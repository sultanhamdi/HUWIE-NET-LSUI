"""Training loop with EMA, AMP, gradient checkpointing, crash-safe resume,
per-epoch logging, qualitative samples, and curve plots.

Designed so a re-run resumes from the last checkpoint (latest.pt).
"""
import csv
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.amp import GradScaler, autocast
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from .config import save_snapshot
from .data import make_loader
from .losses import DINOv2Perceptual, compute_loss
from .model import build_model
from .pretrained import load_pretrained


class _Tee:
    """Mirror stdout to a log file."""
    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()})

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd, device=None):
        self.shadow = {k: v.float().to(device) if device else v.float() for k, v in sd.items()}


def _set_encoder_trainable(model, flag):
    """Freeze/unfreeze the pretrained I2IM and PIM modules.

    During the freeze warmup the Fusion module, PINN and output heads stay
    trainable, so the randomly-initialised heads adapt to the task before
    gradients are allowed to disturb the pretrained features.
    """
    # I2IM module layers
    for name, param in model.named_parameters():
        if name.startswith("i2im_") or name.startswith("pim_"):
            param.requires_grad = flag


def _psnr(pred, gt):
    mse = torch.mean((pred.clamp(0, 1) - gt.clamp(0, 1)) ** 2)
    return (10 * torch.log10(1.0 / (mse + 1e-10))).item()


@torch.no_grad()
def _validate(model, ema, loader, device):
    """Validate with EMA weights; return psnr, ssim, and mean beta/d if PINN."""
    base = model
    backup = {k: v.detach().clone() for k, v in base.state_dict().items()}
    ema.copy_to(base)
    model.eval()

    ps, ss, n = 0.0, 0.0, 0
    betas, depths = [], []
    for inp, gt, _ in tqdm(loader, desc="  val", leave=False, dynamic_ncols=True):
        inp, gt = inp.to(device), gt.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            out, t, beta, d = model(inp)
        out = out.float().clamp(0, 1)
        ps += _psnr(out, gt)
        ss += ssim_fn(out, gt.float(), data_range=1.0).item()
        n += 1
        if beta is not None:
            betas.append(beta.float().mean(0).cpu())
            depths.append(d.float().mean().cpu())

    base.load_state_dict(backup)
    model.train()
    res = {"psnr": ps / n, "ssim": ss / n}
    if betas:
        res["beta"] = torch.stack(betas).mean(0).tolist()
        res["depth"] = float(torch.stack(depths).mean())
    return res


def _save_ckpt(path, epoch, model, ema, opt, sched, scaler, best):
    torch.save({
        "epoch": epoch, "model": model.state_dict(), "ema": ema.state_dict(),
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(), "best_psnr": best,
    }, path)


@torch.no_grad()
def _save_samples(model, ema, loader, device, out_path, n_samples, use_tta):
    base = model
    backup = {k: v.detach().clone() for k, v in base.state_dict().items()}
    ema.copy_to(base)
    model.eval()

    rows = []
    for inp, gt, _ in loader:
        for b in range(inp.size(0)):
            rows.append((inp[b], gt[b]))
            if len(rows) >= n_samples:
                break
        if len(rows) >= n_samples:
            break

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for i, (x, g) in enumerate(rows):
        xb = x.unsqueeze(0).to(device)
        if use_tta:
            preds = []
            for flip in (False, True):
                for rot in (0, 1, 2, 3):
                    xi = TF.hflip(xb) if flip else xb
                    if rot:
                        xi = torch.rot90(xi, rot, dims=[2, 3])
                    with autocast("cuda", dtype=torch.bfloat16):
                        o, *_ = model(xi)
                    o = o.float().clamp(0, 1)
                    if rot:
                        o = torch.rot90(o, -rot, dims=[2, 3])
                    if flip:
                        o = TF.hflip(o)
                    preds.append(o)
            out = torch.stack(preds).mean(0)
        else:
            with autocast("cuda", dtype=torch.bfloat16):
                out, *_ = model(xb)
            out = out.float().clamp(0, 1)
        out = out.squeeze(0).cpu()
        for col, (im, title) in enumerate([(x, "Input"), (out, "Enhanced (EMA+TTA)"), (g, "Reference")]):
            axes[i, col].imshow(im.permute(1, 2, 0).numpy())
            axes[i, col].set_title(title)
            axes[i, col].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    base.load_state_dict(backup)
    model.train()


def _plot_curves(csv_path, curves_dir):
    epochs, loss, char, phys, percep, vpsnr, vssim = [], [], [], [], [], [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            epochs.append(int(r["epoch"]))
            loss.append(float(r["loss"]))
            char.append(float(r["char"]))
            phys.append(float(r["phys"]))
            percep.append(float(r["percep"]))
            vpsnr.append(float(r["val_psnr"]) if r["val_psnr"] else None)
            vssim.append(float(r["val_ssim"]) if r["val_ssim"] else None)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="total")
    plt.plot(epochs, char, label="charbonnier")
    plt.plot(epochs, phys, label="physics")
    plt.plot(epochs, percep, label="perceptual")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Training loss")
    plt.savefig(os.path.join(curves_dir, "loss.png"), dpi=130, bbox_inches="tight")
    plt.close()

    ve = [(e, p, s) for e, p, s in zip(epochs, vpsnr, vssim) if p is not None]
    if ve:
        e, p, s = zip(*ve)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(e, p, "b-o", label="PSNR"); ax1.set_xlabel("epoch")
        ax1.set_ylabel("PSNR (dB)", color="b")
        ax2 = ax1.twinx(); ax2.plot(e, s, "r-s", label="SSIM")
        ax2.set_ylabel("SSIM", color="r")
        plt.title("Validation (UIEB)")
        plt.savefig(os.path.join(curves_dir, "val_metrics.png"), dpi=130, bbox_inches="tight")
        plt.close()


def run_training(cfg, smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU-specific optimisations.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tr = cfg["train"]
    paths = cfg["paths"]

    out_dir = cfg["out_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    curves_dir = os.path.join(out_dir, "curves")
    samples_dir = os.path.join(out_dir, "samples")
    phys_dir = os.path.join(out_dir, "physics")
    for d in (ckpt_dir, curves_dir, samples_dir, phys_dir):
        os.makedirs(d, exist_ok=True)

    sys.stdout = _Tee(os.path.join(out_dir, "train_log.txt"))
    save_snapshot(cfg, os.path.join(out_dir, "config_snapshot.yaml"))

    num_epochs = 1 if smoke else tr["num_epochs"]
    print(f"=== Variant {cfg['variant']} | pinn={cfg['use_pinn']} dino={cfg['use_dino']} "
          f"| epochs={num_epochs} | device={device} ===")

    # Model + pretrained.
    model = build_model(cfg).to(device)
    if paths.get("pretrained") and os.path.exists(paths["pretrained"]):
        load_pretrained(model, paths["pretrained"])
    else:
        print(f"[pretrained] WARNING not found at {paths.get('pretrained')} — random init")

    dino_fn = DINOv2Perceptual().to(device) if cfg["use_dino"] else None
    if dino_fn:
        print("[dino] DINOv2-S/14 loaded (frozen)")

    ema = EMA(model, tr["ema_decay"])

    # Compile only the training forward. `model` stays uncompiled so EMA / state_dict /
    # best.pt keep clean keys (no _orig_mod prefix); compiled wrapper shares the same params.
    fwd = torch.compile(model) if tr.get("compile", False) else model

    # Differential LR: PINN is random-init so it needs a higher LR than the pretrained backbone.
    pinn_ids = {id(p) for p in (model.pinn.parameters() if model.pinn else [])}
    backbone_params = [p for p in model.parameters() if id(p) not in pinn_ids]
    pinn_params     = [p for p in model.parameters() if id(p) in pinn_ids]
    param_groups = [{"params": backbone_params, "lr": tr["lr"]}]
    if pinn_params:
        param_groups.append({"params": pinn_params, "lr": tr.get("pinn_lr", tr["lr"] * 5)})
    opt = optim.AdamW(param_groups, weight_decay=tr["weight_decay"],
                      betas=tuple(tr.get("betas", (0.9, 0.999))))

    def lr_lambda(ep):
        if ep < tr["warmup_epochs"]:
            return (ep + 1) / tr["warmup_epochs"]
        t = (ep - tr["warmup_epochs"]) / max(1, num_epochs - tr["warmup_epochs"])
        r = tr["min_lr"] / tr["lr"]
        return r + 0.5 * (1 - r) * (1 + math.cos(math.pi * t))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler("cuda", enabled=False)  # bfloat16 has fp32 range — no scaling needed

    start_epoch, best = 0, 0.0
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        s = torch.load(latest, map_location="cpu")
        model.load_state_dict(s["model"])
        ema.load_state_dict(s["ema"], device=device)
        opt.load_state_dict(s["optimizer"])
        sched.load_state_dict(s["scheduler"])
        scaler.load_state_dict(s["scaler"])
        start_epoch, best = s["epoch"], s["best_psnr"]
        print(f"[resume] from epoch {start_epoch}, best_psnr={best:.2f}")

    # Use local data paths directly (no staging needed for Kaggle/local).
    lsui_raw = paths["lsui_raw"]
    lsui_ref = paths["lsui_ref"]
    uieb_raw = paths["uieb_raw"]
    uieb_ref = paths["uieb_ref"]

    train_loader = make_loader(lsui_raw, lsui_ref, tr["patch_size"],
                               tr["batch_size"], tr["num_workers"], augment=True,
                               shuffle=True, train=True)
    val_loader = make_loader(uieb_raw, uieb_ref, tr["patch_size"],
                             tr["batch_size"], tr["num_workers"], augment=False,
                             shuffle=False, train=False, drop_last=False)
    print(f"[data] train={len(train_loader.dataset)} pairs  val={len(val_loader.dataset)} pairs")

    csv_path = os.path.join(out_dir, "train_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "char", "phys", "percep",
                                    "lr", "epoch_time_s", "val_psnr", "val_ssim"])
    phys_csv = os.path.join(phys_dir, "beta_depth.csv")
    if cfg["use_pinn"] and not os.path.exists(phys_csv):
        with open(phys_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "beta_R", "beta_G", "beta_B", "depth"])

    accum = tr["accum_steps"]
    freeze_ep = tr.get("freeze_encoder_epochs", 0)
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        model.train()

        if freeze_ep:
            enc_trainable = epoch >= freeze_ep
            _set_encoder_trainable(model, enc_trainable)
            if epoch == 0 and not enc_trainable:
                print(f"[freeze] I2IM+PIM frozen for first {freeze_ep} epoch(s)")
            elif epoch == freeze_ep:
                print(f"[freeze] I2IM+PIM unfrozen at epoch {epoch+1} — full fine-tune")

        opt.zero_grad()
        phys_scale = min(1.0, (epoch + 1) / max(1, tr["phys_warmup_epochs"]))
        run = {"loss": 0.0, "char": 0.0, "phys": 0.0, "percep": 0.0}
        steps = 0

        pbar = tqdm(train_loader, desc=f"[{epoch+1:3d}/{num_epochs}]", leave=False,
                    dynamic_ncols=True)
        for i, (inp, gt, _) in enumerate(pbar):
            inp, gt = inp.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                out, t, beta, d = fwd(inp)
                loss, comps = compute_loss(inp, out, gt, t, dino_fn,
                                           cfg["loss"]["w_char"], cfg["loss"]["w_phys"],
                                           cfg["loss"]["w_percep"], phys_scale=phys_scale)
                loss = loss / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(model)
            run["loss"] += loss.item() * accum
            run["char"] += comps["char"].item()
            run["phys"] += comps["phys"].item()
            run["percep"] += comps["percep"].item()
            steps += 1
            pbar.set_postfix(loss=f"{run['loss']/steps:.4f}",
                             char=f"{comps['char'].item():.4f}",
                             phys=f"{comps['phys'].item():.4f}",
                             percep=f"{comps['percep'].item():.4f}")
            if smoke and steps >= 5:
                break
        pbar.close()

        sched.step()
        for k in run:
            run[k] /= max(1, steps)
        lr = opt.param_groups[0]["lr"]
        dt = time.time() - t0

        vp = vs = ""
        do_val = ((epoch + 1) % cfg["eval"]["eval_every"] == 0) or (epoch + 1 == num_epochs) or smoke
        if do_val:
            v = _validate(model, ema, val_loader, device)
            vp, vs = f"{v['psnr']:.4f}", f"{v['ssim']:.4f}"
            if cfg["use_pinn"] and "beta" in v:
                with open(phys_csv, "a", newline="") as f:
                    csv.writer(f).writerow([epoch + 1, *[f"{x:.5f}" for x in v["beta"]], f"{v['depth']:.5f}"])
            if v["psnr"] > best:
                best = v["psnr"]
                bb = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
                ema.copy_to(model)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
                model.load_state_dict(bb)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{run['loss']:.5f}", f"{run['char']:.5f}",
                                    f"{run['phys']:.5f}", f"{run['percep']:.5f}",
                                    f"{lr:.2e}", f"{dt:.0f}", vp, vs])
        msg = (f"[{epoch+1:3d}/{num_epochs}] loss={run['loss']:.4f} "
               f"(char={run['char']:.4f} phys={run['phys']:.4f} percep={run['percep']:.4f}) "
               f"lr={lr:.2e} {dt:.0f}s")
        if do_val:
            msg += f"  val_psnr={vp} val_ssim={vs}" + ("  <<BEST" if vp and float(vp) >= best else "")
        if torch.cuda.is_available():
            msg += f"  vram={torch.cuda.max_memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB"
        print(msg)

        if ((epoch + 1) % cfg["eval"]["save_every"] == 0) or (epoch + 1 == num_epochs):
            _save_ckpt(os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pt"), epoch + 1,
                       model, ema, opt, sched, scaler, best)
        _save_ckpt(latest, epoch + 1, model, ema, opt, sched, scaler, best)

    # Final artefacts.
    _save_samples(model, ema, val_loader, device,
                  os.path.join(samples_dir, "grid.png"),
                  cfg["eval"]["num_samples"], cfg["eval"]["use_tta"])
    _plot_curves(csv_path, curves_dir)
    if cfg["use_pinn"] and os.path.exists(phys_csv):
        _plot_physics(phys_csv, phys_dir)
    print(f"=== {cfg['variant']} done. best_psnr={best:.2f} ===")
    return best


def _plot_physics(phys_csv, phys_dir):
    ep, bR, bG, bB, dp = [], [], [], [], []
    with open(phys_csv) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); bR.append(float(r["beta_R"]))
            bG.append(float(r["beta_G"])); bB.append(float(r["beta_B"]))
            dp.append(float(r["depth"]))
    plt.figure(figsize=(8, 5))
    plt.plot(ep, bR, "r-o", label="beta_R")
    plt.plot(ep, bG, "g-s", label="beta_G")
    plt.plot(ep, bB, "b-^", label="beta_B")
    plt.plot(ep, dp, "k--", label="depth")
    plt.xlabel("epoch"); plt.ylabel("value")
    plt.title("Learned Beer-Lambert params (expect beta_R > beta_G > beta_B)")
    plt.legend()
    plt.savefig(os.path.join(phys_dir, "beta_depth.png"), dpi=130, bbox_inches="tight")
    plt.close()

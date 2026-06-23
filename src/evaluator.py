"""Final evaluation on UIEB-890 + aggregation into report tables.

Loads best.pt (EMA weights), runs full + no-reference metrics over the whole UIEB
set per image, writes metrics.json + metrics_per_image.csv. Aggregation builds the
ablation summary and a comparison table vs published baselines.
"""
import csv
import json
import os

import torch
import torchvision.transforms.functional as TF
from torch.amp import autocast
from tqdm import tqdm

from .data import make_loader
from .metrics import full_reference, no_reference
from .model import build_model


# Published baseline numbers on UIEB (Test-U90 unless noted). Sources in references.md.
# NOTE: baselines use a random 90-image test split; ours uses full UIEB-890 — not
# directly comparable, shown for context only.
BASELINES = [
    ("UGAN",                2020, 20.68, 0.840, "U-Shape paper (Test-U90)"),
    ("FUnIE-GAN",           2020, 19.45, 0.850, "U-Shape paper (Test-U90)"),
    ("Water-Net",           2020, 19.81, 0.860, "U-Shape paper (Test-U90)"),
    ("Ucolor",              2021, 20.78, 0.870, "U-Shape paper (Test-U90)"),
    ("PUIE-Net",            2022, 21.38, 0.882, "paper (Test-U90)"),
    ("U-Shape Transformer", 2023, 22.91, 0.910, "paper (Test-U90)"),
    ("MuLA-GAN",            2023, 25.59, 0.893, "leaderboard"),
    ("DGNet",               2023, 25.62, 0.929, "leaderboard"),
    ("Mamba-UIE",           2024, 27.13, 0.930, "leaderboard"),
]


@torch.no_grad()
def _enhance(model, x, device, use_tta):
    if not use_tta:
        with autocast("cuda", dtype=torch.bfloat16):
            out, *_ = model(x.to(device))
        return out.float().clamp(0, 1)
    preds = []
    for flip in (False, True):
        for rot in (0, 1, 2, 3):
            xi = TF.hflip(x) if flip else x
            if rot:
                xi = torch.rot90(xi, rot, dims=[2, 3])
            with autocast("cuda", dtype=torch.bfloat16):
                o, *_ = model(xi.to(device))
            o = o.float().clamp(0, 1)
            if rot:
                o = torch.rot90(o, -rot, dims=[2, 3])
            if flip:
                o = TF.hflip(o)
            preds.append(o.cpu())
    return torch.stack(preds).mean(0).to(device)


def run_evaluation(cfg, use_tta=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_tta = cfg["eval"]["use_tta"] if use_tta is None else use_tta
    paths = cfg["paths"]
    out_dir = cfg["out_dir"]

    model = build_model(cfg).to(device).eval()
    ckpt = os.path.join(out_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(out_dir, "checkpoints", "latest.pt")
        state = torch.load(ckpt, map_location="cpu")["model"]
    else:
        state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    print(f"[eval] {cfg['variant']} loaded {ckpt}  tta={use_tta}")

    loader = make_loader(paths["uieb_raw"], paths["uieb_ref"], cfg["train"]["patch_size"],
                         batch_size=1, num_workers=cfg["train"]["num_workers"],
                         augment=False, shuffle=False, train=False, drop_last=False)

    acc = {k: [] for k in ("psnr", "ssim", "lpips", "uciqe", "uiqm")}
    per_image_rows = []
    betas, depths = [], []

    for inp, gt, fnames in tqdm(loader, desc=f"eval {cfg['variant']}", dynamic_ncols=True):
        out = _enhance(model, inp, device, use_tta)
        gt_dev = gt.to(device)
        fr = full_reference(out, gt_dev)
        nr = no_reference(out)
        for k, v in {**fr, **nr}.items():
            acc[k].append(v)
        fname = fnames[0] if isinstance(fnames, (list, tuple)) else fnames
        per_image_rows.append({"fname": fname, **fr, **nr})

        # Physics params: separate single-pass (no TTA) to capture beta/depth.
        if model.use_pinn:
            with torch.no_grad():
                with autocast("cuda", dtype=torch.bfloat16):
                    _, _, beta_val, d_val = model(inp.to(device))
            betas.append(beta_val.float().mean(0).cpu())
            depths.append(d_val.float().mean().cpu())

    metrics = {"variant": cfg["variant"], "n_images": len(loader.dataset),
               "use_tta": use_tta, "eval_set": "UIEB-890"}
    for k, v in acc.items():
        t = torch.tensor(v)
        metrics[f"{k}_mean"] = float(t.mean())
        metrics[f"{k}_std"] = float(t.std())

    # Physics params in metrics.json (A2/A4 only).
    if betas:
        mean_beta = torch.stack(betas).mean(0).tolist()
        metrics["beta_R_mean"] = round(mean_beta[0], 5)
        metrics["beta_G_mean"] = round(mean_beta[1], 5)
        metrics["beta_B_mean"] = round(mean_beta[2], 5)
        metrics["depth_mean"]  = round(float(torch.stack(depths).mean()), 5)
        metrics["beta_ordered_R_G_B"] = bool(mean_beta[0] > mean_beta[1] > mean_beta[2])

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Per-image CSV for statistical analysis.
    csv_path = os.path.join(out_dir, "metrics_per_image.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fname", "psnr", "ssim", "lpips", "uciqe", "uiqm"])
        w.writeheader()
        for row in per_image_rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in row.items()})

    print(f"[eval] {cfg['variant']}: " + "  ".join(
        f"{k}={metrics[f'{k}_mean']:.4f}" for k in acc)
        + (f"  beta=({metrics.get('beta_R_mean',''):.4f},{metrics.get('beta_G_mean',''):.4f},{metrics.get('beta_B_mean',''):.4f})"
           if betas else ""))
    return metrics


def build_summary(variants, outputs_dir):
    """Read each variant's metrics.json -> ablation_summary.{csv,md} + comparison_table.md."""
    rows = []
    for v in variants:
        p = os.path.join(outputs_dir, v, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                rows.append(json.load(f))
    if not rows:
        print("[summary] no metrics.json found — run evaluate first")
        return

    cols = ["psnr", "ssim", "lpips", "uciqe", "uiqm"]
    labels = {"A1": "HuWie-Net", "A2": "+ PINN", "A3": "+ DINOv2", "A4": "Full (PINN+DINOv2)"}

    # CSV
    csv_path = os.path.join(outputs_dir, "ablation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("variant,components," + ",".join(f"{c}_mean,{c}_std" for c in cols) + "\n")
        for r in rows:
            f.write(f"{r['variant']},{labels.get(r['variant'],'')}," +
                    ",".join(f"{r[f'{c}_mean']:.4f},{r[f'{c}_std']:.4f}" for c in cols) + "\n")

    # Markdown ablation table
    md = ["# Ablation Summary (UIEB-890, mean ± std)", "",
          "| Variant | Components | PSNR ↑ | SSIM ↑ | LPIPS ↓ | UCIQE ↑ | UIQM ↑ |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['variant']} | {labels.get(r['variant'],'')} | "
            f"{r['psnr_mean']:.2f}±{r['psnr_std']:.2f} | "
            f"{r['ssim_mean']:.3f}±{r['ssim_std']:.3f} | "
            f"{r['lpips_mean']:.3f}±{r['lpips_std']:.3f} | "
            f"{r['uciqe_mean']:.3f} | {r['uiqm_mean']:.3f} |")
    md += ["", "UCIQE/UIQM: single reference implementation, internal comparison only.",
           "LPIPS lower = better (perceptual). DINOv2's contribution shows here.",
           "Learned β ordering (β_R>β_G>β_B) in physics/ plots evidences the PINN contribution."]
    with open(os.path.join(outputs_dir, "ablation_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # Comparison vs baselines
    ours = next((r for r in rows if r["variant"] == "A4"), rows[-1])
    cmp = ["# Comparison vs Published Baselines (UIEB)", "",
           "| Method | Year | PSNR | SSIM | Source |",
           "|---|---|---|---|---|"]
    for name, yr, ps, ss, src in BASELINES:
        cmp.append(f"| {name} | {yr} | {ps:.2f} | {ss:.3f} | {src} |")
    cmp.append(f"| **HuWie-Net-PINN (ours, {ours['variant']})** | 2025 | "
               f"**{ours['psnr_mean']:.2f}** | **{ours['ssim_mean']:.3f}** | this work (full UIEB-890) |")
    cmp += ["",
            "> ⚠️ Split differs: baselines use a *random* 90-image UIEB test split; ours is",
            "> evaluated on the full UIEB-890. Numbers are context, not a like-for-like ranking."]
    with open(os.path.join(outputs_dir, "comparison_table.md"), "w") as f:
        f.write("\n".join(cmp) + "\n")

    print(f"[summary] wrote ablation_summary.csv/.md and comparison_table.md to {outputs_dir}")

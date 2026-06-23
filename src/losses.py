"""Loss functions: Charbonnier reconstruction, Beer-Lambert physics consistency,
DINOv2 perceptual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred, gt, eps=1e-3):
    return torch.mean(torch.sqrt((pred - gt) ** 2 + eps ** 2))


def physics_consistency_loss(inp, out, t):
    """Beer-Lambert self-consistency: re-degrading the enhanced output with the
    estimated transmittance should reproduce the input.

        inp ~= out * t + A * (1 - t)

    A (ambient light) is estimated from the brightest 0.1% pixels per channel.
    t is global per image (B, 3) from the PINN branch.
    """
    B, C, H, W = inp.shape
    flat = inp.view(B, C, -1)
    k = max(1, int(H * W * 0.001))
    A = flat.topk(k, dim=2).values.mean(dim=2).view(B, C, 1, 1)
    t_s = t.unsqueeze(-1).unsqueeze(-1)
    recon = out * t_s + A * (1 - t_s)
    return F.l1_loss(recon, inp)


class DINOv2Perceptual(nn.Module):
    """Frozen DINOv2-S/14 perceptual loss on patch tokens.

    Params are frozen but gradients flow THROUGH the network back to the model on
    the prediction path. The reference path is wrapped in no_grad (fixed target).
    """
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def _pre(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, pred, gt):
        f_pred = self.dino.forward_features(self._pre(pred))["x_norm_patchtokens"]
        with torch.no_grad():
            f_gt = self.dino.forward_features(self._pre(gt))["x_norm_patchtokens"]
        return F.mse_loss(f_pred, f_gt.detach())


def compute_loss(inp, out, gt, t, dino_fn, w_char, w_phys, w_percep, phys_scale=1.0):
    """Return (total, dict of components). t/dino_fn may be None (ablation toggles).
    phys_scale ramps the physics term during warmup.
    """
    char = charbonnier_loss(out, gt).mean()
    total = w_char * char
    comps = {"char": char.detach()}

    if t is not None:
        phys = physics_consistency_loss(inp, out, t).mean()
        total = total + (w_phys * phys_scale) * phys
        comps["phys"] = phys.detach()
    else:
        comps["phys"] = torch.zeros((), device=inp.device)

    if dino_fn is not None:
        percep = dino_fn(out, gt).mean()  # DataParallel returns vector of scalars
        total = total + w_percep * percep
        comps["percep"] = percep.detach()
    else:
        comps["percep"] = torch.zeros((), device=inp.device)

    return total.mean(), comps

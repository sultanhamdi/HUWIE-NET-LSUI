"""Evaluation metrics.

Full-reference : PSNR, SSIM (torchmetrics), LPIPS (lpips package).
No-reference   : UCIQE, UIQM (numpy).

UCIQE/UIQM follow the standard formulations (Yang & Sowmya 2015; Panetta et al.
2016). Many ports differ slightly in block size / constants, so absolute values
are NOT directly comparable to other papers — they are used here for INTERNAL
ablation comparison only (identical implementation across all variants).
"""
import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2lab

# Lazy singletons (loaded once, on the eval device).
_ssim_fn = None
_psnr_fn = None
_lpips_fn = None


def _ensure_torch_metrics(device):
    global _ssim_fn, _psnr_fn, _lpips_fn
    if _ssim_fn is None:
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        _psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
        _ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        _lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)


@torch.no_grad()
def full_reference(pred, gt):
    """pred, gt: (B, 3, H, W) in [0, 1]. Returns dict of float means over batch."""
    _ensure_torch_metrics(pred.device)
    pred = pred.clamp(0, 1).float()
    gt = gt.clamp(0, 1).float()
    return {
        "psnr": _psnr_fn(pred, gt).item(),
        "ssim": _ssim_fn(pred, gt).item(),
        "lpips": _lpips_fn(pred, gt).item(),
    }


# ── No-reference: UCIQE ────────────────────────────────────────────────────────

def uciqe(img):
    """img: HWC RGB float in [0, 1]. Higher is better."""
    lab = rgb2lab(np.clip(img, 0, 1))
    L = lab[:, :, 0] / 100.0
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt(a ** 2 + b ** 2)

    sigma_c = np.std(chroma)
    # luminance contrast: top 1% minus bottom 1%
    Lf = np.sort(L.ravel())
    n = len(Lf)
    top = Lf[int(0.99 * n):].mean()
    bot = Lf[:max(1, int(0.01 * n))].mean()
    con_l = top - bot
    # average saturation
    sat = np.divide(chroma, lab[:, :, 0] + 1e-6)
    mu_s = np.mean(sat)

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


# ── No-reference: UIQM ─────────────────────────────────────────────────────────

def _alpha_trimmed_stats(x, a_l=0.1, a_r=0.1):
    x = np.sort(x.ravel())
    n = len(x)
    lo, hi = int(a_l * n), int((1 - a_r) * n)
    x = x[lo:hi] if hi > lo else x
    return x.mean(), x.var()


def _uicm(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = R - G
    yb = (R + G) / 2.0 - B
    mu_rg, var_rg = _alpha_trimmed_stats(rg)
    mu_yb, var_yb = _alpha_trimmed_stats(yb)
    return -0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * np.sqrt(var_rg + var_yb)


def _eme(ch, k=8):
    h, w = ch.shape
    bh, bw = max(1, h // k), max(1, w // k)
    total = 0.0
    cnt = 0
    for i in range(0, h - bh + 1, bh):
        for j in range(0, w - bw + 1, bw):
            blk = ch[i:i + bh, j:j + bw]
            mx, mn = blk.max(), blk.min()
            if mn <= 0:
                mn = 1e-4
            if mx <= 0:
                mx = 1e-4
            total += np.log(mx / mn)
            cnt += 1
    return (2.0 / cnt) * total if cnt else 0.0


def _uism(img):
    from scipy.ndimage import sobel
    lam = (0.299, 0.587, 0.114)
    s = 0.0
    for c in range(3):
        ch = img[:, :, c]
        gx = sobel(ch, axis=0)
        gy = sobel(ch, axis=1)
        edge = np.sqrt(gx ** 2 + gy ** 2) * ch
        s += lam[c] * _eme(edge)
    return s


def _logamee(ch, k=8):
    h, w = ch.shape
    bh, bw = max(1, h // k), max(1, w // k)
    total = 0.0
    cnt = 0
    for i in range(0, h - bh + 1, bh):
        for j in range(0, w - bw + 1, bw):
            blk = ch[i:i + bh, j:j + bw]
            mx, mn = blk.max(), blk.min()
            if mx + mn < 1e-6:
                continue
            phi = (mx - mn) / (mx + mn + 1e-6)
            if phi > 0:
                total += phi * np.log(phi + 1e-6)
            cnt += 1
    return (1.0 / cnt) * total if cnt else 0.0


def _uiconm(img):
    inten = img.mean(axis=2)
    return _logamee(inten)


def uiqm(img):
    """img: HWC RGB float in [0, 1]. Higher is better."""
    img = np.clip(img, 0, 1).astype(np.float64)
    uicm = _uicm(img)
    uism = _uism(img)
    uiconm = _uiconm(img)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


@torch.no_grad()
def no_reference(pred):
    """pred: (B, 3, H, W) in [0, 1]. Returns dict of float means over batch."""
    pred = pred.clamp(0, 1).float().cpu().numpy()
    uciqe_v, uiqm_v = [], []
    for k in range(pred.shape[0]):
        img = np.transpose(pred[k], (1, 2, 0))
        uciqe_v.append(uciqe(img))
        uiqm_v.append(uiqm(img))
    return {"uciqe": float(np.mean(uciqe_v)), "uiqm": float(np.mean(uiqm_v))}

"""HuWieNetPINN — official HUWIE-Net backbone + optional Beer-Lambert PINN branch.

HUWIE-Net architecture from UIE-Lab (https://github.com/UIE-Lab/HUWIE-Net):
Three modules — Image-to-Image (I2IM), Physics-Informed (PIM), and Fusion —
with Dark Channel Prior and Atmospheric Light estimation built into PIM.

The PINN branch (Beer-Lambert global attenuation) is an additional component
we attach to the fusion module's latent features.  It is randomly initialised
and provides a learnable physics prior on top of the HUWIE-Net output.

Ablation variants:
  A1 — HUWIE-Net only (baseline, Charbonnier loss)
  A2 — HUWIE-Net + PINN (Beer-Lambert physics branch)
  A3 — HUWIE-Net + DINOv2 perceptual loss
  A4 — Full model (HUWIE-Net + PINN + DINOv2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── PINN branch: global Beer-Lambert attenuation ───────────────────────────────

class PINNBranch(nn.Module):
    """Estimate per-image attenuation beta_RGB and a depth scalar from latent
    features, then transmittance t = exp(-beta * d).

    Note (honest scope): beta and d are GLOBAL per image (no per-pixel depth map),
    so this acts as a physically-grounded global colour-attenuation prior, not a
    full scene-depth model.

    Initialised so t ~= 1 at the start (d ~= 0): the J/t inversion begins as
    near-identity and does not distort the backbone output early on.
    """

    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.beta_head = nn.Sequential(
            nn.Flatten(), nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 3), nn.Softplus(),
        )
        self.depth_head = nn.Sequential(
            nn.Flatten(), nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
        # Make d ~= sigmoid(-4) ~= 0.018 at init  ->  t ~= 1.
        nn.init.zeros_(self.depth_head[3].weight)
        nn.init.constant_(self.depth_head[3].bias, -4.0)

    def forward(self, feat):
        p = self.pool(feat)
        beta = self.beta_head(p)          # (B, 3)
        d = self.depth_head(p)            # (B, 1)
        t = torch.exp(-beta * d)          # (B, 3)
        return beta, d, t


# ── Full model ─────────────────────────────────────────────────────────────────

class HuWieNetPINN(nn.Module):
    """HUWIE-Net (I2IM + PIM + Fusion) + optional PINNBranch.

    The PINNBranch reads the 64-channel latent from the Fusion module (before
    the final 1x1 conv) and produces transmittance t, which is used to rescale
    the output via the Beer-Lambert equation.
    """

    def __init__(self, use_pinn=True):
        super().__init__()
        self.use_pinn = use_pinn

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # ── Image-to-Image Module ──────────────────────────────────────────
        self.i2im_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.i2im_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.i2im_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.i2im_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.i2im_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.i2im_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.i2im_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv7 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

        # ── Physics-Informed Module ────────────────────────────────────────
        self.pim_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pim_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pim_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pim_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pim_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pim_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pim_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv7 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

        # ── Fusion Module ──────────────────────────────────────────────────
        self.con1 = nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con7 = nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=0)

        # ── PINN (Beer-Lambert) ────────────────────────────────────────────
        # Reads the 64-channel fusion latent (after con6, before con7).
        self.pinn = PINNBranch(feat_dim=64) if use_pinn else None

    def forward(self, x):
        # ── I2IM branch ────────────────────────────────────────────────────
        h = self.relu(self.i2im_in1(self.i2im_conv1(x)))
        h = self.relu(self.i2im_in2(self.i2im_conv2(h)))
        h = self.relu(self.i2im_in3(self.i2im_conv3(h)))
        h = self.relu(self.i2im_in4(self.i2im_conv4(h)))
        h = self.relu(self.i2im_in5(self.i2im_conv5(h)))
        h = self.relu(self.i2im_in6(self.i2im_conv6(h)))
        h = self.i2im_conv7(h)
        h = h + x
        i2im_out = self.sigmoid(h)

        # ── PIM branch ─────────────────────────────────────────────────────
        h2 = self.relu(self.pim_in1(self.pim_conv1(x)))
        h2 = self.relu(self.pim_in2(self.pim_conv2(h2)))
        h2 = self.relu(self.pim_in3(self.pim_conv3(h2)))
        h2 = self.relu(self.pim_in4(self.pim_conv4(h2)))
        h2 = self.relu(self.pim_in5(self.pim_conv5(h2)))
        h2 = self.relu(self.pim_in6(self.pim_conv6(h2)))
        t_pim = self.pim_conv7(h2)
        t_pim = t_pim + x
        t_pim = self.sigmoid(t_pim)

        dark = self._dark_channel(x)
        A = self._atmospheric_light(x, dark)

        eps = 1e-05
        fb_1 = torch.div((x - A), (t_pim + eps)) + A
        pim_out = self.sigmoid(fb_1)

        # ── Fusion branch ──────────────────────────────────────────────────
        att_in = torch.cat([x, i2im_out, pim_out], dim=1)
        h3 = self.relu(self.in1(self.con1(att_in)))
        h3 = self.relu(self.in2(self.con2(h3)))
        h3 = self.relu(self.in3(self.con3(h3)))
        h3 = self.relu(self.in4(self.con4(h3)))
        h3 = self.relu(self.in5(self.con5(h3)))
        h3 = self.relu(self.in6(self.con6(h3)))

        # PINN reads the 64-ch fusion latent BEFORE the final 1x1 conv.
        beta = d = t = None
        if self.use_pinn:
            beta, d, t = self.pinn(h3)

        h3 = self.con7(h3)
        att_out = self.sigmoid(h3)

        m1, m2 = torch.split(att_out, 3, dim=1)
        output = 0.5 * torch.mul(m1, i2im_out) + 0.5 * torch.mul(m2, pim_out)

        # Apply Beer-Lambert transmittance correction.
        if self.use_pinn:
            t_s = t.unsqueeze(-1).unsqueeze(-1)   # (B, 3, 1, 1)
            output = (output / t_s.clamp(min=0.05)).clamp(0, 1)

        return output, t, beta, d

    def _dark_channel(self, img):
        patch_size = 15
        # Disable the red channel, use only blue and green channels
        no_red_img = img[:, 1:, :, :]
        min_img, _ = torch.min(no_red_img, dim=1, keepdim=True)
        dark = -F.max_pool2d(-min_img, kernel_size=patch_size, stride=1, padding=patch_size // 2)
        return dark

    def _atmospheric_light(self, img, dark_channel):
        flat_img = img.view(img.size(0), img.size(1), -1)
        flat_dark = dark_channel.view(dark_channel.size(0), dark_channel.size(1), -1)
        num_pixels = flat_dark.size(dim=2)
        num_top_pixels = max(1, int(0.001 * num_pixels))
        _, indices = torch.topk(flat_dark, k=num_top_pixels, dim=2, largest=True, sorted=False)
        A = torch.gather(flat_img, 2, indices.expand(-1, img.size(1), -1)).max(dim=2)[0]
        A = A.unsqueeze(2).unsqueeze(3)
        return A


def build_model(cfg):
    return HuWieNetPINN(
        use_pinn=cfg["use_pinn"],
    )


if __name__ == "__main__":
    # Smoke test (CPU): shapes + PINN toggle, no GPU needed.
    for use_pinn in (False, True):
        net = HuWieNetPINN(use_pinn=use_pinn).eval()
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out, t, beta, d = net(x)
        assert out.shape == x.shape, out.shape
        if use_pinn:
            assert t.shape == (1, 3) and beta.shape == (1, 3) and d.shape == (1, 1)
            assert t.mean() > 0.9, f"t should start ~1, got {t.mean():.3f}"
        n = sum(p.numel() for p in net.parameters())
        print(f"use_pinn={use_pinn}: out {tuple(out.shape)}  params {n/1e6:.2f}M  "
              f"t~{None if t is None else round(t.mean().item(),3)}")
    print("model smoke test OK")

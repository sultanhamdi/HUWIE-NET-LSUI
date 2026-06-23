"""Paired underwater dataset: (raw degraded, clean reference).

Augmentation is applied identically to both images. No colour jitter — colour is
the signal the model must learn to correct.
"""
import glob
import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG")


def _list_images(d):
    files = []
    for e in _EXTS:
        files += glob.glob(os.path.join(d, e))
    return sorted(files)


class UIEDataset(Dataset):
    """Pairs raw[i] with ref[i] by sorted filename order. Validates counts match."""

    def __init__(self, raw_dir, ref_dir, patch_size=256, augment=True, train=True):
        self.raws = _list_images(raw_dir)
        self.refs = _list_images(ref_dir)
        assert self.raws, f"No images found in {raw_dir}"
        assert len(self.raws) == len(self.refs), (
            f"Pair count mismatch: {len(self.raws)} raw in {raw_dir} vs "
            f"{len(self.refs)} ref in {ref_dir}"
        )
        self.patch_size = patch_size
        self.augment = augment
        self.train = train

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, idx):
        raw = Image.open(self.raws[idx]).convert("RGB")
        ref = Image.open(self.refs[idx]).convert("RGB")

        # Align sizes (some pairs differ by a pixel).
        w, h = min(raw.width, ref.width), min(raw.height, ref.height)
        raw, ref = raw.crop((0, 0, w, h)), ref.crop((0, 0, w, h))

        p = self.patch_size
        if self.train:
            if w >= p and h >= p:
                i = random.randint(0, h - p)
                j = random.randint(0, w - p)
                raw, ref = TF.crop(raw, i, j, p, p), TF.crop(ref, i, j, p, p)
            else:
                raw = TF.resize(raw, (p, p), interpolation=TF.InterpolationMode.BICUBIC)
                ref = TF.resize(ref, (p, p), interpolation=TF.InterpolationMode.BICUBIC)
            if self.augment:
                if random.random() > 0.5:
                    raw, ref = TF.hflip(raw), TF.hflip(ref)
                if random.random() > 0.5:
                    raw, ref = TF.vflip(raw), TF.vflip(ref)
                k = random.randint(0, 3)
                if k:
                    raw, ref = TF.rotate(raw, 90 * k), TF.rotate(ref, 90 * k)
        else:
            # Eval: fixed size so PSNR/SSIM are comparable across images.
            raw = TF.resize(raw, (p, p), interpolation=TF.InterpolationMode.BICUBIC)
            ref = TF.resize(ref, (p, p), interpolation=TF.InterpolationMode.BICUBIC)

        return TF.to_tensor(raw), TF.to_tensor(ref), os.path.basename(self.raws[idx])


def make_loader(raw_dir, ref_dir, patch_size, batch_size, num_workers=4,
                augment=True, shuffle=True, train=True, drop_last=True):
    ds = UIEDataset(raw_dir, ref_dir, patch_size, augment=augment, train=train)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0), drop_last=drop_last,
    )

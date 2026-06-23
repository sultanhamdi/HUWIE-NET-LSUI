"""Helpers to locate and arrange downloaded datasets into canonical folders.

Kaggle dataset zips have varying internal structure, so we detect the
image-containing folders by name patterns and image counts rather than hardcoding.
"""
import glob
import os
import shutil

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
RAW_NAMES = ("input", "raw", "traina", "lr", "degraded", "underwater")
REF_NAMES = ("gt", "reference", "target", "trainb", "hr", "ref", "clean", "groundtruth")


def _count_images(d):
    return sum(1 for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(_IMG_EXTS))


def _all_dirs(root):
    out = [root]
    for d, subdirs, _ in os.walk(root):
        for s in subdirs:
            out.append(os.path.join(d, s))
    return out


def find_image_dir(root):
    """Return the directory (recursively) holding the most images."""
    best, best_n = None, 0
    for d in _all_dirs(root):
        n = _count_images(d)
        if n > best_n:
            best, best_n = d, n
    return best, best_n


def find_pair_dirs(root):
    """Find a (raw, ref) directory pair by name patterns; require both have images."""
    raw = ref = None
    for d in _all_dirs(root):
        if _count_images(d) == 0:
            continue
        name = os.path.basename(d).lower()
        if raw is None and any(k in name for k in RAW_NAMES):
            raw = d
        if ref is None and any(k in name for k in REF_NAMES):
            ref = d
    return raw, ref


def move_dir(src, dst):
    """Move src directory contents to dst (created fresh)."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def arrange_paired(dl_root, raw_dst, ref_dst):
    """Detect a raw/ref pair under dl_root and move into canonical dsts."""
    raw, ref = find_pair_dirs(dl_root)
    if not (raw and ref):
        raise RuntimeError(
            f"Could not find raw/ref folders under {dl_root}. "
            f"Found dirs: {[os.path.basename(d) for d in _all_dirs(dl_root) if _count_images(d)]}"
        )
    move_dir(raw, raw_dst)
    move_dir(ref, ref_dst)
    return _count_images(raw_dst), _count_images(ref_dst)


def arrange_single(dl_root, dst):
    """Move the largest image folder under dl_root into dst."""
    d, n = find_image_dir(dl_root)
    if not d or n == 0:
        raise RuntimeError(f"No images found under {dl_root}")
    move_dir(d, dst)
    return _count_images(dst)

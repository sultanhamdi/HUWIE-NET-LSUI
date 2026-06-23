"""Download LSUI + UIEB + pretrained HUWIE-Net weights to local ./data directory.

Usage:
    python download.py
"""
import os
import subprocess
import sys
import urllib.request

import yaml


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "configs", "base.yaml")) as f:
        cfg = yaml.safe_load(f)

    a = cfg["assets"]
    p = cfg["paths"]
    dl = os.path.join(p["data_root"], "_dl")
    os.makedirs(dl, exist_ok=True)

    from src.assets import arrange_paired, arrange_single, _count_images

    def kaggle_dl(slug, sub):
        dst = os.path.join(dl, sub)
        if os.path.exists(dst) and any(os.scandir(dst)):
            print(f"[download] {sub} already present, skip")
            return dst
        os.makedirs(dst, exist_ok=True)
        print(f"[download] kaggle {slug} -> {dst}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", dst, "--unzip"],
            check=True,
        )
        return dst

    # LSUI (raw + ref in one dataset)
    if not (os.path.isdir(p["lsui_raw"]) and os.path.isdir(p["lsui_ref"])):
        root = kaggle_dl(a["kaggle"]["lsui"], "lsui")
        nr, nf = arrange_paired(root, p["lsui_raw"], p["lsui_ref"])
        print(f"[download] LSUI arranged: {nr} raw / {nf} ref")
    else:
        print("[download] LSUI already arranged, skip")

    # UIEB raw + reference (two separate datasets)
    if not os.path.isdir(p["uieb_raw"]):
        root = kaggle_dl(a["kaggle"]["uieb_raw"], "uieb_raw")
        n = arrange_single(root, p["uieb_raw"])
        print(f"[download] UIEB raw arranged: {n}")
    else:
        print("[download] UIEB raw already arranged, skip")

    if not os.path.isdir(p["uieb_ref"]):
        root = kaggle_dl(a["kaggle"]["uieb_ref"], "uieb_ref")
        n = arrange_single(root, p["uieb_ref"])
        print(f"[download] UIEB ref arranged: {n}")
    else:
        print("[download] UIEB ref already arranged, skip")

    # Pretrained HUWIE-Net weights
    if not os.path.exists(p["pretrained"]):
        os.makedirs(os.path.dirname(p["pretrained"]), exist_ok=True)
        url = a["pretrained_url"]
        print(f"[download] pretrained {url}")
        urllib.request.urlretrieve(url, p["pretrained"])
        print(f"[download] pretrained -> {p['pretrained']} "
              f"({os.path.getsize(p['pretrained'])/1e6:.1f} MB)")
    else:
        print("[download] pretrained already present, skip")

    # Validate
    counts = {
        "lsui_raw": _count_images(p["lsui_raw"]),
        "lsui_ref": _count_images(p["lsui_ref"]),
        "uieb_raw": _count_images(p["uieb_raw"]),
        "uieb_ref": _count_images(p["uieb_ref"]),
    }
    print(f"[download] counts: {counts}")
    assert counts["lsui_raw"] == counts["lsui_ref"] > 0, "LSUI pair count mismatch"
    assert counts["uieb_raw"] == counts["uieb_ref"] > 0, "UIEB pair count mismatch"
    assert os.path.exists(p["pretrained"]), "pretrained missing"
    print("[download] DONE")


if __name__ == "__main__":
    main()

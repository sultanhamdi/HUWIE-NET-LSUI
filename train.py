"""Train ablation variant(s).

Usage:
    python train.py --variant A4        # one variant
    python train.py --all               # A1 -> A2 -> A3 -> A4 sequentially
    python train.py --variant A1 --smoke  # 1-epoch sanity check
"""
import argparse

from src.config import load_config
from src.trainer import run_training

VARIANTS = ["A1", "A2", "A3", "A4"]


def main():
    parser = argparse.ArgumentParser(description="Train HuWie-Net ablation variants")
    parser.add_argument("--variant", type=str, default="A4", choices=VARIANTS,
                        help="Which variant to train (default: A4)")
    parser.add_argument("--all", action="store_true",
                        help="Train all variants sequentially (A1->A4)")
    parser.add_argument("--smoke", action="store_true",
                        help="1-epoch sanity check")
    args = parser.parse_args()

    targets = VARIANTS if args.all else [args.variant]
    for v in targets:
        print(f">>> training {v}")
        cfg = load_config(v)
        best = run_training(cfg, smoke=args.smoke)
        print(f">>> {v} best PSNR = {best:.2f}")


if __name__ == "__main__":
    main()

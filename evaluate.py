"""Evaluate trained variant(s) on UIEB-890 and build report tables.

Usage:
    python evaluate.py --variant A4
    python evaluate.py --all            # eval A1..A4 + write summary tables
"""
import argparse

from src.config import load_config
from src.evaluator import run_evaluation, build_summary

VARIANTS = ["A1", "A2", "A3", "A4"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate HuWie-Net ablation variants")
    parser.add_argument("--variant", type=str, default="A4", choices=VARIANTS,
                        help="Which variant to evaluate (default: A4)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all variants and build summary tables")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable test-time augmentation")
    args = parser.parse_args()

    targets = VARIANTS if args.all else [args.variant]
    for v in targets:
        print(f">>> evaluating {v}")
        cfg = load_config(v)
        run_evaluation(cfg, use_tta=not args.no_tta)

    if args.all:
        cfg = load_config(VARIANTS[0])
        build_summary(VARIANTS, cfg["paths"]["outputs"])


if __name__ == "__main__":
    main()

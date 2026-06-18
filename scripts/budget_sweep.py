#!/usr/bin/env python
"""Performance-budget sweep. Evaluates a checkpoint over the configured budget
ratios for the learned policy and the random/energy baselines, writing a CSV and
a Macro-F1 vs observed-% plot.

    python scripts/budget_sweep.py --ckpt results/checkpoints/best.pt \
        [--split test] [--modes learned random energy] [--out results/sweep]
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activesleep.utils import get_logger
from activesleep.data.dataset import make_loader
from activesleep.trainer import Trainer
from activesleep.budget import ratio_to_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--modes", nargs="+", default=["learned", "random", "energy"])
    ap.add_argument("--out", default="results/sweep")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = get_logger("budget_sweep")

    model, cfg, meta = Trainer.load_model(args.ckpt)
    P = meta["n_patches"]
    ratios = cfg["budget"]["eval_ratios"]
    loader, _ = make_loader(cfg, args.split, shuffle=False)
    trainer = Trainer(cfg, model, meta)

    rows = []
    for mode in args.modes:
        for r in ratios:
            B = ratio_to_patches(r, P)
            res = trainer.evaluate(loader, budget=B, mode=mode, with_calibration=True)
            rows.append({
                "mode": mode, "ratio": r, "budget": B,
                "observed_pct": res["observed_pct"],
                "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
                "kappa": res["kappa"], "ece": res.get("ece", float("nan")),
            })
            log.info(f"  {mode:7s} B={B:2d} ({100*r:4.0f}%) "
                     f"mF1 {res['macro_f1']:.4f} acc {res['accuracy']:.4f}")

    csv_path = os.path.join(args.out, f"sweep_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        for mode in args.modes:
            mr = [x for x in rows if x["mode"] == mode]
            mr.sort(key=lambda x: x["observed_pct"])
            plt.plot([x["observed_pct"] for x in mr],
                     [x["macro_f1"] for x in mr], marker="o", label=mode)
        plt.xlabel("observed patches (%)")
        plt.ylabel("Macro-F1")
        plt.title(f"Performance-budget ({meta['dataset']}, {args.split})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        png = os.path.join(args.out, f"sweep_{args.split}.png")
        plt.savefig(png, dpi=150)
        log.info(f"wrote {png}")
    except ImportError:
        log.info("matplotlib not available; skipped plot")


if __name__ == "__main__":
    main()

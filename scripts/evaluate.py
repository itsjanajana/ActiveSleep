#!/usr/bin/env python
"""Evaluate a checkpoint on a split (defaults to test), with calibration and, when
the dataset has CAP labels, CAP metrics. Supports cross-dataset evaluation by
pointing --data at a different processed dir.

    python scripts/evaluate.py --ckpt results/checkpoints/best.pt \
        [--split test] [--budget 6] [--mode learned] \
        [--data data/processed/cap]      # cross-dataset
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activesleep.utils import get_logger
from activesleep.data.dataset import make_loader
from activesleep.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--mode", default="learned", choices=["learned", "random", "energy"])
    ap.add_argument("--data", default=None, help="override processed dir (cross-dataset)")
    args = ap.parse_args()
    log = get_logger("evaluate")

    model, cfg, meta = Trainer.load_model(args.ckpt)
    if args.data:
        cfg["data"]["processed_dir"] = args.data
        eval_meta = json.load(open(os.path.join(args.data, "meta.json")))
    else:
        eval_meta = meta
    budget = args.budget if args.budget is not None else cfg["budget"]["target_patches"]

    loader, _ = make_loader(cfg, args.split, shuffle=False)
    trainer = Trainer(cfg, model, meta)
    res = trainer.evaluate(loader, budget=budget, mode=args.mode,
                           with_calibration=True, with_cap=eval_meta["has_cap"])

    log.info(f"split={args.split} budget={budget}/{meta['n_patches']} mode={args.mode}")
    log.info(f"  acc {res['accuracy']:.4f} | macro-F1 {res['macro_f1']:.4f} | "
             f"kappa {res['kappa']:.4f} | ECE {res.get('ece', float('nan')):.4f}")
    log.info(f"  per-class F1: " +
             ", ".join(f"{k} {v:.3f}" for k, v in res["per_class_f1"].items()))
    if "cap" in res:
        log.info(f"  CAP: AUPRC {res['cap']['auprc']:.4f} | F1 {res['cap']['f1']:.4f}")


if __name__ == "__main__":
    main()

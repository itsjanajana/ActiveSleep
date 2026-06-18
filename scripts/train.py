#!/usr/bin/env python
"""Train ActiveSleep.

    python scripts/train.py --config configs/base.yaml configs/sleepedf.yaml \
        [configs/experiment.yaml] [--set budget.target_patches=6 model.crossview=true]

Reads the processed dir's meta.json, builds the model from it, and runs the
three-phase curriculum. CAP is enabled purely by pointing --config at cap.yaml
(which sets the processed dir + lambda_cap); no code changes.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activesleep.utils import load_config, load_json, seed_everything, get_logger
from activesleep.data.dataset import make_loader
from activesleep.trainer import Trainer, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    log = get_logger("train")

    meta = load_json(os.path.join(cfg["data"]["processed_dir"], "meta.json"))
    train_loader, _ = make_loader(cfg, "train", shuffle=True)
    val_loader, _ = make_loader(cfg, "val", shuffle=False)
    log.info(f"dataset={meta['dataset']} P={meta['n_patches']} "
             f"summary_dim={meta['summary_dim']} has_cap={meta['has_cap']}")
    log.info(f"target budget = {cfg['budget']['target_patches']}/{meta['n_patches']} "
             f"patches | crossview={cfg['model']['crossview']} "
             f"lambda_cap={cfg['loss']['lambda_cap']}")

    model = build_model(cfg, meta)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"model params: {n_params/1e6:.2f}M")

    trainer = Trainer(cfg, model, meta)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

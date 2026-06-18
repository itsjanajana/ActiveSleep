#!/usr/bin/env python
"""Ablation driver. Trains a small matrix of configurations and collects test
metrics into one CSV. Each row toggles one design element so the table maps
directly to the proposal's ablations:

    full            : learned policy + stability + (crossview/CAP per config)
    no_stability    : beta_stability = 0
    no_crossview    : model.crossview = false
    random_select   : selection forced to random (policy disabled) at eval

CAP is included automatically when the processed dir has CAP labels.

    python scripts/ablate.py --config configs/base.yaml configs/sleepedf.yaml \
        [--out results/ablation]
"""
import os
import sys
import csv
import copy
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activesleep.utils import load_config, load_json, seed_everything, get_logger
from activesleep.data.dataset import make_loader
from activesleep.trainer import Trainer, build_model


VARIANTS = {
    "full": {},
    "no_stability": {"loss.beta_stability": 0.0},
    "no_crossview": {"model.crossview": False},
}


def _apply(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        node = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node[kk]
        node[keys[-1]] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="results/ablation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base = load_config(args.config, args.set)
    log = get_logger("ablate")
    meta = load_json(os.path.join(base["data"]["processed_dir"], "meta.json"))

    rows = []
    for name, ov in VARIANTS.items():
        seed_everything(base["seed"])
        cfg = _apply(base, ov)
        cfg["train"]["ckpt_dir"] = os.path.join(args.out, name)
        log.info(f"=== variant: {name} ===")
        tr_loader, _ = make_loader(cfg, "train", shuffle=True)
        va_loader, _ = make_loader(cfg, "val", shuffle=False)
        te_loader, _ = make_loader(cfg, "test", shuffle=False)

        model = build_model(cfg, meta)
        trainer = Trainer(cfg, model, meta)
        trainer.fit(tr_loader, va_loader)

        for eval_mode in (["learned", "random"] if name == "full" else ["learned"]):
            tag = name if eval_mode == "learned" else "random_select"
            res = trainer.evaluate(te_loader, budget=cfg["budget"]["target_patches"],
                                   mode=eval_mode, with_calibration=True,
                                   with_cap=meta["has_cap"])
            row = {"variant": tag, "accuracy": res["accuracy"],
                   "macro_f1": res["macro_f1"], "kappa": res["kappa"],
                   "ece": res.get("ece", float("nan"))}
            if "cap" in res:
                row["cap_auprc"] = res["cap"]["auprc"]
            rows.append(row)
            log.info(f"  [{tag}] mF1 {res['macro_f1']:.4f} acc {res['accuracy']:.4f}")

    csv_path = os.path.join(args.out, "ablation.csv")
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant"] + [k for k in keys if k != "variant"])
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

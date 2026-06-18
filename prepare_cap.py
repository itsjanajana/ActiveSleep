#!/usr/bin/env python
"""Preprocess the CAP Sleep Database into the same cache format as Sleep-EDF,
adding per-epoch or per-patch CAP Phase-A labels.

This is the ONLY file you need to run to bring CAP online — the model already
carries a CAP head and the trainer already does the masked multi-task loss.

Usage (after downloading CAP and setting cap.raw_root in configs/cap.yaml):
    python scripts/prepare_cap.py --config configs/base.yaml configs/cap.yaml

Output (config: cap.out_dir): one <stem>__<stem>.npz per recording with a `cap`
array, meta.json (has_cap=true), and splits.json. Then train with:
    python scripts/train.py --config configs/base.yaml configs/cap.yaml
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activesleep.utils import load_config, save_json, seed_everything, get_logger
from activesleep.data import cap as capmod
from activesleep.data.splits import make_splits, write_splits
from activesleep.data.signal import STAGE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    log = get_logger("prepare_cap")

    cc = cfg["cap"]
    out_dir = cc["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    gran = cfg["model"]["cap_granularity"]

    recs = capmod.list_recordings(cc["raw_root"], cc["subjects"])
    if not recs:
        log.info(f"no .edf/.txt pairs under {cc['raw_root']}")
        sys.exit(1)
    log.info(f"{len(recs)} recordings to process (cap_granularity={gran})")

    s = cfg["signal"]
    P = int(round(s["epoch_sec"] / s["patch_sec"]))
    patch_len = int(round(s["fs"] * s["patch_sec"]))
    summary_dim = None
    stage_hist = np.zeros(len(STAGE_NAMES), dtype=np.int64)
    cap_pos = cap_scored = 0
    subject_ids, n_ok = [], 0

    for k, (edf, ann, stem) in enumerate(recs):
        try:
            arr = capmod.process_recording(edf, ann, cfg)
        except Exception as e:                         # noqa: BLE001
            log.info(f"  [skip] {stem}: {e}")
            continue
        if arr["patches"] is None:
            log.info(f"  [skip] {stem}: no labelled epochs after trim")
            continue

        assert arr["patches"].shape[1:] == (P, patch_len)
        assert arr["labels"].min() >= 0 and arr["labels"].max() <= 4
        assert arr["cap"] is not None, "CAP recording produced no cap array"
        if gran == "patch":
            assert arr["cap"].shape == (len(arr["labels"]), P)
        else:
            assert arr["cap"].shape == (len(arr["labels"]),)
        summary_dim = arr["patch_summary"].shape[-1]

        scored = arr["cap"] >= 0
        cap_scored += int(scored.sum())
        cap_pos += int((arr["cap"] == 1).sum())
        for c in range(len(STAGE_NAMES)):
            stage_hist[c] += int((arr["labels"] == c).sum())
        subject_ids.append(stem)
        n_ok += 1

        np.savez_compressed(
            os.path.join(out_dir, f"{stem}__{stem}.npz"),
            patches=arr["patches"], labels=arr["labels"],
            patch_summary=arr["patch_summary"], cap=arr["cap"].astype(np.int8),
            channel=np.array(arr["channel"]), fs=np.array(arr["fs"]),
        )
        log.info(f"  [{k+1}/{len(recs)}] {stem}: {len(arr['labels'])} epochs "
                 f"(ch={arr['channel']})")

    if n_ok == 0:
        log.info("nothing processed")
        sys.exit(1)

    splits = make_splits(subject_ids, seed=cfg["seed"])
    write_splits(splits, out_dir)
    save_json({
        "dataset": "cap",
        "fs": s["fs"], "epoch_sec": s["epoch_sec"], "patch_sec": s["patch_sec"],
        "n_patches": P, "patch_len": patch_len,
        "summary_dim": int(summary_dim), "summary_type": s["summary_type"],
        "norm": s["norm"], "classes": STAGE_NAMES,
        "has_cap": True, "cap_granularity": gran,
        "n_recordings": n_ok, "n_subjects": len(set(subject_ids)),
    }, os.path.join(out_dir, "meta.json"))

    total = stage_hist.sum()
    dist = ", ".join(f"{STAGE_NAMES[i]} {100*stage_hist[i]/total:.1f}%"
                     for i in range(len(STAGE_NAMES)))
    log.info(f"stage distribution: {dist}")
    if cap_scored:
        log.info(f"CAP Phase-A positives: {100*cap_pos/cap_scored:.1f}% of scored units")
    log.info(f"splits: train {len(splits['train'])} / val {len(splits['val'])} / "
             f"test {len(splits['test'])} subjects")
    log.info(f"wrote {n_ok} recordings to {out_dir}")


if __name__ == "__main__":
    main()

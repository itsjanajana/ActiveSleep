#!/usr/bin/env python
"""Preprocess Sleep-EDF Expanded into per-recording .npz + meta.json + splits.json.

Usage:
    python scripts/prepare_sleepedf.py --config configs/base.yaml configs/sleepedf.yaml \
        [--set sleepedf.raw_root=/data/sleep-edf-database-expanded-1.0.0]

Output (config: sleepedf.out_dir): one <subject>__<stem>.npz per recording, a
meta.json the loader/model read, and splits.json (the leakage guard).
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activesleep.utils import load_config, save_json, seed_everything, get_logger
from activesleep.data import sleepedf as sedf
from activesleep.data.splits import make_splits, write_splits
from activesleep.data.signal import STAGE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    log = get_logger("prepare_sleepedf")

    sc = cfg["sleepedf"]
    out_dir = sc["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    recs = sedf.list_recordings(sc["raw_root"], sc["cohorts"])
    recs = sedf.resolve_subjects(recs, sc["subjects"])
    if not recs:
        log.info(f"no recordings under {sc['raw_root']} for cohorts={sc['cohorts']}")
        sys.exit(1)
    log.info(f"{len(recs)} recordings to process")

    s = cfg["signal"]
    P = int(round(s["epoch_sec"] / s["patch_sec"]))
    patch_len = int(round(s["fs"] * s["patch_sec"]))
    stage_hist = np.zeros(len(STAGE_NAMES), dtype=np.int64)
    summary_dim = None
    subject_ids, n_ok = [], 0

    for k, (psg, hyp, stem) in enumerate(recs):
        sid = sedf.subject_id(stem)
        try:
            arr = sedf.process_recording(psg, hyp, cfg)
        except Exception as e:                         # noqa: BLE001
            log.info(f"  [skip] {stem}: {e}")
            continue
        if arr["patches"] is None:
            log.info(f"  [skip] {stem}: no labelled epochs after trim")
            continue

        # invariants (the only "tests" we keep)
        assert arr["patches"].ndim == 3 and arr["patches"].shape[1:] == (P, patch_len), \
            f"patch shape {arr['patches'].shape} != (*, {P}, {patch_len})"
        assert arr["labels"].min() >= 0 and arr["labels"].max() <= 4
        assert not np.isnan(arr["patches"]).any()
        summary_dim = arr["patch_summary"].shape[-1]

        for c in range(len(STAGE_NAMES)):
            stage_hist[c] += int((arr["labels"] == c).sum())
        subject_ids.append(sid)
        n_ok += 1

        np.savez_compressed(
            os.path.join(out_dir, f"{sid}__{stem}.npz"),
            patches=arr["patches"], labels=arr["labels"],
            patch_summary=arr["patch_summary"],
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
        "dataset": "sleepedf",
        "fs": s["fs"], "epoch_sec": s["epoch_sec"], "patch_sec": s["patch_sec"],
        "n_patches": P, "patch_len": patch_len,
        "summary_dim": int(summary_dim), "summary_type": s["summary_type"],
        "norm": s["norm"], "classes": STAGE_NAMES,
        "has_cap": False, "cap_granularity": cfg["model"]["cap_granularity"],
        "n_recordings": n_ok, "n_subjects": len(set(subject_ids)),
    }, os.path.join(out_dir, "meta.json"))

    total = stage_hist.sum()
    dist = ", ".join(f"{STAGE_NAMES[i]} {100*stage_hist[i]/total:.1f}%"
                     for i in range(len(STAGE_NAMES)))
    log.info(f"stage distribution: {dist}")
    log.info(f"splits: train {len(splits['train'])} / val {len(splits['val'])} / "
             f"test {len(splits['test'])} subjects")
    log.info(f"wrote {n_ok} recordings to {out_dir}")


if __name__ == "__main__":
    main()

"""Subject-wise splits. Deterministic from the seed, written once, read by every
script — the single leakage guard (no subject in two splits)."""
import os

import numpy as np

from ..utils import save_json, load_json


def make_splits(subject_ids, ratios=(0.6, 0.2, 0.2), seed=1):
    """Partition unique subject ids into train/val/test by the given ratios."""
    subs = sorted(set(subject_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    return {
        "train": sorted(subs[:n_tr]),
        "val": sorted(subs[n_tr:n_tr + n_va]),
        "test": sorted(subs[n_tr + n_va:]),
    }


def write_splits(splits, processed_dir):
    save_json(splits, os.path.join(processed_dir, "splits.json"))


def read_splits(processed_dir):
    return load_json(os.path.join(processed_dir, "splits.json"))

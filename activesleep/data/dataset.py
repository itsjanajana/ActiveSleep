"""Torch Dataset over cached per-subject .npz files.

Yields a K-epoch context window; the center epoch is the prediction target. Every
epoch in the window gets its own budgeted patch selection inside the model. Context
is edge-clamped *within a recording* so windows never cross subject/recording
boundaries (leakage-safe) and every epoch still receives a label.

Robust to missing CAP: if a recording has no `cap` array, the loader returns -1
(ignore index), so a model/trainer built for multi-task learning runs unchanged on
Sleep-EDF. This is what lets CAP drop in with no code edits.
"""
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from .splits import read_splits


class SleepWindowDataset(Dataset):
    def __init__(self, processed_dir, split, context, cap_granularity="epoch",
                 preload=True):
        self.dir = processed_dir
        self.K = context
        self.half = context // 2
        self.cap_granularity = cap_granularity
        self.preload = preload

        split_subs = set(read_splits(processed_dir)[split])
        meta_path = os.path.join(processed_dir, "meta.json")
        import json
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.P = self.meta["n_patches"]

        files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
        self.recordings = []   # list of dict handles
        self.index = []        # (rec_idx, center_epoch_idx)
        for fp in files:
            sid = os.path.basename(fp).split("__")[0]
            if sid not in split_subs:
                continue
            rec = self._open(fp)
            n = len(rec["labels"])
            if n == 0:
                continue
            ridx = len(self.recordings)
            self.recordings.append(rec)
            self.index.extend((ridx, i) for i in range(n))

        if not self.index:
            raise RuntimeError(f"no epochs for split={split!r} in {processed_dir}")

    def _open(self, fp):
        if self.preload:
            z = np.load(fp, allow_pickle=False)
            rec = {k: z[k] for k in z.files}
        else:
            rec = {"_path": fp, "_mmap": np.load(fp, mmap_mode="r")}
            for k in rec["_mmap"].files:
                rec[k] = rec["_mmap"][k]
        rec["has_cap"] = "cap" in rec and rec["cap"].size > 0
        return rec

    def __len__(self):
        return len(self.index)

    def _window_idx(self, n, center):
        return [min(max(0, j), n - 1) for j in range(center - self.half, center + self.half + 1)]

    def __getitem__(self, i):
        ridx, center = self.index[i]
        rec = self.recordings[ridx]
        n = len(rec["labels"])
        idx = self._window_idx(n, center)

        patches = rec["patches"][idx]            # [K, P, L]
        summary = rec["patch_summary"][idx]      # [K, P, D]
        label = int(rec["labels"][center])       # center target

        if rec["has_cap"]:
            cap_center = rec["cap"][center]
            cap = np.asarray(cap_center, dtype=np.int64)
        else:
            cap = (np.full(self.P, -1, dtype=np.int64)
                   if self.cap_granularity == "patch"
                   else np.int64(-1))

        return {
            "patches": torch.from_numpy(np.ascontiguousarray(patches)).float(),
            "summary": torch.from_numpy(np.ascontiguousarray(summary)).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "cap": torch.as_tensor(cap, dtype=torch.long),
            "has_cap": bool(rec["has_cap"]),
        }


def make_loader(cfg, split, shuffle):
    from torch.utils.data import DataLoader
    ds = SleepWindowDataset(
        cfg["data"]["processed_dir"], split,
        context=cfg["data"]["context"],
        cap_granularity=cfg["model"]["cap_granularity"],
        preload=cfg["data"]["preload"],
    )
    loader = DataLoader(
        ds, batch_size=cfg["data"]["batch_size"], shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"], drop_last=shuffle,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, ds

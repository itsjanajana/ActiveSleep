"""Sleep-EDF Expanded specifics: PSG/Hypnogram pairing, stage mapping.

Filename convention (both cohorts), e.g. SC4001E0-PSG.edf / SC4001EC-Hypnogram.edf:
    stem[0:2] cohort (SC=cassette, ST=telemetry)
    stem[3:5] subject number   -> two recordings of one subject share these
    stem[5]   night (1/2)
So the subject id (unique across cohorts) is stem[:5], e.g. "SC400", "ST701".
"""
import os
import glob

import numpy as np

from .signal import read_channel, intervals_to_epoch_labels, process_signal

# Sleep-EDF hypnogram descriptions -> 5-class (S3+S4 merged into N3).
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage REM": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}

_COHORT_DIR = {"cassette": "sleep-cassette", "telemetry": "sleep-telemetry"}


def subject_id(stem):
    """stem like 'SC4001E0-PSG' or 'SC4001E0' -> 'SC400'."""
    return os.path.basename(stem)[:5]


def list_recordings(raw_root, cohorts):
    """Return list of (psg_path, hyp_path, stem) for the requested cohorts."""
    out = []
    for c in cohorts:
        d = os.path.join(raw_root, _COHORT_DIR[c])
        for psg in sorted(glob.glob(os.path.join(d, "*-PSG.edf"))):
            stem = os.path.basename(psg).replace("-PSG.edf", "")
            # hypnogram shares the first 7 chars (e.g. SC4001E), trailing letter differs
            matches = sorted(glob.glob(os.path.join(d, stem[:7] + "*-Hypnogram.edf")))
            if not matches:
                continue
            out.append((psg, matches[0], stem))
    return out


def resolve_subjects(recordings, subjects):
    """Filter recordings by the `subjects` config (all / alias / explicit list)."""
    if subjects == "all":
        return recordings
    if subjects == "edf20":
        keep = {f"SC4{n:02d}" for n in range(20)}
    elif subjects == "edf78":
        keep = {subject_id(s) for _, _, s in recordings if s.startswith("SC")}
    elif isinstance(subjects, (list, tuple)):
        keep = set(subjects)
    else:
        raise ValueError(f"unknown subjects spec {subjects!r}")
    return [r for r in recordings if subject_id(r[2]) in keep]


def process_recording(psg_path, hyp_path, cfg):
    """Read one Sleep-EDF recording -> cached-array dict (+ channel, fs)."""
    import mne

    s = cfg["signal"]
    sig, fs, ch, _ = read_channel(
        psg_path, cfg["sleepedf"]["channel"],
        s["fs"], s["l_freq"], s["h_freq"], s["notch"],
    )

    ann = mne.read_annotations(hyp_path)
    intervals = [
        (on, dur, STAGE_MAP.get(desc, -1))
        for on, dur, desc in zip(ann.onset, ann.duration, ann.description)
    ]
    n_total = len(sig) // int(round(fs * s["epoch_sec"]))
    labels_full = intervals_to_epoch_labels(intervals, n_total, s["epoch_sec"])

    arrays = process_signal(sig, fs, labels_full, cfg, phase_a_intervals=None)
    arrays["channel"] = ch
    arrays["fs"] = fs
    return arrays

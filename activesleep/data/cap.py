"""CAP Sleep Database specifics: channel choice, stage mapping, Phase-A parsing.

Shipped now so that bringing CAP online needs no edits elsewhere — only running
scripts/prepare_cap.py with configs/cap.yaml. The CAP .txt annotation layout is
auto-detected (header row containing Time/Event/Duration). If a particular file
in your copy uses a different column order, only `_parse_annotation_file` below
needs adjusting; the rest of the pipeline (and the model/trainer) are untouched.

Macro-stage events:  SLEEP-S0(=W) S1 S2 S3 S4 REM MT
Phase-A events:      MCAP-A1 / A2 / A3   (matched loosely as A[123])
"""
import os
import re
import glob
import datetime as dt

import numpy as np

from .signal import read_channel, intervals_to_epoch_labels, process_signal

# CAP macro stage token -> 5-class.
_STAGE_TOKEN = {
    "S0": 0, "W": 0, "WK": 0, "WAKE": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3, "S4": 3,
    "REM": 4, "R": 4,
    "MT": -1,
}
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
# Phase-A and stage markers are matched as WHOLE columns (re.fullmatch), never as
# substrings — otherwise the channel/location field (e.g. "C4-A1", where A1 is the
# mastoid reference) is mistaken for a Phase-A event. This was a real bug caught in
# testing; do not relax these to search().
_PHASE_A_RE = re.compile(r"(?:MCAP-)?A[123]", re.I)        # use with fullmatch
_SLEEP_EVENT_RE = re.compile(r"SLEEP-([A-Za-z0-9]+)")      # event-column macro stage
_INT_RE = re.compile(r"\d+")


def list_recordings(raw_root, subjects):
    """Return list of (edf_path, ann_path, stem). Pairs *.edf with same-stem *.txt."""
    out = []
    for edf in sorted(glob.glob(os.path.join(raw_root, "*.edf"))):
        stem = os.path.splitext(os.path.basename(edf))[0]
        ann = os.path.join(raw_root, stem + ".txt")
        if not os.path.exists(ann):
            continue
        if subjects != "all" and stem not in set(subjects):
            continue
        out.append((edf, ann, stem))
    return out


def _to_seconds(time_str, start_time):
    """Wall-clock hh:mm:ss -> seconds since recording start (handles midnight wrap)."""
    m = _TIME_RE.search(time_str)
    if not m:
        return None
    h, mi, sec = (int(g) for g in m.groups())
    t = h * 3600 + mi * 60 + sec
    delta = t - start_time
    if delta < 0:
        delta += 24 * 3600
    return delta


def _row_cols(line):
    cols = re.split(r"\t|\s{2,}", line.strip())
    return [c for c in cols if c != ""]


def _row_onset_duration(cols, start_time):
    onset = None
    for c in cols:
        if _TIME_RE.search(c):
            onset = _to_seconds(c, start_time)
            break
    duration = 30
    for c in reversed(cols):
        if _INT_RE.fullmatch(c):
            duration = int(c)
            break
    return onset, duration


def _parse_annotation_file(ann_path, start_time):
    """Parse a CAP scoring .txt into macro-stage intervals and Phase-A spans.

    Returns (intervals, phase_a) where
        intervals = list of (onset_sec, duration_sec, stage_int)
        phase_a   = list of (onset_sec, offset_sec)

    Markers are matched as whole columns. Phase-A events are coded MCAP-A1/A2/A3
    (the macro hypnogram comes from SLEEP-Sx events); if a file lacks SLEEP-x
    events we fall back to the first "Sleep Stage" column on non-Phase-A rows.
    """
    with open(ann_path, "r", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "duration" in low and ("event" in low or "type" in low) and "time" in low:
            header_idx = i
            break
    rows = lines[header_idx + 1:] if header_idx is not None else lines

    phase_a = []
    intervals_evt = []   # from SLEEP-x event column (preferred)
    intervals_col0 = []  # from the first stage column (fallback)
    for ln in rows:
        if not ln.strip():
            continue
        cols = _row_cols(ln)
        if len(cols) < 2:
            continue
        onset, duration = _row_onset_duration(cols, start_time)
        if onset is None:
            continue

        is_phase_a = any(_PHASE_A_RE.fullmatch(c) for c in cols)
        if is_phase_a:
            phase_a.append((onset, onset + duration))

        for c in cols:                                   # preferred: SLEEP-x event
            m = _SLEEP_EVENT_RE.fullmatch(c)
            if m:
                st = _STAGE_TOKEN.get(m.group(1).upper())
                if st is not None:
                    intervals_evt.append((onset, duration, st))
                break

        if not is_phase_a:                               # fallback: column 0 stage
            st = _STAGE_TOKEN.get(cols[0].upper())
            if st is not None:
                intervals_col0.append((onset, duration, st))

    intervals = intervals_evt if intervals_evt else intervals_col0
    return intervals, phase_a


def _alignment_warning(phase_a, labels_full, epoch_sec):
    """Sanity check: most Phase-A spans should land in NREM. Print if they don't."""
    if not phase_a:
        return
    nrem = (labels_full >= 1) & (labels_full <= 3)
    hits = 0
    for on, _ in phase_a:
        i = int(on / epoch_sec)
        if 0 <= i < len(nrem) and nrem[i]:
            hits += 1
    frac = hits / max(1, len(phase_a))
    if frac < 0.7:
        print(
            f"  [warn] only {frac:.0%} of Phase-A spans fall in NREM — check the "
            f"annotation/signal time alignment for this recording."
        )


def process_recording(edf_path, ann_path, cfg):
    """Read one CAP recording -> cached-array dict (+ channel, fs), with cap labels."""
    s = cfg["signal"]
    sig, fs, ch, meas_date = read_channel(
        edf_path, cfg["cap"]["channel_priority"],
        s["fs"], s["l_freq"], s["h_freq"], s["notch"],
    )

    if isinstance(meas_date, dt.datetime):
        start_time = meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second
    else:
        start_time = 0  # fall back to assuming annotation clock starts at 0

    intervals, phase_a = _parse_annotation_file(ann_path, start_time)
    n_total = len(sig) // int(round(fs * s["epoch_sec"]))
    labels_full = intervals_to_epoch_labels(intervals, n_total, s["epoch_sec"])
    _alignment_warning(phase_a, labels_full, s["epoch_sec"])

    arrays = process_signal(sig, fs, labels_full, cfg, phase_a_intervals=phase_a)
    arrays["channel"] = ch
    arrays["fs"] = fs
    return arrays

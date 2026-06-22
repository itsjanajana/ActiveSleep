"""Dataset-agnostic signal preprocessing.

Everything that is shared between Sleep-EDF and CAP lives here so that adding a
new dataset only means writing a thin `prepare_*.py` + a dataset-specific module
(annotation parsing / channel choice). Nothing in this file is Sleep-EDF- or
CAP-specific.

Cache schema (one .npz per recording) produced via `process_signal` + caller:
    patches        float32 [N, P, L]   normalized epochs split into P patches
    labels         int8    [N]         0..4 = W,N1,N2,N3,REM
    patch_summary  float32 [N, P, D]   cheap per-patch features for the policy
    cap            int8    [N] or [N,P] optional; -1 = unscored/absent
    subject_id, channel, fs            metadata
"""
import warnings

import numpy as np

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES = 5

# Bands (Hz) for the band-power summary. Sigma overlaps alpha/beta intentionally
# (spindle band) — these are features, not a partition.
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (11.0, 16.0),
    "beta": (16.0, 30.0),
}
_EPS = 1e-8


# ----------------------------------------------------------- EDF reading ----
def read_channel(edf_path, channel_candidates, fs, l_freq, h_freq, notch=None):
    """Read one EEG channel, resample, bandpass (+ optional notch).

    channel_candidates: a single name (str) or a priority list; the first present
    channel is used. Raises if none are found.

    Returns: (signal float32 [n_samples], fs int, used_channel str, meas_date)
    """
    import mne

    if isinstance(channel_candidates, str):
        channel_candidates = [channel_candidates]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    used = None
    for cand in channel_candidates:
        if cand in raw.ch_names:
            used = cand
            break
    if used is None:
        raise ValueError(
            f"None of {channel_candidates} in {edf_path}. Available: {raw.ch_names}"
        )

    raw.pick_channels([used])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.resample(fs)
        raw.filter(l_freq, h_freq, verbose="ERROR")
        if notch:
            raw.notch_filter(float(notch), verbose="ERROR")

    sig = raw.get_data()[0].astype(np.float32)
    meas_date = raw.info.get("meas_date", None)
    return sig, fs, used, meas_date


# --------------------------------------------------- labels from intervals --
def intervals_to_epoch_labels(intervals, n_epochs, epoch_sec):
    """Rasterize (onset_sec, duration_sec, stage_int) intervals to per-epoch labels.

    Unset epochs are -1. stage_int < 0 entries are skipped.
    """
    labels = np.full(n_epochs, -1, dtype=np.int8)
    for onset, dur, stage in intervals:
        if stage is None or stage < 0:
            continue
        s = int(round(onset / epoch_sec))
        e = int(round((onset + dur) / epoch_sec))
        s = max(0, s)
        e = min(n_epochs, e)
        if e > s:
            labels[s:e] = stage
    return labels


# -------------------------------------------------------------- wake trim --
def wake_trim_range(labels, epoch_sec, trim_min):
    """Index range [lo, hi) keeping only +/- trim_min around the sleep period.

    Sleep = any valid non-Wake epoch (label in 1..4). Returns (0, len) if no sleep.
    """
    sleep = np.where((labels >= 1) & (labels <= 4))[0]
    if sleep.size == 0:
        return 0, len(labels)
    pad = int(round(trim_min * 60 / epoch_sec))
    lo = max(0, sleep[0] - pad)
    hi = min(len(labels), sleep[-1] + pad + 1)
    return lo, hi


# ------------------------------------------------------ patch-level summary --
def compute_summary(patches, fs, summary_type="bandpower", raw_len=16):
    """Per-patch cheap features, standardized per-recording.

    patches: [N, P, L]  ->  [N, P, D]
        bandpower: D = 7  (log-energy, 5 relative band powers, zero-crossing rate)
        raw:       D = raw_len  (uniformly downsampled waveform)
    """
    N, P, L = patches.shape
    flat = patches.reshape(N * P, L)

    if summary_type == "raw":
        idx = np.linspace(0, L - 1, raw_len).round().astype(int)
        feats = flat[:, idx].astype(np.float32)
    elif summary_type == "bandpower":
        log_energy = np.log(flat.var(axis=1) + _EPS)[:, None]
        spec = np.abs(np.fft.rfft(flat, axis=1)) ** 2
        freqs = np.fft.rfftfreq(L, 1.0 / fs)
        total = spec.sum(axis=1) + _EPS
        bands = []
        for lo, hi in _BANDS.values():
            m = (freqs >= lo) & (freqs < hi)
            bands.append(spec[:, m].sum(axis=1) / total)
        bands = np.stack(bands, axis=1)
        zcr = (np.abs(np.diff(np.sign(flat), axis=1)) > 0).mean(axis=1)[:, None]
        feats = np.concatenate([log_energy, bands, zcr], axis=1).astype(np.float32)
    else:
        raise ValueError(f"unknown summary_type {summary_type!r}")

    # Standardize per-recording (controls cross-subject amplitude; keeps which
    # patches stand out within a recording — the cue the policy needs).
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + _EPS
    feats = (feats - mu) / sd
    return feats.reshape(N, P, -1)


# ------------------------------------------------------------ normalization --
def normalize_patches(patches, mode="epoch"):
    """patches: [N, P, L] -> normalized copy."""
    if mode == "none":
        return patches.astype(np.float32)
    if mode == "epoch":
        N, P, L = patches.shape
        x = patches.reshape(N, P * L)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + _EPS
        return ((x - mu) / sd).reshape(N, P, L).astype(np.float32)
    if mode == "patch":
        mu = patches.mean(axis=2, keepdims=True)
        sd = patches.std(axis=2, keepdims=True) + _EPS
        return ((patches - mu) / sd).astype(np.float32)
    raise ValueError(f"unknown norm mode {mode!r}")


# ------------------------------------------------- end-to-end recording op --
def process_signal(signal, fs, labels_full, cfg, phase_a_intervals=None):
    """Turn a continuous channel + per-epoch labels into cached arrays.

    signal:            float32 [n_samples] (already resampled/filtered)
    labels_full:       int8    [n_total_epochs] (-1 = unknown)
    cfg:               the merged config dict (uses cfg['signal'])
    phase_a_intervals: optional list of (onset_sec, offset_sec) CAP Phase-A spans;
                       if given, a `cap` array is returned (epoch or patch level
                       per cfg['model']['cap_granularity']).

    Returns dict with keys: patches, labels, patch_summary, cap (or None).
    """
    s = cfg["signal"]
    epoch_sec = s["epoch_sec"]
    spe = int(round(fs * epoch_sec))               # samples per epoch
    patch_len = int(round(fs * s["patch_sec"]))    # samples per patch
    if spe % patch_len != 0:
        raise ValueError(
            f"epoch ({spe} samples) not divisible by patch ({patch_len} samples)"
        )
    P = spe // patch_len

    n_ep = len(signal) // spe
    signal = signal[: n_ep * spe]
    labels_full = labels_full[:n_ep].copy()
    epochs = signal.reshape(n_ep, spe)             # [n_ep, spe]

    # epoch -> patch level Phase-A labels computed on the full timeline first,
    # so trimming/dropping stays consistent with staging labels.
    cap_full = None
    if phase_a_intervals is not None:
        cap_full = _phase_a_labels(
            phase_a_intervals, labels_full, n_ep, P, fs, epoch_sec, patch_len,
            granularity=cfg["model"]["cap_granularity"],
        )

    # wake trim, then drop unknown-label epochs
    lo, hi = wake_trim_range(labels_full, epoch_sec, s["wake_trim_min"])
    epochs, labels_full = epochs[lo:hi], labels_full[lo:hi]
    if cap_full is not None:
        cap_full = cap_full[lo:hi]
    keep = labels_full >= 0
    epochs, labels = epochs[keep], labels_full[keep].astype(np.int8)
    cap = cap_full[keep] if cap_full is not None else None

    if len(epochs) == 0:
        return {"patches": None, "labels": None, "patch_summary": None, "cap": None}

    patches = epochs.reshape(len(epochs), P, patch_len)               # [N,P,L]
    summary = compute_summary(
        patches, fs, s["summary_type"], s.get("raw_summary_len", 16)
    )
    patches = normalize_patches(patches, s["norm"])
    return {"patches": patches, "labels": labels, "patch_summary": summary, "cap": cap}


def _phase_a_labels(intervals, labels_full, n_ep, P, fs, epoch_sec, patch_len,
                    granularity):
    """Binary Phase-A overlap labels; -1 outside NREM-scored region.

    Phase-A is only defined during NREM, so epochs that are Wake/REM/unknown are
    left as -1 (ignore index) rather than 0, to avoid teaching "unscored == no CAP".
    """
    nrem = (labels_full >= 1) & (labels_full <= 3)  # N1..N3
    if granularity == "epoch":
        cap = np.full(n_ep, -1, dtype=np.int8)
        cap[nrem] = 0
        for on, off in intervals:
            s = max(0, int(np.floor(on / epoch_sec)))
            e = min(n_ep, int(np.ceil(off / epoch_sec)))
            for i in range(s, e):
                if nrem[i]:
                    cap[i] = 1
        return cap
    elif granularity == "patch":
        cap = np.full((n_ep, P), -1, dtype=np.int8)
        cap[nrem, :] = 0
        patch_sec = patch_len / fs
        for on, off in intervals:
            s = max(0, int(np.floor(on / patch_sec)))
            e = min(n_ep * P, int(np.ceil(off / patch_sec)))
            for g in range(s, e):
                ei, pi = divmod(g, P)
                if ei < n_ep and nrem[ei]:
                    cap[ei, pi] = 1
        return cap
    raise ValueError(f"unknown cap_granularity {granularity!r}")

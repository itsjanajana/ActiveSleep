"""Metrics. Staging: accuracy, macro-F1, Cohen's kappa, per-class F1. CAP: AUPRC,
F1 (masked). Calibration: Expected Calibration Error + reliability bins."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    average_precision_score, precision_recall_fscore_support,
)

from .data.signal import STAGE_NAMES


def staging_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class = f1_score(y_true, y_pred, labels=list(range(len(STAGE_NAMES))),
                         average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_f1": {STAGE_NAMES[i]: float(per_class[i])
                         for i in range(len(STAGE_NAMES))},
    }


def cap_metrics(y_true, y_prob, thresh=0.5):
    """y_true/y_prob flattened, with -1 entries already removed by the caller."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {"auprc": float("nan"), "f1": float("nan")}
    y_pred = (y_prob >= thresh).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"auprc": float(average_precision_score(y_true, y_prob)), "f1": float(f1)}


def expected_calibration_error(y_true, probs, n_bins=15):
    """probs: [N, C] softmax. Returns (ece, bin_confidence, bin_accuracy)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_conf, bin_acc = [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            continue
        c, a = conf[m].mean(), correct[m].mean()
        ece += (m.mean()) * abs(a - c)
        bin_conf.append(float(c))
        bin_acc.append(float(a))
    return float(ece), bin_conf, bin_acc

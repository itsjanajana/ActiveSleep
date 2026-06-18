"""Config loading, seeding, logging, and a FLOPs proxy."""
import os
import sys
import json
import random
import logging
from copy import deepcopy

import numpy as np


# ---------------------------------------------------------------- config ----
def _deep_merge(base, over):
    out = deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _coerce(v):
    """Best-effort string -> typed value for CLI overrides."""
    if not isinstance(v, str):
        return v
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x.strip()) for x in inner.split(",")]
    return v


def _set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _coerce(value)


def load_config(paths, overrides=None):
    """Merge a list of YAML files (left-to-right), then apply dotted overrides.

    overrides: list of "a.b.c=value" strings.
    """
    import yaml

    cfg = {}
    for p in paths:
        with open(p, "r") as f:
            part = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, part)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value: {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), v.strip())
    return cfg


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------- seeding ----
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------- logging ----
def get_logger(name="activesleep", logfile=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ----------------------------------------------------------- flops proxy ----
def flops_proxy(observed_patches, n_patches):
    """Relative compute proxy under sparse acquisition.

    A true sparse deployment encodes only the B observed patches, so encoder cost
    scales with observed/total. Returned value is in [0, 1]; multiply by the
    full-budget FLOPs to get an estimate.
    """
    return float(observed_patches) / float(n_patches)

"""Training curriculum: phase boundaries, budget annealing, and per-phase mode.

phase 1  (full-signal warmup)        : B = P,        mode = random (== full, no mask)
phase 2  (learned acquisition)       : B = P -> target (annealed), mode = learned,
                                       stability loss on
phase 3  (+ alignment + microstructure): B = target,  mode = learned,
                                       contrastive + CAP losses on
"""


def phase_of(epoch, total, p1_frac, p2_frac):
    """Return 1, 2, or 3 for the given epoch index."""
    e1 = p1_frac * total
    e2 = p2_frac * total
    if epoch < e1:
        return 1
    if epoch < e2:
        return 2
    return 3


def budget_of(epoch, total, p1_frac, p2_frac, n_patches, target):
    """Patches to keep this epoch (anneal P -> target across phase 2)."""
    ph = phase_of(epoch, total, p1_frac, p2_frac)
    if ph == 1:
        return n_patches
    if ph == 3:
        return target
    e1, e2 = p1_frac * total, p2_frac * total
    frac = (epoch - e1) / max(1e-6, (e2 - e1))             # 0 -> 1 across phase 2
    return int(round(n_patches + frac * (target - n_patches)))


def mode_of(epoch, total, p1_frac, p2_frac):
    return "random" if phase_of(epoch, total, p1_frac, p2_frac) == 1 else "learned"


def ratio_to_patches(ratio, n_patches):
    return max(1, int(round(ratio * n_patches)))

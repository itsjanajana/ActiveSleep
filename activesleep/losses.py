"""Loss terms: L = L_stage + lambda*L_cap + alpha*L_contrast + beta*L_stability."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def staging_loss(logits, labels):
    """5-class cross-entropy; ignores label == -1."""
    return F.cross_entropy(logits, labels, ignore_index=-1)


def cap_loss(logits, targets):
    """Binary CE over CAP labels, masked where target == -1.

    Works for epoch-level ([B] / [B]) and patch-level ([B,P] / [B,P]). Returns 0
    (no grad contribution) when nothing is scored, so the call is a no-op on
    datasets without CAP labels.
    """
    targets = targets.to(logits.dtype) if targets.dtype != torch.long else targets
    valid = targets >= 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[valid], targets[valid].float()
    )


def info_nce(z_time, z_tf, temp=0.1):
    """Symmetric InfoNCE aligning time-view and TF-view embeddings (in-batch negs)."""
    zt = F.normalize(z_time, dim=-1)
    zf = F.normalize(z_tf, dim=-1)
    logits = zt @ zf.t() / temp
    target = torch.arange(zt.size(0), device=zt.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def stability_loss(mask, probs, target_ratio, entropy_w=0.1):
    """Anti-collapse + entropy regularizer on patch selection.

    mask  [B,K,P] hard 0/1; probs [B,K,P] softmax scores.
    - diversity: per-position selection frequency over the batch should stay near
      the budget ratio (stops the policy always picking the same temporal region).
    - entropy bonus: keeps per-epoch selection distributions from degenerating.
    """
    pos_freq = mask.mean(dim=(0, 1))                       # [P]
    diversity = ((pos_freq - target_ratio) ** 2).sum()
    ent = -(probs * (probs.clamp_min(1e-8)).log()).sum(-1).mean()
    return diversity - entropy_w * ent

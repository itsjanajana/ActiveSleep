"""Acquisition policy: score patches from a cheap per-patch summary and select B
of P under a hard budget.

`mode` unifies the proposed method and the two baselines so they share the entire
pipeline (only the selection differs):
    learned : scored by the policy net, differentiable Top-B (straight-through)
    random  : B random patches (MASS-style stochastic masking control)
    energy  : Top-B by a per-patch energy proxy (heuristic acquisition control)

Forward returns (mask, scores, probs):
    mask  [.., P] hard 0/1 with exactly B ones; gradients flow via the soft term
    scores[.., P] raw scores (energy proxy under mode='energy')
    probs [.., P] softmax(scores) for the stability/entropy loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def differentiable_topb(scores, B, tau=1.0, gumbel=False, training=True):
    """Straight-through Top-B over the last dim. Forward = hard mask, backward = soft."""
    P = scores.shape[-1]
    B = max(1, min(int(B), P))
    s = scores
    if gumbel and training:
        u = torch.rand_like(s).clamp_(1e-6, 1 - 1e-6)
        s = s + (-torch.log(-torch.log(u)))
    soft = F.softmax(s / tau, dim=-1)
    idx = torch.topk(s, B, dim=-1).indices
    hard = torch.zeros_like(s).scatter(-1, idx, 1.0)
    mask = hard + (soft - soft.detach())   # value == hard; grad flows through soft
    return mask, soft


class AcquisitionPolicy(nn.Module):
    def __init__(self, summary_dim, hidden=64, tau=1.0, gumbel=False):
        super().__init__()
        self.tau = tau
        self.gumbel = gumbel
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, summary, budget, mode="learned"):
        """summary: [B, K, P, D] -> mask/scores/probs each [B, K, P]."""
        if mode == "learned":
            scores = self.net(summary).squeeze(-1)               # [B,K,P]
            mask, probs = differentiable_topb(
                scores, budget, self.tau, self.gumbel, self.training
            )
            return mask, scores, probs

        if mode == "random":
            scores = torch.rand(summary.shape[:-1], device=summary.device)
        elif mode == "energy":
            # energy proxy = norm of the (standardized) summary vector per patch
            scores = summary.norm(dim=-1)
        else:
            raise ValueError(f"unknown selection mode {mode!r}")

        P = scores.shape[-1]
        B = max(1, min(int(budget), P))
        idx = torch.topk(scores, B, dim=-1).indices
        mask = torch.zeros_like(scores).scatter(-1, idx, 1.0)
        probs = F.softmax(scores, dim=-1)
        return mask, scores, probs

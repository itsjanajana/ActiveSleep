from typing import Tuple

import torch
import torch.nn.functional as F

from torch_geometric.data import Data

from .config import ActiveLearningConfig


def mc_dropout_predict(
    model,
    data: Data,
    mc_passes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Monte Carlo dropout prediction.

    Returns mean predicted probabilities over mc_passes:
        probs: [N, C]
    """
    model.eval()
    original_training_mode = model.training

    all_probs = []
    with torch.no_grad():
        for _ in range(mc_passes):
            model.train()  # activate dropout for MC sampling
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

    # Restore original mode
    model.train(original_training_mode)

    stacked = torch.stack(all_probs, dim=0)  # [T, N, C]
    mean_probs = stacked.mean(dim=0)        # [N, C]
    return mean_probs


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [N, C]
    returns: [N] predictive entropy
    """
    logp = torch.log(torch.clamp(probs, min=1e-8))
    entropy = -(probs * logp).sum(dim=1)
    return entropy


def _normalize_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Normalize scores over masked entries into [0, 1]. Others are set to 0.
    """
    normalized = torch.zeros_like(scores)
    if mask.any():
        masked_scores = scores[mask]
        min_v = masked_scores.min()
        max_v = masked_scores.max()
        if max_v > min_v:
            normalized[mask] = (masked_scores - min_v) / (max_v - min_v)
    return normalized


def compute_acquisition_scores(
    mean_probs: torch.Tensor,          # [N, C]
    embeddings: torch.Tensor,          # [N, d]
    labels: torch.Tensor,              # [N]
    domains: torch.Tensor,             # [N]
    labeled_mask: torch.Tensor,        # [N] bool
    train_mask: torch.Tensor,          # [N] bool (training pool)
    al_cfg: ActiveLearningConfig,
) -> torch.Tensor:
    """
    Computes s(i) = λ_u U(i) + λ_c C(i) + λ_d D(i) over all nodes.
    Labeled nodes receive very negative scores to avoid re-selection.
    """
    device = embeddings.device
    N = embeddings.shape[0]

    # Masks for candidate unlabeled training samples
    unlabeled_train_mask = train_mask & (~labeled_mask)

    # 1. Uncertainty term U(i): predictive entropy
    entropy = predictive_entropy(mean_probs)  # [N]
    U = _normalize_scores(entropy, unlabeled_train_mask)

    # 2. Coverage term C(i): distance to nearest labeled sample in embedding space
    C = torch.zeros(N, device=device)
    if labeled_mask.any():
        labeled_indices = torch.nonzero(labeled_mask & train_mask, as_tuple=False).view(-1)
        unlabeled_indices = torch.nonzero(unlabeled_train_mask, as_tuple=False).view(-1)

        if len(unlabeled_indices) > 0 and len(labeled_indices) > 0:
            emb_labeled = embeddings[labeled_indices].detach()
            emb_unlabeled = embeddings[unlabeled_indices].detach()

            dists = torch.cdist(emb_unlabeled, emb_labeled, p=2)  # [Nu, Nl]
            min_dists, _ = torch.min(dists, dim=1)                # [Nu]
            C_unlabeled = torch.zeros(N, device=device)
            C_unlabeled[unlabeled_indices] = min_dists
            C = _normalize_scores(C_unlabeled, unlabeled_train_mask)

    # 3. Domain-balance term D(i): favor under-represented domains in labeled set
    D = torch.zeros(N, device=device)
    train_domains = domains[train_mask]
    train_domain_total = torch.bincount(train_domains, minlength=train_domains.max().item() + 1)

    labeled_domains = domains[labeled_mask & train_mask]
    if labeled_domains.numel() > 0:
        labeled_domain_counts = torch.bincount(
            labeled_domains,
            minlength=train_domain_total.shape[0],
        ).float()
        total_counts = train_domain_total.float().clamp(min=1.0)
        ratios = labeled_domain_counts / total_counts  # [num_domains]
        max_ratio = ratios.max()
        domain_penalty = max_ratio - ratios           # higher for under-labeled domains

        per_sample_penalty = domain_penalty[domains]
        D = _normalize_scores(per_sample_penalty, unlabeled_train_mask)

    # Combine
    scores = (
        al_cfg.lambda_uncertainty * U
        + al_cfg.lambda_coverage * C
        + al_cfg.lambda_domain * D
    )

    # Do not re-select labeled samples
    scores[labeled_mask] = -1e9
    # Also exclude non-training nodes from selection
    scores[~train_mask] = -1e9

    return scores

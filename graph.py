from typing import Tuple

import torch
from torch_geometric.data import Data


def build_knn_graph(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    k: int = 10,
) -> Data:
    """
    Build a k-NN graph over all embeddings.

    embeddings: [N, d]
    labels: [N]
    domains: [N]
    """
    device = embeddings.device
    N, d = embeddings.shape

    # Compute pairwise distances on CPU for robustness
    emb_cpu = embeddings.detach().cpu()
    dists = torch.cdist(emb_cpu, emb_cpu, p=2)  # [N, N]
    # Exclude self
    diag_idx = torch.arange(N)
    dists[diag_idx, diag_idx] = float("inf")

    # k nearest neighbors for each node
    knn_dists, knn_indices = torch.topk(dists, k=k, largest=False, dim=1)  # [N, k]

    # Edge index: directed edges (i -> j) for neighbors j
    row_indices = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)
    col_indices = knn_indices.reshape(-1)
    edge_index = torch.stack([row_indices, col_indices], dim=0).long()

    # Compute Gaussian weights
    sigma = torch.mean(knn_dists[knn_dists < float("inf")])
    if sigma <= 0:
        # fallback: uniform weights
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    else:
        flat_dists = knn_dists.reshape(-1)
        valid_dists = flat_dists[flat_dists < float("inf")]
        weights = torch.exp(- (valid_dists ** 2) / (2 * sigma ** 2))
        # Place into full array (non-neighbor distances are absent)
        edge_weight = weights

    data = Data(
        x=embeddings.to(device),
        y=labels.to(device),
        domain=domains.to(device),
        edge_index=edge_index.to(device),
        edge_weight=edge_weight.to(device),
        num_nodes=N,
    )

    return data

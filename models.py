from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import timm


class USFMAEBackbone(nn.Module):
    """
    Wrapper around a ViT-style backbone (compatible with USF-MAE).
    Uses timm to instantiate a model with global pooled features.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = False) -> None:
        super().__init__()
        # num_classes=0 returns features instead of logits
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.embedding_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] normalized input
        returns: [B, embedding_dim]
        """
        return self.backbone(x)


class GraphClassifier(nn.Module):
    """
    GNN-based classifier over a k-NN graph with domain embeddings.

    Input Data should have:
        - x: [N, embedding_dim] node features
        - edge_index: [2, E] graph edges
        - domain: [N] domain indices
        - y: [N] labels (for loss)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_domains: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        domain_emb_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.domain_emb = nn.Embedding(num_domains, domain_emb_dim)
        self.dropout_p = dropout

        self.convs = nn.ModuleList()
        in_dim = embedding_dim + domain_emb_dim
        for layer_idx in range(num_layers):
            out_dim = hidden_dim
            self.convs.append(SAGEConv(in_dim if layer_idx == 0 else hidden_dim, out_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """
        data: PyG Data object with x, edge_index, domain
        returns: logits [N, num_classes]
        """
        x = data.x  # [N, embedding_dim]
        edge_index = data.edge_index  # [2, E]
        domain = data.domain  # [N]

        domain_feat = self.domain_emb(domain)  # [N, domain_emb_dim]
        h = torch.cat([x, domain_feat], dim=1)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        logits = self.classifier(h)
        return logits

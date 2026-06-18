"""MASS-style hierarchical backbone with a global prompt.

Same structure as MASS (patch-level + epoch-level modeling, global prompt that
aggregates visible content) but visibility comes from the acquisition policy
instead of random masking.

    1. patch transformer  : self-attention over P tokens within each epoch
    2. epoch embedding     : mask-weighted pool of patch tokens -> one vector/epoch
    3. global prompt       : a learned query cross-attends over all visible patch
                             tokens in the K-epoch window -> context vector
    4. epoch transformer   : attention over [prompt, K epoch embeddings]

Exposes the center epoch's contextual representation (for staging), the center
epoch's patch tokens (for patch-level CAP), and the center epoch embedding
(time-view for cross-view contrastive alignment).
"""
import torch
import torch.nn as nn


def _encoder(d_model, n_heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
        dropout=dropout, batch_first=True, activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class PromptBackbone(nn.Module):
    def __init__(self, d_model, n_heads, patch_layers, epoch_layers, context,
                 dropout=0.1):
        super().__init__()
        self.K = context
        self.patch_tf = _encoder(d_model, n_heads, patch_layers, dropout)
        self.prompt_q = nn.Parameter(torch.zeros(1, 1, d_model))
        self.prompt_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.epoch_tf = _encoder(d_model, n_heads, epoch_layers, dropout)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.prompt_q, std=0.02)

    def forward(self, tokens, mask):
        """tokens [B,K,P,d], mask [B,K,P] -> dict of representations."""
        B, K, P, d = tokens.shape
        center = K // 2

        # 1. patch-level self-attention within each epoch
        ptok = self.patch_tf(tokens.reshape(B * K, P, d)).reshape(B, K, P, d)

        # 2. epoch embedding = mask-weighted mean over patch tokens
        w = mask.unsqueeze(-1)
        denom = w.sum(dim=2).clamp_min(1.0)
        epoch_emb = (ptok * w).sum(dim=2) / denom            # [B,K,d]

        # 3. global prompt from all visible patch tokens in the window
        vis = ptok.reshape(B, K * P, d)
        key_pad = (mask.reshape(B, K * P) < 0.5)             # True where masked-out
        q = self.prompt_q.expand(B, -1, -1)
        prompt, _ = self.prompt_attn(q, vis, vis, key_padding_mask=key_pad)  # [B,1,d]

        # 4. epoch-level attention over [prompt, epoch embeddings]
        seq = torch.cat([prompt, epoch_emb], dim=1)          # [B,1+K,d]
        seq = self.epoch_tf(seq)
        ctx_center = self.norm(seq[:, 1 + center])           # contextual center rep

        return {
            "ctx_center": ctx_center,                        # [B,d] -> staging
            "patch_center": ptok[:, center],                 # [B,P,d] -> patch CAP
            "epoch_center": epoch_emb[:, center],            # [B,d] -> time view
        }

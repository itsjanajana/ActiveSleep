"""Per-patch encoder. A 1D conv stem maps each acquired patch waveform to a token;
unselected patches are replaced by a learned mask token so positions stay fixed
(P tokens per epoch regardless of budget). Patch- and epoch-position embeddings
are added here.
"""
import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    def __init__(self, patch_len, d_model, n_patches, context, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.patch_pos = nn.Parameter(torch.zeros(n_patches, d_model))
        self.epoch_pos = nn.Parameter(torch.zeros(context, d_model))
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.epoch_pos, std=0.02)

    def forward(self, patches, mask):
        """patches [B,K,P,L], mask [B,K,P] -> tokens [B,K,P,d], plus mask.

        For batched training every patch is encoded then gated by the mask. A true
        sparse deployment would encode only the B selected patches; that cost is
        reported via the FLOPs proxy (utils.flops_proxy).
        """
        B, K, P, L = patches.shape
        x = patches.reshape(B * K * P, 1, L)
        tok = self.stem(x).squeeze(-1).reshape(B, K, P, self.d_model)

        m = mask.unsqueeze(-1)                              # [B,K,P,1]
        tok = m * tok + (1.0 - m) * self.mask_token         # gate unselected
        tok = tok + self.patch_pos.view(1, 1, P, -1)
        tok = tok + self.epoch_pos.view(1, K, 1, -1)
        return self.drop(tok)

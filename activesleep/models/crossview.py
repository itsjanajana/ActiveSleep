"""Cross-view (MC2SleepNet-style) time-frequency branch.

Builds a spectrogram of the center epoch and encodes it; the contrastive loss
(in losses.py) aligns this TF embedding with the backbone's time-view epoch
embedding. Under a strict sensing budget the spectrogram is built only from the
selected patches; with `full_epoch=True` (default proxy) it uses the whole epoch.
"""
import torch
import torch.nn as nn


class CrossViewEncoder(nn.Module):
    def __init__(self, d_model, patch_len, n_patches, n_fft=256, full_epoch=True):
        super().__init__()
        self.full_epoch = full_epoch
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, d_model)

    def forward(self, patches, mask):
        """patches [B,K,P,L], mask [B,K,P] -> TF embedding [B,d]."""
        B, K, P, L = patches.shape
        center = K // 2
        ep = patches[:, center]                     # [B,P,L]
        if not self.full_epoch:
            ep = ep * mask[:, center].unsqueeze(-1)  # zero unselected patches
        wav = ep.reshape(B, P * L)                   # [B, spe]

        spec = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop,
            window=self.window.to(wav.device), return_complex=True, center=True,
        ).abs()                                      # [B, F, T]
        spec = torch.log1p(spec).unsqueeze(1)        # [B,1,F,T]
        feat = self.cnn(spec).flatten(1)             # [B,32]
        return self.proj(feat)                       # [B,d]

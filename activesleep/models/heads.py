"""Task heads. The CAP head is always constructed (epoch- or patch-level per
config) so the architecture is identical whether or not CAP labels are present;
without CAP data the trainer simply doesn't apply its loss."""
import torch.nn as nn

from ..data.signal import N_CLASSES


class StageHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, N_CLASSES)

    def forward(self, ctx):           # ctx [B,d] -> [B,5]
        return self.fc(ctx)


class CapHead(nn.Module):
    def __init__(self, d_model, granularity="epoch"):
        super().__init__()
        self.granularity = granularity
        self.fc = nn.Linear(d_model, 1)

    def forward(self, epoch_emb, patch_tokens):
        """epoch-level: [B] from epoch_emb; patch-level: [B,P] from patch tokens."""
        if self.granularity == "patch":
            return self.fc(patch_tokens).squeeze(-1)   # [B,P]
        return self.fc(epoch_emb).squeeze(-1)          # [B]

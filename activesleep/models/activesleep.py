"""ActiveSleep: policy -> encoder -> prompt backbone -> {staging, CAP} heads,
with an optional cross-view TF branch.

Forward signature carries the runtime budget and selection mode so training
(curriculum, budget anneal) and the baseline sweeps all use one code path.
"""
import torch.nn as nn

from .acquisition import AcquisitionPolicy
from .encoder import PatchEncoder
from .backbone import PromptBackbone
from .crossview import CrossViewEncoder
from .heads import StageHead, CapHead


class ActiveSleep(nn.Module):
    def __init__(self, cfg, meta):
        super().__init__()
        m = cfg["model"]
        d = m["d_model"]
        P = meta["n_patches"]
        L = meta["patch_len"]
        K = cfg["data"]["context"]
        D = meta["summary_dim"]

        self.use_crossview = m["crossview"]
        self.cap_granularity = m["cap_granularity"]

        self.policy = AcquisitionPolicy(
            D, hidden=m["policy_hidden"], tau=m["policy_tau"],
            gumbel=m["policy_gumbel"],
        )
        self.encoder = PatchEncoder(L, d, P, K, dropout=m["dropout"])
        self.backbone = PromptBackbone(
            d, m["n_heads"], m["patch_layers"], m["epoch_layers"], K,
            dropout=m["dropout"],
        )
        self.stage_head = StageHead(d)
        self.cap_head = CapHead(d, self.cap_granularity)
        if self.use_crossview:
            self.crossview = CrossViewEncoder(
                d, L, P, n_fft=m["stft_n_fft"],
                full_epoch=m["crossview_full_epoch"],
            )

    def forward(self, patches, summary, budget, mode="learned"):
        mask, scores, probs = self.policy(summary, budget, mode)
        tokens = self.encoder(patches, mask)
        rep = self.backbone(tokens, mask)

        out = {
            "stage_logits": self.stage_head(rep["ctx_center"]),
            "cap_logits": self.cap_head(rep["epoch_center"], rep["patch_center"]),
            "time_emb": rep["epoch_center"],
            "mask": mask,
            "scores": scores,
            "probs": probs,
        }
        out["tf_emb"] = self.crossview(patches, mask) if self.use_crossview else None
        return out

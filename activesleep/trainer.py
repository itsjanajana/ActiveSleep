"""Training loop with the three-phase curriculum, AMP, and checkpointing, plus a
shared evaluation routine used by train/evaluate/budget_sweep.
"""
import os

import numpy as np
import torch

from .losses import staging_loss, cap_loss, info_nce, stability_loss
from .metrics import staging_metrics, cap_metrics, expected_calibration_error
from . import budget as B
from .utils import get_logger, flops_proxy


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg, meta):
    from .models.activesleep import ActiveSleep
    return ActiveSleep(cfg, meta).to(_device())


class Trainer:
    def __init__(self, cfg, model, meta):
        self.cfg = cfg
        self.meta = meta
        self.model = model
        self.dev = _device()
        t = cfg["train"]
        self.epochs = t["epochs"]
        self.p1, self.p2 = t["phase1_frac"], t["phase2_frac"]
        self.P = meta["n_patches"]
        self.target = cfg["budget"]["target_patches"]
        self.opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                                     weight_decay=t["weight_decay"])
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.epochs)
        self.use_amp = t["amp"] and self.dev.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.clip = t["grad_clip"]
        self.ckpt_dir = t["ckpt_dir"]
        self.log = get_logger("train", os.path.join(t["log_dir"], "train.log"))

    # -------------------------------------------------------------- train --
    def fit(self, train_loader, val_loader):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        best = -1.0
        for ep in range(self.epochs):
            ph = B.phase_of(ep, self.epochs, self.p1, self.p2)
            bud = B.budget_of(ep, self.epochs, self.p1, self.p2, self.P, self.target)
            mode = B.mode_of(ep, self.epochs, self.p1, self.p2)
            tr = self._train_epoch(train_loader, ep, ph, bud, mode)
            va = self.evaluate(val_loader, budget=self.target, mode="learned")
            self.sched.step()
            self.log.info(
                f"ep {ep:02d} | phase {ph} B={bud} mode={mode} | "
                f"loss {tr['loss']:.4f} | val acc {va['accuracy']:.4f} "
                f"mf1 {va['macro_f1']:.4f} kappa {va['kappa']:.4f}"
            )
            if va["macro_f1"] > best:
                best = va["macro_f1"]
                self.save(os.path.join(self.ckpt_dir, "best.pt"), ep, best)
        self.save(os.path.join(self.ckpt_dir, "last.pt"), self.epochs - 1, best)
        self.log.info(f"done. best val macro-F1 {best:.4f}")
        return best

    def _train_epoch(self, loader, ep, phase, bud, mode):
        self.model.train()
        c, l = self.cfg["loss"], self.cfg["model"]
        running, n = 0.0, 0
        for batch in loader:
            patches = batch["patches"].to(self.dev, non_blocking=True)
            summary = batch["summary"].to(self.dev, non_blocking=True)
            label = batch["label"].to(self.dev, non_blocking=True)
            cap = batch["cap"].to(self.dev, non_blocking=True)
            has_cap = bool(batch["has_cap"][0]) if "has_cap" in batch else False

            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(patches, summary, bud, mode)
                loss = staging_loss(out["stage_logits"], label)
                if phase >= 2:
                    ratio = bud / self.P
                    loss = loss + c["beta_stability"] * stability_loss(
                        out["mask"], out["probs"], ratio, c["stability_entropy_w"]
                    )
                if phase >= 3:
                    if l["crossview"] and out["tf_emb"] is not None:
                        loss = loss + c["alpha_contrast"] * info_nce(
                            out["time_emb"], out["tf_emb"], c["contrast_temp"]
                        )
                    if has_cap and c["lambda_cap"] > 0:
                        loss = loss + c["lambda_cap"] * cap_loss(out["cap_logits"], cap)

            self.scaler.scale(loss).backward()
            if self.clip:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            running += loss.item() * label.size(0)
            n += label.size(0)
        return {"loss": running / max(1, n)}

    # ----------------------------------------------------------- evaluate --
    @torch.no_grad()
    def evaluate(self, loader, budget, mode="learned", with_calibration=False,
                 with_cap=False):
        self.model.eval()
        ys, ps, probs_all = [], [], []
        cap_t, cap_p = [], []
        for batch in loader:
            patches = batch["patches"].to(self.dev)
            summary = batch["summary"].to(self.dev)
            label = batch["label"]
            out = self.model(patches, summary, budget, mode)
            sm = torch.softmax(out["stage_logits"], dim=-1).cpu().numpy()
            ys.append(label.numpy())
            ps.append(sm.argmax(1))
            probs_all.append(sm)
            if with_cap and bool(batch["has_cap"][0]):
                ct = batch["cap"].numpy().reshape(-1)
                cp = torch.sigmoid(out["cap_logits"]).cpu().numpy().reshape(-1)
                keep = ct >= 0
                cap_t.append(ct[keep])
                cap_p.append(cp[keep])

        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        res = staging_metrics(y_true, y_pred)
        res["observed_pct"] = 100.0 * flops_proxy(budget, self.P)
        if with_calibration:
            ece, bc, ba = expected_calibration_error(y_true, np.concatenate(probs_all))
            res["ece"] = ece
            res["reliability"] = {"confidence": bc, "accuracy": ba}
        if with_cap and cap_t:
            res["cap"] = cap_metrics(np.concatenate(cap_t), np.concatenate(cap_p))
        return res

    # ----------------------------------------------------------- io ----
    def save(self, path, epoch, best):
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg,
                    "meta": self.meta, "epoch": epoch, "best": best}, path)

    @staticmethod
    def load_model(path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or _device())
        model = build_model(ckpt["cfg"], ckpt["meta"])
        model.load_state_dict(ckpt["model"])
        return model, ckpt["cfg"], ckpt["meta"]

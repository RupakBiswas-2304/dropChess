"""Gradient steps on the replay buffer."""

import math

import numpy as np
import torch
import torch.nn.functional as F

import dropchess

NUM_ACTIONS = dropchess.NUM_ACTIONS
_mt = dropchess.mirror_table()
MIRROR_PERM = torch.from_numpy(np.where(_mt >= 0, _mt, np.arange(NUM_ACTIONS))).long()
PLANE_SCALE = torch.from_numpy(dropchess.plane_scale()).view(1, -1, 1, 1)
_BITS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)


def unpack_mask(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 624] (np.packbits order) -> bool [B, 4992]."""
    bits = _BITS.to(packed.device)
    return (packed.unsqueeze(-1) & bits).ne(0).reshape(packed.shape[0], -1)


class Trainer:
    def __init__(self, model, cfg: dict, device: torch.device):
        self.model, self.cfg, self.device = model, cfg, device
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.amp = device.type == "cuda" and cfg.get("amp", True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.perm = MIRROR_PERM.to(device)
        self.scale = PLANE_SCALE.to(device)
        self.rng = np.random.default_rng()

    def batch(self, buf, n):
        planes, mask, pidx, pp, z, q = buf.sample(n, self.rng)
        a = self.cfg.get("value_q_mix", 0.0)
        if a:
            z = (1 - a) * z + a * q
        d = self.device
        x = torch.from_numpy(planes).to(d, non_blocking=True).float() * self.scale
        m = unpack_mask(torch.from_numpy(mask).to(d))
        target = torch.zeros(n, NUM_ACTIONS, device=d)
        target.scatter_add_(1, torch.from_numpy(pidx.astype(np.int64)).to(d), torch.from_numpy(pp.astype(np.float32)).to(d))
        z = torch.from_numpy(z).to(d)
        if self.cfg.get("mirror_augment", True):
            flip = torch.rand(n, device=d) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
            m = torch.where(flip.view(-1, 1), m[:, self.perm], m)
            target = torch.where(flip.view(-1, 1), target[:, self.perm], target)
        return x, m, target, z

    def steps(self, buf, n_steps: int) -> dict:
        cfg = self.cfg
        self.model.train()
        lr0, lr_min = cfg["lr"], cfg["lr"] * cfg.get("lr_min_frac", 0.1)
        tot_p = tot_v = 0.0
        for s in range(n_steps):
            lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * s / max(1, n_steps)))
            for g in self.opt.param_groups:
                g["lr"] = lr
            x, m, target, z = self.batch(buf, cfg["batch_size"])
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                logits, v = self.model(x)
            logits = logits.float().masked_fill(~m, -1e9)
            loss_p = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
            loss_v = F.mse_loss(v.float(), z)
            loss = loss_p + cfg.get("value_weight", 1.0) * loss_v
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            tot_p += loss_p.item()
            tot_v += loss_v.item()
        self.model.eval()
        n = max(1, n_steps)
        return {"loss_p": tot_p / n, "loss_v": tot_v / n}

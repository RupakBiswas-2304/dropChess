"""Residual policy/value network (AlphaZero-lite).

Input  planes [N, 31, 8, 8] float32 (dropchess Game.encode()).
Output policy logits [N, 4992] (unmasked) and value [N] in [-1, 1] for the side to move.

The policy head is convolutional: 78 output planes = 73 move types per
from-square + 5 drop pieces per to-square, rearranged into the action layout
(from_sq * 73 + type, then 4672 + piece * 64 + to_sq). This keeps the model
at ~1-2M parameters instead of ~10M with a dense 2048 -> 4992 layer, which
matters for the browser download.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PLANES = 31
NUM_ACTIONS = 4992


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, blocks: int = 6, channels: int = 64):
        super().__init__()
        self.blocks_n, self.channels = blocks, channels
        c = channels
        self.stem = nn.Sequential(nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(blocks)])
        self.pol = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, 78, 1))
        self.val = nn.Sequential(nn.Conv2d(c, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.val_fc = nn.Sequential(nn.Linear(4 * 64, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.body(self.stem(x))
        p = self.pol(h)  # [N, 78, 8, 8]
        n = p.shape[0]
        moves = p[:, :73].reshape(n, 73, 64).transpose(1, 2).reshape(n, 73 * 64)
        drops = p[:, 73:].reshape(n, 5 * 64)
        logits = torch.cat([moves, drops], dim=1)
        v = torch.tanh(self.val_fc(self.val(h).reshape(n, -1))).squeeze(1)
        return logits, v

    def config(self) -> dict:
        return {"blocks": self.blocks_n, "channels": self.channels}


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

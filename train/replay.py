"""Fixed-size ring buffer of training positions, stored compactly.

Per position: raw planes uint8 [31, 8, 8] (exact; scaled to floats on the GPU),
legal mask packed to 624 bytes, the top-K visit distribution, and the value
target. ~3.4 KB per position, so 500k positions is about 1.7 GB of RAM.
"""

import os
from pathlib import Path

import numpy as np

PLANE_SHAPE = (31, 8, 8)
MASK_BYTES = 624


class ReplayBuffer:
    def __init__(self, capacity: int, topk: int):
        self.capacity, self.topk = capacity, topk
        self.planes = np.zeros((capacity, *PLANE_SHAPE), np.uint8)
        self.mask = np.zeros((capacity, MASK_BYTES), np.uint8)
        self.pol_idx = np.zeros((capacity, topk), np.int16)
        self.pol_p = np.zeros((capacity, topk), np.float16)
        self.z = np.zeros(capacity, np.float32)
        self.q = np.zeros(capacity, np.float32)  # root search value
        self.size = 0
        self.head = 0
        self.total_added = 0

    def __len__(self):
        return self.size

    def add_game(self, g: dict, adjudicate: bool) -> int:
        n = g["n"]
        if n == 0:
            return 0
        z = g["z"].copy()
        if g["status"] == "maxplies" and not adjudicate:
            z[:] = 0.0
        q = g["q"].astype(np.float32) if "q" in g else z
        arrays = (
            g["planes"].reshape(n, *PLANE_SHAPE),
            g["mask"].reshape(n, MASK_BYTES),
            g["pol_idx"].reshape(n, self.topk).astype(np.int16),
            g["pol_p"].reshape(n, self.topk).astype(np.float16),
            z,
            q,
        )
        idx = (self.head + np.arange(n)) % self.capacity
        for dst, src in zip((self.planes, self.mask, self.pol_idx, self.pol_p, self.z, self.q), arrays):
            dst[idx] = src
        self.head = (self.head + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.total_added += n
        return n

    def sample(self, batch: int, rng: np.random.Generator):
        i = rng.integers(0, self.size, batch)
        return self.planes[i], self.mask[i], self.pol_idx[i], self.pol_p[i], self.z[i], self.q[i]

    def save(self, path: Path):
        tmp = Path(str(path) + ".tmp.npz")
        s = self.size
        np.savez(
            tmp,
            planes=self.planes[:s], mask=self.mask[:s], pol_idx=self.pol_idx[:s], pol_p=self.pol_p[:s], z=self.z[:s], q=self.q[:s],
            meta=np.array([self.head, self.total_added, self.capacity, self.topk], np.int64),
        )
        os.replace(tmp, path)

    def load(self, path: Path) -> bool:
        if not Path(path).exists():
            return False
        d = np.load(path)
        head, total, cap, topk = (int(x) for x in d["meta"])
        if topk != self.topk:
            print(f"replay buffer on disk has topk={topk}, config says {self.topk}: starting with an empty buffer")
            return False
        s = min(len(d["z"]), self.capacity)
        # Keep the most recent `s` positions if the capacity shrank.
        order = np.arange(len(d["z"]))
        if cap == len(d["z"]):  # buffer was full: oldest entry sits at head
            order = np.roll(order, -head)
        order = order[-s:]
        self.planes[:s] = d["planes"][order]
        self.mask[:s] = d["mask"][order]
        self.pol_idx[:s] = d["pol_idx"][order]
        self.pol_p[:s] = d["pol_p"][order]
        self.z[:s] = d["z"][order]
        self.q[:s] = (d["q"] if "q" in d else d["z"])[order]  # older buffers have no q
        self.size, self.head, self.total_added = s, s % self.capacity, total
        return True

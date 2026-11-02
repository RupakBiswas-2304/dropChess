"""Drives a dropchess.Runner: batched network evaluation for MCTS leaves."""

import numpy as np
import torch

import dropchess

PLANE_SCALE = torch.from_numpy(dropchess.plane_scale()).view(1, -1, 1, 1)


class Evaluator:
    """Runs a network on raw uint8 planes coming from the Runner."""

    def __init__(self, model: torch.nn.Module, device: torch.device, half: bool):
        self.model, self.device = model, device
        self.half = half and device.type == "cuda"
        self.scale = PLANE_SCALE.to(device)

    @torch.inference_mode()
    def __call__(self, planes_u8: np.ndarray):
        x = torch.from_numpy(planes_u8).to(self.device, non_blocking=True).float() * self.scale
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
            logits, v = self.model(x)
        return logits.float().cpu().numpy(), v.float().cpu().numpy()


def drive_step(runner, evaluators: dict) -> int:
    """One step: advance all games, evaluate their leaves, feed back. Returns batch size."""
    planes, nets = runner.step()
    b = len(nets)
    if b == 0:
        return 0
    if len(evaluators) == 1 or (nets == nets[0]).all():
        logits, values = evaluators[int(nets[0])](planes)
    else:
        logits = np.empty((b, dropchess.NUM_ACTIONS), np.float32)
        values = np.empty(b, np.float32)
        for nid in np.unique(nets):
            sel = nets == nid
            logits[sel], values[sel] = evaluators[int(nid)](planes[sel])
    runner.feed(np.ascontiguousarray(logits), np.ascontiguousarray(values))
    return b


def run_to_completion(runner, evaluators: dict) -> list:
    games = []
    while not runner.done():
        drive_step(runner, evaluators)
        games.extend(runner.take_finished())
    games.extend(runner.take_finished())
    return games


def runner_config(cfg: dict, **overrides) -> dict:
    keys = ["sims", "sims_fast", "p_full", "c_puct", "fpu_reduction", "fpu_root", "dirichlet_alpha", "dirichlet_frac",
            "temp_plies", "max_plies", "topk"]
    out = {k: cfg[k] for k in keys if k in cfg}
    out.update(overrides)
    return out

"""Matches between a network (MCTS) and fixed opponents.

As a script:
    python evaluate.py --ckpt checkpoints/gen_020.pt --vs greedy --games 100
    python evaluate.py --ckpt checkpoints/gen_020.pt --vs checkpoints/gen_010.pt
"""

import argparse
import csv
from pathlib import Path

import torch

import dropchess
from selfplay import Evaluator, run_to_completion, runner_config

VARIANTS = ("drop", "crazy")


def match(evaluators: dict, opponent: str, games: int, cfg: dict, variants=VARIANTS, seed: int = 0) -> dict:
    """net0 vs `opponent` ("random", "greedy" or "net1"), half the games as White.
    Returns {variant: (wins, draws, losses)} from net0's side."""
    rc = runner_config(cfg, sims=cfg["eval_sims"], sims_fast=cfg["eval_sims"], p_full=1.0,
                       temp_plies=cfg.get("eval_temp_plies", 8), noise=False, record=False, adjudicate=False)
    runner = dropchess.Runner(rc, parallel=cfg.get("eval_parallel", 256), seed=seed)
    for v in variants:
        for i in range(games):
            if i % 2 == 0:
                runner.add_game(v, "net0", opponent)
            else:
                runner.add_game(v, opponent, "net0")
    out = {v: [0, 0, 0] for v in variants}
    for g in run_to_completion(runner, evaluators):
        me_white = g["white"] == "net0"
        r = g["result"] if me_white else -g["result"]
        out[g["variant"]][0 if r > 0 else 1 if r == 0 else 2] += 1
    return {v: tuple(x) for v, x in out.items()}


def append_results(path: Path, gen: int, opponent: str, res: dict):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["generation", "variant", "opponent", "W", "D", "L"])
        for v, (wi, d, l) in res.items():
            w.writerow([gen, v, opponent, wi, d, l])


def fmt(res: dict) -> str:
    return "  ".join(f"{v} {w}-{d}-{l}" for v, (w, d, l) in res.items())


def main():
    from run import load_model, pick_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vs", default="greedy", help="random | greedy | path to another checkpoint")
    ap.add_argument("--games", type=int, default=100, help="games per variant")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    model, _ = load_model(args.ckpt, device)
    evs = {0: Evaluator(model, device, half=True)}
    opp = args.vs
    if opp not in ("random", "greedy"):
        other, _ = load_model(opp, device)
        evs[1] = Evaluator(other, device, half=True)
        opp = "net1"
    cfg = {"eval_sims": args.sims, "eval_temp_plies": 8, "max_plies": 300}
    res = match(evs, opp, args.games, cfg)
    print(f"{Path(args.ckpt).name} vs {args.vs}: {fmt(res)}  (W-D-L)")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

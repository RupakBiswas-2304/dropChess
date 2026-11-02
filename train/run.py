"""Self-play training loop. One entry point, resumable:

    python run.py --config configs/gpu.yaml

Each generation: play `games_per_gen` self-play games with the current
network, add them to the replay buffer, train, save checkpoints/gen_NNN.pt.
Every `eval_every` generations: matches against random, greedy and the
checkpoint `eval_every` generations older, appended to eval/results.csv.
Ctrl-C at any time; re-running the same command resumes from the latest
checkpoint (games in progress are lost, everything else is kept).
"""

import argparse
import csv
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import dropchess
from evaluate import append_results, fmt, match
from model import Net, param_count
from replay import ReplayBuffer
from selfplay import Evaluator, drive_step, runner_config
from train import Trainer

HERE = Path(__file__).parent


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def latest_checkpoint(d: Path):
    cks = sorted(Path(d).glob("gen_*.pt"))
    return cks[-1] if cks else None


def ckpt_path(d: Path, gen: int) -> Path:
    return Path(d) / f"gen_{gen:03d}.pt"


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Net(**ck["net"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def save_checkpoint(path: Path, gen: int, model, trainer, cfg):
    tmp = Path(str(path) + ".tmp")
    torch.save({"gen": gen, "net": model.config(), "model": model.state_dict(),
                "opt": trainer.opt.state_dict(), "scaler": trainer.scaler.state_dict(), "cfg": cfg}, tmp)
    tmp.replace(path)


def log_row(path: Path, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def material_blend(cfg: dict, gen: int) -> float:
    """Material score weight in self-play search: full for the first gens, then fades to 0."""
    b0 = cfg.get("material_blend", 0.0)
    full, zero = cfg.get("blend_full_gens", 0), cfg.get("blend_zero_gen", 0)
    if b0 <= 0 or gen > zero:
        return 0.0
    if gen <= full:
        return b0
    return b0 * (zero - gen) / max(1, zero - full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--generations", type=int, default=None, help="stop after this generation (default: config)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dir", default=None, help="checkpoint directory (default: config 'dir' or checkpoints/)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(args.device)
    ck_dir = Path(args.dir or cfg.get("dir", HERE / "checkpoints"))
    ck_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = Path(cfg.get("eval_csv", HERE / "eval" / "results.csv"))
    max_gen = args.generations or cfg.get("generations", 1000)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- model, optimiser, buffer (resume if possible) ----
    buf = ReplayBuffer(cfg["buffer_size"], cfg["topk"])
    last = latest_checkpoint(ck_dir)
    if last is not None:
        model, ck = load_model(last, device)
        trainer = Trainer(model, cfg, device)
        trainer.opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            trainer.scaler.load_state_dict(ck["scaler"])
        gen = ck["gen"]
        loaded = buf.load(ck_dir / "replay.npz")
        print(f"resumed from {last.name} (gen {gen}); replay buffer {'%d positions' % len(buf) if loaded else 'empty'}")
    else:
        model = Net(cfg["blocks"], cfg["channels"]).to(device)
        trainer = Trainer(model, cfg, device)
        gen = 0
        save_checkpoint(ckpt_path(ck_dir, 0), 0, model, trainer, cfg)
        print(f"new network: {cfg['blocks']} blocks x {cfg['channels']} channels, {param_count(model) / 1e6:.2f}M parameters")
    model.eval()
    print(f"device {device}; {cfg['games_per_gen']} games/gen, {cfg['parallel']} in parallel, {cfg['sims']} sims "
          f"(fast {cfg.get('sims_fast', cfg['sims'])}, p_full {cfg.get('p_full', 1.0)})")

    evaluator = Evaluator(model, device, half=cfg.get("amp", True))
    rc = runner_config(cfg, noise=True, record=True, adjudicate=gen < cfg["adjudicate_gens"])
    seed = random.randrange(2**32)
    runner = dropchess.Runner(rc, parallel=cfg["parallel"], seed=seed)
    variants = cfg.get("variants", ["drop", "crazy"])
    vi = 0

    def top_up():
        nonlocal vi
        while runner.active() + runner.queued() < cfg["parallel"] + max(8, cfg["parallel"] // 8):
            runner.add_game(variants[vi % len(variants)], "net0", "net0")
            vi += 1

    while gen < max_gen:
        gen += 1
        adjudicate = gen <= cfg["adjudicate_gens"]
        runner.set_adjudicate(adjudicate)
        blend = material_blend(cfg, gen)
        runner.set_material_blend(blend)
        t0 = time.time()
        games, new_pos = [], 0
        while len(games) < cfg["games_per_gen"]:
            top_up()
            drive_step(runner, {0: evaluator})
            for g in runner.take_finished():
                games.append(g)
                new_pos += buf.add_game(g, adjudicate)
        t_play = time.time() - t0

        # ---- train ----
        t1 = time.time()
        steps = 0
        losses = {"loss_p": float("nan"), "loss_v": float("nan")}
        if len(buf) >= cfg["min_buffer"]:
            steps = int(np.clip(round(new_pos * cfg["sample_reuse"] / cfg["batch_size"]), cfg["min_steps"], cfg["max_steps"]))
            losses = trainer.steps(buf, steps)
        t_train = time.time() - t1
        save_checkpoint(ckpt_path(ck_dir, gen), gen, model, trainer, cfg)
        if gen % cfg.get("save_buffer_every", 1) == 0:
            buf.save(ck_dir / "replay.npz")

        # ---- log ----
        decisive = sum(g["status"] == "checkmate" for g in games) / len(games)
        capped = sum(g["status"] == "maxplies" for g in games) / len(games)
        avg_len = float(np.mean([g["plies"] for g in games]))
        by_v = {v: [g for g in games if g["variant"] == v] for v in variants}
        white_wins = sum(g["result"] == 1 for g in games) / len(games)
        row = {
            "gen": gen, "games": len(games), "positions": new_pos, "buffer": len(buf),
            "decisive": round(decisive, 3), "capped": round(capped, 3), "white_wins": round(white_wins, 3),
            "avg_plies": round(avg_len, 1),
            **{f"avg_plies_{v}": round(float(np.mean([g["plies"] for g in gs])), 1) if gs else "" for v, gs in by_v.items()},
            "steps": steps, "loss_p": round(losses["loss_p"], 4), "loss_v": round(losses["loss_v"], 4),
            "play_s": round(t_play), "train_s": round(t_train), "adjudicate": int(adjudicate), "blend": round(blend, 3),
        }
        log_row(ck_dir / "log.csv", row)
        print(f"gen {gen:3d} | {len(games)} games, {new_pos} pos, buffer {len(buf)} | decisive {decisive:.0%} "
              f"capped {capped:.0%} avg {avg_len:.0f} plies | {steps} steps loss p {losses['loss_p']:.3f} "
              f"v {losses['loss_v']:.3f} | play {t_play:.0f}s train {t_train:.0f}s"
              + (f" | blend {blend:.2f}" if blend else "") + (" | adjudication on" if adjudicate else ""), flush=True)

        # ---- evaluate ----
        every = cfg.get("eval_every", 5)
        if every and gen % every == 0:
            t2 = time.time()
            evs = {0: Evaluator(model, device, half=cfg.get("amp", True))}
            n = cfg["eval_games"]
            for opp in ("random", "greedy"):
                res = match(evs, opp, n, cfg, variants, seed=gen)
                append_results(eval_csv, gen, opp, res)
                print(f"   eval vs {opp:7s} {fmt(res)}  (W-D-L)", flush=True)
            old = ckpt_path(ck_dir, gen - every)
            if old.exists():
                old_model, _ = load_model(old, device)
                evs[1] = Evaluator(old_model, device, half=cfg.get("amp", True))
                res = match(evs, "net1", n, cfg, variants, seed=gen + 1)
                append_results(eval_csv, gen, f"gen_{gen - every:03d}", res)
                print(f"   eval vs gen {gen - every:3d} {fmt(res)}  (W-D-L)", flush=True)
                del old_model
            print(f"   eval took {time.time() - t2:.0f}s", flush=True)

        keep = cfg.get("keep_checkpoints", 0)
        if keep:
            # Keep every eval_every-th checkpoint (eval needs them) and the last `keep`.
            for p in sorted(ck_dir.glob("gen_*.pt"))[:-keep]:
                g = int(re.findall(r"\d+", p.stem)[0])
                if g % every:
                    p.unlink()

    print("done")


if __name__ == "__main__":
    main()

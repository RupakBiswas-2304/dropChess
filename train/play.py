"""Play against a checkpoint in the terminal (for playtesting before the browser bot exists).

    python play.py                                   # latest checkpoint, Drop, you play White
    python play.py --variant crazy --color black --sims 400
    python play.py --ckpt checkpoints/gen_040.pt

Type moves in UCI: e2e4, e7e8q, N@f3. Commands: moves (list legal), quit.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import dropchess
from run import latest_checkpoint, load_model, pick_device
from selfplay import Evaluator, run_to_completion

GLYPH = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
         "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}


def show(fen: str, flip: bool):
    place = fen.split()[0]
    board, res = place.split("[")
    rows = []
    for row in board.split("/"):
        cells = []
        for c in row:
            if c == "~":
                continue
            cells += ["·"] * int(c) if c.isdigit() else [GLYPH[c]]
        rows.append(cells)
    ranks = list(range(8, 0, -1))
    if flip:
        rows = [r[::-1] for r in rows[::-1]]
        ranks = ranks[::-1]
    for rank, cells in zip(ranks, rows):
        print(f" {rank}  " + " ".join(cells))
    files = "abcdefgh"[::-1] if flip else "abcdefgh"
    print("    " + " ".join(files))
    res = res.rstrip("]")
    print(f" reserve  White: {''.join(c for c in res if c.isupper()) or '-'}   Black: {''.join(c for c in res if c.islower()) or '-'}")


def bot_move(evs, game, sims: int) -> str:
    r = dropchess.Runner({"sims": sims, "temp_plies": 0}, parallel=1)
    r.add_game(game.variant, "net0", "net0", fen=game.fen(), max_moves=1)
    return run_to_completion(r, evs)[0]["moves"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--variant", default="drop", choices=["drop", "crazy"])
    ap.add_argument("--color", default="white", choices=["white", "black"])
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    ckpt = args.ckpt or latest_checkpoint(Path(__file__).parent / "checkpoints")
    if ckpt is None:
        raise SystemExit("no checkpoint yet; start training with run.py first")
    device = pick_device(args.device)
    model, meta = load_model(ckpt, device)
    evs = {0: Evaluator(model, device, half=True)}
    print(f"bot: {Path(ckpt).name} (gen {meta['gen']}), {args.sims} simulations per move")
    g = dropchess.Game.new(args.variant)
    me = 0 if args.color == "white" else 1
    while not g.is_terminal():
        show(g.fen(), flip=me == 1)
        if g.to_play() == me:
            uci = input("your move: ").strip()
            if uci == "quit":
                return
            if uci == "moves":
                print(" ".join(g.legal_uci()))
                continue
            if uci not in g.legal_uci():
                print("not legal here (type 'moves' to list them)")
                continue
            g.apply_uci(uci)
        else:
            m = bot_move(evs, g, args.sims)
            print(f"bot plays {m}")
            g.apply_uci(m)
    show(g.fen(), flip=me == 1)
    res = g.result()
    print(f"game over: {g.status()}", "- you win" if res == (1 if me == 0 else -1) else "- bot wins" if res else "- draw")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    np.set_printoptions(suppress=True)
    main()

"""Writes fixtures/perft.json and fixtures/encoding.jsonl with the Python reference engine.

The Rust engine (engine/tests, train/tests) and the web client are tested
against these files. Re-run only if the rules or the encoding change on purpose:

    python make_fixtures.py
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from dropchess_py import Game, START_FEN

HERE = Path(__file__).parent
FIX = HERE / "fixtures"

# Hand-made positions: mid-game, check, en passant, promotion, promoted pieces.
POSITIONS = [
    START_FEN,
    "r3k3/pp6/2n5/8/3Q4/2N2B2/PP6/4K3[RBNPPPPPPrbbnpppppp] w - - 0 12",
    "4k3/8/8/8/8/8/3q4/4K3[QRN] w - - 0 1",
    "4k3/8/8/8/3pP3/8/8/4K3[Pp] b - e3 0 1",
    "4k3/1P6/8/8/8/8/6p1/4K3[NNnn] w - - 0 1",
    "r1b1k2r/ppp2ppp/2n5/3q4/8/2N2N2/PPP2PPP/R1BQK2R[BPbp] w - - 0 10",
    "4k3/8/8/8/8/8/8/q~3K3[Rr] w - - 0 1",
]


def perft(g: Game, depth: int) -> int:
    moves = g.legal_moves()
    if depth == 1:
        return len(moves)
    total = 0
    for m in moves:
        c = g.clone()
        c.push(m)
        total += perft(c, depth - 1)
    return total


def planes_hash(g: Game) -> str:
    return hashlib.sha256(np.ascontiguousarray(g.encode(), dtype=np.float32).tobytes()).hexdigest()


def line(g: Game, moves: list[str], fen0: str) -> dict:
    return {
        "start": fen0,
        "variant": g.variant,
        "moves": moves,
        "fen": g.fen(),
        "status": g.status(),
        "legal": sorted(m.uci() for m in g.legal_moves()),
        "actions": g.legal_actions(),
        "planes_sha256": planes_hash(g),
    }


def main():
    FIX.mkdir(exist_ok=True)

    perfts = []
    for fen in POSITIONS:
        for variant in ("drop", "crazy"):
            g = Game.from_fen(fen, variant)
            depths = {d: perft(g, d) for d in (1, 2, 3)}
            perfts.append({"fen": fen, "variant": variant, "perft": depths})
            print(variant, fen, depths)
    (FIX / "perft.json").write_text(json.dumps(perfts, indent=1) + "\n")

    rng = random.Random(2304)
    out = []
    for fen in POSITIONS:
        for variant in ("drop", "crazy"):
            out.append(line(Game.from_fen(fen, variant), [], fen))
    for i in range(60):
        variant = ("drop", "crazy")[i % 2]
        g = Game.new(variant)
        moves = []
        stop = rng.randint(4, 300)
        while not g.is_terminal() and len(moves) < stop:
            ms = g.legal_moves()
            # Bias towards board moves so games get captures and promotions, not just drops.
            board = [m for m in ms if not m.drop]
            m = rng.choice(board if board and rng.random() < 0.7 else ms)
            g.push(m)
            moves.append(m.uci())
            if len(moves) % 15 == 0 or g.is_terminal():
                out.append(line(g, list(moves), START_FEN))
    with open(FIX / "encoding.jsonl", "w") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(perfts)} perft entries, {len(out)} encoding lines")


if __name__ == "__main__":
    main()

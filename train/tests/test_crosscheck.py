"""The Rust engine must agree with the fixtures (written by the Python reference)
and with the Python engine move by move on random games."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from engine import game_class
from dropchess_py.encoding import mirror_table as py_mirror_table, PLANE_SCALE

FIX = Path(__file__).resolve().parents[1] / "fixtures"
RustGame = game_class("rust")
PyGame = game_class("py")


def test_perft_fixtures():
    import dropchess

    for e in json.loads((FIX / "perft.json").read_text()):
        for d, n in e["perft"].items():
            assert dropchess.perft(e["fen"], e["variant"], int(d)) == n, (e["fen"], e["variant"], d)


def test_encoding_fixtures():
    lines = [json.loads(l) for l in (FIX / "encoding.jsonl").read_text().splitlines()]
    assert len(lines) > 100
    for d in lines:
        g = RustGame.from_fen(d["start"], d["variant"])
        for m in d["moves"]:
            g.apply_uci(m)
        assert g.fen() == d["fen"]
        assert g.status() == d["status"], d["fen"]
        assert g.legal_uci() == d["legal"], d["fen"]
        assert g.legal_actions() == d["actions"], d["fen"]
        h = hashlib.sha256(np.ascontiguousarray(g.encode(), dtype=np.float32).tobytes()).hexdigest()
        assert h == d["planes_sha256"], d["fen"]


def test_mirror_table_and_scale_match():
    import dropchess

    assert (dropchess.mirror_table() == py_mirror_table()).all()
    assert np.allclose(dropchess.plane_scale(), PLANE_SCALE)


@pytest.mark.parametrize("variant", ["drop", "crazy"])
def test_random_games_agree(variant):
    rng = random.Random(7 if variant == "drop" else 8)
    for _ in range(12):
        r, p = RustGame.new(variant), PyGame.new(variant)
        for _ in range(300):
            assert r.fen() == p.fen()
            assert r.status() == p.status()
            if r.is_terminal():
                assert r.result() == p.result()
                break
            assert r.legal_actions() == p.legal_actions()
            assert (r.encode_raw() == p.encode_raw()).all()
            a = rng.choice(r.legal_actions())
            assert r.action_to_uci(a) == p.action_to_uci(a)
            r.apply(a)
            p.apply(a)


def test_mirror_maps_legal_to_legal():
    """With no castling the rules are left-right symmetric: mirroring a position
    mirrors its legal action set."""
    import dropchess

    table = dropchess.mirror_table()

    def mirror_fen(fen):
        place, rest = fen.split(" ", 1)
        board, res = place.split("[")
        rows = []
        for row in board.split("/"):
            cells = []
            for i, c in enumerate(row):
                if c == "~":
                    cells[-1] += "~"
                elif c.isdigit():
                    cells += ["1"] * int(c)
                else:
                    cells.append(c)
            rows.append("".join(reversed(cells)))
        parts = rest.split()
        if parts[2] != "-":
            parts[2] = "abcdefgh"[7 - "abcdefgh".index(parts[2][0])] + parts[2][1]
        return "/".join(rows) + "[" + res + " " + " ".join(parts)

    for d in (json.loads(l) for l in (FIX / "encoding.jsonl").read_text().splitlines()):
        g = RustGame.from_fen(d["fen"], d["variant"])
        m = RustGame.from_fen(mirror_fen(d["fen"]), d["variant"])
        assert sorted(table[a] for a in g.legal_actions()) == m.legal_actions(), d["fen"]
        assert (g.encode_raw()[:, :, ::-1][:24] == m.encode_raw()[:24]).all()

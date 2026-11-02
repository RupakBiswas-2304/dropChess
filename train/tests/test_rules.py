"""Rule tests from the training brief, run against both engines."""

import numpy as np

START = "4k3/8/8/8/8/8/8/4K3[QRRBBNNPPPPPPPPqrrbbnnpppppppp] w - - 0 1"


def test_perft1_start_is_153(Game):
    for v in ("drop", "crazy"):
        assert len(Game.new(v).legal_actions()) == 153


def test_no_pawn_drop_on_own_back_rank_or_enemy_half(Game):
    g = Game.new("drop")
    legal = set(g.legal_uci())
    assert "P@a1" not in legal and "P@a2" in legal
    assert "N@a1" in legal and "Q@h4" in legal and "Q@h5" not in legal
    g.apply_uci("N@c3")
    legal = set(g.legal_uci())
    assert "P@a8" not in legal and "P@a7" in legal
    assert "Q@d5" in legal and "Q@d4" not in legal


def test_drop_can_block_check(Game):
    g = Game.from_fen("4r2k/8/8/8/8/8/8/4K3[Q] w - - 0 1", "drop")
    legal = set(g.legal_uci())
    assert {"Q@e2", "Q@e4"} <= legal
    assert "Q@a2" not in legal
    assert "e1e2" not in legal


def test_drop_mate_is_legal(Game):
    g = Game.from_fen("8/8/8/8/8/8/2K5/k7[R] w - - 0 1", "drop")
    g.apply_uci("R@a3")
    assert g.is_terminal() and g.result() == 1 and g.status() == "checkmate"


def test_capture_goes_to_reserve_only_in_crazy(Game):
    fen = "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1"
    crazy = Game.from_fen(fen, "crazy")
    crazy.apply_uci("e1d2")
    assert crazy.fen().startswith("4k3/8/8/8/8/8/3K4/8[P]")
    drop = Game.from_fen(fen, "drop")
    drop.apply_uci("e1d2")
    assert drop.fen().startswith("4k3/8/8/8/8/8/3K4/8[]")


def test_captured_promoted_piece_returns_as_pawn(Game):
    g = Game.from_fen("4k3/8/8/8/8/8/8/3q~K3 w - - 0 1", "crazy")
    g.apply_uci("e1d1")
    assert "[P]" in g.fen()


def test_dropped_pawn_double_step_and_en_passant(Game):
    g = Game.from_fen("4k3/8/8/8/3p4/8/8/4K3[P] w - - 0 1", "drop")
    for m in ("P@e2", "e8d8", "e2e4"):
        g.apply_uci(m)
    assert "d4e3" in g.legal_uci()
    g.apply_uci("d4e3")
    assert g.fen().startswith("3k4/8/8/8/8/4p3/8/4K3")


def test_drop_resets_fifty_move_clock(Game):
    g = Game.from_fen("4k3/8/8/8/8/8/8/4K3[N] w - - 40 30", "drop")
    g.apply_uci("N@c3")
    assert g.fen().split()[4] == "0"


def test_fifty_move_rule(Game):
    g = Game.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 80", "drop")
    g.apply_uci("a1a2")
    assert g.status() == "fifty"


def test_kings_only_is_insufficient(Game):
    assert Game.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "drop").status() == "insufficient"
    assert Game.new("drop").status() == "ongoing"


def test_threefold_repetition(Game):
    g = Game.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "drop")
    for _ in range(2):
        for m in ("e1f1", "e8f8", "f1e1", "f8e8"):
            g.apply_uci(m)
    assert g.status() == "repetition"


def test_repetition_planes(Game):
    g = Game.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "drop")
    assert g.encode()[24].sum() == 0
    for m in ("e1f1", "e8f8", "f1e1", "f8e8"):
        g.apply_uci(m)
    x = g.encode()
    assert x[24].min() == 1 and x[25].sum() == 0


def test_encoding_is_from_side_to_move(Game):
    g = Game.new("drop")
    g.apply_uci("Q@d4")
    x = g.encode()  # Black to move, board mirrored: White's queen d4 -> d5 = square 35
    assert x[10].reshape(-1)[35] == 1  # opponent queen plane
    assert x[5].reshape(-1)[4] == 1  # my king e8 -> e1
    assert x[27].sum() == 0  # Black to move
    assert np.isclose(x[16, 0, 0], 1 / 8)  # my (Black's) queen in reserve
    assert x[21, 0, 0] == 0  # White's queen left the reserve


def test_action_roundtrip(Game):
    g = Game.from_fen("4k3/1P6/8/8/8/8/6p1/4K3[NNnn] w - - 0 1", "drop")
    for a in g.legal_actions():
        assert g.uci_to_action(g.action_to_uci(a)) == a
    ucis = g.legal_uci()
    assert {"b7b8q", "b7b8n", "b7b8r", "b7b8b"} <= set(ucis)
    assert len(set(g.legal_actions())) == len(ucis)


def test_max_plies_cap(Game):
    g = Game.new("drop")
    g.max_plies = 2
    g.apply_uci("N@c3")
    g.apply_uci("N@c6")
    assert g.status() == "maxplies" and g.result() == 0

"""Batched MCTS / Runner behaviour, using fake networks."""

import numpy as np

import dropchess
from replay import ReplayBuffer


def uniform(planes):
    b = planes.shape[0]
    return np.zeros((b, dropchess.NUM_ACTIONS), np.float32), np.zeros(b, np.float32)


def drive(runner, net=uniform):
    games = []
    while not runner.done():
        planes, nets = runner.step()
        if len(nets):
            runner.feed(*net(planes))
        games += runner.take_finished()
    return games + runner.take_finished()


def test_mcts_finds_drop_mate_with_uninformed_net():
    r = dropchess.Runner({"sims": 400}, parallel=4, seed=1)
    for _ in range(4):
        r.add_game("drop", "net0", "random", fen="8/8/8/8/8/8/2K5/k7[R] w - - 0 1")
    games = drive(r)
    assert len(games) == 4
    for g in games:
        assert g["status"] == "checkmate" and g["result"] == 1 and g["plies"] == 1, g["moves"]
        assert g["moves"][0] in ("R@a3", "R@a4")


def test_greedy_beats_random():
    r = dropchess.Runner({"max_plies": 300}, parallel=64, seed=5)
    for i in range(64):
        r.add_game("drop" if i % 2 else "crazy", "greedy", "random") if i % 4 < 2 else r.add_game(
            "drop" if i % 2 else "crazy", "random", "greedy")
    games = drive(r)
    greedy_score = sum((g["result"] if g["white"] == "greedy" else -g["result"]) for g in games)
    assert greedy_score > 20, greedy_score


def test_selfplay_records_and_targets():
    topk = 32
    r = dropchess.Runner({"sims": 24, "record": True, "noise": True, "temp_plies": 30, "max_plies": 60,
                          "topk": topk, "adjudicate": True}, parallel=8, seed=7)
    for i in range(8):
        r.add_game(("drop", "crazy")[i % 2], "net0", "net0")
    games = drive(r)
    assert len(games) == 8
    for g in games:
        n = g["n"]
        assert n == g["plies"] or n < g["plies"]  # forced single-move positions aren't recorded
        assert g["planes"].shape == (n * 31 * 64,) and g["mask"].shape == (n * 624,)
        p = g["pol_p"].reshape(n, topk)
        assert np.allclose(p.sum(1), 1, atol=1e-5)
        mask = np.unpackbits(g["mask"].reshape(n, 624), axis=1).astype(bool)
        idx = g["pol_idx"].reshape(n, topk)
        for i in range(n):
            assert mask[i, idx[i][p[i] > 0]].all()
        if g["status"] == "maxplies":
            assert np.all(np.abs(g["z"]) <= 0.5)
        buf = ReplayBuffer(1000, topk)
        buf.add_game(g, adjudicate=False)
        if g["status"] == "maxplies":
            assert (buf.z[: buf.size] == 0).all()


def test_replay_save_load(tmp_path):
    topk = 8
    buf = ReplayBuffer(10, topk)
    rng = np.random.default_rng(0)
    for k in range(3):
        n = 4
        g = {"n": n, "status": "checkmate", "planes": rng.integers(0, 2, n * 1984, dtype=np.uint8),
             "mask": rng.integers(0, 255, n * 624, dtype=np.uint8), "pol_idx": np.arange(n * topk),
             "pol_p": np.full(n * topk, 1 / topk, np.float32), "z": np.full(n, float(k), np.float32)}
        buf.add_game(g, adjudicate=False)
    assert len(buf) == 10
    buf.save(tmp_path / "r.npz")
    b2 = ReplayBuffer(10, topk)
    assert b2.load(tmp_path / "r.npz")
    assert sorted(b2.z.tolist()) == sorted(buf.z.tolist())
    b3 = ReplayBuffer(6, topk)  # smaller: keeps the newest
    assert b3.load(tmp_path / "r.npz")
    assert sorted(b3.z.tolist()) == [1, 1, 2, 2, 2, 2]


def test_model_policy_layout_matches_actions():
    """The conv policy head must put each action's logit at the right index."""
    import torch
    from model import Net

    m = Net(1, 8).eval()
    with torch.no_grad():
        for mod in m.pol:
            if isinstance(mod, torch.nn.Conv2d) and mod.out_channels == 78:
                mod.weight.zero_()
                mod.bias.copy_(torch.arange(78, dtype=torch.float32) * 1000)
        logits, _ = m(torch.zeros(1, 31, 8, 8))
    # board action a = sq*73 + type -> channel type; drop 4672 + p*64 + sq -> channel 73 + p
    a = 10 * 73 + 56
    assert logits[0, a].item() == 56 * 1000
    assert logits[0, 4672 + 3 * 64 + 9].item() == 76 * 1000

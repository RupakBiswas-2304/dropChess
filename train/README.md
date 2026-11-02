# Drop Chess bot training

AlphaZero-lite self-play: a small residual network plus MCTS. One network plays both variants.

The rules, state encoding and batched MCTS run in Rust (`../pyengine`, a PyO3 module called `dropchess` built on `../engine`). PyTorch only runs the network. `dropchess_py/` is a separate pure-Python reference engine on python-chess. It's too slow to train with, but the tests hold the Rust engine to it so the rules can't drift.

## Setup (once)

```bash
cd ~/Desktop/Games/drop-chess/train
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
maturin develop --release -m ../pyengine/Cargo.toml
pytest -q                     # 41 tests: rules, Rust vs Python cross-check, MCTS, buffer
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True` and your GPU name. If it prints `False`, the wrong torch wheel was installed; reinstall from the selector at pytorch.org.

After any change under `engine/` or `pyengine/`, run `maturin develop --release -m ../pyengine/Cargo.toml` again.

## Train

```bash
python run.py --config configs/smoke.yaml     # 2-minute check that everything runs (writes checkpoints_smoke/)
python run.py --config configs/gpu.yaml       # the real run; Ctrl-C any time, same command resumes
```

To train unattended (survives closing the terminal, keeps the laptop awake):

```bash
setsid nohup systemd-inhibit --what=idle:sleep --why="drop chess training" \
  .venv/bin/python run.py --config configs/gpu.yaml > train.log 2>&1 < /dev/null &
tail -f train.log        # Ctrl-C stops watching, not training. Stop training: pkill -f run.py
```

`run.py` resumes from whatever is in `checkpoints/`, including a run trained with another config (the network size comes from the checkpoint).

Each generation prints one line:

```
gen  12 | 256 games, 9800 pos, buffer 120000 | decisive 41% capped 22% avg 163 plies | 150 steps loss p 2.710 v 0.512 | play 140s train 6s
```

- **decisive**: share of games ending in checkmate. Expect it near 0% at first, rising as the bot learns to mate. Material adjudication (scoring 300-ply games by material) is on for the first 10 generations to bootstrap.
- **capped**: share of games that hit 300 plies. This should fall.
- Every 5 generations: 100 games per variant against random, greedy and the checkpoint from 5 generations earlier, written to `eval/results.csv`. **Done-bar: at least 90% wins against greedy in both variants.**
- `checkpoints/log.csv` has every generation's numbers.

Expect a few minutes per generation on a 6 GB+ GPU. Train loss is a poor progress signal; the eval lines are the one to watch.

## Try it, export it

```bash
python play.py --variant drop --color white          # play the latest checkpoint in the terminal
python evaluate.py --ckpt checkpoints/gen_030.pt --vs greedy
python export.py                                     # latest -> ../web/public/models/dropchess.onnx (+ parity check)
python export.py --int8                              # also writes a smaller int8 copy
```

## Files

| | |
|---|---|
| `run.py` | the loop: self-play → replay buffer → train → checkpoint → eval |
| `model.py` | residual net; convolutional policy head (78 planes → 4,992 logits) |
| `selfplay.py` | feeds network outputs to the Rust `Runner` |
| `train.py` | masked policy cross-entropy + value MSE, mirror augmentation |
| `replay.py` | compact ring buffer (~3.4 KB per position), saved to `checkpoints/replay.npz` |
| `evaluate.py` | matches against random / greedy / older checkpoints |
| `export.py` | ONNX opset 17, dynamic batch, parity check against PyTorch |
| `play.py` | terminal play against a checkpoint |
| `engine.py` | `ENGINE=rust` (default) or `ENGINE=py` |
| `dropchess_py/` | python-chess reference engine, encoding, action mapping |
| `make_fixtures.py` | writes `fixtures/perft.json` and `fixtures/encoding.jsonl` |
| `configs/` | `gpu.yaml`, `cpu.yaml`, `smoke.yaml` |

## Changes from the training brief

- **Rust engine in the loop from day one.** The Rust engine already existed, so self-play, MCTS and encoding run in Rust (parallel across CPU cores with rayon). The python-chess engine is the test reference rather than the first training engine. `ENGINE=py` works for tests and the `Game` interface, not for training.
- **Convolutional policy head** instead of 1×1 conv → flatten → dense 4,992. The dense layer alone would be ~10M parameters (40 MB in the browser); this head gives the same 4,992 logits in the same order with ~1.4M parameters in total for 8×96.
- **Playout cap randomisation is on** (25% of moves get 200 simulations and become training data; the rest get 40). This gives about 2.5× more games per hour.
- **Mate-in-1 check in the search** (`mate_check: true`, the default). A leaf where the side to move can mate at once is scored as a proven win instead of by the network, and a network player always takes a mate in one. Without it, the early bot walked into drop-mates and lost almost every game to the greedy baseline.
- **Early-training search aids.** An untrained network gives near-uniform priors over ~150 legal moves, so plain PUCT (FPU reduction 0.2) kept re-visiting the first legal move and never noticed hanging pieces. Fixes: no FPU reduction at the root (`fpu_root: 0`); children shuffled on expansion; and for the first generations, a capture-aware material score blended into leaf values (`material_blend` 0.5, full for `blend_full_gens`, fading to 0 by `blend_zero_gen`). Eval, `play.py` and the browser always search on the network alone.
- **Value target mixes in the search value** (`value_q_mix: 0.5`): target = ½ game result + ½ root search value. Game results alone taught the value head almost nothing about material in the first 20 generations.
- **Raw planes stored as uint8** (reserve counts and halfmove clock as integers) and scaled on the GPU, so the 500k-position buffer fits in ~1.7 GB.
- **Under-promotion order** (the brief left it open): type = 64 + direction × 3 + piece, direction 0 = towards file a, 1 = straight, 2 = towards file h; piece 0 = knight, 1 = bishop, 2 = rook. See `fixtures/README.md`.

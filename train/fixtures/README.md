# Shared fixtures

Written by `train/make_fixtures.py` using the Python reference engine. The Rust engine (`engine/tests/fixtures.rs`, `train/tests/test_crosscheck.py`) and the web client's TypeScript `encode()` must reproduce them exactly. If the bot plays nonsense in the browser, check here first.

## perft.json

`[{fen, variant, perft: {"1": n, "2": n, "3": n}}]`: move counts for the start position and six hand-made positions (check, en passant, promotion, promoted pieces, a crowded middlegame), for each variant.

## encoding.jsonl

One JSON object per line:

| key | meaning |
|---|---|
| `start` | FEN the game starts from |
| `variant` | `drop` or `crazy` |
| `moves` | UCI moves played from `start` (replay them, since the repetition planes need history) |
| `fen` | FEN after the moves |
| `status` | `ongoing`, `checkmate`, `stalemate`, `repetition`, `fifty`, `insufficient` |
| `legal` | sorted legal moves in UCI (`e2e4`, `e7e8q`, `N@f3`) |
| `actions` | sorted legal action indices |
| `planes_sha256` | SHA-256 of `encode()` as float32 `[31, 8, 8]`, C order, little-endian |

## Encoding (summary; the training brief has the full table)

Seen from the side to move. For Black the board is mirrored vertically (square ^ 56). Square = rank × 8 + file, a1 = 0.

Planes: 0–5 my P N B R Q K · 6–11 opponent's · 12–16 my reserve P N B R Q (count / 8) · 17–21 opponent's reserve · 22–23 promoted pieces (mine, theirs) · 24–25 position seen before once / twice · 26 en passant square · 27 white to move · 28 Crazy Drop · 29 halfmove clock / 100 · 30 ones.

Actions (4,992):

- **Board moves:** `from × 73 + type`, squares in the flipped frame.
  - Types 0–55: direction × 7 + (distance − 1), with directions N, NE, E, SE, S, SW, W, NW (file delta, rank delta: (0,1), (1,1), (1,0), (1,−1), (0,−1), (−1,−1), (−1,0), (−1,1)). King moves, pawn moves and queen promotions live here.
  - Types 56–63: knight jumps, in the order (1,2), (2,1), (2,−1), (1,−2), (−1,−2), (−2,−1), (−2,1), (−1,2).
  - Types 64–72: under-promotions, 64 + dir × 3 + piece. dir 0 = towards file a, 1 = straight, 2 = towards file h. Piece 0 = knight, 1 = bishop, 2 = rook.
- **Drops:** `4672 + piece × 64 + to`, with piece order P N B R Q.

Network output `policy` is raw logits over all 4,992 indices. Mask the illegal ones before the softmax.

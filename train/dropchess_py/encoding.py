"""State planes and the 4,992-entry action space (see the training brief).

Everything is seen from the side to move. For Black the board is mirrored
vertically (square ^ 56), so "my half" is always ranks 1-4.

Board moves: index = from_sq * 73 + type, with
  types  0-55  queen-style: direction * 7 + (distance - 1),
               directions N, NE, E, SE, S, SW, W, NW
  types 56-63  knight jumps in KNIGHT order below
  types 64-72  under-promotions: 64 + dir * 3 + piece,
               dir 0 = capture towards file a, 1 = straight, 2 = towards file h,
               piece 0 = knight, 1 = bishop, 2 = rook
Drops: index = 4672 + piece * 64 + to_sq, piece order P, N, B, R, Q.
"""

import numpy as np

NUM_PLANES = 31
NUM_ACTIONS = 4992
DROP_BASE = 4672

# (file delta, rank delta)
DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

# Piece letters in plane / reserve order.
PIECES = "PNBRQK"
DROP_PIECES = "PNBRQ"
UNDERPROMO = "NBR"

# Multiply the raw (integer) planes by this to get the network input.
# Reserve planes hold counts (scaled by 1/8), plane 29 holds the halfmove clock (1/100).
PLANE_SCALE = np.ones(NUM_PLANES, dtype=np.float32)
PLANE_SCALE[12:22] = 1.0 / 8.0
PLANE_SCALE[29] = 1.0 / 100.0


def flip(sq: int, black: bool) -> int:
    return sq ^ 56 if black else sq


def encode_move_action(kind: str, frm: int, to: int, promo: str | None, black: bool) -> int:
    """kind is 'drop' (frm = piece index 0..4) or 'move'. Squares are absolute (a1 = 0)."""
    if kind == "drop":
        return DROP_BASE + frm * 64 + flip(to, black)
    f, t = flip(frm, black), flip(to, black)
    df = (t % 8) - (f % 8)
    dr = (t // 8) - (f // 8)
    if promo is not None and promo in UNDERPROMO:
        return f * 73 + 64 + (df + 1) * 3 + UNDERPROMO.index(promo)
    if (df, dr) in KNIGHT:
        return f * 73 + 56 + KNIGHT.index((df, dr))
    dist = max(abs(df), abs(dr))
    d = (df // dist if df else 0, dr // dist if dr else 0)
    return f * 73 + DIRS.index(d) * 7 + (dist - 1)


def decode_action(a: int):
    """Inverse of encode_move_action in the side-to-move frame.

    Returns ('drop', piece_idx, to) or ('move', from, to, promo_or_None) with
    squares still in the flipped frame. A queen-style move onto the last rank
    by a pawn is a queen promotion; the caller decides that.
    """
    if a >= DROP_BASE:
        r = a - DROP_BASE
        return ("drop", r // 64, r % 64)
    f, t = divmod(a, 73)
    ff, fr = f % 8, f // 8
    if t < 56:
        d, dist = divmod(t, 7)
        df, dr = DIRS[d]
        dist += 1
        promo = None
    elif t < 64:
        df, dr = KNIGHT[t - 56]
        dist = 1
        promo = None
    else:
        u = t - 64
        df, dr, dist = u // 3 - 1, 1, 1
        promo = UNDERPROMO[u % 3]
    tf, tr = ff + df * dist, fr + dr * dist
    if not (0 <= tf < 8 and 0 <= tr < 8):
        return None
    return ("move", f, tr * 8 + tf, promo)


def mirror_action(a: int) -> int:
    """Index of the same move with files mirrored a<->h (rules are left-right symmetric)."""
    d = decode_action(a)
    if d is None:
        return -1
    if d[0] == "drop":
        _, p, to = d
        return DROP_BASE + p * 64 + (to ^ 7)
    _, f, t, promo = d
    mf, mt = f ^ 7, t ^ 7
    if promo is not None:
        return encode_move_action("move", mf, mt, promo, False)
    return encode_move_action("move", mf, mt, None, False)


def mirror_table() -> np.ndarray:
    return np.array([mirror_action(a) for a in range(NUM_ACTIONS)], dtype=np.int64)

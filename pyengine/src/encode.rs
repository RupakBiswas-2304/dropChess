//! State planes and the 4,992-entry action space. Must match
//! train/dropchess_py/encoding.py exactly (checked by train/tests).

use engine::{Color, Kind, Move, Position, Sq};

pub const NUM_PLANES: usize = 31;
pub const NUM_ACTIONS: usize = 4992;
pub const DROP_BASE: usize = 4672;
pub const PLANE_LEN: usize = NUM_PLANES * 64;
pub const MASK_BYTES: usize = NUM_ACTIONS / 8;

/// (file delta, rank delta): N, NE, E, SE, S, SW, W, NW.
const DIRS: [(i8, i8); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
const KNIGHT: [(i8, i8); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

pub fn kind_idx(k: Kind) -> usize {
    match k {
        Kind::Pawn => 0,
        Kind::Knight => 1,
        Kind::Bishop => 2,
        Kind::Rook => 3,
        Kind::Queen => 4,
        Kind::King => 5,
    }
}

#[inline]
fn flip(sq: Sq, black: bool) -> usize {
    if black {
        (sq ^ 56) as usize
    } else {
        sq as usize
    }
}

/// Action index of a move for the side `side` (the mover).
pub fn action_of(m: Move, side: Color) -> usize {
    let black = side == Color::Black;
    match m {
        Move::Drop { kind, to } => DROP_BASE + kind_idx(kind) * 64 + flip(to, black),
        Move::Normal { from, to, promo } => {
            let f = flip(from, black) as i8;
            let t = flip(to, black) as i8;
            let df = (t % 8) - (f % 8);
            let dr = (t / 8) - (f / 8);
            let base = f as usize * 73;
            if let Some(p) = promo {
                let pi = match p {
                    Kind::Knight => Some(0),
                    Kind::Bishop => Some(1),
                    Kind::Rook => Some(2),
                    _ => None,
                };
                if let Some(pi) = pi {
                    return base + 64 + ((df + 1) as usize) * 3 + pi;
                }
            }
            if let Some(i) = KNIGHT.iter().position(|&d| d == (df, dr)) {
                return base + 56 + i;
            }
            let dist = df.abs().max(dr.abs());
            let d = (df.signum(), dr.signum());
            let di = DIRS.iter().position(|&x| x == d).expect("queen-style direction");
            base + di * 7 + (dist as usize - 1)
        }
    }
}

/// Raw planes: uint8 [31 * 64]. Network input = raw * plane_scale().
/// `seen_before` = how many earlier positions in the game equal this one.
pub fn encode_raw(pos: &Position, seen_before: usize, out: &mut [u8]) {
    debug_assert_eq!(out.len(), PLANE_LEN);
    out.fill(0);
    let us = pos.side;
    let black = us == Color::Black;
    let mut set = |plane: usize, sq: usize, v: u8| out[plane * 64 + sq] = v;
    for sq in 0..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            let s = flip(sq, black);
            let base = if p.color == us { 0 } else { 6 };
            set(base + kind_idx(p.kind), s, 1);
            if p.promoted {
                set(if p.color == us { 22 } else { 23 }, s, 1);
            }
        }
    }
    let fill = |out: &mut [u8], plane: usize, v: u8| out[plane * 64..(plane + 1) * 64].fill(v);
    for i in 0..5 {
        fill(out, 12 + i, pos.reserve[us.idx()][i]);
        fill(out, 17 + i, pos.reserve[us.opp().idx()][i]);
    }
    if seen_before >= 1 {
        fill(out, 24, 1);
    }
    if seen_before >= 2 {
        fill(out, 25, 1);
    }
    if let Some(ep) = pos.ep {
        out[26 * 64 + flip(ep, black)] = 1;
    }
    if !black {
        fill(out, 27, 1);
    }
    if pos.variant == engine::Variant::Crazy {
        fill(out, 28, 1);
    }
    fill(out, 29, pos.halfmove.min(255) as u8);
    fill(out, 30, 1);
}

pub fn plane_scale() -> [f32; NUM_PLANES] {
    let mut s = [1.0f32; NUM_PLANES];
    for v in s.iter_mut().take(22).skip(12) {
        *v = 1.0 / 8.0;
    }
    s[29] = 1.0 / 100.0;
    s
}

/// For each action, the action with files mirrored a<->h, or -1 if the index
/// can never be a legal move (it points off the board).
pub fn mirror_table() -> Vec<i64> {
    let mut t = vec![-1i64; NUM_ACTIONS];
    for (a, slot) in t.iter_mut().enumerate() {
        if a >= DROP_BASE {
            let r = a - DROP_BASE;
            *slot = (DROP_BASE + (r / 64) * 64 + ((r % 64) ^ 7)) as i64;
            continue;
        }
        let (f, ty) = (a / 73, a % 73);
        let (ff, fr) = ((f % 8) as i8, (f / 8) as i8);
        let (df, dr, dist, promo): (i8, i8, i8, Option<Kind>) = if ty < 56 {
            let (d, dist) = (ty / 7, ty % 7 + 1);
            (DIRS[d].0, DIRS[d].1, dist as i8, None)
        } else if ty < 64 {
            (KNIGHT[ty - 56].0, KNIGHT[ty - 56].1, 1, None)
        } else {
            let u = ty - 64;
            (u as i8 / 3 - 1, 1, 1, Some([Kind::Knight, Kind::Bishop, Kind::Rook][u % 3]))
        };
        let (tf, tr) = (ff + df * dist, fr + dr * dist);
        if !(0..8).contains(&tf) || !(0..8).contains(&tr) {
            continue;
        }
        let from = (fr * 8 + (7 - ff)) as Sq;
        let to = (tr * 8 + (7 - tf)) as Sq;
        *slot = action_of(Move::Normal { from, to, promo }, Color::White) as i64;
    }
    t
}

pub fn pack_mask(actions: &[u16], out: &mut [u8]) {
    out.fill(0);
    for &a in actions {
        out[a as usize / 8] |= 0x80 >> (a % 8); // numpy packbits bit order (big-endian)
    }
}


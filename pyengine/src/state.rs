//! A game state for search: position + repetition history + ply count.
//! Status rules mirror engine::Game::status, plus a training ply cap.

use crate::encode::{self, action_of, PLANE_LEN};
use engine::{Color, Kind, Move, Position, Variant};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Ongoing,
    Checkmate,
    Stalemate,
    Repetition,
    FiftyMove,
    Insufficient,
    MaxPlies,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Ongoing => "ongoing",
            Status::Checkmate => "checkmate",
            Status::Stalemate => "stalemate",
            Status::Repetition => "repetition",
            Status::FiftyMove => "fifty",
            Status::Insufficient => "insufficient",
            Status::MaxPlies => "maxplies",
        }
    }
}

#[derive(Clone)]
pub struct State {
    pub pos: Position,
    /// Repetition hashes of every position so far, current one last.
    pub hist: Vec<u64>,
    pub plies: u32,
    pub max_plies: u32,
}

fn rep_hash(p: &Position) -> u64 {
    let mut h = DefaultHasher::new();
    p.board.hash(&mut h);
    p.side.hash(&mut h);
    p.reserve.hash(&mut h);
    p.ep.hash(&mut h);
    h.finish()
}

pub fn piece_value(k: Kind) -> i32 {
    match k {
        Kind::Pawn => 1,
        Kind::Knight | Kind::Bishop => 3,
        Kind::Rook => 5,
        Kind::Queen => 9,
        Kind::King => 0,
    }
}

impl State {
    pub fn new(pos: Position, max_plies: u32) -> State {
        State { hist: vec![rep_hash(&pos)], pos, plies: 0, max_plies }
    }

    pub fn start(variant: Variant, max_plies: u32) -> State {
        State::new(Position::start(variant), max_plies)
    }

    pub fn side(&self) -> Color {
        self.pos.side
    }

    pub fn legal(&self) -> Vec<Move> {
        self.pos.legal_moves()
    }

    pub fn apply(&mut self, m: Move) {
        self.pos.make(m);
        self.hist.push(rep_hash(&self.pos));
        self.plies += 1;
    }

    pub fn seen_before(&self) -> usize {
        let cur = *self.hist.last().unwrap();
        self.hist[..self.hist.len() - 1].iter().filter(|&&h| h == cur).count()
    }

    /// Status given the already-generated legal moves of this position.
    pub fn status_with(&self, legal: &[Move]) -> Status {
        if legal.is_empty() {
            return if self.pos.in_check(self.pos.side) { Status::Checkmate } else { Status::Stalemate };
        }
        if self.seen_before() >= 2 {
            return Status::Repetition;
        }
        if self.pos.halfmove >= 100 {
            return Status::FiftyMove;
        }
        if self.pos.insufficient_material() {
            return Status::Insufficient;
        }
        if self.plies >= self.max_plies {
            return Status::MaxPlies;
        }
        Status::Ongoing
    }

    pub fn status(&self) -> Status {
        self.status_with(&self.legal())
    }

    /// A move that checkmates at once, if there is one.
    pub fn mate_in_one(&self, legal: &[Move]) -> Option<Move> {
        let us = self.pos.side;
        legal.iter().copied().find(|&m| {
            let mut n = self.pos;
            n.make(m);
            n.in_check(us.opp()) && !has_legal_move(&n)
        })
    }

    /// Material (board + reserve) of `c` minus the opponent's.
    pub fn material_diff(&self, c: Color) -> i32 {
        let mut d = 0;
        for p in self.pos.board.iter().flatten() {
            let v = piece_value(p.kind);
            d += if p.color == c { v } else { -v };
        }
        for (i, k) in Kind::DROPPABLE.iter().enumerate() {
            let v = piece_value(*k);
            d += v * self.pos.reserve[c.idx()][i] as i32;
            d -= v * self.pos.reserve[c.opp().idx()][i] as i32;
        }
        d
    }

    /// Value of a finished game for the side to move.
    /// A capped game is scored by material when `adjudicate` is on (clamped to +-0.5).
    pub fn terminal_value(&self, s: Status, adjudicate: bool) -> f32 {
        match s {
            Status::Checkmate => -1.0,
            Status::MaxPlies if adjudicate => adjudicated(self.material_diff(self.side())),
            _ => 0.0,
        }
    }

    pub fn encode_raw(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), PLANE_LEN);
        encode::encode_raw(&self.pos, self.seen_before(), out);
    }

    pub fn action(&self, m: Move) -> usize {
        action_of(m, self.pos.side)
    }
}

/// Does the side to move have any legal move? Stops at the first one found.
pub fn has_legal_move(p: &Position) -> bool {
    let side = p.side;
    // King moves first: when in check they are the most likely way out.
    let mut moves = p.pseudo_legal();
    if let Some(k) = p.king_sq(side) {
        moves.sort_by_key(|m| !matches!(m, Move::Normal { from, .. } if *from == k));
    }
    moves.into_iter().any(|m| {
        let mut n = *p;
        n.make(m);
        !n.in_check(side)
    })
}

pub fn adjudicated(diff: i32) -> f32 {
    (diff as f32 / 10.0).clamp(-0.5, 0.5)
}

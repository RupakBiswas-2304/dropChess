//! Drop Chess rules engine.
//!
//! Both sides start with only their kings on the board. A turn is either a
//! normal chess move or a drop of a reserve piece onto the mover's own half
//! (ranks 1-4 for White, 5-8 for Black). In the Crazy variant, captured pieces
//! join the capturer's reserve (promoted pieces return as pawns).
//!
//! Squares are 0..64 with a1 = 0, h1 = 7, a8 = 56, h8 = 63.

use serde::{Deserialize, Serialize};
use std::fmt;

pub type Sq = u8;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opp(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        match self {
            Color::White => "white",
            Color::Black => "black",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum Kind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl Kind {
    /// Pieces that can sit in a reserve, in reserve-index order.
    pub const DROPPABLE: [Kind; 5] = [Kind::Pawn, Kind::Knight, Kind::Bishop, Kind::Rook, Kind::Queen];

    pub fn reserve_idx(self) -> usize {
        match self {
            Kind::Pawn => 0,
            Kind::Knight => 1,
            Kind::Bishop => 2,
            Kind::Rook => 3,
            Kind::Queen => 4,
            Kind::King => panic!("king has no reserve slot"),
        }
    }
    pub fn letter(self) -> char {
        match self {
            Kind::Pawn => 'P',
            Kind::Knight => 'N',
            Kind::Bishop => 'B',
            Kind::Rook => 'R',
            Kind::Queen => 'Q',
            Kind::King => 'K',
        }
    }
    pub fn from_letter(c: char) -> Option<Kind> {
        Some(match c.to_ascii_uppercase() {
            'P' => Kind::Pawn,
            'N' => Kind::Knight,
            'B' => Kind::Bishop,
            'R' => Kind::Rook,
            'Q' => Kind::Queen,
            'K' => Kind::King,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Piece {
    pub color: Color,
    pub kind: Kind,
    /// True for a piece that was promoted from a pawn (returns as a pawn when captured in Crazy).
    pub promoted: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Variant {
    /// Captured pieces leave the game.
    Drop,
    /// Captured pieces go to the capturer's reserve.
    Crazy,
}

impl Variant {
    pub fn parse(s: &str) -> Option<Variant> {
        match s {
            "drop" => Some(Variant::Drop),
            "crazy" => Some(Variant::Crazy),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Move {
    Normal { from: Sq, to: Sq, promo: Option<Kind> },
    Drop { kind: Kind, to: Sq },
}

pub fn file_of(sq: Sq) -> u8 {
    sq % 8
}
pub fn rank_of(sq: Sq) -> u8 {
    sq / 8
}
pub fn sq_name(sq: Sq) -> String {
    format!("{}{}", (b'a' + file_of(sq)) as char, rank_of(sq) + 1)
}
pub fn parse_sq(s: &str) -> Option<Sq> {
    let b = s.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return None;
    }
    Some((b[1] - b'1') * 8 + (b[0] - b'a'))
}

fn offset(sq: Sq, df: i8, dr: i8) -> Option<Sq> {
    let f = file_of(sq) as i8 + df;
    let r = rank_of(sq) as i8 + dr;
    if (0..8).contains(&f) && (0..8).contains(&r) {
        Some((r * 8 + f) as Sq)
    } else {
        None
    }
}

const KNIGHT: [(i8, i8); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
const KING: [(i8, i8); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
const ORTHO: [(i8, i8); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
const DIAG: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, -1), (-1, 1)];

impl Move {
    pub fn uci(&self) -> String {
        match *self {
            Move::Normal { from, to, promo } => {
                let mut s = format!("{}{}", sq_name(from), sq_name(to));
                if let Some(k) = promo {
                    s.push(k.letter().to_ascii_lowercase());
                }
                s
            }
            Move::Drop { kind, to } => format!("{}@{}", kind.letter(), sq_name(to)),
        }
    }

    pub fn parse_uci(s: &str) -> Option<Move> {
        if let Some((p, sq)) = s.split_once('@') {
            let mut chars = p.chars();
            let kind = Kind::from_letter(chars.next()?)?;
            if chars.next().is_some() || kind == Kind::King {
                return None;
            }
            return Some(Move::Drop { kind, to: parse_sq(sq)? });
        }
        if s.len() != 4 && s.len() != 5 {
            return None;
        }
        let from = parse_sq(s.get(0..2)?)?;
        let to = parse_sq(s.get(2..4)?)?;
        let promo = match s.get(4..5) {
            Some(c) => Some(Kind::from_letter(c.chars().next()?)?),
            None => None,
        };
        Some(Move::Normal { from, to, promo })
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.uci())
    }
}

/// A position without history. Cheap to copy, used for legality checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Position {
    pub board: [Option<Piece>; 64],
    pub side: Color,
    /// reserve[color][Kind::reserve_idx()]
    pub reserve: [[u8; 5]; 2],
    pub ep: Option<Sq>,
    pub halfmove: u32,
    pub fullmove: u32,
    pub variant: Variant,
}

pub const START_FEN: &str = "4k3/8/8/8/8/8/8/4K3[QRRBBNNPPPPPPPPqrrbbnnpppppppp] w - - 0 1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenError(pub String);

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid FEN: {}", self.0)
    }
}

impl Position {
    pub fn start(variant: Variant) -> Position {
        Position::from_fen(START_FEN, variant).expect("start FEN is valid")
    }

    /// Parses a Crazyhouse-style FEN: `placement[reserve] side castling ep half full`.
    /// Promoted pieces are marked with `~` after the letter. The castling field is ignored.
    pub fn from_fen(fen: &str, variant: Variant) -> Result<Position, FenError> {
        let err = |m: &str| FenError(m.to_string());
        let mut parts = fen.split_whitespace();
        let placement_full = parts.next().ok_or_else(|| err("empty"))?;
        let (placement, reserve_str) = match placement_full.find('[') {
            Some(i) => {
                let r = placement_full[i + 1..].strip_suffix(']').ok_or_else(|| err("unclosed reserve"))?;
                (&placement_full[..i], r)
            }
            None => (placement_full, ""),
        };
        let mut board = [None; 64];
        let rows: Vec<&str> = placement.split('/').collect();
        if rows.len() != 8 {
            return Err(err("need 8 ranks"));
        }
        for (i, row) in rows.iter().enumerate() {
            let rank = 7 - i as u8;
            let mut file = 0u8;
            let chars: Vec<char> = row.chars().collect();
            let mut j = 0;
            while j < chars.len() {
                let c = chars[j];
                if let Some(d) = c.to_digit(10) {
                    file += d as u8;
                } else {
                    let kind = Kind::from_letter(c).ok_or_else(|| err("bad piece letter"))?;
                    let color = if c.is_ascii_uppercase() { Color::White } else { Color::Black };
                    let promoted = chars.get(j + 1) == Some(&'~');
                    if promoted {
                        j += 1;
                    }
                    if file > 7 {
                        return Err(err("rank too long"));
                    }
                    board[(rank * 8 + file) as usize] = Some(Piece { color, kind, promoted });
                    file += 1;
                }
                j += 1;
            }
            if file != 8 {
                return Err(err("rank has wrong length"));
            }
        }
        let mut reserve = [[0u8; 5]; 2];
        for c in reserve_str.chars() {
            let kind = Kind::from_letter(c).ok_or_else(|| err("bad reserve letter"))?;
            if kind == Kind::King {
                return Err(err("king in reserve"));
            }
            let color = if c.is_ascii_uppercase() { Color::White } else { Color::Black };
            reserve[color.idx()][kind.reserve_idx()] += 1;
        }
        let side = match parts.next().unwrap_or("w") {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(err("bad side")),
        };
        let _castling = parts.next();
        let ep = match parts.next().unwrap_or("-") {
            "-" => None,
            s => Some(parse_sq(s).ok_or_else(|| err("bad ep square"))?),
        };
        let halfmove = parts.next().unwrap_or("0").parse().map_err(|_| err("bad halfmove"))?;
        let fullmove = parts.next().unwrap_or("1").parse().map_err(|_| err("bad fullmove"))?;
        let pos = Position { board, side, reserve, ep, halfmove, fullmove, variant };
        for color in [Color::White, Color::Black] {
            let kings = pos.board.iter().filter(|p| matches!(p, Some(pc) if pc.color == color && pc.kind == Kind::King)).count();
            if kings != 1 {
                return Err(err("each side needs exactly one king"));
            }
        }
        Ok(pos)
    }

    fn placement_key(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match self.board[rank * 8 + file] {
                    None => empty += 1,
                    Some(p) => {
                        if empty > 0 {
                            s.push_str(&empty.to_string());
                            empty = 0;
                        }
                        let c = p.kind.letter();
                        s.push(if p.color == Color::White { c } else { c.to_ascii_lowercase() });
                        if p.promoted {
                            s.push('~');
                        }
                    }
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push('[');
        for color in [Color::White, Color::Black] {
            for kind in [Kind::Queen, Kind::Rook, Kind::Bishop, Kind::Knight, Kind::Pawn] {
                for _ in 0..self.reserve[color.idx()][kind.reserve_idx()] {
                    let c = kind.letter();
                    s.push(if color == Color::White { c } else { c.to_ascii_lowercase() });
                }
            }
        }
        s.push(']');
        s
    }

    pub fn fen(&self) -> String {
        format!(
            "{} {} - {} {} {}",
            self.placement_key(),
            if self.side == Color::White { "w" } else { "b" },
            self.ep.map(sq_name).unwrap_or_else(|| "-".into()),
            self.halfmove,
            self.fullmove
        )
    }

    /// Key used for repetition detection: placement, reserves, side to move, ep square.
    pub fn repetition_key(&self) -> String {
        format!(
            "{} {} {}",
            self.placement_key(),
            if self.side == Color::White { "w" } else { "b" },
            self.ep.map(sq_name).unwrap_or_else(|| "-".into())
        )
    }

    pub fn king_sq(&self, color: Color) -> Option<Sq> {
        (0..64u8).find(|&s| matches!(self.board[s as usize], Some(p) if p.color == color && p.kind == Kind::King))
    }

    /// Is `sq` attacked by any piece of color `by`?
    pub fn attacked(&self, sq: Sq, by: Color) -> bool {
        let is = |s: Option<Sq>, kinds: &[Kind]| -> bool {
            match s.and_then(|s| self.board[s as usize]) {
                Some(p) => p.color == by && kinds.contains(&p.kind),
                None => false,
            }
        };
        // Pawns: a white pawn attacks diagonally upward, so look one rank below the target.
        let pr: i8 = if by == Color::White { -1 } else { 1 };
        if is(offset(sq, -1, pr), &[Kind::Pawn]) || is(offset(sq, 1, pr), &[Kind::Pawn]) {
            return true;
        }
        if KNIGHT.iter().any(|&(df, dr)| is(offset(sq, df, dr), &[Kind::Knight])) {
            return true;
        }
        if KING.iter().any(|&(df, dr)| is(offset(sq, df, dr), &[Kind::King])) {
            return true;
        }
        for (dirs, kinds) in [(&ORTHO, [Kind::Rook, Kind::Queen]), (&DIAG, [Kind::Bishop, Kind::Queen])] {
            for &(df, dr) in dirs.iter() {
                let mut cur = sq;
                while let Some(n) = offset(cur, df, dr) {
                    if let Some(p) = self.board[n as usize] {
                        if p.color == by && kinds.contains(&p.kind) {
                            return true;
                        }
                        break;
                    }
                    cur = n;
                }
            }
        }
        false
    }

    pub fn in_check(&self, color: Color) -> bool {
        match self.king_sq(color) {
            Some(k) => self.attacked(k, color.opp()),
            None => false,
        }
    }

    /// Is `sq` on `color`'s own half (where it may drop)?
    pub fn own_half(color: Color, sq: Sq) -> bool {
        match color {
            Color::White => rank_of(sq) <= 3,
            Color::Black => rank_of(sq) >= 4,
        }
    }

    pub fn pseudo_legal(&self) -> Vec<Move> {
        let us = self.side;
        let mut out = Vec::with_capacity(160);
        for from in 0..64u8 {
            let Some(p) = self.board[from as usize] else { continue };
            if p.color != us {
                continue;
            }
            match p.kind {
                Kind::Pawn => self.pawn_moves(from, &mut out),
                Kind::Knight => self.step_moves(from, &KNIGHT, &mut out),
                Kind::King => self.step_moves(from, &KING, &mut out),
                Kind::Bishop => self.slide_moves(from, &DIAG, &mut out),
                Kind::Rook => self.slide_moves(from, &ORTHO, &mut out),
                Kind::Queen => {
                    self.slide_moves(from, &DIAG, &mut out);
                    self.slide_moves(from, &ORTHO, &mut out);
                }
            }
        }
        let back_rank = if us == Color::White { 0 } else { 7 };
        for kind in Kind::DROPPABLE {
            if self.reserve[us.idx()][kind.reserve_idx()] == 0 {
                continue;
            }
            for to in 0..64u8 {
                if self.board[to as usize].is_some() || !Position::own_half(us, to) {
                    continue;
                }
                if kind == Kind::Pawn && rank_of(to) == back_rank {
                    continue;
                }
                out.push(Move::Drop { kind, to });
            }
        }
        out
    }

    fn pawn_moves(&self, from: Sq, out: &mut Vec<Move>) {
        let us = self.side;
        let dir: i8 = if us == Color::White { 1 } else { -1 };
        let start_rank = if us == Color::White { 1 } else { 6 };
        let promo_rank = if us == Color::White { 7 } else { 0 };
        let push = |to: Sq, out: &mut Vec<Move>| {
            if rank_of(to) == promo_rank {
                for k in [Kind::Queen, Kind::Rook, Kind::Bishop, Kind::Knight] {
                    out.push(Move::Normal { from, to, promo: Some(k) });
                }
            } else {
                out.push(Move::Normal { from, to, promo: None });
            }
        };
        if let Some(one) = offset(from, 0, dir) {
            if self.board[one as usize].is_none() {
                push(one, out);
                if rank_of(from) == start_rank {
                    if let Some(two) = offset(from, 0, 2 * dir) {
                        if self.board[two as usize].is_none() {
                            out.push(Move::Normal { from, to: two, promo: None });
                        }
                    }
                }
            }
        }
        for df in [-1, 1] {
            if let Some(to) = offset(from, df, dir) {
                let enemy = matches!(self.board[to as usize], Some(p) if p.color != us);
                if enemy || self.ep == Some(to) {
                    push(to, out);
                }
            }
        }
    }

    fn step_moves(&self, from: Sq, steps: &[(i8, i8)], out: &mut Vec<Move>) {
        for &(df, dr) in steps {
            if let Some(to) = offset(from, df, dr) {
                match self.board[to as usize] {
                    Some(p) if p.color == self.side => {}
                    _ => out.push(Move::Normal { from, to, promo: None }),
                }
            }
        }
    }

    fn slide_moves(&self, from: Sq, dirs: &[(i8, i8)], out: &mut Vec<Move>) {
        for &(df, dr) in dirs {
            let mut cur = from;
            while let Some(to) = offset(cur, df, dr) {
                match self.board[to as usize] {
                    None => out.push(Move::Normal { from, to, promo: None }),
                    Some(p) => {
                        if p.color != self.side {
                            out.push(Move::Normal { from, to, promo: None });
                        }
                        break;
                    }
                }
                cur = to;
            }
        }
    }

    /// Applies a move without checking legality.
    pub fn make(&mut self, m: Move) {
        let us = self.side;
        let mut reset_clock = false;
        let mut new_ep = None;
        match m {
            Move::Drop { kind, to } => {
                self.reserve[us.idx()][kind.reserve_idx()] -= 1;
                self.board[to as usize] = Some(Piece { color: us, kind, promoted: false });
                reset_clock = true;
            }
            Move::Normal { from, to, promo } => {
                let piece = self.board[from as usize].expect("move from empty square");
                let mut captured = self.board[to as usize];
                if piece.kind == Kind::Pawn {
                    reset_clock = true;
                    if self.ep == Some(to) && captured.is_none() {
                        let victim = if us == Color::White { to - 8 } else { to + 8 };
                        captured = self.board[victim as usize].take();
                    }
                    if (to as i8 - from as i8).abs() == 16 {
                        new_ep = Some((from + to) / 2);
                    }
                }
                if let Some(c) = captured {
                    reset_clock = true;
                    if self.variant == Variant::Crazy && c.kind != Kind::King {
                        let kind = if c.promoted { Kind::Pawn } else { c.kind };
                        self.reserve[us.idx()][kind.reserve_idx()] += 1;
                    }
                }
                self.board[from as usize] = None;
                self.board[to as usize] = Some(match promo {
                    Some(k) => Piece { color: us, kind: k, promoted: true },
                    None => piece,
                });
            }
        }
        self.ep = new_ep;
        if reset_clock {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }
        if us == Color::Black {
            self.fullmove += 1;
        }
        self.side = us.opp();
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let us = self.side;
        self.pseudo_legal()
            .into_iter()
            .filter(|&m| {
                let mut next = *self;
                next.make(m);
                !next.in_check(us)
            })
            .collect()
    }

    /// Only kings (plus at most one knight or bishop) on the board and both reserves empty.
    pub fn insufficient_material(&self) -> bool {
        if self.reserve.iter().flatten().any(|&n| n > 0) {
            return false;
        }
        let mut minors = 0;
        for p in self.board.iter().flatten() {
            match p.kind {
                Kind::King => {}
                Kind::Knight | Kind::Bishop => minors += 1,
                _ => return false,
            }
        }
        minors <= 1
    }

    /// Standard algebraic notation for a legal move (drops as `N@f3`).
    pub fn san(&self, m: Move) -> String {
        let legal = self.legal_moves();
        let mut s = match m {
            Move::Drop { kind, to } => format!("{}@{}", if kind == Kind::Pawn { 'P' } else { kind.letter() }, sq_name(to)),
            Move::Normal { from, to, promo } => {
                let piece = self.board[from as usize].expect("piece");
                let capture = self.board[to as usize].is_some() || (piece.kind == Kind::Pawn && self.ep == Some(to));
                let mut s = String::new();
                if piece.kind == Kind::Pawn {
                    if capture {
                        s.push((b'a' + file_of(from)) as char);
                    }
                } else {
                    s.push(piece.kind.letter());
                    let rivals: Vec<Sq> = legal
                        .iter()
                        .filter_map(|&o| match o {
                            Move::Normal { from: f2, to: t2, .. }
                                if t2 == to && f2 != from && self.board[f2 as usize].map(|p| p.kind) == Some(piece.kind) =>
                            {
                                Some(f2)
                            }
                            _ => None,
                        })
                        .collect();
                    if !rivals.is_empty() {
                        if rivals.iter().all(|&r| file_of(r) != file_of(from)) {
                            s.push((b'a' + file_of(from)) as char);
                        } else if rivals.iter().all(|&r| rank_of(r) != rank_of(from)) {
                            s.push((b'1' + rank_of(from)) as char);
                        } else {
                            s.push_str(&sq_name(from));
                        }
                    }
                }
                if capture {
                    s.push('x');
                }
                s.push_str(&sq_name(to));
                if let Some(k) = promo {
                    s.push('=');
                    s.push(k.letter());
                }
                s
            }
        };
        let mut next = *self;
        next.make(m);
        if next.in_check(next.side) {
            s.push(if next.legal_moves().is_empty() { '#' } else { '+' });
        }
        s
    }
}

pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = pos.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .into_iter()
        .map(|m| {
            let mut next = *pos;
            next.make(m);
            perft(&next, depth - 1)
        })
        .sum()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Status {
    Ongoing,
    Checkmate { winner: Color },
    Stalemate,
    Repetition,
    FiftyMove,
    Insufficient,
    Resigned { winner: Color },
    Abandoned { winner: Color },
    Timeout { winner: Color },
}

impl Status {
    pub fn is_over(&self) -> bool {
        !matches!(self, Status::Ongoing)
    }
}

/// A game with history: tracks repetitions and the moves played.
#[derive(Clone, Debug)]
pub struct Game {
    pub pos: Position,
    history: Vec<String>,
    moves: Vec<Move>,
    sans: Vec<String>,
    forced: Option<Status>,
}

impl Game {
    pub fn new(variant: Variant) -> Game {
        Game::from_position(Position::start(variant))
    }

    pub fn from_fen(fen: &str, variant: Variant) -> Result<Game, FenError> {
        Ok(Game::from_position(Position::from_fen(fen, variant)?))
    }

    pub fn from_position(pos: Position) -> Game {
        Game { history: vec![pos.repetition_key()], pos, moves: vec![], sans: vec![], forced: None }
    }

    pub fn variant(&self) -> Variant {
        self.pos.variant
    }
    pub fn moves(&self) -> &[Move] {
        &self.moves
    }
    pub fn sans(&self) -> &[String] {
        &self.sans
    }
    /// Repetition keys of every position so far, the current one last.
    pub fn history(&self) -> &[String] {
        &self.history
    }
    pub fn legal_moves(&self) -> Vec<Move> {
        if self.status().is_over() {
            vec![]
        } else {
            self.pos.legal_moves()
        }
    }

    /// Plays a move given in UCI notation. Returns an error if it is not legal.
    pub fn play_uci(&mut self, uci: &str) -> Result<Move, String> {
        let m = Move::parse_uci(uci).ok_or_else(|| format!("unreadable move: {uci}"))?;
        self.play(m)?;
        Ok(m)
    }

    pub fn play(&mut self, m: Move) -> Result<(), String> {
        if self.status().is_over() {
            return Err("game is over".into());
        }
        if !self.pos.legal_moves().contains(&m) {
            return Err(format!("illegal move: {}", m.uci()));
        }
        let san = self.pos.san(m);
        self.pos.make(m);
        self.history.push(self.pos.repetition_key());
        self.moves.push(m);
        self.sans.push(san);
        Ok(())
    }

    /// Ends the game for a reason outside the board (resignation, abandonment).
    pub fn force_end(&mut self, status: Status) {
        if !self.status().is_over() {
            self.forced = Some(status);
        }
    }

    pub fn status(&self) -> Status {
        if let Some(s) = self.forced {
            return s;
        }
        let pos = &self.pos;
        if pos.legal_moves().is_empty() {
            return if pos.in_check(pos.side) { Status::Checkmate { winner: pos.side.opp() } } else { Status::Stalemate };
        }
        let key = self.history.last().expect("history never empty");
        if self.history.iter().filter(|k| *k == key).count() >= 3 {
            return Status::Repetition;
        }
        if pos.halfmove >= 100 {
            return Status::FiftyMove;
        }
        if pos.insufficient_material() {
            return Status::Insufficient;
        }
        Status::Ongoing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legal_uci(pos: &Position) -> Vec<String> {
        let mut v: Vec<String> = pos.legal_moves().iter().map(|m| m.uci()).collect();
        v.sort();
        v
    }

    #[test]
    fn perft1_start_is_153() {
        for v in [Variant::Drop, Variant::Crazy] {
            assert_eq!(perft(&Position::start(v), 1), 153);
        }
    }

    #[test]
    fn perft2_matches_between_variants_at_start() {
        // No capture is possible within two plies from the start, so both variants agree.
        let a = perft(&Position::start(Variant::Drop), 2);
        let b = perft(&Position::start(Variant::Crazy), 2);
        assert_eq!(a, b);
        assert!(a > 153 * 100);
    }

    #[test]
    fn fen_round_trip() {
        let pos = Position::start(Variant::Drop);
        assert_eq!(pos.fen(), START_FEN);
        let fen = "4k3/8/8/8/8/8/8/3q~K3[NNp] b - - 3 20";
        assert_eq!(Position::from_fen(fen, Variant::Crazy).unwrap().fen(), fen);
    }

    #[test]
    fn no_pawn_drop_on_own_back_rank_or_enemy_half() {
        let pos = Position::start(Variant::Drop);
        let moves = legal_uci(&pos);
        assert!(!moves.contains(&"P@a1".to_string()));
        assert!(moves.contains(&"P@a2".to_string()));
        assert!(moves.contains(&"N@a1".to_string()));
        assert!(moves.contains(&"Q@h4".to_string()));
        assert!(!moves.contains(&"Q@h5".to_string()));
        let mut black = pos;
        black.make(Move::parse_uci("N@c3").unwrap());
        let moves = legal_uci(&black);
        assert!(!moves.contains(&"P@a8".to_string()));
        assert!(moves.contains(&"P@a7".to_string()));
        assert!(moves.contains(&"Q@d5".to_string()));
        assert!(!moves.contains(&"Q@d4".to_string()));
    }

    #[test]
    fn drop_can_block_check() {
        let pos = Position::from_fen("4r2k/8/8/8/8/8/8/4K3[Q] w - - 0 1", Variant::Drop).unwrap();
        let moves = legal_uci(&pos);
        assert!(moves.contains(&"Q@e2".to_string()));
        assert!(moves.contains(&"Q@e4".to_string()));
        assert!(!moves.contains(&"Q@a2".to_string()), "a drop that ignores check is illegal");
        assert!(!moves.contains(&"e1e2".to_string()), "king cannot stay on the checked file");
    }

    #[test]
    fn drop_mate_is_legal() {
        let mut g = Game::from_fen("8/8/8/8/8/8/2K5/k7[R] w - - 0 1", Variant::Drop).unwrap();
        g.play_uci("R@a3").unwrap();
        assert_eq!(g.status(), Status::Checkmate { winner: Color::White });
        assert_eq!(g.sans()[0], "R@a3#");
    }

    #[test]
    fn capture_goes_to_reserve_only_in_crazy() {
        let fen = "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1";
        let mut crazy = Position::from_fen(fen, Variant::Crazy).unwrap();
        crazy.make(Move::parse_uci("e1d2").unwrap());
        assert_eq!(crazy.reserve[0][Kind::Pawn.reserve_idx()], 1);
        let mut drop = Position::from_fen(fen, Variant::Drop).unwrap();
        drop.make(Move::parse_uci("e1d2").unwrap());
        assert_eq!(drop.reserve[0], [0; 5]);
    }

    #[test]
    fn captured_promoted_piece_returns_as_pawn() {
        let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/3q~K3 w - - 0 1", Variant::Crazy).unwrap();
        pos.make(Move::parse_uci("e1d1").unwrap());
        assert_eq!(pos.reserve[0][Kind::Pawn.reserve_idx()], 1);
        assert_eq!(pos.reserve[0][Kind::Queen.reserve_idx()], 0);
    }

    #[test]
    fn dropped_pawn_on_second_rank_can_double_step_and_be_taken_en_passant() {
        let mut g = Game::from_fen("4k3/8/8/8/3p4/8/8/4K3[P] w - - 0 1", Variant::Drop).unwrap();
        // Black pawn on d4 belongs to Black; White drops a pawn on e2 and pushes it two squares.
        g.play_uci("P@e2").unwrap();
        g.play_uci("e8d8").unwrap();
        g.play_uci("e2e4").unwrap();
        assert_eq!(g.pos.ep, Some(parse_sq("e3").unwrap()));
        assert!(g.legal_moves().iter().any(|m| m.uci() == "d4e3"));
        g.play_uci("d4e3").unwrap();
        assert!(g.pos.board[parse_sq("e4").unwrap() as usize].is_none());
    }

    #[test]
    fn dropped_pawn_on_third_rank_moves_one_square() {
        let pos = Position::from_fen("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1", Variant::Drop).unwrap();
        let moves = legal_uci(&pos);
        assert!(moves.contains(&"e3e4".to_string()));
        assert!(!moves.contains(&"e3e5".to_string()));
    }

    #[test]
    fn promotion_offers_four_pieces() {
        let pos = Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", Variant::Drop).unwrap();
        let moves = legal_uci(&pos);
        for p in ["a7a8q", "a7a8r", "a7a8b", "a7a8n"] {
            assert!(moves.contains(&p.to_string()));
        }
    }

    #[test]
    fn drop_resets_fifty_move_counter() {
        let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3[N] w - - 40 30", Variant::Drop).unwrap();
        pos.make(Move::parse_uci("N@c3").unwrap());
        assert_eq!(pos.halfmove, 0);
    }

    #[test]
    fn kings_only_is_draw_by_insufficient_material() {
        let g = Game::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1", Variant::Drop).unwrap();
        assert_eq!(g.status(), Status::Insufficient);
        let g = Game::new(Variant::Drop);
        assert_eq!(g.status(), Status::Ongoing);
    }

    #[test]
    fn threefold_repetition() {
        let mut g = Game::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", Variant::Drop).unwrap();
        for _ in 0..2 {
            for m in ["e1f1", "e8f8", "f1e1", "f8e8"] {
                g.play_uci(m).unwrap();
            }
        }
        assert_eq!(g.status(), Status::Repetition);
    }

    #[test]
    fn illegal_moves_are_rejected() {
        let mut g = Game::new(Variant::Drop);
        assert!(g.play_uci("Q@e5").is_err());
        assert!(g.play_uci("e1e3").is_err());
        assert!(g.play_uci("K@e4").is_err());
        assert!(g.play_uci("zz").is_err());
        assert!(g.play_uci("Q@e4").is_ok());
    }

    #[test]
    fn random_games_never_panic_and_stay_consistent() {
        // Tiny deterministic PRNG so the test needs no dependencies.
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        for variant in [Variant::Drop, Variant::Crazy] {
            for _ in 0..40 {
                let mut g = Game::new(variant);
                for _ in 0..300 {
                    let moves = g.legal_moves();
                    if moves.is_empty() {
                        break;
                    }
                    let m = moves[rnd(moves.len())];
                    g.play(m).unwrap();
                    let re = Position::from_fen(&g.pos.fen(), variant).unwrap();
                    assert_eq!(re.fen(), g.pos.fen());
                    // Material is conserved in Crazy: 16 pieces per side counting reserves... minus promotions.
                    if variant == Variant::Crazy {
                        let on_board = g.pos.board.iter().flatten().count();
                        let in_hand: u32 = g.pos.reserve.iter().flatten().map(|&n| n as u32).sum();
                        assert_eq!(on_board as u32 + in_hand, 32);
                    }
                }
            }
        }
    }
}

//! Many games played in parallel, each side driven by a network (through MCTS),
//! a random player or the greedy baseline. Used for self-play and evaluation.
//!
//! Python loop:
//!     while not runner.done():
//!         planes, nets = runner.step()      # leaves that need the network
//!         logits, values = net(planes)      # grouped by net id on the Python side
//!         runner.feed(logits, values)
//!     games = runner.take_finished()

use crate::encode::{pack_mask, MASK_BYTES, PLANE_LEN};
use crate::mcts::{MctsCfg, Search, StepOut};
use crate::state::{adjudicated, piece_value, State, Status};
use engine::{Color, Move, Variant};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Player {
    Net(u8),
    Random,
    Greedy,
}

impl Player {
    pub fn parse(s: &str) -> Option<Player> {
        match s {
            "random" => Some(Player::Random),
            "greedy" => Some(Player::Greedy),
            _ => s.strip_prefix("net").and_then(|n| n.parse().ok()).map(Player::Net),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RunCfg {
    pub mcts: MctsCfg,
    pub sims_full: u32,
    pub sims_fast: u32,
    /// Probability that a move gets a full search (playout cap randomisation). 1.0 = always.
    pub p_full: f32,
    /// Dirichlet noise at the root of full searches.
    pub noise: bool,
    /// Sample moves in proportion to visits for this many plies, then play the most visited.
    pub temp_plies: u32,
    pub max_plies: u32,
    /// Record training samples (full searches only).
    pub record: bool,
    /// Keep at most this many policy entries per sample (largest visit counts).
    pub topk: usize,
}

pub struct Sample {
    pub planes: Vec<u8>,
    pub mask: Vec<u8>,
    pub pol_idx: Vec<u16>,
    pub pol_p: Vec<f32>,
    pub side: Color,
    /// Root search value for the side to move.
    pub q: f32,
}

pub struct Finished {
    pub id: u64,
    pub variant: Variant,
    pub players: [Player; 2],
    pub status: Status,
    /// +1 White won, -1 Black won, 0 draw.
    pub result: i8,
    pub plies: u32,
    pub samples: Vec<Sample>,
    /// Value target per sample (same order), from that sample's side to move.
    pub z: Vec<f32>,
    pub moves: Vec<String>,
}

struct GameSpec {
    id: u64,
    variant: Variant,
    players: [Player; 2],
    start: Option<engine::Position>,
    max_moves: Option<u32>,
}

struct Slot {
    spec: Option<GameSpec>,
    state: State,
    search: Option<Search>,
    samples: Vec<Sample>,
    moves: Vec<String>,
    planes: Vec<u8>,
    rng: SmallRng,
    done: Option<Status>,
}

pub struct Runner {
    pub cfg: RunCfg,
    slots: Vec<Slot>,
    queue: VecDeque<GameSpec>,
    finished: Vec<Finished>,
    next_id: u64,
    /// slot index of each row returned by the last step()
    rows: Vec<usize>,
}

impl Runner {
    pub fn new(cfg: RunCfg, parallel: usize, seed: u64) -> Runner {
        let slots = (0..parallel.max(1))
            .map(|i| Slot {
                spec: None,
                state: State::start(Variant::Drop, cfg.max_plies),
                search: None,
                samples: vec![],
                moves: vec![],
                planes: vec![0; PLANE_LEN],
                rng: SmallRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64)),
                done: None,
            })
            .collect();
        Runner { cfg, slots, queue: VecDeque::new(), finished: vec![], next_id: 0, rows: vec![] }
    }

    pub fn add_game(&mut self, variant: Variant, white: Player, black: Player, start: Option<engine::Position>, max_moves: Option<u32>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(GameSpec { id, variant, players: [white, black], start, max_moves });
        id
    }

    pub fn active(&self) -> usize {
        self.slots.iter().filter(|s| s.spec.is_some()).count()
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    pub fn done(&self) -> bool {
        self.queue.is_empty() && self.active() == 0
    }

    pub fn take_finished(&mut self) -> Vec<Finished> {
        std::mem::take(&mut self.finished)
    }

    fn refill(&mut self) -> bool {
        let mut started = false;
        let max_plies = self.cfg.max_plies;
        for s in self.slots.iter_mut() {
            if let Some(status) = s.done.take() {
                let spec = s.spec.take().unwrap();
                self.finished.push(finish(spec, status, s));
            }
            if s.spec.is_none() {
                if let Some(spec) = self.queue.pop_front() {
                    s.state = match spec.start {
                        Some(p) => State::new(p, max_plies),
                        None => State::start(spec.variant, max_plies),
                    };
                    s.search = None;
                    s.samples.clear();
                    s.moves.clear();
                    s.spec = Some(spec);
                    started = true;
                }
            }
        }
        started
    }

    /// Advances every game until it needs the network. Returns the leaves'
    /// planes (rows of 31*64 bytes) and which network each row is for.
    pub fn step(&mut self) -> (Vec<u8>, Vec<u8>) {
        let cfg = self.cfg;
        loop {
            self.refill();
            self.slots.par_iter_mut().for_each(|s| advance(s, &cfg));
            // Finished games are collected (and replaced) by the next refill.
            if !self.slots.iter().any(|s| s.done.is_some()) {
                break;
            }
        }
        // Collect leaves.
        self.rows.clear();
        let mut planes = Vec::new();
        let mut nets = Vec::new();
        for (i, s) in self.slots.iter().enumerate() {
            if s.search.as_ref().is_some_and(|x| x.has_pending()) {
                self.rows.push(i);
                planes.extend_from_slice(&s.planes);
                let side = s.state.side().idx();
                match s.spec.as_ref().unwrap().players[side] {
                    Player::Net(n) => nets.push(n),
                    _ => unreachable!("only network players search"),
                }
            }
        }
        (planes, nets)
    }

    /// Network outputs for the rows of the last step(): logits [rows * 4992], values [rows].
    pub fn feed(&mut self, logits: &[f32], values: &[f32]) {
        let n = crate::encode::NUM_ACTIONS;
        assert_eq!(values.len(), self.rows.len(), "feed: wrong number of values");
        assert_eq!(logits.len(), self.rows.len() * n, "feed: wrong logits size");
        let mut row_of = vec![usize::MAX; self.slots.len()];
        for (r, &i) in self.rows.iter().enumerate() {
            row_of[i] = r;
        }
        let cfg = self.cfg;
        self.slots.par_iter_mut().zip(row_of.par_iter()).for_each(|(s, &r)| {
            if r != usize::MAX {
                let search = s.search.as_mut().unwrap();
                search.feed(&logits[r * n..(r + 1) * n], values[r], &cfg.mcts, &mut s.rng);
            }
        });
        self.rows.clear();
    }
}

/// Plays moves in one slot until it needs a network evaluation or the game ends.
fn advance(s: &mut Slot, cfg: &RunCfg) {
    let Some(spec) = s.spec.as_ref() else { return };
    if s.done.is_some() {
        return;
    }
    let players = spec.players;
    let max_moves = spec.max_moves;
    loop {
        if s.search.is_none() {
            if max_moves.is_some_and(|n| s.moves.len() as u32 >= n) {
                s.done = Some(Status::Ongoing); // stopped early, reported as "ongoing"
                return;
            }
            let legal = s.state.legal();
            let st = s.state.status_with(&legal);
            if st != Status::Ongoing {
                s.done = Some(st);
                return;
            }
            let player = players[s.state.side().idx()];
            let mv = match player {
                Player::Random => Some(legal[s.rng.gen_range(0..legal.len())]),
                Player::Greedy => Some(greedy(&s.state, &legal, &mut s.rng)),
                Player::Net(_) if legal.len() == 1 => Some(legal[0]),
                Player::Net(_) => None,
            };
            if let Some(m) = mv {
                play(s, m);
                continue;
            }
            // A network player always takes a mate in one (and learns it as a policy target).
            if cfg.mcts.mate_check {
                if let Some(m) = s.state.mate_in_one(&legal) {
                    if cfg.record {
                        s.samples.push(one_hot_sample(&s.state, &legal, m));
                    }
                    play(s, m);
                    continue;
                }
            }
            let full = cfg.p_full >= 1.0 || s.rng.gen::<f32>() < cfg.p_full;
            let sims = if full { cfg.sims_full } else { cfg.sims_fast };
            s.search = Some(Search::new(sims, full, cfg.noise && full));
        }
        let search = s.search.as_mut().unwrap();
        if search.has_pending() {
            return;
        }
        match search.step(&s.state, &cfg.mcts, &mut s.planes) {
            StepOut::NeedEval => return,
            StepOut::Ready => {
                let search = s.search.take().unwrap();
                if cfg.record && search.full {
                    s.samples.push(make_sample(&s.state, &search, cfg.topk));
                }
                let sample = s.state.plies < cfg.temp_plies;
                let m = search.choose(sample, &mut s.rng);
                play(s, m);
            }
        }
    }
}

fn play(s: &mut Slot, m: Move) {
    s.moves.push(m.uci());
    s.state.apply(m);
}

fn one_hot_sample(state: &State, legal: &[Move], m: Move) -> Sample {
    let mut planes = vec![0u8; PLANE_LEN];
    state.encode_raw(&mut planes);
    let actions: Vec<u16> = legal.iter().map(|&x| state.action(x) as u16).collect();
    let mut mask = vec![0u8; MASK_BYTES];
    pack_mask(&actions, &mut mask);
    Sample { planes, mask, pol_idx: vec![state.action(m) as u16], pol_p: vec![1.0], side: state.side(), q: 1.0 }
}

fn make_sample(state: &State, search: &Search, topk: usize) -> Sample {
    let mut planes = vec![0u8; PLANE_LEN];
    state.encode_raw(&mut planes);
    let mut visits = search.root_visits();
    let actions: Vec<u16> = visits.iter().map(|v| v.1).collect();
    let mut mask = vec![0u8; MASK_BYTES];
    pack_mask(&actions, &mut mask);
    visits.retain(|v| v.2 > 0);
    visits.sort_by(|a, b| b.2.cmp(&a.2));
    visits.truncate(topk);
    let total: u32 = visits.iter().map(|v| v.2).sum::<u32>().max(1);
    Sample {
        planes,
        mask,
        pol_idx: visits.iter().map(|v| v.1).collect(),
        pol_p: visits.iter().map(|v| v.2 as f32 / total as f32).collect(),
        side: state.side(),
        q: search.root_value(),
    }
}

fn finish(spec: GameSpec, status: Status, s: &mut Slot) -> Finished {
    let mover = s.state.side();
    let result: i8 = match status {
        Status::Checkmate => {
            if mover == Color::White {
                -1
            } else {
                1
            }
        }
        _ => 0,
    };
    let samples = std::mem::take(&mut s.samples);
    let z = samples
        .iter()
        .map(|smp| match status {
            Status::Checkmate => {
                if smp.side == mover {
                    -1.0
                } else {
                    1.0
                }
            }
            Status::MaxPlies => adjudicated_for(&s.state, smp.side),
            _ => 0.0,
        })
        .collect();
    Finished {
        id: spec.id,
        variant: spec.variant,
        players: spec.players,
        status,
        result,
        plies: s.state.plies,
        samples,
        z,
        moves: std::mem::take(&mut s.moves),
    }
}

/// Material-based score of a capped game for `side`. The trainer decides
/// whether to use it (early generations) or treat the game as a 0 draw.
fn adjudicated_for(state: &State, side: Color) -> f32 {
    adjudicated(state.material_diff(side))
}

/// Mate in one if available, else capture the most valuable piece, else random.
pub fn greedy(state: &State, legal: &[Move], rng: &mut SmallRng) -> Move {
    let pos = &state.pos;
    if let Some(m) = state.mate_in_one(legal) {
        return m;
    }
    let value = |m: Move| -> i32 {
        match m {
            Move::Normal { from, to, .. } => match pos.board[to as usize] {
                Some(p) => piece_value(p.kind),
                None => {
                    let pawn = matches!(pos.board[from as usize], Some(p) if p.kind == engine::Kind::Pawn);
                    if pawn && pos.ep == Some(to) {
                        1
                    } else {
                        0
                    }
                }
            },
            Move::Drop { .. } => 0,
        }
    };
    let best = legal.iter().map(|&m| value(m)).max().unwrap_or(0);
    if best > 0 {
        let caps: Vec<Move> = legal.iter().copied().filter(|&m| value(m) == best).collect();
        return caps[rng.gen_range(0..caps.len())];
    }
    legal[rng.gen_range(0..legal.len())]
}

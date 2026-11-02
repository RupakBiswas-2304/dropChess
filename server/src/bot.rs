//! The Drop Chess bot: the self-play-trained network (ONNX, run with tract)
//! plus the same PUCT search used in training (`pyengine/src/mcts.rs`).
//!
//! The state encoding and action indices must match `pyengine/src/encode.rs`
//! and `train/dropchess_py/encoding.py` exactly; the tests at the bottom check
//! them against `train/fixtures/encoding.jsonl`.

use dropchess::{Color, Game, Kind, Move, Position, Sq, Variant};
use rand::Rng;
use std::path::Path;
use std::time::{Duration, Instant};
use tract_onnx::prelude::*;

pub const NUM_PLANES: usize = 31;
#[allow(dead_code)]
pub const NUM_ACTIONS: usize = 4992;
const DROP_BASE: usize = 4672;

/// (file delta, rank delta): N, NE, E, SE, S, SW, W, NW.
const DIRS: [(i8, i8); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
const KNIGHT: [(i8, i8); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

fn kind_idx(k: Kind) -> usize {
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

/// Action index of a move made by `side`.
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
            let under = match promo {
                Some(Kind::Knight) => Some(0),
                Some(Kind::Bishop) => Some(1),
                Some(Kind::Rook) => Some(2),
                _ => None,
            };
            if let Some(pi) = under {
                return base + 64 + ((df + 1) as usize) * 3 + pi;
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

/// Network input planes, float32 [31 * 64], already scaled.
/// `seen_before` = how many earlier positions in the game equal this one.
pub fn encode(pos: &Position, seen_before: usize) -> Vec<f32> {
    let mut out = vec![0f32; NUM_PLANES * 64];
    let us = pos.side;
    let black = us == Color::Black;
    for sq in 0..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            let s = flip(sq, black);
            let base = if p.color == us { 0 } else { 6 };
            out[(base + kind_idx(p.kind)) * 64 + s] = 1.0;
            if p.promoted {
                out[(if p.color == us { 22 } else { 23 }) * 64 + s] = 1.0;
            }
        }
    }
    let fill = |out: &mut [f32], plane: usize, v: f32| out[plane * 64..(plane + 1) * 64].fill(v);
    for i in 0..5 {
        fill(&mut out, 12 + i, pos.reserve[us.idx()][i] as f32 * (1.0f32 / 8.0));
        fill(&mut out, 17 + i, pos.reserve[us.opp().idx()][i] as f32 * (1.0f32 / 8.0));
    }
    if seen_before >= 1 {
        fill(&mut out, 24, 1.0);
    }
    if seen_before >= 2 {
        fill(&mut out, 25, 1.0);
    }
    if let Some(ep) = pos.ep {
        out[26 * 64 + flip(ep, black)] = 1.0;
    }
    if !black {
        fill(&mut out, 27, 1.0);
    }
    if pos.variant == Variant::Crazy {
        fill(&mut out, 28, 1.0);
    }
    // Same arithmetic as training: raw integer x float32(1/100), not a division.
    fill(&mut out, 29, pos.halfmove.min(255) as f32 * (1.0f32 / 100.0));
    fill(&mut out, 30, 1.0);
    out
}

type Plan = TypedRunnableModel<TypedModel>;

/// The policy/value network.
pub struct Net {
    plan: Plan,
}

impl Net {
    pub fn load(path: &Path) -> TractResult<Net> {
        let plan = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, NUM_PLANES, 8, 8]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Net { plan })
    }

    /// Returns (raw policy logits over all 4,992 actions, value for the side to move).
    pub fn eval(&self, planes: Vec<f32>) -> TractResult<(Vec<f32>, f32)> {
        let x = Tensor::from_shape(&[1, NUM_PLANES, 8, 8], &planes)?;
        let out = self.plan.run(tvec!(x.into()))?;
        let policy = out[0].as_slice::<f32>()?.to_vec();
        let value = out[1].as_slice::<f32>()?[0];
        Ok((policy, value))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// No search: samples a move from the network's policy.
    Easy,
    /// 64 simulations (the setting used for evaluation during training).
    Medium,
    /// 400 simulations.
    Hard,
}

impl Level {
    pub fn parse(s: &str) -> Option<Level> {
        match s {
            "easy" => Some(Level::Easy),
            "medium" => Some(Level::Medium),
            "hard" => Some(Level::Hard),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Level::Easy => "Easy",
            Level::Medium => "Medium",
            Level::Hard => "Hard",
        }
    }
    fn sims(self) -> u32 {
        match self {
            Level::Easy => 1,
            Level::Medium => 64,
            Level::Hard => 400,
        }
    }
    fn time_cap(self) -> Duration {
        match self {
            Level::Easy => Duration::from_millis(200),
            Level::Medium => Duration::from_millis(1500),
            Level::Hard => Duration::from_millis(4000),
        }
    }
}

const C_PUCT: f32 = 2.0;
const FPU_REDUCTION: f32 = 0.2;
const FPU_ROOT: f32 = 0.0;

/// A position plus the repetition history needed for draws and the repetition planes.
#[derive(Clone)]
struct State {
    pos: Position,
    history: Vec<String>,
}

#[derive(PartialEq)]
enum Outcome {
    Ongoing,
    /// The side to move is checkmated.
    Lost,
    Draw,
}

impl State {
    fn from_game(game: &Game) -> State {
        State { pos: game.pos, history: game.history().to_vec() }
    }
    fn apply(&mut self, m: Move) {
        self.pos.make(m);
        self.history.push(self.pos.repetition_key());
    }
    fn seen_before(&self) -> usize {
        let key = self.history.last().expect("history");
        self.history.iter().filter(|k| *k == key).count() - 1
    }
    fn outcome(&self, legal: &[Move]) -> Outcome {
        if legal.is_empty() {
            return if self.pos.in_check(self.pos.side) { Outcome::Lost } else { Outcome::Draw };
        }
        if self.seen_before() >= 2 || self.pos.halfmove >= 100 || self.pos.insufficient_material() {
            return Outcome::Draw;
        }
        Outcome::Ongoing
    }
}

/// A move that checkmates at once, if there is one.
fn mate_in_one(pos: &Position, legal: &[Move]) -> Option<Move> {
    legal.iter().copied().find(|&m| {
        let mut n = *pos;
        n.make(m);
        n.in_check(n.side) && n.legal_moves().is_empty()
    })
}

#[derive(Clone, Copy)]
struct Node {
    mv: Move,
    prior: f32,
    n: u32,
    /// Sum of values from the point of view of the player who chose this edge.
    w: f32,
    first: u32,
    nchild: u16,
    expanded: bool,
    term: Option<f32>,
}

impl Node {
    fn new(mv: Move, prior: f32) -> Node {
        Node { mv, prior, n: 0, w: 0.0, first: 0, nchild: 0, expanded: false, term: None }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Choice {
    pub mv: Move,
    pub sims: u32,
    /// Value of the position for the bot, in [-1, 1].
    pub value: f32,
    pub elapsed: Duration,
}

fn softmax_over(logits: &[f32], actions: &[usize]) -> Vec<f32> {
    let maxl = actions.iter().map(|&a| logits[a]).fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = actions.iter().map(|&a| (logits[a] - maxl).exp()).collect();
    let s: f32 = p.iter().sum();
    for x in p.iter_mut() {
        *x /= s;
    }
    p
}

/// Picks the bot's move. `budget` caps thinking time (on top of the level's own cap).
pub fn choose(net: &Net, game: &Game, level: Level, budget: Option<Duration>) -> TractResult<Choice> {
    let t0 = Instant::now();
    let root = State::from_game(game);
    let legal = root.pos.legal_moves();
    if legal.is_empty() {
        anyhow::bail!("no legal moves");
    }
    let side = root.pos.side;
    let mut rng = rand::thread_rng();

    if level == Level::Easy {
        let (logits, value) = net.eval(encode(&root.pos, root.seen_before()))?;
        let actions: Vec<usize> = legal.iter().map(|&m| action_of(m, side)).collect();
        let p = softmax_over(&logits, &actions);
        let mut r: f32 = rng.gen();
        let mut pick = legal[legal.len() - 1];
        for (i, &x) in p.iter().enumerate() {
            if r < x {
                pick = legal[i];
                break;
            }
            r -= x;
        }
        return Ok(Choice { mv: pick, sims: 1, value, elapsed: t0.elapsed() });
    }

    // A network player always takes a mate in one (as in training).
    if let Some(m) = mate_in_one(&root.pos, &legal) {
        return Ok(Choice { mv: m, sims: 0, value: 1.0, elapsed: t0.elapsed() });
    }
    if legal.len() == 1 {
        return Ok(Choice { mv: legal[0], sims: 0, value: 0.0, elapsed: t0.elapsed() });
    }

    let cap = match budget {
        Some(b) => b.min(level.time_cap()),
        None => level.time_cap(),
    };
    let target = level.sims();
    let mut nodes: Vec<Node> = Vec::with_capacity(target as usize * 40);
    nodes.push(Node::new(legal[0], 1.0));
    let mut sims = 0u32;

    let backprop = |nodes: &mut Vec<Node>, path: &[u32], leaf_value: f32| {
        let mut v = leaf_value;
        for &i in path.iter().rev() {
            let n = &mut nodes[i as usize];
            n.n += 1;
            n.w -= v;
            v = -v;
        }
    };

    while sims < target && (sims < 8 || t0.elapsed() < cap) {
        let mut state = root.clone();
        let mut path = vec![0u32];
        let mut i = 0usize;
        while nodes[i].expanded && nodes[i].term.is_none() {
            let node = nodes[i];
            let parent_q = if node.n > 0 { -node.w / node.n as f32 } else { 0.0 };
            let fpu = parent_q - if i == 0 { FPU_ROOT } else { FPU_REDUCTION };
            let sqrt_n = (node.n.max(1) as f32).sqrt();
            let mut best = node.first as usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in node.first as usize..node.first as usize + node.nchild as usize {
                let ch = &nodes[c];
                let q = if ch.n > 0 { ch.w / ch.n as f32 } else { fpu };
                let s = q + C_PUCT * ch.prior * sqrt_n / (1.0 + ch.n as f32);
                if s > best_score {
                    best_score = s;
                    best = c;
                }
            }
            i = best;
            state.apply(nodes[i].mv);
            path.push(i as u32);
        }
        if let Some(v) = nodes[i].term {
            backprop(&mut nodes, &path, v);
            sims += 1;
            continue;
        }
        let leaf_legal = state.pos.legal_moves();
        let outcome = state.outcome(&leaf_legal);
        if outcome != Outcome::Ongoing {
            let v = if outcome == Outcome::Lost { -1.0 } else { 0.0 };
            nodes[i].term = Some(v);
            backprop(&mut nodes, &path, v);
            sims += 1;
            continue;
        }
        if i != 0 && mate_in_one(&state.pos, &leaf_legal).is_some() {
            nodes[i].term = Some(1.0);
            backprop(&mut nodes, &path, 1.0);
            sims += 1;
            continue;
        }
        let (logits, value) = net.eval(encode(&state.pos, state.seen_before()))?;
        let leaf_side = state.pos.side;
        let actions: Vec<usize> = leaf_legal.iter().map(|&m| action_of(m, leaf_side)).collect();
        let priors = softmax_over(&logits, &actions);
        let first = nodes.len() as u32;
        // Shuffled, so ties between equal priors don't always go to the first generated move.
        let mut order: Vec<usize> = (0..leaf_legal.len()).collect();
        for k in (1..order.len()).rev() {
            order.swap(k, rng.gen_range(0..=k));
        }
        for k in order {
            nodes.push(Node::new(leaf_legal[k], priors[k]));
        }
        nodes[i].first = first;
        nodes[i].nchild = leaf_legal.len() as u16;
        nodes[i].expanded = true;
        backprop(&mut nodes, &path, value);
        sims += 1;
    }

    let r = nodes[0];
    let best = (r.first as usize..r.first as usize + r.nchild as usize)
        .max_by(|&a, &b| {
            let (na, nb) = (&nodes[a], &nodes[b]);
            na.n.cmp(&nb.n).then(na.prior.partial_cmp(&nb.prior).unwrap_or(std::cmp::Ordering::Equal))
        })
        .expect("root has children");
    let value = if r.n > 0 { -r.w / r.n as f32 } else { 0.0 };
    Ok(Choice { mv: nodes[best].mv, sims, value, elapsed: t0.elapsed() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../train/fixtures/encoding.jsonl");
    const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/models/dropchess.onnx");

    fn replay(d: &serde_json::Value) -> Game {
        let variant = Variant::parse(d["variant"].as_str().unwrap()).unwrap();
        let mut g = Game::from_fen(d["start"].as_str().unwrap(), variant).unwrap();
        for m in d["moves"].as_array().unwrap() {
            g.play_uci(m.as_str().unwrap()).unwrap();
        }
        g
    }

    /// Planes (as float32 little-endian bytes) and action indices match the Python reference.
    #[test]
    fn encoding_matches_training_fixtures() {
        let Ok(text) = std::fs::read_to_string(FIXTURES) else {
            eprintln!("skipping: no {FIXTURES}");
            return;
        };
        let mut checked = 0;
        for line in text.lines() {
            let d: serde_json::Value = serde_json::from_str(line).unwrap();
            let g = replay(&d);
            assert_eq!(g.pos.fen(), d["fen"].as_str().unwrap());
            let state = State::from_game(&g);
            let planes = encode(&g.pos, state.seen_before());
            let bytes: Vec<u8> = planes.iter().flat_map(|x| x.to_le_bytes()).collect();
            let hash = format!("{:x}", Sha256::digest(&bytes));
            assert_eq!(hash, d["planes_sha256"].as_str().unwrap(), "planes differ at {}", d["fen"]);
            let mut actions: Vec<u64> = g.pos.legal_moves().iter().map(|&m| action_of(m, g.pos.side) as u64).collect();
            actions.sort();
            let want: Vec<u64> = d["actions"].as_array().unwrap().iter().map(|a| a.as_u64().unwrap()).collect();
            assert_eq!(actions, want, "actions differ at {}", d["fen"]);
            checked += 1;
        }
        assert!(checked > 100);
    }

    #[test]
    fn bot_plays_legal_moves_and_takes_mate_in_one() {
        let Ok(net) = Net::load(Path::new(MODEL)) else {
            eprintln!("skipping: no model at {MODEL}");
            return;
        };
        // Drop-mate available: R@a3#.
        let g = Game::from_fen("8/8/8/8/8/8/2K5/k7[R] w - - 0 1", Variant::Drop).unwrap();
        for level in [Level::Medium, Level::Hard] {
            assert_eq!(choose(&net, &g, level, None).unwrap().mv.uci().len(), 4);
            let c = choose(&net, &g, level, None).unwrap();
            let mut g2 = g.clone();
            g2.play(c.mv).unwrap();
            assert!(matches!(g2.status(), dropchess::Status::Checkmate { .. }), "{level:?} missed mate: {}", c.mv);
        }
        for variant in [Variant::Drop, Variant::Crazy] {
            let mut g = Game::new(variant);
            for _ in 0..20 {
                if g.status().is_over() {
                    break;
                }
                let c = choose(&net, &g, Level::Medium, None).unwrap();
                g.play(c.mv).unwrap();
            }
        }
    }

    /// Writes planes and outputs for a few fixtures, for comparison with onnxruntime in Python.
    #[test]
    #[ignore]
    fn dump_outputs_for_parity() {
        let net = Net::load(Path::new(MODEL)).unwrap();
        let text = std::fs::read_to_string(FIXTURES).unwrap();
        let mut rows = vec![];
        for line in text.lines().step_by(25).take(20) {
            let d: serde_json::Value = serde_json::from_str(line).unwrap();
            let g = replay(&d);
            let planes = encode(&g.pos, State::from_game(&g).seen_before());
            let (p, v) = net.eval(planes.clone()).unwrap();
            rows.push(serde_json::json!({ "planes": planes, "policy": p, "value": v }));
        }
        std::fs::write(std::env::var("PARITY_OUT").unwrap_or("/tmp/parity.json".into()), serde_json::to_string(&rows).unwrap()).unwrap();
    }

    /// Medium bot vs the training "greedy" baseline: mate in one, else the biggest capture, else random.
    #[test]
    #[ignore]
    fn sanity_vs_greedy() {
        let net = Net::load(Path::new(MODEL)).unwrap();
        let val = |k: Kind| match k { Kind::Pawn => 1, Kind::Knight | Kind::Bishop => 3, Kind::Rook => 5, Kind::Queen => 9, Kind::King => 0 };
        let mut rng = rand::thread_rng();
        for variant in [Variant::Drop, Variant::Crazy] {
            let (mut w, mut d, mut l) = (0, 0, 0);
            for game_i in 0..6 {
                let bot_white = game_i % 2 == 0;
                let mut g = Game::new(variant);
                let mut plies = 0;
                while !g.status().is_over() && plies < 300 {
                    let bots_turn = (g.pos.side == Color::White) == bot_white;
                    let m = if bots_turn {
                        choose(&net, &g, Level::Medium, None).unwrap().mv
                    } else {
                        let legal = g.pos.legal_moves();
                        mate_in_one(&g.pos, &legal).unwrap_or_else(|| {
                            let best = legal.iter().filter_map(|&m| match m {
                                Move::Normal { to, .. } => g.pos.board[to as usize].map(|p| (val(p.kind), m)),
                                _ => None,
                            }).max_by_key(|x| x.0);
                            best.map(|x| x.1).unwrap_or(legal[rng.gen_range(0..legal.len())])
                        })
                    };
                    g.play(m).unwrap();
                    plies += 1;
                }
                match g.status() {
                    dropchess::Status::Checkmate { winner } if (winner == Color::White) == bot_white => w += 1,
                    dropchess::Status::Checkmate { .. } => l += 1,
                    _ => d += 1,
                }
            }
            println!("{variant:?}: bot {w} wins, {d} draws, {l} losses vs greedy");
        }
    }
}

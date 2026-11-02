//! PUCT tree search with external (batched) leaf evaluation.
//!
//! A search alternates between `step`, which descends to a leaf and either
//! resolves it (terminal) or asks for a network evaluation, and `feed`, which
//! expands the leaf with the network's policy and backs up its value.

use crate::encode::NUM_ACTIONS;
use crate::state::{State, Status};
use engine::Move;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Gamma};

#[derive(Clone, Copy, Debug)]
pub struct MctsCfg {
    pub c_puct: f32,
    pub fpu_reduction: f32,
    /// First-play urgency reduction at the root (0 = unvisited root moves count as the root's value).
    pub fpu_root: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_frac: f32,
    pub adjudicate: bool,
    /// Score a leaf where the side to move can mate in one as a proven win.
    pub mate_check: bool,
    /// Weight of a material score blended into network leaf values (early training only).
    pub material_blend: f32,
}

#[derive(Clone, Copy)]
struct Node {
    mv: Move,
    action: u16,
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
    fn new(mv: Move, action: u16, prior: f32) -> Node {
        Node { mv, action, prior, n: 0, w: 0.0, first: 0, nchild: 0, expanded: false, term: None }
    }
}

struct Pending {
    path: Vec<u32>,
    /// Material score of the leaf for its side to move, in [-1, 1].
    material: f32,
    legal: Vec<Move>,
    actions: Vec<u16>,
}

pub enum StepOut {
    /// A leaf needs the network; its planes were written to the output buffer.
    NeedEval,
    /// The simulation budget is spent; pick a move.
    Ready,
}

pub struct Search {
    nodes: Vec<Node>,
    pub sims_target: u32,
    pub sims_done: u32,
    pub full: bool,
    noise: bool,
    pending: Option<Pending>,
}

impl Search {
    pub fn new(sims: u32, full: bool, noise: bool) -> Search {
        let dummy = Move::Drop { kind: engine::Kind::Pawn, to: 0 };
        let mut nodes = Vec::with_capacity(sims as usize * 40);
        nodes.push(Node::new(dummy, 0, 1.0));
        Search { nodes, sims_target: sims.max(1), sims_done: 0, full, noise, pending: None }
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    fn select_child(&self, i: usize, c_puct: f32, fpu_reduction: f32) -> usize {
        let node = &self.nodes[i];
        let parent_q = if node.n > 0 { -node.w / node.n as f32 } else { 0.0 };
        let fpu = parent_q - fpu_reduction;
        let sqrt_n = (node.n.max(1) as f32).sqrt();
        let first = node.first as usize;
        let mut best = first;
        let mut best_score = f32::NEG_INFINITY;
        for c in first..first + node.nchild as usize {
            let ch = &self.nodes[c];
            let q = if ch.n > 0 { ch.w / ch.n as f32 } else { fpu };
            let s = q + c_puct * ch.prior * sqrt_n / (1.0 + ch.n as f32);
            if s > best_score {
                best_score = s;
                best = c;
            }
        }
        best
    }

    fn backprop(&mut self, path: &[u32], leaf_value: f32) {
        // leaf_value is for the side to move at the leaf.
        let mut v = leaf_value;
        for &i in path.iter().rev() {
            let n = &mut self.nodes[i as usize];
            n.n += 1;
            n.w -= v;
            v = -v;
        }
    }

    /// Runs simulations until one needs the network or the budget is spent.
    pub fn step(&mut self, root: &State, cfg: &MctsCfg, planes_out: &mut [u8]) -> StepOut {
        debug_assert!(self.pending.is_none());
        loop {
            if self.sims_done >= self.sims_target && self.nodes[0].expanded {
                return StepOut::Ready;
            }
            let mut state = root.clone();
            let mut path = vec![0u32];
            let mut i = 0usize;
            while self.nodes[i].expanded && self.nodes[i].term.is_none() {
                let fpu = if i == 0 { cfg.fpu_root } else { cfg.fpu_reduction };
                i = self.select_child(i, cfg.c_puct, fpu);
                state.apply(self.nodes[i].mv);
                path.push(i as u32);
            }
            if let Some(v) = self.nodes[i].term {
                self.backprop(&path, v);
                self.sims_done += 1;
                continue;
            }
            let legal = state.legal();
            let st = state.status_with(&legal);
            if st != Status::Ongoing {
                let v = state.terminal_value(st, cfg.adjudicate);
                self.nodes[i].term = Some(v);
                self.backprop(&path, v);
                self.sims_done += 1;
                continue;
            }
            if cfg.mate_check && i != 0 && state.mate_in_one(&legal).is_some() {
                self.nodes[i].term = Some(1.0);
                self.backprop(&path, 1.0);
                self.sims_done += 1;
                continue;
            }
            state.encode_raw(planes_out);
            let actions = legal.iter().map(|&m| state.action(m) as u16).collect();
            let material = if cfg.material_blend > 0.0 { material_score(&state, &legal) } else { 0.0 };
            self.pending = Some(Pending { path, material, legal, actions });
            return StepOut::NeedEval;
        }
    }

    /// Expands the pending leaf with `logits` (all 4,992) and backs up `value`
    /// (for the side to move at the leaf).
    pub fn feed(&mut self, logits: &[f32], value: f32, cfg: &MctsCfg, rng: &mut SmallRng) {
        debug_assert_eq!(logits.len(), NUM_ACTIONS);
        let p = self.pending.take().expect("feed without a pending leaf");
        let leaf = *p.path.last().unwrap() as usize;
        let mut maxl = f32::NEG_INFINITY;
        for &a in &p.actions {
            maxl = maxl.max(logits[a as usize]);
        }
        let mut priors: Vec<f32> = p.actions.iter().map(|&a| (logits[a as usize] - maxl).exp()).collect();
        let sum: f32 = priors.iter().sum();
        for x in priors.iter_mut() {
            *x /= sum;
        }
        if leaf == 0 && self.noise && priors.len() > 1 {
            let g = Gamma::new(cfg.dirichlet_alpha as f64, 1.0).unwrap();
            let noise: Vec<f64> = (0..priors.len()).map(|_| g.sample(rng).max(1e-12)).collect();
            let ns: f64 = noise.iter().sum();
            let e = cfg.dirichlet_frac;
            for (x, n) in priors.iter_mut().zip(noise) {
                *x = (1.0 - e) * *x + e * (n / ns) as f32;
            }
        }
        let first = self.nodes.len() as u32;
        // Shuffled, so ties between equal priors don't always go to the first generated move.
        let mut order: Vec<usize> = (0..p.legal.len()).collect();
        order.shuffle(rng);
        for k in order {
            self.nodes.push(Node::new(p.legal[k], p.actions[k], priors[k]));
        }
        let n = &mut self.nodes[leaf];
        n.first = first;
        n.nchild = p.legal.len() as u16;
        n.expanded = true;
        let b = cfg.material_blend;
        let v = if b > 0.0 { (1.0 - b) * value + b * p.material } else { value };
        self.backprop(&p.path, v);
        self.sims_done += 1;
    }

    /// Root visit counts: (move, action, visits).
    pub fn root_visits(&self) -> Vec<(Move, u16, u32)> {
        let r = &self.nodes[0];
        (r.first as usize..r.first as usize + r.nchild as usize)
            .map(|c| (self.nodes[c].mv, self.nodes[c].action, self.nodes[c].n))
            .collect()
    }

    /// Mean value of the root for the side to move.
    pub fn root_value(&self) -> f32 {
        let r = &self.nodes[0];
        if r.n == 0 {
            0.0
        } else {
            -r.w / r.n as f32
        }
    }

    /// Picks a move: proportional to visits if `sample`, else the most visited.
    pub fn choose(&self, sample: bool, rng: &mut SmallRng) -> Move {
        let v = self.root_visits();
        if sample {
            let total: u32 = v.iter().map(|x| x.2).sum();
            if total > 0 {
                let mut r = rng.gen_range(0..total);
                for &(m, _, n) in &v {
                    if r < n {
                        return m;
                    }
                    r -= n;
                }
            }
        }
        let r = &self.nodes[0];
        let first = r.first as usize;
        let best = (first..first + r.nchild as usize)
            .max_by(|&a, &b| {
                let (na, nb) = (&self.nodes[a], &self.nodes[b]);
                na.n.cmp(&nb.n).then(na.prior.partial_cmp(&nb.prior).unwrap_or(std::cmp::Ordering::Equal))
            })
            .unwrap();
        self.nodes[best].mv
    }
}


/// Material score for the side to move, in [-1, 1]: board + reserve material
/// (P=1 N=B=3 R=5 Q=9) plus the best capture available right now, minus the
/// capturing piece if the square is defended. In Crazy Drop a capture also adds
/// to the capturer's reserve, so swings count double.
pub fn material_score(state: &State, legal: &[Move]) -> f32 {
    use crate::state::piece_value;
    let pos = &state.pos;
    let us = pos.side;
    let mult = if pos.variant == engine::Variant::Crazy { 2 } else { 1 };
    let mut best = 0;
    for &m in legal {
        if let Move::Normal { from, to, promo } = m {
            let Some(victim) = pos.board[to as usize] else { continue };
            let attacker = pos.board[from as usize].unwrap();
            let mut n = *pos;
            n.make(m);
            let moved = promo.unwrap_or(attacker.kind);
            let risk = if n.attacked(to, us.opp()) { piece_value(moved) } else { 0 };
            let promo_gain = promo.map(|k| piece_value(k) - 1).unwrap_or(0);
            best = best.max(mult * (piece_value(victim.kind) - risk) + promo_gain);
        }
    }
    ((state.material_diff(us) + best) as f32 / 10.0).tanh()
}

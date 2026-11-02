//! Python module `dropchess`: the Drop Chess rules engine, state encoding and
//! batched MCTS, for the trainer in train/. Build with
//! `maturin develop --release -m pyengine/Cargo.toml` from the repo root.

mod encode;
mod mcts;
mod runner;
mod state;

use encode::{action_of, MASK_BYTES, NUM_ACTIONS, NUM_PLANES, PLANE_LEN};
use engine::{Color, Move, Position, Variant};
use mcts::MctsCfg;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use runner::{Player, RunCfg, Runner};
use state::{State, Status};

const DEFAULT_MAX_PLIES: u32 = 300;

fn parse_variant(s: &str) -> PyResult<Variant> {
    Variant::parse(s).ok_or_else(|| PyValueError::new_err("variant must be 'drop' or 'crazy'"))
}

fn variant_name(v: Variant) -> &'static str {
    match v {
        Variant::Drop => "drop",
        Variant::Crazy => "crazy",
    }
}

fn player_name(p: Player) -> String {
    match p {
        Player::Net(n) => format!("net{n}"),
        Player::Random => "random".into(),
        Player::Greedy => "greedy".into(),
    }
}

/// A game with history, implementing the trainer's Game interface.
#[pyclass(name = "Game", module = "dropchess")]
#[derive(Clone)]
struct PyGame {
    st: State,
}

impl PyGame {
    fn find(&self, action: usize) -> PyResult<Move> {
        let side = self.st.side();
        self.st
            .legal()
            .into_iter()
            .find(|&m| action_of(m, side) == action)
            .ok_or_else(|| PyValueError::new_err(format!("illegal action {action}")))
    }
}

#[pymethods]
impl PyGame {
    #[staticmethod]
    fn new(variant: &str) -> PyResult<Self> {
        Ok(PyGame { st: State::start(parse_variant(variant)?, DEFAULT_MAX_PLIES) })
    }

    #[staticmethod]
    fn from_fen(fen: &str, variant: &str) -> PyResult<Self> {
        let pos = Position::from_fen(fen, parse_variant(variant)?).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyGame { st: State::new(pos, DEFAULT_MAX_PLIES) })
    }

    fn fen(&self) -> String {
        self.st.pos.fen()
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    #[getter]
    fn variant(&self) -> &'static str {
        variant_name(self.st.pos.variant)
    }

    fn to_play(&self) -> u8 {
        self.st.side().idx() as u8
    }

    #[getter]
    fn plies(&self) -> u32 {
        self.st.plies
    }

    #[getter]
    fn get_max_plies(&self) -> u32 {
        self.st.max_plies
    }

    #[setter]
    fn set_max_plies(&mut self, v: u32) {
        self.st.max_plies = v;
    }

    fn legal_actions(&self) -> Vec<usize> {
        let side = self.st.side();
        let mut v: Vec<usize> = self.st.legal().into_iter().map(|m| action_of(m, side)).collect();
        v.sort_unstable();
        v
    }

    fn legal_uci(&self) -> Vec<String> {
        let mut v: Vec<String> = self.st.legal().iter().map(|m| m.uci()).collect();
        v.sort();
        v
    }

    fn apply(&mut self, action: usize) -> PyResult<()> {
        let m = self.find(action)?;
        self.st.apply(m);
        Ok(())
    }

    fn apply_uci(&mut self, uci: &str) -> PyResult<()> {
        let a = self.uci_to_action(uci)?;
        self.apply(a)
    }

    fn status(&self) -> &'static str {
        self.st.status().name()
    }

    fn is_terminal(&self) -> bool {
        self.st.status() != Status::Ongoing
    }

    /// +1 White won, -1 Black won, 0 draw or not finished.
    fn result(&self) -> i8 {
        match self.st.status() {
            Status::Checkmate => {
                if self.st.side() == Color::White {
                    -1
                } else {
                    1
                }
            }
            _ => 0,
        }
    }

    /// float32 [31, 8, 8] network input.
    fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut raw = vec![0u8; PLANE_LEN];
        self.st.encode_raw(&mut raw);
        let scale = encode::plane_scale();
        let v: Vec<f32> = raw.iter().enumerate().map(|(i, &x)| x as f32 * scale[i / 64]).collect();
        v.into_pyarray(py).call_method1("reshape", ((NUM_PLANES, 8, 8),))
    }

    /// uint8 [31, 8, 8]: integer planes (reserve counts, halfmove clock); encode() = raw * plane_scale().
    fn encode_raw<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut raw = vec![0u8; PLANE_LEN];
        self.st.encode_raw(&mut raw);
        raw.into_pyarray(py).call_method1("reshape", ((NUM_PLANES, 8, 8),))
    }

    fn action_to_uci(&self, action: usize) -> PyResult<String> {
        Ok(self.find(action)?.uci())
    }

    fn uci_to_action(&self, uci: &str) -> PyResult<usize> {
        let m = Move::parse_uci(uci).ok_or_else(|| PyValueError::new_err(format!("unreadable move {uci}")))?;
        Ok(action_of(m, self.st.side()))
    }

    fn __repr__(&self) -> String {
        format!("dropchess.Game('{}', '{}')", self.st.pos.fen(), self.variant())
    }
}

fn get<'py, T: for<'a> FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<T>().map_err(|_| PyValueError::new_err(format!("config {key} has the wrong type"))),
        _ => Ok(default),
    }
}

/// Plays many games in parallel with batched network evaluation.
#[pyclass(name = "Runner", module = "dropchess")]
struct PyRunner {
    r: Runner,
}

#[pymethods]
impl PyRunner {
    /// cfg keys (all optional): sims, sims_fast, p_full, c_puct, fpu_reduction, fpu_root,
    /// dirichlet_alpha, dirichlet_frac, noise, temp_plies, max_plies, adjudicate,
    /// record, topk, mate_check (default true), material_blend (default 0).
    #[new]
    #[pyo3(signature = (cfg, parallel, seed = 0))]
    fn new(cfg: &Bound<'_, PyDict>, parallel: usize, seed: u64) -> PyResult<Self> {
        let sims: u32 = get(cfg, "sims", 200)?;
        let c = RunCfg {
            mcts: MctsCfg {
                c_puct: get(cfg, "c_puct", 2.0)?,
                fpu_reduction: get(cfg, "fpu_reduction", 0.2)?,
                fpu_root: get(cfg, "fpu_root", 0.0)?,
                dirichlet_alpha: get(cfg, "dirichlet_alpha", 0.1)?,
                dirichlet_frac: get(cfg, "dirichlet_frac", 0.25)?,
                adjudicate: get(cfg, "adjudicate", false)?,
                mate_check: get(cfg, "mate_check", true)?,
                material_blend: get(cfg, "material_blend", 0.0)?,
            },
            sims_full: sims,
            sims_fast: get(cfg, "sims_fast", sims)?,
            p_full: get(cfg, "p_full", 1.0)?,
            noise: get(cfg, "noise", false)?,
            temp_plies: get(cfg, "temp_plies", 0)?,
            max_plies: get(cfg, "max_plies", DEFAULT_MAX_PLIES)?,
            record: get(cfg, "record", false)?,
            topk: get(cfg, "topk", 128)?,
        };
        Ok(PyRunner { r: Runner::new(c, parallel, seed) })
    }

    /// Queues a game. Players: "net0", "net1", ... (MCTS with that network), "random", "greedy".
    /// `fen` optionally sets the starting position; `max_moves` stops the game after that many plies.
    #[pyo3(signature = (variant, white, black, fen = None, max_moves = None))]
    fn add_game(&mut self, variant: &str, white: &str, black: &str, fen: Option<&str>, max_moves: Option<u32>) -> PyResult<u64> {
        let p = |s: &str| Player::parse(s).ok_or_else(|| PyValueError::new_err(format!("unknown player {s}")));
        let v = parse_variant(variant)?;
        let start = match fen {
            Some(f) => Some(Position::from_fen(f, v).map_err(|e| PyValueError::new_err(e.to_string()))?),
            None => None,
        };
        Ok(self.r.add_game(v, p(white)?, p(black)?, start, max_moves))
    }

    /// Weight of the material score blended into leaf values (0 = network only).
    fn set_material_blend(&mut self, b: f32) {
        self.r.cfg.mcts.material_blend = b;
    }

    /// Score games that hit the ply cap by material (early training) instead of as draws.
    fn set_adjudicate(&mut self, on: bool) {
        self.r.cfg.mcts.adjudicate = on;
    }

    fn done(&self) -> bool {
        self.r.done()
    }

    fn active(&self) -> usize {
        self.r.active()
    }

    fn queued(&self) -> usize {
        self.r.queued()
    }

    /// Advances all games to their next network evaluation.
    /// Returns (planes uint8 [B, 31, 8, 8], net_ids uint8 [B]).
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyArray1<u8>>)> {
        let r = &mut self.r;
        let (planes, nets) = py.detach(|| r.step());
        let b = nets.len();
        let planes = planes.into_pyarray(py).call_method1("reshape", ((b, NUM_PLANES, 8, 8),))?;
        Ok((planes, nets.into_pyarray(py)))
    }

    /// Network outputs for the rows of the last step(): logits float32 [B, 4992], values float32 [B]
    /// (value for the side to move).
    fn feed(&mut self, py: Python<'_>, logits: PyReadonlyArray2<'_, f32>, values: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let l: Vec<f32> = logits.as_slice().map_err(|_| PyValueError::new_err("logits must be C-contiguous"))?.to_vec();
        let v: Vec<f32> = values.as_slice().map_err(|_| PyValueError::new_err("values must be contiguous"))?.to_vec();
        if l.len() != v.len() * NUM_ACTIONS {
            return Err(PyValueError::new_err("logits must be [B, 4992] matching values [B]"));
        }
        let r = &mut self.r;
        py.detach(|| r.feed(&l, &v));
        Ok(())
    }

    /// Finished games since the last call, as dicts. Recorded samples come as
    /// flat arrays: planes uint8 [n*1984], mask uint8 [n*624] (np.packbits order),
    /// pol_idx int64 [n*topk] and pol_p float32 [n*topk] (zero-padded), z float32 [n],
    /// q float32 [n] (root search value)
    /// (capped games get a material score here; the trainer may zero it).
    fn take_finished<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let topk = self.r.cfg.topk;
        let out = PyList::empty(py);
        for f in self.r.take_finished() {
            let d = PyDict::new(py);
            d.set_item("id", f.id)?;
            d.set_item("variant", variant_name(f.variant))?;
            d.set_item("white", player_name(f.players[0]))?;
            d.set_item("black", player_name(f.players[1]))?;
            d.set_item("status", f.status.name())?;
            d.set_item("result", f.result)?;
            d.set_item("plies", f.plies)?;
            d.set_item("moves", f.moves)?;
            let n = f.samples.len();
            d.set_item("n", n)?;
            let mut planes = Vec::with_capacity(n * PLANE_LEN);
            let mut mask = Vec::with_capacity(n * MASK_BYTES);
            let mut idx = vec![0i64; n * topk];
            let mut p = vec![0f32; n * topk];
            for (i, s) in f.samples.iter().enumerate() {
                planes.extend_from_slice(&s.planes);
                mask.extend_from_slice(&s.mask);
                for (j, (&a, &q)) in s.pol_idx.iter().zip(&s.pol_p).enumerate() {
                    idx[i * topk + j] = a as i64;
                    p[i * topk + j] = q;
                }
            }
            d.set_item("planes", planes.into_pyarray(py))?;
            d.set_item("mask", mask.into_pyarray(py))?;
            d.set_item("pol_idx", idx.into_pyarray(py))?;
            d.set_item("pol_p", p.into_pyarray(py))?;
            d.set_item("z", f.z.into_pyarray(py))?;
            let q: Vec<f32> = f.samples.iter().map(|s| s.q).collect();
            d.set_item("q", q.into_pyarray(py))?;
            out.append(d)?;
        }
        Ok(out)
    }
}

/// Number of leaf nodes at `depth` plies (move-generation check).
#[pyfunction]
fn perft(fen: &str, variant: &str, depth: u32) -> PyResult<u64> {
    let pos = Position::from_fen(fen, parse_variant(variant)?).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(engine::perft(&pos, depth))
}

/// int64 [4992]: action index with files mirrored a<->h, -1 where no legal move maps.
#[pyfunction]
fn mirror_table(py: Python<'_>) -> Bound<'_, PyArray1<i64>> {
    encode::mirror_table().into_pyarray(py)
}

/// float32 [31]: per-plane factor from encode_raw() to encode().
#[pyfunction]
fn plane_scale(py: Python<'_>) -> Bound<'_, PyArray1<f32>> {
    encode::plane_scale().to_vec().into_pyarray(py)
}

#[pymodule]
fn dropchess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyRunner>()?;
    m.add_function(wrap_pyfunction!(perft, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_table, m)?)?;
    m.add_function(wrap_pyfunction!(plane_scale, m)?)?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("NUM_PLANES", NUM_PLANES)?;
    m.add("START_FEN", engine::START_FEN)?;
    Ok(())
}

//! Drop Chess server (LAN or internet).
//!
//! Serves the web client and runs games over WebSockets. The server is the
//! single source of truth: it validates every move with the rules engine and
//! sends each client the full state plus the list of legal moves.
//!
//! Run from the repo root: `cargo run --release -p dropchess-server`
//! Env: PORT (default 8080), DROPCHESS_WEB (path to the built client, default web/dist).

mod bot;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use dropchess::{Color, Game, Move, Status, Variant};
use futures_util::{SinkExt, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    net::{SocketAddr, UdpSocket},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tower_http::services::{ServeDir, ServeFile};

type Tx = mpsc::UnboundedSender<String>;

struct Seat {
    token: String,
    name: String,
    conn: Option<(u64, Tx)>,
    bot: bool,
}

/// The bot's side of a "vs Bot" room.
struct BotSeat {
    level: bot::Level,
    /// 0 = White, 1 = Black.
    seat: usize,
    thinking: bool,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
struct TimeControl {
    initial_ms: u64,
    inc_ms: u64,
}

/// Chess clock. Neither side's first move is timed; after that the side to move's clock runs.
struct Clock {
    tc: TimeControl,
    remaining: [i64; 2],
    /// When the running clock started, and whose it is.
    running: Option<(usize, Instant)>,
}

impl Clock {
    fn new(tc: TimeControl) -> Clock {
        Clock { tc, remaining: [tc.initial_ms as i64; 2], running: None }
    }
    /// Remaining time for a side right now.
    fn now_ms(&self, side: usize) -> i64 {
        match self.running {
            Some((s, since)) if s == side => self.remaining[side] - since.elapsed().as_millis() as i64,
            _ => self.remaining[side],
        }
    }
    /// Stops the running clock, charging its elapsed time. Returns false if it had already run out.
    fn stop(&mut self) -> bool {
        if let Some((s, since)) = self.running.take() {
            self.remaining[s] -= since.elapsed().as_millis() as i64;
            return self.remaining[s] > 0;
        }
        true
    }
}

struct Room {
    id: String,
    variant: Variant,
    game: Game,
    clock: Option<Clock>,
    /// seats[0] = White, seats[1] = Black. A local (same-screen) room uses one seat for both.
    seats: [Option<Seat>; 2],
    local: bool,
    bot: Option<BotSeat>,
    /// Bumped on every new game in the room, so a stale bot answer is ignored.
    game_no: u32,
    rematch: [bool; 2],
    last_activity: Instant,
}

impl Room {
    /// The seat held by a live connection.
    fn seat_of_conn(&self, conn_id: u64) -> Option<usize> {
        (0..2).find(|&i| matches!(&self.seats[i], Some(s) if matches!(s.conn, Some((c, _)) if c == conn_id)))
    }

    fn seat_of_token(&self, token: &str) -> Option<usize> {
        (0..2).find(|&i| matches!(&self.seats[i], Some(s) if s.token == token))
    }

    fn open_seat(&self) -> Option<usize> {
        if self.local || self.bot.is_some() {
            return None;
        }
        (0..2).find(|&i| self.seats[i].is_none())
    }

    fn broadcast_state(&self) {
        let state = self.state_json();
        let mut sent = Vec::new();
        for seat in self.seats.iter().flatten() {
            if let Some((cid, tx)) = &seat.conn {
                if !sent.contains(cid) {
                    let _ = tx.send(state.clone());
                    sent.push(*cid);
                }
            }
        }
    }

    fn clock_json(&self) -> Value {
        match &self.clock {
            None => Value::Null,
            Some(c) => json!({
                "white": c.now_ms(0).max(0),
                "black": c.now_ms(1).max(0),
                "running": c.running.map(|(s, _)| if s == 0 { "white" } else { "black" }),
                "initialMs": c.tc.initial_ms,
                "incMs": c.tc.inc_ms,
            }),
        }
    }

    /// Called after a legal move by `mover` (0 = White). Returns false if the mover's flag had fallen.
    fn clock_after_move(&mut self, mover: usize) -> bool {
        let moves_played = self.game.moves().len();
        let over = self.game.status().is_over();
        let Some(clock) = self.clock.as_mut() else { return true };
        if !clock.stop() {
            return false;
        }
        if moves_played > 2 {
            clock.remaining[mover] += clock.tc.inc_ms as i64;
        }
        if !over && moves_played >= 2 {
            clock.running = Some((1 - mover, Instant::now()));
        }
        true
    }

    /// Ends the game if the running clock has hit zero. Returns true if it just ended.
    fn check_flag(&mut self) -> bool {
        let Some(clock) = self.clock.as_mut() else { return false };
        let Some((side, _)) = clock.running else { return false };
        if clock.now_ms(side) > 0 {
            return false;
        }
        clock.stop();
        clock.remaining[side] = 0;
        let loser = if side == 0 { Color::White } else { Color::Black };
        self.game.force_end(Status::Timeout { winner: loser.opp() });
        true
    }

    fn reset_game(&mut self) {
        self.game = Game::new(self.variant);
        self.game_no += 1;
        if let Some(b) = self.bot.as_mut() {
            b.thinking = false;
        }
        if let Some(c) = self.clock.as_mut() {
            *c = Clock::new(c.tc);
        }
    }

    fn stop_clock(&mut self) {
        if let Some(c) = self.clock.as_mut() {
            c.stop();
        }
    }

    fn state_json(&self) -> String {
        let pos = &self.game.pos;
        let board: Vec<Value> = pos
            .board
            .iter()
            .map(|p| match p {
                None => Value::Null,
                Some(p) => json!({
                    "c": if p.color == Color::White { "w" } else { "b" },
                    "k": p.kind.letter().to_string(),
                    "promoted": p.promoted,
                }),
            })
            .collect();
        let reserve = |c: Color| {
            let r = pos.reserve[c.idx()];
            json!({ "P": r[0], "N": r[1], "B": r[2], "R": r[3], "Q": r[4] })
        };
        let status = self.game.status();
        let check = if pos.in_check(pos.side) && !matches!(status, Status::Stalemate) { pos.king_sq(pos.side) } else { None };
        let player = |i: usize| match &self.seats[i] {
            Some(s) => json!({ "name": s.name, "online": s.bot || s.conn.is_some(), "bot": s.bot }),
            None => Value::Null,
        };
        json!({
            "t": "state",
            "room": self.id,
            "variant": self.variant,
            "local": self.local,
            "fen": pos.fen(),
            "board": board,
            "side": pos.side,
            "reserve": { "white": reserve(Color::White), "black": reserve(Color::Black) },
            "legal": self.game.legal_moves().iter().map(Move::uci).collect::<Vec<_>>(),
            "status": status,
            "check": check,
            "lastMove": self.game.moves().last().map(Move::uci),
            "sans": self.game.sans(),
            "fens": self.game.fens(),
            "ucis": self.game.moves().iter().map(Move::uci).collect::<Vec<_>>(),
            "players": { "white": player(0), "black": player(1) },
            "rematch": { "white": self.rematch[0], "black": self.rematch[1] },
            "clock": self.clock_json(),
            "bot": self.bot.as_ref().map(|b| json!({
                "color": if b.seat == 0 { "white" } else { "black" },
                "level": b.level.label(),
                "thinking": b.thinking,
            })),
        })
        .to_string()
    }
}

#[derive(Default)]
struct AppState {
    rooms: Mutex<HashMap<String, Room>>,
    next_conn: Mutex<u64>,
    net: Option<Arc<bot::Net>>,
}

type Shared = Arc<AppState>;

#[derive(Deserialize)]
#[serde(tag = "t", rename_all = "lowercase")]
enum ClientMsg {
    List,
    Create {
        variant: String,
        #[serde(default)]
        color: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        local: bool,
        #[serde(default)]
        tc: Option<TimeControl>,
        /// Bot level ("easy", "medium", "hard") for a game against the bot.
        #[serde(default)]
        bot: Option<String>,
    },
    Join {
        room: String,
        #[serde(default)]
        name: Option<String>,
    },
    Rejoin {
        room: String,
        token: String,
    },
    Move {
        uci: String,
    },
    Resign,
    Rematch,
    Leave,
    Ping,
}

#[derive(Serialize)]
struct RoomSummary {
    id: String,
    variant: Variant,
    host: String,
    open: bool,
    over: bool,
    tc: Option<TimeControl>,
}

fn random_room_id(existing: &HashMap<String, Room>) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let mut rng = rand::thread_rng();
    loop {
        let id: String = (0..4).map(|_| ALPHABET[rng.gen_range(0..ALPHABET.len())] as char).collect();
        if !existing.contains_key(&id) {
            return id;
        }
    }
}

fn random_token() -> String {
    let mut rng = rand::thread_rng();
    (0..16).map(|_| format!("{:x}", rng.gen_range(0..16u8))).collect()
}

fn clean_name(name: Option<String>, fallback: &str) -> String {
    let n: String = name.unwrap_or_default().trim().chars().filter(|c| !c.is_control()).take(24).collect();
    if n.is_empty() {
        fallback.to_string()
    } else {
        n
    }
}

fn send(tx: &Tx, v: Value) {
    let _ = tx.send(v.to_string());
}

fn err(tx: &Tx, msg: &str) {
    send(tx, json!({ "t": "error", "msg": msg }));
}

fn joined(tx: &Tx, room: &Room, token: &str, seat: usize) {
    let color = if room.local {
        "both"
    } else if seat == 0 {
        "white"
    } else {
        "black"
    };
    send(tx, json!({ "t": "joined", "room": room.id, "token": token, "color": color, "variant": room.variant, "bot": room.bot.is_some() }));
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Shared>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Shared) {
    let conn_id = {
        let mut n = state.next_conn.lock().unwrap();
        *n += 1;
        *n
    };
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let writer = tokio::spawn(async move {
        while let Some(text) = rx.recv().await {
            if sink.send(Message::Text(text.into())).await.is_err() {
                break;
            }
        }
    });

    // The room this connection is in. Its seat is looked up by connection id on
    // every message, because a rematch swaps seats (issue #5).
    let mut attached: Option<String> = None;

    while let Some(Ok(msg)) = stream.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };
        let parsed: ClientMsg = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                err(&tx, &format!("bad message: {e}"));
                continue;
            }
        };
        handle_msg(&state, conn_id, &tx, &mut attached, parsed);
    }

    detach(&state, conn_id, &mut attached);
    writer.abort();
}

/// Removes this connection from its seat (the seat stays reserved for a rejoin).
fn detach(state: &Shared, conn_id: u64, attached: &mut Option<String>) {
    if let Some(room_id) = attached.take() {
        let mut rooms = state.rooms.lock().unwrap();
        if let Some(room) = rooms.get_mut(&room_id) {
            for seat in room.seats.iter_mut().flatten() {
                if matches!(seat.conn, Some((cid, _)) if cid == conn_id) {
                    seat.conn = None;
                }
            }
            room.last_activity = Instant::now();
            room.broadcast_state();
        }
    }
}

fn handle_msg(state: &Shared, conn_id: u64, tx: &Tx, attached: &mut Option<String>, msg: ClientMsg) {
    match msg {
        ClientMsg::List => {
            let rooms = state.rooms.lock().unwrap();
            let mut list: Vec<RoomSummary> = rooms
                .values()
                .filter(|r| !r.local && r.bot.is_none())
                .map(|r| RoomSummary {
                    id: r.id.clone(),
                    variant: r.variant,
                    host: r.seats.iter().flatten().next().map(|s| s.name.clone()).unwrap_or_default(),
                    open: r.open_seat().is_some(),
                    over: r.game.status().is_over(),
                    tc: r.clock.as_ref().map(|c| c.tc),
                })
                .collect();
            list.sort_by(|a, b| b.open.cmp(&a.open).then(a.id.cmp(&b.id)));
            send(tx, json!({ "t": "rooms", "rooms": list }));
        }
        ClientMsg::Create { variant, color, name, local, tc, bot } => {
            let Some(variant) = Variant::parse(&variant) else { return err(tx, "unknown variant") };
            let bot_level = match bot.as_deref() {
                None => None,
                Some(l) => match (bot::Level::parse(l), &state.net) {
                    (Some(level), Some(_)) => Some(level),
                    (None, _) => return err(tx, "unknown bot level"),
                    (_, None) => return err(tx, "the bot isn't available on this server (no model file)"),
                },
            };
            let local = local && bot_level.is_none();
            if let Some(t) = tc {
                if !(1_000..=3 * 3600 * 1000).contains(&t.initial_ms) || t.inc_ms > 180_000 {
                    return err(tx, "time control out of range");
                }
            }
            detach(state, conn_id, attached);
            let mut rooms = state.rooms.lock().unwrap();
            let id = random_room_id(&rooms);
            let token = random_token();
            let seat_idx = match color.as_deref() {
                Some("black") => 1,
                Some("white") => 0,
                _ => rand::thread_rng().gen_range(0..2),
            };
            let seat_idx = if local { 0 } else { seat_idx };
            let mut seats: [Option<Seat>; 2] = [None, None];
            seats[seat_idx] =
                Some(Seat { token: token.clone(), name: clean_name(name, "Player 1"), conn: Some((conn_id, tx.clone())), bot: false });
            if let Some(level) = bot_level {
                seats[1 - seat_idx] =
                    Some(Seat { token: random_token(), name: format!("Bot · {}", level.label()), conn: None, bot: true });
            }
            let room = Room {
                id: id.clone(),
                variant,
                game: Game::new(variant),
                clock: tc.map(Clock::new),
                seats,
                local,
                bot: bot_level.map(|level| BotSeat { level, seat: 1 - seat_idx, thinking: false }),
                game_no: 0,
                rematch: [false; 2],
                last_activity: Instant::now(),
            };
            joined(tx, &room, &token, seat_idx);
            room.broadcast_state();
            rooms.insert(id.clone(), room);
            if let Some(room) = rooms.get_mut(&id) {
                maybe_bot_move(state, room);
            }
            *attached = Some(id);
        }
        ClientMsg::Join { room, name } => {
            let room_id = room.trim().to_uppercase();
            detach(state, conn_id, attached);
            let mut rooms = state.rooms.lock().unwrap();
            let Some(room) = rooms.get_mut(&room_id) else { return err(tx, "no room with that code") };
            let Some(seat_idx) = room.open_seat() else { return err(tx, "that room is full") };
            let token = random_token();
            room.seats[seat_idx] =
                Some(Seat { token: token.clone(), name: clean_name(name, "Player 2"), conn: Some((conn_id, tx.clone())), bot: false });
            room.last_activity = Instant::now();
            joined(tx, room, &token, seat_idx);
            room.broadcast_state();
            *attached = Some(room_id);
        }
        ClientMsg::Rejoin { room, token } => {
            let room_id = room.trim().to_uppercase();
            detach(state, conn_id, attached);
            let mut rooms = state.rooms.lock().unwrap();
            let Some(room) = rooms.get_mut(&room_id) else { return err(tx, "that game no longer exists") };
            let Some(seat_idx) = room.seat_of_token(&token) else { return err(tx, "not a player in that game") };
            if let Some(seat) = room.seats[seat_idx].as_mut() {
                seat.conn = Some((conn_id, tx.clone()));
            }
            room.last_activity = Instant::now();
            joined(tx, room, &token, seat_idx);
            room.broadcast_state();
            *attached = Some(room_id);
        }
        ClientMsg::Move { uci } => {
            let Some(room_id) = attached.clone() else { return err(tx, "not in a game") };
            let mut rooms = state.rooms.lock().unwrap();
            let Some(room) = rooms.get_mut(&room_id) else { return err(tx, "game not found") };
            let Some(seat_idx) = room.seat_of_conn(conn_id) else { return err(tx, "not a player in that game") };
            if !room.local {
                if room.seats[1 - seat_idx].is_none() {
                    return err(tx, "waiting for an opponent");
                }
                let my_color = if seat_idx == 0 { Color::White } else { Color::Black };
                if room.game.pos.side != my_color {
                    return err(tx, "not your turn");
                }
            }
            if room.check_flag() {
                room.broadcast_state();
                return;
            }
            let mover = room.game.pos.side.idx();
            if let Err(e) = room.game.play_uci(&uci) {
                err(tx, &e);
                // Resend state so the client snaps back to the true position.
                let _ = tx.send(room.state_json());
                return;
            }
            room.clock_after_move(mover);
            room.last_activity = Instant::now();
            room.broadcast_state();
            maybe_bot_move(state, room);
        }
        ClientMsg::Resign => {
            let Some(room_id) = attached.clone() else { return err(tx, "not in a game") };
            let mut rooms = state.rooms.lock().unwrap();
            let Some(room) = rooms.get_mut(&room_id) else { return };
            let Some(seat_idx) = room.seat_of_conn(conn_id) else { return err(tx, "not a player in that game") };
            let loser = if room.local {
                room.game.pos.side
            } else if seat_idx == 0 {
                Color::White
            } else {
                Color::Black
            };
            room.game.force_end(Status::Resigned { winner: loser.opp() });
            room.stop_clock();
            room.broadcast_state();
        }
        ClientMsg::Rematch => {
            let Some(room_id) = attached.clone() else { return err(tx, "not in a game") };
            let mut rooms = state.rooms.lock().unwrap();
            let Some(room) = rooms.get_mut(&room_id) else { return };
            let Some(seat_idx) = room.seat_of_conn(conn_id) else { return err(tx, "not a player in that game") };
            if !room.game.status().is_over() {
                return err(tx, "the game is still going");
            }
            if room.local {
                room.reset_game();
            } else {
                room.rematch[seat_idx] = true;
                if room.bot.is_some() {
                    // The bot always accepts.
                    room.rematch = [true, true];
                    if let Some(b) = room.bot.as_mut() {
                        b.seat = 1 - b.seat;
                    }
                }
                if room.rematch == [true, true] {
                    // New game, colours swapped.
                    room.seats.swap(0, 1);
                    room.rematch = [false; 2];
                    room.reset_game();
                    for (i, seat) in room.seats.iter().enumerate() {
                        if let Some(s) = seat {
                            if let Some((_, t)) = &s.conn {
                                joined(t, room, &s.token, i);
                            }
                        }
                    }
                }
            }
            room.broadcast_state();
            maybe_bot_move(state, room);
        }
        ClientMsg::Ping => send(tx, json!({ "t": "pong" })),
        ClientMsg::Leave => {
            let Some(room_id) = attached.clone() else { return };
            {
                let mut rooms = state.rooms.lock().unwrap();
                if rooms.get(&room_id).is_some_and(|r| r.local || r.bot.is_some()) {
                    rooms.remove(&room_id);
                    *attached = None;
                    return;
                }
                if let Some(room) = rooms.get_mut(&room_id) {
                    let Some(seat_idx) = room.seat_of_conn(conn_id) else {
                        *attached = None;
                        return;
                    };
                    if !room.game.status().is_over() && room.seats[1 - seat_idx].is_some() && !room.game.moves().is_empty() {
                        let me = if seat_idx == 0 { Color::White } else { Color::Black };
                        room.game.force_end(Status::Abandoned { winner: me.opp() });
                        room.stop_clock();
                    }
                    room.seats[seat_idx] = None;
                    room.broadcast_state();
                    if room.seats.iter().all(|s| s.is_none()) {
                        rooms.remove(&room_id);
                    }
                }
            }
            *attached = None;
        }
    }
}

/// If it's the bot's turn, thinks in the background and plays its move.
fn maybe_bot_move(state: &Shared, room: &mut Room) {
    let Some(net) = state.net.clone() else { return };
    let over = room.game.status().is_over();
    let side = room.game.pos.side;
    let clock_ms = room.clock.as_ref().map(|c| c.now_ms(if side == Color::White { 0 } else { 1 }));
    let Some(b) = room.bot.as_mut() else { return };
    let bot_color = if b.seat == 0 { Color::White } else { Color::Black };
    if b.thinking || over || side != bot_color {
        return;
    }
    b.thinking = true;
    let level = b.level;
    let game = room.game.clone();
    let ply = game.moves().len();
    let game_no = room.game_no;
    // With a clock, never spend more than about 1/25 of the remaining time.
    let budget = clock_ms.map(|ms| Duration::from_millis((ms.max(0) as u64 / 25).max(100)));
    let room_id = room.id.clone();
    let st = state.clone();
    room.broadcast_state();
    tokio::spawn(async move {
        let started = Instant::now();
        let result = tokio::task::spawn_blocking(move || bot::choose(&net, &game, level, budget)).await;
        // A short pause so instant replies don't feel jarring.
        let min = Duration::from_millis(350);
        if started.elapsed() < min {
            tokio::time::sleep(min - started.elapsed()).await;
        }
        let mut rooms = st.rooms.lock().unwrap();
        let Some(room) = rooms.get_mut(&room_id) else { return };
        if room.game_no != game_no {
            return;
        }
        if let Some(b) = room.bot.as_mut() {
            b.thinking = false;
        }
        if room.game.moves().len() != ply || room.game.status().is_over() {
            room.broadcast_state();
            return;
        }
        let choice = match result {
            Ok(Ok(c)) => c,
            other => {
                eprintln!("bot failed in room {room_id}: {other:?}");
                room.broadcast_state();
                return;
            }
        };
        if room.check_flag() {
            room.broadcast_state();
            return;
        }
        let mover = room.game.pos.side.idx();
        if room.game.play(choice.mv).is_ok() {
            room.clock_after_move(mover);
            room.last_activity = Instant::now();
        }
        room.broadcast_state();
    });
}

/// Ends games whose running clock has hit zero.
async fn clock_ticker(state: Shared) {
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        tick.tick().await;
        let mut rooms = state.rooms.lock().unwrap();
        for room in rooms.values_mut() {
            if room.check_flag() {
                room.broadcast_state();
            }
        }
    }
}

/// Drops rooms where nobody has been connected for 30 minutes.
async fn sweeper(state: Shared) {
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        let mut rooms = state.rooms.lock().unwrap();
        rooms.retain(|_, r| {
            let anyone = r.seats.iter().flatten().any(|s| s.conn.is_some());
            anyone || r.last_activity.elapsed() < Duration::from_secs(30 * 60)
        });
    }
}

fn lan_ip() -> Option<String> {
    // Connecting a UDP socket sends nothing; it just asks the OS which interface it would use.
    let s = UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("192.168.0.1:9").or_else(|_| s.connect("10.0.0.1:9")).ok()?;
    Some(s.local_addr().ok()?.ip().to_string())
}

fn web_dir() -> PathBuf {
    if let Ok(p) = std::env::var("DROPCHESS_WEB") {
        return PathBuf::from(p);
    }
    for candidate in ["web/dist", "../web/dist"] {
        let p = PathBuf::from(candidate);
        if p.join("index.html").exists() {
            return p;
        }
    }
    PathBuf::from("web/dist")
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let web = web_dir();
    let model_path = std::env::var("DROPCHESS_MODEL").map(PathBuf::from).unwrap_or_else(|_| {
        [web.join("models/dropchess.onnx"), PathBuf::from("web/public/models/dropchess.onnx")]
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| web.join("models/dropchess.onnx"))
    });
    let net = match bot::Net::load(&model_path) {
        Ok(n) => {
            println!("bot: loaded {}", model_path.display());
            Some(Arc::new(n))
        }
        Err(e) => {
            eprintln!("bot: disabled, could not load {} ({e})", model_path.display());
            None
        }
    };
    let bot_available = net.is_some();
    let state: Shared = Arc::new(AppState { net, ..Default::default() });
    tokio::spawn(sweeper(state.clone()));
    tokio::spawn(clock_ticker(state.clone()));

    if !web.join("index.html").exists() {
        eprintln!("warning: no web client at {} (run `npm install && npm run build` in web/)", web.display());
    }
    let static_files = ServeDir::new(&web).fallback(ServeFile::new(web.join("index.html")));

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(|| async { "ok" }))
        .route(
            "/info",
            get(move || async move {
                axum::Json(json!({ "lan": lan_ip().map(|ip| format!("http://{ip}:{port}")), "bot": bot_available }))
            }),
        )
        .fallback_service(static_files)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.expect("could not bind port");
    println!("Drop Chess server running");
    println!("  this computer:  http://localhost:{port}");
    match lan_ip() {
        Some(ip) => println!("  on your network: http://{ip}:{port}"),
        None => println!("  on your network: http://<this computer's IP>:{port}"),
    }
    axum::serve(listener, app).await.expect("server error");
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Client {
        id: u64,
        tx: Tx,
        rx: mpsc::UnboundedReceiver<String>,
        attached: Option<String>,
    }

    impl Client {
        fn new(id: u64) -> Client {
            let (tx, rx) = mpsc::unbounded_channel();
            Client { id, tx, rx, attached: None }
        }
        fn send(&mut self, state: &Shared, msg: Value) {
            let msg: ClientMsg = serde_json::from_value(msg).unwrap();
            handle_msg(state, self.id, &self.tx, &mut self.attached, msg);
        }
        /// All messages received so far, parsed.
        fn drain(&mut self) -> Vec<Value> {
            let mut out = vec![];
            while let Ok(m) = self.rx.try_recv() {
                out.push(serde_json::from_str(&m).unwrap());
            }
            out
        }
        fn last_joined_color(&mut self) -> String {
            let msgs = self.drain();
            let j = msgs.iter().rev().find(|m| m["t"] == "joined").expect("joined message");
            j["color"].as_str().unwrap().to_string()
        }
    }

    fn errors(msgs: &[Value]) -> Vec<String> {
        msgs.iter().filter(|m| m["t"] == "error").map(|m| m["msg"].as_str().unwrap().to_string()).collect()
    }

    /// Issue #5: after a rematch swaps colours, both players must be able to move,
    /// whichever of them sent the final rematch request.
    #[test]
    fn both_players_can_move_after_rematch() {
        for final_by_a in [true, false] {
            let state: Shared = Arc::new(AppState::default());
            let (mut a, mut b) = (Client::new(1), Client::new(2));
            a.send(&state, json!({"t": "create", "variant": "drop", "color": "white", "name": "A"}));
            let room = a.drain().iter().find(|m| m["t"] == "joined").unwrap()["room"].as_str().unwrap().to_string();
            b.send(&state, json!({"t": "join", "room": room, "name": "B"}));
            a.send(&state, json!({"t": "move", "uci": "P@e2"}));
            b.send(&state, json!({"t": "resign"}));
            a.drain();
            b.drain();
            if final_by_a {
                b.send(&state, json!({"t": "rematch"}));
                a.send(&state, json!({"t": "rematch"}));
            } else {
                a.send(&state, json!({"t": "rematch"}));
                b.send(&state, json!({"t": "rematch"}));
            }
            assert_eq!(a.last_joined_color(), "black");
            assert_eq!(b.last_joined_color(), "white");
            b.send(&state, json!({"t": "move", "uci": "N@c3"}));
            assert!(errors(&b.drain()).is_empty(), "new White could not move");
            a.send(&state, json!({"t": "move", "uci": "N@c6"}));
            assert!(errors(&a.drain()).is_empty(), "new Black could not move");
            let rooms = state.rooms.lock().unwrap();
            assert_eq!(rooms[&room].game.moves().len(), 2);
        }
    }

    #[test]
    fn moving_out_of_turn_is_rejected() {
        let state: Shared = Arc::new(AppState::default());
        let (mut a, mut b) = (Client::new(1), Client::new(2));
        a.send(&state, json!({"t": "create", "variant": "drop", "color": "white", "name": "A"}));
        let room = a.drain().iter().find(|m| m["t"] == "joined").unwrap()["room"].as_str().unwrap().to_string();
        b.send(&state, json!({"t": "join", "room": room, "name": "B"}));
        b.send(&state, json!({"t": "move", "uci": "N@c6"}));
        assert_eq!(errors(&b.drain()), vec!["not your turn".to_string()]);
    }
}

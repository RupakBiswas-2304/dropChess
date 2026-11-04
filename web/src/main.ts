import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "@fontsource/inter/latin-600.css";
import "@fontsource/inter/latin-700.css";
import "./style.css";
import {
  Connection,
  type BotLevel,
  type ColorName,
  type GameState,
  type RoomSummary,
  type ServerMsg,
  type TimeControl,
  type Variant,
} from "./net";
import { applyTheme, effectiveTheme, toggleTheme } from "./theme";
import { play, soundEnabled, soundForSan, toggleSound } from "./sound";
import { parseFen, type Reserve } from "./fen";
import { annotationsSvg, clearAnnotations, colorFor, hasAnnotations, toggleMark, type Arrow } from "./annotations";

// ---------- state ----------

type Selection = { type: "sq"; sq: number } | { type: "drop"; kind: string } | null;

interface UI {
  screen: "lobby" | "game";
  name: string;
  variant: Variant;
  tcIndex: number;
  rooms: RoomSummary[];
  myColor: ColorName | "both" | null;
  state: GameState | null;
  selected: Selection;
  promo: { from: number; to: number; moves: string[] } | null;
  flipped: boolean;
  connected: boolean;
  lanUrl: string | null;
  toast: string | null;
  confirmResign: boolean;
  confirmLeave: boolean;
  hideResult: boolean;
  botAvailable: boolean;
  botOpen: boolean;
  botLevel: BotLevel;
  botColor: "white" | "random" | "black";
  /** Clock values from the last state, plus when we received them (performance.now()). */
  clockBase: { white: number; black: number; running: ColorName | null; at: number } | null;
  /** Ply being viewed in the move history, or null for the live position (issue #3). */
  viewPly: number | null;
  /** Right-drag in progress for a planning arrow (issue #4). */
  drawing: { from: number; to: number; color: Arrow["color"] } | null;
  /** The low-time warning sound has played for this game. */
  lowWarned: boolean;
  /** Board square whose piece is being dragged (drawn faded), and the square under the pointer (issue #8). */
  dragFrom: number | null;
  dragOver: number | null;
}

const TIME_CONTROLS: { label: string; speed: string; tc: TimeControl | null }[] = [
  { label: "1 + 0", speed: "Bullet", tc: { initialMs: 60_000, incMs: 0 } },
  { label: "3 + 2", speed: "Blitz", tc: { initialMs: 180_000, incMs: 2_000 } },
  { label: "5 + 3", speed: "Blitz", tc: { initialMs: 300_000, incMs: 3_000 } },
  { label: "10 + 0", speed: "Rapid", tc: { initialMs: 600_000, incMs: 0 } },
  { label: "15 + 10", speed: "Rapid", tc: { initialMs: 900_000, incMs: 10_000 } },
  { label: "∞", speed: "No clock", tc: null },
];

const store = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string | null) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {
      /* storage unavailable */
    }
  },
};

const ui: UI = {
  screen: "lobby",
  name: store.get("dropchess.name") ?? "",
  variant: (store.get("dropchess.variant") as Variant) ?? "drop",
  tcIndex: Number(store.get("dropchess.tc") ?? 2),
  rooms: [],
  myColor: null,
  state: null,
  selected: null,
  promo: null,
  flipped: false,
  connected: false,
  lanUrl: null,
  toast: null,
  confirmResign: false,
  confirmLeave: false,
  hideResult: false,
  botAvailable: false,
  botOpen: false,
  botLevel: (store.get("dropchess.botLevel") as BotLevel) ?? "medium",
  botColor: "random",
  clockBase: null,
  viewPly: null,
  drawing: null,
  lowWarned: false,
  dragFrom: null,
  dragOver: null,
};
applyTheme();
if (!["easy", "medium", "hard"].includes(ui.botLevel)) ui.botLevel = "medium";
if (!(ui.tcIndex >= 0 && ui.tcIndex < TIME_CONTROLS.length)) ui.tcIndex = 2;

interface Session {
  room: string;
  token: string;
}
const loadSession = (): Session | null => {
  const raw = store.get("dropchess.session");
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Session;
  } catch {
    return null;
  }
};
const saveSession = (s: Session | null) => store.set("dropchess.session", s ? JSON.stringify(s) : null);

/** A game code in the URL (/g/K7QM) means "join this game". */
const linkCode = (): string | null => {
  const m = location.pathname.match(/^\/g\/([A-Za-z0-9]{4})\/?$/);
  return m ? m[1].toUpperCase() : null;
};
const setPath = (path: string) => {
  if (location.pathname !== path) history.replaceState(null, "", path);
};

// ---------- connection ----------

const conn = new Connection(onMessage, onConnection, () => {
  const s = loadSession();
  const code = linkCode();
  if (s && (!code || code === s.room)) conn.send({ t: "rejoin", room: s.room, token: s.token });
  else if (code) conn.send({ t: "join", room: code, name: ui.name || "Guest" });
  else if (ui.screen === "lobby") conn.send({ t: "list" });
});

fetch("/info")
  .then((r) => r.json())
  .then((info: { lan: string | null; bot?: boolean }) => {
    ui.lanUrl = info.lan;
    ui.botAvailable = !!info.bot;
    if (ui.screen === "lobby" && document.activeElement?.tagName !== "INPUT") renderLobby();
  })
  .catch(() => {});

setInterval(() => {
  if (ui.screen === "lobby" && conn.connected) conn.send({ t: "list" });
}, 2000);

function onConnection(connected: boolean) {
  ui.connected = connected;
  const dot = document.getElementById("conn");
  if (dot) {
    dot.className = connected ? "conn on" : "conn off";
    dot.title = connected ? "Connected" : "Reconnecting…";
  }
}

function onMessage(msg: ServerMsg) {
  switch (msg.t) {
    case "rooms":
      ui.rooms = msg.rooms;
      if (ui.screen === "lobby") renderRooms();
      break;
    case "joined":
      saveSession({ room: msg.room, token: msg.token });
      ui.myColor = msg.color;
      ui.variant = msg.variant;
      ui.screen = "game";
      ui.selected = null;
      ui.promo = null;
      if (msg.color !== "both" && !msg.bot) setPath(`/g/${msg.room}`);
      else setPath("/");
      break;
    case "state": {
      const prev = ui.state;
      ui.state = msg;
      const sameGame = !!prev && prev.room === msg.room && msg.sans.length >= prev.sans.length;
      if (!sameGame) {
        ui.viewPly = null;
        ui.lowWarned = false;
        clearAnnotations();
      }
      if (ui.viewPly !== null && ui.viewPly >= msg.sans.length) ui.viewPly = null;
      if (prev && sameGame && msg.sans.length > prev.sans.length) {
        clearAnnotations();
        play(soundForSan(msg.sans[msg.sans.length - 1]));
      }
      const startedNow = msg.status.kind === "ongoing" && msg.sans.length === 0 && bothSeated(msg) && (!prev || !sameGame || !bothSeated(prev));
      if (startedNow) play("start");
      if (prev && sameGame && prev.status.kind === "ongoing" && msg.status.kind !== "ongoing") window.setTimeout(() => play("end"), 180);
      ui.clockBase = msg.clock
        ? { white: msg.clock.white, black: msg.clock.black, running: msg.clock.running, at: performance.now() }
        : null;
      if (!prev || prev.fen !== msg.fen) {
        ui.selected = null;
        ui.promo = null;
      }
      if (msg.status.kind === "ongoing") ui.hideResult = false;
      if (ui.screen === "game") renderGame();
      break;
    }
    case "error":
      if (["that game no longer exists", "not a player in that game", "no room with that code", "that room is full"].includes(msg.msg)) {
        saveSession(null);
        setPath("/");
        if (ui.screen !== "lobby" || !document.querySelector(".lobby")) goLobby();
      }
      showToast(friendlyError(msg.msg));
      break;
    case "pong":
      break;
  }
}

function friendlyError(msg: string): string {
  if (msg === "that room is full") return "That game already has two players.";
  if (msg === "no room with that code") return "No game with that code. It may have ended.";
  if (msg === "that game no longer exists") return "That game has ended.";
  return msg[0].toUpperCase() + msg.slice(1);
}

function showToast(text: string) {
  ui.toast = text;
  const el = document.getElementById("toast");
  if (el) {
    el.textContent = text;
    el.classList.add("show");
  }
  window.setTimeout(() => {
    if (ui.toast === text) {
      ui.toast = null;
      document.getElementById("toast")?.classList.remove("show");
    }
  }, 2800);
}

function goLobby() {
  ui.screen = "lobby";
  ui.state = null;
  ui.myColor = null;
  ui.selected = null;
  ui.promo = null;
  ui.flipped = false;
  ui.clockBase = null;
  setPath("/");
  renderLobby();
  conn.send({ t: "list" });
}

// ---------- helpers ----------

// Piece art: "cburnett" set by Colin M.L. Burnett (BSD-3-Clause, see web/src/pieces/LICENSE).
// The SVGs are inlined once as <symbol>s, then drawn with <use>, so nothing flickers on re-render.
const PIECE_SVGS = import.meta.glob("./pieces/*.svg", { query: "?raw", import: "default", eager: true }) as Record<string, string>;
(function installPieceSymbols() {
  let symbols = "";
  for (const [path, svg] of Object.entries(PIECE_SVGS)) {
    const id = path.match(/([wb][KQRBNP])\.svg$/)?.[1];
    if (!id) continue;
    const inner = svg.replace(/^[\s\S]*?<svg[^>]*>/, "").replace(/<\/svg>\s*$/, "");
    symbols += `<symbol id="pc-${id}" viewBox="0 0 45 45">${inner}</symbol>`;
  }
  const holder = document.createElement("div");
  holder.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0" style="position:absolute" aria-hidden="true"><defs>${symbols}</defs></svg>`;
  document.body.prepend(holder.firstElementChild!);
})();
type Side = "w" | "b";
const pieceUse = (c: Side, k: string, x: number, y: number, size: number) =>
  `<use href="#pc-${c}${k}" x="${x}" y="${y}" width="${size}" height="${size}"/>`;
const pieceIcon = (c: Side, k: string, cls = "pi") =>
  `<svg class="${cls}" viewBox="0 0 45 45" aria-hidden="true"><use href="#pc-${c}${k}"/></svg>`;

const ICON = {
  flag: '<path d="M4 22V4"/><path d="M4 4h12l-2 4 2 4H4"/>',
  flip: '<path d="M7 4v16"/><path d="m3 8 4-4 4 4"/><path d="M17 20V4"/><path d="m21 16-4 4-4-4"/>',
  share: '<path d="M4 12v8h16v-8"/><path d="M12 3v13"/><path d="m7 8 5-5 5 5"/>',
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  exit: '<path d="M15 4h4a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-4"/><path d="M10 17l5-5-5-5"/><path d="M15 12H4"/>',
};
const icon = (name: keyof typeof ICON) =>
  `<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[name]}</svg>`;

const BOT_ICON = `<svg class="bot-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="8" width="16" height="12" rx="3"/><path d="M12 4v4"/><circle cx="9" cy="14" r="1" fill="currentColor"/><circle cx="15" cy="14" r="1" fill="currentColor"/></svg>`;
const PIECE_NAME: Record<string, string> = { P: "pawn", N: "knight", B: "bishop", R: "rook", Q: "queen", K: "king" };
const RESERVE_ORDER = ["Q", "R", "B", "N", "P"] as const;

const sqName = (sq: number) => "abcdefgh"[sq % 8] + (Math.floor(sq / 8) + 1);
const parseSq = (s: string) => (s.charCodeAt(1) - 49) * 8 + (s.charCodeAt(0) - 97);
const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const variantLabel = (v: Variant) => (v === "drop" ? "Drop" : "Crazy Drop");
const cap = (c: string) => c[0].toUpperCase() + c.slice(1);
const other = (c: ColorName): ColorName => (c === "white" ? "black" : "white");

function tcLabel(tc: TimeControl | null): string {
  if (!tc) return "∞";
  const min = tc.initialMs / 60000;
  const minStr = Number.isInteger(min) ? String(min) : min < 1 ? `${tc.initialMs / 1000}s` : min.toFixed(1);
  return `${minStr} + ${Math.round(tc.incMs / 1000)}`;
}

function formatClock(ms: number): string {
  ms = Math.max(0, ms);
  if (ms < 10_000) return `${Math.floor(ms / 1000)}.${Math.floor((ms % 1000) / 100)}`;
  const total = Math.ceil(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

function clockNow(color: ColorName): number {
  const b = ui.clockBase;
  if (!b) return 0;
  const base = b[color];
  return b.running === color ? base - (performance.now() - b.at) : base;
}

function inviteUrl(code: string): string {
  const local = ["localhost", "127.0.0.1", "::1"].includes(location.hostname);
  const origin = local && ui.lanUrl ? ui.lanUrl : location.origin;
  return `${origin}/g/${code}`;
}

async function shareInvite(code: string, preferShare: boolean) {
  const url = inviteUrl(code);
  const nav = navigator as Navigator & { share?: (d: ShareData) => Promise<void> };
  if (preferShare && nav.share) {
    try {
      await nav.share({ title: "Drop Chess", text: "Play Drop Chess with me", url });
      return;
    } catch {
      /* cancelled: fall back to copy */
    }
  }
  try {
    await navigator.clipboard.writeText(url);
    showToast("Invite link copied");
  } catch {
    const input = document.getElementById("invite-url") as HTMLInputElement | null;
    input?.select();
    showToast("Select the link and copy it");
  }
}

function whiteAtBottom(): boolean {
  const base = ui.myColor !== "black";
  return ui.flipped ? !base : base;
}

function bothSeated(s: GameState): boolean {
  return s.local || !!(s.players.white && s.players.black);
}

function canMove(): boolean {
  const s = ui.state;
  if (!s || s.status.kind !== "ongoing" || !bothSeated(s) || ui.viewPly !== null) return false;
  return ui.myColor === "both" || ui.myColor === s.side;
}

function winnerTitle(s: GameState, winner: ColorName): string {
  const me = ui.myColor;
  if (s.local || !me || me === "both") return `${cap(winner)} wins`;
  if (winner === me) return "You win";
  return `${s.players[winner]?.name ?? cap(winner)} wins`;
}

function resultText(s: GameState): { label: string; title: string; sub: string } {
  const st = s.status;
  const tc = s.clock ? ` · ${tcLabel(s.clock)}` : "";
  const moves = `${Math.ceil(s.sans.length / 2)} moves`;
  switch (st.kind) {
    case "checkmate":
      return { label: "CHECKMATE", title: winnerTitle(s, st.winner), sub: `${s.sans[s.sans.length - 1] ?? ""} · ${moves}${tc}` };
    case "timeout":
      return { label: "TIME'S UP", title: winnerTitle(s, st.winner), sub: `${cap(other(st.winner))} ran out of time${tc}` };
    case "resigned":
      return { label: "RESIGNED", title: winnerTitle(s, st.winner), sub: `${cap(other(st.winner))} resigned · ${moves}` };
    case "abandoned":
      return { label: "OPPONENT LEFT", title: winnerTitle(s, st.winner), sub: moves };
    case "stalemate":
      return { label: "STALEMATE", title: "Draw", sub: moves };
    case "repetition":
      return { label: "THREEFOLD REPETITION", title: "Draw", sub: moves };
    case "fiftymove":
      return { label: "50-MOVE RULE", title: "Draw", sub: moves };
    case "insufficient":
      return { label: "INSUFFICIENT MATERIAL", title: "Draw", sub: moves };
    default:
      return { label: "", title: "", sub: "" };
  }
}

// ---------- lobby ----------

function renderLobby() {
  const v = ui.variant;
  const chips = TIME_CONTROLS.map(
    (t, i) => `<button class="tc ${i === ui.tcIndex ? "on" : ""}" data-action="tc" data-i="${i}">
      <span class="tc-t">${t.label}</span><span class="tc-s">${t.speed}</span></button>`,
  ).join("");
  app.innerHTML = `
  <main class="lobby">
    <header>
      <div class="brand-row">${wordmark(22)}<div class="brand-tools">${prefButtons()}<span id="conn" class="conn ${ui.connected ? "on" : "off"}"></span></div></div>
      <p class="tag">Start with only your king. Each turn, drop a piece on your half or make a move.</p>
      <details class="how"><summary>How to play</summary>
        <ul>
          <li>Your other 15 pieces wait in your reserve.</li>
          <li>Each turn: drop one reserve piece on an empty square of your half (ranks 1–4 for White, 5–8 for Black), or move a piece.</li>
          <li>No pawn drops on your own back rank. Drops may give check, even checkmate.</li>
          <li><b>Drop</b>: captured pieces are gone. <b>Crazy Drop</b>: pieces you capture join your reserve.</li>
          <li>Everything else is normal chess, without castling.</li>
        </ul>
      </details>
    </header>

    <section>
      <label class="label" for="name">NAME</label>
      <input id="name" class="input" maxlength="24" placeholder="Your name" value="${escapeHtml(ui.name)}" autocomplete="nickname" />
    </section>

    <section>
      <div class="label">VARIANT</div>
      <div class="seg">
        <button data-action="variant" data-v="drop" class="${v === "drop" ? "on" : ""}"><b>Drop</b><span>captures are gone</span></button>
        <button data-action="variant" data-v="crazy" class="${v === "crazy" ? "on" : ""}"><b>Crazy Drop</b><span>captures join your reserve</span></button>
      </div>
    </section>

    <section>
      <div class="label">TIME</div>
      <div class="tcs">${chips}</div>
    </section>

    <section class="actions-col">
      <button class="btn primary big" data-action="create">Create game</button>
      <div class="row">
        <button class="btn" data-action="local">Play on this screen</button>
        <button class="btn ${ui.botOpen ? "on" : ""}" data-action="bot-toggle" ${ui.botAvailable ? "" : "disabled"}>${ui.botAvailable ? "Play vs Bot" : "vs Bot · unavailable"}</button>
      </div>
      ${ui.botOpen && ui.botAvailable ? botPanel() : ""}
    </section>

    <section>
      <div class="label">JOIN A GAME</div>
      <div class="row">
        <input id="code" class="input code-input" maxlength="4" placeholder="CODE" autocomplete="off" />
        <button class="btn" data-action="join">Join</button>
      </div>
      <div id="rooms"></div>
    </section>
  </main>
  <div id="toast" class="toast"></div>`;
  renderRooms();
  const nameEl = document.getElementById("name") as HTMLInputElement;
  nameEl.addEventListener("input", () => {
    ui.name = nameEl.value;
    store.set("dropchess.name", ui.name);
  });
  const codeEl = document.getElementById("code") as HTMLInputElement;
  codeEl.addEventListener("input", () => (codeEl.value = codeEl.value.toUpperCase()));
  codeEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") joinRoom(codeEl.value);
  });
}

function botPanel(): string {
  const lv = (["easy", "medium", "hard"] as BotLevel[])
    .map((l) => `<button data-action="bot-level" data-l="${l}" class="${ui.botLevel === l ? "on" : ""}"><b>${cap(l)}</b><span>${l === "easy" ? "instinct only" : l === "medium" ? "thinks a little" : "thinks hard"}</span></button>`)
    .join("");
  const col = (["white", "random", "black"] as const)
    .map((c) => `<button data-action="bot-color" data-c="${c}" class="${ui.botColor === c ? "on" : ""}"><b>${cap(c)}</b></button>`)
    .join("");
  return `<div class="bot-panel">
    <div class="label">BOT LEVEL</div>
    <div class="seg">${lv}</div>
    <div class="label">YOU PLAY</div>
    <div class="seg seg-center">${col}</div>
    <button class="btn primary big" data-action="bot-start">Start game vs Bot</button>
    <p class="hint">Uses the variant and time picked above.</p>
  </div>`;
}

function wordmark(size = 18): string {
  return `<div class="wordmark" style="--wm:${size}px"><span class="wm-badge">${pieceIcon("w", "N")}</span><span class="wm-text">Drop Chess</span></div>`;
}

function renderRooms() {
  const el = document.getElementById("rooms");
  if (!el) return;
  const open = ui.rooms.filter((r) => r.open && !r.over);
  if (open.length === 0) {
    el.innerHTML = `<p class="empty">No open games right now.</p>`;
    return;
  }
  el.innerHTML = open
    .map(
      (r) => `<div class="room">
        <span class="code">${r.id}</span>
        <span class="host">${escapeHtml(r.host)}</span>
        <span class="meta">${variantLabel(r.variant)} · ${tcLabel(r.tc)}</span>
        <button class="btn sm" data-action="join-room" data-room="${r.id}">Join</button>
      </div>`,
    )
    .join("");
}

function joinRoom(code: string) {
  const room = code.trim().toUpperCase();
  if (room.length !== 4) return showToast("Enter the 4-letter game code");
  conn.send({ t: "join", room, name: ui.name || "Guest" });
}

// ---------- game ----------

/** What the board shows: the live position, or an earlier one from the move history. */
interface BoardView {
  live: boolean;
  ply: number;
  board: GameState["board"];
  reserve: Record<ColorName, Reserve>;
  side: ColorName;
  lastMove: string | null;
  check: number | null;
}

function boardView(): BoardView {
  const s = ui.state!;
  const total = s.sans.length;
  if (ui.viewPly === null || ui.viewPly >= total || !s.fens?.[ui.viewPly]) {
    return { live: true, ply: total, board: s.board, reserve: s.reserve, side: s.side, lastMove: s.lastMove, check: s.check };
  }
  const p = parseFen(s.fens[ui.viewPly]);
  return { live: false, ply: ui.viewPly, board: p.board, reserve: p.reserve, side: p.side, lastMove: s.ucis?.[ui.viewPly - 1] ?? null, check: null };
}

function goToPly(ply: number | null) {
  const s = ui.state;
  if (!s) return;
  const total = s.sans.length;
  ui.viewPly = ply === null || ply >= total ? null : Math.max(0, ply);
  ui.selected = null;
  ui.promo = null;
  renderGame();
}

function navigate(to: string) {
  const s = ui.state;
  if (!s) return;
  const total = s.sans.length;
  const cur = ui.viewPly ?? total;
  if (to === "first") goToPly(0);
  else if (to === "prev") goToPly(cur - 1);
  else if (to === "next") goToPly(cur + 1);
  else goToPly(null);
}

function legalFrom(sq: number): string[] {
  const from = sqName(sq);
  return (ui.state?.legal ?? []).filter((m) => !m.includes("@") && m.startsWith(from));
}

function targets(): Map<number, string[]> {
  const out = new Map<number, string[]>();
  const sel = ui.selected;
  if (!sel || !ui.state) return out;
  const list = sel.type === "sq" ? legalFrom(sel.sq) : ui.state.legal.filter((m) => m.startsWith(sel.kind + "@"));
  for (const m of list) {
    const to = parseSq(m.slice(2, 4));
    out.set(to, [...(out.get(to) ?? []), m]);
  }
  return out;
}

function lastMoveSquares(m: string | null): number[] {
  if (!m) return [];
  if (m.includes("@")) return [parseSq(m.slice(2, 4))];
  return [parseSq(m.slice(0, 2)), parseSq(m.slice(2, 4))];
}

function squareCenter(sq: number, wb: boolean): [number, number] {
  const f = sq % 8;
  const r = Math.floor(sq / 8);
  return [(wb ? f : 7 - f) * 100 + 50, (wb ? 7 - r : r) * 100 + 50];
}

function renderBoard(): string {
  const v = boardView();
  const wb = whiteAtBottom();
  const tg = v.live ? targets() : new Map<number, string[]>();
  const last = lastMoveSquares(v.lastMove);
  const sel = v.live ? ui.selected : null;
  const dropping = sel?.type === "drop";
  let cells = "";
  let pieces = "";
  let marks = "";
  let hits = "";
  for (let sq = 0; sq < 64; sq++) {
    const f = sq % 8;
    const r = Math.floor(sq / 8);
    const x = (wb ? f : 7 - f) * 100;
    const y = (wb ? 7 - r : r) * 100;
    const light = (f + r) % 2 === 1;
    let cls = light ? "sq-l" : "sq-d";
    if (last.includes(sq)) cls = light ? "sq-last-l" : "sq-last-d";
    if (sel?.type === "sq" && sel.sq === sq) cls = light ? "sq-sel-l" : "sq-sel-d";
    cells += `<rect x="${x}" y="${y}" width="100" height="100" class="${cls}"/>`;
    if (dropping && (v.side === "white" ? r <= 3 : r >= 4)) cells += `<rect x="${x}" y="${y}" width="100" height="100" class="zone"/>`;
    if (v.check === sq) cells += `<rect x="${x}" y="${y}" width="100" height="100" fill="url(#checkGlow)"/>`;
    const p = v.board[sq];
    if (p && ui.dragFrom === sq) {
      pieces += `<g opacity="0.3">${pieceUse(p.c, p.k, x + 3, y + 3, 94)}</g>`;
    } else if (p) {
      pieces += pieceUse(p.c, p.k, x + 3, y + 3, 94);
      if (p.promoted) pieces += `<circle cx="${x + 86}" cy="${y + 14}" r="6" class="promo-dot"/>`;
    }
    if (tg.has(sq) && ui.dragOver === sq) marks += `<rect x="${x + 2}" y="${y + 2}" width="96" height="96" class="drag-over"/>`;
    if (tg.has(sq)) {
      marks += p
        ? `<circle cx="${x + 50}" cy="${y + 50}" r="44" class="tgt-ring"/>`
        : `<circle cx="${x + 50}" cy="${y + 50}" r="14" class="tgt-dot"/>`;
    }
    const cc = light ? "on-l" : "on-d";
    if (wb ? r === 0 : r === 7) cells += `<text x="${x + 94}" y="${y + 93}" class="coord ${cc}" text-anchor="end">${"abcdefgh"[f]}</text>`;
    if (wb ? f === 0 : f === 7) cells += `<text x="${x + 6}" y="${y + 19}" class="coord ${cc}">${r + 1}</text>`;
    hits += `<rect x="${x}" y="${y}" width="100" height="100" class="hit" data-action="sq" data-sq="${sq}" aria-label="${sqName(sq)}"/>`;
  }
  const defs = `<defs><radialGradient id="checkGlow"><stop offset="0%" stop-color="#e0402a" stop-opacity="0.9"/><stop offset="60%" stop-color="#e0402a" stop-opacity="0.45"/><stop offset="100%" stop-color="#e0402a" stop-opacity="0"/></radialGradient></defs>`;
  const preview = ui.drawing && ui.drawing.from !== ui.drawing.to ? ui.drawing : null;
  const plan = annotationsSvg((sq) => squareCenter(sq, wb), preview);
  const banner = v.live
    ? ""
    : `<div class="history-banner">Move ${Math.ceil(v.ply / 2)}${v.ply % 2 === 1 ? "" : "…"} · viewing history <button class="link" data-action="nav" data-to="last">Back to live</button></div>`;
  return `<div class="board-wrap ${v.live ? "" : "past"}"><svg viewBox="0 0 800 800" class="board" role="img" aria-label="Chess board">${defs}${cells}${pieces}${marks}${plan}${hits}</svg>${banner}${v.live ? boardOverlay() : ""}</div>`;
}

function boardOverlay(): string {
  const s = ui.state!;
  if (ui.promo) {
    const opts = ["q", "r", "b", "n"].filter((k) => ui.promo!.moves.some((m) => m.endsWith(k)));
    const c: Side = s.side === "white" ? "w" : "b";
    return `<div class="overlay"><div class="card">
      <div class="card-label">PROMOTE TO</div>
      <div class="promo-row">${opts
        .map((k) => `<button class="promo-btn" data-action="promo" data-k="${k}" title="${PIECE_NAME[k.toUpperCase()]}">${pieceIcon(c, k.toUpperCase())}</button>`)
        .join("")}</div>
      <button class="link" data-action="promo-cancel">Cancel</button>
    </div></div>`;
  }
  if (!bothSeated(s) && s.status.kind === "ongoing") {
    const url = inviteUrl(s.room);
    const canShare = "share" in navigator;
    return `<div class="overlay"><div class="card">
      <div class="card-label">WAITING FOR YOUR FRIEND</div>
      <div class="card-title">Send them this link</div>
      <input id="invite-url" class="input invite-input" readonly value="${escapeHtml(url)}" />
      <button class="btn primary big" data-action="copy-invite">${icon("copy")} Copy link</button>
      ${canShare ? `<button class="btn big" data-action="share-invite">${icon("share")} Share…</button>` : ""}
      <div class="card-sub">or tell them the code <b class="code">${s.room}</b></div>
    </div></div>`;
  }
  if (s.status.kind !== "ongoing" && !ui.hideResult) {
    const r = resultText(s);
    const me = ui.myColor;
    const myRematch = me !== "both" && me && s.rematch[me];
    const theirRematch = me !== "both" && me && s.rematch[other(me)];
    const opponentHere = s.local || (me && me !== "both" && s.players[other(me)]?.online);
    const rematchLabel = s.local ? "New game" : s.bot ? "Play again" : myRematch ? "Waiting for opponent…" : theirRematch ? "Accept rematch" : "Rematch";
    return `<div class="overlay soft"><div class="card">
      <div class="card-label">${r.label}</div>
      <div class="card-title big">${r.title}</div>
      <div class="card-sub">${escapeHtml(r.sub)}</div>
      ${theirRematch && !myRematch ? `<div class="card-note">Your opponent wants a rematch</div>` : ""}
      <button class="btn primary big" data-action="rematch" ${myRematch || !opponentHere ? "disabled" : ""}>${rematchLabel}</button>
      <button class="btn big" data-action="leave">Back to lobby</button>
      <button class="link" data-action="view-board">View board</button>
    </div></div>`;
  }
  return "";
}

function renderClock(color: ColorName): string {
  const s = ui.state!;
  if (!s.clock) {
    const turn = s.status.kind === "ongoing" && bothSeated(s) && s.side === color;
    return `<div class="turn-tag ${turn ? "on" : ""}">${turn ? "to move" : ""}</div>`;
  }
  return `<div class="clock" id="clock-${color}" data-color="${color}">${formatClock(clockNow(color))}</div>`;
}

function updateClocks() {
  const s = ui.state;
  if (!s?.clock || ui.screen !== "game") return;
  for (const color of ["white", "black"] as ColorName[]) {
    const el = document.getElementById(`clock-${color}`);
    if (!el) continue;
    const ms = clockNow(color);
    const running = ui.clockBase?.running === color && s.status.kind === "ongoing";
    const text = formatClock(ms);
    if (el.textContent !== text) el.textContent = text;
    const cls = `clock ${running ? (ms < 20_000 ? "low" : "active") : ms < 20_000 && s.status.kind === "ongoing" ? "idle low-idle" : "idle"}`;
    if (el.className !== cls) el.className = cls;
    const mine = ui.myColor === color || (ui.myColor === "both" && running);
    if (running && mine && ms < 10_000 && !ui.lowWarned && s.clock.initialMs > 30_000) {
      ui.lowWarned = true;
      play("lowtime");
    }
  }
}
window.setInterval(updateClocks, 100);

function renderStrip(color: ColorName): string {
  const s = ui.state!;
  const counts = boardView().reserve[color];
  const mine = ui.myColor === "both" ? s.side === color : ui.myColor === color;
  const active = mine && canMove() && s.side === color;
  const sel = ui.selected;
  const player = s.players[color];
  const name = s.local ? cap(color) : player ? escapeHtml(player.name) : "Waiting…";
  const isBot = !!player?.bot;
  const thinking = isBot && s.bot?.thinking && s.status.kind === "ongoing";
  const online = s.local || player?.online;
  const c: Side = color === "white" ? "w" : "b";
  const items = RESERVE_ORDER.map((k) => {
    const n = counts[k];
    const isSel = active && sel?.type === "drop" && sel.kind === k;
    const canDrop = active && n > 0 && s.legal.some((m) => m.startsWith(k + "@"));
    return `<button class="res ${n === 0 ? "empty" : ""} ${isSel ? "on" : ""} ${canDrop ? "live" : ""}"
      ${canDrop ? `data-action="res" data-kind="${k}"` : "disabled"} title="${n} ${PIECE_NAME[k]}${n === 1 ? "" : "s"} in reserve">
      ${pieceIcon(c, k, "g")}${n > 0 ? `<span class="n">${n}</span>` : ""}</button>`;
  }).join("");
  return `<div class="strip">
    <div class="who">
      <span class="avatar ${color}">${isBot ? BOT_ICON : escapeHtml((s.local ? cap(color) : player?.name ?? "?")[0] ?? "?")}</span>
      <span class="pname">${name}</span>
      ${thinking ? `<span class="thinking" title="thinking"><i></i><i></i><i></i></span>` : ""}
      ${!s.local && player && !isBot ? `<span class="dot ${online ? "on" : "off"}" title="${online ? "online" : "offline"}"></span>` : ""}
      ${!s.local && ui.myColor === color ? `<span class="you">YOU</span>` : ""}
    </div>
    <div class="res-row">${items}</div>
    ${renderClock(color)}
  </div>`;
}

function renderMoves(): string {
  const sans = ui.state!.sans;
  if (sans.length === 0) return `<p class="empty">White starts: drop a piece or move the king.</p>`;
  const shown = ui.viewPly ?? sans.length; // the ply whose position is on the board
  const cell = (i: number) =>
    i < sans.length
      ? `<button class="mv ${i + 1 === shown ? "cur" : ""}" data-action="ply" data-ply="${i + 1}">${escapeHtml(sans[i])}</button>`
      : "<span></span>";
  let rows = "";
  for (let i = 0; i < sans.length; i += 2) {
    rows += `<li><span class="num">${i / 2 + 1}.</span>${cell(i)}${cell(i + 1)}</li>`;
  }
  return `<ol class="moves" id="moves">${rows}</ol>`;
}

function renderNav(): string {
  const total = ui.state!.sans.length;
  const cur = ui.viewPly ?? total;
  const b = (to: string, label: string, title: string, disabled: boolean) =>
    `<button class="nav-btn" data-action="nav" data-to="${to}" title="${title}" aria-label="${title}" ${disabled ? "disabled" : ""}>${label}</button>`;
  return `<div class="nav">${b("first", "«", "First position (Home)", cur === 0)}${b("prev", "‹", "Previous move (←)", cur === 0)}${b("next", "›", "Next move (→)", cur >= total)}${b("last", "»", "Live position (End)", cur >= total)}</div>`;
}

const THEME_ICON = {
  light: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  dark: '<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/>',
};
const SOUND_ICON = {
  on: '<path d="M11 5 6 9H3v6h3l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13"/>',
  off: '<path d="M11 5 6 9H3v6h3l5 4z"/><path d="m16 9 6 6M22 9l-6 6"/>',
};
const svgIcon = (paths: string) =>
  `<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;

/** Theme and sound toggles, shown in the lobby and the game's top bar. */
function prefButtons(): string {
  // The button shows where it takes you: a sun in dark mode, a moon in light mode.
  const dark = effectiveTheme() === "dark";
  const label = dark ? "Switch to light mode" : "Switch to dark mode";
  const on = soundEnabled();
  return `<button class="icon-btn" data-action="theme" title="${label}" aria-label="${label}">${svgIcon(dark ? THEME_ICON.light : THEME_ICON.dark)}</button>
    <button class="icon-btn" data-action="sound" title="${on ? "Sound on" : "Sound off"}" aria-label="${on ? "Sound on" : "Sound off"}">${svgIcon(on ? SOUND_ICON.on : SOUND_ICON.off)}</button>`;
}

function renderGame() {
  const s = ui.state;
  if (!s) {
    app.innerHTML = `<main class="loading">Connecting…</main><div id="toast" class="toast"></div>`;
    return;
  }
  const wb = whiteAtBottom();
  const top: ColorName = wb ? "black" : "white";
  const bottom: ColorName = wb ? "white" : "black";
  const over = s.status.kind !== "ongoing";
  const waiting = !bothSeated(s);
  let actions = "";
  if (!over && !waiting && s.sans.length > 0) {
    actions += `<button class="btn ${ui.confirmResign ? "danger" : ""}" data-action="resign">${icon("flag")}${ui.confirmResign ? "Confirm resign" : "Resign"}</button>`;
  }
  actions += `<button class="btn" data-action="flip">${icon("flip")}Flip</button>`;
  if (!s.local && waiting) actions += `<button class="btn" data-action="share-invite">${icon("share")}Invite</button>`;
  if (over && ui.hideResult) actions += `<button class="btn" data-action="show-result">Result</button>`;

  app.innerHTML = `
  <header class="topbar">
    ${wordmark()}
    <div class="top-right">
      <span class="pill">${variantLabel(s.variant)}</span>
      <span class="pill">${tcLabel(s.clock)}</span>
      ${s.local ? "" : `<span class="code top-code" title="Game code">${s.room}</span>`}
      <span id="conn" class="conn ${ui.connected ? "on" : "off"}"></span>
      ${prefButtons()}
      <button class="icon-btn" data-action="leave" title="${over || waiting || s.local ? "Back to lobby" : "Leave game (counts as a loss)"}">${icon("exit")}</button>
    </div>
  </header>
  <main class="game">
    <div class="side-l"></div>
    <div class="play">
      ${renderStrip(top)}
      ${renderBoard()}
      ${renderStrip(bottom)}
    </div>
    <aside class="side-r">
      <div class="label">MOVES</div>
      <div class="moves-wrap">${renderMoves()}</div>
      ${s.sans.length > 0 ? renderNav() : ""}
      <div class="actions">${actions}</div>
    </aside>
  </main>
  <div id="toast" class="toast ${ui.toast ? "show" : ""}">${ui.toast ? escapeHtml(ui.toast) : ""}</div>`;
  const mv = document.getElementById("moves");
  if (mv) {
    const cur = mv.querySelector<HTMLElement>(".mv.cur");
    const wrap = mv.parentElement!;
    if (cur && ui.viewPly !== null) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
    else {
      wrap.scrollTop = wrap.scrollHeight;
      wrap.scrollLeft = wrap.scrollWidth;
    }
  }
  updateClocks();
}

function sendMove(uci: string) {
  ui.selected = null;
  ui.promo = null;
  conn.send({ t: "move", uci });
  renderGame();
}

function onSquare(sq: number) {
  const s = ui.state;
  if (!s) return;
  if (ui.viewPly !== null) return goToPly(null);
  if (!canMove()) return;
  const tg = targets();
  if (tg.has(sq)) {
    const moves = tg.get(sq)!;
    if (moves.length > 1 && ui.selected?.type === "sq") {
      ui.promo = { from: ui.selected.sq, to: sq, moves };
      renderGame();
      return;
    }
    sendMove(moves[0]);
    return;
  }
  const p = s.board[sq];
  const mine = p && (p.c === "w" ? "white" : "black") === s.side;
  if (mine && legalFrom(sq).length > 0) {
    ui.selected = ui.selected?.type === "sq" && ui.selected.sq === sq ? null : { type: "sq", sq };
  } else {
    ui.selected = null;
  }
  renderGame();
}

// ---------- events ----------

const app = document.getElementById("app")!;

app.addEventListener("click", (e) => {
  if (suppressClick) {
    // This click is the tail end of a drag that already made (or cancelled) the move.
    suppressClick = false;
    return;
  }
  const el = (e.target as Element).closest<HTMLElement | SVGElement>("[data-action]");
  if (!el) return;
  const d = (el as HTMLElement).dataset;
  const action = d.action;
  if (action !== "resign" && ui.confirmResign) ui.confirmResign = false;
  switch (action) {
    case "variant":
      ui.variant = d.v as Variant;
      store.set("dropchess.variant", ui.variant);
      renderLobby();
      break;
    case "tc":
      ui.tcIndex = Number(d.i);
      store.set("dropchess.tc", String(ui.tcIndex));
      renderLobby();
      break;
    case "bot-toggle":
      ui.botOpen = !ui.botOpen;
      renderLobby();
      break;
    case "bot-level":
      ui.botLevel = d.l as BotLevel;
      store.set("dropchess.botLevel", ui.botLevel);
      renderLobby();
      break;
    case "bot-color":
      ui.botColor = d.c as UI["botColor"];
      renderLobby();
      break;
    case "bot-start":
      conn.send({ t: "create", variant: ui.variant, name: ui.name, tc: TIME_CONTROLS[ui.tcIndex].tc, bot: ui.botLevel, color: ui.botColor });
      break;
    case "local":
      conn.send({ t: "create", variant: ui.variant, local: true, name: ui.name, tc: TIME_CONTROLS[ui.tcIndex].tc });
      break;
    case "create":
      conn.send({ t: "create", variant: ui.variant, name: ui.name, tc: TIME_CONTROLS[ui.tcIndex].tc });
      break;
    case "join":
      joinRoom((document.getElementById("code") as HTMLInputElement).value);
      break;
    case "join-room":
      joinRoom(d.room!);
      break;
    case "sq":
      if (hasAnnotations()) {
        clearAnnotations();
        if (!canMove() && ui.viewPly === null) {
          renderGame();
          break;
        }
      }
      onSquare(Number(d.sq));
      break;
    case "ply":
      goToPly(Number(d.ply));
      break;
    case "nav":
      navigate(d.to!);
      break;
    case "theme":
      toggleTheme();
      if (ui.screen === "game") renderGame();
      else renderLobby();
      break;
    case "sound":
      showToast(toggleSound() ? "Sound on" : "Sound off");
      if (ui.screen === "game") renderGame();
      else renderLobby();
      break;
    case "res":
      if (!canMove()) return;
      ui.selected = ui.selected?.type === "drop" && ui.selected.kind === d.kind ? null : { type: "drop", kind: d.kind! };
      renderGame();
      break;
    case "promo": {
      const p = ui.promo;
      if (p) sendMove(p.moves.find((m) => m.endsWith(d.k!))!);
      break;
    }
    case "promo-cancel":
      ui.promo = null;
      ui.selected = null;
      renderGame();
      break;
    case "copy-invite":
      if (ui.state) shareInvite(ui.state.room, false);
      break;
    case "share-invite":
      if (ui.state) shareInvite(ui.state.room, true);
      break;
    case "resign":
      if (ui.confirmResign) {
        ui.confirmResign = false;
        conn.send({ t: "resign" });
      } else {
        ui.confirmResign = true;
        renderGame();
      }
      break;
    case "rematch":
      conn.send({ t: "rematch" });
      break;
    case "view-board":
      ui.hideResult = true;
      renderGame();
      break;
    case "show-result":
      ui.hideResult = false;
      renderGame();
      break;
    case "flip":
      ui.flipped = !ui.flipped;
      renderGame();
      break;
    case "leave": {
      if (ui.screen !== "game") break;
      const s = ui.state;
      const live = s && !s.local && s.status.kind === "ongoing" && bothSeated(s) && s.sans.length > 0;
      if (live && !ui.confirmLeave) {
        ui.confirmLeave = true;
        showToast("Leaving now counts as a loss. Tap again to leave.");
        window.setTimeout(() => (ui.confirmLeave = false), 3000);
        break;
      }
      ui.confirmLeave = false;
      conn.send({ t: "leave" });
      saveSession(null);
      goLobby();
      break;
    }
  }
});

document.addEventListener("keydown", (e) => {
  if (ui.screen !== "game" || !ui.state) return;
  const typing = (e.target as HTMLElement)?.closest?.("input, textarea, select");
  if (typing) return;
  if (e.key === "Escape" && (ui.selected || ui.promo || hasAnnotations())) {
    ui.selected = null;
    ui.promo = null;
    clearAnnotations();
    renderGame();
    return;
  }
  const keys: Record<string, string> = { ArrowLeft: "prev", ArrowRight: "next", ArrowUp: "first", ArrowDown: "last", Home: "first", End: "last" };
  if (keys[e.key] && ui.state.sans.length > 0) {
    e.preventDefault();
    navigate(keys[e.key]);
  }
});

// ---------- drag and drop (issue #8) ----------
// Press on one of your pieces (or a reserve piece) and move a few pixels to start
// dragging; release on a highlighted square to play the move. A press without
// movement is an ordinary click, so click-to-move keeps working unchanged.

interface Drag {
  from: NonNullable<Selection>;
  piece: { c: Side; k: string };
  x0: number;
  y0: number;
  active: boolean;
  ghost: HTMLElement | null;
}
let drag: Drag | null = null;
let suppressClick = false;
const DRAG_THRESHOLD = 5;

function dragStartFrom(target: EventTarget | null): Drag["from"] | null {
  const s = ui.state;
  if (!s || !canMove()) return null;
  const el = target as Element | null;
  const res = el?.closest?.("button.res.live") as HTMLElement | null;
  if (res) return { type: "drop", kind: res.dataset.kind! };
  const sq = hitSquare(el);
  if (sq === null) return null;
  const p = s.board[sq];
  if (!p || (p.c === "w" ? "white" : "black") !== s.side || legalFrom(sq).length === 0) return null;
  return { type: "sq", sq };
}

function boardSquareSize(): number {
  const svg = document.querySelector("svg.board");
  return svg ? svg.getBoundingClientRect().width / 8 : 64;
}

function moveGhost(d: Drag, x: number, y: number) {
  if (!d.ghost) return;
  const size = d.ghost.offsetWidth;
  d.ghost.style.transform = `translate(${x - size / 2}px, ${y - size / 2}px)`;
}

function endDrag() {
  drag?.ghost?.remove();
  drag = null;
  ui.dragFrom = null;
  ui.dragOver = null;
}

app.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !e.isPrimary) return;
  const from = dragStartFrom(e.target);
  if (!from) return;
  const s = ui.state!;
  const piece =
    from.type === "sq"
      ? { c: s.board[from.sq]!.c as Side, k: s.board[from.sq]!.k as string }
      : { c: (s.side === "white" ? "w" : "b") as Side, k: from.kind };
  drag = { from, piece, x0: e.clientX, y0: e.clientY, active: false, ghost: null };
  if ((e.target as Element).closest("svg.board")) e.preventDefault(); // no text selection or native image drag
});

window.addEventListener("pointermove", (e) => {
  if (!drag || !e.isPrimary) return;
  if (!drag.active) {
    if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < DRAG_THRESHOLD) return;
    drag.active = true;
    ui.selected = drag.from;
    ui.promo = null;
    ui.dragFrom = drag.from.type === "sq" ? drag.from.sq : null;
    const size = boardSquareSize();
    const ghost = document.createElement("div");
    ghost.className = "drag-ghost";
    ghost.style.width = ghost.style.height = `${size}px`;
    ghost.innerHTML = pieceIcon(drag.piece.c, drag.piece.k);
    document.body.appendChild(ghost);
    drag.ghost = ghost;
    renderGame();
  }
  moveGhost(drag, e.clientX, e.clientY);
  const over = hitSquare(document.elementFromPoint(e.clientX, e.clientY));
  if (over !== ui.dragOver) {
    ui.dragOver = over;
    renderGame();
  }
});

window.addEventListener("pointerup", (e) => {
  if (!drag || e.button !== 0) return;
  const d = drag;
  const wasActive = d.active;
  endDrag();
  if (!wasActive) return; // a plain click: the click handler takes it from here
  suppressClick = true;
  window.setTimeout(() => (suppressClick = false), 0);
  const sq = hitSquare(document.elementFromPoint(e.clientX, e.clientY));
  if (sq !== null && targets().has(sq)) {
    onSquare(sq); // plays the move, or opens the promotion picker
    return;
  }
  // Released on its own square: stay selected, so a click on the target still works.
  // Released anywhere else: cancel.
  if (!(d.from.type === "sq" && sq === d.from.sq)) ui.selected = null;
  renderGame();
});

window.addEventListener("pointercancel", () => {
  if (!drag) return;
  endDrag();
  ui.selected = null;
  if (ui.screen === "game") renderGame();
});

// ---------- planning arrows (right mouse button) ----------

const hitSquare = (target: EventTarget | null): number | null => {
  const el = (target as Element | null)?.closest?.("rect.hit") as SVGElement | null;
  return el ? Number(el.dataset.sq) : null;
};

app.addEventListener("contextmenu", (e) => {
  if ((e.target as Element).closest(".board-wrap")) e.preventDefault();
});

app.addEventListener("pointerdown", (e) => {
  if (e.button !== 2) return;
  const sq = hitSquare(e.target);
  if (sq === null) return;
  e.preventDefault();
  ui.drawing = { from: sq, to: sq, color: colorFor(e) };
});

app.addEventListener("pointermove", (e) => {
  if (!ui.drawing) return;
  const sq = hitSquare(document.elementFromPoint(e.clientX, e.clientY));
  if (sq === null || sq === ui.drawing.to) return;
  ui.drawing.to = sq;
  renderGame();
});

window.addEventListener("pointerup", (e) => {
  if (e.button !== 2 || !ui.drawing) return;
  const end = hitSquare(document.elementFromPoint(e.clientX, e.clientY)) ?? ui.drawing.to;
  toggleMark(ui.drawing.from, end, ui.drawing.color);
  ui.drawing = null;
  if (ui.screen === "game") renderGame();
});

// First paint: show the game shell while rejoining or joining from a link.
if (loadSession() || linkCode()) {
  ui.screen = "game";
  renderGame();
} else {
  renderLobby();
}

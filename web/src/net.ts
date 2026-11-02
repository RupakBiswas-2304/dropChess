// WebSocket connection to the Drop Chess server, with automatic reconnect.

export type ColorName = "white" | "black";
export type Variant = "drop" | "crazy";
export type PieceLetter = "P" | "N" | "B" | "R" | "Q" | "K";

export interface PieceView {
  c: "w" | "b";
  k: PieceLetter;
  promoted: boolean;
}

export type Status =
  | { kind: "ongoing" }
  | { kind: "checkmate"; winner: ColorName }
  | { kind: "stalemate" }
  | { kind: "repetition" }
  | { kind: "fiftymove" }
  | { kind: "insufficient" }
  | { kind: "resigned"; winner: ColorName }
  | { kind: "abandoned"; winner: ColorName }
  | { kind: "timeout"; winner: ColorName };

export interface TimeControl {
  initialMs: number;
  incMs: number;
}

export interface ClockView extends TimeControl {
  white: number;
  black: number;
  running: ColorName | null;
}

export interface PlayerView {
  name: string;
  online: boolean;
  bot?: boolean;
}

export type BotLevel = "easy" | "medium" | "hard";

export interface GameState {
  t: "state";
  room: string;
  variant: Variant;
  local: boolean;
  fen: string;
  board: (PieceView | null)[];
  side: ColorName;
  reserve: Record<ColorName, Record<Exclude<PieceLetter, "K">, number>>;
  legal: string[];
  status: Status;
  check: number | null;
  lastMove: string | null;
  sans: string[];
  players: Record<ColorName, PlayerView | null>;
  rematch: Record<ColorName, boolean>;
  clock: ClockView | null;
  bot: { color: ColorName; level: string; thinking: boolean } | null;
}

export interface RoomSummary {
  id: string;
  variant: Variant;
  host: string;
  open: boolean;
  over: boolean;
  tc: TimeControl | null;
}

export type ServerMsg =
  | GameState
  | { t: "joined"; room: string; token: string; color: ColorName | "both"; variant: Variant; bot?: boolean }
  | { t: "rooms"; rooms: RoomSummary[] }
  | { t: "error"; msg: string }
  | { t: "pong" };

type Listener = (msg: ServerMsg) => void;
type ConnListener = (connected: boolean) => void;

export class Connection {
  private ws: WebSocket | null = null;
  private queue: string[] = [];
  private retry = 0;
  private pinger = 0;
  connected = false;

  constructor(
    private onMessage: Listener,
    private onConnection: ConnListener,
    private onOpen: () => void,
  ) {
    this.open();
  }

  private open() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      // Keep proxies and tunnels (Render, Cloudflare) from closing an idle socket.
      window.clearInterval(this.pinger);
      this.pinger = window.setInterval(() => this.send({ t: "ping" }), 25000);
      this.connected = true;
      this.onConnection(true);
      this.onOpen();
      for (const m of this.queue.splice(0)) ws.send(m);
    };
    ws.onmessage = (ev) => {
      try {
        this.onMessage(JSON.parse(ev.data as string) as ServerMsg);
      } catch (e) {
        console.error("bad server message", e);
      }
    };
    ws.onclose = () => {
      window.clearInterval(this.pinger);
      this.connected = false;
      this.onConnection(false);
      const delay = Math.min(5000, 300 * 2 ** this.retry++);
      setTimeout(() => this.open(), delay);
    };
  }

  send(msg: Record<string, unknown>) {
    const text = JSON.stringify(msg);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(text);
    else this.queue.push(text);
  }
}

// Move sounds (issue #2), synthesised with the Web Audio API so there are no
// audio files to download or license. Browsers only allow audio after the user
// has interacted with the page, so the context is created on the first click.

export type SoundKind = "move" | "drop" | "capture" | "check" | "start" | "end" | "lowtime";

const KEY = "dropchess.sound";
let enabled = true;
try {
  enabled = localStorage.getItem(KEY) !== "off";
} catch {
  /* default on */
}

let ctx: AudioContext | null = null;

function context(): AudioContext | null {
  if (!ctx) {
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
  }
  if (ctx.state === "suspended") void ctx.resume();
  return ctx;
}

// Unlock audio on the first user gesture.
window.addEventListener("pointerdown", () => context(), { once: true, capture: true });

export function soundEnabled(): boolean {
  return enabled;
}

export function toggleSound(): boolean {
  enabled = !enabled;
  try {
    localStorage.setItem(KEY, enabled ? "on" : "off");
  } catch {
    /* lasts for this page only */
  }
  if (enabled) play("move");
  return enabled;
}

/** A short filtered noise burst: the "click" of a piece on wood. */
function knock(c: AudioContext, t: number, freq: number, gain: number, decay: number) {
  const len = Math.floor(c.sampleRate * decay);
  const buf = c.createBuffer(1, len, c.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
  const src = c.createBufferSource();
  src.buffer = buf;
  const filter = c.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.value = freq;
  filter.Q.value = 1.2;
  const g = c.createGain();
  g.gain.value = gain;
  src.connect(filter).connect(g).connect(c.destination);
  src.start(t);
}

/** A soft sine tone with a quick attack and exponential release. */
function tone(c: AudioContext, t: number, freq: number, gain: number, dur: number) {
  const osc = c.createOscillator();
  osc.type = "sine";
  osc.frequency.value = freq;
  const g = c.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  osc.connect(g).connect(c.destination);
  osc.start(t);
  osc.stop(t + dur + 0.02);
}

export function play(kind: SoundKind) {
  if (!enabled) return;
  const c = context();
  if (!c || c.state !== "running") return;
  const t = c.currentTime + 0.005;
  switch (kind) {
    case "move":
      knock(c, t, 1900, 0.9, 0.05);
      tone(c, t, 190, 0.18, 0.06);
      break;
    case "drop":
      // A softer, lower "placing" sound.
      knock(c, t, 1200, 0.7, 0.07);
      tone(c, t, 150, 0.22, 0.09);
      break;
    case "capture":
      knock(c, t, 1500, 1.0, 0.06);
      knock(c, t + 0.045, 2300, 0.7, 0.05);
      tone(c, t, 120, 0.25, 0.1);
      break;
    case "check":
      knock(c, t, 1900, 0.8, 0.05);
      tone(c, t + 0.02, 880, 0.12, 0.12);
      tone(c, t + 0.1, 660, 0.1, 0.14);
      break;
    case "start":
      tone(c, t, 523, 0.1, 0.18);
      tone(c, t + 0.12, 784, 0.1, 0.24);
      break;
    case "end":
      tone(c, t, 784, 0.1, 0.2);
      tone(c, t + 0.14, 659, 0.1, 0.2);
      tone(c, t + 0.28, 523, 0.12, 0.4);
      break;
    case "lowtime":
      tone(c, t, 1000, 0.07, 0.08);
      break;
  }
}

/** The sound for a move, from its SAN (`N@f3`, `Bxe5+`, `R@a3#`). */
export function soundForSan(san: string): SoundKind {
  if (san.endsWith("+") || san.endsWith("#")) return "check";
  if (san.includes("x")) return "capture";
  if (san.includes("@")) return "drop";
  return "move";
}

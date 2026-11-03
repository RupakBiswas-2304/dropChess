// Move planning (issue #4): right-drag between squares draws an arrow,
// right-click on a square circles it. A left click clears everything, and so
// does the next move. Hold Shift for red, Alt or Ctrl for blue.

export type MarkColor = "green" | "red" | "blue";

export interface Arrow {
  from: number;
  to: number;
  color: MarkColor;
}

export interface Circle {
  sq: number;
  color: MarkColor;
}

export const annotations = {
  arrows: [] as Arrow[],
  circles: [] as Circle[],
};

export function hasAnnotations(): boolean {
  return annotations.arrows.length > 0 || annotations.circles.length > 0;
}

export function clearAnnotations() {
  annotations.arrows = [];
  annotations.circles = [];
}

export function colorFor(e: { shiftKey: boolean; altKey: boolean; ctrlKey: boolean }): MarkColor {
  if (e.shiftKey) return "red";
  if (e.altKey || e.ctrlKey) return "blue";
  return "green";
}

/** Adds the mark, or removes it if the same one is already there. */
export function toggleMark(from: number, to: number, color: MarkColor) {
  if (from === to) {
    const i = annotations.circles.findIndex((c) => c.sq === from);
    if (i >= 0 && annotations.circles[i].color === color) annotations.circles.splice(i, 1);
    else if (i >= 0) annotations.circles[i] = { sq: from, color };
    else annotations.circles.push({ sq: from, color });
    return;
  }
  const i = annotations.arrows.findIndex((a) => a.from === from && a.to === to);
  if (i >= 0 && annotations.arrows[i].color === color) annotations.arrows.splice(i, 1);
  else if (i >= 0) annotations.arrows[i] = { from, to, color };
  else annotations.arrows.push({ from, to, color });
}

/**
 * SVG for the marks, in the board's 800x800 coordinate space.
 * `center(sq)` gives a square's centre for the current orientation.
 */
export function annotationsSvg(center: (sq: number) => [number, number], preview?: Arrow | null): string {
  const marker = (color: MarkColor) =>
    `<marker id="ah-${color}" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="mark-${color}-fill"/></marker>`;
  let out = `<defs>${marker("green")}${marker("red")}${marker("blue")}</defs>`;
  for (const c of annotations.circles) {
    const [x, y] = center(c.sq);
    out += `<circle cx="${x}" cy="${y}" r="45" class="mark-circle mark-${c.color}"/>`;
  }
  const arrows = preview ? [...annotations.arrows, preview] : annotations.arrows;
  for (const a of arrows) {
    if (a.from === a.to) continue;
    const [x1, y1] = center(a.from);
    const [x2, y2] = center(a.to);
    // Stop short of the target centre so the head sits inside the square.
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    const ex = x2 - (dx / len) * 22;
    const ey = y2 - (dy / len) * 22;
    out += `<line x1="${x1}" y1="${y1}" x2="${ex}" y2="${ey}" class="mark-arrow mark-${a.color}${a === preview ? " preview" : ""}" marker-end="url(#ah-${a.color})"/>`;
  }
  return `<g class="marks" pointer-events="none">${out}</g>`;
}

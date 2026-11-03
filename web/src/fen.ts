// Reads the server's Crazyhouse-style FEN, so the client can show any earlier
// position of the game (issue #3) without knowing the rules.
//
//   4k3/8/8/8/8/5N2/8/4K3[QRRBBNPPPPPPPPqrrbbnnpppppppp] b - - 0 1
//   board (rank 8 first)  reserve in brackets            side ...
// A "~" after a piece marks a promoted pawn.

import type { ColorName, PieceLetter, PieceView } from "./net";

export type Reserve = Record<Exclude<PieceLetter, "K">, number>;

export interface FenPosition {
  board: (PieceView | null)[]; // index = rank * 8 + file, a1 = 0
  reserve: Record<ColorName, Reserve>;
  side: ColorName;
}

const emptyReserve = (): Reserve => ({ P: 0, N: 0, B: 0, R: 0, Q: 0 });

export function parseFen(fen: string): FenPosition {
  const [placementFull, side = "w"] = fen.split(" ");
  const bracket = placementFull.indexOf("[");
  const placement = bracket >= 0 ? placementFull.slice(0, bracket) : placementFull;
  const reserveStr = bracket >= 0 ? placementFull.slice(bracket + 1, placementFull.indexOf("]")) : "";

  const board: (PieceView | null)[] = new Array(64).fill(null);
  placement.split("/").forEach((row, i) => {
    const rank = 7 - i;
    let file = 0;
    for (let j = 0; j < row.length; j++) {
      const ch = row[j];
      if (ch >= "1" && ch <= "8") {
        file += Number(ch);
        continue;
      }
      const promoted = row[j + 1] === "~";
      if (promoted) j++;
      board[rank * 8 + file] = {
        c: ch === ch.toUpperCase() ? "w" : "b",
        k: ch.toUpperCase() as PieceLetter,
        promoted,
      };
      file++;
    }
  });

  const reserve = { white: emptyReserve(), black: emptyReserve() };
  for (const ch of reserveStr) {
    const k = ch.toUpperCase() as Exclude<PieceLetter, "K">;
    if (!(k in reserve.white)) continue;
    reserve[ch === k ? "white" : "black"][k]++;
  }

  return { board, reserve, side: side === "b" ? "black" : "white" };
}

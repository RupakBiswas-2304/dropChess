"""Reference Game on top of python-chess's CrazyhouseBoard.

python-chess supplies move generation (drops, check evasion by drop, pins, en
passant). Everything that differs from Crazyhouse is done here:
  * drops are limited to the mover's own half,
  * in the Drop variant captured pieces are discarded,
  * the 50-move clock, repetition, insufficient material and the ply cap are
    tracked here, with the same definitions as engine/src/lib.rs.
"""

from __future__ import annotations

import chess
import chess.variant
import numpy as np

from .encoding import (
    DROP_PIECES,
    NUM_PLANES,
    PIECES,
    PLANE_SCALE,
    decode_action,
    encode_move_action,
    flip,
)

START_FEN = "4k3/8/8/8/8/8/8/4K3[QRRBBNNPPPPPPPPqrrbbnnpppppppp] w - - 0 1"
MAX_PLIES = 300

_PT = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
_LETTER_PT = {v: k for k, v in _PT.items()}


class _DropBoard(chess.variant.CrazyhouseBoard):
    """Crazyhouse board where captured pieces leave the game."""

    def _push_capture(self, move, capture_square, piece_type, was_promoted):
        return None


def _own_half(sq: int, white: bool) -> bool:
    return chess.square_rank(sq) <= 3 if white else chess.square_rank(sq) >= 4


class Game:
    def __init__(self, board: chess.variant.CrazyhouseBoard, variant: str, halfmove: int, history: list, plies: int):
        self.board = board
        self.variant = variant
        self.halfmove = halfmove
        self.history = history  # repetition keys, current position last
        self.plies = plies
        self.max_plies = MAX_PLIES
        self._legal = None

    # ---- construction -------------------------------------------------
    @staticmethod
    def new(variant: str) -> "Game":
        return Game.from_fen(START_FEN, variant)

    @staticmethod
    def from_fen(fen: str, variant: str) -> "Game":
        if variant not in ("drop", "crazy"):
            raise ValueError("variant must be 'drop' or 'crazy'")
        parts = fen.split()
        cls = _DropBoard if variant == "drop" else chess.variant.CrazyhouseBoard
        # python-chess only accepts a legal ep square in a FEN; set it afterwards.
        ep = parts[3] if len(parts) > 3 else "-"
        safe = parts[:]
        while len(safe) < 6:
            safe.append(["w", "-", "-", "0", "1"][len(safe) - 1])
        safe[2] = "-"
        safe[3] = "-"
        board = cls(" ".join(safe))
        if ep != "-":
            board.ep_square = chess.parse_square(ep)
        g = Game(board, variant, int(safe[4]), [], 0)
        g.history.append(g._key())
        return g

    def clone(self) -> "Game":
        g = Game(self.board.copy(stack=False), self.variant, self.halfmove, list(self.history), self.plies)
        g.max_plies = self.max_plies
        return g

    # ---- FEN in the same format as the Rust engine ----------------------
    def _placement(self) -> str:
        b = self.board
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for file in range(8):
                sq = rank * 8 + file
                p = b.piece_at(sq)
                if p is None:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                c = _PT[p.piece_type]
                row += c if p.color == chess.WHITE else c.lower()
                if b.promoted & chess.BB_SQUARES[sq]:
                    row += "~"
            if empty:
                row += str(empty)
            rows.append(row)
        res = ""
        for color in (chess.WHITE, chess.BLACK):
            for letter in "QRBNP":
                n = b.pockets[color].count(_LETTER_PT[letter])
                res += (letter if color == chess.WHITE else letter.lower()) * n
        return "/".join(rows) + "[" + res + "]"

    def _ep(self) -> str:
        return chess.square_name(self.board.ep_square) if self.board.ep_square is not None else "-"

    def _key(self) -> str:
        return f"{self._placement()} {'w' if self.board.turn else 'b'} {self._ep()}"

    def fen(self) -> str:
        return f"{self._placement()} {'w' if self.board.turn else 'b'} - {self._ep()} {self.halfmove} {self.board.fullmove_number}"

    # ---- rules ----------------------------------------------------------
    def to_play(self) -> int:
        return 0 if self.board.turn == chess.WHITE else 1

    def legal_moves(self) -> list[chess.Move]:
        if self._legal is None:
            white = self.board.turn == chess.WHITE
            self._legal = [m for m in self.board.legal_moves if not m.drop or _own_half(m.to_square, white)]
        return self._legal

    def _move_action(self, m: chess.Move) -> int:
        black = self.board.turn == chess.BLACK
        if m.drop:
            return encode_move_action("drop", DROP_PIECES.index(_PT[m.drop]), m.to_square, None, black)
        promo = _PT[m.promotion] if m.promotion else None
        return encode_move_action("move", m.from_square, m.to_square, promo, black)

    def legal_actions(self) -> list[int]:
        return sorted(self._move_action(m) for m in self.legal_moves())

    def legal_uci(self) -> list[str]:
        return sorted(m.uci() for m in self.legal_moves())

    def apply_uci(self, uci: str) -> None:
        self.apply(self.uci_to_action(uci))

    def _action_move(self, a: int) -> chess.Move:
        for m in self.legal_moves():
            if self._move_action(m) == a:
                return m
        raise ValueError(f"illegal action {a}")

    def apply(self, action: int) -> None:
        self.push(self._action_move(action))

    def push(self, m: chess.Move) -> None:
        b = self.board
        reset = bool(m.drop) or b.is_capture(m) or b.piece_type_at(m.from_square) == chess.PAWN
        b.push(m)
        self.halfmove = 0 if reset else self.halfmove + 1
        self.plies += 1
        self.history.append(self._key())
        self._legal = None

    def insufficient_material(self) -> bool:
        b = self.board
        if len(b.pockets[chess.WHITE]) or len(b.pockets[chess.BLACK]):
            return False
        minors = 0
        for _, p in b.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            if p.piece_type in (chess.KNIGHT, chess.BISHOP):
                minors += 1
            else:
                return False
        return minors <= 1

    def status(self) -> str:
        """'ongoing', 'checkmate', 'stalemate', 'repetition', 'fifty', 'insufficient', 'maxplies'."""
        if not self.legal_moves():
            return "checkmate" if self.board.is_check() else "stalemate"
        if self.history.count(self.history[-1]) >= 3:
            return "repetition"
        if self.halfmove >= 100:
            return "fifty"
        if self.insufficient_material():
            return "insufficient"
        if self.plies >= self.max_plies:
            return "maxplies"
        return "ongoing"

    def is_terminal(self) -> bool:
        return self.status() != "ongoing"

    def result(self) -> int:
        if self.status() == "checkmate":
            return -1 if self.board.turn == chess.WHITE else 1
        return 0

    # ---- notation --------------------------------------------------------
    def action_to_uci(self, action: int) -> str:
        return self._action_move(action).uci()

    def uci_to_action(self, uci: str) -> int:
        return self._move_action(chess.Move.from_uci(uci))

    # ---- encoding --------------------------------------------------------
    def encode_raw(self) -> np.ndarray:
        """uint8 [31, 8, 8]; multiply by PLANE_SCALE to get encode()."""
        b = self.board
        black = b.turn == chess.BLACK
        us = b.turn
        x = np.zeros((NUM_PLANES, 64), dtype=np.uint8)
        for sq, p in b.piece_map().items():
            s = flip(sq, black)
            base = 0 if p.color == us else 6
            x[base + PIECES.index(_PT[p.piece_type]), s] = 1
            if b.promoted & chess.BB_SQUARES[sq]:
                x[22 if p.color == us else 23, s] = 1
        for i, letter in enumerate(DROP_PIECES):
            x[12 + i, :] = b.pockets[us].count(_LETTER_PT[letter])
            x[17 + i, :] = b.pockets[not us].count(_LETTER_PT[letter])
        seen_before = self.history[:-1].count(self.history[-1])
        if seen_before >= 1:
            x[24, :] = 1
        if seen_before >= 2:
            x[25, :] = 1
        if b.ep_square is not None:
            x[26, flip(b.ep_square, black)] = 1
        if not black:
            x[27, :] = 1
        if self.variant == "crazy":
            x[28, :] = 1
        x[29, :] = min(self.halfmove, 255)
        x[30, :] = 1
        return x.reshape(NUM_PLANES, 8, 8)

    def encode(self) -> np.ndarray:
        return self.encode_raw().astype(np.float32) * PLANE_SCALE[:, None, None]

"""Reference Drop Chess engine in pure Python, built on python-chess.

It is slow and is not used in the training loop. Its job is to give an
independent implementation of the rules, the state encoding and the action
mapping, so the Rust engine (module `dropchess`) can be checked against it.

Interface (shared with the Rust module):

    Game.new(variant) / Game.from_fen(fen, variant)
    fen() clone() to_play() legal_actions() apply(a) is_terminal() result()
    encode() action_to_uci(a) uci_to_action(uci)
"""

from .game import Game, START_FEN
from .encoding import (
    NUM_PLANES,
    NUM_ACTIONS,
    PLANE_SCALE,
    encode_move_action,
    mirror_action,
)

__all__ = [
    "Game",
    "START_FEN",
    "NUM_PLANES",
    "NUM_ACTIONS",
    "PLANE_SCALE",
    "encode_move_action",
    "mirror_action",
]

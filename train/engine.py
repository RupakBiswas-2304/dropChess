"""Picks the rules engine: ENGINE=rust (default, the compiled `dropchess` module)
or ENGINE=py (the slow python-chess reference in dropchess_py/).

Training always uses the Rust module, because batched MCTS lives there; the
switch matters for tests and for poking at positions by hand.
"""

import os

ENGINE = os.environ.get("ENGINE", "rust")


def game_class(name: str | None = None):
    name = name or ENGINE
    if name == "py":
        from dropchess_py import Game
        return Game
    if name == "rust":
        try:
            import dropchess
        except ImportError as e:
            raise ImportError(
                "The Rust module isn't built. From train/: maturin develop --release -m ../pyengine/Cargo.toml"
            ) from e
        return dropchess.Game
    raise ValueError(f"ENGINE must be 'py' or 'rust', got {name!r}")


Game = game_class()

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import game_class  # noqa: E402

ENGINES = ["py", "rust"]


@pytest.fixture(params=ENGINES)
def Game(request):
    return game_class(request.param)

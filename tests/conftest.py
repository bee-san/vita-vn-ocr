"""Shared test setup: the repo's script directories on sys.path, and the pinned CI font."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("ci", "synth", "train"):
    sys.path.insert(0, str(ROOT / d))


def _own_shapes(tm):
    """The shape names train/textdet_model.py defines itself (its _shape(...) calls)."""
    import re
    src = Path(tm.__file__).read_text(encoding="utf-8")
    return frozenset(re.findall(r'^_shape\("([^"]+)"', src, flags=re.M))


@pytest.fixture(autouse=True)
def _textdet_shapes_reset():
    """train/textspot2_shapes.register() adds textspot2's shapes to train/textdet_model.SHAPES, and #97's tests
    compare that dict with their own protocol; so after every test SHAPES holds textdet_model's own shapes only
    (every register() call in the tests is inside a test, so no test relies on another's)."""
    yield
    tm = sys.modules.get("textdet_model")
    if tm is not None:
        own = _own_shapes(tm)
        for name in [k for k in tm.SHAPES if k not in own]:
            del tm.SHAPES[name]


@pytest.fixture(scope="session")
def font():
    """Path of the pinned OFL font (ci/fonts.py). Without network access this skips locally,
    but fails in CI (GitHub Actions sets CI=true), so a CI run never skips the render tests."""
    import fonts
    try:
        return fonts.ensure_font()
    except RuntimeError as e:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"pinned font unavailable ({e}); run: python ci/fonts.py")

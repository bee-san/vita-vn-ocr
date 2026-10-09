"""Shared test setup: the repo's script directories on sys.path, and the pinned CI font."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("ci", "synth", "train"):
    sys.path.insert(0, str(ROOT / d))


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

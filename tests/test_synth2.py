"""synth2 renderer pieces: the mask line finder, kinsoku wrapping, and (where the local
data/ exists) the label invariants of rendered crops."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))

import finders  # noqa: E402
import render  # noqa: E402


def test_mask_finder_geometry_and_furigana():
    ink = np.zeros((120, 300), bool)
    ink[10:16, 40:60] = True          # furigana: 6 rows, under 0.55x the line height -> dropped
    ink[20:44, 30:250] = True         # line 1: 24 rows
    ink[60:84, 30:180] = True         # line 2
    boxes = finders.find_lines_mask(ink)
    assert boxes == [(26, 16, 228, 32), (26, 56, 158, 32)]   # 4 px pad left/right/top, 5 below


def test_mask_finder_merges_close_furigana_and_rejects_tall_runs():
    ink = np.zeros((100, 200), bool)
    ink[8:14, 20:40] = True           # furigana 1 px above the line: part of the line's run
    ink[15:39, 10:150] = True
    assert finders.find_lines_mask(ink) == [(6, 4, 148, 39)]
    tall = np.zeros((120, 50), bool)
    tall[5:94, 5:40] = True           # one run of 89 rows (> 88): the region is rejected, as in common.py
    assert finders.find_lines_mask(tall) is None
    tall[93] = False                  # 88 rows: still a line
    assert finders.find_lines_mask(tall) == [(1, 1, 43, 96)]


def test_mask_finder_is_polarity_agnostic_and_grad_finder_agrees():
    for text, box in ((235, 30), (20, 225)):      # light on dark, dark on light
        img = np.full((60, 200, 3), box, np.uint8)
        img[20:40, 20:180:4] = text                  # glyph-like vertical strokes
        ink = np.zeros((60, 200), bool)
        ink[20:40, 20:180:4] = True
        assert finders.find_lines_mask(ink) == [(16, 16, 165, 28)]
        (x, y, w, h), = finders.find_lines_grad(img)
        assert abs(x - 16) <= 5 and abs(y - 16) <= 3 and abs(w - 165) <= 8 and abs(h - 28) <= 4


def test_wrap_kinsoku():
    rng = np.random.default_rng(0)
    text = "「あいうえおかきくけこ」、さしすせそたち……つてとなにぬねの。"
    for cpl in range(4, 14):
        lines = render.wrap(text, cpl, 99, rng)
        assert "".join(lines) == text
        for ln in lines[1:]:
            assert ln[0] not in render.NO_START, (cpl, lines)
        for ln in lines[:-1]:
            assert ln[-1] not in render.NO_END, (cpl, lines)
            assert len(ln) <= cpl + 2                 # at most two hanging characters
        assert not any(a.endswith("…") and b.startswith("…") for a, b in zip(lines, lines[1:]))


@pytest.fixture(scope="module")
def resources():
    if not (render.DATA / "corpus" / "dialogue.txt").exists() or not list((render.DATA / "fonts").glob("*/*.ttf")):
        pytest.skip("needs the local data/ (corpus and fonts)")
    render.init(20_000)
    return render.R


def test_rendered_crops_have_clean_labels(resources):
    rng = np.random.default_rng(1)
    n = 0
    for _ in range(200):
        r = render.render_region(rng)
        if r is None:
            continue
        img, layout, ink, meta = r
        assert img.dtype == np.uint8 and img.shape[:2] == ink.shape
        for g, label, (x, y, w, h), line in render.extract(img, layout, ink, "mask", rng) or []:
            n += 1
            assert g.shape == (h, w) and h >= 8 and w >= 8
            assert label and "　" not in label and set(label) <= resources.vocab_set
    assert n > 150

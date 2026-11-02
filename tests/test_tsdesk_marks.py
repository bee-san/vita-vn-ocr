"""textspot-desktop round 2 (#173): train/tsdesk_det.py --marks (add_marks, trim_bands): single marks drawn as
background before a positive's start or between two side-by-side positives, never over a label or unlabelled ink, and
the trainer bit for bit unchanged with the option off. No data: synthetic windows only."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "tools", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("torch")
pytest.importorskip("cv2")

import tsdesk_det as TD  # noqa: E402


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _window(lines, bg=40, ink=220):
    img = np.full((256, 512), bg, np.uint8)
    for x0, y0, x1, y1 in lines:
        img[y0:y1, x0:x1:3] = ink                     # a line's strokes
    return img


def test_marks_go_before_the_start_and_off_the_labels():
    rng = np.random.default_rng(0)
    pos = [[100, 120, 400, 144, 0, 1.0, 1]]
    ign = [[10, 10, 60, 40]]
    n_marks = 0
    for _ in range(80):
        img = _window([pos[0][:4]])
        before = img.copy()
        drawn = TD.add_marks(img, pos, ign, rng)
        assert (img[120:144, 100:400] == before[120:144, 100:400]).all()          # the line untouched
        if not drawn:
            assert (img == before).all()
        for b in drawn:
            n_marks += 1
            assert not _overlaps(b, pos[0]) and not _overlaps(b, ign[0])
            lh = 24
            assert b[2] <= 100 and b[0] >= 100 - (0.9 + 0.9) * lh - 3                 # before the start, near it
            assert min(b[3], 144) - max(b[1], 120) > 0                                  # on the line's row
            assert (img[b[1]:b[3], b[0]:b[2]] != before[b[1]:b[3], b[0]:b[2]]).any()
    assert n_marks >= 40


def test_marks_between_side_by_side_buttons(monkeypatch):
    monkeypatch.setitem(TD.MARKS, "before_p", 0.0)
    rng = np.random.default_rng(1)
    pos = [[40, 120, 160, 144, 0, 1.0, 1], [300, 120, 420, 144, 0, 1.0, 1]]
    n = 0
    for _ in range(40):
        img = _window([p[:4] for p in pos])
        for b in TD.add_marks(img, pos, [], rng):
            n += 1
            assert b[0] >= 160 and b[2] <= 300                                          # in the gap
            assert min(b[3], 144) - max(b[1], 120) > 0
    assert n >= 20
    # two lines on different rows are no pair: the marks go before a start
    pos2 = [[40, 60, 160, 84, 0, 1.0, 1], [300, 120, 420, 144, 0, 1.0, 1]]
    for _ in range(20):
        img = _window([p[:4] for p in pos2])
        for b in TD.add_marks(img, pos2, [], rng):
            assert b[2] <= 40 or (b[2] <= 300 and b[0] >= 300 - 1.8 * 24 - 3)


def test_marks_need_flat_background():
    rng = np.random.default_rng(2)
    pos = [[100, 120, 400, 144, 0, 1.0, 1]]
    noise = np.random.default_rng(3).integers(0, 255, (256, 512), dtype=np.uint8)
    for _ in range(30):
        img = noise.copy()
        img[120:144, 100:400] = 40
        img[120:144, 100:400:3] = 220
        assert TD.add_marks(img, pos, [], rng) == [] and (img[:, :100] == noise[:, :100]).all()


def test_marks_skip_vertical_and_tiny_lines():
    rng = np.random.default_rng(4)
    for pos in ([[100, 20, 124, 240, 1, 1.0, 1]], [[100, 120, 400, 124, 0, 1.0, 1]]):
        img = _window([pos[0][:4]])
        assert TD.add_marks(img, pos, [], rng) == []


def test_trim_bands_frees_the_marks_columns_only():
    bands = [[60, 120, 100, 144], [400, 120, 424, 144]]
    out = TD.trim_bands(bands, [[70, 124, 82, 140]])
    assert out == [[60, 120, 70, 144], [82, 120, 100, 144], [400, 120, 424, 144]]
    assert TD.trim_bands(bands, []) == bands
    assert TD.trim_bands([[60, 120, 100, 144]], [[50, 100, 120, 160]]) == []


class _Sources:
    """A stand-in for the trainer's Sources: one fixed sample of a non-exact source."""

    def __init__(self):
        self.canvas = _window([[100, 120, 400, 144], [100, 160, 300, 184]])
        self.smp = {"lines": [[100, 120, 400, 144, 0, "text", 1.0], [100, 160, 300, 184, 0, "text", 1.0]],
                    "ignore": [], "kind": "pc"}

    def pick(self, rng):
        rng.random()
        return "desk_dense", self.canvas, self.smp


def _fake_window(canvas, smp, src, rng, sources, scale, aug=True, ruby_filter=False):
    rng.random()
    pos = [[*ln[:4], 0, 1.0, 1] for ln in smp["lines"]]
    return np.array(canvas, copy=True), pos, []


def test_make_batch_with_marks_off_is_unchanged(monkeypatch):
    import desktop_det as DD
    monkeypatch.setattr(DD, "make_window_ms", _fake_window)
    cfg = {"stride": 8, "batch": 4, "scale": (1.0, 1.0), "aug": True, "recipe": "base", "end_ignore": 1.0,
           "ornaments": 0.0}
    src = _Sources()
    outs, states = [], []
    for extra in ({}, {"marks": 0.0}, {"marks": 1.0}):
        rng = np.random.default_rng(7)
        outs.append(TD.make_batch(src, rng, dict(cfg, **extra)))
        states.append(rng.bit_generator.state["state"]["state"])
    for a, b in zip(outs[0], outs[1]):
        assert np.array_equal(a.numpy(), b.numpy())
    assert states[0] == states[1]
    # on: the windows get marks before the lines' starts, as background, and the end bands lose their columns
    x_on, x_off = outs[2][0].numpy(), outs[0][0].numpy()
    assert (x_on != x_off).any() and states[2] != states[0]
    K_on, v_on = outs[2][1].numpy(), outs[2][8].numpy()
    diff = (x_on[:, 0] != x_off[:, 0])
    cols = np.flatnonzero(diff.any(axis=(0, 1)))
    assert cols.max() < 100                                                   # every change lies before the starts
    cell = (diff.reshape(4, 32, 8, 64, 8).any(axis=(2, 4)))                   # stride-8 cells a mark touches
    cell[:, :, 100 // 8:] = False             # a mark within 8 px of the start shares the line's first cell (K = 1)
    assert cell.any() and (K_on[cell] == 0).all() and v_on[cell].any()        # background, and some of it scored
    v_off = outs[0][8].numpy()
    assert v_on[cell].sum() > v_off[cell].sum()                               # the trimmed bands score the marks


def test_glyph_sets(monkeypatch):
    """v2 (the default) never draws ・ or ※; v1 is the first set; the trainer's cfg picks the set."""
    assert TD.MARK_GLYPHS["v1"] == ("nakaguro", "bullet", "diamond", "square", "triangle", "kome")
    assert TD.MARK_GLYPHS["v2"] == ("bullet", "diamond", "square", "triangle") == TD.MARKS["glyphs"]
    seen = {}
    real = TD._draw_mark

    def spy(img, kind, cx, cy, s, value, rng):
        seen[kind] = seen.get(kind, 0) + 1
        return real(img, kind, cx, cy, s, value, rng)
    monkeypatch.setattr(TD, "_draw_mark", spy)
    monkeypatch.setitem(TD.MARKS, "icon_p", 0.0)
    pos = [[100, 120, 400, 144, 0, 1.0, 1]]
    for glyphs, want_absent in ((None, {"nakaguro", "kome"}), (TD.MARK_GLYPHS["v2"], {"nakaguro", "kome"})):
        seen.clear()
        rng = np.random.default_rng(0)
        for _ in range(300):
            TD.add_marks(_window([pos[0][:4]]), pos, [], rng, glyphs=glyphs)
        assert seen and not (set(seen) & want_absent)
    seen.clear()
    rng = np.random.default_rng(0)
    for _ in range(300):
        TD.add_marks(_window([pos[0][:4]]), pos, [], rng, glyphs=TD.MARK_GLYPHS["v1"])
    assert {"nakaguro", "kome"} <= set(seen)


def test_make_batch_marks_glyph_set(monkeypatch):
    import desktop_det as DD
    monkeypatch.setattr(DD, "make_window_ms", _fake_window)
    cfg = {"stride": 8, "batch": 4, "scale": (1.0, 1.0), "aug": True, "recipe": "base", "end_ignore": 1.0,
           "ornaments": 0.0, "marks": 1.0}
    src = _Sources()
    default = TD.make_batch(src, np.random.default_rng(7), cfg)
    v2 = TD.make_batch(src, np.random.default_rng(7), dict(cfg, marks_glyphs="v2"))
    v1 = TD.make_batch(src, np.random.default_rng(7), dict(cfg, marks_glyphs="v1"))
    assert all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip(default, v2))
    assert not np.array_equal(default[0].numpy(), v1[0].numpy())

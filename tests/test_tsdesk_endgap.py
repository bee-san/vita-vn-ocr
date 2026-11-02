"""textspot-desktop round 2 (#173): train/tsdesk_det.py --end-ignore-gap (end_bands' gap cap): an end band never covers
more than G of the gap to the next positive on the same row, the rest of the gap is background, and with the option
off every band and batch is as before. No data: synthetic windows only."""
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


def _old_end_bands(pos, f):
    """end_bands as it was before the gap cap (the off path's reference)."""
    out = []
    for x0, y0, x1, y1, vertical, *_ in pos:
        if vertical:
            d = f * (x1 - x0)
            out += [[x0, int(y0 - d), x1, y0], [x0, y1, x1, int(y1 + d + 0.999)]]
        else:
            d = f * (y1 - y0)
            out += [[int(x0 - d), y0, x0, y1], [x1, y0, int(x1 + d + 0.999), y1]]
    return out


def test_off_path_is_the_old_bands():
    rng = np.random.default_rng(0)
    for _ in range(200):
        pos = []
        for _ in range(int(rng.integers(1, 6))):
            x0, y0 = int(rng.integers(0, 400)), int(rng.integers(0, 200))
            v = int(rng.random() < 0.2)
            w, h = (int(rng.integers(8, 30)), int(rng.integers(30, 120))) if v else (int(rng.integers(30, 200)),
                                                                                     int(rng.integers(8, 30)))
            pos.append([x0, y0, x0 + w, y0 + h, v, 1.0, 1])
        f = float(rng.uniform(0.3, 1.5))
        assert TD.end_bands(pos, f) == TD.end_bands(pos, f, 0.0) == _old_end_bands(pos, f)


def test_gap_caps_the_bands_between_two_items_on_a_row():
    a = [100, 100, 200, 124, 0, 1.0, 1]
    b = [260, 100, 360, 124, 0, 1.0, 1]                   # 60 px to its right on the same row
    assert TD.end_bands([a, b], 1.0) == [[76, 100, 100, 124], [200, 100, 224, 124], [236, 100, 260, 124],
                                         [360, 100, 384, 124]]
    assert TD.end_bands([a, b], 1.0, 0.25) == [[76, 100, 100, 124], [200, 100, 215, 124], [245, 100, 260, 124],
                                               [360, 100, 384, 124]]
    # the cap never lengthens a band: a wide gap keeps the plain f x height band
    far = [600, 100, 700, 124, 0, 1.0, 1]
    assert TD.end_bands([a, far], 1.0, 0.25) == _old_end_bands([a, far], 1.0)
    # another row (overlap under half the smaller height) does not cap
    below = [210, 113, 300, 137, 0, 1.0, 1]               # overlaps a's rows by 11 of 24 px
    assert TD.end_bands([a, below], 1.0, 0.25) == _old_end_bands([a, below], 1.0)


def test_gap_touching_or_overlapping_neighbours_leave_no_band():
    a = [100, 100, 200, 124, 0, 1.0, 1]
    touch = [200, 102, 260, 122, 0, 1.0, 1]               # starts where a ends
    bands = TD.end_bands([a, touch], 1.0, 0.25)
    assert [76, 100, 100, 124] in bands and not any(b[0] == 200 and b[1] == 100 for b in bands)
    assert not any(b[2] == 200 and b[1] == 102 for b in bands)            # touch's start band is gone too
    assert [260, 102, 280, 122] in bands


def test_gap_in_a_column_of_vertical_lines():
    a = [100, 50, 124, 150, 1, 1.0, 1]
    b = [100, 190, 124, 300, 1, 1.0, 1]                   # 40 px below it in the same column
    assert TD.end_bands([a, b], 1.0, 0.5) == [[100, 26, 124, 50], [100, 150, 124, 170], [100, 170, 124, 190],
                                              [100, 300, 124, 324]]


class _Sources:
    """One fixed sample of a non-exact source: two buttons on a row 40 px apart (the plain end bands, 24 px each,
    cover the whole gap) and a line below."""

    def __init__(self):
        img = np.full((256, 512), 40, np.uint8)
        for x0, y0, x1, y1 in ((60, 60, 160, 84), (200, 60, 300, 84), (60, 140, 400, 164)):
            img[y0:y1, x0:x1:3] = 220
        self.canvas = img
        self.smp = {"lines": [[60, 60, 160, 84, 0, "text", 1.0], [200, 60, 300, 84, 0, "text", 1.0],
                              [60, 140, 400, 164, 0, "text", 1.0]], "ignore": [], "kind": "pc"}

    def pick(self, rng):
        rng.random()
        return "desk_dense", self.canvas, self.smp


def _fake_window(canvas, smp, src, rng, sources, scale, aug=True, ruby_filter=False):
    rng.random()
    return np.array(canvas, copy=True), [[*ln[:4], 0, 1.0, 1] for ln in smp["lines"]], []


def test_make_batch_gap_off_is_unchanged_and_on_frees_the_gap(monkeypatch):
    import desktop_det as DD
    monkeypatch.setattr(DD, "make_window_ms", _fake_window)
    cfg = {"stride": 8, "batch": 2, "scale": (1.0, 1.0), "aug": True, "recipe": "base", "end_ignore": 1.0,
           "ornaments": 0.0}
    src = _Sources()
    outs = []
    for extra in ({}, {"end_ignore_gap": 0.0}, {"end_ignore_gap": 0.25}):
        outs.append(TD.make_batch(src, np.random.default_rng(3), dict(cfg, **extra)))
    for a, b in zip(outs[0], outs[1]):
        assert np.array_equal(a.numpy(), b.numpy())
    v_off, v_on = outs[0][8].numpy(), outs[2][8].numpy()
    k_on = outs[2][9].numpy()
    assert not (v_off & ~v_on).any()                        # nothing scored before is ignored now
    freed = v_on & ~v_off
    assert freed.any() and not (freed & k_on).any()
    ys, xs = np.nonzero(freed[0])
    assert ys.min() >= 60 // 8 and ys.max() <= 83 // 8      # on the buttons' row
    assert xs.min() * 8 >= 160 and (xs.max() + 1) * 8 <= 200   # inside the gap between them
    for a, b in zip(outs[0][:8], outs[2][:8]):              # the image and the other targets are the same
        assert np.array_equal(a.numpy(), b.numpy())

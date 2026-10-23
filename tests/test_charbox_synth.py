"""eval/charbox_synth.py's geometry helpers (no renderer assets needed): the canvas -> region mapping of a pen
position through PIL's own resize, the name pen reconstruction and the typewriter ramp factor."""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import charbox_synth as C  # noqa: E402


def test_nearest_matches_pil_pixel_spans():
    # pixel fonts on the Vita: a canvas of ceil(w / 2) columns, NEAREST to w (odd w samples unevenly)
    for w in range(9, 70):
        wc = math.ceil(w / 2)
        edges = C.resample_positions(list(range(wc + 1)), w, wc, Image.NEAREST)
        for j in range(wc):
            a = np.zeros((1, wc), np.uint8)
            a[0, j] = 255
            on = np.flatnonzero(np.asarray(Image.fromarray(a).resize((w, 1), Image.NEAREST))[0])
            assert (edges[j], edges[j + 1]) == ((on[0], on[-1] + 1) if len(on) else (edges[j], edges[j]))


def test_box_and_bilinear_match_pil_step_response():
    rng = np.random.default_rng(0)
    for resample in (Image.BOX, Image.BILINEAR):
        for _ in range(100):
            w = int(rng.integers(40, 300))
            ss = float(rng.choice([2.0, 32 / 25, 48 / 23, 24 / 17]))
            wc = math.ceil(w * ss)
            c = float(rng.uniform(3, wc - 3))
            cov = np.clip(c - np.arange(wc), 0, 1)
            a = (cov * 255 + 0.5).astype(np.uint8)[None, :].repeat(3, 0)
            out = np.asarray(Image.fromarray(a).resize((w, 3), resample), np.float64)[1] / 255
            assert abs(out.sum() - C.resample_positions([c], w, wc, resample)[0]) < 0.01


def test_identity_and_beyond_canvas():
    assert list(C.resample_positions([0.0, 3.5, 10.0], 10, 10, Image.BOX)) == [0.0, 3.5, 10.0]
    got = C.resample_positions([-4.0, 0.0, 200.0, 204.0], 100, 200, Image.BOX)
    assert abs(got[1]) < 1e-6 and abs(got[2] - 100) < 1e-6
    assert abs(got[0] + 2) < 1e-6 and abs(got[3] - 102) < 1e-6


class _Font:
    def getlength(self, ch):
        return 24.0 if ch == C.SPACE else {"あ": 22.0, "い": 20.0}.get(ch, 22.0)


def test_name_pens_rebuilds_full_width_spaces():
    f = _Font()
    # "あ　い" drawn at x = 10 without tracking: あ at 10, the space at 32, い at 56, ending at 76
    calls = [(10.0, 40.0, "あ", f, 0, False, False), (56.0, 40.0, "い", f, 0, False, False)]
    disp, pens = C.name_pens(calls, 1.0, 76.0)
    assert disp == "あ" + C.SPACE + "い" and pens == [10.0, 32.0, 56.0]


def test_ramp_factors():
    cp = np.array([[0.0, 0.5, 1.0]], np.float32)
    ref = cp.copy()
    ref[0, 1:] *= np.float32(0.25)
    fac = C.ramp_factors({"refs": ([ref], None, None), "copies": ([cp], None, None)})
    assert fac["s1"] is None and np.allclose(fac["fill"][0], [[1.0, 0.25, 0.25]])
    assert C.ramp_factors({"refs": ([cp], None, None), "copies": ([cp.copy()], None, None)}) is None

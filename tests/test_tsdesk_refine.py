"""textspot-desktop's box refinement (issue #173, tools/tsdesk_refine.py, desktop/vocr_tsd_refine.c): the C version
gives the Python twin's boxes bit for bit (plain C and AVX2 builds) on random synthetic lines, and each step and guard
does what the rule says. Synthetic images only (no private data)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import tsdesk_refine as TR  # noqa: E402

cv2 = pytest.importorskip("cv2")


# ----------------------------------------------------------------------------- synthetic lines

def _glyphs(rng, img, x, y, h, n, fill, thick=(1, 3)):
    """n glyph-like cells of h x h px from (x, y): 2-4 random strokes each, inside the cell's inner 80%."""
    for k in range(n):
        cx = x + k * h
        for _ in range(rng.integers(2, 5)):
            p0 = (int(cx + h * rng.uniform(0.1, 0.9)), int(y + h * rng.uniform(0.1, 0.9)))
            p1 = (int(cx + h * rng.uniform(0.1, 0.9)), int(y + h * rng.uniform(0.1, 0.9)))
            cv2.line(img, p0, p1, int(fill), int(rng.integers(thick[0], thick[1] + 1)), cv2.LINE_AA)
        # a horizontal stroke near the cell's top and one near its bottom, so every glyph spans the cell's rows
        cv2.line(img, (int(cx + 0.2 * h), int(y + 0.12 * h)), (int(cx + 0.8 * h), int(y + 0.12 * h)), int(fill), 1)
        cv2.line(img, (int(cx + 0.2 * h), int(y + 0.88 * h)), (int(cx + 0.8 * h), int(y + 0.88 * h)), int(fill), 1)
    return x + n * h


def _background(rng, H, W, kind):
    if kind == "flat":
        return np.full((H, W), int(rng.integers(20, 90)), np.uint8)
    if kind == "gradient":
        a, b = rng.integers(10, 100, 2)
        return np.tile(np.linspace(a, b, W).astype(np.uint8), (H, 1))
    if kind == "dither":
        lo, hi = sorted(rng.integers(10, 110, 2))
        yy, xx = np.mgrid[:H, :W]
        return np.where((xx + yy) % 2 == 0, lo, hi).astype(np.uint8)
    if kind == "noise":
        return np.clip(rng.normal(60, 12, (H, W)), 0, 255).astype(np.uint8)
    raise ValueError(kind)


def make_line(rng, kind="flat", style="plain", end=None, ruby=False, adjacent=False, H=96, W=480, h=None):
    """(grey image, the line's exact ink box [x0, y0, x1, y1]): a line of light glyphs on a dark background, the box
    measured on the line's own ink layer."""
    h = int(h or rng.integers(14, 30))
    n = int(rng.integers(3, 12))
    x = int(rng.integers(h + 4, 3 * h))
    y = (H - h) // 2
    img = _background(rng, H, W, kind)
    layer = np.zeros((H, W), np.uint8)
    fill = int(rng.integers(200, 256))
    xe = _glyphs(rng, layer, x, y, h, n, 255)
    if end == "dot":            # 。 at the lower left of its cell
        cv2.circle(layer, (xe + h // 4, y + int(0.75 * h)), max(2, h // 7), 255, 1, cv2.LINE_AA)
    elif end == "bracket":      # 」 at the left of its cell
        cv2.line(layer, (xe + h // 3, y + h // 6), (xe + h // 3, y + int(0.85 * h)), 255, 1)
        cv2.line(layer, (xe + h // 3, y + int(0.85 * h)), (xe + 2, y + int(0.85 * h)), 255, 1)
    elif end == "dash":         # ―― across its cell
        cv2.line(layer, (xe + 2, y + h // 2), (xe + 2 * h - 2, y + h // 2), 255, 1)
    elif end == "leader":       # …… : dots along the middle
        for k in range(6):
            cv2.circle(layer, (xe + 2 + k * h // 3, y + h // 2), 1, 255, -1)
    ys, xs = np.nonzero(layer > 127)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    other = np.zeros_like(layer)
    if ruby:                    # a band of half-size glyphs over the first cells, touching the line's top
        _glyphs(rng, other, x, box[1] - h // 2, h // 2, 2 * min(n, 3), 255, thick=(1, 1))
    if adjacent:                # lines above and below, 1-3 px of gap
        g = int(rng.integers(1, 4))
        _glyphs(rng, other, x, box[1] - g - h, h, n, 255)
        _glyphs(rng, other, x, box[3] + g, h, n, 255)
    if style == "outline":      # a dark 1-px outline around the fill
        both = np.maximum(layer, other)
        ring = cv2.dilate((both > 127).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        img[ring] = 5
    elif style == "shadow":
        both = np.maximum(layer, other)
        sh = np.zeros_like(both)
        sh[2:, 2:] = both[:-2, :-2]
        img[sh > 127] = 5
    for lay in (layer, other):
        a = lay.astype(np.float32) / 255.0
        img = (img * (1 - a) + fill * a).astype(np.uint8)
    return img, box


def perturb(rng, box, h, scale=0.15):
    e = rng.normal(0, scale * h, 4)
    return [box[0] + e[0], box[1] + e[1], box[2] + e[2], box[3] + e[3]]


# ----------------------------------------------------------------------------- the C library

def _cpu_has(*flags):
    try:
        txt = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    have = set()
    for line in txt.splitlines():
        if line.startswith("flags"):
            have = set(line.split(":", 1)[1].split())
            break
    return all(f in have for f in flags)


def _build(out, arch):
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc or not shutil.which("make"):
        pytest.skip("no C compiler")
    subprocess.run(["make", "-s", "-C", str(ROOT / "desktop"), f"OUT={out}", f"ARCH={arch}", f"CC={cc}",
                    f"{out}/libvocrtsdr.so"], check=True)
    return Path(out) / "libvocrtsdr.so"


@pytest.fixture(scope="module")
def c_refiners(tmp_path_factory):
    d = tmp_path_factory.mktemp("tsdr")
    libs = {"plain": _build(d / "plain", "")}
    if _cpu_has("avx2"):
        libs["avx2"] = _build(d / "avx2", "-march=x86-64-v3")
    return libs


def _random_case(rng):
    kind = ["flat", "gradient", "dither", "noise"][rng.integers(0, 4)]
    style = ["plain", "outline", "shadow"][rng.integers(0, 3)]
    end = [None, "dot", "bracket", "dash", "leader"][rng.integers(0, 5)]
    img, box = make_line(rng, kind, style, end, ruby=bool(rng.random() < 0.3), adjacent=bool(rng.random() < 0.3),
                         H=int(rng.integers(60, 140)), W=int(rng.integers(200, 520)))
    h = box[3] - box[1]
    boxes = [perturb(rng, box, h, scale=float(rng.choice([0.05, 0.15, 0.4]))) + [0] for _ in range(3)]
    H, W = img.shape
    boxes.append([float(rng.uniform(-20, W)), float(rng.uniform(-20, H)), float(rng.uniform(0, W + 20)),
                  float(rng.uniform(0, H + 20)), int(rng.random() < 0.2)])      # anywhere, sometimes vertical
    boxes.append([-5.5, -3.25, W + 7.0, H + 2.5, 0])                            # beyond the frame
    return img, boxes


def test_c_matches_python_on_random_lines(c_refiners):
    rng = np.random.default_rng(20261101)
    cases = [_random_case(rng) for _ in range(150)]
    opts_sets = [None, {"texture_pct": 10, "shrink_l_pct": 50, "row_gap": 1, "valley_pct": 0}]
    for name, lib in c_refiners.items():
        for opts in opts_sets:
            cr = TR.CRefine(lib, opts)
            n = 0
            for img, boxes in cases:
                st_py, st_c = [], []
                ref = TR.refine(img, boxes, opts, st_py)
                got = cr.refine(img, boxes, st_c)
                assert [b[:4] for b in got] == [b[:4] for b in ref], (name, opts)
                assert st_c == st_py, (name, opts)
                n += sum(1 for s in st_py if s & 15)
            assert n > 200          # most boxes were snapped


def test_c_defaults_match_python(c_refiners):
    for lib in c_refiners.values():
        TR.CRefine(lib)             # raises when the C defaults or the option count differ


def test_c_rejects_bad_arguments(c_refiners):
    import ctypes as C
    cr = TR.CRefine(c_refiners["plain"])
    g = np.zeros((10, 10), np.uint8)
    small = np.zeros(16, np.uint8)
    xy = np.zeros(4, np.float64)
    rc = cr.lib.vocr_tsdr_refine(g.ctypes.data, 10, 10, 10, xy.ctypes.data, None, 1, None, None, small.ctypes.data,
                                 C.c_size_t(small.size))
    assert rc == -2                 # the scratch is too small
    rc = cr.lib.vocr_tsdr_refine(g.ctypes.data, 10, 10, 5, xy.ctypes.data, None, 1, None, None, small.ctypes.data,
                                 C.c_size_t(small.size))
    assert rc == -1                 # stride < width


# ----------------------------------------------------------------------------- the steps and guards

def _refine1(img, b, **opts):
    st = []
    out = TR.refine(img, [list(b) + [0]], opts, st)[0]
    return out[:4], st[0]


@pytest.mark.parametrize("kind", ["flat", "gradient", "dither", "noise"])
@pytest.mark.parametrize("style", ["plain", "outline", "shadow"])
def test_snaps_a_perturbed_box_to_the_ink(kind, style):
    rng = np.random.default_rng(abs(hash((kind, style))) % (1 << 32))
    errs = []
    for _ in range(12):
        img, box = make_line(rng, kind, style)
        h = box[3] - box[1]
        out, st = _refine1(img, perturb(rng, box, h, 0.1))
        errs.append([abs(a - b) for a, b in zip(out, box)])
    e = np.array(errs)
    # the shadow and the outline are ink of the line too (they differ from the background): allow them
    tol = 3 if style != "plain" else 1
    assert np.median(e) <= tol and np.percentile(e, 75) <= tol + 1, (kind, style, e.tolist())


@pytest.mark.parametrize("end", ["dot", "bracket", "dash", "leader"])
def test_grows_over_thin_end_punctuation(end):
    rng = np.random.default_rng(7)
    for _ in range(8):
        img, box = make_line(rng, "flat", "plain", end=end)
        h = box[3] - box[1]
        # the detector stopped short of the line's end by 0.6 line heights (the punctuation, or part of the dash);
        # the walk reaches about along_pct - gap_pct (1.1 line heights) beyond the box
        cut = [box[0], box[1], box[2] - 0.6 * h, box[3]]
        out, st = _refine1(img, cut)
        assert st & TR.ST_SNAP_R
        assert abs(out[2] - box[2]) <= 1, (end, out, box)


def test_does_not_cross_a_gap_wider_than_its_bound():
    rng = np.random.default_rng(11)
    img, box = make_line(rng, "flat", "plain", h=20, W=520)
    h = box[3] - box[1]
    # a separate blob (a page cursor) 1.2 line heights after the line's end
    cx = int(box[2] + 1.2 * h)
    cv2.rectangle(img, (cx, box[1] + 3), (cx + 8, box[3] - 3), 250, -1)
    out, st = _refine1(img, [box[0], box[1], box[2] - 2, box[3]])
    assert out[2] == box[2]
    # the same blob within the gap bound is taken in (it could be a glyph of the line)
    img2, box2 = make_line(np.random.default_rng(11), "flat", "plain", h=20, W=520)
    cx = int(box2[2] + 0.4 * h)
    cv2.rectangle(img2, (cx, box2[1] + 3), (cx + 4, box2[3] - 3), 250, -1)
    out2, _ = _refine1(img2, [box2[0], box2[1], box2[2] - 2, box2[3]])
    assert out2[2] == cx + 5


def test_barrier_a_frame_bar_beside_the_line_is_not_taken_in():
    rng = np.random.default_rng(5)
    img, box = make_line(rng, "flat", "plain", h=20)
    bx = box[0] - 6                     # a box frame's vertical bar, taller than the line, 6 px left of the ink
    cv2.line(img, (bx, 0), (bx, img.shape[0] - 1), 240, 2)
    out, st = _refine1(img, [box[0] + 3, box[1], box[2], box[3]])
    assert out[0] == box[0], (out, box)
    out_nobar, _ = _refine1(img, [box[0] + 3, box[1], box[2], box[3]], bar_pct=0)
    assert out_nobar[0] < box[0]        # without the barrier rule the walk takes the bar in


def test_ruby_band_above_the_line_is_left_out():
    rng = np.random.default_rng(3)
    for _ in range(6):
        img, box = make_line(rng, "flat", "plain", ruby=True, h=24, W=520)
        h = box[3] - box[1]
        out, st = _refine1(img, [box[0], box[1] + 0.15 * h, box[2], box[3]])
        assert out[1] >= box[1] - 1, (out, box)
        assert not (st & TR.ST_LIMIT) or out[1] >= box[1]


def test_touching_lines_are_not_merged():
    rng = np.random.default_rng(9)
    for _ in range(10):
        img, box = make_line(rng, "flat", "plain", adjacent=True, h=20, H=120)
        out, st = _refine1(img, [box[0], box[1] + 2, box[2], box[3] - 2])
        # either snapped to this line's ink, or (lines touching through more than the valley rule allows) kept
        assert box[1] - 1 <= out[1] <= box[1] + 2 and box[3] - 2 <= out[3] <= box[3] + 1, (out, box, st)


def test_band_limit_keeps_the_edge():
    # ink that runs on past the across band (a striped block much taller than the box, within its columns): the
    # across walks run out of the window, so the top and bottom keep the detector's values
    img = np.full((120, 300), 30, np.uint8)
    img[10:110, 60:240] = 220
    img[10:110, 60:240:4] = 30            # vertical stripes: every row of the block has ink and edges
    b = [60.0, 50.0, 240.0, 70.0]
    out, st = _refine1(img, b)
    assert st & TR.ST_LIMIT
    assert out[1] == 50.0 and out[3] == 70.0


def test_textured_margins_raise_the_threshold():
    # bright specks on both sides of the line (in its rows) as bright as the text: the noise term lifts the
    # threshold above them and the text alike, so nothing is ink and the box is kept
    rng = np.random.default_rng(4)
    img, box = make_line(rng, "flat", "plain", h=20, W=400)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    speck = ((xx // 2 + yy // 2) % 3 == 0) & ((xx < box[0] - 12) | (xx > box[2] + 12))
    img[speck] = 255
    b = [box[0] + 1.5, box[1] + 1.5, box[2] - 1.5, box[3] - 1.5]
    out, st = _refine1(img, b)
    assert st & TR.ST_NOINK and out == b


def test_busy_band_keeps_the_box():
    # specks above and below the line (beyond an empty row), none beside it: the margins are clean, the across walks
    # stop at the line, but more than busy_pct % of the window around the box is ink
    rng = np.random.default_rng(4)
    img, box = make_line(rng, "flat", "plain", h=20, W=400)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    speck = ((xx // 2) % 2 == 0) & ((yy < box[1] - 1) | (yy >= box[3] + 1))      # 2-px stripes, half the pixels
    img[speck] = 255
    b = [box[0] + 1.5, box[1] + 1.5, box[2] - 1.5, box[3] - 1.5]
    out, st = _refine1(img, b, busy_pct=15)
    assert st & TR.ST_BUSY and out == b
    out2, st2 = _refine1(img, b, busy_pct=100)
    assert not (st2 & TR.ST_BUSY) and out2 == [float(v) for v in box]


def test_small_vertical_and_non_finite_boxes_are_left_alone():
    img = np.random.default_rng(0).integers(0, 255, (50, 80), dtype=np.uint8)
    st = []
    boxes = [[10.0, 10.0, 12.5, 30.0, 0], [5.0, 5.0, 20.0, 45.0, 1], [float("nan"), 1.0, 20.0, 9.0, 0],
             [1.0, 1.0, float("inf"), 9.0, 0]]
    out = TR.refine(img, boxes, None, st)
    assert st == [TR.ST_SMALL, TR.ST_VERT, TR.ST_SMALL, TR.ST_SMALL]
    for a, b in zip(out[:2], boxes[:2]):
        assert a == b


def test_extra_fields_and_order_are_kept():
    rng = np.random.default_rng(1)
    img, box = make_line(rng, "flat", "plain")
    b = [box[0] - 2.5, box[1] - 1.5, box[2] + 1.25, box[3] + 0.5, 0, 0.93, 2]
    out = TR.refine(img, [b, b])
    assert out[0][4:] == [0, 0.93, 2] and out[0] == out[1]
    assert out[0][:4] == [float(v) for v in box]


def test_quantile_and_r100_are_the_integer_rules():
    v = np.array([5, 1, 9, 3, 7], np.int16)
    assert TR.quantile(v, 50) == 5 and TR.quantile(v, 99) == 9 and TR.quantile(v, 1) == 1
    assert TR.quantile(np.array([], np.int16), 90) == 0
    assert TR.r100(7, 50) == 4 and TR.r100(5, 50) == 3 and TR.r100(10, 25) == 3


def test_refine_det_on_box_objects():
    import dataclasses

    @dataclasses.dataclass
    class Box:
        x: float
        y: float
        w: float
        h: float
        vertical: int
        logit: float = 1.0

    rng = np.random.default_rng(6)
    img, box = make_line(rng, "flat", "plain")
    b = Box(box[0] - 2.0, box[1] + 1.0, box[2] - box[0] + 3.0, box[3] - box[1] - 1.0, 0)
    (r,) = TR.refine_det(img, [b])
    assert (r.x, r.y, r.x + r.w, r.y + r.h) == tuple(float(v) for v in box) and r.logit == 1.0


def test_unknown_option_is_refused():
    with pytest.raises(KeyError):
        TR.options({"no_such_option": 1})

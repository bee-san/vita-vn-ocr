"""textspot-desktop (#173): the network family, the training targets, the exact ink box, the decode, the pad rules
and the decision rule's helpers. No data: synthetic images and random networks only."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "tools", ROOT / "eval", ROOT / "synth", ROOT / "teacher", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

import tsdesk as T  # noqa: E402  (tools/tsdesk.py)

import tsdesk_net as N  # noqa: E402
import tsdesk_targets as TT  # noqa: E402

# ----------------------------------------------------------------------------- the network

@pytest.mark.parametrize("name", sorted(N.SHAPES))
def test_shapes_run_and_count(name):
    sh = N.SHAPES[name]
    torch.manual_seed(0)
    m = N.TsdNet(sh).eval()
    x = torch.randint(0, 256, (1, 1, 96, 160), dtype=torch.uint8)
    with torch.no_grad():
        y = m(x)
    hs = sh["hs"]
    assert y.shape == (1, N.OUT, 96 // hs, 160 // hs)
    mc = N.macs(sh, 960, 544)
    assert mc["total"] == sum(v for k, v in mc.items() if k != "total")
    assert 10e6 < mc["total"] < (200e6 if name.startswith("p") else 1e9)   # round 2's q shapes: up to ~0.85 G


def test_fold_bn_keeps_the_function():
    torch.manual_seed(1)
    m = N.TsdNet(N.SHAPES["p8g"])
    m.train()
    x = torch.randint(0, 256, (4, 1, 64, 96), dtype=torch.uint8)
    for _ in range(3):
        m(x)                                     # running statistics move away from their defaults
    m.eval()
    f = N.fold_bn(m)
    with torch.no_grad():
        a, b = m(x), f(x)
    assert float((a - b).abs().max()) < 1e-3


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    torch.manual_seed(2)
    m = N.TsdNet(N.SHAPES["p8s"]).eval()
    worst = N.export_onnx(m, tmp_path / "m.onnx", N.onnx_meta(N.SHAPES["p8s"], "t", "fit:960", T.DEFAULTS))
    assert worst < 1e-3
    det = T.Detector(tmp_path / "m.onnx")
    assert det.stride == 8 and det.size == "fit:960"


# ----------------------------------------------------------------------------- targets

def test_targets_edges_and_ends():
    import tsdesk_det as TD
    s = 8
    line = [40, 64, 300, 88, 0, 1.0, 1]                      # a 24 px line, 260 px long, inside the window
    K, O, A1, A2, E1, E2, v, k, m1, m2, Wt, M = TD.targets([line], [], 512, 256, s)  # noqa: E741
    rows, cols = np.nonzero(k)
    assert len(rows) and cols.min() == 40 // s and cols.max() == (300 - 1) // s
    centres = np.arange(512 // s) * s + (s - 1) / 2
    # the across distances are the edges' distances from each kernel cell's centre
    r0 = rows[0]
    assert A1[r0, cols[0]] == pytest.approx(centres[r0] - 64)
    assert A2[r0, cols[0]] == pytest.approx(88 - centres[r0])
    # E targets near the ends only (within max(2 s, 1.5 x 24) = 36 px), exact distances
    sel1 = np.nonzero(m1)
    assert len(sel1[0]) and all(centres[c] - 40 <= 36 for c in sel1[1])
    assert E1[sel1][0] == pytest.approx(centres[sel1[1][0]] - 40)
    sel2 = np.nonzero(m2)
    assert all(300 - centres[c] <= 36 for c in sel2[1])
    # the end columns carry the edge-kernel weight
    assert TD.EDGE_WEIGHT == 4.0                              # the registered recipe's
    assert Wt[r0, cols.min()] == pytest.approx(4.0) and Wt[r0, cols.max()] == pytest.approx(4.0)
    assert Wt[r0, (cols.min() + cols.max()) // 2] == pytest.approx(1.0)


def test_targets_cut_end_has_no_e_target():
    import tsdesk_det as TD
    line = [-50, 64, 200, 88, 0, 1.0, 1]                     # starts outside the window: its start is not an end
    *_, m1, m2, Wt, M = TD.targets([line], [], 512, 256, 8)
    assert not m1.any() and m2.any()


# ----------------------------------------------------------------------------- the exact ink box

def _page(fg=230, bg=40, dither=False, w=320, h=40):
    img = np.full((h, w, 3), bg, np.uint8)
    if dither:
        yy, xx = np.mgrid[0:h, 0:w]
        img[((yy + xx) % 2) == 0] = bg + 25
    return img


def test_ink_box_keeps_thin_end_marks():
    img = _page()
    img[12:28, 60:200] = 230                                 # the line's body
    img[26:28, 210:214] = 200                                # a faint, small trailing mark (a 。-like dot)
    box, how = TT.ink_box(img, [50, 8, 230, 32])
    assert how == "ink"
    assert box[0] == 60 and box[2] == 214 and box[1] == 12 and box[3] == 28


def test_ink_box_ignores_dither_and_gradients():
    img = _page(dither=True)
    img[12:28, 60:200] = 235
    box, _ = TT.ink_box(img, [40, 8, 240, 32])
    assert box[0] == 60 and box[2] == 200
    g = np.tile(np.linspace(20, 120, 320).astype(np.uint8)[None, :, None], (40, 1, 3))
    g[12:28, 100:180] = 250
    box, _ = TT.ink_box(g, [40, 8, 240, 32])
    assert abs(box[0] - 100) <= 1 and abs(box[2] - 180) <= 1


def test_ink_box_threshold_is_0_4_of_the_body_contrast():
    """k = 0.4 (the registered reference): an end mark at 45% of the line body's contrast is ink, one at 35% is not."""
    for frac, end in ((0.45, 216), (0.35, 200)):
        img = _page(bg=20)
        img[12:28, 60:200] = 230
        img[22:28, 210:216] = int(round(20 + frac * 210))
        box, how = TT.ink_box(img, [50, 8, 230, 32])
        assert how == "ink" and box == [60, 12, end, 28], (frac, box)


def test_ink_box_noise_margin_keeps_the_box_noise_out():
    """The threshold's noise term is the ring's 90th percentile + 12: on a dithered box (ring median 120, every
    background pixel 20 from it) a mark 30 from it is under 20 + 12, so it is not ink; with no margin it would be."""
    yy, xx = np.mgrid[0:40, 0:320]
    img = np.where((yy + xx) % 2 == 0, 100, 140).astype(np.uint8)[:, :, None].repeat(3, 2)
    img[12:28, 60:200] = 160
    img[20:28, 206:214] = 150
    box, _ = TT.ink_box(img, [50, 8, 230, 32])
    assert box == [60, 12, 200, 28]
    assert TT.ink_box(img, [50, 8, 230, 32], noise_margin=0)[0] == [60, 12, 214, 28]


def test_ink_box_needs_a_stroke_edge():
    """A smooth bright bump past the line's end (above the threshold, but no 3x3 contrast above tc) is not ink."""
    img = _page(bg=20)
    img[12:28, 60:200] = 120
    yy, xx = np.mgrid[0:40, 0:320]
    bump = 80.0 * np.exp(-((xx - 236) ** 2 + (yy - 20) ** 2) / (2 * 6.0 ** 2))
    img = np.clip(img + bump[:, :, None], 0, 255).round().astype(np.uint8)
    box, how = TT.ink_box(img, [50, 8, 260, 32])
    assert how == "ink" and box == [60, 12, 200, 28]


def test_strip_weak_drops_the_trailing_meiki_ignores():
    """The junk recipe drops a hooked page's weak MeikiOCR ignore areas, which build_labels appends last."""
    import tsdesk_det as TD
    ign = [[0, 0, 5, 5], [10, 10, 15, 15], [20, 20, 25, 25], [30, 30, 35, 35], [40, 40, 45, 45]]
    meta = {"counts": {"ruby_ignore": 1, "drop_ignore": 1, "uncovered_ignore": 1, "meiki_ignore": 2}}
    smp = {"ignore": list(ign), "meta": meta}
    assert TD.strip_weak(smp, "hook")["ignore"] == ign[:3] and smp["ignore"] == ign
    desk = {"ignore": list(ign), "meta": {"counts": {"meiki_ignore": 2}, "src_ignore": 5}}
    assert TD.strip_weak(desk, "desk_hook")["ignore"] == ign[:3]
    # counts that do not add up, another source, no weak areas: unchanged
    for smp2, src in (({"ignore": ign[:4], "meta": meta}, "hook"), ({"ignore": ign[:4], "meta": desk["meta"]},
                      "desk_hook"), (smp, "realplus"), ({"ignore": ign, "meta": {"counts": {"meiki_ignore": 0}}},
                                                        "hook")):
        assert TD.strip_weak(smp2, src) is smp2


def test_reading_pad_rule():
    """The reference pipeline's pad: max(2, round(k H / 544)) a side (k6v4: 6 across, 4 along the height)."""
    assert T.READ_PAD == "k6v4"
    assert T.pad_of("k6v4", 544) == (6, 4) and T.pad_of("k6v4", 1080) == (12, 8) and T.pad_of("k6v4", 500) == (6, 4)
    assert T.pad_of("k6v4", 100) == 2 and T.pad_of("k6v4", 272) == (3, 2) and T.pad_of("auto", 1080) == 8


def test_cells_union_prefers_cells():
    ln = {"box": [0, 0, 100, 20], "cell_boxes": [[10, 2, 26, 18], [26, 2, 42, 18]]}
    assert TT.cells_union(ln) == [10.0, 2.0, 42.0, 18.0]
    assert TT.cells_union({"box": [5, 6, 10, 12]}) == [5.0, 6.0, 15.0, 18.0]


def test_patch_sample_hook_checks_alignment():
    cache = {"T": {"000001": {"old": [[10, 10, 50, 20]], "new": [[12, 11, 49, 20]]}}}
    smp = {"title": "T", "page": "000001",
           "lines": [[10, 10, 50, 20, 0, "dialogue", 1.0], [0, 0, 5, 5, 0, "notdlg", 0.7]]}
    out = TT.patch_sample(smp, cache, "hook")
    assert out["lines"][0][:4] == [12, 11, 49, 20] and out["lines"][0][4:] == [0, "dialogue", 1.0]
    assert out["lines"][1] == smp["lines"][1]
    bad = dict(smp, lines=[[11, 10, 50, 20, 0, "dialogue", 1.0]])
    assert TT.patch_sample(bad, cache, "hook") is None


# ----------------------------------------------------------------------------- the decode

def _maps(h, w, s, lines):
    """Ideal maps for horizontal lines [x0, y0, x1, y1] at stride s (kernel band at the centre rows)."""
    hm, wm = -(-h // s), -(-w // s)
    m = np.zeros((N.OUT, hm, wm), np.float32)
    m[0] = -5.0
    m[1] = -5.0
    cs = np.arange(max(hm, wm)) * s + (s - 1) / 2
    for x0, y0, x1, y1 in lines:
        c = (y0 + y1) / 2
        rows = [r for r in range(hm) if abs(cs[r] - c) <= max((y1 - y0) / 4, s / 2)]
        cols = list(range(int(x0 // s), int((x1 - 1) // s) + 1))
        for r in rows:
            for q in cols:
                m[0, r, q] = 3.0
                m[2, r, q] = cs[r] - y0
                m[3, r, q] = y1 - cs[r]
                m[4, r, q] = cs[q] - x0
                m[5, r, q] = x1 - cs[q]
    return m


def test_decode_recovers_exact_boxes():
    lines = [[37.0, 50.0, 301.0, 74.0], [37.0, 90.0, 221.0, 114.0]]
    m = _maps(160, 320, 8, lines)
    boxes = T.decode(m, 320, 160, 8)
    got = sorted([(b.x, b.y, b.x + b.w, b.y + b.h) for b in boxes], key=lambda t: t[1])
    for g, want in zip(got, lines):
        assert all(abs(a - b) < 1e-9 for a, b in zip(g, want))
    assert all(b.vertical == 0 and b.cells >= 2 for b in boxes)


def test_decode_clamps_ends_to_the_kernel():
    m = _maps(80, 320, 8, [[40.0, 20.0, 200.0, 44.0]])
    m[4][m[0] > 0] = -100.0                                 # the start regressed far past the first cell
    boxes = T.decode(m, 320, 80, 8)
    assert len(boxes) == 1 and boxes[0].x == pytest.approx(5 * 8 + 8)     # lo + s


def test_decode_order_and_cap():
    m = _maps(400, 400, 8, [[8.0 + 40 * i, 8.0 + 40 * j, 32.0 + 40 * i, 24.0 + 40 * j]
                            for i in range(9) for j in range(9)])
    boxes = T.decode(m, 400, 400, 8, max_boxes=10)
    assert len(boxes) == 10
    keys = [(-b.logit, b.y, b.x) for b in boxes]
    assert keys == sorted(keys)


def test_input_size_and_pad():
    assert T.input_size("fit:960", 1920, 1080) == (960, 540)
    assert T.input_size("fit:960", 480, 272) == (480, 272)
    assert T.input_size("long:960", 480, 272) == (960, 544)
    assert T.pad_of("auto", 544) == 4 and T.pad_of("auto", 272) == 2 and T.pad_of("auto", 1080) == 8
    assert T.crop_of((10.2, 5.5, 20.0, 10.0), 2, 100, 100) == (8, 3, 25, 15)


def test_pad_rules():
    import tsdesk_eval as E
    assert E.pad_px(6, 544) == 6 and E.pad_px(4, 272) == 2 and E.pad_px(6, 1080) == 12
    assert E.crop_hv([10.0, 10.0, 50.0, 20.0], (6, 4), 100, 100) == (4, 6, 52, 18)
    assert E.crop_hv([0.0, 0.0, 1.0, 1.0], (0, 0), 0, 0) is None


def test_match_is_one_to_one():
    import tsdesk_eval as E
    dets = [[0, 0, 100, 20], [0, 0, 100, 20], [0, 30, 100, 50]]
    refs = [[2, 1, 98, 19], [0, 31, 99, 49]]
    pairs = E.match(dets, refs)
    assert sorted(j for _, j in pairs) == [0, 1] and len(pairs) == 2


def test_protocol_is_valid_json():
    doc = json.loads((ROOT / "eval" / "textspot_desktop_protocol.json").read_text(encoding="utf-8"))
    assert doc["kind"] == "textspot-desktop-protocol"
    assert [a["n"] for a in doc["amendments"]] == list(range(1, len(doc["amendments"]) + 1))
    assert math.isclose(sum(v for k, v in doc["recipe"]["mix"].items() if k != "why"), 1.0)


def test_resize_identity_and_constant():
    g = np.full((30, 40), 77, np.uint8)
    assert (T.resize(g, 40, 30) == g).all()
    assert (T.resize(g, 17, 11) == 77).all() and (T.resize(g, 83, 61) == 77).all()


def test_end_bands_leave_the_kernel_and_ignore_past_the_ends():
    import tsdesk_det as TD
    pos = [[100, 60, 300, 84, 0, 1.0, 1]]
    ign = TD.end_bands(pos, 1.0)
    assert ign == [[76, 60, 100, 84], [300, 60, 324, 84]]
    K, O, A1, A2, E1, E2, v, k, m1, m2, Wt, M = TD.targets(pos, ign, 512, 256, 8)  # noqa: E741
    r = int(np.nonzero(k)[0][0])
    assert k[r, 100 // 8:(300 - 1) // 8 + 1].all()            # the line's kernel cells stay
    assert not v[r, 300 // 8 + 1] and not v[r, 76 // 8]        # the cells past its ends are ignored
    assert v[r, 340 // 8]                                      # further on, background again


def test_ornament_rows_stay_off_the_labels():
    import tsdesk_det as TD
    rng = np.random.default_rng(0)
    drawn = 0
    for _ in range(50):
        img = np.full((256, 512), 40, np.uint8)
        pos = [[100, 120, 400, 144, 0, 1.0, 1]]
        ign = [[10, 10, 60, 40]]
        before = img.copy()
        box = TD.add_ornaments(img, pos, ign, rng)
        if box is None:
            assert (img == before).all()
            continue
        drawn += 1
        assert not (box[0] < 400 and 100 < box[2] and box[1] < 144 and 120 < box[3])     # off the line
        assert (img[120:144, 100:400] == 40).all()                                       # the line untouched
        assert (img != before).any()
    assert drawn >= 10


def test_reference_pipeline_pad_rules():
    assert T.pad_of(None, 544) == (6, 4) and T.pad_of("k6v4", 1080) == (12, 8) and T.pad_of("k6v4", 272) == (3, 2)
    assert T.pad_of("k5", 544) == 5 and T.pad_of(3, 544) == 3
    assert T.crop_of((10.0, 10.0, 40.0, 10.0), (6, 4), 100, 100) == (4, 6, 52, 18)


# ----------------------------------------------------------------------------- the post-pass (round 2, boxes helper)

def _glyphs(img, x0, y0, n, size, gap, val=230):
    """n glyph-like marks (a frame and a cross of 2 px strokes) of size x size px from (x0, y0), gap px apart."""
    for k in range(n):
        x = x0 + k * (size + gap)
        cv2.rectangle(img, (x, y0), (x + size - 1, y0 + size - 1), int(val), 2)
        cv2.line(img, (x + size // 2, y0), (x + size // 2, y0 + size - 1), int(val), 2)
        cv2.line(img, (x, y0 + size // 2), (x + size - 1, y0 + size // 2), int(val), 2)
    return x0 + n * (size + gap) - gap


def _box(x0, y0, x1, y1, vertical=0, logit=1.0):
    return T.Box(float(x0), float(y0), float(x1 - x0), float(y1 - y0), vertical, logit, 0, 0.0, 4)


def test_post_off_is_identity():
    g = np.full((60, 200), 20, np.uint8)
    _glyphs(g, 10, 20, 5, 16, 4)
    bs = [_box(8, 18, 110, 38), _box(9, 19, 30, 37)]
    assert T.post_pre(g, bs, {}) == bs
    assert T.post_options({}) == T.POST_DEFAULTS and not any(T.POST_DEFAULTS[k] for k in T.POST_PRE)
    with pytest.raises(KeyError):
        T.post_options({"nope": 1})


def test_merge_overlaps_takes_the_union():
    a, b = _box(10, 10, 200, 34), _box(180, 11, 220, 33, logit=0.5)       # an end fragment overlapping its line
    out = T.merge_overlaps([a, b], 30)
    assert len(out) == 1 and (out[0].x, out[0].y, out[0].x + out[0].w, out[0].y + out[0].h) == (10, 10, 220, 34)
    assert out[0].logit == 1.0 and out[0].cells == 8
    assert len(T.merge_overlaps([a, b], 60)) == 2                           # 20 / 40 of the fragment: under 60 %
    assert len(T.merge_overlaps([a, _box(10, 30, 200, 54)], 10)) == 2       # the next row: overlaps across by 4 px
    assert len(T.merge_overlaps([a, _box(150, 10, 174, 60, vertical=1)], 10)) == 2    # orientations differ
    assert len(T.merge_overlaps([a, _box(201, 10, 260, 34)], 1)) == 2       # side by side, no overlap


def test_split_rows_splits_two_tight_rows_not_ruby():
    g = np.full((80, 260), 30, np.uint8)
    _glyphs(g, 10, 10, 10, 18, 4)                                           # row 1: y 10-27
    _glyphs(g, 10, 31, 10, 18, 4)                                           # row 2: y 31-48, a 3 px gap
    out = T.split_rows(g, _box(8, 8, 230, 51), 65)
    assert len(out) == 2
    top, bot = sorted(out, key=lambda b: b.y)
    assert top.y == 8 and 27 <= top.y + top.h <= 29 and 29 <= bot.y <= 31 and bot.y + bot.h == 51
    assert len(T.split_rows(g, _box(8, 8, 230, 28), 65)) == 1                 # one row: no valley
    assert len(T.split_rows(g, _box(8, 8, 230, 51), 65, rows_len=1300)) == 1   # rows under 13 x the box height
    s2 = np.full((80, 120), 30, np.uint8)
    _glyphs(s2, 10, 10, 2, 18, 4)
    _glyphs(s2, 10, 31, 2, 18, 4)                                           # a two-glyph label over another
    assert len(T.split_rows(s2, _box(8, 8, 52, 51), 65)) == 1                 # 40 px of ink < 3 x 43 px
    r = np.full((80, 260), 30, np.uint8)
    _glyphs(r, 10, 6, 12, 8, 2)                                             # ruby: 8 px marks over the line
    _glyphs(r, 10, 18, 10, 18, 4)                                           # the line: 18 px
    assert len(T.split_rows(r, _box(8, 4, 230, 38), 65)) == 1


def test_split_along_at_wide_gaps_only():
    g = np.full((60, 400), 25, np.uint8)
    e1 = _glyphs(g, 10, 20, 4, 16, 3)                                       # a label
    e2 = _glyphs(g, e1 + 40, 20, 3, 16, 3)                                  # 40 px (2.2 line heights) on: a number
    b = _box(8, 18, e2 + 2, 38)
    out = T.split_along(g, b, 150)
    assert len(out) == 2
    assert out[0].x == b.x and abs(out[0].x + out[0].w - e1) <= 1
    assert abs(out[1].x - (e1 + 40)) <= 1 and out[1].x + out[1].w == b.x + b.w
    assert out[0].y == b.y and out[1].h == b.h
    assert len(T.split_along(g, b, 250)) == 1                              # 40 px < 2.5 x 20 px
    assert T.split_along(g, _box(8, 18, e2 + 2, 38, vertical=1), 150) == [_box(8, 18, e2 + 2, 38, vertical=1)]
    one = _box(8, 18, e1 + 2, 38)
    assert T.split_along(g, one, 100) == [one]                              # 3 px gaps between glyphs
    g[34:37, e1 + 18:e1 + 21] = 70                                          # a faint 、 in the gap: weak ink only
    assert len(T.split_along(g, b, 150)) == 1 and len(T.split_along(g, b, 150, gap_weak=0)) == 2


def test_post_pre_runs_in_order():
    g = np.full((90, 400), 25, np.uint8)
    e1 = _glyphs(g, 10, 10, 4, 18, 3)
    _glyphs(g, e1 + 50, 10, 3, 18, 3)
    _glyphs(g, 10, 31, 9, 18, 3)
    boxes = [_box(8, 8, 300, 51), _box(20, 9, 40, 29, logit=0.2)]
    out = T.post_pre(g, boxes, {"merge_pct": 50, "rows_pct": 65, "gap_pct": 150})
    rows = sorted(out, key=lambda b: (b.y, b.x))
    assert len(rows) == 3 and rows[0].y < rows[2].y and rows[0].y == rows[1].y


def test_drop_set_rules():
    o1, o2 = T.drop_options({"nojp": 1}), T.drop_options({"nojp": 2})

    def items(*rows):
        return [((x, y, w, h), t, c) for x, y, w, h, t, c in rows]
    for t in ("0000O00", "", "ME AGE", "10000"):
        assert T.drop_set(items((0, 0, 50, 20, t, 0.99)), o1) == {0}, t
    for t in ("「…………」", "――", "Aスポーツ", "14日目", "（！？）", "ＡＢＣ", "【ok】"):
        assert T.drop_set(items((0, 0, 50, 20, t, 0.99)), o1) == set(), t
    row = items((10, 10, 12, 20, "A", 0.9), (30, 10, 150, 20, "スポーツドリンク", 0.99), (300, 10, 40, 20, "0000", 0.8),
                (30, 40, 30, 20, "Max", 0.9))
    assert T.drop_set(row, o1) == {0, 2, 3}
    assert T.drop_set(row, o2) == {2, 3}                     # "A" sits 8 px from a Japanese reading in its row
    assert T.drop_set(items((10, 10, 12, 20, "", 0.9), (30, 10, 150, 20, "スポーツ", 0.99)), o2) == {0}   # empty
    far = items((10, 10, 40, 20, "行動力", 0.99), (75, 12, 40, 18, "Max", 0.9))     # 25 px on: 1.25 heights
    assert T.drop_set(far, o2) == {1} and T.drop_set(far, T.drop_options({"nojp": 2, "nojp_gap_pct": 200})) == set()
    strip = items((10, 10, 120, 20, "「000OO000", 0.7), (140, 10, 60, 20, "0000", 0.7))
    assert T.drop_set(strip, o2) == set()                    # 「 makes the ornament an anchor
    assert T.drop_set(strip, T.drop_options({"nojp": 2, "nojp_anchor_pct": 50})) == {1}
    assert T.drop_set(strip, T.drop_options({"nojp": 2, "nojp_gap_pct": 200, "nojp_anchor_pct": 50})) == {1}
    assert T.drop_set(strip, T.drop_options({"nojp": 2, "nojp_gap_pct": 200})) == set()
    assert T.japanese_share("「000") == 0.25 and T.japanese_share("") == 0.0
    assert T.drop_set(row, T.drop_options({})) == set()
    r = T.drop_options({"rep_pct": 50})
    reps = items((0, 0, 99, 20, "イf0u000u00000000000c", 0.9), (0, 30, 99, 20, "【AccessingtheWorld】", 0.9),
                 (0, 60, 99, 20, "……………………", 0.9), (0, 90, 99, 20, "000", 0.9), (0, 120, 99, 20, "所持金100000円", 0.9))
    assert T.drop_set(reps, r) == {0, 4} and T.drop_set(reps, T.drop_options({"rep_pct": 80})) == set()
    short = items((0, 0, 40, 20, "イ00", 0.9), (0, 30, 40, 20, "6億", 0.9))
    assert T.drop_set(short, r) == set() and T.drop_set(short, T.drop_options({"rep_pct": 50, "rep_len": 3})) == {0}
    assert T.drop_set(short, T.drop_options({"rep_pct": 50, "rep_len": 2})) == {0, 1}
    c = T.drop_options({"conf_milli": 600})
    assert T.drop_set(items((0, 0, 9, 9, "あ", 0.59), (0, 20, 9, 9, "あ", 0.61)), c) == {0}
    with pytest.raises(KeyError):
        T.drop_options({"conf": 1})


def test_spec_suffixes():
    import tsdesk_eval as E
    assert E.spec_parts("tsd:/m/a.onnx@long:960+refine") == ("tsd:/m/a.onnx@long:960", True, {}, {})
    assert E.spec_parts("tsd:/m/a.onnx@long:960+post:gap_pct=150,merge_pct=50+refine+drop:nojp=1") == (
        "tsd:/m/a.onnx@long:960", True, {"gap_pct": 150, "merge_pct": 50}, {"nojp": 1})
    assert E.spec_parts("tsd:/m/a.onnx") == ("tsd:/m/a.onnx", False, {}, {})
    assert E.spec_parts("tsd:/m/a.onnx+dec:conf_min_logit=1.5", with_decode=True)[4] == {"conf_min_logit": 1.5}
    with pytest.raises(ValueError):
        E.spec_parts("tsd:/m/a.onnx+refined")
    assert E.drop_of_det("tsd:/m/a.onnx@long:960+refine") is None
    assert E.drop_of_det("meiki+drop:nojp=1")["nojp"] == 1
    assert E.strip_drop("meiki+drop:nojp=1") == "meiki"
    assert E.strip_drop("tsd:/m/a.onnx+refine+drop:nojp=1+post:gap_pct=1") == "tsd:/m/a.onnx+refine+post:gap_pct=1"


def test_extend_ends_over_dash_runs():
    o = T.post_options({"ext_pct": 1200})
    g = np.full((60, 420), 20, np.uint8)
    # 「 at x 10-16 (a corner in the upper half), a dash run x 20-105 at the line's middle, 、 at x 108-112 low, a
    # gap of 11 px (0.6 line heights), two glyphs x 124-163 (the detector's box), ―― x 166-210 and 。 x 213-217 low
    cv2.line(g, (16, 20), (10, 20), 230, 2)
    cv2.line(g, (10, 20), (10, 30), 230, 2)
    cv2.line(g, (20, 29), (105, 29), 230, 2)
    cv2.rectangle(g, (108, 34), (112, 37), 230, -1)
    end = _glyphs(g, 124, 20, 2, 18, 4)
    cv2.line(g, (end + 2, 29), (end + 46, 29), 230, 2)
    cv2.rectangle(g, (end + 50, 33), (end + 54, 37), 230, -1)
    b = _box(124, 20, end, 38)
    out = T.post_post(g, [b], o)[0]
    assert out.y == b.y and out.h == b.h
    ink = np.flatnonzero((g[20:38] > 100).any(0))
    assert (out.x, out.x + out.w) == (ink[0], ink[-1] + 1) == (9, end + 55)
    assert T.post_post(g, [b], {}) == [b]
    # a frame's border just above the run: not taken
    g2 = g.copy()
    cv2.line(g2, (0, 16), (122, 16), 230, 1)
    assert T.post_post(g2, [b], o)[0].x == 124
    # another box beside it stops the extension at its edge; with ext_join the two become one when it reaches it
    nb = _box(10, 20, 60, 38)
    out3 = T.post_post(g, [b, nb], dict(o, ext_join=0))
    assert out3[0].x == 60 and out3[1].x == 10 and out3[1].x + out3[1].w == 113    # nb runs on over the dashes and 、
    out4 = T.post_post(g, [b, nb], o)
    assert len(out4) == 1 and (out4[0].x, out4[0].x + out4[0].w) == (10, end + 55)
    # the dots of an ellipsis (gaps inside the run) are not a dash run
    g4 = np.full((60, 420), 20, np.uint8)
    end4 = _glyphs(g4, 130, 20, 2, 18, 4)
    for k in range(8):
        cv2.rectangle(g4, (end4 + 3 + 7 * k, 28), (end4 + 4 + 7 * k, 29), 230, -1)
    b4 = _box(130, 20, end4, 38)
    assert T.post_post(g4, [b4], o)[0] == b4
    # a run under ext_dash_pct of the height is not a dash run; ordinary glyphs beyond the end are not taken
    g3 = np.full((60, 420), 20, np.uint8)
    end3 = _glyphs(g3, 130, 20, 2, 18, 4)
    cv2.line(g3, (end3 + 3, 29), (end3 + 12, 29), 230, 2)
    _glyphs(g3, end3 + 40, 20, 2, 18, 4)
    b3 = _box(130, 20, end3, 38)
    assert T.post_post(g3, [b3], o)[0] == b3


def test_decode_saddle_split():
    m = _maps(80, 400, 8, [[40.0, 20.0, 360.0, 44.0]])
    on = m[0] > 0
    m[0][on] = 6.0
    col = 24                                                # a weak column inside the kernel: a K saddle
    m[0][:, col][on[:, col]] = 0.5
    one = T.decode(m, 400, 80, 8)
    assert len(one) == 1 and T.decode(m, 400, 80, 8, saddle_pct=5) == one       # 0.5 > 5 % of 6
    two = sorted(T.decode(m, 400, 80, 8, saddle_pct=20), key=lambda b: b.x)
    assert len(two) == 2
    assert two[0].x == pytest.approx(40.0) and two[0].x + two[0].w == pytest.approx(8 * (col + 1))
    assert two[1].x == pytest.approx(8 * (col + 1)) and two[1].x + two[1].w == pytest.approx(360.0)
    assert all(abs(b.y - 20.0) < 1e-9 and abs(b.h - 24.0) < 1e-9 for b in two)
    assert all(v == 0 for v in T.DECODE_EXTRA.values()) and "saddle_pct" not in T.DEFAULTS     # off; C constants kept


def test_decode_vote_split_two_touching_lines():
    s, hm, wm = 8, 12, 40
    m = np.zeros((N.OUT, hm, wm), np.float32)
    m[0] = -5.0
    m[1] = -5.0
    cs = np.arange(max(hm, wm)) * s + (s - 1) / 2
    lines = {range(3, 5): (20.0, 44.0), range(5, 8): (46.0, 70.0)}       # kernel rows 3-7 touch: one component
    for rows_, (y0, y1) in lines.items():
        for r in rows_:
            for q in range(5, 31):
                m[0, r, q] = 4.0
                m[2, r, q], m[3, r, q] = cs[r] - y0, y1 - cs[r]
                m[4, r, q], m[5, r, q] = cs[q] - 40.0, 248.0 - cs[q]
    one = T.decode(m, wm * s, hm * s, s)
    assert len(one) == 1
    two = sorted(T.decode(m, wm * s, hm * s, s, vote_pct=70), key=lambda b: b.y)
    assert len(two) == 2
    assert two[0].y == pytest.approx(20.0) and two[0].y + two[0].h == pytest.approx(44.0)
    assert two[1].y == pytest.approx(46.0) and two[1].y + two[1].h == pytest.approx(70.0)
    assert all(b.x == pytest.approx(40.0) and b.x + b.w == pytest.approx(248.0) for b in two)
    m1 = _maps(160, 320, 8, [[37.0, 50.0, 301.0, 74.0]])                   # one line: its votes agree, no split
    assert len(T.decode(m1, 320, 160, 8, vote_pct=70)) == 1


def test_decode_rescue_weak_lines():
    m = _maps(160, 320, 8, [[37.0, 50.0, 301.0, 74.0], [37.0, 100.0, 221.0, 124.0]])
    weak = m[0][12:16] > 0                                  # the second line's kernel rows: made weak (K -1)
    m[0][12:16][weak] = -1.0
    assert len(T.decode(m, 320, 160, 8)) == 1
    two = sorted(T.decode(m, 320, 160, 8, rescue_milli=1500), key=lambda b: b.y)
    assert len(two) == 2 and two[1].logit == pytest.approx(-1.0)
    assert abs(two[1].y - 100.0) < 1e-9 and abs(two[1].x + two[1].w - 221.0) < 1e-9
    assert len(T.decode(m, 320, 160, 8, rescue_milli=500)) == 1          # -1 is under -0.5
    assert T.decode(m, 320, 160, 8, rescue_milli=1500)[0] == T.decode(m, 320, 160, 8)[0]   # the strong line as before


def test_read_frame_reading_filter():
    import vocr_onnx as V

    class Det:
        def detect(self, g):
            return [_box(10, 10, 60, 30), _box(110, 10, 200, 30), _box(204, 10, 220, 30), _box(10, 60, 30, 90, 1)]

    class Rec:
        def read(self, img, invert=False, lm=None):
            t = {54: "0000000", 94: "あいう", 20: "A"}.get(img.shape[1], "?")       # crop widths (pad 2 px)
            return V.Line(t, t, conf_mean=0.99)

    g = np.full((100, 300), 30, np.uint8)
    texts = lambda drop: [ln.text for ln in T.read_frame(g, Det(), Rec(), drop=drop)]    # noqa: E731
    assert texts(None) == ["0000000", "あいう", "A", ""]
    assert texts({"nojp": 1}) == ["あいう", ""]                           # a vertical box is never read or dropped
    assert texts({"nojp": 2}) == ["あいう", "A", ""]
    assert texts({"rep_pct": 50}) == ["あいう", "A", ""]


def test_decode_clamp_ends_at_a_neighbour():
    m = _maps(80, 400, 8, [[40.0, 20.0, 160.0, 44.0], [200.0, 20.0, 240.0, 44.0]])     # a label and a number
    on = m[0] > 0
    m[5][on & (np.arange(m.shape[2])[None, :] < 20)] += 80.0               # the label's end regressed over the number
    a = sorted(T.decode(m, 400, 80, 8), key=lambda b: b.x)
    assert a[0].x + a[0].w > 200.0                                          # the label's box runs into the number
    b = sorted(T.decode(m, 400, 80, 8, clamp_ends=1), key=lambda b: b.x)
    assert b[0].x + b[0].w == pytest.approx(200.0) and b[1] == a[1]       # it stops at the number's first cell
    one = _maps(80, 400, 8, [[40.0, 20.0, 160.0, 44.0]])
    assert T.decode(one, 400, 80, 8, clamp_ends=1) == T.decode(one, 400, 80, 8)


def test_merge_touching_and_the_guard_band():
    a, b = _box(10, 10, 200, 34), _box(196, 11, 300, 33)              # one line in two boxes, 4 px of overlap
    assert len(T.merge_overlaps([a, b], 70)) == 2
    out = T.post_pre(np.zeros((60, 400), np.uint8), [a, b], {"touch_pct": 70})
    assert len(out) == 1 and (out[0].x, out[0].x + out[0].w, out[0].y, out[0].y + out[0].h) == (10, 300, 10, 34)
    assert len(T.merge_touching([a, _box(201, 10, 300, 34)], 70)) == 2                 # a 1 px gap: apart
    assert len(T.merge_touching([a, _box(196, 30, 300, 54)], 70)) == 2                 # 4 px across: another row
    # the guard: ink 6 px above the run (beyond a 10 % band of an 18 px line) rejects it only with the full margin
    o = T.post_options({"ext_pct": 1200})
    g = np.full((60, 420), 20, np.uint8)
    end = _glyphs(g, 124, 20, 2, 18, 4)
    cv2.line(g, (end + 2, 29), (end + 46, 29), 230, 2)
    cv2.line(g, (end + 2, 15), (end + 46, 15), 230, 1)                 # a rule 5 px above the line's top
    b2 = _box(124, 20, end, 38)
    assert T.post_post(g, [b2], o)[0] == b2
    got = T.post_post(g, [b2], dict(o, ext_band_pct=10))[0]
    assert got.x == b2.x and got.x + got.w > end + 40


def test_vflip_turns_a_gappy_column_into_rows():
    g = np.full((240, 300), 20, np.uint8)
    for col in range(3):                                                 # a 3 x 4 kana grid: 18 px glyphs, 12 px gaps
        for row in range(4):
            _glyphs(g, 20 + 30 * col, 20 + 30 * row, 1, 18, 0)
    cols = [_box(20 + 30 * c - 1, 19, 20 + 30 * c + 19, 19 + 30 * 3 + 20, vertical=1) for c in range(3)]
    out = T.post_pre(g, cols, {"vflip_pct": 30})
    assert all(not b.vertical for b in out) and len(out) == 4               # one horizontal box per grid row
    for k, b in enumerate(sorted(out, key=lambda b: b.y)):
        assert b.x < 21 and b.x + b.w > 97 and abs(b.y - (20 + 30 * k)) <= 2 and b.h <= 22
    t = np.full((240, 60), 20, np.uint8)                                   # vertical text: glyphs 2 px apart
    for row in range(7):
        _glyphs(t, 20, 10 + 20 * row, 1, 18, 0)
    col = [_box(19, 9, 39, 150, vertical=1)]
    assert T.post_pre(t, col, {"vflip_pct": 30}) == col
    assert T.post_pre(g, cols, {}) == cols

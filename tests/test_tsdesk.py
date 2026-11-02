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
    assert 10e6 < mc["total"] < 200e6


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
    assert Wt[r0, cols.min()] == pytest.approx(TD.EDGE_WEIGHT) and Wt[r0, cols.max()] == pytest.approx(TD.EDGE_WEIGHT)


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


def test_resize_twin_matches_the_c_resize():
    lib = ROOT / "desktop" / "build" / "libvocrdesk.so"
    if not lib.exists():
        pytest.skip("desktop/build/libvocrdesk.so is not built")
    import vocr_desktop as VD
    L = VD.c_library(lib)
    rng = np.random.default_rng(0)
    for w, h, dw, dh in ((1920, 1080, 960, 540), (640, 480, 960, 720), (480, 272, 960, 544), (961, 545, 960, 544),
                         (7, 5, 3, 2), (3, 2, 7, 5)):
        g = rng.integers(0, 256, (h, w), dtype=np.uint8)
        out = np.zeros((dh, dw), np.uint8)
        scr = np.zeros(L.vocr_desk_resize_scratch(w, h, dw, dh) + 64, np.uint8)
        L.vocr_desk_resize(g.ctypes.data, w, h, w, out.ctypes.data, dw, dh, scr.ctypes.data)
        assert (T.resize(g, dw, dh) == out).all()


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

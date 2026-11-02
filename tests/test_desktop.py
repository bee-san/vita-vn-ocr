"""The desktop pipeline (issue #169, docs/desktop.md): tools/vocr_desktop.py's size and pad policies, decode and
reading order; train/desktop_det.py's network, BatchNorm folding, ONNX and VDSK exports and ruby filter; the C path
(desktop/: the same network from the VDSK blob, the decode bit for bit on the same maps, the resize, the page
pipeline with runtime/vocr.c); eval/desktop.py's scoring helpers. No private data: synthetic maps, random networks and
generated frames only."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "train", ROOT / "eval", ROOT / "synth", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vocr_desktop as VD  # noqa: E402


def test_scale_for_policies():
    assert VD.scale_for("native", 1920, 1080) == 1.0
    assert VD.scale_for(None, 640, 480) == 1.0
    assert VD.scale_for("fit:960", 1920, 1080) == 0.5
    assert VD.scale_for("fit:960", 480, 272) == 1.0            # fit never scales up
    assert VD.scale_for("960", 1280, 720) == 0.75              # a bare number is fit:N
    assert VD.scale_for("long:960", 480, 272) == 2.0
    with pytest.raises(ValueError):
        VD.scale_for("wide:960", 10, 10)
    assert VD.policy_of(None) == (-1, 0) and VD.policy_of("native") == (0, 0)
    assert VD.policy_of("fit:640") == (1, 640) and VD.policy_of("long:960") == (2, 960)


def _maps_with_line(h, w, rows, cols, a1=6.0, a2=6.0, tag=None, vertical=False):
    """[8, h/4, w/4] maps with one kernel at the given cells (K logit 4), A1 / A2 across, tag maps."""
    m = np.zeros((8, h // 4, w // 4), np.float32)
    m[0] = -8.0
    m[1] = 4.0 if vertical else -4.0
    for r in rows:
        for c in cols:
            m[0, r, c] = 4.0
            m[2, r, c] = a1
            m[3, r, c] = a2
            if tag is not None:
                m[4 + tag - 1, r, c] = 3.0
    return m


def test_decode_one_horizontal_line():
    m = _maps_with_line(64, 128, rows=[5], cols=range(4, 20), a1=6.0, a2=6.0)
    boxes = VD.decode(m, 128, 64)
    assert len(boxes) == 1
    b = boxes[0]
    # across: the cell centre 4 * 5 + 1.5 = 21.5, extent [15.5, 27.5] (12 px), pad max(2, 0.18 * 12) = 2.16
    assert b.vertical == 0 and b.cells == 16
    assert b.y == pytest.approx(21.5 - 6 - 2.16)
    assert b.h == pytest.approx(12 + 2 * 2.16)
    # along: cells 4..19 -> [16, 80), grown by the pad
    assert b.x == pytest.approx(16 - 2.16)
    assert b.w == pytest.approx(64 + 2 * 2.16)
    assert b.tag == 0 and b.logit == pytest.approx(4.0)


def test_decode_tag_vertical_small_and_weak_components():
    m = _maps_with_line(64, 128, rows=range(2, 12), cols=[20], vertical=True, tag=1)
    m[0, 1, 1] = 4.0                       # a single cell: under min_cells, dropped
    boxes = VD.decode(m, 128, 64)
    assert len(boxes) == 1 and boxes[0].vertical == 1 and boxes[0].tag == 1
    assert boxes[0].tconf == pytest.approx(3.0)
    weak = _maps_with_line(64, 128, rows=[5], cols=range(4, 8))
    weak[0][weak[0] > 0] = 0.01            # a mean logit of 0.01
    assert len(VD.decode(weak, 128, 64)) == 1
    assert len(VD.decode(weak, 128, 64, conf_min_logit=0.02)) == 0
    thin = _maps_with_line(64, 128, rows=[5], cols=range(4, 8), a1=1.0, a2=1.0)
    assert len(VD.decode(thin, 128, 64)) == 0          # 2 px across, under min_across


def test_decode_clips_sorts_and_caps():
    m = _maps_with_line(64, 128, rows=[1], cols=range(0, 32), a1=10.0, a2=4.0)
    b = VD.decode(m, 100, 60)[0]                        # only 100 x 60 of the 128 x 64 input is the image
    assert b.x == 0 and b.y == 0
    assert b.x + b.w <= 100 and b.y + b.h <= 60
    many = np.full((8, 64, 64), -8.0, np.float32)
    many[1] = -4.0
    many[2:4] = 3.0
    for i in range(0, 64, 2):                           # 32 x 16 components of 2 cells each
        for j in range(0, 64, 4):
            many[0, i, j:j + 2] = 1.0 + 0.001 * (i * 64 + j)
    boxes = VD.decode(many, 256, 256, max_boxes=100)
    assert len(boxes) == 100
    assert all(boxes[k].logit >= boxes[k + 1].logit for k in range(99))


def test_reading_order_rows_then_left_to_right():
    boxes = [(300, 100, 100, 20), (10, 52, 100, 22), (150, 50, 100, 20), (10, 100, 100, 20)]
    assert VD.reading_order(boxes) == [1, 2, 3, 0]
    import desktop as D
    assert D.reading_order(boxes) == VD.reading_order(boxes)


def test_crop_helpers():
    assert VD.crop_of((1.2, 2.7, 10.1, 5.0), 100, 100) == (1, 2, 11, 6)
    assert VD.crop_of((-5, -5, 3, 3), 100, 100) is None
    import desktop as D
    assert D.pad_of("auto", 544) == 4 and D.pad_of("auto", 272) == 2 and D.pad_of("auto", 1080) == 8
    assert D.pad_of("3", 544) == 3
    assert D.crop_box([10, 20, 50, 40], 4, 100, 100) == (6, 16, 48, 28)
    assert D.crop_box([0, 0, 5, 5], 4, 100, 100) == (0, 0, 9, 9)


def test_unit_boxes_by_centre_and_scoring():
    import desktop as D
    u = {"uid": "u", "cluster": "v1", "group": "vita", "region": [10, 10, 210, 60],
         "lines": [{"text": "あいう", "box": [20, 20, 60, 20], "cells": [[20, 20, 40, 40], [40, 20, 60, 40],
                                                                       [60, 20, 80, 40]]}]}
    crops = [(18, 18, 66, 24, 0), (300, 300, 10, 10, 1), (150, 15, 10, 30, 2)]
    inside = D.unit_boxes(u, crops)
    assert [c[4] for c in inside] == [0, 2]
    grey = np.full((400, 400), 200, np.uint8)
    for x0 in (22, 42, 62):                              # ink in every cell
        grey[24:36, x0:x0 + 14] = 20
    fr = {"set": "dev", "view": "vita", "units": [u]}
    r = D.score_frame_units(fr, grey, crops, ["あいう", "x", ""])[0]
    assert r["errors"] == 0 and r["chars"] == 3 and r["whole"] == 1 and r["boxes"] == 2 and r["junk"] == 1
    assert D.score_frame_units(fr, grey, crops, ["あい", "x", "ん"])[0]["errors"] == 1
    assert D.score_frame_units(fr, grey, crops, ["あいう", "x", "んん"])[0]["errors"] == 2


def test_box_pr_matching():
    import desktop as D
    refs = [[10, 10, 110, 30], [10, 40, 110, 60]]
    dets = [[8, 8, 112, 32], [500, 500, 520, 520], [12, 2, 60, 9]]
    t, f, m = D.box_pr(dets, refs, [[0, 0, 120, 10]])     # the third box lies in an ignore area
    assert (t, f, m) == (1, 1, 1)


def test_ruby_like():
    pytest.importorskip("torch")
    import desktop_det as DD
    lines = [[100, 50, 300, 80, 0, "text", 1.0], [120, 38, 200, 49, 0, "text", 1.0],
             [100, 100, 300, 130, 0, "text", 1.0], [400, 20, 420, 200, 1, "text", 1.0],
             [500, 38, 560, 49, 0, "text", 1.0]]
    assert DD.ruby_like(lines) == [1]                    # the small box over line 0; the lone small one stays


def _random_desknet(shape="d32", seed=0):
    torch = pytest.importorskip("torch")
    import desktop_det as DD
    torch.manual_seed(seed)
    m = DD.DeskNet(DD.SHAPES[shape])
    for mod in m.modules():                              # BatchNorm statistics that are not the identity
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 1.5)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.1, 0.1)
    with torch.no_grad():
        m.head.bias[0] = 0.5                             # some kernel cells above 0
    return m.eval()


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import desktop_det as DD
    d = tmp_path_factory.mktemp("desk")
    m = _random_desknet()
    DD.export_onnx(m, d / "d32.onnx", DD.onnx_meta(DD.SHAPES["d32"], "test"))
    (d / "d32.vdsk").write_bytes(DD.vdsk_bytes(m))
    DD.export_onnx(m, d / "d32f.onnx", DD.onnx_meta(DD.SHAPES["d32"], "test", decode=DD.PAD_RULES["frame"]))
    (d / "d32f.vdsk").write_bytes(DD.vdsk_bytes(m, decode=DD.PAD_RULES["frame"]))
    return m, d


def test_network_fold_and_onnx_export(model_files):
    torch = pytest.importorskip("torch")
    import desktop_det as DD
    m, d = model_files
    x = torch.randint(0, 256, (1, 1, 64, 96), dtype=torch.uint8)
    with torch.no_grad():
        ref = m(x)
        folded = DD.fold_bn(m)(x)
    assert ref.shape == (1, 8, 16, 24)
    assert float((ref - folded).abs().max()) < 1e-4
    det = VD.Detector(d / "d32.onnx")
    assert det.size == DD.DEFAULT_SIZE and det.min_cells == DD.DECODE["min_cells"] and det.conf_min_logit == 0.0
    g = np.random.default_rng(1).integers(0, 256, (272, 480), dtype=np.uint8)
    xin, (fx, fy, ww, hh) = det.preprocess(g)
    assert xin.shape == (1, 1, 288, 480) and (fx, fy, ww, hh) == (1.0, 1.0, 480, 272)
    assert (xin[0, 0, 272:] == xin[0, 0, 271]).all()     # edge replication
    for b in det.detect(g):
        assert 0 <= b.x and b.x + b.w <= 480 and 0 <= b.y and b.y + b.h <= 272


def test_frame_pad_rule(model_files):
    _, d = model_files
    g = np.random.default_rng(2).integers(0, 256, (544, 960), dtype=np.uint8)
    pct, frame = VD.Detector(d / "d32.onnx"), VD.Detector(d / "d32f.onnx")
    assert frame.pad_frame == 4 and frame.pad_pct == 0.0
    ink = VD.Detector(d / "d32.onnx")
    ink.pad_pct, ink.pad_min = 0.0, 0.0
    bi, bf = ink.detect(g), frame.detect(g)
    assert len(bi) == len(bf) and len(pct.detect(g)) == len(bi)
    for a, b in zip(bi, bf):                             # the ink box grown by 4 px at 544 px, clipped
        assert b.x == max(0.0, a.x - 4) and b.y == max(0.0, a.y - 4)
        assert b.x + b.w == pytest.approx(min(960.0, a.x + a.w + 4))


def test_detector_refuses_other_onnx_files(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    p = tmp_path / "other.onnx"
    onnx.save(m, str(p))
    with pytest.raises(ValueError):
        VD.Detector(p)


def test_window_scale_maps_boxes():
    pytest.importorskip("torch")
    import desktop_det as DD
    canvas = np.full((544, 960), 30, np.uint8)
    canvas[100:130, 200:600] = 230
    smp = {"lines": [[200, 100, 600, 130, 0, "dialogue", 1.0]], "ignore": [], "kind": "vita"}
    img, pos, ign = DD.make_window_ms(canvas, smp, "synth", np.random.default_rng(0), None, (1.0, 1.0), aug=False)
    assert img.shape == (256, 512)
    img2, pos2, _ = DD.make_window_ms(canvas, smp, "synth", np.random.default_rng(0), None, (0.5, 0.5), aug=False)
    assert img2.shape == (256, 480)                      # the whole 960 x 544 canvas at half size fits the width
    if pos2:
        x0, y0, x1, y1 = pos2[0][:4]
        assert abs((x1 - x0) - 200) <= 2 and abs((y1 - y0) - 15) <= 2


# ----------------------------------------------------------------------------- the C path

def _cpu_has(*flags):
    try:
        info = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    line = next((ln for ln in info.splitlines() if ln.startswith("flags")), "")
    return all(f in line.split() for f in flags)


def _build(out, arch):
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc or not shutil.which("make"):
        pytest.skip("no C compiler")
    subprocess.run(["make", "-s", "-C", str(ROOT / "desktop"), f"OUT={out}", f"ARCH={arch}", f"CC={cc}"],
                   check=True)
    return Path(out) / "libvocrdesk.so"


@pytest.fixture(scope="module")
def c_libs(tmp_path_factory):
    d = tmp_path_factory.mktemp("cbuild")
    libs = {"plain": _build(d / "plain", "")}
    if _cpu_has("avx2", "fma"):
        libs["avx2"] = _build(d / "avx2", "-march=x86-64-v3")
    return libs


def test_c_network_matches_onnx(model_files, c_libs):
    m, d = model_files
    det = VD.Detector(d / "d32.onnx")
    rng = np.random.default_rng(3)
    for name, lib in c_libs.items():
        cm = VD.CModel(d / "d32.vdsk", lib=lib)
        for h, w in ((96, 160), (288, 480)):
            x = rng.integers(0, 256, (h, w), dtype=np.uint8)
            ref, got = det.maps(x[None, None]), cm.maps(x)
            assert got.shape == ref.shape
            assert float(np.abs(ref - got).max()) <= 1e-4 * max(1.0, float(np.abs(ref).max())), name


def test_c_decode_equals_python_on_the_same_maps(model_files, c_libs):
    _, d = model_files
    rng = np.random.default_rng(4)
    cases = [_maps_with_line(64, 128, rows=[5], cols=range(4, 20)),
             _maps_with_line(64, 128, rows=range(2, 12), cols=[20], vertical=True, tag=2)]
    for _ in range(6):                                   # random fields: many components, ties, all tags
        mp = rng.normal(0, 1.5, (8, 40, 60)).astype(np.float32)
        mp[2:4] = np.abs(mp[2:4]) * 6
        cases.append(mp)
    for lib in c_libs.values():
        cm = VD.CModel(d / "d32.vdsk", lib=lib)
        for mp in cases:
            ih, iw = mp.shape[1] * 4 - 3, mp.shape[2] * 4 - 2
            assert cm.decode(mp, iw, ih) == VD.decode(mp, iw, ih)


def test_c_detect_matches_onnx_crops(model_files, c_libs):
    _, d = model_files
    rng = np.random.default_rng(5)
    for fname in ("d32", "d32f"):
        det = VD.Detector(d / f"{fname}.onnx")
        for lib in c_libs.values():
            cm = VD.CModel(d / f"{fname}.vdsk", lib=lib)
            for h, w in ((272, 480), (544, 960)):        # no resize under fit:960: the same network input
                g = rng.integers(0, 256, (h, w), dtype=np.uint8)
                a = [VD.crop_of((b.x, b.y, b.w, b.h), w, h) for b in det.detect(g)]
                b = [VD.crop_of((x.x, x.y, x.w, x.h), w, h) for x in cm.detect(g)]
                assert sorted(a) == sorted(b)


def test_c_resize_close_to_opencv(model_files, c_libs):
    cv2 = pytest.importorskip("cv2")
    _, d = model_files
    cm = VD.CModel(d / "d32.vdsk", lib=next(iter(c_libs.values())))
    g = np.random.default_rng(6).integers(0, 256, (720, 1280), dtype=np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)                 # smooth content, as frames are
    for dw, dh, interp in ((960, 540, cv2.INTER_AREA), (640, 360, cv2.INTER_AREA), (853, 480, cv2.INTER_AREA)):
        got, ref = cm.resize(g, dw, dh), cv2.resize(g, (dw, dh), interpolation=interp)
        assert int(np.abs(got.astype(int) - ref.astype(int)).max()) <= 1
    small = g[:272, :480]
    got, ref = cm.resize(small, 960, 544), cv2.resize(small, (960, 544), interpolation=cv2.INTER_LINEAR)
    assert int(np.abs(got.astype(int) - ref.astype(int)).max()) <= 1
    flat = np.full((100, 100), 77, np.uint8)
    assert (cm.resize(flat, 37, 23) == 77).all()         # the weights sum to 1


@pytest.fixture(scope="module")
def rec_blob(tmp_path_factory):
    """A small random VOCR recognizer blob (train/export.py) for the page pipeline's tests."""
    pytest.importorskip("torch")
    import torch
    from torch import nn

    import export
    from model import TinyVNOCR
    torch.manual_seed(7)
    rec = TinyVNOCR(60, width=0.25).eval()
    with torch.no_grad():
        for mod in rec.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.uniform_(0.5, 2.0)
    vocab = [chr(0x3042 + i) for i in range(59)]
    blob, _ = export.build_blob(export.model_ops(rec, 32), vocab, 32, True, 1024, b"")
    rp = tmp_path_factory.mktemp("rec") / "rec.vocr"
    rp.write_bytes(blob)
    return rp


def test_c_page_reads_like_the_c_runtime(model_files, c_libs, rec_blob):
    """The page pipeline's lines: the detector's crops in reading order, each read by runtime/vocr.c (W8I) with its
    dark flag, the same text pyvocr gives for the same crop."""
    from pyvocr import Vocr
    _, d = model_files
    rp = rec_blob
    g = np.random.default_rng(8).integers(0, 256, (272, 480), dtype=np.uint8)
    for lib in c_libs.values():
        page = VD.CDesk(d / "d32.vdsk", rp, threads=2, lib=lib)
        lines = page.page(g)
        boxes = page.detect(g)
        assert lines and len(lines) == sum(1 for b in boxes if VD.crop_of((b.x, b.y, b.w, b.h), 480, 272))
        assert any(t for t, *_ in lines)
        for threads in (1, 3):                           # the network's row bands and the lines' threads: same bits
            other = VD.CDesk(d / "d32.vdsk", rp, threads=threads, lib=lib)
            assert other.page(g) == lines and other.detect(g) == boxes
        import vocr_onnx as V
        r = Vocr(rp.read_bytes(), mode="w8i")
        for text, (x, y, w, h), _tag, dark, vertical in lines:
            if vertical:
                assert text == ""
                continue
            assert dark == V.line_dark(g, (x, y, w, h))
            assert text == r.run(np.ascontiguousarray(g[y:y + h, x:x + w]), invert=bool(dark))[0]


# ----------------------------------------------------------------------------- bounds (audit-vocr-desktop F5, N16-N20)

def test_policy_bounds_in_python():
    assert VD.policy_of("long:640") == (2, 640) and VD.policy_of("fit:960") == (1, 960)
    assert VD.policy_of("native") == (0, 0) and VD.policy_of(None) == (-1, 0)
    for bad in ("long:0", "long:8193", "long:4294967278", "fit:-5", "wide:640", "long:6.5"):
        with pytest.raises(ValueError):
            VD.policy_of(bad)
    assert VD.scale_for("long:8192", 1, 1) == 8192.0
    for bad in ("long:0", "fit:8193", "long:4294967278"):
        with pytest.raises(ValueError):
            VD.scale_for(bad, 640, 480)


def _patched_blob(path, kind, n):
    """A VDSK blob with another default policy in its header, the CRC-32 refitted (a crafted model file)."""
    import struct
    import zlib
    b = bytearray(Path(path).read_bytes())
    struct.pack_into("<II", b, 64, kind, n)
    struct.pack_into("<I", b, len(b) - 4, zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
    return bytes(b)


def test_c_refuses_bad_policies_sizes_and_blobs(model_files, c_libs, rec_blob, tmp_path):
    import ctypes as C
    _, d = model_files
    huge = 4294967278                                     # the audit's long:N whose arena size used to wrap
    for name, lib in c_libs.items():
        L = VD.c_library(lib)
        cm = VD.CModel(d / "d32.vdsk", lib=lib)
        m = cm.m.ctypes.data
        assert [L.vocr_desk_policy_ok(k, n) for k, n in ((0, 12345), (1, 1), (2, 8192), (1, 960))] == [1, 1, 1, 1]
        assert [L.vocr_desk_policy_ok(k, n) for k, n in ((2, 0), (2, 8193), (2, huge), (1, 0), (3, 640))] == [0] * 5
        for k, n, w, h in ((2, huge, 1, 1), (2, huge, 640, 480), (2, 9000, 640, 480), (7, 0, 640, 480),
                           (2, 640, 0, 480), (2, 640, 8193, 10)):
            assert L.vocr_desk_arena_bytes(m, k, n, w, h) == 0, (name, k, n, w, h)
        assert L.vocr_desk_arena_bytes(m, 2, 640, 640, 480) > 0
        out = [C.c_uint(7) for _ in range(4)]
        L.vocr_desk_input_size(2, huge, 1, 1, *[C.byref(v) for v in out])
        assert [v.value for v in out] == [0, 0, 0, 0]
        L.vocr_desk_input_size(2, 640, 1920, 1080, *[C.byref(v) for v in out])
        assert [v.value for v in out] == [640, 360, 640, 384]
        _, CSet, _ = VD._c_types()
        g = np.full((1, 1), 128, np.uint8)
        arena = np.zeros(1 << 20, np.uint8)
        rc = L.vocr_desk_detect(m, g.ctypes.data, 1, 1, 1, 2, huge, arena.ctypes.data, arena.size, C.byref(CSet()))
        assert rc == -1                                   # VOCR_DESK_E_ARG, before any resize
        with pytest.raises(ValueError):
            cm.detect(np.full((480, 640), 128, np.uint8), size="long:9000")
        # the decode and the resize check their own arguments (N17)
        assert L.vocr_desk_decode_bytes(4096, 10, 4) == 0 and L.vocr_desk_decode_bytes(2048, 2048, 4) > 0
        assert L.vocr_desk_resize_scratch(5, 3, 0, 1) == 0 and L.vocr_desk_resize_scratch(5, 3, 9000, 1) == 0
        src, dst, scr = np.zeros(15, np.uint8), np.zeros(16, np.uint8), np.zeros(256, np.int32)
        assert L.vocr_desk_resize(src.ctypes.data, 5, 3, 5, dst.ctypes.data, 0, 1, scr.ctypes.data) == -1
        with pytest.raises(ValueError):
            cm.resize(np.zeros((3, 5), np.uint8), 0, 1)
        # a model file whose default policy is out of range, or of no kind, is refused (VOCR_DESK_E_BLOB)
        for k, n in ((2, huge), (2, 0), (3, 640), (1, 8193)):
            bad = tmp_path / f"{name}_{k}_{n}.vdsk"
            bad.write_bytes(_patched_blob(d / "d32.vdsk", k, n))
            with pytest.raises(ValueError, match="-2"):
                VD.CModel(bad, lib=lib)
            with pytest.raises(ValueError, match="-2"):
                VD.CDesk(bad, rec_blob, lib=lib)
        ok = tmp_path / f"{name}_ok.vdsk"
        ok.write_bytes(_patched_blob(d / "d32.vdsk", 2, 8192))
        VD.CModel(ok, lib=lib)
        # the page pipeline: a policy out of range or of no kind (N18), and the frame's sides before any allocation
        det, rec = VD._aligned((d / "d32.vdsk").read_bytes()), VD._aligned(rec_blob.read_bytes())
        for k, n in ((2, huge), (7, 0), (1, 0)):
            err = C.c_int(0)
            p = L.vocr_desk_page_new(det.ctypes.data, len(det), rec.ctypes.data, len(rec), 1, k, n, C.byref(err))
            assert not p and err.value == -1, (k, n)
        page = VD.CDesk(d / "d32.vdsk", rec_blob, lib=lib)
        rgb = np.zeros(48, np.uint8)
        assert L.vocr_desk_page_run_rgb(page.p, rgb.ctypes.data, 65535, 65535, 65535 * 3, 3) == -1


def test_c_decode_keeps_the_first_256_like_python(model_files, c_libs):
    """More than 256 components, with equal logits and equal clipped corners (ties down to the component's index):
    the C decode's bounded selection keeps the boxes, in the order, that sorting everything and truncating keeps."""
    _, d = model_files
    mh, mw = 64, 120
    mp = np.full((8, mh, mw), -4.0, np.float32)
    mp[2:4] = 3.0
    for r in range(0, mh, 2):
        for c in range(0, mw - 1, 3):
            mp[0, r, c:c + 2] = (0.5, 1.5, 2.5)[(r // 2 + c // 3) % 3]
            if c == 0:
                mp[2, r, c:c + 2] = 500.0                 # the box reaches above the frame: clipped to y 0, x 0
    n = (mh // 2) * len(range(0, mw - 1, 3))
    assert n > 256
    ref = VD.decode(mp, mw * 4, mh * 4)
    assert len(ref) == 256
    for lib in c_libs.values():
        cm = VD.CModel(d / "d32.vdsk", lib=lib)
        assert cm.decode(mp, mw * 4, mh * 4) == ref


def test_region_reads_at_the_frame_scale_and_pad(model_files, monkeypatch):
    """read_frame's region (N20): the policy's factor and the frame pad come from the whole frame, so a region reads
    as the same region of a whole-frame read would."""
    _, d = model_files
    det = VD.Detector(d / "d32f.onnx", size="long:640")
    shapes = []
    real_maps = det.maps
    monkeypatch.setattr(det, "maps", lambda x: (shapes.append(x.shape), real_maps(x))[1])
    fixed = [VD.Box(30.0, 10.0, 60.0, 10.0, 0, 1.0, 0, 1.0, 4)]   # one box in network-input px
    monkeypatch.setattr(VD, "decode", lambda *a, **k: list(fixed))
    sub = np.full((150, 900), 128, np.uint8)
    (b,) = det.detect(sub, frame=(1920, 1080))            # 1920 x 1080 at long:640: a factor of 1/3
    assert shapes[-1] == (1, 1, 64, 320)                  # 900 x 150 -> 300 x 50, padded to 320 x 64
    pad = max(2, round(4 * 1080 / 544))                   # the frame's pad: 8 px
    assert (b.x, b.y) == (90.0 - pad, 30.0 - pad)
    assert b.x + b.w == pytest.approx(270.0 + pad) and b.y + b.h == pytest.approx(60.0 + pad)
    (b,) = det.detect(sub)                                # the region alone: its own scale and pad (2 px)
    assert shapes[-1] == (1, 1, 128, 640)
    seen = {}
    monkeypatch.setattr(det, "detect", lambda img, frame=None: seen.setdefault("frame", frame) and [])
    VD.read_frame(np.zeros((1080, 1920), np.uint8), det, None, region=(100, 800, 900, 150))
    assert seen["frame"] == (1920, 1080)


def test_export_writes_the_networks_tag_count(model_files, c_libs):
    """train/desktop_det.py's VDSK header and ONNX metadata carry the network's own tag count (N16)."""
    torch = pytest.importorskip("torch")
    import struct

    import desktop_det as DD
    torch.manual_seed(9)
    m = DD.DeskNet(DD.SHAPES["d32"], tags=2).eval()
    blob = DD.vdsk_bytes(m)
    assert struct.unpack_from("<I", blob, 56)[0] == 2
    meta = DD.onnx_meta(DD.SHAPES["d32"], "t2", tags=2)
    assert meta["vocr.tags"] == 2 and meta["vocr.classes"] == ["dialogue", "name", "choice"]
    assert "[1, 6, H / 4, W / 4]" in meta["vocr.output"]
    assert DD.onnx_meta(DD.SHAPES["d32"], "t4")["vocr.output"] == (
        "maps float32 [1, 8, H / 4, W / 4]: K, O logits, A1, A2 (input px), 4 tag logits")
    x = np.random.default_rng(10).integers(0, 256, (64, 96), dtype=np.uint8)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)[None, None]).numpy()[0]
    for lib in c_libs.values():
        got = VD.CModel(blob, lib=lib).maps(x)
        assert got.shape == ref.shape == (6, 16, 24)
        assert float(np.abs(ref - got).max()) <= 1e-3 * max(1.0, float(np.abs(ref).max()))

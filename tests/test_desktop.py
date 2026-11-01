"""The desktop pipeline (issue #169, docs/desktop.md): tools/vocr_desktop.py's size policy, decode and reading order,
train/desktop_det.py's network, BatchNorm folding and ONNX export, and eval/desktop.py's scoring helpers. No private
data: synthetic maps, random networks and generated frames only."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "train", ROOT / "eval", ROOT / "synth"):
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
    assert b.vertical == 0
    assert b.y == pytest.approx(21.5 - 6 - 2.16)
    assert b.h == pytest.approx(12 + 2 * 2.16)
    # along: cells 4..19 -> [16, 80), grown by the pad
    assert b.x == pytest.approx(16 - 2.16)
    assert b.w == pytest.approx(64 + 2 * 2.16)
    assert b.tag == 0 and b.conf > 0.95


def test_decode_tag_vertical_small_and_weak_components():
    m = _maps_with_line(64, 128, rows=range(2, 12), cols=[20], vertical=True, tag=1)
    m[0, 1, 1] = 4.0                       # a single cell: under min_cells, dropped
    boxes = VD.decode(m, 128, 64)
    assert len(boxes) == 1 and boxes[0].vertical == 1 and boxes[0].tag == 1
    weak = _maps_with_line(64, 128, rows=[5], cols=range(4, 8))
    weak[0][weak[0] > 0] = 0.01            # sigmoid(0.01) ~ 0.5025: kept at conf_min 0.5
    assert len(VD.decode(weak, 128, 64)) == 1
    assert len(VD.decode(weak, 128, 64, conf_min=0.6)) == 0
    thin = _maps_with_line(64, 128, rows=[5], cols=range(4, 8), a1=1.0, a2=1.0)
    assert len(VD.decode(thin, 128, 64)) == 0          # 2 px across, under min_across


def test_decode_clips_to_the_image_part():
    m = _maps_with_line(64, 128, rows=[1], cols=range(0, 32), a1=10.0, a2=4.0)
    b = VD.decode(m, 100, 60)[0]                        # only 100 x 60 of the 128 x 64 input is the image
    assert b.x == 0 and b.y == 0
    assert b.x + b.w <= 100 and b.y + b.h <= 60


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
    rows = D.score_frame_units(fr, grey, crops, ["あいう", "x", ""])
    r = rows[0]
    assert r["errors"] == 0 and r["chars"] == 3 and r["whole"] == 1 and r["boxes"] == 2 and r["junk"] == 1
    rows = D.score_frame_units(fr, grey, crops, ["あい", "x", "ん"])
    assert rows[0]["errors"] == 1                       # "あいん" against "あいう"
    rows = D.score_frame_units(fr, grey, crops, ["あいう", "x", "んん"])
    assert rows[0]["errors"] == 2                       # the junk box's text counts


def test_network_fold_and_onnx_export(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import desktop_det as DD
    torch.manual_seed(0)
    m = DD.DeskNet(DD.SHAPES["d32"])
    for mod in m.modules():                              # BatchNorm statistics that are not the identity
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 1.5)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.1, 0.1)
    m.eval()
    x = torch.randint(0, 256, (1, 1, 64, 96), dtype=torch.uint8)
    with torch.no_grad():
        ref = m(x)
        folded = DD.fold_bn(m)(x)
    assert ref.shape == (1, 8, 16, 24)
    assert float((ref - folded).abs().max()) < 1e-4
    path = tmp_path / "d32.onnx"
    worst = DD.export_onnx(m, path, DD.onnx_meta(DD.SHAPES["d32"], "test"))
    assert worst < 1e-3
    det = VD.Detector(path)
    assert det.size == DD.DEFAULT_SIZE and det.min_cells == DD.DECODE["min_cells"]
    g = np.random.default_rng(1).integers(0, 256, (272, 480), dtype=np.uint8)
    xin, (fx, fy, ww, hh) = det.preprocess(g)
    assert xin.shape == (1, 1, 288, 480) and (fx, fy, ww, hh) == (1.0, 1.0, 480, 272)
    assert (xin[0, 0, 272:] == xin[0, 0, 271]).all()     # edge replication
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].name == "grey"
    for b in det.detect(g):
        assert 0 <= b.x and b.x + b.w <= 480 and 0 <= b.y and b.y + b.h <= 272


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
    rng = np.random.default_rng(0)
    img, pos, ign = DD.make_window_ms(canvas, smp, "synth", rng, None, (1.0, 1.0), aug=False)
    assert img.shape == (256, 512)
    rng = np.random.default_rng(0)
    img2, pos2, _ = DD.make_window_ms(canvas, smp, "synth", rng, None, (0.5, 0.5), aug=False)
    assert img2.shape == (256, 480)                      # the whole 960 x 544 canvas at half size fits the width
    if pos2:
        x0, y0, x1, y1 = pos2[0][:4]
        assert abs((x1 - x0) - 200) <= 2 and abs((y1 - y0) - 15) <= 2

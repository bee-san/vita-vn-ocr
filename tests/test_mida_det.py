"""eval/mida_det.py, the mida_ocr_l.onnx detector backend for the desktop benchmark (#169), on generated fixtures only:
tiny ONNX graphs built here with onnx.helper (no third-party weights) and hand-made maps. The checks: the static
safety checks refuse external data, other operator domains, model-local functions and subgraphs; the size policies;
the decode's geometry (kernel, orientation, edges in stride units, the end extensions, the pads); the backend running
end to end on ONNX Runtime; its registration into a desktop.py-like module."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("cv2")

import mida_det as M  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def tiny_model(domain="", external=False, function=False, subgraph=False):
    """images (N, 3, h, w) -> maps (N, 4, ceil(h/8), ceil(w/8)): Sub 0.5, then a stride-8 8x8 conv (fixed weights)."""
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(0, 0.1, (4, 3, 8, 8)).astype(np.float32), "w")
    b = numpy_helper.from_array(np.array([1.0, -1.0, 0.5, 0.5], np.float32), "b")
    half = numpy_helper.from_array(np.array(0.5, np.float32), "half")
    nodes = [helper.make_node("Sub", ["images", "half"], ["x"]),
             helper.make_node("Conv", ["x", "w", "b"], ["maps"], strides=[8, 8], kernel_shape=[8, 8],
                              pads=[0, 0, 7, 7])]
    if domain:
        nodes.append(helper.make_node("Foo", ["maps"], ["junk"], domain=domain))
    if subgraph:
        g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "sub", [], [])
        nodes.append(helper.make_node("If", ["cond"], ["z"], then_branch=g, else_branch=g))
    graph = helper.make_graph(
        nodes, "tiny", [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["n", 3, "h", "w"])],
        [helper.make_tensor_value_info("maps", TensorProto.FLOAT, ["n", 4, "gh", "gw"])], [w, b, half])
    opsets = [helper.make_opsetid("", 17)] + ([helper.make_opsetid(domain, 1)] if domain else [])
    m = helper.make_model(graph, opset_imports=opsets, producer_name="test")
    m.ir_version = 8
    if external:
        t = m.graph.initializer[0]
        t.data_location = TensorProto.EXTERNAL
        e = t.external_data.add()
        e.key, e.value = "location", "/etc/passwd"
    if function:
        f = helper.make_function("custom", "F", ["a"], ["b"], [helper.make_node("Identity", ["a"], ["b"])],
                                 [helper.make_opsetid("", 17)])
        m.functions.append(f)
    return m


def save(m, tmp_path, name="m.onnx"):
    p = tmp_path / name
    p.write_bytes(m.SerializeToString())
    return p


def test_check_model_accepts_a_plain_graph(tmp_path):
    p = save(tiny_model(), tmp_path)
    facts = M.check_model(p, sha256=None)
    assert facts["input"] == "images" and facts["output"] == "maps"
    assert facts["opsets"] == {"ai.onnx": 17} and facts["ops"] == {"Sub": 1, "Conv": 1}
    assert facts["params"] == 4 * 3 * 8 * 8 + 4 + 1


@pytest.mark.parametrize("kw, why", [({"external": True}, "external data"), ({"domain": "com.example"}, "domain"),
                                     ({"function": True}, "functions"), ({"subgraph": True}, "subgraph")])
def test_check_model_refuses(tmp_path, kw, why):
    m = tiny_model(**kw)
    p = tmp_path / "bad.onnx"
    p.write_bytes(m.SerializeToString())
    with pytest.raises(ValueError, match=why):
        M.check_model(p, sha256=None)


def test_check_model_refuses_other_bytes(tmp_path):
    p = save(tiny_model(), tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        M.check_model(p)                      # the default expects the published file's sha256


def test_parse_spec():
    path, cfg = M.parse_spec("mida")
    assert path is None and cfg == M.DEFAULTS
    path, cfg = M.parse_spec("mida:/x/m.onnx@long:640@along=1.0@min_cells=2@order=rgb@pad=pct")
    assert path == "/x/m.onnx" and cfg["policy"] == "long:640" and cfg["along"] == 1.0
    assert cfg["min_cells"] == 2 and isinstance(cfg["min_cells"], int) and cfg["order"] == "rgb" and cfg["pad"] == "pct"
    for bad in ("mida@foo=1", "mida@order=yuv", "mida@pad=big", "meiki"):
        with pytest.raises(ValueError):
            M.parse_spec(bad)


def test_input_size():
    assert M.input_size("native", 480, 272) == (480, 272, 480, 272)
    assert M.input_size("long:640", 960, 544) == (640, 363, 640, 363)
    assert M.input_size("long:960", 480, 272) == (960, 544, 960, 544)
    assert M.input_size("fit:960", 480, 272) == (480, 272, 480, 272)          # fit never scales up
    assert M.input_size("fit:960", 1920, 1080) == (960, 540, 960, 540)
    assert M.input_size("meiki", 800, 600) == (960, 544, 725, 544)            # meikiocr's fit, zero padded
    assert M.input_size("box:960x544", 800, 600) == (725, 544, 725, 544)
    assert M.input_size("box:960x544", 480, 272) == (960, 544, 960, 544)
    assert M.input_size("stretch:960x544", 800, 600) == (960, 544, 960, 544)


def test_preprocess_order_and_padding():
    rgb = np.zeros((6, 10, 3), np.uint8)
    rgb[..., 0] = 255                                         # red
    x, fx, fy = M.preprocess(rgb, dict(M.DEFAULTS))           # bgr: red is the last channel
    assert x.shape == (1, 3, 6, 10) and (fx, fy) == (1.0, 1.0)
    assert x[0, 2].min() == 1.0 and x[0, 0].max() == 0.0
    x, _, _ = M.preprocess(rgb, dict(M.DEFAULTS, order="rgb"))
    assert x[0, 0].min() == 1.0
    x, fx, fy = M.preprocess(np.full((272, 480, 3), 128, np.uint8), dict(M.DEFAULTS, policy="meiki"))
    assert x.shape == (1, 3, 544, 960) and fx == fy == 2.0


def line_maps(gh, gw, rows, c0, c1, a1, a2, vertical=False):
    maps = np.full((4, gh, gw), -10.0, np.float32)
    maps[1] = 5.0 if vertical else -5.0
    maps[2], maps[3] = a1, a2
    for r in rows:
        if vertical:
            maps[0, c0:c1 + 1, r] = 3.0
        else:
            maps[0, r, c0:c1 + 1] = 3.0
    return maps


def test_decode_horizontal_line():
    maps = line_maps(20, 40, [5], 3, 12, 1.5, 1.25)
    cfg = dict(M.DEFAULTS, pad="none")
    (b,) = M.decode(maps, 1.0, 1.0, 320, 160, cfg)
    cy = 8 * 5 + 3.5
    assert b[:5] == [24.0, cy - 12.0, 104.0, cy + 10.0, 0] and b[5] == 3.0 and b[6] == -1
    (b,) = M.decode(maps, 1.0, 1.0, 320, 160, dict(cfg, along=0.5))           # both ends + 0.5 x 22 px
    assert b[0] == 24.0 - 11.0 and b[2] == 104.0 + 11.0
    (b,) = M.decode(maps, 1.0, 1.0, 320, 160, dict(cfg, along=0.5, ext_start=0.0))
    assert b[0] == 24.0 and b[2] == 104.0 + 11.0
    (b,) = M.decode(maps, 1.0, 1.0, 320, 160, dict(cfg, pad="frame"))         # max(2, round(4 * 160 / 544)) = 2
    assert b[:4] == [22.0, cy - 14.0, 106.0, cy + 12.0]
    (b,) = M.decode(maps, 2.0, 2.0, 160, 80, cfg)                             # an input at twice the frame's size
    assert b[:4] == [12.0, (cy - 12.0) / 2, 52.0, (cy + 10.0) / 2]


def test_decode_vertical_and_filters():
    maps = line_maps(30, 30, [7], 2, 20, 1.0, 1.0, vertical=True)
    (b,) = M.decode(maps, 1.0, 1.0, 240, 240, dict(M.DEFAULTS, pad="none"))
    cx = 8 * 7 + 3.5
    assert b[4] == 1 and b[:4] == [cx - 8.0, 16.0, cx + 8.0, 168.0]
    one = line_maps(10, 10, [4], 4, 4, 1.0, 1.0)                              # a single kernel cell
    assert len(M.decode(one, 1.0, 1.0, 80, 80, dict(M.DEFAULTS))) == 1
    assert M.decode(one, 1.0, 1.0, 80, 80, dict(M.DEFAULTS, min_cells=2)) == []
    assert M.decode(one, 1.0, 1.0, 80, 80, dict(M.DEFAULTS, thr=3.5)) == []
    two = line_maps(20, 40, [5, 9], 3, 12, 1.0, 1.0)                          # two lines, sorted by top edge
    bs = M.decode(two, 1.0, 1.0, 320, 160, dict(M.DEFAULTS))
    assert len(bs) == 2 and bs[0][1] < bs[1][1]


def test_backend_runs_end_to_end(tmp_path):
    p = save(tiny_model(), tmp_path)
    det = M.MidaDet(p, threads=1, sha256=None, policy="long:96")
    rgb = np.random.default_rng(1).integers(0, 256, (54, 96, 3), dtype=np.uint8)
    x, _, _ = M.preprocess(rgb, det.cfg)
    assert det.maps(x).shape == (4, 7, 12)
    for b in det.boxes(rgb):
        assert 0 <= b[0] < b[2] <= 96 and 0 <= b[1] < b[3] <= 54 and b[4] in (0, 1)
    r = M.macs(p, 96, 54, sha256=None)
    assert r["maps"] == [4, 7, 12] and r["macs"] == r["conv_macs"] == 4 * 7 * 12 * 3 * 8 * 8 and r["matmul_macs"] == 0
    with pytest.raises(ValueError, match="sha256"):
        M.macs(p, 96, 54)                                      # the default expects the published file


def test_install_into_a_desktop_like_module(tmp_path, monkeypatch):
    p = save(tiny_model(), tmp_path)
    calls = []
    D = types.SimpleNamespace(detector=lambda spec, threads=1, opt="all", spinning=None: calls.append(spec) or spec,
                              _init_read=lambda rec, threads: calls.append(rec), _W={},
                              session_options=lambda threads=1, opt="all", spinning=None: None)
    M.install(D, str(p))
    assert D.detector("ts8") == "ts8" and calls == ["ts8"]
    orig = M.check_model
    monkeypatch.setattr(M, "check_model", lambda path, sha256=M.SHA256: orig(path, None))
    det = D.detector("mida@long:96@min_cells=2")
    assert isinstance(det, M.MidaDet) and det.cfg["min_cells"] == 2 and det.cfg["policy"] == "long:96"
    D._init_read("x.onnx", 1)
    assert calls[-1] == "x.onnx"
    M.install(D, str(p))                                     # a second install changes nothing
    assert D.detector("ts8") == "ts8" and calls[-1] == "ts8"

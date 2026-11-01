"""eval/mida_det.py, the mida_ocr_l.onnx detector backend for the desktop benchmark (#169), on generated fixtures only:
tiny ONNX graphs built here with onnx.helper (no third-party weights) and hand-made maps. The checks: the static
safety checks refuse external data anywhere (initializers, sparse initializers, tensor, tensors and sparse-tensor
attributes, untyped attributes), attribute types outside the allowlist, other operator domains, model-local functions,
subgraphs and training_info, and ONNX Runtime only ever gets the checked bytes; the size policies and the preprocessing;
the decode's geometry (4-connected kernels, orientation, edges in stride units, the end extensions, both pads, the
mapping back per axis, clipping, the order of the boxes); the backend end to end on ONNX Runtime; its registration
into a desktop.py-like module."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("cv2")

import mida_det as M  # noqa: E402
from onnx import AttributeProto, TensorProto, helper, numpy_helper  # noqa: E402

C4 = np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1)


def ext_tensor(name, arr, location="weights.bin", entries_only=False):
    """A TensorProto whose data lives in an external file (no raw data), as an attacker would write it."""
    t = numpy_helper.from_array(arr, name)
    t.ClearField("raw_data")
    if not entries_only:
        t.data_location = TensorProto.EXTERNAL
    e = t.external_data.add()
    e.key, e.value = "location", location
    return t


def tiny_model(domain="", external=False, function=False, subgraph=False, extra_nodes=(), training_info=False):
    """images (N, 3, h, w) -> maps (N, 4, ceil(h/8), ceil(w/8)): Sub 0.5, then a stride-8 8x8 conv (fixed weights),
    then (optionally) maps + a constant c."""
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(0, 0.1, (4, 3, 8, 8)).astype(np.float32), "w")
    b = numpy_helper.from_array(np.array([1.0, -1.0, 0.5, 0.5], np.float32), "b")
    half = numpy_helper.from_array(np.array(0.5, np.float32), "half")
    out = "maps" if not extra_nodes else "m0"
    nodes = [helper.make_node("Sub", ["images", "half"], ["x"]),
             helper.make_node("Conv", ["x", "w", "b"], [out], strides=[8, 8], kernel_shape=[8, 8],
                              pads=[0, 0, 7, 7])]
    nodes += list(extra_nodes)
    if extra_nodes:
        nodes.append(helper.make_node("Add", ["m0", "c"], ["maps"]))
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
        m.graph.initializer[0].CopyFrom(ext_tensor("w", np.zeros((4, 3, 8, 8), np.float32), "/etc/passwd"))
    if function:
        f = helper.make_function("custom", "F", ["a"], ["b"], [helper.make_node("Identity", ["a"], ["b"])],
                                 [helper.make_opsetid("", 17)])
        m.functions.append(f)
    if training_info:
        ti = m.training_info.add()
        ti.initialization.CopyFrom(helper.make_graph([], "init", [], [], [ext_tensor("ti", C4)]))
    return m


def save(m, tmp_path, name="m.onnx"):
    p = tmp_path / name
    p.write_bytes(m.SerializeToString())
    return p


def sparse_constant(external):
    vals = ext_tensor("v", C4.reshape(-1)) if external else numpy_helper.from_array(C4.reshape(-1), "v")
    idx = numpy_helper.from_array(np.arange(4, dtype=np.int64), "i")
    return helper.make_node("Constant", [], ["c"], sparse_value=helper.make_sparse_tensor(vals, idx, [1, 4, 1, 1]))


def untyped_constant():
    """A Constant whose 'value' attribute holds an external tensor but declares no type (UNDEFINED)."""
    n = helper.make_node("Constant", [], ["c"])
    a = n.attribute.add()
    a.name = "value"
    a.t.CopyFrom(ext_tensor("u", C4))
    return n


def typed_constant(external, entries_only=False):
    t = ext_tensor("t", C4, entries_only=entries_only) if external else numpy_helper.from_array(C4, "t")
    return helper.make_node("Constant", [], ["c"], value=t)


@pytest.fixture
def no_sessions(monkeypatch):
    """Records every onnxruntime.InferenceSession construction (by source: a path or a bytes object)."""
    made = []
    real = ort.InferenceSession

    def spy(src, *a, **k):
        made.append(src)
        return real(src, *a, **k)

    monkeypatch.setattr(ort, "InferenceSession", spy)
    return made


def test_check_model_accepts_a_plain_graph(tmp_path):
    p = save(tiny_model(), tmp_path)
    facts, data = M.check_model(p, sha256=None)
    assert data == p.read_bytes() and facts["bytes"] == len(data)
    assert facts["input"] == "images" and facts["output"] == "maps"
    assert facts["opsets"] == {"ai.onnx": 17} and facts["ops"] == {"Sub": 1, "Conv": 1}
    assert facts["params"] == 4 * 3 * 8 * 8 + 4 + 1
    facts, _ = M.check_model(save(tiny_model(extra_nodes=[typed_constant(False)]), tmp_path, "t.onnx"), sha256=None)
    assert facts["ops"]["Constant"] == 1                      # an embedded TENSOR attribute is allowed


@pytest.mark.parametrize("make, why", [
    (lambda: tiny_model(external=True), "external data"),
    (lambda: tiny_model(domain="com.example"), "domain"),
    (lambda: tiny_model(function=True), "functions"),
    (lambda: tiny_model(subgraph=True), "subgraph"),
    (lambda: tiny_model(training_info=True), "training_info"),
    (lambda: tiny_model(extra_nodes=[typed_constant(True)]), "external data"),
    (lambda: tiny_model(extra_nodes=[typed_constant(True, entries_only=True)]), "external data"),
    (lambda: tiny_model(extra_nodes=[sparse_constant(True)]), "SPARSE_TENSOR attribute"),
    (lambda: tiny_model(extra_nodes=[sparse_constant(False)]), "SPARSE_TENSOR attribute"),
    (lambda: tiny_model(extra_nodes=[untyped_constant()]), "UNDEFINED attribute"),
])
def test_check_model_refuses(tmp_path, no_sessions, make, why):
    p = save(make(), tmp_path, "bad.onnx")
    with pytest.raises(ValueError, match=why):
        M.check_model(p, sha256=None)
    with pytest.raises(ValueError, match=why):
        M.MidaDet(p, sha256=None)
    with pytest.raises(ValueError, match=why):
        M.macs(p, 16, 16, sha256=None)
    assert no_sessions == []                                 # refused before ONNX Runtime saw it


def test_check_model_refuses_other_bytes(tmp_path, no_sessions):
    p = save(tiny_model(), tmp_path)
    for call in (lambda: M.check_model(p), lambda: M.MidaDet(p), lambda: M.macs(p, 16, 16),
                 lambda: M.detector_from_spec(f"mida:{p}")):
        with pytest.raises(ValueError, match="sha256"):
            call()                                           # the default expects the published file's sha256
    assert no_sessions == []


def test_onnxruntime_gets_the_checked_bytes(tmp_path, no_sessions):
    p = save(tiny_model(), tmp_path)
    M.MidaDet(p, sha256=None)
    assert len(no_sessions) == 1 and no_sessions[0] == p.read_bytes()     # not the path: no second read


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
    assert M.input_size("meiki", 500, 300) == (960, 544, 906, 544)            # meikiocr truncates (906.67: not 907)
    assert M.input_size("box:960x544", 800, 600) == (725, 544, 725, 544)
    assert M.input_size("box:960x544", 480, 272) == (960, 544, 960, 544)
    assert M.input_size("box:960x544", 404, 217) == (961, 516, 961, 516)     # desktop.resample_frame: the width
    assert M.input_size("box:960x544", 417, 200) == (959, 460, 959, 460)     # from the rounded height, as fit960
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
    x, fx, fy = M.preprocess(np.full((600, 800, 3), 255, np.uint8), dict(M.DEFAULTS, policy="meiki"))
    assert x.shape == (1, 3, 544, 960) and (fx, fy) == (725 / 800, 544 / 600)
    assert x[0, :, :, :725].min() == 1.0 and x[0, :, :, 725:].max() == 0.0   # content top left, zeros to the right
    x, _, _ = M.preprocess(np.full((800, 600, 3), 255, np.uint8), dict(M.DEFAULTS, policy="meiki"))
    assert x[0, :, :544, :408].min() == 1.0 and x[0, :, :, 408:].max() == 0.0


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
    (b,) = M.decode(maps, 1.0, 1.0, 320, 160, dict(cfg, along=0.5, ext_end=0.0))
    assert b[0] == 24.0 - 11.0 and b[2] == 104.0
    (b,) = M.decode(maps, 1.0, 1.0, 320, 160, dict(cfg, pad="frame"))         # max(2, round(4 * 160 / 544)) = 2
    assert b[:4] == [22.0, cy - 14.0, 106.0, cy + 12.0]
    (b,) = M.decode(maps, 1.0, 1.0, 960, 1088, dict(cfg, pad="frame"))        # round(4 * 1088 / 544) = 8: by H
    assert b[:4] == [16.0, cy - 20.0, 112.0, cy + 18.0]
    (b,) = M.decode(maps, 1.0, 1.0, 320, 160, dict(cfg, pad="pct"))           # 18% of 22 = 3.96 px on all sides
    assert b[:4] == pytest.approx([24.0 - 3.96, cy - 12.0 - 3.96, 104.0 + 3.96, cy + 10.0 + 3.96], abs=0.006)
    thin = line_maps(20, 40, [5], 3, 12, 0.5, 0.5)                            # 8 px across: the 2 px minimum
    (b,) = M.decode(thin, 1.0, 1.0, 320, 160, dict(cfg, pad="pct"))
    assert b[:4] == [22.0, cy - 6.0, 106.0, cy + 6.0]
    (b,) = M.decode(maps, 2.0, 2.0, 160, 80, cfg)                             # an input at twice the frame's size
    assert b[:4] == [12.0, (cy - 12.0) / 2, 52.0, (cy + 10.0) / 2]
    (b,) = M.decode(maps, 2.0, 4.0, 160, 40, cfg)                             # x and y scaled apart (stretch)
    assert b[:4] == [12.0, round((cy - 12.0) / 4, 2), 52.0, round((cy + 10.0) / 4, 2)]


def test_decode_across_is_the_mean():
    maps = line_maps(20, 40, [5], 3, 12, 1.0, 1.0)
    maps[2, 5, 3:6] = 4.0                                     # 3 of 10 cells think the top is 32 px up, not 8
    (b,) = M.decode(maps, 1.0, 1.0, 320, 160, dict(M.DEFAULTS, pad="none"))
    cy = 8 * 5 + 3.5
    assert b[1] == pytest.approx(cy - 8.0 * (0.7 * 1.0 + 0.3 * 4.0), abs=0.006)   # median would give cy - 8


def test_decode_vertical_and_filters():
    maps = line_maps(30, 30, [7], 2, 20, 1.0, 1.0, vertical=True)
    (b,) = M.decode(maps, 1.0, 1.0, 240, 240, dict(M.DEFAULTS, pad="none"))
    cx = 8 * 7 + 3.5
    assert b[4] == 1 and b[:4] == [cx - 8.0, 16.0, cx + 8.0, 168.0]
    (b,) = M.decode(maps, 2.0, 4.0, 120, 60, dict(M.DEFAULTS, pad="none"))    # vertical, x and y scaled apart
    assert b[:4] == [round((cx - 8.0) / 2, 2), 4.0, round((cx + 8.0) / 2, 2), 42.0]
    weak = line_maps(30, 30, [7], 2, 20, 1.0, 1.0)                            # O summed: 19 x 0.3 > 0 but mean < 0.5
    weak[1] = 0.3
    (b,) = M.decode(weak, 1.0, 1.0, 240, 240, dict(M.DEFAULTS, pad="none"))
    assert b[4] == 1
    one = line_maps(10, 10, [4], 4, 4, 1.0, 1.0)                              # a single kernel cell
    assert len(M.decode(one, 1.0, 1.0, 80, 80, dict(M.DEFAULTS))) == 1
    assert M.decode(one, 1.0, 1.0, 80, 80, dict(M.DEFAULTS, min_cells=2)) == []
    assert M.decode(one, 1.0, 1.0, 80, 80, dict(M.DEFAULTS, thr=3.5)) == []
    assert M.decode(one, 1.0, 1.0, 80, 80, dict(M.DEFAULTS, thr=3.0)) == []   # K > thr, strictly
    two = line_maps(20, 40, [5, 9], 3, 12, 1.0, 1.0)                          # two lines, sorted by top edge
    bs = M.decode(two, 1.0, 1.0, 320, 160, dict(M.DEFAULTS))
    assert len(bs) == 2 and bs[0][1] < bs[1][1]


def test_decode_is_4_connected():
    maps = np.full((4, 10, 10), -10.0, np.float32)
    maps[1], maps[2], maps[3] = -5.0, 1.0, 1.0
    maps[0, 3, 3] = maps[0, 4, 4] = 3.0                       # two cells touching only at a corner
    bs = M.decode(maps, 1.0, 1.0, 80, 80, dict(M.DEFAULTS, pad="none"))
    assert len(bs) == 2                                       # 8-connectivity would give one box


def test_decode_order_conf_and_clipping():
    maps = np.full((4, 20, 40), -10.0, np.float32)
    maps[1], maps[2], maps[3] = -5.0, 1.0, 1.0
    maps[0, 9, 2:6] = [1.0, 2.0, 3.0, 6.0]                    # lower line on the left: mean 3, max 6
    maps[0, 3, 30:34] = 3.0                                   # upper line on the right
    bs = M.decode(maps, 1.0, 1.0, 320, 160, dict(M.DEFAULTS, pad="none"))
    assert [b[0] for b in bs] == [240.0, 16.0]                # sorted by top edge, not by x
    assert bs[1][5] == 3.0                                    # conf: the mean K logit, not the max
    edge = line_maps(5, 5, [0], 0, 4, 2.0, 2.0)               # 16 px up from row 0's centre: clipped at 0
    (b,) = M.decode(edge, 1.0, 1.0, 36, 30, dict(M.DEFAULTS, along=1.0, pad="none"))
    assert b[:4] == [0.0, 0.0, 36.0, 19.5]


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

    def base_detector(spec, threads=1, opt="all", spinning=None):
        calls.append((spec, threads, opt, spinning))
        return spec

    D = types.SimpleNamespace(detector=base_detector, _init_read=lambda rec, threads: calls.append(rec), _W={},
                              session_options=lambda threads=1, opt="all", spinning=None: None)
    M.install(D, str(p))
    assert D.detector("ts8", 3, "basic", False) == "ts8" and calls == [("ts8", 3, "basic", False)]
    orig = M.check_model
    monkeypatch.setattr(M, "check_model", lambda path, sha256=M.SHA256: orig(path, None))
    det = D.detector("mida@long:96@min_cells=2")
    assert isinstance(det, M.MidaDet) and det.cfg["min_cells"] == 2 and det.cfg["policy"] == "long:96"
    D._init_read("x.onnx", 1)
    assert calls[-1] == "x.onnx"
    first = D.detector
    M.install(D, str(p))                                     # a second install changes nothing
    assert D.detector is first and D.detector("ts8") == "ts8" and calls[-1] == ("ts8", 1, "all", None)


def test_attribute_allowlist_is_what_the_docstring_says():
    assert set(M.ALLOWED_ATTRIBUTE_TYPES) == {"FLOAT", "INT", "STRING", "TENSOR", "FLOATS", "INTS", "STRINGS",
                                              "TENSORS"}
    for name in M.ALLOWED_ATTRIBUTE_TYPES:
        assert hasattr(AttributeProto, name)


@pytest.mark.parametrize("field", ["sparse_tensor", "tp"])
def test_an_allowed_type_carrying_another_field_is_refused(tmp_path, no_sessions, field):
    """The declared type is not trusted: a TENSOR attribute that also holds a sparse tensor or a type is refused."""
    n = typed_constant(False)
    a = n.attribute[0]
    if field == "sparse_tensor":
        idx = numpy_helper.from_array(np.arange(4, dtype=np.int64), "i")
        a.sparse_tensor.CopyFrom(helper.make_sparse_tensor(numpy_helper.from_array(C4.reshape(-1), "v"), idx,
                                                           [1, 4, 1, 1]))
    else:
        a.tp.CopyFrom(helper.make_tensor_type_proto(TensorProto.FLOAT, [1]))
    assert AttributeProto.AttributeType.Name(a.type) == "TENSOR"
    p = save(tiny_model(extra_nodes=[n]), tmp_path, "bad.onnx")
    with pytest.raises(ValueError, match="sparse tensor or a type"):
        M.MidaDet(p, sha256=None)
    assert no_sessions == []


@pytest.mark.parametrize("field, why", [("graphs", "subgraph"), ("sparse_tensors", "sparse tensor or a type"),
                                        ("type_protos", "sparse tensor or a type")])
def test_an_allowed_type_carrying_a_repeated_field_is_refused(tmp_path, no_sessions, field, why):
    n = typed_constant(False)
    a = n.attribute[0]
    if field == "graphs":
        a.graphs.add().CopyFrom(helper.make_graph([], "g", [], []))
    elif field == "sparse_tensors":
        idx = numpy_helper.from_array(np.arange(4, dtype=np.int64), "i")
        a.sparse_tensors.add().CopyFrom(helper.make_sparse_tensor(numpy_helper.from_array(C4.reshape(-1), "v"), idx,
                                                                  [1, 4, 1, 1]))
    else:
        a.type_protos.add().CopyFrom(helper.make_tensor_type_proto(TensorProto.FLOAT, [1]))
    assert AttributeProto.AttributeType.Name(a.type) == "TENSOR"
    p = save(tiny_model(extra_nodes=[n]), tmp_path, "bad.onnx")
    with pytest.raises(ValueError, match=why):
        M.MidaDet(p, sha256=None)
    assert no_sessions == []


def sparse_initializer_model(external):
    """A model whose constant c is a sparse initializer; external: 'values' or 'indices' live in a file next to it
    (a relative location: the case ONNX Runtime 1.30 reads)."""
    vals = ext_tensor("c_v", C4.reshape(-1), "weights.bin") if external == "values" else \
        numpy_helper.from_array(C4.reshape(-1), "c_v")
    idx = ext_tensor("c_i", np.arange(4, dtype=np.int64), "weights.bin") if external == "indices" else \
        numpy_helper.from_array(np.arange(4, dtype=np.int64), "c_i")
    m = tiny_model(extra_nodes=[helper.make_node("Identity", ["cs"], ["c"])])
    st = m.graph.sparse_initializer.add()
    st.CopyFrom(helper.make_sparse_tensor(vals, idx, [1, 4, 1, 1]))
    st.values.name = "cs"
    return m


@pytest.mark.parametrize("external", ["values", "indices"])
def test_a_sparse_initializer_with_external_data_is_refused(tmp_path, no_sessions, external):
    p = save(sparse_initializer_model(external), tmp_path, "bad.onnx")
    (tmp_path / "weights.bin").write_bytes(np.zeros(64, np.uint8).tobytes())
    with pytest.raises(ValueError, match="sparse initializer has external data"):
        M.MidaDet(p, sha256=None)
    with pytest.raises(ValueError, match="sparse initializer has external data"):
        M.macs(p, 16, 16, sha256=None)
    assert no_sessions == []


def test_a_sparse_initializer_with_embedded_data_is_allowed(tmp_path):
    facts, _ = M.check_model(save(sparse_initializer_model(None), tmp_path), sha256=None)
    assert facts["ops"]["Identity"] == 1


def test_a_tensors_attribute_with_one_external_member_is_refused(tmp_path, no_sessions):
    n = helper.make_node("Identity", ["m0"], ["unused"])
    n.attribute.append(helper.make_attribute("extra", [numpy_helper.from_array(C4, "ok"), ext_tensor("bad", C4)]))
    assert AttributeProto.AttributeType.Name(n.attribute[0].type) == "TENSORS"
    m = tiny_model(extra_nodes=[typed_constant(False), n])
    with pytest.raises(ValueError, match="extra has external data"):
        M.MidaDet(save(m, tmp_path, "bad.onnx"), sha256=None)
    assert no_sessions == []


def test_the_external_flag_alone_is_refused(tmp_path, no_sessions):
    m = tiny_model()
    t = m.graph.initializer[0]
    t.ClearField("raw_data")
    t.data_location = TensorProto.EXTERNAL                    # no external_data entries at all
    assert len(t.external_data) == 0
    with pytest.raises(ValueError, match="external data"):
        M.MidaDet(save(m, tmp_path, "bad.onnx"), sha256=None)
    assert no_sessions == []


def test_an_opset_import_alone_is_refused(tmp_path, no_sessions):
    m = tiny_model()
    m.opset_import.append(helper.make_opsetid("com.example", 1))   # no node of that domain
    with pytest.raises(ValueError, match="opset imports"):
        M.MidaDet(save(m, tmp_path, "bad.onnx"), sha256=None)
    assert no_sessions == []


def test_macs_gives_onnxruntime_bytes(tmp_path, no_sessions):
    p = save(tiny_model(), tmp_path)
    M.macs(p, 16, 16, sha256=None)
    assert len(no_sessions) == 1 and isinstance(no_sessions[0], bytes)


def test_frame_pad_rounds_as_desktop_does():
    """eval/desktop.py's auto pad, max(2, round(4 H / 544)): 8 px at H = 1080 (7.94), where int() would give 7."""
    maps = line_maps(20, 40, [5], 3, 12, 1.5, 1.25)
    cy = 8 * 5 + 3.5
    (b,) = M.decode(maps, 1.0, 1.0, 1920, 1080, dict(M.DEFAULTS, pad="frame"))
    assert b[:4] == [16.0, cy - 20.0, 112.0, cy + 18.0]
    (b,) = M.decode(maps, 1.0, 1.0, 1366, 768, dict(M.DEFAULTS, pad="frame"))    # 5.65 -> 6, int() 5
    assert b[:4] == [18.0, cy - 18.0, 110.0, cy + 16.0]


def test_decode_drops_inverted_and_empty_boxes():
    inverted = line_maps(20, 40, [5], 3, 12, -2.0, -2.0)                    # far <= near
    assert M.decode(inverted, 1.0, 1.0, 320, 160, dict(M.DEFAULTS, pad="none")) == []
    slight = line_maps(20, 40, [5], 3, 12, -0.1875, -0.1875)                # far - near = -3 px: an 8 px frame pad
    assert M.decode(slight, 1.0, 1.0, 960, 1088, dict(M.DEFAULTS, pad="frame")) == []   # would leave it 13 px tall
    outside = line_maps(20, 40, [5], 30, 35, 1.0, 1.0)                      # x 240-288, clipped by a 200 px frame
    assert M.decode(outside, 1.0, 1.0, 200, 160, dict(M.DEFAULTS, pad="none")) == []


def test_area_interpolation_only_shrinks():
    rgb = np.random.default_rng(2).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    up_area, _, _ = M.preprocess(rgb, dict(M.DEFAULTS, policy="long:120", interp="area"))
    up_lin, _, _ = M.preprocess(rgb, dict(M.DEFAULTS, policy="long:120", interp="linear"))
    assert np.array_equal(up_area, up_lin)                    # upscaling stays bilinear
    dn_area, _, _ = M.preprocess(rgb, dict(M.DEFAULTS, policy="long:23", interp="area"))   # not an exact halving:
    dn_lin, _, _ = M.preprocess(rgb, dict(M.DEFAULTS, policy="long:23", interp="linear"))  # there the two agree
    assert dn_area.shape == dn_lin.shape and not np.array_equal(dn_area, dn_lin)

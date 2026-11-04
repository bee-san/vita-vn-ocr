"""textspot-desktop's C ensemble detector (issue #173, protocol v2 amendment 21): vocr_tsd_ens_detect /
vocr_tsd_ens_maps (desktop/vocr_tsd.c), tools/tsdesk_c.CModel on several blobs and eval's ctsdens: spec, against
tools/tsdesk.EnsembleDetector. EnsembleDetector runs its members through ONNX Runtime; here its sessions are replaced by
shims that return each member's own C maps (float32 blobs) or tsdesk_quant.q8_maps (int8 blobs), so the Python
ensemble averages, decodes, refines and post-processes exactly the maps the C members compute: the C ensemble's maps
and boxes must equal it bit for bit. No private data: random networks and generated frames only; the C libraries are
built in temporary directories (skipped without a C compiler)."""
import ctypes as C
from pathlib import Path

import numpy as np
import pytest

from test_tsdesk_c import C6_DECODE, C6_POST, TC, T, _build, _cpu_has, _random_net, _text_frame

VNNI = ("avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx2", "fma")


@pytest.fixture(scope="module")
def elibs(tmp_path_factory):
    """{name: library}: plain C, AVX2 and AVX-512 VNNI (each when the CPU has it)."""
    d = tmp_path_factory.mktemp("tsdens")
    out = {"plain": _build(d / "plain", "")}
    if _cpu_has("avx2", "fma"):
        out["avx2"] = _build(d / "avx2", "-march=x86-64-v3")
    if _cpu_has(*VNNI):
        out["vnni"] = _build(d / "vnni", "-march=x86-64-v4 -mavx512vnni")
    return out


@pytest.fixture(scope="module")
def members(tmp_path_factory):
    """{name: (onnx path, float32 blob, int8 blob)}: two p8a networks of different seeds and a p8g (one output format:
    head stride 8, 10 maps) and a p4a (head stride 4: another format)."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import tsdesk_blob as TB
    import tsdesk_net as N
    import tsdesk_quant as Q
    d = tmp_path_factory.mktemp("tsdensnets")
    rng = np.random.default_rng(0)
    xs = [rng.integers(0, 256, (h, w), dtype=np.uint8) for h, w in ((96, 160), (64, 224))]
    out = {}
    for name, shape, seed in (("a", "p8a", 21), ("b", "p8a", 22), ("g", "p8g", 23), ("s4", "p4a", 24)):
        m = _random_net(shape, seed=seed, k_quantile=0.45)
        p = d / f"{name}.onnx"
        N.export_onnx(m, p, N.onnx_meta(m.shape, name, "fit:960", T.DEFAULTS), check=False)
        v1 = TB.blob_bytes(m, "fit:960", T.DEFAULTS)
        cal = Q.Calib(Q.FloatNet(v1, 2))
        cal.pass_max(xs)
        cal.pass_hist(xs)
        out[name] = (p, v1, Q.q8_blob(v1, cal.scales("mse")))
    return out


class _Shim:
    """An ONNX Runtime session's interface (run, get_inputs, get_modelmeta) returning a C member's maps."""

    class _In:
        name = "grey"

    def __init__(self, maps_fn, meta_sess):
        self.maps_fn, self.meta_sess = maps_fn, meta_sess

    def get_inputs(self):
        return [self._In()]

    def get_modelmeta(self):
        return self.meta_sess.get_modelmeta()

    def run(self, _outputs, feeds):
        return [self.maps_fn(next(iter(feeds.values())))[None]]


def _py_ensemble(members, names, kind, lib, **kw):
    """tools/tsdesk.EnsembleDetector of the members' ONNX files whose sessions return the members' C maps
    (kind "f32": the float32 blob through the C network) or the numpy twin's (kind "q8": tsdesk_quant.q8_maps)."""
    import tsdesk_quant as Q
    det = T.EnsembleDetector([members[n][0] for n in names], resize_mode="exact", **kw)

    def maps_of(name, k):
        if k == "q8":
            blob = members[name][2]
            return lambda x: Q.q8_maps(blob, x)
        cm = TC.CModel(members[name][1], lib=lib)
        return lambda x: cm.maps(x)
    kinds = kind if isinstance(kind, (list, tuple)) else [kind] * len(names)
    shims = [_Shim(maps_of(n, k), det.sess) for n, k in zip(names, kinds)]
    det.sess, det.others = shims[0], shims[1:]
    det.input_name = "grey"
    return det


def _blobs(members, names, kind):
    kinds = kind if isinstance(kind, (list, tuple)) else [kind] * len(names)
    return [members[n][2] if k == "q8" else members[n][1] for n, k in zip(names, kinds)]


@pytest.mark.parametrize("kind", ["f32", "q8", ("f32", "q8")])
def test_c_ensemble_maps_equal_ensemble_detector(members, elibs, kind):
    """CModel([A, B]).maps(x) (vocr_tsd_ens_maps) == EnsembleDetector.maps(x) on the members' own maps, bit for bit:
    two and three members, float32, int8 or mixed, at 1 and 3 threads."""
    rng = np.random.default_rng(31)
    lib = elibs.get("avx2") or elibs["plain"]
    for names in (("a", "b"), ("a", "g", "b")):
        k = kind if isinstance(kind, str) else (kind * 2)[:len(names)]
        det = _py_ensemble(members, names, k, lib)
        for h, w in ((96, 160), (288, 512)):
            x = rng.integers(0, 256, (h, w), dtype=np.uint8)
            x[h // 3:h // 3 + 10, 16:w - 16] = 230
            want = det.maps(x[None, None])
            for threads in (1, 3):
                got = TC.CModel(_blobs(members, names, k), threads=threads, lib=lib).maps(x)
                assert got.dtype == np.float32 and got.shape == want.shape
                assert np.array_equal(got, want), (names, k, h, w, threads)


def test_c_ensemble_detect_equals_ensemble_detector(members, elibs):
    """CModel([A, B]).detect == EnsembleDetector.detect on the members' own maps, bit for bit: the frame's luma, the
    integer resize and padding once, the averaged maps, one decode (vote_pct), the frame-px boxes, post_pre, the
    refinement and post_post; RGB and grey frames, resized or not, float32 and int8 members."""
    lib = elibs.get("avx2") or elibs["plain"]
    n_boxes = 0
    for kind in ("f32", "q8"):
        for size in ("native", "long:960"):
            det = _py_ensemble(members, ("a", "b"), kind, lib, size=size, refine=True, post=C6_POST,
                               decode_opts=C6_DECODE)
            cm = TC.CModel(_blobs(members, ("a", "b"), kind), lib=lib)
            for (h, w) in ((300, 541), (544, 960)):
                rgb = _text_frame(w, h, w + h)
                want = det.detect(rgb)
                got = cm.detect(rgb, size, refine=True, decode=C6_DECODE, post=C6_POST)
                assert got == want, (kind, size, (h, w))
                assert cm.detect(T.grey_of(rgb), size, refine=True, decode=C6_DECODE, post=C6_POST) == got
                n_boxes += len(got)
            plain = _py_ensemble(members, ("a", "b"), kind, lib, size=size)    # no options: the plain decode
            rgb = _text_frame(541, 300, 7)
            assert cm.detect(rgb, size) == plain.detect(rgb), (kind, size)
    assert n_boxes > 20


def test_c_ensemble_builds_and_threads_same_boxes(members, elibs):
    """The int8 ensemble gives the same boxes in every build (plain, AVX2, VNNI) at 1 and 4 threads; the float32 one
    at every thread count of a build."""
    frame = _text_frame(541, 300, 11)
    ref = TC.CModel(_blobs(members, ("a", "b"), "q8"), lib=elibs["plain"]).detect(frame, "long:960", True, C6_DECODE,
                                                                                  C6_POST)
    for name, lib in elibs.items():
        for threads in (1, 4):
            cm = TC.CModel(_blobs(members, ("a", "b"), "q8"), threads=threads, lib=lib)
            assert cm.detect(frame, "long:960", True, C6_DECODE, C6_POST) == ref, (name, threads)
        f1 = TC.CModel(_blobs(members, ("a", "g"), "f32"), lib=lib).detect(frame, "long:960", True)
        assert TC.CModel(_blobs(members, ("a", "g"), "f32"), threads=3, lib=lib).detect(frame, "long:960", True) == f1


def _ens_detect(cm_models, lib, frame, size, opt_owner):
    """vocr_tsd_ens_detect called directly on the models' structs (any count, also 1)."""
    L = TC.c_library(lib)
    a, w, h, stride, ch = TC._image(frame)
    kind, n = TC.policy_of(size, opt_owner.default_policy)
    ptrs = (C.c_void_p * len(cm_models))(*[m.ctypes.data for m in cm_models])
    opt, keep = opt_owner.det_opts(True, C6_DECODE, C6_POST)
    need = L.vocr_tsd_ens_arena_bytes(ptrs, len(cm_models), kind, n, w, h, ch, 1, C.byref(opt))
    if not need:
        return None
    arena = np.zeros(need + 64, np.uint8)
    st = TC.CSet()
    rc = L.vocr_tsd_ens_detect(ptrs, len(cm_models), a.ctypes.data, w, h, stride, ch, kind, n, C.byref(opt),
                               arena.ctypes.data, arena.size, C.byref(st), None, None, 1)
    del keep
    return rc, TC._boxes(st)


def test_c_ensemble_of_one_is_the_detector_and_formats_must_match(members, elibs):
    """vocr_tsd_ens_detect with one model == vocr_tsd_detect_opt bit for bit (float32: the lazy head too); models of
    different output formats (head stride 4 and 8) and counts outside 1 .. 8 are refused."""
    lib = elibs.get("avx2") or elibs["plain"]
    frame = _text_frame(541, 300, 5)
    for blob in (members["a"][1], members["a"][2]):
        cm = TC.CModel(blob, lib=lib)
        rc, got = _ens_detect([cm.m], lib, frame, "long:960", cm)
        assert rc == 0 and got == cm.detect(frame, "long:960", True, C6_DECODE, C6_POST)
    a, s4 = TC.CModel(members["a"][1], lib=lib), TC.CModel(members["s4"][1], lib=lib)
    assert _ens_detect([a.m, s4.m], lib, frame, "long:960", a) is None        # 0 arena bytes: refused
    assert _ens_detect([a.m] * 9, lib, frame, "long:960", a) is None
    with pytest.raises(ValueError, match="one output format"):
        TC.CModel([members["a"][1], members["s4"][1]], lib=lib)
    with pytest.raises(ValueError, match="1 to 8"):
        TC.CModel([members["a"][1]] * 9, lib=lib)


def test_ctsdens_spec_in_eval(members, elibs, tmp_path, monkeypatch):
    """eval's ctsdens:A.vtsd|B.vtsd@SIZE+options spec is the C ensemble (CModel on the members, the options parsed as
    ctsd:'s)."""
    import importlib.util
    import sys
    pytest.importorskip("cv2")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(1, str(root / "eval"))
    try:
        spec = importlib.util.spec_from_file_location("tsdesk_eval", root / "eval" / "tsdesk_eval.py")
        E = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(E)
    except ImportError as e:
        pytest.skip(f"eval/tsdesk_eval.py: {e}")
    finally:
        sys.path.remove(str(root / "eval"))
    lib = elibs.get("avx2") or elibs["plain"]
    monkeypatch.setenv("VOCR_DESK_LIB", str(lib))
    pa, pb = tmp_path / "a.vtsd", tmp_path / "b.vtsd"
    pa.write_bytes(members["a"][2])
    pb.write_bytes(members["b"][2])
    det = E.detector_from_spec(f"ctsdens:{pa}|{pb}@long:960+refine+dec:vote_pct=70+post:merge_pct=70,rows_pct=65,"
                               "valley_pct=0,ext_pct=1200")
    frame = _text_frame(541, 300, 9)
    want = TC.CModel([members["a"][2], members["b"][2]], lib=lib).detect(frame, "long:960", True, C6_DECODE, C6_POST)
    assert det.boxes(frame) == [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag] for b in want]
    single = E.detector_from_spec(f"ctsd:{pa}@long:960")
    assert single.m.nmod == 1 and det.m.nmod == 2

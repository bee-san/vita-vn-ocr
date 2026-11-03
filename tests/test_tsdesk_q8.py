"""textspot-desktop's int8 detector (issue #173, round 3): train/tsdesk_quant.py's quantization and numpy twin, the
VTSD version-2 blob (train/tsdesk_blob.py) and desktop/vocr_tsd_q8.c. The C int8 network gives the twin's maps bit for
bit in every build (plain C, AVX2, AVX-512 VNNI when the CPU has it) at every thread count, for every shape of
train/tsdesk_net.SHAPES and the extra shapes; its detector runs the same decode on those maps. No private data: random
networks calibrated on random frames; the C libraries are built in temporary directories (skipped without a C
compiler)."""
import struct
import zlib

import numpy as np
import pytest

from test_tsdesk_c import TC, T, _build, _cpu_has, _random_net, _shapes, _text_frame

VNNI = ("avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx2", "fma")


@pytest.fixture(scope="module")
def qlibs(tmp_path_factory):
    """{name: library}: plain C, AVX2 (when the CPU has it) and AVX-512 VNNI (when the CPU has it)."""
    d = tmp_path_factory.mktemp("tsdq")
    out = {"plain": _build(d / "plain", "")}
    if _cpu_has("avx2", "fma"):
        out["avx2"] = _build(d / "avx2", "-march=x86-64-v3")
    if _cpu_has(*VNNI):
        out["vnni"] = _build(d / "vnni", "-march=x86-64-v4 -mavx512vnni")
    return out


def _calibrated(blob, seed=0, method="mse"):
    """(v2 blob, scales) of a v1 blob calibrated on two random frames."""
    import tsdesk_quant as Q
    rng = np.random.default_rng(seed)
    xs = [rng.integers(0, 256, (h, w), dtype=np.uint8) for h, w in ((96, 160), (64, 224))]
    cal = Q.Calib(Q.FloatNet(blob, 2))
    cal.pass_max(xs)
    cal.pass_hist(xs)
    sa = cal.scales(method)
    return Q.q8_blob(blob, sa), sa


@pytest.fixture(scope="module")
def qnets():
    """{shape: (v1 blob, v2 blob)} of random networks (K calibrated to ~45% kernel cells)."""
    pytest.importorskip("torch")
    import tsdesk_blob as TB
    out = {}
    for i, shape in enumerate(sorted(_shapes())):
        m = _random_net(shape, seed=100 + i, k_quantile=0.45)
        v1 = TB.blob_bytes(m, "fit:960", T.DEFAULTS)
        out[shape] = (v1, _calibrated(v1, seed=i)[0])
    return out


def test_q8_blob_round_trip_and_checks(qnets, qlibs):
    import tsdesk_blob as TB
    import tsdesk_quant as Q
    v1, v2 = qnets["q8b"]
    h1, h2 = TB.read_header(v1), TB.read_header(v2)
    assert h1["version"] == 1 and h2["version"] == 2
    assert {k: v for k, v in h1.items() if k not in ("version", "bytes")} == \
        {k: v for k, v in h2.items() if k not in ("version", "bytes")}
    assert h2["bytes"] < h1["bytes"] * 0.45     # int8 weights (gc keeps its float32 weights; the scales add a bit)
    _, sa = _calibrated(v1, seed=3)
    q = Q.quantize(v1, sa)
    hq, back = TB.read_q8(TB.blob_bytes_q8(v1, q))
    assert hq == TB.read_header(TB.blob_bytes_q8(v1, q))
    for name, e in q.items():
        for k, v in e.items():
            want = np.asarray(v, np.float32) if k in ("m", "r", "sx", "so", "b") or name == "gc" else np.asarray(v)
            assert np.array_equal(np.asarray(back[name][k]).astype(want.dtype), want), (name, k)
    assert all(back[L["name"]]["w"].dtype == np.int8 for L in TB.layer_list(h2) if L["kind"] != "gc")
    lib = next(iter(qlibs.values()))
    TC.CModel(v2, lib=lib)
    bad = bytearray(v2)
    bad[400] ^= 1
    with pytest.raises(ValueError, match="-3"):                 # the CRC
        TC.CModel(bytes(bad), lib=lib)
    with pytest.raises(ValueError, match="-2"):                 # truncated
        TC.CModel(v2[:-8], lib=lib)
    for off, val in ((156, 0), (156, 2), (4, 3)):              # the scheme, the version
        b = bytearray(v2)
        struct.pack_into("<I", b, off, val)
        struct.pack_into("<I", b, len(b) - 4, zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
        with pytest.raises(ValueError, match="-2"):
            TC.CModel(bytes(b), lib=lib)
    with pytest.raises(ValueError):                             # tsdesk_blob refuses them too
        TB.read_header(bytes(bad))


def test_q8_loader_refuses_bad_scales(qnets, qlibs):
    """A multiplier that is negative or not finite, or a bias past 2^30, is refused (vocr_tsd_load: -2)."""
    import tsdesk_blob as TB
    lib = next(iter(qlibs.values()))
    v1, v2 = qnets["x4"]
    h, q = TB.read_q8(v2)
    off = TB.HEADER + q["stem"]["w"].size + 4 * len(q["stem"]["bq"])       # the stem's first multiplier
    for val in (-1.0, float("inf"), float("nan")):
        b = bytearray(v2)
        struct.pack_into("<f", b, off, val)
        struct.pack_into("<I", b, len(b) - 4, zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
        with pytest.raises(ValueError, match="-2"):
            TC.CModel(bytes(b), lib=lib)
    b = bytearray(v2)
    struct.pack_into("<i", b, TB.HEADER + q["stem"]["w"].size, (1 << 30) + 1)
    struct.pack_into("<I", b, len(b) - 4, zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
    with pytest.raises(ValueError, match="-2"):
        TC.CModel(bytes(b), lib=lib)


@pytest.mark.parametrize("shape", sorted(_shapes()))
def test_q8_c_maps_equal_twin(shape, qnets, qlibs):
    """Every build at 1 and 3 threads gives tsdesk_quant.q8_maps's maps bit for bit."""
    import tsdesk_quant as Q
    _, v2 = qnets[shape]
    rng = np.random.default_rng(11)
    sizes = [(64, 96), (96, 160), (288, 512)] + ([(544, 960)] if shape in ("q8b", "x4") else [])
    for h, w in sizes:
        x = rng.integers(0, 256, (h, w), dtype=np.uint8)
        x[h // 3:h // 3 + 12, w // 5:w - w // 5] = 240        # some structure on the noise
        want = Q.q8_maps(v2, x)
        for name, lib in qlibs.items():
            for threads in (1, 3):
                got = TC.CModel(v2, threads=threads, lib=lib).maps(x)
                assert got.dtype == np.float32 and got.shape == want.shape
                assert np.array_equal(got, want), (shape, h, w, name, threads)


def test_q8_maps_close_to_float(qnets):
    """The int8 maps follow the float network's: on frames of the calibration's kind, |dK| is a few percent of K's
    spread and the K sign agrees on most cells."""
    import tsdesk_quant as Q
    rng = np.random.default_rng(7)
    for shape in ("q8b", "q4b", "x8"):
        v1, v2 = qnets[shape]
        net = Q.FloatNet(v1, 2)
        x = rng.integers(0, 256, (96, 160), dtype=np.uint8)
        fm, _ = net(x)
        qm = Q.q8_maps(v2, x)
        k = fm[0]
        assert float(np.abs(qm[0] - k).mean()) < 0.1 * float(k.std()), shape
        assert float(((qm[0] > 0) == (k > 0)).mean()) > 0.9, shape


def test_q8_detect_same_boxes_every_build_and_thread_count(qnets, qlibs):
    """The detector on an int8 blob: the decode of its own maps mapped to frame px (vocr_tsd_detect's rule), the same
    boxes in every build and at every thread count, refined or not, RGB or grey."""
    import tsdesk_quant as Q
    for shape in ("q8b", "x4"):
        _, v2 = qnets[shape]
        frame = _text_frame(541, 300, 3)
        ref = TC.CModel(v2, lib=qlibs["plain"])
        x, (fx, fy, iw, ih) = ref.preprocess(frame, "long:960")
        want = []
        for b in ref.decode(Q.q8_maps(v2, x), iw, ih):
            x0, y0 = max(0.0, b.x / fx), max(0.0, b.y / fy)
            x1, y1 = min(541.0, (b.x + b.w) / fx), min(300.0, (b.y + b.h) / fy)
            if x1 > x0 and y1 > y0:
                want.append((x0, y0, x1 - x0, y1 - y0, b.vertical, b.tag))
        got = [(b.x, b.y, b.w, b.h, b.vertical, b.tag) for b in ref.detect(frame, "long:960")]
        assert got == want, shape
        grey = T.grey_of(frame)
        for size, refine in (("long:960", None), ("native", True)):
            base = ref.detect(frame, size, refine)
            assert ref.detect(grey, size, refine) == base, (shape, size)
            for name, lib in qlibs.items():
                for threads in (1, 4):
                    assert TC.CModel(v2, threads=threads, lib=lib).detect(frame, size, refine) == base, \
                        (shape, size, name, threads)


def test_q8_simd_reports_the_build(qlibs):
    want = {"plain": 0, "avx2": 1, "vnni": 2}
    for name, lib in qlibs.items():
        assert TC.c_library(lib).vocr_tsd_q8_simd() == want[name], name


def test_calib_methods_clip_inside_the_range():
    """Every calibration method clips at most at the tensor's max; percentiles grow with p; mse's clip is positive."""
    pytest.importorskip("torch")
    import tsdesk_blob as TB
    import tsdesk_quant as Q
    m = _random_net("p8a", seed=5)
    blob = TB.blob_bytes(m, "fit:960", T.DEFAULTS)
    rng = np.random.default_rng(2)
    cal = Q.Calib(Q.FloatNet(blob, 2))
    xs = [rng.integers(0, 256, (64, 96), dtype=np.uint8)]
    cal.pass_max(xs)
    cal.pass_hist(xs)
    for k in cal.names:
        c = {meth: cal.clip(k, meth) for meth in Q.CALIB_METHODS}
        assert all(0 < v <= cal.max[k] * (1 + 1e-9) for v in c.values()), k
        assert c["p99.9"] <= c["p99.99"] <= c["p99.999"] <= c["max"], k
    with pytest.raises(ValueError):
        cal.clip(cal.names[0], "p")

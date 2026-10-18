"""Tests of the learned text detector's integer twin (synth/textdet_twin.py), its blob (train/textdet_export.py), the
protocol's sweep table (eval/textdet_protocol.json) and the C runtime (runtime/vocr_textdet.c) against the twin.

The decode on hand-made maps (a horizontal line, a vertical one, lines close together, the 4-px drop, padding and
clipping, the reading order of side-by-side boxes, the 64-candidate and 16-line caps, floor and ceil of negative
sums), the blob's round trip and every loader rejection, export's quantize_multiplier, and the sweep's MACs and
parameters. The C parts compile the host driver with the system compiler (cc, or $CC) and are skipped without one:
the C loader must refuse exactly the blobs the twin refuses, and a small tests/textdet_parity.py run must pass. No
screenshots, fonts or models; no network.
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_parity as P  # noqa: E402
import textdet_twin as TW  # noqa: E402

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
PROTOCOL = json.loads((ROOT / "eval" / "textdet_protocol.json").read_text())


# ---------------------------------------------------------------- the decode on hand-made maps

def maps_of(hm, wm):
    """Empty maps: K = -16 (no kernel), O = -16 (horizontal), A1 = A2 = 0."""
    m = np.zeros((4, hm, wm), np.int64)
    m[0] = -16
    m[1] = -16
    return m


def put(m, rows, cols, k=10, o=-16, a1=12, a2=12):
    m[0][rows, cols] = k
    m[1][rows, cols] = o
    m[2][rows, cols] = a1
    m[3][rows, cols] = a2


S4 = SimpleNamespace(stride=4, min_cells=2)
S8 = SimpleNamespace(stride=8, min_cells=1)


def test_a_horizontal_line():
    m = maps_of(10, 30)
    put(m, 3, slice(5, 15))           # c2 = 2*4*3 + 3 = 27: near floor((270-120)/20) = 7, far ceil(390/20) = 20
    assert TW.decode(S4, m, 40, 120) == ([(18, 5, 44, 17, 0)], False)   # pad 2; along [20, 60)


def test_a_vertical_line():
    m = maps_of(10, 30)
    put(m, slice(1, 9), 20, o=16, a1=8, a2=8)   # c2 = 163: near floor(1240/16) = 77, far ceil(1368/16) = 86
    assert TW.decode(S4, m, 40, 120) == ([(75, 2, 13, 36, 1)], False)


def test_orientation_is_the_sign_of_the_summed_votes():
    m = maps_of(10, 30)
    put(m, 3, slice(5, 15), o=16)
    m[1][3, 5:9] = -40                # 4 x -40 + 6 x 16 = -64: horizontal
    assert TW.decode(S4, m, 40, 120)[0][0][4] == 0
    m[1][3, 5:9] = -24                # -96 + 96 = 0: not above 0, horizontal
    assert TW.decode(S4, m, 40, 120)[0][0][4] == 0
    m[1][3, 5] = -23                  # +1: vertical
    assert TW.decode(S4, m, 40, 120)[0][0][4] == 1


def test_two_lines_close_together_stay_apart_and_touching_ones_merge():
    m = maps_of(10, 30)
    put(m, 3, slice(5, 15), a1=4, a2=4)
    put(m, 5, slice(5, 15), a1=4, a2=4)        # one row of non-kernel cells between them
    boxes, _ = TW.decode(S4, m, 40, 120)
    assert [b[1] for b in boxes] == [9, 17] and len(boxes) == 2
    m[0][4, 9] = 10                             # one kernel cell joins them (4-connected)
    boxes, _ = TW.decode(S4, m, 40, 120)
    assert len(boxes) == 1
    m[0][4, 9] = -16
    m[0][4, 4] = 10                             # touches row 3 and row 5 only by its corners: no join
    m[0][2, 3] = m[0][6, 15] = 10               # single cells: under min_cells
    assert len(TW.decode(S4, m, 40, 120)[0]) == 2


def test_the_4px_drop():
    m = maps_of(10, 30)
    put(m, 3, slice(5, 15), a1=4, a2=3)         # near floor((270-40)/20) = 11, far ceil(300/20) = 15: 4 px, kept
    assert len(TW.decode(S4, m, 40, 120)[0]) == 1
    put(m, 3, slice(5, 15), a1=3, a2=3)         # 3 px across: dropped
    assert TW.decode(S4, m, 40, 120) == ([], False)


def test_pad_and_clipping():
    m = maps_of(20, 40)
    put(m, 10, slice(10, 20), a1=50, a2=50)     # c2 = 83: near floor(330/20) = 16, far ceil(1330/20) = 67
    (x, y, w, h, v), = TW.decode(S4, m, 80, 160)[0]
    assert (y, h) == (16 - 9, 67 + 9 - 7) and (x, w) == (40 - 9, 80 + 9 - 31)   # pad (51 * 18 + 50) // 100 = 9
    m = maps_of(5, 8)
    put(m, 0, slice(0, 8), a1=40, a2=40)        # at the top-left and right borders: clipped to the region
    (x, y, w, h, v), = TW.decode(S4, m, 18, 30)[0]
    assert (x, y) == (0, 0) and x + w == 30 and y + h == 18


def test_a_box_clipped_to_nothing_is_dropped():
    m = maps_of(5, 8)
    put(m, 4, slice(0, 8), a1=-120, a2=127)     # pushed below the region's last row
    assert TW.decode(S4, m, 18, 32) == ([], False)


def test_floor_and_ceil_of_negative_sums():
    m = maps_of(5, 20)
    put(m, 0, slice(5, 7), a2=17)
    m[2][0, 5], m[2][0, 6] = 9, 10              # near floor((6 - 19) / 4) = floor(-3.25) = -4 (not -3)
    # far = ceil((6 + 34) / 4) = 10; across 14 gives pad (14 * 18 + 50) // 100 = 3 (13 would give 2)
    assert TW.decode(S4, m, 40, 80) == ([(17, 0, 14, 13, 0)], False)
    assert TW._floor_div(-13, 4) == -4 and TW._ceil_div(-13, 4) == -3 and TW._ceil_div(13, 4) == 4


def test_row_order_of_side_by_side_boxes():
    m = maps_of(20, 60)
    put(m, 4, slice(30, 40), a1=12, a2=12)      # right box, its top at row 4's centre - 6
    put(m, 4, slice(2, 12), a1=6, a2=10)        # left box, starting lower: still the same row
    put(m, 12, slice(10, 20))                   # a line further down: a row of its own
    boxes, _ = TW.decode(S4, m, 80, 240)
    assert [b[0] for b in boxes] == [6, 118, 38]
    assert boxes[0][1] > boxes[1][1]             # left one first although it starts lower


def test_a_line_overlapping_less_than_half_starts_a_new_row():
    m = maps_of(20, 60)
    put(m, 4, slice(30, 40), a1=8, a2=8)        # rows [13, 22) before the pad (2): [11, 24)
    put(m, 6, slice(2, 12), a1=8, a2=8)         # rows [21, 30) padded [19, 32): overlap 5 of 13, under half
    boxes, _ = TW.decode(S4, m, 80, 240)
    assert [b[0] for b in boxes] == [118, 6]


def test_vertical_lines_go_right_to_left_after_the_horizontal_ones():
    m = maps_of(20, 40)
    put(m, slice(2, 12), 5, o=16, a1=8, a2=8)
    put(m, slice(2, 12), 30, o=16, a1=8, a2=8)
    put(m, 15, slice(10, 20))
    boxes, _ = TW.decode(S4, m, 80, 160)
    assert [b[4] for b in boxes] == [0, 1, 1] and boxes[1][0] > boxes[2][0]


def test_the_64_candidate_and_16_line_caps():
    m = maps_of(60, 60)
    for k in range(70):                          # 70 two-cell lines, 10 per map row band
        put(m, 2 * (k // 10) * 4, slice(6 * (k % 10), 6 * (k % 10) + 2), a1=4, a2=4)
    boxes, truncated = TW.decode(S4, m, 240, 240)
    assert truncated and len(boxes) == 16
    m = maps_of(60, 60)
    for k in range(20):
        put(m, 2 * (k // 10) * 4, slice(6 * (k % 10), 6 * (k % 10) + 2), a1=4, a2=4)
    boxes, truncated = TW.decode(S4, m, 240, 240)
    assert truncated and len(boxes) == 16 and boxes[0][0] < boxes[1][0]
    m = maps_of(60, 60)
    for k in range(16):
        put(m, 2 * (k // 10) * 4, slice(6 * (k % 10), 6 * (k % 10) + 2), a1=4, a2=4)
    assert TW.decode(S4, m, 240, 240)[1] is False


def test_the_64_kept_candidates_are_the_smallest_by_vertical_y_x_h_w():
    vert = maps_of(40, 140)
    for k in range(66):                          # 66 vertical lines, and one horizontal line below them
        put(vert, slice(0, 3), 2 * k, o=16, a1=4, a2=4)
    put(vert, 30, slice(0, 10))
    boxes, truncated = TW.decode(S4, vert, 160, 560)
    assert truncated and boxes[0][4] == 0        # the horizontal line survives the 64 cut: vertical sorts last
    assert sum(b[4] for b in boxes) == 15


def test_stride_8_and_single_cells():
    m = maps_of(5, 10)
    put(m, 2, 4, a1=8, a2=8)                     # one cell at stride 8: c2 = 2*8*2 + 7 = 39, near 15, far 24
    assert TW.decode(S8, m, 40, 80) == ([(30, 13, 12, 13, 0)], False)   # along [32, 40), pad 2


# ---------------------------------------------------------------- blob and loader

def layer_ok(L, cin, cout, kind="conv", k=3, stride=1, dil=1, relu=True, head=False, rng=None):
    rng = rng or np.random.default_rng(0)
    per = 1 if kind == "dw" else cin
    return {"kind": kind, "k": k, "stride": stride, "dil": dil, "relu": relu, "cin": cin, "cout": cout, "head": head,
            "wq": rng.integers(-127, 128, (cout, per, k, k)).astype(np.int8), "bq": rng.integers(-500, 500, cout),
            "m": rng.integers(1 << 30, 1 << 31, cout), "sh": rng.integers(30, 45, cout)}


def small_blob():
    shape, q = E.random_int_layers("s4_c4", 3, n_cal=2)
    return E.pack(shape, q, hashlib.sha256(b"small").digest()), shape, q


def test_blob_round_trip():
    blob, shape, q = small_blob()
    m = TW.Model(blob)
    assert m.name == "s4_c4" and m.stride == 4 and m.min_cells == 2 and m.blob_bytes == len(blob)
    assert m.ckpt == hashlib.sha256(b"small").digest()[:8].hex()
    assert len(m.layers) == len(TM.layers(TM.SHAPES["s4_c4"])) == len(q)
    for L, ql in zip(m.layers, q):
        assert (L.kind, L.k, L.stride, L.dil, L.relu, L.cin, L.cout) == \
            (1 if ql["kind"] == "dw" else 0, ql["k"], ql["stride"], ql["dil"], int(ql["relu"]), ql["cin"], ql["cout"])
        assert (L.w == ql["wq"]).all() and (L.b == ql["bq"]).all() and (L.m == ql["m"]).all()
        assert (L.sh == ql["sh"]).all()
    assert E.pack(shape, q, hashlib.sha256(b"small").digest()) == blob      # deterministic


def recrc(b):
    b = bytearray(b)
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


def record(blob, i):
    return TW.HEADER + i * TW.RECORD


def bad_blobs():
    """(name, blob) pairs every loader must refuse."""
    blob, shape, q = small_blob()
    n = len(q)
    out = []

    def edit(name, fn, crc=True):
        b = bytearray(blob)
        fn(b)
        out.append((name, recrc(b) if crc else bytes(b)))

    edit("crc", lambda b: b.__setitem__(len(b) - 1, b[-1] ^ 1), crc=False)
    edit("magic", lambda b: b.__setitem__(0, ord("X")))
    out.append(("short", blob[:40]))
    out.append(("truncated", recrc(blob[:-4])))
    edit("version", lambda b: struct.pack_into("<I", b, 4, 2))
    edit("size", lambda b: struct.pack_into("<I", b, 8, len(blob) + 4))
    edit("stride 5", lambda b: b.__setitem__(18, 5))
    edit("outputs 3", lambda b: b.__setitem__(19, 3))
    edit("a codes 3", lambda b: b.__setitem__(20, 3))
    edit("min cells 0", lambda b: b.__setitem__(21, 0))
    edit("min cells 5", lambda b: b.__setitem__(21, 5))
    edit("1 layer", lambda b: struct.pack_into("<H", b, 16, 1))
    edit("33 layers", lambda b: struct.pack_into("<H", b, 16, 33))
    edit("non-ASCII name", lambda b: b.__setitem__(33, 0xC3))
    r0, r1, rl = record(blob, 0), record(blob, 1), record(blob, n - 1)
    edit("kind 2", lambda b: b.__setitem__(r0, 2))
    edit("k 2", lambda b: b.__setitem__(r0 + 1, 2))
    edit("stride 3", lambda b: b.__setitem__(r0 + 2, 3))
    edit("dilation 0", lambda b: b.__setitem__(r0 + 3, 0))
    edit("dilation 9", lambda b: b.__setitem__(r0 + 3, 9))
    edit("relu 2", lambda b: b.__setitem__(r0 + 4, 2))
    edit("cin 2", lambda b: struct.pack_into("<H", b, r0 + 8, 2))
    edit("cout 0", lambda b: struct.pack_into("<H", b, r0 + 10, 0))
    edit("depthwise cout != cin", lambda b: struct.pack_into("<H", b, r1 + 10, 5))       # layer 1 is depthwise
    edit("depthwise 1x1", lambda b: b.__setitem__(r1 + 1, 1))
    edit("stride 2 dilated", lambda b: b.__setitem__(r0 + 3, 2))
    edit("stride 2 1x1", lambda b: b.__setitem__(r0 + 1, 1))
    wo = struct.unpack_from("<I", blob, r0 + 12)[0]
    edit("misaligned weights", lambda b: struct.pack_into("<I", b, r0 + 12, wo + 1))
    edit("weights in the records", lambda b: struct.pack_into("<I", b, r0 + 12, TW.HEADER))
    edit("bias past the end", lambda b: struct.pack_into("<I", b, r0 + 16, len(blob) - 4))
    edit("offset overflow", lambda b: struct.pack_into("<I", b, r0 + 20, 0xFFFFFFFC))
    mo = struct.unpack_from("<I", blob, r0 + 20)[0]
    so = struct.unpack_from("<I", blob, r0 + 24)[0]
    edit("multiplier 0", lambda b: struct.pack_into("<i", b, mo, 0))
    edit("multiplier < 0", lambda b: struct.pack_into("<i", b, mo, -5))
    edit("shift 0", lambda b: b.__setitem__(so, 0))
    edit("shift 63", lambda b: b.__setitem__(so, 63))
    edit("head with relu", lambda b: b.__setitem__(rl + 4, 1))
    edit("chain stride", lambda b: b.__setitem__(18, 8))
    # a chain that does not end in the 4-map head: the header names one layer fewer (the last record ignored)
    out.append(("no head", recrc(bytes(blob[:16]) + struct.pack("<H", n - 1) + blob[18:])))
    return out


@pytest.mark.parametrize("name,blob", bad_blobs(), ids=[n for n, _ in bad_blobs()])
def test_the_twin_refuses(name, blob):
    with pytest.raises(ValueError):
        TW.Model(blob)


def test_the_good_blob_and_edge_values_load():
    blob, shape, q = small_blob()
    TW.Model(blob)
    for L in q:                                 # the extremes the loader allows
        L["sh"] = L["sh"].copy()
        L["sh"][0], L["m"][0] = 62, (1 << 31) - 1
    q[0]["sh"][-1] = 1
    TW.Model(E.pack(shape, q))


# ---------------------------------------------------------------- export

def test_quantize_multiplier():
    for M in (1.0, 0.5, 0.25, 0.75, 1e-3, 3.7e-5, 0.999999999, 1.5, 123.0, 1e-17):
        m, sh = E.quantize_multiplier(M)
        assert 1 <= sh <= 62
        if sh < 62:
            assert (1 << 30) <= m < (1 << 31)
            assert abs(m / 2.0 ** sh - M) <= 2.0 ** -31 * M * 1.0000001
    assert E.quantize_multiplier(1.0) == (1 << 30, 30)
    assert E.quantize_multiplier(0.5) == (1 << 30, 31)
    assert E.quantize_multiplier(1 - 2 ** -40) == (1 << 30, 30)       # rounds up to 2^31: renormalised
    m, sh = E.quantize_multiplier(1e-25)                              # below 2^-31 * 2^-31: shift capped at 62
    assert sh == 62 and 1 <= m < (1 << 30)
    for bad in (0.0, -1.0, float("nan"), 2.0 ** 31):
        with pytest.raises(ValueError):
            E.quantize_multiplier(bad)


def test_random_blobs_are_valid_for_every_shape():
    for name in TM.SHAPES:
        shape, q = E.random_int_layers(name, 1, n_cal=1)
        m = TW.Model(E.pack(shape, q))
        assert m.name == name and m.stride == TM.SHAPES[name]["stride"]


# ---------------------------------------------------------------- the protocol's sweep table

def test_protocol_sweep_equals_the_model_cost():
    sweep = PROTOCOL["sweep"]["shapes"]
    assert set(sweep) == set(TM.SHAPES)
    for name, row in sweep.items():
        s = TM.SHAPES[name]
        for k in ("stride", "c0", "c1", "c2", "chans", "dil", "block", "head"):
            assert row[k] == s[k], (name, k)
        assert row["macs"] == TM.cost(s, 900, 150)["macs"]
        assert row["params"] == TM.cost(s, 900, 150)["params"]
        for w, h in ((960, 544), (480, 272), (450, 80)):
            assert row[f"macs_{w}x{h}"] == TM.cost(s, w, h)["macs"]
        assert row["receptive_field"] == TM.receptive_field(s)


def test_frozen_finder_sources_are_unchanged():
    """runtime/vocr_textdet.c adds a finder mode beside the frozen GRAD finder and refinement, never inside them."""
    pinned = PROTOCOL["sources_sha256"]
    for f in ("runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.c", "runtime/vocr_lines_rows.h",
              "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h"):
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == pinned[f], f


# ---------------------------------------------------------------- the C runtime

@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("textdet") / "vocr_textdet_test"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    *(str(ROOT / "runtime" / f) for f in ("vocr_lines.c", "vocr_lines_rows.c", "vocr_lines_refine.c",
                                                          "vocr_textdet.c", "vocr_textdet_test.c")),
                    "-o", str(out)], check=True)
    return out


def test_c_loader_refuses_what_the_twin_refuses(driver, tmp_path):
    blob, _, _ = small_blob()
    cases = [("good", blob)] + bad_blobs()
    paths = []
    for i, (_, b) in enumerate(cases):
        p = tmp_path / f"b{i}.vtd"
        p.write_bytes(b)
        paths.append(str(p))
    rcs = [int(v) for v in subprocess.run([str(driver), "--load", *paths], check=True, capture_output=True,
                                          text=True).stdout.split()]
    assert rcs[0] == 0
    assert all(rc == -5 for rc in rcs[1:]), [(n, rc) for (n, _), rc in zip(cases[1:], rcs[1:]) if rc != -5]


def test_c_selftest(driver, tmp_path):
    a, b = tmp_path / "a.vtd", tmp_path / "b.vtd"
    a.write_bytes(small_blob()[0])
    b.write_bytes(P.probe_blob(8, 1))
    out = subprocess.run([str(driver), "--selftest", "4", str(a), str(b)], check=True, capture_output=True,
                         text=True, timeout=600).stdout
    assert out.startswith("TEXTDET_SELFTEST_OK 2 blobs")


def test_c_matches_the_twin(driver):
    assert P.main(["--bin", str(driver), "--seeds", "1", "--regions", "4", "--probe-regions", "6",
                   "--shapes", "s4_c4,s4p_c4,s8_c8,s8_c16_ko", "--screen-every", "6", "--wild", "1",
                   "--seed", "3"]) == 0


def test_c_arena_fits_the_always_safe_tier_and_follows_the_width(driver, tmp_path):
    p = tmp_path / "x.vtd"
    for name in TM.SHAPES:
        shape, q = E.random_int_layers(name, 1, n_cal=1)
        p.write_bytes(E.pack(shape, q))
        run = lambda w, h: int(subprocess.run([str(driver), "--arena", str(p), str(w), str(h)],  # noqa: E731
                                              check=True, capture_output=True, text=True).stdout)
        a = run(960, 544)
        assert a <= 131072, (name, a)
        assert run(960, 1) == run(960, 2048) == a and run(480, 272) < a


def test_c_batch_contract(driver, tmp_path):
    """eval/textdet.py's contract: --batch BLOB FILE, records <4I (w, h, bpp, flags 0/1) + pixels; one line each:
    rc count truncated, then x y w h dark vertical per line."""
    blob = P.probe_blob(4, 0)
    (tmp_path / "p.vtd").write_bytes(blob)
    rng = np.random.default_rng(1)
    g = P.designed_region(rng, 4)
    P.write_batch(tmp_path / "r.bin", [(g, 0), (g, 1), (np.zeros((3, 3), np.uint8), 0)])
    out = subprocess.run([str(driver), "--batch", str(tmp_path / "p.vtd"), str(tmp_path / "r.bin")], check=True,
                         capture_output=True, text=True).stdout.splitlines()
    want = P.twin_line(TW.Model(blob), g)
    assert out == [want, want, "0 0 0"]
    v = [int(t) for t in want.split()]
    assert len(v) == 3 + 6 * v[1]


def test_c_det_refine_contract(driver, tmp_path):
    """Record flag 2 (det+refine): the detector's horizontal lines through VOCR_REFINE_DEFAULT, in-memory and rows
    forms, equal synth/lines_refine.refine_lines on the twin's horizontal lines; --arena-refine prints
    max(vocr_textdet_arena_bytes, vocr_refine_arena_bytes)."""
    blob = P.probe_blob(4, 0)
    p = tmp_path / "p.vtd"
    p.write_bytes(blob)
    rng = np.random.default_rng(5)
    imgs = [P.designed_region(rng, 4) for _ in range(6)]
    P.write_batch(tmp_path / "r.bin", [(g, P.REFINE | form) for g in imgs for form in (0, P.ROWS)])
    out = subprocess.run([str(driver), "--batch", str(p), str(tmp_path / "r.bin")], check=True,
                         capture_output=True, text=True).stdout.splitlines()
    want = [P.refine_line(TW.Model(blob), g) for g in imgs for _ in (0, 1)]
    assert out == want
    for w, h in ((960, 544), (480, 272), (8, 8)):
        det, ref = (int(subprocess.run([str(driver), mode, str(p), str(w), str(h)], check=True, capture_output=True,
                                       text=True).stdout) for mode in ("--arena", "--arena-refine"))
        assert ref >= det

"""Tests of the text-presence gate: its twin's helpers (eval/textgate.py: kth, box_mask, score, threshold_at,
rubi_ink_edges, rule8_score), the frozen sources it must not touch, the committed code sizes, and the C runtime
(runtime/vocr_textgate.c) against the twin: the loader must refuse exactly the blobs synth/textdet_twin.Model refuses,
the driver's self-test must pass and a small tests/textgate_parity.py run must give the twin's scores and K maps.
The C parts compile the host driver with the system compiler (cc, or $CC) and are skipped without one. No
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

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textgate as TG  # noqa: E402

import test_textdet as TD  # noqa: E402
import textdet_parity as TP  # noqa: E402
import textdet_twin as TW  # noqa: E402
import textgate_parity as P  # noqa: E402
import textgate_train as G  # noqa: E402

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
PROTOCOL = json.loads((ROOT / "eval" / "textgate_protocol.json").read_text())


# ---------------------------------------------------------------- kth, box_mask, score

def test_kth_largest_ties_and_fewer_cells():
    v = np.array([5, 1, 9, 9, 3])
    assert [TG.kth(v, k) for k in (1, 2, 3, 4, 5)] == [9, 9, 5, 3, 1]
    assert TG.kth(v, 8) == 1                         # fewer cells than k: the smallest
    assert TG.kth(np.array([[4, 4], [4, 4]]), 3) == 4    # all equal, 2-D
    assert TG.kth(np.array([], np.int64), 1) == TG.NO_CELLS == -129
    assert TG.kth(np.array([-128]), 4) == -128


def test_box_mask_edges():
    shape = (3, 5)                                   # a map of 3 x 5 cells at stride 4: blocks up to 12 x 20 px
    one = TG.box_mask(shape, 4, [(0, 0, 1, 1)])
    assert one.sum() == 1 and one[0, 0]
    m = TG.box_mask(shape, 4, [(3, 3, 2, 2)])        # pixels 3..4 cross the block border: 2 x 2 cells
    assert m[:2, :2].all() and m.sum() == 4
    assert not TG.box_mask(shape, 4, [(5, 5, 0, 3), (5, 5, 3, 0)]).any()      # empty boxes meet nothing
    assert not TG.box_mask(shape, 4, [(100, 0, 5, 5), (0, 40, 5, 5)]).any()   # beyond the map
    edge = TG.box_mask(shape, 4, [(18, 10, 100, 100)])   # past the region's corner: the last block still meets
    assert edge.sum() == 1 and edge[2, 4]
    union = TG.box_mask(shape, 4, [(0, 0, 8, 4), (4, 0, 8, 4)])                # overlapping: each cell once
    assert union.sum() == 3 and union[0, :3].all()
    assert TG.box_mask(shape, 8, [(7, 7, 1, 1), (8, 8, 1, 1)]).sum() == 2


def test_score_both_modes():
    km = np.array([[10, -5, 3], [7, 7, -128]])
    assert TG.score(km, "region", 1, 4) == 10 and TG.score(km, "region", 2, 4) == 7
    assert TG.score(km, "region", 6, 4) == -128 and TG.score(km, "region", 8, 4) == -128
    assert TG.score(km, "boxes", 1, 4, [(4, 0, 8, 8)]) == 7            # cells (0,1), (0,2), (1,1), (1,2)
    assert TG.score(km, "boxes", 3, 4, [(4, 0, 8, 8)]) == -5
    assert TG.score(km, "boxes", 8, 4, [(4, 0, 8, 8)]) == -128          # fewer cells than k
    assert TG.score(km, "boxes", 1, 4, []) == TG.score(km, "boxes", 1, 4, None) == TG.NO_CELLS
    assert TG.score(km, "boxes", 1, 4, [(40, 40, 4, 4)]) == TG.NO_CELLS


def test_threshold_at():
    s = np.arange(1, 1001)
    t = TG.threshold_at(s, 0.998)
    assert t == 3 and (s >= t).mean() >= 0.998 and (s >= t + 1).mean() < 0.998
    assert TG.threshold_at(s, 0.999) == 2
    ties = np.array([5] * 10 + [9] * 990)
    assert TG.threshold_at(ties, 0.998) == 5                    # the third smallest is in the tie
    assert TG.threshold_at(np.array([-129, 4, 4, 6]), 0.75) == 4
    assert TG.threshold_at(np.array([7]), 0.999) == 7


# ---------------------------------------------------------------- the baselines

def test_rubi_ink_edges_by_hand():
    crop = np.full((3, 4), 10, np.uint8)
    crop[0, 2] = 200
    # n 12: lo = 10, hi = 200 (want 12 of 12), edge max(24, 190 // 4) = 47; across: (0,0) at 2 steps, (0,1), (0,2);
    # down: (0,2) only (steps look right and down); 4 edges: 4 * 10,000 / 12 = 3,333.3
    assert TG.rubi_ink_edges(crop) == 3333


def test_rubi_ink_edges_rubi_examples():
    """Rubi_for_vita host/tests/test_vocr_policy.c test_ink_and_blank (core/vocr_ocr.c vjo_vocr_ink), the cases the
    twin models (Rubi's guards for crops over its pixel and width limits are not part of the twin)."""
    blank = TG.BLANK_EDGES
    px = np.full((40, 300), 90, np.uint8)
    assert TG.rubi_ink_edges(px) == 0                                         # an even area
    ramp = (20 + np.arange(300) * 200 // 300).astype(np.uint8)[None].repeat(40, 0)
    assert TG.rubi_ink_edges(ramp) == 0                                       # a smooth ramp
    strokes = np.full((40, 300), 20, np.uint8)
    for x in range(10, 290, 12):
        strokes[8:32, x:x + 2] = 235
    assert TG.rubi_ink_edges(strokes) >= 400 and TG.rubi_ink_edges(255 - strokes) >= blank
    low = np.full((40, 300), 100, np.uint8)
    for x in range(10, 290, 12):
        low[8:32, x] = 120
    assert TG.rubi_ink_edges(low) == 0                                        # 20 levels: under the floor
    col = np.full((10, 97), 20, np.uint8)
    col[:, 50] = 235
    assert TG.rubi_ink_edges(col) == 309                                      # 30 * 10,000 / 970
    dot = np.full((10, 97), 20, np.uint8)
    dot[5, 50] = 235
    assert TG.rubi_ink_edges(dot) == 61                                       # 6 * 10,000 / 970
    mark = np.full((40, 40), 20, np.uint8)
    mark[16:24, 16:24] = 20
    mark[16, 16:24] = mark[23, 16:24] = mark[16:24, 16] = mark[16:24, 23] = 230
    assert TG.rubi_ink_edges(mark) >= blank                                   # one small mark: not blank


def test_rule8_score():
    g = np.full((16, 64), 30, np.uint8)
    assert TG.rule8_score(g) == 0
    stripes = np.where(np.arange(8) % 2, 255, 0).astype(np.uint8)
    for bx in (1, 2, 3, 6):                      # blocks 1-3 in a run, block 6 alone
        g[0:8, 8 * bx:8 * bx + 8] = stripes
    assert TG.rule8_score(g) == 3
    soft = np.full((8, 32), 100, np.uint8)
    soft[:, ::2] = 140                           # 40 levels: edges (>= 24) but under the contrast of 48
    assert TG.rule8_score(soft) == 0
    sparse = np.full((8, 8), 0, np.uint8)
    sparse[0, 1] = 255                           # 2 edge pixels (dx at x 1 and 2... only those): under 16
    assert TG.rule8_score(sparse) == 0
    odd = np.full((5, 13), 0, np.uint8)          # partial blocks: padded with 0 (edges) and 255 (min)
    odd[:, ::2] = 200
    assert TG.rule8_score(odd) in (1, 2)


# ---------------------------------------------------------------- frozen sources and committed sizes

def test_frozen_sources_are_unchanged():
    """The gate is a new file beside the finder, its refinement and #97's detector, never inside them."""
    pinned = PROTOCOL["sources_sha256"]
    for f in ("runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.c", "runtime/vocr_lines_rows.h",
              "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h", "runtime/vocr_textdet.c",
              "runtime/vocr_textdet.h", "eval/textgate.py", "synth/textdet_twin.py"):
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == pinned[f], f


def test_committed_sizes_are_of_this_code():
    sizes = sorted((ROOT / "eval").glob("textgate_sizes_*.json"))
    assert sizes, "eval/textgate_sizes_<date>.json (vita/textgate_size.py --out)"
    doc = json.loads(sizes[-1].read_text())
    for f in ("vocr_textgate.c", "vocr_textgate.h"):
        assert doc["sources_sha256"][f] == hashlib.sha256((ROOT / "runtime" / f).read_bytes()).hexdigest(), \
            f"{sizes[-1].name} was measured on another {f}: re-run vita/textgate_size.py"
    cb = doc["code_bytes"]
    assert cb["region"] <= cb["boxes"] and cb["blank"] == 0
    assert all(doc["vita"]["levels"][lv][m]["rodata"] == 0 for lv in ("-Os", "-O2") for m in ("region", "boxes"))


# ---------------------------------------------------------------- random blobs of the presence shapes

def test_random_blobs_of_every_presence_shape_load():
    for name, shape in G.SHAPES.items():
        m = TW.Model(P.random_blob(shape, 1, n_cal=1))
        assert m.name == name and m.stride == shape["stride"] and m.layers[-1].cout == 4
    m = TW.Model(P.chain_blob(3))
    assert m.layers[0].cin == 1 and m.layers[-1].cout == 4


# ---------------------------------------------------------------- the C runtime

@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("textgate") / "vocr_textgate_test"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    *(str(ROOT / "runtime" / f) for f in ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textgate.c",
                                                          "vocr_textgate_test.c")),
                    "-o", str(out)], check=True)
    return out


def gate_bad_blobs():
    """#97's loader corruptions (tests/test_textdet.bad_blobs) plus the name's edge cases, on a presence-shape blob
    too: (name, blob, refused)."""
    out = [(n, b, True) for n, b in TD.bad_blobs()]
    good = P.random_blob(G.SHAPES["p8_c4"], 2, n_cal=1)
    out.append(("p8_c4", good, False))

    def edit(name, fn, refused):
        b = bytearray(good)
        fn(b)
        struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
        out.append((name, bytes(b), refused))
    edit("name: last byte non-ASCII", lambda b: b.__setitem__(63, 0x80), True)
    edit("name: a NUL inside", lambda b: b.__setitem__(33, 0), False)
    edit("name: all NUL", lambda b: b.__setitem__(slice(32, 64), bytes(32)), False)
    edit("reserved u16 and checkpoint bytes", lambda b: b.__setitem__(slice(22, 32), b"\x7f" * 10), False)
    edit("min cells 4", lambda b: b.__setitem__(21, 4), False)
    edit("head relu", lambda b: b.__setitem__(TW.HEADER + (len(TW.Model(good).layers) - 1) * TW.RECORD + 4, 1), True)
    return out


def test_c_loader_refuses_what_the_twin_refuses(driver, tmp_path):
    cases = gate_bad_blobs()
    paths = []
    for i, (name, b, refused) in enumerate(cases):
        try:
            TW.Model(b)
            twin = False
        except ValueError:
            twin = True
        assert twin == refused, name
        p = tmp_path / f"b{i}.vtd"
        p.write_bytes(b)
        paths.append(str(p))
    rcs = [int(v) for v in subprocess.run([str(driver), "--load", *paths], check=True, capture_output=True,
                                          text=True).stdout.split()]
    want = [-5 if refused else 0 for _, _, refused in cases]
    assert rcs == want, [(n, rc) for (n, _, _), rc, w in zip(cases, rcs, want) if rc != w]


def test_c_selftest(driver, tmp_path):
    a, b, c = tmp_path / "a.vtd", tmp_path / "b.vtd", tmp_path / "c.vtd"
    a.write_bytes(P.random_blob(G.SHAPES["p8_c4"], 4, n_cal=1))
    b.write_bytes(TP.probe_blob(4, 0))
    c.write_bytes(P.chain_blob(11))
    out = subprocess.run([str(driver), "--selftest", "8", str(a), str(b), str(c)], check=True, capture_output=True,
                         text=True, timeout=600).stdout
    assert out.startswith("TEXTGATE_SELFTEST_OK 3 blobs")


def test_c_matches_the_twin(driver):
    assert P.main(["--bin", str(driver), "--seeds", "1", "--regions", "4", "--shapes", "p8_c4,p8_c8,p4_c4,s8_c8",
                   "--wild", "1", "--chains", "2", "--probes", "1", "--screen-every", "5", "--seed", "3"]) == 0


def test_c_arena_follows_the_width_in_region_mode(driver, tmp_path):
    p = tmp_path / "x.vtd"
    p.write_bytes(P.random_blob(G.SHAPES["p8_c8"], 1, n_cal=1))

    def arena(mode, w, h):
        return int(subprocess.run([str(driver), "--arena", str(p), str(mode), str(w), str(h)], check=True,
                                  capture_output=True, text=True).stdout)
    a = arena(0, 960, 544)
    assert arena(0, 960, 1) == arena(0, 960, 2048) == a and arena(0, 480, 272) < a
    assert arena(1, 960, 544) - a == 120 * 68 - 120          # the K map instead of the K row
    assert arena(0, 0, 5) == 0 and arena(2, 5, 5) == 0 and arena(0, 2049, 5) == 0


def test_c_batch_contract(driver, tmp_path):
    """eval/textgate_c.py's contract: --batch BLOB FILE, records <4I (w, h, bpp, flags) + pixels (+ boxes in boxes
    mode: <I n, n x <4I x y w h); flags bits 0-1 the form, 4 boxes mode, bits 4-7 k; one line each: rc score."""
    blob = TP.probe_blob(8, 0)
    (tmp_path / "p.vtd").write_bytes(blob)
    rng = np.random.default_rng(2)
    g = TP.designed_region(rng, 8)
    km = TG.kmap(TW.Model(blob), g)
    boxes = [(0, 0, 9, 9), (20, 4, 30, 2)]
    recs = [P.record(g, form | (k << 4)) for form in (0, P.ROWS, P.PUSH) for k in (1, 2)]
    recs += [P.record(g, P.BOXES | form | (2 << 4), boxes) for form in (0, P.ROWS, P.PUSH)]
    recs.append(P.record(np.zeros((1, 1), np.uint8), P.BOXES | (1 << 4), []))
    (tmp_path / "r.bin").write_bytes(b"".join(recs))
    out = subprocess.run([str(driver), "--batch", str(tmp_path / "p.vtd"), str(tmp_path / "r.bin")], check=True,
                         capture_output=True, text=True).stdout.splitlines()
    want = [f"0 {TG.score(km, 'region', k, 8)}" for _ in range(3) for k in (1, 2)]
    want += [f"0 {TG.score(km, 'boxes', 2, 8, boxes)}"] * 3 + ["0 -129"]
    assert out == want

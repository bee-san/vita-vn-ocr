"""Tests of runtime/vocr_textspot6.c, the spotter of VTSP version 1, 2, 3 and 4 (tagged) blobs, against its twin
(synth/textspot6_twin.py) and, for versions 1-3, against textspot3's frozen runtime (runtime/vocr_textspot3.c): the C
loader refuses exactly the blobs the twin refuses (tests/textspot6_parity.bad_v4, textspot3's and textspot2's refusals,
field mutants with both CRCs fixed), the VOCR_TEXTSPOT6_NO_MS build refuses every multi-scale network and gives the
full build's results otherwise, --info, the driver's self-test, a small tests/textspot6_parity.py run, the twin tests'
hand-made maps through a probe network (C gives what tests/test_textspot6_twin.py expects of them), the --batch line
format, the arenas (versions 1-3 equal vocr_textspot3.c's; version 4: the up-to arena is one frame's need) and the
pinned sources. The C parts compile the host drivers with the system compiler (cc, or $CC) and are skipped without
one. No screenshots, fonts or models; no network.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_parity as TP  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_twin as T2  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_parity as Q3  # noqa: E402
import textspot3_twin as T3  # noqa: E402
import textspot6_model as M6  # noqa: E402
import textspot6_parity as Q6  # noqa: E402
import textspot6_twin as T6  # noqa: E402
import textspot_parity as P  # noqa: E402
import textspot_twin as S  # noqa: E402
from test_textspot2_twin import bad_blobs as bad_v2_blobs  # noqa: E402
from test_textspot2_twin import good_blobs as good_v2_blobs  # noqa: E402
from test_textspot3_twin import BAD as BAD_V3  # noqa: E402
from test_textspot3_twin import good_blobs as good_v3_blobs  # noqa: E402

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
LIB = ("vocr_lines.c", "vocr_lines_rows.c")
SOURCES = {"vocr_textspot6_test": (*LIB, "vocr_textspot6.c", "vocr_textspot6_test.c"),
           "vocr_textspot6_test_noms": (*LIB, "vocr_textspot6.c", "vocr_textspot6_test.c"),
           "vocr_textspot3_test": (*LIB, "vocr_textspot3.c", "vocr_textspot3_test.c")}
PROTOCOLS = [json.loads((ROOT / "eval" / p).read_text(encoding="utf-8"))
             for p in ("textspot_protocol.json", "textspot2_protocol.json", "textspot3_protocol.json")]
TINY_T, TINY_M = Q6.TINY_T, Q6.TINY_M
TINY3 = M3.m_shape("tiny", 4, 8, 8, [8], [1], [(16, [1]), (16, [2])], 8, 8, 1)


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("textspot6")
    out = {}
    for name, srcs in SOURCES.items():
        out[name] = d / name
        extra = ["-DVOCR_TEXTSPOT6_NO_MS"] if name.endswith("_noms") else []
        subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", *extra,
                        *(str(ROOT / "runtime" / f) for f in srcs), "-o", str(out[name])], check=True)
    return out


def run(binary, *args):
    return subprocess.run([str(binary), *map(str, args)], check=True, capture_output=True, text=True,
                          timeout=600).stdout


def load_codes(binary, tmp_path, blobs):
    paths = []
    for i, b in enumerate(blobs):
        p = tmp_path / f"b{i}.vtsp"
        p.write_bytes(b)
        paths.append(p)
    return [int(v) for v in run(binary, "--load", *paths).split()]


def test_c_loader_refuses_what_the_twin_refuses(drivers, tmp_path):
    good = Q6.good_v4() + good_v3_blobs() + good_v2_blobs()
    bad = Q6.bad_v4() + list(BAD_V3) + bad_v2_blobs()
    for name, b in bad:
        with pytest.raises(ValueError):
            T6.Model(b)
    for name, b in good:
        T6.Model(b)
    rcs = load_codes(drivers["vocr_textspot6_test"], tmp_path, [b for _, b in good + bad])
    assert rcs[:len(good)] == [0] * len(good)
    wrong = [(n, rc) for (n, _), rc in zip(bad, rcs[len(good):]) if rc != Q6.E_BLOB]
    assert not wrong, wrong
    # textspot3's frozen loader refuses every version-4 blob, good or bad
    old = load_codes(drivers["vocr_textspot3_test"], tmp_path, [b for _, b in Q6.good_v4()])
    assert old and all(rc == Q6.E_BLOB for rc in old)


def test_c_loader_agrees_with_the_twin_on_field_mutants(drivers, tmp_path):
    rng = np.random.default_rng(29)
    blobs, want = [], []
    for name, good in Q6.good_v4()[:6]:
        for _, b in Q6.mutants(good, rng, 40):
            blobs.append(b)
            want.append(Q6.twin_code(b))
    assert load_codes(drivers["vocr_textspot6_test"], tmp_path, blobs) == want
    assert 0 < want.count(0) < len(want)


def test_no_ms_build(drivers, tmp_path):
    """-DVOCR_TEXTSPOT6_NO_MS refuses version 3 and version 4's multi-scale form, loads everything else and gives the
    full build's lines for it."""
    ms = [b for n, b in Q6.good_v4() if T6.Model(b).form == "ms"] + [b for _, b in good_v3_blobs()]
    trunk = [b for n, b in Q6.good_v4() if T6.Model(b).form == "trunk"] + [b for _, b in good_v2_blobs()]
    assert load_codes(drivers["vocr_textspot6_test_noms"], tmp_path, ms) == [Q6.E_BLOB] * len(ms)
    assert load_codes(drivers["vocr_textspot6_test_noms"], tmp_path, trunk) == [0] * len(trunk)
    rng = np.random.default_rng(3)
    P.write_batch(tmp_path / "r.bin", [(P.tiled(rng, 960, 544), 0), (P.tiled(rng, 480, 272), 1),
                                       (P.tiled(rng, 37, 61), 2)])
    for i, b in enumerate(trunk[:4]):
        (tmp_path / f"t{i}.vtsp").write_bytes(b)
        assert run(drivers["vocr_textspot6_test_noms"], "--batch", tmp_path / f"t{i}.vtsp", tmp_path / "r.bin") == \
            run(drivers["vocr_textspot6_test"], "--batch", tmp_path / f"t{i}.vtsp", tmp_path / "r.bin")


def test_c_info(drivers, tmp_path):
    t = M6.random_blob(TINY_T, 5, 3, conf_min=3, n_cal=2)[0]
    m = M6.random_blob(TINY_M, 5, 4, n_cal=2)[0]
    v3 = T3.pack(M3.random_net(TINY3, 5, n_cal=2), 480, 3, 272)
    v2 = T2.pack(X.random_vtdt("s8_c24n4x2", 5, n_cal=2), 600, 3)
    for name, b in (("a", t), ("b", m), ("c", v3), ("d", v2), ("e", t[:-1])):
        (tmp_path / f"{name}.vtsp").write_bytes(b)
    lines = run(drivers["vocr_textspot6_test"], "--info", *(tmp_path / f"{n}.vtsp" for n in "abcde")).splitlines()
    mt, mm = T6.Model(t), T6.Model(m)
    assert lines[0].split() == [str(v) for v in (0, 4, 28, 8, 4, mt.min_cells, 480, 3, len(mt.net.layers), len(t),
                                                 len(t) - 32, 0, 0, 0, 3, 1)]
    head = lines[1].split()
    assert head[:9] == [str(v) for v in (0, 4, 32, 8, 4, mm.min_cells, 480, 1, len(mm.net.stream))]
    assert head[11:13] == ["272", str(len(mm.net.ops))] and head[-2:] == ["4", "2"]
    assert lines[2].split()[1] == "3" and lines[2].split()[-2:] == ["0", "0"]
    assert lines[3].split()[1] == "2" and lines[3].split()[-2:] == ["0", "0"]
    assert lines[4] == "-5"


def test_c_selftest(drivers, tmp_path):
    paths = []
    blobs = (M6.random_blob(TINY_T, 8, 1, n_cal=2)[0], M6.random_blob(TINY_T, 9, 4, n_cal=2)[0],
             M6.random_blob(TINY_M, 8, 2, n_cal=2)[0], T6.pack(Q6.probe6_net("ms", 4, 2), 4, 0, 1, 300),
             T6.pack(Q6.probe6_net("trunk", 2), 2, 600, 1, 0), T3.pack(Q3.probe_net(2, 3), 0, 4, 0),
             T2.pack(X.random_vtdt("s8_c24n4x2", 8, n_cal=2), 600, 1), S.pack(TP.random_blob("s4_c4", 9, 2), 600, 1))
    for i, b in enumerate(blobs):
        paths.append(tmp_path / f"s{i}.vtsp")
        paths[-1].write_bytes(b)
    out = run(drivers["vocr_textspot6_test"], "--selftest", 3, *paths)
    assert out.startswith("TEXTSPOT6_SELFTEST_OK 8 blobs (1 of version 2, 1 of version 3, 5 of version 4)")
    trunk = [p for p, b in zip(paths, blobs) if not Q6.is_ms_blob(b)]
    out = run(drivers["vocr_textspot6_test_noms"], "--selftest", 2, *trunk)
    assert out.startswith(f"TEXTSPOT6_SELFTEST_OK {len(trunk)} blobs")


def test_c_matches_the_twin(drivers):
    assert Q6.main(["--bin", str(drivers["vocr_textspot6_test"]), "--v3-bin", str(drivers["vocr_textspot3_test"]),
                    "--noms-bin", str(drivers["vocr_textspot6_test_noms"]), "--t-shapes", "tinyt", "--m-shapes",
                    "tinym", "--t-tags", "1,2,3,4", "--m-tags", "2,4", "--frames", "1", "--probe-frames", "1",
                    "--probe-variants", "0,1,2", "--probe-mc", "1", "--probe-cells", "30x60", "--full-every", "6",
                    "--maps-every", "2", "--graphs", "1", "--mutants", "6", "--arena-every", "5", "--arena-frames", "1",
                    "--v3-probe-shapes", "tiny3", "--v3-graphs", "1", "--v3-probes", "1", "--v12-probes", "1",
                    "--v1-shapes", "s4_c4", "--d2s-shapes", "s8_c24n4x2", "--seed", "4"]) == 0


def parse_line(line):
    """A --batch line: (head [rc, count, block_count, pick, truncated, presence], boxes [10 fields], blocks [8])."""
    v = [int(t) for t in line.split()]
    n, nb = v[1], v[2]
    assert len(v) == 6 + 10 * n + 8 * nb
    return v[:6], [v[6 + 10 * i:16 + 10 * i] for i in range(n)], [v[6 + 10 * n + 8 * i:14 + 10 * n + 8 * i]
                                                                 for i in range(nb)]


@pytest.mark.parametrize("form", ["trunk", "ms"])
def test_c_on_the_twin_tests_hand_made_maps(drivers, tmp_path, form):
    """tests/test_textspot6_twin.py's hand-made maps, carried by a probe network: C gives what that test expects."""
    rng = np.random.default_rng(11)
    for name, tags, mp, check in Q6.twin_cases():
        if form == "ms" and tags % 2:
            continue
        blob = T6.pack(Q6.probe6_net(form, tags), tags, 0, 1, 0)
        (tmp_path / "p.vtsp").write_bytes(blob)
        g = Q6.encode(mp, rng)
        P.write_batch(tmp_path / "r.bin", [(g, 0), (g, 2)])
        out = run(drivers["vocr_textspot6_test"], "--batch", tmp_path / "p.vtsp", tmp_path / "r.bin").splitlines()
        r = T6.spot(T6.Model(blob), g)
        assert out == [Q6.twin_line6(r), Q6.twin_line6(r, dark=False)] and not check(r), name
        head, boxes, blocks = parse_line(out[0])
        if name == "scores_ties_tconf":
            by_y = sorted(boxes, key=lambda b: b[1])
            assert [b[8] for b in by_y] == [0, 1, 1, 0, 4] and [b[9] for b in by_y] == [16, 32, 0, 0, 16]
        elif name == "blocks_by_tag_pick":
            assert sorted(b[8] for b in boxes) == [0, 0, 1, 3, 3, 3] and head[5] == 64
            assert blocks[head[3]][7] == 0 and blocks[head[3]][6] == 2
            for blk in blocks:
                assert {b[8] for b in boxes[blk[5]:blk[5] + blk[6]]} == {blk[7]}
        elif name == "no_dialogue":
            assert head[1] == 1 and boxes[0][8] == 1 and head[3] == -1 and head[5] == 0
        else:
            assert head[1] == 64 and head[4] == 1 and min(b[5] for b in boxes) == 7


def test_c_batch_contract(drivers, tmp_path):
    """--batch: one line per record, rc count block_count pick truncated presence, then 10 integers per box and 8 per
    block; flag 2 (VOCR_TEXTSPOT_NO_DARK) the same with every dark 0; --dry spells the empty set; --batch3 prints
    vocr_textspot3_test's line."""
    blob = M6.random_blob(TINY_T, 3, 4, n_cal=2)[0]
    (tmp_path / "b.vtsp").write_bytes(blob)
    g = P.tiled(np.random.default_rng(2), 960, 544)
    P.write_batch(tmp_path / "r.bin", [(g, 0), (g, 1), (g, 2), (np.zeros((3, 3), np.uint8), 0)])
    out = run(drivers["vocr_textspot6_test"], "--batch", tmp_path / "b.vtsp", tmp_path / "r.bin").splitlines()
    r = T6.spot(T6.Model(blob), g)
    assert out == [Q6.twin_line6(r), Q6.twin_line6(r), Q6.twin_line6(r, dark=False), "0 0 0 -1 0 0"]
    head, boxes, blocks = parse_line(out[0])
    assert head[1] > 1 and all(b[6] == 0 for b in parse_line(out[2])[1])
    assert run(drivers["vocr_textspot6_test"], "--dry", tmp_path / "b.vtsp", tmp_path / "r.bin").splitlines() == \
        ["0 0 0 -1 0 0"] * 4
    v2 = T2.pack(X.random_vtdt("s8_c24n4x2", 3, n_cal=2), 600, 1)
    (tmp_path / "v2.vtsp").write_bytes(v2)
    assert run(drivers["vocr_textspot6_test"], "--batch3", tmp_path / "v2.vtsp", tmp_path / "r.bin") == \
        run(drivers["vocr_textspot3_test"], "--batch", tmp_path / "v2.vtsp", tmp_path / "r.bin")


def arena(binary, blob_path, w, h, kind="--frame-arena"):
    return int(run(binary, kind, blob_path, w, h))


def test_c_arena_of_versions_1_to_3_is_textspot3s(drivers, tmp_path):
    p = tmp_path / "x.vtsp"
    for blob in (S.pack(TP.random_blob("s4_c24n4", 1, 2), 600, 1), T2.pack(X.random_vtdt("s8_c48n4x2", 1, n_cal=2),
                                                                            480, 1),
                 T3.pack(M3.random_net(M3.PROBES["pS1"], 1, n_cal=1), 480, 1, 272),
                 T3.pack(Q3.graph_net(31), 0, 1, 0)):
        p.write_bytes(blob)
        for w, h in ((960, 544), (480, 272), (600, 544), (601, 3), (1, 1), (2048, 9), (725, 300)):
            for kind in ("--arena", "--frame-arena"):
                assert arena(drivers["vocr_textspot6_test"], p, w, h, kind) == \
                    arena(drivers["vocr_textspot3_test"], p, w, h, kind), (w, h, kind)


def test_c_arena_of_version_4(drivers, tmp_path):
    """Both forms: the up-to arena is one frame's need (the largest frame read at full size, or the frame itself when
    halved), covers smaller frames and grows with each side; a trunk's tag maps cost (4 + T) wm bytes of map row and
    T int32 per component slot, its head 4 T more planes, nothing else."""
    p = tmp_path / "m.vtsp"
    d = drivers["vocr_textspot6_test"]
    for blob in (M6.random_blob(TINY_T, 1, 4, n_cal=1)[0], M6.random_blob(TINY_M, 1, 2, n_cal=1)[0],
                 Q6.repack(M6.random_blob(TINY_T, 1, 2, n_cal=1)[0], 480, 1, 272)):
        p.write_bytes(blob)
        m = T6.Model(blob)
        up = arena(d, p, 960, 544, "--arena")
        w1 = m.half_above if m.half_above and 960 > m.half_above else 960
        h1 = m.half_above_h if m.half_above_h and 544 > m.half_above_h else 544
        assert up == max(arena(d, p, w1, h1), arena(d, p, 960, 544))
        assert up >= arena(d, p, 725, 544) and up >= arena(d, p, 480, 300)
    # a trunk halved by its height alone (half_above 0, half_above_h 272): the frame of 300 x 273 reads 150 x 137
    blob = Q6.repack(M6.random_blob(TINY_T, 2, 4, n_cal=1)[0], 0, 1, 272)
    p.write_bytes(blob)
    assert arena(d, p, 300, 273) < arena(d, p, 300, 272)
    assert arena(d, p, 300, 544, "--arena") == max(arena(d, p, 300, 272), arena(d, p, 300, 544))
    # the tag maps' bytes: one trunk as version 2 (its head's first 16 channels) and as version 4 (16 + 4 T). A frame
    # read at full size needs 4 T more head planes, T more map-row planes, T int32 per component slot, 4 T more
    # requantisation records and (when the head is the widest layer) a wider block scratch, nothing else
    for tags in (1, 4):
        q = M6.random_int_layers(TINY_T, 1, tags, n_cal=1)
        q16 = [dict(L, cout=16, wq=L["wq"][:16], bq=L["bq"][:16], m=L["m"][:16], sh=L["sh"][:16]) if L["head"] else L
               for L in q]
        b2 = T2.pack(M6.pack_trunk(TINY_T, q16, 0), 480, 1)
        b4 = T6.pack(M6.pack_trunk(TINY_T, q, tags), tags, 480, 1, 0)
        (tmp_path / "v2.vtsp").write_bytes(b2)
        (tmp_path / "v4.vtsp").write_bytes(b4)
        out, wn, wm = 16 + 4 * tags, 60, 120                  # a 480 x 272 frame, read as it is

        def up8(n):
            return (n + 7) & ~7
        chans = sum(L["cout"] for L in q16)
        widest = max(L["cout"] for L in q16)
        diff = (up8(16 * (chans + 4 * tags)) - up8(16 * chans) + up8(24 * max(widest, out)) - up8(24 * widest) +
                up8(out * wn) - up8(16 * wn) + up8((4 + tags) * wm) - up8(4 * wm) + up8(4 * tags * wm))
        assert arena(d, tmp_path / "v4.vtsp", 480, 272) - arena(d, tmp_path / "v2.vtsp", 480, 272) == diff, tags


def test_new_runtime_files_and_frozen_sources():
    """The version-4 runtime is files of its own; every source a protocol pins is byte for byte as pinned; the earlier
    runtimes, their drivers and twins are unchanged; the Makefile runs check-textspot6 in check."""
    for f in ("vocr_textspot6.c", "vocr_textspot6.h", "vocr_textspot6_test.c", "vocr_textspot6.mk"):
        assert (ROOT / "runtime" / f).exists()
    pins = {}
    for p in PROTOCOLS:
        pins.update(p["sources_sha256"])
        for amend in p.get("amendments", []):
            pins.update(amend.get("sources_sha256", {}))
    assert {"runtime/vocr_textspot.c", "runtime/vocr_textspot2.c", "runtime/vocr_textspot3.c",
            "synth/textspot3_twin.py"} <= set(pins)
    for f, digest in pins.items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == digest, f
    mk = (ROOT / "runtime" / "Makefile").read_text()
    check = re.search(r"^check:(.*(?:\n\s+.*)*)$", mk, re.M)
    assert "include vocr_textspot6.mk" in mk and check and "check-textspot6" in check.group(0)
    src = (ROOT / "runtime" / "vocr_textspot6.c").read_text()
    for frozen in ("vocr_textspot.h", "vocr_textspot3.h"):
        assert f'#include "{frozen}"' not in src


def test_v4_needs_the_new_runtime():
    """A version-4 blob is refused by every earlier loader (the twins of textspot, textspot2 and textspot3)."""
    blob = M6.random_blob(TINY_T, 2, 4, n_cal=1)[0]
    for loader in (S.Model, T2.Model, T3.Model):
        with pytest.raises(ValueError):
            loader(blob)

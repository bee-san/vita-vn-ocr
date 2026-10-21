"""Tests of runtime/vocr_textspot3.c, the spotter of VTSP version 1, 2 and 3 blobs, against its twin
(synth/textspot3_twin.py) and, for versions 1 and 2, against textspot2's frozen runtime (runtime/vocr_textspot2.c):
the C loader refuses exactly the blobs the twin refuses (tests/test_textspot3_twin's refusals, textspot2's, random
damage with both CRCs fixed), --info, the driver's self-test, a small tests/textspot3_parity.py run, the --batch
contract on a version-3 probe network (whose result is #97's stride-4 probe's), the arenas (versions 1 and 2 equal
vocr_textspot2.c's; version 3: the up-to arena is one frame's need, grows with both sides, the probe and registered
M shapes within the protocol's 262,144 B at 960 x 544) and the pinned sources. The C parts compile the host drivers
with the system compiler (cc, or $CC) and are skipped without one. No screenshots, fonts or models; no network.
"""
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_parity as TP  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_parity as Q2  # noqa: E402
import textspot2_twin as T2  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_parity as Q3  # noqa: E402
import textspot3_shapes as S3  # noqa: E402
import textspot3_twin as T3  # noqa: E402
import textspot_parity as P  # noqa: E402
import textspot_twin as S  # noqa: E402
from test_textspot2_twin import bad_blobs as bad_v2_blobs  # noqa: E402
from test_textspot2_twin import good_blobs as good_v2_blobs  # noqa: E402
from test_textspot3_twin import BAD as BAD_V3  # noqa: E402
from test_textspot3_twin import good_blobs as good_v3_blobs  # noqa: E402

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
PROTOCOLS = [json.loads((ROOT / "eval" / p).read_text(encoding="utf-8"))
             for p in ("textspot_protocol.json", "textspot2_protocol.json", "textspot3_protocol.json")]
SOURCES = {"vocr_textspot3_test": ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot3.c", "vocr_textspot3_test.c"),
           "vocr_textspot2_test": ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot2.c", "vocr_textspot2_test.c")}
ARENA_MAX = PROTOCOLS[2]["budget"]["arena_max"]
TINY = M3.m_shape("tiny", 4, 8, 8, [8], [1], [(16, [1]), (16, [2])], 8, 8, 1)


def recrc(b):
    """Both CRCs fixed: the VTMS section's (when it is one) and the VTSP blob's."""
    b = bytearray(b)
    if len(b) > 48 and b[32:36] == b"VTMS":
        struct.pack_into("<I", b, 44, zlib.crc32(bytes(b[48:])) & 0xFFFFFFFF)
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("textspot3")
    out = {}
    for name, srcs in SOURCES.items():
        out[name] = d / name
        subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
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
    good = good_v3_blobs() + good_v2_blobs()
    bad = list(BAD_V3) + bad_v2_blobs()
    for _, b in bad:
        with pytest.raises(ValueError):
            T3.Model(b)
    rcs = load_codes(drivers["vocr_textspot3_test"], tmp_path, [b for _, b in good + bad])
    assert rcs[:len(good)] == [0] * len(good)
    wrong = [(n, rc) for (n, _), rc in zip(bad, rcs[len(good):]) if rc != -5]
    assert not wrong, wrong
    # textspot2's frozen loader refuses every version-3 blob, good or bad
    old = load_codes(drivers["vocr_textspot2_test"], tmp_path, [b for _, b in good_v3_blobs()])
    assert old and all(rc == -5 for rc in old)


def test_c_loader_agrees_with_the_twin_on_random_damage(drivers, tmp_path):
    rng = np.random.default_rng(23)
    blobs, want = [], []
    for name, good in good_v3_blobs() + good_v2_blobs()[:2]:
        n_rec = 64 + 32 * 40
        for _ in range(24):
            b = bytearray(good)
            for _ in range(int(rng.integers(1, 3))):
                r = rng.random()
                at = int(rng.integers(4, 64)) if r < 0.3 else int(rng.integers(32, min(len(b), 32 + n_rec))) \
                    if r < 0.7 else int(rng.integers(0, len(b)))
                b[at] = int(rng.integers(0, 256))
            b = recrc(b)
            try:
                T3.Model(b)
                want.append(0)
            except ValueError:
                want.append(-5)
            blobs.append(b)
    assert load_codes(drivers["vocr_textspot3_test"], tmp_path, blobs) == want
    assert 0 < want.count(0) < len(want)


def test_c_info(drivers, tmp_path):
    v3 = T3.pack(M3.random_net(TINY, 5, n_cal=2), 480, 3, 272)
    v2 = T2.pack(X.random_vtdt("s8_c24n4x2", 5, n_cal=2), 600, 3)
    v1 = S.pack(TP.random_blob("s8_c8", 5, 2), 0, 1)
    for name, b in (("a", v3), ("b", v2), ("c", v1), ("d", v3[:-1])):
        (tmp_path / f"{name}.vtsp").write_bytes(b)
    lines = run(drivers["vocr_textspot3_test"], "--info", *(tmp_path / f"{n}.vtsp" for n in "abcd")).splitlines()
    m = T3.Model(v3)
    assert lines[0].split() == [str(v) for v in (0, 3, 16, 8, 4, m.min_cells, 480, 3, len(m.net.stream), len(v3),
                                                 len(v3) - 32, 272, len(m.net.ops), 3)]
    m2 = T2.Model(v2)
    assert lines[1].split() == [str(v) for v in (0, 2, 16, 8, 4, m2.min_cells, 600, 3, len(m2.net.layers), len(v2),
                                                 len(v2) - 32, 0, 0, 0)]
    assert lines[2].split()[:3] == ["0", "1", "4"] and lines[2].split()[-3:] == ["0", "0", "0"]
    assert lines[3] == "-5"


def test_c_selftest(drivers, tmp_path):
    paths = []
    for i, b in enumerate((T3.pack(M3.random_net(TINY, 8, n_cal=2), 480, 1, 272),
                           T3.pack(Q3.probe_net(2, 3), 0, 4, 0), T3.pack(Q3.graph_net(9), 600, 1, 300),
                           T2.pack(X.random_vtdt("s8_c24n4x2", 8, n_cal=2), 600, 1),
                           S.pack(TP.random_blob("s4_c4", 9, 2), 600, 1))):
        paths.append(tmp_path / f"s{i}.vtsp")
        paths[-1].write_bytes(b)
    out = run(drivers["vocr_textspot3_test"], "--selftest", 4, *paths)
    assert out.startswith("TEXTSPOT3_SELFTEST_OK 5 blobs (1 of version 2, 3 of version 3)")


def test_c_matches_the_twin(drivers):
    assert Q3.main(["--bin", str(drivers["vocr_textspot3_test"]), "--v2-bin", str(drivers["vocr_textspot2_test"]),
                    "--seeds", "1", "--frames", "2", "--probe-frames", "2", "--probe-shapes", "pS1",
                    "--graphs", "2", "--full-every", "8", "--maps-every", "3", "--arena-every", "3",
                    "--v1-shapes", "s4_c4,s8_c8", "--d2s-shapes", "s8_c24n4x2", "--seed", "4"]) == 0


def arena(binary, blob_path, w, h, kind="--frame-arena"):
    return int(run(binary, kind, blob_path, w, h))


def test_c_arena_of_versions_1_and_2_is_textspot2s(drivers, tmp_path):
    p = tmp_path / "x.vtsp"
    for blob in (S.pack(TP.random_blob("s4_c24n4", 1, 2), 600, 1), S.pack(TP.random_blob("s8_c8", 1, 2), 0, 1),
                 T2.pack(X.random_vtdt("s8_c48n4x2", 1, n_cal=2), 600, 1),
                 T2.pack(X.random_vtdt("s8_c24n4x2", 1, n_cal=2), 480, 1)):
        p.write_bytes(blob)
        for w, h in ((960, 544), (480, 272), (600, 544), (601, 3), (1, 1), (2048, 9)):
            for kind in ("--arena", "--frame-arena"):
                assert arena(drivers["vocr_textspot3_test"], p, w, h, kind) == \
                    arena(drivers["vocr_textspot2_test"], p, w, h, kind), (w, h, kind)


def m_shapes():
    out = dict(M3.PROBES)
    out.update({k: v for k, v in S3.M.items()})
    return out


def test_c_arena_of_version_3(drivers, tmp_path):
    """The up-to arena is one frame's need (the largest frame read at full size, or the frame itself when halved),
    covers every smaller frame, grows with both sides; every probe and registered M shape fits the protocol's
    262,144 B at 960 x 544 but the two the probe table marks over it."""
    p = tmp_path / "m.vtsp"
    over = {"pL4", "qL5n6"}
    for name, shape in m_shapes().items():
        p.write_bytes(M3.random_blob(shape, 1, n_cal=1)[0])
        up = arena(drivers["vocr_textspot3_test"], p, 960, 544, "--arena")
        full = arena(drivers["vocr_textspot3_test"], p, 480, 272)
        halved = arena(drivers["vocr_textspot3_test"], p, 960, 544)
        assert up == max(full, halved) and up >= arena(drivers["vocr_textspot3_test"], p, 725, 544)
        assert (up <= ARENA_MAX) == (name not in over), (name, up)
    p.write_bytes(T3.pack(Q3.graph_net(31), 0, 1, 0))           # never halved: the need grows with each side
    prev = 0
    for w, h in ((1, 1), (17, 9), (64, 64), (300, 200), (960, 544), (2048, 2048)):
        a = arena(drivers["vocr_textspot3_test"], p, w, h)
        assert a >= prev and a == arena(drivers["vocr_textspot3_test"], p, w, h, "--arena")
        prev = a
    assert arena(drivers["vocr_textspot3_test"], p, 960, 545) >= arena(drivers["vocr_textspot3_test"], p, 960, 544)
    assert arena(drivers["vocr_textspot3_test"], p, 961, 544) >= arena(drivers["vocr_textspot3_test"], p, 960, 544)


def test_c_batch_contract(drivers, tmp_path):
    """--batch BLOB FILE as vocr_textspot2_test's: one line per record, rc count block_count pick truncated, then
    8 integers per box and 7 per block; flag 2 (VOCR_TEXTSPOT_NO_DARK) the same with every dark 0. The version-3
    probe gives the stride-4 probe's result."""
    blob = T3.pack(Q3.probe_net(0), 480, 1, 272)
    (tmp_path / "p.vtsp").write_bytes(blob)
    model = T3.Model(blob)
    g = Q3.probe_frame3(np.random.default_rng(1), model, 960, 544)
    P.write_batch(tmp_path / "r.bin", [(g, 0), (g, 1), (g, 2), (np.zeros((3, 3), np.uint8), 0)])
    out = run(drivers["vocr_textspot3_test"], "--batch", tmp_path / "p.vtsp", tmp_path / "r.bin").splitlines()
    r = T3.spot(model, g)
    assert out == [P.twin_line(r), P.twin_line(r), P.twin_line(r, dark=False), "0 0 0 -1 0"]
    v = [int(t) for t in out[0].split()]
    assert len(v) == 5 + 8 * v[1] + 7 * v[2] and v[1] > 1 and v[3] >= 0
    ref = Q3.probe_reference(model)
    x, f = T3.net_input(model, g)
    import textdet_twin as TD
    assert out[0] == P.twin_line(T2.spot_maps(ref, TD.run_net(ref.net, x), f, g))


def test_new_runtime_files_and_frozen_sources():
    """The version-3 runtime is files of its own; every source a protocol pins is byte for byte as pinned."""
    for f in ("vocr_textspot3.c", "vocr_textspot3.h", "vocr_textspot3_test.c", "vocr_textspot3.mk"):
        assert (ROOT / "runtime" / f).exists()
    pins = {}
    for p in PROTOCOLS:
        pins.update(p["sources_sha256"])
        for amend in p.get("amendments", []):
            pins.update(amend.get("sources_sha256", {}))
    assert {"runtime/vocr_textspot.c", "runtime/vocr_textspot2.c", "runtime/vocr_textspot2.h",
            "synth/textspot2_twin.py", "train/textspot2_export.py", "eval/textspot2_a9.py"} <= set(pins)
    for f, digest in pins.items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == digest, f
    mk = (ROOT / "runtime" / "Makefile").read_text()
    check = re.search(r"^check:(.*(?:\n\s+.*)*)$", mk, re.M)
    assert "include vocr_textspot3.mk" in mk and check and "check-textspot3" in check.group(0)


def test_v3_needs_the_new_runtime():
    """A version-3 blob is refused by every earlier loader (the twins of textspot and textspot2)."""
    blob = T3.pack(M3.random_net(TINY, 2, n_cal=1), 480, 1, 272)
    for loader in (S.Model, T2.Model):
        with pytest.raises(ValueError):
            loader(blob)
    assert Q2.probe_vtdt(0)[:4] == b"VTDT"

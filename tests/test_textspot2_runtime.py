"""Tests of runtime/vocr_textspot2.c, the spotter of VTSP version 1 and 2 (d2s) blobs, against its twin
(synth/textspot2_twin.py) and, for version 1, against textspot's frozen runtime (runtime/vocr_textspot.c): the C
loader refuses exactly the blobs the twin refuses (tests/test_textspot2_twin.bad_blobs, textspot's
tests/test_textspot_runtime.bad_blobs, random damage with the outer CRC fixed), --info, the driver's self-test, a
small tests/textspot2_parity.py run, the --batch contract, the arenas (a version-1 blob's equal vocr_textspot.c's;
the d2s shapes within the always-safe tier) and the pinned sources. The C parts compile the host drivers with the
system compiler (cc, or $CC) and are skipped without one. No screenshots, fonts or models; no network.
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

# The recognizer runtime (runtime/vocr.c, vocr.h) is the harness these protocols read text with, not part of the
# spotter; it may change with other work as long as its output does not (runtime/tests/test_vocr.c pins its logits
# by known-answer hashes, runtime/tests/parity.py against runtime/tests/vocr_ref.py), so it is not pinned here, as
# in tests/test_lines_split.py. The head gate (#117) changed it with every output bit for bit the same.
HARNESS = {"runtime/vocr.c", "runtime/vocr.h"}

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_parity as TP  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_parity as Q  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_twin as T2  # noqa: E402
import textspot_parity as P  # noqa: E402
import textspot_twin as S  # noqa: E402
from test_textspot2_twin import bad_blobs as bad_v2_blobs  # noqa: E402
from test_textspot2_twin import good_blobs as good_v2_blobs  # noqa: E402
from test_textspot_runtime import bad_blobs as bad_v1_blobs  # noqa: E402
from test_textspot_runtime import good_blobs as good_v1_blobs  # noqa: E402

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
PROTOCOLS = [json.loads((ROOT / "eval" / p).read_text(encoding="utf-8"))
             for p in ("textspot_protocol.json", "textspot2_protocol.json")]
SOURCES = {"vocr_textspot2_test": ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot2.c", "vocr_textspot2_test.c"),
           "vocr_textspot_test": ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot.c", "vocr_textspot_test.c")}
ARENA_MAX = PROTOCOLS[1]["rule"]["arena_max"]


def recrc(b):
    b = bytearray(b)
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("textspot2")
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
    good = good_v2_blobs() + good_v1_blobs()
    bad = bad_v2_blobs() + bad_v1_blobs()
    for _, b in bad:
        with pytest.raises(ValueError):
            T2.Model(b)
    rcs = load_codes(drivers["vocr_textspot2_test"], tmp_path, [b for _, b in good + bad])
    assert rcs[:len(good)] == [0] * len(good)
    wrong = [(n, rc) for (n, _), rc in zip(bad, rcs[len(good):]) if rc != -5]
    assert not wrong, wrong
    # textspot's frozen loader refuses every version-2 blob, good or bad
    old = load_codes(drivers["vocr_textspot_test"], tmp_path, [b for _, b in good_v2_blobs() if b[4] == 2])
    assert old and all(rc == -5 for rc in old)


def test_c_loader_agrees_with_the_twin_on_random_damage(drivers, tmp_path):
    rng = np.random.default_rng(17)
    blobs, want = [], []
    for name, good in good_v2_blobs() + good_v1_blobs()[:2]:
        for _ in range(30):
            b = bytearray(good)
            for _ in range(int(rng.integers(1, 3))):
                at = int(rng.integers(4, 64)) if rng.random() < 0.5 else int(rng.integers(0, len(b)))
                b[at] = int(rng.integers(0, 256))
            b = recrc(b)
            try:
                T2.Model(b)
                want.append(0)
            except ValueError:
                want.append(-5)
            blobs.append(b)
    assert load_codes(drivers["vocr_textspot2_test"], tmp_path, blobs) == want
    assert 0 < want.count(0) < len(want)


def test_c_info(drivers, tmp_path):
    v2 = T2.pack(X.random_vtdt("s8_c24n4x2", 5, n_cal=2), 600, 3)
    v1 = S.pack(TP.random_blob("s8_c8", 5, 2), 0, 1)
    (tmp_path / "a.vtsp").write_bytes(v2)
    (tmp_path / "b.vtsp").write_bytes(v1)
    (tmp_path / "c.vtsp").write_bytes(v2[:-1])
    lines = run(drivers["vocr_textspot2_test"], "--info", tmp_path / "a.vtsp", tmp_path / "b.vtsp",
                tmp_path / "c.vtsp").splitlines()
    m = T2.Model(v2)
    assert lines[0].split() == [str(v) for v in (0, 2, 16, 8, 4, m.min_cells, 600, 3, len(m.net.layers), len(v2),
                                                 len(v2) - 32)]
    assert lines[1].split() == [str(v) for v in (0, 1, 4, 8, 8, 1, 0, 1, len(S.Model(v1).net.layers), len(v1),
                                                 len(v1) - 32)]
    assert lines[2] == "-5"


def test_c_selftest(drivers, tmp_path):
    paths = []
    for i, b in enumerate((T2.pack(X.random_vtdt("s8_c24n4x2", 8, n_cal=2), 600, 1),
                           T2.pack(Q.probe_vtdt(2), 0, 4), S.pack(TP.random_blob("s4_c4", 9, 2), 600, 1))):
        paths.append(tmp_path / f"s{i}.vtsp")
        paths[-1].write_bytes(b)
    out = run(drivers["vocr_textspot2_test"], "--selftest", 5, *paths)
    assert out.startswith("TEXTSPOT2_SELFTEST_OK 3 blobs (2 of version 2)")


def test_c_matches_the_twin(drivers):
    assert Q.main(["--bin", str(drivers["vocr_textspot2_test"]), "--v1-bin", str(drivers["vocr_textspot_test"]),
                   "--seeds", "1", "--frames", "3", "--probe-frames", "3", "--shapes", "s4_c4,s8_c8,s4p_c6n4",
                   "--full-every", "6", "--wild", "1", "--seed", "4"]) == 0


def arena(binary, blob_path, w, h, kind="--frame-arena"):
    return int(run(binary, kind, blob_path, w, h))


def test_c_arena_of_version_1_is_textspots_and_d2s_fits_the_tier(drivers, tmp_path):
    p = tmp_path / "x.vtsp"
    S2.register()
    for name, s in S2.SHAPES.items():
        for ha in (0, 600):
            if s["head"] == "kroa":
                p.write_bytes(S.pack(TP.random_blob(name, 1, 2), ha, 1))
                for w, h in ((960, 544), (480, 272), (600, 544), (601, 3), (1, 1), (2048, 9)):
                    for kind in ("--arena", "--frame-arena"):
                        assert arena(drivers["vocr_textspot2_test"], p, w, h, kind) == \
                            arena(drivers["vocr_textspot_test"], p, w, h, kind), (name, ha, w, h, kind)
            else:
                p.write_bytes(T2.pack(X.random_vtdt(name, 1, n_cal=2), ha, 1))
            f, a = arena(drivers["vocr_textspot2_test"], p, 960, 544), \
                arena(drivers["vocr_textspot2_test"], p, 960, 544, "--arena")
            assert f <= a and (ha == 0 or a <= ARENA_MAX), (name, ha, f, a)
            assert arena(drivers["vocr_textspot2_test"], p, 960, 1) == \
                arena(drivers["vocr_textspot2_test"], p, 960, 2048) == f
            if ha:
                full = arena(drivers["vocr_textspot2_test"], p, ha, 544)
                assert a == max(f, full)


def test_c_batch_contract(drivers, tmp_path):
    """--batch BLOB FILE as vocr_textspot_test's: one line per record, rc count block_count pick truncated, then
    8 integers per box and 7 per block; flag 2 (VOCR_TEXTSPOT_NO_DARK) the same with every dark 0."""
    blob = T2.pack(Q.probe_vtdt(0), 600, 1)
    (tmp_path / "p.vtsp").write_bytes(blob)
    model = T2.Model(blob)
    g = P.probe_frame(np.random.default_rng(1), model, 960, 544)
    P.write_batch(tmp_path / "r.bin", [(g, 0), (g, 1), (g, 2), (np.zeros((3, 3), np.uint8), 0)])
    out = run(drivers["vocr_textspot2_test"], "--batch", tmp_path / "p.vtsp", tmp_path / "r.bin").splitlines()
    r = T2.spot(model, g)
    assert out == [P.twin_line(r), P.twin_line(r), P.twin_line(r, dark=False), "0 0 0 -1 0"]
    v = [int(t) for t in out[0].split()]
    assert len(v) == 5 + 8 * v[1] + 7 * v[2] and v[1] > 1 and v[3] >= 0
    assert out[0] == P.twin_line(S.spot(Q.probe_twin(blob), g))          # the stride-4 probe's result


def test_new_runtime_files_and_frozen_sources():
    """The d2s runtime is files of its own; every source either protocol pins is byte for byte as frozen."""
    for f in ("vocr_textspot2.c", "vocr_textspot2.h", "vocr_textspot2_test.c", "vocr_textspot2.mk"):
        assert (ROOT / "runtime" / f).exists()
    pins = {}
    for p in PROTOCOLS:
        pins.update(p["sources_sha256"])
        for amend in p.get("amendments", []):
            pins.update(amend.get("sources_sha256", {}))
    assert {"runtime/vocr_textspot.c", "runtime/vocr_textspot.h", "synth/textspot_twin.py",
            "synth/textdet_twin.py", "train/textdet_export.py", "eval/textspot_a9.py"} <= set(pins)
    for f, digest in pins.items():
        if f in HARNESS:
            continue
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == digest, f
    mk = (ROOT / "runtime" / "Makefile").read_text()
    check = re.search(r"^check:(.*)$", mk, re.M)
    assert "include vocr_textspot2.mk" in mk and check and "check-textspot2" in check.group(1)


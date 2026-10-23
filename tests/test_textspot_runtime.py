"""Tests of the full-frame text spotter's C runtime (runtime/vocr_textspot.c) against its twin
(synth/textspot_twin.py): the C loader refuses exactly the blobs the twin refuses (every VTSP header check, every
inner VTDT check of tests/test_textdet.bad_blobs, random damage with the outer CRC fixed), the driver's self-test,
the --batch contract, the arena (width only, the frame and up-to forms, family B within the always-safe tier) and a
small tests/textspot_parity.py run. The C parts compile the host driver with the system compiler (cc, or $CC) and
are skipped without one. No screenshots, fonts or models; no network.
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
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as E  # noqa: E402
import textdet_parity as TP  # noqa: E402
import textspot_parity as P  # noqa: E402
import textspot_twin as S  # noqa: E402
from test_textdet import bad_blobs as bad_vtdt_blobs  # noqa: E402

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
PROTOCOL = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text(encoding="utf-8"))
SOURCES = ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot.c", "vocr_textspot_test.c")


def vtdt(shape="s4_c4", seed=3):
    sh, q = E.random_int_layers(shape, seed, n_cal=2)
    return E.pack(sh, q, hashlib.sha256(f"{shape}:{seed}".encode()).digest())


def raw_vtsp(inner, half_above=600, conf_min=1, r0=0, off=32, size=None, r1=0, version=1):
    """A VTSP header around any bytes, without the twin's checks (so a bad inner blob can be wrapped)."""
    size = len(inner) if size is None else size
    total = 32 + len(inner)
    body = struct.pack("<HBBIII", half_above, conf_min, r0, off, size, r1) + inner
    return b"VTSP" + struct.pack("<III", version, total, zlib.crc32(body) & 0xFFFFFFFF) + body


def recrc(b):
    b = bytearray(b)
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


def bad_blobs():
    """(name, blob) pairs every VTSP loader must refuse."""
    good = S.pack(vtdt(), 600, 1)
    out = [("short", good[:31]), ("truncated", recrc(good[:-4])), ("empty", b"")]

    def edit(name, fn, crc=True):
        b = bytearray(good)
        fn(b)
        out.append((name, recrc(b) if crc else bytes(b)))

    edit("magic", lambda b: b.__setitem__(3, ord("T")))
    edit("version", lambda b: struct.pack_into("<I", b, 4, 2))
    edit("size", lambda b: struct.pack_into("<I", b, 8, len(good) + 4))
    edit("crc", lambda b: b.__setitem__(len(b) - 1, b[-1] ^ 1), crc=False)
    edit("conf_min 0", lambda b: b.__setitem__(18, 0))
    edit("conf_min 128", lambda b: b.__setitem__(18, 128))
    edit("conf_min 255", lambda b: b.__setitem__(18, 255))
    edit("reserved u8", lambda b: b.__setitem__(19, 1))
    edit("offset 36", lambda b: struct.pack_into("<I", b, 20, 36))
    edit("offset 0", lambda b: struct.pack_into("<I", b, 20, 0))
    edit("net size short", lambda b: struct.pack_into("<I", b, 24, len(good) - 36))
    edit("net size long", lambda b: struct.pack_into("<I", b, 24, len(good) - 28))
    edit("net size huge", lambda b: struct.pack_into("<I", b, 24, 0xFFFFFFF0))
    edit("reserved u32", lambda b: struct.pack_into("<I", b, 28, 1 << 31))
    edit("inner magic", lambda b: b.__setitem__(32, ord("X")))
    edit("inner crc", lambda b: b.__setitem__(len(b) - 1, b[-1] ^ 0x40))   # outer CRC fixed: the inner one fails
    out.append(("no network", raw_vtsp(b"")))
    out.append(("short network", raw_vtsp(vtdt()[:63])))
    for name, bad in bad_vtdt_blobs():                                     # every inner check of textdet_twin.Model
        out.append((f"inner {name}", raw_vtsp(bad)))
    return out


def good_blobs():
    """Edge values the loader accepts."""
    v = vtdt()
    return [("h0", S.pack(v, 0, 1)), ("h600", S.pack(v, 600, 1)), ("h65535 c127", S.pack(v, 65535, 127)),
            ("h1 c2", S.pack(v, 1, 2)), ("s8", S.pack(vtdt("s8_c8", 1), 600, 5))]


@pytest.mark.parametrize("name,blob", bad_blobs(), ids=[n for n, _ in bad_blobs()])
def test_the_twin_refuses(name, blob):
    with pytest.raises(ValueError):
        S.Model(blob)


def test_the_twin_accepts_the_edge_values():
    for _, blob in good_blobs():
        S.Model(blob)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("textspot") / "vocr_textspot_test"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    *(str(ROOT / "runtime" / f) for f in SOURCES), "-o", str(out)], check=True)
    return out


def load_codes(driver, tmp_path, blobs):
    paths = []
    for i, b in enumerate(blobs):
        p = tmp_path / f"b{i}.vtsp"
        p.write_bytes(b)
        paths.append(str(p))
    return [int(v) for v in subprocess.run([str(driver), "--load", *paths], check=True, capture_output=True,
                                           text=True).stdout.split()]


def test_c_loader_refuses_what_the_twin_refuses(driver, tmp_path):
    bad, good = bad_blobs(), good_blobs()
    rcs = load_codes(driver, tmp_path, [b for _, b in good + bad])
    assert rcs[:len(good)] == [0] * len(good)
    wrong = [(n, rc) for (n, _), rc in zip(bad, rcs[len(good):]) if rc != -5]
    assert not wrong, wrong


def test_c_loader_agrees_with_the_twin_on_random_damage(driver, tmp_path):
    """Random byte edits anywhere, with the outer CRC fixed: the C loader accepts exactly what the twin accepts (an
    edit of half_above is accepted by both, a conf_min of 0 or a bad layer field refused by both, ...)."""
    rng = np.random.default_rng(7)
    blobs, want = [], []
    for k, (name, good) in enumerate(good_blobs()):
        for _ in range(40):
            b = bytearray(good)
            for _ in range(int(rng.integers(1, 3))):
                at = int(rng.integers(16, 32)) if rng.random() < 0.4 else int(rng.integers(0, len(b)))
                b[at] = int(rng.integers(0, 256))
            b = recrc(b)
            try:
                S.Model(b)
                want.append(0)
            except ValueError:
                want.append(-5)
            blobs.append(b)
    got = load_codes(driver, tmp_path, blobs)
    assert got == want
    assert 0 < want.count(0) < len(want)      # both outcomes occur


def test_c_selftest(driver, tmp_path):
    a, b = tmp_path / "a.vtsp", tmp_path / "b.vtsp"
    a.write_bytes(S.pack(vtdt(), 600, 1))
    b.write_bytes(S.pack(TP.probe_blob(8, 1), 0, 3))
    out = subprocess.run([str(driver), "--selftest", "6", str(a), str(b)], check=True, capture_output=True,
                         text=True, timeout=600).stdout
    assert out.startswith("TEXTSPOT_SELFTEST_OK 2 blobs")


def test_c_matches_the_twin(driver):
    assert P.main(["--bin", str(driver), "--seeds", "1", "--frames", "3", "--probe-frames", "4",
                   "--shapes", "s4_c4,s8_c8,s4_c8,s8_c16n4", "--full-every", "5", "--wild", "1", "--seed", "3"]) == 0


def arena(driver, blob_path, w, h, kind="--frame-arena"):
    return int(subprocess.run([str(driver), kind, str(blob_path), str(w), str(h)], check=True, capture_output=True,
                              text=True).stdout)


def test_c_arena_follows_the_width_and_fits_the_tier_for_family_b(driver, tmp_path):
    p = tmp_path / "x.vtsp"
    fam = PROTOCOL["candidates"]["family_b"]["shapes"]
    for scale, shapes in fam.items():
        ha = PROTOCOL["candidates"]["family_b"]["scales"][scale]["half_above"]
        for name in shapes:
            p.write_bytes(S.pack(vtdt(name, 1), ha, 1))
            f, a = arena(driver, p, 960, 544), arena(driver, p, 960, 544, "--arena")
            assert f <= a <= PROTOCOL["rule"]["arena_max"], (scale, name, f, a)
            assert arena(driver, p, 960, 1) == arena(driver, p, 960, 2048) == f
            if ha:          # every frame up to 960 wide: the 600-px frame (read at full size) needs the most
                full = arena(driver, p, ha, 544)
                assert a == max(f, full) and arena(driver, p, ha + 1, 544) < full
            else:
                assert a == f
            assert arena(driver, p, 480, 272) == arena(driver, p, 480, 272, "--arena") < f


def test_c_batch_contract(driver, tmp_path):
    """--batch BLOB FILE, records <4I (w, h, bpp, flags) + pixels; one line each: rc count block_count pick truncated,
    then x y w h vertical conf dark block per box and x y w h vertical first count per block. Flag 2
    (VOCR_TEXTSPOT_NO_DARK): the same with every dark 0."""
    blob = S.pack(TP.probe_blob(4, 0), 600, 1)
    (tmp_path / "p.vtsp").write_bytes(blob)
    model = S.Model(blob)
    rng = np.random.default_rng(1)
    g = P.probe_frame(rng, model, 960, 544)
    P.write_batch(tmp_path / "r.bin", [(g, 0), (g, 1), (g, 2), (np.zeros((3, 3), np.uint8), 0)])
    out = subprocess.run([str(driver), "--batch", str(tmp_path / "p.vtsp"), str(tmp_path / "r.bin")], check=True,
                         capture_output=True, text=True).stdout.splitlines()
    r = S.spot(model, g)
    assert out == [P.twin_line(r), P.twin_line(r), P.twin_line(r, dark=False), "0 0 0 -1 0"]
    v = [int(t) for t in out[0].split()]
    assert len(v) == 5 + 8 * v[1] + 7 * v[2] and v[1] > 1 and v[3] >= 0


def test_frozen_runtime_sources_are_unchanged():
    """The spotter is a file of its own: the finders, #97's detector, the gate, the column finder and every header the
    protocol pins are byte for byte as frozen."""
    pins = dict(PROTOCOL["sources_sha256"])
    for amend in PROTOCOL.get("amendments", []):      # an amendment re-pins a source (tests/test_textspot.py)
        pins.update(amend.get("sources_sha256", {}))
    frozen = [f for f in pins if f.startswith("runtime/")]
    assert {"runtime/vocr_textdet.c", "runtime/vocr_textgate.c", "runtime/vocr_vertical.c",
            "runtime/vocr_lines.c", "runtime/vocr_lines_rows.c"} <= set(frozen)
    for f in frozen:
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == pins[f], f

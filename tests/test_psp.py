"""Tests of the PSP harness's host-side tools (psp/): crop packs, the expected-output file, the
PPSSPP driver's vocr_cli-compatible command line, the app config keys, the stack report and
the D-cache model. None of them needs the PSP toolchain or PPSSPP; the cache model needs a C
compiler and make and is skipped without them.
"""
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "psp"), str(ROOT / "runtime"), str(ROOT / "runtime" / "tests")]
import pack  # noqa: E402
import ppsspp_cli  # noqa: E402
import stack  # noqa: E402

MAIN_C = (ROOT / "psp" / "main.c").read_text()


def app_cfg_keys():
    """The keys psp/main.c's cfg_set() accepts."""
    body = MAIN_C[MAIN_C.index("static int cfg_set"):MAIN_C.index("static int cfg_read")]
    return set(re.findall(r'\{"(\w+)", c->', body)) | set(re.findall(r'!strcmp\(k, "(\w+)"\)', body))


def test_pack_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (21, 37), dtype=np.uint8), rng.integers(0, 256, (16, 5, 4), dtype=np.uint8)]
    path = tmp_path / "lines.bin"
    pack.write_pack(path, crops)
    raw = path.read_bytes()
    assert struct.unpack_from("<I", raw) == (2,)
    assert struct.unpack_from("<III", raw, 4) == (21, 37, 1)          # vocr_cli's pack format
    back = pack.read_pack(path)
    assert [b.shape for b in back] == [c.shape for c in crops]
    assert all(np.array_equal(a, b) for a, b in zip(back, crops))
    path.write_bytes(raw + b"x")
    with pytest.raises(ValueError):
        pack.read_pack(path)


def test_expect_file_format(tmp_path):
    path = tmp_path / "expect.txt"
    pack.write_expect(path, [(0x1234abcd, "「テスト」"), (0, "")])
    assert path.read_text(encoding="utf-8") == "1234abcd\t「テスト」\n00000000\t\n"
    with pytest.raises(ValueError):
        pack.write_expect(path, [(1, "a\tb")])


def test_read_pnm_grey_and_rgb(tmp_path):
    p5 = tmp_path / "a.pgm"
    p5.write_bytes(b"P5\n# comment\n3 2\n255\n" + bytes(range(6)))
    assert ppsspp_cli.read_pnm(p5) == (2, 3, 1, bytes(range(6)))
    p6 = tmp_path / "b.ppm"
    p6.write_bytes(b"P6 2 1 255\n" + bytes([1, 2, 3, 4, 5, 6]))
    assert ppsspp_cli.read_pnm(p6) == (1, 2, 4, bytes([1, 2, 3, 255, 4, 5, 6, 255]))   # RGBA, as vocr_cli


def test_driver_takes_vocr_cli_arguments():
    """The argument lists runtime/tests/parity.py (CliBackend) and eval's cmd: protocol pass."""
    known = app_cfg_keys()
    for argv in (["m.vocr", "--mode", "w8i", "--crc", "--tile", "3", "--head", "7", "--invert",
                  "--pack", "in.bin", "--logits", "l.bin", "--input-out", "i.bin"],
                 ["m.vocr", "--mode", "f32", "--tile", "1", "--head", "1", "--arena", "8,100,1024"],
                 ["m.vocr", "--mode", "w8f", "--stdin"],
                 ["m.vocr", "a.pgm", "b.ppm", "--repeat", "3"]):
        a = ppsspp_cli.parse_args(argv)
        cfg = ppsspp_cli.make_cfg(a)
        assert set(cfg) <= known, set(cfg) - known
    a = ppsspp_cli.parse_args(["m.vocr", "--mode", "w8i", "--tile", "3", "--head", "7", "--invert"])
    cfg = ppsspp_cli.make_cfg(a)
    assert (cfg["mode"], cfg["tile"], cfg["head"], cfg["invert"], cfg["wait"]) == ("w8i", 3, 7, 1, 0)


def test_bundle_config_keys_are_the_apps():
    cfg = pack.bundle_cfg("w8f", invert=True, repeat=2, clock=222, tile=16)
    assert set(cfg) <= app_cfg_keys()
    assert cfg == {"mode": "w8f", "invert": 1, "repeat": 2, "crc": 1, "clock": 222, "wait": 1, "tile": 16, "head": 0}


def test_stack_report_follows_the_deepest_chain(tmp_path):
    ci = tmp_path / "x.ci"
    ci.write_text('graph: { title: "x.c"\n'
                  'node: { title: "vocr_run" label: "vocr_run\\nx.c:1:1\\n40 bytes (static)\\n0 dynamic objects" }\n'
                  'node: { title: "step" label: "step\\nx.c:2:1\\n100 bytes (static)\\n0 dynamic objects" }\n'
                  'node: { title: "small" label: "small\\nx.c:3:1\\n8 bytes (static)\\n0 dynamic objects" }\n'
                  'node: { title: "memset" label: "memset\\n<built-in>" }\n'
                  'edge: { sourcename: "vocr_run" targetname: "small" label: "x.c:1:2" }\n'
                  'edge: { sourcename: "vocr_run" targetname: "step" label: "x.c:1:3" }\n'
                  'edge: { sourcename: "step" targetname: "memset" label: "x.c:2:2" }\n}\n')
    nodes, edges = stack.parse(ci.read_text())
    assert nodes["step"] == (100, "static") and nodes["memset"] is None
    assert edges["vocr_run"] == ["small", "step"]
    out = subprocess.run([sys.executable, str(ROOT / "psp" / "stack.py"), str(ci), "--json"],
                         capture_output=True, text=True, check=True).stdout
    rep = json.loads(out)
    assert rep["entries"]["vocr_run"]["bytes"] == 140
    assert rep["entries"]["vocr_run"]["chain"] == ["vocr_run 40", "step 100", "memset (extern)"]
    assert rep["externs"] == ["memset"]


@pytest.mark.skipif(not (shutil.which("make") and shutil.which("gcc")),
                    reason="needs make and gcc (cachesim uses gcc's KASAN callbacks)")
def test_cachesim_counts_misses(tmp_path):
    from parity import fixture_lines, fixture_model

    import export as E
    subprocess.run(["make", "-s", "-C", str(ROOT / "psp"), "cachesim", "PSPDEV=/nonexistent"], check=True,
                   capture_output=True)
    model, vocab = fixture_model(0, 0.5)
    blob, _ = E.build_blob(E.model_ops(model, 32), vocab, 32, True, 1024, b"fixture")
    (tmp_path / "m.vocr").write_bytes(blob)
    pack.write_pack(tmp_path / "lines.bin", [img for _, img, _ in fixture_lines(4, 1)])
    out = subprocess.run([str(ROOT / "psp" / "build" / "cachesim"), str(tmp_path / "m.vocr"), "--mode", "w8f",
                          "--pack", str(tmp_path / "lines.bin")], capture_output=True, text=True, check=True).stdout
    rep = json.loads(out)
    assert rep["lines"] == 4 and rep["frames"] > 0 and rep["load_bytes"] > 0
    for c in rep["configs"]:
        assert 0 < c["misses"] < c["accesses"] and c["writebacks"] <= c["misses"]

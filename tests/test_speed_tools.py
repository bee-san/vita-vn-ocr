"""The speed tools of vita/: the per-layer profile (a9_layers.py), the development pack (dev_pack.py) and the
parity driver (speed_parity.py). None of it needs VitaSDK, qemu or screenshots: the profile's parsing and
cost model run on synthetic counts, the pack's CTC frame count against the C runtime's own resize rule."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VITA = ROOT / "vita"
sys.path[:0] = [str(VITA)]
import a9_cycles  # noqa: E402
import a9_layers  # noqa: E402
import dev_pack  # noqa: E402


def test_layers_file_round_trip(tmp_path):
    p = tmp_path / "layers.txt"
    p.write_text("0 10 5\n0 14 7\n19 10 2\n# 19 switches_l_free 3\n# 19 switches_l_vfp 9\n# 19 l1d_misses 4\n"
                 "# 19 prefetches 6\n")
    counts, extra = a9_layers.read_layers(p)
    assert counts == {0: {0x10: 5, 0x14: 7}, 19: {0x10: 2}}
    assert extra[19] == {"switches_l_free": 3, "switches_l_vfp": 9, "l1d_misses": 4, "prefetches": 6}


def test_phase_cost_is_a9_cycles_model_for_one_phase():
    """The same bounds as vita/a9_cycles.model_mode, for the instructions of one phase."""
    rows = {0x10: ("mm_i16", "vmlal.s16", "q0, d1, d2[0]"), 0x14: ("mm_i16", "vstr", "d24, [sp, #8]"),
            0x18: ("mm_i16", "add.w", "r1, r1, #8"), 0x1c: ("main", "add.w", "r1, r1, #8")}
    counts = {0x10: 800, 0x14: 100, 0x18: 400, 0x1c: 10**6}        # main is the CLI: not counted
    ex = {"switches_l_free": 0, "switches_l_vfp": 200, "l1d_misses": 10, "l2_misses": 1}
    r = a9_layers.phase_cost(rows, counts, ex, {"mm_i16"}, 444)
    ref = a9_cycles.model_mode(rows, counts, {"switches_l_free": 0, "switches_l_vfp": 200}, {"mm_i16"}, frames=1,
                               cache={"l1d_misses": 10, "l2_misses": 1})
    assert r["insns"] == 1300
    assert r["low"] == pytest.approx(ref["cycles_per_frame"][444][0])
    assert r["high"] == pytest.approx(ref["cycles_per_frame"][444][1])
    assert r["functions"][0] == ("mm_i16", 1300)


def test_calibration_maps_the_summed_high_bound_onto_the_console():
    class A:
        console_ms, console_mhz, scale = 5.07, 333, 0.0
    res = {0: {"high": 1000.0}, 1: {"high": 3000.0}}
    # 4,000 cycles over 2 frames = 2,000 per frame; the console's 5.07 ms at 333 MHz is 1,688,310
    assert a9_layers.calib_scale(res, 2, A) == pytest.approx(5.07 * 333 * 1000 / 2000)
    A.scale = 1.5                                         # a fixed factor (another model, the same console)
    assert a9_layers.calib_scale(res, 2, A) == 1.5
    A.console_ms, A.scale = 0.0, 0.0
    assert a9_layers.calib_scale(res, 2, A) == 1.0


def write_pack(path, crops):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(crops)))
        for c in crops:
            f.write(struct.pack("<III", c.shape[0], c.shape[1], 1))
            f.write(c.tobytes())


def test_subset_pack_keeps_lines_by_ctc_frames(tmp_path):
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (h, w), dtype=np.uint8) for h, w in ((32, 64), (32, 640), (16, 640), (30, 100))]
    write_pack(tmp_path / "all.bin", crops)
    frames = [dev_pack.frames(c.shape[1], c.shape[0]) for c in crops]
    assert frames == [8, 80, 128, 14]                     # 640 px at height 16 is 1,280 px: clamped to 1,024
    n, nf = a9_layers.subset_pack(tmp_path / "all.bin", tmp_path / "sub.bin", 10, 100, 0)
    assert (n, nf) == (2, 94)
    data = (tmp_path / "sub.bin").read_bytes()
    assert struct.unpack_from("<I", data)[0] == 2
    h, w, ch = struct.unpack_from("<III", data, 4)
    assert (h, w, ch) == (32, 640, 1) and data[16:16 + h * w] == crops[1].tobytes()
    assert a9_layers.subset_pack(tmp_path / "all.bin", tmp_path / "one.bin", 0, 10**6, 1) == (1, 8)


def test_dev_pack_frames_match_the_runtime(tmp_path):
    """dev_pack.frames is vocr_resized_width (round half to even, clamped to [8, 1024]) padded to 8."""
    sys.path[:0] = [str(ROOT / "train"), str(ROOT / "runtime")]
    import budget_sweep as BS
    from pyvocr import Vocr

    import export
    data, _ = export.build_blob(export.model_ops(BS.random_model(0.5, 48, 1, 2, 5, 101), 32), BS.vocab_of(101), 32,
                                int8=True)
    (tmp_path / "m.vocr").write_bytes(data)
    r = Vocr(str(tmp_path / "m.vocr"), mode="w8i")
    for w, h in ((1, 1), (5, 16), (21, 16), (23, 16), (640, 32), (883, 35), (3000, 20), (100, 7), (999, 37),
                 (835, 35), (876, 45)):
        assert dev_pack.frames(w, h) == (r.resized_width(w, h) + 7) // 8, (w, h)


def test_dev_pack_refuses_protected_vns():
    bad = dev_pack.protected_vns()
    t = json.loads((ROOT / "eval/manifests/test_games.json").read_text(encoding="utf-8"))
    reserved = json.loads((ROOT / "eval/manifests/reserved.json").read_text(encoding="utf-8"))
    assert "v4" in bad                                    # CLANNAD
    assert set(t["vita-test"]) <= bad and set(t["psp-test"]) <= bad and set(t["all"]) <= bad
    assert set(reserved["test"]) <= bad                   # the committed test reservation (CLANNAD family, partners)
    assert {"v4060", "v36", "v1267"} <= bad               # the three reserved VNs the dev manifests hold
    kept, refused = dev_pack.dev_rows(ROOT / "eval/linefind/manifests")
    rows = [json.loads(ln) for name in dev_pack.DEV_MANIFESTS
            for ln in (ROOT / "eval/linefind/manifests" / name).read_text().splitlines()]
    assert not {r["vn"] for _, r in kept} & bad           # no kept row of a reserved VN
    assert refused == sum(r["vn"] in bad for r in rows) and len(kept) + refused == len(rows)


def test_pld_sites_parse(tmp_path):
    """build() writes 'address reg offset' for the PLD instructions of a disassembly."""
    import re
    regs = {**{f"r{i}": i for i in range(13)}, "sb": 9, "sl": 10, "fp": 11, "ip": 12, "sp": 13, "lr": 14}
    pat = re.compile(r"\[(\w+)(?:,\s*#(-?\d+))?\]$")
    got = []
    for mn, ops in (("pld", "[r3, #128]"), ("pld", "[ip]"), ("pld", "[r1, r2]"), ("vld1.8", "{d16}, [r1]")):
        m = pat.match(ops) if mn.startswith("pld") else None
        if m and m.group(1) in regs:
            got.append((regs[m.group(1)], int(m.group(2) or 0)))
    assert got == [(3, 128), (12, 0)]
    assert "re.match(r\"\\[(\\w+)(?:,\\s*#(-?\\d+))?\\]$\"" in (VITA / "a9_layers.py").read_text()

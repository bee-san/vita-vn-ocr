"""Tests of the per-character positions (issue #114, docs/char-boxes.md): the twin eval/charbox.py, its C
counterpart runtime/vocr_charbox.c (self-test and parity through the driver, tests/charbox_parity.py), the
committed calibration and protocol pins, and the sets' rules. The C parts compile with the system compiler and
are skipped without one. No screenshots, fonts or models.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "synth"), str(ROOT / "tests")]
import charbox as CB  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
Q = CB.Q


def test_spans_follow_greedy_ctc():
    # a blank separates two equal characters; a run of one class is one character
    assert CB.spans([0, 1, 1, 0, 1, 2, 2, 0, 0, 3]) == [(1, 1, 2), (1, 4, 4), (2, 5, 6), (3, 9, 9)]
    assert CB.spans([0, 0, 0]) == []
    assert CB.spans([]) == []


def test_resized_width_is_the_runtimes():
    # vocr_resized_width: round(w * 32 / h) half to even, clamped to [8, 1024]
    assert CB.resized_width(80, 32) == 80
    assert CB.resized_width(1, 32) == 8
    assert CB.resized_width(5000, 32) == 1024
    assert CB.resized_width(5, 64) == 8
    assert CB.resized_width(3, 4) == 24
    assert CB.resized_width(1, 64) == 8
    assert CB.resized_width(101, 64) == 50          # 50.5 -> 50 (half to even)
    assert CB.resized_width(103, 64) == 52          # 51.5 -> 52


def test_frame_centres_map_through_the_resize():
    # an 80 x 32 crop is read at 80 resized px: frame f's centre is 8 f + 4 px
    out = CB.positions([1, 0, 2, 2, 0, 0, 3, 0, 0, 0], 80, 32, 80)
    assert [o["c"] for o in out] == [4 * Q, 20 * Q, 52 * Q]
    # twice the crop, the same resized line: twice the coordinates
    out2 = CB.positions([1, 0, 2, 2, 0, 0, 3, 0, 0, 0], 160, 64, 80)
    assert [o["c"] for o in out2] == [8 * Q, 40 * Q, 104 * Q]
    # the run's centre
    out3 = CB.positions([1, 0, 2, 2, 0, 0, 3, 0, 0, 0], 80, 32, 80, CB.CENTRE_RUN)
    assert out3[1]["c"] == 24 * Q


def test_cells_tile_the_line():
    rng = np.random.default_rng(0)
    for _ in range(200):
        best = rng.choice([0, 0, 1, 2, 3], size=int(rng.integers(1, 120)))
        w, h = int(rng.integers(8, 1000)), int(rng.integers(8, 60))
        out = CB.positions(best, w, h, CB.resized_width(w, h))
        for a, b in zip(out, out[1:]):
            assert a["x1"] == b["x0"] and a["x0"] <= a["c"] <= a["x1"]
        for o in out:
            assert 0 <= o["x0"] <= o["x1"] <= w * Q


def test_calibration_shift_and_groups():
    groups = {1: CB.G_OTHER, 2: CB.G_CLOSE}
    cal = [-16, 0, 32, 0, 0, 0, 0, 0, 0]          # shift -1 px, close +2 px (resized px x 16)
    out = CB.positions([1, 0, 2, 0], 32, 32, 32, 0, None, cal, groups)
    assert out[0]["c"] == 4 * Q - 16 and out[1]["c"] == 20 * Q - 16 + 32
    clipped = CB.positions([1, 0], 32, 32, 32, 0, None, [-32768] + [0] * 8, groups)
    assert clipped[0]["c"] == 0


def test_groups():
    assert CB.group("、") == CB.G_CLOSE and CB.group("」") == CB.G_CLOSE
    assert CB.group("「") == CB.G_OPEN and CB.group("…") == CB.G_DOTS
    assert CB.group("っ") == CB.G_SMALL and CB.group("ー") == CB.G_LONG
    assert CB.group("！") == CB.G_MARK and CB.group("A") == CB.G_HALF and CB.group("漢") == CB.G_OTHER
    assert CB.group(0x3002) == CB.G_CLOSE and CB.group("") == CB.G_OTHER


def test_ink_moves_cuts_to_gaps_and_boxes_to_ink():
    grey = np.zeros((32, 40), np.uint8)
    grey[10, 3:15] = grey[10, 22:36] = 200
    grey[11, 3:15] = grey[11, 22:36] = 100
    prof = CB.ink_profile(grey)
    assert prof[5] == 400 and prof[0] == 0 and (CB.ink_profile(255 - grey) == prof).all()
    out = CB.positions([0, 1, 0, 2, 0], 40, 32, 40, CB.INK, prof)
    assert out[0]["x1"] == out[1]["x0"] == 296                 # the middle of the gap 15..21
    assert (out[0]["i0"], out[0]["i1"], out[0]["c"]) == (4 * Q, 15 * Q, 152)
    assert (out[1]["i0"], out[1]["i1"], out[1]["c"]) == (22 * Q, 36 * Q, 464)


def test_pitch_snaps_a_regular_line_only():
    best = [0] * 64
    for k, f in enumerate([1, 5, 8, 12, 15, 19]):              # frame spacings 4 / 3: within 25% of 4
        best[f] = k % 3 + 1
    out = CB.positions(best, 512, 32, 512, CB.PITCH)
    d = [b["c"] - a["c"] for a, b in zip(out, out[1:])]
    assert max(d) - min(d) <= 1                                 # one grid after the snap
    irregular = [0] * 64
    for k, f in enumerate([1, 2, 8, 20, 22]):
        irregular[f] = k % 3 + 1
    raw = CB.positions(irregular, 512, 32, 512)
    assert CB.positions(irregular, 512, 32, 512, CB.PITCH) == raw


def test_align_map():
    assert CB.align_map("abcde", "abxde") == [0, 1, 2, 3, 4]
    assert CB.align_map("abcde", "abxcde") == [0, 1, -1, 2, 3, 4]
    assert CB.align_map("abcde", "abde") == [0, 1, 3, 4]
    assert CB.align_map("", "ab") == [-1, -1]
    assert CB.align_map("ab", "") == []
    # the beam + LM text of a line: a substitution and an inserted long-vowel mark
    assert CB.align_map("そうだね", "そーだね") == [0, 1, 2, 3]
    assert CB.align_map("そうだね", "そうーだね") == [0, 1, -1, 2, 3]


def test_to_screen_horizontal_and_vertical():
    out = CB.positions([1, 0, 2, 0], 32, 32, 32)
    hz = CB.to_screen(out, (100, 200, 32, 32))
    assert hz[0][1] == (104.0, 216.0) and hz[0][0][1] == 200 and hz[0][0][3] == 232
    vt = CB.to_screen(out, (300, 50, 20, 32), vertical=True)
    assert vt[0][1] == (310.0, 54.0) and vt[0][0][0] == 300 and vt[0][0][2] == 320


def test_committed_calibration_and_protocol():
    cal = json.loads((ROOT / "eval" / "charboxes_cal.json").read_text(encoding="utf-8"))
    proto = json.loads((ROOT / "eval" / "charboxes_protocol.json").read_text(encoding="utf-8"))
    for t in ("S", "N", "L", "V"):
        v = cal["models"][t]["cal"]
        assert len(v) == CB.N_GROUPS + 1 and all(-32768 <= x <= 32767 for x in v)
        assert proto["calibration"][t] == v
    assert proto["bar"]["model"] == "N" and proto["bar"]["arm"] == "cal+ink"
    assert proto["bar"]["parts"] == ["vita", "psp", "adrenaline"]


def test_protocol_pins():
    import charbox_protocol as P
    assert P.check() == []


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("charbox") / "vocr_charbox_test"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    *(str(ROOT / "runtime" / f) for f in ("vocr.c", "vocr_charbox.c", "vocr_charbox_test.c")),
                    "-lm", "-o", str(out)], check=True)
    return out


def test_c_selftest(driver):
    r = subprocess.run([str(driver)], capture_output=True, text=True, check=True)
    assert "CHARBOX_SELFTEST_OK" in r.stdout


def test_c_parity(driver):
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "charbox_parity.py"), "--bin", str(driver),
                        "--lines", "400", "--aligns", "400", "--seed", "11"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CHARBOX_PARITY OK" in r.stdout

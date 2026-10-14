"""Tests of the post-finder line refinement: runtime/vocr_lines_refine.c and its twin synth/lines_refine.py.

The twin's behaviour on drawn regions (each step on the case it is for, and flags 0 as the identity), and the C
driver against the twin on procedural regions (tests/lines_refine_parity.py, both the in-memory and the
row-callback forms). The C parts compile with the system compiler (cc, or $CC) and are skipped without one.
No screenshots, fonts or models.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "synth"), str(ROOT / "tests")]
import lines_refine as R  # noqa: E402
import lines_refine_parity  # noqa: E402
from common import find_lines_ex  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("refine") / "vocr_lines_refine_test"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    *(str(ROOT / "runtime" / f) for f in ("vocr_lines.c", "vocr_lines_refine.c",
                                                          "vocr_lines_refine_test.c")), "-o", str(out)], check=True)
    return out


def glyph_row(img, top, height, x0, x1, ink, pitch=None):
    """A row of dense block glyphs, `height` rows tall, between columns x0 and x1."""
    pitch = pitch or height + 4
    cells = height
    for x in range(x0, x1 - cells + 1, pitch):
        g = np.zeros((height, cells), bool)
        g[:, ::3] = True
        g[::4, :] = True
        g[0, :] = g[-1, :] = True
        img[top:top + height, x:x + cells][g] = ink
    return img


def region(h=60, w=300, paper=40):
    return np.full((h, w), paper, np.uint8)


def refined(img, flags=R.DEFAULT):
    b, d, _ = find_lines_ex(img, "grad")
    return b, R.refine_lines(img, b, d, flags)


def test_flags_zero_is_the_identity():
    img = glyph_row(region(), 20, 16, 10, 280, 230)
    b, d, _ = find_lines_ex(img, "grad")
    assert R.refine_lines(img, b, d, 0) == (b, d, list(range(len(b))))


def test_lines_only_shrink_keep_order_and_dark():
    for img in lines_refine_parity.make_cases(4, 60):
        b, d, _ = find_lines_ex(img, "grad")
        for flags in (R.DEFAULT, R.TIGHTEN, R.SPLIT, R.CURSOR | R.PORTRAIT):
            rb, rd, org = R.refine_lines(img, b, d, flags)
            assert org == sorted(org) and set(org) == set(range(len(b))) and len(rb) <= 16
            for (x, y, w, h), dk, o in zip(rb, rd, org):
                bx, by, bw, bh = b[o]
                assert w > 0 and h > 0 and dk == d[o]
                assert bx <= x and by <= y and x + w <= bx + bw and y + h <= by + bh


def test_tighten_drops_a_frame_line_and_scenery_rows():
    img = region(70, 320)
    img[8, 5:315] = 220                      # a box border across the region
    img[10:14, 5:315] = 120                  # a stripe of scenery
    glyph_row(img, 30, 16, 12, 300, 230)
    b, (rb, _, _) = refined(img, R.TIGHTEN | R.FRAMES)
    assert len(b) == 1 and len(rb) == 1
    x, y, w, h = rb[0]
    assert y >= 15 and y + h <= 30 + 16 + 4    # the text rows plus the 16% margin, no border
    assert y <= 30 - 2                        # the margin above the ink is kept


def test_split_separates_two_close_rows_but_not_furigana():
    img = region(60, 320)
    glyph_row(img, 10, 16, 12, 300, 230)
    glyph_row(img, 28, 16, 12, 200, 230)      # a second line 2 px below: one GRAD box
    b, (rb, _, org) = refined(img)
    assert len(b) == 1 and len(rb) == 2 and org == [0, 0]
    assert rb[0][1] + rb[0][3] <= rb[1][1] + 1
    img = region(60, 320)
    glyph_row(img, 12, 7, 40, 120, 230, pitch=9)     # furigana: glyphs under half the line's height
    glyph_row(img, 22, 16, 12, 300, 230)
    b, (rb, _, _) = refined(img)
    assert len(rb) == len(b) == 1


def test_cursor_after_the_line_is_cut():
    img = region(50, 360)
    glyph_row(img, 16, 16, 10, 230, 230)      # glyphs end at column 226
    img[18:30, 254:272] = 230                 # a solid blob 28 px (1.75 line heights) after them: the cursor
    b, (rb, _, _) = refined(img, R.TIGHTEN | R.CURSOR)
    assert len(b) == 1 and b[0][0] + b[0][2] > 272   # the finder keeps it in the line
    assert 226 < rb[0][0] + rb[0][2] < 254
    b, (rb, _, _) = refined(img, R.TIGHTEN)
    assert rb[0][0] + rb[0][2] == b[0][0] + b[0][2]


def test_portrait_edge_at_the_start_is_cut():
    img = region(60, 360)
    img[:, 4:40] = 200                        # a tall bright slab: a portrait's edge crossing the line
    img[::3, 4:40] = 90
    glyph_row(img, 22, 16, 80, 340, 230)
    b, (rb, _, _) = refined(img, R.TIGHTEN | R.PORTRAIT)
    assert rb[0][0] > 40 > b[0][0]
    b, (rb, _, _) = refined(img, R.TIGHTEN)
    assert rb[0][0] == b[0][0]                # without the flag the columns are kept


def test_parse_flags():
    assert R.parse_flags("default") == R.DEFAULT == R.ALL
    assert R.parse_flags("tighten+split") == R.TIGHTEN | R.SPLIT
    assert R.parse_flags("none") == 0 and R.parse_flags("0x3") == 3


def test_bad_lines_raise():
    img = region()
    with pytest.raises(ValueError):
        R.refine_lines(img, [(0, 0, 400, 10)], [0])
    with pytest.raises(ValueError):
        R.refine_lines(img, [(0, 0, 10, 10)], [0], 0x40)


def test_c_selftest(driver):
    out = subprocess.run([str(driver), "--selftest", "40"], check=True, capture_output=True, text=True).stdout
    assert out.startswith("REFINE_SELFTEST_OK")


@pytest.mark.parametrize("rows", [False, True])
def test_c_matches_twin(driver, rows):
    rng = np.random.default_rng(13)
    imgs = lines_refine_parity.make_cases(13, 120)
    cases = [(img, int(lines_refine_parity.FLAG_SETS[int(rng.integers(0, len(lines_refine_parity.FLAG_SETS)))]))
             for img in imgs]
    assert lines_refine_parity.run(driver, cases, seed=13, rows=rows) == []


def test_c_arena_fits_in_the_finders(driver):
    for w, h in ((960, 544), (480, 272), (8, 8), (1, 3000), (3000, 1)):
        refine, finder = map(int, subprocess.run([str(driver), "--arena", str(w), str(h)], check=True,
                                                 capture_output=True, text=True).stdout.split())
        assert 0 < refine <= finder


def test_frozen_sources_match_the_protocol():
    """The committed results are for this code: a changed refinement needs a new protocol and new results."""
    import hashlib
    import json
    proto = json.loads((ROOT / "eval" / "finder_tighten_protocol.json").read_text())
    for f in ("runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h", "synth/lines_refine.py"):
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == proto["refinement"]["source_sha256"][f], f
    assert proto["refinement"]["flags_default"] == R.DEFAULT


def test_committed_results_hold_no_text():
    import json
    forbidden = {"text", "pred", "preds", "ref", "reference_text", "label", "lens", "readings"}

    def walk(v, where):
        if isinstance(v, dict):
            for k, x in v.items():
                assert k not in forbidden, f"{where}.{k}"
                walk(x, f"{where}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{where}[{i}]")
    for name in ("dev", "protected", "clannad", "a9"):
        walk(json.loads((ROOT / "eval" / f"finder_tighten_{name}_20261014.json").read_text()), name)

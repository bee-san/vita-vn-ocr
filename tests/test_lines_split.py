"""Tests of the stroke split: runtime/vocr_lines_split.c and its twin synth/lines_split.py, and the translucent-box
classifier (eval/translucent_box.py, issue #98).

The twin's behaviour on drawn regions (two lines over a busy scene become two lines, one large line or a line with
furigana stays one, flags 0 is the identity, boxes only shrink and keep their columns and flags), and the C driver
against the twin on procedural regions (tests/lines_split_parity.py, both forms). The C parts compile with the
system compiler (cc, or $CC) and are skipped without one. No screenshots, fonts or models.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "synth"), str(ROOT / "tests"), str(ROOT / "eval")]
import lines_split as S  # noqa: E402
import lines_split_parity  # noqa: E402
from common import find_lines_ex  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
PROTOCOL = ROOT / "eval" / "translucent_box_protocol.json"


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("split") / "vocr_lines_split_test"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    *(str(ROOT / "runtime" / f) for f in ("vocr_lines.c", "vocr_lines_refine.c", "vocr_lines_split.c",
                                                          "vocr_lines_split_test.c")), "-o", str(out)], check=True)
    return out


def glyph_row(img, top, height, x0, x1, ink, outline=None, pitch=None):
    """A row of block glyphs (vertical strokes 2 px wide with gaps, top and bottom bars), optionally outlined."""
    pitch = pitch or height + 4
    for x in range(x0, x1 - height + 1, pitch):
        g = np.zeros((height, height), bool)
        g[:, 1::4] = True
        g[:, 2::4] = True
        g[0, :] = g[-1, :] = True
        cell = img[top:top + height, x:x + height]
        if outline is not None:
            ring = np.zeros_like(g)
            ring[:, :-1] |= g[:, 1:]
            ring[:, 1:] |= g[:, :-1]
            cell[ring & ~g] = outline
        cell[g] = ink
    return img


def scene(h, w, seed=3, n=100, base=190, ink=80):
    """Line art behind a translucent box: short thin strokes of every direction, at a contrast the GRAD finder
    takes for text (above its threshold) but below the outlined glyphs'."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), base, np.int32)
    for _ in range(n):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        length, ang = int(rng.integers(6, 40)), rng.random() * np.pi
        for t in range(length):
            x, y = int(x0 + t * np.cos(ang)), int(y0 + t * np.sin(ang))
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = ink
    return img.astype(np.uint8)


def split(img, flags=S.DEFAULT):
    b, d, _ = find_lines_ex(img, "grad")
    return b, S.split_lines(img, b, d, flags)


def test_flags_zero_is_the_identity():
    img = glyph_row(scene(60, 320), 10, 16, 8, 300, 255, outline=30)
    b, d, _ = find_lines_ex(img, "grad")
    assert S.split_lines(img, b, d, 0) == (b, d, list(range(len(b))))


def test_two_outlined_lines_over_a_busy_scene_become_two():
    img = scene(64, 330)
    glyph_row(img, 10, 16, 8, 320, 255, outline=30)
    glyph_row(img, 32, 16, 8, 200, 255, outline=30)          # a shorter second line, 6 rows below
    b, d, _ = find_lines_ex(img, "grad")
    sb, sd, org = S.split_lines(img, b, d)
    assert len(b) == 1                                        # GRAD merges them through the scene
    assert len(sb) == 2 and org == [0, 0] and sd == [d[0], d[0]]
    (x0, y0, w0, h0), (x1, y1, w1, h1) = sb
    assert (x0, w0) == (x1, w1) == (b[0][0], b[0][2])         # the box's columns
    assert y0 + h0 <= y1 and y0 <= 10 and y0 + h0 >= 10 + 16 and y1 <= 32 and y1 + h1 >= 32 + 16


def test_one_large_line_and_furigana_stay_one():
    img = np.full((70, 320), 30, np.uint8)
    glyph_row(img, 8, 48, 8, 310, 230, pitch=56)              # one tall line (large text)
    b, (sb, _, _) = split(img)
    assert sb == b
    img = np.full((50, 320), 30, np.uint8)
    glyph_row(img, 8, 7, 40, 150, 230, pitch=9)               # furigana over a line
    glyph_row(img, 18, 18, 8, 310, 230)
    b, (sb, _, _) = split(img)
    assert sb == b


def test_lines_only_shrink_keep_columns_order_and_dark():
    rng = np.random.default_rng(5)
    imgs = lines_split_parity.make_cases(5, 40) + [lines_split_parity.busy_region(rng) for _ in range(60)]
    for img in imgs:
        b, d, _ = find_lines_ex(img, "grad")
        sb, sd, org = S.split_lines(img, b, d)
        assert org == sorted(org) and set(org) == set(range(len(b))) and len(sb) <= 16
        for (x, y, w, h), dk, o in zip(sb, sd, org):
            bx, by, bw, bh = b[o]
            assert (x, w, dk) == (bx, bw, d[o]) and h > 0 and by <= y and y + h <= by + bh


def test_per_box_on_a_strip_equals_the_whole_set():
    """A caller can split each box on the strip it already holds (box made strip-relative)."""
    rng = np.random.default_rng(6)
    for img in [lines_split_parity.busy_region(rng) for _ in range(40)]:
        b, d, _ = find_lines_ex(img, "grad")
        whole = S.split_lines(img, b, d)
        per = ([], [], [])
        for i, ((x, y, w, h), dk) in enumerate(zip(b, d)):
            sb, sd, _ = S.split_lines(img[y:y + h], [(x, 0, w, h)], [dk])
            for (xx, yy, ww, hh), ddk in zip(sb, sd):
                per[0].append((xx, yy + y, ww, hh))
                per[1].append(ddk)
                per[2].append(i)
        if len(per[0]) <= 16:
            assert per == whole


def test_parse_flags_and_bad_lines():
    assert S.parse_flags("default") == S.DEFAULT == S.ALL == S.STROKE
    assert S.parse_flags("off") == 0 and S.parse_flags("1") == 1
    with pytest.raises(ValueError):
        S.split_lines(np.zeros((20, 20), np.uint8), [(0, 0, 30, 10)], [0])
    with pytest.raises(ValueError):
        S.split_lines(np.zeros((20, 20), np.uint8), [(0, 0, 10, 10)], [0], 0x2)


def test_c_selftest(driver):
    out = subprocess.run([str(driver), "--selftest", "60"], check=True, capture_output=True, text=True).stdout
    assert out.startswith("SPLIT_SELFTEST_OK")


@pytest.mark.parametrize("rows", [False, True])
def test_c_matches_twin(driver, rows):
    rng = np.random.default_rng(17)
    imgs = lines_split_parity.make_cases(17, 60) + [lines_split_parity.busy_region(rng) for _ in range(80)]
    cases = [(img, lines_split_parity.FLAG_SETS[int(rng.integers(0, len(lines_split_parity.FLAG_SETS)))])
             for img in imgs]
    bad, _ = lines_split_parity.run(driver, cases, seed=17, rows=rows)
    assert bad == []


@pytest.mark.parametrize("rows", [False, True])
def test_c_matches_twin_where_it_splits(driver, rows):
    """The splitting branch itself: merged-line regions, most of which the split acts on."""
    import lines_split_merged
    cases = lines_split_merged.cases(19, 60)
    bad, splits = lines_split_parity.run(driver, cases, seed=19, rows=rows)
    assert bad == [] and splits >= 20


def test_c_arena_fits_in_the_finders(driver):
    for w, h in ((960, 544), (480, 272), (8, 8), (1, 3000), (3000, 1), (960, 16)):
        split_b, finder = map(int, subprocess.run([str(driver), "--arena", str(w), str(h)], check=True,
                                                  capture_output=True, text=True).stdout.split())
        assert 0 < split_b <= finder


def test_classifier_tags_a_busy_background_not_a_plain_box():
    import translucent_box as TB
    plain = np.full((60, 300), 40, np.uint8)
    glyph_row(plain, 22, 16, 10, 290, 230)
    busy = scene(60, 300, n=40, base=170, ink=60)
    glyph_row(busy, 22, 16, 10, 290, 250, outline=10)
    ink = [(10, 22, 290, 38)]
    assert not TB.busyness(plain, ink, "vita")["tagged"]
    assert TB.busyness(busy, ink, "vita")["tagged"]


def test_no_module_constant_is_bound_twice():
    """eval/translucent_box.py once bound SOURCES twice, so the tag sources were shadowed by the split's files."""
    import ast
    tree = ast.parse((ROOT / "eval" / "translucent_box.py").read_text())
    names = [t.id for node in tree.body if isinstance(node, ast.Assign) for t in node.targets
             if isinstance(t, ast.Name) and t.id.isupper()]
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})


def test_prevalence_record_covers_every_source():
    import translucent_box as TB
    rec = json.loads((ROOT / "eval" / "translucent_box_prevalence_20261018.json").read_text())
    assert set(rec["sources"]) == set(TB.SOURCES) and rec["sources"]["wd"]["tagged"] > 0


FINDER_FILES = ("runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.c",
                "runtime/vocr_lines_rows.h", "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h",
                "synth/lines_refine.py")


def test_frozen_sources_match_the_protocol():
    """The committed results are for this code: a changed split needs a new protocol and new results. Pinned: the
    split's own files and the finder and refinement it runs between (which tests/test_textdet.py and
    tests/test_lines_refine.py pin too). The protocol also records the harness files it was scored with
    (runtime/vocr.c, synth/common.py, eval/textdet.py, ...); those may change with other work, so they are not
    pinned here."""
    if not PROTOCOL.exists():
        pytest.skip("protocol not written yet")
    proto = json.loads(PROTOCOL.read_text())
    for f, digest in proto["sources_sha256"].items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == digest, f
    for f in FINDER_FILES:
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == proto["unchanged_sha256"][f], f
    assert proto["candidate"]["split_flags"] == S.DEFAULT


def test_ridge_reach_is_three_in_c_and_twin():
    """The C code takes a pixel's left and right minima from the 3-pixel window centred S_R - 1 away, which equals
    the twin's reach-S_R minima only for S_R = 3 (audit n9; the frozen C source is not edited for it): both must say
    3, so a changed reach fails here before it can make the two disagree."""
    import re
    c = (ROOT / "runtime" / "vocr_lines_split.c").read_text()
    assert S.S_R == 3
    assert re.search(r"^#define S_R 3u\b", c, re.M)


def test_committed_records_hold_no_text():
    forbidden = {"text", "pred", "preds", "ref", "reference_text", "label", "labels", "lines_text", "readings"}

    def walk(v, where):
        if isinstance(v, dict):
            for k, x in v.items():
                assert k not in forbidden, f"{where}.{k}"
                walk(x, f"{where}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{where}[{i}]")
    for p in sorted((ROOT / "eval").glob("translucent_box_*.json")):
        walk(json.loads(p.read_text()), p.name)

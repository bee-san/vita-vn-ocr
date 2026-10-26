"""runtime/vocr_lines_fix.c (issue #129) through eval/pagepath.py's driver, and the committed results.

The page-path driver built with -DPAGEPATH_FIX (`--finder rubi,fix=N`) on procedural regions:
- fix 0 gives the frozen finder's boxes;
- VOCR_FIX_COLGAP keeps a line whole across a wide gap;
- VOCR_FIX_JUNK drops a frame strip on the region's top edge;
- the default (COLGAP | JUNK) leaves a plain region alone.

It also checks that eval/finder_fix_20261026.json's decisions follow #129's pre-registered rule. The frozen finder
files keep their pinned sha256. Procedural regions only: no screenshots, fonts or models.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "synth"))
import pagepath as PP  # noqa: E402

CC = shutil.which("gcc") or shutil.which("cc")
PAPER, INK = 24, 220
FROZEN = {"runtime/vocr_lines.c": "90c30a1c1e1707b818c2bd919243ee589700ff44ceac8b8d7a842b9d55d8d9ca",
          "runtime/vocr_lines_rows.c": "3207edb85d1cd7c79899390bb2ee028839dd120ca5d2278a28edd5fa9ca2ee6e"}


def glyphs(g, x0, y0, n, size):
    """n glyph-like blocks (vertical strokes every 4 px, a stroke row every 6), pitch size + size / 5."""
    pitch = size + size // 5
    for i in range(n):
        for y in range(size):
            for x in range(size):
                if x % 4 < 2 or y % 6 == 0 or y + 1 == size:
                    g[y0 + y, x0 + i * pitch + x] = INK


def test_frozen_finder_files_unchanged():
    for rel, want in FROZEN.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want, rel


def test_default_flags():
    h = (ROOT / "runtime" / "vocr_lines_fix.h").read_text(encoding="utf-8")
    flags = {k: int(v, 16) for k, v in re.findall(r"#define (VOCR_FIX_[A-Z]+) (0x[0-9a-f]+)u", h)}
    assert flags == {"VOCR_FIX_COLGAP": 1, "VOCR_FIX_VALLEY": 2, "VOCR_FIX_JUNK": 4}
    assert "#define VOCR_FIX_DEFAULT (VOCR_FIX_COLGAP | VOCR_FIX_JUNK)" in h


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("drv")
    frozen, _ = PP.build_driver(PP.parse_finder("rubi"), d, cc=CC)
    fixed, meta = PP.build_driver(PP.parse_finder("rubi,fix=0"), d, cc=CC)
    assert "runtime/vocr_lines_fix.c" in meta["sources"] and frozen != fixed
    return frozen, fixed


def run(exe, g, fix, refine=0):
    return PP.run_driver(exe, [(g, PP.FINDER_MODES["textbox_recovery"], fix, 1, refine)])[0]


def test_fix0_is_the_frozen_finder(drivers):
    frozen, fixed = drivers
    rng = np.random.default_rng(129)
    for k in range(40):
        g = np.full((int(rng.integers(30, 160)), int(rng.integers(40, 640))), PAPER, np.uint8)
        for _ in range(int(rng.integers(0, 4))):
            n, size = int(rng.integers(1, 20)), int(rng.integers(10, 26))
            x0 = int(rng.integers(0, max(1, g.shape[1] - n * (size + size // 5))))
            y0 = int(rng.integers(0, max(1, g.shape[0] - size)))
            if x0 + n * (size + size // 5) <= g.shape[1] and y0 + size <= g.shape[0]:
                glyphs(g, x0, y0, n, size)
        if k % 3 == 0:
            g = 255 - g                                     # dark text on a light box
        for refine in (0, PP.REFINE_ALL):
            assert run(fixed, g, 0, refine) == run(frozen, g, 0, refine), k


def test_colgap_keeps_a_line_across_a_wide_gap(drivers):
    frozen, fixed = drivers
    g = np.full((120, 640), PAPER, np.uint8)
    glyphs(g, 20, 40, 6, 20)
    glyphs(g, 20 + 6 * 24 + 60, 40, 3, 20)                  # a gap of 3 glyph widths
    end = 20 + 6 * 24 + 60 + 2 * 24 + 20
    rc, found, probed, boxes = run(frozen, g, 0)
    assert rc == 0 and not any(b[0] <= 20 and b[0] + b[2] >= end for b in boxes)
    rc, found, probed, boxes = run(fixed, g, 1)
    assert rc == 0 and len(boxes) == 1 and boxes[0][0] <= 20 and boxes[0][0] + boxes[0][2] >= end


def test_junk_drops_an_edge_strip(drivers):
    frozen, fixed = drivers
    g = np.full((120, 640), PAPER, np.uint8)
    for y in range(12):
        for x in range(640):
            if x % 8 < 3 or y % 5 == 0:
                g[y, x] = INK
    glyphs(g, 20, 40, 20, 20)
    rc, found, probed, boxes = run(frozen, g, 0)
    assert rc == 0 and len(boxes) == 2 and boxes[0][1] == 0
    rc, found, probed, boxes = run(fixed, g, 4)
    assert rc == 0 and len(boxes) == 1 and boxes[0][1] > 20
    assert run(fixed, g, 5)[3] == boxes                       # the default drops it too


def test_default_leaves_a_plain_region_alone(drivers):
    frozen, fixed = drivers
    g = np.full((120, 640), PAPER, np.uint8)
    glyphs(g, 20, 20, 20, 20)
    glyphs(g, 20, 60, 12, 20)
    for refine in (0, PP.REFINE_ALL):
        assert run(fixed, g, 5, refine) == run(frozen, g, 0, refine)


def test_results_follow_the_preregistered_rule():
    d = json.loads((ROOT / "eval" / "finder_fix_20261026.json").read_text(encoding="utf-8"))
    st = d["steps"]

    def adopted(arm, base):
        hv, dv = st[f"{arm}_vs_{base}_hval"]["all"], st[f"{arm}_vs_{base}_dev"]["all"]
        return hv["ci95"][1] < 0 and dv["ci95"][1] <= 0.05

    # COLGAP over Rubi today; VALLEY (with or without JUNK) not over COLGAP; COLGAP + JUNK over COLGAP
    assert adopted("F1", "base") and not adopted("F12", "F1") and not adopted("F123", "F1")
    assert adopted("F13", "F1")
    r = d["runs"]
    assert (r["base_hval"]["cer"], r["F13_hval"]["cer"]) == pytest.approx((3.875, 1.898), abs=6e-4)
    assert (r["base_dev"]["cer"], r["F13_dev"]["cer"]) == pytest.approx((3.036, 2.325), abs=6e-4)
    for k in ("base_hval", "F13_hval", "base_dev", "F13_dev"):
        assert r[k]["errors"] / r[k]["chars"] * 100 == pytest.approx(r[k]["cer"], abs=1e-4)
    # the result file holds counts and hashes only: no corpus text
    assert not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", json.dumps(d, ensure_ascii=False))

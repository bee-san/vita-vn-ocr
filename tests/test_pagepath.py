"""eval/pagepath.py (issue #129): the scorer's pieces without game data or models.

The Levenshtein count, Rubi's per-crop composition (cascade, empty lines), the non-Japanese line filter, the line
statistics, the cluster bootstrap, finder specs and units, and eval/pagepath_find.c built against the frozen finder
on procedural regions (Rubi's short-line probe and per-box refinement included)."""
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "synth"))
import pagepath as PP  # noqa: E402
from metrics import lev as slow_lev  # noqa: E402

CC = shutil.which("gcc") or shutil.which("cc")


def test_lev_equals_metrics_lev():
    rng = random.Random(0)
    alphabet = "あいうえお「」。、！？…一ーabc"
    for _ in range(600):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        assert PP.lev(a, b) == slow_lev(a, b), (a, b)
    long_a = "".join(rng.choice(alphabet) for _ in range(300))
    long_b = long_a[:100] + "ん" + long_a[140:]
    assert PP.lev(long_a, long_b) == slow_lev(long_a, long_b)
    assert PP.lev("", "") == 0 and PP.lev("abc", "") == 3 and PP.lev("", "ab") == 2


def test_compose_follows_rubi():
    # (greedy, beam, conf_min) of the job's tier, then of the next tier up
    assert PP.compose(("", "x", 0.1), ("y", "Y", 0.9)) == ""            # read as empty: stays empty
    assert PP.compose(("a", "A", 0.95), ("b", "B", 0.9)) == "A"         # confident: its own beam text
    assert PP.compose(("a", "A", 0.5), ("b", "B", 0.9)) == "B"          # doubtful: the next tier's beam text
    assert PP.compose(("a", "A", 0.5), ("", "B", 0.9)) == ""            # ... empty when its greedy text is
    assert PP.compose(("a", "A", 0.5), None) == "A"                     # no next tier (Large)
    assert PP.line_conf("a", 0.6995) == 700 and PP.compose(("a", "A", 0.6995), ("b", "B", 1)) == "A"
    assert PP.line_conf("a", 0.6994) == 699 and PP.compose(("a", "A", 0.6994), ("b", "B", 1)) == "B"
    assert PP.line_conf("", 0.0) == 1000


def test_japanese_filter():
    assert PP.has_japanese("あ") and PP.has_japanese("カ") and PP.has_japanese("漢")
    assert not PP.has_japanese("……") and not PP.has_japanese("「？」") and not PP.has_japanese("1001")


def test_line_stats():
    pts = [[(10, 10), (20, 10), None, (40, 10)], [(10, 30), (20, 30)]]
    status, kinds, _ = PP.line_stats([(0, 0, 50, 20), (0, 22, 50, 20)], pts)
    assert status == ["whole", "whole"] and kinds == ["line", "line"]
    status, kinds, _ = PP.line_stats([(0, 0, 25, 20), (60, 0, 20, 40)], pts)
    assert status == ["cut", "missed"] and kinds == ["line", "junk"]
    status, kinds, _ = PP.line_stats([(0, 0, 50, 45)], pts)
    assert status == ["whole", "whole"] and kinds == ["merged"]


def _rows(errors, chars, clusters):
    return [{"uid": f"u{i}", "errors": e, "chars": c, "cluster": k} for i, (e, c, k) in
            enumerate(zip(errors, chars, clusters))]


def test_cluster_bootstrap():
    base = _rows([5, 3, 2, 7, 1, 0], [50, 40, 30, 60, 20, 10], list("aabbcc"))
    same = PP.cluster_bootstrap(base, base)
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0] and same["clusters"] == 3
    arm = _rows([1, 1, 0, 2, 0, 0], [50, 40, 30, 60, 20, 10], list("aabbcc"))
    r = PP.cluster_bootstrap(arm, base)
    assert r["delta"] < 0 and r["ci95"][1] < 0 and r["clusters_better"] == 3
    with pytest.raises(ValueError):
        PP.cluster_bootstrap(arm[:-1], base)


def test_finder_specs():
    c = PP.parse_finder("rubi")
    assert (c["mode"], c["probe"], c["refine"], c["fix"]) == ("textbox_recovery", 1, "psp", 0)
    assert PP.refine_flags(c, "psp") == PP.REFINE_ALL and PP.refine_flags(c, "vita") == 0
    c = PP.parse_finder("rubi,mode=grad,probe=0,refine=all,fix=0x3")
    assert (c["mode"], c["probe"], c["fix"]) == ("grad", 0, 3) and PP.refine_flags(c, "vita") == PP.REFINE_ALL
    with pytest.raises(SystemExit):
        PP.parse_finder("nope")
    with pytest.raises(SystemExit):
        PP.parse_finder("rubi,colour=1")


def test_hooked_units_from_a_page_pool(tmp_path):
    pool = tmp_path / "pagepool"
    pool.mkdir()
    lines = [{"id": "T:000001:line1", "kind": "line", "text": "「あい」", "box": [20, 30, 60, 20], "box_main": None,
              "pad": 4, "cell_boxes": [[20, 30, 35, 50], [35, 30, 50, 50], [50, 30, 65, 50], [65, 30, 80, 50]]},
             {"id": "T:000001:line2", "kind": "line", "text": "うえ", "box": [20, 52, 30, 20], "box_main": None,
              "pad": 4, "cell_boxes": []}]
    regions = [{"variant": v, "region": [8 - e, 18 - e, 92 + e, 84 + e], "lines": ["T:000001:line1", "T:000001:line2"]}
               for v, e in (("base", 0), ("expanded-4", 4))]
    page = {"title_id": "T", "page": "000001", "split": "hval", "platform": "vita", "image_sha256": "0" * 64,
            "lines": lines, "regions": regions}
    train = dict(page, page="000002", split="train",
                 regions=[dict(g, lines=["T:000002:line1"]) for g in regions],
                 lines=[dict(lines[0], id="T:000002:line1")])
    (pool / "T.jsonl").write_text("\n".join(json.dumps(p) for p in (page, train)) + "\n")
    (pool / "manifest.json").write_text(json.dumps({"titles": {"T": {}}}))
    units, meta = PP.hooked_units(pool, {"hval"})
    assert [u["uid"] for u in units] == ["T:000001:0:base", "T:000001:0:expanded-4"]
    assert units[0]["lines"][0]["cells"][0] == [20.0, 30.0, 35.0, 50.0]
    assert len(units[0]["lines"][1]["cells"]) == 2                     # no cell boxes: one cell per character
    tune, _ = PP.hooked_units(pool, {"tune"}, per_title=5)
    assert {u["set"] for u in tune} == {"tune"} and len(tune) == 2
    cells, boxes = PP.ref_geometry(units[0], 2, (16, 36))
    assert boxes[0] == [24, 24, 120, 40] and cells[0][0] == [24.0, 24.0, 54.0, 64.0]


def _region():
    """A dark region with two rows of bright glyph-like strokes and a short third row."""
    g = np.full((90, 320), 20, np.uint8)
    for row, (y0, n) in enumerate(((10, 12), (34, 12), (58, 2))):
        for i in range(n):
            x0 = 10 + i * 24
            for y in range(y0, y0 + 18):
                for x in range(x0, x0 + 18):
                    if x % 4 < 2 or (y - y0) % 6 == 0 or y == y0 + 17:
                        g[y, x] = 230
    return g


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_driver_on_the_frozen_finder(tmp_path):
    cfg = PP.parse_finder("rubi")
    exe, meta = PP.build_driver(cfg, tmp_path, cc=CC)
    assert "runtime/vocr_lines.c" in meta["sources"]
    g = _region()
    out = PP.run_driver(exe, [(g, PP.FINDER_MODES["grad"], 0, 0, 0), (g, PP.FINDER_MODES["textbox_recovery"], 0, 1, 0),
                              (g, PP.FINDER_MODES["textbox_recovery"], 0, 1, PP.REFINE_ALL)])
    for rc, found, probed, boxes in out:
        assert rc == 0 and found >= 2 and len(boxes) >= 2
        for x, y, w, h, dark in boxes:
            assert x + w <= g.shape[1] and y + h <= g.shape[0] and dark == 0
    # the two long rows are found by GRAD itself
    rc, found, probed, boxes = out[0]
    assert sum(1 for b in boxes if b[2] > 200) == 2

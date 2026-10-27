"""eval/pagepath.py (issue #129): the scorer's pieces without game data or models.

The Levenshtein count, Rubi's per-crop composition (cascade, empty lines), the line filters, the line statistics,
the cluster bootstrap and the checks of `compare`, finder specs and units, external boxes (clipping, page ids), the
LM settings per LM, the cache tag, the page hash check, the textnorm tool's source put in place whole, and
eval/pagepath_find.c built against the frozen finder on procedural regions: Rubi's short-line probe (its boxes
frozen from Rubi's own host tool) and per-box refinement."""
import json
import os
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
    keep = PP.FILTERS["lines+punct"]
    assert keep("あ") and keep("「……」") and keep("……") and keep("。") and not keep("1001") and not keep("abc")
    assert PP.FILTERS["none"]("1001") and PP.FILTERS["lines"] is PP.has_japanese


def test_line_stats():
    pts = [[(10, 10), (20, 10), None, (40, 10)], [(10, 30), (20, 30)]]
    status, kinds, _ = PP.line_stats([(0, 0, 50, 20), (0, 22, 50, 20)], pts)
    assert status == ["whole", "whole"] and kinds == ["line", "line"]
    status, kinds, _ = PP.line_stats([(0, 0, 25, 20), (60, 0, 20, 40)], pts)
    assert status == ["cut", "missed"] and kinds == ["line", "junk"]
    status, kinds, _ = PP.line_stats([(0, 0, 50, 45)], pts)
    assert status == ["whole", "whole"] and kinds == ["merged"]
    status, kinds, _ = PP.line_stats([(0, 0, 50, 20)], [[None, None], [(10, 10)]])   # a line with no ink
    assert status == ["noink", "whole"] and kinds == ["line"]


def _rows(errors, chars, clusters):
    return [{"uid": f"u{i}", "errors": e, "chars": c, "cluster": k} for i, (e, c, k) in
            enumerate(zip(errors, chars, clusters))]


def test_cluster_bootstrap():
    base = _rows([5, 3, 2, 7, 1, 4], [50, 40, 30, 60, 20, 10], list("abcdef"))
    same = PP.cluster_bootstrap(base, base)
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0] and same["clusters"] == 6
    arm = _rows([1, 1, 0, 2, 0, 1], [50, 40, 30, 60, 20, 10], list("abcdef"))
    r = PP.cluster_bootstrap(arm, base)
    assert r["delta"] < 0 and r["ci95"][1] < 0 and r["clusters_better"] == 6
    with pytest.raises(ValueError):
        PP.cluster_bootstrap(arm[:-1], base)
    # under 5 clusters there is no interval (eval/vocr3_report.cluster_bootstrap's rule)
    few = _rows([5, 3, 2, 7, 1, 0], [50, 40, 30, 60, 20, 10], list("aabbcc"))
    assert PP.cluster_bootstrap(few, few)["ci95"] is None


def _run(units, **kw):
    return dict({"units_sha256": "u1", "view": "console", "tier": "N", "lm": {"name": "x.vlm", "sha256": "0"},
                 "finder": {"preset": "rubi"}, "units": units}, **kw)


def test_compare_refuses_different_units():
    us = [{"uid": f"u{i}", "chars": 10, "errors": 1, "cluster": "c", "set": "hval", "platform": "vita"}
          for i in range(3)]
    assert PP.check_pair(_run(us), _run(us)) == {}
    differs = PP.check_pair(_run(us, finder={"preset": "rubi", "fix": 5}, tier="L"), _run(us))
    assert sorted(differs) == ["finder", "tier"] and differs["tier"] == ["N", "L"]      # [base, arm]
    with pytest.raises(SystemExit):
        PP.check_pair(_run(us, units_sha256="u2"), _run(us))                             # other unit files
    with pytest.raises(SystemExit):
        PP.check_pair(_run(us[:2]), _run(us))                                            # --limit / --set differ
    with pytest.raises(SystemExit):
        PP.check_pair(_run([dict(us[0], chars=11)] + us[1:]), _run(us))                  # other references


def test_external_boxes_are_clipped():
    u = {"uid": "T:000001:0:base", "title": "T", "page": "000001", "platform": "vita", "_w": 100, "_h": 50}
    stats = PP.Counter()
    boxes = {u["uid"]: [[-5, 0, 20, 10], [90, 40, 30, 30, 1], [200, 0, 10, 10], [10.4, 5.6, 20.2, 10]]}
    out = PP.external_boxes(u, 1, (0, 0), boxes, None, stats)
    assert out == [(0, 0, 15, 10), (90, 40, 10, 10, 1), (10, 5, 21, 11)]
    assert stats["boxes_clipped"] == 3 and stats["boxes_dropped"] == 1
    # frame boxes: the unit keeps those whose centre is inside it, clipped; a numeric page id is zero-padded
    fb = {("T", PP.page_id(1), "vita"): [[95, 95, 20, 20], [150, 120, 10, 10], [0, 0, 300, 20], [175, 125, 20, 20],
                                         [180, 130, 20, 20]]}
    out = PP.external_boxes(u, 1, (90, 90), None, fb, stats)
    assert out == [(5, 5, 20, 20), (60, 30, 10, 10), (85, 35, 15, 15)]      # centre outside: dropped
    assert PP.page_id(7) == "000007" and PP.page_id("12") == "000012" and PP.page_id("p3") == "p3"
    assert PP.clip_box((1.5, 2.5, 3, 3), 10, 10) == (1, 2, 4, 4) and PP.clip_box((12, 0, 3, 3), 10, 10) is None


def test_lm_settings_follow_rubi(capsys):
    o = PP.lm_settings("/x/vocr_lm_o3_500k.vlm")
    assert (o["alpha"], o["beta"], o["flags"], o["gate"], o["bracket"], o["beam"]) == (500, 3000, 2, 8000, 8000, 8)
    assert PP.lm_settings("vocr_lm_o3_1000k.vlm")["beta"] == 3500 == PP.lm_settings("vocr_lm_o3_4000k.vlm")["beta"]
    assert PP.lm_settings("vocr_lm_o3_2000k.vlm")["beta"] == 3000
    o = PP.lm_settings("game_h64.vlm", alpha=600, beta=2500, flags=1)
    assert (o["alpha"], o["beta"], o["flags"]) == (600, 2500, 1) and "warning" not in capsys.readouterr().err
    o = PP.lm_settings("game_h64.vlm")
    assert o == PP.LM_OPTS and "warning" in capsys.readouterr().err


def test_model_tag_names_the_runtime(tmp_path):
    blob, lm = tmp_path / "b.vocr", tmp_path / "l.vlm"
    blob.write_bytes(b"blob")
    lm.write_bytes(b"lm")
    t1 = PP.model_tag(blob, lm, PP.LM_OPTS, "a" * 64)
    assert t1 != PP.model_tag(blob, lm, PP.LM_OPTS, "b" * 64)                       # another runtime build
    assert t1 != PP.model_tag(blob, lm, dict(PP.LM_OPTS, beta=3000), "a" * 64)      # another decode
    assert t1 == PP.model_tag(blob, lm, dict(PP.LM_OPTS), "a" * 64)


def test_pages_are_checked_against_their_hash(tmp_path):
    cv2 = pytest.importorskip("cv2")
    d = tmp_path / "T" / "pages"
    d.mkdir(parents=True)
    img = np.zeros((8, 12, 3), np.uint8)
    img[2:5, 3:9] = 200
    cv2.imwrite(str(d / "000001.png"), img)
    good = PP.sha256_file(d / "000001.png")
    fr = PP.Frames(tmp_path, "console")
    assert fr.grey("T", "000001", "vita", good).shape == (8, 12)
    assert fr.grey("T", "000001", "psp", good).shape == (8, 12)                     # cached; checked once
    with pytest.raises(SystemExit):
        PP.Frames(tmp_path, "console").grey("T", "000001", "vita", "0" * 64)


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


def _glyph(g, x0, y0, w, h):
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            if x % 4 < 2 or (y - y0) % 6 == 0 or y == y0 + h - 1:
                g[y, x] = 230


def _probe_region():
    """Two rows of glyphs and, under them, a short line of two narrow glyphs that no finder mode keeps."""
    g = np.full((100, 320), 20, np.uint8)
    for y0 in (10, 36):
        for i in range(12):
            _glyph(g, 10 + i * 24, y0, 18, 20)
    _glyph(g, 27, 67, 12, 15)
    _glyph(g, 50, 67, 12, 15)
    return g


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_driver_short_line_probe(tmp_path):
    """Rubi's probe adds the short last line. The expected boxes are Rubi's own: vjo-ocr (Rubi bd130a8's host tool,
    with vita-vn-ocr 817d203's runtime; --refine off / on) on this region as RGBA returned exactly these."""
    exe, _ = PP.build_driver(PP.parse_finder("rubi"), tmp_path, cc=CC)
    g = _probe_region()
    tr = PP.FINDER_MODES["textbox_recovery"]
    out = PP.run_driver(exe, [(g, tr, 0, 1, 0), (g, tr, 0, 1, PP.REFINE_ALL), (g, tr, 0, 0, 0)])
    assert out[0] == (0, 2, 1, [(6, 6, 290, 28, 0), (6, 32, 290, 28, 0), (23, 63, 43, 23, 0)])
    assert out[1] == (0, 2, 1, [(6, 7, 290, 26, 0), (6, 33, 290, 26, 0), (23, 65, 43, 19, 0)])
    assert out[2] == (0, 2, 0, [(6, 6, 290, 28, 0), (6, 32, 290, 28, 0)])                     # probe off


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_rubi_norm_source_is_replaced_whole(tmp_path):
    """rubi_norm_tool puts its generated source in place with a rename, never by rewriting the file under its name:
    another process's compiler may be reading it (audit-vocr6-finder-g g1: the O_TRUNC rewrite broke parallel cold
    builds). Run on a stand-in for Rubi's core/textnorm.c."""
    core = tmp_path / "rubi" / "core"
    core.mkdir(parents=True)
    (core / "textnorm.h").write_text("void vjo_text_normalize(char *s, void *stats);\n")
    (core / "textnorm.c").write_text('#include "textnorm.h"\n'
                                     "void vjo_text_normalize(char *s, void *stats) {\n"
                                     "  (void)stats; for (; *s; s++) if (*s == 'a') *s = 'b'; }\n")
    exe, meta = PP.rubi_norm_tool(tmp_path / "rubi", tmp_path / "bin", cc=CC)
    assert PP.textnorm(exe, ["abc", ""]) == ["bbc", ""] and meta["textnorm_sha256"]
    src = exe.with_name(exe.name + ".c")
    exe.unlink()                                                        # a cold cache again, the source left behind
    with open(src, "rb") as held:                                       # as another process's compiler holds it
        before = held.read()
        exe2, _ = PP.rubi_norm_tool(tmp_path / "rubi", tmp_path / "bin", cc=CC)
        held.seek(0)
        assert held.read() == before                                    # the reader's file was not rewritten
        assert os.stat(src).st_ino != os.fstat(held.fileno()).st_ino       # a new file took its name
    assert exe2 == exe and PP.textnorm(exe2, ["aa"]) == ["bb"]
    assert sorted(p.name for p in (tmp_path / "bin").iterdir()) == sorted([exe.name, src.name])  # no temp left

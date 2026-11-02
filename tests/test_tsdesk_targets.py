"""textspot-desktop's ink-box targets (#173) with #177's gap clip: two tight lines whose glyph cells overlap, which the
clip separates; the clip off (targets_v1's rule) unchanged; the neighbour rule; the build and the compare. No data:
synthetic pages only."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "teacher", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

cv2 = pytest.importorskip("cv2")

import tsdesk_targets as TT  # noqa: E402

INF = math.inf


def _two_lines(h=64, w=320):
    """Two lines of glyph-like ink (1 px strokes every 6 px and full-width bars) whose em cells overlap by 4 px:
    line A's ink on rows 11-29 (a bottom bar on row 29), cells y 8-34; line B's on rows 33-51 (a top bar on row 33),
    cells y 30-56. Each line's cells hold the other's nearest bar."""
    img = np.full((h, w, 3), 40, np.uint8)
    xs = np.arange(w)
    strokes = (xs % 6 == 0) & (xs >= 60) & (xs < 200)
    img[11:30][:, strokes] = 230
    img[20, 60:200] = 230
    img[29, 60:200] = 230
    img[33:52][:, strokes] = 230
    img[33, 60:200] = 230
    img[42, 60:200] = 230
    return img, [50.0, 8.0, 230.0, 34.0], [50.0, 30.0, 230.0, 56.0]


def _overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0


def test_gap_clip_separates_two_tight_lines():
    img, A, B = _two_lines()
    a1, ha = TT.ink_box(img, A)
    b1, hb = TT.ink_box(img, B)
    # targets_v1's rule: each box takes in the other line's nearest bar (A down to row 34, B up from row 29)
    assert (a1, ha) == ([60, 11, 200, 35], "ink") and (b1, hb) == ([60, 29, 200, 52], "ink")
    assert _overlap(a1, b1)
    ta, tb = TT.gap_bounds(A, [B]), TT.gap_bounds(B, [A])
    assert ta == (-INF, 32.0, "clip") and tb == (32.0, INF, "clip")       # the middle of the 30-34 overlap
    a2, _ = TT.ink_box(img, A, across=ta[:2])
    b2, _ = TT.ink_box(img, B, across=tb[:2])
    assert a2 == [60, 11, 200, 30] and b2 == [60, 33, 200, 52]           # each line's own ink rows
    assert not _overlap(a2, b2)


def test_gap_clip_off_and_far_bounds_keep_targets_v1():
    img, A, B = _two_lines()
    for c in (A, B):
        v1 = TT.ink_box(img, c)
        assert TT.ink_box(img, c, across=None) == v1
        assert TT.ink_box(img, c, across=(-INF, INF)) == v1
        assert TT.ink_box(img, c, across=(c[1] - 3.0, c[3] + 3.0)) == v1    # bounds beyond the spill: no change
    # a line with no neighbour: gap_bounds gives no bound and the box is v1's
    t = TT.gap_bounds(A, [[300.0, 30.0, 320.0, 56.0]])                    # no horizontal overlap
    assert t == (-INF, INF, "none")


def test_gap_clip_keeps_the_cells_fallback_inside_the_bounds():
    blank = np.full((64, 320, 3), 40, np.uint8)
    A, B = [50.0, 8.0, 230.0, 34.0], [50.0, 30.0, 230.0, 56.0]
    assert TT.ink_box(blank, A) == ([50, 8, 230, 34], "cells")
    assert TT.ink_box(blank, A, across=TT.gap_bounds(A, [B])[:2]) == ([50, 8, 230, 32], "cells")
    assert TT.ink_box(blank, B, across=TT.gap_bounds(B, [A])[:2]) == ([50, 32, 230, 56], "cells")


def test_gap_bounds_rules():
    c = [100.0, 100.0, 400.0, 130.0]
    # rows that overlap c across: the middle of the overlap; the nearest row above and below wins
    assert TT.gap_bounds(c, [[100.0, 74.0, 300.0, 104.0]]) == (102.0, INF, "clip")
    assert TT.gap_bounds(c, [[100.0, 126.0, 500.0, 156.0]]) == (-INF, 128.0, "clip")
    assert TT.gap_bounds(c, [[100.0, 40.0, 300.0, 70.0], [100.0, 74.0, 300.0, 104.0],
                             [100.0, 126.0, 500.0, 156.0], [100.0, 160.0, 500.0, 190.0]]) == (102.0, 128.0, "clip")
    # a positive gap: half of it, outside c's rows ("none": nothing of c is cut)
    assert TT.gap_bounds(c, [[100.0, 60.0, 300.0, 90.0]]) == (95.0, INF, "none")
    # no bound from: a row beside c (no horizontal overlap), a row sharing c's row (centres nearer than half the
    # smaller height), a row spanning c across (a vertical column, a big box)
    for o in ([401.0, 74.0, 600.0, 104.0], [380.0, 104.0, 600.0, 134.0], [200.0, 60.0, 230.0, 200.0]):
        assert TT.gap_bounds(c, [o]) == (-INF, INF, "none"), o
    # a small ruby row above: its centre is far enough although its height is small
    assert TT.gap_bounds(c, [[120.0, 88.0, 200.0, 104.0]]) == (102.0, INF, "clip")
    # bounds that together would keep less than half of c's height are not applied (one bound alone always keeps
    # more than half: it lies between c's start and its centre)
    pair = [[100.0, 60.0, 300.0, 120.0], [100.0, 106.0, 300.0, 170.0]]          # bounds 110 and 118
    assert TT.gap_bounds(c, pair) == (-INF, INF, "guard")
    assert TT.gap_bounds(c, pair[:1]) == (110.0, INF, "clip")


@pytest.mark.parametrize("bound", [32.0, 32.5, 31.7, 32.49, 32.51])
def test_across_rows_two_lines_share_no_row(bound):
    _, hi_upper = TT.across_rows((-INF, bound), 64)
    lo_lower, _ = TT.across_rows((bound, INF), 64)
    assert hi_upper == lo_lower
    assert TT.across_rows(None, 64) == (0, 64) and TT.across_rows((-INF, INF), 64) == (0, 64)


def _record(tmp_path, img, A, B, split="hval"):
    """A pool record of one page holding the two lines (cell boxes cut from their cells' unions), a ruby row above A
    and an uncovered renderer box at the right."""
    path = tmp_path / "pages" / "T" / "000001.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[:, :, ::-1])

    def cells(c, n=6):
        xs = np.linspace(c[0], c[2], n + 1)
        return [[float(xs[i]), c[1], float(xs[i + 1]), c[3]] for i in range(n)]

    def line(i, c, kind="line", host=None):
        x0, y0, x1, y1 = (int(v) for v in c)
        return {"id": f"T:000001:{kind}{i}", "kind": kind, "vertical": False, "box": [x0, y0, x1 - x0, y1 - y0],
                "box_main": None, "furigana": False, "tall_band": False, "box_index": [i], "pad": 0, "text": "テスト",
                "source": "glyph_cells", "host_line": host, "drop": None, "cell_boxes": cells(c)}

    rec = {"title_id": "T", "page": "000001", "image": "T/000001.png", "dims": [img.shape[1], img.shape[0]],
           "split": split, "uncovered": [9], "boxes": [{"index": 9, "xyxy": [240.0, 30.0, 300.0, 56.0],
                                                        "kind": "line"}],
           "lines": [line(1, A), line(2, B)]}
    return rec, path


def test_page_targets_gap_clip(tmp_path):
    img, A, B = _two_lines()
    rec, path = _record(tmp_path, img, A, B)
    _, _, v1, h1 = TT._page_targets(("T", rec, str(path), False))
    _, _, v2, h2 = TT._page_targets(("T", rec, str(path), True))
    assert v1["old"] == v2["old"]                                       # the trainer's alignment key is unchanged
    assert h1["gap"] == [None, None] and h2["gap"] == ["clip", "clip"] and h2["fix"] == [None, None]
    assert _overlap(*v1["new"]) and not _overlap(*v2["new"])
    assert v2["new"] == [[60, 11, 200, 30], [60, 33, 200, 52]]


def test_build_and_compare(tmp_path):
    import argparse
    img, A, B = _two_lines()
    rec, _ = _record(tmp_path, img, A, B)
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "manifest.json").write_text("{}\n")
    (pool / "T.jsonl").write_text(json.dumps(rec) + "\n")
    args = dict(pool=str(pool), pages=str(tmp_path / "pages"), splits="train,guard,hval", workers=1)
    TT.cmd_build(argparse.Namespace(out=str(tmp_path / "v1"), gap_clip=False, cells_fix=False, **args))
    TT.cmd_build(argparse.Namespace(out=str(tmp_path / "v2"), gap_clip=True, cells_fix=False, **args))
    d1 = json.loads((tmp_path / "v1" / "targets.json").read_text())
    d2 = json.loads((tmp_path / "v2" / "targets.json").read_text())
    assert "gap_clip" not in d1["rule"] and d2["rule"]["gap_clip"]["horizontal_lines"] == {"clip": 2}
    assert d1["lines"] == d2["lines"] == 2
    c1, _ = TT.load_cache(tmp_path / "v1")
    c2, _ = TT.load_cache(tmp_path / "v2")
    res = TT.compare(c1, c2, {("T", "000001"): "hval"}, ("T",))
    for k in ("all", "split:hval", "title:T"):
        g = res[k]
        assert g["lines"] == 2 and g["changed"] == 2 and g["overlap_a"] == 2 and g["overlap_b"] == 0
        assert g["top_down"] == 1 and g["bottom_up"] == 1 and g.get("grew", 0) == 0
        assert g["move_bottom"]["hist"] == {-5: 1} and g["move_top"]["hist"] == {4: 1}
    with pytest.raises(SystemExit):                                     # the test split is never read
        TT.cmd_build(argparse.Namespace(out=str(tmp_path / "x"), gap_clip=True, cells_fix=False,
                                        **dict(args, splits="train,test")))


# ----------------------------------------------------------------------------- targets_v3's cells fix

def test_ring_noise_fix_leaves_sparse_ink_out():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 3, 400).clip(0, None)
    inky = np.concatenate([clean, np.full(100, 150.0)])                   # 20% of the ring through ink
    assert TT.ring_noise(inky, 60.0) == pytest.approx(np.percentile(inky, 90))
    assert TT.ring_noise(inky, 60.0, fix=True) == pytest.approx(np.percentile(clean, 90))
    dither = np.full(400, 20.0)                                          # a dither as strong as the body term
    assert TT.ring_noise(dither, 16.0, fix=True) == TT.ring_noise(dither, 16.0) == 20.0


def _ring_through_ink():
    """A line (strokes 230 on 40, rows 12-28, x 60-200) with a fainter word (grey 160: above the body term, x 210-250),
    and the window's bottom ring row (cells y 8-31, grow 2: row 32) running through a neighbour's ink (grey 180) on
    110 px, a fifth of the ring: the plain noise term (its 90th percentile, 140, + 12) drops the fainter word."""
    img = np.full((40, 320, 3), 40, np.uint8)
    img[12:28, 60:200:3] = 230
    img[20, 60:200] = 230
    img[14:26, 210:250:3] = 160
    img[18, 210:250] = 160
    img[32, 60:170] = 180
    return img


def test_noise_fix_keeps_a_faint_word_the_ring_noise_dropped():
    img = _ring_through_ink()
    c = [50.0, 8.0, 260.0, 31.0]
    v2, _ = TT.ink_box(img, c)
    v3, _ = TT.ink_box(img, c, noise_fix=True)
    assert v2[2] == 200 and v3[2] == 250                                 # the faint word drops, then is kept
    assert v3[:2] == v2[:2] and v3[3] == v2[3]


def test_colour_edges_keep_a_coloured_word():
    img = np.full((40, 320, 3), 100, np.uint8)
    img[12:28, 60:200:3] = 230
    img[20, 60:200] = 230
    img[14:26, 210:250:3] = (220, 40, 40)                                # luma ~ the background's: no grey edge
    img[18, 210:250] = (220, 40, 40)
    c = [50.0, 8.0, 260.0, 32.0]
    assert TT.ink_box(img, c)[0][2] == 200
    assert TT.ink_box(img, c, colour_edges=True)[0][2] == 250
    grey = np.repeat(img.mean(axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
    assert TT.ink_box(grey, c, colour_edges=True) == TT.ink_box(grey, c)    # grey frames: unchanged


def test_glyph_cells_keep_a_small_glyph_the_speck_filter_breaks():
    img = np.full((40, 320, 3), 64, np.uint8)
    img[12:28, 60:200:3] = 255
    img[20, 60:200] = 255
    for y, x in ((22, 210), (22, 214), (24, 209), (24, 215), (26, 212)):  # a small 。 in 2 px pieces
        img[y, x:x + 2] = 255
    c = [50.0, 8.0, 228.0, 32.0]
    v2, _ = TT.ink_box(img, c)
    assert v2[2] == 200
    v3, _ = TT.ink_box(img, c, glyph_cells=[[60.0, 8.0, 200.0, 32.0], [208.0, 8.0, 226.0, 32.0]])
    assert v3[2] == 217 and v3[0] == v2[0]
    # a cell that already holds ink, or a cell with fewer than min_blob raw pixels, is left as it is
    assert TT.ink_box(img, c, glyph_cells=[[60.0, 8.0, 200.0, 32.0]]) == TT.ink_box(img, c)


def test_page_targets_cells_fix_only_touches_cut_lines(tmp_path):
    img = _ring_through_ink()
    path = tmp_path / "pages" / "T" / "000002.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[:, :, ::-1])

    def line(i, c, text, n):
        xs = np.linspace(c[0], c[2], n + 1)
        x0, y0, x1, y1 = (int(v) for v in c)
        return {"id": f"T:000002:line{i}", "kind": "line", "vertical": False, "box": [x0, y0, x1 - x0, y1 - y0],
                "box_main": None, "furigana": False, "tall_band": False, "box_index": [i], "pad": 0, "text": text,
                "source": "glyph_cells", "host_line": None, "drop": None,
                "cell_boxes": [[float(xs[k]), c[1], float(xs[k + 1]), c[3]] for k in range(n)]}
    # the line's 7 cells of 30 px: the last two (x 200-260) hold the fainter word
    rec = {"title_id": "T", "page": "000002", "image": "T/000002.png", "dims": [320, 40], "split": "train",
           "uncovered": [], "boxes": [], "lines": [line(1, [50.0, 8.0, 260.0, 31.0], "あいうえおかき", 7)]}
    _, _, v2, i2 = TT._page_targets(("T", rec, str(path), True, False))
    _, _, v3, i3 = TT._page_targets(("T", rec, str(path), True, True))
    assert v2["old"] == v3["old"] and i2["fix"] == [None] and i3["fix"] == ["fixed"]
    ln = rec["lines"][0]
    assert TT.missed_cells(ln, v2["new"][0]) == [5, 6] and TT.missed_cells(ln, v3["new"][0]) == []
    rec2 = dict(rec, lines=[line(1, [50.0, 8.0, 200.0, 31.0], "あいうえお", 5)])  # nothing cut: v2's box exactly
    _, _, w2, _ = TT._page_targets(("T", rec2, str(path), True, False))
    _, _, w3, j3 = TT._page_targets(("T", rec2, str(path), True, True))
    assert w2 == w3 and j3["fix"] == [None]

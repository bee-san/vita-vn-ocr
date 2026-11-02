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
    assert h1 == [None, None] and h2 == ["clip", "clip"]
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
    TT.cmd_build(argparse.Namespace(out=str(tmp_path / "v1"), gap_clip=False, **args))
    TT.cmd_build(argparse.Namespace(out=str(tmp_path / "v2"), gap_clip=True, **args))
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
        TT.cmd_build(argparse.Namespace(out=str(tmp_path / "x"), gap_clip=True, **dict(args, splits="train,test")))

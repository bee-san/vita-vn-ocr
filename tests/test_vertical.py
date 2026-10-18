"""Tests of the vertical-text tools (issue #99): synth/vertical_twin.py, eval/vertical.py, eval/vertical_data.py's
column geometry, synth/vertical_render.py's typesetting helpers and the frozen protocol. No data or network needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "eval"))

import vertical_twin as V  # noqa: E402


def _eval_vertical():
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_vertical", ROOT / "eval" / "vertical.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _columns_region(n=3, cw=20, pitch=24, chars=8, gap=14, fg=230, bg=20, h=None, seed=0):
    """A region of n vertical columns of glyph-like blocks (each cell two strokes), right to left."""
    rng = np.random.default_rng(seed)
    w = 2 * 12 + n * cw + (n - 1) * gap
    h = h or 2 * 12 + chars * pitch
    g = np.full((h, w), bg, np.uint8)
    boxes = []
    for c in range(n):
        x0 = w - 12 - (c + 1) * cw - c * gap
        for k in range(chars):
            y0 = 12 + k * pitch
            g[y0 + 2:y0 + 4, x0 + 2:x0 + cw - 2] = fg                      # a horizontal stroke
            xs = x0 + 3 + int(rng.integers(0, cw - 8))
            g[y0 + 2:y0 + pitch - 6, xs:xs + 2] = fg                        # a vertical stroke
            g[y0 + pitch - 9:y0 + pitch - 7, x0 + 4:x0 + cw - 4] = fg      # another horizontal
        boxes.append((x0, 12, cw, chars * pitch))
    return g, boxes


def test_rotate_ccw_matches_rot90():
    g = np.arange(6 * 4, dtype=np.uint8).reshape(6, 4)
    r = V.rotate_ccw(g)
    assert r.shape == (4, 6)
    assert np.array_equal(r, np.rot90(g, 1))
    # out[i][j] = g[j][w - 1 - i]: the column's top goes to the left, its right side to the top
    assert r[0, 0] == g[0, 3] and r[3, 5] == g[5, 0]


@pytest.mark.parametrize("variant", [0, 1])
def test_find_columns_reading_order_and_dark(variant):
    g, boxes = _columns_region(n=3)
    found, dark, trunc = V.find_columns(g, variant)
    assert not trunc
    assert len(found) == 3
    # right to left
    rights = [x + w for x, _, w, _ in found]
    assert rights == sorted(rights, reverse=True)
    # each found box covers its column
    for (fx, fy, fw, fh), (x, y, w, h) in zip(found, boxes):
        assert fx <= x + 2 and fx + fw >= x + w - 2 and fy <= y + 4 and fy + fh >= y + h - 10
    assert dark == [0, 0, 0]
    inv, _ = _columns_region(n=3, fg=20, bg=230)
    _, dark2, _ = V.find_columns(inv, variant)
    assert dark2 and all(d == 1 for d in dark2)


def test_find_columns_drops_horizontal_text():
    h, w = 60, 400
    g = np.full((h, w), 10, np.uint8)
    for x in range(20, 380, 26):                    # one horizontal line of glyphs
        g[20:40, x + 2:x + 4] = 240
        g[22:24, x:x + 20] = 240
        g[36:38, x:x + 20] = 240
    found, _, _ = V.find_columns(g)
    assert found == []


def test_find_columns_empty_and_tiny():
    assert V.find_columns(np.zeros((40, 40), np.uint8)) == ([], [], False)
    assert V.find_columns(np.zeros((0, 0), np.uint8)) == ([], [], False)


def test_retile_lays_cells_left_to_right():
    g, boxes = _columns_region(n=1, chars=5)
    x, y, w, h = boxes[0]
    crop = g[y - 4:y + h + 4, x - 4:x + w + 4]
    cells = V.cells(crop)
    assert len(cells) == 5
    strip = V.retile(crop, "a0")
    W = crop.shape[1]
    assert strip.shape[0] == W
    assert strip.shape[1] == 2 + 5 * (W + 2)
    strip1 = V.retile(crop, "a1")
    assert strip1.shape == strip.shape


def test_retile_a1_turns_a_vertical_bar():
    W = 24
    crop = np.full((3 * W, W), 0, np.uint8)
    crop[4:W - 4, 6:18] = 255                        # a square-ish glyph
    crop[W + 3:2 * W - 3, 11:13] = 255               # a tall thin bar: a vertical long-vowel mark
    crop[2 * W + 4:3 * W - 4, 6:18] = 255
    a0, a1 = V.retile(crop, "a0"), V.retile(crop, "a1")
    bar0 = a0[:, 2 + W + 2:2 + 2 * W + 2]
    bar1 = a1[:, 2 + W + 2:2 + 2 * W + 2]
    ys0, xs0 = np.nonzero(bar0 > 128)
    ys1, xs1 = np.nonzero(bar1 > 128)
    assert (ys0.max() - ys0.min()) > (xs0.max() - xs0.min())     # upright in a0
    assert (xs1.max() - xs1.min()) > (ys1.max() - ys1.min())     # turned horizontal in a1


def test_match_and_bootstrap():
    ev = _eval_vertical()
    dets = [(0, 0, 10, 50), (20, 0, 10, 50), (40, 0, 10, 50)]
    refs = [[1, 1, 9, 48], [21, 2, 9, 47]]
    pairs, fd, fr = ev.match(dets, refs, ignore=[[38, 0, 14, 60]])
    assert sorted(pairs) == [(0, 0), (1, 1)]
    assert fd == [] and fr == []                    # the third detection is in an ignore box
    rows_a = [{"id": f"c{i}", "cluster": f"k{i % 3}", "cohort": "vn", "text": "あいうえお", "pred": "あいうえお"}
              for i in range(9)]
    rows_b = [dict(r, pred="あいうえ") for r in rows_a]
    d = ev.cluster_diff(rows_a, rows_b, "vn", n=200)
    assert d["clusters"] == 3 and abs(d["diff"] - 20.0) < 1e-9
    assert d["lo"] == pytest.approx(20.0) and d["hi"] == pytest.approx(20.0)


def test_score_rows_excludes_names_and_skips():
    ev = _eval_vertical()
    rows = [{"cohort": "vn", "cluster": "a", "text": "あい", "pred": "あい"},
            {"cohort": "vn", "cluster": "a", "text": "うえ", "pred": "うお", "name": True},
            {"cohort": "manga", "cluster": "b", "text": "かき", "pred": "か"},
            {"cohort": "vn", "cluster": "a", "text": None, "skip": "ui"}]
    s = ev.score_rows(rows)
    assert s["vn"]["chars"] == 2 and s["vn"]["errs"] == 0
    assert s["manga"]["errs"] == 1 and s["all"]["chars"] == 4


def test_column_box_excludes_furigana():
    import vertical_data as VD
    g = np.full((120, 60), 250, np.uint8)
    g[10:110, 20:22] = 10                           # the main column's strokes (20..38)
    g[10:110, 36:38] = 10
    for y in range(10, 110, 20):
        g[y:y + 2, 20:38] = 10
    for y in range(20, 60, 8):                      # furigana: small marks right of the column
        g[y:y + 4, 41:45] = 10
    ink, crop = VD.column_box(g, [18, 8, 47, 112], 4)
    assert ink[0] >= 19 and ink[2] <= 39            # furigana (x 41..45) left out
    assert ink[1] <= 10 and ink[3] >= 109
    assert crop == [ink[0] - 4, ink[1] - 4, ink[2] + 4, ink[3] + 4]


def test_protocol_is_frozen_and_complete():
    p = json.loads((ROOT / "eval" / "vertical_protocol.json").read_text())
    assert p["kind"] == "vita-vn-ocr-vertical-protocol"
    assert "primary" in p["metric"] and "VN development dialogue CER" in p["metric"]["primary"]
    for k in ("code_only", "adopt_V", "finder", "test", "manga_never_trades_vn"):
        assert k in p["rules"]
    m = json.loads((ROOT / "data" / "manifests" / "vertical_shards_v1.json").read_text())
    assert m["crops"] == sum(s["crops"] for s in m["shards"]) and len(m["shards"]) >= 100
    assert all(len(s["sha256"]) == 64 for s in m["shards"])

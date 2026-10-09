"""synth2 renderer pieces: the mask line finder, kinsoku wrapping, the background exclusions
and asset pins, and (where the local data/ exists) the label invariants of rendered crops."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))

import mask_finder  # noqa: E402
import render  # noqa: E402
from common import find_lines, is_test_vn  # noqa: E402


def test_mask_finder_geometry_and_furigana():
    ink = np.zeros((120, 300), bool)
    ink[10:16, 40:60] = True          # furigana: 6 rows, under 0.55x the line height -> dropped
    ink[20:44, 30:250] = True         # line 1: 24 rows
    ink[60:84, 30:180] = True         # line 2
    boxes = mask_finder.find_lines_mask(ink)
    assert boxes == [(26, 16, 228, 32), (26, 56, 158, 32)]   # 4 px pad left/right/top, 5 below


def test_mask_finder_merges_close_furigana_and_rejects_tall_runs():
    ink = np.zeros((100, 200), bool)
    ink[8:14, 20:40] = True           # furigana 1 px above the line: part of the line's run
    ink[15:39, 10:150] = True
    assert mask_finder.find_lines_mask(ink) == [(6, 4, 148, 39)]
    tall = np.zeros((120, 50), bool)
    tall[5:94, 5:40] = True           # one run of 89 rows (> 88): the region is rejected, as in common.py
    assert mask_finder.find_lines_mask(tall) is None
    tall[93] = False                  # 88 rows: still a line
    assert mask_finder.find_lines_mask(tall) == [(1, 1, 43, 96)]


def test_mask_finder_is_polarity_agnostic_and_the_console_finder_agrees():
    for text, box in ((235, 30), (20, 225)):      # light on dark, dark on light
        img = np.full((60, 200, 3), box, np.uint8)
        img[20:40, 20:180:4] = text                  # glyph-like vertical strokes
        ink = np.zeros((60, 200), bool)
        ink[20:40, 20:180:4] = True
        assert mask_finder.find_lines_mask(ink) == [(16, 16, 165, 28)]
        (x, y, w, h), = find_lines(img, "grad")
        assert abs(x - 16) <= 5 and abs(y - 16) <= 3 and abs(w - 165) <= 8 and abs(h - 28) <= 4


def test_background_exclusions_cover_eval_reserved_games():
    """Every VN eval reserves (eval/manifests/test_games.json "all") is in the committed
    background exclusion manifest; when eval's list grows, re-run synth/bg_exclude.py."""
    games = json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text(encoding="utf-8"))
    path = ROOT / "data" / "manifests" / "synth2_bg_exclude.txt"
    listed = {x.split()[0] for x in path.read_text().splitlines() if x.strip() and x[0] != "#"}
    missing = sorted(set(games["all"]) - listed)
    assert not missing, f"re-run synth/bg_exclude.py; not excluded: {missing[:10]}"


def test_psp_background_manifest_is_training_side():
    games = json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (ROOT / "data" / "manifests" / "synth2_psp_shots.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    assert rows
    for r in rows:
        assert not is_test_vn(r["vn"]) and r["vn"] not in set(games["all"]), r
        assert r["release"] not in set(games["sessions"]), r
        assert re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) and r["url"].startswith("https://t.vndb.org/"), r


def test_fetch_rejects_a_file_that_differs_from_its_pin():
    import fetch_synth2
    fetch_synth2.check("x", "a" * 64, None)                # nothing pinned yet (--refresh)
    fetch_synth2.check("x", "a" * 64, "a" * 64)
    with pytest.raises(SystemExit):
        fetch_synth2.check("x", "a" * 64, "b" * 64)


def test_wrap_kinsoku():
    rng = np.random.default_rng(0)
    text = "「あいうえおかきくけこ」、さしすせそたち……つてとなにぬねの。"
    for cpl in range(4, 14):
        lines = render.wrap(text, cpl, 99, rng)
        assert "".join(lines) == text
        for ln in lines[1:]:
            assert ln[0] not in render.NO_START, (cpl, lines)
        for ln in lines[:-1]:
            assert ln[-1] not in render.NO_END, (cpl, lines)
            assert len(ln) <= cpl + 2                 # at most two hanging characters
        assert not any(a.endswith("…") and b.startswith("…") for a, b in zip(lines, lines[1:]))


@pytest.fixture(scope="module")
def resources():
    if not (render.DATA / "corpus" / "dialogue.txt").exists() or not list((render.DATA / "fonts").glob("*/*.ttf")):
        pytest.skip("needs the local data/ (corpus and fonts)")
    render.init(20_000)
    return render.R


def test_rendered_crops_have_clean_labels(resources):
    rng = np.random.default_rng(1)
    n = 0
    for _ in range(200):
        r = render.render_region(rng)
        if r is None:
            continue
        img, layout, ink, meta = r
        assert img.dtype == np.uint8 and img.shape[:2] == ink.shape
        for g, label, (x, y, w, h), line, dark in render.extract(img, layout, ink, "mask", rng) or []:
            n += 1
            assert g.shape == (h, w) and h >= 8 and w >= 8 and dark in (0, 1)
            assert label and "　" not in label and set(label) <= resources.vocab_set
    assert n > 150

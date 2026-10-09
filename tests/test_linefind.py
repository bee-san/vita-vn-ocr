"""Tests for the dialogue line finder: runtime/vocr_lines.c, its Python twin and the
ground-truth manifests in eval/linefind/manifests.

The C parts compile runtime/vocr_lines.c with the system compiler (cc, or $CC) and
are skipped when no compiler is available.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "eval" / "linefind"))
import build_gt  # noqa: E402  (as a module: its test_games() must not be collected here)

import common  # noqa: E402
import linefind_parity  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("vocr") / "vocr_lines_test"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "runtime" / "vocr_lines.c"), str(ROOT / "runtime" / "vocr_lines_test.c"),
                    "-o", str(out)], check=True)
    return out


def lines_image(ink, paper, outline=None, h=80, w=320, tops=(12, 46), thin=False):
    """Two lines of dense block 'glyphs' (rows 12..31 and 46..65, columns 16..259) on a
    flat box: 18-px cells every 24 px, 3-px strokes every 6 px and three 3-px bars, the
    same pattern as the C selftest, so every glyph row has ink at most 3 px apart.
    `tops` moves the lines (each is 20 rows tall). The 3-px bars leave rows of zero
    contrast inside a line; `thin` draws a 1-px grid every 4 px instead, which has none."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = paper
    yy, xx = np.mgrid[0:20, 0:w - 76]
    cell = xx % 24
    if thin:
        ys, xs = np.nonzero((cell < 18) & ((cell % 4 == 0) | (yy % 4 == 0)))
    else:
        ys, xs = np.nonzero((cell < 18) & ((cell % 6 < 3) | (yy % 8 < 3)))
    xs = xs + 16
    for top in tops:
        if outline is not None:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    img[top + ys + dy, xs + dx] = outline
        img[top + ys, xs] = ink
    return img


def legacy_frozen(rgb):
    """The overlay finder exactly as the seed commit shipped it (synth/common.py)."""
    h_img, w_img = rgb.shape[:2]
    if not w_img or not h_img or w_img > 960 or h_img > 544:
        return None
    bright = (rgb[..., 0] >= 180) & (rgb[..., 1] >= 180) & (rgb[..., 2] >= 180)
    counts = bright.sum(axis=1)
    if (counts > w_img * 3 // 4).any():
        return None
    has = counts > 0
    left = np.where(has, bright.argmax(axis=1), w_img)
    right = np.where(has, w_img - bright[:, ::-1].argmax(axis=1), 0)
    lines, active, start, last = [], False, 0, 0
    for y in range(h_img + 1):
        count = counts[y] if y < h_img else 0
        if count >= 2:
            if not active:
                start, active = y, True
            last = y
        if active and (y == h_img or y - last >= 3):
            h = last - start + 1
            if h >= 8:
                if h > 88 or len(lines) == 8:
                    return None
                x1, x2 = int(left[start:last + 1].min()), int(right[start:last + 1].max())
                if x2 > x1:
                    lx = x1 - 4 if x1 > 4 else 0
                    ly = start - 4 if start > 4 else 0
                    ex = x2 + 4 if x2 + 4 < w_img else w_img
                    ey = last + 5 if last + 5 < h_img else h_img
                    lines.append((lx, ly, ex - lx, ey - ly))
            active = False
    return lines


def test_grad_is_polarity_agnostic():
    light = lines_image(240, 30)
    dark = lines_image(30, 230)
    bl, dl, _ = common.find_lines_ex(light)
    bd, dd, _ = common.find_lines_ex(dark)
    assert len(bl) == 2 and bl == bd
    assert dl == [0, 0] and dd == [1, 1]
    for (x, y, w, h), top in zip(bl, (12, 46)):
        assert y <= top - 3 and y + h >= top + 20 + 3       # whole line plus padding
        assert x <= 16 - 3 and x + w >= 260 + 3
    assert common.find_lines(dark, "legacy") is None        # the old finder rejects light boxes
    assert len(common.find_lines(light, "legacy")) == 2


def test_outlined_light_text_on_light_box_stays_light():
    # White fill, black outline, light box: the dark outline outnumbers nothing 8:1.
    img = lines_image(255, 180, outline=20)
    boxes, dark, _ = common.find_lines_ex(img)
    assert len(boxes) == 2 and dark == [0, 0]
    crops = common.line_crops(img)
    assert all(d == 0 for _, _, d in crops)


def test_closely_set_lines_stay_apart():
    # 3 and 4 px of paper between the lines leave 1 and 2 empty contrast rows: within
    # MERGE, but an empty row between two 22-row runs splits the band (PSP dialogue).
    for gap in (3, 4):
        tops = (12, 32 + gap)
        boxes = common.find_lines(lines_image(240, 30, tops=tops, thin=True), "grad")
        assert len(boxes) == 2, gap
        for (x, y, w, h), top in zip(boxes, tops):
            assert y <= top and y + h >= top + 20               # each box holds its whole line
            assert y + h <= top + 20 + gap + 3 and y >= top - gap - 3
    # 2 px of paper is no gap at all after the 3x3 contrast: one band
    assert len(common.find_lines(lines_image(240, 30, tops=(12, 34), thin=True), "grad")) == 1
    # the thick-bar glyphs have empty rows inside each line, so the runs next to the
    # gap are only 3 rows long: bridged as before, and lines must be 5 px apart
    assert len(common.find_lines(lines_image(240, 30, tops=(12, 35)), "grad")) == 1
    assert len(common.find_lines(lines_image(240, 30, tops=(12, 37)), "grad")) == 2


def test_line_crops_invert_dark_text():
    dark = lines_image(30, 230)
    crops = common.line_crops(dark)
    assert len(crops) == 2 and all(d == 1 for _, _, d in crops)
    for crop, (x, y, w, h), _ in crops:
        assert crop.shape == (h, w)
        assert np.median(crop) < 64                          # inverted: dark ground, light text
    kept = common.line_crops(dark, invert_dark=False)
    assert np.median(kept[0][0]) > 192


def test_grey_and_rgba_inputs_agree():
    rgb = lines_image(30, 230)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 7, np.uint8)], axis=2)
    assert common.find_lines_ex(rgb) == common.find_lines_ex(rgba) == common.find_lines_ex(common.to_gray(rgb))


def test_empty_and_tiny_regions():
    assert common.find_lines(np.full((60, 200, 3), 128, np.uint8), "grad") == []
    for w, h in ((1, 1), (3, 1), (1, 9), (5, 5)):
        assert common.find_lines(np.zeros((h, w, 3), np.uint8), "grad") == []
    noise = np.random.default_rng(0).integers(0, 256, (100, 300, 3), dtype=np.uint8)
    assert len(common.find_lines(noise, "grad")) <= common.LINES_MAX


def test_legacy_twin_unchanged():
    for img in linefind_parity.make_cases(3, 120):
        if img.ndim == 3:
            assert common.find_lines(img[..., :3], "legacy") == legacy_frozen(img[..., :3])


def test_c_selftest(driver):
    out = subprocess.run([str(driver), "--selftest"], check=True, capture_output=True, text=True).stdout
    assert "selftest ok" in out


def test_c_matches_twin(driver):
    imgs = linefind_parity.make_cases(11, 160) + [lines_image(30, 230), lines_image(255, 180, outline=20),
                                                  lines_image(250, 200, outline=20)]
    imgs += [lines_image(240, 30, tops=(12, 32 + gap), thin=t) for gap in (2, 3, 4, 5) for t in (False, True)]
    cases = [(img, f) for img in imgs for f in ("grad", "legacy")]
    assert linefind_parity.run(driver, cases, seed=11) == []


def test_c_arena_bytes(driver):
    # The figures docs/linefind.md and vocr_lines.h quote, exactly.
    for (w, h), need in {(960, 544): 83_323, (480, 272): 25_355}.items():
        out = subprocess.run([str(driver), "--arena", str(w), str(h)], check=True, capture_output=True, text=True)
        assert int(out.stdout) == need, (w, h)


def test_find_lines_needs_a_finder():
    # find_lines(img) used to mean the legacy finder; a caller must now name the finder,
    # so old code fails here instead of silently switching to grad.
    with pytest.raises(TypeError):
        common.find_lines(lines_image(240, 30))


def manifests():
    return {p.stem: build_gt.read(p) for p in sorted(build_gt.MAN.glob("*.jsonl"))
            if not p.name.endswith(".candidates.jsonl")}


def test_manifests_hold_no_test_games():
    """No detector-built region and no PSP pool screenshot is from one of eval's test games:
    is_test_vn, plus eval/manifests/test_games.json once eval has merged (the shared copy
    is not in CI). research_a/b are the tuning sets and hold test games by design
    (docs/linefind.md). When eval's list grows, rerun eval/linefind/build_gt.py finalize."""
    path = ROOT / "eval" / "manifests" / "test_games.json"
    listed = set(json.loads(path.read_text())["all"]) if path.exists() else set()
    for name, rows in manifests().items():
        if name.startswith("research_"):
            continue
        bad = sorted({r["vn"] for r in rows if r["vn"] in listed or common.is_test_vn(r["vn"])})
        assert not bad, f"{name} has test games {bad}: rerun eval/linefind/build_gt.py finalize"


def test_holdouts_are_held_out():
    """dev and holdout share no VN, and no holdout VN is in research_a/b (the tuning sets)."""
    sets, tuning = manifests(), build_gt.tuning_vns()
    for plat in ("vita", "psp"):
        dev = {r["vn"] for r in sets[f"{plat}_dev"]}
        hold = {r["vn"] for r in sets[f"{plat}_holdout"]}
        assert hold and not dev & hold, plat
        assert not hold & tuning, (plat, sorted(hold & tuning))

"""synth/synth4_gap.py: crop measures, detector calibration, Jiten weights, tier costs, colour.

No shards, screenshots or fonts are needed: every crop here is drawn with numpy."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
import synth4_gap as G  # noqa: E402


def text_crop(h=32, w=160, bg=40, fg=230, blur=False, outline=None, seed=0):
    """A crop of bright 'glyphs' (bars and boxes) on a dark background."""
    rng = np.random.default_rng(seed)
    g = np.full((h, w), float(bg), np.float32)
    top, bot = h // 4, h - h // 4
    x = 6
    while x < w - 14:
        g[top:bot, x:x + 2] = fg                    # a vertical stroke
        g[top:top + 2, x:x + 10] = fg               # a horizontal stroke
        g[(top + bot) // 2:(top + bot) // 2 + 2, x + 4:x + 12] = fg
        g[top:bot, x + 10:x + 12] = fg
        x += 16 + int(rng.integers(0, 3))
    if outline is not None:
        import cv2
        fill = (g > (bg + fg) / 2).astype(np.uint8)
        ring = cv2.dilate(fill, np.ones((3, 3), np.uint8), iterations=1).astype(bool) & ~fill.astype(bool)
        g[ring] = outline
    if blur:
        import cv2
        g = cv2.GaussianBlur(g, (0, 0), 0.9)
    return np.clip(g, 0, 255).astype(np.uint8)


def test_binary_vs_antialiased():
    crisp = G.crop_features(text_crop())
    soft = G.crop_features(text_crop(blur=True))
    assert crisp["aa"] < 0.1 < soft["aa"]
    assert crisp["ink_h"] >= 14


def test_orientation_flip():
    g = 255 - text_crop()
    f = G.crop_features(g)
    assert f["flipped"] and f["aa"] < 0.1


def test_outline_halo():
    plain = G.crop_features(text_crop(bg=90))
    outlined = G.crop_features(text_crop(bg=90, outline=0))
    assert outlined["halo"] > plain["halo"] + 0.1


def test_nearest2x():
    small = text_crop(h=16, w=80, blur=True)
    up = np.repeat(np.repeat(small, 2, 0), 2, 1)
    f_up = G.crop_features(up)
    f_plain = G.crop_features(text_crop(h=32, w=160, blur=True))
    assert f_up["nearest2x"] > 0.2 > f_plain["nearest2x"]


def test_ruby_band():
    g = np.full((40, 160), 30, np.uint8)
    g[16:36, 10:150:12] = 230                      # main line: tall strokes over the width
    for x in range(10, 150, 12):
        g[16:18, x:x + 8] = 230
    g[6:12, 40:70] = 230                           # a short band above (furigana), 6 rows, a gap of 4 rows
    f = G.crop_features(g)
    assert f["ruby"]
    g2 = g.copy()
    g2[6:12, 40:70] = 30
    assert not G.crop_features(g2)["ruby"]


def test_rogan_gladen_and_youden():
    assert G.rogan_gladen(0.1, 0.5, 0.1) == 0.0
    assert abs(G.rogan_gladen(0.3, 0.5, 0.1) - 0.5) < 1e-9
    assert G.rogan_gladen(0.3, 0.12, 0.1) is None
    pos = np.array([0.0, 0.05, 0.1, 0.2])
    neg = np.array([0.5, 0.6, 0.7, 0.9])
    t, tpr, fpr = G.youden(pos, neg, lower=True)
    assert tpr == 1.0 and fpr == 0.0 and 0.2 <= t <= 0.5


def test_jiten_weights_and_fit():
    words = [("見る", 1), ("人", 2), ("鬱", 50)]
    w = G.jiten_char_weights(words, 1.0)
    assert abs(w["見"] - 1.0) < 1e-12 and abs(w["人"] - 0.5) < 1e-12 and abs(w["鬱"] - 0.02) < 1e-12
    real = {"見": 100.0, "人": 50.0, "鬱": 2.0}
    s, curve = G.fit_exponent(words, real, [0.5, 1.0, 1.5])
    assert s == 1.0 and len(curve) == 3


def test_coverage_and_kanji():
    assert G.is_kanji("漢") and G.is_kanji("々") and not G.is_kanji("あ")
    assert G.coverage({"a": 3, "b": 1}, {"a"}) == 0.75


def test_tier_cost_reproduces_the_budget_sweep():
    # docs/vita-memory-budget.md: w100 (FL10's topology with 5,355 classes) is 859,904 B, 149-280 ms;
    # w150v7000 is 1,335,600 B, 241-438 ms
    b, ms = G.tier_cost("FL10", 5355)
    assert abs(b - 859904) <= 2 and abs(ms - 280.0) < 0.2
    b, ms = G.tier_cost("H15", 5355)
    assert b == 1155968 and ms == 398.0
    assert G.tier_cost("FL10", 3501) == (657664, 235.0)


def test_colour_features():
    g = text_crop()
    rgb = np.repeat(g[:, :, None], 3, 2).astype(np.uint8)
    f = G.colour_features(rgb, g)
    assert not f["coloured"] and not f["grad"]
    red = rgb.copy()
    m = g > 128
    red[m] = [230, 40, 40]
    f = G.colour_features(red, g)
    assert f["coloured"] and f["hue"] == "r"
    # a vertical gradient fill: bright at the top of the ink rows, darker at the bottom
    gg = g.astype(np.float32)
    rows = np.arange(g.shape[0], dtype=np.float32)[:, None] * np.ones((1, g.shape[1]), np.float32)
    grad = np.where(m, 250 - 5.0 * (rows - 8), gg)
    gu = np.clip(grad, 0, 255).astype(np.uint8)
    f = G.colour_features(np.repeat(gu[:, :, None], 3, 2), gu)
    assert f["grad"]


def test_reserved_vns_hold_the_test_games():
    res = G.reserved_vns()
    test = set(json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())["all"])
    assert test <= res
    assert not G.usable("v4", res)          # CLANNAD (exclusion manifest)
    assert G.usable("v661", res) and G.usable("v19373", res)   # training data since 2026-10-14
    assert not G.usable("v3396", res) and not G.usable("v33", res)   # a psp-test game; Kanon (on CLANNAD's box)
    assert not G.usable(None, res)


def test_synth_truth_from_metadata():
    t = G.synth_truth({"raster": "pixel2", "cat": "pixel", "platform": "psp2x", "capture": "nearest",
                       "sw": 0, "shadow": True, "ruby": False})
    assert t["binary"] and t["pixel_font"] and t["nearest2x"] and t["shadow"] and not t["outline"]
    t = G.synth_truth({"raster": "aa8", "cat": "gothic", "platform": "vita", "capture": "clean", "sw": 1.0})
    assert not t["binary"] and t["outline"] and t["clean"]

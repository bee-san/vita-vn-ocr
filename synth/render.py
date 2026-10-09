#!/usr/bin/env python3
"""Synthetic PS Vita / PSP visual-novel dialogue renderer (synth2).

A sample is a dialogue *region*, like the one a user selects with Square: one to
eight horizontal lines in some VN style over a scene. The style model was
calibrated on contact sheets of real training-side Vita and PSP screenshots
(docs/synth2.md):

  layouts   ADV box (with or without a visible border), translucent band, NVL
            page, speech balloon, choice bars, bare text over the scene
  polarity  light text on dark boxes, dark text on light / pastel boxes, light
            text with an outline on light boxes, white text on saturated boxes
  effects   outlines (1-3 px, dark, light or coloured; sometimes doubled),
            hard or soft drop shadows, glows, vertical fill gradients
  names     a separate name tab above the box, a name line inside it, or an
            inline name before the bracket; often in an accent colour
  colour    accent-coloured keywords and whole lines
  extras    furigana and emphasis dots (distractors, never labelled), page
            cursors inline or in the box corner (never labelled), typewriter-
            partial text with a faint next glyph, kinsoku line breaks with
            hanging punctuation and indented continuation lines
  screens   Vita 960x544, PSP 480x272 at native size (12-16 px glyphs), and PSP
            upscaled 2x as on a Vita; sub-native rendering, blur, noise and JPEG
  fonts     OFL gothic, rounded, mincho, pen and display faces, and pixel fonts
            (DotGothic16, Fusion Pixel 10/12 px) drawn without anti-aliasing at
            multiples of their pixel size; ordinary faces are also drawn
            without anti-aliasing sometimes (bitmap-like PSP text)

Lines are cut with finders.find_lines_mask on the renderer's own text-ink mask:
the overlay's projection rules, applied to text of any polarity. A crop is kept
only when it matches one rendered line unambiguously; its label is that line's
text (furigana, emphasis dots, page cursors and faint typewriter glyphs are
distractors and never part of a label; full-width spaces are not labelled).
`--finder bright` cuts with the overlay's bright-pixel finder instead (bright
text only), `--finder grad` with the image-only gradient finder.

  render.py preview [--n 12]
      -> synth/preview_regions.png, synth/preview_crops.png (both <= 1280 px)
  render.py shards --out DIR --shards N --per-shard M [--workers W] [--seed S]
      -> DIR/synth_00000.npz ... (uint8 grey crops at native size + labels + meta)
"""
import argparse
import colorsys
import io
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, find_lines, is_test_vn, load_vocab, to_gray  # noqa: E402
from finders import find_lines_grad, find_lines_mask  # noqa: E402

FONT_DIRS = [DATA / "fonts", DATA / "fonts_extra"]   # the first one is never written to
CATEGORY = {
    "gothic": ["notosansjp", "mplus1p", "mplus2", "sawarabigothic", "zenkakugothicnew",
               "zenkakugothicantique", "bizudgothic", "bizudpgothic", "murecho", "ibmplexsansjp"],
    "rounded": ["mplusrounded1c", "zenmarugothic", "kiwimaru"],
    "mincho": ["notoserifjp", "sawarabimincho", "zenoldmincho", "zenantique", "shipporimincho",
               "shipporiantique", "bizudmincho", "hinamincho", "kaiseiopti", "kaiseitokumin",
               "yujisyuku"],
    "pen": ["kleeone", "yuseimagic", "yomogi", "zenkurenaido"],
    "pixel": ["dotgothic16", "fusionpixel10", "fusionpixel12"],
    "display": ["rocknrollone", "mochiypopone", "hachimarupop"],
}
FAMILY_WEIGHT = {"fusionpixel10": 0.4}
PIXEL_EM = {"dotgothic16": 16, "fusionpixel10": 10, "fusionpixel12": 12}
CAT_WEIGHT = {
    "vita": {"gothic": 0.45, "rounded": 0.20, "mincho": 0.21, "pen": 0.04, "pixel": 0.06, "display": 0.04},
    "psp": {"gothic": 0.44, "rounded": 0.16, "mincho": 0.18, "pen": 0.03, "pixel": 0.16, "display": 0.03},
}
POOL_WEIGHT = {"dialogue": 0.50, "narration": 0.34, "wiki": 0.16}
LAYOUT_WEIGHT = {"adv": 0.50, "band": 0.15, "nvl": 0.10, "balloon": 0.07, "choice": 0.08, "bare": 0.10}
SCHEME_WEIGHT = {   # per layout: text/box polarity
    "adv": {"light_on_dark": 0.53, "dark_on_light": 0.31, "outlined_on_light": 0.09, "on_colour": 0.07},
    "band": {"light_on_dark": 0.62, "dark_on_light": 0.28, "outlined_on_light": 0.10},
    "nvl": {"light_on_dark": 0.85, "dark_on_light": 0.15},
    "balloon": {"dark_on_light": 1.0},
    "choice": {"light_on_dark": 0.55, "dark_on_light": 0.30, "on_colour": 0.15},
    "bare": {"outlined_on_light": 0.75, "dark_outlined": 0.25},
}
PSP_SCHEME_BIAS = {"light_on_dark": 1.4, "dark_on_light": 0.45, "outlined_on_light": 1.3}
PLATFORM_WEIGHT = {"vita": 0.64, "psp": 0.26, "psp2x": 0.10}
SCREEN = {"vita": (960, 544), "psp": (480, 272), "psp2x": (480, 272)}

NO_START = set("、。，．・：；？！ー―…‥」』）】〕〉》］｝〙〗ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々ゝゞヽヾ"
               "!?),.:;～")
NO_END = set("「『（【〔〈《［｛〘〖(")
PAIRS = set("…‥―")
ACCENT_ON_DARK = [(0.93, 0.38), (0.58, 0.40), (0.14, 0.50), (0.08, 0.50), (0.30, 0.42), (0.76, 0.32),
                  (0.50, 0.45), (0.00, 0.35)]
ACCENT_ON_LIGHT = [(0.00, 0.85, 0.78), (0.62, 0.85, 0.72), (0.90, 0.80, 0.72), (0.35, 0.85, 0.48),
                   (0.07, 0.90, 0.62), (0.78, 0.70, 0.58), (0.55, 0.90, 0.55)]
INTERJ = ["あ、", "えっ、", "ふふっ、", "うん、", "ねえ、", "ちょっと、", "はぁ……", "あの、", "えへへ、",
          "なっ……", "へえ、", "よし、", "おい、", "うーん……", "ほら、", "わぁ、", "そっか、", "ええと、"]
ENDINGS = ["！", "？", "！？", "……", "っ！", "♪", "～", "ーっ！", "！！", "……？", "♥", "っ"]

R = None          # per-process resources, loaded before workers fork
_FONTS = {}
_AXES = {}
_BG = {}


class Resources:
    def __init__(self, pool_limit=300_000):
        self.vocab = load_vocab()
        self.vocab_set = set(self.vocab)
        self.pools = {}
        for name in POOL_WEIGHT:
            text = (DATA / "corpus" / f"{name}.txt").read_text(encoding="utf-8")
            self.pools[name] = [s for s in text.split("\n", pool_limit)[:pool_limit] if s]
        self.fonts = self._fonts()
        self.by_cat = {c: [f for f in self.fonts if f[0] == c] for c in CATEGORY}
        self.by_cat = {c: fs for c, fs in self.by_cat.items() if fs}
        # Backgrounds: training-side screenshots only. The exclusion manifest holds VN ids
        # (eval's reserved test games) and single screenshots (hand-labelled real sets).
        exclude = set()
        ex_path = HERE.parent / "data" / "manifests" / "synth2_bg_exclude.txt"
        if ex_path.exists():
            exclude = {x.split()[0] for x in ex_path.read_text().splitlines() if x.strip() and x[0] != "#"}

        def usable(p):
            vn = p.name.split("_")[0]
            return not is_test_vn(vn) and vn not in exclude and p.name not in exclude
        self.bg_vita = [p for p in sorted((DATA / "vndb" / "shots").glob("*.jpg")) if usable(p)]
        self.bg_psp = [p for p in sorted((DATA / "vndb_psp" / "shots").glob("*.jpg")) if usable(p)]
        self.katakana = [c for c in self.vocab if 0x30A1 <= ord(c) <= 0x30FA and c not in "ヵヶヮ"]
        self.hiragana = [c for c in self.vocab if 0x3041 <= ord(c) <= 0x3093 and c not in "ゎゐゑ"]
        freq = json.loads((DATA / "corpus" / "char_freq.json").read_text(encoding="utf-8"))
        self.kanji = [c for c in freq if is_kanji(c) and c in self.vocab_set][:2000]
        self.name_kanji = self.kanji[:1200]

    def _fonts(self):
        out = []
        for root in FONT_DIRS:
            cover_path = root / "coverage.json"
            cover = json.loads(cover_path.read_text(encoding="utf-8")) if cover_path.exists() else {}
            changed = False
            for cat, fams in CATEGORY.items():
                for fam in fams:
                    for path in sorted((root / fam).glob("*.[to]tf")):
                        key = f"{fam}/{path.name}"
                        if key not in cover:
                            from fontTools.ttLib import TTFont
                            cmap = TTFont(str(path), lazy=True).getBestCmap() or {}
                            cover[key] = "".join(c for c in self.vocab if ord(c) in cmap)
                            changed = True
                        chars = set(cover[key])
                        if len(chars) > 1000:
                            out.append((cat, fam, path, chars))
            if changed and root != FONT_DIRS[0]:
                cover_path.write_text(json.dumps(cover, ensure_ascii=False), encoding="utf-8")
        return out


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or c == "々"


def choose(rng, weights):
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=np.float64)
    return names[rng.choice(len(names), p=p / p.sum())]


def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0), 1), min(max(v, 0), 1)), np.float32) * 255


def luma(c):
    c = np.asarray(c, np.float32)
    return float(77 * c[0] + 150 * c[1] + 29 * c[2]) / 256.0


def dark_colour(rng, vmax=0.26):
    return hsv(rng.random(), rng.uniform(0, 0.85), rng.uniform(0.0, vmax))


def light_colour(rng, smax=0.18, vmin=0.88):
    return hsv(rng.random(), rng.uniform(0, smax), rng.uniform(vmin, 1.0))


def text_light(rng):
    """Light text fill: mostly pure white, as in most games."""
    return light_colour(rng, 0.04, 0.97) if rng.random() < 0.6 else light_colour(rng, 0.12, 0.92)


def text_dark(rng):
    """Dark text fill: mostly near-black, sometimes dark brown, navy or wine."""
    return dark_colour(rng, 0.12) if rng.random() < 0.55 else dark_colour(rng, 0.30)


def accent(rng, on_dark):
    if on_dark:
        h, s = ACCENT_ON_DARK[rng.integers(len(ACCENT_ON_DARK))]
        return hsv(h + rng.uniform(-0.03, 0.03), s * rng.uniform(0.8, 1.25), rng.uniform(0.93, 1.0))
    h, s, v = ACCENT_ON_LIGHT[rng.integers(len(ACCENT_ON_LIGHT))]
    return hsv(h + rng.uniform(-0.03, 0.03), s, v * rng.uniform(0.8, 1.05))


# ----------------------------------------------------------------------------- text

def get_font(path, size, wght):
    key = (str(path), size, wght)
    font = _FONTS.get(key)
    if font is None:
        if len(_FONTS) > 400:
            _FONTS.clear()
        font = ImageFont.truetype(str(path), size)
        if wght is not None:
            try:
                font.set_variation_by_axes([wght])
            except (OSError, ValueError):
                pass
        _FONTS[key] = font
    return font


def pick_weight(rng, path):
    if "[wght]" not in path.name:
        return None
    if path not in _AXES:
        try:
            ax = ImageFont.truetype(str(path), 20).get_variation_axes()[0]
            _AXES[path] = (max(ax["minimum"], 300), min(ax["maximum"], 800))
        except (OSError, IndexError, KeyError):
            _AXES[path] = None
    if _AXES[path] is None:
        return None
    lo, hi = _AXES[path]
    return int(round(rng.uniform(lo, hi) / 100.0) * 100)


def pick_font(rng, chars, platform, cat=None):
    weights = CAT_WEIGHT["vita" if platform == "vita" else "psp"]
    cat = cat or choose(rng, {c: weights[c] for c in R.by_cat})
    cands = [f for f in R.by_cat.get(cat, []) if chars <= f[3]]
    if cands and cat == "pixel":
        wt = np.array([FAMILY_WEIGHT.get(f[1], 1.0) for f in cands])
        return cands[rng.choice(len(cands), p=wt / wt.sum())]
    if not cands:
        cands = [f for f in R.fonts if chars <= f[3] and f[0] != "pixel"]
    return cands[rng.integers(len(cands))] if cands else None


def clean(s):
    return "".join(c for c in s if c in R.vocab_set)


def sample_text(rng, kind="dialogue"):
    """VN-like text: corpus chunks with VN punctuation, interjections and endings."""
    if kind == "choice":
        items = R.pools["dialogue" if rng.random() < 0.5 else "narration"]
        s = clean(items[rng.integers(len(items))]).strip("「」『』")
        cut = [i + 1 for i, c in enumerate(s) if c in "、。！？" and 3 <= i < 18]
        s = s[:cut[0]] if cut else s[:int(rng.integers(5, 16))]
        return s.rstrip("、。") or s
    pool = choose(rng, POOL_WEIGHT)
    items = R.pools[pool]
    s = clean(items[rng.integers(len(items))])
    if not s:
        return s
    if pool == "narration":
        r = rng.random()
        if r < 0.15:
            s = "「" + s + "」"
        elif r < 0.22:
            s = "（" + s.rstrip("。") + "）"
    opener = s[0] if s[0] in "「『（" else ""
    body = s[1:] if opener else s
    r = rng.random()
    if r < 0.05:
        body = "……" + body
    elif r < 0.08:
        body = "――" + body
    elif r < 0.14:
        body = INTERJ[rng.integers(len(INTERJ))] + body
    if rng.random() < 0.16:
        close = body[-1] if body and body[-1] in "」』）" else ""
        core = body[:-1] if close else body
        if core.endswith("。"):
            end = ENDINGS[rng.integers(len(ENDINGS))]
            if all(c in R.vocab_set for c in end):
                core = core[:-1] + end
        body = core + close
    return clean(opener + body)


def random_name(rng):
    """(display, label): display may hold a full-width space, which is not labelled."""
    r = rng.random()
    if r < 0.04:
        name = "？？？"
    elif r < 0.44:
        name = "".join(rng.choice(R.katakana, size=int(rng.integers(2, 6))))
        if rng.random() < 0.12:
            name += "・" + "".join(rng.choice(R.katakana, size=int(rng.integers(2, 5))))
    elif r < 0.90:
        fam = "".join(rng.choice(R.name_kanji, size=int(rng.choice([1, 2, 2, 2, 3]))))
        giv = "".join(rng.choice(R.name_kanji, size=int(rng.choice([1, 2, 2]))))
        name = fam + ("　" if rng.random() < 0.35 else "") + giv
        if rng.random() < 0.15:
            name = giv
    else:
        name = "".join(rng.choice(R.hiragana, size=int(rng.integers(2, 5))))
    if rng.random() < 0.18:
        name = "【" + name + "】"
    return name, name.replace("　", "")


def add_spaces(text, rng):
    """Typesetting: a full-width space after ！ and ？ when more text follows."""
    if rng.random() > 0.6:
        return text
    out = []
    for i, c in enumerate(text):
        out.append(c)
        if c in "！？" and i + 1 < len(text) and text[i + 1] not in "！？」』）…―ー～♪♥":
            out.append("　")
    return "".join(out)


def wrap(text, cpl, max_lines, rng, at_punct=False):
    """Greedy wrap with kinsoku: closing marks and small kana hang on the previous
    line (up to 2), opening brackets never end a line, …… and ―― pairs and runs of
    ASCII alphanumerics are not split, and optionally break after 、。！？ near the end."""
    lines, i = [], 0
    while i < len(text) and len(lines) < max_lines:
        j = min(i + cpl, len(text))
        if at_punct and j < len(text):
            for k in range(j, i + max(2, int(cpl * 0.6)), -1):
                if text[k - 1] in "、。！？" and text[k - 1:k + 1] != "！？":
                    j = k
                    break
        if j < len(text) and text[j - 1] in PAIRS and text[j] == text[j - 1]:
            j += 1 if j + 1 - i <= cpl + 1 else -1
        while j - i > 1 and j < len(text) and text[j].isascii() and text[j].isalnum() \
                and text[j - 1].isascii() and text[j - 1].isalnum() and j - i > cpl // 2:
            j -= 1
        hang = 0
        while j < len(text) and (text[j] in NO_START or text[j] == "　") and hang < 2:
            j, hang = j + 1, hang + 1
        while j - i > 1 and j < len(text) and text[j - 1] in NO_END:
            j -= 1
        lines.append(text[i:j].lstrip("　"))
        i = j
    return [ln for ln in lines if ln]


# ----------------------------------------------------------------------------- style

def sample_style(rng, platform):
    st = {"platform": platform, "layout": choose(rng, LAYOUT_WEIGHT)}
    weights = dict(SCHEME_WEIGHT[st["layout"]])
    if platform != "vita":          # PSP shots: far more light text than dark (docs/synth2.md)
        weights = {k: v * PSP_SCHEME_BIAS.get(k, 1.0) for k, v in weights.items()}
    st["scheme"] = choose(rng, weights)
    psp = platform != "vita"
    sch = st["scheme"]
    # Box colours.
    if sch == "light_on_dark":
        box = dark_colour(rng, 0.32)
        if rng.random() < 0.25:     # navy / wine / green tinted dark boxes
            box = hsv(rng.random(), rng.uniform(0.4, 0.9), rng.uniform(0.15, 0.42))
        fill = text_light(rng) if rng.random() < 0.85 else accent(rng, True)
    elif sch == "dark_on_light":
        box = light_colour(rng, 0.35, 0.82) if rng.random() < 0.7 else hsv(
            rng.uniform(0.05, 0.16), rng.uniform(0.15, 0.4), rng.uniform(0.75, 0.95))   # parchment
        fill = text_dark(rng) if rng.random() < 0.88 else accent(rng, False)
    elif sch == "outlined_on_light":
        box = light_colour(rng, 0.4, 0.70)
        fill = text_light(rng) if rng.random() < 0.8 else accent(rng, True)
    elif sch == "dark_outlined":
        box = light_colour(rng, 0.3, 0.6)
        fill = text_dark(rng)
    else:   # on_colour: white text on a saturated box
        box = hsv(rng.random(), rng.uniform(0.5, 0.95), rng.uniform(0.35, 0.72))
        fill = text_light(rng)
    st["box"] = box
    st["fill"] = fill
    on_dark = luma(fill) > luma(box)
    # Outline / shadow / glow.
    fs_ref = 14 if psp else 24
    sw = 0
    stroke_c = None
    if sch in ("outlined_on_light", "dark_outlined") or (sch == "on_colour" and rng.random() < 0.7):
        sw = 1 if psp else int(rng.choice([1, 2, 2, 3], p=[0.3, 0.35, 0.2, 0.15]))
    elif rng.random() < (0.34 if sch == "light_on_dark" else 0.16):
        sw = 1 if psp else int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
    if sw:
        if on_dark:
            stroke_c = dark_colour(rng, 0.25) if rng.random() < 0.75 else hsv(
                rng.random(), rng.uniform(0.5, 0.9), rng.uniform(0.25, 0.55))
            if sch == "on_colour":
                hb, sb, vb = colorsys.rgb_to_hsv(*(box / 255))
                stroke_c = hsv(hb, min(1, sb + 0.1), vb * rng.uniform(0.35, 0.65))
        else:
            stroke_c = light_colour(rng, 0.15, 0.9) if rng.random() < 0.85 else accent(rng, True)
    st["stroke"] = {"w": sw, "c": stroke_c} if sw else None
    st["stroke2"] = None
    if sw and not psp and rng.random() < 0.08:
        st["stroke2"] = {"w": sw + int(rng.integers(1, 3)),
                         "c": dark_colour(rng, 0.2) if luma(stroke_c) > 128 else light_colour(rng)}
    st["shadow"] = None
    p_shadow = {"light_on_dark": 0.38, "dark_on_light": 0.10, "outlined_on_light": 0.3,
                "dark_outlined": 0.1, "on_colour": 0.3}[sch]
    if st["layout"] in ("nvl", "bare"):
        p_shadow += 0.3
    if rng.random() < p_shadow:
        d = 1 if psp else int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
        st["shadow"] = {"dx": d if rng.random() < 0.85 else 0, "dy": d,
                        "blur": float(rng.uniform(0.4, 1.6)) if rng.random() < 0.45 else 0.0,
                        "c": dark_colour(rng, 0.15) if on_dark else light_colour(rng, 0.1, 0.85),
                        "a": float(rng.uniform(0.55, 1.0))}
    st["glow"] = None
    if rng.random() < 0.05:
        st["glow"] = {"s": float(rng.uniform(1.0, 2.5)) * (0.6 if psp else 1),
                      "c": dark_colour(rng, 0.2) if on_dark else light_colour(rng),
                      "a": float(rng.uniform(0.5, 0.9))}
    st["grad"] = None
    if rng.random() < 0.10:
        h = rng.random()
        c1 = hsv(h, rng.uniform(0.15, 0.45), 1.0) if on_dark else hsv(h, rng.uniform(0.5, 0.9),
                                                                      rng.uniform(0.25, 0.5))
        st["grad"] = c1
    st["accent"] = accent(rng, on_dark)
    st["on_dark"] = on_dark
    st["fs_ref"] = fs_ref
    return st


# ----------------------------------------------------------------------------- raster

def over(img, m, col, a=1.0):
    """Composite colour col (vec3 or broadcastable image) over img with coverage m in [0,1]."""
    m3 = (m * a)[..., None] if a != 1.0 else m[..., None]
    img *= 1.0 - m3
    img += np.asarray(col, np.float32) * m3
    return img


def shift(m, dx, dy):
    out = np.zeros_like(m)
    h, w = m.shape
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        m[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def load_bg(path, scale):
    key = (str(path), round(scale, 2))
    im = _BG.get(key)
    if im is None:
        if len(_BG) > 40:
            _BG.clear()
        pil = Image.open(path).convert("RGB")
        if scale != 1.0:
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                             Image.BILINEAR)
        im = np.asarray(pil)
        _BG[key] = im
    return im


def background(rng, w, h, platform):
    """Scene behind the box: a crop of the upper part of a training-side screenshot,
    or a synthetic gradient with blobs. Screenshot crops avoid the lower 38% of the
    screen, where the screenshot's own dialogue text usually is."""
    if rng.random() < 0.75:
        if platform != "vita" and R.bg_psp and rng.random() < 0.75:
            path, base = R.bg_psp[rng.integers(len(R.bg_psp))], 1.0
        elif R.bg_vita:
            path, base = R.bg_vita[rng.integers(len(R.bg_vita))], 0.5 if platform != "vita" else 1.0
        else:
            path = None
        if path is not None:
            try:
                im = load_bg(path, base * round(float(rng.uniform(1.0, 1.4)), 1))
                H, W = im.shape[:2]
                if W >= w and H >= h:
                    x0 = int(rng.integers(0, W - w + 1))
                    ymax = max(0, min(H - h, int(0.62 * H) - h))
                    y0 = int(rng.integers(0, ymax + 1))
                    arr = im[y0:y0 + h, x0:x0 + w].astype(np.float32)
                    return arr[:, ::-1].copy() if rng.random() < 0.5 else arr
            except OSError:
                pass
    c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    if rng.random() < 0.5:
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    else:
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    img = np.broadcast_to(c0 * (1 - t) + c1 * t, (h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(0, 6))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rx, ry = rng.uniform(10, w / 2 + 10), rng.uniform(10, h / 2 + 10)
        a = (np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)) * rng.uniform(0.3, 1))[..., None]
        img = img * (1 - a) + rng.uniform(0, 255, 3).astype(np.float32) * a
    if rng.random() < 0.5:
        small = rng.normal(0, rng.uniform(3, 15), (h // 8 + 2, w // 8 + 2, 3)).astype(np.float32)
        img += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 0, 255)


def pattern(rng, w, h):
    """Low-contrast box texture in [-1, 1]: checks, diagonal stripes, dots or noise."""
    kind = choose(rng, {"checker": 0.4, "stripes": 0.3, "dots": 0.15, "noise": 0.15})
    p = int(rng.integers(6, 26))
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "checker":
        return (((xx // p + yy // p) % 2) * 2 - 1).astype(np.float32)
    if kind == "stripes":
        return np.sin(2 * np.pi * (xx + yy) / (2 * p)).astype(np.float32)
    if kind == "dots":
        return ((((xx % p) - p / 2) ** 2 + ((yy % p) - p / 2) ** 2 < (p / 4) ** 2) * 2 - 1).astype(np.float32)
    small = rng.normal(0, 1, (h // p + 2, w // p + 2)).astype(np.float32)
    return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), -1, 1)


def shape_mask(w, h, kind, rect, radius=0, width=None, tail=None):
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), np.float32)
    fill, outline = (255, None) if width is None else (None, 255)
    if kind == "ellipse":
        d.ellipse((x0, y0, x1, y1), fill=fill, outline=outline, width=width or 1)
    elif kind == "round" and radius > 0:
        r = int(min(radius, (x1 - x0) / 2 - 1, (y1 - y0) / 2 - 1))
        d.rounded_rectangle((x0, y0, x1, y1), radius=max(r, 0), fill=fill, outline=outline, width=width or 1)
    else:
        d.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=width or 1)
    if tail is not None:
        d.polygon(tail, fill=255 if width is None else None, outline=outline)
    return np.asarray(im, np.float32) / 255.0


def draw_box(img, rng, b, st):
    """Composite one box: shape, colour (solid, gradient or patterned), alpha
    (flat or vertical fade) and an optional border."""
    h, w = img.shape[:2]
    m = shape_mask(w, h, b["shape"], b["rect"], b.get("radius", 0), tail=b.get("tail"))
    col = np.asarray(b["colour"], np.float32)
    if b.get("colour2") is not None:
        t = np.clip((np.arange(h, dtype=np.float32) - b["rect"][1]) / max(1.0, b["rect"][3] - b["rect"][1]),
                    0, 1)[:, None, None]
        col = col * (1 - t) + np.asarray(b["colour2"], np.float32) * t
    if b.get("pattern"):
        pat = pattern(rng, w, h)[..., None] * b["pattern"]
        col = col + pat * (1 if luma(b["colour"]) < 128 else -1)
    a = b["alpha"]
    if isinstance(a, tuple):
        t = np.clip((np.arange(h, dtype=np.float32) - b["rect"][1]) / max(1.0, b["rect"][3] - b["rect"][1]),
                    0, 1)
        am = (a[0] * (1 - t) + a[1] * t)[:, None]
    else:
        am = np.float32(a)
    over(img, m * am, np.clip(col, 0, 255))
    if b.get("border"):
        bw, bc = b["border"]
        bm = shape_mask(w, h, b["shape"], b["rect"], b.get("radius", 0), width=bw, tail=b.get("tail"))
        over(img, bm, bc)
    return img


class Layer:
    """Masks of one text style group (fills per colour index, outlines), drawn at
    supersampled size ss, with or without anti-aliasing."""

    def __init__(self, w, h, ss, aa, n_fill, sw, sw2):
        self.ss, self.sw, self.sw2 = ss, sw, sw2
        size = (w * ss, h * ss)
        self.fill = [Image.new("L", size, 0) for _ in range(n_fill)]
        self.s1 = Image.new("L", size, 0) if sw else None
        self.s2 = Image.new("L", size, 0) if sw2 else None
        self.d = [ImageDraw.Draw(m) for m in self.fill]
        self.d1 = ImageDraw.Draw(self.s1) if sw else None
        self.d2 = ImageDraw.Draw(self.s2) if sw2 else None
        for dr in self.d + [self.d1, self.d2]:
            if dr is not None and not aa:
                dr.fontmode = "1"

    def glyph(self, x, base, ch, font, k=0, small=False):
        ss = self.ss
        xy = (x * ss, base * ss)
        self.d[k].text(xy, ch, font=font, fill=255, anchor="ls")
        if self.d1 is not None:
            sw = max(self.sw - 1, 1) if small else self.sw
            self.d1.text(xy, ch, font=font, fill=255, anchor="ls", stroke_width=sw * ss, stroke_fill=255)
        if self.d2 is not None:
            sw = max(self.sw2 - 1, 1) if small else self.sw2
            self.d2.text(xy, ch, font=font, fill=255, anchor="ls", stroke_width=sw * ss, stroke_fill=255)

    def poly(self, pts, k=0, ellipse=False):
        ss = self.ss
        p = [(x * ss, y * ss) for x, y in pts]
        for dr in [self.d[k], self.d1, self.d2]:
            if dr is None:
                continue
            if ellipse:
                dr.ellipse((p[0][0], p[0][1], p[1][0], p[1][1]), fill=255)
            else:
                dr.polygon(p, fill=255)

    def arrays(self):
        def arr(im):
            if im is None:
                return None
            if self.ss > 1:
                im = im.resize((im.width // self.ss, im.height // self.ss), Image.BOX)
            return np.asarray(im, np.float32) / 255.0
        return [arr(m) for m in self.fill], arr(self.s1), arr(self.s2)


def compose_text(img, fills, s1, s2, cols, st, grad_rows=None):
    """Shadow, glow, outer and inner outline, then the fills (in colour order)."""
    union = fills[0].copy()
    for f in fills[1:]:
        np.maximum(union, f, out=union)
    outer = s2 if s2 is not None else (s1 if s1 is not None else union)
    sh = st.get("shadow")
    if sh:
        m = shift(outer, sh["dx"], sh["dy"])
        if sh["blur"] > 0:
            m = cv2.GaussianBlur(m, (0, 0), sh["blur"]) * 1.3
        over(img, np.clip(m, 0, 1), sh["c"], sh["a"])
    gl = st.get("glow")
    if gl:
        m = np.clip(cv2.GaussianBlur(outer, (0, 0), gl["s"]) * 2.2, 0, 1)
        over(img, m, gl["c"], gl["a"])
    if s2 is not None:
        over(img, s2, st["stroke2"]["c"])
    if s1 is not None:
        over(img, s1, st["stroke"]["c"])
    for k, f in enumerate(fills):
        c = cols[k]
        if k == 0 and grad_rows is not None:
            c = grad_rows
        over(img, f, c)
    ink = np.maximum(union, outer)
    if sh and sh["a"] >= 0.5 and sh["blur"] <= 1.0:
        np.maximum(ink, shift(outer, sh["dx"], sh["dy"]), out=ink)
    return union, ink


# ----------------------------------------------------------------------------- layout

CURSORS = ["tri", "tri", "chevron", "circle_tri", "diamond", "sparkle", "star", "arrow", "snow", "feather"]


def cursor_shape(kind, cx, cy, s):
    """Polygons (or ellipse boxes) for a page-cursor icon of size s centred at (cx, cy)."""
    r = s / 2
    if kind == "tri":
        return [("poly", [(cx - r, cy - r * 0.7), (cx + r, cy - r * 0.7), (cx, cy + r * 0.8)])]
    if kind == "chevron":
        t = s * 0.22
        return [("poly", [(cx - r, cy - r * 0.5), (cx - r + t, cy - r * 0.5 - t * 0.3), (cx, cy + r * 0.2),
                          (cx + r - t, cy - r * 0.5 - t * 0.3), (cx + r, cy - r * 0.5), (cx, cy + r * 0.7)])]
    if kind == "circle_tri":
        return [("ring", [(cx - r, cy - r), (cx + r, cy + r)]),
                ("poly", [(cx - r * 0.45, cy - r * 0.25), (cx + r * 0.45, cy - r * 0.25), (cx, cy + r * 0.45)])]
    if kind == "diamond":
        return [("poly", [(cx, cy - r), (cx + r * 0.75, cy), (cx, cy + r), (cx - r * 0.75, cy)])]
    if kind == "sparkle":
        q = r * 0.22
        return [("poly", [(cx, cy - r), (cx + q, cy - q), (cx + r, cy), (cx + q, cy + q), (cx, cy + r),
                          (cx - q, cy + q), (cx - r, cy), (cx - q, cy - q)])]
    if kind == "star":
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.42
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return [("poly", pts)]
    if kind == "arrow":
        t = r * 0.3
        return [("poly", [(cx - t, cy - r), (cx + t, cy - r), (cx + t, cy), (cx + r * 0.8, cy),
                          (cx, cy + r), (cx - r * 0.8, cy), (cx - t, cy)])]
    if kind == "snow":
        out = []
        for i in range(3):
            a = i * math.pi / 3
            dx, dy, nx, ny = r * math.cos(a), r * math.sin(a), -r * 0.12 * math.sin(a), r * 0.12 * math.cos(a)
            out.append(("poly", [(cx - dx + nx, cy - dy + ny), (cx + dx + nx, cy + dy + ny),
                                 (cx + dx - nx, cy + dy - ny), (cx - dx - nx, cy - dy - ny)]))
        return out
    # feather: a slanted leaf with a quill
    pts = []
    for i in range(16):
        a = 2 * math.pi * i / 16
        x, y = r * 0.35 * math.cos(a), r * math.sin(a)
        pts.append((cx + x * 0.9 + y * 0.35, cy + y * 0.9 - x * 0.2))
    return [("poly", pts)]


def draw_cursor(layer, kind, cx, cy, s):
    for typ, pts in cursor_shape(kind, cx, cy, s):
        if typ == "ring":
            (x0, y0), (x1, y1) = pts
            layer.poly([(x0, y0), (x1, y1)], k=layer.cursor_k, ellipse=True)
            inner = (x1 - x0) * 0.16
            for dr in [layer.d[layer.cursor_k], layer.d1, layer.d2]:
                if dr is not None:
                    ss = layer.ss
                    dr.ellipse(((x0 + inner) * ss, (y0 + inner) * ss, (x1 - inner) * ss, (y1 - inner) * ss),
                               fill=0)
        else:
            layer.poly(pts, k=layer.cursor_k)


def measure(font, ss, text, track):
    return sum(font.getlength(c) for c in text) / ss + track * max(len(text) - 1, 0)


def render_region(rng, platform=None):
    """Returns (rgb uint8 HxWx3, layout, ink HxW bool, meta) or None for an unusable draw."""
    platform = platform or choose(rng, PLATFORM_WEIGHT)
    st = sample_style(rng, platform)
    lay, sch = st["layout"], st["scheme"]
    psp = platform != "vita"
    SW, SH = SCREEN[platform]

    # ---- text and font
    if lay == "choice":
        n_lines = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.2]))
        texts = [sample_text(rng, "choice") for _ in range(n_lines)]
        if min(len(t) for t in texts) < 2:
            return None
        chars = set("".join(texts))
    else:
        text = sample_text(rng)
        if len(text) < 2:
            return None
        chars = set(text)
    name_mode = None
    if lay in ("adv", "band", "bare"):
        name_mode = choose(rng, {"tab": 0.32 if lay == "adv" else 0.0, "line": 0.20, "inline": 0.10,
                                 None: 0.38 if lay == "adv" else 0.70})
    elif lay == "nvl" and rng.random() < 0.15:
        name_mode = "line"
    name_disp = name_label = None
    if name_mode:
        name_disp, name_label = random_name(rng)
        chars |= set(name_disp.replace("　", ""))
    font = pick_font(rng, chars, platform)
    if font is None:
        return None
    cat, fam, path, fchars = font
    wght = pick_weight(rng, path)
    if fam in PIXEL_EM:
        em = PIXEL_EM[fam]
        if psp:
            mult = 1 if em >= 12 else int(rng.choice([1, 2], p=[0.3, 0.7]))
        else:
            mult = int(rng.choice([2, 2, 3])) if em <= 12 else int(rng.choice([1, 2], p=[0.5, 0.5]))
            if em == 16 and mult == 1:
                mult = 1.5
        fs = int(round(em * mult))
        aa = float(mult).is_integer() is False
        ss = 1
    else:
        fs = int(rng.integers(12, 17)) if psp else int(np.clip(rng.normal(24, 3.2), 16, 34))
        aa = rng.random() < (0.55 if psp else 0.95)
        ss = 2 if aa and rng.random() < 0.45 else 1
    track = 0 if rng.random() < 0.6 else int(rng.integers(-1, 3 if psp else 5))
    f = get_font(path, fs * ss, wght)
    sw = st["stroke"]["w"] if st["stroke"] else 0
    sw2 = st["stroke2"]["w"] if st["stroke2"] else 0
    sh = st["shadow"]
    dy = sh["dy"] if sh else 0

    # ---- lines
    def pad(a, b):              # box paddings, halved on the PSP's half-size screen
        return int(rng.integers(a, b)) // (2 if psp else 1)
    if lay == "choice":
        lines = [add_spaces(t, rng) for t in texts]
    else:
        box_w = rng.uniform(0.55, 0.97) * SW
        if lay == "nvl":
            box_w = rng.uniform(0.6, 0.95) * SW
        cap = max(4, int((box_w - 2 * fs) // max(fs + track, 6)))
        cpl = max(6, int(cap * rng.uniform(0.62, 1.0)))
        max_lines = {"adv": int(rng.choice([1, 2, 3, 4], p=[0.22, 0.33, 0.35, 0.10])),
                     "band": int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])),
                     "nvl": int(rng.integers(3, 8)),
                     "balloon": int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])),
                     "bare": int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))}[lay]
        if lay == "balloon":
            cpl = min(cpl, int(rng.integers(8, 20)))
        lines = wrap(add_spaces(text, rng), cpl, max_lines, rng, at_punct=rng.random() < 0.35)
        if lay == "nvl" and rng.random() < 0.5 and len(lines) > 1:
            # NVL pages hold several short paragraphs.
            lines = lines + wrap(sample_text(rng), cpl, max(0, max_lines - len(lines)), rng)
            lines = [ln for ln in lines if set(ln) <= fchars | {"　"}]
    if not lines:
        return None
    indent = rng.random() < 0.5 and lines[0][0] in "「『（" and lay not in ("choice", "balloon")
    inline_name = name_mode == "inline"
    if inline_name:
        gap = "　" * int(rng.choice([0, 0, 1, 2]))
        lines[0] = name_disp + gap + lines[0]
    labels = [ln.replace("　", "") for ln in lines]

    # ---- spans of accent colour (keywords, whole coloured lines)
    n_fill = 1
    colour_of = [[0] * len(ln) for ln in lines]
    if lay != "choice" and rng.random() < 0.10:
        li = int(rng.integers(len(lines)))
        if len(lines[li]) >= 4:
            a = int(rng.integers(0, len(lines[li]) - 2))
            b = min(len(lines[li]), a + int(rng.integers(2, 7)))
            for k in range(a, b):
                colour_of[li][k] = 1
            n_fill = 2
    elif lay != "choice" and rng.random() < 0.06:
        li = int(rng.integers(len(lines)))
        colour_of[li] = [1] * len(lines[li])
        n_fill = 2
    name_coloured = name_mode is not None and rng.random() < 0.55
    if inline_name and name_coloured:
        for k in range(len(name_disp)):
            colour_of[0][k] = 1
        n_fill = 2
    cursor_k = n_fill
    n_fill += 1        # the cursor gets its own fill colour slot

    # ---- ruby / partial / cursor plans
    ruby = lay in ("adv", "band", "nvl", "bare") and rng.random() < 0.11
    ruby_fs = max(7 if psp else 9, int(round(fs * rng.uniform(0.42, 0.55))))
    ruby_gap = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))
    ruby_space = ruby_fs + ruby_gap if ruby else 0
    boten = not ruby and lay in ("adv", "band", "nvl") and rng.random() < 0.02
    if boten:
        ruby_space = max(ruby_space, int(fs * 0.3) + 1)
    partial = None
    long_lines = [i for i, ln in enumerate(lines) if len(ln) >= 2]
    if lay in ("adv", "band", "nvl") and rng.random() < 0.06 and long_lines and sum(map(len, labels)) > 4:
        partial = {"line": int(long_lines[rng.integers(len(long_lines))]), "ghost": rng.random() < 0.4,
                   "ghost_a": float(rng.uniform(0.06, 0.2))}
        li = partial["line"]
        keep = int(rng.integers(1, len(lines[li])))
        partial["rest"] = lines[li][keep]
        lines, labels, colour_of = lines[:li + 1], labels[:li + 1], colour_of[:li + 1]
        lines[li] = lines[li][:keep]
        labels[li] = lines[li].replace("　", "")
        colour_of[li] = colour_of[li][:keep]
        if not labels[li]:
            return None
    cursor = None
    if lay in ("adv", "band", "balloon", "nvl") and partial is None and rng.random() < 0.45:
        cursor = {"kind": CURSORS[rng.integers(len(CURSORS))],
                  "inline": rng.random() < 0.35, "size": fs * rng.uniform(0.45, 0.8),
                  "same": rng.random() < 0.4}

    # ---- geometry (canvas = the user's region)
    widths = [measure(f, ss, ln, track) for ln in lines]
    pitch = max(fs * rng.uniform(1.3, 1.85), fs + 2 * max(sw, sw2) + dy + 6) + ruby_space
    if lay == "nvl":
        pitch = max(pitch, fs * rng.uniform(1.4, 2.0) + ruby_space)
    pl, pr = pad(8, 48), pad(8, 48)
    pt, pb = pad(6, 26), pad(6, 28)
    if lay == "nvl":
        pl, pt = pad(16, 70), pad(10, 40)
    out_l = out_r = out_t = out_b = 0
    show_border = lay in ("adv", "balloon", "choice") and rng.random() < 0.55
    if show_border or lay in ("bare", "balloon", "choice"):
        out_l, out_r = pad(0, 26), pad(0, 26)
        out_t, out_b = pad(0, 22), pad(0, 22)
    name_fs = max(10 if psp else 14, int(round(fs * rng.uniform(0.8, 1.02))))
    nf = get_font(path, name_fs * ss, wght) if name_mode in ("tab", "line") else None
    tab = None
    if name_mode == "tab":
        nw = measure(nf, ss, name_disp, 0)
        tab_h = name_fs * rng.uniform(1.35, 1.9)
        tab_w = max(nw + 2 * name_fs * rng.uniform(0.35, 1.4), name_fs * rng.uniform(3, 7))
        overlap = tab_h * rng.uniform(-0.15, 0.55)
        out_t = max(out_t, int(math.ceil(tab_h - overlap)) + pad(0, 8))
        tab = {"w": tab_w, "h": tab_h, "overlap": overlap, "dx": name_fs * rng.uniform(0, 2.5),
               "nw": nw}
    name_line = name_mode == "line"
    name_line_h = (name_fs * 1.25 + pitch - fs) if name_line else 0
    inner_w = max(widths) + (fs if indent else 0) + ((cursor["size"] + fs * 0.8) if cursor else 0)
    if lay in ("adv", "band", "nvl") and rng.random() < 0.6:
        inner_w = max(inner_w, box_w - pl - pr)
    if lay == "choice":
        bar_w = max(widths) + 2 * fs * rng.uniform(0.8, 3.0)
        bar_h = fs * rng.uniform(1.5, 2.1) + 2 * max(sw, sw2)
        bar_gap = fs * rng.uniform(0.25, 1.0)
        w = int(out_l + bar_w + out_r)
        h = int(out_t + len(lines) * bar_h + (len(lines) - 1) * bar_gap + out_b)
    else:
        text_h = name_line_h + ruby_space + (len(lines) - 1) * pitch + fs + 2 * max(sw, sw2) + dy
        box_inner_h = pt + text_h + pb
        w = int(out_l + pl + inner_w + pr + out_r)
        h = int(out_t + box_inner_h + out_b)
    if lay == "balloon":
        h += int(fs * 0.8)          # room for the tail
    if w > SW or h > SH or w < 24 or h < 16:
        return None

    # ---- boxes
    img = background(rng, w, h, platform)
    boxes = []
    rect = (out_l, out_t, w - out_r - 1, h - out_b - 1 - (int(fs * 0.8) if lay == "balloon" else 0))
    box_alpha = float(rng.uniform(0.5, 0.92)) if st["on_dark"] else float(rng.uniform(0.82, 1.0))
    if sch == "on_colour":
        box_alpha = float(rng.uniform(0.7, 0.97))
    border = None
    if show_border:
        bw = 1 if psp else int(rng.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.2, 0.1]))
        bc = light_colour(rng, 0.5, 0.75) if st["on_dark"] else (dark_colour(rng, 0.4) if rng.random() < 0.5
                                                                  else accent(rng, False))
        border = (bw, bc)
    pat = float(rng.uniform(4, 14)) if rng.random() < 0.2 else 0.0
    col2 = None
    if rng.random() < 0.25:
        col2 = np.clip(st["box"] + rng.normal(0, 25, 3), 0, 255)
    if lay in ("adv", "nvl"):
        radius = 0 if rng.random() < 0.4 else int(rng.uniform(4, 22) / (2 if psp else 1))
        alpha = box_alpha
        if lay == "nvl":
            alpha = float(rng.uniform(0.35, 0.75))
        elif rng.random() < 0.18:
            alpha = (float(rng.uniform(0.15, 0.6)) * box_alpha, box_alpha)
        boxes.append({"shape": "round" if radius else "rect", "rect": rect, "radius": radius,
                      "colour": st["box"], "colour2": col2, "alpha": alpha, "pattern": pat,
                      "border": border})
    elif lay == "band":
        a0 = float(rng.uniform(0.0, 0.7)) * box_alpha if rng.random() < 0.5 else box_alpha
        boxes.append({"shape": "rect", "rect": (-2, rect[1], w + 2, h + 2), "colour": st["box"],
                      "colour2": col2, "alpha": (a0, box_alpha), "pattern": pat, "border": None})
    elif lay == "balloon":
        x0, y0, x1, y1 = rect
        tx = rng.uniform(x0 + (x1 - x0) * 0.2, x0 + (x1 - x0) * 0.8)
        tail = [(tx - fs * 0.5, y1 - 2), (tx + fs * 0.5, y1 - 2), (tx + rng.uniform(-1, 1) * fs, y1 + fs * 0.75)]
        boxes.append({"shape": "ellipse" if rng.random() < 0.3 else "round", "rect": rect,
                      "radius": int((y1 - y0) * rng.uniform(0.2, 0.5)), "colour": st["box"],
                      "alpha": float(rng.uniform(0.9, 1.0)), "pattern": 0.0, "tail": tail,
                      "border": (1 if psp else int(rng.integers(2, 4)), dark_colour(rng, 0.2))})
    elif lay == "choice":
        radius = int(rng.uniform(0, 0.5) * bar_h)
        hl = int(rng.integers(len(lines))) if rng.random() < 0.5 else -1
        for i in range(len(lines)):
            y0 = out_t + i * (bar_h + bar_gap)
            c = st["box"] if i != hl else np.clip(st["box"] + rng.normal(0, 45, 3), 0, 255)
            boxes.append({"shape": "round" if radius else "rect",
                          "rect": (out_l, y0, w - out_r - 1, y0 + bar_h), "radius": radius,
                          "colour": c, "alpha": box_alpha, "pattern": 0.0, "border": border})
    for b in boxes:
        draw_box(img, rng, b, st)
    if tab is not None:
        tab_c = st["box"] if rng.random() < 0.3 else (dark_colour(rng, 0.35) if rng.random() < 0.4 else hsv(
            rng.random(), rng.uniform(0.3, 0.9), rng.uniform(0.35, 0.95)))
        tx0 = out_l + tab["dx"]
        ty1 = out_t + tab["overlap"]
        tab["rect"] = (tx0, ty1 - tab["h"], min(w - 1, tx0 + tab["w"]), ty1)
        tr = int(rng.uniform(0, 0.5) * tab["h"])
        draw_box(img, rng, {"shape": "round" if tr else "rect", "rect": tab["rect"], "radius": tr,
                            "colour": tab_c, "colour2": None, "alpha": float(rng.uniform(0.8, 1.0)),
                            "pattern": 0.0, "border": border if rng.random() < 0.5 else None}, st)
        tab["c"] = tab_c

    # ---- text
    layer = Layer(w, h, ss, aa, n_fill, sw, sw2)
    layer.cursor_k = cursor_k
    layout = []
    x_left = out_l + pl
    y = out_t + pt + ruby_space
    if name_line:
        nx = x_left
        base = y + 0.88 * name_fs
        for ch in name_disp:
            layer.glyph(nx, base, ch, nf, 1 if name_coloured else 0)
            nx += nf.getlength(ch) / ss
        if name_coloured:
            n_fill = max(n_fill, 2)
        layout.append({"text": name_label, "top": y, "fs": name_fs, "x0": x_left, "x1": nx,
                       "kind": "name"})
        y += name_line_h
    if lay == "choice":
        tops = [out_t + i * (bar_h + bar_gap) + (bar_h - fs) / 2 for i in range(len(lines))]
    else:
        tops = [y + i * pitch for i in range(len(lines))]
    cursor_xy = None
    ink_extra = []
    for i, ln in enumerate(lines):
        top = tops[i]
        base = top + 0.88 * fs
        if lay in ("choice", "balloon"):
            x = (w - widths[i]) / 2
        else:
            x = x_left + (fs if indent and i > 0 else 0)
        xs = []
        for k, ch in enumerate(ln):
            xs.append(x)
            if ch != "　":
                layer.glyph(x, base, ch, f, colour_of[i][k])
            x += f.getlength(ch) / ss + track
        x_end = x - track
        if ruby:
            draw_ruby(rng, layer, ln, xs, f, ss, top, ruby_fs, ruby_gap, path, wght)
        if boten and i == 0:
            draw_boten(rng, layer, ln, xs, f, ss, top, fs)
        if partial is not None and i == len(lines) - 1 and partial["ghost"]:
            ink_extra.append((x, base, partial["rest"]))
        if cursor is not None and i == len(lines) - 1:
            s = cursor["size"]
            if cursor["inline"]:
                cx = x_end + fs * rng.uniform(0.15, 0.6) + s / 2
                cy = top + fs * rng.uniform(0.45, 0.8)
            else:
                cx = w - out_r - pr * rng.uniform(0.2, 0.8) - s / 2
                cy = h - out_b - pb * rng.uniform(0.3, 0.9) - s / 2
            if cx + s / 2 < w - 1 and cy + s / 2 < h - 1:
                cursor_xy = (cx, cy)
                draw_cursor(layer, cursor["kind"], cx, cy, s)
                if cursor["inline"]:
                    x_end = cx + s / 2
        layout.append({"text": labels[i], "top": top, "fs": fs, "x0": xs[0] if xs else x, "x1": x_end,
                       "kind": "choice" if lay == "choice" else "dialogue",
                       "ruby": bool(ruby and any(is_kanji(c) for c in ln)),
                       "cursor": bool(cursor_xy is not None and cursor["inline"] and i == len(lines) - 1),
                       "coloured": bool(any(colour_of[i]))})
    fills, s1, s2 = layer.arrays()
    cols = [st["fill"], st["accent"]] + [st["fill"]] * max(0, len(fills) - 2)
    if cursor is not None:
        cols[cursor_k] = st["fill"] if cursor["same"] else accent(rng, st["on_dark"])
    grad_rows = None
    if st["grad"] is not None:
        t = np.full(h, 0.5, np.float32)
        for L in layout:
            r0, r1 = int(L["top"]), int(L["top"] + L["fs"])
            t[max(r0, 0):max(r1, 0)] = np.linspace(0, 1, max(r1, 0) - max(r0, 0), dtype=np.float32)
        grad_rows = (np.asarray(st["fill"], np.float32)[None, :] * (1 - t[:, None])
                     + np.asarray(st["grad"], np.float32)[None, :] * t[:, None])[:, None, :]
    core, ink = compose_text(img, fills, s1, s2, cols, st, grad_rows)
    for gx, gbase, ch in ink_extra:          # faint typewriter glyph: drawn, never inked or labelled
        g = Image.new("L", (w * ss, h * ss), 0)
        gd = ImageDraw.Draw(g)
        gd.text((gx * ss, gbase * ss), ch, font=f, fill=255, anchor="ls")
        if ss > 1:
            g = g.resize((w, h), Image.BOX)
        over(img, np.asarray(g, np.float32) / 255.0, st["fill"], partial["ghost_a"])

    if tab is not None:      # the name tab's text: its own colours, contrasting the tab
        tl = Layer(w, h, ss, aa, 1, sw if rng.random() < 0.5 else 0, 0)
        on_tab_dark = luma(tab["c"]) < 140
        if rng.random() < 0.5:
            nx = tab["rect"][0] + (tab["w"] - tab["nw"]) / 2
        else:
            nx = tab["rect"][0] + name_fs * 0.5
        ntop = tab["rect"][1] + (tab["h"] - name_fs) / 2
        nbase = ntop + 0.88 * name_fs
        x = nx
        for ch in name_disp:
            if ch != "　":
                tl.glyph(x, nbase, ch, nf)
            x += nf.getlength(ch) / ss
        nfill = (light_colour(rng, 0.1, 0.9) if on_tab_dark else dark_colour(rng, 0.25)) \
            if not name_coloured else accent(rng, on_tab_dark)
        tst = {"shadow": None, "glow": None,
               "stroke": {"w": tl.sw, "c": dark_colour(rng, 0.2) if on_tab_dark else light_colour(rng)},
               "stroke2": None}
        tf, ts1, _ = tl.arrays()
        tcore, tink = compose_text(img, tf, ts1, None, [nfill], tst)
        np.maximum(core, tcore, out=core)
        np.maximum(ink, tink, out=ink)
        layout.insert(0, {"text": name_label, "top": ntop, "fs": name_fs, "x0": nx, "x1": x, "kind": "name"})

    # ---- finder geometry: an ideal finder sees either the text core (like the overlay's
    # bright finder on outlined or shadowed text) or the full ink with outline and shadow
    geom = "core" if rng.random() < 0.5 else "full"
    if geom == "core":
        ink = core

    # ---- screen artefacts
    up2x = platform == "psp2x"
    scale = 1.0
    if platform == "vita" and rng.random() < 0.12:       # sub-native rendering, GPU upscale
        scale = float(rng.uniform(0.6, 0.9))
        small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 0.8)))
    if rng.random() < 0.2:
        img = img + rng.normal(0, rng.uniform(1, 4), img.shape).astype(np.float32)
    out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    if up2x:
        interp = cv2.INTER_NEAREST if rng.random() < 0.4 else cv2.INTER_LINEAR
        out = cv2.resize(out, (w * 2, h * 2), interpolation=interp)
        ink = cv2.resize(ink, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        for L in layout:
            for key in ("top", "x0", "x1"):
                L[key] *= 2
            L["fs"] *= 2
        sw, sw2, dy, ruby_space = 2 * sw, 2 * sw2, 2 * dy, 2 * ruby_space
    q = 0
    if rng.random() < (0.6 if psp else 0.5):
        q = int(np.clip(rng.beta(5, 1.6) * 100, 35, 97))
        out = jpeg(out, q, rng)
        if rng.random() < 0.1:
            out = jpeg(out, int(rng.integers(60, 95)), rng)

    ink_b = ink >= 0.5
    for L in layout:     # vertical band of each line's own text ink (furigana excluded)
        L["y0"] = L["top"] - max(sw, sw2) - 1
        L["y1"] = L["top"] + L["fs"] + max(sw, sw2) + dy + 1
        rows = ink_b[max(int(L["y0"]), 0):max(int(L["y1"]), 0)]
        cols = np.flatnonzero(rows.any(axis=0)) if rows.size else []
        L["ix0"], L["ix1"] = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (-99, -99)
    meta = {"platform": platform, "layout": lay, "scheme": sch, "cat": cat, "fam": fam, "fs": fs,
            "aa": bool(aa), "sw": sw // (2 if up2x else 1), "shadow": bool(sh), "glow": bool(st["glow"]),
            "grad": st["grad"] is not None, "name": name_mode, "partial": partial is not None,
            "cursor": None if cursor is None else ("inline" if cursor["inline"] else "corner"),
            "jpeg": q, "scale": round(scale, 2), "border": bool(show_border), "geom": geom}
    return out, layout, ink_b, meta


def draw_ruby(rng, layer, ln, xs, f, ss, top, ruby_fs, gap, path, wght):
    rf = get_font(path, ruby_fs * ss, wght)
    i = 0
    while i < len(ln):
        if not is_kanji(ln[i]):
            i += 1
            continue
        j = i
        while j < len(ln) and is_kanji(ln[j]):
            j += 1
        if rng.random() < 0.6:
            kana = R.hiragana if rng.random() < 0.85 else R.katakana
            reading = "".join(rng.choice(kana, size=(j - i) * int(rng.integers(1, 3))))
            span0, span1 = xs[i], xs[j - 1] + f.getlength(ln[j - 1]) / ss
            width = sum(rf.getlength(c) for c in reading) / ss
            x = (span0 + span1 - width) / 2
            base = top - gap - 0.12 * ruby_fs
            for ch in reading:
                layer.glyph(x, base, ch, rf, 0, small=True)
                x += rf.getlength(ch) / ss
        i = j


def draw_boten(rng, layer, ln, xs, f, ss, top, fs):
    """Emphasis dots (傍点) over a short run of characters."""
    if len(ln) < 3:
        return
    a = int(rng.integers(0, len(ln) - 2))
    b = min(len(ln), a + int(rng.integers(2, 6)))
    r = max(1.0, fs * 0.07)
    for k in range(a, b):
        cx = xs[k] + f.getlength(ln[k]) / ss / 2
        cy = top - fs * 0.16
        layer.poly([(cx - r, cy - r), (cx + r, cy + r)], ellipse=True)


def jpeg(arr, q, rng):
    """JPEG round trip with a random 8x8 grid phase."""
    oy, ox = int(rng.integers(0, 8)), int(rng.integers(0, 8))
    h, w = arr.shape[:2]
    pad = np.pad(arr, ((oy, 0), (ox, 0), (0, 0)), mode="edge")
    buf = io.BytesIO()
    Image.fromarray(pad).save(buf, "JPEG", quality=int(q))
    return np.asarray(Image.open(buf).convert("RGB"))[oy:oy + h, ox:ox + w]


# ----------------------------------------------------------------------------- crops

def extract(img, layout, ink=None, finder="mask", rng=None):
    """Match finder detections to rendered lines; return [(grey crop, label, box, line)]."""
    if finder == "mask":
        dets = find_lines_mask(ink)
    elif finder == "grad":
        dets = find_lines_grad(img)
    else:
        dets = find_lines(img)
    if dets is None:
        return None
    out = []
    if not len(dets) or not layout:
        return out
    h_img, w_img = img.shape[:2]
    rows = [(y + (4 if y > 0 else 0), y + h - (5 if y + h < h_img else 0)) for (x, y, w, h) in dets]
    ov = np.array([[min(b, L["y1"]) - max(a, L["y0"]) for L in layout] for a, b in rows], np.float32)
    for d, (x, y, w, h) in enumerate(dets):
        li = int(ov[d].argmax())
        L = layout[li]
        if ov[d, li] <= 0 or int(ov[:, li].argmax()) != d:
            continue
        if (np.delete(ov[d], li) > 1).any():           # spans two rendered lines
            continue
        if ov[d, li] < 0.5 * min(b - a for a, b in [rows[d]] + [(L["y0"], L["y1"])]):
            continue
        if finder != "mask":
            # The crop must hug this line's own ink: no cut-off line, and no bright
            # background text pulled into the crop beside it.
            if x > 0 and abs(x + 4 - L["ix0"]) > 3:
                continue
            if x + w < w_img and abs(x + w - 4 - L["ix1"]) > 3:
                continue
        if rng is not None and rng.random() < 0.3:      # finder-to-finder geometry jitter
            j = rng.integers(-1, 4, 4)
            x0, y0 = max(0, x - int(j[0])), max(0, y - int(j[1]))
            x1, y1 = min(w_img, x + w + int(j[2])), min(h_img, y + h + int(j[3]))
            if x1 - x0 > 8 and y1 - y0 > 8:
                x, y, w, h = x0, y0, x1 - x0, y1 - y0
        out.append((to_gray(img[y:y + h, x:x + w]), L["text"], (x, y, w, h), L))
    return out


def crop_meta(meta, L):
    m = dict(meta)
    m["kind"] = L["kind"]
    for k in ("ruby", "cursor", "coloured"):
        m[k] = bool(L.get(k, False))
    return m


def init(pool_limit=300_000):
    global R
    R = Resources(pool_limit)


def run_shard(job):
    idx, n_lines, seed, out_dir, finder = job
    cv2.setNumThreads(1)        # workers stay single-threaded (heavy.sh gives the job 5 threads)
    path = Path(out_dir) / f"synth_{idx:05d}.npz"
    if path.exists():
        return idx, 0, 0, 0, 0.0
    rng = np.random.default_rng(seed + idx * 1_000_003)
    t0, crops, labels, metas, regions, errors = time.time(), [], [], [], 0, 0
    while len(crops) < n_lines:
        try:
            r = render_region(rng)
        except Exception as e:  # noqa: BLE001 - a rare bad draw must not kill a long render
            errors += 1
            if errors <= 3:
                print(f"shard {idx}: render error {type(e).__name__}: {e}", flush=True)
            if errors > 1000:
                raise
            continue
        if r is None:
            continue
        img, layout, ink, meta = r
        got = extract(img, layout, ink, finder, rng)
        regions += 1
        for g, label, _, L in got or []:
            crops.append(g)
            labels.append(label)
            metas.append(crop_meta(meta, L))
    crops, labels, metas = crops[:n_lines], labels[:n_lines], metas[:n_lines]
    shapes = np.array([c.shape for c in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=np.concatenate([c.ravel() for c in crops]), offsets=offsets,
                        shapes=shapes, labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(metas)))
    tmp.rename(path)
    return idx, len(crops), regions, errors, time.time() - t0


def preview(n, finder="mask", seed=7, out_dir=HERE):
    """Contact sheets (both <= 1280 px wide and tall) of regions and their crops."""
    rng = np.random.default_rng(seed)
    label_font = ImageFont.truetype(str(sorted((DATA / "fonts" / "notosansjp").glob("*.ttf"))[0]), 16)
    regions, crops = [], []
    while len(regions) < n:
        r = render_region(rng)
        if r is None:
            continue
        img, layout, ink, meta = r
        got = extract(img, layout, ink, finder)
        if not got:
            continue
        vis = Image.fromarray(img)
        d = ImageDraw.Draw(vis)
        for _, _, (x, y, w, h), _ in got:
            d.rectangle((x, y, x + w - 1, y + h - 1), outline=(0, 255, 0))
        regions.append(vis)
        crops += [(g, lab, crop_meta(meta, L)) for g, lab, _, L in got]
    sheets, cur, y = [], [], 0
    for im in regions:
        im = im if im.width <= 1280 else im.resize((1280, im.height * 1280 // im.width))
        if y + im.height > 1280 and cur:
            sheets.append(cur)
            cur, y = [], 0
        cur.append(im)
        y += im.height + 4
    sheets.append(cur)
    for s, ims in enumerate(sheets[:4]):
        sheet = Image.new("RGB", (1280, min(1280, sum(im.height + 4 for im in ims))), (40, 40, 40))
        yy = 0
        for im in ims:
            sheet.paste(im, (0, yy))
            yy += im.height + 4
        sheet.save(Path(out_dir) / f"preview_regions{'' if s == 0 else s}.png")
    rows = []
    for g, lab, meta in crops[:36]:
        g32 = Image.fromarray(g).resize((max(8, round(g.shape[1] * 32 / g.shape[0])), 32), Image.BILINEAR)
        g32 = g32.crop((0, 0, min(g32.width, 760), 32))
        row = Image.new("RGB", (1280, 34), (0, 0, 0))
        row.paste(g32.convert("RGB"), (0, 1))
        tag = f"{lab[:22]} [{meta['platform']} {meta['layout']} {meta['scheme']} {meta['fam']} {meta['fs']}]"
        ImageDraw.Draw(row).text((770, 6), tag, font=label_font, fill=(255, 255, 160))
        rows.append(row)
    sheet = Image.new("RGB", (1280, 34 * len(rows)), (0, 0, 0))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, 34 * i))
    sheet.save(Path(out_dir) / "preview_crops.png")
    print(f"preview: {len(regions)} regions, {len(crops)} crops", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preview", "shards"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=DATA / "synth2")
    ap.add_argument("--shards", type=int, default=40)
    ap.add_argument("--per-shard", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--finder", choices=["mask", "grad", "bright"], default="mask")
    args = ap.parse_args()
    cv2.setNumThreads(1)
    init()
    print(f"fonts {len(R.fonts)} in {sorted(R.by_cat)}, bg vita {len(R.bg_vita)} psp {len(R.bg_psp)}, "
          f"vocab {len(R.vocab)}", flush=True)
    if args.mode == "preview":
        preview(args.n, args.finder, args.seed)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.per_shard, args.seed, str(args.out), args.finder) for i in range(args.shards)]
    t0, total = time.time(), 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, regions, errors, dt in pool.imap_unordered(run_shard, jobs):
            total += n
            print(f"shard {idx}: {n} lines from {regions} regions ({errors} errors) in {dt:.0f}s "
                  f"(total {total}, {time.time() - t0:.0f}s)", flush=True)
    print("SHARDS_DONE", flush=True)


if __name__ == "__main__":
    main()

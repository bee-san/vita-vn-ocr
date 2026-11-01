#!/usr/bin/env python3
"""1c: synthetic desktop frames at native desktop resolutions (docs/desktop-data.md).

  synth.py render   --out DIR --shards N [--per-shard 250] [--first F] [--seed S] [--workers W]
  synth.py manifest --dir DIR --out data/manifests/desktop_data/synth_desktop_v1.json

Each frame is a whole PC VN screen, drawn at the screen's own resolution (1280 x 720 to 3840 x 2160, 16:9, 16:10,
21:9 and 4:3), its text rasterised at that size (no upscale):
  scene      synth/textspot_data.load_scene: a training-side VNDB screenshot from textspot's clean scene pool,
             cover-cropped to the frame, its known text inpainted away (or synth2's synthetic background)
  dialogue   an ADV box at the bottom (or the top), a full-screen NVL page, a band, or text with no box at all, over
             the scene; boxes translucent (alpha 0.15-0.95) or opaque, rounded or square, some with a border; 1-4
             lines (NVL 3-10), a name plate, a name line, an inline 【name】 or no name; synth3's dialogue,
             narration and name sources (synth/synth3.py, its `personal` profile's text pools)
  fonts      synth3's font pool without its pixel (bitmap) families: gothic, mincho, rounded, display and pen faces,
             every one OFL or similar ("open" licence class, data/manifests/synth3_fonts.json); about 22% thin
             weights (100-300) where a family has them, else regular to black
  sizes      the main text's em 24-64 px: 2.8-5.0% of the screen height, clipped to [24, 64], or (15%) uniform
             in [24, 64] on any screen
  effects    outlines 1-5 px (about a third of the lines, most of the box-less ones), hard or soft shadows,
             coloured names and spans, furigana over some words (an ignore area), page cursors (background)
  choices    2-4 bars of in-story choices (class choice); UI labels at the edges: quick-menu words, status and
             chapter labels, 14-28 px (class ui)
  empty      7% text-free frames: the scene, or the scene with an empty box
  guard      MeikiOCR's detector (threshold 0.3) on the finished frame: a box that meets no labelled line and no
             ignore area becomes an ignore area (scene text the inpainting missed is never taught as background)
  capture    70% native (the game runs at the screen's resolution; 20% of those saved as JPEG q 75-95); 30% drawn at
             a game resolution (800 x 600 to 1920 x 1080) and put through desktopdata/capture.py's capture model to
             a screen (letterbox / pillarbox, resampler, DPI, downscale, JPEG), boxes mapped exactly
Boxes are each instance's ink box from its own coverage (fill, outline, hard shadow; synth/textspot_data's rule:
coverage >= 0.5, or half its peak for a thin face), an instance under 6 px across an ignore area. Shards are #97's
frames (train/textspot_train.FramePool reads them as source synth): DIR/desk_synth_NNNN.{npy,json}, src
"synth:desktop", kind "pc", lines [x0, y0, x1, y1, vertical, role].
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from desktopdata import capture as CAP  # noqa: E402
from desktopdata import shards as SH  # noqa: E402

GEN_VERSION = 1
MIN_ACROSS = 6
MIX = {
    "screen": {"1280x720": 0.15, "1366x768": 0.05, "1600x900": 0.05, "1920x1080": 0.35, "2560x1440": 0.12,
               "3840x2160": 0.04, "1280x800": 0.03, "1440x900": 0.02, "1920x1200": 0.05, "2560x1080": 0.03,
               "800x600": 0.03, "1024x768": 0.05, "1280x960": 0.03},
    "game": {"800x600": 0.15, "1024x576": 0.1, "1024x768": 0.1, "1280x720": 0.4, "1600x900": 0.1, "1920x1080": 0.15},
    "captured": 0.30,
    "native_jpeg": {"p": 0.2, "quality": [75, 95]},
    "empty": 0.07,
    "em": {"frac": [0.028, 0.050], "clip": [24, 64], "uniform_p": 0.15},
    "layout": {"adv": 0.52, "adv_top": 0.03, "nvl": 0.13, "band": 0.08, "bare": 0.18, "none": 0.06},
    "box_alpha": {"translucent": [0.15, 0.8], "opaque": [0.8, 0.97], "p_translucent": 0.65},
    "category": {"gothic": 0.42, "mincho": 0.28, "rounded": 0.14, "display": 0.08, "pen": 0.08},
    "thin_p": 0.22,
    "outline": {"box": 0.25, "bare": 0.85, "width_em": [0.03, 0.09]},
    "shadow": {"p": 0.3, "hard_p": 0.6},
    "name": {"tab": 0.3, "line": 0.25, "inline": 0.1, "none": 0.35},
    "furigana": 0.12,
    "cursor": 0.3,
    "choice": {"with_main": 0.08, "without_main": 0.6},
    "ui_n": [0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05],
    "ui_em": [14, 28],
    "scenes": {"hires": 0.92, "synthetic": 0.08},
}
TS = None
S3 = None
GUARD_ON = True


def choose(rng, w):
    return CAP.choose(rng, w)


def init(scenes=True):
    global TS, S3
    import textspot_data
    textspot_data.init(scenes=scenes)
    TS, S3 = textspot_data, textspot_data.S3
    return TS


# ----------------------------------------------------------------------------- fonts and text

def pick_font(rng, chars, cat=None):
    """(Font, wght or None, thin): a non-pixel synth3 font file covering chars, by MIX['category']."""
    R = S3.R
    for _ in range(6):
        c = cat or choose(rng, {k: v for k, v in MIX["category"].items() if k in R.by_cat})
        fams = [f for f in R.by_cat[c].values() if any(chars <= x.chars for x in f)]
        if not fams:
            cat = None
            continue
        fam = fams[int(rng.integers(len(fams)))]
        files = [x for x in fam if chars <= x.chars]
        thin = [x for x in files if x.weight and x.weight <= 300]
        if thin and rng.random() < MIX["thin_p"]:
            f = thin[int(rng.integers(len(thin)))]
            return f, None, True
        normal = [x for x in files if not x.weight or x.weight > 300] or files
        w = np.array([1.0 + (0.8 if x.weight and 500 <= x.weight <= 750 else 0.0) for x in normal])
        f = normal[int(rng.choice(len(normal), p=w / w.sum()))]
        wght = S3.S2.pick_weight(rng, f.path)
        return f, wght, bool(wght is not None and wght <= 300)
    return None, None, False


def get_font(f, size, wght):
    return S3.S2.get_font(f.path, int(size), wght)


def text_masks(text, font, stroke=0, shadow=None):
    """Float masks (fill, outline, shadow) of one line drawn left-baseline at (pad, pad + ascent) in a tight local
    canvas, and the canvas' offset of the pen origin: (fill, outline or None, shadow or None, (ox, oy), advance)."""
    from PIL import Image, ImageDraw
    asc, desc = font.getmetrics()
    adv = font.getlength(text)
    pad = stroke + 4 + (max(abs(shadow["dx"]), abs(shadow["dy"])) + int(math.ceil(2 * shadow["blur"])) if shadow
                        else 0)
    w, h = int(math.ceil(adv)) + 2 * pad + 4, asc + desc + 2 * pad + 4
    org = (pad, pad + asc)
    fill = Image.new("L", (w, h), 0)
    ImageDraw.Draw(fill).text(org, text, font=font, fill=255, anchor="ls")
    out = None
    if stroke:
        out = Image.new("L", (w, h), 0)
        ImageDraw.Draw(out).text(org, text, font=font, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
    sh = None
    if shadow:
        import cv2
        base = np.asarray(out if out is not None else fill, np.float32) / 255.0
        m = np.zeros_like(base)
        dx, dy = shadow["dx"], shadow["dy"]
        H, W = base.shape
        m[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = \
            base[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        if shadow["blur"] > 0:
            m = cv2.GaussianBlur(m, (0, 0), shadow["blur"])
        sh = m
    f = np.asarray(fill, np.float32) / 255.0
    o = None if out is None else np.asarray(out, np.float32) / 255.0
    return f, o, sh, org, adv


def composite(img, x, y, m, colour, alpha=1.0):
    """img (float32 H x W x 3) <- colour over it with coverage m at (x, y) (clipped to the frame)."""
    H, W = img.shape[:2]
    h, w = m.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = (m[y0 - y:y1 - y, x0 - x:x1 - x] * alpha)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + np.asarray(colour, np.float32) * a


def draw_text(fr, rng, x, base, text, font, style, role, add=True):
    """One line of text with its pen origin at (x, base) in frame px: composited, and (add) its coverage added as an
    instance of `role`. Returns (x0, y0, x1, y1) of what was drawn, or None when it does not fit the frame."""
    stroke = style.get("stroke", 0)
    sh = style.get("shadow")
    f, o, s, (ox, oy), adv = text_masks(text, font, stroke, sh)
    X, Y = int(round(x)) - ox, int(round(base)) - oy
    h, w = f.shape
    if X < 0 or Y < 0 or X + w > fr.W or Y + h > fr.H:
        return None
    if s is not None:
        composite(fr.img, X, Y, s, sh["c"], sh["a"])
    if o is not None:
        composite(fr.img, X, Y, o, style["stroke_c"])
    composite(fr.img, X, Y, f, style["fill"])
    if add:
        cov = f if o is None else np.maximum(f, o)
        if s is not None and sh["blur"] == 0:
            cov = np.maximum(cov, s * sh["a"])
        fr.add(X, Y, cov, 0, role)
    ys, xs = np.nonzero(np.maximum(f, o if o is not None else 0) > 0.1)
    if not len(ys):
        return None
    return X + int(xs.min()), Y + int(ys.min()), X + int(xs.max()) + 1, Y + int(ys.max()) + 1


def draw_box(img, rect, colour, alpha, radius=0, border=None, gradient=None):
    """A filled (rounded) rectangle, alpha blended; gradient (a0, a1): alpha from top to bottom."""
    import cv2
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    r = int(min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    if r > 0:
        cv2.rectangle(m, (r, 0), (m.shape[1] - 1 - r, m.shape[0] - 1), 255, -1)
        cv2.rectangle(m, (0, r), (m.shape[1] - 1, m.shape[0] - 1 - r), 255, -1)
        w1, h1 = m.shape[1] - 1 - r, m.shape[0] - 1 - r
        for cx, cy in ((r, r), (w1, r), (r, h1), (w1, h1)):
            cv2.circle(m, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
    else:
        m[:] = 255
    a = m.astype(np.float32) / 255.0
    if gradient:
        a = a * np.linspace(gradient[0], gradient[1], a.shape[0], dtype=np.float32)[:, None]
    else:
        a = a * alpha
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a[..., None]) + np.asarray(colour, np.float32) * a[..., None]
    if border:
        bw, bc = border
        e = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((2 * bw + 1, 2 * bw + 1), np.uint8))
        eb = (e.astype(np.float32) / 255.0)[..., None]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - eb) + np.asarray(bc, np.float32) * eb


def em_of(rng, W, H):
    lo, hi = MIX["em"]["clip"]
    if rng.random() < MIX["em"]["uniform_p"]:
        return int(rng.integers(lo, hi + 1))
    return int(np.clip(round(H * rng.uniform(*MIX["em"]["frac"])), lo, hi))


def text_style(rng, on_box, box_luma=None):
    """Fill / outline / shadow colours and widths for text on a box of luma box_luma, or on the scene (bare)."""
    S2 = S3.S2
    dark_text = on_box and box_luma is not None and box_luma > 150
    fill = S2.text_dark(rng) if dark_text else S2.text_light(rng)
    st = {"fill": fill}
    p_out = MIX["outline"]["box" if on_box else "bare"]
    if rng.random() < p_out:
        st["stroke_frac"] = float(rng.uniform(*MIX["outline"]["width_em"]))
        st["stroke_c"] = S2.light_colour(rng, 0.15, 0.9) if dark_text else S2.dark_colour(rng, 0.25)
    if rng.random() < MIX["shadow"]["p"] or (not on_box and "stroke_frac" not in st):
        hard = rng.random() < MIX["shadow"]["hard_p"]
        st["shadow_frac"] = float(rng.uniform(0.04, 0.09))
        st["shadow"] = {"c": S2.dark_colour(rng, 0.15) if not dark_text else S2.light_colour(rng, 0.1, 0.85),
                        "a": float(rng.uniform(0.55, 1.0)), "blur": 0.0 if hard else float(rng.uniform(0.8, 2.0))}
    return st


def sized_style(st, em):
    out = dict(st)
    out["stroke"] = max(1, int(round(st["stroke_frac"] * em))) if "stroke_frac" in st else 0
    if "shadow" in st:
        d = max(1, int(round(st["shadow_frac"] * em)))
        out["shadow"] = dict(st["shadow"], dx=d, dy=d)
    return out


# ----------------------------------------------------------------------------- elements

def furigana(fr, rng, x0, base, line, font, em, style, fam_font):
    """Ruby over a run of kanji of a drawn line: small kana above it (an ignore area, as on the eval sets)."""
    import re
    runs = [(m.start(), m.end()) for m in re.finditer(r"[\u4e00-\u9fff]{1,4}", line)]
    if not runs:
        return
    a, b = runs[int(rng.integers(len(runs)))]
    pool = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわん")
    kana = "".join(rng.choice(pool, size=int(rng.integers(2, 2 * (b - a) + 2))))
    rs = max(9, int(round(em * rng.uniform(0.42, 0.55))))
    rf = get_font(fam_font[0], rs, fam_font[1])
    xa = x0 + font.getlength(line[:a])
    width = font.getlength(line[a:b])
    rw = rf.getlength(kana)
    rx = xa + (width - rw) / 2
    rbase = base - em * 0.92 - rs * 0.15
    rst = dict(style, stroke=max(0, style["stroke"] // 2), shadow=None)
    bb = draw_text(fr, rng, rx, rbase, kana, rf, rst, None, add=False)
    if bb:
        fr.add_ignore(bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2)


def wrap_lines(rng, text, cpl, max_lines):
    S2 = S3.S2
    return S2.wrap(S2.add_spaces(text, rng), max(4, cpl), max_lines, rng, at_punct=rng.random() < 0.35)


def dialogue_element(fr, rng, layout):
    """A dialogue box (or bare text) with its lines, name and effects. True when a line was added."""
    S2 = S3.S2
    W, H = fr.W, fr.H
    em = fr.em
    named = layout in ("adv", "adv_top", "band", "bare") and choose(rng, MIX["name"]) != "none"
    name_mode = choose(rng, {k: v for k, v in MIX["name"].items() if k != "none"}) if named else None
    if layout == "nvl":
        text, _, _ = S3.narration_text(rng) if rng.random() < 0.6 else S3.dialogue_text(rng, False)
        more = [S3.narration_text(rng)[0] for _ in range(int(rng.integers(1, 4)))]
        text = "".join([text] + more)
    elif rng.random() < 0.25:
        text, _, _ = S3.narration_text(rng)
    else:
        text, _, _ = S3.dialogue_text(rng, named)
        if named and rng.random() < 0.7:
            text = "「" + text + "」"
    if not text:
        return False
    disp = label = None
    if named:
        disp, label, _ = S3.sample_name(rng)
        if not label:
            named, name_mode = False, None
    chars = set(text) | (set(disp) if disp else set())
    f, wght, thin = pick_font(rng, chars - {"　"})
    if f is None:
        return False
    font = get_font(f, em, wght)
    on_box = layout not in ("bare",)
    box_c = None
    if on_box:
        box_c = S2.dark_colour(rng, 0.3) if rng.random() < 0.72 else S2.light_colour(rng, 0.2, 0.8)
    st = sized_style(text_style(rng, on_box, S2.luma(box_c) if box_c is not None else None), em)
    pitch = em * rng.uniform(1.25, 1.7)
    if layout == "nvl":
        bw = W * rng.uniform(0.82, 0.96)
        max_lines = int(np.clip((H * 0.8) // pitch, 3, 10))
    elif layout == "band":
        bw = W
        max_lines = int(rng.integers(1, 4))
    else:
        bw = W * rng.uniform(0.6, 0.98)
        max_lines = int(rng.choice([1, 2, 3, 4], p=[0.2, 0.35, 0.33, 0.12]))
    pad_x = em * rng.uniform(0.6, 2.0)
    pad_y = em * rng.uniform(0.4, 1.2)
    cpl = int((bw - 2 * pad_x) // max(em, 1))
    if layout != "nvl":
        cpl = max(6, int(cpl * rng.uniform(0.75, 1.0)))
    lines = wrap_lines(rng, text, cpl, max_lines)
    if not lines:
        return False
    if name_mode == "inline":
        lines[0] = disp + lines[0]
    name_h = em * 1.4 if name_mode == "line" else 0.0
    ruby_space = em * 0.55 if rng.random() < MIX["furigana"] else 0.0
    text_h = name_h + ruby_space + (len(lines) - 1) * pitch + em * 1.15
    bh = text_h + 2 * pad_y
    widths = [font.getlength(ln) for ln in lines]
    if layout in ("bare",):
        bw = max(widths) + 2 * pad_x
    bw = min(bw, W - 2)
    where = {"adv": "bottom", "band": "bottom", "nvl": "centre", "bare": "any", "adv_top": "upper"}[layout]
    if layout == "bare" and rng.random() < 0.6:
        where = "bottom"
    xy = TS.place(fr, rng, int(math.ceil(bw)), int(math.ceil(bh)), where)
    if xy is None:
        return False
    bx, by = xy
    if layout == "band":
        bx = 0
    fr.reserve(bx, by, bx + bw, by + bh)
    if on_box:
        p_t = MIX["box_alpha"]["p_translucent"]
        a = float(rng.uniform(*MIX["box_alpha"]["translucent" if rng.random() < p_t else "opaque"]))
        grad = (a * rng.uniform(0.0, 0.6), a) if layout == "band" and rng.random() < 0.5 else None
        border = None
        if layout in ("adv", "adv_top") and rng.random() < 0.4:
            border = (int(rng.integers(1, max(2, em // 12) + 1)), S2.light_colour(rng, 0.5, 0.7))
        radius = 0 if rng.random() < 0.35 else int(em * rng.uniform(0.2, 1.0))
        draw_box(fr.img, (bx, by, bx + bw, by + bh), box_c, a, radius, border, grad)
        fr.box_alpha = a
    tab_y = by
    if name_mode == "tab":
        nf = get_font(f, max(14, int(em * rng.uniform(0.8, 1.0))), wght)
        tw = nf.getlength(disp) + em * rng.uniform(0.8, 2.4)
        th = em * rng.uniform(1.3, 1.8)
        tx = bx + em * rng.uniform(0.0, 2.0)
        ty = by - th * rng.uniform(0.5, 1.05)
        if ty >= 0 and tx + tw <= W:
            plate = box_c if box_c is not None and rng.random() < 0.4 else S2.dark_colour(rng, 0.35)
            draw_box(fr.img, (tx, ty, tx + tw, ty + th), plate, float(rng.uniform(0.6, 1.0)), int(th * 0.3))
            nst = sized_style(text_style(rng, True, S2.luma(plate)), nf.size)
            if rng.random() < 0.5:
                nst["fill"] = S2.accent(rng, S2.luma(plate) < 128)
            draw_text(fr, rng, tx + (tw - nf.getlength(disp)) / 2, ty + th * 0.5 + nf.size * 0.38, disp, nf, nst,
                      "name")
            tab_y = ty
    x0 = bx + pad_x
    y = by + pad_y + ruby_space
    added = False
    if name_mode == "line":
        nst = dict(st)
        if rng.random() < 0.5:
            nst["fill"] = S2.accent(rng, box_c is None or S2.luma(box_c) < 128)
        if draw_text(fr, rng, x0, y + em * 0.92, disp, font, nst, "name"):
            added = True
        y += name_h
    indent = em if lines[0][:1] in "「『（" and rng.random() < 0.4 else 0
    for i, ln in enumerate(lines):
        lx = x0 + (indent if i and lines[0][:1] in "「『（" else 0)
        base = y + em * 0.92 + i * pitch
        if not ln.strip("　"):
            continue
        role = "dialogue"
        bb = draw_text(fr, rng, lx, base, ln, font, st, role)
        if bb:
            added = True
            if ruby_space and i == 0:
                furigana(fr, rng, lx, base, ln, font, em, st, (f, wght))
    if added and rng.random() < MIX["cursor"] and layout != "bare":
        last = lines[-1]
        cx = x0 + font.getlength(last) + em * rng.uniform(0.2, 0.6)
        cy = y + em * 0.92 + (len(lines) - 1) * pitch
        cur = "▼▽◆◇▲" [int(rng.integers(5))]
        cf = get_font(f, max(10, int(em * rng.uniform(0.5, 0.8))), wght) if cur in f.chars else None
        if cf is not None:
            draw_text(fr, rng, cx, cy, cur, cf, dict(st, stroke=0, shadow=None), None, add=False)
    fr.elements.append(f"dialogue:{layout}:{name_mode}:{'thin' if thin else 'reg'}:{f.cat}")
    del tab_y
    return added


def choice_element(fr, rng):
    S2 = S3.S2
    n = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.2]))
    texts = [S3.choice_text(rng)[0] for _ in range(n)]
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return False
    f, wght, _ = pick_font(rng, set("".join(texts)))
    if f is None:
        return False
    em = int(np.clip(fr.em * rng.uniform(0.85, 1.05), 20, 64))
    font = get_font(f, em, wght)
    bar_w = max(font.getlength(t) for t in texts) + 2 * em * rng.uniform(1.0, 3.0)
    bar_h = em * rng.uniform(1.5, 2.1)
    gap = em * rng.uniform(0.3, 1.0)
    h = n * bar_h + (n - 1) * gap
    xy = TS.place(fr, rng, int(math.ceil(bar_w)), int(math.ceil(h)), "centre")
    if xy is None:
        return False
    x, y = xy
    fr.reserve(x, y, x + bar_w, y + h)
    c = S2.dark_colour(rng, 0.35) if rng.random() < 0.7 else S2.light_colour(rng, 0.2, 0.85)
    st = sized_style(text_style(rng, True, S2.luma(c)), em)
    a = float(rng.uniform(0.5, 0.95))
    added = False
    for i, t in enumerate(texts):
        ty = y + i * (bar_h + gap)
        draw_box(fr.img, (x, ty, x + bar_w, ty + bar_h), c, a, int(bar_h * rng.uniform(0, 0.5)))
        if draw_text(fr, rng, x + (bar_w - font.getlength(t)) / 2, ty + bar_h / 2 + em * 0.38, t, font, st,
                     "choice"):
            added = True
    fr.elements.append("choice")
    return added


def ui_element(fr, rng):
    S2 = S3.S2
    kind = rng.random()
    if kind < 0.45:
        words = [TS.QUICK_JP[int(rng.integers(len(TS.QUICK_JP)))] if rng.random() < 0.6 else
                 TS.QUICK_LATIN[int(rng.integers(len(TS.QUICK_LATIN)))] for _ in range(int(rng.integers(1, 6)))]
    else:
        s = TS.STATUS[int(rng.integers(len(TS.STATUS)))]
        n_, m_ = int(rng.integers(1, 30)), int(rng.integers(30, 99))           # the draw order: n, m, nn
        words = [s.format(n=n_, m=m_, nn=f"{int(rng.integers(1, 99)):02d}")]
    em = int(rng.integers(MIX["ui_em"][0], MIX["ui_em"][1] + 1))
    f, wght, _ = pick_font(rng, set("".join(words)) - {" "}, cat="gothic" if rng.random() < 0.7 else None)
    if f is None:
        return False
    font = get_font(f, em, wght)
    gap = em * rng.uniform(1.6, 3.0)
    width = sum(font.getlength(w) for w in words) + gap * (len(words) - 1) + em
    h = em * 1.6
    xy = TS.place(fr, rng, int(math.ceil(width)), int(math.ceil(h)), "edge")
    if xy is None:
        return False
    x, y = xy
    fr.reserve(x, y, x + width, y + h)
    if rng.random() < 0.5:
        draw_box(fr.img, (x, y, x + width, y + h), S2.dark_colour(rng, 0.3), float(rng.uniform(0.4, 0.9)),
                 int(h * 0.3))
    st = sized_style(text_style(rng, True, 40), em)
    cx = x + em / 2
    for w in words:
        draw_text(fr, rng, cx, y + h / 2 + em * 0.38, w, font, st, "ui")
        cx += font.getlength(w) + gap
    fr.elements.append("ui")
    return True


def empty_box(fr, rng):
    S2 = S3.S2
    W, H = fr.W, fr.H
    bw, bh = W * rng.uniform(0.6, 0.98), H * rng.uniform(0.15, 0.3)
    xy = TS.place(fr, rng, int(bw), int(bh), "bottom")
    if xy:
        draw_box(fr.img, (xy[0], xy[1], xy[0] + bw, xy[1] + bh), S2.dark_colour(rng, 0.3),
                 float(rng.uniform(0.3, 0.9)), int(fr.em * rng.uniform(0, 1)))
        fr.elements.append("empty:box")


# ----------------------------------------------------------------------------- frames

def new_frame(rng, W, H):
    fr = TS.Frame("pc", W, H, "vita", H / 544.0, None)
    pools = dict(MIX["scenes"])
    a = W / H
    key = "16:9" if abs(a - 16 / 9) < 0.06 else "4:3" if abs(a - 4 / 3) < 0.06 else "hires"
    if key != "hires":
        pools[key] = pools.pop("hires")
    fr.img, fr.scene = TS.load_scene(rng, W, H, pools)
    fr.em = em_of(rng, W, H)
    fr.box_alpha = None
    return fr


def compose(rng, W, H):
    """A frame with its instances (or text-free): the TS.Frame."""
    fr = new_frame(rng, W, H)
    if rng.random() < MIX["empty"]:
        if rng.random() < 0.6:
            empty_box(fr, rng)
        fr.elements.append("empty")
        fr.empty = True
        return fr
    fr.empty = False
    layout = choose(rng, MIX["layout"])
    has_main = layout != "none" and dialogue_element(fr, rng, layout)
    if rng.random() < MIX["choice"]["with_main" if has_main else "without_main"]:
        choice_element(fr, rng)
    n_ui = int(rng.choice(len(MIX["ui_n"]), p=np.asarray(MIX["ui_n"]) / np.sum(MIX["ui_n"])))
    for _ in range(n_ui):
        ui_element(fr, rng)
    return fr


def instance_boxes(fr):
    """(lines, ignore) in frame px from the instances' own coverage (TS's rule)."""
    lines, ignore = [], [list(b) for b in fr.ignore]
    for x, y, cov, theta, v, role, _ in fr.inst:
        b = TS._box(x, y, cov, theta)
        across = (b[2] - b[0]) if v else (b[3] - b[1])
        if across < MIN_ACROSS:
            ignore.append(b)
            continue
        lines.append([b[0], b[1], b[2], b[3], v, role])
    return lines, ignore


GUARD = {"meiki_det_thr": 0.3, "meet_iou": 0.2, "meet_inside": 0.3, "meet_grow": 3, "ignore_grow": 2}
_MEIKI = {}


def guard_ignore(rgb, lines, ignore):
    """MeikiOCR's detector (threshold 0.3) on the finished frame: every box that meets no labelled line and no
    ignore area (IoU < 0.2 and under 30% of its area inside one grown by 3 px) becomes an ignore area (grown by 2
    px): text the scene kept after the inpainting (a sign, a logo, an unlisted caption) is never taught as
    background. Returns the added areas."""
    import textspot2_labels as L2
    if "m" not in _MEIKI:
        from teacher import engines as E
        _MEIKI["m"] = E.Meiki(det_thr=GUARD["meiki_det_thr"], n_threads=1)
    H, W = rgb.shape[:2]
    have = [ln[:4] for ln in lines] + [list(b) for b in ignore]
    g = GUARD["meet_grow"]
    grown = [[max(0, b[0] - g), max(0, b[1] - g), min(W, b[2] + g), min(H, b[3] + g)] for b in have]

    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def inter(a, b):
        return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    out = []
    for raw, _sc in L2.meiki_detect(_MEIKI["m"], np.ascontiguousarray(rgb[:, :, ::-1])):
        meets = False
        for b, bg in zip(have, grown):
            i = inter(raw, b)
            u = area(raw) + area(b) - i
            if (u and i / u >= GUARD["meet_iou"]) or inter(raw, bg) >= GUARD["meet_inside"] * max(1, area(raw)):
                meets = True
                break
        if not meets:
            e = GUARD["ignore_grow"]
            b = [max(0, raw[0] - e), max(0, raw[1] - e), min(W, raw[2] + e), min(H, raw[3] + e)]
            if b[2] > b[0] and b[3] > b[1]:
                out.append(b)
    return out


def sample_frame(rng):
    """One synthetic desktop frame (a sample dict and its grey canvas), or None for an unusable draw."""
    captured = rng.random() < MIX["captured"]
    if captured:
        gw, gh = CAP.wh(choose(rng, MIX["game"]))
        fr = compose(rng, gw, gh)
    else:
        sw, sh = CAP.wh(choose(rng, MIX["screen"]))
        fr = compose(rng, sw, sh)
    if not fr.empty and not fr.inst:
        return None
    lines, ignore = instance_boxes(fr)
    if not fr.empty and not lines:
        return None
    rgb = np.clip(fr.img + 0.5, 0, 255).astype(np.uint8)
    guard = guard_ignore(rgb, lines, ignore) if GUARD_ON else []
    ignore = ignore + guard
    cap = None
    tags = []
    if captured:
        cap = CAP.sample(rng, fr.W, fr.H, None)
        rgb = cap.apply(rgb)
        lines = [[*b, ln[4], ln[5]] for ln in lines for b in [cap.box(ln[:4])] if b is not None]
        ignore = [b for b in (cap.box(x) for x in ignore) if b is not None]
        tags.append("captured")
    elif rng.random() < MIX["native_jpeg"]["p"]:
        import cv2
        q = int(rng.integers(MIX["native_jpeg"]["quality"][0], MIX["native_jpeg"]["quality"][1] + 1))
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        rgb = np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])
        tags.append(f"jpeg{q}")
    H, W = rgb.shape[:2]
    meta = {"generator": GEN_VERSION, "render": [fr.W, fr.H], "size": [W, H], "em": fr.em, "scene": fr.scene,
            "elements": fr.elements, "capture": cap.record() if cap else None, "tags": tags,
            "box_alpha": round(fr.box_alpha, 3) if fr.box_alpha is not None else None, "guard_ignore": len(guard)}
    s = {"src": "synth:desktop" if not fr.empty else "synth:empty", "kind": "pc", "vn": None, "shot": None,
         "lines": lines, "ignore": ignore, "meta": meta}
    return s, SH.to_gray(rgb)


def run_shard(job):
    idx, n, seed, out_dir = job
    path = Path(out_dir) / f"desk_synth_{idx:04d}"
    if path.with_suffix(".json").exists():
        return idx, 0, 0, 0.0
    rng = np.random.default_rng([seed, idx])
    t0, samples, greys, errors = time.time(), [], [], 0
    while len(samples) < n:
        try:
            r = sample_frame(rng)
        except Exception as e:  # noqa: BLE001 - a rare bad draw must not end a long render
            errors += 1
            if errors <= 3:
                import traceback
                print(f"shard {idx}: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            if errors > 200:
                raise
            continue
        if r is not None:
            samples.append(r[0])
            greys.append(r[1])
    SH.write_shard(path, samples, greys)
    return idx, len(samples), errors, time.time() - t0


def cmd_render(args):
    import cv2
    cv2.setNumThreads(1)
    global GUARD_ON
    GUARD_ON = not args.no_guard
    init(scenes=not args.no_scenes)
    if GUARD_ON:
        guard_ignore(np.zeros((64, 64, 3), np.uint8), [], [])       # load MeikiOCR before the workers fork
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.per_shard, args.seed, str(out)) for i in range(args.first, args.first + args.shards)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, errors, dt in pool.imap_unordered(run_shard, jobs):
            print(f"desk synth shard {idx}: {n} frames ({errors} errors) in {dt:.0f}s ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"desk synth render: {len(jobs)} shards in {time.time() - t0:.0f} s", flush=True)


def cmd_manifest(args):
    d = Path(args.dir)
    shards = [SH.shard_record(js.with_suffix(""), d.name) for js in sorted(d.glob("desk_synth_*.json"))]
    doc = {"kind": "desktop-frames", "schema": 1, "issue": 169, "generator": "desktopdata/synth.py",
           "generator_version": GEN_VERSION, "created": SH.utc(),
           "format": "#97's frames (synth/textdet_data.py); train/textspot_train.FramePool reads them as source synth",
           "mix": MIX, "capture_mix": CAP.MIX, "guard": GUARD, "seed": args.seed,
           "fonts": "synth3's pool (data/manifests/synth3_fonts.json) without its pixel families: every face used is "
                    "licence class open",
           "scenes": "synth/textspot_data.scene_pool (textspot's clean scene pool) and synth2's synthetic backgrounds",
           "code_sha256": SH.code_sha256(ROOT, ["desktopdata/synth.py", "desktopdata/capture.py",
                                               "desktopdata/shards.py", "synth/textspot_data.py", "synth/synth3.py"]),
           "shards": shards, "totals": SH.totals(shards),
           "note": "shards hold rendered frames over VNDB screenshot scenes and stay local; this manifest holds names, "
                   "counts, sha256"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {doc['totals']['all']['samples'] if shards else 0} frames", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--per-shard", type=int, default=250)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--no-scenes", action="store_true")
    p.add_argument("--no-guard", action="store_true", help="no MeikiOCR guard (tests: no teacher env)")
    p = sub.add_parser("manifest")
    p.add_argument("--dir", required=True)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"render": cmd_render, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

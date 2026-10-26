#!/usr/bin/env python3
"""synth6: synthetic VN dialogue lines in the real game fonts, drawn the way each training title draws them, on
the titles' own screens.

synth3/synth4 drew free fonts in a sampled palette over VNDB scenes. synth6 uses what the owner's hooked corpus
(vita-vn-ocr #125) holds: the fonts the farm harvested from the games themselves (`synth6_fonts.py`), each train
title's measured text style and layout (`synth6_assets.py`: rows, ink height, pen advance, fill, outline or halo,
shadow, box colour and opacity) and the title's real pages, with every text box on them masked, as the background.

  matched   the title's verified dialogue font at its native size or scale and letter spacing, in its measured
            style, on a text row its page leaves empty: a new line exactly as the game would draw it
  cross     any other drawable harvested font (game and Vita/PSP system fonts; never a test title's), sized to the
            title's pen advance, in the title's style: the games' real typefaces in other games' boxes
  rows      lines on the title's dialogue rows that the page's own text leaves free (names on its name row)
  scene     lines on the page's scene away from every box, bare (outline, halo or shadow as the title draws
            them) or on a box of the title's measured colour and opacity (scene_box)
  views     vita (960x544 native), psp (480x272 native: model B) and psp2x (the PSP frame upscaled 2x as Rubi
            captures a PSP game through Adrenaline: bilinear, with Adrenaline's sharp (nearest) filter sometimes)
  crops     per line: the renderer-cell box + pad (the hooked pool's convention), the ink box + pad (eval's), or
            the console's GRAD finder box on the dialogue region (common.find_lines_ex), all in console polarity
  text      synth6_text.py: synth3's VN pools, other titles' hook text, rare-character targets, #119's confusions

  synth6.py assets --hooked DIR --pages DIR --splits SPLITS --harvest DIR --out ASSETS  styles, layout, inventory
  synth6.py font-check --assets ASSETS --ckpt CKPT --out ASSETS/font_check.json     common lines read back by OCR
  synth6.py glyph-check --assets ASSETS --ckpt CKPT --out ASSETS/glyph_check.json   every kana/kanji, per font
  synth6.py glyph-geometry --assets ASSETS --out ASSETS/glyph_geometry.json        glyphs outside their pen cell
  synth6.py inventory --assets ASSETS                rebuild the font inventory alone
  synth6.py manifests --assets ASSETS        data/manifests/synth6_fonts.json + synth6_styles.json (committed)
  synth6.py preview --assets ASSETS --view vita --out DIR [--n 40]                  contact sheets
  synth6.py compare --assets ASSETS --title T --out sheet.png                       real vs synthetic, one title
  synth6.py shards --assets ASSETS --view vita --out DIR --shards N --per-shard M [--first F] [--workers W]
  synth6.py manifest --dir DIR [--dir DIR2] --out MANIFEST.json

Everything the corpus provides (pages, fonts, hook text) is private training input: never committed, never
uploaded. Shards are local; data/manifests/synth6_shards_v1.json pins them by sha256.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "teacher"))
import synth6_fonts as F6  # noqa: E402
from common import find_lines_ex, line_dark, to_gray  # noqa: E402
from vote import canon  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
DEFAULTS = {"hooked": HERD / "shared" / "private" / "hooked-b2",
            "pages": ART / "vocr6-ingest" / "pages",
            "splits": ART / "vocr6-ingest" / "splits_v1.json",
            "harvest": ART / "vocr6-synth6" / "fontpack" / "font-harvest",
            "assets": ART / "vocr6-synth6" / "assets",
            "curriculum": ART / "vocr5-train" / "render" / "curriculum_v5631.json"}
VOCAB = ROOT / "data" / "vocab_vn_5631.txt"

# Every probability of the generator, recorded verbatim in each shard set's recipe.
CFG = {
    "generator": "synth6",
    "version": 1,
    "matched": 0.55,            # the title's own verified font, when the farm harvested one
    "placement": {"rows": 0.8, "scene": 0.1, "scene_box": 0.1},
    "lines": [0.45, 0.35, 0.15, 0.05],          # lines per region (1..4), capped by the free rows
    "name": 0.35,               # a free name row gets a speaker name
    "text": {"pool": 0.40, "hook": 0.25, "target": 0.25, "confusion": 0.10},
    "fullwidth_space": 0.2,     # a 　 after a mid-line ！ or ？ (the cut-line case of #119)
    "crop": {"cell": 0.35, "ink": 0.35, "grad": 0.30},
    "grad_cover": 0.9,          # a GRAD box must hold this share of its line's ink height (else the ink crop)
    "pad": {"vita": 4, "psp": 2, "psp2x": 4},
    "pad_extra": {"vita": {"top": 0.04, "bottom": 0.06, "left": 0.12, "right": 0.04},
                  "psp": {"top": 0.0, "bottom": 0.02, "left": 0.2, "right": 0.0}},
    "region_margin": {"vita": 12, "psp": 6, "psp2x": 12},
    "adrenaline": {"bilinear": 0.8, "nearest": 0.15, "bicubic": 0.05},
    "cross_size": [0.9, 1.12],  # the cross font's advance against the title's, uniform
    "accent": 0.5,              # of the title's non-plain fill share: a line in one of its accent colours
    "style_jitter": {"rgb": 10, "alpha": 0.12, "drop_halo": 0.05, "drop_shadow": 0.05},
    "borrow_style": 0.12,       # cross mode: another same-polarity title's halo and shadow
    "spacing_jitter": 0.6,      # px, uniform +-, on the matched spacing
    "y_jitter": 1,              # px
    "scene_band": [0.06, 0.72],  # scene placements' text top, as a share of the frame height
    "min_label": 1,
    "min_contrast": 60,         # luma levels between the fill and its surround (legibility)
    "max_busy": 45,             # luma sd around unframed text (no halo or shadow of half opacity)
    "min_char_contrast": 40,    # the lowest per-character contrast (a line partly over a bright patch)
    "scene_box_alpha": [0.8, 1.0],
    "page_tries": 4,            # pages tried for a free dialogue row before a scene placement
}
MASK_PAD = {"vita": 4, "psp": 2}
SKIP_LABELS = set("「」『』（）()、。・…―ー ")
CLOSE = set("、。」』）！？…‥ーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々")
W = None        # per-process render state (init)


def sha256(path):
    return F6.sha256_file(path)


def choose(rng, weights):
    names = list(weights)
    p = np.array([weights[n] for n in names], np.float64)
    return names[int(rng.choice(len(names), p=p / p.sum()))]


# ----------------------------------------------------------------------------- state

class World:
    """Per-process render state: titles (layout, style, pages), fonts and the text sampler."""

    def __init__(self, assets, harvest, vocab=VOCAB, curriculum=None, text=True, page_cache=48):
        assets = Path(assets)
        self.assets, self.harvest = assets, Path(harvest)
        doc = json.loads((assets / "titles.json").read_text(encoding="utf-8"))
        self.titles_doc = doc
        self.titles = doc["titles"]
        self.pages_root = Path(doc.get("pages_root") or DEFAULTS["pages"])
        # a title draws synthetic lines when it has a measured horizontal style and dialogue rows (not the
        # vertical PCSG00653, not a title without verified lines)
        usable = {t for t, i in self.titles.items() if (i["style"].get("line") or {}).get("lines")
                  and i["layout"].get("line")}
        self.by_platform = {p: sorted(t for t in usable if self.titles[t]["platform"] == p) for p in ("vita", "psp")}
        inv = json.loads((assets / "fonts_inventory.json").read_text(encoding="utf-8"))
        check = {}
        if (assets / "font_check.json").exists():
            check = json.loads((assets / "font_check.json").read_text(encoding="utf-8"))["fonts"]
        self.font_check = check
        gc = assets / "glyph_check.json"
        self.deny = {k: set(v["suspect"]) for k, v in json.loads(gc.read_text(encoding="utf-8"))["fonts"].items()} \
            if gc.exists() else {}
        gg = assets / "glyph_geometry.json"
        if gg.exists():
            for k, v in json.loads(gg.read_text(encoding="utf-8"))["fonts"].items():
                self.deny.setdefault(k, set()).update(v["misplaced"] + v.get("swapped", "") + v.get("manual", ""))
        self.inventory = {r["font"]: r for r in inv}
        self.cross = {}
        for r in inv:
            if r["excluded"] or (r["glyphs"] or 0) < 2000:
                continue
            c = check.get(r["font"])
            if c is not None and not c.get("pass"):
                continue
            if c is None and not r["trusted_map"]:
                continue
            face = f"{r['title'] if not r['system'] else 'SYSTEM'}:{r['face']}"
            self.cross.setdefault(face, []).append(r)
        self.cross_faces = sorted(self.cross)
        self.vocab_set = set(c for c in Path(vocab).read_text(encoding="utf-8").split("\n") if c) - {"〓"}
        self._fonts = {}
        self._pages = OrderedDict()
        self.page_cache = page_cache
        self.text = None
        if text:
            import synth6_text as T6
            self.text = T6.Text6(assets, vocab, curriculum=curriculum)

    def font(self, fid):
        if fid not in self._fonts:
            f = F6.open_font(self.harvest / fid)
            f.deny = set(self.deny.get(fid, ""))
            self._fonts[fid] = f
        return self._fonts[fid]

    def page(self, title, pid):
        key = (title, pid)
        if key in self._pages:
            self._pages.move_to_end(key)
            return self._pages[key]
        im = cv2.imread(str(self.pages_root / title / "pages" / f"{pid}.png"), cv2.IMREAD_COLOR)
        if im is None:
            return None
        im = np.ascontiguousarray(im[..., ::-1])
        self._pages[key] = im
        if len(self._pages) > self.page_cache:
            self._pages.popitem(last=False)
        return im


def init(assets, harvest, curriculum=None, text=True):
    global W
    W = World(assets, harvest, curriculum=curriculum, text=text)
    return W


# ----------------------------------------------------------------------------- fonts and style

def title_advance(T):
    st = (T["style"].get("line") or {})
    adv = st.get("advance_p50")
    if adv:
        return float(adv)
    rows = T["layout"].get("line") or []
    return float(rows[0]["h"]) * 0.85 if rows else 24.0


def matched_font(T):
    f = T.get("font")
    if not f:
        return None
    r = W.inventory.get(f["font"])
    if r is None or r["excluded"]:
        return None
    c = W.font_check.get(f["font"])
    if c is not None and not c.get("pass"):
        return None
    return f


PROBE = "あいうえおかきくけこ国語漢字時間"
_METRICS = {}


def font_metrics(font):
    """(ink width, advance) of the font's full-width glyphs at scale 1 (vector fonts at 32 px), medians over
    PROBE. Atlases often store a cell with room for an outline, so the advance can be 1.5-1.9x the ink; an engine
    then draws with a negative letter spacing (the fidelity fits find -3 to -7 px)."""
    key = id(font)
    if key not in _METRICS:
        size0 = None
        if font.kind == "vector":
            size0 = font.size
            font.set_size(32)
        ink, adv = [], []
        for c in PROBE:
            if not font.has(c):
                continue
            g, _, _, a = font.glyph(c)
            cols = np.flatnonzero((g > 0.2).any(axis=0))
            if len(cols):
                ink.append(cols[-1] - cols[0] + 1)
                adv.append(a)
        if size0:
            font.set_size(size0)
        _METRICS[key] = (float(np.median(ink)) if ink else 28.0, float(np.median(adv)) if adv else 32.0)
    return _METRICS[key]


def pick_font(rng, T):
    """(font, scale, spacing, mode, font id): the title's matched font, or a cross font whose full-width ink is
    the title's pen advance times U(0.82, 0.93) and whose letter spacing makes its pen advance the title's
    (times CFG cross_size)."""
    mf = matched_font(T)
    if mf and rng.random() < CFG["matched"]:
        font = W.font(mf["font"])
        size = float(mf["size_or_scale"] or 1.0)
        sp = float(mf["spacing"] or 0.0) + float(rng.uniform(-1, 1)) * CFG["spacing_jitter"]
        if font.kind == "vector":
            font.set_size(size)
            return font, 1.0, sp, "matched", mf["font"]
        return font, size, sp, "matched", mf["font"]
    target = title_advance(T) * float(rng.uniform(*CFG["cross_size"]))
    want_ink = target * float(rng.uniform(0.82, 0.93))
    for _ in range(8):           # a face whose strikes all need a scale outside [0.5, 1.7] is redrawn
        face = W.cross_faces[int(rng.integers(len(W.cross_faces)))]
        cands = W.cross[face]
        if cands[0]["kind"] == "vector" or any(0.5 <= want_ink / font_metrics(W.font(c["font"]))[0] <= 1.7
                                               for c in cands):
            break
    if cands[0]["kind"] == "vector":
        r = cands[int(rng.integers(len(cands)))]
        font = W.font(r["font"])
        ink, adv = font_metrics(font)
        size = 32.0 * want_ink / ink
        font.set_size(size)
        return font, 1.0, float(target - adv * size / 32.0), "cross", r["font"]

    def ink_at(c):
        f = W.font(c["font"])
        return font_metrics(f)[0]
    r = min(cands, key=lambda c: abs(ink_at(c) - want_ink))
    font = W.font(r["font"])
    ink, adv = font_metrics(font)
    scale = float(np.clip(want_ink / ink, 0.5, 1.7))
    return font, scale, float(target - adv * scale), "cross", r["font"]


def jitter_rgb(rng, c, amt):
    return np.clip(np.asarray(c, np.float32) + rng.uniform(-amt, amt, 3), 0, 255)


def pick_style(rng, T, mode, kind="line"):
    """The title's measured style for one region, jittered: fill, halo and shadow (colour, opacity, geometry)."""
    st = T["style"].get(kind) or T["style"].get("line") or {}
    J = CFG["style_jitter"]
    fill = st.get("fill_rgb") or [240, 240, 240]
    acc = st.get("accent_fills") or []
    if acc and rng.random() < CFG["accent"] * (1.0 - float(st.get("fill_plain_share", 1.0))):
        fill = acc[int(rng.integers(len(acc)))]
    out = {"fill": jitter_rgb(rng, fill, J["rgb"]), "halo": None, "shadow": None,
           "dark": float(st.get("dark_share", 0.0)) >= 0.5}
    src = st
    if mode == "cross" and rng.random() < CFG["borrow_style"]:
        same = [t for t, i in W.titles.items()
                if ((i["style"].get("line") or {}).get("dark_share", 0.0) >= 0.5) == out["dark"]]
        if same:
            src = W.titles[same[int(rng.integers(len(same)))]]["style"].get("line") or st
    h = src.get("halo")
    if h and rng.random() >= J["drop_halo"]:
        out["halo"] = {"radius": int(h["radius"]), "width": int(max(1, h["width"])), "hard": bool(h["hard"]),
                       "alpha": float(np.clip(h["alpha"] + rng.uniform(-J["alpha"], J["alpha"]), 0.05, 1.0)),
                       "rgb": jitter_rgb(rng, h["rgb"] or ([0, 0, 0] if not out["dark"] else [255] * 3), J["rgb"])}
    s = src.get("shadow")
    if s and rng.random() >= J["drop_shadow"]:
        out["shadow"] = {"offset": [int(v) for v in s["offset"]],
                         "alpha": float(np.clip(s["alpha"] + rng.uniform(-J["alpha"], J["alpha"]), 0.05, 1.0)),
                         "rgb": jitter_rgb(rng, s["rgb"], J["rgb"])}
    return out


# ----------------------------------------------------------------------------- text layout

def wrap(font, text, scale, spacing, max_w, rng):
    """Greedy VN wrap: as many characters as fit in max_w px; one closing mark that would start a line hangs on the
    previous one (engines let 、。」 overhang). Drawable characters only (the caller checked)."""
    lines, cur, w = [], "", 0.0
    s = scale if font.kind == "bitmap" else 1.0
    for ch in text:
        adv = font.glyph(ch)[3] * s + spacing
        # a closing mark may hang past max_w, as engines let 、。」 do, but only one: a second breaks the line
        if cur and w + adv > max_w and (ch not in CLOSE or w > max_w):
            lines.append(cur)
            cur, w = "", 0.0
        cur += ch
        w += adv
    if cur:
        lines.append(cur)
    return [ln.lstrip("\u3000 ") for ln in lines]


def add_spaces(rng, s):
    if rng.random() >= CFG["fullwidth_space"]:
        return s
    out = []
    for i, ch in enumerate(s):
        out.append(ch)
        if ch in "！？" and 0 < i < len(s) - 1 and s[i + 1] not in "！？」』）…\u3000" and rng.random() < 0.6:
            out.append("\u3000")
    return "".join(out)


def drawable(font, s):
    return all(font.has(c) for c in s)


# ----------------------------------------------------------------------------- compositing

def soft(mask, radius, width, hard):
    """A halo mask from glyph coverage: dilated to `radius` (+ width - 1), softened (hard: 0.5 px, soft: 1.2)."""
    r = max(1, radius + width - 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.dilate(mask, k)
    sig = 0.5 if hard else 1.2
    return np.clip(cv2.GaussianBlur(m, (0, 0), sig) * (1.0 if hard else 1.15), 0, 1)


def shift(m, dx, dy):
    """m moved by (dx, dy) >= 0 px (down-right), zero filled."""
    out = np.zeros_like(m)
    H, W_ = m.shape
    out[dy:, dx:] = m[:H - dy, :W_ - dx]
    return out


def composite(img, cov, x, y, style):
    """Draw coverage `cov` (float32, its top-left at frame (x, y)) onto img (float32 RGB) in `style`.
    Returns the alpha of everything drawn (fill, halo, shadow) in cov's frame, for the ink box."""
    H, W_ = img.shape[:2]
    h, w = cov.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros_like(cov)
    region = img[y0:y1, x0:x1]
    sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    total = cov.copy()
    if style.get("shadow"):
        s = style["shadow"]
        sm = shift(cov, *s["offset"])
        if style.get("halo"):
            sm = shift(soft(cov, 1, 1, True), *s["offset"])
        sm = cv2.GaussianBlur(sm, (0, 0), 0.7) * s["alpha"]
        a = sm[sl][..., None]
        region[:] = region * (1 - a) + s["rgb"] * a
        total = np.maximum(total, sm)
    if style.get("halo"):
        hl = style["halo"]
        hm = soft(cov, hl["radius"], hl["width"], hl["hard"]) * hl["alpha"]
        a = hm[sl][..., None]
        region[:] = region * (1 - a) + hl["rgb"] * a
        total = np.maximum(total, hm)
    a = cov[sl][..., None]
    region[:] = region * (1 - a) + style["fill"] * a
    return total


def draw_box(img, rng, x0, y0, x1, y1, rgb, alpha):
    """A flat dialogue box of the given colour and opacity, with rounded corners (no rim, no texture)."""
    H, W_ = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W_, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    rad = int(rng.integers(0, max(1, min(12, (y1 - y0) // 3))))
    cv2.rectangle(m, (rad, 0), (m.shape[1] - 1 - rad, m.shape[0] - 1), 1.0, -1)
    cv2.rectangle(m, (0, rad), (m.shape[1] - 1, m.shape[0] - 1 - rad), 1.0, -1)
    for cx, cy in ((rad, rad), (m.shape[1] - 1 - rad, rad), (rad, m.shape[0] - 1 - rad),
                   (m.shape[1] - 1 - rad, m.shape[0] - 1 - rad)):
        cv2.circle(m, (cx, cy), rad, 1.0, -1)
    a = (m * alpha)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + np.asarray(rgb, np.float32) * a


# ----------------------------------------------------------------------------- one region

def free_rows(rows, boxes, platform):
    """Rows (dialogue or name) of the title that no box of the page touches, with their usable width."""
    m = MASK_PAD[platform]
    out = []
    for r in rows:
        x0 = min(v for v, _ in r["x0"])
        x1 = max(r["x1"], x0 + 2 * r["h"])
        ya, yb = r["y0"] - m, r["y0"] + r["h"] + m
        if all(bx + bw + m <= x0 or bx - m >= x1 or by + bh <= ya or by >= yb for bx, by, bw, bh in boxes):
            out.append(r)
    return out


def row_chain(rng, free, k_max):
    """Up to k_max free rows, one region's lines: from a random free row, each next row the nearest one below that
    does not overlap the previous row (a title with several layouts has rows 5-10 px apart: drawing two of those
    would stack two lines) and starts within 2.5 row heights of its top (the next line of the same box)."""
    free = sorted(free, key=lambda r: (r["y0"], -r["count"]))
    if not free:
        return []
    chain = [free[int(rng.integers(len(free)))]]
    while len(chain) < k_max:
        last = chain[-1]
        nxt = [r for r in free if r["y0"] >= last["y0"] + last["h"] and r["y0"] - last["y0"] <= 2.5 * last["h"]]
        if not nxt:
            break
        chain.append(min(nxt, key=lambda r: (r["y0"], -r["count"])))
    return chain


def overlaps(a, b):
    """Do two [x, y, w, h] boxes overlap?"""
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def scene_slot(rng, T, boxes, platform, n_lines, pitch, width):
    """A free rectangle for n_lines on the page's scene: top in CFG scene_band, no box within a line height."""
    Wf, Hf = T["dims"]
    m = MASK_PAD[platform]
    h = int(math.ceil(pitch * n_lines))
    for _ in range(24):
        w = int(min(Wf - 2 * m - 4, width * rng.uniform(0.5, 1.0)))
        if w < 40:
            return None
        x = int(rng.integers(m + 2, max(m + 3, Wf - w - m - 2)))
        y = int(rng.uniform(*CFG["scene_band"]) * Hf)
        if y + h + m >= Hf:
            continue
        pad = int(pitch)
        if all(bx + bw + pad <= x or bx - pad >= x + w or by + bh + pad <= y or by - pad >= y + h
               for bx, by, bw, bh in boxes):
            return x, y, w
    return None


def line_text(rng, T, font, title):
    """An utterance the font can draw, in the title's convention, and its source."""
    for _ in range(12):
        s, src, focus = W.text.utterance(rng, CFG["text"], title=title, font=font)
        if not s:
            continue
        s = add_spaces(rng, s)
        if drawable(font, s):
            return s, src, focus
    return None, None, None


def render_region(rng, view):
    """(frame float32 RGB at the view's resolution, lines [{...}], meta) or None."""
    plat = "vita" if view == "vita" else "psp"
    titles = W.by_platform[plat]
    title = titles[int(rng.integers(len(titles)))]
    T = W.titles[title]
    pids = list(T["pages"])
    rows = T["layout"].get("line") or []
    placement = choose(rng, CFG["placement"])
    for _ in range(CFG["page_tries"] if placement == "rows" else 1):
        pid = pids[int(rng.integers(len(pids)))]
        boxes = T["pages"][pid]
        free = free_rows(rows, boxes, plat) if placement == "rows" else []
        if placement != "rows" or free:
            break
    page = W.page(title, pid)
    if page is None:
        return None
    if placement == "rows" and not free:
        placement = "scene"
    font, scale, spacing, mode, fid = pick_font(rng, T)
    style = pick_style(rng, T, mode)
    k_max = 1 + int(choose(rng, {i: p for i, p in enumerate(CFG["lines"])}))
    lh = font.line_height * (scale if font.kind == "bitmap" else 1.0)
    pitch = (rows[1]["y0"] - rows[0]["y0"]) if len(rows) >= 2 else 1.35 * lh
    pitch = max(pitch, lh + 2)
    width = max([r["x1"] - min(v for v, _ in r["x0"]) for r in rows] or [0.7 * T["dims"][0]])
    slots = []                       # (x0, top, max_w, kind)
    if placement == "rows":
        for r in row_chain(rng, free, k_max):
            xs = [v for v, _ in r["x0"]]
            ws = np.array([n for _, n in r["x0"]], np.float64)
            x0 = xs[int(rng.choice(len(xs), p=ws / ws.sum()))]
            slots.append((x0, r["y0"], r["x1"] - x0, "line", r["h"]))
    else:
        got = scene_slot(rng, T, boxes, plat, k_max, pitch, width)
        if got is None:
            return None
        x, y, w = got
        for i in range(k_max):
            slots.append((x, int(y + i * pitch), w, "line", int(lh)))
    text, src, focus = line_text(rng, T, font, title)
    if text is None:
        return None
    max_w = min(s[2] for s in slots)
    wrapped = wrap(font, text, scale, spacing, max_w, rng)
    if not wrapped:
        return None
    if len(wrapped) > len(slots):          # a later page of a long utterance (synth3 v3's pages)
        page_i = int(rng.integers(0, (len(wrapped) + len(slots) - 1) // len(slots)))
        wrapped = wrapped[page_i * len(slots):(page_i + 1) * len(slots)]
    img = page.astype(np.float32)
    fill_l = float(to_gray(style["fill"].reshape(1, 1, 3).astype(np.uint8))[0, 0])
    if placement == "scene":
        # bare text over a scene only where the title's own look keeps it readable: a hard halo, or a calm
        # background far from the fill's luma; otherwise the title's box goes behind it (scene_box)
        x0, y0 = slots[0][0], slots[0][1]
        blk = to_gray(page[max(0, y0):y0 + int(pitch * len(wrapped)) + 1, max(0, x0):x0 + slots[0][2]])
        hard = style.get("halo") and style["halo"]["hard"] and style["halo"]["alpha"] >= 0.5
        if not blk.size or not (hard or (abs(fill_l - float(blk.mean())) >= 100 and float(blk.std()) <= 35)):
            placement = "scene_box"
    if placement == "scene_box":
        # the title's box over another part of its own screens: the colour real text sits on in the title's
        # crops (bg_rgb_p50: box and scene through it together), nearly opaque, so the real contrast holds
        st = T["style"].get("line") or {}
        rgb = st.get("bg_rgb_p50") or (T.get("box") or {}).get("rgb") or [20, 20, 40]
        if abs(float(to_gray(np.asarray(rgb, np.uint8).reshape(1, 1, 3))[0, 0]) - fill_l) < 90:
            rgb = [235, 235, 235] if fill_l < 128 else [20, 20, 30]
        alpha = float(rng.uniform(*CFG["scene_box_alpha"]))
        x0 = slots[0][0] - int(rng.uniform(0.5, 1.2) * lh)
        y0 = slots[0][1] - int(rng.uniform(0.35, 0.8) * lh)
        x1 = slots[0][0] + slots[0][2] + int(rng.uniform(0.3, 1.0) * lh)
        y1 = slots[len(wrapped) - 1][1] + int(lh * rng.uniform(1.3, 1.9))
        draw_box(img, rng, x0, y0, x1, y1, jitter_rgb(rng, rgb, 8), alpha)
    lines = []
    for (x0, top, mw, kind, rh), s in zip(slots, wrapped):
        y = top + int(rng.integers(-CFG["y_jitter"], CFG["y_jitter"] + 1))
        lines.append(draw_text_line(img, font, s, scale, spacing, x0, y, style, kind))
    # a speaker name on a free name row
    nrows = T["layout"].get("name") or []
    if nrows and placement == "rows" and rng.random() < CFG["name"]:
        drawn = [ln["cell"] for ln in lines if ln.get("cell")]

        def row_box(r):
            x = min(v for v, _ in r["x0"])
            return [x, r["y0"], r["x1"] - x, r["h"]]
        nf = [r for r in free_rows(nrows, boxes, plat) if not any(overlaps(row_box(r), c) for c in drawn)]
        if nf:
            r = nf[int(rng.integers(len(nf)))]
            for _ in range(6):
                disp, nsrc = W.text.name(rng)
                if disp and drawable(font, disp):
                    nst = pick_style(rng, T, mode, "name")
                    ratio = float(np.clip(r["h"] / max(1.0, rows[0]["h"] if rows else r["h"]), 0.75, 1.1))
                    nscale = scale * ratio if font.kind == "bitmap" else scale
                    size0 = font.size if font.kind == "vector" else None
                    if size0:
                        font.set_size(size0 * ratio)
                    xs = [v for v, _ in r["x0"]]
                    ln = draw_text_line(img, font, disp, nscale, spacing, xs[0], r["y0"], nst, "name")
                    ln["source"] = nsrc
                    lines.append(ln)
                    if size0:
                        font.set_size(size0)
                    break
    for ln in lines:
        if ln["kind"] == "line":
            ln["source"], ln["focus"] = src, focus
    meta = {"title": title, "page": pid, "platform": plat, "view": view, "font": fid, "font_kind": font.kind,
            "mode": mode, "scale": round(float(scale), 4), "spacing": round(float(spacing), 3),
            "size": round(float(font.size), 2) if font.kind == "vector" else None, "placement": placement,
            "halo": None if not style["halo"] else {k: (round(v, 3) if isinstance(v, float) else v)
                                                    for k, v in style["halo"].items() if k != "rgb"},
            "shadow": None if not style["shadow"] else style["shadow"]["offset"],
            "real_boxes": boxes}
    return img, lines, meta


def draw_text_line(img, font, s, scale, spacing, x0, y, style, kind):
    """Draw one line onto the frame; returns its record: text, cells box, ink box (frame coordinates)."""
    m = 8
    cov, cells = F6.draw_line(font, s, scale=scale, spacing=spacing, x=m, y=m)
    # bitmap glyph quads (the renderer's boxes): pen + offset, bitmap size
    sc = scale if font.kind == "bitmap" else 1.0
    qx0, qy0, qx1, qy1 = 1e9, 1e9, -1e9, -1e9
    pen = float(m)
    for ch in s:
        g, ox, oy, adv = font.glyph(ch)
        if ch not in F6.SPACES and g.size > 1:
            gx, gy = pen + ox * sc, m + oy * sc
            qx0, qy0 = min(qx0, gx), min(qy0, gy)
            qx1, qy1 = max(qx1, gx + g.shape[1] * sc), max(qy1, gy + g.shape[0] * sc)
        pen += adv * sc + spacing
    if qx1 < qx0:
        return {"text": s, "kind": kind, "cell": None, "ink": None}
    # place so that the quads' top sits on the row's top y
    ox_, oy_ = int(round(x0 - m)), int(round(y - qy0))
    total = composite(img, cov, ox_, oy_, style)
    contrast, busy, worst = legibility(img, cov, total, ox_, oy_, cells)
    framed = any((style.get(k) or {}).get("alpha", 0.0) >= 0.5 for k in ("halo", "shadow"))
    ys, xs = np.nonzero(total > 0.1)
    ink = None if not len(xs) else [int(xs.min()) + ox_, int(ys.min()) + oy_, int(xs.max() - xs.min() + 1),
                                    int(ys.max() - ys.min() + 1)]
    cell = [int(math.floor(qx0)) + ox_, int(math.floor(qy0)) + oy_, int(math.ceil(qx1 - qx0)),
            int(math.ceil(qy1 - qy0))]
    return {"text": s, "kind": kind, "cell": cell, "ink": ink, "contrast": contrast, "busy": busy, "worst": worst,
            "legible": bool(contrast >= CFG["min_contrast"] and worst >= CFG["min_char_contrast"]
                            and (framed or busy <= CFG["max_busy"]))}


def legibility(img, cov, total, x, y, cells=()):
    """(contrast, busy, worst) of a drawn line after compositing. contrast: the luma contrast of the glyphs' fill core
    against the pixels just outside everything drawn (fill, halo, shadow), or against the halo where one is drawn
    (medians). busy: the luma spread (sd) of those outside pixels. worst: the lowest such contrast over the line's
    character cells (pen to pen + advance; cells with fewer than 6 core pixels, such as a small 、, are skipped), so a
    line partly drawn over a bright patch is caught. A line under CFG min_contrast or min_char_contrast, or over
    CFG max_busy without a halo or shadow of at least half opacity, is not cropped: real games keep text legible."""
    H, W_ = img.shape[:2]
    h, w = cov.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 255.0, 0.0
    sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    lum = 0.299 * img[y0:y1, x0:x1, 0] + 0.587 * img[y0:y1, x0:x1, 1] + 0.114 * img[y0:y1, x0:x1, 2]
    c, t = cov[sl], total[sl]
    core = c >= 0.8
    drawn = t > 0.08
    ring = cv2.dilate(drawn.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~drawn
    if core.sum() < 8 or ring.sum() < 8:
        return 0.0, 255.0, 0.0
    halo = (t > 0.3) & ~(c > 0.1)
    use_halo = (t - c).max() > 0.3 and halo.sum() >= 8

    def con(a, b, lo, hi):
        """contrast of the core in columns [a, b) against the ring and halo in columns [lo, hi)"""
        cm, rm, hm = core[:, a:b], ring[:, lo:hi], halo[:, lo:hi]
        fill = float(np.median(lum[:, a:b][cm]))
        v = abs(fill - float(np.median(lum[:, lo:hi][rm]))) if rm.sum() >= 4 else 0.0
        if use_halo and hm.sum() >= 4:
            v = max(v, abs(fill - float(np.median(lum[:, lo:hi][hm]))))
        return v
    contrast = con(0, core.shape[1], 0, core.shape[1])
    busy = float(lum[ring].std())
    worst = contrast
    for ch, cx0, cx1, _, _ in cells:
        if ch in F6.SPACES:
            continue
        a, b = int(math.floor(cx0)) - (x0 - x), int(math.ceil(cx1)) - (x0 - x)
        a, b = max(0, a), min(core.shape[1], b)
        if b - a < 2 or core[:, a:b].sum() < 6:
            continue
        worst = min(worst, con(a, b, max(0, a - 3), min(core.shape[1], b + 3)))
    return contrast, busy, worst


# ----------------------------------------------------------------------------- views and crops

def to_view(rng, img, view):
    """The frame as the recognizer's host sees it: psp2x is Adrenaline's 2x picture of a PSP frame."""
    if view != "psp2x":
        return img, 1, None
    f = choose(rng, CFG["adrenaline"])
    interp = {"bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "bicubic": cv2.INTER_CUBIC}[f]
    return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=interp), 2, f


def scaled(box, k):
    return None if box is None else [v * k for v in box]


def pad_box(box, pad, shape, extra=None, ih=0, rng=None):
    H, W_ = shape[:2]
    x, y, w, h = box
    more = [0, 0, 0, 0]
    if extra and rng is not None:
        more = [int(rng.uniform(0, extra[k]) * ih + 0.5) for k in ("left", "top", "right", "bottom")]
    x0, y0 = max(0, x - pad - more[0]), max(0, y - pad - more[1])
    x1, y1 = min(W_, x + w + pad + more[2]), min(H, y + h + pad + more[3])
    return [x0, y0, x1 - x0, y1 - y0]


def grad_boxes(grey, lines, real, margin, k):
    """GRAD finder boxes on the dialogue region (all drawn lines' ink plus the real boxes near them, + margin)
    that hold exactly one drawn line (synth3.grad_crops's acceptance rule, real boxes as unlabelled lines)."""
    from common import PAD
    inks = [ln["ink_v"] for ln in lines if ln.get("ink_v")]
    if not inks:
        return {}
    H, W_ = grey.shape
    rx0 = max(0, min(b[0] for b in inks) - margin)
    ry0 = max(0, min(b[1] for b in inks) - margin)
    rx1 = min(W_, max(b[0] + b[2] for b in inks) + margin)
    ry1 = min(H, max(b[1] + b[3] for b in inks) + margin)
    reg = grey[ry0:ry1, rx0:rx1]
    r = find_lines_ex(reg, "grad")
    if r is None or not r[0]:
        return {}
    dets = [(x + rx0, y + ry0, w, h) for x, y, w, h in r[0]]
    bands = []
    for i, ln in enumerate(lines):
        if ln.get("ink_v"):
            x, y, w, h = ln["ink_v"]
            bands.append(("l", i, y, y + h, x, x + w))
    for bx, by, bw, bh in real:
        bx, by, bw, bh = bx * k, by * k, bw * k, bh * k
        if by + bh > ry0 and by < ry1 and bx + bw > rx0 and bx < rx1:
            bands.append(("r", -1, by, by + bh, bx, bx + bw))
    out = {}
    for x, y, w, h in dets:
        a, b = y + (PAD if y > ry0 else 0), y + h - (PAD if y + h < ry1 else 0)
        ov = [min(b, e[3]) - max(a, e[2]) if (e[5] > x and e[4] < x + w) else 0 for e in bands]
        j = int(np.argmax(ov))
        if ov[j] <= 0 or bands[j][0] != "l":
            continue
        if any(o > 1 for i2, o in enumerate(ov) if i2 != j):
            continue
        e = bands[j]
        band = e[3] - e[2]
        if ov[j] < 0.5 * min(b - a, band) or h > 1.6 * (band + 2 * PAD):
            continue
        if min(y + h, e[3]) - max(y, e[2]) < CFG["grad_cover"] * band:      # a band cut through the glyphs
            continue
        if x > e[4] + 1 or x + w < e[5] - 1:          # the box must hold the line's ink across, even at the region edge
            continue
        if e[4] > rx0 + PAD + 1 and abs(x + PAD - e[4]) > 3:
            continue
        if e[5] < rx1 - PAD - 1 and abs(x + w - PAD - e[5]) > 3:
            continue
        if e[1] in out:
            out.pop(e[1])           # two boxes on one line: neither
            continue
        out[e[1]] = (x, y, w, h)
    return out


def extract(rng, img, lines, meta):
    """[(grey crop in console polarity, label, meta)] of a rendered region."""
    view = meta["view"]
    frame, k, filt = to_view(rng, img, view)
    meta["adrenaline_filter"] = filt
    frame8 = np.clip(frame + 0.5, 0, 255).astype(np.uint8)
    grey = to_gray(frame8)
    pad = CFG["pad"][view]
    extra = CFG["pad_extra"]["psp" if view == "psp" else "vita"]
    for ln in lines:
        ln["ink_v"] = scaled(ln.get("ink"), k)
        ln["cell_v"] = scaled(ln.get("cell"), k)
    mode = choose(rng, CFG["crop"])
    grads = grad_boxes(grey, lines, meta["real_boxes"], CFG["region_margin"][view], k) if mode == "grad" else {}
    Hn, Wn = img.shape[:2]
    out = []
    for i, ln in enumerate(lines):
        # a glyph cut off at the frame edge, or a line stacked on another drawn line, would make the label wrong
        inside = all(bx is not None and bx[0] >= 0 and bx[1] >= 0 and bx[0] + bx[2] <= Wn and bx[1] + bx[3] <= Hn
                     for bx in (ln.get("ink"), ln.get("cell")))
        if not inside or any(j != i and stacked(ln.get("ink"), o.get("ink")) for j, o in enumerate(lines)):
            continue
        lab = canon(ln["text"])
        if len(lab) < CFG["min_label"] or set(lab) <= SKIP_LABELS or \
                not W.text.clean_label(lab, ln.get("source")):
            continue
        if ln.get("ink_v") is None or not ln.get("legible", True):
            continue
        kind = mode
        if mode == "grad" and i in grads:
            box = list(grads[i])
        else:
            if mode == "grad":
                kind = "ink"                    # synth3 v3's fallback: a line without a GRAD box gets its gt crop
            if kind == "cell" and ln.get("cell_v"):
                box = pad_box(ln["cell_v"], pad, grey.shape)
            else:
                kind = "ink"
                box = pad_box(ln["ink_v"], pad, grey.shape, extra, ln["ink_v"][3], rng)
        x, y, w, h = box
        if w < 8 or h < 8:
            continue
        g = grey[y:y + h, x:x + w]
        dark = line_dark(g, (0, 0, w, h))
        crop = 255 - g if dark else g.copy()
        if not crop_feasible(crop.shape, lab):
            continue
        m = {k2: v for k2, v in meta.items() if k2 != "real_boxes"}
        m.update(kind=ln["kind"], crop=kind, crop_mode=mode, dark=int(dark), inverted=int(dark), box=box,
                 ink=ln["ink_v"], cell=ln["cell_v"], source=ln.get("source"), focus=ln.get("focus"),
                 contrast=round(float(ln.get("contrast", 0.0)), 1), busy=round(float(ln.get("busy", 0.0)), 1),
                 worst=round(float(ln.get("worst", 0.0)), 1))
        out.append((crop, lab, m))
    return out


def stacked(a, b):
    """Do two drawn lines' ink boxes overlap by a quarter of the smaller height or more (lines drawn on top of each
    other, not neighbours whose halos touch)?"""
    if a is None or b is None or a[0] >= b[0] + b[2] or b[0] >= a[0] + a[2]:
        return False
    ov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return ov >= 0.25 * min(a[3], b[3])


def crop_feasible(shape, label):
    """FL10's height-32 resize, width cap and stride-8 CTC frames (render.crop_ctc_feasible)."""
    h, w = shape
    width = max(8, min(1024, round(w * 32 / h)))
    needed = len(label) + sum(a == b for a, b in zip(label, label[1:]))
    return bool(label) and math.ceil(width / 8) >= needed


# ----------------------------------------------------------------------------- shards

def run_shard(job):
    idx, n_lines, seed, view, out_dir, prefix = job
    cv2.setNumThreads(1)
    path = Path(out_dir) / f"{prefix}_{idx:05d}.npz"
    if path.exists():
        return idx, 0, 0, 0, 0.0
    rng = np.random.default_rng([seed, idx, {"vita": 1, "psp": 2, "psp2x": 3}[view]])
    t0, crops, labels, metas, regions, errors = time.time(), [], [], [], 0, 0
    while len(crops) < n_lines:
        try:
            r = render_region(rng, view)
            if r is None:
                continue
            got = extract(rng, *r)
        except Exception as e:  # noqa: BLE001 - a rare bad draw must not end a long render
            errors += 1
            if errors <= 3:
                print(f"shard {idx}: render error {type(e).__name__}: {e}", flush=True)
            if errors > 1000:
                raise
            continue
        regions += 1
        for crop, lab, m in got:
            crops.append(crop)
            labels.append(lab)
            metas.append(m)
            if len(crops) >= n_lines:
                break
    shapes = np.array([c.shape for c in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=np.concatenate([c.ravel() for c in crops]), offsets=offsets, shapes=shapes,
                        labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(metas, ensure_ascii=False)))
    tmp.rename(path)
    return idx, len(crops), regions, errors, time.time() - t0


def recipe(args):
    code = {p: sha256(HERE / p) for p in ("synth6.py", "synth6_fonts.py", "synth6_assets.py", "synth6_text.py",
                                          "synth3.py", "common.py")}
    doc = W.titles_doc
    return {"kind": "synth6-recipe", "schema": 1, "view": args.view, "seed": args.seed, "per_shard": args.per_shard,
            "cfg": CFG, "code_sha256": code, "vocab": str(VOCAB.relative_to(ROOT)), "vocab_sha256": sha256(VOCAB),
            "assets": {"titles_sha256": sha256(W.assets / "titles.json"),
                       "hooktext_sha256": sha256(W.assets / "hooktext.jsonl"),
                       "real_counts_sha256": sha256(W.assets / "real_counts.json"),
                       "fonts_inventory_sha256": sha256(W.assets / "fonts_inventory.json"),
                       "font_check_sha256": sha256(W.assets / "font_check.json")
                       if (W.assets / "font_check.json").exists() else None,
                       "glyph_check_sha256": sha256(W.assets / "glyph_check.json")
                       if (W.assets / "glyph_check.json").exists() else None,
                       "glyph_geometry_sha256": sha256(W.assets / "glyph_geometry.json")
                       if (W.assets / "glyph_geometry.json").exists() else None,
                       "test_text_sha256": sha256(W.assets / "test_text.json")
                       if (W.assets / "test_text.json").exists() else None,
                       "hval_text_sha256": sha256(W.assets / "hval_text.json")
                       if (W.assets / "hval_text.json").exists() else None,
                       "splits_sha256": doc.get("splits_sha256")},
            "titles": sorted(W.by_platform["vita" if args.view == "vita" else "psp"]),
            "cross_faces": len(W.cross_faces), "cross_fonts": sum(len(v) for v in W.cross.values()),
            "text": W.text.R.text_identity if W.text else None,
            "versions": {"numpy": np.__version__, "opencv": cv2.__version__}}


def cmd_shards(args):
    init(args.assets, args.harvest, curriculum=args.curriculum)
    args.out.mkdir(parents=True, exist_ok=True)
    rec = json.loads(json.dumps(recipe(args), ensure_ascii=False))
    rp = args.out / "recipe.json"
    if rp.exists() and json.loads(rp.read_text(encoding="utf-8")) != rec:
        raise SystemExit(f"{rp} holds a different recipe: use a new --out")
    rp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    prefix = f"synth6_{args.view}"
    jobs = [(i, args.per_shard, args.seed, args.view, str(args.out), prefix)
            for i in range(args.first, args.first + args.shards)]
    t0, total = time.time(), 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, regions, errors, dt in pool.imap_unordered(run_shard, jobs):
            total += n
            print(f"shard {idx}: {n} lines from {regions} regions ({errors} errors) in {dt:.0f}s "
                  f"(total {total}, {time.time() - t0:.0f}s)", flush=True)
    print("shards finished", flush=True)


def cmd_manifest(args):
    shards, recipes = [], {}
    for d in args.dir:
        d = Path(d)
        recipes[d.name] = json.loads((d / "recipe.json").read_text(encoding="utf-8"))
        for p in sorted(d.glob("*.npz")):
            z = np.load(p)
            shards.append({"set": d.name, "file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
                           "samples": int(len(z["shapes"]))})
    doc = {"kind": "synth6-shards", "schema": 1, "sets": recipes, "shards": shards,
           "samples": sum(s["samples"] for s in shards), "bytes": sum(s["bytes"] for s in shards)}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(shards)} shards, {doc['samples']} samples, {doc['bytes'] / 1e9:.2f} GB -> {args.out}")


# ----------------------------------------------------------------------------- sheets

def sheet_rows(items, width=1280, row_h=34, label=True):
    """Contact-sheet rows (<= 1280 px wide): each crop at height 32 next to its label."""
    from PIL import Image, ImageDraw, ImageFont
    fp = sorted((HERD / "shared" / "data" / "fonts" / "notosansjp").glob("*.ttf"))
    lf = ImageFont.truetype(str(fp[0]), 15) if fp else ImageFont.load_default()
    rows = []
    for crop, text in items:
        g = cv2.resize(crop, (max(8, round(crop.shape[1] * 32 / crop.shape[0])), 32), interpolation=cv2.INTER_LINEAR)
        row = Image.new("RGB", (width, row_h), (0, 0, 0))
        row.paste(Image.fromarray(g[:, :760]).convert("RGB"), (0, 1))
        if label:
            ImageDraw.Draw(row).text((770, 7), text[:56], font=lf, fill=(255, 255, 160))
        rows.append(row)
    return rows


def save_sheets(rows, out, stem, per=37):
    from PIL import Image
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(rows), per):
        part = rows[s:s + per]
        sheet = Image.new("RGB", (part[0].width, sum(r.height for r in part)), (0, 0, 0))
        y = 0
        for r in part:
            sheet.paste(r, (0, y))
            y += r.height
        p = out / f"{stem}_{s // per:02d}.png"
        sheet.save(p)
        paths.append(p)
    return paths


def cmd_preview(args):
    init(args.assets, args.harvest, curriculum=args.curriculum)
    rng = np.random.default_rng(args.seed)
    items, regions = [], []
    while len(items) < args.n:
        r = render_region(rng, args.view)
        if r is None:
            continue
        img, lines, meta = r
        got = extract(rng, img, lines, meta)
        if len(regions) < 6:
            vis = np.clip(img + 0.5, 0, 255).astype(np.uint8).copy()
            for ln in lines:
                if ln.get("ink"):
                    x, y, w, h = ln["ink"]
                    cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
            regions.append(vis)
        for crop, lab, m in got:
            items.append((crop, f"{lab[:16]} [{m['font']} {m['mode'][0]}{m['placement'][:3]} {m['crop'][:2]}]"))
    paths = save_sheets(sheet_rows(items), args.out, f"crops_{args.view}")
    for i, im in enumerate(regions):
        im = im if im.shape[1] <= 1280 else cv2.resize(im, (1280, im.shape[0] * 1280 // im.shape[1]))
        cv2.imwrite(str(Path(args.out) / f"region_{args.view}_{i:02d}.png"), im[..., ::-1])
    print("preview:", [str(p) for p in paths], flush=True)


def cmd_compare(args):
    """Real crops of each title (--title, comma-separated) next to synth6 crops in that title's style, for an
    eye check: pairs of rows, the real line (hook label, R) then a synthetic one (S), --n pairs per title; the
    synthetic lines use the title's own font with --matched. PSP titles in --psp-view."""
    import synth6_assets as A6
    init(args.assets, args.harvest, curriculum=args.curriculum)
    dirs = A6.title_dirs(args.hooked)
    items = []
    rng = np.random.default_rng(args.seed)
    keep = dict(W.by_platform)
    if args.matched:
        CFG["matched"] = 1.0
    for title in args.title.split(","):
        T = W.titles[title]
        view = "vita" if T["platform"] == "vita" else args.psp_view
        pages = set(T["pages"])
        real = [r for r in A6.read_jsonl_zst(dirs[title][0] / "lines.jsonl.zst")
                if r["page"] in pages and r["kind"] == "line" and not r.get("vertical")]
        pick = [real[i] for i in rng.choice(len(real), size=min(args.n, len(real)), replace=False)]
        W.by_platform = {"vita": [title], "psp": [title]}
        synth, tries = [], 0
        while len(synth) < len(pick) and tries < 400:
            tries += 1
            r = render_region(rng, view)
            if r is None:
                continue
            for crop, lab, m in extract(rng, *r):
                if m["kind"] == "line":
                    synth.append((crop, f"S {lab[:14]} {title} {m['mode'][0]}{m['placement'][:3]} {m['crop'][:2]} "
                                        f"{m['font']}"))
        k = 2 if view == "psp2x" else 1
        for rr, sy in zip(pick, synth):
            page = W.page(title, rr["page"])
            if k == 2:
                page = cv2.resize(page, (page.shape[1] * 2, page.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
            x, y, w, h = (v * k for v in rr["box"])
            p = CFG["pad"][view]
            g = to_gray(page)[max(0, y - p):y + h + p, max(0, x - p):x + w + p]
            dark = line_dark(g, (0, 0, g.shape[1], g.shape[0]))
            items.append((255 - g if dark else g, f"R {rr['text'][:14]} {title}"))
            items.append(sy)
    W.by_platform = keep
    paths = save_sheets(sheet_rows(items), Path(args.out).parent, Path(args.out).stem)
    print("compare:", [str(p) for p in paths], flush=True)


# ----------------------------------------------------------------------------- fonts, assets, font check

def clannad_raster(inv, harvest):
    """{font id: {"font": CLANNAD font, "share": raster_share}} of every bitmap font of 28-36 px (CLANNAD's fonts are
    32 px) that holds one of CLANNAD's four fonts' raster (synth6_fonts.raster_share >= RASTER_SHARE)."""
    refs = {r["font"]: F6.open_font(harvest / r["font"]) for r in inv if r["title"] == "PCSG00415"}
    out = {}
    for r in inv:
        if r["title"] == "PCSG00415" or r["kind"] != "bitmap" or not 28 <= float(r["pixel_size"] or 0) <= 36:
            continue
        font = F6.open_font(harvest / r["font"])
        best = max(((F6.raster_share(font, ref), k) for k, ref in refs.items()), default=(0.0, None))
        if best[0] >= F6.RASTER_SHARE:
            out[r["font"]] = {"font": best[1], "share": round(best[0], 3)}
    return out


def cmd_manifests(args):
    """The committed records of a built asset set: data/manifests/synth6_fonts.json (every harvested font: ids,
    sha256s, fidelity verdicts, why excluded, the OCR check and the glyph check's denials) and
    data/manifests/synth6_styles.json (every train title's measured layout and style, its matched font, the page
    and line counts it was measured on). Numbers and hashes only: no font, page or text."""
    A = Path(args.assets)
    inv = json.loads((A / "fonts_inventory.json").read_text(encoding="utf-8"))
    inv_by = {r["font"]: r for r in inv}
    fc = json.loads((A / "font_check.json").read_text(encoding="utf-8"))
    gc = json.loads((A / "glyph_check.json").read_text(encoding="utf-8"))
    gg = json.loads((A / "glyph_geometry.json").read_text(encoding="utf-8"))["fonts"] \
        if (A / "glyph_geometry.json").exists() else {}
    raster = clannad_raster(inv, Path(args.harvest))
    rows = []
    for r in inv:
        c, g = fc["fonts"].get(r["font"]), gc["fonts"].get(r["font"])
        row = {k: r.get(k) for k in ("font", "kind", "format", "face", "system", "pixel_size", "glyphs",
                                     "vocr5_kanji", "content_sha256", "files", "excluded", "trusted_map",
                                     "farm_train_titles")}
        row["fidelity"] = {t: v["verdict"] for t, v in r["fidelity"].items()}
        row["ocr_check"] = None if c is None else {k: c.get(k) for k in ("pass", "cer", "lines")}
        row["glyph_check"] = None if g is None else {"checked_font": g["checked_font"], "checked": g["checked"],
                                                     "read_ok_share": g["read_ok_share"],
                                                     "placeholder": g["placeholder"], "wrong": g["wrong"],
                                                     "denied": g["suspect"]}
        row["misplaced"] = (gg.get(r["font"]) or {}).get("misplaced")
        row["swapped"] = (gg.get(r["font"]) or {}).get("swapped")
        row["manual_deny"] = (gg.get(r["font"]) or {}).get("manual")
        row["near_test_font"] = r.get("near_test_font")
        row["clannad_raster"] = raster.get(r["font"])
        row["drawn"] = not r["excluded"] and (r["glyphs"] or 0) >= 2000 and bool(c and c.get("pass"))
        rows.append(row)
    doc = {"kind": "synth6-fonts", "schema": 1,
           "source": "fonts/font-pack.tar.zst of the owner's private hooked corpus (vita-vn-ocr #125); never committed",
           "test_titles": list(F6.TEST_TITLES), "fonts": len(rows), "drawn": sum(r["drawn"] for r in rows),
           "excluded": {w: sum(1 for r in rows if w in r["excluded"]) for w in sorted({w for r in rows
                                                                                       for w in r["excluded"]})},
           "ocr_check": {"ckpt_sha256": fc["ckpt_sha256"], "max_cer": fc["max_cer"],
                         "text": FONT_CHECK_TEXT},
           "glyph_check": {"ckpt_sha256": gc["ckpt_sha256"], "references": gc["references"]},
           "glyph_geometry": {"rule": "synth6_fonts.misplaced, synth6_fonts.swapped, synth6.MANUAL_DENY"},
           "clannad_raster": {"rule": "synth6_fonts.raster_share against CLANNAD's (PCSG00415) four fonts, bitmap "
                                      "fonts of 28-36 px; flagged at RASTER_SHARE or above (the font holds CLANNAD's "
                                      "glyph raster, re-quantised or not)", "threshold": F6.RASTER_SHARE,
                              "flagged": sorted(raster), "drawn_flagged": sorted(
                                  f for f in raster if not inv_by[f]["excluded"])},
           "assets_sha256": {n: sha256(A / n) for n in ("fonts_inventory.json", "font_check.json",
                                                        "glyph_check.json", "glyph_geometry.json")
                             if (A / n).exists()},
           "rows": rows}
    Path(args.fonts_out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    t = json.loads((A / "titles.json").read_text(encoding="utf-8"))
    titles = {}
    for tid, i in sorted(t["titles"].items()):
        titles[tid] = {"name": i["name"], "platform": i["platform"], "dims": i["dims"], "vndb": i["vndb"],
                       "pages_train": i["pages_train"], "lines_train": i["lines_train"],
                       "vertical_lines": i["vertical_lines"], "font": i["font"], "box": i["box"],
                       "layout": {k: [{"y0": r["y0"], "h": r["h"], "x0": r["x0"][0][0], "x1": r["x1"],
                                       "lines": r["count"]} for r in v] for k, v in i["layout"].items()},
                       "style": i["style"],
                       "drawn": bool((i["style"].get("line") or {}).get("lines") and i["layout"].get("line"))}
    sdoc = {"kind": "synth6-styles", "schema": 1, "splits_sha256": t["splits_sha256"],
            "test_titles": t["test_titles"], "titles_sha256": sha256(A / "titles.json"),
            "measured_on": "train pages only (vocr6-ingest splits_v1: never an hval, guard or test page)",
            "titles": titles}
    Path(args.styles_out).write_text(json.dumps(sdoc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} fonts ({doc['drawn']} drawn) -> {args.fonts_out}; {len(titles)} titles -> {args.styles_out}")


def cmd_inventory(args):
    """Rebuild ASSETS/fonts_inventory.json alone (synth6_fonts.inventory: exclusions by hash, glyph identity,
    fidelity, glyph count and tategaki forms), without re-measuring the titles."""
    import synth6_assets as A6
    dirs = A6.title_dirs(args.hooked)
    train = {t for t, (_, md) in dirs.items() if md.get("split") == "train"}
    rows = F6.inventory(args.harvest, train_titles=train)
    (Path(args.assets) / "fonts_inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0) + "\n",
                                                            encoding="utf-8")
    from collections import Counter
    print(f"{len(rows)} fonts; excluded {dict(Counter(w for r in rows for w in r['excluded']))}", flush=True)


# Single wrong glyphs at rarely used codes, found by the second audit round's per-symbol screen (shape cosine against
# the reference fonts) and confirmed by eye (reports/audit-vocr6-synth6-b.md, F8). No automatic check covers a
# single wrong symbol outside a shifted run, so these are listed by hand.
# Round 3 added the 8 more its screen found over every drawable pair (reports/audit-vocr6-synth6-c.md, F11), plus ─
# in the two font24 fonts and NPJH50621's ④⑤ (outside the vocab today).
MANUAL_DENY = {"PCSG00617/font24": "┐└─", "PCSG00838/font24": "┐\"└─", "NPJH50621/font16": "≪≫①②③④⑤",
               "PCSG01083/font_system": "〒△▲"}


def cmd_glyph_geometry(args):
    """ASSETS/glyph_geometry.json: per drawn-candidate font, the vocab characters it would draw outside its own pen
    cell or not visibly (synth6_fonts.misplaced). The renderer denies them, as it denies the glyph check's."""
    inv = json.loads((Path(args.assets) / "fonts_inventory.json").read_text(encoding="utf-8"))
    vocab = [c for c in VOCAB.read_text(encoding="utf-8").split("\n") if c and c != "〓"]
    # symbols, punctuation, Latin and digits as text draws them (full-width forms too: labels fold them)
    shapes = sorted({c for c in vocab if not (0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF)}
                    | {chr(c) for c in range(0xFF01, 0xFF5F)})
    refs = [F6.open_font(Path(args.harvest) / spec, size=48) for spec in ("SYSTEM-VITA/pvf-jpn0",
                                                                          "SYSTEM-VITA/pvf-jpn1")]
    noto = sorted((HERD / "shared" / "data" / "fonts" / "notosansjp").glob("*.ttf"))
    if noto:
        refs.append(F6.OutlineFont(noto[0], size=48))
    out = {}
    for r in inv:
        if r["excluded"] or (r["glyphs"] or 0) < 2000:
            continue
        font = F6.open_font(Path(args.harvest) / r["font"])
        if font.kind == "vector":
            font.set_size(32)
        bad = F6.misplaced(font, vocab)
        sw = F6.swapped(font, shapes, refs)
        out[r["font"]] = {"misplaced": "".join(bad), "swapped": "".join(sw), "manual": MANUAL_DENY.get(r["font"], "")}
    doc = {"kind": "synth6-glyph-geometry", "schema": 1, "rule": F6.misplaced.__doc__.split("\n\n")[0].strip(),
           "fonts": out}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n = sorted(len(v["misplaced"]) for v in out.values())
    w = sorted(len(v["swapped"]) for v in out.values())
    print(f"glyph geometry: {len(out)} fonts, {sum(1 for x in n if x)} with misplaced glyphs (median {n[len(n) // 2]}, "
          f"max {n[-1]}), {sum(1 for x in w if x)} with swapped symbols (max {w[-1]}) -> {args.out}", flush=True)


def cmd_assets(args):
    import synth6_assets as A6
    A6.build(args.hooked, args.pages, args.splits, args.harvest, args.out, workers=args.workers)
    doc = json.loads((Path(args.out) / "titles.json").read_text(encoding="utf-8"))
    doc["pages_root"] = str(args.pages)
    (Path(args.out) / "titles.json").write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")


FONT_CHECK_TEXT = [
    "今日はいい天気ですね。", "「そんなこと、言わないでよ……」", "彼女は静かに窓の外を見つめていた。",
    "ありがとう、また明日。", "俺はどうすればいいんだ？", "教室の扉が音を立てて開いた。",
    "まさか、本当に来るとは思わなかった。", "「ちょっと待って！」", "世界の終わりを見届けるために。",
    "時計の針は午後三時を指している。", "雨が降り始めた。傘を持ってこなかった。", "君の名前を教えてくれないか。",
    "『本当に？』と彼は言った。", "（それは、違う……）", "ねえ――待って！\u3000どこ行くの？",
    "１２３４５６７８９０円", "ＡＢＣ\u3000ＤＥＦ", "ラーメン・カレー・パスタ", "ヴァイオリンとチェロ"]


def cmd_font_check(args):
    """Draw common lines in every candidate font and read them with a released recognizer: a font whose
    glyph-to-character map is wrong reads as other characters. A font passes at a CER of at most --max-cer."""
    sys.path.insert(0, str(ROOT / "eval"))
    sys.path.insert(0, str(ROOT / "train"))
    from backends import TinyTorch
    from metrics import lev, norm
    rec = TinyTorch(args.ckpt)
    inv = json.loads((Path(args.assets) / "fonts_inventory.json").read_text(encoding="utf-8"))
    out = {}
    for r in inv:
        if r["excluded"] or (r["glyphs"] or 0) < 2000:
            continue
        try:
            font = F6.open_font(Path(args.harvest) / r["font"])
        except Exception as e:  # noqa: BLE001
            out[r["font"]] = {"pass": False, "error": str(e)}
            continue
        if font.kind == "vector":
            font.set_size(26)
            scale = 1.0
        else:
            scale = float(np.clip(28.0 / max(8.0, float(r["pixel_size"] or font.line_height)), 0.6, 2.0))
        crops, refs = [], []
        for s in FONT_CHECK_TEXT:
            if not drawable(font, s):
                continue
            cov, _ = F6.draw_line(font, s, scale=scale, spacing=0.5, x=6, y=6)
            g = np.clip(cov * 255, 0, 255).astype(np.uint8)
            ys, xs = np.nonzero(g > 30)
            if not len(xs):
                continue
            g = g[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5]
            crops.append(g)
            refs.append(canon(s))
        if len(crops) < 6:
            out[r["font"]] = {"pass": False, "lines": len(crops), "why": "too few drawable check lines"}
            continue
        hyp = rec.read(crops)
        err = sum(lev(norm(h), norm(t)) for h, t in zip(hyp, refs))
        n = sum(len(norm(t)) for t in refs)
        cer = err / max(1, n)
        out[r["font"]] = {"pass": bool(cer <= args.max_cer), "cer": round(cer, 4), "lines": len(crops),
                          "sample": [hyp[0], refs[0]]}
        print(f"{r['font']}: {len(crops)} lines CER {cer:.3f} {'ok' if cer <= args.max_cer else 'FAIL'} "
              f"{hyp[0]!r}", flush=True)
    doc = {"kind": "synth6-font-check", "schema": 1, "ckpt": str(args.ckpt), "ckpt_sha256": sha256(args.ckpt),
           "max_cer": args.max_cer, "fonts": out,
           "passed": sum(1 for v in out.values() if v.get("pass")), "failed": sum(1 for v in out.values()
                                                                                if not v.get("pass"))}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"font check: {doc['passed']} pass, {doc['failed']} fail -> {args.out}")


def sjis_key(c):
    try:
        return c.encode("cp932")
    except UnicodeEncodeError:
        return b"\xff\xff" + c.encode("utf-8")


def glyph_vec(font, ch, size=32):
    """The glyph's ink cropped to its bounding box and resized to size x size, zero-mean unit-norm (or None)."""
    g = font.glyph(ch)[0]
    ys, xs = np.nonzero(g > 0.25)
    if len(xs) < 4:
        return None
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    v = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).ravel().astype(np.float32)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


class BatchReader:
    """A train/model.py checkpoint read in batches (eval/backends.TinyTorch's preprocessing, edge-padded to the
    batch's width; each line decoded over its own frames, greedy CTC)."""

    def __init__(self, ckpt):
        sys.path.insert(0, str(ROOT / "eval"))
        sys.path.insert(0, str(ROOT / "train"))
        import torch
        from backends import TinyTorch
        self.torch, self.tt = torch, TinyTorch(ckpt)

    def read(self, crops, batch=64):
        from backends import ctc_greedy
        out = []
        for s in range(0, len(crops), batch):
            xs = [self.tt.prep(c, self.tt.H) for c in crops[s:s + batch]]
            W_ = max(x.shape[3] for x, _ in xs)
            X = np.concatenate([np.pad(x, ((0, 0), (0, 0), (0, 0), (0, W_ - x.shape[3])), mode="edge")
                                for x, _ in xs])
            with self.torch.no_grad():
                ids = self.tt.model(self.torch.from_numpy(X)).argmax(2).numpy()      # (T, B)
            out += [ctc_greedy(ids[:t, i], self.tt.vocab) for i, (_, t) in enumerate(xs)]
        return out


def align(ref, hyp):
    """Levenshtein alignment: for each ref character, the hyp character aligned to it (None if deleted)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [None] * n, n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out[i - 1] = hyp[j - 1]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def cmd_glyph_check(args):
    """Per font, the vocab characters it must not draw. A decoder's character map can be wrong in a block (the
    font pack notes an error in the MAGES SC3 charset; Custom Drive's lt18 draws other kanji at JIS level-2
    codes), and a font can draw a placeholder for a character it lacks. The OCR check reads common text only, so
    here every covered kana and kanji is drawn (lines of 8 in random order, about 28 px) and read by a released
    recognizer (--ckpt), and compared on its glyph's ink box (32 x 32) with three reference fonts (the Vita
    system gothic and mincho, Noto Sans JP):

      placeholder  the same glyph image drawn for 3 or more vocab characters
      wrong        the recognizer reads the character as another kana or kanji d (not a lookalike of it, such
                   as ー and 一), and the glyph matches d's references better than its own by 0.05 or more: two
                   independent readings of another character
      block        40 consecutive covered characters in Shift-JIS order of which 40% or more are wrong: all

    Fonts of one title and face share a decoder and its table: the largest strike is checked and its denials
    apply to every strike. The renderer never draws a denied character in that font (synth6_fonts: font.deny)."""
    from collections import defaultdict
    inv = json.loads((Path(args.assets) / "fonts_inventory.json").read_text(encoding="utf-8"))
    vocab = [c for c in VOCAB.read_text(encoding="utf-8").split("\n") if c and c != "〓"]
    cjk = sorted((c for c in vocab if 0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF), key=sjis_key)
    cpos = {c: j for j, c in enumerate(cjk)}
    refs = [F6.open_font(Path(args.harvest) / spec, size=48) for spec in ("SYSTEM-VITA/pvf-jpn0",
                                                                          "SYSTEM-VITA/pvf-jpn1")]
    noto = sorted((HERD / "shared" / "data" / "fonts" / "notosansjp").glob("*.ttf"))
    if noto:
        refs.append(F6.OutlineFont(noto[0], size=48))
    R = np.zeros((len(refs), len(cjk), 1024), np.float32)
    have = np.zeros((len(refs), len(cjk)), bool)
    for i, f in enumerate(refs):
        for j, c in enumerate(cjk):
            if f.has(c):
                v = glyph_vec(f, c)
                if v is not None:
                    R[i, j], have[i, j] = v, True

    def ref_corr(v, j):
        return max((float(R[i, j] @ v) for i in range(len(refs)) if have[i, j]), default=-1.0)

    sys.path.insert(0, str(HERE))
    from synth3 import CONFUSABLE
    look = {}
    for grp in list(CONFUSABLE) + ["ー一―‐", "ロ口□", "工エ", "力カ", "二ニ", "八ハ", "卜ト", "夕タ", "へヘ", "千干"]:
        for c in grp:
            look.setdefault(c, set()).update(grp)
    reader = BatchReader(args.ckpt)
    groups = defaultdict(list)
    for r in inv:
        if not r["excluded"] and (r["glyphs"] or 0) >= 2000:
            groups[(r["title"], r["face"], r["format"])].append(r)
    rng = np.random.default_rng(0)
    out = {}
    for key, rows in sorted(groups.items()):
        r = max(rows, key=lambda x: float(x["pixel_size"] or 0))
        font = F6.open_font(Path(args.harvest) / r["font"])
        if font.kind == "vector":
            font.set_size(30)
            scale = 1.0
        else:
            scale = float(np.clip(28.0 / max(8.0, font_metrics(font)[0]), 0.6, 2.5))
        digest = defaultdict(list)
        for c in vocab:
            if font.has(c) and c not in F6.SPACES:
                g = font.glyph(c)[0]
                if g.size > 1:
                    h = hashlib.sha1(np.round(g * 255).astype(np.uint8).tobytes() + bytes(g.shape)).hexdigest()
                    digest[h].append(c)
        placeholder = {c for cs in digest.values() if len(cs) >= 3 for c in cs}
        chars = [c for c in cjk if font.has(c) and c not in placeholder]
        order = list(rng.permutation(len(chars)))
        lines = ["".join(chars[k] for k in order[s:s + 8]) for s in range(0, len(order), 8)]
        crops = []
        for ln in lines:
            cov, _ = F6.draw_line(font, ln, scale=scale, spacing=3.0, x=8, y=8)
            g = np.clip(cov * 255, 0, 255).astype(np.uint8)
            ys, xs = np.nonzero(g > 30)
            crops.append(g[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5] if len(xs) else g)
        hyps = reader.read(crops)
        wrong, read_ok = set(), 0
        for ln, hy in zip(lines, hyps):
            for c, d in zip(ln, align(ln, hy)):
                if d == c:
                    read_ok += 1
                elif d is not None and d in cpos and d not in look.get(c, ()):    # lookalikes: no evidence
                    v = glyph_vec(font, c)
                    if v is not None and ref_corr(v, cpos[d]) >= ref_corr(v, cpos[c]) + 0.05:
                        wrong.add(c)
        seq = np.array([c in wrong for c in chars], bool)
        if len(seq) >= 40:
            share = np.convolve(seq.astype(np.float32), np.ones(40, np.float32) / 40, mode="valid")
            grow = np.zeros_like(seq)
            for i in np.flatnonzero(share >= 0.4):
                grow[i:i + 40] = True
            wrong |= {c for c, g in zip(chars, grow) if g}
        deny = "".join(sorted(placeholder)) + "".join(c for c in chars if c in wrong)
        rec = {"checked_font": r["font"], "checked": len(chars), "read_ok": read_ok,
               "read_ok_share": round(read_ok / max(1, len(chars)), 4), "placeholder": len(placeholder),
               "wrong": len(wrong), "suspect": deny}
        for x in rows:
            out[x["font"]] = rec
        print(f"{key[0]}/{key[1]} ({len(rows)} strikes, {r['font']}): {len(chars)} checked, read {read_ok} "
              f"({rec['read_ok_share']:.1%}), {len(placeholder)} placeholder, {len(wrong)} wrong {deny[:40]}",
              flush=True)
    doc = {"kind": "synth6-glyph-check", "schema": 2, "ckpt": str(args.ckpt), "ckpt_sha256": sha256(args.ckpt),
           "references": ["SYSTEM-VITA/pvf-jpn0", "SYSTEM-VITA/pvf-jpn1"] + ([noto[0].name] if noto else []),
           "fonts": out}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"glyph check: {len(out)} fonts in {len(groups)} groups, "
          f"{sum(1 for v in out.values() if v['suspect'])} with denied characters -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["assets", "inventory", "font-check", "glyph-check", "glyph-geometry", "manifests",
                                     "preview", "compare", "shards", "manifest"])
    ap.add_argument("--hooked", type=Path, default=DEFAULTS["hooked"])
    ap.add_argument("--pages", type=Path, default=DEFAULTS["pages"])
    ap.add_argument("--splits", type=Path, default=DEFAULTS["splits"])
    ap.add_argument("--harvest", type=Path, default=DEFAULTS["harvest"])
    ap.add_argument("--assets", type=Path, default=DEFAULTS["assets"])
    ap.add_argument("--curriculum", type=Path, default=DEFAULTS["curriculum"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dir", action="append")
    ap.add_argument("--view", choices=["vita", "psp", "psp2x"], default="vita")
    ap.add_argument("--psp-view", choices=["psp", "psp2x"], default="psp")
    ap.add_argument("--title")
    ap.add_argument("--matched", action="store_true", help="compare: always the title's own font")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--max-cer", type=float, default=0.08)
    ap.add_argument("--shards", type=int, default=50)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--per-shard", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--fonts-out", type=Path, default=ROOT / "data" / "manifests" / "synth6_fonts.json")
    ap.add_argument("--styles-out", type=Path, default=ROOT / "data" / "manifests" / "synth6_styles.json")
    args = ap.parse_args()
    cv2.setNumThreads(1)
    {"assets": cmd_assets, "inventory": cmd_inventory, "font-check": cmd_font_check, "glyph-check": cmd_glyph_check,
     "glyph-geometry": cmd_glyph_geometry, "manifests": cmd_manifests, "preview": cmd_preview,
     "compare": cmd_compare, "shards": cmd_shards, "manifest": cmd_manifest}[args.mode](args)


if __name__ == "__main__":
    main()

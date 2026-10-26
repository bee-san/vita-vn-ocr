"""synth6 fonts: the real game fonts the owner's farm harvested, read and drawn the way the engines draw them.

The font pack (`fonts/font-pack.tar.zst` of the owner's private hooked corpus, schema `vn-font-harvest/1`) holds one
directory per font, `<TITLE_ID>/<font_id>/`, with `font.json` (format, face, pixel size, coverage, source sha256,
fidelity) and either the original vector file (`.otf`, `.pvf`; drawn by FreeType) or a bitmap atlas (`atlas.png`,
8-bit coverage) with `glyphs.json`: per character `[x, y, w, h, ox, oy, adv]`, the atlas rectangle, its offset from
the pen at the line top and the advance. The pack is private training input: nothing of it is committed, only the
inventory (`data/manifests/synth6_fonts.json`: font ids, sha256, titles, fidelity, why a font is excluded).

Drawing follows the engines: a bitmap glyph is placed at pen + offset and scaled by the title's scale with bilinear
sampling at sub-pixel positions (cv2.warpAffine, as a GPU samples a glyph texture without mipmaps); a vector glyph
is rasterised by FreeType at the pixel size; the pen advances by the glyph's advance plus a letter spacing.

Exclusion (`inventory`): every font of a test title (CLANNAD PCSG00415, Dance with Devils PCSG00807: `test` in the
corpus index, and CLANNAD by name) and every other font whose content or file sha256 equals one of them is never
drawn; so is a font whose decoder disagreed with its own title's pages (fidelity `mismatch`) unless it is verified
on another title, and a font with too few glyphs to draw a line.
"""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

SPACES = {" ", "\u3000"}
# The farm's test titles (index.json split "test", and CLANNAD by name: the fonts README holds its fonts out).
TEST_TITLES = ("PCSG00415", "PCSG00807")
# A font is drawable only with at least this many glyphs (a 100-glyph bold or quest font draws no line).
MIN_GLYPHS = 300
# Fidelity verdicts whose decoders are trusted for the glyph-to-character map (font_render.py fidelity): the font
# matched its title's pages (verified), or it is another face/size of a verified font's file (decoder-verified).
TRUSTED = ("verified", "decoder-verified")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(font_dir):
    return json.loads((Path(font_dir) / "font.json").read_text(encoding="utf-8"))


def fidelity_of(meta):
    """{title: fidelity record} of a font: `fidelity_by_title` (bitmap fonts) or the single `fidelity`."""
    out = dict(meta.get("fidelity_by_title") or {})
    f = meta.get("fidelity")
    if f and f.get("title_tested"):
        out.setdefault(f["title_tested"], f)
    return out


class AtlasFont:
    """A bitmap font from its atlas: glyph(ch) -> (float32 coverage, ox, oy, adv) at scale 1."""

    kind = "bitmap"

    def __init__(self, font_dir):
        self.dir = Path(font_dir)
        self.meta = load_meta(self.dir)
        table = json.loads((self.dir / self.meta.get("glyphs_file", "glyphs.json")).read_text(encoding="utf-8"))
        self.chars = table["chars"]
        self.cell = table.get("cell") or self.meta.get("cell")
        atlas = cv2.imread(str(self.dir / table.get("atlas", "atlas.png")), cv2.IMREAD_UNCHANGED)
        if atlas is None:
            raise FileNotFoundError(self.dir / "atlas.png")
        if atlas.ndim == 3:
            atlas = atlas[..., -1] if atlas.shape[2] == 4 else cv2.cvtColor(atlas, cv2.COLOR_BGR2GRAY)
        self.atlas = atlas
        self.line_height = float(self.meta.get("line_height") or (self.cell[1] if self.cell else
                                 max(r[3] for r in self.chars.values())))
        self.pixel_size = float(self.meta.get("pixel_size") or self.line_height)
        self.deny = set()           # characters the glyph check found drawn as another character (synth6.py)
        self._cache = {}

    @property
    def name(self):
        return f"{self.meta['title_id']}/{self.meta['font_id']}"

    def has(self, ch):
        return (ch in self.chars or ch in SPACES) and ch not in self.deny

    def glyph(self, ch):
        if ch in self._cache:
            return self._cache[ch]
        if ch not in self.chars:
            if ch in SPACES:
                adv = (self.cell[0] if self.cell else self.line_height) * (0.5 if ch == " " else 1.0)
                return np.zeros((1, 1), np.float32), 0.0, 0.0, float(adv)
            raise KeyError(ch)
        x, y, w, h, ox, oy, adv = self.chars[ch]
        img = self.atlas[y:y + h, x:x + w].astype(np.float32) / 255.0
        out = (img, float(ox), float(oy), float(adv))
        self._cache[ch] = out
        return out


class OutlineFont:
    """A vector font (OTF, TTF, PVF) rasterised by FreeType at a pixel size (Pillow's ImageFont)."""

    kind = "vector"

    def __init__(self, path, meta=None, size=24.0):
        self.path = Path(path)
        self.meta = meta or {"title_id": "-", "font_id": self.path.stem}
        chars_file = self.path.parent / "chars.txt"
        if chars_file.exists():
            self.cmap = set(chars_file.read_text(encoding="utf-8"))
        else:
            from fontTools.ttLib import TTFont
            self.cmap = {chr(c) for c in (TTFont(str(self.path), lazy=True).getBestCmap() or {})}
        self.size = None
        self.deny = set()
        self.set_size(size)

    @property
    def name(self):
        return f"{self.meta['title_id']}/{self.meta['font_id']}"

    def has(self, ch):
        return (ch in self.cmap or ch in SPACES) and ch not in self.deny

    def set_size(self, size):
        from PIL import ImageFont
        size = round(float(size) * 4) / 4          # FreeType takes fractional sizes; libpvf games use them
        if size != self.size:
            self.size = size
            self.font = ImageFont.truetype(str(self.path), size)
            asc, desc = self.font.getmetrics()
            self.ascent, self.line_height = float(asc), float(asc + desc)
            self.pixel_size = size
            self._cache = {}
        return self

    def glyph(self, ch):
        if ch in self._cache:
            return self._cache[ch]
        adv = float(self.font.getlength(ch))
        if ch in SPACES:
            out = (np.zeros((1, 1), np.float32), 0.0, 0.0, adv)
        else:
            mask, (ox, oy) = self.font.getmask2(ch, mode="L", anchor="la")
            w, h = mask.size
            img = np.asarray(mask, dtype=np.uint8).reshape(h, w).astype(np.float32) / 255.0
            out = (img, float(ox), float(oy), adv)
        self._cache[ch] = out
        return out


def open_font(font_dir, size=None):
    """AtlasFont or OutlineFont for a harvested font directory (size: a vector font's pixel size)."""
    meta = load_meta(font_dir)
    if meta["kind"] == "bitmap":
        return AtlasFont(font_dir)
    return OutlineFont(Path(font_dir) / meta["file"], meta, size or meta.get("pixel_size") or 24.0)


def draw_line(font, text, scale=1.0, spacing=0.0, x=0.0, y=0.0, shape=None):
    """Draw `text` with its pen starting at (x, y) (the line top), like a VN engine: each glyph at pen + offset
    (bitmap offsets and advances times `scale`), advancing by the advance plus `spacing` px.

    Returns (coverage float32 of `shape` or of the line's own extent, cells): cells[i] = (char, x0, x1, y0, y1),
    the glyph's pen cell (pen to pen + advance, line top to line top + line height) in the canvas's coordinates.
    A vector font is drawn at its current size (scale is ignored: set_size picks the size)."""
    s = scale if font.kind == "bitmap" else 1.0
    lh = font.line_height * s
    items, pen = [], float(x)
    for ch in text:
        img, ox, oy, adv = font.glyph(ch)
        items.append((ch, img, pen + ox * s, y + oy * s, pen, adv * s))
        pen += adv * s + spacing
    if shape is None:
        w = int(math.ceil(max([pen] + [gx + g.shape[1] * s for _, g, gx, _, _, _ in items]))) + 2
        h = int(math.ceil(max([y + lh] + [gy + g.shape[0] * s for _, g, _, gy, _, _ in items]))) + 2
        shape = (h, w)
    canvas = np.zeros(shape, np.float32)
    cells = []
    H, W = shape
    for ch, img, gx, gy, x0, adv in items:
        cells.append((ch, x0, x0 + adv, y, y + lh))
        if ch in SPACES or img.size <= 1:
            continue
        gh, gw = img.shape
        if s == 1.0 and gx == int(gx) and gy == int(gy):
            xi, yi = int(gx), int(gy)
            patch, px, py = img, xi, yi
        else:
            xi, yi = int(math.floor(gx)), int(math.floor(gy))
            ow, oh = int(math.ceil(gx + gw * s)) - xi + 1, int(math.ceil(gy + gh * s)) - yi + 1
            M = np.array([[s, 0.0, gx - xi], [0.0, s, gy - yi]], np.float32)
            patch = cv2.warpAffine(img, M, (max(1, ow), max(1, oh)), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            px, py = xi, yi
        ys, xs = max(0, py), max(0, px)
        ye, xe = min(H, py + patch.shape[0]), min(W, px + patch.shape[1])
        if ye > ys and xe > xs:
            np.maximum(canvas[ys:ye, xs:xe], patch[ys - py:ye - py, xs - px:xe - px], out=canvas[ys:ye, xs:xe])
    return canvas, cells


def line_advance(font, text, scale=1.0, spacing=0.0):
    """The pen distance of `text` (the sum of advances plus spacing between glyphs)."""
    s = scale if font.kind == "bitmap" else 1.0
    return sum(font.glyph(c)[3] * s for c in text) + spacing * max(0, len(text) - 1)


# ----------------------------------------------------------------------------- inventory

def vertical_forms(font):
    """True when the font stores vertical-writing punctuation at the horizontal code points (、 and 。 in the
    top-right of their cell, as a tategaki game's font does): such a font would draw a horizontal line wrongly."""
    if font.kind == "vector":
        return False
    hits = 0
    for c in "、。":
        if c not in font.chars:
            continue
        g, ox, oy, adv = font.glyph(c)
        ys, xs = np.nonzero(g > 0.3)
        if not len(xs):
            continue
        cx = (ox + xs.mean()) / max(1.0, adv)
        cy = (oy + ys.mean()) / max(1.0, font.line_height)
        hits += cx > 0.55 and cy < 0.45
    return hits > 0


def scan(harvest):
    """Every font directory of a harvest root: [(dir, meta)] sorted by id."""
    out = []
    for fj in sorted(Path(harvest).glob("*/*/font.json")):
        out.append((fj.parent, load_meta(fj.parent)))
    return out


def inventory(harvest, test_titles=TEST_TITLES, train_titles=None):
    """The synth6 font inventory: one row per harvested font with its sha256s, fidelity and the decision.

    A font is excluded when it belongs to a test title, when its content sha256 or any file's sha256 equals a
    test-title font's (a byte-identical copy shipped by another game), when its decoder disagreed with its own
    title's pages (`mismatch`) and no title verified it, or when it has fewer than MIN_GLYPHS glyphs.
    train_titles: the farm titles marked `train`, recorded per font (`farm_train_titles`)."""
    rows = []
    for d, m in scan(harvest):
        files = {}
        for key in ("file", "glyphs_file"):
            n = m.get(key)
            if n and (d / n).exists():
                files[n] = sha256_file(d / n)
        if m["kind"] == "bitmap" and (d / "atlas.png").exists() and "atlas.png" not in files:
            files["atlas.png"] = sha256_file(d / "atlas.png")
        fid = fidelity_of(m)
        rows.append({
            "font": f"{m['title_id']}/{m['font_id']}", "title": m["title_id"], "kind": m["kind"],
            "format": m.get("format"), "face": m.get("face"), "system": bool(m.get("system")),
            "pixel_size": m.get("pixel_size"), "glyphs": m.get("glyph_count"),
            "vocr5_kanji": (m.get("coverage") or {}).get("vocr5_kanji"),
            "content_sha256": m.get("content_sha256"), "source_sha256": (m.get("source") or {}).get("sha256"),
            "files": files,
            "fidelity": {t: {"verdict": v.get("verdict"), "median_glyph_corr": v.get("median_glyph_corr"),
                             "native_size_or_scale": v.get("native_size_or_scale"),
                             "native_spacing": v.get("native_spacing"), "native_px": v.get("native_px")}
                         for t, v in sorted(fid.items())},
        })
    test_hashes = set()
    for r in rows:
        if r["title"] in test_titles:
            test_hashes |= {r["content_sha256"]} | set(r["files"].values())
    test_hashes.discard(None)
    for r in rows:
        why = []
        if r["title"] in test_titles:
            why.append("test-title font")
        elif ({r["content_sha256"]} | set(r["files"].values())) & test_hashes:
            why.append("byte-identical to a test-title font")
        verdicts = {v["verdict"] for v in r["fidelity"].values()}
        if "mismatch" in verdicts and not verdicts & set(TRUSTED):
            why.append("decoder mismatch on its own title's pages")
        if (r["glyphs"] or 0) < MIN_GLYPHS:
            why.append(f"fewer than {MIN_GLYPHS} glyphs")
        elif not why and r["kind"] == "bitmap" and vertical_forms(open_font(Path(harvest) / r["font"])):
            why.append("vertical punctuation forms (a tategaki font)")
        r["excluded"] = why
        r["trusted_map"] = bool(verdicts & set(TRUSTED)) or r["kind"] == "vector" or r["system"]
        if train_titles is not None:
            r["farm_train_titles"] = sorted(t for t, v in r["fidelity"].items()
                                            if t in train_titles and v["verdict"] in TRUSTED)
    return rows


def dialogue_fonts(rows, titles):
    """{title: (font id, size_or_scale, spacing, corr)}: each title's verified dialogue font (the best median
    glyph correlation among fonts verified on that title), for titles in `titles`."""
    best = {}
    for r in rows:
        if r["excluded"]:
            continue
        for t, v in r["fidelity"].items():
            if t not in titles or v["verdict"] != "verified":
                continue
            corr = v["median_glyph_corr"] or 0.0
            if t not in best or corr > best[t][3]:
                best[t] = (r["font"], v["native_size_or_scale"], v["native_spacing"] or 0.0, corr)
    return best


@lru_cache(maxsize=None)
def cached_font(harvest, font_id):
    return open_font(Path(harvest) / font_id)

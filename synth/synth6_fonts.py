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
corpus index, and CLANNAD by name), every other font whose content or file sha256 equals one of them, and every
bitmap font whose glyph bitmaps equal a test-title font's for 95% of their common characters (the same font files
another release ships) is never drawn; so is a font whose decoder disagreed with its own title's pages (fidelity
`mismatch`) unless it is verified on another title, and a font with too few glyphs to draw a line. `misplaced`
lists the characters a font would draw outside its own pen cell or not visibly; the renderer never draws those.
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


def misplaced(font, chars):
    """The characters of `chars` that `font` would not draw where a reader sees them: a glyph whose ink lies mostly
    (more than 40% of its coverage) outside its own pen cell (pen - 1 to pen + advance + 1 px), such as a middle dot
    stored in a full cell with a half-width advance, which lands inside the next character; or a glyph with no
    visible ink (maximum coverage under 0.3). Spaces are never listed."""
    out = []
    for c in chars:
        if c in SPACES or not font.has(c):
            continue
        g, ox, _, adv = font.glyph(c)
        if g.size <= 1 or float(g.max()) < 0.3:
            out.append(c)
            continue
        w = (g * (g >= 0.3)).sum(axis=0)
        xs = ox + np.arange(g.shape[1])
        if float(w[(xs < -1) | (xs > adv + 1)].sum()) > 0.4 * float(w.sum()):
            out.append(c)
    return out


def shape_vec(font, ch, size=24):
    """A glyph's ink box padded to a square (so a dash stays a dash) and resized to size x size, unit norm, without
    centring (so a filled box differs from a hollow one); None for an empty glyph."""
    g = font.glyph(ch)[0]
    if g.size <= 1:
        return None
    ys, xs = np.nonzero(g > 0.25)
    if len(xs) < 2:
        return None
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    n = max(h, w)
    sq = np.zeros((n, n), np.float32)
    sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = g
    v = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA).ravel()
    nv = float(np.linalg.norm(v))
    return v / nv if nv > 1e-6 else None


@lru_cache(maxsize=None)
def _cp932_group(lead):
    """The assigned characters of one Shift-JIS group in code order: a two-byte lead byte's row, or (lead 0) ASCII."""
    out = []
    if lead == 0:
        for b in range(0x20, 0x7F):
            out.append(bytes([b]).decode("cp932"))
        return tuple(out)
    for t in list(range(0x40, 0x7F)) + list(range(0x80, 0xFD)):
        try:
            c = bytes([lead, t]).decode("cp932")
        except UnicodeDecodeError:
            continue
        if len(c) == 1:
            out.append(c)
    return tuple(out)


def cp932_shift(ch, d):
    """The character d *assigned* Shift-JIS (cp932) codes away from ch in its row (unassigned codes are skipped, as a
    decoder's glyph table skips them), or None."""
    try:
        b = ch.encode("cp932")
    except UnicodeEncodeError:
        return None
    if len(b) == 1 and not 0x20 <= b[0] < 0x7F:
        return None
    g = _cp932_group(b[0] if len(b) == 2 else 0)
    if ch not in g:
        return None
    i = g.index(ch) + d
    return g[i] if 0 <= i < len(g) else None


SHIFTS = (-2, -1, 1, 2)


def swapped(font, chars, refs, min_run=8, min_gain=0.05, max_gap=2):
    """The characters of `chars` (symbols, punctuation, Latin and digits: shapes the OCR checks barely read) that
    `font` draws from a decoder table off by one or two codes (NPJH50649's lt18 draws 」 as 「, ＝ as ÷, ○ as ★
    ... from 0x8176 on). For each character in Shift-JIS order, its glyph (shape_vec) is compared with the
    reference glyphs (any of `refs`) of itself and of its neighbours 1-2 codes away; a character whose best match is
    a neighbour at shift d (by `min_gain` or more) votes for d. A run of `min_run` or more votes for one d, in code
    order within a Shift-JIS lead byte (single-byte codes form one group; up to `max_gap` other characters between
    votes), is a shifted table: every
    character of the run is listed. A single vote is not enough: tiny glyphs (。．) match their neighbours by chance."""
    rv = {}

    def ref_vec(c):
        if c not in rv:
            rv[c] = [v for v in (shape_vec(r, c) for r in refs if r.has(c)) if v is not None]
        return rv[c]
    rows = []
    for c in chars:
        if c in SPACES or not font.has(c):
            continue
        try:
            code = c.encode("cp932")
        except UnicodeEncodeError:
            continue
        v = shape_vec(font, c)
        own = ref_vec(c)
        if v is None or not own:
            continue
        o = max(float(v @ r) for r in own)
        best, gain = 0, 0.0
        for d in SHIFTS:
            t = cp932_shift(c, d)
            if t is not None and ref_vec(t):
                g = max(float(v @ r) for r in ref_vec(t)) - o
                if g > gain:
                    best, gain = d, g
        vote = best if gain >= min_gain else 0
        rows.append((len(code), code[0] if len(code) == 2 else 0, int.from_bytes(code, "big"), c, vote))
    rows.sort()
    out = []
    i = 0
    while i < len(rows):
        if rows[i][4] == 0:
            i += 1
            continue
        d, lead, j, gap, last = rows[i][4], rows[i][:2], i, 0, i
        votes = 0
        while j < len(rows) and rows[j][:2] == lead and gap <= max_gap:
            if rows[j][4] == d:
                votes, gap, last = votes + 1, 0, j
            else:
                gap += 1
            j += 1
        if votes >= min_run:
            out += [r[3] for r in rows[i:last + 1]]
        i = last + 1
    return out


def glyph_digests(font_dir):
    """{character: sha1 of its glyph bitmap and its size} of a bitmap font (None for a vector font)."""
    meta = load_meta(font_dir)
    if meta["kind"] != "bitmap":
        return None
    f = AtlasFont(font_dir)
    out = {}
    for c, (x, y, w, h, _, _, _) in f.chars.items():
        b = f.atlas[y:y + h, x:x + w]
        out[c] = hashlib.sha1(b.tobytes() + bytes(f"{w},{h}", "ascii")).hexdigest()
    return out


def _levels(font):
    """A bitmap font's atlas as level indices: each grey value's rank among the atlas's distinct values (a 4-bit
    font has 16), or 16 rank bins when it has more distinct values (an anti-aliased 8-bit atlas)."""
    a = font.atlas
    vals = np.unique(a)
    if len(vals) <= 32:
        lut = np.zeros(256, np.int16)
        lut[vals] = np.arange(len(vals))
        return lut[a]
    q = np.quantile(a[a > 0], np.linspace(0, 1, 17)[1:-1]) if (a > 0).any() else np.zeros(15)
    out = np.searchsorted(q, a, side="right").astype(np.int16) + 1
    out[a == 0] = 0
    return out


def raster_share(font, ref, n=600, seed=0):
    """The share of `ref`'s glyphs that `font` holds as the same raster, whatever the palette.

    For up to `n` common characters, both glyphs are taken as level indices (_levels) and placed at their pen offsets
    (ox, oy) on one canvas; one global shift of `font`'s glyphs (dx, dy in -1..1) is chosen where their inked pixels
    overlap `ref`'s most. The tone map from `ref`'s levels to `font`'s is the modal font level of each ref level over
    pixels inked in both. A glyph counts when every pixel inked in either, with `ref`'s levels mapped, lies within
    one level of `font`'s. The same raster re-quantised through another palette (NEAR_SHARE's byte check misses it)
    scores near 1; another raster of the same typeface near 0."""
    if font.kind != "bitmap" or ref.kind != "bitmap":
        return 0.0
    la, lb = _levels(font), _levels(ref)
    common = sorted(c for c in font.chars.keys() & ref.chars.keys() if c not in SPACES)
    if not common:
        return 0.0
    rng = np.random.default_rng(seed)
    pick = [common[i] for i in rng.choice(len(common), size=min(n, len(common)), replace=False)]

    def placed(f, lv, c, H, W, dx=0, dy=0):
        x, y, w, h, ox, oy, _ = f.chars[c]
        out = np.zeros((H, W), np.int16)
        px, py = int(round(ox)) + 2 + dx, int(round(oy)) + 2 + dy
        g = lv[y:y + h, x:x + w]
        ys, xs = max(0, py), max(0, px)
        ye, xe = min(H, py + h), min(W, px + w)
        if ye > ys and xe > xs:
            out[ys:ye, xs:xe] = g[ys - py:ye - py, xs - px:xe - px]
        return out
    H = int(max(max(f.chars[c][3] + abs(f.chars[c][5]) for c in pick) for f in (font, ref))) + 6
    W = int(max(max(f.chars[c][2] + abs(f.chars[c][4]) for c in pick) for f in (font, ref))) + 6
    refs = [placed(ref, lb, c, H, W) for c in pick]
    best = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ov = sum(int(((placed(font, la, c, H, W, dx, dy) > 0) & (r > 0)).sum()) for c, r in zip(pick[:80], refs))
            if best is None or ov > best[0]:
                best = (ov, dx, dy)
    fonts = [placed(font, la, c, H, W, best[1], best[2]) for c in pick]
    nb = int(max(int(r.max()) for r in refs)) + 1
    na = int(max(int(g.max()) for g in fonts)) + 1
    joint = np.zeros((nb, na), np.int64)
    for g, r in zip(fonts, refs):
        m = (g > 0) | (r > 0)           # zeros count: a re-quantisation may drop the faintest levels to 0
        np.add.at(joint, (r[m], g[m]), 1)
    tone = np.where(joint.sum(axis=1) > 0, joint.argmax(axis=1), 0)
    tone[0] = 0
    ok = 0
    for g, r in zip(fonts, refs):
        m = (g > 0) | (r > 0)
        if m.any() and int(np.abs(tone[r[m]] - g[m]).max()) <= 1:
            ok += 1
    return ok / len(pick)


RASTER_SHARE = 0.5      # raster_share at or above this marks a font as holding a test-title font's raster


NEAR_SHARE = 0.95       # a font whose glyphs are this identical to a test-title font's is excluded too
NEAR_MIN_COMMON = 500


def near_identical(harvest, rows, test_titles):
    """{font id: (test font id, share)} of every bitmap font that is not a test title's whose glyph bitmaps equal a
    test-title font's for at least NEAR_SHARE of the characters both hold (another release shipping the same font
    files, re-encoded: an exact-hash check misses those)."""
    tests = {r["font"]: glyph_digests(Path(harvest) / r["font"]) for r in rows if r["title"] in test_titles}
    tests = {k: v for k, v in tests.items() if v}
    out = {}
    for r in rows:
        if r["title"] in test_titles or r["kind"] != "bitmap":
            continue
        d = glyph_digests(Path(harvest) / r["font"])
        for tf, td in tests.items():
            common = d.keys() & td.keys()
            if len(common) < NEAR_MIN_COMMON:
                continue
            share = sum(d[c] == td[c] for c in common) / len(common)
            if share >= NEAR_SHARE and share > out.get(r["font"], ("", 0.0))[1]:
                out[r["font"]] = (tf, round(share, 4))
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
    near = near_identical(harvest, rows, test_titles)
    for r in rows:
        why = []
        if r["title"] in test_titles:
            why.append("test-title font")
        elif ({r["content_sha256"]} | set(r["files"].values())) & test_hashes:
            why.append("byte-identical to a test-title font")
        elif r["font"] in near:
            why.append("near-identical to a test-title font (glyph bitmaps)")
            r["near_test_font"] = {"font": near[r["font"]][0], "identical_glyph_share": near[r["font"]][1]}
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

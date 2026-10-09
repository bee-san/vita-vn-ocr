#!/usr/bin/env python3
"""Synthetic PS Vita visual-novel dialogue renderer.

A sample is a dialogue-box *region*, like the one a user selects with Square:
bright text on a translucent dark box over a background. Lines are cropped with
common.find_lines (an exact port of the overlay's line finder), so the crops are
the crops the recognizer sees on the Vita. A crop is kept only when it matches one
rendered line unambiguously; its label is that line's text (ruby and the
page-cursor icon are distractors and never part of a label).

  render.py preview [--n 12]
      -> synth/preview_regions.png, synth/preview_crops.png
  render.py shards --out DIR --shards N --per-shard M [--workers W] [--seed S]
      -> DIR/synth_00000.npz ... (uint8 grey crops at native size + labels)
"""
import argparse
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, find_lines, is_test_vn, load_vocab, to_gray  # noqa: E402

FONT_DIR = DATA / "fonts"
# Style buckets approximate what Vita VNs ship: Shin Go/Rodin-like gothics,
# Seurat-like rounded faces, Ryumin/Matisse-like minchos, plus a few pen, pixel
# and display faces for robustness.
CATEGORY = {
    "gothic": ["notosansjp", "mplus1p", "mplus2", "sawarabigothic", "zenkakugothicnew",
               "zenkakugothicantique", "bizudgothic", "bizudpgothic", "murecho", "ibmplexsansjp"],
    "rounded": ["mplusrounded1c", "zenmarugothic", "kiwimaru"],
    "mincho": ["notoserifjp", "sawarabimincho", "zenoldmincho", "zenantique", "shipporimincho",
               "shipporiantique", "bizudmincho", "hinamincho", "kaiseiopti", "kaiseitokumin",
               "yujisyuku"],
    "pen": ["kleeone", "yuseimagic", "yomogi", "zenkurenaido"],
    "pixel": ["dotgothic16"],
    "display": ["rocknrollone", "mochiypopone", "hachimarupop"],
}
CAT_WEIGHT = {"gothic": 0.40, "rounded": 0.18, "mincho": 0.27, "pen": 0.07, "pixel": 0.05,
              "display": 0.03}
POOL_WEIGHT = {"dialogue": 0.45, "narration": 0.35, "wiki": 0.20}
NO_START = set("、。，．・：；？！ー―…‥」』）】〕〉》ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々")
NO_END = set("「『（【〔〈《")

R = None          # per-process resources, loaded before workers fork
_FONTS = {}
_AXES = {}


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
        shots = sorted((DATA / "vndb" / "shots").glob("*.jpg"))
        self.bg_shots = [p for p in shots if not is_test_vn(p.name.split("_")[0])]
        self.katakana = [c for c in self.vocab if 0x30A1 <= ord(c) <= 0x30FA]
        self.hiragana = [c for c in self.vocab if 0x3041 <= ord(c) <= 0x3096]
        freq = json.loads((DATA / "corpus" / "char_freq.json").read_text(encoding="utf-8"))
        self.kanji = [c for c in freq if is_kanji(c) and c in self.vocab_set][:2000]

    def _fonts(self):
        cover_path = FONT_DIR / "coverage.json"
        cover = json.loads(cover_path.read_text(encoding="utf-8")) if cover_path.exists() else {}
        out, changed = [], False
        for cat, fams in CATEGORY.items():
            for fam in fams:
                for path in sorted((FONT_DIR / fam).glob("*.[to]tf")):
                    key = f"{fam}/{path.name}"
                    if key not in cover:
                        from fontTools.ttLib import TTFont
                        cmap = TTFont(str(path), lazy=True).getBestCmap() or {}
                        cover[key] = "".join(c for c in self.vocab if ord(c) in cmap)
                        changed = True
                    chars = set(cover[key])
                    if len(chars) > 1000:
                        out.append((cat, fam, path, chars))
        if changed:
            cover_path.write_text(json.dumps(cover, ensure_ascii=False), encoding="utf-8")
        return out


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or c == "々"


def get_font(path, size, wght):
    key = (str(path), size, wght)
    font = _FONTS.get(key)
    if font is None:
        if len(_FONTS) > 300:
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


def choose(rng, weights):
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=np.float64)
    return names[rng.choice(len(names), p=p / p.sum())]


def sample_text(rng):
    pool = choose(rng, POOL_WEIGHT)
    items = R.pools[pool]
    s = "".join(c for c in items[rng.integers(len(items))] if c in R.vocab_set)
    if pool == "narration" and rng.random() < 0.15:
        s = "「" + s + "」"
    r = rng.random()
    if r < 0.05:
        s = "……" + s
    elif r < 0.08:
        s = "――" + s
    elif r < 0.11 and s.endswith("。"):
        s = s[:-1] + ("！？" if rng.random() < 0.5 else "……")
    return s


def wrap(text, cpl, max_lines):
    """Greedy wrap with simple kinsoku: closing marks hang on the previous line,
    opening brackets never end a line."""
    lines, i = [], 0
    while i < len(text) and len(lines) < max_lines:
        j = min(i + cpl, len(text))
        hang = 0
        while j < len(text) and text[j] in NO_START and hang < 2:
            j, hang = j + 1, hang + 1
        while j - i > 1 and j < len(text) and text[j - 1] in NO_END:
            j -= 1
        lines.append(text[i:j])
        i = j
    return [ln for ln in lines if ln]


def random_name(rng):
    if rng.random() < 0.5:
        name = "".join(rng.choice(R.katakana, size=int(rng.integers(2, 6))))
    else:
        name = "".join(rng.choice(R.kanji, size=int(rng.integers(1, 4))))
    return f"【{name}】" if rng.random() < 0.5 else name


def background(rng, w, h):
    if R.bg_shots and rng.random() < 0.5:
        try:
            im = Image.open(R.bg_shots[rng.integers(len(R.bg_shots))]).convert("RGB")
            s = rng.uniform(1.0, 1.6)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
            if im.width >= w and im.height >= h:
                x0 = int(rng.integers(0, im.width - w + 1))
                y0 = int(rng.integers(0, im.height - h + 1))
                arr = np.asarray(im.crop((x0, y0, x0 + w, y0 + h)), np.float32)
                return arr[:, ::-1].copy() if rng.random() < 0.5 else arr.copy()
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
        small = rng.normal(0, rng.uniform(5, 30), (h // 8 + 2, w // 8 + 2, 3)).astype(np.float32)
        img += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 0, 255)


def render_region(rng):
    """Returns (rgb uint8 HxWx3, layout, meta) or None when the draw is unusable."""
    text = sample_text(rng)
    if len(text) < 2:
        return None
    fs = int(np.clip(rng.normal(24, 3.5), 16, 34))
    track = 0 if rng.random() < 0.5 else int(rng.integers(-1, 5))
    sw = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.40, 0.25, 0.05]))
    shadow = rng.random() < 0.35
    dx, dy = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    ml, mr = int(rng.integers(4, 31)), int(rng.integers(4, 41))
    cpl_cap = max(4, (930 - ml - mr - fs) // max(fs + track, 8))
    cpl = int(min(rng.integers(12, 33), cpl_cap))
    lines = wrap(text, cpl, int(rng.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.38, 0.12])))
    if rng.random() < 0.08:
        lines = [random_name(rng)] + lines[:3]
    if not lines:
        return None
    font = pick_font(rng, set("".join(lines)))
    if font is None:
        return None
    cat, fam, path, chars = font
    wght = pick_weight(rng, path)
    ss = 2 if rng.random() < 0.4 else 1
    f = get_font(path, fs * ss, wght)
    ruby = rng.random() < 0.08
    ruby_fs = max(8, int(fs * 0.45))
    ruby_gap = int(rng.choice([1, 2, 5], p=[0.4, 0.4, 0.2]))
    ruby_space = ruby_fs + ruby_gap if ruby else 0
    cursor = rng.random() < 0.12
    pitch = max(int(fs * rng.uniform(1.3, 1.9)), fs + 2 * sw + (dy if shadow else 0) + 6) + ruby_space
    mt = int(rng.integers(3, 21)) + ruby_space
    mb = int(rng.integers(3, 25))
    widths = [sum(f.getlength(c) for c in ln) / ss + track * (len(ln) - 1) for ln in lines]
    w = int(ml + max(widths) + mr + (fs if cursor else 0))
    h = int(mt + len(lines) * pitch + mb)
    if w > 960 or h > 544 or w < 16:
        return None

    fill = Image.new("L", (w * ss, h * ss), 0)
    stroke = Image.new("L", (w * ss, h * ss), 0) if (sw or shadow) else None
    dfill = ImageDraw.Draw(fill)
    dstroke = ImageDraw.Draw(stroke) if stroke is not None else None
    rf = get_font(path, ruby_fs * ss, wght) if ruby else None
    layout = []
    for i, ln in enumerate(lines):
        top = mt + i * pitch
        base = top + 0.88 * fs
        x, xs = float(ml), []
        for ch in ln:
            xs.append(x)
            dfill.text((x * ss, base * ss), ch, font=f, fill=255, anchor="ls")
            if dstroke is not None:
                dstroke.text((x * ss, base * ss), ch, font=f, fill=255, anchor="ls",
                             stroke_width=sw * ss, stroke_fill=255)
            x += f.getlength(ch) / ss + track
        x_end = x - track
        if ruby:
            draw_ruby(rng, ln, xs, f, ss, top, ruby_fs, ruby_gap, rf, dfill, dstroke, max(sw - 1, 0))
        if cursor and i == len(lines) - 1:
            s, cx, cy = fs * 0.5, x_end + fs * 0.3, top + fs * 0.3
            tri = [(cx * ss, cy * ss), ((cx + s) * ss, cy * ss), ((cx + s / 2) * ss, (cy + s * 0.8) * ss)]
            dfill.polygon(tri, fill=255)
            if dstroke is not None:
                dstroke.polygon(tri, fill=255)
            x_end = cx + s
        layout.append({"text": ln, "x0": ml, "x1": x_end, "top": top,
                       "y0": top - ruby_space - sw - 1, "y1": top + fs + sw + (dy if shadow else 0) + 1})

    if ss > 1:
        fill = fill.resize((w, h), Image.BOX)
        stroke = stroke.resize((w, h), Image.BOX) if stroke is not None else None
    if rng.random() < 0.10:
        fill = fill.point(lambda v: 255 if v >= 128 else 0)
        stroke = stroke.point(lambda v: 255 if v >= 128 else 0) if stroke is not None else None
    # Measured ink extents of each line's main text (bracket and comma glyphs sit
    # well inside their advance box, so the layout x0/x1 are not the ink edges).
    fill_arr = np.asarray(fill)
    for L in layout:
        cols = np.flatnonzero(fill_arr[int(L["top"]):int(L["top"]) + fs].max(axis=0) >= 128)
        L["ix0"], L["ix1"] = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (-99, -99)

    img = background(rng, w, h)
    box = (rng.uniform(0, 60, 3) * rng.uniform(0.3, 1.0, 3)).astype(np.float32)
    if rng.random() < 0.2:
        a = np.linspace(rng.uniform(0.45, 0.95), rng.uniform(0.45, 0.95), h, dtype=np.float32)[:, None, None]
    else:
        a = np.float32(rng.uniform(0.55, 0.93))
    img = img * (1 - a) + box * a
    fill_c = rng.uniform(232, 255, 3) if rng.random() < 0.85 else rng.uniform(195, 255, 3)
    stroke_c = rng.uniform(0, 70, 3) * rng.uniform(0.3, 1.0, 3)
    if shadow:
        src = np.asarray(stroke if sw else fill, np.float32) / 255.0
        sh = np.zeros_like(src)
        sh[dy:, dx:] = src[:h - dy, :w - dx]
        if rng.random() < 0.5:
            sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(0.5, 1.2))
        sa = sh[..., None] * rng.uniform(0.5, 1.0)
        img = img * (1 - sa) + rng.uniform(0, 30, 3) * sa
    if sw:
        sm = np.asarray(stroke, np.float32)[..., None] / 255.0
        img = img * (1 - sm) + stroke_c * sm
    fm = np.asarray(fill, np.float32)[..., None] / 255.0
    img = img * (1 - fm) + fill_c * fm
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(0.3, 0.8))
    if rng.random() < 0.25:
        img = img + rng.normal(0, rng.uniform(1, 4), img.shape)
    out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "JPEG", quality=int(rng.integers(55, 96)))
        out = np.asarray(Image.open(buf).convert("RGB"))
    meta = {"fam": fam, "cat": cat, "fs": fs, "ss": ss, "sw": sw, "shadow": bool(shadow),
            "ruby": bool(ruby), "cursor": bool(cursor)}
    return out, layout, meta


def pick_font(rng, chars):
    cands = [f for f in R.by_cat[choose(rng, {c: CAT_WEIGHT[c] for c in R.by_cat})]
             if chars <= f[3]]
    if not cands:
        cands = [f for f in R.fonts if chars <= f[3]]
    return cands[rng.integers(len(cands))] if cands else None


def draw_ruby(rng, ln, xs, f, ss, top, ruby_fs, gap, rf, dfill, dstroke, rsw):
    i = 0
    while i < len(ln):
        if not is_kanji(ln[i]):
            i += 1
            continue
        j = i
        while j < len(ln) and is_kanji(ln[j]):
            j += 1
        if rng.random() < 0.6:
            reading = "".join(rng.choice(R.hiragana, size=(j - i) * int(rng.integers(1, 3))))
            span0, span1 = xs[i], xs[j - 1] + f.getlength(ln[j - 1]) / ss
            width = sum(rf.getlength(c) for c in reading) / ss
            x = (span0 + span1 - width) / 2
            base = top - gap - 0.12 * ruby_fs
            for ch in reading:
                dfill.text((x * ss, base * ss), ch, font=rf, fill=255, anchor="ls")
                if dstroke is not None:
                    dstroke.text((x * ss, base * ss), ch, font=rf, fill=255, anchor="ls",
                                 stroke_width=rsw * ss, stroke_fill=255)
                x += rf.getlength(ch) / ss
        i = j


def extract(img, layout):
    """Match find_lines detections to rendered lines; return [(grey crop, label, box)]."""
    dets = find_lines(img)
    if dets is None:
        return None
    ink = [(y + 4, y + h - 5) for (x, y, w, h) in dets]
    ov = np.array([[min(b, L["y1"]) - max(a, L["y0"]) for L in layout] for a, b in ink], np.float32)
    out = []
    if not len(dets):
        return out
    for d, (x, y, w, h) in enumerate(dets):
        li = int(ov[d].argmax())
        L = layout[li]
        if ov[d, li] <= 0 or int(ov[:, li].argmax()) != d:
            continue
        if (np.delete(ov[d], li) > 1).any():           # spans two rendered lines
            continue
        if ov[d, li] < 0.5 * min(b - a for a, b in [ink[d]] + [(L["y0"], L["y1"])]):
            continue
        # find_lines pads ink by exactly 4 px unless clamped at the region edge, so
        # the crop must hug this line's own ink: no cut-off line, and no bright
        # background text pulled into the crop beside it.
        if x > 0 and abs(x + 4 - L["ix0"]) > 3:
            continue
        if x + w < img.shape[1] and abs(x + w - 4 - L["ix1"]) > 3:
            continue
        out.append((to_gray(img[y:y + h, x:x + w]), L["text"], (x, y, w, h)))
    return out


def init(pool_limit=300_000):
    global R
    R = Resources(pool_limit)


def run_shard(job):
    idx, n_lines, seed, out_dir = job
    path = Path(out_dir) / f"synth_{idx:05d}.npz"
    if path.exists():
        return idx, 0, 0, 0.0
    rng = np.random.default_rng(seed + idx * 1_000_003)
    t0, crops, labels, metas, regions, rejected = time.time(), [], [], [], 0, 0
    while len(crops) < n_lines:
        r = render_region(rng)
        if r is None:
            rejected += 1
            continue
        img, layout, meta = r
        got = extract(img, layout)
        regions += 1
        if not got:
            rejected += 1
            continue
        for g, label, _ in got:
            crops.append(g)
            labels.append(label)
            metas.append(meta)
    shapes = np.array([c.shape for c in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=np.concatenate([c.ravel() for c in crops]), offsets=offsets,
                        shapes=shapes, labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(metas)))
    tmp.rename(path)
    return idx, len(crops), regions, time.time() - t0


def preview(n):
    rng = np.random.default_rng(7)
    label_font = ImageFont.truetype(str(sorted((FONT_DIR / "notosansjp").glob("*.ttf"))[0]), 20)
    regions, crops = [], []
    while len(regions) < n:
        r = render_region(rng)
        if r is None:
            continue
        img, layout, meta = r
        got = extract(img, layout)
        if not got:
            continue
        vis = Image.fromarray(img)
        d = ImageDraw.Draw(vis)
        for (x, y, w, h) in find_lines(img):
            d.rectangle((x, y, x + w - 1, y + h - 1), outline=(255, 0, 0))
        for _, _, (x, y, w, h) in got:
            d.rectangle((x, y, x + w - 1, y + h - 1), outline=(0, 255, 0))
        regions.append(vis)
        crops += [(g, lab, meta) for g, lab, _ in got]
    width = max(im.width for im in regions)
    sheet = Image.new("RGB", (width, sum(im.height + 6 for im in regions)), (40, 40, 40))
    y = 0
    for im in regions:
        sheet.paste(im, (0, y))
        y += im.height + 6
    sheet.save(HERE / "preview_regions.png")
    rows = []
    for g, lab, meta in crops[:40]:
        g32 = Image.fromarray(g).resize((max(8, round(g.shape[1] * 32 / g.shape[0])), 32), Image.BILINEAR)
        row = Image.new("RGB", (1500, 36), (0, 0, 0))
        row.paste(g32.convert("RGB"), (0, 2))
        ImageDraw.Draw(row).text((min(g32.width, 900) + 10, 4), f"{lab}  [{meta['fam']} {meta['fs']}px]",
                                 font=label_font, fill=(255, 255, 160))
        rows.append(row)
    sheet = Image.new("RGB", (1500, 36 * len(rows)), (0, 0, 0))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, 36 * i))
    sheet.save(HERE / "preview_crops.png")
    print(f"preview: {len(regions)} regions, {len(crops)} crops", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preview", "shards"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=DATA / "synth")
    ap.add_argument("--shards", type=int, default=40)
    ap.add_argument("--per-shard", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()
    init()
    print(f"fonts {len(R.fonts)} in {sorted(R.by_cat)}, bg shots {len(R.bg_shots)}, "
          f"vocab {len(R.vocab)}", flush=True)
    if args.mode == "preview":
        preview(args.n)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.per_shard, args.seed, str(args.out)) for i in range(args.shards)]
    t0, total = time.time(), 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, regions, dt in pool.imap_unordered(run_shard, jobs):
            total += n
            print(f"shard {idx}: {n} lines from {regions} regions in {dt:.0f}s "
                  f"(total {total}, {time.time() - t0:.0f}s)", flush=True)
    print("SHARDS_DONE", flush=True)


if __name__ == "__main__":
    main()

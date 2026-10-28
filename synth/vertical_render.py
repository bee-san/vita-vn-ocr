#!/usr/bin/env python3
"""Synthetic vertical text columns for the dedicated vertical recognizer VOCR5-V (issue #99, docs/vertical.md).

  vertical_render.py shards --out DIR --shards N --per-shard M [--first F] [--seed S] [--workers W] [--vocab V]
  vertical_render.py preview --out PNG [--n 40] [--seed S]

One sample is one vertical column as the console's column finder cuts it (the column's main-text ink box plus a
pad of 4 px on the Vita and in manga views, 2 px on the PSP, with a little jitter), in the console's polarity (dark
text inverted, synth/common.line_dark's rule, as runtime/vocr_vertical.c hands columns to the recognizer) and
turned 90 degrees anticlockwise, so the column's top is the strip's left end: the recognizer reads it as a
horizontal line of characters lying on their side. Its label is the column's text, top to bottom.

Text, fonts, scenes and exclusions are synth3's (synth3.Resources: the `personal` licence profile VOCR5 used, every
protected title's names and scenes excluded). Typesetting is the fonts' own vertical layout through HarfBuzz
(Pillow's raqm layout, direction "ttb": the `vert` forms of ー ― … ～ 「」 （） and the vertical positions of 、。 and
small kana), one character at a time so the pitch can vary. Looks:
  vita    a Vita ADV / NVL column: a scene or synthetic background, optionally a translucent box, light or dark text,
          outline or shadow, 16-34 px
  psp     the same at PSP sizes (10-16 px), sometimes 1-bit glyphs, sometimes shown 2x and scaled back (Adrenaline)
  manga   a speech-balloon column: dark text on a white or toned balloon (sometimes light text outlined in black over
          a dark panel), 16-34 px, as NOBORU shows a page scaled to the Vita's width
Sometimes the crop also catches the edges of neighbouring columns or a run of furigana at its right, as finder
crops do. Shards are train2's format (data / offsets / shapes / labels / meta npz), crops at native size.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

GEN_VERSION = 3       # 1: the pinned v1 shards (data/manifests/vertical_shards_v1.json); 2: the scenes of the vertical
                      # development and test pages excluded (data/manifests/vertical_exclude_scenes.json); 3: also
                      # every other screenshot of the vertical sets' VNs (issue #159, docs/vertical.md "Limits")
EXCLUDE_SCENES = ROOT / "data" / "manifests" / "vertical_exclude_scenes.json"
VERTICAL_SETS = sorted((ROOT / "eval" / "manifests").glob("vertical-*.jsonl"))
OWNER_TEST_VNS = ("v11713",)   # Trick x Logic: the owner's private PSP test frames of the vertical test set


def vertical_set_vns():
    """Every VNDB VN with a page in the vertical development or test set (eval/manifests/vertical-*.jsonl, origin
    vndb) and the owner's private test VN: generation 3 cuts no scene from any of their screenshots."""
    vns = set(OWNER_TEST_VNS)
    for path in VERTICAL_SETS:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("origin") == "vndb":
                vns.add(row["vn"])
    return vns
PAD = {"vita": 4, "psp": 2, "manga": 4}
LOOK = {"vita": 0.45, "psp": 0.25, "manga": 0.30}
NO_START = set("、。，．」』）】〕〉》ー…‥～・！？!?ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
FURI_KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
S3 = None
_FONT_CACHE = {}


def init(vocab=None, gen=GEN_VERSION):
    """synth3's resources. From generation 2 on, the scene pools leave out every screenshot of the vertical
    development and test sets and their byte-identical copies (version 1, the pinned shards, drew from the whole
    pools: two test and three development pages were scenes; docs/vertical.md). From generation 3 on, they also
    leave out every other screenshot of those sets' VNs (vertical_set_vns), as docs/vertical.md asks of a later
    vertical model."""
    global S3
    import cv2

    import synth3
    cv2.setNumThreads(1)
    synth3.init(vocab=vocab)
    if gen >= 2:
        ex = set(json.loads(EXCLUDE_SCENES.read_text(encoding="utf-8"))["shots"])
        for k in list(synth3.R.bg):
            synth3.R.bg[k] = [p for p in synth3.R.bg[k] if Path(p).stem not in ex]
    if gen >= 3:
        vns = vertical_set_vns()
        for k in list(synth3.R.bg):
            synth3.R.bg[k] = [p for p in synth3.R.bg[k] if Path(p).name.split("_")[0] not in vns]
    S3 = synth3
    return synth3


def font(path, size):
    k = (path, size)
    f = _FONT_CACHE.get(k)
    if f is None:
        if len(_FONT_CACHE) > 400:
            _FONT_CACHE.clear()
        f = _FONT_CACHE[k] = ImageFont.truetype(str(path), size)
    return f


def to_gray(rgb):
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def line_dark(g, box):
    """synth/common.line_dark / vocr_line_dark: 1 when the box's text is darker than its border's median."""
    from common import line_dark as _ld
    return _ld(g, box)


def column_text(rng):
    """(text, kind): a run of characters a vertical column holds (1-24, most 5-16)."""
    r = rng.random()
    if r < 0.55:
        s, _, _ = S3.dialogue_text(rng, rng.random() < 0.3)
        kind = "dialogue"
    elif r < 0.85:
        s, _, _ = S3.narration_text(rng)
        kind = "narration"
    else:
        s = S3.clean(S3.template_line(rng))
        kind = "short"
    s = s.replace("　", "").replace(" ", "")
    if not s:
        return "", kind
    n = int(np.clip(round(rng.gamma(3.0, 3.6)), 1, 24))
    if len(s) <= n:
        return s, kind
    for _ in range(6):
        i = int(rng.integers(0, len(s) - n + 1)) if rng.random() < 0.6 else 0
        if s[i] not in NO_START or i == 0:
            return s[i:i + n], kind
    return s[:n], kind


def draw_column(layer, x_c, y0, text, f, fs, pitch, stroke=0, stroke_layer=None, ss=1):
    """Draws `text` top to bottom centred on x = x_c (canvas px) from y0, one character per cell of `pitch` px.
    Returns the y after the last cell."""
    d = ImageDraw.Draw(layer)
    ds = ImageDraw.Draw(stroke_layer) if stroke_layer is not None else None
    y = y0
    for ch in text:
        d.text((x_c, y), ch, font=f, fill=255, direction="ttb", anchor="mt")
        if ds is not None and stroke:
            ds.text((x_c, y), ch, font=f, fill=255, direction="ttb", anchor="mt", stroke_width=stroke * ss,
                    stroke_fill=255)
        y += pitch
    return y


def look_colours(rng, look, bg_mean):
    """(text rgb, outline rgb or None, shadow?, box) for a look."""
    if look == "manga":
        if rng.random() < 0.85:
            return np.array([rng.uniform(0, 45)] * 3), None, False, ("balloon", rng.uniform(225, 255))
        return np.array([rng.uniform(230, 255)] * 3), np.array([rng.uniform(0, 30)] * 3), False, ("panel", None)
    rng.random()          # drawn and unused: keeps the random stream the pinned v1 shards were rendered with
    box = None
    if rng.random() < 0.45:
        box = ("dark", rng.uniform(0.45, 0.9)) if rng.random() < 0.7 else ("light", rng.uniform(0.5, 0.95))
    light_bg = (bg_mean > 140 and box is None) or (box is not None and box[0] == "light")
    if light_bg and rng.random() < 0.8:
        tc = np.array([rng.uniform(0, 70)] * 3)
        if rng.random() < 0.25:
            tc = np.array([rng.uniform(40, 140), rng.uniform(0, 90), rng.uniform(0, 120)])
        oc = np.array([rng.uniform(220, 255)] * 3) if rng.random() < 0.2 else None
    else:
        tc = np.array([rng.uniform(215, 255)] * 3)
        if rng.random() < 0.2:
            tc = np.array([rng.uniform(150, 255), rng.uniform(150, 255), rng.uniform(120, 255)])
        oc = np.array([rng.uniform(0, 50)] * 3) if (box is None or rng.random() < 0.3) and rng.random() < 0.6 else None
    shadow = oc is None and rng.random() < 0.3
    return tc, oc, shadow, box


def sample(rng, vocab_set=None):
    """One column: {"crop": uint8 (h, w) turned anticlockwise, "text", "meta"} or None."""
    import cv2
    look = S3.choose(rng, LOOK)
    plat = "psp" if look == "psp" else "vita"
    text, kind = column_text(rng)
    if not text:
        return None
    fnt = S3.pick_font(rng, set(text), "vita" if plat == "vita" else "psp", cat=None)
    if fnt is None:
        return None
    if look == "psp":
        fs = int(rng.choice([10, 11, 12, 12, 13, 13, 14, 14, 15, 16]))
    elif look == "manga":
        fs = int(np.clip(round(rng.normal(24, 4.5)), 16, 34))
    else:
        fs = int(np.clip(round(rng.normal(25, 3.5)), 16, 34))
    mono = look == "psp" and rng.random() < 0.3
    ss = 1 if mono else 2
    F = font(fnt.path, fs * ss)
    pitch = int(round(fs * ss * rng.uniform(1.0, 1.18)))
    margin_x = int(round(fs * ss * rng.uniform(0.9, 1.6)))
    furi = rng.random() < 0.12 and fs >= 12
    neigh = rng.random() < 0.25
    W = 2 * margin_x + fs * ss + (int(fs * ss * 0.7) if furi else 0)
    top = int(round(fs * ss * rng.uniform(0.3, 1.0)))
    H = top * 2 + pitch * len(text)
    fill = Image.new("L", (W, H), 0)
    strk = Image.new("L", (W, H), 0)
    x_c = margin_x + fs * ss / 2
    tc_ = look_colours(rng, look, 128.0)
    stroke = 0
    if tc_[1] is not None:
        stroke = 1 if fs < 20 or rng.random() < 0.6 else 2
    draw_column(fill, x_c, top, text, F, fs * ss, pitch, stroke, strk if stroke else None, ss)
    main_fill = np.asarray(fill, np.float32) / 255.0
    extra = Image.new("L", (W, H), 0)
    if furi:                                  # a run of ruby beside some of the column, to its right
        fF = font(fnt.path, max(5, int(round(fs * ss * 0.5))))
        k0 = int(rng.integers(0, max(1, len(text) - 1)))
        k1 = int(min(len(text), k0 + rng.integers(1, 4)))
        ruby = "".join(FURI_KANA[int(rng.integers(len(FURI_KANA)))] for _ in range(2 * (k1 - k0)))
        draw_column(extra, x_c + fs * ss * 0.5 + fs * ss * 0.28 + 1 * ss, top + k0 * pitch, ruby, fF,
                    int(fs * ss * 0.5), int(round(pitch / 2)))
    if neigh:                                 # the edges of the neighbouring columns
        gap = int(round(fs * ss * rng.uniform(0.3, 0.9)))
        for side in (-1, 1):
            if rng.random() < 0.6:
                t2, _ = column_text(rng)
                if t2:
                    draw_column(extra, x_c + side * (fs * ss + gap), top + int(rng.integers(-pitch, pitch)),
                                t2[:len(text) + 2], F, fs * ss, pitch)
    ex = np.asarray(extra, np.float32) / 255.0
    allfill = np.maximum(main_fill, ex)
    sk = np.asarray(strk, np.float32) / 255.0 if stroke else None
    if stroke:
        grown = cv2.dilate((allfill > 0.05).astype(np.uint8), np.ones((2 * stroke * ss + 1,) * 2, np.uint8))
        sk = np.maximum(sk, grown.astype(np.float32) * (allfill > 0.05).max())
    # down to output size
    w, h = int(math.ceil(W / ss)), int(math.ceil(H / ss))

    def down(a):
        if a is None:
            return None
        if ss == 1:
            return a[:h, :w]
        im = Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)).resize((w, h), Image.BOX)
        return np.asarray(im, np.float32) / 255.0
    fa, ma, sa = down(allfill), down(main_fill), down(sk)
    if mono:
        fa = (fa > 0.5).astype(np.float32)
        ma = (ma > 0.5).astype(np.float32)
    # background
    tc, oc, shadow, box = tc_
    if look == "manga" and box[0] == "balloon":
        bg = np.full((h, w, 3), box[1], np.float32)
        if rng.random() < 0.05:                 # a light screentone
            yy, xx = np.mgrid[0:h, 0:w]
            p = int(rng.integers(3, 6))
            bg -= (((yy % p == 0) & (xx % p == 0)) * rng.uniform(15, 45))[..., None]
    else:
        bg, _tag = S3.background(rng, w, h, plat, plat, 1.0)
        bg = np.asarray(bg, np.float32)
        if bg.shape[:2] != (h, w):
            bg = cv2.resize(bg, (w, h))
        if box is not None and box[0] in ("dark", "light"):
            base = np.array([rng.uniform(0, 40)] * 3) if box[0] == "dark" else np.array([rng.uniform(215, 255)] * 3)
            bg = bg * (1 - box[1]) + base * box[1]
        elif box is not None and box[0] == "panel":
            bg = bg * 0.4
    img = bg.copy()
    if shadow:
        d = 1 if plat == "psp" else int(rng.integers(1, 3))
        sh = np.zeros_like(fa)
        sh[d:, d:] = fa[:-d, :-d]
        img = img * (1 - sh[..., None] * 0.8) + np.array([10, 10, 10]) * (sh[..., None] * 0.8)
    if sa is not None and oc is not None:
        img = img * (1 - sa[..., None]) + oc * sa[..., None]
    img = img * (1 - fa[..., None]) + tc * fa[..., None]
    img = np.clip(img, 0, 255)
    # capture
    if look == "psp" and rng.random() < 0.25:     # shown 2x (Adrenaline) and scaled back by the finder's caller
        up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.3:
        q = int(rng.integers(55, 96))
        ok, enc = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    if rng.random() < 0.1:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.7))
    g = to_gray(np.clip(img + 0.5, 0, 255).astype(np.uint8))
    # the crop: the main column's ink (fill and outline) plus the pad, with jitter
    ink = ma > 0.25
    if sa is not None:
        ink |= (sa > 0.25) & (cv2.dilate((ma > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return None
    pad = PAD[look]
    x0 = int(xs.min()) - pad + int(rng.integers(-1, 2))
    x1 = int(xs.max()) + 1 + pad + int(rng.integers(-1, 2))
    y0 = int(ys.min()) - pad + int(rng.integers(-1, 2))
    y1 = int(ys.max()) + 1 + pad + int(rng.integers(-1, 3))
    if rng.random() < 0.15:                      # a looser crop
        x1 += int(rng.integers(0, max(1, fs // 3)))
        x0 -= int(rng.integers(0, max(1, fs // 4)))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    dark = line_dark(g, (x0, y0, x1 - x0, y1 - y0))
    crop = g[y0:y1, x0:x1]
    if dark:
        crop = 255 - crop
    crop = np.ascontiguousarray(np.rot90(crop, 1))
    return {"crop": crop, "text": text,
            "meta": {"look": look, "fs": fs, "font": fnt.fam, "mono": mono, "furi": furi, "neigh": neigh,
                     "outline": oc is not None, "shadow": bool(shadow), "box": box[0] if box else None,
                     "dark": int(dark), "kind": kind}}


def run_shard(job):
    idx, n, seed, out, vocab, gen = job
    path = Path(out) / f"vsyn_{idx:05d}.npz"
    if path.exists():
        return idx, "exists"
    if S3 is None:
        init(vocab, gen)
    rng = np.random.default_rng([seed, idx])
    crops, labels, metas = [], [], []
    while len(crops) < n:
        s = sample(rng)
        if s is None or not s["text"]:
            continue
        crops.append(s["crop"])
        labels.append(s["text"])
        metas.append(s["meta"])
    shapes = np.array([c.shape for c in crops], np.int32)
    offs = np.concatenate([[0], np.cumsum([c.size for c in crops])]).astype(np.int64)   # n + 1 entries
    data = np.concatenate([c.ravel() for c in crops])
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, data=data, offsets=offs, shapes=shapes, labels=json.dumps(labels, ensure_ascii=False),
             meta=json.dumps({"gen": gen, "seed": seed, "shard": idx, "items": metas}, ensure_ascii=False))
    os.replace(tmp, path)
    return idx, "ok"


def cmd_shards(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.per_shard, args.seed, str(out), args.vocab, args.gen)
            for i in range(args.first, args.first + args.shards)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for idx, st in pool.imap_unordered(run_shard, jobs):
            print(idx, st, flush=True)
    print("SHARDS_DONE", flush=True)


def cmd_preview(args):
    init(args.vocab, args.gen)
    rng = np.random.default_rng(args.seed)
    items = []
    while len(items) < args.n:
        s = sample(rng)
        if s is not None:
            items.append(s)
    # show each crop turned back upright, side by side, zoomed to ~40 px wide; labels in a JSON next to it
    x, y, rowh = 0, 0, 0
    canvas = Image.new("L", (1280, 1280), 255)
    meta = []
    for i, s in enumerate(items):
        up = np.rot90(s["crop"], -1)
        z = max(1, min(3, int(round(40 / up.shape[1]))))
        im = Image.fromarray(up).resize((up.shape[1] * z, up.shape[0] * z), Image.NEAREST)
        if im.height > 600:
            im = im.resize((max(1, im.width * 600 // im.height), 600))
        if x + im.width + 6 > 1280:
            x, y, rowh = 0, y + rowh + 16, 0
        if y + im.height + 16 > 1280:
            break
        canvas.paste(im, (x + 2, y + 14))
        ImageDraw.Draw(canvas).text((x + 2, y), str(i + 1), fill=0)
        meta.append({"n": i + 1, "text": s["text"], **s["meta"]})
        x += im.width + 6
        rowh = max(rowh, im.height)
    canvas.save(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=0))
    print(len(meta), "->", args.out)


def cmd_manifest(args):
    """Pins rendered shards: file, sha256, bytes, crops; the generator version and the code's sha256."""
    import hashlib

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    shards = []
    for d in args.dir:
        for p in sorted(Path(d).glob("vsyn_*.npz")):
            z = np.load(p)
            shards.append({"dir": Path(d).name, "file": p.name, "sha256": sha(p), "bytes": p.stat().st_size,
                           "crops": int(len(z["shapes"]))})
    doc = {"kind": "vita-vn-ocr-vertical-shards", "generator": "synth/vertical_render.py", "gen_version": args.gen,
           "code_sha256": sha(Path(__file__)), "vocab": args.vocab, "seed": args.seed, "shards": shards,
           "crops": sum(s["crops"] for s in shards), "bytes": sum(s["bytes"] for s in shards)}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(len(shards), "shards,", doc["crops"], "crops ->", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shards")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--per-shard", type=int, default=10000)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--vocab")
    p.add_argument("--gen", type=int, default=GEN_VERSION, choices=(1, 2, 3),
                   help="1 reproduces the pinned v1 shards; 2 excludes the vertical sets' pages from the scenes; "
                        "3 (default) also every other screenshot of their VNs")
    p = sub.add_parser("preview")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vocab")
    p.add_argument("--gen", type=int, default=GEN_VERSION, choices=(1, 2, 3))
    p = sub.add_parser("manifest")
    p.add_argument("--dir", action="append", required=True)
    p.add_argument("--gen", type=int, default=GEN_VERSION, choices=(1, 2, 3))
    p.add_argument("--vocab", default="data/vocab_4987_proof.txt")
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"shards": cmd_shards, "preview": cmd_preview, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    main()

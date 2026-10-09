"""Japanese line crops from NVIDIA OCR-Synthetic-Multilingual-v1 pages, for pretraining.

A page is one HDF5 sample of the dataset: a JPEG plus a JSON annotation whose
line_bboxes hold each text line's box, quad and text. check() decides which lines
the recognizer can learn from; crop() cuts one the way the console sees a line:

  * horizontal, axis-aligned, fully on the page, box at least MIN_BOX_H px tall;
  * the text after the corpus normalisation (synth/build_corpus.py norm(): NFC,
    half-width kana to full-width, 〜 to ～, no spaces) has 2+ characters, all in
    data/vocab.txt, at least one kana or kanji and at most half ASCII;
  * the crop is the line's ink extent (row and column profile of gradient energy,
    polarity-agnostic like the line finders) plus PAD_FRAC of the ink height on every
    side, which is the 4 px the Vita line finder adds around 20-26 px glyphs;
  * integer grey (common.to_gray, the C side's luma), height 32, width <= 1024, CTC
    feasible at stride 8 (train.py's ctc_ok), and still visible in grey: enough
    contrast (CONTRAST) and not hue-on-hue text that vanishes in luma (GREY_RATIO).

Crops at most 48 px tall go to height 32 with bilinear interpolation, exactly as
train.py and the runtime resize. Taller ones are first area-downscaled to a random
28-40 px, a native Vita line height, so a page's big headings don't reach the model
as aliased bilinear 2-4x downscales the console never produces.

This module needs numpy, OpenCV and Pillow only; nvsynth/fetch.py adds h5py and
huggingface_hub to read and download the shards.
"""
import io
import json
import math
import sys
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from build_corpus import norm  # noqa: E402
from common import load_vocab, to_gray  # noqa: E402

H, MAX_W, STRIDE = 32, 1024, 8      # train.py's input height, width clamp and CTC stride
MIN_BOX_H = 14                      # annotated box height; ink is about 0.66 of it
MIN_CHARS = 2
MAX_ASCII = 0.5
PAD_FRAC = (0.10, 0.24)             # crop margin per side, as a fraction of the ink height
CONTRAST = 16                       # min of max(p99 - p50, p50 - p1) over the grey crop: below
                                    # it most lines are invisible once converted to grey
GREY_RATIO = 0.25                   # min grey_ratio() of the ink band: below it the text is
                                    # mostly hue on hue and largely vanishes in grey
AREA_ABOVE = 48                     # taller crops are area-downscaled first ...
MID_H = (28, 40)                    # ... to this native-looking height, then bilinear to 32
REASONS = ("rotated", "vertical", "clipped", "small", "short", "oov", "ascii")
CROP_REASONS = ("faint", "wide", "ctc")


def is_jp(c):
    o = ord(c)
    return 0x3041 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or c in "々〆〇ヶ"


def axis_aligned(quad, tol):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
    return max(abs(y0 - y1), abs(y3 - y2), abs(x0 - x3), abs(x1 - x2)) <= tol


def check(line, page_w, page_h, vocab):
    """(reason, label) for one annotated line; reason is None when the line is usable.
    The label is the normalised text whenever the line is horizontal (for OOV stats)."""
    x, y, w, h = line["bbox"]
    quad = line.get("quad")
    if quad is not None and not axis_aligned(quad, max(1.0, 0.02 * h)):
        return "rotated", None
    label = norm(line.get("text") or "")
    if h > w and len(label) > 1:
        return "vertical", None
    if x < -1 or y < -1 or x + w > page_w + 1 or y + h > page_h + 1:
        return "clipped", label
    if h < MIN_BOX_H:
        return "small", label
    if len(label) < MIN_CHARS:
        return "short", label
    if any(c not in vocab for c in label):
        return "oov", label
    if not any(map(is_jp, label)) or sum(ord(c) < 128 for c in label) > MAX_ASCII * len(label):
        return "ascii", label
    return None, label


def runs(on, gap):
    """[start, end) runs of True in a 1-D bool array, merging gaps of at most `gap`."""
    idx = np.flatnonzero(on)
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.concatenate([[idx[0]], idx[cut + 1]])
    ends = np.concatenate([idx[cut], [idx[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def energy(win):
    """|dx| + |dy| of a grey window (int32), the line finders' edge-strength measure."""
    g = win.astype(np.int16)
    e = np.zeros(g.shape, np.int32)
    e[:, :-1] += np.abs(np.diff(g, axis=1))
    e[:-1, :] += np.abs(np.diff(g, axis=0))
    return e


def ink_box(gray, bbox):
    """Ink extent (x0, y0, x1, y1) of a line inside its annotated box, plus whether the
    gradient profile gave a clean answer. NVIDIA boxes span the font's line height, so
    the ink sits about 0.17 of the box height inside the top and bottom edges; when the
    profile is unclear (busy backgrounds) that median trim is used instead."""
    ph, pw = gray.shape
    x, y, w, h = bbox
    bx0, by0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    bx1, by1 = min(pw, int(math.ceil(x + w))), min(ph, int(math.ceil(y + h)))
    fallback = (bx0, int(round(y + 0.17 * h)), bx1, int(round(y + h - 0.16 * h)))
    m = int(round(0.25 * h))
    ya, yb = max(0, by0 - m), min(ph, by1 + m)
    if bx1 - bx0 < 4 or by1 - by0 < 4:
        return fallback, False
    e = energy(gray[ya:yb, bx0:bx1])
    rows = e.sum(axis=1)
    peak = rows[by0 - ya:by1 - ya].max()
    if peak <= 0:
        return fallback, False
    core0, core1 = by0 - ya + int(0.1 * h), by1 - ya - int(0.1 * h)
    hit = [(a, b) for a, b in runs(rows > 0.2 * peak, 2) if a < core1 and b > core0]
    if not hit:
        return fallback, False
    top = max(hit[0][0] + ya, int(y - 0.1 * h))
    bot = min(hit[-1][1] + ya, int(math.ceil(y + h + 0.1 * h)))
    if not 0.4 * h <= bot - top <= 1.05 * h:
        return fallback, False
    cols = e[top - ya:bot - ya].sum(axis=0)
    on = np.flatnonzero(cols > 0.05 * cols.max()) if cols.max() > 0 else []
    if not len(on):
        return fallback, False
    return (bx0 + int(on[0]), top, bx0 + int(on[-1]) + 1, bot), True


def contrast(g):
    """(contrast, polarity) of a grey crop. Ink covers a minority of the pixels (a
    few percent for dots and thin strokes), so the median is background and the 1st
    or 99th percentile is ink."""
    p1, p50, p99 = np.percentile(g, (1, 50, 99))
    dark, light = p50 - p1, p99 - p50
    return float(max(dark, light)), ("dark-on-light" if dark >= light else "light-on-dark")


def grey_ratio(rgb_band, grey_band):
    """Edge energy of the grey ink band over the strongest colour channel's. About 1
    for achromatic text; near 0 when text and background differ in hue but not in
    luma, i.e. text that disappears once the crop is converted to grey."""
    e_grey = energy(grey_band).sum()
    e_col = np.max(np.stack([energy(rgb_band[..., c]) for c in range(3)]), axis=0).sum()
    return float(e_grey / e_col) if e_col > 0 else 0.0


def ctc_ok(label, width):
    """train.py's ctc_ok: CTC needs a frame per label and per repeated pair."""
    repeats = sum(1 for a, b in zip(label, label[1:]) if a == b)
    return len(label) + repeats <= math.ceil(width / STRIDE)


def to_height(g, rng):
    """Grey crop to height 32: bilinear (train.py's resize32) for crops up to AREA_ABOVE
    px, otherwise area-downscale to a native-looking height first."""
    h, w = g.shape
    if h > AREA_ABOVE:
        mid = int(rng.integers(MID_H[0], MID_H[1] + 1))
        g = cv2.resize(g, (max(1, int(round(w * mid / h))), mid), interpolation=cv2.INTER_AREA)
        h, w = g.shape
    nw = max(8, int(round(w * H / h)))
    return cv2.resize(g, (nw, H), interpolation=cv2.INTER_LINEAR)


def crop(rgb, gray, line, label, rng):
    """(grey 32xW uint8 crop, meta) for one usable line, or (None, reason)."""
    (x0, y0, x1, y1), clean = ink_box(gray, line["bbox"])
    ink_h = y1 - y0
    pad = int(round(ink_h * rng.uniform(*PAD_FRAC)))
    ph, pw = gray.shape
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(pw, x1 + pad), min(ph, y1 + pad)
    g = gray[cy0:cy1, cx0:cx1]
    if g.shape[0] < 4 or g.shape[1] < 4 or x1 <= x0 or y1 <= y0:
        return None, "faint"
    c, polarity = contrast(g)
    ratio = grey_ratio(rgb[y0:y1, x0:x1], gray[y0:y1, x0:x1])
    if c < CONTRAST or ratio < GREY_RATIO:
        return None, "faint"
    g32 = to_height(g, rng)
    if g32.shape[1] > MAX_W:
        return None, "wide"
    if not ctc_ok(label, g32.shape[1]):
        return None, "ctc"
    meta = {"crop": [cx0, cy0, cx1 - cx0, cy1 - cy0], "ink_h": ink_h, "clean": clean,
            "contrast": round(c, 1), "grey_ratio": round(ratio, 3), "polarity": polarity}
    return g32, meta


def page_rng(seed, tag, sample_id):
    """Per-page generator: the same page always gets the same lines and margins,
    however the pages are split across worker processes."""
    return np.random.default_rng([seed, zlib.crc32(tag.encode()), int(sample_id)])


def process_page(jpeg, ann, dims, sample_id, tag, per_page, seed, vocab, stats, oov):
    """Picks up to per_page usable lines of one page at random and crops them.
    dims is the page's [H, W] (the HDF5 'dimensions' row). Updates stats (reason
    counts) and oov (out-of-vocab character counts) in place.
    Returns [(grey crop, label, meta)]."""
    ann = json.loads(ann) if isinstance(ann, (str, bytes)) else ann
    rng = page_rng(seed, tag, sample_id)
    stats["pages"] += 1
    usable = []
    lines = ann.get("line_bboxes") or []
    page_h, page_w = int(dims[0]), int(dims[1])
    for li, line in enumerate(lines):
        stats["lines"] += 1
        reason, label = check(line, page_w, page_h, vocab)
        if label is not None:                              # horizontal: OOV statistics
            stats["h_lines"] += 1
            stats["h_chars"] += len(label)
            missing = [c for c in label if c not in vocab]
            if missing:
                stats["h_oov_lines"] += 1
                stats["h_oov_chars"] += len(missing)
                oov.update(missing)
        stats[reason or "usable"] += 1
        if reason is None:
            usable.append((li, line, label))
    if not usable or per_page <= 0:
        return []
    rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    gray = to_gray(rgb)
    if gray.shape != (page_h, page_w):
        stats["bad_dims"] += 1
        return []
    out = []
    for j in rng.permutation(len(usable)):
        if len(out) >= per_page:
            break
        li, line, label = usable[j]
        g, meta = crop(rgb, gray, line, label, rng)
        if g is None:
            stats[meta] += 1
            continue
        stats["kept"] += 1
        meta.update({"src": tag, "sample": int(sample_id), "line": li})
        out.append((g, label, meta))
    if out:
        stats["pages_kept"] += 1
    return out


def new_stats():
    return Counter({k: 0 for k in ("pages", "pages_kept", "lines", "usable", "kept", "h_lines", "h_chars",
                                   "h_oov_lines", "h_oov_chars", "bad_dims", *REASONS, *CROP_REASONS)})


def save_shard(path, crops, labels, metas):
    """train.py / render.py shard format: one flat uint8 array plus offsets and shapes."""
    shapes = np.array([c.shape for c in crops], np.int32).reshape(-1, 2)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    data = np.concatenate([c.ravel() for c in crops]) if crops else np.zeros(0, np.uint8)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=data, offsets=offsets, shapes=shapes,
                        labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(metas, ensure_ascii=False, separators=(",", ":"))))
    tmp.rename(path)


def vocab_set(path=None):
    return set(load_vocab(path) if path else load_vocab())

"""Shared pieces for the Vita VN OCR toolchain.

find_lines(img, finder) is a bit-exact twin of runtime/vocr_lines.c, so the training
crops are the crops the console hands its recognizer:
  finder="grad"    the polarity-agnostic dialogue line finder (the console's)
  finder="legacy"  an exact port of vita-jp-overlay's vjo_ocr_find_lines
                   (core/ocr_lines.c), the bright-pixel projection used for v1's data
The finder has no default in find_lines(), which used to mean the legacy finder: a
caller written before grad existed fails loudly instead of silently switching finder.
find_lines_ex() and line_crops() default to grad. line_crops() turns a region into
the grey line crops the recognizer reads, inverting dark-text lines by default so the
recognizer always sees light text on a dark ground. docs/linefind.md describes the
method and its evaluation.
"""
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # vita-ocr/
DATA = ROOT / "data"
MAX_W, MAX_H, MAX_LINES, MAX_LINE_H = 960, 544, 8, 96   # legacy finder limits

# Grad finder constants: keep in step with runtime/vocr_lines.c.
LINES_MAX = 16      # lines returned per region
T_MIN = 40          # lowest contrast threshold
GAP = 10            # on-pixels at most this far apart join into one run
LMIN = 16           # shortest run that counts as text
ROW_DIV = 16        # active rows: kept length >= max(LMIN, peak // ROW_DIV)
MIN_H = 8           # shortest band
MERGE = 2           # inactive rows bridged inside a band
GAPC = 2            # column clusters split at gaps over GAPC * band height
PAD_X = 7          # horizontal crop padding: previous 3 px plus 4 px, clamped to ROI
PAD_Y = 3          # vertical crop padding: 4 px around ink, less the 1 px dilation
FURI_PCT = 55       # lines under this % of the median height are dropped


def grey_of(img):
    """uint8 grey of an HxW grey or HxWx3/4 RGB(A) image, as the C side computes it."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.uint8, copy=False)
    return to_gray(img)


def find_lines(img, finder):
    """img: HxW grey or HxWx3/4 RGB(A) uint8 region; finder: "grad" or "legacy" (no
    default, see the module docstring). Returns [(x, y, w, h), ...] top to bottom, or
    None where the legacy finder rejects the region."""
    r = find_lines_ex(img, finder)
    return None if r is None else r[0]


def find_lines_ex(img, finder="grad"):
    """Like find_lines but returns (boxes, dark, truncated): dark[i] is 1 when line i's
    text is darker than its background; truncated when more than LINES_MAX lines were
    found (only the first LINES_MAX are returned). None where legacy rejects."""
    if finder == "grad":
        return _find_grad(np.asarray(img))
    if finder == "legacy":
        boxes = _find_legacy(np.asarray(img))
        return None if boxes is None else (boxes, [0] * len(boxes), False)
    raise ValueError(f"unknown finder {finder!r}")


def line_crops(img, finder="grad", invert_dark=True):
    """[(grey crop, (x, y, w, h), dark)] for each line, or None where legacy rejects.
    With invert_dark, dark-text crops are inverted (255 - g) so that every crop shows
    light text, which is what the recognizer is trained on."""
    r = find_lines_ex(img, finder)
    if r is None:
        return None
    g = grey_of(img)
    out = []
    for (x, y, w, h), dark in zip(r[0], r[1]):
        crop = g[y:y + h, x:x + w]
        out.append((255 - crop if (dark and invert_dark) else crop.copy(), (x, y, w, h), dark))
    return out


def _find_legacy(rgb):
    """Exact port of vjo_ocr_find_lines. Returns None where the C code returns
    VJO_E_OCR_REGION (oversized region, light background, a line taller than 88 px,
    or more than 8 lines)."""
    h_img, w_img = rgb.shape[:2]
    if not w_img or not h_img or w_img > MAX_W or h_img > MAX_H:
        return None
    if rgb.ndim == 2:
        bright = rgb >= 180
    else:
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
                if h > MAX_LINE_H - 8 or len(lines) == MAX_LINES:
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


def contrast3(g):
    """max - min of each 3x3 window, edges replicated."""
    p = np.pad(g, 1, mode="edge")
    mx = np.maximum(np.maximum(p[:, :-2], p[:, 1:-1]), p[:, 2:])
    mn = np.minimum(np.minimum(p[:, :-2], p[:, 1:-1]), p[:, 2:])
    mx = np.maximum(np.maximum(mx[:-2], mx[1:-1]), mx[2:])
    mn = np.minimum(np.minimum(mn[:-2], mn[1:-1]), mn[2:])
    return mx - mn


def kept_runs(on):
    """Per row, join on-pixels at most GAP apart into runs; True on runs >= LMIN."""
    H, W = on.shape
    idx = np.arange(W, dtype=np.int32)
    prev = np.maximum.accumulate(np.where(on, idx, -(1 << 20)), axis=1)
    nxt = np.minimum.accumulate(np.where(on, idx, 1 << 20)[:, ::-1], axis=1)[:, ::-1]
    filled = on | (nxt - prev <= GAP)
    z = np.zeros((H, 1), bool)
    st = filled & ~np.concatenate([z, filled[:, :-1]], axis=1)
    en = filled & ~np.concatenate([filled[:, 1:], z], axis=1)
    s_idx = np.maximum.accumulate(np.where(st, idx, -1), axis=1)
    e_idx = np.minimum.accumulate(np.where(en, idx, W)[:, ::-1], axis=1)[:, ::-1]
    return filled & (e_idx - s_idx + 1 >= LMIN)


def _first_cum(cum, k):
    """Smallest value v with cum[v] >= k (the C loops' percentile rule)."""
    return int(np.argmax(cum >= k))


def _widen(oc, x1, x2, d):
    """Grow [x1, x2) over on-columns at most d apart (C: the widening loops)."""
    w = len(oc)
    r = np.flatnonzero(oc[x2:]) + x2
    chain = np.concatenate([[x2 - 1], r])
    stop = np.flatnonzero(np.diff(chain) > d)
    x2 = int(chain[stop[0] if len(stop) else len(r)]) + 1
    lft = np.flatnonzero(oc[:x1])[::-1]
    chain = np.concatenate([[x1], lft])
    stop = np.flatnonzero(-np.diff(chain) > d)
    x1 = int(chain[stop[0] if len(stop) else len(lft)])
    return max(0, x1), min(w, x2)


def _find_grad(img):
    g = grey_of(img)
    H, W = g.shape
    if not H or not W:
        return [], [], False
    c = contrast3(g)
    cum = np.cumsum(np.bincount(c.ravel(), minlength=256)).astype(np.int64)
    n = H * W
    p99 = _first_cum(cum * 100, n * 99)
    T = max(T_MIN, 2 * p99 // 5)
    on = c >= T
    kept = kept_runs(on)
    prof = kept.sum(axis=1)
    th = max(LMIN, int(prof.max()) // ROW_DIV)
    act = prof >= th
    # bands: gaps of up to MERGE rows bridged, except a gap holding an empty row
    # between runs of >= MIN_H active rows (C: the band loop)
    bands, y = [], 0
    while y < H:
        if not act[y]:
            y += 1
            continue
        s = rs = last = y
        y += 1
        while y < H and (act[y] or y - last <= MERGE):
            if act[y]:
                if y - last > 1:
                    k = y
                    while k < H and act[k]:
                        k += 1
                    if not prof[last + 1:y].all() and last + 1 - rs >= MIN_H and k - y >= MIN_H:
                        bands.append((s, last + 1))
                        s = y
                    rs = y
                last = y
            y += 1
        if last + 1 - s >= MIN_H:
            bands.append((s, last + 1))
    cands = []
    for i, (a, b) in enumerate(bands):
        bh = b - a
        kc = kept[a:b].sum(axis=0)
        cols = np.flatnonzero(kc)
        if not len(cols):
            continue
        br = np.flatnonzero(np.diff(cols) > GAPC * bh)
        starts = np.concatenate([[0], br + 1])
        ends = np.concatenate([br, [len(cols) - 1]])
        mass = [int(kc[cols[s]:cols[e] + 1].sum()) for s, e in zip(starts, ends)]
        k = int(np.argmax(mass))
        x1, x2 = int(cols[starts[k]]), int(cols[ends[k]]) + 1
        oc = on[a:b].sum(axis=0)
        x1, x2 = _widen(oc, x1, x2, max(4, bh))
        up = min(bh // 2, (a - bands[i - 1][1]) // 2) if i else min(bh // 2, a)
        dn = min(bh // 2, (bands[i + 1][0] - b) // 2) if i + 1 < len(bands) else min(bh // 2, H - b)
        rows_on = on[:, x1:x2].sum(axis=1)
        top, bot = a, b
        while a - top < up and rows_on[top - 1] >= 2:
            top -= 1
        while bot - b < dn and rows_on[bot] >= 2:
            bot += 1
        cands.append((x1, top, x2, bot))
    boxes, dark, truncated = [], [], False
    if cands:
        med = sorted(bot - top for _, top, _, bot in cands)[len(cands) // 2]
        for x1, top, x2, bot in cands:
            if 100 * (bot - top) < FURI_PCT * med:
                continue
            if len(boxes) == LINES_MAX:
                truncated = True
                break
            lx, ly = max(0, x1 - PAD_X), max(0, top - PAD_Y)
            ex, ey = min(W, x2 + PAD_X), min(H, bot + PAD_Y)
            boxes.append((lx, ly, ex - lx, ey - ly))
            dark.append(line_dark(g, (lx, ly, ex - lx, ey - ly)))
    return boxes, dark, truncated


def line_dark(img, box):
    """1 if the text in box (x, y, w, h) of img is darker than its background (C:
    vocr_line_dark). Background = median of the box's border pixels; text pixels
    deviate from it by more than max(16, a quarter of the box's 2..98% grey range).
    Dark only when bright deviations are nearly absent: outlined light text on a
    light box has a dark outline but a bright fill, and stays light."""
    x, y, w, h = box
    g = grey_of(img)[y:y + h, x:x + w].astype(np.int32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = int(np.sort(border)[len(border) // 2])
    cum = np.cumsum(np.bincount(g.ravel(), minlength=256)).astype(np.int64)
    n = w * h
    lo = _first_cum(cum, n * 2 // 100)
    hi = _first_cum(cum, n * 98 // 100)
    d = max(16, (hi - lo) // 4)
    nd = int((g < bg - d).sum())
    nb = int((g > bg + d).sum())
    return 1 if nd > 8 * nb else 0


def line_polarity(grey):
    """Polarity of a whole grey line crop (line_dark over the full crop)."""
    return line_dark(grey, (0, 0, grey.shape[1], grey.shape[0]))


def to_gray(rgb):
    """Integer luma, identical to the C side: (77 R + 150 G + 29 B) >> 8."""
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def load_vocab(path=DATA / "vocab.txt"):
    """Class i + 1 is vocab[i]; class 0 is the CTC blank."""
    return [c for c in Path(path).read_text(encoding="utf-8").split("\n") if c]


def is_test_vn(vn_id):
    """Deterministic VN-level split: about 15% of VNs are held out for evaluation,
    so no screenshot of a test VN is ever used for training or as a background."""
    return int(hashlib.md5(vn_id.encode()).hexdigest(), 16) % 100 < 15

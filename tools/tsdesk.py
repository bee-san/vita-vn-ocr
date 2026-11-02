#!/usr/bin/env python3
"""textspot-desktop: VOCR's fast desktop line detector with tight boxes (issue #173, docs/textspot-desktop.md), and
its page pipeline with a VOCR6 recognizer, on ONNX Runtime's CPU provider. The C path (desktop/, AVX2 + FMA with a
scalar fallback) runs the same network and gives the same boxes from the same maps.

    from tsdesk import Detector, read_frame
    from vocr_onnx import Recognizer
    det = Detector("textspot_desktop.onnx")                 # any frame size; `size` picks the input policy
    rec = Recognizer("VOCR6.2_L_fast.onnx")
    lines = read_frame(frame, det, rec, workers=4)          # frame: H x W grey or H x W x 3/4 RGB(A) uint8
    for ln in lines:
        ln.text, ln.box, ln.crop, ln.tag                    # box: the line's tight ink box (x, y, w, h), frame px

    python3 tsdesk.py --det textspot_desktop.onnx --rec VOCR6.2_L_fast.onnx [--size fit:960] IMAGE...

The detector (train/tsdesk_net.py) is a small fully convolutional network trained at many text sizes, so it reads a
frame at any size. `size` scales the frame first, keeping its aspect ratio: "fit:N" brings the longer side down to N px
(never up), "long:N" scales it to N px up or down, "native" keeps it. The network reads the grey frame (integer luma,
(77 R + 150 G + 29 B) >> 8, as VOCR's recognizer) padded right and bottom to a multiple of 32 by repeating its last
column and row. It returns maps at the head's stride s (8): K (a line's kernel, a logit), O (orientation: vertical when
> 0), A1 and A2 (the distances from a cell's centre to the line's two edges across it), E1 and E2 (the distances from
a cell's centre to the line's start and end along it), all in input px, and 4 tag logits.

The decode gives each line's ink box with no margin: the 4-connected components of K > 0 with at least `min_cells`
cells and a mean K logit of at least `conf_min_logit`; across, the mean over the component's cells of centre - A1 and
centre + A2; along, the mean of centre - E1 over the cells of its first column (row, for a vertical line) and of
centre + E2 over its last, each kept within the component's cells' reach (`along_reach` x its size across beyond
them, and never inside the first or last cell's far edge). The recognizer reads a box grown by `pad` (default
READ_PAD = "k6v4", the protocol's: max(2, round(6 H / 544)) frame px left and right and max(2, round(4 H / 544)) top
and bottom for an H px tall frame; "auto" is k4, the pad VOCR's eval crops and MeikiOCR's reference pipeline use).
"""
import json
import math
from dataclasses import dataclass

import numpy as np

ALIGN = 32
CLASSES = ("dialogue", "name", "choice", "ui", "scene")
MAX_BOXES = 256
DEFAULTS = {"min_cells": 2, "conf_min_logit": 0.0, "min_across": 4.0, "along_reach": 3.0, "max_boxes": MAX_BOXES}


def _session(path, threads=1, providers=None, options=None):
    import onnxruntime as ort
    so = options or ort.SessionOptions()
    if options is None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])


def grey_of(img):
    """uint8 grey of an H x W grey or H x W x 3/4 RGB(A) image: (77 R + 150 G + 29 B) >> 8 (runtime/vocr.c's)."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise TypeError(f"images are uint8, not {a.dtype}")
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] in (3, 4):
        r, g, b = (a[..., i].astype(np.uint16) for i in range(3))
        return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
    raise ValueError("an H x W grey or H x W x 3 / 4 RGB(A) image")


def scale_for(policy, w, h):
    """The factor a w x h frame is resized by before the network, for a size policy (module docstring)."""
    if policy in (None, "", "native"):
        return 1.0
    kind, _, n = str(policy).partition(":")
    if not n:
        kind, n = "fit", kind
    n = float(n)
    long = max(w, h)
    if kind == "fit":
        return min(1.0, n / long)
    if kind == "long":
        return n / long
    raise ValueError(f"size policy {policy!r}: fit:N, long:N or native")


def input_size(policy, w, h):
    """(iw, ih) of the network input's image part for a w x h frame (each side round(side x f), at least 1)."""
    f = scale_for(policy, w, h)
    if f == 1.0:
        return w, h
    return max(1, int(round(w * f))), max(1, int(round(h * f)))


Q14 = 16384


def _axis(n, m):
    """(idx [m, K], weights [m, K]) of one axis of desktop/vocr_desk.c's vocr_desk_resize: Q14 area weights when the
    axis shrinks (m < n), bilinear on pixel centres when it grows; the weights of an output sum to 16384."""
    if m == n:
        return np.arange(m, dtype=np.int64)[:, None], np.full((m, 1), Q14, np.int64)
    K = n // m + 2 if m < n else 2
    idx = np.zeros((m, K), np.int64)
    wt = np.zeros((m, K), np.int64)
    for i in range(m):
        if m < n:
            lo, hi = i * n, i * n + n
            j, k, cum, prev = lo // m, 0, 0, 0
            while j * m < hi and k < K:
                a, b = max(lo, j * m), min(hi, (j + 1) * m)
                cum += b - a
                q = (cum * 2 * Q14 + n) // (2 * n)
                idx[i, k], wt[i, k] = j, q - prev
                prev = q
                j += 1
                k += 1
            for kk in range(k, K):
                idx[i, kk] = idx[i, kk - 1]
        else:
            num = (2 * i + 1) * n - m
            if num <= 0:
                idx[i, :2], wt[i, 0] = 0, Q14
            else:
                x0, r = num // (2 * m), num - (num // (2 * m)) * 2 * m
                if x0 >= n - 1:
                    idx[i, :2], wt[i, 0] = n - 1, Q14
                else:
                    w1 = (r * Q14 + m) // (2 * m)
                    idx[i, 0], idx[i, 1], wt[i, 0], wt[i, 1] = x0, x0 + 1, Q14 - w1, w1
    return idx, wt


def resize(g, dw, dh):
    """desktop/vocr_desk.c's vocr_desk_resize on a uint8 grey image, bit for bit: separable integer weights (the rows
    first, then the columns), (acc + 2^27) >> 28, clamped. The detector's C path resizes this way, so the ONNX path
    and the C path see the same network input."""
    g = np.asarray(g, np.uint8)
    h, w = g.shape
    if (w, h) == (dw, dh):
        return g.copy()
    ix, wx = _axis_cached(w, dw)
    iy, wy = _axis_cached(h, dh)
    tmp = np.zeros((h, dw), np.int32)
    for k in range(ix.shape[1]):
        tmp += g[:, ix[:, k]].astype(np.int32) * wx[:, k].astype(np.int32)[None, :]
    acc = np.zeros((dh, dw), np.int64)
    for k in range(iy.shape[1]):
        acc += tmp[iy[:, k]].astype(np.int64) * wy[:, k][:, None]
    return np.clip((acc + (1 << 27)) >> 28, 0, 255).astype(np.uint8)


_AXES = {}


def _axis_cached(n, m):
    key = (int(n), int(m))
    if key not in _AXES:
        if len(_AXES) > 64:
            _AXES.clear()
        _AXES[key] = _axis(n, m)
    return _AXES[key]


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    vertical: int
    logit: float            # the mean K logit over the box's kernel cells (the decode's confidence)
    tag: int
    tconf: float            # (best tag score - second) / cells
    cells: int = 0


@dataclass
class Line:
    text: str
    box: tuple              # the detector's tight box (x, y, w, h), frame px
    crop: tuple             # the crop read: the box grown by the pad, integer frame px
    tag: int
    logit: float
    dark: int
    vertical: int = 0


def decode(maps, w, h, stride, min_cells=2, conf_min_logit=0.0, min_across=4.0, along_reach=3.0, tags=4,
           max_boxes=MAX_BOXES):
    """[Box] in network-input px (the image's w x h part) from float maps [6 + T, H / s, W / s] (module docstring).
    The C decode (desktop/vocr_tsd.c) is the same rule with the same sums in the same order (in double, over the
    cells in raster order), so given the same maps the two give the same boxes bit for bit. Components are labelled
    in the order of each one's first cell; boxes are sorted by mean logit (descending), then y, then x, stable, and
    the first max_boxes kept."""
    import cv2
    s = int(stride)
    hm, wm = -(-h // s), -(-w // s)
    k = np.ascontiguousarray(maps[0, :hm, :wm])
    n, lab, stats, _ = cv2.connectedComponentsWithStats((k > 0).astype(np.uint8), connectivity=4)
    if n <= 1:
        return []
    flat = lab.ravel()
    on = np.flatnonzero(flat)
    L = flat[on]
    rows, cols = np.divmod(on, wm)

    def m_of(c):
        return np.ascontiguousarray(maps[c, :hm, :wm]).ravel()[on].astype(np.float64)

    cnt = np.bincount(L, minlength=n).astype(np.float64)
    cnt[0] = 1.0
    logit = np.bincount(L, m_of(0), minlength=n) / cnt
    vert = np.bincount(L, m_of(1), minlength=n) / cnt > 0
    vL = vert[L]
    half = (s - 1) / 2.0
    cc = s * np.where(vL, cols, rows).astype(np.float64) + half
    near = np.bincount(L, cc - m_of(2), minlength=n) / cnt
    far = np.bincount(L, cc + m_of(3), minlength=n) / cnt
    ai = np.where(vL, rows, cols)
    first = np.where(vert, stats[:, cv2.CC_STAT_TOP], stats[:, cv2.CC_STAT_LEFT])
    last = first + np.where(vert, stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_WIDTH]) - 1
    ca = s * ai.astype(np.float64) + half
    s1 = ai == first[L]
    s2 = ai == last[L]
    n1 = np.bincount(L[s1], minlength=n).astype(np.float64)
    n2 = np.bincount(L[s2], minlength=n).astype(np.float64)
    n1[0] = n2[0] = 1.0
    start = np.bincount(L[s1], (ca - m_of(4))[s1], minlength=n) / n1
    end = np.bincount(L[s2], (ca + m_of(5))[s2], minlength=n) / n2
    tsum = [np.bincount(L, m_of(6 + i), minlength=n) for i in range(tags)]
    out = []
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_cells or logit[c] < conf_min_logit:
            continue
        size = float(far[c] - near[c])
        if size < min_across:
            continue
        lo, hi = float(s * int(first[c])), float(s * (int(last[c]) + 1))
        b0, b1 = float(start[c]), float(end[c])
        if b0 > lo + s:
            b0 = lo + s
        if b0 < lo - along_reach * size:
            b0 = lo - along_reach * size
        if b1 < hi - s:
            b1 = hi - s
        if b1 > hi + along_reach * size:
            b1 = hi + along_reach * size
        a0, a1 = float(near[c]), float(far[c])
        x0, x1, y0, y1 = (a0, a1, b0, b1) if vert[c] else (b0, b1, a0, a1)
        x0, y0 = (0.0 if x0 < 0.0 else x0), (0.0 if y0 < 0.0 else y0)
        x1, y1 = (float(w) if x1 > w else x1), (float(h) if y1 > h else y1)
        if x1 <= x0 or y1 <= y0:
            continue
        sc = [0.0] + [float(t[c]) for t in tsum]
        tag = 0
        for i in range(1, len(sc)):
            if sc[i] > sc[tag]:
                tag = i
        second = max((v for i, v in enumerate(sc) if i != tag), default=0.0) if tags else 0.0
        out.append(Box(x0, y0, x1 - x0, y1 - y0, int(vert[c]), float(logit[c]), tag, (sc[tag] - second) / cnt[c],
                       int(cnt[c])))
    out.sort(key=lambda b: (-b.logit, b.y, b.x))
    return out[:max_boxes]


# ----------------------------------------------------------------------------- the post-pass (#173 round 2)
#
# Options on the decode's frame-px boxes (helper textspot-desktop-boxes). Every option is an integer and 0 turns it
# off, so with POST_DEFAULTS the boxes are the decode's and the refinement's bit for bit. The pre-refinement options
# run in this order on the decode's boxes, before tools/tsdesk_refine.py:
#   merge_pct  two boxes of one orientation that overlap across by at least half the smaller one's size across and
#              whose intersection is at least merge_pct % of the smaller one's area become their union (the larger
#              box's fields): a fragment the decode found inside or beside its line (a 「, a ―。 at the end);
#   rows_pct   a horizontal box split across at an ink valley (rows with at most valley_pct % of the median ink row's
#              count) between two ink bands whose heights are within rows_pct % of each other, each at least
#              max(4, 20 % of the box's height) rows tall and twice its height long in ink: two tightly spaced rows
#              boxed as one (a band of ruby, about half a line's height, is not split off); at most two levels;
#   gap_pct    a horizontal box split along at every run of at least max(2, r(h, gap_pct)) empty ink columns between
#              its first and last ink column (h its height): neighbouring buttons, labels and numbers in one box.
# The ink is the refinement's test (tools/tsdesk_refine.py steps 3-6, without its margins' noise term): on the box
# grown by max(2, r(h, 25)) px, bg the lower median of that window's border ring, a pixel is ink when |g - bg| and
# its 3 x 3 contrast both exceed max(28, 40 % of their 99th percentile over the box) and a 4-neighbour is ink too.

POST_DEFAULTS = {"merge_pct": 0, "rows_pct": 0, "valley_pct": 10, "gap_pct": 0}
POST_PRE = ("merge_pct", "rows_pct", "gap_pct")      # the options that run before the refinement


def post_options(opts=None):
    o = dict(POST_DEFAULTS)
    for k, v in (opts or {}).items():
        if k not in POST_DEFAULTS:
            raise KeyError(f"unknown post-pass option {k!r}")
        o[k] = int(v)
    return o


def _r100(v, pct):
    return (int(v) * int(pct) + 50) // 100


def _pixel_box(b, W, H):
    """(ix0, iy0, ix1, iy1) of a Box: floor / ceil of its edges, clipped to the frame."""
    ix0 = min(max(int(math.floor(b.x)), 0), W)
    iy0 = min(max(int(math.floor(b.y)), 0), H)
    ix1 = min(max(int(math.ceil(b.x + b.w)), 0), W)
    iy1 = min(max(int(math.ceil(b.y + b.h)), 0), H)
    return ix0, iy0, ix1, iy1


def ink_mask(g, ix0, iy0, ix1, iy1, floor=28, k_pct=40):
    """The ink of the pixel box [ix0, ix1) x [iy0, iy1) of grey g (the post-pass's test, above): a bool array of the
    box's shape."""
    import tsdesk_refine as TR
    H, W = g.shape
    h, w = iy1 - iy0, ix1 - ix0
    A = max(2, _r100(h, 25))
    wx0, wy0, wx1, wy1 = max(0, ix0 - A), max(0, iy0 - A), min(W, ix1 + A), min(H, iy1 + A)
    win = g[wy0:wy1, wx0:wx1].astype(np.int16)
    ring = np.concatenate([win[0], win[-1], win[1:-1, 0], win[1:-1, -1]])
    bg = int(np.partition(ring, (ring.size - 1) // 2)[(ring.size - 1) // 2])
    d = np.abs(win - bg)
    c3 = TR._c3(g, wx0, wy0, wx1, wy1)
    by0, bx0 = iy0 - wy0, ix0 - wx0
    t = max(floor, _r100(TR.quantile(d[by0:by0 + h, bx0:bx0 + w], 99), k_pct))
    tc = max(floor, _r100(TR.quantile(c3[by0:by0 + h, bx0:bx0 + w], 99), k_pct))
    m0 = (d > t) & (c3 > tc)
    nb = np.zeros_like(m0)
    nb[1:] |= m0[:-1]
    nb[:-1] |= m0[1:]
    nb[:, 1:] |= m0[:, :-1]
    nb[:, :-1] |= m0[:, 1:]
    return (m0 & nb)[by0:by0 + h, bx0:bx0 + w]


def merge_overlaps(boxes, pct):
    """merge_pct: the pair with the largest intersection (as a share of the smaller box's area) merged first, until no
    pair qualifies. The order of the boxes is kept (a merged box takes the place of the earlier of the two)."""
    import dataclasses
    out = list(boxes)
    while True:
        best = None
        for i in range(len(out)):
            a = out[i]
            for j in range(i + 1, len(out)):
                b = out[j]
                if a.vertical != b.vertical:
                    continue
                dx = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
                dy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                if dx <= 0 or dy <= 0:
                    continue
                across, size = (dx, min(a.w, b.w)) if a.vertical else (dy, min(a.h, b.h))
                if across < 0.5 * size:
                    continue
                small = min(a.w * a.h, b.w * b.h)
                if small <= 0 or dx * dy * 100 < pct * small:
                    continue
                share = dx * dy / small
                if best is None or share > best[0]:
                    best = (share, i, j)
        if best is None:
            return out
        _, i, j = best
        a, b = out[i], out[j]
        big = a if a.w * a.h >= b.w * b.h else b
        x0, y0 = min(a.x, b.x), min(a.y, b.y)
        x1, y1 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
        out[i] = dataclasses.replace(big, x=x0, y=y0, w=x1 - x0, h=y1 - y0, cells=a.cells + b.cells)
        del out[j]


def split_rows(g, b, rows_pct, valley_pct=10, depth=0):
    """rows_pct: [Box] of a horizontal box split across at an ink valley (above), recursively to two levels."""
    import dataclasses
    H, W = g.shape
    ix0, iy0, ix1, iy1 = _pixel_box(b, W, H)
    h, w = iy1 - iy0, ix1 - ix0
    if b.vertical or h < 8 or w < 8 or depth >= 2:
        return [b]
    m = ink_mask(g, ix0, iy0, ix1, iy1)
    rc = m.sum(1)
    ink = np.flatnonzero(rc)
    if len(ink) < 2:
        return [b]
    vals = rc[ink]
    med = int(np.partition(vals, (vals.size - 1) // 2)[(vals.size - 1) // 2])
    low = _r100(med, valley_pct)
    dense = np.flatnonzero(rc > low)
    if len(dense) < 2:
        return [b]
    lo, hi = int(dense[0]), int(dense[-1]) + 1
    min_band = max(4, _r100(h, 20))
    best = None
    y = lo
    while y < hi:
        if rc[y] > low:
            y += 1
            continue
        v0 = y
        while y < hi and rc[y] <= low:
            y += 1
        v1 = y                                   # the valley's rows [v0, v1), dense rows on both sides
        ha, hb = v0 - lo, hi - v1
        if min(ha, hb) < min_band or min(ha, hb) * 100 < rows_pct * max(ha, hb):
            continue
        ca = np.flatnonzero(m[lo:v0].any(0))
        cb = np.flatnonzero(m[v1:hi].any(0))
        if not len(ca) or not len(cb) or ca[-1] - ca[0] + 1 < 2 * ha or cb[-1] - cb[0] + 1 < 2 * hb:
            continue
        key = (min(ha, hb) / max(ha, hb), v1 - v0)
        if best is None or key > best[0]:
            best = (key, v0, v1)
    if best is None:
        return [b]
    _, v0, v1 = best
    top = dataclasses.replace(b, h=float(iy0 + v0) - b.y)
    bot = dataclasses.replace(b, y=float(iy0 + v1), h=b.y + b.h - float(iy0 + v1))
    return split_rows(g, top, rows_pct, valley_pct, depth + 1) + split_rows(g, bot, rows_pct, valley_pct, depth + 1)


def split_along(g, b, gap_pct):
    """gap_pct: [Box] of a horizontal box split along at its long runs of empty ink columns (above). The first part
    keeps the box's start and the last its end; the edges at a split lie on the ink either side of the gap."""
    import dataclasses
    H, W = g.shape
    ix0, iy0, ix1, iy1 = _pixel_box(b, W, H)
    h, w = iy1 - iy0, ix1 - ix0
    if b.vertical or h < 4 or w < 4:
        return [b]
    cc = ink_mask(g, ix0, iy0, ix1, iy1).sum(0)
    ink = np.flatnonzero(cc)
    if len(ink) < 2:
        return [b]
    G = max(2, _r100(h, gap_pct))
    cut = np.flatnonzero(np.diff(ink) - 1 >= G)
    if not len(cut):
        return [b]
    starts = [b.x] + [float(ix0 + int(ink[c + 1])) for c in cut]
    ends = [float(ix0 + int(ink[c]) + 1) for c in cut] + [b.x + b.w]
    return [dataclasses.replace(b, x=s, w=e - s) for s, e in zip(starts, ends) if e > s]


def post_pre(g, boxes, opts):
    """The pre-refinement options (above) on frame-px Box objects of grey frame g."""
    o = post_options(opts)
    out = list(boxes)
    if o["merge_pct"]:
        out = merge_overlaps(out, o["merge_pct"])
    if o["rows_pct"]:
        out = [p for b in out for p in split_rows(g, b, o["rows_pct"], o["valley_pct"])]
    if o["gap_pct"]:
        out = [p for b in out for p in split_along(g, b, o["gap_pct"])]
    return out


# ----------------------------------------------------------------------------- the recognizer as a junk filter

# a Japanese character: kana, kanji, CJK symbols and punctuation, full-width forms, …, ― and —
JAPANESE = "\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff01-\uff60\u2025\u2026\u2015\u2014"
DROP_DEFAULTS = {"nojp": 0, "conf_milli": 0}


def drop_options(opts=None):
    o = dict(DROP_DEFAULTS)
    for k, v in (opts or {}).items():
        if k not in DROP_DEFAULTS:
            raise KeyError(f"unknown reading-filter option {k!r}")
        o[k] = int(v)
    return o


_JP_RE = None


def drop_reading(line, o):
    """True when a horizontal box's reading (a vocr_onnx.Line) drops the box, for options o (drop_options(); every
    rule off by default): nojp, a reading with no Japanese character (JAPANESE; the empty reading too); conf_milli, a
    reading whose mean character confidence (runtime/vocr.c's conf_mean) is under conf_milli / 1000."""
    global _JP_RE
    if _JP_RE is None:
        import re
        _JP_RE = re.compile(f"[{JAPANESE}]")
    if o["nojp"] and not _JP_RE.search(line.text):
        return True
    if o["conf_milli"] and line.conf_mean * 1000.0 < o["conf_milli"]:
        return True
    return False


class Detector:
    """textspot-desktop's detector (an ONNX file train/tsdesk_det.py exports)."""

    def __init__(self, path, threads=1, providers=None, session_options=None, size=None, decode_opts=None,
                 resize_mode="exact", refine=None, post=None):
        if resize_mode not in ("exact", "cv2"):
            raise ValueError(f"resize_mode {resize_mode!r}: exact (the C path's integer resize) or cv2")
        self.resize_mode = resize_mode
        # refine: None (off), True (tools/tsdesk_refine.DEFAULTS) or a dict of its options: every horizontal box's
        # edges snapped to its line's ink on the full-resolution grey frame after the decode
        self.refine = None if refine in (None, False) else ({} if refine is True else dict(refine))
        # post: None (off) or a dict of POST_DEFAULTS options (the post-pass; every option 0 = off)
        self.post = post_options(post) if post else None
        self.sess = _session(path, threads, providers, session_options)
        meta = {k: json.loads(v) for k, v in self.sess.get_modelmeta().custom_metadata_map.items()
                if k.startswith("vocr.")}
        if meta.get("vocr.kind") != "textspot-desktop-detector":
            raise ValueError(f"{path}: not a textspot-desktop detector")
        self.meta = meta
        self.size = size or meta.get("vocr.size", "fit:960")
        self.stride = int(meta["vocr.stride"])
        self.tags = int(meta.get("vocr.tags", 4))
        d = dict(DEFAULTS)
        d.update({k: v for k, v in (meta.get("vocr.decode") or {}).items() if k in DEFAULTS})
        d.update(decode_opts or {})
        self.opts = d
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, g):
        """(network input [1, 1, H', W'] uint8, (fx, fy, iw, ih)) of a grey frame: resized by the size policy with
        the C path's integer resize (resize(): area weights down, bilinear up), padded right and bottom to a multiple
        of 32 by edge replication."""
        h, w = g.shape
        iw, ih = input_size(self.size, w, h)
        if (iw, ih) != (w, h):
            if self.resize_mode == "cv2":      # an app's resize: within a grey level of the integer one, faster here
                import cv2
                g = cv2.resize(g, (iw, ih), interpolation=cv2.INTER_AREA if iw < w else cv2.INTER_LINEAR)
            else:
                g = resize(g, iw, ih)
        ph, pw = -ih % ALIGN, -iw % ALIGN
        if ph or pw:
            g = np.pad(g, ((0, ph), (0, pw)), mode="edge")
        return np.ascontiguousarray(g, np.uint8)[None, None], (iw / w, ih / h, iw, ih)

    def maps(self, x):
        return self.sess.run(None, {self.input_name: x})[0][0]

    def detect(self, img):
        """[Box] in the frame's px (tight ink boxes), in the decode's order (by confidence)."""
        g = grey_of(img)
        x, (fx, fy, iw, ih) = self.preprocess(g)
        o = self.opts
        boxes = decode(self.maps(x), iw, ih, self.stride, o["min_cells"], o["conf_min_logit"], o["min_across"],
                       o["along_reach"], self.tags, o["max_boxes"])
        H, W = g.shape
        out = []
        for b in boxes:
            x0, y0 = max(0.0, b.x / fx), max(0.0, b.y / fy)
            x1, y1 = min(float(W), (b.x + b.w) / fx), min(float(H), (b.y + b.h) / fy)
            if x1 > x0 and y1 > y0:
                out.append(Box(x0, y0, x1 - x0, y1 - y0, b.vertical, b.logit, b.tag, b.tconf, b.cells))
        if self.post is not None and any(self.post[k] for k in POST_PRE):
            out = post_pre(g, out, self.post)
        if self.refine is not None:
            import tsdesk_refine
            out = tsdesk_refine.refine_det(g, out, self.refine)
        return out

    def boxes(self, rgb):
        """eval/desktop.py's detector interface: [[x0, y0, x1, y1, vertical, logit, tag]] in frame px."""
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag] for b in self.detect(rgb)]


READ_PAD = "k6v4"          # the reading pad the protocol chose on exact boxes (#173 amendment 1)


def pad_of(pad, H):
    """The reading pad in frame px for an H px tall frame: "kN" = max(2, round(N H / 544)) on every side, "kAvB" = A
    horizontally and B vertically (a (horizontal, vertical) tuple; "k6v4", the default, is textspot-desktop's), "auto"
    = k4 (eval/desktop.py's), or an integer."""
    if pad is None:
        pad = READ_PAD
    if pad == "auto":
        pad = "k4"
    s = str(pad)
    if s.startswith("k"):
        a, _, b = s[1:].partition("v")
        ph, pv = max(2, int(round(float(a) * H / 544))), max(2, int(round(float(b or a) * H / 544)))
        return ph if ph == pv else (ph, pv)
    return int(pad)


def crop_of(b, p, W, H):
    """A float box (x, y, w, h) grown by p px (or a (horizontal, vertical) pair), rounded outward and clipped: integer
    (x, y, w, h), or None."""
    ph, pv = p if isinstance(p, tuple) else (p, p)
    x0, y0 = max(0, int(math.floor(b[0] - ph))), max(0, int(math.floor(b[1] - pv)))
    x1, y1 = min(W, int(math.ceil(b[0] + b[2] + ph))), min(H, int(math.ceil(b[1] + b[3] + pv)))
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


def reading_order(boxes):
    """Indices of boxes (x, y, w, h first) in reading order: rows top to bottom (a box joins a row when it overlaps
    the row's first box vertically by at least half the smaller height), left to right within a row."""
    idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3] / 2.0, boxes[i][0]))
    rows = []
    for i in idx:
        x, y, w, h = boxes[i][:4]
        if rows:
            fx, fy, fw, fh = boxes[rows[-1][0]][:4]
            if min(fy + fh, y + h) - max(fy, y) >= 0.5 * min(fh, h):
                rows[-1].append(i)
                continue
        rows.append([i])
    return [i for r in rows for i in sorted(r, key=lambda j: boxes[j][0])]


def read_frame(frame, det, rec, pad=READ_PAD, lm=None, tags=None, workers=1, executor=None, drop=None):
    """Every line of a frame, top to bottom: [Line] in frame px. det: a Detector; rec: a vocr_onnx.Recognizer;
    lm: a vocr_onnx.CharLM; tags: the tag classes to read (None: all); workers: lines read on that many threads;
    drop: None (off) or DROP_DEFAULTS options: a horizontal line whose reading drop_reading() rejects is left out."""
    import vocr_onnx as V
    g = grey_of(frame)
    H, W = g.shape
    dropo = drop_options(drop) if drop else None
    boxes = det.detect(g)
    if tags is not None:
        boxes = [b for b in boxes if b.tag in tags]
    p = pad_of(pad, H)
    order = reading_order([(b.x, b.y, b.w, b.h) for b in boxes])
    jobs = []
    for i in order:
        b = boxes[i]
        cb = crop_of((b.x, b.y, b.w, b.h), p, W, H)
        if cb is not None:
            jobs.append((b, cb))

    def one(job):
        b, (cx, cy, cw, ch) = job
        tight = (b.x, b.y, b.w, b.h)
        if b.vertical:
            return Line("", tight, (cx, cy, cw, ch), b.tag, b.logit, 0, 1)
        dark = V.line_dark(g, (cx, cy, cw, ch))
        r = rec.read(g[cy:cy + ch, cx:cx + cw], invert=bool(dark), lm=lm)
        if dropo is not None and drop_reading(r, dropo):
            return None
        return Line(r.text, tight, (cx, cy, cw, ch), b.tag, b.logit, int(dark), 0)

    if executor is not None and len(jobs) > 1:
        return [ln for ln in executor.map(one, jobs) if ln is not None]
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return [ln for ln in ex.map(one, jobs) if ln is not None]
    return [ln for ln in map(one, jobs) if ln is not None]


def _load_image(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def main(argv=None):
    import argparse
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vocr_onnx as V
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, help="textspot-desktop's ONNX file")
    ap.add_argument("--rec", required=True, help="a VOCR6 recognizer's ONNX file (VOCR6.2_L_fast.onnx)")
    ap.add_argument("--lm", help="a VOCR character LM (.vlm): beam 8 + the LM at Rubi's settings")
    ap.add_argument("--size", help="the detector's input policy: fit:N, long:N or native (default: the file's)")
    ap.add_argument("--pad", default="k6v4", help="the reading pad: k6v4 (the default), kN, auto or N frame px")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args(argv)
    det = Detector(a.det, threads=1, size=a.size)
    rec = V.Recognizer(a.rec, threads=1)
    lm = V.CharLM(a.lm, tier=rec.name[-1:] if rec.name else None) if a.lm else None
    for p in a.images:
        img = _load_image(p)
        t0 = time.perf_counter()
        lines = read_frame(img, det, rec, pad=a.pad, lm=lm, workers=a.threads)
        ms = (time.perf_counter() - t0) * 1e3
        print(json.dumps({"image": p, "ms": round(ms, 1), "lines": [
            {"text": ln.text, "box": [round(v, 1) for v in ln.box], "crop": list(ln.crop), "tag": CLASSES[ln.tag],
             "vertical": ln.vertical} for ln in lines]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

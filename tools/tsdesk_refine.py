#!/usr/bin/env python3
"""textspot-desktop's box refinement (issue #173, eval/textspot_desktop_protocol.json detector.refinement): each
horizontal box's four edges snapped to its line's ink on the full-resolution grey frame. This file is the reference
(the Python twin); desktop/vocr_tsd_refine.c is the C version and gives the same boxes bit for bit: every step is
integer arithmetic on the grey frame (the input box's floor / ceil aside), so the two cannot drift by rounding.

    from tsdesk_refine import refine
    boxes = refine(grey, boxes)          # [[x0, y0, x1, y1, vertical, ...]] frame px; vertical boxes unchanged

    python3 tools/tsdesk_refine.py boxes --frames F.pkl --boxes B.jsonl --out B_refined.jsonl [--impl c]
        a detector's boxes file (eval/desktop.py detect's format) refined on every frame's grey (tools/tsdesk.grey_of)

For one horizontal box (frame px, floats), with r(v, p) = (v p + 50) div 100 (a percentage, rounded half up):

  1. the pixel box: ix0 = floor(x0), iy0 = floor(y0), ix1 = ceil(x1), iy1 = ceil(y1), clipped to the frame; h, w its
     height and width (a box under min_h x min_w px is kept: SMALL);
  2. the window: the pixel box grown by A = max(band_min, r(h, band_pct)) rows across, by
     max(along_min, r(h, start_pct)) columns before its start and max(along_min, r(h, end_pct)) after its end (a
     line's end may lack a long ―― or …… that its start never has), clipped to the frame;
  3. the background bg: the lower median (the element (n - 1) div 2 of the sorted values) of the window's one-pixel
     border ring; d = |g - bg| on the window;
  4. the local contrast c3: the 3 x 3 max - min of the grey on the window (the frame's edge pixels repeated beyond
     its border);
  5. the thresholds: t = max(floor, r(q99(d), k_pct), noise) and tc = max(floor, r(q99(c3), k_pct)), q99 over the
     pixel box (qP of n values: the element ceil(P n / 100) - 1 of the sorted values); noise = the lower of the
     margin_pct quantiles of d on the along margins (the box's rows, the window's columns more than
     max(2, r(h, margin_gap_pct)) px beyond either end of the box; a margin under margin_min columns is not used)
     plus noise_add, or 0 when neither margin is used: a textured background (a dithered or patterned box, a scene)
     raises it above its own texture;
  6. the ink: d > t and c3 > tc, without the pixels that have no 4-neighbour in it (specks, dither);
  7. across: the rows' ink counts over the box's columns; the body is the core's ink rows (the pixel box less
     min(r(h, core_pct), (h - 1) div 2) rows each side); the top edge walks up from the body's first row through ink
     rows (gaps of at most row_gap empty rows) to its last ink row, the bottom edge down the same way;
  8. along: the columns' ink counts over the new rows [rt, rb), H' = rb - rt, a barrier column (its ink, a column
     either side included, unbroken from max(1, r(H', bar_pct)) rows above the new rows to as many below them: a box
     frame, a border) counted empty; the body is the along core's ink columns (the pixel box less
     min(r(H', along_core_pct), (w - 1) div 2) columns at each end); the left edge walks left from the body's first
     column, crossing gaps of at most G = max(gap_min, r(H', gap_pct)) empty columns and stopping at a barrier, to
     its last ink column; the right edge the same way right;
  9. across again (step 7) over the new columns.

Guards (the edge that fails keeps the detector's value; `status` says why):
  - NOINK: no ink in the body (step 7 or 8): the box is kept;
  - LIMIT: an edge whose walk ran out of the window within its gap of ink (the window's side not being the frame's
    border): the line goes on beyond the search band (an adjacent line, ruby taller than the band, a busy
    background), so that edge is kept;
  - SHRINK: an edge never moves inward by more than r(h, shrink_a_pct) rows or r(H', shrink_l_pct) columns;
  - BUSY: when more than busy_pct % of the window's pixels outside the new box are ink, the box is kept.
Vertical boxes are left alone.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

DEFAULTS = {
    "min_h": 4, "min_w": 4,
    "band_pct": 25, "band_min": 2,
    "start_pct": 180, "end_pct": 400, "along_min": 3,
    "core_pct": 25, "along_core_pct": 50,
    "row_gap": 0, "valley_pct": 8, "low_run_pct": 20, "rise_pct": 50, "beyond_pct": 15,
    "gap_pct": 70, "gap_min": 2, "gap_start_pct": 35,
    "bar_pct": 15,
    "shrink_a_pct": 10, "shrink_l_pct": 25,
    "floor": 28, "k_pct": 40,
    "margin_pct": 90, "margin_gap_pct": 50, "margin_min": 3, "noise_add": 12, "texture_pct": 0,
    "busy_pct": 25,
    "neighbours": 1,
}
KEYS = tuple(DEFAULTS)      # the C struct's field order (desktop/vocr_tsd_refine.h)

# status bits (one word per box)
ST_SNAP_L, ST_SNAP_T, ST_SNAP_R, ST_SNAP_B = 1, 2, 4, 8        # the edge was snapped to ink
ST_SMALL = 16
ST_NOINK = 32
ST_LIMIT = 64
ST_SHRINK = 128
ST_BUSY = 512
ST_VERT = 1024
STATUS_NAMES = {ST_SMALL: "small", ST_NOINK: "noink", ST_LIMIT: "limit", ST_SHRINK: "shrink", ST_BUSY: "busy",
                ST_VERT: "vertical"}


def r100(v, pct):
    """(v pct + 50) div 100: v pct / 100 rounded half up (v, pct >= 0 integers)."""
    return (int(v) * int(pct) + 50) // 100


def quantile(vals, pct):
    """The element ceil(pct n / 100) - 1 (clipped to [0, n - 1]) of the sorted integer values (0 when empty)."""
    a = np.asarray(vals).ravel()
    n = int(a.size)
    if n == 0:
        return 0
    idx = (int(pct) * n + 99) // 100 - 1
    idx = 0 if idx < 0 else (n - 1 if idx > n - 1 else idx)
    return int(np.partition(a, idx)[idx])


def _c3(g, x0, y0, x1, y1):
    """The 3 x 3 max - min of grey g over [y0, y1) x [x0, x1) (int16), the frame's edge pixels repeated."""
    H, W = g.shape
    xs = np.clip(np.arange(x0 - 1, x1 + 1), 0, W - 1)
    ys = np.clip(np.arange(y0 - 1, y1 + 1), 0, H - 1)
    s = g[np.ix_(ys, xs)].astype(np.int16)
    hmax = np.maximum(np.maximum(s[:, :-2], s[:, 1:-1]), s[:, 2:])
    hmin = np.minimum(np.minimum(s[:, :-2], s[:, 1:-1]), s[:, 2:])
    vmax = np.maximum(np.maximum(hmax[:-2], hmax[1:-1]), hmax[2:])
    vmin = np.minimum(np.minimum(hmin[:-2], hmin[1:-1]), hmin[2:])
    return vmax - vmin


def _walk(cnt, start, step, n, gap, stop=None, gap2=None, edge=None):
    """From index `start` (an ink entry) move by `step` (-1 or +1) through entries with cnt > 0, crossing runs of at
    most `gap` empty entries (`gap2` once the walk is past index `edge` in its direction), within [0, n); an entry
    with stop[i] set ends the walk. (the last ink index, True when the walk ran out of [0, n) within the gap + 1 of
    its last ink: the ink may go on beyond)."""
    last = start
    i = start + step
    run = 0
    while 0 <= i < n:
        if stop is not None and stop[i]:
            return last, False
        if cnt[i] > 0:
            last = i
            run = 0
        else:
            run += 1
            g_ = gap2 if gap2 is not None and (i - edge) * step > 0 else gap
            if run > g_:
                return last, False
        i += step
    return last, True


def _walk_rows(cnt, start, step, n, gap, low, rise, low_max, beyond_max, stop=None):
    """The across walk (_walk with `gap` empty rows) that also ends at a valley: a row with 0 < cnt <= low is a low
    row, and a valley the first low row of a run of them. The walk ends at the latest valley (that row included) when
    the run grows longer than low_max rows (a sparse band: ruby, a texture), when a row with cnt >= rise follows it (a
    touching band: ruby, an adjacent line) or when the walk goes on more than beyond_max rows past it."""
    last = start
    i = start + step
    run = 0
    lows = 0
    valley = -1
    while 0 <= i < n:
        if stop is not None and stop[i]:
            return last, False
        c = int(cnt[i])
        if c > 0:
            if c <= low:
                if lows == 0:
                    valley = i
                lows += 1
                if lows > low_max:
                    return valley, False
            else:
                lows = 0
                if valley >= 0 and c >= rise:
                    return valley, False
            if valley >= 0 and (i - valley) * step > beyond_max:
                return valley, False
            last = i
            run = 0
        else:
            run += 1
            if run > gap:
                return last, False
        i += step
    return last, True


def refine_box(g, b, o=None, dbg=None, lim=None):
    """(x0, y0, x1, y1, status) of one horizontal box b = (x0, y0, x1, y1) on grey g (H x W uint8): the module
    docstring's steps. o: options (DEFAULTS); dbg: a dict that receives the internals (tests and diagnostics); lim:
    (left, top, right, bottom) frame px the edges may not grow past (the other boxes, neighbour_limits)."""
    if o is None:
        o = DEFAULTS
    H, W = g.shape
    x0, y0, x1, y1 = (float(v) for v in b[:4])
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return x0, y0, x1, y1, ST_SMALL
    ix0 = min(max(int(math.floor(x0)), 0), W)
    iy0 = min(max(int(math.floor(y0)), 0), H)
    ix1 = min(max(int(math.ceil(x1)), 0), W)
    iy1 = min(max(int(math.ceil(y1)), 0), H)
    h, w = iy1 - iy0, ix1 - ix0
    if h < o["min_h"] or w < o["min_w"]:
        return x0, y0, x1, y1, ST_SMALL
    A = max(o["band_min"], r100(h, o["band_pct"]))
    Ls = max(o["along_min"], r100(h, o["start_pct"]))
    Le = max(o["along_min"], r100(h, o["end_pct"]))
    wx0, wx1 = max(0, ix0 - Ls), min(W, ix1 + Le)
    wy0, wy1 = max(0, iy0 - A), min(H, iy1 + A)
    mh, mw = wy1 - wy0, wx1 - wx0
    bx0, by0 = ix0 - wx0, iy0 - wy0                     # the pixel box in window coordinates
    win = g[wy0:wy1, wx0:wx1].astype(np.int16)
    ring = np.concatenate([win[0], win[-1], win[1:-1, 0], win[1:-1, -1]])
    bg = int(np.partition(ring, (ring.size - 1) // 2)[(ring.size - 1) // 2])
    d = np.abs(win - bg)
    c3 = _c3(g, wx0, wy0, wx1, wy1)
    tc = max(o["floor"], r100(quantile(c3[by0:by0 + h, bx0:bx0 + w], 99), o["k_pct"]))
    mg = max(2, r100(h, o["margin_gap_pct"]))
    qs = []
    for sx0, sx1 in ((0, bx0 - mg), (bx0 + w + mg, mw)):
        if sx1 - sx0 < o["margin_min"]:
            continue
        dm, cm = d[by0:by0 + h, sx0:sx1], c3[by0:by0 + h, sx0:sx1]
        tex = dm[cm > tc]
        if o["texture_pct"] > 0 and tex.size * 100 < o["texture_pct"] * dm.size:
            qs.append(0)                                 # an untextured margin: no noise term
        else:
            qs.append(quantile(tex if o["texture_pct"] > 0 else dm, o["margin_pct"]) + o["noise_add"])
    noise = min(qs) if qs else 0
    t = max(o["floor"], r100(quantile(d[by0:by0 + h, bx0:bx0 + w], 99), o["k_pct"]), noise)
    m0 = (d > t) & (c3 > tc)
    nb = np.zeros_like(m0)
    nb[1:] |= m0[:-1]
    nb[:-1] |= m0[1:]
    nb[:, 1:] |= m0[:, :-1]
    nb[:, :-1] |= m0[:, 1:]
    m = m0 & nb
    if dbg is not None:
        dbg.update(win=(wx0, wy0, wx1, wy1), bg=bg, t=t, tc=tc, noise=noise, mask=m, A=A, Ls=Ls, Le=Le)
    lim_t, lim_b, lim_l, lim_r = wy0 > 0, wy1 < H, wx0 > 0, wx1 < W     # window sides that are band limits
    S_a = r100(h, o["shrink_a_pct"])
    rstop = cstop = None
    if lim is not None:                     # rows / columns of the window beyond another box: the walks stop there
        nl, nt, nr, nb_ = (int(v) for v in lim)
        ys = np.arange(mh) + wy0
        xs = np.arange(mw) + wx0
        rstop = (ys < nt) | (ys >= nb_)
        cstop = (xs < nl) | (xs >= nr)

    def across(cx0, cx1):
        """(top or None, bottom or None, status) in window rows over window columns [cx0, cx1); None: no ink."""
        rc = m[:, cx0:cx1].sum(1)
        q = min(r100(h, o["core_pct"]), (h - 1) // 2)
        k0, k1 = by0 + q, by0 + h - q
        ink = np.flatnonzero(rc[k0:k1] > 0)
        if not len(ink):
            return None
        vals = rc[k0:k1][ink]
        med = int(np.partition(vals, (vals.size - 1) // 2)[(vals.size - 1) // 2])
        low, rise = r100(med, o["valley_pct"]), r100(med, o["rise_pct"])
        if o["valley_pct"] <= 0:
            low = 0
        low_max = max(2, r100(h, o["low_run_pct"]))
        beyond = max(2, r100(h, o["beyond_pct"]))
        st = 0
        top, more = _walk_rows(rc, k0 + int(ink[0]), -1, mh, o["row_gap"], low, rise, low_max, beyond, rstop)
        if more and lim_t:
            top, st = None, st | ST_LIMIT
        elif top - by0 > S_a:
            top, st = None, st | ST_SHRINK
        bot, more = _walk_rows(rc, k0 + int(ink[-1]), +1, mh, o["row_gap"], low, rise, low_max, beyond, rstop)
        if more and lim_b:
            bot, st = None, st | ST_LIMIT
        elif (by0 + h - 1) - bot > S_a:
            bot, st = None, st | ST_SHRINK
        return top, bot, st

    a = across(bx0, bx0 + w)
    if a is None:
        return x0, y0, x1, y1, ST_NOINK
    top, bot, status = a
    rt = top if top is not None else by0
    rb = bot + 1 if bot is not None else by0 + h
    hh = rb - rt
    cc = m[rt:rb].sum(0)
    bar = max(1, r100(hh, o["bar_pct"]))
    barrier = None
    if o["bar_pct"] > 0 and rt - bar >= 0 and rb + bar <= mh:
        s3 = m[rt - bar:rb + bar]
        m3 = s3.copy()
        m3[:, 1:] |= s3[:, :-1]
        m3[:, :-1] |= s3[:, 1:]
        barrier = m3.all(0)
        cc = np.where(barrier, 0, cc)
    c = min(r100(hh, o["along_core_pct"]), (w - 1) // 2)
    k0, k1 = bx0 + c, bx0 + w - c
    ink = np.flatnonzero(cc[k0:k1] > 0)
    if not len(ink):
        return x0, y0, x1, y1, status | ST_NOINK
    G = max(o["gap_min"], r100(hh, o["gap_pct"]))
    S_l = r100(hh, o["shrink_l_pct"])
    if cstop is not None:
        barrier = cstop if barrier is None else (barrier | cstop)
    Gs = max(o["gap_min"], r100(hh, o["gap_start_pct"])) if o["gap_start_pct"] > 0 else None
    left, more = _walk(cc, k0 + int(ink[0]), -1, mw, G, barrier, Gs, bx0)
    if more and lim_l:
        left, status = None, status | ST_LIMIT
    elif left - bx0 > S_l:
        left, status = None, status | ST_SHRINK
    right, more = _walk(cc, k0 + int(ink[-1]), +1, mw, G, barrier)
    if more and lim_r:
        right, status = None, status | ST_LIMIT
    elif (bx0 + w - 1) - right > S_l:
        right, status = None, status | ST_SHRINK
    cl = left if left is not None else bx0
    cr = right + 1 if right is not None else bx0 + w
    a = across(cl, cr)
    if a is not None:
        top, bot, st = a
        status |= st
    bt = top if top is not None else by0
    bb = bot + 1 if bot is not None else by0 + h
    outside = mh * mw - (bb - bt) * (cr - cl)
    ink_out = int(m.sum()) - int(m[bt:bb, cl:cr].sum())
    if outside > 0 and ink_out * 100 > o["busy_pct"] * outside:
        return x0, y0, x1, y1, status | ST_BUSY
    out = [x0, y0, x1, y1]
    if left is not None:
        out[0], status = float(wx0 + left), status | ST_SNAP_L
    if top is not None:
        out[1], status = float(wy0 + top), status | ST_SNAP_T
    if right is not None:
        out[2], status = float(wx0 + right + 1), status | ST_SNAP_R
    if bot is not None:
        out[3], status = float(wy0 + bot + 1), status | ST_SNAP_B
    return out[0], out[1], out[2], out[3], status


def neighbour_limits(boxes, W, H):
    """[(left, top, right, bottom)] for every box: the frame px its edges may not grow past, from the other boxes
    (vertical ones too) that lie beside it (a vertical overlap of at least half the smaller height: left and right)
    or above / below it (a horizontal overlap of at least half the smaller width: top and bottom). A box that
    overlaps it already stops that edge at the box's own pixel edge."""
    n = len(boxes)
    px = []
    for b in boxes:
        v = [float(t) for t in b[:4]]
        if not all(math.isfinite(t) for t in v):
            px.append(None)
            continue
        px.append((v, min(max(int(math.floor(v[0])), 0), W), min(max(int(math.floor(v[1])), 0), H),
                   min(max(int(math.ceil(v[2])), 0), W), min(max(int(math.ceil(v[3])), 0), H)))
    out = []
    for i in range(n):
        if px[i] is None:
            out.append((0, 0, W, H))
            continue
        (a0, a1, a2, a3), ix0, iy0, ix1, iy1 = px[i]
        nl, nt, nr, nb = 0, 0, W, H
        for j in range(n):
            if j == i or px[j] is None:
                continue
            (b0, b1, b2, b3), jx0, jy0, jx1, jy1 = px[j]
            oy = min(a3, b3) - max(a1, b1)
            ox = min(a2, b2) - max(a0, b0)
            if oy >= 0.5 * min(a3 - a1, b3 - b1):
                if b0 > a0:
                    nr = min(nr, max(ix1, jx0))
                if b2 < a2:
                    nl = max(nl, min(ix0, jx1))
            if ox >= 0.5 * min(a2 - a0, b2 - b0):
                if b1 < a1:
                    nt = max(nt, min(iy0, jy1))
                if b3 > a3:
                    nb = min(nb, max(iy1, jy0))
        out.append((nl, nt, nr, nb))
    return out


def options(opts=None):
    o = dict(DEFAULTS)
    for k, v in (opts or {}).items():
        if k not in DEFAULTS:
            raise KeyError(f"unknown refinement option {k!r}")
        o[k] = int(v)
    return o


def refine(grey, boxes, opts=None, status=None):
    """The boxes ([[x0, y0, x1, y1, vertical, ...]] frame px) with every horizontal box's edges snapped to its ink
    (module docstring); vertical boxes and every entry after the fourth unchanged. status: a list that receives each
    box's status bits."""
    o = options(opts)
    g = np.ascontiguousarray(grey, np.uint8)
    if g.ndim != 2:
        raise ValueError("refine reads an H x W grey frame (tools/tsdesk.grey_of)")
    lims = neighbour_limits(boxes, g.shape[1], g.shape[0]) if o["neighbours"] else [None] * len(boxes)
    out = []
    for b, lim in zip(boxes, lims):
        nb = list(b)
        if len(b) > 4 and b[4]:
            if status is not None:
                status.append(ST_VERT)
            out.append(nb)
            continue
        x0, y0, x1, y1, st = refine_box(g, b, o, lim=lim)
        nb[:4] = [x0, y0, x1, y1]
        if status is not None:
            status.append(st)
        out.append(nb)
    return out


def refine_det(grey, boxes, opts=None):
    """tools/tsdesk.Box objects (x, y, w, h) refined: new Box objects, the same order."""
    import dataclasses
    rows = [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical] for b in boxes]
    out = []
    for b, r in zip(boxes, refine(grey, rows, opts)):
        out.append(dataclasses.replace(b, x=r[0], y=r[1], w=r[2] - r[0], h=r[3] - r[1]))
    return out


# ----------------------------------------------------------------------------- the C version (ctypes)

class CRefine:
    """desktop/vocr_tsd_refine.c through ctypes (a shared library built from it: desktop/Makefile's
    build/libvocrtsdr.so, or libvocrdesk.so, which links it)."""

    def __init__(self, lib=None, opts=None):
        import ctypes as C
        if lib is None:
            here = Path(__file__).resolve().parents[1] / "desktop" / "build"
            lib = next((p for p in (here / "libvocrtsdr.so", here / "libvocrdesk.so") if p.exists()), None)
            if lib is None:
                raise FileNotFoundError("build desktop/ first (make -C desktop build/libvocrtsdr.so)")
        self.C = C
        self.lib = C.CDLL(str(lib))
        self.lib.vocr_tsdr_defaults.argtypes = [C.POINTER(C.c_int)]
        self.lib.vocr_tsdr_nopts.restype = C.c_int
        self.lib.vocr_tsdr_scratch_bytes.restype = C.c_size_t
        self.lib.vocr_tsdr_scratch_bytes.argtypes = [C.c_uint, C.c_uint, C.c_uint]
        self.lib.vocr_tsdr_refine.restype = C.c_int
        self.lib.vocr_tsdr_refine.argtypes = [C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_void_p,
                                              C.c_void_p, C.c_uint, C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t]
        n = self.lib.vocr_tsdr_nopts()
        if n != len(KEYS):
            raise RuntimeError(f"the C refinement has {n} options, this file {len(KEYS)}")
        d = (C.c_int * n)()
        self.lib.vocr_tsdr_defaults(d)
        if list(d) != [DEFAULTS[k] for k in KEYS]:
            raise RuntimeError("the C refinement's defaults differ from tools/tsdesk_refine.DEFAULTS")
        o = options(opts)
        self.opts = (C.c_int * n)(*[o[k] for k in KEYS])
        self._scratch = None

    def refine(self, grey, boxes, status=None):
        g = np.ascontiguousarray(grey, np.uint8)
        H, W = g.shape
        n = len(boxes)
        xy = np.zeros((max(1, n), 4), np.float64)
        vert = np.zeros(max(1, n), np.uint8)
        for i, b in enumerate(boxes):
            xy[i] = [float(v) for v in b[:4]]
            vert[i] = 1 if (len(b) > 4 and b[4]) else 0
        st = np.zeros(max(1, n), np.uint32)
        need = self.lib.vocr_tsdr_scratch_bytes(W, H, n)
        if self._scratch is None or self._scratch.size < need:
            self._scratch = np.zeros(need, np.uint8)
        rc = self.lib.vocr_tsdr_refine(g.ctypes.data, W, H, W, xy.ctypes.data, vert.ctypes.data, n,
                                       self.opts, st.ctypes.data, self._scratch.ctypes.data, self._scratch.size)
        if rc != 0:
            raise RuntimeError(f"vocr_tsdr_refine: {rc}")
        out = []
        for i, b in enumerate(boxes):
            nb = list(b)
            nb[:4] = [float(v) for v in xy[i]]
            out.append(nb)
        if status is not None:
            status.extend(int(v) for v in st[:n])
        return out


# ----------------------------------------------------------------------------- command line

_W = {}


def _init(impl, opts):
    _W["impl"] = CRefine(opts=opts) if impl == "c" else None
    _W["opts"] = opts


def _job(job):
    import tsdesk as T

    import desktop as D
    fr, boxes = job
    g = T.grey_of(D.load_rgb(fr))
    st = []
    if _W["impl"] is not None:
        nb = _W["impl"].refine(g, boxes, st)
    else:
        nb = refine(g, boxes, _W["opts"], st)
    return fr["key"], nb, st


def cmd_boxes(a):
    import time
    from collections import Counter
    root = Path(__file__).resolve().parents[1]
    for p in (root / "eval", root / "tools", root / "synth", root / "train", root / "teacher", root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import desktop as D
    frames, _ = D.load_frames(a.frames)
    boxes, bmeta = D.load_boxes(a.boxes)
    if bmeta["frames_sha256"] != D.sha256_file(a.frames):
        raise SystemExit("the boxes were found on another frames file")
    opts = options(json.loads(a.opts) if a.opts else None)
    jobs = [(fr, boxes[fr["key"]]) for fr in frames if fr["key"] in boxes]
    t0 = time.time()
    res, counts = {}, Counter()
    with D._pool(a.workers, _init, (a.impl, opts)) as pool:
        for key, nb, st in pool.imap(_job, jobs, chunksize=8):
            res[key] = nb
            for v in st:
                counts["boxes"] += 1
                counts["edges_snapped"] += bin(v & 15).count("1")
                for bit, name in STATUS_NAMES.items():
                    if v & bit:
                        counts[name] += 1
    rows = []
    src = {r["key"]: r for r in D.read_jsonl(a.boxes)[1:]}
    for fr in frames:
        if fr["key"] in res:
            rows.append(dict(src[fr["key"]], boxes=res[fr["key"]]))
    meta = dict(bmeta, det=f"{bmeta['det']}+refine", refine={
        "impl": a.impl, "opts": opts, "code_sha256": D.sha256_file(__file__), "source_boxes": str(a.boxes),
        "source_boxes_sha256": D.sha256_file(a.boxes), "counts": dict(counts), "seconds": round(time.time() - t0, 1)})
    D.write_jsonl(a.out, [{"meta": meta}, *rows])
    print(f"refine: {len(rows)} frames, {dict(counts)} -> {a.out}", flush=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("boxes")
    p.add_argument("--frames", required=True)
    p.add_argument("--boxes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opts", help="JSON: options that differ from DEFAULTS")
    p.add_argument("--impl", default="py", choices=("py", "c"))
    p.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    {"boxes": cmd_boxes}[a.cmd](a)


if __name__ == "__main__":
    main()

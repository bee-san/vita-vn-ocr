"""Bit-exact Python twin of runtime/vocr_lines_refine.c, the post-finder refinement of the line finder's boxes.

  boxes, dark, origin = refine_lines(img, boxes, dark, flags=DEFAULT)

img is the region the finder ran on (HxW grey or HxWx3/4 RGB(A) uint8), boxes and dark are what
common.find_lines_ex(img, "grad") returned, and the result is the refined set: boxes [(x, y, w, h)],
their dark flags, and for each output line the index of the finder line it came from (a split box
gives several). flags = 0 returns the input unchanged. See docs/finder-tighten.md for the method and
its measurements. DEFAULT is every step (TIGHTEN | FRAMES | SPLIT | CURSOR | PORTRAIT). Keep the
constants in step with runtime/vocr_lines_refine.c.
"""
import numpy as np

from common import LINES_MAX, grey_of

TIGHTEN, FRAMES, SPLIT, CURSOR, PORTRAIT = 0x01, 0x02, 0x04, 0x08, 0x10
ALL = 0x1F
DEFAULT = ALL
NAMES = {"tighten": TIGHTEN, "frames": FRAMES, "split": SPLIT, "cursor": CURSOR, "portrait": PORTRAIT}

R_DMIN = 16
R_FLIP = 4
R_MINROW = 2
R_MINRUNS = 2
R_SPLIT_H = 80
R_SPLIT_M = 25
R_SPLIT_MINH = 6
R_SPLIT_G = 70
R_MARGIN = 16
R_CUR_GAP = 15
R_CUR_W = 15
R_POR_GAP = 10
R_POR_SPAN = 19
R_POR_EXT = 5
R_POR_W = 15
R_SOLID = 6
R_XMARGIN = 50


def parse_flags(text):
    """'default', 'all', 'none', an integer, or names joined by '+' (tighten+frames+split)."""
    text = str(text).strip().lower()
    if text in ("default", ""):
        return DEFAULT
    if text == "all":
        return ALL
    if text in ("none", "off", "0"):
        return 0
    if text.isdigit() or text.startswith("0x"):
        return int(text, 0)
    flags = 0
    for part in text.split("+"):
        flags |= NAMES[part]
    return flags


def _pct(hist, k):
    """Smallest value whose cumulative count reaches k (k = 0 gives 0)."""
    return int(np.argmax(np.cumsum(hist) >= k))


def _stats(mid, dark):
    """bg, d and the mask's polarity from the box's middle rows (C: stats_of)."""
    hist = np.bincount(mid.ravel(), minlength=256).astype(np.int64)
    n = int(mid.size)
    bg = _pct(hist, n // 2 + 1)
    lo = _pct(hist, n * 2 // 100)
    hi = _pct(hist, n * 98 // 100)
    d = max(R_DMIN, (hi - lo) // 4)
    v = np.arange(256)
    nd = int(hist[v + d < bg].sum())
    nb = int(hist[v > bg + d].sum())
    mine, other = (nd, nb) if dark else (nb, nd)
    mdark = (0 if dark else 1) if other > R_FLIP * mine else (1 if dark else 0)
    return bg, d, mdark


def _mask(g, dark, frames):
    """The box's ink mask (frame rows cleared) and its row counts (0 for a row of fewer than R_MINRUNS runs)."""
    h, w = g.shape
    q = h // 4
    bg, d, mdark = _stats(g[q:h - q], dark)
    gi = g.astype(np.int32)
    m = (gi + d < bg) if mdark else (gi > bg + d)
    cnt = m.sum(1).astype(np.int64)
    for y in np.flatnonzero(cnt):
        e = np.diff(np.concatenate([[0], m[y].astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
        if frames and 2 * int((ends - starts).max()) >= w:
            m[y] = False
            cnt[y] = 0
        elif len(starts) < R_MINRUNS:
            cnt[y] = 0
    return m, cnt


def _bands(cnt):
    """[(a, b, mass)]: runs of rows with at least R_MINROW ink pixels."""
    on = (cnt >= R_MINROW).astype(np.int8)
    e = np.diff(np.concatenate([[0], on, [0]]))
    return [(int(a), int(b), int(cnt[a:b].sum())) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]


def _clusters(col, lo, hi):
    """Runs of columns with ink within [lo, hi)."""
    on = (col[lo:hi] > 0).astype(np.int8)
    e = np.diff(np.concatenate([[0], on, [0]]))
    return [(lo + int(s), lo + int(t)) for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]


def _glyph_height(m, a, b):
    """Upper median of the column clusters' ink heights within band rows [a, b) (C: glyph_height)."""
    band = m[a:b]
    hs = []
    for s, e in _clusters(band.sum(0), 0, band.shape[1]):
        rows = np.flatnonzero(band[:, s:e].any(1))
        hs.append(int(rows[-1] + 1 - rows[0]))
    hs.sort()
    return hs[len(hs) // 2] if hs else 0


def _part_rows(m, ya, yb, s, e):
    """(t, b, up, dn): the band rows with ink in columns [s, e), and how far the run of rows with ink there
    continues above and below them; None if no ink."""
    has = m[:, s:e].any(1)
    rows = np.flatnonzero(has[ya:yb])
    if not len(rows):
        return None
    t, b = ya + int(rows[0]), ya + int(rows[-1]) + 1
    top, bot = t, b
    while top > 0 and has[top - 1]:
        top -= 1
    while bot < len(has) and has[bot]:
        bot += 1
    return t, b, t - top, bot - b


def _portrait_part(m, ya, yb, col, s, e):
    hb = yb - ya
    r = _part_rows(m, ya, yb, s, e)
    if r is None:
        return False
    t, b, up, dn = r
    pw = e - s
    mass = int(col[s:e].sum())
    if 10 * (up + (b - t) + dn) >= R_POR_SPAN * hb and 10 * up >= R_POR_EXT * hb and 10 * dn >= R_POR_EXT * hb:
        return True
    return 10 * pw > R_POR_W * hb and 10 * mass >= R_SOLID * pw * (b - t)


def _trim_columns(m, ya, yb, flags):
    h, w = m.shape
    hb = yb - ya
    mx = (R_XMARGIN * hb + 50) // 100
    col = m[ya:yb].sum(0).astype(np.int64)
    x0, x1 = 0, w
    cl = _clusters(col, 0, w)
    if len(cl) < 2:
        return x0, x1
    if flags & CURSOR:
        (_, pe), (ls, le) = cl[-2], cl[-1]
        gap, cw = ls - pe, le - ls
        r = _part_rows(m, ya, yb, ls, le) if 10 * gap > R_CUR_GAP * hb and 10 * cw <= R_CUR_W * hb else None
        if r is not None:
            ph = r[1] - r[0]
            mass = int(col[ls:le].sum())
            if 4 * ph <= hb or (10 * mass >= R_SOLID * cw * ph and 10 * cw >= 6 * ph):
                x1 = min(pe + mx, (pe + ls) // 2)
    if flags & PORTRAIT:
        lim = x1
        cs = _clusters(col, 0, lim)
        if cs:
            gs, ge = cs[0]
            for s, e in cs[1:]:
                if 10 * (s - ge) > R_POR_GAP * hb:
                    if _portrait_part(m, ya, yb, col, gs, ge):
                        x0 = max(s - mx if s > mx else 0, (ge + s + 1) // 2)
                    break
                ge = e
        cs = _clusters(col, x0, lim)
        prev_e, part_s, gap_s = None, 0, 0
        for s, e in cs:
            if prev_e is not None and 10 * (s - prev_e) > R_POR_GAP * hb:
                gap_s, part_s = prev_e, s
            prev_e = e
        if part_s and _portrait_part(m, ya, yb, col, part_s, prev_e):
            x1 = min(gap_s + mx, (gap_s + part_s) // 2)
    return x0, x1


def refine_box(g, dark, flags, room):
    """Box-relative refined boxes [(x0, y0, x1, y1)] of one finder box (grey crop g), or None for 'as it is'."""
    h, w = g.shape
    m, cnt = _mask(g, dark, bool(flags & FRAMES))
    bands = _bands(cnt)
    if not bands:
        return None
    ka, kb, km = bands[0]
    for a, b, mass in bands[1:]:
        if mass > km:
            ka, kb, km = a, b, mass
    sel = []
    if flags & SPLIT:
        kg = _glyph_height(m, ka, kb)
        sel = [(a, b) for a, b, mass in bands
               if a == ka or (100 * (b - a) >= R_SPLIT_H * (kb - ka) and 100 * mass >= R_SPLIT_M * km
                              and b - a >= R_SPLIT_MINH and 100 * _glyph_height(m, a, b) >= R_SPLIT_G * kg)]
    if len(sel) < 2 or len(sel) > room:
        sel = [(ka, kb)]
    out = []
    for i, (a, b) in enumerate(sel):
        ny0 = (sel[i - 1][1] + a + 1) // 2 if i else 0
        ny1 = (b + sel[i + 1][0]) // 2 if i + 1 < len(sel) else h
        if flags & TIGHTEN:
            mg = (R_MARGIN * (b - a) + 50) // 100
            ny0 = max(ny0, a - mg if a > mg else 0)
            ny1 = min(ny1, b + mg)
        x0, x1 = (0, w)
        if flags & (CURSOR | PORTRAIT):
            x0, x1 = _trim_columns(m, a, b, flags)
        out.append((x0, ny0, x1, ny1))
    return out


def refine_lines(img, boxes, dark, flags=DEFAULT):
    """The refined line set of a region: (boxes, dark, origin). flags = 0 returns the input unchanged."""
    boxes, dark = [tuple(int(v) for v in b) for b in boxes], [int(d) for d in dark]
    if not flags:
        return boxes, dark, list(range(len(boxes)))
    if flags & ~ALL:
        raise ValueError(f"unknown refinement flags {flags:#x}")
    g = grey_of(img)
    H, W = g.shape
    out_b, out_d, origin = [], [], []
    for i, ((x, y, w, h), d) in enumerate(zip(boxes, dark)):
        if not w or not h or x >= W or y >= H or w > W - x or h > H - y:
            raise ValueError(f"line {i} {(x, y, w, h)} is outside the {W}x{H} region")
        room = LINES_MAX - len(out_b) - (len(boxes) - i - 1)
        r = refine_box(g[y:y + h, x:x + w], d, flags, room)
        for x0, y0, x1, y1 in r if r is not None else [(0, 0, w, h)]:
            out_b.append((x + x0, y + y0, x1 - x0, y1 - y0))
            out_d.append(d)
            origin.append(i)
    return out_b, out_d, origin

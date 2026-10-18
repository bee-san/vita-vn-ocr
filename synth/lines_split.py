"""Bit-exact Python twin of runtime/vocr_lines_split.c: the stroke split of the line finder's boxes (issue #98,
docs/translucent-box.md).

  boxes, dark, origin = split_lines(img, boxes, dark, flags=DEFAULT)

img is the region the finder ran on (HxW grey or HxWx3/4 RGB(A) uint8), boxes and dark what
common.find_lines_ex(img, "grad") returned; the result is the new set: boxes [(x, y, w, h)], their dark flags, and
for each output line the index of the finder line it came from (a split box gives several). flags = 0 returns the
input unchanged. Keep the constants in step with runtime/vocr_lines_split.c.

Per box (x, y, w, h) of at least S_MIN_H rows, from the box's own grey pixels only (indices clamped at its edges):
  1. Stroke maps, along the box's rows. B (bright ridge) = how far a pixel is brighter than the brighter of the
     darkest pixels within S_R on its left and on its right; D (dark ridge) the same for darker. A text row crosses
     the vertical strokes of its glyphs, thin lines of one polarity with the other on both sides (the fill against
     the outline or the box; the gaps between strokes the other way), so it holds both; a scene seen through a
     translucent box holds weaker or one-sided structure, and a step edge or a broad area is no ridge at all.
     C = max - min of the pixel and its left and right neighbours.
  2. Levels: tb, td = max(S_TFLOOR, 2/5 of the box's 99th percentile of B, D); tc = max(S_CFLOOR, 2/5 of C's).
     Per row: S(y) = min(#B >= tb, #D >= td), On(y) = #C >= tc.
  3. Segments: the box's rows split at valleys, deepest first, while S[y] * S_VDEN <= S_VNUM * min(the peak of S
     above y, the peak below it) within the segment, at most S_SEGS segments. A segment's band: its rows from the
     first to the last with S * S_BDEN >= S_BNUM * the segment's peak.
  4. Text bands: at least S_BAND_H % as tall as the tallest band, S_BAND_M % of the heaviest band's mass and
     S_BAND_MINH rows. A band follows the previous one as a line of its own only if the smallest On of the rows
     between them is at most S_CONT % of the upper median of On over both bands' rows (strokes crossing the gap,
     as when a band boundary runs through large glyphs, merge them again).
  5. Two or more text bands, and room in the set: the box becomes one line per band, rows = the band plus
     S_MARGIN % of its height above and below, never past the midpoint to the next band; the box's columns and
     dark flag. Otherwise (and for a box under S_MIN_H rows or over S_MAXW columns) the box stays as it is.
"""
import numpy as np

from common import LINES_MAX, grey_of

STROKE = 0x01
ALL = 0x01
DEFAULT = STROKE

S_MIN_H = 16
S_MAXW = 65535
S_R = 3
S_TFLOOR = 24
S_CFLOOR = 40
S_VNUM, S_VDEN = 1, 3
S_BNUM, S_BDEN = 1, 4
S_BAND_H = 70
S_BAND_M = 10
S_BAND_MINH = 6
S_CONT = 60
S_MARGIN = 16
S_SEGS = 16


def parse_flags(text):
    """'default', 'all', 'none' / 'off' / '0', or an integer."""
    text = str(text).strip().lower()
    if text in ("default", "all", "stroke", ""):
        return DEFAULT
    if text in ("none", "off", "0"):
        return 0
    return int(text, 0)


def _shift(a, k):
    """a[:, clip(x + k)]: columns shifted by k, edges replicated."""
    n = a.shape[1]
    return a[:, np.clip(np.arange(n) + k, 0, n - 1)]


def maps(g):
    """(B, D, C) of a grey box, along its rows: bright and dark ridge strength and the horizontal 3-pixel contrast
    (C: row_maps)."""
    gi = np.asarray(g, dtype=np.int16)
    left = [_shift(gi, -k) for k in range(1, S_R + 1)]
    right = [_shift(gi, k) for k in range(1, S_R + 1)]
    hi = np.maximum(np.minimum.reduce(left), np.minimum.reduce(right))
    lo = np.minimum(np.maximum.reduce(left), np.maximum.reduce(right))
    B = np.maximum(gi - hi, 0)
    D = np.maximum(lo - gi, 0)
    win = [_shift(gi, -1), gi, _shift(gi, 1)]
    C = np.maximum.reduce(win) - np.minimum.reduce(win)
    return B.astype(np.int32), D.astype(np.int32), C.astype(np.int32)


def level(m, floor):
    """max(floor, 2/5 of the 99th percentile): GRAD's threshold rule (smallest v with cum[v] * 100 >= n * 99)."""
    cum = np.cumsum(np.bincount(m.ravel(), minlength=256)).astype(np.int64)
    p99 = int(np.argmax(cum * 100 >= m.size * 99))
    return max(floor, 2 * p99 // 5)


def profiles(g):
    """(S, On) per row of a grey box."""
    B, D, C = maps(g)
    tb, td, tc = level(B, S_TFLOOR), level(D, S_TFLOOR), level(C, S_CFLOOR)
    S = np.minimum((B >= tb).sum(1), (D >= td).sum(1)).astype(np.int64)
    On = (C >= tc).sum(1).astype(np.int64)
    return S, On


def segments(S):
    """[(a, b)]: the rows split at valleys (C: segment), in order."""
    segs = [(0, len(S))]
    while True:
        out, split, count = [], False, len(segs)
        for a, b in segs:
            best, by = None, -1
            for y in range(a + 1, b - 1):
                m = min(int(S[a:y].max()), int(S[y + 1:b].max()))
                if m <= 0 or int(S[y]) * S_VDEN > S_VNUM * m:
                    continue
                r = int(S[y]) * 4096 // m
                if best is None or r < best:
                    best, by = r, y
            if by >= 0 and count < S_SEGS:
                out += [(a, by), (by + 1, b)]
                count += 1
                split = True
            else:
                out.append((a, b))
        segs = out
        if not split:
            return segs


def bands(S):
    """[(a, b, mass)] of the segments' bands (C: band_of)."""
    out = []
    for a, b in segments(S):
        pk = int(S[a:b].max()) if b > a else 0
        if pk <= 0:
            continue
        rows = [y for y in range(a, b) if int(S[y]) * S_BDEN >= S_BNUM * pk]
        out.append((rows[0], rows[-1] + 1, int(S[rows[0]:rows[-1] + 1].sum())))
    return out


def text_bands(S, On):
    """The bands that become lines (step 4), in order."""
    bs = bands(S)
    if len(bs) < 2:
        return bs
    hmax = max(b - a for a, b, _ in bs)
    mmax = max(m for _, _, m in bs)
    sel = [(a, b, m) for a, b, m in bs
           if 100 * (b - a) >= S_BAND_H * hmax and 100 * m >= S_BAND_M * mmax and b - a >= S_BAND_MINH]
    if len(sel) < 2:
        return sel
    keep = [sel[0]]
    for a, b, m in sel[1:]:
        pa, pb, pm = keep[-1]
        vals = sorted(int(v) for v in np.concatenate([On[pa:pb], On[a:b]]))
        med = vals[len(vals) // 2]
        if 100 * int(On[pb:a].min()) <= S_CONT * med:
            keep.append((a, b, m))
        else:
            keep[-1] = (pa, b, pm + m)
    return keep


def split_box(g, room):
    """Box-relative lines [(y0, y1)] of one finder box (grey crop g), or None for 'as it is'."""
    h, w = g.shape
    if h < S_MIN_H or w > S_MAXW:
        return None
    S, On = profiles(g)
    sel = text_bands(S, On)
    if len(sel) < 2 or len(sel) > room:
        return None
    out = []
    for i, (a, b, _) in enumerate(sel):
        mg = (S_MARGIN * (b - a) + 50) // 100
        lo = (sel[i - 1][1] + a + 1) // 2 if i else 0
        hi = (b + sel[i + 1][0]) // 2 if i + 1 < len(sel) else h
        out.append((max(lo, a - mg), min(hi, b + mg)))
    return out


def split_lines(img, boxes, dark, flags=DEFAULT):
    """The split line set of a region: (boxes, dark, origin). flags = 0 returns the input unchanged."""
    boxes, dark = [tuple(int(v) for v in b) for b in boxes], [int(d) for d in dark]
    if not flags:
        return boxes, dark, list(range(len(boxes)))
    if flags & ~ALL:
        raise ValueError(f"unknown split flags {flags:#x}")
    g = grey_of(img)
    H, W = g.shape
    out_b, out_d, origin = [], [], []
    for i, ((x, y, w, h), d) in enumerate(zip(boxes, dark)):
        if not w or not h or x >= W or y >= H or w > W - x or h > H - y:
            raise ValueError(f"line {i} {(x, y, w, h)} is outside the {W}x{H} region")
        room = LINES_MAX - len(out_b) - (len(boxes) - i - 1)
        r = split_box(g[y:y + h, x:x + w], room)
        for y0, y1 in r if r is not None else [(0, h)]:
            out_b.append((x, y + y0, w, y1 - y0))
            out_d.append(d)
            origin.append(i)
    return out_b, out_d, origin

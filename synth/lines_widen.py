"""Python twin of runtime/vocr_lines_widen.c (issue #109): a found line that the region's left or right edge cuts,
continued past the edge through the ink the edge cuts. Same integer arithmetic, so tests/lines_widen_parity.py
can require identical boxes.

    x, w, sides = widen(strip, rx, rw, x, w)   # strip: the line's rows over the region and its margins (grey or RGBA)
"""
import numpy as np

PAD = 3
MIN_INK = 16
LEFT, RIGHT = 1, 2


def grey(img):
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.int32)
    p = img.astype(np.int32)
    return (77 * p[..., 0] + 150 * p[..., 1] + 29 * p[..., 2]) >> 8


def widen(img, rx, rw, x, w):
    """(new x, new w, sides) in the strip's coordinates, as vocr_line_widen()."""
    g = grey(img)
    H, W = g.shape
    if not rw or rx > W or rw > W - rx or not w or x > rw or w > rw - x or not H or not W:
        raise ValueError("box or region outside the strip")
    rng = g.max(axis=0) - g.min(axis=0)
    left, right = rx + x, rx + x + w
    cmax = int(rng[left:right].max())
    t = max(cmax // 3, MIN_INK)
    gap_max = max(H // 8, 2)
    sides = 0
    if x == 0 and rx > 0 and (rng[rx] >= t or (rw > 1 and rng[rx + 1] >= t)):
        last, gap, ended = rx, 0, False
        for c in range(rx - 1, -1, -1):
            if rng[c] >= t:
                last, gap = c, 0
            else:
                gap += 1
                if gap >= gap_max:
                    ended = True
                    break
        if ended and last < rx:                      # a cut glyph ends inside the margin
            left = last - PAD if last > PAD else 0
            sides |= LEFT
    if x + w == rw and rx + rw < W and (rng[rx + rw - 1] >= t or (rw > 1 and rng[rx + rw - 2] >= t)):
        last, gap, ended = rx + rw - 1, 0, False
        for c in range(rx + rw, W):
            if rng[c] >= t:
                last, gap = c, 0
            else:
                gap += 1
                if gap >= gap_max:
                    ended = True
                    break
        if ended and last >= rx + rw:
            right = min(last + 1 + PAD, W)
            sides |= RIGHT
    return left, right - left, sides

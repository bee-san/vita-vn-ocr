"""Polarity-agnostic line finders for cutting synthetic training crops.

The overlay's own finder (common.find_lines) only sees bright pixels, so it
rejects dark-on-light boxes. Until the linefind work ships its C finder and a
bit-exact Python twin, render.py cuts crops with one of these:

find_lines_mask(ink)
    The overlay's projection rules (rows with >= 2 ink pixels form runs, a gap of
    3 rows ends a run, runs under 8 rows are ignored, a run over 88 rows or more
    than 8 lines rejects the region, boxes padded by 4 px and 5 px below), plus
    the research finder's furigana rule (runs under 0.55x the median run height
    are dropped). It runs on the renderer's exact text-ink mask instead of on
    bright pixels, so it behaves like an ideal finder of either polarity: the
    crop geometry matches what the overlay hands the recognizer for bright text,
    and the same geometry is used for dark, outlined and coloured text.

find_lines_grad(rgb)
    The research's unprivileged gradient-energy finder (docs/research-findings.md:
    rows above 25% of the maximum of |dx|+|dy| form runs, gaps under 3 px merge,
    runs under 8 px or under 0.55x the median height are dropped), with the line's
    horizontal extent trimmed to its edge-energy columns and the same padding.
    Used to measure how often a real image-only finder agrees with the mask crops.
"""
import numpy as np

MAX_LINES = 8
TOO_TALL = 88      # common.find_lines rejects a line run taller than MAX_LINE_H - 8 = 88 rows


def _runs(on, gap=3):
    """[start, last] row runs of a boolean profile; a run ends `gap` rows after its last on-row."""
    runs, start, last, active = [], 0, 0, False
    n = len(on)
    for y in range(n + 1):
        if y < n and on[y]:
            if not active:
                start, active = y, True
            last = y
        if active and (y == n or y - last >= gap):
            runs.append((start, last))
            active = False
    return runs


def _pad(x1, x2, start, last, w_img, h_img):
    lx = x1 - 4 if x1 > 4 else 0
    ly = start - 4 if start > 4 else 0
    ex = x2 + 4 if x2 + 4 < w_img else w_img
    ey = last + 5 if last + 5 < h_img else h_img
    return lx, ly, ex - lx, ey - ly


def _median_filter(runs, ratio=0.55):
    if not runs:
        return runs
    med = float(np.median([b - a + 1 for a, b in runs]))
    return [(a, b) for a, b in runs if b - a + 1 >= ratio * med]


def find_lines_mask(ink):
    """ink: HxW bool text mask. Returns [(x, y, w, h)] or None (region rejected)."""
    h_img, w_img = ink.shape
    counts = ink.sum(axis=1)
    runs = [(a, b) for a, b in _runs(counts >= 2) if b - a + 1 >= 8]
    if any(b - a + 1 > TOO_TALL for a, b in runs):
        return None
    runs = _median_filter(runs)
    if len(runs) > MAX_LINES:
        return None
    out = []
    for a, b in runs:
        cols = np.flatnonzero(ink[a:b + 1].any(axis=0))
        if len(cols):
            out.append(_pad(int(cols[0]), int(cols[-1]) + 1, a, b, w_img, h_img))
    return out


def edge_energy(rgb):
    c = rgb.astype(np.float32)
    g = (77 * c[..., 0] + 150 * c[..., 1] + 29 * c[..., 2]) / 256.0   # no BLAS: workers stay 1-thread
    e = np.zeros_like(g)
    e[:-1, :-1] = np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]
    return e


def find_lines_grad(rgb, col_frac=0.12):
    """rgb: HxWx3 uint8. Image-only polarity-agnostic finder; same output format."""
    h_img, w_img = rgb.shape[:2]
    e = edge_energy(rgb)
    prof = np.convolve(e.sum(axis=1), np.ones(3) / 3, mode="same")
    if prof.max() <= 0:
        return []
    runs = [(a, b) for a, b in _runs(prof > 0.25 * prof.max()) if b - a + 1 >= 8]
    runs = _median_filter(runs)
    if len(runs) > MAX_LINES or any(b - a + 1 > TOO_TALL for a, b in runs):
        return None
    out = []
    for a, b in runs:
        col = np.convolve(e[a:b + 1].sum(axis=0), np.ones(5) / 5, mode="same")
        cols = np.flatnonzero(col > col_frac * col.max()) if col.max() > 0 else []
        if len(cols):
            out.append(_pad(int(cols[0]), int(cols[-1]) + 1, a, b, w_img, h_img))
    return out

"""Polarity-agnostic line finders for cutting synthetic training crops.

The console's line finder (common.find_lines(img, "grad"), the twin of
runtime/vocr_lines.c) misses some lines of hard styles, and the overlay's legacy
finder only sees bright pixels. render.py's default finder is an ideal one:

find_lines_mask(ink)
    The overlay's projection rules (rows with >= 2 ink pixels form runs, a gap of
    3 rows ends a run, runs under 8 rows are ignored, a run over 88 rows or more
    than 8 lines rejects the region, boxes padded by 4 px and 5 px below), plus
    the research finder's furigana rule (runs under 0.55x the median run height
    are dropped). It runs on the renderer's exact text-ink mask instead of on
    bright pixels, so it behaves like an ideal finder of either polarity: the
    crop geometry matches what the overlay hands the recognizer for bright text,
    and the same geometry is used for dark, outlined and coloured text.
"""
import numpy as np

MAX_LINES = 8
TOO_TALL = 88      # the legacy finder rejects a line run taller than MAX_LINE_H - 8 = 88 rows


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

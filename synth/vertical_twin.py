"""Vertical text columns: the column finder, the column rotation the dedicated recognizer reads, and the re-tiling
arm (issue #99, docs/vertical.md).

find_columns(img) is a bit-exact twin of runtime/vocr_vertical.c's vocr_find_columns: the GRAD finder of
runtime/vocr_lines.c (synth/common.find_lines, "grad") run on the region's transpose, so every step that finds a
text row finds a text column instead:
  1. grey and the 3x3 local contrast (max - min), the threshold T = max(40, 2/5 of the 99th percentile): the
     contrast is symmetric, so this is GRAD's on-mask, transposed
  2. per column, on-pixels at most 10 px apart join into vertical runs and runs of at least 16 px are kept;
     prof(x) = the kept run length of column x
  3. columns with prof >= max(16, max(prof) / 16) are active; active columns with gaps of up to 2 form bands,
     bands under 8 columns wide are dropped, and a bridged gap holding an empty column between two runs of at least
     8 active columns ends the band (columns set 1-2 px apart stay separate)
  4. per band: the row cluster with the most kept pixels (clusters split at gaps over twice the band's width) is
     the column; it is lengthened over on-pixels at most max(4, band width) rows away (punctuation and small kana at
     its ends) and widened by up to half the band's width (never past half the gap to the neighbouring band) while
     columns hold 2+ on-pixels
  5. columns narrower than 55% of the median column width are dropped (furigana, rules, icons)
  6. NEW for columns: a box wider than 5/4 of its height is horizontal text, not a column, and is dropped
  7. boxes are padded by 3 px and get GRAD's polarity flag (vocr_line_dark)
  8. order: right to left (a column's right edge, larger first; on ties the upper one first): the reading order
At most 16 columns (VOCR_LINES_MAX); more set the truncated flag.

rotate_ccw(grey) turns a column crop 90 degrees anticlockwise (np.rot90 k=1): the column's top becomes the strip's
left end, its right side the strip's top. The dedicated recognizer VOCR5-V reads that strip as a horizontal line.

retile(grey, mode) cuts a column crop into character cells by its row profile and lays them out left to right
(characters stay upright) for the horizontal VOCR5 recognizers: mode "a0" as they are, "a1" with vertical forms
undone (a cell holding one tall thin stroke, a vertical long-vowel bar or dash, is turned 90 degrees clockwise...
back to horizontal; small marks in a cell's upper right, the vertical positions of 、。 and small kana, move to its
lower left). retile is a measured research arm; the console runs it only if the protocol selects it.
"""
import numpy as np

from common import (
    FURI_PCT,
    GAPC,
    LINES_MAX,
    LMIN,
    MERGE,
    MIN_H,
    PAD,
    ROW_DIV,
    T_MIN,
    _first_cum,
    _widen,
    contrast3,
    grey_of,
    kept_runs,
    line_dark,
)

ASPECT_NUM, ASPECT_DEN = 5, 4      # a box wider than 5/4 of its height is not a column


VALLEY_DIV = 4       # a band splits where a column's on-pixels fall under 1/VALLEY_DIV of the band's median
DENSE_DIV = 8        # lengthening and the column's own rows need max(2, band width / DENSE_DIV) on-pixels


def _split_valleys(oc, a, b):
    """Splits band [a, b) (positions across the columns) at runs of columns whose on-pixel count is under
    max(1, median / VALLEY_DIV), keeping pieces at least MIN_H wide: [(a', b')]."""
    seg = oc[a:b]
    med = int(np.sort(seg)[len(seg) // 2])
    th = max(1, med // VALLEY_DIV)
    low = seg < th
    out, s0 = [], None
    for i in range(len(seg) + 1):
        hi = i < len(seg) and not low[i]
        if hi and s0 is None:
            s0 = i
        elif not hi and s0 is not None:
            if i - s0 >= MIN_H:
                out.append((a + s0, a + i))
            s0 = None
    return out or [(a, b)]


def find_columns(img, variant=0):
    """img: HxW grey or HxWx3/4 RGB(A) uint8 region. Returns (boxes, dark, truncated): boxes [(x, y, w, h)] in
    reading order (right to left), dark[i] 1 when column i's text is darker than its background.
    variant: runtime/vocr_vertical.h's flags. 0: GRAD transposed, as described above; bit 0 (VOCR_COLUMNS_VALLEYS,
    the selected P1): also split bands at on-pixel valleys and lengthen / cluster a column only over rows dense
    enough to be text; bit 1 (VOCR_COLUMNS_NO_DARK): every dark flag 0 (the caller computes line_dark itself)."""
    g = grey_of(np.asarray(img))
    H, W = g.shape
    if not H or not W:
        return [], [], False
    c = contrast3(g)
    cum = np.cumsum(np.bincount(c.ravel(), minlength=256)).astype(np.int64)
    n = H * W
    p99 = _first_cum(cum * 100, n * 99)
    T = max(T_MIN, 2 * p99 // 5)
    on = np.ascontiguousarray((c >= T).T)          # transposed: row x of `on` is column x of the region
    Ht, Wt = on.shape                               # Ht = W (columns), Wt = H (rows)
    kept = kept_runs(on)
    prof = kept.sum(axis=1)
    th = max(LMIN, int(prof.max()) // ROW_DIV)
    act = prof >= th
    bands, y = [], 0
    while y < Ht:
        if not act[y]:
            y += 1
            continue
        s = rs = last = y
        y += 1
        while y < Ht and (act[y] or y - last <= MERGE):
            if act[y]:
                if y - last > 1:
                    k = y
                    while k < Ht and act[k]:
                        k += 1
                    if not prof[last + 1:y].all() and last + 1 - rs >= MIN_H and k - y >= MIN_H:
                        bands.append((s, last + 1))
                        s = y
                    rs = y
                last = y
            y += 1
        if last + 1 - s >= MIN_H:
            bands.append((s, last + 1))
    no_dark = bool(variant & 2)
    variant &= 1
    if variant:
        oc_all = on.sum(axis=1)
        bands = [q for a, b in bands for q in _split_valleys(oc_all, a, b)]
    cands = []
    for i, (a, b) in enumerate(bands):
        bh = b - a
        kc = kept[a:b].sum(axis=0)
        if variant:
            kc = np.where(kc >= max(2, bh // DENSE_DIV), kc, 0)
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
        if variant:
            oc = np.where(oc >= max(2, bh // DENSE_DIV), oc, 0)
        x1, x2 = _widen(oc, x1, x2, max(4, bh))
        up = min(bh // 2, (a - bands[i - 1][1]) // 2) if i else min(bh // 2, a)
        dn = min(bh // 2, (bands[i + 1][0] - b) // 2) if i + 1 < len(bands) else min(bh // 2, Ht - b)
        rows_on = on[:, x1:x2].sum(axis=1)
        top, bot = a, b
        while a - top < up and rows_on[top - 1] >= 2:
            top -= 1
        while bot - b < dn and rows_on[bot] >= 2:
            bot += 1
        cands.append((x1, top, x2, bot))           # transposed coordinates: x along the column, y across
    out = []
    if cands:
        med = sorted(bot - top for _, top, _, bot in cands)[len(cands) // 2]
        for x1, top, x2, bot in cands:
            if 100 * (bot - top) < FURI_PCT * med:
                continue
            # back to region coordinates: the column spans rows [x1, x2) and columns [top, bot)
            cw, ch = bot - top, x2 - x1
            if ASPECT_DEN * cw > ASPECT_NUM * ch:
                continue
            lx, ly = max(0, top - PAD), max(0, x1 - PAD)
            ex, ey = min(W, bot + PAD), min(H, x2 + PAD)
            out.append((lx, ly, ex - lx, ey - ly))
    # reading order: right edge descending, then top ascending
    order = sorted(range(len(out)), key=lambda i: (-(out[i][0] + out[i][2]), out[i][1]))
    boxes, dark, truncated = [], [], False
    for i in order:
        if len(boxes) == LINES_MAX:
            truncated = True
            break
        boxes.append(out[i])
        dark.append(0 if no_dark else line_dark(g, out[i]))
    return boxes, dark, truncated


def rotate_ccw(grey):
    """The column crop turned 90 degrees anticlockwise (out[i][j] = grey[j][w - 1 - i])."""
    return np.ascontiguousarray(np.rot90(np.asarray(grey), 1))


# ----------------------------------------------------------------------------- re-tiling (research arm)

def _ink(g):
    c = contrast3(g)
    cum = np.cumsum(np.bincount(c.ravel(), minlength=256)).astype(np.int64)
    p99 = _first_cum(cum * 100, g.size * 99)
    return c >= max(T_MIN, 2 * p99 // 5)


def cells(g):
    """Character cells [(y0, y1)] of a column crop: runs of rows holding ink, small ones merged with a neighbour
    while the merge stays under 1.15 column widths, tall ones split into equal parts of about one width."""
    on = _ink(g)
    H, W = g.shape
    rows = on.sum(axis=1) > 0
    runs, s = [], None
    for y in range(H + 1):
        r = y < H and rows[y]
        if r and s is None:
            s = y
        elif not r and s is not None:
            runs.append([s, y])
            s = None
    if not runs:
        return []
    em = max(4, W - 2 * 3)
    merged = [runs[0]]
    for a, b in runs[1:]:
        pa, pb = merged[-1]
        if (b - pa) <= 1.15 * em and ((pb - pa) < 0.55 * em or (b - a) < 0.55 * em):
            merged[-1] = [pa, b]
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        h = b - a
        k = int(round(h / em)) if h > 1.4 * em else 1
        for i in range(k):
            out.append((a + h * i // k, a + h * (i + 1) // k))
    return out


def _square(cell, W, bg):
    """A cell (h rows x W columns) in a W x W square: centred vertically, cropped to its middle W rows if taller."""
    h = cell.shape[0]
    if h >= W:
        o = (h - W) // 2
        return cell[o:o + W].copy()
    sq = np.full((W, W), bg, np.uint8)
    o = (W - h) // 2
    sq[o:o + h] = cell
    return sq


def retile(g, mode="a0", gap=2):
    """The column's character cells as W x W squares laid out left to right (W = the crop's width), separated by
    `gap` columns of the background grey (the median of the crop's border): characters stay upright, so a
    horizontal recognizer reads the strip. mode "a1" also undoes the vertical forms a cell can hold:
      - one tall thin stroke (ink at least 2.2 times as tall as wide and under 0.35 W wide: the long-vowel bar and
        dashes as set vertically) is turned 90 degrees, back to a horizontal bar;
      - a small mark (ink under 0.45 W both ways) whose ink sits right of centre in the upper half (、。 and small
        kana as set vertically) is moved to the square's lower left, where horizontal text sets it."""
    g = np.asarray(g)
    H, W = g.shape
    cs = cells(g)
    if not cs:
        return g.copy()
    bg = int(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])))
    on = _ink(g)
    blocks = [np.full((W, gap), bg, np.uint8)]
    for a, b in cs:
        a0, b0 = max(0, a - 1), min(H, b + 1)
        sq = _square(g[a0:b0], W, bg)
        if mode == "a1":
            m = _square(on[a0:b0].astype(np.uint8), W, 0) > 0
            ys, xs = np.nonzero(m)
            if len(ys):
                ih, iw = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
                if ih > 2.2 * iw and iw < 0.35 * W:
                    sq = np.ascontiguousarray(np.rot90(sq, -1))
                elif ih < 0.45 * W and iw < 0.45 * W and xs.mean() > 0.55 * W and ys.mean() < 0.5 * W:
                    y0c, x0c = int(ys.min()), int(xs.min())
                    patch = sq[y0c:y0c + ih, x0c:x0c + iw].copy()
                    sq = np.full((W, W), bg, np.uint8)
                    py, px = max(0, W - ih - max(1, W // 10)), min(max(1, W // 10), W - iw)
                    sq[py:py + ih, px:px + iw] = patch
        blocks.append(sq)
        blocks.append(np.full((W, gap), bg, np.uint8))
    return np.ascontiguousarray(np.concatenate(blocks, axis=1))

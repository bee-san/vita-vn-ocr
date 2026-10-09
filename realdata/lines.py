"""Line geometry for real screenshots: text detection, rows, ink boxes and crops.

A screenshot is read at its source resolution. PP-OCRv5 mobile det finds text boxes;
boxes on the same text row are merged (a dialogue line split at a gap is one line for
the console's line finder too). Each row's ink box is then refined with a
polarity-agnostic gradient-energy profile (the same signal the research's gradient line
finder uses), so dark-on-light, outlined and coloured text are all handled.

Two crops come out of every ink box:

  voter view   the ink box plus 20% of its height on every side, at source resolution.
               The OCR voters read this; for PC screenshots it is the sharp original.
  train crop   the ink box at the target resolution (Vita 960x544 or PSP 480x272) padded
               2 px left and right, 3 px above and 4 px below: on 2 x 2,000 accepted lines the
               console's grad finder (synth/common.find_lines_ex, runtime/vocr_lines.c)
               then gives boxes whose median edge difference is 0 px on every side (+1 at
               x0 on one of two 2,000-line samples), though only 39-72% of edges are within
               1 px (p10-p90 about -25..+18 px at x0, -18..+5 at x1; check_finder.py). PC
               screenshots are first scaled to fit the target screen (cv2.INTER_AREA), so
               their glyphs end up at console size.
"""
import math

import cv2
import numpy as np

TARGETS = {"vita": (960, 544), "psp": (480, 272)}
# Ink height limits per target, in target pixels: below the minimum a glyph is not
# legible at that resolution (PSP bitmap fonts are 12-16 px); above the maximum it is a
# title or logo, not dialogue.
MIN_INK = {"vita": 14, "psp": 10}
MAX_INK = 72
PAD = (2, 3, 2, 4)                      # left, top, right, bottom


def make_detector(threads):
    from rapidocr import ModelType, OCRVersion, RapidOCR
    return RapidOCR(params={"Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
                            "Global.log_level": "error",
                            "EngineConfig.onnxruntime.intra_op_num_threads": threads})


def detect(det, bgr):
    """Axis-aligned boxes [x0, y0, x1, y1] (floats, source pixels) of near-horizontal text."""
    # rapidocr remembers use_det/use_rec between calls: always pass all three.
    r = det(bgr, use_det=True, use_cls=False, use_rec=False)
    quads = getattr(r, "boxes", None)
    out = []
    for q in (quads if quads is not None else []):
        q = np.asarray(q, np.float32)
        top = q[1] - q[0]
        if abs(math.degrees(math.atan2(float(top[1]), float(top[0]) + 1e-6))) > 4:
            continue                                     # slanted decoration text
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        if x1 - x0 < 0.9 * (y1 - y0):                    # vertical text (tategaki)
            continue
        out.append([float(x0), float(y0), float(x1), float(y1)])
    return out


def merge_rows(boxes):
    """Union boxes that sit on the same text row: similar height, mostly overlapping
    vertically, and at most 0.8 line heights apart horizontally."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        boxes.sort(key=lambda b: (b[1] + b[3], b[0]))
        out = []
        for b in boxes:
            for a in out:
                ha, hb = a[3] - a[1], b[3] - b[1]
                ov = min(a[3], b[3]) - max(a[1], b[1])
                gap = max(a[0], b[0]) - min(a[2], b[2])
                if ov >= 0.6 * min(ha, hb) and max(ha, hb) <= 1.4 * min(ha, hb) \
                        and gap <= 0.8 * max(ha, hb):
                    a[:] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def energy(rgb):
    """|dx| + |dy| per pixel, max over the colour channels (so colour-on-colour text of
    equal luma still shows up)."""
    f = rgb.astype(np.int16)
    dx = np.zeros(f.shape[:2], np.int16)
    dy = np.zeros(f.shape[:2], np.int16)
    dx[:, :-1] = np.abs(np.diff(f, axis=1)).max(axis=2)
    dy[:-1] = np.abs(np.diff(f, axis=0)).max(axis=2)
    return (dx + dy).astype(np.float32)


def runs(mask, max_gap):
    """[(start, end_exclusive)] of True runs, merging gaps shorter than max_gap."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, s, last = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - last > max_gap:
            out.append((s, last + 1))
            s = i
        last = i
    out.append((s, last + 1))
    return out


SPLIT_SIDE, SPLIT_DEPTH, SPLIT_KEEP = 3, 0.5, 0.5


def text_rows(prof, mid, min_rows):
    """The text row through row `mid` of a per-row energy profile: the run above 15% of
    the maximum (gaps of up to two rows bridged) that holds `mid`, split at a valley
    deeper than half of the lower of the two peaks beside it when the far part runs into
    the profile's edge (a neighbouring line with tight leading), keeping mid's side if
    that is at least min_rows tall (glyphs like 二 or 八 have valleys of their own)."""
    pmax = float(prof.max())
    hit = [r for r in runs(prof >= 0.15 * pmax, 2) if r[0] <= mid < r[1]]
    if not hit:
        return None
    s, e = (int(v) for v in hit[0])
    n = len(prof)
    while e - s >= 6:
        best = None
        for r in range(s + SPLIT_SIDE, e - SPLIT_SIDE):
            # only a part that runs into the window's edge can be another line; a part that
            # ends inside the window is this line's own strokes (short words like 充実, 情報)
            if not ((r < mid and s == 0) or (r > mid and e == n)):
                continue
            side = min(float(prof[s:r].max()), float(prof[r + 1:e].max()))
            if prof[r] < SPLIT_DEPTH * side and (best is None or prof[r] < prof[best]):
                best = r
        if best is None or best == mid:
            break
        ns, ne = (s, best) if mid < best else (best + 1, e)
        if ne - ns < min_rows:
            break
        s, e = ns, ne
    return s, e


def ink_box(rgb, box):
    """Refine a detector row box to the ink it holds. Returns (x0, y0, x1, y1) ints in
    source pixels (exclusive ends), or None when no clean text row is found."""
    H, W = rgb.shape[:2]
    bx0, by0, bx1, by1 = box
    bh = by1 - by0
    wx0, wx1 = int(max(0, math.floor(bx0 - 0.6 * bh))), int(min(W, math.ceil(bx1 + 0.6 * bh)))
    wy0, wy1 = int(max(0, math.floor(by0 - 0.25 * bh))), int(min(H, math.ceil(by1 + 0.25 * bh)))
    if wx1 - wx0 < 4 or wy1 - wy0 < 4:
        return None
    e = energy(rgb[wy0:wy1, wx0:wx1])
    # rows: energy over the detector's columns; the text row is the run through the box middle
    cx0, cx1 = int(max(0, bx0 - wx0)), int(min(wx1 - wx0, bx1 - wx0))
    prof = e[:, cx0:cx1].mean(axis=1)
    if prof.max() <= 0:
        return None
    ry = text_rows(prof, int(round((by0 + by1) / 2 - wy0)), int(math.ceil(SPLIT_KEEP * bh)))
    if ry is None:
        return None
    ry0, ry1 = ry
    if not 0.5 * bh <= ry1 - ry0 <= 1.25 * bh:          # border or texture merged in
        return None
    # columns: energy within the text row, relative to the text's own level; keep the
    # spans that touch the detector box (gaps between glyphs are bridged)
    col = e[ry0:ry1].mean(axis=0)
    floor = max(0.1 * float(np.percentile(col[cx0:cx1], 90)), 1.0)
    spans = [s for s in runs(col >= floor, int(0.8 * (ry1 - ry0)) + 1) if s[1] > cx0 and s[0] < cx1]
    if not spans:
        return None
    rx0, rx1 = spans[0][0], spans[-1][1]
    if rx1 - rx0 < 0.5 * (ry1 - ry0):
        return None
    x0, y0, x1, y1 = wx0 + rx0, wy0 + ry0, wx0 + rx1, wy0 + ry1
    if x0 <= 0 or y0 <= 0 or x1 >= W or y1 >= H:     # text may continue off screen
        return None
    return int(x0), int(y0), int(x1), int(y1)


def target_scale(src_w, src_h, target):
    tw, th = TARGETS[target]
    return min(tw / src_w, th / src_h)


def scale_image(rgb, s):
    if abs(s - 1.0) < 1e-6:
        return rgb
    w, h = rgb.shape[1], rgb.shape[0]
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def train_box(ink, s, img_w, img_h):
    """Ink box in source pixels -> padded crop box [x0, y0, x1, y1) at scale s."""
    x0, y0 = int(math.floor(ink[0] * s)), int(math.floor(ink[1] * s))
    x1, y1 = int(math.ceil(ink[2] * s)), int(math.ceil(ink[3] * s))
    return (max(0, x0 - PAD[0]), max(0, y0 - PAD[1]), min(img_w, x1 + PAD[2]), min(img_h, y1 + PAD[3]))


def voter_box(ink, img_w, img_h, rgb=None):
    """The ink box plus 20% of its height on every side. With the image, the top and
    bottom margins stop halfway to a neighbouring text row, so the readers don't see
    half of the line above or below (tight leading, PSP fonts)."""
    m = max(3, int(round(0.2 * (ink[3] - ink[1]))))
    top = bottom = m
    if rgb is not None:
        top, bottom = clear_margins(rgb, ink, m)
    return (max(0, ink[0] - m), max(0, ink[1] - top), min(img_w, ink[2] + m), min(img_h, ink[3] + bottom))


def clear_margins(rgb, ink, m):
    """(top, bottom) margins of at most m rows. Moving away from the ink box, the per-row
    mean |dx| falls into the gap between lines; if it rises again within m rows (a
    neighbouring line), the margin ends at the lowest row before the rise."""
    x0, y0, x1, y1 = ink
    a, b = max(0, y0 - m), min(rgb.shape[0], y1 + m)
    f = rgb[a:b, x0:x1].astype(np.int16)
    if f.shape[1] < 2:
        return m, m
    dx = np.abs(np.diff(f, axis=1)).max(axis=2).mean(axis=1)
    ref = float(np.median(dx[y0 - a:y1 - a]))
    out = []
    for rows in (dx[:y0 - a][::-1], dx[y1 - a:]):
        lo, lo_i, cut = float("inf"), 0, len(rows)
        for i, v in enumerate(rows):
            if v < lo:
                lo, lo_i = float(v), i
            elif v >= lo + 0.25 * (ref - lo) and v >= 0.2 * ref:
                cut = lo_i + 1
                break
        out.append(min(m, max(1, cut)))
    return tuple(out)


def polarity(rgb):
    """'light-on-dark' or 'dark-on-light' for a line crop: Otsu split of the grey crop,
    the minority class is the ink."""
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    t, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = float((m > 0).mean())
    return "light-on-dark" if bright < 0.5 else "dark-on-light"

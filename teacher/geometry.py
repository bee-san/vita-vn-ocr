"""Line geometry for the teacher: MeikiOCR's line boxes -> ink boxes, voter views, train crops.

The conventions are eval's (eval/tools/assemble.py), so a training crop looks like a
benchmark crop:
  * detector boxes that are pieces of one text row are merged (vertical overlap >= 0.6 of
    the smaller height, gap <= 1.2 line heights, heights within 0.6x), as the console's row
    projection would see one line there;
  * each box is refined to the ink it holds (refine(): contrast against the box's border
    ring, near-white and near-black masks; the smallest valid box wins);
  * small boxes sitting just above a line are its furigana: not labelled, but inside the
    line's crop, as on screen;
  * the train crop is the ink box (with furigana) plus PAD px (4 on Vita, 2 on PSP).
The voters read a wider view (VIEW_X line heights left and right, VIEW_Y above and below,
stopping short of a neighbouring row), so a line MeikiOCR cut short reads longer for them
and the vote fails rather than labelling half a line.
"""
import math

import numpy as np

TARGETS = {"vita": (960, 544), "psp": (480, 272)}
PAD = {"vita": 4, "psp": 2}
MIN_INK = {"vita": 10, "psp": 10}       # ink height in target px; below this a line is unreadable
MAX_INK = 72                            # above: titles, logos
VIEW_X, VIEW_Y = 0.6, 0.25


def to_gray(rgb):
    """Integer luma, identical to synth/common.to_gray and the C side."""
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def vov(a, b):
    """Vertical overlap of two x0,y0,x1,y1 boxes as a share of the smaller height."""
    inter = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, inter) / max(1, min(a[3] - a[1], b[3] - b[1]))


def merge_rows(lines):
    """Merge MeikiOCR lines that are pieces of one horizontal text row (eval's rule).
    Each line: dict with det [x0, y0, x1, y1], det_conf, text, chars. Texts and characters
    are joined left to right; the merged line keeps the lowest detector score."""
    lines = [dict(ln, parts=[ln["det"]]) for ln in lines]
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i]["det"], lines[j]["det"]
                h = max(a[3] - a[1], b[3] - b[1])
                gap = max(a[0], b[0]) - min(a[2], b[2])
                hr = min(a[3] - a[1], b[3] - b[1]) / max(1, h)
                if vov(a, b) >= 0.6 and gap <= 1.2 * h and hr >= 0.6:
                    left, right = (lines[i], lines[j]) if a[0] <= b[0] else (lines[j], lines[i])
                    lines[i] = {"det": [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                                "det_conf": min(lines[i]["det_conf"], lines[j]["det_conf"]),
                                "vertical": False, "text": left["text"] + right["text"],
                                "chars": left["chars"] + right["chars"],
                                "parts": left["parts"] + right["parts"]}
                    del lines[j]
                    changed = True
                    break
            if changed:
                break
    return lines


def _ink_box(m, bbox, win):
    """Box of the longest run of ink rows of mask m that touches the det box rows."""
    x0, y0, x1, y1 = bbox
    wx0, wy0 = win
    rows = np.flatnonzero(m.sum(1) >= max(2, 0.01 * m.shape[1]))
    if not len(rows):
        return None
    runs, s, last = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - last > 3:
            runs.append((s, last))
            s = r
        last = r
    runs.append((s, last))
    by0, by1 = y0 - wy0, y1 - wy0
    runs = [r for r in runs if r[1] >= by0 and r[0] < by1]
    if not runs:
        return None
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    cols = np.flatnonzero(m[r0:r1 + 1].sum(0) >= 1)
    if not len(cols):
        return None
    ib = [wx0 + int(cols[0]), wy0 + int(r0), wx0 + int(cols[-1]) + 1, wy0 + int(r1) + 1]
    if ib[3] - ib[1] < 0.45 * (y1 - y0) or ib[2] - ib[0] < 0.75 * (x1 - x0):
        return None
    if m[r0:r1 + 1, cols[0]:cols[-1] + 1].mean() < 0.03:
        return None
    return ib


def refine(rgb, bbox):
    """Ink extent of a detector box (eval/tools/assemble.py refine()). Returns (box, hint):
    hint has bright_frac (share of ink brighter than the background) for polarity."""
    import cv2
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    wx0, wy0, wx1, wy1 = max(0, x0 - 3), max(0, y0 - 3), min(W, x1 + 3), min(H, y1 + 3)
    win = rgb[wy0:wy1, wx0:wx1]
    g = to_gray(win).astype(np.float32)
    if g.shape[0] < 6 or g.shape[1] < 6:
        return list(bbox), {}
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = float(np.median(ring))
    d = np.abs(g - bg)
    t = max(28.0, 0.4 * float(np.percentile(d, 99)))
    k = np.ones((3, 3), np.uint8)
    masks = [d > t,
             cv2.dilate((win.min(2) >= 200).astype(np.uint8), k).astype(bool),
             cv2.dilate((win.max(2) <= 70).astype(np.uint8), k).astype(bool)]
    boxes = [b for b in (_ink_box(m, bbox, (wx0, wy0)) for m in masks) if b is not None]
    if not boxes:
        return list(bbox), {}
    ib = min(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    gm = g[ib[1] - wy0:ib[3] - wy0, ib[0] - wx0:ib[2] - wx0]
    mm = masks[0][ib[1] - wy0:ib[3] - wy0, ib[0] - wx0:ib[2] - wx0]
    hint = {}
    if mm.sum() >= 10:
        hint["bright_frac"] = round(float(((gm > bg) & mm).sum() / mm.sum()), 3)
        hint["bg"] = round(bg, 1)
    return ib, hint


def attach_furigana(lines):
    """Small lines directly above a bigger one become its furigana (eval's rule): they are
    not labelled, and their box joins the host's crop box ("crop")."""
    if not lines:
        return lines
    hs = sorted(ln["ink"][3] - ln["ink"][1] for ln in lines)
    med = hs[len(hs) // 2]
    keep = []
    for b in lines:
        x0, y0, x1, y1 = b["ink"]
        h = y1 - y0
        host = None
        if h <= 0.65 * med:
            for o in lines:
                if o is b:
                    continue
                ox0, oy0, ox1, oy1 = o["ink"]
                oh = oy1 - oy0
                if h <= 0.65 * oh and 0 <= oy0 - y1 <= 0.5 * oh and x0 >= ox0 - 4 and x1 <= ox1 + 4:
                    host = o
                    break
        if host is None:
            keep.append(b)
        else:
            host.setdefault("furi", []).append(b["ink"])
            host.setdefault("furi_text", []).append(b.get("text", ""))
    for b in keep:
        c = list(b["ink"])
        for f in b.get("furi", []):
            c = [min(c[0], f[0]), min(c[1], f[1]), max(c[2], f[2]), max(c[3], f[3])]
        b["crop"] = c
    return keep


def voter_view(ink, others, W, H):
    """The ink box widened by VIEW_X line heights left/right and VIEW_Y above/below; the
    vertical margins stop halfway to any other line that overlaps it horizontally."""
    x0, y0, x1, y1 = ink
    h = y1 - y0
    mx, my = max(3, int(round(VIEW_X * h))), max(2, int(round(VIEW_Y * h)))
    top, bot = y0 - my, y1 + my
    for o in others:
        if o is ink or o[2] <= x0 - mx or o[0] >= x1 + mx:
            continue
        if o[3] <= y0:
            top = max(top, (o[3] + y0) // 2)
        elif o[1] >= y1:
            bot = min(bot, (o[1] + y1 + 1) // 2)
    left, right = x0 - mx, x1 + mx
    for o in others:                       # nor run into a neighbour on the same row
        if o is ink or vov(o, ink) < 0.5:
            continue
        if o[2] <= x0:
            left = max(left, (o[2] + x0) // 2)
        elif o[0] >= x1:
            right = min(right, (o[0] + x1 + 1) // 2)
    return [max(0, left), max(0, min(top, y0)), min(W, right), min(H, max(bot, y1))]


def target_scale(src_w, src_h, target):
    tw, th = TARGETS[target]
    return min(tw / src_w, th / src_h)


def targets_of(kind, dims, ink):
    """Target screens a line is legible on: native shots only their own; PC shots any
    target whose scaled ink height is within [MIN_INK, MAX_INK]."""
    ih = ink[3] - ink[1]
    if kind in TARGETS:
        return [kind] if MIN_INK[kind] <= ih <= MAX_INK else []
    return [t for t in TARGETS if MIN_INK[t] <= ih * target_scale(dims[0], dims[1], t) <= MAX_INK]


def scale_image(rgb, s):
    import cv2
    if abs(s - 1.0) < 1e-6:
        return rgb
    w, h = rgb.shape[1], rgb.shape[0]
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def crop_box(box, s, pad, img_w, img_h):
    """A source-pixel box at scale s, padded: [x0, y0, x1, y1) in the scaled image."""
    x0, y0 = int(math.floor(box[0] * s)), int(math.floor(box[1] * s))
    x1, y1 = int(math.ceil(box[2] * s)), int(math.ceil(box[3] * s))
    return [max(0, x0 - pad), max(0, y0 - pad), min(img_w, x1 + pad), min(img_h, y1 + pad)]


def polarity(hint):
    f = hint.get("bright_frac")
    if f is None:
        return None
    return "bright-on-dark" if f >= 0.5 else "dark-on-light"


def lines_of(meiki_lines, rgb):
    """Screenshot lines from MeikiOCR's output: horizontal rows merged, ink boxes, furigana
    attached, voter views. Vertical lines are dropped (the console reads horizontal text)."""
    H, W = rgb.shape[:2]
    rows = merge_rows([ln for ln in meiki_lines if not ln["vertical"] and ln["text"]])
    for r in rows:
        r["ink"], r["hint"] = refine(rgb, r["det"])
    rows = attach_furigana(rows)
    boxes = [r["ink"] for r in rows]
    for r in rows:
        r["view"] = voter_view(r["ink"], [b for b in boxes if b is not r["ink"]], W, H)
        r["edge"] = r["ink"][0] <= 1 or r["ink"][1] <= 1 or r["ink"][2] >= W - 1 or r["ink"][3] >= H - 1
    rows.sort(key=lambda r: (r["ink"][1], r["ink"][0]))
    return rows

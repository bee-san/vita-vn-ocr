"""Line geometry of the devnative cohort: candidate lines, dialogue regions, the console's GRAD
finder on each region, finder-to-line matching, the two crop styles and the console polarity.

Line boxes come from MeikiOCR's detector through teacher/geometry.py (rows merged, each box
refined to its ink, furigana attached), the conventions eval used to build vita-test and
psp-test. From a screenshot's kept lines this module derives:
  * regions: lines that sit together (vertical gap <= 1.6 line heights) and their union plus
    12 px (Vita) / 6 px (PSP), the dialogue box a player would draw (eval/tools/manifest.py).
    devnative's GRAD crops grouped the raw reads lines, whose "crop" is read.py's padded box, so
    their regions are 16 / 8 px around the text and the gap is measured between padded boxes;
  * the ink crop: the line's ink box with its furigana, plus 4 px (Vita) / 2 px (PSP), as
    teacher and eval crop (eval_line(); on a raw reads line crop_box() pads twice, which is the
    contact sheets' review frame);
  * the GRAD crop: what the console's line finder (synth/common.find_lines_ex, the bit-exact
    twin of runtime/vocr_lines.c) returns for the region, matched one-to-one to the lines
    (eval/finders.match: vertical overlap >= 0.5 of the smaller height, any horizontal
    overlap). A GRAD crop takes its line's label only when it holds the whole main line
    (status "ok"): not "partial" (an edge of the ink outside the box) and not "mixed" (a
    quarter or more of another detected text box inside it). Lines with no finder box are
    "missed";
  * the console polarity: a crop whose text common.line_dark calls dark is stored inverted
    (255 - grey), as teacher, realdata and the console do.
All boxes are [x0, y0, x1, y1) in screenshot pixels.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from common import find_lines_ex, line_dark, to_gray  # noqa: E402
from devnative import paths as P  # noqa: E402

GROUP_GAP = 1.6       # eval/tools/manifest.py groups(): vertical gap in line heights
MATCH = 0.5           # eval/finders.match threshold
MIXED = 0.25          # share of another text box inside a GRAD box that makes it mixed
MIN_INK, MAX_INK = 10, 72   # ink height at native size (teacher/geometry.py)


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def text_box(line):
    """The line's ink box with its furigana (eval's "box"), from line["crop"]. A raw reads.jsonl
    line holds read.py's padded crop there (ink + furigana + 4/2 px; the unpadded box is
    "crop_ink"), so on a raw line this is the padded box and crop_box() pads it again (the sheets'
    review frame, ink + furigana + 8/4 px). eval_line() gives eval's geometry."""
    return list(line.get("crop") or line["ink"])


def eval_line(line):
    """The line with "crop" = its unpadded ink + furigana box, so text_box() is eval's box and
    crop_box() / final_crop() pad once (ink + 4/2 px), as eval and teacher crop."""
    return dict(line, crop=list(line.get("crop_ink") or line["ink"]))


def crop_box(line, kind, W, H):
    """The ink crop: ink + furigana, plus PAD px, clipped to the screen."""
    x0, y0, x1, y1 = text_box(line)
    p = P.PAD[kind]
    return [max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)]


def ink_height(line):
    return line["ink"][3] - line["ink"][1]


def size_ok(line):
    return MIN_INK <= ink_height(line) <= MAX_INK


def regions(lines, kind, W, H):
    """[(region box, [indices into lines])]: eval's groups() over the lines' text boxes, top to
    bottom, each group's union plus the platform margin."""
    order = sorted(range(len(lines)), key=lambda i: (text_box(lines[i])[1], text_box(lines[i])[0]))
    groups = []
    for i in order:
        b = text_box(lines[i])
        if groups:
            g = groups[-1]
            bottom = max(text_box(lines[j])[3] for j in g)
            hmax = max(text_box(lines[j])[3] - text_box(lines[j])[1] for j in g + [i])
            if b[1] - bottom <= GROUP_GAP * hmax:
                g.append(i)
                continue
        groups.append([i])
    m = P.MARGIN[kind]
    out = []
    for g in groups:
        u = union([text_box(lines[j]) for j in g])
        out.append(([max(0, u[0] - m), max(0, u[1] - m), min(W, u[2] + m), min(H, u[3] + m)], sorted(g)))
    return out


def grad_lines(rgb, region):
    """The console finder on one region: [(box in screen px, dark)], top to bottom."""
    x0, y0, x1, y1 = region
    boxes, dark, _ = find_lines_ex(np.ascontiguousarray(rgb[y0:y1, x0:x1]), "grad")
    return [([x0 + x, y0 + y, x0 + x + w, y0 + y + h], int(d)) for (x, y, w, h), d in zip(boxes, dark)]


def overlap(d, g):
    """eval/finders.overlap on [x0, y0, x1, y1) boxes: 0 without horizontal overlap, else the
    vertical overlap as a share of the smaller height."""
    if min(d[2], g[2]) <= max(d[0], g[0]):
        return 0.0
    ov = min(d[3], g[3]) - max(d[1], g[1])
    return max(0.0, ov / max(1, min(d[3] - d[1], g[3] - g[1])))


def match(dets, gts, thresh=MATCH):
    """eval/finders.match: one-to-one greedy by overlap. Returns {gt index: det index}."""
    cand = sorted(((overlap(d, g), i, j) for i, d in enumerate(dets) for j, g in enumerate(gts)), reverse=True)
    used_d, used_g, out = set(), set(), {}
    for ov, i, j in cand:
        if ov < thresh or i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        out[j] = i
    return out


def own_boxes(line):
    """Detector boxes that are this line or its furigana."""
    return [list(b) for b in line.get("parts") or []] + [list(b) for b in line.get("furi") or []]


def others_of(line, dets):
    """Every other detected text box on the screen (any orientation or language)."""
    mine = own_boxes(line)
    out = []
    for d in dets:
        b = d["det"] if isinstance(d, dict) else d
        if area(b) and any(area(inter(b, m)) > 0.5 * area(b) for m in mine):
            continue
        out.append(list(b))
    return out


def grad_status(gbox, line, others):
    """"ok" when the GRAD box holds the whole main line and no other text, else why not."""
    x0, y0, x1, y1 = line["ink"]
    t = max(1, round(0.12 * (y1 - y0)))
    if gbox[0] > x0 + 1 or gbox[2] < x1 - 1 or gbox[1] > y0 + t or gbox[3] < y1 - t:
        return "partial"
    for o in others:
        if area(o) and area(inter(o, gbox)) >= MIXED * area(o):
            return "mixed"
    return "ok"


def grad_assign(rgb, lines, dets, kind):
    """For a screenshot's kept lines: regions, finder boxes and one GRAD result per line:
    {"region": box, "box": box or None, "dark": 0/1, "status": ok|partial|mixed|missed}."""
    H, W = rgb.shape[:2]
    res = [None] * len(lines)
    for reg, members in regions(lines, kind, W, H):
        found = grad_lines(rgb, reg)
        gts = [text_box(lines[j]) for j in members]
        pairs = match([b for b, _ in found], gts)
        for jj, j in enumerate(members):
            if jj not in pairs:
                res[j] = {"region": reg, "box": None, "dark": None, "status": "missed", "found": len(found)}
                continue
            b, d = found[pairs[jj]]
            res[j] = {"region": reg, "box": b, "dark": d, "found": len(found),
                      "status": grad_status(b, lines[j], others_of(lines[j], dets))}
    return res


def cut(rgb, box, invert=True):
    """Grey crop (integer luma) of box, with the console's polarity flag. Returns
    (crop as stored, dark, inverted)."""
    x0, y0, x1, y1 = box
    g = to_gray(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
    dark = int(line_dark(g, (0, 0, g.shape[1], g.shape[0])))
    inverted = bool(dark and invert)
    return (255 - g if inverted else g), dark, inverted


def box_iou(a, b):
    i = area(inter(a, b))
    return i / max(1, area(a) + area(b) - i)


def label_chars(line, key_of, label):
    """MeikiOCR's character boxes (screen px) of the reading whose agreement key equals the
    label's: its own box first, then its reading of the view. None when neither matches."""
    k = key_of(label)
    for e in ("meiki", "meikiv"):
        r = (line.get("reads") or {}).get(e)
        if not r:
            continue
        chars = r[1] if e == "meiki" else r[3]
        if chars and key_of(r[0]) == k:
            return chars
    return None


def final_crop(line, kind, W, H, chars=None):
    """The ink crop widened left and right to the labelled characters' boxes (plus PAD), never
    beyond the line's view: teacher/build.py's crop_of rule, so every labelled glyph MeikiOCR
    located is inside the crop."""
    c = crop_box(line, kind, W, H)
    if not chars:
        return c
    p, v = P.PAD[kind], line.get("view") or c
    x0 = max(min(v[0], c[0]), min(c[0], min(ch[1] for ch in chars) - p))
    x1 = min(max(v[2], c[2]), max(c[2], max(ch[3] for ch in chars) + p))
    return [max(0, x0), c[1], min(W, x1), c[3]]


def ink_beyond(gray, line, crop):
    """(left, right) px of ink that continues the line's rows beyond `crop`: columns whose
    contrast against the local background exceeds the refine threshold, chained with gaps of
    at most 0.6 ink heights, searched within the line's view. A bracket or 。 the detector
    left out shows up here (so do plate borders and decoration)."""
    x0, y0, x1, y1 = line["ink"]
    h = max(1, y1 - y0)
    v = line.get("view") or crop
    H, W = gray.shape[:2]
    vx0, vx1 = max(0, min(v[0], crop[0])), min(W, max(v[2], crop[2]))
    wy0, wy1 = max(0, y0 - 2), min(H, y1 + 2)
    win = gray[wy0:wy1, vx0:vx1].astype(np.int32)
    if win.size == 0:
        return 0, 0
    ring = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]])
    d = np.abs(win - float(np.median(ring)))
    t = max(28.0, 0.4 * float(np.percentile(d, 99)))
    on = (d[y0 - wy0:y1 - wy0] > t).sum(0) >= 2
    gap = max(3, round(0.6 * h))
    last = crop[2] - 1 - vx0
    for c in range(crop[2] - vx0, len(on)):
        if on[c]:
            if c - last > gap:
                break
            last = c
    first = crop[0] - vx0
    for c in range(crop[0] - 1 - vx0, -1, -1):
        if on[c]:
            if first - c > gap:
                break
            first = c
    return max(0, crop[0] - (vx0 + first)), max(0, vx0 + last + 1 - crop[2])

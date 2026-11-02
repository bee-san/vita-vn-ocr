#!/usr/bin/env python3
"""textspot-desktop's exact line boxes (issue #173, eval/textspot_desktop_protocol.json amendment 1): the ink extent
of a hooked line inside its glyph cells.

Every verified line of the hooked pages carries the renderer's geometry: its box (the extent of its glyph quads) and
its cells (cell_boxes: one per character of the label, the glyph quads grouped by character; on image-row titles the
title's pitch grid from the first to the last ink column). Neither is tight to the ink: a cell is a glyph's whole
em box, so 「 sits in the right part of its cell and 。 in the lower left, and the box can also hold an indent space
or a cursor quad that is no character of the label. The repository's ink refinement (teacher/geometry.refine, which
synth/textspot6_pages.refined_line applies to the box) takes the smallest of three masks' boxes; on titles whose
punctuation or a coloured word is fainter than the other glyphs it drops a leading 「 or ――, or a trailing 。, 」 or
……, and on dithered or patterned boxes the plain contrast extent runs into the background. This function:

  1. c = the union of the line's cells (its box when it has none), clipped to the frame;
  2. a window of c grown by `grow` px; the background: the per-channel median of the window's one-pixel border ring;
  3. d = each pixel's largest per-channel absolute difference from the background;
  4. the threshold t = max(floor, k x the 99th percentile of d inside c, the 90th percentile of d on the ring +
     `noise_margin`): above the background's own texture (a dither or a pattern on the ring);
  5. the ink: pixels with d > t inside c grown by `spill` px, without the 4-connected components of fewer than
     `min_blob` pixels (dither and noise specks);
  6. the box: the first and last columns and rows holding ink, as an integer half-open [x0, y0, x1, y1] in frame px.
A line with no ink keeps c ("cells"). The along-line ends are each within a cell of c's (the first and last
characters' cells), the across extent within c's rows grown by `spill`.

The gap clip (#177; `build --gap-clip`, targets_v2; off by default, so targets_v1 rebuilds as it was): on titles whose
line pitch is smaller than a glyph's em box, c reaches into the next line and that line's glyph tops or bottoms count
as ink. With the clip, the rows searched for ink (step 5) and the "cells" box stop at half the gap to the nearest row
above and below on the same page that overlaps the line horizontally (gap_bounds, as eval/tsdesk_dense.regions bounds
a region; the rows: every other horizontal line of any kind and drop, ruby included, and every uncovered renderer
box), in the middle of the overlap where the cells overlap. Steps 2-4 (the background and the thresholds) are
unchanged, so a line no bound cuts keeps targets_v1's box exactly.

The cells fix (`build --cells-fix`, targets_v3 = targets_v2 + this; off by default): a horizontal line whose glyph
cells match its label's characters, and whose box leaves a non-space character's cell wholly outside (part of the
line cut: a trailing 。 or a leading 「, a faint or coloured word), is boxed again with three changes, each a cause
of such cuts: ring_noise's fix (a ring through the line's outline or a neighbour's ink no longer lifts the threshold),
colour_edges (a coloured glyph's stroke edges on its own channel) and glyph_cells (a small glyph whose ink the speck
filter breaks up is kept: its cell must hold ink). Every other line keeps its targets_v2 box.
"""
import numpy as np

GROW = 2
SPILL = 1
K = 0.4
FLOOR = 28.0
NOISE_MARGIN = 12.0
MIN_BLOB = 3
GAP_MIN_OFFSET = 0.5     # a row is above / below a line when the centres lie this x the smaller height apart or more
GAP_MIN_KEEP = 0.5       # bounds that would keep less of the cells' height than this are not applied
NOISE_INK_SHARE = 0.5    # noise_fix: the ring's ink-like pixels leave its noise when they are under this share


def cells_union(line):
    """[x0, y0, x1, y1] (floats) of a pool line's cells, or of its box (box_main for a line with furigana)."""
    cb = line.get("cell_boxes") or []
    if cb:
        a = np.asarray([b[:4] for b in cb], np.float64)
        return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())]
    x, y, w, h = line.get("box_main") if line.get("furigana") and line.get("box_main") else line["box"]
    return [float(x), float(y), float(x + w), float(y + h)]


def _drop_small(m, min_blob):
    if min_blob <= 1 or not m.any():
        return m
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob
    return keep[lab]


def row_unions(rec):
    """[(line id or None, [x0, y0, x1, y1])] of a pool record's text rows, the candidates for gap_bounds: the cells'
    union of every horizontal line of every kind and drop (lines, names, ruby, dropped lines), and every uncovered
    renderer box (a box no verified line holds; its quad, id None)."""
    out = [(ln["id"], cells_union(ln)) for ln in rec["lines"] if not ln["vertical"]]
    unc = set(rec.get("uncovered") or [])
    for b in rec.get("boxes") or []:
        if b["index"] in unc:
            out.append((None, [float(v) for v in b["xyxy"][:4]]))
    return out


def gap_bounds(c, others, min_offset=GAP_MIN_OFFSET, min_keep=GAP_MIN_KEEP):
    """(top, bottom, how): the across bounds of a horizontal line's cells' union c = [x0, y0, x1, y1] on its page,
    in frame px (floats; -inf / +inf where there is none), from the page's other rows (others: boxes of the same
    form) that overlap c horizontally, as eval/tsdesk_dense.regions bounds a region: half the gap to the nearest row
    above and to the nearest row below. A row is above c when it starts above c's start, ends above c's end and its
    centre lies at least min_offset x the smaller of the two heights above c's centre (a nearer row shares c's row,
    and a row that spans c's is no row above or below); below likewise. When the two overlap (cells taller than the
    line pitch), the gap is negative and the bound lies in the middle of the overlap: for two lines of one em size,
    halfway between their centres. how: "none" (no bound inside c's rows), "clip", or "guard" when the bounds would
    keep less than min_keep of c's height (then no bound is applied)."""
    import math
    top, bot = -math.inf, math.inf
    cy, ch = (c[1] + c[3]) / 2.0, c[3] - c[1]
    for o in others:
        if min(c[2], o[2]) - max(c[0], o[0]) <= 0:
            continue
        oy = (o[1] + o[3]) / 2.0
        sep = min_offset * min(ch, o[3] - o[1])
        if o[1] < c[1] and o[3] < c[3] and cy - oy >= sep:
            top = max(top, (c[1] + o[3]) / 2.0)
        elif o[1] > c[1] and o[3] > c[3] and oy - cy >= sep:
            bot = min(bot, (c[3] + o[1]) / 2.0)
    if top <= c[1] and bot >= c[3]:
        return top, bot, "none"
    if min(bot, c[3]) - max(top, c[1]) < min_keep * ch:
        return -math.inf, math.inf, "guard"
    return top, bot, "clip"


def across_rows(across, H):
    """The pixel rows [lo, hi) between the across bounds (top, bottom): the rows whose centre lies at or below top
    and above bottom, so two lines that share a bound never share a row."""
    import math
    lo, hi = 0, H
    if across is not None:
        if math.isfinite(across[0]):
            lo = max(lo, int(math.ceil(across[0] - 0.5)))
        if math.isfinite(across[1]):
            hi = min(hi, int(math.ceil(across[1] - 0.5)))
    return lo, hi


def ring_noise(dr, body, fix=False):
    """The ring's noise level (step 4): the 90th percentile of the ring's differences. fix (targets_v3): without the
    ring pixels at or above the body term (ink-like: the line's outline past its cells, the neighbouring line, a plate
    border) when they are under NOISE_INK_SHARE of the ring, so a ring through ink no longer lifts the threshold over
    the line's faint parts; a ring mostly above the body term (a dither or pattern as strong as the text) keeps the
    plain 90th percentile."""
    if fix:
        inky = dr >= body
        if inky.mean() < NOISE_INK_SHARE and (~inky).sum() >= 8:
            return float(np.percentile(dr[~inky], 90))
    return float(np.percentile(dr, 90))


def ink_box(rgb, c, grow=GROW, spill=SPILL, k=K, floor=FLOOR, noise_margin=NOISE_MARGIN, min_blob=MIN_BLOB,
            across=None, noise_fix=False, colour_edges=False, glyph_cells=None):
    """(box, how): the ink box of a line inside c = [x0, y0, x1, y1] (its cells' union, module docstring); how is
    "ink" or "cells". across: None (targets_v1's rule), or the line's across bounds (top, bottom) in frame px
    (gap_bounds: #177's gap clip): the ink is searched, and the "cells" box kept, only on the rows between them
    (across_rows); the background and the thresholds are the unclipped window's, so a line no bound cuts gets
    exactly targets_v1's box. noise_fix (targets_v3): ring_noise's fix. colour_edges (targets_v3): the stroke-edge test
    on the largest per-channel 3x3 contrast instead of the channel mean's (a coloured glyph's edge is as strong as its
    difference from the background on its own channel, as d already measures it); grey frames are unchanged.
    glyph_cells (targets_v3): the cells [x0, y0, x1, y1] of the line's non-space characters; a cell left with no ink
    after the speck filter keeps its own raw ink when that holds at least min_blob pixels (a small 。 or 、 at PSP
    size whose ring falls apart into 1-2 px pieces: a glyph is known to be there)."""
    import math
    H, W = rgb.shape[:2]
    x0, y0 = max(0, int(math.floor(c[0]))), max(0, int(math.floor(c[1])))
    x1, y1 = min(W, int(math.ceil(c[2]))), min(H, int(math.ceil(c[3])))
    lo, hi = across_rows(across, H)
    cells = [x0, max(y0, lo), x1, min(y1, hi)]
    if x1 - x0 < 2 or y1 - y0 < 2 or cells[3] - cells[1] < 2:
        return cells, "cells"
    wx0, wy0, wx1, wy1 = max(0, x0 - grow), max(0, y0 - grow), min(W, x1 + grow), min(H, y1 + grow)
    win = np.asarray(rgb[wy0:wy1, wx0:wx1]).astype(np.int16)
    if win.ndim == 2:
        win = win[:, :, None]
    ring = np.concatenate([win[0], win[-1], win[1:-1, 0], win[1:-1, -1]], 0)
    bg = np.median(ring, axis=0)
    d = np.abs(win - bg[None, None, :]).max(axis=2).astype(np.float32)
    dr = np.concatenate([d[0], d[-1], d[1:-1, 0], d[1:-1, -1]])
    inner = d[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0]
    body = float(k) * float(np.percentile(inner, 99))
    t = max(float(floor), body, ring_noise(dr, body, noise_fix) + noise_margin)
    # local contrast (the 3x3 max - min of the luma): a smooth gradient behind the text differs from the ring's
    # median without being ink; a stroke edge has both
    if colour_edges and win.shape[2] > 1:
        c3 = None
        for ch in range(win.shape[2]):
            gp = np.pad(win[:, :, ch].astype(np.float32), 1, mode="edge")
            sh = [gp[i:i + win.shape[0], j:j + win.shape[1]] for i in range(3) for j in range(3)]
            e = np.max(sh, 0) - np.min(sh, 0)
            c3 = e if c3 is None else np.maximum(c3, e)
    else:
        g = win.mean(axis=2).astype(np.float32)
        gp = np.pad(g, 1, mode="edge")
        sh = [gp[i:i + g.shape[0], j:j + g.shape[1]] for i in range(3) for j in range(3)]
        c3 = np.max(sh, 0) - np.min(sh, 0)
    c3i = c3[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0]
    tc = max(float(floor), float(k) * float(np.percentile(c3i, 99)))
    sx0, sy0 = max(wx0, x0 - spill), max(wy0, y0 - spill, lo)
    sx1, sy1 = min(wx1, x1 + spill), min(wy1, y1 + spill, hi)
    if sy1 <= sy0:
        return cells, "cells"
    sl = (slice(sy0 - wy0, sy1 - wy0), slice(sx0 - wx0, sx1 - wx0))
    # a pixel is ink when it differs from the background; it counts toward the extent when it also lies within one
    # pixel of a stroke edge (local contrast above tc), so a gradient alone never makes ink
    near_edge = c3[sl] > tc
    m0 = (d[sl] > t) & near_edge
    m = _drop_small(m0, min_blob)
    for cb in glyph_cells or ():
        a0 = max(0, int(math.floor(cb[0])) - sx0)
        a1 = min(m.shape[1], int(math.ceil(cb[2])) - sx0)
        if a1 > a0 and not m[:, a0:a1].any() and int(m0[:, a0:a1].sum()) >= min_blob:
            m[:, a0:a1] |= m0[:, a0:a1]
    cols = np.flatnonzero(m.any(0))
    rows = np.flatnonzero(m.any(1))
    if not len(cols) or not len(rows):
        return cells, "cells"
    return [int(sx0 + cols[0]), int(sy0 + rows[0]), int(sx0 + cols[-1] + 1), int(sy0 + rows[-1] + 1)], "ink"


def line_ink(rgb, line, across=None, **kw):
    """(box, how) of a pool line: ink_box on its cells' union (across: its gap bounds, or None; kw: ink_box's
    targets_v3 options)."""
    return ink_box(rgb, cells_union(line), across=across, **kw)


# ----------------------------------------------------------------------------- the training targets (amendment 1)

def positive_lines(rec):
    """The pool record's lines that synth/textspot6_pages.build_labels makes positives, in its order."""
    import textspot6_pages as TP
    R = TP.RULES
    out = []
    for ln in rec["lines"]:
        if ln["kind"] == "ruby" or ln.get("drop") in R["ignore_drops"]:
            continue
        if ln.get("drop") not in R["positive_drops"] or ln["kind"] not in R["classes"]:
            continue
        out.append(ln)
    return out


def glyph_line(ln):
    """True when a pool line's cells are its label's characters, one cell each (source glyph_cells)."""
    return ln.get("source") == "glyph_cells" and len(ln.get("cell_boxes") or []) == len(ln["text"])


def missed_cells(ln, box):
    """The indices of a glyph line's non-space characters whose cell lies wholly outside box along the line."""
    return [j for j, (ch, cb) in enumerate(zip(ln["text"], ln["cell_boxes"]))
            if not ch.isspace() and (cb[2] <= box[0] or cb[0] >= box[2])]


def _page_targets(job):
    import cv2

    import textspot6_pages as TP
    tid, rec, path, gap = job[:4]
    fix = bool(job[4]) if len(job) > 4 else False
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return tid, rec["page"], None, None
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    H, W = rgb.shape[:2]
    rows = row_unions(rec) if gap else None
    old, new, hows, fixes = [], [], [], []
    for ln in positive_lines(rec):
        b = TP.xyxy(ln["box_main"] if ln.get("furigana") and ln.get("box_main") else ln["box"])
        ob, _ = TP.refined_line(rgb, b, bool(ln["vertical"]))
        ob = [max(0, ob[0]), max(0, ob[1]), min(W, ob[2]), min(H, ob[3])]
        gh = fx = None
        if ln["vertical"]:
            nb = ob                                  # vertical lines keep their targets (horizontal reading only)
        else:
            across = None
            if gap:
                top, bot, gh = gap_bounds(cells_union(ln), [u for i, u in rows if i != ln["id"]])
                across = (top, bot)
            nb, _ = line_ink(rgb, ln, across=across)
            if fix and glyph_line(ln) and missed_cells(ln, nb):
                gc = [cb for ch, cb in zip(ln["text"], ln["cell_boxes"]) if not ch.isspace()]
                fb, _ = line_ink(rgb, ln, across=across, noise_fix=True, colour_edges=True, glyph_cells=gc)
                fx = ("fixed" if not missed_cells(ln, fb) else "partly" if len(missed_cells(ln, fb)) <
                      len(missed_cells(ln, nb)) else "kept") if fb != nb else "kept"
                nb = fb
        old.append([int(v) for v in ob])
        new.append([int(v) for v in nb])
        hows.append(gh)
        fixes.append(fx)
    return tid, rec["page"], {"old": old, "new": new}, {"gap": hows, "fix": fixes}


def cmd_build(a):
    """The target cache: per title, {page: {"old": refined_line boxes, "new": ink boxes}} of the positives of every
    page of the given splits, in build_labels order (the trainer checks "old" against the page shards' boxes).
    --gap-clip: the new boxes with #177's gap clip (targets_v2); without it, targets_v1's rule."""
    import json
    import multiprocessing as mp
    import time
    from collections import Counter
    from pathlib import Path
    pool = Path(a.pool)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = set(a.splits.split(","))
    if "test" in splits:
        raise SystemExit("the test titles' pages are never read here")
    jobs = []
    for f in sorted(pool.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["split"] in splits:
                jobs.append((f.stem, r, str(Path(a.pages) / r["image"]), bool(a.gap_clip), bool(a.cells_fix)))
    t0 = time.time()
    res = {}
    gap_how, fix_how = Counter(), Counter()
    with mp.get_context("fork").Pool(a.workers) as p:
        for tid, page, v, info in p.imap(_page_targets, jobs, chunksize=16):
            if v is not None:
                res.setdefault(tid, {})[page] = v
                gap_how.update(h for h in info["gap"] if h is not None)
                fix_how.update(h for h in info["fix"] if h is not None)
    n_lines = n_changed = 0
    for tid, pages in sorted(res.items()):
        for v in pages.values():
            n_lines += len(v["new"])
            n_changed += sum(1 for o, nw in zip(v["old"], v["new"]) if o != nw)
        (out / f"{tid}.json").write_text(json.dumps(pages, separators=(",", ":")) + "\n", encoding="utf-8")
    rule = {"grow": GROW, "spill": SPILL, "k": K, "floor": FLOOR, "noise_margin": NOISE_MARGIN, "min_blob": MIN_BLOB}
    if a.gap_clip:
        rule["gap_clip"] = {"issue": 177, "min_offset": GAP_MIN_OFFSET, "min_keep": GAP_MIN_KEEP,
                            "rows": "every other horizontal line's cells' union (any kind and drop) and every "
                                    "uncovered renderer box",
                            "horizontal_lines": dict(sorted(gap_how.items()))}
    if a.cells_fix:
        rule["cells_fix"] = {"gate": "a horizontal glyph-cell line whose box leaves a non-space character's cell "
                                     "wholly outside",
                             "changes": [f"ring_noise fix (NOISE_INK_SHARE {NOISE_INK_SHARE})", "colour_edges",
                                         "glyph_cells"],
                             "lines_gated": dict(sorted(fix_how.items()))}
    rec = {"kind": "textspot-desktop-targets", "pool_manifest_sha256": _sha(pool / "manifest.json"),
           "splits": sorted(splits), "titles": len(res), "pages": sum(len(v) for v in res.values()),
           "lines": n_lines, "changed": n_changed, "rule": rule,
           "code_sha256": _sha(Path(__file__)), "seconds": round(time.time() - t0, 1),
           "files": {f"{t}.json": _sha(out / f"{t}.json") for t in sorted(res)}}
    (out / "targets.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("titles", "pages", "lines", "changed", "seconds")}), flush=True)


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(d):
    import json
    from pathlib import Path
    d = Path(d)
    doc = json.loads((d / "targets.json").read_text(encoding="utf-8"))
    out = {}
    for name in doc["files"]:
        out[name[:-5]] = json.loads((d / name).read_text(encoding="utf-8"))
    return out, doc


DESK_HOOK_SEED = 20261101           # #171's 1a plan seed (desktopdata/hooked.py build: one capture per page)


def capture_of(smp, seed=DESK_HOOK_SEED):
    """The exact desktopdata/capture.Capture a 1a page was made with (its seeded draw, as hooked.py build draws it),
    or None when it does not reproduce the sample's recorded capture."""
    import numpy as np_

    from desktopdata import capture as C
    from desktopdata.hooked import seed_of
    meta = smp.get("meta") or {}
    w, h = meta.get("src_dims") or (None, None)
    if not w:
        return None
    cap = C.sample(np_.random.default_rng(seed_of(seed, smp["title"], smp["page"])), int(w), int(h),
                   meta.get("src_kind"))
    if cap.record() != meta.get("capture"):
        return None
    return cap


def patch_sample(smp, cache, src):
    """A hooked sample (src "hook": a page shard's; "desk_hook": #171's 1a capture of one) with its positives' boxes
    replaced by the cache's ink boxes, or None when the sample's boxes do not match the cache's refined_line boxes
    (then it is left as it is). Only the first len(cache) lines are touched (the verified positives; MeikiOCR's other
    text follows them)."""
    pages = cache.get(smp.get("title"))
    v = pages.get(smp.get("page")) if pages else None
    if v is None:
        return None
    old, new = v["old"], v["new"]
    n = len(old)
    lines = smp["lines"]
    if len(lines) < n:
        return None
    if src == "hook":
        if any([int(x) for x in lines[i][:4]] != old[i] for i in range(n)):
            return None
        mapped = new
    else:
        meta = smp.get("meta") or {}
        if meta.get("lines_outside"):
            return None
        cap = capture_of(smp)
        if cap is None or any(cap.box(old[i]) != [int(x) for x in lines[i][:4]] for i in range(n)):
            return None
        mapped = [cap.box(b) for b in new]
        if any(m is None for m in mapped):
            return None
    out = [list(m) + list(lines[i][4:]) for i, m in enumerate(mapped)] + [list(x) for x in lines[n:]]
    return dict(smp, lines=out)


def _overlapping(boxes):
    """[bool]: the boxes [x0, y0, x1, y1] that overlap another of the list (positive area)."""
    hit = [False] * len(boxes)
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0:
                hit[i] = hit[j] = True
    return hit


def compare(cache_a, cache_b, splits_of=None, titles=()):
    """Counts of how cache_b's ink boxes differ from cache_a's (two target caches of the same pages; no pixels):
    lines changed, the edges' moves (b - a, px), and the positives that overlap another positive of their page
    (refined_line's boxes, a's and b's ink boxes), overall, per split (splits_of: {(title, page): split}) and for the
    given titles."""
    from collections import Counter, defaultdict
    groups = defaultdict(Counter)
    moves = defaultdict(lambda: defaultdict(list))
    for t in sorted(cache_a):
        pa, pb = cache_a[t], cache_b.get(t, {})
        for page in sorted(pa):
            va, vb = pa[page], pb.get(page)
            if vb is None or vb["old"] != va["old"]:
                raise SystemExit(f"{t}:{page}: the two caches do not hold the same positives")
            keys = ["all"] + ([f"split:{splits_of[(t, page)]}"] if splits_of else []) + (
                [f"title:{t}"] if t in titles else [])
            ho, ha, hb = _overlapping(va["old"]), _overlapping(va["new"]), _overlapping(vb["new"])
            for i, (a, b) in enumerate(zip(va["new"], vb["new"])):
                for k in keys:
                    g = groups[k]
                    g["lines"] += 1
                    g["overlap_refined_line"] += ho[i]
                    g["overlap_a"] += ha[i]
                    g["overlap_b"] += hb[i]
                    g["overlap_a_not_refined_line"] += ha[i] and not ho[i]
                    g["overlap_b_not_refined_line"] += hb[i] and not ho[i]
                    if a != b:
                        g["changed"] += 1
                        g["changed_x"] += a[0] != b[0] or a[2] != b[2]
                        g["top_down"] += b[1] > a[1]
                        g["bottom_up"] += b[3] < a[3]
                        g["grew"] += b[1] < a[1] or b[3] > a[3] or b[0] < a[0] or b[2] > a[2]
                        for e, n in enumerate(("left", "top", "right", "bottom")):
                            if b[e] != a[e]:
                                moves[k][n].append(b[e] - a[e])
                        moves[k]["height"].append((b[3] - b[1]) - (a[3] - a[1]))
    out = {}
    for k, g in groups.items():
        rec = dict(g)
        for n, v in moves[k].items():
            a = np.abs(np.asarray(v, np.float64))
            rec[f"move_{n}"] = {"n": len(v), "median_abs": float(np.median(a)), "p90_abs": float(np.percentile(a, 90)),
                                "max_abs": float(a.max()), "hist": dict(sorted(Counter(int(x) for x in v).items()))}
        out[k] = rec
    return out


def cmd_compare(a):
    import json
    from pathlib import Path
    ca, da = load_cache(a.a)
    cb, db = load_cache(a.b)
    splits_of = None
    if a.pool:
        splits_of = {}
        for f in sorted(Path(a.pool).glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                splits_of[(f.stem, r["page"])] = r["split"]
    res = compare(ca, cb, splits_of, tuple(a.titles.split(",")) if a.titles else ())
    rec = {"kind": "textspot-desktop-targets-compare", "a": str(a.a), "b": str(a.b),
           "a_targets_sha256": _sha(Path(a.a) / "targets.json"), "b_targets_sha256": _sha(Path(a.b) / "targets.json"),
           "a_rule": da["rule"], "b_rule": db["rule"], "groups": res}
    txt = json.dumps(rec, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(json.dumps({k: {n: res[k].get(n, 0) for n in ("lines", "changed", "overlap_a", "overlap_b")}
                      for k in sorted(res)}), flush=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pool", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", default="train,guard,hval")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--gap-clip", action="store_true", help="#177: clip each horizontal line's ink search across at "
                                                           "half the gap to its neighbouring rows (targets_v2)")
    p.add_argument("--cells-fix", action="store_true", help="targets_v3: box again the lines whose box leaves a "
                                                            "non-space character's cell outside (module docstring)")
    p = sub.add_parser("compare", help="counts of how cache B's ink boxes differ from cache A's (no pixels)")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--pool", help="the page pool (for the per-split counts)")
    p.add_argument("--titles", default="", help="titles to count apart (comma-separated)")
    p.add_argument("--out")
    a = ap.parse_args(argv)
    {"build": cmd_build, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    for _p in (_root / "synth", _root / "teacher", _root / "train", _root):
        sys.path.insert(0, str(_p))
    main()

#!/usr/bin/env python3
"""textspot-desktop round 2 (#173): the label coverage of the detector's hooked and desktop training sources. Text that
no positive and no ignore area covers is background to the trainer (train/tsdesk_det.targets); this finds it with
MeikiOCR's detector boxes (score >= 0.5), each refined to ink as the pools refine them (synth/textspot6_pages.py
meiki: textspot2_labels.refine_h / refine_t, refined_of P1).

--measure ink (the default): the ink inside the box (ink_in: train/tsdesk_targets.ink_box's stroke rule on the stored
grey frame, without its ring-noise term) that lies in no positive and no ignore area (each grown by 1 px), as
8-connected components after a 3x3 dilation; a component counts when it is at least INK_ALONG x the box's size across
long along the line and INK_ACROSS x it across (about half a glyph: a shadow sliver above or below a labelled line
does not count), and the box counts when its counted components hold at least INK_SHARE of its ink: text the trainer
sees as background, whatever the box's own offset. Each counted box is placed (where): "cells" when its unlabelled ink
lies mostly inside a positive line's own glyph cells (hook / desk_hook with --pool: the target box cut its line),
"row" when it shares the row of a positive or an ignore area within WHERE_REACH line heights of its ends (a page
cursor, an icon, an end glyph past a renderer box), "apart" otherwise (other text: UI, names, scene text).

--measure box: desktop-data 1b v3's geometric test (desktopdata/realplus.uncovered_spans: a run of unlabelled columns
at least 0.4 x the box's height long, or of unlabelled rows at least 0.2 x it tall), spans thinner than MIN_SPAN px
dropped; a box offset from its line by a quarter of its height fails it although its line is labelled (MeikiOCR's
PSP boxes), so it overcounts.

  tsdesk_coverage.py meiki --manifest M --root DIR --out CACHE [--workers 5] [--shards N]       (one heavy slot)
      MeikiOCR's detector (teacher/engines.Meiki: its pinned files, threshold 0.2) on every sample of a shard set
      (#97's frames or textspot6-pages shards; the stored grey frame in three channels), each box refined as the pools
      refine theirs; CACHE/<shard>.jsonl, one line per sample {"shard", "i", "meiki": [{"raw", "score", "box",
      "vertical"}]} (the pools' format); a finished shard is skipped
  tsdesk_coverage.py run --source hook|desk_hook|desk_synth --manifest M --root DIR --meiki CACHE
                     [--targets DIR] [--splits train] [--measure ink|box] [--pool P] --out COUNTS.json
                     --rows ROWS.jsonl [--workers 5]
      the labels of each sample as the trainer sees them (on hook and desk_hook the positives patched with the
      --targets cache, train/tsdesk_targets.patch_sample), under recipe base and under recipe junk (tsdesk_det.
      strip_weak), against each MeikiOCR box. hook: CACHE = the pools' cache (synth/textspot6_pages.py meiki: one
      {"title", "page", "meiki": [{"raw", "score", ...}]} line per page, the boxes the labels were built from);
      desk_hook: the same cache, each box mapped through its page's capture (train/tsdesk_targets.capture_of);
      desk_synth: a cache of `meiki` above. COUNTS holds counts only; ROWS (private) each uncovered box and each
      weak MeikiOCR ignore area the junk recipe turns to background, with its sample, for the sheets
  tsdesk_coverage.py edits --rows ROWS.jsonl ... --shards DIR --meiki CACHE [--targets DIR] --pool P --out EDITS
      the label-edit cache: each hook page's 'apart' boxes as notdlg positives (edit_candidates: not on a verified
      line's row, not ruby-like, not over a positive), per title {page: [lines]} in native px, and EDITS/edits.json
      (counts, sha256); train/tsdesk_det.py --label-edits EDITS appends them to the hook and desk_hook samples
  tsdesk_coverage.py sheet --rows ROWS.jsonl --root DIR --out DIR --name NAME [--kind uncovered|weak] [--n 24]
                     [--seed 0] [--min-score S] [--max-score S] [--source S] [--exclude-titles T,...] [--distinct]
                     [--min-frac F] [--max-frac F]
      private contact sheets (<= 1280 px) of a seeded draw of those rows: a crop of the stored grey frame, the
      MeikiOCR box orange, the positives green, the ignore areas blue, the unlabelled spans red; NAME.json lists the
      tiles
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "teacher", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

STRONG = 0.5        # synth/textspot6_pages.RULES meiki_pos_min: a box this sure is text
DET_THR = 0.2       # the pools' MeikiOCR cache threshold (textspot6_pages.RULES meiki_det_thr)
ROW_RUN = 0.2       # desktopdata/realplus.RULES_V3 row_run["meiki"]
MIN_SPAN = 4        # px: a span thinner than this on either side is not counted
GRID = 8            # px: distinct elements are counted per title on this grid of the box's native corners
INK_SHARE = 0.15    # the unlabelled share of a box's ink that counts (ink measure)
INK_ALONG = 0.4     # the unlabelled ink's extent along the line, x the box's size across
INK_ACROSS = 0.3    # its extent across, x the box's size across
WHERE_REACH = 1.5   # "row": within this many line heights past a positive's ends on its row
BANDS = (0.2, 0.3, 0.4, 0.5, 0.7, 0.9)


def band(score):
    lo = max([b for b in BANDS if b <= score], default=0.0)
    return f"{lo:.1f}+"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def clip(b, W, H):
    x0, y0, x1, y1 = max(0, int(b[0])), max(0, int(b[1])), min(W, int(b[2])), min(H, int(b[3]))
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def label_mask(lines, ignore, W, H):
    """(mask, boxes): what is labelled, the positives' boxes and the ignore areas (realplus.labels_of_v3's mask)."""
    lab = np.zeros((H, W), bool)
    boxes = []
    for b in [ln[:4] for ln in lines] + [g[:4] for g in ignore]:
        c = clip(b, W, H)
        if c is not None:
            lab[c[1]:c[3], c[0]:c[2]] = True
            boxes.append(c)
    return lab, boxes


def uncovered(b, lab, boxes):
    """The unlabelled spans of a detector box b (desktopdata/realplus.uncovered_spans, MeikiOCR's row run)."""
    from desktopdata import realplus as RP
    H, W = lab.shape
    c = clip(b, W, H)
    if c is None or c[2] - c[0] < 2 or c[3] - c[1] < 2:
        return []
    return [sp for sp in RP.uncovered_spans(c, lab, boxes, ROW_RUN)
            if sp[2] - sp[0] >= MIN_SPAN and sp[3] - sp[1] >= MIN_SPAN]


def span_frac(b, spans, W, H):
    """The share of box b's area its spans cover (their union)."""
    c = clip(b, W, H)
    if c is None or not spans:
        return 0.0
    m = np.zeros((c[3] - c[1], c[2] - c[0]), bool)
    for s in spans:
        m[max(0, s[1] - c[1]):max(0, s[3] - c[1]), max(0, s[0] - c[0]):max(0, s[2] - c[0])] = True
    return float(m.mean())


def ink_in(g, b, grow=2, floor=28.0, k=0.4, min_blob=3):
    """The stroke pixels of grey frame g inside box b = [x0, y0, x1, y1] (a bool array over b): differing from the
    median of the one-pixel ring of b grown by `grow` by more than max(floor, k x the 99th percentile of that
    difference inside b), and on a 3x3 local-contrast edge above max(floor, k x its 99th percentile inside b)
    (train/tsdesk_targets.ink_box's rule without its ring-noise term); specks under min_blob px dropped."""
    import tsdesk_targets as TT
    H, W = g.shape
    x0, y0, x1, y1 = b
    wx0, wy0, wx1, wy1 = max(0, x0 - grow), max(0, y0 - grow), min(W, x1 + grow), min(H, y1 + grow)
    win = np.asarray(g[wy0:wy1, wx0:wx1], np.float32)
    if win.shape[0] < 3 or win.shape[1] < 3:
        return np.zeros((y1 - y0, x1 - x0), bool)
    ring = np.concatenate([win[0], win[-1], win[1:-1, 0], win[1:-1, -1]])
    d = np.abs(win - np.median(ring))
    sl = (slice(y0 - wy0, y1 - wy0), slice(x0 - wx0, x1 - wx0))
    t = max(floor, k * float(np.percentile(d[sl], 99)))
    gp = np.pad(win, 1, mode="edge")
    sh = [gp[i:i + win.shape[0], j:j + win.shape[1]] for i in range(3) for j in range(3)]
    c3 = (np.max(sh, 0) - np.min(sh, 0))[sl]
    tc = max(floor, k * float(np.percentile(c3, 99)))
    return TT._drop_small((d[sl] > t) & (c3 > tc), min_blob)


def labelled_in(b, boxes, g=1):
    """A bool array over box b: the pixels inside any of boxes (positives and ignore areas), each grown by g px."""
    x0, y0, x1, y1 = b
    m = np.zeros((y1 - y0, x1 - x0), bool)
    for o in boxes:
        a0, a1 = max(x0, o[0] - g), min(x1, o[2] + g)
        c0, c1 = max(y0, o[1] - g), min(y1, o[3] + g)
        if a1 > a0 and c1 > c0:
            m[c0 - y0:c1 - y0, a0 - x0:a1 - x0] = True
    return m


def uncovered_ink(g, b, boxes):
    """(spans, share): the counted components of box b's unlabelled ink (their boxes [[x0, y0, x1, y1]] in frame px,
    [] when none counts or their share is under INK_SHARE) and their share of the box's ink (module docstring)."""
    import cv2
    H, W = g.shape
    c = clip(b, W, H)
    if c is None or c[2] - c[0] < 2 or c[3] - c[1] < 2:
        return [], 0.0
    ink = ink_in(g, c)
    n = int(ink.sum())
    if not n:
        return [], 0.0
    u = ink & ~labelled_in(c, boxes)
    if not u.any():
        return [], 0.0
    vertical = (c[3] - c[1]) > 2.5 * (c[2] - c[0])
    across = (c[2] - c[0]) if vertical else (c[3] - c[1])
    ud = cv2.dilate(u.astype(np.uint8), np.ones((3, 3), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(ud, connectivity=8)
    keep, spans = [], []
    for j in range(1, k):
        x, y, w, h = (int(v) for v in st[j, :4])
        along, acr = (h, w) if vertical else (w, h)
        if along >= INK_ALONG * across and acr >= INK_ACROSS * across:
            keep.append(j)
    if not keep:
        return [], 0.0
    m = np.isin(lab, keep) & u
    share = float(m.sum()) / n
    if share < INK_SHARE:
        return [], share
    for j in keep:
        mj = (lab == j) & u
        rows, cols = np.flatnonzero(mj.any(1)), np.flatnonzero(mj.any(0))
        if len(rows):
            spans.append([int(c[0] + cols[0]), int(c[1] + rows[0]), int(c[0] + cols[-1] + 1), int(c[1] + rows[-1] + 1)])
    return spans, share


def where_of(spans, pos, cells):
    """Where a box's unlabelled ink lies (module docstring): "cells", "row" or "apart". spans: its counted
    components' boxes; pos: the sample's labelled rows (its horizontal positives' boxes and its ignore areas); cells:
    the positive lines' cells' unions in the sample's px (the verified lines', in shard order; [] when unknown)."""
    def area(q):
        return max(0, q[2] - q[0]) * max(0, q[3] - q[1])

    def inter(p, q):
        return area([max(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), min(p[3], q[3])])
    tot = sum(area(q) for q in spans) or 1
    if cells and sum(max((inter(q, cu) for cu in cells), default=0) for q in spans) >= 0.5 * tot:
        return "cells"
    for q in spans:
        for p in pos:
            hp = p[3] - p[1]
            if hp <= 0 or min(q[3], p[3]) - max(q[1], p[1]) < 0.5 * min(hp, q[3] - q[1]):
                continue
            if q[2] > p[0] - WHERE_REACH * hp and q[0] < p[2] + WHERE_REACH * hp:
                return "row"
    return "apart"


def weak_areas(smp, src):
    """(the ignore areas the junk recipe keeps, the weak MeikiOCR areas it turns to background)."""
    import tsdesk_det as TD
    out = TD.strip_weak(smp, src)
    ign = smp["ignore"]
    return out["ignore"], ign[len(out["ignore"]):]


# ----------------------------------------------------------------------------- MeikiOCR on stored frames

_M = {}


def _init_meiki():
    from teacher import engines as E
    _M["m"] = E.Meiki(det_thr=DET_THR, n_threads=1)


def _meiki_shard(job):
    import textspot2_labels as L2
    root, s, out = job
    path = Path(out) / f"{s['name']}.jsonl"
    if path.exists():
        return s["name"], 0, 0.0
    t0 = time.time()
    base = Path(root) / s.get("dir", "") / s["name"]
    data = np.load(base.with_suffix(".npy"), mmap_mode="r")
    samples = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["samples"]
    rows = []
    for i, smp in enumerate(samples):
        g = np.asarray(data[smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(smp["h"], smp["w"])
        rgb = np.ascontiguousarray(np.repeat(g[:, :, None], 3, axis=2))
        out = []
        for raw, sc in L2.meiki_detect(_M["m"], rgb):          # grey in three channels: BGR and RGB are the same
            ref = {"raw": raw, "score": sc, "ref_h": L2.refine_h(rgb, raw), "ref_t": L2.refine_t(rgb, raw)}
            box, vertical = L2.refined_of(ref, "P1")
            out.append({"raw": [int(v) for v in raw], "score": round(float(sc), 4), "box": [int(v) for v in box],
                        "vertical": bool(vertical)})
        rows.append({"shard": s["name"], "i": i, "meiki": out})
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    return s["name"], len(rows), time.time() - t0


def cmd_meiki(a):
    doc = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    shards = doc["shards"][:a.shards] if a.shards else doc["shards"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.root, s, str(out)) for s in shards]
    t0 = time.time()
    n = 0
    with mp.get_context("fork").Pool(a.workers, initializer=_init_meiki) as p:
        for name, k, sec in p.imap_unordered(_meiki_shard, jobs):
            n += k
            print(json.dumps({"shard": name, "samples": k, "sec": round(sec, 1), "total": n,
                              "elapsed": round(time.time() - t0, 1)}), flush=True)
    rec = {"kind": "tsdesk-coverage-meiki", "manifest": str(a.manifest), "manifest_sha256": sha256_file(a.manifest),
           "root": str(a.root), "shards": len(shards), "det_thr": DET_THR, "seconds": round(time.time() - t0, 1),
           "input": "the stored grey frame in three channels (BGR, as teacher/label.py feeds Meiki)"}
    (out / "meiki.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- the coverage of one sample

def sample_coverage(smp, meiki, src, cache=None, grey=None, measure="ink", cells=None):
    """(counts, rows) of one sample. meiki: [{"raw", "score", "box", ...}] MeikiOCR boxes in the sample's px (box: the
    raw box refined to ink; each may carry "native": its box on the native page); src: the trainer's source name
    (hook, desk_hook, desk_synth); cache: a target cache for hook / desk_hook; grey: the sample's grey frame (needed
    by measure "ink", the default; "box": desktop-data 1b v3's geometric test); cells: the verified positive lines'
    cells' unions in the sample's px (where_of's "cells"; None: unknown)."""
    import tsdesk_targets as TT
    c = Counter()
    rows = []
    if cache is not None and src in ("hook", "desk_hook"):
        ps = TT.patch_sample(smp, cache, src)
        c["patched" if ps is not None else "unpatched"] += 1
        if ps is not None:
            smp = ps
    W, H = int(smp["w"]), int(smp["h"])
    keep, weak = weak_areas(smp, src) if src in ("hook", "desk_hook") else (smp["ignore"], [])
    lab_b, boxes_b = label_mask(smp["lines"], smp["ignore"], W, H)
    lab_j, boxes_j = label_mask(smp["lines"], keep, W, H) if weak else (lab_b, boxes_b)
    c["samples"] += 1
    c["weak_areas_junk"] += len(weak)
    any_b = any_j = False
    for m in meiki:
        sc = float(m["score"])
        c[f"meiki_{band(sc)}"] += 1
        if sc < STRONG:
            continue
        b = m.get("box") or m["raw"]
        c["strong"] += 1
        if measure == "ink":
            sp_b, sh_b = uncovered_ink(grey, b, boxes_b)
            sp_j, sh_j = uncovered_ink(grey, b, boxes_j) if weak else (sp_b, sh_b)
        else:
            sp_b = uncovered(b, lab_b, boxes_b)
            sp_j = uncovered(b, lab_j, boxes_j) if weak else sp_b
            sh_b, sh_j = span_frac(b, sp_b, W, H), span_frac(b, sp_j, W, H)
        base = {"box": [int(v) for v in b[:4]], "raw": [int(v) for v in m["raw"][:4]], "score": round(sc, 4),
                "native": m.get("native") or [int(v) for v in b[:4]]}
        if sp_b:
            any_b = True
            wh = where_of(sp_b, [ln[:4] for ln in smp["lines"] if not ln[4]] + [g[:4] for g in smp["ignore"]],
                          cells or [])
            c[f"strong_uncovered_where_{wh}"] += 1
            whole = sh_b >= 0.9 if measure == "ink" else sp_b == [clip(b, W, H)]
            c["strong_uncovered"] += 1
            c["strong_uncovered_whole"] += whole
            c[f"strong_uncovered_{band(sc)}"] += 1
            rows.append(dict(base, kind="uncovered", spans=sp_b, frac=round(sh_b, 3), whole=bool(whole), where=wh))
        if sp_j:
            any_j = True
            c["strong_uncovered_junk"] += 1
            if not sp_b:
                c["strong_uncovered_only_junk"] += 1
                rows.append(dict(base, kind="uncovered_junk_only", spans=sp_j, frac=round(sh_j, 3)))
    c["samples_uncovered"] += any_b
    c["samples_uncovered_junk"] += any_j
    # the weak areas the junk recipe turns to background, each with the weak MeikiOCR box it came from (the pools'
    # rule: the refined box, clipped; the best overlap)
    for g in weak:
        best, hit = 0.0, None
        for m in meiki:
            if float(m["score"]) >= STRONG:
                continue
            bx = m.get("box") or m["raw"]
            ix = max(0, min(g[2], bx[2]) - max(g[0], bx[0])) * max(0, min(g[3], bx[3]) - max(g[1], bx[1]))
            u = max(1, (g[2] - g[0]) * (g[3] - g[1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - ix)
            if ix / u > best:
                best, hit = ix / u, m
        sc = float(hit["score"]) if hit is not None and best >= 0.5 else None
        c[f"weak_{band(sc) if sc is not None else 'unmatched'}"] += 1
        rows.append({"kind": "weak", "box": [int(v) for v in g[:4]], "score": round(sc, 4) if sc is not None else None,
                     "native": (hit.get("native") or [int(v) for v in g[:4]]) if sc is not None else None})
    return c, rows


# ----------------------------------------------------------------------------- the sources

def _load_meiki(path, key):
    """{key: [{"raw", "score", "box", "vertical"}]} of a MeikiOCR cache file (the pools' per title, keyed by page, or
    this tool's per shard, keyed by sample index)."""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r[key]] = r.get("meiki") or []
    return out


_RUN = {}


def _init_run(targets):
    import tsdesk_targets as TT
    _RUN["cache"] = TT.load_cache(targets)[0] if targets else None


def element_key(r, title):
    """A distinct on-screen element: the title and the box's native corners on a GRID px grid (a UI label repeated on
    every page of a title is one element)."""
    nb = r.get("native")
    if nb is None:
        return None
    return f"{title}:" + ",".join(str(int(round(v / GRID))) for v in nb)


def _run_shard(job):
    import tsdesk_targets as TT
    src, root, s, meiki, splits, measure, pool = job
    base = Path(root) / s.get("dir", "") / s["name"]
    samples = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["samples"]
    data = np.load(base.with_suffix(".npy"), mmap_mode="r") if measure == "ink" else None
    recs = {}
    if src in ("hook", "desk_hook"):
        title = s.get("title") or samples[0]["title"]
        pm = _load_meiki(Path(meiki) / f"{title}.jsonl", "page")
        if pool:
            for line in (Path(pool) / f"{title}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["split"] != "test":
                    recs[r["page"]] = r
    else:
        pm = _load_meiki(Path(meiki) / f"{s['name']}.jsonl", "i")
    tot = Counter()
    by_title = defaultdict(Counter)
    keys = defaultdict(set)
    rows = []
    for i, smp in enumerate(samples):
        if splits and smp.get("split") and smp["split"] not in splits:
            continue
        cells = None
        rec = recs.get(smp.get("page")) if recs else None
        if rec is not None:
            cells = [TT.cells_union(ln) for ln in TT.positive_lines(rec) if not ln["vertical"]]
        if src == "hook":
            mk = pm.get(smp["page"])
        elif src == "desk_hook":
            native = pm.get(smp["page"])
            cap = TT.capture_of(smp) if native is not None else None
            if cap is None:
                tot["no_capture" if native is not None else "no_meiki"] += 1
                continue
            if cells is not None:
                cells = [q for q in (cap.box(cu) for cu in cells) if q is not None]
            mk = []
            for m in native:
                raw, bx = cap.box(m["raw"]), cap.box(m["box"])
                if raw is not None and bx is not None:
                    mk.append(dict(m, raw=raw, box=bx, native=[int(v) for v in m["box"]]))
        else:
            mk = pm.get(i)
        if mk is None:
            tot["no_meiki"] += 1
            continue
        g = (np.asarray(data[smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(smp["h"], smp["w"])
             if data is not None else None)
        c, rs = sample_coverage(smp, mk, src, _RUN.get("cache"), g, measure, cells)
        tot.update(c)
        t = smp.get("title") or "synth"
        by_title[t].update({k: c[k] for k in ("samples", "strong", "strong_uncovered", "weak_areas_junk")})
        for r in rs:
            r.update(source=src, shard=s["name"], dir=s.get("dir", ""), i=i, w=int(smp["w"]), h=int(smp["h"]),
                     title=smp.get("title"), page=smp.get("page"))
            k = element_key(r, t) if src != "desk_synth" else f"{s['name']}:{i}:{r['box']}"
            r["element"] = k
            if k is not None:
                keys[r["kind"]].add(k)
            rows.append(r)
    return tot, {k: dict(v) for k, v in by_title.items()}, rows, {k: sorted(v) for k, v in keys.items()}


def cmd_run(a):
    doc = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    splits = tuple(a.splits.split(",")) if a.splits else None
    if splits and "test" in splits:
        raise SystemExit("the test titles' pages are never read here")
    shards = [s for s in doc["shards"] if not (splits and s.get("split") and s["split"] not in splits)]
    jobs = [(a.source, a.root, s, a.meiki, splits, a.measure, a.pool) for s in shards]
    t0 = time.time()
    tot = Counter()
    by_title = defaultdict(Counter)
    keys = defaultdict(set)
    n_rows = 0
    with open(a.rows, "w", encoding="utf-8") as fo, \
            mp.get_context("fork").Pool(a.workers, initializer=_init_run, initargs=(a.targets,)) as p:
        for c, bt, rows, ks in p.imap_unordered(_run_shard, jobs, chunksize=1):
            tot.update(c)
            for k, v in bt.items():
                by_title[k].update(v)
            for k, v in ks.items():
                keys[k].update(v)
            for r in rows:
                fo.write(json.dumps(r) + "\n")
            n_rows += len(rows)
    distinct = {k: len(v) for k, v in sorted(keys.items())}
    distinct_titles = {k: len({x.split(":")[0] for x in v}) for k, v in sorted(keys.items())}
    rec = {"kind": "tsdesk-coverage", "source": a.source, "manifest": str(a.manifest),
           "manifest_sha256": sha256_file(a.manifest), "splits": list(splits) if splits else None,
           "meiki": str(a.meiki), "targets": a.targets, "pool": a.pool,
           "targets_sha256": sha256_file(Path(a.targets) / "targets.json") if a.targets else None,
           "rule": {"measure": a.measure, "strong": STRONG, "row_run": ROW_RUN, "min_span_px": MIN_SPAN,
                    "grid_px": GRID, "where_reach": WHERE_REACH,
                    "ink": {"share": INK_SHARE, "along": INK_ALONG, "across": INK_ACROSS,
                                             "label_grow_px": 1, "rule": "ink_in: tsdesk_targets.ink_box's stroke "
                                                                         "rule on the stored grey, no ring-noise term"},
                    "box": "the MeikiOCR box refined to ink (the pools' box), raw where absent",
                    "test": "desktopdata/realplus.uncovered_spans",
                    "labels": "positives (patched with --targets on hook / desk_hook) and ignore areas; junk: "
                              "tsdesk_det.strip_weak"},
           "code_sha256": sha256_file(Path(__file__)), "seconds": round(time.time() - t0, 1),
           "totals": dict(sorted(tot.items())), "distinct_elements": distinct,
           "titles_with_elements": distinct_titles,
           "by_title": {k: dict(v) for k, v in sorted(by_title.items())}, "rows_file": str(a.rows), "rows": n_rows}
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({**{k: tot[k] for k in ("samples", "strong", "strong_uncovered", "strong_uncovered_whole",
                                              "samples_uncovered", "strong_uncovered_junk", "weak_areas_junk")},
                      "distinct": distinct}), flush=True)


# ----------------------------------------------------------------------------- label edits (apart -> notdlg)

EDIT_ROW_OVERLAP = 0.5      # a box shares a verified line's row: vertical overlap >= this x the smaller height


def edit_candidates(smp, rows, meiki):
    """(adds, reasons): the notdlg positives [x0, y0, x1, y1, vertical, "notdlg", weight] a hook sample gets from its
    'apart' coverage rows, and Counter of the rows left out. smp: the sample as the trainer sees it (positives patched
    with the target cache); rows: its rows of kind uncovered and where apart; meiki: the page's pool MeikiOCR boxes
    ({"raw", "score", "box", "vertical"}). Left out: a box on a verified dialogue or name line's row (a page cursor or
    an icon far past the line's end: background, as the 'row' kind), a ruby-like box over a line (furigana the hook
    gives no ruby line for: desktop_det.ruby_like; the pools make ruby an ignore area, never a positive), a box that
    overlaps a positive, and the lower-scored of two that overlap. The box is the pools' refined box, its weight the
    pools' notdlg weight (synth/textspot6_pages.meiki_weight)."""
    import desktop_det as DD
    import textspot6_pages as TP
    by_raw = {tuple(m["raw"]): m for m in meiki}
    verified = [ln for ln in smp["lines"] if not ln[4] and ln[5] in ("dialogue", "name")]
    pos = [ln[:4] for ln in smp["lines"]]
    out, why = [], Counter()

    def ov(a, b):
        return min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0
    for r in sorted(rows, key=lambda r: -r["score"]):
        m = by_raw.get(tuple(r["raw"]))
        if m is None:
            why["no_meiki_box"] += 1
            continue
        b = [int(v) for v in m["box"]]
        hb = b[3] - b[1]
        if any(min(b[3], ln[3]) - max(b[1], ln[1]) >= EDIT_ROW_OVERLAP * max(1, min(hb, ln[3] - ln[1]))
               for ln in verified):
            why["row"] += 1
            continue
        if not m["vertical"] and 0 in DD.ruby_like([[*b, 0]] + [[*ln[:4], ln[4]] for ln in verified]):
            why["ruby"] += 1
            continue
        if any(ov(b, p) for p in pos):
            why["overlap"] += 1
            continue
        if any(ov(b, a[:4]) for a in out):
            why["duplicate"] += 1
            continue
        out.append([*b, int(bool(m["vertical"])), "notdlg", round(TP.meiki_weight(m["score"]), 4)])
    return out, why


def _edits_title(job):
    import tsdesk_targets as TT
    title, rows, shards_dir, meiki_dir, targets = job
    cache = TT.load_cache(targets)[0] if targets else None
    pm = _load_meiki(Path(meiki_dir) / f"{title}.jsonl", "page")
    by_shard = defaultdict(list)
    for r in rows:
        by_shard[(r["dir"], r["shard"])].append(r)
    out, why = {}, Counter()
    for (d, shard), rs in sorted(by_shard.items()):
        samples = json.loads((Path(shards_dir) / d / f"{shard}.json").read_text(encoding="utf-8"))["samples"]
        by_i = defaultdict(list)
        for r in rs:
            by_i[r["i"]].append(r)
        for i, rr in sorted(by_i.items()):
            smp = samples[i]
            if cache is not None:
                smp = TT.patch_sample(smp, cache, "hook") or smp
            adds, w = edit_candidates(smp, rr, pm.get(smp["page"]) or [])
            why.update(w)
            if adds:
                out[smp["page"]] = adds
    return title, out, why


def cmd_edits(a):
    """The label-edit cache: per title {page: [notdlg positives]} in native page px (edit_candidates), from the
    hook coverage rows' 'apart' boxes; train/tsdesk_det.py --label-edits appends them to hook and desk_hook samples."""
    rows = defaultdict(list)
    for p in a.rows:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["kind"] == "uncovered" and r.get("where") == "apart" and r["source"] == "hook":
                rows[r["title"]].append(r)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res, why = {}, Counter()
    jobs = [(t, rs, a.shards, a.meiki, a.targets) for t, rs in sorted(rows.items())]
    with mp.get_context("fork").Pool(a.workers) as p:
        for title, pages, w in p.imap_unordered(_edits_title, jobs):
            why.update(w)
            if pages:
                res[title] = pages
    splits = {}
    for f in sorted(Path(a.pool).glob("*.jsonl")):
        if f.stem in res:
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                splits[(f.stem, r["page"])] = r["split"]
    by_split, by_title, elements = Counter(), Counter(), set()
    for t, pages in sorted(res.items()):
        for page, adds in pages.items():
            by_split[splits.get((t, page), "?")] += len(adds)
            by_title[t] += len(adds)
            for b in adds:
                elements.add((t, *(int(round(v / GRID)) for v in b[:4])))
        (out / f"{t}.json").write_text(json.dumps(pages, separators=(",", ":")) + "\n", encoding="utf-8")
    rec = {"kind": "textspot-desktop-label-edits", "issue": 173,
           "rule": {"from": "tsdesk_coverage run rows of kind uncovered, where apart (strong MeikiOCR boxes >= 0.5 "
                            "with unlabelled ink away from every labelled row)",
                    "add": "a notdlg positive: the pools' refined box and orientation, weight meiki_weight(score)",
                    "left_out": ["row: on a verified dialogue or name line's row", "ruby: desktop_det.ruby_like over "
                                 "a verified line", "overlap: overlaps a positive", "duplicate: overlaps a kept edit"],
                    "row_overlap": EDIT_ROW_OVERLAP,
                    "coords": "native page px; desk_hook maps them through its capture"},
           "rows_files": {str(p): sha256_file(p) for p in a.rows}, "targets": a.targets,
           "targets_sha256": sha256_file(Path(a.targets) / "targets.json") if a.targets else None,
           "candidates": sum(len(v) for v in rows.values()), "left_out": dict(sorted(why.items())),
           "added": sum(by_title.values()), "pages": sum(len(v) for v in res.values()), "titles": len(res),
           "distinct_elements": len(elements), "by_split": dict(sorted(by_split.items())),
           "by_title": dict(by_title.most_common()), "code_sha256": sha256_file(Path(__file__)),
           "seconds": round(time.time() - t0, 1),
           "files": {f"{t}.json": sha256_file(out / f"{t}.json") for t in sorted(res)}}
    (out / "edits.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("candidates", "added", "pages", "titles", "distinct_elements", "left_out",
                                           "by_split")}), flush=True)


def load_edits(d):
    """({title: {page: [lines]}}, the edits.json record) of a label-edit cache (cmd_edits); every file's sha256
    checked."""
    d = Path(d)
    doc = json.loads((d / "edits.json").read_text(encoding="utf-8"))
    out = {}
    for name, sha in doc["files"].items():
        if sha256_file(d / name) != sha:
            raise SystemExit(f"{d / name}: sha256 differs from edits.json's")
        out[name[:-5]] = json.loads((d / name).read_text(encoding="utf-8"))
    return out, doc


def apply_edits(smp, edits, src):
    """A hook (src "hook") or desk_hook sample with its label edits appended to its lines (desk_hook: mapped through
    the page's capture, train/tsdesk_targets.capture_of), or None when it has none (or its capture does not
    reproduce). Its other lines and its ignore areas are untouched (the junk recipe's counts stay right)."""
    import tsdesk_targets as TT
    adds = (edits.get(smp.get("title")) or {}).get(smp.get("page"))
    if not adds:
        return None
    if src == "desk_hook":
        cap = TT.capture_of(smp)
        if cap is None:
            return None
        adds = [[*m, *b[4:]] for b in adds for m in [cap.box(b[:4])] if m is not None]
        if not adds:
            return None
    return dict(smp, lines=[list(x) for x in smp["lines"]] + [list(b) for b in adds])


# ----------------------------------------------------------------------------- sheets

def _frame(root, r, cache):
    key = (r["dir"], r["shard"])
    if key not in cache:
        base = Path(root) / r["dir"] / r["shard"]
        cache[key] = (np.load(base.with_suffix(".npy"), mmap_mode="r"),
                      json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["samples"])
    data, samples = cache[key]
    smp = samples[r["i"]]
    return np.asarray(data[smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(smp["h"], smp["w"]), smp


def cmd_sheet(a):
    import random

    import cv2
    rows = [json.loads(x) for x in Path(a.rows).read_text(encoding="utf-8").splitlines()]
    ex = set(a.exclude_titles.split(",")) if a.exclude_titles else set()
    rows = [r for r in rows if r["kind"] == a.kind and (a.source is None or r["source"] == a.source)
            and r.get("title") not in ex and (r["score"] is None or a.min_score <= r["score"] < a.max_score)
            and a.min_frac <= r.get("frac", 1.0) <= a.max_frac]
    rng = random.Random(a.seed)
    if a.distinct:                      # one row per element (a UI label repeated on every page counts once)
        by = defaultdict(list)
        for r in rows:
            by[r.get("element") or id(r)].append(r)
        rows = [rng.choice(v) for _, v in sorted(by.items(), key=lambda kv: str(kv[0]))]
    pick = rng.sample(rows, min(a.n, len(rows)))
    roots = dict(kv.split("=", 1) for kv in a.root)
    TW, TH, cols = 312, 150, 4
    tiles, meta, cache = [], [], {}
    for k, r in enumerate(pick):
        g, smp = _frame(roots[r["source"]], r, cache)
        H, W = g.shape
        x0, y0, x1, y1 = r["box"]
        bh = max(8, y1 - y0)
        cx0, cx1 = max(0, x0 - max(16, bh)), min(W, x1 + max(16, bh))
        cy0, cy1 = max(0, y0 - max(10, int(1.2 * bh))), min(H, y1 + max(10, int(1.2 * bh)))
        crop = cv2.cvtColor(np.ascontiguousarray(g[cy0:cy1, cx0:cx1]), cv2.COLOR_GRAY2BGR)
        f = min(2.0, (TW - 4) / crop.shape[1], (TH - 20) / crop.shape[0])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)

        def rect(b, col):
            p0 = (int(round((b[0] - cx0) * f)), int(round((b[1] - cy0) * f)))
            p1 = (int(round((b[2] - cx0) * f)) - 1, int(round((b[3] - cy0) * f)) - 1)
            cv2.rectangle(crop, p0, p1, col, 1)
        for ln in smp["lines"]:
            rect(ln[:4], (0, 200, 0))
        for b in smp["ignore"]:
            rect(b[:4], (255, 120, 0))
        for b in r.get("spans") or []:
            rect(b, (0, 0, 255))
        rect(r["box"], (0, 165, 255))
        tile = np.full((TH, TW, 3), 255, np.uint8)
        tile[18:18 + crop.shape[0], 2:2 + crop.shape[1]] = crop
        lab = f"{k + 1}. {r['source']} {r.get('title') or ''} s{r['score']}"
        cv2.putText(tile, lab, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        tiles.append(tile)
        meta.append({"n": k + 1, **{x: r.get(x) for x in ("source", "shard", "i", "title", "page", "box", "score",
                                                          "frac", "whole", "element")}})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_img = [np.concatenate(tiles[i:i + cols] + [np.full((TH, TW, 3), 255, np.uint8)] *
                               (cols - len(tiles[i:i + cols])), 1) for i in range(0, len(tiles), cols)]
    sheet = np.concatenate(rows_img, 0) if rows_img else np.full((TH, TW * cols, 3), 255, np.uint8)
    cv2.imwrite(str(out / f"{a.name}.png"), sheet)
    (out / f"{a.name}.json").write_text(json.dumps({"rows_file": str(a.rows), "kind": a.kind, "candidates": len(rows),
                                                    "seed": a.seed, "tiles": meta}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"sheet": str(out / f"{a.name}.png"), "size": list(sheet.shape[:2]), "candidates": len(rows)}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("meiki")
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--shards", type=int, default=0)
    p = sub.add_parser("run")
    p.add_argument("--source", required=True, choices=("hook", "desk_hook", "desk_synth"))
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--meiki", required=True)
    p.add_argument("--targets")
    p.add_argument("--measure", default="ink", choices=("ink", "box"))
    p.add_argument("--pool", help="the page pool (hook, desk_hook): the positives' glyph cells for where_of")
    p.add_argument("--splits", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--workers", type=int, default=5)
    p = sub.add_parser("sheet")
    p.add_argument("--rows", required=True)
    p.add_argument("--root", nargs="+", required=True, help="SOURCE=DIR for each source in the rows")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--kind", default="uncovered", choices=("uncovered", "uncovered_junk_only", "weak"))
    p.add_argument("--source")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--max-score", type=float, default=2.0)
    p.add_argument("--exclude-titles", default="")
    p.add_argument("--distinct", action="store_true", help="draw distinct elements (one row per element)")
    p.add_argument("--min-frac", type=float, default=0.0, help="the unlabelled spans' share of the box, at least")
    p.add_argument("--max-frac", type=float, default=1.0, help="... at most")
    p = sub.add_parser("edits", help="the label-edit cache: the 'apart' rows as notdlg positives (edit_candidates)")
    p.add_argument("--rows", nargs="+", required=True, help="hook coverage rows (run --source hook)")
    p.add_argument("--shards", required=True, help="the hook page shards the rows were measured on")
    p.add_argument("--meiki", required=True, help="the pools' MeikiOCR cache")
    p.add_argument("--targets", help="the target cache the rows were measured with")
    p.add_argument("--pool", required=True, help="the page pool (for the per-split counts)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    {"meiki": cmd_meiki, "run": cmd_run, "sheet": cmd_sheet, "edits": cmd_edits}[a.cmd](a)


if __name__ == "__main__":
    main()

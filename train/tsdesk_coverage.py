#!/usr/bin/env python3
"""textspot-desktop round 2 (#173): the label coverage of the detector's hooked and desktop training sources. Text that
no positive and no ignore area covers is background to the trainer (train/tsdesk_det.targets); this finds it with
MeikiOCR's detector boxes (score >= 0.5), each refined to ink as the pools refine them (synth/textspot6_pages.py
meiki: textspot2_labels.refine_h / refine_t, refined_of P1).

--measure ink (the default): the ink inside the box (ink_in: train/tsdesk_targets.ink_box's stroke rule on the stored
grey frame, without its ring-noise term) that lies in no positive and no ignore area (each grown by 1 px); the box
counts when that unlabelled ink holds at least INK_SHARE of the box's ink and its extent is at least 0.4 x the box's
size across along the line and 0.3 x it across (about half a glyph): text the trainer sees as background, whatever
the box's own offset. --measure box: desktop-data 1b v3's geometric test (desktopdata/realplus.uncovered_spans: a run
of unlabelled columns at least 0.4 x the box's height long, or of unlabelled rows at least 0.2 x it tall), spans
thinner than MIN_SPAN px dropped; a box offset from its line by a quarter of its height fails it although its line is
labelled (MeikiOCR's PSP boxes), so it overcounts.

  tsdesk_coverage.py meiki --manifest M --root DIR --out CACHE [--workers 5] [--shards N]       (one heavy slot)
      MeikiOCR's detector (teacher/engines.Meiki: its pinned files, threshold 0.2) on every sample of a shard set
      (#97's frames or textspot6-pages shards; the stored grey frame in three channels), each box refined as the pools
      refine theirs; CACHE/<shard>.jsonl, one line per sample {"shard", "i", "meiki": [{"raw", "score", "box",
      "vertical"}]} (the pools' format); a finished shard is skipped
  tsdesk_coverage.py run --source hook|desk_hook|desk_synth --manifest M --root DIR --meiki CACHE
                     [--targets DIR] [--splits train] [--measure ink|box] --out COUNTS.json --rows ROWS.jsonl
                     [--workers 5]
      the labels of each sample as the trainer sees them (on hook and desk_hook the positives patched with the
      --targets cache, train/tsdesk_targets.patch_sample), under recipe base and under recipe junk (tsdesk_det.
      strip_weak), against each MeikiOCR box. hook: CACHE = the pools' cache (synth/textspot6_pages.py meiki: one
      {"title", "page", "meiki": [{"raw", "score", ...}]} line per page, the boxes the labels were built from);
      desk_hook: the same cache, each box mapped through its page's capture (train/tsdesk_targets.capture_of);
      desk_synth: a cache of `meiki` above. COUNTS holds counts only; ROWS (private) each uncovered box and each
      weak MeikiOCR ignore area the junk recipe turns to background, with its sample, for the sheets
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
    """(spans, share): the unlabelled ink of box b (the bounding box of its ink outside every label, as [[x0, y0, x1,
    y1]], or [] when it is under the INK_* bars) and that ink's share of the box's ink."""
    H, W = g.shape
    c = clip(b, W, H)
    if c is None or c[2] - c[0] < 2 or c[3] - c[1] < 2:
        return [], 0.0
    ink = ink_in(g, c)
    n = int(ink.sum())
    if not n:
        return [], 0.0
    u = ink & ~labelled_in(c, boxes)
    share = float(u.sum()) / n
    if share < INK_SHARE:
        return [], share
    rows, cols = np.flatnonzero(u.any(1)), np.flatnonzero(u.any(0))
    vertical = (c[3] - c[1]) > 2.5 * (c[2] - c[0])
    across = (c[2] - c[0]) if vertical else (c[3] - c[1])
    along_ext = (rows[-1] - rows[0] + 1) if vertical else (cols[-1] - cols[0] + 1)
    across_ext = (cols[-1] - cols[0] + 1) if vertical else (rows[-1] - rows[0] + 1)
    if along_ext < INK_ALONG * across or across_ext < INK_ACROSS * across:
        return [], share
    return [[int(c[0] + cols[0]), int(c[1] + rows[0]), int(c[0] + cols[-1] + 1), int(c[1] + rows[-1] + 1)]], share


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

def sample_coverage(smp, meiki, src, cache=None, grey=None, measure="ink"):
    """(counts, rows) of one sample. meiki: [{"raw", "score", "box", ...}] MeikiOCR boxes in the sample's px (box: the
    raw box refined to ink; each may carry "native": its box on the native page); src: the trainer's source name
    (hook, desk_hook, desk_synth); cache: a target cache for hook / desk_hook; grey: the sample's grey frame (needed
    by measure "ink", the default; "box": desktop-data 1b v3's geometric test)."""
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
            whole = sh_b >= 0.9 if measure == "ink" else sp_b == [clip(b, W, H)]
            c["strong_uncovered"] += 1
            c["strong_uncovered_whole"] += whole
            c[f"strong_uncovered_{band(sc)}"] += 1
            rows.append(dict(base, kind="uncovered", spans=sp_b, frac=round(sh_b, 3), whole=bool(whole)))
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
    src, root, s, meiki, splits, measure = job
    base = Path(root) / s.get("dir", "") / s["name"]
    samples = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["samples"]
    data = np.load(base.with_suffix(".npy"), mmap_mode="r") if measure == "ink" else None
    if src in ("hook", "desk_hook"):
        pm = _load_meiki(Path(meiki) / f"{s.get('title') or samples[0]['title']}.jsonl", "page")
    else:
        pm = _load_meiki(Path(meiki) / f"{s['name']}.jsonl", "i")
    tot = Counter()
    by_title = defaultdict(Counter)
    keys = defaultdict(set)
    rows = []
    for i, smp in enumerate(samples):
        if splits and smp.get("split") and smp["split"] not in splits:
            continue
        if src == "hook":
            mk = pm.get(smp["page"])
        elif src == "desk_hook":
            native = pm.get(smp["page"])
            cap = TT.capture_of(smp) if native is not None else None
            if cap is None:
                tot["no_capture" if native is not None else "no_meiki"] += 1
                continue
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
        c, rs = sample_coverage(smp, mk, src, _RUN.get("cache"), g, measure)
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
    jobs = [(a.source, a.root, s, a.meiki, splits, a.measure) for s in shards]
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
           "meiki": str(a.meiki), "targets": a.targets,
           "targets_sha256": sha256_file(Path(a.targets) / "targets.json") if a.targets else None,
           "rule": {"measure": a.measure, "strong": STRONG, "row_run": ROW_RUN, "min_span_px": MIN_SPAN,
                    "grid_px": GRID, "ink": {"share": INK_SHARE, "along": INK_ALONG, "across": INK_ACROSS,
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
    a = ap.parse_args(argv)
    {"meiki": cmd_meiki, "run": cmd_run, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()

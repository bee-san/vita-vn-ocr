#!/usr/bin/env python3
"""Console crops (issue #129, VOCR6 #125 T3): the line crops the console's finder cuts from the hooked pages, labelled
from the verified lines they hold, as a pool for crop-in-the-loop training.

  console_crops.py build --pagepool DIR --pages DIR --finder SPEC --out OUT
                         [--sets train] [--variants base,expanded-8] [--views console,native] [--jitter 2]
                         [--titles T,...] [--procs 5]

For every page of the chosen splits that has dialogue regions (the page pool's regions, eval/finder_tighten's rule),
every chosen region variant and view (console: Vita frames as they are, a PSP frame upscaled 2x bilinear as
Adrenaline shows it; native: every frame at its own size, PSP only), the finder (eval/pagepath.py's --finder SPEC,
e.g. `rubi,fix=5`, #129's default) returns its line boxes exactly as Rubi cuts them (TEXTBOX_RECOVERY, the short-line
probe, the refinement on PSP titles). Each box is matched to the region's verified lines by their inked character
cells (eval/pagepath.ink_points: the centroid of each cell's high-contrast pixels):
  whole    the box holds every inked cell of one line and none of another: label = the line's text
  partial  the box holds some of one line's inked cells and none of another: left out (a cell whose ink centroid
           lies outside the box can still show in it, so no exact label can be given)
  empty    the box holds no inked cell and overlaps no renderer text box of the page (furigana rows, name plates,
           unverified rows are text): label "" (a negative: a frame, an icon, scenery)
  merged   the box holds cells of two lines: left out
Each whole crop also gets `--jitter` variants: the box moved by random amounts (top and bottom -0.15 to +0.35 of its
height, left and right -0.3 to +0.8 of its height, seeded by the crop), kept only when it still holds the line's
whole ink extent (the bounding box of its cells' high-contrast pixels) and no other line's inked cell, clipped to
the region; its dark flag is synth/common.line_dark's. Neighbouring lines' fragments may show in it, as they do in
the finder's own boxes.

Crops are stored as the console hands them to the recognizer: integer luma, a dark-text crop inverted (the finder's
dark flag, as Rubi's VOCR_INVERT; meta `inverted`). Identical crops of one page (the same pixels and polarity from
two region variants) are stored once. Pools (OUT/<split>/<pool>/<title>.npz, train/train2.py's shard format:
data, offsets, shapes, labels, meta): console, console_empty, console_jitter, and the same with _native for PSP
pages at 480x272. A crop whose label has a character outside data/vocab_vn_5631.txt, or that
synth/render.crop_ctc_feasible rejects, goes to the matching drop_<pool> pool instead. OUT/manifest.json records
the inputs' and every file's sha256 and the counts.

Everything read and written here is the owner's private, copyrighted game data: it stays under $HERD and is never
committed.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import pagepath as PP  # noqa: E402

SCHEMA = "vocr6-console-crops/1"
VOCAB = ROOT / "data" / "vocab_vn_5631.txt"
UNK = "〓"
JITTER_TRIES = 6


def sha256_file(path):
    return PP.sha256_file(path)


def ink_extents(img, cells, boxes):
    """Per reference line the bounding box [x0, y0, x1, y1) of its cells' high-contrast pixels
    (eval/pagepath.ink_points' test), or None."""
    H, W = img.shape
    c = PP.contrast3(img)
    out = []
    for cl, b in zip(cells, boxes):
        bx0, by0 = max(0, int(b[0])), max(0, int(b[1]))
        bx1, by1 = min(W, int(np.ceil(b[0] + b[2]))), min(H, int(np.ceil(b[1] + b[3])))
        sub = c[by0:by1, bx0:bx1]
        t = max(40, 2 * int(np.percentile(sub, 99)) // 5) if sub.size else 255
        ext = None
        for x0, y0, x1, y1 in cl:
            xa, ya = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
            xb, yb = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
            if xb <= xa or yb <= ya:
                continue
            m = c[ya:yb, xa:xb] >= t
            if m.sum() < 2:
                continue
            ys, xs = np.nonzero(m)
            e = [xa + int(xs.min()), ya + int(ys.min()), xa + int(xs.max()) + 1, ya + int(ys.max()) + 1]
            ext = e if ext is None else [min(ext[0], e[0]), min(ext[1], e[1]), max(ext[2], e[2]), max(ext[3], e[3])]
        out.append(ext)
    return out


def overlaps(box, xyxy):
    x, y, w, h = box
    return min(x + w, xyxy[2]) > max(x, xyxy[0]) and min(y + h, xyxy[3]) > max(y, xyxy[1])


def classify(box, points):
    """(kind, line index, [cell indices]) of a finder box against the region's inked cell points."""
    held = []
    for li, pl in enumerate(points):
        idx = [ci for ci, p in enumerate(pl) if p is not None and PP.inside(box, p)]
        if idx:
            held.append((li, idx))
    if not held:
        return "empty", None, []
    if len(held) > 1:
        return "merged", None, []
    li, idx = held[0]
    inked = [ci for ci, p in enumerate(points[li]) if p is not None]
    return ("whole" if len(idx) == len(inked) else "partial"), li, idx


def jitter_boxes(box, ink, avoid_pts, W, H, n, seed):
    x, y, w, h = box
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n * JITTER_TRIES):
        if len(out) == n:
            break
        t, b = rng.uniform(-0.15, 0.35, 2) * h
        lft, rgt = rng.uniform(-0.3, 0.8, 2) * h
        x0, y0 = max(0, int(round(x - lft))), max(0, int(round(y - t)))
        x1, y1 = min(W, int(round(x + w + rgt))), min(H, int(round(y + h + b)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        jb = (x0, y0, x1 - x0, y1 - y0)
        if jb == tuple(box) or jb in out:
            continue
        holds = ink is not None and x0 <= ink[0] and y0 <= ink[1] and x1 >= ink[2] and y1 >= ink[3]
        if holds and not any(PP.inside(jb, p) for p in avoid_pts):
            out.append(jb)
    return out


def title_job(args):
    (tid, pagepool, pages_dir, spec, sets, variants, views, jitter, work, max_pages) = args
    from common import line_dark
    from render import crop_ctc_feasible
    vocab = {c for c in VOCAB.read_text(encoding="utf-8").split("\n") if c and c != UNK}
    cfg = PP.parse_finder(spec)
    exe, dmeta = PP.build_driver(cfg, Path(work) / "bin")
    rows = PP.read_jsonl(Path(pagepool) / f"{tid}.jsonl")
    if max_pages:
        rows = rows[:max_pages]
    pools = defaultdict(lambda: ([], [], []))           # (split, pool) -> crops, labels, metas
    counts = Counter()
    for view in views:
        frames = PP.Frames(pages_dir, view)
        units = []
        for r in rows:
            if r["split"] not in sets or not r.get("regions"):
                continue
            if view == "native" and r["platform"] != "psp":
                continue
            by_id = {ln["id"]: ln for ln in r["lines"]}
            groups = []
            for g in r["regions"]:
                if g["variant"] not in variants:
                    continue
                key = tuple(g["lines"])
                if key not in groups:
                    groups.append(key)
                lines = sorted((by_id[i] for i in g["lines"]), key=lambda ln: (ln["box"][1], ln["box"][0]))
                units.append({"uid": f"{tid}:{r['page']}:{groups.index(key)}:{g['variant']}", "split": r["split"],
                              "title": tid, "page": r["page"], "platform": r["platform"], "variant": g["variant"],
                              "image_sha256": r.get("image_sha256"),
                              "region": [int(v) for v in g["region"]],
                              "text_boxes": [b["xyxy"] for b in r.get("boxes") or []],
                              "lines": [{"id": ln["id"], "text": ln["text"], "box": ln.get("box_main") or ln["box"],
                                         "cells": PP.cells_of(ln)} for ln in lines]})
        seen = set()
        batch = []

        def flush():
            res = PP.run_driver(exe, [(img, PP.FINDER_MODES[cfg["mode"]], cfg["fix"], cfg["probe"],
                                       PP.refine_flags(cfg, u["platform"])) for u, img, s, o in batch])
            for (u, img, s, o), (rc, found, probed, bx) in zip(batch, res):
                harvest(u, img, s, o, bx)
            batch.clear()

        def store(u, img, box, dark, label, pool, meta):
            x, y, w, h = box
            c = np.ascontiguousarray(img[y:y + h, x:x + w])
            c = (255 - c) if dark else c
            key = (u["page"], hashlib.blake2b(c.tobytes() + bytes(c.shape[:1]), digest_size=12).hexdigest(),
                   c.shape, pool)
            if key in seen:
                counts[f"{view}:{pool}:duplicate"] += 1
                return
            seen.add(key)
            oov = sorted({ch for ch in label if ch not in vocab})
            ok = crop_ctc_feasible(c.shape, label) if label else True
            name = (pool if view == "console" else pool + "_native")
            if oov or not ok:
                name = "drop_" + name
            meta = dict(meta, title=tid, page=u["page"], platform=u["platform"], view=view, variant=u["variant"],
                        box=[int(v) for v in box], inverted=bool(dark), oov=oov, ctc_ok=bool(ok))
            crops, labels, metas = pools[(u["split"], name)]
            crops.append(c)
            labels.append(label)
            metas.append(meta)
            counts[f"{view}:{name}"] += 1

        def harvest(u, img, s, o, bx):
            H, W = img.shape
            cells, rboxes = PP.ref_geometry(u, s, o)
            points = PP.ink_points(img, cells, rboxes)
            ink = ink_extents(img, cells, rboxes)
            ox, oy = o
            text_boxes = [[b[0] * s - ox, b[1] * s - oy, b[2] * s - ox, b[3] * s - oy] for b in u["text_boxes"]]
            for bi, (x, y, w, h, dark) in enumerate(bx):
                kind, li, idx = classify((x, y, w, h), points)
                counts[f"{view}:box:{kind}"] += 1
                if kind in ("merged", "partial"):
                    continue
                if kind == "empty":
                    if any(overlaps((x, y, w, h), tb) for tb in text_boxes):
                        counts[f"{view}:empty_on_text"] += 1
                        continue
                    store(u, img, (x, y, w, h), dark, "", "console_empty", {"kind": "empty"})
                    continue
                line = u["lines"][li]
                label = line["text"]
                meta = {"kind": kind, "line": line["id"]}
                store(u, img, (x, y, w, h), dark, label, "console", meta)
                avoid = [p for lj, pl in enumerate(points) if lj != li for p in pl if p is not None]
                seed = int(hashlib.sha256(f"{u['uid']}:{view}:{bi}".encode()).hexdigest()[:8], 16)
                for k, jb in enumerate(jitter_boxes((x, y, w, h), ink[li], avoid, W, H, jitter, seed)):
                    jd = line_dark(img, jb)
                    store(u, img, jb, jd, label, "console_jitter", dict(meta, jitter=k + 1, finder_box=[x, y, w, h]))

        for u, img, s, o in PP.unit_images(units, frames):
            batch.append((u, img, s, o))
            if len(batch) >= 256:
                flush()
        if batch:
            flush()
    return tid, {f"{k[0]}/{k[1]}": v for k, v in pools.items()}, dict(counts), dmeta


def write_npz(path, crops, labels, metas):
    """train/train2.py's shard format (data/import_hooked.py's): data, offsets, shapes, labels, meta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shapes = np.asarray([g.shape for g in crops], np.int32).reshape(-1, 2)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    data = np.concatenate([g.ravel() for g in crops]) if crops else np.zeros(0, np.uint8)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=data.astype(np.uint8), offsets=offsets, shapes=shapes,
                        labels=np.asarray(json.dumps(labels, ensure_ascii=False)),
                        meta=np.asarray(json.dumps(metas, ensure_ascii=False)))
    os.replace(tmp, path)


def read_npz(path):
    """[(meta, label, crop)] of a pool file."""
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels, metas = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    return [(m, lab, data[offs[i]:offs[i + 1]].reshape(int(shapes[i][0]), int(shapes[i][1])))
            for i, (lab, m) in enumerate(zip(labels, metas))]


def cmd_build(a):
    man = json.loads((Path(a.pagepool) / "manifest.json").read_text(encoding="utf-8"))
    titles = sorted(man["titles"]) if not a.titles else a.titles.split(",")
    sets, variants, views = set(a.sets.split(",")), set(a.variants.split(",")), a.views.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, a.pagepool, a.pages, a.finder, sets, variants, views, a.jitter, a.work, a.max_pages)
            for t in titles]
    doc = {"schema": SCHEMA, "kind": "vocr6-console-crops", "finder": a.finder, "sets": sorted(sets),
           "variants": sorted(variants), "views": views, "jitter": a.jitter,
           "pagepool_manifest_sha256": sha256_file(Path(a.pagepool) / "manifest.json"),
           "console_crops_sha256": sha256_file(__file__), "pagepath_sha256": sha256_file(PP.__file__),
           "vocab_sha256": sha256_file(VOCAB), "titles": {}, "files": {}, "counts": {}}
    total = Counter()
    with ProcessPoolExecutor(a.procs) as ex:
        for tid, pools, counts, dmeta in ex.map(title_job, jobs):
            doc["driver"] = dmeta
            doc["titles"][tid] = counts
            total.update(counts)
            for key, (crops, labels, metas) in pools.items():
                path = out / key / f"{tid}.npz"
                write_npz(path, crops, labels, metas)
                doc["files"][str(path.relative_to(out))] = {"crops": len(crops), "sha256": sha256_file(path)}
            print(tid, {k: v for k, v in sorted(counts.items()) if ":box:" in k or k.endswith(":console")},
                  flush=True)
    doc["counts"] = dict(sorted(total.items()))
    (out / "manifest.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("console crops: wrote", out, json.dumps({k: v for k, v in doc["counts"].items() if ":box:" not in k}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pagepool", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--finder", required=True)
    p.add_argument("--sets", default="train")
    p.add_argument("--variants", default="base,expanded-8")
    p.add_argument("--views", default="console,native")
    p.add_argument("--jitter", type=int, default=2)
    p.add_argument("--titles")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--max-pages", type=int, default=0, help="the first N pages of each title (smoke runs)")
    p.add_argument("--work", default="/tmp/console_crops")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cmd_build(a)


if __name__ == "__main__":
    main()

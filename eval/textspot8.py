#!/usr/bin/env python3
"""textspot8 (vocr6-spotter2, docs/textspot8.md): MeikiOCR's text detector as the reference line spotter, scored the
way textspot6 scores its spotters, so that every number pairs with textspot6's (#128) and GRAD + refine's.

  textspot8.py meiki-spot --set dev|hval|tsdev --out RAW.jsonl [--workers 5] [--threads 1] [--limit N]
      MeikiOCR's detector (teacher/engines.Meiki: the pinned meiki.text.detect.v0.1 ONNX, MeikiOCR 0.3.5's own
      preprocessing, fit to 960 x 544 with INTER_LINEAR; BGR input as teacher/label.py and textspot6's cache feed it)
      on every frame of a set, in colour and in the set's view (eval/textspot6.py's frames and keys):
        dev    #119's whole-dialogue units' frames, console view: a PSP frame upscaled 2x bilinear on its colour
        hval   the hval pages of vocr6-ingest's page pool, views vita, psp2x (2x bilinear on the colour) and psp
        tsdev  textspot's 188 development frames
      One JSON line per frame and view: its key, W, H, every box with score > 0.2 ([x0, y0, x1, y1] clipped to the
      frame, as Meiki.ocr casts them) with its score, and the detector's host time (ms).
  textspot8.py meiki-decode --raw RAW.jsonl --out SPOT.jsonl [--thr 0.5] [--decode raw|pad|ink] [--pad N]
      a textspot6 spot file of Meiki's boxes (eval/textspot6.py's schema: boxes [x, y, w, h, vertical, conf, dark,
      block, tag, tconf] and rboxes), so eval/textspot6.py pagepath-boxes / hval-boxes / tsdev read it unchanged:
        boxes     the boxes with score > thr; vertical as Meiki decides it (taller than wide); decode raw = Meiki's
                  box, pad = grown by --pad px (default 4 at Vita / console scale, 2 for a native PSP view) on every
                  side and clipped, ink = refined to ink as textspot2 refines meiki boxes (textspot2_labels.refine_h,
                  refine_t, refined_of P1) and then grown by the pad; conf = clip(round(127 score), 1, 127); dark =
                  synth/common.line_dark on the frame's grey; tag 0 (Meiki has no classes), tconf 0
        order     textspot_twin.group_and_order's blocks, order and pick (the order textspot6's twin gives its own
                  boxes; eval/pagepath.py reads a region's boxes in this order)
        rboxes    VOCR_REFINE_DEFAULT of the horizontal boxes on the frame's grey (eval/textspot6.refined): the
                  det+refine decode
  textspot8.py meiki-speed --n 100 [--threads 1,5] --out SPEED.json
      the detector's host time per 960 x 544 frame (dev frames), its file size and parameter count

Everything read here is private (game frames); RAW and SPOT files hold coordinates only and stay local.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MEIKI_THR_CACHE = 0.2            # every box above it is kept in RAW (textspot2's and textspot6's cache threshold)
MEIKI_THR = 0.5                  # MeikiOCR's own default (run_ocr det_threshold) and teacher/engines.Meiki's
PAD_CONSOLE, PAD_PSP = 4, 2      # eval's crop pad: 4 px on a Vita (console) frame, 2 px on a native PSP frame


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ----------------------------------------------------------------------------- colour frames, textspot6's keys

def dev_colour_frames(path=None):
    """[(key, rgb, grey)] of #119's dev units' frames in the console view, keyed as eval/textspot6.dev_frames: the
    colour frame (a PSP frame upscaled 2x bilinear on its colour, Meiki's input) and the grey frame textspot6's
    dev_frames gives (a PSP frame upscaled 2x on its grey: what eval/pagepath.py's regions are cut from)."""
    import glob
    from collections import defaultdict

    import cv2
    import geometry as G
    import textspot6 as T6E

    from common import to_gray
    units = T6E.dev_units(path or T6E.DEV_UNITS)
    by = defaultdict(list)
    for u in units:
        by[T6E.dev_frame_key(u)].append(u)
    for key, us in sorted(by.items()):
        u0 = us[0]
        cands = [p for d in T6E.SHOT_DIRS[u0["src_kind"]] for p in glob.glob(f"{d}/{u0['shot']}.*")]
        rgb = None
        for p in cands:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            if u0["src_kind"] == "hires":
                rgb = np.ascontiguousarray(G.scale_image(rgb, float(u0["scale"])))
            break
        if rgb is None:
            raise FileNotFoundError(f"{u0['shot']}: no screenshot")
        g = to_gray(rgb)
        for u in us:
            x0, y0, x1, y1 = u["region"]
            if g[y0:y1, x0:x1].shape != u["image"].shape or not (g[y0:y1, x0:x1] == u["image"]).all():
                raise SystemExit(f"{u['unit']}: the rebuilt frame's region is not the unit's image")
        if u0["kind"] == "psp":
            g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        yield ({"frame": key, "view": "psp2x" if u0["kind"] == "psp" else "vita", "shot": u0["shot"],
                "vn": u0["vn"]}, np.ascontiguousarray(rgb), np.ascontiguousarray(g))


def hval_colour_frames(split="hval", titles=None):
    """[(key, rgb, grey)] of the page pool's pages of a split in their views, keyed as eval/textspot6.hval_frames."""
    import cv2
    import textspot6 as T6E

    from common import to_gray
    for r in T6E.hval_records(T6E.POOL, split, titles):
        if split == "test" or r["split"] == "test":
            raise SystemExit("textspot8: never reads a test page here")
        bgr = cv2.imread(str(Path(T6E.PAGES) / r["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(r["image"])
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        base = {"title": r["title_id"], "page": r["page"]}
        if r["platform"] == "psp":
            up = np.ascontiguousarray(cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR))
            yield dict(base, view="psp2x"), up, to_gray(up)
            yield dict(base, view="psp"), rgb, to_gray(rgb)
        else:
            yield dict(base, view="vita"), rgb, to_gray(rgb)


def tsdev_colour_frames():
    import textspot as TSE

    from common import to_gray
    rows = read_jsonl(ROOT / "eval" / "manifests" / "textspot-dev.jsonl")
    dirs = list(TSE.SHOT_DIRS)
    for r in rows:
        rgb = np.ascontiguousarray(TSE.scale_frame(r, TSE.read_rgb(TSE.source_path(r, dirs))))
        yield {"frame": r["id"], "vn": r["vn"], "part": r["part"], "view": r["part"]}, rgb, to_gray(rgb)


COLOUR_FRAMES = {"dev": dev_colour_frames, "hval": hval_colour_frames, "tsdev": tsdev_colour_frames}


# ----------------------------------------------------------------------------- MeikiOCR's detector

_W = {}


def _init_meiki(threads):
    import engines as E
    _W["meiki"] = E.Meiki(det_thr=MEIKI_THR_CACHE, n_threads=threads)


def meiki_boxes(meiki, rgb):
    """MeikiOCR's detector on an RGB frame, fed BGR (as teacher/label.py and textspot6's cache feed it):
    [([x0, y0, x1, y1], score)] with score > the cache threshold, clipped and cast as Meiki.ocr does, in Meiki's
    order (stable by top edge). (textspot2_labels.meiki_detect, which takes the BGR array.)"""
    import textspot2_labels as L2
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return L2.meiki_detect(meiki, bgr)


def _meiki_one(job):
    key, rgb, _g = job
    t0 = time.perf_counter()
    found = meiki_boxes(_W["meiki"], rgb)
    ms = (time.perf_counter() - t0) * 1000.0
    return {**key, "W": int(rgb.shape[1]), "H": int(rgb.shape[0]), "ms": round(ms, 2),
            "meiki": [[*b, round(float(s), 4)] for b, s in found]}


def cmd_meiki_spot(args):
    import multiprocessing as mp

    import engines as E
    jobs = list(COLOUR_FRAMES[args.set]())
    if args.limit:
        jobs = jobs[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    det = E.model_path("meiki_det")
    meta = {"set": args.set, "frames": len(jobs), "detector": det.name, "detector_sha256": sha256_file(det),
            "pin": list(E.PINS["meiki_det"][0]), "thr_kept": MEIKI_THR_CACHE, "input": "BGR (cv2's order)",
            "threads_per_worker": args.threads, "workers": args.workers,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        with mp.get_context("fork").Pool(args.workers, initializer=_init_meiki, initargs=(args.threads,)) as pool:
            for rec in pool.imap(_meiki_one, jobs, chunksize=4):
                f.write(json.dumps(rec) + "\n")
    os.replace(tmp, out)
    print(f"textspot8 meiki-spot finished: {args.set} {len(jobs)} frames {time.time() - t0:.0f} s -> {out}",
          flush=True)


def load_raw(path):
    rows = read_jsonl(path)
    if not rows or "meta" not in rows[0]:
        raise SystemExit(f"{path}: not a textspot8 meiki-spot file")
    return rows[0]["meta"], rows[1:]


# ----------------------------------------------------------------------------- Meiki's boxes as a spot file

def grow(box, pad, W, H):
    x0, y0, x1, y1 = box
    return [max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)]


def decode_meiki(rec, rgb, grey, thr=MEIKI_THR, decode="raw", pad=None):
    """One frame's spot row of Meiki's boxes (score > thr) in a decode (module docstring)."""
    import textspot6 as T6E

    import textspot2_labels as L2
    import textspot_twin as S
    from common import line_dark
    W, H = rec["W"], rec["H"]
    if pad is None:
        pad = PAD_PSP if rec.get("view") == "psp" else PAD_CONSOLE
    cand = []
    for x0, y0, x1, y1, score in rec["meiki"]:
        if score <= thr:
            continue
        raw = [int(x0), int(y0), int(x1), int(y1)]
        vertical = (raw[3] - raw[1]) > (raw[2] - raw[0])
        if decode == "ink":
            ref = {"raw": raw, "score": score, "ref_h": L2.refine_h(rgb, raw), "ref_t": L2.refine_t(rgb, raw)}
            box, vertical = L2.refined_of(ref, "P1")
            box = grow([int(v) for v in box], pad, W, H)
        elif decode == "pad":
            box = grow(raw, pad, W, H)
        elif decode == "raw":
            box = raw
        else:
            raise ValueError(decode)
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        conf = int(min(127, max(1, round(127 * float(score)))))
        cand.append((x0, y0, x1 - x0, y1 - y0, int(bool(vertical)), conf))
    cand = sorted(set(cand))
    dark = {b: line_dark(grey, b[:4]) for b in cand}
    ordered, blocks, pick = S.group_and_order([(*b, dark[b]) for b in cand], H)
    boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]), int(b[5]), int(b[6]), int(b[7]), 0, 0]
             for b in ordered]
    row = {k: rec[k] for k in rec if k not in ("meiki", "ms")}
    row.update(boxes=boxes, blocks=[[*k[:5], k[5], k[6], 0] for k in blocks], pick=pick,
               presence=max([b[5] for b in boxes], default=0), truncated=0, rboxes=T6E.refined(boxes, grey))
    return row


def cmd_meiki_decode(args):
    meta, rows = load_raw(args.raw)
    want = {json.dumps({k: r[k] for k in r if k not in ("meiki", "ms", "W", "H")}, sort_keys=True): r for r in rows}
    out = Path(args.out)
    tmp = Path(str(out) + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": {"blob": "meiki:" + meta["detector"], "blob_sha256": meta["detector_sha256"],
                                     "set": meta["set"], "frames": len(rows), "twin": "eval/textspot8.decode_meiki",
                                     "thr": args.thr, "decode": args.decode, "pad": args.pad, "raw": str(args.raw),
                                     "raw_sha256": sha256_file(args.raw),
                                     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}) + "\n")
        for key, rgb, grey in COLOUR_FRAMES[meta["set"]]():
            rec = want.get(json.dumps(key, sort_keys=True))
            if rec is None:
                continue
            f.write(json.dumps(decode_meiki(rec, rgb, grey, args.thr, args.decode, args.pad)) + "\n")
            n += 1
    if n != len(rows):
        raise SystemExit(f"meiki-decode: {n} frames rebuilt for {len(rows)} rows")
    os.replace(tmp, out)
    print(f"textspot8 meiki-decode finished: {meta['set']} {n} frames thr {args.thr} {args.decode} -> {out}",
          flush=True)


# ----------------------------------------------------------------------------- crop-geometry diagnostics

def _floor_ceil_box(x0, y0, x1, y1, W, H):
    import math
    a0, b0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    a1, b1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
    return [a0, b0, a1 - a0, b1 - b0] if a1 > a0 and b1 > b0 else None


def oracle_rows(units_path, pad_x, pad_y):
    """eval/pagepath.py --boxes rows: each reference line's own box (pagepath.ref_geometry's line box: box_main for a
    furigana line) grown by pad_x / pad_y px at the view's scale (console: a PSP unit at 2x), clipped to the region.
    The recognizer's floor at a chosen crop margin (pagepath's own --finder oracle uses the line record's pad, else
    4 px Vita / 2 px PSP native)."""
    import pagepath as PP
    units, _ = PP.load_units(units_path)
    out = []
    for u in units:
        if u.get("image") is not None:                       # a dev unit: region pixels, its image
            s = 2 if u["platform"] == "psp" else 1
            H, W = (v * s for v in u["image"].shape)
            origin = (0, 0)
        else:
            s = 2 if u["platform"] == "psp" else 1
            x0, y0, x1, y1 = (v * s for v in u["region"])
            W, H, origin = x1 - x0, y1 - y0, (x0, y0)
        _, boxes = PP.ref_geometry(u, s, origin)
        bx = []
        for x, y, w, h in boxes:
            b = _floor_ceil_box(x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y, W, H)
            if b is not None:
                bx.append(b)
        out.append({"unit": u["uid"], "boxes": bx})
    return out


def cmd_oracle_boxes(args):
    sys.path.insert(0, str(Path(args.pagepath).resolve().parent))
    rows = oracle_rows(args.units, args.pad_x, args.pad_y)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"textspot8 oracle-boxes written: {len(rows)} units pad {args.pad_x}/{args.pad_y} -> {args.out}", flush=True)


def cmd_shift_boxes(args):
    """A pagepath rows file's boxes (its boxes_xywh, dark kept) grown by dx / dy px on each side (negative:
    shrunk; a box never shrinks below 2 px), as eval/pagepath.py --boxes rows. Diagnostic only."""
    n = 0
    with open(args.rows, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            bx = []
            for (x, y, w, h), dk in zip(r["boxes_xywh"], r["dark"]):
                nx, ny = x - args.dx, y - args.dy
                nw, nh = w + 2 * args.dx, h + 2 * args.dy
                if nw < 2:
                    nx, nw = x + (w - 2) // 2, 2
                if nh < 2:
                    ny, nh = y + (h - 2) // 2, 2
                bx.append([max(0, nx), max(0, ny), nw + min(0, nx), nh + min(0, ny), int(dk)])
            g.write(json.dumps({"unit": r["uid"], "boxes": bx}) + "\n")
            n += 1
    print(f"textspot8 shift-boxes written: {n} units dx {args.dx} dy {args.dy} -> {args.out}", flush=True)


# ----------------------------------------------------------------------------- size and host speed

def onnx_params(path):
    import onnx
    m = onnx.load(str(path))
    n = 0
    for t in m.graph.initializer:
        n += int(np.prod(t.dims)) if len(t.dims) else 1
    return n


def cmd_meiki_speed(args):
    import engines as E
    jobs = list(dev_colour_frames())[:args.n]
    det = E.model_path("meiki_det")
    out = {"detector": det.name, "sha256": sha256_file(det), "bytes": det.stat().st_size, "frames": len(jobs),
           "host": None, "threads": {}}
    try:
        out["params"] = onnx_params(det)
    except Exception as e:                      # onnx not installed in this venv
        out["params"] = None
        out["params_error"] = str(e)[:200]
    try:
        import hostinfo
        out["host"] = hostinfo.host_label()
    except Exception:
        pass
    for th in [int(t) for t in args.threads.split(",")]:
        m = E.Meiki(det_thr=MEIKI_THR_CACHE, n_threads=th)
        meiki_boxes(m, jobs[0][1])                                       # warm-up
        ts = []
        for _k, rgb, _g in jobs:
            t0 = time.perf_counter()
            meiki_boxes(m, rgb)
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        out["threads"][str(th)] = {"mean_ms": round(float(ts.mean()), 1), "median_ms": round(float(np.median(ts)), 1),
                                   "p90_ms": round(float(np.percentile(ts, 90)), 1)}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"textspot8 meiki-speed written: {args.out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("meiki-spot")
    a.add_argument("--set", required=True, choices=sorted(COLOUR_FRAMES))
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=5)
    a.add_argument("--threads", type=int, default=1)
    a.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("meiki-decode")
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--thr", type=float, default=MEIKI_THR)
    b.add_argument("--decode", default="raw", choices=("raw", "pad", "ink"))
    b.add_argument("--pad", type=int, default=None)
    c = sub.add_parser("meiki-speed")
    c.add_argument("--n", type=int, default=100)
    c.add_argument("--threads", default="1,5")
    c.add_argument("--out", required=True)
    d = sub.add_parser("oracle-boxes")
    d.add_argument("--units", required=True)
    d.add_argument("--pagepath", required=True, help="the pinned eval/pagepath.py (its units and ref_geometry)")
    d.add_argument("--pad-x", type=int, default=4)
    d.add_argument("--pad-y", type=int, default=4)
    d.add_argument("--out", required=True)
    e = sub.add_parser("shift-boxes")
    e.add_argument("--rows", required=True)
    e.add_argument("--dx", type=int, default=0)
    e.add_argument("--dy", type=int, default=0)
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"meiki-spot": cmd_meiki_spot, "meiki-decode": cmd_meiki_decode, "meiki-speed": cmd_meiki_speed,
     "oracle-boxes": cmd_oracle_boxes, "shift-boxes": cmd_shift_boxes}[args.cmd](args)


if __name__ == "__main__":
    main()

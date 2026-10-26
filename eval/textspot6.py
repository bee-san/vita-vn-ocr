#!/usr/bin/env python3
"""textspot6's evaluation (issue #128, docs/textspot6.md "Measure"): the spotter's boxes on every scored set, the
files eval/pagepath.py (vocr6-finder's page-path benchmark, #129) reads them from, and the box metrics.

  textspot6.py spot --blob B.vtsp --set hval|dev|tsdev --out SPOT.jsonl [--workers 5]
      the spotter (synth/textspot6_twin.spot, the specification; the C runtime must give the same boxes) on every
      frame of the set, one JSON line per frame and view:
        hval   vocr6-ingest's hval pages (the page pool's split "hval"); views "vita" (a Vita page as it is),
               "psp2x" (a PSP page upscaled 2x bilinear on its colour, then grey: what Rubi gets through Adrenaline)
               and "psp" (a PSP page at its own 480 x 272)
        dev    the frames of #119's whole-dialogue development units ($A/vocr5-analysis/work/units.pkl: other games
               than the corpus), rebuilt from the local screenshots (each unit's region of the rebuilt frame equals
               its stored image, checked), console view: a PSP frame upscaled 2x bilinear on its grey, as
               eval/pagepath.py upscales the units' grey regions
        tsdev  textspot's 188 development frames (eval/manifests/textspot-dev.jsonl: 81 other games, every text
               labelled with its class)
  textspot6.py pagepath-boxes --spot SPOT.jsonl --set hval|dev --out BOXES.jsonl [--tags all|dialogue]
      eval/pagepath.py's inputs: hval -> --frame-boxes {title, page, view, boxes [[x, y, w, h, dark]]} in frame px
      (a unit keeps the boxes whose centre lies in its region); dev -> --boxes {unit, boxes} in the unit's region
      px at its view scale (pagepath.dev_units' uid "dev:<i>:<unit>:<variant>"). Horizontal boxes only (the page
      path reads horizontal lines). --tags dialogue keeps the boxes of tag 0 only
  textspot6.py spot-crop --blob B.vtsp --set hval|dev --out BOXES.jsonl [--units UNITS.jsonl] [--decode det|detref]
      the crop decode: the spotter on each unit's region alone, cut at the view's scale exactly as eval/pagepath.py
      cuts it (dev: the unit's grey region, a PSP one upscaled 2x bilinear; hval: --units, eval/pagepath.py units'
      file, each region cut from the page at the view's scale) and read at the scale the blob's rule gives the
      frame it was cut from (textspot6_twin.spot(scale=f), f = scale_of(the frame's width and height); amendment 3),
      every horizontal box of the region read (no tag filter, as GRAD + refine reads every line of a region):
      eval/pagepath.py --boxes; BOXES.meta.json beside it
  textspot6.py hval-boxes --spot SPOT.jsonl --shards DIR --manifest M --out RES.json
      box P / R / F1 against the hval pages' verified lines (synth/textspot6_pages.py's hval shards: dialogue and
      name positives; "notdlg" boxes and ignore areas are left out: textspot_metrics.match_frame's ignore rule),
      recall per class, the tag of matched boxes (dialogue / name accuracy), per view, platform and title
  textspot6.py tsdev --spot SPOT.jsonl --out RES.json [--cache DIR] [--units UNITS.json] [--models N]
      textspot's whole-frame metrics on its development frames (eval/textspot.py's reading and scoring: every box
      read by the C runtime in W8I, greedy, VOCR5-N for horizontal and VOCR5-V for vertical boxes;
      textspot_metrics' P / R / F1, all-text CER, dialogue CER, the pick), plus the tagged dialogue CER: what Rubi
      shows as dialogue when it reads the boxes of tag 0 only: the dialogue references' errors with tag-0 boxes,
      plus every character of a tag-0 box that goes to another reference or to none, over the dialogue
      references' characters; and the presence check on the text-free frames

Everything read here is private (game frames); SPOT and BOXES files hold coordinates only and stay local.
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
INGEST = ART / "vocr6-ingest"
POOL = INGEST / "pagepool" / "v1"
PAGES = INGEST / "pages"
DEV_UNITS = ART / "vocr5-analysis" / "work" / "units.pkl"
SHOT_DIRS = {"vita": [ART / "realdata" / "shots" / "vita", ART / "devnative" / "shots" / "vita"],
             "psp": [ART / "realdata" / "shots" / "psp", ART / "devnative" / "shots" / "psp"],
             "hires": [ART / "realdata" / "shots" / "hires", ART / "realplus" / "shots" / "hires"]}
SETS = ("hval", "dev", "tsdev")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ----------------------------------------------------------------------------- frames

def hval_records(pool=POOL, split="hval", titles=None):
    """The page pool's records of one split (default "hval"; eval/textspot6_test.py reads "test"), optionally of
    some titles only."""
    out = []
    for f in sorted(Path(pool).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["split"] == split and (titles is None or r["title_id"] in titles):
                out.append(r)
    return out


def hval_frames(pool=POOL, pages=PAGES, split="hval", titles=None):
    """[(key, grey)] of every page of a split (default hval) in its views: key {"title", "page", "view"}."""
    import cv2

    from common import to_gray
    for r in hval_records(pool, split, titles):
        bgr = cv2.imread(str(Path(pages) / r["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(r["image"])
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        base = {"title": r["title_id"], "page": r["page"]}
        if r["platform"] == "psp":
            up = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            yield dict(base, view="psp2x"), to_gray(np.ascontiguousarray(up))
            yield dict(base, view="psp"), to_gray(rgb)
        else:
            yield dict(base, view="vita"), to_gray(rgb)


def dev_units(path=DEV_UNITS):
    return pickle.load(open(path, "rb"))["units"]


def dev_frame_key(u):
    return f"{u['shot']}|{u['src_kind']}|{round(float(u['scale']), 6)}|{u['kind']}"


def dev_frames(path=DEV_UNITS):
    """[(key, grey)] of every development unit's frame in the console view, each rebuilt from its screenshot and
    checked: every unit's region of the native rebuilt frame equals its stored image."""
    import cv2
    import geometry as G

    from common import to_gray
    units = dev_units(path)
    by = defaultdict(list)
    for u in units:
        by[dev_frame_key(u)].append(u)
    for key, us in sorted(by.items()):
        u0 = us[0]
        cands = [p for d in SHOT_DIRS[u0["src_kind"]] for p in glob.glob(f"{d}/{u0['shot']}.*")]
        g = None
        for p in cands:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            if u0["src_kind"] == "hires":
                rgb = G.scale_image(rgb, float(u0["scale"]))
            g = to_gray(np.ascontiguousarray(rgb))
            break
        if g is None:
            raise FileNotFoundError(f"{u0['shot']}: no screenshot in {SHOT_DIRS[u0['src_kind']]}")
        for u in us:
            x0, y0, x1, y1 = u["region"]
            if g[y0:y1, x0:x1].shape != u["image"].shape or not (g[y0:y1, x0:x1] == u["image"]).all():
                raise SystemExit(f"{u['unit']}: the rebuilt frame's region is not the unit's image")
        if u0["kind"] == "psp":
            g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        yield {"frame": key, "view": "psp2x" if u0["kind"] == "psp" else "vita", "shot": u0["shot"],
               "vn": u0["vn"]}, np.ascontiguousarray(g)


def tsdev_frames():
    import textspot as TSE
    rows = [json.loads(x) for x in (ROOT / "eval" / "manifests" / "textspot-dev.jsonl").read_text().splitlines()]
    dirs = list(TSE.SHOT_DIRS)
    for r in rows:
        f = TSE.load_frame(r, dirs)
        yield {"frame": r["id"], "vn": r["vn"], "part": r["part"], "view": r["part"]}, f.grey


FRAMES = {"hval": hval_frames, "dev": dev_frames, "tsdev": tsdev_frames}


# ----------------------------------------------------------------------------- the spotter

_W = {}


def refined(boxes, g):
    """det+refine: the decode's horizontal boxes, in its order, through refine_lines (VOCR_REFINE_DEFAULT) on the
    frame, as eval/textspot.refine_spot_boxes; a refined box keeps its source box's confidence and tag; vertical
    boxes unchanged. -> [[x, y, w, h, vertical, conf, dark, tag]]."""
    import lines_refine as LR
    hz = [b for b in boxes if not b[4]]
    rb, rd, org = LR.refine_lines(g, [b[:4] for b in hz], [b[6] for b in hz], LR.DEFAULT)
    out = [[*(int(v) for v in bx), 0, int(hz[o][5]), int(dk), int(hz[o][8])] for bx, dk, o in zip(rb, rd, org)]
    return out + [[*b[:7], b[8]] for b in boxes if b[4]]


def _spot_one(job):
    import textspot6_twin as T6
    key, g = job
    r = T6.spot(_W["model"], g)
    boxes = [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, b.tag, b.tconf] for b in r.boxes]
    return {**key, "W": int(g.shape[1]), "H": int(g.shape[0]), "boxes": boxes,
            "blocks": [[k.x, k.y, k.w, k.h, k.vertical, k.first, k.count, k.tag] for k in r.blocks],
            "pick": r.pick, "presence": r.presence, "truncated": r.truncated, "rboxes": refined(boxes, g)}


def write_spot(blob_path, jobs, out, set_name, workers):
    """`spot`'s file: the twin on every (key, grey) job, a meta line then one row per frame (_spot_one)."""
    import multiprocessing as mp

    import textspot6_twin as T6
    blob = Path(blob_path).read_bytes()
    _W["model"] = T6.Model(blob)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": {"blob": str(blob_path), "blob_sha256": hashlib.sha256(blob).hexdigest(),
                                     "set": set_name, "frames": len(jobs), "twin": "synth/textspot6_twin.spot",
                                     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}) + "\n")
        with mp.get_context("fork").Pool(workers) as pool:
            for rec in pool.imap(_spot_one, jobs, chunksize=4):
                f.write(json.dumps(rec) + "\n")
    os.replace(tmp, out)
    return out


def cmd_spot(args):
    t0 = time.time()
    jobs = list(FRAMES[args.set]())
    if args.limit:
        jobs = jobs[:args.limit]
    out = write_spot(args.blob, jobs, args.out, args.set, args.workers)
    print(f"textspot6 spot finished: {args.set} {len(jobs)} frames {time.time() - t0:.0f} s -> {out}", flush=True)


def load_spot(path):
    rows = read_jsonl(path)
    if not rows or "meta" not in rows[0]:
        raise SystemExit(f"{path}: not a textspot6 spot file")
    return rows[0]["meta"], rows[1:]


# ----------------------------------------------------------------------------- eval/pagepath.py's inputs

def keep_box(b, tags):
    return not b[4] and (tags == "all" or b[8] == 0)


def decoded(r, decode):
    """The boxes of a spot row in eval/pagepath.py's decode: det [[x, y, w, h, vertical, conf, dark, block, tag,
    tconf]] as they are; detref: the refined boxes as [x, y, w, h, vertical, conf, dark, -1, tag, 0]."""
    if decode == "det":
        return r["boxes"]
    return [[*b[:7], -1, b[7], 0] for b in r["rboxes"]]


def frame_boxes(rows, decode, tags="all"):
    """eval/pagepath.py --frame-boxes rows of a pool split's spot rows: {title, page, view, boxes [[x, y, w, h,
    dark]]} in frame px, horizontal boxes only."""
    return [{"title": r["title"], "page": r["page"], "view": r["view"],
             "boxes": [[b[0], b[1], b[2], b[3], b[6]] for b in decoded(r, decode) if keep_box(b, tags)]}
            for r in rows]


def region_boxes(boxes, region, tags="all"):
    """A whole-frame decode's boxes inside one region (x0, y0, x1, y1 in the frame's px): the horizontal boxes whose
    centre lies in it, clipped to it, in region px [[x, y, w, h, dark]], in the decode's order."""
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    bx = []
    for b in boxes:
        if not keep_box(b, tags):
            continue
        cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            continue
        a0, b0 = max(0, b[0] - x0), max(0, b[1] - y0)
        a1, b1 = min(w, b[0] + b[2] - x0), min(h, b[1] + b[3] - y0)
        if a1 > a0 and b1 > b0:
            bx.append([int(a0), int(b0), int(a1 - a0), int(b1 - b0), int(b[6])])
    return bx


def cmd_pagepath_boxes(args):
    meta, rows = load_spot(args.spot)
    out = []
    if args.set == "hval":
        out = frame_boxes(rows, args.decode, args.tags)
    elif args.set == "dev":
        by = {r["frame"]: r for r in rows}
        for i, u in enumerate(dev_units()):
            r = by[dev_frame_key(u)]
            s = 2 if u["kind"] == "psp" else 1
            reg = tuple(v * s for v in u["region"])
            out.append({"unit": f"dev:{i}:{u['unit']}:{u['variant']}",
                        "boxes": region_boxes(decoded(r, args.decode), reg, args.tags)})
    else:
        raise SystemExit("pagepath-boxes: --set hval or dev")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{args.out}: {len(out)} rows ({args.set}, {args.decode}, tags {args.tags}) from "
          f"{meta['blob_sha256'][:12]}")


# ----------------------------------------------------------------------------- the crop decode

def crop_boxes(model, img, fw, fh, decode):
    """The crop decode on one region's grey (cut at the view's scale) of a fw x fh frame: the spotter at the frame's
    scale (amendment 3), det or detref, its horizontal boxes [[x, y, w, h, dark]] in region px."""
    import textspot6_twin as T6
    r = T6.spot(model, img, scale=T6.scale_of(model, fw, fh))
    boxes = [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, b.tag, b.tconf] for b in r.boxes]
    if decode == "detref":
        boxes = [[*b[:7], -1, b[7], 0] for b in refined(boxes, img)]
    return [[b[0], b[1], b[2], b[3], b[6]] for b in boxes if not b[4]]


def _crop_one(job):
    uid, img, fw, fh = job
    return {"unit": uid, "boxes": crop_boxes(_W["model"], img, fw, fh, _W["decode"])}


def crop_jobs(set_name, units_path=None, pages=PAGES):
    import cv2

    from common import to_gray
    if set_name == "dev":
        dims = {k["frame"]: (g.shape[1], g.shape[0]) for k, g in dev_frames()}     # console-view frame sizes
        for i, u in enumerate(dev_units()):
            img = u["image"]
            if u["kind"] == "psp":
                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            fw, fh = dims[dev_frame_key(u)]
            yield f"dev:{i}:{u['unit']}:{u['variant']}", np.ascontiguousarray(img), fw, fh
        return
    rows = read_jsonl(units_path)
    units = [r for r in rows if "uid" in r]
    cache = {}
    for u in units:
        k = (u["title"], u["page"])
        if k not in cache:
            cache.clear()
            bgr = cv2.imread(str(Path(pages) / u["title"] / "pages" / f"{u['page']}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"{u['title']}:{u['page']}")
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            if u["platform"] == "psp":
                rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cache[k] = to_gray(np.ascontiguousarray(rgb))
        g = cache[k]
        sc = 2 if u["platform"] == "psp" else 1
        x0, y0, x1, y1 = (v * sc for v in u["region"])
        yield u["uid"], np.ascontiguousarray(g[y0:y1, x0:x1]), g.shape[1], g.shape[0]


def write_crop(blob, jobs, decode, out, workers):
    """--boxes rows of the crop decode for (uid, region grey, frame width, frame height) jobs (_crop_one)."""
    import multiprocessing as mp

    import textspot6_twin as T6
    _W["model"], _W["decode"] = T6.Model(blob), decode
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f, mp.get_context("fork").Pool(workers) as pool:
        for rec in pool.imap(_crop_one, jobs, chunksize=8):
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, out)
    return out


def cmd_spot_crop(args):
    blob = Path(args.blob).read_bytes()
    t0 = time.time()
    jobs = list(crop_jobs(args.set, args.units))
    out = write_crop(blob, jobs, args.decode, args.out, args.workers)
    meta = {"blob": str(args.blob), "blob_sha256": hashlib.sha256(blob).hexdigest(), "set": args.set,
            "decode": f"crop{'+refine' if args.decode == 'detref' else ''}", "units": len(jobs),
            "scale": "the frame's (amendment 3): textspot6_twin.spot(scale=scale_of(frame width, frame height))",
            "units_file": str(args.units) if args.units else str(DEV_UNITS), "twin": "synth/textspot6_twin.spot",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(str(out).replace(".jsonl", ".meta.json")).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"textspot6 spot-crop finished: {args.set} {len(jobs)} units {time.time() - t0:.0f} s -> {out}", flush=True)


# ----------------------------------------------------------------------------- hval box metrics

def hval_refs(shards, manifest):
    """{(title, page): (refs at native px [{box, cls, vertical, ignore}], W, H, platform)} from the hval shards."""
    doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
    out = {}
    for s in doc["shards"]:
        if s["split"] != "hval":
            continue
        samples = json.loads((Path(shards) / s["dir"] / f"{s['name']}.json").read_text(encoding="utf-8"))["samples"]
        for smp in samples:
            refs = []
            for ln in smp["lines"]:
                x0, y0, x1, y1 = ln[:4]
                ign = ln[5] not in ("dialogue", "name")
                refs.append({"box": [x0, y0, x1 - x0, y1 - y0], "cls": ln[5] if not ign else "ui",
                             "vertical": bool(ln[4]), "ignore": ign})
            for b in smp["ignore"]:
                refs.append({"box": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "cls": None, "vertical": False,
                             "ignore": True})
            out[(smp["title"], smp["page"])] = (refs, smp["w"], smp["h"], smp["kind"])
    return out


def cmd_hval_boxes(args):
    import textspot_metrics as M
    meta, rows = load_spot(args.spot)
    refs = hval_refs(args.shards, args.manifest)
    acc = defaultdict(lambda: Counter())
    per_title = defaultdict(lambda: Counter())
    for r in rows:
        k = (r["title"], r["page"])
        if k not in refs:
            continue
        rf, W, H, plat = refs[k]
        s = 2 if r["view"] == "psp2x" else 1
        rs = [dict(x, box=[v * s for v in x["box"]]) for x in rf]
        dets = [{"box": b[:4], "vertical": bool(b[4]), "tag": b[8]} for b in r["boxes"]]
        m = M.match_frame(dets, rs, W * s, H * s)
        c = acc[r["view"]]
        c["frames"] += 1
        c["tp"] += m["tp"]
        c["fp"] += m["fp"]
        c["fn"] += m["fn"]
        for cl in ("dialogue", "name"):
            c[f"tp_{cl}"] += m["tp_cls"][cl]
            c[f"fn_{cl}"] += m["fn_cls"][cl]
        for i, j, _, _ in m["pairs"]:
            cl = rs[j]["cls"]
            want = 0 if cl == "dialogue" else 1
            c[f"tag_{cl}_n"] += 1
            c[f"tag_{cl}_ok"] += int(dets[i]["tag"] == want)
        c["frames_pick"] += int(r["pick"] >= 0)
        t = per_title[(r["view"], r["title"])]
        t["tp"] += m["tp"]
        t["fp"] += m["fp"]
        t["fn"] += m["fn"]

    def summ(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out = {"frames": c["frames"], "tp": tp, "fp": fp, "fn": fn,
               "precision": round(tp / (tp + fp), 4) if tp + fp else None,
               "recall": round(tp / (tp + fn), 4) if tp + fn else None,
               "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None}
        for cl in ("dialogue", "name"):
            n = c[f"tp_{cl}"] + c[f"fn_{cl}"]
            out[f"recall_{cl}"] = round(c[f"tp_{cl}"] / n, 4) if n else None
            out[f"refs_{cl}"] = n
            nt = c[f"tag_{cl}_n"]
            out[f"tag_acc_{cl}"] = round(c[f"tag_{cl}_ok"] / nt, 4) if nt else None
        if "frames_pick" in c:
            out["frames_with_pick"] = c["frames_pick"]
        return out

    res = {"spot": meta, "views": {v: summ(c) for v, c in sorted(acc.items())},
           "per_title": {f"{v}:{t}": summ(c) for (v, t), c in sorted(per_title.items())}}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for v, s in res["views"].items():
        print(v, s)


# ----------------------------------------------------------------------------- textspot's development frames

def tagged_dialogue(dets, refs, width, height):
    """(errors, chars) of the tagged dialogue path on one frame: dets [{box, vertical, text, tag}], refs as
    textspot_metrics'. Only tag-0 detections are read; each goes to a reference as textspot_metrics.assign says;
    a dialogue reference scores Levenshtein against its tag-0 readings joined in order; a tag-0 reading that goes to
    a non-dialogue reference or to none adds its characters."""
    import textspot_metrics as M
    from metrics import lev, norm
    d0 = [d for d in dets if d["tag"] == 0]
    a = M.assign(d0, refs, width, height)
    hyp = defaultdict(list)
    extra = 0
    for d, t in zip(d0, a):
        if t == "out":
            continue
        if t is None or refs[t]["cls"] != "dialogue":
            extra += len(norm(d.get("text") or ""))
        else:
            hyp[t].append(d)
    e = n = 0
    for j, r in enumerate(refs):
        if r.get("ignore") or r["cls"] != "dialogue":
            continue
        vert = bool(r.get("vertical"))
        parts = sorted(hyp[j], key=(lambda d: (d["box"][1], d["box"][0])) if vert else
                       (lambda d: (d["box"][0], d["box"][1])))
        ref_t = norm(r["text"])
        e += lev(ref_t, norm("".join(d.get("text") or "" for d in parts)))
        n += len(ref_t)
    return e + extra, n


def arm_of(r, decode, height):
    """(textspot's arm dict, the boxes with their tags) of one spot row in a whole-frame decode: det, the decoded
    boxes, blocks and pick as they are; detref, the refined boxes regrouped and ordered (textspot6_twin.
    group_and_order on the frame's height). The boxes are [x, y, w, h, vertical, conf, dark, block, tag]."""
    import textspot6_twin as T6
    if decode == "det":
        bx = [list(b) for b in r["boxes"]]
        return {"boxes": [tuple(b[:8]) for b in bx], "blocks": [tuple(k[:7]) for k in r["blocks"]],
                "pick": r["pick"], "truncated": r["truncated"]}, bx
    ob, bl, pk = T6.group_and_order([(*b[:7], b[7], 0) for b in r["rboxes"]], height)
    bx = [[*b[:7], b[9], b[7], b[8]] for b in ob]
    return {"boxes": [tuple(b[:8]) for b in bx], "blocks": [tuple(k[:7]) for k in bl], "pick": pk,
            "truncated": r["truncated"]}, bx


def cmd_tsdev(args):
    import shutil

    import textspot as TSE
    import textspot_metrics as M
    meta, rows = load_spot(args.spot)
    by = {r["frame"]: r for r in rows}
    man = [json.loads(x) for x in (ROOT / "eval" / "manifests" / "textspot-dev.jsonl").read_text().splitlines()]
    frames = TSE.load_frames(man, list(TSE.SHOT_DIRS))
    cache_dir = Path(args.cache)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
        seed = ART / "textspot3" / "eval" / "cache"
        for f in sorted(seed.glob("*.jsonl")):             # textspot3's development readings, copied byte for byte
            shutil.copyfile(f, cache_dir / f.name)
    cache = TSE.Cache(cache_dir)
    models = [TSE.SHORT.get(m, m) for m in args.models.split(",")]
    readers = TSE.c_readers({m: TSE.BLOBS[m] for m in [*models, TSE.V_MODEL]}, Path(args.cbuild))
    arms, tagged_boxes = [], []
    for f in frames:
        arm, bx = arm_of(by[f.id], args.decode, f.height)
        arms.append({"spot": arm})
        tagged_boxes.append(bx)
    n_read = TSE.read_all(frames, arms, models, readers, cache, workers=args.workers)
    out = {"spot": meta, "decode": args.decode, "read": n_read, "models": {}}
    for m in models:
        units, tagged = [], []
        for f, a, r, bx in zip(frames, arms, (by[f.id] for f in frames), tagged_boxes):
            texts = TSE.texts_of(f, a["spot"]["boxes"], m, readers, cache)
            u = TSE.frame_unit(f, a["spot"], texts)
            units.append(u)
            dets = [{"box": list(b[:4]), "vertical": bool(b[4]), "text": t, "tag": b[8]} for b, t in zip(bx, texts)]
            e, n = tagged_dialogue(dets, f.refs, f.width, f.height)
            tagged.append({"unit": f.id, "vn": f.vn, "part": f.part, "errors": e, "chars": n,
                           "presence": r["presence"], "text_free": f.text_free})
        summ = TSE.summarize(units)
        tg = M.cer_of(tagged)
        tf = [t for t in tagged if t["text_free"]]
        summ["tagged_dialogue"] = tg
        summ["tagged_dialogue_per_part"] = {p: M.cer_of([t for t in tagged if t["part"] == p])
                                            for p in sorted({t["part"] for t in tagged})}
        summ["presence_text_free"] = {"frames": len(tf), "with_dialogue_box": sum(1 for t in tf if t["presence"]),
                                      "presence_values": sorted(t["presence"] for t in tf if t["presence"])}
        out["models"][m] = {"summary": summ}
        if args.units:
            Path(args.units).parent.mkdir(parents=True, exist_ok=True)
            Path(str(args.units).replace(".json", f"_{m}.json")).write_text(
                json.dumps({"units": units, "tagged": tagged}) + "\n", encoding="utf-8")
        d = summ["detection"]
        print(f"{m}: F1 {d['f1']:.4f} P {d['precision']:.4f} R {d['recall']:.4f} | CER all "
              f"{summ['cer_all']['cer']} dialogue {summ['cer_dialogue']['cer']} | tagged dialogue {tg['cer']} | "
              f"text-free with a dialogue box {summ['presence_text_free']['with_dialogue_box']}/{len(tf)}")
    cache.close()
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("spot")
    a.add_argument("--blob", required=True)
    a.add_argument("--set", required=True, choices=SETS)
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "5")))
    a.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("pagepath-boxes")
    b.add_argument("--spot", required=True)
    b.add_argument("--set", required=True, choices=("hval", "dev"))
    b.add_argument("--out", required=True)
    b.add_argument("--tags", default="all", choices=("all", "dialogue"))
    b.add_argument("--decode", default="det", choices=("det", "detref"))
    e = sub.add_parser("spot-crop")
    e.add_argument("--blob", required=True)
    e.add_argument("--set", required=True, choices=("hval", "dev"))
    e.add_argument("--units", help="hval: eval/pagepath.py units' file")
    e.add_argument("--out", required=True)
    e.add_argument("--decode", default="det", choices=("det", "detref"))
    e.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "5")))
    c = sub.add_parser("hval-boxes")
    c.add_argument("--spot", required=True)
    c.add_argument("--shards", required=True)
    c.add_argument("--manifest", required=True)
    c.add_argument("--out", required=True)
    d = sub.add_parser("tsdev")
    d.add_argument("--spot", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--cache", default=str(ART / "vocr6-spotter" / "eval" / "tsdev_cache"))
    d.add_argument("--cbuild", default=str(ART / "vocr6-spotter" / "eval" / "cbuild"))
    d.add_argument("--units")
    d.add_argument("--models", default="N")
    d.add_argument("--decode", default="det", choices=("det", "detref"))
    d.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "5")))
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    {"spot": cmd_spot, "pagepath-boxes": cmd_pagepath_boxes, "hval-boxes": cmd_hval_boxes, "tsdev": cmd_tsdev,
     "spot-crop": cmd_spot_crop}[args.cmd](args)


if __name__ == "__main__":
    main()

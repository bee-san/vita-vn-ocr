#!/usr/bin/env python3
"""1b: realplus's PC screenshots at their own size, with realplus's line boxes and ignore areas for every region its
labels do not cover (docs/desktop-data.md).

  realplus.py plan     --out PLAN.json                          the screenshots, their labelled lines (no text)
  realplus.py meiki    --plan PLAN.json --out CACHE [--part i/n] [--workers 5]   MeikiOCR's detector at 0.2, native
  realplus.py build    --plan PLAN.json --meiki CACHE --out SHARDS [--part i/n] [--workers 5]
  realplus.py manifest --plan PLAN.json --shards SHARDS --out data/manifests/desktop_data/realplus_native_v1.json

realplus v3 (docs/realplus.md) labelled its lines on each screenshot scaled to fit 960 x 544 (cv2.INTER_AREA); the
originals are on disk at their own size (VNDB's JPEGs: $ART/realplus/shots/hires for v1-v2's 43,198 screenshots,
$ART/reserve-policy/realplus/shots/hires for the reclaim tier's 20,572). This keeps every v3 screenshot that holds a
kept line (49,285 of them) at its own size, its sha256 checked against realplus/manifests/shots_*.jsonl.

A frame's labels (s = the 960 x 544 fit scale of the screenshot; boxes in native px are the fit-scale boxes / s):
  positives  every kept v3 line: its reads line's ink box (the main text, without furigana) widened left and right to
             v3's final ink crop (realplus's crops.vita_ink less its 4 px pad: the crop was widened to the labelled
             characters), refined to ink at native size (textspot2_labels.refine_h, kept only inside the box grown by
             max(3, 3 / s) px and at least 45% of its height, as synth/textspot6_pages.refined_line), class "name"
             for a speaker name, else "text" (no tag target: realplus does not tell dialogue from choices)
  ignore     the kept lines' furigana; every other line the reads found (not kept: skipped as ui / cut / bad, unsure,
             a label outside the vocab, the repeat caps, ...) as its ink + furigana box; every MeikiOCR detector box,
             of the reads (threshold 0.5) and of a fresh native pass at threshold 0.2 (`meiki`), that meets no
             positive (IoU < 0.2 and under 30% of its area inside a positive grown by max(3, 3 / s) px); each grown
             by max(2, 2 / s) px. So text the labels do not cover is never taught as background.
  skipped    screenshots of VNs in vocr-desktop's development, PC-sample and confirmation sets (#119's units,
             textspot7's confirmation set), the reservation's test and dev_sample, the PC test set, and textspot2's
             skip lists (eval screenshots, eval/linefind VNs), so this pool never holds a frame vocr-desktop decides on.
Shards are #97's frames (train/textspot_train.FramePool reads them unchanged): SHARDS/desk_realplus_v1/
rp_PP_NNN.{npy,json}, src "realplus", kind "pc", at most 400 frames each.
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, ROOT / "synth", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from desktopdata import shards as SH  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
LINES_V3 = ART / "realplus-v3" / "lines_full_v3.jsonl"
READS = [ART / "realplus" / "reads" / "reads.jsonl", ART / "realplus-v3" / "reclaim" / "reads" / "reads.jsonl"]
SHOT_DIRS = [ART / "realplus" / "shots" / "hires", ART / "reserve-policy" / "realplus" / "shots" / "hires"]
DIR = "desk_realplus_v1"
SHARD_FRAMES = 400
RULES = {"meiki_det_thr": 0.2, "meet_iou": 0.2, "meet_inside": 0.3, "meet_grow": 3, "refine_grow": 3,
         "refine_min_across": 0.45, "ignore_grow": 2, "furigana_grow": 1, "crop_pad": 4, "fit": [960, 544]}
CODE = ["desktopdata/realplus.py", "desktopdata/shards.py", "synth/textspot2_labels.py", "teacher/geometry.py"]


def fit_scale(w, h):
    return min(RULES["fit"][0] / w, RULES["fit"][1] / h)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def excluded_vns():
    """{vn: why} of the VNs this pool must not hold (module docstring)."""
    import pickle
    out = {}
    res = json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text(encoding="utf-8"))
    for part in ("test", "dev_sample"):
        for v in res.get(part, {}):
            out.setdefault(v, f"reserved-{part}")
    units = ART / "vocr5-analysis" / "work" / "units.pkl"
    if units.exists():
        with open(units, "rb") as f:
            d = pickle.load(f)
        for u in d.get("units", []):
            out.setdefault(u["vn"], "dev119")
        for u in d.get("fixed", []) if isinstance(d.get("fixed"), list) else []:
            if isinstance(u, dict) and u.get("vn"):
                out.setdefault(u["vn"], "dev119")
    conf = ROOT / "eval" / "textspot7_confirm_units_v1.json"
    if conf.exists():
        doc = json.loads(conf.read_text(encoding="utf-8"))
        for u in doc.get("units_public", []):
            if u.get("vn"):
                out.setdefault(u["vn"], "confirm")
    pc = ROOT / "eval" / "manifests" / "desktop-pc-test-v1.jsonl"
    if pc.exists():
        for r in read_jsonl(pc):
            out.setdefault(r["vn"], "pc-test")
    pc_vns = ART / "desktop-data" / "pctest" / "candidates.jsonl"
    if pc_vns.exists():
        for r in read_jsonl(pc_vns):
            out.setdefault(r["vn"], "pc-test-candidate")
    t2 = json.loads((ROOT / "data" / "manifests" / "textspot2_shards_v1.json").read_text(encoding="utf-8"))
    for v in t2["real"]["measured"]["skip"]["linefind_list_vn"]["skip_vns"]:
        out.setdefault(v, "linefind-list")
    return out


def skipped_shots():
    t2 = json.loads((ROOT / "data" / "manifests" / "textspot2_shards_v1.json").read_text(encoding="utf-8"))
    return set(t2["real"]["measured"]["skip"]["eval_screenshot"]["skip_shots"])


def cmd_plan(args):
    shots = []
    for p in sorted((ROOT / "realplus" / "manifests").glob("shots_*.jsonl")):
        shots += read_jsonl(p)
    ex, skip = excluded_vns(), skipped_shots()
    kept = collections.defaultdict(list)
    for r in read_jsonl(LINES_V3):
        kept[r["shot"]].append({"id": r["id"], "k": int(r["id"].rsplit("_", 1)[1]), "name": bool(r.get("name")),
                                "vita_ink": r["crops"].get("vita_ink")})
    why = collections.Counter()
    rows = []
    for s in shots:
        if s["vn"] in ex:
            why[f"vn:{ex[s['vn']]}"] += 1
            continue
        if s["shot"] in skip:
            why["eval-screenshot"] += 1
            continue
        if not kept.get(s["shot"]):
            why["no-kept-line"] += 1
            continue
        path = next((d / f"{s['shot']}.jpg" for d in SHOT_DIRS if (d / f"{s['shot']}.jpg").exists()), None)
        if path is None:
            why["no-file"] += 1
            continue
        rows.append({"shot": s["shot"], "vn": s["vn"], "url": s["url"], "sha256": s["sha256"], "dims": s["dims"],
                     "release": s.get("release"), "path": str(path),
                     "lines": sorted(kept[s["shot"]], key=lambda x: x["k"])})
    doc = {"kind": "desktop-realplus-plan", "created": SH.utc(), "rules": RULES, "frames": len(rows),
           "vns": len({r["vn"] for r in rows}), "lines": sum(len(r["lines"]) for r in rows), "dropped": dict(why),
           "lines_v3_sha256": SH.sha256_file(LINES_V3), "excluded_vns": len(ex), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc) + "\n", encoding="utf-8")
    print(f"plan: {len(rows)} frames, {doc['vns']} VNs, {doc['lines']} lines; dropped {dict(why)}", flush=True)


def part_of(rows, part):
    if not part:
        return rows
    i, n = (int(v) for v in part.split("/"))
    return rows[i::n]


# ----------------------------------------------------------------------------- MeikiOCR at 0.2

_W = {}


def _init_meiki():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from teacher import engines as E
    _W["meiki"] = E.Meiki(det_thr=RULES["meiki_det_thr"], n_threads=1)


def _meiki_one(row):
    import cv2

    import textspot2_labels as L2
    bgr = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if bgr is None:
        return {"shot": row["shot"], "error": "unreadable"}
    return {"shot": row["shot"], "boxes": [[*b, round(float(s), 4)] for b, s in L2.meiki_detect(_W["meiki"], bgr)]}


def cmd_meiki(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = part_of(plan["rows"], args.part)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.part.replace("/", "of") if args.part else "all"
    path = out / f"meiki_{tag}.jsonl"
    done = {r["shot"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [r for r in rows if r["shot"] not in done]
    print(f"meiki: {len(rows)} frames, {len(done)} cached, {len(todo)} to run", flush=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo, mp.get_context("spawn").Pool(args.workers,
                                                                               initializer=_init_meiki) as pool:
        for i, r in enumerate(pool.imap(_meiki_one, todo, chunksize=8)):
            fo.write(json.dumps(r) + "\n")
            if (i + 1) % 2000 == 0:
                fo.flush()
                print(f"  {i + 1}/{len(todo)} {time.time() - t0:.0f} s", flush=True)
    print(f"meiki: {len(todo)} frames in {time.time() - t0:.0f} s", flush=True)


def load_meiki(cache):
    out = {}
    for p in sorted(Path(cache).glob("meiki_*.jsonl")):
        for r in read_jsonl(p):
            if "boxes" in r:
                out[r["shot"]] = r["boxes"]
    return out


# ----------------------------------------------------------------------------- PP-OCRv6 det small (v2's third source)

PPDET = {"engine": "rapidocr 3.10.0, PP-OCRv6 det small (its defaults: limit_side_len 736 min, thresh 0.3, "
                   "box_thresh 0.5, unclip_ratio 1.6)", "file": "PP-OCRv6_det_small.onnx",
         "sha256": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"}


def _init_ppdet():
    import importlib.util

    from rapidocr import RapidOCR
    spec = importlib.util.find_spec("rapidocr")
    model = Path(spec.origin).parent / "models" / PPDET["file"]
    if SH.sha256_file(model) != PPDET["sha256"]:
        raise SystemExit(f"{model}: sha256 differs from the pin")
    _W["ppdet"] = RapidOCR(params={"Global.log_level": "error", "EngineConfig.onnxruntime.intra_op_num_threads": 1})


def _ppdet_one(row):
    import cv2
    bgr = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if bgr is None:
        return {"shot": row["shot"], "error": "unreadable"}
    r = _W["ppdet"](bgr, use_det=True, use_cls=False, use_rec=False)
    quads = [] if r.boxes is None else [np.asarray(q, np.float32) for q in r.boxes]
    scores = list(r.scores) if r.scores is not None else [None] * len(quads)
    out = []
    for q, sc in zip(quads, scores):
        x0, y0 = int(np.floor(q[:, 0].min())), int(np.floor(q[:, 1].min()))
        x1, y1 = int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max()))
        out.append([x0, y0, x1, y1, None if sc is None else round(float(sc), 4)])
    return {"shot": row["shot"], "boxes": out}


def cmd_ppdet(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = part_of(plan["rows"], args.part)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.part.replace("/", "of") if args.part else "all"
    path = out / f"ppdet_{tag}.jsonl"
    done = {r["shot"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [r for r in rows if r["shot"] not in done]
    print(f"ppdet: {len(rows)} frames, {len(done)} cached, {len(todo)} to run", flush=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo, mp.get_context("spawn").Pool(args.workers,
                                                                               initializer=_init_ppdet) as pool:
        for i, r in enumerate(pool.imap(_ppdet_one, todo, chunksize=8)):
            fo.write(json.dumps(r) + "\n")
            if (i + 1) % 2000 == 0:
                fo.flush()
                print(f"  {i + 1}/{len(todo)} {time.time() - t0:.0f} s", flush=True)
    print(f"ppdet: {len(todo)} frames in {time.time() - t0:.0f} s", flush=True)


def load_ppdet(cache):
    out = {}
    for p in sorted(Path(cache).glob("ppdet_*.jsonl")):
        for r in read_jsonl(p):
            if "boxes" in r:
                out[r["shot"]] = r["boxes"]
    return out


def load_reads(shots):
    """{shot: reads row} of the realplus reads (v1-v2 and the reclaim tier) for the given shots."""
    out = {}
    for p in READS:
        with open(p, encoding="utf-8") as f:
            for x in f:
                if not x.strip():
                    continue
                # cheap pre-filter on the shot key before a full parse
                k = x[x.find('"shot": "') + 9:x.find('"', x.find('"shot": "') + 9)] if '"shot": "' in x else None
                if k is not None and k not in shots:
                    continue
                r = json.loads(x)
                if r.get("shot") in shots:
                    out[r["shot"]] = r
    return out


# ----------------------------------------------------------------------------- labels

def scale_box(b, s):
    """A fit-scale box [x0, y0, x1, y1) -> native px (start floor, end ceil)."""
    return [int(math.floor(b[0] / s)), int(math.floor(b[1] / s)), int(math.ceil(b[2] / s)), int(math.ceil(b[3] / s))]


def clip(b, W, H):
    b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
    return b if b[2] > b[0] and b[3] > b[1] else None


def grow(b, g, W, H):
    return clip([b[0] - g, b[1] - g, b[2] + g, b[3] + g], W, H)


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u else 0.0


def positive_box(ln, kept, s, W, H):
    """A kept line's native box before refinement: its reads ink box widened to v3's crop (less the pad)."""
    x0, y0, x1, y1 = ln["ink"]
    vi = kept.get("vita_ink")
    if vi:
        p = RULES["crop_pad"]
        x0, x1 = min(x0, vi[0] + p), max(x1, vi[2] - p)
    return clip(scale_box([x0, y0, x1, y1], s), W, H)


def refined(rgb, q, s):
    import textspot2_labels as L2
    H, W = rgb.shape[:2]
    r = L2.refine_h(rgb, q)
    lim = grow(q, max(RULES["refine_grow"], int(round(RULES["refine_grow"] / s))), W, H) or q
    ok = (r[0] >= lim[0] and r[1] >= lim[1] and r[2] <= lim[2] and r[3] <= lim[3] and r[2] > r[0] and r[3] > r[1]
          and (r[3] - r[1]) >= RULES["refine_min_across"] * (q[3] - q[1]))
    return ([int(v) for v in r], "ink") if ok else (list(q), "box")


def labels_of(rgb, row, rd, meiki):
    """v1's labels (realplus_native_v1, kept to reproduce it): (lines, ignore, counts) of one frame. Superseded by
    labels_of_v2: its meets() test let an unkept neighbour line hide inside an over-tall positive (audit-desktop-data
    F1)."""
    H, W = rgb.shape[:2]
    s = fit_scale(W, H)
    rlines = {ln["k"]: ln for ln in rd.get("lines", [])}
    kept = {x["k"]: x for x in row["lines"]}
    g_ign = max(RULES["ignore_grow"], int(round(RULES["ignore_grow"] / s)))
    g_meet = max(RULES["meet_grow"], int(round(RULES["meet_grow"] / s)))
    lines, ignore = [], []
    c = collections.Counter()
    for k, kx in kept.items():
        ln = rlines.get(k)
        if ln is None:
            c["kept_without_read"] += 1
            continue
        q = positive_box(ln, kx, s, W, H)
        if q is None:
            continue
        b, how = refined(rgb, q, s)
        lines.append([*b, 0, "name" if kx["name"] else "text", 1.0])
        c[f"positive_{how}"] += 1
        for fb in ln.get("furi") or []:
            g = grow(scale_box(fb, s), max(RULES["furigana_grow"], int(round(RULES["furigana_grow"] / s))), W, H)
            if g:
                ignore.append(g)
                c["furigana_ignore"] += 1
    pos = [ln[:4] for ln in lines]
    pos_g = [grow(p, g_meet, W, H) or p for p in pos]

    def meets(b):
        return any(iou(b, p) >= RULES["meet_iou"] or inter(b, pg) >= RULES["meet_inside"] * max(1, area(b))
                   for p, pg in zip(pos, pos_g))

    def add_ignore(g, key):
        # an area already (almost) inside another ignore area adds nothing: the same text found by the reads'
        # lines, the 0.5 detector and the 0.2 detector is one area
        if any(inter(g, o) >= 0.9 * area(g) for o in ignore):
            c[f"{key}_dup"] += 1
            return
        ignore.append(g)
        c[key] += 1
    for k, ln in rlines.items():
        if k in kept:
            continue
        b = clip(scale_box(ln.get("crop_ink") or ln["ink"], s), W, H)
        if b and not meets(b):
            g = grow(b, g_ign, W, H)
            if g:
                add_ignore(g, "other_line_ignore")
    for d in rd.get("dets", []):
        b = clip(scale_box(d["det"], s), W, H)
        if b and not meets(b):
            g = grow(b, g_ign, W, H)
            if g:
                add_ignore(g, "det05_ignore")
    for x0, y0, x1, y1, _sc in meiki or []:
        b = clip([x0, y0, x1, y1], W, H)
        if b and not meets(b):
            g = grow(b, g_ign, W, H)
            if g:
                add_ignore(g, "det02_ignore")
    return lines, ignore, c


RULES_V2 = dict(RULES, version=2, same_row_iou=0.5, same_row_vov=0.6, same_row_inside=0.6, same_row_height=1.7,
                row_vov=0.5, min_height=0.6, anchor_cover=0.7, overlap_vov=1 / 3, covered=0.5, ppdet=PPDET)


def inner_box(b):
    """A box's inner part: shrunk by a quarter of its height at the top and bottom and by min(w, h) / 4 at the ends
    (audit-desktop-data's coverage measure)."""
    x0, y0, x1, y1 = b[:4]
    h, w = y1 - y0, x1 - x0
    dy, dx = h / 4, min(w / 4, h / 4)
    return [int(round(x0 + dx)), int(round(y0 + dy)), int(round(x1 - dx)), int(round(y1 - dy))]


def vov(a, b):
    """Vertical overlap of two boxes as a share of the smaller height."""
    o = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, o) / max(1, min(a[3] - a[1], b[3] - b[1]))


def hov(a, b):
    """Horizontal overlap of two boxes in px."""
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def same_row(b, p, g):
    """v2: box b is positive p's own text (its detector box, a piece of the line): IoU >= 0.5, or b no taller than
    1.7 x p (+ 2 g), overlapping p vertically by >= 0.6 of the smaller height, with >= 0.6 of its width inside p's
    columns grown by g. A neighbouring row, a box over two rows, or text beside the line on its row is not."""
    if iou(b, p) >= RULES_V2["same_row_iou"]:
        return True
    if (b[3] - b[1]) > RULES_V2["same_row_height"] * (p[3] - p[1]) + 2 * g:
        return False
    inside = max(0, min(b[2], p[2] + g) - max(b[0], p[0] - g)) / max(1, b[2] - b[0])
    return vov(b, p) >= RULES_V2["same_row_vov"] and inside >= RULES_V2["same_row_inside"]


def clip_rows(p, anchor, others):
    """v2: p clipped at the boundary to every other text row that overlaps it horizontally and reaches into it. A
    row is another row when it overlaps the anchor (the line's own detector box) vertically by under half the smaller
    height; the boundary is the midpoint between the anchor's edge and the row's facing edge (rounded away from the
    anchor), so two tight rows' positives meet there and overlap by at most 1 px."""
    x0, y0, x1, y1 = p
    ac = (anchor[1] + anchor[3]) / 2
    for o in others:
        if hov(o, p) <= 0 or vov(o, anchor) >= RULES_V2["row_vov"]:
            continue
        oc = (o[1] + o[3]) / 2
        if oc > ac and o[1] < y1:
            y1 = min(y1, int(math.ceil((anchor[3] + o[1]) / 2)))
        elif oc < ac and o[3] > y0:
            y0 = max(y0, int(math.floor((o[3] + anchor[1]) / 2)))
    return [x0, y0, x1, y1]


def labels_of_v2(rgb, row, rd, meiki, ppdet):
    """(lines, ignore, counts) of one frame, v2 (realplus_native_v2):
      positives  each kept line's v1 box (its reads ink box widened to v3's crop, at native size), clipped at the
                 boundaries to the other rows (clip_rows: every other reads line's detector box, every MeikiOCR box
                 of score >= 0.5 of the native pass, the line's own furigana), refined to ink inside that band and
                 clipped again. A refined box under 0.6 of the line's own detector box's height, or covering under
                 0.7 of that box's rows (the refinement took part of the glyphs, or their lower half and a drop
                 shadow: two-tone, outlined and shadowed fills), gives way to the clipped reads box, and a reads box
                 that short to the detector box's rows (with the reads box's columns). A positive still under
                 0.6 of its detector box's height, or one that overlaps another positive (horizontally, and
                 vertically by over a third of the smaller height), becomes an ignore area instead
      ignore     the kept lines' furigana; every box that is no positive's own text (same_row) from four sources:
                 the reads lines realplus did not keep (ink + furigana), the reads' detector boxes (MeikiOCR, 0.5),
                 MeikiOCR's native pass (0.2) and PP-OCRv6 det small (an independent detector); a detector box
                 only when under half of its inner part (inner_box) is labelled already (by an ignore area or a
                 positive's inner band), so a box over two labelled rows does not erase the gap between them;
                 each grown by max(2, 2 / s) px; an area at least 90% inside an earlier one is not added again"""
    H, W = rgb.shape[:2]
    s = fit_scale(W, H)
    rlines = {ln["k"]: ln for ln in rd.get("lines", [])}
    kept = {x["k"]: x for x in row["lines"]}
    g_ign = max(RULES["ignore_grow"], int(round(RULES["ignore_grow"] / s)))
    g_meet = max(RULES["meet_grow"], int(round(RULES["meet_grow"] / s)))
    g_furi = max(RULES["furigana_grow"], int(round(RULES["furigana_grow"] / s)))
    c = collections.Counter()
    rows_all = {k: clip(scale_box(ln["det"], s), W, H) for k, ln in rlines.items()}
    meiki_rows = [clip(b[:4], W, H) for b in meiki or [] if b[4] >= 0.5]
    cand = []                                   # (positive box, kind, anchor)
    ignore = []
    for k, kx in kept.items():
        ln = rlines.get(k)
        if ln is None:
            c["kept_without_read"] += 1
            continue
        q = positive_box(ln, kx, s, W, H)
        anchor = rows_all.get(k)
        if q is None or anchor is None:
            continue
        furi = [b for b in (clip(scale_box(fb, s), W, H) for fb in ln.get("furi") or []) if b]
        others = [b for j, b in rows_all.items() if j != k and b] + [b for b in meiki_rows if b] + furi
        q2 = clip(clip_rows(q, anchor, others), W, H)
        if q2 is None:
            ignore.append(grow(q, g_ign, W, H) or q)
            c["positive_to_ignore_height"] += 1
            continue
        ah = anchor[3] - anchor[1]
        min_h = RULES_V2["min_height"] * ah
        r, how = refined(rgb, q2, s)
        rows_cov = max(0, min(r[3], anchor[3]) - max(r[1], anchor[1])) / max(1, ah)
        if ((r[3] - r[1]) < min_h or rows_cov < RULES_V2["anchor_cover"]) and (q2[3] - q2[1]) >= min_h:
            r, how = list(q2), "box_refine_off"             # the refinement took part of the glyphs: the reads box
        b = clip(clip_rows(r, anchor, others), W, H) or q2
        c["positive_clipped"] += int(q2 != q or b != r)       # a row boundary cut the box before or after refining
        if (b[3] - b[1]) < min_h:                             # the reads box is short too: the detector box's rows
            a2 = clip(clip_rows([q2[0], anchor[1], q2[2], anchor[3]], anchor, others), W, H)
            if a2 is not None and (a2[3] - a2[1]) >= min_h:
                b, how = a2, "box_anchor"
        if (b[3] - b[1]) < min_h:
            ignore.append(grow(q, g_ign, W, H) or q)
            c["positive_to_ignore_height"] += 1
            continue
        cand.append((b, "name" if kx["name"] else "text", how))
        for fb in furi:
            g = grow(fb, g_furi, W, H)
            if g:
                ignore.append(g)
                c["furigana_ignore"] += 1
    bad = set()
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b2 = cand[i][0], cand[j][0]
            if hov(a, b2) > 0 and vov(a, b2) > RULES_V2["overlap_vov"]:
                bad |= {i, j}
    lines = []
    for i, (b, cls, how) in enumerate(cand):
        if i in bad:
            ignore.append(grow(b, g_ign, W, H) or b)
            c["positive_to_ignore_overlap"] += 1
            continue
        lines.append([*b, 0, cls, 1.0])
        c[f"positive_{how}"] += 1
    pos = [ln[:4] for ln in lines]
    cov = np.zeros((H, W), bool)                    # what is labelled: ignore areas and the positives' centre bands
    for b in [inner_box(p) for p in pos] + ignore:
        if b[2] > b[0] and b[3] > b[1]:
            cov[b[1]:b[3], b[0]:b[2]] = True

    def own(b):
        return any(same_row(b, p, g_meet) for p in pos)

    def add_ignore(b, key, detector=False):
        b = clip(b, W, H)
        if b is None or own(b):
            return
        if detector:                                # a detector box: only text the labels so far leave uncovered
            ib = inner_box(b)
            if ib[2] > ib[0] and ib[3] > ib[1] and cov[ib[1]:ib[3], ib[0]:ib[2]].mean() >= RULES_V2["covered"]:
                c[f"{key}_covered"] += 1
                return
        g = grow(b, g_ign, W, H)
        if g is None:
            return
        if any(inter(g, o) >= 0.9 * area(g) for o in ignore):
            c[f"{key}_dup"] += 1
            return
        ignore.append(g)
        cov[g[1]:g[3], g[0]:g[2]] = True
        c[key] += 1
    for k, ln in rlines.items():
        if k not in kept:
            add_ignore(scale_box(ln.get("crop_ink") or ln["ink"], s), "other_line_ignore")
    for d in rd.get("dets", []):
        add_ignore(scale_box(d["det"], s), "det05_ignore", True)
    for b in meiki or []:
        add_ignore(b[:4], "det02_ignore", True)
    for b in ppdet or []:
        add_ignore(b[:4], "ppdet_ignore", True)
    return lines, ignore, c


def _build_one(job):
    import hashlib

    import cv2
    row, rd, meiki, ppdet, rules = job
    raw = Path(row["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        return None, None, {"sha256_mismatch": 1}
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    H, W = rgb.shape[:2]
    if rules == "v1":
        lines, ignore, c = labels_of(rgb, row, rd, meiki)
    else:
        lines, ignore, c = labels_of_v2(rgb, row, rd, meiki, ppdet)
    meta = {"url": row["url"], "sha256": row["sha256"], "release": row.get("release"), "dims": [W, H],
            "fit": round(fit_scale(W, H), 6), "kept_lines": len(row["lines"]), "counts": dict(c)}
    if rules != "v1":
        meta["rules"] = 2
    smp = {"src": "realplus", "kind": "pc", "vn": row["vn"], "shot": row["shot"], "lines": lines, "ignore": ignore,
           "meta": meta}
    return smp, SH.to_gray(rgb), dict(c)


DIRS = {"v1": "desk_realplus_v1", "v2": "desk_realplus_v2"}


def cmd_build(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = part_of(plan["rows"], args.part)
    meiki = load_meiki(args.meiki)
    if args.rules != "v1" and not args.ppdet:
        raise SystemExit("rules v2 needs --ppdet (realplus.py ppdet's cache)")
    ppdet = load_ppdet(args.ppdet) if args.rules != "v1" else {}
    if args.rules != "v1":
        missing = [r["shot"] for r in rows if r["shot"] not in ppdet]
        if missing:
            raise SystemExit(f"{len(missing)} frames have no PP-OCRv6 det boxes in {args.ppdet} (run ppdet first)")
    reads = load_reads({r["shot"] for r in rows})
    out = Path(args.out) / DIRS[args.rules]
    pi = int(args.part.split("/")[0]) if args.part else 0
    tot = collections.Counter()
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k in range(0, len(rows), SHARD_FRAMES):
            name = f"rp_{pi:02d}_{k // SHARD_FRAMES:03d}"
            if (out / f"{name}.json").exists() and not args.force:
                continue
            jobs = [(r, reads.get(r["shot"], {}), meiki.get(r["shot"]), ppdet.get(r["shot"]), args.rules)
                    for r in rows[k:k + SHARD_FRAMES]]
            res = [x for x in pool.imap(_build_one, jobs, chunksize=4)]
            ok = [(s, g) for s, g, _ in res if s is not None]
            for _, _, c in res:
                tot.update(c)
            SH.write_shard(out / name, [s for s, _ in ok], [g for _, g in ok])
            print(f"  {DIRS[args.rules]}/{name}: {len(ok)} frames ({time.time() - t0:.0f} s)", flush=True)
    print(f"realplus build: {dict(tot)} in {time.time() - t0:.0f} s", flush=True)


def cmd_manifest(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    d = DIRS[args.rules]
    root = Path(args.shards) / d
    shards = [SH.shard_record(js.with_suffix(""), d) for js in sorted(root.glob("*.json"))]
    n = sum(r["samples"] for r in shards)
    doc = {"kind": "desktop-frames", "schema": 1, "issue": 169, "generator": "desktopdata/realplus.py",
           "created": SH.utc(), "format": "#97's frames (synth/textdet_data.py); train/textspot_train.FramePool "
                                          "reads them as source realplus",
           "rules": RULES if args.rules == "v1" else RULES_V2,
           "label_rule": "labels_of" if args.rules == "v1" else "labels_of_v2",
           "plan": {k: plan[k] for k in ("frames", "vns", "lines", "dropped", "lines_v3_sha256", "excluded_vns")},
           "plan_sha256": SH.sha256_file(args.plan), "code_sha256": SH.code_sha256(ROOT, CODE),
           "shards": shards, "totals": SH.totals(shards),
           "note": "shards hold copyrighted VNDB screenshots and stay local; this manifest holds names, counts, sha256"}
    if n != plan["frames"]:
        doc["missing_frames"] = plan["frames"] - n
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {n} frames of {plan['frames']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p = sub.add_parser("meiki")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part")
    p.add_argument("--workers", type=int, default=5)
    p = sub.add_parser("ppdet")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part")
    p.add_argument("--workers", type=int, default=5)
    p = sub.add_parser("build")
    p.add_argument("--plan", required=True)
    p.add_argument("--meiki", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ppdet", help="realplus.py ppdet's cache (rules v2)")
    p.add_argument("--rules", choices=("v1", "v2"), default="v2", help="v1: realplus_native_v1's labels_of")
    p.add_argument("--part")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("manifest")
    p.add_argument("--plan", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--rules", choices=("v1", "v2"), default="v2")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    {"plan": cmd_plan, "meiki": cmd_meiki, "ppdet": cmd_ppdet, "build": cmd_build,
     "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

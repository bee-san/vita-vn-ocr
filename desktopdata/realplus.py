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
    """(lines, ignore, counts) of one frame (module docstring)."""
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


def _build_one(job):
    import cv2
    row, rd, meiki = job
    raw = Path(row["path"]).read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        return None, None, {"sha256_mismatch": 1}
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    H, W = rgb.shape[:2]
    lines, ignore, c = labels_of(rgb, row, rd, meiki)
    smp = {"src": "realplus", "kind": "pc", "vn": row["vn"], "shot": row["shot"], "lines": lines, "ignore": ignore,
           "meta": {"url": row["url"], "sha256": row["sha256"], "release": row.get("release"), "dims": [W, H],
                    "fit": round(fit_scale(W, H), 6), "kept_lines": len(row["lines"]), "counts": dict(c)}}
    return smp, SH.to_gray(rgb), dict(c)


def cmd_build(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = part_of(plan["rows"], args.part)
    meiki = load_meiki(args.meiki)
    reads = load_reads({r["shot"] for r in rows})
    out = Path(args.out) / DIR
    pi = int(args.part.split("/")[0]) if args.part else 0
    tot = collections.Counter()
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k in range(0, len(rows), SHARD_FRAMES):
            name = f"rp_{pi:02d}_{k // SHARD_FRAMES:03d}"
            if (out / f"{name}.json").exists() and not args.force:
                continue
            jobs = [(r, reads.get(r["shot"], {}), meiki.get(r["shot"])) for r in rows[k:k + SHARD_FRAMES]]
            res = [x for x in pool.imap(_build_one, jobs, chunksize=4)]
            ok = [(s, g) for s, g, _ in res if s is not None]
            for _, _, c in res:
                tot.update(c)
            SH.write_shard(out / name, [s for s, _ in ok], [g for _, g in ok])
            print(f"  {DIR}/{name}: {len(ok)} frames ({time.time() - t0:.0f} s)", flush=True)
    print(f"realplus build: {dict(tot)} in {time.time() - t0:.0f} s", flush=True)


def cmd_manifest(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.shards) / DIR
    shards = [SH.shard_record(js.with_suffix(""), DIR) for js in sorted(root.glob("*.json"))]
    n = sum(r["samples"] for r in shards)
    doc = {"kind": "desktop-frames", "schema": 1, "issue": 169, "generator": "desktopdata/realplus.py",
           "created": SH.utc(), "format": "#97's frames (synth/textdet_data.py); train/textspot_train.FramePool "
                                          "reads them as source realplus",
           "rules": RULES, "plan": {k: plan[k] for k in ("frames", "vns", "lines", "dropped", "lines_v3_sha256",
                                                         "excluded_vns")},
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
    p = sub.add_parser("build")
    p.add_argument("--plan", required=True)
    p.add_argument("--meiki", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("manifest")
    p.add_argument("--plan", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    {"plan": cmd_plan, "meiki": cmd_meiki, "build": cmd_build, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

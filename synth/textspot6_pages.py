#!/usr/bin/env python3
"""textspot6's hooked page shards (issue #128, docs/textspot6.md "Data"): whole native frames of the owner's hooked
corpus (#125) with the spotter's targets, in #97's shard format, plus MeikiOCR's detector boxes for the text the hook
does not cover.

  textspot6_pages.py meiki --pool P --pages D --out CACHE [--titles T,...] [--splits train,hval,guard] [--workers 5]
  textspot6_pages.py build --pool P --pages D --meiki CACHE --out SHARDS [--titles T,...] [--workers 5]
  textspot6_pages.py manifest --shards SHARDS --pool P --out data/manifests/textspot6_pages_v1.json
  textspot6_pages.py sheet --shards SHARDS --dir DIR --name NAME --out PNG [--n 6] [--seed 0]   QA (<= 1280 px)

Input: vocr6-ingest's page pool (P = $A/vocr6-ingest/pagepool/<version>/<TID>.jsonl: one record per page with its
split, dims, the renderer's page boxes, the verified lines with kind, box, box_main, vertical, drop and box_index,
the uncovered page boxes and the untrusted flag) and its extracted pages (D = $A/vocr6-ingest/pages: <image> of a
record). Test pages are never read here (the build refuses split "test"). Everything stays local: shards hold
copyrighted game frames (never committed); the manifest holds only names, counts and sha256.

meiki: teacher/engines.Meiki(det_thr=0.2), one thread per worker, its detector only (textspot2_labels.meiki_detect:
the boxes with score > 0.2, BGR input as teacher/label.py feeds it). Each box is refined to ink as textspot2 refines
meiki boxes (textspot2_labels.refine_h on the colour page, refine_t on its transpose; refined_of P1 decides the
orientation). One JSON line per page in CACHE/<TID>.jsonl, appended, so a killed job resumes.

The labels of a page (RULES holds every setting; build_labels is pure, boxes in, boxes out):
  positives  every verified line of kind "line" (class "dialogue") or "name" (class "name") whose drop is None or
             "oov" (a character outside VOCR5's vocabulary does not change the box), weight 1. Its box is box_main
             when the line has furigana, else box ([x, y, w, h] in native px: the renderer's glyph quads), refined to
             ink: refine_h for a horizontal line, refine_t for a vertical one, kept only when the refined box lies
             inside the quad box grown by 3 px and its size across is at least 45% of the quad's (else the quad box)
  ignore     ruby lines (their box grown by 1 px), lines whose drop is "geometry" or "conflict" (box grown by 2 px),
             every uncovered page box (a renderer box no verified line holds: its quad grown by 2 px)
  other text a meiki box that meets no page box (IoU < 0.2 with every page box and less than 30% of its area inside
             any one page box grown by 3 px): with score >= 0.5 a positive of class "notdlg" (some text that is not
             dialogue: a name in a pre-rendered plate, a choice, UI or scene text; the trainer's tag loss reads it as
             the class set {name, choice, ui, scene}) with textspot2's weight 0.5 + 0.5 min(1, (score - 0.5) / 0.3),
             its refined box and orientation; below 0.5 an ignore area (its refined box)
  pages      an untrusted page (the ingest's audit) is left out and counted
A shard line is [x0, y0, x1, y1, vertical, class, weight] in the canvas's native px, an ignore area [x0, y0, x1, y1];
a sample is {off, h, w, src "hook", kind "vita" | "psp", vn, shot "<TID>:<page>", title, page, split, lines, ignore,
meta}. Shards are per title and split: SHARDS/<split>/<TID>_NNN.{npy,json} (at most 500 pages each).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RULES = {
    "meiki_det_thr": 0.2,
    "meiki_pos_min": 0.5,
    "meiki_weight_span": 0.3,
    "meet_iou": 0.2,
    "meet_inside": 0.3,
    "meet_grow": 3,
    "refine_grow": 3,
    "refine_min_across": 0.45,
    "ruby_grow": 1,
    "dropped_grow": 2,
    "uncovered_grow": 2,
    "positive_drops": [None, "oov"],
    "ignore_drops": ["geometry", "conflict"],
    "classes": {"line": "dialogue", "name": "name"},
    "other_class": "notdlg",
    "shard_pages": 500,
}
SPLITS = ("train", "hval", "guard")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pool(pool, titles=None, splits=SPLITS):
    """[(title, record)] of the pool's pages in title and page order, of the given splits (never test)."""
    if "test" in splits:
        raise SystemExit("textspot6_pages never reads test pages")
    out = []
    for f in sorted(Path(pool).glob("*.jsonl")):
        tid = f.stem
        if titles and tid not in titles:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["split"] in splits:
                out.append((tid, r))
    return out


# ----------------------------------------------------------------------------- geometry

def xyxy(b):
    x, y, w, h = (int(v) for v in b)
    return [x, y, x + w, y + h]


def grow(b, g, W, H):
    return [max(0, b[0] - g), max(0, b[1] - g), min(W, b[2] + g), min(H, b[3] + g)]


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def quad_box(xy):
    """A page box's float xyxy -> the integer box that holds it (floor / ceil)."""
    import math
    return [int(math.floor(xy[0])), int(math.floor(xy[1])), int(math.ceil(xy[2])), int(math.ceil(xy[3]))]


def refined_line(rgb, q, vertical, rules=RULES):
    """(box, how) of a verified line's quad box q [x0, y0, x1, y1]: the ink box (refine_h / refine_t) when it passes
    the two checks, else q."""
    import textspot2_labels as L2
    H, W = rgb.shape[:2]
    r = L2.refine_t(rgb, q) if vertical else L2.refine_h(rgb, q)
    lim = grow(q, rules["refine_grow"], W, H)
    inside = r[0] >= lim[0] and r[1] >= lim[1] and r[2] <= lim[2] and r[3] <= lim[3] and r[2] > r[0] and r[3] > r[1]
    across_q = (q[2] - q[0]) if vertical else (q[3] - q[1])
    across_r = (r[2] - r[0]) if vertical else (r[3] - r[1])
    if inside and across_r >= rules["refine_min_across"] * across_q:
        return [int(v) for v in r], "ink"
    return [int(v) for v in q], "quad"


def meiki_weight(score, rules=RULES):
    return 0.5 + 0.5 * min(1.0, (float(score) - rules["meiki_pos_min"]) / rules["meiki_weight_span"])


def build_labels(rec, refined, meiki, W, H, rules=RULES):
    """The labels of one page: (lines, ignore, counts). rec: the pool record; refined: {line id: (box, how)};
    meiki: [{"raw", "score", "box", "vertical"}] (the cache's boxes, refined) or None."""
    lines, ignore = [], []
    c = {"positive": 0, "dialogue": 0, "name": 0, "ruby_ignore": 0, "drop_ignore": 0, "uncovered_ignore": 0,
         "meiki": 0, "meiki_meets": 0, "meiki_other": 0, "meiki_ignore": 0, "ink": 0, "quad": 0}
    for ln in rec["lines"]:
        b = xyxy(ln["box_main"] if ln.get("furigana") and ln.get("box_main") else ln["box"])
        if ln["kind"] == "ruby":
            ignore.append(grow(xyxy(ln["box"]), rules["ruby_grow"], W, H))
            c["ruby_ignore"] += 1
            continue
        if ln.get("drop") in rules["ignore_drops"]:
            ignore.append(grow(b, rules["dropped_grow"], W, H))
            c["drop_ignore"] += 1
            continue
        if ln.get("drop") not in rules["positive_drops"] or ln["kind"] not in rules["classes"]:
            ignore.append(grow(b, rules["dropped_grow"], W, H))
            c["drop_ignore"] += 1
            continue
        box, how = refined[ln["id"]]
        box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        cls = rules["classes"][ln["kind"]]
        lines.append([*box, int(bool(ln["vertical"])), cls, 1.0])
        c["positive"] += 1
        c[cls] += 1
        c[how] += 1
    page_boxes = [quad_box(b["xyxy"]) for b in rec["boxes"]]
    uncovered = set(rec.get("uncovered") or [])
    for b in rec["boxes"]:
        if b["index"] in uncovered:
            ignore.append(grow(quad_box(b["xyxy"]), rules["uncovered_grow"], W, H))
            c["uncovered_ignore"] += 1
    for m in meiki or []:
        c["meiki"] += 1
        raw = m["raw"]
        meets = False
        for q in page_boxes:
            gq = grow(q, rules["meet_grow"], W, H)
            if iou(raw, q) >= rules["meet_iou"] or inter(raw, gq) >= rules["meet_inside"] * max(1, area(raw)):
                meets = True
                break
        if meets:
            c["meiki_meets"] += 1
            continue
        box = [max(0, m["box"][0]), max(0, m["box"][1]), min(W, m["box"][2]), min(H, m["box"][3])]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if m["score"] >= rules["meiki_pos_min"]:
            lines.append([*box, int(bool(m["vertical"])), rules["other_class"], round(meiki_weight(m["score"]), 4)])
            c["meiki_other"] += 1
        else:
            ignore.append(box)
            c["meiki_ignore"] += 1
    return lines, ignore, c


# ----------------------------------------------------------------------------- meiki

_W = {}


def _init_meiki():
    from teacher import engines as E
    _W["meiki"] = E.Meiki(det_thr=RULES["meiki_det_thr"], n_threads=1)


def _meiki_one(job):
    import cv2

    import textspot2_labels as L2
    tid, page, path = job
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"title": tid, "page": page, "error": "unreadable"}
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    out = []
    for raw, score in L2.meiki_detect(_W["meiki"], bgr):
        ref = {"raw": raw, "score": score, "ref_h": L2.refine_h(rgb, raw), "ref_t": L2.refine_t(rgb, raw)}
        box, vertical = L2.refined_of(ref, "P1")
        out.append({"raw": raw, "score": round(float(score), 4), "box": [int(v) for v in box],
                    "vertical": bool(vertical)})
    return {"title": tid, "page": page, "meiki": out}


def cmd_meiki(args):
    titles = set(args.titles.split(",")) if args.titles else None
    splits = tuple(args.splits.split(","))
    rows = read_pool(args.pool, titles, splits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["title"], r["page"]))
    jobs = [(tid, r["page"], str(Path(args.pages) / r["image"])) for tid, r in rows if (tid, r["page"]) not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"meiki: {len(rows)} pages, {len(done)} cached, {len(jobs)} to run, {args.workers} workers", flush=True)
    t0 = time.time()
    files = {}
    with mp.get_context("fork").Pool(args.workers, initializer=_init_meiki) as pool:
        for i, r in enumerate(pool.imap(_meiki_one, jobs, chunksize=8)):
            f = files.get(r["title"])
            if f is None:
                f = files[r["title"]] = open(out / f"{r['title']}.jsonl", "a", encoding="utf-8")
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 2000 == 0:
                for g in files.values():
                    g.flush()
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
    for f in files.values():
        f.close()
    print(f"TEXTSPOT6_MEIKI_DONE {len(jobs)} pages in {time.time() - t0:.0f} s", flush=True)


def load_meiki(cache):
    out = {}
    for f in Path(cache).glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "meiki" in r:
                out[(r["title"], r["page"])] = r["meiki"]
    return out


# ----------------------------------------------------------------------------- build

def _page_one(job):
    """(sample, grey bytes) of one page."""
    import cv2

    from common import to_gray
    tid, rec, path, meiki, vn = job
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    H, W = rgb.shape[:2]
    if [W, H] != list(rec["dims"]):
        raise SystemExit(f"{path}: {W}x{H}, the pool says {rec['dims']}")
    refined = {}
    for ln in rec["lines"]:
        if ln["kind"] in RULES["classes"]:
            b = xyxy(ln["box_main"] if ln.get("furigana") and ln.get("box_main") else ln["box"])
            refined[ln["id"]] = refined_line(rgb, b, bool(ln["vertical"]))
    lines, ignore, counts = build_labels(rec, refined, meiki, W, H)
    grey = to_gray(rgb)
    smp = {"h": H, "w": W, "src": "hook", "kind": rec["platform"], "vn": vn, "shot": f"{tid}:{rec['page']}",
           "title": tid, "page": rec["page"], "split": rec["split"], "lines": lines, "ignore": ignore,
           "meta": {"image_sha256": rec["image_sha256"], "complete": rec["complete"], "counts": counts,
                    "roi": rec.get("roi")}}
    return smp, np.ascontiguousarray(grey).tobytes()


def title_vns(pool_dir):
    """{TID: VNDB id} from the hooked corpus' metadata (the pool records do not carry it)."""
    root = Path(os.environ.get("HOOKED", "/local/home/skerraut/herd-vita-ocr/shared/private/hooked-b2"))
    out = {}
    for meta in root.glob("*/*/metadata.json"):
        d = json.loads(meta.read_text(encoding="utf-8"))
        out[d["title_id"]] = (d.get("vndb") or {}).get("vn")
    return out


def cmd_build(args):
    titles = set(args.titles.split(",")) if args.titles else None
    rows = read_pool(args.pool, titles, SPLITS)
    meiki = load_meiki(args.meiki) if args.meiki else {}
    vns = title_vns(args.pool)
    out = Path(args.out)
    skipped = {"untrusted": 0, "no_meiki": 0, "unreadable": 0}
    groups = {}
    for tid, r in rows:
        if r.get("untrusted"):
            skipped["untrusted"] += 1
            continue
        if args.meiki and (tid, r["page"]) not in meiki:
            skipped["no_meiki"] += 1
            if not args.allow_missing_meiki:
                raise SystemExit(f"{tid}:{r['page']}: no meiki boxes in {args.meiki} (run meiki first)")
        groups.setdefault((r["split"], tid), []).append(r)
    t0 = time.time()
    written = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for (split, tid), recs in sorted(groups.items()):
            d = out / split
            d.mkdir(parents=True, exist_ok=True)
            per = RULES["shard_pages"]
            for k in range(0, len(recs), per):
                name = f"{tid}_{k // per:03d}"
                if (d / f"{name}.json").exists() and not args.force:
                    written.append((split, name))
                    continue
                jobs = [(tid, r, str(Path(args.pages) / r["image"]), meiki.get((tid, r["page"])), vns.get(tid))
                        for r in recs[k:k + per]]
                samples, off = [], 0
                with open(d / f"{name}.npy.part", "wb") as fb:
                    # an .npy header for a flat uint8 array whose length is known at the end: write the data first
                    blobs = []
                    for res in pool.imap(_page_one, jobs, chunksize=4):
                        if res is None:
                            skipped["unreadable"] += 1
                            continue
                        smp, g = res
                        smp["off"] = off
                        off += len(g)
                        samples.append(smp)
                        blobs.append(g)
                    arr = np.frombuffer(b"".join(blobs), np.uint8)
                    np.save(fb, arr)
                os.replace(d / f"{name}.npy.part", d / f"{name}.npy")
                (d / f"{name}.json").write_text(json.dumps({"samples": samples}) + "\n", encoding="utf-8")
                written.append((split, name))
                print(f"  {split}/{name}: {len(samples)} pages, {time.time() - t0:.0f} s", flush=True)
    print(f"TEXTSPOT6_PAGES_BUILT {len(written)} shards, skipped {skipped}, {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- manifest

def cmd_manifest(args):
    root = Path(args.shards)
    shards, totals = [], {}
    for split in SPLITS:
        for js in sorted((root / split).glob("*.json")):
            npy = js.with_suffix(".npy")
            samples = json.loads(js.read_text(encoding="utf-8"))["samples"]
            by_class, counts = {}, {}
            for s in samples:
                for ln in s["lines"]:
                    by_class[ln[5]] = by_class.get(ln[5], 0) + 1
                for k, v in s["meta"]["counts"].items():
                    counts[k] = counts.get(k, 0) + v
            rec = {"dir": split, "name": js.stem, "split": split, "title": samples[0]["title"] if samples else None,
                   "npy_sha256": sha256_file(npy), "json_sha256": sha256_file(js), "samples": len(samples),
                   "lines": sum(len(s["lines"]) for s in samples),
                   "vertical": sum(int(ln[4]) for s in samples for ln in s["lines"]),
                   "ignore": sum(len(s["ignore"]) for s in samples), "by_class": by_class, "counts": counts,
                   "by_kind": {k: sum(1 for s in samples if s["kind"] == k) for k in ("vita", "psp")}}
            shards.append(rec)
            t = totals.setdefault(split, {"shards": 0, "samples": 0, "lines": 0, "ignore": 0, "by_class": {}})
            t["shards"] += 1
            t["samples"] += rec["samples"]
            t["lines"] += rec["lines"]
            t["ignore"] += rec["ignore"]
            for k, v in by_class.items():
                t["by_class"][k] = t["by_class"].get(k, 0) + v
    code = {p: sha256_file(ROOT / p) for p in ("synth/textspot6_pages.py", "synth/textspot2_labels.py",
                                                "teacher/geometry.py", "teacher/engines.py")}
    pool_files = {f.name: sha256_file(f) for f in sorted(Path(args.pool).glob("*.jsonl"))}
    doc = {"kind": "textspot6-pages", "schema": 1, "issue": 128, "generator": "synth/textspot6_pages.py",
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rules": RULES, "code_sha256": code,
           "pool": {"dir": str(args.pool), "files_sha256": pool_files}, "shards": shards, "totals": totals,
           "note": "shards hold copyrighted game frames and stay local; this manifest holds names, counts, sha256"}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {totals}")


# ----------------------------------------------------------------------------- QA sheet

def cmd_sheet(args):
    import cv2
    d = Path(args.shards) / args.dir
    samples = json.loads((d / f"{args.name}.json").read_text(encoding="utf-8"))["samples"]
    data = np.load(d / f"{args.name}.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(samples), min(args.n, len(samples)), replace=False)
    colour = {"dialogue": (255, 128, 0), "name": (255, 0, 255), "notdlg": (0, 200, 255)}
    tiles = []
    for i in sorted(pick):
        s = samples[i]
        g = np.asarray(data[s["off"]:s["off"] + s["h"] * s["w"]]).reshape(s["h"], s["w"])
        im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for b in s["ignore"]:
            cv2.rectangle(im, (b[0], b[1]), (b[2] - 1, b[3] - 1), (0, 0, 255), 1)
        for ln in s["lines"]:
            cv2.rectangle(im, (ln[0], ln[1]), (ln[2] - 1, ln[3] - 1), colour.get(ln[5], (0, 255, 0)), 1)
        if s["w"] < 960:
            im = cv2.resize(im, (960, 544), interpolation=cv2.INTER_NEAREST)
        cv2.putText(im, s["shot"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])
    s = min(1.0, 1280 / max(sheet.shape[:2]))
    sheet = cv2.resize(sheet, (int(sheet.shape[1] * s), int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.out, sheet)
    print(args.out, sheet.shape)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("meiki")
    a.add_argument("--pool", required=True)
    a.add_argument("--pages", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--titles")
    a.add_argument("--splits", default="train,hval,guard")
    a.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "5")))
    a.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--pool", required=True)
    b.add_argument("--pages", required=True)
    b.add_argument("--meiki")
    b.add_argument("--out", required=True)
    b.add_argument("--titles")
    b.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "5")))
    b.add_argument("--force", action="store_true")
    b.add_argument("--allow-missing-meiki", action="store_true")
    c = sub.add_parser("manifest")
    c.add_argument("--shards", required=True)
    c.add_argument("--pool", required=True)
    c.add_argument("--out", required=True)
    d = sub.add_parser("sheet")
    d.add_argument("--shards", required=True)
    d.add_argument("--dir", required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=6)
    d.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    {"meiki": cmd_meiki, "build": cmd_build, "manifest": cmd_manifest, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    main()

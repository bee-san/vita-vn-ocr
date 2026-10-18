#!/usr/bin/env python3
"""Vertical-text evaluation (issue #99, docs/vertical.md): manifests, fixed-column CER, the column finder, bootstraps.

  vertical.py build    --reads R[,R2] --labels L[,L2] --out DIR          columns.jsonl + regions.jsonl per split
  vertical.py columns  --manifest DIR/columns.jsonl --arm ARM --model 'cmd:...' --out OUT.json
  vertical.py finder   --manifest DIR/regions.jsonl --finder py|c:BIN|det:BLOB --out OUT.json
  vertical.py e2e      --manifest DIR/regions.jsonl --finder ... --arm ARM --model 'cmd:...' --out OUT.json
  vertical.py diff     A.json B.json [--cohort vn|manga] [--n 10000]   cluster-bootstrap difference of two runs

Arms (how a column crop reaches a recognizer):
  rot   turned 90 degrees anticlockwise (synth/vertical.rotate_ccw): the dedicated recognizer VOCR5-V's input
  a0    re-tiled into upright character cells (synth/vertical.retile, "a0") for the horizontal VOCR5 models
  a1    the same with the vertical forms undone ("a1")
Every crop is first put in the console's polarity (inverted when synth/common.line_dark calls its text dark,
the flag runtime/vocr_vertical.c passes on as VOCR_INVERT). A recognizer is `cmd:COMMAND` as in eval/run.py
(runtime/build/vocr_cli BLOB --stdin reads one PGM path per line and prints one line each).

Scoring: eval/metrics.norm on both sides, Levenshtein over reference characters (CER); dialogue = columns not
marked as a speaker name. Cohorts: "vn" (console and PC visual-novel pages: the primary metric) and "manga".
Intervals: bootstrap over series clusters (each page's `cluster`), 10,000 resamples, seed 0.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "synth"))
from metrics import lev, norm  # noqa: E402

import common  # noqa: E402
import vertical_twin as V  # noqa: E402

PAD = {"vita": 4, "psp": 2}
REGION_MARGIN = {"vita": 12, "psp": 6}


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_page_grey(row):
    sys.path.insert(0, str(HERE))
    from vertical_data import load_page
    rgb = load_page(row)
    return None if rgb is None else common.to_gray(rgb)


def cohort(row):
    return "manga" if row.get("origin") == "manga" else "vn"


def boot_unit(row):
    """The bootstrap unit: the series cluster, or for manga (one series per split) the book page (amendment 1)."""
    if row.get("cohort") == "manga" or row.get("origin") == "manga":
        return row["page"].rsplit("_w", 1)[0] if row.get("page") else row["id"].rsplit("_w", 1)[0]
    return row["cluster"]


# ----------------------------------------------------------------------------- manifests

def merge_pieces(cols):
    """MeikiOCR's detector sometimes cuts one column in two at a full-width space: pieces whose columns overlap by
    70% of the narrower width, one below the other within 1.2 column widths, become one column (texts top to bottom,
    boxes united), as the column finder sees them."""
    out = []
    for c in sorted(cols, key=lambda c: c["box"][1]):
        x, y, w, h = c["box"]
        for o in out:
            ox, oy, ow, oh = o["box"]
            ov = min(x + w, ox + ow) - max(x, ox)
            gap = y - (oy + oh)
            if ov >= 0.7 * min(w, ow) and -0.2 * max(w, ow) <= gap <= 1.2 * max(w, ow):
                x0, y0 = min(x, ox), min(y, oy)
                x1, y1 = max(x + w, ox + ow), max(y + h, oy + oh)
                o["box"], o["text"] = [x0, y0, x1 - x0, y1 - y0], o["text"] + c["text"]
                o["pieces"] = o.get("pieces", [o["id"]]) + [c["id"]]
                break
        else:
            out.append(dict(c))
    return sorted(out, key=lambda c: (-(c["box"][0] + c["box"][2]), c["box"][1]))


def cmd_build(args):
    """Columns and regions of the reviewed pages. A column's box is its main-text ink box (column_box); a label
    row says its text, or why it has none (skip: ui, sfx, cut, bad, furigana, unsure, disagree). Regions: the
    labelled dialogue columns of a page grouped by adjacency (horizontal gap at most 1.5 column widths, vertical
    overlap at least 30% of the shorter), their union plus a 12 / 6 px margin; any other column inside a region
    (a name, a skipped one) is an ignore area there, as is every horizontal text box."""
    reads, labels = [], {}
    for p in args.reads.split(","):
        reads += jsonl(p)
    for p in args.labels.split(","):
        for r in jsonl(p):
            labels[r["id"]] = r
    cols, regions = [], []
    for r in reads:
        if r.get("error"):
            continue
        W, H = r["dims"]
        page_cols = []
        for c in r["columns"]:
            cid = f"{r['page']}_c{c['k']}"
            lab = labels.get(cid)
            if lab is None:
                continue
            x0, y0, x1, y1 = c["ink"]
            row = {"id": cid, "page": r["page"], "vn": r["vn"], "cluster": r["cluster"], "split": r["split"],
                   "origin": r["origin"], "plat": r["plat"], "kind": r["kind"], "cohort": cohort(r),
                   "box": [x0, y0, x1 - x0, y1 - y0], "text": lab.get("text"), "skip": lab.get("skip"),
                   "name": bool(lab.get("name")), "path": r["path"], "url": r.get("url"), "sha256": r["sha256"],
                   "dims": [W, H]}
            cols.append(row)
            page_cols.append(row)
        dia = [c for c in page_cols if c["text"] and not c["skip"] and not c["name"]]
        parent = list(range(len(dia)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(dia)):
            for j in range(i + 1, len(dia)):
                a, b = dia[i]["box"], dia[j]["box"]
                gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
                ov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if gap <= 1.5 * max(a[2], b[2]) and ov >= 0.3 * min(a[3], b[3]):
                    parent[find(i)] = find(j)
        groups = {}
        for i in range(len(dia)):
            groups.setdefault(find(i), []).append(dia[i])
        m = REGION_MARGIN[r["kind"]]
        for gi, g in enumerate(sorted(groups.values(), key=lambda g: -max(c["box"][0] + c["box"][2] for c in g))):
            x0 = max(0, min(c["box"][0] for c in g) - m)
            y0 = max(0, min(c["box"][1] for c in g) - m)
            x1 = min(W, max(c["box"][0] + c["box"][2] for c in g) + m)
            y1 = min(H, max(c["box"][1] + c["box"][3] for c in g) + m)
            inside = lambda b: b[0] < x1 and b[0] + b[2] > x0 and b[1] < y1 and b[1] + b[3] > y0  # noqa: E731
            order = merge_pieces(sorted(g, key=lambda c: (-(c["box"][0] + c["box"][2]), c["box"][1])))
            ignore = [c["box"] for c in page_cols if c not in g and inside(c["box"])]
            ignore += [[h["det"][0], h["det"][1], h["det"][2] - h["det"][0], h["det"][3] - h["det"][1]]
                       for h in r.get("horizontal", []) if inside([h["det"][0], h["det"][1], h["det"][2] - h["det"][0],
                                                                    h["det"][3] - h["det"][1]])]
            regions.append({"id": f"{r['page']}_r{gi}", "page": r["page"], "vn": r["vn"], "cluster": r["cluster"],
                            "split": r["split"], "origin": r["origin"], "plat": r["plat"], "kind": r["kind"],
                            "cohort": cohort(r), "region": [x0, y0, x1 - x0, y1 - y0],
                            "columns": [{"id": c["id"], "box": c["box"], "text": c["text"]} for c in order],
                            "ignore": ignore, "path": r["path"], "url": r.get("url"), "sha256": r["sha256"],
                            "dims": [W, H]})
    out = Path(args.out)
    for split in sorted({c["split"] for c in cols}):
        write_jsonl(out / f"columns_{split}.jsonl", [c for c in cols if c["split"] == split])
        write_jsonl(out / f"regions_{split}.jsonl", [g for g in regions if g["split"] == split])
    from collections import Counter
    print(Counter((c["split"], c["cohort"], c["skip"] or ("name" if c["name"] else "label")) for c in cols))
    print(Counter((g["split"], g["cohort"]) for g in regions))


# ----------------------------------------------------------------------------- crops and recognizers

def column_crop(grey, box, kind, pad=None):
    """The column's crop as the console cuts it (box plus pad, clipped) in the console's polarity: (crop, dark)."""
    x, y, w, h = box
    p = PAD[kind] if pad is None else pad
    H, W = grey.shape
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    dark = common.line_dark(grey, (x0, y0, x1 - x0, y1 - y0))
    crop = grey[y0:y1, x0:x1]
    return (255 - crop if dark else crop.copy()), dark


def arm_input(crop, arm):
    if arm == "rot":
        return V.rotate_ccw(crop)
    if arm in ("a0", "a1"):
        return V.retile(crop, arm)
    raise SystemExit(f"unknown arm {arm}")


def write_pgm(path, g):
    g = np.ascontiguousarray(g, np.uint8)
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (g.shape[1], g.shape[0]))
        f.write(g.tobytes())


def recognize(model, images):
    """Texts of `images` (grey uint8 arrays) read by `cmd:COMMAND` (one PGM path per stdin line)."""
    if not model.startswith("cmd:"):
        raise SystemExit("only cmd: recognizers")
    with tempfile.TemporaryDirectory(prefix="vert_") as td:
        paths = []
        for i, g in enumerate(images):
            p = os.path.join(td, f"{i:06d}.pgm")
            write_pgm(p, g)
            paths.append(p)
        r = subprocess.run(model[4:], shell=True, input="\n".join(paths) + "\n", capture_output=True, text=True,
                           timeout=3600)
        if r.returncode:
            raise SystemExit(f"recognizer failed: {r.stderr[-2000:]}")
        out = r.stdout.split("\n")
        if len(out) < len(images):
            raise SystemExit(f"recognizer printed {len(out)} lines for {len(images)} crops")
        return out[:len(images)]


def score_rows(rows):
    """{cohort: {"chars", "errs", "cer", "lines", "exact"}} and per-cluster sums, over dialogue rows."""
    out = {}
    for r in rows:
        if r.get("name") or r.get("skip") or not r.get("text"):
            continue
        for k in (r["cohort"], "all"):
            o = out.setdefault(k, {"chars": 0, "errs": 0, "lines": 0, "exact": 0, "clusters": {}})
            ref, hyp = norm(r["text"]), norm(r.get("pred", ""))
            e = lev(hyp, ref)
            o["chars"] += len(ref)
            o["errs"] += e
            o["lines"] += 1
            o["exact"] += int(e == 0)
            c = o["clusters"].setdefault(r["cluster"], [0, 0])
            c[0] += e
            c[1] += len(ref)
    for o in out.values():
        o["cer"] = o["errs"] / max(1, o["chars"])
    return out


def cmd_columns(args):
    rows = jsonl(args.manifest)
    rows = [r for r in rows if r.get("text") and not r.get("skip")]
    if args.limit:
        rows = rows[:args.limit]
    cache, imgs = {}, []
    for r in rows:
        g = cache.get(r["page"])
        if g is None:
            cache.clear()
            g = cache[r["page"]] = load_page_grey({"path": r["path"], "kind": r["kind"]})
        crop, dark = column_crop(g, r["box"], r["kind"])
        imgs.append(arm_input(crop, args.arm))
        r["dark"] = dark
    preds = recognize(args.model, imgs)
    for r, p in zip(rows, preds):
        r["pred"] = p
    res = {"arm": args.arm, "model": args.model, "manifest": args.manifest, "scores": score_rows(rows),
           "rows": [{**{k: r[k] for k in ("id", "cohort", "name", "text", "pred", "dark")}, "cluster": boot_unit(r)}
                    for r in rows]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=0)
    for k, o in sorted(res["scores"].items()):
        print(f"{k}: CER {100 * o['cer']:.2f}% ({o['errs']}/{o['chars']}), exact {o['exact']}/{o['lines']}")


# ----------------------------------------------------------------------------- finder

def find(finder, region_grey):
    if finder.startswith("py"):
        b, d, t = V.find_columns(region_grey, int(finder[2:] or 0))
        return b, d
    if finder.startswith("c:"):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "r.pgm")
            write_pgm(p, region_grey)
            r = subprocess.run([finder[2:], p], capture_output=True, text=True, timeout=60)
            boxes, dark = [], []
            for line in r.stdout.splitlines():
                v = [int(x) for x in line.split()]
                if len(v) >= 4:
                    boxes.append(tuple(v[:4]))
                    dark.append(v[4] if len(v) > 4 else 0)
            return boxes, dark
    raise SystemExit(f"unknown finder {finder}")


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u else 0.0


def match(dets, refs, ignore, thr=0.5):
    """One-to-one matching at IoU >= thr (greedy by IoU); detections over an ignore box (half their area inside)
    are left out. Returns (pairs [(det, ref)], unmatched dets, unmatched refs)."""
    def inside(d, b):
        ix = max(0, min(d[0] + d[2], b[0] + b[2]) - max(d[0], b[0]))
        iy = max(0, min(d[1] + d[3], b[1] + b[3]) - max(d[1], b[1]))
        return ix * iy >= 0.5 * d[2] * d[3]
    cand = sorted(((iou(d, r), i, j) for i, d in enumerate(dets) for j, r in enumerate(refs)), reverse=True)
    used_d, used_r, pairs = set(), set(), []
    for v, i, j in cand:
        if v < thr or i in used_d or j in used_r:
            continue
        used_d.add(i)
        used_r.add(j)
        pairs.append((i, j))
    free_d = [i for i in range(len(dets)) if i not in used_d and not any(inside(dets[i], b) for b in ignore)]
    free_r = [j for j in range(len(refs)) if j not in used_r]
    return pairs, free_d, free_r


def cmd_finder(args):
    rows = jsonl(args.manifest)
    tot = {"tp": 0, "fp": 0, "fn": 0, "regions": 0, "exact": 0}
    per = []
    cache = {}
    for r in rows:
        g = cache.get(r["page"])
        if g is None:
            cache.clear()
            g = cache[r["page"]] = load_page_grey({"path": r["path"], "kind": r["kind"]})
        x, y, w, h = r["region"]
        reg = g[y:y + h, x:x + w]
        dets, dark = find(args.finder, reg)
        refs = [[c["box"][0] - x, c["box"][1] - y, c["box"][2], c["box"][3]] for c in r["columns"]]
        ign = [[b[0] - x, b[1] - y, b[2], b[3]] for b in r["ignore"]]
        pairs, fd, fr = match(dets, refs, ign, args.iou)
        tot["tp"] += len(pairs)
        tot["fp"] += len(fd)
        tot["fn"] += len(fr)
        tot["regions"] += 1
        order_ok = [j for _, j in sorted(pairs)] == sorted(j for _, j in pairs)
        tot["exact"] += int(not fd and not fr and order_ok)
        per.append({"id": r["id"], "cohort": r["cohort"], "cluster": r["cluster"], "dets": dets, "dark": dark,
                    "tp": len(pairs), "fp": len(fd), "fn": len(fr), "order_ok": order_ok})
    p = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    rc = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    tot.update(precision=p, recall=rc, f1=2 * p * rc / max(1e-9, p + rc),
               region_exact=tot["exact"] / max(1, tot["regions"]))
    json.dump({"finder": args.finder, "totals": tot, "regions": per}, open(args.out, "w"), indent=0)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in tot.items()})


def cmd_e2e(args):
    """Whole regions: finder -> columns (reading order) -> crops -> recognizer; the joined text against the region's
    joined reference (dialogue columns, right to left). A detection over an ignore box is not read."""
    rows = jsonl(args.manifest)
    cache, imgs, owners = {}, [], []
    plans = []
    for k, r in enumerate(rows):
        g = cache.get(r["page"])
        if g is None:
            cache.clear()
            g = cache[r["page"]] = load_page_grey({"path": r["path"], "kind": r["kind"]})
        x, y, w, h = r["region"]
        reg = g[y:y + h, x:x + w]
        dets, _ = find(args.finder, reg)
        ign = [[b[0] - x, b[1] - y, b[2], b[3]] for b in r["ignore"]]
        _, keep, _ = match(dets, [], ign, 1.1)
        keep = sorted(keep)
        for i in keep:
            crop, dark = column_crop(reg, dets[i], r["kind"], pad=0)
            imgs.append(arm_input(crop, args.arm))
            owners.append(k)
        plans.append(len(keep))
    preds = recognize(args.model, imgs) if imgs else []
    texts = [""] * len(rows)
    for k, p in zip(owners, preds):
        texts[k] += norm(p)
    out_rows = []
    for r, t in zip(rows, texts):
        ref = "".join(norm(c["text"]) for c in r["columns"])
        out_rows.append({"id": r["id"], "cluster": boot_unit(r), "cohort": r["cohort"], "text": ref, "pred": t})
    res = {"finder": args.finder, "arm": args.arm, "model": args.model, "scores": score_rows(out_rows),
           "rows": out_rows}
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=0)
    for k, o in sorted(res["scores"].items()):
        print(f"{k}: CER {100 * o['cer']:.2f}% ({o['errs']}/{o['chars']}), exact {o['exact']}/{o['lines']}")


# ----------------------------------------------------------------------------- bootstrap

def cluster_diff(a_rows, b_rows, cohort_name, n=10000, seed=0):
    """CER(b) - CER(a) in points over the rows both scored, with a 95% bootstrap interval over clusters."""
    def per_cluster(rows):
        out = {}
        for r in rows:
            if r.get("name") or (cohort_name != "all" and r["cohort"] != cohort_name):
                continue
            ref, hyp = norm(r["text"]), norm(r.get("pred", ""))
            c = out.setdefault(r["cluster"], {})
            c[r["id"]] = (lev(hyp, ref), len(ref))
        return out
    A, B = per_cluster(a_rows), per_cluster(b_rows)
    keys = sorted(set(A) & set(B))
    ea = np.array([sum(A[k][i][0] for i in A[k] if i in B[k]) for k in keys], np.float64)
    eb = np.array([sum(B[k][i][0] for i in B[k] if i in A[k]) for k in keys], np.float64)
    nc = np.array([sum(A[k][i][1] for i in A[k] if i in B[k]) for k in keys], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    d = (eb[idx].sum(1) - ea[idx].sum(1)) / np.maximum(1, nc[idx].sum(1)) * 100
    point = (eb.sum() - ea.sum()) / max(1, nc.sum()) * 100
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"clusters": len(keys), "chars": int(nc.sum()), "a": ea.sum() / max(1, nc.sum()) * 100,
            "b": eb.sum() / max(1, nc.sum()) * 100, "diff": point, "lo": float(lo), "hi": float(hi)}


def cmd_diff(args):
    a, b = json.load(open(args.a)), json.load(open(args.b))
    r = cluster_diff(a["rows"], b["rows"], args.cohort, args.n)
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--reads", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("columns")
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("finder")
    p.add_argument("--manifest", required=True)
    p.add_argument("--finder", default="py")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("e2e")
    p.add_argument("--manifest", required=True)
    p.add_argument("--finder", default="py")
    p.add_argument("--arm", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("diff")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--cohort", default="vn")
    p.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    {"build": cmd_build, "columns": cmd_columns, "finder": cmd_finder, "e2e": cmd_e2e, "diff": cmd_diff}[args.cmd](args)


if __name__ == "__main__":
    main()

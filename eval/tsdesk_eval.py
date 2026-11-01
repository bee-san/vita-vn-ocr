#!/usr/bin/env python3
"""textspot-desktop's evaluation (issue #173, eval/textspot_desktop_protocol.json), on vocr-desktop's harness
(eval/desktop.py, eval/desktop_time.py), which this file installs into at run time instead of editing it.

  tsdesk_eval.py desktop CMD ...  eval/desktop.py's command line (frames / detect / read / compare / boxes-pr / meiki)
                                 with this work's detectors installed: tsd:MODEL.onnx[@SIZE] (tools/tsdesk.Detector),
                                 ctsd:BLOB.vtsd[@SIZE] (the C path, when built), and the dense sets' ignore filter
                                 (eval/tsdesk_dense.install_ignore: a box at least half inside a 1b ignore area is
                                 dropped before reading)
  tsdesk.py edges --frames HVAL.pkl --boxes BOXES.jsonl --out E.json [--rec L_fast.onnx] [--workers 5]
      the protocol's edge error and tightness cost on hooked pages: every reference line of the units, its box
      (box_main for a line with furigana) refined to ink by synth/textspot6_pages.refined_line on the page's colour
      frame (a line the refinement leaves at its quad is counted and left out); the detector's horizontal boxes as they
      are (no pad) matched one to one (vertical overlap >= half the smaller height, horizontal overlap >= half the
      smaller width, largest overlap first); per edge e = detector - reference (x0, y0, x1, y1) in frame px; then every
      matched line read twice by the recognizer (greedy), from its detector box + pad auto and from its exact box + pad
      auto: the CER of each over those lines and the paired title-cluster interval of the difference
  tsdesk.py screen --runs DIR [--arms A,B,...] [--out S.json]
      the screening table: each arm's four reads (hval and dense_screen at fit:960 and long:960), their mean (the
      protocol's score), PCSG00854's junk boxes, and the shape rule's network time and MACs

Everything read here is private (copyrighted frames): BOXES and per-line rows stay on the shared box; E.json and S.json
hold counts only.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "tools", ROOT / "synth", ROOT / "train", ROOT / "teacher", ROOT / "runtime", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import desktop as D  # noqa: E402

PROTOCOL = ROOT / "eval" / "textspot_desktop_protocol.json"
READ_PAD = os.environ.get("TSDESK_PAD", "auto")      # the reading pad of the tsd: / ctsd: timing pipelines
SCHEMA = "textspot-desktop/1"


# ----------------------------------------------------------------------------- the harness hooks

class CTsdDet:
    """The C path's detector (desktop/, tools/tsdesk_c.py) on a VTSD blob."""

    name = "ctsd"

    def __init__(self, blob, size=None, threads=1):
        import tsdesk_c as TC
        self.m, self.size = TC.CModel(blob, threads=threads), size

    def boxes(self, rgb):
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag] for b in
                self.m.detect(rgb, self.size)]


def detector_from_spec(spec, threads=1, so=None):
    import tsdesk as T
    if spec.startswith("tsd:"):
        path, _, size = spec[4:].partition("@")
        return T.Detector(path, session_options=so, size=size or None)
    if spec.startswith("ctsd:"):
        path, _, size = spec[5:].partition("@")
        return CTsdDet(path, size or None, threads)
    raise ValueError(spec)


def install(Dm=D):
    """Register the tsd: / ctsd: detector specs in eval/desktop.py's detector() (which desktop.py detect and
    desktop_time.py's pipelines call), give them R's reading pad in desktop_time.py's pipelines, and install the dense
    sets' ignore filter. desktop.py itself is not changed."""
    if getattr(Dm, "_tsdesk_installed", False):
        return
    import tsdesk_dense
    base = Dm.detector

    def detector(spec, threads=1, opt="all", spinning=None):
        if spec.startswith(("tsd:", "ctsd:")):
            return detector_from_spec(spec, threads, Dm.session_options(threads, opt, spinning))
        return base(spec, threads, opt, spinning)

    Dm.detector, Dm._tsdesk_installed = detector, True
    tsdesk_dense.install_ignore(Dm)
    base_pad, base_crop = Dm.pad_of, Dm.crop_box

    def pad_of(spec, H):
        """eval/desktop.py's pad specs plus the protocol's rules: 'kN' = max(2, round(N H / 544)) px on every side,
        'kAvB' = A horizontally and B vertically (a tuple)."""
        s = str(spec)
        if s.startswith("k"):
            kh, _, kv = s[1:].partition("v")
            ph, pv = pad_px(float(kh), H), pad_px(float(kv or kh), H)
            return ph if ph == pv else (ph, pv)
        return base_pad(spec, H)

    def crop_box(b, pad, W, H):
        if isinstance(pad, tuple):
            return crop_hv(b, pad, W, H)
        return base_crop(b, pad, W, H)

    Dm.pad_of, Dm.crop_box = pad_of, crop_box
    try:
        import desktop_time as DT
    except ImportError:
        return
    if "tsdesk_eval" not in DT.MEM_CHILD:      # the fresh processes of desktop_time.py mem know the tsd: specs too
        DT.MEM_CHILD = DT.MEM_CHILD.replace("import desktop_time as DT\n",
                                            "import desktop_time as DT\nimport tsdesk_eval\ntsdesk_eval.install(D)\n")
    init = DT.Pipeline.__init__

    def pipeline_init(self, spec, threads, mode, *a, **k):
        init(self, spec, threads, mode, *a, **k)
        if spec.startswith(("tsd:", "ctsd:")):
            self.pad = READ_PAD
    DT.Pipeline.__init__ = pipeline_init


# ----------------------------------------------------------------------------- edges and the tightness cost

def match(dets, refs):
    """One-to-one pairs (i, j) of detector boxes and reference boxes ([x0, y0, x1, y1]): vertical overlap >= half the
    smaller height and horizontal overlap >= half the smaller width, the largest overlap area first."""
    pairs = []
    for i, d in enumerate(dets):
        for j, r in enumerate(refs):
            dx = min(d[2], r[2]) - max(d[0], r[0])
            dy = min(d[3], r[3]) - max(d[1], r[1])
            if dx >= 0.5 * min(d[2] - d[0], r[2] - r[0]) and dy >= 0.5 * min(d[3] - d[1], r[3] - r[1]):
                pairs.append((dx * dy, i, j))
    used_d, used_r, out = set(), set(), []
    for _, i, j in sorted(pairs, key=lambda t: (-t[0], t[1], t[2])):
        if i not in used_d and j not in used_r:
            used_d.add(i)
            used_r.add(j)
            out.append((i, j))
    return out


_E = {}


def _init_edges(rec_path, pad=(4, 4)):
    import vocr_onnx as V
    _E["rec"] = V.Recognizer(rec_path, session_options=D.session_options(1))
    _E["V"] = V
    _E["pad"] = tuple(pad)


def exact_ref(fr, rgb, ln):
    """(box [x0, y0, x1, y1], how) of a unit line's exact box (protocol amendment 1): on hooked frames the ink box
    inside the line's glyph cells (train/tsdesk_targets.ink_box on the cells' union); elsewhere the line's box as it
    is (the dense sets' 1b boxes are ink boxes already), how "box"."""
    import tsdesk_targets as TT
    if fr["src"]["kind"] == "hooked":
        cl = np.asarray(ln["cells"], np.float64) if ln.get("cells") else None
        if cl is not None and len(cl):
            c = [float(cl[:, 0].min()), float(cl[:, 1].min()), float(cl[:, 2].max()), float(cl[:, 3].max())]
        else:
            x, y, w, h = ln["box"]
            c = [x, y, x + w, y + h]
        return TT.ink_box(rgb, c)
    x, y, w, h = ln["box"]
    return [x, y, x + w, y + h], "box"


def crop_hv(b, p, W, H):
    """[x0, y0, x1, y1] grown by p = (horizontal, vertical) px, rounded outward, clipped: (x, y, w, h)."""
    import math
    ph, pv = p
    x0, y0 = max(0, int(math.floor(b[0] - ph))), max(0, int(math.floor(b[1] - pv)))
    x1, y1 = min(W, int(math.ceil(b[2] + ph))), min(H, int(math.ceil(b[3] + pv)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _edges_job(job):
    """One frame: (rows, skipped): a row per matched reference line."""
    import metrics
    import pagepath as PP
    fr, boxes = job
    V = _E["V"]
    rgb = D.load_rgb(fr)
    g = V.grey_of(rgb)
    H, W = g.shape
    p = pad_px(_E["pad"][0], H), pad_px(_E["pad"][1], H)
    refs, meta, skipped = [], [], 0
    for u in fr["units"]:
        for ln in u["lines"]:
            box, how = exact_ref(fr, rgb, ln)
            if how not in ("ink", "box"):
                skipped += 1
                continue
            refs.append([float(v) for v in box])
            meta.append((u["uid"], u["cluster"], ln["text"]))
    dets = [[float(v) for v in b[:4]] for b in boxes if not b[4]]
    rows = []
    for i, j in match(dets, refs):
        d, r = dets[i], refs[j]
        uid, cluster, text = meta[j]
        ref = metrics.norm(text)
        texts = []
        for b in (d, r):
            cb = crop_hv(b, p, W, H)
            dark = V.line_dark(g, cb)
            texts.append(_E["rec"].read(g[cb[1]:cb[1] + cb[3], cb[0]:cb[0] + cb[2]], invert=bool(dark)).text)
        rows.append({"frame": fr["key"], "uid": uid, "cluster": cluster, "view": fr["view"], "ref_len": len(ref),
                     "e": [d[0] - r[0], d[1] - r[1], d[2] - r[2], d[3] - r[3]], "height": r[3] - r[1],
                     "err_det": PP.lev(metrics.norm(texts[0]), ref), "err_exact": PP.lev(metrics.norm(texts[1]), ref)})
    return rows, skipped, len(refs)


def edge_stats(rows):
    if not rows:
        return {"lines": 0}
    e = np.array([r["e"] for r in rows], np.float64)
    out = {"lines": len(rows)}
    for k, name in enumerate(("left", "top", "right", "bottom")):
        a = np.abs(e[:, k])
        out[name] = {"median_abs": round(float(np.median(a)), 3), "p90_abs": round(float(np.percentile(a, 90)), 3),
                     "mean": round(float(e[:, k].mean()), 3), "share_within_1px": round(float((a <= 1).mean()), 4),
                     "share_within_2px": round(float((a <= 2).mean()), 4)}
    out["mean_median_abs"] = round(float(np.mean([out[n]["median_abs"] for n in ("left", "top", "right",
                                                                                  "bottom")])), 3)
    return out


def cmd_edges(a):
    import pagepath as PP
    frames, fmeta = D.load_frames(a.frames)
    boxes, bmeta = D.load_boxes(a.boxes)
    if bmeta["frames_sha256"] != D.sha256_file(a.frames):
        raise SystemExit("the boxes were found on another frames file")
    jobs = [(fr, boxes[fr["key"]]) for fr in frames]
    t0 = time.time()
    rows, skipped, n_refs = [], 0, 0
    pad = PAD_RULES[a.pad]
    with D._pool(a.workers, _init_edges, (a.rec, pad)) as pool:
        for rs, sk, nr in pool.imap(_edges_job, jobs, chunksize=4):
            rows.extend(rs)
            skipped += sk
            n_refs += nr
    res = {"all": edge_stats(rows)}
    for v in sorted({r["view"] for r in rows}):
        res[v] = edge_stats([r for r in rows if r["view"] == v])
    by_line_a = [{"uid": f"{r['frame']}|{k}", "cluster": r["cluster"], "errors": r["err_det"], "chars": r["ref_len"]}
                 for k, r in enumerate(rows)]
    by_line_b = [{"uid": f"{r['frame']}|{k}", "cluster": r["cluster"], "errors": r["err_exact"], "chars": r["ref_len"]}
                 for k, r in enumerate(rows)]
    cost = {"all": PP.cluster_bootstrap(by_line_a, by_line_b)}
    for v in sorted({r["view"] for r in rows}):
        ia = [x for x, r in zip(by_line_a, rows) if r["view"] == v]
        ib = [x for x, r in zip(by_line_b, rows) if r["view"] == v]
        cost[v] = PP.cluster_bootstrap(ia, ib)
    x854 = [i for i, r in enumerate(rows) if r["cluster"] != "PCSG00854"]
    cost["all_without_PCSG00854"] = PP.cluster_bootstrap([by_line_a[i] for i in x854], [by_line_b[i] for i in x854])
    doc = {"schema": SCHEMA, "kind": "textspot-desktop-edges", "frames_sha256": D.sha256_file(a.frames),
           "boxes_sha256": D.sha256_file(a.boxes), "det": bmeta["det"], "rec": Path(a.rec).name,
           "rec_sha256": D.sha256_file(a.rec), "pad": a.pad, "frames_meta": fmeta,
           "references": {"lines": n_refs + skipped, "ink": n_refs, "left_at_quad": skipped,
                          "matched": len(rows), "matched_share": round(len(rows) / max(1, n_refs), 4)},
           "edges": res, "tightness_cost": cost, "seconds": round(time.time() - t0, 1),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    if a.rows:
        D.write_jsonl(a.rows, rows)
    e = res["all"]
    print(f"edges: {e['lines']} lines matched of {n_refs} (+{skipped} at quad); median |e| L/T/R/B "
          f"{e['left']['median_abs']}/{e['top']['median_abs']}/{e['right']['median_abs']}/{e['bottom']['median_abs']} "
          f"p90 {e['left']['p90_abs']}/{e['top']['p90_abs']}/{e['right']['p90_abs']}/{e['bottom']['p90_abs']}; "
          f"cost {cost['all']['arm_cer']:.3f} vs {cost['all']['base_cer']:.3f} delta {cost['all']['delta']:+.3f} "
          f"CI {cost['all']['ci95']}", flush=True)


# ----------------------------------------------------------------------------- the reading pad on exact boxes

PAD_RULES = {"k3": (3, 3), "k4": (4, 4), "k5": (5, 5), "k6": (6, 6), "k8": (8, 8), "k4v6": (4, 6), "k6v4": (6, 4)}


def pad_px(k, H):
    """max(2, round(k H / 544)) frame px (k = 4: eval/desktop.py's pad 'auto')."""
    return max(2, int(round(k * H / 544)))


def exact_boxes(fr, rgb):
    """[(unit index, line index, [x0, y0, x1, y1], how)] of a frame's reference lines (exact_ref)."""
    out = []
    for ui, u in enumerate(fr["units"]):
        for li, ln in enumerate(u["lines"]):
            box, how = exact_ref(fr, rgb, ln)
            out.append((ui, li, [float(v) for v in box], how))
    return out


def _pad_job(fr):
    import math

    import metrics
    import pagepath as PP
    V = _E["V"]
    rgb = D.load_rgb(fr)
    g = V.grey_of(rgb)
    H, W = g.shape
    rows = []
    for ui, li, b, how in exact_boxes(fr, rgb):
        u = fr["units"][ui]
        ref = metrics.norm(u["lines"][li]["text"])
        r = {"uid": f"{u['uid']}|{li}", "cluster": u["cluster"], "view": fr["view"], "chars": len(ref), "how": how}
        for name, (kh, kv) in PAD_RULES.items():
            ph, pv = pad_px(kh, H), pad_px(kv, H)
            x0, y0 = max(0, int(math.floor(b[0] - ph))), max(0, int(math.floor(b[1] - pv)))
            x1, y1 = min(W, int(math.ceil(b[2] + ph))), min(H, int(math.ceil(b[3] + pv)))
            cb = (x0, y0, x1 - x0, y1 - y0)
            dark = V.line_dark(g, cb)
            t = _E["rec"].read(g[y0:y1, x0:x1], invert=bool(dark)).text
            r[name] = PP.lev(metrics.norm(t), ref)
        rows.append(r)
    return rows


def cmd_pads(a):
    """The protocol's 'exact' reading, detector-free: every reference line's exact box read at each pad rule."""
    import pagepath as PP
    out = {"schema": SCHEMA, "kind": "textspot-desktop-pads", "rules": PAD_RULES, "rec": Path(a.rec).name,
           "rec_sha256": D.sha256_file(a.rec), "sets": {},
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for fpath in a.frames:
        frames, _ = D.load_frames(fpath)
        rows = []
        with D._pool(a.workers, _init_edges, (a.rec,)) as pool:
            for rs in pool.imap(_pad_job, frames, chunksize=4):
                rows.extend(rs)
        st = {"frames_sha256": D.sha256_file(fpath), "lines": len(rows),
              "how": {k: sum(1 for r in rows if r["how"] == k) for k in sorted({r["how"] for r in rows})}}
        subs = [("all", None), ("without_PCSG00854", "x854")] + [(v, v) for v in sorted({r["view"] for r in rows})]
        for sub, key in subs:
            if key is None:
                rr = rows
            elif key == "x854":
                rr = [r for r in rows if r["cluster"] != "PCSG00854"]
            else:
                rr = [r for r in rows if r["view"] == key]
            c = sum(r["chars"] for r in rr)
            st[sub] = {name: round(100.0 * sum(r[name] for r in rr) / max(1, c), 4) for name in PAD_RULES}
        base = [{"uid": r["uid"], "cluster": r["cluster"], "errors": r["k4"], "chars": r["chars"]} for r in rows]
        st["vs_k4"] = {name: PP.cluster_bootstrap([{"uid": r["uid"], "cluster": r["cluster"], "errors": r[name],
                                                    "chars": r["chars"]} for r in rows], base)
                       for name in PAD_RULES if name != "k4"}
        by = {}
        for r in rows:
            b = by.setdefault(r["cluster"], {n: 0 for n in PAD_RULES})
            for n in PAD_RULES:
                b[n] += r[n]
        st["worst_clusters_k4"] = sorted(([k, v] for k, v in by.items()), key=lambda kv: -kv[1]["k4"])[:8]
        out["sets"][Path(fpath).stem] = st
        if a.rows:
            D.write_jsonl(f"{a.rows}.{Path(fpath).stem}.jsonl", rows)
        print(Path(fpath).stem, json.dumps({k: st[k] for k in ("lines", "how", "all", "without_PCSG00854")}),
              flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def _exact_job(fr):
    rgb = D.load_rgb(fr)
    out = []
    for _, _, b, _how in exact_boxes(fr, rgb):
        out.append([b[0], b[1], b[2], b[3], 0, 9.0, 0])
    return {"key": fr["key"], "W": fr["W"], "H": fr["H"], "boxes": out, "ms": 0.0}


def cmd_exact_boxes(a):
    """The exact boxes of every unit line as a detector's boxes file (eval/desktop.py detect's format): the perfect
    boxes row (protocol: baselines.exact)."""
    frames, _ = D.load_frames(a.frames)
    t0 = time.time()
    rows = []
    with D._pool(a.workers, None, ()) as pool:
        for r in pool.imap(_exact_job, frames, chunksize=4):
            rows.append(r)
    meta = {"meta": {"det": "exact", "frames_file": str(a.frames), "frames_sha256": D.sha256_file(a.frames),
                     "rule": "eval/tsdesk.exact_ref (amendment 1)", "repo_head": D.git_head(),
                     "seconds": round(time.time() - t0, 1)}}
    D.write_jsonl(a.out, [meta, *rows])
    print(f"exact boxes: {len(rows)} frames, {sum(len(r['boxes']) for r in rows)} boxes -> {a.out}", flush=True)


# ----------------------------------------------------------------------------- timing pages

def cmd_pages(a):
    """The protocol's timing pages (metrics.timing_pages), in eval/desktop_time.py's pages format: seed 20261101, 60
    hval Vita frames (960x544), 60 hval PSP frames (480x272), the same 60 Vita frames resampled to 1920x1080
    (cv2.INTER_LINEAR), and 30 1b v3 training frames of exactly 1920x1080 (none of the dense sets' VNs)."""
    import pickle
    import random

    import cv2
    rng = random.Random(a.seed)
    frames, _ = D.load_frames(a.hval)
    pages = []
    picked = {}
    for view in ("vita", "psp"):
        fs = sorted((f for f in frames if f["view"] == view), key=lambda f: f["key"])
        picked[view] = sorted(rng.sample(fs, min(a.n, len(fs))), key=lambda f: f["key"])
        for f in picked[view]:
            D.check_sha(f)
            pages.append({"key": f["key"], "set": "hval", "view": view, "W": f["W"], "H": f["H"], "rgb": D.load_rgb(f)})
    for f in picked["vita"]:
        rgb = cv2.resize(D.load_rgb(f), (1920, 1080), interpolation=cv2.INTER_LINEAR)
        pages.append({"key": f["key"] + "@1080", "set": "hval", "view": "vita1080", "W": 1920, "H": 1080,
                      "rgb": np.ascontiguousarray(rgb)})
    skip = set(Path(a.skip_vns).read_text(encoding="utf-8").split())
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    plan = {r["shot"]: r for r in json.loads(Path(a.plan).read_text(encoding="utf-8"))["rows"]}
    cands = []
    for s in man["shards"]:
        js = json.loads((Path(a.shards) / s["dir"] / f"{s['name']}.json").read_text(encoding="utf-8"))
        for smp in js["samples"]:
            if smp["w"] == 1920 and smp["h"] == 1080 and smp["vn"] not in skip and smp["shot"] in plan:
                cands.append(smp["shot"])
    for shot in sorted(rng.sample(sorted(cands), min(a.n_pc, len(cands)))):
        pr = plan[shot]
        if D.sha256_file(pr["path"]) != pr["sha256"]:
            raise SystemExit(f"{pr['path']}: sha256 differs from the plan's")
        bgr = cv2.imread(pr["path"], cv2.IMREAD_COLOR)
        pages.append({"key": shot, "set": "1b", "view": "pc1080", "W": bgr.shape[1], "H": bgr.shape[0],
                      "rgb": np.ascontiguousarray(bgr[:, :, ::-1])})
    with open(a.out, "wb") as f:
        pickle.dump({"pages": pages, "seed": a.seed, "source": "eval/tsdesk_eval.py pages"}, f)
    from collections import Counter
    print(json.dumps({"pages": len(pages), "views": dict(Counter(p["view"] for p in pages))}), flush=True)


# ----------------------------------------------------------------------------- the screening table

def _cer(doc, drop=None):
    U = [u for u in doc["units"] if drop is None or u["cluster"] != drop]
    e, c = sum(u["errors"] for u in U), sum(u["chars"] for u in U)
    return 100.0 * e / max(1, c)


def _junk(doc, cluster=None):
    return sum(u["junk"] for u in doc["units"] if cluster is None or u["cluster"] == cluster)


def cmd_screen(a):
    res = Path(a.res)
    arms = a.arms.split(",")
    table = {}
    for arm in arms:
        row = {}
        vals = []
        for st in ("hval", "dense_screen"):
            for pol in ("fit960", "long960"):
                p = res / f"{st}_{arm}_{pol}_L_fast.json"
                if not p.exists():
                    row[f"{st}_{pol}"] = None
                    continue
                doc = json.loads(p.read_text(encoding="utf-8"))
                row[f"{st}_{pol}"] = round(_cer(doc), 4)
                vals.append(row[f"{st}_{pol}"])
                if st == "hval":
                    row[f"{st}_{pol}_without_PCSG00854"] = round(_cer(doc, "PCSG00854"), 4)
                    row[f"{st}_{pol}_junk_PCSG00854"] = _junk(doc, "PCSG00854")
                    row[f"{st}_{pol}_junk"] = _junk(doc)
                    a_ = doc["summary"]["all"]
                    row[f"{st}_{pol}_line_recall"] = a_["line_recall"]
                else:
                    row[f"{st}_{pol}_line_recall"] = doc["summary"]["all"]["line_recall"]
        row["score"] = round(float(np.mean(vals)), 4) if len(vals) == 4 else None
        table[arm] = row
    out = {"schema": SCHEMA, "kind": "textspot-desktop-screen", "arms": table,
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for arm, row in table.items():
        print(arm, json.dumps(row), flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "desktop":
        install(D)
        return D.main(argv[1:])
    if argv and argv[0] == "time":
        install(D)
        import desktop_time as DT
        return DT.main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("edges")
    p.add_argument("--frames", required=True)
    p.add_argument("--boxes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows")
    p.add_argument("--rec", default=str(D.ONNX_DIR / "VOCR6.2_L_fast.onnx"))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--pad", default="k4", help="the reading pad rule (PAD_RULES; k4 = eval/desktop.py's auto)")
    p = sub.add_parser("screen")
    p.add_argument("--res", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--out")
    p = sub.add_parser("pads")
    p.add_argument("--frames", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--rows")
    p.add_argument("--rec", default=str(D.ONNX_DIR / "VOCR6.2_L_fast.onnx"))
    p.add_argument("--workers", type=int, default=5)
    p = sub.add_parser("pages")
    p.add_argument("--hval", default=str(D.PV / "vocr-desktop" / "frames" / "hval.pkl"))
    p.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "desktop_data" / "realplus_native_v3.json"))
    p.add_argument("--shards", default=str(D.ART / "desktop-data" / "pages" / "realplus_v3"))
    p.add_argument("--plan", default=str(D.ART / "desktop-data" / "realplus" / "plan_v1.json"))
    p.add_argument("--skip-vns", default=str(D.PV / "textspot-desktop" / "frames" / "skip_vns.txt"))
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--n-pc", type=int, default=30)
    p.add_argument("--out", required=True)
    p = sub.add_parser("exact-boxes")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    {"edges": cmd_edges, "screen": cmd_screen, "pads": cmd_pads, "exact-boxes": cmd_exact_boxes,
     "pages": cmd_pages}[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

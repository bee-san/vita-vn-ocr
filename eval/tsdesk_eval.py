#!/usr/bin/env python3
"""textspot-desktop's evaluation (issue #173, eval/textspot_desktop_protocol.json), on vocr-desktop's harness
(eval/desktop.py, eval/desktop_time.py), which this file installs into at run time instead of editing it.

  tsdesk_eval.py desktop CMD ...  eval/desktop.py's command line (frames / detect / read / compare / boxes-pr / meiki)
                                 with this work's detectors installed: tsd:MODEL.onnx[@SIZE] (tools/tsdesk.Detector),
                                 ctsd:BLOB.vtsd[@SIZE] (the C path, when built), and the dense sets' ignore filter
                                 (eval/tsdesk_dense.install_ignore: a box at least half inside a 1b ignore area is
                                 dropped before reading)
  tsdesk.py edges --frames HVAL.pkl --boxes BOXES.jsonl --out E.json [--rec L_fast.onnx] [--workers 5]
      the protocol's edge error and tightness cost on hooked pages: every reference line of the units, its exact ink
      box (amendment 1: train/tsdesk_targets.ink_box on the line's glyph cells; the protocol's first reference,
      synth/textspot6_pages.refined_line, cut end glyphs); the detector's horizontal boxes as they are (no pad) matched
      one to one (vertical overlap >= half the smaller height, horizontal overlap >= half the smaller width, largest
      overlap first); per edge e = detector - reference (x0, y0, x1, y1) in frame px; then every matched line read twice
      by the recognizer (greedy), from its detector box and from its exact box, each grown by the reading pad (--pad;
      the protocol's k6v4): the CER of each over those lines and the paired title-cluster interval of the difference
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
# the reading pad of the tsd: / ctsd: timing pipelines when no --pad is given (eval/desktop.py's "auto" = k4); every
# protocol read and timing passes the protocol's k6v4 (eval/reproducibility/textspot-desktop-20261101/)
READ_PAD = os.environ.get("TSDESK_PAD", "auto")
SCHEMA = "textspot-desktop/1"


# ----------------------------------------------------------------------------- the harness hooks

class CTsdDet:
    """The C path's detector (desktop/, tools/tsdesk_c.py) on a VTSD blob."""

    name = "ctsd"

    def __init__(self, blob, size=None, threads=1, refine=None):
        import tsdesk_c as TC
        self.m, self.size, self.refine = TC.CModel(blob, threads=threads), size, refine

    def boxes(self, rgb):
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag] for b in
                self.m.detect(rgb, self.size, self.refine)]


def parse_kv(s):
    """{key: int} of "k=v,k=v" (the post-pass's and the reading filter's integer options)."""
    out = {}
    for part in filter(None, s.split(",")):
        k, eq, v = part.partition("=")
        if not eq:
            raise ValueError(f"{part!r}: key=value")
        out[k] = int(v)
    return out


def spec_parts(spec, with_decode=False):
    """(base, refine, post, drop[, decode]) of a detector spec: tsd:MODEL.onnx[@SIZE[@MODE]] or ctsd:BLOB.vtsd[@SIZE],
    then any of the suffixes +refine (tools/tsdesk_refine on the boxes), +post:K=V,... (tools/tsdesk.POST_DEFAULTS
    options), +drop:K=V,... (tools/tsdesk.DROP_DEFAULTS: the reading filter, applied when the boxes are read) and
    +dec:K=V,... (tools/tsdesk.DEFAULTS: the decode's options, numbers)."""
    base, *toks = str(spec).split("+")
    refine, post, drop, dec = False, {}, {}, {}
    for t in toks:
        if t == "refine":
            refine = True
        elif t.startswith("post:"):
            post.update(parse_kv(t[5:]))
        elif t.startswith("drop:"):
            drop.update(parse_kv(t[5:]))
        elif t.startswith("dec:"):
            for part in filter(None, t[4:].split(",")):
                k, _, v = part.partition("=")
                dec[k] = float(v)
        else:
            raise ValueError(f"{spec}: unknown suffix +{t}")
    return (base, refine, post, drop, dec) if with_decode else (base, refine, post, drop)


def detector_from_spec(spec, threads=1, so=None):
    import tsdesk as T
    base, refine, post, drop, dec = spec_parts(spec, with_decode=True)
    if drop:
        T.drop_options(drop)                    # checked here; applied when the boxes are read (install's read job)
    bad = set(dec) - set(T.DEFAULTS) - set(T.DECODE_EXTRA)
    if bad:
        raise ValueError(f"{spec}: unknown decode options {sorted(bad)}")
    dec = {k: int(v) if k in ("min_cells", "max_boxes") or k in T.DECODE_EXTRA else v for k, v in dec.items()}
    if base.startswith("tsd:"):
        path, _, rest = base[4:].partition("@")
        size, _, mode = rest.partition("@")
        return T.Detector(path, session_options=so, size=size or None, resize_mode=mode or "exact",
                          refine=True if refine else None, post=post or None, decode_opts=dec or None)
    if base.startswith("ctsd:"):
        if post or dec:
            raise ValueError(f"{spec}: the post-pass and decode options have no C path here")
        path, _, size = base[5:].partition("@")
        return CTsdDet(path, size or None, threads, refine=True if refine else None)
    if refine or post or drop:
        raise ValueError(f"{spec}: the suffixes are for tsd: and ctsd: specs")
    raise ValueError(spec)


_DROP = {"opts": None, "base": None}


def drop_of_det(det):
    """The reading filter's options (tools/tsdesk.drop_options) of a boxes file's detector spec, or None."""
    import tsdesk as T
    drop = {}
    for t in str(det).split("+")[1:]:
        if t.startswith("drop:"):
            drop.update(parse_kv(t[5:]))
    return T.drop_options(drop) if drop else None


def strip_drop(spec):
    """A detector spec without its +drop: suffixes (the detectors never see the reading filter)."""
    base, *toks = str(spec).split("+")
    return "+".join([base] + [t for t in toks if not t.startswith("drop:")])


def _drop_read_job(job):
    """eval/desktop.py's _read_job with the reading filter of the boxes' detector spec (+drop:...): the horizontal
    boxes whose readings tools/tsdesk.drop_set rejects are not scored, neither their text nor their boxes. The filter
    sees the readings of every box read (with nojp 2 also the horizontal boxes outside the units, as a page reads them
    all). With no filter it is _read_job itself."""
    o = _DROP["opts"]
    if not o:
        return _DROP["base"](job)
    import tsdesk as T
    fr, boxes, pad = job
    V = D._W["V"]
    rgb = D.load_rgb(fr)
    g = V.grey_of(rgb)
    H, W = g.shape
    p = D.pad_of(pad, H) if pad is not None else 0
    want = set()
    for u in fr["units"]:
        x0, y0, x1, y1 = u["region"]
        for k, b in enumerate(boxes):
            cb = D.crop_box(b, p, W, H)
            if cb and x0 <= cb[0] + cb[2] / 2.0 < x1 and y0 <= cb[1] + cb[3] / 2.0 < y1:
                want.add(k)
    read = {}
    for k, b in enumerate(boxes):
        if b[4] or not (k in want or o["nojp"] >= 2):
            continue
        cb = D.crop_box(b, p, W, H)
        if not cb:
            continue
        dark = V.line_dark(g, cb)
        ln = D._W["rec"].read(g[cb[1]:cb[1] + cb[3], cb[0]:cb[0] + cb[2]], invert=bool(dark))
        read[k] = (ln.text, ln.conf_mean)
    ks = sorted(read)
    items = [((boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]), *read[k]) for k in ks]
    gone = {ks[i] for i in T.drop_set(items, o)}
    crops, texts = [], []
    for k, b in enumerate(boxes):
        if k not in want or k in gone:
            continue
        cb = D.crop_box(b, p, W, H)
        crops.append((*cb, len(texts)))
        texts.append("" if b[4] else read[k][0])
    return D.score_frame_units(fr, g, crops, texts)


def install(Dm=D):
    """Register the tsd: / ctsd: detector specs in eval/desktop.py's detector() (which desktop.py detect and
    desktop_time.py's pipelines call), give them R's reading pad in desktop_time.py's pipelines, install the reading
    filter of a boxes file's +drop: suffix in desktop.py read, and install the dense sets' ignore filter. desktop.py
    itself is not changed."""
    if getattr(Dm, "_tsdesk_installed", False):
        return
    import tsdesk_dense
    base = Dm.detector

    def detector(spec, threads=1, opt="all", spinning=None):
        if spec.startswith(("tsd:", "ctsd:")):
            return detector_from_spec(spec, threads, Dm.session_options(threads, opt, spinning))
        return base(strip_drop(spec), threads, opt, spinning)

    Dm.detector, Dm._tsdesk_installed = detector, True
    base_load = Dm.load_boxes

    def load_boxes(path):
        boxes, meta = base_load(path)
        _DROP["opts"] = drop_of_det(meta.get("det", ""))
        return boxes, meta

    _DROP["base"] = Dm._read_job
    Dm.load_boxes, Dm._read_job = load_boxes, _drop_read_job
    tsdesk_dense.install_ignore(Dm)
    if os.environ.get("MIDA_MODEL"):          # MeikiOCR's author's detector, a report-only arm (eval/mida_det.py)
        import mida_det
        mida_det.install(Dm, os.environ["MIDA_MODEL"])
    base_pad, base_crop = Dm.pad_of, Dm.crop_box

    def pad_of(spec, H):
        """eval/desktop.py's pad specs plus the protocol's rules: 'kN' = max(2, round(N H / 544)) px on every side,
        'kAvB' = A horizontally and B vertically (a tuple); 'hXvY' = X and Y times each box's own size across it
        (protocol v2 amendment 1)."""
        s = str(spec)
        rel = rel_rule(s)
        if rel:
            return rel
        fl = floor_rule(s)
        if fl:
            return floor_hv(fl, H)
        if s.startswith("k"):
            kh, _, kv = s[1:].partition("v")
            ph, pv = pad_px(float(kh), H), pad_px(float(kv or kh), H)
            return ph if ph == pv else (ph, pv)
        return base_pad(spec, H)

    def crop_box(b, pad, W, H):
        if isinstance(pad, tuple) and pad and pad[0] == "clip":
            ph, pv = rel_hv(pad, b)
            return crop_sides(b, clip_sides(b, _CLIP["boxes"], ph, pv), W, H)
        if isinstance(pad, tuple) and pad and pad[0] == "rel":
            return crop_hv(b, rel_hv(pad, b), W, H)
        if isinstance(pad, tuple):
            return crop_hv(b, pad, W, H)
        return base_crop(b, pad, W, H)

    Dm.pad_of, Dm.crop_box = pad_of, crop_box
    _CLIP["base_read"] = Dm._read_job
    Dm._read_job = _clip_read_job
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


def margin_hv(rule, b, H):
    """(horizontal, vertical) frame px of a reading rule for the box b = [x0, y0, x1, y1] in an H px tall frame:
    a PAD_RULES name (the frame's height: max(2, round(k H / 544))), or "hXvY" (protocol v2 amendment 1: X and Y
    times the box's height, at least 1 px)."""
    if rule in PAD_RULES:
        kh, kv = PAD_RULES[rule]
        return pad_px(kh, H), pad_px(kv, H)
    fl = floor_rule(rule)
    if fl:
        return floor_hv(fl, H)
    rel = rel_rule(rule)
    if rel:
        return rel_hv(rel, b)
    raise ValueError(f"reading rule {rule!r}")


def floor_rule(spec):
    """(A, B, C) of a frame rule with a horizontal floor "kAvBfC" (protocol v2 amendment 2), else None."""
    s = str(spec)
    if s.startswith("k") and "v" in s and "f" in s:
        try:
            a, _, rest = s[1:].partition("v")
            b, _, c = rest.partition("f")
            return (float(a), float(b), float(c))
        except ValueError:
            return None
    return None


def floor_hv(fl, H):
    """(horizontal, vertical) px of "kAvBfC" in an H px tall frame: max(C, round(A H / 544)) and max(2, round(B H /
    544))."""
    a, b, c = fl
    return max(c, float(int(round(a * H / 544)))), float(pad_px(b, H))


def rel_rule(spec):
    """("rel", X, Y) of a line-height reading rule "hXvY", ("clip", X, Y) of "cXvY" (the same margins, each side
    clipped at half the gap to the nearest other box of the frame on that side; protocol v2 amendment 3), else None."""
    s = str(spec)
    if s[:1] in ("h", "c") and "v" in s:
        x, _, y = s[1:].partition("v")
        try:
            return ("rel" if s[0] == "h" else "clip", float(x), float(y))
        except ValueError:
            return None
    return None


_CLIP = {"boxes": (), "base_read": None}


def _clip_read_job(job):
    """eval/desktop.py's read job with the frame's boxes kept for the clip rules (module level: the pool pickles it)."""
    _CLIP["boxes"] = tuple(job[1])
    return _CLIP["base_read"](job)


def clip_sides(b, others, ph, pv):
    """(left, top, right, bottom) margins of b = [x0, y0, x1, y1]: ph / pv, each cut to half the gap to the nearest
    other box on that side (a box in the same row, overlapping b across by at least half the smaller height, for the
    left and right; a box overlapping b along it, for the top and bottom), at least 1 px or the gap's half if
    smaller."""
    x0, y0, x1, y1 = (float(v) for v in b[:4])
    left, right, top, bottom = ph, ph, pv, pv
    for o in others:
        ox0, oy0, ox1, oy1 = (float(v) for v in o[:4])
        if (ox0, oy0, ox1, oy1) == (x0, y0, x1, y1):
            continue
        across = min(y1, oy1) - max(y0, oy0)
        along = min(x1, ox1) - max(x0, ox0)
        if across >= 0.5 * min(y1 - y0, oy1 - oy0) and along <= 0:
            if ox0 >= x1:
                right = min(right, (ox0 - x1) / 2.0)
            elif ox1 <= x0:
                left = min(left, (x0 - ox1) / 2.0)
        elif along > 0 and across <= 0:
            if oy0 >= y1:
                bottom = min(bottom, (oy0 - y1) / 2.0)
            elif oy1 <= y0:
                top = min(top, (y0 - oy1) / 2.0)
    return left, top, right, bottom


def crop_sides(b, sides, W, H):
    import math
    left, top, right, bottom = sides
    x0, y0 = max(0, int(math.floor(b[0] - left))), max(0, int(math.floor(b[1] - top)))
    x1, y1 = min(W, int(math.ceil(b[2] + right))), min(H, int(math.ceil(b[3] + bottom)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def rel_hv(rel, b):
    """(horizontal, vertical) px of ("rel", X, Y) for b = [x0, y0, x1, y1]: X times the line's size across it along
    the line and Y times it across (a horizontal line's height; a vertical line's width), at least 1 px."""
    _, x, y = rel
    w, h = max(1.0, float(b[2]) - float(b[0])), max(1.0, float(b[3]) - float(b[1]))
    if w >= h:
        return max(1.0, x * h), max(1.0, y * h)
    return max(1.0, y * w), max(1.0, x * w)


def exact_boxes(fr, rgb):
    """[(unit index, line index, [x0, y0, x1, y1], how)] of a frame's reference lines (exact_ref)."""
    out = []
    for ui, u in enumerate(fr["units"]):
        for li, ln in enumerate(u["lines"]):
            box, how = exact_ref(fr, rgb, ln)
            out.append((ui, li, [float(v) for v in box], how))
    return out


def _pad_job(job):
    import math

    import metrics
    import pagepath as PP
    fr, rules = job if isinstance(job, tuple) else (job, list(PAD_RULES))
    V = _E["V"]
    rgb = D.load_rgb(fr)
    g = V.grey_of(rgb)
    H, W = g.shape
    rows = []
    ex = exact_boxes(fr, rgb)
    others = [e[2] for e in ex]
    for ui, li, b, how in ex:
        u = fr["units"][ui]
        ref = metrics.norm(u["lines"][li]["text"])
        r = {"uid": f"{u['uid']}|{li}", "cluster": u["cluster"], "view": fr["view"], "chars": len(ref), "how": how}
        for name in rules:
            rel = rel_rule(name)
            if rel and rel[0] == "clip":
                cb = crop_sides(b, clip_sides(b, others, *rel_hv(rel, b)), W, H)
            else:
                ph, pv = margin_hv(name, b, H)
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
    rules = a.rules.split(",") if a.rules else list(PAD_RULES)
    if "k4" not in rules:
        rules = ["k4"] + rules
    out = {"schema": SCHEMA, "kind": "textspot-desktop-pads", "rules": rules, "rec": Path(a.rec).name,
           "rec_sha256": D.sha256_file(a.rec), "sets": {},
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for fpath in a.frames:
        frames, _ = D.load_frames(fpath)
        rows = []
        with D._pool(a.workers, _init_edges, (a.rec,)) as pool:
            for rs in pool.imap(_pad_job, [(f, rules) for f in frames], chunksize=4):
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
            st[sub] = {name: round(100.0 * sum(r[name] for r in rr) / max(1, c), 4) for name in rules}
        base = [{"uid": r["uid"], "cluster": r["cluster"], "errors": r["k4"], "chars": r["chars"]} for r in rows]
        st["vs_k4"] = {name: PP.cluster_bootstrap([{"uid": r["uid"], "cluster": r["cluster"], "errors": r[name],
                                                    "chars": r["chars"]} for r in rows], base)
                       for name in rules if name != "k4"}
        by = {}
        for r in rows:
            b = by.setdefault(r["cluster"], {n: 0 for n in rules})
            for n in rules:
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


# ----------------------------------------------------------------------------- the PC test set, once

PCTEST_ATTEMPT = D.PV / "textspot-desktop" / "pctest" / "ATTEMPT.json"
DECISION = ROOT / "eval" / "textspot_desktop_decision.json"


def committed_unchanged(path):
    import subprocess
    rel = Path(path).resolve().relative_to(ROOT)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=ROOT, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(rel)], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return not tracked.returncode and not dirty


def cmd_pctest_gate(a):
    """The PC test set is read once (protocol rule.pctest_guard): only when the protocol and the decision record are
    committed unchanged, and only if this protocol's attempt record does not exist yet (this call writes it)."""
    for p in (PROTOCOL, DECISION):
        if not p.exists() or not committed_unchanged(p):
            raise SystemExit(f"{p.relative_to(ROOT)}: not committed unchanged")
    if PCTEST_ATTEMPT.exists():
        raise SystemExit(f"the PC test set was read already under this protocol ({PCTEST_ATTEMPT})")
    PCTEST_ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "head": D.git_head(),
           "protocol_sha256": D.sha256_file(PROTOCOL), "decision_sha256": D.sha256_file(DECISION),
           "what": a.what}
    PCTEST_ATTEMPT.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    print(json.dumps(rec), flush=True)


def errata_boxes(path):
    """{(unit uid, box_v1 tuple): box_fixed} of the PC test set's erratum (eval/desktop_pc_test_v1_errata.json)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(ln["unit"], tuple(float(v) for v in ln["box_v1"])): [float(v) for v in ln["box_fixed"]]
            for ln in doc["lines"]}


def cmd_exact_pctest(a):
    """The exact-boxes row of the PC test set: its line boxes with the erratum's 7 corrected boxes."""
    fix = errata_boxes(a.errata)
    frames, _ = D.load_frames(a.frames)
    rows, n_fixed = [], 0
    for fr in frames:
        out = []
        for u in fr["units"]:
            for ln in u["lines"]:
                b = [float(v) for v in ln["box"]]
                k = (u["uid"], tuple(b))
                if k in fix:
                    b = fix[k]
                    n_fixed += 1
                out.append([b[0], b[1], b[0] + b[2], b[1] + b[3], 0, 9.0, 0])
        rows.append({"key": fr["key"], "W": fr["W"], "H": fr["H"], "boxes": out, "ms": 0.0})
    meta = {"meta": {"det": "exact", "frames_file": str(a.frames), "frames_sha256": D.sha256_file(a.frames),
                     "rule": "the PC test set's line boxes, the erratum's corrected boxes", "fixed": n_fixed,
                     "repo_head": D.git_head()}}
    D.write_jsonl(a.out, [meta, *rows])
    print(f"exact pctest boxes: {len(rows)} frames, {n_fixed} boxes from the erratum -> {a.out}", flush=True)


# ----------------------------------------------------------------------------- R6: C path against ONNX path

def cmd_parity(a):
    """R6's crop equality: two boxes files of the same frames (the C path's ctsd: and the ONNX path's tsd:); per frame,
    the sorted crops (each horizontal box grown by the pad rule, rounded outward, clipped) are all equal or not."""
    fa, ma = D.load_boxes(a.a)
    fb, mb = D.load_boxes(a.b)
    if ma["frames_sha256"] != mb["frames_sha256"]:
        raise SystemExit("the two boxes files were found on different frames files")
    frames, _ = D.load_frames(a.frames)
    if D.sha256_file(a.frames) != ma["frames_sha256"]:
        raise SystemExit("the boxes were found on another frames file")
    kh, kv = PAD_RULES[a.pad]
    eq = n = 0
    worst = 0.0
    diff_frames = []
    for fr in frames:
        W, H = fr["W"], fr["H"]
        p = (pad_px(kh, H), pad_px(kv, H))

        def crops(bs):
            return sorted(c for c in (crop_hv([float(v) for v in b[:4]], p, W, H) for b in bs if not b[4]) if c)
        A, B = fa[fr["key"]], fb[fr["key"]]
        n += 1
        if crops(A) == crops(B):
            eq += 1
        else:
            diff_frames.append(fr["key"])
        if len(A) == len(B):
            sa = sorted([[float(v) for v in b[:4]] for b in A])
            sb = sorted([[float(v) for v in b[:4]] for b in B])
            for x, y in zip(sa, sb):
                worst = max(worst, max(abs(u - v) for u, v in zip(x, y)))
    doc = {"kind": "textspot-desktop-parity", "a": str(a.a), "a_det": ma["det"], "b": str(a.b), "b_det": mb["det"],
           "frames_sha256": ma["frames_sha256"], "pad": a.pad, "frames": n, "crops_equal_frames": eq,
           "crops_equal_share": round(eq / max(1, n), 5), "max_abs_box_diff_px": round(worst, 6),
           "frames_differing": len(diff_frames)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(doc), flush=True)


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
    p.add_argument("--rules", help="comma-separated reading rules (PAD_RULES names and hXvY); default PAD_RULES")
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
    p = sub.add_parser("parity")
    p.add_argument("--frames", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--pad", default="k6v4")
    p.add_argument("--out", required=True)
    p = sub.add_parser("pctest-gate")
    p.add_argument("--what", required=True)
    p = sub.add_parser("exact-pctest")
    p.add_argument("--frames", required=True)
    p.add_argument("--errata", default=str(ROOT / "eval" / "desktop_pc_test_v1_errata.json"))
    p.add_argument("--out", required=True)
    p = sub.add_parser("exact-boxes")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    {"edges": cmd_edges, "screen": cmd_screen, "pads": cmd_pads, "exact-boxes": cmd_exact_boxes,
     "pages": cmd_pages, "pctest-gate": cmd_pctest_gate, "exact-pctest": cmd_exact_pctest,
     "parity": cmd_parity}[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

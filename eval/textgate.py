#!/usr/bin/env python3
"""textgate: the text-presence gate in front of GRAD + refine (docs/textgate.md, eval/textgate_protocol.json).

A gate answers one question per region: does it hold text? When it says no, the finder returns no line (the
recognizer and its weights are never touched); when it says yes, GRAD + refine run unchanged. A learned gate is
an int8 network in #97's VTDT layout (synth/textdet_twin.py: run by runtime/vocr_textgate.c, and by
runtime/vocr_textdet.c's vocr_textdet_maps) whose head's first map K is a text logit per output cell:

    region mode   score = the k-th largest K code over the region's map (the smallest when it has fewer cells)
    boxes mode    score = the k-th largest K code over the cells whose block meets a GRAD box (-129 when none)
    text          iff score >= t

(k, t) are fixed per candidate on the calibration cohorts (training-side VNs only, eval/textgate_data.py): t is the
largest code keeping at least the target share of cal_dlg cases, k in {1, 2, 4} the one that then rejects the most
cal_empty regions (the smaller k on a tie). Non-learned baselines: `blank` (Rubi's blank-line check,
core/vocr_ocr.c vjo_vocr_ink: a line whose crop has fewer than 50 edges per 10,000 pixels is blank; the region
holds text iff a GRAD + refine line is not blank) and `rule8` (the longest horizontal run of 8 x 8 blocks with at
least 16 edge pixels and a contrast of at least 48; text iff the run reaches R blocks, R calibrated as t is).

  textgate.py calibrate --cand NAME=BLOB ... --cohorts DIR --out JSON [--cbin DRIVER]
  textgate.py dev --calibration JSON --out DIR [--cgate DRIVER]          development (after the protocol is pushed)
  textgate.py select --protocol JSON --dev DIR --sizes JSON --out JSON    the pre-registered rule
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
K_SET = (1, 2, 4)
TARGETS = (0.998, 0.999)
MODES = ("region", "boxes")
NO_CELLS = -129
SCREEN_EMPTY = 0.85
MODELS = ("VOCR5-S", "VOCR5-N", "VOCR5-L")
WD_COHORTS = ("tv_dlg", "dn_dlg")
BLANK_EDGES, EDGE_MIN, EDGE_DIV, INK_SPAN = 50, 24, 4, 2       # Rubi core/vocr_ocr.h
RULE8 = {"block": 8, "edges": 16, "contrast": 48, "tau": 24}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(spec):
    name, _, path = spec.partition("=")
    if not name:
        raise SystemExit(f"NAME=PATH, not {spec!r}")
    return name, path


# ---------------------------------------------------------------- the learned gate (the spec)

def kmap(model, grey):
    """The K map (int64 codes) of a VTDT model on a grey region: synth/textdet_twin.run_net's first map."""
    import textdet_twin as TW
    return TW.run_net(model, np.ascontiguousarray(grey, dtype=np.uint8))[0]


def kth(values, k):
    """The k-th largest of a 1-D integer array (the smallest when it holds fewer than k), NO_CELLS when empty."""
    v = np.asarray(values).ravel()
    if not v.size:
        return NO_CELLS
    k = min(k, v.size)
    return int(np.partition(v, v.size - k)[v.size - k])


def box_mask(shape, stride, boxes):
    """Cells (i, j) whose block [s j, s j + s) x [s i, s i + s) meets any box (x, y, w, h)."""
    hm, wm = shape
    m = np.zeros((hm, wm), bool)
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        m[y // stride:min(hm, (y + h - 1) // stride + 1), x // stride:min(wm, (x + w - 1) // stride + 1)] = True
    return m


def score(km, mode, k, stride, grad_boxes=None):
    if mode == "region":
        return kth(km, k)
    return kth(km[box_mask(km.shape, stride, grad_boxes or [])], k)


def c_kmaps(cbin, blob, images):
    """K maps through runtime/vocr_textdet_test --maps (runtime/vocr_textdet.c, vocr_textdet_maps)."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img in images:
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape
            f.write(struct.pack("<4I", w, h, 1, 0))
            f.write(img.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(cbin), "--maps", str(blob), path], check=True, capture_output=True,
                             text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    res = []
    for a, b in zip(out[0::2], out[1::2]):
        rc, hm, wm = (int(t) for t in b.split())
        if rc:
            raise RuntimeError(f"vocr_textdet_maps rc={rc}")
        v = np.array([int(t) for t in a[1:].split()], np.int64).reshape(hm, wm, 4)
        res.append(v[..., 0])
    return res


# ---------------------------------------------------------------- non-learned baselines

def rubi_ink_edges(crop):
    """Rubi's vjo_vocr_ink (core/vocr_ocr.c): edges per 10,000 pixels of a grey crop (integer arithmetic)."""
    g = np.asarray(crop, np.int64)
    h, w = g.shape
    n = h * w
    hist = np.bincount(g.ravel(), minlength=256)

    def pct(part):
        want = max(1, (n * part + 999) // 1000)
        acc = np.cumsum(hist[:255])
        idx = np.flatnonzero(acc >= want)
        return int(idx[0]) if len(idx) else 255
    lo, hi = pct(20), pct(980)
    e = max(EDGE_MIN, (hi - lo) // EDGE_DIV)
    across = np.zeros((h, w), bool)
    down = np.zeros((h, w), bool)
    for s in range(1, INK_SPAN + 1):
        if s < w:
            across[:, :w - s] |= np.abs(g[:, :w - s] - g[:, s:]) >= e
        if s < h:
            down[:h - s, :] |= np.abs(g[:h - s, :] - g[s:, :]) >= e
    edges = int(across.sum() + down.sum())
    q, r = divmod(edges, n)
    a = r * 100 // n
    r = r * 100 % n
    return q * 10000 + a * 100 + r * 100 // n


def blank_text(grey, refined_boxes):
    """`blank`: the region holds text iff some GRAD + refine line's crop is not blank."""
    return any(rubi_ink_edges(grey[y:y + h, x:x + w]) >= BLANK_EDGES for x, y, w, h in refined_boxes)


def rule8_score(grey):
    """`rule8`: the longest horizontal run of 8 x 8 blocks with >= 16 edge pixels (|dx| or |dy| >= 24) and a
    contrast (max - min) >= 48."""
    g = np.asarray(grey, np.int32)
    H, W = g.shape
    s = RULE8["block"]
    dx = np.zeros_like(g)
    dx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
    dy = np.zeros_like(g)
    dy[1:] = np.abs(g[1:] - g[:-1])
    bh, bw = -(-H // s), -(-W // s)

    def pad(a, v):
        return np.pad(a, ((0, bh * s - H), (0, bw * s - W)), constant_values=v).reshape(bh, s, bw, s)
    e = pad((np.maximum(dx, dy) >= RULE8["tau"]).astype(np.int32), 0).sum((1, 3))
    c = pad(g, 0).max((1, 3)) - pad(g, 255).min((1, 3))
    ok = (e >= RULE8["edges"]) & (c >= RULE8["contrast"])
    best = 0
    for row in ok:
        run = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
    return best


# ---------------------------------------------------------------- the finder's boxes (C, twin-checked)

def finder_driver(out):
    import finder_tighten as FT
    return FT.build(Path(out) / "build")


def grad_refine(driver, images):
    """[(GRAD boxes, refined boxes)] per grey image (the C finder and refinement, checked against the twin)."""
    import finder_tighten as FT
    g = FT.lines_of(driver, images, 0)
    r = FT.lines_of(driver, images, 0x1F)
    return [(gg[0], rr[0]) for gg, rr in zip(g, r)]


# ---------------------------------------------------------------- calibration (training-side)

def threshold_at(scores, target):
    """The largest t with mean(scores >= t) >= target: the (q+1)-th smallest score, q = floor((1 - target) n)."""
    s = np.sort(np.asarray(scores))
    q = int(np.floor((1 - target) * len(s) + 1e-9))
    return int(s[q])


def calibrate_scores(dlg_scores, emp_scores, dlg_found, emp_found, kinds, sources):
    """{target: {k*, t, recall (pooled / per kind / per source), empty_rate (gate or no finder line)}} from
    {k: scores} of cal_dlg and cal_empty. Recall counts the cal_dlg cases GRAD + refine returns a line for (the gate
    cannot lose the others)."""
    out = {}
    live = np.asarray(dlg_found) > 0
    kinds, sources = np.asarray(kinds)[live], np.asarray(sources)[live]
    for target in TARGETS:
        best = None
        for k in sorted(dlg_scores):
            d = np.asarray(dlg_scores[k])[live]
            t = threshold_at(d, target)
            e = np.asarray(emp_scores[k])
            empty = (e < t) | (np.asarray(emp_found) == 0)
            rate = float(empty.mean())
            if best is None or rate > best["empty_rate"]:
                best = {"k": int(k), "t": int(t), "empty_rate": round(rate, 4),
                        "gate_rejects": round(float((e < t).mean()), 4),
                        "recall": round(float((d >= t).mean()), 5), "cases": int(live.sum()),
                        "recall_by_kind": {kd: round(float((d[kinds == kd] >= t).mean()), 5)
                                           for kd in sorted(set(kinds))},
                        "recall_by_source": {sc: round(float((d[sources == sc] >= t).mean()), 5)
                                             for sc in sorted(set(sources))},
                        "dropped": int((d < t).sum())}
        best["screen_pass"] = best["empty_rate"] >= SCREEN_EMPTY
        out[str(target)] = best
    return out


def cmd_calibrate(args):
    import textdet_twin as TW
    coh = Path(args.cohorts)
    dlg = pickle.load(open(coh / "cal_dlg.pkl", "rb"))
    emp = pickle.load(open(coh / "cal_empty.pkl", "rb"))
    kinds = [c["kind"] for c in dlg]
    sources = [c["source"] for c in dlg]
    ident = finder_driver(Path(args.out).parent / f"cal-build-{os.getpid()}")
    fd = grad_refine(ident["driver"], [c["image"] for c in dlg])
    fe = grad_refine(ident["driver"], [c["image"] for c in emp])
    emp_found = [len(r) for _, r in fe]
    dlg_found = [len(r) for _, r in fd]
    doc = {"kind": "textgate-calibration", "computed": utc_now(), "cohorts": json.loads((coh / "cohorts.json")
                                                                                      .read_text()),
           "targets": TARGETS, "k_set": K_SET, "screen_empty": SCREEN_EMPTY, "finder": ident,
           "cal_empty_finder_none": int(sum(n == 0 for n in emp_found)),
           "cal_dlg_finder_none": int(sum(n == 0 for n in dlg_found)), "candidates": {}}
    if Path(args.out).exists():
        doc["candidates"] = json.loads(Path(args.out).read_text())["candidates"]
    for spec in args.cand:
        name, blob = parse(spec)
        if name in doc["candidates"] and not args.force:
            continue
        entry = {}
        if name == "blank":
            d = {1: [int(blank_text(c["image"], r)) for c, (_, r) in zip(dlg, fd)]}
            e = {1: [int(blank_text(c["image"], r)) for c, (_, r) in zip(emp, fe)]}
            entry = {"family": "rule", "bytes": 0, "modes": {"region": calibrate_scores(d, e, dlg_found, emp_found,
                                                                                       kinds, sources)}}
        elif name == "rule8":
            d = {1: [rule8_score(c["image"]) for c in dlg]}
            e = {1: [rule8_score(c["image"]) for c in emp]}
            entry = {"family": "rule", "bytes": 0, "modes": {"region": calibrate_scores(d, e, dlg_found, emp_found,
                                                                                       kinds, sources)}}
        else:
            model = TW.Model(Path(blob).read_bytes())
            kd = c_kmaps(args.cbin, blob, [c["image"] for c in dlg])
            ke = c_kmaps(args.cbin, blob, [c["image"] for c in emp])
            rng = np.random.default_rng(0)
            for i in rng.choice(len(dlg), min(40, len(dlg)), replace=False):     # the C maps equal the spec's
                if not np.array_equal(kd[i], kmap(model, dlg[i]["image"])):
                    raise RuntimeError(f"{name}: C K map differs from the twin on cal_dlg case {i}")
            entry = {"family": "learned", "blob": str(blob), "blob_sha256": sha256(blob), "bytes": model.blob_bytes,
                     "shape": model.name, "stride": model.stride, "modes": {}}
            for mode in MODES:
                d = {k: [score(m, mode, k, model.stride, g) for m, (g, _) in zip(kd, fd)] for k in K_SET}
                e = {k: [score(m, mode, k, model.stride, g) for m, (g, _) in zip(ke, fe)] for k in K_SET}
                entry["modes"][mode] = calibrate_scores(d, e, dlg_found, emp_found, kinds, sources)
        doc["candidates"][name] = entry
        for mode, ops in entry["modes"].items():
            print(name, mode, " ".join(f"{t}: k{o['k']} t{o['t']} rec {o['recall']} empty {o['empty_rate']}"
                                       f"{' PASS' if o['screen_pass'] else ''}" for t, o in ops.items()), flush=True)
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- development

def linefind_empties():
    """(development empties, report empties): eval/linefind's `empty` regions of vita_dev + psp_dev and of the two
    holdouts, a reserved test VN's regions left out (textdet.linefind_rows), with their grey regions."""
    import textdet as TD
    shots = [str(d) for d in TD._linefind("build_gt").SHOT_DIRS]
    test = TD.reserved_test_vns()
    out = {}
    for role, sets in (("development", ("vita_dev", "psp_dev")), ("report", ("vita_holdout", "psp_holdout"))):
        rows = []
        for s in sets:
            keep, _ = TD.linefind_rows(s, test)
            rows += [dict(r, linefind_set=s) for r in keep if "empty" in r.get("tags", [])]
        out[role] = [(r, TD.region_grey(r, shots)) for r in rows]
    return out


def scene_pool_ids():
    """Screenshot ids and names in synth3's default scene pool (#97's synthetic canvases drew from it)."""
    import re

    import synth3 as S3
    R = S3.init()
    names = {Path(p).name for paths in R.bg.values() for p in paths}
    ids = {m.group(2) for n in names for m in [re.match(r"(v\d+)_(\d+)", n)] if m}
    return ids, names


def cmd_dev(args):
    """Every screened candidate on the development data: per wd case and per empty region, its score and decision
    (no recognition: a region the gate passes reads as GRAD + refine read it, one it stops reads nothing)."""
    import textdet as TD

    import textdet_twin as TW
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [(n, c) for n in WD_COHORTS for c in TD.load_cohort(args.cohorts, n)]
    empties = linefind_empties()
    ident = finder_driver(out)
    fw = grad_refine(ident["driver"], [c["image"] for _, c in cases])
    fe = {role: grad_refine(ident["driver"], [g for _, g in rows]) for role, rows in empties.items()}
    ids, names = scene_pool_ids()

    def in_pool(r):
        sid = r["url"].rsplit("/", 1)[-1].split(".")[0]
        return sid in ids or r["shot"] in names
    doc = {"kind": "textgate-dev-decisions", "computed": utc_now(), "calibration_sha256": sha256(args.calibration),
           "finder": ident, "wd_cases": [TD.case_key(n, c) for n, c in cases],
           "empties": {role: [{"id": r["id"], "vn": r["vn"], "set": r["linefind_set"], "in_scene_pool": in_pool(r),
                               "finder_lines": len(f[1])} for (r, _), f in zip(rows, fe[role])]
                       for role, rows in empties.items()},
           "candidates": {}}
    for name, entry in cal["candidates"].items():
        for mode, ops in entry["modes"].items():
            for target, op in ops.items():
                if not op["screen_pass"] and not args.all:
                    continue
                key = f"{name}/{mode}/{target}"
                if entry["family"] == "learned":
                    model = TW.Model(Path(entry["blob"]).read_bytes())
                    kms = [kmap(model, c["image"]) for _, c in cases]
                    sw = [score(m, mode, op["k"], model.stride, g) for m, (g, _) in zip(kms, fw)]
                    se = {role: [score(kmap(model, g), mode, op["k"], model.stride, f[0]) for (_, g), f
                                 in zip(rows, fe[role])] for role, rows in empties.items()}
                    keep_w = [s >= op["t"] for s in sw]
                    keep_e = {role: [s >= op["t"] for s in v] for role, v in se.items()}
                elif name == "blank":
                    sw = [int(blank_text(c["image"], r)) for (_, c), (_, r) in zip(cases, fw)]
                    se = {role: [int(blank_text(g, f[1])) for (_, g), f in zip(rows, fe[role])]
                          for role, rows in empties.items()}
                    keep_w = [s >= op["t"] for s in sw]
                    keep_e = {role: [s >= op["t"] for s in v] for role, v in se.items()}
                else:
                    sw = [rule8_score(c["image"]) for _, c in cases]
                    se = {role: [rule8_score(g) for _, g in rows] for role, rows in empties.items()}
                    keep_w = [s >= op["t"] for s in sw]
                    keep_e = {role: [s >= op["t"] for s in v] for role, v in se.items()}
                doc["candidates"][key] = {"name": name, "mode": mode, "target": target, "k": op["k"], "t": op["t"],
                                          "family": entry["family"], "bytes": entry["bytes"],
                                          "blob_sha256": entry.get("blob_sha256"), "wd_scores": sw,
                                          "wd_keep": keep_w, "empty_scores": se, "empty_keep": keep_e}
                kept_e = sum(k and f["finder_lines"] > 0 for k, f in zip(keep_e["development"],
                                                                         doc["empties"]["development"]))
                print(f"{key}: wd dropped {len(keep_w) - sum(keep_w)} of {len(keep_w)}; development empties with a "
                      f"line {kept_e} of {len(keep_e['development'])}", flush=True)
    (out / "decisions.json").write_text(json.dumps(doc) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- the rule

def gated_units(base_units, keep):
    """A model's development units with the gate: the refine unit where the gate keeps the case, else every
    character missed. base_units: finder_tighten's units (cohort -> {"refine": [...]}); keep: case key -> bool."""
    out = {}
    for cohort in WD_COHORTS:
        units = []
        for u in base_units[cohort]["refine"]:
            k = keep[f"{cohort}|{u['unit']}"]
            units.append(dict(u) if k else dict(u, errors=u["chars"], hira_errors=u["hira_chars"],
                                                detections=0, gated=True))
        out[cohort] = {"gate": units, "refine": base_units[cohort]["refine"]}
    return out


def evaluate(cand, dec, base, native):
    import textdet as TD
    keep = dict(zip(dec["wd_cases"], cand["wd_keep"]))
    rep = {"models": {}}
    for m in MODELS:
        units = gated_units(base[m], keep)
        g, r = TD.wd_units(units, "gate"), TD.wd_units(units, "refine")
        rep["models"][m] = {
            "wd": {"gate": TD.cer_of(g), "refine": TD.cer_of(r)},
            "gate_minus_refine": TD.vn_diff([(g, r)]),
            "native_vita": {"gate": TD.cer_of(g, lambda u: u["shot"] in native),
                            "refine": TD.cer_of(r, lambda u: u["shot"] in native)},
            "native_gate_minus_refine": TD.vn_diff([(g, r)], lambda u: u["shot"] in native),
            "per_platform": {p: {"gate": TD.cer_of(g, lambda u, p=p: u["platform"] == p),
                                 "refine": TD.cer_of(r, lambda u, p=p: u["platform"] == p)} for p in ("vita", "psp")},
            "units_gated": sum(bool(u.get("gated")) for u in g)}
    rep["wd_cases_dropped"] = len(cand["wd_keep"]) - sum(cand["wd_keep"])
    for role in ("development", "report"):
        meta = dec["empties"][role]
        none = [not k or e["finder_lines"] == 0 for k, e in zip(cand["empty_keep"][role], meta)]
        rep[f"empty_{role}"] = {
            "regions": len(meta), "none_returned": sum(none), "rate": round(sum(none) / len(meta), 4) if meta else None,
            "in_scene_pool": [sum(n for n, e in zip(none, meta) if e["in_scene_pool"]),
                              sum(e["in_scene_pool"] for e in meta)],
            "not_in_scene_pool": [sum(n for n, e in zip(none, meta) if not e["in_scene_pool"]),
                                  sum(not e["in_scene_pool"] for e in meta)]}
    return rep


def cmd_select(args):
    import textdet as TD
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    rule = proto["selection_rule"]
    dec = json.loads((Path(args.dev) / "decisions.json").read_text(encoding="utf-8"))
    sizes = json.loads(Path(args.sizes).read_text(encoding="utf-8"))
    parity = json.loads(Path(args.parity).read_text(encoding="utf-8")) if args.parity else {}
    base = TD.load_baseline(args.baseline)
    native = TD.native_vita_shots()
    reps = {}
    for key, cand in dec["candidates"].items():
        rep = evaluate(cand, dec, base, native)
        code = sizes["code_bytes"][cand["mode"] if cand["family"] == "learned" else cand["name"]]
        rep.update(name=cand["name"], mode=cand["mode"], target=cand["target"], k=cand["k"], t=cand["t"],
                   family=cand["family"], weight_bytes=cand["bytes"], code_bytes=code,
                   total_bytes=cand["bytes"] + code, blob_sha256=cand.get("blob_sha256"))
        mm = rep["models"]
        rep["conditions"] = {
            "wd_not_resolved_worse": all(mm[m]["gate_minus_refine"]["ci95"] is None or
                                         mm[m]["gate_minus_refine"]["ci95"][0] <= 0 for m in MODELS),
            "wd_cost_le_cap": all(mm[m]["gate_minus_refine"]["delta"] <= rule["wd_cost_cap_points"] for m in MODELS),
            "native_vita_not_resolved_worse": all(mm[m]["native_gate_minus_refine"]["ci95"] is None or
                                                  mm[m]["native_gate_minus_refine"]["ci95"][0] <= 0 for m in MODELS),
            "empty_rate_ge": rep["empty_development"]["rate"] >= rule["empty_rate_min"]}
        rep["eligible"] = cand["family"] != "learned" or bool(parity.get(cand["name"], {}).get("c_equals_twin"))
        rep["qualifies"] = rep["eligible"] and all(rep["conditions"].values())
        reps[key] = rep
    tmode = {m: i for i, m in enumerate(MODES)}
    order = sorted(reps, key=lambda c: (reps[c]["total_bytes"], reps[c]["weight_bytes"], tmode[reps[c]["mode"]],
                                        -float(reps[c]["target"]), c))
    qual = [c for c in order if reps[c]["qualifies"]]
    winner = qual[0] if qual else None
    doc = {"kind": "textgate-selection", "computed": utc_now(), "protocol_sha256": sha256(args.protocol),
           "decisions_sha256": sha256(Path(args.dev) / "decisions.json"), "sizes": sizes, "order": order,
           "qualifying": qual, "winner": winner, "candidates": reps,
           "baseline": {m: TD.cer_of(TD.wd_units({c: {"refine": base[m][c]["refine"]} for c in WD_COHORTS},
                                                 "refine")) for m in MODELS}}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for c in order:
        r = reps[c]
        mm = r["models"]
        print(f"{c:34s} {r['total_bytes']:6d} B  dropped {r['wd_cases_dropped']:2d}  " +
              " ".join(f"{mm[m]['gate_minus_refine']['delta']:+.3f}{mm[m]['gate_minus_refine']['ci95']}"
                       for m in MODELS) +
              f"  empty {r['empty_development']['none_returned']}/{r['empty_development']['regions']}"
              f" (out of pool {r['empty_development']['not_in_scene_pool']})  qual {r['qualifies']}", flush=True)
    print("winner:", winner)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--cand", action="append", required=True)
    p.add_argument("--cohorts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cbin", default=str(ART / "textdet" / "cbuild" / "000bdca" / "vocr_textdet_test"))
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("dev")
    p.add_argument("--calibration", required=True)
    p.add_argument("--cohorts", default=str(ART / "vocr5-train" / "cohorts"))
    p.add_argument("--out", required=True)
    p.add_argument("--all", action="store_true", help="also the candidates the calibration screen stopped")
    p = sub.add_parser("select")
    p.add_argument("--protocol", default=str(ROOT / "eval" / "textgate_protocol.json"))
    p.add_argument("--dev", required=True)
    p.add_argument("--baseline", default=str(ART / "textdet" / "baseline" / "wd"))
    p.add_argument("--sizes", required=True)
    p.add_argument("--parity", default="")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"calibrate": cmd_calibrate, "dev": cmd_dev, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()

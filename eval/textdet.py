#!/usr/bin/env python3
"""textdet: the evaluation harness of the learned text detector (issue #96; docs/textdet.md;
eval/textdet_protocol.json).

  textdet.py linefind --finder NAME ... --out JSON [--rows JSON]
      Detection only, on eval/linefind's detector-built sets: vita_dev and psp_dev (development: the protocol's
      empty-region gate reads these) and vita_holdout and psp_holdout (report only). Every region of a VN that
      eval/manifests/reserved.json reserves as a test VN is left out (vita_dev v36, psp_dev v1267 and v4060 when
      written); research_a/b (the GRAD finder's tuning sets, which hold vita-test screenshots) are never read.
      Per set and per region tag (tight / loose / empty): line recall / precision / F1 at IoU >= 0.5 one to one
      (eval/linefind/metrics.py), region exact, merged / split / cut, false alarms, misses, and the empty-region
      rate (the share of `empty` regions for which the finder returns no line at all).

Finders (NAME):
  grad      runtime/vocr_lines.c, VOCR_LINES_GRAD, through its bit-exact twin synth/common.find_lines_ex (with
            --check-c the C driver runtime/vocr_lines_test.c runs too and must give the same boxes)
  refine    GRAD, then runtime/vocr_lines_refine.c with VOCR_REFINE_DEFAULT (twin synth/lines_refine.py)

  det:NAME=BLOB  a detector candidate (synth/textdet_twin.py, the C runtime's bit-exact twin); every line counts in
            linefind (vertical ones too), only horizontal lines are read by a recognizer

Every finder reads the region as the console does: synth/common.to_gray of the RGB region (the recognizer's luma).

  textdet.py baseline --units NAME=DIR ... --linefind JSON --out JSON          the frozen baseline (see above)
  textdet.py detect --cand NAME=BLOB ... --cohorts DIR --out DIR [--cbin DRIVER]
      Every detector candidate on every development whole-dialogue case (VOCR5's tv_dlg / dn_dlg cohorts): its lines
      per case to DIR/dets_NAME.json. With --cbin, the C runtime's driver (runtime/vocr_textdet_test batch mode) runs
      too and must equal the twin on every case (the protocol's eligibility condition).
  textdet.py read --model NAME=BLOB --dets DIR --cohorts DIR --out DIR [--cand NAME ...]
      One recognizer (the C runtime in W8I, eval/finder_tighten.Reader) on every candidate's horizontal lines, read
      in the detector's order, joined and scored by transcript_score: DIR/NAME/units_CAND.json in
      eval/finder_tighten.py's unit format (cohort -> {"det": [units]}), so eval/textdet.wd_units pools them.
  textdet.py select --protocol JSON --baseline DIR --units DIR --linefind JSON ... --blobs DIR --out JSON
      The pre-registered promotion rule (eval/textdet_protocol.json "promotion_rule") over every candidate.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _linefind(module):
    """eval/linefind's modules by path, under their own names (its metrics.py is not eval/metrics.py)."""
    import importlib.util
    name = f"linefind_{module}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / "linefind" / f"{module}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
LINEFIND = ROOT / "eval" / "linefind" / "manifests"
LINEFIND_SETS = {"vita_dev": "development", "psp_dev": "development",
                 "vita_holdout": "report", "psp_holdout": "report"}
TAGS = ("tight", "loose", "empty")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reserved_test_vns():
    from common import test_set_vns
    return set(test_set_vns())


# ---------------------------------------------------------------- finders

class Grad:
    """GRAD (runtime/vocr_lines.c) through its twin; check_c runs the C driver as well."""

    name = "grad"

    def __init__(self, check_c=False):
        self.c = None
        if check_c:
            self.c = _linefind("cfinder").Finder("grad")

    def __call__(self, grey):
        from common import find_lines_ex
        boxes, dark, _ = find_lines_ex(grey, "grad")
        if self.c is not None:
            cb = self.c(grey)
            if [tuple(b) for b in cb] != [tuple(b) for b in boxes] or list(self.c.last_dark) != list(dark):
                raise RuntimeError(f"C GRAD differs from the twin: {cb} / {boxes}")
        return [tuple(int(v) for v in b) for b in boxes], [int(d) for d in dark]


class Refine(Grad):
    """GRAD + VOCR_REFINE_DEFAULT (runtime/vocr_lines_refine.c) through the twin."""

    name = "refine"

    def __call__(self, grey):
        import lines_refine as LR
        boxes, dark = super().__call__(grey)
        b, d, _ = LR.refine_lines(grey, boxes, dark, LR.DEFAULT)
        return [tuple(int(v) for v in x) for x in b], [int(v) for v in d]


def det_refine(lines, grey):
    """Round 2's det+refine (protocol amendment 3): the detector's horizontal lines through VOCR_REFINE_DEFAULT
    (synth/lines_refine.py, the twin of runtime/vocr_lines_refine.c); its vertical lines are kept as they are (never
    read; they keep the region's line count, so the empty rate is the detector's)."""
    import lines_refine as LR
    hz = [ln for ln in lines if not ln[5]]
    b, d, _ = LR.refine_lines(grey, [ln[:4] for ln in hz], [ln[4] for ln in hz], LR.DEFAULT)
    return [(*map(int, bx), int(dk), 0) for bx, dk in zip(b, d)] + [tuple(ln) for ln in lines if ln[5]]


class Det:
    """A detector candidate through the twin: every line (horizontal and vertical); last_vertical holds the flags.
    refine: round 2's det+refine."""

    def __init__(self, name, blob, refine=False):
        import textdet_twin as TW
        self.name = name
        self.model = TW.Model(Path(blob).read_bytes())
        self.refine = refine
        self.last_vertical = []

    def lines(self, grey):
        import textdet_twin as TW
        lines, truncated = TW.find_lines(self.model, grey)
        return (det_refine(lines, grey) if self.refine else lines), truncated

    def __call__(self, grey):
        lines, _ = self.lines(grey)
        self.last_vertical = [ln[5] for ln in lines]
        return [tuple(ln[:4]) for ln in lines], [ln[4] for ln in lines]


def make_finder(spec, check_c=False):
    if spec == "grad":
        return Grad(check_c)
    if spec == "refine":
        return Refine(check_c)
    if spec.startswith(("det:", "detref:")):
        kind, _, rest = spec.partition(":")
        name, _, blob = rest.partition("=")
        return Det(name, blob, refine=kind == "detref")
    raise SystemExit(f"unknown finder {spec!r}")


# ---------------------------------------------------------------- eval/linefind

def linefind_rows(name, test_vns):
    """(kept rows, left-out rows) of one linefind set: regions of a reserved test VN are left out."""
    rows = _linefind("build_gt").read(LINEFIND / f"{name}.jsonl")
    keep = [r for r in rows if r["vn"] not in test_vns]
    return keep, [r for r in rows if r["vn"] in test_vns]


def region_grey(row, shots):
    from common import to_gray
    return np.ascontiguousarray(to_gray(np.ascontiguousarray(_linefind("build_gt").region_rgb(row, shots))))


def gt_of(row):
    x0, y0 = row["region"][:2]
    keep, ign = [], []
    for ln in row["lines"]:
        if not ln.get("box"):
            continue
        b = ln["box"]
        (ign if ln.get("ignore") else keep).append((b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0))
    return keep, ign


def score_linefind(rows, greys, finder):
    M = _linefind("metrics")
    out = []
    for r, g in zip(rows, greys):
        boxes, _ = finder(g)
        keep, ign = gt_of(r)
        m = M.match([(x, y, x + w, y + h) for (x, y, w, h) in boxes], keep, ign)
        out.append({"id": r["id"], "vn": r["vn"], "tags": r.get("tags", []), "found": len(boxes), "gt": len(keep),
                    **{k: m[k] for k in ("tp", "fp", "fn", "merged", "split", "cut")}})
    return out


def summarize(per):
    tp, fp, fn = (sum(o[k] for o in per) for k in ("tp", "fp", "fn"))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = round(2 * p * r / (p + r), 4) if p + r else 0.0
    s = {"regions": len(per), "gt_lines": tp + fn, "found": sum(o["found"] for o in per), "tp": tp, "fp": fp,
         "fn": fn, "line_recall": round(r, 4), "line_precision": round(p, 4), "line_f1": f1,
         "region_exact": round(sum(o["fp"] == 0 and o["fn"] == 0 for o in per) / len(per), 4) if per else None,
         "merged": sum(o["merged"] for o in per), "split": sum(o["split"] for o in per),
         "cut": sum(o["cut"] for o in per)}
    empty = [o for o in per if "empty" in o["tags"]]
    if empty:
        s["empty_regions"] = len(empty)
        s["empty_none_returned"] = sum(o["found"] == 0 for o in empty)
        s["empty_rate"] = round(s["empty_none_returned"] / len(empty), 4)
    return s


def cmd_linefind(args):
    SHOT_DIRS = _linefind("build_gt").SHOT_DIRS
    shots = [str(d) for d in args.shots] + [str(d) for d in SHOT_DIRS]
    test_vns = reserved_test_vns()
    finders = [make_finder(f, args.check_c) for f in args.finder]
    doc = {"kind": "textdet-linefind", "computed": utc_now(),
           "matching": "IoU >= 0.5, one to one (eval/linefind/metrics.py)",
           "reserved_sha256": sha256(ROOT / "eval" / "manifests" / "reserved.json"), "sets": {}}
    per_rows = {}
    for name, role in LINEFIND_SETS.items():
        rows, dropped = linefind_rows(name, test_vns)
        greys = [region_grey(r, shots) for r in rows]
        e = {"role": role, "manifest_sha256": sha256(LINEFIND / f"{name}.jsonl"), "regions": len(rows),
             "left_out_test_vn_regions": len(dropped), "left_out_vns": sorted({r["vn"] for r in dropped}),
             "finders": {}}
        for f in finders:
            per = score_linefind(rows, greys, f)
            per_rows.setdefault(name, {})[f.name] = per
            e["finders"][f.name] = {"all": summarize(per),
                                    **{t: summarize([o for o in per if t in o["tags"]]) for t in TAGS}}
            s = e["finders"][f.name]
            a = s["all"]
            print(f"{name} {f.name}: F1 {a['line_f1']} exact {a['region_exact']} empty {a.get('empty_rate')} "
                  f"m/s/c {a['merged']}/{a['split']}/{a['cut']}", flush=True)
        doc["sets"][name] = e
    pooled = {}
    for f in finders:
        for role in ("development", "report"):
            per = [o for n, r in LINEFIND_SETS.items() if r == role for o in per_rows[n][f.name]]
            pooled.setdefault(role, {})[f.name] = {"all": summarize(per),
                                                   **{t: summarize([o for o in per if t in o["tags"]]) for t in TAGS}}
    doc["pooled"] = pooled
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    if args.rows:
        Path(args.rows).write_text(json.dumps(per_rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- whole dialogue (development)

WD_COHORTS = ("tv_dlg", "dn_dlg")
VARIANTS = ("base", "expanded-4", "expanded-8")
BOOT = {"reps": 10000, "seed": 0, "min_clusters": 5}


def native_vita_shots():
    """Shot ids of the development frames that are native PS Vita screenshots (the CLANNAD-dev-style subset):
    teacher_val frames whose teacher read is of kind 'vita' (not a scaled PC screenshot) and devnative's Vita
    frames (devnative is native by construction)."""
    import glob
    shots = set()
    for p in sorted(glob.glob(str(HERD / "shared" / "artifacts" / "teacher" / "reads" / "train.*of5.jsonl"))):
        for x in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(x)
            except json.JSONDecodeError:
                continue
            if r["kind"] == "vita":
                shots.add(r["shot"])
    for x in (HERD / "shared" / "artifacts" / "devnative" / "reads" / "reads.jsonl").read_text().splitlines():
        r = json.loads(x)
        if r["kind"] == "vita":
            shots.add(r["shot"])
    return shots


def wd_units(res, mode):
    """The pooled development whole dialogue of one arm and finder mode (VOCR5's wd, vocr5_select.load_wd):
    tv_dlg plus the dn_dlg frames tv_dlg does not hold, every variant. Unit keys are cohort-qualified."""
    tv_shots = {u["shot"] for u in res["tv_dlg"][mode]}
    out = []
    for name in WD_COHORTS:
        for u in res[name][mode]:
            if name == "dn_dlg" and u["shot"] in tv_shots:
                continue
            out.append(dict(u, unit=f"{name}|{u['unit']}", cohort=name))
    return sorted(out, key=lambda u: u["unit"])


def cer_of(units, keep=lambda u: True):
    e = sum(u["errors"] for u in units if keep(u))
    n = sum(u["chars"] for u in units if keep(u))
    return {"errors": e, "chars": n, "cer": round(100 * e / n, 4) if n else None}


def vn_diff(pairs, keep=lambda u: True, boot=BOOT):
    """Mean over the (cand_units, base_units) pairs of CER(cand) - CER(base) in points, with a VN-cluster
    bootstrap that resamples the VNs once for all pairs (sorted VN ids, numpy default_rng(seed).integers,
    linear 2.5 / 97.5 percentiles, as eval/vocr3_report.cluster_bootstrap). One pair is exactly
    cluster_bootstrap's interval."""
    ca, cb, chars, vns = [], [], None, None
    for cand, base in pairs:
        a = {u["unit"]: u for u in cand if keep(u)}
        b = {u["unit"]: u for u in base if keep(u)}
        if set(a) != set(b) or any(a[k]["chars"] != b[k]["chars"] for k in a):
            raise ValueError("paired comparison needs identical units and denominators")
        keys = sorted(a)
        if chars is None:
            chars = {k: a[k]["chars"] for k in keys}
            vns = {k: a[k]["vn"] for k in keys}
        elif sorted(chars) != keys:
            raise ValueError("every pair must be scored on the same units")
        ca.append({k: a[k]["errors"] for k in keys})
        cb.append({k: b[k]["errors"] for k in keys})
    clusters = sorted(set(vns.values()))
    ci = {c: i for i, c in enumerate(clusters)}
    K = len(clusters)
    n = np.zeros(K)
    for k, v in chars.items():
        n[ci[vns[k]]] += v
    d = np.zeros(K)
    for ea, eb in zip(ca, cb):
        for k in ea:
            d[ci[vns[k]]] += (ea[k] - eb[k]) / len(ca)
    point = 100 * d.sum() / n.sum() if n.sum() else None
    out = {"delta": round(point, 4) if point is not None else None, "clusters": K, "chars": int(n.sum()),
           "pairs": len(ca), **boot}
    if K < boot["min_clusters"] or not n.sum():
        return dict(out, ci95=None)
    idx = np.random.default_rng(boot["seed"]).integers(0, K, (boot["reps"], K))
    tot = n[idx].sum(1)
    ok = tot > 0
    ds = d[idx].sum(1)[ok] / tot[ok]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(out, ci95=[round(100 * float(lo), 4), round(100 * float(hi), 4)])


def load_units(spec):
    """NAME=DIR: an eval/finder_tighten.py or textdet.py wd score directory holding units_NAME.json."""
    name, _, d = spec.partition("=")
    return name, json.loads((Path(d) / f"units_{name}.json").read_text(encoding="utf-8"))


def wd_tables(arms, modes, native):
    """CER per arm and mode: pooled over variants, per variant, per platform and on the native-Vita subset."""
    keeps = {"all": lambda u: True, **{v: (lambda u, v=v: u["variant"] == v) for v in VARIANTS},
             "vita": lambda u: u["platform"] == "vita", "psp": lambda u: u["platform"] == "psp",
             "native_vita": lambda u: u["shot"] in native}
    tab = {}
    for arm, res in arms.items():
        for mode in modes:
            units = wd_units(res, mode)
            tab.setdefault(arm, {})[mode] = {k: cer_of(units, f) for k, f in keeps.items()}
            tab[arm][mode]["units"] = len(units)
            tab[arm][mode]["vns"] = len({u["vn"] for u in units})
            tab[arm][mode]["native_vita_vns"] = len({u["vn"] for u in units if u["shot"] in native})
    return tab, keeps


def cmd_baseline(args):
    """The frozen baseline: GRAD and GRAD + refine on the development whole dialogue with each VOCR5 model
    (units from finder_tighten.py score), and the linefind detection summary (textdet.py linefind)."""
    native = native_vita_shots()
    arms = dict(load_units(s) for s in args.units)
    tab, keeps = wd_tables(arms, ("grad", "refine"), native)
    cmp_ = {}
    for arm, res in arms.items():
        g, r = wd_units(res, "grad"), wd_units(res, "refine")
        cmp_[arm] = {k: vn_diff([(r, g)], keeps[k]) for k in ("all", "vita", "psp", "native_vita")}
    lf = json.loads(Path(args.linefind).read_text(encoding="utf-8"))
    meta = {a: json.loads((Path(s.partition("=")[2]) / "meta.json").read_text(encoding="utf-8")) for a, s in
            ((s.partition("=")[0], s) for s in args.units)}
    doc = {"kind": "textdet-baseline", "computed": utc_now(),
           "wd": {"definition": "VOCR5's development whole dialogue (eval/vocr5_select.load_wd): tv_dlg plus the "
                                "dn_dlg frames tv_dlg does not hold, base / +4 / +8 px, dialogue text "
                                "(transcript_score all/full); GRAD = VOCR_LINES_GRAD, refine = GRAD + "
                                "VOCR_REFINE_DEFAULT; C finder, C runtime W8I",
                  "cohorts": next(iter(meta.values()))["cohorts"],
                  "arms": {a: m["arms"][a] for a, m in meta.items()},
                  "native_vita_shots": len(native & {u["shot"] for res in arms.values()
                                                     for u in wd_units(res, "grad")}),
                  "cer": tab, "refine_minus_grad": cmp_},
           "linefind": {s: {"regions": e["regions"], "left_out_test_vn_regions": e["left_out_test_vn_regions"],
                            "finders": e["finders"]} for s, e in lf["sets"].items()},
           "linefind_pooled": lf["pooled"]}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for arm in arms:
        g, r = tab[arm]["grad"], tab[arm]["refine"]
        print(f"{arm}: wd GRAD {g['all']['cer']} refine {r['all']['cer']} (Δ {cmp_[arm]['all']['delta']} "
              f"{cmp_[arm]['all']['ci95']}); native Vita {g['native_vita']['cer']} -> {r['native_vita']['cer']} "
              f"({tab[arm]['grad']['native_vita_vns']} VNs)")


# ---------------------------------------------------------------- detector candidates on the development wd

def load_cohort(directory, name):
    import pickle
    with open(Path(directory) / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def case_key(name, c):
    return f"{name}|{c['unit']}|{c['variant']}"


def c_detect(cbin, blob, images, flags=0):
    """The C driver's batch mode: [(lines [(x, y, w, h, dark, vertical)], truncated)] per grey image. flags 2: the
    driver's det+refine mode (the refined horizontal lines only)."""
    import struct
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img in images:
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape
            f.write(struct.pack("<4I", w, h, 1, flags))
            f.write(img.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(cbin), "--batch", str(blob), path], check=True, capture_output=True,
                             text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(out) != len(images):
        raise RuntimeError(f"C driver returned {len(out)} results for {len(images)} images")
    res = []
    for line in out:
        v = [int(t) for t in line.split()]
        if v[0] != 0:
            raise RuntimeError(f"C detector rc={v[0]}")
        n = v[1]
        res.append(([tuple(v[3 + 6 * i:9 + 6 * i]) for i in range(n)], bool(v[2])))
    return res


def cmd_detect(args):
    import textdet_twin as TW
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [(name, c) for name in WD_COHORTS for c in load_cohort(args.cohorts, name)]
    for spec in args.cand:
        name, _, blob = spec.partition("=")
        target = out / f"dets_{name}.json"
        if target.exists() and not args.force:
            print(f"{name}: {target} exists", flush=True)
            continue
        model = TW.Model(Path(blob).read_bytes())
        dets = {}
        for cname, c in cases:
            lines, truncated = TW.find_lines(model, c["image"])
            if args.refine:
                lines = det_refine(lines, c["image"])
            dets[case_key(cname, c)] = {"lines": [list(map(int, ln)) for ln in lines], "truncated": bool(truncated)}
        c_equal = None
        if args.cbin:
            cres = c_detect(args.cbin, blob, [c["image"] for _, c in cases], 2 if args.refine else 0)
            if args.refine:                  # the C driver's det+refine mode drops the vertical lines
                ref = {k: [ln for ln in v["lines"] if not ln[5]] for k, v in dets.items()}
                c_equal = all([list(map(int, ln)) for ln in cl] == ref[case_key(cn, c)] and
                              ct == dets[case_key(cn, c)]["truncated"] for (cn, c), (cl, ct) in zip(cases, cres))
                cres = None
            if cres is not None:
                c_equal = all([list(map(int, ln)) for ln in cl] == dets[case_key(cn, c)]["lines"] and
                              ct == dets[case_key(cn, c)]["truncated"] for (cn, c), (cl, ct) in zip(cases, cres))
            if not c_equal:
                print(f"{name}: C and twin DIFFER", flush=True)
        doc = {"kind": "textdet-dets", "candidate": name, "blob": str(blob), "blob_sha256": sha256(blob),
               "blob_bytes": model.blob_bytes, "shape": model.name, "cohorts": args.cohorts,
               "finder": "det+refine" if args.refine else "det", "c_equals_twin": c_equal,
               "c_driver_sha256": sha256(args.cbin) if args.cbin else None, "computed": utc_now(), "dets": dets}
        target.write_text(json.dumps(doc) + "\n", encoding="utf-8")
        n = sum(len(d["lines"]) for d in dets.values())
        print(f"{name}: {len(dets)} cases, {n} lines, C == twin: {c_equal}", flush=True)


def cmd_read(args):
    import finder_tighten as FT

    from general_vn_textbox_guard import transcript_score
    name, _, blob = args.model.partition("=")
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    ident = FT.build(out / f"build-{os.getpid()}")       # one per process: several may read for one model
    reader = FT.Reader(Path(blob), ident["library"])
    cohorts = {cn: load_cohort(args.cohorts, cn) for cn in WD_COHORTS}
    cache = {}

    def read(g, dark):
        key = (hashlib.blake2b(np.ascontiguousarray(g).tobytes(), digest_size=16).hexdigest(), g.shape, int(dark))
        if key not in cache:
            cache[key] = reader.read(g, dark)
        return cache[key]

    files = sorted(Path(args.dets).glob("dets_*.json"))
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        cand = doc["candidate"]
        if args.cand and cand not in args.cand:
            continue
        target = out / f"units_{cand}.json"
        if target.exists() and not args.force:
            continue
        res = {}
        for cn, cases in cohorts.items():
            units = []
            for c in cases:
                d = doc["dets"][case_key(cn, c)]
                img = c["image"]
                hz = [ln for ln in d["lines"] if not ln[5]]
                preds = [read(img[y:y + h, x:x + w], dk) for x, y, w, h, dk, _ in hz]
                sc = transcript_score(c["lines"], preds)
                units.append({"unit": f"{c['unit']}|{c['variant']}", "shot": c["shot"], "vn": c["vn"],
                              "platform": c["kind"], "variant": c["variant"],
                              "errors": sc["all"]["full"]["errors"], "chars": sc["all"]["full"]["chars"],
                              "hira_errors": sc["all"]["hira"]["errors"], "hira_chars": sc["all"]["hira"]["chars"],
                              "detections": len(hz), "vertical": len(d["lines"]) - len(hz)})
            res[cn] = {"det": units}
        meta = {"model": name, "blob_sha256": reader.sha256, "candidate": cand, "dets_sha256": sha256(f),
                "native": ident, "computed": utc_now()}
        target.write_text(json.dumps({"meta": meta, **res}) + "\n", encoding="utf-8")
        e = sum(u["errors"] for cn in res for u in res[cn]["det"])
        print(f"{name} {cand}: {e} errors, cache {len(cache)} crops", flush=True)


# ---------------------------------------------------------------- the promotion rule

MODELS = ("VOCR5-S", "VOCR5-N", "VOCR5-L")
H15_MACS, H15_MS = 1564992, (5.0, 5.2)          # VOCR3-H15 on the console: MACs per CTC frame, ms per frame
REGIONS = {"vita_box": (900, 150), "psp_box": (450, 80), "vita_screen": (960, 544), "psp_screen": (480, 272)}


def est_ms(macs):
    """MAC-rate estimate (the protocol's): {333: [lo, hi], 444: [lo, hi]} ms. Estimates, not measurements."""
    lo, hi = macs * H15_MS[0] / H15_MACS, macs * H15_MS[1] / H15_MACS
    return {"333": [round(lo, 2), round(hi, 2)], "444": [round(lo * 333 / 444, 2), round(hi * 333 / 444, 2)]}


def costs(shape_name):
    import textdet_model as TM
    shape = TM.SHAPES[shape_name]
    out = {}
    for k, (w, h) in REGIONS.items():
        m = TM.cost(shape, w, h)["macs"]
        out[k] = {"macs": m, "est_ms": est_ms(m)}
    c = TM.cost(shape, 960, 544)
    out.update(params=c["params"], weights_int8=c["weights_int8"])
    return out


def load_baseline(directory):
    """{model: units (finder_tighten format: cohort -> grad / refine)} of the frozen baseline."""
    return {m: json.loads((Path(directory) / m / f"units_{m}.json").read_text(encoding="utf-8")) for m in MODELS}


def load_det_units(directory, cand):
    return {m: json.loads((Path(directory) / m / f"units_{cand}.json").read_text(encoding="utf-8")) for m in MODELS}


def candidate_report(cand, det, base, native, empty_rate, blob_bytes, arena, c_equal, shape):
    """Every number the rule reads for one candidate, plus the rule's conditions."""
    keep_all = lambda u: True  # noqa: E731
    keep_nat = lambda u: u["shot"] in native  # noqa: E731
    rep = {"candidate": cand, "shape": shape, "blob_bytes": blob_bytes, "arena_960x544": arena,
           "c_equals_twin": c_equal, "empty_rate": empty_rate, "costs": costs(shape), "models": {}}
    for m in MODELS:
        d = wd_units(det[m], "det")
        g, r = wd_units(base[m], "grad"), wd_units(base[m], "refine")
        rep["models"][m] = {
            "wd": {"det": cer_of(d), "grad": cer_of(g), "refine": cer_of(r)},
            "native_vita": {"det": cer_of(d, keep_nat), "grad": cer_of(g, keep_nat), "refine": cer_of(r, keep_nat)},
            "det_minus_grad": vn_diff([(d, g)], keep_all), "det_minus_refine": vn_diff([(d, r)], keep_all),
            "native_det_minus_grad": vn_diff([(d, g)], keep_nat),
            "native_det_minus_refine": vn_diff([(d, r)], keep_nat),
            "per_platform": {p: {"det": cer_of(d, lambda u, p=p: u["platform"] == p),
                                 "grad": cer_of(g, lambda u, p=p: u["platform"] == p),
                                 "refine": cer_of(r, lambda u, p=p: u["platform"] == p)} for p in ("vita", "psp")},
            "zero_detection_units": sum(u["detections"] == 0 for u in d)}
    mm = rep["models"]
    below = all(mm[m]["det_minus_grad"]["ci95"][1] < 0 and mm[m]["det_minus_refine"]["ci95"][1] < 0 for m in MODELS)
    native_ok = all(not (mm[m][k]["ci95"] is not None and mm[m][k]["ci95"][0] > 0)
                    for m in MODELS for k in ("native_det_minus_grad", "native_det_minus_refine"))
    rep["mean_wd_cer"] = round(sum(mm[m]["wd"]["det"]["cer"] for m in MODELS) / len(MODELS), 4)
    rep["conditions"] = {"wd_resolved_below_grad_and_refine": below, "no_resolved_native_vita_regression": native_ok,
                         "empty_rate_ge_0_75": empty_rate is not None and empty_rate >= 0.75}
    rep["eligible"] = {"blob_lt_100000": blob_bytes < 100000, "arena_le_131072": arena is not None and arena <= 131072,
                       "macs_900x150_le_30M": rep["costs"]["vita_box"]["macs"] <= 30_000_000,
                       "c_equals_twin": bool(c_equal)}
    rep["is_eligible"] = all(rep["eligible"].values())
    rep["qualifies"] = rep["is_eligible"] and all(rep["conditions"].values())
    rep["point_below_both"] = all(mm[m]["wd"]["det"]["cer"] < min(mm[m]["wd"]["grad"]["cer"],
                                                                  mm[m]["wd"]["refine"]["cer"]) for m in MODELS)
    return rep


def apply_rule(reps, det_units):
    """The protocol's winner: the smallest qualifying candidate; a larger one only by a resolved mean margin."""
    order = sorted(reps, key=lambda c: (reps[c]["blob_bytes"], reps[c]["costs"]["vita_box"]["macs"], c))
    qual = [c for c in order if reps[c]["qualifies"]]
    steps = []
    if not qual:
        elig = [c for c in order if reps[c]["is_eligible"]]
        best = min(elig, key=lambda c: (reps[c]["mean_wd_cer"], c)) if elig else None
        trigger = any(reps[c]["is_eligible"] and reps[c]["conditions"]["no_resolved_native_vita_regression"] and
                      reps[c]["conditions"]["empty_rate_ge_0_75"] and reps[c]["point_below_both"] for c in order)
        return {"winner": None, "order": order, "qualifying": [], "report_candidate": best,
                "fine_tune_triggered": trigger, "steps": steps}
    w = qual[0]
    steps.append({"start": w})
    while True:
        moved = None
        for c in qual[qual.index(w) + 1:]:
            pairs = [(wd_units(det_units[c][m], "det"), wd_units(det_units[w][m], "det")) for m in MODELS]
            d = vn_diff(pairs)
            steps.append({"from": w, "to": c, "mean_delta": d["delta"], "ci95": d["ci95"]})
            if d["ci95"] is not None and d["ci95"][1] < 0:
                moved = c
                break
        if moved is None:
            break
        w = moved
    return {"winner": w, "order": order, "qualifying": qual, "fine_tune_triggered": False, "steps": steps}


def pareto(reps, key="mean_wd_cer"):
    """Candidates no other candidate beats on both blob bytes and the key (lower is better on both)."""
    front = []
    for c, r in reps.items():
        if not any(o != c and q["blob_bytes"] <= r["blob_bytes"] and q[key] <= r[key] and
                   (q["blob_bytes"] < r["blob_bytes"] or q[key] < r[key]) for o, q in reps.items()):
            front.append(c)
    return sorted(front, key=lambda c: reps[c]["blob_bytes"])


def cmd_select(args):
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    runs = next(a for a in proto["amendments"] if a["n"] == args.runs_from)["runs"]
    base = load_baseline(args.baseline)
    native = native_vita_shots()
    arenas = json.loads(Path(args.arena).read_text(encoding="utf-8")) if args.arena else {}
    reps, det_units = {}, {}
    for cand in runs:
        dets = json.loads((Path(args.dets) / f"dets_{cand}.json").read_text(encoding="utf-8"))
        lf = json.loads((Path(args.linefind) / f"linefind_{cand}.json").read_text(encoding="utf-8"))
        empty = lf["pooled"]["development"][cand]["empty"].get("empty_rate")
        det_units[cand] = load_det_units(args.units, cand)
        reps[cand] = candidate_report(cand, det_units[cand], base, native, empty, dets["blob_bytes"],
                                      arenas.get(cand), dets["c_equals_twin"], dets["shape"])
        reps[cand]["linefind_dev"] = lf["pooled"]["development"][cand]["all"]
        reps[cand]["linefind_report"] = lf["pooled"]["report"][cand]["all"]
    rule = apply_rule(reps, det_units)
    doc = {"kind": "textdet-selection", "computed": utc_now(), "protocol_sha256": sha256(args.protocol),
           "runs_from_amendment": args.runs_from, "runs": runs, "rule": rule, "pareto_mean_wd_cer": pareto(reps),
           "candidates": reps,
           "baseline": {m: {"grad": cer_of(wd_units(base[m], "grad")), "refine": cer_of(wd_units(base[m], "refine"))}
                        for m in MODELS}}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for c in rule["order"]:
        r = reps[c]
        mm = r["models"]
        print(f"{c:16s} {r['blob_bytes']:6d} B  wd " + " ".join(f"{mm[m]['wd']['det']['cer']:.2f}" for m in MODELS) +
              "  d-grad " + " ".join(f"{mm[m]['det_minus_grad']['delta']:+.2f}{mm[m]['det_minus_grad']['ci95']}"
                                     for m in MODELS) +
              "  d-ref " + " ".join(f"{mm[m]['det_minus_refine']['delta']:+.2f}{mm[m]['det_minus_refine']['ci95']}"
                                    for m in MODELS) +
              f"  empty {r['empty_rate']}  qual {r['qualifies']}", flush=True)
    print("winner:", rule["winner"], "| report candidate:", rule.get("report_candidate"), "| fine-tune:",
          rule["fine_tune_triggered"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("linefind")
    p.add_argument("--finder", action="append", required=True)
    p.add_argument("--shots", action="append", default=[], type=Path)
    p.add_argument("--check-c", action="store_true", help="also run the C GRAD driver and require equal boxes")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", default="")
    p = sub.add_parser("baseline")
    p.add_argument("--units", action="append", required=True, help="NAME=DIR of a finder_tighten.py score output")
    p.add_argument("--linefind", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--cand", action="append", required=True, help="NAME=BLOB")
    p.add_argument("--cohorts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cbin", default="")
    p.add_argument("--refine", action="store_true", help="round 2's det+refine")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("read")
    p.add_argument("--model", required=True, help="NAME=BLOB")
    p.add_argument("--dets", required=True)
    p.add_argument("--cohorts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cand", action="append", default=[])
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("select")
    p.add_argument("--protocol", default=str(ROOT / "eval" / "textdet_protocol.json"))
    p.add_argument("--baseline", required=True)
    p.add_argument("--units", required=True)
    p.add_argument("--dets", required=True)
    p.add_argument("--linefind", required=True)
    p.add_argument("--arena", default="")
    p.add_argument("--runs-from", type=int, default=2, help="the protocol amendment that lists the round's runs")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "linefind":
        cmd_linefind(args)
    elif args.cmd == "baseline":
        cmd_baseline(args)
    elif args.cmd == "detect":
        cmd_detect(args)
    elif args.cmd == "read":
        cmd_read(args)
    elif args.cmd == "select":
        cmd_select(args)


if __name__ == "__main__":
    main()

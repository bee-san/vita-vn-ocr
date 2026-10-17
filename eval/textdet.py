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

Every finder reads the region as the console does: synth/common.to_gray of the RGB region (the recognizer's luma).
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
for _p in (ROOT, ROOT / "eval", ROOT / "eval" / "linefind", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
            import cfinder
            self.c = cfinder.Finder("grad")

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


def make_finder(spec, check_c=False):
    if spec == "grad":
        return Grad(check_c)
    if spec == "refine":
        return Refine(check_c)
    raise SystemExit(f"unknown finder {spec!r}")


# ---------------------------------------------------------------- eval/linefind

def linefind_rows(name, test_vns):
    """(kept rows, left-out rows) of one linefind set: regions of a reserved test VN are left out."""
    from build_gt import read
    rows = read(LINEFIND / f"{name}.jsonl")
    keep = [r for r in rows if r["vn"] not in test_vns]
    return keep, [r for r in rows if r["vn"] in test_vns]


def region_grey(row, shots):
    from build_gt import region_rgb

    from common import to_gray
    return np.ascontiguousarray(to_gray(np.ascontiguousarray(region_rgb(row, shots))))


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
    import metrics as M
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
    from build_gt import SHOT_DIRS
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
    args = ap.parse_args(argv)
    if args.cmd == "linefind":
        cmd_linefind(args)
    elif args.cmd == "baseline":
        cmd_baseline(args)


if __name__ == "__main__":
    main()

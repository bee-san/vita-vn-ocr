#!/usr/bin/env python3
"""VOCR6 T1 decisions (issue #125, eval/vocr6_protocol.json "decision"): paired arm comparisons with game-cluster
bootstraps, over eval/vocr6_score.py score files.

  vocr6_select.py compare --scores DIR [DIR...] --x RUN[,RUN...] --y RUN[,RUN...] [--path rubi] [--out OUT.json]
  vocr6_select.py table --scores DIR [DIR...] --runs RUN [RUN...] [--path rubi]

A run is named by its score file (score_<RUN>.json under one of the --scores directories). X and Y list runs paired
by position (seed pairs). Each component's difference is mean(Y) - mean(X) of the per-row errors (runs averaged within
an arm), over the component's reference characters, in CER points:

  hval_page   every trusted hval page's whole-dialogue regions (base, +4, +8 px) through the C finder, over the units
              whose finder output has no junk box (one covering no reference line) and that hold no line whose
              duplicate cluster has a train copy (SEEN_UNITS; stage 1 amendment 3): the decision metric; clusters
              = titles
  hval_page_seen  the junk-free units with the train-seen ones kept (report)
  hval_page_nomerge  hval_page without the units where the finder merged two lines into one box (report)
  hval_page_all  every unit, junk boxes included (report)
  hval_clean  hval dialogue lines as clean crops (report); clusters = titles
  dev         0.25 tv + 0.25 dn + 0.5 wd (VOCR5's development score, console view, without the hooked titles' VNs);
              tv, dn and wd resampled independently over VNs; a replicate is the weighted sum

10,000 resamples; numpy seeds hval_page 10, hval_page_seen 13, hval_page_nomerge 14, hval_page_all 12, hval_clean 11,
tv 0, dn 1, wd 2.
Y is adopted over X iff hval_page's interval lies entirely below 0 and every pair's hval_page point difference is
below 0 ("better on hval page-path dialogue CER, across seeds") and the dev score's interval does not lie entirely
above 0 ("not worse on the unseen-game dev sets").
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

N_BOOT = 10000
SEEDS = {"hval_page": 10, "hval_page_seen": 13, "hval_page_nomerge": 14, "hval_page_all": 12, "hval_clean": 11,
         "tv": 0, "dn": 1, "wd": 2}
DEV_WEIGHTS = {"tv": 0.25, "dn": 0.25, "wd": 0.5}
# hval units (page region variants) holding a line whose duplicate cluster has a train copy (vocr6-ingest decisions
# pools.line.seen_train; audit-vocr6-ingest F3): out of the decision metric (stage 1 amendment 3), reported apart
SEEN_UNITS = Path(os.environ.get("VOCR6_SEEN_UNITS", "/local/home/skerraut/herd-vita-ocr/shared/artifacts/vocr6-lead/"
                                 "plans/hval_seen_units_v1.json"))
SEEN_SHA256 = os.environ.get("VOCR6_SEEN_SHA256", "574ee0cdeea1cfb2abc232a4f43eacd674cc5bdf1e59d575eb5bf540cf6a2d19")
_SEEN = {}
# region units left out of every hval_page component of one comparison (--exclude-units; stage 1 amendment 5: the
# units arm A's init trained on, for A against B)
_EXCLUDE = {"units": frozenset()}


def seen_units():
    """The pinned set of hval unit keys that hold a train-seen line (empty set when VOCR6_SEEN_UNITS is "")."""
    if "units" not in _SEEN:
        if not str(SEEN_UNITS) or str(SEEN_UNITS) == ".":
            _SEEN["units"] = frozenset()
        else:
            raw = SEEN_UNITS.read_bytes()
            if SEEN_SHA256 and hashlib.sha256(raw).hexdigest() != SEEN_SHA256:
                raise SystemExit(f"{SEEN_UNITS} differs from its pinned sha256")
            _SEEN["units"] = frozenset(json.loads(raw)["units"])
    return _SEEN["units"]


# hval plans whose region units carry the ids SEEN_UNITS (and the exposed list) were built on: v1 and v2 hold the
# same 6,327 units in the same order (amendment 5); a score read on any other plan is refused
KNOWN_PLANS = {"dfefdd96c0f4c6596f964189a996dd719e35f7dde941fabc61f15cf0e743c8ff": "hval_plan_v1",
               "3a6f80e1dfe4e4f5cf61d2b56aef4253e2039939a76231d344e17a4d351d97ba": "hval_plan_v2",
               "17b054496a612df9b152a16309e7e21237ba86d27f566e7d1980bdde945d063a": "hval_plan_v3"}


def find_score(dirs, run):
    for d in dirs:
        p = Path(d) / f"score_{run}.json"
        if p.exists():
            doc = json.loads(p.read_text(encoding="utf-8"))
            sha = doc.get("hval_plan_sha256")
            if sha and sha not in KNOWN_PLANS and os.environ.get("VOCR6_ANY_PLAN") != "1":
                raise SystemExit(f"{p} was read on an unknown hval plan {sha[:12]}: its unit ids may not be the "
                                 f"seen list's (set VOCR6_ANY_PLAN=1 to override)")
            return doc
    raise SystemExit(f"no score_{run}.json in {dirs}")


def component(doc, name, path):
    """(errors, chars, clusters, row keys) of one component's dialogue rows. hval_page is the page path over the
    region units whose finder output has no junk box (a box that covers no reference line): junk boxes are the
    finder's (T3) and model-independent in where they fall, while what a model reads into them is a lottery that
    swamps every recognizer difference (eval/vocr6_protocol.json decision.page_metric_evidence); hval_page_all
    keeps every unit (report)."""
    r = doc["rows"]
    if name in ("hval_page", "hval_page_all", "hval_page_seen", "hval_page_nomerge"):
        x = r["hval_page"]
        if name == "hval_page_all":
            keep = np.ones(len(x["chars"]), bool)
        else:
            keep = np.asarray(x["junk"]) == 0
            if name in ("hval_page", "hval_page_nomerge"):
                seen = seen_units()
                keep &= np.asarray([u not in seen for u in x["unit"]], bool)
            if name == "hval_page_nomerge":
                keep &= np.asarray(x["merged"]) == 0
        if _EXCLUDE["units"]:
            keep &= np.asarray([u not in _EXCLUDE["units"] for u in x["unit"]], bool)
        cl, keys = np.asarray(x["title"]), x["unit"]
    elif name == "hval_clean":
        x = r["hval_clean"]
        keep = np.asarray(x["kind"]) == "line"
        cl, keys = np.asarray(x["title"]), x["id"]
    elif name in ("tv", "dn"):
        x = r[name]
        keep = ~np.asarray(x["name"], bool)
        cl, keys = np.asarray(x["vn"]), x["id"]
    elif name == "wd":
        x = r["wd"]
        keep = np.ones(len(x["chars"]), bool)
        cl, keys = np.asarray(x["vn"]), x["unit"]
    else:
        raise ValueError(name)
    e = np.asarray(x["errors"][path], float)[keep]
    c = np.asarray(x["chars"], float)[keep]
    return e, c, cl[keep], [k for k, m in zip(keys, keep) if m]


def platform_mask(doc, name, platform):
    x = doc["rows"]["hval_page" if name.startswith("hval_page") else name]
    if name in ("hval_page", "hval_page_seen", "hval_page_nomerge"):
        m = np.asarray(x["junk"]) == 0
        if name in ("hval_page", "hval_page_nomerge"):
            seen = seen_units()
            m &= np.asarray([u not in seen for u in x["unit"]], bool)
        if name == "hval_page_nomerge":
            m &= np.asarray(x["merged"]) == 0
    elif name == "hval_page_all":
        m = np.ones(len(x["chars"]), bool)
    else:
        m = np.asarray(x["kind"]) == "line"
    if name.startswith("hval_page") and _EXCLUDE["units"]:
        m &= np.asarray([u not in _EXCLUDE["units"] for u in x["unit"]], bool)
    return (np.asarray(x["platform"]) == platform)[m]


def mean_errors(docs, name, path, keys=None):
    parts = [component(d, name, path) for d in docs]
    keys = parts[0][3] if keys is None else keys
    for p, d in zip(parts, docs):
        if p[3] != keys:
            raise SystemExit(f"{name}: {d['name']} was scored on other rows")
    return sum(p[0] for p in parts) / len(parts), parts[0][1], parts[0][2]


def paired(xdocs, ydocs, name, path):
    """mean(X) and mean(Y) errors of one component, every run checked to hold the same rows."""
    keys = component(xdocs[0], name, path)[3]
    ex, c, cl = mean_errors(xdocs, name, path, keys)
    ey, _, _ = mean_errors(ydocs, name, path, keys)
    return ex, ey, c, cl


def boot(ex, ey, chars, clusters, seed, n_boot=N_BOOT):
    """Point difference and its replicates (CER points), clusters resampled with replacement."""
    point = (ey.sum() - ex.sum()) / chars.sum() * 100
    ug, inv = np.unique(clusters, return_inverse=True)
    gx = np.bincount(inv, weights=ex, minlength=len(ug))
    gy = np.bincount(inv, weights=ey, minlength=len(ug))
    gc = np.bincount(inv, weights=chars, minlength=len(ug))
    idx = np.random.default_rng(seed).integers(0, len(ug), size=(n_boot, len(ug)))
    return float(point), (gy[idx].sum(1) - gx[idx].sum(1)) / gc[idx].sum(1) * 100, len(ug)


def ci(reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def compare(xdocs, ydocs, path="rubi", n_boot=N_BOOT):
    out = {}
    for name in ("hval_page", "hval_page_seen", "hval_page_nomerge", "hval_page_all", "hval_clean"):
        ex, ey, c, cl = paired(xdocs, ydocs, name, path)
        p, r, n = boot(ex, ey, c, cl, SEEDS[name], n_boot)
        out[name] = {"delta_points": round(p, 3), "ci95": ci(r), "clusters": n,
                     "cer_x": round(100 * ex.sum() / c.sum(), 4), "cer_y": round(100 * ey.sum() / c.sum(), 4)}
        for plat in ("vita", "psp"):
            m = platform_mask(xdocs[0], name, plat)
            if m.any():
                pp, rr, nn = boot(ex[m], ey[m], c[m], cl[m], SEEDS[name], n_boot)
                out[name][plat] = {"delta_points": round(pp, 3), "ci95": ci(rr), "clusters": nn,
                                   "cer_x": round(100 * ex[m].sum() / c[m].sum(), 4),
                                   "cer_y": round(100 * ey[m].sum() / c[m].sum(), 4)}
    point, reps = 0.0, 0.0
    for name, w in DEV_WEIGHTS.items():
        ex, ey, c, cl = paired(xdocs, ydocs, name, path)
        p, r, n = boot(ex, ey, c, cl, SEEDS[name], n_boot)
        out[name] = {"delta_points": round(p, 3), "ci95": ci(r), "clusters": n,
                     "cer_x": round(100 * ex.sum() / c.sum(), 4), "cer_y": round(100 * ey.sum() / c.sum(), 4)}
        point, reps = point + w * p, reps + w * r
    out["dev"] = {"delta_points": round(point, 3), "ci95": ci(reps)}
    return out


def same_plan(docs):
    """Every run of a comparison must have been read on the same hval plan (v1 and v2 share their unit ids, so the
    row check alone would not catch a mix)."""
    plans = {d.get("hval_plan_sha256") for d in docs}
    if len(plans) > 1:
        raise SystemExit("the runs were read on different hval plans: " +
                         ", ".join(f"{d['name']}={str(d.get('hval_plan_sha256'))[:12]}" for d in docs))


def decide(xdocs, ydocs, path="rubi", n_boot=N_BOOT):
    if len(xdocs) != len(ydocs):
        raise SystemExit("X and Y need the same number of runs (seed pairs)")
    same_plan(list(xdocs) + list(ydocs))
    total = compare(xdocs, ydocs, path, n_boot)
    pairs = {f"{x['name']}->{y['name']}": compare([x], [y], path, n_boot) for x, y in zip(xdocs, ydocs)}
    checks = {"hval_page_ci_below_zero": total["hval_page"]["ci95"][1] < 0,
              "every_pair_hval_page_below_zero": all(p["hval_page"]["delta_points"] < 0 for p in pairs.values()),
              "dev_not_resolved_worse": not total["dev"]["ci95"][0] > 0}
    return {"x": [d["name"] for d in xdocs], "y": [d["name"] for d in ydocs], "path": path, "total": total,
            "pairs": {k: {"hval_page": v["hval_page"]["delta_points"], "hval_clean": v["hval_clean"]["delta_points"],
                          "dev": v["dev"]["delta_points"]} for k, v in pairs.items()},
            "checks": checks, "adopt_y": all(checks.values())}


def table(docs, path="rubi"):
    head = ("| run | hval page | (Vita / PSP) | page, junk incl. | hval clean | (Vita / PSP) | dev score | tv | dn "
            "| wd |\n|---|---:|---|---:|---:|---|---:|---:|---:|---:|")
    rows = [head]
    for d in docs:
        s = d["summary"][path]
        rows.append(f"| {d['name']} | {s['hval_page']:.2f}% | {s['hval_page_vita']:.2f} / {s['hval_page_psp']:.2f} | "
                    f"{s['hval_page_all']:.2f}% | {s['hval_clean_dlg']:.2f}% | "
                    f"{s['hval_clean_dlg_vita']:.2f} / {s['hval_clean_dlg_psp']:.2f} | {s['dev_score']:.3f} | "
                    f"{s['tv']:.2f} | {s['dn']:.2f} | {s['wd']:.2f} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--path", default="rubi", choices=("rubi", "greedy"))
    p.add_argument("--out")
    p.add_argument("--exclude-units", help="a JSON file with {\"units\": [...]}: region units left out of every "
                                            "hval_page component (e.g. the units arm A's init trained on)")
    p = sub.add_parser("table")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--path", default="rubi", choices=("rubi", "greedy"))
    a = ap.parse_args()
    if a.cmd == "compare":
        if a.exclude_units:
            _EXCLUDE["units"] = frozenset(json.loads(Path(a.exclude_units).read_text(encoding="utf-8"))["units"])
        xs = [find_score(a.scores, r) for r in a.x.split(",")]
        ys = [find_score(a.scores, r) for r in a.y.split(",")]
        doc = decide(xs, ys, a.path)
        if a.exclude_units:
            doc["excluded_units"] = {"file": a.exclude_units, "n": len(_EXCLUDE["units"]),
                                     "sha256": hashlib.sha256(Path(a.exclude_units).read_bytes()).hexdigest()}
        txt = json.dumps(doc, ensure_ascii=False, indent=1)
        if a.out:
            Path(a.out).write_text(txt + "\n", encoding="utf-8")
        print(txt)
    else:
        print(table([find_score(a.scores, r) for r in a.runs], a.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

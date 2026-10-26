#!/usr/bin/env python3
"""VOCR6 T1 decisions (issue #125, eval/vocr6_protocol.json "decision"): paired arm comparisons with game-cluster
bootstraps, over eval/vocr6_score.py score files.

  vocr6_select.py compare --scores DIR [DIR...] --x RUN[,RUN...] --y RUN[,RUN...] [--path rubi] [--out OUT.json]
  vocr6_select.py table --scores DIR [DIR...] --runs RUN [RUN...] [--path rubi]

A run is named by its score file (score_<RUN>.json under one of the --scores directories). X and Y list runs paired
by position (seed pairs). Each component's difference is mean(Y) - mean(X) of the per-row errors (runs averaged within
an arm), over the component's reference characters, in CER points:

  hval_page   every complete hval page's whole-dialogue regions (base, +4, +8 px) through the C finder: the
              decision metric; clusters = titles
  hval_clean  hval dialogue lines as clean crops (report); clusters = titles
  dev         0.25 tv + 0.25 dn + 0.5 wd (VOCR5's development score, console view, without the hooked titles' VNs);
              tv, dn and wd resampled independently over VNs; a replicate is the weighted sum

10,000 resamples; numpy seeds hval_page 10, hval_clean 11, tv 0, dn 1, wd 2. Y is adopted over X iff hval_page's
interval lies entirely below 0 and every pair's hval_page point difference is below 0 ("better on hval page-path
dialogue CER, across seeds") and the dev score's interval does not lie entirely above 0 ("not worse on the
unseen-game dev sets").
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_BOOT = 10000
SEEDS = {"hval_page": 10, "hval_clean": 11, "tv": 0, "dn": 1, "wd": 2}
DEV_WEIGHTS = {"tv": 0.25, "dn": 0.25, "wd": 0.5}


def find_score(dirs, run):
    for d in dirs:
        p = Path(d) / f"score_{run}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise SystemExit(f"no score_{run}.json in {dirs}")


def component(doc, name, path):
    """(errors, chars, clusters, row keys) of one component's dialogue rows."""
    r = doc["rows"]
    if name == "hval_page":
        x = r["hval_page"]
        keep = np.ones(len(x["chars"]), bool)
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
    x = doc["rows"][name]
    m = np.ones(len(x["chars"]), bool) if name == "hval_page" else np.asarray(x["kind"]) == "line"
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
    for name in ("hval_page", "hval_clean"):
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


def decide(xdocs, ydocs, path="rubi", n_boot=N_BOOT):
    if len(xdocs) != len(ydocs):
        raise SystemExit("X and Y need the same number of runs (seed pairs)")
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
    head = ("| run | hval page | (Vita / PSP) | hval clean | (Vita / PSP) | hval names | dev score | tv | dn | wd |\n"
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|")
    rows = [head]
    for d in docs:
        s = d["summary"][path]
        rows.append(f"| {d['name']} | {s['hval_page']:.2f}% | {s['hval_page_vita']:.2f} / {s['hval_page_psp']:.2f} | "
                    f"{s['hval_clean_dlg']:.2f}% | {s['hval_clean_dlg_vita']:.2f} / {s['hval_clean_dlg_psp']:.2f} | "
                    f"{s['hval_clean_name']:.2f}% | {s['dev_score']:.3f} | {s['tv']:.2f} | {s['dn']:.2f} | "
                    f"{s['wd']:.2f} |")
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
    p = sub.add_parser("table")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--path", default="rubi", choices=("rubi", "greedy"))
    a = ap.parse_args()
    if a.cmd == "compare":
        xs = [find_score(a.scores, r) for r in a.x.split(",")]
        ys = [find_score(a.scores, r) for r in a.y.split(",")]
        doc = decide(xs, ys, a.path)
        txt = json.dumps(doc, ensure_ascii=False, indent=1)
        if a.out:
            Path(a.out).write_text(txt + "\n", encoding="utf-8")
        print(txt)
    else:
        print(table([find_score(a.scores, r) for r in a.runs], a.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

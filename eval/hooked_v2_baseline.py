#!/usr/bin/env python3
"""The VOCR6.2 baseline on the hooked corpus v2 (issue #166, docs/vocr6-data-v2.md): eval/pagepath.py's results on pools
v2.1's units, aggregated into one counts-only record.

  hooked_v2_baseline.py score --res DIR --overlap OVERLAP.json --protocol LINE.json --out RESULT.json

DIR holds pagepath.py run outputs named <set>_<finder>_<tier>.json: set hval_v21 (every hval page's regions, base /
+4 / +8 px) or new_all_base_v21 (every page of the train titles new in v2, base regions), finder spot (Rubi's default:
the textspot8 spotter's crop decode), fix1 (rubi,fix=1: Rubi's GRAD path), oracle (each reference line's own box +
pad: the clean crop) or oraclenf (the clean crop with Rubi's non-Japanese line filter off: --filter none), tier S, N
or L (each with Rubi's cascade to the next tier up); spotlinespunct / spotnone: the spotter's path at N with the line
filter relaxed (--filter lines+punct) or off (report only).

Groups: a title is "new" when pools v1 did not hold it (OVERLAP.json `in_pools_v1`: VOCR6.2 never trained on it) and
"v1" otherwise (VOCR6.2's training titles), each per platform. A group's dialogue CER is pooled (errors / reference
characters over its units) with a 95% bootstrap interval (10,000 resamples, seed 0, numpy's linear percentiles) over
its titles, or over its pages when it has fewer than 5 titles (`ci_over`). Per title (tier N): errors, characters, CER.
The record holds counts, rates, ids and hashes only.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPS, SEED, MIN_TITLES = 10000, 0, 5
SETS = ("hval_v21", "new_all_base_v21")
FINDERS = ("spot", "fix1", "oracle", "oraclenf", "spotlinespunct", "spotnone")
TIERS = ("S", "N", "L")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pooled_ci(units, key, reps=REPS, seed=SEED):
    """95% bootstrap interval of the pooled CER (in %) of `units` resampled by `key(unit)` clusters, or None."""
    sums = defaultdict(lambda: [0, 0])
    for u in units:
        s = sums[key(u)]
        s[0] += u["errors"]
        s[1] += u["chars"]
    if len(sums) < 2:
        return None
    ks = sorted(sums)
    e = np.array([sums[k][0] for k in ks], np.float64)
    n = np.array([sums[k][1] for k in ks], np.float64)
    idx = np.random.default_rng(seed).integers(0, len(ks), (reps, len(ks)))
    tot = n[idx].sum(1)
    ok = tot > 0
    r = 100.0 * e[idx].sum(1)[ok] / tot[ok]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def group_summary(units):
    errors, chars = sum(u["errors"] for u in units), sum(u["chars"] for u in units)
    titles = {u["cluster"] for u in units}
    by_title = len(titles) >= MIN_TITLES
    key = (lambda u: u["cluster"]) if by_title else (lambda u: ":".join(u["uid"].split(":")[:2]))
    return {"units": len(units), "titles": len(titles), "errors": errors, "chars": chars,
            "cer": round(100.0 * errors / chars, 4) if chars else None,
            "ci95": pooled_ci(units, key) if chars else None, "ci_over": "titles" if by_title else "pages",
            "lines": sum(u["lines"] for u in units), "whole": sum(u["whole"] for u in units),
            "cut": sum(u["cut"] for u in units), "missed": sum(u["missed"] for u in units),
            "merged": sum(u["merged"] for u in units), "junk": sum(u["junk"] for u in units)}


def summarize(res, new_titles):
    """{group: summary} of one pagepath result: all, new, v1, each also per platform."""
    groups = defaultdict(list)
    for u in res["units"]:
        g = "new" if u["cluster"] in new_titles else "v1"
        for k in ("all", f"all:{u['platform']}", g, f"{g}:{u['platform']}"):
            groups[k].append(u)
    return {k: group_summary(v) for k, v in sorted(groups.items())}


def per_title(res):
    t = defaultdict(lambda: [0, 0, 0])
    for u in res["units"]:
        s = t[u["cluster"]]
        s[0] += u["errors"]
        s[1] += u["chars"]
        s[2] += 1
    return {k: {"errors": e, "chars": c, "units": n, "cer": round(100.0 * e / c, 4) if c else None}
            for k, (e, c, n) in sorted(t.items())}


def cmd_score(a):
    ov = json.loads(Path(a.overlap).read_text(encoding="utf-8"))
    new_titles = {t for t, v in ov["titles"].items() if not v["in_pools_v1"]}
    out = {"kind": "b2-ingest2-baseline", "schema": 1, "overlap_sha256": sha256_file(a.overlap),
           "protocol": json.loads(Path(a.protocol).read_text(encoding="utf-8")) if a.protocol else None,
           "groups_rule": "new: titles pools v1 did not hold (VOCR6.2 never trained on them); v1: pools v1's titles",
           "ci": f"95% bootstrap, {REPS} resamples, seed {SEED}, over titles (pages when a group has < {MIN_TITLES})",
           "sets": {}}
    for s in SETS:
        for f in FINDERS:
            for t in TIERS:
                p = Path(a.res) / f"{s}_{f}_{t}.json"
                if not p.exists():
                    continue
                res = json.loads(p.read_text(encoding="utf-8"))
                rec = {"result_sha256": sha256_file(p), "units_sha256": res["units_sha256"],
                       "pagepool_manifest_sha256": res.get("pagepool_manifest_sha256"),
                       "blobs": res["blobs"], "lm": res["lm"], "lm_opts": res["lm_opts"], "heads": res.get("heads"),
                       "cascade": res["cascade"], "textnorm": res["textnorm"], "filter": res["filter"],
                       "runtime_lib_sha256": res["runtime_lib_sha256"],
                       "finder": {k: v for k, v in res["finder"].items() if k in ("boxes_sha256", "mode", "fix",
                                                                                  "probe", "refine", "oracle",
                                                                                  "driver_sha256")},
                       "groups": summarize(res, new_titles)}
                if t == "N":                       # per title at the default tier (Normal) only: a compact record
                    rec["per_title"] = per_title(res)
                out["sets"].setdefault(s, {}).setdefault(f, {})[t] = rec
                g = rec["groups"]
                print(f"{s} {f} {t}: " + "  ".join(f"{k} {g[k]['cer']:.2f}%" for k in ("new:vita", "new:psp", "v1:vita",
                                                                                        "v1:psp") if k in g),
                      flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--res", required=True)
    p.add_argument("--overlap", required=True)
    p.add_argument("--protocol", default="", help="the experiment log's protocol line (JSON) to record")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

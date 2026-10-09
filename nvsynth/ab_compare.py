#!/usr/bin/env python3
"""Compare A/B runs scored by eval/run.py (--out JSON), as Markdown tables.

  ab_compare.py RESULTS_DIR [--base synth] [--sets vita-test,psp-test,synth-val] [--reps 10000]

RESULTS_DIR holds <arm>_s<seed>.json files. For every set: CER and exact-line
accuracy per run, then per arm (errors pooled over its seeds), and the difference
to the base arm with a 95% interval from a paired bootstrap that resamples whole
games (so a game with many lines can't dominate) over the same scored lines. Then
CER per style tag, pooled over seeds.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)\.json$")


def load(results_dir):
    runs = defaultdict(dict)                 # arm -> seed -> {set: [line rows]}
    for p in sorted(Path(results_dir).glob("*.json")):
        m = NAME.match(p.name)
        if not m:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        runs[m["arm"]][int(m["seed"])] = {s: r["lines"] for s, r in d["sets"].items() if "lines" in r}
    return runs


def pooled(arm_runs, set_name):
    """{line id: (mean errors over seeds, n, vn, styles)} for the seeds that have the set."""
    per = [r[set_name] for r in arm_runs.values() if set_name in r]
    if not per:
        return None
    out = {}
    for rows in per:
        for r in rows:
            e, n, vn, st = out.get(r["id"], (0.0, r["n"], r.get("vn") or "-", r.get("styles") or []))
            out[r["id"]] = (e + r["errors"] / len(per), n, vn, st)
    return out


def cer(lines, ids=None):
    ids = lines.keys() if ids is None else ids
    e = sum(lines[i][0] for i in ids)
    n = sum(lines[i][1] for i in ids)
    return e / n if n else float("nan")


def bootstrap(a, b, reps, seed=0):
    """CER(b) - CER(a) over the shared lines, games resampled with replacement (lines, for
    sets without games such as synth-val)."""
    ids = sorted(set(a) & set(b))
    games = defaultdict(list)
    for i in ids:
        games[a[i][2]].append(i)
    if len(games) < 2:
        games = {i: [i] for i in ids}
    keys = sorted(games)
    ea = np.array([sum(a[i][0] for i in games[g]) for g in keys])
    eb = np.array([sum(b[i][0] for i in games[g]) for g in keys])
    n = np.array([sum(a[i][1] for i in games[g]) for g in keys], np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), (reps, len(keys)))
    diff = eb[pick].sum(1) / n[pick].sum(1) - ea[pick].sum(1) / n[pick].sum(1)
    point = eb.sum() / n.sum() - ea.sum() / n.sum()
    better, worse = int((eb < ea).sum()), int((eb > ea).sum())
    return point, np.percentile(diff, 2.5), np.percentile(diff, 97.5), better, worse, len(keys)


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--base", default="synth")
    ap.add_argument("--sets", default="vita-test,psp-test,synth-val")
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()
    runs = load(args.results_dir)
    arms = sorted(runs, key=lambda a: (a != args.base, a))
    for s in args.sets.split(","):
        have = {a: pooled(runs[a], s) for a in arms}
        have = {a: v for a, v in have.items() if v}
        if not have:
            continue
        print(f"\n### {s}\n\n| arm | seeds | CER per seed | exact per seed | pooled CER | "
              f"vs {args.base}: CER difference (95% CI, games resampled) |")
        print("|---|---|---|---|---|---|")
        base = have.get(args.base)
        for a, lines in have.items():
            per_cer, per_exact = [], []
            for seed in sorted(runs[a]):
                rows = runs[a][seed].get(s)
                if rows:
                    per_cer.append(sum(r["errors"] for r in rows) / max(1, sum(r["n"] for r in rows)))
                    per_exact.append(sum(r["errors"] == 0 for r in rows) / len(rows))
            vs = "-"
            if base is not None and a != args.base:
                d, lo, hi, better, worse, g = bootstrap(base, lines, args.reps)
                unit = "games" if len({v[2] for v in lines.values()}) > 1 else "lines"
                vs = (f"{100 * d:+.2f} pts ({100 * lo:+.2f} to {100 * hi:+.2f}); "
                      f"better in {better}, worse in {worse} of {g} {unit}")
            print(f"| {a} | {len(per_cer)} | {' / '.join(map(pct, per_cer))} | {' / '.join(map(pct, per_exact))} "
                  f"| {pct(cer(lines))} | {vs} |")
        styles = sorted({t for v in have.values() for (_, _, _, st) in v.values() for t in st})
        if styles:
            print(f"\n| style ({s}) | lines | " + " | ".join(have) + " |")
            print("|---|---|" + "---|" * len(have))
            for t in styles:
                ids = [i for i, v in next(iter(have.values())).items() if t in v[3]]
                print(f"| {t} | {len(ids)} | " + " | ".join(pct(cer(v, [i for i in ids if i in v]))
                                                         for v in have.values()) + " |")


if __name__ == "__main__":
    main()

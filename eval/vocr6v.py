"""VOCR6-V (issue #159): dialogue CER of vertical text through Rubi's own job, paired bootstraps and the rule.

  vocr6v.py score  --units U.jsonl --results R.jsonl [--manifest M.jsonl] [--out S.json]
  vocr6v.py diff   --units U.jsonl --x X.jsonl [--x ...] --y Y.jsonl [--y ...] --unit page|cluster|book
                   [--manifest M.jsonl] [--cohort vn|manga] [--paired] [--out D.json]
  vocr6v.py decide --protocol eval/vocr6v_protocol.json --comparison NAME --hval H.json --vn V.json --manga G.json

The units and results are those of bee-san/Rubi_for_vita's tools/vertical_eval.py (`units` / `ext-units`, and `run`:
host/page_ocr.c, the game worker's job). A unit's dialogue error is computed as that tool's `score` does
(score_unit): the job's lines in its order, those mostly inside another text box of the page (a speaker's name) left
out, each normalised with eval/metrics.norm and joined, against the unit's dialogue columns right to left;
Levenshtein distance over the reference characters. A job that failed (rc != 0) reads nothing.

Intervals: 10,000 bootstrap resamples (seed 0) of the cluster units; a resample's statistic is
sum(errors) / sum(characters). `diff` gives Y - X in points: X and Y may each be one or several runs (seeds); with
several, each side's statistic is the mean over its runs of their CERs on the resample. --paired pairs the i-th X
run with the i-th Y run and also reports each pair's point difference. Cluster units: `page` (a hooked page: the
unit's page), `cluster` (the series cluster of eval/manifests/vertical-dev-regions.jsonl, by the unit's region id)
and `book` (a manga book page: the region's page without its window suffix, docs/vertical.md amendment 1).
Everything here is counts and rates; the results files themselves are private (copyrighted frames' text).
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import lev, norm  # noqa: E402

N_BOOT = 10000


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mostly_inside(d, b):
    ix = max(0, min(d[0] + d[2], b[0] + b[2]) - max(d[0], b[0]))
    iy = max(0, min(d[1] + d[3], b[1] + b[3]) - max(d[1], b[1]))
    return ix * iy >= 0.5 * d[2] * d[3]


def unit_errors(u, r):
    """(errors, characters) of one vertical unit's job result, as Rubi tools/vertical_eval.py score_unit."""
    ox, oy = u["region"][0], u["region"][1]
    lines = r.get("lines") or []
    dets = [[ln["box"][0] + ox, ln["box"][1] + oy, ln["box"][2], ln["box"][3]] for ln in lines]
    ref = "".join(norm(c["text"]) for c in u["refs"])
    keep = [] if r["rc"] else [i for i in range(len(dets)) if not any(mostly_inside(dets[i], b) for b in u["ignore"])]
    hyp = "".join(norm(lines[i]["text"]) for i in keep)
    return lev(ref, hyp), len(ref)


def region_index(manifest):
    """Region id -> its row of eval/manifests/vertical-*-regions.jsonl (cluster, cohort, page)."""
    return {r["id"]: r for r in read_jsonl(manifest)} if manifest else {}


def unit_key(u, how, regions):
    """The bootstrap cluster of a unit."""
    if how == "page":
        return f"{u['title']}:{u['page']}"
    rid = u["id"].rsplit(":", 1)[0]
    row = regions[rid]
    if how == "cluster":
        return row["cluster"]
    if how == "book":
        return re.sub(r"_w\d+$", "", row["page"])
    raise ValueError(how)


def cohort_of(u, regions):
    rid = u["id"].rsplit(":", 1)[0]
    return regions[rid]["cohort"] if rid in regions else "vn"


def per_cluster(units, results, how, regions, cohort=None):
    """{cluster: [errors, characters]} of one run."""
    by_id = {r["id"]: r for r in results}
    out = {}
    for u in units:
        if cohort and cohort_of(u, regions) != cohort:
            continue
        if u["id"] not in by_id:
            raise SystemExit(f"no result for unit {u['id']}")
        e, c = unit_errors(u, by_id[u["id"]])
        k = unit_key(u, how, regions)
        acc = out.setdefault(k, [0, 0])
        acc[0] += e
        acc[1] += c
    return out


def cer(cl):
    e = sum(v[0] for v in cl.values())
    c = sum(v[1] for v in cl.values())
    return 100.0 * e / max(1, c)


def boot_diff(xs, ys, n=N_BOOT, seed=0):
    """Point estimate and 95% interval of mean_j CER(Y_j) - mean_i CER(X_i) (points) over resampled clusters."""
    keys = sorted(set().union(*[set(c) for c in xs + ys]))
    rng = random.Random(seed)
    point = sum(cer(y) for y in ys) / len(ys) - sum(cer(x) for x in xs) / len(xs)
    stats = []
    for _ in range(n):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]

        def mean_cer(runs):
            tot = 0.0
            for run in runs:
                e = c = 0
                for k in pick:
                    v = run.get(k)
                    if v:
                        e += v[0]
                        c += v[1]
                tot += 100.0 * e / max(1, c)
            return tot / len(runs)
        stats.append(mean_cer(ys) - mean_cer(xs))
    stats.sort()
    return round(point, 3), [round(stats[int(0.025 * n)], 3), round(stats[int(0.975 * n) - 1], 3)]


def boot_cer(run, n=N_BOOT, seed=0):
    keys = sorted(run)
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        e = sum(run[k][0] for k in pick)
        c = sum(run[k][1] for k in pick)
        stats.append(100.0 * e / max(1, c))
    stats.sort()
    return [round(stats[int(0.025 * n)], 2), round(stats[int(0.975 * n) - 1], 2)]


def cmd_score(a):
    units = [u for u in read_jsonl(a.units) if u.get("vertical")]
    regions = region_index(a.manifest)
    res = read_jsonl(a.results)
    out = {}
    for cohort in (["vn", "manga"] if regions else [None]):
        how = "book" if cohort == "manga" and a.unit == "cluster" else a.unit   # manga's unit: the book page
        cl = per_cluster(units, res, how, regions, cohort)
        if not cl:
            continue
        out[cohort or "all"] = {"cer": round(cer(cl), 2), "ci95": boot_cer(cl), "unit": how, "clusters": len(cl),
                                "chars": sum(v[1] for v in cl.values()), "errors": sum(v[0] for v in cl.values())}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


def cmd_diff(a):
    units = [u for u in read_jsonl(a.units) if u.get("vertical")]
    regions = region_index(a.manifest)
    xs = [per_cluster(units, read_jsonl(p), a.unit, regions, a.cohort) for p in a.x]
    ys = [per_cluster(units, read_jsonl(p), a.unit, regions, a.cohort) for p in a.y]
    point, ci = boot_diff(xs, ys)
    out = {"unit": a.unit, "cohort": a.cohort, "clusters": len(set().union(*[set(c) for c in xs + ys])),
           "chars": sum(v[1] for v in xs[0].values()),
           "x": [round(cer(x), 3) for x in xs], "y": [round(cer(y), 3) for y in ys],
           "diff": point, "ci95": ci}
    if a.paired:
        if len(xs) != len(ys):
            raise SystemExit("--paired needs as many X runs as Y runs")
        out["pairs"] = [round(cer(y) - cer(x), 3) for x, y in zip(xs, ys)]
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


def cmd_decide(a):
    proto = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    rule = proto["comparisons"][a.comparison]["rule"]
    h, v, g = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.hval, a.vn, a.manga))
    checks = {}
    if rule == "adopt":
        checks["hval_interval_below_0"] = h["ci95"][1] < 0
        checks["every_hval_pair_below_0"] = all(d < 0 for d in h.get("pairs") or [h["diff"]])
        checks["dev_vn_not_above_0"] = not v["ci95"][0] > 0
        checks["dev_manga_not_above_0"] = not g["ci95"][0] > 0
    elif rule == "manga":
        checks["dev_manga_interval_below_0"] = g["ci95"][1] < 0
        checks["hval_not_above_0"] = not h["ci95"][0] > 0
        checks["dev_vn_not_above_0"] = not v["ci95"][0] > 0
    elif rule == "fold":
        checks["dev_vn_not_above_0"] = not v["ci95"][0] > 0
        checks["dev_manga_not_above_0"] = not g["ci95"][0] > 0
    else:
        raise SystemExit(f"unknown rule {rule}")
    out = {"comparison": a.comparison, "rule": rule, "checks": checks, "adopted": all(checks.values()),
           "hval": h, "dev_vn": v, "dev_manga": g}
    print(json.dumps(out, indent=1))


def cmd_crops(a):
    """Diagnostic: CER of a train2-format shard's crops (already turned and in console polarity) read by the C
    runtime's CLI (runtime/build/vocr_cli BLOB --mode w8i --stdin: greedy), one PGM per crop."""
    import subprocess
    import tempfile

    import numpy as np
    z = np.load(a.shard)
    labels = json.loads(str(z["labels"]))
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, lab in enumerate(labels):
            h, w = (int(v) for v in z["shapes"][i])
            g = z["data"][int(z["offsets"][i]):int(z["offsets"][i + 1])].reshape(h, w)
            p = Path(td) / f"{i:06d}.pgm"
            p.write_bytes(f"P5\n{w} {h}\n255\n".encode() + g.tobytes())
            paths.append(str(p))
        out = subprocess.run([a.cli, a.blob, "--mode", "w8i", "--stdin"], input="\n".join(paths) + "\n",
                             capture_output=True, text=True, check=True).stdout.splitlines()
    if len(out) != len(labels):
        raise SystemExit(f"{len(out)} readings for {len(labels)} crops")
    errs = sum(lev(norm(t), norm(p)) for t, p in zip(labels, out))
    chars = sum(len(norm(t)) for t in labels)
    res = {"shard": Path(a.shard).name, "crops": len(labels), "chars": chars, "errors": errs,
           "cer": round(100.0 * errs / max(1, chars), 2),
           "exact": round(sum(norm(t) == norm(p) for t, p in zip(labels, out)) / max(1, len(labels)), 4)}
    print(json.dumps(res))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("crops")
    p.add_argument("--shard", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--cli", default=str(HERE.parent / "runtime" / "build" / "vocr_cli"))
    p = sub.add_parser("score")
    p.add_argument("--units", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--manifest")
    p.add_argument("--unit", default="page", choices=("page", "cluster", "book"))
    p.add_argument("--out")
    p = sub.add_parser("diff")
    p.add_argument("--units", required=True)
    p.add_argument("--x", action="append", required=True)
    p.add_argument("--y", action="append", required=True)
    p.add_argument("--unit", required=True, choices=("page", "cluster", "book"))
    p.add_argument("--manifest")
    p.add_argument("--cohort", choices=("vn", "manga"))
    p.add_argument("--paired", action="store_true")
    p.add_argument("--out")
    p = sub.add_parser("decide")
    p.add_argument("--protocol", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--hval", required=True)
    p.add_argument("--vn", required=True)
    p.add_argument("--manga", required=True)
    a = ap.parse_args(argv)
    {"score": cmd_score, "diff": cmd_diff, "decide": cmd_decide, "crops": cmd_crops}[a.cmd](a)


if __name__ == "__main__":
    main()

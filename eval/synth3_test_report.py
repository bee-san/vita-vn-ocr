#!/usr/bin/env python3
"""The synth3 protocols' locked test report: arms and runs on vita-test and psp-test.

  synth3_test_report.py --protocol eval/synth3_ablation_protocol.json --dir RESULTS_DIR \
      [--spec SPEC.json] [--arms A,B,...] [--pair X=Y ...] [--ref NAME ...] --out result.json

RESULTS_DIR holds one `eval/run.py --out` file per run and polarity, NAME_console.json and
NAME_native.json, each scored once on vita-test and psp-test through the C runtime (vocr:
backend, W8I). Runs are named as in the protocol's arms (A_s2_seed1, ...).

For each polarity (console, the primary; native) and each set (vita-test, psp-test, and both
pooled), it reports every run's and every arm's CER and exact-line rate with integer errors and
denominators (an arm's errors are summed over its seeds), the protocol's comparisons as paired
bootstraps of CER(Y) - CER(X) in points (VNs resampled as clusters, and lines), per eval style
tag, and any extra single-run pairs (--pair X=Y, e.g. the protocol's secondary runs). --ref adds
reference runs (released blobs) to the tables without comparing them. --spec adds the arms and
comparisons of a reported-only spec (eval/synth3_study2_reported.json); --arms keeps only the named
arms, and only the comparisons between two kept arms (one architecture of study 2, say).

These numbers are reported only: the protocols fix that the test sets cannot change any
decision, recipe or dataset.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth3_ablation import bootstrap  # noqa: E402

SETS = ("vita-test", "psp-test")
POLARITIES = ("console", "native")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(directory, name, pol):
    p = Path(directory) / f"{name}_{pol}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    lines = {s: d["sets"][s]["lines"] for s in SETS}
    return {"file": p.name, "sha256": sha256(p), "model_sha256": d["model_info"]["sha256"],
            "blob_bytes": d["model_info"]["blob_bytes"], "arena_bytes": d["model_info"]["arena_bytes"],
            "git": d["env"]["git"], "lines": lines}


def set_lines(run, s):
    return run["lines"]["vita-test"] + run["lines"]["psp-test"] if s == "pooled" else run["lines"][s]


def check_same_lines(runs, s):
    ids = None
    for name, r in runs.items():
        cur = [x["id"] for x in set_lines(r, s)]
        if ids is None:
            ids = cur
        elif cur != ids:
            raise SystemExit(f"{name}: {s} holds different lines")


def summary(errs, exact, chars, scale):
    n = int(chars.sum()) * scale
    return {"errors": int(errs.sum()), "chars": n, "cer": round(float(errs.sum() / n), 5),
            "exact": round(float(exact.sum() / (len(chars) * scale)), 5), "lines": len(chars) * scale}


def arrays(runs, names, s):
    """Per-line errors and exact counts summed over `names`, plus chars, VN ids and styles."""
    base = set_lines(runs[names[0]], s)
    e = np.zeros(len(base))
    x = np.zeros(len(base))
    for name in names:
        ln = set_lines(runs[name], s)
        e += np.array([r["errors"] for r in ln], np.float64)
        x += np.array([r["errors"] == 0 for r in ln], np.float64)
    chars = np.array([r["n"] for r in base], np.float64)
    vns = np.array([r["vn"] for r in base])
    styles = [r["styles"] for r in base]
    return e, x, chars, vns, styles


def compare(runs, a_names, b_names, s, n_boot, seed):
    ea, xa, chars, vns, styles = arrays(runs, a_names, s)
    eb, xb, _, _, _ = arrays(runs, b_names, s)
    scale = len(a_names)
    out = {"vn_cluster": bootstrap(ea, eb, chars, vns, n_boot, seed, scale),
           "line": bootstrap(ea, eb, chars, np.arange(len(chars)), n_boot, seed, scale),
           "exact_a": summary(ea, xa, chars, scale)["exact"], "exact_b": summary(eb, xb, chars, scale)["exact"],
           "styles": {}}
    for t in sorted({t for st in styles for t in st}):
        m = np.array([t in st for st in styles])
        if m.sum() < 3:
            continue
        r = bootstrap(ea[m], eb[m], chars[m], np.arange(int(m.sum())), n_boot, seed, scale)
        r.update({"lines": int(m.sum()), "errors_a": int(ea[m].sum()), "errors_b": int(eb[m].sum()),
                  "chars": int(chars[m].sum()) * scale})
        out["styles"][t] = r
    return out


def build(args):
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    n_boot, seed = proto["bootstrap"]["resamples"], proto["bootstrap"]["seed"]
    arms = {a: rs for a, rs in proto["arms"].items()}
    comparisons = dict(proto["comparisons"])
    if getattr(args, "spec", ""):
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        arms.update(spec["arms"])
        comparisons.update(spec["comparisons"])
    keep = [a for a in getattr(args, "arms", "").split(",") if a]
    if keep:
        arms = {a: arms[a] for a in keep}
        comparisons = {k: v for k, v in comparisons.items() if v[0] in arms and v[1] in arms}
    pairs = [p.split("=", 1) for p in args.pair]
    names = sorted({r for rs in arms.values() for r in rs} | {x for p in pairs for x in p} | set(args.ref))
    res = {"kind": "synth3-test-report", "protocol": Path(args.protocol).name,
           "protocol_sha256": sha256(args.protocol), "bootstrap": proto["bootstrap"],
           "note": "reported only; cannot change any decision, recipe or dataset", "polarities": {}}
    for pol in POLARITIES:
        runs = {n: load_run(args.dir, n, pol) for n in names}
        blobs = {n: {k: r[k] for k in ("file", "sha256", "model_sha256", "blob_bytes", "arena_bytes", "git")}
                 for n, r in runs.items()}
        P = {"runs_files": blobs, "sets": {}}
        for s in SETS + ("pooled",):
            check_same_lines(runs, s)
            S = {"runs": {}, "arms": {}, "comparisons": {}, "pairs": {}, "styles": {}}
            for n in names:
                e, x, chars, _, _ = arrays(runs, [n], s)
                S["runs"][n] = summary(e, x, chars, 1)
            for a, rs in arms.items():
                e, x, chars, _, styles = arrays(runs, rs, s)
                S["arms"][a] = summary(e, x, chars, len(rs))
                S["styles"][a] = {}
                for t in sorted({t for st in styles for t in st}):
                    m = np.array([t in st for st in styles])
                    S["styles"][a][t] = summary(e[m], x[m], chars[m], len(rs))
            for cname, (a, b) in comparisons.items():
                S["comparisons"][cname] = {"a": a, "b": b, **compare(runs, arms[a], arms[b], s, n_boot, seed)}
            for a, b in pairs:
                S["pairs"][f"{a}->{b}"] = {"a": a, "b": b, **compare(runs, [a], [b], s, n_boot, seed)}
            P["sets"][s] = S
        res["polarities"][pol] = P
    return res


def markdown(res):
    out = []
    for pol, P in res["polarities"].items():
        out.append(f"### {pol} polarity\n")
        out.append("| run or arm | vita-test CER / exact | psp-test CER / exact | pooled CER |")
        out.append("|---|---|---|---|")
        sets = P["sets"]
        for kind in ("arms", "runs"):
            for n in sets["pooled"][kind]:
                v, p, q = (sets[s][kind][n] for s in ("vita-test", "psp-test", "pooled"))
                out.append(f"| {n} | {v['cer']:.2%} / {v['exact']:.1%} ({v['errors']}/{v['chars']}) | "
                           f"{p['cer']:.2%} / {p['exact']:.1%} ({p['errors']}/{p['chars']}) | {q['cer']:.2%} |")
        out.append("")
        out.append("| comparison | vita-test (VN cluster) | psp-test (VN cluster) | pooled (VN cluster) "
                   "| pooled (lines) |")
        out.append("|---|---|---|---|---|")
        for kind in ("comparisons", "pairs"):
            for c in sets["pooled"][kind]:
                cells = []
                for s in ("vita-test", "psp-test", "pooled"):
                    r = sets[s][kind][c]["vn_cluster"]
                    cells.append(f"{r['delta_points']:+.2f} ({r['ci95'][0]:+.2f} to {r['ci95'][1]:+.2f})")
                r = sets["pooled"][kind][c]["line"]
                cells.append(f"{r['delta_points']:+.2f} ({r['ci95'][0]:+.2f} to {r['ci95'][1]:+.2f})")
                label = c if kind == "comparisons" else c.replace("->", " → ")
                out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--dir", required=True, help="eval/run.py --out files, NAME_console.json / NAME_native.json")
    ap.add_argument("--spec", default="", help="reported-only spec whose arms and comparisons are added")
    ap.add_argument("--arms", default="", help="comma list: keep only these arms and their comparisons")
    ap.add_argument("--pair", action="append", default=[], help="extra single-run comparison X=Y (Y - X)")
    ap.add_argument("--ref", action="append", default=[], help="reference run reported without comparisons")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="", help="also write the markdown tables here")
    args = ap.parse_args()
    res = build(args)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    md = markdown(res)
    if args.md:
        Path(args.md).write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tables of docs/synth3.md from the stored score files, rounded once from exact values.

  doc_tables.py --scores DIR --test DIR [--md OUT.md]

DIR holds eval/synth3_ablation.py score outputs (A_s2_seed1.json ... secondary_ft.json,
secondary_h15.json); the test DIR holds eval/run.py --out files NAME_<console|native>.json. The
statistics are those of eval/synth3_ablation.py (CER summed over seeds and views, paired bootstrap
with VNs as clusters, 10,000 resamples, seed 0) and eval/synth3_test_report.py, without their
3-decimal rounding, so each printed value is rounded once.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

B, SEED = 10000, 0
VIEWS = ("rv", "rvn")


def boot(ea, eb, chars, groups, scale):
    d = (eb.sum() - ea.sum()) / (chars.sum() * scale) * 100
    ug, inv = np.unique(groups, return_inverse=True)
    ga, gb, gc = (np.bincount(inv, weights=w, minlength=len(ug)) for w in (ea, eb, chars))
    idx = np.random.default_rng(SEED).integers(0, len(ug), size=(B, len(ug)))
    diffs = (gb[idx].sum(1) - ga[idx].sum(1)) / (gc[idx].sum(1) * scale) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return d, lo, hi


def f(d, lo, hi):
    return f"{d:+.2f} ({lo:+.2f} to {hi:+.2f})".replace("-", "−")


def pct(x):
    return f"{100 * x:.2f}%"


def load_scores(paths):
    models, lines = {}, None
    for p in paths:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        lines = lines or s["lines"]
        models.update(s["models"])
    return lines, models


def errs(models, runs, views=VIEWS):
    return sum(np.asarray(models[r]["errors"][v], float) for r in runs for v in views)


def dev_tables(sdir, out):
    lines, models = load_scores(sorted(Path(sdir).glob("*.json")))
    chars = np.array([len(x["ref"]) for x in lines], float)
    vns = np.array([x["vn"] for x in lines])
    n = chars.sum()
    out.append("Study 1 runs (CER, errors / characters):\n")
    for r in ("A_s2_seed1", "A_s2_seed2", "B_s3_seed1", "B_s3_seed2", "C_mix_seed1", "C_mix_seed2"):
        e = {v: np.sum(models[r]["errors"][v]) for v in VIEWS}
        out.append(f"| {r} | {pct(e['rv'] / n)} | {pct(e['rvn'] / n)} |")
    arms = {"A": ["A_s2_seed1", "A_s2_seed2"], "B": ["B_s3_seed1", "B_s3_seed2"], "C": ["C_mix_seed1", "C_mix_seed2"]}
    out.append("\nStudy 1 comparisons (VN cluster; line CI; per seed):\n")
    for name, (a, b) in {"B − A": ("A", "B"), "C − A": ("A", "C"), "C − B": ("B", "C")}.items():
        ea, eb = errs(models, arms[a]), errs(models, arms[b])
        v = boot(ea, eb, chars, vns, 4)
        ln = boot(ea, eb, chars, np.arange(len(chars)), 4)
        seeds = [boot(errs(models, [ra]), errs(models, [rb]), chars, vns, 2)[0] for ra, rb in zip(arms[a], arms[b])]
        nat = np.array([("kind:vita" in x["tags"]) or ("kind:psp" in x["tags"]) for x in lines])
        nv = boot(ea[nat], eb[nat], chars[nat], vns[nat], 4)
        out.append(f"| {name} | {pct(ea.sum() / (4 * n))} → {pct(eb.sum() / (4 * n))} | {f(*v)} | "
                   f"{ln[1]:+.2f} to {ln[2]:+.2f} | {seeds[0]:+.2f} | {seeds[1]:+.2f} | "
                   f"native Vita + PSP ({int(nat.sum())} lines): {f(*nv)} |")
    out.append("\nStrata (arm CER; differences with VN-cluster CI):\n")
    tags = sorted({t for x in lines for t in x["tags"]})
    for t in tags:
        m = np.array([t in x["tags"] for x in lines])
        if m.sum() < 5:
            continue
        e = {k: errs(models, r)[m] for k, r in arms.items()}
        c = chars[m]
        cells = [pct(e[k].sum() / (4 * c.sum())) for k in "ABC"]
        diffs = [f(*boot(e[a], e[b], c, vns[m], 4)) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))]
        out.append(f"| {t} | {int(m.sum())} | " + " | ".join(cells + diffs) + " |")
    out.append("\nSecondaries (one seed; VN cluster):\n")
    sec = {"ft B − ft A": (["ft_A_s2_seed1"], ["ft_B_s3_seed1"]),
           "ft C − ft A": (["ft_A_s2_seed1"], ["ft_C_mix_seed1"]),
           "ft C − ft B": (["ft_B_s3_seed1"], ["ft_C_mix_seed1"]), "ft A − A": (["A_s2_seed1"], ["ft_A_s2_seed1"]),
           "ft B − B": (["B_s3_seed1"], ["ft_B_s3_seed1"]), "ft C − C": (["C_mix_seed1"], ["ft_C_mix_seed1"]),
           "C − A seed 1 (before fine-tuning)": (["A_s2_seed1"], ["C_mix_seed1"]),
           "H15 B − H15 A": (["A_s2_w15_seed1"], ["B_s3_w15_seed1"])}
    native = np.array([("kind:vita" in x["tags"]) or ("kind:psp" in x["tags"]) for x in lines])
    for name, (a, b) in sec.items():
        if not all(r in models for r in a + b):
            continue
        ea, eb = errs(models, a), errs(models, b)
        nat = boot(ea[native], eb[native], chars[native], vns[native], 2)
        out.append(f"| {name} | {pct(ea.sum() / (2 * n))} → {pct(eb.sum() / (2 * n))} | "
                   f"{f(*boot(ea, eb, chars, vns, 2))} "
                   f"| native Vita + PSP ({int(native.sum())} lines): {f(*nat)} |")


def load_test(tdir, name, pol):
    d = json.loads((Path(tdir) / f"{name}_{pol}.json").read_text(encoding="utf-8"))
    return {s: d["sets"][s]["lines"] for s in ("vita-test", "psp-test")}


def test_tables(tdir, out):
    arms = {"A_s2": ["A_s2_seed1", "A_s2_seed2"], "B_s3": ["B_s3_seed1", "B_s3_seed2"],
            "C_mix": ["C_mix_seed1", "C_mix_seed2"]}
    singles = ["ft_A_s2_seed1", "ft_B_s3_seed1", "ft_C_mix_seed1", "A_s2_w15_seed1", "B_s3_w15_seed1", "REF_FL10_v020"]
    for pol in ("console", "native"):
        runs = {}
        for r in [x for rs in arms.values() for x in rs] + singles:
            if (Path(tdir) / f"{r}_{pol}.json").exists():
                runs[r] = load_test(tdir, r, pol)

        def arr(names, s):
            sets = ("vita-test", "psp-test") if s == "pooled" else (s,)
            base = [ln for ss in sets for ln in runs[names[0]][ss]]
            e = sum(np.array([ln["errors"] for ss in sets for ln in runs[r][ss]], float) for r in names)
            c = np.array([ln["n"] for ln in base], float)
            v = np.array([ln["vn"] for ln in base])
            return e, c, v
        out.append(f"\nTest sets, {pol} polarity (CER over seeds):\n")
        groups = {**arms, **{r: [r] for r in singles if r in runs}}
        for g, names in groups.items():
            cells = []
            for s in ("vita-test", "psp-test", "pooled"):
                e, c, _ = arr(names, s)
                cells.append(f"{pct(e.sum() / (len(names) * c.sum()))} ({int(e.sum())}/{int(len(names) * c.sum())})")
            out.append(f"| {g} | " + " | ".join(cells) + " |")
        out.append("")
        pairs = {"B − A": (arms["A_s2"], arms["B_s3"]), "C − A": (arms["A_s2"], arms["C_mix"]),
                 "C − B": (arms["B_s3"], arms["C_mix"]),
                 "ft B − ft A": (["ft_A_s2_seed1"], ["ft_B_s3_seed1"]),
                 "ft C − ft A": (["ft_A_s2_seed1"], ["ft_C_mix_seed1"]),
                 "ft C − ft B": (["ft_B_s3_seed1"], ["ft_C_mix_seed1"]),
                 "H15 B − H15 A": (["A_s2_w15_seed1"], ["B_s3_w15_seed1"])}
        for name, (a, b) in pairs.items():
            if not all(r in runs for r in a + b):
                continue
            cells = []
            for s in ("vita-test", "psp-test", "pooled"):
                ea, c, v = arr(a, s)
                eb, _, _ = arr(b, s)
                cells.append(f(*boot(ea, eb, c, v, len(a))))
            out.append(f"| {name} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    out = []
    dev_tables(args.scores, out)
    test_tables(args.test, out)
    text = "\n".join(out) + "\n"
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

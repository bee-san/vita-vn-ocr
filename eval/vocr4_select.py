#!/usr/bin/env python3
"""VOCR4's decisions and selection (eval/vocr4_protocol.json "decisions", "selection"), development data only.

  vocr4_select.py decide --protocol eval/vocr4_protocol.json --size SIZE --decision real|synth \
                         --tv TV.json [TV2.json ...] --dn DN.json [DN.json ...] [--real REAL.json] --out DECISION.json
  vocr4_select.py select --protocol eval/vocr4_protocol.json --size SIZE --tv TV.json ... --dn DN.json ... \
                         --budget-dir DIR [--budget-dir DIR2] --real REAL.json --synth SYNTH.json --out SELECTION.json
  vocr4_select.py strata --protocol eval/vocr4_protocol.json --size SIZE --tv ... --dn ... --x C --y A --out S.json

Scores are eval/synth3_ablation.py score outputs (C runtime W8I; views rv and rvn) on teacher_val (--tv) and on
devnative's files (--dn: strict and tv_native, ink and GRAD; pooled into one cohort). A run's development score is
the mean of its teacher_val CER and its devnative CER (errors over both views / (2 x reference characters)).

decide: the arm comparison X -> Y over the protocol's seed pairs (errors summed over the pairs' runs and views). The
difference of the development score, Y - X in points, gets a stratified VN-cluster bootstrap: the VNs of
teacher_val and of devnative are resampled independently (10,000 resamples; seeds 0 and 1), and each replicate's
difference is the mean of the two cohorts' differences. Y is adopted iff that interval lies entirely below 0,
every pair's own development-score difference is below 0, and neither cohort's own interval lies entirely above 0.
The real decision compares C -> A; the synth decision compares the real decision's adopted arm (C or A) -> M.
select: the candidates are the runs of the arm the decisions end on (M if the synth decision adopts it, else the
real decision's arm); every candidate's blob must pass the budget (vita/budget_sweep.py check outputs, as
eval/vocr3_select.py reads them); the lowest development score wins (ties at 5 decimals: the lower seed). Every
run of the size is scored and listed; only the candidates can be selected.
strata: report only (decides nothing): Y - X per cohort and per line tag (eval/synth3_ablation.py's strata: capture
kind and target, polarity, ink height, length, name-like, dakuten / small kana, rare kanji, Latin), tags with at
least 20 lines, VN-cluster intervals.
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth3_ablation import sha256  # noqa: E402
from vocr3_select import VIEWS, budget_of, cer, load_scores, pool_cohorts  # noqa: E402

N_BOOT = 10000
ARMS = ("C", "A", "M")
SEEDS = {"teacher_val": 0, "devnative": 1}


def arm_runs(proto, size, arm):
    return list(proto["arms"][arm]["runs"][size])


def errors(models, runs):
    """Per-line errors of a list of runs, summed over runs and views."""
    return sum(models[r]["errors"][v] for r in runs for v in VIEWS)


def replicates(ea, eb, chars, groups, n_boot, seed, scale):
    """(point, replicate array) of CER(B) - CER(A) in points, resampling `groups` as clusters."""
    point = (eb.sum() - ea.sum()) / (chars.sum() * scale) * 100
    ug, inv = np.unique(groups, return_inverse=True)
    ga = np.bincount(inv, weights=ea, minlength=len(ug))
    gb = np.bincount(inv, weights=eb, minlength=len(ug))
    gc = np.bincount(inv, weights=chars, minlength=len(ug))
    idx = np.random.default_rng(seed).integers(0, len(ug), size=(n_boot, len(ug)))
    return float(point), (gb[idx].sum(1) - ga[idx].sum(1)) / (gc[idx].sum(1) * scale) * 100


def summary(point, reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"delta_points": round(point, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def cohort_arrays(lines):
    return (np.array([len(x["ref"]) for x in lines], float), np.array([x["vn"] for x in lines]))


def compare(cohorts, runs_x, runs_y, n_boot=N_BOOT):
    """The development-score difference Y - X (points) with its stratified VN-cluster interval, and per cohort."""
    out, reps = {}, []
    for name, (models, lines) in cohorts.items():
        chars, vns = cohort_arrays(lines)
        scale = len(runs_x) * len(VIEWS)
        p, r = replicates(errors(models, runs_x), errors(models, runs_y), chars, vns, n_boot, SEEDS[name], scale)
        out[name] = summary(p, r)
        out[name].update(cer_x=round(float(errors(models, runs_x).sum() / (chars.sum() * scale)), 5),
                         cer_y=round(float(errors(models, runs_y).sum() / (chars.sum() * scale)), 5))
        reps.append((p, r))
    point = sum(p for p, _ in reps) / len(reps)
    out["development_score"] = summary(point, sum(r for _, r in reps) / len(reps))
    return out


def load_cohorts(tv, dn):
    """teacher_val's score files (all on its lines) and devnative's (any number per subset file: they are grouped
    by the cohort file each was scored on, then the subsets are pooled)."""
    groups = {}
    for p in dn:
        key = json.loads(Path(p).read_text(encoding="utf-8"))["dev"]["file"]
        groups.setdefault(key, []).append(p)
    return {"teacher_val": load_scores(tv), "devnative": pool_cohorts([groups[k] for k in sorted(groups)])}


def decide(proto, size, kind, cohorts, real=None, n_boot=N_BOOT):
    if kind == "real":
        x, y = "C", "A"
    else:
        if real is None or real["size"] != size or real["decision"] != "real":
            raise SystemExit("the synth decision needs this size's real decision (--real)")
        x, y = real["adopted_arm"], "M"
    rx, ry = arm_runs(proto, size, x), arm_runs(proto, size, y)
    if len(rx) != len(ry):
        raise SystemExit("arms need the same number of runs")
    for name, (models, _) in cohorts.items():
        missing = [r for r in rx + ry if r not in models]
        if missing:
            raise SystemExit(f"{name}: not scored {missing}")
    total = compare(cohorts, rx, ry, n_boot)
    pairs = {f"{a}->{b}": compare(cohorts, [a], [b], n_boot) for a, b in zip(rx, ry)}
    d = total["development_score"]
    ok_ci = d["ci95"][1] < 0
    ok_pairs = all(p["development_score"]["delta_points"] < 0 for p in pairs.values())
    ok_guard = all(not total[c]["ci95"][0] > 0 for c in cohorts)
    adopted = bool(ok_ci and ok_pairs and ok_guard)
    return {"kind": "vocr4-decision", "decision": kind, "size": size, "x": x, "y": y, "runs_x": rx, "runs_y": ry,
            "total": total, "pairs": pairs,
            "rule": {"ci_upper_below_zero": bool(ok_ci), "every_pair_below_zero": bool(ok_pairs),
                     "no_cohort_entirely_above_zero": bool(ok_guard)},
            "adopted": adopted, "adopted_arm": y if adopted else x}


def seed_of(run):
    return int(re.search(r"_seed(\d+)$", run).group(1))


def select(proto, size, cohorts, budget_dirs, real, synth):
    if real["decision"] != "real" or synth["decision"] != "synth" or {real["size"], synth["size"]} != {size}:
        raise SystemExit("decisions are for another size or kind")
    if synth["x"] != real["adopted_arm"]:
        raise SystemExit("the synth decision was not taken against the real decision's arm")
    final = synth["adopted_arm"]
    lim = proto["budget"]
    tv, tv_lines = cohorts["teacher_val"]
    dn, dn_lines = cohorts["devnative"]
    cands = set(arm_runs(proto, size, final))
    rows = []
    for arm in ARMS:
        for r in arm_runs(proto, size, arm):
            b = budget_of(budget_dirs, tv[r]["blob_sha256"])
            if b is None:
                raise SystemExit(f"{r}: no budget check for blob {tv[r]['blob_sha256'][:12]}")
            ok = (b["blob_bytes"] <= lim["weights_max_bytes"] and b["arena_w8i"] <= lim["arena_max_bytes"]
                  and b["ms_333_high"] <= lim["ms_max_per_640px_line"])
            t, d = cer(tv[r], tv_lines), cer(dn[r], dn_lines)
            rows.append({"run": r, "arm": arm, "candidate": r in cands, "blob": tv[r]["blob"],
                         "blob_sha256": tv[r]["blob_sha256"], "teacher_val_cer": round(t, 6),
                         "devnative_cer": round(d, 6), "score": round((t + d) / 2, 6), "budget": b,
                         "budget_ok": bool(ok)})
    ok_rows = [r for r in rows if r["candidate"] and r["budget_ok"]]
    if not ok_rows:
        raise SystemExit("no candidate passes the budget")
    best = min(ok_rows, key=lambda r: (round(r["score"], 5), seed_of(r["run"])))
    return {"kind": "vocr4-selection", "size": size, "final_arm": final, "candidates": sorted(cands), "runs": rows,
            "selected": best["run"], "selected_blob": best["blob"], "selected_blob_sha256": best["blob_sha256"]}


def strata(proto, size, cohorts, x, y, n_boot=N_BOOT, min_lines=20):
    """Report only: Y - X (points, VN-cluster interval) per cohort and per line tag (synth3_ablation's strata)."""
    rx, ry = arm_runs(proto, size, x), arm_runs(proto, size, y)
    out = {}
    for name, (models, lines) in cohorts.items():
        chars, vns = cohort_arrays(lines)
        ea, eb = errors(models, rx), errors(models, ry)
        scale = len(rx) * len(VIEWS)
        tags = sorted({t for ln in lines for t in ln.get("tags", [])})
        rows = {}
        for t in tags:
            m = np.array([t in ln.get("tags", []) for ln in lines])
            if m.sum() < min_lines:
                continue
            p, r = replicates(ea[m], eb[m], chars[m], vns[m], n_boot, SEEDS[name], scale)
            rows[t] = {**summary(p, r), "lines": int(m.sum()), "vns": int(len(set(vns[m]))),
                       "cer_x": round(float(ea[m].sum() / (chars[m].sum() * scale)), 5),
                       "cer_y": round(float(eb[m].sum() / (chars[m].sum() * scale)), 5)}
        out[name] = rows
    return {"kind": "vocr4-strata", "size": size, "x": x, "y": y, "runs_x": rx, "runs_y": ry, "report_only": True,
            "strata": out}


def stamp(doc, args, files):
    doc.update(protocol_sha256=sha256(args.protocol),
               computed=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               inputs={str(p): sha256(p) for p in files})
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("decide", "select", "strata"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", required=True)
        p.add_argument("--size", required=True, choices=("FL10", "H15", "W20"))
        p.add_argument("--tv", nargs="+", required=True)
        p.add_argument("--dn", nargs="+", required=True)
        p.add_argument("--real", default="")
        p.add_argument("--out", required=True)
        if name == "decide":
            p.add_argument("--decision", required=True, choices=("real", "synth"))
        elif name == "strata":
            p.add_argument("--x", required=True, choices=ARMS)
            p.add_argument("--y", required=True, choices=ARMS)
        else:
            p.add_argument("--synth", required=True)
            p.add_argument("--budget-dir", action="append", required=True)
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    cohorts = load_cohorts(args.tv, args.dn)
    real = json.loads(Path(args.real).read_text(encoding="utf-8")) if args.real else None
    files = list(args.tv) + list(args.dn) + ([args.real] if args.real else [])
    if args.cmd == "strata":
        doc = stamp(strata(proto, args.size, cohorts, args.x, args.y), args, files)
        for name, rows in doc["strata"].items():
            for t, r in rows.items():
                print(f"{name} {t}: {r['delta_points']:+.3f} {r['ci95']} ({r['lines']} lines, {r['vns']} VNs)")
    elif args.cmd == "decide":
        doc = stamp(decide(proto, args.size, args.decision, cohorts, real), args, files)
        t = doc["total"]
        print(f"{args.size} {args.decision}: {doc['x']} -> {doc['y']}: development score "
              f"{t['development_score']['delta_points']:+.3f} {t['development_score']['ci95']}; teacher_val "
              f"{t['teacher_val']['delta_points']:+.3f} {t['teacher_val']['ci95']}; devnative "
              f"{t['devnative']['delta_points']:+.3f} {t['devnative']['ci95']}; pairs "
              + ", ".join(f"{p['development_score']['delta_points']:+.3f}" for p in doc["pairs"].values())
              + f" -> adopted {doc['adopted']} (arm {doc['adopted_arm']})")
    else:
        synth = json.loads(Path(args.synth).read_text(encoding="utf-8"))
        doc = stamp(select(proto, args.size, cohorts, args.budget_dir, real, synth), args, files + [args.synth])
        for r in sorted(doc["runs"], key=lambda r: r["score"]):
            print(f"{r['run']}: score {r['score']:.5f} (teacher_val {r['teacher_val_cer']:.5f}, devnative "
                  f"{r['devnative_cer']:.5f}) {'candidate' if r['candidate'] else '-'} budget "
                  f"{'ok' if r['budget_ok'] else 'OVER'} {r['budget']['ms_333_high']:.1f} ms")
        print(f"selected {doc['selected']} (arm {doc['final_arm']})")
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

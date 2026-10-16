#!/usr/bin/env python3
"""VOCR5's decisions and selection (eval/vocr5_protocol.json "score", "decisions", "selection"), development data only.

  vocr5_select.py decide --protocol eval/vocr5_protocol.json --tier TIER --decision arm|replace \
                         --tv TV.json ... --dn DN.json ... --wd WD_DIR ... [--arm ARM.json] --out DECISION.json
  vocr5_select.py select --protocol eval/vocr5_protocol.json --tier TIER --tv ... --dn ... --wd ... \
                         --arm ARM.json --replace REPLACE.json --out SELECTION.json
  vocr5_select.py scores --protocol eval/vocr5_protocol.json --tier TIER --tv ... --dn ... --wd ... --out SCORES.json

Inputs (a copy of eval/vocr4_select.py's design with VOCR5's score):
  --tv  eval/synth3_ablation.py score outputs on teacher_val's development part (C runtime W8I, views rv and rvn)
  --dn  the same on devnative's four development files (pooled into one cohort)
  --wd  eval/finder_tighten.py score output directories on the development sample's whole-dialogue cohorts
        (tv_dlg, dn_dlg: units_<run>.json, the C GRAD finder with flags 0 and with the refinement)
A model is named by its run (the score files' model names, the finder_tighten arm names); the incumbent by its tier's
protocol name (e.g. VOCR4-H15).

The development score (protocol "score"): dialogue only. tv = teacher_val's non-name-like lines (name-like: the
`namelike` tag of eval/synth3_ablation.strata), errors of both views over both views' reference characters; dn = the
same over devnative's lines that are not tagged `name` in their file's meta styles; wd = tv_dlg plus the dn_dlg frames
tv_dlg does not hold (finder-tighten's pooling), every unit of both finder modes and all three region variants.
score = 0.25 tv + 0.25 dn + 0.5 wd (in CER points: ½ the fixed-line half, ½ whole dialogue).

decide: Y - X over the seed pairs. Each component's difference gets a VN-cluster bootstrap (10,000 resamples; tv, dn
and wd resampled independently with seeds 0, 1 and 2); a replicate's score difference is 0.25 tv + 0.25 dn + 0.5 wd.
The difference is the mean over the seed pairs (runs averaged within an arm). Y is adopted iff the score interval
lies entirely below 0, every pair's score difference is below 0, and no component's own interval lies entirely
above 0. `arm`: X = the tier's P runs, Y = its F runs (paired by seed). `replace`: X = the incumbent (one blob, the
same for every pair), Y = the arm the arm decision adopted.
select: if the replace decision adopted Y, the run of Y with the lowest score whose blob is within the tier's limits
(weights and W8I arena exact; the box estimate is the shape's, protocol "estimates.table"); ties at 5 decimals go to
the lower seed. Otherwise the incumbent stays.
scores: every run's components and score, and the names column (report only).
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

N_BOOT = 10000
VIEWS = ("rv", "rvn")
SEEDS = {"tv": 0, "dn": 1, "wd": 2}
WEIGHTS = {"tv": 0.25, "dn": 0.25, "wd": 0.5}
TIER_LETTER = {"Small": "S", "Normal": "N", "Large": "L"}
SHAPE = {"Small": "s_w100v4987", "Normal": "n_w150v5631", "Large": "l_lw300v5631"}


class Component:
    """One cohort as rows (lines or units): per-model errors, reference characters, VN clusters, a name mask."""

    def __init__(self, name):
        self.name, self.models, self.chars, self.vns, self.is_name, self.meta = name, {}, None, None, None, {}

    def add(self, model, errors, chars, vns, is_name, meta):
        errors, chars = np.asarray(errors, float), np.asarray(chars, float)
        if self.chars is None:
            self.chars, self.vns, self.is_name = chars, np.asarray(vns), np.asarray(is_name, bool)
        elif len(chars) != len(self.chars) or not (chars == self.chars).all() or list(vns) != list(self.vns):
            raise SystemExit(f"{self.name}: {model} was scored on other rows")
        if model in self.models:
            raise SystemExit(f"{self.name}: {model} scored twice")
        self.models[model], self.meta[model] = errors, meta

    def rate(self, runs, part="dialogue"):
        m = ~self.is_name if part == "dialogue" else self.is_name
        return float(sum(self.models[r][m].sum() for r in runs) / (len(runs) * self.chars[m].sum()))


def _dn_names(score_doc):
    """{line id: True when devnative's meta styles tag it a name} of a devnative development file."""
    z = np.load(score_doc["dev"]["file"])
    return {m["id"]: "name" in (m.get("styles") or []) for m in json.loads(str(z["meta"]))}


def load_fixed(paths, comp, devnative):
    """Score files of one cohort (several files over different lines are pooled, each model's rows in file order)."""
    by_model = {}
    groups = {}
    for p in paths:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        groups.setdefault(s["dev"]["file"], []).append(s)
    for f in sorted(groups):
        names = _dn_names(groups[f][0]) if devnative else None
        for s in groups[f]:
            lines = s["lines"]
            chars = [len(x["ref"]) * len(VIEWS) for x in lines]
            vns = [x["vn"] for x in lines]
            is_name = [names[x["id"]] if devnative else "namelike" in x.get("tags", []) for x in lines]
            for model, m in s["models"].items():
                e = sum(np.asarray(m["errors"][v], float) for v in VIEWS)
                by_model.setdefault(model, []).append((f, e, chars, vns, is_name,
                                                       {k: m[k] for k in ("blob", "blob_sha256", "blob_bytes",
                                                                          "arena_bytes")}))
    for model, parts in by_model.items():
        if [p[0] for p in parts] != sorted(groups):
            raise SystemExit(f"{comp.name}: {model} lacks a cohort file")
        comp.add(model, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
                 sum((list(p[3]) for p in parts), []), np.concatenate([p[4] for p in parts]), parts[0][5])


def load_wd(dirs, comp):
    """finder_tighten score directories: units_<arm>.json, pooled as finder-tighten's pooled_dialogue."""
    for d in dirs:
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        for arm, info in meta["arms"].items():
            res = json.loads((d / f"units_{arm}.json").read_text(encoding="utf-8"))
            tv_shots = {u["shot"] for u in res.get("tv_dlg", {}).get("grad", [])}
            rows = []
            for name in ("tv_dlg", "dn_dlg"):
                for mode in ("grad", "refine"):
                    for u in res.get(name, {}).get(mode, []):
                        if name == "dn_dlg" and u["shot"] in tv_shots:
                            continue
                        rows.append((f"{name}|{mode}|{u['unit']}", u["errors"], u["chars"], u["vn"]))
            rows.sort()
            comp.add(arm, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                     np.zeros(len(rows), bool), {"blob_sha256": info["sha256"], "blob_bytes": info["blob_bytes"],
                                                 "arena_bytes": info["arena_w8i"], "flags": meta["flags"]})


def load(args):
    tv, dn, wd = Component("tv"), Component("dn"), Component("wd")
    load_fixed(args.tv, tv, devnative=False)
    load_fixed(args.dn, dn, devnative=True)
    load_wd(args.wd, wd)
    return {"tv": tv, "dn": dn, "wd": wd}


def replicates(comp, runs_x, runs_y, seed, n_boot):
    """(point, replicates) of the dialogue CER difference mean(Y) - mean(X), points, VN clusters resampled."""
    m = ~comp.is_name
    ex = sum(comp.models[r][m] for r in runs_x) / len(runs_x)
    ey = sum(comp.models[r][m] for r in runs_y) / len(runs_y)
    chars, vns = comp.chars[m], comp.vns[m]
    point = (ey.sum() - ex.sum()) / chars.sum() * 100
    ug, inv = np.unique(vns, return_inverse=True)
    gx = np.bincount(inv, weights=ex, minlength=len(ug))
    gy = np.bincount(inv, weights=ey, minlength=len(ug))
    gc = np.bincount(inv, weights=chars, minlength=len(ug))
    idx = np.random.default_rng(seed).integers(0, len(ug), size=(n_boot, len(ug)))
    return float(point), (gy[idx].sum(1) - gx[idx].sum(1)) / gc[idx].sum(1) * 100, len(ug)


def summary(point, reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"delta_points": round(float(point), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def compare(comps, runs_x, runs_y, n_boot=N_BOOT):
    out, point, reps = {}, 0.0, 0.0
    for name, comp in comps.items():
        p, r, n = replicates(comp, runs_x, runs_y, SEEDS[name], n_boot)
        out[name] = {**summary(p, r), "vns": n, "cer_x": round(comp.rate(runs_x), 6),
                     "cer_y": round(comp.rate(runs_y), 6)}
        point, reps = point + WEIGHTS[name] * p, reps + WEIGHTS[name] * r
    out["score"] = summary(point, reps)
    return out


def score_of(comps, run):
    return sum(WEIGHTS[n] * c.rate([run]) * 100 for n, c in comps.items())


def seed_of(run):
    return int(re.search(r"_seed(\d+)$", run).group(1))


def tier_runs(proto, tier, arm):
    return list(proto["runs"]["fine_tunes"][tier][arm])


def check_scored(comps, runs):
    for name, c in comps.items():
        missing = [r for r in runs if r not in c.models]
        if missing:
            raise SystemExit(f"{name}: not scored {missing}")


def rule(total, pairs, comps):
    ok_ci = total["score"]["ci95"][1] < 0
    ok_pairs = all(p["score"]["delta_points"] < 0 for p in pairs.values())
    ok_guard = all(not total[c]["ci95"][0] > 0 for c in comps)
    return {"ci_upper_below_zero": bool(ok_ci), "every_pair_below_zero": bool(ok_pairs),
            "no_component_entirely_above_zero": bool(ok_guard)}, bool(ok_ci and ok_pairs and ok_guard)


def decide(proto, tier, kind, comps, arm_doc=None, n_boot=N_BOOT):
    inc = proto["tiers"][tier]["incumbent"]["name"]
    if kind == "arm":
        x, y = "P", "F"
        rx, ry = tier_runs(proto, tier, "P"), tier_runs(proto, tier, "F")
    else:
        if arm_doc is None or arm_doc["tier"] != tier or arm_doc["decision"] != "arm":
            raise SystemExit("the replace decision needs this tier's arm decision (--arm)")
        x, y = inc, arm_doc["adopted_arm"]
        ry = tier_runs(proto, tier, y)
        rx = [inc] * len(ry)
    check_scored(comps, sorted(set(rx + ry)))
    total = compare(comps, rx, ry, n_boot)
    pairs = {f"{a}->{b}": compare(comps, [a], [b], n_boot) for a, b in zip(rx, ry)}
    r, adopted = rule(total, pairs, comps)
    adopted_arm = (y if adopted else x) if kind == "arm" else None
    return {"kind": "vocr5-decision", "decision": kind, "tier": tier, "x": x, "y": y, "runs_x": rx, "runs_y": ry,
            "total": total, "pairs": pairs, "rule": r, "adopted": adopted,
            **({"adopted_arm": adopted_arm} if kind == "arm" else {"replaces": adopted})}


def within_limits(proto, tier, meta):
    t = proto["tiers"][tier]
    lim, exp = t["limits"], t["expected"]
    w_ok = meta["blob_bytes"] < lim["weights_lt"] if "weights_lt" in lim else meta["blob_bytes"] <= lim["weights_le"]
    shape_ok = meta["blob_bytes"] == exp["blob_bytes"] and meta["arena_bytes"] == exp["arena_w8i_bytes"]
    est = proto["estimates"]["table"][SHAPE[tier]]
    box_ok = all(v[1] <= lim["box_s_le"] for k in ("main", "#85", "#85 no-PLD") for v in est[k].values())
    return bool(w_ok and meta["arena_bytes"] <= lim["arena_le"] and shape_ok and box_ok)


def scores(proto, tier, comps):
    inc = proto["tiers"][tier]["incumbent"]["name"]
    runs = [r for arm in ("P", "F") for r in tier_runs(proto, tier, arm)] + [inc]
    rows = []
    for r in runs:
        if not all(r in c.models for c in comps.values()):
            continue
        meta = comps["tv"].meta[r]
        rows.append({"run": r, "arm": "P" if r.startswith("v5p") else "F" if r.startswith("v5f") else "incumbent",
                     "score": round(score_of(comps, r), 5),
                     **{f"{n}_cer": round(c.rate([r]), 6) for n, c in comps.items()},
                     **{f"{n}_names_cer": round(c.rate([r], "names"), 6) for n, c in comps.items() if n != "wd"},
                     "blob": meta.get("blob"), "blob_sha256": meta["blob_sha256"], "blob_bytes": meta["blob_bytes"],
                     "arena_w8i": meta["arena_bytes"],
                     "within_limits": within_limits(proto, tier, meta) if r != inc else None})
    return rows


def select(proto, tier, comps, arm_doc, rep_doc):
    if arm_doc["decision"] != "arm" or rep_doc["decision"] != "replace" or {arm_doc["tier"], rep_doc["tier"]} != {tier}:
        raise SystemExit("decisions are for another tier or kind")
    if rep_doc["y"] != arm_doc["adopted_arm"]:
        raise SystemExit("the replace decision was not taken for the adopted arm")
    rows = scores(proto, tier, comps)
    inc = proto["tiers"][tier]["incumbent"]
    doc = {"kind": "vocr5-selection", "tier": tier, "adopted_arm": arm_doc["adopted_arm"],
           "replaces": rep_doc["replaces"], "runs": rows}
    if not rep_doc["replaces"]:
        return {**doc, "selected": inc["name"], "selected_blob": inc["blob"], "selected_blob_sha256": inc["sha256"]}
    cands = set(tier_runs(proto, tier, arm_doc["adopted_arm"]))
    ok = [r for r in rows if r["run"] in cands and r["within_limits"]]
    if not ok:
        raise SystemExit("no candidate is within the tier's limits")
    best = min(ok, key=lambda r: (round(r["score"], 5), seed_of(r["run"])))
    return {**doc, "candidates": sorted(cands), "selected": best["run"], "selected_blob": best["blob"],
            "selected_blob_sha256": best["blob_sha256"]}


def stamp(doc, args, files):
    doc.update(protocol_sha256=sha256(args.protocol),
               computed=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               inputs={str(p): sha256(p) for p in files})
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("decide", "select", "scores"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", required=True)
        p.add_argument("--tier", required=True, choices=tuple(TIER_LETTER))
        p.add_argument("--tv", nargs="+", required=True)
        p.add_argument("--dn", nargs="+", required=True)
        p.add_argument("--wd", nargs="+", required=True)
        p.add_argument("--out", required=True)
        if name == "decide":
            p.add_argument("--decision", required=True, choices=("arm", "replace"))
            p.add_argument("--arm", default="")
        elif name == "select":
            p.add_argument("--arm", required=True)
            p.add_argument("--replace", required=True)
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    comps = load(args)
    wd_files = [str(Path(d) / f) for d in args.wd for f in ("meta.json",)] + \
        [str(p) for d in args.wd for p in sorted(Path(d).glob("units_*.json"))]
    files = list(args.tv) + list(args.dn) + wd_files
    if args.cmd == "decide":
        arm_doc = json.loads(Path(args.arm).read_text(encoding="utf-8")) if args.arm else None
        doc = stamp(decide(proto, args.tier, args.decision, comps, arm_doc), args,
                    files + ([args.arm] if args.arm else []))
        t = doc["total"]
        print(f"{args.tier} {args.decision}: {doc['x']} -> {doc['y']}: score {t['score']['delta_points']:+.3f} "
              f"{t['score']['ci95']}; " + "; ".join(f"{c} {t[c]['delta_points']:+.3f} {t[c]['ci95']}" for c in comps)
              + "; pairs " + ", ".join(f"{p['score']['delta_points']:+.3f}" for p in doc["pairs"].values())
              + f" -> adopted {doc['adopted']}")
    elif args.cmd == "select":
        arm_doc = json.loads(Path(args.arm).read_text(encoding="utf-8"))
        rep_doc = json.loads(Path(args.replace).read_text(encoding="utf-8"))
        doc = stamp(select(proto, args.tier, comps, arm_doc, rep_doc), args, files + [args.arm, args.replace])
        for r in sorted(doc["runs"], key=lambda r: r["score"]):
            print(f"{r['run']}: score {r['score']:.4f} (tv {r['tv_cer']:.5f} dn {r['dn_cer']:.5f} "
                  f"wd {r['wd_cer']:.5f}) limits {r['within_limits']}")
        print(f"selected {doc['selected']}")
    else:
        doc = stamp({"kind": "vocr5-scores", "tier": args.tier, "runs": scores(proto, args.tier, comps)}, args, files)
        for r in sorted(doc["runs"], key=lambda r: r["score"]):
            print(f"{r['run']}: score {r['score']:.4f} (tv {r['tv_cer']:.5f} dn {r['dn_cer']:.5f} "
                  f"wd {r['wd_cer']:.5f})")
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

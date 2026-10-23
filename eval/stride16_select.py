#!/usr/bin/env python3
"""stride16's decisions and selection (eval/stride16_protocol.json "score", "floor", "decisions", "selection"),
development data only. A copy of eval/vocr5_select.py's scoring (the same cohorts, rows, pooling and VN-cluster
bootstrap) with stride16's rule: a coarser-stride candidate replaces its tier's VOCR5 incumbent when it is NOT
RESOLVED WORSE, not when it is resolved better.

  stride16_select.py decide --protocol eval/stride16_protocol.json --arm S16|N16|L16|N12 --floor FLOOR.json \\
                            --tv TV.json ... --dn DN.json ... --wd WD_DIR ... --out DECISION.json
  stride16_select.py select --protocol ... --tier Small|Normal|Large --decision DECISION.json [DECISION.json] \\
                            --floor ... --tv ... --dn ... --wd ... --out SELECTION.json
  stride16_select.py scores --protocol ... --tier TIER --floor ... --tv ... --dn ... --wd ... --out SCORES.json

Inputs: --tv / --dn eval/synth3_ablation.py score outputs (C runtime W8I, views rv and rvn) on teacher_val's and
devnative's development files; --wd eval/finder_tighten.py score directories on the whole-dialogue cohorts (tv_dlg,
dn_dlg; GRAD with flags 0 and with the refinement); --floor eval/stride16_floor.py's frame budget of every row. A
model is named by its run, the incumbent by its protocol name (VOCR5-S, VOCR5-N, VOCR5-L).

Components (VOCR5's, dialogue only): tv = teacher_val's non-name-like lines, errors of both views over both views'
reference characters; dn = the same over devnative's lines not tagged `name`; wd = tv_dlg plus the dn_dlg frames
tv_dlg does not hold, every unit of both finder modes and all three region variants. Score = 0.25 tv + 0.25 dn +
0.5 wd in CER points. Strata (protocol "floor", at the candidate's stride s): a row is TIGHT when T_s < 1.25 NR
(NR = reference characters + adjacent repeats; T_s its frames at stride s) and BELOW THE FLOOR when T_s < NR; the
fixed-line tight stratum pools tv's and dn's tight dialogue lines, the whole-dialogue one wd's tight units.

decide (one candidate arm against its tier's incumbent; the difference is the mean over the arm's seeds of
candidate - incumbent, a stratified VN-cluster bootstrap of 10,000 resamples: tv, dn and wd with numpy seeds 0, 1
and 2 as VOCR5's, the fixed-line tight stratum seed 3, the whole-dialogue tight stratum seed 4). The arm is adopted
iff (a) wd's interval does not lie entirely above 0, (b) neither tv's nor dn's does, (c) neither tight stratum's
does (a stratum holding fewer than 5 VN clusters cannot be resolved and passes, flagged), and (d) every run of the
arm is within the tier's limits (protocol tiers.<T>.limits: exact weights, the W8I arena at the runtime's default
tile, the shape's box estimate).
select: Small and Large take their arm's decision; Normal takes N16's, and N12's only when N16 is not adopted. The
adopted arm's run with the lowest score (ties at 5 decimals to the lower seed) is selected, else the incumbent stays.
scores: every scored run's components, names, strata and per-platform rates (report only).
"""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

import numpy as np

N_BOOT = 10000
VIEWS = ("rv", "rvn")
SEEDS = {"tv": 0, "dn": 1, "wd": 2, "fixed_tight": 3, "wd_tight": 4}
WEIGHTS = {"tv": 0.25, "dn": 0.25, "wd": 0.5}
MIN_CLUSTERS = 5
TIGHT = 1.25


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Component:
    """One cohort as rows (lines or units): per-model errors, reference characters, VN clusters, a name mask, and
    each row's key, platform, frames per stride and CTC need (from the floor file)."""

    def __init__(self, name):
        self.name, self.models, self.meta = name, {}, {}
        self.chars = self.vns = self.is_name = self.keys = None
        self.platform, self.frames, self.nr = None, None, None

    def add(self, model, errors, chars, vns, is_name, keys, meta):
        errors, chars = np.asarray(errors, float), np.asarray(chars, float)
        if self.chars is None:
            self.chars, self.vns, self.is_name = chars, np.asarray(vns), np.asarray(is_name, bool)
            self.keys = list(keys)
        elif len(chars) != len(self.chars) or not (chars == self.chars).all() or list(vns) != list(self.vns) \
                or list(keys) != self.keys:
            raise SystemExit(f"{self.name}: {model} was scored on other rows")
        if model in self.models:
            raise SystemExit(f"{self.name}: {model} scored twice")
        self.models[model], self.meta[model] = errors, meta

    def attach_floor(self, rows):
        """rows: key -> floor record (T per stride, nr, platform)."""
        missing = [k for k in self.keys if k not in rows]
        if missing:
            raise SystemExit(f"{self.name}: {len(missing)} rows lack a frame budget, e.g. {missing[:3]}")
        self.platform = np.asarray([rows[k]["platform"] for k in self.keys])
        self.nr = np.asarray([rows[k]["nr"] for k in self.keys], float)
        self.frames = {s: np.asarray([rows[k]["T"][str(s)] for k in self.keys], float) for s in (8, 12, 16)}

    def ratio(self, stride):
        return self.frames[stride] / np.maximum(self.nr, 1.0)

    def mask(self, part="dialogue"):
        return ~self.is_name if part == "dialogue" else self.is_name

    def rate(self, runs, part="dialogue", extra=None):
        m = self.mask(part) if extra is None else self.mask(part) & extra
        if not m.any():
            return None
        return float(sum(self.models[r][m].sum() for r in runs) / (len(runs) * self.chars[m].sum()))


def _dn_names(score_doc):
    """{line id: True when devnative's meta styles tag it a name} of a devnative development file."""
    z = np.load(score_doc["dev"]["file"])
    return {m["id"]: "name" in (m.get("styles") or []) for m in json.loads(str(z["meta"]))}


def load_fixed(paths, comp, devnative):
    """vocr5_select.load_fixed with each row's key: (file basename, line id)."""
    by_model, groups = {}, {}
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
            keys = [f"{Path(f).name}|{k}|{x['id']}" for k, x in enumerate(lines)]   # a row is its position
            for model, m in s["models"].items():
                e = sum(np.asarray(m["errors"][v], float) for v in VIEWS)
                by_model.setdefault(model, []).append((f, e, chars, vns, is_name, keys,
                                                       {k: m[k] for k in ("blob", "blob_sha256", "blob_bytes",
                                                                          "arena_bytes")}))
    for model, parts in by_model.items():
        if [p[0] for p in parts] != sorted(groups):
            raise SystemExit(f"{comp.name}: {model} lacks a cohort file")
        comp.add(model, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
                 sum((list(p[3]) for p in parts), []), np.concatenate([p[4] for p in parts]),
                 sum((list(p[5]) for p in parts), []), parts[0][6])


def load_wd(dirs, comp):
    """vocr5_select.load_wd (finder-tighten's pooled_dialogue) with each row's key."""
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
                     np.zeros(len(rows), bool), [r[0] for r in rows],
                     {"blob_sha256": info["sha256"], "blob_bytes": info["blob_bytes"],
                      "arena_bytes": info["arena_w8i"], "flags": meta["flags"]})


def load(args):
    floor = json.loads(Path(args.floor).read_text(encoding="utf-8"))
    tv, dn, wd = Component("tv"), Component("dn"), Component("wd")
    load_fixed(args.tv, tv, devnative=False)
    load_fixed(args.dn, dn, devnative=True)
    load_wd(args.wd, wd)
    tv.attach_floor({f"teacher_val_dev.npz|{k}|{v['id']}": v for k, v in enumerate(floor["tv"])})
    dn.attach_floor({f"{f}|{k}|{v['id']}": v for f, rows in floor["dn"].items() for k, v in enumerate(rows)})
    wd.attach_floor(floor["wd"])
    return {"tv": tv, "dn": dn, "wd": wd}, floor


def _clusters(errors_x, errors_y, chars, vns, seed, n_boot):
    point = (errors_y.sum() - errors_x.sum()) / chars.sum() * 100
    ug, inv = np.unique(vns, return_inverse=True)
    gx = np.bincount(inv, weights=errors_x, minlength=len(ug))
    gy = np.bincount(inv, weights=errors_y, minlength=len(ug))
    gc = np.bincount(inv, weights=chars, minlength=len(ug))
    idx = np.random.default_rng(seed).integers(0, len(ug), size=(n_boot, len(ug)))
    return float(point), (gy[idx].sum(1) - gx[idx].sum(1)) / gc[idx].sum(1) * 100, len(ug)


def replicates(comp, runs_x, runs_y, seed, n_boot, extra=None):
    """(point, replicates, clusters) of the dialogue CER difference mean(Y) - mean(X), points, VN clusters
    resampled; extra: an optional row mask (a stratum)."""
    m = comp.mask() if extra is None else comp.mask() & extra
    ex = sum(comp.models[r][m] for r in runs_x) / len(runs_x)
    ey = sum(comp.models[r][m] for r in runs_y) / len(runs_y)
    return _clusters(ex, ey, comp.chars[m], comp.vns[m], seed, n_boot)


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


def stratum_rows(comps, stride, which):
    """The tight (T_s < 1.25 NR) or floor (T_s < NR) dialogue rows of each component at `stride`."""
    lim = TIGHT if which == "tight" else 1.0
    return {n: c.ratio(stride) < lim for n, c in comps.items()}


def compare_stratum(comps, names, stride, runs_x, runs_y, seed, which="tight", n_boot=N_BOOT):
    """One pooled stratum over components `names` (rows concatenated; a VN in several is one cluster)."""
    masks = stratum_rows(comps, stride, which)
    ex, ey, ch, vn, rows = [], [], [], [], 0
    for n in names:
        c = comps[n]
        m = c.mask() & masks[n]
        rows += int(m.sum())
        ex.append(sum(c.models[r][m] for r in runs_x) / len(runs_x))
        ey.append(sum(c.models[r][m] for r in runs_y) / len(runs_y))
        ch.append(c.chars[m])
        vn.append(c.vns[m])
    ex, ey, ch, vn = (np.concatenate(v) for v in (ex, ey, ch, vn))
    rec = {"stratum": which, "stride": stride, "components": list(names), "rows": rows,
           "chars": int(ch.sum()), "vns": int(len(np.unique(vn)))}
    if rows == 0 or ch.sum() == 0:
        return {**rec, "resolvable": False}
    p, reps, n = _clusters(ex, ey, ch, vn, seed, n_boot)
    return {**rec, **summary(p, reps), "resolvable": n >= MIN_CLUSTERS,
            "cer_x": round(float(ex.sum() / ch.sum()), 6), "cer_y": round(float(ey.sum() / ch.sum()), 6)}


def seed_of(run):
    return int(re.search(r"_seed(\d+)$", run).group(1))


def check_scored(comps, runs):
    for name, c in comps.items():
        missing = [r for r in runs if r not in c.models]
        if missing:
            raise SystemExit(f"{name}: not scored {missing}")


def within_limits(proto, arm, meta):
    a = proto["arms"][arm]
    lim = proto["tiers"][a["tier"]]["limits"]
    w_ok = meta["blob_bytes"] == lim["weights_eq"]
    arena_ok = meta["arena_bytes"] <= lim["arena_w8i_le"]
    est = proto["estimates"]["table"][arm]
    box_ok = all(est[k][mhz][1] <= lim["box_s_le"] for k in ("#85 no-PLD",) for mhz in ("333", "444"))
    return bool(w_ok and arena_ok and box_ok)


def not_above(ci):
    return not ci[0] > 0


def decide(proto, arm, comps, n_boot=N_BOOT):
    a = proto["arms"][arm]
    tier = a["tier"]
    inc = proto["tiers"][tier]["incumbent"]["name"]
    ry = list(a["runs"]["fine_tunes"])
    rx = [inc] * len(ry)
    check_scored(comps, sorted(set(rx + ry)))
    stride = int(a["stride"])
    total = compare(comps, rx, ry, n_boot)
    pairs = {f"{x}->{y}": compare(comps, [x], [y], n_boot) for x, y in zip(rx, ry)}
    strata = {"fixed_tight": compare_stratum(comps, ("tv", "dn"), stride, rx, ry, SEEDS["fixed_tight"], "tight",
                                             n_boot),
              "wd_tight": compare_stratum(comps, ("wd",), stride, rx, ry, SEEDS["wd_tight"], "tight", n_boot),
              "fixed_floor": compare_stratum(comps, ("tv", "dn"), stride, rx, ry, SEEDS["fixed_tight"], "floor",
                                             n_boot),
              "wd_floor": compare_stratum(comps, ("wd",), stride, rx, ry, SEEDS["wd_tight"], "floor", n_boot)}
    limits = {r: within_limits(proto, arm, comps["tv"].meta[r]) for r in ry}
    rule = {"a_wd_not_resolved_worse": not_above(total["wd"]["ci95"]),
            "b_tv_not_resolved_worse": not_above(total["tv"]["ci95"]),
            "b_dn_not_resolved_worse": not_above(total["dn"]["ci95"]),
            "c_fixed_tight_not_resolved_worse": (not strata["fixed_tight"]["resolvable"]
                                                 or not_above(strata["fixed_tight"]["ci95"])),
            "c_wd_tight_not_resolved_worse": (not strata["wd_tight"]["resolvable"]
                                              or not_above(strata["wd_tight"]["ci95"])),
            "d_within_limits": all(limits.values())}
    adopted = all(rule.values())
    return {"kind": "stride16-decision", "arm": arm, "tier": tier, "stride": stride, "x": inc, "y": arm,
            "runs_x": rx, "runs_y": ry, "total": total, "pairs": pairs, "strata": strata, "limits": limits,
            "rule": rule, "adopted": bool(adopted),
            "flags": [k for k in ("fixed_tight", "wd_tight") if not strata[k]["resolvable"]]}


def platform_rates(comp, runs):
    out = {}
    for plat in sorted(set(comp.platform.tolist())):
        r = comp.rate(runs, extra=comp.platform == plat)
        if r is not None:
            out[plat] = round(r, 6)
    return out


def scores(proto, tier, comps):
    inc = proto["tiers"][tier]["incumbent"]["name"]
    arms = [k for k, a in proto["arms"].items() if a["tier"] == tier]
    runs = [(r, k) for k in arms for r in proto["arms"][k]["runs"]["fine_tunes"]] + [(inc, "incumbent")]
    rows = []
    for r, arm in runs:
        if not all(r in c.models for c in comps.values()):
            continue
        meta = comps["tv"].meta[r]
        stride = int(proto["arms"][arm]["stride"]) if arm != "incumbent" else 8
        strata = {}
        for s in (12, 16):
            for which, lim in (("tight", TIGHT), ("floor", 1.0)):
                strata[f"s{s}_{which}"] = {n: c.rate([r], extra=c.ratio(s) < lim) for n, c in comps.items()}
        rows.append({"run": r, "arm": arm, "stride": stride,
                     "score": round(sum(WEIGHTS[n] * c.rate([r]) * 100 for n, c in comps.items()), 5),
                     **{f"{n}_cer": round(c.rate([r]), 6) for n, c in comps.items()},
                     **{f"{n}_names_cer": round(c.rate([r], "names"), 6) for n, c in comps.items() if n != "wd"},
                     "platform": {n: platform_rates(c, [r]) for n, c in comps.items()},
                     "strata": strata,
                     "blob": meta.get("blob"), "blob_sha256": meta["blob_sha256"], "blob_bytes": meta["blob_bytes"],
                     "arena_w8i": meta["arena_bytes"],
                     "within_limits": within_limits(proto, arm, meta) if arm != "incumbent" else None})
    return rows


def select(proto, tier, comps, decisions):
    order = proto["selection"]["order"][tier]
    by_arm = {d["arm"]: d for d in decisions}
    if any(d["tier"] != tier for d in decisions):
        raise SystemExit("a decision is for another tier")
    chosen, considered = None, []
    for arm in order:
        if arm not in by_arm:
            raise SystemExit(f"{tier}: the decision for {arm} is needed (protocol selection.order)")
        considered.append(arm)
        if by_arm[arm]["adopted"]:
            chosen = arm
            break
    rows = scores(proto, tier, comps)
    inc = proto["tiers"][tier]["incumbent"]
    doc = {"kind": "stride16-selection", "tier": tier, "considered": considered, "adopted_arm": chosen,
           "replaces": chosen is not None, "runs": rows}
    if chosen is None:
        return {**doc, "selected": inc["name"], "selected_blob": inc["blob"], "selected_blob_sha256": inc["sha256"]}
    cands = set(proto["arms"][chosen]["runs"]["fine_tunes"])
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("decide", "select", "scores"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", required=True)
        p.add_argument("--floor", required=True)
        p.add_argument("--tv", nargs="+", required=True)
        p.add_argument("--dn", nargs="+", required=True)
        p.add_argument("--wd", nargs="+", required=True)
        p.add_argument("--out", required=True)
        if name == "decide":
            p.add_argument("--arm", required=True)
        else:
            p.add_argument("--tier", required=True, choices=("Small", "Normal", "Large"))
        if name == "select":
            p.add_argument("--decision", nargs="+", required=True)
    args = ap.parse_args(argv)
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    comps, _ = load(args)
    wd_files = [str(Path(d) / "meta.json") for d in args.wd] + \
        [str(p) for d in args.wd for p in sorted(Path(d).glob("units_*.json"))]
    files = list(args.tv) + list(args.dn) + wd_files + [args.floor]
    if args.cmd == "decide":
        doc = stamp(decide(proto, args.arm, comps), args, files)
        t, st = doc["total"], doc["strata"]
        print(f"{args.arm} vs {doc['x']}: score {t['score']['delta_points']:+.3f} {t['score']['ci95']}; "
              + "; ".join(f"{c} {t[c]['delta_points']:+.3f} {t[c]['ci95']}" for c in comps)
              + "; " + "; ".join(f"{k} {v.get('delta_points', 'n/a')} {v.get('ci95', '')} ({v['rows']} rows, "
                                 f"{v['vns']} VNs)" for k, v in st.items())
              + f" -> rule {doc['rule']} adopted {doc['adopted']}")
    elif args.cmd == "select":
        decisions = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.decision]
        doc = stamp(select(proto, args.tier, comps, decisions), args, files + list(args.decision))
        for r in sorted(doc["runs"], key=lambda r: r["score"]):
            print(f"{r['run']}: score {r['score']:.4f} (tv {r['tv_cer']:.5f} dn {r['dn_cer']:.5f} "
                  f"wd {r['wd_cer']:.5f}) limits {r['within_limits']}")
        print(f"selected {doc['selected']}")
    else:
        doc = stamp({"kind": "stride16-scores", "tier": args.tier, "runs": scores(proto, args.tier, comps)}, args,
                    files)
        for r in sorted(doc["runs"], key=lambda r: r["score"]):
            print(f"{r['run']}: score {r['score']:.4f} (tv {r['tv_cer']:.5f} dn {r['dn_cer']:.5f} "
                  f"wd {r['wd_cer']:.5f})")
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

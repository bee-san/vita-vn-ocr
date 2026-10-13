#!/usr/bin/env python3
"""VOCR3 selection (eval/vocr3_protocol.json "selection", "budget", "candidates.variants").

  vocr3_select.py base   --protocol eval/vocr3_protocol.json --size FL10|H15|W20 \
                         --tv TV_SCORES.json [TV_SCORES2.json ...] [--dn DN_SCORES.json ...] [--dn-commit SHA] \
                         --budget-dir DIR [--budget-dir DIR2 ...] --out selection.json
  vocr3_select.py realplus --protocol eval/vocr3_protocol.json --size SIZE --tv TV.json ... [--dn DN.json ...] \
                         --out realplus.json
  vocr3_select.py base ... --realplus realplus.json      (the candidates follow the realplus decision)
  vocr3_select.py variant --selection selection.json --name RUN --tv TV.json ... [--dn DN.json ...] \
                         --out variant.json

Scores are eval/synth3_ablation.py score outputs (C runtime W8I, views rv and rvn). A candidate's
development score is its CER over teacher_val in both views (errors / reference characters); with
devnative (any number of its per-subset score files: strict and tv_native, ink and GRAD crops), the
mean of teacher_val's CER and devnative's pooled CER. Every candidate must also pass the budget:
vita/budget_sweep.py check's sizes.json and a9 result in a --budget-dir (blob bytes, W8I arena at the
default tiling, and the high bound of the estimated ms per 640-px line at 333 MHz). The base rule
picks the lowest score among the size's budget-passing candidates (ties at 5 decimals: the shorter
pretrain, then the lower seed). `variant` applies the variant ablation rule against the base
selection: the variant replaces it iff the VN-cluster 95% interval of its teacher_val CER minus the
base's lies entirely below 0 and, when devnative selects, devnative's interval is not entirely above 0.
`realplus` applies amendment 4: over the size's decision pairs (base fine-tune, rp run), errors summed,
realplus is adopted iff teacher_val's VN-cluster interval lies entirely below 0, every pair's difference is
below 0, and devnative's interval (when given) is not entirely above 0; `base --realplus` then takes the rp
runs (decision pairs plus extra runs) as the candidates when adopted, the protocol's candidates otherwise.
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
from synth3_ablation import bootstrap, sha256  # noqa: E402

VIEWS = ("rv", "rvn")
CEIL = None


def load_scores(paths):
    """{model: {view: per-line errors}}, lines (ids, vn, chars) of one cohort from score files on the same lines."""
    models, lines = {}, None
    for p in paths:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        if lines is None:
            lines = s["lines"]
        elif [x["id"] for x in s["lines"]] != [x["id"] for x in lines]:
            raise SystemExit(f"{p}: holds other lines than {paths[0]}")
        for name, m in s["models"].items():
            models[name] = {"errors": {v: np.asarray(m["errors"][v], float) for v in VIEWS},
                            "blob": m.get("blob"), "blob_sha256": m["blob_sha256"], "blob_bytes": m["blob_bytes"],
                            "arena_bytes": m["arena_bytes"]}
    return models, lines


def pool_cohorts(groups):
    """Join several cohorts (lists of score files over different lines, the same models) into one."""
    models, lines = {}, []
    for paths in groups:
        m, ln = load_scores(paths)
        tag = Path(paths[0]).stem
        lines += [{**x, "id": f"{tag}:{x['id']}"} for x in ln]
        for name, v in m.items():
            if name not in models:
                models[name] = {**v, "errors": {k: [] for k in VIEWS}}
            for k in VIEWS:
                models[name]["errors"][k].append(v["errors"][k])
    for v in models.values():
        v["errors"] = {k: np.concatenate(e) for k, e in v["errors"].items()}
    return models, lines


def cer(model, lines):
    n = sum(len(x["ref"]) for x in lines)
    return float(sum(model["errors"][v].sum() for v in VIEWS) / (len(VIEWS) * n))


def budget_of(dirs, blob_sha):
    """(blob bytes, W8I arena, ms high at 333 MHz) of the blob from vita/budget_sweep.py check outputs."""
    for d in dirs:
        d = Path(d)
        sizes = json.loads((d / "sizes.json").read_text(encoding="utf-8"))
        for name, info in sizes.items():
            for r in sorted((d / "a9").glob(f"{name}_l2*.json")):
                a = json.loads(r.read_text(encoding="utf-8"))
                if a.get("blob_sha256") == blob_sha and a.get("tile") in (0, 4) and a.get("l2_kib") == 2048:
                    return {"blob_bytes": info["blob_bytes"], "arena_w8i": info["arena"]["t4h16"]["w8i"],
                            "ms_333_high": float(a["ms_640px"]["333"][1]), "check": str(r)}
    return None


def run_key(name):
    m = re.search(r"_pre(\d+)k", name)
    pre = int(m.group(1)) * 1000 if m else 0
    seed = int(re.search(r"_seed(\d+)", name).group(1))
    return pre, seed


def candidate_names(proto, size, realplus=None):
    cands = proto["candidates"][size]
    names = list(cands.get("study2", [])) + list(cands.get("step_curve", [])) + list(cands.get("runs", [])) + \
        list(cands.get("extra", []))
    if realplus is not None:
        if realplus["size"] != size:
            raise SystemExit("the realplus decision is for another size")
        if realplus["adopted"]:
            rp = proto["candidates"]["realplus"][size]
            names = [b for _, b in rp["decision_pairs"]] + list(rp["extra_runs"])
    return names


def cmd_realplus(args):
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    pairs = proto["candidates"]["realplus"][args.size]["decision_pairs"]
    tv, tv_lines = load_scores(args.tv)
    chars = np.array([len(x["ref"]) for x in tv_lines], float)
    vns = np.array([x["vn"] for x in tv_lines])
    base_runs, rp_runs = [a for a, _ in pairs], [b for _, b in pairs]
    ea = sum(tv[r]["errors"][v] for r in base_runs for v in VIEWS)
    eb = sum(tv[r]["errors"][v] for r in rp_runs for v in VIEWS)
    t = bootstrap(ea, eb, chars, vns, 10000, 0, len(pairs) * len(VIEWS))
    per_pair = {f"{a}->{b}": bootstrap(sum(tv[a]["errors"][v] for v in VIEWS), sum(tv[b]["errors"][v] for v in VIEWS),
                                       chars, vns, 10000, 0, len(VIEWS)) for a, b in pairs}
    res = {"kind": "vocr3-realplus-decision", "size": args.size, "protocol_sha256": sha256(args.protocol),
           "computed": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "pairs": pairs, "teacher_val": t, "per_pair": per_pair, "devnative": None,
           "devnative_files": [str(x) for x in args.dn], "devnative_sha256": [sha256(x) for x in args.dn]}
    ok = t["ci95"][1] < 0 and all(x["delta_points"] < 0 for x in per_pair.values())
    if args.dn:
        dn, dn_lines = pool_cohorts([[x] for x in args.dn])
        c = np.array([len(x["ref"]) for x in dn_lines], float)
        g = np.array([x["vn"] for x in dn_lines])
        d = bootstrap(sum(dn[r]["errors"][v] for r in base_runs for v in VIEWS),
                      sum(dn[r]["errors"][v] for r in rp_runs for v in VIEWS), c, g, 10000, 0, len(pairs) * len(VIEWS))
        res["devnative"] = d
        ok = ok and not d["ci95"][0] > 0
    res["adopted"] = bool(ok)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"{args.size} realplus: teacher_val {t['delta_points']:+.3f} {t['ci95']}; pairs "
          + ", ".join(f"{x['delta_points']:+.2f}" for x in per_pair.values())
          + (f"; devnative {res['devnative']['delta_points']:+.3f} {res['devnative']['ci95']}" if args.dn else "")
          + f" -> adopted {res['adopted']}")


def cmd_base(args):
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    rp = json.loads(Path(args.realplus).read_text(encoding="utf-8")) if getattr(args, "realplus", "") else None
    names = candidate_names(proto, args.size, rp)
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    lim = proto["budget"]
    tv, tv_lines = load_scores(args.tv)
    dn = dn_lines = None
    if args.dn:
        dn, dn_lines = pool_cohorts([[p] for p in args.dn])
    rows = []
    for n in names:
        if n not in tv:
            raise SystemExit(f"{n}: not scored on teacher_val")
        b = budget_of(args.budget_dir, tv[n]["blob_sha256"])
        if b is None:
            raise SystemExit(f"{n}: no budget check for blob {tv[n]['blob_sha256'][:12]}")
        ok = (b["blob_bytes"] <= lim["weights_max_bytes"] and b["arena_w8i"] <= lim["arena_max_bytes"]
              and b["ms_333_high"] <= lim["ms_max_per_640px_line"])
        t = cer(tv[n], tv_lines)
        d = cer(dn[n], dn_lines) if dn else None
        score = (t + d) / 2 if dn else t
        rows.append({"run": n, "blob": tv[n]["blob"], "blob_sha256": tv[n]["blob_sha256"],
                     "teacher_val_cer": round(t, 6),
                     "devnative_cer": None if d is None else round(d, 6), "score": round(score, 6),
                     "budget": b, "budget_ok": bool(ok)})
    ok_rows = [r for r in rows if r["budget_ok"]]
    if not ok_rows:
        raise SystemExit("no candidate passes the budget")
    best = min(ok_rows, key=lambda r: (round(r["score"], 5), *run_key(r["run"])))
    out = {"kind": "vocr3-selection", "size": args.size, "protocol_sha256": sha256(args.protocol),
           "computed": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "teacher_val": {"files": [str(p) for p in args.tv], "lines": len(tv_lines)},
           "devnative": {"files": [str(p) for p in args.dn], "commit": args.dn_commit or None,
                         "lines": len(dn_lines) if dn_lines else 0, "sha256": [sha256(p) for p in args.dn]},
           "realplus_decision": None if rp is None else {"file": str(args.realplus), "adopted": rp["adopted"]},
           "candidates": rows, "selected": best["run"], "selected_blob": best["blob"],
           "selected_blob_sha256": best["blob_sha256"]}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for r in sorted(rows, key=lambda r: r["score"]):
        print(f"{r['run']}: score {r['score']:.5f} (teacher_val {r['teacher_val_cer']:.5f}"
              + (f", devnative {r['devnative_cer']:.5f}" if r["devnative_cer"] is not None else "")
              + f") budget {'ok' if r['budget_ok'] else 'OVER'} {r['budget']['ms_333_high']:.1f} ms")
    print(f"selected {best['run']}")


def cmd_variant(args):
    sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    base = sel["selected"]
    tv, tv_lines = load_scores(args.tv)
    chars = np.array([len(x["ref"]) for x in tv_lines], float)
    vns = np.array([x["vn"] for x in tv_lines])
    ea = sum(tv[base]["errors"][v] for v in VIEWS)
    eb = sum(tv[args.name]["errors"][v] for v in VIEWS)
    t = bootstrap(ea, eb, chars, vns, 10000, 0, len(VIEWS))
    res = {"kind": "vocr3-variant", "size": sel["size"], "base": base, "variant": args.name,
           "teacher_val": t, "devnative": None}
    replaces = t["ci95"][1] < 0
    if args.dn:
        dn, dn_lines = pool_cohorts([[p] for p in args.dn])
        c = np.array([len(x["ref"]) for x in dn_lines], float)
        g = np.array([x["vn"] for x in dn_lines])
        d = bootstrap(sum(dn[base]["errors"][v] for v in VIEWS), sum(dn[args.name]["errors"][v] for v in VIEWS),
                      c, g, 10000, 0, len(VIEWS))
        res["devnative"] = d
        replaces = replaces and not d["ci95"][0] > 0
    res["replaces_base"] = bool(replaces)
    res["selected"] = args.name if replaces else base
    res["selected_blob"] = tv[args.name]["blob"] if replaces else sel["selected_blob"]
    res["selected_blob_sha256"] = tv[args.name]["blob_sha256"] if replaces else sel["selected_blob_sha256"]
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base")
    b.add_argument("--protocol", required=True)
    b.add_argument("--size", required=True, choices=("FL10", "H15", "W20"))
    b.add_argument("--tv", nargs="+", required=True)
    b.add_argument("--dn", nargs="*", default=[])
    b.add_argument("--dn-commit", default="")
    b.add_argument("--budget-dir", action="append", required=True)
    b.add_argument("--only", default="", help="restrict to these candidates (tests only)")
    b.add_argument("--realplus", default="", help="the size's realplus decision (vocr3_select.py realplus)")
    b.add_argument("--out", required=True)
    r = sub.add_parser("realplus")
    r.add_argument("--protocol", required=True)
    r.add_argument("--size", required=True, choices=("FL10", "H15", "W20"))
    r.add_argument("--tv", nargs="+", required=True)
    r.add_argument("--dn", nargs="*", default=[])
    r.add_argument("--out", required=True)
    v = sub.add_parser("variant")
    v.add_argument("--selection", required=True)
    v.add_argument("--name", required=True)
    v.add_argument("--tv", nargs="+", required=True)
    v.add_argument("--dn", nargs="*", default=[])
    v.add_argument("--out", required=True)
    args = ap.parse_args()
    {"base": cmd_base, "realplus": cmd_realplus, "variant": cmd_variant}[args.cmd](args)


if __name__ == "__main__":
    main()

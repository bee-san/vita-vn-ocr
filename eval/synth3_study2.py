#!/usr/bin/env python3
"""synth3 study 2's result: development comparisons and the frozen decision rule.

  synth3_study2.py merge  --scores A.json B.json ... --out merged.json
  synth3_study2.py result --protocol eval/synth3_study2_protocol.json --tv TV_SCORES.json \
                          [--dn DN_MERGED.json --dn-commit SHA] [--reported eval/synth3_study2_reported.json \
                          --reported-scores S.json ...] --out result.json

Scores come from `eval/synth3_ablation.py score` (C runtime W8I, views rv and rvn). `merge` joins
score files of different development line sets that hold the same models (devnative's subsets
and crops, one file per npz) into one file whose line ids carry their source file's name, so the
pooled set is compared like one cohort (VNs resampled as clusters).

`result` runs study 1's comparison statistics (`synth3_ablation.cmd_compare`: VN-cluster and line
bootstraps, per view, per seed, strata, per VN) on teacher_val and, when given, on devnative's
decision subsets, for every protocol comparison whose runs are all scored, and applies the
protocol's decision: C_mix is better than A_s2 iff teacher_val's VN-cluster 95% interval of
CER(C_mix) - CER(A_s2) lies entirely below 0, every same-seed difference is below 0, and (when
devnative is co-primary) devnative's VN-cluster interval also lies entirely below 0; A_s2 is better
under the mirrored rule; otherwise neither. `--reported` adds the protocol's reported-only
comparisons (pretrain-only, fine-tuning gain, step and learning curves) from their own scores.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth3_ablation as ABL  # noqa: E402


def merge(paths):
    """One score file from several score files over different lines and the same models."""
    out = {"kind": "synth3-dev-scores", "dev": {"merged": []}, "lines": [], "models": {}}
    names = None
    for p in paths:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        tag = Path(p).stem
        if names is None:
            names = sorted(s["models"])
            for n in names:
                m = s["models"][n]
                out["models"][n] = {k: m[k] for k in ("blob", "blob_sha256", "blob_bytes", "arena_bytes")}
                out["models"][n].update(pred={v: [] for v in m["pred"]}, errors={v: [] for v in m["errors"]})
        elif sorted(s["models"]) != names:
            raise SystemExit(f"{p}: holds other models than {paths[0]}")
        out["dev"]["merged"].append({"name": tag, **s["dev"]})
        out["lines"] += [{**x, "id": f"{tag}:{x['id']}"} for x in s["lines"]]
        for n in names:
            for kind in ("pred", "errors"):
                for v, vals in s["models"][n][kind].items():
                    out["models"][n][kind][v] += vals
    n_chars = sum(len(x["ref"]) for x in out["lines"])
    out["dev"].update(lines=len(out["lines"]), chars=n_chars, vns=len({x["vn"] for x in out["lines"]}))
    for m in out["models"].values():
        m["cer"] = {v: round(sum(e) / n_chars, 5) for v, e in m["errors"].items()}
    return out


def compare(score_paths, spec, tmp):
    """synth3_ablation's comparisons of `spec` (a protocol-format dict) over the given score files."""
    sp = Path(tmp) / "spec.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    out = Path(tmp) / "cmp.json"
    ABL.cmd_compare(type("A", (), {"scores": [str(p) for p in score_paths], "protocol": str(sp), "out": str(out)})())
    return json.loads(out.read_text(encoding="utf-8"))


def scored(paths):
    names = set()
    for p in paths:
        names |= set(json.loads(Path(p).read_text(encoding="utf-8"))["models"])
    return names


def side(c, sign):
    """Whether one comparison's VN-cluster interval and every seed lie strictly on one side of 0."""
    lo, hi = c["all_vn_cluster"]["ci95"]
    seeds = [x["delta_points"] for x in c["per_seed"].values()]
    if sign < 0:
        return hi < 0, all(d < 0 for d in seeds)
    return lo > 0, all(d > 0 for d in seeds)


def decide(tv, dn):
    """The protocol's rule for one comparison (A = runs_a = A_s2, B = runs_b = C_mix)."""
    out = {}
    for name, sign in (("c_better", -1), ("a_better", +1)):
        ci, seeds = side(tv, sign)
        ok = ci and seeds
        dn_ok = None
        if dn is not None:
            lo, hi = dn["all_vn_cluster"]["ci95"]
            dn_ok = hi < 0 if sign < 0 else lo > 0
            ok = ok and dn_ok
        out[name] = {"holds": bool(ok), "tv_ci_side": bool(ci), "tv_every_seed": bool(seeds),
                     "devnative_ci_side": dn_ok}
    better = "C_mix" if out["c_better"]["holds"] else "A_s2" if out["a_better"]["holds"] else None
    out["better"] = better
    out["recommended"] = better or "A_s2"
    out["reason"] = (f"{better} is better under the frozen rule" if better else
                     "no measured difference: synth2 (the released recipe) stays")
    return out


def cmd_merge(args):
    Path(args.out).write_text(json.dumps(merge(args.scores), ensure_ascii=False), encoding="utf-8")


def cmd_result(args):
    import tempfile
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    base = {k: proto[k] for k in ("views", "bootstrap", "decision", "arms")}
    have = scored(args.tv)
    cmps = {k: v for k, v in proto["comparisons"].items()
            if all(r in have for r in proto["arms"][v[0]] + proto["arms"][v[1]])}
    if not cmps:
        raise SystemExit("no protocol comparison has all its runs scored")
    res = {"kind": "synth3-study2-result", "protocol_sha256": ABL.sha256(args.protocol),
           "computed": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "devnative": {"co_primary": bool(args.dn), "commit": args.dn_commit or None}, "comparisons": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tv = compare(args.tv, {**base, "comparisons": cmps}, tmp)
        res["teacher_val"] = {"dev": tv["dev"], "models": tv["models"]}
        dn = compare([args.dn], {**base, "comparisons": cmps}, tmp) if args.dn else None
        if dn:
            res["devnative_decision_set"] = {"dev": dn["dev"], "models": dn["models"]}
        for name in cmps:
            c = {"teacher_val": tv["comparisons"][name]}
            if dn:
                c["devnative"] = dn["comparisons"][name]
            c["decision"] = decide(c["teacher_val"], c.get("devnative"))
            for k in ("teacher_val", "devnative"):
                if k in c:
                    c[k].pop("decision", None)
            res["comparisons"][name] = c
            t = c["teacher_val"]["all_vn_cluster"]
            print(f"{name}: teacher_val {t['delta_points']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]"
                  + (f"; devnative {c['devnative']['all_vn_cluster']['delta_points']:+.3f} "
                     f"{c['devnative']['all_vn_cluster']['ci95']}" if dn else "")
                  + f" -> {c['decision']['reason']}", flush=True)
        if args.reported:
            rep = json.loads(Path(args.reported).read_text(encoding="utf-8"))
            have_r = scored(args.reported_scores)
            rc = {k: v for k, v in rep["comparisons"].items()
                  if all(r in have_r for r in rep["arms"][v[0]] + rep["arms"][v[1]])}
            if rc:
                r = compare(args.reported_scores, {**base, "arms": rep["arms"], "comparisons": rc}, tmp)
                for c in r["comparisons"].values():
                    c.pop("decision", None)
                res["reported"] = {"spec_sha256": ABL.sha256(args.reported), "models": r["models"],
                                   "comparisons": r["comparisons"]}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--scores", nargs="+", required=True)
    m.add_argument("--out", required=True)
    r = sub.add_parser("result")
    r.add_argument("--protocol", required=True)
    r.add_argument("--tv", nargs="+", required=True, help="teacher_val score files (same lines)")
    r.add_argument("--dn", default="", help="devnative's merged decision-set scores (only when co-primary)")
    r.add_argument("--dn-commit", default="")
    r.add_argument("--reported", default="", help="the reported-only comparison spec")
    r.add_argument("--reported-scores", nargs="*", default=[])
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"merge": cmd_merge, "result": cmd_result}[args.cmd](args)


if __name__ == "__main__":
    main()

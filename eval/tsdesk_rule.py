#!/usr/bin/env python3
"""textspot-desktop's decision rule as code (issue #173, eval/textspot_desktop_protocol.json "rule", amendment 1).

  tsdesk_rule.py decide --res DIR --arm ARM --seeds 1,2,3 [--policies fit960,long960] --timing T.json
                        --parity P.json --out DECISION.json

Reads (DIR holds eval/desktop.py read results and eval/tsdesk_eval.py edges results, named as the screening and decision
scripts write them):
  dev_{ARM}_s{N}_{POL}_L_fast.json, dev_T8_L_fast.json          R1 (dev-v2: CER, and each view's paired interval)
  dense_dev_{ARM}_s{N}_{POL}_L_fast.json                        R2 (CER <= 2.00%)
  edges_hval_{ARM}_s{N}_{POL}.json                              R3 (per edge median |e| <= 1, p90 <= 2; the cost)
  hval_{ARM}_s{N}_{POL}_L_fast.json                             R4 (PCSG00854's junk boxes <= 5)
  T.json: {"POL": {"vita": {"p50": ms}, "pc1080": {"p50": ms}}}, the C AVX2 build at 1 thread   R5
  P.json: {"POL": {"crops_equal_share": x, "cer_onnx": a, "cer_c": b}} on dev-v2                R6
The representative seed is the median of the seeds by dev-v2 CER at fit:960 (ties: the lower seed). Each policy of
the representative is a candidate; the pick is the cheapest qualifier (C 1-thread p50, Vita + pc1080); a costlier
one replaces it only when it is resolved better on dev-v2 (the paired interval entirely below 0).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "tools", ROOT / "synth", ROOT / "train", ROOT / "teacher", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

THRESH = {"dense_dev_cer": 2.00, "edge_median": 1.0, "edge_p90": 2.0, "junk_854": 5, "ms_960": 3.0, "ms_1080": 8.0,
          "parity_share": 0.99, "parity_cer": 0.05}
EDGES = ("left", "top", "right", "bottom")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cer(doc, drop=None):
    U = [u for u in doc["units"] if drop is None or u["cluster"] != drop]
    e, c = sum(u["errors"] for u in U), sum(u["chars"] for u in U)
    return 100.0 * e / max(1, c)


def junk(doc, cluster):
    return sum(u["junk"] for u in doc["units"] if u["cluster"] == cluster)


def not_above_zero(ci):
    """A paired interval not entirely above 0 (None, with too few clusters, counts as not resolved)."""
    return ci is None or ci[0] <= 0.0


def entirely_below_zero(ci):
    return ci is not None and ci[1] < 0.0


def candidate(res, arm, seed, pol, t8, timing, parity):
    import desktop as D
    tag = f"{arm}_s{seed}_{pol}"
    dev = load(res / f"dev_{tag}_L_fast.json")
    cmp = D.compare_docs(dev, t8)
    out = {"tag": tag, "policy": pol}
    c_dev, c_t8 = cer(dev), cer(t8)
    views = {v: cmp.get(f"dev:{v}") for v in ("vita", "psp")}
    out["R1"] = {"cer": round(c_dev, 4), "t8": round(c_t8, 4), "point_ok": c_dev <= c_t8 + 1e-12,
                 "views": {v: {"delta": c["delta"], "ci95": c["ci95"], "ok": not_above_zero(c["ci95"])}
                           for v, c in views.items() if c is not None},
                 "all_vs_t8": {"delta": cmp["dev:all"]["delta"], "ci95": cmp["dev:all"]["ci95"]}}
    out["R1"]["pass"] = out["R1"]["point_ok"] and all(v["ok"] for v in out["R1"]["views"].values()) and \
        len(out["R1"]["views"]) == 2
    dd = load(res / f"dense_dev_{tag}_L_fast.json")
    out["R2"] = {"cer": round(cer(dd), 4), "pass": cer(dd) <= THRESH["dense_dev_cer"]}
    ed = load(res / f"edges_hval_{tag}.json")
    e_all = ed["edges"]["all"]
    edges_ok = all(e_all[k]["median_abs"] <= THRESH["edge_median"] and e_all[k]["p90_abs"] <= THRESH["edge_p90"]
                   for k in EDGES)
    cost = ed["tightness_cost"]["all"]
    out["R3"] = {"edges": {k: {"median_abs": e_all[k]["median_abs"], "p90_abs": e_all[k]["p90_abs"]} for k in EDGES},
                 "matched_share": ed["references"]["matched_share"], "cost": {"det_cer": cost["arm_cer"],
                                                                             "exact_cer": cost["base_cer"],
                                                                             "delta": cost["delta"],
                                                                             "ci95": cost["ci95"]},
                 "pass": edges_ok and not_above_zero(cost["ci95"])}
    hv = load(res / f"hval_{tag}_L_fast.json")
    out["R4"] = {"junk_PCSG00854": junk(hv, "PCSG00854"), "pass": junk(hv, "PCSG00854") <= THRESH["junk_854"]}
    tm = (timing or {}).get(pol)
    if tm:
        out["R5"] = {"ms_960": tm["vita"]["p50"], "ms_1080": tm["pc1080"]["p50"],
                     "pass": tm["vita"]["p50"] <= THRESH["ms_960"] and tm["pc1080"]["p50"] <= THRESH["ms_1080"]}
    else:
        out["R5"] = {"pass": False, "why": "no C timing for this policy"}
    pr = (parity or {}).get(pol)
    if pr:
        out["R6"] = {**pr, "pass": pr["crops_equal_share"] >= THRESH["parity_share"] and
                     abs(pr["cer_c"] - pr["cer_onnx"]) <= THRESH["parity_cer"]}
    else:
        out["R6"] = {"pass": False, "why": "no C parity record for this policy"}
    out["qualifies"] = all(out[k]["pass"] for k in ("R1", "R2", "R3", "R4", "R5", "R6"))
    out["cost_ms"] = (tm["vita"]["p50"] + tm["pc1080"]["p50"]) if tm else None
    out["_dev"] = dev
    return out


def cmd_decide(a):
    import desktop as D
    res = Path(a.res)
    seeds = [int(s) for s in a.seeds.split(",")]
    pols = a.policies.split(",")
    t8 = load(res / "dev_T8_L_fast.json")
    by_seed = {s: cer(load(res / f"dev_{a.arm}_s{s}_fit960_L_fast.json")) for s in seeds}
    order = sorted(seeds, key=lambda s: (by_seed[s], s))
    rep = order[len(order) // 2]
    timing = load(a.timing) if a.timing and Path(a.timing).exists() else None
    parity = load(a.parity) if a.parity and Path(a.parity).exists() else None
    cands = [candidate(res, a.arm, rep, pol, t8, timing, parity) for pol in pols]
    quals = [c for c in cands if c["qualifies"]]
    pick = None
    if quals:
        quals.sort(key=lambda c: c["cost_ms"])
        pick = quals[0]
        for c in quals[1:]:
            cmp = D.compare_docs(c["_dev"], pick["_dev"])["dev:all"]
            if entirely_below_zero(cmp["ci95"]):
                pick = c
    for c in cands:
        c.pop("_dev", None)
    doc = {"kind": "textspot-desktop-decision", "protocol": "eval/textspot_desktop_protocol.json",
           "thresholds": THRESH, "arm": a.arm,
           "seeds_dev_cer_fit960": {str(s): round(v, 4) for s, v in by_seed.items()},
           "representative_seed": rep, "candidates": cands, "pick": pick["tag"] if pick else None,
           "verdict": "qualifies" if pick else "no qualifier"}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"representative": rep, "pick": doc["pick"],
                      "candidates": {c["tag"]: {k: c[k]["pass"] for k in ("R1", "R2", "R3", "R4", "R5", "R6")}
                                     for c in cands}}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--res", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--policies", default="fit960,long960")
    p.add_argument("--timing")
    p.add_argument("--parity")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()

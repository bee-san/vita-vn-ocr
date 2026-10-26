#!/usr/bin/env python3
"""textspot6's selection (issue #128, docs/textspot6.md "The rule"; eval/textspot6_protocol.json "rule", amendment 2),
run once, after every stage's runs are scored through eval/pagepath.py.

  textspot6_select.py --table TABLE.json --out eval/textspot6_dev_<date>.json

TABLE.json (written by the scoring step):
  {"baseline": {"dev": RESULT.json, "hval": RESULT.json},              eval/pagepath.py run --finder rubi
   "seen_dev_clusters": ["v12188"],                                     left out of the headline
   "candidates": [{"arm", "seed", "decode": "crop|crop+refine|det|det+refine", "dev": RESULT.json,
                   "hval": RESULT.json, "vtsp_bytes", "est_ms_333", "est_ms_444", "arena", "c_equal": bool}]}

Each RESULT.json is eval/pagepath.py's (its "units": one row per unit with uid, cluster, platform, errors, chars).

The rule, as registered:
  representative  per (arm, decode): the median seed by unseen-dev page-path dialogue CER (sorted by that CER, then
                  hval CER, then seed; index n // 2, the worse middle one when n is even); never the best
  eligible        VTSP bytes <= 512,000, est. ms @333 <= 300, arena <= 262,144, C == twin on every dev frame
  qualifies       unseen dev (the seen clusters left out): candidate - GRAD + refine with a 95% VN-cluster interval
                  entirely below 0; hval: the title-cluster interval's lower bound <= 0 (not resolved above 0)
  cost order      VTSP bytes, est. ms @333, arena, decode (crop, crop+refine, det, det+refine), name
  pick            the cheapest qualifier; a costlier qualifier replaces it when, against it, unseen dev is resolved
                  below 0 and hval is not resolved above 0; repeated until none does
Intervals: eval/vocr3_report.cluster_bootstrap (10,000 resamples, seed 0, sorted cluster ids, cluster sums).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import vocr3_report as VR  # noqa: E402

DECODES = ("crop", "crop+refine", "det", "det+refine")
VTSP_MAX, MS_MAX, ARENA_MAX = 512000, 300.0, 262144


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rows(path, drop_clusters=()):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {u["uid"]: u for u in doc["units"] if u["cluster"] not in drop_clusters}


def cer(rows):
    e = sum(r["errors"] for r in rows.values())
    n = sum(r["chars"] for r in rows.values())
    return round(100 * e / n, 4) if n else None


def diff(a, b):
    """CER(a) - CER(b) in points over the same units, with the cluster bootstrap."""
    if set(a) != set(b):
        raise SystemExit("paired comparison needs the same units")
    if any(a[k]["chars"] != b[k]["chars"] for k in a):
        raise SystemExit("paired comparison needs the same references")
    pa = {k: (a[k]["errors"], a[k]["chars"]) for k in a}
    pb = {k: (b[k]["errors"], b[k]["chars"]) for k in a}
    cl = {k: a[k]["cluster"] for k in a}
    n = sum(v[1] for v in pa.values())
    d = 100 * (sum(v[0] for v in pa.values()) - sum(v[0] for v in pb.values())) / n if n else None
    out = VR.cluster_bootstrap(pa, pb, cl)
    return {"delta": round(d, 4) if d is not None else None, **out}


def resolved_below(c):
    return c.get("ci95") is not None and c["ci95"][1] < 0


def resolved_above(c):
    return c.get("ci95") is not None and c["ci95"][0] > 0


def per_platform(rows):
    out = {}
    for p in sorted({r["platform"] for r in rows.values()}):
        out[p] = cer({k: r for k, r in rows.items() if r["platform"] == p})
    return out


def cost_key(c):
    return (c["vtsp_bytes"], c["est_ms_333"], c["arena"], DECODES.index(c["decode"]), c["arm"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tab = json.loads(Path(a.table).read_text(encoding="utf-8"))
    seen = tuple(tab.get("seen_dev_clusters") or ())
    base_dev = load_rows(tab["baseline"]["dev"], seen)
    base_dev_all = load_rows(tab["baseline"]["dev"])
    base_hval = load_rows(tab["baseline"]["hval"])
    seeds = []
    for c in tab["candidates"]:
        dev, hval = load_rows(c["dev"], seen), load_rows(c["hval"])
        rec = dict(c, dev_cer=cer(dev), dev_cer_all=cer(load_rows(c["dev"])), hval_cer=cer(hval),
                   dev_platform=per_platform(dev), hval_platform=per_platform(hval),
                   dev_sha256=sha256_file(c["dev"]), hval_sha256=sha256_file(c["hval"]))
        rec["eligible"] = bool(c["vtsp_bytes"] <= VTSP_MAX and c["est_ms_333"] <= MS_MAX and c["arena"] <= ARENA_MAX
                               and c["c_equal"])
        seeds.append(rec)
    groups = {}
    for r in seeds:
        groups.setdefault((r["arm"], r["decode"]), []).append(r)
    cands = []
    for (arm, dec), rs in sorted(groups.items()):
        order = sorted(rs, key=lambda r: (r["dev_cer"], r["hval_cer"], r["seed"]))
        rep = order[len(order) // 2]
        dev, hval = load_rows(rep["dev"], seen), load_rows(rep["hval"])
        cmp_dev = diff(dev, base_dev)
        cmp_dev_all = diff(load_rows(rep["dev"]), base_dev_all)
        cmp_hval = diff(hval, base_hval)
        qual = rep["eligible"] and resolved_below(cmp_dev) and not resolved_above(cmp_hval)
        cands.append({"arm": arm, "decode": dec, "representative_seed": rep["seed"],
                      "seeds": [{"seed": r["seed"], "dev_cer": r["dev_cer"], "hval_cer": r["hval_cer"]} for r in order],
                      "vtsp_bytes": rep["vtsp_bytes"], "est_ms_333": rep["est_ms_333"],
                      "est_ms_444": rep.get("est_ms_444"), "arena": rep["arena"], "c_equal": rep["c_equal"],
                      "eligible": rep["eligible"], "dev_cer": rep["dev_cer"], "dev_cer_all": rep["dev_cer_all"],
                      "hval_cer": rep["hval_cer"], "dev_platform": rep["dev_platform"],
                      "hval_platform": rep["hval_platform"], "vs_grad_refine": {"dev": cmp_dev, "dev_all": cmp_dev_all,
                                                                                "hval": cmp_hval},
                      "qualifies": qual, "_dev": rep["dev"], "_hval": rep["hval"]})
    quals = sorted([c for c in cands if c["qualifies"]], key=cost_key)
    chain = []
    pick = quals[0] if quals else None
    if pick:
        chain.append({"step": "cheapest qualifier", "pick": f"{pick['arm']}|{pick['decode']}"})
        while True:
            pd, ph = load_rows(pick["_dev"], seen), load_rows(pick["_hval"])
            nxt = None
            for c in quals:
                if cost_key(c) <= cost_key(pick):
                    continue
                cd = diff(load_rows(c["_dev"], seen), pd)
                ch = diff(load_rows(c["_hval"]), ph)
                if resolved_below(cd) and not resolved_above(ch):
                    nxt = (c, cd, ch)
                    break
            if nxt is None:
                break
            chain.append({"step": "replaced", "pick": f"{nxt[0]['arm']}|{nxt[0]['decode']}", "dev": nxt[1],
                          "hval": nxt[2]})
            pick = nxt[0]
    for c in cands:
        c.pop("_dev")
        c.pop("_hval")
    out = {"kind": "textspot6-selection", "computed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "table": str(a.table), "table_sha256": sha256_file(a.table),
           "protocol_sha256": sha256_file(ROOT / "eval" / "textspot6_protocol.json"),
           "seen_dev_clusters": list(seen),
           "baseline": {"dev_cer": cer(base_dev), "dev_cer_all": cer(base_dev_all), "hval_cer": cer(base_hval),
                        "dev_platform": per_platform(base_dev), "hval_platform": per_platform(base_hval),
                        "dev_sha256": sha256_file(tab["baseline"]["dev"]),
                        "hval_sha256": sha256_file(tab["baseline"]["hval"])},
           "seeds": [{k: v for k, v in r.items() if k not in ("dev", "hval")} for r in seeds],
           "candidates": cands, "qualifiers": [f"{c['arm']}|{c['decode']}" for c in quals], "chain": chain,
           "pick": None if pick is None else {"arm": pick["arm"], "decode": pick["decode"],
                                              "seed": pick["representative_seed"]},
           "verdict": "positive" if pick else "negative"}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"baseline GRAD + refine: dev {out['baseline']['dev_cer']}% hval {out['baseline']['hval_cer']}%")
    for c in sorted(cands, key=cost_key):
        v = c["vs_grad_refine"]
        print(f"{c['arm']:>20s} {c['decode']:>11s} s{c['representative_seed']} {'E' if c['eligible'] else '-'} "
              f"dev {c['dev_cer']}% ({v['dev']['delta']:+.3f} {v['dev'].get('ci95')}) hval {c['hval_cer']}% "
              f"({v['hval']['delta']:+.3f} {v['hval'].get('ci95')}) {'QUALIFIES' if c['qualifies'] else ''}")
    print("PICK", out["pick"], "verdict", out["verdict"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The desktop decision (issue #169): eval/desktop_protocol.json's rule, with its amendments, as code over the
eval/desktop.py result records and the eval/desktop_time.py timing records. It reads records only (no frame).

  desktop_rule.py --res DIR --timing T4.json --candidates C.json --out DECISION.json [--pc-res DIR]

C.json lists the candidates: {"reference": {"dev": NAME, "hval": NAME, "pc": NAME, "confirm": NAME,
"timing": PIPELINE}, "seeds": {SEED: {POLICY: {"dev": NAME, "hval": NAME, "pc": NAME, "confirm": NAME,
"timing": PIPELINE}}}} where NAME is a result file's stem under --res (e.g. dev_X_fit640_L_fast) and PIPELINE a
timing pipeline name (desktop_time.py run's "spec|mode"); an entry may list several timing pipelines (the
detector's intra-op variants, amendment 1: "#d1", "#d2", "#d4"), and each view takes the fastest.

The rule:
- representative seed: the median of the seeds by dev-v2 CER at fit:960 (an odd number of seeds);
- accuracy (dev-v2): CER(C) <= 1.20 CER(R); the paired 95% VN-cluster interval of CER(C) - CER(R) not entirely above
  0; the Vita and the PSP view each CER(C) <= 1.5 CER(R) of that view;
- guards: hval (title clusters) and confirm-v2 (VN clusters): C - R not resolved above 0 (the interval's lower bound
  > 0 fails); a missing guard record leaves the candidate undecided;
- the PC decision set (amendment 2): CER(C) <= 1.20 CER(R) and the interval not entirely above 0;
- speed: at the 4-thread budget, each view's p50 ms per page (the fastest of a pipeline's variants) of C <= 1/4 of
  R's, on the Vita and the PSP timing pages separately;
- pick: among the qualifying policies the cheapest (Vita + PSP p50 at 4 threads); a costlier one replaces it only when
  resolved better on dev-v2 (its interval against the cheapest entirely below 0).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import desktop as D  # noqa: E402


def load(res_dir, name):
    if not name:
        return None
    p = Path(res_dir) / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cer_of(doc, view=None):
    e = c = 0
    for u in doc["units"]:
        if view is None or u["view"] == view:
            e += u["errors"]
            c += u["chars"]
    return 100.0 * e / max(1, c)


def paired(a, b, group):
    """compare_docs(arm a, base b)[group]."""
    return D.compare_docs(a, b).get(group)


def speed_of(timing, names, view):
    names = [names] if isinstance(names, str) else list(names)
    vals = [timing["results"][n][view]["p50"] for n in names if n in timing["results"] and view in timing["results"][n]]
    return min(vals) if vals else None


def judge(c, r, timing, res, pc_res):
    """The clauses for one candidate entry c against the reference r: {clause: {..., "ok": bool | None}}."""
    out = {}
    dev_c, dev_r = load(res, c.get("dev")), load(res, r.get("dev"))
    if dev_c is None or dev_r is None:
        return {"error": "missing dev-v2 records"}
    a, b = cer_of(dev_c), cer_of(dev_r)
    cmp_all = paired(dev_c, dev_r, "dev:all")
    views = {}
    for v in ("vita", "psp"):
        cv, rv = cer_of(dev_c, v), cer_of(dev_r, v)
        views[v] = {"cer": round(cv, 4), "ref": round(rv, 4), "limit": round(1.5 * rv, 4), "ok": cv <= 1.5 * rv}
    ci = cmp_all["ci95"]
    out["dev"] = {"cer": round(a, 4), "ref": round(b, 4), "limit": round(1.2 * b, 4), "delta": cmp_all["delta"],
                  "ci95": ci, "views": views,
                  "ok": a <= 1.2 * b and not (ci and ci[0] > 0) and all(x["ok"] for x in views.values())}
    for name, docs, grp in (("hval", (load(res, c.get("hval")), load(res, r.get("hval"))), "hval:all"),
                            ("confirm", (load(res, c.get("confirm")), load(res, r.get("confirm"))), "confirm:all")):
        if docs[0] is None or docs[1] is None:
            out[name] = {"ok": None, "note": "not read"}
            continue
        cm = paired(docs[0], docs[1], grp)
        out[name] = {"cer": cm["arm_cer"], "ref": cm["base_cer"], "delta": cm["delta"], "ci95": cm["ci95"],
                     "ok": not (cm["ci95"] and cm["ci95"][0] > 0)}
    pc_c, pc_r = load(pc_res or res, c.get("pc")), load(pc_res or res, r.get("pc"))
    if pc_c is None or pc_r is None:
        out["pc"] = {"ok": None, "note": "not read"}
    else:
        grp = f"{pc_c['units'][0]['set']}:all"
        cm = paired(pc_c, pc_r, grp)
        out["pc"] = {"cer": cm["arm_cer"], "ref": cm["base_cer"], "limit": round(1.2 * cm["base_cer"], 4),
                     "delta": cm["delta"], "ci95": cm["ci95"],
                     "ok": cm["arm_cer"] <= 1.2 * cm["base_cer"] and not (cm["ci95"] and cm["ci95"][0] > 0)}
    sp = {}
    for v in ("vita", "psp"):
        tc, tr = speed_of(timing, c.get("timing", []), v), speed_of(timing, r.get("timing", []), v)
        sp[v] = {"p50": tc, "ref_p50": tr, "limit": round(tr / 4, 3) if tr else None,
                 "ok": None if tc is None or tr is None else tc <= tr / 4}
    out["speed"] = {"views": sp, "cost": (sp["vita"]["p50"] or 0) + (sp["psp"]["p50"] or 0),
                    "ok": None if any(x["ok"] is None for x in sp.values()) else all(x["ok"] for x in sp.values())}
    oks = [out[k]["ok"] for k in ("dev", "hval", "confirm", "pc", "speed")]
    out["qualifies"] = None if any(o is None for o in oks) else all(oks)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True)
    ap.add_argument("--pc-res")
    ap.add_argument("--timing", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cand = json.loads(Path(a.candidates).read_text(encoding="utf-8"))
    timing = json.loads(Path(a.timing).read_text(encoding="utf-8"))
    if timing.get("threads") != 4:
        raise SystemExit("the speed clause reads the 4-thread run")
    r = cand["reference"]
    seeds = cand["seeds"]
    by_seed = {}
    for s, pols in seeds.items():
        doc = load(a.res, pols["fit:960"]["dev"])
        by_seed[s] = cer_of(doc) if doc else None
    if any(v is None for v in by_seed.values()) or len(by_seed) % 2 == 0:
        raise SystemExit(f"the representative needs an odd number of seeds with a dev-v2 record at fit:960: {by_seed}")
    order = sorted(by_seed, key=lambda s: (by_seed[s], s))
    rep = order[len(order) // 2]
    results = {pol: judge(c, r, timing, a.res, a.pc_res) for pol, c in seeds[rep].items()}
    qual = [p for p, v in results.items() if v.get("qualifies")]
    pick = None
    if qual:
        qual.sort(key=lambda p: results[p]["speed"]["cost"])
        pick = qual[0]
        for p in qual[1:]:
            cm = paired(load(a.res, seeds[rep][p]["dev"]), load(a.res, seeds[rep][pick]["dev"]), "dev:all")
            if cm["ci95"] and cm["ci95"][1] < 0:
                pick = p
    undecided = [p for p, v in results.items() if v.get("qualifies") is None]
    doc = {"kind": "vocr-desktop-decision", "protocol_sha256": D.sha256_file(ROOT / "eval" / "desktop_protocol.json"),
           "seeds_dev_cer_fit960": {s: round(v, 4) for s, v in by_seed.items()}, "representative": rep,
           "results": results, "qualifying": qual, "undecided": undecided, "pick": pick,
           "verdict": "pick" if pick else ("undecided" if undecided else "no qualifier")}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("representative", "qualifying", "undecided", "pick", "verdict")}))


if __name__ == "__main__":
    main()

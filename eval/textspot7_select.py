#!/usr/bin/env python3
"""textspot7's selection (issue #128, eval/textspot7_protocol.json "rule"): textspot6's rule unchanged, with the
fallback's cost counted and the confirmation condition added. Two phases, each run once:

  textspot7_select.py --table TABLE.json --phase dev --out eval/textspot7_dev_<date>.json
      every candidate's eligibility and its dev / hval comparisons; the qualifiers (beat GRAD + refine on unseen dev,
      hval not resolved above 0). Written and committed before the confirmation set is read
  textspot7_select.py --table TABLE.json --phase confirm --dev-record eval/textspot7_dev_<date>.json --out ...
      the confirmation condition for every qualifier (not resolved worse than rubi,fix=1 on the confirmation set),
      the pick (the cheapest qualifier that meets it; a costlier one replaces it when resolved better on unseen dev
      and not resolved worse on hval), Rubi's default (against rubi,fix=1: unseen dev resolved below 0, hval not above
      0), and the VOCR6 guard when the table has one

TABLE.json:
  {"baseline": {"dev": RESULT, "hval": RESULT},                              GRAD + refine (eval/pagepath.py --finder rubi)
   "rubi_next": {"name": "rubi,fix=1", "dev": RESULT, "hval": RESULT, "confirm": RESULT (phase confirm)},
   "seen_dev_clusters": ["v12188"],
   "candidates": [{"name", "arm", "run", "decode": "D|F|F+D", "dev": RESULT, "hval": RESULT, "confirm": RESULT (phase
                   confirm, qualifiers only), "vtsp_bytes", "est_ms_333", "est_ms_444", "est_ms_parts": {...}, "arena",
                   "c_equal": bool, "c_equal_parts": {...}, "params": {...}}],
   "vocr6_guard": {"name", "fix1": {"dev", "hval"}, "pick": {"dev", "hval"}} (optional)}
Each RESULT is an eval/pagepath.py result (its "units" rows: uid, cluster, platform, errors, chars).
Intervals: eval/vocr3_report.cluster_bootstrap (10,000 resamples, seed 0) through eval/textspot6_select.diff.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))
import textspot6_select as S6  # noqa: E402

DECODE_ORDER = ("D", "F", "F+D")
VTSP_MAX, MS_MAX, ARENA_MAX = 512000, 300.0, 262144


def cost_key(c):
    return (c["vtsp_bytes"], c["est_ms_333"], c["arena"], DECODE_ORDER.index(c["decode"]), c["arm"])


def candidate_record(c, base_dev, base_dev_all, base_hval, nxt, seen):
    dev, dev_all, hval = S6.load_rows(c["dev"], seen), S6.load_rows(c["dev"]), S6.load_rows(c["hval"])
    rec = {k: c.get(k) for k in ("name", "arm", "run", "decode", "params", "vtsp_bytes", "est_ms_333", "est_ms_444",
                                 "est_ms_parts", "arena", "c_equal", "c_equal_parts")}
    rec["eligible"] = bool(c["vtsp_bytes"] <= VTSP_MAX and c["est_ms_333"] <= MS_MAX and c["arena"] <= ARENA_MAX
                           and c["c_equal"])
    rec.update(dev_cer=S6.cer(dev), dev_cer_all=S6.cer(dev_all), hval_cer=S6.cer(hval),
               dev_platform=S6.per_platform(dev), hval_platform=S6.per_platform(hval),
               dev_sha256=S6.sha256_file(c["dev"]), hval_sha256=S6.sha256_file(c["hval"]))
    rec["vs_grad_refine"] = {"dev": S6.diff(dev, base_dev), "dev_all": S6.diff(dev_all, base_dev_all),
                             "hval": S6.diff(hval, base_hval)}
    if nxt:
        rec["vs_rubi_next"] = {"dev": S6.diff(dev, nxt["dev"]), "hval": S6.diff(hval, nxt["hval"])}
    v = rec["vs_grad_refine"]
    rec["qualifies"] = rec["eligible"] and S6.resolved_below(v["dev"]) and not S6.resolved_above(v["hval"])
    return rec


def phase_dev(tab):
    seen = tuple(tab.get("seen_dev_clusters") or ())
    base_dev = S6.load_rows(tab["baseline"]["dev"], seen)
    base_dev_all = S6.load_rows(tab["baseline"]["dev"])
    base_hval = S6.load_rows(tab["baseline"]["hval"])
    nxt = None
    if tab.get("rubi_next"):
        rn = tab["rubi_next"]
        nxt = {"dev": S6.load_rows(rn["dev"], seen), "hval": S6.load_rows(rn["hval"])}
    cands = [candidate_record(c, base_dev, base_dev_all, base_hval, nxt, seen) for c in tab["candidates"]]
    out = {"kind": "textspot7-selection-dev", "schema": 1, "issue": 128,
           "protocol": "eval/textspot7_protocol.json", "protocol_sha256": S6.sha256_file(
               ROOT / "eval" / "textspot7_protocol.json"),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seen_dev_clusters": list(seen),
           "baseline": {"name": "GRAD + refine (rubi)", "dev_cer": S6.cer(base_dev), "dev_cer_all": S6.cer(
               base_dev_all), "hval_cer": S6.cer(base_hval), "dev_sha256": S6.sha256_file(tab["baseline"]["dev"]),
               "hval_sha256": S6.sha256_file(tab["baseline"]["hval"])}, "candidates": cands,
           "qualifiers": [c["name"] for c in sorted([c for c in cands if c["qualifies"]], key=cost_key)]}
    if nxt:
        out["rubi_next"] = {"name": tab["rubi_next"].get("name"), "dev_cer": S6.cer(nxt["dev"]),
                            "hval_cer": S6.cer(nxt["hval"]),
                            "vs_grad_refine": {"dev": S6.diff(nxt["dev"], base_dev),
                                               "hval": S6.diff(nxt["hval"], base_hval)}}
    out["verdict_dev"] = ("qualifiers: " + ", ".join(out["qualifiers"])) if out["qualifiers"] else \
        "no qualifier: negative verdict (no confirmation read needed)"
    return out


def phase_confirm(tab, dev_rec):
    seen = tuple(tab.get("seen_dev_clusters") or ())
    by_name = {c["name"]: c for c in tab["candidates"]}
    rec = {"kind": "textspot7-selection-confirm", "schema": 1, "issue": 128,
           "dev_record_sha256": None, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    quals = [c for c in dev_rec["candidates"] if c["qualifies"]]
    fix_c = S6.load_rows(tab["rubi_next"]["confirm"])
    rec["rubi_next_confirm_cer"] = S6.cer(fix_c)
    passed = []
    for c in quals:
        conf = S6.load_rows(by_name[c["name"]]["confirm"])
        d = S6.diff(conf, fix_c)
        c = dict(c, confirm_cer=S6.cer(conf), vs_rubi_next_confirm=d, confirm_sha256=S6.sha256_file(
            by_name[c["name"]]["confirm"]), confirmed=not S6.resolved_above(d))
        passed.append(c)
    rec["qualifiers"] = passed
    ok = sorted([c for c in passed if c["confirmed"]], key=cost_key)
    chain, pick = [], (ok[0] if ok else None)
    if pick:
        chain.append({"step": "cheapest confirmed qualifier", "pick": pick["name"]})
        while True:
            pd = S6.load_rows(by_name[pick["name"]]["dev"], seen)
            ph = S6.load_rows(by_name[pick["name"]]["hval"])
            nxt = None
            for c in ok:
                if cost_key(c) <= cost_key(pick):
                    continue
                cd = S6.diff(S6.load_rows(by_name[c["name"]]["dev"], seen), pd)
                ch = S6.diff(S6.load_rows(by_name[c["name"]]["hval"]), ph)
                if S6.resolved_below(cd) and not S6.resolved_above(ch):
                    nxt = (c, cd, ch)
                    break
            if nxt is None:
                break
            chain.append({"step": "replaced", "pick": nxt[0]["name"], "dev": nxt[1], "hval": nxt[2]})
            pick = nxt[0]
    rec["chain"] = chain
    rec["pick"] = pick["name"] if pick else None
    if pick:
        v = pick["vs_rubi_next"]
        default = S6.resolved_below(v["dev"]) and not S6.resolved_above(v["hval"])
        g = tab.get("vocr6_guard")
        guard = None
        if g:
            gd = S6.diff(S6.load_rows(g["pick"]["dev"], seen), S6.load_rows(g["fix1"]["dev"], seen))
            gh = S6.diff(S6.load_rows(g["pick"]["hval"]), S6.load_rows(g["fix1"]["hval"]))
            guard = {"name": g.get("name"), "dev": gd, "hval": gh,
                     "ok": not S6.resolved_above(gd) and not S6.resolved_above(gh)}
            if not guard["ok"]:
                rec["pick"] = None
        rec["vocr6_guard"] = guard if g else "moot: no VOCR6 release at the decision"
        rec["rubi_default"] = {"against": "rubi,fix=1", "finder": "spotter" if (default and rec["pick"]) else "grad",
                               "dev": v["dev"], "hval": v["hval"]}
    rec["verdict"] = (f"pick {rec['pick']}" if rec["pick"] else
                      "negative: no qualifier meets the confirmation condition" if quals else "negative: no qualifier")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--phase", choices=("dev", "confirm"), required=True)
    ap.add_argument("--dev-record")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tab = json.loads(Path(a.table).read_text(encoding="utf-8"))
    if a.phase == "dev":
        rec = phase_dev(tab)
    else:
        dev_rec = json.loads(Path(a.dev_record).read_text(encoding="utf-8"))
        rec = phase_confirm(tab, dev_rec)
        rec["dev_record_sha256"] = S6.sha256_file(a.dev_record)
    Path(a.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{a.phase}: {rec.get('verdict_dev') or rec.get('verdict')} -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""textspot8's selection table (eval/textspot8_protocol.json "rule"): eval/textspot6_select.py's TABLE.json for
textspot8's arms, so the rule runs as textspot6's code (eval/textspot6_select.py, unchanged).

  textspot8_table.py --eval $A/vocr6-spotter2/eval --runs $A/vocr6-spotter2/runs2 --base-eval $A/vocr6-spotter/eval \
      --out TABLE.json [--allow-partial]

Candidates: amendment 1's t8_t64n8c16, seeds 1-3, in textspot6's four decodes, with textspot6's cost rows
(eval/textspot6_table.candidate_rows: the A9 rows of record of the shape, the refine decodes' extra time and arena,
the parity check), plus amendment 4's ctx (the crop row; parity on the grown regions) and amendment 3's det+D (the
det row plus D's largest-region A9 cost; parity of the spotter and of D), F and F+D (the det row plus rubi,fix=1 on
the largest dev region, textspot7's cost rule: eval/textspot7_a9_20261027.json). Baselines: textspot6's stored
GRAD + refine (rubi) and rubi,fix=1 results (the same units and pins). Report rows: textspot6's control (base_t64n8c16 seeds 1-3, every decode) and every MeikiOCR reference row
found under EVAL/meiki.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6_table as T6T  # noqa: E402

ARMS = ("t8_t64n8c16",)
SEEDS = (1, 2, 3)
DECODES = T6T.DECODES
MEIKI = ("raw", "pad", "ink", "raw+refine", "pad+refine", "ink+refine")
GRAD_MS_333 = 122.4          # textspot7's A9 row of rubi,fix=1 on the largest of #119's base regions (F's added cost)


def extra_rows(name, arm, seed, ev, run, det, crop):
    """amendment 3 and 4's decodes of one scored seed: ctx, det+D, F, F+D (rows that exist only)."""
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    shape = meta["shape"]["name"]
    blob = run / "final.vtsp"
    par = json.loads((ev / "parity.json").read_text(encoding="utf-8")) if (ev / "parity.json").exists() else {}

    def ok(*files):
        out = bool(par.get("all_equal"))
        for f in files:
            if not (ev / f).exists():
                return None
            out = out and bool(json.loads((ev / f).read_text(encoding="utf-8")).get("all_equal"))
        return out
    rows = []
    base = {"arm": arm, "seed": seed, "run": name, "shape": shape, "vtsp_bytes": blob.stat().st_size,
            "blob_sha256": T6T.sha256_file(blob)}
    if (ev / "dev_ctx.json").exists():
        r = crop[shape]
        rows.append(dict(base, decode="ctx", dev=str(ev / "dev_ctx.json"), hval=str(ev / "hval_ctx.json"),
                         est_ms_333=round(r["ms_333"], 1), est_ms_444=round(r["ms_444"], 1), arena=int(r["arena"]),
                         c_equal=ok("parity_ctx_dev.json", "parity_ctx_hval.json")))
    dcost = ev / "a9" / "out" / "a9_t7_D.json"
    if (ev / "dev_det+D.json").exists() and dcost.exists():
        r = det[shape]
        dms = json.loads(dcost.read_text(encoding="utf-8"))["max_one_ms_333"]
        rows.append(dict(base, decode="det+D", dev=str(ev / "dev_det+D.json"), hval=str(ev / "hval_det+D.json"),
                         est_ms_333=round(r["ms_333"] + dms, 1), est_ms_444=round((r["ms_333"] + dms) * 333 / 444, 1),
                         arena=int(r["arena"]), c_equal=ok("parity_D_dev.json", "parity_D_hval.json"), d_ms_333=dms))
    for dec in ("F", "F+D"):
        if (ev / f"dev_{dec}.json").exists():
            r = det[shape]
            ms = r["ms_333"] + GRAD_MS_333
            rows.append(dict(base, decode=dec, dev=str(ev / f"dev_{dec}.json"), hval=str(ev / f"hval_{dec}.json"),
                             est_ms_333=round(ms, 1), est_ms_444=round(ms * 333 / 444, 1), arena=int(r["arena"]),
                             c_equal=None))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--base-eval", required=True, help="textspot6's eval directory (its baselines and control)")
    ap.add_argument("--a9", default=str(T6T.A9))
    ap.add_argument("--dev-units", default=str(T6T.DEV_UNITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(argv)
    E, R, B = Path(a.eval), Path(a.runs), Path(a.base_eval)
    det, crop = T6T.a9_rows(a.a9)
    kinds = [u["kind"] for u in pickle.load(open(a.dev_units, "rb"))["units"]]
    cands, missing = [], []
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_seed{seed}"
            ev = E / name
            if not (ev / "SCORED").exists():
                missing.append(name)
                continue
            cands.extend(T6T.candidate_rows(name, arm, seed, ev, R / name, det, crop, kinds))
            cands.extend(extra_rows(name, arm, seed, ev, R / name, det, crop))
    unchecked = sorted({c["run"] for c in cands if c["c_equal"] is None and c["decode"] not in ("F", "F+D")})
    if (missing or unchecked) and not a.allow_partial:
        raise SystemExit(f"not scored yet: {missing}; no parity check: {unchecked}")
    reports = {}
    for seed in SEEDS:
        for d in DECODES:
            p = B / f"base_t64n8c16_seed{seed}"
            reports[f"textspot6_base_t64n8c16|s{seed}|{d}"] = {"dev": str(p / f"dev_{d}.json"),
                                                               "hval": str(p / f"hval_{d}.json")}
    for d in MEIKI:
        dv, hv = E / "meiki" / f"dev_{d}.json", E / "meiki" / f"hval_{d}.json"
        if dv.exists() and hv.exists():
            reports[f"meiki|{d}"] = {"dev": str(dv), "hval": str(hv)}
    tab = {"kind": "textspot6-selection-table", "study": "textspot8",
           "baseline": {"dev": str(B / "grad_refine" / "dev.json"), "hval": str(B / "grad_refine" / "hval.json")},
           "rubi_next": {"name": "rubi,fix=1", "dev": str(B / "grad_refine_fix1" / "dev.json"),
                         "hval": str(B / "grad_refine_fix1" / "hval.json")},
           "seen_dev_clusters": ["v12188"], "candidates": cands, "reports": reports, "missing": missing,
           "no_parity": unchecked, "a9": str(a.a9), "a9_sha256": T6T.sha256_file(a.a9)}
    Path(a.out).write_text(json.dumps(tab, indent=1) + "\n", encoding="utf-8")
    print(f"{a.out}: {len(cands)} candidate rows, {len(reports)} report rows, missing {missing}")


if __name__ == "__main__":
    sys.exit(main())

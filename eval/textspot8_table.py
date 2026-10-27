#!/usr/bin/env python3
"""textspot8's selection table (eval/textspot8_protocol.json "rule"): eval/textspot6_select.py's TABLE.json for
textspot8's arms, so the rule runs as textspot6's code (eval/textspot6_select.py, unchanged).

  textspot8_table.py --eval $A/vocr6-spotter2/eval --runs $A/vocr6-spotter2/runs --base-eval $A/vocr6-spotter/eval \
      --out TABLE.json [--allow-partial]

Candidates: amendment 1's t8_t64n8c16, seeds 1-3, in textspot6's four decodes, with textspot6's cost rows
(eval/textspot6_table.candidate_rows: the A9 rows of record of the shape, the refine decodes' extra time and arena,
the parity check). Baselines: textspot6's stored GRAD + refine (rubi) and rubi,fix=1 results (the same units and
pins). Report rows: textspot6's control (base_t64n8c16 seeds 1-3, every decode) and every MeikiOCR reference row
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
    unchecked = sorted({c["run"] for c in cands if c["c_equal"] is None})
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

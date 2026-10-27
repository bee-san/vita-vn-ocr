#!/usr/bin/env python3
"""textspot6's selection table (issue #128; eval/textspot6_select.py --table): every scored candidate seed in every
decode with its cost rows, from the scoring outputs, the A9 rows of record and the parity checks.

  textspot6_table.py --eval $A/vocr6-spotter/eval --runs $A/vocr6-spotter/runs --out TABLE.json [--allow-partial]

Candidates: the arms of the rule (stage 1's base_t64n8c16, base_qS4, base_qL5; stage 2's kd_t64n8c16), seeds 1-3
each (amendment 4), x {crop, crop+refine, det, det+refine}; the nohook control (amendment 8) and the references
(rubi,fix=5, textspot3's pick) are report rows. Costs (protocol "budget", amendments 2 and 10):
  vtsp_bytes   the run's final.vtsp
  est_ms_333   det decodes: the A9 row of record of the shape (eval/textspot6_a9_20261026.json "summary", tags 4, O2,
               dark pass, 960 x 544); crop decodes: the forced SCALE2 row of a 960 x 544 region ("forced", the bound
               of any region of the frame); +1.4 ms per refined horizontal box at the run's median for the refine
               decodes: det+refine over the development Vita frames (spot_dev.jsonl's refined boxes), crop+refine
               over the development Vita units (boxes_dev_crop+refine.jsonl)
  arena        the same rows' arena up to 960 x 544; the refine decodes the larger of it and the refinement's
               72,391 B (textspot2's rule)
  c_equal      parity.json's all_equal (the C runtime against the twin on every development frame and crop)
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A9 = ROOT / "eval" / "textspot6_a9_20261026.json"
DEV_UNITS = Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts/vocr5-analysis/work/units.pkl")
ARMS = ("base_t64n8c16", "base_qS4", "base_qL5", "kd_t64n8c16")
REPORT_ARMS = ("nohook_t64n8c16",)
SEEDS = (1, 2, 3)
DECODES = ("crop", "crop+refine", "det", "det+refine")
REFINE_MS, REFINE_ARENA = 1.4, 72391


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def a9_rows(path=A9):
    a9 = json.loads(Path(path).read_text(encoding="utf-8"))
    det = {r["shape"]: {"ms_333": r["ms"]["O2"]["dark"]["960x544"]["333"],
                        "ms_444": r["ms"]["O2"]["dark"]["960x544"]["444"], "arena": r["arena"]["960x544"]["up_to"],
                        "vtsp_bytes": r["vtsp_bytes"]} for r in a9["summary"] if r["tags"] == 4}
    crop = {r["shape"]: {"ms_333": r["ms_333_dark"], "ms_444": r["ms_444_dark"], "arena": r["arena_up_to_bytes"]}
            for r in a9["forced"]["rows"] if r["region"] == "960x544" and r["build"] == "O2" and r["force"] == 2}
    return det, crop


def refined_median(ev, decode, kinds):
    """The run's median number of refined horizontal boxes per development Vita frame (det+refine) or unit
    (crop+refine)."""
    if decode == "det+refine":
        rows = [json.loads(x) for x in (ev / "spot_dev.jsonl").read_text(encoding="utf-8").splitlines()[1:]]
        n = [sum(1 for b in r["rboxes"] if not b[4]) for r in rows if r["view"] == "vita"]
    else:
        n = []
        for x in (ev / "boxes_dev_crop+refine.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            if kinds[int(r["unit"].split(":")[1])] == "vita":
                n.append(len(r["boxes"]))
    return float(np.median(n)), len(n)


def candidate_rows(name, arm, seed, ev, run, det, crop, kinds):
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    shape = meta["shape"]["name"]
    blob = run / "final.vtsp"
    par = json.loads((ev / "parity.json").read_text(encoding="utf-8")) if (ev / "parity.json").exists() else {}
    out = []
    for dec in DECODES:
        row = (crop if dec.startswith("crop") else det)[shape]
        ms3, ms4, arena = row["ms_333"], row["ms_444"], row["arena"]
        extra = {}
        if dec.endswith("refine"):
            med, n = refined_median(ev, dec, kinds)
            ms3, ms4 = ms3 + REFINE_MS * med, ms4 + REFINE_MS * med * 333 / 444
            arena = max(arena, REFINE_ARENA)
            extra = {"refined_boxes_median": med, "refined_over": n}
        out.append({"arm": arm, "seed": seed, "run": name, "decode": dec, "shape": shape,
                    "dev": str(ev / f"dev_{dec}.json"), "hval": str(ev / f"hval_{dec}.json"),
                    "vtsp_bytes": blob.stat().st_size, "blob_sha256": sha256_file(blob),
                    "est_ms_333": round(ms3, 1), "est_ms_444": round(ms4, 1), "arena": int(arena),
                    "c_equal": bool(par["all_equal"]) if par else None, "parity_sets": sorted(par.get("sets", {})),
                    **extra})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--a9", default=str(A9))
    ap.add_argument("--dev-units", default=str(DEV_UNITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="a preview: unscored seeds are left out")
    a = ap.parse_args(argv)
    E, R = Path(a.eval), Path(a.runs)
    det, crop = a9_rows(a.a9)
    kinds = [u["kind"] for u in pickle.load(open(a.dev_units, "rb"))["units"]]
    cands, reports, missing, missing_reports = [], [], [], []
    for arms, into, miss in ((ARMS, cands, missing), (REPORT_ARMS, reports, missing_reports)):
        for arm in arms:
            for seed in SEEDS if arms is ARMS else (1, 2):
                name = f"{arm}_seed{seed}"
                ev = E / name
                if not (ev / "SCORED").exists():
                    miss.append(name)
                    continue
                into.extend(candidate_rows(name, arm, seed, ev, R / name, det, crop, kinds))
    unchecked = sorted({c["run"] for c in cands if c["c_equal"] is None})
    if (missing or unchecked) and not a.allow_partial:
        raise SystemExit(f"not scored yet: {missing}; no parity check: {unchecked}")
    tab = {"kind": "textspot6-selection-table", "baseline": {"dev": str(E / "grad_refine" / "dev.json"),
                                                            "hval": str(E / "grad_refine" / "hval.json")},
           "rubi_next": {"name": "rubi,fix=1", "dev": str(E / "grad_refine_fix1" / "dev.json"),
                         "hval": str(E / "grad_refine_fix1" / "hval.json")},
           "seen_dev_clusters": ["v12188"], "candidates": cands,
           "reports": {"rubi,fix=5": {"dev": str(E / "grad_refine_fix5" / "dev.json"),
                                      "hval": str(E / "grad_refine_fix5" / "hval.json")},
                       **{f"textspot3_pick|{d}": {"dev": str(E / "textspot3_pick" / f"dev_{d}.json"),
                                                  "hval": str(E / "textspot3_pick" / f"hval_{d}.json")}
                          for d in DECODES},
                       **{f"{r['arm']}|s{r['seed']}|{r['decode']}": {"dev": r["dev"], "hval": r["hval"]}
                          for r in reports}},
           "missing": missing, "missing_reports": missing_reports, "no_parity": unchecked, "a9": str(a.a9),
           "a9_sha256": sha256_file(a.a9)}
    Path(a.out).write_text(json.dumps(tab, indent=1) + "\n", encoding="utf-8")
    print(f"{a.out}: {len(cands)} candidate rows, {len(reports)} report rows, missing {missing}, "
          f"report rows not scored yet {missing_reports}")


if __name__ == "__main__":
    sys.exit(main())

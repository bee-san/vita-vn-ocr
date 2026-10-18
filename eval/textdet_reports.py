#!/usr/bin/env python3
"""textdet: the producers of the report-only records of docs/textdet.md (none of them decides anything).

  textdet_reports.py own-dlg --private DIR --out eval/textdet_own_dlg_20261018.json
      The owner's CHAOS;CHILD (v14018) native Vita whole dialogue: finder-tighten's own_dlg cases of kind 'vita'
      (DIR/cohort/own_dlg.pkl), GRAD and GRAD + refine units from eval/finder_tighten.py score (DIR/base/<model>/),
      det+refine units from eval/textdet.py read (DIR/units/<model>/units_s8_c24_seed4.json, dets in DIR/dets/).
      Intervals: a paired bootstrap over the 145 regions (one VN), 10,000 resamples, numpy default_rng(0).
  textdet_reports.py int4 --artifacts T --out eval/textdet_int4_20261018.json
      The int4 study: the three blobs re-exported with train/textdet_export.py --wbits 4 (T/int4/blobs), det+refine,
      VOCR5-N, development wd, against the same blobs' int8 units (round 3's s8_c24_seed4, round 2's other two).
  textdet_reports.py round1-final --artifacts T --cbin DRIVER --out eval/textdet_round1_final_runtime_20261018.json
      Round 1 under the final runtime: each blob's arena at 960 x 544 (DRIVER --arena) and its development lines
      (T/eval/dets_c, the final driver's detect --cbin) against round 1's record (T/eval/dets).
  textdet_reports.py empty-pool --artifacts T --out eval/textdet_empty_pool_check_20261018.json
      eval/linefind's empty regions split by whether their screenshot is in synth3's scene pool (synth3.init(), train
      split), GRAD and two det+refine blobs.
The private inputs are read in place; the outputs hold counts, rates and hashes only.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet as TD  # noqa: E402

OWN_NOTE = ("report only, run after the protected read: the owner's CHAOS;CHILD (v14018) native Vita frames as whole "
            "dialogue (finder-tighten's own_dlg Vita units: 145 regions x base/+4/+8); the VOCR5 models trained on "
            "these frames (in-sample for the recognizers), the detector did not (three VNDB PC screenshots of v14018 "
            "are in its synthetic scene pool); intervals: paired bootstrap over the 145 regions (one VN), 10,000 "
            "resamples, seed 0")
INT4_NOTE = ("report only: weights limited to +-7 codes (train/textdet_export.py --wbits 4), det+refine, VOCR5-N, "
             "development wd; the runtime runs int8")
INT4 = (("s8_c24_seed4", "eval_r3"), ("s8_c24_seed1", "eval_r2"), ("s4_c16n4_seed1", "eval_r2"))
ROUND1_NOTE = ("round 1's record (eval/textdet_round1_dev_20261018.json) used the early, bit-exact runtime f9d4ad3 "
               "for its C == twin check and arenas (protocol amendment 3); with the final runtime the detector's lines "
               "are identical on every development case and the arenas below apply; no candidate's eligibility "
               "changes")
POOL_NOTE = ("report only (found after the protected read): eval/linefind empty regions split by whether their "
             "screenshot is in synth3's scene pool, which the synthetic training canvases draw backgrounds and empty "
             "negatives from")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def own_dlg(args):
    P = Path(args.private)
    out = {"kind": "textdet-own-dlg-report", "note": OWN_NOTE, "models": {}, "inputs": {}}
    for m in TD.MODELS:
        fb, fd = P / "base" / m / f"units_{m}.json", P / "units" / m / "units_s8_c24_seed4.json"
        out["inputs"][f"base/{m}/units_{m}.json"] = sha256(fb)
        out["inputs"][f"units/{m}/units_s8_c24_seed4.json"] = sha256(fd)
        base = json.loads(fb.read_text(encoding="utf-8"))["own_dlg"]
        det = json.loads(fd.read_text(encoding="utf-8"))["own_dlg"]["det"]
        g = {u["unit"]: u for u in base["grad"]}
        r = {u["unit"]: u for u in base["refine"]}
        d = {u["unit"]: u for u in det}
        if not set(g) == set(d) == set(r):
            raise SystemExit(f"{m}: the units differ")
        keys = sorted(d)
        reg = sorted({k.rsplit("|", 1)[0] for k in keys})
        ri = {x: i for i, x in enumerate(reg)}

        def agg(src):
            e, n = np.zeros(len(reg)), np.zeros(len(reg))
            for k in keys:
                e[ri[k.rsplit("|", 1)[0]]] += src[k]["errors"]
                n[ri[k.rsplit("|", 1)[0]]] += src[k]["chars"]
            return e, n

        eg, n = agg(g)
        er, _ = agg(r)
        ed, _ = agg(d)
        idx = np.random.default_rng(0).integers(0, len(reg), (10000, len(reg)))

        def ci(a, b):
            ds = (a[idx].sum(1) - b[idx].sum(1)) / n[idx].sum(1) * 100
            lo, hi = np.percentile(ds, [2.5, 97.5])
            return round(100 * (a.sum() - b.sum()) / n.sum(), 3), [round(float(lo), 3), round(float(hi), 3)]

        out["models"][m] = {"chars": int(n.sum()), "regions": len(reg), "cases": len(keys),
                            "cer": {"grad": round(100 * eg.sum() / n.sum(), 3),
                                    "refine": round(100 * er.sum() / n.sum(), 3),
                                    "det_refine": round(100 * ed.sum() / n.sum(), 3)},
                            "det_refine_minus_grad": ci(ed, eg), "det_refine_minus_refine": ci(ed, er),
                            "zero_detection_cases": sum(1 for u in det if u["detections"] == 0)}
    for f in ("cohort/own_dlg.pkl", "dets/dets_s8_c24_seed4.json"):
        out["inputs"][f] = sha256(P / f)
    dets = json.loads((P / "dets" / "dets_s8_c24_seed4.json").read_text(encoding="utf-8"))
    out["inputs"]["blob_sha256"] = dets["blob_sha256"]
    out["inputs"]["c_equals_twin"] = dets["c_equals_twin"]
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def int4(args):
    T = Path(args.artifacts)
    res, inputs = {}, {}
    for c, rnd in INT4:
        fa = T / "int4" / "units" / "VOCR5-N" / f"units_{c}_w4.json"
        fb = T / rnd / "units" / "VOCR5-N" / f"units_{c}.json"
        inputs[f"int4/units/VOCR5-N/units_{c}_w4.json"] = sha256(fa)
        inputs[f"{rnd}/units/VOCR5-N/units_{c}.json"] = sha256(fb)
        inputs[f"int4/blobs/{c}_w4.vtd"] = sha256(T / "int4" / "blobs" / f"{c}_w4.vtd")
        a = TD.wd_units(json.loads(fa.read_text(encoding="utf-8")), "det")
        b = TD.wd_units(json.loads(fb.read_text(encoding="utf-8")), "det")
        d = TD.vn_diff([(a, b)])
        res[c] = {"int8_wd_cer": TD.cer_of(b)["cer"], "int4_wd_cer": TD.cer_of(a)["cer"],
                  "int4_minus_int8": d["delta"], "ci95": d["ci95"]}
    doc = {"kind": "textdet-int4-study", "note": INT4_NOTE, "results": res, "inputs": inputs}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def round1_final(args):
    T = Path(args.artifacts)
    runs = json.loads((ROOT / "eval" / "textdet_protocol.json").read_text(encoding="utf-8"))["amendments"][1]["runs"]
    r1 = json.loads((ROOT / "eval" / "textdet_round1_dev_20261018.json").read_text(encoding="utf-8"))
    out = {"kind": "textdet-round1-final-runtime", "runtime": "feat/textdet-runtime 000bdca (merged in PR #97)",
           "driver_sha256": sha256(args.cbin), "note": ROUND1_NOTE, "runs": {}}
    for r in runs:
        a = int(subprocess.run([args.cbin, "--arena", str(T / "sweep" / r / "final.vtd"), "960", "544"], check=True,
                               capture_output=True, text=True).stdout.split()[0])
        d0 = json.loads((T / "eval" / "dets" / f"dets_{r}.json").read_text(encoding="utf-8"))
        d1 = json.loads((T / "eval" / "dets_c" / f"dets_{r}.json").read_text(encoding="utf-8"))
        out["runs"][r] = {"arena_960x544_final": a, "arena_960x544_record": r1["candidates"][r]["arena_960x544"],
                          "lines_identical": d0["dets"] == d1["dets"], "c_equals_twin_final": d1["c_equals_twin"]}
    if not all(v["lines_identical"] and v["c_equals_twin_final"] and v["arena_960x544_final"] <= 131072
               for v in out["runs"].values()):
        raise SystemExit("a round-1 blob differs under the final runtime")
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def empty_pool(args):
    import synth3 as S3
    T = Path(args.artifacts)
    R = S3.init()
    names = {Path(p).name for paths in R.bg.values() for p in paths}
    ids = {m.group(2) for n in names for m in [re.match(r"(v\d+)_(\d+)", n)] if m}
    finders = {"grad": TD.make_finder("grad"),
               "det+refine s8_c24_seed4": TD.make_finder(f"detref:s8_c24_seed4={T}/sweep_r3/s8_c24_seed4/final.vtd"),
               "det+refine s8_c24_seed1": TD.make_finder(f"detref:s8_c24_seed1={T}/sweep/s8_c24_seed1/final.vtd")}
    shots = [str(d) for d in TD._linefind("build_gt").SHOT_DIRS]
    test = TD.reserved_test_vns()
    out = {}
    for role, sets in (("development", ["vita_dev", "psp_dev"]), ("report", ["vita_holdout", "psp_holdout"])):
        rows = []
        for s in sets:
            keep, _ = TD.linefind_rows(s, test)
            rows += [r for r in keep if "empty" in r.get("tags", [])]
        for name, f in finders.items():
            res = {"in_pool": [0, 0], "not_in_pool": [0, 0]}
            for r in rows:
                boxes, _ = f(TD.region_grey(r, shots))
                sid = r["url"].rsplit("/", 1)[-1].split(".")[0]
                k = "in_pool" if (sid in ids or r["shot"] in names) else "not_in_pool"
                res[k][0] += len(boxes) == 0
                res[k][1] += 1
            out[f"{role}/{name}"] = res
    doc = {"kind": "textdet-empty-scene-pool-check", "note": POOL_NOTE, "results": out}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("own-dlg")
    p.add_argument("--private", required=True)
    p.add_argument("--out", required=True)
    for name in ("int4", "round1-final", "empty-pool"):
        p = sub.add_parser(name)
        p.add_argument("--artifacts", required=True)
        p.add_argument("--out", required=True)
        if name == "round1-final":
            p.add_argument("--cbin", required=True)
    args = ap.parse_args()
    {"own-dlg": own_dlg, "int4": int4, "round1-final": round1_final, "empty-pool": empty_pool}[args.cmd](args)


if __name__ == "__main__":
    main()

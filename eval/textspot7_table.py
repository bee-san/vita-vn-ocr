#!/usr/bin/env python3
"""textspot7's selection table (issue #128): eval/textspot7_select.py's TABLE.json from the scored runs, the A9 rows
and the C == twin records.

  textspot7_table.py --out TABLE.json [--eval DIR] [--stored DIR] [--a9 eval/textspot7_a9_20261027.json]
                     [--a9-t7 DIR] [--confirm]

Per representative and arm (D, F, F+D):
  est. ms @333 = the spotter's row of record (eval/textspot6_a9_20261026.json, -O2, the dark pass, 960 x 544)
               + (F, F+D) rubi,fix=1 on the largest development region (eval/textspot7_a9_20261027.json)
               + the largest of the five largest regions' runtime/vocr_textspot7.c records (eval/textspot7_a9.py t7:
                 D's post-pass and F's choice; F alone: the region assignment and the choice)
  arena        = the largest of the spotter's, the finder's (vocr_lines_arena_bytes(960, 544)) and the post-pass's
                 (vocr_textspot7_arena_bytes(960, 544)): they run one after another
  c_equal      = textspot6's parity of the spotter (dev, dev-crop, tsdev) and this round's on every dev and hval unit
                 (D's boxes, F's choice, F+D's choice over D's boxes)
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
REPS = {"base_t64n8c16": "base_t64n8c16_seed1", "kd_t64n8c16": "kd_t64n8c16_seed2", "base_qS4": "base_qS4_seed2",
        "base_qL5": "base_qL5_seed1"}
SHAPE = {"base_t64n8c16": "t64n8c16_t4", "kd_t64n8c16": "t64n8c16_t4", "base_qS4": "qS4_t4", "base_qL5": "qL5_t4"}
LINES_ARENA_960, T7_ARENA_960 = 83323, 72391          # measured: vocr_lines_arena_bytes / vocr_textspot7_arena_bytes


def spotter_rows():
    d = json.loads((ROOT / "eval" / "textspot6_a9_20261026.json").read_text(encoding="utf-8"))
    out = {}
    for r in d["summary"]:
        if r.get("tags") == 4:
            out[r["row"]] = {"ms_333": r["ms"]["O2"]["dark"]["960x544"]["333"],
                             "ms_444": r["ms"]["O2"]["dark"]["960x544"]["444"],
                             "arena": r["arena"]["960x544"]["up_to"], "vtsp_bytes": r["vtsp_bytes"]}
    return out


def t7_max(a9dir, run, kind):
    p = Path(a9dir) / f"out_{run}" / f"a9_t7_{run}_{kind}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    ones = [v["ms_333"] for k, v in d["rows"].items() if k.startswith("one")]
    return max(ones), d["rows"]["all"]["ms_333"]


def eq(path, key):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return bool(d.get(key, {}).get("equal"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", default=str(ART / "vocr6-spotter3" / "eval"))
    ap.add_argument("--stored", default=str(ART / "vocr6-spotter" / "eval"))
    ap.add_argument("--a9", default=str(ROOT / "eval" / "textspot7_a9_20261027.json"))
    ap.add_argument("--a9-t7", default=str(ART / "vocr6-spotter3" / "eval" / "a9" / "t7"))
    ap.add_argument("--confirm", action="store_true", help="add the confirmation results (phase confirm)")
    a = ap.parse_args(argv)
    E, S = Path(a.eval), Path(a.stored)
    rows = spotter_rows()
    grad_ms = json.loads(Path(a.a9).read_text(encoding="utf-8"))["grad"]["largest_ms_333"]
    tab = {"baseline": {"dev": str(S / "grad_refine" / "dev.json"), "hval": str(S / "grad_refine" / "hval.json")},
           "rubi_next": {"name": "rubi,fix=1", "dev": str(S / "grad_refine_fix1" / "dev.json"),
                         "hval": str(S / "grad_refine_fix1" / "hval.json")},
           "seen_dev_clusters": ["v12188"], "candidates": []}
    if a.confirm:
        tab["rubi_next"]["confirm"] = str(E / "confirm_grad_refine_fix1" / "confirm.json")
        tab["baseline"]["confirm"] = str(E / "confirm_grad_refine" / "confirm.json")
    fits = json.loads((ROOT / "eval" / "textspot7_fit_20261027.json").read_text(encoding="utf-8"))
    for arm, run in REPS.items():
        sp = rows[SHAPE[arm]]
        p6 = json.loads((S / run / "parity.json").read_text(encoding="utf-8"))["all_equal"]
        fd_max, fd_mean = t7_max(a.a9_t7, run, "FD")
        f_max, f_mean = t7_max(a.a9_t7, run, "F")
        arena = max(sp["arena"], LINES_ARENA_960, T7_ARENA_960)
        ceq = {"spotter": p6,
               "D": eq(E / run / "parity_dev_D.json", "post") and eq(E / run / "parity_hval_D.json", "post"),
               "F": eq(E / run / "parity_dev_F.json", "choose") and eq(E / run / "parity_hval_F.json", "choose"),
               "F+D": eq(E / run / "parity_dev_FD.json", "choose") and eq(E / run / "parity_hval_FD.json", "choose")}
        fd = json.loads((E / "fit" / f"FD_{run}.json").read_text(encoding="utf-8"))["fit"]
        arms = {
            "D": (sp["ms_333"] + fd_max, {"spotter": sp["ms_333"], "post_and_choice_max": fd_max,
                                          "post_and_choice_mean": fd_mean}, ceq["spotter"] and ceq["D"],
                  {"steps": fits["D"]["D"]}, "D"),
            "F": (sp["ms_333"] + grad_ms + f_max, {"spotter": sp["ms_333"], "grad_largest": grad_ms,
                                                   "assign_and_choice_max": f_max, "assign_and_choice_mean": f_mean},
                  ceq["spotter"] and ceq["F"], {"F": fits["F"][run]["fit"]["params"]}, "F"),
            "F+D": (sp["ms_333"] + grad_ms + fd_max, {"spotter": sp["ms_333"], "grad_largest": grad_ms,
                                                      "post_and_choice_max": fd_max},
                    ceq["spotter"] and ceq["D"] and ceq["F+D"], {"steps": fits["D"]["D"], "F": fd["params"]}, "FD"),
        }
        for dec, (ms, parts, ok, params, tag) in arms.items():
            c = {"name": f"{arm}|{dec}", "arm": arm, "run": run, "decode": dec, "params": params,
                 "dev": str(E / run / f"dev_{tag}.json"), "hval": str(E / run / f"hval_{tag}.json"),
                 "vtsp_bytes": sp["vtsp_bytes"], "est_ms_333": round(ms, 2), "est_ms_444": round(ms * 333 / 444, 2),
                 "est_ms_parts": {k: round(v, 3) for k, v in parts.items()}, "arena": arena, "c_equal": ok,
                 "c_equal_parts": ceq}
            if a.confirm:
                c["confirm"] = str(E / run / f"confirm_{tag}.json")
            tab["candidates"].append(c)
    Path(a.out).write_text(json.dumps(tab, indent=1) + "\n", encoding="utf-8")
    print(f"table: {len(tab['candidates'])} candidates -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""dev_scores.py: every VOCR3 candidate's development scores (reported only; the selections are
eval/vocr3_selection_*.json).

  dev_scores.py --select-dir $S3/vocr3/select --out eval/vocr3_dev_scores_20261013.json

Per size, from the selection driver's score files (teacher_val.json, devnative_*.json over the amendment-8 cohorts) and
budget check (budget/sizes.json, budget/a9/*_l22048.json): teacher_val CER, devnative CER, the development score (their
mean), blob bytes, W8I arena, the A9 estimate's high bound at 333 MHz and the docs/vita-memory-budget.md tier. Integer
error counts and character totals only; no line text.
"""
import argparse
import json
from pathlib import Path

VIEWS = ("rv", "rvn")
DN = ("strict_ink", "strict_grad", "tv_native_ink", "tv_native_grad")


def errors(path):
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    chars = sum(len(x["ref"]) for x in s["lines"])
    errs = {m: sum(sum(v["errors"][k]) for k in VIEWS) for m, v in s["models"].items()}
    return errs, chars, {m: v["blob_sha256"] for m, v in s["models"].items()}


def tier(blob, arena, ms):
    if blob <= 900000 and arena <= 131072 and ms <= 300:
        return "always"
    if blob <= 1600000 and arena <= 262144 and ms <= 550:
        return "preflight"
    return "never"


def size_table(d):
    tv, tv_chars, shas = errors(d / "teacher_val.json")
    dn_err, dn_chars = {}, 0
    for f in DN:
        e, c, _ = errors(d / f"devnative_{f}.json")
        dn_chars += c
        for m, v in e.items():
            dn_err[m] = dn_err.get(m, 0) + v
    sizes = json.loads((d / "budget" / "sizes.json").read_text(encoding="utf-8"))
    a9 = {}
    for p in sorted((d / "budget" / "a9").glob("*_l22048.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        a9[r["name"]] = float(r["ms_640px"]["333"][1])
    rows = []
    for m in tv:
        t = tv[m] / (2 * tv_chars)
        n = dn_err[m] / (2 * dn_chars)
        info = sizes[m]
        blob, arena, ms = info["blob_bytes"], info["arena"]["t4h16"]["w8i"], a9[m]
        rows.append({"run": m, "blob_sha256": shas[m],
                     "teacher_val": {"errors": tv[m], "chars_x_views": 2 * tv_chars, "cer": round(t, 6)},
                     "devnative_v3": {"errors": dn_err[m], "chars_x_views": 2 * dn_chars, "cer": round(n, 6)},
                     "development_score": round((t + n) / 2, 6), "blob_bytes": blob, "arena_w8i_bytes": arena,
                     "a9_ms_333_high": round(ms, 1), "tier": tier(blob, arena, ms)})
    rows.sort(key=lambda r: r["development_score"])
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"kind": "vocr3-development-scores", "schema": 1,
           "note": "reported only: every candidate and rp run of each size on the amendment-8 cohorts "
                   "(teacher_val 1,643 lines, devnative v3 986 crops; both views), with the budget check; "
                   "the selections are eval/vocr3_selection_*_20261013.json",
           "sizes": {}}
    for size in ("FL10", "H15", "W20"):
        d = Path(args.select_dir) / size
        if (d / "teacher_val.json").exists():
            out["sizes"][size] = size_table(d)["rows"]
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for size, rows in out["sizes"].items():
        for r in rows:
            tv, dn = 100 * r["teacher_val"]["cer"], 100 * r["devnative_v3"]["cer"]
            print(f"{size} {r['run']}: tv {tv:.2f}% dn {dn:.2f}% score {100 * r['development_score']:.2f}% {r['tier']}")


if __name__ == "__main__":
    main()

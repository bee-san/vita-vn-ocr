#!/usr/bin/env python3
"""eval/finder_fix_tables.py (issue #129): the committed results of the finder fixes, from eval/pagepath.py's result
files (counts only: no text, no pixels).

  finder_fix_tables.py --runs DIR --out eval/finder_fix_20261026.json

DIR holds <arm>_<set>.json of `pagepath.py run` (arms base, F1, F12, F123, F13, oracle; S_*/L_* for the tiers; sets
hval, dev, tune, test, empty). Every comparison is pagepath.py's cluster bootstrap (titles on hval and tune, VNs on
dev, pages on the single test title).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import pagepath as PP  # noqa: E402

STEPS = (("F1", "base"), ("F12", "F1"), ("F123", "F1"), ("F13", "F1"), ("F13", "base"), ("F123", "base"))


def load(runs, name):
    p = Path(runs) / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cmp(a, b, by_page=False, platform=None):
    ua = [dict(u) for u in a["units"] if platform is None or u["platform"] == platform]
    ub = [dict(u) for u in b["units"] if platform is None or u["platform"] == platform]
    if by_page:
        for u in ua + ub:
            u["cluster"] = u["uid"].split(":")[1]
    return PP.cluster_bootstrap(ua, ub)


def totals(r):
    s = r["summary"]["all"]["all"]
    out = {k: s[k] for k in ("units", "errors", "chars", "cer", "lines", "boxes", "whole", "cut", "missed", "merged",
                             "junk", "line_recall", "box_precision", "frames")}
    out["by_platform"] = {k.split(":")[1]: {f: v[f] for f in ("units", "errors", "chars", "cer")}
                          for k, v in r["summary"]["by_platform"].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doc = {"schema": "vocr6-finder-fix-results/1", "issue": 129,
           "path": "eval/pagepath.py (#131): TEXTBOX_RECOVERY + Rubi's short-line probe + the refinement on PSP "
                   "titles, VOCR5 W8I, counter re-decode 6/4, beam 8 + vocr_lm_o3_1000k, cascade < 0.7, Rubi's "
                   "core/textnorm.c, the non-Japanese line filter; console view (PSP frames 2x bilinear)",
           "arms": {"base": "Rubi today: the frozen finder", "F1": "fix COLGAP", "F12": "fix COLGAP|VALLEY",
                    "F123": "fix COLGAP|VALLEY|JUNK", "F13": "fix COLGAP|JUNK (adopted: VOCR_FIX_DEFAULT)",
                    "oracle": "each reference line's own box plus its pad"},
           "runs": {}, "steps": {}, "tiers": {}, "test_once": {}, "empty_regions": {}, "per_title_hval": {}}
    first = None
    for arm in ("base", "F1", "F12", "F123", "F13", "oracle"):
        for s in ("hval", "dev", "tune"):
            r = load(a.runs, f"{arm}_{s}")
            if r is None:
                continue
            first = first or r
            doc["runs"][f"{arm}_{s}"] = dict(totals(r), units_sha256=r["units_sha256"],
                                             finder=r["finder"].get("sources", r["finder"]))
    for arm, base in STEPS:
        for s in ("hval", "dev", "tune"):
            x, y = load(a.runs, f"{arm}_{s}"), load(a.runs, f"{base}_{s}")
            if x and y:
                doc["steps"][f"{arm}_vs_{base}_{s}"] = {"all": cmp(x, y), "vita": cmp(x, y, platform="vita"),
                                                         "psp": cmp(x, y, platform="psp")}
    for t in ("S", "L"):
        for s in ("hval", "dev"):
            x, y = load(a.runs, f"{t}_F13_{s}"), load(a.runs, f"{t}_base_{s}")
            if x and y:
                doc["tiers"][f"{t}_{s}"] = {"base": totals(y), "F13": totals(x), "delta": cmp(x, y)}
    x, y = load(a.runs, "F13_test"), load(a.runs, "base_test")
    if x and y:
        doc["test_once"] = {"title": "PCSG00807", "base": totals(y), "F13": totals(x),
                            "delta_page_clusters": cmp(x, y, by_page=True)}
    for arm in ("base", "F13", "F123"):
        r = load(a.runs, f"{arm}_empty")
        if r:
            doc["empty_regions"][arm] = {"regions": len(r["units"]),
                                         "regions_with_text": sum(1 for u in r["units"] if u["errors"]),
                                         "chars_read": sum(u["errors"] for u in r["units"]),
                                         "boxes": sum(u["boxes"] for u in r["units"])}
    b, f = load(a.runs, "base_hval"), load(a.runs, "F13_hval")
    if b and f:
        fa = {u["uid"]: u for u in f["units"]}
        per = defaultdict(lambda: {"platform": "", "chars": 0, "base_errors": 0, "F13_errors": 0})
        for u in b["units"]:
            t = per[u["cluster"]]
            t["platform"] = u["platform"]
            t["chars"] += u["chars"]
            t["base_errors"] += u["errors"]
            t["F13_errors"] += fa[u["uid"]]["errors"]
        doc["per_title_hval"] = dict(sorted(per.items()))
    if first:
        doc["models"] = {"blobs": first["blobs"], "lm": first["lm"], "lm_opts": first["lm_opts"],
                         "textnorm": first["textnorm"],
                         "pagepool_manifest_sha256": first.get("pagepool_manifest_sha256")}
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(a.out, len(doc["runs"]), "runs", len(doc["steps"]), "comparisons")


if __name__ == "__main__":
    main()

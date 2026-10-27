#!/usr/bin/env python3
"""textspot8 step 2 (eval/textspot8_protocol.json "step2_geometry_diagnostic", amendment 1 "across_shrink_rule"): the
hval page-path CER of every geometry arm, and the across shrink the rule picks.

  geom_rule.py --geom $A/vocr6-spotter2/eval/geom --base-eval $A/vocr6-spotter/eval --out GEOM.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))
import textspot8_report as R  # noqa: E402

ARMS = {
    "fix1": ("grad_refine_fix1", "hval", [("dx0_dy1", 0, 1), ("dx0_dy2", 0, 2), ("dx4_dy0", 4, 0)]),
    "t64det": ("base_t64n8c16_seed1", "hval_det", [("dx0_dy-1", 0, -1), ("dx0_dy-2", 0, -2)]),
    "qS4crop": ("base_qS4_seed2", "hval_crop", [("dx0_dy-1", 0, -1), ("dx0_dy-2", 0, -2)]),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True)
    ap.add_argument("--base-eval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    G, B = Path(a.geom), Path(a.base_eval)
    out = {"arms": {}}
    for arm, (run, f, shifts) in ARMS.items():
        base = R.load(B / run / f"{f}.json")
        bu = R.units_of(base)
        rows = {"stored": {"dx": 0, "dy": 0, **R.summary(bu)}}
        for tag, dx, dy in shifts:
            p = G / f"hval_{arm}_{tag}.json"
            if not p.exists():
                continue
            res = R.load(p)
            ru = R.units_of(res)
            rows[tag] = {"dx": dx, "dy": dy, **R.summary(ru), "vs_stored": R.pair(ru, bu)}
        out["arms"][arm] = rows
    t = out["arms"]["t64det"]
    g = {0: t["stored"]["cer"]}
    for tag, d in (("dx0_dy-1", 1), ("dx0_dy-2", 2)):
        if tag in t:
            g[d] = t[tag]["cer"]
    best = min(sorted(g), key=lambda d: (g[d], d))
    out["rule"] = {"G": {str(k): v for k, v in g.items()}, "d_star": best, "across_shrink": best / 2,
                   "complete": len(g) == 3}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for arm, rows in out["arms"].items():
        for tag, r in rows.items():
            v = r.get("vs_stored")
            ci = f" vs stored {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" if v else ""
            print(f"{arm:8s} {tag:9s} hval CER {r['cer']:.3f}{ci}")
    print("rule:", json.dumps(out["rule"]))


if __name__ == "__main__":
    main()

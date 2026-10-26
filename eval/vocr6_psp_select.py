#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): the screening decisions of eval/vocr6_psp_protocol.json.

  vocr6_psp_select.py stage1 --runs DIR --sweep SWEEP.json --out DECISION.json
  vocr6_psp_select.py stage2 --runs DIR --winner ARM --cand ARM --out DECISION.json

stage1: per arm the mean over its seeds of the screening score SC (0.5 hp_grad + 0.25 wd_grad + 0.25 fixed, CER points,
from each run's run_record.json), and its feasibility (budgets.screening_feasibility): the est. console seconds per
nominal box from psp/modelb_sweep.py's PPSSPP costs of the arm's shape (scalar W8F, tile 16, head 64: `gate1` = the
backbone and the gate on every frame, `g1l1:63:0` - gate1 = level 1 + the escape per kept frame, `g1l1:63:1` -
`g1l1:63:0` = the rare rows per fired frame) at a skip share of 0.9 x the run's development blank-argmax share and an
escape share of 0.20 of the kept frames, x the nominal frames, + the finder (0.562 emulated us per pixel, measured,
x 480 x 90), x 1.45. The winner: the lowest mean SC among feasible arms; within 0.05 points, the faster.
stage2: CAND (e.g. Q, QAT) against WINNER: adopted iff its mean SC is lower and neither wd_grad's nor fixed's paired
VN-cluster 95% interval of the mean over seeds of CAND - WINNER lies entirely above 0 (eval/modelb_eval.py compare).
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINDER_US_PER_PX = 0.562            # PPSSPP, VOCR5-S smoke (psp/modelb_verify.py): 197,442 us over 351,260 px
NOMINAL_REGION_PX = 480 * 90
FACTOR = 1.45
CONFIG = "w8f:16:64"
ESC_SHARE = 0.20
SKIP_OF_BLANK = 0.9


def nominal_frames(height):
    return 3 * math.ceil(int(round(320 * height / 19)) / 8)


def runs_of(runs_dir, arm):
    out = []
    for p in sorted(Path(runs_dir).glob(f"mb_{arm}_seed*/run_record.json")):
        out.append(json.loads(p.read_text()))
    return out


def costs(sweep, shape):
    by = {(r["name"], r["eboot"], r["config"], r["variant"]): r["psp"]["us_frame"] for r in sweep}
    g1 = by[(shape, "scalar", CONFIG, "gate1")]
    l1 = by[(shape, "scalar", CONFIG, "g1l1:63:0")] - g1
    rare = by[(shape, "scalar", CONFIG, "g1l1:63:1")] - by[(shape, "scalar", CONFIG, "g1l1:63:0")]
    return g1, l1, rare


def est_box(sweep, shape, height, blank_share):
    g1, l1, rare = costs(sweep, shape)
    skip = SKIP_OF_BLANK * blank_share
    us = g1 + (1 - skip) * (l1 + ESC_SHARE * rare)
    emu = (us * nominal_frames(height) + FINDER_US_PER_PX * NOMINAL_REGION_PX) / 1e6
    return {"us_per_frame": round(us, 1), "frames": nominal_frames(height), "emulated_s": round(emu, 4),
            "est_console_s": round(FACTOR * emu, 4), "skip_share": round(skip, 4), "escape_share": ESC_SHARE,
            "terms_us": {"backbone_and_gate": g1, "level1_and_escape": round(l1, 1), "rare_rows": round(rare, 1)}}


def stage1(a):
    proto = json.loads((ROOT / "eval/vocr6_psp_protocol.json").read_text())
    sweep = json.loads(Path(a.sweep).read_text())
    out = {"kind": "vocr6-psp-stage1", "protocol_sha256": None, "arms": {}}
    for arm, spec in proto["arms"].items():
        if arm == "Q":
            continue
        rs = runs_of(a.runs, arm)
        if len(rs) != len(spec["seeds"]):
            raise SystemExit(f"{arm}: {len(rs)} runs, the protocol has {len(spec['seeds'])} seeds")
        h = int(spec["train2_args"].split("--height")[1].split()[0])
        sc = sum(r["screening_score"] for r in rs) / len(rs)
        blank = sum(r["blank_share_dev_wd"] for r in rs) / len(rs)
        est = est_box(sweep, arm, h, blank)
        out["arms"][arm] = {"seeds": [r["seed"] for r in rs], "SC": round(sc, 4),
                            "SC_per_seed": [r["screening_score"] for r in rs],
                            "hp_grad": [r["hval_cer"]["page_grad_refine"] for r in rs],
                            "hl": [r["hval_cer"]["clean_lines"] for r in rs],
                            "wd_grad": [r["dev_cer"]["native_psp_wd_grad_refine"] for r in rs],
                            "fixed": [r["dev_cer"]["native_psp_fixed_dialogue"] for r in rs],
                            "blank_share_dev_wd": round(blank, 4), "est": est,
                            "feasible": est["est_console_s"] <= proto["budgets"]["time_s_le"] and arm != "B32",
                            "blobs": [r["blob_sha256"] for r in rs]}
    feas = [(v["SC"], v["est"]["est_console_s"], k) for k, v in out["arms"].items() if v["feasible"]]
    if not feas:
        out["winner"] = None
    else:
        best = min(feas)
        near = [f for f in feas if f[0] - best[0] < 0.05]
        out["winner"] = min(near, key=lambda f: (f[1], f[0]))[2]
        out["lowest_SC"] = best[2]
    out["protocol_sha256"] = subprocess.run(["git", "hash-object", "eval/vocr6_psp_protocol.json"], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip()
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def stage2(a):
    cand, win = runs_of(a.runs, a.cand), runs_of(a.runs, a.winner)
    if not cand or not win:
        raise SystemExit("no runs")
    tmp = Path(a.out).with_suffix(".cmp.json")
    py = str(ROOT / ".venv/bin/python")
    res = {}
    for plan in ("dev", "hval"):
        subprocess.run([py, str(ROOT / "eval/modelb_eval.py"), "compare", "--cand",
                        *[str(Path(r["blob"]).parent / f"{plan}.json") for r in cand], "--base",
                        *[str(Path(r["blob"]).parent / f"{plan}.json") for r in win], "--out",
                        str(tmp.with_suffix(f".{plan}.json"))], check=True, capture_output=True)
        res[plan] = json.loads(tmp.with_suffix(f".{plan}.json").read_text())["components"]
    sc_c = sum(r["screening_score"] for r in cand) / len(cand)
    sc_w = sum(r["screening_score"] for r in win) / len(win)
    worse = [c for c in ("wd_grad", "fixed") if res["dev"][c]["vn_ci95"] and res["dev"][c]["vn_ci95"][0] > 0]
    out = {"kind": "vocr6-psp-stage2", "cand": a.cand, "winner": a.winner, "SC_cand": round(sc_c, 4),
           "SC_winner": round(sc_w, 4), "dev": res["dev"], "hval": res["hval"], "resolved_worse": worse,
           "adopted": sc_c < sc_w and not worse}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("cand", "winner", "SC_cand", "SC_winner", "resolved_worse", "adopted")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("stage1")
    s1.add_argument("--runs", required=True)
    s1.add_argument("--sweep", required=True)
    s1.add_argument("--out", required=True)
    s2 = sub.add_parser("stage2")
    s2.add_argument("--runs", required=True)
    s2.add_argument("--winner", required=True)
    s2.add_argument("--cand", required=True)
    s2.add_argument("--out", required=True)
    a = ap.parse_args()
    {"stage1": stage1, "stage2": stage2}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

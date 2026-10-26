#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): the protocol's "head" decision for one run (eval/vocr6_psp_protocol.json decisions.head).

  vocr6_psp_head.py RUN_DIR [--lines 12000]

1. the run's int8 blob with its head in the model B frequency order (train/modelb_head.py permute; rank_psp.json)
2. W8F features of training-side console crops (the console crops of the hooked PSP train pages and VOCR5's native
   PSP GRAD / refined crops; a fixed sample of --lines rows) and the gate + escape fit (train/modelb_head.py fit)
3. reads of the dev and hval plans: the permuted plain blob (W8F), the gate alone (W8F, head_gate on) and the gate
   with the two-level head (W8F, head_gate on, two_level 1)
4. the rule: gated + two-level minus plain on dev wd_grad and fixed, paired VN-cluster 95% intervals with upper
   bounds at most +0.05 points: keep both; else the gate alone under the same rule; else neither (not time-feasible)
Writes RUN_DIR/head/decision.json and logs it in vocr6/experiments.jsonl.
"""
import argparse
import datetime
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts/vocr6-psp")
P = Path("/local/home/skerraut/herd-vita-ocr/shared/private/vocr6-psp/pools/native")
EXPLOG = Path("/local/home/skerraut/herd-synth3/vocr6/experiments.jsonl")
UPPER = 0.05
PY = str(ROOT / ".venv/bin/python")


def run(*args):
    print("+", " ".join(map(str, args))[:300], flush=True)
    subprocess.run([PY, *map(str, args)], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--lines", type=int, default=12000)
    a = ap.parse_args()
    rd = Path(a.run_dir)
    rec = json.loads((rd / "run_record.json").read_text())
    blob = Path(rec["blob"])
    hd = rd / "head"
    hd.mkdir(exist_ok=True)
    perm = hd / "permuted.vocr"
    run(ROOT / "train/modelb_head.py", "permute", "--blob", blob, "--rank", A / "rank_psp.json", "--out", perm)
    srcs = [str(A / "pools/console/train/*.npz")] + [str(A / f"pools/native/{n}_psp.npz") for n in
                                                     ("nat_grad", "nat_ref", "dntv_grad", "dntv_ref", "dnst_grad",
                                                      "dnst_ref")] + \
        [str(P / f"{n}_psp.npz") for n in ("own_grad", "own_ref")]
    feats = hd / "features.npz"
    run(ROOT / "train/modelb_head.py", "features", "--blob", perm, "--sources", *srcs, "--lines", a.lines,
        "--out", feats)
    run(ROOT / "train/modelb_head.py", "fit", "--blob", perm, "--features", feats, "--out", hd)
    reads = {"plain": (perm, []), "gate": (hd / "gated.vocr", ["--head-gate", "on"]),
             "gate2": (hd / "gated2.vocr", ["--head-gate", "on", "--two-level"])}
    for name, (b, opts) in reads.items():
        for plan in ("dev", "hval"):
            run(ROOT / "eval/modelb_eval.py", "read", "--plan", A / f"eval/{plan}_plan.pkl", "--blob", b, "--mode",
                "w8f", *opts, "--out", hd / f"{plan}_{name}.json")
    # the permuted plain blob reads exactly as the run's own blob (a permutation changes no logit)
    same = (json.loads((hd / "dev_plain.json").read_text())["totals"] ==
            json.loads((rd / "dev.json").read_text())["totals"])
    out = {"run": rd.name, "fit": json.loads((hd / "fit.json").read_text()), "permuted_reads_as_run": same, "arms": {}}
    for name in ("gate2", "gate"):
        for plan in ("dev", "hval"):
            run(ROOT / "eval/modelb_eval.py", "compare", "--cand", hd / f"{plan}_{name}.json", "--base",
                hd / f"{plan}_plain.json", "--out", hd / f"cmp_{plan}_{name}.json")
        cmp_dev = json.loads((hd / f"cmp_dev_{name}.json").read_text())["components"]
        cmp_hv = json.loads((hd / f"cmp_hval_{name}.json").read_text())["components"]
        ok = all(cmp_dev[c]["vn_ci95"] is not None and cmp_dev[c]["vn_ci95"][1] <= UPPER
                 for c in ("wd_grad", "fixed"))
        tot = json.loads((hd / f"dev_{name}.json").read_text())["totals"]
        out["arms"][name] = {"passes": ok, "dev": {c: cmp_dev[c] for c in ("wd_grad", "fixed", "wd_recovery")},
                             "hval": {c: cmp_hv[c] for c in ("hp_grad", "hl")},
                             "gated_share_dev_wd": tot["wd_grad"]["gated_share"],
                             "escaped_share_of_kept_dev_wd": tot["wd_grad"]["escaped_share_of_kept"]}
    out["kept"] = "gate2" if out["arms"]["gate2"]["passes"] else "gate" if out["arms"]["gate"]["passes"] else None
    (hd / "decision.json").write_text(json.dumps(out, indent=1) + "\n")
    log = {"label": "vocr6-psp", "arm": f"head({rd.name})", "tier": "modelB", "seed": rec["seed"],
           "config_hash": out["fit"]["blobs"]["gated2"]["sha256"][:16],
           "data_manifest_hash": out["fit"]["features_sha256"][:16],
           "steps": 0, "wall_clock_s": None, "slot_hours": None, "checkpoint": str(hd / "gated2.vocr"),
           "hval_cer": {k: out["arms"]["gate2"]["hval"][k]["cand_cer"] for k in ("hp_grad", "hl")},
           "dev_cer": {k: out["arms"]["gate2"]["dev"][k]["cand_cer"] for k in ("wd_grad", "fixed")},
           "notes": "gate + two-level (W8F) vs plain: kept={}; dev wd {} {}, fixed {} {}".format(
               out["kept"], out["arms"]["gate2"]["dev"]["wd_grad"]["delta"],
               out["arms"]["gate2"]["dev"]["wd_grad"]["vn_ci95"], out["arms"]["gate2"]["dev"]["fixed"]["delta"],
               out["arms"]["gate2"]["dev"]["fixed"]["vn_ci95"]),
           "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(EXPLOG, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(log, ensure_ascii=False) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)
    print(json.dumps({"run": rd.name, "kept": out["kept"], "permuted_reads_as_run": same,
                      **{n: {"passes": v["passes"], "wd": v["dev"]["wd_grad"]["vn_ci95"],
                             "fixed": v["dev"]["fixed"]["vn_ci95"], "gated": v["gated_share_dev_wd"],
                             "esc": v["escaped_share_of_kept_dev_wd"]} for n, v in out["arms"].items()}}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""textspot6's stage-2 teacher (issue #128; eval/textspot6_protocol.json "stages", amendment 5), applied once after
stage 1's M arms are scored, before any stage-2 run:

  textspot6_teacher.py --eval $A/vocr6-spotter/eval --runs $A/vocr6-spotter/runs \
      --out eval/textspot6_teacher_<date>.json

The rule: the teacher is the stage-1 M arm (base_qS4 or base_qL5) whose representative has the lower unseen-dev
page-path dialogue CER in any of its four decodes (ties: the smaller shape); a representative is the median seed of
an (arm, decode) as the selection takes it (eval/textspot6_select.py: sorted by unseen-dev CER, then hval CER, then
seed; index n // 2). The teacher model is that representative seed's final.pt. The student is kd_t64n8c16, seeds
1-3 (amendment 4).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))
import textspot6_select as SEL  # noqa: E402

M_ARMS = ("base_qS4", "base_qL5")          # smaller shape first (the tie-break)
SEEDS = (1, 2, 3)
SEEN = ("v12188",)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def representative(ev, arm, dec, seeds=SEEDS):
    rows = []
    for s in seeds:
        d = Path(ev) / f"{arm}_seed{s}"
        if not (d / "SCORED").exists():
            raise SystemExit(f"{d}: not scored")
        dev, hval = SEL.load_rows(d / f"dev_{dec}.json", SEEN), SEL.load_rows(d / f"hval_{dec}.json")
        rows.append({"seed": s, "dev_cer": SEL.cer(dev), "hval_cer": SEL.cer(hval),
                     "dev_sha256": sha256_file(d / f"dev_{dec}.json")})
    order = sorted(rows, key=lambda r: (r["dev_cer"], r["hval_cer"], r["seed"]))
    return order[len(order) // 2], order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    table, best = {}, {}
    for arm in M_ARMS:
        table[arm] = {}
        for dec in SEL.DECODES:
            rep, order = representative(a.eval, arm, dec)
            table[arm][dec] = {"representative_seed": rep["seed"], "dev_cer": rep["dev_cer"],
                               "hval_cer": rep["hval_cer"], "seeds": order}
        dec = min(SEL.DECODES, key=lambda k: (table[arm][k]["dev_cer"], SEL.DECODES.index(k)))
        best[arm] = (table[arm][dec]["dev_cer"], dec, table[arm][dec]["representative_seed"])
    arm = min(M_ARMS, key=lambda k: (best[k][0], M_ARMS.index(k)))
    cer, dec, seed = best[arm]
    run = Path(a.runs) / f"{arm}_seed{seed}"
    pt = run / "final.pt"
    out = {"kind": "textspot6-teacher", "computed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rule": "amendment 5", "seen_dev_clusters": list(SEEN), "table": table,
           "best": {k: {"dev_cer": v[0], "decode": v[1], "representative_seed": v[2]} for k, v in best.items()},
           "teacher": {"arm": arm, "decode": dec, "seed": seed, "dev_cer": cer, "run": run.name,
                       "final_pt_sha256": sha256_file(pt)},
           "student": {"arm": "kd_t64n8c16", "seeds": list(SEEDS)},
           "protocol_sha256": sha256_file(ROOT / "eval" / "textspot6_protocol.json")}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for k, v in out["best"].items():
        print(f"{k}: best representative {v['dev_cer']}% ({v['decode']}, seed {v['representative_seed']})")
    print(f"teacher: {arm} seed {seed} ({dec}, {cer}%) -> {run.name}/final.pt")
    return out


if __name__ == "__main__":
    main()

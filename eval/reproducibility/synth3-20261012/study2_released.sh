#!/usr/bin/env bash
# study2_released.sh: the released FL10/H15 blobs scored W8I on teacher_val (development set), the paired
# released-vs-arm intervals and the 0.25M-2M learning-curve interval cited in docs/synth3.md (study 2).
# A reconstruction of the commands run by hand on 2026-10-13 (heavy.log 15:0xZ); it reproduces the numbers.
set -euo pipefail
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; S3=$A/synth3; SC=$S3/study2/scores
cd "$(dirname "$0")/../../.."
[ -s $SC/released_teacher_val.json ] || .venv-ci/bin/python eval/synth3_ablation.py score \
  --blob REF_FL10_v020=$A/train2/release/FL10_w8.vocr --blob REF_H15_v020=$A/train2/release/H15_w8.vocr \
  --dev $A/teacher/teacher_val.npz --out $SC/released_teacher_val.json
.venv-ci/bin/python - "$SC" "$S3" <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "eval")
import synth3_ablation as ABL
SC, S3 = sys.argv[1], sys.argv[2]
rel = json.load(open(f"{SC}/released_teacher_val.json"))
chars = np.array([len(x["ref"]) for x in rel["lines"]], float); vns = np.array([x["vn"] for x in rel["lines"]])
def mean_err(s, runs):   # per-line errors averaged over runs and the two views
    return sum(np.asarray(s["models"][r]["errors"][v], float) for r in runs for v in ("rv", "rvn")) / (2 * len(runs))
out = {}
for arch, seeds, ref in (("FL10", (1, 2, 3), "REF_FL10_v020"), ("H15", (1, 2), "REF_H15_v020")):
    s = json.load(open(f"{SC}/{arch}_teacher_val.json"))
    assert [x["id"] for x in s["lines"]] == [x["id"] for x in rel["lines"]]
    for arm in ("A_s2", "C_mix"):
        b = ABL.bootstrap(mean_err(rel, [ref]), mean_err(s, [f"ft_{arch}_{arm}_seed{i}" for i in seeds]), chars, vns, 10000, 0, 1.0)
        out[f"{arch}_{arm}_minus_released_W8I"] = b
        print(arch, arm, "- released:", b["delta_points"], b["ci95"], "released", f"{100*b['cer_a']:.2f}%", "arm", f"{100*b['cer_b']:.2f}%")
cv = json.load(open(f"{S3}/curve/scores/curve_teacher_val.json"))
assert [x["id"] for x in cv["lines"]] == [x["id"] for x in rel["lines"]]
b = ABL.bootstrap(mean_err(cv, ["B_s3_025M_seed1", "B_s3_025M_seed2"]), mean_err(cv, ["B_s3_2M_seed1", "B_s3_2M_seed2"]), chars, vns, 10000, 0, 1.0)
out["curve_2M_minus_025M"] = b
print("curve 2M - 0.25M:", b["delta_points"], b["ci95"])
json.dump({"kind": "released-vs-study2-arms", "note": "teacher_val (development set), C W8I both views; per-line errors averaged over seeds and views; VN-cluster bootstrap 10000 resamples, seed 0; curve_2M_minus_025M is the end-to-end learning-curve interval", "comparisons": out}, open(f"{SC}/released_vs_arms.json", "w"), indent=1)
PY

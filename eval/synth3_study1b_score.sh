#!/usr/bin/env bash
# synth3_study1b_score.sh: eval/synth3_study1b_protocol.json's scoring, run once all four ckpt_step16000.pt exist.
# Exports the EMA step-16000 checkpoints (per-channel int8), reads teacher_val (decides) and devnative's four files
# (reported) through the C runtime in W8I, and applies the protocol's comparison. Through heavy.sh:
#   HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 bash eval/synth3_study1b_score.sh < /dev/null
set -euo pipefail
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
AB=$A/synth3-v3/ablation
PROTO=eval/synth3_study1b_protocol.json
cd "$REPO"
echo "git $(git rev-parse HEAD)"
mkdir -p "$AB/blobs" "$AB/scores" "$AB/compare"
BLOBS=()
for r in B_s3 D_s3v3; do for s in 1 2; do
  ck=$AB/runs/${r}_seed$s/ckpt_step16000.pt
  [ -f "$ck" ] || { echo "missing $ck"; exit 3; }
  .venv-ci/bin/python eval/synth3_ablation.py export --ckpt "$ck" --out "$AB/blobs/${r}_seed${s}_w8.vocr"
  BLOBS+=(--blob "${r}_seed$s=$AB/blobs/${r}_seed${s}_w8.vocr")
done; done
.venv-ci/bin/python eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$A/teacher/teacher_val.npz" \
  --out "$AB/scores/teacher_val.json"
.venv-ci/bin/python eval/synth3_ablation.py compare --scores "$AB/scores/teacher_val.json" --protocol "$PROTO" \
  --out "$AB/compare/teacher_val.json"
DN=()
for f in strict_ink strict_grad tv_native_ink tv_native_grad; do
  .venv-ci/bin/python eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$A/devnative/devnative_$f.npz" \
    --out "$AB/scores/devnative_$f.json"
  .venv-ci/bin/python eval/synth3_ablation.py compare --scores "$AB/scores/devnative_$f.json" --protocol "$PROTO" \
    --out "$AB/compare/devnative_$f.json"
  DN+=("$AB/scores/devnative_$f.json")
done
.venv-ci/bin/python eval/synth3_study2.py merge --scores "${DN[@]}" --out "$AB/scores/devnative_pooled.json"
.venv-ci/bin/python eval/synth3_ablation.py compare --scores "$AB/scores/devnative_pooled.json" --protocol "$PROTO" \
  --out "$AB/compare/devnative_pooled.json"
AB="$AB" PROTO="$PROTO" .venv-ci/bin/python - <<'PY'
import datetime, hashlib, json, os, subprocess
from pathlib import Path
ab, proto = Path(os.environ["AB"]), os.environ["PROTO"]
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
tv = load(ab / "compare" / "teacher_val.json")
dn = {f: load(ab / "compare" / f"devnative_{f}.json") for f in
      ("strict_ink", "strict_grad", "tv_native_ink", "tv_native_grad", "pooled")}
c = tv["comparisons"]["D_vs_B"]
study1 = load("eval/synth3_ablation_20261012.json")["models"]
out = {"kind": "synth3-study1b-result",
       "protocol": {"file": proto, "sha256": hashlib.sha256(Path(proto).read_bytes()).hexdigest()},
       "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
       "scored": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "decision": {"d_better": c["decision"]["b_better"], "rule": c["decision"]["rule"],
                    "delta_points": c["all_vn_cluster"]["delta_points"], "ci95": c["all_vn_cluster"]["ci95"],
                    "per_seed": {k: v["delta_points"] for k, v in c["per_seed"].items()},
                    "mirrored_b_better": bool(c["all_vn_cluster"]["ci95"][0] > 0
                                              and all(v["delta_points"] > 0 for v in c["per_seed"].values()))},
       "teacher_val": tv,
       "devnative_reported_only": dn,
       "study1_B_s3_on_the_32_core_box": {k: v for k, v in study1.items() if k.startswith("B_s3")}}
Path("eval/synth3_study1b_20261012.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                                     encoding="utf-8")
print(json.dumps(out["decision"], indent=1))
PY
echo SCORED

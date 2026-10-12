#!/usr/bin/env bash
# Study 1 scoring per eval/synth3_ablation_protocol.json: export the six step-16000 checkpoints,
# score each on teacher_val through C W8I (one process per blob), then the protocol comparison.
set -euo pipefail
cd /local/home/skerraut/herd-vita-oct12/clones/synth-dataset
S3=/local/home/skerraut/herd-vita-ocr/shared/artifacts/synth3
DEV=/local/home/skerraut/herd-vita-ocr/shared/artifacts/teacher/teacher_val.npz
mkdir -p "$S3/ablation/blobs" "$S3/ablation/scores"
echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ)"
for r in A_s2 B_s3 C_mix; do for s in 1 2; do
  .venv-ci/bin/python eval/synth3_ablation.py export --ckpt "$S3/ablation/runs/${r}_seed$s/ckpt_step16000.pt" \
    --out "$S3/ablation/blobs/${r}_seed${s}_w8.vocr"
done; done
pids=()
for r in A_s2 B_s3 C_mix; do for s in 1 2; do
  .venv-ci/bin/python eval/synth3_ablation.py score --blob "${r}_seed$s=$S3/ablation/blobs/${r}_seed${s}_w8.vocr" \
    --dev "$DEV" --out "$S3/ablation/scores/${r}_seed$s.json" &
  pids+=($!)
done; done
for p in "${pids[@]}"; do wait "$p"; done
.venv-ci/bin/python eval/synth3_ablation.py compare --scores "$S3"/ablation/scores/*.json \
  --protocol eval/synth3_ablation_protocol.json --out eval/synth3_ablation_20261012.json
echo "SCORE_DONE $(date -u +%FT%TZ)"

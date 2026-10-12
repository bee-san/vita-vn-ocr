#!/usr/bin/env bash
set -euo pipefail
HERD=/local/home/skerraut/herd-vita-ocr; S3=$HERD/shared/artifacts/synth3
cd /local/home/skerraut/herd-vita-oct12/clones/synth-dataset
echo "git $(git rev-parse HEAD)"
for a in A_s2 B_s3; do
  .venv-ci/bin/python eval/synth3_ablation.py export --ckpt $S3/ablation/runs/${a}_w15_seed1/ckpt_step16000.pt --out $S3/ablation/blobs/${a}_w15_seed1_w8.vocr
done
.venv-ci/bin/python eval/synth3_ablation.py score --blob A_s2_w15_seed1=$S3/ablation/blobs/A_s2_w15_seed1_w8.vocr \
  --blob B_s3_w15_seed1=$S3/ablation/blobs/B_s3_w15_seed1_w8.vocr --dev $HERD/shared/artifacts/teacher/teacher_val.npz \
  --out $S3/ablation/scores/secondary_h15.json
echo SCORE_H15_FINISHED

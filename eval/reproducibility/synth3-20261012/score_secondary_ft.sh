#!/usr/bin/env bash
set -euo pipefail
HERD=/local/home/skerraut/herd-vita-ocr; S3=$HERD/shared/artifacts/synth3
cd /local/home/skerraut/herd-vita-oct12/clones/synth-dataset
echo "git $(git rev-parse HEAD)"
for a in A_s2 B_s3 C_mix; do
  .venv-ci/bin/python eval/synth3_ablation.py export --ckpt $S3/ablation/runs/ft_${a}_seed1/ckpt_step5000.pt --out $S3/ablation/blobs/ft_${a}_seed1_w8.vocr
done
.venv-ci/bin/python eval/synth3_ablation.py score --blob ft_A_s2_seed1=$S3/ablation/blobs/ft_A_s2_seed1_w8.vocr \
  --blob ft_B_s3_seed1=$S3/ablation/blobs/ft_B_s3_seed1_w8.vocr --blob ft_C_mix_seed1=$S3/ablation/blobs/ft_C_mix_seed1_w8.vocr \
  --dev $HERD/shared/artifacts/teacher/teacher_val.npz --out $S3/ablation/scores/secondary_ft.json
echo SCORE_FT_FINISHED

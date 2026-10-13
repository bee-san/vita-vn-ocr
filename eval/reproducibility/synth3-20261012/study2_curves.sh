#!/usr/bin/env bash
# study2_curves.sh: study 2's reported-only learning curve and pretrain-length (step) curve: exports, teacher_val
# scores and eval/synth3_study2.py result with the reported spec -> $S3/study2/FL10_curves_result.json.
# Learning-curve exports are never scored on a test set (protocol); the step-curve exports are scored once on the
# test sets by a separate step after this result is committed.
set -euo pipefail
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; S3=$A/synth3; export HERD
REPO=/local/home/skerraut/herd-vita-oct12/clones/study2-wt; PY=$REPO/.venv-ci/bin/python
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ)"
mkdir -p $S3/curve/blobs $S3/curve/scores
CV=()
for r in B_s3_025M_seed1 B_s3_025M_seed2 B_s3_050M_seed1 B_s3_050M_seed2 B_s3_2M_seed1 B_s3_2M_seed2; do
  [ -s $S3/curve/blobs/${r}_w8.vocr ] || $PY eval/synth3_ablation.py export --ckpt $S3/curve/runs/$r/ckpt_step16000.pt --out $S3/curve/blobs/${r}_w8.vocr | tail -n 1
  CV+=(--blob "$r=$S3/curve/blobs/${r}_w8.vocr")
done
$PY eval/synth3_ablation.py score "${CV[@]}" --dev $A/teacher/teacher_val.npz --out $S3/curve/scores/curve_teacher_val.json | tail -n 6
ST=()
for spec in pre_FL10_C_mix_pre64k_seed1:64000 ft_FL10_C_mix_pre64k_seed1:15000 pre_FL10_C_mix_pre128k_seed1:128000 ft_FL10_C_mix_pre128k_seed1:15000; do
  r=${spec%%:*}; step=${spec#*:}
  [ -s $S3/study2/blobs/${r}_w8.vocr ] || $PY eval/synth3_ablation.py export --ckpt $S3/study2/runs/$r/ckpt_step$step.pt --out $S3/study2/blobs/${r}_w8.vocr | tail -n 1
  ST+=(--blob "$r=$S3/study2/blobs/${r}_w8.vocr")
done
$PY eval/synth3_ablation.py score "${ST[@]}" --dev $A/teacher/teacher_val.npz --out $S3/study2/scores/FL10_step_teacher_val.json | tail -n 4
SC=$S3/study2/scores
$PY eval/synth3_study2.py result --protocol eval/synth3_study2_protocol.json --tv $SC/FL10_teacher_val.json \
  --dn $SC/FL10_devnative_decision_set.json --dn-commit f5a1dbd0ad15317e73e1b1e030eb88acf3d79861 \
  --reported eval/synth3_study2_reported.json \
  --reported-scores $SC/FL10_teacher_val.json $SC/FL10_step_teacher_val.json $S3/curve/scores/curve_teacher_val.json \
                    $S3/ablation/scores/B_s3_seed1.json $S3/ablation/scores/B_s3_seed2.json \
  --out $S3/study2/FL10_curves_result.json
echo "CURVES_RESULT_WRITTEN $(date -u +%FT%TZ)"

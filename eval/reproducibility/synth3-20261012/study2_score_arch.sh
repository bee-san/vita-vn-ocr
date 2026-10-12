#!/usr/bin/env bash
# score_arch.sh ARCH: export every study-2 run of ARCH (pretrain final + fine-tune final) and score them on
# teacher_val, and on devnative's npz files when they exist (co-primary or not is decided by the caller).
set -euo pipefail
ARCH=$1
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; S3=$A/synth3; R=$S3/study2/runs
B=$S3/study2/blobs; SC=$S3/study2/scores; DN=$A/devnative
cd /local/home/skerraut/herd-vita-oct12/clones/synth-dataset
echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ)"
case $ARCH in FL10) PRE=32000; SEEDS="1 2 3" ;; H15) PRE=40000; SEEDS="1 2" ;; esac
BLOBS=()
for arm in A_s2 C_mix; do for s in $SEEDS; do
  for kind in pre ft; do
    run=${kind}_${ARCH}_${arm}_seed$s
    ck=$R/$run/ckpt_step$([ $kind = pre ] && echo $PRE || echo 15000).pt
    [ -s "$B/${run}_w8.vocr" ] || .venv-ci/bin/python eval/synth3_ablation.py export --ckpt "$ck" --out "$B/${run}_w8.vocr"
    BLOBS+=(--blob "$run=$B/${run}_w8.vocr")
  done
done; done
.venv-ci/bin/python eval/synth3_ablation.py score "${BLOBS[@]}" --dev $A/teacher/teacher_val.npz --out $SC/${ARCH}_teacher_val.json
for f in $DN/devnative_*_ink.npz $DN/devnative_*_grad.npz; do
  [ -e "$f" ] || continue
  .venv-ci/bin/python eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$f" --out $SC/${ARCH}_$(basename "$f" .npz).json
done
echo "SCORE_ARCH_FINISHED $(date -u +%FT%TZ)"

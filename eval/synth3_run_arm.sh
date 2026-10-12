#!/usr/bin/env bash
# run_arm.sh ARM SEED [WIDTH VOCAB]: one primary run of eval/synth3_ablation_protocol.json (train2.py, 16k steps)
set -euo pipefail
ARM=$1; SEED=$2; WIDTH=${3:-1.0}; VOCAB=${4:-/home/skerraut/herd-vita-ocr/shared/artifacts/train2/release/FL10_vocab_3500.txt}
A=/home/skerraut/herd-vita-ocr/shared/artifacts
REPO=/home/skerraut/herd-vita-oct12/clones/synth-dataset
OUT=$A/synth3/ablation/runs
NV="nv:0.5:$A/nvsynth/shards/nv_train_0*.npz:stream=1,pool=3,inv=0.2"
S2G="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
S3G="$A/synth3/shards/train_v2/synth3_000??.npz"
case $ARM in
  A_s2)  SRC=(--src "$NV" --src "s2:0.5:$S2G:stream=1,pool=3,inv=0.2") ;;
  B_s3)  SRC=(--src "$NV" --src "s3:0.5:$S3G:stream=1,pool=3,inv=0.2") ;;
  C_mix) SRC=(--src "$NV" --src "s2:0.25:$S2G:stream=1,pool=3,inv=0.2" --src "s3:0.25:$S3G:stream=1,pool=3,inv=0.2") ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
TAG=""; [ "$WIDTH" != "1.0" ] && TAG="_w${WIDTH/./}"
RUN=${ARM}${TAG}_seed${SEED}
cd "$REPO"
echo "git $(git rev-parse HEAD) run $RUN"
exec .venv-ci/bin/python train/train2.py --run "$RUN" --out "$OUT" --steps 16000 --batch 96 --lr 2e-3 --warmup 1000 \
  --final-lr 0.05 --wd 0.01 --ema 0.999 --width "$WIDTH" --vocab "$VOCAB" --seed "$SEED" --threads 5 \
  --eval-every 4000 --keep-evals "${SRC[@]}" \
  --val "rv:$A/teacher/teacher_val.npz" --val "rvn:$A/teacher/teacher_val.npz:native=1" \
  --val "nvv:$A/nvsynth/shards/nv_val.npz:n=2000" --val "s2v:$A/synth2/shards/synth_00039.npz:n=2000" \
  --select rv,rvn

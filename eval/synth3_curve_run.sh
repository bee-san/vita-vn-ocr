#!/usr/bin/env bash
# synth3_curve_run.sh SIZE SEED: one learning-curve run of eval/synth3_study2_protocol.json ("learning_curve"):
# study 1's B_s3 run (eval/synth3_run_arm.sh: train2.py, 16k steps, nvsynth one-line set 0.5 + synth3 0.5)
# with the synth3 source restricted to the first 0.25M or 0.5M crops of the v2 render, or extended to 2M with
# the pinned v2 extension (shards 100-199 of the same generator). The 1M point is study 1's B_s3 runs.
set -euo pipefail
SIZE=$1; SEED=$2
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$A/synth3/curve/runs
V2=$A/synth3/shards/train_v2
case $SIZE in
  025M) S3G="$V2/synth3_000[01]?.npz+$V2/synth3_0002[0-4].npz" ;;
  050M) S3G="$V2/synth3_000[0-4]?.npz" ;;
  2M)   S3G="$V2/synth3_000??.npz+$A/synth3/shards/train_v2_ext/synth3_001??.npz" ;;
  *) echo "unknown size $SIZE"; exit 2 ;;
esac
RUN=B_s3_${SIZE}_seed${SEED}
cd "$REPO"
echo "git $(git rev-parse HEAD) run $RUN"
if [ -s "$OUT/$RUN/ckpt_step16000.pt" ]; then echo "$RUN is complete"; exit 0; fi
if [ -e "$OUT/$RUN" ]; then echo "$OUT/$RUN is partial: delete it and restart (runs never resume)"; exit 3; fi
exec .venv-ci/bin/python train/train2.py --run "$RUN" --out "$OUT" --steps 16000 --batch 96 --lr 2e-3 --warmup 1000 \
  --final-lr 0.05 --wd 0.01 --ema 0.999 --width 1.0 --vocab "$A/train2/release/FL10_vocab_3500.txt" --seed "$SEED" \
  --threads 5 --eval-every 4000 --keep-evals \
  --src "nv:0.5:$A/nvsynth/shards/nv_train_0*.npz:stream=1,pool=3,inv=0.2" --src "s3:0.5:$S3G:stream=1,pool=3,inv=0.2" \
  --val "rv:$A/teacher/teacher_val.npz" --val "rvn:$A/teacher/teacher_val.npz:native=1" \
  --val "nvv:$A/nvsynth/shards/nv_val.npz:n=2000" --val "s2v:$A/synth2/shards/synth_00039.npz:n=2000" \
  --select rv,rvn

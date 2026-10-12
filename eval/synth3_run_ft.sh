#!/usr/bin/env bash
# synth3_run_ft.sh ARM: the protocol's secondary fine-tune (eval/synth3_ablation_protocol.json "secondary.finetune"):
# 5,000 steps from ARM's seed-1 step-16000 checkpoint on nv + ARM's synthetic data + teacher_train + real_train.
set -euo pipefail
ARM=$1
A=/home/skerraut/herd-vita-ocr/shared/artifacts
REPO=/home/skerraut/herd-vita-oct12/clones/synth-dataset
OUT=$A/synth3/ablation/runs
VOCAB=$A/train2/release/FL10_vocab_3500.txt
NV="nv:0.25:$A/nvsynth/shards/nv_train_0*.npz:stream=1,pool=3,inv=0.2"
S2G="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
S3G="$A/synth3/shards/train_v2/synth3_000??.npz"
case $ARM in
  A_s2)  SYN=(--src "s2:0.25:$S2G:stream=1,pool=3,inv=0.2") ;;
  B_s3)  SYN=(--src "s3:0.25:$S3G:stream=1,pool=3,inv=0.2") ;;
  C_mix) SYN=(--src "s2:0.125:$S2G:stream=1,pool=3,inv=0.2" --src "s3:0.125:$S3G:stream=1,pool=3,inv=0.2") ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
cd "$REPO"
echo "git $(git rev-parse HEAD) run ft_${ARM}_seed1"
exec .venv-ci/bin/python train/train2.py --run "ft_${ARM}_seed1" --out "$OUT" --init "$OUT/${ARM}_seed1/ckpt_step16000.pt" \
  --steps 5000 --batch 96 --lr 1.2e-3 --warmup 500 --final-lr 0.05 --wd 0.01 --ema 0.999 --width 1.0 --vocab "$VOCAB" \
  --seed 1 --threads 5 --eval-every 2500 --keep-evals --src "$NV" "${SYN[@]}" \
  --src "tt:0.35:$A/teacher/teacher_train.npz:inv=0.5" --src "rd:0.15:$A/realdata/real_train.npz:inv=0.5" \
  --val "rv:$A/teacher/teacher_val.npz" --val "rvn:$A/teacher/teacher_val.npz:native=1" --select rv,rvn

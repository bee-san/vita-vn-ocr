#!/usr/bin/env bash
# synth3_study1b_run.sh ARM SEED: one run of eval/synth3_study1b_protocol.json (study 1's recipe: train2.py, 16k steps,
# FL10, nv 0.5 + VN-style synthetic 0.5). ARM: B_s3 (generator version 2, as in study 1) or D_s3v3 (version 3).
# Launch: HEAVY_SLOTS=10 $HERD/heavy.sh timeout 36000 bash eval/synth3_study1b_run.sh ARM SEED < /dev/null
set -euo pipefail
ARM=$1; SEED=$2
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$A/synth3-v3/ablation/runs
VOCAB=$A/train2/release/FL10_vocab_3500.txt
NV="nv:0.5:$A/nvsynth/shards/nv_train_0*.npz:stream=1,pool=3,inv=0.2"
S3G="$A/synth3/shards/train_v2/synth3_000??.npz"
S3V3G="$A/synth3-v3/shards/train_v3/synth3_000??.npz"
case $ARM in
  B_s3)   SRC=(--src "$NV" --src "s3:0.5:$S3G:stream=1,pool=3,inv=0.2") ;;
  D_s3v3) [ -f "$REPO/data/manifests/synth3_shards_v3.json" ] || { echo "v3 shards are not pinned yet"; exit 3; }
          SRC=(--src "$NV" --src "s3v3:0.5:$S3V3G:stream=1,pool=3,inv=0.2") ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
RUN=${ARM}_seed${SEED}
mkdir -p "$OUT"
cd "$REPO"
echo "git $(git rev-parse HEAD) run $RUN"
exec .venv-ci/bin/python train/train2.py --run "$RUN" --out "$OUT" --steps 16000 --batch 96 --lr 2e-3 --warmup 1000 \
  --final-lr 0.05 --wd 0.01 --ema 0.999 --width 1.0 --vocab "$VOCAB" --seed "$SEED" --threads 5 \
  --eval-every 4000 --keep-evals "${SRC[@]}" \
  --val "rv:$A/teacher/teacher_val.npz" --val "rvn:$A/teacher/teacher_val.npz:native=1" \
  --val "nvv:$A/nvsynth/shards/nv_val.npz:n=2000" --val "s2v:$A/synth2/shards/synth_00039.npz:n=2000" \
  --select rv,rvn

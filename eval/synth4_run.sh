#!/usr/bin/env bash
# synth4_run.sh SEED: one F_s4 run of eval/synth4_protocol.json (study 1's recipe: train2.py at 2b132aa, 16k steps,
# FL10, nv 0.5 + synth4's pinned pilot shards 0.5). D_s3v3 is study 1b's runs, reused (not retrained).
# Launch: TRAIN_REPO=<worktree at 2b132aa> HEAVY_SLOTS=10 $HERD/heavy.sh timeout 36000 bash eval/synth4_run.sh SEED < /dev/null
set -euo pipefail
SEED=$1
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
WANT=2b132aaba600ab7dbceeeb1f3a71877d3202e853          # the commit D_s3v3 trained at (its logs' first line)
TRAIN=${TRAIN_REPO:?set TRAIN_REPO to a worktree at $WANT}
PY=${PY:-$REPO/.venv-ci/bin/python}                    # requirements-ci.lock is unchanged since $WANT
[ "$(git -C "$TRAIN" rev-parse HEAD)" = "$WANT" ] || { echo "TRAIN_REPO is not at $WANT"; exit 3; }
[ -f "$REPO/data/manifests/synth4_shards_v1.json" ] || { echo "synth4 shards are not pinned yet"; exit 3; }
OUT=$A/synth4-design/ablation/runs
VOCAB=$A/train2/release/FL10_vocab_3500.txt
NV="nv:0.5:$A/nvsynth/shards/nv_train_0*.npz:stream=1,pool=3,inv=0.2"
S4G="$A/synth4-design/shards/train_v1/synth4_000??.npz"
SRC=(--src "$NV" --src "s4:0.5:$S4G:stream=1,pool=3,inv=0.2")
RUN=F_s4_seed${SEED}
mkdir -p "$OUT"
cd "$TRAIN"
echo "git $(git rev-parse HEAD) run $RUN (protocol repo $(git -C "$REPO" rev-parse HEAD))"
exec "$PY" train/train2.py --run "$RUN" --out "$OUT" --steps 16000 --batch 96 --lr 2e-3 --warmup 1000 \
  --final-lr 0.05 --wd 0.01 --ema 0.999 --width 1.0 --vocab "$VOCAB" --seed "$SEED" --threads 5 \
  --eval-every 4000 --keep-evals "${SRC[@]}" \
  --val "rv:$A/teacher/teacher_val.npz" --val "rvn:$A/teacher/teacher_val.npz:native=1" \
  --val "nvv:$A/nvsynth/shards/nv_val.npz:n=2000" --val "s2v:$A/synth2/shards/synth_00039.npz:n=2000" \
  --select rv,rvn

#!/usr/bin/env bash
# One textspot8 training run (train/textspot8_train.py) on textspot6's data: the hooked page shards (train split,
# or train,guard,hval for a final model), textspot's synthetic frames and textspot2's MeikiOCR-labelled real half.
# Run it through heavy.sh (5 threads) from a worktree at the registered commit:
#   HEAVY_SLOTS=12 $HERD/heavy.sh timeout 21600 eval/reproducibility/textspot8-20261027/run_train8.sh SHAPE SEED RUNS_DIR \
#       [T8_JSON] [SPLITS]
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
PV=$HERD/shared/private
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SHAPE=${1:?shape}; SEED=${2:?seed}; RUNS=${3:?runs dir}; T8JSON=${4:-'{}'}; SPLITS=${5:-train}
NAME=t8_${SHAPE}_seed${SEED}
[ "$SPLITS" = train ] || NAME=${NAME}_final
OUT=$RUNS/$NAME
mkdir -p "$RUNS/logs"
SHARDS=("$A/textspot/shards/synth_frames" "$A/textspot2/shards/lab/lab_realplus" "$A/textspot2/shards/lab/lab_teacher"
        "$A/textspot2/shards/lab/lab_devnative" "$PV/textspot2/shards/lab/lab_owner" "$A/textspot2/shards/more/more_realplus"
        "$A/textspot2/shards/more/more_teacher" "$PV/textspot2/shards/more/more_owner")
exec "$REPO/.venv/bin/python" "$REPO/train/textspot8_train.py" --recipe t8 --shape "$SHAPE" --seed "$SEED" --out "$OUT" \
  --hook-manifest "$REPO/data/manifests/textspot6_pages_v1.json" --hook-shards "$PV/vocr6-spotter/shards_v1" \
  --hook-splits "$SPLITS" --manifest "$REPO/data/manifests/textspot_shards_v1.json" \
  "$REPO/data/manifests/textspot2_shards_v1.json" --shards "${SHARDS[@]}" --threads 5 --t8-json "$T8JSON" \
  > "$RUNS/logs/$NAME.log" 2>&1

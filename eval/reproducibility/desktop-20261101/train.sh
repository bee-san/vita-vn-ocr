#!/usr/bin/env bash
# One desktop detector run (#169, train/desktop_det.py) on pools v2.1's hooked page shards, textspot's synthetic
# frames and textspot2's MeikiOCR-labelled real half. Through heavy.sh (5 threads: 4 torch + 1 data worker):
#   HEAVY_SLOTS=12 heavy.sh timeout 43200 train.sh NAME SHAPE SEED SPLITS [extra desktop_det.py args...]
set -euo pipefail
NAME=${1:?name}; SHAPE=${2:?shape}; SEED=${3:?seed}; SPLITS=${4:?splits}; shift 4
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts; PV=$HERD/shared/private
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
RUNS=$A/vocr-desktop/runs
mkdir -p $RUNS/logs
SHARDS=("$A/textspot/shards/synth_frames" "$A/textspot2/shards/lab/lab_realplus" "$A/textspot2/shards/lab/lab_teacher"
        "$A/textspot2/shards/lab/lab_devnative" "$PV/textspot2/shards/lab/lab_owner" "$A/textspot2/shards/more/more_realplus"
        "$A/textspot2/shards/more/more_teacher" "$PV/textspot2/shards/more/more_owner")
exec "$REPO/.venv/bin/python" "$REPO/train/desktop_det.py" --shape "$SHAPE" --seed "$SEED" --out "$RUNS/$NAME" \
  --hook-manifest "$REPO/data/manifests/desktop_pages_v21.json" --hook-shards "$PV/vocr-desktop/shards_v21" \
  --hook-splits "$SPLITS" --manifest "$REPO/data/manifests/textspot_shards_v1.json" \
  "$REPO/data/manifests/textspot2_shards_v1.json" --shards "${SHARDS[@]}" --threads 4 --workers 1 "$@" \
  > "$RUNS/logs/$NAME.log" 2>&1

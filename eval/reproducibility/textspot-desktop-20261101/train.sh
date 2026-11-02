#!/usr/bin/env bash
# One textspot-desktop detector run (#173, train/tsdesk_det.py) on vocr-desktop's sources (#169, amendment 3's mix)
# with #171's 1b v3 and the dense sets' VNs skipped. Through heavy.sh (5 threads: 4 torch + 1 data worker):
#   HEAVY_SLOTS=12 heavy.sh timeout 43200 train.sh NAME SHAPE SEED SPLITS RECIPE [extra tsdesk_det.py args...]
set -euo pipefail
NAME=${1:?name}; SHAPE=${2:?shape}; SEED=${3:?seed}; SPLITS=${4:?splits}; RECIPE=${5:?recipe}; shift 5
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts; PV=$HERD/shared/private
REPO=${REPO:-/home/skerraut/herd-synth3/clones/textspot-desktop}
RUNS=${RUNS:-$A/textspot-desktop/runs}
DD=$A/desktop-data
M=$REPO/data/manifests/desktop_data
mkdir -p $RUNS/logs
SHARDS=("$A/textspot/shards/synth_frames" "$A/textspot2/shards/lab/lab_realplus" "$A/textspot2/shards/lab/lab_teacher"
        "$A/textspot2/shards/lab/lab_devnative" "$PV/textspot2/shards/lab/lab_owner" "$A/textspot2/shards/more/more_realplus"
        "$A/textspot2/shards/more/more_teacher" "$PV/textspot2/shards/more/more_owner")
exec "$REPO/.venv/bin/python" "$REPO/train/tsdesk_det.py" --shape "$SHAPE" --seed "$SEED" --out "$RUNS/$NAME" \
  --recipe "$RECIPE" --skip-vns "$PV/textspot-desktop/frames/skip_vns.txt" --targets "$PV/textspot-desktop/targets_v1" --e-exact \
  --hook-manifest "$REPO/data/manifests/desktop_pages_v21.json" --hook-shards "$PV/vocr-desktop/shards_v21" \
  --hook-splits "$SPLITS" --manifest "$REPO/data/manifests/textspot_shards_v1.json" \
  "$REPO/data/manifests/textspot2_shards_v1.json" --shards "${SHARDS[@]}" \
  --mix hook=0.3375,synth=0.15,real=0.2625,desk_hook=0.1,desk_realplus=0.1,desk_synth=0.05 \
  --extra "desk_hook=$M/hooked_desktop_v1.json:$DD/pages/hooked_v1" "desk_realplus=$M/realplus_native_v3.json:$DD/pages/realplus_v3" \
          "desk_synth=$M/synth_desktop_v1.json:$DD/pages/synth_v1" \
  --threads 4 --workers 1 "$@" > "$RUNS/logs/$NAME.log" 2>&1

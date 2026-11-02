#!/usr/bin/env bash
# The screening reads of one arm (#173 protocol: hval and dense_screen at fit:960 and long:960, the reading pad k6v4;
# the edge error and tightness cost on hval at fit:960 and long:960). One heavy slot, 5 workers:
#   HEAVY_SLOTS=12 heavy.sh timeout 7200 screen_arm.sh ARM [MODEL.onnx]
set -euo pipefail
ARM=${1:?arm}
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop; PV=$HERD/shared/private
R=${REPO:-/home/skerraut/herd-synth3/clones/textspot-desktop}
PY=$R/.venv/bin/python
M=${2:-$A/runs/$ARM/final.onnx}
L=$HERD/shared/artifacts/vocr-onnx/release/VOCR6.2_L_fast.onnx
PAD=${PAD:-k6v4}
B=$PV/textspot-desktop/boxes; RES=$A/res; ROWS=$PV/textspot-desktop/rows
mkdir -p $B $RES $ROWS
export OMP_NUM_THREADS=1
for SET in hval dense_screen; do
  if [ $SET = hval ]; then F=$PV/vocr-desktop/frames/hval.pkl; else F=$PV/textspot-desktop/frames/dense_screen.pkl; fi
  for POL in fit:960 long:960; do
    P=${POL/:/}
    $PY $R/eval/tsdesk_eval.py desktop detect --frames $F --det "tsd:$M@$POL" --out $B/${SET}_${ARM}_${P}.jsonl --workers 5
    $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${SET}_${ARM}_${P}.jsonl --rec $L --pad $PAD \
        --out $RES/${SET}_${ARM}_${P}_L_fast.json --rows $ROWS/${SET}_${ARM}_${P}.jsonl --workers 5
    if [ $SET = hval ]; then
      $PY $R/eval/tsdesk_eval.py edges --frames $F --boxes $B/${SET}_${ARM}_${P}.jsonl --pad $PAD \
          --out $RES/edges_${SET}_${ARM}_${P}.json --rows $ROWS/edges_${SET}_${ARM}_${P}.jsonl --workers 5
      $PY $R/eval/tsdesk_eval.py desktop boxes-pr --frames $F --boxes $B/${SET}_${ARM}_${P}.jsonl --pad none \
          --out $RES/boxes_pr_${SET}_${ARM}_${P}.json
    fi
  done
done
OMP_NUM_THREADS=1 $PY $A/scripts/net_time.py $M $RES/net_time_${ARM}.json
echo "screen_arm: $ARM finished"

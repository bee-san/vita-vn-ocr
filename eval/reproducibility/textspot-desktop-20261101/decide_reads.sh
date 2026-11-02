#!/usr/bin/env bash
# The decision reads of one seed of the chosen arm (#173 protocol "rule"): dev-v2 and dense_dev at fit:960 and
# long:960 (R1, R2), hval reads and edges (R3, R4), the reading pad k6v4. Registered before it runs; one heavy slot:
#   HEAVY_SLOTS=12 heavy.sh timeout 7200 decide_reads.sh ARM SEED MODEL.onnx
set -euo pipefail
ARM=${1:?arm}; SEED=${2:?seed}; M=${3:?model}
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop; PV=$HERD/shared/private
R=${REPO:-/home/skerraut/herd-synth3/clones/textspot-desktop}
PY=$R/.venv/bin/python
L=$HERD/shared/artifacts/vocr-onnx/release/VOCR6.2_L_fast.onnx
PAD=k6v4
B=$PV/textspot-desktop/boxes; RES=$A/res; ROWS=$PV/textspot-desktop/rows
export OMP_NUM_THREADS=1
TAG=${ARM}_s${SEED}
for SET in dev dense_dev hval; do
  case $SET in
    dev) F=$PV/vocr-desktop/frames/dev.pkl ;;
    dense_dev) F=$PV/textspot-desktop/frames/dense_dev.pkl ;;
    hval) F=$PV/vocr-desktop/frames/hval.pkl ;;
  esac
  for POL in fit:960 long:960; do
    P=${POL/:/}
    $PY $R/eval/tsdesk_eval.py desktop detect --frames $F --det "tsd:$M@$POL${REF:-+refine}" --out $B/${SET}_${TAG}_${P}.jsonl --workers 5
    $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${SET}_${TAG}_${P}.jsonl --rec $L --pad $PAD \
        --out $RES/${SET}_${TAG}_${P}_L_fast.json --rows $ROWS/${SET}_${TAG}_${P}.jsonl --workers 5
    if [ $SET = hval ]; then
      $PY $R/eval/tsdesk_eval.py edges --frames $F --boxes $B/${SET}_${TAG}_${P}.jsonl --pad $PAD \
          --out $RES/edges_${SET}_${TAG}_${P}.json --rows $ROWS/edges_${SET}_${TAG}_${P}.jsonl --workers 5
    fi
  done
done
echo "decide_reads: $TAG finished"

#!/usr/bin/env bash
# The baselines on the decision sets (#173): R (MeikiOCR's detector, pad auto), T8 (textspot8, pad none; R1's base) and
# the exact boxes (pad k6v4) on dev-v2 and dense_dev; dev-v2 reuses vocr-desktop's stored MeikiOCR / textspot8 boxes
# (found on the same frames file). One heavy slot:  HEAVY_SLOTS=12 heavy.sh timeout 7200 baselines_decide.sh
set -euo pipefail
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop; PV=$HERD/shared/private
R=${REPO:-/home/skerraut/herd-synth3/clones/textspot-desktop}
PY=$R/.venv/bin/python
L=$HERD/shared/artifacts/vocr-onnx/release/VOCR6.2_L_fast.onnx
B=$PV/textspot-desktop/boxes; RES=$A/res; ROWS=$PV/textspot-desktop/rows
export OMP_NUM_THREADS=1
DV=$PV/vocr-desktop/frames/dev.pkl; DD=$PV/textspot-desktop/frames/dense_dev.pkl
cp $PV/vocr-desktop/boxes/dev_meiki.jsonl $B/dev_R.jsonl; cp $PV/vocr-desktop/boxes/dev_ts8.jsonl $B/dev_T8.jsonl
$PY $R/eval/tsdesk_eval.py desktop detect --frames $DD --det meiki --out $B/dense_dev_R.jsonl --workers 5
$PY $R/eval/tsdesk_eval.py desktop detect --frames $DD --det ts8 --out $B/dense_dev_T8.jsonl --workers 5
for SET in dev dense_dev; do
  if [ $SET = dev ]; then F=$DV; else F=$DD; fi
  $PY $R/eval/tsdesk_eval.py exact-boxes --frames $F --out $B/${SET}_exact.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${SET}_R.jsonl --rec $L --pad auto \
      --out $RES/${SET}_R_L_fast.json --rows $ROWS/${SET}_R.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${SET}_T8.jsonl --rec $L --pad none \
      --out $RES/${SET}_T8_L_fast.json --rows $ROWS/${SET}_T8.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${SET}_exact.jsonl --rec $L --pad k6v4 \
      --out $RES/${SET}_exact_L_fast.json --rows $ROWS/${SET}_exact.jsonl --workers 5
done
echo "baselines_decide finished"

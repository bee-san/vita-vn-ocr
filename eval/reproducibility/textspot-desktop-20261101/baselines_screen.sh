#!/usr/bin/env bash
# The baselines on the screening sets (#173): R (MeikiOCR's detector, pad auto), T8 (textspot8, pad none) and the exact
# boxes (pad k6v4), read by VOCR6.2-L fast; hval reuses vocr-desktop's stored MeikiOCR / textspot8 boxes (found on the
# same frames file). One heavy slot:  HEAVY_SLOTS=12 heavy.sh timeout 7200 baselines_screen.sh
set -euo pipefail
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop; PV=$HERD/shared/private
R=${REPO:-/home/skerraut/herd-synth3/clones/textspot-desktop}
PY=$R/.venv/bin/python
L=$HERD/shared/artifacts/vocr-onnx/release/VOCR6.2_L_fast.onnx
B=$PV/textspot-desktop/boxes; RES=$A/res; ROWS=$PV/textspot-desktop/rows
mkdir -p $B $RES $ROWS
export OMP_NUM_THREADS=1
HV=$PV/vocr-desktop/frames/hval.pkl; DS=$PV/textspot-desktop/frames/dense_screen.pkl
cp $PV/vocr-desktop/boxes/hval_meiki.jsonl $B/hval_R.jsonl; cp $PV/vocr-desktop/boxes/hval_ts8.jsonl $B/hval_T8.jsonl
$PY $R/eval/tsdesk_eval.py desktop detect --frames $DS --det meiki --out $B/dense_screen_R.jsonl --workers 5
$PY $R/eval/tsdesk_eval.py desktop detect --frames $DS --det ts8 --out $B/dense_screen_T8.jsonl --workers 5
for SET in hval dense_screen; do
  if [ $SET = hval ]; then F=$HV; else F=$DS; fi
  $PY $R/eval/tsdesk_eval.py exact-boxes --frames $F --out $B/${SET}_exact.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${SET}_R.jsonl --rec $L --pad auto \
      --out $RES/${SET}_R_L_fast.json --rows $ROWS/${SET}_R.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${SET}_T8.jsonl --rec $L --pad none \
      --out $RES/${SET}_T8_L_fast.json --rows $ROWS/${SET}_T8.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${SET}_exact.jsonl --rec $L --pad k6v4 \
      --out $RES/${SET}_exact_L_fast.json --rows $ROWS/${SET}_exact.jsonl --workers 5
done
$PY $R/eval/tsdesk_eval.py desktop boxes-pr --frames $HV --boxes $B/hval_R.jsonl --pad auto --out $RES/boxes_pr_hval_R.json
$PY $R/eval/tsdesk_eval.py desktop boxes-pr --frames $HV --boxes $B/hval_T8.jsonl --pad none --out $RES/boxes_pr_hval_T8.json
for D in R T8; do
  $PY $R/eval/tsdesk_eval.py edges --frames $HV --boxes $B/hval_$D.jsonl --pad k6v4 --out $RES/edges_hval_$D.json --workers 5
done
echo "baselines_screen finished"

#!/usr/bin/env bash
# Screening rule 3 for one arm (#173): hval at fit:960 and long:960 with the refinement (tsd:MODEL@SIZE+refine),
# the reads (the score's hval half) and the edges (the pooled median |e|), pad k6v4. One heavy slot:
#   HEAVY_SLOTS=12 heavy.sh timeout 7200 screen_refine.sh ARM [MODEL.onnx]
set -euo pipefail
ARM=${1:?arm}
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop; PV=$HERD/shared/private
R=${REPO:-/home/skerraut/herd-synth3/clones/textspot-desktop}
PY=$R/.venv/bin/python
M=${2:-$A/runs/$ARM/final.onnx}
L=$HERD/shared/artifacts/vocr-onnx/release/VOCR6.2_L_fast.onnx
B=$PV/textspot-desktop/boxes; RES=$A/res; ROWS=$PV/textspot-desktop/rows
export OMP_NUM_THREADS=1
F=$PV/vocr-desktop/frames/hval.pkl
for POL in fit:960 long:960; do
  P=${POL/:/}
  $PY $R/eval/tsdesk_eval.py desktop detect --frames $F --det "tsd:$M@$POL+refine" --out $B/hval_${ARM}_${P}_refine.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/hval_${ARM}_${P}_refine.jsonl --rec $L --pad k6v4 \
      --out $RES/hval_${ARM}_${P}_refine_L_fast.json --rows $ROWS/hval_${ARM}_${P}_refine.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py edges --frames $F --boxes $B/hval_${ARM}_${P}_refine.jsonl --pad k6v4 \
      --out $RES/edges_hval_${ARM}_${P}_refine.json --rows $ROWS/edges_hval_${ARM}_${P}_refine.jsonl --workers 5
done
echo "screen_refine: $ARM finished"

#!/usr/bin/env bash
# R6 for the representative seed (#173): the C path (ctsd:BLOB@POL+refine) against the ONNX path's boxes of the
# decision reads on dev-v2, crops equal per frame (pad k6v4), and the C boxes' dev-v2 CER. One heavy slot:
#   HEAVY_SLOTS=12 heavy.sh timeout 7200 parity_dev.sh TAG BLOB.vtsd      (TAG e.g. Si_s2: the decision reads' tag)
set -euo pipefail
TAG=${1:?tag}; BLOB=${2:?blob}
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop; PV=$HERD/shared/private
R=${REPO:-/home/skerraut/herd-synth3/clones/textspot-desktop}
PY=$R/.venv/bin/python
L=$HERD/shared/artifacts/vocr-onnx/release/VOCR6.2_L_fast.onnx
B=$PV/textspot-desktop/boxes; RES=$A/res; ROWS=$PV/textspot-desktop/rows
export OMP_NUM_THREADS=1
F=$PV/vocr-desktop/frames/dev.pkl
for POL in fit:960 long:960; do
  P=${POL/:/}
  $PY $R/eval/tsdesk_eval.py desktop detect --frames $F --det "ctsd:$BLOB@$POL+refine" --out $B/dev_${TAG}_${P}_c.jsonl --workers 5
  $PY $R/eval/tsdesk_eval.py parity --frames $F --a $B/dev_${TAG}_${P}_c.jsonl --b $B/dev_${TAG}_${P}.jsonl --pad k6v4 \
      --out $RES/parity_dev_${TAG}_${P}.json
  $PY $R/eval/tsdesk_eval.py desktop read --frames $F --boxes $B/dev_${TAG}_${P}_c.jsonl --rec $L --pad k6v4 \
      --out $RES/dev_${TAG}_${P}_c_L_fast.json --rows $ROWS/dev_${TAG}_${P}_c.jsonl --workers 5
done
echo "parity_dev: $TAG finished"

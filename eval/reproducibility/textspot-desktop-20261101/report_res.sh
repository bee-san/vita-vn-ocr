#!/usr/bin/env bash
# Report-only reads of the research candidate (#173): dev-v2 at 720 / 1080 / 1440 / 2160 px (vocr-desktop's dev_h*.pkl)
# and the like-for-like frames fit into 960x544 (dev_l544, hval_l544), +refine, pad k6v4.
set -euo pipefail
M=${1:?model}
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop; PV=$HERD/shared/private
R=/home/skerraut/herd-synth3/clones/textspot-desktop; PY=$R/.venv/bin/python
L=$HERD/shared/artifacts/vocr-onnx/release/VOCR6.2_L_fast.onnx
B=$PV/textspot-desktop/boxes; RES=$A/res/report; mkdir -p $RES
export OMP_NUM_THREADS=1
cd $R
for S in dev_h720 dev_h1080 dev_h1440 dev_h2160 dev_l544 hval_l544; do
  F=$PV/vocr-desktop/frames/$S.pkl
  POL=long:960; [ $S = dev_l544 ] || [ $S = hval_l544 ] && POL=fit:960
  $PY eval/tsdesk_eval.py desktop detect --frames $F --det "tsd:$M@$POL+refine" --out $B/${S}_final_s2.jsonl --workers 5
  $PY eval/tsdesk_eval.py desktop read --frames $F --boxes $B/${S}_final_s2.jsonl --rec $L --pad k6v4 --out $RES/${S}_final_s2.json --workers 5
done
echo "report_res finished"

#!/usr/bin/env bash
# vocr-desktop (#169, amendment 4): the baselines on the like-for-like frames (fit into 960x544): R, textspot8 + L,
# MeikiOCR itself. heavy.sh timeout 10800 baselines_l544.sh SET...
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=$REPO/.venv/bin/python; D="$PY $REPO/eval/desktop.py"
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
R=$HERD/shared/artifacts/vocr-onnx/release
for S in "$@"; do
  F=$PV/frames/$S.pkl
  for det in meiki ts8; do [ -s $PV/boxes/${S}_$det.jsonl ] || $D detect --frames $F --det $det --out $PV/boxes/${S}_$det.jsonl --workers 5; done
  [ -s $A/res/${S}_meiki4_L_fast.json ] || $D read --frames $F --boxes $PV/boxes/${S}_meiki.jsonl --pad auto --rec $R/VOCR6.2_L_fast.onnx --out $A/res/${S}_meiki4_L_fast.json --rows $PV/rows/${S}_meiki4_L_fast.rows.jsonl --workers 5 | tail -n 1
  [ -s $A/res/${S}_ts8_L_fast.json ] || $D read --frames $F --boxes $PV/boxes/${S}_ts8.jsonl --rec $R/VOCR6.2_L_fast.onnx --out $A/res/${S}_ts8_L_fast.json --rows $PV/rows/${S}_ts8_L_fast.rows.jsonl --workers 5 | tail -n 1
  [ -s $A/res/${S}_meikiocr.json ] || $D meiki --frames $F --out $A/res/${S}_meikiocr.json --rows $PV/rows/${S}_meikiocr.rows.jsonl --workers 5 | tail -n 1
done
echo "baselines_l544 finished"

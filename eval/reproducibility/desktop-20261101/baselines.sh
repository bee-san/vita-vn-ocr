#!/usr/bin/env bash
# vocr-desktop baselines on one set (#169): MeikiOCR 0.3.5 itself, MeikiOCR's detector + VOCR6.2 (fast/exact ONNX),
# textspot8 + VOCR6.2 S/N/L (fast/exact). Run in one heavy slot: HEAVY_SLOTS=12 heavy.sh timeout 7200 baselines.sh SET
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
SET=${1:?set}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=$REPO/.venv/bin/python
D="$PY $REPO/eval/desktop.py"
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
R=$HERD/shared/artifacts/vocr-onnx/release
F=$PV/frames/$SET.pkl
mkdir -p $PV/boxes $PV/rows $A/res
for det in meiki ts8; do
  [ -s $PV/boxes/${SET}_$det.jsonl ] || $D detect --frames $F --det $det --out $PV/boxes/${SET}_$det.jsonl --workers 5
done
rd() { local name=$1; shift
  [ -s $A/res/${SET}_$name.json ] && return 0
  $D read --frames $F --out $A/res/${SET}_$name.json --rows $PV/rows/${SET}_$name.rows.jsonl --workers 5 "$@" > $A/logs/${SET}_$name.log 2>&1
  echo "$SET $name: $(tail -n 1 $A/logs/${SET}_$name.log)"
}
rd meiki4_L_fast --boxes $PV/boxes/${SET}_meiki.jsonl --pad auto --rec $R/VOCR6.2_L_fast.onnx
rd meiki4_L_exact --boxes $PV/boxes/${SET}_meiki.jsonl --pad auto --rec $R/VOCR6.2_L_exact.onnx
rd meiki0_L_fast --boxes $PV/boxes/${SET}_meiki.jsonl --pad none --rec $R/VOCR6.2_L_fast.onnx
for T in S N; do rd meiki4_${T}_fast --boxes $PV/boxes/${SET}_meiki.jsonl --pad auto --rec $R/VOCR6.2_${T}_fast.onnx; done
for T in S N L; do for M in fast exact; do
  rd ts8_${T}_$M --boxes $PV/boxes/${SET}_ts8.jsonl --rec $R/VOCR6.2_${T}_$M.onnx
done; done
[ -s $A/res/${SET}_meikiocr.json ] || { $D meiki --frames $F --out $A/res/${SET}_meikiocr.json --rows $PV/rows/${SET}_meikiocr.rows.jsonl --workers 5 > $A/logs/${SET}_meikiocr.log 2>&1; echo "$SET meikiocr: $(tail -n 1 $A/logs/${SET}_meikiocr.log)"; }
echo "baselines $SET finished"

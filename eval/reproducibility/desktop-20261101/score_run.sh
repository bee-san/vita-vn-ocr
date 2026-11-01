#!/usr/bin/env bash
# Score one desktop detector run on a set at input-size policies (#169): detect (desk:RUN/final.onnx@SIZE[@PAD]) then
# read with VOCR6.2_L_fast. heavy.sh timeout 7200 score_run.sh RUN SET "fit:960 fit:640@frame ..." [TAG]
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
RUN=${1:?run}; SET=${2:?set}; SIZES=${3:?sizes}; TAG=${4:-$(basename $RUN)}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=$REPO/.venv/bin/python; D="$PY $REPO/eval/desktop.py"
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
R=$HERD/shared/artifacts/vocr-onnx/release
F=$PV/frames/$SET.pkl
[ -s "$RUN/final.onnx" ] || { echo "no $RUN/final.onnx"; exit 2; }
for S in $SIZES; do
  s=${S//:/}; s=${s//@/_}
  B=$PV/boxes/${SET}_${TAG}_$s.jsonl
  [ -s $B ] || $D detect --frames $F --det "desk:$RUN/final.onnx@$S" --out $B --workers 5 > $A/logs/det_${SET}_${TAG}_$s.log 2>&1
  O=$A/res/${SET}_${TAG}_${s}_L_fast.json
  [ -s $O ] || $D read --frames $F --boxes $B --rec $R/VOCR6.2_L_fast.onnx --out $O --rows $PV/rows/${SET}_${TAG}_${s}_L_fast.rows.jsonl --workers 5 > $A/logs/read_${SET}_${TAG}_$s.log 2>&1
  echo "$SET $TAG $S: $(tail -n 1 $A/logs/read_${SET}_${TAG}_$s.log)"
done

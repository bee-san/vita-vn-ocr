#!/usr/bin/env bash
# vocr-desktop (#169, amendment 6): the research model final3 (final_d64_seed3: train + guard + hval) on dev-v2 at the
# 7 policies, the like-for-like frames, dev-v2 resampled to 720-2160 px (per input resolution) and the PC sample.
#   heavy.sh timeout 14400 final3_reads.sh
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=$REPO/.venv/bin/python; D="$PY $REPO/eval/desktop.py"
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
R=$HERD/shared/artifacts/vocr-onnx/release
M=$A/models/final3/vocr_desktop_det.onnx
rd() { local set=$1 tag=$2 spec=$3
  local B=$PV/boxes/${set}_$tag.jsonl O=$A/res/${set}_${tag}_L_fast.json
  [ -s $B ] || $D detect --frames $PV/frames/$set.pkl --det "$spec" --out $B --workers 5 > $A/logs/det_${set}_$tag.log 2>&1
  [ -s $O ] || $D read --frames $PV/frames/$set.pkl --boxes $B --rec $R/VOCR6.2_L_fast.onnx --out $O \
      --rows $PV/rows/${set}_${tag}_L_fast.rows.jsonl --workers 5 > $A/logs/read_${set}_$tag.log 2>&1
  echo "$set $tag: $(tail -n 1 $A/logs/read_${set}_$tag.log)"; }
for P in fit:960 fit:800 fit:640 fit:480 long:960 long:800 long:640; do rd dev final3_${P//:/} "desk:$M@$P"; done
rd dev_l544 final3_fit960 "desk:$M@fit:960"
for h in 720 1080 1440 2160; do for P in long:640 fit:640 fit:960; do rd dev_h$h final3_${P//:/} "desk:$M@$P"; done; done
for P in long:640 fit:960; do rd pc final3_${P//:/} "desk:$M@$P"; done
echo "final3_reads finished"

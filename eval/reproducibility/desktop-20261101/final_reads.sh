#!/usr/bin/env bash
# vocr-desktop (#169): the final reads of one set, every arm in one pass (amendments 4-6): R (MeikiOCR's detector +
# VOCR6.2-L), MeikiOCR itself, textspot8 + VOCR6.2-L, mida (report only), the representative F3 at its policies and the
# research model final3. A like-for-like set (*_l544) reads F3 and final3 at fit:960 only (their frames are already
# fit into 960x544, so they read them as MeikiOCR's detector does).
#   heavy.sh timeout 14400 final_reads.sh SET
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
SET=${1:?set}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=$REPO/.venv/bin/python; D="$PY $REPO/eval/desktop.py"
MIDA="$PY $REPO/eval/mida_det.py desktop --repo $REPO --model ${MIDA_MODEL:-/local/home/skerraut/herd-synth3/shared/mida-ocr/mida_ocr_l.onnx} --"
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
R=$HERD/shared/artifacts/vocr-onnx/release
F=$PV/frames/$SET.pkl
case $SET in *_l544) POL="fit:960";; *) POL="fit:960 fit:800 fit:640 fit:480 long:960 long:800 long:640";; esac
[ "${SKIP_F3:-0}" = 1 ] && POL=""
det() { local tag=$1 spec=$2 runner=${3:-$D}
  [ -s $PV/boxes/${SET}_$tag.jsonl ] || $runner detect --frames $F --det "$spec" --out $PV/boxes/${SET}_$tag.jsonl --workers 5 > $A/logs/det_${SET}_$tag.log 2>&1; }
rd() { local btag=$1 tag=$2 pad=$3
  local O=$A/res/${SET}_${tag}_L_fast.json
  [ -s $O ] || $D read --frames $F --boxes $PV/boxes/${SET}_$btag.jsonl --pad $pad --rec $R/VOCR6.2_L_fast.onnx --out $O \
      --rows $PV/rows/${SET}_${tag}_L_fast.rows.jsonl --workers 5 > $A/logs/read_${SET}_$tag.log 2>&1
  echo "$SET $tag: $(tail -n 1 $A/logs/read_${SET}_$tag.log)"; }
det meiki meiki; rd meiki meiki4 auto
det ts8 ts8; rd ts8 ts8 none
det mida "mida@box:960x544@along=1.0@min_cells=2" "$MIDA"; rd mida mida none
for P in $POL; do det F3_${P//:/} "desk:$A/models/F3/det.onnx@$P"; rd F3_${P//:/} F3_${P//:/} none; done
case $SET in *_l544) FP=fit:960;; *) FP=long:640;; esac
det final3_${FP//:/} "desk:$A/models/final3/vocr_desktop_det.onnx@$FP"; rd final3_${FP//:/} final3_${FP//:/} none
[ -s $A/res/${SET}_meikiocr.json ] || { $D meiki --frames $F --out $A/res/${SET}_meikiocr.json --rows $PV/rows/${SET}_meikiocr.rows.jsonl --workers 5 > $A/logs/${SET}_meikiocr.log 2>&1; echo "$SET meikiocr: $(tail -n 1 $A/logs/${SET}_meikiocr.log)"; }
echo "final_reads $SET finished"

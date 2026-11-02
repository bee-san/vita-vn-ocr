#!/usr/bin/env bash
# The PC test set, read once (#173 protocol rule.pctest_guard, amendment 6): the gate (eval/tsdesk_eval.py pctest-gate,
# which refuses a second read and needs the protocol and the decision record committed unchanged), then every row:
# the candidate (tsd:MODEL@POL+refine, pad k6v4), T8, R, the exact boxes, MeikiOCR itself, mida (report only), and the
# like-for-like rows on the frames fit into 960x544. One heavy slot:
#   HEAVY_SLOTS=12 heavy.sh timeout 10800 pctest_once.sh MODEL.onnx POL WHAT
set -euo pipefail
M=${1:?model}; POL=${2:?policy}; WHAT=${3:?what}
HERD=/local/home/skerraut/herd-vita-ocr
A=$HERD/shared/artifacts/textspot-desktop; PV=$HERD/shared/private
R=${REPO:-/home/skerraut/herd-synth3/clones/textspot-desktop}
PY=$R/.venv/bin/python
L=$HERD/shared/artifacts/vocr-onnx/release/VOCR6.2_L_fast.onnx
O=$PV/textspot-desktop/pctest; mkdir -p $O
RES=$A/res/pctest; mkdir -p $RES
export OMP_NUM_THREADS=1 MIDA_MODEL=/local/home/skerraut/herd-synth3/shared/mida-ocr/mida_ocr_l.onnx
F=$HERD/shared/artifacts/desktop-data/frozen_v3/frames_pctest_v1.pkl
FL=$PV/vocr-desktop/frames/pctest_l544.pkl
P=${POL/:/}
cd $R
$PY eval/tsdesk_eval.py pctest-gate --what "$WHAT"
$PY eval/tsdesk_eval.py desktop detect --frames $F --det "tsd:$M@$POL+refine" --out $O/cand.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop read --frames $F --boxes $O/cand.jsonl --rec $L --pad k6v4 --out $RES/pctest_cand_${P}.json --rows $O/cand_rows.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop detect --frames $F --det ts8 --out $O/T8.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop read --frames $F --boxes $O/T8.jsonl --rec $L --pad none --out $RES/pctest_T8.json --rows $O/T8_rows.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop detect --frames $F --det meiki --out $O/R.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop read --frames $F --boxes $O/R.jsonl --rec $L --pad auto --out $RES/pctest_R.json --rows $O/R_rows.jsonl --workers 5
$PY eval/tsdesk_eval.py exact-pctest --frames $F --out $O/exact.jsonl
$PY eval/tsdesk_eval.py desktop read --frames $F --boxes $O/exact.jsonl --rec $L --pad k6v4 --out $RES/pctest_exact.json --rows $O/exact_rows.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop meiki --frames $F --out $RES/pctest_meikiocr.json --rows $O/meikiocr_rows.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop detect --frames $F --det "mida@box:960x544@along=1.0@min_cells=2" --out $O/mida.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop read --frames $F --boxes $O/mida.jsonl --rec $L --pad none --out $RES/pctest_mida.json --rows $O/mida_rows.jsonl --workers 5
# like-for-like: every pipeline on the frames fit into 960x544 (MeikiOCR's own detector scale)
$PY eval/tsdesk_eval.py desktop detect --frames $FL --det "tsd:$M@fit:960+refine" --out $O/cand_l544.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop read --frames $FL --boxes $O/cand_l544.jsonl --rec $L --pad k6v4 --out $RES/pctest_l544_cand.json --workers 5
$PY eval/tsdesk_eval.py desktop detect --frames $FL --det ts8 --out $O/T8_l544.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop read --frames $FL --boxes $O/T8_l544.jsonl --rec $L --pad none --out $RES/pctest_l544_T8.json --workers 5
$PY eval/tsdesk_eval.py desktop detect --frames $FL --det meiki --out $O/R_l544.jsonl --workers 5
$PY eval/tsdesk_eval.py desktop read --frames $FL --boxes $O/R_l544.jsonl --rec $L --pad auto --out $RES/pctest_l544_R.json --workers 5
$PY eval/tsdesk_eval.py desktop meiki --frames $FL --out $RES/pctest_l544_meikiocr.json --workers 5
for X in T8 R exact meikiocr mida; do
  $PY eval/tsdesk_eval.py desktop compare --base $RES/pctest_T8.json --arm $RES/pctest_$X.json --out $RES/cmp_$X-T8.json > /dev/null 2>&1 || true
done
$PY eval/tsdesk_eval.py desktop compare --base $RES/pctest_T8.json --arm $RES/pctest_cand_${P}.json --out $RES/cmp_cand-T8.json
$PY eval/tsdesk_eval.py desktop compare --base $RES/pctest_R.json --arm $RES/pctest_cand_${P}.json --out $RES/cmp_cand-R.json
echo "pctest_once finished"

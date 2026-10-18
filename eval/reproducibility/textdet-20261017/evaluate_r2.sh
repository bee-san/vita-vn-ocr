#!/usr/bin/env bash
# Round 2 (protocol amendment 3): det+refine for every round-1 blob on the development data, then the rule.
#   evaluate_r2.sh SWEEP OUT [CBIN]      (CBIN: the C driver with the det+refine batch mode; without it the twin only,
#                                         and the selection waits for the C check)
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SWEEP=$1; OUT=$2; CBIN=${3:-}
T=$HERD/shared/artifacts/textdet; V5=$HERD/shared/artifacts/vocr5-train
H="env HEAVY_SLOTS=12 $HERD/heavy.sh"
cd "$REPO"; PY=.venv/bin/python
RUNS=$($PY -c 'import json; print(" ".join(json.load(open("eval/textdet_protocol.json"))["amendments"][1]["runs"]))')
mkdir -p "$OUT/dets" "$OUT/linefind" "$OUT/units" "$OUT/logs"
CANDS=(); for r in $RUNS; do CANDS+=(--cand "$r=$SWEEP/$r/final.vtd"); done
$H timeout 7200 $PY eval/textdet.py detect --refine "${CANDS[@]}" --cohorts "$V5/cohorts" --out "$OUT/dets" \
    ${CBIN:+--cbin "$CBIN"} < /dev/null > "$OUT/logs/detect.log" 2>&1
ALL=($RUNS)
for ((i = 0; i < ${#ALL[@]}; i += 4)); do
  batch="${ALL[*]:i:4}"
  $H timeout 7200 bash -c "for r in $batch; do [ -e '$OUT/linefind/'linefind_\$r.json ] || $PY eval/textdet.py \
      linefind --finder detref:\$r='$SWEEP'/\$r/final.vtd --out '$OUT/linefind/'linefind_\$r.json || exit 1; done" \
      < /dev/null > "$OUT/logs/linefind_$i.log" 2>&1 &
done
for m in S N L; do
  case $m in S) b=v5f_S_seed2;; N) b=v5f_N_seed3;; L) b=v5f_L_seed2;; esac
  for ((g = 0; g < 3; g++)); do
    sel=(); k=0
    for r in $RUNS; do [ $((k % 3)) = $g ] && sel+=(--cand "$r"); k=$((k + 1)); done
    $H timeout 21600 $PY eval/textdet.py read --model "VOCR5-$m=$V5/blobs/${b}_w8.vocr" --dets "$OUT/dets" \
        --cohorts "$V5/cohorts" --out "$OUT/units" "${sel[@]}" < /dev/null > "$OUT/logs/read_${m}_$g.log" 2>&1 &
  done
done
wait
echo EVALUATE_R2_SCORED

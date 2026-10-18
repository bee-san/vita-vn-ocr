#!/usr/bin/env bash
# Scores sweep runs as they finish (the steps of evaluate.sh, skipping what exists), until every run is scored.
#   evaluate_incremental.sh SWEEP OUT CBIN
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SWEEP=$1; OUT=$2; CBIN=$3
V5=$HERD/shared/artifacts/vocr5-train
H="env HEAVY_SLOTS=12 $HERD/heavy.sh"
cd "$REPO"; PY=.venv/bin/python
RUNS=$($PY -c 'import json; print(" ".join(json.load(open("eval/textdet_protocol.json"))["amendments"][1]["runs"]))')
mkdir -p "$OUT/dets" "$OUT/linefind" "$OUT/units" "$OUT/logs"
while :; do
  todo=(); all=1
  for r in $RUNS; do
    if [ -e "$SWEEP/$r/done.json" ]; then
      [ -e "$OUT/units/VOCR5-L/units_$r.json" ] && [ -e "$OUT/units/VOCR5-S/units_$r.json" ] && \
        [ -e "$OUT/units/VOCR5-N/units_$r.json" ] && [ -e "$OUT/linefind/linefind_$r.json" ] || todo+=("$r")
    else
      all=0
    fi
  done
  if [ ${#todo[@]} -gt 0 ]; then
    cands=(); for r in "${todo[@]}"; do cands+=(--cand "$r=$SWEEP/$r/final.vtd"); done
    $H timeout 7200 $PY eval/textdet.py detect "${cands[@]}" --cohorts "$V5/cohorts" --out "$OUT/dets" --cbin "$CBIN" \
        < /dev/null >> "$OUT/logs/detect.log" 2>&1
    $H timeout 7200 bash -c "for r in ${todo[*]}; do [ -e '$OUT/linefind/'linefind_\$r.json ] || $PY eval/textdet.py \
        linefind --finder det:\$r='$SWEEP'/\$r/final.vtd --out '$OUT/linefind/'linefind_\$r.json || exit 1; done" \
        < /dev/null >> "$OUT/logs/linefind.log" 2>&1 &
    for m in S N L; do
      case $m in S) b=v5f_S_seed2;; N) b=v5f_N_seed3;; L) b=v5f_L_seed2;; esac
      sel=(); for r in "${todo[@]}"; do sel+=(--cand "$r"); done
      $H timeout 21600 $PY eval/textdet.py read --model "VOCR5-$m=$V5/blobs/${b}_w8.vocr" --dets "$OUT/dets" \
          --cohorts "$V5/cohorts" --out "$OUT/units" "${sel[@]}" < /dev/null >> "$OUT/logs/read_$m.log" 2>&1 &
    done
    wait
  elif [ $all = 1 ]; then
    break
  else
    sleep 120
  fi
done
echo INCREMENTAL_DONE

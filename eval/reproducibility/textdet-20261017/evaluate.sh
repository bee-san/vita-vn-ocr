#!/usr/bin/env bash
# Every sweep run on the development data, then the pre-registered selection (eval/textdet_protocol.json).
#   evaluate.sh SWEEP OUT CBIN
#     SWEEP  sweep.sh's output (RUN/final.vtd per run)
#     OUT    dets/, linefind/, units/<model>/, arena.json, selection.json
#     CBIN   the C runtime's driver (runtime/vocr_textdet_test): C == twin on every development case, exact arenas
# Every step runs through heavy.sh (5 threads a slot).
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SWEEP=${1:?usage: evaluate.sh SWEEP OUT CBIN}; OUT=${2:?}; CBIN=${3:?}
T=$HERD/shared/artifacts/textdet; V5=$HERD/shared/artifacts/vocr5-train
H="env HEAVY_SLOTS=12 $HERD/heavy.sh"
cd "$REPO"
PY=.venv/bin/python
RUNS=$($PY -c 'import json; print(" ".join(json.load(open("eval/textdet_protocol.json"))["amendments"][1]["runs"]))')
mkdir -p "$OUT/dets" "$OUT/linefind" "$OUT/units" "$OUT/logs"
CANDS=()
for r in $RUNS; do CANDS+=(--cand "$r=$SWEEP/$r/final.vtd"); done
# 1. every candidate on every development case: the twin, checked against the C driver
$H timeout 7200 $PY eval/textdet.py detect "${CANDS[@]}" --cohorts "$V5/cohorts" --out "$OUT/dets" --cbin "$CBIN" \
    < /dev/null > "$OUT/logs/detect.log" 2>&1
# 2. eval/linefind detection and the empty regions, per candidate (four candidates a slot)
ALL=($RUNS)
for ((i = 0; i < ${#ALL[@]}; i += 4)); do
  batch="${ALL[*]:i:4}"
  $H timeout 7200 bash -c "for r in $batch; do [ -e '$OUT/linefind/'linefind_\$r.json ] || $PY eval/textdet.py \
      linefind --finder det:\$r='$SWEEP'/\$r/final.vtd --out '$OUT/linefind/'linefind_\$r.json || exit 1; done" \
      < /dev/null > "$OUT/logs/linefind_$i.log" 2>&1 &
done
wait
# 3. the three VOCR5 recognizers on every candidate's lines (each model's candidates in four groups, a slot each)
for m in S N L; do
  case $m in S) b=v5f_S_seed2;; N) b=v5f_N_seed3;; L) b=v5f_L_seed2;; esac
  for ((g = 0; g < 4; g++)); do
    sel=(); k=0
    for r in $RUNS; do [ $((k % 4)) = $g ] && sel+=(--cand "$r"); k=$((k + 1)); done
    $H timeout 21600 $PY eval/textdet.py read --model "VOCR5-$m=$V5/blobs/${b}_w8.vocr" --dets "$OUT/dets" \
        --cohorts "$V5/cohorts" --out "$OUT/units" "${sel[@]}" < /dev/null > "$OUT/logs/read_${m}_$g.log" 2>&1 &
  done
done
wait
# 4. exact arenas at 960 x 544 from the C runtime, then the rule
$PY - "$SWEEP" "$CBIN" "$OUT/arena.json" $RUNS <<'EOF'
import json, subprocess, sys
sweep, cbin, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
arena = {r: int(subprocess.run([cbin, "--arena", f"{sweep}/{r}/final.vtd", "960", "544"], check=True,
                               capture_output=True, text=True).stdout.split()[0]) for r in runs}
open(out, "w").write(json.dumps(arena, indent=1) + "\n")
EOF
$PY eval/textdet.py select --baseline "$T/baseline/wd" --units "$OUT/units" --dets "$OUT/dets" \
    --linefind "$OUT/linefind" --arena "$OUT/arena.json" --out "$OUT/selection.json" | tee "$OUT/logs/select.log"
echo EVALUATE_DONE

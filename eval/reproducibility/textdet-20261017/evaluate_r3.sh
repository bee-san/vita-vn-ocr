#!/usr/bin/env bash
# Round 3 (protocol amendment 4): det+refine for round 3's runs on the development data, the C check, the rule.
#   evaluate_r3.sh SWEEP_R3 OUT CBIN
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SWEEP=$1; OUT=$2; CBIN=$3
T=$HERD/shared/artifacts/textdet; V5=$HERD/shared/artifacts/vocr5-train
H="env HEAVY_SLOTS=12 $HERD/heavy.sh"
cd "$REPO"; PY=.venv/bin/python
RUNS=$($PY -c 'import json; print(" ".join(next(a for a in json.load(open("eval/textdet_protocol.json"))["amendments"] if a["n"] == 4)["runs"]))')
mkdir -p "$OUT/dets" "$OUT/linefind" "$OUT/units" "$OUT/logs"
CANDS=(); for r in $RUNS; do CANDS+=(--cand "$r=$SWEEP/$r/final.vtd"); done
$H timeout 7200 $PY eval/textdet.py detect --refine "${CANDS[@]}" --cohorts "$V5/cohorts" --out "$OUT/dets" \
    --cbin "$CBIN" < /dev/null > "$OUT/logs/detect.log" 2>&1
ALL=($RUNS)
for ((i = 0; i < ${#ALL[@]}; i += 3)); do
  batch="${ALL[*]:i:3}"
  $H timeout 7200 bash -c "for r in $batch; do [ -e '$OUT/linefind/'linefind_\$r.json ] || $PY eval/textdet.py \
      linefind --finder detref:\$r='$SWEEP'/\$r/final.vtd --out '$OUT/linefind/'linefind_\$r.json || exit 1; done" \
      < /dev/null > "$OUT/logs/linefind_$i.log" 2>&1 &
done
for m in S N L; do
  case $m in S) b=v5f_S_seed2;; N) b=v5f_N_seed3;; L) b=v5f_L_seed2;; esac
  for ((g = 0; g < 2; g++)); do
    sel=(); k=0
    for r in $RUNS; do [ $((k % 2)) = $g ] && sel+=(--cand "$r"); k=$((k + 1)); done
    $H timeout 21600 $PY eval/textdet.py read --model "VOCR5-$m=$V5/blobs/${b}_w8.vocr" --dets "$OUT/dets" \
        --cohorts "$V5/cohorts" --out "$OUT/units" "${sel[@]}" < /dev/null > "$OUT/logs/read_${m}_$g.log" 2>&1 &
  done
done
wait
$PY - "$SWEEP" "$CBIN" "$OUT/arena.json" $RUNS <<'PYEOF'
import json, subprocess, sys
sweep, cbin, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
arena = {r: int(subprocess.run([cbin, "--arena-refine", f"{sweep}/{r}/final.vtd", "960", "544"], check=True,
                               capture_output=True, text=True).stdout.split()[0]) for r in runs}
open(out, "w").write(json.dumps(arena, indent=1) + "\n")
PYEOF
$PY eval/textdet.py select --runs-from 4 --baseline "$T/baseline/wd" --units "$OUT/units" --dets "$OUT/dets" \
    --linefind "$OUT/linefind" --arena "$OUT/arena.json" --out "$OUT/selection.json" | tee "$OUT/logs/select.log"
echo EVALUATE_R3_DONE

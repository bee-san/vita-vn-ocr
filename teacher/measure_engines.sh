#!/usr/bin/env bash
# Recognizer-only scores of every teacher engine on eval's hand-checked crops, through eval/run.py's
# cmd: backend (the engine reads eval's own crop: the line box plus 4 px on Vita, 2 on PSP).
#   $HERD/heavy.sh teacher/measure_engines.sh [SETS] [ENGINES]
# SETS default vita-test,psp-test (sets without a manifest are skipped by run.py); ENGINES default
# every engine. Runs offline (unshare -rn), one engine per core of the slot, results to
# $TEACHER_OUT/results/engine_<engine>.json and .log. EVAL_ROOT points at a checkout with eval/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
HERD="${HERD:-/local/home/skerraut/herd-vita-ocr}"
OUT="${TEACHER_OUT:-$HERD/shared/artifacts/teacher}/results"
EVAL_ROOT="${EVAL_ROOT:-$ROOT}"
SETS="${1:-vita-test,psp-test}"
ENGINES="${2:-meiki meiki-rgb v6s v5s manga screenai}"
PY="$HERE/.venv/bin/python"
export VOCR_EVAL_DATA="${VOCR_EVAL_DATA:-$HERD/shared/artifacts/eval}" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT"
cd "$EVAL_ROOT"
pids=()
for e in $ENGINES; do
  unshare -rn "$PY" eval/run.py --model "cmd:$PY $HERE/engine_cli.py $e" --set "$SETS" --errors 30 \
    --out "$OUT/engine_$e.json" < /dev/null > "$OUT/engine_$e.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h "lines, .* chars | CER" "$OUT"/engine_*.log || true
echo "MEASURE_ENGINES_DONE rc=$rc"
exit $rc

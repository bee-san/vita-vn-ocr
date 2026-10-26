#!/usr/bin/env bash
# The test reads' dry runs (eval/textspot6_test.py --dry-run): the same code on non-protected stand-ins, before the
# one real read. title: two hval titles (one Vita, one PSP); tests: textspot's development manifest; clannad: a
# stand-in work directory from 30 hval pages of two Vita titles. Run through heavy.sh (5 threads).
#   test_dryrun.sh RUN_DIR DECODE...     a finished stage-1 run as the "final" weights, e.g. base_qS4_seed1 crop det
# Env: PAGEPATH (the pinned eval/pagepath.py), OUT (default $A/vocr6-spotter/dryrun).
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$REPO/.venv/bin/python
PAGEPATH=${PAGEPATH:?set PAGEPATH to the pinned eval/pagepath.py}
OUT=${OUT:-$A/vocr6-spotter/dryrun}
RUN=${1:?usage: test_dryrun.sh RUN_DIR DECODE...}; shift
UNITS_HVAL=$A/vocr6-spotter/eval/units/hval_v1.jsonl
mkdir -p "$OUT"
[ -d "$OUT/standin" ] || "$PY" "$REPO/eval/textspot6_test.py" standin --units "$UNITS_HVAL" \
  --titles PCSG00935,PCSG00468 --pages 30 --no-text 3 --out "$OUT/standin"
first=1
for dec in "$@"; do
  arm=$(basename "$RUN" | sed 's/_seed[0-9]*$//'); seed=$(basename "$RUN" | sed 's/.*_seed//')
  SEL=$OUT/selection_$dec.json
  printf '{"kind": "textspot6-selection", "dry_run": true, "pick": {"arm": "%s", "decode": "%s", "seed": %s}}\n' \
    "$arm" "$dec" "$seed" > "$SEL"
  D=$OUT/$dec
  rm -rf "$D"
  "$PY" "$REPO/eval/textspot6_test.py" title --dry-run --split hval --titles PCSG01110,NPJG00016 --selection "$SEL" \
    --final "$RUN" --out "$D/title" --pagepath "$PAGEPATH"
  if [ $first = 1 ]; then
    "$PY" "$REPO/eval/textspot6_test.py" tests --dry-run --selection "$SEL" --final "$RUN" --out "$D/tests"
    first=0
  fi
  "$PY" "$REPO/eval/textspot6_test.py" clannad --dry-run --work "$OUT/standin" --selection "$SEL" --final "$RUN" \
    --out "$D/clannad" --pagepath "$PAGEPATH"
done
echo "test_dryrun finished $OUT"

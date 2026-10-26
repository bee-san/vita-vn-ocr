#!/usr/bin/env bash
# textspot6's scoring queue (docs/textspot6.md "Metrics"): scores every finished run of RUNS (a run with done.json and
# no OUT/<run>/SCORED marker), one at a time, each through heavy.sh (the label's scoring slot), oldest finished first.
#   score_queue.sh RUNS [OUT]           stop it with: touch RUNS/score_queue.stop
# It starts once WAIT_FOR (a file) holds WAIT_LINE (default: the orphan watcher's ORPHAN_DONE, amendment 4), so the
# label's slots stay at 3 (2 training runs, 1 scoring). A run that failed to score (OUT/<run>/FAILED) is skipped.
# Every scoring job holds SLOT_LOCK (flock; default OUT/../slot3.lock), which the runtime helper's heavy jobs also take,
# so the label's third slot runs one of them at a time (vocr6/slots.txt 11:58Z: the helper is inside the share).
# Env: PAGEPATH (the pinned eval/pagepath.py), HEAVY_SLOTS (15).
set -u
RUNS=${1:?usage: score_queue.sh RUNS [OUT]}
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
OUT=${2:-$HERD/shared/artifacts/vocr6-spotter/eval}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PAGEPATH=${PAGEPATH:?set PAGEPATH}
LOGS=$OUT/logs
mkdir -p "$LOGS"
SLOT_LOCK=${SLOT_LOCK:-$(dirname "$OUT")/slot3.lock}
WAIT_FOR=${WAIT_FOR:-$RUNS/watch_orphan.log}
WAIT_LINE=${WAIT_LINE:-ORPHAN_DONE}
until grep -q "$WAIT_LINE" "$WAIT_FOR" 2>/dev/null || [ -e "$RUNS/score_queue.stop" ]; do sleep 60; done
while [ ! -e "$RUNS/score_queue.stop" ]; do
  next=""
  for d in $(ls -1tr -d "$RUNS"/*/done.json 2>/dev/null | xargs -r -n1 dirname); do
    name=$(basename "$d")
    { [ -e "$OUT/$name/SCORED" ] || [ -e "$OUT/$name/FAILED" ]; } && continue
    next=$d; break
  done
  if [ -z "$next" ]; then sleep 60; continue; fi
  name=$(basename "$next")
  echo "$(date -u +%FT%TZ) scoring $name"
  if flock "$SLOT_LOCK" env HEAVY_SLOTS="${HEAVY_SLOTS:-15}" "$HERD/heavy.sh" timeout 10800 env PAGEPATH="$PAGEPATH" OUT="$OUT" \
       "$REPO/eval/reproducibility/textspot6-20261026/score_run.sh" "$next" "$name" < /dev/null \
       > "$LOGS/score_$name.log" 2>&1; then
    date -u +%FT%TZ > "$OUT/$name/SCORED"
    echo "$(date -u +%FT%TZ) scored $name"
  else
    echo "$(date -u +%FT%TZ) FAILED $name (log $LOGS/score_$name.log)"
    mkdir -p "$OUT/$name" && date -u +%FT%TZ > "$OUT/$name/FAILED"
    sleep 300
  fi
done
echo "score_queue stopped"

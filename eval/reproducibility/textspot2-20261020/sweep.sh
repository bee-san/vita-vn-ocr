#!/usr/bin/env bash
# textspot2's training sweep (docs/textspot2.md "Arms and stages"; issue #106): every run of a plan (textspot2_plan.py stage1
# or stage2) through heavy.sh, 5 threads, at most MAX_PAR runs at once, in the plan's order (longest first).
#   sweep.sh OUT PLAN [RUN ...]          RUN: names of the plan (default: all of them)
# Every run trains to its last step; nothing is selected or stopped early on any score. A run that exits without
# done.json (a crash or its timeout: 6 h, 12 h for 60,000 steps) is moved to OUT/<run>.failed<k> and restarted from
# scratch with the same seed, at most 3 attempts; OUT/crashes.jsonl and OUT/restarts.jsonl record it. A finished
# run is checked by check_run.py (final.vtsp loads in synth/textspot_twin.Model, or synth/textspot2_twin.Model for a
# d2s run, and runs on procedural frames; the C driver runtime/vocr_textspot2_test, built once into OUT/cbuild from
# this checkout's sources, gives the twin's result there) and appended to OUT/finished.txt (name, sha256 of
# final.vtsp, bytes, steps, wall seconds). OUT/locks/<run>.lock is held by the run's processes, so a second sweep
# skips a live run. Run it from a dedicated worktree at the pinned commit (the conductor, 2026-10-20T10:00Z).
# Optional: NOTIFY_PANE (a herdr pane: a line at every 5th finished run, at every crash and at the end),
# STATUS_FILE (one status line, overwritten), STAGE (the label in both, e.g. "stage 1").
# The tree must be clean (tracked files): every run records the commit it ran from.
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$REPO/eval/reproducibility/textspot2-20261020
OUT=${1:?usage: sweep.sh OUT PLAN [RUN ...]}; PLAN=${2:?usage: sweep.sh OUT PLAN [RUN ...]}; shift 2
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
PLAN=$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")
MAX_PAR=${MAX_PAR:-5}
STAGE=${STAGE:-sweep}
PY="$REPO/.venv-ci/bin/python"
mkdir -p "$OUT/logs" "$OUT/locks"
utc() { date -u +%FT%TZ; }

if [ -n "$(cd "$REPO" && git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
  echo "the checkout has modified tracked files: commit first (the runs record the commit)" >&2; exit 2
fi

CBIN=$OUT/cbuild/vocr_textspot2_test
mkdir -p "$OUT/cbuild"
if ! ( cd "$REPO/runtime" && ${CC:-cc} -std=c99 -O2 -Wall -Wextra -Werror vocr_lines.c vocr_lines_rows.c \
         vocr_textspot2.c vocr_textspot2_test.c -o "$CBIN.tmp" ); then
  echo "the C driver runtime/vocr_textspot2_test does not build" >&2; exit 2
fi
mv "$CBIN.tmp" "$CBIN"
ALL=$("$PY" "$HERE/textspot2_plan.py" list "$PLAN")
if [ $# -gt 0 ]; then RUNS="$*"; else RUNS=$ALL; fi
N_ALL=$(echo "$ALL" | wc -w)

notify() {         # one line to the conductor's pane, if NOTIFY_PANE is set (never fails the sweep)
  [ -n "${NOTIFY_PANE:-}" ] || return 0
  timeout 30 herdr pane send-text "$NOTIFY_PANE" "[textspot2-train -> textspot2] $*" < /dev/null > /dev/null 2>&1 || true
  timeout 30 herdr pane send-keys "$NOTIFY_PANE" Enter < /dev/null > /dev/null 2>&1 || true
}

status() {         # "STAGE: d/N done, r running, c crashes" to STATUS_FILE (and back to the caller)
  local done running crashes r
  done=0; running=0
  for r in $ALL; do
    if [ -e "$OUT/$r/done.json" ]; then done=$((done + 1)); elif [ -e "$OUT/$r/run.json" ]; then running=$((running + 1)); fi
  done
  crashes=0; [ -e "$OUT/crashes.jsonl" ] && crashes=$(wc -l < "$OUT/crashes.jsonl")
  local line="$STAGE: $done/$N_ALL done, $running running, $crashes crashes ($(utc))"
  [ -n "${STATUS_FILE:-}" ] && echo "textspot2-train $line" > "$STATUS_FILE"
  echo "$line"
}

finished() {       # after a run's check: the status, and the conductor's line at every 5th finished run
  local run=$1 line n
  exec 8>"$OUT/locks/notify.lock"
  flock 8
  line=$(status)
  n=$( [ -e "$OUT/finished.txt" ] && wc -l < "$OUT/finished.txt" || echo 0)
  if [ $((n % 5)) -eq 0 ] && [ "$n" -gt 0 ] && ! grep -qx "$n" "$OUT/locks/notified" 2>/dev/null; then
    echo "$n" >> "$OUT/locks/notified"
    notify "$line; last: $run ($(tail -n 1 "$OUT/finished.txt" | cut -d' ' -f2 | cut -c1-12)..., $(tail -n 1 "$OUT/finished.txt" | cut -d' ' -f3) B)"
  fi
  flock -u 8
}

one() {            # one run: up to 3 attempts, each from scratch, then the check
  local run=$1 attempt rc k secs
  exec 7>"$OUT/locks/$run.lock"
  if ! flock -n 7; then echo "$(utc) $run: live in another sweep, skipped" >> "$OUT/sweep.log"; return 0; fi
  secs=$("$PY" "$HERE/textspot2_plan.py" timeout "$PLAN" "$run")
  for attempt in 1 2 3; do
    [ -e "$OUT/$run/done.json" ] && break
    if [ -e "$OUT/$run" ]; then
      k=1; while [ -e "$OUT/$run.failed$k" ]; do k=$((k + 1)); done
      mv "$OUT/$run" "$OUT/$run.failed$k"
      if [ -e "$OUT/logs/$run.log" ]; then mv "$OUT/logs/$run.log" "$OUT/logs/$run.failed$k.log"; fi
      echo "{\"run\": \"$run\", \"moved_to\": \"$run.failed$k\", \"restart\": \"from scratch, the same seed\", \"utc\": \"$(utc)\"}" >> "$OUT/restarts.jsonl"
    fi
    local -a ARGV=()
    mapfile -d '' -t ARGV < <("$PY" "$HERE/textspot2_plan.py" args "$PLAN" "$run" --out-dir "$OUT")
    if [ "${#ARGV[@]}" -lt 4 ]; then echo "$(utc) $run: no command line from plan.py" >> "$OUT/sweep.log"; rc=97; break; fi
    echo "$(utc) $run: attempt $attempt (timeout ${secs}s): ${ARGV[*]}" >> "$OUT/sweep.log"
    rc=0
    ( cd "$REPO" && HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout "$secs" .venv/bin/python "${ARGV[@]}" \
        < /dev/null > "$OUT/logs/$run.log" 2>&1 ) || rc=$?
    echo "$(utc) $run: attempt $attempt exit $rc" >> "$OUT/sweep.log"
    [ -e "$OUT/$run/done.json" ] && break
    echo "{\"run\": \"$run\", \"attempt\": $attempt, \"exit\": $rc, \"utc\": \"$(utc)\"}" >> "$OUT/crashes.jsonl"
    notify "$STAGE: CRASH $run attempt $attempt exit $rc ($(tail -n 2 "$OUT/logs/$run.log" 2>/dev/null | tr '\n' ' ' | cut -c1-200)); restarting from scratch if attempts remain"
  done
  if [ -e "$OUT/$run/done.json" ]; then
    ( cd "$REPO" && "$PY" "$HERE/check_run.py" "$OUT/$run" --finished "$OUT/finished.txt" --cbin "$CBIN" ) \
      >> "$OUT/sweep.log" 2>&1 \
      || { echo "$(utc) $run: CHECK FAILED" >> "$OUT/sweep.log"; notify "$STAGE: CHECK FAILED $run (see $OUT/sweep.log)"; }
    finished "$run"
  else
    echo "$(utc) $run: NOT FINISHED after 3 attempts" >> "$OUT/sweep.log"
    notify "$STAGE: $run NOT FINISHED after 3 attempts"
    status > /dev/null
  fi
}

echo "$(utc) sweep start: $STAGE, MAX_PAR $MAX_PAR, head $(cd "$REPO" && git rev-parse HEAD), plan $PLAN ($(sha256sum "$PLAN" | cut -c1-64)), runs $RUNS" >> "$OUT/sweep.log"
status > /dev/null
for run in $RUNS; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 15; done
  one "$run" &
  sleep 2
done
wait
line=$(status)
echo "$(utc) SWEEP_DONE $line" >> "$OUT/sweep.log"
notify "$STAGE: SWEEP_DONE: $line"
echo "SWEEP_DONE $line"

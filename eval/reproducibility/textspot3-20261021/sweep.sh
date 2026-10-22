#!/usr/bin/env bash
# textspot3's training sweep (docs/textspot3.md "Arms and stages"; issue #108): every run of a plan (textspot3_plan.py
# stage1 --family T|M, or stage2) through heavy.sh, 5 threads, in the plan's order (longest first), at most MAX_PAR
# runs at once across every sweep that shares OUT (slot locks OUT/locks/slot-<k>.lock, k = 1..MAX_PAR: a T and an M
# sweep of stage 1 running side by side hold MAX_PAR runs between them).
#   sweep.sh OUT PLAN [RUN ...]          RUN: names of the plan (default: all of them)
# Every run trains to its last step; nothing is selected or stopped early on any score. A run that exits without
# done.json (a crash or its timeout, the plan's hang guard) is moved to OUT/<run>.failed<k> and restarted from scratch
# with the same seed, at most 3 attempts; OUT/crashes.jsonl and OUT/restarts.jsonl record it. A finished run is
# checked by check_run.py (its blob in the twin of its VTSP version on procedural frames, and the C driver of that
# version, built once into OUT/cbuild from this checkout's sources, giving the twin's result: runtime/vocr_textspot2_test
# for family T, runtime/vocr_textspot3_test for family M) and appended to OUT/finished.txt (name, sha256 of final.vtsp,
# bytes, steps, wall seconds); the check also holds the run to its plan record (recipe, family, shape, seed, steps,
# extras, teacher). OUT/locks/<run>.lock is held by the run's processes, so a second sweep skips a live run.
# Run it from a detached worktree at the pinned commit (the conductor's GO).
# EXTRA_ARGS (scratch sweeps only, never a stage's sweep): arguments appended to every run's trainer command line, e.g.
# "--steps 300 --ema-start 100 --log-every 20"; they are written in the start line, and the plan-record check is
# skipped (the runs are not the plan's). SWEEP_POLL: seconds between slot polls (15).
# Optional: NOTIFY_PANE (a herdr pane: a line at every multiple of 5 finished runs of the plan, each once, at every
# crash and at the end),
# STATUS_FILE (one status line, overwritten), STAGE (the label in both, e.g. "stage 1 T"), SWEEP_TMPDIR (TMPDIR for
# the runs and checks, default /local/home/skerraut/kt/t3s: short, because the loader's AF_UNIX sockets live under it).
# Finished runs and crashes are counted per plan, so several plans can share OUT.
# The tree must be clean (tracked files): every run records the commit it ran from.
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$REPO/eval/reproducibility/textspot3-20261021
OUT=${1:?usage: sweep.sh OUT PLAN [RUN ...]}; PLAN=${2:?usage: sweep.sh OUT PLAN [RUN ...]}; shift 2
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
PLAN=$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")
MAX_PAR=${MAX_PAR:-8}
STAGE=${STAGE:-sweep}
POLL=${SWEEP_POLL:-15}
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
PY="$REPO/.venv-ci/bin/python"
TMPDIR=${SWEEP_TMPDIR:-/local/home/skerraut/kt/t3s}
if [ "${#TMPDIR}" -gt 60 ]; then echo "SWEEP_TMPDIR $TMPDIR: over 60 characters" >&2; exit 2; fi
mkdir -p "$TMPDIR" "$OUT/logs" "$OUT/locks"
export TMPDIR
utc() { date -u +%FT%TZ; }

if [ -n "$(cd "$REPO" && git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
  echo "the checkout has modified tracked files: commit first (the runs record the commit)" >&2; exit 2
fi

ALL=$("$PY" "$HERE/textspot3_plan.py" list "$PLAN")
FAMS=$(for r in $ALL; do "$PY" "$HERE/textspot3_plan.py" family "$PLAN" "$r"; done | sort -u | tr '\n' ' ')
mkdir -p "$OUT/cbuild"
CBIN2=$OUT/cbuild/vocr_textspot2_test
CBIN3=$OUT/cbuild/vocr_textspot3_test
if ! ( cd "$REPO/runtime" && ${CC:-cc} -std=c99 -O2 -Wall -Wextra -Werror vocr_lines.c vocr_lines_rows.c \
         vocr_textspot2.c vocr_textspot2_test.c -o "$CBIN2.tmp" ); then
  echo "the C driver runtime/vocr_textspot2_test does not build" >&2; exit 2
fi
mv "$CBIN2.tmp" "$CBIN2"
CHECK_C="--cbin2 $CBIN2"
if [ "${#EXTRA[@]}" -eq 0 ]; then CHECK_C="$CHECK_C --plan $PLAN"; fi
if [ -e "$REPO/runtime/vocr_textspot3.c" ] && [ -e "$REPO/runtime/vocr_textspot3_test.c" ]; then
  if ! ( cd "$REPO/runtime" && ${CC:-cc} -std=c99 -O2 -Wall -Wextra -Werror vocr_lines.c vocr_lines_rows.c \
           vocr_textspot3.c vocr_textspot3_test.c -o "$CBIN3.tmp" ); then
    echo "the C driver runtime/vocr_textspot3_test does not build" >&2; exit 2
  fi
  mv "$CBIN3.tmp" "$CBIN3"
  CHECK_C="$CHECK_C --cbin3 $CBIN3"
elif [[ " $FAMS " == *" M "* ]]; then
  echo "the plan has family M runs and runtime/vocr_textspot3_test.c is not in this checkout" >&2; exit 2
fi
NOTIFIED=$OUT/locks/notified-$(basename "$PLAN" .jsonl)
if [ $# -gt 0 ]; then RUNS="$*"; else RUNS=$ALL; fi
N_ALL=$(echo "$ALL" | wc -w)

notify() {         # one line to the conductor's pane, if NOTIFY_PANE is set (never fails the sweep)
  [ -n "${NOTIFY_PANE:-}" ] || return 0
  timeout 30 herdr pane send-text "$NOTIFY_PANE" "[textspot3-train -> textspot3] $*" < /dev/null > /dev/null 2>&1 || true
  timeout 30 herdr pane send-keys "$NOTIFY_PANE" Enter < /dev/null > /dev/null 2>&1 || true
}

status() {         # "STAGE: d/N done, r running, c crashes" (this plan's runs) to STATUS_FILE and the caller
  local done running crashes r c
  done=0; running=0; crashes=0
  for r in $ALL; do
    if [ -e "$OUT/$r/done.json" ]; then done=$((done + 1)); elif [ -e "$OUT/$r/run.json" ]; then running=$((running + 1)); fi
    if [ -e "$OUT/crashes.jsonl" ]; then
      c=$(grep -cF "\"run\": \"$r\"," "$OUT/crashes.jsonl" || true); crashes=$((crashes + ${c:-0}))
    fi
  done
  local line="$STAGE: $done/$N_ALL done, $running running, $crashes crashes ($(utc))"
  [ -n "${STATUS_FILE:-}" ] && echo "textspot3-train $line" > "$STATUS_FILE"
  echo "$line"
}

finished() {       # after a run's check: the status, and the conductor's line at every multiple of 5 finished runs
  local run=$1 line n r k  # of the plan, each once (two runs can finish within one check: the count goes 4 -> 6)
  exec 8>"$OUT/locks/notify.lock"
  flock 8
  line=$(status)
  n=0; for r in $ALL; do [ -e "$OUT/$r/done.json" ] && n=$((n + 1)); done
  for k in $(seq 5 5 "$n"); do
    if ! grep -qx "$k" "$NOTIFIED" 2>/dev/null; then
      echo "$k" >> "$NOTIFIED"
      notify "$line ($k-run notice); last: $run ($(tail -n 1 "$OUT/finished.txt" | cut -d' ' -f2 | cut -c1-12)..., $(tail -n 1 "$OUT/finished.txt" | cut -d' ' -f3) B)"
    fi
  done
  flock -u 8
}

one() {            # one run: up to 3 attempts, each from scratch, then the check
  local run=$1 attempt rc k secs
  exec 7>"$OUT/locks/$run.lock"
  if ! flock -n 7; then echo "$(utc) $run: live in another sweep, skipped" >> "$OUT/sweep.log"; return 0; fi
  secs=$("$PY" "$HERE/textspot3_plan.py" timeout "$PLAN" "$run")
  for attempt in 1 2 3; do
    [ -e "$OUT/$run/done.json" ] && break
    if [ -e "$OUT/$run" ]; then
      k=1; while [ -e "$OUT/$run.failed$k" ]; do k=$((k + 1)); done
      mv "$OUT/$run" "$OUT/$run.failed$k"
      if [ -e "$OUT/logs/$run.log" ]; then mv "$OUT/logs/$run.log" "$OUT/logs/$run.failed$k.log"; fi
      echo "{\"run\": \"$run\", \"moved_to\": \"$run.failed$k\", \"restart\": \"from scratch, the same seed\", \"utc\": \"$(utc)\"}" >> "$OUT/restarts.jsonl"
    fi
    local -a ARGV=()
    mapfile -d '' -t ARGV < <("$PY" "$HERE/textspot3_plan.py" args "$PLAN" "$run" --out-dir "$OUT")
    if [ "${#ARGV[@]}" -lt 4 ]; then echo "$(utc) $run: no command line from textspot3_plan.py" >> "$OUT/sweep.log"; rc=97; break; fi
    ARGV+=("${EXTRA[@]}")
    echo "$(utc) $run: attempt $attempt (timeout ${secs}s, slot $SLOTK): ${ARGV[*]}" >> "$OUT/sweep.log"
    rc=0
    ( cd "$REPO" && HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout "$secs" .venv/bin/python "${ARGV[@]}" \
        < /dev/null > "$OUT/logs/$run.log" 2>&1 ) || rc=$?
    echo "$(utc) $run: attempt $attempt exit $rc" >> "$OUT/sweep.log"
    [ -e "$OUT/$run/done.json" ] && break
    echo "{\"run\": \"$run\", \"attempt\": $attempt, \"exit\": $rc, \"utc\": \"$(utc)\"}" >> "$OUT/crashes.jsonl"
    notify "$STAGE: CRASH $run attempt $attempt exit $rc ($(tail -n 2 "$OUT/logs/$run.log" 2>/dev/null | tr '\n' ' ' | cut -c1-200)); restarting from scratch if attempts remain"
  done
  if [ -e "$OUT/$run/done.json" ]; then
    # shellcheck disable=SC2086
    ( cd "$REPO" && "$PY" "$HERE/check_run.py" "$OUT/$run" --finished "$OUT/finished.txt" $CHECK_C ) \
      >> "$OUT/sweep.log" 2>&1 \
      || { echo "$(utc) $run: CHECK FAILED" >> "$OUT/sweep.log"; notify "$STAGE: CHECK FAILED $run (see $OUT/sweep.log)"; }
    finished "$run"
  else
    echo "$(utc) $run: NOT FINISHED after 3 attempts" >> "$OUT/sweep.log"
    notify "$STAGE: $run NOT FINISHED after 3 attempts"
    status > /dev/null
  fi
}

take_slot() {      # a free slot of MAX_PAR, shared by every sweep of OUT: its descriptor in SLOTFD, its number in SLOTK
  local k fd
  for k in $(seq 1 "$MAX_PAR"); do
    exec {fd}>"$OUT/locks/slot-$k.lock"
    if flock -n "$fd"; then SLOTFD=$fd; SLOTK=$k; return 0; fi
    exec {fd}>&-
  done
  return 1
}

echo "$(utc) sweep start: $STAGE, MAX_PAR $MAX_PAR, head $(cd "$REPO" && git rev-parse HEAD), plan $PLAN ($(sha256sum "$PLAN" | cut -c1-64)), runs $RUNS${EXTRA_ARGS:+, extra args $EXTRA_ARGS}" >> "$OUT/sweep.log"
status > /dev/null
SLOTFD=; SLOTK=
for run in $RUNS; do
  until take_slot; do sleep "$POLL"; done
  one "$run" &              # the run inherits the slot's descriptor and holds the slot until it ends
  exec {SLOTFD}>&-
  sleep 2
done
wait
line=$(status)
echo "$(utc) SWEEP_DONE $line" >> "$OUT/sweep.log"
notify "$STAGE: SWEEP_DONE: $line"
echo "SWEEP_DONE $line"

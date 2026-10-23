#!/usr/bin/env bash
# queue.sh: run stride16's jobs from jobs.txt in file order, at most MAX (default 8) at once, each through heavy.sh
# (HEAVY_SLOTS=12, 5 threads). A copy of VOCR5's queue.sh (eval/reproducibility/vocr5-20261015/queue.sh) with
# stride16's paths and eval/stride16_run.sh. jobs.txt is re-read every loop, so jobs can be appended or reordered
# while it runs. A line: NAME RUNNER_DIR TIMEOUT_S ENV TIER STRIDE STAGE SEED   (ENV: '-' or VAR=VAL[,VAR=VAL...];
# '#' starts a comment). State: state/NAME.running | .done | .failed (with the exit status); logs/NAME.log.
# touch state/STOP to stop starting new jobs (running ones go on).
set -u
S16=/local/home/skerraut/herd-vita-ocr/shared/artifacts/stride16
HERD=/local/home/skerraut/herd-vita-ocr
MAX=${MAX:-8}
mkdir -p "$S16/logs" "$S16/state"
echo "$(date -u +%FT%TZ) queue start pid $$ max $MAX" >> "$S16/logs/queue.log"
run_job() {   # name runner tmo envs tier stride stage seed
  local name=$1 runner=$2 tmo=$3 envs=$4 tier=$5 stride=$6 stage=$7 seed=$8 rc
  local -a ev=()
  [ "$envs" != "-" ] && IFS=',' read -r -a ev <<< "$envs"
  echo "$(date -u +%FT%TZ) start $name ($runner $tier $stride $stage $seed ${envs})" >> "$S16/logs/queue.log"
  env "${ev[@]}" HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout "$tmo" "$runner/eval/stride16_run.sh" "$tier" "$stride" \
    "$stage" "$seed" >> "$S16/logs/$name.log" 2>&1 < /dev/null
  rc=$?
  if [ "$rc" = 0 ]; then mv "$S16/state/$name.running" "$S16/state/$name.done"
  else echo "$rc" > "$S16/state/$name.failed"; rm -f "$S16/state/$name.running"; fi
  echo "$(date -u +%FT%TZ) end $name rc=$rc" >> "$S16/logs/queue.log"
}
while :; do
  running=$(find "$S16/state" -name '*.running' | wc -l)
  if [ ! -e "$S16/state/STOP" ] && [ "$running" -lt "$MAX" ] && [ -s "$S16/jobs.txt" ]; then
    while read -r name runner tmo envs tier stride stage seed _; do
      case "$name" in ''|\#*) continue ;; esac
      [ -e "$S16/state/$name.running" ] || [ -e "$S16/state/$name.done" ] || [ -e "$S16/state/$name.failed" ] && continue
      touch "$S16/state/$name.running"
      run_job "$name" "$runner" "$tmo" "$envs" "$tier" "$stride" "$stage" "$seed" &
      break
    done < "$S16/jobs.txt"
  fi
  sleep 20
done

#!/usr/bin/env bash
# queue.sh: run VOCR4's jobs from jobs.txt in file order, at most MAX (default 8) at once, each through heavy.sh
# (HEAVY_SLOTS=10, 5 threads). jobs.txt is re-read every loop, so jobs can be appended or reordered while it runs.
# A line: NAME RUNNER_DIR TIMEOUT_S ENV SIZE ARM SEED   (ENV: '-' or VAR=VAL[,VAR=VAL...]; '#' starts a comment)
# State: state/NAME.running | .done | .failed (with the exit status); logs/NAME.log. touch state/STOP to stop
# starting new jobs (running ones go on).
set -u
V4=/local/home/skerraut/herd-vita-ocr/shared/artifacts/vocr4-train
HERD=/local/home/skerraut/herd-vita-ocr
MAX=${MAX:-8}
mkdir -p "$V4/logs" "$V4/state"
echo "$(date -u +%FT%TZ) queue start pid $$ max $MAX" >> "$V4/logs/queue.log"
run_job() {   # name runner tmo envs size arm seed
  local name=$1 runner=$2 tmo=$3 envs=$4 size=$5 arm=$6 seed=$7 rc
  local -a ev=()
  [ "$envs" != "-" ] && IFS=',' read -r -a ev <<< "$envs"
  echo "$(date -u +%FT%TZ) start $name ($runner $size $arm $seed ${envs})" >> "$V4/logs/queue.log"
  env "${ev[@]}" HEAVY_SLOTS=10 "$HERD/heavy.sh" timeout "$tmo" "$runner/eval/vocr4_run.sh" "$size" "$arm" "$seed" \
    > "$V4/logs/$name.log" 2>&1 < /dev/null
  rc=$?
  if [ "$rc" = 0 ]; then mv "$V4/state/$name.running" "$V4/state/$name.done"
  else echo "$rc" > "$V4/state/$name.failed"; rm -f "$V4/state/$name.running"; fi
  echo "$(date -u +%FT%TZ) end $name rc=$rc" >> "$V4/logs/queue.log"
}
while :; do
  running=$(find "$V4/state" -name '*.running' | wc -l)
  if [ ! -e "$V4/state/STOP" ] && [ "$running" -lt "$MAX" ] && [ -s "$V4/jobs.txt" ]; then
    while read -r name runner tmo envs size arm seed _; do
      case "$name" in ''|\#*) continue ;; esac
      [ -e "$V4/state/$name.running" ] || [ -e "$V4/state/$name.done" ] || [ -e "$V4/state/$name.failed" ] && continue
      touch "$V4/state/$name.running"
      run_job "$name" "$runner" "$tmo" "$envs" "$size" "$arm" "$seed" &
      break
    done < "$V4/jobs.txt"
  fi
  sleep 20
done

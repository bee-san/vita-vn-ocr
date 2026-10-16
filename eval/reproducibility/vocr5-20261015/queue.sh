#!/usr/bin/env bash
# queue.sh: run VOCR5's jobs from jobs.txt in file order, at most MAX (default 8) at once, each through heavy.sh
# (HEAVY_SLOTS=10, 5 threads). A copy of VOCR4's queue.sh (eval/reproducibility/vocr4-20261013/queue.sh) with VOCR5's
# paths and eval/vocr5_run.sh. jobs.txt is re-read every loop, so jobs can be appended or reordered while it runs.
# A line: NAME RUNNER_DIR TIMEOUT_S ENV TIER STAGE SEED   (ENV: '-' or VAR=VAL[,VAR=VAL...]; '#' starts a comment)
# State: state/NAME.running | .done | .failed (with the exit status); logs/NAME.log. touch state/STOP to stop
# starting new jobs (running ones go on).
set -u
V5=/local/home/skerraut/herd-vita-ocr/shared/artifacts/vocr5-train
HERD=/local/home/skerraut/herd-vita-ocr
MAX=${MAX:-8}
mkdir -p "$V5/logs" "$V5/state"
echo "$(date -u +%FT%TZ) queue start pid $$ max $MAX" >> "$V5/logs/queue.log"
run_job() {   # name runner tmo envs tier stage seed
  local name=$1 runner=$2 tmo=$3 envs=$4 tier=$5 stage=$6 seed=$7 rc
  local -a ev=()
  [ "$envs" != "-" ] && IFS=',' read -r -a ev <<< "$envs"
  echo "$(date -u +%FT%TZ) start $name ($runner $tier $stage $seed ${envs})" >> "$V5/logs/queue.log"
  env "${ev[@]}" HEAVY_SLOTS=10 "$HERD/heavy.sh" timeout "$tmo" "$runner/eval/vocr5_run.sh" "$tier" "$stage" "$seed" \
    >> "$V5/logs/$name.log" 2>&1 < /dev/null
  rc=$?
  if [ "$rc" = 0 ]; then mv "$V5/state/$name.running" "$V5/state/$name.done"
  else echo "$rc" > "$V5/state/$name.failed"; rm -f "$V5/state/$name.running"; fi
  echo "$(date -u +%FT%TZ) end $name rc=$rc" >> "$V5/logs/queue.log"
}
while :; do
  running=$(find "$V5/state" -name '*.running' | wc -l)
  if [ ! -e "$V5/state/STOP" ] && [ "$running" -lt "$MAX" ] && [ -s "$V5/jobs.txt" ]; then
    while read -r name runner tmo envs tier stage seed _; do
      case "$name" in ''|\#*) continue ;; esac
      [ -e "$V5/state/$name.running" ] || [ -e "$V5/state/$name.done" ] || [ -e "$V5/state/$name.failed" ] && continue
      touch "$V5/state/$name.running"
      run_job "$name" "$runner" "$tmo" "$envs" "$tier" "$stage" "$seed" &
      break
    done < "$V5/jobs.txt"
  fi
  sleep 20
done

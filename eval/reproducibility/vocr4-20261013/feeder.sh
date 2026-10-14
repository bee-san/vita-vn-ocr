#!/usr/bin/env bash
# feeder.sh RUNNER: every 60 s, for each finished M pretrain (state/pre_<SIZE>_M3_seed<S>.done), put its fine-tune job
# v4m_<SIZE>_seed<S> at the top of jobs.txt (once), so queue.sh starts it next. Exits when all six are queued.
set -u
V4=/local/home/skerraut/herd-vita-ocr/shared/artifacts/vocr4-train
RUNNER=$1
declare -A TMO=([FL10]=14400 [H15]=25200 [W20]=32400)
while :; do
  left=0
  for size in FL10 H15 W20; do
    for s in 1 2; do
      job=v4m_${size}_seed$s
      grep -q "^$job " "$V4/jobs.txt" && continue
      left=$((left + 1))
      [ -e "$V4/state/pre_${size}_M3_seed$s.done" ] || continue
      { head -n 1 "$V4/jobs.txt"; echo "$job $RUNNER ${TMO[$size]} - $size M $s"; tail -n +2 "$V4/jobs.txt"; } \
        > "$V4/jobs.txt.feed" && mv "$V4/jobs.txt.feed" "$V4/jobs.txt"
      echo "$(date -u +%FT%TZ) feeder queued $job" >> "$V4/logs/queue.log"
      left=$((left - 1))
    done
  done
  [ "$left" = 0 ] && { echo "$(date -u +%FT%TZ) feeder done" >> "$V4/logs/queue.log"; exit 0; }
  sleep 60
done

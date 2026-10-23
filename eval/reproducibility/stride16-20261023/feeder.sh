#!/usr/bin/env bash
# feeder.sh RUNNER: every 60 s, for each finished stride16 pretrain (state/pre<STRIDE>_<T>_seed<S>.done), put its
# fine-tune (s<STRIDE>f_<T>_seed<S>; Large first) at the top of jobs.txt, once, from RUNNER (a worktree tagged
# stride16-runner-<sha>). A copy of VOCR5's feeder.sh with stride16's names (arm F only). Exits when all are queued.
set -u
S16=/local/home/skerraut/herd-vita-ocr/shared/artifacts/stride16
RUNNER=$1
declare -A TMO=([S]=14400 [N]=25200 [L]=36000)
ARMS="L:16 N:16 N:12 S:16"
while :; do
  left=0
  for a in $ARMS; do
    tier=${a%%:*}; stride=${a##*:}
    for s in 1 2; do
      job=s${stride}f_${tier}_seed$s
      grep -q "^$job " "$S16/jobs.txt" && continue
      left=$((left + 1))
      [ -e "$S16/state/pre${stride}_${tier}_seed$s.done" ] || continue
      { echo "$job $RUNNER ${TMO[$tier]} - $tier $stride F $s"; cat "$S16/jobs.txt"; } > "$S16/jobs.txt.feed" \
        && mv "$S16/jobs.txt.feed" "$S16/jobs.txt"
      echo "$(date -u +%FT%TZ) feeder queued $job" >> "$S16/logs/queue.log"
      left=$((left - 1))
    done
  done
  [ "$left" = 0 ] && { echo "$(date -u +%FT%TZ) feeder done" >> "$S16/logs/queue.log"; exit 0; }
  sleep 60
done

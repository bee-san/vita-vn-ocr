#!/usr/bin/env bash
# feeder.sh RUNNER: every 60 s, for each finished VOCR5 pretrain (state/pre5_<T>_seed<S>.done), put its two fine-tunes
# (v5p_<T>_seed<S>, v5f_<T>_seed<S>; Large first) at the top of jobs.txt, once, from RUNNER (a worktree whose
# eval/vocr5_pools.sha256 is committed). A copy of VOCR4's feeder.sh with VOCR5's names. Exits when all are queued.
set -u
V5=/local/home/skerraut/herd-vita-ocr/shared/artifacts/vocr5-train
RUNNER=$1
declare -A TMO=([S]=14400 [N]=25200 [L]=36000)
declare -A SEEDS=([S]="1 2" [N]="1 2 3" [L]="1 2 3")
while :; do
  left=0
  for tier in S N L; do
    for s in ${SEEDS[$tier]}; do
      for stage in P F; do
        job=v5$(echo $stage | tr 'PF' 'pf')_${tier}_seed$s
        grep -q "^$job " "$V5/jobs.txt" && continue
        left=$((left + 1))
        [ -e "$V5/state/pre5_${tier}_seed$s.done" ] || continue
        { echo "$job $RUNNER ${TMO[$tier]} - $tier $stage $s"; cat "$V5/jobs.txt"; } > "$V5/jobs.txt.feed" \
          && mv "$V5/jobs.txt.feed" "$V5/jobs.txt"
        echo "$(date -u +%FT%TZ) feeder queued $job" >> "$V5/logs/queue.log"
        left=$((left - 1))
      done
    done
  done
  [ "$left" = 0 ] && { echo "$(date -u +%FT%TZ) feeder done" >> "$V5/logs/queue.log"; exit 0; }
  sleep 60
done

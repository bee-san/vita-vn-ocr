#!/usr/bin/env bash
# Run order after 21:22Z (replaces queue.sh, which had launched the 10 study-2 FL10/H15 runs): wait for study 2's
# last H15 run (pid 740844) to take its slot, then VOCR3's width-2.0 candidates (eval/vocr3_protocol.json
# amendment 1), study 2's step-curve runs (64k, 128k; VOCR3-FL10 candidates), then the learning curve. Each job is handed
# to heavy.sh (HEAVY_SLOTS=10: training never takes slots 11-13) only after the previous one has taken a slot.
HERD=/local/home/skerraut/herd-vita-ocr; S3=$HERD/shared/artifacts/synth3; export HERD
REPO=/local/home/skerraut/herd-vita-oct12/clones/runner-c464aa8; L=$HERD/locks/heavy.log
JOBS=(
 "v3 W20 C_mix 1" "v3 W20 A_s2 1" "v3 W20 B_s3 1"
 "s2 FL10 C_mix 1 64000" "s2 FL10 C_mix 1 128000"
 "cv 025M 1" "cv 025M 2" "cv 050M 1" "cv 050M 2" "cv 2M 1" "cv 2M 2"
)
cd "$REPO"
until grep -q " START 740844 " $L || ! kill -0 740844 2>/dev/null; do sleep 15; done
echo "$(date -u +%FT%TZ) study2_H15_C_mix_2 has its slot"
for j in "${JOBS[@]}"; do
  set -- $j; kind=$1; shift
  case $kind in
    v3) name="vocr3_$(echo "$*" | tr ' ' '_')"; log=$S3/vocr3/logs/$name.log; cmd=(timeout 129600 bash eval/vocr3_run.sh "$@") ;;
    s2) name="study2_$(echo "$*" | tr ' ' '_')"; log=$S3/study2/logs/$name.log; cmd=(timeout 172800 bash eval/synth3_study2_run.sh "$@") ;;
    cv) name="curve_$1_seed$2"; log=$S3/curve/logs/$name.log; cmd=(timeout 36000 bash eval/synth3_curve_run.sh "$@") ;;
  esac
  HEAVY_SLOTS=10 nohup $HERD/heavy.sh "${cmd[@]}" > "$log" 2>&1 < /dev/null &
  pid=$!
  echo "$(date -u +%FT%TZ) queued $name pid $pid"
  until grep -q " START $pid " $L || ! kill -0 $pid 2>/dev/null; do sleep 15; done
  echo "$(date -u +%FT%TZ) started $name ($(grep " START $pid " $L | awk '{print $2}'))"
  sleep 5
done
echo "$(date -u +%FT%TZ) QUEUE_EMPTY"

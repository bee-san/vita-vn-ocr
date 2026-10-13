#!/usr/bin/env bash
# Run order from 2026-10-13 00:20Z (replaces queue2.sh after it launched curve_025M_seed1): VOCR3 amendment 2's H15
# B_s3 candidate first, then the rest of the learning curve. HEAVY_SLOTS=10; each job is handed to heavy.sh only
# after the previous one has taken a slot.
HERD=/local/home/skerraut/herd-vita-ocr; S3=$HERD/shared/artifacts/synth3; export HERD
REPO=/local/home/skerraut/herd-vita-oct12/clones/runner-c464aa8; L=$HERD/locks/heavy.log
JOBS=("v3 H15 B_s3 1" "cv 025M 2" "cv 050M 1" "cv 050M 2" "cv 2M 1" "cv 2M 2")
cd "$REPO"
for j in "${JOBS[@]}"; do
  set -- $j; kind=$1; shift
  case $kind in
    v3) name="vocr3_$(echo "$*" | tr ' ' '_')"; log=$S3/vocr3/logs/$name.log; cmd=(timeout 172800 bash eval/vocr3_run.sh "$@") ;;
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

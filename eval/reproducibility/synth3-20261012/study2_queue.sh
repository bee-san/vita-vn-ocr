#!/usr/bin/env bash
# Study 2's run order (eval/synth3_study2_protocol.json "order"): each job is handed to heavy.sh (HEAVY_SLOTS=10)
# only after the previous one has taken a slot, so my runs start in protocol order.
HERD=/local/home/skerraut/herd-vita-ocr; S3=$HERD/shared/artifacts/synth3; export HERD
REPO=/local/home/skerraut/herd-vita-oct12/clones/synth-dataset; L=$HERD/locks/heavy.log
JOBS=(
 "s2 FL10 A_s2 1" "s2 FL10 C_mix 1" "s2 FL10 A_s2 2" "s2 FL10 C_mix 2" "s2 FL10 A_s2 3" "s2 FL10 C_mix 3"
 "s2 H15 A_s2 1" "s2 H15 C_mix 1" "s2 H15 A_s2 2" "s2 H15 C_mix 2"
 "s2 FL10 C_mix 1 64000" "s2 FL10 C_mix 1 128000"
 "cv 025M 1" "cv 025M 2" "cv 050M 1" "cv 050M 2" "cv 2M 1" "cv 2M 2"
)
cd "$REPO"
for j in "${JOBS[@]}"; do
  set -- $j; kind=$1; shift
  if [ "$kind" = s2 ]; then
    name="study2_$(echo "$*" | tr ' ' '_')"; log=$S3/study2/logs/$name.log
    cmd=(timeout 86400 bash eval/synth3_study2_run.sh "$@")
  else
    [ "$1" = 2M ] && until [ -e $S3/curve/V2EXT_PINNED ]; do sleep 60; done
    name="curve_$1_seed$2"; log=$S3/curve/logs/$name.log
    cmd=(timeout 18000 bash eval/synth3_curve_run.sh "$@")
  fi
  HEAVY_SLOTS=10 nohup $HERD/heavy.sh "${cmd[@]}" > "$log" 2>&1 < /dev/null &
  pid=$!
  echo "$(date -u +%FT%TZ) queued $name pid $pid"
  until grep -q " START $pid " $L || ! kill -0 $pid 2>/dev/null; do sleep 15; done
  echo "$(date -u +%FT%TZ) started $name ($(grep " START $pid " $L | awk '{print $2}'))"
  sleep 5
done
echo "$(date -u +%FT%TZ) QUEUE_EMPTY"

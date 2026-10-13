#!/usr/bin/env bash
# Run order from 2026-10-13 00:46Z (replaces queue3.sh, which queued vocr3_H15_B_s3_1 as pid 2470254): the learning
# curve waits until VOCR3's H15 B_s3 run and study 1b's two D_s3v3 runs (agent synth3-v3, pids 2375558 and 2375666)
# have their slots. HEAVY_SLOTS=10; each job is handed to heavy.sh only after the previous one has taken a slot.
HERD=/local/home/skerraut/herd-vita-ocr; S3=$HERD/shared/artifacts/synth3; export HERD
REPO=/local/home/skerraut/herd-vita-oct12/clones/runner-c464aa8; L=$HERD/locks/heavy.log
for p in 2470254 2375558 2375666; do
  until grep -q " START $p " $L || ! kill -0 $p 2>/dev/null; do sleep 15; done
  echo "$(date -u +%FT%TZ) pid $p has its slot (or ended)"
done
JOBS=("cv 025M 2" "cv 050M 1" "cv 050M 2" "cv 2M 1" "cv 2M 2")
cd "$REPO"
for j in "${JOBS[@]}"; do
  set -- $j; shift
  name="curve_$1_seed$2"; log=$S3/curve/logs/$name.log
  HEAVY_SLOTS=10 nohup $HERD/heavy.sh timeout 36000 bash eval/synth3_curve_run.sh "$@" > "$log" 2>&1 < /dev/null &
  pid=$!
  echo "$(date -u +%FT%TZ) queued $name pid $pid"
  until grep -q " START $pid " $L || ! kill -0 $pid 2>/dev/null; do sleep 15; done
  echo "$(date -u +%FT%TZ) started $name ($(grep " START $pid " $L | awk '{print $2}'))"
  sleep 5
done
echo "$(date -u +%FT%TZ) QUEUE_EMPTY"

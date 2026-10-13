#!/usr/bin/env bash
# Run order from 2026-10-13 01:05Z (replaces queue4.sh): after VOCR3's H15 B_s3 run and study 1b's two D_s3v3 runs
# (pids 2470254, 2375558, 2375666) have their slots, VOCR3 amendment 4's rp fine-tunes in the order their
# pretrains finish, then study 2's learning curve, then the two rp runs whose pretrains finish last.
# Each "rp" item waits for its pretrain checkpoint first. HEAVY_SLOTS=10; one waiter at a time.
HERD=/local/home/skerraut/herd-vita-ocr; S3=$HERD/shared/artifacts/synth3; export HERD
REPO=/local/home/skerraut/herd-vita-oct12/clones/runner-0669f38; L=$HERD/locks/heavy.log
for p in 2470254 2375558 2375666; do
  until grep -q " START $p " $L || ! kill -0 $p 2>/dev/null; do sleep 15; done
  echo "$(date -u +%FT%TZ) pid $p has its slot (or ended)"
done
JOBS=(
 "rp FL10 A_s2 1" "rp FL10 C_mix 1" "rp FL10 A_s2 2" "rp FL10 C_mix 2" "rp FL10 A_s2 3" "rp FL10 C_mix 3"
 "rp H15 A_s2 1" "rp H15 C_mix 1" "rp H15 A_s2 2" "rp H15 C_mix 2"
 "rp FL10 C_mix 1 64000" "rp W20 C_mix 1" "rp W20 A_s2 1" "rp W20 B_s3 1"
 "cv 025M 2" "cv 050M 1" "cv 050M 2" "cv 2M 1" "cv 2M 2"
 "rp H15 B_s3 1" "rp FL10 C_mix 1 128000"
)
pre_ckpt() {   # ARCH ARM SEED [PRE]: path of the pretrain's final checkpoint (study 2's or VOCR3's)
  local def; case $1 in FL10|W20) def=32000 ;; H15) def=40000 ;; esac
  local pre=${4:-$def} tag=""; [ "$pre" != "$def" ] && tag="_pre$((pre / 1000))k"
  for d in $S3/study2/runs $S3/vocr3/runs; do
    [ -s "$d/pre_$1_$2${tag}_seed$3/ckpt_step$pre.pt" ] && { echo "$d/pre_$1_$2${tag}_seed$3/ckpt_step$pre.pt"; return 0; }
  done; return 1
}
cd "$REPO"
for j in "${JOBS[@]}"; do
  set -- $j; kind=$1; shift
  case $kind in
    rp) until pre_ckpt "$@" >/dev/null; do sleep 60; done; sleep 90   # the pretrain's checkpoint is complete
        name="vocr3_rp_$(echo "$*" | tr ' ' '_')"; log=$S3/vocr3/logs/$name.log; cmd=(timeout 43200 bash eval/vocr3_rp_run.sh "$@") ;;
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

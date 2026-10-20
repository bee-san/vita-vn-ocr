#!/usr/bin/env bash
# Before the GO, and not runs of the sweep: speed probes (one warm-up step, then 200 timed steps of the recipe's
# schedule; 5 threads) of every stage-1 shape under data and of s4_c24n4 under ema, cls, long and kd, then a
# 300-step smoke run of every recipe on s4_c24n4 (QAT from step 240, export, VTSP, check_run.py), r0 included. On
# textspot's shards (data/manifests/textspot_shards_v1.json + data/manifests/textdet_shards_v1.json: the textspot2
# shards are not pinned yet; --any-shards, never --allow-missing-sources). The EMA starts at step 1 in the probes (so
# its update is timed) and at step 100 in the smoke runs (so the swap runs before QAT); kd's teacher stand-in is
# textspot's winner run (h2_s4_c24n4_seed2, the same shape and stride). Through heavy.sh, MAX_PAR (1) at once.
# PASS=k (k > 1) repeats the speed probes alone into OUT/probe<k> (repeated samples; probe_summary.py averages them).
#   probe.sh OUT          then: probe_summary.py OUT --out JSON
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$REPO/eval/reproducibility/textspot2-20261020
OUT=${1:?usage: probe.sh OUT}
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
MAX_PAR=${MAX_PAR:-1}
PASS=${PASS:-1}
PDIR=probe; [ "$PASS" -gt 1 ] && PDIR=probe$PASS
TS=$HERD/shared/artifacts/textspot; TD=$HERD/shared/artifacts/textdet; PD=$HERD/shared/private/textdet
MANIFESTS="data/manifests/textspot_shards_v1.json data/manifests/textdet_shards_v1.json"
SHARDS="$TS/shards/synth_frames $TD/shards/realplus $TD/shards/teacher $TD/shards/devnative $PD/shards/owner"
TEACHER=$TS/runs/h2_s4_c24n4_seed2
mkdir -p "$OUT/logs" "$OUT/$PDIR" "$OUT/smoke"
utc() { date -u +%FT%TZ; }

job() {            # job probe|smoke RECIPE SHAPE RUNNING
  local kind=$1 recipe=$2 shape=$3 running=$4 seed=1 name dir rc trainer extra
  [ "$recipe" = r0 ] && seed=3
  name=${recipe}_h2_${shape}_seed${seed}; dir=$OUT/$kind/$name
  [ "$kind" = probe ] && dir=$OUT/$PDIR/$name
  if [ -e "$dir/probe.json" ] || [ -e "$dir/done.json" ]; then return 0; fi
  if [ -e "$dir" ]; then mv "$dir" "$dir.failed.$(date -u +%H%M%S)"; fi
  if [ "$recipe" = r0 ]; then
    trainer="train/textspot_train.py --scale h2"
  else
    trainer="train/textspot2_train.py --recipe $recipe --any-shards"
  fi
  if [ "$kind" = probe ]; then extra="--probe 200"; else extra="--steps 300 --log-every 20"; fi
  case "$recipe" in
    ema|long) extra="$extra --ema-start $([ "$kind" = probe ] && echo 1 || echo 100)";;
    kd) extra="$extra --teacher $TEACHER --extras ema,cls --ema-start $([ "$kind" = probe ] && echo 1 || echo 100)";;
  esac
  echo "$(utc) $kind $name start (other jobs: $running; load $(cut -d' ' -f1-3 /proc/loadavg))" >> "$OUT/probe.log"
  rc=0
  # shellcheck disable=SC2086
  ( cd "$REPO" && HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout 3600 .venv/bin/python $trainer --shape "$shape" \
      --seed "$seed" --manifest $MANIFESTS --shards $SHARDS --out "$dir" --threads 5 $extra \
      < /dev/null > "$OUT/logs/$([ "$kind" = probe ] && echo "$PDIR" || echo "$kind")-$name.log" 2>&1 ) || rc=$?
  echo "$(utc) $kind $name exit $rc (load $(cut -d' ' -f1-3 /proc/loadavg))" >> "$OUT/probe.log"
  if [ "$kind" = smoke ] && [ -e "$dir/done.json" ]; then
    ( cd "$REPO" && .venv-ci/bin/python "$HERE/check_run.py" "$dir" ) >> "$OUT/probe.log" 2>&1 \
      || echo "$(utc) smoke $name: CHECK FAILED" >> "$OUT/probe.log"
  fi
}

PROBES="data:s4_c24n4 data:s4p_c8 data:s4p_c6n4 data:s4p_c8n4 data:s4p_c10n4 data:s4p_c12 data:s4_c32n3
        ema:s4_c24n4 cls:s4_c24n4 long:s4_c24n4 kd:s4_c24n4"
SMOKES="lab:s4_c24n4 data:s4_c24n4 ema:s4_c24n4 cls:s4_c24n4 long:s4_c24n4 kd:s4_c24n4 r0:s4_c24n4"
echo "$(utc) probe start: MAX_PAR $MAX_PAR, pass $PASS, head $(cd "$REPO" && git rev-parse HEAD)" >> "$OUT/probe.log"
KINDS="probe smoke"; [ "$PASS" -gt 1 ] && KINDS=probe
for kind in $KINDS; do
  if [ "$kind" = probe ]; then LIST=$PROBES; else LIST=$SMOKES; fi
  for s in $LIST; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
    n=$(jobs -rp | wc -l)
    job "$kind" "${s%%:*}" "${s#*:}" "$n" &
    sleep 2
  done
done
wait
echo "$(utc) PROBE_DONE" >> "$OUT/probe.log"
echo PROBE_DONE

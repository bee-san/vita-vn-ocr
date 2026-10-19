#!/usr/bin/env bash
# Before the GO, and not runs of the sweep: for every family B shape at its scale (seed 1), a speed probe (one
# warm-up step, then 200 steps of a 30,000-step run, timed; 5 threads) and a 300-step smoke run of the whole recipe
# (QAT from step 240, export, VTSP, then check_run.py), on #97's shards: its synthetic regions stand in for the
# full-frame shards, which were not pinned yet. Through heavy.sh, at most MAX_PAR (5) at once, probes first.
#   probe.sh OUT          then: probe_summary.py OUT --out JSON
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:?usage: probe.sh OUT}
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
MAX_PAR=${MAX_PAR:-5}
TD=$HERD/shared/artifacts/textdet; PD=$HERD/shared/private/textdet
MANIFESTS="data/manifests/textdet_shards_v1.json"
SHARDS="$TD/shards/synth $TD/shards/realplus $TD/shards/teacher $TD/shards/devnative $PD/shards/owner"
mkdir -p "$OUT/logs" "$OUT/probe" "$OUT/smoke"
utc() { date -u +%FT%TZ; }

job() {            # job probe|smoke SCALE SHAPE RUNNING (the other jobs running when it was queued)
  local kind=$1 scale=$2 shape=$3 running=$4 name=${2}_${3}_seed1 dir rc extra
  dir=$OUT/$kind/$name
  if [ -e "$dir/probe.json" ] || [ -e "$dir/done.json" ]; then return 0; fi
  if [ -e "$dir" ]; then mv "$dir" "$dir.failed.$(date -u +%H%M%S)"; fi
  if [ "$kind" = probe ]; then extra="--steps 30000 --probe 200"; else extra="--steps 300 --log-every 20"; fi
  echo "$(utc) $kind $name start (other jobs: $running; load $(cut -d' ' -f1-3 /proc/loadavg))" >> "$OUT/probe.log"
  rc=0
  ( cd "$REPO" && HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout 3600 .venv/bin/python train/textspot_train.py \
      --shape "$shape" --scale "$scale" --seed 1 --manifest $MANIFESTS --shards $SHARDS --out "$dir" --threads 5 \
      $extra < /dev/null > "$OUT/logs/$kind-$name.log" 2>&1 ) || rc=$?
  echo "$(utc) $kind $name exit $rc (load $(cut -d' ' -f1-3 /proc/loadavg))" >> "$OUT/probe.log"
  if [ "$kind" = smoke ] && [ -e "$dir/done.json" ]; then
    ( cd "$REPO" && .venv-ci/bin/python eval/reproducibility/textspot-20261019/check_run.py "$dir" ) \
      >> "$OUT/probe.log" 2>&1 || echo "$(utc) smoke $name: CHECK FAILED" >> "$OUT/probe.log"
  fi
}

SHAPES=$(cd "$REPO" && .venv-ci/bin/python -c '
import json
fb = json.load(open("eval/textspot_protocol.json"))["candidates"]["family_b"]
print(" ".join(f"{sc}:{sh}" for sc, shapes in fb["shapes"].items() for sh in shapes))')
echo "$(utc) probe start: MAX_PAR $MAX_PAR, head $(cd "$REPO" && git rev-parse HEAD), shapes $SHAPES" >> "$OUT/probe.log"
for kind in probe smoke; do
  for s in $SHAPES; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
    n=$(jobs -rp | wc -l)
    job "$kind" "${s%%:*}" "${s#*:}" "$n" &
    sleep 2
  done
done
wait
echo "$(utc) PROBE_DONE" >> "$OUT/probe.log"
echo PROBE_DONE

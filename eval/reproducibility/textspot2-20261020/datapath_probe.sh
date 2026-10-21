#!/usr/bin/env bash
# Before the GO, and not runs of the sweep: the data path of the pinned shards (data/manifests/textspot2_shards_v1.json)
# through the sweep's own command lines (textspot2_plan.py args), each with --probe 200 appended (one warm-up step,
# then 200 timed steps; no model): lab_h2_s4_c24n4_seed1 (synth_frames + lab_<source>), data_h2_s4_c24n4_seed1
# (+ more_<source>), and the d2s shapes under data (data_h2_s8_c24n4x2_seed1, data_h2_s8_c48n4x2_seed1); then a
# 300-step d2s smoke run (data_h2_s8_c24n4x2_seed1 with ema from step 100; QAT from step 240, export, VTSP version 2)
# checked by check_run.py with the C driver. One job at a time through heavy.sh (1 heavy slot).
#   datapath_probe.sh OUT          then: datapath_summary.py OUT --out JSON
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$REPO/eval/reproducibility/textspot2-20261020
OUT=${1:?usage: datapath_probe.sh OUT}
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
PY="$REPO/.venv-ci/bin/python"
mkdir -p "$OUT/logs" "$OUT/probe" "$OUT/smoke" "$OUT/cbuild"
utc() { date -u +%FT%TZ; }
"$PY" "$HERE/textspot2_plan.py" stage1 --out "$OUT/stage1.jsonl" > /dev/null
"$PY" "$HERE/textspot2_plan.py" stage2 --extras none --d2s-shapes s8_c24n4x2,s8_c48n4x2 --out "$OUT/d2s.jsonl" > /dev/null
"$PY" "$HERE/textspot2_plan.py" stage2 --extras ema --d2s-shapes s8_c24n4x2 --out "$OUT/d2s_ema.jsonl" > /dev/null
CBIN=$OUT/cbuild/vocr_textspot2_test
( cd "$REPO/runtime" && ${CC:-cc} -std=c99 -O2 -Wall -Wextra -Werror vocr_lines.c vocr_lines_rows.c vocr_textspot2.c \
    vocr_textspot2_test.c -o "$CBIN" )

job() {            # job KIND PLAN RUN EXTRA...
  local kind=$1 plan=$2 run=$3 rc; shift 3
  local -a ARGV=()
  mapfile -d '' -t ARGV < <("$PY" "$HERE/textspot2_plan.py" args "$plan" "$run" --out-dir "$OUT/$kind")
  if [ -e "$OUT/$kind/$run" ]; then mv "$OUT/$kind/$run" "$OUT/$kind/$run.failed.$(date -u +%H%M%S)"; fi
  echo "$(utc) $kind $run start (load $(cut -d' ' -f1-3 /proc/loadavg)): ${ARGV[*]} $*" >> "$OUT/probe.log"
  rc=0
  ( cd "$REPO" && HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout 3600 .venv/bin/python "${ARGV[@]}" "$@" \
      < /dev/null > "$OUT/logs/$kind-$run.log" 2>&1 ) || rc=$?
  echo "$(utc) $kind $run exit $rc (load $(cut -d' ' -f1-3 /proc/loadavg))" >> "$OUT/probe.log"
  if [ "$kind" = smoke ] && [ -e "$OUT/smoke/$run/done.json" ]; then
    ( cd "$REPO" && "$PY" "$HERE/check_run.py" "$OUT/smoke/$run" --cbin "$CBIN" ) >> "$OUT/probe.log" 2>&1 \
      || echo "$(utc) smoke $run: CHECK FAILED" >> "$OUT/probe.log"
  fi
}

echo "$(utc) datapath probe start: head $(cd "$REPO" && git rev-parse HEAD)" >> "$OUT/probe.log"
job probe "$OUT/stage1.jsonl" lab_h2_s4_c24n4_seed1 --probe 200
job probe "$OUT/stage1.jsonl" data_h2_s4_c24n4_seed1 --probe 200
job probe "$OUT/d2s.jsonl" data_h2_s8_c24n4x2_seed1 --probe 200
job probe "$OUT/d2s.jsonl" data_h2_s8_c48n4x2_seed1 --probe 200
job smoke "$OUT/d2s_ema.jsonl" data_h2_s8_c24n4x2_seed1 --steps 300 --log-every 20 --ema-start 100
echo "$(utc) DATAPATH_DONE" >> "$OUT/probe.log"
echo DATAPATH_DONE

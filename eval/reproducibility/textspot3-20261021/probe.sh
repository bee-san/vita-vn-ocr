#!/usr/bin/env bash
# Before the GO, and not runs of the sweep: speed probes (one warm-up step, then 200 timed steps of the recipe's
# schedule; 5 threads) and 300-step smoke runs (QAT from step 240, export, VTSP, check_run.py with the C driver), on the
# pinned shards with the sweep's own command lines (textspot3_plan.py args, seed 1), through heavy.sh, at most MAX_PAR
# (2) at once. The EMA starts at step 1 in the probes (so its update is timed) and at step 100 in the smoke runs (so the
# swap, the BatchNorm re-estimation and the QAT average run before the end). kd's teacher is TEACHER (default: the
# base smoke run of KD_SHAPE, a stand-in of the same family and shape; stage 2's teacher is an M run), with
# --extras none.
#   probe.sh OUT          then: probe_summary.py OUT --out JSON
# PROBES / SMOKES: "recipe:shape ..." lists (default: base on the three registered T shapes and size, clean, long and
# kd on t64n4c16; smokes of base, size, clean, long and kd on t64n4c16). PASS=k (k > 1) repeats the speed probes alone
# into OUT/probe<k> (repeated samples; probe_summary.py averages them).
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$REPO/eval/reproducibility/textspot3-20261021
OUT=${1:?usage: probe.sh OUT}
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
MAX_PAR=${MAX_PAR:-2}
PASS=${PASS:-1}
PDIR=probe; [ "$PASS" -gt 1 ] && PDIR=probe$PASS
KD_SHAPE=${KD_SHAPE:-t64n4c16}
TEACHER=${TEACHER:-$OUT/smoke/base_${KD_SHAPE}_seed1}
PROBES=${PROBES:-"base:t64n4c16 base:t48n8c16d8 base:t64n8c16 size:t64n4c16 clean:t64n4c16 long:t64n4c16"}
SMOKES=${SMOKES:-"base:t64n4c16 size:t64n4c16 clean:t64n4c16 long:t64n4c16"}
KD=${KD-"kd:$KD_SHAPE"}                   # KD="" probes no kd
PY="$REPO/.venv-ci/bin/python"
export TMPDIR=${PROBE_TMPDIR:-/local/home/skerraut/kt/t3p}
mkdir -p "$TMPDIR" "$OUT/logs" "$OUT/$PDIR" "$OUT/smoke" "$OUT/cbuild"
utc() { date -u +%FT%TZ; }

CBIN2=$OUT/cbuild/vocr_textspot2_test
if [ ! -x "$CBIN2" ]; then
  ( cd "$REPO/runtime" && ${CC:-cc} -std=c99 -O2 -Wall -Wextra -Werror vocr_lines.c vocr_lines_rows.c \
      vocr_textspot2.c vocr_textspot2_test.c -o "$CBIN2" )
fi
CBIN3=$OUT/cbuild/vocr_textspot3_test
if [ ! -x "$CBIN3" ] && [ -e "$REPO/runtime/vocr_textspot3.c" ] && [ -e "$REPO/runtime/vocr_textspot3_test.c" ]; then
  ( cd "$REPO/runtime" && ${CC:-cc} -std=c99 -O2 -Wall -Wextra -Werror vocr_lines.c vocr_lines_rows.c \
      vocr_textspot3.c vocr_textspot3_test.c -o "$CBIN3" ) || rm -f "$CBIN3"
fi
PLAN=$OUT/probe_plan${PLAN_TAG:+_$PLAN_TAG}.jsonl
"$PY" "$HERE/probe_summary.py" plan --out "$PLAN" --teacher "$TEACHER" $PROBES $SMOKES $KD > /dev/null

job() {            # job probe|smoke RECIPE SHAPE
  local kind=$1 recipe=$2 shape=$3 name dir rc extra
  name=${recipe}_${shape}_seed1
  dir=$OUT/$kind/$name; [ "$kind" = probe ] && dir=$OUT/$PDIR/$name
  if [ -e "$dir/probe.json" ] || [ -e "$dir/done.json" ]; then return 0; fi
  if [ -e "$dir" ]; then mv "$dir" "$dir.failed.$(date -u +%H%M%S)"; fi
  local -a ARGV=()
  mapfile -d '' -t ARGV < <("$PY" "$HERE/probe_summary.py" argv "$PLAN" "$name" --out-dir "$(dirname "$dir")")
  if [ "$kind" = probe ]; then extra=(--probe 200 --ema-start 1); else extra=(--steps 300 --log-every 20 --ema-start 100); fi
  echo "$(utc) $kind $name start (load $(cut -d' ' -f1-3 /proc/loadavg)): ${ARGV[*]} ${extra[*]}" >> "$OUT/probe.log"
  rc=0
  ( cd "$REPO" && HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout 5400 .venv/bin/python "${ARGV[@]}" "${extra[@]}" \
      < /dev/null > "$OUT/logs/$([ "$kind" = probe ] && echo "$PDIR" || echo "$kind")-$name.log" 2>&1 ) || rc=$?
  echo "$(utc) $kind $name exit $rc (load $(cut -d' ' -f1-3 /proc/loadavg))" >> "$OUT/probe.log"
  if [ "$kind" = smoke ] && [ -e "$dir/done.json" ]; then
    local -a C=(--cbin2 "$CBIN2"); [ -x "$CBIN3" ] && C+=(--cbin3 "$CBIN3")
    ( cd "$REPO" && "$PY" "$HERE/check_run.py" "$dir" "${C[@]}" ) >> "$OUT/probe.log" 2>&1 \
      || echo "$(utc) smoke $name: CHECK FAILED" >> "$OUT/probe.log"
  fi
}

launch() {         # launch KIND LIST: every recipe:shape of LIST, at most MAX_PAR jobs of this script at once
  local kind=$1 s
  for s in $2; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
    job "$kind" "${s%%:*}" "${s#*:}" &
    sleep 2
  done
}

echo "$(utc) probe start: MAX_PAR $MAX_PAR, pass $PASS, head $(cd "$REPO" && git rev-parse HEAD), teacher $TEACHER" >> "$OUT/probe.log"
if [ "$PASS" -gt 1 ]; then
  launch probe "$PROBES"
else
  # the base smoke of KD_SHAPE first (kd's stand-in teacher), the probes beside it, then the other smokes, then kd
  launch smoke "base:$KD_SHAPE"
  launch probe "$PROBES"
  launch smoke "$(echo "$SMOKES" | tr ' ' '\n' | { grep -vx "base:$KD_SHAPE" || true; } | tr '\n' ' ')"
  while [ ! -e "$TEACHER/done.json" ] && [ -n "$(jobs -rp)" ]; do sleep 5; done
  if [ -e "$TEACHER/done.json" ]; then
    launch probe "$KD"
    launch smoke "$KD"
  else
    echo "$(utc) no teacher at $TEACHER: kd not probed" >> "$OUT/probe.log"
  fi
fi
wait
echo "$(utc) PROBE_DONE" >> "$OUT/probe.log"
echo PROBE_DONE

#!/usr/bin/env bash
# The textspot family B sweep (eval/textspot_protocol.json "candidates.family_b": every scale x shape, seeds 1 and
# 2; 22 runs): each run through heavy.sh, 5 threads, at most MAX_PAR (5) runs at once, longest first.
#   sweep.sh OUT [RUN ...]          RUN: <scale>_<shape>_seed<n> (default: all 22)
# Every run trains to its last step; nothing is selected or stopped on any score. A run that exits without
# done.json (a crash or the 4 h timeout) is moved to OUT/<run>.failed<k> and restarted from scratch with the same
# seed, at most 3 attempts; OUT/crashes.jsonl and OUT/restarts.jsonl record it. A finished run is checked by
# check_run.py (final.vtsp loads in textspot_twin.Model, the twin runs on procedural frames) and appended to
# OUT/finished.txt. OUT/locks/<run>.lock is held by the run's processes, so a second sweep skips a live run.
# Data (the defaults, overridable): the full-frame synthetic shards (data/manifests/textspot_shards_v1.json) and
# #97's real canvases (data/manifests/textdet_shards_v1.json, its synth shards not read).
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:?usage: sweep.sh OUT [RUN ...]}; shift || true
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
MAX_PAR=${MAX_PAR:-5}
TS=$HERD/shared/artifacts/textspot; TD=$HERD/shared/artifacts/textdet; PD=$HERD/shared/private/textdet
MANIFESTS=${MANIFESTS:-"data/manifests/textspot_shards_v1.json data/manifests/textdet_shards_v1.json"}
SHARDS=${SHARDS:-"$TS/shards/synth_frames $TD/shards/realplus $TD/shards/teacher $TD/shards/devnative $PD/shards/owner"}
mkdir -p "$OUT/logs" "$OUT/locks"
utc() { date -u +%FT%TZ; }

default_runs() {   # the protocol's 22 runs, longest first: a step costs the shape's MACs on one 256 x 512 window
  (cd "$REPO" && .venv-ci/bin/python -c '
import json, sys
sys.path.insert(0, "train")
import textdet_model as TM
fb = json.load(open("eval/textspot_protocol.json"))["candidates"]["family_b"]
runs = [(sc, sh, n) for sc, shapes in fb["shapes"].items() for sh in shapes for n in fb["seeds"]]
runs.sort(key=lambda r: (-TM.cost(TM.SHAPES[r[1]], 512, 256)["macs"], r[0], r[1], r[2]))
print(" ".join(f"{sc}_{sh}_seed{n}" for sc, sh, n in runs))')
}

one() {            # one run: up to 3 attempts, each from scratch, then the check
  local run=$1 scale rest shape seed attempt rc k
  scale=${run%%_*}; rest=${run#*_}; shape=${rest%_seed*}; seed=${run##*_seed}
  exec 7>"$OUT/locks/$run.lock"
  if ! flock -n 7; then echo "$(utc) $run: live in another sweep, skipped" >> "$OUT/sweep.log"; return 0; fi
  for attempt in 1 2 3; do
    [ -e "$OUT/$run/done.json" ] && break
    if [ -e "$OUT/$run" ]; then
      k=1; while [ -e "$OUT/$run.failed$k" ]; do k=$((k + 1)); done
      mv "$OUT/$run" "$OUT/$run.failed$k"
      if [ -e "$OUT/logs/$run.log" ]; then mv "$OUT/logs/$run.log" "$OUT/logs/$run.failed$k.log"; fi
      echo "{\"run\": \"$run\", \"moved_to\": \"$run.failed$k\", \"restart\": \"from scratch, seed $seed\", \"utc\": \"$(utc)\"}" >> "$OUT/restarts.jsonl"
    fi
    echo "$(utc) $run: attempt $attempt" >> "$OUT/sweep.log"
    rc=0
    ( cd "$REPO" && HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout 14400 .venv/bin/python train/textspot_train.py \
        --shape "$shape" --scale "$scale" --seed "$seed" --manifest $MANIFESTS --shards $SHARDS \
        --out "$OUT/$run" --threads 5 < /dev/null > "$OUT/logs/$run.log" 2>&1 ) || rc=$?
    echo "$(utc) $run: attempt $attempt exit $rc" >> "$OUT/sweep.log"
    [ -e "$OUT/$run/done.json" ] && break
    echo "{\"run\": \"$run\", \"attempt\": $attempt, \"exit\": $rc, \"utc\": \"$(utc)\"}" >> "$OUT/crashes.jsonl"
  done
  if [ -e "$OUT/$run/done.json" ]; then
    ( cd "$REPO" && .venv-ci/bin/python eval/reproducibility/textspot-20261019/check_run.py "$OUT/$run" \
        --finished "$OUT/finished.txt" ) >> "$OUT/sweep.log" 2>&1 || echo "$(utc) $run: CHECK FAILED" >> "$OUT/sweep.log"
  else
    echo "$(utc) $run: NOT FINISHED after 3 attempts" >> "$OUT/sweep.log"
  fi
}

if [ $# -gt 0 ]; then RUNS="$*"; else RUNS=$(default_runs); fi
echo "$(utc) sweep start: MAX_PAR $MAX_PAR, head $(cd "$REPO" && git rev-parse HEAD), runs $RUNS" >> "$OUT/sweep.log"
for run in $RUNS; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 15; done
  one "$run" &
  sleep 2
done
wait
echo "$(utc) SWEEP_DONE" >> "$OUT/sweep.log"
echo SWEEP_DONE

#!/usr/bin/env bash
# Round 3 (protocol amendment 4): its 12 runs through heavy.sh, 5 threads, --strict-kernel.
#   sweep_r3.sh OUT
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:?usage: sweep_r3.sh OUT}
T=$HERD/shared/artifacts/textdet; P=$HERD/shared/private/textdet
SHARDS="$T/shards/synth $T/shards/realplus $T/shards/teacher $T/shards/devnative $P/shards/owner"
RUNS=$(cd "$REPO" && .venv-ci/bin/python -c 'import json; print(" ".join(next(a for a in json.load(open("eval/textdet_protocol.json"))["amendments"] if a["n"] == 4)["runs"]))')
mkdir -p "$OUT/logs"
for run in $RUNS; do
  shape=${run%_seed*}; seed=${run##*_seed}
  [ -e "$OUT/$run/done.json" ] && continue
  ( cd "$REPO" && HEAVY_SLOTS=12 "$HERD/heavy.sh" timeout 14400 .venv/bin/python train/textdet_train.py --shape "$shape" \
      --seed "$seed" --manifest data/manifests/textdet_shards_v1.json --shards $SHARDS --out "$OUT/$run" --threads 5 \
      --strict-kernel < /dev/null > "$OUT/logs/$run.log" 2>&1 ) &
  sleep 2
done
wait
echo SWEEP_R3_DONE

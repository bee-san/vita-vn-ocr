#!/usr/bin/env bash
# The textgate sweep (eval/textgate_protocol.json "candidates.new_presence_runs"): 8 presence-only runs, each through
# heavy.sh with 5 threads, all at once. Usage: sweep.sh OUT_DIR
set -u
OUT=$(realpath -m "$1"); mkdir -p "$OUT/logs"
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
H=${HERD:-/local/home/skerraut/herd-vita-ocr}
T=$H/shared/artifacts/textdet/shards
P=$H/shared/private/textdet/shards
S=$H/shared/artifacts/textgate/shards/synth_clean
cd "$ROOT" || exit 2
for run in p8_c4:1 p8_c6:1 p8_c6:2 p8_c8:1 p8_c8:2 p8_c12:1 p4_c4:1 p4_c8:1; do
  shape=${run%%:*}; seed=${run##*:}; name=${shape}_seed$seed
  [ -e "$OUT/$name/done.json" ] && continue
  HEAVY_SLOTS=12 setsid "$H/heavy.sh" timeout 14400 .venv/bin/python train/textgate_train.py --shape "$shape" \
    --seed "$seed" --real-manifest data/manifests/textdet_shards_v1.json \
    --synth-manifest data/manifests/textgate_shards_v1.json --shards "$T/realplus" "$T/teacher" "$T/devnative" \
    "$P/owner" "$S" --out "$OUT/$name" --steps 24000 --threads 5 < /dev/null > "$OUT/logs/$name.log" 2>&1 &
  sleep 2
done
wait
echo SWEEP_DONE

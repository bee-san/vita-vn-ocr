#!/usr/bin/env bash
# A/B of the pretraining data: the v1 architecture trained for the same steps from the same
# seed on (synth) the v1 synthetic shards, (nv) NVIDIA line crops, or (mix) both, then scored
# by the eval harness on the real test sets. Run each arm through heavy.sh:
#
#   $HERD/heavy.sh nvsynth/ab.sh ARM SEED          ARM: synth | nv | mix
#
# Environment (defaults in brackets):
#   TRAIN_PY   a train/train.py with --steps/--seed/--out    [train/train.py]
#   EVAL_PY    eval/run.py                                     [eval/run.py]
#   PY         python                                          [$HERD/shared/venv/bin/python]
#   AB_DIR     runs, shard lists and results                   [$HERD/shared/artifacts/nvsynth/ab]
#   NV_DIR     nvsynth shards (fetch.py run + merge-val)       [$HERD/shared/artifacts/nvsynth/shards]
#   SYNTH_DIR  v1 synthetic shards synth_00000..00039          [$HERD/shared/data/synth]
#   NV_SHARDS  NVIDIA train shards per arm (about 20k lines each)              [20]
#   STEPS      [5000]   EVAL_SETS  [vita-test,synth-val]
#   VOCR_EVAL_DATA (eval screenshots) [$HERD/shared/artifacts/eval]
#
# train.py validates on the last shard of its sorted --synth glob: synth_00039 for synth and
# mix, nv_val for nv. That only picks ckpt_best; the comparison uses ckpt_last (step STEPS),
# scored on EVAL_SETS (results/<arm>_s<seed>.json) and on nv_val.npz (results/nv-val/).
set -euo pipefail
arm=${1:?arm: synth|nv|mix}
seed=${2:?seed}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
PY=${PY:-$HERD/shared/venv/bin/python}
TRAIN_PY=${TRAIN_PY:-$ROOT/train/train.py}
EVAL_PY=${EVAL_PY:-$ROOT/eval/run.py}
AB_DIR=${AB_DIR:-$HERD/shared/artifacts/nvsynth/ab}
NV_DIR=${NV_DIR:-$HERD/shared/artifacts/nvsynth/shards}
SYNTH_DIR=${SYNTH_DIR:-$HERD/shared/data/synth}
NV_SHARDS=${NV_SHARDS:-20}
STEPS=${STEPS:-5000}
EVAL_SETS=${EVAL_SETS:-vita-test,synth-val}
export VOCR_EVAL_DATA=${VOCR_EVAL_DATA:-$HERD/shared/artifacts/eval}
export VOCR_SYNTH_VAL=${VOCR_SYNTH_VAL:-$SYNTH_DIR/synth_00039.npz}

set_dir=$AB_DIR/sets/$arm
mkdir -p "$set_dir" "$AB_DIR/runs" "$AB_DIR/results"
rm -f "$set_dir"/*.npz
link_synth() { for f in "$SYNTH_DIR"/synth_000[0-3][0-9].npz; do ln -s "$f" "$set_dir/"; done; }
link_nv() {
  for i in $(seq 0 $((NV_SHARDS - 1))); do
    f=$(printf '%s/nv_train_%03d.npz' "$NV_DIR" "$i")
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
    ln -s "$f" "$set_dir/"
  done
}
case $arm in
  synth) link_synth ;;
  nv) link_nv; ln -s "$NV_DIR/nv_val.npz" "$set_dir/" ;;
  mix) link_synth; link_nv ;;             # sorted: nv_train_* < synth_*, so synth_00039 validates
  *) echo "unknown arm $arm" >&2; exit 1 ;;
esac
echo "arm $arm: $(ls "$set_dir" | wc -l) shards, validation $(ls "$set_dir" | sort | tail -n 1)"

run=${arm}_s$seed
"$PY" "$TRAIN_PY" --run "$run" --out "$AB_DIR/runs" --synth "$set_dir/*.npz" --steps "$STEPS" \
  --seed "$seed" --eval-every 2500 > "$AB_DIR/runs/$run.train.log" 2>&1 < /dev/null
grep -q TRAIN_DONE "$AB_DIR/runs/$run.train.log"
"$PY" "$EVAL_PY" --model "$AB_DIR/runs/$run/ckpt_last.pt" --set "$EVAL_SETS" \
  --out "$AB_DIR/results/$run.json" > "$AB_DIR/results/$run.log" 2>&1 < /dev/null
# The NVIDIA held-out shard through the same harness (its synth-val slot; style tags meaningless).
mkdir -p "$AB_DIR/results/nv-val"
VOCR_SYNTH_VAL=$NV_DIR/nv_val.npz "$PY" "$EVAL_PY" --model "$AB_DIR/runs/$run/ckpt_last.pt" --set synth-val \
  --out "$AB_DIR/results/nv-val/$run.json" > "$AB_DIR/results/nv-val/$run.log" 2>&1 < /dev/null
tail -n 30 "$AB_DIR/results/$run.log"
grep -h "synth-val:" "$AB_DIR/results/nv-val/$run.log" | sed 's/^synth-val/nv-val/'
echo "AB_DONE $run"

#!/usr/bin/env bash
# textspot8 steps 1-2 (docs/textspot8.md, eval/textspot8_protocol.json): MeikiOCR's detector as the reference
# spotter through textspot6's pinned page path, and the crop-geometry diagnostic on hval. One heavy slot (5 threads):
#   HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 eval/reproducibility/textspot8-20261027/bench_meiki.sh [STEP...]
# STEP: spot decode pp tsdev boxes speed geom (default: all but geom, in that order). Re-running a step overwrites its outputs.
# Env: OUT (default $A/vocr6-spotter2/eval), PAGEPATH (textspot6's pin), RUBI (textspot6's textnorm pin), WORKERS (5).
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$REPO/.venv/bin/python
TPY=$REPO/teacher/.venv/bin/python
OUT=${OUT:-$A/vocr6-spotter2/eval}
PAGEPATH=${PAGEPATH:-$A/vocr6-spotter/pagepath_snap135/repo/eval/pagepath.py}
RUBI=${RUBI:-$A/vocr6-spotter/rubi_textnorm}
WORKERS=${WORKERS:-5}
UNITS_HVAL=$A/vocr6-spotter/eval/units/hval_v1.jsonl
UNITS_DEV=$A/vocr5-analysis/work/units.pkl
MODELS=$A/vvo-public/v030
LM=$MODELS/vocr_lm_o3_1000k.vlm
CACHE=$OUT/pp_cache/reads.sqlite
WORK=$OUT/pp_work
PAGES=$A/vocr6-ingest/pages
T6=$A/vocr6-spotter/eval
M=$OUT/meiki
mkdir -p "$M" "$OUT/geom" "$(dirname "$CACHE")" "$WORK"
STEPS=${*:-"spot decode pp tsdev boxes speed"}
export OMP_NUM_THREADS=1

pp() {   # pp SET UNITS OUTBASE ARGS...: one eval/pagepath.py run (textspot6's score_run.sh pp())
  local set=$1 units=$2 base=$3; shift 3
  local pages=()
  [ "$set" = hval ] && pages=(--pages "$PAGES")
  "$PY" "$PAGEPATH" run --units "$units" "${pages[@]}" "$@" --models "$MODELS" --lm "$LM" --rubi "$RUBI" \
    --cache "$CACHE" --work "$WORK" --workers "$WORKERS" --rows "$base.rows.jsonl" --out "$base.json"
}

for step in $STEPS; do
  case $step in
    spot)
      for s in dev tsdev hval; do
        "$TPY" "$REPO/eval/textspot8.py" meiki-spot --set $s --out "$M/raw_$s.jsonl" --workers "$WORKERS"
      done;;
    decode)
      for d in raw pad ink; do
        for s in dev tsdev hval; do
          "$PY" "$REPO/eval/textspot8.py" meiki-decode --raw "$M/raw_$s.jsonl" --decode $d --out "$M/spot_${s}_$d.jsonl"
        done
      done;;
    pp)
      for d in raw pad ink; do
        for k in det detref; do
          name=$d; [ $k = detref ] && name=$d+refine
          "$PY" "$REPO/eval/textspot6.py" pagepath-boxes --spot "$M/spot_dev_$d.jsonl" --set dev --decode $k \
            --out "$M/boxes_dev_$name.jsonl"
          "$PY" "$REPO/eval/textspot6.py" pagepath-boxes --spot "$M/spot_hval_$d.jsonl" --set hval --decode $k \
            --out "$M/boxes_hval_$name.jsonl"
          pp dev "$UNITS_DEV" "$M/dev_$name" --boxes "$M/boxes_dev_$name.jsonl"
          pp hval "$UNITS_HVAL" "$M/hval_$name" --frame-boxes "$M/boxes_hval_$name.jsonl"
        done
      done;;
    tsdev)
      for d in raw pad ink; do
        for k in det detref; do
          "$PY" "$REPO/eval/textspot6.py" tsdev --spot "$M/spot_tsdev_$d.jsonl" --decode $k --out "$M/tsdev_${d}_$k.json" \
            --cache "$OUT/tsdev_cache" --cbuild "$OUT/cbuild" --workers "$WORKERS"
        done
      done;;
    boxes)
      for d in raw pad ink; do
        "$PY" "$REPO/eval/textspot6.py" hval-boxes --spot "$M/spot_hval_$d.jsonl" \
          --shards "$HERD/shared/private/vocr6-spotter/shards_v1" --manifest "$REPO/data/manifests/textspot6_pages_v1.json" \
          --out "$M/hval_boxes_$d.json"
      done;;
    speed)
      OMP_NUM_THREADS=5 "$TPY" "$REPO/eval/textspot8.py" meiki-speed --n 100 --threads 1,5 --out "$M/speed.json";;
    geom)
      G=$OUT/geom
      for a in "fix1 $T6/grad_refine_fix1/hval.rows.jsonl 0 1" "fix1 $T6/grad_refine_fix1/hval.rows.jsonl 0 2" \
               "fix1 $T6/grad_refine_fix1/hval.rows.jsonl 4 0" \
               "t64det $T6/base_t64n8c16_seed1/hval_det.rows.jsonl 0 -1" \
               "t64det $T6/base_t64n8c16_seed1/hval_det.rows.jsonl 0 -2" \
               "qS4crop $T6/base_qS4_seed2/hval_crop.rows.jsonl 0 -1" \
               "qS4crop $T6/base_qS4_seed2/hval_crop.rows.jsonl 0 -2"; do
        set -- $a
        n="$1_dx$3_dy$4"
        "$PY" "$REPO/eval/textspot8.py" shift-boxes --rows "$2" --dx "$3" --dy "$4" --out "$G/boxes_hval_$n.jsonl"
        pp hval "$UNITS_HVAL" "$G/hval_$n" --boxes "$G/boxes_hval_$n.jsonl"
      done;;
    *) echo "unknown step $step" >&2; exit 2;;
  esac
done
echo "bench_meiki finished: $STEPS"

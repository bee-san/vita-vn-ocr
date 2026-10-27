#!/usr/bin/env bash
# textspot8 amendment 4: the context decode (eval/textspot8_ctx.py: the spotter on the region grown by 48 px, the boxes
# whose centre lies in the region) of a t8 run, through the pinned page path, and its C == twin check. One heavy slot:
#   score_ctx8.sh RUN_DIR NAME      (outputs in $A/vocr6-spotter2/eval/NAME)
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$REPO/.venv/bin/python
PAGEPATH=${PAGEPATH:-$A/vocr6-spotter/pagepath_snap135/repo/eval/pagepath.py}
RUBI=${RUBI:-$A/vocr6-spotter/rubi_textnorm}
OUT=${OUT:-$A/vocr6-spotter2/eval}
WORKERS=${WORKERS:-5}
UNITS_HVAL=$A/vocr6-spotter/eval/units/hval_v1.jsonl
UNITS_DEV=$A/vocr5-analysis/work/units.pkl
MODELS=$A/vvo-public/v030
LM=$MODELS/vocr_lm_o3_1000k.vlm
RUN=${1:?run dir}; NAME=${2:?name}; D=$OUT/$NAME; BLOB=$RUN/final.vtsp
mkdir -p "$D"
pp() {
  local set=$1 units=$2 base=$3; shift 3
  local pages=()
  [ "$set" = hval ] && pages=(--pages "$A/vocr6-ingest/pages")
  "$PY" "$PAGEPATH" run --units "$units" "${pages[@]}" "$@" --models "$MODELS" --lm "$LM" --rubi "$RUBI" \
    --cache "$OUT/pp_cache/reads.sqlite" --work "$OUT/pp_work" --workers "$WORKERS" --rows "$base.rows.jsonl" \
    --out "$base.json"
}
for s in dev hval; do
  "$PY" "$REPO/eval/textspot8_ctx.py" boxes --blob "$BLOB" --set $s --out "$D/boxes_${s}_ctx.jsonl" --workers "$WORKERS"
done
pp dev "$UNITS_DEV" "$D/dev_ctx" --boxes "$D/boxes_dev_ctx.jsonl"
pp hval "$UNITS_HVAL" "$D/hval_ctx" --boxes "$D/boxes_hval_ctx.jsonl"
for s in dev hval; do
  "$PY" "$REPO/eval/textspot8_ctx.py" parity --blob "$BLOB" --set $s --out "$D/parity_ctx_$s.json" --cbin "$OUT/cbuild6" \
    --workers "$WORKERS"
done
date -u +%FT%TZ > "$D/SCORED_CTX"
echo "score_ctx8 finished $NAME"

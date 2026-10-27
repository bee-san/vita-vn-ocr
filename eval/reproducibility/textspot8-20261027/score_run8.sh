#!/usr/bin/env bash
# textspot8's scoring of one run: textspot6's score_run.sh (eval/reproducibility/textspot6-20261026/score_run.sh)
# with textspot8's own output, reading cache and build directories, and textspot6's pins (page path, textnorm,
# recognizers, LM, units), so that every number pairs with textspot6's. Run it through heavy.sh (5 threads):
#   score_run8.sh RUN_DIR|BLOB.vtsp NAME [DECODES]      DECODES default "crop crop+refine det det+refine"
# Env: OUT (default $A/vocr6-spotter2/eval), PAGEPATH, RUBI, WORKERS (5).
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
CACHE=$OUT/pp_cache/reads.sqlite
WORK=$OUT/pp_work
PAGES=$A/vocr6-ingest/pages
mkdir -p "$OUT" "$(dirname "$CACHE")" "$WORK"

pp() {   # pp SET UNITS OUTBASE ARGS...: one eval/pagepath.py run
  local set=$1 units=$2 base=$3; shift 3
  local pages=()
  [ "$set" = hval ] && pages=(--pages "$PAGES")
  "$PY" "$PAGEPATH" run --units "$units" "${pages[@]}" "$@" --models "$MODELS" --lm "$LM" --rubi "$RUBI" \
    --cache "$CACHE" --work "$WORK" --workers "$WORKERS" --rows "$base.rows.jsonl" --out "$base.json"
}

SRC=${1:?usage: score_run8.sh RUN_DIR|BLOB.vtsp NAME [DECODES]}; NAME=${2:?name}
DECODES=${3:-"crop crop+refine det det+refine"}
if [ -d "$SRC" ]; then BLOB=$SRC/final.vtsp; else BLOB=$SRC; fi
D=$OUT/$NAME; mkdir -p "$D"
sha256sum "$BLOB" > "$D/blob.sha256"
"$PY" "$REPO/eval/textspot6.py" spot --blob "$BLOB" --set dev --out "$D/spot_dev.jsonl" --workers "$WORKERS"
"$PY" "$REPO/eval/textspot6.py" spot --blob "$BLOB" --set hval --out "$D/spot_hval.jsonl" --workers "$WORKERS"
for dec in $DECODES; do
  case $dec in
    crop|crop+refine)
      k=det; [ "$dec" = crop+refine ] && k=detref
      "$PY" "$REPO/eval/textspot6.py" spot-crop --blob "$BLOB" --set dev --decode $k --out "$D/boxes_dev_$dec.jsonl" \
        --workers "$WORKERS"
      "$PY" "$REPO/eval/textspot6.py" spot-crop --blob "$BLOB" --set hval --units "$UNITS_HVAL" --decode $k \
        --out "$D/boxes_hval_$dec.jsonl" --workers "$WORKERS"
      pp dev "$UNITS_DEV" "$D/dev_$dec" --boxes "$D/boxes_dev_$dec.jsonl"
      pp hval "$UNITS_HVAL" "$D/hval_$dec" --boxes "$D/boxes_hval_$dec.jsonl";;
    det|det+refine)
      k=det; [ "$dec" = det+refine ] && k=detref
      "$PY" "$REPO/eval/textspot6.py" pagepath-boxes --spot "$D/spot_dev.jsonl" --set dev --decode $k \
        --out "$D/boxes_dev_$dec.jsonl"
      "$PY" "$REPO/eval/textspot6.py" pagepath-boxes --spot "$D/spot_hval.jsonl" --set hval --decode $k \
        --out "$D/boxes_hval_$dec.jsonl"
      pp dev "$UNITS_DEV" "$D/dev_$dec" --boxes "$D/boxes_dev_$dec.jsonl"
      pp hval "$UNITS_HVAL" "$D/hval_$dec" --frame-boxes "$D/boxes_hval_$dec.jsonl";;
    *) echo "unknown decode $dec" >&2; exit 2;;
  esac
done
"$PY" "$REPO/eval/textspot6.py" spot --blob "$BLOB" --set tsdev --out "$D/spot_tsdev.jsonl" --workers "$WORKERS"
for k in det detref; do
  "$PY" "$REPO/eval/textspot6.py" tsdev --spot "$D/spot_tsdev.jsonl" --decode $k --out "$D/tsdev_$k.json" \
    --cache "$OUT/tsdev_cache" --cbuild "$OUT/cbuild" --workers "$WORKERS"
done
"$PY" "$REPO/eval/textspot6.py" hval-boxes --spot "$D/spot_hval.jsonl" \
  --shards "$HERD/shared/private/vocr6-spotter/shards_v1" --manifest "$REPO/data/manifests/textspot6_pages_v1.json" \
  --out "$D/hval_boxes.json"
"$PY" "$REPO/eval/textspot6_parity.py" --blob "$BLOB" --out "$D/parity.json" --sets dev,dev-crop,tsdev \
  --cbin "$OUT/cbuild6" --workers "$WORKERS"
date -u +%FT%TZ > "$D/SCORED"
echo "score_run8 finished $NAME $D"

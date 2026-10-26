#!/usr/bin/env bash
# textspot6's scoring of one run (docs/textspot6.md "Metrics", amendments 2-3): the spotter's boxes on the unseen dev
# units and the hval regions in the four decodes, then eval/pagepath.py (vocr6-finder's page-path benchmark, #129) on
# each, and the report-only whole-frame metrics. Run it through heavy.sh (5 threads).
#   score_run.sh RUN_DIR|BLOB.vtsp NAME [DECODES]      DECODES default "crop crop+refine det det+refine"
#   score_run.sh --baseline [NAME SPEC]                 a finder on dev and hval: default grad_refine = --finder rubi
#                                                       (Rubi today); e.g. grad_refine_fix5 "rubi,fix=5" (#135)
# Env: PAGEPATH (the pinned eval/pagepath.py; its directory's pagepath_find.c is built by it), OUT (default
# $A/vocr6-spotter/eval), WORKERS (5). Results: OUT/<NAME>/{dev,hval}_<decode>.json (+ .rows.jsonl), tsdev_*.json.
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$REPO/.venv/bin/python
PAGEPATH=${PAGEPATH:?set PAGEPATH to the pinned eval/pagepath.py}
OUT=${OUT:-$A/vocr6-spotter/eval}
WORKERS=${WORKERS:-5}
UNITS_HVAL=$A/vocr6-spotter/eval/units/hval_v1.jsonl
UNITS_DEV=$A/vocr5-analysis/work/units.pkl
MODELS=$A/vvo-public/v030
LM=$MODELS/vocr_lm_o3_1000k.vlm
RUBI=${RUBI:-/home/skerraut/herd-synth3/clones/vocr6-spotter-rubi}
CACHE=$A/vocr6-spotter/eval/pp_cache/reads.sqlite
WORK=$A/vocr6-spotter/eval/pp_work
PAGES=$A/vocr6-ingest/pages
mkdir -p "$OUT" "$(dirname "$CACHE")" "$WORK"

pp() {   # pp SET UNITS OUTBASE ARGS...: one eval/pagepath.py run
  local set=$1 units=$2 base=$3; shift 3
  local pages=()
  [ "$set" = hval ] && pages=(--pages "$PAGES")
  "$PY" "$PAGEPATH" run --units "$units" "${pages[@]}" "$@" --models "$MODELS" --lm "$LM" --rubi "$RUBI" \
    --cache "$CACHE" --work "$WORK" --workers "$WORKERS" --rows "$base.rows.jsonl" --out "$base.json"
}

if [ "${1:-}" = --baseline ]; then
  BNAME=${2:-grad_refine}; SPEC=${3:-rubi}
  D=$OUT/$BNAME; mkdir -p "$D"
  pp dev "$UNITS_DEV" "$D/dev" --finder "$SPEC"
  pp hval "$UNITS_HVAL" "$D/hval" --finder "$SPEC"
  echo "score_baseline finished $D"; exit 0
fi

SRC=${1:?usage: score_run.sh RUN_DIR|BLOB.vtsp NAME [DECODES]}; NAME=${2:?name}
DECODES=${3:-"crop crop+refine det det+refine"}
if [ -d "$SRC" ]; then BLOB=$SRC/final.vtsp; else BLOB=$SRC; fi
D=$OUT/$NAME; mkdir -p "$D"
sha256sum "$BLOB" > "$D/blob.sha256"
# 1. whole-frame boxes (det, det+refine), once per set
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
# 2. report only: textspot's whole-frame metrics (both whole-frame decodes) and hval box metrics
"$PY" "$REPO/eval/textspot6.py" spot --blob "$BLOB" --set tsdev --out "$D/spot_tsdev.jsonl" --workers "$WORKERS"
for k in det detref; do
  "$PY" "$REPO/eval/textspot6.py" tsdev --spot "$D/spot_tsdev.jsonl" --decode $k --out "$D/tsdev_$k.json" \
    --workers "$WORKERS"
done
"$PY" "$REPO/eval/textspot6.py" hval-boxes --spot "$D/spot_hval.jsonl" \
  --shards "$HERD/shared/private/vocr6-spotter/shards_v1" --manifest "$REPO/data/manifests/textspot6_pages_v1.json" \
  --out "$D/hval_boxes.json"
echo "score_run finished $NAME $D"

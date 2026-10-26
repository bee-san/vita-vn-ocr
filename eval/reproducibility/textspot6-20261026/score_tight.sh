#!/usr/bin/env bash
# textspot6's tight-region rows (amendment 12, report only): the pick and the baselines on vocr6-finder's
# tight-region units, the regions a player draws tight to the text: hval_tight (0 px around the ink), hval_tight2
# (2 px) and dev_tight (1 px), all arms read on the same units through eval/pagepath.py. Run through heavy.sh.
#   score_tight.sh --baseline NAME SPEC          a finder: e.g. grad_refine rubi, grad_refine_fix1 "rubi,fix=1"
#   score_tight.sh RUN_DIR|BLOB NAME DECODES     a spotter in its decodes; the det decodes reuse the scoring's
#                                                spot files (OUT/NAME/spot_{dev,hval}.jsonl)
# Env: PAGEPATH (the pinned eval/pagepath.py), RUBI (the pinned textnorm), OUT (default $A/vocr6-spotter/eval),
# TIGHT (default $A/vocr6-finder/units_v1). Results: OUT/tight/NAME/{dev_tight,hval_tight,hval_tight2}_<decode>.json
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$REPO/.venv/bin/python
PAGEPATH=${PAGEPATH:?set PAGEPATH to the pinned eval/pagepath.py}
RUBI=${RUBI:?set RUBI to the pinned textnorm copy}
OUT=${OUT:-$A/vocr6-spotter/eval}
TIGHT=${TIGHT:-$A/vocr6-finder/units_v1}
WORKERS=${WORKERS:-5}
MODELS=$A/vvo-public/v030
LM=$MODELS/vocr_lm_o3_1000k.vlm
CACHE=$A/vocr6-spotter/eval/pp_cache/reads.sqlite
WORK=$A/vocr6-spotter/eval/pp_work
PAGES=$A/vocr6-ingest/pages
SETS="dev_tight:$TIGHT/dev_tight.pkl hval_tight:$TIGHT/hval_tight.jsonl hval_tight2:$TIGHT/hval_tight2.jsonl"

pp() {   # pp SETNAME UNITS OUTBASE ARGS...
  local set=$1 units=$2 base=$3; shift 3
  local pages=()
  [ "${set#hval}" != "$set" ] && pages=(--pages "$PAGES")
  "$PY" "$PAGEPATH" run --units "$units" "${pages[@]}" "$@" --models "$MODELS" --lm "$LM" --rubi "$RUBI" \
    --cache "$CACHE" --work "$WORK" --workers "$WORKERS" --rows "$base.rows.jsonl" --out "$base.json"
}

if [ "${1:-}" = --baseline ]; then
  NAME=${2:?name}; SPEC=${3:?finder spec}
  D=$OUT/tight/$NAME; mkdir -p "$D"
  for su in $SETS; do pp "${su%%:*}" "${su#*:}" "$D/${su%%:*}" --finder "$SPEC"; done
  echo "score_tight finished $NAME"; exit 0
fi

SRC=${1:?usage: score_tight.sh RUN_DIR|BLOB NAME DECODES}; NAME=${2:?name}; DECODES=${3:?decodes}
if [ -d "$SRC" ]; then BLOB=$SRC/final.vtsp; else BLOB=$SRC; fi
E=$OUT/$NAME
D=$OUT/tight/$NAME; mkdir -p "$D"
sha256sum "$BLOB" > "$D/blob.sha256"
for dec in $DECODES; do
  k=det; [ "${dec%+refine}" != "$dec" ] && k=detref
  for su in $SETS; do
    set=${su%%:*}; units=${su#*:}; b=$D/boxes_${set}_$dec.jsonl
    case $dec in
      crop|crop+refine)
        if [ "$set" = dev_tight ]; then s=dev; else s=hval; fi
        "$PY" "$REPO/eval/textspot6.py" spot-crop --blob "$BLOB" --set $s --units "$units" --decode $k --out "$b" \
          --workers "$WORKERS"
        pp "$set" "$units" "$D/${set}_$dec" --boxes "$b";;
      det|det+refine)
        if [ "$set" = dev_tight ]; then
          "$PY" "$REPO/eval/textspot6.py" pagepath-boxes --spot "$E/spot_dev.jsonl" --set dev --units "$units" \
            --decode $k --out "$b"
          pp "$set" "$units" "$D/${set}_$dec" --boxes "$b"
        else
          "$PY" "$REPO/eval/textspot6.py" pagepath-boxes --spot "$E/spot_hval.jsonl" --set hval --decode $k --out "$b"
          pp "$set" "$units" "$D/${set}_$dec" --frame-boxes "$b"
        fi;;
      *) echo "unknown decode $dec" >&2; exit 2;;
    esac
  done
done
echo "score_tight finished $NAME"

#!/usr/bin/env bash
# textspot8's animated-icon check (report only; eval/textspot8_spinner.py): a finder or spotter on the dev units with
# an icon drawn after the last dialogue line, both icon styles. One heavy slot (5 threads):
#   spinner_runs.sh finder NAME SPEC          e.g. grad_fix1 rubi,fix=1
#   spinner_runs.sh blob NAME BLOB [DECODES]  DECODES default "crop det"
# Env: SET (default $HERD/shared/private/vocr6-spotter2/spinner), OUT (default $A/vocr6-spotter2/eval/spinner).
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$REPO/.venv/bin/python
PAGEPATH=${PAGEPATH:-$A/vocr6-spotter/pagepath_snap135/repo/eval/pagepath.py}
RUBI=${RUBI:-$A/vocr6-spotter/rubi_textnorm}
SET=${SET:-$HERD/shared/private/vocr6-spotter2/spinner}
OUT=${OUT:-$A/vocr6-spotter2/eval/spinner}
EV=$A/vocr6-spotter2/eval
WORKERS=${WORKERS:-5}
MODELS=$A/vvo-public/v030
LM=$MODELS/vocr_lm_o3_1000k.vlm
pp() {   # pp UNITS OUTBASE ARGS...
  local units=$1 base=$2; shift 2
  "$PY" "$PAGEPATH" run --units "$units" "$@" --models "$MODELS" --lm "$LM" --rubi "$RUBI" \
    --cache "$EV/pp_cache/reads.sqlite" --work "$EV/pp_work" --workers "$WORKERS" --rows "$base.rows.jsonl" \
    --out "$base.json"
}
KIND=${1:?finder|blob}; NAME=${2:?name}
for style in trained unseen; do
  S=$SET/$style; D=$OUT/$style/$NAME; mkdir -p "$D"
  if [ "$KIND" = finder ]; then
    pp "$S/units.pkl" "$D/dev" --finder "${3:?spec}"
    "$PY" "$REPO/eval/textspot8_spinner.py" overlap --rows "$D/dev.rows.jsonl" --icons "$S/icons.json" --out "$D/overlap.json"
  else
    BLOB=${3:?blob}; DECODES=${4:-"crop det"}
    for dec in $DECODES; do
      case $dec in
        det)
          "$PY" "$REPO/eval/textspot8_spinner.py" spot --blob "$BLOB" --dir "$S" --out "$D/spot.jsonl" --workers "$WORKERS"
          "$PY" "$REPO/eval/textspot6.py" pagepath-boxes --spot "$D/spot.jsonl" --set dev --decode det --units "$S/units.pkl" \
            --out "$D/boxes_det.jsonl";;
        crop)
          "$PY" "$REPO/eval/textspot6.py" spot-crop --blob "$BLOB" --set dev --units "$S/units.pkl" --decode det \
            --out "$D/boxes_crop.jsonl" --workers "$WORKERS";;
        *) echo "unknown decode $dec" >&2; exit 2;;
      esac
      pp "$S/units.pkl" "$D/dev_$dec" --boxes "$D/boxes_$dec.jsonl"
      "$PY" "$REPO/eval/textspot8_spinner.py" overlap --rows "$D/dev_$dec.rows.jsonl" --icons "$S/icons.json" \
        --out "$D/overlap_$dec.json"
    done
  fi
done
echo "spinner_runs finished $NAME"

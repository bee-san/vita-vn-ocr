#!/usr/bin/env bash
# textspot8 amendment 6: the VOCR6 recognizer guard. The pick's boxes (its stored --boxes files) and rubi,fix=1 (the
# pinned finder) read with VOCR6-N, cascade VOCR6-L (v0.4.0-vocr6.1's plain _w8 blobs), Rubi's VOCR6 LM settings
# (alpha 600, beta 2500, flags 1; dev: the shipped 1 MB game-text LM, hval: the screening build), through the pinned
# page path. One heavy slot.
#   guard_vocr6.sh NAME DECODE     (the pick's eval dir $A/vocr6-spotter2/eval/NAME, e.g. t8_t64n8c16_seed3 crop)
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
PV=$HERD/shared/private
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$REPO/.venv/bin/python
PAGEPATH=$A/vocr6-spotter/pagepath_snap135/repo/eval/pagepath.py
RUBI=$A/vocr6-spotter/rubi_textnorm
ASSETS=$A/vocr6-lead/release/assets
LMD=$A/vocr6-lm/release/v0.3.7-lm6.1/vocr_lm6_o3_1000k.vlm
LMH=$PV/vocr6-lm/lm/h64_1000k.vlm
EV=$A/vocr6-spotter2/eval
OUT=$EV/vocr6_guard
NAME=${1:?name}; DEC=${2:?decode}
M=$OUT/models
mkdir -p "$M" "$OUT"
# pagepath reads the tiers by VOCR5's file names: VOCR6's blobs under them (S is never read: tier N, cascade L)
ln -sfn "$ASSETS/VOCR6_N_w8.vocr" "$M/VOCR5_N_w8.vocr"
ln -sfn "$ASSETS/VOCR6_L_w8.vocr" "$M/VOCR5_L_w8.vocr"
ln -sfn "$A/vvo-public/v030/VOCR5_S_w8.vocr" "$M/VOCR5_S_w8.vocr"
( cd "$ASSETS" && sha256sum -c --quiet <(grep -E " VOCR6_(N|L)_w8.vocr$" SHA256SUMS) )
pp() {   # pp SET UNITS LM OUTBASE TIER ARGS...
  local set=$1 units=$2 lm=$3 base=$4 tier=$5; shift 5
  local pages=()
  [ "$set" = hval ] && pages=(--pages "$A/vocr6-ingest/pages")
  "$PY" "$PAGEPATH" run --units "$units" "${pages[@]}" "$@" --models "$M" --lm "$lm" --lm-alpha 600 --lm-beta 2500 \
    --lm-flags 1 --tier "$tier" --rubi "$RUBI" --cache "$EV/pp_cache/reads.sqlite" --work "$EV/pp_work" --workers 5 \
    --rows "$base.rows.jsonl" --out "$base.json"
}
UD=$A/vocr5-analysis/work/units.pkl
UH=$A/vocr6-spotter/eval/units/hval_v1.jsonl
B=$EV/$NAME
case $DEC in
  crop|crop+refine|ctx|det+D) BH=(--boxes "$B/boxes_hval_$DEC.jsonl");;
  det|det+refine) BH=(--frame-boxes "$B/boxes_hval_$DEC.jsonl");;
esac
for tier in N L; do
  pp dev "$UD" "$LMD" "$OUT/dev_${NAME}_${DEC}_$tier" $tier --boxes "$B/boxes_dev_$DEC.jsonl"
  pp hval "$UH" "$LMH" "$OUT/hval_${NAME}_${DEC}_$tier" $tier "${BH[@]}"
  [ -s "$OUT/dev_fix1_$tier.json" ] || pp dev "$UD" "$LMD" "$OUT/dev_fix1_$tier" $tier --finder rubi,fix=1
  [ -s "$OUT/hval_fix1_$tier.json" ] || pp hval "$UH" "$LMH" "$OUT/hval_fix1_$tier" $tier --finder rubi,fix=1
done
echo "guard_vocr6 finished $NAME $DEC"

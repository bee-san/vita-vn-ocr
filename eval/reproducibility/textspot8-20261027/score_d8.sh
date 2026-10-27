#!/usr/bin/env bash
# textspot8 amendment 3: vocr6-spotter3's D (join10, extend: its hval fit, feat/textspot7 a2fa039) on a scored t8 run's
# det boxes, as a candidate decode, and its F and F+D (report only: ineligible by its cost rule for a T trunk).
# Needs the run's score_run8.sh outputs (spot_dev/hval.jsonl, dev/hval_det rows). One heavy slot:
#   score_d8.sh NAME     (EVAL dir $A/vocr6-spotter2/eval/NAME)
# Env: T7 (the textspot7 worktree at a2fa039), OUT, PAGEPATH, RUBI, WORKERS.
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
T7=${T7:-/local/home/skerraut/herd-synth3/clones/vocr6-spotter2-wt7}
PY=$T7/.venv/bin/python
PAGEPATH=${PAGEPATH:-$A/vocr6-spotter/pagepath_snap135/repo/eval/pagepath.py}
RUBI=${RUBI:-$A/vocr6-spotter/rubi_textnorm}
OUT=${OUT:-$A/vocr6-spotter2/eval}
WORKERS=${WORKERS:-5}
UNITS_HVAL=$A/vocr6-spotter/eval/units/hval_v1.jsonl
UNITS_DEV=$A/vocr5-analysis/work/units.pkl
MODELS=$A/vvo-public/v030
LM=$MODELS/vocr_lm_o3_1000k.vlm
G=$A/vocr6-spotter/eval/grad_refine_fix1
NAME=${1:?name}; D=$OUT/$NAME
STEPS=join10,extend
[ "$(git -C "$T7" rev-parse --short HEAD)" = a2fa039 ] || { echo "T7 worktree is not at a2fa039" >&2; exit 2; }
pp() {
  local set=$1 units=$2 base=$3; shift 3
  local pages=()
  [ "$set" = hval ] && pages=(--pages "$A/vocr6-ingest/pages")
  "$PY" "$PAGEPATH" run --units "$units" "${pages[@]}" "$@" --models "$MODELS" --lm "$LM" --rubi "$RUBI" \
    --cache "$OUT/pp_cache/reads.sqlite" --work "$OUT/pp_work" --workers "$WORKERS" --rows "$base.rows.jsonl" \
    --out "$base.json"
}
# det+D: the candidate
for s in dev hval; do
  "$PY" "$T7/eval/textspot7.py" boxes-d --spot "$D/spot_$s.jsonl" --set $s --steps $STEPS --out "$D/boxes_${s}_det+D.jsonl" \
    --workers "$WORKERS"
done
pp dev "$UNITS_DEV" "$D/dev_det+D" --boxes "$D/boxes_dev_det+D.jsonl"
pp hval "$UNITS_HVAL" "$D/hval_det+D" --boxes "$D/boxes_hval_det+D.jsonl"
for s in dev hval; do
  "$PY" "$T7/eval/textspot7.py" parity --spot "$D/spot_$s.jsonl" --set $s --steps $STEPS --out "$D/parity_D_$s.json" \
    --work "$OUT/bin7" --workers "$WORKERS"
done
# D's cost (textspot7's rule cost_of_D: its A9 method on the dev units' boxes, the largest region of the five largest)
mkdir -p "$D/a9"
"$PY" "$T7/eval/textspot7.py" a9-batch --spot "$D/spot_dev.jsonl" --set dev --steps $STEPS --out "$D/a9/D.bin"
(cd "$T7" && "$PY" eval/textspot7_a9.py t7 --out "$D/a9/out" --batch "$D/a9/D.bin" --name D --units 387 \
  --one "$D/a9/D.bin.one1.bin" "$D/a9/D.bin.one2.bin" "$D/a9/D.bin.one3.bin" "$D/a9/D.bin.one4.bin" "$D/a9/D.bin.one5.bin")
# F and F+D: report only (textspot7's cost rule makes them ineligible for a T trunk)
"$PY" "$T7/eval/textspot7.py" fit-f --spot "$D/spot_hval.jsonl" --s-rows "$D/hval_det.rows.jsonl" --g-rows "$G/hval.rows.jsonl" \
  --set hval --out "$D/fit_F.json"
P=$("$PY" -c "import json;print(json.load(open('$D/fit_F.json'))['fit']['params'])")
"$PY" "$T7/eval/textspot7.py" boxes-f --spot "$D/spot_hval.jsonl" --s-rows "$D/hval_det.rows.jsonl" --g-rows "$G/hval.rows.jsonl" \
  --set hval --params "$P" --out "$D/boxes_hval_F.jsonl"
"$PY" "$T7/eval/textspot7.py" boxes-f --spot "$D/spot_dev.jsonl" --s-rows "$D/dev_det.rows.jsonl" --g-rows "$G/dev.rows.jsonl" \
  --set dev --params "$P" --out "$D/boxes_dev_F.jsonl"
pp dev "$UNITS_DEV" "$D/dev_F" --boxes "$D/boxes_dev_F.jsonl"
pp hval "$UNITS_HVAL" "$D/hval_F" --boxes "$D/boxes_hval_F.jsonl"
"$PY" "$T7/eval/textspot7.py" fit-f --spot "$D/spot_hval.jsonl" --s-rows "$D/hval_det+D.rows.jsonl" \
  --g-rows "$G/hval.rows.jsonl" --s-boxes "$D/boxes_hval_det+D.jsonl" --set hval --out "$D/fit_FD.json"
P=$("$PY" -c "import json;print(json.load(open('$D/fit_FD.json'))['fit']['params'])")
"$PY" "$T7/eval/textspot7.py" boxes-f --spot "$D/spot_hval.jsonl" --s-rows "$D/hval_det+D.rows.jsonl" \
  --g-rows "$G/hval.rows.jsonl" --s-boxes "$D/boxes_hval_det+D.jsonl" --set hval --params "$P" --out "$D/boxes_hval_F+D.jsonl"
"$PY" "$T7/eval/textspot7.py" boxes-f --spot "$D/spot_dev.jsonl" --s-rows "$D/dev_det+D.rows.jsonl" \
  --g-rows "$G/dev.rows.jsonl" --s-boxes "$D/boxes_dev_det+D.jsonl" --set dev --params "$P" --out "$D/boxes_dev_F+D.jsonl"
pp dev "$UNITS_DEV" "$D/dev_F+D" --boxes "$D/boxes_dev_F+D.jsonl"
pp hval "$UNITS_HVAL" "$D/hval_F+D" --boxes "$D/boxes_hval_F+D.jsonl"
date -u +%FT%TZ > "$D/SCORED_D"
echo "score_d8 finished $NAME"

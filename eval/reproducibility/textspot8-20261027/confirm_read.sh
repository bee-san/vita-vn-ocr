#!/usr/bin/env bash
# textspot8's confirmation read (eval/textspot8_protocol.json rule.guard_confirmation and amendment 5): every qualifier
# of eval/textspot8_dev_20261027.json, its representative seed, on vocr6-spotter3's frozen confirmation units, ONCE,
# through the pinned page path; paired later against vocr6-spotter3's stored rubi,fix=1 read of the same units.
# One heavy slot. An attempt marker refuses a second read.
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$REPO/.venv/bin/python
T7=${T7:-/local/home/skerraut/herd-synth3/clones/vocr6-spotter2-wt7}
PAGEPATH=$A/vocr6-spotter/pagepath_snap135/repo/eval/pagepath.py
RUBI=$A/vocr6-spotter/rubi_textnorm
MODELS=$A/vvo-public/v030
LM=$MODELS/vocr_lm_o3_1000k.vlm
UNITS=$HERD/shared/private/vocr6-spotter3/confirm/units_v1.pkl
OUT=$HERD/shared/private/vocr6-spotter2/confirm
EV=$A/vocr6-spotter2/eval
RUNS=$A/vocr6-spotter2/runs2
mkdir -p "$OUT"
[ -e "$OUT/ATTEMPT" ] && { echo "the confirmation set was read already ($OUT/ATTEMPT)" >&2; exit 3; }
[ "$(sha256sum "$UNITS" | cut -c1-64)" = afc93c818ba29dfa3f96177feeb7c3fac32645c219f53a2cc0f96f898e41de89 ] || exit 4
pp() {   # pp NAME BOXES
  "$PY" "$PAGEPATH" run --units "$UNITS" --boxes "$2" --models "$MODELS" --lm "$LM" --rubi "$RUBI" \
    --cache "$EV/pp_cache/reads.sqlite" --work "$EV/pp_work" --workers 5 --rows "$OUT/$1.rows.jsonl" --out "$OUT/$1.json"
}
# the qualifiers and their representative seeds (from the dev / hval record)
QUALS=$("$PY" -c "
import json
d = json.load(open('$REPO/eval/textspot8_dev_20261027.json'))
q = set(d['qualifiers'])
for c in d['candidates']:
    if f\"{c['arm']}|{c['decode']}\" in q:
        print(f\"{c['decode']}:{c['representative_seed']}\")")
echo "qualifiers: $QUALS"
for qs in $QUALS; do
  dec=${qs%%:*}; seed=${qs##*:}; n=t8_t64n8c16_seed$seed; blob=$RUNS/$n/final.vtsp; tag=${dec}_s$seed
  case $dec in
    crop|crop+refine|ctx)
      "$PY" "$REPO/eval/textspot8_confirm.py" boxes --blob "$blob" --decode "$dec" --out "$OUT/boxes_$tag.jsonl";;
    det|det+refine|det+D)
      [ -s "$OUT/spot_s$seed.jsonl" ] || "$PY" "$T7/eval/textspot7.py" spot-confirm --blob "$blob" --out "$OUT/spot_s$seed.jsonl"
      if [ "$dec" = det+D ]; then
        T7_CONFIRM_UNITS="$UNITS" "$PY" "$T7/eval/textspot7.py" boxes-d --spot "$OUT/spot_s$seed.jsonl" --set confirm \
          --steps join10,extend --out "$OUT/boxes_$tag.jsonl"
      else
        "$PY" "$REPO/eval/textspot8_confirm.py" boxes --spot "$OUT/spot_s$seed.jsonl" --decode "$dec" --out "$OUT/boxes_$tag.jsonl"
      fi;;
  esac
done
# the read: every qualifier's boxes exist; from here on the set counts as read (a second attempt is refused)
date -u +%FT%TZ > "$OUT/ATTEMPT"
for qs in $QUALS; do
  dec=${qs%%:*}; seed=${qs##*:}; tag=${dec}_s$seed
  pp "confirm_$tag" "$OUT/boxes_$tag.jsonl"
done
date -u +%FT%TZ > "$OUT/READ_COMPLETE"
echo "confirm read finished"

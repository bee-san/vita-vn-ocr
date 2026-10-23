#!/usr/bin/env bash
# select_tier.sh TIER STEP: stride16's development-only scoring, decisions and selection for one tier
# (eval/stride16_protocol.json). A copy of VOCR5's select_tier.sh with stride16's arms and eval/stride16_select.py.
#   export   every finished fine-tune of the tier's arms -> $S16/blobs/<run>_w8.vocr (eval/synth3_ablation.py export:
#            train/export.py, int8, the max width the stride implies); the incumbent's blob is linked as
#            $S16/blobs/<incumbent>_w8.vocr after a sha256 check
#   score    every blob without scores yet: teacher_val's and devnative's development files (eval/synth3_ablation.py
#            score) and the whole-dialogue cohorts (eval/finder_tighten.py score, tv_dlg + dn_dlg, GRAD and refined)
#   decide   eval/stride16_select.py decide for each arm of the tier (protocol selection.order) -> decision_<ARM>.json
#   select   eval/stride16_select.py select -> selection.json (needs the decisions selection.order asks for)
#   scores   eval/stride16_select.py scores -> scores.json (every scored run; report only)
set -euo pipefail
TIER=$1; STEP=$2
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; export HERD
REPO=${REPO:-/local/home/skerraut/herd-synth3/clones/stride16}; PY=$REPO/.venv-ci/bin/python
S16=$A/stride16; SEL=$S16/select/$TIER; B=$S16/blobs; mkdir -p "$SEL/scores" "$SEL/wd" "$B"
DV=$A/owner-drive-ingest-2/reclaim/pools; WDC=$A/vocr5-train/cohorts; FLOOR=$REPO/eval/stride16_floor_dev_20261023.json
DNF=(devnative_tv_native_ink_dev devnative_tv_native_grad_dev devnative_strict_ink_dev devnative_strict_grad_dev)
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) tier $TIER step $STEP"
mapfile -t ARMS < <($PY -c "
import json; p = json.load(open('eval/stride16_protocol.json'))
print('\n'.join(p['selection']['order']['$TIER']))")
mapfile -t RUNS < <($PY -c "
import json; p = json.load(open('eval/stride16_protocol.json'))
print('\n'.join(r for a in p['selection']['order']['$TIER'] for r in p['arms'][a]['runs']['fine_tunes']))")
read -r INC INC_BLOB INC_SHA < <($PY -c "
import json, os; i = json.load(open('eval/stride16_protocol.json'))['tiers']['$TIER']['incumbent']
print(i['name'], os.path.expandvars(i['blob']), i['sha256'])")
ALL=("${RUNS[@]}" "$INC")
# the floor file's pin: the latest amendment that re-pins it (amendment 1: rows keyed by position), else floor.sha256
WANT_FLOOR=$($PY -c "
import json; p = json.load(open('eval/stride16_protocol.json'))
print(([a['floor_sha256'] for a in p['amendments'] if a.get('floor_sha256')] or [p['floor']['sha256']])[-1])")
[ "$(sha256sum "$FLOOR" | cut -c1-64)" = "$WANT_FLOOR" ] || { echo "$FLOOR differs from the protocol"; exit 4; }
case $STEP in
export)
  [ "$(sha256sum "$INC_BLOB" | cut -c1-64)" = "$INC_SHA" ] || { echo "$INC_BLOB differs from the protocol"; exit 4; }
  ln -sfn "$INC_BLOB" "$B/${INC}_w8.vocr"
  for r in "${RUNS[@]}"; do
    ck=$S16/runs/$r/ckpt_step15000.pt
    [ -s "$B/${r}_w8.vocr" ] && continue
    [ -s "$ck" ] || { echo "pending $r"; continue; }
    $PY eval/synth3_ablation.py export --ckpt "$ck" --out "$B/${r}_w8.vocr" | tail -n 1
  done ;;
score)
  $PY - "$DV" "$WDC" <<'PY'
import hashlib, json, sys
from pathlib import Path
c = json.load(open("eval/vocr5_protocol.json"))["cohorts"]; dv, wd = Path(sys.argv[1]), Path(sys.argv[2])
want = {dv / "teacher_val_dev.npz": c["tv"]["sha256"]}
want.update({dv / f: s for f, s in c["dn"]["files"].items()})
want.update({wd / f: s for f, s in c["wd"]["files"].items()})
bad = [str(f) for f, s in want.items() if hashlib.sha256(f.read_bytes()).hexdigest() != s]
print(f"cohorts: {len(want) - len(bad)} match VOCR5's protocol, {len(bad)} differ {bad}"); sys.exit(1 if bad else 0)
PY
  for r in "${ALL[@]}"; do
    [ -e "$B/${r}_w8.vocr" ] || { echo "not exported: $r"; continue; }
    if [ ! -s "$SEL/scores/tv_$r.json" ]; then
      for f in "${DNF[@]}"; do
        $PY eval/synth3_ablation.py score --blob "$r=$B/${r}_w8.vocr" --dev "$DV/$f.npz" \
          --out "$SEL/scores/dn_${f}_$r.json" | tail -n 1
      done
      $PY eval/synth3_ablation.py score --blob "$r=$B/${r}_w8.vocr" --dev "$DV/teacher_val_dev.npz" \
        --out "$SEL/scores/tv_$r.tmp.json" | tail -n 1
      mv "$SEL/scores/tv_$r.tmp.json" "$SEL/scores/tv_$r.json"
    fi
    if [ ! -s "$SEL/wd/$r/meta.json" ]; then
      rm -rf "$SEL/wd/$r.tmp"
      $PY eval/finder_tighten.py score --cohorts "$WDC" --cohort tv_dlg --cohort dn_dlg --arm "$r=$B/${r}_w8.vocr" \
        --flags default --out "$SEL/wd/$r.tmp" | tail -n 2
      rm -rf "$SEL/wd/$r"; mv "$SEL/wd/$r.tmp" "$SEL/wd/$r"
    fi
  done ;;
decide|select|scores)
  TV=(); DNS=(); WD=()
  for r in "${ALL[@]}"; do
    [ -s "$SEL/scores/tv_$r.json" ] && [ -s "$SEL/wd/$r/meta.json" ] || continue
    TV+=("$SEL/scores/tv_$r.json"); WD+=("$SEL/wd/$r")
    for f in "${DNF[@]}"; do DNS+=("$SEL/scores/dn_${f}_$r.json"); done
  done
  C=(--protocol eval/stride16_protocol.json --floor "$FLOOR" --tv "${TV[@]}" --dn "${DNS[@]}" --wd "${WD[@]}")
  case $STEP in
  decide)
    for arm in "${ARMS[@]}"; do
      [ -s "$SEL/decision_$arm.json" ] && { echo "decided already: $arm"; continue; }
      $PY eval/stride16_select.py decide "${C[@]}" --arm "$arm" --out "$SEL/decision_$arm.json"
      $PY -c "import json, sys; sys.exit(0 if json.load(open('$SEL/decision_$arm.json'))['adopted'] else 1)" && break
    done ;;
  select)
    DEC=(); for arm in "${ARMS[@]}"; do [ -s "$SEL/decision_$arm.json" ] && DEC+=("$SEL/decision_$arm.json"); done
    $PY eval/stride16_select.py select "${C[@]}" --tier "$TIER" --decision "${DEC[@]}" --out "$SEL/selection.json" ;;
  scores)  $PY eval/stride16_select.py scores "${C[@]}" --tier "$TIER" --out "$SEL/scores.json" ;;
  esac ;;
*) echo "unknown step $STEP"; exit 2 ;;
esac
echo "SELECT_STEP_FINISHED $TIER $STEP $(date -u +%FT%TZ)"

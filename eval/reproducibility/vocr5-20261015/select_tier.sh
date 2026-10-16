#!/usr/bin/env bash
# select_tier.sh TIER STEP: VOCR5's development-only scoring, decisions and selection for one tier
# (eval/vocr5_protocol.json). A copy of VOCR4's select_size.sh with VOCR5's cohorts and eval/vocr5_select.py.
#   export   every finished fine-tune of the tier -> $V5/blobs/<run>_w8.vocr (eval/synth3_ablation.py export); the
#            incumbent's blob is linked as $V5/blobs/<incumbent>_w8.vocr after a sha256 check
#   score    every blob without scores yet: teacher_val's and devnative's development files (eval/synth3_ablation.py
#            score) and the whole-dialogue cohorts (eval/finder_tighten.py score, tv_dlg + dn_dlg, GRAD and refined);
#            refuses before eval/vocr5_pools.sha256 is committed
#   arm      eval/vocr5_select.py decide --decision arm -> arm_decision.json (needs the tier's P and F runs scored)
#   replace  eval/vocr5_select.py decide --decision replace -> replace_decision.json (needs the incumbent scored)
#   select   eval/vocr5_select.py select -> selection.json
#   scores   eval/vocr5_select.py scores -> scores.json (every scored run; report only)
set -euo pipefail
TIER=$1; STEP=$2
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; export HERD
REPO=${REPO:-/local/home/skerraut/herd-synth3/clones/vocr5-train}; PY=$REPO/.venv-ci/bin/python
V5=$A/vocr5-train; SEL=$V5/select/$TIER; B=$V5/blobs; mkdir -p "$SEL/scores" "$SEL/wd" "$B"
DV=$A/owner-drive-ingest-2/reclaim/pools; WDC=$V5/cohorts
DNF=(devnative_tv_native_ink_dev devnative_tv_native_grad_dev devnative_strict_ink_dev devnative_strict_grad_dev)
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) tier $TIER step $STEP"
mapfile -t RUNS < <($PY -c "
import json; p = json.load(open('eval/vocr5_protocol.json'))
print('\n'.join(r for arm in ('P', 'F') for r in p['runs']['fine_tunes']['$TIER'][arm]))")
read -r INC INC_BLOB INC_SHA < <($PY -c "
import json, os; i = json.load(open('eval/vocr5_protocol.json'))['tiers']['$TIER']['incumbent']
print(i['name'], os.path.expandvars(i['blob']), i['sha256'])")
ALL=("${RUNS[@]}" "$INC")
case $STEP in
export)
  [ "$(sha256sum "$INC_BLOB" | cut -c1-64)" = "$INC_SHA" ] || { echo "$INC_BLOB differs from the protocol"; exit 4; }
  ln -sfn "$INC_BLOB" "$B/${INC}_w8.vocr"
  for r in "${RUNS[@]}"; do
    ck=$V5/runs/$r/ckpt_step15000.pt
    [ -s "$B/${r}_w8.vocr" ] && continue
    [ -s "$ck" ] || { echo "pending $r"; continue; }
    $PY eval/synth3_ablation.py export --ckpt "$ck" --out "$B/${r}_w8.vocr" | tail -n 1
  done ;;
score)
  git ls-files --error-unmatch eval/vocr5_pools.sha256 >/dev/null 2>&1 \
    || { echo "eval/vocr5_pools.sha256 is not committed: no VOCR5 score before the pools are pinned"; exit 2; }
  $PY - "$DV" "$WDC" <<'PY'
import hashlib, json, sys
from pathlib import Path
c = json.load(open("eval/vocr5_protocol.json"))["cohorts"]; dv, wd = Path(sys.argv[1]), Path(sys.argv[2])
want = {dv / "teacher_val_dev.npz": c["tv"]["sha256"]}
want.update({dv / f: s for f, s in c["dn"]["files"].items()})
want.update({wd / f: s for f, s in c["wd"]["files"].items()})
bad = [str(f) for f, s in want.items() if hashlib.sha256(f.read_bytes()).hexdigest() != s]
print(f"cohorts: {len(want) - len(bad)} match the protocol, {len(bad)} differ {bad}"); sys.exit(1 if bad else 0)
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
arm|replace|select|scores)
  TV=(); DNS=(); WD=()
  for r in "${ALL[@]}"; do
    [ -s "$SEL/scores/tv_$r.json" ] && [ -s "$SEL/wd/$r/meta.json" ] || continue
    TV+=("$SEL/scores/tv_$r.json"); WD+=("$SEL/wd/$r")
    for f in "${DNF[@]}"; do DNS+=("$SEL/scores/dn_${f}_$r.json"); done
  done
  C=(--protocol eval/vocr5_protocol.json --tier "$TIER" --tv "${TV[@]}" --dn "${DNS[@]}" --wd "${WD[@]}")
  case $STEP in
  arm)     $PY eval/vocr5_select.py decide "${C[@]}" --decision arm --out "$SEL/arm_decision.json" ;;
  replace) $PY eval/vocr5_select.py decide "${C[@]}" --decision replace --arm "$SEL/arm_decision.json" \
             --out "$SEL/replace_decision.json" ;;
  select)  $PY eval/vocr5_select.py select "${C[@]}" --arm "$SEL/arm_decision.json" \
             --replace "$SEL/replace_decision.json" --out "$SEL/selection.json" ;;
  scores)  $PY eval/vocr5_select.py scores "${C[@]}" --out "$SEL/scores.json" ;;
  esac ;;
*) echo "unknown step $STEP"; exit 2 ;;
esac
echo "SELECT_STEP_FINISHED $TIER $STEP $(date -u +%FT%TZ)"

#!/usr/bin/env bash
# select_size.sh SIZE STEP: VOCR4's development-only scoring, decisions and selection for one size
# (eval/vocr4_protocol.json). STEP:
#   export  every finished run of the size -> $V4/blobs/<run>_w8.vocr (eval/synth3_ablation.py export)
#   score   every exported blob without scores yet, on teacher_val and devnative's four files (one file per run
#           and cohort file under $V4/select/SIZE/scores/); refuses before eval/vocr4_pools_a.sha256 is committed
#   budget  vita/budget_sweep.py check on the size's blobs -> $V4/select/SIZE/budget
#   real    eval/vocr4_select.py decide --decision real -> real_decision.json (needs C and A scored)
#   synth   eval/vocr4_select.py decide --decision synth -> synth_decision.json (needs M scored too)
#   select  eval/vocr4_select.py select -> selection.json
#   strata X Y  report only: eval/vocr4_select.py strata (arm X -> arm Y per tag) -> strata_X_Y.json
set -euo pipefail
SIZE=$1; STEP=$2
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; export HERD
REPO=${REPO:-/home/skerraut/herd-synth3/clones/vocr4-train}; PY=$REPO/.venv-ci/bin/python
V4=$A/vocr4-train; SEL=$V4/select/$SIZE; B=$V4/blobs; mkdir -p "$SEL/scores" "$B"
SP=$A/synth3/vocr3/select_pools; DN=$SP/devnative_v3
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) size $SIZE step $STEP"
mapfile -t RUNS < <($PY -c "
import json; p = json.load(open('eval/vocr4_protocol.json'))
print('\n'.join(r for arm in ('C', 'A', 'M') for r in p['arms'][arm]['runs']['$SIZE']))")
case $STEP in
export)
  for r in "${RUNS[@]}"; do
    ck=$V4/runs/$r/ckpt_step15000.pt
    [ -s "$B/${r}_w8.vocr" ] && continue
    [ -s "$ck" ] || { echo "pending $r"; continue; }
    $PY eval/synth3_ablation.py export --ckpt "$ck" --out "$B/${r}_w8.vocr" | tail -n 1
  done ;;
score)
  git ls-files --error-unmatch eval/vocr4_pools_a.sha256 >/dev/null 2>&1 \
    || { echo "eval/vocr4_pools_a.sha256 is not committed: no VOCR4 score before the A pools are pinned"; exit 2; }
  $PY - "$SP" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = json.load(open("eval/vocr4_protocol.json"))["cohorts"]; sp = Path(sys.argv[1])
want = {sp / "teacher_val.npz": p["teacher_val"]["sha256"]}
want.update({sp / "devnative_v3" / f: s for f, s in p["devnative"]["files"].items()})
bad = [str(f) for f, s in want.items() if hashlib.sha256(f.read_bytes()).hexdigest() != s]
print(f"cohorts: {len(want) - len(bad)} match the protocol, {len(bad)} differ {bad}"); sys.exit(1 if bad else 0)
PY
  for r in "${RUNS[@]}"; do
    [ -s "$B/${r}_w8.vocr" ] || { echo "not exported: $r"; continue; }
    [ -s "$SEL/scores/tv_$r.json" ] && continue
    for f in strict_ink strict_grad tv_native_ink tv_native_grad; do
      $PY eval/synth3_ablation.py score --blob "$r=$B/${r}_w8.vocr" --dev "$DN/devnative_$f.npz" \
        --out "$SEL/scores/dn_${f}_$r.json" | tail -n 1
    done
    $PY eval/synth3_ablation.py score --blob "$r=$B/${r}_w8.vocr" --dev "$SP/teacher_val.npz" \
      --out "$SEL/scores/tv_$r.tmp.json" | tail -n 1
    mv "$SEL/scores/tv_$r.tmp.json" "$SEL/scores/tv_$r.json"
  done ;;
budget)
  ALL=(); for r in "${RUNS[@]}"; do [ -s "$B/${r}_w8.vocr" ] && ALL+=("$B/${r}_w8.vocr"); done
  rm -rf "$SEL/budget"
  VITASDK=$A/vita/vitasdk VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz $PY vita/budget_sweep.py check \
    "${ALL[@]}" --out "$SEL/budget" --qemu "$A/vita/qemu" --vitasdk "$A/vita/vitasdk" \
    --arm-cc "$A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc" \
    2>&1 | tail -n $((${#ALL[@]} + 4)) ;;
real|synth|select|strata)
  TV=(); DNS=()
  for r in "${RUNS[@]}"; do
    [ -s "$SEL/scores/tv_$r.json" ] || continue
    TV+=("$SEL/scores/tv_$r.json")
    for f in strict_ink strict_grad tv_native_ink tv_native_grad; do DNS+=("$SEL/scores/dn_${f}_$r.json"); done
  done
  case $STEP in
  real)  $PY eval/vocr4_select.py decide --protocol eval/vocr4_protocol.json --size "$SIZE" --decision real \
           --tv "${TV[@]}" --dn "${DNS[@]}" --out "$SEL/real_decision.json" ;;
  synth) $PY eval/vocr4_select.py decide --protocol eval/vocr4_protocol.json --size "$SIZE" --decision synth \
           --tv "${TV[@]}" --dn "${DNS[@]}" --real "$SEL/real_decision.json" --out "$SEL/synth_decision.json" ;;
  strata) $PY eval/vocr4_select.py strata --protocol eval/vocr4_protocol.json --size "$SIZE" --tv "${TV[@]}" \
           --dn "${DNS[@]}" --x "$3" --y "$4" --out "$SEL/strata_$3_$4.json" | grep -E "kind:|target:|namelike|rare|len:" ;;
  select) $PY eval/vocr4_select.py select --protocol eval/vocr4_protocol.json --size "$SIZE" --tv "${TV[@]}" \
           --dn "${DNS[@]}" --real "$SEL/real_decision.json" --synth "$SEL/synth_decision.json" \
           --budget-dir "$SEL/budget" --out "$SEL/selection.json" ;;
  esac ;;
*) echo "unknown step $STEP"; exit 2 ;;
esac
echo "SELECT_STEP_FINISHED $SIZE $STEP $(date -u +%FT%TZ)"

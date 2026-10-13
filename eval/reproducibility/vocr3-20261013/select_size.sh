#!/usr/bin/env bash
# vocr3_select_size.sh SIZE [STEP]: VOCR3 selection for one size class (eval/vocr3_protocol.json), development data only.
# Steps: export (every candidate + rp run -> W8I blob), score (teacher_val both views; devnative v3's four npz), budget
# (vita/budget_sweep.py check from the main worktree), realplus decision, base selection. Outputs: $S3/vocr3/select/SIZE/.
set -euo pipefail
SIZE=$1; STEP=${2:-all}
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; S3=$A/synth3; export HERD
REPO=/local/home/skerraut/herd-vita-oct12/clones/vocr3-wt; PY=$REPO/.venv-ci/bin/python
SEL=$S3/vocr3/select/$SIZE; B=$S3/vocr3/blobs; mkdir -p "$SEL" "$B"
SP=$S3/vocr3/select_pools; DN=$SP/devnative_v3; DN_COMMIT=14ae22d48d5d100bc681763df3beabcc64d38fa7   # amendment 8: owner-series-free copies
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) size $SIZE step $STEP"
# candidate runs (base + rp) from the protocol
mapfile -t RUNS < <($PY - "$SIZE" <<'PY'
import json, sys
P = json.load(open("eval/vocr3_protocol.json")); size = sys.argv[1]; c = P["candidates"][size]
base = list(c.get("study2", [])) + list(c.get("step_curve", [])) + list(c.get("runs", [])) + list(c.get("extra", []))
rp = [b for _, b in P["candidates"]["realplus"][size]["decision_pairs"]] + list(P["candidates"]["realplus"][size]["extra_runs"])
print("\n".join(base + rp))
PY
)
ckpt_of() {   # run -> final fine-tune checkpoint
  for d in $S3/study2/runs $S3/vocr3/runs; do [ -s "$d/$1/ckpt_step15000.pt" ] && { echo "$d/$1/ckpt_step15000.pt"; return; }; done
  echo "MISSING $1" >&2; return 1
}
if [ "$STEP" = all ] || [ "$STEP" = export ]; then
  for r in "${RUNS[@]}"; do
    [ -s "$B/${r}_w8.vocr" ] && continue
    ck=$(ckpt_of "$r" 2>/dev/null) || { echo "pending $r (no final checkpoint yet)"; continue; }
    [ -s "$S3/study2/blobs/${r}_w8.vocr" ] && { cp -p "$S3/study2/blobs/${r}_w8.vocr" "$B/${r}_w8.vocr"; echo "copied study-2 export $r"; continue; }
    $PY eval/synth3_ablation.py export --ckpt "$ck" --out "$B/${r}_w8.vocr" | tail -n 1
  done
fi
BLOBS=(); for r in "${RUNS[@]}"; do BLOBS+=(--blob "$r=$B/${r}_w8.vocr"); done
if [ "$STEP" = all ] || [ "$STEP" = score ]; then
  for r in "${RUNS[@]}"; do [ -s "$B/${r}_w8.vocr" ] || { echo "cannot score: $r has no blob"; exit 4; }; done
  $PY eval/synth3_ablation.py score "${BLOBS[@]}" --dev $SP/teacher_val.npz --out $SEL/teacher_val.json | tail -n 3
  for f in strict_ink strict_grad tv_native_ink tv_native_grad; do
    $PY eval/synth3_ablation.py score "${BLOBS[@]}" --dev $DN/devnative_$f.npz --out $SEL/devnative_$f.json | tail -n 1
  done
  (cd $SP && sha256sum --quiet -c "$REPO/eval/vocr3_select_pools.sha256") && sha256sum $SP/teacher_val.npz $DN/devnative_*.npz > $SEL/cohort_npz.sha256
  cp /local/home/skerraut/herd-synth3/devnative.status $SEL/devnative_status_at_scoring.txt
fi
if [ "$STEP" = all ] || [ "$STEP" = budget ]; then
  # one check call per size: budget_sweep.py check rewrites sizes.json, so all blobs go in together
  ALL=(); for r in "${RUNS[@]}"; do [ -s "$B/${r}_w8.vocr" ] && ALL+=("$B/${r}_w8.vocr"); done
  rm -rf "$SEL/budget"; bash $S3/vocr3/budget/check.sh $SEL/budget "${ALL[@]}" 2>&1 | tail -n $((${#ALL[@]} + 4))
fi
DNS=($SEL/devnative_strict_ink.json $SEL/devnative_strict_grad.json $SEL/devnative_tv_native_ink.json $SEL/devnative_tv_native_grad.json)
if [ "$STEP" = all ] || [ "$STEP" = select ]; then
  $PY eval/vocr3_select.py realplus --protocol eval/vocr3_protocol.json --size $SIZE --tv $SEL/teacher_val.json --dn "${DNS[@]}" --out $SEL/realplus_decision.json | tail -n 6
  $PY eval/vocr3_select.py base --protocol eval/vocr3_protocol.json --size $SIZE --tv $SEL/teacher_val.json --dn "${DNS[@]}" --dn-commit $DN_COMMIT \
     --budget-dir $SEL/budget --realplus $SEL/realplus_decision.json --out $SEL/selection.json
fi
echo "SELECT_SIZE_FINISHED $SIZE $(date -u +%FT%TZ)"

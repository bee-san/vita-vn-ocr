#!/usr/bin/env bash
# vocab_proof_score.sh: eval/vocab_proof_protocol.json's scoring, run once both G_v4987 ckpt_step16000.pt exist.
# Exports G_v4987's two EMA step-16000 checkpoints, and reuses F_s4's two published blobs (hash-checked against the
# published blobs), reads teacher_val and devnative v3 (the VOCR3 selection's amendment-8 copies) through the
# C runtime W8I, compares G against F with eval/synth3_ablation.py (the synth4 ablation's machinery), and adds
# the character strata and the synthetic probe (train/vocab_expand.py proof).
#   HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 bash eval/vocab_proof_score.sh < /dev/null
set -euo pipefail
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
PR=$A/vocab-expand/proof
SEL=$A/synth3/vocr3/select_pools
PROTO=eval/vocab_proof_protocol.json
EXP=$A/synth4-design/gap/exposures_s4v1.json
PY=.venv-ci/bin/python
cd "$REPO"
echo "git $(git rev-parse HEAD)"
mkdir -p "$PR/blobs" "$PR/scores" "$PR/compare"
# verify F_s4's reused blobs are the published bytes, and the proof vocab matches the protocol
$PY - "$PROTO" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()  # noqa: E731
assert sha(Path("data") / Path(p["arms_detail"]["G_v4987"]["vocab"]["file"]).name) == p["arms_detail"]["G_v4987"]["vocab"]["sha256"]
for s in (1, 2):
    rec = p["arms_detail"]["F_s4"]["checkpoints"][f"F_s4_seed{s}"]
    assert sha(rec["blob"]) == rec["blob_sha256"], f"F_s4 seed{s} blob differs from the protocol"
    assert sha(rec["ckpt"]) == rec["ckpt_sha256"], f"F_s4 seed{s} checkpoint differs from the protocol"
for name, d in p["dev"].items():
    for f, rec in d["files"].items():
        assert sha(rec["set"]) == rec["sha256"], f"{name}/{f} differs from the protocol"
print("F_s4 blobs, checkpoints, the proof vocab and the dev files match the protocol")
PY
BLOBS=()
for s in 1 2; do
  cp "$A/synth4-design/ablation/blobs/F_s4_seed${s}_w8.vocr" "$PR/blobs/F_s4_seed${s}_w8.vocr"
  $PY eval/synth3_ablation.py export --ckpt "$PR/runs/G_v4987_seed${s}/ckpt_step16000.pt" \
      --out "$PR/blobs/G_v4987_seed${s}_w8.vocr"
  BLOBS+=(--blob "F_s4_seed${s}=$PR/blobs/F_s4_seed${s}_w8.vocr" --blob "G_v4987_seed${s}=$PR/blobs/G_v4987_seed${s}_w8.vocr")
done
# teacher_val (decides) and devnative v3 pooled (guard)
$PY eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$SEL/teacher_val.npz" --out "$PR/scores/teacher_val.json"
$PY eval/synth3_ablation.py compare --scores "$PR/scores/teacher_val.json" --protocol "$PROTO" \
  --out "$PR/compare/teacher_val.json"
DN=()
for f in strict_ink strict_grad tv_native_ink tv_native_grad; do
  $PY eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$SEL/devnative_v3/devnative_$f.npz" \
    --out "$PR/scores/devnative_$f.json"
  DN+=("$PR/scores/devnative_$f.json")
done
$PY eval/synth3_study2.py merge --scores "${DN[@]}" --out "$PR/scores/devnative_pooled.json"
$PY eval/synth3_ablation.py compare --scores "$PR/scores/devnative_pooled.json" --protocol "$PROTO" \
  --out "$PR/compare/devnative_pooled.json"
# character-level strata (added / shared / dropped recall) on teacher_val and devnative
$PY train/vocab_expand.py proof strata --protocol "$PROTO" \
  --scores "$PR/scores/teacher_val.json" "$PR/scores"/devnative_*.json --out eval/vocab_proof_strata_20261014.json
# the synthetic probe: per-class recall of the added classes by exposure bin
PROBE=()
for s in 1 2; do
  PROBE+=(--blob "F_s4_seed${s}=$PR/blobs/F_s4_seed${s}_w8.vocr" --blob "G_v4987_seed${s}=$PR/blobs/G_v4987_seed${s}_w8.vocr")
done
$PY train/vocab_expand.py proof probe --protocol "$PROTO" --exposures "$EXP" "${PROBE[@]}" \
  --out eval/vocab_proof_probe_20261014.json
echo PROOF_SCORE_DONE

#!/usr/bin/env bash
# synth4_score.sh: eval/synth4_protocol.json's scoring, run once both F_s4 ckpt_step16000.pt exist. Exports the four
# EMA step-16000 checkpoints (per-channel int8; D_s3v3's again, checked against the protocol's pinned blob sha256s),
# reads teacher_val and devnative v3 (the VOCR3 selection's amendment-8 copies: no owner-reserved title) through the
# C runtime in W8I, adds the protocol's curriculum strata, and applies the comparison and the two-part decision.
#   HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 bash eval/synth4_score.sh < /dev/null
set -euo pipefail
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
AB=$A/synth4-design/ablation
PROTO=eval/synth4_protocol.json
SEL=$A/synth3/vocr3/select_pools
CUR=$A/synth4-design/curriculum/curriculum_v1.json
V3500=$A/train2/release/FL10_vocab_3500.txt
PY=.venv-ci/bin/python
cd "$REPO"
echo "git $(git rev-parse HEAD)"
mkdir -p "$AB/blobs" "$AB/scores" "$AB/compare"
$PY - "$PROTO" "$SEL" "$CUR" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()  # noqa: E731
for name, d in p["dev"].items():
    for f, rec in d["files"].items():
        assert sha(rec["set"]) == rec["sha256"], f"{name}/{f} differs from the protocol"
assert sha(sys.argv[3]) == p["strata"]["curriculum_sha256"], "curriculum differs from the protocol"
for run, rec in p["runs"]["D_s3v3"]["checkpoints"].items():
    assert sha(rec["ckpt"]) == rec["ckpt_sha256"], f"{run} checkpoint differs from the protocol"
print("inputs match the protocol")
PY
BLOBS=()
for r in D_s3v3 F_s4; do for s in 1 2; do
  if [ $r = D_s3v3 ]; then ck=$A/synth3-v3/ablation/runs/${r}_seed$s/ckpt_step16000.pt; else ck=$AB/runs/${r}_seed$s/ckpt_step16000.pt; fi
  [ -f "$ck" ] || { echo "missing $ck"; exit 3; }
  $PY eval/synth3_ablation.py export --ckpt "$ck" --out "$AB/blobs/${r}_seed${s}_w8.vocr"
  BLOBS+=(--blob "${r}_seed$s=$AB/blobs/${r}_seed${s}_w8.vocr")
done; done
$PY - "$PROTO" "$AB/blobs" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for run, rec in p["runs"]["D_s3v3"]["checkpoints"].items():
    got = hashlib.sha256((Path(sys.argv[2]) / f"{run}_w8.vocr").read_bytes()).hexdigest()
    assert got == rec["blob_sha256"], f"{run}: re-export {got} differs from study 1b's blob {rec['blob_sha256']}"
print("D_s3v3 re-exports are study 1b's blobs, byte for byte")
PY
tag() { $PY eval/synth4_ablation.py tag --scores "$1" --curriculum "$CUR" --fl10-vocab "$V3500" --out "$1"; }
$PY eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$SEL/teacher_val.npz" --out "$AB/scores/teacher_val.json"
tag "$AB/scores/teacher_val.json"
$PY eval/synth3_ablation.py compare --scores "$AB/scores/teacher_val.json" --protocol "$PROTO" \
  --out "$AB/compare/teacher_val.json"
DN=(); DNF=()
for f in strict_ink strict_grad tv_native_ink tv_native_grad; do
  $PY eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$SEL/devnative_v3/devnative_$f.npz" \
    --out "$AB/scores/devnative_$f.json"
  tag "$AB/scores/devnative_$f.json"
  $PY eval/synth3_ablation.py compare --scores "$AB/scores/devnative_$f.json" --protocol "$PROTO" \
    --out "$AB/compare/devnative_$f.json"
  DN+=("$AB/scores/devnative_$f.json"); DNF+=(--dn-file "$f=$AB/compare/devnative_$f.json")
done
$PY eval/synth3_study2.py merge --scores "${DN[@]}" --out "$AB/scores/devnative_pooled.json"
$PY eval/synth3_ablation.py compare --scores "$AB/scores/devnative_pooled.json" --protocol "$PROTO" \
  --out "$AB/compare/devnative_pooled.json"
$PY eval/synth4_ablation.py result --protocol "$PROTO" --tv "$AB/compare/teacher_val.json" \
  --dn "$AB/compare/devnative_pooled.json" "${DNF[@]}" --out eval/synth4_ablation_20261013.json
echo SCORE_DONE

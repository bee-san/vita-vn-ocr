#!/usr/bin/env bash
# synth3_native_score.sh: eval/synth3_native_protocol.json's scoring, run once both E_s3nat ckpt_step16000.pt exist.
# Exports all four step-16000 EMA checkpoints (D_s3v3 again, by the same code), scores devnative v3's four files without the
# owner's test series (decides, pooled) and teacher_val likewise (reported) through the C runtime in W8I, and applies the
# protocol's comparison.
#   HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 bash eval/synth3_native_score.sh < /dev/null
set -euo pipefail
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
ST=$A/native-frames/study
PROTO=eval/synth3_native_protocol.json
PY=.venv-ci/bin/python
cd "$REPO"
echo "git $(git rev-parse HEAD)"
mkdir -p "$ST/blobs" "$ST/scores" "$ST/compare"
# the development files the protocol pins (devnative v3 and teacher_val without the owner's test series)
HERD_ART="$A" $PY - "$PROTO" <<'PY'
import hashlib, json, os, sys
p = json.load(open(sys.argv[1]))
files = list(p["dev"]["devnative_v3"]["files"].values()) + [p["dev"]["teacher_val"]]
for f in files:
    path = f["set"].replace("$HERD/shared/artifacts", os.environ["HERD_ART"])
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != f["sha256"]:
        sys.exit(f"{path}: sha256 {got} is not the protocol's {f['sha256']}")
print("development files match the protocol")
PY
BLOBS=()
for r in D_s3v3 E_s3nat; do for s in 1 2; do
  if [ "$r" = D_s3v3 ]; then ck=$A/synth3-v3/ablation/runs/${r}_seed$s/ckpt_step16000.pt
  else ck=$ST/runs/${r}_seed$s/ckpt_step16000.pt; fi
  [ -f "$ck" ] || { echo "missing $ck"; exit 3; }
  $PY eval/synth3_ablation.py export --ckpt "$ck" --out "$ST/blobs/${r}_seed${s}_w8.vocr"
  BLOBS+=(--blob "${r}_seed$s=$ST/blobs/${r}_seed${s}_w8.vocr")
done; done
DN=()
for f in strict_ink strict_grad tv_native_ink tv_native_grad; do
  $PY eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$ST/dev/devnative_${f}_noowner.npz" --out "$ST/scores/devnative_v3_$f.json"
  $PY eval/synth3_ablation.py compare --scores "$ST/scores/devnative_v3_$f.json" --protocol "$PROTO" \
    --out "$ST/compare/devnative_v3_$f.json"
  DN+=("$ST/scores/devnative_v3_$f.json")
done
$PY eval/synth3_study2.py merge --scores "${DN[@]}" --out "$ST/scores/devnative_v3_pooled.json"
$PY eval/synth3_ablation.py compare --scores "$ST/scores/devnative_v3_pooled.json" --protocol "$PROTO" \
  --out "$ST/compare/devnative_v3_pooled.json"
$PY eval/synth3_ablation.py score "${BLOBS[@]}" --dev "$ST/dev/teacher_val_noowner.npz" --out "$ST/scores/teacher_val.json"
$PY eval/synth3_ablation.py compare --scores "$ST/scores/teacher_val.json" --protocol "$PROTO" \
  --out "$ST/compare/teacher_val.json"
ST="$ST" PROTO="$PROTO" $PY - <<'PY'
import datetime, hashlib, json, os, subprocess
from pathlib import Path
st, proto = Path(os.environ["ST"]), os.environ["PROTO"]
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
pooled = load(st / "compare" / "devnative_v3_pooled.json")
c = pooled["comparisons"]["E_vs_D"]
d_better = bool(c["all_vn_cluster"]["ci95"][0] > 0 and all(v["delta_points"] > 0 for v in c["per_seed"].values()))
out = {"kind": "synth3-native-result",
       "protocol": {"file": proto, "sha256": hashlib.sha256(Path(proto).read_bytes()).hexdigest()},
       "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
       "scored": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "decision": {"e_better": c["decision"]["b_better"], "d_better_mirrored": d_better, "rule": c["decision"]["rule"],
                    "delta_points": c["all_vn_cluster"]["delta_points"], "ci95": c["all_vn_cluster"]["ci95"],
                    "per_seed": {k: v["delta_points"] for k, v in c["per_seed"].items()}},
       "devnative_v3_pooled": pooled,
       "devnative_v3_files": {f: load(st / "compare" / f"devnative_v3_{f}.json")
                              for f in ("strict_ink", "strict_grad", "tv_native_ink", "tv_native_grad")},
       "teacher_val_reported_only": load(st / "compare" / "teacher_val.json")}
Path("eval/synth3_native_20261013.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out["decision"], indent=1))
PY
echo SCORED

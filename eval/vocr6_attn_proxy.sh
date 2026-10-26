#!/usr/bin/env bash
# vocr6_attn_proxy.sh ATTN [SEED] [OUT]: one run of the attention block's training smoke test (docs/vocr6-attn.md,
# "Proxy"), in one heavy.sh slot. A second 15,000-step fine-tune of VOCR5-N from v5f_N_seed1's final checkpoint, with
# VOCR5-N's own fine-tune recipe and mix (eval/vocr5_run.sh N F: the M recipe's synthetic half with finder
# augmentation, the pinned real pools of eval/vocr5_ft_sources.json with their refined copies), --attn ATTN (0: the
# control, 1: the attention block, which starts at gate 0, i.e. as v5f_N_seed1 itself). Logs teacher_val_dev as
# VOCR5 did (rv, rvn); the run's final checkpoint is the run. A smoke test, not a selection: nothing is decided on it.
#   ATTN: 0 or 1    SEED: the fine-tune's seed (default 1)    OUT: runs directory (default the label's artifacts)
set -euo pipefail
ATTN=${1:?usage: vocr6_attn_proxy.sh ATTN [SEED] [OUT]}; SEED=${2:-1}
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}; A=$HERD/shared/artifacts
OUT=${3:-$A/vocr6-s3-attn/runs}
REPO=$(cd "$(dirname "$0")/.." && pwd)
V5=$A/vocr5-train
INIT=$V5/runs/v5f_N_seed1/ckpt_step15000.pt
VOCAB=$REPO/data/vocab_vn_5631.txt
case $ATTN in 0|1) ;; *) echo "ATTN must be 0 or 1"; exit 2 ;; esac
cd "$REPO"
[ "$(sha256sum "$VOCAB" | cut -c1-64)" = 22216fc5ac73160f941548b7e7acff6024fe4e27fb8b6eb2081afe98041c74f0 ] \
  || { echo "$VOCAB differs from VOCR5's (eval/vocr5_protocol.json)"; exit 4; }
TVD=$A/owner-drive-ingest-2/reclaim/pools/teacher_val_dev.npz
[ "$(sha256sum "$TVD" | cut -c1-64)" = e2f0300fd9f417b6c3c01e1dab36641bfd72749e2cac5db896226ef0d29b65dd ] \
  || { echo "teacher_val_dev.npz differs from the protocol"; exit 4; }
[ -s "$INIT" ] || { echo "no $INIT"; exit 5; }
(cd / && sha256sum --quiet -c "$REPO/eval/vocr5_pools.sha256")
S3V3D=$A/synth3-v3/shards/train_v3; S4D=$V5/render/s4v5631/shards
for m in "data/manifests/synth3_shards_v3.json $S3V3D" "data/manifests/synth4_shards_v5631.json $S4D"; do
  set -- $m
  .venv-ci/bin/python - "$1" "$2" <<'PY'
import hashlib, json, sys
from pathlib import Path
m = json.load(open(sys.argv[1])); d = Path(sys.argv[2])
bad = [s["file"] for s in m["shards"] if hashlib.sha256((d / s["file"]).read_bytes()).hexdigest() != s["sha256"]]
print(f"{sys.argv[1]}: {len(m['shards']) - len(bad)} shards match, {len(bad)} differ")
sys.exit(1 if bad else 0)
PY
done
NVG="$A/nvsynth/full/train/nv_train_0[0-3]?_b*.npz+$A/nvsynth/full/train/nv_train_04[01]_b*.npz"
S2G="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
FSYN=(--src "nv:0.225:$NVG:stream=1,pool=3,inv=0.2,fa=0.15" --src "s2:0.1125:$S2G:stream=1,pool=3,inv=0.2,fa=0.15"
      --src "s3v3:0.1125:$S3V3D/synth3_000??.npz:stream=1,pool=3,inv=0.2" --src "s4:0.05:$S4D/synth4_000??.npz:stream=1,pool=3,inv=0.2")
mapfile -t REAL < <(.venv-ci/bin/python - eval/vocr5_ft_sources.json <<'PY'
import json, sys
for s in json.load(open(sys.argv[1]))["real"]:
    opts = "kd=0,inv=0.5" + (f",fa={s['fa']}" if s.get("fa") else "")
    print("--src"); print(f"{s['name']}:{s['weight']}:{'+'.join(s['files'] + s.get('refined', []))}:{opts}")
PY
)
RUN=vocr6attn_N_a${ATTN}_s${SEED}
if [ -s "$OUT/$RUN/ckpt_step15000.pt" ]; then echo "$RUN is complete"; exit 0; fi
if [ -e "$OUT/$RUN" ]; then echo "$OUT/$RUN is partial: delete it and restart (fine-tunes never resume)"; exit 3; fi
echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) $RUN from $INIT"
exec .venv-ci/bin/python train/train2.py --run "$RUN" --out "$OUT" --init "$INIT" \
  --steps 15000 --batch 96 --lr 1.2e-3 --warmup 500 --final-lr 0.05 --wd 0.01 --ema 0.999 --width 1.5 \
  --attn "$ATTN" --vocab "$VOCAB" --seed "$SEED" --threads 5 --eval-every 2500 --keep-evals "${FSYN[@]}" "${REAL[@]}" \
  --val "rv:$TVD" --val "rvn:$TVD:native=1" --select rv,rvn

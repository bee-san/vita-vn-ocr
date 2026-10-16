#!/usr/bin/env bash
# vocr5_run.sh TIER STAGE SEED: one VOCR5 run (eval/vocr5_protocol.json), in one heavy.sh slot.
#   TIER:  S  Small,  s_w100v4987: width 1.0, data/vocab_4987_proof.txt (4,987 classes), recipe C
#          N  Normal, n_w150v5631: width 1.5, data/vocab_vn_5631.txt (5,631 classes), recipe M
#          L  Large,  l_lw300v5631: width 2.0 early, 3.0 late (--chans 64,128,256,480,576), data/vocab_vn_5631.txt, M
#   STAGE: pre  the tier's pretrain pre5_<TIER>_seed<SEED> from scratch
#          P    the plain fine-tune v5p_<TIER>_seed<SEED> from that pretrain (every pool, no finder augmentation)
#          F    the finder-robust fine-tune v5f_<TIER>_seed<SEED> (P's sources and weights, fa= on the ink sources,
#               the GRAD sources with their refined copies)
#   SEED:  1 or 2 (Small), 1-3 (Normal, Large): the pretrain's seed and its fine-tunes' seed
# Recipes: C = nv 0.5 + synth2 0.25 + synth3 v2 0.25 (pretrain); M = nv 0.45 + synth2 0.225 + synth3 v3 0.225 +
# s4v5631 0.1. Fine-tunes: synthetic half 0.5 (the recipe's sources at half weight), real half 0.5 from the pinned
# pools (eval/vocr5_pools.sha256) with the per-source weights of eval/vocr5_ft_sources.json.
# Inputs are checked first: the vocab's sha256 (the protocol), synth3's shard manifests, the s4v5631 render's manifest
# (data/manifests/synth4_shards_v5631.json) for N and L, the development file the runs log on, and for a fine-tune
# the pool pin and the pretrain checkpoint. A pretrain that dies after its first checkpoint resumes from
# ckpt_last.pt (VOCR5_RESUME=1, recorded in the log); any other partial directory stops the script.
set -euo pipefail
TIER=$1; STAGE=$2; SEED=$3
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}; A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
V5=$A/vocr5-train; OUT=$V5/runs; P5=$V5/pools; Q5=$HERD/shared/private/vocr5-train/pools
case $TIER in
  S) NAME=Small;  ARCH=(--width 1.0);                                   VOCAB=$REPO/data/vocab_4987_proof.txt; RECIPE=C ;;
  N) NAME=Normal; ARCH=(--width 1.5);                                   VOCAB=$REPO/data/vocab_vn_5631.txt;    RECIPE=M ;;
  L) NAME=Large;  ARCH=(--width 2.0 --chans 64,128,256,480,576);        VOCAB=$REPO/data/vocab_vn_5631.txt;    RECIPE=M ;;
  *) echo "unknown tier $TIER"; exit 2 ;;
esac
case $STAGE in pre|P|F) ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
cd "$REPO"
PROTO=eval/vocr5_protocol.json
read -r STEPS SEEDS WANT_VOCAB < <(.venv-ci/bin/python - "$PROTO" "$NAME" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))["tiers"][sys.argv[2]]
print(t["pretrain_steps"], ",".join(map(str, t["seeds"])), t["vocab"]["sha256"])
PY
)
case ",$SEEDS," in *",$SEED,"*) ;; *) echo "seed $SEED is not one of tier $TIER's ($SEEDS)"; exit 2 ;; esac
[ "$(sha256sum "$VOCAB" | cut -c1-64)" = "$WANT_VOCAB" ] || { echo "$VOCAB differs from the protocol's sha256"; exit 4; }
echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) vocr5 $TIER $STAGE seed $SEED"
NVG="$A/nvsynth/full/train/nv_train_0[0-3]?_b*.npz+$A/nvsynth/full/train/nv_train_04[01]_b*.npz"
S2G="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
S3D=$A/synth3/shards/train_v2; S3G="$S3D/synth3_000??.npz"
S3V3D=$A/synth3-v3/shards/train_v3; S3V3G="$S3V3D/synth3_000??.npz"
S4D=$V5/render/s4v5631/shards; S4G="$S4D/synth4_000??.npz"
TVD=$A/owner-drive-ingest-2/reclaim/pools/teacher_val_dev.npz      # logged only: the final checkpoint is the run
VALS=(--val "rv:$TVD" --val "rvn:$TVD:native=1" --select rv,rvn)
check_shards() {   # check_shards MANIFEST DIR: every shard of a pinned manifest has its sha256
  .venv-ci/bin/python - "$1" "$2" <<'PY'
import hashlib, json, sys
from pathlib import Path
m = json.load(open(sys.argv[1])); d = Path(sys.argv[2])
bad = [s["file"] for s in m["shards"] if hashlib.sha256((d / s["file"]).read_bytes()).hexdigest() != s["sha256"]]
print(f"{sys.argv[1]}: {len(m['shards']) - len(bad)} shards match, {len(bad)} differ")
sys.exit(1 if bad else 0)
PY
}
[ "$(sha256sum "$TVD" | cut -c1-64)" = "e2f0300fd9f417b6c3c01e1dab36641bfd72749e2cac5db896226ef0d29b65dd" ] \
  || { echo "teacher_val_dev.npz differs from the protocol"; exit 4; }
if [ "$RECIPE" = C ]; then
  check_shards data/manifests/synth3_shards_v2.json "$S3D"
  PSRC=(--src "nv:0.5:$NVG:stream=1,pool=3,inv=0.2" --src "s2:0.25:$S2G:stream=1,pool=3,inv=0.2"
        --src "s3:0.25:$S3G:stream=1,pool=3,inv=0.2")
else
  check_shards data/manifests/synth3_shards_v3.json "$S3V3D"
  [ -s data/manifests/synth4_shards_v5631.json ] || { echo "the s4v5631 render is not pinned yet"; exit 5; }
  check_shards data/manifests/synth4_shards_v5631.json "$S4D"
  PSRC=(--src "nv:0.45:$NVG:stream=1,pool=3,inv=0.2" --src "s2:0.225:$S2G:stream=1,pool=3,inv=0.2"
        --src "s3v3:0.225:$S3V3G:stream=1,pool=3,inv=0.2" --src "s4:0.1:$S4G:stream=1,pool=3,inv=0.2")
fi
PRUN=pre5_${TIER}_seed${SEED}; INIT=$OUT/$PRUN/ckpt_step$STEPS.pt
if [ "$STAGE" = pre ]; then
  if [ -s "$INIT" ]; then echo "$PRUN is complete"; exit 0; fi
  if [ -e "$OUT/$PRUN" ]; then
    if [ "${VOCR5_RESUME:-0}" = 1 ] && [ -s "$OUT/$PRUN/ckpt_last.pt" ]; then
      echo "RESUME $PRUN from ckpt_last.pt (train2's resume reseeds the batch stream; the protocol records it)"
    else
      echo "$OUT/$PRUN is partial: resume with VOCR5_RESUME=1 (pretrain, after a checkpoint) or delete it"; exit 3
    fi
  fi
  exec .venv-ci/bin/python train/train2.py --run "$PRUN" --out "$OUT" --steps "$STEPS" --batch 96 --lr 2e-3 \
    --warmup 1000 --final-lr 0.05 --wd 0.01 --ema 0.999 "${ARCH[@]}" --vocab "$VOCAB" --seed "$SEED" --threads 5 \
    --eval-every 4000 --keep-evals "${PSRC[@]}" "${VALS[@]}" \
    --val "nvv:$A/nvsynth/shards/nv_val.npz:n=2000" --val "s2v:$A/synth2/shards/synth_00039.npz:n=2000"
fi
# ---------------------------------------------------------------- fine-tunes
[ -s "$INIT" ] || { echo "$PRUN has not finished ($INIT)"; exit 5; }
PIN=$REPO/eval/vocr5_pools.sha256; SRCS=$REPO/eval/vocr5_ft_sources.json
git ls-files --error-unmatch eval/vocr5_pools.sha256 eval/vocr5_ft_sources.json >/dev/null 2>&1 \
  || { echo "the fine-tune pools are not pinned (committed) yet"; exit 5; }
(cd / && sha256sum --quiet -c "$PIN")
mapfile -t REAL < <(.venv-ci/bin/python - "$SRCS" "$STAGE" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1])); arm = sys.argv[2]
for s in doc["real"]:
    opts = "kd=0,inv=0.5" + (f",fa={s['fa']}" if arm == "F" and s.get("fa") else "")
    files = s["files"] + (s.get("refined", []) if arm == "F" else [])
    print("--src"); print(f"{s['name']}:{s['weight']}:{'+'.join(files)}:{opts}")
PY
)
FA_NV=""
if [ "$STAGE" = F ]; then
  FA_NV=",fa=0.15"
  grep -q "finder_aug" train/train2.py || { echo "this checkout's train2 has no fa= option (train/finder_aug.py)"; exit 5; }
fi
if [ "$RECIPE" = C ]; then
  FSYN=(--src "nv:0.25:$NVG:stream=1,pool=3,inv=0.2$FA_NV" --src "s2:0.125:$S2G:stream=1,pool=3,inv=0.2$FA_NV"
        --src "s3:0.125:$S3G:stream=1,pool=3,inv=0.2")
else
  FSYN=(--src "nv:0.225:$NVG:stream=1,pool=3,inv=0.2$FA_NV" --src "s2:0.1125:$S2G:stream=1,pool=3,inv=0.2$FA_NV"
        --src "s3v3:0.1125:$S3V3G:stream=1,pool=3,inv=0.2" --src "s4:0.05:$S4G:stream=1,pool=3,inv=0.2")
fi
RUN=v5$(echo "$STAGE" | tr 'PF' 'pf')_${TIER}_seed${SEED}
if [ -s "$OUT/$RUN/ckpt_step15000.pt" ]; then echo "$RUN is complete"; exit 0; fi
if [ -e "$OUT/$RUN" ]; then echo "$OUT/$RUN is partial: delete it and restart (fine-tunes never resume)"; exit 3; fi
echo "run $RUN from $INIT"
exec .venv-ci/bin/python train/train2.py --run "$RUN" --out "$OUT" --init "$INIT" \
  --steps 15000 --batch 96 --lr 1.2e-3 --warmup 500 --final-lr 0.05 --wd 0.01 --ema 0.999 "${ARCH[@]}" \
  --vocab "$VOCAB" --seed "$SEED" --threads 5 --eval-every 2500 --keep-evals "${FSYN[@]}" "${REAL[@]}" "${VALS[@]}"

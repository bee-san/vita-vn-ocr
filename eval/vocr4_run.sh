#!/usr/bin/env bash
# vocr4_run.sh SIZE ARM SEED: one VOCR4 run (eval/vocr4_protocol.json), in one heavy.sh slot.
#   SIZE: FL10 (width 1.0, FL10's 3,500 vocab), H15 (width 1.5, data/vocab.txt), W20 (width 2.0, data/vocab.txt)
#   ARM:  C  the control: VOCR3's realplus fine-tune (eval/vocr3_rp_run.sh: nv 0.25 + synth2 0.125 + synth3 v2 0.125,
#            tt 0.2 + rd 0.1 + realplus v2 0.2) on VOCR4's cleaned pools, from the pretrain of VOCR3's selection
#         A  the new real pools: C with the real half re-split as tt 0.175 + rd 0.1 + realplus v3i 0.175 + native
#            pool v1 0.025 + owner frames 0.025 (the pinned files decide; the protocol's fallbacks below)
#         M  synth3 v3: a new pretrain (nv 0.5 + synth2 0.25 + synth3 v3 0.25; VOCR3's steps for the size; seed SEED),
#            then A's fine-tune with synth3 v3 in place of synth3 v2
#   SEED: the fine-tune seed (1 or 2); for M also the pretrain's
# Inputs are checked first: the reused pretrains against eval/vocr4_inputs.sha256, synth3 v3's shards against
# data/manifests/synth3_shards_v3.json, the pools against eval/vocr4_pools_c.sha256 (and eval/vocr4_pools_a.sha256
# for an A or M fine-tune). VOCR4_PRETRAIN_ONLY=1 stops an M run after its pretrain.
# A run never resumes: a partial directory stops the script (delete it and restart with the same arguments).
set -euo pipefail
SIZE=$1; ARM=$2; SEED=$3
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}; A=$HERD/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
V4=$A/vocr4-train; OUT=$V4/runs; P4=$V4/pools
case $SIZE in
  FL10) WIDTH=1.0; VOCAB=$A/train2/release/FL10_vocab_3500.txt; PRE=128000
        BASE=synth3/study2/runs/pre_FL10_C_mix_pre128k_seed1/ckpt_step128000.pt ;;
  H15)  WIDTH=1.5; VOCAB=$REPO/data/vocab.txt; PRE=40000
        BASE=synth3/study2/runs/pre_H15_C_mix_seed2/ckpt_step40000.pt ;;
  W20)  WIDTH=2.0; VOCAB=$REPO/data/vocab.txt; PRE=32000
        BASE=synth3/vocr3/runs/pre_W20_C_mix_seed1/ckpt_step32000.pt ;;
  *) echo "unknown size $SIZE"; exit 2 ;;
esac
case $ARM in C|A|M) ;; *) echo "unknown arm $ARM"; exit 2 ;; esac
case $SEED in 1|2) ;; *) echo "seed must be 1 or 2"; exit 2 ;; esac
NVG="$A/nvsynth/full/train/nv_train_0[0-3]?_b*.npz+$A/nvsynth/full/train/nv_train_04[01]_b*.npz"
S2G="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
S3G="$A/synth3/shards/train_v2/synth3_000??.npz"
S3V3D=$A/synth3-v3/shards/train_v3; S3V3G="$S3V3D/synth3_000??.npz"
TV=$A/synth3/vocr3/select_pools/teacher_val.npz       # logged only: the final checkpoint is the run, none is selected
VALS=(--val "rv:$TV" --val "rvn:$TV:native=1" --select rv,rvn)
cd "$REPO"
echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) vocr4 $SIZE $ARM seed $SEED"
grep -qF "  $BASE" eval/vocr4_inputs.sha256 || { echo "$BASE is not pinned in eval/vocr4_inputs.sha256"; exit 4; }
(cd "$A" && sha256sum --quiet -c "$REPO/eval/vocr4_inputs.sha256")
(cd "$P4" && sha256sum --quiet -c "$REPO/eval/vocr4_pools_c.sha256")
INIT=$A/$BASE
FSYN=(--src "s2:0.125:$S2G:stream=1,pool=3,inv=0.2" --src "s3:0.125:$S3G:stream=1,pool=3,inv=0.2")
if [ "$ARM" = M ]; then
  .venv-ci/bin/python - "$S3V3D" <<'PY'
import hashlib, json, sys
from pathlib import Path
m = json.load(open("data/manifests/synth3_shards_v3.json")); d = Path(sys.argv[1])
bad = [s["file"] for s in m["shards"] if hashlib.sha256((d / s["file"]).read_bytes()).hexdigest() != s["sha256"]]
print(f"synth3 v3 shards: {len(m['shards']) - len(bad)} match data/manifests/synth3_shards_v3.json, {len(bad)} differ")
sys.exit(1 if bad else 0)
PY
  PRUN=pre_${SIZE}_M3_seed${SEED}; INIT=$OUT/$PRUN/ckpt_step$PRE.pt
  FSYN=(--src "s2:0.125:$S2G:stream=1,pool=3,inv=0.2" --src "s3v3:0.125:$S3V3G:stream=1,pool=3,inv=0.2")
  if [ ! -s "$INIT" ]; then
    if [ -e "$OUT/$PRUN" ]; then echo "$OUT/$PRUN is partial: delete it and restart (runs never resume)"; exit 3; fi
    .venv-ci/bin/python train/train2.py --run "$PRUN" --out "$OUT" --steps "$PRE" --batch 96 --lr 2e-3 --warmup 1000 \
      --final-lr 0.05 --wd 0.01 --ema 0.999 --width "$WIDTH" --vocab "$VOCAB" --seed "$SEED" --threads 5 \
      --eval-every 4000 --keep-evals --src "nv:0.5:$NVG:stream=1,pool=3,inv=0.2" \
      --src "s2:0.25:$S2G:stream=1,pool=3,inv=0.2" --src "s3v3:0.25:$S3V3G:stream=1,pool=3,inv=0.2" "${VALS[@]}" \
      --val "nvv:$A/nvsynth/shards/nv_val.npz:n=2000" --val "s2v:$A/synth2/shards/synth_00039.npz:n=2000"
  fi
  if [ "${VOCR4_PRETRAIN_ONLY:-0}" = 1 ]; then echo "pretrain $PRUN complete; fine-tune not started"; exit 0; fi
fi
if [ "$ARM" = C ]; then
  REAL=(--src "tt:0.2:$P4/tt.npz:kd=0,inv=0.5" --src "rd:0.1:$P4/rd.npz:kd=0,inv=0.5"
        --src "rp:0.2:$P4/rp2_ink.npz+$P4/rp2_grad.npz:kd=0,inv=0.5")
else
  PA=$REPO/eval/vocr4_pools_a.sha256
  [ -s "$PA" ] || { echo "the A pools are not pinned yet ($PA)"; exit 5; }
  (cd "$P4" && sha256sum --quiet -c "$PA")
  pinned() { grep -q "  $1\$" "$PA"; }
  RP="$P4/rp2_ink.npz+$P4/rp2_grad.npz"                              # fallback: realplus v3i not pinned
  if pinned rp3i_ink.npz && pinned rp3i_grad.npz; then RP="$P4/rp3i_ink.npz+$P4/rp3i_grad.npz"; fi
  REAL=(--src "tt:0.175:$P4/tt.npz:kd=0,inv=0.5" --src "rd:0.1:$P4/rd.npz:kd=0,inv=0.5" --src "rp:0.175:$RP:kd=0,inv=0.5")
  if pinned own_ink.npz && pinned own_grad.npz; then
    REAL+=(--src "nat:0.025:$P4/nat_ink.npz+$P4/nat_grad.npz:kd=0,inv=0.5"
           --src "own:0.025:$P4/own_ink.npz+$P4/own_grad.npz:kd=0,inv=0.5")
  else                                                               # fallback: no owner pool pinned
    REAL+=(--src "nat:0.05:$P4/nat_ink.npz+$P4/nat_grad.npz:kd=0,inv=0.5")
  fi
fi
RUN=v4$(echo "$ARM" | tr 'CAM' 'cam')_${SIZE}_seed${SEED}
if [ -s "$OUT/$RUN/ckpt_step15000.pt" ]; then echo "$RUN is complete"; exit 0; fi
if [ -e "$OUT/$RUN" ]; then echo "$OUT/$RUN is partial: delete it and restart (runs never resume)"; exit 3; fi
echo "run $RUN from $INIT"
exec .venv-ci/bin/python train/train2.py --run "$RUN" --out "$OUT" --init "$INIT" \
  --steps 15000 --batch 96 --lr 1.2e-3 --warmup 500 --final-lr 0.05 --wd 0.01 --ema 0.999 --width "$WIDTH" \
  --vocab "$VOCAB" --seed "$SEED" --threads 5 --eval-every 2500 --keep-evals \
  --src "nv:0.25:$NVG:stream=1,pool=3,inv=0.2" "${FSYN[@]}" "${REAL[@]}" "${VALS[@]}"

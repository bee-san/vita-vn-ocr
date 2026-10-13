#!/usr/bin/env bash
# vocr3_run.sh ARCH ARM SEED [PRE_STEPS]: one VOCR3 candidate run (eval/vocr3_protocol.json), in one heavy.sh slot.
# The study-2 recipe (eval/synth3_study2_run.sh): pretrain from scratch on nvsynth's every-line set (0.5) plus the
# arm's VN-style synthetic data (0.5), then the 15,000-step real fine-tune with hard labels (kd=0) on the
# CLANNAD-free pools (eval/synth3_study2_pools.sha256) from the pretrain's final checkpoint.
#   ARCH: FL10 (width 1.0, FL10's 3,500 vocab, 32k), H15 (width 1.5, data/vocab.txt, 40k),
#         W20 (width 2.0, data/vocab.txt, 32k: train2's p20/F20 size, inside the vita-budget ceiling),
#         LW (--chans 48,96,192,320,384, data/vocab.txt, 32k: the late-wide W20-class candidate of amendment 6)
#   ARM:  A_s2 (synth2), B_s3 (synth3 v2), C_mix (synth2 + synth3 v2)
# A run never resumes: a partial directory stops the script (delete it and restart).
set -euo pipefail
ARCH=$1; ARM=$2; SEED=$3
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$A/synth3/vocr3/runs
case $ARCH in
  FL10) WIDTH=1.0; VOCAB=$A/train2/release/FL10_vocab_3500.txt; DEF=32000 ;;
  H15)  WIDTH=1.5; VOCAB=$REPO/data/vocab.txt; DEF=40000 ;;
  W20)  WIDTH=2.0; VOCAB=$REPO/data/vocab.txt; DEF=32000 ;;
  LW)   WIDTH=1.5; CHANS=48,96,192,320,384; VOCAB=$REPO/data/vocab.txt; DEF=32000 ;;   # amendment 6: late-wide, train2 --chans
  *) echo "unknown arch $ARCH"; exit 2 ;;
esac
PRE=${4:-$DEF}
CH=(); [ -n "${CHANS:-}" ] && CH=(--chans "$CHANS")   # explicit per-stage channels (train/train2.py --chans, PR #58)
NVG="$A/nvsynth/full/train/nv_train_0[0-3]?_b*.npz+$A/nvsynth/full/train/nv_train_04[01]_b*.npz"
S2G="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
S3G="$A/synth3/shards/train_v2/synth3_000??.npz"
case $ARM in
  A_s2)  PSYN=(--src "s2:0.5:$S2G:stream=1,pool=3,inv=0.2")
         FSYN=(--src "s2:0.25:$S2G:stream=1,pool=3,inv=0.2") ;;
  B_s3)  PSYN=(--src "s3:0.5:$S3G:stream=1,pool=3,inv=0.2")
         FSYN=(--src "s3:0.25:$S3G:stream=1,pool=3,inv=0.2") ;;
  C_mix) PSYN=(--src "s2:0.25:$S2G:stream=1,pool=3,inv=0.2" --src "s3:0.25:$S3G:stream=1,pool=3,inv=0.2")
         FSYN=(--src "s2:0.125:$S2G:stream=1,pool=3,inv=0.2" --src "s3:0.125:$S3G:stream=1,pool=3,inv=0.2") ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
TAG=""; [ "$PRE" != "$DEF" ] && TAG="_pre$((PRE / 1000))k"
PRUN=pre_${ARCH}_${ARM}${TAG}_seed${SEED}
FRUN=ft_${ARCH}_${ARM}${TAG}_seed${SEED}
VALS=(--val "rv:$A/teacher/teacher_val.npz" --val "rvn:$A/teacher/teacher_val.npz:native=1" --select rv,rvn)
FP=$A/synth3/study2/pools
cd "$REPO"
echo "git $(git rev-parse HEAD) runs $PRUN -> $FRUN"
(cd "$FP" && sha256sum --quiet -c "$REPO/eval/synth3_study2_pools.sha256")
if [ -s "$OUT/$FRUN/ckpt_step15000.pt" ]; then echo "$FRUN is complete"; exit 0; fi
if [ -e "$OUT/$FRUN" ]; then echo "$OUT/$FRUN is partial: delete it and restart (runs never resume)"; exit 3; fi
if [ ! -s "$OUT/$PRUN/ckpt_step$PRE.pt" ]; then
  if [ -e "$OUT/$PRUN" ]; then echo "$OUT/$PRUN is partial: delete it and restart (runs never resume)"; exit 3; fi
  .venv-ci/bin/python train/train2.py --run "$PRUN" --out "$OUT" --steps "$PRE" --batch 96 --lr 2e-3 --warmup 1000 \
    --final-lr 0.05 --wd 0.01 --ema 0.999 --width "$WIDTH" "${CH[@]}" --vocab "$VOCAB" --seed "$SEED" --threads 5 \
    --eval-every 4000 --keep-evals --src "nv:0.5:$NVG:stream=1,pool=3,inv=0.2" "${PSYN[@]}" "${VALS[@]}" \
    --val "nvv:$A/nvsynth/shards/nv_val.npz:n=2000" --val "s2v:$A/synth2/shards/synth_00039.npz:n=2000"
fi
exec .venv-ci/bin/python train/train2.py --run "$FRUN" --out "$OUT" --init "$OUT/$PRUN/ckpt_step$PRE.pt" \
  --steps 15000 --batch 96 --lr 1.2e-3 --warmup 500 --final-lr 0.05 --wd 0.01 --ema 0.999 --width "$WIDTH" "${CH[@]}" \
  --vocab "$VOCAB" --seed "$SEED" --threads 5 --eval-every 2500 --keep-evals \
  --src "nv:0.25:$NVG:stream=1,pool=3,inv=0.2" "${FSYN[@]}" \
  --src "tt:0.275:$FP/teacher_train.npz:targets=$FP/targets,kd=0,inv=0.5" \
  --src "rd:0.125:$FP/real_train.npz:targets=$FP/targets,kd=0,inv=0.5" \
  --src "ps:0.1:$FP/teacher_distill_train.npz+$FP/real_unlabelled.npz:targets=$FP/targets,label=teacher,conf=0.7,kd=0,inv=0.5" \
  "${VALS[@]}"

#!/usr/bin/env bash
# vocr3_rp_run.sh ARCH ARM SEED [PRE_STEPS]: the realplus fine-tune of one VOCR3 candidate's pretrain
# (eval/vocr3_protocol.json amendment 4). It is the candidate's own fine-tune (eval/synth3_study2_run.sh /
# eval/vocr3_run.sh: 15,000 steps, lr 1.2e-3, kd=0, nv 0.25 + the arm's synthetic data 0.25) with the real half
# re-split as tt 0.2 + rd 0.1 + rp 0.2: tt and rd are the CLANNAD-free copies (eval/synth3_study2_pools.sha256),
# rp is the realplus pool, ink and GRAD crops (eval/vocr3_realplus.sha256), and there are no PP-OCRv6 pseudo-labels.
# The pretrain is read from study 2's runs or VOCR3's own; the output is $HERD/shared/artifacts/synth3/vocr3/runs/
# rp_<ARCH>_<ARM>[_preNk]_seed<SEED>. A run never resumes: a partial directory stops the script.
set -euo pipefail
ARCH=$1; ARM=$2; SEED=$3
A=${HERD:-/local/home/skerraut/herd-vita-ocr}/shared/artifacts
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$A/synth3/vocr3/runs
case $ARCH in
  FL10) WIDTH=1.0; VOCAB=$A/train2/release/FL10_vocab_3500.txt; DEF=32000 ;;
  H15)  WIDTH=1.5; VOCAB=$REPO/data/vocab.txt; DEF=40000 ;;
  W20)  WIDTH=2.0; VOCAB=$REPO/data/vocab.txt; DEF=32000 ;;
  *) echo "unknown arch $ARCH"; exit 2 ;;
esac
PRE=${4:-$DEF}
NVG="$A/nvsynth/full/train/nv_train_0[0-3]?_b*.npz+$A/nvsynth/full/train/nv_train_04[01]_b*.npz"
S2G="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
S3G="$A/synth3/shards/train_v2/synth3_000??.npz"
case $ARM in
  A_s2)  FSYN=(--src "s2:0.25:$S2G:stream=1,pool=3,inv=0.2") ;;
  B_s3)  FSYN=(--src "s3:0.25:$S3G:stream=1,pool=3,inv=0.2") ;;
  C_mix) FSYN=(--src "s2:0.125:$S2G:stream=1,pool=3,inv=0.2" --src "s3:0.125:$S3G:stream=1,pool=3,inv=0.2") ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
TAG=""; [ "$PRE" != "$DEF" ] && TAG="_pre$((PRE / 1000))k"
PRUN=pre_${ARCH}_${ARM}${TAG}_seed${SEED}
RUN=rp_${ARCH}_${ARM}${TAG}_seed${SEED}
INIT=""
for d in "$A/synth3/study2/runs" "$A/synth3/vocr3/runs"; do
  [ -s "$d/$PRUN/ckpt_step$PRE.pt" ] && INIT="$d/$PRUN/ckpt_step$PRE.pt" && break
done
[ -n "$INIT" ] || { echo "no finished pretrain $PRUN (ckpt_step$PRE.pt)"; exit 4; }
FP=$A/synth3/study2/pools
RP=$A/realplus
VALS=(--val "rv:$A/teacher/teacher_val.npz" --val "rvn:$A/teacher/teacher_val.npz:native=1" --select rv,rvn)
cd "$REPO"
echo "git $(git rev-parse HEAD) run $RUN from $INIT"
(cd "$FP" && sha256sum --quiet -c "$REPO/eval/synth3_study2_pools.sha256")
(cd "$RP" && sha256sum --quiet -c "$REPO/eval/vocr3_realplus.sha256")
if [ -s "$OUT/$RUN/ckpt_step15000.pt" ]; then echo "$RUN is complete"; exit 0; fi
if [ -e "$OUT/$RUN" ]; then echo "$OUT/$RUN is partial: delete it and restart (runs never resume)"; exit 3; fi
exec .venv-ci/bin/python train/train2.py --run "$RUN" --out "$OUT" --init "$INIT" \
  --steps 15000 --batch 96 --lr 1.2e-3 --warmup 500 --final-lr 0.05 --wd 0.01 --ema 0.999 --width "$WIDTH" \
  --vocab "$VOCAB" --seed "$SEED" --threads 5 --eval-every 2500 --keep-evals \
  --src "nv:0.25:$NVG:stream=1,pool=3,inv=0.2" "${FSYN[@]}" \
  --src "tt:0.2:$FP/teacher_train.npz:targets=$FP/targets,kd=0,inv=0.5" \
  --src "rd:0.1:$FP/real_train.npz:targets=$FP/targets,kd=0,inv=0.5" \
  --src "rp:0.2:$RP/realplus_train_ink.npz+$RP/realplus_train_grad.npz:kd=0,inv=0.5" \
  "${VALS[@]}"

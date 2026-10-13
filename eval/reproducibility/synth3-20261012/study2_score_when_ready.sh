#!/usr/bin/env bash
# score_when_ready.sh ARCH: wait for every study-2 fine-tune of ARCH, record devnative's status, then score (heavy.sh)
# and compute the protocol result (eval/synth3_study2.py) into $S3/study2/ARCH_result.json.
set -uo pipefail
ARCH=$1
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; S3=$A/synth3; R=$S3/study2/runs; SC=$S3/study2/scores
REPO=/local/home/skerraut/herd-vita-oct12/clones/runner-c464aa8
case $ARCH in FL10) SEEDS="1 2 3" ;; H15) SEEDS="1 2" ;; esac
for arm in A_s2 C_mix; do for s in $SEEDS; do
  until [ -s "$R/ft_${ARCH}_${arm}_seed$s/ckpt_step15000.pt" ]; do sleep 60; done
done; done
sleep 120   # let the last fine-tune write its final files and exit
STATUS=$(cat /local/home/skerraut/herd-synth3/devnative.status)
echo "$(date -u +%FT%TZ) all $ARCH fine-tunes done; devnative.status: $STATUS"
echo "$STATUS" > $SC/${ARCH}_devnative_status_at_scoring.txt
sha256sum $A/devnative/devnative_*.npz > $SC/${ARCH}_devnative_npz.sha256
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 14400 bash $S3/study2/score_arch.sh $ARCH < /dev/null
DN=(); for f in strict_ink strict_grad tv_native_ink tv_native_grad; do [ -s $SC/${ARCH}_devnative_$f.json ] && DN+=($SC/${ARCH}_devnative_$f.json); done
cd "$REPO"
SHA=$(echo "$STATUS" | awk '/COHORT_FROZEN/ {print $2}')
if [ ${#DN[@]} -gt 0 ] && [ -n "$SHA" ]; then
  .venv-ci/bin/python eval/synth3_study2.py merge --scores "${DN[@]}" --out $SC/${ARCH}_devnative_decision_set.json
  DNARGS=(--dn $SC/${ARCH}_devnative_decision_set.json --dn-commit "$SHA")
else
  DNARGS=()
fi
.venv-ci/bin/python eval/synth3_study2.py result --protocol eval/synth3_study2_protocol.json --tv $SC/${ARCH}_teacher_val.json \
  "${DNARGS[@]}" --reported eval/synth3_study2_reported.json --reported-scores $SC/${ARCH}_teacher_val.json \
  --out $S3/study2/${ARCH}_result.json
echo "$(date -u +%FT%TZ) RESULT_WRITTEN $S3/study2/${ARCH}_result.json"

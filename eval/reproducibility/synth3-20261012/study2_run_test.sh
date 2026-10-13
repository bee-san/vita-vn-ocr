#!/usr/bin/env bash
# The study-1 protocol's locked vita-test/psp-test scoring ("protected_test"): each export once per polarity.
# Usage: run_test.sh NAME=BLOB ...   (writes $S3/ablation/test/NAME_<pol>.json; skips existing outputs)
set -euo pipefail
HERD=/local/home/skerraut/herd-vita-ocr; S3=$HERD/shared/artifacts/synth3; T=$S3/study2/test
cd /local/home/skerraut/herd-vita-oct12/clones/synth-dataset
export VOCR_EVAL_DATA=$HERD/shared/artifacts/eval
echo "git $(git rev-parse HEAD)"
for spec in "$@"; do
  name=${spec%%=*}; blob=${spec#*=}
  for pol in console native; do
    out=$T/${name}_${pol}.json
    [ -s "$out" ] && { echo "skip $out (exists: scored once already)"; continue; }
    echo "== $name $pol $(sha256sum "$blob" | cut -c1-16)"
    .venv-ci/bin/python eval/run.py --model "vocr:$blob" --set vita-test,psp-test --polarity $pol --out "$out" 2>&1 | grep -E 'CER|cer|exact|lines' | head -8
  done
done
echo "TEST_SCORING_FINISHED"

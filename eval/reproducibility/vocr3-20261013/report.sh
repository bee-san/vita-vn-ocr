#!/usr/bin/env bash
# vocr3_report.sh STEP: the VOCR3 locked report (eval/vocr3_protocol.json "locked_report"), run ONCE after the three
# selections are committed on feat/vocr3. STEP = spec | shared | a9 | overlay | render | all.
set -euo pipefail
STEP=${1:-all}
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; S3=$A/synth3; export HERD
REPO=/local/home/skerraut/herd-vita-oct12/clones/vocr3-wt; PY=$REPO/.venv-ci/bin/python
R=$S3/vocr3/report; mkdir -p $R/blobs $R/out
export VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz
VITASDK=$A/vita/vitasdk; QEMU=$A/vita/qemu
ARMCC=$A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
OVERLAY=$A/vocr3-harness/overlay; CMAKE=$A/vita-budget/tools-venv/bin/cmake; NINJA=$HOME/.local/bin/ninja
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) step $STEP"
if [ "$STEP" = all ] || [ "$STEP" = spec ]; then
  for s in FL10 H15 W20; do [ -s eval/vocr3_selection_${s}_20261013.json ] || { echo "missing selection $s"; exit 2; }; done
  git status --porcelain -- eval/vocr3_selection_*_20261013.json | grep -q . && { echo "selection files not committed"; exit 2; }
  $PY eval/vocr3_make_spec.py --protocol eval/vocr3_protocol.json \
    --selection FL10=eval/vocr3_selection_FL10_20261013.json --selection H15=eval/vocr3_selection_H15_20261013.json \
    --selection W20=eval/vocr3_selection_W20_20261013.json --blob-dir $R/blobs --release-dir $R/release --out $R/spec.json
  $PY -c "import json; d=json.load(open('$R/spec.json')); print([a['name'] for a in d['arms']]); print(len(d['pairs']), 'pairs', len(d['gates']), 'gates')"
  (cd $R/blobs && sha256sum *.vocr) | tee $R/blobs.sha256
fi
if [ "$STEP" = all ] || [ "$STEP" = shared ]; then
  $PY eval/vocr3_report.py shared --spec $R/spec.json --blobs $R/blobs --data $A/eval --out $R/out 2>&1 | tail -n 30
fi
if [ "$STEP" = all ] || [ "$STEP" = a9 ]; then
  $PY eval/vocr3_report.py a9 --spec $R/spec.json --blobs $R/blobs --out $R/out --data $A/eval --vitasdk $VITASDK --qemu $QEMU --arm-cc $ARMCC 2>&1 | tail -n 12
fi
if [ "$STEP" = all ] || [ "$STEP" = overlay ]; then
  $PY eval/vocr3_report.py overlay --spec $R/spec.json --blobs $R/blobs --out $R/out --data $A/eval --overlay $OVERLAY --cmake $CMAKE --ninja $NINJA --extend-table 2>&1 | tail -n 12
fi
if [ "$STEP" = all ] || [ "$STEP" = render ]; then
  $PY eval/vocr3_report.py render --out $R/out 2>&1 | tail -n 5
  ls -la $R/out/report.json $R/out/report_summary.json $R/out/report.md
fi
echo "REPORT_STEP_FINISHED $STEP $(date -u +%FT%TZ)"

#!/usr/bin/env bash
# report.sh STEP: the VOCR4 locked report (eval/vocr4_protocol.json "locked_report"), run ONCE after the three
# selections are committed on feat/vocr4. STEP = spec | shared | a9 | overlay | render | clannad.
set -euo pipefail
STEP=$1
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; S3=$A/synth3; export HERD
REPO=${REPO:-/home/skerraut/herd-synth3/clones/vocr4-train}; PY=$REPO/.venv-ci/bin/python
V4=$A/vocr4-train; R=$V4/report; mkdir -p "$R/blobs" "$R/out"
export VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz
VITASDK=$A/vita/vitasdk; QEMU=$A/vita/qemu
ARMCC=$A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
OVERLAY=$A/vocr3-harness/overlay; CMAKE=$A/vita-budget/tools-venv/bin/cmake; NINJA=$HOME/.local/bin/ninja
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) step $STEP"
SELS=(); for s in FL10 H15 W20; do SELS+=(--selection "$s=eval/vocr4_selection_${s}_20261014.json"); done
case $STEP in
spec)
  for s in FL10 H15 W20; do
    f=eval/vocr4_selection_${s}_20261014.json
    [ -s "$f" ] || { echo "missing $f"; exit 2; }
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "$f is not committed"; exit 2; }
  done
  git status --porcelain -- eval/vocr4_selection_*.json | grep -q . && { echo "selection files not committed"; exit 2; }
  $PY eval/vocr4_make_spec.py --protocol eval/vocr4_protocol.json "${SELS[@]}" \
    --vocr3-spec eval/vocr3_report_spec_20261013.json --vocr3-blobs "$S3/vocr3/report/blobs" \
    --release-dir "$S3/vocr3/report/release" --blob-dir "$R/blobs" --out "$R/spec.json"
  (cd "$R/blobs" && sha256sum ./*.vocr) | tee "$R/blobs.sha256" ;;
shared)
  $PY eval/vocr3_report.py shared --spec "$R/spec.json" --blobs "$R/blobs" --data "$A/eval" --out "$R/out" 2>&1 | tail -n 30 ;;
a9)
  $PY eval/vocr3_report.py a9 --spec "$R/spec.json" --blobs "$R/blobs" --out "$R/out" --data "$A/eval" \
    --vitasdk "$VITASDK" --qemu "$QEMU" --arm-cc "$ARMCC" 2>&1 | tail -n 14 ;;
overlay)
  $PY eval/vocr3_report.py overlay --spec "$R/spec.json" --blobs "$R/blobs" --out "$R/out" --data "$A/eval" \
    --overlay "$OVERLAY" --cmake "$CMAKE" --ninja "$NINJA" --extend-table 2>&1 | tail -n 14 ;;
render)
  $PY eval/vocr3_report.py render --out "$R/out" 2>&1 | tail -n 5
  ls -la "$R/out/report.json" "$R/out/report_summary.json" "$R/out/report.md" ;;
clannad)
  # owner-drive-ingest's extra-arms path (read-only on its WORK; refuses before its references are frozen)
  CR=${CLANNAD_REPO:-$REPO}                       # main has eval/clannad_refs.py since #73 (1c6a350)
  WORK=$HERD/shared/private/owner-drive-20261013/clannad-compare; C=$V4/clannad; mkdir -p "$C"
  EX=(); for s in FL10 H15 W20; do EX+=(--extra "VOCR4_$s=VOCR4_${s}_w8.vocr#$(sha256sum "$R/blobs/VOCR4_${s}_w8.vocr" | cut -c1-64)"); done
  (cd "$CR" && .venv-ci/bin/python eval/clannad_refs.py arms --work "$WORK" --blobs "$R/blobs" "${EX[@]}" --suffix vocr4 --into "$C")
  (cd "$CR" && .venv-ci/bin/python eval/clannad_refs.py score --work "$WORK" --suffix vocr4 --into "$C" \
     --pair VOCR4_FL10:VOCR3_FL10 --pair VOCR4_H15:VOCR3_H15 --pair VOCR4_W20:VOCR3_W20 --pair VOCR4_FL10:VOCR1 \
     --pair VOCR4_H15:VOCR_H15 --pair VOCR4_W20:VOCR_H15 --pair VOCR4_FL10:Lens --pair VOCR4_H15:Lens \
     --pair VOCR4_W20:Lens --out "$C/clannad_vocr4.json") ;;
*) echo "unknown step $STEP"; exit 2 ;;
esac
echo "REPORT_STEP_FINISHED $STEP $(date -u +%FT%TZ)"

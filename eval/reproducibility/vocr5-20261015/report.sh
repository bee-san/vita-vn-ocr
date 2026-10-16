#!/usr/bin/env bash
# report.sh STEP: the VOCR5 locked report (eval/vocr5_protocol.json "locked_report"), run ONCE after the three
# selections are committed on feat/vocr5. A copy of VOCR4's report.sh. STEP = spec | shared | a9 | overlay | render |
# lm | lens | dialogue | clannad.
set -euo pipefail
STEP=$1
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; export HERD
REPO=${REPO:-/local/home/skerraut/herd-synth3/clones/vocr5-train}; PY=$REPO/.venv-ci/bin/python
V5=$A/vocr5-train; R=$V5/report; mkdir -p "$R/blobs" "$R/out"
export VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz
VITASDK=$A/vita/vitasdk; QEMU=$A/vita/qemu
ARMCC=$A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
OVERLAY=$A/vocr3-harness/overlay; CMAKE=$A/vita-budget/tools-venv/bin/cmake; NINJA=$HOME/.local/bin/ninja
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) step $STEP"
SELS=(); for t in Small Normal Large; do SELS+=(--selection "$t=eval/vocr5_selection_${t}_20261016.json"); done
case $STEP in
spec)
  for t in Small Normal Large; do
    f=eval/vocr5_selection_${t}_20261016.json
    [ -s "$f" ] || { echo "missing $f"; exit 2; }
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "$f is not committed"; exit 2; }
  done
  git status --porcelain -- eval/vocr5_selection_*.json | grep -q . && { echo "selection files not committed"; exit 2; }
  $PY eval/vocr5_make_spec.py --protocol eval/vocr5_protocol.json "${SELS[@]}" \
    --vocr4-spec eval/vocr4_report_spec_20261014.json --blobs "$A/vocr4-train/report/blobs" \
    --blob-dir "$R/blobs" --out "$R/spec.json"
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
lm)
  # report only: the deployed decode (beam + the overlay's default 1 MB LM) on the fixed lines, VOCR4 and VOCR5
  # vocr_lm_o3_1000k.vlm of v0.2.6-ctclm.2 (sha256 99300061..., the overlay's pinned 1 MB tier) and its
  # docs/ctc-lm-decode.json "tiers" 1MB config: beam 8, top-8, alpha 500, beta 3500, gate 8000, flags 2, bracket 8000
  LM=$A/ctc-lm-decode/release_pub2/vocr_lm_o3_1000k.vlm; mkdir -p "$R/lm"
  [ "$(sha256sum "$LM" | cut -c1-16)" = 993000611555171d ] || { echo "$LM is not the pinned 1 MB LM"; exit 4; }
  OPTS="--stdin --topk 8 --beam 8 --alpha 500 --beta 3500 --gate 8000 --lm-flags 2 --bracket 8000 --lm $LM"
  make -C runtime build/vocr_cli >/dev/null
  for arm in VOCR4_FL10 VOCR4_H15 VOCR4_W20 VOCR5_S VOCR5_N VOCR5_L; do
    for s in vita-test psp-test; do
      $PY eval/run.py --model "cmd:runtime/build/vocr_cli $R/blobs/${arm}_w8.vocr $OPTS" \
        --set "$s" --data "$A/eval" --polarity console --out "$R/lm/${arm}_${s}.json" 2>&1 | tail -n 2
    done
  done ;;
lens)
  # protocol amendment 2 (report only): Google Lens reads each scored line of vita-test and psp-test once (public VNDB
  # screenshots), screen polarity; readings cached privately; scored by eval/run.py like every arm (native polarity)
  L=$HERD/shared/private/vocr5-train/lens; mkdir -p "$L" "$R/lens"
  [ -x .venv-owocr/bin/python ] || { echo "no .venv-owocr (eval/lens_lines.py's install steps)"; exit 2; }
  for s in vita-test psp-test; do
    $PY eval/lens_lines.py read --set "$s" --data "$A/eval" --owocr-python .venv-owocr/bin/python --out "$L/$s.jsonl"
    $PY eval/run.py --set "$s" --data "$A/eval" --polarity native \
      --model "cmd:$PY eval/lens_lines.py serve --cache $L/$s.jsonl" --out "$L/run_$s.json" 2>&1 | tail -n 3
  done ;;
dialogue)
  # the dialogue-primary fixed-line tables and pairs (lines without the 'name' style), Lens included
  # (run after the lm step: the deployed decode's arms enter as --extra console:<arm>+LM:<set>)
  LA=(); L=$HERD/shared/private/vocr5-train/lens
  for s in vita-test psp-test; do [ -s "$L/run_$s.json" ] && LA+=(--lens "$s=$L/run_$s.json"); done
  for arm in VOCR4_FL10 VOCR4_H15 VOCR4_W20 VOCR5_S VOCR5_N VOCR5_L; do
    for s in vita-test psp-test; do
      [ -s "$R/lm/${arm}_$s.json" ] && LA+=(--extra "console:${arm}+LM:$s=$R/lm/${arm}_$s.json")
    done
  done
  $PY eval/vocr5_dialogue.py tables --report "$R/out/report.json" "${LA[@]}" --out "$R/out/dialogue.json" \
    --md "$R/out/dialogue.md" | head -n 40 ;;
clannad)
  # owner-drive-ingest's extra-arms path (read-only on its WORK; one attempt per suffix)
  WORK=$HERD/shared/private/owner-drive-20261013/clannad-compare; C=$HERD/shared/private/vocr5-train/clannad; mkdir -p "$C"
  # VOCR4's three are read again beside VOCR5's (the runtime is deterministic), so every pair's baseline is in this
  # attempt's arms file; VOCR1, VOCR_H15 and Lens come from the references' own arms
  EX=(); for a in VOCR4_FL10 VOCR4_H15 VOCR4_W20 VOCR5_S VOCR5_N VOCR5_L; do
    EX+=(--extra "$a=${a}_w8.vocr#$(sha256sum "$R/blobs/${a}_w8.vocr" | cut -c1-64)"); done
  $PY eval/clannad_refs.py arms --work "$WORK" --blobs "$R/blobs" "${EX[@]}" --suffix vocr5 --into "$C"
  $PY eval/clannad_refs.py score --work "$WORK" --suffix vocr5 --into "$C" \
     --pair VOCR5_S:VOCR4_FL10 --pair VOCR5_N:VOCR4_H15 --pair VOCR5_L:VOCR4_W20 --pair VOCR5_N:VOCR4_W20 \
     --pair VOCR5_L:VOCR5_N --pair VOCR5_S:Lens --pair VOCR5_N:Lens --pair VOCR5_L:Lens --out "$C/clannad_vocr5.json" ;;
*) echo "unknown step $STEP"; exit 2 ;;
esac
echo "REPORT_STEP_FINISHED $STEP $(date -u +%FT%TZ)"

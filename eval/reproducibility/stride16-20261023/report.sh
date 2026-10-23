#!/usr/bin/env bash
# report.sh STEP: the stride16 locked report (eval/stride16_protocol.json "locked_report"), run ONCE after the three
# selections are committed on feat/stride16. A copy of VOCR5's report.sh (eval/reproducibility/vocr5-20261015) with
# stride16's arms. STEP = spec | shared | render | lm | dialogue | floor | clannad.
#   spec      eval/stride16_make_spec.py from the committed selections: VOCR5_S / N / L and every arm's best run
#   shared    eval/vocr3_report.py shared: vita-test + psp-test fixed lines (console and native polarity) and whole
#             dialogue (each region at base, +4 and +8 px, the arm's C GRAD finder), the scalar C W8I runtime
#   render    report.json / report_summary.json / report.md
#   lm        report only: the deployed decode (beam 8 + the overlay's 1 MB LM at its tier settings) on the fixed lines
#   dialogue  the dialogue-primary tables and pairs (eval/vocr5_dialogue.py tables, stride16's arms and pairs)
#   floor     report only: the protected fixed lines' frame budget strata (eval/stride16_floor.py protected)
#   clannad   owner-drive-ingest's frozen CLANNAD references, the arms read once through clannad_refs' extra-arms path
set -euo pipefail
STEP=$1
HERD=/local/home/skerraut/herd-vita-ocr; A=$HERD/shared/artifacts; export HERD
REPO=${REPO:-/local/home/skerraut/herd-synth3/clones/stride16}; PY=$REPO/.venv-ci/bin/python
S16=$A/stride16; R=$S16/report; mkdir -p "$R/blobs" "$R/out"
cd "$REPO"; echo "git $(git rev-parse HEAD) $(date -u +%FT%TZ) step $STEP"
ARMS=(VOCR5_S VOCR5_N VOCR5_L S16 N16 N12 L16)
SELS=(); for t in Small Normal Large; do SELS+=(--selection "$t=eval/stride16_selection_${t}_20261024.json"); done
case $STEP in
spec)
  for t in Small Normal Large; do
    f=eval/stride16_selection_${t}_20261024.json
    [ -s "$f" ] || { echo "missing $f"; exit 2; }
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "$f is not committed"; exit 2; }
  done
  git status --porcelain -- eval/stride16_selection_*.json | grep -q . && { echo "selection files not committed"; exit 2; }
  [ -e "$R/out/shared.json" ] && { echo "the locked report has run already: $R/out/shared.json"; exit 2; }
  $PY eval/stride16_make_spec.py --protocol eval/stride16_protocol.json "${SELS[@]}" \
    --vocr5-spec eval/vocr5_report_spec_20261016.json --blobs "$A/vocr5-train/report/blobs" \
    --blob-dir "$R/blobs" --out "$R/spec.json"
  (cd "$R/blobs" && sha256sum ./*.vocr) | tee "$R/blobs.sha256" ;;
shared)
  [ -e "$R/out/shared.json" ] && { echo "the locked report has run already"; exit 2; }
  $PY eval/vocr3_report.py shared --spec "$R/spec.json" --blobs "$R/blobs" --data "$A/eval" --out "$R/out" 2>&1 | tail -n 30 ;;
render)
  $PY eval/vocr3_report.py render --out "$R/out" 2>&1 | tail -n 5
  ls -la "$R/out/report.json" "$R/out/report_summary.json" "$R/out/report.md" ;;
lm)
  # v0.2.6-ctclm.2's 1 MB tier (sha256 99300061...) at docs/ctc-lm-decode.json's 1MB settings, as VOCR5's report
  LM=$A/ctc-lm-decode/release_pub2/vocr_lm_o3_1000k.vlm; mkdir -p "$R/lm"
  [ "$(sha256sum "$LM" | cut -c1-16)" = 993000611555171d ] || { echo "$LM is not the pinned 1 MB LM"; exit 4; }
  OPTS="--stdin --topk 8 --beam 8 --alpha 500 --beta 3500 --gate 8000 --lm-flags 2 --bracket 8000 --lm $LM"
  make -C runtime build/vocr_cli >/dev/null
  for arm in "${ARMS[@]}"; do
    for s in vita-test psp-test; do
      [ -s "$R/lm/${arm}_${s}.json" ] && continue
      $PY eval/run.py --model "cmd:runtime/build/vocr_cli $R/blobs/${arm}_w8.vocr $OPTS" \
        --set "$s" --data "$A/eval" --polarity console --out "$R/lm/${arm}_${s}.json" 2>&1 | tail -n 2
    done
  done ;;
dialogue)
  $PY - "$R/spec.json" "$R/out/dialogue_pairs.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
json.dump({"locked_report": {"arms": [a["name"] for a in spec["arms"]],
                             "pairs": [[p["candidate"], p["baseline"]] for p in spec["pairs"]]}},
          open(sys.argv[2], "w"), indent=1)
PY
  LA=()
  for arm in "${ARMS[@]}"; do
    for s in vita-test psp-test; do
      [ -s "$R/lm/${arm}_$s.json" ] && LA+=(--extra "console:${arm}+LM:$s=$R/lm/${arm}_$s.json")
    done
  done
  $PY eval/vocr5_dialogue.py tables --protocol "$R/out/dialogue_pairs.json" --report "$R/out/report.json" "${LA[@]}" \
    --out "$R/out/dialogue.json" --md "$R/out/dialogue.md" > "$R/out/dialogue.stdout"
  head -n 60 "$R/out/dialogue.stdout" ;;
floor)
  $PY eval/stride16_floor.py protected --report "$R/out/report.json" --data "$A/eval" --out "$R/out/floor_protected.json" ;;
clannad)
  WORK=$HERD/shared/private/owner-drive-20261013/clannad-compare; C=$HERD/shared/private/stride16/clannad; mkdir -p "$C"
  EX=(); for a in "${ARMS[@]}"; do
    EX+=(--extra "$a=${a}_w8.vocr#$(sha256sum "$R/blobs/${a}_w8.vocr" | cut -c1-64)"); done
  $PY eval/clannad_refs.py arms --work "$WORK" --blobs "$R/blobs" "${EX[@]}" --suffix stride16 --into "$C"
  $PY eval/clannad_refs.py score --work "$WORK" --suffix stride16 --into "$C" \
     --pair S16:VOCR5_S --pair N16:VOCR5_N --pair N12:VOCR5_N --pair L16:VOCR5_L --pair N16:N12 \
     --out "$C/clannad_stride16.json" ;;
*) echo "unknown step $STEP"; exit 2 ;;
esac
echo "REPORT_STEP_FINISHED $STEP $(date -u +%FT%TZ)"

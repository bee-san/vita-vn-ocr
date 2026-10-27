#!/usr/bin/env bash
# a9.sh BLOBS OUT: vita/a9_layers.py measure + report of the attention op's cost (docs/vocr6-attn.md "Time"): per tier
# the adopted speed blob (S: head gate; N, L: gate + two-level head) and the same with the op at gate 0 (BLOBS from
# eval/vocr6_attn_cost.py blobs), on ocr-speed's 24 development lines of 60-100 frames (sub24.bin, sha256 3162151a...),
# --topk 8 and the tier's options, with the PLD model (pld) and without it (nopld, the safe basis the tiers are sized
# on). Instruction counts under qemu-arm of the VitaSDK -O2 build; every millisecond derived from them is an estimate.
# Run inside ONE heavy.sh slot: up to 4 qemu processes at a time (one thread each). A finished directory is skipped.
set -u
BLOBS=${1:?usage: a9.sh BLOBS OUT}; OUT=${2:?usage: a9.sh BLOBS OUT}
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
. "$HERD/shared/artifacts/ocr-speed/env.sh"          # VITASDK, ARM_CC, VOCR_A9_PLUGINS, A (ocr-speed's artifacts)
export PYTHONDONTWRITEBYTECODE=1
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PACK=$HERD/shared/artifacts/ocr-speed-resume/packs/sub24.bin
export REPO BLOBS OUT PACK
one() {
  IFS=: read -r basis tier name cli <<< "$1"
  extra=""; [ "$basis" = nopld ] && extra="--no-pld"
  out=$OUT/${basis}_${tier}_${name}
  [ -f "$out/report.json" ] && { echo "skip $out"; return 0; }
  mkdir -p "$out"; start=$(date +%s)
  "$REPO/.venv-ci/bin/python" "$REPO/vita/a9_layers.py" measure --blob "$BLOBS/${tier}_${name}.vocr" --pack "$PACK" \
    --out "$out" --mode w8i --min-frames 60 --max-frames 100 --limit 24 --cli-opts "--topk 8 $cli" $extra \
    > "$out/measure.log" 2>&1 || { echo "FAIL measure $basis $tier $name"; return 1; }
  "$REPO/.venv-ci/bin/python" "$REPO/vita/a9_layers.py" report "$out" --json "$out/report.json" --scale 1.000676 \
    > "$out/layers.md" 2>&1 || { echo "FAIL report $basis $tier $name"; return 1; }
  echo "done $basis $tier $name in $(( $(date +%s) - start ))s"
}
export -f one
jobs=()
for basis in nopld pld; do
  for name in ref attn; do
    jobs+=("$basis:S:$name:--head-gate on" "$basis:N:$name:--two-level --head-gate on"
           "$basis:L:$name:--two-level --head-gate on")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 4 -I{} bash -c 'one "$@"' _ {}
echo A9_RUN_FINISHED

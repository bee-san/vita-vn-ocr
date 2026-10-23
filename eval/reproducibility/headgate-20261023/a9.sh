#!/usr/bin/env bash
# a9.sh TIER CONFIG PLD OUTDIR: one vita/a9_layers.py measurement of a gated VOCR5 blob (eval/headgate_protocol.json
# "estimates"): ocr-speed's 24 development lines of 60-100 frames (pack sha256 3162151a...), --topk 8, the head gate
# off | on | score, with the PLD model (PLD=1) or without (PLD=0, --no-pld: the safe basis). Run through heavy.sh.
# Instruction counts under qemu-arm of the VitaSDK -O2 build; every millisecond derived from them is an estimate.
set -euo pipefail
tier=$1; config=$2; pld=$3; out=$4
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
. "$HERD/shared/artifacts/ocr-speed/env.sh"          # VITASDK, ARM_CC, VOCR_A9_PLUGINS, A (ocr-speed's artifacts)
repo=$(cd "$(dirname "$0")/../../.." && pwd)
blob=$HERD/shared/artifacts/head-gate/blobs/VOCR5_${tier}_w8g.vocr
extra=()
[ "$pld" = 0 ] && extra=(--no-pld)
dir=$out/${tier}_${config}_pld${pld}
[ -f "$dir/layers.txt" ] && exit 0
exec "$repo/.venv-ci/bin/python" "$repo/vita/a9_layers.py" measure --blob "$blob" --pack "$A/packs/dev_res.bin" \
  --out "$dir" --mode w8i --min-frames 60 --max-frames 100 --limit 24 \
  --cli-opts "--topk 8 --head-gate $config" "${extra[@]}"

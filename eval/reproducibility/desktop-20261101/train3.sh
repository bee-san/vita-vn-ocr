#!/usr/bin/env bash
# A final desktop detector run (#169, amendment 3): train.sh's sources + desktop-data's frozen pages (desk_hook,
# desk_realplus, desk_synth) at the registered mix. heavy.sh timeout 43200 train3.sh NAME SHAPE SEED SPLITS [args...]
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
NAME=${1:?name}; SHAPE=${2:?shape}; SEED=${3:?seed}; SPLITS=${4:?splits}; shift 4
DD=$HERD/shared/artifacts/desktop-data; F=$DD/frozen_v1
exec "$(dirname "$0")/train.sh" "$NAME" "$SHAPE" "$SEED" "$SPLITS" \
  --steps 40000 --mix hook=0.3375,synth=0.15,real=0.2625,desk_hook=0.1,desk_realplus=0.1,desk_synth=0.05 \
  --extra "desk_hook=$F/hooked_desktop_v1.json:$DD/pages/hooked_v1" "desk_realplus=$F/realplus_native_v1.json:$DD/pages/realplus_v1" \
          "desk_synth=$F/synth_desktop_v1.json:$DD/pages/synth_v1" "$@"

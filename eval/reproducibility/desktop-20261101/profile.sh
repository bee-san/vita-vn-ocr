#!/usr/bin/env bash
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
TAG=${1:?tag}; M=${2:-meiki,ts8,L,L:exact}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
exec $REPO/.venv/bin/python $REPO/eval/desktop_time.py profile --pages $PV/timing_pages.pkl --models "$M" \
  --cases all:1,disable:1,all:4 --inputs 40 --work $PV/prof_$TAG --out $A/timing/${TAG}_profile.json

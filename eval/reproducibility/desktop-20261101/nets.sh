#!/usr/bin/env bash
# vocr-desktop network decomposition (#169): each network alone at opt levels x threads, and the C runtime's builds.
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
TAG=${1:?tag}; M=${2:-meiki,ts8,L,L:exact}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
mkdir -p $A/timing
exec $REPO/.venv/bin/python $REPO/eval/desktop_time.py nets --pages $PV/timing_pages.pkl --models "$M" --threads 1,2,4,8 \
  --opts disable,basic,extended,all --passes 5 --warmup 1 --cbin $PV/cbuild --out $A/timing/${TAG}_nets.json

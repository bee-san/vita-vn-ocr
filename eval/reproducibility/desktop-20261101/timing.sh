#!/usr/bin/env bash
# vocr-desktop timing (#169): one intra-op thread count, every baseline pipeline interleaved, seq and par modes.
#   heavy.sh timeout 7200 timing.sh THREADS TAG [PIPELINES]     (THREADS 8: run inside two nested heavy.sh slots)
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
N=${1:?threads}; TAG=${2:?tag}
P=${3:-meikiocr,meikiocr-nospin,meiki+S,meiki+N,meiki+L,meiki+L:exact,ts8+S,ts8+N,ts8+L,ts8+L:exact}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
mkdir -p $A/timing
exec $REPO/.venv/bin/python $REPO/eval/desktop_time.py run --pages $PV/timing_pages.pkl --pipelines "$P" --threads $N \
  --modes seq,par --passes 3 --warmup 1 --out $A/timing/${TAG}_t$N.json

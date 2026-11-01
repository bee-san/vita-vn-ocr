#!/usr/bin/env bash
# vocr-desktop (#169): the decision's timing (amendment 1): a 4-thread budget, mode par; R (MeikiOCR's detector +
# VOCR6.2-L) and the representative's 7 policies, each detector at 1, 2 and 4 intra-op threads (#d1/#d2/#d4), all
# interleaved per page in one run on the timing pages. One heavy slot (5 threads).
#   heavy.sh timeout 7200 timing_decide.sh MODEL_DIR TAG
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
M=${1:?model dir}; TAG=${2:?tag}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
P="meiki+L#d1,meiki+L#d2,meiki+L#d4"
for pol in fit:960 fit:800 fit:640 fit:480 long:960 long:800 long:640; do
  for d in 1 2 4; do P="$P,desk:$M/det.onnx@$pol+L#d$d"; done
done
mkdir -p $A/timing
exec $REPO/.venv/bin/python $REPO/eval/desktop_time.py run --pages $PV/timing_pages.pkl --pipelines "$P" --threads 4 \
  --modes par --passes 3 --warmup 1 --ref "meiki+L#d4|par" --out $A/timing/${TAG}_decide_t4.json

#!/usr/bin/env bash
# vocr-desktop (#169): the report's timings and memory, every arm interleaved per page in each run (eval/desktop_time.py
# through eval/mida_det.py's runner, which adds the mida detector). One heavy slot per run; the 8-thread run needs two.
#   heavy.sh timeout 14400 timing_report.sh KIND THREADS
#     KIND grid:  the timing pages (60 Vita, 60 PSP, 30 PC) at THREADS
#     KIND l544:  the same pages fit into 960x544 (amendment 4), THREADS 4
#     KIND res:   dev-v2 resampled to 720 / 1080 / 1440 / 2160 px (20 Vita + 20 PSP frames each), THREADS 4
#     KIND mem:   peak memory, a fresh process per pipeline on 60 dev-v2 frames, THREADS 4
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
KIND=${1:?kind}; N=${2:-4}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=$REPO/.venv/bin/python
MD="$PY $REPO/eval/mida_det.py"
MODEL=${MIDA_MODEL:-/local/home/skerraut/herd-synth3/shared/mida-ocr/mida_ocr_l.onnx}
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
M=$A/models/final3
FD="desk:$M/vocr_desktop_det.onnx"; CD="cdesk:$M/vocr_desktop_det.vdsk"; MI="mida@box:960x544@along=1.0@min_cells=2"
mkdir -p $A/timing
case $KIND in
  grid)
    P="meikiocr,meiki+L,ts8+L,$MI+L,$FD@long:640+L,$FD@fit:960+L,$CD@long:640+c:L,cts8+c:L"
    if [ "$N" = 4 ]; then P="$P,meiki+L#d1,meiki+L#d2,$FD@long:640+L#d1,$FD@long:640+L#d2,$MI+L#d1,$MI+L#d2,ts8+L#d1,ts8+L#d2"; fi
    exec $MD time --repo $REPO --model $MODEL -- run --pages $PV/timing_pages.pkl --pipelines "$P" --threads $N \
      --modes par --passes 3 --warmup 1 --ref "meiki+L|par" --out $A/timing/report_grid_t$N.json ;;
  l544)
    P="meikiocr,meiki+L,ts8+L,$MI+L,$FD@fit:960+L,$CD@fit:960+c:L"
    exec $MD time --repo $REPO --model $MODEL -- run --pages $PV/timing_pages_l544.pkl --pipelines "$P" --threads $N \
      --modes par --passes 3 --warmup 1 --ref "meiki+L|par" --out $A/timing/report_l544_t$N.json ;;
  res)
    P="meikiocr,meiki+L,$MI+L,$FD@long:640+L,$FD@fit:960+L,$CD@long:640+c:L"
    exec $MD time --repo $REPO --model $MODEL -- run --pages $PV/timing_pages_res.pkl --pipelines "$P" --threads $N \
      --modes par --passes 3 --warmup 1 --ref "meiki+L|par" --out $A/timing/report_res_t$N.json ;;
  mem)
    P="meikiocr,meiki+L,ts8+L,$MI+L,$FD@long:640+L,$FD@fit:960+L,$CD@long:640+c:L,cts8+c:L"
    exec $MD mem --repo $REPO --model $MODEL --frames $PV/frames/dev.pkl --pipelines "$P" --threads $N --limit 60 \
      --out $A/timing/report_mem_t$N.json ;;
esac

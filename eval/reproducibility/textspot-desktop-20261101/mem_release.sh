#!/usr/bin/env bash
set -euo pipefail
cd /tmp/tsd-pr
export OMP_NUM_THREADS=1
P='cdet:/local/home/skerraut/herd-vita-ocr/shared/artifacts/textspot-desktop/release_files/textspot_desktop.vtsd@long:960+refine,odet:/local/home/skerraut/herd-vita-ocr/shared/artifacts/textspot-desktop/release_files/textspot_desktop.onnx@long:960@cv2+refine,cpage:/local/home/skerraut/herd-vita-ocr/shared/artifacts/textspot-desktop/release_files/textspot_desktop.vtsd@long:960+refine+c:L,opage:/local/home/skerraut/herd-vita-ocr/shared/artifacts/textspot-desktop/release_files/textspot_desktop.onnx@long:960@cv2+refine+L'
for T in 1 4; do for V in vita pc1080; do
  .venv/bin/python eval/tsdesk_time.py mem --pages /local/home/skerraut/herd-vita-ocr/shared/private/textspot-desktop/timing_pages.pkl --pipelines "$P" --threads $T --lib /tmp/tsd-pr/desktop/build/libvocrdesk.so --pad k6v4 --view $V --out /local/home/skerraut/herd-vita-ocr/shared/artifacts/textspot-desktop/res/timing/mem_release_e35bf00_t${T}_$V.json
done; done
echo "mem_release finished"

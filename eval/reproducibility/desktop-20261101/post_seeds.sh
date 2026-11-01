#!/usr/bin/env bash
# vocr-desktop (#169) after the final seeds: export (frame pad) and dev-v2 reads at the 7 candidate policies for
# every seed, then hval for the representative (the median seed by dev-v2 at fit:960). One heavy slot.
#   heavy.sh timeout 14400 post_seeds.sh
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=$REPO/.venv/bin/python; D="$PY $REPO/eval/desktop.py"
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
R=$HERD/shared/artifacts/vocr-onnx/release
POL="fit:960 fit:800 fit:640 fit:480 long:960 long:800 long:640"
mkdir -p $A/models
for s in 1 2 3; do
  RUN=$A/runs/F${s}_d64_seed$s; M=$A/models/F$s
  [ -s $RUN/done.json ] || { echo "F$s not finished"; exit 2; }
  [ -s $M/det.onnx ] || $PY $REPO/train/desktop_det.py export --run $RUN --out $M --name det --pad frame > $M.export.json
done
rd() { local set=$1 tag=$2 spec=$3
  local B=$PV/boxes/${set}_$tag.jsonl O=$A/res/${set}_${tag}_L_fast.json
  [ -s $B ] || $D detect --frames $PV/frames/$set.pkl --det "$spec" --out $B --workers 5 > $A/logs/det_${set}_$tag.log 2>&1
  [ -s $O ] || $D read --frames $PV/frames/$set.pkl --boxes $B --rec $R/VOCR6.2_L_fast.onnx --out $O \
      --rows $PV/rows/${set}_${tag}_L_fast.rows.jsonl --workers 5 > $A/logs/read_${set}_$tag.log 2>&1
  echo "$set $tag: $(tail -n 1 $A/logs/read_${set}_$tag.log)"
}
for s in 1 2 3; do for P in $POL; do rd dev F${s}_${P//:/} "desk:$A/models/F$s/det.onnx@$P"; done; done
REP=$($PY - <<EOF
import json
c = {s: json.load(open(f"$A/res/dev_F{s}_fit960_L_fast.json"))["summary"]["all"]["cer"] for s in (1, 2, 3)}
print(sorted(c, key=lambda s: (c[s], s))[1])
EOF
)
echo "representative seed: $REP"
echo $REP > $A/models/REPRESENTATIVE
for P in $POL; do rd hval F${REP}_${P//:/} "desk:$A/models/F$REP/det.onnx@$P"; done
# amendment 4's like-for-like rows (frames fit into 960x544; the detector reads them as they are)
rd dev_l544 F${REP}_fit960 "desk:$A/models/F$REP/det.onnx@fit:960"
rd hval_l544 F${REP}_fit960 "desk:$A/models/F$REP/det.onnx@fit:960"
echo "post_seeds finished"

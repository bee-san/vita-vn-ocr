#!/usr/bin/env bash
# vocr-desktop (#169, amendment 2's C-path requirement, for the research model): the C detector (desktop/, VDSK) against
# the ONNX file on dev-v2 at long:640: crops per frame, CER with VOCR6.2_L_fast.onnx, and the whole C page's CER
# (C detector + runtime/vocr.c W8I with the blob's head sections); then the INT8 check (calibrated on hval pages).
#   heavy.sh timeout 7200 c_parity.sh
set -euo pipefail
HERD=${HERD:-/local/home/skerraut/herd-vita-ocr}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=$REPO/.venv/bin/python; D="$PY $REPO/eval/desktop.py"
MD="$PY $REPO/eval/mida_det.py desktop --repo $REPO --model ${MIDA_MODEL:-/local/home/skerraut/herd-synth3/shared/mida-ocr/mida_ocr_l.onnx} --"
PV=$HERD/shared/private/vocr-desktop
A=$HERD/shared/artifacts/vocr-desktop
R=$HERD/shared/artifacts/vocr-onnx/release
M=$A/models/final3
F=$PV/frames/dev.pkl
[ -s $PV/boxes/dev_final3c_long640.jsonl ] || $D detect --frames $F --det "cdesk:$M/vocr_desktop_det.vdsk@long:640" --out $PV/boxes/dev_final3c_long640.jsonl --workers 5
$PY - <<EOF
import json, sys
sys.path.insert(0, "$REPO/eval")
import desktop as D
a, _ = D.load_boxes("$PV/boxes/dev_final3_long640.jsonl")
b, _ = D.load_boxes("$PV/boxes/dev_final3c_long640.jsonl")
fr, _ = D.load_frames("$F")
same = 0; na = nb = 0
for f in fr:
    ca = sorted(D.crop_box(x[:4], 0, f["W"], f["H"]) for x in a[f["key"]] if D.crop_box(x[:4], 0, f["W"], f["H"]))
    cb = sorted(D.crop_box(x[:4], 0, f["W"], f["H"]) for x in b[f["key"]] if D.crop_box(x[:4], 0, f["W"], f["H"]))
    same += ca == cb; na += len(ca); nb += len(cb)
rec = {"frames": len(fr), "frames_same_crops": same, "crops_onnx": na, "crops_c": nb}
json.dump(rec, open("$A/res/c_parity_dev_long640.json", "w"), indent=1)
print("c parity:", rec)
EOF
[ -s $A/res/dev_final3c_long640_L_fast.json ] || $D read --frames $F --boxes $PV/boxes/dev_final3c_long640.jsonl --rec $R/VOCR6.2_L_fast.onnx --out $A/res/dev_final3c_long640_L_fast.json --workers 5 | tail -n 1
[ -s $A/res/dev_final3c_long640_L_c.json ] || $MD read --frames $F --boxes $PV/boxes/dev_final3c_long640.jsonl --rec $A/models/VOCR6.2_L_w8_2l.vocr --out $A/res/dev_final3c_long640_L_c.json --workers 5 | tail -n 1
[ -s $A/res/int8_check_long640.json ] || $PY $REPO/eval/desktop_int8.py --onnx $M/vocr_desktop_det.onnx --calib $PV/frames/hval_calib64.pkl --calib-n 64 --frames $F --size long:640 --out $M/vocr_desktop_det_int8.onnx --json $A/res/int8_check_long640.json | tail -n 1
echo "c_parity finished"

#!/usr/bin/env python3
"""Does an INT8 desktop detector keep the FP32 file's outputs? (issue #169; the task ships an INT8 variant only if it
does, as docs/onnx.md measured for the recognizers.)

  desktop_int8.py --onnx DET.onnx --calib FRAMES.pkl --frames FRAMES.pkl --size fit:640 --out INT8.onnx --json R.json
                  [--calib-n 64]

onnxruntime.quantization.quantize_static (QDQ, per-channel int8 weights, uint8 activations, MinMax) calibrated on the
first --calib-n frames of --calib (training-side pages: hval), at the policy's input; the FP32 file's vocr.* metadata
copied over. Then on every frame of --frames: the detector's integer crops (eval/desktop.crop_box of each box, pad 0)
from the FP32 and the INT8 file, frames whose crops are all the same, and both files' network time at 1 thread.
Development frames are read here for agreement and speed only; no choice is made on them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "eval", ROOT / "synth", ROOT / "teacher", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import desktop as D  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--calib-n", type=int, default=64)
    a = ap.parse_args(argv)
    import onnx
    import onnxruntime as ort
    import vocr_desktop as VD
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    det = VD.Detector(a.onnx, size=a.size)
    calib, _ = D.load_frames(a.calib)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib[:a.calib_n])

        def get_next(self):
            fr = next(self.it, None)
            if fr is None:
                return None
            x, _ = det.preprocess(VD.grey_of(D.load_rgb(fr)))
            return {det.input_name: x}

    tmp = Path(a.out).with_suffix(".tmp.onnx")
    quantize_static(a.onnx, str(tmp), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    src, q = onnx.load(a.onnx), onnx.load(str(tmp))
    del q.metadata_props[:]
    for p in src.metadata_props:
        q.metadata_props.add().CopyFrom(p)
    onnx.save(q, a.out)
    tmp.unlink()
    d32 = VD.Detector(a.onnx, session_options=D.session_options(1), size=a.size)
    d8 = VD.Detector(a.out, session_options=D.session_options(1), size=a.size)
    frames, _ = D.load_frames(a.frames)
    same = 0
    boxes32 = boxes8 = 0
    t32, t8 = [], []
    for fr in frames:
        rgb = D.load_rgb(fr)
        g = VD.grey_of(rgb)
        H, W = g.shape
        c32 = sorted(D.crop_box(b[:4], 0, W, H) for b in d32.boxes(rgb) if D.crop_box(b[:4], 0, W, H))
        c8 = sorted(D.crop_box(b[:4], 0, W, H) for b in d8.boxes(rgb) if D.crop_box(b[:4], 0, W, H))
        same += int(c32 == c8)
        boxes32 += len(c32)
        boxes8 += len(c8)
        x, _ = d32.preprocess(g)
        for dd, tl in ((d32, t32), (d8, t8)):
            dd.maps(x)
            t0 = time.perf_counter()
            dd.maps(x)
            tl.append((time.perf_counter() - t0) * 1e3)
    res = {"kind": "vocr-desktop-int8-check", "onnx": Path(a.onnx).name, "onnx_sha256": D.sha256_file(a.onnx),
           "int8": Path(a.out).name, "int8_sha256": D.sha256_file(a.out), "int8_bytes": Path(a.out).stat().st_size,
           "fp32_bytes": Path(a.onnx).stat().st_size, "size": a.size, "calib_frames": min(a.calib_n, len(calib)),
           "calib_sha256": D.sha256_file(a.calib), "frames": len(frames), "frames_sha256": D.sha256_file(a.frames),
           "frames_same_crops": same, "boxes_fp32": boxes32, "boxes_int8": boxes8,
           "network_ms_1thread_median": {"fp32": round(float(np.median(t32)), 3),
                                         "int8": round(float(np.median(t8)), 3)},
           "onnxruntime": ort.__version__}
    Path(a.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

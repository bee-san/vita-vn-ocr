#!/usr/bin/env python3
"""Make private loader-probe data from the manual bundle and ARM baseline.

The reference floats are round-tripped from the standalone native probe's
nine-significant-digit JSON output. They are never host estimates of ARM output.
"""

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path


def detector_fixtures(model, eval_data, out, oom_model):
    import cv2
    import numpy as np

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "eval"))
    import sets

    if (
        hashlib.sha256(model.read_bytes()).hexdigest()
        != "f0865ed291846a130905527f67c33f17e64d0f2c5eb38378ed1faea57f74b1ba"
    ):
        raise ValueError("Detector does not match pinned conversion")
    shutil.copyfile(model, out / "meiki-detect-int8.mnn")
    cache = sets.ImageCache()
    cases = []
    for dataset in ("vita-test", "psp-test"):
        path, (x, y, w, h), _ = sets.regions(dataset, eval_data)[0]
        rgb = cache.get(path)[y : y + h, x : x + w]
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        name = f"detect-{dataset}"
        (out / f"{name}.rgba").write_bytes(struct.pack("<II", w, h) + rgba.tobytes())
        scale = min(320 / w, 192 / h)
        nw, nh = int(w * scale), int(h * scale)
        tensor = np.zeros((192, 320, 3), np.float32)
        tensor[:nh, :nw] = (
            cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (nw, nh), interpolation=cv2.INTER_LINEAR).astype(
                np.float32
            )
            / 255
        )
        tensor.transpose(2, 0, 1).astype("<f4").tofile(out / f"{name}.f32")
        cases.append(f"{name}\t{w}\t{h}\n")
    (out / "detect.tsv").write_text("".join(cases))
    if oom_model:
        if (
            hashlib.sha256(oom_model.read_bytes()).hexdigest()
            != "126b26ce33e03c76bb9b2168d76b96c95cb2600a51798024d4c64dfd298b8e4a"
        ):
            raise ValueError("OOM fixture is not the pinned existing960x544 detector")
        shutil.copyfile(oom_model, out / "meiki-detect-oom.mnn")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--oom-detector", type=Path)
    args = parser.parse_args()
    if (
        hashlib.sha256((args.bundle / "meiki-stream-int8.mnn").read_bytes()).hexdigest()
        != "b7ba3219cd1bfeef5cb6b42ef6686683b991c6561b7ad3b1c41d32817deba21e"
    ):
        raise ValueError("Model does not match pinned int8 conversion")
    rows = [json.loads(line) for line in args.baseline.read_text().splitlines()]
    terminal = rows[-1]
    cases = [row for row in rows if "index" in row]
    if terminal.get("status") != "complete" or terminal.get("failed") or len(cases) != 8:
        raise ValueError("Need a completed eight-run standalone ARM baseline")
    raw = []
    for i, row in enumerate(cases):
        if row["rc"] or row["heap_remaining"] or row["index"] != i:
            raise ValueError("Baseline inference/cleanup failed")
        if len(row["codes"]) != 48 or len(row["boxes"]) != 192 or len(row["scores"]) != 48:
            raise ValueError("Malformed raw baseline")
        raw.append(struct.pack("<48i240f", *row["codes"], *row["boxes"], *row["scores"]))
    if raw[:4] != raw[4:]:
        raise ValueError("Baseline repeats differ")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("cases.tsv", "meiki-stream-int8.mnn", "font.bin", "font-OFL.txt"):
        shutil.copyfile(args.bundle / name, args.out / name)
    for tensor in args.bundle.glob("*.f32"):
        shutil.copyfile(tensor, args.out / tensor.name)
    (args.out / "reference.raw").write_bytes(b"".join(raw[:4]))
    if args.detector_model:
        detector_fixtures(
            args.detector_model,
            args.eval_data or Path(__file__).resolve().parents[3] / "data/eval",
            args.out,
            args.oom_detector,
        )
    manifest = {
        "kind": "Private actual ARM baseline for isolated module probe",
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256((args.out / "reference.raw").read_bytes()).hexdigest(),
        "reference_bytes": len(b"".join(raw[:4])),
        "cases": 4,
    }
    (args.out / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

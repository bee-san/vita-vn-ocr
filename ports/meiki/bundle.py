#!/usr/bin/env python3
"""Build the private, self-contained Meiki manual-test VPK's data.

Uses upstream preprocessing and four real test crops; never publish this
bundle. The VPK needs no separate weights transfer or remote-control plugin.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "vita"), str(ROOT / "ci")]
from bundle import write_font  # noqa: E402
from sets import load_real  # noqa: E402

MODEL_SHA256 = "b7ba3219cd1bfeef5cb6b42ef6686683b991c6561b7ad3b1c41d32817deba21e"


def reply(process):
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("MNN probe stopped; see its stderr log")
        if line.startswith('{"ms":'):
            result = json.loads(line)
            if result["rc"] or result["allocation_failed"] or result["heap_remaining"]:
                raise RuntimeError(f"Bounded runtime failed: {result}")
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/eval")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Model does not match the pinned weight-only int8 conversion")
    sys.path.insert(0, str(args.upstream))
    from meikiocr import MeikiOCR

    upstream = MeikiOCR.__new__(MeikiOCR)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.model, args.out / args.model.name)
    entries, cases = [], []
    with (args.out / "host.log").open("w") as log:
        process = subprocess.Popen(
            [str(args.probe), str(args.model), "16"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            for dataset in ("vita-test", "psp-test"):
                lines, _ = load_real(dataset, args.data_dir)
                chosen = [line for line in lines if "name" not in line.styles and len(line.text) >= 8][:2]
                if len(chosen) != 2:
                    raise ValueError(f"Need two dialogue crops from {dataset}")
                for i, line in enumerate(chosen):
                    image = cv2.cvtColor(line.crop_rgb, cv2.COLOR_RGB2BGR)
                    h, w = image.shape[:2]
                    batch, indices, metadata = upstream._preprocess_for_recognition(
                        image, [{"bbox": [0, 0, w, h]}], [0], False
                    )
                    filename = f"{dataset}-{i}.f32"
                    path = args.out / filename
                    batch.astype("<f4").tofile(path)
                    effective = metadata[0]["effective_w"]
                    process.stdin.write(f"{path}\t{w}\t{effective}\n")
                    process.stdin.flush()
                    result = reply(process)
                    raw = [
                        np.array(result["codes"], dtype=np.int64).reshape(1, 48),
                        np.array(result["boxes"], dtype=np.float32).reshape(1, 48, 4),
                        np.array(result["scores"], dtype=np.float32).reshape(1, 48),
                    ]
                    decoded = [{}]
                    upstream._postprocess_recognition_results(raw, indices, metadata, 0.1, decoded, 1.0, False)
                    text = decoded[0].get("text", "")
                    if result["decoded"] < 0 or bytes.fromhex(result["text_hex"]).decode() != text:
                        raise ValueError("C decoder disagrees with the upstream decoder")
                    if any(c in text for c in "\t\r\n"):
                        raise ValueError("Expected text cannot be represented as a TSV field")
                    cases.append(f"{filename}\t{w}\t{effective}\t{text}\n")
                    entries.append(
                        {
                            "id": line.id,
                            "set": dataset,
                            "filename": filename,
                            "crop_width": w,
                            "effective_width": effective,
                            "ground_truth": line.text,
                            "expected_text": text,
                            "raw": result,
                        }
                    )
        finally:
            process.stdin.close()
            rc = process.wait(timeout=20)
            process.stdout.close()
        if rc:
            raise RuntimeError(f"Probe exited {rc}")
    (args.out / "cases.tsv").write_text("".join(cases + cases), encoding="utf-8")
    write_font(args.out, [entry["expected_text"] for entry in entries], [])
    (args.out / "expected.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(entries),
                "runs": len(entries) * 2,
                "model_sha256": MODEL_SHA256,
                "host_pool_peak_bytes": max(entry["raw"]["heap_peak"] for entry in entries),
                "font_bytes": (args.out / "font.bin").stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check the bounded binding and C decoder against upstream on real crops.

The reference JSONL is evaluate.py's unbounded MNN int8 run. This separately
checks repeated load/unload and decoder parity, without retraining a model.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import score_lines  # noqa: E402
from sets import load_real  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/eval")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream))
    from meikiocr import MeikiOCR

    upstream = MeikiOCR.__new__(MeikiOCR)
    references = {row["id"]: row["pred"] for row in map(json.loads, args.reference.read_text().splitlines())}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    peaks, retained, c_differences, reference_differences = [], [], 0, 0
    with (args.out / "stderr.log").open("w") as log, (args.out / "runtime.jsonl").open("w") as raw_log:
        process = subprocess.Popen(
            [str(args.probe), str(args.model), "16"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            for dataset in ("vita-test", "psp-test"):
                rows[dataset] = []
                lines, _ = load_real(dataset, args.data_dir)
                for index, line in enumerate(lines, 1):
                    image = cv2.cvtColor(line.crop_rgb, cv2.COLOR_RGB2BGR)
                    h, w = image.shape[:2]
                    batch, indices, metadata = upstream._preprocess_for_recognition(
                        image, [{"bbox": [0, 0, w, h]}], [0], False
                    )
                    path = args.out / "input.f32"
                    batch.astype("<f4").tofile(path)
                    process.stdin.write(f"{path}\t{w}\t{metadata[0]['effective_w']}\n")
                    process.stdin.flush()
                    while True:
                        response = process.stdout.readline()
                        if not response:
                            raise RuntimeError(f"Probe stopped at {line.id}; see stderr.log")
                        if response.startswith('{"ms":'):
                            result = json.loads(response)
                            break
                    if result["rc"] or result["allocation_failed"] or result["decoded"] < 0:
                        raise RuntimeError(f"Runtime failed at {line.id}: {result}")
                    raw_log.write(json.dumps({"id": line.id, **result}) + "\n")
                    raw = [
                        np.array(result["codes"], dtype=np.int64).reshape(1, 48),
                        np.array(result["boxes"], dtype=np.float32).reshape(1, 48, 4),
                        np.array(result["scores"], dtype=np.float32).reshape(1, 48),
                    ]
                    decoded = [{}]
                    upstream._postprocess_recognition_results(raw, indices, metadata, 0.1, decoded, 1.0, False)
                    text = bytes.fromhex(result["text_hex"]).decode()
                    c_differences += text != decoded[0].get("text", "")
                    reference_differences += text != references[line.id]
                    peaks.append(result["heap_peak"])
                    retained.append(result["heap_remaining"])
                    rows[dataset].append(
                        {"id": line.id, "pred": text, "text": line.text, "styles": line.styles, "vn": line.vn}
                    )
                    if index % 25 == 0:
                        print(f"{dataset}: {index}/{len(lines)}", flush=True)
        finally:
            process.stdin.close()
            rc = process.wait(timeout=20)
            process.stdout.close()
        if rc:
            raise RuntimeError(f"Probe exited {rc}")
    summary = {
        "kind": "host static MNN, same C binding; not physical Vita RAM or speed",
        "cases": len(peaks),
        "workspace_bytes": 16 * 1024 * 1024,
        "peak_model_tensor_bytes": max(peaks),
        "max_retained_pool_bytes": max(retained),
        "c_decoder_text_differences": c_differences,
        "reference_text_differences": reference_differences,
        "datasets": {name: score_lines(items) for name, items in rows.items()},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "datasets"}, indent=2))
    if c_differences or reference_differences or max(retained):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

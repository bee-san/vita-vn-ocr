#!/usr/bin/env python3
"""Compare a prepared Meiki model on the repo's hand-checked real test sets.

Uses the pinned upstream Meiki preprocessing and postprocessing, without
loading its detector or vertical model. Test crops are never calibration data.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import score_lines  # noqa: E402
from sets import load_real  # noqa: E402


def ort_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--mnn-probe", type=Path)
    parser.add_argument("--mnn-model", type=Path, action="append", default=[])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/eval")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.mnn_model and not args.mnn_probe:
        parser.error("--mnn-model requires --mnn-probe")
    sys.path.insert(0, str(args.upstream))
    from meikiocr import MeikiOCR

    upstream = MeikiOCR.__new__(MeikiOCR)
    sessions = {"original": ort_session(args.original), "prepared": ort_session(args.prepared)}
    args.output.mkdir(parents=True, exist_ok=True)
    processes, logs = {}, []
    for path in args.mnn_model:
        log = (args.output / (path.stem + ".log")).open("w")
        logs.append(log)
        processes[path.stem] = subprocess.Popen(
            [str(args.mnn_probe), str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True
        )
    names = list(sessions) + list(processes)
    rows = {name: {} for name in names}
    timings = {name: [] for name in names}
    parity = {"raw_code_differences": 0, "text_differences": 0, "max_box_delta": 0.0, "max_score_delta": 0.0}
    tensor_file = args.output / "input.f32"
    try:
        for set_name in ("vita-test", "psp-test"):
            lines, _ = load_real(set_name, args.data_dir)
            if args.limit is not None:
                lines = lines[: args.limit]
            for name in names:
                rows[name][set_name] = []
            for count, line in enumerate(lines, 1):
                image = cv2.cvtColor(line.crop_rgb, cv2.COLOR_RGB2BGR)
                h, w = image.shape[:2]
                batch, indices, metadata = upstream._preprocess_for_recognition(
                    image, [{"bbox": [0, 0, w, h]}], [0], False
                )
                raw, text = {}, {}
                for name, session in sessions.items():
                    feed = {"images": batch}
                    if name == "original":
                        feed["orig_target_sizes"] = np.array([[960, 32]], dtype=np.int64)
                    started = time.perf_counter()
                    raw[name] = session.run(None, feed)
                    timings[name].append((time.perf_counter() - started) * 1000)
                batch.astype("<f4").tofile(tensor_file)
                for name, process in processes.items():
                    process.stdin.write(str(tensor_file) + "\n")
                    process.stdin.flush()
                    while True:
                        response = process.stdout.readline()
                        if not response:
                            raise RuntimeError(f"{name} probe exited {process.poll()}; see its log")
                        if response.startswith('{"ms":'):
                            result = json.loads(response)
                            break
                    raw[name] = [
                        np.array(result["codes"], dtype=np.int64).reshape(1, 48),
                        np.array(result["boxes"], dtype=np.float32).reshape(1, 48, 4),
                        np.array(result["scores"], dtype=np.float32).reshape(1, 48),
                    ]
                    timings[name].append(result["ms"])
                for name in names:
                    result = [{}]
                    upstream._postprocess_recognition_results(raw[name], indices, metadata, 0.1, result, 1.0, False)
                    text[name] = result[0].get("text", "")
                    rows[name][set_name].append(
                        {"id": line.id, "pred": text[name], "text": line.text, "styles": line.styles, "vn": line.vn}
                    )
                parity["raw_code_differences"] += int(np.count_nonzero(raw["original"][0] != raw["prepared"][0]))
                parity["text_differences"] += int(text["original"] != text["prepared"])
                parity["max_box_delta"] = max(
                    parity["max_box_delta"], float(np.max(np.abs(raw["original"][1] - raw["prepared"][1])))
                )
                parity["max_score_delta"] = max(
                    parity["max_score_delta"], float(np.max(np.abs(raw["original"][2] - raw["prepared"][2])))
                )
                if count % 25 == 0:
                    print(f"{set_name}: {count}/{len(lines)}", flush=True)
        summary = {
            "environment": {
                "kind": "host CPU, one thread; not Vita speed or total Vita memory",
                "onnxruntime": ort.__version__,
            },
            "graph_parity": parity,
            "models": {},
        }
        for name in names:
            summary["models"][name] = {set_name: score_lines(items) for set_name, items in rows[name].items()}
            summary["models"][name]["host_ms"] = {
                "median": float(np.median(timings[name])),
                "p95": float(np.percentile(timings[name], 95)),
            }
            (args.output / (name + ".jsonl")).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for items in rows[name].values() for row in items),
                encoding="utf-8",
            )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        for process in processes.values():
            process.stdin.close()
            process.wait(timeout=10)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()

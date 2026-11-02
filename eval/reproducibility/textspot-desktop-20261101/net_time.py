"""The screening rule's network time (#173 protocol screening.rule 2): ONNX Runtime 1 thread, random uint8 input,
the median of 25 runs after 3 warm-ups, at 960x544 (and 1920x1088, 480x288 for the report).

  net_time.py MODEL OUT
"""
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort


def main():
    m, out = sys.argv[1], sys.argv[2]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    s = ort.InferenceSession(m, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    res = {"model": m, "load_1min": round(os.getloadavg()[0], 1), "ort": ort.__version__}
    for w, h in ((960, 544), (1920, 1088), (480, 288)):
        x = np.random.default_rng(0).integers(0, 256, (1, 1, h, w), dtype=np.uint8)
        for _ in range(3):
            s.run(None, {name: x})
        ts = []
        for _ in range(25):
            t0 = time.perf_counter()
            s.run(None, {name: x})
            ts.append((time.perf_counter() - t0) * 1e3)
        res[f"{w}x{h}"] = {"median_ms": round(float(np.median(ts)), 3),
                           "p95_ms": round(float(np.percentile(ts, 95)), 3)}
    with open(out, "w") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()

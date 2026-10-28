#!/usr/bin/env python3
"""MeikiOCR as an eval/run.py `cmd:` recognizer, for report-only comparisons (eval/other_readers_protocol.json).

  eval/run.py --model 'cmd:.venv-meiki/bin/python eval/meiki_cmd.py' --set vita-test --polarity native ...

One crop path per stdin line in, one UTF-8 line per crop out, in order (eval/README.md's cmd: protocol). The engine is
eval/clannad_meiki.py's: rtr46/meikiocr 0.3.5 at its pinned commit, with the pinned model files (checked by size and
sha256 before the first read; nothing is downloaded), one onnxruntime thread, and its arm (a)'s call
run_recognition([bgr], conf_threshold=0.1, punct_conf_factor=1.0) on the crop as BGR uint8 (cv2.imread: a crop
written with three identical channels reads as one grey line). A line break inside a reading is dropped.

Run it in .venv-meiki (uv pip sync --require-hashes eval/clannad_meiki_requirements.lock).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    import clannad_meiki as CM
    import cv2
    import numpy as np
    CM.pinned_engine()                      # meikiocr is the pinned 0.3.5 / 52aa607, or this exits
    m = CM.load_meiki(n_threads=1)
    for line in sys.stdin:
        path = line.rstrip("\r\n")
        if not path:
            continue
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"meiki_cmd: cannot read {path}")
        r = m.run_recognition([np.ascontiguousarray(bgr)], conf_threshold=CM.REC_THRESHOLD,
                              punct_conf_factor=CM.PUNCT_FACTOR)[0]
        sys.stdout.write(r["text"].replace("\r", "").replace("\n", "") + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

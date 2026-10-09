#!/usr/bin/env python3
"""One teacher engine as an eval/run.py `cmd:` recognizer: reads crop paths on stdin, prints
one UTF-8 line of text per crop, in order.

  eval/run.py --model 'cmd:unshare -rn teacher/.venv/bin/python teacher/engine_cli.py meiki' \
              --set vita-test,psp-test --data $HERD/shared/artifacts/eval

ENGINE: meiki | meiki-rgb (MeikiOCR's recognizer on the crop, BGR or RGB input) |
        screenai (Chrome Screen AI through owocr, every horizontal line it finds, top to
        bottom) | v6s | v5s | manga (the voters, preprocessed as in teacher/label.py).
Text is printed as the engine returns it; eval scores it with its own norm().
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    import cv2
    name = sys.argv[1]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import engines
    eng = engines.load([name])[name]
    out = sys.stdout
    for path in sys.stdin.read().split("\n"):
        if not path:
            continue
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        text = eng(bgr)[0] if bgr is not None and bgr.size else ""
        out.write(text.replace("\n", " ").replace("\r", " ") + "\n")
    out.flush()


if __name__ == "__main__":
    main()

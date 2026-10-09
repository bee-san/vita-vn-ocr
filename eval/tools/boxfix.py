#!/usr/bin/env python3
"""Refine hand-given line windows to ink boxes and show them (build tool).

  boxfix.py vita SHOT x0,y0,x1,y1 [SHOT x0,y0,x1,y1 ...]

For each window, prints the ink box (assemble.refine on that window) as
@x,y,w,h for review.py, and writes data/eval/sheets/<platform>_boxfix.png with
every refined crop (box + 4 px, scaled for reading) to check by eye.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from assemble import refine  # noqa: E402
from fetch import data_dir  # noqa: E402


def main(which, args):
    data = data_dir()
    tiles = []
    for shot, win in zip(args[::2], args[1::2]):
        raw = win.startswith("=")
        x0, y0, x1, y1 = (int(v) for v in win.lstrip("=").split(","))
        rgb = np.asarray(Image.open(data / which / "shots" / shot).convert("RGB"))
        if raw:
            (bx0, by0, bx1, by1), hint = (x0, y0, x1, y1), {}
        else:
            (bx0, by0, bx1, by1), hint = refine(rgb, [x0, y0, x1, y1])
        print(f"{shot} window {win} -> @{bx0},{by0},{bx1 - bx0},{by1 - by0} {hint}")
        H, W = rgb.shape[:2]
        crop = Image.fromarray(rgb[max(0, by0 - 4):min(H, by1 + 4), max(0, bx0 - 4):min(W, bx1 + 4)])
        s = min(2.0 if which == "vita" else 3.0, 1200 / crop.width)
        tiles.append(crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS))
    sheet = Image.new("RGB", (1280, min(1280, sum(t.height + 6 for t in tiles))), (128, 128, 128))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height + 6
    out = data / "sheets" / f"{which}_boxfix.png"
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

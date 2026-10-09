#!/usr/bin/env python3
"""Host CPU time of runtime/vocr_lines.c per region (vocr_lines_test --bench).

  bench_c.py --manifest M.jsonl [--manifest ...] [--reps 20] [--full 40] [--bin PATH]

Every manifest region is passed as RGBA (as the overlay passes its framebuffer copy) to
both finders, REPS times in a row; the best of the REPS runs is that region's time. With
--full N, the first N distinct screenshots are also timed whole (960x544 or 480x272), the
largest input the finder accepts. Prints median / 95th percentile / max microseconds per
set and finder. These are host numbers (x86-64, -O2), not console numbers.
"""
import argparse
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from build_gt import SHOT_DIRS, load_shot, read, region_rgb
from cfinder import build

FINDERS = {"grad": 0, "legacy": 1}


def rgba(img):
    return np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], axis=2)


def bench(binary, imgs, reps):
    """[(img, finder)] -> best microseconds per record."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        for img, finder in imgs:
            h, w = img.shape[:2]
            f.write(struct.pack("<5I", w, h, 4, FINDERS[finder], w * 4))
            f.write(np.ascontiguousarray(rgba(img)).tobytes())
        f.flush()
        out = subprocess.run([str(binary), "--bench", f.name, str(reps)], check=True,
                             capture_output=True, text=True).stdout.splitlines()
    return [float(line.split()[3]) for line in out if not line.startswith("total")]


def stats(us):
    return f"{np.median(us):.0f} | {np.percentile(us, 95):.0f} | {np.max(us):.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--full", type=int, default=0)
    ap.add_argument("--bin")
    args = ap.parse_args()
    binary = Path(args.bin) if args.bin else build()
    print("| set | finder | regions | median us | p95 us | max us |")
    print("|---|---|---|---|---|---|")
    shots = {}
    for m in args.manifest:
        rows = read(m)
        imgs = [region_rgb(r, SHOT_DIRS) for r in rows]
        for r in rows:
            shots.setdefault(r["platform"], {}).setdefault(r["shot"], r)
        for finder in FINDERS:
            us = bench(binary, [(img, finder) for img in imgs], args.reps)
            print(f"| {Path(m).stem} | {finder} | {len(imgs)} | {stats(us)} |")
    if args.full:
        for plat, rows in sorted(shots.items()):
            full = [load_shot(r, SHOT_DIRS) for r in list(rows.values())[:args.full]]
            size = "x".join(map(str, full[0].shape[1::-1]))
            for finder in FINDERS:
                us = bench(binary, [(img, finder) for img in full], args.reps)
                print(f"| full screen {plat} {size} | {finder} | {len(full)} | {stats(us)} |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the VPK's LiveArea images at build time (vita/CMakeLists.txt calls this).

  mkicon.py OUTDIR    writes OUTDIR/icon0.png (128x128), bg.png (840x500), startup.png (280x158)

The Vita wants 8-bit palette PNGs here. These are plain generated pictures (a dark box
with three light "text lines"), made with the standard library so no image file is ever
committed (tests/test_repo_policy.py).
"""
import struct
import sys
import zlib
from pathlib import Path

PALETTE = [(16, 24, 48), (230, 236, 255), (90, 160, 255)]   # background, text, accent


def png(width, height, pixel):
    """An 8-bit palette PNG; pixel(x, y) returns a palette index."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    rows = b"".join(b"\0" + bytes(pixel(x, y) for x in range(width)) for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0))
            + chunk(b"PLTE", b"".join(bytes(c) for c in PALETTE))
            + chunk(b"IDAT", zlib.compress(rows, 9))
            + chunk(b"IEND", b""))


def lines_picture(width, height):
    """Three bars like dialogue lines in a box, with an accent frame."""
    m = max(2, width // 16)

    def pixel(x, y):
        if x < m // 2 or y < m // 2 or x >= width - m // 2 or y >= height - m // 2:
            return 2
        band = (y - m) * 6 // max(1, height - 2 * m)
        in_line = band in (1, 3, 5) and m <= x < width - m - (band == 5) * width // 3
        return 1 if in_line and ((x // max(1, width // 40)) % 5) != 4 else 0
    return png(width, height, pixel)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in {"icon0.png": (128, 128), "bg.png": (840, 500), "startup.png": (280, 158)}.items():
        (out / name).write_bytes(lines_picture(w, h))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parity of runtime/vocr_vertical.c's column finder with its Python twin, synth/vertical_twin.py find_columns.

  vertical_parity.py --bin runtime/build/.../vocr_vertical_test [--cases N] [--seed S] [--regions FILE.jsonl]

Procedural regions (columns of stroke glyphs over flat, gradient or noisy backgrounds, either polarity, with
furigana beside some columns, horizontal lines and clutter; grey and RGB), and optionally real regions (eval's
vertical regions manifest: each row's page cut to its region), through the C driver (in-memory and row forms, both
flag sets: valleys and no-dark) and the twin: every box, dark flag, the order and the truncated flag must be equal. The rotation is
checked too (vocr_vertical_test --rotate against np.rot90). Exits non-zero on any difference.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
import common  # noqa: E402
import vertical_twin as V  # noqa: E402


def region(rng):
    h = int(rng.integers(16, 545))
    w = int(rng.integers(16, 961))
    kind = rng.integers(3)
    bg = int(rng.integers(0, 256))
    if kind == 0:
        g = np.full((h, w), bg, np.int32)
    elif kind == 1:
        yy, xx = np.mgrid[0:h, 0:w]
        g = (bg + xx // 4 + yy // 3) % 256
    else:
        g = (bg + rng.integers(0, 40, size=(h, w))) % 256
    fg = int(rng.integers(0, 60)) if bg > 128 else int(rng.integers(200, 256))
    for _ in range(int(rng.integers(1, 9))):
        cw = int(rng.integers(8, 34))
        pitch = cw + int(rng.integers(0, cw // 3 + 1))
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, max(1, h // 2)))
        for k in range(int(rng.integers(1, 16))):
            cy = y0 + k * pitch
            for _ in range(int(rng.integers(2, 7))):
                vert = rng.random() < 0.5
                ln = int(rng.integers(2, cw + 1))
                ox, oy = int(rng.integers(0, cw)), int(rng.integers(0, cw))
                t = np.arange(ln)
                ys = cy + (oy + t if vert else oy + 0 * t)
                xs = x0 + (ox + 0 * t if vert else ox + t)
                ok = (ys < h) & (xs < w)
                g[ys[ok], xs[ok]] = fg
                if rng.random() < 0.5:
                    ok2 = ok & (xs + 1 < w)
                    g[ys[ok2], xs[ok2] + 1] = fg
        if rng.random() < 0.3:                      # furigana beside the column
            fx = x0 + cw + 2
            for k in range(int(rng.integers(1, 10))):
                fy = y0 + k * (pitch // 2)
                if fy + 4 < h and fx + 4 < w:
                    g[fy:fy + 4, fx:fx + 4] = fg
    if rng.random() < 0.3:                          # a horizontal line of strokes
        y = int(rng.integers(0, h))
        for x in range(int(rng.integers(0, w)), w, 3):
            g[y:min(h, y + int(rng.integers(1, 3))), x:min(w, x + 2)] = fg
    g = np.clip(g, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                          # RGB: grey through the RGBA path
        rgb = np.stack([g, np.clip(g.astype(int) + rng.integers(-20, 21), 0, 255).astype(np.uint8), g], -1)
        return rgb
    return g


def write_pnm(path, img):
    img = np.ascontiguousarray(img)
    with open(path, "wb") as f:
        if img.ndim == 2:
            f.write(b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
        else:
            f.write(b"P6\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
        f.write(img.tobytes())


def read_pgm(path):
    """A binary PGM (P5, maxval 255): the magic, width, height and maxval tokens, one whitespace byte, the rows."""
    data = open(path, "rb").read()
    tokens, i = [], 0
    while len(tokens) < 4:
        while data[i:i + 1].isspace():
            i += 1
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    if tokens[0] != b"P5" or tokens[3] != b"255":
        raise SystemExit(f"{path}: not a P5 maxval-255 PGM")
    w, h = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data[i + 1:i + 1 + w * h], np.uint8).reshape(h, w)


def c_columns(binpath, path, flags, rows):
    cmd = [*binpath, path, "--flags", str(flags)] + (["--rows"] if rows else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise SystemExit(f"driver failed on {path}: {r.stderr}")
    boxes, dark, trunc = [], [], None
    for line in r.stdout.splitlines():
        if line.startswith("truncated"):
            trunc = bool(int(line.split()[1]))
            continue
        x, y, w, h, d = (int(v) for v in line.split())
        boxes.append((x, y, w, h))
        dark.append(d)
    return boxes, dark, trunc


def real_regions(path, limit):
    sys.path.insert(0, str(ROOT / "eval"))
    from vertical_data import load_page
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines()[:limit]:
        r = json.loads(line)
        rgb = load_page({"path": r["path"], "kind": r["kind"]})
        if rgb is None:
            continue
        x, y, w, h = r["region"]
        out.append(np.ascontiguousarray(rgb[y:y + h, x:x + w]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="vocr_vertical_test (may be 'ENV=... path')")
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--regions", help="eval vertical regions manifest (real regions; local data only)")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    binpath = args.bin.split()
    rng = np.random.default_rng(args.seed)
    imgs = [region(rng) for _ in range(args.cases)]
    if args.regions:
        imgs += real_regions(args.regions, args.limit)
    bad, cols = 0, 0
    with tempfile.TemporaryDirectory(prefix="vpar_") as td:
        for i, img in enumerate(imgs):
            p = os.path.join(td, f"r{i}.pnm")
            write_pnm(p, img)
            for flags in (0, 1, 2, 3):
                want = V.find_columns(img, flags)
                cols += len(want[0])
                for rows in (False, True):
                    got = c_columns(binpath, p, flags, rows)
                    if (list(map(tuple, want[0])), list(want[1]), bool(want[2])) != got:
                        bad += 1
                        if bad <= 5:
                            print(f"region {i} {img.shape} flags {flags} rows {rows}:\n  twin {want}\n  C    {got}")
            if i < 40:                               # the rotation
                g = img if img.ndim == 2 else common.to_gray(img)
                q = os.path.join(td, f"o{i}.pgm")
                subprocess.run([*binpath, "--rotate", p, q], check=True, timeout=60)
                rot = read_pgm(q)
                if not np.array_equal(rot, V.rotate_ccw(g)):
                    bad += 1
                    print(f"region {i}: rotation differs")
    print(f"vertical parity: {len(imgs)} regions, {cols} twin columns, {bad} differences")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

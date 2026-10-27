#!/usr/bin/env python3
"""runtime/vocr_textspot7.c against its twin (synth/textspot7_twin.py) through a built driver (runtime/
vocr_textspot7_test.c): F's choice over random spotter / GRAD box sets and every grid parameter, D's post-pass over
random regions, boxes and grey pixels for every step set, and the row-streamed post-pass (kind 3) against the
in-memory one. Used by `make -C runtime check-textspot7` (under ASan + UBSan in CI).

  textspot7_parity.py --bin DRIVER [--seed 0] [--choose 1500] [--post 400]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "eval", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot7 as E  # noqa: E402
import textspot7_twin as T7  # noqa: E402

COMBOS = [[], ["assign"], ["join5"], ["join10"], ["extend"], ["tighten"], ["assign", "join5", "extend"],
          ["assign", "join10", "extend", "tighten"], ["join5", "tighten"], ["join10", "extend"]]


def boxes(rng, n, W, H, conf=True):
    out = []
    for _ in range(n):
        w, h = int(rng.integers(4, max(5, W // 2))), int(rng.integers(4, 60))
        x, y = int(rng.integers(-20, W)), int(rng.integers(-20, H))
        out.append((x, y, w, h, int(rng.integers(0, 128)), int(rng.integers(0, 2))) if conf else (x, y, w, h))
    return out


def image(rng, H, W):
    g = np.full((H, W), int(rng.integers(0, 256)), np.uint8)
    for _ in range(int(rng.integers(1, 6))):
        y, x = int(rng.integers(0, H)), int(rng.integers(0, W))
        h, w = int(rng.integers(2, max(3, H // 2))), int(rng.integers(2, max(3, W)))
        step = int(rng.integers(6, 16))
        for xx in range(x, min(W, x + w), step):
            g[y:y + h, xx:xx + step // 2] = int(rng.integers(0, 256))
    if rng.random() < 0.5:
        g = (g.astype(np.int16) + rng.integers(-25, 25, g.shape)).clip(0, 255).astype(np.uint8)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choose", type=int, default=1500)
    ap.add_argument("--post", type=int, default=400)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    grid = E.grid()
    recs, want = [], []
    for k in range(a.choose):
        S = [b[:5] for b in boxes(rng, int(rng.integers(0, 9)), 500, 200)]
        G = [b[:4] for b in boxes(rng, int(rng.integers(0, 7)), 500, 200, conf=False)]
        if k % 3 == 0 and S:
            G = [(0, 0, 500, 200)] + G
        p = grid[int(rng.integers(0, len(grid)))]
        recs.append(E.rec_choose(S, G, p))
        want.append("C 0" if T7.choose(S, G, p) == "S" else "C 1")
    got = E.run_c(a.bin, recs)
    bad = sum(1 for x, y in zip(got, want) if x != y) + abs(len(got) - len(want))
    print(f"choose: {len(want)} records, {bad} differ")
    recs2, recs3, wantp = [], [], []
    for k in range(a.post):
        W, H = int(rng.integers(8, 420)), int(rng.integers(8, 170))
        X0, Y0 = int(rng.integers(0, 300)), int(rng.integers(0, 200))
        region = (X0, Y0, X0 + W, Y0 + H)
        g = image(rng, H, W)
        fb = [(x + X0, y + Y0, w, h, c, d) for x, y, w, h, c, d in boxes(rng, int(rng.integers(0, 10)), W, H)]
        steps = COMBOS[k % len(COMBOS)]
        r = E.rec_post(fb, region, g, steps)
        recs2.append(r)
        recs3.append(E.i32(3) + r[4:])
        wantp.append([tuple(b) for b in T7.post(fb, region, g, steps)])
    got2 = [E.parse_post(line) for line in E.run_c(a.bin, recs2)]
    got3 = [E.parse_post(line) for line in E.run_c(a.bin, recs3)]
    bad2 = sum(1 for x, y in zip(got2, wantp) if x != y) + abs(len(got2) - len(wantp))
    bad3 = sum(1 for x, y in zip(got3, got2) if x != y) + abs(len(got3) - len(got2))
    print(f"post: {len(wantp)} records, {bad2} differ from the twin; post_rows: {bad3} differ from post")
    if bad or bad2 or bad3:
        sys.exit(1)
    print("textspot7 parity: every record equal")


if __name__ == "__main__":
    main()

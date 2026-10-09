#!/usr/bin/env python3
"""Look for training screenshots that nearly duplicate a test screenshot (audit tool).

  leakcheck.py POOL_DIR... [--min-corr 0.95] [--test DIR ...] [--keep-reserved] [--data DIR]

Every screenshot in the test manifests (or every image under the --test dirs) is
compared with every *.jpg / *.png under the pool dirs, recursively, by the
correlation of 32x18 grey thumbnails (each normalised to zero mean and unit
variance). Measured on the 74 vita-test screenshots: the same scene with its
dialogue painted out, re-saved as JPEG q60 and scaled to 480x272 and back still
correlates at least 0.958 with the original, while 3,000 random pairs from
different VNs peak at 0.78 (Vita) and 0.74 (PSP). Each pair at or above
--min-corr is printed, tagged "same-vn" when both file names start with the same VNDB id
(v123_...), "cross-vn" when they differ, then the spread of each test
screenshot's best match. A same-vn pair means the pool holds a test game; a
cross-vn pair is one capture filed under two ids (or a shared stock image) and
needs a look. Pool images of reserved games (manifests/test_games.json "all", or
synth/common.py is_test_vn) are skipped first, as training must skip them;
--keep-reserved audits the raw pool. Training-data agents can run it on their pools.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "synth"))
import sets  # noqa: E402
from fetch import data_dir, shot_path  # noqa: E402

from common import is_test_vn  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}
VN = re.compile(r"^(v\d+)_")


def thumb(path):
    from PIL import Image
    with Image.open(path) as im:
        im.draft("L", (128, 72))  # JPEG: decode at reduced scale
        g = np.asarray(im.convert("L").resize((32, 18), Image.BILINEAR), dtype=np.float32).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def vn_of(path):
    m = VN.match(Path(path).name)
    return m.group(1) if m else None


def test_shots(data, test_dirs):
    if test_dirs:
        return sorted(p for d in test_dirs for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
    out = set()
    for name, platform in sets.SETS.items():
        if platform and (sets.MANIFESTS / f"{name}.jsonl").exists():
            out |= {shot_path(data, platform, r["vn"], r["url"]) for r in sets.read_manifest(name)}
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+", type=Path, help="training screenshot dirs")
    ap.add_argument("--test", action="append", default=[], help="test image dir (default: manifest screenshots)")
    ap.add_argument("--min-corr", type=float, default=0.95)
    ap.add_argument("--keep-reserved", action="store_true", help="do not skip pool images of reserved games")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    tests = test_shots(data_dir(args.data), args.test)
    tset = {p.resolve() for p in tests}
    pool = sorted(p for d in args.pools for p in d.rglob("*")
                  if p.suffix.lower() in EXTS and p.resolve() not in tset)
    n_raw = len(pool)
    if not args.keep_reserved:
        reserved = set(json.loads((sets.MANIFESTS / "test_games.json").read_text(encoding="utf-8"))["all"])
        pool = [p for p in pool if not (vn_of(p) and (vn_of(p) in reserved or is_test_vn(vn_of(p))))]
    print(f"{len(tests)} test screenshots, {len(pool)} pool images ({n_raw - len(pool)} of reserved games skipped)")
    if not tests or not pool:
        return
    corr = np.stack([thumb(p) for p in tests]) @ np.stack([thumb(p) for p in pool]).T
    best = corr.max(1)
    edges = [-1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    hist = np.histogram(best, bins=edges)[0]
    print("best pool correlation per test screenshot: " +
          ", ".join(f"[{lo:.2f},{hi:.2f}):{n}" for lo, hi, n in zip(edges, edges[1:], hist) if n))
    pairs = sorted(((float(corr[i, j]), tests[i], pool[j]) for i, j in zip(*np.nonzero(corr >= args.min_corr))),
                   key=lambda t: -t[0])
    kinds = {"same-vn": 0, "cross-vn": 0}
    for c, t, p in pairs:
        kind = "same-vn" if vn_of(t) and vn_of(t) == vn_of(p) else "cross-vn"
        kinds[kind] += 1
        print(f"  {c:.3f} {kind:8s} {t.name}  ~  {p}")
    print(json.dumps({"tests": len(tests), "pool": len(pool), "min_corr": args.min_corr, "pairs": len(pairs),
                      **kinds, "test_shots_hit": len({str(t) for _, t, _ in pairs})}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The ink-geometry table of docs/synth3.md ("Teacher's character boxes give the real ink geometry").

  ink_geometry.py [--pool $HERD/shared/artifacts/teacher/teacher_train.npz]

Per (capture kind, target) of teacher_train: crops, crop height p50, ink height p10 / p50 / p90 and
the margins above / below the ink as a fraction of the ink height (p50), from the character boxes
teacher stores in each line's meta (`chars`: [char, x0, y0, x1, y1, ...] in crop pixels). The
table was first computed inline; this script recomputes it (the first audit checked every cell).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

POOL = Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts/teacher/teacher_train.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=POOL)
    args = ap.parse_args()
    z = np.load(args.pool)
    meta = json.loads(str(z["meta"]))
    shapes = z["shapes"]
    groups, crops = {}, Counter((m["kind"], m["target"]) for m in meta)
    for m, (h, _) in zip(meta, shapes):
        ch = m.get("chars") or []
        if not ch:
            continue
        top, bottom = min(c[2] for c in ch), max(c[4] for c in ch)
        ink = bottom - top
        if ink <= 0:
            continue
        g = groups.setdefault((m["kind"], m["target"]), {"h": [], "ink": [], "above": [], "below": []})
        g["h"].append(h)
        g["ink"].append(ink)
        g["above"].append(top / ink)
        g["below"].append((h - bottom) / ink)
    print("| kind, target | crops | crop height p50 | ink height p10 / p50 / p90 "
          "| margin above / below ink, x ink height |")
    print("|---|---|---|---|---|")
    for k in sorted(groups):
        g = groups[k]
        q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
        print(f"| {k[0]}, {k[1]} | {crops[k]:,} | {q(g['h'], 50):g} | {q(g['ink'], 10):.1f} / {q(g['ink'], 50):.1f} / "
              f"{q(g['ink'], 90):.1f} | {q(g['above'], 50):.2f} / {q(g['below'], 50):.2f} |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Development line crops as the console's game worker hands them to the recognizer, in vocr_cli's pack format.

  dev_pack.py --shots DIR [--shots DIR ...] --out PACK [--limit N] [--manifests DIR]

For every region of the line finder's development manifests (eval/linefind/manifests vita_dev.jsonl and
psp_dev.jsonl, never the holdout or any test set), as vita-jp-overlay's game worker does on the console:
  1. the region's grey rows, (77 R + 150 G + 29 B) >> 8 (the display hook's conversion), masks filled as
     eval/linefind/build_gt.load_shot fills them;
  2. runtime/vocr_lines.c's GRAD finder on that grey region (the worker's vocr_find_lines_rows gives the same
     lines: runtime/tests/test_vocr_lines_rows.c);
  3. each found line's grey crop, inverted (255 - v) when the finder calls it dark. Inverting the source crop is
     exactly VOCR_INVERT, which reads every source pixel as 255 - grey before the resize (runtime/vocr.h).
Rows of protected sets are refused: every VN of eval/manifests/test_games.json ("all", "vita-test",
"psp-test" and their "family" members) and CLANNAD (v4). The dev manifests hold none of them today; the check
stays so a future manifest cannot slip one in.

The pack holds crops of copyrighted screenshots: write it outside the repository and never commit or share
it. --index writes a pixel-free JSON sidecar (region id, box, dark flag, size, CTC frames per line).
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval" / "linefind")]
import cfinder  # noqa: E402

DEV_MANIFESTS = ("vita_dev.jsonl", "psp_dev.jsonl")
HEIGHT = 32          # every VOCR model's input height
STRIDE = 8           # input columns per CTC frame
MAX_WIDTH = 1024


def protected_vns():
    t = json.loads((ROOT / "eval/manifests/test_games.json").read_text(encoding="utf-8"))
    vns = set(t["all"]) | set(t["vita-test"]) | set(t["psp-test"]) | {"v4"}
    for k, members in t.get("family", {}).items():
        if k in vns or vns.intersection(members):
            vns |= {k, *members}
    return vns


def frames(w, h):
    """CTC frames vocr_run runs for a w x h crop (vocr_resized_width, padded to the stride)."""
    num = w * HEIGHT
    q = (2 * num + h) // (2 * h)
    if (2 * num + h) % (2 * h) == 0 and q & 1:
        q -= 1
    q = min(max(q, 8), MAX_WIDTH)
    return (q + STRIDE - 1) // STRIDE


def load_grey(path, mask):
    a = np.asarray(Image.open(path).convert("RGB")).copy()
    for x0, y0, x1, y1 in mask or []:
        ref = a[max(0, y0 - 6):y0, x0:x1].reshape(-1, 3)
        a[y0:y1, x0:x1] = np.median(ref, axis=0).astype(np.uint8)
    a = a.astype(np.uint32)
    return ((77 * a[..., 0] + 150 * a[..., 1] + 29 * a[..., 2]) >> 8).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shots", action="append", required=True, type=Path, help="directory of the screenshots")
    ap.add_argument("--manifests", type=Path, default=ROOT / "eval/linefind/manifests")
    ap.add_argument("--out", required=True, type=Path, help="the pack (vocr_cli --pack format)")
    ap.add_argument("--index", type=Path, help="pixel-free JSON sidecar")
    ap.add_argument("--limit", type=int, default=0, help="at most this many regions (0: all)")
    args = ap.parse_args(argv)
    bad = protected_vns()
    finder = cfinder.Finder("grad")
    crops, index, refused = [], [], 0
    for name in DEV_MANIFESTS:
        for line in (args.manifests / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["vn"] in bad or str(row.get("set", "")).lower() in ("vita-test", "psp-test", "clannad"):
                refused += 1
                continue
            if args.limit and len(index) >= args.limit:
                break
            shot = next((d / row["shot"] for d in args.shots if (d / row["shot"]).exists()), None)
            if shot is None:
                raise SystemExit(f"{row['shot']} not found in {args.shots}")
            x0, y0, x1, y1 = row["region"]
            g = load_grey(shot, row.get("mask"))[y0:y1, x0:x1]
            boxes = finder(np.ascontiguousarray(g)) or []
            lines = []
            for (x, y, w, h), dark in zip(boxes, finder.last_dark):
                c = np.ascontiguousarray(g[y:y + h, x:x + w])
                if dark:
                    c = 255 - c
                crops.append(c)
                lines.append({"box": [x, y, w, h], "dark": int(dark), "frames": frames(w, h)})
            index.append({"manifest": name, "id": row["id"], "region": [x1 - x0, y1 - y0], "lines": lines})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", len(crops)))
        for c in crops:
            f.write(struct.pack("<III", c.shape[0], c.shape[1], 1))
            f.write(c.tobytes())
    n_frames = sum(ln["frames"] for r in index for ln in r["lines"])
    summary = {"regions": len(index), "lines": len(crops), "frames": n_frames, "refused_protected_rows": refused}
    if args.index:
        args.index.write_text(json.dumps({"summary": summary, "regions": index}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture a local native-resolution Vita3K session, or pack its hand-checked labels.

Use a separate emulator preference directory. Images, labels and NPZs stay local.
The screenshots are emulator captures, never physical-Vita memory/speed evidence.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
import sets  # noqa: E402

from common import line_polarity, to_gray  # noqa: E402


def capture(args):
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DISPLAY=args.display)
    subprocess.run(["xdotool", "windowfocus", args.window], env=env, check=True)
    seen, captures = set(), []
    for i in range(args.count):
        path = args.out / f"{args.session}-{i:04d}.png"
        subprocess.run(["import", "-window", args.window, str(path)], env=env, check=True)
        with Image.open(path) as im:
            if im.size != (960, 544):
                raise ValueError(f"expected native 960x544 window; got {im.size}")
            pixels = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        if pixels in seen:
            path.unlink()
        else:
            seen.add(pixels)
            captures.append({"image": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "session": args.session, "split": args.split, "dims": [960, 544]})
        if i + 1 < args.count:
            subprocess.run(["xdotool", "keydown", args.key], env=env, check=True)
            time.sleep(.18)  # held long enough for the game's controller polling
            subprocess.run(["xdotool", "keyup", args.key], env=env, check=True)
            time.sleep(args.wait)
    (args.out / "captures.json").write_text(json.dumps(captures, indent=2) + "\n")
    print(f"{len(captures)} distinct captures; manually label dialogue boxes before scoring")


def pack(args):
    name = "local-dev"
    sets.register_local(name, args.manifest)
    rows = sets.read_manifest(name)
    if rows[0]["split"] != "dev":
        raise ValueError("only development captures may be packed for checkpoint selection")
    lines, _ = sets.load(name, None)
    crops, labels, meta = [], [], []
    by_id = {r["id"]: r for r in rows}
    for line in lines:
        g = to_gray(line.crop_rgb)
        dark = bool(line_polarity(g))
        crops.append(255 - g if dark else g)
        labels.append(line.text)
        r = by_id[line.id]
        meta.append({"split": r["split"], "session": r["session"], "hand_checked": True,
                     "image_sha256": r["sha256"], "id": r["id"], "inverted": dark})
    shapes = np.array([g.shape for g in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes.prod(axis=1))]).astype(np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, data=np.concatenate([g.ravel() for g in crops]), shapes=shapes,
                        offsets=offsets, labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(meta)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--display", required=True)
    cap.add_argument("--window", required=True)
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--session", required=True)
    cap.add_argument("--split", required=True, choices=("dev", "test"))
    cap.add_argument("--count", type=int, default=20)
    cap.add_argument("--key", default="c")
    cap.add_argument("--wait", type=float, default=2)
    pk = sub.add_parser("pack")
    pk.add_argument("--manifest", type=Path, required=True)
    pk.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "capture":
        capture(args)
    else:
        pack(args)


if __name__ == "__main__":
    main()

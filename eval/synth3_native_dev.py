#!/usr/bin/env python3
"""Development sets of eval/synth3_native_protocol.json without the owner's test series.

  synth3_native_dev.py filter --manifest data/manifests/synth3_exclude_owner_games.json --out DIR NPZ...

The owner's 10:51Z rule reserves the Memories Off series for a fresh native test: nothing may be
selected on it. devnative v3 and teacher_val hold lines of one series VN (v1331, Memories Off After
Rain). This writes a copy of each development file without the rows of any VN of the manifest
(teacher_val's npz format: data, offsets, shapes, labels, meta; rows otherwise byte-identical and in
order) and prints each copy's line and VN counts and sha256.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_vns(path, vns, out):
    """Copy an npz without the rows whose meta vn is in `vns`; returns (kept, dropped)."""
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels, meta = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    keep = [i for i, m in enumerate(meta) if m.get("vn") not in vns]
    crops = [data[offs[i]:offs[i + 1]] for i in keep]
    sh = shapes[keep].astype(shapes.dtype)
    new_offs = np.concatenate([[0], np.cumsum(sh[:, 0].astype(np.int64) * sh[:, 1])]).astype(offs.dtype)
    np.savez_compressed(out, data=np.concatenate(crops).astype(data.dtype) if crops else data[:0], offsets=new_offs,
                        shapes=sh, labels=np.array(json.dumps([labels[i] for i in keep], ensure_ascii=False)),
                        meta=np.array(json.dumps([meta[i] for i in keep], ensure_ascii=False)))
    return len(keep), len(meta) - len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("filter")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("npz", nargs="+")
    a = ap.parse_args()
    vns = set(json.loads(Path(a.manifest).read_text(encoding="utf-8"))["vns"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for f in a.npz:
        out = Path(a.out) / (Path(f).stem + "_noowner.npz")
        kept, dropped = drop_vns(f, vns, out)
        meta = json.loads(str(np.load(out)["meta"]))
        print(json.dumps({"source": f, "source_sha256": sha256(f), "file": str(out), "sha256": sha256(out),
                          "lines": kept, "dropped": dropped, "vns": len({m["vn"] for m in meta})}), flush=True)


if __name__ == "__main__":
    main()

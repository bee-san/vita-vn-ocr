#!/usr/bin/env python3
"""Make a train.py --synth directory from the every-line NVIDIA shards without loading them.

  subset.py --out DIR [--full $HERD/shared/artifacts/nvsynth/full] [--buckets 0]
            [--sources 0-66] [--val validation:0] [--extra 'GLOB' ...]

Each source file's lines are spread over its shards nv_train_NNN_bBB.npz by a hash of
(file, page, line), so bucket BB of every source is a uniform sample of all 1.5M pages.
--buckets picks which buckets (e.g. 0, or 0-2 for 3/12 of the lines), --sources which
train files. --val adds one held-out shard as zz_val.npz, which sorts after nv_* and
synth_*, so train.py (it validates on the last shard of its sorted glob) validates on
it: 'validation:N' is bucket 0 of ja/validation file N, or give a path. --extra links
other shards too (e.g. synth2's), which then train alongside. Prints the line count.

  python train/train.py --synth "DIR/*.npz" ...
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nv_fetch as fetch  # noqa: E402


def records(split_dir):
    recs = {Path(p).stem: json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{split_dir}/nv_*.json"))}
    return {k: r for k, r in recs.items() if "error" not in r}   # unreadable sources have no shards


def shard_list(rec):
    return rec.get("shards") or [{"shard": rec["shard"], "lines": rec["lines"]}]


def bucket(name):
    stem = Path(name).stem
    return int(stem.rsplit("_b", 1)[1]) if "_b" in stem else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", default=str(fetch.ARTIFACTS / "full"))
    ap.add_argument("--buckets", default="0")
    ap.add_argument("--sources", default="")
    ap.add_argument("--val", default="validation:0")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    full, out = Path(args.full), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = records(full / "train")
    want_src = set(fetch.parse_range(args.sources, 67))
    want_b = set(fetch.parse_range(args.buckets, 100))
    n_lines = n_links = 0
    for tag, rec in recs.items():
        if int(tag.rsplit("_", 1)[1]) not in want_src:
            continue
        for s in shard_list(rec):
            if bucket(s["shard"]) in want_b:
                link = out / s["shard"]
                if not link.exists():
                    os.symlink(full / "train" / s["shard"], link)
                n_lines += s["lines"]
                n_links += 1
    for pattern in args.extra:
        for p in sorted(glob.glob(pattern)):
            link = out / Path(p).name
            if not link.exists():
                os.symlink(Path(p).resolve(), link)
            n_links += 1
    if args.val:
        if args.val.startswith("validation:"):
            n = int(args.val.split(":", 1)[1])
            rec = records(full / "validation")[f"nv_validation_{n:03d}"]
            val = full / "validation" / sorted(s["shard"] for s in shard_list(rec))[0]
        else:
            val = Path(args.val)
        link = out / "zz_val.npz"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(val.resolve(), link)
        print(f"validation shard: {val}")
    print(f"{out}: {n_links} shards, {n_lines:,} NVIDIA train lines (plus any --extra shards)")


if __name__ == "__main__":
    main()

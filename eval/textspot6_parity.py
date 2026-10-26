#!/usr/bin/env python3
"""textspot6's C == twin check on the scored development data (docs/textspot6.md "The rule": eligible needs the C
runtime to give the twin's result on every development frame): runtime/vocr_textspot6.c through its host driver
(runtime/vocr_textspot6_test --batch) against synth/textspot6_twin.spot, field for field (rc, every box with its tag
and tconf, every block with its tag, pick, truncated, presence).

  textspot6_parity.py --blob B.vtsp --out PARITY.json [--sets dev,dev-crop,tsdev,hval,hval-crop] [--units HVAL_UNITS]
                      [--cbin DIR] [--workers 5]

Sets: dev (the rebuilt console-view frames of #119's units, as eval/textspot6.py spot reads them), dev-crop (each
unit's region at its frame's scale, VOCR_TEXTSPOT6_SCALE1 / _SCALE2 as eval/textspot6.py spot-crop reads it),
tsdev (textspot's 188 development frames), hval (the hval pages in their views) and hval-crop (the hval region
units). Records go to the driver in batches; the twin runs in worker processes. PARITY.json holds counts only.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6 as E6  # noqa: E402

SCALE_FLAG = {1: 32, 2: 64}            # the driver's record flags for VOCR_TEXTSPOT6_SCALE1 / _SCALE2
SRC = ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot6.c", "vocr_textspot6_test.c")


def build(cdir):
    cdir = Path(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    exe = cdir / "vocr_textspot6_test"
    h = hashlib.sha256()
    for s in SRC + ("vocr_textspot6.h", "vocr_textspot.h", "vocr_lines.h", "vocr_lines_rows.h"):
        h.update((ROOT / "runtime" / s).read_bytes())
    tag = cdir / "sources.sha256"
    if exe.exists() and tag.exists() and tag.read_text().strip() == h.hexdigest():
        return exe, h.hexdigest()
    cc = os.environ.get("CC", "cc")
    subprocess.run([cc, "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    *[str(ROOT / "runtime" / s) for s in SRC], "-o", str(exe)], check=True)
    tag.write_text(h.hexdigest() + "\n")
    return exe, h.hexdigest()


def jobs_of(set_name, units=None):
    """[(key, grey, scale or None)] of a set, with the scale the scoring used (None: the blob's rule)."""
    import textspot6_twin as T6
    if set_name == "dev":
        for k, g in E6.dev_frames():
            yield k["frame"], g, None
    elif set_name == "tsdev":
        for k, g in E6.tsdev_frames():
            yield k["frame"], g, None
    elif set_name == "hval":
        for k, g in E6.hval_frames():
            yield f"{k['title']}:{k['page']}:{k['view']}", g, None
    elif set_name in ("dev-crop", "hval-crop"):
        m = _W["model"]
        for uid, img, fw, fh in E6.crop_jobs("dev" if set_name == "dev-crop" else "hval", units):
            yield uid, img, T6.scale_of(m, fw, fh)
    else:
        raise SystemExit(f"unknown set {set_name}")


_W = {}


def twin_line(job):
    import textspot6_twin as T6
    key, g, scale = job
    r = T6.spot(_W["model"], g, scale=scale)
    parts = [0, len(r.boxes), len(r.blocks), r.pick, r.truncated, r.presence]
    for b in r.boxes:
        parts += [b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, b.tag, b.tconf]
    for k in r.blocks:
        parts += [k.x, k.y, k.w, k.h, k.vertical, k.first, k.count, k.tag]
    return key, " ".join(str(int(v)) for v in parts)


def c_lines(exe, blob_path, jobs):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for _, g, scale in jobs:
            g = np.ascontiguousarray(g, np.uint8)
            fl = SCALE_FLAG[scale] if scale else 0
            f.write(struct.pack("<4I", g.shape[1], g.shape[0], 1, fl) + g.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(exe), "--batch", str(blob_path), path], capture_output=True, text=True,
                             check=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(out) != len(jobs):
        raise SystemExit(f"the driver printed {len(out)} lines for {len(jobs)} records")
    return [x.strip() for x in out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="dev,dev-crop,tsdev,hval,hval-crop")
    ap.add_argument("--units", default=str(E6.ART / "vocr6-spotter" / "eval" / "units" / "hval_v1.jsonl"))
    ap.add_argument("--cbin", default=str(E6.ART / "vocr6-spotter" / "eval" / "cbuild6"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "5")))
    ap.add_argument("--batch", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: the first N records of each set")
    a = ap.parse_args(argv)
    import textspot6_twin as T6
    blob = Path(a.blob).read_bytes()
    _W["model"] = T6.Model(blob)
    exe, src_sha = build(a.cbin)
    t0 = time.time()
    res = {"kind": "textspot6-parity", "blob": str(a.blob), "blob_sha256": hashlib.sha256(blob).hexdigest(),
           "runtime_sources_sha256": src_sha, "sets": {}}
    for s in a.sets.split(","):
        jobs = list(jobs_of(s, a.units))
        if a.limit:
            jobs = jobs[:a.limit]
        n_eq = n = 0
        first_diff = None
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k in range(0, len(jobs), a.batch):
                chunk = jobs[k:k + a.batch]
                tw = pool.map(twin_line, chunk, chunksize=8)
                cl = c_lines(exe, a.blob, chunk)
                for (key, t), c in zip(tw, cl):
                    n += 1
                    if t == c:
                        n_eq += 1
                    elif first_diff is None:
                        first_diff = {"key": key, "twin": t[:300], "c": c[:300]}
        res["sets"][s] = {"records": n, "equal": n_eq, "all_equal": n == n_eq, "first_difference": first_diff}
        print(f"{s}: {n_eq}/{n} equal", flush=True)
    res["all_equal"] = all(v["all_equal"] for v in res["sets"].values())
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("parity", "equal" if res["all_equal"] else "DIFFERS", flush=True)


if __name__ == "__main__":
    main()

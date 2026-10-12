#!/usr/bin/env python3
"""Copy real crop pools without the rows of given VNs, together with their aligned teacher targets.

  synth3_filter_pools.py --vns v4 v12 ... --out DIR [--targets-in TDIR --targets-out TDIR2] POOL.npz ...

Each POOL.npz (teacher's / realdata's format: data, offsets, shapes, labels, meta) is written to
DIR/POOL.npz without the rows whose meta `vn` is in --vns; every other row is kept in order, byte
for byte. If TDIR holds POOL's teacher-target file (train/teacher_targets.py: POOL.v6s.npz, one row
per crop: text, stext, conf, T, off/kid/kp frame targets, soft_ok), the same rows are dropped from
it into TDIR2, so train2.py's `targets=` option reads the copy exactly as it reads the original.
A JSON record (input and output sha256, rows dropped per VN) goes to DIR/filter_record.json.

Used by eval/synth3_study2_protocol.json's amendment: the fine-tune pools without CLANNAD (v4) and
its VNDB family, which the protocols and the fleet rules keep out of training.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def keep_rows(z, keep):
    """Crop arrays of the kept rows (data re-packed, offsets rebuilt)."""
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]     # each z[...] reads the file: once only
    idx = np.flatnonzero(keep)
    parts = [data[offs[i]:offs[i + 1]] for i in idx]
    sizes = np.array([len(p) for p in parts], np.int64)
    new_offs = np.concatenate([[0], np.cumsum(sizes)]).astype(offs.dtype)
    data = np.concatenate(parts) if parts else np.zeros(0, data.dtype)
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    return {"data": data, "offsets": new_offs, "shapes": shapes[idx],
            "labels": np.array(json.dumps([labels[i] for i in idx], ensure_ascii=False)),
            "meta": np.array(json.dumps([meta[i] for i in idx], ensure_ascii=False))}


def keep_targets(t, keep):
    t = {k: t[k] for k in t.files}                                  # read every array once
    idx = np.flatnonzero(keep)
    off = t["off"]
    rows = [np.arange(off[i], off[i + 1]) for i in idx]
    frames = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    sizes = np.array([len(r) for r in rows], np.int64)
    out = {"off": np.concatenate([[0], np.cumsum(sizes)]).astype(off.dtype),
           "kid": t["kid"][frames], "kp": t["kp"][frames]}
    for k in ("text", "stext"):
        vals = json.loads(str(t[k]))
        out[k] = np.array(json.dumps([vals[i] for i in idx], ensure_ascii=False))
    for k in ("conf", "T", "soft_ok"):
        out[k] = t[k][idx]
    out["info"] = t["info"]
    return out


def save(path, arrays, compressed):
    tmp = path.with_name(path.name + ".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    tmp.rename(path)


def filter_pool(pool, vns, out_dir, targets_in=None, targets_out=None):
    pool = Path(pool)
    z = np.load(pool)
    meta = json.loads(str(z["meta"]))
    keep = np.array([m.get("vn") not in vns for m in meta], bool)
    dropped = Counter(m.get("vn") for m, k in zip(meta, keep) if not k)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / pool.name
    save(out, keep_rows(z, keep), compressed=True)
    rec = {"pool": str(pool), "sha256": sha256(pool), "rows": len(meta), "out": str(out),
           "out_sha256": sha256(out), "out_rows": int(keep.sum()), "dropped": dict(sorted(dropped.items()))}
    if targets_in:
        tp = Path(targets_in) / (pool.stem + ".v6s.npz")
        if tp.exists():
            t = np.load(tp)
            if len(t["T"]) != len(meta):
                raise SystemExit(f"{tp}: {len(t['T'])} target rows for {len(meta)} crops")
            Path(targets_out).mkdir(parents=True, exist_ok=True)
            to = Path(targets_out) / tp.name
            save(to, keep_targets(t, keep), compressed=False)
            rec.update(targets=str(tp), targets_sha256=sha256(tp), targets_out=str(to),
                       targets_out_sha256=sha256(to))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--vns", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets-in", default="")
    ap.add_argument("--targets-out", default="")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.targets_out:
        Path(args.targets_out).mkdir(parents=True, exist_ok=True)
    recs = [filter_pool(p, set(args.vns), args.out, args.targets_in or None, args.targets_out or None)
            for p in args.pools]
    doc = {"kind": "synth3-filtered-pools", "vns": sorted(set(args.vns)), "pools": recs}
    (Path(args.out) / "filter_record.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                                                       encoding="utf-8")
    for r in recs:
        print(f"{Path(r['pool']).name}: {r['rows']} -> {r['out_rows']} rows, dropped {r['dropped']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""s3e: version 3's pinned shards with their native console crops replaced by version 4's (study
eval/synth3_native_protocol.json).

  synth3_native_merge.py merge --v3 DIR --native DIR --out DIR [--workers N]
  synth3_native_merge.py check --v3 DIR --native DIR --out DIR

Shard i of s3e is version 3's shard i, crop for crop: a crop of platform "scaled" (a PC render area-scaled
to the console, 47%) is kept byte for byte; every crop of platform vita, psp or psp2x is replaced, in
place and in order, by the next crop of version 4's native shard i (native console frames). Both arms then
stream 1,000,000 crops in the same 100 x 10,000 positions and differ only in how the native ones were
rendered. merge writes s3e_DIR/synth3e_NNNNN.npz and recipe.json (the inputs' recipes and sha256, the
per-shard replacement counts); check re-derives every merged shard from its inputs and compares.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

NATIVE = {"vita", "psp", "psp2x"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    z = np.load(path)
    return {"data": z["data"], "offsets": z["offsets"], "shapes": z["shapes"],
            "labels": json.loads(str(z["labels"])), "meta": json.loads(str(z["meta"]))}


def crop(s, i):
    h, w = (int(v) for v in s["shapes"][i])
    return s["data"][s["offsets"][i]:s["offsets"][i + 1]].reshape(h, w)


def merge_one(v3, nat):
    """(crops, labels, metas, replaced): v3's crops with each native-platform crop swapped for the next
    of `nat` (which must hold at least as many)."""
    crops, labels, metas = [], [], []
    k = 0
    for i, m in enumerate(v3["meta"]):
        if m["platform"] in NATIVE:
            if k >= len(nat["meta"]):
                raise ValueError(f"native shard holds {len(nat['meta'])} crops, version 3 needs more")
            if nat["meta"][k]["platform"] not in NATIVE:
                raise ValueError("a native shard crop is not of a native platform")
            crops.append(crop(nat, k))
            labels.append(nat["labels"][k])
            metas.append(nat["meta"][k])
            k += 1
        else:
            crops.append(crop(v3, i))
            labels.append(v3["labels"][i])
            metas.append(m)
    return crops, labels, metas, k


def save(path, crops, labels, metas):
    shapes = np.array([c.shape for c in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    tmp = Path(path).with_name(Path(path).name + ".tmp.npz")
    np.savez_compressed(tmp, data=np.concatenate([c.ravel() for c in crops]), offsets=offsets, shapes=shapes,
                        labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(metas, ensure_ascii=False)))
    tmp.rename(path)


def _job(args):
    i, v3_dir, nat_dir, out_dir = args
    v3p, natp = Path(v3_dir) / f"synth3_{i:05d}.npz", Path(nat_dir) / f"synth3_{i:05d}.npz"
    outp = Path(out_dir) / f"synth3e_{i:05d}.npz"
    crops, labels, metas, k = merge_one(load(v3p), load(natp))
    if not outp.exists():
        save(outp, crops, labels, metas)
    return i, {"v3": v3p.name, "v3_sha256": sha256(v3p), "native": natp.name, "native_sha256": sha256(natp),
               "native_crops": len(load(natp)["meta"]), "replaced": k, "kept": len(crops) - k,
               "file": outp.name, "sha256": sha256(outp), "samples": len(crops)}


def cmd_merge(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = sorted(int(p.stem.split("_")[1]) for p in Path(args.v3).glob("synth3_?????.npz"))
    jobs = [(i, args.v3, args.native, str(out)) for i in idx]
    with mp.get_context("fork").Pool(args.workers) as pool:
        rows = dict(pool.imap_unordered(_job, jobs))
    rec = {"kind": "synth3-e-merge", "schema": 1,
           "rule": "version 3's shard i with every vita / psp / psp2x crop replaced in place, in order, by the next "
                   "crop of version 4's native shard i; 'scaled' crops kept byte for byte",
           "v3": {"dir": str(args.v3),
                  "recipe": json.loads((Path(args.v3) / "recipe.json").read_text(encoding="utf-8"))},
           "native": {"dir": str(args.native),
                      "recipe": json.loads((Path(args.native) / "recipe.json").read_text(encoding="utf-8"))},
           "shards": [rows[i] for i in idx],
           "replaced": sum(r["replaced"] for r in rows.values()), "kept": sum(r["kept"] for r in rows.values())}
    (out / "recipe.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(idx)} shards: {rec['replaced']} native crops replaced, {rec['kept']} kept -> {out}", flush=True)


def cmd_check(args):
    rec = json.loads((Path(args.out) / "recipe.json").read_text(encoding="utf-8"))
    bad = 0
    for r in rec["shards"]:
        i = int(r["file"].split("_")[1].split(".")[0])
        crops, labels, metas, k = merge_one(load(Path(args.v3) / r["v3"]), load(Path(args.native) / r["native"]))
        got = load(Path(args.out) / r["file"])
        same = (k == r["replaced"] and labels == got["labels"] and metas == got["meta"]
                and all(np.array_equal(c, crop(got, j)) for j, c in enumerate(crops)))
        bad += not same
        if not same:
            print(f"shard {i}: differs", flush=True)
    print("MERGE_CHECK_OK" if not bad else f"MERGE_CHECK_FAILED {bad}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("merge", "check"):
        p = sub.add_parser(c)
        p.add_argument("--v3", required=True)
        p.add_argument("--native", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    {"merge": cmd_merge, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()

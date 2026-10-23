#!/usr/bin/env python3
"""head2level: the training-side frames of the escape (eval/head2level_protocol.json "escape_training").

  head2level_extract.py train --tier N --blob PERM.vocr --out PRIV/train_N.npz [--limit 200]

Rows: every row of head-gate's eight console-cut files (own_grad, own_ref, nat_grad, nat_ref, dntv_grad, dntv_ref,
dnst_grad, dnst_ref) and a seeded 40,000-row sample of rp3_grad.npz, as stored (console polarity), every row
(dialogue and names); the pinned hashes and the reservation are asserted first. Each crop runs through the C
runtime (W8I, max width 1024) with the tier's permuted blob (train/head2level.py permute): per frame the head's
input (vocr_extra.head_in, as the head reads it), the argmax (permuted class), the blank logit b0 and, for every K
of the protocol's grid, level 1's largest non-blank logit l1 (level 1 = the first ceil((K + 1) / 8) blocks). No
text label is used. The output holds features only (no pixels, no text) and stays private: the owner's captures.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "runtime", ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import head2level as H2  # noqa: E402

KGRID = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500]
CONSOLE_FILES = ["shared/private/vocr5-train/pools/v3/own_grad.npz", "shared/private/vocr5-train/pools/v3/own_ref.npz",
                 "shared/artifacts/vocr5-train/pools/v3/nat_grad.npz",
                 "shared/artifacts/vocr5-train/pools/v3/nat_ref.npz",
                 "shared/artifacts/vocr5-train/pools/v3/dntv_grad.npz",
                 "shared/artifacts/vocr5-train/pools/v3/dntv_ref.npz",
                 "shared/artifacts/vocr5-train/pools/v3/dnst_grad.npz",
                 "shared/artifacts/vocr5-train/pools/v3/dnst_ref.npz"]
RP3 = "shared/artifacts/vocr5-train/pools/v3/rp3_grad.npz"
RP3_N, RP3_SEED = 40000, 20261023


def l1_rows(K):
    return 8 * ((K + 1 + 7) // 8)


def calib(vn):
    return int(hashlib.sha1(vn.encode()).hexdigest(), 16) % 5 == 0


def load_rows(limit=0):
    """[(file tag, row, vn, kind, name, crop)] in a fixed order; asserts pins and the reservation."""
    from common import is_reserved_vn
    pins = H2.pinned_hashes()
    out = []
    for rel in CONSOLE_FILES + [RP3]:
        f = str(H2.HERD / rel)
        if pins.get(f) != H2.sha256(f):
            raise SystemExit(f"{f}: not the file eval/vocr5_pools.sha256 pins")
        z = np.load(f)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        meta = json.loads(str(z["meta"]))
        rows = range(len(meta))
        if rel == RP3:
            rows = sorted(np.random.default_rng(RP3_SEED).choice(len(meta), RP3_N, replace=False).tolist())
        for i in rows:
            m = meta[i]
            if is_reserved_vn(m["vn"]):
                raise SystemExit(f"{f}: row {i} of reserved VN {m['vn']}")
            h, w = (int(v) for v in shapes[i])
            out.append((Path(f).stem, int(i), m["vn"], m.get("kind"), bool(m.get("name")),
                        np.ascontiguousarray(data[offs[i]:offs[i + 1]].reshape(h, w))))
    if limit:
        out = out[::max(1, len(out) // limit)][:limit]
    return out


W = {}


def _init(blob):
    from pyvocr import Vocr
    W["rt"] = Vocr(blob, mode="w8i", max_width=1024)
    W["L1"] = [l1_rows(K) for K in KGRID]


def _work(job):
    i, img = job
    rt = W["rt"]
    if img.shape[0] < 2 or img.shape[1] < 2:
        return i, None
    text, lg = rt.run(img, logits=True, head_in=True)
    x = rt.last_head_in
    am = lg.argmax(1)
    assert np.array_equal(am, rt.last_best), "argmax differs from vocr_run's"
    b0 = lg[:, 0].astype(np.float32)
    l1 = np.stack([lg[:, 1:L].max(1) for L in W["L1"]], 1).astype(np.float32)
    return i, (x.astype(np.float32), am.astype(np.uint16), b0, l1)


def cmd_train(a):
    t0 = time.time()
    rows = load_rows(a.limit)
    jobs = [(i, r[5]) for i, r in enumerate(rows)]
    res = [None] * len(jobs)
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.blob,)) as pool:
        for n, (i, o) in enumerate(pool.imap_unordered(_work, jobs, chunksize=16)):
            res[i] = o
            if n % 5000 == 0:
                print(f"  {a.tier}: {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    keep = [i for i, o in enumerate(res) if o is not None]
    off = np.zeros(len(keep) + 1, np.int64)
    off[1:] = np.cumsum([len(res[i][1]) for i in keep])
    meta = {"src": [rows[i][0] for i in keep], "row": [rows[i][1] for i in keep], "vn": [rows[i][2] for i in keep],
            "kind": [rows[i][3] for i in keep], "name": [rows[i][4] for i in keep],
            "cal": [calib(rows[i][2]) for i in keep]}
    np.savez(a.out, off=off, x=np.concatenate([res[i][0] for i in keep]), am=np.concatenate([res[i][1] for i in keep]),
             b0=np.concatenate([res[i][2] for i in keep]), l1=np.concatenate([res[i][3] for i in keep]),
             kgrid=np.array(KGRID), meta=np.array(json.dumps(meta)))
    info = {"tier": a.tier, "blob": a.blob, "blob_sha256": H2.sha256(a.blob), "lines": len(keep),
            "skipped_tiny": len(rows) - len(keep), "frames": int(off[-1]), "vns": len(set(meta["vn"])),
            "cal_vns": len({v for v, c in zip(meta["vn"], meta["cal"]) if c}),
            "cal_frames": int(sum(off[k + 1] - off[k] for k in range(len(keep)) if meta["cal"][k])),
            "by_src": {s: meta["src"].count(s) for s in sorted(set(meta["src"]))},
            "seconds": round(time.time() - t0, 1)}
    Path(a.out + ".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


def _work_x(job):
    i, img = job
    W["rt"].run(img, head_in=True)
    x = W["rt"].last_head_in
    return i, x.astype(np.float32)


SYNTH = {"shared/artifacts/vocr5-train/render/s4v5631/shards/synth4_00097.npz":
         "c31c6f7be8ed3d7ad97fd26dacb91a45331f132431d6858429dcf3b601ac1e30",
         "shared/artifacts/vocr5-train/render/s4v5631/shards/synth4_00098.npz":
         "fe61282dd898f9db7b9e6dc2afdf831ce2c12fe79ed49e0273d0f0bcb6ce2712",
         "shared/artifacts/vocr5-train/render/s4v5631/shards/synth4_00099.npz":
         "1821662f3f075a6fc17ad601bf6f5d1babddaf1eb7b4e9c5333b51d25f48d98b"}


def cmd_synth(a):
    """Head inputs of the first 10,000 crops of each of the protocol's three synth4 shards (r48's KD data)."""
    t0 = time.time()
    imgs = []
    for rel, h in SYNTH.items():
        f = str(H2.HERD / rel)
        if H2.sha256(f) != h:
            raise SystemExit(f"{f}: not the protocol's file")
        z = np.load(f)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        for i in range(min(10000, len(shapes))):
            hh, ww = (int(v) for v in shapes[i])
            imgs.append(np.ascontiguousarray(data[offs[i]:offs[i + 1]].reshape(hh, ww)))
    if a.limit:
        imgs = imgs[:a.limit]
    res = [None] * len(imgs)
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.blob,)) as pool:
        for n, (i, x) in enumerate(pool.imap_unordered(_work_x, list(enumerate(imgs)), chunksize=16)):
            res[i] = x
            if n % 5000 == 0:
                print(f"  {a.tier} synth: {n}/{len(imgs)} {time.time() - t0:.0f}s", flush=True)
    off = np.zeros(len(res) + 1, np.int64)
    off[1:] = np.cumsum([len(x) for x in res])
    np.savez(a.out, off=off, x=np.concatenate(res))
    info = {"tier": a.tier, "blob_sha256": H2.sha256(a.blob), "crops": len(res), "frames": int(off[-1]),
            "files": SYNTH, "seconds": round(time.time() - t0, 1)}
    Path(a.out + ".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "synth"):
        p = sub.add_parser(name)
        p.add_argument("--tier", required=True)
        p.add_argument("--blob", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    {"train": cmd_train, "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    main()

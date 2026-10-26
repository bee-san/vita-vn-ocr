#!/usr/bin/env python3
"""VOCR6 T1's hval guard (issue #125, eval/vocr6_protocol.json "data_pins.guarded"): training pools other than the
hooked corpus may hold the very lines of an hval block, captured another time (the owner's console captures of the
same scenes). For screening runs, every such row is taken out of a copy of its pool; final models (hval folded back
in) use the pools as they are.

  vocr6_guard.py --hval-plan HVAL_PLAN.pkl --vn-of POOLS_HVAL_DIR --pool NAME=IN.npz=OUT.npz ... --record REC.json

A row is dropped when its VNDB id is a hooked title's and its label, eval/metrics.norm'd, equals one of that title's
hval dialogue or name lines. Everything else is copied byte for byte (same order, same meta).
"""
import argparse
import glob
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import norm  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hval-plan", required=True)
    ap.add_argument("--vn-of", required=True, help="vocr6-ingest's pools/<v>/hval directory (title -> VNDB id)")
    ap.add_argument("--pool", action="append", required=True, help="NAME=IN.npz=OUT.npz")
    ap.add_argument("--record", required=True)
    a = ap.parse_args()
    t2v = {}
    for f in glob.glob(str(Path(a.vn_of) / "*" / "*.npz")):
        z = np.load(f)
        for m in json.loads(str(z["meta"])):
            if m.get("vn"):
                t2v[m["title_id"]] = m["vn"]
    plan = pickle.load(open(a.hval_plan, "rb"))
    texts = {}
    for f in plan["fixed"]:
        v = t2v.get(f["title"])
        if v:
            texts.setdefault(v, set()).add(norm(f["ref"]))
    rec = {"kind": "vocr6-hval-guard", "hval_plan_sha256": sha256(a.hval_plan), "vns": sorted(texts),
           "hval_texts": sum(len(s) for s in texts.values()), "pools": {}}
    for spec in a.pool:
        name, src, dst = spec.split("=")
        z = np.load(src)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        labels, meta = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
        keep = [i for i, (lab, m) in enumerate(zip(labels, meta))
                if not (lab and m.get("vn") in texts and norm(lab) in texts[m["vn"]])]
        ks = set(keep)
        dropped = Counter(meta[i].get("vn") for i in range(len(labels)) if i not in ks)
        parts = [data[offs[i]:offs[i + 1]] for i in keep]
        out_data = np.concatenate(parts) if parts else np.zeros(0, np.uint8)
        out_offs = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(offs.dtype)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, data=out_data, offsets=out_offs, shapes=shapes[keep],
                 labels=np.asarray(json.dumps([labels[i] for i in keep], ensure_ascii=False)),
                 meta=np.asarray(json.dumps([meta[i] for i in keep], ensure_ascii=False)))
        rec["pools"][name] = {"in": src, "in_sha256": sha256(src), "out": dst, "out_sha256": sha256(dst),
                              "rows_in": len(labels), "rows_out": len(keep), "dropped_by_vn": dict(dropped)}
        print(f"{name}: {len(labels)} -> {len(keep)} rows ({dict(dropped)})", flush=True)
    Path(a.record).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("GUARD_DONE", a.record)


if __name__ == "__main__":
    main()

"""Shard files of the desktop data, in the two formats the repository's detector trainers read unchanged:

  textspot6-pages (synth/textspot6_pages.py; train/textspot6_train.HookPool): SHARDS/<split>/<name>.{npy,json}, a
      flat uint8 grey array and {"samples": [...]}, one sample per page: off, h, w, src "hook", kind, vn, shot,
      title, page, split, lines [[x0, y0, x1, y1, vertical, class, weight]], ignore [[x0, y0, x1, y1]], meta
  #97's frames (synth/textdet_data.py; train/textspot_train.FramePool): DIR/<name>.{npy,json}, the same arrays,
      samples with off, h, w, src, kind, vn, shot, lines [[x0, y0, x1, y1, vertical, role(, weight)]], ignore, meta

Boxes are [x0, x1) x [y0, y1) in the canvas's pixels. The grey is integer luma ((77 R + 150 G + 29 B) >> 8, the
runtime's and synth/common.to_gray's). A manifest pins every shard's npy and json sha256 with its counts.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def to_gray(rgb):
    """Integer luma, identical to synth/common.to_gray and runtime/vocr.c."""
    a = np.asarray(rgb)
    r, g, b = (a[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def write_shard(path, samples, greys):
    """Write <path>.npy (the greys concatenated, row-major) and <path>.json ({"samples"}): each sample gets its "off",
    "h", "w" from its grey. Atomic (temporary files, then renamed). Returns (npy sha256, json sha256)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    off = 0
    blobs = []
    for s, g in zip(samples, greys):
        g = np.ascontiguousarray(g, np.uint8)
        s["off"], s["h"], s["w"] = off, int(g.shape[0]), int(g.shape[1])
        off += g.size
        blobs.append(g.reshape(-1))
    arr = np.concatenate(blobs) if blobs else np.zeros(0, np.uint8)
    tmp_npy = path.with_name(path.name + ".npy.part")
    with open(tmp_npy, "wb") as f:
        np.save(f, arr)
    tmp_js = path.with_name(path.name + ".json.part")
    tmp_js.write_text(json.dumps({"samples": samples}, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_npy, path.with_suffix(".npy"))
    os.replace(tmp_js, path.with_suffix(".json"))
    return sha256_file(path.with_suffix(".npy")), sha256_file(path.with_suffix(".json"))


def read_shard(path):
    """(memory-mapped data, samples) of a shard written by write_shard."""
    path = Path(path)
    data = np.load(path.with_suffix(".npy"), mmap_mode="r")
    samples = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["samples"]
    return data, samples


def canvas(data, s):
    return np.asarray(data[s["off"]:s["off"] + s["h"] * s["w"]]).reshape(s["h"], s["w"])


def shard_record(path, dir_name, extra=None):
    """A manifest entry for one shard: names, sha256, counts (samples, lines, ignore, by class, by size)."""
    path = Path(path)
    samples = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["samples"]
    by_class, by_size = {}, {}
    for s in samples:
        for ln in s["lines"]:
            by_class[ln[5]] = by_class.get(ln[5], 0) + 1
        k = f"{s['w']}x{s['h']}"
        by_size[k] = by_size.get(k, 0) + 1
    rec = {"dir": dir_name, "name": path.name, "npy_sha256": sha256_file(path.with_suffix(".npy")),
           "json_sha256": sha256_file(path.with_suffix(".json")), "samples": len(samples),
           "lines": sum(len(s["lines"]) for s in samples), "vertical": sum(int(ln[4]) for s in samples
                                                                         for ln in s["lines"]),
           "ignore": sum(len(s["ignore"]) for s in samples), "by_class": by_class, "by_size": by_size,
           "pixels": int(sum(s["w"] * s["h"] for s in samples))}
    if extra:
        rec.update(extra)
    return rec


def totals(shards, key=None):
    """Sum of the shard records' counts (by `key` when given, e.g. "split")."""
    out = {}
    for r in shards:
        k = r.get(key) if key else "all"
        t = out.setdefault(k, {"shards": 0, "samples": 0, "lines": 0, "ignore": 0, "pixels": 0, "by_class": {},
                               "by_size": {}})
        t["shards"] += 1
        for f in ("samples", "lines", "ignore", "pixels"):
            t[f] += r[f]
        for f in ("by_class", "by_size"):
            for a, b in r[f].items():
                t[f][a] = t[f].get(a, 0) + b
    return out


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha256(root, files):
    return {f: sha256_file(Path(root) / f) for f in files if (Path(root) / f).exists()}

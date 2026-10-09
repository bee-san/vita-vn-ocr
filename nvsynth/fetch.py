#!/usr/bin/env python3
"""Download the Japanese shards of NVIDIA OCR-Synthetic-Multilingual-v1 one by one,
cut their text lines into train.py shards, and delete each raw file afterwards.

  fetch.py run --split train|validation [--files 0-66] [--per-page 1] [--seed S]
               [--out DIR] [--raw DIR] [--max-raw-gb 32] [--downloads 2] [--procs N]
      For every ja/<split>/<split>_NNN.h5 at the pinned REVISION: download it with
      huggingface_hub, check its sha256 against the Hub's LFS record, crop up to
      --per-page usable lines per page (nvsynth/extract.py) into
      DIR/nv_<split>_NNN.npz plus a .json of counts (validation: DIR/val_parts/),
      then delete the raw file.
      Downloads run ahead of the extraction, but the raw files on disk (finished
      or in flight) never exceed --max-raw-gb. Resumable: finished shards are
      skipped and a complete raw file is not downloaded again.
  fetch.py merge-val [--out DIR]
      Concatenates DIR/val_parts/nv_validation_*.npz into DIR/nv_val.npz. With
      train.py's convention (the last shard of the --synth glob is the validation
      shard), --synth 'DIR/nv_*.npz' trains on nv_train_* and validates on nv_val.npz.
  fetch.py manifest [--out DIR] [--write nvsynth/manifests/nvsynth_ja.json]
      Collects the per-shard counts, sha256s and OOV statistics into the manifest.

The dataset is CC BY 4.0 (NVIDIA); see README.md for the attribution. Raw shards
are about 9.8 GB (train) and 1.2 GB (validation) each. Threads follow
OMP_NUM_THREADS (heavy.sh sets 5) unless --procs is given.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract as ex  # noqa: E402

REPO = "nvidia/OCR-Synthetic-Multilingual-v1"
REVISION = "69696a1cc543ef3a0f8e9892a89c17293e915263"     # main on 2026-10-09 (last change 2026-04-20)
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
DEFAULT_OUT = HERD / "shared" / "artifacts" / "nvsynth" / "shards"
DEFAULT_SEED = 20261009
PAGES_PER_TASK = 500


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def hub_files(split):
    """{path: (size, lfs sha256)} for ja/<split>/*.h5 at REVISION."""
    from huggingface_hub import HfApi
    out = {}
    for e in HfApi().list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"ja/{split}", revision=REVISION):
        lfs = getattr(e, "lfs", None)
        if e.path.endswith(".h5") and lfs is not None:
            out[e.path] = (int(e.size), lfs.sha256)
    return out


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def parse_range(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


class Budget:
    """Bytes of raw shards allowed on disk at once (finished downloads plus in-flight ones)."""

    def __init__(self, limit):
        self.limit, self.used, self.cond = limit, 0, threading.Condition()

    def take(self, n):
        with self.cond:
            while self.used and self.used + n > self.limit:
                self.cond.wait()
            self.used += n

    def give(self, n):
        with self.cond:
            self.used -= n
            self.cond.notify_all()


def download(path, size, sha, raw, budget):
    """Fetch one shard into raw/ (unless already complete) and verify its sha256."""
    from huggingface_hub import hf_hub_download
    budget.take(size)
    local = raw / path
    t0 = time.time()
    if not (local.exists() and local.stat().st_size == size):
        for attempt in range(5):
            try:
                hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION, local_dir=raw)
                break
            except Exception as e:  # network hiccups: retry with backoff
                log(f"download {path} failed ({e!r}), retry {attempt + 1}")
                time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError(f"could not download {path}")
    dt = time.time() - t0
    got = sha256_file(local)
    if got != sha:
        local.unlink()
        raise RuntimeError(f"{path}: sha256 {got} != hub {sha}")
    log(f"downloaded {path} ({size / 1e9:.2f} GB, {size / max(dt, 1e-3) / 1e6:.0f} MB/s), sha256 ok")
    return local, dt


_VOCAB = None


def _init_worker():
    global _VOCAB
    import cv2
    cv2.setNumThreads(1)          # one process per core; no nested OpenCV thread pools
    _VOCAB = ex.vocab_set()


def _pages(job):
    """Worker: crop pages [a, b) of one shard. Returns (items, stats, oov, chars)."""
    import h5py
    path, tag, a, b, per_page, seed = job
    stats, oov, chars, items = ex.new_stats(), Counter(), Counter(), []
    with h5py.File(path, "r") as f:
        anns, dims, sids = f["annotations"][a:b], f["dimensions"][a:b], f["sample_ids"][a:b]
        images = f["images"]
        for k in range(b - a):
            items += ex.process_page(images[a + k].tobytes(), anns[k], dims[k], sids[k], tag, per_page, seed,
                                     _VOCAB, stats, oov, chars)
    return items, stats, oov, chars


def extract_file(pool, local, tag, per_page, seed):
    import h5py
    with h5py.File(local, "r") as f:
        n = f["images"].shape[0]
    jobs = [(str(local), tag, a, min(n, a + PAGES_PER_TASK), per_page, seed) for a in range(0, n, PAGES_PER_TASK)]
    stats, oov, chars, crops, labels, metas = ex.new_stats(), Counter(), Counter(), [], [], []
    for items, st, ov, ch in pool.imap(_pages, jobs):
        stats.update(st)
        oov.update(ov)
        chars.update(ch)
        for g, lab, meta in items:
            crops.append(g)
            labels.append(lab)
            metas.append(meta)
    return crops, labels, metas, stats, oov, chars


def cmd_run(args):
    out, raw = Path(args.out), Path(args.raw)
    if args.split == "validation":      # merged into nv_val.npz by merge-val; kept out of nv_*.npz
        out = out / "val_parts"
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    files = hub_files(args.split)
    names = sorted(files)
    todo = []
    for i in parse_range(args.files, len(names)):
        path = f"ja/{args.split}/{args.split}_{i:03d}.h5"
        if path not in files:
            raise SystemExit(f"{path} is not in {REPO}@{REVISION}")
        tag = Path(path).stem
        if (out / f"nv_{tag}.npz").exists() and (out / f"nv_{tag}.json").exists():
            continue
        todo.append((path, tag))
    log(f"{len(todo)} {args.split} shards to do, {args.procs} procs, {args.downloads} downloads, "
        f"raw budget {args.max_raw_gb} GB")
    budget = Budget(int(args.max_raw_gb * 1e9))
    t_all = time.time()
    with ThreadPoolExecutor(args.downloads) as dl, mp.get_context("fork").Pool(args.procs, _init_worker) as pool:
        futures = [(path, tag, dl.submit(download, path, *files[path], raw, budget)) for path, tag in todo]
        for path, tag, fut in futures:
            local, dl_s = fut.result()
            size, sha = files[path]
            t0 = time.time()
            crops, labels, metas, stats, oov, chars = extract_file(pool, local, tag, args.per_page, args.seed)
            npz = out / f"nv_{tag}.npz"
            ex.save_shard(npz, crops, labels, metas)
            rec = {"source": path, "source_bytes": size, "source_sha256": sha, "shard": npz.name,
                   "shard_bytes": npz.stat().st_size, "shard_sha256": sha256_file(npz), "lines": len(labels),
                   "per_page": args.per_page, "seed": args.seed, "revision": REVISION, "git": git_sha(),
                   "download_s": round(dl_s, 1), "extract_s": round(time.time() - t0, 1),
                   "stats": dict(stats), "oov": dict(oov.most_common()), "chars": dict(chars.most_common())}
            tmp = out / f"nv_{tag}.json.tmp"
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            tmp.rename(out / f"nv_{tag}.json")
            local.unlink()
            budget.give(size)
            log(f"{tag}: {len(labels)} lines from {stats['pages']} pages in {rec['extract_s']}s, "
                f"{rec['shard_bytes'] / 1e6:.0f} MB, total {time.time() - t_all:.0f}s")
    log("RUN_DONE")


def load_npz(path):
    z = np.load(path)
    return (z["data"], z["offsets"], z["shapes"], json.loads(str(z["labels"])), json.loads(str(z["meta"])))


def cmd_merge_val(args):
    out = Path(args.out)
    parts = sorted((out / "val_parts").glob("nv_validation_*.npz"))
    if not parts:
        raise SystemExit("no val_parts/nv_validation_*.npz to merge")
    crops, labels, metas = [], [], []
    for p in parts:
        data, offs, shapes, labs, mets = load_npz(p)
        crops += [data[offs[i]:offs[i + 1]].reshape(shapes[i]) for i in range(len(labs))]
        labels += labs
        metas += mets
    ex.save_shard(out / "nv_val.npz", crops, labels, metas)
    log(f"nv_val.npz: {len(labels)} lines from {len(parts)} shards")


def cmd_manifest(args):
    out = Path(args.out)
    recs = [json.loads(p.read_text()) for p in [*sorted(out.glob("nv_*.json")),
                                               *sorted((out / "val_parts").glob("nv_*.json"))]]
    shards, total, oov = [], ex.new_stats(), Counter()
    for r in recs:
        total.update(r["stats"])
        oov.update(r["oov"])
        shards.append({k: r[k] for k in ("source", "source_bytes", "source_sha256", "shard", "shard_bytes",
                                         "shard_sha256", "lines", "git")} | {"pages": r["stats"]["pages"]})
    val = out / "nv_val.npz"
    by_split = Counter()
    for r in recs:
        by_split[r["source"].split("/")[1]] += r["lines"]
    st = dict(total)
    man = {
        "dataset": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), (c) NVIDIA Corporation",
        "subset": "ja (Japanese): train and validation splits only",
        "made_by": "nvsynth/fetch.py run, then merge-val and manifest",
        "format": "train.py shards: data (uint8, flat), offsets, shapes (32 x W), labels (JSON list), meta "
                  "(JSON list: src, sample, line, crop [x, y, w, h] on the page, ink_h, clean, contrast, "
                  "grey_ratio, polarity)",
        "filters": {"min_box_h": ex.MIN_BOX_H, "min_chars": ex.MIN_CHARS, "max_ascii": ex.MAX_ASCII,
                    "pad_frac": list(ex.PAD_FRAC), "min_contrast": ex.CONTRAST, "min_grey_ratio": ex.GREY_RATIO,
                    "height": ex.H, "max_width": ex.MAX_W, "area_above": ex.AREA_ABOVE, "mid_h": list(ex.MID_H),
                    "per_page": sorted({r["per_page"] for r in recs}), "seed": sorted({r["seed"] for r in recs}),
                    "vocab": "data/vocab.txt (5,354 characters + CTC blank)",
                    "label_norm": "synth/build_corpus.py norm()"},
        "lines": dict(by_split),
        "val_shard": {"shard": val.name, "shard_sha256": sha256_file(val), "shard_bytes": val.stat().st_size}
        if val.exists() else None,
        "stats": st,
        "oov": {"char_rate": st["h_oov_chars"] / max(1, st["h_chars"]),
                "line_rate": st["h_oov_lines"] / max(1, st["h_lines"]),
                "note": "over every horizontal line of every processed page, after norm()",
                "top": [[c, f"U+{ord(c):04X}", n] for c, n in oov.most_common(300)]},
        "shards": shards,
    }
    Path(args.write).parent.mkdir(parents=True, exist_ok=True)
    Path(args.write).write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n")
    log(f"manifest: {len(shards)} shards, {sum(by_split.values())} lines -> {args.write}")


def default_procs():
    v = os.environ.get("OMP_NUM_THREADS", "")
    return int(v) if re.fullmatch(r"\d+", v) else 5


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=("train", "validation"), required=True)
    r.add_argument("--files", default="", help="shard numbers, e.g. 0-66 or 0,3,5-9 (default: all)")
    r.add_argument("--per-page", type=int, default=1)
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--raw", default=str(HERE.parent / "scratch" / "raw"))
    r.add_argument("--max-raw-gb", type=float, default=32)
    r.add_argument("--downloads", type=int, default=2)
    r.add_argument("--procs", type=int, default=default_procs())
    m = sub.add_parser("merge-val")
    m.add_argument("--out", default=str(DEFAULT_OUT))
    f = sub.add_parser("manifest")
    f.add_argument("--out", default=str(DEFAULT_OUT))
    f.add_argument("--write", default=str(HERE / "manifests" / "nvsynth_ja.json"))
    args = ap.parse_args()
    {"run": cmd_run, "merge-val": cmd_merge_val, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")   # no xet chunk cache: raw files only
    main()

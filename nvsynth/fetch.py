#!/usr/bin/env python3
"""Download the Japanese shards of NVIDIA OCR-Synthetic-Multilingual-v1 one by one,
cut their text lines into train.py shards, and delete each raw file afterwards.

  fetch.py run --split train|validation|test --out DIR [--files 0-66] [--per-page 1]
               [--seed S] [--lines-per-shard 20000] [--raw DIR] [--max-raw-gb 36]
               [--min-free-gb 70] [--downloads 3] [--procs N]
      For every ja/<split>/<split>_NNN.h5 at the pinned REVISION: download it with
      huggingface_hub, check its sha256 against the Hub's LFS record, crop up to
      --per-page usable lines per page (0 = every usable line; nvsynth/extract.py),
      write them to DIR, record counts in DIR/nv_<split>_NNN.json, delete the raw file.
      With --per-page 1 a source file becomes one shard, DIR/nv_<split>_NNN.npz. With
      --per-page 0 its lines are spread over B = round(lines / --lines-per-shard)
      shards DIR/nv_<split>_NNN_bBB.npz by a hash of (file, page, line), so bucket BB
      of every file is a uniform sample of all pages: a glob such as '*_b00.npz' picks
      1/B of the lines without loading the rest.
      Downloads run ahead of the cutting, but raw files on disk (finished or in flight)
      never exceed --max-raw-gb, and a download waits while it would leave less than
      --min-free-gb free on the disk. Resumable: finished sources are skipped and a
      complete raw file is not downloaded again.
  fetch.py merge-val --parts DIR --to FILE
      Concatenates DIR/nv_validation_*.npz into one shard (the held-out set).
  fetch.py manifest --dirs DIR [DIR ...] --write FILE [--val FILE]
      Collects the per-source counts, sha256s and OOV statistics into the manifest.

Commands that made the shared artifacts ($HERD/shared/artifacts/nvsynth, see docs/nvsynth.md):
  one line per page:  run --split validation --files 0-3 --out shards/val_parts;
                      merge-val --parts shards/val_parts --to shards/nv_val.npz;
                      run --split train --out shards; manifest --dirs shards shards/val_parts
                      --val shards/nv_val.npz --write nvsynth/manifests/nvsynth_ja.json
  every line:         run --split S --per-page 0 --out full/S for S in train, validation, test;
                      manifest --dirs full/train full/validation full/test
                      --write nvsynth/manifests/nvsynth_ja_full.json

The dataset is CC BY 4.0 (NVIDIA); see README.md for the attribution. Raw shards
are about 9.8 GB (train) and 1.2 GB (validation, test) each. Worker processes follow
OMP_NUM_THREADS (heavy.sh sets 5) unless --procs is given.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract as ex  # noqa: E402

REPO = "nvidia/OCR-Synthetic-Multilingual-v1"
REVISION = "69696a1cc543ef3a0f8e9892a89c17293e915263"     # main on 2026-10-09 (last change 2026-04-20)
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ARTIFACTS = HERD / "shared" / "artifacts" / "nvsynth"
DEFAULT_SEED = 20261009
PAGES_PER_TASK = 500
OUT_FRACTION = 0.2          # output bytes per raw byte, for the disk-free check (measured: 0.013 at one
                            # line per page, about 0.16 with every line)


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
    """Raw-file bytes allowed on disk at once (finished downloads plus in-flight ones), and a
    floor of free disk space: a download starts only if the disk keeps min_free bytes free once
    every pending download and the output still to be written for it have landed."""

    def __init__(self, limit, min_free=0, path="/", free=None):
        self.limit, self.min_free, self.path = limit, min_free, path
        self.used = 0                 # raw bytes held (on disk or in flight)
        self.pending = 0              # bytes not yet on disk: in-flight downloads plus their output
        self.cond = threading.Condition()
        self.free = free or (lambda: shutil.disk_usage(self.path).free)
        self.waited = 0.0

    def _ok(self, n):
        if self.used and self.used + n > self.limit:
            return False
        return self.free() - self.pending - n - int(n * OUT_FRACTION) >= self.min_free

    def take(self, n):
        t0 = time.time()
        with self.cond:
            warned = False
            while not self._ok(n):
                if not warned and time.time() - t0 > 60:
                    log(f"waiting for disk: {self.free() / 1e9:.0f} GB free, raw held {self.used / 1e9:.0f} GB")
                    warned = True
                self.cond.wait(timeout=30)
            self.used += n
            self.pending += n + int(n * OUT_FRACTION)
        self.waited += time.time() - t0

    def landed(self, n):
        """A download of n bytes finished (now on disk); its output is still pending."""
        with self.cond:
            self.pending -= n
            self.cond.notify_all()

    def give(self, n):
        """The raw file was cut and deleted; its output is on disk."""
        with self.cond:
            self.used -= n
            self.pending -= int(n * OUT_FRACTION)
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
    budget.landed(size)
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


def extract_file(workers, local, tag, per_page, seed, timeout=1800, tries=3):
    """Cuts one source file in page ranges on the worker processes. A worker that dies (the
    OOM killer, a stray signal) or a file that takes longer than timeout seconds restarts the
    workers and the file, up to tries times."""
    import h5py
    with h5py.File(local, "r") as f:
        n = f["images"].shape[0]
    jobs = [(str(local), tag, a, min(n, a + PAGES_PER_TASK), per_page, seed) for a in range(0, n, PAGES_PER_TASK)]
    for attempt in range(1, tries + 1):
        try:
            results = workers.map(_pages, jobs, timeout)
            break
        except (BrokenProcessPool, FuturesTimeout, OSError) as e:
            log(f"{tag}: workers failed ({e!r}), restarting them (attempt {attempt} of {tries})")
            workers.reset()
    else:
        raise RuntimeError(f"{tag}: workers failed {tries} times")
    stats, oov, chars, crops, labels, metas = ex.new_stats(), Counter(), Counter(), [], [], []
    for items, st, ov, ch in results:
        stats.update(st)
        oov.update(ov)
        chars.update(ch)
        for g, lab, meta in items:
            crops.append(g)
            labels.append(lab)
            metas.append(meta)
    return crops, labels, metas, stats, oov, chars


class Workers:
    """A process pool (spawn: no fork of a process that has download threads) that can be
    thrown away and rebuilt when a worker dies."""

    def __init__(self, n):
        self.n, self.pool = n, None
        self.reset()

    def reset(self):
        if self.pool is not None:
            procs = list((getattr(self.pool, "_processes", None) or {}).values())
            self.pool.shutdown(wait=False, cancel_futures=True)
            for p in procs:                                   # stuck workers: stop them
                if p.is_alive():
                    p.kill()
        self.pool = ProcessPoolExecutor(self.n, mp_context=mp.get_context("spawn"), initializer=_init_worker)

    def map(self, fn, jobs, timeout):
        """Results in job order; raises BrokenProcessPool or TimeoutError."""
        futs = [self.pool.submit(fn, j) for j in jobs]
        done = {}
        for fut in as_completed(futs, timeout=timeout):
            done[fut] = fut.result()
        return [done[f] for f in futs]

    def close(self):
        self.pool.shutdown(wait=True)


def bucket_of(meta, n_buckets):
    """Sub-shard of one line: a hash of (source file, page, line), so every bucket of a file is a
    uniform sample of its pages, and a line's bucket depends only on the bucket count."""
    key = f"{meta['src']}:{meta['sample']}:{meta['line']}".encode()
    return zlib.crc32(key) % n_buckets


def write_shards(out, tag, crops, labels, metas, n_buckets, writers):
    """Writes the lines as n_buckets shards (in parallel threads: zlib releases the GIL).
    Returns [{shard, lines, shard_bytes, shard_sha256}]."""
    groups = [[] for _ in range(n_buckets)]
    for i, m in enumerate(metas):
        groups[bucket_of(m, n_buckets) if n_buckets > 1 else 0].append(i)
    names = [f"nv_{tag}.npz"] if n_buckets == 1 else [f"nv_{tag}_b{b:02d}.npz" for b in range(n_buckets)]

    def one(b):
        idx = groups[b]
        path = out / names[b]
        ex.save_shard(path, [crops[i] for i in idx], [labels[i] for i in idx], [metas[i] for i in idx])
        return {"shard": names[b], "lines": len(idx), "shard_bytes": path.stat().st_size,
                "shard_sha256": sha256_file(path)}

    return list(writers.map(one, range(n_buckets)))


def record_path(out, tag):
    return out / f"nv_{tag}.json"


def cmd_run(args):
    out, raw = Path(args.out), Path(args.raw)
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    files = hub_files(args.split)
    todo = []
    for i in parse_range(args.files, len(files)):
        path = f"ja/{args.split}/{args.split}_{i:03d}.h5"
        if path not in files:
            raise SystemExit(f"{path} is not in {REPO}@{REVISION}")
        tag = Path(path).stem
        if not record_path(out, tag).exists():
            todo.append((path, tag))
    log(f"{len(todo)} {args.split} sources to do -> {out}, per page {args.per_page or 'all'}, {args.procs} procs, "
        f"{args.downloads} downloads, raw budget {args.max_raw_gb} GB, keep {args.min_free_gb} GB free")
    budget = Budget(int(args.max_raw_gb * 1e9), int(args.min_free_gb * 1e9), path=str(out))
    t_all, total = time.time(), 0
    workers = Workers(args.procs)
    with ThreadPoolExecutor(args.downloads) as dl, ThreadPoolExecutor(args.procs) as writers:
        futures = [(path, tag, dl.submit(download, path, *files[path], raw, budget)) for path, tag in todo]
        for path, tag, fut in futures:
            local, dl_s = fut.result()
            size, sha = files[path]
            t0 = time.time()
            crops, labels, metas, stats, oov, chars = extract_file(workers, local, tag, args.per_page, args.seed)
            t_cut = time.time() - t0
            n_buckets = 1 if args.per_page else max(1, round(len(labels) / args.lines_per_shard))
            shards = write_shards(out, tag, crops, labels, metas, n_buckets, writers)
            del crops, metas
            rec = {"source": path, "source_bytes": size, "source_sha256": sha, "lines": len(labels),
                   "per_page": args.per_page, "seed": args.seed, "revision": REVISION, "git": git_sha(),
                   "download_s": round(dl_s, 1), "extract_s": round(t_cut, 1),
                   "write_s": round(time.time() - t0 - t_cut, 1), "shards": shards,
                   "stats": dict(stats), "oov": dict(oov.most_common()), "chars": dict(chars.most_common())}
            if n_buckets == 1:      # one-line-per-page layout: the shard's fields at the top level, as before
                rec.update(shards[0])
            tmp = record_path(out, tag).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            tmp.rename(record_path(out, tag))
            local.unlink()
            budget.give(size)
            total += len(labels)
            log(f"{tag}: {len(labels)} lines from {stats['pages']} pages in {t_cut:.0f}s cut + "
                f"{rec['write_s']:.0f}s write, {n_buckets} shard(s), "
                f"{sum(s['shard_bytes'] for s in shards) / 1e6:.0f} MB; {total} lines so far, "
                f"{time.time() - t_all:.0f}s, {shutil.disk_usage(str(out)).free / 1e9:.0f} GB free")
    workers.close()
    log(f"RUN_DONE {args.split}: {total} lines")


def load_npz(path):
    z = np.load(path)
    return (z["data"], z["offsets"], z["shapes"], json.loads(str(z["labels"])), json.loads(str(z["meta"])))


def cmd_merge_val(args):
    parts = sorted(Path(args.parts).glob("nv_validation_*.npz"))
    if not parts:
        raise SystemExit(f"no nv_validation_*.npz in {args.parts}")
    crops, labels, metas = [], [], []
    for p in parts:
        data, offs, shapes, labs, mets = load_npz(p)
        crops += [data[offs[i]:offs[i + 1]].reshape(shapes[i]) for i in range(len(labs))]
        labels += labs
        metas += mets
    ex.save_shard(Path(args.to), crops, labels, metas)
    log(f"{args.to}: {len(labels)} lines from {len(parts)} shards")


def cmd_manifest(args):
    recs = []
    for d in args.dirs:
        recs += [json.loads(p.read_text()) for p in sorted(Path(d).glob("nv_*.json"))]
    total, oov, by_split, shards_by_split = ex.new_stats(), Counter(), Counter(), Counter()
    sources = []
    for r in recs:
        total.update(r["stats"])
        oov.update(r["oov"])
        split = r["source"].split("/")[1]
        shards = r.get("shards") or [{k: r[k] for k in ("shard", "lines", "shard_bytes", "shard_sha256")}]
        by_split[split] += r["lines"]
        shards_by_split[split] += len(shards)
        sources.append({"source": r["source"], "source_bytes": r["source_bytes"],
                        "source_sha256": r["source_sha256"], "pages": r["stats"]["pages"], "lines": r["lines"],
                        "git": r["git"], "shards": shards})
    st = dict(total)
    man = {
        "dataset": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), (c) NVIDIA Corporation",
        "subset": "ja (Japanese) only: " + ", ".join(sorted(by_split)),
        "made_by": "nvsynth/fetch.py (see its docstring for the exact commands)",
        "format": "train.py shards: data (uint8, flat), offsets, shapes (32 x W), labels (JSON list), meta "
                  "(JSON list: src, sample, line, crop [x, y, w, h] on the page, ink_h, clean, contrast, "
                  "grey_ratio, polarity)",
        "per_page": sorted({r["per_page"] for r in recs}),
        "filters": {"min_box_h": ex.MIN_BOX_H, "min_chars": ex.MIN_CHARS, "max_ascii": ex.MAX_ASCII,
                    "one_word_box": True, "pad_frac": list(ex.PAD_FRAC), "min_contrast": ex.CONTRAST,
                    "min_grey_ratio": ex.GREY_RATIO, "height": ex.H, "max_width": ex.MAX_W,
                    "area_above": ex.AREA_ABOVE, "mid_h": list(ex.MID_H), "seed": sorted({r["seed"] for r in recs}),
                    "vocab": "data/vocab.txt (5,354 characters + CTC blank)",
                    "label_norm": "synth/build_corpus.py norm()"},
        "lines": dict(by_split),
        "lines_total": sum(by_split.values()),
        "shards": dict(shards_by_split),
        "val_shard": None,
        "stats": st,
        "oov": {"char_rate": st["h_oov_chars"] / max(1, st["h_chars"]),
                "line_rate": st["h_oov_lines"] / max(1, st["h_lines"]),
                "note": "over every horizontal line of every processed page, after norm()",
                "top": [[c, f"U+{ord(c):04X}", n] for c, n in oov.most_common(300)]},
        "sources": sources,
    }
    if args.val:
        val = Path(args.val)
        man["val_shard"] = {"shard": val.name, "shard_sha256": sha256_file(val), "shard_bytes": val.stat().st_size,
                            "lines": len(json.loads(str(np.load(val)["labels"])))}
    Path(args.write).parent.mkdir(parents=True, exist_ok=True)
    Path(args.write).write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n")
    log(f"manifest: {len(sources)} sources, {sum(shards_by_split.values())} shards, "
        f"{man['lines_total']} lines -> {args.write}")


def default_procs():
    v = os.environ.get("OMP_NUM_THREADS", "")
    return int(v) if re.fullmatch(r"\d+", v) else 5


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=("train", "validation", "test"), required=True)
    r.add_argument("--out", required=True, help="directory for this split's shards and counts")
    r.add_argument("--files", default="", help="source numbers, e.g. 0-66 or 0,3,5-9 (default: all)")
    r.add_argument("--per-page", type=int, default=1, help="lines per page (0 = every usable line)")
    r.add_argument("--lines-per-shard", type=int, default=20000, help="with --per-page 0")
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--raw", default=str(HERE.parent / "scratch" / "raw"))
    r.add_argument("--max-raw-gb", type=float, default=36)
    r.add_argument("--min-free-gb", type=float, default=70)
    r.add_argument("--downloads", type=int, default=3)
    r.add_argument("--procs", type=int, default=default_procs())
    m = sub.add_parser("merge-val")
    m.add_argument("--parts", required=True)
    m.add_argument("--to", required=True)
    f = sub.add_parser("manifest")
    f.add_argument("--dirs", nargs="+", required=True)
    f.add_argument("--val", default="")
    f.add_argument("--write", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "merge-val": cmd_merge_val, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")   # no xet chunk cache: raw files only
    main()

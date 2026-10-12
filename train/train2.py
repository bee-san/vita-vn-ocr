#!/usr/bin/env python3
"""train2: the release recipe for the tiny VN recognizer (pretrain, fine-tune, distil).

  train2.py --run NAME --out runs [--init CKPT] --steps N \
      --src 'nv:0.5:/a/full/train/nv_train_0[0-3]?_b*.npz' \
      --src 's2:0.3:/a/synth2/shards/synth_000[0-3][0-8].npz' \
      --src 'real:0.2:/a/teacher/teacher_train.npz:targets=DIR,kd=1,inv=0.5' \
      --val 'rv:/a/teacher/teacher_val.npz' --val 'rvn:/a/teacher/teacher_val.npz:native=1' \
      --select rv,rvn

Each --src is NAME:WEIGHT:GLOBS[:OPTS]. GLOBS is one or more globs joined by '+'. Every
training sample is drawn from source NAME with probability WEIGHT / sum(WEIGHT). OPTS is a
comma-separated list of key=value:

  stream=1      stream the shards (a pool of `pool` shards in RAM, refilled in a background
                thread, cycling through a reshuffled shard list) instead of loading them all;
                the default for sources over 2 GB on disk
  pool=3        shards held at once when streaming
  targets=DIR   teacher targets (train/teacher_targets.py: DIR/<shard stem>.v6s.npz)
  label=gt      the hard label: gt (the shard's own label; lines without one are dropped),
                teacher (the teacher's reading), or gt|teacher (gt when present)
  conf=0.5      with label=teacher: keep only lines whose lowest character confidence is at
                least this (lines below it are dropped, or with kdonly=1 kept for KD alone)
  kd=0          1: add the frame-level KL to the teacher's targets for this source's lines
                whose targets are usable (soft_ok; and, for gt labels, a teacher reading that
                scores the same as the label)
  kdonly=0      1: lines under the conf threshold are kept with the KL term only (no CTC)
  inv=0.2       probability of inverting the crop (255 - g): every source then shows both
                polarities (real crops are stored as the console hands them over, dark text
                inverted; eval feeds them as they are on screen)
  aug=1         0 turns augmentation off for the source

Augmentation (train.py's plus a PSP-like down-up resize and JPEG): lines with frame
targets get only augmentations that keep the frame grid (no horizontal stretch).

Each --val is NAME:PATH[:OPTS] with OPTS n=N (first N lines, default 2000) and native=1
(un-invert crops whose meta says inverted, i.e. score real crops in screen polarity).
Validation CER is eval's: eval/metrics.norm on both sides, Levenshtein over total GT
characters, one forward pass per padded width (identical to one-line inference). The
checkpoint with the lowest mean CER over --select sets is ckpt_best.pt. With --ema the
EMA weights are what is evaluated and saved as "model" (the raw ones as "model_raw").

Checkpoints are train.py's format (eval/run.py, train/export.py, sweep_eval.py read them).
A run resumes from <out>/<run>/ckpt_last.pt. Threads: $TORCH_NUM_THREADS (heavy.sh).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import platform
import queue
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "eval"))
from metrics import lev, norm, score_hiragana  # noqa: E402

from common import DATA, load_vocab  # noqa: E402
from model import TinyVNOCR, set_weight_quant  # noqa: E402
from source_policy import (  # noqa: E402
    CAPS,
    ExposureJournal,
    ExposureLedger,
    RunWriter,
    SourcePolicy,
    digest,
    freeze_json,
    json_digest,
)
from source_policy import sha256 as file_sha256  # noqa: E402
from train import UNK, ctc_ok, default_threads, encoder, resize_h  # noqa: E402

H, MAX_W, STRIDE = 32, 1024, 8
# Teacher targets index data/vocab.txt's classes. With another --vocab, KD_REMAP maps each of
# those classes to the training vocab's (a reduced vocab's <unk> takes the ones it lacks).
KD_REMAP = None


def kd_remap(vocab):
    full = load_vocab(DATA / "vocab.txt")
    if list(vocab) == full:
        return None
    ids = {c: i + 1 for i, c in enumerate(vocab)}
    unk = ids.get(UNK) if vocab[-1] == UNK else None
    if unk is None:
        raise SystemExit("KD targets with a reduced vocab need its <unk> class (last entry 〓)")
    return np.asarray([0] + [ids.get(c, unk) for c in full], np.int64)

# ----------------------------------------------------------------------------- sources

def parse_opts(s):
    out = {}
    for kv in filter(None, (s or "").split(",")):
        k, _, v = kv.partition("=")
        out[k.strip()] = v.strip()
    return out


def expand(globs):
    paths = []
    for g in globs.split("+"):
        hit = sorted(glob.glob(g))
        if not hit:
            raise SystemExit(f"no files match {g}")
        paths += hit
    return paths


class Item:
    __slots__ = ("img", "ids", "kid", "kp", "src", "provenance")

    def __init__(self, img, ids, kid=None, kp=None, src=0, provenance=None):
        self.img, self.ids, self.kid, self.kp, self.src = img, ids, kid, kp, src
        self.provenance = provenance


def load_shard(path, encode, opts, vocab_key, provenance=None):
    """Items of one shard under a source's options. vocab_key(text) -> eval-scoring key."""
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    if provenance is not None and (len(provenance) != len(shapes) or len(labels) != len(shapes)):
        raise ValueError("source policy row count differs from crop/label arrays")
    tg = None
    if opts.get("targets"):
        tp = Path(opts["targets"]) / (Path(path).stem + ".v6s.npz")
        if tp.exists():
            t = np.load(tp)
            tg = {"text": json.loads(str(t["text"])), "conf": t["conf"], "T": t["T"], "off": t["off"],
                  "kid": t["kid"], "kp": t["kp"], "ok": t["soft_ok"]}
        elif opts.get("label", "gt") != "gt" or opts.get("kd") == "1":
            raise SystemExit(f"{path}: no teacher targets {tp}")
    mode = opts.get("label", "gt")
    conf_min = float(opts.get("conf", 0.5))
    kd, kdonly = opts.get("kd") == "1", opts.get("kdonly") == "1"
    items, stats = [], {"kept": 0, "dropped": 0, "kd": 0, "kdonly": 0}
    n = len(shapes) if tg is None else min(len(shapes), len(tg["text"]))
    for i in range(n):
        lab = labels[i]
        use_kd_only = False
        if mode == "teacher" or (mode == "gt|teacher" and not lab):
            if tg is None:
                stats["dropped"] += 1
                continue
            lab = tg["text"][i]
            if float(tg["conf"][i][0]) < conf_min or not lab:
                if kdonly and tg["ok"][i]:
                    use_kd_only = True
                else:
                    stats["dropped"] += 1
                    continue
        if not lab and not use_kd_only:
            stats["dropped"] += 1
            continue
        h, w = (int(v) for v in shapes[i])
        g = resize_h(data[offs[i]:offs[i + 1]].reshape(h, w), H)
        ids = None
        if not use_kd_only:
            ids = encode(lab)
            if not ids or -1 in ids or not ctc_ok(ids, g.shape[1]):
                stats["dropped"] += 1
                continue
            ids = np.asarray(ids, np.int64)
        kid = kp = None
        if (kd or use_kd_only) and tg is not None and tg["ok"][i]:
            same = use_kd_only or mode != "gt" and lab == tg["text"][i] or \
                vocab_key(lab) == vocab_key(tg["text"][i])
            T = int(tg["T"][i])
            if same and T == (g.shape[1] + STRIDE - 1) // STRIDE:
                a, b = int(tg["off"][i]), int(tg["off"][i + 1])
                kid, kp = tg["kid"][a:b], tg["kp"][a:b]
                if KD_REMAP is not None:
                    kid = KD_REMAP[kid.astype(np.int64)].astype(np.int16)
                stats["kd"] += 1
        if use_kd_only and kid is None:
            stats["dropped"] += 1
            continue
        stats["kdonly"] += use_kd_only
        items.append(Item(g, ids, kid, kp, provenance=None if provenance is None else provenance[i]))
        stats["kept"] += 1
    return items, stats


class RamSource:
    def __init__(self, name, paths, encode, opts, key, rng, provenance=None):
        self.name, self.opts, self.rng = name, opts, rng
        self.items, self.stats = [], {}
        for p in paths:
            it, st = load_shard(p, encode, opts, key,
                                None if provenance is None else provenance[Path(p).resolve()])
            self.items += it
            for k, v in st.items():
                self.stats[k] = self.stats.get(k, 0) + v
        if not self.items:
            raise SystemExit(f"source {name}: no usable lines in {paths}")
        self.order, self.pos = rng.permutation(len(self.items)), 0

    def sample(self):
        if self.pos >= len(self.order):
            self.order, self.pos = self.rng.permutation(len(self.items)), 0
        it = self.items[self.order[self.pos]]
        self.pos += 1
        return it

    def describe(self):
        return f"{self.name}: {len(self.items)} lines in RAM {self.stats}"


class StreamSource:
    """A pool of shards in RAM; an exhausted shard is replaced by the next one of a shuffled,
    endlessly cycling shard list, loaded ahead by a background thread."""

    def __init__(self, name, paths, encode, opts, key, rng):
        self.name, self.paths, self.encode, self.opts, self.key = name, paths, encode, opts, key
        self.rng = rng                                   # main thread only
        self.lrng = np.random.default_rng(int(rng.integers(1 << 31)))   # loader thread only
        self.n_pool = int(opts.get("pool", 3))
        self.order, self.next = list(self.lrng.permutation(len(paths))), 0
        self.q = queue.Queue(maxsize=2)
        self.loaded = 0
        self.thread = threading.Thread(target=self._loader, daemon=True)
        self.thread.start()
        self.pool = [self._take() for _ in range(min(self.n_pool, len(paths)))]

    def _next_path(self):
        if self.next >= len(self.order):
            self.order, self.next = list(self.lrng.permutation(len(self.paths))), 0
        p = self.paths[self.order[self.next]]
        self.next += 1
        return p

    def _loader(self):
        # Any failure (load_shard's SystemExit for missing targets, a corrupt shard, a whole
        # pass over the shard list without a usable line) is handed to the training thread,
        # which raises it, instead of ending this thread silently and leaving _take() blocked.
        try:
            empty = 0
            while True:
                p = self._next_path()
                items, _ = load_shard(p, self.encode, self.opts, self.key)
                if not items:
                    empty += 1
                    if empty >= len(self.paths):
                        raise SystemExit(f"source {self.name}: no usable lines in any of its "
                                         f"{len(self.paths)} shards")
                    continue
                empty = 0
                perm = self.lrng.permutation(len(items))
                self.q.put([items[j] for j in perm])
        except BaseException as e:   # SystemExit included: a thread would swallow it
            self.q.put(e)

    def _take(self):
        self.loaded += 1
        got = self.q.get()
        if isinstance(got, BaseException):
            raise RuntimeError(f"source {self.name}: shard loader failed: {got}") from got
        return got

    def sample(self):
        k = int(self.rng.integers(len(self.pool)))
        if not self.pool[k]:
            self.pool[k] = self._take()
        return self.pool[k].pop()

    def describe(self):
        return f"{self.name}: {len(self.paths)} shards streamed, pool {self.n_pool}"


# ----------------------------------------------------------------------------- augmentation

def augment(g, rng, geo=True, inv=0.0, min_width=8):
    height = g.shape[0]
    g = g.astype(np.float32)
    if geo and rng.random() < 0.3:                           # horizontal stretch
        nw = max(8, min_width, min(MAX_W, int(g.shape[1] * rng.uniform(0.88, 1.12))))
        g = cv2.resize(g, (nw, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:                                   # vertical jitter (width kept)
        d = int(rng.integers(1, 3))
        g = g[d:] if rng.random() < 0.5 else np.pad(g, ((d, 0), (0, 0)), mode="edge")
        g = cv2.resize(g, (g.shape[1], height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.12:                                  # PSP-size text: down, then up
        s = rng.uniform(0.42, 0.75)
        w = g.shape[1]
        small = cv2.resize(g, (max(4, int(w * s)), max(8, int(round(height * s)))),
                           interpolation=cv2.INTER_AREA)
        g = cv2.resize(small, (w, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.2:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 0.9))
    if rng.random() < 0.08:                                  # JPEG
        ok, buf = cv2.imencode(".jpg", np.clip(g, 0, 255).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        if ok:
            g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if rng.random() < 0.5:
        g = g * rng.uniform(0.7, 1.15) + rng.uniform(-25, 25)
    if rng.random() < 0.2:
        g = g + rng.normal(0, rng.uniform(2, 8), g.shape)
    g = np.clip(g, 0, 255)
    if inv and rng.random() < inv:
        g = 255.0 - g
    return g


def make_batch(items, rng, src_opts):
    gs = []
    for it in items:
        o = src_opts[it.src]
        if rng is not None and o.get("aug", "1") != "0":
            # Shrinking a tight crop must retain enough frames for the label, including
            # a blank between adjacent repeated characters. Otherwise zero_infinity
            # silently removes that sample's CTC gradient.
            frames = 1 if it.ids is None else len(it.ids) + sum(
                a == b for a, b in zip(it.ids, it.ids[1:]))
            min_width = max(8, (frames - 1) * STRIDE + 1)
            g = augment(it.img, rng, geo=it.kid is None, inv=float(o.get("inv", 0.2)),
                        min_width=min_width)
        else:
            g = it.img.astype(np.float32)
        gs.append(g)
    W = (max(g.shape[1] for g in gs) + STRIDE - 1) // STRIDE * STRIDE
    x = np.empty((len(gs), 1, H, W), np.float32)
    for j, g in enumerate(gs):
        w = g.shape[1]
        x[j, 0, :, :w] = g
        x[j, 0, :, w:] = g[:, -1:]
    x = torch.from_numpy(x / 127.5 - 1.0).contiguous(memory_format=torch.channels_last)
    in_len = torch.tensor([math.ceil(g.shape[1] / STRIDE) for g in gs], dtype=torch.long)
    lab = [j for j, it in enumerate(items) if it.ids is not None]
    tgt = torch.from_numpy(np.concatenate([items[j].ids for j in lab])) if lab else torch.zeros(0, dtype=torch.long)
    tgt_len = torch.tensor([len(items[j].ids) for j in lab], dtype=torch.long)
    kd = None
    kd_rows = [j for j, it in enumerate(items) if it.kid is not None]
    if kd_rows:
        Tm = W // STRIDE
        k = items[kd_rows[0]].kid.shape[1]
        kid = np.zeros((len(items), Tm, k), np.int64)
        kp = np.zeros((len(items), Tm, k), np.float32)
        mask = np.zeros((len(items), Tm), np.float32)
        for j in kd_rows:
            t = items[j].kid.shape[0]
            kid[j, :t], kp[j, :t], mask[j, :t] = items[j].kid, items[j].kp, 1.0
        kd = (torch.from_numpy(kid), torch.from_numpy(kp), torch.from_numpy(mask))
    return x, in_len, torch.tensor(lab, dtype=torch.long), tgt, tgt_len, kd


class Mixer:
    """Draws samples from the sources by weight, buckets them by width into batches."""

    def __init__(self, sources, weights, batch, rng, bucket=24):
        self.sources, self.batch, self.rng, self.bucket = sources, batch, rng, bucket
        w = np.asarray(weights, np.float64)
        self.p = w / w.sum()
        self.ready = []
        self.counts = np.zeros(len(sources), np.int64)

    def _fill(self):
        n = self.batch * self.bucket
        pick = self.rng.choice(len(self.sources), size=n, p=self.p)
        items = []
        for s in pick:
            it = self.sources[s].sample()
            items.append(Item(it.img, it.ids, it.kid, it.kp, int(s), it.provenance))
            self.counts[s] += 1
        order = np.argsort(np.array([it.img.shape[1] for it in items]) + self.rng.uniform(0, 24, n))
        chunks = [[items[j] for j in order[i:i + self.batch]] for i in range(0, n, self.batch)]
        self.rng.shuffle(chunks)
        self.ready = chunks

    def next(self):
        if not self.ready:
            self._fill()
        return self.ready.pop()


class PolicyMixer(Mixer):
    """Check final consumption order, replacing capped/unknown rows with broad fallback.

    Prefetch is never credited as model exposure. A caller must record every successful
    forward, including a forward that produces no optimizer update, before asking for more.
    """
    def __init__(self, sources, weights, batch, rng, policy, src_opts, bucket=24):
        if any(not isinstance(source, RamSource) for source in sources):
            raise ValueError("source policy supports deterministic RAM sources only")
        for opts in src_opts:
            validate_policy_options(opts)
        if (len(weights) != len(sources) or not weights or any(not math.isfinite(w) or w < 0 for w in weights)
                or sum(weights) <= 0 or type(batch) is not int or batch <= 0
                or type(bucket) is not int or bucket <= 0):
            raise ValueError("source policy requires valid weights and positive batch/bucket sizes")
        super().__init__(sources, weights, batch, rng, bucket)
        self.policy = policy
        names = [source.name for source in sources]
        if len(set(names)) != len(names):
            raise ValueError("source policy requires unique source names")
        if any(item.provenance is None for source in sources for item in source.items):
            raise ValueError("source policy requires bound immutable row provenance")
        self.fallback = names.index(policy.fallback_source)
        if any(item.provenance is None or not item.provenance.fallback_verified or item.ids is None
               for item in sources[self.fallback].items):
            raise ValueError("all usable fallback rows require verified broad provenance and CTC labels")
        self.settings = {"sources": names, "weights": [float(w) for w in weights],
                         "batch": batch, "bucket": bucket, "options": src_opts}
        self.lookup = {(s, item.provenance.sample_id): item for s, source in enumerate(sources)
                       for item in source.items}
        self.ledger = ExposureLedger()
        self.rejected, self.fallback_draws = Counter(), 0
        self.pending = self.optimizer_pending = None

    def next(self):
        if self.pending is not None:
            raise ValueError("previous batch has not been recorded as forwarded")
        self.optimizer_pending = None
        reserved, selected = self.ledger.reserve(), []
        for item in super().next():
            reason = reserved.reason(item.provenance)
            if reason:
                self.rejected[reason] += 1
                source_item = self.sources[self.fallback].sample()
                item = Item(source_item.img, source_item.ids, source_item.kid, source_item.kp,
                            self.fallback, source_item.provenance)
                self.fallback_draws += 1
            reserved.consume(item.provenance, self.sources[item.src].name)
            selected.append(item)
        self.pending = selected
        return selected

    def record_forward(self, items):
        if items is not self.pending:
            raise ValueError("forwarded batch differs from its policy reservation")
        for item in items:
            self.ledger.consume(item.provenance, self.sources[item.src].name)
        self.pending = None
        self.optimizer_pending = items

    def record_optimizer(self, items):
        if items is not self.optimizer_pending:
            raise ValueError("optimizer batch must be the last forwarded batch, counted once")
        self.ledger.optimized([self.sources[item.src].name for item in items])
        self.optimizer_pending = None

    def report(self):
        return {"policy_sha256": self.policy.sha256, "caps": CAPS,
                "prefetch_draws": {source.name: int(count) for source, count in zip(self.sources, self.counts)},
                "fallback_draws": self.fallback_draws, "rejected": dict(self.rejected),
                "buffered_rows": sum(map(len, self.ready)), "consumed": self.ledger.state_dict()}

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("cannot checkpoint a batch before recording its forward")
        return {"schema": 1, "policy_sha256": self.policy.sha256, "settings": self.settings,
                "ledger": self.ledger.state_dict(), "rng": self.rng.bit_generator.state,
                "counts": self.counts.tolist(), "fallback_draws": self.fallback_draws,
                "rejected": dict(self.rejected),
                "ready": [[[item.src, item.provenance.sample_id] for item in batch] for batch in self.ready],
                "sources": [{"order": source.order.tolist(), "pos": int(source.pos),
                             "rng": source.rng.bit_generator.state} for source in self.sources]}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get("schema") != 1
                or state.get("policy_sha256") != self.policy.sha256 or state.get("settings") != self.settings):
            raise ValueError("resumed source policy or sampler configuration differs")
        ledger = ExposureLedger.from_state(state["ledger"])
        counts = state["counts"]
        if (not isinstance(counts, list) or len(counts) != len(self.sources)
                or any(type(n) is not int or n < 0 for n in counts)
                or type(state["fallback_draws"]) is not int or state["fallback_draws"] < 0):
            raise ValueError("invalid resumed draw counts")
        rejected = state["rejected"]
        if not isinstance(rejected, dict) or any(type(n) is not int or n < 0 for n in rejected.values()):
            raise ValueError("invalid resumed rejection counts")
        source_states = state["sources"]
        if not isinstance(source_states, list) or len(source_states) != len(self.sources):
            raise ValueError("invalid resumed source states")
        for source, saved in zip(self.sources, source_states):
            order, pos = saved["order"], saved["pos"]
            if (not isinstance(order, list) or any(type(n) is not int for n in order)
                    or sorted(order) != list(range(len(source.items)))
                    or type(pos) is not int or not 0 <= pos <= len(order)):
                raise ValueError("invalid resumed source cursor")
        ready = []
        if not isinstance(state["ready"], list):
            raise ValueError("invalid resumed prefetch buffer")
        for batch in state["ready"]:
            if not isinstance(batch, list) or len(batch) != self.batch:
                raise ValueError("invalid resumed prefetch batch")
            restored = []
            for reference in batch:
                if (not isinstance(reference, list) or len(reference) != 2
                        or type(reference[0]) is not int or tuple(reference) not in self.lookup):
                    raise ValueError("resumed immutable row identity is absent")
                s, sample_id = reference
                item = self.lookup[(s, sample_id)]
                restored.append(Item(item.img, item.ids, item.kid, item.kp, s, item.provenance))
            ready.append(restored)
        if (sum(counts) != ledger.forwarded + sum(map(len, ready))
                or state["fallback_draws"] != sum(rejected.values())):
            raise ValueError("resumed draws/forwards/buffers/rejections do not balance")
        # Validate RNG states before mutating any source/ledger state.
        for saved_rng in [state["rng"], *(saved["rng"] for saved in source_states)]:
            trial = np.random.default_rng()
            trial.bit_generator.state = saved_rng
        self.ledger, self.counts, self.ready = ledger, np.asarray(counts, np.int64), ready
        self.fallback_draws, self.rejected = state["fallback_draws"], Counter(rejected)
        self.rng.bit_generator.state = state["rng"]
        for source, saved in zip(self.sources, source_states):
            source.order, source.pos = np.asarray(saved["order"], np.int64), saved["pos"]
            source.rng.bit_generator.state = saved["rng"]


def validate_policy_options(opts):
    if opts.get("stream", "0") != "0":
        raise ValueError("source policy is RAM-only: use stream=0")
    if ("targets" in opts or opts.get("label", "gt") != "gt"
            or any(opts.get(key, "0") != "0" for key in ("kd", "kdonly"))):
        raise ValueError("source policy requires GT labels without unpinned teacher/KD auxiliaries")


def execution_identity(vocab_path, vocab, context=None):
    files = [HERE / name for name in ("train2.py", "source_policy.py", "train.py", "model.py", "restore_blob.py")]
    files += [ROOT / "synth/common.py", ROOT / "eval/metrics.py"]
    identity = {"schema": 1, "vocabulary_file_sha256": file_sha256(vocab_path),
            "ordered_vocabulary_sha256": json_digest(vocab),
            "geometry": {"height": H, "max_width": MAX_W, "stride": STRIDE},
            "code": {str(path.relative_to(ROOT)): file_sha256(path) for path in files},
            "dependencies": {"python": sys.version, "python_executable_sha256": file_sha256(sys.executable),
                             "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                             "lock_sha256": {name: file_sha256(ROOT / name)
                                             for name in ("requirements.lock", "requirements-ci.lock")}},
            "host": {"platform": platform.platform(), "system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "processor": platform.processor(), "node": platform.node(),
                     "libc": list(platform.libc_ver())},
            "math_backend": {"torch_build_sha256": json_digest(torch.__config__.show()),
                             "torch_parallel_sha256": json_digest(torch.__config__.parallel_info()),
                             "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                             "float32_matmul_precision": torch.get_float32_matmul_precision(),
                             "mkldnn_enabled": torch.backends.mkldnn.enabled,
                             "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
                             "mkl_available": torch.backends.mkl.is_available(),
                             "openmp_available": torch.backends.openmp.is_available(),
                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                             "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                             "torch_threads": torch.get_num_threads(),
                             "torch_interop_threads": torch.get_num_interop_threads(),
                             "flush_denormal_requested": True,
                             "opencv_build_sha256": json_digest(cv2.getBuildInformation()),
                             "opencv_threads": cv2.getNumThreads(), "opencv_optimized": cv2.useOptimized(),
                             "opencv_opencl": cv2.ocl.useOpenCL()},
            "optimizer": "torch.optim.AdamW", "dropout": "TinyVNOCR.drop; pinned model.py"}
    if context is not None:
        identity["prospective_driver"] = json.loads(json.dumps(context))
    return identity


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train_scope(model, scope):
    """A head-only adaptation keeps every encoder and collapse parameter fixed."""
    if scope not in ("all", "head"):
        raise ValueError("unknown train scope")
    model.requires_grad_(scope == "all")
    if scope == "head":
        model.head.requires_grad_(True)


# ----------------------------------------------------------------------------- eval

class ValSet:
    def __init__(self, name, path, opts, height=H):
        self.name = name
        z = np.load(path)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        labels = json.loads(str(z["labels"]))
        meta = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(labels)
        n = int(opts.get("n", 2000))
        native = opts.get("native") == "1"
        self.imgs, self.refs = [], []
        for i in range(len(labels)):
            if len(self.imgs) >= n:
                break
            if not labels[i]:
                continue
            h, w = (int(v) for v in shapes[i])
            g = data[offs[i]:offs[i + 1]].reshape(h, w)
            if native and meta[i].get("inverted"):
                g = 255 - g
            self.imgs.append(resize_h(g, height))
            self.refs.append(norm(labels[i]))
        groups = {}
        for i, g in enumerate(self.imgs):
            groups.setdefault((g.shape[1] + STRIDE - 1) // STRIDE, []).append(i)
        self.groups = groups

    @torch.no_grad()
    def score(self, model, vocab, batch=64):
        chars = [""] + [c if len(c) == 1 else "\ufffd" for c in vocab]
        errs = tot = exact = 0
        rows = []
        for T, idx in self.groups.items():
            for s in range(0, len(idx), batch):
                part = idx[s:s + batch]
                x = np.empty((len(part), 1, H, T * STRIDE), np.float32)
                for j, i in enumerate(part):
                    g = self.imgs[i]
                    x[j, 0, :, :g.shape[1]] = g
                    x[j, 0, :, g.shape[1]:] = g[:, -1:]
                xt = torch.from_numpy(x / 127.5 - 1.0).contiguous(memory_format=torch.channels_last)
                best = model(xt).argmax(2).T.numpy()
                for j, i in enumerate(part):
                    out, last = [], 0
                    for t in best[j, :T]:
                        if t and t != last:
                            out.append(chars[t])
                        last = t
                    pred = norm("".join(out))
                    rows.append({"text": self.refs[i], "pred": pred})
                    e = lev(pred, self.refs[i])
                    errs, tot, exact = errs + e, tot + len(self.refs[i]), exact + (e == 0)
        return errs / max(tot, 1), exact / max(len(self.imgs), 1), score_hiragana(rows)["cer"]


# ----------------------------------------------------------------------------- training

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def kd_loss(logits, kd, tau):
    kid, kp, mask = kd
    lp = (logits.float() / tau).log_softmax(2).permute(1, 0, 2)     # N x T x C
    Tm = min(lp.shape[1], kid.shape[1])
    lp, kid, kp, mask = lp[:, :Tm], kid[:, :Tm], kp[:, :Tm], mask[:, :Tm]
    q = kp.clamp_min(1e-8)
    if tau != 1.0:
        q = q ** (1.0 / tau)
    q = q / q.sum(2, keepdim=True)
    g = lp.gather(2, kid)
    kl = (q * (q.log() - g)).sum(2)
    return (kl * mask).sum() / mask.sum().clamp_min(1.0) * tau * tau


def main(argv=None, *, execution_context=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--src", action="append", required=True, help="NAME:WEIGHT:GLOBS[:OPTS]")
    ap.add_argument("--source-policy", type=Path,
                    help="opt-in exact-prefix general-VN provenance/caps (RAM sources only)")
    ap.add_argument("--val", action="append", default=[], help="NAME:PATH[:OPTS]")
    ap.add_argument("--select", default="", help="comma list of --val names (mean CER)")
    ap.add_argument("--select-metric", choices=("cer", "hiragana"), default="cer",
                    help="select checkpoints on development sets only")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--final-lr", type=float, default=0.05, help="cosine floor, fraction of --lr")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--width", type=float, default=1.5)
    ap.add_argument("--bottleneck", type=int, default=96)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=2)
    ap.add_argument("--ctx-k", type=int, default=5, choices=(3, 5, 7))
    ap.add_argument("--vocab", type=Path, default=DATA / "vocab.txt")
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--kd-tau", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 = off), e.g. 0.999")
    ap.add_argument("--qat", type=float, default=0.0, help="int8 weight fake quant from this fraction")
    ap.add_argument("--init", default="", help="start weights from this checkpoint")
    ap.add_argument("--train-scope", choices=("all", "head"), default="all",
                    help="head freezes encoder/collapse weights and BN statistics")
    ap.add_argument("--freeze-bn-stats", action="store_true",
                    help="freeze BN running statistics; automatic for restored release blobs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=default_threads())
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--keep-evals", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)
    if execution_context is not None and not args.source_policy:
        raise ValueError("prospective execution context requires a source policy")
    if args.source_policy and (args.val or args.select):
        raise ValueError("--source-policy does not accept --val/--select; evaluate retained checkpoints with the "
                         "separately frozen general-VN selector")
    args.freeze_bn_stats |= args.train_scope == "head"
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    torch.manual_seed(args.seed)
    out = args.out / args.run
    out.mkdir(parents=True, exist_ok=True)
    # Hold until process exit, before writing args/logs/checkpoints; also prevents a
    # non-policy invocation of this driver from racing a policy run in the same dir.
    _writer = RunWriter(out)

    vocab = load_vocab(args.vocab)
    encode = encoder(vocab)
    unk = UNK if vocab[-1] == UNK else None
    global KD_REMAP
    if any("kd=1" in s or "kdonly=1" in s for s in args.src):
        KD_REMAP = kd_remap(vocab)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    sources, weights, src_opts, specifications = [], [], [], []
    for spec in args.src:
        parts = spec.split(":", 3)
        name, wt, globs = parts[0], float(parts[1]), parts[2]
        opts = parse_opts(parts[3] if len(parts) > 3 else "")
        paths = expand(globs)
        specifications.append((name, wt, paths, opts))
    policy = SourcePolicy(args.source_policy) if args.source_policy else None
    execution = execution_identity(args.vocab, vocab, execution_context) if policy else None
    provenance = None
    if policy:
        if len({name for name, _, _, _ in specifications}) != len(specifications):
            raise ValueError("source policy requires unique source names")
        for _, _, _, opts in specifications:
            validate_policy_options(opts)
        provenance = policy.bind({name: paths for name, _, paths, _ in specifications})
    for name, wt, paths, opts in specifications:
        big = sum(os.path.getsize(p) for p in paths) > 2e9
        cls = RamSource if policy else (StreamSource if opts.get("stream", "1" if big else "0") == "1" else RamSource)
        srng = np.random.default_rng([args.seed, len(sources), 7])
        sources.append(cls(name, paths, encode, opts, norm, srng,
                           **({"provenance": provenance} if policy else {})))
        weights.append(wt)
        src_opts.append(opts)
        print(f"source {sources[-1].describe()} weight {wt} opts {opts}", flush=True)
    if policy:
        policy.verify_inputs()
    vals = []
    for spec in args.val:
        parts = spec.split(":", 2)
        vals.append(ValSet(parts[0], parts[1], parse_opts(parts[2] if len(parts) > 2 else "")))
        print(f"val {parts[0]}: {len(vals[-1].imgs)} lines from {parts[1]}", flush=True)
    select = [s for s in args.select.split(",") if s] or [v.name for v in vals]
    print(f"data ready in {time.time() - t0:.0f}s", flush=True)

    model = TinyVNOCR(len(vocab) + 1, width=args.width, bottleneck=args.bottleneck,
                      depth=args.depth, ctx=args.ctx, ctx_k=args.ctx_k)
    model = model.to(memory_format=torch.channels_last)
    train_scope(model, args.train_scope)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M, classes {len(vocab) + 1}, "
          f"threads {args.threads}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    step, best, best_step = 0, 1e9, 0
    last = out / "ckpt_last.pt"
    ema = None
    resume_policy, resume_torch_rng, checkpoint_tail = None, None, None
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if bool(policy) != ("source_policy" in ck):
            raise ValueError("resume cannot add or remove the source exposure policy")
        if policy:
            previous_execution = ck.get("source_policy_execution")
            if (not isinstance(previous_execution, dict) or any(previous_execution.get(key) != value
                    for key, value in execution.items()) or "initial_model" not in previous_execution):
                raise ValueError("resume source-policy execution/vocabulary identity differs")
            initial = previous_execution["initial_model"]
            if not isinstance(initial, dict) or initial.get("kind") not in ("checkpoint", "seeded_random"):
                raise ValueError("resume requires the frozen initial model identity")
            digest(initial.get("sha256") if initial["kind"] == "checkpoint" else initial.get("state_sha256"))
            if initial["kind"] == "seeded_random" and initial.get("seed") != args.seed:
                raise ValueError("resume initial model seed differs")
            if args.init and previous_execution["initial_model"].get("sha256") != file_sha256(args.init):
                raise ValueError("resume initial model bytes differ")
            execution = previous_execution
            resume_policy = ck["source_policy"]
            resume_torch_rng = ck.get("torch_rng")
            checkpoint_tail = ck.get("source_policy_journal_sha256")
            if resume_torch_rng is None:
                raise ValueError("policy resume requires saved torch RNG state")
        if ck.get("args", {}).get("train_scope", "all") != args.train_scope:
            raise ValueError("resume train scope differs from checkpoint")
        if ck.get("args", {}).get("select_metric", "cer") != args.select_metric:
            raise ValueError("resume selection metric differs from checkpoint")
        if ck["vocab"] != vocab:
            raise ValueError("resume vocabulary order differs from checkpoint")
        args.freeze_bn_stats |= ck.get("freeze_bn_stats", False)
        model.load_state_dict(ck.get("model_raw", ck["model"]))
        opt.load_state_dict(ck["opt"])
        step, best, best_step = ck["step"], ck["best_sel"], ck.get("best_step", 0)
        if args.ema:
            ema = EMA(model, args.ema)
            ema.shadow = {k: v.clone() for k, v in ck["model"].items()}
        print(f"resumed at step {step}, best {best:.4f}", flush=True)
        if not policy:
            rng = np.random.default_rng([args.seed, step])
    elif args.init:
        init_sha256 = file_sha256(args.init) if policy else None
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if policy:
            if file_sha256(args.init) != init_sha256:
                raise ValueError("initial model bytes changed during loading")
            execution["initial_model"] = {"kind": "checkpoint", "sha256": init_sha256}
        if ck["vocab"] != vocab:
            raise ValueError("init vocabulary order differs from checkpoint")
        args.freeze_bn_stats |= ck.get("freeze_bn_stats", False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {args.init} (step {ck.get('step')})", flush=True)
    if policy and "initial_model" not in execution:
        execution["initial_model"] = {"kind": "seeded_random", "seed": args.seed,
                                      "state_sha256": model_state_sha256(model)}
    if args.ema and ema is None:
        ema = EMA(model, args.ema)
    mixer = (PolicyMixer(sources, weights, args.batch, rng, policy, src_opts)
             if policy else Mixer(sources, weights, args.batch, rng))
    if resume_policy is not None:
        mixer.load_state_dict(resume_policy)
    if policy:
        freeze_json(out / "source_execution.json", execution)
    journal = (ExposureJournal(out / "source_exposures.jsonl", execution, policy.sha256,
                               mixer.ledger.state_dict() if resume_policy is not None else None, checkpoint_tail)
               if policy else None)
    qat_from = int(args.qat * args.steps) if args.qat else None
    cfg = {k: str(v) for k, v in vars(args).items()}
    if policy and resume_policy is not None:
        recipe_keys = ("steps", "seed", "threads", "lr", "final_lr", "warmup", "wd", "width", "bottleneck",
                       "depth", "ctx", "ctx_k", "kd_weight", "kd_tau", "ema", "qat", "freeze_bn_stats")
        if any(ck["args"].get(key) != cfg[key] for key in recipe_keys):
            raise ValueError("source-policy resume training recipe differs from checkpoint")
    (out / "args.json").write_text(json.dumps(cfg, indent=1))
    metrics = open(out / "metrics.jsonl", "a")
    eval_model = TinyVNOCR(len(vocab) + 1, width=args.width, bottleneck=args.bottleneck,
                           depth=args.depth, ctx=args.ctx, ctx_k=args.ctx_k).to(memory_format=torch.channels_last)
    if resume_torch_rng is not None:
        torch.set_rng_state(resume_torch_rng)

    def lr_at(s):
        warm = min(1.0, (s + 1) / max(1, args.warmup))
        prog = min(1.0, max(0, s - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * warm * (args.final_lr + (1 - args.final_lr) * 0.5 * (1 + math.cos(math.pi * prog)))

    t_log, seen, acc = time.time(), 0, {"ctc": 0.0, "kd": 0.0, "n": 0}
    model.train()
    if args.freeze_bn_stats:
        from restore_blob import freeze_bn_stats
        freeze_bn_stats(model)
    while step < args.steps:
        if qat_from is not None and step >= qat_from:
            set_weight_quant(model, 8)
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        items = mixer.next()
        x, in_len, lab, tgt, tgt_len, kd = make_batch(items, rng, src_opts)
        if policy:
            journal.append("intent", mixer.ledger)
        logits = model(x)                                    # T x N x C
        if policy:
            mixer.record_forward(items)
            journal.append("forward", mixer.ledger)
        loss = torch.zeros(())
        if len(lab):
            logp = logits.float().log_softmax(2)
            if len(lab) < len(items):
                logp, il = logp[:, lab], in_len[lab]
            else:
                il = in_len
            c = F.ctc_loss(logp, tgt, il, tgt_len, blank=0, reduction="none", zero_infinity=True)
            c = (c / tgt_len.clamp_min(1)).mean()
            loss = loss + c
            acc["ctc"] += float(c)
        if kd is not None and args.kd_weight > 0:
            k = kd_loss(logits, kd, args.kd_tau)
            loss = loss + args.kd_weight * k
            acc["kd"] += float(k)
        if not loss.requires_grad:
            continue
        if policy and not torch.isfinite(loss):
            raise ValueError("nonfinite training loss; forwarded exposure was journaled, run aborted")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if policy:
            mixer.record_optimizer(items)
            journal.append("optimizer", mixer.ledger)
        if ema is not None:
            ema.update(model)
        step += 1
        seen += len(items)
        acc["n"] += 1
        if step % args.log_every == 0:
            dt = time.time() - t_log
            mix = " ".join(f"{s.name}={c}" for s, c in zip(sources, mixer.counts))
            if policy:
                mix += f" forwarded={mixer.ledger.forwarded} optimizer_rows={mixer.ledger.optimizer}"
            print(f"step {step}/{args.steps} ctc {acc['ctc'] / acc['n']:.4f} kd {acc['kd'] / acc['n']:.4f} "
                  f"lr {lr:.2e} {seen / dt:.0f} lines/s [{mix}]", flush=True)
            t_log, seen, acc = time.time(), 0, {"ctc": 0.0, "kd": 0.0, "n": 0}
        if step % args.eval_every == 0 or step == args.steps:
            sd = ema.shadow if ema is not None else model.state_dict()
            eval_model.load_state_dict(sd)
            if qat_from is not None and step >= qat_from:
                set_weight_quant(eval_model, 8)
            eval_model.eval()
            t1 = time.time()
            res = {v.name: v.score(eval_model, vocab) for v in vals}
            metric_index = 2 if args.select_metric == "hiragana" else 0
            selected = [res[s][metric_index] for s in select]
            if any(v is None for v in selected):
                raise ValueError("selected development set has no hiragana references")
            sel = float(np.mean(selected)) if vals else -float(step)
            rec = {"step": step, "lr": lr, "time": time.time(), "eval_s": round(time.time() - t1, 1),
                   **{f"{k}_cer": v[0] for k, v in res.items()}, **{f"{k}_exact": v[1] for k, v in res.items()},
                   **{f"{k}_hiragana_cer": v[2] for k, v in res.items()},
                   "sel": sel, "mix": {s.name: int(c) for s, c in zip(sources, mixer.counts)}}
            if policy:
                rec["source_policy"] = mixer.report()
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            if sel < best:
                best, best_step = sel, step
            ck = {"model": {k: v.contiguous() for k, v in sd.items()},
                  "opt": opt.state_dict(), "step": step, "best_sel": best, "best_step": best_step,
                  "vocab": vocab, "width": args.width, "bottleneck": args.bottleneck,
                  "depth": args.depth, "ctx": args.ctx, "ctx_k": args.ctx_k, "height": H, "unk": unk,
                  "freeze_bn_stats": args.freeze_bn_stats,
                  "qat": {"bits": 8, "from_step": qat_from} if qat_from is not None else None,
                  "args": cfg, "val": {k: list(v) for k, v in res.items()}}
            if policy:
                ck["source_policy"] = mixer.state_dict()
                ck["torch_rng"] = torch.get_rng_state()
                ck["source_policy_execution"] = execution
                ck["source_policy_journal_sha256"] = journal.tail
            if ema is not None:
                ck["model_raw"] = {k: v.contiguous() for k, v in model.state_dict().items()}
            if best_step == step:
                torch.save(ck, out / "ckpt_best.tmp")
                (out / "ckpt_best.tmp").replace(out / "ckpt_best.pt")
            torch.save(ck, out / "ckpt_last.tmp")
            (out / "ckpt_last.tmp").replace(last)
            if args.keep_evals:
                retained = out / f"ckpt_step{step}.pt"
                torch.save(ck, retained.with_suffix(".tmp"))
                retained.with_suffix(".tmp").replace(retained)
            vs = " ".join(f"{k} {v[0]:.4f}/{v[1]:.3f}" for k, v in res.items())
            print(f"eval step {step}: {vs} sel {sel:.4f} (best {best:.4f} @ {best_step}) "
                  f"[{time.time() - t1:.0f}s]", flush=True)
    print("TRAIN_DONE", flush=True)
    os._exit(0)   # the stream loaders are daemon threads blocked on full queues


if __name__ == "__main__":
    main()

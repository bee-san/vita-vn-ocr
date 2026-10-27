#!/usr/bin/env python3
"""Training throughput of a train/train2.py recipe on this host: steps/s, lines/s, and where a step's time goes.

  bench_train2.py --src NAME:W:GLOBS[:OPTS] ... [--width 1.5] [--chans ...] [--vocab data/vocab_vn_5631.txt]
                  [--batch 96] [--ema 0.999] [--threads 5] [--warmup 30] [--steps 300] --out RESULT.json

The loop is train2's step exactly (its sources, Mixer.next, make_batch with the sources' augmentation and finder
augmentation, the model, CTC as train2 computes it, gradient clipping, AdamW, the EMA update), timed in two parts:
data (Mixer.next + make_batch, train2's main thread) and compute (forward, loss, backward, optimizer, EMA). Stream
sources load their shards in train2's background thread, which competes for the same cores; its cost shows up in
both parts. Startup (loading the RAM sources) is timed separately. The first --warmup steps are not measured.
Nothing is saved but the result: it is a measurement, not a training run.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import hostinfo  # noqa: E402
import train2 as T2  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--width", type=float, default=1.5)
    ap.add_argument("--chans", default="")
    ap.add_argument("--vocab", type=Path, default=ROOT / "data" / "vocab_vn_5631.txt")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    torch.manual_seed(a.seed)
    T2.set_geometry(8)
    vocab = T2.load_vocab(a.vocab)
    encode = T2.encoder(vocab)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    sources, weights, src_opts, described = [], [], [], []
    for spec in a.src:
        parts = spec.split(":", 3)
        name, wt, globs = parts[0], float(parts[1]), parts[2]
        opts = T2.parse_opts(parts[3] if len(parts) > 3 else "")
        paths = T2.expand(globs)
        big = sum(Path(p).stat().st_size for p in paths) > 2e9
        cls = T2.StreamSource if opts.get("stream", "1" if big else "0") == "1" else T2.RamSource
        sources.append(cls(name, paths, encode, opts, T2.norm, np.random.default_rng([a.seed, len(sources), 7])))
        weights.append(wt)
        src_opts.append(opts)
        described.append(sources[-1].describe())
        print(described[-1], flush=True)
    startup = time.time() - t0
    chans = tuple(int(c) for c in a.chans.split(",")) if a.chans else None
    model = T2.TinyVNOCR(len(vocab) + 1, width=a.width, bottleneck=96, depth=1, ctx=2, ctx_k=5, chans=chans,
                         stride=8).to(memory_format=torch.channels_last)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    ema = T2.EMA(model, a.ema) if a.ema else None
    mixer = T2.Mixer(sources, weights, a.batch, rng)
    model.train()
    t_data, t_comp, widths, lines = [], [], [], 0
    for step in range(a.warmup + a.steps):
        s0 = time.perf_counter()
        items = mixer.next()
        x, in_len, lab, tgt, tgt_len, kd = T2.make_batch(items, rng, src_opts)
        s1 = time.perf_counter()
        logits = model(x)
        logp = logits.float().log_softmax(2)
        il = in_len[lab] if len(lab) < len(items) else in_len
        if len(lab) < len(items):
            logp = logp[:, lab]
        c = F.ctc_loss(logp, tgt, il, tgt_len, blank=0, reduction="none", zero_infinity=True)
        loss = (c / tgt_len.clamp_min(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if ema is not None:
            ema.update(model)
        s2 = time.perf_counter()
        if step >= a.warmup:
            t_data.append(s1 - s0)
            t_comp.append(s2 - s1)
            widths.append(int(x.shape[3]))
            lines += len(items)
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}: data {np.mean(t_data[-50:] or [0]):.3f}s compute "
                  f"{np.mean(t_comp[-50:] or [0]):.3f}s loss {float(loss):.3f}", flush=True)
    tot = np.asarray(t_data) + np.asarray(t_comp)
    sps = len(tot) / tot.sum()
    res = {"kind": "train2-throughput", "label": a.label, "host": hostinfo.host_label(), "threads": a.threads,
           "torch": torch.__version__, "gpu": torch.cuda.is_available(), "width": a.width, "chans": a.chans or None,
           "classes": len(vocab) + 1, "params_m": round(sum(p.numel() for p in params) / 1e6, 3),
           "batch": a.batch, "ema": a.ema, "sources": described, "src": a.src, "startup_s": round(startup, 1),
           "warmup_steps": a.warmup, "measured_steps": len(tot), "steps_per_s": round(float(sps), 4),
           "lines_per_s": round(lines / tot.sum(), 1),
           "s_per_step": {"mean": round(float(tot.mean()), 4), "p50": round(float(np.median(tot)), 4),
                          "p90": round(float(np.percentile(tot, 90)), 4)},
           "data_s_per_step": round(float(np.mean(t_data)), 4), "compute_s_per_step": round(float(np.mean(t_comp)), 4),
           "data_share": round(float(np.sum(t_data) / tot.sum()), 4),
           "batch_width_px": {"mean": round(float(np.mean(widths)), 1), "p90": float(np.percentile(widths, 90))},
           "hours_per_10k_steps_one_slot": round(10000 / sps / 3600, 3),
           "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("label", "steps_per_s", "lines_per_s", "data_share",
                                          "hours_per_10k_steps_one_slot", "startup_s")}), flush=True)
    import os
    os._exit(0)                                          # the stream loaders are daemon threads


if __name__ == "__main__":
    main()

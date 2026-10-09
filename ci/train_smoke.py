#!/usr/bin/env python3
"""CI training smoke: render a tiny synthetic set from the pinned OFL font, run exactly
--steps CTC steps of train/train.py on CPU, and check what it wrote.

  train_smoke.py [--workdir DIR] [--steps 50] [--batch 16] [--per-shard 160] [--threads N]

render.py makes two shards in a sandbox copy of the repo (ci/sandbox.py); train.py holds the
last one out for validation, as it does for real runs. The checks:
  - train.py exits 0 and prints TRAIN_DONE;
  - ckpt_last.pt and the last metrics.jsonl record are at step == --steps, with finite numbers;
  - the checkpoint carries data/vocab.txt, loads into TinyVNOCR and has only finite weights;
  - the CTC loss on a fixed set of training lines is lower than at the seeded initialisation
    (compared with batch-statistics BN and no dropout, so only the learned weights differ).
Exit status 0 means every check passed. The workdir is kept on failure or with --keep.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fonts
import sandbox


def default_threads():
    for var in ("TORCH_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    return min(4, os.cpu_count() or 1)


WORK = None


def fail(msg):
    raise SystemExit(f"TRAIN_SMOKE_FAILED: {msg} (workdir kept: {WORK})")


def ctc_loss(model, T, imgs, labels, idx, batch):
    """Mean CTC loss over idx, BN on batch statistics, dropout off, no augmentation."""
    import torch
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum")
    total = n = 0
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x, in_len, tgt, tgt_len = T.make_batch(imgs, labels, idx[i:i + batch])
            total += float(ctc(model(x).log_softmax(2), tgt, in_len, tgt_len))
            n += len(in_len)
    return total / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--keep", action="store_true", help="keep the workdir even on success")
    ap.add_argument("--font-cache", type=Path, default=fonts.DEFAULT_CACHE)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--per-shard", type=int, default=160)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--threads", type=int, default=default_threads())
    ap.add_argument("--timeout", type=int, default=900, help="seconds each for render.py and train.py")
    args = ap.parse_args()
    global WORK
    work = WORK = args.workdir or Path(tempfile.mkdtemp(prefix="vocr-smoke-"))
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        fail(f"workdir {work} is not empty")
    os.environ.update({k: str(args.threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    t0 = time.time()

    font = fonts.ensure_font(args.font_cache)
    repo = sandbox.build(work / "repo", font)
    t = time.time()
    shards = sandbox.render(repo, shards=2, per_shard=args.per_shard, seed=args.seed,
                            workers=min(2, args.threads), timeout=args.timeout)
    render_s = time.time() - t

    # The sandbox's own train.py and model.py, so the checks see exactly what train.py ran.
    sys.path[:0] = [str(repo / "train"), str(repo / "synth")]
    import numpy as np
    import torch

    import train as T
    from model import TinyVNOCR
    torch.set_num_threads(args.threads)
    vocab = T.load_vocab()
    char2id = {c: i + 1 for i, c in enumerate(vocab)}
    tr_imgs, tr_labels, dropped = T.load(shards[:-1], char2id)
    steps_per_epoch = len(tr_imgs) // args.batch
    if steps_per_epoch < 1:
        fail(f"only {len(tr_imgs)} training lines for batch {args.batch}")
    epochs = (args.steps + 0.5) / steps_per_epoch      # train.py runs int(epochs * steps_per_epoch) steps

    torch.manual_seed(args.seed)
    init = TinyVNOCR(len(vocab) + 1, width=args.width)
    init_path = work / "init.pt"
    torch.save({"model": init.state_dict()}, init_path)
    probe = list(range(min(64, len(tr_imgs))))
    loss_init = ctc_loss(init, T, tr_imgs, tr_labels, probe, args.batch)

    cmd = [sys.executable, str(repo / "train" / "train.py"), "--run", "smoke",
           "--synth", str(repo / "data" / "synth" / "synth_*.npz"), "--epochs", repr(epochs),
           "--batch", str(args.batch), "--width", repr(args.width), "--threads", str(args.threads),
           "--eval-every", str(max(1, args.steps // 2)), "--init", str(init_path)]
    print("+", " ".join(cmd), flush=True)
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        print(e.stdout or "", e.stderr or "", sep="\n")
        fail(f"train.py did not finish in {args.timeout}s")
    train_s = time.time() - t
    (work / "train.log").write_text(r.stdout + r.stderr)
    print(r.stdout, r.stderr, sep="", flush=True)
    if r.returncode != 0 or "TRAIN_DONE" not in r.stdout:
        fail(f"train.py exited {r.returncode} (TRAIN_DONE {'seen' if 'TRAIN_DONE' in r.stdout else 'missing'})")

    out = repo / "models" / "smoke"
    recs = [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines() if s]
    if not recs or recs[-1]["step"] != args.steps:
        fail(f"metrics.jsonl ends at step {recs[-1]['step'] if recs else None}, expected {args.steps}")
    if not all(math.isfinite(v) for rec in recs for v in rec.values()):
        fail(f"non-finite metric in {recs}")
    ck = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    if ck["step"] != args.steps:
        fail(f"ckpt_last.pt is at step {ck['step']}, expected {args.steps}")
    if ck["vocab"] != vocab or ck["width"] != args.width:
        fail("ckpt_last.pt vocab/width do not match the run")
    bad = [k for k, v in ck["model"].items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        fail(f"non-finite weights after training: {bad}")
    final = TinyVNOCR(len(vocab) + 1, width=args.width)
    final.load_state_dict(ck["model"])
    final_params = dict(final.named_parameters())
    moved = max(float((final_params[k] - p).abs().max()) for k, p in init.named_parameters())
    loss_final = ctc_loss(final, T, tr_imgs, tr_labels, probe, args.batch)
    if not loss_final < loss_init:
        fail(f"CTC loss on {len(probe)} training lines did not drop: {loss_init:.4f} -> {loss_final:.4f}")

    summary = {"steps": args.steps, "batch": args.batch, "threads": args.threads,
               "train_lines": len(tr_imgs), "val_lines": len(json.loads(str(np.load(shards[-1])["labels"]))),
               "dropped": dropped, "params": sum(p.numel() for p in final.parameters()),
               "ctc_loss_init": round(loss_init, 4), "ctc_loss_final": round(loss_final, 4),
               "max_weight_change": round(moved, 6), "val_cer": round(recs[-1]["val_cer"], 4),
               "render_s": round(render_s, 1), "train_s": round(train_s, 1), "total_s": round(time.time() - t0, 1)}
    print("TRAIN_SMOKE_OK", json.dumps(summary), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("### Training smoke\n\n| | |\n|---|---|\n")
            f.write("".join(f"| {k} | {v} |\n" for k, v in summary.items()) + "\n")
    if not args.keep and args.workdir is None:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()

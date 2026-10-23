#!/usr/bin/env python3
"""Train the tiny Vita VN recognizer with CTC.

  train.py --run v1 [--synth 'data/synth/*.npz'] [--real 'data/real/*.npz' --real-weight 0.3]

Crops (render.py / label_real.py shards) are resized to --height (32) with bilinear
interpolation, the same resize the Vita runtime does. The last synthetic shard is
held out for validation. Writes models/<run>/ckpt_last.pt (resumed automatically),
ckpt_best.pt and metrics.jsonl.

Architecture-sweep options (defaults reproduce v1): --width, --bottleneck, --depth,
--ctx, --ctx-k, --height,
--vocab (a vocab whose last entry is the <unk> token 〓 maps every other character
to it instead of dropping the line), --steps (a fixed budget instead of --epochs),
--seed (seeds torch and numpy) and --qat F (int8 weight fake quantisation from
fraction F of the steps onwards). Threads default to $TORCH_NUM_THREADS (heavy.sh).
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synth"))
from common import DATA, ROOT, load_vocab  # noqa: E402
from model import TinyVNOCR, set_weight_quant  # noqa: E402

H, MAX_W, STRIDE = 32, 1024, 8
UNK = "〓"   # GETA MARK: the <unk> class of a reduced vocab, always its last entry


def resize_h(g, height=H, max_w=MAX_W):
    h, w = g.shape
    nw = max(8, min(max_w, int(round(w * height / h))))
    return cv2.resize(g, (nw, height), interpolation=cv2.INTER_LINEAR)


def resize32(g):
    return resize_h(g, 32)


def ctc_ok(ids, width, stride=STRIDE):
    """True when a label fits the frames of a width-px line: characters plus a blank between
    adjacent repeats (CTC's floor), at `stride` input columns per frame."""
    repeats = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    return len(ids) + repeats <= math.ceil(width / stride)


def encoder(vocab):
    """char -> class id (vocab[i] is class i + 1). With an <unk> vocab (last entry
    UNK) every other character maps to UNK's id; otherwise unknown chars are -1."""
    char2id = {c: i + 1 for i, c in enumerate(vocab)}
    unk = char2id.get(UNK) if vocab and vocab[-1] == UNK else None
    return lambda s: [char2id.get(c, -1 if unk is None else unk) for c in s]


def load(paths, encode, height=H):
    """encode: encoder(vocab), or a plain char2id dict as in v1 (unknown chars drop)."""
    if isinstance(encode, dict):
        char2id = encode
        encode = lambda s: [char2id.get(c, -1) for c in s]  # noqa: E731
    imgs, labels, dropped = [], [], 0
    for p in paths:
        z = np.load(p)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        for i, lab in enumerate(json.loads(str(z["labels"]))):
            ids = encode(lab)
            if not ids or -1 in ids:
                dropped += 1
                continue
            h, w = (int(v) for v in shapes[i])
            g = resize_h(data[offs[i]:offs[i + 1]].reshape(h, w), height)
            if not ctc_ok(ids, g.shape[1]):
                dropped += 1
                continue
            imgs.append(g)
            labels.append(np.asarray(ids, np.int64))
    return imgs, labels, dropped


def augment(g, rng):
    height = g.shape[0]
    g = g.astype(np.float32)
    if rng.random() < 0.3:                                   # horizontal stretch
        nw = max(8, min(MAX_W, int(g.shape[1] * rng.uniform(0.88, 1.12))))
        g = cv2.resize(g, (nw, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:                                   # vertical jitter of the crop
        d = int(rng.integers(1, 3))
        g = g[d:] if rng.random() < 0.5 else np.pad(g, ((d, 0), (0, 0)), mode="edge")
        g = cv2.resize(g, (g.shape[1], height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.2:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 0.9))
    if rng.random() < 0.5:
        g = g * rng.uniform(0.7, 1.15) + rng.uniform(-25, 25)
    if rng.random() < 0.2:
        g = g + rng.normal(0, rng.uniform(2, 8), g.shape)
    return np.clip(g, 0, 255)


def batches(widths, batch, rng):
    """Width-bucketed batches in random order."""
    order = np.argsort(np.asarray(widths) + rng.uniform(0, 24, len(widths)))
    chunks = [order[i:i + batch] for i in range(0, len(order), batch)]
    rng.shuffle(chunks)
    return chunks


def make_batch(imgs, labels, idx, rng=None):
    gs = [augment(imgs[i], rng) if rng is not None else imgs[i].astype(np.float32) for i in idx]
    W = (max(g.shape[1] for g in gs) + STRIDE - 1) // STRIDE * STRIDE
    x = np.empty((len(gs), 1, gs[0].shape[0], W), np.float32)
    for j, g in enumerate(gs):
        w = g.shape[1]
        x[j, 0, :, :w] = g
        x[j, 0, :, w:] = g[:, -1:]
    # channels_last: same values, ~1.4x faster CPU convs (N x 1 x H x W has one channel,
    # so this only tags the layout that the conv outputs then inherit)
    x = torch.from_numpy(x / 127.5 - 1.0).contiguous(memory_format=torch.channels_last)
    in_len = torch.tensor([math.ceil(g.shape[1] / STRIDE) for g in gs], dtype=torch.long)
    tgt = [labels[i] for i in idx]
    return x, in_len, torch.from_numpy(np.concatenate(tgt)), torch.tensor([len(t) for t in tgt])


def greedy(logits, in_len):
    best = logits.argmax(2).T.numpy()            # N x T
    out = []
    for row, n in zip(best, in_len.tolist()):
        seq, last = [], 0
        for t in row[:n]:
            if t and t != last:
                seq.append(int(t))
            last = t
        out.append(seq)
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, imgs, labels, n=3000, batch=128):
    model.eval()
    idx = np.arange(min(n, len(imgs)))
    errs = chars = exact = 0
    for chunk in [idx[i:i + batch] for i in range(0, len(idx), batch)]:
        x, in_len, _, _ = make_batch(imgs, labels, chunk)
        for pred, i in zip(greedy(model(x), in_len), chunk):
            ref = labels[i].tolist()
            e = edit_distance(pred, ref)
            errs, chars, exact = errs + e, chars + len(ref), exact + (e == 0)
    model.train()
    return errs / max(chars, 1), exact / max(len(idx), 1)


def default_threads():
    for var in ("TORCH_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    return 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    ap.add_argument("--out", type=Path, default=ROOT / "models", help="runs directory")
    ap.add_argument("--synth", default=str(DATA / "synth" / "synth_*.npz"))
    ap.add_argument("--real", default="")
    ap.add_argument("--real-weight", type=float, default=0.3)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--steps", type=int, default=0, help="total steps (overrides --epochs)")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--bottleneck", type=int, default=96)
    ap.add_argument("--depth", type=int, default=1, help="stride-1 DS blocks per stage (v1: 1)")
    ap.add_argument("--ctx", type=int, default=2, help="temporal Ctx blocks (v1: 2)")
    ap.add_argument("--ctx-k", type=int, default=5, choices=(3, 5, 7), help="Ctx kernel width")
    ap.add_argument("--height", type=int, default=H, choices=(24, 32))
    ap.add_argument("--vocab", type=Path, default=DATA / "vocab.txt")
    ap.add_argument("--qat", type=float, default=0.0,
                    help="int8 weight fake quantisation from this fraction of the steps (0 = off)")
    ap.add_argument("--seed", type=int, default=None, help="seed torch and numpy (default: v1)")
    ap.add_argument("--threads", type=int, default=default_threads())
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--keep-evals", action="store_true",
                    help="also keep ckpt_step<N>.pt at every eval (learning curves)")
    ap.add_argument("--init", default="", help="start weights from this checkpoint")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    out = args.out / args.run
    out.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.vocab)
    encode = encoder(vocab)
    unk = UNK if vocab[-1] == UNK else None
    shards = sorted(glob.glob(args.synth))
    t0 = time.time()
    tr_imgs, tr_labels, d1 = load(shards[:-1], encode, args.height)
    va_imgs, va_labels, d2 = load(shards[-1:], encode, args.height)
    src = [0] * len(tr_imgs)
    if args.real:
        real = sorted(glob.glob(args.real))
        r_imgs, r_labels, d3 = load(real, encode, args.height)
        tr_imgs += r_imgs
        tr_labels += r_labels
        src += [1] * len(r_imgs)
        print(f"real: {len(r_imgs)} lines from {len(real)} shards ({d3} dropped)", flush=True)
    print(f"train {len(tr_imgs)} / val {len(va_imgs)} lines, dropped {d1 + d2}, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    model = TinyVNOCR(len(vocab) + 1, width=args.width, bottleneck=args.bottleneck,
                      depth=args.depth, ctx=args.ctx, ctx_k=args.ctx_k)
    model = model.to(memory_format=torch.channels_last)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M, classes "
          f"{len(vocab) + 1}, height {args.height}, threads {args.threads}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    step, best = 0, 1.0
    last = out / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best_cer"]
        print(f"resumed at step {step}, best CER {best:.4f}", flush=True)
    elif args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {args.init}", flush=True)

    rng = np.random.default_rng(step + 1 if args.seed is None else [args.seed, step])
    widths = [g.shape[1] for g in tr_imgs]
    weights = np.where(np.asarray(src) == 1, args.real_weight, 1.0)
    steps_per_epoch = len(tr_imgs) // args.batch
    total = args.steps or int(args.epochs * steps_per_epoch)
    qat_from = int(args.qat * total) if args.qat else None
    warm = 1000
    metrics = open(out / "metrics.jsonl", "a")
    t_log, seen, loss_acc = time.time(), 0, 0.0
    model.train()
    while step < total:
        if args.real:
            # Resample the epoch so real lines appear at the requested weight.
            p = weights / weights.sum()
            pool = rng.choice(len(tr_imgs), size=len(tr_imgs), p=p)
            chunks = [pool[c] for c in batches([widths[i] for i in pool], args.batch, rng)]
        else:
            chunks = batches(widths, args.batch, rng)
        for idx in chunks:
            if step >= total:
                break
            if qat_from is not None and step >= qat_from:
                set_weight_quant(model, 8)
            lr = args.lr * min(1.0, (step + 1) / warm) * (
                0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
            for g in opt.param_groups:
                g["lr"] = lr
            x, in_len, tgt, tgt_len = make_batch(tr_imgs, tr_labels, idx, rng)
            logp = model(x).log_softmax(2)
            loss = ctc(logp, tgt, in_len, tgt_len)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            seen += len(idx)
            loss_acc += float(loss)
            if step % 100 == 0:
                dt = time.time() - t_log
                print(f"step {step}/{total} loss {loss_acc / 100:.4f} lr {lr:.2e} "
                      f"{seen / dt:.0f} lines/s", flush=True)
                t_log, seen, loss_acc = time.time(), 0, 0.0
            if step % args.eval_every == 0 or step == total:
                cer, acc = evaluate(model, va_imgs, va_labels)
                rec = {"step": step, "epoch": step / steps_per_epoch, "val_cer": cer,
                       "val_line_acc": acc, "lr": lr, "time": time.time()}
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                ck = {"model": {k: v.contiguous() for k, v in model.state_dict().items()},
                      "opt": opt.state_dict(), "step": step,
                      "best_cer": min(best, cer), "vocab": vocab, "width": args.width,
                      "bottleneck": args.bottleneck, "depth": args.depth, "ctx": args.ctx,
                      "ctx_k": args.ctx_k, "height": args.height, "unk": unk,
                      "qat": {"bits": 8, "from_step": qat_from} if qat_from is not None else None,
                      "args": {k: str(v) for k, v in vars(args).items()}}
                if cer < best:
                    best = cer
                    torch.save(ck, out / "ckpt_best.pt")
                torch.save(ck, out / "ckpt_last.tmp")
                (out / "ckpt_last.tmp").replace(last)
                if args.keep_evals:
                    torch.save(ck, out / f"ckpt_step{step}.pt")
                print(f"eval step {step}: val CER {cer:.4f} line acc {acc:.4f} (best {best:.4f})",
                      flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()

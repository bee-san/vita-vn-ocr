#!/usr/bin/env python3
"""Train one text detector shape (issue #96, docs/textdet.md, eval/textdet_protocol.json "training").

  textdet_train.py --shape s4_c8 --seed 1 --manifest data/manifests/textdet_shards_v1.json --shards DIR ... \
                   --out RUN [--steps 30000] [--threads 5]

The run writes RUN/log.jsonl (losses every 200 steps), RUN/final.pt (the quantisation-aware weights: folded
convs, activation scales) and RUN/final.vtd (the int8 blob, train/textdet_export.py). The last step is the run's
model; nothing is selected inside a run.

Batches: 16 windows of at most 192 x 512 px, half synthetic and half real (realplus 0.35, teacher 0.35, owner
0.25, devnative 0.05 of the real half). A canvas that fits is used whole; a bigger one gives a random window, or
(real canvases, half the time) a player-like region around a group of lines (their union plus a margin of 4-20 px,
half that at the PSP target). A window smaller than 192 x 512 sits in the top left and every layer's output is
masked to it, so the network computes exactly what the runtime computes on the window alone. A line cut across by
the window's edge becomes an ignore area; one cut along its length keeps the part inside.

Targets at the output stride s (train/textdet_model.py): a line's kernel cells are those whose centre
s*i + (s-1)/2 lies within max(h/4, s/2) of the line's centre across it (the nearest cell when none) and whose
block meets the line along it; K = 1 there, A1 / A2 = the distances from the cell centre to the line's near / far
edge, O = 1 for vertical lines. A cell two lines claim is ignored. Cells meeting an ignore area carry no K loss
unless they are a line's kernel. Loss: K BCE + dice over the valid cells; O BCE on kernel cells x 0.5; A1 / A2
smooth L1 of (pred - target) / 8 px on kernel cells.

Optimiser: AdamW 4e-3, 500 warm-up steps, cosine to 1e-4, weight decay 1e-4 on conv weights only. QAT from step
24,000 (80 %): activation scales from the 99.99th percentile of each layer's output on 32 batches, BatchNorm
folded, int8 fake quantisation of weights and activations for the rest of the run.
"""
import argparse
import datetime
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as EX  # noqa: E402
import textdet_model as TM  # noqa: E402

WIN_H, WIN_W = 192, 512
BATCH = 16
MIX = {"synth": 0.5, "realplus": 0.175, "teacher": 0.175, "owner": 0.125, "devnative": 0.025}
LR, LR_MIN, WARMUP, WD = 4e-3, 1e-4, 500, 1e-4
QAT_FRAC = 0.8
CAL_BATCHES = 32


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- data

class Pool:
    """Every shard of the manifest, memory-mapped, grouped by source (synth or a real source name)."""

    def __init__(self, manifest, dirs):
        man = json.loads(Path(manifest).read_text(encoding="utf-8"))
        roots = {Path(d).name: Path(d) for d in dirs}
        self.items = {k: [] for k in MIX}
        self.data = []
        for s in man["shards"]:
            base = roots[s["dir"]] / s["name"]
            data = np.load(base.with_suffix(".npy"), mmap_mode="r")
            idx = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["samples"]
            d = len(self.data)
            self.data.append(data)
            for k, smp in enumerate(idx):
                src = "synth" if smp["src"].startswith("synth") else smp["src"]
                self.items[src].append((d, smp))
        self.keys = [k for k in MIX if self.items[k]]
        p = np.array([MIX[k] for k in self.keys])
        self.p = p / p.sum()

    def canvas(self, d, smp):
        return np.asarray(self.data[d][smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(smp["h"], smp["w"])


def group_region(lines, k, rng, H, W, psp):
    """A player-like region around horizontal line k: the lines stacked with it (gap <= 1.6 line heights,
    overlapping columns), their union plus a margin."""
    hz = [ln for ln in lines if not ln[4]]
    if not hz:
        return None
    g = [hz[k % len(hz)]]
    changed = True
    while changed:
        changed = False
        x0, y0 = min(b[0] for b in g), min(b[1] for b in g)
        x1, y1 = max(b[2] for b in g), max(b[3] for b in g)
        hmax = max(b[3] - b[1] for b in g)
        for b in hz:
            if b in g:
                continue
            gap = max(b[1] - y1, y0 - b[3])
            if gap <= 1.6 * max(hmax, b[3] - b[1]) and min(b[2], x1) > max(b[0], x0):
                g.append(b)
                changed = True
    x0, y0 = min(b[0] for b in g), min(b[1] for b in g)
    x1, y1 = max(b[2] for b in g), max(b[3] for b in g)
    m = int(rng.integers(4, 21)) // (2 if psp else 1)
    extra = int(rng.choice([0, 0, 4, 8])) // (2 if psp else 1)
    m += extra
    return max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)


def window_of(rng, smp, canvas_shape, real):
    H, W = canvas_shape
    if real and smp["lines"] and rng.random() < 0.5:
        reg = group_region(smp["lines"], int(rng.integers(1 << 30)), rng, H, W, smp.get("kind") == "psp")
        if reg is not None:
            x0, y0, x1, y1 = reg
            h, w = min(WIN_H, y1 - y0), min(WIN_W, x1 - x0)
            oy = y0 + int(rng.integers(0, y1 - y0 - h + 1))
            ox = x0 + int(rng.integers(0, x1 - x0 - w + 1))
            return ox, oy, w, h
    h, w = min(WIN_H, H), min(WIN_W, W)
    return int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1)), w, h


def targets(lines, ignore, wx, wy, w, h, s):
    """Window targets at stride s: K, O, A1, A2 (float32), valid and kernel masks (bool), (WIN_H/s, WIN_W/s)."""
    ho, wo = WIN_H // s, WIN_W // s
    K = np.zeros((ho, wo), np.float32)
    O = np.zeros((ho, wo), np.float32)  # noqa: E741
    A1 = np.zeros((ho, wo), np.float32)
    A2 = np.zeros((ho, wo), np.float32)
    owner = np.full((ho, wo), -1, np.int32)
    conflict = np.zeros((ho, wo), bool)
    valid = np.zeros((ho, wo), bool)
    vh, vw = (h + s - 1) // s, (w + s - 1) // s
    valid[:vh, :vw] = True
    ign = np.zeros((ho, wo), bool)

    def block_mark(m, b):
        x0, y0, x1, y1 = b
        if x1 <= x0 or y1 <= y0:
            return
        m[max(0, y0 // s):min(ho, (y1 - 1) // s + 1), max(0, x0 // s):min(wo, (x1 - 1) // s + 1)] = True

    for b in ignore:
        block_mark(ign, (b[0] - wx, b[1] - wy, b[2] - wx, b[3] - wy))
    centres = np.arange(max(ho, wo)) * s + (s - 1) / 2
    for n, ln in enumerate(lines):
        x0, y0, x1, y1 = ln[0] - wx, ln[1] - wy, ln[2] - wx, ln[3] - wy
        vert = bool(ln[4])
        a0, a1, b0, b1 = (x0, x1, y0, y1) if vert else (y0, y1, x0, x1)       # across, along
        lim_a = w if vert else h
        lim_b = h if vert else w
        if a1 <= 0 or a0 >= lim_a or b1 <= 0 or b0 >= lim_b:
            continue
        if a0 < 0 or a1 > lim_a:                     # cut across by the window: unknown extent
            block_mark(ign, (max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
            continue
        b0, b1 = max(0, b0), min(lim_b, b1)
        across = a1 - a0
        c = (a0 + a1) / 2
        r = max(across / 4, s / 2)
        n_across = ho if not vert else wo
        idx = np.flatnonzero(np.abs(centres[:n_across] - c) <= r)
        if not len(idx):
            idx = np.array([int(np.clip(round((c - (s - 1) / 2) / s), 0, n_across - 1))])
        along = np.arange(b0 // s, (b1 - 1) // s + 1)
        along = along[along < (wo if not vert else ho)]
        if not len(along):
            continue
        if vert:
            ii, jj = np.meshgrid(along, idx, indexing="ij")
        else:
            ii, jj = np.meshgrid(idx, along, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        cc = centres[jj] if vert else centres[ii]
        taken = owner[ii, jj] >= 0
        conflict[ii[taken], jj[taken]] = True
        owner[ii, jj] = n
        K[ii, jj] = 1
        O[ii, jj] = float(vert)
        A1[ii, jj] = cc - a0
        A2[ii, jj] = a1 - cc
    kern = (K > 0) & ~conflict & valid
    valid_k = valid & ~conflict & (~ign | kern)
    K[conflict] = 0
    return K, O, A1, A2, valid_k, kern


class Batches(torch.utils.data.IterableDataset):
    def __init__(self, manifest, dirs, stride, seed):
        super().__init__()
        self.manifest, self.dirs, self.stride, self.seed = manifest, dirs, stride, seed

    def __iter__(self):
        pool = Pool(self.manifest, self.dirs)
        rng = np.random.default_rng(self.seed)
        s = self.stride
        while True:
            x = np.full((BATCH, 1, WIN_H, WIN_W), 128, np.uint8)
            valid_hw = []
            tg = [np.zeros((BATCH, WIN_H // s, WIN_W // s), np.float32) for _ in range(4)]
            vm = np.zeros((BATCH, WIN_H // s, WIN_W // s), bool)
            km = np.zeros_like(vm)
            for b in range(BATCH):
                src = pool.keys[int(rng.choice(len(pool.keys), p=pool.p))]
                lst = pool.items[src]
                d, smp = lst[int(rng.integers(len(lst)))]
                cv = pool.canvas(d, smp)
                wx, wy, w, h = window_of(rng, smp, cv.shape, src != "synth")
                x[b, 0, :h, :w] = cv[wy:wy + h, wx:wx + w]
                valid_hw.append((h, w))
                K, O, A1, A2, v, k = targets(smp["lines"], smp["ignore"], wx, wy, w, h, s)  # noqa: E741
                for t, a in zip(tg, (K, O, A1, A2)):
                    t[b] = a
                vm[b], km[b] = v, k
            yield (torch.from_numpy(x), torch.tensor(valid_hw), *(torch.from_numpy(t) for t in tg),
                   torch.from_numpy(vm), torch.from_numpy(km))


# ----------------------------------------------------------------------------- training

def lr_at(step, steps):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, steps - WARMUP)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * min(1.0, t)))


def loss_of(out, K, O, A1, A2, vm, km):  # noqa: E741
    k_logit = out[:, 0]
    vmf = vm.float()
    n_valid = vmf.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(k_logit, K, reduction="none") * vmf).sum() / n_valid
    p = torch.sigmoid(k_logit) * vmf
    dice = 1 - (2 * (p * K).sum() + 1) / (p.sum() + (K * vmf).sum() + 1)
    kmf = km.float()
    n_k = kmf.sum().clamp(min=1)
    lo = (F.binary_cross_entropy_with_logits(out[:, 1], O, reduction="none") * kmf).sum() / n_k
    la = ((F.smooth_l1_loss((out[:, 2] - A1) / 8, torch.zeros_like(A1), reduction="none") +
           F.smooth_l1_loss((out[:, 3] - A2) / 8, torch.zeros_like(A2), reduction="none")) * kmf).sum() / n_k
    total = bce + dice + 0.5 * lo + la
    return total, {"k_bce": float(bce), "dice": float(dice), "o": float(lo), "a": float(la)}


def param_groups(params):
    decay, no = [], []
    for p in params:
        (decay if p.dim() == 4 else no).append(p)
    return [{"params": decay, "weight_decay": WD}, {"params": no, "weight_decay": 0.0}]


def to_input(x):
    return (x.float() - 128) / 128


def calibrate(model, it, shape, n=CAL_BATCHES):
    """Per layer: mean over n batches of the 99.99th percentile of |output| over the valid positions."""
    model.eval()
    acc = None
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _ in range(n):
            x, vhw = next(it)[:2]
            masks = TM.layer_masks(shape, vhw.tolist(), WIN_H, WIN_W)
            taps = []
            model(to_input(x), masks=masks, taps=taps)
            vals = []
            for t, m in zip(taps, masks):
                a = t.abs()[m.expand_as(t) > 0]
                if a.numel() > 200_000:
                    a = a[torch.randint(0, a.numel(), (200_000,), generator=gen)]
                vals.append(float(torch.quantile(a, 0.9999)) if a.numel() else 1.0)
            acc = vals if acc is None else [p + q for p, q in zip(acc, vals)]
    model.train()
    return [max(v / n, 1e-4) / 127 for v in acc]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", required=True, choices=sorted(TM.SHAPES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = TM.SHAPES[args.shape]
    s = shape["stride"]
    model = TM.TextDet(shape)
    opt = torch.optim.AdamW(param_groups(list(model.parameters())), lr=LR)
    loader = torch.utils.data.DataLoader(Batches(args.manifest, args.shards, s, args.seed), batch_size=None,
                                         num_workers=1, persistent_workers=True, prefetch_factor=4)
    it = iter(loader)
    qat_start = int(QAT_FRAC * args.steps)
    run = {"kind": "textdet-run", "shape": shape, "seed": args.seed, "steps": args.steps, "qat_start": qat_start,
           "batch": BATCH, "window": [WIN_H, WIN_W], "mix": MIX, "lr": [LR, LR_MIN, WARMUP], "wd": WD,
           "manifest_sha256": sha256(args.manifest),
           "code_sha256": {f: sha256(ROOT / f) for f in ("train/textdet_train.py", "train/textdet_model.py",
                                                         "train/textdet_export.py", "synth/textdet_twin.py")},
           "torch": torch.__version__, "threads": args.threads,
           "started": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    agg, n_agg = {}, 0
    act_scale = None
    for step in range(args.steps):
        if step == qat_start:
            act_scale = calibrate(model, it, shape)
            model.eval()
            folded = TM.fold(model)
            model = TM.QTextDet(shape, folded, act_scale)
            opt = torch.optim.AdamW(param_groups(list(model.parameters())), lr=lr_at(step, args.steps))
            log.write(json.dumps({"step": step, "qat": True, "act_scale": act_scale}) + "\n")
            log.flush()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps)
        x, vhw, K, O, A1, A2, vm, km = next(it)  # noqa: E741
        masks = TM.layer_masks(shape, vhw.tolist(), WIN_H, WIN_W)
        y = model(to_input(x), masks=masks)
        loss, parts = loss_of(y, K, O, A1, A2, vm, km)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n_agg += 1
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            rec = {"step": step + 1, "lr": round(lr_at(step, args.steps), 6), "sec": round(time.time() - t0, 1),
                   **{k: round(v / n_agg, 5) for k, v in agg.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            agg, n_agg = {}, 0
    ck = {"kind": "textdet-qat", "shape": shape, "seed": args.seed, "steps": args.steps,
          "w": [w.detach().double() for w in model.w], "b": [b.detach().double() for b in model.b],
          "act_scale": act_scale, "run": run}
    torch.save(ck, out / "final.pt")
    blob = EX.export_checkpoint(out / "final.pt")
    (out / "final.vtd").write_bytes(blob)
    done = {"finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seconds": round(time.time() - t0, 1), "blob_bytes": len(blob),
            "blob_sha256": hashlib.sha256(blob).hexdigest()}
    (out / "done.json").write_text(json.dumps(done, indent=1) + "\n")
    print("TEXTDET_TRAIN_DONE", json.dumps(done), flush=True)


if __name__ == "__main__":
    main()

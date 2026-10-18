#!/usr/bin/env python3
"""Train one text-presence gate (docs/textgate.md, eval/textgate_protocol.json "training").

  textgate_train.py --shape p8_c8 --seed 1 --real-manifest data/manifests/textdet_shards_v1.json \
                    --synth-manifest data/manifests/textgate_shards_v1.json --shards DIR ... --out RUN

A gate is a text detector network of train/textdet_model.py (a 3x3 stride-2 stem, separable downsampling to output
stride 4 or 8, a few separable blocks, a 1x1 head) trained for presence only: the head's first map (K) is a text
logit per output cell; its other three maps carry no loss (the blob keeps the VTDT layout of synth/textdet_twin.py
and runtime/vocr_textdet.c, so either runs it). The gate's decision is made from the K map (eval/textgate.py).

Data: #97's real shards (realplus, teacher, owner, devnative) without the calibration split's VNs
(eval/textgate_data.calib_vn), and the clean synthetic shards (eval/textgate_data.py synth), half and half per
batch as #97 mixes them; #97's windows (train/textdet_train.window_of: 192 x 512 px at most, player-like regions
around line groups half the time on real canvases), its validity masks and its QAT.

Targets at the output stride s: a cell is text (1) when its block's centre lies inside a horizontal line's box
(for a line no centre falls in, the cells of the row nearest the line's centre that the line spans); a cell whose
block meets a horizontal line's box but is not text, or meets a vertical line or an ignore area, carries no loss;
every other valid cell is not text (0). Loss: BCE + dice over the valid cells, plus 0.5 x the BCE of each window's
highest logit against "the window holds text" (a multiple-instance term: the gate reads the highest cells).

Optimiser, schedule and QAT: #97's (AdamW 4e-3, 500 warm-up steps, cosine to 1e-4, weight decay on conv weights,
int8 fake quantisation from 80 % of the run, activation scales from 32 batches).
"""
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as EX  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_train as TT  # noqa: E402

BATCH, WIN_H, WIN_W = TT.BATCH, TT.WIN_H, TT.WIN_W
MIX = TT.MIX
MIL_WEIGHT = 0.5
SALT = "textgate-split:"


def calib_vn(vn):
    """eval/textgate_data.calib_vn (copied here so training needs no eval import)."""
    return int(hashlib.sha256(f"{SALT}{vn}".encode()).hexdigest()[:8], 16) % 5 == 0


def shape(name, stride, c0, c1, chans, dil, c2=None, block="sep"):
    return {"name": name, "stride": stride, "c0": c0, "c1": c1, "c2": c2, "chans": list(chans), "dil": list(dil),
            "block": block, "head": "kroa"}


# The presence shapes of the protocol's sweep (eval/textgate_protocol.json "sweep"), all much smaller than #97's
# s8_c24 (5,700 B): output stride 8 with 4-12 channels, and stride 4 with 4-8.
SHAPES = {s["name"]: s for s in (
    shape("p8_c4", 8, 4, 4, [4], [1], c2=4),
    shape("p8_c6", 8, 4, 6, [6, 6], [1, 2], c2=6),
    shape("p8_c8", 8, 4, 8, [8, 8], [1, 2], c2=8),
    shape("p8_c12", 8, 4, 8, [12, 12], [1, 2], c2=12),
    shape("p4_c4", 4, 4, 4, [4, 4], [1, 2]),
    shape("p4_c8", 4, 4, 8, [8, 8], [1, 2]),
)}


class Pool(TT.Pool):
    """#97's pool from two manifests (real and synthetic), without the calibration split's VNs."""

    def __init__(self, real_manifest, synth_manifest, dirs):
        self.items = {k: [] for k in MIX}
        self.data = []
        roots = {Path(d).name: Path(d) for d in dirs}
        self.dropped = 0
        for man_path, want in ((real_manifest, "real"), (synth_manifest, "synth")):
            man = json.loads(Path(man_path).read_text(encoding="utf-8"))
            for s in man["shards"]:
                if (s["dir"] == "synth") != (want == "synth") and want == "real":
                    continue                       # #97's own synthetic shards are not used (their scene pool)
                base = roots[s["dir"]] / s["name"]
                data = np.load(base.with_suffix(".npy"), mmap_mode="r")
                idx = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["samples"]
                d = len(self.data)
                self.data.append(data)
                for smp in idx:
                    src = "synth" if smp["src"].startswith("synth") else smp["src"]
                    if src != "synth" and calib_vn(smp["vn"]):
                        self.dropped += 1
                        continue
                    self.items[src].append((d, smp))
        self.keys = [k for k in MIX if self.items[k]]
        p = np.array([MIX[k] for k in self.keys])
        self.p = p / p.sum()


def targets(lines, ignore, wx, wy, w, h, s):
    """(P, valid) at stride s for a window: P 1 text / 0 not; valid False where there is no loss."""
    ho, wo = WIN_H // s, WIN_W // s
    P = np.zeros((ho, wo), np.float32)
    nol = np.zeros((ho, wo), bool)
    valid = np.zeros((ho, wo), bool)
    vh, vw = (h + s - 1) // s, (w + s - 1) // s
    valid[:vh, :vw] = True
    cy = np.arange(ho) * s + (s - 1) / 2
    cx = np.arange(wo) * s + (s - 1) / 2

    def meets(b):
        x0, y0, x1, y1 = b
        if x1 <= x0 or y1 <= y0:
            return None
        return (slice(max(0, y0 // s), min(ho, (y1 - 1) // s + 1)), slice(max(0, x0 // s), min(wo, (x1 - 1) // s + 1)))

    for b in ignore:
        sl = meets((b[0] - wx, b[1] - wy, b[2] - wx, b[3] - wy))
        if sl:
            nol[sl] = True
    for ln in lines:
        x0, y0, x1, y1 = ln[0] - wx, ln[1] - wy, ln[2] - wx, ln[3] - wy
        sl = meets((max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
        if sl is None or x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        nol[sl] = True
        if ln[4] or y0 < 0 or y1 > h:                 # vertical text, or cut across by the window: no loss
            continue
        rows = np.flatnonzero((cy >= y0) & (cy < y1))
        if not len(rows):
            rows = np.array([int(np.clip(round(((y0 + y1) / 2 - (s - 1) / 2) / s), 0, ho - 1))])
        cols = np.flatnonzero((cx >= max(0, x0)) & (cx < min(w, x1)))
        if not len(cols):
            cols = np.array([int(np.clip(round(((x0 + x1) / 2 - (s - 1) / 2) / s), 0, wo - 1))])
        P[np.ix_(rows, cols)] = 1
    valid &= (P > 0) | ~nol
    return P, valid


class Batches(torch.utils.data.IterableDataset):
    def __init__(self, real_manifest, synth_manifest, dirs, stride, seed):
        super().__init__()
        self.args = real_manifest, synth_manifest, dirs
        self.stride, self.seed = stride, seed

    def __iter__(self):
        pool = Pool(*self.args)
        rng = np.random.default_rng(self.seed)
        s = self.stride
        while True:
            x = np.full((BATCH, 1, WIN_H, WIN_W), 128, np.uint8)
            valid_hw = []
            P = np.zeros((BATCH, WIN_H // s, WIN_W // s), np.float32)
            V = np.zeros((BATCH, WIN_H // s, WIN_W // s), bool)
            for b in range(BATCH):
                src = pool.keys[int(rng.choice(len(pool.keys), p=pool.p))]
                lst = pool.items[src]
                d, smp = lst[int(rng.integers(len(lst)))]
                cv = pool.canvas(d, smp)
                wx, wy, w, h = TT.window_of(rng, smp, cv.shape, src != "synth")
                x[b, 0, :h, :w] = cv[wy:wy + h, wx:wx + w]
                valid_hw.append((h, w))
                P[b], V[b] = targets(smp["lines"], smp["ignore"], wx, wy, w, h, s)
            yield torch.from_numpy(x), torch.tensor(valid_hw), torch.from_numpy(P), torch.from_numpy(V)


def loss_of(out, P, V):
    k = out[:, 0]
    vf = V.float()
    n = vf.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(k, P, reduction="none") * vf).sum() / n
    p = torch.sigmoid(k) * vf
    dice = 1 - (2 * (p * P).sum() + 1) / (p.sum() + (P * vf).sum() + 1)
    masked = torch.where(V, k, torch.full_like(k, -1e4))
    top = masked.flatten(1).max(1).values
    has = ((P * vf).flatten(1).sum(1) > 0).float()
    keep = V.flatten(1).any(1)
    mil = F.binary_cross_entropy_with_logits(top[keep], has[keep]) if keep.any() else top.sum() * 0
    return bce + dice + MIL_WEIGHT * mil, {"bce": float(bce), "dice": float(dice), "mil": float(mil)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", required=True, choices=sorted(SHAPES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--real-manifest", required=True)
    ap.add_argument("--synth-manifest", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shp = SHAPES[args.shape]
    s = shp["stride"]
    model = TM.TextDet(shp)
    opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=TT.LR)
    loader = torch.utils.data.DataLoader(Batches(args.real_manifest, args.synth_manifest, args.shards, s, args.seed),
                                         batch_size=None, num_workers=1, persistent_workers=True, prefetch_factor=4)
    it = iter(loader)
    qat_start = int(TT.QAT_FRAC * args.steps)
    run = {"kind": "textgate-run", "shape": shp, "seed": args.seed, "steps": args.steps, "qat_start": qat_start,
           "batch": BATCH, "window": [WIN_H, WIN_W], "mix": MIX, "mil_weight": MIL_WEIGHT,
           "lr": [TT.LR, TT.LR_MIN, TT.WARMUP], "wd": TT.WD,
           "real_manifest_sha256": TT.sha256(args.real_manifest),
           "synth_manifest_sha256": TT.sha256(args.synth_manifest),
           "code_sha256": {f: TT.sha256(ROOT / f) for f in ("train/textgate_train.py", "train/textdet_train.py",
                                                            "train/textdet_model.py", "train/textdet_export.py")},
           "torch": torch.__version__, "threads": args.threads,
           "started": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    agg, n_agg = {}, 0
    act_scale = None
    for step in range(args.steps):
        if step == qat_start:
            act_scale = TT.calibrate(model, it, shp)
            model.eval()
            folded = TM.fold(model)
            model = TM.QTextDet(shp, folded, act_scale)
            opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=TT.lr_at(step, args.steps))
            log.write(json.dumps({"step": step, "qat": True, "act_scale": act_scale}) + "\n")
            log.flush()
        for g in opt.param_groups:
            g["lr"] = TT.lr_at(step, args.steps)
        x, vhw, P, V = next(it)
        masks = TM.layer_masks(shp, vhw.tolist(), WIN_H, WIN_W)
        y = model(TT.to_input(x), masks=masks)
        loss, parts = loss_of(y, P, V)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n_agg += 1
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            rec = {"step": step + 1, "lr": round(TT.lr_at(step, args.steps), 6), "sec": round(time.time() - t0, 1),
                   **{k: round(v / n_agg, 5) for k, v in agg.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            agg, n_agg = {}, 0
    ck = {"kind": "textdet-qat", "shape": shp, "seed": args.seed, "steps": args.steps,
          "w": [w.detach().double() for w in model.w], "b": [b.detach().double() for b in model.b],
          "act_scale": act_scale, "run": run}
    torch.save(ck, out / "final.pt")
    blob = EX.export_checkpoint(out / "final.pt")
    (out / "final.vtd").write_bytes(blob)
    done = {"finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seconds": round(time.time() - t0, 1), "blob_bytes": len(blob),
            "blob_sha256": hashlib.sha256(blob).hexdigest()}
    (out / "done.json").write_text(json.dumps(done, indent=1) + "\n")
    print("TEXTGATE_TRAIN_DONE", json.dumps(done), flush=True)


if __name__ == "__main__":
    main()

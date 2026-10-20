#!/usr/bin/env python3
"""Train one textspot2 run of the full-frame text spotter (issue #106, docs/textspot2.md "Recipes" and "Shapes",
eval/textspot2_protocol.json "recipes"); runs are named <recipe>_h2_<shape>_seed<n>.

  textspot2_train.py --recipe lab|data|ema|cls|long|kd --shape S --seed N --manifest M ... --shards DIR ... \
                     --out RUN [--teacher RUN_DIR] [--extras ema,cls|none] [--steps N] [--threads 5]
  textspot2_train.py ... --probe 200     speed probe: one warm-up step, then 200 steps of the run, timed
                                         (RUN/probe.json; no model is written)

textspot's trainer (train/textspot_train.py) and #97's (train/textdet_train.py) are imported and used unchanged
wherever the recipe does not change them: the pool with its sha256 checks and skip lists (FramePool, skip_lists,
skip_record), the window cut at h2 (window and half_box: textspot_twin.halve's exact 2x2 mean), the 4-px ignore rule,
the mix, the optimiser groups and weight decay, the gradient clip, #97's QAT calibration, the BatchNorm fold,
QTextDet, the checkpoint, the export (train/textdet_export.py) and the VTSP pack (synth/textspot_twin.py). The shapes
are the protocol's (train/textspot2_shapes.SHAPES, used as dicts: train/textdet_model.SHAPES is left as it is). Every
run is h2 (half_above 600). r0, textspot's recipe, is train/textspot_train.py itself and is not run here.

What the recipes change (docs/textspot2.md "Recipes", "Data" b):
- weights (every recipe here): a line's 7th field (index 6, default 1) multiplies the K BCE, O and A losses of its
  kernel cells; every other valid cell weighs 1; dice and every normalisation are #97's. targets() is a copy of
  textdet_train.targets (the round-1 kernel rule) that also returns the weight map; with every weight 1 it equals
  textdet_train.targets and loss_of equals textdet_train.loss_of exactly, so a lab run on #97's labels is textspot's
  run bit for bit (tests/test_textspot2_train.py checks it).
- cls: weight 2 on the lines of class dialogue, name and choice (index 5), times the line's own weight. The
  synthetic frames carry classes; the real canvases are "text".
- ema: an exponential moving average (decay 0.999) of the parameters, started from the weights after step 1,000
  and updated after every later step. At the QAT start the average replaces the live parameters and the BatchNorm
  statistics are re-estimated on 32 batches (the training-mode forward without gradient; running statistics reset,
  then their cumulative average), then come #97's calibration (32 batches), fold and QTextDet. During QAT a second
  average (decay 0.999) of the QAT parameters runs, started from the folded weights at the QAT start; it is the
  run's model (final.pt's w and b).
- long: 60,000 steps, peak LR 2e-3, 1,000 warm-up steps, cosine to 1e-4, QAT from step 48,000 (80 %), with ema.
- kd: distillation from a teacher run's final.pt (#97's QAT checkpoint, kind textdet-qat, run as QTextDet in torch
  with its fake quantisation on the same window at the same scale and output stride): loss = the label loss +
  BCE(K, sigmoid(K_t)) over the valid cells (the label loss's valid mask and normalisation) + 0.5 BCE(O,
  sigmoid(O_t)) + smooth L1((A1 - A1_t) / 8) + smooth L1((A2 - A2_t) / 8), the last three averaged over the cells
  where sigmoid(K_t) > 0.5. The teacher's terms carry no line weights.
- --extras sets the extras of data, long and kd, the recipes stage 2 combines with "the kept extras": data none,
  long ema, kd required (none, ema, cls or ema,cls).

Data: --manifest and --shards as in textspot_train.py (a shard is read when its manifest's dir is the basename of a
--shards directory; every shard's npy and json sha256 is checked first; the skip lists of every data pin are applied,
and a textspot2 pin without them is refused). The protocol's directories (the conductor, 2026-10-20T09:00Z): lab
reads synth_frames and lab_<source> and leaves only more_<source> shards unread; every other recipe reads
synth_frames, lab_<source> and more_<source> and every shard of its manifests; #97's real directories never.
--any-shards lifts this for probes and tests (recorded).

Outputs (the last step is the run's model; nothing is selected inside a run; RUN must not exist or be empty, so a
crashed run restarts from scratch in a fresh directory):
  RUN/run.json    started (UTC), the arguments, the recipe and every setting, the name, the shape, the seed, the
                  manifests' sha256, the skip record, the shards read, the pool per source with its line weights,
                  the teacher (kd), the sha256 of every train/ and synth/ file imported, both protocols' sha256 and
                  the checkout's commit (with any modified tracked file)
  RUN/log.jsonl   #97's records every 200 steps (step, lr, sec, k_bce, dice, o, a; kd_k, kd_o, kd_a for kd; loss,
                  the total optimised), the EMA's start and swap records and the QAT record
  RUN/final.pt    #97's quantisation-aware checkpoint (kind textdet-qat; with ema, the QAT average)
  RUN/final.vtd   the int8 VTDT blob (train/textdet_export.export_checkpoint)
  RUN/final.vtsp  textspot_twin.pack(final.vtd, 600, 1)
  RUN/done.json   finished (UTC), steps, wall time, the blobs' bytes and sha256
"""
import argparse
import collections
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
import textdet_train as TT  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot_train as ST  # noqa: E402
import textspot_twin as TS  # noqa: E402

SCALE = "h2"
HALF_ABOVE = ST.HALF_ABOVE[SCALE]       # 600
WIN_H, WIN_W = ST.WIN_H, ST.WIN_W       # 256 x 512
BATCH, MIX, MIN_ACROSS, CLIP = ST.BATCH, ST.MIX, ST.MIN_ACROSS, ST.CLIP
REAL = tuple(k for k in MIX if k != "synth")
CLS_CLASSES = ("dialogue", "name", "choice")
CLS_FACTOR = 2.0
EMA_DECAY = 0.999
EMA_START = 1000
BN_BATCHES = 32
EXTRAS = ("ema", "cls")
SET_EXTRAS = ("data", "long", "kd")     # the recipes stage 2 combines with "the kept extras"
RECIPES = {
    "lab": {"arm": "b", "extras": (), "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
            "what": "r0 with the real half's labels replaced by the textspot2 labels (the same canvases)"},
    "data": {"arm": "c", "extras": (), "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
             "what": "lab plus the enlarged real pool"},
    "ema": {"arm": "e", "extras": ("ema",), "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
            "what": "data + an exponential moving average of the weights"},
    "cls": {"arm": "f", "extras": ("cls",), "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
            "what": "data + class weights (2 on dialogue, name and choice)"},
    "long": {"arm": "e", "extras": ("ema",), "steps": 60000, "lr": 2e-3, "warmup": 1000,
             "what": "data + ema + 60,000 steps, peak LR 2e-3, 1,000 warm-up steps, cosine to 1e-4"},
    "kd": {"arm": "e", "extras": None, "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
           "what": "data + the kept extras + distillation from a teacher spotter"},
}
SYNTH_DIR = "synth_frames"
PROTOCOL = ROOT / "eval" / "textspot2_protocol.json"
TEXTSPOT_PROTOCOL = ROOT / "eval" / "textspot_protocol.json"
SHAPE_FIELDS = ("name", "stride", "c0", "c1", "c2", "chans", "dil", "block", "head")


def shape_of(name):
    """A protocol shape as textdet_model's functions take it (TextDet, layers, layer_masks, cost, the export): the
    same dict textspot2_shapes.register() would add, and textdet_model's own where it defines the name (textspot's
    s4p_c8 and s4_c24n4)."""
    s = {k: S2.SHAPES[name][k] for k in SHAPE_FIELDS}
    s["chans"], s["dil"] = list(s["chans"]), list(s["dil"])
    have = TM.SHAPES.get(name)
    if have is not None and {k: have[k] for k in SHAPE_FIELDS} != s:
        raise ValueError(f"{name}: train/textdet_model.SHAPES defines another shape under this name")
    return s


# ----------------------------------------------------------------------------- the schedule

def lr_at(step, steps, peak=TT.LR, warmup=TT.WARMUP, lr_min=TT.LR_MIN):
    """textdet_train.lr_at with the recipe's peak and warm-up: linear warm-up to peak over warmup steps, then cosine
    from peak to lr_min over the remaining steps. With #97's constants it is textdet_train.lr_at exactly."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return lr_min + 0.5 * (peak - lr_min) * (1 + math.cos(math.pi * min(1.0, t)))


# ----------------------------------------------------------------------------- weights and targets

def line_weight(ln, cls=False):
    """A shard line's loss weight: its 7th field (index 6; 1 when it has none), times CLS_FACTOR when cls is on and
    its class (index 5) is dialogue, name or choice."""
    w = 1.0 if len(ln) < 7 or ln[6] is None else float(ln[6])
    if not (math.isfinite(w) and w > 0):
        raise ValueError(f"line {list(ln[:5])}: weight {ln[6]!r} is not a positive number")
    if cls and len(ln) > 5 and ln[5] in CLS_CLASSES:
        w *= CLS_FACTOR
    return w


def at_scale(lines, ignore, f, cls=False):
    """textspot_train.at_scale with each positive's weight: (positives [[x0, y0, x1, y1, vertical, weight]],
    ignore areas). The same boxes, the same 4-px rule (a line under MIN_ACROSS px across at the network's scale
    becomes an ignore area)."""
    def to(b):
        return ST.half_box(b) if f == 2 else [int(v) for v in b[:4]]

    ign = [to(b) for b in ignore]
    pos = []
    for ln in lines:
        b = to(ln)
        vertical = int(ln[4])
        if (b[2] - b[0] if vertical else b[3] - b[1]) < MIN_ACROSS:
            ign.append(b)
        else:
            pos.append([*b, vertical, line_weight(ln, cls)])
    return pos, ign


def targets(lines, ignore, wx, wy, w, h, s):
    """textdet_train.targets (round-1 kernel rule) on the recipe's WIN_H x WIN_W window, plus the weight map:
    (K, O, A1, A2, valid, kernel, W). lines: [[x0, y0, x1, y1, vertical, weight]] (at_scale's positives; a line
    without index 5 weighs 1). W is a kernel cell's line weight and 1 on every other cell. Everything else is
    #97's code, line for line."""
    ho, wo = WIN_H // s, WIN_W // s
    K = np.zeros((ho, wo), np.float32)
    O = np.zeros((ho, wo), np.float32)  # noqa: E741
    A1 = np.zeros((ho, wo), np.float32)
    A2 = np.zeros((ho, wo), np.float32)
    Wt = np.ones((ho, wo), np.float32)
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
        dist = np.abs(centres[:n_across] - c)
        idx = np.flatnonzero(dist <= r)
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
        Wt[ii, jj] = float(ln[5]) if len(ln) > 5 else 1.0
    kern = (K > 0) & ~conflict & valid
    valid_k = valid & ~conflict & (~ign | kern)
    K[conflict] = 0
    Wt[~kern] = 1.0
    return K, O, A1, A2, valid_k, kern, Wt


def make_batch(pool, rng, s, cls=False):
    """textspot_train.make_batch at h2 (the same draws in the same order, so the same windows) plus the weight map:
    (x, valid (h, w), K, O, A1, A2, valid mask, kernel mask, W)."""
    x = np.full((BATCH, 1, WIN_H, WIN_W), 128, np.uint8)
    valid_hw = []
    tg = [np.zeros((BATCH, WIN_H // s, WIN_W // s), np.float32) for _ in range(5)]
    vm = np.zeros((BATCH, WIN_H // s, WIN_W // s), bool)
    km = np.zeros_like(vm)
    for b in range(BATCH):
        src = pool.keys[int(rng.choice(len(pool.keys), p=pool.p))]
        lst = pool.items[src]
        d, smp = lst[int(rng.integers(len(lst)))]
        img, ox, oy, w, h, f = ST.window(rng, pool.canvas(d, smp), SCALE)
        x[b, 0, :h, :w] = img
        valid_hw.append((h, w))
        lines, ignore = at_scale(smp["lines"], smp["ignore"], f, cls)
        K, O, A1, A2, v, k, Wt = targets(lines, ignore, ox, oy, w, h, s)  # noqa: E741
        for t, a in zip(tg, (K, O, A1, A2, Wt)):
            t[b] = a
        vm[b], km[b] = v, k
    K, O, A1, A2, Wt = (torch.from_numpy(t) for t in tg)  # noqa: E741
    return torch.from_numpy(x), torch.tensor(valid_hw), K, O, A1, A2, torch.from_numpy(vm), torch.from_numpy(km), Wt


class FrameBatches(torch.utils.data.IterableDataset):
    def __init__(self, manifests, dirs, stride, seed, cls=False, skip=(frozenset(), frozenset())):
        super().__init__()
        self.manifests, self.dirs = list(manifests), list(dirs)
        self.stride, self.seed, self.cls = stride, seed, cls
        self.skip = tuple(frozenset(x) for x in skip)

    def __iter__(self):
        pool = ST.FramePool(self.manifests, self.dirs, skip=self.skip)
        rng = np.random.default_rng(self.seed)
        while True:
            yield make_batch(pool, rng, self.stride, self.cls)


# ----------------------------------------------------------------------------- losses

def loss_of(out, K, O, A1, A2, vm, km, Wt):  # noqa: E741
    """textdet_train.loss_of with the weight map: the K BCE, O BCE and A smooth-L1 terms of every cell are multiplied
    by W (a kernel cell's line weight, 1 elsewhere); dice and the normalisations (valid cells, kernel cells) are
    #97's. With W = 1 it is textdet_train.loss_of exactly."""
    k_logit = out[:, 0]
    vmf = vm.float()
    n_valid = vmf.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(k_logit, K, reduction="none") * vmf * Wt).sum() / n_valid
    p = torch.sigmoid(k_logit) * vmf
    dice = 1 - (2 * (p * K).sum() + 1) / (p.sum() + (K * vmf).sum() + 1)
    kmf = km.float()
    n_k = kmf.sum().clamp(min=1)
    lo = (F.binary_cross_entropy_with_logits(out[:, 1], O, reduction="none") * kmf * Wt).sum() / n_k
    la = ((F.smooth_l1_loss((out[:, 2] - A1) / 8, torch.zeros_like(A1), reduction="none") +
           F.smooth_l1_loss((out[:, 3] - A2) / 8, torch.zeros_like(A2), reduction="none")) * kmf * Wt).sum() / n_k
    total = bce + dice + 0.5 * lo + la
    return total, {"k_bce": float(bce), "dice": float(dice), "o": float(lo), "a": float(la)}


def kd_loss(out, t, vm):
    """The distillation terms: BCE(K, sigmoid(K_t)) over the valid cells (vm, normalised by their count as the
    label K BCE is) + 0.5 BCE(O, sigmoid(O_t)) + smooth L1((A1 - A1_t) / 8) + smooth L1((A2 - A2_t) / 8), the last
    three averaged over the cells where sigmoid(K_t) > 0.5. t: the teacher's maps (no gradient)."""
    t = t.detach()
    vmf = vm.float()
    n_valid = vmf.sum().clamp(min=1)
    pk = torch.sigmoid(t[:, 0])
    kd_k = (F.binary_cross_entropy_with_logits(out[:, 0], pk, reduction="none") * vmf).sum() / n_valid
    tm = (pk > 0.5).float()
    n_t = tm.sum().clamp(min=1)
    kd_o = (F.binary_cross_entropy_with_logits(out[:, 1], torch.sigmoid(t[:, 1]), reduction="none") * tm).sum() / n_t
    zero = torch.zeros_like(out[:, 2])
    kd_a = ((F.smooth_l1_loss((out[:, 2] - t[:, 2]) / 8, zero, reduction="none") +
             F.smooth_l1_loss((out[:, 3] - t[:, 3]) / 8, zero, reduction="none")) * tm).sum() / n_t
    total = kd_k + 0.5 * kd_o + kd_a
    return total, {"kd_k": float(kd_k), "kd_o": float(kd_o), "kd_a": float(kd_a), "kd_cells": int(tm.sum())}


# ----------------------------------------------------------------------------- EMA

class EMA:
    """An exponential moving average of a parameter list, in float64: the first step() copies the parameters,
    every later one sets avg = decay * avg + (1 - decay) * p. copy_to() writes the average into parameters of the
    same order."""

    def __init__(self, decay=EMA_DECAY):
        self.decay = float(decay)
        self.avg = None
        self.steps = 0

    @torch.no_grad()
    def step(self, params):
        params = list(params)
        if self.avg is None:
            self.avg = [p.detach().double().clone() for p in params]
        else:
            if len(params) != len(self.avg):
                raise ValueError("EMA: another parameter list")
            for a, p in zip(self.avg, params):
                a.mul_(self.decay).add_(p.detach().double(), alpha=1.0 - self.decay)
        self.steps += 1
        return self

    @torch.no_grad()
    def copy_to(self, params):
        params = list(params)
        if self.avg is None or len(params) != len(self.avg):
            raise ValueError("EMA: nothing averaged, or another parameter list")
        for a, p in zip(self.avg, params):
            p.copy_(a.to(p.dtype))


def reestimate_bn(model, it, shape, n):
    """The BatchNorm statistics re-estimated on n batches: running statistics reset, momentum None (a cumulative
    average of the batch statistics), the training-mode forward with the layer masks and no gradient; the model's
    mode and momenta are restored. -> the number of BatchNorm layers."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    was_training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(n):
            x, vhw = next(it)[:2]
            masks = TM.layer_masks(shape, vhw.tolist(), WIN_H, WIN_W)
            model(TT.to_input(x), masks=masks)
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    model.train(was_training)
    return len(bns)


# ----------------------------------------------------------------------------- the teacher (kd)

def load_teacher(run_dir, stride):
    """A teacher run: its final.pt (kind textdet-qat) as a QTextDet with fake quantisation, frozen; its run.json must
    say h2 (half_above 600) and its shape the student's output stride. -> (model, shape, record)."""
    d = Path(run_dir)
    pt, info_path = d / "final.pt", d / "run.json"
    if not pt.exists() or not info_path.exists():
        raise SystemExit(f"{d}: a teacher run needs final.pt and run.json")
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    if ck.get("kind") != "textdet-qat":
        raise SystemExit(f"{pt}: not a quantisation-aware textdet checkpoint")
    shape = ck["shape"]
    if shape.get("head", "kroa") != "kroa" or shape["stride"] != stride:
        raise SystemExit(f"{pt}: a {shape.get('head', 'kroa')} head at stride {shape['stride']}, not kroa at {stride}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("half_above") != HALF_ABOVE:
        raise SystemExit(f"{info_path}: half_above {info.get('half_above')}, not {HALF_ABOVE} (h2)")
    model = TM.QTextDet(shape, list(zip(ck["w"], ck["b"])), ck["act_scale"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vtsp = d / "final.vtsp"
    rec = {"dir": str(d), "name": info.get("name"), "kind": info.get("kind"), "recipe": info.get("recipe"),
           "shape": shape["name"], "stride": shape["stride"], "half_above": info.get("half_above"),
           "steps": ck.get("steps"), "seed": ck.get("seed"), "pt_sha256": TT.sha256(pt),
           "vtsp_sha256": TT.sha256(vtsp) if vtsp.exists() else None,
           "how": "final.pt as TM.QTextDet (fake quantisation on), eval mode, no gradient, on the student's window "
                  "with its own layer masks"}
    return model, shape, rec


# ----------------------------------------------------------------------------- data checks and records

def is_textspot2_pin(path, doc):
    return doc.get("kind") == "textspot2-shards" or Path(path).name.startswith("textspot2_shards")


def skip_lists(manifests):
    """textspot_train.skip_lists, after refusing a textspot2 data pin without real.measured.skip's two lists."""
    for man in manifests:
        doc = json.loads(Path(man).read_text(encoding="utf-8"))
        if is_textspot2_pin(man, doc):
            skip = ((doc.get("real") or {}).get("measured") or {}).get("skip") or {}
            if not ((skip.get("eval_screenshot") or {}).get("skip_shots") is not None and
                    (skip.get("linefind_list_vn") or {}).get("skip_vns") is not None):
                raise SystemExit(f"{man}: a textspot2 data pin without real.measured.skip's two lists")
    return ST.skip_lists(manifests)


def skip_per_manifest(manifests, read):
    """Expected against dropped, per manifest: the canvases the skip lists dropped from its shards (FramePool.read's
    samples - kept) and, for a textspot2 pin, 0 expected: its shards were built without the canvases its lists name
    (amendment 1), and its counts are textspot's (#97's real half), carried for reference. So the skip record's
    pin_differs (textspot's counts against what this run dropped) is explained here."""
    per = {}
    for r in read:
        d = per.setdefault(r["manifest"], {"shards_read": 0, "samples": 0, "kept": 0})
        d["shards_read"] += 1
        d["samples"] += r["samples"]
        d["kept"] += r["kept"]
    out = []
    for man in manifests:
        doc = json.loads(Path(man).read_text(encoding="utf-8"))
        skip = ((doc.get("real") or {}).get("measured") or {}).get("skip")
        t2 = is_textspot2_pin(man, doc)
        p = per.get(str(man), {"shards_read": 0, "samples": 0, "kept": 0})
        rec = {"manifest": str(man), "kind": doc.get("kind"), "textspot2_pin": t2,
               "carries_skip_lists": skip is not None, "shards_read": p["shards_read"],
               "samples_read": p["samples"], "dropped": p["samples"] - p["kept"],
               "expected_dropped": 0 if t2 else None,
               "pin_counts": {k: {kk: vv for kk, vv in (skip.get(k) or {}).items()
                                  if kk not in ("skip_shots", "skip_vns")}
                              for k in ("eval_screenshot", "linefind_list_vn")} if skip else None}
        if t2:
            rec["as_expected"] = rec["dropped"] == 0
            rec["why"] = ("a textspot2 pin's shards were built without the canvases its skip lists name (amendment "
                          "1), so the trainer drops none of them; the lists are applied again here all the same, and "
                          "the pin's counts are textspot's (#97's real half), kept for reference")
            if not rec["as_expected"]:
                print(f"WARNING {man}: the skip lists dropped {rec['dropped']} canvases of a textspot2 pin's shards "
                      "(expected 0)", flush=True)
        elif skip is not None:
            rec["why"] = ("a textspot data pin: its lists name canvases of #97's real half "
                          "(data/manifests/textdet_shards_v1.json), not of its own shards")
        out.append(rec)
    return out


def data_check(recipe, manifests, dirs, pool):
    """The protocol's directories: synth_frames and lab_<source> for lab (only more_<source> shards left unread);
    synth_frames, lab_<source> and more_<source> for every other recipe, with every shard of its manifests read;
    nothing else (never #97's real directories). -> the record; SystemExit when it does not hold."""
    names = [Path(d).name for d in dirs]
    lab = [f"lab_{s}" for s in REAL]
    more = [f"more_{s}" for s in REAL]
    allowed = {SYNTH_DIR, *lab, *(more if recipe != "lab" else ())}
    bad = sorted(set(names) - allowed)
    if bad:
        raise SystemExit(f"{recipe}: --shards {bad} are not the protocol's directories ({sorted(allowed)}); "
                         "--any-shards is for probes and tests")
    if SYNTH_DIR not in names or not set(lab) & set(names) or (recipe != "lab" and not set(more) & set(names)):
        raise SystemExit(f"{recipe}: the protocol's run reads {SYNTH_DIR}, lab_<source>"
                         f"{' and more_<source>' if recipe != 'lab' else ''} directories")
    unread = {}
    for key, n in pool.skipped.items():
        man, _, d = key.rpartition(":")
        unread.setdefault(d, 0)
        unread[d] += n
    wrong = sorted(d for d in unread if recipe != "lab" or d not in more)
    if wrong:
        raise SystemExit(f"{recipe}: the manifests' shards in {wrong} are not read (pass their directories)")
    return {"rule": "lab: synth_frames + lab_<source> (only more_<source> unread); other recipes: synth_frames + "
                    "lab_<source> + more_<source>, every manifest shard read; never #97's real directories",
            "dirs": names, "unread": unread, "manifests": [str(m) for m in manifests]}


def pool_weights(pool):
    """Per source: lines, how many weigh other than 1, their weight sum, min and max (every weight checked by
    line_weight), and the lines per class (cls doubles dialogue, name and choice)."""
    out = {}
    for src, items in pool.items.items():
        ws = [line_weight(ln) for _, s in items for ln in s["lines"]]
        cls = collections.Counter(str(ln[5]) if len(ln) > 5 else "" for _, s in items for ln in s["lines"])
        out[src] = {"lines": len(ws), "weighted": sum(1 for w in ws if w != 1.0),
                    "weight_sum": round(float(sum(ws)), 3), "weight_min": min(ws) if ws else None,
                    "weight_max": max(ws) if ws else None, "classes": dict(sorted(cls.items())),
                    "cls_lines": sum(n for c, n in cls.items() if c in CLS_CLASSES)}
    return out


def parse_extras(text):
    """'ema,cls' | 'ema' | 'cls' | 'none' -> a tuple in EXTRAS order; None stays None."""
    if text is None:
        return None
    items = [x.strip() for x in text.split(",") if x.strip()]
    if items == ["none"]:
        return ()
    if not items or any(x not in EXTRAS for x in items) or len(set(items)) != len(items):
        raise ValueError(f"--extras {text!r}: a comma list of {', '.join(EXTRAS)}, or none")
    return tuple(x for x in EXTRAS if x in items)


def run_name(recipe, shape, seed):
    return f"{recipe}_{SCALE}_{shape}_seed{seed}"


# ----------------------------------------------------------------------------- the run

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, choices=list(RECIPES))
    ap.add_argument("--shape", required=True, choices=sorted(S2.SHAPES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", help="kd: the teacher run's directory (its final.pt and run.json)")
    ap.add_argument("--extras", help="data, long and kd: the extras, a comma list of ema, cls, or none (stage 2's "
                                     "kept extras; default: data none, long ema; kd needs it)")
    ap.add_argument("--steps", type=int, help="default: the recipe's (30,000; long 60,000)")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="speed probe: one warm-up step, then N steps of the run's schedule, timed; RUN/probe.json")
    ap.add_argument("--cal-batches", type=int, default=TT.CAL_BATCHES, help="QAT calibration batches (#97: 32)")
    ap.add_argument("--bn-batches", type=int, default=BN_BATCHES, help="ema: BatchNorm re-estimation batches (32)")
    ap.add_argument("--ema-start", type=int, default=EMA_START,
                    help="ema: the average starts from the weights after this many steps (1,000)")
    ap.add_argument("--workers", type=int, default=1, help="loader processes (#97: 1)")
    ap.add_argument("--any-shards", action="store_true",
                    help="probes and tests: read any shard directories (not the protocol's data rule)")
    ap.add_argument("--allow-missing-sources", action="store_true",
                    help="tests: train on whichever of the mix's sources the shards hold")
    args = ap.parse_args(argv)
    r = RECIPES[args.recipe]
    if args.shape in S2.D2S:
        ap.error(f"{args.shape}: a d2s shape (head kroa_x2) needs the d2s format (synth/textspot2_twin.py, "
                 "train/textspot2_export.py), which is not on this branch yet")
    try:
        extras = parse_extras(args.extras)
    except ValueError as e:
        ap.error(str(e))
    if extras is not None and args.recipe not in SET_EXTRAS:
        ap.error(f"--extras is for {', '.join(SET_EXTRAS)}; {args.recipe}'s extras are the protocol's")
    if args.recipe == "kd" and extras is None:
        ap.error("kd: --extras is required (the kept extras: none, ema, cls or ema,cls)")
    if (args.recipe == "kd") != bool(args.teacher):
        ap.error("--teacher is required for kd and only for kd")
    args.extras_set = r["extras"] if extras is None else extras
    args.steps = r["steps"] if args.steps is None else args.steps
    if (args.steps < 1 or args.probe < 0 or args.cal_batches < 1 or args.bn_batches < 1 or args.ema_start < 1
            or args.workers < 0 or args.log_every < 1):
        ap.error("--steps, --cal-batches, --bn-batches, --ema-start, --log-every >= 1; --probe, --workers >= 0")
    if "ema" in args.extras_set and args.ema_start > int(TT.QAT_FRAC * args.steps):
        ap.error(f"ema: --ema-start {args.ema_start} is after the QAT start {int(TT.QAT_FRAC * args.steps)}")
    return args


def recipe_record(args, qat_start, peak, warmup, teacher):
    ex = args.extras_set
    return {
        "recipe": args.recipe, "arm": RECIPES[args.recipe]["arm"], "what": RECIPES[args.recipe]["what"],
        "extras": list(ex), "batch": BATCH, "window": [WIN_H, WIN_W], "mix": MIX, "scale": SCALE,
        "half_above": HALF_ABOVE, "steps": args.steps,
        "lr": {"peak": peak, "min": TT.LR_MIN, "warmup": warmup,
               "form": "peak (step + 1) / warmup during the warm-up, then cosine from peak to min over the "
                       "remaining steps (textdet_train.lr_at's form)"},
        "wd": TT.WD, "clip": CLIP, "qat_frac": TT.QAT_FRAC, "qat_start": qat_start, "cal_batches": args.cal_batches,
        "kernel_rule": "round 1 (textdet_train.targets strict=False), copied with the weight map",
        "min_across": MIN_ACROSS,
        "halve": "textspot_twin.halve when the canvas is wider than half_above; boxes start floor(v/2), end ceil(v/2)",
        "windows": "uniformly random offset; a side that fits is whole; no region crops",
        "positives": "every line of every class; under min_across px across: ignore",
        "weights": {"field": 6, "default": 1.0,
                    "rule": "a line's weight multiplies the K BCE, O BCE and A smooth-L1 terms of its kernel cells; "
                            "every other valid cell weighs 1; dice and the normalisations are #97's"},
        "cls": {"classes": list(CLS_CLASSES), "factor": CLS_FACTOR,
                "rule": "x2 on the lines of these classes (index 5), times the line's own weight"}
        if "cls" in ex else None,
        "ema": {"decay": EMA_DECAY, "start_after_step": args.ema_start, "bn_batches": args.bn_batches,
                "qat_decay": EMA_DECAY,
                "rule": "the average starts from the weights after start_after_step steps and is updated after every "
                        "later step; at the QAT start it replaces the live parameters, the BatchNorm statistics are "
                        "re-estimated (reset, momentum None, bn_batches training-mode forwards without gradient), "
                        "then #97's calibration, fold and QTextDet; during QAT an average (qat_decay) of the QAT "
                        "parameters from the folded weights on is the run's model"}
        if "ema" in ex else None,
        "kd": {"teacher": teacher,
               "loss": "label loss + BCE(K, sigmoid(K_t)) over the valid cells (/ their count) + 0.5 BCE(O, "
                       "sigmoid(O_t)) + smooth L1((A1 - A1_t) / 8) + smooth L1((A2 - A2_t) / 8), the last three "
                       "averaged over the cells where sigmoid(K_t) > 0.5; no line weights"}
        if args.recipe == "kd" else None,
        "loader_workers": args.workers,
    }


def main(argv=None):
    args = parse(argv)
    t_proc = time.time()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: a run starts from scratch in a fresh directory")
    started = ST.utc()
    r = RECIPES[args.recipe]
    shape = shape_of(args.shape)
    s = shape["stride"]
    teacher = t_shape = t_rec = None
    if args.recipe == "kd":
        teacher, t_shape, t_rec = load_teacher(args.teacher, s)
    skip_shots, skip_vns, lists = skip_lists(args.manifest)
    pool = ST.FramePool(args.manifest, args.shards, verify=True, skip=(skip_shots, skip_vns))
    missing = [k for k in MIX if not pool.items[k]]
    if missing and not args.allow_missing_sources:
        raise SystemExit(f"no samples of {missing} (the mix needs all of {list(MIX)})")
    data_rec = ("any shard directories (--any-shards: a probe or a test)" if args.any_shards
                else data_check(args.recipe, args.manifest, args.shards, pool))
    pool_summary = pool.summary()
    pool_summary["weights"] = pool_weights(pool)
    skip = ST.skip_record(lists, pool.dropped)
    skip["per_manifest"] = skip_per_manifest(args.manifest, pool.read)
    skip["pin_differs_note"] = ("pin_expects sums the counts every list-carrying manifest records (textspot's, of "
                                "#97's real half); per_manifest gives what each manifest's own shards lost")
    del pool
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cls, ema_on = "cls" in args.extras_set, "ema" in args.extras_set
    peak, warmup = r["lr"], r["warmup"]
    model = TM.TextDet(shape)
    opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=peak)
    loader = torch.utils.data.DataLoader(FrameBatches(args.manifest, args.shards, s, args.seed, cls,
                                                      (skip_shots, skip_vns)),
                                         batch_size=None, num_workers=args.workers,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None)
    it = iter(loader)
    qat_start = int(TT.QAT_FRAC * args.steps)
    name = run_name(args.recipe, args.shape, args.seed)
    run = {"kind": "textspot2-run", "name": name, "started": started,
           "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
           "argv": sys.argv[1:] if argv is None else list(argv), "recipe": args.recipe,
           "extras": list(args.extras_set), "shape": shape, "scale": SCALE, "half_above": HALF_ABOVE,
           "conf_min": 1, "seed": args.seed, "steps": args.steps, "qat_start": qat_start, "probe": args.probe,
           "settings": recipe_record(args, qat_start, peak, warmup, t_rec), "data_rule": data_rec,
           "manifests_sha256": {m: TT.sha256(m) for m in args.manifest}, "skip": skip,
           "shards_dirs": [str(Path(d)) for d in args.shards], "shards_verified": True, "pool": pool_summary,
           "code_sha256": ST.code_sha256(),
           "protocol_sha256": TT.sha256(PROTOCOL) if PROTOCOL.exists() else None,
           "textspot_protocol_sha256": TT.sha256(TEXTSPOT_PROTOCOL) if TEXTSPOT_PROTOCOL.exists() else None,
           "git": ST.git_state(), "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    log = open(out / "log.jsonl", "a", encoding="utf-8")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    t0 = time.time()
    agg, n_agg = {}, 0
    act_scale = None
    ema = EMA() if ema_on else None
    qema = None
    n_steps = min(args.steps, args.probe + 1) if args.probe else args.steps
    times = []
    for step in range(n_steps):
        if step == qat_start:
            if ema is not None:
                ema.copy_to(model.parameters())
                n_bn = reestimate_bn(model, it, shape, args.bn_batches)
                write({"step": step, "ema_swap": True, "ema_steps": ema.steps, "bn_layers": n_bn,
                       "bn_batches": args.bn_batches})
            with ST.window_size():
                act_scale = TT.calibrate(model, it, shape, args.cal_batches)
            model.eval()
            folded = TM.fold(model)
            model = TM.QTextDet(shape, folded, act_scale)
            opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())),
                                    lr=lr_at(step, args.steps, peak, warmup))
            if ema is not None:
                qema = EMA().step(model.parameters())
            write({"step": step, "qat": True, "act_scale": act_scale, "qat_average": qema is not None})
        ts = time.perf_counter()
        lr = lr_at(step, args.steps, peak, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, vhw, K, O, A1, A2, vm, km, Wt = next(it)  # noqa: E741
        td = time.perf_counter()
        masks = TM.layer_masks(shape, vhw.tolist(), WIN_H, WIN_W)
        xin = TT.to_input(x)
        y = model(xin, masks=masks)
        loss, parts = loss_of(y, K, O, A1, A2, vm, km, Wt)
        if teacher is not None:
            with torch.no_grad():
                t_masks = masks if t_shape == shape else TM.layer_masks(t_shape, vhw.tolist(), WIN_H, WIN_W)
                t_out = teacher(xin, masks=t_masks)
            kl, kparts = kd_loss(y, t_out, vm)
            loss = loss + kl
            parts.update(kparts)
        parts["loss"] = float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if qema is not None:
            qema.step(model.parameters())
        elif ema is not None and step + 1 >= args.ema_start:
            ema.step(model.parameters())
            if ema.steps == 1:
                write({"step": step + 1, "ema": "start", "decay": EMA_DECAY})
        if args.probe:
            times.append((td - ts, time.perf_counter() - ts))
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n_agg += 1
        if (step + 1) % args.log_every == 0 or step + 1 == n_steps:
            rec = {"step": step + 1, "lr": round(lr, 6), "sec": round(time.time() - t0, 1),
                   **{k: round(v / n_agg, 5) for k, v in agg.items()}}
            write(rec)
            print(json.dumps(rec), flush=True)
            agg, n_agg = {}, 0
    if args.probe:
        timed = times[1:]
        ms = np.array([t for _, t in timed]) * 1000
        wait = np.array([t for t, _ in timed]) * 1000
        probe = {"finished": ST.utc(), "name": name, "recipe": args.recipe, "extras": list(args.extras_set),
                 "steps_timed": len(timed), "warmup_ms": round(times[0][1] * 1000, 1),
                 "ms_per_step": round(float(ms.mean()), 2), "ms_per_step_median": round(float(np.median(ms)), 2),
                 "ms_per_step_p90": round(float(np.percentile(ms, 90)), 2),
                 "ms_waiting_for_data": round(float(wait.mean()), 2), "threads": args.threads,
                 "startup_s": round(t0 - t_proc, 1)}
        (out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n", encoding="utf-8")
        log.close()
        print("TEXTSPOT2_PROBE_DONE", json.dumps(probe), flush=True)
        return probe
    if qema is not None:
        qema.copy_to(model.parameters())
        write({"step": n_steps, "qat_average": "the run's model", "qat_average_steps": qema.steps})
    ck = {"kind": "textdet-qat", "shape": shape, "seed": args.seed, "steps": args.steps,
          "w": [w.detach().double() for w in model.w], "b": [b.detach().double() for b in model.b],
          "act_scale": act_scale, "run": run}
    torch.save(ck, out / "final.pt")
    vtd = EX.export_checkpoint(out / "final.pt")
    (out / "final.vtd").write_bytes(vtd)
    vtsp = TS.pack(vtd, HALF_ABOVE, 1)
    m = TS.Model(vtsp)
    if (m.half_above, m.conf_min, m.name) != (HALF_ABOVE, 1, shape["name"]):
        raise SystemExit("the VTSP blob does not read back")
    (out / "final.vtsp").write_bytes(vtsp)
    done = {"finished": ST.utc(), "name": name, "steps": args.steps, "seconds": round(time.time() - t0, 1),
            "wall_seconds": round(time.time() - t_proc, 1), "pt_sha256": TT.sha256(out / "final.pt"),
            "vtd_bytes": len(vtd), "vtd_sha256": hashlib.sha256(vtd).hexdigest(),
            "vtsp_bytes": len(vtsp), "vtsp_sha256": hashlib.sha256(vtsp).hexdigest()}
    (out / "done.json").write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
    log.close()
    print("TEXTSPOT2_TRAIN_DONE", json.dumps(done), flush=True)
    return done


if __name__ == "__main__":
    main()

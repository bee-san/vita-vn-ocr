#!/usr/bin/env python3
"""Train one textspot3 run of the full-frame text spotter (issue #108, docs/textspot3.md "Two families", "Recipes",
"Seeds", "Arms and stages"; eval/textspot3_protocol.json "families", "recipes", "half_above"); runs are named
<recipe>_<shape>_seed<n>.

  textspot3_train.py --recipe base|size|clean|long|kd --family T|M --shape S --seed N --manifest M ... \
                     --shards DIR ... --out RUN [--teacher RUN_DIR] [--extras size,clean|none] [--steps N] \
                     [--threads 5]
  textspot3_train.py ... --probe 200     speed probe: one warm-up step, then 200 steps of the run, timed
                                         (RUN/probe.json; no model is written)

textspot2's trainer (train/textspot2_train.py) and what it imports are used unchanged wherever the recipe does not
change them: the pool with its sha256 checks and skip lists (textspot_train.FramePool, textspot2_train.skip_lists,
skip_record, skip_per_manifest, pool_weights), the box halving (textspot_train.half_box through
textspot2_train.at_scale), the 4-px ignore rule, the mix, the line weights (textspot2_train.line_weight: a line's
7th field), the weighted loss (textspot2_train.loss_of), the distillation loss (textspot2_train.kd_loss), the EMA
(textspot2_train.EMA, reestimate_bn), the schedule (textspot2_train.lr_at), the optimiser groups and weight decay,
the gradient clip, #97's QAT calibration (textdet_train.calibrate), the BatchNorm fold, the checkpoint and, for
family T, textspot2's d2s network and export (train/textspot2_export.py: TextDetD2S, QTextDetD2S, to_stride4,
export_vtsp). None of their files changes.

**base** is textspot2's `data` + `ema`, exactly: textspot2's real labels (every lab_<source> and more_<source> shard
of data/manifests/textspot2_shards_v1.json, its skip lists applied) with textspot's synthetic shards
(data/manifests/textspot_shards_v1.json) in textspot's mix; 30,000 steps of 16 windows of 256 x 512 at the network's
scale; AdamW 4e-3, 500 warm-up steps, cosine to 1e-4, QAT from step 24,000 (80 %); an EMA (decay 0.999) of the
parameters from the weights after step 1,000; at the QAT start the average replaces the live parameters, the
BatchNorm statistics are re-estimated on 32 batches, then #97's calibration (32 batches), the fold and the QAT
network; during QAT an average (0.999) of the QAT parameters from the folded weights on is the run's model. Its
batches, losses and updates are textspot2's ema recipe's draw for draw (tests/test_textspot3_train.py runs both on
the same shards: the logged losses and LRs of every step, the QAT checkpoint and the int8 network are equal), with
one difference by design: the scale rule (below).

What the other recipes change:
- **size** (UI/menu emphasis): every positive line's kernel cells weigh clamp(sqrt(64 / n), 1, 4) times the line's
  own weight, where n is that line's kernel cell count in the window's target map at the map's stride (the cells
  that end up as its kernel cells: inside the window, valid, claimed by no other line; counted in targets(), this
  module's weighted copy of the targets); dice and every normalisation are #97's. A line with n >= 64 keeps its own
  weight; n = 16 doubles it; n <= 4 quadruples it.
- **clean** (cleaner meiki labels), at load time on the real half only (sources realplus, teacher, owner,
  devnative; the synthetic frames are never touched): a positive of label weight < 1 (a meiki box of score < 0.8
  that neither a reading-pipeline line nor a textspot winner box agrees with) becomes an ignore area; a vertical
  positive longer than 16 x its width (y1 - y0 > 16 (x1 - x0)) or narrower than 6 px (x1 - x0 < 6), both at the
  canvas's scale (the shard's own coordinates), becomes an ignore area. The canvases stay (so a clean run draws
  the windows a base run draws); what the rules change over the whole pool, per source and per rule, is in
  run.json ("clean").
- **long**: base at 60,000 steps, peak LR 2e-3, 1,000 warm-up steps, cosine to 1e-4, QAT from step 48,000.
- **kd** (stage 2): base + --extras + textspot2's distillation loss on the stride-4 maps: the label loss +
  BCE(K, sigmoid(K_t)) over the valid cells + 0.5 BCE(O, sigmoid(O_t)) + smooth L1((A1 - A1_t) / 8) +
  smooth L1((A2 - A2_t) / 8), the last three on the cells where sigmoid(K_t) > 0.5. The teacher is a textspot3 run's
  final.pt (T: kind textdet-qat with head kroa_x2, as QTextDetD2S; M: kind textspot3-qat, as QTextSpot3) in fake
  quantisation, frozen (eval mode, no gradient), on the student's window with its own layer masks for the window's
  valid region; the student's and the teacher's 16-channel heads both go through to_stride4. A teacher of the other
  family is taken only when the two scale rules halve the same canvases of the pool (counted, recorded; a canvas
  they disagree on refuses the run), so its maps come from the same window at the same scale.
- **--extras** (size, clean, or none) adds stage 2's "kept extras" to base or kd; kd needs it. A base run with
  extras is named after them: base + size is the size recipe (named size_...), base + clean the clean recipe, base
  + size + clean is named sizeclean_... (stage 2's "base + the kept extras on L"; a stage-1 base run keeps its name).

**Family T** (VTSP version 2, textspot2's d2s format): the registered T shapes (train/textspot3_shapes.py, which reads
eval/textspot3_protocol.json "shapes"): textdet_model.layers' grammar at stride 8 (stem, down, down8, separable
blocks) with up to 64 channels and the 2 x 2 sub-pixel head of 16 channels; train/textspot2_export.TextDetD2S, then
QTextDetD2S at the QAT start; the head through to_stride4 (stride-4 maps) against stride-4 targets; the scale
half_above 480 (a canvas wider than 480 px is halved: textspot_twin.halve's exact 2 x 2 mean, boxes start
floor(v / 2) and end ceil(v / 2)); export textspot2_export.export_vtsp(final.pt, 480, 1): final.vtd (the 16-output
VTDT network) and final.vtsp (VTSP version 2, read back by synth/textspot2_twin.Model).

**Family M** (VTSP version 3, the multi-scale family): train/textspot3_model.py (textspot3-runtime's: TextSpot3,
fold, QTextSpot3, layer_masks, to_stride4, export_vtsp; checkpoint kind textspot3-qat) and synth/textspot3_twin.py,
imported only for an M run or teacher, so family T needs neither; the scale: halved when the canvas is wider than
480 px or taller than 272 px. On every canvas of the pinned pool the two rules halve the same canvases (none is at
most 480 px wide and taller than 272 px).

Data: --manifest and --shards as in textspot2_train.py (a shard is read when its manifest's dir is the basename of a
--shards directory; every shard's npy and json sha256 is checked first; the skip lists of every data pin are
applied, and a textspot2 pin without them is refused). The protocol's directories: synth_frames, lab_<source> and
more_<source>, every shard of the manifests read, nothing else. --any-shards lifts this for probes and tests, and
--shape-json lets a probe or a test train a shape that is not registered (both recorded).

Outputs (the last step is the run's model; nothing is selected inside a run; RUN must not exist or be empty, so a
crashed run restarts from scratch in a fresh directory):
  RUN/run.json    started (UTC), the arguments, the recipe, the family and every setting, the name, the shape, the
                  seed, the manifests' sha256, the skip record, the shards read, the pool per source with its line
                  weights, clean's counts, the teacher (kd), the sha256 of every train/ and synth/ file imported, the
                  three protocols' sha256 and the checkout's commit (with any modified tracked file)
  RUN/log.jsonl   #97's records every 200 steps (step, lr, sec, k_bce, dice, o, a; kd_k, kd_o, kd_a, kd_cells for kd;
                  loss, the total optimised), the EMA's start and swap records and the QAT record
  RUN/final.pt    the quantisation-aware checkpoint (T: kind textdet-qat, M: textspot3-qat), the QAT average
  RUN/final.vtd   T: the int8 VTDT network (textspot2_export); M: whatever textspot3_model's export writes beside
                  the VTSP, if anything
  RUN/final.vtsp  T: VTSP version 2 (half_above 480, conf_min 1); M: VTSP version 3 (conf_min 1)
  RUN/done.json   finished (UTC), steps, wall time, the blobs' bytes and sha256, the VTSP version
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_model as TM  # noqa: E402
import textdet_train as TT  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_train as T2  # noqa: E402
import textspot2_twin as TS2  # noqa: E402
import textspot_train as ST  # noqa: E402
import textspot_twin as TS  # noqa: E402

WIN_H, WIN_W = T2.WIN_H, T2.WIN_W       # 256 x 512
BATCH, MIX, MIN_ACROSS, CLIP = T2.BATCH, T2.MIX, T2.MIN_ACROSS, T2.CLIP
REAL = T2.REAL                          # realplus, teacher, owner, devnative
SYNTH_DIR = T2.SYNTH_DIR
EMA_DECAY, EMA_START, BN_BATCHES = T2.EMA_DECAY, T2.EMA_START, T2.BN_BATCHES
MAP_STRIDE = TS2.D2S_MAP_STRIDE         # 4: both families' heads are read as stride-4 maps
NET_STRIDE = TS2.D2S_NET_STRIDE         # 8
N_HEAD = TS2.D2S_OUTPUTS                # 16
HALF_ABOVE_T = 480                      # T: a canvas wider than 480 px is halved (eval/textspot3_protocol.json)
HALF_ABOVE_M = (480, 272)               # M: wider than 480 px or taller than 272 px
T_MAX_CHANNELS = 64                     # the version-2 runtime's limit (docs/textspot3.md "Two families")
SIZE_N, SIZE_MIN, SIZE_MAX = 64, 1.0, 4.0
CLEAN_WEIGHT = 1.0                      # a positive of weight below this becomes an ignore area
CLEAN_RATIO, CLEAN_MIN_WIDTH = 16, 6    # a vertical positive longer than 16 x its width, or under 6 px wide
EXTRAS = ("size", "clean")
SET_EXTRAS = ("base", "kd")             # the recipes stage 2 combines with "the kept extras"
FAMILIES = ("T", "M")
T_KIND, M_KIND = "textdet-qat", "textspot3-qat"
RECIPES = {
    "base": {"extras": (), "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
             "what": "textspot2's data + ema: its real labels with textspot's synthetic half, an EMA of the weights"},
    "size": {"extras": ("size",), "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
             "what": "base + every positive line's kernel cells weighted by clamp(sqrt(64 / n), 1, 4)"},
    "clean": {"extras": ("clean",), "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
              "what": "base + agreement filtering and the vertical-strip rule on the real half, at load time"},
    "long": {"extras": (), "steps": 60000, "lr": 2e-3, "warmup": 1000,
             "what": "base at 60,000 steps, peak LR 2e-3, 1,000 warm-up steps, cosine to 1e-4"},
    "kd": {"extras": None, "steps": 30000, "lr": TT.LR, "warmup": TT.WARMUP,
           "what": "base + the kept extras + distillation from a teacher spotter on the stride-4 maps"},
}
PROTOCOL = ROOT / "eval" / "textspot3_protocol.json"
TEXTSPOT2_PROTOCOL = ROOT / "eval" / "textspot2_protocol.json"
TEXTSPOT_PROTOCOL = ROOT / "eval" / "textspot_protocol.json"
T_FIELDS = T2.SHAPE_FIELDS              # name, stride, c0, c1, c2, chans, dil, block, head


# ----------------------------------------------------------------------------- shapes and families

def registry():
    """The registered textspot3 shapes: train/textspot3_shapes.SHAPES once it says REGISTERED (the shapes of
    eval/textspot3_protocol.json, registered by an amendment); {} before that (its SHAPES are then the T_ROWS
    candidates, which no run may train) and while that module does not exist."""
    try:
        import textspot3_shapes as S3
    except ImportError:
        return {}
    if not getattr(S3, "REGISTERED", False):
        return {}
    return {k: dict(v) for k, v in S3.SHAPES.items()}


def family_of(shape):
    """'T' for a d2s trunk (head kroa_x2), 'M' for a multi-scale shape (family M or head ms_x2), else None."""
    fam = shape.get("family")
    if fam in FAMILIES:
        return fam
    if shape.get("head") == X.HEAD:
        return "T"
    if shape.get("head") == "ms_x2":
        return "M"
    return None


def t_shape(s):
    """A T shape as textdet_model's and textspot2_export's functions take it: exactly textspot2's shape fields
    (a d2s checkpoint of textspot2's form), checked: stride 8, head kroa_x2, separable blocks, at most 64 channels,
    the family's half_above (480) when the shape names one."""
    if s.get("half_above", HALF_ABOVE_T) != HALF_ABOVE_T:
        raise ValueError(f"{s.get('name')}: half_above {s.get('half_above')}, family T's is {HALF_ABOVE_T}")
    out = {k: s[k] for k in T_FIELDS}
    out["chans"], out["dil"] = list(out["chans"]), list(out["dil"])
    if out["stride"] != NET_STRIDE or out["head"] != X.HEAD or out["block"] != "sep":
        raise ValueError(f"{out['name']}: a T shape is a separable stride-8 trunk with head {X.HEAD!r}")
    if len(out["chans"]) != len(out["dil"]) or not out["chans"]:
        raise ValueError(f"{out['name']}: chans and dil must be equally long and not empty")
    widest = max(out["c0"], out["c1"], out["c2"] or 0, *out["chans"])
    if widest > T_MAX_CHANNELS:
        raise ValueError(f"{out['name']}: {widest} channels, the version-2 runtime's limit is {T_MAX_CHANNELS}")
    X.layers(out)
    return out


def resolve_shape(name, spec=None):
    """(shape, family, source): the registered shape of that name, or (probes and tests) the --shape-json spec."""
    if spec is not None:
        s = dict(spec)
        if s.get("name") != name:
            raise SystemExit(f"--shape-json names {s.get('name')!r}, not --shape {name!r}")
        src = "--shape-json (a probe or a test; not a registered shape)"
    else:
        reg = registry()
        if name not in reg:
            raise SystemExit(f"{name}: not a registered textspot3 shape ({sorted(reg) or 'none registered yet'}; "
                             "train/textspot3_shapes.py reads eval/textspot3_protocol.json \"shapes\")")
        s = reg[name]
        src = "train/textspot3_shapes.SHAPES (eval/textspot3_protocol.json \"shapes\")"
    fam = family_of(s)
    if fam is None:
        raise SystemExit(f"{name}: neither a T (head {X.HEAD}) nor an M shape")
    if fam == "T":
        try:
            s = t_shape(s)
        except (KeyError, ValueError) as e:
            raise SystemExit(f"{name}: {e}") from None
    return s, fam, src


def half_t(width, height):
    """T's scale: 2 when the canvas is wider than HALF_ABOVE_T (480) px."""
    return 2 if width > HALF_ABOVE_T else 1


def half_m(width, height):
    """M's scale: 2 when the canvas is wider than 480 px or taller than 272 px."""
    return 2 if width > HALF_ABOVE_M[0] or height > HALF_ABOVE_M[1] else 1


HALF = {"T": half_t, "M": half_m}


def scale_record(family):
    if family == "T":
        return {"rule": "half_above", "half_above": HALF_ABOVE_T,
                "what": "a canvas wider than 480 px is halved (textspot_twin.halve)"}
    return {"rule": "wider or taller", "half_above": HALF_ABOVE_M[0], "half_above_h": HALF_ABOVE_M[1],
            "what": "a canvas wider than 480 px or taller than 272 px is halved (textspot_twin.halve)"}


def m_modules():
    """textspot3-runtime's train/textspot3_model.py and synth/textspot3_twin.py (family M, VTSP version 3)."""
    try:
        import textspot3_model as M3
        import textspot3_twin as TW3
    except ImportError as e:
        raise SystemExit(f"family M needs train/textspot3_model.py and synth/textspot3_twin.py (textspot3-runtime's "
                         f"VTSP version 3, FORMAT_READY): {e}") from None
    return M3, TW3


class FamilyT:
    """Family T: textspot2's d2s network (textdet_model's trunk, textspot2_export's 16-channel head), VTSP version 2."""
    name, kind, vtsp_version = "T", T_KIND, TS2.VERSION_D2S

    def __init__(self, shape):
        self.shape = shape

    def model(self):
        return X.TextDetD2S(self.shape)

    def masks(self, vhw):
        return TM.layer_masks(self.shape, vhw, WIN_H, WIN_W)

    def maps(self, y):
        return X.to_stride4(y, WIN_H, WIN_W)

    def reestimate_bn(self, model, it, n):
        return T2.reestimate_bn(model, it, self.shape, n)

    def calibrate(self, model, it, n):
        with ST.window_size():
            return TT.calibrate(model, it, self.shape, n)

    def fold(self, model):
        return TM.fold(model)

    def qmodel(self, folded, act_scale):
        return X.QTextDetD2S(self.shape, folded, act_scale)

    def checkpoint(self, model, act_scale, seed, steps, run):
        return {"kind": T_KIND, "shape": self.shape, "seed": seed, "steps": steps,
                "w": [w.detach().double() for w in model.w], "b": [b.detach().double() for b in model.b],
                "act_scale": act_scale, "run": run}

    def export(self, pt):
        """-> (VTDT bytes, VTSP version 2 bytes), read back: version 2, half_above 480, conf_min 1, the shape."""
        vtd, vtsp = X.export_vtsp(pt, HALF_ABOVE_T, 1)
        m = TS2.Model(vtsp)
        if (m.version, m.half_above, m.conf_min, m.name) != (TS2.VERSION_D2S, HALF_ABOVE_T, 1, self.shape["name"]):
            raise SystemExit("the VTSP version-2 blob does not read back")
        return vtd, vtsp

    def record(self):
        return {"family": "T", "head": X.HEAD, "outputs": X.N_OUT, "net_stride": X.STRIDE, "map_stride": MAP_STRIDE,
                "network": "train/textspot2_export.TextDetD2S, then QTextDetD2S at the QAT start (head scales per "
                           "channel: K, O 1/16; A1, A2 1/2 px)",
                "maps": "textspot2_export.to_stride4 (channel 4 (2 dy + dx) + m: map m of sub-cell (dy, dx)) against "
                        "the stride-4 targets",
                "export": f"textspot2_export.export_vtsp(final.pt, {HALF_ABOVE_T}, 1): VTSP version 2"}


class FamilyM:
    """Family M: textspot3-runtime's multi-scale network (train/textspot3_model.py), VTSP version 3."""
    name, kind = "M", M_KIND

    def __init__(self, shape):
        self.M3, self.TW3 = m_modules()
        self.shape = shape
        self.vtsp_version = getattr(self.TW3, "VERSION_MS", 3)

    def model(self):
        return self.M3.TextSpot3(self.shape)

    def masks(self, vhw):
        return self.M3.layer_masks(self.shape, vhw, WIN_H, WIN_W)

    def maps(self, y):
        return self.M3.to_stride4(y, WIN_H, WIN_W)

    def reestimate_bn(self, model, it, n):
        return reestimate_bn(model, it, self.masks, n)

    def calibrate(self, model, it, n):
        return calibrate(model, it, self.masks, n)

    def fold(self, model):
        return self.M3.fold(model)

    def qmodel(self, folded, act_scale):
        return self.M3.QTextSpot3(self.shape, folded, act_scale)

    def checkpoint(self, model, act_scale, seed, steps, run):
        return {"kind": M_KIND, "shape": self.shape, "seed": seed, "steps": steps,
                "w": [w.detach().double() for w in model.w], "b": [b.detach().double() for b in model.b],
                "act_scale": act_scale, "run": run}

    def export(self, pt):
        out = self.M3.export_vtsp(pt, conf_min=1)
        vtd, vtsp = out if isinstance(out, tuple) else (None, out)
        m = self.TW3.Model(vtsp)
        if (m.version, m.conf_min, m.name) != (self.vtsp_version, 1, self.shape["name"]):
            raise SystemExit("the VTSP version-3 blob does not read back")
        return vtd, vtsp

    def record(self):
        return {"family": "M", "head": self.shape.get("head"), "outputs": N_HEAD, "net_stride": NET_STRIDE,
                "map_stride": MAP_STRIDE,
                "network": "train/textspot3_model.TextSpot3, then QTextSpot3 at the QAT start",
                "maps": "textspot3_model.to_stride4 against the stride-4 targets",
                "export": "textspot3_model.export_vtsp(final.pt, conf_min=1): VTSP version 3"}


def family(name, shape):
    return FamilyT(shape) if name == "T" else FamilyM(shape)


# ----------------------------------------------------------------------------- M's calibration and BatchNorm
# (textdet_train.calibrate and textspot2_train.reestimate_bn read textdet_model.layer_masks; these are the same code
# with the family's masks; family T calls #97's and textspot2's own functions)

def calibrate(model, it, masks_of, n=TT.CAL_BATCHES):
    """textdet_train.calibrate with masks_of(valid) for the layer masks: per tap, the mean over n batches of the
    99.99th percentile of |output| over its valid positions, / 127 (at least 1e-4 / 127)."""
    model.eval()
    acc = None
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _ in range(n):
            x, vhw = next(it)[:2]
            masks = masks_of(vhw.tolist())
            taps = []
            model(TT.to_input(x), masks=masks, taps=taps)
            if len(taps) != len(masks):
                raise ValueError(f"{len(taps)} taps for {len(masks)} masks")
            vals = []
            for t, m in zip(taps, masks):
                a = t.abs()[m.expand_as(t) > 0]
                if a.numel() > 200_000:
                    a = a[torch.randint(0, a.numel(), (200_000,), generator=gen)]
                vals.append(float(torch.quantile(a, 0.9999)) if a.numel() else 1.0)
            acc = vals if acc is None else [p + q for p, q in zip(acc, vals)]
    model.train()
    return [max(v / n, 1e-4) / 127 for v in acc]


def reestimate_bn(model, it, masks_of, n):
    """textspot2_train.reestimate_bn with masks_of(valid) for the layer masks."""
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
            model(TT.to_input(x), masks=masks_of(vhw.tolist()))
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    model.train(was_training)
    return len(bns)


# ----------------------------------------------------------------------------- clean (at load time)

def clean_rule(ln):
    """The rules a shard line [x0, y0, x1, y1, vertical, class, weight] breaks (clean): 'weight' (label weight < 1),
    'vertical_long' (vertical, y1 - y0 > 16 (x1 - x0)), 'vertical_narrow' (vertical, x1 - x0 < 6); () when none."""
    hit = []
    if T2.line_weight(ln) < CLEAN_WEIGHT:
        hit.append("weight")
    if int(ln[4]):
        width, length = int(ln[2]) - int(ln[0]), int(ln[3]) - int(ln[1])
        if length > CLEAN_RATIO * width:
            hit.append("vertical_long")
        if width < CLEAN_MIN_WIDTH:
            hit.append("vertical_narrow")
    return tuple(hit)


def clean_sample(smp):
    """(lines kept, ignore areas with the removed positives' boxes appended, [(removed line, the rules it broke)])."""
    kept, removed = [], []
    for ln in smp["lines"]:
        hit = clean_rule(ln)
        if hit:
            removed.append((ln, hit))
        else:
            kept.append(ln)
    return kept, [*smp["ignore"], *([int(v) for v in ln[:4]] for ln, _ in removed)], removed


def clean_pool(pool):
    """Applies clean to every real canvas of the pool in place; -> the counts per source (canvases, canvases changed,
    lines before and kept, lines removed, the removed vertical lines, every rule's hits; a line can break more than
    one rule) and their totals. The synthetic frames are never touched."""
    per = {}
    for src in REAL:
        c = {"canvases": 0, "canvases_changed": 0, "lines": 0, "vertical": 0, "removed": 0, "removed_vertical": 0,
             "kept": 0, "hits": {"weight": 0, "vertical_long": 0, "vertical_narrow": 0},
             "removed_by_vertical_rules_only": 0}
        for _, smp in pool.items.get(src, []):
            c["canvases"] += 1
            c["lines"] += len(smp["lines"])
            c["vertical"] += sum(int(ln[4]) for ln in smp["lines"])
            kept, ignore, removed = clean_sample(smp)
            if removed:
                c["canvases_changed"] += 1
                for ln, hit in removed:
                    for h in hit:
                        c["hits"][h] += 1
                    c["removed_vertical"] += int(ln[4])
                    c["removed_by_vertical_rules_only"] += "weight" not in hit
                smp["lines"], smp["ignore"] = kept, ignore
            c["removed"] += len(removed)
            c["kept"] += len(kept)
        per[src] = c
    tot = {k: sum(c[k] for c in per.values()) for k in ("canvases", "canvases_changed", "lines", "vertical", "removed",
                                                         "removed_vertical", "kept",
                                                         "removed_by_vertical_rules_only")}
    tot["hits"] = {h: sum(c["hits"][h] for c in per.values()) for h in ("weight", "vertical_long", "vertical_narrow")}
    return {"by_source": per, "total": tot,
            "rules": {"weight": f"label weight < {CLEAN_WEIGHT:g} (index 6)",
                      "vertical_long": f"vertical and y1 - y0 > {CLEAN_RATIO} (x1 - x0)",
                      "vertical_narrow": f"vertical and x1 - x0 < {CLEAN_MIN_WIDTH} px"},
            "how": "at load time, on every real canvas of the pool (src not 'synth...'), in the shard's coordinates "
                   "(the canvas's scale): a positive that breaks a rule becomes an ignore area (its box appended to "
                   "the canvas's ignore areas); the canvases stay"}


class Pool(ST.FramePool):
    """textspot_train.FramePool (sha256 checks, skip lists, #97's sources and mix), then clean in place when on."""

    def __init__(self, manifests, dirs, verify=False, skip=(frozenset(), frozenset()), clean=False):
        super().__init__(manifests, dirs, verify=verify, skip=skip)
        self.cleaned = clean_pool(self) if clean else None


# ----------------------------------------------------------------------------- windows, targets, batches

def window(rng, canvas, half):
    """textspot_train.window with the family's scale rule half(W, H) -> f: the window of a canvas at the network's
    scale, at a uniformly random offset (x drawn first). With half = 2 for a canvas wider than 600 px it is
    textspot_train.window(rng, canvas, 'h2') exactly."""
    H, W = canvas.shape
    f = half(W, H)
    Hs, Ws = -(-H // f), -(-W // f)
    h, w = min(WIN_H, Hs), min(WIN_W, Ws)
    ox, oy = int(rng.integers(0, Ws - w + 1)), int(rng.integers(0, Hs - h + 1))
    if f == 2:
        img = TS.halve(canvas[2 * oy:2 * (oy + h), 2 * ox:2 * (ox + w)])
    else:
        img = np.asarray(canvas[oy:oy + h, ox:ox + w])
    return img, ox, oy, w, h, f


def size_factor(n):
    """clamp(sqrt(64 / n), 1, 4) per kernel-cell count n (n >= 1), float64."""
    n = np.asarray(n, np.float64)
    return np.clip(np.sqrt(SIZE_N / np.maximum(n, 1.0)), SIZE_MIN, SIZE_MAX)


def targets(lines, ignore, wx, wy, w, h, s, size=False):
    """textspot2_train.targets (the round-1 kernel rule and the weight map) on the WIN_H x WIN_W window, line for line,
    plus size: (K, O, A1, A2, valid, kernel, W). lines: [[x0, y0, x1, y1, vertical, weight]] (at_scale's positives).
    Without size it is textspot2_train.targets exactly. With size, a kernel cell of line l weighs
    float32(w_l x clamp(sqrt(64 / n_l), 1, 4)), n_l = the number of kernel cells line l owns in this window's map
    (inside the window, valid, claimed by no other line)."""
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
    if size and kern.any():
        own = owner[kern]
        n_cells = np.bincount(own, minlength=len(lines))
        Wt[kern] = (Wt[kern].astype(np.float64) * size_factor(n_cells)[own]).astype(np.float32)
    return K, O, A1, A2, valid_k, kern, Wt


def make_batch(pool, rng, s, half, size=False):
    """textspot2_train.make_batch (the same draws in the same order: source, canvas, window) with the family's scale
    rule and size: (x, valid (h, w), K, O, A1, A2, valid mask, kernel mask, W)."""
    x = np.full((BATCH, 1, WIN_H, WIN_W), 128, np.uint8)
    valid_hw = []
    tg = [np.zeros((BATCH, WIN_H // s, WIN_W // s), np.float32) for _ in range(5)]
    vm = np.zeros((BATCH, WIN_H // s, WIN_W // s), bool)
    km = np.zeros_like(vm)
    for b in range(BATCH):
        src = pool.keys[int(rng.choice(len(pool.keys), p=pool.p))]
        lst = pool.items[src]
        d, smp = lst[int(rng.integers(len(lst)))]
        img, ox, oy, w, h, f = window(rng, pool.canvas(d, smp), half)
        x[b, 0, :h, :w] = img
        valid_hw.append((h, w))
        lines, ignore = T2.at_scale(smp["lines"], smp["ignore"], f)
        K, O, A1, A2, v, k, Wt = targets(lines, ignore, ox, oy, w, h, s, size)  # noqa: E741
        for t, a in zip(tg, (K, O, A1, A2, Wt)):
            t[b] = a
        vm[b], km[b] = v, k
    K, O, A1, A2, Wt = (torch.from_numpy(t) for t in tg)  # noqa: E741
    return torch.from_numpy(x), torch.tensor(valid_hw), K, O, A1, A2, torch.from_numpy(vm), torch.from_numpy(km), Wt


class FrameBatches(torch.utils.data.IterableDataset):
    def __init__(self, manifests, dirs, seed, family_name, size=False, clean=False,
                 skip=(frozenset(), frozenset())):
        super().__init__()
        self.manifests, self.dirs = list(manifests), list(dirs)
        self.seed, self.family_name, self.size, self.clean = seed, family_name, size, clean
        self.skip = tuple(frozenset(x) for x in skip)

    def __iter__(self):
        pool = Pool(self.manifests, self.dirs, skip=self.skip, clean=self.clean)
        rng = np.random.default_rng(self.seed)
        half = HALF[self.family_name]
        while True:
            yield make_batch(pool, rng, MAP_STRIDE, half, self.size)


# ----------------------------------------------------------------------------- the teacher (kd)

def scale_disagreements(pool, a, b):
    """Canvases of the pool (every source) that scale rules a and b halve differently: {source: count}."""
    out = {}
    for src, items in pool.items.items():
        n = sum(1 for _, smp in items if HALF[a](smp["w"], smp["h"]) != HALF[b](smp["w"], smp["h"]))
        if n:
            out[src] = n
    return out


def load_teacher(run_dir):
    """A textspot3 teacher run: its final.pt as the quantisation-aware network in fake quantisation, eval mode, no
    gradient (T: kind textdet-qat with head kroa_x2, as QTextDetD2S; M: kind textspot3-qat, as QTextSpot3); its
    run.json must be a textspot3 run of that family with the family's scale. -> (model, family adapter, record)."""
    d = Path(run_dir)
    pt, info_path = d / "final.pt", d / "run.json"
    if not pt.exists() or not info_path.exists():
        raise SystemExit(f"{d}: a teacher run needs final.pt and run.json")
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("kind") != "textspot3-run":
        raise SystemExit(f"{info_path}: not a textspot3 run (kind {info.get('kind')!r})")
    if ck.get("kind") == T_KIND and X.is_d2s(ck.get("shape")):
        fam = FamilyT(t_shape(ck["shape"]))
    elif ck.get("kind") == M_KIND:
        fam = FamilyM(ck["shape"])
    else:
        raise SystemExit(f"{pt}: neither a d2s (kind {T_KIND}, head {X.HEAD}) nor an M ({M_KIND}) checkpoint")
    if info.get("family") != fam.name or info.get("scale") != scale_record(fam.name):
        raise SystemExit(f"{info_path}: family {info.get('family')}, scale {info.get('scale')}: not a family "
                         f"{fam.name} run at its scale")
    model = fam.qmodel(list(zip(ck["w"], ck["b"])), ck["act_scale"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vtsp = d / "final.vtsp"
    rec = {"dir": str(d), "name": info.get("name"), "kind": info.get("kind"), "recipe": info.get("recipe"),
           "family": fam.name, "shape": ck["shape"]["name"], "scale": info.get("scale"), "steps": ck.get("steps"),
           "seed": ck.get("seed"), "pt_sha256": TT.sha256(pt),
           "vtsp_sha256": TT.sha256(vtsp) if vtsp.exists() else None,
           "how": "final.pt as the family's quantisation-aware network (fake quantisation on), eval mode, no "
                  "gradient, on the student's window with its own layer masks; its 16-channel head through "
                  "to_stride4"}
    return model, fam, rec


# ----------------------------------------------------------------------------- arguments and names

def parse_extras(text):
    """'size,clean' | 'size' | 'clean' | 'none' -> a tuple in EXTRAS order; None stays None."""
    if text is None:
        return None
    items = [x.strip() for x in text.split(",") if x.strip()]
    if items == ["none"]:
        return ()
    if not items or any(x not in EXTRAS for x in items) or len(set(items)) != len(items):
        raise ValueError(f"--extras {text!r}: a comma list of {', '.join(EXTRAS)}, or none")
    return tuple(x for x in EXTRAS if x in items)


def name_recipe(recipe, extras):
    """The recipe part of a run's name: kd and long by their recipe; base, size and clean by their extras (base with
    none, size, clean, or sizeclean for base + both)."""
    if recipe in ("kd", "long"):
        return recipe
    return "".join(x for x in EXTRAS if x in extras) if extras else "base"


def run_name(recipe, shape, seed, extras=None):
    ex = RECIPES[recipe]["extras"] if extras is None else extras
    return f"{name_recipe(recipe, ex)}_{shape}_seed{seed}"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, choices=list(RECIPES))
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", help="kd: the teacher run's directory (its final.pt and run.json)")
    ap.add_argument("--extras", help="base and kd: the kept extras, a comma list of size, clean, or none (kd needs it)")
    ap.add_argument("--steps", type=int, help="default: the recipe's (30,000; long 60,000)")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="speed probe: one warm-up step, then N steps of the run's schedule, timed; RUN/probe.json")
    ap.add_argument("--cal-batches", type=int, default=TT.CAL_BATCHES, help="QAT calibration batches (#97: 32)")
    ap.add_argument("--bn-batches", type=int, default=BN_BATCHES, help="BatchNorm re-estimation batches (32)")
    ap.add_argument("--ema-start", type=int, default=EMA_START,
                    help="the average starts from the weights after this many steps (1,000)")
    ap.add_argument("--workers", type=int, default=1, help="loader processes (#97: 1)")
    ap.add_argument("--any-shards", action="store_true",
                    help="probes and tests: read any shard directories (not the protocol's data rule)")
    ap.add_argument("--shape-json", help="probes and tests: the shape as a JSON object (not a registered shape)")
    ap.add_argument("--allow-missing-sources", action="store_true",
                    help="tests: train on whichever of the mix's sources the shards hold")
    args = ap.parse_args(argv)
    r = RECIPES[args.recipe]
    try:
        extras = parse_extras(args.extras)
    except ValueError as e:
        ap.error(str(e))
    if extras is not None and args.recipe not in SET_EXTRAS:
        ap.error(f"--extras is for {', '.join(SET_EXTRAS)}; {args.recipe}'s extras are the protocol's")
    if args.recipe == "kd" and extras is None:
        ap.error("kd: --extras is required (the kept extras: none, size, clean or size,clean)")
    if (args.recipe == "kd") != bool(args.teacher):
        ap.error("--teacher is required for kd and only for kd")
    args.extras_set = r["extras"] if extras is None else extras
    args.steps = r["steps"] if args.steps is None else args.steps
    if (args.steps < 1 or args.probe < 0 or args.cal_batches < 1 or args.bn_batches < 1 or args.ema_start < 1
            or args.workers < 0 or args.log_every < 1 or args.threads < 1):
        ap.error("--steps, --cal-batches, --bn-batches, --ema-start, --log-every, --threads >= 1; --probe, "
                 "--workers >= 0")
    if args.ema_start > int(TT.QAT_FRAC * args.steps):
        ap.error(f"--ema-start {args.ema_start} is after the QAT start {int(TT.QAT_FRAC * args.steps)}")
    if args.shape_json is not None:
        try:
            args.shape_spec = json.loads(args.shape_json)
        except ValueError as e:
            ap.error(f"--shape-json: {e}")
        if not isinstance(args.shape_spec, dict):
            ap.error("--shape-json: a JSON object")
    else:
        args.shape_spec = None
    args.name = run_name(args.recipe, args.shape, args.seed, args.extras_set)
    return args


def data_check(manifests, dirs, pool):
    """The protocol's directories (textspot2's rule for every recipe but lab): synth_frames, lab_<source> and
    more_<source>, every shard of the manifests read; nothing else. -> the record; SystemExit when it does not hold."""
    names = [Path(d).name for d in dirs]
    lab = [f"lab_{s}" for s in REAL]
    more = [f"more_{s}" for s in REAL]
    allowed = {SYNTH_DIR, *lab, *more}
    bad = sorted(set(names) - allowed)
    if bad:
        raise SystemExit(f"--shards {bad} are not the protocol's directories ({sorted(allowed)}); --any-shards is "
                         "for probes and tests")
    if SYNTH_DIR not in names or not set(lab) & set(names) or not set(more) & set(names):
        raise SystemExit(f"the protocol's run reads {SYNTH_DIR}, lab_<source> and more_<source> directories")
    unread = {}
    for key, n in pool.skipped.items():
        d = key.rpartition(":")[2]
        unread[d] = unread.get(d, 0) + n
    if unread:
        raise SystemExit(f"the manifests' shards in {sorted(unread)} are not read (pass their directories)")
    return {"rule": "synth_frames + lab_<source> + more_<source>, every manifest shard read (textspot2's data rule "
                    "for its data recipe); never #97's real directories",
            "dirs": names, "unread": unread, "manifests": [str(m) for m in manifests]}


def recipe_record(args, fam, qat_start, peak, warmup, teacher, clean_counts):
    ex = args.extras_set
    return {
        "recipe": args.recipe, "name_recipe": name_recipe(args.recipe, ex), "what": RECIPES[args.recipe]["what"],
        "extras": list(ex), "family": fam.name, "scale": scale_record(fam.name), "batch": BATCH,
        "window": [WIN_H, WIN_W], "mix": MIX, "steps": args.steps,
        "lr": {"peak": peak, "min": TT.LR_MIN, "warmup": warmup,
               "form": "peak (step + 1) / warmup during the warm-up, then cosine from peak to min over the "
                       "remaining steps (textdet_train.lr_at's form; textspot2_train.lr_at)"},
        "wd": TT.WD, "clip": CLIP, "qat_frac": TT.QAT_FRAC, "qat_start": qat_start, "cal_batches": args.cal_batches,
        "kernel_rule": "round 1 (textdet_train.targets strict=False), copied with the weight map (textspot2's) and "
                       "size", "map_stride": MAP_STRIDE, "min_across": MIN_ACROSS,
        "halve": "textspot_twin.halve when the family's scale rule says so; boxes start floor(v/2), end ceil(v/2)",
        "windows": "uniformly random offset; a side that fits is whole; no region crops",
        "positives": "every line of every class; under min_across px across at the network's scale: ignore",
        "weights": {"field": 6, "default": 1.0,
                    "rule": "a line's weight multiplies the K BCE, O BCE and A smooth-L1 terms of its kernel cells; "
                            "every other valid cell weighs 1; dice and the normalisations are #97's"},
        "ema": {"decay": EMA_DECAY, "start_after_step": args.ema_start, "bn_batches": args.bn_batches,
                "qat_decay": EMA_DECAY,
                "rule": "the average starts from the weights after start_after_step steps and is updated after every "
                        "later step; at the QAT start it replaces the live parameters, the BatchNorm statistics are "
                        "re-estimated (reset, momentum None, bn_batches training-mode forwards without gradient), "
                        "then #97's calibration, fold and the QAT network; during QAT an average (qat_decay) of the "
                        "QAT parameters from the folded weights on is the run's model"},
        "size": {"n": SIZE_N, "min": SIZE_MIN, "max": SIZE_MAX,
                 "rule": "a kernel cell of line l weighs w_l x clamp(sqrt(64 / n_l), 1, 4) (float32), n_l = the "
                         "kernel cells line l owns in the window's stride-4 map (valid, claimed by no other line); "
                         "dice and the normalisations unchanged"} if "size" in ex else None,
        "clean": clean_counts,
        "kd": {"teacher": teacher,
               "loss": "label loss + BCE(K, sigmoid(K_t)) over the valid cells (/ their count) + 0.5 BCE(O, "
                       "sigmoid(O_t)) + smooth L1((A1 - A1_t) / 8) + smooth L1((A2 - A2_t) / 8), the last three "
                       "averaged over the cells where sigmoid(K_t) > 0.5 (textspot2_train.kd_loss) on both networks' "
                       "stride-4 maps; no line weights"} if args.recipe == "kd" else None,
        "network": fam.record(), "loader_workers": args.workers,
    }


# ----------------------------------------------------------------------------- the run

def main(argv=None):
    args = parse(argv)
    t_proc = time.time()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: a run starts from scratch in a fresh directory")
    started = ST.utc()
    r = RECIPES[args.recipe]
    shape, fam_name, shape_src = resolve_shape(args.shape, args.shape_spec)
    if fam_name != args.family:
        raise SystemExit(f"--family {args.family}: {args.shape} is a family {fam_name} shape")
    fam = family(fam_name, shape)
    size_on, clean_on = "size" in args.extras_set, "clean" in args.extras_set
    teacher = t_fam = t_rec = None
    if args.recipe == "kd":
        teacher, t_fam, t_rec = load_teacher(args.teacher)
    skip_shots, skip_vns, lists = T2.skip_lists(args.manifest)
    pool = Pool(args.manifest, args.shards, verify=True, skip=(skip_shots, skip_vns), clean=clean_on)
    missing = [k for k in MIX if not pool.items[k]]
    if missing and not args.allow_missing_sources:
        raise SystemExit(f"no samples of {missing} (the mix needs all of {list(MIX)})")
    data_rec = ("any shard directories (--any-shards: a probe or a test)" if args.any_shards
                else data_check(args.manifest, args.shards, pool))
    if t_rec is not None:
        t_rec["scale_disagreements"] = scale_disagreements(pool, fam.name, t_fam.name)
        if t_rec["scale_disagreements"]:
            raise SystemExit(f"kd: the teacher's scale ({t_fam.name}) and the student's ({fam.name}) halve "
                             f"{t_rec['scale_disagreements']} canvases differently")
    pool_summary = pool.summary()
    pool_summary["weights"] = T2.pool_weights(pool)
    skip = ST.skip_record(lists, pool.dropped)
    skip["per_manifest"] = T2.skip_per_manifest(args.manifest, pool.read)
    skip["pin_differs_note"] = ("pin_expects sums the counts every list-carrying manifest records (textspot's, of "
                                "#97's real half); per_manifest gives what each manifest's own shards lost")
    clean_counts = pool.cleaned
    del pool
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    peak, warmup = r["lr"], r["warmup"]
    model = fam.model()
    opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=peak)
    loader = torch.utils.data.DataLoader(FrameBatches(args.manifest, args.shards, args.seed, fam.name, size_on,
                                                      clean_on, (skip_shots, skip_vns)),
                                         batch_size=None, num_workers=args.workers,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None)
    it = iter(loader)
    qat_start = int(TT.QAT_FRAC * args.steps)
    run = {"kind": "textspot3-run", "name": args.name, "started": started,
           "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
           "argv": sys.argv[1:] if argv is None else list(argv), "recipe": args.recipe,
           "extras": list(args.extras_set), "family": fam.name, "shape": shape, "shape_source": shape_src,
           "scale": scale_record(fam.name), "conf_min": 1, "seed": args.seed, "steps": args.steps,
           "qat_start": qat_start, "probe": args.probe,
           "settings": recipe_record(args, fam, qat_start, peak, warmup, t_rec, clean_counts),
           "data_rule": data_rec, "manifests_sha256": {m: TT.sha256(m) for m in args.manifest}, "skip": skip,
           "shards_dirs": [str(Path(d)) for d in args.shards], "shards_verified": True, "pool": pool_summary,
           "code_sha256": ST.code_sha256(),
           "protocol_sha256": TT.sha256(PROTOCOL) if PROTOCOL.exists() else None,
           "textspot2_protocol_sha256": TT.sha256(TEXTSPOT2_PROTOCOL) if TEXTSPOT2_PROTOCOL.exists() else None,
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
    ema = T2.EMA()
    qema = None
    n_steps = min(args.steps, args.probe + 1) if args.probe else args.steps
    times = []
    for step in range(n_steps):
        if step == qat_start:
            ema.copy_to(model.parameters())
            n_bn = fam.reestimate_bn(model, it, args.bn_batches)
            write({"step": step, "ema_swap": True, "ema_steps": ema.steps, "bn_layers": n_bn,
                   "bn_batches": args.bn_batches})
            act_scale = fam.calibrate(model, it, args.cal_batches)
            model.eval()
            folded = fam.fold(model)
            model = fam.qmodel(folded, act_scale)
            opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())),
                                    lr=T2.lr_at(step, args.steps, peak, warmup))
            qema = T2.EMA().step(model.parameters())
            write({"step": step, "qat": True, "act_scale": act_scale, "qat_average": True})
        ts = time.perf_counter()
        lr = T2.lr_at(step, args.steps, peak, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, vhw, K, O, A1, A2, vm, km, Wt = next(it)  # noqa: E741
        td = time.perf_counter()
        masks = fam.masks(vhw.tolist())
        xin = TT.to_input(x)
        y = fam.maps(model(xin, masks=masks))
        loss, parts = T2.loss_of(y, K, O, A1, A2, vm, km, Wt)
        if teacher is not None:
            with torch.no_grad():
                t_out = t_fam.maps(teacher(xin, masks=t_fam.masks(vhw.tolist())))
            kl, kparts = T2.kd_loss(y, t_out, vm)
            loss = loss + kl
            parts.update(kparts)
        parts["loss"] = float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if qema is not None:
            qema.step(model.parameters())
        elif step + 1 >= args.ema_start:
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
        probe = {"finished": ST.utc(), "name": args.name, "recipe": args.recipe, "family": fam.name,
                 "shape": shape["name"], "extras": list(args.extras_set), "steps_timed": len(timed),
                 "warmup_ms": round(times[0][1] * 1000, 1), "ms_per_step": round(float(ms.mean()), 2),
                 "ms_per_step_median": round(float(np.median(ms)), 2),
                 "ms_per_step_p90": round(float(np.percentile(ms, 90)), 2),
                 "ms_waiting_for_data": round(float(wait.mean()), 2), "threads": args.threads,
                 "startup_s": round(t0 - t_proc, 1)}
        (out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n", encoding="utf-8")
        log.close()
        print("TEXTSPOT3_PROBE_DONE", json.dumps(probe), flush=True)
        return probe
    if qema is not None:
        qema.copy_to(model.parameters())
        write({"step": n_steps, "qat_average": "the run's model", "qat_average_steps": qema.steps})
    torch.save(fam.checkpoint(model, act_scale, args.seed, args.steps, run), out / "final.pt")
    vtd, vtsp = fam.export(out / "final.pt")
    if vtd is not None:
        (out / "final.vtd").write_bytes(vtd)
    (out / "final.vtsp").write_bytes(vtsp)
    done = {"finished": ST.utc(), "name": args.name, "family": fam.name, "steps": args.steps,
            "seconds": round(time.time() - t0, 1), "wall_seconds": round(time.time() - t_proc, 1),
            "pt_sha256": TT.sha256(out / "final.pt"),
            "vtd_bytes": len(vtd) if vtd is not None else None,
            "vtd_sha256": hashlib.sha256(vtd).hexdigest() if vtd is not None else None,
            "vtsp_bytes": len(vtsp), "vtsp_sha256": hashlib.sha256(vtsp).hexdigest(),
            "vtsp_version": fam.vtsp_version}
    (out / "done.json").write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
    log.close()
    print("TEXTSPOT3_TRAIN_DONE", json.dumps(done), flush=True)
    return done


if __name__ == "__main__":
    main()

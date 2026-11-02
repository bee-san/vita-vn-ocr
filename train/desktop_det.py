#!/usr/bin/env python3
"""Train the desktop line detector (issue #169, docs/desktop.md): a float32 fully convolutional network for ONNX
Runtime on a PC, read by tools/vocr_desktop.py, trained at many text sizes from the start so that any input size the
app picks reads well (MeikiOCR's author found that a fixed-size detector made to take dynamic sizes loses accuracy
when it was never trained at them).

  desktop_det.py --shape d48 --seed N --out RUN --hook-manifest M --hook-shards DIR [--hook-splits train]
                 --manifest M ... --shards DIR ... [--steps 40000] [--lr 3e-3] [--mix hook=0.45,synth=0.2,real=0.35]
                 [--scale 0.42,1.9] [--threads 4] [--workers 1] [--probe N]
  desktop_det.py export --run RUN --out DIR [--name NAME] [--size fit:960] [--pad pct|frame]
      a run's final.pt as NAME.onnx (the ONNX file tools/vocr_desktop.Detector reads) and NAME.vdsk (the same
      weights for the C path, desktop/vocr_desk.c)

The network (SHAPES): a 3x3 stem at stride 2, then four stages at strides 4, 8, 16 and 32, each a separable
down-sampling block (3x3 depthwise at stride 2, BatchNorm, ReLU, 1x1, BatchNorm, ReLU) and n separable blocks
(3x3 depthwise with a dilation, 1x1, a residual add, ReLU); a top-down feature pyramid (1x1 laterals to F channels,
nearest 2x up-sampling and addition) back to stride 4; one separable smoothing block; a 1x1 head with 4 + T outputs
at stride 4: K and O logits, A1 and A2 (4 x the head's value, in input px) and T = 4 tag logits (textspot6's
classes after dialogue). The input is the grey frame as uint8 [N, 1, H, W] (H, W multiples of 32), normalised in
the graph as (g - 128) / 128.

Data: textspot6's sources (train/textspot6_train.py: the hooked page shards, textspot's synthetic frames, textspot2's
MeikiOCR-labelled real screenshots, the real half's skip lists), drawn by --mix; textspot6's targets with textspot8's
edge-kernel weights (train/textspot8_train.targets: the end cells of a line x 4), its tag loss and textspot2's loss.
The windows are textspot6's 256 x 512 with one change, the scale: every window's canvas (a PSP hooked page first
2x bilinear with p 0.5, as Adrenaline shows it; a desktop-size page of an --extra source named desk_* first to a
960 px long side, the scale the input policies read at) is resized by a factor drawn log-uniform from --scale before
the window is cut (INTER_AREA down, INTER_LINEAR up), so the network sees text from about 6 to 60 px tall; then
textspot6's photometric augmentation (with textspot8's blends), textspot8's icons (p 0.35) and tight cuts (p 0.15).
A window smaller than 256 x 512 is padded by edge replication, outside the loss.

Schedule: AdamW (textdet_train.param_groups' weight decay), linear warm-up, cosine to 1e-4, gradient clip 5; an EMA
(0.999) of the parameters from step 1,000, which is the run's model, its BatchNorm statistics re-estimated on 32
batches; BatchNorm folded into the convolutions; exported to ONNX (opset 17, dynamic height and width) and checked
against the torch network.

Outputs (RUN must not exist or be empty): run.json, log.jsonl, final.pt (state dict, shape), final.onnx, done.json.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

KIND = "desktop-detector"
TAGS = 4
SHAPES = {
    "d32": {"name": "d32", "c": [16, 24, 48, 64, 96], "n": [1, 2, 2, 1], "F": 32},
    "d48": {"name": "d48", "c": [16, 32, 64, 96, 128], "n": [1, 3, 3, 2], "F": 48},
    "d64": {"name": "d64", "c": [16, 32, 80, 128, 160], "n": [1, 4, 3, 2], "F": 64},
}
DILATIONS = {4: [1], 8: [1, 2, 4, 1], 16: [1, 2, 1], 32: [1, 2]}
DECODE = {"min_cells": 2, "conf_min_logit": 0.0, "pad_pct": 0.18, "pad_min": 2.0, "min_across": 4.0, "pad_frame": 0,
          "pad_frame_min": 2}
# the pad rules (tools/vocr_desktop.py): pct = textspot8's (18% of the size across, at least 2 network px);
# frame = MeikiOCR's reference pipeline's (the ink box grown by max(2, round(4 H / 544)) frame px)
PAD_RULES = {"pct": {"pad_pct": 0.18, "pad_min": 2.0, "pad_frame": 0},
             "frame": {"pad_pct": 0.0, "pad_min": 0.0, "pad_frame": 4}}
VDSK_HEADER = 144
DEFAULT_SIZE = "fit:960"
ATTRIBUTION = ("VOCR desktop line detector, bee-san/vita-vn-ocr (GPL-3.0-or-later). Trained on textspot's synthetic "
               "frames, MeikiOCR's detector boxes on VNDB and owner screenshots, and the owner's private hooked PS "
               "Vita / PSP captures (training input only): for non-commercial use only. See DESKTOP-ATTRIBUTION.md.")


# ----------------------------------------------------------------------------- the network

class Sep(nn.Module):
    """A separable block: 3x3 depthwise (stride, dilation), BN, ReLU, 1x1, BN, (+ x when shapes allow), ReLU."""

    def __init__(self, cin, cout, stride=1, dil=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, padding=dil, dilation=dil, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = F.relu(self.bn1(self.dw(x)))
        y = self.bn2(self.pw(y))
        return F.relu(y + x if self.res else y)


class Lateral(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class DeskNet(nn.Module):
    def __init__(self, shape, tags=TAGS):
        super().__init__()
        c0, c1, c2, c3, c4 = shape["c"]
        n4, n8, n16, n32 = shape["n"]
        fc = shape["F"]
        self.shape = dict(shape)
        self.stem = nn.Sequential(nn.Conv2d(1, c0, 3, 2, padding=1, bias=False), nn.BatchNorm2d(c0), nn.ReLU())

        def stage(cin, cout, n, s):
            d = DILATIONS[s]
            return nn.Sequential(Sep(cin, cout, 2), *[Sep(cout, cout, 1, d[i % len(d)]) for i in range(n)])
        self.s4, self.s8 = stage(c0, c1, n4, 4), stage(c1, c2, n8, 8)
        self.s16, self.s32 = stage(c2, c3, n16, 16), stage(c3, c4, n32, 32)
        self.lat = nn.ModuleList([Lateral(c, fc) for c in (c1, c2, c3, c4)])
        self.smooth = Sep(fc, fc)
        self.head = nn.Conv2d(fc, 4 + tags, 1, bias=True)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = -2.0
            self.head.bias[2:4] = 2.0

    def forward(self, g):
        x = (g.float() - 128.0) / 128.0
        f4 = self.s4(self.stem(x))
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        p = self.lat[3](f32)
        p = self.lat[2](f16) + F.interpolate(p, scale_factor=2.0, mode="nearest")
        p = self.lat[1](f8) + F.interpolate(p, scale_factor=2.0, mode="nearest")
        p = self.lat[0](f4) + F.interpolate(p, scale_factor=2.0, mode="nearest")
        y = self.head(self.smooth(p))
        return torch.cat([y[:, :2], 4.0 * y[:, 2:4], y[:, 4:]], 1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _fold(conv, bn):
    w = conv.weight.detach().double()
    s = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
    b = bn.bias.detach().double() - bn.running_mean.detach().double() * s
    if conv.bias is not None:
        b = b + conv.bias.detach().double() * s
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                    conv.groups, bias=True)
    with torch.no_grad():
        new.weight.copy_((w * s.view(-1, 1, 1, 1)).float())
        new.bias.copy_(b.float())
    return new


def fold_bn(model):
    """An eval-mode copy with every BatchNorm folded into its convolution (Identity in its place)."""
    m = copy.deepcopy(model).eval()
    for mod in m.modules():
        if isinstance(mod, Sep):
            mod.dw, mod.bn1 = _fold(mod.dw, mod.bn1), nn.Identity()
            mod.pw, mod.bn2 = _fold(mod.pw, mod.bn2), nn.Identity()
        elif isinstance(mod, Lateral):
            mod.conv, mod.bn = _fold(mod.conv, mod.bn), nn.Identity()
    m.stem = nn.Sequential(_fold(m.stem[0], m.stem[1]), nn.ReLU())
    return m


def export_onnx(model, path, meta, check=True):
    """Fold, export (opset 17, dynamic H / W) with vocr.* metadata, and check against torch on two random inputs."""
    import onnx
    import onnxruntime as ort
    m = fold_bn(model)
    dummy = torch.randint(0, 256, (1, 1, 256, 512), dtype=torch.uint8)
    torch.onnx.export(m, (dummy,), str(path), input_names=["grey"], output_names=["maps"], opset_version=17,
                      dynamic_axes={"grey": {2: "height", 3: "width"}, "maps": {2: "map_height", 3: "map_width"}},
                      do_constant_folding=True, dynamo=False)
    mo = onnx.load(str(path))
    del mo.metadata_props[:]
    for k, v in meta.items():
        p = mo.metadata_props.add()
        p.key, p.value = k, json.dumps(v, ensure_ascii=False)
    mo.producer_name = "vita-vn-ocr train/desktop_det.py"
    mo.doc_string = ATTRIBUTION
    onnx.checker.check_model(mo)
    onnx.save(mo, str(path))
    if check:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        worst = 0.0
        for h, w in ((96, 160), (544, 960)):
            x = torch.randint(0, 256, (1, 1, h, w), dtype=torch.uint8)
            with torch.no_grad():
                ref = model.eval()(x).numpy()
            got = sess.run(None, {"grey": x.numpy()})[0]
            worst = max(worst, float(np.abs(ref - got).max()))
        if worst > 1e-2:
            raise SystemExit(f"the ONNX export differs from the torch network by {worst}")
        return worst
    return None


def policy(size):
    """(kind, n) of a size policy for the VDSK header: native 0, fit:N 1, long:N 2."""
    if size in (None, "", "native"):
        return 0, 0
    kind, _, n = str(size).partition(":")
    if not n:
        kind, n = "fit", kind
    return {"fit": 1, "long": 2}[kind], int(n)


def vdsk_bytes(model, size=DEFAULT_SIZE, decode=None):
    """The VDSK version-1 blob of a DeskNet for desktop/vocr_desk.c (vocr_desk.h): a 144-byte header (shape, tags,
    the decode's constants as float64, the default size policy, the stages' dilation patterns), the BatchNorm-folded
    float32 weights in the C kernels' layouts (stem [3][3][c0]; depthwise [3][3][C]; pointwise [Cin][Cout]; the head
    [F][4 + T] with A1 and A2 times 4 folded in) in the network's order, then the CRC-32 of everything before it."""
    import struct
    import zlib
    m = fold_bn(model)
    sh = model.shape
    d = dict(DECODE, **(decode or {}))
    parts = []

    def add(t):
        parts.append(np.ascontiguousarray(t.detach().cpu().numpy(), np.float32).ravel())

    def dw(conv):
        add(conv.weight[:, 0].permute(1, 2, 0))
        add(conv.bias)

    def pw(conv):
        add(conv.weight[:, :, 0, 0].t())
        add(conv.bias)
    dw(m.stem[0])
    for stage in (m.s4, m.s8, m.s16, m.s32):
        for blk in stage:
            dw(blk.dw)
            pw(blk.pw)
    for lat in m.lat:
        pw(lat.conv)
    dw(m.smooth.dw)
    pw(m.smooth.pw)
    w = m.head.weight[:, :, 0, 0].t().clone()
    b = m.head.bias.clone()
    w[:, 2:4] *= 4.0
    b[2:4] *= 4.0
    add(w)
    add(b)
    body = np.concatenate(parts).astype("<f4").tobytes()
    kind, n = policy(size)
    dl = [DILATIONS[s] for s in (4, 8, 16, 32)]
    hdr = b"VDSK" + struct.pack("<III", 1, VDSK_HEADER, VDSK_HEADER + len(body) + 4)
    hdr += struct.pack("<5I", *sh["c"]) + struct.pack("<4I", *sh["n"])
    hdr += struct.pack("<5I", sh["F"], model.head.out_channels - 4, int(d["min_cells"]), kind, n)
    hdr += struct.pack("<4I", *[len(x) for x in dl]) + bytes(v for x in dl for v in (x + [0, 0, 0, 0])[:4])
    hdr += struct.pack("<4d", float(d["conf_min_logit"]), float(d["pad_pct"]), float(d["pad_min"]),
                       float(d["min_across"]))
    hdr += struct.pack("<2I", int(d["pad_frame"]), int(d["pad_frame_min"]))
    hdr += bytes(VDSK_HEADER - len(hdr))
    if len(hdr) != VDSK_HEADER:
        raise AssertionError(len(hdr))
    blob = hdr + body
    return blob + struct.pack("<I", zlib.crc32(blob) & 0xFFFFFFFF)


def load_run(run_dir):
    """(DeskNet in eval mode, checkpoint) of a run's final.pt."""
    ck = torch.load(Path(run_dir) / "final.pt", map_location="cpu", weights_only=False)
    if ck.get("kind") != KIND:
        raise SystemExit(f"{run_dir}/final.pt: not a desktop detector checkpoint")
    m = DeskNet(ck["shape"], int(ck.get("tags", TAGS)))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def cmd_export(argv):
    """desktop_det.py export --run RUN --out DIR [--name NAME] [--size fit:960]: NAME.onnx and NAME.vdsk."""
    ap = argparse.ArgumentParser(prog="desktop_det.py export")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="vocr_desktop_det")
    ap.add_argument("--size", default=DEFAULT_SIZE)
    ap.add_argument("--pad", default="pct", choices=sorted(PAD_RULES))
    a = ap.parse_args(argv)
    m, ck = load_run(a.run)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    worst = export_onnx(m, out / f"{a.name}.onnx", onnx_meta(ck["shape"], a.name, a.size, PAD_RULES[a.pad],
                                                           m.head.out_channels - 4))
    blob = vdsk_bytes(m, a.size, PAD_RULES[a.pad])
    (out / f"{a.name}.vdsk").write_bytes(blob)
    rec = {"run": str(a.run), "pt_sha256": sha256_file(Path(a.run) / "final.pt"), "size": a.size,
           "onnx": f"{a.name}.onnx", "onnx_sha256": sha256_file(out / f"{a.name}.onnx"),
           "onnx_bytes": (out / f"{a.name}.onnx").stat().st_size, "vdsk": f"{a.name}.vdsk",
           "vdsk_sha256": sha256_file(out / f"{a.name}.vdsk"), "vdsk_bytes": len(blob),
           "onnx_vs_torch_max_abs": worst, "params": n_params(m), "pad": a.pad}
    print(json.dumps(rec), flush=True)
    return rec


def onnx_meta(shape, run_name, size=DEFAULT_SIZE, decode=None, tags=TAGS):
    d = dict(DECODE, **(decode or {}))
    out = f"maps float32 [1, {4 + tags}, H / 4, W / 4]: K, O logits, A1, A2 (input px), {tags} tag logits"
    return {"vocr.kind": KIND, "vocr.name": run_name, "vocr.shape": shape, "vocr.stride": 4, "vocr.align": 32,
            "vocr.tags": tags, "vocr.classes": ["dialogue", "name", "choice", "ui", "scene"][:1 + tags],
            "vocr.size": size, **{f"vocr.{k}": v for k, v in d.items()},
            "vocr.input": "grey uint8 [1, 1, H, W], H and W multiples of 32; integer luma (77 R + 150 G + 29 B) >> 8",
            "vocr.output": out, "vocr.licence": "GPL-3.0-or-later; non-commercial use only (training inputs)",
            "vocr.attribution": ATTRIBUTION}


# ----------------------------------------------------------------------------- windows

def _setup():
    import textspot2_train as T2
    import textspot6_train as B
    import textspot8_train as T8M
    import textspot_train as ST
    B.AUG.update(blend_p=T8M.T8["blend_p"], blend_alpha=list(T8M.T8["blend_alpha"]))
    return B, T2, T8M, ST


RUBY = {"max_rel": 0.6, "gap": 0.5, "overlap": 0.25, "inside": 0.8}
DESK_LONG = 960.0       # a "desk_*" source's canvas is scaled to this long side before the window's scale factor


def ruby_like(lines, rules=RUBY):
    """Indices of the horizontal lines that look like ruby (furigana) over another line: at most max_rel of its
    height, its bottom within `overlap` of that line's height below the line's top and at most `gap` of it above,
    and at least `inside` of its width over the line (MeikiOCR's boxes, the real half's labels, box ruby as lines)."""
    out = []
    hz = [(i, ln) for i, ln in enumerate(lines) if not int(ln[4])]
    for i, a in hz:
        ha, wa = a[3] - a[1], a[2] - a[0]
        if ha <= 0 or wa <= 0:
            continue
        for j, b in hz:
            if j == i:
                continue
            hb = b[3] - b[1]
            if hb <= 0 or ha > rules["max_rel"] * hb:
                continue
            if not (a[3] <= b[1] + rules["overlap"] * hb and b[1] - a[3] <= rules["gap"] * hb):
                continue
            if min(a[2], b[2]) - max(a[0], b[0]) >= rules["inside"] * wa:
                out.append(i)
                break
    return out


def make_window_ms(canvas, smp, src, rng, sources, scale, aug=True, ruby_filter=False):
    """One 256 x 512 (at most) window: (img uint8, positives [[x0, y0, x1, y1, vertical, weight, mask]], ignore)
    in window px, the canvas resized by a log-uniform factor in `scale` (module docstring)."""
    import cv2
    B, T2, T8M, _ = _setup()
    lines, ignore = smp["lines"], smp["ignore"]
    if ruby_filter and not src.startswith("hook") and not src.startswith("desk_hook"):
        rb = set(ruby_like(lines))
        if rb:
            ignore = list(ignore) + [ln[:4] for i, ln in enumerate(lines) if i in rb]
            lines = [ln for i, ln in enumerate(lines) if i not in rb]
    up = 1
    if aug and src == "hook" and smp["kind"] == "psp" and rng.random() < B.AUG["adrenaline_p"]:
        canvas = cv2.resize(np.ascontiguousarray(canvas), (2 * canvas.shape[1], 2 * canvas.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        up = 2
    H, W = canvas.shape
    s = float(math.exp(rng.uniform(math.log(scale[0]), math.log(scale[1]))))
    if src.startswith("desk_"):         # a desktop-size page: first to the 960 px long side the policies read at
        s *= DESK_LONG / max(H, W)
    rh, rw = min(H, max(1, int(round(B.WIN_H / s)))), min(W, max(1, int(round(B.WIN_W / s))))
    ox, oy = int(rng.integers(0, W - rw + 1)), int(rng.integers(0, H - rh + 1))
    img = np.asarray(canvas[oy:oy + rh, ox:ox + rw])
    nh, nw = min(B.WIN_H, max(1, int(round(rh * s)))), min(B.WIN_W, max(1, int(round(rw * s))))
    if (nh, nw) != (rh, rw):
        img = cv2.resize(np.ascontiguousarray(img), (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else
                         cv2.INTER_LINEAR)
    zx, zy = nw / rw, nh / rh
    h, w = img.shape
    empty_win = None
    if aug and sources is not None and B.AUG["blend_p"] > 0:
        e = sources.empty_canvas(rng)
        if e is not None and e.shape[0] >= h and e.shape[1] >= w:
            ey, ex = int(rng.integers(0, e.shape[0] - h + 1)), int(rng.integers(0, e.shape[1] - w + 1))
            empty_win = e[ey:ey + h, ex:ex + w]
    if aug:
        img = B._photometric(img, rng, empty_win)

    def to(b):
        x0, y0, x1, y1 = (float(v) * up for v in b[:4])
        return [int(math.floor((x0 - ox) * zx)), int(math.floor((y0 - oy) * zy)),
                int(math.ceil((x1 - ox) * zx)), int(math.ceil((y1 - oy) * zy))]

    ign = [to(b) for b in ignore]
    pos = []
    for ln in lines:
        b = to(ln)
        vertical = int(ln[4])
        if (b[2] - b[0] if vertical else b[3] - b[1]) < B.MIN_ACROSS:
            ign.append(b)
            continue
        cls = ln[5] if len(ln) > 5 else "text"
        pos.append([*b, vertical, T2.line_weight(ln), B.CLASS_MASK[cls]])
    img = np.ascontiguousarray(img)
    if aug:
        if rng.random() < T8M.T8["icon_p"]:
            img = np.array(img, copy=True)
            for x0, y0, x1, y1 in T8M.add_icons(img, pos, ign, rng):
                if rng.random() < 0.5:
                    patch = np.ascontiguousarray(img[y0:y1, x0:x1])
                    img[y0:y1, x0:x1] = cv2.GaussianBlur(patch, (0, 0), rng.uniform(0.4, 1.0))
        if rng.random() < T8M.T8["tight_p"]:
            img, pos, ign = T8M.tight_crop(img, pos, ign, rng)
    return img, pos, ign


def make_batch_ms(sources, rng, scale, batch, aug=True, ruby_filter=False):
    B, _, T8M, _ = _setup()
    WH, WW = B.WIN_H, B.WIN_W
    x = np.zeros((batch, 1, WH, WW), np.uint8)
    ho, wo = WH // B.MAP_STRIDE, WW // B.MAP_STRIDE
    tg = [np.zeros((batch, ho, wo), np.float32) for _ in range(5)]
    vm = np.zeros((batch, ho, wo), bool)
    km = np.zeros_like(vm)
    mk = np.zeros((batch, ho, wo), np.int64)
    for b in range(batch):
        src, canvas, smp = sources.pick(rng)
        img, pos, ign = make_window_ms(canvas, smp, src, rng, sources, scale, aug, ruby_filter)
        h, w = img.shape
        x[b, 0] = np.pad(img, ((0, WH - h), (0, WW - w)), mode="edge")
        K, O, A1, A2, v, k, Wt, M = T8M.targets(pos, ign, w, h)  # noqa: E741
        for t, a in zip(tg, (K, O, A1, A2, Wt)):
            t[b] = a
        vm[b], km[b], mk[b] = v, k, M
    return (torch.from_numpy(x), *(torch.from_numpy(t) for t in tg[:4]), torch.from_numpy(vm), torch.from_numpy(km),
            torch.from_numpy(tg[4]), torch.from_numpy(mk))


class ExtraPool:
    """The shards of one manifest (textspot6-pages or #97's frames: entries with "dir", "name", npy / json sha256)
    under ROOT, memory-mapped, as one source of the mix (--extra NAME=MANIFEST:ROOT[:SPLITS]). A sample needs off, h,
    w, kind, lines and ignore; its window is cut like a synthetic frame's (no Adrenaline upscale)."""

    def __init__(self, manifest, root, splits=None, verify=False):
        doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.data, self.items = [], []
        for s in doc["shards"]:
            if splits and s.get("split") and s["split"] not in splits:
                continue
            base = Path(root) / s.get("dir", "") / s["name"]
            npy, js = base.with_suffix(".npy"), base.with_suffix(".json")
            if verify:
                for path, want in ((npy, s["npy_sha256"]), (js, s["json_sha256"])):
                    if sha256_file(path) != want:
                        raise SystemExit(f"{path}: sha256 is not {manifest}'s")
            d = len(self.data)
            self.data.append(np.load(npy, mmap_mode="r"))
            for smp in json.loads(js.read_text(encoding="utf-8"))["samples"]:
                if splits and smp.get("split") and smp["split"] not in splits:
                    continue
                self.items.append((d, smp))
        if not self.items:
            raise SystemExit(f"{manifest}: no samples under {root}")

    def draw(self, rng):
        d, smp = self.items[int(rng.integers(len(self.items)))]
        return np.asarray(self.data[d][smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(smp["h"], smp["w"]), smp


def parse_extra(specs):
    """--extra NAME=MANIFEST:ROOT[:SPLITS] -> {name: (manifest, root, splits or None)}."""
    out = {}
    for sp in specs or []:
        name, _, rest = sp.partition("=")
        parts = rest.split(":")
        if len(parts) < 2:
            raise SystemExit(f"--extra {sp}: NAME=MANIFEST:ROOT[:SPLITS]")
        out[name] = (parts[0], parts[1], tuple(parts[2].split(",")) if len(parts) > 2 and parts[2] else None)
    return out


def add_extra_sources(sources, mix, extra):
    """Add every --extra pool to a textspot6_train.Sources with its --mix weight, and renormalise."""
    for name, (man, root, splits) in extra.items():
        if name not in mix:
            raise SystemExit(f"--extra {name}: no weight for it in --mix")
        pool = ExtraPool(man, root, splits)
        sources.names.append(name)
        sources.p = np.append(np.asarray(sources.p, np.float64) * 1.0, float(mix[name]))
        sources.draw[name] = pool.draw
    p = np.asarray(sources.p, np.float64)
    sources.p = p / p.sum()
    return sources


class Batches(torch.utils.data.IterableDataset):
    def __init__(self, cfg, seed):
        super().__init__()
        self.cfg, self.seed = cfg, seed

    def __iter__(self):
        B, _, _, ST = _setup()
        c = self.cfg
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        hook = B.HookPool(c["hook_manifest"], c["hook_shards"], c["hook_splits"]) if c["mix"].get("hook") else None
        frames = ST.FramePool(c["manifests"], c["shards"], skip=c["skip"]) if c["manifests"] else None
        base_mix = {k: v for k, v in c["mix"].items() if k in ("hook", "synth", "real")}
        sources = B.Sources(base_mix, hook, frames, c["title_temp"])
        sources.p = np.asarray(sources.p) * sum(base_mix.values())     # back to the --mix scale before the extras
        add_extra_sources(sources, c["mix"], c.get("extra", {}))
        rng = np.random.default_rng([self.seed, wid])
        while True:
            yield make_batch_ms(sources, rng, c["scale"], c["batch"], c.get("aug", True),
                                c.get("ruby_filter", False))


# ----------------------------------------------------------------------------- the run

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", required=True, choices=sorted(SHAPES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hook-manifest")
    ap.add_argument("--hook-shards")
    ap.add_argument("--hook-splits", default="train")
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mix", default="hook=0.45,synth=0.2,real=0.35")
    ap.add_argument("--extra", nargs="*", default=[], help="NAME=MANIFEST:ROOT[:SPLITS]: more sources for --mix")
    ap.add_argument("--scale", default="0.42,1.9")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--ruby-filter", action="store_true",
                    help="ruby-like boxes of the non-hooked sources: ignore areas")
    ap.add_argument("--bn-batches", type=int, default=32)
    ap.add_argument("--ema-start", type=int, default=1000)
    a = ap.parse_args(argv)
    a.mix_d = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    a.scale_t = tuple(float(v) for v in a.scale.split(","))
    a.name = f"{a.shape}_seed{a.seed}"
    return a


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256():
    files = ["train/desktop_det.py", "tools/vocr_desktop.py", "train/textspot6_train.py", "train/textspot8_train.py",
             "train/textspot2_train.py", "train/textspot3_train.py", "train/textspot_train.py",
             "train/textdet_train.py"]
    return {f: sha256_file(ROOT / f) for f in files if (ROOT / f).exists()}


@torch.no_grad()
def reestimate_bn(model, it, n):
    """BatchNorm running statistics as the cumulative average over n training batches (no augmentation change)."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for _ in range(n):
        model(next(it)[0])
    for m, mo in zip(bns, saved):
        m.momentum = mo
    model.eval()
    return len(bns)


def main(argv=None):
    a = parse(argv)
    B, T2, T8M, ST = _setup()
    import textdet_train as TT
    t_proc = time.time()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.probe:
        raise SystemExit(f"{out} is not empty: a run starts from scratch in a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    shape = SHAPES[a.shape]
    skip_shots, skip_vns, lists = T2.skip_lists(a.manifest) if a.manifest else (frozenset(), frozenset(), [])
    cfg = {"mix": a.mix_d, "hook_manifest": a.hook_manifest, "hook_shards": a.hook_shards,
           "hook_splits": tuple(a.hook_splits.split(",")), "manifests": list(a.manifest), "shards": list(a.shards),
           "skip": (skip_shots, skip_vns), "title_temp": B.TITLE_TEMP, "scale": a.scale_t, "batch": a.batch,
           "aug": not a.no_aug, "ruby_filter": a.ruby_filter, "extra": parse_extra(a.extra)}
    hook = B.HookPool(a.hook_manifest, a.hook_shards, cfg["hook_splits"]) if a.mix_d.get("hook") else None
    frames = ST.FramePool(a.manifest, a.shards, skip=(skip_shots, skip_vns)) if a.manifest else None
    base_mix = {k: v for k, v in a.mix_d.items() if k in ("hook", "synth", "real")}
    sources = B.Sources(base_mix, hook, frames, B.TITLE_TEMP)
    sources.p = np.asarray(sources.p) * sum(base_mix.values())
    add_extra_sources(sources, a.mix_d, cfg["extra"])
    pool_rec = {"sources": dict(zip(sources.names, (round(float(q), 6) for q in sources.p))),
                "extra": {k: {"manifest": v[0], "root": v[1], "splits": v[2],
                              "manifest_sha256": sha256_file(v[0])} for k, v in cfg["extra"].items()},
                "hook": hook.summary(B.TITLE_TEMP) if hook else None,
                "frames": frames.summary() if frames else None, "text_free_synthetic": len(sources.empty)}
    skip_rec = ST.skip_record(lists, frames.dropped) if lists and frames is not None else None
    del hook, frames, sources
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    model = DeskNet(shape)
    opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=a.lr)
    loader = torch.utils.data.DataLoader(Batches(cfg, a.seed), batch_size=None, num_workers=a.workers,
                                         persistent_workers=a.workers > 0,
                                         prefetch_factor=4 if a.workers > 0 else None)
    it = iter(loader)
    run = {"kind": "desktop-detector-run", "name": a.name, "started": ST.utc(), "shape": shape,
           "params": n_params(model), "args": {k: v for k, v in vars(a).items() if k not in ("mix_d", "scale_t")},
           "mix": a.mix_d, "scale": list(a.scale_t), "window": [B.WIN_H, B.WIN_W], "batch": a.batch,
           "ruby_filter": RUBY if a.ruby_filter else None,
           "aug": B.AUG if not a.no_aug else None, "t8": {k: T8M.T8[k] for k in ("edge_cells", "edge_weight", "icon_p",
                                                                            "tight_p", "tight_margin")},
           "class_mask": B.CLASS_MASK, "pool": pool_rec, "skip": skip_rec,
           "manifests_sha256": {m: sha256_file(m) for m in [*a.manifest, *([a.hook_manifest] if a.hook_manifest
                                                                           else [])]},
           "code_sha256": code_sha256(), "git": ST.git_state(), "torch": torch.__version__, "threads": a.threads,
           "workers": a.workers}
    if not a.probe:
        (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    log = open(out / ("probe.jsonl" if a.probe else "log.jsonl"), "a", encoding="utf-8")
    ema = T2.EMA(0.999)
    n_steps = a.probe if a.probe else a.steps
    t0 = time.time()
    agg, n_agg, times = {}, 0, []
    model.train()
    for step in range(n_steps):
        ts = time.perf_counter()
        lr = T2.lr_at(step, a.steps, a.lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, K, O, A1, A2, vm, km, Wt, Mk = next(it)  # noqa: E741
        td = time.perf_counter()
        y = model(x)
        loss, parts = T2.loss_of(y[:, :4], K, O, A1, A2, vm, km, Wt)
        lt, tparts = B.tag_loss(y[:, 4:], Mk, Wt)
        loss = loss + B.TAG_WEIGHT * lt
        parts.update(tparts)
        parts["loss"] = float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), B.CLIP)
        opt.step()
        if step + 1 >= a.ema_start:
            ema.step(model.parameters())
        times.append((td - ts, time.perf_counter() - ts))
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n_agg += 1
        if (step + 1) % a.log_every == 0 or step + 1 == n_steps:
            rec = {"step": step + 1, "lr": round(lr, 6), "sec": round(time.time() - t0, 1),
                   **{k: round(v / n_agg, 5) for k, v in agg.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            agg, n_agg = {}, 0
    if a.probe:
        ms = np.array([t for _, t in times[2:]]) * 1e3
        wait = np.array([t for t, _ in times[2:]]) * 1e3
        probe = {"name": a.name, "params": n_params(model), "ms_per_step": round(float(ms.mean()), 1),
                 "ms_waiting_for_data": round(float(wait.mean()), 1), "threads": a.threads, "workers": a.workers,
                 "startup_s": round(t0 - t_proc, 1)}
        (out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n")
        print("desktop_det probe:", json.dumps(probe), flush=True)
        return probe
    ema.copy_to(model.parameters())
    n_bn = reestimate_bn(model, it, a.bn_batches)
    log.write(json.dumps({"step": n_steps, "ema": "the run's model", "ema_steps": ema.steps, "bn_layers": n_bn}) + "\n")
    torch.save({"kind": KIND, "shape": shape, "tags": TAGS, "seed": a.seed, "steps": a.steps,
                "state_dict": model.state_dict(), "run": run}, out / "final.pt")
    worst = export_onnx(model, out / "final.onnx", onnx_meta(shape, a.name))
    done = {"finished": ST.utc(), "name": a.name, "steps": a.steps, "seconds": round(time.time() - t0, 1),
            "wall_seconds": round(time.time() - t_proc, 1), "params": n_params(model),
            "onnx_bytes": (out / "final.onnx").stat().st_size, "onnx_sha256": sha256_file(out / "final.onnx"),
            "pt_sha256": sha256_file(out / "final.pt"), "onnx_vs_torch_max_abs": worst}
    (out / "done.json").write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
    log.close()
    print("desktop_det finished:", json.dumps(done), flush=True)
    return done


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        cmd_export(sys.argv[2:])
    else:
        main()

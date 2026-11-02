#!/usr/bin/env python3
"""textspot-desktop's network family (issue #169's speed tier, docs/textspot-desktop.md): a very small float32 fully
convolutional line detector for desktop frames, built for a C AVX2 runtime first and ONNX second.

The family (SHAPES):
  stem    a k x k convolution at stride k (k = `patch`, 4): the grey frame cut into k x k patches, each mapped to c[0]
          channels (a space-to-depth and a 1x1 convolution in one), BatchNorm, ReLU
  stages  at strides 4, 8, 16 and 32 (c[0..3] channels): n[0] separable blocks at stride 4 after the stem; for each
          later stage a separable down-sampling block (3x3 depthwise at stride 2, BN, ReLU, 1x1 to the stage's
          channels, BN, ReLU) and n[i] separable blocks (3x3 depthwise with the stage's dilation pattern, BN, ReLU,
          1x1, BN, a residual add, ReLU)
  gc      optional global context at stride 32 (gc = 1): the mean of every position's features, a 1x1 layer and ReLU,
          added to every position (page-level context for a few hundred MACs)
  pyramid 1x1 laterals to F channels at every stride from 32 down to the head's stride, nearest 2x up-sampling and
          addition
  head    `smooth` separable blocks at the head's stride (hs: 8 or 4), then a 1x1 layer with OUT = 6 + TAGS maps
The maps (channel order): K (line kernel logit), O (orientation logit: vertical when > 0), A1, A2 (distances from a
cell's centre to the line's near and far edges across it, input px), E1, E2 (distances from a cell's centre to the
line's start and end along it, input px; trained on the cells near each end), then TAGS tag logits (textspot6's
classes after dialogue). A1, A2, E1 and E2 are the head's values times `hs` (the scale is folded into the head's
weights for export). The input is the grey frame as uint8 [N, 1, H, W] (H, W multiples of 32), normalised in the
graph as (g - 128) / 128.
"""
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KIND = "textspot-desktop-detector"
TAGS = 4
OUT = 6 + TAGS
ALIGN = 32
CLASSES = ("dialogue", "name", "choice", "ui", "scene")
DILATIONS = {4: [1], 8: [1, 2, 1, 2], 16: [1, 2, 1], 32: [1, 2]}

SHAPES = {
    # the screening family: c = channels at strides 4, 8, 16, 32; n = residual blocks per stage; F = pyramid channels;
    # hs = the head's stride; smooth = separable blocks at the head's stride; gc = global context at stride 32
    "p8a": {"patch": 4, "c": [16, 32, 64, 96], "n": [0, 2, 2, 1], "F": 32, "hs": 8, "smooth": 1, "gc": 0},
    "p8s": {"patch": 4, "c": [16, 24, 48, 64], "n": [0, 1, 2, 1], "F": 24, "hs": 8, "smooth": 1, "gc": 0},
    "p8t": {"patch": 4, "c": [8, 24, 48, 64], "n": [0, 1, 1, 1], "F": 24, "hs": 8, "smooth": 0, "gc": 0},
    "p8g": {"patch": 4, "c": [16, 32, 64, 96], "n": [0, 2, 2, 1], "F": 32, "hs": 8, "smooth": 1, "gc": 1},
    "p4a": {"patch": 4, "c": [16, 32, 64, 96], "n": [0, 2, 2, 1], "F": 16, "hs": 4, "smooth": 0, "gc": 0},
    # round 2 (protocol v2, amendment 1): more capacity within the speed goal (C AVX2, 1 thread, 960x544, random
    # weights: q8b 9.7 ms, q8c 16.7, q4b 11.3, q4c 16.2; MeikiOCR's detector takes ~100-110 ms, so the bar is ~20-22)
    "q8b": {"patch": 4, "c": [32, 64, 128, 192], "n": [0, 3, 3, 2], "F": 64, "hs": 8, "smooth": 2, "gc": 1},
    "q8c": {"patch": 4, "c": [32, 96, 160, 224], "n": [1, 4, 3, 2], "F": 64, "hs": 8, "smooth": 2, "gc": 1},
    "q4b": {"patch": 4, "c": [32, 64, 128, 160], "n": [1, 3, 3, 2], "F": 48, "hs": 4, "smooth": 1, "gc": 1},
    "q4c": {"patch": 4, "c": [32, 80, 128, 160], "n": [1, 4, 3, 2], "F": 64, "hs": 4, "smooth": 1, "gc": 1},
}


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


class GlobalCtx(nn.Module):
    """x + relu(W mean(x) + b), broadcast over every position."""

    def __init__(self, c):
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1, bias=True)
        with torch.no_grad():
            self.fc.weight.mul_(0.1)
            self.fc.bias.zero_()

    def forward(self, x):
        return x + F.relu(self.fc(x.mean((2, 3), keepdim=True)))


class TsdNet(nn.Module):
    def __init__(self, shape, tags=TAGS):
        super().__init__()
        self.shape = dict(shape)
        c, n, fc, hs, k = shape["c"], shape["n"], shape["F"], shape["hs"], shape["patch"]
        if k != 4 or hs not in (4, 8):
            raise ValueError("the family has a 4 x 4 patch stem and a head at stride 4 or 8")
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], k, k, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU())
        self.s4 = nn.Sequential(*[Sep(c[0], c[0], 1, DILATIONS[4][i % 1]) for i in range(n[0])])

        def stage(cin, cout, m, s):
            d = DILATIONS[s]
            return nn.Sequential(Sep(cin, cout, 2), *[Sep(cout, cout, 1, d[i % len(d)]) for i in range(m)])
        self.s8 = stage(c[0], c[1], n[1], 8)
        self.s16 = stage(c[1], c[2], n[2], 16)
        self.s32 = stage(c[2], c[3], n[3], 32)
        self.gc = GlobalCtx(c[3]) if shape.get("gc") else None
        lat_c = ([c[0]] if hs == 4 else []) + [c[1], c[2], c[3]]
        self.lat = nn.ModuleList([Lateral(ci, fc) for ci in lat_c])
        self.smooth = nn.Sequential(*[Sep(fc, fc) for _ in range(int(shape.get("smooth", 0)))])
        self.head = nn.Conv2d(fc, 6 + tags, 1, bias=True)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = -2.0
            self.head.bias[2:6] = 1.0

    def features(self, g):
        x = (g.float() - 128.0) / 128.0
        f4 = self.s4(self.stem(x))
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        if self.gc is not None:
            f32 = self.gc(f32)
        fs = ([f4] if self.shape["hs"] == 4 else []) + [f8, f16, f32]
        p = self.lat[-1](fs[-1])
        for i in range(len(fs) - 2, -1, -1):
            p = self.lat[i](fs[i]) + F.interpolate(p, scale_factor=2.0, mode="nearest")
        return self.smooth(p)

    def forward(self, g):
        y = self.head(self.features(g))
        hs = float(self.shape["hs"])
        return torch.cat([y[:, :2], hs * y[:, 2:6], y[:, 6:]], 1)


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


def macs(shape, w, h, tags=TAGS):
    """Multiply-accumulates of the network on a w x h input (multiples of 32), by layer kind, counted from the
    shapes (a convolution's MACs: output cells x its kernel taps x input channels per group x output channels)."""
    c, n, fc, hs = shape["c"], shape["n"], shape["F"], shape["hs"]
    cells = {s: (h // s) * (w // s) for s in (4, 8, 16, 32)}
    out = {"stem": cells[4] * 16 * c[0], "stages": 0, "pyramid": 0, "head": 0, "gc": 0}

    def sep(cin, cout, s_out):
        return cells[s_out] * (9 * cin + cin * cout)
    out["stages"] += n[0] * sep(c[0], c[0], 4)
    for i, s in enumerate((8, 16, 32)):
        out["stages"] += sep(c[i], c[i + 1], s) + n[i + 1] * sep(c[i + 1], c[i + 1], s)
    if shape.get("gc"):
        out["gc"] = c[3] * c[3]
    lat = ([(c[0], 4)] if hs == 4 else []) + [(c[1], 8), (c[2], 16), (c[3], 32)]
    out["pyramid"] = sum(cells[s] * ci * fc for ci, s in lat)
    out["head"] = int(shape.get("smooth", 0)) * sep(fc, fc, hs) + cells[hs] * fc * (6 + tags)
    out["total"] = sum(v for k, v in out.items())
    return out


def onnx_meta(shape, name, size, decode):
    return {"vocr.kind": KIND, "vocr.name": name, "vocr.shape": shape, "vocr.stride": int(shape["hs"]),
            "vocr.align": ALIGN, "vocr.tags": TAGS, "vocr.classes": list(CLASSES), "vocr.size": size,
            "vocr.decode": decode, "vocr.input": "grey uint8 [1, 1, H, W], H and W multiples of 32; integer luma "
            "(77 R + 150 G + 29 B) >> 8", "vocr.output": f"maps float32 [1, {OUT}, H / hs, W / hs]: K, O logits, A1, "
            "A2, E1, E2 (input px), 4 tag logits"}


def export_onnx(model, path, meta, check=True, doc=""):
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
    mo.producer_name = "vita-vn-ocr train/tsdesk_det.py"
    mo.doc_string = doc
    onnx.checker.check_model(mo)
    onnx.save(mo, str(path))
    if not check:
        return None
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    worst = 0.0
    for hh, ww in ((96, 160), (544, 960)):
        x = torch.randint(0, 256, (1, 1, hh, ww), dtype=torch.uint8)
        with torch.no_grad():
            ref = model.eval()(x).numpy()
        got = sess.run(None, {"grey": x.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    if worst > 1e-2:
        raise SystemExit(f"the ONNX export differs from the torch network by {worst}")
    return worst

"""The learned text detector (issue #96, docs/textdet.md): a tiny fully convolutional network on the grey region.

A shape is a stem, a downsampling stage and a few blocks at the output stride (4 or 8 input pixels per output
cell), then a 1x1 head with four maps per output cell:

  K   text kernel logit: the middle band of a line across it (max(h/4, s/2) either side of its centre line), its
      whole length along it
  O   orientation logit: > 0 for vertical text
  A1  distance from the cell centre back to the line's near edge (top, or left for vertical text), in px
  A2  distance from the cell centre to the far edge (bottom, or right)

Output cell (i, j) stands for the input block [s*i, s*i + s) x [s*j, s*j + s); its centre, s*i + (s - 1)/2, is
the point A1 and A2 are measured from (a labelling convention: the network learns the offset to its receptive
field's centre, s*i). Outputs are ceil(H/s) x ceil(W/s) for an H x W region.

Layers (Conv2d + BatchNorm, then ReLU except the depthwise halves of separable blocks and the head):

  stem   3x3 stride 2, 1 -> c0                                   (1/2)
  down   'sep': 3x3 depthwise stride 2, then 1x1 c0 -> c1        (1/4)
         'plain': 3x3 stride 2, c0 -> c1
  down8  (stride 8 only) the same again, c1 -> c2                 (1/8)
  blocks 'sep': 3x3 depthwise with dilation d, then 1x1; 'plain': 3x3 with dilation d
  head   1x1 -> 4 (no BatchNorm, no ReLU)

Training places a region in the top-left corner of a fixed-size window and multiplies every layer's output by the
region's validity mask at that layer's resolution, so the padded window computes exactly what the runtime computes
on the region alone (zero padding at every layer). QTextDet is the BatchNorm-folded network with int8 fake
quantisation (weights per output channel, activations per tensor with frozen scales; the head's K and O maps at
1/16 per code, A1 and A2 at 1/2 px per code), the arithmetic train/textdet_export.py writes into the blob.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = {}


def _shape(name, stride, c0, c1, chans, dil, block="sep", head="kroa", c2=None):
    SHAPES[name] = {"name": name, "stride": stride, "c0": c0, "c1": c1, "c2": c2, "chans": list(chans),
                    "dil": list(dil), "block": block, "head": head}


# stride 4, separable blocks: the size ladder
_shape("s4_c4", 4, 4, 4, [4, 4, 4], [1, 2, 4])
_shape("s4_c6", 4, 4, 6, [6, 6, 6], [1, 2, 4])
_shape("s4_c8", 4, 4, 8, [8, 8, 8], [1, 2, 4])
_shape("s4_c8n5", 4, 4, 8, [8, 8, 8, 8, 8], [1, 2, 4, 1, 2])
_shape("s4_c12", 4, 6, 12, [12, 12, 12], [1, 2, 4])
_shape("s4_c16", 4, 8, 16, [16, 16, 16], [1, 2, 4])
_shape("s4_c16n4", 4, 8, 16, [16, 16, 16, 16], [1, 2, 4, 1])
_shape("s4_c24", 4, 8, 16, [24, 24, 24], [1, 2, 2])
# stride 4, plain 3x3 blocks
_shape("s4p_c4", 4, 4, 4, [4, 4, 4], [1, 2, 4], block="plain")
_shape("s4p_c8", 4, 4, 8, [8, 8, 8], [1, 2, 4], block="plain")
# stride 8, separable blocks
_shape("s8_c8", 8, 4, 8, [8, 8, 8], [1, 2, 4], c2=8)
_shape("s8_c16", 8, 4, 8, [16, 16, 16], [1, 2, 4], c2=16)
_shape("s8_c24", 8, 8, 16, [24, 24, 24], [1, 2, 4], c2=24)
_shape("s8_c32", 8, 8, 16, [32, 32, 32], [1, 2, 4], c2=32)
_shape("s8_c32n4", 8, 8, 16, [32, 32, 32, 32], [1, 2, 2, 2], c2=32)
_shape("s8_c48", 8, 8, 24, [48, 48, 48], [1, 2, 2], c2=48)
# heads ablation: kernel + orientation only (dropped from the sweep by protocol amendment 2)
_shape("s4_c8_ko", 4, 4, 8, [8, 8, 8], [1, 2, 4], head="ko")
_shape("s8_c16_ko", 8, 4, 8, [16, 16, 16], [1, 2, 4], c2=16, head="ko")
# round 3 (protocol amendment 4): deeper shapes around round 2's front
_shape("s4_c12n4", 4, 6, 12, [12, 12, 12, 12], [1, 2, 4, 1])
_shape("s4_c20n4", 4, 8, 16, [20, 20, 20, 20], [1, 2, 4, 1])
_shape("s4_c24n4", 4, 8, 16, [24, 24, 24, 24], [1, 2, 2, 1])
_shape("s8_c16n4", 8, 4, 8, [16, 16, 16, 16], [1, 2, 4, 2], c2=16)
_shape("s8_c24n4", 8, 8, 16, [24, 24, 24, 24], [1, 2, 4, 2], c2=24)

N_OUT = 4                     # K, O, A1, A2
A_SCALE = 2                   # A1/A2 codes per pixel in the int8 blob
KO_SCALE = 16                 # K/O codes per logit unit
MIN_CELLS = {4: 2, 8: 1}      # the decode drops smaller kernel components


def layers(shape):
    """The layer list: dicts with kind ('conv' | 'dw'), cin, cout, k, stride, dil, relu, head."""
    s, out = shape, []
    out.append(dict(kind="conv", cin=1, cout=s["c0"], k=3, stride=2, dil=1, relu=True))

    def down(cin, cout):
        if s["block"] == "sep":
            out.append(dict(kind="dw", cin=cin, cout=cin, k=3, stride=2, dil=1, relu=False))
            out.append(dict(kind="conv", cin=cin, cout=cout, k=1, stride=1, dil=1, relu=True))
        else:
            out.append(dict(kind="conv", cin=cin, cout=cout, k=3, stride=2, dil=1, relu=True))

    down(s["c0"], s["c1"])
    c = s["c1"]
    if s["stride"] == 8:
        down(c, s["c2"])
        c = s["c2"]
    for cout, d in zip(s["chans"], s["dil"]):
        if s["block"] == "sep":
            out.append(dict(kind="dw", cin=c, cout=c, k=3, stride=1, dil=d, relu=False))
            out.append(dict(kind="conv", cin=c, cout=cout, k=1, stride=1, dil=1, relu=True))
        else:
            out.append(dict(kind="conv", cin=c, cout=cout, k=3, stride=1, dil=d, relu=True))
        c = cout
    out.append(dict(kind="conv", cin=c, cout=N_OUT, k=1, stride=1, dil=1, relu=False, head=True))
    for L in out:
        L.setdefault("head", False)
    return out


def out_size(n, stride):
    return (n + stride - 1) // stride


def layer_masks(shape, valid, height, width):
    """Per layer, the (N, 1, h_l, w_l) validity mask of regions of valid = [(h, w)] placed top left in a window of
    height x width."""
    out = []
    h, w = height, width
    vh = [v[0] for v in valid]
    vw = [v[1] for v in valid]
    for L in layers(shape):
        h, w = out_size(h, L["stride"]), out_size(w, L["stride"])
        vh = [out_size(x, L["stride"]) for x in vh]
        vw = [out_size(x, L["stride"]) for x in vw]
        m = torch.zeros(len(valid), 1, h, w)
        for i, (a, b) in enumerate(zip(vh, vw)):
            m[i, :, :a, :b] = 1
        out.append(m)
    return out


class TextDet(nn.Module):
    """The float network with BatchNorm (training before QAT)."""

    def __init__(self, shape):
        super().__init__()
        self.shape = dict(shape)
        self.spec = layers(shape)
        mods = []
        for L in self.spec:
            groups = L["cin"] if L["kind"] == "dw" else 1
            pad = (L["k"] - 1) // 2 * L["dil"]
            conv = nn.Conv2d(L["cin"], L["cout"], L["k"], stride=L["stride"], padding=pad, dilation=L["dil"],
                             groups=groups, bias=L["head"])
            seq = [conv]
            if not L["head"]:
                seq.append(nn.BatchNorm2d(L["cout"]))
            if L["relu"]:
                seq.append(nn.ReLU(inplace=True))
            mods.append(nn.Sequential(*seq))
        self.body = nn.ModuleList(mods)

    def forward(self, x, masks=None, taps=None):
        """x: N x 1 x H x W float, the grey region as (g - 128) / 128. taps: a list that receives every layer's
        output (for activation-range calibration)."""
        for i, m in enumerate(self.body):
            x = m(x)
            if masks is not None:
                x = x * masks[i]
            if taps is not None:
                taps.append(x)
        return x


def fold(model):
    """BatchNorm folded into each conv: [(weight, bias)] float64 tensors, one per layer."""
    out = []
    for seq, L in zip(model.body, model.spec):
        conv = seq[0]
        w = conv.weight.detach().double()
        b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(L["cout"], dtype=torch.float64)
        if not L["head"]:
            bn = seq[1]
            k = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
            w = w * k.view(-1, 1, 1, 1)
            b = (b - bn.running_mean.detach().double()) * k + bn.bias.detach().double()
        out.append((w, b))
    return out


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x + 0.5)

    @staticmethod
    def backward(ctx, g):
        return g


def fq_weight(w):
    """Symmetric per-output-channel int8: scale max|w_c| / 127, codes clamped to +-127 (train/textdet_export.py)."""
    s = w.detach().abs().flatten(1).amax(1).clamp(min=1e-12) / 127
    s = s.view(-1, *([1] * (w.dim() - 1)))
    return torch.clamp(_RoundSTE.apply(w / s), -127, 127) * s


def fq_act(x, scale, lo, hi):
    """Per-tensor (or per-channel for the head) activation codes: round(x / scale) clamped to [lo, hi]."""
    return torch.clamp(_RoundSTE.apply(x / scale), lo, hi) * scale


class QTextDet(nn.Module):
    """The folded network with int8 fake quantisation; act_scale[i] is layer i's output scale (float per code)."""

    def __init__(self, shape, folded, act_scale):
        super().__init__()
        self.shape = dict(shape)
        self.spec = layers(shape)
        self.w = nn.ParameterList([nn.Parameter(w.float().clone()) for w, _ in folded])
        self.b = nn.ParameterList([nn.Parameter(b.float().clone()) for _, b in folded])
        scales = []
        for L, s in zip(self.spec, act_scale):
            if L["head"]:
                scales.append(torch.tensor([1 / KO_SCALE, 1 / KO_SCALE, 1 / A_SCALE, 1 / A_SCALE]).view(1, 4, 1, 1))
            else:
                scales.append(torch.tensor(float(s)).view(1, 1, 1, 1))
        for i, s in enumerate(scales):
            self.register_buffer(f"s{i}", s)
        self.quant = True

    def scale(self, i):
        return getattr(self, f"s{i}")

    def forward(self, x, masks=None, taps=None):
        for i, L in enumerate(self.spec):
            w = fq_weight(self.w[i]) if self.quant else self.w[i]
            pad = (L["k"] - 1) // 2 * L["dil"]
            x = F.conv2d(x, w, self.b[i], stride=L["stride"], padding=pad, dilation=L["dil"],
                         groups=L["cin"] if L["kind"] == "dw" else 1)
            if L["relu"]:
                x = F.relu(x)
            if self.quant:
                x = fq_act(x, self.scale(i), 0 if L["relu"] else -128, 127)
            if masks is not None:
                x = x * masks[i]
            if taps is not None:
                taps.append(x)
        return x


def cost(shape, width, height):
    """Exact multiply-accumulates for a width x height region, parameters, int8 weight count and the row-buffer
    bytes of a streaming runtime (each 3x3 layer keeps (k - 1) * dilation + 1 rows of its input; the exact arena
    is runtime/vocr_textdet.c's vocr_textdet_arena_bytes)."""
    macs = params = 0
    w, h = width, height
    arena = 0
    for L in layers(shape):
        ow, oh = out_size(w, L["stride"]), out_size(h, L["stride"])
        per = L["k"] * L["k"] * (1 if L["kind"] == "dw" else L["cin"])
        macs += ow * oh * L["cout"] * per
        params += L["cout"] * per + L["cout"]
        rows = (L["k"] - 1) * L["dil"] + 1
        arena += rows * w * L["cin"]
        w, h = ow, oh
    biases = sum(L["cout"] for L in layers(shape))
    return {"macs": int(macs), "params": int(params), "weights_int8": int(params - biases), "arena_rows": int(arena),
            "out": [w, h]}


def receptive_field(shape):
    rf, jump = 1, 1
    for L in layers(shape):
        rf += (L["k"] - 1) * L["dil"] * jump
        jump *= L["stride"]
    return rf


def table(regions=((900, 150), (450, 80), (960, 544), (480, 272))):
    rows = []
    for name, s in SHAPES.items():
        r = {"shape": name, "rf": receptive_field(s)}
        for w, h in regions:
            r[f"macs_{w}x{h}"] = cost(s, w, h)["macs"]
        c = cost(s, 960, 544)
        r.update(params=c["params"], weights_int8=c["weights_int8"], arena_rows_960=c["arena_rows"])
        rows.append(r)
    return rows


if __name__ == "__main__":
    for r in table():
        print(f"{r['shape']:12s} rf {r['rf']:4d} params {r['params']:6d} w8 {r['weights_int8']:6d} "
              f"box {r['macs_900x150'] / 1e6:6.2f}M psp-box {r['macs_450x80'] / 1e6:5.2f}M "
              f"vita {r['macs_960x544'] / 1e6:6.1f}M psp {r['macs_480x272'] / 1e6:5.1f}M "
              f"arena-rows@960 {r['arena_rows_960']:6d} {math.ceil(r['arena_rows_960'] / 1024)}K")

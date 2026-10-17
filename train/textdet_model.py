"""The learned text detector (issue #96, docs/textdet.md): a tiny fully convolutional network on the grey region.

A shape is a stem, a downsampling stage and a few blocks at the output stride (4 or 8 input pixels per output
cell), then a 1x1 head with four maps per output cell:

  K   text kernel logit: the middle half of a line's height (across the line), its whole length along it
  O   orientation logit: > 0 for vertical text
  A1  distance from the cell centre back to the line's near edge (top, or left for vertical text), in px
  A2  distance from the cell centre to the far edge (bottom, or right)

`train/textdet_export.py` folds BatchNorm into the convolutions and writes the int8 blob that the C runtime
(runtime/vocr_textdet.c) and its bit-exact twin (synth/textdet_twin.py) run with integer arithmetic only.

Layers (every one is Conv2d + BatchNorm, then ReLU except the depthwise halves of separable blocks and the head):

  stem   3x3 stride 2, 1 -> c0                                   (1/2)
  down   'sep': 3x3 depthwise stride 2, then 1x1 c0 -> c1        (1/4)
         'plain': 3x3 stride 2, c0 -> c1
  down8  (stride 8 only) the same again, c1 -> c2                 (1/8)
  blocks 'sep': 3x3 depthwise with dilation d, then 1x1; 'plain': 3x3 with dilation d
  head   1x1 -> 4 (no BatchNorm, no ReLU)

Padding is (k - 1) / 2 * dilation on every side, so an output cell (i, j) of stride s is centred on input pixel
(s*i + (s-1)/2, s*j + (s-1)/2) whatever the region size: outputs are ceil(H/s) x ceil(W/s).
"""
import math

import torch.nn as nn

# Named shapes of the size sweep (eval/textdet_protocol.json "sweep"). c: block widths, d: their dilations.
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
# heads ablation: kernel + orientation only (no edge regression; the box is the kernel's extent unshrunk)
_shape("s4_c8_ko", 4, 4, 8, [8, 8, 8], [1, 2, 4], head="ko")
_shape("s8_c16_ko", 8, 4, 8, [16, 16, 16], [1, 2, 4], c2=16, head="ko")

N_OUT = 4                     # K, O, A1, A2 (a 'ko' head trains A1/A2 at zero weight; they are never read)
A_SCALE = 2                   # A1/A2 output code units per pixel in the int8 blob (0.5 px per code)


def layers(shape):
    """The layer list: dicts with kind ('conv' | 'dw'), cin, cout, k, stride, dil, relu, groups."""
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
    return out


class TextDet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = dict(shape)
        mods = []
        for L in layers(shape):
            groups = L["cin"] if L["kind"] == "dw" else 1
            pad = (L["k"] - 1) // 2 * L["dil"]
            conv = nn.Conv2d(L["cin"], L["cout"], L["k"], stride=L["stride"], padding=pad, dilation=L["dil"],
                             groups=groups, bias=bool(L.get("head")))
            seq = [conv]
            if not L.get("head"):
                seq.append(nn.BatchNorm2d(L["cout"]))
            if L["relu"]:
                seq.append(nn.ReLU(inplace=True))
            mods.append(nn.Sequential(*seq))
        self.body = nn.ModuleList(mods)

    def forward(self, x):
        """x: N x 1 x H x W float, the grey region as (g - 128) / 128."""
        for m in self.body:
            x = m(x)
        return x


def out_size(n, stride):
    return (n + stride - 1) // stride


def cost(shape, width, height):
    """Exact multiply-accumulates for a width x height region, parameters, int8 weight bytes (weights only) and
    the C runtime's streaming arena estimate (runtime/vocr_textdet.c is exact; this is the same formula)."""
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
            c = cost(s, w, h)
            r[f"macs_{w}x{h}"] = c["macs"]
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

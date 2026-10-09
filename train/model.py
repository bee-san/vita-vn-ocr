"""Tiny CTC line recognizer for Vita VN dialogue.

Grey input, height 32 (or 24), any width W (multiple of 8). Output: W/8 frames of
class logits (class 0 = CTC blank). Only Conv2d, BatchNorm (folded at export), ReLU
and residual adds, so the Vita runtime is a handful of plain C kernels: 3x3 stem,
3x3 depthwise, 1x1 pointwise, a 2x1 height collapse and 1x5 temporal depthwise.
Height 24 uses exactly the same layers: 24 -> 12 -> 6 -> 3 -> 2 -> 1.

About 0.76M parameters at width 1.0 with the 5,355-class head; the head is
factorised through a 96-wide bottleneck because at stride 8 it dominates compute.

Every conv is a QConv2d: a plain nn.Conv2d (same parameters and state_dict keys)
that can fake-quantise its weights to symmetric per-output-channel int8 with a
straight-through gradient, for quantisation-aware training (set_weight_quant).
"""
import torch
import torch.nn as nn


def fake_quant_weight(w, bits=8):
    """Symmetric per-output-channel quantise-dequantise: scale = max|w_c| / 127,
    q = clamp(round(w / scale), -127, 127). The gradient passes straight through.

    Per output channel, this commutes with BatchNorm folding (folding multiplies
    channel c by one scalar k_c, and round(-x) = -round(x)), so quantising the raw
    conv weight here gives exactly the int8 weights of the BN-folded export."""
    qmax = 2 ** (bits - 1) - 1
    s = w.detach().abs().flatten(1).amax(1).clamp(min=1e-12) / qmax
    s = s.view(-1, *([1] * (w.dim() - 1)))
    q = torch.clamp(torch.round(w / s), -qmax, qmax) * s
    return w + (q - w).detach()


class QConv2d(nn.Conv2d):
    """nn.Conv2d with optional int8 weight fake quantisation (off by default)."""

    wbits = 0

    def forward(self, x):
        w = fake_quant_weight(self.weight, self.wbits) if self.wbits else self.weight
        return self._conv_forward(x, w, self.bias)


def set_weight_quant(model, bits):
    """bits=8 turns on int8 weight fake quantisation in every conv, 0 turns it off."""
    for m in model.modules():
        if isinstance(m, QConv2d):
            m.wbits = bits


def cbr(cin, cout, k, stride=1, groups=1, pad=None):
    if pad is None:
        pad = (k[0] // 2, k[1] // 2) if isinstance(k, tuple) else k // 2
    return nn.Sequential(QConv2d(cin, cout, k, stride, pad, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DS(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, residual when the shape is unchanged."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = cbr(cin, cin, 3, stride, groups=cin)
        self.pw = cbr(cin, cout, 1)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return x + y if self.res else y


class Ctx(nn.Module):
    """Temporal context on the 1 x T feature row: depthwise 1x5 + pointwise, residual."""

    def __init__(self, c, k=5):
        super().__init__()
        self.dw = cbr(c, c, (1, k), groups=c, pad=(0, k // 2))
        self.pw = cbr(c, c, 1)

    def forward(self, x):
        return x + self.pw(self.dw(x))


def channels(width):
    """The five distinct channel counts (nominally 32, 64, 128, 160, 192) at a width
    multiplier, each rounded to a multiple of 8 and at least 8."""
    c = lambda n: max(8, int(round(n * width / 8)) * 8)  # noqa: E731
    return [c(n) for n in (32, 64, 128, 160, 192)]


class TinyVNOCR(nn.Module):
    def __init__(self, n_classes, width=1.0, bottleneck=96):
        super().__init__()
        c32, c64, c128, c160, c192 = channels(width)
        self.features = nn.Sequential(
            cbr(1, c32, 3, 2),                           # 16 x W/2  (12 at height 24)
            DS(c32, c64, 2),                             # 8 x W/4
            DS(c64, c64, 1),
            DS(c64, c128, 2),                            # 4 x W/8
            DS(c128, c128, 1),
            DS(c128, c160, (2, 1)),                      # 2 x W/8
            DS(c160, c160, 1),
            cbr(c160, c192, (2, 1), pad=(0, 0)),         # 1 x W/8
            Ctx(c192),
            Ctx(c192),
        )
        self.drop = nn.Dropout(0.1)
        self.bottleneck = QConv2d(c192, bottleneck, 1, bias=False)
        self.head = QConv2d(bottleneck, n_classes, 1)

    def forward(self, x):
        """x: N x 1 x H x W (H = 32 or 24), values in [-1, 1]. Returns logits T x N x C."""
        f = self.head(self.bottleneck(self.drop(self.features(x))))
        assert f.shape[2] == 1, f"input height {x.shape[2]} does not collapse to 1 row"
        return f.squeeze(2).permute(2, 0, 1)


def from_checkpoint(ck):
    """Rebuild the model a train.py checkpoint (dict or path) describes, v1 included.
    Returns (model in eval mode, meta) with meta = {vocab, width, bottleneck, height, unk}."""
    if not isinstance(ck, dict):
        ck = torch.load(ck, map_location="cpu", weights_only=False)
    meta = {"vocab": ck["vocab"], "width": ck.get("width", 1.0),
            "bottleneck": ck.get("bottleneck", 96), "height": ck.get("height", 32),
            "unk": ck.get("unk")}
    model = TinyVNOCR(len(meta["vocab"]) + 1, width=meta["width"], bottleneck=meta["bottleneck"])
    model.load_state_dict(ck["model"])
    return model.eval(), meta

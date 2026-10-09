"""Tiny CTC line recognizer for Vita VN dialogue.

Grey input, height 32, any width W (multiple of 8). Output: W/8 frames of class
logits (class 0 = CTC blank). Only Conv2d, BatchNorm (folded at export), ReLU and
residual adds, so the Vita runtime is a handful of plain C kernels: 3x3 stem,
3x3 depthwise, 1x1 pointwise, a 2x1 height collapse and 1x5 temporal depthwise.

About 0.76M parameters at width 1.0 with the 5,355-class head; the head is
factorised through a 96-wide bottleneck because at stride 8 it dominates compute.
"""
import torch.nn as nn


def cbr(cin, cout, k, stride=1, groups=1, pad=None):
    if pad is None:
        pad = (k[0] // 2, k[1] // 2) if isinstance(k, tuple) else k // 2
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, pad, groups=groups, bias=False),
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


class TinyVNOCR(nn.Module):
    def __init__(self, n_classes, width=1.0, bottleneck=96):
        super().__init__()
        c = lambda n: max(8, int(round(n * width / 8)) * 8)  # noqa: E731
        self.features = nn.Sequential(
            cbr(1, c(32), 3, 2),                         # 16 x W/2
            DS(c(32), c(64), 2),                         # 8 x W/4
            DS(c(64), c(64), 1),
            DS(c(64), c(128), 2),                        # 4 x W/8
            DS(c(128), c(128), 1),
            DS(c(128), c(160), (2, 1)),                  # 2 x W/8
            DS(c(160), c(160), 1),
            cbr(c(160), c(192), (2, 1), pad=(0, 0)),     # 1 x W/8
            Ctx(c(192)),
            Ctx(c(192)),
        )
        self.drop = nn.Dropout(0.1)
        self.bottleneck = nn.Conv2d(c(192), bottleneck, 1, bias=False)
        self.head = nn.Conv2d(bottleneck, n_classes, 1)

    def forward(self, x):
        """x: N x 1 x 32 x W, values in [-1, 1]. Returns logits T x N x C."""
        f = self.head(self.bottleneck(self.drop(self.features(x))))
        return f.squeeze(2).permute(2, 0, 1)

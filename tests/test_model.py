"""train/model.py: output shape, size, and the layer set the plain-C runtime has kernels for."""
import math

import pytest
import torch
from torch import nn

from common import load_vocab
from model import TinyVNOCR

N_CLASSES = len(load_vocab()) + 1          # + CTC blank


@pytest.fixture(scope="module", params=[0.25, 0.5, 1.0], ids=lambda w: f"width{w}")
def model(request):
    torch.manual_seed(0)
    return TinyVNOCR(N_CLASSES, width=request.param).eval()


@pytest.mark.parametrize("w", [8, 100, 1024])
def test_output_is_time_batch_classes(model, w):
    """Height 32 in, one frame per 8 columns out: T = ceil(W / 8), as train.py's in_len assumes."""
    with torch.no_grad():
        y = model(torch.zeros(3, 1, 32, w))
    assert tuple(y.shape) == (math.ceil(w / 8), 3, N_CLASSES)


def test_width_one_stays_tiny():
    """Memory comes first (AGENTS.md). Raise this bound only on purpose, with the reason in the PR."""
    params = sum(p.numel() for p in TinyVNOCR(N_CLASSES).parameters())
    assert params < 1_000_000, f"TinyVNOCR(width=1.0) has {params:,} parameters (0.76M in README.md)"


def conv_kind(m):
    """Name of the runtime kernel that runs this conv, or None."""
    if m.dilation != (1, 1) or m.padding_mode != "zeros":
        return None
    dw = m.groups == m.in_channels == m.out_channels and m.groups > 1
    k, s, p = m.kernel_size, m.stride, m.padding
    if k == (3, 3) and m.groups == 1 and m.in_channels == 1 and s == (2, 2) and p == (1, 1):
        return "3x3 stem"
    if k == (3, 3) and dw and s in {(1, 1), (2, 2), (2, 1)} and p == (1, 1):
        return "3x3 depthwise"
    if k == (1, 1) and m.groups == 1 and s == (1, 1) and p == (0, 0):
        return "1x1 pointwise"
    if k == (2, 1) and m.groups == 1 and s == (1, 1) and p == (0, 0):
        return "2x1 height collapse"
    if k == (1, 5) and dw and s == (1, 1) and p == (0, 2):
        return "1x5 temporal depthwise"
    return None


def test_only_layers_the_c_runtime_implements(model):
    """runtime/ (plain C) implements exactly these layers; BatchNorm folds into the convs at
    export and Dropout is the identity at inference. A new layer needs a runtime kernel first:
    coordinate with runtime/ and extend conv_kind() in the same PR."""
    leaves = [m for m in model.modules() if not list(m.children())]
    other = sorted({type(m).__name__ for m in leaves
                    if not isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout))})
    assert not other, f"layers without a C runtime kernel: {other}"
    unknown = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d) and conv_kind(m) is None]
    assert not unknown, f"convolutions without a C runtime kernel: {unknown}"
    assert model.head.out_channels == N_CLASSES

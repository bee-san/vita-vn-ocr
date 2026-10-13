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
straight-through gradient, for quantisation-aware training (set_weight_quant). The
quantised set matches the int8 export: every conv but the stem and the depthwise ones.
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


def int8_conv(m):
    """True for the convs an int8 export stores as int8 (runtime/train/export.py --int8):
    pointwise, row collapse, bottleneck and head. The stem (one input channel) and the
    depthwise convs stay float32, as do all biases and scales."""
    return isinstance(m, nn.Conv2d) and m.groups == 1 and m.in_channels > 1


def set_weight_quant(model, bits):
    """bits=8 turns on int8 weight fake quantisation in every int8_conv, 0 turns it off."""
    for m in model.modules():
        if isinstance(m, QConv2d):
            m.wbits = bits if int8_conv(m) else 0


def cbr(cin, cout, k, stride=1, groups=1, pad=None):
    if pad is None:
        pad = (k[0] // 2, k[1] // 2) if isinstance(k, tuple) else k // 2
    return nn.Sequential(QConv2d(cin, cout, k, stride, pad, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class RepCBR(nn.Module):
    """Training-time multi-branch conv + BN + ReLU (structural re-parameterisation, as in RepVGG,
    Ding et al. 2021, and MobileOne, Vasu et al. 2023): n parallel k x k conv + BN branches, a
    1 x 1 conv + BN branch on the kernel's centre tap, and an identity BN branch when the shape
    is unchanged, summed before the ReLU. In eval mode the sum is one linear conv, so fused()
    turns it back into cbr()'s single conv + BN: the deployed model, the export and the C
    runtime are exactly the plain TinyVNOCR's.

    Used only where the runtime keeps float32 weights (the stem and the depthwise convs), so the
    fused weights never meet int8 quantisation (RepVGG's merged weights are known to quantise
    badly, Chu et al. 2022, arXiv:2212.01593)."""

    def __init__(self, cin, cout, k, stride=1, groups=1, pad=None, n=1):
        super().__init__()
        k, stride = _pair(k), _pair(stride)
        pad = (k[0] // 2, k[1] // 2) if pad is None else _pair(pad)
        if pad != (k[0] // 2, k[1] // 2) or k[0] % 2 == 0 or k[1] % 2 == 0:
            raise ValueError("RepCBR needs an odd kernel with 'same' padding (centre-aligned branches)")
        self.k, self.stride, self.pad, self.groups, self.cin, self.cout = k, stride, pad, groups, cin, cout
        self.convs = nn.ModuleList(
            nn.Sequential(QConv2d(cin, cout, k, stride, pad, groups=groups, bias=False), nn.BatchNorm2d(cout))
            for _ in range(n))
        self.scale = (nn.Sequential(QConv2d(cin, cout, 1, stride, 0, groups=groups, bias=False),
                                    nn.BatchNorm2d(cout)) if k != (1, 1) else None)
        self.skip = nn.BatchNorm2d(cout) if stride == (1, 1) and cin == cout else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.convs[0](x)
        for b in self.convs[1:]:
            y = y + b(x)
        if self.scale is not None:
            y = y + self.scale(x)
        if self.skip is not None:
            y = y + self.skip(x)
        return self.act(y)

    @torch.no_grad()
    def fused(self):
        """(weight, bias), float64: the single conv every branch together equals in eval mode."""
        kh, kw = self.k
        cin_g = self.cin // self.groups

        def fold(w, bn):
            s = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
            return w.double() * s.view(-1, 1, 1, 1), bn.bias.double() - bn.running_mean.double() * s

        w, b = torch.zeros(self.cout, cin_g, kh, kw, dtype=torch.float64), torch.zeros(self.cout, dtype=torch.float64)
        centre = (slice(None), slice(None), slice(kh // 2, kh // 2 + 1), slice(kw // 2, kw // 2 + 1))
        for br in self.convs:
            dw, db = fold(br[0].weight, br[1])
            w, b = w + dw, b + db
        if self.scale is not None:
            dw, db = fold(self.scale[0].weight, self.scale[1])
            w[centre] += dw
            b = b + db
        if self.skip is not None:
            eye = torch.zeros(self.cout, cin_g, 1, 1, dtype=torch.float64)
            eye[torch.arange(self.cout), torch.arange(self.cout) % cin_g] = 1.0
            dw, db = fold(eye, self.skip)
            w[centre] += dw
            b = b + db
        return w, b

    @torch.no_grad()
    def load_plain(self, seq):
        """Start from a plain cbr()'s conv + BN: branch 0 takes them and every other branch
        contributes nothing (BN weight and bias 0), so the function is unchanged."""
        self.convs[0][0].weight.copy_(seq[0].weight)
        self.convs[0][1].load_state_dict(seq[1].state_dict())
        extra = [br[1] for br in self.convs[1:]] + ([self.scale[1]] if self.scale is not None else []) + \
            ([self.skip] if self.skip is not None else [])
        for bn in extra:
            bn.weight.zero_()
            bn.bias.zero_()


def conv_bn(cin, cout, k, stride=1, groups=1, pad=None, rep=0):
    """cbr(), or with rep >= 1 its training-time multi-branch form RepCBR (rep parallel k x k
    branches plus the 1 x 1 and identity branches)."""
    return RepCBR(cin, cout, k, stride, groups, pad, n=rep) if rep else cbr(cin, cout, k, stride, groups, pad)


class DS(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, residual when the shape is unchanged."""

    def __init__(self, cin, cout, stride, rep=0):
        super().__init__()
        self.dw = conv_bn(cin, cin, 3, stride, groups=cin, rep=rep)
        self.pw = cbr(cin, cout, 1)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return x + y if self.res else y


class Ctx(nn.Module):
    """Temporal context on the 1 x T feature row: depthwise 1x5 + pointwise, residual."""

    def __init__(self, c, k=5, rep=0):
        super().__init__()
        self.dw = conv_bn(c, c, (1, k), groups=c, pad=(0, k // 2), rep=rep)
        self.pw = cbr(c, c, 1)

    def forward(self, x):
        return x + self.pw(self.dw(x))


def channels(width):
    """The five distinct channel counts (nominally 32, 64, 128, 160, 192) at a width
    multiplier, each rounded to a multiple of 8 and at least 8."""
    c = lambda n: max(8, int(round(n * width / 8)) * 8)  # noqa: E731
    return [c(n) for n in (32, 64, 128, 160, 192)]


class TinyVNOCR(nn.Module):
    """width scales every channel count; depth is the number of stride-1 DS blocks after
    each of the three downsampling DS blocks (v1: 1); ctx is the number of temporal Ctx
    blocks (v1: 2) and ctx_k their kernel width (v1: 5; the runtime allows up to 7).
    chans, when given, replaces channels(width) with five explicit channel counts (stem,
    8-row, 4-row, 2-row stages and the frame row; multiples of 8), e.g. a late-wide shape.
    rep >= 1 trains the stem and every depthwise conv with RepCBR's extra branches (rep
    parallel k x k branches); fuse_rep() turns such a model into the plain one it equals.
    The defaults build v1 exactly (same layers, same state_dict keys)."""

    def __init__(self, n_classes, width=1.0, bottleneck=96, depth=1, ctx=2, ctx_k=5, chans=None, rep=0):
        super().__init__()
        if chans is not None:
            chans = tuple(int(c) for c in chans)
            if len(chans) != 5 or any(c < 8 or c % 8 for c in chans):
                raise ValueError(f"chans must be five multiples of 8, got {chans}")
        c32, c64, c128, c160, c192 = chans or channels(width)
        self.arch = dict(width=width, bottleneck=bottleneck, depth=depth, ctx=ctx, ctx_k=ctx_k,
                         chans=chans, rep=rep)
        self.features = nn.Sequential(
            conv_bn(1, c32, 3, 2, rep=rep),              # 16 x W/2  (12 at height 24)
            DS(c32, c64, 2, rep),                        # 8 x W/4
            *[DS(c64, c64, 1, rep) for _ in range(depth)],
            DS(c64, c128, 2, rep),                       # 4 x W/8
            *[DS(c128, c128, 1, rep) for _ in range(depth)],
            DS(c128, c160, (2, 1), rep),                 # 2 x W/8
            *[DS(c160, c160, 1, rep) for _ in range(depth)],
            cbr(c160, c192, (2, 1), pad=(0, 0)),         # 1 x W/8
            *[Ctx(c192, ctx_k, rep) for _ in range(ctx)],
        )
        self.drop = nn.Dropout(0.1)
        self.bottleneck = QConv2d(c192, bottleneck, 1, bias=False)
        self.head = QConv2d(bottleneck, n_classes, 1)

    def forward(self, x, return_features=False):
        """x: N x 1 x H x W (H = 32 or 24), values in [-1, 1]. Returns logits T x N x C (and,
        with return_features, the N x C192 x 1 x T frame-row features the head reads)."""
        feats = self.features(x)
        f = self.head(self.bottleneck(self.drop(feats)))
        assert f.shape[2] == 1, f"input height {x.shape[2]} does not collapse to 1 row"
        logits = f.squeeze(2).permute(2, 0, 1)
        return (logits, feats) if return_features else logits


ARCH_KEYS = {"width": 1.0, "bottleneck": 96, "depth": 1, "ctx": 2, "ctx_k": 5}


def _plain_pairs(rep_model, plain):
    """(rep module, plain cbr Sequential) for every place the two layouts differ."""
    for a, b in zip(rep_model.features, plain.features):
        if isinstance(a, RepCBR):
            yield a, b
        elif isinstance(a, (DS, Ctx)) and isinstance(a.dw, RepCBR):
            yield a.dw, b.dw


@torch.no_grad()
def fuse_rep(model):
    """The plain TinyVNOCR (rep=0, same shape) a RepCBR-trained model equals in eval mode: each
    RepCBR becomes one conv with BN weight 1, bias = the fused bias, mean 0 and var 1 - eps (so
    the BN is the identity plus the bias). Every other parameter and buffer is copied."""
    kw = dict(model.arch)
    kw["rep"] = 0
    plain = TinyVNOCR(model.head.out_channels, **kw)
    reps = {id(a): b for a, b in _plain_pairs(model, plain)}
    sd = plain.state_dict()
    src = model.state_dict()
    rep_names = {n for n, m in model.named_modules() if isinstance(m, RepCBR)}
    for k in sd:
        if not any(k.startswith(n + ".") for n in rep_names):
            sd[k] = src[k].clone()
    plain.load_state_dict(sd)
    for a in (m for m in model.modules() if isinstance(m, RepCBR)):
        seq = reps[id(a)]
        w, b = a.fused()
        seq[0].weight.copy_(w.float())
        seq[1].weight.fill_(1.0)
        seq[1].bias.copy_(b.float())
        seq[1].running_mean.zero_()
        seq[1].running_var.fill_(1.0 - seq[1].eps)
        seq[1].num_batches_tracked.copy_(a.convs[0][1].num_batches_tracked)
    return plain.eval()


@torch.no_grad()
def load_plain_into_rep(model, plain_sd):
    """Load a plain checkpoint's state_dict into a rep model of the same shape, as a function-
    preserving start (RepCBR.load_plain): fine-tuning a rep model from a plain pretrain."""
    kw = dict(model.arch)
    kw["rep"] = 0
    plain = TinyVNOCR(model.head.out_channels, **kw)
    plain.load_state_dict(plain_sd)
    rep_names = {n for n, m in model.named_modules() if isinstance(m, RepCBR)}
    sd = model.state_dict()
    for k, v in plain.state_dict().items():
        if k in sd and not any(k.startswith(n + ".") for n in rep_names):
            sd[k] = v.clone()
    model.load_state_dict(sd)
    for a, seq in _plain_pairs(model, plain):
        a.load_plain(seq)
    return model


def from_checkpoint(ck):
    """Rebuild the model a train.py checkpoint (dict or path) describes, v1 included.
    Returns (model in eval mode, meta) with meta = {vocab, height, unk} plus ARCH_KEYS
    (and chans, when the checkpoint records explicit channel counts)."""
    if not isinstance(ck, dict):
        ck = torch.load(ck, map_location="cpu", weights_only=False)
    meta = {"vocab": ck["vocab"], "height": ck.get("height", 32), "unk": ck.get("unk"),
            **{k: ck.get(k, v) for k, v in ARCH_KEYS.items()}}
    extra = {}
    if ck.get("chans"):
        meta["chans"] = extra["chans"] = tuple(int(c) for c in ck["chans"])
    model = TinyVNOCR(len(meta["vocab"]) + 1, **{k: meta[k] for k in ARCH_KEYS}, **extra)
    model.load_state_dict(ck["model"])
    return model.eval(), meta

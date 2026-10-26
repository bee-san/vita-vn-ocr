"""Tiny CTC line recognizer for Vita VN dialogue.

Grey input, height 32 (or 24), any width W (multiple of 8). Output: W/8 frames of
class logits (class 0 = CTC blank). With stride=16 (or 12) one DS block downsamples the
width once more (stage_strides()), so W (a multiple of 16 or 12) gives W/16 (W/12) frames
from the same parameters. Only Conv2d, BatchNorm (folded at export), ReLU
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


class GlobalAttn(nn.Module):
    """One light global self-attention layer over the frame row (VOCR6 stage 3, docs/vocr6-attn.md), on the
    bottleneck's output z (N x d x 1 x T):

        z' = z + g * O(concat_h softmax(Q_h(z) K_h(z)^T / sqrt(dh)) V_h(z))

    Q, K, V (d -> dim) and O (dim -> d) are 1x1 QConv2d, so the int8 export and QAT treat them like every other
    pointwise conv. dim is split into `heads` heads of dh = dim / heads channels, a multiple of 8 (the runtime's
    8-wide blocks). K has no bias: a bias on K adds the same score to every key of a query, which the softmax cancels.
    There is no positional embedding (the depthwise temporal convs before it give position, as conditional positional
    encodings do: Chu et al. 2021, arXiv:2102.10882) and no LayerNorm. g is a learnt scalar initialised to 0 (ReZero:
    Bachlechner et al. 2020, arXiv:2003.04887), so a model built with the block and loaded from a checkpoint without
    it (load_state) starts as exactly that checkpoint's model. The export folds 1 / sqrt(dh) into Q and g into O.

    lengths (N frame counts) masks a training batch's right padding out of the keys; a line read on its own (the
    runtime, train2's validation, eval) has none."""

    def __init__(self, d, dim=96, heads=2):
        super().__init__()
        dim, heads = int(dim), int(heads)
        if heads < 1 or dim < 8 or dim % heads or (dim // heads) % 8:
            raise ValueError(f"attention: dim {dim} must split into {heads} heads of a multiple of 8 channels")
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.q = QConv2d(d, dim, 1)
        self.k = QConv2d(d, dim, 1, bias=False)
        self.v = QConv2d(d, dim, 1)
        self.o = QConv2d(dim, d, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, z, lengths=None):
        n, _, h, t = z.shape
        assert h == 1, "attention runs on the 1-row frame row"
        q, k, v = (m(z).reshape(n, self.heads, self.dh, t) for m in (self.q, self.k, self.v))
        s = torch.matmul(q.transpose(2, 3), k) * (self.dh ** -0.5)            # N x heads x T(query) x T(key)
        if lengths is not None:
            pad = torch.arange(t, device=z.device)[None, :] >= torch.as_tensor(lengths, device=z.device)[:, None]
            s = s.masked_fill(pad[:, None, None, :], float("-inf"))
        a = torch.matmul(s.softmax(-1), v.transpose(2, 3))                    # N x heads x T x dh
        y = a.permute(0, 1, 3, 2).reshape(n, self.dim, 1, t)
        return z + self.gate * self.o(y)


ATTN_DEFAULTS = {"attn": 0, "attn_dim": 96, "attn_heads": 2}


def attn_arch(src):
    """The attention keys of a checkpoint or arch dict: {} without the block (attn 0 or absent: every checkpoint
    before train2's --attn), else {attn: 1, attn_dim, attn_heads}."""
    if not int(src.get("attn") or 0):
        return {}
    if int(src["attn"]) != 1:
        raise ValueError(f"attn {src['attn']}: one attention layer (1) or none (0)")
    return {"attn": 1, **{k: int(src.get(k) or ATTN_DEFAULTS[k]) for k in ("attn_dim", "attn_heads")}}


@torch.no_grad()
def load_state(model, sd):
    """model.load_state_dict(sd), except that a state_dict without the attention block (a checkpoint from before
    --attn, or of an --attn 0 run) loads into a model with it: the block keeps its initialisation with the gate at 0,
    so the model computes exactly what the checkpoint's model does. Any other missing or unexpected key is an error.
    Returns "fresh-attn" in that case, else "loaded"."""
    attn = getattr(model, "attn", None)
    if attn is not None and not any(k.startswith("attn.") for k in sd):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected or any(not k.startswith("attn.") for k in missing):
            raise RuntimeError(f"state_dict does not fit the model: missing {missing}, unexpected {unexpected}")
        attn.gate.zero_()
        return "fresh-attn"
    model.load_state_dict(sd)
    return "loaded"


STRIDES = (8, 12, 16)


def stage_strides(stride):
    """(4-row stage's, 2-row stage's) downsampling DS strides for a horizontal frame step of 8, 12
    or 16 input columns. 8: (2, (2, 1)), the released shape. 16 ('ds22'): the 2-row stage's DS at
    (2, 2), so the 2-row stage and the frame row run at W/16. 12 ('ds3w3'): the 4-row stage's DS at
    (2, 3), so everything from the 4-row stage on runs at W/12 (the runtime takes op strides up to 3).
    Neither changes a parameter: a stride-12 or stride-16 model has a stride-8 model's state_dict."""
    if stride not in STRIDES:
        raise ValueError(f"stride must be one of {STRIDES}, got {stride}")
    return ((2, 3) if stride == 12 else 2), ((2, 2) if stride == 16 else (2, 1))


class TinyVNOCR(nn.Module):
    """width scales every channel count; depth is the number of stride-1 DS blocks after
    each of the three downsampling DS blocks (v1: 1); ctx is the number of temporal Ctx
    blocks (v1: 2) and ctx_k their kernel width (v1: 5; the runtime allows up to 7).
    chans, when given, replaces channels(width) with five explicit channel counts (stem,
    8-row, 4-row, 2-row stages and the frame row; multiples of 8), e.g. a late-wide shape.
    rep >= 1 trains the stem and every depthwise conv with RepCBR's extra branches (rep
    parallel k x k branches); fuse_rep() turns such a model into the plain one it equals.
    stride is the CTC frame step in input columns (8, 12 or 16; stage_strides()).
    attn = 1 adds GlobalAttn (attn_dim channels in attn_heads heads) between the bottleneck and the head; the head
    then reads z'. It is built after the head, so the other modules draw the same initial values for a seed.
    The defaults build v1 exactly (same layers, same state_dict keys)."""

    def __init__(self, n_classes, width=1.0, bottleneck=96, depth=1, ctx=2, ctx_k=5, chans=None, rep=0,
                 stride=8, attn=0, attn_dim=96, attn_heads=2):
        super().__init__()
        if chans is not None:
            chans = tuple(int(c) for c in chans)
            if len(chans) != 5 or any(c < 8 or c % 8 for c in chans):
                raise ValueError(f"chans must be five multiples of 8, got {chans}")
        s4, s2 = stage_strides(int(stride))
        c32, c64, c128, c160, c192 = chans or channels(width)
        self.stride = int(stride)
        self.arch = dict(width=width, bottleneck=bottleneck, depth=depth, ctx=ctx, ctx_k=ctx_k,
                         chans=chans, rep=rep, stride=self.stride)
        self.arch.update(attn_arch(dict(attn=attn, attn_dim=attn_dim, attn_heads=attn_heads)))
        self.features = nn.Sequential(
            conv_bn(1, c32, 3, 2, rep=rep),              # 16 x W/2  (12 at height 24)
            DS(c32, c64, 2, rep),                        # 8 x W/4
            *[DS(c64, c64, 1, rep) for _ in range(depth)],
            DS(c64, c128, s4, rep),                      # 4 x W/8 (W/12 at stride 12)
            *[DS(c128, c128, 1, rep) for _ in range(depth)],
            DS(c128, c160, s2, rep),                     # 2 x W/8 (W/12, W/16)
            *[DS(c160, c160, 1, rep) for _ in range(depth)],
            cbr(c160, c192, (2, 1), pad=(0, 0)),         # 1 x W/8 (W/12, W/16)
            *[Ctx(c192, ctx_k, rep) for _ in range(ctx)],
        )
        self.drop = nn.Dropout(0.1)
        self.bottleneck = QConv2d(c192, bottleneck, 1, bias=False)
        self.head = QConv2d(bottleneck, n_classes, 1)
        a = attn_arch(self.arch)
        self.attn = GlobalAttn(bottleneck, a["attn_dim"], a["attn_heads"]) if a else None

    def forward(self, x, return_features=False, lengths=None):
        """x: N x 1 x H x W (H = 32 or 24), values in [-1, 1]. Returns logits T x N x C (and,
        with return_features, the N x C192 x 1 x T frame-row features the bottleneck reads).
        lengths: N frame counts of a right-padded batch, which the attention block (attn = 1) masks out of its keys;
        a model without the block ignores it."""
        feats = self.features(x)
        z = self.bottleneck(self.drop(feats))
        if self.attn is not None:
            z = self.attn(z, lengths)
        f = self.head(z)
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
    load_state(plain, plain_sd)
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
    Returns (model in eval mode, meta) with meta = {vocab, height, unk, stride} plus ARCH_KEYS and ATTN_DEFAULTS'
    keys (and chans, when the checkpoint records explicit channel counts). A checkpoint without a
    stride (every one before train2's --stride) is stride 8; one without attn (every one before --attn) has no
    attention block (attn 0)."""
    if not isinstance(ck, dict):
        ck = torch.load(ck, map_location="cpu", weights_only=False)
    meta = {"vocab": ck["vocab"], "height": ck.get("height", 32), "unk": ck.get("unk"),
            "stride": int(ck.get("stride") or 8), **{k: ck.get(k, v) for k, v in ARCH_KEYS.items()}}
    extra = {"stride": meta["stride"], **attn_arch(ck)}
    meta.update({**ATTN_DEFAULTS, **attn_arch(ck)})
    if ck.get("chans"):
        meta["chans"] = extra["chans"] = tuple(int(c) for c in ck["chans"])
    model = TinyVNOCR(len(meta["vocab"]) + 1, **{k: meta[k] for k in ARCH_KEYS}, **extra)
    model.load_state_dict(ck["model"])
    return model.eval(), meta

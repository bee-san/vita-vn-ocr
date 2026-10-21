#!/usr/bin/env python3
"""Parity check: runtime/vocr_textspot3.c against its Python twin synth/textspot3_twin.py (the specification), for
VTSP version 1, 2 and 3 blobs. This part: the version-3 network builders the tests and the parity share.

  probe_net(variant)        a version-3 network that carries the d2s probe's stride-8 codes through identity map ops
                            (a dilated depthwise, a stride-2 chain to stride 64, zero-weight branches read through
                            upsamplings by 2, 4 and 8, a concatenation) into the d2s probe's head: its maps and
                            results must be #97's stride-4 probe's exactly
  wild_net(shape, seed)     random_int_layers with extreme requantisation corners (biases over the int32 range,
                            multipliers 1 .. 2^31 - 1, shifts 1 .. 62) and weights at +-127
  graph_net(seed)           a random op graph (not the grammar): pw ops over 1..3 earlier maps at any upsampling
                            that meets, dw ops of strides 1 and 2 and dilations 1..8, channels 8..64
"""
import hashlib
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_parity as TP  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_twin as T3  # noqa: E402

ONE = 1 << 30


def _q(L, wq, bq, m, sh):
    return {**L, "wq": np.asarray(wq, np.int8), "bq": np.asarray(bq, np.int64), "m": np.asarray(m, np.int64),
            "sh": np.asarray(sh, np.int64)}


def stream_layer(kind, k, stride, cin, cout, wq, bq, m, sh, relu=False, dil=1):
    return _q({"kind": kind, "k": k, "stride": stride, "dil": dil, "relu": relu, "cin": cin, "cout": cout,
               "head": False, "part": "stream"}, wq, bq, m, sh)


def op_layer(kind, src, up, cin, cout, wq, bq, m, sh, relu=False, stride=1, dil=1, head=False, out_stride=8):
    return _q({"kind": kind, "k": 3 if kind == "dw" else 1, "stride": stride, "dil": dil, "relu": relu,
               "cin": cin, "cout": cout, "src": list(src), "up": list(up), "head": head, "part": "op",
               "out_stride": out_stride}, wq, bq, m, sh)


def identity_dw(c):
    w = np.zeros((c, 1, 3, 3), np.int8)
    w[:, 0, 1, 1] = 1
    return w


def identity_pw(cout, cin, at=0):
    w = np.zeros((cout, cin, 1, 1), np.int8)
    for i in range(min(cout, cin - at)):
        w[i, at + i, 0, 0] = 1
    return w


def probe_layers(variant=0):
    """probe_net's integer layers (pack_net's dicts)."""
    centre = np.zeros((1, 1, 3, 3), np.int8)
    centre[0, 0, 1, 1] = 1
    taps = np.zeros((8, 1, 3, 3), np.int8)
    for dy in (0, 1):
        for dx in (0, 1):
            taps[2 * dy + dx, 0, 1 + dy, 1 + dx] = 1
    out = [stream_layer("conv", 3, 2, 1, 1, centre, [0], [ONE], [30]),
           stream_layer("dw", 3, 2, 1, 1, centre, [0], [ONE], [30]),
           stream_layer("conv", 3, 2, 1, 8, taps, [0] * 8, [ONE] * 8, [30] * 8)]     # F8: channels 4..7 zero
    ops = [op_layer("dw", [0], [0], 8, 8, identity_dw(8), [0] * 8, [ONE] * 8, [30] * 8, dil=2),          # map 1
           op_layer("dw", [1], [0], 8, 8, identity_dw(8), [0] * 8, [ONE] * 8, [30] * 8, stride=2,
                    out_stride=16),                                                                     # map 2
           op_layer("dw", [2], [0], 8, 8, identity_dw(8), [0] * 8, [ONE] * 8, [30] * 8, stride=2,
                    out_stride=32),                                                                     # map 3
           op_layer("dw", [3], [0], 8, 8, identity_dw(8), [0] * 8, [ONE] * 8, [30] * 8, stride=2,
                    out_stride=64),                                                                     # map 4
           op_layer("pw", [4], [0], 8, 8, np.zeros((8, 8, 1, 1)), [5] * 8, [ONE] * 8, [30] * 8, relu=True,
                    out_stride=64),                                                                     # map 5: 5s
           op_layer("pw", [1, 5, 3], [0, 3, 2], 24, 8, np.concatenate(
               [identity_pw(8, 8), np.zeros((8, 16, 1, 1), np.int8)], 1), [0] * 8, [ONE] * 8, [30] * 8)]   # map 6
    slope1, slope2, off1, off2 = TP.PROBE_VARIANTS[variant]
    w = np.zeros((16, 8, 1, 1), np.int8)
    bias, mult, shift = [], [], []
    for dy in (0, 1):
        for dx in (0, 1):
            q = 2 * dy + dx
            for mp, (wt, b, sh) in enumerate(((1, 0, 30), (1, -64, 30), (slope1, 4 * off1, 32),
                                              (slope2, 4 * off2, 32))):
                w[T3.channel(dy, dx, mp), q, 0, 0] = wt
                bias.append(b)
                mult.append(ONE)
                shift.append(sh)
    ops.append(op_layer("pw", [6], [0], 8, 16, w, bias, mult, shift, head=True))
    return out + ops


def probe_net(variant=0, min_cells=None):
    shape = {"name": f"probe_ms_v{variant}"}
    net = M3.pack_net(shape, probe_layers(variant), hashlib.sha256(shape["name"].encode()).digest())
    return net if min_cells is None else set_min_cells(net, min_cells)


def set_min_cells(net, n):
    b = bytearray(net)
    b[21] = n
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    T3.Net(bytes(b))
    return bytes(b)


def probe_reference(model):
    """The stride-4 version-1 probe (#97's) whose maps a version-3 probe must reproduce, as a textspot_twin Model
    with the probe's min cells and conf_min (never halving: the caller feeds it the version-3 network's input)."""
    import textspot_twin as S
    v = int(model.name[len("probe_ms_v"):])
    return S.Model(S.pack(TP.set_min_cells(TP.probe_blob(4, v), model.min_cells), 0, model.conf_min))


def wild_net(shape, seed):
    """random_int_layers of an M shape with some channels' requantisation at the loader's extremes (weights stay in
    +-127, the version-3 range): most such channels saturate."""
    rng = np.random.default_rng(seed)
    q = M3.random_int_layers(shape, seed, n_cal=2)
    for L in q:
        wild = rng.random(L["cout"]) < 0.35
        L["wq"] = L["wq"].copy()
        L["wq"][rng.random(L["wq"].shape) < 0.05] = 127 * (1 - 2 * (seed % 2))
        L["bq"] = np.where(wild, rng.integers(-(1 << 31), 1 << 31, L["cout"]), L["bq"])
        L["m"] = np.where(wild, rng.integers(1, 1 << 31, L["cout"]), L["m"])
        L["sh"] = np.where(wild, rng.integers(1, 63, L["cout"]), L["sh"])
        if wild.any():
            k = int(rng.integers(0, L["cout"]))
            L["bq"][k], L["m"][k], L["sh"][k] = (-(1 << 31), (1 << 31) - 1, 1) if seed % 2 else ((1 << 31) - 1, 1, 62)
    return M3.pack_net({"name": f"wild_{shape['name']}"[:31]}, q,
                       hashlib.sha256(f"wild:{shape['name']}:{seed}".encode()).digest())


def graph_layers(seed, n_ops=None):
    """A random version-3 network outside the grammar: a small stream to stride 8 (F8 of 8..32 channels), then
    random ops (pw over 1..3 earlier maps that meet at one stride, some sources repeated; dw of stride 1 with any
    dilation or stride 2), every map's channels a multiple of 8 up to 64, then the head over 1..3 maps of stride
    8 or deeper; random weights in +-127 and random requantisation (some identity-like, some wild)."""
    rng = np.random.default_rng(seed)
    c8 = int(rng.choice([8, 16, 24, 32]))
    c0 = int(rng.integers(1, 9))
    layers = []

    def wts(shape):
        w = rng.integers(-127, 128, shape).astype(np.int8)
        w[rng.random(shape) < 0.3] = 0
        return w

    def rq(cout, cin_k):
        # spread: an accumulator of sd ~ 60 sqrt(cin_k) to ~30 codes
        target = 30.0 / (60.0 * np.sqrt(cin_k) + 1)
        m, sh = zip(*[M3.E.quantize_multiplier(float(target * rng.uniform(0.3, 3))) for _ in range(cout)])
        b = rng.integers(-2000, 2000, cout)
        if rng.random() < 0.2:                                   # a wild channel or two
            k = int(rng.integers(0, cout))
            m = list(m)
            sh = list(sh)
            m[k], sh[k], b[k] = int(rng.integers(1, 1 << 31)), int(rng.integers(1, 63)), \
                int(rng.integers(-(1 << 31), 1 << 31))
        return b, m, sh

    layers.append(stream_layer("conv", 3, 2, 1, c0, wts((c0, 1, 3, 3)), *rq(c0, 9), relu=True))
    layers.append(stream_layer("dw", 3, 2, c0, c0, wts((c0, 1, 3, 3)), *rq(c0, 9)))
    layers.append(stream_layer("conv", 1, 1, c0, c8, wts((c8, c0, 1, 1)), *rq(c8, c0), relu=True))
    layers.append(stream_layer("dw", 3, 2, c8, c8, wts((c8, 1, 3, 3)), *rq(c8, 9),
                               dil=1))
    maps = [(8, c8)]
    n = int(rng.integers(2, 12)) if n_ops is None else n_ops
    for k in range(n):
        if rng.random() < 0.45:
            s = int(rng.integers(0, len(maps)))
            st = 2 if (rng.random() < 0.4 and maps[s][0] < 64) else 1
            d = 1 if st == 2 else int(rng.integers(1, 9))
            c = maps[s][1]
            layers.append(op_layer("dw", [s], [0], c, c, wts((c, 1, 3, 3)), *rq(c, 9), relu=bool(rng.random() < 0.5),
                                   stride=st, dil=d, out_stride=maps[s][0] * st))
            maps.append((maps[s][0] * st, c))
        else:
            s0 = int(rng.integers(0, len(maps)))
            s_op = maps[s0][0] >> int(rng.integers(0, min(3, int(np.log2(maps[s0][0] // 8))) + 1))
            cands = [i for i, (s, _) in enumerate(maps) if s >= s_op and s // s_op in (1, 2, 4, 8)]
            src = [int(rng.choice(cands)) for _ in range(int(rng.integers(1, 4)))]
            up = [int(np.log2(maps[i][0] // s_op)) for i in src]
            cin = sum(maps[i][1] for i in src)
            if cin > T3.MAX_CHANS:
                src, up, cin = src[:1], up[:1], maps[src[0]][1]
            c = int(rng.integers(1, 9)) * 8
            layers.append(op_layer("pw", src, up, cin, c, wts((c, cin, 1, 1)), *rq(c, cin),
                                   relu=bool(rng.random() < 0.7), out_stride=s_op))
            maps.append((s_op, c))
    cands = [i for i, (s, _) in enumerate(maps) if s in (8, 16, 32, 64)]
    src = [len(maps) - 1] if maps[-1][0] in (8, 16, 32, 64) else []
    src += [int(rng.choice(cands)) for _ in range(int(rng.integers(1, 4)) - len(src))]
    src = src or [int(rng.choice(cands))]
    up = [int(np.log2(maps[i][0] // 8)) for i in src]
    cin = sum(maps[i][1] for i in src)
    b = np.array([0, 0, 32, 32] * 4) + rng.integers(-8, 9, 16)
    m, sh = zip(*[M3.E.quantize_multiplier(float(20.0 / (60.0 * np.sqrt(cin) + 1) * rng.uniform(0.5, 2)))
                  for _ in range(16)])
    layers.append(op_layer("pw", src, up, cin, 16, wts((16, cin, 1, 1)), b, m, sh, head=True))
    return layers


def graph_net(seed, n_ops=None):
    return M3.pack_net({"name": f"graph_{seed}"}, graph_layers(seed, n_ops),
                       hashlib.sha256(f"graph:{seed}".encode()).digest())

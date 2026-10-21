#!/usr/bin/env python3
"""textspot3's multi-scale family M (issue #108, docs/textspot3.md "Two families"): the shape grammar, its layers,
exact costs, window masks, the float and fake-quantised torch networks, the export to VTSP version 3
(synth/textspot3_twin.py, the specification) and random version-3 blobs for the runtime's tests and A9 tables.

  textspot3_model.py --ckpt RUN/final.pt --out spot.vtsp [--net RUN/final.vtm] [--conf-min 1]
  textspot3_model.py --random SHAPE --seed N --out spot.vtsp            (a PROBES name or a JSON shape file)
  textspot3_model.py --info spot.vtsp | --cost SHAPE

The shape grammar (one dict; layers(shape) expands it into the stream layers and the map ops):

  {"name", "family": "M", "stride": 8, "c0", "c1", "c2", "chans", "dil", "block": "sep",
   "stages": [{"c": C, "n": N, "dil": [d, ...]}, ...], "neck": {"t": T, "t8": T8, "smooth": S}, "head": "ms_x2",
   "point": "S" | "M" | "L" (optional: the protocol's size point)}

  stream  train/textdet_model.layers' grammar at stride 8 without its head: stem 3x3 stride 2 1 -> c0, down sep
          (3x3 depthwise stride 2, then 1x1 c0 -> c1), down8 sep c1 -> c2, then one separable block per entry of
          chans / dil (3x3 depthwise with that dilation, then 1x1 to that many channels). Its output is F8 (map 0;
          chans[-1] channels, or c2 without blocks), streamed by rows as version 1 streams its network
  stages  for strides 16, 32, 64 (, 128) in order, each {"c", "n", "dil"} (len(dil) = n): a 3x3 depthwise stride 2
          (no ReLU) + a 1x1 C_prev -> c (ReLU), then n blocks of a 3x3 depthwise with dilation dil[i] (no ReLU) +
          a 1x1 c -> c (ReLU): outputs F16, F32, F64
  neck    top-down: P_deep = 1x1 (F_deepest -> T, ReLU); for each shallower level from the second deepest down to
          16, P_l = 1x1 over concat(F_l, up2(P_{l+1})) -> T (ReLU); then P8 = 1x1 over concat(F8, up2(P16)) -> T8
          (ReLU); then S separable smoothing blocks at stride 8 on P8 (3x3 depthwise dilation 1 + 1x1 T8 -> T8, ReLU)
  head    1x1 (P8 -> 16), no BatchNorm, no ReLU, version 2's d2s channel order (channel 4 (2 dy + dx) + m), K and O
          at 1/16 per code, A1 and A2 at 1/2 network px per code; to_stride4 gives the decode's stride-4 maps

Every map op uses version 1's integer arithmetic (int8 weights per output channel in +-127, int32 bias, the
fixed-point requantisation). The sources of one 1x1 op share one activation scale: the layers producing maps that
some 1x1 op concatenates form a group (union-find over every multi-source op), and the group's scale is the
largest of its members' calibrated scales (shared_scales; QTextSpot3 and the export apply it alike). Training
places a region top left in a padded window and multiplies every layer's output by the region's validity mask at
that layer's resolution (layer_masks), as train/textdet_model does; with the nearest upsampling read cropped to
the op's size, the window computes on the region exactly what the runtime computes on the region alone.

The checkpoint (kind "textspot3-qat", what the trainer writes): "shape" (this grammar's dict), "w" and "b" (the
BatchNorm-folded weights and biases of every layer of layers(shape), in order), "act_scale" (every layer's output
scale; the head's is not read; shared_scales is applied on export, so raw or shared scales export alike).
"""
import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_twin as TD  # noqa: E402
import textspot3_twin as T3  # noqa: E402

FAMILY, HEAD, BLOCK = "M", "ms_x2", "sep"
N_OUT = T3.OUTPUTS                                   # 16
MIN_CELLS = TM.MIN_CELLS[T3.MAP_STRIDE]              # 2: the decode reads a stride-4 map
HALF_ABOVE, HALF_ABOVE_H = 480, 272
HEAD_SCALES = tuple((1 / TM.KO_SCALE, 1 / TM.KO_SCALE, 1 / TM.A_SCALE, 1 / TM.A_SCALE)[c % 4] for c in range(N_OUT))
HEAD_TARGET_MED = (0.0, 0.0, 16.0, 16.0)             # random_int_layers' head targets per map (K, O, A1, A2)
HEAD_TARGET_SD = (24.0, 24.0, 6.0, 6.0)
SIZES = ((960, 544), (480, 272))
KEYS = ("name", "family", "stride", "c0", "c1", "c2", "chans", "dil", "block", "stages", "neck", "head")


# ----------------------------------------------------------------------------- the grammar

def m_shape(name, c0, c1, c2, chans, dil, stages, t, t8, smooth, point=None):
    """A family-M shape dict. stages: [(c, [d, ...]), ...] (n = len of the dilation list)."""
    s = {"name": name, "family": FAMILY, "stride": 8, "c0": c0, "c1": c1, "c2": c2, "chans": list(chans),
         "dil": list(dil), "block": BLOCK, "stages": [{"c": c, "n": len(d), "dil": list(d)} for c, d in stages],
         "neck": {"t": t, "t8": t8, "smooth": smooth}, "head": HEAD}
    if point:
        s["point"] = point
    return s


def check_shape(s):
    """The shape dict itself (raises ValueError): the grammar's keys, channel limits of the format."""
    missing = [k for k in KEYS if k not in s]
    if missing:
        raise ValueError(f"{s.get('name')}: missing {missing}")
    if s["family"] != FAMILY or s["stride"] != 8 or s["block"] != BLOCK or s["head"] != HEAD:
        raise ValueError(f"{s['name']}: family M, stride 8, block sep, head ms_x2")
    if len(s["chans"]) != len(s["dil"]) or not 1 <= len(s["stages"]) <= 4:
        raise ValueError(f"{s['name']}: chans / dil lengths, 1..4 stages")
    for st in s["stages"]:
        if st["n"] != len(st["dil"]) or not 8 <= st["c"] <= T3.MAX_CHANS or st["c"] % 8:
            raise ValueError(f"{s['name']}: a stage {st}")
    nk = s["neck"]
    for k in ("t", "t8"):
        if not 8 <= nk[k] <= T3.MAX_CHANS or nk[k] % 8:
            raise ValueError(f"{s['name']}: neck {k} {nk[k]}")
    if nk["smooth"] < 0:
        raise ValueError(f"{s['name']}: smooth >= 0")
    return s


def stream_layers(shape):
    """The stream: train/textdet_model.layers of the trunk (stride 8, sep), without its 1x1 head."""
    trunk = {"name": shape["name"], "stride": 8, "c0": shape["c0"], "c1": shape["c1"], "c2": shape["c2"],
             "chans": list(shape["chans"]), "dil": list(shape["dil"]), "block": BLOCK, "head": "kroa"}
    out = [dict(L, part="stream") for L in TM.layers(trunk)[:-1]]
    if out[-1]["cout"] % 8 or max(L["cout"] for L in out) > 64:
        raise ValueError(f"{shape['name']}: the stream's channels (<= 64, F8 a multiple of 8)")
    return out


def map_ops(shape):
    """The map ops of a shape, in order: dicts with kind ("dw" | "pw"), cin, cout, k, stride, dil, relu, src (map
    indices: 0 F8, k + 1 op k), up (upsampling exponents), out_stride, head, part "op", role (a label)."""
    check_shape(shape)
    c8 = stream_layers(shape)[-1]["cout"]
    maps = [(8, c8)]
    ops = []

    def add(kind, src, up, cout, relu, role, stride=1, dil=1, head=False):
        if kind == "dw":
            s_out, cin = maps[src[0]][0] * stride, maps[src[0]][1]
        else:
            s_out, cin = maps[src[0]][0] >> up[0], sum(maps[s][1] for s in src)
        ops.append({"kind": kind, "cin": cin, "cout": cout, "k": 3 if kind == "dw" else 1, "stride": stride,
                    "dil": dil, "relu": relu, "src": list(src), "up": list(up), "out_stride": s_out, "head": head,
                    "part": "op", "role": role})
        maps.append((s_out, cout))
        return len(maps) - 1

    feats, prev = [0], 0
    for li, st in enumerate(shape["stages"]):
        tag = f"s{8 << (li + 1)}"
        m = add("dw", [prev], [0], maps[prev][1], False, f"{tag} down", stride=2)
        m = add("pw", [m], [0], st["c"], True, f"{tag} down")
        for bi, d in enumerate(st["dil"]):
            m = add("dw", [m], [0], st["c"], False, f"{tag} block {bi}", dil=d)
            m = add("pw", [m], [0], st["c"], True, f"{tag} block {bi}")
        feats.append(m)
        prev = m
    nk = shape["neck"]
    p = add("pw", [feats[-1]], [0], nk["t"], True, f"neck P{maps[feats[-1]][0]}")
    for lvl in range(len(feats) - 2, 0, -1):
        p = add("pw", [feats[lvl], p], [0, 1], nk["t"], True, f"neck P{maps[feats[lvl]][0]}")
    p = add("pw", [0, p], [0, 1], nk["t8"], True, "neck P8")
    for k in range(nk["smooth"]):
        m = add("dw", [p], [0], nk["t8"], False, f"smooth {k}")
        p = add("pw", [m], [0], nk["t8"], True, f"smooth {k}")
    add("pw", [p], [0], N_OUT, False, "head", head=True)
    for op in ops:
        if op["cin"] > T3.MAX_CHANS or op["out_stride"] > T3.MAX_MAP_STRIDE:
            raise ValueError(f"{shape['name']}: {op['role']} cin {op['cin']} / stride {op['out_stride']}")
    if len(ops) > T3.MAX_OPS:
        raise ValueError(f"{shape['name']}: {len(ops)} ops > {T3.MAX_OPS}")
    return ops


def layers(shape):
    """Every layer in forward order (one per tap): the stream layers (textdet_model's dicts, part "stream"), then
    the map ops (part "op")."""
    out = stream_layers(shape) + map_ops(shape)
    for L in out:
        L.setdefault("head", False)
    return out


def maps_of(shape):
    """[(stride, channels)] per map: map 0 F8, then every op's output."""
    sl = stream_layers(shape)
    return [(8, sl[-1]["cout"])] + [(op["out_stride"], op["cout"]) for op in map_ops(shape)]


def producers(shape):
    """The layer index (into layers(shape)) that produces each map."""
    n_s = len(stream_layers(shape))
    return [n_s - 1] + [n_s + k for k in range(len(map_ops(shape)))]


def scale_groups(shape):
    """[group id per layer]: layers whose output maps one 1x1 op concatenates share a group (union-find); every
    other layer is its own group."""
    spec = layers(shape)
    parent = list(range(len(spec)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    prod = producers(shape)
    for L in spec:
        if L["part"] == "op" and L["kind"] == "pw":
            ids = [find(prod[s]) for s in L["src"]]
            for i in ids[1:]:
                a, b = find(ids[0]), find(i)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(len(spec))]


def shared_scales(shape, act_scale):
    """act_scale (one per layer; the head's may be missing) with every group's scale its members' largest."""
    spec = layers(shape)
    sc = [float(act_scale[i]) if i < len(act_scale) else 1.0 for i in range(len(spec))]
    groups = scale_groups(shape)
    best = {}
    for i, g in enumerate(groups):
        if not spec[i]["head"]:
            best[g] = max(best.get(g, 0.0), sc[i])
    return [sc[i] if spec[i]["head"] else best[groups[i]] for i in range(len(spec))]


# ----------------------------------------------------------------------------- costs

def out_size(n, s):
    return -(-n // s)


def vtsp_bytes(shape):
    """The exact VTSP version-3 size (textspot3_twin's layout: 32 + 64 + 32 per record, each section 4-aligned)."""
    def a4(n):
        return (n + 3) & ~3
    spec = layers(shape)
    total = T3.HEADER + T3.NET_HEADER + T3.RECORD * len(spec)
    for L in spec:
        per = 1 if L["kind"] == "dw" else L["cin"]
        total += a4(L["cout"] * per * L["k"] * L["k"]) + 8 * L["cout"] + a4(L["cout"])
    return total


def cost(shape, width, height):
    """Exact figures for a width x height network input: MACs (and per part: the stream, the map ops at each
    stride, the head), parameters, int8 weights and the VTSP bytes; weights per stride."""
    spec = layers(shape)
    macs = params = 0
    part, wpart = {}, {}
    w, h = width, height
    for L in spec:
        per = L["k"] * L["k"] * (1 if L["kind"] == "dw" else L["cin"])
        if L["part"] == "stream":
            ow, oh = out_size(w, L["stride"]), out_size(h, L["stride"])
            w, h = ow, oh
            key = "stream"
        else:
            ow, oh = out_size(width, L["out_stride"]), out_size(height, L["out_stride"])
            key = "head" if L["head"] else f"s{L['out_stride']}"
        n = ow * oh * L["cout"] * per
        macs += n
        part[key] = part.get(key, 0) + n
        params += L["cout"] * per + L["cout"]
        wpart[key] = wpart.get(key, 0) + L["cout"] * per
    biases = sum(L["cout"] for L in spec)
    return {"macs": int(macs), "macs_by_part": part, "params": int(params), "weights_int8": int(params - biases),
            "weights_by_part": wpart, "vtsp_bytes": vtsp_bytes(shape), "layers": len(spec),
            "ops": sum(L["part"] == "op" for L in spec)}


def net_input_size(width, height, half_above=HALF_ABOVE, half_above_h=HALF_ABOVE_H):
    """Version 3's rule: the network input of a width x height frame."""
    halve = (half_above and width > half_above) or (half_above_h and height > half_above_h)
    return ((width + 1) // 2, (height + 1) // 2) if halve else (width, height)


def receptive_field(shape):
    """The head cell's receptive field in network-input px (the largest over its paths; a nearest upsampling keeps
    its source cell's)."""
    rf, jump = 1, 1
    for L in stream_layers(shape):
        rf += (L["k"] - 1) * L["dil"] * jump
        jump *= L["stride"]
    rfs = [rf]
    for op in map_ops(shape):
        if op["kind"] == "dw":
            src = op["src"][0]
            rfs.append(rfs[src] + 2 * op["dil"] * (op["out_stride"] // op["stride"]))
        else:
            rfs.append(max(rfs[s] for s in op["src"]))
    return rfs[-1]


def costs(shape, sizes=SIZES):
    """train/textspot3_shapes.costs' figures of an M shape: MACs per frame at version 3's network input,
    parameters, int8 weights, VTSP bytes, the receptive field, the layer and op counts."""
    out = {}
    for w, h in sizes:
        iw, ih = net_input_size(w, h)
        out[f"{w}x{h}"] = cost(shape, iw, ih)["macs"]
    c = cost(shape, 480, 272)
    return {"family": FAMILY, "macs": out, "macs_by_part_480x272": c["macs_by_part"], "params": c["params"],
            "weights_int8": c["weights_int8"], "weights_by_part": c["weights_by_part"], "vtsp_bytes": c["vtsp_bytes"],
            "receptive_field": receptive_field(shape), "layers": c["layers"], "ops": c["ops"],
            "half_above": HALF_ABOVE, "half_above_h": HALF_ABOVE_H}


# ----------------------------------------------------------------------------- window masks

def layer_masks(shape, valid, height, width):
    """Per layer of layers(shape), the (N, 1, h_l, w_l) validity mask of regions valid = [(h, w)] placed top left
    in a window of height x width: the stream's as train/textdet_model.layer_masks; an op's at its stride from the
    window's and the regions' sizes (ceil(n / s): a stride-2 depthwise op's ceil(ceil(n / s) / 2) is the same)."""
    import torch
    out = []
    h, w = height, width
    vh = [v[0] for v in valid]
    vw = [v[1] for v in valid]
    for L in layers(shape):
        if L["part"] == "stream":
            h, w = out_size(h, L["stride"]), out_size(w, L["stride"])
            vh = [out_size(x, L["stride"]) for x in vh]
            vw = [out_size(x, L["stride"]) for x in vw]
            hh, ww, rh, rw = h, w, vh, vw
        else:
            s = L["out_stride"]
            hh, ww = out_size(height, s), out_size(width, s)
            rh, rw = [out_size(v[0], s) for v in valid], [out_size(v[1], s) for v in valid]
        m = torch.zeros(len(valid), 1, hh, ww)
        for i, (a, b) in enumerate(zip(rh, rw)):
            m[i, :, :a, :b] = 1
        out.append(m)
    return out


# ----------------------------------------------------------------------------- torch

def _up(m, u, rows, cols):
    """A map read through a nearest upsampling by 2^u, cropped to rows x cols (textspot3_twin.upsampled)."""
    import torch
    if u:
        m = m.index_select(2, torch.arange(rows, device=m.device) >> u)
        m = m.index_select(3, torch.arange(cols, device=m.device) >> u)
    return m[:, :, :rows, :cols]


def _op_input(L, maps, height, width):
    import torch
    if L["kind"] == "dw":
        return maps[L["src"][0]]
    rows, cols = out_size(height, L["out_stride"]), out_size(width, L["out_stride"])
    xs = [_up(maps[s], u, rows, cols) for s, u in zip(L["src"], L["up"])]
    return xs[0] if len(xs) == 1 else torch.cat(xs, 1)


def _make_torch():
    import torch.nn as nn

    class TextSpot3(nn.Module):
        """The float network with BatchNorm (training before QAT). forward(x, masks=None, taps=None) has
        train/textdet_model.TextDet's signature: x N x 1 x H x W, (g - 128) / 128; it returns the 16-channel head at
        stride 8 (to_stride4 gives the decode's maps). .body and .spec as TextDet's, so train/textdet_model.fold
        folds it (fold below)."""

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
            height, width = x.shape[2], x.shape[3]
            maps, y = [], x
            for i, (L, m) in enumerate(zip(self.spec, self.body)):
                y = m(y if L["part"] == "stream" else _op_input(L, maps, height, width))
                if masks is not None:
                    y = y * masks[i]
                if taps is not None:
                    taps.append(y)
                if L["part"] == "op" or i + 1 == len(self.spec) or self.spec[i + 1]["part"] == "op":
                    maps.append(y)
            return y

    class QTextSpot3(nn.Module):
        """The folded network with int8 fake quantisation, as train/textdet_model.QTextDet: weights per output
        channel (fq_weight), each layer's output at its scale (shared_scales: one scale per group of
        concatenated maps), the head's per channel (HEAD_SCALES)."""

        def __init__(self, shape, folded, act_scale):
            import torch
            super().__init__()
            self.shape = dict(shape)
            self.spec = layers(shape)
            if len(folded) != len(self.spec):
                raise ValueError(f"{len(folded)} folded layers for {len(self.spec)}")
            self.w = nn.ParameterList([nn.Parameter(torch.as_tensor(w).float().clone()) for w, _ in folded])
            self.b = nn.ParameterList([nn.Parameter(torch.as_tensor(b).float().clone()) for _, b in folded])
            sc = shared_scales(shape, act_scale)
            for i, L in enumerate(self.spec):
                s = torch.tensor(HEAD_SCALES).view(1, N_OUT, 1, 1) if L["head"] else \
                    torch.tensor(float(sc[i])).view(1, 1, 1, 1)
                self.register_buffer(f"s{i}", s)
            self.quant = True

        def scale(self, i):
            return getattr(self, f"s{i}")

        def forward(self, x, masks=None, taps=None):
            import torch.nn.functional as F
            height, width = x.shape[2], x.shape[3]
            maps, y = [], x
            for i, L in enumerate(self.spec):
                inp = y if L["part"] == "stream" else _op_input(L, maps, height, width)
                w = TM.fq_weight(self.w[i]) if self.quant else self.w[i]
                pad = (L["k"] - 1) // 2 * L["dil"]
                y = F.conv2d(inp, w, self.b[i], stride=L["stride"], padding=pad, dilation=L["dil"],
                             groups=L["cin"] if L["kind"] == "dw" else 1)
                if L["relu"]:
                    y = F.relu(y)
                if self.quant:
                    y = TM.fq_act(y, self.scale(i), 0 if L["relu"] else -128, 127)
                if masks is not None:
                    y = y * masks[i]
                if taps is not None:
                    taps.append(y)
                if L["part"] == "op" or i + 1 == len(self.spec) or self.spec[i + 1]["part"] == "op":
                    maps.append(y)
            return y

    return TextSpot3, QTextSpot3


def __getattr__(name):
    """TextSpot3 and QTextSpot3 load torch only when asked for."""
    if name in ("TextSpot3", "QTextSpot3"):
        a, b = _make_torch()
        globals().update(TextSpot3=a, QTextSpot3=b)
        return globals()[name]
    raise AttributeError(name)


def fold(model):
    """BatchNorm folded into each conv: [(weight, bias)] float64 tensors, one per layer (train/textdet_model.fold:
    TextSpot3 has its .body and .spec form)."""
    return TM.fold(model)


def to_stride4(y, height, width):
    """(N, 16, ceil(h/8), ceil(w/8)) head outputs of an h x w input -> (N, 4, ceil(h/4), ceil(w/4)) stride-4 maps:
    train/textspot2_export.to_stride4 (textspot2_twin.head_to_maps in torch)."""
    import textspot2_export as X
    return X.to_stride4(y, height, width)


# ----------------------------------------------------------------------------- integer layers and blobs

def quantize_layers(shape, folded, act_scale, wbits=8):
    """Integer layers from folded float weights and per-layer output scales (train/textdet_export.quantize_layers'
    arithmetic; the input at 1/128): [dict(layer, wq, bq, m, sh)]. A stream layer's input scale is the previous
    stream layer's output scale; an op's is its sources' (shared_scales makes them one)."""
    spec = layers(shape)
    if len(folded) != len(spec) or len(act_scale) not in (len(spec), len(spec) - 1):
        raise ValueError(f"{len(folded)} folded layers / {len(act_scale)} scales for {len(spec)} layers")
    sc = shared_scales(shape, act_scale)
    prod = producers(shape)
    qmax = 2 ** (wbits - 1) - 1
    out = []
    s_x = E.IN_SCALE
    for i, (L, (w, b)) in enumerate(zip(spec, folded)):
        w = np.asarray(w, np.float64)
        b = np.asarray(b, np.float64)
        cout = L["cout"]
        per = 1 if L["kind"] == "dw" else L["cin"]
        if w.shape != (cout, per, L["k"], L["k"]) or b.shape != (cout,):
            raise ValueError(f"layer {i}: weights {w.shape} / bias {b.shape} for {(cout, per, L['k'], L['k'])}")
        if L["part"] == "op":
            ins = {sc[prod[s]] for s in L["src"]}
            if len(ins) != 1:
                raise AssertionError(f"layer {i}: its sources' scales differ")
            s_in = ins.pop()
        else:
            s_in = s_x
        s_out = np.array(HEAD_SCALES) if L["head"] else np.full(cout, float(sc[i]))
        mx = np.abs(w.reshape(cout, -1)).max(1)
        s_w = np.where(mx > 0, mx / qmax, 1.0)
        wq = np.clip(np.floor(w / s_w.reshape(-1, 1, 1, 1) + 0.5), -qmax, qmax).astype(np.int8)
        bq = np.clip(np.floor(b / (s_w * s_in) + 0.5), -(2 ** 31 - 1), 2 ** 31 - 1).astype(np.int64)
        ms = [E.quantize_multiplier(float(s_w[c] * s_in / s_out[c])) for c in range(cout)]
        out.append({**L, "wq": wq, "bq": bq, "m": np.array([m for m, _ in ms], np.int64),
                    "sh": np.array([s for _, s in ms], np.int64)})
        if L["part"] == "stream":
            s_x = float(sc[i])
    return out


def pack_net(shape, qlayers, ckpt_sha256=b"\0" * 8):
    """The VTMS network section's bytes (synth/textspot3_twin.py's layout); textspot3_twin.Net checks it."""
    n_s = sum(L["part"] == "stream" for L in qlayers)
    n_o = len(qlayers) - n_s
    base = T3.NET_HEADER + T3.RECORD * len(qlayers)
    data = bytearray()
    recs = []

    def put(arr):
        while (base + len(data)) % 4:
            data.append(0)
        off = base + len(data)
        data.extend(arr.tobytes())
        return off

    for L in qlayers:
        wo = put(np.ascontiguousarray(L["wq"], np.int8))
        bo = put(np.asarray(L["bq"], "<i4"))
        mo = put(np.asarray(L["m"], "<i4"))
        so = put(np.asarray(L["sh"], np.uint8))
        if L["part"] == "stream":
            recs.append(struct.pack("<BBBBB3xHHIIIII", 1 if L["kind"] == "dw" else 0, L["k"], L["stride"], L["dil"],
                                    1 if L["relu"] else 0, L["cin"], L["cout"], wo, bo, mo, so, 0))
        else:
            src = list(L["src"]) + [0] * (T3.MAX_SOURCES - len(L["src"]))
            up = list(L["up"]) + [0] * (T3.MAX_SOURCES - len(L["up"]))
            recs.append(struct.pack("<BBBBBBH3B3BHIIII", T3.DW if L["kind"] == "dw" else T3.PW, L["stride"],
                                    L["dil"], 1 if L["relu"] else 0, len(L["src"]), 0, L["cout"], *src, *up, 0,
                                    wo, bo, mo, so))
    while len(data) % 4:
        data.append(0)
    total = base + len(data)
    head = bytearray(T3.NET_HEADER)
    head[:4] = T3.NET_MAGIC
    struct.pack_into("<HBBBBH", head, 16, n_s, T3.STREAM_STRIDE, N_OUT, TM.A_SCALE, MIN_CELLS, n_o)
    head[24:32] = ckpt_sha256[:8]
    head[32:64] = shape["name"].encode("ascii")[:31].ljust(32, b"\0")
    body = bytes(head[16:]) + b"".join(recs) + bytes(data)
    struct.pack_into("<III", head, 4, T3.NET_VERSION, total, zlib.crc32(body) & 0xFFFFFFFF)
    blob = bytes(head) + b"".join(recs) + bytes(data)
    assert len(blob) == total
    T3.Net(blob)
    return blob


def pack(net, half_above=HALF_ABOVE, conf_min=1, half_above_h=HALF_ABOVE_H):
    """The VTSP version-3 blob around a VTMS section (textspot3_twin.pack)."""
    return T3.pack(net, half_above, conf_min, half_above_h, version=T3.VERSION_MS)


def export_checkpoint(path, wbits=8):
    """A textspot3 quantisation-aware checkpoint (kind textspot3-qat; the module docstring) -> the VTMS section's
    bytes (RUN/final.vtm)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "textspot3-qat":
        raise SystemExit(f"{path}: not a textspot3 quantisation-aware checkpoint (kind textspot3-qat)")
    shp = check_shape(ck["shape"])
    folded = [(w.double(), b.double()) for w, b in zip(ck["w"], ck["b"])]
    if len(ck["w"]) != len(ck["b"]) or len(folded) != len(layers(shp)):
        raise SystemExit(f"{path}: {len(ck['w'])} layers, {len(layers(shp))} expected for {shp['name']}")
    sha = hashlib.sha256(Path(path).read_bytes()).digest()
    return pack_net(shp, quantize_layers(shp, folded, ck["act_scale"], wbits), sha)


def export_vtsp(path, conf_min=1, half_above=HALF_ABOVE, half_above_h=HALF_ABOVE_H):
    """(VTMS section bytes, VTSP version-3 bytes) of a textspot3 checkpoint (RUN/final.vtm, RUN/final.vtsp)."""
    net = export_checkpoint(path)
    return net, pack(net, half_above, conf_min, half_above_h)


def _int_layer(L, wq, bq, m, sh, relu):
    t = TD.Layer()
    t.kind, t.k, t.stride, t.dil, t.relu = (1 if L["kind"] == "dw" else 0), L["k"], L["stride"], L["dil"], relu
    t.cin, t.cout = L["cin"], L["cout"]
    t.w, t.b, t.m, t.sh = wq.astype(np.int64), np.asarray(bq, np.int64), np.asarray(m, np.int64), \
        np.asarray(sh, np.int64)
    return t


def _forward_int(L, t, maps, x, h, w):
    """One integer layer (a textdet_twin.Layer t of layer dict L) on the stream's x or the maps so far."""
    if L["part"] == "stream":
        return TD.run_layer(t, x)
    if L["kind"] == "dw":
        return TD.run_layer(t, maps[L["src"][0]])
    rows, cols = out_size(h, L["out_stride"]), out_size(w, L["out_stride"])
    xin = np.concatenate([T3.upsampled(maps[s], u, rows, cols) for s, u in zip(L["src"], L["up"])], 0)
    return TD.run_layer(t, xin)


def random_int_layers(shape, seed, n_cal=6):
    """textspot2_export.random_int_layers' idea for an M shape: random int8 weights (+-127, 30% zeros) whose
    activations spread over the int8 range on procedural regions (tests of the C runtime and the A9 table, not a
    spotter): each layer's identity-requantised accumulators set its bias and multiplier for a median of 10 (ReLU)
    or 0 and a spread of 30 codes; the head's K and O channels centred on 0, A1 and A2 on about 8 px."""
    rng = np.random.default_rng(seed)
    regions = [E.procedural_region(rng) for _ in range(n_cal)]
    xs = [g.astype(np.int64)[None] - 128 for g in regions]
    maps = [[] for _ in regions]
    out = []
    spec = layers(shape)
    for i, L in enumerate(spec):
        per = 1 if L["kind"] == "dw" else L["cin"]
        wq = rng.integers(-127, 128, (L["cout"], per, L["k"], L["k"])).astype(np.int8)
        wq[rng.random(wq.shape) < 0.3] = 0
        ident = _int_layer(L, wq, np.zeros(L["cout"]), np.full(L["cout"], 1 << 30), np.full(L["cout"], 30), 0)
        accs = []
        for r, g in enumerate(regions):
            h, w = g.shape
            a = _accumulate(L, ident, maps[r], xs[r], h, w)
            accs.append(a.reshape(L["cout"], -1))
        a = np.concatenate(accs, 1).astype(np.float64)
        med, sd = np.median(a, 1), a.std(1) + 1.0
        if L["head"]:
            target_med = np.array([HEAD_TARGET_MED[c % 4] for c in range(L["cout"])])
            target_sd = np.array([HEAD_TARGET_SD[c % 4] for c in range(L["cout"])])
        else:
            target_med = np.full(L["cout"], 10.0 if L["relu"] else 0.0)
            target_sd = np.full(L["cout"], 30.0)
        M = target_sd / sd
        ms = [E.quantize_multiplier(float(v)) for v in M]
        bq = np.floor(target_med / M - med + 0.5).astype(np.int64)
        q = {**L, "wq": wq, "bq": bq, "m": np.array([m for m, _ in ms], np.int64),
             "sh": np.array([s for _, s in ms], np.int64)}
        out.append(q)
        t = _int_layer(L, wq, bq, q["m"], q["sh"], 1 if L["relu"] else 0)
        for r, g in enumerate(regions):
            h, w = g.shape
            y = _forward_int(L, t, maps[r], xs[r], h, w)
            if L["part"] == "stream":
                xs[r] = y
                if i + 1 == len(spec) or spec[i + 1]["part"] == "op":
                    maps[r].append(y)
            else:
                maps[r].append(y)
    return out


def _accumulate(L, ident, maps, x, h, w):
    """The accumulator (bias 0, before requantisation) of layer L on a region (identity requantisation: the
    multiplier 2^30 and shift 30 return it exactly, unclamped)."""
    t = ident
    if L["part"] == "stream":
        return E._accumulate(t, x)
    if L["kind"] == "dw":
        return E._accumulate(t, maps[L["src"][0]])
    rows, cols = out_size(h, L["out_stride"]), out_size(w, L["out_stride"])
    xin = np.concatenate([T3.upsampled(maps[s], u, rows, cols) for s, u in zip(L["src"], L["up"])], 0)
    return E._accumulate(t, xin)


def random_net(shape, seed, n_cal=6):
    return pack_net(shape, random_int_layers(shape, seed, n_cal),
                    hashlib.sha256(f"random:{shape['name']}:{seed}".encode()).digest())


def random_blob(shape, seed, conf_min=1, n_cal=6, half_above=HALF_ABOVE, half_above_h=HALF_ABOVE_H):
    """(a random VTSP version-3 blob of an M shape (a dict or a PROBES name), how it was made)."""
    shp = shape_of(shape)
    blob = pack(random_net(shp, seed, n_cal), half_above, conf_min, half_above_h)
    return blob, (f"textspot3_model.random_int_layers seed {seed}, textspot3_model.pack(half_above {half_above}, "
                  f"conf_min {conf_min}, half_above_h {half_above_h})")


# ----------------------------------------------------------------------------- probe shapes (not registered)

def _probes():
    """The PROBE_ROWS grid (probe shapes for the A9 rows, not registered: the conductor registers S, M and L by an
    amendment): around 100, 250 and <= 500 KB of VTSP, each size point from compute-light (most weights at stride
    64) to compute-heavy (wide stride-16 stages); stream c2 48 or 64 with 2 or 3 stride-8 blocks (dilations 1, 2,
    4), neck t 48-128, t8 32-48, one smoothing block."""
    out = {}
    grid = (("pS1", 48, 2, [(48, []), (96, [1]), (192, [])], 48, 32, "S"),
            ("pS2", 48, 2, [(64, [1]), (96, [1]), (128, [1])], 64, 48, "S"),
            ("pS3", 64, 2, [(64, [1]), (128, []), (192, [])], 64, 48, "S"),
            ("pS4", 64, 2, [(96, [1]), (96, []), (128, [])], 64, 48, "S"),
            ("pM1", 48, 2, [(64, []), (192, []), (320, [1])], 64, 32, "M"),
            ("pM2", 48, 2, [(96, [1]), (192, [1]), (256, [1])], 96, 48, "M"),
            ("pM3", 64, 2, [(128, [1, 2]), (192, [1]), (256, [])], 96, 48, "M"),
            ("pM4", 64, 2, [(128, [1, 2]), (128, [1, 2]), (192, [])], 128, 48, "M"),
            ("pL1", 48, 2, [(64, []), (192, []), (512, [1])], 48, 32, "L"),
            ("pL2", 48, 2, [(96, [1]), (192, [1]), (448, [1])], 96, 48, "L"),
            ("pL3", 64, 2, [(128, [1, 2]), (256, [1, 2]), (320, [])], 128, 48, "L"),
            ("pL4", 64, 2, [(160, [1, 2]), (256, [1, 2]), (256, [])], 128, 48, "L"),
            ("pL5", 48, 3, [(160, []), (192, []), (448, [1])], 96, 48, "L"))
    for name, c2, nb, stages, t, t8, point in grid:
        out[name] = m_shape(name, 8, 16, c2, [c2] * nb, [1, 2, 4][:nb], stages, t, t8, 1, point)
    return out


PROBES = _probes()


def shape_of(s):
    """A shape dict from a dict, a PROBES name, train/textspot3_shapes' registered M names, or a JSON file."""
    if isinstance(s, dict):
        return check_shape(s)
    if s in PROBES:
        return PROBES[s]
    import textspot3_shapes as S3
    if s in S3.SHAPES and S3.SHAPES[s]["family"] == FAMILY:
        return check_shape(S3.SHAPES[s])
    p = Path(s)
    if p.suffix == ".json" and p.exists():
        return check_shape(json.loads(p.read_text(encoding="utf-8")))
    raise KeyError(f"{s}: not an M shape")


# ----------------------------------------------------------------------------- CLI

def info(path):
    blob = Path(path).read_bytes()
    m = T3.Model(blob)
    print(f"{path}: VTSP version {m.version} ({m.head}), {m.blob_bytes} bytes, half_above {m.half_above} / "
          f"{m.half_above_h}, conf_min {m.conf_min}, {m.name}, min cells {m.min_cells}")
    if m.version == T3.VERSION_MS:
        n = m.net
        print(f"  stream {len(n.stream)} layers -> F8 {n.map_chans[0]} ch; {len(n.ops)} ops; "
              f"maps {list(zip(n.map_stride, n.map_chans))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--random")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--net", help="also write the VTMS section here (--ckpt)")
    ap.add_argument("--conf-min", type=int, default=1)
    ap.add_argument("--info")
    ap.add_argument("--cost")
    args = ap.parse_args(argv)
    if args.info:
        info(args.info)
        return 0
    if args.cost:
        s = shape_of(args.cost)
        print(json.dumps({"shape": s, **costs(s)}, indent=1))
        return 0
    if args.ckpt:
        net, blob = export_vtsp(args.ckpt, args.conf_min)
        if args.net:
            Path(args.net).write_bytes(net)
    elif args.random:
        blob, _ = random_blob(args.random, args.seed, args.conf_min)
    else:
        ap.error("--ckpt, --random, --info or --cost")
    if not args.out:
        ap.error("--out")
    Path(args.out).write_bytes(blob)
    print(f"{args.out}: {len(blob)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

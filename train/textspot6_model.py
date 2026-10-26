#!/usr/bin/env python3
"""textspot6's tagged spotters (issue #128, docs/textspot6.md): the networks of textspot3's two families with a
tagged d2s head, their fake-quantised torch forms, the export to VTSP version 4 (synth/textspot6_twin.py, the
specification) and random version-4 blobs for the runtime's tests and A9 tables.

  textspot6_model.py --ckpt RUN/final.pt --out spot.vtsp [--net RUN/final.vtn]
  textspot6_model.py --random SHAPE --family T|M --seed N --out spot.vtsp [--tags 4] [--half-above 480]
  textspot6_model.py --info spot.vtsp | --cost SHAPE --family T|M

A tagged network is textspot3's network with one change: the head (the last 1x1 layer) has 16 + 4 T channels.
Channels 0..15 are version 2's d2s head (K, O, A1, A2 per sub-cell, train/textspot2_export.to_stride4's order);
channel 16 + T (2 dy + dx) + t is tag map t of the sub-cell (dy, dx) (textspot6_twin.tag_channel). The exporter
writes K, O and the tag maps at 1/16 per code and A1, A2 at 1/2 network-input px per code (head_scales).

  family T  train/textspot2_export's d2s trunk (train/textdet_model.layers' grammar at stride 8; at most 64 channels)
            -> a VTDT network with outputs 16 + 4 T (textspot6_twin.TrunkNet)
  family M  train/textspot3_model's multi-scale grammar (stream, stages, neck) -> a VTMS section with outputs
            16 + 4 T (16 + 4 T a multiple of 8, so T = 2 or 4)

The scale (VTSP header): half_above and half_above_h as the shape says (T: half_above, default 480, half_above_h 0;
M: 480 and 272). The checkpoint (kind "textspot6-qat", what train/textspot6_train.py writes): "shape" (with
"family" T or M), "tags", "w" and "b" (the BatchNorm-folded weights and biases of every layer of layers(shape,
tags)), "act_scale" (every layer's output scale; the head's is not read).
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
import textspot2_export as X  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_twin as T3  # noqa: E402
import textspot6_twin as T6  # noqa: E402

KIND = "textspot6-qat"
TAGS = 4                                  # the classes of textspot6_twin.CLASSES: dialogue, name, choice, ui, scene
FAMILIES = ("T", "M")
HALF_ABOVE_T, HALF_ABOVE_M, HALF_ABOVE_H_M = 480, 480, 272
MIN_CELLS = TM.MIN_CELLS[T6.MAP_STRIDE]   # 2
T_MAX_CHANNELS = 64
# random blobs: the head's accumulator targets per map (K, O, A1, A2, then every tag map like K)
HEAD_TARGET_MED = {"K": 0.0, "O": 0.0, "A1": 16.0, "A2": 16.0, "T": 0.0}
HEAD_TARGET_SD = {"K": 24.0, "O": 24.0, "A1": 6.0, "A2": 6.0, "T": 24.0}


def outputs(tags):
    return T6.outputs_of(tags)


def channel_role(c):
    """'K', 'O', 'A1', 'A2' for channels 0..15 (c % 4), 'T' for a tag channel."""
    return ("K", "O", "A1", "A2")[c % 4] if c < T6.BASE_OUTPUTS else "T"


def head_scales(tags):
    """The head's output scale per channel: K, O and tags 1/16 per code, A1 and A2 1/2 px per code."""
    return tuple({"K": 1 / TM.KO_SCALE, "O": 1 / TM.KO_SCALE, "A1": 1 / TM.A_SCALE, "A2": 1 / TM.A_SCALE,
                  "T": 1 / TM.KO_SCALE}[channel_role(c)] for c in range(outputs(tags)))


# ----------------------------------------------------------------------------- shapes and layers

def family_of(shape):
    fam = shape.get("family")
    if fam in FAMILIES:
        return fam
    if shape.get("head") == X.HEAD:
        return "T"
    if shape.get("head") == M3.HEAD:
        return "M"
    raise ValueError(f"{shape.get('name')}: neither a T (head {X.HEAD}) nor an M (head {M3.HEAD}) shape")


def check(shape, tags):
    """The shape and tags as a tagged network takes them (raises ValueError)."""
    fam = family_of(shape)
    if not 1 <= tags <= T6.MAX_TAGS:
        raise ValueError(f"tags {tags}: 1..{T6.MAX_TAGS}")
    if fam == "T":
        s = {k: shape[k] for k in ("name", "stride", "c0", "c1", "c2", "chans", "dil", "block", "head")}
        if s["stride"] != 8 or s["head"] != X.HEAD or s["block"] != "sep":
            raise ValueError(f"{s['name']}: a T shape is a separable stride-8 trunk with head {X.HEAD!r}")
        widest = max(s["c0"], s["c1"], s["c2"] or 0, *s["chans"])
        if widest > T_MAX_CHANNELS:
            raise ValueError(f"{s['name']}: {widest} channels > {T_MAX_CHANNELS}")
    else:
        M3.check_shape(shape)
        if outputs(tags) % 8:
            raise ValueError(f"family M: 16 + 4 x {tags} = {outputs(tags)} is not a multiple of 8")
    return fam


def layers(shape, tags=TAGS):
    """Every layer in forward order: textspot2_export.layers (T) or textspot3_model.layers (M), the head with
    16 + 4 tags channels."""
    fam = check(shape, tags)
    out = X.layers(dict(shape, head=X.HEAD)) if fam == "T" else M3.layers(shape)
    if not out[-1]["head"] or out[-1]["k"] != 1:
        raise AssertionError("the networks end in a 1x1 head")
    out = [dict(L) for L in out]
    out[-1]["cout"] = outputs(tags)
    return out


def scales_of(shape):
    """(half_above, half_above_h) of a shape's VTSP header."""
    if family_of(shape) == "T":
        return int(shape.get("half_above", HALF_ABOVE_T)), int(shape.get("half_above_h", 0))
    return int(shape.get("half_above", HALF_ABOVE_M)), int(shape.get("half_above_h", HALF_ABOVE_H_M))


def net_input_size(shape, width, height):
    ha, hh = scales_of(shape)
    if (ha and width > ha) or (hh and height > hh):
        return (width + 1) // 2, (height + 1) // 2
    return width, height


def cost(shape, width, height, tags=TAGS):
    """Exact MACs, parameters and int8 weights of the network on a width x height network input."""
    fam = family_of(shape)
    if fam == "T":
        macs = params = 0
        w, h = width, height
        for L in layers(shape, tags):
            ow, oh = TM.out_size(w, L["stride"]), TM.out_size(h, L["stride"])
            per = L["k"] * L["k"] * (1 if L["kind"] == "dw" else L["cin"])
            macs += ow * oh * L["cout"] * per
            params += L["cout"] * per + L["cout"]
            w, h = ow, oh
    else:
        macs = params = 0
        w, h = width, height
        for L in layers(shape, tags):
            if L["part"] == "stream":
                ow, oh = TM.out_size(w, L["stride"]), TM.out_size(h, L["stride"])
                w, h = ow, oh
            else:
                ow, oh = M3.out_size(width, L["out_stride"]), M3.out_size(height, L["out_stride"])
            per = L["k"] * L["k"] * (1 if L["kind"] == "dw" else L["cin"])
            macs += ow * oh * L["cout"] * per
            params += L["cout"] * per + L["cout"]
    biases = sum(L["cout"] for L in layers(shape, tags))
    return {"macs": int(macs), "params": int(params), "weights_int8": int(params - biases)}


def layer_masks(shape, valid, height, width):
    """The layers' validity masks (the head's channels do not change them)."""
    return TM.layer_masks(dict(shape, head=X.HEAD), valid, height, width) if family_of(shape) == "T" else \
        M3.layer_masks(shape, valid, height, width)


# ----------------------------------------------------------------------------- torch

def to_maps(y, height, width, tags=TAGS):
    """(N, 16 + 4 T, ceil(h/8), ceil(w/8)) head outputs of an h x w input -> ((N, 4, h4, w4) K, O, A1, A2 maps,
    (N, T, h4, w4) tag maps) at stride 4: textspot6_twin.head_to_maps in torch, differentiable."""
    n, c, hn, wn = y.shape
    if c != outputs(tags):
        raise ValueError(f"head {tuple(y.shape)}: {outputs(tags)} channels expected")
    base = X.to_stride4(y[:, :T6.BASE_OUTPUTS], height, width)
    t = y[:, T6.BASE_OUTPUTS:].reshape(n, 2, 2, tags, hn, wn).permute(0, 3, 4, 1, 5, 2).reshape(n, tags, 2 * hn,
                                                                                                2 * wn)
    return base, t[:, :, :base.shape[2], :base.shape[3]]


def _make_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TextSpot6T(X.TextDetD2S):
        """Family T's float network: textspot2_export.TextDetD2S with a 16 + 4 T channel head."""

        def __init__(self, shape, tags=TAGS):
            super().__init__(dict(shape, head=X.HEAD))
            self.tags = tags
            self.spec = layers(shape, tags)
            L = self.spec[-1]
            self.body[-1] = nn.Sequential(nn.Conv2d(L["cin"], L["cout"], 1, bias=True))

    class QTextSpot6T(TM.QTextDet):
        """Family T's folded network with int8 fake quantisation: textspot2_export.QTextDetD2S with the tagged
        head's scales."""

        def __init__(self, shape, folded, act_scale, tags=TAGS):
            nn.Module.__init__(self)
            self.shape = dict(shape, head=X.HEAD)
            self.tags = tags
            self.spec = layers(shape, tags)
            self.w = nn.ParameterList([nn.Parameter(torch.as_tensor(w).float().clone()) for w, _ in folded])
            self.b = nn.ParameterList([nn.Parameter(torch.as_tensor(b).float().clone()) for _, b in folded])
            for i, L in enumerate(self.spec):
                s = torch.tensor(head_scales(tags)).view(1, outputs(tags), 1, 1) if L["head"] else \
                    torch.tensor(float(act_scale[i])).view(1, 1, 1, 1)
                self.register_buffer(f"s{i}", s)
            self.quant = True

    TS3, _ = M3._make_torch()

    class TextSpot6M(TS3):
        """Family M's float network: textspot3_model.TextSpot3 with a 16 + 4 T channel head."""

        def __init__(self, shape, tags=TAGS):
            super().__init__(shape)
            self.tags = tags
            self.spec = layers(shape, tags)
            L = self.spec[-1]
            self.body[-1] = nn.Sequential(nn.Conv2d(L["cin"], L["cout"], 1, bias=True))

    class QTextSpot6M(nn.Module):
        """Family M's folded network with int8 fake quantisation: textspot3_model.QTextSpot3 with the tagged
        head's scales (shared_scales for the concatenated maps, as there)."""

        def __init__(self, shape, folded, act_scale, tags=TAGS):
            super().__init__()
            self.shape = dict(shape)
            self.tags = tags
            self.spec = layers(shape, tags)
            if len(folded) != len(self.spec):
                raise ValueError(f"{len(folded)} folded layers for {len(self.spec)}")
            self.w = nn.ParameterList([nn.Parameter(torch.as_tensor(w).float().clone()) for w, _ in folded])
            self.b = nn.ParameterList([nn.Parameter(torch.as_tensor(b).float().clone()) for _, b in folded])
            sc = M3.shared_scales(shape, act_scale)
            for i, L in enumerate(self.spec):
                s = torch.tensor(head_scales(tags)).view(1, outputs(tags), 1, 1) if L["head"] else \
                    torch.tensor(float(sc[i])).view(1, 1, 1, 1)
                self.register_buffer(f"s{i}", s)
            self.quant = True

        def scale(self, i):
            return getattr(self, f"s{i}")

        def forward(self, x, masks=None, taps=None):
            height, width = x.shape[2], x.shape[3]
            maps, y = [], x
            for i, L in enumerate(self.spec):
                inp = y if L["part"] == "stream" else M3._op_input(L, maps, height, width)
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

    return TextSpot6T, QTextSpot6T, TextSpot6M, QTextSpot6M


_TORCH = {}


def torch_classes():
    if not _TORCH:
        a, b, c, d = _make_torch()
        _TORCH.update(TextSpot6T=a, QTextSpot6T=b, TextSpot6M=c, QTextSpot6M=d)
    return _TORCH


def float_model(shape, tags=TAGS):
    k = "TextSpot6T" if family_of(shape) == "T" else "TextSpot6M"
    return torch_classes()[k](shape, tags)


def q_model(shape, folded, act_scale, tags=TAGS):
    k = "QTextSpot6T" if family_of(shape) == "T" else "QTextSpot6M"
    return torch_classes()[k](shape, folded, act_scale, tags)


def fold(model):
    return TM.fold(model)


# ----------------------------------------------------------------------------- integer layers and blobs

def quantize_layers(shape, folded, act_scale, tags=TAGS, wbits=8):
    """Integer layers from folded float weights and per-layer output scales (textspot2_export.quantize_layers'
    arithmetic for T, textspot3_model.quantize_layers' for M), the head's scales head_scales(tags)."""
    fam = family_of(shape)
    spec = layers(shape, tags)
    if len(folded) != len(spec) or len(act_scale) not in (len(spec), len(spec) - 1):
        raise ValueError(f"{len(folded)} folded layers / {len(act_scale)} scales for {len(spec)} layers")
    sc = list(act_scale) if fam == "T" else M3.shared_scales(shape, act_scale)
    prod = M3.producers(shape) if fam == "M" else None
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
        if fam == "M" and L["part"] == "op":
            ins = {sc[prod[s]] for s in L["src"]}
            if len(ins) != 1:
                raise AssertionError(f"layer {i}: its sources' scales differ")
            s_in = ins.pop()
        else:
            s_in = s_x
        s_out = np.array(head_scales(tags)) if L["head"] else np.full(cout, float(sc[i]))
        mx = np.abs(w.reshape(cout, -1)).max(1)
        s_w = np.where(mx > 0, mx / qmax, 1.0)
        wq = np.clip(np.floor(w / s_w.reshape(-1, 1, 1, 1) + 0.5), -qmax, qmax).astype(np.int8)
        bq = np.clip(np.floor(b / (s_w * s_in) + 0.5), -(2 ** 31 - 1), 2 ** 31 - 1).astype(np.int64)
        ms = [E.quantize_multiplier(float(s_w[c] * s_in / s_out[c])) for c in range(cout)]
        out.append({**L, "wq": wq, "bq": bq, "m": np.array([m for m, _ in ms], np.int64),
                    "sh": np.array([s for _, s in ms], np.int64)})
        if fam == "T" or L["part"] == "stream":
            if not L["head"]:
                s_x = float(sc[i])
    return out


def _put(base, data, arr):
    while (base + len(data)) % 4:
        data.append(0)
    off = base + len(data)
    data.extend(arr.tobytes())
    return off


def pack_trunk(shape, qlayers, tags=TAGS, ckpt_sha256=b"\0" * 8):
    """A family-T network's VTDT bytes (textspot2_export.pack's layout) with outputs 16 + 4 tags."""
    n = len(qlayers)
    data = bytearray()
    base = TD.HEADER + n * TD.RECORD
    recs = []
    for L in qlayers:
        wo = _put(base, data, np.ascontiguousarray(L["wq"], np.int8))
        bo = _put(base, data, np.asarray(L["bq"], "<i4"))
        mo = _put(base, data, np.asarray(L["m"], "<i4"))
        so = _put(base, data, np.asarray(L["sh"], np.uint8))
        recs.append(struct.pack("<BBBBB3xHHIIIII", 1 if L["kind"] == "dw" else 0, L["k"], L["stride"], L["dil"],
                                1 if L["relu"] else 0, L["cin"], L["cout"], wo, bo, mo, so, 0))
    while len(data) % 4:
        data.append(0)
    total = base + len(data)
    head = bytearray(TD.HEADER)
    head[:4] = TD.MAGIC
    struct.pack_into("<HBBBBH", head, 16, n, 8, outputs(tags), TM.A_SCALE, MIN_CELLS, 0)
    head[24:32] = ckpt_sha256[:8]
    head[32:64] = shape["name"].encode("ascii")[:31].ljust(32, b"\0")
    body = bytes(head[16:]) + b"".join(recs) + bytes(data)
    struct.pack_into("<III", head, 4, TD.VERSION, total, zlib.crc32(body) & 0xFFFFFFFF)
    blob = bytes(head) + b"".join(recs) + bytes(data)
    assert len(blob) == total
    T6.TrunkNet(blob, tags)
    return blob


def pack_ms(shape, qlayers, tags=TAGS, ckpt_sha256=b"\0" * 8):
    """A family-M network's VTMS section (textspot3_model.pack_net's layout) with outputs 16 + 4 tags."""
    n_s = sum(L["part"] == "stream" for L in qlayers)
    n_o = len(qlayers) - n_s
    base = T3.NET_HEADER + T3.RECORD * len(qlayers)
    data = bytearray()
    recs = []
    for L in qlayers:
        wo = _put(base, data, np.ascontiguousarray(L["wq"], np.int8))
        bo = _put(base, data, np.asarray(L["bq"], "<i4"))
        mo = _put(base, data, np.asarray(L["m"], "<i4"))
        so = _put(base, data, np.asarray(L["sh"], np.uint8))
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
    struct.pack_into("<HBBBBH", head, 16, n_s, T3.STREAM_STRIDE, outputs(tags), TM.A_SCALE, MIN_CELLS, n_o)
    head[24:32] = ckpt_sha256[:8]
    head[32:64] = shape["name"].encode("ascii")[:31].ljust(32, b"\0")
    body = bytes(head[16:]) + b"".join(recs) + bytes(data)
    struct.pack_into("<III", head, 4, T3.NET_VERSION, total, zlib.crc32(body) & 0xFFFFFFFF)
    blob = bytes(head) + b"".join(recs) + bytes(data)
    assert len(blob) == total
    T6._ms_net(blob, tags)
    return blob


def pack_net(shape, qlayers, tags=TAGS, ckpt_sha256=b"\0" * 8):
    if family_of(shape) == "T":
        return pack_trunk(shape, qlayers, tags, ckpt_sha256)
    return pack_ms(shape, qlayers, tags, ckpt_sha256)


def pack(shape, net, tags=TAGS, conf_min=1):
    """The VTSP version-4 blob of a shape's network (its scales from the shape)."""
    ha, hh = scales_of(shape)
    return T6.pack(net, tags, ha, conf_min, hh)


def export_checkpoint(path, wbits=8):
    """A textspot6 quantisation-aware checkpoint -> (the network's bytes, the VTSP version-4 blob)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != KIND:
        raise SystemExit(f"{path}: not a textspot6 quantisation-aware checkpoint (kind {KIND})")
    shp, tags = ck["shape"], int(ck["tags"])
    folded = [(w.double(), b.double()) for w, b in zip(ck["w"], ck["b"])]
    if len(ck["w"]) != len(ck["b"]) or len(folded) != len(layers(shp, tags)):
        raise SystemExit(f"{path}: {len(ck['w'])} layers, {len(layers(shp, tags))} expected for {shp['name']}")
    sha = hashlib.sha256(Path(path).read_bytes()).digest()
    net = pack_net(shp, quantize_layers(shp, folded, ck["act_scale"], tags, wbits), tags, sha)
    return net, pack(shp, net, tags)


# ----------------------------------------------------------------------------- random blobs (runtime tests, A9)

def _int_layer(L, wq, bq, m, sh, relu):
    t = TD.Layer()
    t.kind, t.k, t.stride, t.dil, t.relu = (1 if L["kind"] == "dw" else 0), L["k"], L["stride"], L["dil"], relu
    t.cin, t.cout = L["cin"], L["cout"]
    t.w, t.b, t.m, t.sh = wq.astype(np.int64), np.asarray(bq, np.int64), np.asarray(m, np.int64), \
        np.asarray(sh, np.int64)
    return t


def random_int_layers(shape, seed, tags=TAGS, n_cal=6):
    """Random int8 layers whose activations spread over the int8 range on procedural regions (tests of the C
    runtime, not a spotter): textspot2_export.random_int_layers' and textspot3_model.random_int_layers' rule, the
    head's K, O and tag channels centred on 0, A1 and A2 on about 8 px."""
    fam = family_of(shape)
    rng = np.random.default_rng(seed)
    regions = [E.procedural_region(rng) for _ in range(n_cal)]
    xs = [g.astype(np.int64)[None] - 128 for g in regions]
    maps = [[] for _ in regions]
    out = []
    spec = layers(shape, tags)
    for i, L in enumerate(spec):
        per = 1 if L["kind"] == "dw" else L["cin"]
        wq = rng.integers(-127, 128, (L["cout"], per, L["k"], L["k"])).astype(np.int8)
        wq[rng.random(wq.shape) < 0.3] = 0
        ident = _int_layer(L, wq, np.zeros(L["cout"]), np.full(L["cout"], 1 << 30), np.full(L["cout"], 30), 0)
        accs = []
        for r, g in enumerate(regions):
            h, w = g.shape
            if fam == "T" or L.get("part") == "stream":
                a = E._accumulate(ident, xs[r])
            else:
                a = M3._accumulate(L, ident, maps[r], xs[r], h, w)
            accs.append(a.reshape(L["cout"], -1))
        a = np.concatenate(accs, 1).astype(np.float64)
        med, sd = np.median(a, 1), a.std(1) + 1.0
        if L["head"]:
            target_med = np.array([HEAD_TARGET_MED[channel_role(c)] for c in range(L["cout"])])
            target_sd = np.array([HEAD_TARGET_SD[channel_role(c)] for c in range(L["cout"])])
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
            if fam == "T":
                xs[r] = TD.run_layer(t, xs[r])
                continue
            y = M3._forward_int(L, t, maps[r], xs[r], h, w)
            if L["part"] == "stream":
                xs[r] = y
                if i + 1 == len(spec) or spec[i + 1]["part"] == "op":
                    maps[r].append(y)
            else:
                maps[r].append(y)
    return out


def random_blob(shape, seed, tags=TAGS, conf_min=1, n_cal=6):
    """(a random VTSP version-4 blob of a shape, how it was made)."""
    net = pack_net(shape, random_int_layers(shape, seed, tags, n_cal), tags,
                   hashlib.sha256(f"random:{shape['name']}:{seed}:{tags}".encode()).digest())
    ha, hh = scales_of(shape)
    return pack(shape, net, tags, conf_min), (f"textspot6_model.random_int_layers seed {seed}, tags {tags}, "
                                              f"textspot6_twin.pack(half_above {ha}, half_above_h {hh}, "
                                              f"conf_min {conf_min})")


# ----------------------------------------------------------------------------- shapes by name

def shape_by_name(name, family):
    """A registered textspot6 shape (train/textspot6_shapes.py), else textspot3's T or M shapes and probes."""
    try:
        import textspot6_shapes as S6
        if name in S6.SHAPES:
            return dict(S6.SHAPES[name])
    except ImportError:
        pass
    if family == "T":
        import textspot3_shapes as S3
        if name in S3.SHAPES:
            return dict(S3.SHAPES[name])
        return X.shape(name)
    return M3.shape_of(name)


def info(path):
    blob = Path(path).read_bytes()
    m = T6.Model(blob)
    print(f"{path}: VTSP version {m.version} ({m.head}, tags {m.tags}, form {m.form}), {m.blob_bytes} bytes, "
          f"half_above {m.half_above}, half_above_h {m.half_above_h}, conf_min {m.conf_min}; network {m.name}, "
          f"min cells {m.min_cells}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--random")
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--shape-json", help="the shape as a JSON object (instead of a name)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tags", type=int, default=TAGS)
    ap.add_argument("--out")
    ap.add_argument("--net", help="also write the network's bytes here (--ckpt)")
    ap.add_argument("--info")
    ap.add_argument("--cost")
    args = ap.parse_args(argv)
    if args.info:
        info(args.info)
        return 0
    if args.cost:
        shp = json.loads(args.shape_json) if args.shape_json else shape_by_name(args.cost, args.family)
        for w, h in ((960, 544), (480, 272)):
            iw, ih = net_input_size(shp, w, h)
            print(f"{w}x{h} (net {iw}x{ih}): {cost(shp, iw, ih, args.tags)}")
        return 0
    if args.ckpt:
        net, blob = export_checkpoint(args.ckpt)
        if args.net:
            Path(args.net).write_bytes(net)
    elif args.random:
        shp = json.loads(args.shape_json) if args.shape_json else shape_by_name(args.random, args.family)
        blob, _ = random_blob(shp, args.seed, args.tags)
    else:
        ap.error("--ckpt, --random, --cost or --info")
    if not args.out:
        ap.error("--out")
    Path(args.out).write_bytes(blob)
    print(f"{args.out}: {len(blob)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

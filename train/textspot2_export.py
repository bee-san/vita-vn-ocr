#!/usr/bin/env python3
"""Export textspot2's d2s spotters (issue #106, docs/textspot2.md "d2s"): a stride-8 trunk with a 2 x 2 sub-pixel
head of 16 channels, to the VTSP version 2 blob that synth/textspot2_twin.py specifies and runtime/vocr_textspot2.c
runs; and random version-2 blobs for the runtime's tests and the A9 table.

  textspot2_export.py --ckpt RUN/final.pt --out spot.vtsp [--vtd RUN/final.vtd] [--half-above 600] [--conf-min 1]
  textspot2_export.py --random SHAPE --seed N --out spot.vtsp [--half-above 600]
  textspot2_export.py --info spot.vtsp

The network (layers()): train/textdet_model.layers' grammar for the trunk, unchanged (stem 3x3 stride 2, down,
down8, the blocks), then a 1x1 head to 16 channels: channel 4 (2 dy + dx) + m is map m (0 K, 1 O, 2 A1, 3 A2) of the
sub-cell (dy, dx) of a stride-8 cell (textspot2_twin.channel), so the decode reads the four maps at stride 4
(textspot2_twin.head_to_maps; to_stride4 is the same rearrangement in torch, for training against stride-4
targets).

The checkpoint is the trainer's quantisation-aware one, as train/textdet_train.py writes it: kind "textdet-qat",
"shape" a "kroa_x2" shape of train/textspot2_shapes.py, "w" and "b" the BatchNorm-folded weights and biases of
every layer of layers(shape), "act_scale" every layer's output scale (the head's entry is not read). The integer
layers are train/textdet_export.quantize_layers' arithmetic (int8 weights per output channel, int32 biases in the
accumulator's units, a fixed-point multiplier and shift per channel to the next scale; the input at 1/128) with
the head's output scales fixed per channel: K and O at 1/16 per code, A1 and A2 at 1/2 network-input px per code
(HEAD_SCALES), as #97's 4-map head. The VTDT header (train/textdet_export.pack's layout) says stride 8, outputs 16,
2 A codes per px and min cells 2 (a stride-4 map's); the VTSP wrapper is version 2 (textspot2_twin.pack).
TextDetD2S and QTextDetD2S are train/textdet_model's TextDet and QTextDet with this head; train/textdet_model.fold
and layer_masks work on them unchanged.
"""
import argparse
import hashlib
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_twin as TW  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_twin as T2  # noqa: E402

HEAD = "kroa_x2"
N_OUT = T2.D2S_OUTPUTS                       # 16
STRIDE = T2.D2S_NET_STRIDE                   # 8: the trunk's
MIN_CELLS = TM.MIN_CELLS[T2.D2S_MAP_STRIDE]  # 2: the decode reads a stride-4 map
HEAD_SCALES = tuple((1 / TM.KO_SCALE, 1 / TM.KO_SCALE, 1 / TM.A_SCALE, 1 / TM.A_SCALE)[c % 4] for c in range(N_OUT))
# random_int_layers' targets for the head's accumulators, per map (K, O, A1, A2), as textdet_export's
HEAD_TARGET_MED = (0.0, 0.0, 16.0, 16.0)
HEAD_TARGET_SD = (24.0, 24.0, 6.0, 6.0)


def is_d2s(shape):
    return isinstance(shape, dict) and shape.get("head") == HEAD


def shape(name):
    """The d2s shape of the protocol (train/textspot2_shapes.SHAPES) as a plain dict."""
    s = S2.SHAPES.get(name)
    if s is None or s["head"] != HEAD:
        raise ValueError(f"{name}: not a d2s shape of eval/textspot2_protocol.json")
    return {"name": name, **{k: (list(s[k]) if isinstance(s[k], list) else s[k]) for k in S2.FIELDS}}


def layers(shape):
    """The layer list (train/textdet_model.layers' dicts) of a d2s shape: the trunk as textdet_model builds it, the
    head 1x1 to 16 channels."""
    if not is_d2s(shape) or shape["stride"] != STRIDE:
        raise ValueError(f"{shape.get('name')}: a d2s shape is a stride-8 trunk with head {HEAD!r}")
    out = TM.layers(shape)
    if not out[-1]["head"] or out[-1]["k"] != 1:
        raise AssertionError("train/textdet_model.layers ends in a 1x1 head")
    out[-1] = dict(out[-1], cout=N_OUT)
    return out


def cost(shape, width, height):
    """train/textdet_model.cost's figures (exact MACs, parameters, int8 weights, row-buffer bytes, output size) for
    a d2s shape on a width x height network input."""
    macs = params = arena = 0
    w, h = width, height
    for L in layers(shape):
        ow, oh = TM.out_size(w, L["stride"]), TM.out_size(h, L["stride"])
        per = L["k"] * L["k"] * (1 if L["kind"] == "dw" else L["cin"])
        macs += ow * oh * L["cout"] * per
        params += L["cout"] * per + L["cout"]
        arena += ((L["k"] - 1) * L["dil"] + 1) * w * L["cin"]
        w, h = ow, oh
    biases = sum(L["cout"] for L in layers(shape))
    return {"macs": int(macs), "params": int(params), "weights_int8": int(params - biases), "arena_rows": int(arena),
            "out": [w, h]}


def costs(name, sizes=((960, 544), (480, 272)), half_above=600):
    """train/textspot2_shapes.costs for a d2s shape: MACs per frame at the network's input size, parameters, int8
    weights and the receptive field (the head is 1x1: textdet_model.receptive_field's)."""
    s = shape(name)
    out = {}
    for w, h in sizes:
        iw, ih = ((w + 1) // 2, (h + 1) // 2) if half_above and w > half_above else (w, h)
        out[f"{w}x{h}"] = cost(s, iw, ih)["macs"]
    c = cost(s, 480, 272)
    return {"macs": out, "params": c["params"], "weights_int8": c["weights_int8"],
            "receptive_field": TM.receptive_field(s)}


# ----------------------------------------------------------------------------- integer layers and blobs

def quantize_layers(shape, folded, act_scale, wbits=8):
    """train/textdet_export.quantize_layers for a d2s shape: the same arithmetic, the head's output scales
    HEAD_SCALES per channel. act_scale: one scale per layer (the head's entry, if any, is not read)."""
    spec = layers(shape)
    if len(folded) != len(spec) or len(act_scale) not in (len(spec), len(spec) - 1):
        raise ValueError(f"{len(folded)} folded layers / {len(act_scale)} scales for {len(spec)} layers")
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
        s_out = np.array(HEAD_SCALES) if L["head"] else np.full(cout, float(act_scale[i]))
        mx = np.abs(w.reshape(cout, -1)).max(1)
        s_w = np.where(mx > 0, mx / qmax, 1.0)
        wq = np.clip(np.floor(w / s_w.reshape(-1, 1, 1, 1) + 0.5), -qmax, qmax).astype(np.int8)
        bq = np.clip(np.floor(b / (s_w * s_x) + 0.5), -(2 ** 31 - 1), 2 ** 31 - 1).astype(np.int64)
        ms = [E.quantize_multiplier(float(s_w[c] * s_x / s_out[c])) for c in range(cout)]
        out.append({**L, "wq": wq, "bq": bq, "m": np.array([m for m, _ in ms], np.int64),
                    "sh": np.array([s for _, s in ms], np.int64)})
        if not L["head"]:
            s_x = float(act_scale[i])
    return out


def pack(shape, qlayers, ckpt_sha256=b"\0" * 8):
    """The d2s network's VTDT bytes: train/textdet_export.pack's layout and sections with stride 8, outputs 16,
    2 A codes per px and min cells 2 in the header (textspot2_twin.Net checks it)."""
    n = len(qlayers)
    data = bytearray()
    base = TW.HEADER + n * TW.RECORD
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
        recs.append(struct.pack("<BBBBB3xHHIIIII", 1 if L["kind"] == "dw" else 0, L["k"], L["stride"], L["dil"],
                                1 if L["relu"] else 0, L["cin"], L["cout"], wo, bo, mo, so, 0))
    while len(data) % 4:
        data.append(0)
    total = base + len(data)
    name = shape["name"].encode("ascii")[:31].ljust(32, b"\0")
    head = bytearray(TW.HEADER)
    head[:4] = TW.MAGIC
    struct.pack_into("<HBBBBH", head, 16, n, STRIDE, N_OUT, TM.A_SCALE, MIN_CELLS, 0)
    head[24:32] = ckpt_sha256[:8]
    head[32:64] = name
    body = bytes(head[16:]) + b"".join(recs) + bytes(data)
    struct.pack_into("<III", head, 4, TW.VERSION, total, zlib.crc32(body) & 0xFFFFFFFF)
    blob = bytes(head) + b"".join(recs) + bytes(data)
    assert len(blob) == total
    T2.Net(blob)              # the loader's checks
    return blob


def export_checkpoint(path, wbits=8):
    """A d2s quantisation-aware checkpoint (the module docstring) -> the network's VTDT bytes (RUN/final.vtd);
    textspot2_twin.pack(vtd, half_above, 1) wraps it as the VTSP version 2 blob (RUN/final.vtsp)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "textdet-qat" or not is_d2s(ck.get("shape")):
        raise SystemExit(f"{path}: not a quantisation-aware d2s checkpoint (kind textdet-qat, head {HEAD})")
    shp = ck["shape"]
    folded = [(w.double(), b.double()) for w, b in zip(ck["w"], ck["b"])]
    if len(ck["w"]) != len(ck["b"]) or len(folded) != len(layers(shp)):
        raise SystemExit(f"{path}: {len(ck['w'])} layers, {len(layers(shp))} expected for {shp['name']}")
    sha = hashlib.sha256(Path(path).read_bytes()).digest()
    return pack(shp, quantize_layers(shp, folded, ck["act_scale"], wbits), sha)


def export_vtsp(path, half_above=600, conf_min=1):
    """(VTDT bytes, VTSP version 2 bytes) of a d2s checkpoint."""
    vtd = export_checkpoint(path)
    return vtd, T2.pack(vtd, half_above, conf_min, version=T2.VERSION_D2S)


def random_int_layers(shape_name, seed, n_cal=6):
    """train/textdet_export.random_int_layers for a d2s shape: random int8 weights whose activations spread over the
    int8 range on procedural regions (tests of the C runtime, not a detector); the head's K and O channels
    centred on 0, A1 and A2 on about 8 px, for every sub-cell."""
    rng = np.random.default_rng(seed)
    shp = shape(shape_name)
    regions = [E.procedural_region(rng) for _ in range(n_cal)]
    xs = [g.astype(np.int64)[None] - 128 for g in regions]
    out = []
    for L in layers(shp):
        per = 1 if L["kind"] == "dw" else L["cin"]
        wq = rng.integers(-127, 128, (L["cout"], per, L["k"], L["k"])).astype(np.int8)
        wq[rng.random(wq.shape) < 0.3] = 0
        tl = TW.Layer()
        tl.kind, tl.k, tl.stride, tl.dil, tl.relu = (1 if L["kind"] == "dw" else 0), L["k"], L["stride"], L["dil"], 0
        tl.cin, tl.cout = L["cin"], L["cout"]
        tl.w = wq.astype(np.int64)
        tl.b = np.zeros(L["cout"], np.int64)
        tl.m = np.full(L["cout"], 1 << 30, np.int64)
        tl.sh = np.full(L["cout"], 30, np.int64)
        a = np.concatenate([E._accumulate(tl, x).reshape(L["cout"], -1) for x in xs], 1).astype(np.float64)
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
        tl.relu, tl.b, tl.m, tl.sh = (1 if L["relu"] else 0), bq, q["m"], q["sh"]
        xs = [TW.run_layer(tl, x) for x in xs]
    return shp, out


def random_vtdt(shape_name, seed, n_cal=6):
    shp, q = random_int_layers(shape_name, seed, n_cal)
    return pack(shp, q, hashlib.sha256(f"random:{shape_name}:{seed}".encode()).digest())


def random_blob(shape_name, seed, half_above=600, conf_min=1, n_cal=6):
    """(a random VTSP version 2 blob of a d2s shape, how it was made): eval/textspot_a9.blob_for's form."""
    blob = T2.pack(random_vtdt(shape_name, seed, n_cal), half_above, conf_min, version=T2.VERSION_D2S)
    return blob, (f"textspot2_export.random_int_layers seed {seed}, textspot2_twin.pack(half_above {half_above}, "
                  f"conf_min {conf_min}, version 2)")


# ----------------------------------------------------------------------------- torch: training and QAT

def to_stride4(y, height, width):
    """(N, 16, ceil(h/8), ceil(w/8)) head outputs of an h x w input -> (N, 4, ceil(h/4), ceil(w/4)) stride-4 maps
    (K, O, A1, A2): textspot2_twin.head_to_maps in torch, differentiable."""
    n, c, hn, wn = y.shape
    if c != N_OUT or (hn, wn) != T2.map_size(height, width, STRIDE):
        raise ValueError(f"head {tuple(y.shape)} for a {height}x{width} input")
    z = y.reshape(n, 2, 2, 4, hn, wn).permute(0, 3, 4, 1, 5, 2).reshape(n, 4, 2 * hn, 2 * wn)
    hd, wd = T2.map_size(height, width, T2.D2S_MAP_STRIDE)
    return z[:, :, :hd, :wd]


class TextDetD2S(TM.TextDet):
    """train/textdet_model.TextDet (the float network with BatchNorm) of a d2s shape: its trunk, a 16-channel head."""

    def __init__(self, shape):
        super().__init__(shape)
        self.spec = layers(shape)
        L = self.spec[-1]
        self.body[-1] = nn.Sequential(nn.Conv2d(L["cin"], L["cout"], 1, bias=True))


class QTextDetD2S(TM.QTextDet):
    """train/textdet_model.QTextDet (the folded network with int8 fake quantisation) of a d2s shape: the head's
    output scales HEAD_SCALES per channel, the export's arithmetic."""

    def __init__(self, shape, folded, act_scale):
        nn.Module.__init__(self)
        self.shape = dict(shape)
        self.spec = layers(shape)
        self.w = nn.ParameterList([nn.Parameter(torch.as_tensor(w).float().clone()) for w, _ in folded])
        self.b = nn.ParameterList([nn.Parameter(torch.as_tensor(b).float().clone()) for _, b in folded])
        for i, L in enumerate(self.spec):
            s = torch.tensor(HEAD_SCALES).view(1, N_OUT, 1, 1) if L["head"] else \
                torch.tensor(float(act_scale[i])).view(1, 1, 1, 1)
            self.register_buffer(f"s{i}", s)
        self.quant = True


# ----------------------------------------------------------------------------- CLI

def info(path):
    blob = Path(path).read_bytes()
    m = T2.Model(blob)
    print(f"{path}: VTSP version {m.version} ({m.head}), {m.blob_bytes} bytes, half_above {m.half_above}, conf_min "
          f"{m.conf_min}; network {m.name}, stride {m.net_stride} (maps at {m.stride}), {len(m.net.layers)} layers, "
          f"min cells {m.min_cells}")
    if m.version == T2.VERSION_D2S and m.name in S2.D2S:
        for k, v in costs(m.name, half_above=m.half_above)["macs"].items():
            print(f"  {k}: {v:,} MACs")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--vtd", help="also write the network's VTDT bytes here (--ckpt)")
    ap.add_argument("--half-above", type=int, default=600)
    ap.add_argument("--conf-min", type=int, default=1)
    ap.add_argument("--info")
    args = ap.parse_args(argv)
    if args.info:
        info(args.info)
        return 0
    if args.ckpt:
        vtd, blob = export_vtsp(args.ckpt, args.half_above, args.conf_min)
        if args.vtd:
            Path(args.vtd).write_bytes(vtd)
    elif args.random:
        blob, _ = random_blob(args.random, args.seed, args.half_above, args.conf_min)
    else:
        ap.error("--ckpt, --random or --info")
    if not args.out:
        ap.error("--out")
    Path(args.out).write_bytes(blob)
    print(f"{args.out}: {len(blob)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

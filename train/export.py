#!/usr/bin/env python3
"""Export a TinyVNOCR checkpoint to the weights blob of the plain-C runtime (runtime/vocr.h).

  export.py --ckpt models/v1/ckpt_best.pt --out v1_f32.vocr
  export.py --ckpt models/v1/ckpt_best.pt --out v1_w8.vocr --int8
  export.py --info v1_w8.vocr [--widths 256,640,1024] [--tile 4 --head 16]

BatchNorm is folded into the preceding convolution (in float64, stored as float32).
With --int8 every pointwise / row-collapse / head weight matrix is stored as int8 with
one float32 scale per output channel (symmetric, max-abs); the stem and depthwise
weights, all biases and scales stay float32. The vocabulary is embedded. The runtime
reads the blob in place: nothing is copied, so the file size is the weights memory.

Blob format, version 1 (little-endian; every section 16-byte aligned):

  header, 64 bytes
    0  char[4] "VOCR"         4  u16 version (1)         6  u16 header bytes (64)
    8  u32 file bytes        12  u32 flags (bit 0: int8 matrices present)
   16  u16 input height      18  u16 max resized width (training clamp, 1024; 1020 at stride 12)
   20  u16 width stride (8; 12 or 16 for train2 --stride)  22  u16 number of ops
   24  u32 classes (incl. blank)                        28  u32 op table offset
   32  u32 vocab offset      36  u32 vocab UTF-8 bytes  40  u32 CRC-32 of bytes [64, end)
   44  u32 parameter count   48  u8[16] first 16 bytes of the checkpoint's SHA-256
  op table: one 64-byte record per op, in execution order
    0 u8 type (1 CONV with one input channel, 2 DW depthwise, 3 PW pointwise, 4 HEAD)
    1 u8 flags (1 ReLU, 2 residual add, 4 bias, 8 int8 weights, 16 row collapse)
    2 u8 kh  3 u8 kw  4 u8 stride h  5 u8 stride w  6 u8 pad h  7 u8 pad w
    8 u16 cin  10 u16 cout  12 u16 rows in  14 u16 rows out
   16 u32 weights offset  20 u32 weights bytes  24 u32 bias offset  28 u32 scales offset
   32 u16 K (reduction length)  34 u16 cout padded to a multiple of 8  36..63 zero
  weights
    CONV  f32 [kh*kw][cout]            DW  f32 [kh*kw][c]
    PW / HEAD  f32 or i8 [cout_pad/8][K][8], K = cin (or rows_in*cin with row collapse,
               index dy*cin + c); int8 adds f32 scales[cout_pad]; bias f32[cout_pad]
  vocab: u16 offsets[classes + 1], then the UTF-8 bytes; class c is bytes
         [off[c], off[c+1]); class 0 (CTC blank) is empty.

A residual PW adds the input of the depthwise op right before it. The runtime's
activation layout is [column][row][channel] with channels innermost.

plan_arena() mirrors vocr_arena_bytes() in runtime/vocr.c exactly (tested by
runtime/tests/parity.py), so the arena size can be computed without building C.
"""
import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

MAGIC, VERSION, HEADER_BYTES, OP_BYTES = b"VOCR", 1, 64, 64
T_CONV, T_DW, T_PW, T_HEAD = 1, 2, 3, 4
F_RELU, F_RES, F_BIAS, F_INT8, F_COLLAPSE = 1, 2, 4, 8, 16
TYPE_NAME = {T_CONV: "conv", T_DW: "dw", T_PW: "pw", T_HEAD: "head"}
MODE_F32, MODE_W8F, MODE_W8I = 1, 2, 3
MODE_NAME = {MODE_F32: "f32", MODE_W8F: "w8f", MODE_W8I: "w8i"}
DENORMAL = 1e-30          # weights smaller than this are flushed to zero


def _align(n, a=16):
    return (n + a - 1) // a * a


# ---------------------------------------------------------------- checkpoint -> ops

def _fold(conv, bn):
    """Conv (no bias) + BatchNorm (eval) -> weight, bias in float64."""
    import torch
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else None
    if bn is None:
        return w, b
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
    w = w * scale.view(-1, 1, 1, 1)
    shift = bn.bias.detach().double() - bn.running_mean.detach().double() * scale
    b = shift if b is None else shift + b * scale
    return w, b


def model_ops(model, height):
    """Walk a TinyVNOCR (train/model.py) and return the runtime op list (float64 numpy)."""
    import torch.nn as nn

    from model import DS, Ctx
    ops, h = [], height

    def add(conv, bn, relu, res=False, head=False):
        nonlocal h
        w, b = _fold(conv, bn)
        w = w.numpy()
        b = None if b is None else b.numpy()
        cout, cin_g, kh, kw = w.shape
        groups = conv.groups
        cin = cin_g * groups
        (sh, sw), (ph, pw) = conv.stride, conv.padding
        h_out = (h + 2 * ph - kh) // sh + 1
        op = dict(kh=kh, kw=kw, sh=sh, sw=sw, ph=ph, pw=pw, cin=cin, cout=cout, h_in=h,
                  h_out=h_out, relu=relu, res=res, collapse=False, bias=b)
        if head:
            assert kh == kw == 1 and groups == 1 and h == 1, "head must be 1x1 on one row"
            op.update(type=T_HEAD, w=w.reshape(cout, cin))
        elif groups > 1:
            assert groups == cin == cout, "only depthwise grouping is supported"
            op.update(type=T_DW, w=w.reshape(cout, kh * kw))
        elif kh == kw == 1:
            op.update(type=T_PW, w=w.reshape(cout, cin))
        elif cin == 1:
            op.update(type=T_CONV, w=w.reshape(cout, kh * kw))
        elif kw == 1 and kh == h and ph == pw == 0 and sh == sw == 1:
            op.update(type=T_PW, collapse=True,
                      w=w[:, :, :, 0].transpose(0, 2, 1).reshape(cout, kh * cin))
        else:
            raise ValueError(f"conv {tuple(w.shape)} stride {conv.stride} pad {conv.padding} "
                             f"on {h} rows is not a runtime op")
        if op["type"] in (T_PW, T_HEAD):
            assert (op["sh"], op["sw"]) == (1, 1)
        ops.append(op)
        h = h_out

    for m in model.features:
        if isinstance(m, nn.Sequential):
            add(m[0], m[1], relu=True)
        elif isinstance(m, (DS, Ctx)):
            add(m.dw[0], m.dw[1], relu=True)
            add(m.pw[0], m.pw[1], relu=True, res=bool(getattr(m, "res", True)))
        else:
            raise ValueError(f"unsupported feature module {type(m).__name__}")
    add(model.bottleneck, None, relu=False)
    add(model.head, None, relu=False, head=True)
    for i, op in enumerate(ops):
        if op["res"]:
            d = ops[i - 1]
            assert d["type"] == T_DW and (d["sh"], d["sw"]) == (1, 1) and d["h_in"] == d["h_out"] \
                and d["cin"] == op["cin"] == op["cout"], f"op {i}: unsupported residual"
    return ops


def load_checkpoint(path, height=None):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import TinyVNOCR
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    n_classes = sd["head.weight"].shape[0]
    bottleneck = sd["bottleneck.weight"].shape[0]
    # depth / ctx / ctx_k: arch's deeper students (train.py records them; v1 has none)
    arch = {k: int(ck[k]) for k in ("depth", "ctx", "ctx_k") if ck.get(k) is not None}
    if ck.get("chans"):                     # explicit per-stage channel counts (train2.py --chans)
        arch["chans"] = tuple(int(c) for c in ck["chans"])
    if ck.get("stride"):                    # the CTC frame step (train2.py --stride; absent = 8)
        arch["stride"] = int(ck["stride"])
    model = TinyVNOCR(n_classes, width=float(ck.get("width", 1.0)), bottleneck=bottleneck, **arch)
    model.load_state_dict(sd)
    model.eval()
    vocab = list(ck["vocab"])
    assert len(vocab) + 1 == n_classes, "checkpoint vocab does not match the head"
    return model, vocab, int(height or ck.get("height", 32))


# ---------------------------------------------------------------- blob writer

def _pack_matrix(w, cout_pad):
    """[cout][K] -> [cout_pad/8][K][8]."""
    cout, k = w.shape
    full = np.zeros((cout_pad, k), w.dtype)
    full[:cout] = w
    return np.ascontiguousarray(full.reshape(cout_pad // 8, 8, k).transpose(0, 2, 1))


def quantize_rows(w):
    """Symmetric per-row int8: returns q int8 [rows][K] and scale float32 [rows]."""
    w = w.astype(np.float32)
    scale = (np.abs(w).max(axis=1) / np.float32(127)).astype(np.float32)
    safe = np.where(scale > 0, scale, np.float32(1))
    q = np.clip(np.rint(w / safe[:, None]), -127, 127).astype(np.int8)
    q[scale == 0] = 0
    return q, scale


def default_max_width(stride, limit=1024):
    """The width clamp of a blob: the largest multiple of its stride up to `limit` (1,024 at
    strides 8 and 16, 1,020 at 12), since the runtime needs max_width % stride == 0."""
    return limit // stride * stride


def build_blob(ops, vocab, height, int8=False, max_width=None, source_sha=b""):
    """The blob of `ops`. max_width None = default_max_width(the ops' stride): 1,024 for every
    stride-8 or stride-16 model; a max_width that is not a multiple of the stride is refused."""
    total_stride = 1
    for op in ops:
        total_stride *= op["sw"]
    if max_width is None:
        max_width = default_max_width(total_stride)
    if max_width % total_stride:
        raise ValueError(f"max_width {max_width} is not a multiple of the stride {total_stride}")
    sections = []                       # (offset, bytes)
    pos = [HEADER_BYTES + _align(OP_BYTES * len(ops))]

    def put(arr):
        data = np.ascontiguousarray(arr).tobytes()
        off = pos[0]
        sections.append((off, data))
        pos[0] = _align(off + len(data))
        return off, len(data)

    records, n_params, flushed, stride = [], 0, 0, 1
    for op in ops:
        t = op["type"]
        flags = (F_RELU if op["relu"] else 0) | (F_RES if op["res"] else 0) | \
                (F_COLLAPSE if op["collapse"] else 0)
        cout = op["cout"]
        w = np.asarray(op["w"], np.float64).astype(np.float32)
        tiny = (np.abs(w) < DENORMAL) & (w != 0)
        flushed += int(tiny.sum())
        w[tiny] = 0
        n_params += w.size + (0 if op["bias"] is None else cout)
        stride *= op["sw"]
        s_off = 0
        if t in (T_CONV, T_DW):
            k, cout_pad = op["kh"] * op["kw"], cout
            w_off, w_bytes = put(np.ascontiguousarray(w.T))           # [tap][channel]
        else:
            k = w.shape[1]
            cout_pad = _align(cout, 8)
            if t == T_PW:
                assert cout % 8 == 0, "pointwise outputs must be a multiple of 8"
            if int8:
                q, scale = quantize_rows(w)
                w_off, w_bytes = put(_pack_matrix(q, cout_pad))
                s_off, _ = put(np.concatenate([scale, np.zeros(cout_pad - cout, np.float32)]))
                flags |= F_INT8
            else:
                w_off, w_bytes = put(_pack_matrix(w, cout_pad))
        b_off = 0
        if op["bias"] is not None:
            b = np.zeros(cout_pad, np.float32)
            b[:cout] = np.asarray(op["bias"], np.float64).astype(np.float32)
            b_off, _ = put(b)
            flags |= F_BIAS
        records.append(struct.pack("<8B4H4I2H28x", t, flags, op["kh"], op["kw"], op["sh"], op["sw"],
                                   op["ph"], op["pw"], op["cin"], cout, op["h_in"], op["h_out"],
                                   w_off, w_bytes, b_off, s_off, k, cout_pad))
    n_classes = len(vocab) + 1
    enc = [b""] + [c.encode("utf-8") for c in vocab]
    offs = np.zeros(n_classes + 1, np.int64)
    offs[1:] = np.cumsum([len(e) for e in enc])
    assert offs[-1] < 65536, "vocab too large for 16-bit offsets"
    vocab_bytes = int(offs[-1])
    v_off, _ = put(np.frombuffer(offs.astype("<u2").tobytes() + b"".join(enc), np.uint8))
    total = pos[0]
    blob = bytearray(total)
    blob[HEADER_BYTES:HEADER_BYTES + OP_BYTES * len(ops)] = b"".join(records)
    for off, data in sections:
        blob[off:off + len(data)] = data
    crc = zlib.crc32(bytes(blob[HEADER_BYTES:])) & 0xFFFFFFFF
    sha = (source_sha + bytes(16))[:16]
    blob[:HEADER_BYTES] = struct.pack("<4sHHII4HIIIIII16s", MAGIC, VERSION, HEADER_BYTES, total,
                                      1 if int8 else 0, height, max_width, stride, len(ops),
                                      n_classes, HEADER_BYTES, v_off, vocab_bytes, crc, n_params, sha)
    return bytes(blob), dict(params=n_params, flushed=flushed, stride=stride)


# ---------------------------------------------------------------- blob reader

def read_blob(data):
    """Parse and check a blob. Returns a dict with the header fields, the op list (numpy
    arrays in logical layout: CONV/DW w [cout][taps]; PW/HEAD w [cout][K] (float32, or
    int8 plus scale)) and the vocabulary."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    (magic, ver, hb, total, flags, height, max_w, stride, n_ops, n_classes, ops_off, v_off,
     v_bytes, crc, n_params, sha) = struct.unpack_from("<4sHHII4HIIIIII16s", data, 0)
    if magic != MAGIC or ver != VERSION or hb != HEADER_BYTES or total > len(data):
        raise ValueError("not a version-1 VOCR blob")
    if zlib.crc32(data[HEADER_BYTES:total]) & 0xFFFFFFFF != crc:
        raise ValueError("CRC mismatch")
    ops = []
    for i in range(n_ops):
        (t, fl, kh, kw, sh, sw, ph, pw, cin, cout, h_in, h_out, w_off, w_bytes, b_off, s_off, k,
         cout_pad) = struct.unpack_from("<8B4H4I2H", data, ops_off + OP_BYTES * i)
        op = dict(type=t, relu=bool(fl & F_RELU), res=bool(fl & F_RES), int8=bool(fl & F_INT8),
                  collapse=bool(fl & F_COLLAPSE), kh=kh, kw=kw, sh=sh, sw=sw, ph=ph, pw=pw,
                  cin=cin, cout=cout, h_in=h_in, h_out=h_out, k=k, cout_pad=cout_pad,
                  w_bytes=w_bytes)
        if t in (T_CONV, T_DW):
            op["w"] = np.frombuffer(data, "<f4", k * cout, w_off).reshape(k, cout).T.copy()
        else:
            dt = np.int8 if op["int8"] else np.dtype("<f4")
            packed = np.frombuffer(data, dt, cout_pad * k, w_off).reshape(cout_pad // 8, k, 8)
            op["w"] = packed.transpose(0, 2, 1).reshape(cout_pad, k)[:cout].copy()
            if op["int8"]:
                op["scale"] = np.frombuffer(data, "<f4", cout_pad, s_off)[:cout].copy()
        op["bias"] = np.frombuffer(data, "<f4", cout_pad, b_off)[:cout].copy() if fl & F_BIAS else None
        ops.append(op)
    offs = np.frombuffer(data, "<u2", n_classes + 1, v_off).astype(np.int64)
    raw = data[v_off + 2 * (n_classes + 1):v_off + 2 * (n_classes + 1) + v_bytes]
    vocab = [raw[offs[c]:offs[c + 1]].decode("utf-8") for c in range(n_classes)]
    return dict(bytes=total, int8=bool(flags & 1), height=height, max_width=max_w, stride=stride,
                n_classes=n_classes, params=n_params, sha=sha.hex(), ops=ops, vocab=vocab)


def dequant(op):
    """float32 weight matrix as the runtime's float paths see it."""
    if op.get("int8"):
        return (op["w"].astype(np.float32) * op["scale"][:, None]).astype(np.float32)
    return op["w"]


def macs_per_frame(blob):
    """Multiply-accumulates the runtime does per CTC frame (the blob's stride of input columns):
    a line of width W costs ceil(W / stride) times this. Returns (total, head)."""
    S, s, tot, head = blob["stride"], 1, 0, 0
    for op in blob["ops"]:
        s *= op["sw"]
        cols = S // s                                   # output columns per frame
        if op["type"] in (T_CONV, T_DW):
            macs = cols * op["h_out"] * op["cout"] * op["kh"] * op["kw"]
        else:
            macs = cols * (1 if op["collapse"] else op["h_in"]) * op["k"] * op["cout"]
        tot += macs
        head += macs if op["type"] == T_HEAD else 0
    return tot, head


# ---------------------------------------------------------------- arena plan (mirror of C)

def default_tile(stride):
    """The runtime's default tile in CTC frames (vocr.c plan()): 4, or for a stride over 8 the
    whole frames nearest 32 input columns, at least 1: 3 at stride 12, 2 at 16."""
    return 4 if stride <= 8 else 1 if stride >= 64 else (32 + stride // 2) // stride


def plan_arena(blob, max_width=0, tile=0, head=0, mode=0):
    """Exact arena bytes vocr_arena_bytes() returns. tile/head in frames (0 = defaults
    default_tile(stride) and 16; tile >= max_width / stride = whole line); mode 0 = the blob's
    default (f32 for float blobs, w8f for int8)."""
    ops, S, H0 = blob["ops"], blob["stride"], blob["height"]
    if mode == 0:
        mode = MODE_W8F if blob["int8"] else MODE_F32
    if mode != MODE_F32 and not blob["int8"]:
        raise ValueError("int8 modes need an int8 blob")
    if mode == MODE_F32 and blob["int8"]:
        raise ValueError("an int8 blob has no float32 weights")
    if not max_width or max_width > blob["max_width"]:
        max_width = blob["max_width"]
    max_width = _align(max_width, S)
    t_max = max_width // S
    tile, head = tile or default_tile(S), head or 16
    F = min(tile, t_max)
    hb = min(head, t_max)
    L = len(ops)
    # front group: leading CONV/DW ops, minus the last one if a residual PW follows it
    f = 0
    while f < L and ops[f]["type"] in (T_CONV, T_DW):
        f += 1
    if f < L and ops[f]["res"]:
        f -= 1
    f -= 1                                   # index of the last front op
    s_out, s = [], 1
    for op in ops:
        s *= op["sw"]
        s_out.append(s)
    n_tile = [F * S // s_out[i] for i in range(L)]        # output columns per tile
    n_step = [s_out[f] // s_out[i] if i <= f else n_tile[i] for i in range(L)]   # per step
    n_src = s_out[f]                                       # source columns per front step
    lag = []
    prev = 0
    for op in ops:
        r = op["kw"] - 1 - op["pw"]
        d = -(-(prev + r + 1) // op["sw"]) - 1
        lag.append(d)
        prev = d
    persistent = _align(8 * H0)              # resize row table
    for i, op in enumerate(ops):
        if op["type"] in (T_CONV, T_DW):
            n_prev = n_src if i == 0 else (n_step[i - 1] if i <= f else n_tile[i - 1])
            halo = lag[i] * op["sw"] + op["pw"] - (lag[i - 1] if i else 0)
            cin = 1 if i == 0 else op["cin"]
            persistent += _align((n_prev + halo) * op["h_in"] * cin * 4)
    k_head = ops[-1]["k"]
    acc_frames = hb + F - 1
    persistent += _align(acc_frames * k_head * 4) + _align(acc_frames * 4) * 2
    trans = []                               # transient bytes in and out of each op
    for i, op in enumerate(ops):
        tin = n_tile[i - 1] * ops[i - 1]["h_out"] * ops[i - 1]["cout"] * 4 \
            if op["type"] == T_PW else 0
        tout = n_tile[i] * op["h_out"] * op["cout"] * 4 \
            if i + 1 < L and ops[i + 1]["type"] == T_PW else 0
        scratch = 0
        if op["type"] in (T_PW, T_HEAD):
            pos = acc_frames if op["type"] == T_HEAD else \
                n_tile[i] * (1 if op["collapse"] else op["h_in"])
            if mode == MODE_W8F:
                scratch = op["k"] * 8 * 4
            elif mode == MODE_W8I:
                scratch = _align(pos * op["k"] * 2) + _align(pos * 4)
        trans.append(_align(tin) + _align(tout) + _align(scratch))
    return persistent + max(trans)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--height", type=int, default=0, help="input height (default: ckpt or 32)")
    ap.add_argument("--max-width", type=int, default=0,
                    help="width clamp (default: the largest multiple of the stride up to 1024)")
    ap.add_argument("--info", help="print a blob's sizes and arena bytes")
    ap.add_argument("--widths", default="256,640,1024")
    ap.add_argument("--tile", type=int, default=0, help="tile frames (0: the runtime's default_tile(stride))")
    ap.add_argument("--head", type=int, default=16)
    args = ap.parse_args()
    if args.ckpt:
        model, vocab, height = load_checkpoint(args.ckpt, args.height)
        sha = hashlib.sha256(Path(args.ckpt).read_bytes()).digest()
        blob, meta = build_blob(model_ops(model, height), vocab, height, args.int8, args.max_width or None, sha)
        Path(args.out).write_bytes(blob)
        print(json.dumps({"out": args.out, "bytes": len(blob), "int8": args.int8, **meta}))
        args.info = args.info or args.out
    if args.info:
        b = read_blob(args.info)
        modes = [MODE_W8F, MODE_W8I] if b["int8"] else [MODE_F32]
        rows = {MODE_NAME[m]: {w: plan_arena(b, w, args.tile, args.head, m)
                               for w in map(int, args.widths.split(","))} for m in modes}
        macs, head = macs_per_frame(b)
        print(json.dumps({"blob": str(args.info), "bytes": b["bytes"], "params": b["params"],
                          "classes": b["n_classes"], "height": b["height"], "ops": len(b["ops"]),
                          "macs_per_frame": macs, "head_macs_per_frame": head,
                          "macs_per_line": {w: macs * -(-w // b["stride"])
                                            for w in map(int, args.widths.split(","))},
                          "tile": args.tile or default_tile(b["stride"]), "head": args.head or 16,
                          "arena": rows}))


if __name__ == "__main__":
    main()

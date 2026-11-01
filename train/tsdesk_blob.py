#!/usr/bin/env python3
"""textspot-desktop's VTSD blob (issue #173, docs/textspot-desktop.md): the C path's weights file for desktop/vocr_tsd.c
(vocr_tsd.h), written from a train/tsdesk_net.TsdNet. train/tsdesk_det.py export writes it beside the ONNX file.

    import tsdesk_blob as TB
    blob = TB.blob_bytes(model, "fit:960", tsdesk_det.DECODE)       # bytes; TB.read_header(blob) -> dict

Version 1, little-endian. A 160-byte header, then the float32 weights, then the CRC-32 (zlib) of everything before it:

    0   "VTSD"              4   u32 version (1)       8   u32 header bytes (160)   12  u32 total bytes
    16  u32 patch (4)       20  u32 hs (the head's stride: 4 or 8)
    24  u32 c[4]            the channels at strides 4, 8, 16, 32
    40  u32 n[4]            the residual separable blocks per stage (stage 4's need no down block)
    56  u32 F               the pyramid's channels
    60  u32 smooth          separable blocks at the head's stride
    64  u32 gc              1: global context at stride 32
    68  u32 tags            72  u32 out (6 + tags maps)          76  u32 outp (out rounded up to 8)
    80  u32 min_cells       84  u32 max_boxes                    the decode's integer constants
    88  u32 size kind       92  u32 size n                       the default input policy (0 native, 1 fit, 2 long)
    96  u8 dil[4][8]        each stage's residual block i's dilation (0 past n[stage])
    128 f64 conf_min_logit  136 f64 min_across   144 f64 along_reach   the decode's float constants
    152 u32 align (32)      156 u32 0

The weights, BatchNorm folded (tsdesk_net.fold_bn), in the network's order and the C kernels' layouts (each weight
array followed by its bias):
    stem        [4][4][c0] (ky, kx, out)
    blocks      stage 4's n[0] blocks; then for strides 8, 16 and 32 the down block (3x3 depthwise at stride 2, then
                1x1) and n[s] residual blocks: depthwise [3][3][C], pointwise [Cin][Cout]
    gc          (gc = 1) [c3][c3]
    laterals    in TsdNet.lat's order ((stride 4 when hs = 4), 8, 16, 32): [Cin][F]
    smooth      `smooth` blocks: depthwise [3][3][F], pointwise [F][F]
    head        [F][outp] (columns past out are 0), with A1, A2, E1 and E2 (maps 2..5) times hs folded in: exact,
                hs being a power of 2, so the C head gives the ONNX graph's `hs * y` without a multiply.
The C network normalises the grey input itself ((g - 128) / 128, as the graph does).
"""
import struct
import zlib

import numpy as np

MAGIC = b"VTSD"
VERSION = 1
HEADER = 160
ALIGN = 32
MAX_BLOCKS = 8
MAX_CHANNELS = 256
MAX_SMOOTH = 4
MAX_TAGS = 8
MAX_BOXES = 256
SIZE_KINDS = {"native": 0, "fit": 1, "long": 2}
DECODE_DEFAULTS = {"min_cells": 2, "conf_min_logit": 0.0, "min_across": 4.0, "along_reach": 3.0, "max_boxes": 256}


def policy(size):
    """(kind, n) of an input policy: "native" -> (0, 0), "fit:N" (or a bare "N") -> (1, N), "long:N" -> (2, N)."""
    if size in (None, "", "native"):
        return 0, 0
    kind, _, n = str(size).partition(":")
    if not n:
        kind, n = "fit", kind
    if kind not in ("fit", "long") or int(n) <= 0:
        raise ValueError(f"size policy {size!r}: fit:N, long:N or native")
    return SIZE_KINDS[kind], int(n)


def _np(t):
    return np.ascontiguousarray(t.detach().cpu().numpy(), np.float32)


def blob_bytes(model, size="fit:960", decode=None):
    """The VTSD version-1 blob (module docstring) of a TsdNet (train or eval mode; a folded copy is written)."""
    import tsdesk_net as N
    sh = dict(model.shape)
    c, n, fc, hs = [int(v) for v in sh["c"]], [int(v) for v in sh["n"]], int(sh["F"]), int(sh["hs"])
    smooth, gc = int(sh.get("smooth", 0)), int(bool(sh.get("gc")))
    if int(sh["patch"]) != 4 or hs not in (4, 8):
        raise ValueError("VTSD holds a 4 x 4 patch stem and a head at stride 4 or 8")
    if any(v % 8 or not 0 < v <= MAX_CHANNELS for v in c + [fc]):
        raise ValueError(f"channels {c} / F {fc}: multiples of 8 up to {MAX_CHANNELS}")
    if any(v > MAX_BLOCKS for v in n) or smooth > MAX_SMOOTH:
        raise ValueError(f"at most {MAX_BLOCKS} blocks per stage and {MAX_SMOOTH} smoothing blocks")
    m = N.fold_bn(model)
    out = int(m.head.out_channels)
    tags = out - 6
    if not 0 <= tags <= MAX_TAGS:
        raise ValueError(f"{tags} tags")
    outp = (out + 7) // 8 * 8
    d = dict(DECODE_DEFAULTS)
    d.update({k: v for k, v in (decode or {}).items() if k in DECODE_DEFAULTS})
    if not 1 <= int(d["max_boxes"]) <= MAX_BOXES or int(d["min_cells"]) < 1:
        raise ValueError(f"decode constants {d}")
    parts = []

    def add(a):
        parts.append(np.ascontiguousarray(a, np.float32).ravel())

    def conv_check(conv, k, stride, dil, groups):
        if (tuple(conv.kernel_size) != (k, k) or tuple(conv.stride) != (stride, stride) or
                tuple(conv.dilation) != (dil, dil) or conv.groups != groups or
                tuple(conv.padding) != ((dil, dil) if k == 3 else (0, 0))):
            raise ValueError(f"unexpected convolution {conv}")

    def sep(blk, cin, cout, stride):
        dil = int(blk.dw.dilation[0])
        conv_check(blk.dw, 3, stride, dil, cin)
        conv_check(blk.pw, 1, 1, 1, 1)
        if blk.pw.in_channels != cin or blk.pw.out_channels != cout:
            raise ValueError(f"block {cin} -> {cout}: {blk.pw}")
        add(_np(blk.dw.weight)[:, 0].transpose(1, 2, 0))
        add(_np(blk.dw.bias))
        add(_np(blk.pw.weight)[:, :, 0, 0].T)
        add(_np(blk.pw.bias))
        return dil

    def pw(conv, cin, cout):
        conv_check(conv, 1, 1, 1, 1)
        if conv.in_channels != cin or conv.out_channels != cout:
            raise ValueError(f"1x1 {cin} -> {cout}: {conv}")
        add(_np(conv.weight)[:, :, 0, 0].T)
        add(_np(conv.bias))

    stem = m.stem[0]
    if tuple(stem.kernel_size) != (4, 4) or tuple(stem.stride) != (4, 4) or stem.in_channels != 1:
        raise ValueError(f"unexpected stem {stem}")
    add(_np(stem.weight)[:, 0].transpose(1, 2, 0))
    add(_np(stem.bias))
    dil = np.zeros((4, MAX_BLOCKS), np.uint8)
    stages = (m.s4, m.s8, m.s16, m.s32)
    if len(stages[0]) != n[0]:
        raise ValueError("stage 4's blocks")
    for i, blk in enumerate(stages[0]):
        dil[0, i] = sep(blk, c[0], c[0], 1)
    for s in (1, 2, 3):
        blocks = list(stages[s])
        if len(blocks) != 1 + n[s]:
            raise ValueError(f"stage {4 << s}'s blocks")
        if sep(blocks[0], c[s - 1], c[s], 2) != 1:
            raise ValueError("a down block at dilation 1")
        for i, blk in enumerate(blocks[1:]):
            dil[s, i] = sep(blk, c[s], c[s], 1)
    if gc:
        pw(m.gc.fc, c[3], c[3])
    elif m.gc is not None:
        raise ValueError("gc")
    lat_c = ([c[0]] if hs == 4 else []) + [c[1], c[2], c[3]]
    if len(m.lat) != len(lat_c):
        raise ValueError("laterals")
    for lat, ci in zip(m.lat, lat_c):
        pw(lat.conv, ci, fc)
    if len(m.smooth) != smooth:
        raise ValueError("smoothing blocks")
    for blk in m.smooth:
        if sep(blk, fc, fc, 1) != 1:
            raise ValueError("a smoothing block at dilation 1")
    conv_check(m.head, 1, 1, 1, 1)
    w = _np(m.head.weight)[:, :, 0, 0].T.copy()
    b = _np(m.head.bias).copy()
    w[:, 2:6] *= np.float32(hs)
    b[2:6] *= np.float32(hs)
    wp = np.zeros((fc, outp), np.float32)
    wp[:, :out] = w
    bp = np.zeros(outp, np.float32)
    bp[:out] = b
    add(wp)
    add(bp)
    body = np.concatenate(parts).astype("<f4").tobytes()
    kind, nsz = policy(size)
    hdr = MAGIC + struct.pack("<III", VERSION, HEADER, HEADER + len(body) + 4)
    hdr += struct.pack("<II", 4, hs) + struct.pack("<4I", *c) + struct.pack("<4I", *n)
    hdr += struct.pack("<10I", fc, smooth, gc, tags, out, outp, int(d["min_cells"]), int(d["max_boxes"]), kind, nsz)
    hdr += dil.tobytes()
    hdr += struct.pack("<3d", float(d["conf_min_logit"]), float(d["min_across"]), float(d["along_reach"]))
    hdr += struct.pack("<II", ALIGN, 0)
    if len(hdr) != HEADER:
        raise AssertionError(len(hdr))
    blob = hdr + body
    return blob + struct.pack("<I", zlib.crc32(blob) & 0xFFFFFFFF)


def read_header(blob):
    """The header of a VTSD blob as a dict (checks the magic, version, size and CRC-32)."""
    b = bytes(blob)
    if len(b) < HEADER + 4 or b[:4] != MAGIC:
        raise ValueError("not a VTSD blob")
    version, header, total = struct.unpack_from("<III", b, 4)
    if version != VERSION or header != HEADER or total != len(b):
        raise ValueError(f"VTSD version {version}, header {header}, {total} bytes (file {len(b)})")
    if zlib.crc32(b[:-4]) & 0xFFFFFFFF != struct.unpack_from("<I", b, len(b) - 4)[0]:
        raise ValueError("VTSD CRC-32 mismatch")
    patch, hs = struct.unpack_from("<II", b, 16)
    c = list(struct.unpack_from("<4I", b, 24))
    n = list(struct.unpack_from("<4I", b, 40))
    fc, smooth, gc, tags, out, outp, min_cells, max_boxes, kind, nsz = struct.unpack_from("<10I", b, 56)
    dil = np.frombuffer(b, np.uint8, 4 * MAX_BLOCKS, 96).reshape(4, MAX_BLOCKS)
    conf, min_across, reach = struct.unpack_from("<3d", b, 128)
    size = "native" if kind == 0 else f"{('fit', 'long')[kind - 1]}:{nsz}"
    return {"patch": patch, "hs": hs, "c": c, "n": n, "F": fc, "smooth": smooth, "gc": gc, "tags": tags, "out": out,
            "outp": outp, "size": size, "size_kind": kind, "size_n": nsz,
            "dil": [[int(v) for v in dil[s, :n[s]]] for s in range(4)],
            "decode": {"min_cells": min_cells, "conf_min_logit": conf, "min_across": min_across,
                       "along_reach": reach, "max_boxes": max_boxes}, "bytes": len(b)}

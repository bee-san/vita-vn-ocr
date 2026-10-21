"""Bit-exact Python twin of runtime/vocr_textspot3.c (issue #108, docs/textspot3.md "Two families"): the full-frame
text spotter for VTSP version 1 and 2 blobs (synth/textspot2_twin.py's, unchanged) and for VTSP version 3, the
multi-scale family M: a stride-8 trunk streamed by rows, then layers on whole maps at strides 8 to 256 (3x3
depthwise layers and 1x1 layers over the concatenation of up to three upsampled maps), ending in version 2's d2s
head, which the decode reads as stride-4 maps.

    from textspot3_twin import Model, spot
    m = Model(open("spot.vtsp", "rb").read())   # a VTSP version 1, 2 or 3 blob
    r = spot(m, frame)                           # frame: H x W grey or H x W x 3/4 RGB(A) uint8
    r.boxes, r.blocks, r.pick, r.truncated       # textspot_twin's Result, Box and Block, the same fields

Versions 1 and 2 are synth/textspot2_twin.py exactly: Model keeps textspot2_twin.Model(blob) as .t2 (with its
fields copied), and spot() returns textspot2_twin.spot(m.t2, frame); nothing of them is re-implemented here. A
version-1 or version-2 loader (textspot_twin, textspot2_twin, runtime/vocr_textspot.c, runtime/vocr_textspot2.c)
refuses every version-3 blob by its version field.

VERSION 3. The VTSP blob (little-endian; train/textspot3_model.py writes it):

    0   char[4]  magic "VTSP"
    4   u32      version: 3
    8   u32      total bytes
    12  u32      CRC-32 (zlib) of bytes [16, total)
    16  u16      half_above: a frame wider than this many px is halved (0: the width never halves it)
    18  u8       conf_min, 1..127 (as version 1)
    19  u8       reserved (0)
    20  u32      offset of the network (32)
    24  u32      bytes of the network (total - 32)
    28  u16      half_above_h: a frame taller than this many px is halved (0: the height never halves it)
    30  u16      reserved (0)
    32  the network, a VTMS section:

    0   char[4]  magic "VTMS"
    4   u32      version (1)
    8   u32      total bytes of the section
    12  u32      CRC-32 (zlib) of the section's bytes [16, total)
    16  u16      stream layers n_s, 1..32
    18  u8       the stream's output stride: 8
    19  u8       the head's outputs: 16
    20  u8       A codes per px: 2
    21  u8       min cells, 1..4 (the exporter writes 2, a stride-4 map's)
    22  u16      map ops n_o, 1..64
    24  u8[8]    sha256 prefix of the source checkpoint
    32  char[32] shape name, NUL padded (ASCII once the trailing NULs go)
    64  n_s stream layer records, 32 bytes each: synth/textdet_twin.py's layer records, unchanged (u8 kind 0 dense
        / 1 depthwise, u8 k 1 or 3, u8 stride 1 or 2, u8 dilation, u8 relu, u8[3] reserved, u16 cin, u16 cout,
        u32 weights, bias, multiplier and shift offsets, u32 reserved)
    64 + 32 n_s: n_o map op records, 32 bytes each:
          0   u8      kind: 0 a 1x1 layer over 1..3 sources ("pw"), 1 a 3x3 depthwise layer ("dw")
          1   u8      stride: 1, or 2 (dw only, then dilation 1)
          2   u8      dilation: 1..8 (dw); 1 (pw)
          3   u8      relu: 1 outputs clamped to [0, 127], 0 to [-128, 127]
          4   u8      sources: 1 (dw), 1..3 (pw)
          5   u8      reserved (0)
          6   u16     cout: 8..1024, a multiple of 8 (dw: its source's channels)
          8   u8[3]   the source maps (below; each an earlier map; an unused entry 0)
          11  u8[3]   their upsampling exponents u, 0..3 (dw: 0; an unused entry 0)
          14  u16     reserved (0)
          16  u32     weights offset, 20 u32 bias offset, 24 u32 multiplier offset, 28 u32 shift offset
    data: per stream layer, then per op: int8 weights [cout][cin, or 1 if depthwise][k][k], int32 bias[cout],
          int32 multiplier[cout] (> 0), u8 shift[cout] (1..62), each section 4-aligned, inside the data part
          [64 + 32 (n_s + n_o), total). Offsets count from the section's first byte. Every weight of a version-3
          network lies in [-127, 127] (the exporter's range; -128 is refused).

The maps. Map 0 (F8) is the stream's output; map k + 1 is op k's output. Every map has a stride s (network-input
px per cell) and c channels: map 0 has s = 8 and the last stream layer's cout channels (8..64, a multiple of 8);
a dw op's output has its source's stride times its own stride; a pw op's sources must all have the strides
s_op 2^u (s_op >= 8, the op's stride) and the op's output has s_op and its cout. Every stride is 8 2^e with
s <= 256. A pw op's cin, the sum of its sources' channels, is at most 1,024 (a source may appear more than once).
The last op is the head: a pw op of 16 outputs, relu 0, at stride 8. For an h x w network input a map of stride s
is ceil(h / s) x ceil(w / s) cells (a stride-2 dw of a ceil(h / s)-row map gives ceil(h / 2s) rows).

The frame, step by step (W x H; integer arithmetic only):

  grey      g = (77 R + 150 G + 29 B) >> 8 for RGB(A) (synth/common.to_gray), the frame itself when grey
  input     f = 2 when (half_above > 0 and W > half_above) or (half_above_h > 0 and H > half_above_h), else 1.
            f = 2: textspot's exact 2x2 mean (textspot_twin.halve: (4 px + 2) >> 2, the last row or column
            repeated), ceil(H / 2) x ceil(W / 2); f = 1: g. With the exporter's 480 and 272 any frame up to
            960 x 544 gives an input of at most 480 x 272 (a Vita frame at the PSP's scale, a PSP frame as it is)
  x0        = x - 128 (int8)
  stream    the n_s stream layers, textdet_twin.run_layer each (version 1's arithmetic: a dense or depthwise 1x1
            or 3x3 conv with stride 1 or 2, dilation d and zero padding, int32 accumulator, + bias, then
            out = clamp(floor((acc * mult + 2^(shift - 1)) / 2^shift), relu ? 0 : -128, 127)): map 0
  ops       in order, op k gives map k + 1:
            dw   textdet_twin.run_layer as a 3x3 depthwise layer (stride, dilation, zero padding) on its source
            pw   x = the concatenation, channel by channel in source order, of each source m_i read through its
                 upsampling: x_i[c][a][b] = m_i[c][a >> u_i][b >> u_i] for a < ceil(h / s_op), b < ceil(w / s_op)
                 (nearest, by 2^u_i, cropped to the op's size); then textdet_twin.run_layer as a 1x1 dense layer
            the requantisation of both is version 1's, exactly
  head      the last map, 16 channels at stride 8: textspot2_twin's d2s head; channel 4 (2 dy + dx) + m of cell
            (i, j) is map m (0 K, 1 O, 2 A1, 3 A2) of cell (2 i + dy, 2 j + dx) of the decode's maps at stride 4,
            cropped to ceil(h / 4) x ceil(w / 4) (textspot2_twin.head_to_maps). The exporter writes K and O at
            1/16 per code and A1, A2 at 1/2 network-input px per code
  decode    textspot_twin's, unchanged at stride 4 (textspot2_twin.spot_maps: kernel cells K > 0, 4-connected
            components of at least min cells, conf = floor(sum K / n) >= conf_min, vertical = sum O > 0, near /
            far across the line at the doubled cell centre c2 = 2 * 4 * I + 3, along [4 f first, 4 f (last + 1)),
            the 4 f drop, the pad, the clip; then the 64-box cap, the dark flags (common.line_dark on the frame's
            grey), blocks, order and pick). Boxes are in frame px (f as in version 1)

A version-3 runtime may compute the maps in any order and layout; only these values are specified. The C runtime
(runtime/vocr_textspot3.c) streams the stream part by rows into a whole map 0, runs the ops on whole maps in an
arena, and streams the head's rows into the decode.

Model fields: version (1, 2 or 3), head ("kroa", "kroa_x2", or "ms_x2" for version 3), half_above, half_above_h
(version 3; 0 otherwise), conf_min, min_cells, name, blob_bytes, net (version 3: .stream, the stream's
textdet_twin.Layer objects, .ops, the Op objects, .map_stride and .map_chans per map), net_stride (8 for
versions 2 and 3) and stride, the stride of the maps the decode reads (4 for versions 2 and 3).
"""
import struct
import zlib

import numpy as np

import textdet_twin as TD
import textspot2_twin as T2
import textspot_twin as S

MAGIC = S.MAGIC
HEADER = S.HEADER                 # 32: the VTSP header
VERSION_MS = 3
NET_MAGIC = b"VTMS"
NET_VERSION = 1
NET_HEADER = 64
RECORD = 32
MAX_STREAM = 32
MAX_OPS = 64
MAX_CHANS = 1024                  # a map's channels and a pw op's cin
MAX_MAP_STRIDE = 256
MAX_UP = 3
MAX_SOURCES = 3
STREAM_STRIDE = 8
OUTPUTS = T2.D2S_OUTPUTS          # 16
MAP_STRIDE = T2.D2S_MAP_STRIDE    # 4: the decode reads stride-4 maps
W_MAX = 127                       # every version-3 weight lies in [-127, 127]
PW, DW = 0, 1


class Op:
    """A map op: kind (PW or DW), stride, dil, relu, cout, cin, src (source map indices), up (their upsampling
    exponents), out_stride, w, b, m, sh (int64 arrays, w of shape (cout, cin or 1, k, k))."""
    __slots__ = ("kind", "stride", "dil", "relu", "cout", "cin", "src", "up", "out_stride", "k", "w", "b", "m", "sh")

    def layer(self):
        """The op as a textdet_twin.Layer (for textdet_twin.run_layer)."""
        L = TD.Layer()
        L.kind, L.k, L.stride, L.dil, L.relu = self.kind, self.k, self.stride, self.dil, self.relu
        L.cin, L.cout, L.w, L.b, L.m, L.sh = (self.cout if self.kind == DW else self.cin), self.cout, self.w, \
            self.b, self.m, self.sh
        return L


def _sections(blob, total, data, specs, what):
    for off, size in specs:
        if off % 4 or off < data or off + size > total:
            raise ValueError(f"{what}: section outside the blob")


def _requant(blob, nw, wo, bo, mo, so, cout, shape, what):
    w = np.frombuffer(blob, np.int8, nw, wo).astype(np.int64).reshape(shape)
    if (w < -W_MAX).any():
        raise ValueError(f"{what}: a weight of -128")
    b = np.frombuffer(blob, "<i4", cout, bo).astype(np.int64)
    m = np.frombuffer(blob, "<i4", cout, mo).astype(np.int64)
    sh = np.frombuffer(blob, np.uint8, cout, so).astype(np.int64)
    if (m <= 0).any() or (sh < 1).any() or (sh > 62).any():
        raise ValueError(f"{what}: bad requantisation")
    return w, b, m, sh


class Net:
    """A VTMS network section. Raises ValueError on anything the C loader refuses."""

    def __init__(self, blob):
        blob = bytes(blob)
        if len(blob) < NET_HEADER or blob[:4] != NET_MAGIC:
            raise ValueError("not a VTMS section")
        version, total, crc = struct.unpack_from("<III", blob, 4)
        if version != NET_VERSION or total != len(blob):
            raise ValueError(f"version {version} / size {total} != {NET_VERSION} / {len(blob)}")
        if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
            raise ValueError("CRC mismatch")
        n_s, self.stride, self.outputs, self.a_scale, self.min_cells, n_o = struct.unpack_from("<HBBBBH", blob, 16)
        if self.stride != STREAM_STRIDE or self.outputs != OUTPUTS or self.a_scale != 2 or \
                not 1 <= self.min_cells <= 4:
            raise ValueError("unsupported header (stride 8, 16 outputs, 2 A codes per px, min cells 1..4)")
        if not 1 <= n_s <= MAX_STREAM or not 1 <= n_o <= MAX_OPS or NET_HEADER + (n_s + n_o) * RECORD > total:
            raise ValueError("bad layer or op count")
        self.ckpt = blob[24:32].hex()
        self.name = blob[32:64].rstrip(b"\0").decode("ascii")
        data = NET_HEADER + (n_s + n_o) * RECORD
        self.stream = []
        stride, cin = 1, 1
        for i in range(n_s):
            r = struct.unpack_from("<BBBBB3xHHIIIII", blob, NET_HEADER + i * RECORD)
            L = TD.Layer()
            L.kind, L.k, L.stride, L.dil, L.relu, L.cin, L.cout = r[:7]
            wo, bo, mo, so = r[7:11]
            if L.kind not in (0, 1) or L.k not in (1, 3) or L.stride not in (1, 2) or not 1 <= L.dil <= 8 \
                    or L.relu not in (0, 1) or L.cin != cin or not 1 <= L.cout <= 64:
                raise ValueError(f"stream layer {i}: unsupported")
            if L.kind == 1 and (L.cout != L.cin or L.k != 3):
                raise ValueError(f"stream layer {i}: bad depthwise layer")
            if L.stride == 2 and (L.k != 3 or L.dil != 1):
                raise ValueError(f"stream layer {i}: a stride-2 layer is 3x3, dilation 1")
            per = 1 if L.kind == 1 else L.cin
            nw = L.cout * per * L.k * L.k
            _sections(blob, total, data, ((wo, nw), (bo, 4 * L.cout), (mo, 4 * L.cout), (so, L.cout)),
                      f"stream layer {i}")
            L.w, L.b, L.m, L.sh = _requant(blob, nw, wo, bo, mo, so, L.cout, (L.cout, per, L.k, L.k),
                                           f"stream layer {i}")
            stride *= L.stride
            cin = L.cout
            self.stream.append(L)
        if stride != STREAM_STRIDE or cin % 8:
            raise ValueError("the stream does not end at stride 8 in a multiple of 8 channels")
        self.map_stride, self.map_chans = [STREAM_STRIDE], [cin]
        self.ops = []
        for k in range(n_o):
            base = NET_HEADER + (n_s + k) * RECORD
            kind, st, dil, relu, nsrc, r0, cout = struct.unpack_from("<BBBBBBH", blob, base)
            src = list(blob[base + 8:base + 11])
            up = list(blob[base + 11:base + 14])
            r1, wo, bo, mo, so = struct.unpack_from("<HIIII", blob, base + 14)
            what = f"op {k}"
            if kind not in (PW, DW) or st not in (1, 2) or not 1 <= dil <= 8 or relu not in (0, 1) or r0 or r1:
                raise ValueError(f"{what}: unsupported")
            if not 8 <= cout <= MAX_CHANS or cout % 8:
                raise ValueError(f"{what}: cout {cout} (8..1024, a multiple of 8)")
            if not 1 <= nsrc <= (1 if kind == DW else MAX_SOURCES):
                raise ValueError(f"{what}: {nsrc} sources")
            if any(src[i] or up[i] for i in range(nsrc, MAX_SOURCES)):
                raise ValueError(f"{what}: an unused source entry is not 0")
            src, up = src[:nsrc], up[:nsrc]
            if any(s > k for s in src) or any(u > MAX_UP for u in up):
                raise ValueError(f"{what}: a source is not an earlier map, or u > 3")
            op = Op()
            op.kind, op.stride, op.dil, op.relu, op.cout, op.src, op.up = kind, st, dil, relu, cout, src, up
            if kind == DW:
                s0 = src[0]
                if up[0] or cout != self.map_chans[s0] or (st == 2 and dil != 1):
                    raise ValueError(f"{what}: bad depthwise op")
                op.k, op.cin = 3, cout
                op.out_stride = self.map_stride[s0] * st
                per = 1
            else:
                if st != 1 or dil != 1:
                    raise ValueError(f"{what}: a 1x1 op has stride 1 and dilation 1")
                outs = {self.map_stride[s] >> u if self.map_stride[s] >> u << u == self.map_stride[s] else -1
                        for s, u in zip(src, up)}
                if len(outs) != 1 or min(outs) < STREAM_STRIDE:
                    raise ValueError(f"{what}: the sources do not meet at one stride >= 8")
                op.k, op.cin = 1, sum(self.map_chans[s] for s in src)
                op.out_stride = outs.pop()
                if op.cin > MAX_CHANS:
                    raise ValueError(f"{what}: cin {op.cin} > 1024")
                per = op.cin
            if op.out_stride > MAX_MAP_STRIDE:
                raise ValueError(f"{what}: stride {op.out_stride} > 256")
            nw = cout * per * op.k * op.k
            _sections(blob, total, data, ((wo, nw), (bo, 4 * cout), (mo, 4 * cout), (so, cout)), what)
            op.w, op.b, op.m, op.sh = _requant(blob, nw, wo, bo, mo, so, cout, (cout, per, op.k, op.k), what)
            self.ops.append(op)
            self.map_stride.append(op.out_stride)
            self.map_chans.append(cout)
        head = self.ops[-1]
        if head.kind != PW or head.cout != OUTPUTS or head.relu or head.out_stride != STREAM_STRIDE:
            raise ValueError("the last op is not the 16-channel head at stride 8 without ReLU")
        self.blob_bytes = total


class Model:
    """A parsed VTSP blob of version 1 or 2 (textspot2_twin.Model, kept as .t2) or 3 (multi-scale). Raises
    ValueError on anything the C loader (runtime/vocr_textspot3.c) refuses."""

    def __init__(self, blob):
        blob = bytes(blob)
        if len(blob) < HEADER or blob[:4] != MAGIC:
            raise ValueError("not a VTSP blob")
        version = struct.unpack_from("<I", blob, 4)[0]
        if version in (S.VERSION, T2.VERSION_D2S):
            m = T2.Model(blob)
            self.t2, self.version, self.head, self.half_above_h = m, m.version, m.head, 0
            for k in ("half_above", "conf_min", "net", "stride", "net_stride", "min_cells", "name", "blob_bytes"):
                setattr(self, k, getattr(m, k))
            return
        total, crc = struct.unpack_from("<II", blob, 8)
        if version != VERSION_MS or total != len(blob):
            raise ValueError(f"version {version} / size {total} != 1, 2 or {VERSION_MS} / {len(blob)}")
        if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
            raise ValueError("CRC mismatch")
        self.half_above, self.conf_min, r0, off, size, self.half_above_h, r1 = struct.unpack_from("<HBBIIHH",
                                                                                                  blob, 16)
        if r0 or r1 or off != HEADER or off + size != total or not 1 <= self.conf_min <= 127:
            raise ValueError("bad VTSP header")
        self.t2, self.version, self.head = None, VERSION_MS, "ms_x2"
        self.net = Net(blob[off:off + size])
        self.net_stride, self.stride = STREAM_STRIDE, MAP_STRIDE
        self.min_cells, self.name, self.blob_bytes = self.net.min_cells, self.net.name, total


def pack(net, half_above=480, conf_min=1, half_above_h=272, version=None):
    """A VTSP blob around a network section: version 3 for a VTMS section (half_above and half_above_h as given);
    otherwise textspot2_twin.pack (a VTDT network: version 1 or 2 by its outputs field; half_above_h unused).
    version=None: the section's magic decides."""
    net = bytes(net)
    if version is None:
        version = VERSION_MS if net[:4] == NET_MAGIC else None
    if version != VERSION_MS:
        return T2.pack(net, half_above, conf_min, version)
    Net(net)
    if not 0 <= half_above <= 0xFFFF or not 0 <= half_above_h <= 0xFFFF or not 1 <= conf_min <= 127:
        raise ValueError("half_above and half_above_h 0..65535, conf_min 1..127")
    total = HEADER + len(net)
    body = struct.pack("<HBBIIHH", half_above, conf_min, 0, HEADER, len(net), half_above_h, 0) + net
    return MAGIC + struct.pack("<III", VERSION_MS, total, zlib.crc32(body) & 0xFFFFFFFF) + body


def scale_of(model, width, height=1):
    """f: 2 when the frame is halved before the network, else 1 (versions 1 and 2: by its width alone)."""
    if model.version != VERSION_MS:
        return S.scale_of(model, width)
    wide = model.half_above and width > model.half_above
    tall = model.half_above_h and height > model.half_above_h
    return 2 if wide or tall else 1


def net_input(model, grey):
    """(the network's input image, f)."""
    g = np.ascontiguousarray(grey)
    f = scale_of(model, g.shape[1], g.shape[0])
    return (S.halve(g) if f == 2 else g), f


def map_size(h, w, stride):
    """The cells of a map at this stride for an h x w network input: (rows, columns)."""
    return -(-h // stride), -(-w // stride)


def _dense(L, x):
    """textdet_twin.run_layer for a 1x1 dense layer, its accumulator by a float64 product (exact: every product and
    partial sum is an integer below 2^24 < 2^53), the bias and requantisation in int64 as run_layer's."""
    cin, h, w = x.shape
    acc = (L.w[:, :, 0, 0].astype(np.float64) @ x.reshape(cin, -1).astype(np.float64)).astype(np.int64)
    acc = acc.reshape(L.cout, h, w) + L.b[:, None, None]
    y = TD._round_shift(acc * L.m[:, None, None], L.sh[:, None, None])
    return np.clip(y, 0 if L.relu else -128, 127)


def upsampled(m, u, rows, cols):
    """A map read through a nearest upsampling by 2^u, cropped to rows x cols: m[:, a >> u, b >> u]."""
    return m[:, (np.arange(rows) >> u)[:, None], (np.arange(cols) >> u)[None, :]]


def run_op(op, maps, h, w):
    """Op op on the maps so far (a list of (c, rows, cols) int64 codes) for an h x w network input."""
    if op.kind == DW:
        return TD.run_layer(op.layer(), maps[op.src[0]])
    rows, cols = map_size(h, w, op.out_stride)
    x = np.concatenate([upsampled(maps[s], u, rows, cols) for s, u in zip(op.src, op.up)], 0)
    return _dense(op.layer(), x)


def run_all(model, x):
    """Every map of a version-3 network on its input x (h x w uint8): [map 0, ..., the head], int64 codes."""
    g = np.asarray(x)
    if model.version != VERSION_MS:
        raise ValueError("a version-3 model")
    if g.ndim != 2 or g.dtype != np.uint8 or not g.size:
        raise ValueError("a non-empty HxW uint8 input")
    h, w = g.shape
    y = g.astype(np.int64)[None] - 128
    for L in model.net.stream:
        y = TD.run_layer(L, y)
    maps = [y]
    for op in model.net.ops:
        maps.append(run_op(op, maps, h, w))
    return maps


def run_head(model, x):
    """The network's head on its input x: version 3 (16, ceil(h/8), ceil(w/8)) codes; versions 1 and 2 as
    textspot2_twin.run_head."""
    if model.version != VERSION_MS:
        return T2.run_head(model.t2, x)
    return run_all(model, x)[-1]


def run_maps(model, x):
    """The four maps the decode reads, (4, rows, columns) int64 codes at model.stride."""
    if model.version != VERSION_MS:
        return T2.run_maps(model.t2, x)
    return T2.head_to_maps(run_head(model, x), *np.asarray(x).shape)


def spot_maps(model, maps, f, grey):
    """textspot_twin.spot after its network (textspot2_twin.spot_maps): decode, cap, dark flags, blocks, order,
    pick. A Result."""
    return T2.spot_maps(model if model.version == VERSION_MS else model.t2, maps, f, grey)


def spot(model, img):
    """The spotter on one frame (grey or RGB(A) uint8): a textspot_twin Result. Versions 1 and 2:
    textspot2_twin.spot."""
    if model.version != VERSION_MS:
        return T2.spot(model.t2, img)
    from common import grey_of
    g = np.ascontiguousarray(grey_of(img))
    if g.ndim != 2 or not g.size:
        raise ValueError("a non-empty frame")
    x, f = net_input(model, g)
    return T2.spot_maps(model, run_maps(model, x), f, g)


# the same names as textspot_twin / textspot2_twin, so a caller can switch modules
Box, Block, Result = S.Box, S.Block, S.Result
group_and_order, halve, BOX_MAX = S.group_and_order, S.halve, S.BOX_MAX
channel, head_to_maps, maps_to_head = T2.channel, T2.head_to_maps, T2.maps_to_head

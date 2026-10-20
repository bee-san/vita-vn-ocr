"""Bit-exact Python twin of runtime/vocr_textspot2.c (issue #106, docs/textspot2.md "d2s"): the full-frame text
spotter of textspot (synth/textspot_twin.py) for VTSP version 1 blobs, unchanged, and for VTSP version 2 blobs,
whose network ends in a 2 x 2 sub-pixel head (d2s): a stride-8 trunk that gives the decode stride-4 maps.

    from textspot2_twin import Model, spot
    m = Model(open("spot.vtsp", "rb").read())   # a VTSP version 1 or 2 blob
    r = spot(m, frame)                           # frame: H x W grey or H x W x 3/4 RGB(A) uint8
    r.boxes, r.blocks, r.pick, r.truncated       # textspot_twin's Result, Box and Block, the same fields

Version 1 is synth/textspot_twin.py exactly: Model keeps textspot_twin.Model(blob) as .v1, and spot() returns
textspot_twin.spot(m.v1, frame); nothing of it is re-implemented here.

Version 2, the d2s format: the smallest change of version 1 that keeps every version-1 blob, loader and result
bit for bit as it is. A version-1 loader (textspot_twin.Model, runtime/vocr_textspot.c) refuses every version-2
blob (its version field), and the version-2 network is never taken for a 4-map one (its outputs field, below).

The VTSP blob (little-endian; train/textspot2_export.py writes it), textspot_twin's 32-byte header with version 2:

    0   char[4]  magic "VTSP"
    4   u32      version: 2 = the network's head is a 2 x 2 sub-pixel head (1: textspot_twin's 4-map head)
    8   u32      total bytes
    12  u32      CRC-32 (zlib) of bytes [16, total)
    16  u16      half_above (as version 1)
    18  u8       conf_min, 1..127 (as version 1)
    19  u8       reserved (0)
    20  u32      offset of the network (32)
    24  u32      bytes of the network (total - 32)
    28  u32      reserved (0)
    32  the network: #97's VTDT layout (synth/textdet_twin.py: its 64-byte header, CRC, 32-byte layer records,
        sections and every check of textdet_twin.Model), version 1, with these header values:
          18  u8  output stride   8: the trunk's (a version-2 network is a stride-8 trunk)
          19  u8  outputs         16: the head's channels (textdet_twin.Model and every version-1 VTDT and VTSP
                                  loader refuse outputs != 4, so this network never loads as a 4-map one)
          20  u8  A codes / px    2
          21  u8  min cells       1..4; the exporter writes 2, a stride-4 map's (textdet_model.MIN_CELLS[4])
        and the layer chain multiplies to stride 8 and ends in 16 output channels without ReLU (any layer kind,
        as version 1; the exporter's head is a 1x1 conv). Its requantisation is per channel, as every layer's:
        the exporter writes K and O at 1/16 per code and A1 and A2 at 1/2 network-input px per code.

The head. Channel c = 4 (2 dy + dx) + m of network cell (i, j) (m = 0 K, 1 O, 2 A1, 3 A2; dy, dx in {0, 1}) is
map m of the sub-cell (dy, dx): cell (2 i + dy, 2 j + dx) of a map at stride s / 2 = 4 network-input px. For an
h x w network input the head is ceil(h / 8) x ceil(w / 8) cells of 16 channels, and the decode's four maps are
the ceil(h / 4) x ceil(w / 4) cells of that stride-4 map (a sub-cell row or column past it, which covers no input
pixel, is dropped):

    maps[m][I][J] = head[4 (2 (I % 2) + J % 2) + m][I // 2][J // 2],   I < ceil(h / 4), J < ceil(w / 4)

The decode is textspot_twin's on these maps with stride 4: textspot_twin.candidates (kernel cells K > 0,
4-connected components of at least the blob's min cells, conf = floor(sum K / n) >= conf_min, vertical = sum O > 0,
the doubled cell centre across the line c2 = 2 * 4 * I + 3 (A1 and A2 are 1/2 px from the sub-cell's own centre,
as a stride-4 map's), near / far / along at f s = 4 f frame px per cell, the 4 f drop, the pad, the clip), then the
cap (64), the dark flags, blocks, order and pick of textspot_twin.spot, unchanged. In code: spot() = net_input,
run_maps (textdet_twin.run_net, then head_to_maps) and spot_maps.

Model fields: version (1 or 2), head ("kroa" or "kroa_x2"), half_above, conf_min, min_cells, name, blob_bytes,
net (the network: .layers, .stride, .outputs), net_stride (the network's output stride) and stride, the stride of
the maps the decode reads: the network's for version 1, net_stride / 2 = 4 for version 2.
"""
import struct
import zlib

import numpy as np

import textdet_twin as TD
import textspot_twin as S

MAGIC = S.MAGIC
HEADER = S.HEADER
VERSION_D2S = 2
D2S_OUTPUTS = 16              # 4 sub-cells x (K, O, A1, A2)
D2S_NET_STRIDE = 8            # the trunk's stride
D2S_MAP_STRIDE = D2S_NET_STRIDE // 2
MAPS = 4                      # K, O, A1, A2


def channel(dy, dx, m):
    """The head channel of map m (0 K, 1 O, 2 A1, 3 A2) of the sub-cell (dy, dx)."""
    return 4 * (2 * dy + dx) + m


class Net:
    """A version-2 network: textdet_twin.Model's checks with outputs 16 at stride 8. Raises ValueError on anything
    the C loader refuses."""

    def __init__(self, blob):
        blob = bytes(blob)
        if len(blob) < TD.HEADER or blob[:4] != TD.MAGIC:
            raise ValueError("not a VTDT blob")
        version, total, crc = struct.unpack_from("<III", blob, 4)
        if version != TD.VERSION or total != len(blob):
            raise ValueError(f"version {version} / size {total} != {TD.VERSION} / {len(blob)}")
        if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
            raise ValueError("CRC mismatch")
        n, self.stride, self.outputs, self.a_scale, self.min_cells = struct.unpack_from("<HBBBB", blob, 16)
        if self.stride != D2S_NET_STRIDE or self.outputs != D2S_OUTPUTS or self.a_scale != 2 or \
                not 1 <= self.min_cells <= 4:
            raise ValueError("unsupported header (a d2s network: stride 8, 16 outputs, 2 A codes per px)")
        if n < 2 or n > 32 or TD.HEADER + n * TD.RECORD > total:
            raise ValueError("bad layer count")
        self.ckpt = blob[24:32].hex()
        self.name = blob[32:64].rstrip(b"\0").decode("ascii")
        self.layers = []
        stride, cin = 1, 1
        for i in range(n):
            r = struct.unpack_from("<BBBBB3xHHIIIII", blob, TD.HEADER + i * TD.RECORD)
            L = TD.Layer()
            L.kind, L.k, L.stride, L.dil, L.relu, L.cin, L.cout = r[:7]
            wo, bo, mo, so = r[7:11]
            if L.kind not in (0, 1) or L.k not in (1, 3) or L.stride not in (1, 2) or not 1 <= L.dil <= 8 \
                    or L.relu not in (0, 1) or L.cin != cin or not 1 <= L.cout <= 64:
                raise ValueError(f"layer {i}: unsupported")
            if L.kind == 1 and (L.cout != L.cin or L.k != 3):
                raise ValueError(f"layer {i}: bad depthwise layer")
            if L.stride == 2 and (L.k != 3 or L.dil != 1):
                raise ValueError(f"layer {i}: a stride-2 layer is 3x3, dilation 1")
            nw = L.cout * (1 if L.kind == 1 else L.cin) * L.k * L.k
            for off, size in ((wo, nw), (bo, 4 * L.cout), (mo, 4 * L.cout), (so, L.cout)):
                if off % 4 or off < TD.HEADER + n * TD.RECORD or off + size > total:
                    raise ValueError(f"layer {i}: section outside the blob")
            L.w = np.frombuffer(blob, np.int8, nw, wo).astype(np.int64).reshape(
                L.cout, 1 if L.kind == 1 else L.cin, L.k, L.k)
            L.b = np.frombuffer(blob, "<i4", L.cout, bo).astype(np.int64)
            L.m = np.frombuffer(blob, "<i4", L.cout, mo).astype(np.int64)
            L.sh = np.frombuffer(blob, np.uint8, L.cout, so).astype(np.int64)
            if (L.m <= 0).any() or (L.sh < 1).any() or (L.sh > 62).any():
                raise ValueError(f"layer {i}: bad requantisation")
            stride *= L.stride
            cin = L.cout
            self.layers.append(L)
        if stride != self.stride or cin != D2S_OUTPUTS or self.layers[-1].relu:
            raise ValueError("the layers do not end in the 16-channel head at stride 8")
        self.blob_bytes = total


class Model:
    """A parsed VTSP blob of version 1 (textspot_twin.Model, kept as .v1) or 2 (d2s). Raises ValueError on
    anything the C loader (runtime/vocr_textspot2.c) refuses."""

    def __init__(self, blob):
        blob = bytes(blob)
        if len(blob) < HEADER or blob[:4] != MAGIC:
            raise ValueError("not a VTSP blob")
        version = struct.unpack_from("<I", blob, 4)[0]
        if version == S.VERSION:
            m = S.Model(blob)
            self.v1, self.version, self.head = m, S.VERSION, "kroa"
            self.half_above, self.conf_min, self.net = m.half_above, m.conf_min, m.net
            self.stride, self.net_stride, self.min_cells = m.stride, m.net.stride, m.min_cells
            self.name, self.blob_bytes = m.name, m.blob_bytes
            return
        total, crc = struct.unpack_from("<II", blob, 8)
        if version != VERSION_D2S or total != len(blob):
            raise ValueError(f"version {version} / size {total} != 1 or {VERSION_D2S} / {len(blob)}")
        if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
            raise ValueError("CRC mismatch")
        self.half_above, self.conf_min, r0, off, size, r1 = struct.unpack_from("<HBBIII", blob, 16)
        if r0 or r1 or off != HEADER or off + size != total or not 1 <= self.conf_min <= 127:
            raise ValueError("bad VTSP header")
        self.v1, self.version, self.head = None, VERSION_D2S, "kroa_x2"
        self.net = Net(blob[off:off + size])
        self.net_stride, self.stride = self.net.stride, self.net.stride // 2
        self.min_cells, self.name, self.blob_bytes = self.net.min_cells, self.net.name, total


def pack(vtdt, half_above=0, conf_min=1, version=None):
    """A VTSP blob around a network blob: version 1 (textspot_twin.pack) for a 4-map VTDT, version 2 for a d2s one
    (version=None: the network's outputs field decides)."""
    vtdt = bytes(vtdt)
    if version is None:
        version = VERSION_D2S if len(vtdt) > 19 and vtdt[19] == D2S_OUTPUTS else S.VERSION
    if version == S.VERSION:
        return S.pack(vtdt, half_above, conf_min)
    if version != VERSION_D2S:
        raise ValueError(f"version {version}")
    Net(vtdt)
    if not 0 <= half_above <= 0xFFFF or not 1 <= conf_min <= 127:
        raise ValueError("half_above 0..65535, conf_min 1..127")
    total = HEADER + len(vtdt)
    body = struct.pack("<HBBIII", half_above, conf_min, 0, HEADER, len(vtdt), 0) + vtdt
    return MAGIC + struct.pack("<III", VERSION_D2S, total, zlib.crc32(body) & 0xFFFFFFFF) + body


def map_size(h, w, stride):
    """The decode maps' rows and columns for an h x w network input at the maps' stride."""
    return -(-h // stride), -(-w // stride)


def head_to_maps(head, h, w):
    """A d2s head's (16, ceil(h/8), ceil(w/8)) codes -> the four (ceil(h/4), ceil(w/4)) maps at stride 4 of an
    h x w network input (the module docstring's maps[m][I][J])."""
    head = np.asarray(head)
    if head.ndim != 3 or head.shape[0] != D2S_OUTPUTS:
        raise ValueError("a (16, rows, columns) head")
    hn, wn = head.shape[1:]
    if (hn, wn) != map_size(h, w, D2S_NET_STRIDE):
        raise ValueError(f"head {hn}x{wn} for a {h}x{w} input")
    out = np.empty((MAPS, 2 * hn, 2 * wn), head.dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            out[:, dy::2, dx::2] = head[channel(dy, dx, 0):channel(dy, dx, 0) + MAPS]
    hd, wd = map_size(h, w, D2S_MAP_STRIDE)
    return out[:, :hd, :wd]


def maps_to_head(maps, h, w):
    """The inverse of head_to_maps for hand-made stride-4 maps (4, ceil(h/4), ceil(w/4)): the d2s head whose decode
    reads them; a sub-cell past the maps (dropped by head_to_maps) gets K = -128 and 0 elsewhere."""
    maps = np.asarray(maps)
    hd, wd = map_size(h, w, D2S_MAP_STRIDE)
    if maps.shape != (MAPS, hd, wd):
        raise ValueError(f"maps {maps.shape} for a {h}x{w} input")
    hn, wn = map_size(h, w, D2S_NET_STRIDE)
    full = np.zeros((MAPS, 2 * hn, 2 * wn), maps.dtype)
    full[0] = -128
    full[:, :hd, :wd] = maps
    head = np.empty((D2S_OUTPUTS, hn, wn), maps.dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            head[channel(dy, dx, 0):channel(dy, dx, 0) + MAPS] = full[:, dy::2, dx::2]
    return head


def run_head(model, x):
    """The network on its input x (uint8): (outputs, rows, columns) int64 codes (4 maps for version 1, the 16
    channels of the d2s head for version 2)."""
    return TD.run_net(model.net, x)


def run_maps(model, x):
    """The four maps the decode reads, (4, rows, columns) int64 codes at model.stride: the head itself for version
    1, head_to_maps of it for version 2."""
    head = run_head(model, x)
    return head if model.version == S.VERSION else head_to_maps(head, *np.asarray(x).shape)


def spot_maps(model, maps, f, grey):
    """textspot_twin.spot after its network: the decode of the maps (at model.stride, frame scale f), the cap, the
    dark flags on the frame's grey, blocks, order and pick. A Result."""
    from common import line_dark
    g = np.ascontiguousarray(grey)
    H, W = g.shape
    cands = S.candidates(model, maps, f, H, W)
    truncated = int(len(cands) > S.BOX_MAX)
    kept = [c for _, c in sorted(cands)[:S.BOX_MAX]]
    with_dark = [(bx, by, bw, bh, v, conf, int(line_dark(g, (bx, by, bw, bh)))) for bx, by, bw, bh, v, conf in kept]
    boxes, blocks, pick = S.group_and_order(with_dark, H)
    return S.Result(tuple(S.Box(*b) for b in boxes), tuple(S.Block(*b) for b in blocks), pick, truncated)


def spot(model, img):
    """The spotter on one frame (grey or RGB(A) uint8): a textspot_twin Result. Version 1: textspot_twin.spot."""
    if model.version == S.VERSION:
        return S.spot(model.v1, img)
    from common import grey_of
    g = np.ascontiguousarray(grey_of(img))
    if g.ndim != 2 or not g.size:
        raise ValueError("a non-empty frame")
    x, f = S.net_input(model, g)
    return spot_maps(model, run_maps(model, x), f, g)


# the same names as textspot_twin, so a caller can switch modules
Box, Block, Result = S.Box, S.Block, S.Result
group_and_order, net_input, scale_of, halve, BOX_MAX = S.group_and_order, S.net_input, S.scale_of, S.halve, S.BOX_MAX

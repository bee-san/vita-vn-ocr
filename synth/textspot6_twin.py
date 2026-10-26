"""Bit-exact Python twin of runtime/vocr_textspot6.c (issue #128, docs/textspot6.md): the full-frame text spotter
for VTSP versions 1, 2 and 3 (synth/textspot3_twin.py's networks and decode, unchanged) and for VTSP version 4,
the TAGGED spotter. A version-4 network is a version-2 trunk (a VTDT network) or a version-3 multi-scale network (a
VTMS section) whose d2s head carries, beside version 2's 16 channels, T tag maps per sub-cell. The decode gives
every box a class, joins only boxes of one class into a block, picks the dialogue block and reports the frame's
dialogue presence:

    from textspot6_twin import Model, spot
    m = Model(open("spot.vtsp", "rb").read())   # a VTSP version 1, 2, 3 or 4 blob
    r = spot(m, frame)                           # frame: H x W grey or H x W x 3/4 RGB(A) uint8
    r.boxes        # [Box(x, y, w, h, vertical, conf, dark, block, tag, tconf)], in reading order
    r.blocks       # [Block(x, y, w, h, vertical, first, count, tag)], in reading order
    r.pick         # the picked dialogue block (tag 0), -1 when there is none
    r.truncated    # 1 when candidates were dropped (more than BOX_MAX)
    r.presence     # the largest confidence of a tag-0 box (1..127), 0 when there is none

The classes (the exporter's convention, CLASSES): 0 dialogue (and narration), 1 speaker name, 2 choice, 3 UI text,
4 scene text (signs, writing in a CG). A version-4 blob with T tag maps has the classes 0..T.

VERSIONS 1, 2 AND 3. synth/textspot3_twin.py's Model, networks and decode, with every box of tag 0 and tconf 0.
Blocks, order and pick are this module's (below), which with every tag 0 are textspot_twin.group_and_order's
exactly; so boxes, blocks, order, pick and truncated equal textspot3_twin.spot's, and presence is the largest box
confidence (0 without a box).

VERSION 4. The VTSP blob (little-endian; train/textspot6_model.py writes it):

    0   char[4]  magic "VTSP"
    4   u32      version: 4
    8   u32      total bytes
    12  u32      CRC-32 (zlib) of bytes [16, total)
    16  u16      half_above: a frame wider than this many px is halved (0: the width never halves it)
    18  u8       conf_min, 1..127 (as version 1)
    19  u8       tags T, 1..4: the head's tag maps (classes 0..T)
    20  u32      offset of the network (32)
    24  u32      bytes of the network (total - 32)
    28  u16      half_above_h: a frame taller than this many px is halved (0: the height never halves it)
    30  u16      reserved (0)
    32  the network, by its magic:
          "VTDT"  a trunk: synth/textdet_twin.py's layout and every check of textspot2_twin.Net (stride 8, 2 A codes
                  per px, min cells 1..4, layers chained to stride 8, at most 64 channels), with outputs 16 + 4 T in
                  its header and as the last layer's cout (no ReLU)
          "VTMS"  a multi-scale network: synth/textspot3_twin.py's VTMS section and every check of textspot3_twin.Net,
                  with outputs 16 + 4 T in its header and as the head op's cout (a 1x1 op at stride 8, no ReLU);
                  16 + 4 T must be a multiple of 8 (T = 2 or 4), as every version-3 op's cout

The head. For an h x w network input the head is (16 + 4 T, ceil(h / 8), ceil(w / 8)) codes. Channels 0..15 are
version 2's d2s head exactly (channel 4 (2 dy + dx) + m is map m, 0 K, 1 O, 2 A1, 3 A2, of the sub-cell (dy, dx)).
Channel 16 + T (2 dy + dx) + t is tag map t (t = 0..T-1: class t + 1's logit against class 0's) of the sub-cell
(dy, dx). The decode reads 4 + T maps at stride 4, cropped to ceil(h / 4) x ceil(w / 4) (head_to_maps):

    maps[m][I][J]     = head[4 (2 (I % 2) + J % 2) + m][I // 2][J // 2]        m = 0..3
    maps[4 + t][I][J] = head[16 + T (2 (I % 2) + J % 2) + t][I // 2][J // 2]   t = 0..T-1

The exporter writes K, O and the tag maps at 1/16 per code, A1 and A2 at 1/2 network-input px per code.

The decode (integers only; frame W x H, f the scale of textspot3_twin.scale_of with this header's half_above and
half_above_h). textspot_twin.candidates on maps 0..3 at stride 4, unchanged (kernel cells K > 0, 4-connected
components of at least min cells, conf = floor(sum K / n), dropped below conf_min, vertical = sum O > 0, near / far /
along, the 4 f drop, the pad, the clip), and per kept component of n cells:

    S_t   = the sum of tag map t's codes over its cells (t = 0..T-1)
    score = (0, S_0, ..., S_{T-1}): class 0's score is 0, class t + 1's is S_t
    tag   = the class of the largest score (the lowest class on a tie)
    tconf = min(127, floor((best - second) / n)), best and second the two largest scores (equal scores: 0)

Then:

    caps      at most BOX_MAX = 64 candidates: the smallest by (-conf, -n, vertical, y, x, h, w, tag); truncated = 1
              when any was dropped
    dark      synth/common.line_dark on the frame's grey in the box (vocr_line_dark)
    blocks    textspot_twin's rule plus one condition: two boxes join a block when they have one orientation AND
              one tag, overlap along the line, the gap between them across the line is at most the larger one's
              size across, and the larger size across is at most 3/2 of the smaller (union-find over every pair of
              the boxes sorted by (x, y, w, h, vertical, conf, dark, tag, tconf)); a block's tag is its boxes'
    order     textspot_twin's inside a block (horizontal lines top to bottom in rows, a row left to right; vertical
              columns right to left in groups, a group top to bottom). Blocks go top to bottom in rows of blocks
              (sorted by (y, x, h, w, tag); a block joins the current row when its rows overlap the row's first
              block's by at least half the smaller height); a row is sorted by (x, y, h, w, tag), or by (-(x + w), y,
              h, w, tag) when every block in it is vertical
    pick      among the blocks of tag 0: the largest score = sum of its boxes' w h, doubled when the block's centre
              is in the lower half of the frame (2 y + h >= H); ties: the larger top y, then the earlier block; -1
              when no block has tag 0
    presence  the largest conf of a box of tag 0; 0 when there is none

Model fields: version (1..4), head ("kroa", "kroa_x2", "ms_x2", or "tag_x2" / "tagms_x2" for version 4), tags (T;
0 for versions 1-3), form (version 4: "trunk" or "ms"), half_above, half_above_h, conf_min, min_cells, name,
blob_bytes, net, net_stride (8 for versions 2-4) and stride (the decode maps' stride: 4 for versions 2-4).
"""
import struct
import zlib
from dataclasses import dataclass

import numpy as np

import textdet_twin as TD
import textspot2_twin as T2
import textspot3_twin as T3
import textspot_twin as S

MAGIC = S.MAGIC
HEADER = S.HEADER                    # 32
VERSION_TAG = 4
MAX_TAGS = 4
CLASSES = ("dialogue", "name", "choice", "ui", "scene")
DIALOGUE = 0
BASE_OUTPUTS = T2.D2S_OUTPUTS        # 16: K, O, A1, A2 per sub-cell
NET_STRIDE = T2.D2S_NET_STRIDE       # 8
MAP_STRIDE = T2.D2S_MAP_STRIDE       # 4
BOX_MAX = S.BOX_MAX                  # 64
TCONF_MAX = 127


def outputs_of(tags):
    """The head's channels for T tag maps."""
    return BASE_OUTPUTS + 4 * tags


def tag_channel(dy, dx, t, tags):
    """The head channel of tag map t of the sub-cell (dy, dx)."""
    return BASE_OUTPUTS + tags * (2 * dy + dx) + t


# ----------------------------------------------------------------------------- the version-4 networks

class TrunkNet:
    """A version-4 trunk: textspot2_twin.Net's checks with outputs 16 + 4 T. Raises ValueError on anything the C
    loader refuses."""

    def __init__(self, blob, tags):
        blob = bytes(blob)
        if len(blob) < TD.HEADER or blob[:4] != TD.MAGIC:
            raise ValueError("not a VTDT blob")
        version, total, crc = struct.unpack_from("<III", blob, 4)
        if version != TD.VERSION or total != len(blob):
            raise ValueError(f"version {version} / size {total} != {TD.VERSION} / {len(blob)}")
        if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
            raise ValueError("CRC mismatch")
        n, self.stride, self.outputs, self.a_scale, self.min_cells = struct.unpack_from("<HBBBB", blob, 16)
        if self.stride != NET_STRIDE or self.outputs != outputs_of(tags) or self.a_scale != 2 or \
                not 1 <= self.min_cells <= 4:
            raise ValueError(f"unsupported header (a tagged trunk: stride 8, {outputs_of(tags)} outputs, 2 A codes "
                             "per px, min cells 1..4)")
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
        if stride != self.stride or cin != outputs_of(tags) or self.layers[-1].relu:
            raise ValueError(f"the layers do not end in the {outputs_of(tags)}-channel head at stride 8")
        self.blob_bytes = total


def _ms_net(blob, tags):
    """A version-4 multi-scale network: textspot3_twin.Net's checks with outputs 16 + 4 T (a multiple of 8). The
    section is checked by textspot3_twin.Net on a copy whose header says 16 outputs and whose head op is cut to its
    first 16 channels; then the head op's own record, sections and requantisation are read here. Raises
    ValueError on anything the C loader refuses."""
    blob = bytes(blob)
    out = outputs_of(tags)
    if out % 8:
        raise ValueError(f"a multi-scale head of {out} channels (16 + 4 T must be a multiple of 8: T = 2 or 4)")
    if len(blob) < T3.NET_HEADER or blob[:4] != T3.NET_MAGIC:
        raise ValueError("not a VTMS section")
    version, total, crc = struct.unpack_from("<III", blob, 4)
    if version != T3.NET_VERSION or total != len(blob):
        raise ValueError(f"version {version} / size {total} != {T3.NET_VERSION} / {len(blob)}")
    if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
        raise ValueError("CRC mismatch")
    n_s, stride, outputs, a_scale, min_cells, n_o = struct.unpack_from("<HBBBBH", blob, 16)
    if outputs != out:
        raise ValueError(f"the section's header says {outputs} outputs, the VTSP header {out}")
    if not 1 <= n_s <= T3.MAX_STREAM or not 1 <= n_o <= T3.MAX_OPS or \
            T3.NET_HEADER + (n_s + n_o) * T3.RECORD > total:
        raise ValueError("bad layer or op count")
    head_rec = T3.NET_HEADER + (n_s + n_o - 1) * T3.RECORD
    kind, st, dil, relu, nsrc, r0, cout = struct.unpack_from("<BBBBBBH", blob, head_rec)
    if kind != T3.PW or cout != out or relu:
        raise ValueError(f"the last op is not the {out}-channel head (a 1x1 op without ReLU)")
    # the copy: header outputs 16, head op cout 16 (its sections stay where they are; the first 16 channels of each
    # are the same bytes), CRC recomputed
    cp = bytearray(blob)
    cp[19] = BASE_OUTPUTS
    struct.pack_into("<H", cp, head_rec + 6, BASE_OUTPUTS)
    struct.pack_into("<I", cp, 12, zlib.crc32(bytes(cp[16:])) & 0xFFFFFFFF)
    net = T3.Net(bytes(cp))
    head = net.ops[-1]
    r1, wo, bo, mo, so = struct.unpack_from("<HIIII", blob, head_rec + 14)
    data = T3.NET_HEADER + (n_s + n_o) * T3.RECORD
    nw = out * head.cin
    T3._sections(blob, total, data, ((wo, nw), (bo, 4 * out), (mo, 4 * out), (so, out)), f"op {n_o - 1}")
    head.w, head.b, head.m, head.sh = T3._requant(blob, nw, wo, bo, mo, so, out, (out, head.cin, 1, 1),
                                                  f"op {n_o - 1}")
    head.cout = out
    net.outputs = out
    net.map_chans[-1] = out
    net.blob_bytes = total
    return net


class Model:
    """A parsed VTSP blob of version 1, 2 or 3 (textspot3_twin.Model, kept as .t3) or 4 (tagged). Raises
    ValueError on anything the C loader (runtime/vocr_textspot6.c) refuses."""

    def __init__(self, blob):
        blob = bytes(blob)
        if len(blob) < HEADER or blob[:4] != MAGIC:
            raise ValueError("not a VTSP blob")
        version = struct.unpack_from("<I", blob, 4)[0]
        if version in (S.VERSION, T2.VERSION_D2S, T3.VERSION_MS):
            m = T3.Model(blob)
            self.t3, self.version, self.head, self.tags, self.form = m, m.version, m.head, 0, None
            for k in ("half_above", "half_above_h", "conf_min", "net", "stride", "net_stride", "min_cells", "name",
                      "blob_bytes"):
                setattr(self, k, getattr(m, k))
            return
        total, crc = struct.unpack_from("<II", blob, 8)
        if version != VERSION_TAG or total != len(blob):
            raise ValueError(f"version {version} / size {total} != 1..{VERSION_TAG} / {len(blob)}")
        if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
            raise ValueError("CRC mismatch")
        self.half_above, self.conf_min, self.tags, off, size, self.half_above_h, r1 = \
            struct.unpack_from("<HBBIIHH", blob, 16)
        if r1 or off != HEADER or off + size != total or not 1 <= self.conf_min <= 127 or \
                not 1 <= self.tags <= MAX_TAGS:
            raise ValueError("bad VTSP header")
        net = blob[off:off + size]
        self.t3, self.version = None, VERSION_TAG
        if net[:4] == TD.MAGIC:
            self.form, self.head = "trunk", "tag_x2"
            self.net = TrunkNet(net, self.tags)
        elif net[:4] == T3.NET_MAGIC:
            self.form, self.head = "ms", "tagms_x2"
            self.net = _ms_net(net, self.tags)
        else:
            raise ValueError("the network is neither a VTDT nor a VTMS section")
        self.net_stride, self.stride = NET_STRIDE, MAP_STRIDE
        self.min_cells, self.name, self.blob_bytes = self.net.min_cells, self.net.name, total


def pack(net, tags, half_above=480, conf_min=1, half_above_h=0):
    """A VTSP version-4 blob around a tagged network (a VTDT trunk or a VTMS section with 16 + 4 tags outputs)."""
    net = bytes(net)
    if not 1 <= tags <= MAX_TAGS or not 0 <= half_above <= 0xFFFF or not 0 <= half_above_h <= 0xFFFF or \
            not 1 <= conf_min <= 127:
        raise ValueError("tags 1..4, half_above and half_above_h 0..65535, conf_min 1..127")
    if net[:4] == TD.MAGIC:
        TrunkNet(net, tags)
    elif net[:4] == T3.NET_MAGIC:
        _ms_net(net, tags)
    else:
        raise ValueError("a VTDT or VTMS network")
    total = HEADER + len(net)
    body = struct.pack("<HBBIIHH", half_above, conf_min, tags, HEADER, len(net), half_above_h, 0) + net
    return MAGIC + struct.pack("<III", VERSION_TAG, total, zlib.crc32(body) & 0xFFFFFFFF) + body


# ----------------------------------------------------------------------------- the input and the maps

def scale_of(model, width, height=1):
    """f: 2 when the frame is halved before the network, else 1 (versions 1 and 2: by the width alone)."""
    if model.version in (S.VERSION, T2.VERSION_D2S):
        return S.scale_of(model.t3.t2, width)
    wide = model.half_above and width > model.half_above
    tall = model.half_above_h and height > model.half_above_h
    return 2 if wide or tall else 1


def net_input(model, grey):
    """(the network's input image, f)."""
    g = np.ascontiguousarray(grey)
    f = scale_of(model, g.shape[1], g.shape[0])
    return (S.halve(g) if f == 2 else g), f


def run_head(model, x):
    """The network's head on its input x (h x w uint8): (outputs, rows, columns) int64 codes."""
    if model.version != VERSION_TAG:
        return T3.run_head(model.t3, x)
    g = np.asarray(x)
    if g.ndim != 2 or g.dtype != np.uint8 or not g.size:
        raise ValueError("a non-empty HxW uint8 input")
    if model.form == "trunk":
        return TD.run_net(model.net, g)
    h, w = g.shape
    y = g.astype(np.int64)[None] - 128
    for L in model.net.stream:
        y = TD.run_layer(L, y)
    maps = [y]
    for op in model.net.ops:
        maps.append(T3.run_op(op, maps, h, w))
    return maps[-1]


def head_to_maps(head, h, w, tags):
    """A tagged head's (16 + 4 T, ceil(h/8), ceil(w/8)) codes -> the 4 + T decode maps (ceil(h/4), ceil(w/4)) at
    stride 4 of an h x w network input (the module docstring's maps)."""
    head = np.asarray(head)
    if head.ndim != 3 or head.shape[0] != outputs_of(tags):
        raise ValueError(f"a ({outputs_of(tags)}, rows, columns) head")
    hn, wn = head.shape[1:]
    if (hn, wn) != T2.map_size(h, w, NET_STRIDE):
        raise ValueError(f"head {hn}x{wn} for a {h}x{w} input")
    out = np.empty((4 + tags, 2 * hn, 2 * wn), head.dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            c = T2.channel(dy, dx, 0)
            out[:4, dy::2, dx::2] = head[c:c + 4]
            c = tag_channel(dy, dx, 0, tags)
            out[4:, dy::2, dx::2] = head[c:c + tags]
    hd, wd = T2.map_size(h, w, MAP_STRIDE)
    return out[:, :hd, :wd]


def maps_to_head(maps, h, w, tags):
    """The inverse of head_to_maps for hand-made stride-4 maps (4 + T, ceil(h/4), ceil(w/4)): the head whose decode
    reads them; a sub-cell past the maps gets K = -128 and 0 elsewhere."""
    maps = np.asarray(maps)
    hd, wd = T2.map_size(h, w, MAP_STRIDE)
    if maps.shape != (4 + tags, hd, wd):
        raise ValueError(f"maps {maps.shape} for a {h}x{w} input with {tags} tags")
    hn, wn = T2.map_size(h, w, NET_STRIDE)
    full = np.zeros((4 + tags, 2 * hn, 2 * wn), maps.dtype)
    full[0] = -128
    full[:, :hd, :wd] = maps
    head = np.empty((outputs_of(tags), hn, wn), maps.dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            c = T2.channel(dy, dx, 0)
            head[c:c + 4] = full[:4, dy::2, dx::2]
            c = tag_channel(dy, dx, 0, tags)
            head[c:c + tags] = full[4:, dy::2, dx::2]
    return head


def run_maps(model, x):
    """The maps the decode reads, (4 + T, rows, columns) int64 codes at model.stride (T = 0 for versions 1-3)."""
    if model.version != VERSION_TAG:
        return T3.run_maps(model.t3, x)
    return head_to_maps(run_head(model, x), *np.asarray(x).shape, model.tags)


# ----------------------------------------------------------------------------- the decode

def _tag_of(sums, n):
    """(tag, tconf) of a component of n cells with tag sums S_0..S_{T-1} (the module docstring)."""
    scores = [0] + [int(v) for v in sums]
    best = 0
    for c in range(1, len(scores)):
        if scores[c] > scores[best]:
            best = c
    if len(scores) == 1:
        return 0, 0
    second = max(scores[c] for c in range(len(scores)) if c != best)
    return best, min(TCONF_MAX, (scores[best] - second) // n)


def candidates(model, maps, f, height, width):
    """Every kept candidate before the cap: [(key, (x, y, w, h, vertical, conf, tag, tconf))]; key = the cap's sort
    key. textspot_twin.candidates' boxes, line for line, with the tag of each component."""
    s = model.stride
    tags = maps.shape[0] - 4
    kmap, omap, a1map, a2map = (maps[c] for c in range(4))
    out = []
    for runs in TD.components(kmap > 0):
        n = sum(b - a for _, a, b in runs)
        if n < model.min_cells:
            continue
        rows = np.concatenate([np.full(b - a, i, np.int64) for i, a, b in runs])
        cols = np.concatenate([np.arange(a, b, dtype=np.int64) for _, a, b in runs])
        conf = int(kmap[rows, cols].sum()) // n
        if conf < model.conf_min:
            continue
        vertical = int(omap[rows, cols].sum()) > 0
        a1, a2 = a1map[rows, cols], a2map[rows, cols]
        across_idx, along_idx = (cols, rows) if vertical else (rows, cols)
        c2 = 2 * s * across_idx + s - 1
        near = (f * int((c2 - a1).sum())) // (2 * n)
        far = -((-(f * int((c2 + a2).sum()))) // (2 * n))
        lo, hi = f * s * int(along_idx.min()), f * s * (int(along_idx.max()) + 1)
        across = far - near
        if across < S.MIN_ACROSS * f:
            continue
        pad = max(S.PAD_MIN, (across * S.PAD_PCT + 50) // 100)
        a0, a1_, b0, b1 = near - pad, far + pad, lo - pad, hi + pad
        if vertical:
            x0, x1, y0, y1 = a0, a1_, b0, b1
        else:
            x0, x1, y0, y1 = b0, b1, a0, a1_
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        tag, tconf = _tag_of([int(maps[4 + t][rows, cols].sum()) for t in range(tags)], n)
        key = (-conf, -n, int(vertical), y0, x0, y1 - y0, x1 - x0, tag)
        out.append((key, (x0, y0, x1 - x0, y1 - y0, int(vertical), conf, tag, tconf)))
    return out


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    vertical: int
    conf: int
    dark: int
    block: int
    tag: int
    tconf: int


@dataclass(frozen=True)
class Block:
    x: int
    y: int
    w: int
    h: int
    vertical: int
    first: int
    count: int
    tag: int


@dataclass(frozen=True)
class Result:
    boxes: tuple
    blocks: tuple
    pick: int
    truncated: int
    presence: int


# a box in the blocks' code: (x, y, w, h, vertical, conf, dark, tag, tconf); X, Y, W, H, V, TAG index it
X, Y, W, H, V, CONF, DARK, TAG, TCONF = range(9)


def _joins(a, b):
    """textspot_twin._joins plus one tag."""
    return a[TAG] == b[TAG] and S._joins(a, b)


def group_and_order(boxes, height):
    """boxes: [(x, y, w, h, vertical, conf, dark, tag, tconf)] -> (ordered boxes [(..., block)], blocks
    [(x, y, w, h, vertical, first, count, tag)], pick). With every tag equal it is textspot_twin.group_and_order's
    result for the same boxes (their first seven fields). Deterministic for any input order of distinct boxes."""
    boxes = sorted(tuple(b) for b in boxes)
    n = len(boxes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if _joins(boxes[i], boxes[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    blocks = []
    for members in groups.values():
        x0 = min(b[X] for b in members)
        y0 = min(b[Y] for b in members)
        x1 = max(b[X] + b[W] for b in members)
        y1 = max(b[Y] + b[H] for b in members)
        blocks.append(((x0, y0, x1 - x0, y1 - y0, members[0][V], members[0][TAG]), S._order_block(members)))
    rows, run = [], []
    for bk in sorted(blocks, key=lambda t: (t[0][1], t[0][0], t[0][3], t[0][2], t[0][5])):
        if run:
            f = run[0][0]
            b = bk[0]
            ov = min(b[1] + b[3], f[1] + f[3]) - max(b[1], f[1])
            if 2 * ov >= min(b[3], f[3]):
                run.append(bk)
                continue
            rows.append(run)
        run = [bk]
    if run:
        rows.append(run)
    ordered = []
    for run in rows:
        if all(t[0][4] for t in run):
            ordered += sorted(run, key=lambda t: (-(t[0][0] + t[0][2]), t[0][1], t[0][3], t[0][2], t[0][5]))
        else:
            ordered += sorted(run, key=lambda t: (t[0][0], t[0][1], t[0][3], t[0][2], t[0][5]))
    out_boxes, out_blocks = [], []
    for k, (bb, members) in enumerate(ordered):
        out_blocks.append((*bb[:5], len(out_boxes), len(members), bb[5]))
        out_boxes += [(*m, k) for m in members]
    pick, best = -1, None
    for k, (bb, members) in enumerate(ordered):
        if bb[5] != DIALOGUE:
            continue
        score = sum(m[W] * m[H] for m in members) * (2 if 2 * bb[1] + bb[3] >= height else 1)
        key = (score, bb[1])
        if best is None or key > best:
            pick, best = k, key
    return out_boxes, out_blocks, pick


def spot_maps(model, maps, f, grey):
    """The decode of the maps (4 + T at model.stride, frame scale f), the cap, the dark flags on the frame's grey,
    blocks, order, pick and presence. A Result."""
    from common import line_dark
    g = np.ascontiguousarray(grey)
    H_, W_ = g.shape
    maps = np.asarray(maps)
    cands = candidates(model, maps, f, H_, W_)
    truncated = int(len(cands) > BOX_MAX)
    kept = [c for _, c in sorted(cands)[:BOX_MAX]]
    boxes6 = [(bx, by, bw, bh, v, conf, int(line_dark(g, (bx, by, bw, bh))), tag, tconf)
              for bx, by, bw, bh, v, conf, tag, tconf in kept]
    boxes, blocks, pick = group_and_order(boxes6, H_)
    presence = max([b[CONF] for b in boxes if b[TAG] == DIALOGUE], default=0)
    return Result(tuple(Box(x, y, w, h, v, c, d, blk, t, tc) for x, y, w, h, v, c, d, t, tc, blk in boxes),
                  tuple(Block(*b) for b in blocks), pick, truncated, presence)


def spot(model, img):
    """The spotter on one frame (grey or RGB(A) uint8): a Result."""
    from common import grey_of
    g = np.ascontiguousarray(grey_of(img))
    if g.ndim != 2 or not g.size:
        raise ValueError("a non-empty frame")
    x, f = net_input(model, g)
    return spot_maps(model, run_maps(model, x), f, g)


def from_v3(result, presence=None):
    """A textspot3_twin Result as this module's (every tag 0, tconf 0; presence the largest box confidence)."""
    boxes = tuple(Box(b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, 0, 0) for b in result.boxes)
    blocks = tuple(Block(b.x, b.y, b.w, b.h, b.vertical, b.first, b.count, 0) for b in result.blocks)
    if presence is None:
        presence = max([b.conf for b in result.boxes], default=0)
    return Result(boxes, blocks, result.pick, result.truncated, presence)

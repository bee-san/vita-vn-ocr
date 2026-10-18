"""Bit-exact Python twin of the learned text detector's C runtime (runtime/vocr_textdet.c; docs/textdet.md).

The detector reads a grey region (synth/common.to_gray of RGBA, as every finder does), runs an int8 network with
integer arithmetic only and decodes its output maps into line boxes, top to bottom:

    from textdet_twin import Model, find_lines
    m = Model(open("det.vtd", "rb").read())
    lines, truncated = find_lines(m, grey)            # [(x, y, w, h, dark, vertical)], truncated flag

The blob (little-endian; train/textdet_export.py writes it):

    0   char[4]  magic "VTDT"
    4   u32      version (1)
    8   u32      total bytes
    12  u32      CRC-32 (zlib) of bytes [16, total)
    16  u16      layers
    18  u8       output stride (4 or 8)
    19  u8       outputs (4: K, O, A1, A2)
    20  u8       A codes per pixel (2)
    21  u8       min kernel cells of a line (2 at stride 4, 1 at stride 8)
    22  u16      reserved (0)
    24  u8[8]    sha256 prefix of the source checkpoint
    32  char[32] shape name, NUL padded
    64  layer records, 32 bytes each:
          u8 kind (0: dense conv, 1: depthwise), u8 k (1 or 3), u8 stride (1 or 2), u8 dilation,
          u8 relu (1: outputs clamped to [0, 127], 0: [-128, 127]), u8[3] reserved,
          u16 cin, u16 cout, u32 weights offset, u32 bias offset, u32 multiplier offset, u32 shift offset,
          u32 reserved
    data: per layer int8 weights [cout][cin or 1][k][k], int32 bias[cout], int32 multiplier[cout] (> 0),
          u8 shift[cout] (1..62), each section starting on a 4-byte boundary

The network, layer by layer (the region is h x w; a layer's output is ceil(h/stride) x ceil(w/stride)):

    x0[y][x]        = grey - 128                                         (int8)
    acc[c][i][j]    = bias[c] + sum w[c][q][ky][kx] * x[q][i*s + (ky - (k-1)/2)*d][j*s + (kx - (k-1)/2)*d]
                      over the inputs q (all of them, or only q = c for a depthwise layer); positions outside the
                      input are 0                                         (int32)
    out[c][i][j]    = clamp(floor((acc * mult[c] + 2^(shift[c]-1)) / 2^shift[c]), relu ? 0 : -128, 127)

The last layer gives four int8 maps at the output stride s: K (text kernel logit), O (vertical logit), A1 and A2
(distances from the cell centre to the line's near and far edges across the line, in 1/2 px). Decode:

    kernel cells: K > 0; 4-connected components
    a component of n cells (n >= min cells) with cell centres c2 = 2*s*i + s - 1 (twice the centre, across the line):
        vertical  = sum(O) > 0
        near      = floor(sum(c2 - A1) / 2n),  far = ceil(sum(c2 + A2) / 2n)          (across the line)
        along     = [first cell * s, (last cell + 1) * s)
        dropped when far - near < 4
        pad       = max(2, ((far - near) * 18 + 50) / 100), on every side, then clipped to the region
    order: horizontal lines by (y, x, h, w), gathered into rows (a line joins the current row when its rows overlap
    the row's first line by at least half the smaller height; a row is sorted by (x, y, h, w)); then vertical lines
    right to left, by (-(x + w), y, x, w, h). At most 64 candidates are kept (the smallest by (vertical, y, x, h, w)),
    then at most VOCR_LINES_MAX = 16 lines (truncated is set when any was dropped). dark = common.line_dark on the
    box.
"""
import struct
import zlib

import numpy as np

MAGIC = b"VTDT"
VERSION = 1
HEADER = 64
RECORD = 32
LINES_MAX = 16
CANDIDATES_MAX = 64
KERNEL, ORIENT, NEAR, FAR = 0, 1, 2, 3
PAD_PCT, PAD_MIN, MIN_ACROSS = 18, 2, 4


class Layer:
    __slots__ = ("kind", "k", "stride", "dil", "relu", "cin", "cout", "w", "b", "m", "sh")


class Model:
    """A parsed blob. Raises ValueError on anything the C loader would refuse."""

    def __init__(self, blob):
        blob = bytes(blob)
        if len(blob) < HEADER or blob[:4] != MAGIC:
            raise ValueError("not a VTDT blob")
        version, total, crc = struct.unpack_from("<III", blob, 4)
        if version != VERSION or total != len(blob):
            raise ValueError(f"version {version} / size {total} != {VERSION} / {len(blob)}")
        if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
            raise ValueError("CRC mismatch")
        n, self.stride, n_out, self.a_scale, self.min_cells = struct.unpack_from("<HBBBB", blob, 16)
        if self.stride not in (4, 8) or n_out != 4 or self.a_scale != 2 or not 1 <= self.min_cells <= 4:
            raise ValueError("unsupported header")
        if n < 2 or n > 32 or HEADER + n * RECORD > total:
            raise ValueError("bad layer count")
        self.ckpt = blob[24:32].hex()
        self.name = blob[32:64].rstrip(b"\0").decode("ascii")
        self.layers = []
        stride = 1
        cin = 1
        for i in range(n):
            r = struct.unpack_from("<BBBBB3xHHIIIII", blob, HEADER + i * RECORD)
            L = Layer()
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
                if off % 4 or off < HEADER + n * RECORD or off + size > total:
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
        if stride != self.stride or cin != 4 or self.layers[-1].relu:
            raise ValueError("the layers do not end in the 4-map head at the output stride")
        self.blob_bytes = total


def _round_shift(v, sh):
    """floor((v + 2^(sh-1)) / 2^sh) for int64 arrays (numpy's >> floors negative values)."""
    return (v + (np.int64(1) << (sh - 1))) >> sh


def run_layer(L, x):
    """One layer on x (cin, h, w) int64 codes -> (cout, ceil(h/s), ceil(w/s)) int64 codes."""
    cin, h, w = x.shape
    s, d, k = L.stride, L.dil, L.k
    p = (k - 1) // 2 * d
    ho, wo = (h + s - 1) // s, (w + s - 1) // s
    xp = np.pad(x, ((0, 0), (p, p), (p, p))) if p else x
    acc = np.zeros((L.cout, ho, wo), np.int64)
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, ky * d: ky * d + s * (ho - 1) + 1: s, kx * d: kx * d + s * (wo - 1) + 1: s]
            if L.kind == 1:
                acc += L.w[:, 0, ky, kx][:, None, None] * patch
            else:
                acc += np.tensordot(L.w[:, :, ky, kx], patch, axes=(1, 0))
    acc += L.b[:, None, None]
    y = _round_shift(acc * L.m[:, None, None], L.sh[:, None, None])
    return np.clip(y, 0 if L.relu else -128, 127)


def run_net(model, grey):
    """The four output maps (4, ceil(h/s), ceil(w/s)) as int64 codes."""
    g = np.asarray(grey)
    if g.ndim != 2 or g.dtype != np.uint8 or not g.size:
        raise ValueError("a non-empty HxW uint8 grey region")
    x = g.astype(np.int64)[None] - 128
    for L in model.layers:
        x = run_layer(L, x)
    return x


def _floor_div(a, b):
    return a // b


def _ceil_div(a, b):
    return -((-a) // b)


def components(kmask):
    """4-connected components of a boolean map: list of (rows, cols) index arrays, in no particular order."""
    h, w = kmask.shape
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    runs_prev, label = [], {}
    runs_all = []
    for i in range(h):
        row = kmask[i]
        runs = []
        j = 0
        while j < w:
            if row[j]:
                j0 = j
                while j < w and row[j]:
                    j += 1
                rid = len(runs_all)
                runs_all.append((i, j0, j))
                parent[rid] = rid
                runs.append(rid)
            else:
                j += 1
        for r in runs:
            _, a0, a1 = runs_all[r]
            for q in runs_prev:
                _, b0, b1 = runs_all[q]
                if a0 < b1 and b0 < a1:
                    ra, rb = find(r), find(q)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        runs_prev = runs
    for r in range(len(runs_all)):
        label.setdefault(find(r), []).append(runs_all[r])
    return list(label.values())


def decode(model, maps, height, width):
    """Line boxes from the output maps: ([(x, y, w, h, vertical)], truncated), in reading order."""
    s = model.stride
    kmap, omap, a1map, a2map = (maps[c] for c in range(4))
    cands = []
    for runs in components(kmap > 0):
        n = sum(b - a for _, a, b in runs)
        if n < model.min_cells:
            continue
        rows = np.concatenate([np.full(b - a, i, np.int64) for i, a, b in runs])
        cols = np.concatenate([np.arange(a, b, dtype=np.int64) for _, a, b in runs])
        o = int(omap[rows, cols].sum())
        a1, a2 = a1map[rows, cols], a2map[rows, cols]
        vertical = o > 0
        if vertical:
            c2 = 2 * s * cols + s - 1
            lo, hi = int(rows.min()) * s, (int(rows.max()) + 1) * s
        else:
            c2 = 2 * s * rows + s - 1
            lo, hi = int(cols.min()) * s, (int(cols.max()) + 1) * s
        near = _floor_div(int((c2 - a1).sum()), 2 * n)
        far = _ceil_div(int((c2 + a2).sum()), 2 * n)
        across = far - near
        if across < MIN_ACROSS:
            continue
        pad = max(PAD_MIN, (across * PAD_PCT + 50) // 100)
        a0, a1_, b0, b1 = near - pad, far + pad, lo - pad, hi + pad
        if vertical:
            x0, x1, y0, y1 = a0, a1_, b0, b1
        else:
            x0, x1, y0, y1 = b0, b1, a0, a1_
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        cands.append((int(vertical), y0, x0, y1 - y0, x1 - x0))
    truncated = len(cands) > CANDIDATES_MAX
    cands = sorted(cands)[:CANDIDATES_MAX]
    horiz = sorted(((x, y, w, h) for v, y, x, h, w in cands if not v), key=lambda b: (b[1], b[0], b[3], b[2]))
    ordered = []
    row = []
    for b in horiz:
        if row:
            f = row[0]
            ov = min(b[1] + b[3], f[1] + f[3]) - max(b[1], f[1])
            if 2 * ov >= min(b[3], f[3]):
                row.append(b)
                continue
            ordered += sorted(row, key=lambda r: (r[0], r[1], r[3], r[2]))
        row = [b]
    ordered += sorted(row, key=lambda r: (r[0], r[1], r[3], r[2]))
    out = [(x, y, w, h, 0) for x, y, w, h in ordered]
    vert = sorted(((x, y, w, h) for v, y, x, h, w in cands if v),
                  key=lambda b: (-(b[0] + b[2]), b[1], b[0], b[2], b[3]))
    out += [(x, y, w, h, 1) for x, y, w, h in vert]
    if len(out) > LINES_MAX:
        out, truncated = out[:LINES_MAX], True
    return out, truncated


def find_lines(model, img):
    """The detector on one region (grey or RGB(A) uint8): ([(x, y, w, h, dark, vertical)], truncated)."""
    from common import grey_of, line_dark
    g = np.ascontiguousarray(grey_of(img))
    maps = run_net(model, g)
    boxes, truncated = decode(model, maps, g.shape[0], g.shape[1])
    return [(x, y, w, h, int(line_dark(g, (x, y, w, h))), v) for x, y, w, h, v in boxes], truncated


def find_lines_ex(model, img):
    """common.find_lines_ex's form for the horizontal lines: (boxes, dark, truncated); vertical lines are left out
    (the horizontal recognizers never read them)."""
    lines, truncated = find_lines(model, img)
    hz = [ln for ln in lines if not ln[5]]
    return [ln[:4] for ln in hz], [ln[4] for ln in hz], truncated

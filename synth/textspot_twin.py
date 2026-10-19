"""Bit-exact Python twin of the full-frame text spotter's C runtime (runtime/vocr_textspot.c; docs/textspot.md).

The spotter reads a whole frame (PS Vita 960x544, PSP 480x272, Adrenaline's PSP picture at 2x, a PC screenshot
scaled to the Vita), runs #97's int8 detector network (synth/textdet_twin.py: the VTDT layout and its integer
arithmetic, unchanged) and decodes its four maps into every text box of the frame, each with an orientation, a
confidence and a dark flag; the boxes are grouped into blocks and put in reading order, and one block is picked as
the likely dialogue:

    from textspot_twin import Model, spot
    m = Model(open("spot.vtsp", "rb").read())
    r = spot(m, frame)          # frame: H x W grey or H x W x 3/4 RGB(A) uint8
    r.boxes                     # [Box(x, y, w, h, vertical, conf, dark, block)], in reading order
    r.blocks                    # [Block(x, y, w, h, vertical, first, count)], in reading order
    r.pick                      # index of the picked block, -1 when there is no box
    r.truncated                 # 1 when candidates were dropped (more than BOX_MAX)

The blob (little-endian; train/textspot_export.py or pack() writes it):

    0   char[4]  magic "VTSP"
    4   u32      version (1)
    8   u32      total bytes
    12  u32      CRC-32 (zlib) of bytes [16, total)
    16  u16      half_above: a frame wider than this many pixels is halved before the network (0: never)
    18  u8       conf_min: candidates whose confidence is below it are dropped (1..127; 1 keeps every one)
    19  u8       reserved (0)
    20  u32      offset of the VTDT blob (32)
    24  u32      bytes of the VTDT blob (total - 32)
    28  u32      reserved (0)
    32  the VTDT blob of synth/textdet_twin.py (its own header, CRC and layer records)

The frame, step by step (W x H; every coordinate below is in frame pixels unless it says otherwise):

  grey      g = (77 R + 150 G + 29 B) >> 8 for RGB(A) (synth/common.to_gray), the frame itself when grey
  input     f = 2 when half_above > 0 and W > half_above, else 1. With f = 2 the network reads the
            ceil(H/2) x ceil(W/2) image x[i][j] = (g[2i][2j] + g[2i][2j'] + g[2i'][2j] + g[2i'][2j'] + 2) >> 2,
            with 2i' = min(2i + 1, H - 1) and 2j' = min(2j + 1, W - 1) (the last row or column repeats); with
            f = 1 it reads g. So a Vita frame is read at the PSP's scale, a PSP frame as it is
  network   textdet_twin.run_net on the input: maps K, O, A1, A2 at the output stride s (4 or 8 input pixels)
  kernel    cells with K > 0, 4-connected components (textdet_twin.components)
  a component of n cells (n >= the blob's min cells), with K, O, A1, A2 its cells' codes:
            conf      = floor(sum K / n)                      (1..127; dropped when below conf_min)
            vertical  = sum O > 0
            across the line (rows for a horizontal line, columns for a vertical one), each cell's doubled
            input-pixel centre c2 = 2 s i + s - 1 (i its row, or column, index):
            near      = floor(f sum(c2 - A1) / 2n),  far = ceil(f sum(c2 + A2) / 2n)
            along     = [f s first, f s (last + 1)), first and last the component's along-line cell indices
            dropped when far - near < 4 f
            pad       = max(2, ((far - near) 18 + 50) / 100) on every side, then clipped to the frame; dropped
                        when the clipped box is empty
  caps      at most BOX_MAX = 64 candidates are kept: the smallest by (-conf, -n, vertical, y, x, h, w);
            truncated = 1 when any was dropped
  dark      synth/common.line_dark on the frame's grey in the box (vocr_line_dark)
  blocks    two boxes of one orientation join a block when they overlap along the line, the gap between them
            across the line is at most the larger one's size across, and the larger size across is at most 3/2
            of the smaller (union-find over every pair; horizontal and vertical boxes never join)
  order     inside a block, horizontal boxes go top to bottom in rows (a box joins the current row when its
            rows overlap the row's first box by at least half the smaller height; a row runs left to right) and
            vertical boxes right to left in column groups (a box joins the current group when its columns
            overlap the group's first box by at least half the smaller width; a group runs top to bottom).
            Blocks go top to bottom in rows of blocks (a block joins the current row when its rows overlap the
            row's first block by at least half the smaller height), a row left to right, or right to left when
            every block in it is vertical
  pick      the block with the largest score = sum of its boxes' w h, doubled when the block's centre is in the
            lower half of the frame (y0 + y1 >= H); ties: the larger top y, then the earlier block

The same block, order and pick rules apply to any detector's boxes (eval/textspot.py runs them on the
baselines' boxes too): group_and_order(boxes, height).
"""
import struct
import zlib
from dataclasses import dataclass

import numpy as np

import textdet_twin as TD

MAGIC = b"VTSP"
VERSION = 1
HEADER = 32
BOX_MAX = 64
PAD_PCT, PAD_MIN, MIN_ACROSS = 18, 2, 4


class Model:
    """A parsed VTSP blob. Raises ValueError on anything the C loader would refuse."""

    def __init__(self, blob):
        blob = bytes(blob)
        if len(blob) < HEADER or blob[:4] != MAGIC:
            raise ValueError("not a VTSP blob")
        version, total, crc = struct.unpack_from("<III", blob, 4)
        if version != VERSION or total != len(blob):
            raise ValueError(f"version {version} / size {total} != {VERSION} / {len(blob)}")
        if zlib.crc32(blob[16:]) & 0xFFFFFFFF != crc:
            raise ValueError("CRC mismatch")
        self.half_above, self.conf_min, r0, off, size, r1 = struct.unpack_from("<HBBIII", blob, 16)
        if r0 or r1 or off != HEADER or off + size != total or not 1 <= self.conf_min <= 127:
            raise ValueError("bad VTSP header")
        self.net = TD.Model(blob[off:off + size])
        self.stride, self.min_cells, self.name = self.net.stride, self.net.min_cells, self.net.name
        self.blob_bytes = total


def pack(vtdt, half_above=0, conf_min=1):
    """A VTSP blob around a VTDT blob (bytes)."""
    vtdt = bytes(vtdt)
    TD.Model(vtdt)
    if not 0 <= half_above <= 0xFFFF or not 1 <= conf_min <= 127:
        raise ValueError("half_above 0..65535, conf_min 1..127")
    total = HEADER + len(vtdt)
    body = struct.pack("<HBBIII", half_above, conf_min, 0, HEADER, len(vtdt), 0) + vtdt
    return MAGIC + struct.pack("<III", VERSION, total, zlib.crc32(body) & 0xFFFFFFFF) + body


def scale_of(model, width):
    return 2 if model.half_above and width > model.half_above else 1


def halve(grey):
    """The 2x2 mean of the module docstring: ceil(H/2) x ceil(W/2) uint8."""
    g = np.asarray(grey, np.int32)
    if g.shape[0] % 2:
        g = np.vstack([g, g[-1:]])
    if g.shape[1] % 2:
        g = np.hstack([g, g[:, -1:]])
    return ((g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2).astype(np.uint8)


def net_input(model, grey):
    """(the network's input image, f)."""
    f = scale_of(model, grey.shape[1])
    return (halve(grey) if f == 2 else np.ascontiguousarray(grey)), f


def _floor_div(a, b):
    return a // b


def _ceil_div(a, b):
    return -((-a) // b)


def candidates(model, maps, f, height, width):
    """Every kept candidate before the cap: [(key, (x, y, w, h, vertical, conf))], key = the cap's sort key."""
    s = model.stride
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
        near = _floor_div(f * int((c2 - a1).sum()), 2 * n)
        far = _ceil_div(f * int((c2 + a2).sum()), 2 * n)
        lo, hi = f * s * int(along_idx.min()), f * s * (int(along_idx.max()) + 1)
        across = far - near
        if across < MIN_ACROSS * f:
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
        key = (-conf, -n, int(vertical), y0, x0, y1 - y0, x1 - x0)
        out.append((key, (x0, y0, x1 - x0, y1 - y0, int(vertical), conf)))
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


@dataclass(frozen=True)
class Block:
    x: int
    y: int
    w: int
    h: int
    vertical: int
    first: int
    count: int


@dataclass(frozen=True)
class Result:
    boxes: tuple
    blocks: tuple
    pick: int
    truncated: int


def _joins(a, b):
    """a, b: (x, y, w, h, vertical, ...) of one orientation."""
    if a[4] != b[4]:
        return False
    if a[4]:
        ov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
        sa, sb = a[2], b[2]
    else:
        ov = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        gap = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
        sa, sb = a[3], b[3]
    return ov > 0 and gap <= max(sa, sb) and 2 * max(sa, sb) <= 3 * min(sa, sb)


def _rows(items, key_sort, lo, size, key_row):
    """Sort items by key_sort, gather them into runs (an item joins the current run when its [lo, lo + size)
    overlaps the run's first item's by at least half the smaller size), each run sorted by key_row."""
    out, run = [], []
    for it in sorted(items, key=key_sort):
        if run:
            f = run[0]
            ov = min(lo(it) + size(it), lo(f) + size(f)) - max(lo(it), lo(f))
            if 2 * ov >= min(size(it), size(f)):
                run.append(it)
                continue
            out += sorted(run, key=key_row)
        run = [it]
    return out + sorted(run, key=key_row)


def _order_block(members):
    """members: [(x, y, w, h, vertical, ...)] of one block -> reading order."""
    if members[0][4]:
        return _rows(members, key_sort=lambda b: (-(b[0] + b[2]), b[1], b[0], b[2], b[3]),
                     lo=lambda b: b[0], size=lambda b: b[2], key_row=lambda b: (b[1], -(b[0] + b[2]), b[3], b[2]))
    return _rows(members, key_sort=lambda b: (b[1], b[0], b[3], b[2]),
                 lo=lambda b: b[1], size=lambda b: b[3], key_row=lambda b: (b[0], b[1], b[3], b[2]))


def group_and_order(boxes, height):
    """boxes: [(x, y, w, h, vertical, *rest)] -> (ordered boxes [(x, y, w, h, vertical, *rest, block)],
    blocks [(x, y, w, h, vertical, first, count)], pick). Deterministic for any input order of distinct boxes."""
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
        x0 = min(b[0] for b in members)
        y0 = min(b[1] for b in members)
        x1 = max(b[0] + b[2] for b in members)
        y1 = max(b[1] + b[3] for b in members)
        blocks.append(((x0, y0, x1 - x0, y1 - y0, members[0][4]), _order_block(members)))
    # rows of blocks, top to bottom; a row left to right, or right to left when all its blocks are vertical
    rows, run = [], []
    for bk in sorted(blocks, key=lambda t: (t[0][1], t[0][0], t[0][3], t[0][2])):
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
    ordered_blocks = []
    for run in rows:
        if all(t[0][4] for t in run):
            ordered_blocks += sorted(run, key=lambda t: (-(t[0][0] + t[0][2]), t[0][1], t[0][3], t[0][2]))
        else:
            ordered_blocks += sorted(run, key=lambda t: (t[0][0], t[0][1], t[0][3], t[0][2]))
    out_boxes, out_blocks = [], []
    for k, (bb, members) in enumerate(ordered_blocks):
        out_blocks.append((*bb, len(out_boxes), len(members)))
        out_boxes += [(*m, k) for m in members]
    pick, best = -1, None
    for k, (bb, members) in enumerate(ordered_blocks):
        score = sum(m[2] * m[3] for m in members) * (2 if 2 * bb[1] + bb[3] >= height else 1)
        key = (score, bb[1])
        if best is None or key > best:
            pick, best = k, key
    return out_boxes, out_blocks, pick


def spot(model, img):
    """The spotter on one frame (grey or RGB(A) uint8): a Result."""
    from common import grey_of, line_dark
    g = np.ascontiguousarray(grey_of(img))
    if g.ndim != 2 or not g.size:
        raise ValueError("a non-empty frame")
    H, W = g.shape
    x, f = net_input(model, g)
    maps = TD.run_net(model.net, x)
    cands = candidates(model, maps, f, H, W)
    truncated = int(len(cands) > BOX_MAX)
    kept = [c for _, c in sorted(cands)[:BOX_MAX]]
    with_dark = [(bx, by, bw, bh, v, conf, int(line_dark(g, (bx, by, bw, bh)))) for bx, by, bw, bh, v, conf in kept]
    boxes, blocks, pick = group_and_order(with_dark, H)
    return Result(tuple(Box(*b) for b in boxes), tuple(Block(*b) for b in blocks), pick, truncated)

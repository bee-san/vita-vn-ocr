#!/usr/bin/env python3
"""Parity check: runtime/vocr_textspot6.c against its Python twin synth/textspot6_twin.py (the specification), for
VTSP version 1, 2, 3 and 4 blobs; versions 1-3 also against textspot3's frozen runtime (runtime/vocr_textspot3.c).

  textspot6_parity.py --bin build/textspot6/vocr_textspot6_test [--v3-bin build/textspot6/vocr_textspot3_test]
                      [--noms-bin build/textspot6/vocr_textspot6_test_noms] [--t-shapes tinyt,t64n8c16]
                      [--m-shapes tinym,qS4,qM4,qL5] [--t-tags 1,2,3,4] [--m-tags 2,4] [--seeds 1] [--frames 3]
                      [--probe-frames 4] [--full-every 3] [--maps-every 2] [--arena-every 2] [--graphs 2]
                      [--selftest 8] [--selftest-blobs 8] [--mutants 24] [--v3-seeds 1] [--v3-frames 2]
                      [--v3-probe-shapes pS1] [--v3-graphs 1] [--v1-shapes s4_c4,s8_c8] [--d2s-shapes s8_c24n4x2]
                      [--seed 0] [--blob FILE ...]

Version 4 (the tagged spotter):
  - random blobs (train/textspot6_model.random_blob) of the T shapes (train/textspot3_shapes' registered and
    candidate T shapes, the full-resolution T shapes of eval/textspot6_a9.py, and a tiny one) with every tag count of
    --t-tags, and of the M shapes (train/textspot3_model's probes and the registered qS4 / qM4 / qL5, and a tiny one)
    with --m-tags (2 and 4: a multi-scale head of 16 + 4 T channels is a multiple of 8); each at its own scales and,
    for every third, at (0, 0), (600, 0), (0, 300) and (480, 272), some with a conf_min > 1;
  - wild-requantisation networks of both forms (biases over the int32 range, multipliers 1 .. 2^31 - 1, shifts 1 ..
    62; the trunk's weights at -128, the multi-scale ones at +-127) and random op graphs outside the grammar with a
    tagged head (tests/textspot3_parity.graph_layers, its head widened to 16 + 4 T channels, over 1..3 sources at any
    upsampling: the head gathers);
  - probe networks (probe6_net: a space-to-depth trunk that carries the grey of 4 + T chosen pixels of each 4 x 4
    sub-block to the K, O, A1, A2 and tag maps of that stride-4 cell; as a trunk, and as a VTMS section with no op
    but the head, with identity ops in place, and with a gathered head over an upsampled map), on frames whose maps
    are designed cell by cell: the twin tests' hand-made maps (tests/test_textspot6_twin: score ties, tconf,
    tag-separated blocks, the dialogue pick, presence 0, truncation) and random designed maps (lines of every class
    with tag codes from a small set, so ties happen, joined and separated by class, vertical columns, more than 64
    components), read at f = 1 and, doubled, at f = 2 (halved by the width or by the height). The twin's maps on such
    a frame must be the designed maps; C must give the twin's result, and on the twin tests' maps the twin tests'
    expectations.
Frames: tests/textspot_parity's procedural full frames (960x544, 480x272, 725x544, 870x544 on every --full-every-th
blob), edge and odd sizes, GREY and RGBA; the in-memory and row-callback forms, VOCR_TEXTSPOT_NO_DARK, padded
strides, odd arenas, the arena for every frame up to the size. C (--batch) and the twin (textspot6_twin.spot; maps:
textspot6_twin.run_maps) must agree exactly on the maps the decode reads (4 + T planes), the boxes with their tag and
tconf, the blocks with their tag, order, pick, truncated and presence.
Versions 1-3: tests/textspot3_parity's blobs (random, graph, wild and probe version-3 networks; textspot2_parity's
version-1 and version-2 blobs): --batch must be the twin's (every tag 0, presence the largest confidence), and with
--v3-bin every batch's --batch3 lines, every --maps output and every blob's arenas (for the frame and up to it) must be
exactly runtime/vocr_textspot3_test's.
The loader: C must refuse exactly what textspot6_twin.Model refuses: bad_v4() (header, tags, reserved bytes, both
network forms, odd T in a multi-scale head, the head's channels beyond the 16th), tests/test_textspot3_twin's and
tests/test_textspot2_twin's refusals, field mutants of good version-4 blobs (every header and record field set to
boundary values, weights -128 and requantisation corners in the head's tag channels; both CRCs fixed, so the field
checks decide) and random damage; versions 1-3 also give vocr_textspot3_test's codes. --noms-bin (the driver built
with -DVOCR_TEXTSPOT6_NO_MS) must refuse every version-3 blob and version-4 multi-scale network, give the full
build's codes on everything else, and the full build's lines on the trunk blobs' batches. --selftest N runs the
driver's own invariants on N frames of a few blobs of every version (and the NO_MS driver's on trunk blobs).
Prints TEXTSPOT6_PARITY_OK <runs> runs (...).
"""
import argparse
import hashlib
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as E  # noqa: E402
import textdet_parity as TP  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_parity as Q3  # noqa: E402
import textspot3_twin as T3  # noqa: E402
import textspot6_model as M6  # noqa: E402
import textspot6_twin as T6  # noqa: E402
import textspot_parity as P  # noqa: E402

ONE = 1 << 30
SCALE1, SCALE2 = 32, 64                     # vocr_textspot6_test's record flags of VOCR_TEXTSPOT6_SCALE1 / _SCALE2
TINY_T = {"name": "tinyt", "family": "T", "stride": 8, "c0": 4, "c1": 8, "c2": 16, "chans": [16, 16], "dil": [1, 2],
          "block": "sep", "head": "kroa_x2", "half_above": 480}
TINY_M = M3.m_shape("tinym", 4, 8, 8, [8], [1], [(16, [1]), (16, [2])], 8, 8, 1)
SCALES = ((0, 0), (600, 0), (0, 300), (480, 272))
E_BLOB = -5


# ---------------------------------------------------------------- shapes

def t_shapes_all():
    """Every T shape this check knows: the tiny one, train/textspot3_shapes' (registered and candidates) and the
    full-resolution T shapes of eval/textspot6_a9.py (half_above 0)."""
    import textspot3_shapes as S3
    out = {"tinyt": TINY_T}
    for k, v in {**S3.CANDIDATES, **S3.T}.items():
        out[k] = dict(v)
    try:
        import textspot6_a9 as A6
        for k, v in {**A6.T_FULL, **A6.T_FULL_PROBE}.items():
            out.setdefault(k, dict(v))
    except ImportError:
        pass
    return out


def m_shapes_all():
    """Every M shape: the tiny one, train/textspot3_model.PROBES and the registered qS4 / qM4 / qL5."""
    import textspot3_shapes as S3
    out = {"tinym": TINY_M}
    out.update(M3.PROBES)
    for k, v in S3.M.items():
        out[k] = M3.check_shape(dict(v))
    return out


# ---------------------------------------------------------------- blob builders

def repack(blob, half_above, conf_min, half_above_h):
    """A version-4 blob's network under another VTSP header."""
    m = T6.Model(blob)
    return T6.pack(blob[32:], m.tags, half_above, conf_min, half_above_h)


def set_min_cells(net, n):
    """A network (VTDT or VTMS) with another min cells (byte 21), its CRC fixed."""
    b = bytearray(net)
    b[21] = n
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


def _q(L, wq, bq, m, sh):
    return {**L, "wq": np.asarray(wq, np.int8), "bq": np.asarray(bq, np.int64), "m": np.asarray(m, np.int64),
            "sh": np.asarray(sh, np.int64)}


def _ident(cout):
    return [0] * cout, [ONE] * cout, [30] * cout


def probe6_channel(dy, dx, q, tags):
    """The head channel of decode map q (0 K, 1 O, 2 A1, 3 A2, 4 + t tag map t) of the sub-cell (dy, dx)."""
    return 4 * (2 * dy + dx) + q if q < 4 else T6.tag_channel(dy, dx, q - 4, tags)


def probe6_source(dy, dx, q):
    """The channel of the probe's stride-8 map (32 channels) that holds the pixel of decode map q (q < 8) of the
    sub-cell (dy, dx): pixel (4 dy + 2 (q // 4), 4 dx + q % 4) of the cell's 8 x 8 block (encode())."""
    py, px = 4 * dy + 2 * (q // 4), 4 * dx + q % 4
    return 8 * (2 * (py >> 2) + (px >> 2)) + 2 * (2 * ((py >> 1) & 1) + ((px >> 1) & 1)) + (px & 1)


def probe6_stream(part):
    """The space-to-depth stream: three 3x3 stride-2 dense layers 1 -> 2 -> 8 -> 32, each output channel one tap of
    its input (identity requantisation, no ReLU), carrying the even rows of every 2 x 2 block only: channel
    8 (2 dy2 + dx2) + 2 (2 dy1 + dx1) + dx0 of the stride-8 map is pixel (4 dy2 + 2 dy1, 4 dx2 + 2 dx1 + dx0) of the
    cell's 8 x 8 block, minus 128 (the 32 pixels the head reads, so the probe costs about 4,000 MACs per cell)."""
    out = []
    for cin, cout in ((1, 2), (2, 8), (8, 32)):
        w = np.zeros((cout, cin, 3, 3), np.int8)
        if cin == 1:                                  # the even row of each 2 x 2: pixel (2 i, 2 j + dx)
            for dx in (0, 1):
                w[dx, 0, 1, dx + 1] = 1
        else:                                         # every tap of the 2 x 2 below: channel cin (2 dy + dx) + c
            for dy in (0, 1):
                for dx in (0, 1):
                    for c in range(cin):
                        w[cin * (2 * dy + dx) + c, c, dy + 1, dx + 1] = 1
        L = {"kind": "conv", "k": 3, "stride": 2, "dil": 1, "relu": False, "cin": cin, "cout": cout, "head": False}
        if part:
            L["part"] = part
        out.append(_q(L, w, *_ident(cout)))
    return out


def probe6_head_weights(tags, cin=32):
    out = T6.outputs_of(tags)
    w = np.zeros((out, cin, 1, 1), np.int8)
    for dy in (0, 1):
        for dx in (0, 1):
            for q in range(4 + tags):
                w[probe6_channel(dy, dx, q, tags), probe6_source(dy, dx, q % 8), 0, 0] = 1
    return w


def probe6_net(form, tags, variant=0):
    """A version-4 probe network: its decode maps at stride 4 are the codes of chosen frame pixels (encode()).
    form "trunk": the space-to-depth stream and a 1x1 head (a VTDT network). form "ms": the same stream as a VTMS
    stream, then variant 0: the head alone; 1: a dilated identity depthwise op and an identity 1x1 op in place before
    the head; 2: a stride-2 identity depthwise op to stride 16 and the head over [F8, it at u 1, it at u 1] with zero
    weights for the upsampled sources (the head gathers its cells)."""
    out = T6.outputs_of(tags)
    name = f"probe6_{form}_v{variant}"
    if form == "trunk":
        layers = probe6_stream(None)
        layers.append(_q({"kind": "conv", "k": 1, "stride": 1, "dil": 1, "relu": False, "cin": 32, "cout": out,
                          "head": True}, probe6_head_weights(tags), *_ident(out)))
        return M6.pack_trunk({"name": name}, layers, tags, hashlib.sha256(name.encode()).digest())
    layers = probe6_stream("stream")
    src, up, cin = [0], [0], 32
    if variant == 1:
        layers.append(Q3.op_layer("dw", [0], [0], 32, 32, Q3.identity_dw(32), *_ident(32), dil=2))
        layers.append(Q3.op_layer("pw", [1], [0], 32, 32, Q3.identity_pw(32, 32), *_ident(32)))
        src = [2]
    elif variant == 2:
        layers.append(Q3.op_layer("dw", [0], [0], 32, 32, Q3.identity_dw(32), *_ident(32), stride=2, out_stride=16))
        src, up, cin = [0, 1, 1], [0, 1, 1], 96
    w = np.zeros((out, cin, 1, 1), np.int8)
    w[:, :32] = probe6_head_weights(tags)
    layers.append(Q3.op_layer("pw", src, up, cin, out, w, *_ident(out), head=True))
    return M6.pack_ms({"name": name}, layers, tags, hashlib.sha256(name.encode()).digest())


def encode(maps, rng):
    """The (4 hd) x (4 wd) network input whose probe maps are `maps` ((4 + T, hd, wd) codes): pixel
    (4 I + 2 (q // 4), 4 J + q % 4) is maps[q][I][J] + 128, every other pixel random."""
    nm, hd, wd = maps.shape
    g = rng.integers(0, 256, (4 * hd, 4 * wd))
    for q in range(nm):
        g[2 * (q // 4)::4, q % 4::4] = np.asarray(maps[q]) + 128
    return g.astype(np.uint8)


def wild6(shape, seed, tags):
    """random_int_layers of a tagged shape with some channels' requantisation at the loader's extremes (a trunk's
    weights also at -128; a multi-scale network's stay in +-127)."""
    rng = np.random.default_rng(seed)
    q = M6.random_int_layers(shape, seed, tags, n_cal=2)
    trunk = M6.family_of(shape) == "T"
    for L in q:
        wild = rng.random(L["cout"]) < 0.35
        L["wq"] = L["wq"].copy()
        L["wq"][rng.random(L["wq"].shape) < 0.05] = -128 if trunk else 127 * (1 - 2 * (seed % 2))
        L["bq"] = np.where(wild, rng.integers(-(1 << 31), 1 << 31, L["cout"]), L["bq"])
        L["m"] = np.where(wild, rng.integers(1, 1 << 31, L["cout"]), L["m"])
        L["sh"] = np.where(wild, rng.integers(1, 63, L["cout"]), L["sh"])
        if wild.any():
            k = int(rng.integers(0, L["cout"]))
            L["bq"][k], L["m"][k], L["sh"][k] = (-(1 << 31), (1 << 31) - 1, 1) if seed % 2 else ((1 << 31) - 1, 1, 62)
    net = M6.pack_net(dict(shape, name=f"wild_{shape['name']}"[:31]), q, tags,
                      hashlib.sha256(f"wild6:{shape['name']}:{seed}".encode()).digest())
    ha, hh = M6.scales_of(shape)
    return T6.pack(net, tags, ha, 1, hh)


def graph6(seed, tags):
    """tests/textspot3_parity.graph_layers' random op graph with its head widened to 16 + 4 T channels (random
    weights, its requantisation spread as the graph's head)."""
    rng = np.random.default_rng(10_000 + seed)
    layers = Q3.graph_layers(seed)
    h = layers[-1]
    out = T6.outputs_of(tags)
    extra = out - h["cout"]
    w = rng.integers(-127, 128, (extra, h["cin"], 1, 1)).astype(np.int8)
    w[rng.random(w.shape) < 0.3] = 0
    ms = [E.quantize_multiplier(float(20.0 / (60.0 * np.sqrt(h["cin"]) + 1) * rng.uniform(0.5, 2)))
          for _ in range(extra)]
    head = dict(h, cout=out, wq=np.concatenate([h["wq"], w], 0),
                bq=np.concatenate([h["bq"], rng.integers(-8, 9, extra)]),
                m=np.concatenate([h["m"], [m for m, _ in ms]]), sh=np.concatenate([h["sh"], [s for _, s in ms]]))
    net = M6.pack_ms({"name": f"graph6_{seed}"}, layers[:-1] + [head], tags,
                     hashlib.sha256(f"graph6:{seed}:{tags}".encode()).digest())
    return T6.pack(net, tags, 480, 1, 272)


def v4_blobs(args):
    """[(name, VTSP version-4 bytes)]: random T and M blobs at their scales and others, wild and graph networks,
    the probes (at scales the designed frames choose; see probe_runs)."""
    rng = np.random.default_rng(6543)
    ts, ms = t_shapes_all(), m_shapes_all()
    out = []
    j = 0
    for fam, names, tags_list in (("T", args.t_shapes, args.t_tags), ("M", args.m_shapes, args.m_tags)):
        for i, name in enumerate(names):
            shape = (ts if fam == "T" else ms)[name]
            for t in tags_list:
                for k in range(args.seeds):
                    seed = 6000 * (k + 1) + 10 * i + t
                    blob, _ = M6.random_blob(shape, seed, t, n_cal=args.n_cal)
                    ha, hh = M6.scales_of(shape)
                    cm = int(rng.integers(2, 40)) if j % 4 == 3 else 1
                    out.append((f"{name}_t{t}_s{seed}_h{ha}x{hh}_c{cm}", repack(blob, ha, cm, hh)))
                    if j % 3 == 0:
                        for sa, sh in SCALES:
                            if (sa, sh) != (ha, hh):
                                out.append((f"{name}_t{t}_s{seed}_h{sa}x{sh}_c1", repack(blob, sa, 1, sh)))
                    j += 1
    for k, t in enumerate(args.t_tags[:2]):
        out.append((f"wild6_tinyt_t{t}", wild6(TINY_T, 1 + k, t)))
    for k, t in enumerate(args.m_tags[:2]):
        out.append((f"wild6_tinym_t{t}", wild6(TINY_M, 3 + k, t)))
    for g in range(args.graphs):
        t = args.m_tags[g % len(args.m_tags)]
        out.append((f"graph6_{g}_t{t}", graph6(700 + g, t)))
    for p in args.blob:
        out.append((Path(p).name, Path(p).read_bytes()))
    return out


# ---------------------------------------------------------------- designed maps

def blank_maps(hd, wd, tags, rng=None):
    mp = np.zeros((4 + tags, hd, wd), np.int64)
    mp[0] = -100 if rng is None else rng.integers(-128, 1, (hd, wd))
    mp[2] = mp[3] = 4
    return mp


def put_line(mp, rows, cols, tag_codes, k=64, o=-16, a=(4, 4)):
    """Kernel cells rows x cols with K = k, O = o, A1, A2 = a and the tag maps' codes (a code per map or an array)."""
    mp[0, rows, cols] = k
    mp[1, rows, cols] = o
    mp[2, rows, cols] = a[0]
    mp[3, rows, cols] = a[1]
    for t, v in enumerate(tag_codes):
        mp[4 + t, rows, cols] = v


def twin_cases():
    """The hand-made maps of tests/test_textspot6_twin.py on a 272 x 480 input, with what that test expects of each:
    [(name, tags, maps, check(result) -> error string or "")]."""
    h, w = 272, 480
    hd, wd = h // 4, w // 4
    out = []
    mp = blank_maps(hd, wd, 4)
    for row, codes in ((10, [-16, -16, -16, -16]), (20, [32, -16, 0, 0]), (30, [16, 16, 0, 0]), (40, [0, 0, 0, 0]),
                       (50, [-8, -8, 8, 24])):
        put_line(mp, row, slice(5, 40), codes, o=0)

    def scores(r):
        by_y = {b.y: b for b in r.boxes}
        got = ([by_y[k].tag for k in sorted(by_y)], [by_y[k].tconf for k in sorted(by_y)])
        return "" if got == ([0, 1, 1, 0, 4], [16, 32, 0, 0, 16]) else f"tags/tconf {got}"
    out.append(("scores_ties_tconf", 4, mp, scores))
    mp = blank_maps(hd, wd, 4)
    put_line(mp, 50, slice(10, 20), [40, -10, -10, -10], o=0)
    put_line(mp, 53, slice(10, 60), [-10, -10, -10, -10], o=0)
    put_line(mp, 56, slice(10, 60), [-10, -10, -10, -10], o=0)
    for row in (60, 63, 66):
        put_line(mp, row, slice(70, 119), [-10, -10, 40, -10], o=0)

    def blocks(r):
        if sorted(b.tag for b in r.boxes) != [0, 0, 1, 3, 3, 3]:
            return f"tags {sorted(b.tag for b in r.boxes)}"
        for blk in r.blocks:
            if {b.tag for b in r.boxes[blk.first:blk.first + blk.count]} != {blk.tag}:
                return "a block of two classes"
        if r.pick < 0 or r.blocks[r.pick].tag != 0 or r.blocks[r.pick].count != 2 or r.presence != 64:
            return f"pick {r.pick} presence {r.presence}"
        return ""
    out.append(("blocks_by_tag_pick", 4, mp, blocks))
    mp = blank_maps(hd, wd, 4)
    put_line(mp, 30, slice(10, 60), [40, -10, -10, -10], o=0)
    out.append(("no_dialogue", 4, mp, lambda r: "" if (len(r.boxes), r.boxes[0].tag if r.boxes else -1, r.pick,
                                                        r.presence) == (1, 1, -1, 0) else "pick/presence"))
    mp = blank_maps(hd, wd, 2)
    for i in range(70):
        row, col = 2 + 2 * (i // 8), 2 + 14 * (i % 8)
        put_line(mp, row, slice(col, col + 3), [(-1) ** i * 20, 0], k=1 + i, o=0)
    out.append(("truncation", 2, mp, lambda r: "" if (len(r.boxes), r.truncated, min(b.conf for b in r.boxes)) ==
                (64, 1, 7) else f"truncation {len(r.boxes)} {r.truncated}"))
    return out


def _cls_codes(tags, cls, margin):
    """Tag codes per cell that give class cls with a margin (tconf) of `margin` over the next class."""
    if cls == 0:
        return [-margin] * tags
    codes = [0] * tags
    codes[cls - 1] = margin
    return codes


def tiebreak_cases(tags):
    """Designed maps (a 160 x 480 input) whose result turns on the version-4 tie-breaks:
      cap_tag    two components of one box (rows 10 and 12, A codes 16 apart; the same n and conf), the earlier one of
                 the higher class, behind 63 more confident fillers: the cap keeps the lower class;
      cap_tconf  the same with one class, the earlier one of the larger tconf: the cap keeps the lower tconf;
      full_tconf two components of one box, one class and conf, n 10 and 20, the smaller one of the smaller tconf:
                 the block lists them by (x, y, w, h, vertical, conf, tag, tconf), the smaller tconf first;
      row_tag    lines on rows 10 and 14 of one class (one block) and between them a thicker line of a lower class
                 whose box is their union, across the whole width (horizontal) or, transposed, the whole height
                 (vertical): two blocks of one rectangle in one row, the lower class first.
    [(name, tags, maps, check)]."""
    hd, wd = 40, 120
    hi = min(2, tags)
    lo = hi - 1
    out = []

    def fillers(mp):
        for i in range(63):
            row, col = 2 * (i // 21), 1 + 5 * (i % 21)
            put_line(mp, row, slice(col, col + 3), [-10] * tags, k=100, o=-16)

    def pair_box(r):
        return [b for b in r.boxes if (b.y, b.h) == (39, 13) and b.conf == 50]

    mp = blank_maps(hd, wd, tags)
    fillers(mp)
    put_line(mp, 10, slice(20, 30), _cls_codes(tags, hi, 20), k=50, o=-16, a=(0, 16))
    put_line(mp, 12, slice(20, 30), _cls_codes(tags, lo, 20), k=50, o=-16, a=(16, 0))
    out.append(("cap_tag", tags, mp, lambda r: "" if r.truncated and [b.tag for b in pair_box(r)] == [lo] else
                f"kept {[(b.tag, b.tconf) for b in pair_box(r)]} truncated {r.truncated}"))
    mp = blank_maps(hd, wd, tags)
    fillers(mp)
    put_line(mp, 10, slice(20, 30), _cls_codes(tags, tags, 30), k=50, o=-16, a=(0, 16))
    put_line(mp, 12, slice(20, 30), _cls_codes(tags, tags, 20), k=50, o=-16, a=(16, 0))
    out.append(("cap_tconf", tags, mp, lambda r: "" if r.truncated and [b.tconf for b in pair_box(r)] == [20] else
                f"kept {[(b.tag, b.tconf) for b in pair_box(r)]} truncated {r.truncated}"))
    mp = blank_maps(hd, wd, tags)
    put_line(mp, 10, slice(20, 30), _cls_codes(tags, tags, 20), k=50, o=-16, a=(0, 16))
    put_line(mp, 12, slice(20, 30), _cls_codes(tags, tags, 30), k=50, o=-16, a=(20, -4))
    put_line(mp, 13, slice(20, 30), _cls_codes(tags, tags, 30), k=50, o=-16, a=(20, -4))
    out.append(("full_tconf", tags, mp, lambda r: "" if [b.tconf for b in pair_box(r)] == [20, 30] and
                len({b.block for b in pair_box(r)}) == 1 else f"order {[(b.tconf, b.block) for b in pair_box(r)]}"))
    for vertical in (0, 1):
        mp = blank_maps(hd, wd, tags) if not vertical else blank_maps(wd, hd, tags)
        span = slice(0, mp.shape[2])
        for row, tag, a in ((10, hi, 4), (14, hi, 4), (12, lo, 18)):
            put_line(mp, row, span, _cls_codes(tags, tag, 10), k=40, o=-16, a=(a, a))
        if vertical:
            mp = mp.transpose(0, 2, 1).copy()
            mp[1] = np.where(mp[0] > 0, 16, mp[1])

        def row_tag(r, v=vertical):
            blk = [(b.tag, b.count) for b in r.blocks if b.vertical == v and
                   ((b.x, b.w) == (37, 25) if v else (b.y, b.h) == (37, 25))]
            return "" if blk == [(lo, 1), (hi, 2)] else f"blocks {blk}"
        out.append((f"row_tag_{'v' if vertical else 'h'}", tags, mp, row_tag))
    return out


TAG_CODES = (-24, -8, 0, 8, 24, 40)


def designed6(rng, tags, hd, wd):
    """Random designed maps: lines (1-3 rows high) and vertical columns of every class, tag codes per line from a
    small set (ties), some per-cell noise, lines close together (blocks; across classes they must stay apart), now and
    then more than 64 components or none."""
    mp = blank_maps(hd, wd, tags, rng)
    mode = rng.random()
    n = 0 if mode < 0.05 else int(rng.integers(60, 120)) if mode < 0.15 else int(rng.integers(1, 24))
    for _ in range(n):
        vertical = rng.random() < 0.25
        cls = int(rng.integers(0, tags + 1))
        codes = []
        for t in range(tags):
            v = int(rng.choice(TAG_CODES))
            if cls and t == cls - 1:
                v = max(v, int(rng.choice(TAG_CODES[2:])))
            codes.append(v)
        if rng.random() < 0.3:
            codes = [np.int64(c) + rng.integers(-3, 4, 1) for c in codes]
        k = int(rng.integers(1, 128))
        a = (int(rng.integers(0, 24)), int(rng.integers(0, 24)))
        stack = 1 if rng.random() < 0.5 else int(rng.integers(2, 5))
        thick = int(rng.integers(1, 4))
        if vertical:
            j0, i0 = int(rng.integers(0, wd)), int(rng.integers(0, hd))
            ln = int(rng.integers(2, 40))
            for s in range(stack):
                cols = slice(j0 + s * (thick + 1 + int(rng.integers(0, 3))), j0 + s * (thick + 1) + thick)
                put_line(mp, slice(i0, i0 + ln), cols, codes if s % 2 == 0 or rng.random() < 0.6 else
                         [int(rng.choice(TAG_CODES)) for _ in range(tags)], k=k, o=int(rng.integers(1, 64)), a=a)
        else:
            i0, j0 = int(rng.integers(0, hd)), int(rng.integers(0, wd))
            ln = int(rng.integers(2, 80))
            for s in range(stack):
                rows = slice(i0 + s * (thick + 1 + int(rng.integers(0, 3))), i0 + s * (thick + 1) + thick)
                put_line(mp, rows, slice(j0, j0 + ln), codes if s % 2 == 0 or rng.random() < 0.6 else
                         [int(rng.choice(TAG_CODES)) for _ in range(tags)], k=k, o=int(rng.integers(-64, 1)), a=a)
    return mp


def probe_runs(rng, args, v4list):
    """[(name, blob, [(frame, designed maps or None, f, check or None)])]: the probe networks on designed frames."""
    out = []
    cases = twin_cases()
    for form, variants, tags_list in (("trunk", (0,), args.t_tags), ("ms", args.probe_variants, args.m_tags)):
        for v in variants:
            for t in tags_list:
                net = probe6_net(form, t, v)
                for mc in (2, 1 + (t + v) % 4)[:args.probe_mc]:
                    name = f"probe6_{form}_v{v}_t{t}_mc{mc}"
                    blob1 = T6.pack(set_min_cells(net, mc), t, 0, 1, 0)
                    frames = []
                    if mc == 2 and v == 0:                        # hand-made maps and what they must give
                        for cname, ct, mp, check in [c for c in cases if c[1] == t] + tiebreak_cases(t):
                            frames.append((encode(mp, rng), mp, 1, (cname, check)))
                    for _ in range(args.probe_frames):
                        hd, wd = int(rng.integers(2, args.probe_cells[0])), int(rng.integers(2, args.probe_cells[1]))
                        mp = designed6(rng, t, hd, wd)
                        frames.append((encode(mp, rng), mp, 1, None))
                    out.append((name, blob1, frames))
                    # f = 2: the designed input doubled, halved by the width (half_above) or by the height
                    hd, wd = int(rng.integers(2, args.probe_cells[0])), int(rng.integers(2, args.probe_cells[1]))
                    mp = designed6(rng, t, hd, wd)
                    g = encode(mp, rng)
                    big = np.repeat(np.repeat(g, 2, 0), 2, 1)
                    by_h = (t + v + mc) % 2 == 1
                    ha, hh = (0, g.shape[0]) if by_h else (g.shape[1], 0)
                    out.append((f"{name}_f2{'h' if by_h else 'w'}", T6.pack(set_min_cells(net, mc), t, ha, 1, hh),
                                [(big, mp, 2, None)]))
    return out


# ---------------------------------------------------------------- the driver and the twin

def twin_line6(r, dark=True):
    out = [0, len(r.boxes), len(r.blocks), r.pick, r.truncated, r.presence]
    for b in r.boxes:
        out += [b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark if dark else 0, b.block, b.tag, b.tconf]
    for b in r.blocks:
        out += [b.x, b.y, b.w, b.h, b.vertical, b.first, b.count, b.tag]
    return " ".join(map(str, out))


class Coverage:
    def __init__(self):
        self.truncated = self.halved = self.vertical = self.boxes = self.blocks = self.multi = self.empty = 0
        self.tags = [0] * 5
        self.tconf = self.nopick = self.presence0 = self.mixed = self.forced = 0

    def add(self, model, img, r):
        self.truncated += r.truncated
        self.halved += T6.scale_of(model, img.shape[1], img.shape[0]) == 2
        self.vertical += sum(b.vertical for b in r.boxes)
        self.boxes += len(r.boxes)
        self.blocks += len(r.blocks)
        self.multi += sum(b.count > 1 for b in r.blocks)
        self.empty += not r.boxes
        for b in r.boxes:
            self.tags[b.tag] += 1
        self.tconf += sum(b.tconf > 0 for b in r.boxes)
        self.nopick += bool(r.boxes) and r.pick < 0
        self.presence0 += bool(r.boxes) and r.presence == 0
        self.mixed += len({b.tag for b in r.boxes}) > 1

    def __str__(self):
        return (f"{self.truncated} truncated, {self.halved} halved, {self.empty} empty, {self.boxes} boxes "
                f"({self.vertical} vertical; classes {self.tags}; {self.tconf} with tconf > 0) in {self.blocks} blocks "
                f"({self.multi} of several boxes); {self.mixed} frames of several classes, {self.nopick} with boxes "
                f"and no dialogue pick, {self.presence0} with boxes and presence 0; {self.forced} forced-scale "
                "records (crops included)")


def multi(binary, jobs, td, tag):
    """The driver's --multi on jobs [(mode, blob path, records path)]: the outputs of the jobs, in order."""
    lst = Path(td) / f"{tag}.jobs"
    lst.write_text("".join(f"{m} {b} {f}\n" for m, b, f in jobs))
    out, cur = [], None
    for line in P.run_driver(binary, ["--multi", lst]).splitlines():
        if line.startswith("@@ "):
            cur = []
            out.append(cur)
        else:
            cur.append(line)
    if len(out) != len(jobs):
        raise RuntimeError(f"--multi gave {len(out)} outputs for {len(jobs)} jobs")
    return out


def check_frames(binary, name, blob, imgs, rng, td, cov, maps_every=1, noms=None, v3=None, arenas=False,
                 arena_frames=99, forced_p=0.3):
    """Every frame through C and the twin: (runs, maps, extra runs, mismatches). imgs: [(frame, designed maps or
    None, f, (case name, check) or None)]. noms: the NO_MS driver (a trunk blob's lines must be the full build's).
    v3: vocr_textspot3_test (a version-1, 2 or 3 blob: --batch3, --maps and the arenas must be its own; arenas at 960 x
    544, 480 x 272, 600 x 7 and up to arena_frames of the frames' sizes)."""
    model = T6.Model(blob)
    bpath = Path(td) / f"{name}.vtsp"
    bpath.write_bytes(blob)
    cases, exp, bad = [], [], []
    forced_cases, forced_exp = [], []
    for k, (g, mp, f, check) in enumerate(imgs):
        img = TP.to_rgba(g, rng) if rng.random() < 0.5 else g
        r = T6.spot(model, img)
        cov.add(model, img, r)
        if mp is not None:
            x, f2 = T6.net_input(model, np.ascontiguousarray(g))
            got = T6.run_maps(model, x)
            if f2 != f or got.shape != mp.shape or not (got == mp).all():
                bad.append((name, k, f"the probe's maps are not the designed maps (f {f2} vs {f})", ""))
        if check is not None:
            err = check[1](r)
            if err:
                bad.append((name, k, f"twin case {check[0]}: {err}", ""))
        line, line_nd = twin_line6(r), twin_line6(r, dark=False)
        base = P.ODD * int(rng.random() < 0.3) | P.PADDED * int(rng.random() < 0.3) | P.MOST * int(rng.random() < 0.3)
        for form in (0, P.ROWS):
            cases.append((img, base | form))
            exp.append((k, form, line))
        if rng.random() < 0.5:
            form = int(rng.integers(0, 2))
            cases.append((img, base | P.NO_DARK | form))
            exp.append((k, f"nodark{form}", line_nd))
        # the scale forced (VOCR_TEXTSPOT6_SCALE1 / _SCALE2, textspot6_twin.spot(..., scale=f)): the other scale
        # than the rule's, or the rule's own; and on a frame of 400 x 200 or more a region cut from it read at the
        # frame's scale, as the crop decodes read a unit's region
        rule = T6.scale_of(model, g.shape[1], g.shape[0])
        scales = (1, 2) if k == 0 else (3 - rule if rng.random() < 0.7 else rule,) if rng.random() < forced_p else ()
        for f in scales:                      # both scales on every blob's first frame
            fl = (SCALE1 if f == 1 else SCALE2) | (P.NO_DARK if rng.random() < 0.3 else 0)
            rf = T6.spot(model, img, scale=f)
            forced_cases.append((img, base | fl | int(rng.integers(0, 2))))
            forced_exp.append((k, f"scale{f}", twin_line6(rf, dark=not fl & P.NO_DARK)))
            cov.forced += 1
        if g.shape[0] >= 200 and g.shape[1] >= 400 and rng.random() < forced_p:
            w0, h0 = int(rng.integers(40, g.shape[1] // 2)), int(rng.integers(16, g.shape[0] // 2))
            x0, y0 = int(rng.integers(0, g.shape[1] - w0)), int(rng.integers(0, g.shape[0] - h0))
            region = np.ascontiguousarray(img[y0:y0 + h0, x0:x0 + w0])
            f = T6.scale_of(model, g.shape[1], g.shape[0])
            rf = T6.spot(model, region, scale=f)
            forced_cases.append((region, base | (SCALE1 if f == 1 else SCALE2) | int(rng.integers(0, 2))))
            forced_exp.append((k, f"crop{w0}x{h0}@{f}", twin_line6(rf)))
            cov.forced += 1
    batch = Path(td) / f"{name}.bin"
    P.write_batch(batch, cases)
    old = v3 and model.version != T6.VERSION_TAG
    sel = [g for g, _, _, _ in imgs][::maps_every] if maps_every else []
    # the maps at the rule's scale, and (version 4, now and then) at the other scale forced
    msc = [None if old or rng.random() < 0.7 else 3 - T6.scale_of(model, g.shape[1], g.shape[0]) for g in sel]
    mb = Path(td) / f"{name}_maps.bin"
    fbatch = Path(td) / f"{name}_forced.bin"
    jobs = [("batch", bpath, batch)]
    if sel:
        P.write_batch(mb, [(g, 0 if f is None else SCALE1 if f == 1 else SCALE2) for g, f in zip(sel, msc)])
        jobs.append(("maps", bpath, mb))
    if old:
        jobs.append(("batch3", bpath, batch))
    if forced_cases:
        P.write_batch(fbatch, forced_cases)
        jobs.append(("batch", bpath, fbatch))
    outs = dict(zip((m if m != "batch" or i == 0 else "forced" for i, (m, _, _) in enumerate(jobs)),
                    multi(binary, jobs, td, name)))
    got = outs["batch"]
    fgot = outs.get("forced", [])
    if len(got) != len(cases) or len(fgot) != len(forced_cases):
        return len(cases), 0, 0, [(name, -1, f"{len(got)} + {len(fgot)} output lines for {len(cases)} + "
                                             f"{len(forced_cases)} records", "")]
    for (k, form, e), g in list(zip(exp, got)) + list(zip(forced_exp, fgot)):
        if g != e:
            bad.append((name, k, f"form {form}, shape {imgs[k][0].shape}: C  {g[:400]}", f"py {e[:400]}"))
    extra = 0
    if noms and model.form != "ms" and model.version != T3.VERSION_MS:
        other = P.run_driver(noms, ["--batch", bpath, batch]).splitlines()
        extra += len(other)
        if other != got:
            bad.append((name, -1, "the NO_MS driver prints other lines", ""))
        if forced_cases and P.run_driver(noms, ["--batch", bpath, fbatch]).splitlines() != fgot:
            bad.append((name, -1, "the NO_MS driver prints other lines at a forced scale", ""))
    if old:
        b = P.run_driver(v3, ["--batch", bpath, batch]).splitlines()
        extra += len(b)
        if outs["batch3"] != b:
            bad.append((name, -1, "--batch3 differs from vocr_textspot3_test --batch", ""))
        if sel and P.run_driver(v3, ["--maps", bpath, mb]).splitlines() != outs["maps"]:
            bad.append((name, -1, "--maps differs from vocr_textspot3_test's", ""))
        if arenas:
            sizes = [(960, 544), (480, 272), (600, 7)]
            sizes += sorted({(g.shape[1], g.shape[0]) for g, _, _, _ in imgs} - set(sizes))[:arena_frames]
            mine = P.run_driver(binary, ["--arenas", bpath, *[v for wh in sizes for v in wh]]).splitlines()
            for (w, h), a in zip(sizes, mine):
                ref = " ".join(P.run_driver(v3, [kind, bpath, w, h]).strip() for kind in ("--frame-arena", "--arena"))
                if a != ref:
                    bad.append((name, -1, f"arenas {w}x{h}: {a} != vocr_textspot3's {ref}", ""))
    nmaps = 0
    if sel:
        out = outs["maps"]
        planes = 4 + model.tags
        for i, g in enumerate(sel):
            rc, hm, wm = map(int, out[2 * i + 1].split())
            ref = T6.run_maps(model, T6.net_input(model, np.ascontiguousarray(g), msc[i])[0])
            vals = np.array(out[2 * i].split()[1:], np.int64)
            ok = rc == 0 and vals.size == planes * hm * wm and (planes, hm, wm) == ref.shape and \
                (vals.reshape(hm, wm, planes).transpose(2, 0, 1) == ref).all()
            nmaps += 1
            if not ok:
                bad.append((name, i * maps_every, f"maps differ (rc {rc}, {hm}x{wm} vs {ref.shape[1:]})", ""))
    return len(cases) + len(forced_cases), nmaps, extra, bad


# ---------------------------------------------------------------- the loader

def recrc_net(b):
    """Both CRCs fixed: the network's (VTDT or VTMS, at 32) and the VTSP blob's."""
    b = bytearray(b)
    if len(b) >= 48 and b[32:36] in (b"VTDT", b"VTMS"):
        struct.pack_into("<I", b, 44, zlib.crc32(bytes(b[48:])) & 0xFFFFFFFF)
    if len(b) >= 16:
        struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


def bad_v4():
    """(name, blob) pairs a version-4 loader must refuse: tests/test_textspot6_twin.test_refusals' and more."""
    t = M6.random_blob(TINY_T, 1, 4, n_cal=2)[0]
    m = M6.random_blob(TINY_M, 1, 4, n_cal=2)[0]
    m2 = M6.random_blob(TINY_M, 2, 2, n_cal=2)[0]
    out = [("v4 short", t[:31]), ("v4 truncated", recrc_net(t[:-4])), ("v4 empty net", T6.MAGIC + t[4:32])]

    def edit(name, blob, fn, crc=True):
        b = bytearray(blob)
        fn(b)
        out.append((name, recrc_net(b) if crc else bytes(b)))

    for tags in (0, 5, 255):
        edit(f"tags {tags}", t, lambda b, v=tags: b.__setitem__(19, v))
    edit("tags 3 under a 32-output trunk", t, lambda b: b.__setitem__(19, 3))
    edit("tags 2 under a 32-output VTMS", m, lambda b: b.__setitem__(19, 2))
    edit("tags 4 under a 24-output VTMS", m2, lambda b: b.__setitem__(19, 4))
    edit("reserved 30", t, lambda b: b.__setitem__(30, 1))
    edit("reserved 31", m, lambda b: b.__setitem__(31, 1))
    edit("conf_min 0", t, lambda b: b.__setitem__(18, 0))
    edit("conf_min 128", t, lambda b: b.__setitem__(18, 128))
    edit("offset 36", t, lambda b: struct.pack_into("<I", b, 20, 36))
    edit("size - 1", t, lambda b: struct.pack_into("<I", b, 24, len(b) - 33))
    edit("version 5", t, lambda b: struct.pack_into("<I", b, 4, 5))
    edit("version 0", t, lambda b: struct.pack_into("<I", b, 4, 0))
    edit("total + 1", t, lambda b: struct.pack_into("<I", b, 8, len(b) + 1))
    edit("CRC", t, lambda b: b.__setitem__(40, b[40] ^ 1), crc=False)
    edit("outer CRC only", m, lambda b: b.__setitem__(12, b[12] ^ 1), crc=False)
    edit("neither magic", t, lambda b: b.__setitem__(slice(32, 36), b"XXXX"))
    edit("VTDT magic on a VTMS section", m, lambda b: b.__setitem__(slice(32, 36), b"VTDT"))
    edit("VTMS magic on a trunk", t, lambda b: b.__setitem__(slice(32, 36), b"VTMS"))
    edit("trunk outputs 28", t, lambda b: b.__setitem__(32 + 19, 28))
    edit("trunk outputs 16", t, lambda b: b.__setitem__(32 + 19, 16))
    edit("trunk stride 4", t, lambda b: b.__setitem__(32 + 18, 4))
    edit("VTMS outputs 16", m, lambda b: b.__setitem__(32 + 19, 16))
    edit("VTMS outputs 24 under 4 tags", m, lambda b: b.__setitem__(32 + 19, 24))
    # odd T in a multi-scale head (16 + 4 T not a multiple of 8): a VTMS section whose head has 20 or 28 channels
    for tags in (1, 3):
        net = bytearray(m[32:])
        n_s, n_o = struct.unpack_from("<H", net, 16)[0], struct.unpack_from("<H", net, 22)[0]
        rec = 64 + 32 * (n_s + n_o - 1)
        net[19] = 16 + 4 * tags
        struct.pack_into("<H", net, rec + 6, 16 + 4 * tags)
        struct.pack_into("<I", net, 12, zlib.crc32(bytes(net[16:])) & 0xFFFFFFFF)
        body = struct.pack("<HBBIIHH", 480, 1, tags, 32, len(net), 272, 0) + bytes(net)
        out.append((f"VTMS head of {16 + 4 * tags} channels (T {tags})",
                    T6.MAGIC + struct.pack("<III", 4, 32 + len(net), zlib.crc32(body) & 0xFFFFFFFF) + body))
    # the multi-scale head's channels beyond the 16th: a -128 weight, a multiplier 0, a shift 0 or 63, relu, its
    # sections past the blob
    net = m[32:]
    n_s, n_o = struct.unpack_from("<H", net, 16)[0], struct.unpack_from("<H", net, 22)[0]
    rec = 32 + 64 + 32 * (n_s + n_o - 1)
    wo, bo, mo, so = struct.unpack_from("<IIII", m, rec + 16)
    cin = T6.Model(m).net.ops[-1].cin
    edit("VTMS head weight -128 in channel 20", m, lambda b: b.__setitem__(32 + wo + 20 * cin + 1, 0x80))
    edit("VTMS head weight -128 in channel 31", m, lambda b: b.__setitem__(32 + wo + 31 * cin, 0x80))
    edit("VTMS head multiplier 0 in channel 17", m, lambda b: struct.pack_into("<i", b, 32 + mo + 4 * 17, 0))
    edit("VTMS head multiplier < 0 in channel 30", m, lambda b: struct.pack_into("<i", b, 32 + mo + 4 * 30, -5))
    edit("VTMS head shift 0 in channel 24", m, lambda b: b.__setitem__(32 + so + 24, 0))
    edit("VTMS head shift 63 in channel 31", m, lambda b: b.__setitem__(32 + so + 31, 63))
    edit("VTMS head relu", m, lambda b: b.__setitem__(rec + 3, 1))
    edit("VTMS head kind dw", m, lambda b: b.__setitem__(rec, 1))
    edit("VTMS head cout 24", m, lambda b: struct.pack_into("<H", b, rec + 6, 24))
    edit("VTMS head shift section past the blob", m, lambda b: struct.pack_into("<I", b, rec + 28, len(m) - 32 - 20))
    edit("VTMS head weights past the blob", m, lambda b: struct.pack_into("<I", b, rec + 16, len(m) - 32 - 16 * cin))
    # the trunk's head: relu, the last layer's cout
    tn = t[32:]
    n = struct.unpack_from("<H", tn, 16)[0]
    trec = 32 + 64 + 32 * (n - 1)
    edit("trunk head relu", t, lambda b: b.__setitem__(trec + 4, 1))
    edit("trunk head shift 0 in channel 31", t, lambda b: b.__setitem__(32 + struct.unpack_from("<I", b, trec + 24)[0]
                                                                         + 31, 0))
    edit("trunk head multiplier 0 in channel 30", t,
         lambda b: struct.pack_into("<i", b, 32 + struct.unpack_from("<I", b, trec + 20)[0] + 4 * 30, 0))
    edit("trunk min cells 0", t, lambda b: b.__setitem__(32 + 21, 0))
    edit("VTMS min cells 5", m, lambda b: b.__setitem__(32 + 21, 5))
    edit("trunk name byte 0x80", t, lambda b: b.__setitem__(32 + 40, 0x80))
    edit("VTMS name byte 0xff", m, lambda b: b.__setitem__(32 + 33, 0xFF))
    # 5 tags: a trunk of 16 + 4 x 5 = 36 outputs, otherwise valid (textspot6_twin.TrunkNet takes it), under a header
    # that says 5 tags (the twin refuses tags > 4 at the header)
    for tags, form in ((5, "trunk"), (6, "trunk"), (6, "ms")):
        out.append((f"{tags} tags, a {form} of {16 + 4 * tags} outputs", raw_v4(probe6_net(form, tags), tags)))
    return out


def raw_v4(net, tags, half_above=480, conf_min=1, half_above_h=0, r1=0):
    """A version-4 VTSP blob around a network, without textspot6_twin.pack's checks."""
    body = struct.pack("<HBBIIHH", half_above, conf_min, tags, 32, len(net), half_above_h, r1) + bytes(net)
    return T6.MAGIC + struct.pack("<III", 4, 32 + len(net), zlib.crc32(body) & 0xFFFFFFFF) + body


def good_v4():
    """(name, blob) version-4 blobs every loader of version 4 accepts (edge values)."""
    t = M6.random_blob(TINY_T, 1, 4, n_cal=2)[0]
    m = M6.random_blob(TINY_M, 1, 4, n_cal=2)[0]
    out = [("v4 trunk", t), ("v4 ms", m)]
    for name, blob in (("trunk", t), ("ms", m)):
        mm = T6.Model(blob)
        net = blob[32:]
        out += [(f"{name} h0 0", T6.pack(net, mm.tags, 0, 1, 0)),
                (f"{name} h65535 c127", T6.pack(net, mm.tags, 65535, 127, 65535)),
                (f"{name} mc1", T6.pack(set_min_cells(net, 1), mm.tags, 480, 1, 272)),
                (f"{name} mc4", T6.pack(set_min_cells(net, 4), mm.tags, 1, 2, 1))]
    for t_ in (1, 2, 3):
        out.append((f"trunk tags {t_}", M6.random_blob(TINY_T, 2, t_, n_cal=2)[0]))
    out.append(("ms tags 2", M6.random_blob(TINY_M, 2, 2, n_cal=2)[0]))
    return out


BOUNDARY = (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 20, 24, 28, 32, 33, 63, 64, 65, 127, 128, 254, 255)


def mutants(blob, rng, n):
    """n field mutants of a good version-4 blob, both CRCs fixed: a header or record byte set to a boundary value, a
    record's u16 / u32 field nudged, a weight set to -128 or a multiplier / shift set to a corner (most of them in the
    head, and there in its tag channels)."""
    m = T6.Model(blob)
    net = blob[32:]
    ms = net[:4] == b"VTMS"
    n_s = struct.unpack_from("<H", net, 16)[0]
    n_o = struct.unpack_from("<H", net, 22)[0] if ms else 0
    nrec = n_s + n_o
    layers = (list(m.net.stream) + list(m.net.ops)) if ms else list(m.net.layers)
    out = []
    for k in range(n):
        b = bytearray(blob)
        kind = rng.random()
        if kind < 0.15:                                       # the VTSP header
            b[int(rng.integers(16, 32))] = int(rng.choice(BOUNDARY))
        elif kind < 0.3:                                      # the network's header
            b[32 + int(rng.integers(16, 24))] = int(rng.choice(BOUNDARY))
        elif kind < 0.7:                                      # a record's field
            ri = nrec - 1 - int(rng.integers(0, 2)) if rng.random() < 0.4 else int(rng.integers(0, nrec))
            r = 32 + 64 + 32 * ri
            op = ms and ri >= n_s
            at = int(rng.integers(0, 32))
            if at < (8 if op else 8) or rng.random() < 0.4:
                b[r + at] = int(rng.choice(BOUNDARY))
            else:
                if op:
                    fields = (6, 14, 16, 20, 24, 28)          # cout, reserved, weight / bias / mult / shift offsets
                else:
                    fields = (8, 10, 12, 16, 20, 24, 28)      # cin, cout, the offsets, reserved
                at = int(rng.choice(fields))
                size = 2 if at < 16 and (op or at < 12) else 4
                v = int.from_bytes(b[r + at:r + at + size], "little") + int(rng.choice((-4, -3, -1, 1, 2, 4, 8, 64)))
                b[r + at:r + at + size] = (v % (1 << (8 * size))).to_bytes(size, "little")
        else:                                                 # a weight, multiplier or shift
            ri = nrec - 1 if rng.random() < 0.6 else int(rng.integers(0, nrec))
            r = 32 + 64 + 32 * ri
            op = ms and ri >= n_s
            wo, bo, mo, so = struct.unpack_from("<IIII", b, r + (16 if op else 12))
            L = layers[ri]
            cout = L.w.shape[0]
            per = int(np.prod(L.w.shape[1:]))
            c = int(rng.integers(16, cout)) if cout > 16 and rng.random() < 0.7 else int(rng.integers(0, cout))
            what = rng.random()
            if what < 0.4:
                b[32 + wo + c * per + int(rng.integers(0, per))] = 0x80
            elif what < 0.7:
                struct.pack_into("<i", b, 32 + mo + 4 * c, int(rng.choice((0, -1, -(1 << 31), 1, (1 << 31) - 1))))
            else:
                b[32 + so + c] = int(rng.choice((0, 1, 62, 63, 255)))
        out.append((f"mutant{k}", recrc_net(b)))
    return out


def twin_code(blob):
    try:
        T6.Model(blob)
        return 0
    except ValueError:
        return E_BLOB


def is_ms_blob(blob):
    try:
        m = T6.Model(blob)
    except ValueError:
        return False
    return m.version == T3.VERSION_MS or m.form == "ms"


def check_loader(binary, v3, noms, blobs, args, td):
    """The C loaders against the twin's refusals: (mismatches, cases)."""
    import test_textspot2_twin as TT2
    import test_textspot3_twin as TT3
    rng = np.random.default_rng(55)
    cases = [(n, b) for n, b in bad_v4() + good_v4()]
    cases += [(f"t3 {n}", b) for n, b in TT3.BAD + TT3.good_blobs()]
    cases += [(f"t2 {n}", b) for n, b in TT2.bad_blobs() + TT2.good_blobs()]
    v4 = [b for b in blobs if b[1][4] == 4]
    picks = v4[::max(1, len(v4) // 6)][:6] + [b for b in blobs if b[1][4] in (1, 2, 3)][:4]
    for name, blob in picks:
        if blob[4] == 4:
            cases += [(f"{name}_{n}", b) for n, b in mutants(blob, rng, args.mutants)]
        for k in range(8):
            b = bytearray(blob)
            at = int(rng.integers(0, len(b))) if k < 4 else int(rng.integers(4, min(len(b), 64 + 32 * 8)))
            b[at] ^= int(rng.integers(1, 256))
            cases.append((f"{name}_dmg{k}", recrc_net(b) if k % 2 else bytes(b)))
        cases.append((f"{name}_ok", blob))
    paths, want = [], []
    for i, (name, b) in enumerate(cases):
        pth = Path(td) / f"load_{i}.bin"
        pth.write_bytes(b)
        paths.append(pth)
        want.append(twin_code(b))

    def codes(drv):
        got = []
        for i in range(0, len(paths), 200):
            got += [int(v) for v in P.run_driver(drv, ["--load", *paths[i:i + 200]]).split()]
        return got
    bad = [(cases[i][0], w, g) for i, (w, g) in enumerate(zip(want, codes(binary))) if w != g]
    if v3:
        old = codes(v3)
        for i, (name, b) in enumerate(cases):
            if len(b) >= 8 and b[:4] == T6.MAGIC and struct.unpack_from("<I", b, 4)[0] in (1, 2, 3) and \
                    old[i] != want[i]:
                bad.append((name, want[i], f"vocr_textspot3_test {old[i]}"))
    nn = 0
    if noms:
        for i, (g, (name, b)) in enumerate(zip(codes(noms), cases)):
            w = E_BLOB if is_ms_blob(b) else want[i]
            nn += w == E_BLOB and want[i] == 0
            if g != w:
                bad.append((name, w, f"NO_MS {g}"))
    return bad, len(cases), nn


# ---------------------------------------------------------------- versions 1-3

def old_blobs(args):
    """[(name, VTSP bytes)] of versions 1-3: random version-3 networks of --v3-probe-shapes (train/textspot3_model's
    PROBES, the registered M shapes, or tiny3), random op graphs, a wild network and every version-3 probe network
    (tests/textspot3_parity's builders), at (480, 272) and every third also at (0, 0), (600, 0) and (0, 300), some with
    conf_min > 1; then textspot2_parity's version-1 and version-2 blobs of --v1-shapes / --d2s-shapes."""
    import textspot2_parity as Q2
    rng = np.random.default_rng(4321)
    nets = []
    shapes = {"tiny3": M3.m_shape("tiny3", 4, 8, 8, [8], [1], [(16, [1]), (16, [2])], 8, 8, 1), **m_shapes_all()}
    for i, name in enumerate(args.v3_probe_shapes):
        for k in range(args.v3_seeds):
            seed = 3000 * (k + 1) + i
            nets.append((f"{name}_seed{seed}", M3.random_net(shapes[name], seed, args.n_cal)))
    for g in range(args.v3_graphs):
        nets.append((f"graph_{g}", Q3.graph_net(500 + g)))
    if args.v3_graphs or args.v3_probe_shapes:
        nets.append(("wild_tiny3_1", Q3.wild_net(shapes["tiny3"], 1)))
        for v in range(min(args.v3_probes, len(TP.PROBE_VARIANTS))):
            nets.append((f"probe_ms_v{v}", Q3.probe_net(v, None if v % 2 else 1 + v % 4)))
    out = []
    for j, (name, net) in enumerate(nets):
        for ha, hh in (((480, 272), (0, 0), (600, 0), (0, 300)) if j % 3 == 0 else ((480, 272),)):
            cm = int(rng.integers(2, 48)) if (j + ha // 300 + hh // 300) % 4 == 3 else 1
            out.append((f"{name}_h{ha}x{hh}_c{cm}", T3.pack(net, ha, cm, hh)))
    if args.v1_shapes or args.d2s_shapes:
        nets = Q2.nets(1, args.v1_shapes, args.d2s_shapes, wild=1, n_cal=args.n_cal)
        keep = {f"probe_{k}_v{v}" for v in range(min(args.v12_probes, len(TP.PROBE_VARIANTS))) for k in
                ("s4", "s8", "d2s")}
        for j, (name, vtdt) in enumerate(n for n in nets if not n[0].startswith("probe") or n[0] in keep):
            for ha in (0, 600):
                cm = int(rng.integers(2, 48)) if (j + ha // 600) % 4 == 3 else 1
                out.append((f"{name}_h{ha}_c{cm}", T3.pack(vtdt, ha, cm)))
    return out


# ---------------------------------------------------------------- the run

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="the driver (may carry a prefix, e.g. 'qemu-arm -cpu cortex-a9 ./x')")
    ap.add_argument("--v3-bin", default="", help="runtime/vocr_textspot3_test (version-1/2/3 lines and arenas)")
    ap.add_argument("--noms-bin", default="", help="the driver built with -DVOCR_TEXTSPOT6_NO_MS")
    ap.add_argument("--t-shapes", default="tinyt,t64n8c16", help="T shapes (tinyt, train/textspot3_shapes', "
                    "eval/textspot6_a9.T_FULL's)")
    ap.add_argument("--m-shapes", default="tinym,qS4,qM4,qL5", help="M shapes (tinym, textspot3_model.PROBES, the "
                    "registered qS4 / qM4 / qL5)")
    ap.add_argument("--t-tags", default="1,2,3,4")
    ap.add_argument("--m-tags", default="2,4")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--probe-frames", type=int, default=4)
    ap.add_argument("--full-every", type=int, default=3)
    ap.add_argument("--maps-every", type=int, default=2)
    ap.add_argument("--arena-every", type=int, default=2, help="versions 1-3: every k-th blob's arenas compared")
    ap.add_argument("--arena-frames", type=int, default=99, help="... at 3 sizes and up to this many frame sizes")
    ap.add_argument("--graphs", type=int, default=2)
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--selftest-blobs", type=int, default=8)
    ap.add_argument("--mutants", type=int, default=24)
    ap.add_argument("--n-cal", type=int, default=2)
    ap.add_argument("--v3-seeds", type=int, default=1)
    ap.add_argument("--v3-frames", type=int, default=2)
    ap.add_argument("--v3-probe-shapes", default="tiny3,pS1", help="version-3 shapes (tiny3, PROBES, qS4 ...)")
    ap.add_argument("--v3-probes", type=int, default=6, help="version-3 probe networks (variants 0 ..)")
    ap.add_argument("--v12-probes", type=int, default=6, help="version-1 and version-2 probe variants (0 ..)")
    ap.add_argument("--probe-variants", default="0,1,2", help="the version-4 multi-scale probes' variants")
    ap.add_argument("--probe-mc", type=int, default=2, help="min-cells variants per version-4 probe (1 or 2)")
    ap.add_argument("--probe-cells", default="70x130",
                    help="the random designed maps' largest rows x columns (stride-4 cells; 4 px each)")
    ap.add_argument("--v3-graphs", type=int, default=1)
    ap.add_argument("--v1-shapes", default="s4_c4,s8_c8")
    ap.add_argument("--d2s-shapes", default="s8_c24n4x2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blob", action="append", default=[])
    args = ap.parse_args(argv)
    t0 = time.time()
    args.t_shapes = [s for s in args.t_shapes.split(",") if s]
    args.m_shapes = [s for s in args.m_shapes.split(",") if s]
    args.t_tags = [int(s) for s in args.t_tags.split(",") if s]
    args.m_tags = [int(s) for s in args.m_tags.split(",") if s]
    args.probe_variants = [int(s) for s in args.probe_variants.split(",") if s]
    args.probe_cells = tuple(int(v) for v in args.probe_cells.split("x"))
    import textspot2_shapes as S2
    S2.register()
    rng = np.random.default_rng(args.seed)
    args.v3_probe_shapes = [s for s in args.v3_probe_shapes.split(",") if s]
    args.v1_shapes = [s for s in args.v1_shapes.split(",") if s]
    args.d2s_shapes = [s for s in args.d2s_shapes.split(",") if s]
    v4 = v4_blobs(args)
    probes = probe_runs(rng, args, v4)
    old = old_blobs(args)
    runs = nframes = nmaps = nextra = 0
    bad = []
    cov = Coverage()
    with tempfile.TemporaryDirectory(prefix="textspot6-parity-") as td:
        if args.selftest:
            pick = []
            for want in (lambda n, b: b[4] == 4 and n.startswith("probe6_trunk"),
                         lambda n, b: b[4] == 4 and n.startswith("probe6_ms_v2"),
                         lambda n, b: b[4] == 4 and T6.Model(b).form == "trunk" and not n.startswith(("probe", "wild")),
                         lambda n, b: b[4] == 4 and T6.Model(b).form == "ms" and not n.startswith(("probe", "wild")),
                         lambda n, b: n.startswith("wild6"), lambda n, b: n.startswith("graph6"),
                         lambda n, b: b[4] == 3, lambda n, b: b[4] == 2, lambda n, b: b[4] == 1):
                pick += [x for x in [(n, b) for n, b, _ in probes] + v4 + old if want(*x) and x not in pick][:1]
            paths, trunk = [], []
            for name, blob in pick[:args.selftest_blobs]:
                pth = Path(td) / f"self_{name}.vtsp"
                pth.write_bytes(blob)
                paths.append(pth)
                if not is_ms_blob(blob):
                    trunk.append(pth)
            print(P.run_driver(args.bin, ["--selftest", args.selftest, *paths]).strip())
            if args.noms_bin and trunk:
                print("NO_MS: " + P.run_driver(args.noms_bin, ["--selftest", max(2, args.selftest // 2),
                                                               *trunk]).strip())
        lbad, nload, nnoms = check_loader(args.bin, args.v3_bin, args.noms_bin, v4 + [(n, b) for n, b, _ in probes]
                                          + old, args, td)
        for name, w, g in lbad[:12]:
            bad.append((name, -1, f"loader: C {g}", f"twin {w}"))
        for bi, (name, blob) in enumerate(v4):
            model = T6.Model(blob)
            screen = P.SCREENS[(bi // args.full_every) % len(P.SCREENS)] if args.full_every and \
                bi % args.full_every == 0 else None
            imgs = [(g, None, 0, None) for g in P.frames_for(rng, model, args.frames, False, screen)]
            r, mp, ex, b = check_frames(args.bin, name, blob, imgs, rng, td, cov, args.maps_every, args.noms_bin)
            runs, nframes, nmaps, nextra = runs + r, nframes + len(imgs), nmaps + mp, nextra + ex
            bad += b
        for name, blob, imgs in probes:
            r, mp, ex, b = check_frames(args.bin, name, blob, imgs, rng, td, cov, 1, args.noms_bin)
            runs, nframes, nmaps, nextra = runs + r, nframes + len(imgs), nmaps + mp, nextra + ex
            bad += b
        for bi, (name, blob) in enumerate(old):
            model = T6.Model(blob)
            probe = name.startswith("probe")
            screen = P.SCREENS[(bi // args.full_every) % len(P.SCREENS)] if args.full_every and \
                bi % args.full_every == 0 else None
            if model.version == T3.VERSION_MS:
                frames = Q3.frames_for(rng, model.t3, args.v3_frames + probe, probe, screen)
            else:
                frames = P.frames_for(rng, model.t3.t2.v1 if model.version == 1 else model.t3.t2,
                                      args.v3_frames + probe, probe, screen)
            imgs = [(g, None, 0, None) for g in frames]
            arenas = bool(args.arena_every) and bi % args.arena_every == 0
            r, mp, ex, b = check_frames(args.bin, name, blob, imgs, rng, td, cov, args.maps_every, args.noms_bin,
                                        args.v3_bin, arenas, args.arena_frames)
            runs, nframes, nmaps, nextra = runs + r, nframes + len(imgs), nmaps + mp, nextra + ex
            bad += b
    n4 = len(v4) + len(probes)
    if bad:
        for name, k, c, p in bad[:12]:
            print(f"MISMATCH {name} frame {k}: {c}\n    {p}")
        print(f"TEXTSPOT6_PARITY_FAIL {len(bad)} of {runs} runs")
        return 1
    print(f"coverage: {cov}")
    print(f"TEXTSPOT6_PARITY_OK {runs} runs ({nframes} frames, {n4 + len(old)} blobs ({n4} of version 4: "
          f"{len(probes)} probes on designed maps; {len(old)} of versions 1-3), {nmaps} maps, {nload} loader cases "
          f"({nnoms} refused by NO_MS only), {nextra} runs of vocr_textspot3_test or the NO_MS driver equal; "
          f"{time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

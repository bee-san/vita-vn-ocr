#!/usr/bin/env python3
"""Parity check: runtime/vocr_textspot3.c against its Python twin synth/textspot3_twin.py (the specification), for
VTSP version 1, 2 and 3 blobs; versions 1 and 2 also against textspot2's frozen runtime (runtime/vocr_textspot2.c).

  textspot3_parity.py --bin build/textspot3/vocr_textspot3_test [--v2-bin build/textspot3/vocr_textspot2_test]
                      [--seeds 1] [--frames 4] [--probe-frames 6] [--full-every 4] [--selftest 12]
                      [--selftest-blobs 5] [--maps-every 2] [--arena-every 1] [--seed 0] [--probe-shapes S,...]
                      [--graphs 4] [--v1-shapes S,...] [--d2s-shapes S,...] [--blob FILE ...]

Blobs:
  - version 3: random networks (train/textspot3_model.random_int_layers) of --probe-shapes (default: every shape of
    train/textspot3_model.PROBES) and of the protocol's registered M shapes; random op graphs outside the grammar
    (graph_net: up to 3 sources at any upsampling that meets, dilations 1..8, stride-2 chains, unused maps);
    wild-requantisation networks; the version-3 probe networks (probe_net, every variant, min cells 1-4), whose
    maps and results the twin must give as #97's stride-4 probe does; each at the exporter's half_above 480 /
    half_above_h 272 and some at (0, 0), (600, 0) and (0, 300), every fourth with a conf_min > 1;
  - versions 1 and 2: tests/textspot2_parity's random, wild and probe networks of a few shapes (--v1-shapes,
    --d2s-shapes), wrapped as textspot2_parity wraps them;
  - --blob adds VTSP files.
Frames: tests/textspot_parity's procedural full frames (960x544, 480x272, 725x544, 870x544 on every --full-every-th
blob), edge and odd sizes, and for the probes frames designed cell by cell at stride 4 at the network's input (more
than 64 candidates, truncated), GREY and RGBA; the in-memory and row-callback forms, VOCR_TEXTSPOT_NO_DARK, padded
strides, odd arenas, the arena for every frame up to the size. The C driver and the twin (textspot3_twin.spot; maps:
textspot3_twin.run_maps) must agree exactly on the maps the decode reads, the boxes, blocks, order, pick and
truncated flag; with --v2-bin every version-1 and version-2 batch must also print exactly what
runtime/vocr_textspot2_test prints for it, and every such blob's arena (for the frame and up to it) must be that
driver's. The C loader must refuse exactly the blobs the twin refuses (tests/test_textspot3_twin's refusals and
damaged copies of blobs of every version). --selftest N also runs the driver's own invariants on N frames of a few
blobs of every version. Prints TEXTSPOT3_PARITY_OK <runs> runs (<frames> frames, ...).

Builders the tests share:
  probe_net(variant)        a version-3 network that carries the d2s probe's stride-8 codes through identity map ops
                            (a dilated depthwise, a stride-2 chain to stride 64, zero-weight branches read through
                            upsamplings by 2, 4 and 8, a concatenation) into the d2s probe's head: its maps and
                            results must be #97's stride-4 probe's exactly
  wild_net(shape, seed)     random_int_layers with extreme requantisation corners (biases over the int32 range,
                            multipliers 1 .. 2^31 - 1, shifts 1 .. 62) and weights at +-127
  graph_net(seed)           a random op graph (not the grammar): pw ops over 1..3 earlier maps at any upsampling
                            that meets, dw ops of strides 1 and 2 and dilations 1..8, channels 8..64
"""
import hashlib
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_parity as TP  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_twin as T3  # noqa: E402

ONE = 1 << 30


def _q(L, wq, bq, m, sh):
    return {**L, "wq": np.asarray(wq, np.int8), "bq": np.asarray(bq, np.int64), "m": np.asarray(m, np.int64),
            "sh": np.asarray(sh, np.int64)}


def stream_layer(kind, k, stride, cin, cout, wq, bq, m, sh, relu=False, dil=1):
    return _q({"kind": kind, "k": k, "stride": stride, "dil": dil, "relu": relu, "cin": cin, "cout": cout,
               "head": False, "part": "stream"}, wq, bq, m, sh)


def op_layer(kind, src, up, cin, cout, wq, bq, m, sh, relu=False, stride=1, dil=1, head=False, out_stride=8):
    return _q({"kind": kind, "k": 3 if kind == "dw" else 1, "stride": stride, "dil": dil, "relu": relu,
               "cin": cin, "cout": cout, "src": list(src), "up": list(up), "head": head, "part": "op",
               "out_stride": out_stride}, wq, bq, m, sh)


def identity_dw(c):
    w = np.zeros((c, 1, 3, 3), np.int8)
    w[:, 0, 1, 1] = 1
    return w


def identity_pw(cout, cin, at=0):
    w = np.zeros((cout, cin, 1, 1), np.int8)
    for i in range(min(cout, cin - at)):
        w[i, at + i, 0, 0] = 1
    return w


def probe_layers(variant=0):
    """probe_net's integer layers (pack_net's dicts)."""
    centre = np.zeros((1, 1, 3, 3), np.int8)
    centre[0, 0, 1, 1] = 1
    taps = np.zeros((8, 1, 3, 3), np.int8)
    for dy in (0, 1):
        for dx in (0, 1):
            taps[2 * dy + dx, 0, 1 + dy, 1 + dx] = 1
    out = [stream_layer("conv", 3, 2, 1, 1, centre, [0], [ONE], [30]),
           stream_layer("dw", 3, 2, 1, 1, centre, [0], [ONE], [30]),
           stream_layer("conv", 3, 2, 1, 8, taps, [0] * 8, [ONE] * 8, [30] * 8)]     # F8: channels 4..7 zero
    ops = [op_layer("dw", [0], [0], 8, 8, identity_dw(8), [0] * 8, [ONE] * 8, [30] * 8, dil=2),          # map 1
           op_layer("dw", [1], [0], 8, 8, identity_dw(8), [0] * 8, [ONE] * 8, [30] * 8, stride=2,
                    out_stride=16),                                                                     # map 2
           op_layer("dw", [2], [0], 8, 8, identity_dw(8), [0] * 8, [ONE] * 8, [30] * 8, stride=2,
                    out_stride=32),                                                                     # map 3
           op_layer("dw", [3], [0], 8, 8, identity_dw(8), [0] * 8, [ONE] * 8, [30] * 8, stride=2,
                    out_stride=64),                                                                     # map 4
           op_layer("pw", [4], [0], 8, 8, np.zeros((8, 8, 1, 1)), [5] * 8, [ONE] * 8, [30] * 8, relu=True,
                    out_stride=64),                                                                     # map 5: 5s
           op_layer("pw", [1, 5, 3], [0, 3, 2], 24, 8, np.concatenate(
               [identity_pw(8, 8), np.zeros((8, 16, 1, 1), np.int8)], 1), [0] * 8, [ONE] * 8, [30] * 8)]   # map 6
    slope1, slope2, off1, off2 = TP.PROBE_VARIANTS[variant]
    w = np.zeros((16, 8, 1, 1), np.int8)
    bias, mult, shift = [], [], []
    for dy in (0, 1):
        for dx in (0, 1):
            q = 2 * dy + dx
            for mp, (wt, b, sh) in enumerate(((1, 0, 30), (1, -64, 30), (slope1, 4 * off1, 32),
                                              (slope2, 4 * off2, 32))):
                w[T3.channel(dy, dx, mp), q, 0, 0] = wt
                bias.append(b)
                mult.append(ONE)
                shift.append(sh)
    ops.append(op_layer("pw", [6], [0], 8, 16, w, bias, mult, shift, head=True))
    return out + ops


def probe_net(variant=0, min_cells=None):
    shape = {"name": f"probe_ms_v{variant}"}
    net = M3.pack_net(shape, probe_layers(variant), hashlib.sha256(shape["name"].encode()).digest())
    return net if min_cells is None else set_min_cells(net, min_cells)


def set_min_cells(net, n):
    b = bytearray(net)
    b[21] = n
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    T3.Net(bytes(b))
    return bytes(b)


def probe_reference(model):
    """The stride-4 version-1 probe (#97's) whose maps a version-3 probe must reproduce, as a textspot_twin Model
    with the probe's min cells and conf_min (never halving: the caller feeds it the version-3 network's input)."""
    import textspot_twin as S
    v = int(model.name[len("probe_ms_v"):])
    return S.Model(S.pack(TP.set_min_cells(TP.probe_blob(4, v), model.min_cells), 0, model.conf_min))


def wild_net(shape, seed):
    """random_int_layers of an M shape with some channels' requantisation at the loader's extremes (weights stay in
    +-127, the version-3 range): most such channels saturate."""
    rng = np.random.default_rng(seed)
    q = M3.random_int_layers(shape, seed, n_cal=2)
    for L in q:
        wild = rng.random(L["cout"]) < 0.35
        L["wq"] = L["wq"].copy()
        L["wq"][rng.random(L["wq"].shape) < 0.05] = 127 * (1 - 2 * (seed % 2))
        L["bq"] = np.where(wild, rng.integers(-(1 << 31), 1 << 31, L["cout"]), L["bq"])
        L["m"] = np.where(wild, rng.integers(1, 1 << 31, L["cout"]), L["m"])
        L["sh"] = np.where(wild, rng.integers(1, 63, L["cout"]), L["sh"])
        if wild.any():
            k = int(rng.integers(0, L["cout"]))
            L["bq"][k], L["m"][k], L["sh"][k] = (-(1 << 31), (1 << 31) - 1, 1) if seed % 2 else ((1 << 31) - 1, 1, 62)
    return M3.pack_net({"name": f"wild_{shape['name']}"[:31]}, q,
                       hashlib.sha256(f"wild:{shape['name']}:{seed}".encode()).digest())


def graph_layers(seed, n_ops=None):
    """A random version-3 network outside the grammar: a small stream to stride 8 (F8 of 8..32 channels), then
    random ops (pw over 1..3 earlier maps that meet at one stride, some sources repeated; dw of stride 1 with any
    dilation or stride 2), every map's channels a multiple of 8 up to 64, then the head over 1..3 maps of stride
    8 or deeper; random weights in +-127 and random requantisation (some identity-like, some wild)."""
    rng = np.random.default_rng(seed)
    c8 = int(rng.choice([8, 16, 24, 32]))
    c0 = int(rng.integers(1, 9))
    layers = []

    def wts(shape):
        w = rng.integers(-127, 128, shape).astype(np.int8)
        w[rng.random(shape) < 0.3] = 0
        return w

    def rq(cout, cin_k):
        # spread: an accumulator of sd ~ 60 sqrt(cin_k) to ~30 codes
        target = 30.0 / (60.0 * np.sqrt(cin_k) + 1)
        m, sh = zip(*[M3.E.quantize_multiplier(float(target * rng.uniform(0.3, 3))) for _ in range(cout)])
        b = rng.integers(-2000, 2000, cout)
        if rng.random() < 0.2:                                   # a wild channel or two
            k = int(rng.integers(0, cout))
            m = list(m)
            sh = list(sh)
            m[k], sh[k], b[k] = int(rng.integers(1, 1 << 31)), int(rng.integers(1, 63)), \
                int(rng.integers(-(1 << 31), 1 << 31))
        return b, m, sh

    layers.append(stream_layer("conv", 3, 2, 1, c0, wts((c0, 1, 3, 3)), *rq(c0, 9), relu=True))
    layers.append(stream_layer("dw", 3, 2, c0, c0, wts((c0, 1, 3, 3)), *rq(c0, 9)))
    layers.append(stream_layer("conv", 1, 1, c0, c8, wts((c8, c0, 1, 1)), *rq(c8, c0), relu=True))
    layers.append(stream_layer("dw", 3, 2, c8, c8, wts((c8, 1, 3, 3)), *rq(c8, 9),
                               dil=1))
    maps = [(8, c8)]
    n = int(rng.integers(2, 12)) if n_ops is None else n_ops
    for k in range(n):
        if rng.random() < 0.45:
            s = int(rng.integers(0, len(maps)))
            st = 2 if (rng.random() < 0.4 and maps[s][0] < 64) else 1
            d = 1 if st == 2 else int(rng.integers(1, 9))
            c = maps[s][1]
            layers.append(op_layer("dw", [s], [0], c, c, wts((c, 1, 3, 3)), *rq(c, 9), relu=bool(rng.random() < 0.5),
                                   stride=st, dil=d, out_stride=maps[s][0] * st))
            maps.append((maps[s][0] * st, c))
        else:
            s0 = int(rng.integers(0, len(maps)))
            s_op = maps[s0][0] >> int(rng.integers(0, min(3, int(np.log2(maps[s0][0] // 8))) + 1))
            cands = [i for i, (s, _) in enumerate(maps) if s >= s_op and s // s_op in (1, 2, 4, 8)]
            src = [int(rng.choice(cands)) for _ in range(int(rng.integers(1, 4)))]
            up = [int(np.log2(maps[i][0] // s_op)) for i in src]
            cin = sum(maps[i][1] for i in src)
            if cin > T3.MAX_CHANS:
                src, up, cin = src[:1], up[:1], maps[src[0]][1]
            c = int(rng.integers(1, 9)) * 8
            layers.append(op_layer("pw", src, up, cin, c, wts((c, cin, 1, 1)), *rq(c, cin),
                                   relu=bool(rng.random() < 0.7), out_stride=s_op))
            maps.append((s_op, c))
    cands = [i for i, (s, _) in enumerate(maps) if s in (8, 16, 32, 64)]
    src = [len(maps) - 1] if maps[-1][0] in (8, 16, 32, 64) else []
    src += [int(rng.choice(cands)) for _ in range(int(rng.integers(1, 4)) - len(src))]
    src = src or [int(rng.choice(cands))]
    up = [int(np.log2(maps[i][0] // 8)) for i in src]
    cin = sum(maps[i][1] for i in src)
    b = np.array([0, 0, 32, 32] * 4) + rng.integers(-8, 9, 16)
    m, sh = zip(*[M3.E.quantize_multiplier(float(20.0 / (60.0 * np.sqrt(cin) + 1) * rng.uniform(0.5, 2)))
                  for _ in range(16)])
    layers.append(op_layer("pw", src, up, cin, 16, wts((16, cin, 1, 1)), b, m, sh, head=True))
    return layers


def graph_net(seed, n_ops=None):
    return M3.pack_net({"name": f"graph_{seed}"}, graph_layers(seed, n_ops),
                       hashlib.sha256(f"graph:{seed}".encode()).digest())


# ---------------------------------------------------------------- the parity run

SCALES3 = ((480, 272), (0, 0), (600, 0), (0, 300))


def v3_nets(seeds, probe_shapes, graphs, n_cal=6):
    """[(name, VTMS section)]: random networks of the probe and registered shapes, graphs, wild and probe nets."""
    import textspot3_shapes as S3
    out = []
    shapes = [M3.shape_of(n) for n in probe_shapes] + [M3.check_shape(s) for s in S3.M.values()]
    for i, sh in enumerate(shapes):
        for k in range(seeds):
            seed = 3000 * (k + 1) + i
            out.append((f"{sh['name']}_seed{seed}", M3.random_net(sh, seed, n_cal)))
    for g in range(graphs):
        out.append((f"graph_{g}", graph_net(500 + g)))
    tiny = M3.m_shape("tiny3", 4, 8, 8, [8], [1], [(16, [1]), (16, [2])], 8, 8, 1)
    out.append(("wild_tiny3_1", wild_net(tiny, 1)))
    if shapes:
        out.append((f"wild_{shapes[0]['name']}_2", wild_net(shapes[0], 2)))
    for v in range(len(TP.PROBE_VARIANTS)):
        out.append((f"probe_ms_v{v}", probe_net(v, None if v % 2 else 1 + v % 4)))
    return out


def all_blobs(args):
    """[(name, VTSP bytes)]: version 3 at the scales above (every fourth with a conf_min > 1), then textspot2's
    version-1 and version-2 blobs, then --blob files."""
    import textspot2_parity as Q2
    rng = np.random.default_rng(4321)
    out = []
    for j, (name, net) in enumerate(v3_nets(args.seeds, args.probe_shapes, args.graphs, args.n_cal)):
        scales = SCALES3 if j % 3 == 0 or name.startswith("probe") else SCALES3[:2]
        for ha, hh in scales:
            cm = int(rng.integers(2, 48)) if (j + ha // 300 + hh // 300) % 4 == 3 else 1
            out.append((f"{name}_h{ha}x{hh}_c{cm}", T3.pack(net, ha, cm, hh)))
    if args.v1_shapes or args.d2s_shapes:
        out += Q2.all_blobs(1, args.v1_shapes, args.d2s_shapes, wild=1, n_cal=args.n_cal)
    for p in args.blob:
        out.append((Path(p).name, Path(p).read_bytes()))
    return out


def probe_frame3(rng, model, w, h):
    """A w x h frame whose network input is designed cell by cell at stride 4 (tests/textspot_parity.designed):
    drawn at the input's scale and doubled when the frame is halved (version 3's rule)."""
    import textspot_parity as P
    f = T3.scale_of(model, w, h)
    if f == 1:
        return P.designed(rng, 4, h, w)
    d = P.designed(rng, 4, -(-h // 2), -(-w // 2))
    return np.repeat(np.repeat(d, 2, 0), 2, 1)[:h, :w]


def frames_for(rng, model, n, probe, screen):
    """tests/textspot_parity.frames_for (which reads the model only for probe frames) with version 3's probe frames:
    edge sizes, then designed frames of random sizes, and the screen."""
    import textspot_parity as P
    if not probe:
        return P.frames_for(rng, model, n, False, screen)
    sizes = list(P.EDGE[:max(0, min(len(P.EDGE), n // 3))])
    while len(sizes) < n:
        sizes.append((int(rng.integers(8, 720)), int(rng.integers(8, 560))))
    if screen is not None:
        sizes.append(screen)
    return [probe_frame3(rng, model, w, h) for w, h in sizes]


def check_blob(binary, v2_binary, name, blob, imgs, rng, td, cov, maps_every=1, arenas=True):
    """Every frame through C and the twin (and, for versions 1 and 2, textspot2's driver). (runs, maps, v12 runs,
    mismatches)."""
    import textspot_parity as P
    model = T3.Model(blob)
    ref4 = probe_reference(model) if model.version == T3.VERSION_MS and model.name.startswith("probe_ms_v") else None
    bpath = Path(td) / f"{name}.vtsp"
    bpath.write_bytes(blob)
    cases, exp, bad = [], [], []
    for k, g in enumerate(imgs):
        img = TP.to_rgba(g, rng) if rng.random() < 0.5 else g
        r = T3.spot(model, img)
        cov.add(model, img, r)
        if ref4 is not None:
            import textdet_twin as TD
            import textspot2_twin as T2
            x, f = T3.net_input(model, np.ascontiguousarray(g))
            if T2.spot_maps(ref4, TD.run_net(ref4.net, x), f, np.ascontiguousarray(g)) != T3.spot(model, g):
                bad.append((name, k, "the version-3 probe differs from its stride-4 probe in the twin", ""))
        line, line_nd = P.twin_line(r), P.twin_line(r, dark=False)
        base = P.ODD * int(rng.random() < 0.3) | P.PADDED * int(rng.random() < 0.3) | P.MOST * int(rng.random() < 0.3)
        for form in (0, P.ROWS):
            cases.append((img, base | form))
            exp.append((k, form, line))
        if rng.random() < 0.5:
            form = int(rng.integers(0, 2))
            cases.append((img, base | P.NO_DARK | form))
            exp.append((k, f"nodark{form}", line_nd))
    batch = Path(td) / f"{name}.bin"
    P.write_batch(batch, cases)
    got = P.run_driver(binary, ["--batch", bpath, batch]).splitlines()
    if len(got) != len(cases):
        return len(cases), 0, 0, [(name, -1, f"{len(got)} output lines for {len(cases)} records", "")]
    for (k, form, e), g in zip(exp, got):
        if g != e:
            bad.append((name, k, f"form {form}, shape {imgs[k].shape}: C  {g[:400]}", f"py {e[:400]}"))
    nv = 0
    if v2_binary and model.version != T3.VERSION_MS:
        old = P.run_driver(v2_binary, ["--batch", bpath, batch]).splitlines()
        nv = len(old)
        if old != got:
            bad.append((name, -1, "vocr_textspot3_test and vocr_textspot2_test print different lines", ""))
        sizes = sorted({(g.shape[1], g.shape[0]) for g in imgs} | {(960, 544), (480, 272), (600, 7)})
        for w, h in sizes if arenas else []:
            for kind in ("--arena", "--frame-arena"):
                a1 = P.run_driver(v2_binary, [kind, bpath, w, h]).strip()
                a2 = P.run_driver(binary, [kind, bpath, w, h]).strip()
                if a1 != a2:
                    bad.append((name, -1, f"{kind} {w}x{h}: {a2} != vocr_textspot2's {a1}", ""))
    nmaps = 0
    if maps_every:
        sel = imgs[::maps_every]
        mb = Path(td) / f"{name}_maps.bin"
        P.write_batch(mb, [(g, 0) for g in sel])
        out = P.run_driver(binary, ["--maps", bpath, mb]).splitlines()
        for i, g in enumerate(sel):
            rc, hm, wm = map(int, out[2 * i + 1].split())
            ref = T3.run_maps(model, T3.net_input(model, np.ascontiguousarray(g))[0])
            vals = np.array(out[2 * i].split()[1:], np.int64)
            ok = rc == 0 and vals.size == 4 * hm * wm and (hm, wm) == ref.shape[1:] and \
                (vals.reshape(hm, wm, 4).transpose(2, 0, 1) == ref).all()
            nmaps += 1
            if not ok:
                bad.append((name, i * maps_every, f"maps differ (rc {rc}, {hm}x{wm} vs {ref.shape[1:]})", ""))
    return len(cases), nmaps, nv, bad


def check_loader(binary, blobs, td, per=8):
    """The C loader accepts every blob and refuses exactly what the twin refuses: tests/test_textspot3_twin's
    refusals, and damaged copies (a few edits per blob of each version; every other one with the outer CRC fixed so
    the inner checks decide)."""
    import textspot_parity as P
    rng = np.random.default_rng(5)
    cases = []
    try:
        import test_textspot3_twin as TT
        cases += [(f"bad_{i}", b) for i, (_, b) in enumerate(TT.BAD)]
        cases += [(f"good_{i}", b) for i, (_, b) in enumerate(TT.good_blobs())]
    except ImportError:
        pass
    picks = [b for b in blobs if b[1][4] == 3][:6] + [b for b in blobs if b[1][4] in (1, 2)][:4]
    for name, blob in picks:
        for k in range(per):
            b = bytearray(blob)
            at = int(rng.integers(0, len(b))) if k < per // 2 else int(rng.integers(4, min(len(b), 64 + 32 * 8)))
            b[at] ^= int(rng.integers(1, 256))
            if k % 2:
                struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
                if len(b) > 48 and b[32:36] == b"VTMS":       # the inner CRC too: the field checks decide
                    struct.pack_into("<I", b, 44, zlib.crc32(bytes(b[48:])) & 0xFFFFFFFF)
                    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
            cases.append((f"{name}_dmg{k}", bytes(b)))
        cases.append((f"{name}_ok", blob))
    paths, want = [], []
    for i, (name, b) in enumerate(cases):
        try:
            T3.Model(b)
            ok = 0
        except ValueError:
            ok = -5
        pth = Path(td) / f"load_{i}.bin"
        pth.write_bytes(b)
        paths.append(pth)
        want.append(ok)
    got = []
    for i in range(0, len(paths), 200):
        got += [int(v) for v in P.run_driver(binary, ["--load", *paths[i:i + 200]]).split()]
    return [(cases[i][0], w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g], len(paths)


class Coverage:
    def __init__(self):
        self.truncated = self.halved = self.vertical = self.boxes = self.blocks = self.multi = self.empty = 0

    def add(self, model, img, r):
        self.truncated += r.truncated
        self.halved += T3.scale_of(model, img.shape[1], img.shape[0]) == 2
        self.vertical += sum(b.vertical for b in r.boxes)
        self.boxes += len(r.boxes)
        self.blocks += len(r.blocks)
        self.multi += sum(b.count > 1 for b in r.blocks)
        self.empty += not r.boxes

    def __str__(self):
        return (f"{self.truncated} truncated, {self.halved} halved, {self.empty} empty, {self.boxes} boxes "
                f"({self.vertical} vertical) in {self.blocks} blocks ({self.multi} of several boxes)")


def main(argv=None):
    import argparse
    import tempfile
    import time

    import textspot_parity as P
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="the driver (may carry a prefix, e.g. 'qemu-arm -cpu cortex-a9 ./x')")
    ap.add_argument("--v2-bin", default="", help="runtime/vocr_textspot2_test (version-1 and -2 batches compared)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--probe-frames", type=int, default=6)
    ap.add_argument("--full-every", type=int, default=4)
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--selftest-blobs", type=int, default=5)
    ap.add_argument("--maps-every", type=int, default=2)
    ap.add_argument("--arena-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cal", type=int, default=6)
    ap.add_argument("--probe-shapes", default="", help="M shapes (train/textspot3_model.PROBES names; default all)")
    ap.add_argument("--graphs", type=int, default=4)
    ap.add_argument("--v1-shapes", default="s4_c24n4,s4p_c8,s8_c8")
    ap.add_argument("--d2s-shapes", default="s8_c24n4x2")
    ap.add_argument("--blob", action="append", default=[])
    args = ap.parse_args(argv)
    t0 = time.time()
    args.probe_shapes = [s for s in args.probe_shapes.split(",") if s] or list(M3.PROBES)
    args.v1_shapes = [s for s in args.v1_shapes.split(",") if s]
    args.d2s_shapes = [s for s in args.d2s_shapes.split(",") if s]
    import textspot2_shapes as S2
    S2.register()
    blobs = all_blobs(args)
    rng = np.random.default_rng(args.seed)
    runs = nframes = nmaps = nv = 0
    bad = []
    cov = Coverage()
    with tempfile.TemporaryDirectory(prefix="textspot3-parity-") as td:
        if args.selftest:
            pick = []
            for want in (lambda n, b: b[4] == 3 and n.startswith("probe_ms"), lambda n, b: b[4] == 3 and
                         n.startswith("graph"), lambda n, b: b[4] == 3 and n.startswith(tuple(args.probe_shapes)),
                         lambda n, b: b[4] == 2, lambda n, b: b[4] == 1, lambda n, b: n.startswith("wild_")):
                pick += [x for x in blobs if want(*x) and x not in pick][:1]
            paths = []
            for name, blob in pick[:args.selftest_blobs]:
                pth = Path(td) / f"self_{name}.vtsp"
                pth.write_bytes(blob)
                paths.append(pth)
            print(P.run_driver(args.bin, ["--selftest", args.selftest, *paths]).strip())
        lbad, nload = check_loader(args.bin, blobs, td)
        for name, w, g in lbad[:12]:
            bad.append((name, -1, f"loader: C {g}", f"twin {w}"))
        for bi, (name, blob) in enumerate(blobs):
            model = T3.Model(blob)
            probe = name.startswith("probe")
            screen = P.SCREENS[(bi // args.full_every) % len(P.SCREENS)] if args.full_every and \
                bi % args.full_every == 0 else None
            if model.version == T3.VERSION_MS:
                imgs = frames_for(rng, model, args.probe_frames if probe else args.frames, probe, screen)
            else:
                imgs = P.frames_for(rng, model.t2.v1 if model.version == 1 else model.t2,
                                    args.probe_frames if probe else args.frames, probe, screen)
            arenas = bool(args.arena_every) and bi % args.arena_every == 0
            r, mp, v, b = check_blob(args.bin, args.v2_bin, name, blob, imgs, rng, td, cov, args.maps_every, arenas)
            runs += r
            nframes += len(imgs)
            nmaps += mp
            nv += v
            bad += b
    n2 = sum(1 for _, b in blobs if b[4] == 2)
    n3 = sum(1 for _, b in blobs if b[4] == 3)
    if bad:
        for name, k, c, p in bad[:12]:
            print(f"MISMATCH {name} frame {k}: {c}\n    {p}")
        print(f"TEXTSPOT3_PARITY_FAIL {len(bad)} of {runs} runs")
        return 1
    print(f"coverage: {cov}")
    print(f"TEXTSPOT3_PARITY_OK {runs} runs ({nframes} frames, {len(blobs)} blobs ({n2} of version 2, {n3} of version "
          f"3), {nmaps} maps, {nload} loader cases, {nv} version-1/2 runs equal to vocr_textspot2_test's; "
          f"{time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

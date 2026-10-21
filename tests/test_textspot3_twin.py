"""The version-3 spotter's twin (synth/textspot3_twin.py), no data and no C: versions 1 and 2 are textspot2_twin's
exactly; version 3's halving rule, map ops (upsampled concatenation, depthwise strides and dilations), the d2s head
read at stride 4 (the version-3 probe network gives #97's stride-4 probe's maps and results), and every refusal of
the loader (each also refused by textspot2_twin and textspot_twin)."""
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_parity as TP  # noqa: E402
import textdet_twin as TD  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_parity as Q2  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_twin as T2  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_parity as Q3  # noqa: E402
import textspot3_twin as T3  # noqa: E402
import textspot_parity as P  # noqa: E402
import textspot_twin as S  # noqa: E402

TINY = M3.m_shape("tiny", 4, 8, 8, [8], [1], [(16, [1]), (16, [2])], 8, 8, 1)


def recrc(b, at=12, start=16):
    b = bytearray(b)
    struct.pack_into("<I", b, at, zlib.crc32(bytes(b[start:])) & 0xFFFFFFFF)
    return bytes(b)


def tiny_net(seed=3):
    return M3.random_net(TINY, seed, n_cal=2)


# ---------------------------------------------------------------- versions 1 and 2 are textspot2_twin's

def test_versions_1_and_2_are_textspot2_twin():
    rng = np.random.default_rng(11)
    S2.register()
    blobs = [S.pack(TP.random_blob(s, 40 + i, 2), ha, 1 + 3 * (i % 2))
             for i, s in enumerate(("s4_c24n4", "s4p_c8", "s8_c24n4")) for ha in (0, 600)]
    blobs += [T2.pack(X.random_vtdt(s, 7, 2), ha, 2) for s in S2.D2S for ha in (0, 600)]
    blobs += [T2.pack(Q2.probe_vtdt(1), 600, 1)]
    for blob in blobs:
        a, b = T2.Model(blob), T3.Model(blob)
        assert b.t2 is not None and b.version == a.version and b.head == a.head and b.half_above_h == 0
        for k in ("half_above", "conf_min", "stride", "net_stride", "min_cells", "name", "blob_bytes"):
            assert getattr(a, k) == getattr(b, k), k
        for g in (P.tiled(rng, 960, 544), P.tiled(rng, 600, 300), P.tiled(rng, 37, 601), P.tiled(rng, 3, 2)):
            img = TP.to_rgba(g, rng) if rng.random() < 0.5 else g
            assert T3.spot(b, img) == T2.spot(a, img)
            x, f = T3.net_input(b, g)
            assert (x == T2.net_input(a, g)[0]).all() and f == T2.net_input(a, g)[1]
            assert (T3.run_maps(b, x) == T2.run_maps(a, x)).all()
            assert T3.scale_of(b, g.shape[1], g.shape[0]) == S.scale_of(a, g.shape[1])


# ---------------------------------------------------------------- version 3: input, ops, head

def test_the_halving_rule():
    m = T3.Model(T3.pack(tiny_net(), 480, 1, 272))
    for (w, h), f in (((480, 272), 1), ((481, 272), 2), ((480, 273), 2), ((960, 544), 2), ((1, 1), 1),
                      ((479, 600), 2), ((725, 544), 2), ((2048, 3), 2), ((3, 2048), 2), ((400, 200), 1)):
        assert T3.scale_of(m, w, h) == f, (w, h)
        x, ff = T3.net_input(m, np.zeros((h, w), np.uint8))
        assert ff == f and x.shape == ((h + f - 1) // f, (w + f - 1) // f)
    g = P.tiled(np.random.default_rng(2), 961, 545)
    assert (T3.net_input(m, g)[0] == S.halve(g)).all()           # textspot's exact 2x2 mean
    only_w = T3.Model(T3.pack(tiny_net(), 600, 1, 0))
    assert T3.scale_of(only_w, 600, 2048) == 1 and T3.scale_of(only_w, 601, 1) == 2
    only_h = T3.Model(T3.pack(tiny_net(), 0, 1, 100))
    assert T3.scale_of(only_h, 2048, 100) == 1 and T3.scale_of(only_h, 1, 101) == 2
    never = T3.Model(T3.pack(tiny_net(), 0, 1, 0))
    assert T3.scale_of(never, 2048, 2048) == 1
    # every frame up to 960 x 544 reaches the network at most 480 x 272
    for w, h in ((960, 544), (959, 543), (481, 1), (1, 273), (480, 272)):
        x, _ = T3.net_input(m, np.zeros((h, w), np.uint8))
        assert x.shape[0] <= 272 and x.shape[1] <= 480


def test_map_ops_against_a_direct_computation():
    """pw over three upsampled sources and dw ops of strides 1 and 2 with dilations, against loops written from the
    docstring's formulas (not textdet_twin)."""
    rng = np.random.default_rng(4)
    net = T3.Net(Q3.graph_net(17, n_ops=10))
    for h, w in ((37, 61), (8, 8), (1, 1), (65, 17)):
        x = rng.integers(0, 256, (h, w)).astype(np.uint8)
        m = T3.Model(T3.pack(Q3.graph_net(17, n_ops=10), 0, 1, 0))
        maps = T3.run_all(m, x)
        for k, op in enumerate(net.ops):
            rows, cols = T3.map_size(h, w, op.out_stride)
            ref = np.zeros((op.cout, rows, cols), np.int64)
            for i in range(rows):
                for j in range(cols):
                    if op.kind == T3.DW:
                        src = maps[op.src[0]]
                        acc = op.b.copy()
                        for ky in range(3):
                            for kx in range(3):
                                a, b = i * op.stride + (ky - 1) * op.dil, j * op.stride + (kx - 1) * op.dil
                                if 0 <= a < src.shape[1] and 0 <= b < src.shape[2]:
                                    acc += op.w[:, 0, ky, kx] * src[:, a, b]
                    else:
                        v = np.concatenate([maps[s][:, i >> u, j >> u] for s, u in zip(op.src, op.up)])
                        acc = op.b + op.w[:, :, 0, 0] @ v
                    q = (acc * op.m + (1 << (op.sh - 1))) >> op.sh
                    ref[:, i, j] = np.clip(q, 0 if op.relu else -128, 127)
            assert (maps[k + 1] == ref).all(), (h, w, k)
        assert maps[-1].shape == (16, -(-h // 8), -(-w // 8))


def test_the_dense_product_is_exact():
    """_dense's float64 product equals textdet_twin.run_layer's int64 one at the largest cin and extreme codes."""
    rng = np.random.default_rng(9)
    L = TD.Layer()
    L.kind, L.k, L.stride, L.dil, L.relu, L.cin, L.cout = 0, 1, 1, 1, 0, 1024, 16
    L.w = rng.choice([-127, 127, 0, 5], (16, 1024, 1, 1)).astype(np.int64)
    L.b = rng.integers(-(1 << 31), 1 << 31, 16)
    L.m = rng.integers(1, 1 << 31, 16)
    L.sh = rng.integers(1, 63, 16)
    x = rng.choice([-128, 127, 0, -3], (1024, 3, 5)).astype(np.int64)
    assert (T3._dense(L, x) == TD.run_layer(L, x)).all()


def test_the_probe_is_the_stride_4_probe():
    """probe_net carries the d2s probe's stride-8 codes through identity ops, a stride-2 chain to stride 64 and
    zero-weight branches upsampled by 2, 4 and 8: its maps and results are #97's stride-4 probe's on every frame,
    at both scales, through truncation."""
    rng = np.random.default_rng(5)
    for v in range(len(TP.PROBE_VARIANTS)):
        for ha, hh in ((0, 0), (480, 272)):
            mc = 1 + v % 4
            blob = T3.pack(Q3.probe_net(v, mc), ha, 1 + v % 3, hh)
            m = T3.Model(blob)
            ref = Q3.probe_reference(m)
            assert (m.version, m.head, m.stride, m.net_stride, m.min_cells) == (3, "ms_x2", 4, 8, mc)
            assert m.net.map_stride == [8, 8, 16, 32, 64, 64, 8, 8]
            for g in P.frames_for(rng, m, 3, True, (960, 544) if v == 1 else None):
                x, f = T3.net_input(m, g)
                assert (T3.run_maps(m, x) == TD.run_net(ref.net, x)).all()
                rr = T2.spot_maps(ref, TD.run_net(ref.net, x), f, np.ascontiguousarray(g))
                assert T3.spot(m, g) == rr


def test_d2s_channel_order_through_version_3():
    """The head's channel 4 (2 dy + dx) + m is map m of the stride-4 cell (2 i + dy, 2 j + dx): a head whose only
    non-zero channel is K of one sub-cell lights that sub-cell of every head cell, and nothing else."""
    for dy in (0, 1):
        for dx in (0, 1):
            layers = Q3.probe_layers(0)
            head = layers[-1]
            head["wq"] = np.zeros_like(head["wq"])
            head["bq"] = np.where(np.arange(16) % 4 == 0, -64, 0)
            head["bq"][T3.channel(dy, dx, 0)] = 40
            head["bq"][[T3.channel(dy, dx, 2), T3.channel(dy, dx, 3)]] = 4 * 30
            head["sh"] = np.where(np.arange(16) % 4 >= 2, 32, 30)
            net = M3.pack_net({"name": "order"}, layers)
            m = T3.Model(T3.pack(net, 0, 1, 0))
            x = np.zeros((40, 64), np.uint8)
            mp = T3.run_maps(m, x)
            k = mp[0] > 0
            assert k[dy::2, dx::2].all() and k.sum() == k[dy::2, dx::2].size


def test_random_version_3_blobs():
    blob, src = M3.random_blob(TINY, 1)
    m = T3.Model(blob)
    assert (m.version, m.head, m.name, m.half_above, m.half_above_h, m.conf_min) == (3, "ms_x2", "tiny", 480, 272, 1)
    assert blob == M3.random_blob(TINY, 1)[0] and "seed 1" in src
    assert len(blob) == M3.vtsp_bytes(TINY)
    r = T3.spot(m, P.tiled(np.random.default_rng(1), 300, 200))
    assert all(0 < b.conf <= 127 for b in r.boxes)
    for loader in (S.Model, T2.Model):
        with pytest.raises(ValueError):
            loader(blob)                                            # versions 1 and 2 refuse version 3


# ---------------------------------------------------------------- the loader

def raw(net, version=3, half_above=480, conf_min=1, r0=0, off=32, size=None, hh=272, r1=0):
    size = len(net) if size is None else size
    body = struct.pack("<HBBIIHH", half_above, conf_min, r0, off, size, hh, r1) + net
    return b"VTSP" + struct.pack("<III", version, 32 + len(net), zlib.crc32(body) & 0xFFFFFFFF) + body


def bad_blobs():
    """(name, blob) pairs every version-3 loader must refuse (synth/textspot3_twin.Model, vocr_textspot3.c)."""
    net = tiny_net()
    good = T3.pack(net, 480, 1, 272)
    out = [("short", good[:31]), ("truncated", recrc(good[:-4])), ("empty", b"")]
    n_s, n_o = struct.unpack_from("<H", net, 16)[0], struct.unpack_from("<H", net, 22)[0]
    r_s = lambda i: T3.NET_HEADER + i * T3.RECORD             # noqa: E731
    r_o = lambda k: T3.NET_HEADER + (n_s + k) * T3.RECORD     # noqa: E731

    def edit(name, fn, crc=True):
        b = bytearray(good)
        fn(b)
        out.append((name, recrc(b) if crc else bytes(b)))

    def inner(name, fn):
        b = bytearray(net)
        fn(b)
        out.append((f"net {name}", raw(recrc(b))))

    edit("magic", lambda b: b.__setitem__(2, ord("X")))
    edit("version 4", lambda b: struct.pack_into("<I", b, 4, 4))
    edit("version 2 around a version-3 network", lambda b: struct.pack_into("<I", b, 4, 2))
    edit("version 1 around a version-3 network", lambda b: struct.pack_into("<I", b, 4, 1))
    edit("size", lambda b: struct.pack_into("<I", b, 8, len(good) + 4))
    edit("crc", lambda b: b.__setitem__(len(b) - 1, b[-1] ^ 1), crc=False)
    edit("conf_min 0", lambda b: b.__setitem__(18, 0))
    edit("conf_min 128", lambda b: b.__setitem__(18, 128))
    edit("reserved u8", lambda b: b.__setitem__(19, 1))
    edit("offset 36", lambda b: struct.pack_into("<I", b, 20, 36))
    edit("net size short", lambda b: struct.pack_into("<I", b, 24, len(good) - 36))
    edit("reserved u16", lambda b: struct.pack_into("<H", b, 30, 1))
    edit("net crc", lambda b: b.__setitem__(len(b) - 1, b[-1] ^ 0x40))
    out.append(("version 3 around a VTDT network", raw(TP.random_blob("s8_c8", 3, 2))))
    out.append(("version 3 around a d2s network", raw(X.random_vtdt("s8_c24n4x2", 3, 2))))
    out.append(("no network", raw(b"")))
    out.append(("short network", raw(net[:63])))
    inner("magic", lambda b: b.__setitem__(0, ord("X")))
    inner("version 2", lambda b: struct.pack_into("<I", b, 4, 2))
    inner("stride 4", lambda b: b.__setitem__(18, 4))
    inner("outputs 4", lambda b: b.__setitem__(19, 4))
    inner("a codes 1", lambda b: b.__setitem__(20, 1))
    inner("min cells 0", lambda b: b.__setitem__(21, 0))
    inner("min cells 5", lambda b: b.__setitem__(21, 5))
    inner("0 stream layers", lambda b: struct.pack_into("<H", b, 16, 0))
    inner("33 stream layers", lambda b: struct.pack_into("<H", b, 16, 33))
    inner("0 ops", lambda b: struct.pack_into("<H", b, 22, 0))
    inner("65 ops", lambda b: struct.pack_into("<H", b, 22, 65))
    inner("records past the end", lambda b: struct.pack_into("<H", b, 22, (len(b) - 64) // 32))
    inner("non-ASCII name", lambda b: b.__setitem__(33, 0xC3))
    inner("stream kind 2", lambda b: b.__setitem__(r_s(0), 2))
    inner("stream dilation 9", lambda b: b.__setitem__(r_s(1) + 3, 9))
    inner("stream stride 2 dilated", lambda b: b.__setitem__(r_s(0) + 3, 2))
    inner("stream cin chain", lambda b: struct.pack_into("<H", b, r_s(1) + 8, 3))
    inner("stream cout 65", lambda b: struct.pack_into("<H", b, r_s(0) + 10, 65))
    s2 = next(i for i in range(n_s) if net[r_s(i) + 2] == 2)
    inner("stream at stride 4", lambda b: b.__setitem__(r_s(s2) + 2, 1))
    wo = struct.unpack_from("<I", net, r_s(0) + 12)[0]
    inner("stream weight -128", lambda b: b.__setitem__(wo, 0x80))
    inner("stream misaligned weights", lambda b: struct.pack_into("<I", b, r_s(0) + 12, wo + 1))
    so = struct.unpack_from("<I", net, r_s(0) + 24)[0]
    inner("stream shift 0", lambda b: b.__setitem__(so, 0))
    # the op records
    inner("op kind 2", lambda b: b.__setitem__(r_o(0), 2))
    inner("op stride 3", lambda b: b.__setitem__(r_o(0) + 1, 3))
    inner("op dilation 0", lambda b: b.__setitem__(r_o(1) + 2, 0))
    inner("op dilation 9", lambda b: b.__setitem__(r_o(2) + 2, 9))
    inner("op relu 2", lambda b: b.__setitem__(r_o(0) + 3, 2))
    inner("op reserved u8", lambda b: b.__setitem__(r_o(0) + 5, 1))
    inner("op reserved u16", lambda b: struct.pack_into("<H", b, r_o(0) + 14, 1))
    kinds = [net[r_o(k)] for k in range(n_o)]
    dw0 = kinds.index(T3.DW)
    pw0 = kinds.index(T3.PW)
    inner("dw of 2 sources", lambda b: b.__setitem__(r_o(dw0) + 4, 2))
    inner("dw with u 1", lambda b: b.__setitem__(r_o(dw0) + 11, 1))
    inner("dw stride 2 dilated", lambda b: b.__setitem__(r_o(dw0) + 2, 2))
    inner("dw cout != its source's", lambda b: struct.pack_into("<H", b, r_o(dw0) + 6, 16))
    inner("pw stride 2", lambda b: b.__setitem__(r_o(pw0) + 1, 2))
    inner("pw dilation 2", lambda b: b.__setitem__(r_o(pw0) + 2, 2))
    inner("pw 0 sources", lambda b: b.__setitem__(r_o(pw0) + 4, 0))
    inner("pw 4 sources", lambda b: b.__setitem__(r_o(pw0) + 4, 4))
    inner("an unused source entry", lambda b: b.__setitem__(r_o(pw0) + 9, 1))
    inner("an unused u entry", lambda b: b.__setitem__(r_o(pw0) + 12, 1))
    inner("a later source", lambda b: b.__setitem__(r_o(pw0) + 8, pw0 + 2))
    inner("its own map as source", lambda b: b.__setitem__(r_o(pw0) + 8, pw0 + 1))
    inner("u 4", lambda b: b.__setitem__(r_o(pw0) + 11, 4))
    inner("sources that do not meet", lambda b: b.__setitem__(r_o(pw0) + 11, 1))
    inner("cout 12", lambda b: struct.pack_into("<H", b, r_o(pw0) + 6, 12))
    inner("cout 0", lambda b: struct.pack_into("<H", b, r_o(pw0) + 6, 0))
    inner("cout 1032", lambda b: struct.pack_into("<H", b, r_o(pw0) + 6, 1032))
    ow = struct.unpack_from("<I", net, r_o(pw0) + 16)[0]
    inner("op weight -128", lambda b: b.__setitem__(ow + 1, 0x80))
    inner("op misaligned weights", lambda b: struct.pack_into("<I", b, r_o(pw0) + 16, ow + 2))
    inner("op weights before the data", lambda b: struct.pack_into("<I", b, r_o(pw0) + 16, 64))
    inner("op weights past the end", lambda b: struct.pack_into("<I", b, r_o(pw0) + 16, len(b) - 4))
    om = struct.unpack_from("<I", net, r_o(pw0) + 24)[0]
    inner("op multiplier 0", lambda b: struct.pack_into("<i", b, om, 0))
    inner("op multiplier < 0", lambda b: struct.pack_into("<i", b, om + 4, -5))
    osh = struct.unpack_from("<I", net, r_o(pw0) + 28)[0]
    inner("op shift 63", lambda b: b.__setitem__(osh, 63))
    inner("head with relu", lambda b: b.__setitem__(r_o(n_o - 1) + 3, 1))
    inner("head of 8 channels", lambda b: struct.pack_into("<H", b, r_o(n_o - 1) + 6, 8))
    inner("head a dw", lambda b: b.__setitem__(r_o(n_o - 1), T3.DW))
    inner("no head (an op short)", lambda b: struct.pack_into("<H", b, 22, n_o - 1))
    # networks the grammar never builds: a pw cin over 1,024, a stride over 256, a head at stride 16
    one = [1 << 30]

    def rq(c):
        return [0] * c, one * c, [30] * c

    def pw(src, up, cin, cout, s=8, head=False):
        return Q3.op_layer("pw", src, up, cin, cout, np.ones((cout, cin, 1, 1)), *rq(cout), out_stride=s, head=head)

    def dw(src, c, s_out, stride=2):
        return Q3.op_layer("dw", [src], [0], c, c, Q3.identity_dw(c), *rq(c), stride=stride, out_stride=s_out)

    wide = [pw([0], [0], 8, 512), pw([0, 1, 1], [0, 0, 0], 1032, 8), pw([2], [0], 8, 16, head=True)]
    out.append(("pw cin 1032", raw(_pack_unchecked(_stream8() + wide))))
    chain = [dw(k, 8, 16 << k) for k in range(6)]               # strides 16 .. 512
    out.append(("stride 512", raw(_pack_unchecked(_stream8() + chain + [pw([0], [0], 8, 16, head=True)]))))
    out.append(("head at stride 16", raw(_pack_unchecked(_stream8() + [dw(0, 8, 16), pw([1], [0], 8, 16, 16,
                                                                                         head=True)]))))
    out.append(("head of a later map", raw(_pack_unchecked(_stream8() + [pw([1], [0], 8, 16, head=True)]))))
    return out


def _stream8(c=8):
    """A stride-8 stream of three layers ending in c channels (identity-like requantisation)."""
    one = 1 << 30
    return [Q3.stream_layer("conv", 3, 2, 1, 1, np.ones((1, 1, 3, 3)), [0], [one], [30]),
            Q3.stream_layer("dw", 3, 2, 1, 1, np.ones((1, 1, 3, 3)), [0], [one], [30]),
            Q3.stream_layer("conv", 3, 2, 1, c, np.ones((c, 1, 3, 3)), [0] * c, [one] * c, [30] * c)]


def _pack_unchecked(qlayers):
    """pack_net without its read-back check (for blobs the loader must refuse)."""
    saved = T3.Net
    try:
        T3.Net = lambda b: None
        return M3.pack_net({"name": "bad"}, qlayers)
    finally:
        T3.Net = saved


def edge_nets():
    """Networks at the format's limits, which every loader must accept: a pw cin of exactly 1,024 over three
    sources, a stride-2 chain to stride 256, a head reading stride 64 through u = 3."""
    one = 1 << 30

    def rq(c):
        return [0] * c, [one] * c, [30] * c

    def pw(src, up, cin, cout, s=8, head=False):
        return Q3.op_layer("pw", src, up, cin, cout, np.ones((cout, cin, 1, 1)), *rq(cout), out_stride=s, head=head)

    def dw(src, c, s_out):
        return Q3.op_layer("dw", [src], [0], c, c, Q3.identity_dw(c), *rq(c), stride=2, out_stride=s_out)

    wide = _stream8() + [pw([0], [0], 8, 512), pw([0], [0], 8, 504), pw([0, 1, 2], [0, 0, 0], 1024, 8),
                         pw([3], [0], 8, 16, head=True)]
    deep = _stream8() + [dw(k, 8, 16 << k) for k in range(5)] + [pw([3, 0], [3, 0], 16, 16, head=True)]
    return [M3.pack_net({"name": "wide"}, wide), M3.pack_net({"name": "deep"}, deep)]


def good_blobs():
    net = tiny_net()
    return [(f"edge {i}", T3.pack(n, 480, 1, 272)) for i, n in enumerate(edge_nets())] + [
            ("h480 272", T3.pack(net, 480, 1, 272)), ("h0 0", T3.pack(net, 0, 1, 0)),
            ("h65535 c127", T3.pack(net, 65535, 127, 65535)), ("mc1", T3.pack(Q3.set_min_cells(net, 1), 480, 1, 272)),
            ("mc4", T3.pack(Q3.set_min_cells(net, 4), 1, 2, 1)), ("probe", T3.pack(Q3.probe_net(0), 480, 1, 272)),
            ("graph", T3.pack(Q3.graph_net(3), 480, 1, 272)), ("wild", T3.pack(Q3.wild_net(TINY, 2), 480, 1, 272))]


BAD = bad_blobs()


@pytest.mark.parametrize("name,blob", BAD, ids=[n for n, _ in BAD])
def test_the_twin_refuses(name, blob):
    with pytest.raises(ValueError):
        T3.Model(blob)
    with pytest.raises(ValueError):                         # and the earlier loaders refuse it too
        T2.Model(blob)
    with pytest.raises(ValueError):
        S.Model(blob)


def test_the_twin_accepts_the_edge_values():
    for _, blob in good_blobs():
        T3.Model(blob)
    with pytest.raises(ValueError):
        T3.pack(tiny_net(), 480, 0, 272)
    with pytest.raises(ValueError):
        T3.pack(tiny_net(), 480, 1, 65536)
    assert T3.pack(TP.random_blob("s4_c4", 3, 2), 600, 1) == S.pack(TP.random_blob("s4_c4", 3, 2), 600, 1)

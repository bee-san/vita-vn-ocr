"""The d2s spotter's twin (synth/textspot2_twin.py) and exporter (train/textspot2_export.py), no data and no C:
version 1 is textspot_twin exactly; the version-2 decode on hand-made maps (a line across sub-cells, two lines two
stride-4 cells apart that a stride-8 map would merge, vertical text), the head's channel order and crop, the d2s
probe network against #97's stride-4 probe, every refusal of the loader, and the exporter (random blobs, the
checkpoint export, the torch rearrangement and fake-quantised network)."""
import hashlib
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_parity as TP  # noqa: E402
import textdet_twin as TD  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_parity as Q  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_twin as T2  # noqa: E402
import textspot_parity as P  # noqa: E402
import textspot_twin as S  # noqa: E402

D4 = SimpleNamespace(stride=4, min_cells=2, conf_min=1)     # the decode's view of a version-2 model


def recrc(b, at=12, start=16):
    b = bytearray(b)
    struct.pack_into("<I", b, at, zlib.crc32(bytes(b[start:])) & 0xFFFFFFFF)
    return bytes(b)


def d2s_vtdt(name="s8_c24n4x2", seed=3):
    return X.random_vtdt(name, seed, n_cal=2)


# ---------------------------------------------------------------- version 1 is textspot_twin

def test_version_1_is_textspot_twin():
    rng = np.random.default_rng(11)
    S2.register()
    for i, shape in enumerate(("s4_c24n4", "s4p_c8", "s4p_c6n4", "s4_c32n3", "s8_c24n4", "s4_c4")):
        for ha in (0, 600):
            blob = S.pack(TP.random_blob(shape, 40 + i, 2), ha, 1 + 3 * (i % 2))
            a, b = S.Model(blob), T2.Model(blob)
            assert (b.version, b.head, b.v1.blob_bytes) == (1, "kroa", len(blob))
            for k in ("half_above", "conf_min", "stride", "min_cells", "name", "blob_bytes"):
                assert getattr(a, k) == getattr(b, k), k
            assert b.net_stride == a.net.stride == b.stride
            for g in (P.tiled(rng, 960, 544), P.tiled(rng, 480, 272), P.tiled(rng, 37, 601), P.tiled(rng, 3, 2)):
                img = TP.to_rgba(g, rng) if rng.random() < 0.5 else g
                r = T2.spot(b, img)
                assert r == S.spot(a, img)
                x, f = T2.net_input(b, g)
                assert T2.spot_maps(b, T2.run_maps(b, x), f, g) == S.spot(a, g)    # the same tail as textspot_twin


# ---------------------------------------------------------------- the head's channels and the crop

def test_channel_order_and_crop():
    assert [T2.channel(dy, dx, m) for dy in (0, 1) for dx in (0, 1) for m in range(4)] == list(range(16))
    for h in range(1, 19):
        for w in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17):
            hn, wn = -(-h // 8), -(-w // 8)
            head = (np.arange(16)[:, None, None] * 10000 + np.arange(hn)[None, :, None] * 100 +
                    np.arange(wn)[None, None, :])
            maps = T2.head_to_maps(head, h, w)
            assert maps.shape == (4, -(-h // 4), -(-w // 4))
            for m in range(4):
                for i in range(maps.shape[1]):
                    for j in range(maps.shape[2]):
                        assert maps[m, i, j] == head[4 * (2 * (i % 2) + j % 2) + m, i // 2, j // 2]
            back = T2.maps_to_head(maps, h, w)
            assert (T2.head_to_maps(back, h, w) == maps).all()
    with pytest.raises(ValueError):
        T2.head_to_maps(np.zeros((16, 2, 2)), 17, 9)            # a 17x9 input has a 3x2 head
    with pytest.raises(ValueError):
        T2.head_to_maps(np.zeros((4, 2, 2)), 9, 9)


# ---------------------------------------------------------------- the version-2 decode on hand-made maps

def maps_of(hm, wm):
    """Empty stride-4 maps: K = -16 (no kernel), O = -16 (horizontal), A1 = A2 = 0."""
    m = np.zeros((4, hm, wm), np.int64)
    m[0] = -16
    m[1] = -16
    return m


def put(m, rows, cols, k=10, o=-16, a1=12, a2=12):
    m[0][rows, cols] = k
    m[1][rows, cols] = o
    m[2][rows, cols] = a1
    m[3][rows, cols] = a2


def decode_v2(maps, h, w, f=1):
    """The decode of hand-made stride-4 maps through a d2s head: maps -> the 16-channel head -> head_to_maps ->
    textspot2_twin.spot_maps (frame f h x f w, grey 0: no dark text)."""
    head = T2.maps_to_head(maps, h, w)
    back = T2.head_to_maps(head, h, w)
    assert (back == maps).all()
    return T2.spot_maps(D4, back, f, np.zeros((f * h, f * w), np.uint8))


def boxes(r):
    return [(b.x, b.y, b.w, b.h, b.vertical) for b in r.boxes]


def test_a_line_across_sub_cells():
    """Cells (3, 5..14): odd row (dy = 1) and both column parities (dx = 0, 1) of head cells (1, 2..7)."""
    m = maps_of(10, 30)
    put(m, 3, slice(5, 15))     # c2 = 2*4*3 + 3 = 27: near floor((270-120)/20) = 7, far ceil(390/20) = 20
    r = decode_v2(m, 40, 120)
    assert boxes(r) == [(18, 5, 44, 17, 0)]                  # pad 2; along [20, 60): textdet's stride-4 line
    assert r.boxes[0].conf == 10 and r.pick == 0 and not r.truncated
    head = T2.maps_to_head(m, 40, 120)
    assert (head[T2.channel(1, 0, 0), 1, 3:7] == 10).all() and (head[T2.channel(1, 1, 0), 1, 2:7] == 10).all()
    assert (head[T2.channel(0, 0, 0)] == -16).all()          # no kernel in the even rows
    m2 = maps_of(10, 30)
    put(m2, 3, slice(5, 15))
    assert S.candidates(D4, m2, 1, 40, 120) == S.candidates(D4, T2.head_to_maps(head, 40, 120), 1, 40, 120)
    r2 = decode_v2(m, 40, 120, f=2)                          # a halved frame: everything at twice the scale
    assert boxes(r2) == [(36, 11, 88, 32, 0)]                # near 15, far 39, pad 4; along [40, 120)


def test_two_lines_two_cells_apart_stay_apart():
    """Lines in stride-4 rows 3 and 5 lie in head rows 1 and 2, which touch: a stride-8 map of the same text (a
    head cell is kernel when any of its sub-cells is) merges them into one component; the d2s maps keep two."""
    m = maps_of(10, 30)
    put(m, 3, slice(5, 15), a1=4, a2=4)
    put(m, 5, slice(5, 15), a1=4, a2=4)
    r = decode_v2(m, 40, 120)
    assert [b[1] for b in boxes(r)] == [9, 17] and len(r.boxes) == 2
    assert len(r.blocks) == 1 and r.blocks[0].count == 2      # one block of two lines, top to bottom
    head = T2.maps_to_head(m, 40, 120)
    k8 = head[[T2.channel(dy, dx, 0) for dy in (0, 1) for dx in (0, 1)]].max(0) > 0
    assert len(TD.components(k8)) == 1                       # at stride 8 the two lines are one component
    m[0][4, 9] = 10                                          # one kernel cell between them joins them
    assert len(decode_v2(m, 40, 120).boxes) == 1


def test_vertical_text():
    m = maps_of(10, 30)
    put(m, slice(1, 9), 20, o=16, a1=8, a2=8)    # column 20 (dx = 0), rows 1..8: c2 = 163, near 77, far 86
    r = decode_v2(m, 40, 120)
    assert boxes(r) == [(75, 2, 13, 36, 1)] and r.blocks[0].vertical == 1
    put(m, slice(1, 9), 23, o=16, a1=8, a2=8)    # a second column three cells to the right (dx = 1)
    r = decode_v2(m, 40, 120)
    assert [b[0] for b in boxes(r)] == [87, 75] and all(b.vertical for b in r.boxes)   # right to left
    assert len(r.blocks) == 1


def test_the_crop_drops_sub_cells_past_the_input():
    """A 9-row input: ceil(9/4) = 3 map rows, though the head's 2 rows hold 4 sub-rows; a kernel in the 4th never
    reaches the decode."""
    head = T2.maps_to_head(maps_of(3, 3), 9, 9)
    head[T2.channel(1, 0, 0), 1, :] = 40                    # sub-row 3: past the input
    head[T2.channel(1, 0, 2), 1, :] = head[T2.channel(1, 0, 3), 1, :] = 30
    maps = T2.head_to_maps(head, 9, 9)
    assert maps.shape == (4, 3, 3) and (maps[0] < 0).all()
    assert not T2.spot_maps(D4, maps, 1, np.zeros((9, 9), np.uint8)).boxes


# ---------------------------------------------------------------- through the network

def test_the_d2s_probe_is_the_stride_4_probe():
    """probe_vtdt carries pixel (8 i + 4 dy, 8 j + 4 dx) to sub-cell (dy, dx): its maps and results are #97's
    stride-4 probe's on every frame, at both scales, through truncation."""
    rng = np.random.default_rng(5)
    for v in range(len(TP.PROBE_VARIANTS)):
        for ha in (0, 600):
            mc = 1 + v % 4
            blob = T2.pack(Q.set_min_cells(Q.probe_vtdt(v), mc), ha, 1 + v % 3)
            m = T2.Model(blob)
            ref = Q.probe_twin(blob)
            assert (m.version, m.stride, m.net_stride, m.min_cells) == (2, 4, 8, mc) and ref.stride == 4
            for g in P.frames_for(rng, m, 3, True, (960, 544) if v == 1 else None):
                x, f = T2.net_input(m, g)
                assert (T2.run_maps(m, x) == TD.run_net(ref.net, x)).all()
                assert T2.spot(m, g) == S.spot(ref, g)


def test_random_d2s_blobs():
    S2.register()
    for name, want in (("s8_c24n4x2", 4350), ("s8_c48n4x2", 13420)):
        blob, src = X.random_blob(name, 1)
        m = T2.Model(blob)
        assert (m.version, m.head, m.name, m.stride, m.net_stride, m.min_cells) == (2, "kroa_x2", name, 4, 8, 2)
        assert m.net.outputs == 16 and m.net.layers[-1].cout == 16 and m.net.layers[-1].k == 1
        assert "seed 1" in src and blob == X.random_blob(name, 1)[0]                     # deterministic
        c = X.costs(name)
        assert abs(c["weights_int8"] - want) < 10                                         # the protocol's ~figure
        assert c["macs"]["960x544"] == c["macs"]["480x272"]                               # h2: a Vita frame halved
        g = P.tiled(np.random.default_rng(1), 300, 200)
        r = T2.spot(m, g)
        assert r.boxes and all(0 < b.conf <= 127 for b in r.boxes)
        with pytest.raises(ValueError):
            S.Model(blob)                                                                 # version 1 refuses it


# ---------------------------------------------------------------- the loader

def bad_blobs():
    """(name, blob) pairs every version-2-aware loader must refuse (synth/textspot2_twin.Model, vocr_textspot2.c)."""
    vtdt = d2s_vtdt()
    good = T2.pack(vtdt, 600, 1)
    out = [("short", good[:31]), ("truncated", recrc(good[:-4])), ("empty", b"")]

    def edit(name, fn, crc=True, blob=good):
        b = bytearray(blob)
        fn(b)
        out.append((name, recrc(b) if crc else bytes(b)))

    def raw(inner, version=2, half_above=600, conf_min=1, r0=0, off=32, size=None, r1=0):
        size = len(inner) if size is None else size
        body = struct.pack("<HBBIII", half_above, conf_min, r0, off, size, r1) + inner
        return b"VTSP" + struct.pack("<III", version, 32 + len(inner), zlib.crc32(body) & 0xFFFFFFFF) + body

    def inner(name, fn):
        b = bytearray(vtdt)
        fn(b)
        out.append((f"inner {name}", raw(recrc(b))))

    edit("magic", lambda b: b.__setitem__(1, ord("X")))
    edit("version 0", lambda b: struct.pack_into("<I", b, 4, 0))
    edit("version 3", lambda b: struct.pack_into("<I", b, 4, 3))
    edit("version 1 around a d2s network", lambda b: struct.pack_into("<I", b, 4, 1))
    edit("size", lambda b: struct.pack_into("<I", b, 8, len(good) + 4))
    edit("crc", lambda b: b.__setitem__(len(b) - 1, b[-1] ^ 1), crc=False)
    edit("conf_min 0", lambda b: b.__setitem__(18, 0))
    edit("conf_min 128", lambda b: b.__setitem__(18, 128))
    edit("reserved u8", lambda b: b.__setitem__(19, 2))
    edit("offset 36", lambda b: struct.pack_into("<I", b, 20, 36))
    edit("net size short", lambda b: struct.pack_into("<I", b, 24, len(good) - 36))
    edit("reserved u32", lambda b: struct.pack_into("<I", b, 28, 1))
    edit("inner crc", lambda b: b.__setitem__(len(b) - 1, b[-1] ^ 0x40))
    out.append(("version 2 around a 4-map network", raw(TP.random_blob("s4_c4", 3, 2))))
    out.append(("version 2 around a 4-map stride-8 network", raw(TP.random_blob("s8_c8", 3, 2))))
    out.append(("no network", raw(b"")))
    out.append(("short network", raw(vtdt[:63])))
    inner("magic", lambda b: b.__setitem__(0, ord("X")))
    inner("version 2", lambda b: struct.pack_into("<I", b, 4, 2))
    inner("stride 4", lambda b: b.__setitem__(18, 4))
    inner("stride 5", lambda b: b.__setitem__(18, 5))
    inner("outputs 4", lambda b: b.__setitem__(19, 4))
    inner("outputs 8", lambda b: b.__setitem__(19, 8))
    inner("a codes 3", lambda b: b.__setitem__(20, 3))
    inner("min cells 0", lambda b: b.__setitem__(21, 0))
    inner("min cells 5", lambda b: b.__setitem__(21, 5))
    inner("1 layer", lambda b: struct.pack_into("<H", b, 16, 1))
    inner("33 layers", lambda b: struct.pack_into("<H", b, 16, 33))
    inner("non-ASCII name", lambda b: b.__setitem__(33, 0xC3))
    n = struct.unpack_from("<H", vtdt, 16)[0]
    rl, r0 = TD.HEADER + (n - 1) * TD.RECORD, TD.HEADER
    inner("head with relu", lambda b: b.__setitem__(rl + 4, 1))
    inner("head of 15 channels", lambda b: struct.pack_into("<H", b, rl + 10, 15))
    inner("kind 2", lambda b: b.__setitem__(r0, 2))
    inner("stride 2 dilated", lambda b: b.__setitem__(r0 + 3, 2))
    inner("dilation 9", lambda b: b.__setitem__(r0 + 3, 9))
    so = struct.unpack_from("<I", vtdt, r0 + 24)[0]
    mo = struct.unpack_from("<I", vtdt, r0 + 20)[0]
    inner("shift 63", lambda b: b.__setitem__(so, 63))
    inner("multiplier 0", lambda b: struct.pack_into("<i", b, mo, 0))
    wo = struct.unpack_from("<I", vtdt, r0 + 12)[0]
    inner("misaligned weights", lambda b: struct.pack_into("<I", b, r0 + 12, wo + 1))
    out.append(("inner no head", raw(recrc(bytes(vtdt[:16]) + struct.pack("<H", n - 1) + vtdt[18:]))))
    # the stride-8 chain with one stride-2 layer made stride 1 (stride 4 overall, header still 8)
    s2 = next(i for i in range(n) if vtdt[TD.HEADER + i * TD.RECORD + 2] == 2)
    inner("chain stride 4", lambda b: b.__setitem__(TD.HEADER + s2 * TD.RECORD + 2, 1))
    return out


def good_blobs():
    v = d2s_vtdt()
    return [("h0", T2.pack(v, 0, 1)), ("h600", T2.pack(v, 600, 1)), ("h65535 c127", T2.pack(v, 65535, 127)),
            ("mc1", T2.pack(Q.set_min_cells(v, 1), 600, 1)), ("mc4", T2.pack(Q.set_min_cells(v, 4), 1, 2)),
            ("probe", T2.pack(Q.probe_vtdt(0), 600, 1)), ("v1", S.pack(TP.random_blob("s4_c4", 3, 2), 600, 1))]


@pytest.mark.parametrize("name,blob", bad_blobs(), ids=[n for n, _ in bad_blobs()])
def test_the_twin_refuses(name, blob):
    with pytest.raises(ValueError):
        T2.Model(blob)
    with pytest.raises(ValueError):                         # and textspot's version-1 loader refuses it too
        S.Model(blob)


def test_the_twin_accepts_the_edge_values():
    for _, blob in good_blobs():
        T2.Model(blob)
    with pytest.raises(ValueError):
        T2.pack(TP.random_blob("s4_c4", 3, 2), 600, 1, version=2)   # a 4-map network is never version 2
    with pytest.raises(ValueError):
        T2.pack(d2s_vtdt(), 600, 1, version=1)                     # nor a d2s one version 1
    with pytest.raises(ValueError):
        T2.pack(d2s_vtdt(), 600, 0)


# ---------------------------------------------------------------- the exporter

def test_layers_and_costs():
    S2.register()
    for name in S2.D2S:
        s = X.shape(name)
        L, base = X.layers(s), X.TM.layers(s)
        assert len(L) == len(base) and L[:-1] == base[:-1]
        assert L[-1] == dict(base[-1], cout=16) and L[-1]["k"] == 1 and L[-1]["head"]
        assert np.prod([x["stride"] for x in L]) == 8
        c = X.cost(s, 480, 272)
        assert c["out"] == [60, 34]
        # the head's extra 12 channels over a 4-map head on the same trunk
        cin = L[-1]["cin"]
        assert c["macs"] - X.TM.cost(s, 480, 272)["macs"] == 12 * cin * 60 * 34
    with pytest.raises(ValueError):
        X.shape("s4_c24n4")
    with pytest.raises(ValueError):
        X.layers(dict(X.shape("s8_c24n4x2"), stride=4))


def fake_checkpoint(name, seed):
    """A d2s checkpoint as the trainer writes it: a random-init TextDetD2S with random BatchNorm statistics, folded,
    with activation scales calibrated on procedural regions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    shape = X.shape(name)
    model = X.TextDetD2S(shape)
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.2, 0.2)
    model.eval()
    folded = X.TM.fold(model)
    q = X.QTextDetD2S(shape, folded, [1.0] * len(folded))
    q.quant = False
    mx = [0.0] * len(folded)
    with torch.no_grad():
        for _ in range(4):
            g = X.E.procedural_region(rng).astype(np.float32)[None, None]
            taps = []
            q((torch.from_numpy(g) - 128) / 128, taps=taps)
            mx = [max(a, float(t.abs().max())) for a, t in zip(mx, taps)]
    scales = [max(v, 1e-3) / 127 for v in mx]
    return {"kind": "textdet-qat", "shape": shape, "seed": seed, "steps": 0, "w": [w for w, _ in folded],
            "b": [b for _, b in folded], "act_scale": scales}


def test_checkpoint_export(tmp_path):
    ck = fake_checkpoint("s8_c24n4x2", 7)
    torch.save(ck, tmp_path / "final.pt")
    vtd, vtsp = X.export_vtsp(tmp_path / "final.pt", 600, 1)
    m = T2.Model(vtsp)
    assert (m.version, m.half_above, m.conf_min, m.name, m.net.blob_bytes) == (2, 600, 1, "s8_c24n4x2", len(vtd))
    assert m.net.ckpt == hashlib.sha256((tmp_path / "final.pt").read_bytes()).digest()[:8].hex()
    q = X.quantize_layers(ck["shape"], [(w.double(), b.double()) for w, b in zip(ck["w"], ck["b"])], ck["act_scale"])
    for L, ql in zip(m.net.layers, q):
        assert (L.w == ql["wq"]).all() and (L.b == ql["bq"]).all() and (L.m == ql["m"]).all()
        assert (L.sh == ql["sh"]).all()
    # the head's requantisation: per channel, K and O at 1/16 per code, A1 and A2 at 1/2 px
    w = ck["w"][-1].double().numpy()
    s_w = np.abs(w.reshape(16, -1)).max(1) / 127
    s_x = ck["act_scale"][-2]
    for c in range(16):
        want = s_w[c] * s_x / X.HEAD_SCALES[c]
        assert abs(q[-1]["m"][c] / 2.0 ** q[-1]["sh"][c] / want - 1) < 1e-8
    assert X.HEAD_SCALES[:4] == (1 / 16, 1 / 16, 1 / 2, 1 / 2) and len(set(X.HEAD_SCALES[4:])) == 2
    bad = dict(ck, shape=dict(ck["shape"], head="kroa"))
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(SystemExit):
        X.export_checkpoint(tmp_path / "bad.pt")
    torch.save(dict(ck, w=ck["w"][:-1], b=ck["b"][:-1]), tmp_path / "short.pt")
    with pytest.raises(SystemExit):
        X.export_checkpoint(tmp_path / "short.pt")


def test_the_fake_quantised_network_is_the_blob():
    """QTextDetD2S (the export's arithmetic in float) and the twin's integer network give the same head codes up to
    the usual rounding: at least 98% within one code; to_stride4 is head_to_maps."""
    ck = fake_checkpoint("s8_c24n4x2", 9)
    shape = ck["shape"]
    folded = list(zip(ck["w"], ck["b"]))
    blob = X.pack(shape, X.quantize_layers(shape, [(w.double(), b.double()) for w, b in folded], ck["act_scale"]))
    net = T2.Net(blob)
    q = X.QTextDetD2S(shape, folded, ck["act_scale"]).eval()
    g = X.E.procedural_region(np.random.default_rng(3))
    with torch.no_grad():
        y = q((torch.from_numpy(g.astype(np.float32))[None, None] - 128) / 128)
    codes = torch.round(y / getattr(q, f"s{len(q.spec) - 1}"))[0].numpy().astype(np.int64)
    ref = TD.run_net(net, g)
    assert codes.shape == ref.shape and (np.abs(codes - ref) <= 1).mean() >= 0.98
    yy = torch.from_numpy(ref[None].astype(np.float64))
    assert (X.to_stride4(yy, *g.shape)[0].numpy() == T2.head_to_maps(ref, *g.shape)).all()
    with pytest.raises(ValueError):
        X.to_stride4(yy[:, :4], *g.shape)

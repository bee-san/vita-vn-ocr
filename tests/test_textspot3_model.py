"""textspot3's family M (train/textspot3_model.py), no data and no C: the grammar's layers and map ops, exact costs
and VTSP bytes, the receptive field, the shared-scale groups, window masks (a region in a padded window computes
what the region alone computes, through the upsamplings and crops), the BatchNorm fold, and the export: the
fake-quantised torch network and the exported blob's integer network give the same head codes up to the usual
rounding."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_model as TM  # noqa: E402
import textspot2_twin as T2  # noqa: E402
import textspot3_model as M3  # noqa: E402
import textspot3_shapes as S3  # noqa: E402
import textspot3_twin as T3  # noqa: E402

TINY = M3.m_shape("tiny", 4, 8, 8, [8], [1], [(16, [1]), (16, [2])], 8, 8, 1)


@pytest.fixture(autouse=True)
def few_threads():
    """Small tensors: a few intra-op threads (a box with many cores otherwise spends seconds per forward)."""
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


SMALL = M3.m_shape("small", 4, 8, 16, [16, 16], [1, 2], [(24, [1]), (32, [2]), (40, [1])], 16, 16, 1)


# ---------------------------------------------------------------- the grammar

def test_layers_of_a_three_stage_shape():
    s = SMALL
    stream, ops = M3.stream_layers(s), M3.map_ops(s)
    trunk = TM.layers({**{k: s[k] for k in ("stride", "c0", "c1", "c2", "chans", "dil", "block")}, "head": "kroa",
                       "name": "x"})
    assert [dict(L, part="stream") for L in trunk[:-1]] == stream
    assert np.prod([L["stride"] for L in stream]) == 8 and stream[-1]["cout"] == 16
    roles = [op["role"] for op in ops]
    assert roles == ["s16 down", "s16 down", "s16 block 0", "s16 block 0",
                     "s32 down", "s32 down", "s32 block 0", "s32 block 0",
                     "s64 down", "s64 down", "s64 block 0", "s64 block 0",
                     "neck P64", "neck P32", "neck P16", "neck P8", "smooth 0", "smooth 0", "head"]
    maps = M3.maps_of(s)
    assert maps[0] == (8, 16) and [m for m in maps[1:5]] == [(16, 16), (16, 24), (16, 24), (16, 24)]
    f16, f32, f64 = 4, 8, 12                                # the stages' last maps
    assert maps[f16] == (16, 24) and maps[f32] == (32, 32) and maps[f64] == (64, 40)
    p64, p32, p16, p8 = (ops[k] for k in (12, 13, 14, 15))
    assert p64["src"] == [f64] and p64["up"] == [0] and p64["cout"] == 16 and p64["out_stride"] == 64
    assert p32["src"] == [f32, 13] and p32["up"] == [0, 1] and p32["cin"] == 32 + 16
    assert p16["src"] == [f16, 14] and p16["up"] == [0, 1] and p16["out_stride"] == 16
    assert p8["src"] == [0, 15] and p8["up"] == [0, 1] and p8["cout"] == 16 and p8["out_stride"] == 8
    dws = [op for op in ops if op["kind"] == "dw"]
    assert [op["stride"] for op in dws] == [2, 1, 2, 1, 2, 1, 1] and [op["dil"] for op in dws] == [1, 1, 1, 2, 1, 1, 1]
    assert all(not op["relu"] for op in dws) and all(op["relu"] for op in ops if op["kind"] == "pw" and
                                                     not op["head"])
    head = ops[-1]
    assert head["head"] and head["cout"] == 16 and not head["relu"] and head["src"] == [18]
    # one stage: P16 alone, then P8
    one = M3.m_shape("one", 4, 8, 8, [], [], [(16, [])], 8, 8, 0)
    assert [op["role"] for op in M3.map_ops(one)] == ["s16 down", "s16 down", "neck P16", "neck P8", "head"]


def test_check_shape_refuses():
    for bad in (dict(SMALL, family="T"), dict(SMALL, head="kroa_x2"), dict(SMALL, stages=[]),
                dict(SMALL, stages=[{"c": 12, "n": 0, "dil": []}]), dict(SMALL, stages=[{"c": 16, "n": 2, "dil": [1]}]),
                dict(SMALL, neck={"t": 20, "t8": 8, "smooth": 0}), dict(SMALL, chans=[16]),
                {k: v for k, v in SMALL.items() if k != "neck"}):
        with pytest.raises((ValueError, KeyError)):
            M3.check_shape(bad)
            M3.layers(bad)
    with pytest.raises(ValueError):
        M3.layers(dict(SMALL, c2=12, chans=[], dil=[]))      # F8 not a multiple of 8
    with pytest.raises(ValueError):
        M3.layers(dict(SMALL, chans=[72], dil=[1]))          # a stream layer over 64 channels


def test_scale_groups():
    s = SMALL
    spec, prod, groups = M3.layers(s), M3.producers(s), M3.scale_groups(s)
    n_s = len(M3.stream_layers(s))
    ops = M3.map_ops(s)
    for op in ops:
        if op["kind"] == "pw" and len(op["src"]) > 1:
            assert len({groups[prod[k]] for k in op["src"]}) == 1
    # F8 (the stream's last layer) shares P16's producer's group; F64's producer is alone
    assert groups[n_s - 1] == groups[n_s + 14] and groups[n_s + 11] == n_s + 11
    sc = M3.shared_scales(s, [0.01 * (i + 1) for i in range(len(spec))])
    for op in ops:
        assert len({sc[prod[k]] for k in op["src"]}) == 1
    assert sc[n_s - 1] == max(0.01 * n_s, 0.01 * (n_s + 15))
    assert M3.shared_scales(s, sc) == sc                     # idempotent


# ---------------------------------------------------------------- costs

def test_costs_are_exact():
    for s in (TINY, SMALL, *M3.PROBES.values()):
        for w, h in ((480, 272), (481, 3), (7, 9)):
            c = M3.cost(s, w, h)
            macs = 0
            ww, hh = w, h
            for L in M3.layers(s):
                per = L["k"] * L["k"] * (1 if L["kind"] == "dw" else L["cin"])
                if L["part"] == "stream":
                    ww, hh = -(-ww // L["stride"]), -(-hh // L["stride"])
                    cells = ww * hh
                else:
                    cells = -(-w // L["out_stride"]) * -(-h // L["out_stride"])
                macs += cells * L["cout"] * per
            assert c["macs"] == macs and sum(c["macs_by_part"].values()) == macs
            assert sum(c["weights_by_part"].values()) == c["weights_int8"]
    for s in (TINY, SMALL):
        blob, _ = M3.random_blob(s, 2, n_cal=1)
        assert len(blob) == M3.vtsp_bytes(s) == M3.costs(s)["vtsp_bytes"]
        m = T3.Model(blob)
        n_w = sum(L.w.size for L in m.net.stream) + sum(op.w.size for op in m.net.ops)
        assert n_w == M3.costs(s)["weights_int8"]
    c = M3.costs(SMALL)
    assert c["macs"]["960x544"] == c["macs"]["480x272"] == M3.cost(SMALL, 480, 272)["macs"]
    assert M3.net_input_size(960, 544) == (480, 272) and M3.net_input_size(480, 273) == (240, 137)
    assert M3.net_input_size(480, 272) == (480, 272)


def test_receptive_field():
    one = M3.m_shape("one", 4, 8, 8, [], [], [(16, [])], 8, 8, 0)
    # stem 3, down 3x3 s2 (+2*2), down8 3x3 s2 (+2*4): 15; the stage's stride-2 dw at stride 8 adds 2 * 8
    assert M3.receptive_field(one) == 15 + 16
    two = M3.m_shape("two", 4, 8, 8, [8], [2], [(16, [3])], 8, 8, 1)
    # + the stream block at dilation 2 (2*2*8 = 32), the stage's down (16), its block at dilation 3 (2*3*16 = 96),
    # and the smoothing block at stride 8 after the neck (2*8 = 16)
    assert M3.receptive_field(two) == 15 + 32 + 16 + 96 + 16


def test_shapes_registry_reaches_m_shapes(monkeypatch):
    monkeypatch.setitem(S3.SHAPES, "tiny_m", dict(TINY, name="tiny_m"))
    c = S3.costs("tiny_m")
    assert c["family"] == "M" and c["vtsp_bytes"] == M3.vtsp_bytes(TINY)
    assert set(c) >= {"macs", "params", "weights_int8", "vtsp_bytes", "receptive_field"}


# ---------------------------------------------------------------- torch: masks, fold, export

def random_bn(model, seed):
    torch.manual_seed(seed)
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.2, 0.2)
    return model.eval()


def test_masks_a_region_in_a_window_is_the_region_alone():
    torch.manual_seed(0)
    model = random_bn(M3.TextSpot3(SMALL), 1)
    rng = np.random.default_rng(2)
    H, W = 96, 200
    for vh, vw in ((96, 200), (37, 61), (9, 17), (1, 1), (65, 130), (88, 3)):
        g = rng.integers(0, 256, (vh, vw)).astype(np.float32)
        win = np.full((H, W), 128, np.float32)
        win[:vh, :vw] = g
        win[vh:, :] = rng.integers(0, 256, (H - vh, W))       # junk outside the region: masks must hide it
        win[:, vw:] = rng.integers(0, 256, (H, W - vw))
        x = (torch.from_numpy(win)[None, None] - 128) / 128
        masks = M3.layer_masks(SMALL, [(vh, vw)], H, W)
        assert len(masks) == len(M3.layers(SMALL))
        with torch.no_grad():
            y = model(x, masks=masks)
            alone = model((torch.from_numpy(g)[None, None] - 128) / 128)
        hn, wn = -(-vh // 8), -(-vw // 8)
        assert y.shape == (1, 16, H // 8, W // 8) and alone.shape == (1, 16, hn, wn)
        assert torch.allclose(y[:, :, :hn, :wn], alone, atol=1e-5)
        assert (y[:, :, hn:, :] == 0).all() and (y[:, :, :, wn:] == 0).all()


def test_fold_is_the_eval_network():
    model = random_bn(M3.TextSpot3(TINY), 3)
    folded = M3.fold(model)
    q = M3.QTextSpot3(TINY, folded, [1.0] * len(folded))
    q.quant = False
    x = torch.randn(2, 1, 40, 72)
    with torch.no_grad():
        assert torch.allclose(model(x), q(x), atol=1e-5)
    y = q(x)
    assert M3.to_stride4(y, 40, 72).shape == (2, 4, 10, 18)


def fake_checkpoint(shape, seed):
    """A checkpoint as the trainer writes it: a random-init TextSpot3 with random BatchNorm statistics, folded,
    with scales calibrated on procedural regions (the largest |output| per layer, / 127)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = random_bn(M3.TextSpot3(shape), seed)
    folded = M3.fold(model)
    q = M3.QTextSpot3(shape, folded, [1.0] * len(folded))
    q.quant = False
    mx = [0.0] * len(folded)
    with torch.no_grad():
        for _ in range(4):
            g = M3.E.procedural_region(rng).astype(np.float32)[None, None]
            taps = []
            q((torch.from_numpy(g) - 128) / 128, taps=taps)
            mx = [max(a, float(t.abs().max())) for a, t in zip(mx, taps)]
    scales = [max(v, 1e-3) / 127 for v in mx]
    return {"kind": "textspot3-qat", "shape": shape, "seed": seed, "steps": 0, "w": [w for w, _ in folded],
            "b": [b for _, b in folded], "act_scale": scales}


def head_codes(q, g, masks=None, window=None):
    x = torch.from_numpy(g.astype(np.float32))[None, None]
    if window is not None:
        x = window
    with torch.no_grad():
        y = q((x - 128) / 128, masks=masks)
    return torch.round(y / getattr(q, f"s{len(q.spec) - 1}"))[0].numpy().astype(np.int64)


@pytest.mark.parametrize("shape,seed", [(TINY, 5), (SMALL, 6), (SMALL, 7)])
def test_the_fake_quantised_network_is_the_blob(tmp_path, shape, seed):
    """QTextSpot3 (the export's arithmetic in float, shared scales) and the twin's integer network give the same
    head codes up to the usual rounding: all within one code, most exactly; in a padded window with masks too."""
    ck = fake_checkpoint(shape, seed)
    torch.save(ck, tmp_path / "final.pt")
    net, vtsp = M3.export_vtsp(tmp_path / "final.pt")
    m = T3.Model(vtsp)
    assert (m.version, m.half_above, m.half_above_h, m.conf_min, m.name) == (3, 480, 272, 1, shape["name"])
    assert m.net.ckpt == hashlib.sha256((tmp_path / "final.pt").read_bytes()).digest()[:8].hex()
    assert m.net.blob_bytes == len(net) and len(vtsp) == M3.vtsp_bytes(shape)
    q = M3.QTextSpot3(shape, list(zip(ck["w"], ck["b"])), ck["act_scale"]).eval()
    rng = np.random.default_rng(seed)
    exact = near = total = 0
    for _ in range(3):
        g = M3.E.procedural_region(rng)
        codes = head_codes(q, g)
        ref = T3.run_head(m, g)
        assert codes.shape == ref.shape
        near += int((np.abs(codes - ref) <= 1).sum())
        exact += int((codes == ref).sum())
        total += ref.size
        # the same region in a padded window, masked: the region's cells are the region's own codes
        H, W = g.shape[0] + 21, g.shape[1] + 13
        win = np.full((H, W), 128, np.float32)
        win[:g.shape[0], :g.shape[1]] = g
        masks = M3.layer_masks(shape, [g.shape], H, W)
        wc = head_codes(q, g, masks, torch.from_numpy(win)[None, None])
        assert (wc[:, :ref.shape[1], :ref.shape[2]] == codes).all()
    assert near == total and exact >= 0.95 * total, (exact, near, total)
    assert (M3.to_stride4(torch.from_numpy(ref[None]).double(), *g.shape)[0].numpy() ==
            T2.head_to_maps(ref, *g.shape)).all()


def test_export_refuses_other_checkpoints(tmp_path):
    ck = fake_checkpoint(TINY, 1)
    torch.save(dict(ck, kind="textdet-qat"), tmp_path / "a.pt")
    with pytest.raises(SystemExit):
        M3.export_checkpoint(tmp_path / "a.pt")
    torch.save(dict(ck, w=ck["w"][:-1], b=ck["b"][:-1]), tmp_path / "b.pt")
    with pytest.raises(SystemExit):
        M3.export_checkpoint(tmp_path / "b.pt")
    folded = [(w.double(), b.double()) for w, b in zip(ck["w"], ck["b"])]
    q = M3.quantize_layers(TINY, folded, ck["act_scale"])
    assert all(np.abs(L["wq"]).max() <= 127 for L in q)
    # the head's requantisation: per channel, K and O at 1/16 per code, A1 and A2 at 1/2 px
    assert M3.HEAD_SCALES[:4] == (1 / 16, 1 / 16, 1 / 2, 1 / 2)


def test_random_blobs_of_every_probe():
    for name, s in M3.PROBES.items():
        c = M3.costs(s)
        assert c["vtsp_bytes"] <= 512000 and s["point"] in ("S", "M", "L"), name
    blob, src = M3.random_blob(next(iter(M3.PROBES)), 1, n_cal=1)
    m = T3.Model(blob)
    assert m.version == 3 and "seed 1" in src

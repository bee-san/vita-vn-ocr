"""Tests of textspot2's trainer (train/textspot2_train.py, issue #106): the weighted targets and loss against #97's at
weight 1, the class weights, the EMA arithmetic, the BatchNorm re-estimation and the swap at the QAT start, the
long schedule's LR, the distillation loss on a tiny teacher, the data rule, the refusals, and a 3-step run of every
recipe on tiny hand-made shards in #97's format (a lab run on weight-1 labels is textspot's run bit for bit). No
screenshots, fonts or models; no network.
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_data as DATA  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_train as TT  # noqa: E402
import textdet_twin as TD  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_train as T2  # noqa: E402
import textspot_train as ST  # noqa: E402
import textspot_twin as TS  # noqa: E402

PROTOCOL = json.loads((ROOT / "eval" / "textspot2_protocol.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def shapes_unchanged():
    """No test here may leave train/textdet_model.SHAPES changed (other tests compare it with #97's protocol)."""
    before = dict(TM.SHAPES)
    yield
    assert dict(TM.SHAPES) == before


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the protocol's recipes

def test_recipes_are_the_protocols():
    assert set(T2.RECIPES) == set(PROTOCOL["recipes"]) - {"r0", "common"}
    for name, r in T2.RECIPES.items():
        assert r["arm"] == PROTOCOL["recipes"][name]["arm"]
    assert (T2.SCALE, T2.HALF_ABOVE, T2.WIN_H, T2.WIN_W, T2.BATCH) == ("h2", 600, 256, 512, 16)
    assert T2.RECIPES["long"]["steps"] == 60000 and T2.RECIPES["long"]["lr"] == 2e-3
    assert T2.RECIPES["long"]["warmup"] == 1000 and T2.RECIPES["long"]["extras"] == ("ema",)
    for name in ("lab", "data", "ema", "cls", "kd"):
        r = T2.RECIPES[name]
        assert (r["steps"], r["lr"], r["warmup"]) == (30000, 4e-3, 500)
    assert T2.RECIPES["ema"]["extras"] == ("ema",) and T2.RECIPES["cls"]["extras"] == ("cls",)
    assert T2.RECIPES["lab"]["extras"] == T2.RECIPES["data"]["extras"] == () and T2.RECIPES["kd"]["extras"] is None
    assert int(TT.QAT_FRAC * 30000) == 24000 and int(TT.QAT_FRAC * 60000) == 48000
    assert (T2.EMA_DECAY, T2.EMA_START, T2.BN_BATCHES, TT.CAL_BATCHES) == (0.999, 1000, 32, 32)
    assert T2.CLS_CLASSES == ("dialogue", "name", "choice") and T2.CLS_FACTOR == 2
    assert T2.MIX == TT.MIX and T2.MIN_ACROSS == 4 and T2.CLIP == 5.0


def test_the_shapes_are_the_protocols_and_textdet_model_is_left_alone():
    before = dict(TM.SHAPES)
    for name in S2.VTDT:
        s = T2.shape_of(name)
        assert s == {k: (list(v) if isinstance(v, list) else v) for k, v in S2.SHAPES[name].items()
                     if k in T2.SHAPE_FIELDS}
        if name in TM.SHAPES:                                      # textspot's s4p_c8 and s4_c24n4
            assert s == TM.SHAPES[name]
        assert TM.layers(s)[-1]["cout"] == TM.N_OUT
    assert dict(TM.SHAPES) == before and not set(S2.D2S) & set(TM.SHAPES)
    with pytest.raises(KeyError):
        T2.shape_of("s4_c99")


def test_run_names_and_extras():
    assert T2.run_name("data", "s4p_c8", 2) == "data_h2_s4p_c8_seed2"
    assert T2.parse_extras("cls,ema") == ("ema", "cls") and T2.parse_extras("none") == ()
    assert T2.parse_extras("ema") == ("ema",) and T2.parse_extras(None) is None
    for bad in ("", "ema,ema", "dropout", "none,ema"):
        with pytest.raises(ValueError):
            T2.parse_extras(bad)


# ---------------------------------------------------------------- the schedule

def test_the_default_schedule_is_textdet_lr_at():
    for step in range(30000):
        assert T2.lr_at(step, 30000) == TT.lr_at(step, 30000)


def test_the_long_schedule():
    lr = [T2.lr_at(s, 60000, 2e-3, 1000) for s in range(60000)]
    assert lr[0] == pytest.approx(2e-6) and lr[499] == pytest.approx(1e-3)
    assert lr[999] == pytest.approx(2e-3) and lr[1000] == pytest.approx(2e-3)
    assert all(a < b for a, b in zip(lr[:999], lr[1:1000]))                 # linear warm-up
    assert all(a >= b for a, b in zip(lr[1000:], lr[1001:]))                # then the cosine only falls
    assert lr[30500] == pytest.approx((2e-3 + 1e-4) / 2)                    # half way through the cosine
    assert lr[48000] == pytest.approx(1e-4 + 0.5 * 1.9e-3 * (1 + math.cos(math.pi * 47000 / 59000)))
    assert lr[-1] == pytest.approx(1e-4, abs=1e-8) and min(lr[1000:]) >= 1e-4
    assert max(lr) == pytest.approx(2e-3)


# ---------------------------------------------------------------- weights and targets

def test_line_weights_and_class_weights():
    assert T2.line_weight([0, 0, 9, 9, 0, "dialogue"]) == 1.0
    assert T2.line_weight([0, 0, 9, 9, 0, "dialogue"], cls=True) == 2.0
    assert T2.line_weight([0, 0, 9, 9, 0, "text", 0.75]) == 0.75
    assert T2.line_weight([0, 0, 9, 9, 0, "text", 0.75], cls=True) == 0.75          # real lines: class "text"
    assert T2.line_weight([0, 0, 9, 9, 0, "name", 0.6], cls=True) == pytest.approx(1.2)
    assert T2.line_weight([0, 0, 9, 9, 1, "choice"], cls=True) == 2.0
    for c in ("ui", "scene", "text"):
        assert T2.line_weight([0, 0, 9, 9, 0, c], cls=True) == 1.0
    assert T2.line_weight([0, 0, 9, 9, 0]) == 1.0 and T2.line_weight([0, 0, 9, 9, 0, "ui", None]) == 1.0
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            T2.line_weight([0, 0, 9, 9, 0, "text", bad])


def random_lines(rng, n, H, W, weights=False):
    out = []
    for _ in range(n):
        vertical = int(rng.random() < 0.3)
        if vertical:
            x0, w = int(rng.integers(-20, W)), int(rng.integers(2, 40))
            y0, h = int(rng.integers(-40, H)), int(rng.integers(10, 300))
        else:
            x0, w = int(rng.integers(-40, W)), int(rng.integers(10, 400))
            y0, h = int(rng.integers(-20, H)), int(rng.integers(2, 40))
        ln = [x0, y0, x0 + w, y0 + h, vertical, str(rng.choice(["dialogue", "name", "ui", "choice", "scene", "text"]))]
        if weights:
            ln.append(float(rng.choice([1.0, 0.5, 0.8, 0.95])))
        out.append(ln)
    return out


def test_at_scale_is_textspots_with_the_weights():
    rng = np.random.default_rng(0)
    for f in (1, 2):
        for _ in range(50):
            lines = random_lines(rng, 12, 600, 1000, weights=True)
            ignore = [[a, b, a + 30, b + 10] for a, b in zip(rng.integers(0, 900, 3).tolist(),
                                                             rng.integers(0, 500, 3).tolist())]
            pos, ign = T2.at_scale(lines, ignore, f)
            want_pos, want_ign = ST.at_scale(lines, ignore, f)
            assert [p[:5] for p in pos] == want_pos and ign == want_ign
            kept = [ln for ln in lines if [*((ST.half_box(ln) if f == 2 else ln[:4])), ln[4]] in want_pos]
            assert [p[5] for p in pos] == [T2.line_weight(ln) for ln in kept]
            pos_c, _ = T2.at_scale(lines, ignore, f, cls=True)
            assert [p[5] for p in pos_c] == [T2.line_weight(ln, True) for ln in kept]


@pytest.mark.parametrize("s", [4, 8])
def test_weighted_targets_equal_textdet_targets_at_weight_1(s):
    rng = np.random.default_rng(s)
    for k in range(150):
        h, w = int(rng.integers(20, 257)), int(rng.integers(20, 513))
        wx, wy = int(rng.integers(0, 300)), int(rng.integers(0, 200))
        lines = [[ln[0] + wx, ln[1] + wy, ln[2] + wx, ln[3] + wy, ln[4]] for ln in random_lines(rng, 14, h, w)]
        ignore = [[wx + int(a), wy + int(b), wx + int(a) + 40, wy + int(b) + 12]
                  for a, b in zip(rng.integers(-20, w, 3), rng.integers(-10, h, 3))]
        with ST.window_size():
            want = TT.targets(lines, ignore, wx, wy, w, h, s)
        got = T2.targets(lines, ignore, wx, wy, w, h, s)
        for a, b in zip(got[:6], want):
            assert a.dtype == b.dtype and np.array_equal(a, b)
        assert got[6].shape == want[0].shape and np.all(got[6] == 1)
        # explicit weight 1 on every line: the same
        got1 = T2.targets([[*ln, 1.0] for ln in lines], ignore, wx, wy, w, h, s)
        for a, b in zip(got1, got):
            assert np.array_equal(a, b)
    assert (TT.WIN_H, TT.WIN_W) == (192, 512)


def test_the_weight_map():
    # two horizontal lines (weights 0.5 and 2), a vertical one (0.8) and two lines that claim one cell
    lines = [[10, 10, 200, 26, 0, 0.5], [10, 60, 200, 76, 0, 2.0], [300, 20, 316, 200, 1, 0.8],
             [10, 120, 100, 136, 0, 0.7], [60, 122, 160, 138, 0, 3.0]]
    K, O, A1, A2, v, k, W = T2.targets(lines, [], 0, 0, 512, 256, 4)  # noqa: E741
    base = T2.targets([ln[:5] for ln in lines], [], 0, 0, 512, 256, 4)
    for a, b in zip((K, O, A1, A2, v, k), base[:6]):
        assert np.array_equal(a, b)                      # the weights change nothing but W
    for n, ln in enumerate(lines[:3]):
        one = T2.targets([ln], [], 0, 0, 512, 256, 4)[5]
        assert np.all(W[one & k] == np.float32(ln[5])) and (one & k).any()
    pair = T2.targets(lines[3:], [], 0, 0, 512, 256, 4)
    conflict = T2.targets(lines[3:4], [], 0, 0, 512, 256, 4)[5] & T2.targets(lines[4:5], [], 0, 0, 512, 256, 4)[5]
    assert conflict.any() and not pair[5][conflict].any() and np.all(pair[6][conflict] == 1)
    assert np.all(W[~k] == 1)


class StubPool:
    def __init__(self, canvas, smp):
        self.keys, self.p, self.items, self._canvas = ["synth"], np.array([1.0]), {"synth": [(0, smp)]}, canvas

    def canvas(self, d, smp):
        return self._canvas


def test_the_batch_is_textspots_at_weight_1_and_the_classes_weigh_2():
    rng = np.random.default_rng(9)
    g = rng.integers(0, 60, (545, 961)).astype(np.uint8)
    g[401:431, 100:500] = 240
    smp = {"lines": [[100, 401, 500, 431, 0, "dialogue"], [700, 10, 703, 400, 1, "ui"],
                     [100, 300, 500, 330, 0, "ui"], [600, 200, 640, 500, 1, "name"]],
           "ignore": [[800, 300, 900, 340]], "src": "synth:frame"}
    for s in (4, 8):
        got = T2.make_batch(StubPool(g, smp), np.random.default_rng(10), s)
        want = ST.make_batch(StubPool(g, smp), np.random.default_rng(10), s, "h2")
        assert len(got) == 9
        for a, b in zip(got[:8], want):
            assert torch.equal(a, b)
        assert torch.all(got[8] == 1)
        cls = T2.make_batch(StubPool(g, smp), np.random.default_rng(10), s, cls=True)
        for a, b in zip(cls[:8], want):
            assert torch.equal(a, b)
        km, W = cls[7], cls[8]
        assert torch.all(W[~km] == 1) and torch.all((W[km] == 1) | (W[km] == 2)) and (W[km] == 2).any()
        assert (W[km] == 1).any()                         # the ui line keeps 1
    assert (TT.WIN_H, TT.WIN_W) == (192, 512)


# ---------------------------------------------------------------- the losses

def fake_maps(seed, n=3, h=16, w=32):
    g = torch.Generator().manual_seed(seed)
    out = torch.randn(n, 4, h, w, generator=g) * 3
    K = (torch.rand(n, h, w, generator=g) < 0.2).float()
    O = ((torch.rand(n, h, w, generator=g) < 0.3).float() * K)  # noqa: E741
    A1 = torch.rand(n, h, w, generator=g) * 12 * K
    A2 = torch.rand(n, h, w, generator=g) * 12 * K
    vm = torch.rand(n, h, w, generator=g) < 0.9
    km = K.bool() & vm
    return out, K, O, A1, A2, vm, km


def test_the_weighted_loss_is_textdets_at_weight_1():
    for seed in range(5):
        out, K, O, A1, A2, vm, km = fake_maps(seed)  # noqa: E741
        t, p = T2.loss_of(out, K, O, A1, A2, vm, km, torch.ones_like(K))
        t0, p0 = TT.loss_of(out, K, O, A1, A2, vm, km)
        assert torch.equal(t, t0) and p == p0


def test_the_weights_multiply_the_kernel_cells_terms():
    out, K, O, A1, A2, vm, km = fake_maps(7)  # noqa: E741
    W = torch.where(km, torch.full_like(K, 0.5), torch.ones_like(K))
    _, p = T2.loss_of(out, K, O, A1, A2, vm, km, W)
    _, p0 = TT.loss_of(out, K, O, A1, A2, vm, km)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], K, reduction="none")
    n_valid = vm.float().sum()
    want_k = float(((bce * vm) * torch.where(km, 0.5, 1.0)).sum() / n_valid)
    assert p["k_bce"] == pytest.approx(want_k, rel=1e-6)
    assert p["dice"] == p0["dice"]                                   # dice is #97's
    assert p["o"] == pytest.approx(0.5 * p0["o"], rel=1e-6) and p["a"] == pytest.approx(0.5 * p0["a"], rel=1e-6)


def test_the_kd_loss():
    g = torch.Generator().manual_seed(3)
    t = torch.randn(2, 4, 8, 16, generator=g) * 2
    t[:, 2:] = t[:, 2:].abs() * 6
    vm = torch.rand(2, 8, 16, generator=g) < 0.8
    y = torch.randn(2, 4, 8, 16, generator=g, requires_grad=True)
    total, p = T2.kd_loss(y, t, vm)
    F = torch.nn.functional
    pk = torch.sigmoid(t[:, 0])
    tm = pk > 0.5
    want_k = float((F.binary_cross_entropy_with_logits(y[:, 0], pk, reduction="none")[vm]).mean())
    want_o = float(F.binary_cross_entropy_with_logits(y[:, 1], torch.sigmoid(t[:, 1]), reduction="none")[tm].mean())
    d1, d2 = (y[:, 2] - t[:, 2]) / 8, (y[:, 3] - t[:, 3]) / 8
    sl1 = torch.where(d1.abs() < 1, 0.5 * d1 ** 2, d1.abs() - 0.5) + torch.where(d2.abs() < 1, 0.5 * d2 ** 2,
                                                                                 d2.abs() - 0.5)
    want_a = float(sl1[tm].mean())
    assert p["kd_k"] == pytest.approx(want_k, rel=1e-5) and p["kd_o"] == pytest.approx(want_o, rel=1e-5)
    assert p["kd_a"] == pytest.approx(want_a, rel=1e-5) and p["kd_cells"] == int(tm.sum())
    assert float(total) == pytest.approx(want_k + 0.5 * want_o + want_a, rel=1e-5)
    # no gradient reaches the teacher, and a student equal to its teacher has a zero gradient
    s = t.clone().requires_grad_(True)
    tt = t.clone().requires_grad_(True)
    total, _ = T2.kd_loss(s, tt, vm)
    total.backward()
    assert tt.grad is None and float(s.grad.abs().max()) < 1e-6
    # O and A only on the teacher's kernel cells; K only on the valid cells
    s = t.clone().requires_grad_(True)
    s2 = s * 1.0
    total, _ = T2.kd_loss(s2 + 1.0, t, vm)
    total.backward()
    assert torch.all(s.grad[:, 1:][~tm.unsqueeze(1).expand(-1, 3, -1, -1)] == 0)
    assert torch.all(s.grad[:, 0][~vm] == 0)


# ---------------------------------------------------------------- EMA, the BatchNorm re-estimation, the swap

def test_the_ema_arithmetic():
    p = [torch.nn.Parameter(torch.full((3,), 2.0)), torch.nn.Parameter(torch.zeros(2, 2))]
    e = T2.EMA(0.5).step(p)
    assert e.steps == 1 and torch.equal(e.avg[0], torch.full((3,), 2.0, dtype=torch.float64))
    with torch.no_grad():
        p[0].fill_(4.0)
        p[1].fill_(1.0)
    e.step(p)
    assert torch.allclose(e.avg[0], torch.full((3,), 3.0, dtype=torch.float64))
    assert torch.allclose(e.avg[1], torch.full((2, 2), 0.5, dtype=torch.float64))
    with torch.no_grad():
        p[0].fill_(0.0)
    e.step(p)
    assert torch.allclose(e.avg[0], torch.full((3,), 1.5, dtype=torch.float64))
    # decay 0.999, from c to a constant c': avg = c' + (c - c') d^n after n updates
    q = [torch.nn.Parameter(torch.tensor([1.0, -3.0]))]
    e = T2.EMA().step(q)
    with torch.no_grad():
        q[0].copy_(torch.tensor([5.0, 1.0]))
    for _ in range(2500):
        e.step(q)
    d = 0.999 ** 2500
    assert torch.allclose(e.avg[0], torch.tensor([5.0 + (1 - 5) * d, 1.0 + (-3 - 1) * d], dtype=torch.float64))
    target = [torch.nn.Parameter(torch.zeros(2))]
    e.copy_to(target)
    assert torch.allclose(target[0], e.avg[0].float())
    with pytest.raises(ValueError):
        e.copy_to(target + target)
    with pytest.raises(ValueError):
        T2.EMA().copy_to(target)


def test_the_batchnorm_reestimation():
    torch.manual_seed(0)
    shape = T2.shape_of("s4p_c6n4")
    model = TM.TextDet(shape)
    g = torch.Generator().manual_seed(1)
    batches = []
    for _ in range(3):
        x = torch.randint(0, 256, (4, 1, 256, 512), generator=g, dtype=torch.uint8)
        batches.append((x, torch.tensor([[256, 512], [200, 300], [256, 100], [64, 512]])))
    bn0 = model.body[0][1]
    bn0.running_mean.fill_(7.0)
    bn0.momentum = 0.1
    model.eval()
    n = T2.reestimate_bn(model, iter(batches), shape, 3)
    assert n == len([m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])
    assert not model.training and bn0.momentum == 0.1
    conv = model.body[0][0]
    with torch.no_grad():
        ys = [conv(TT.to_input(x)) for x, _ in batches]
    want_mean = torch.stack([y.mean((0, 2, 3)) for y in ys]).mean(0)
    want_var = torch.stack([y.var((0, 2, 3), unbiased=True) for y in ys]).mean(0)
    assert torch.allclose(bn0.running_mean, want_mean, atol=1e-5)
    assert torch.allclose(bn0.running_var, want_var, rtol=1e-4)


# ---------------------------------------------------------------- shards and runs

def test_the_swap_at_the_qat_start_and_the_qat_average_is_the_model(tmp_path, monkeypatch):
    mans, dirs = tiny_shards(tmp_path)
    made, events, folded_from = [], [], []

    class Recording(T2.EMA):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.copied = []
            made.append(self)

        def copy_to(self, params):
            params = list(params)
            super().copy_to(params)
            self.copied.append([p.detach().clone() for p in params])
            events.append(f"copy_to{made.index(self)}")

    real_fold, real_bn, real_cal = TM.fold, T2.reestimate_bn, TT.calibrate

    def fold(model):
        folded_from.append([p.detach().clone() for p in model.parameters()])
        events.append("fold")
        return real_fold(model)

    def bn(*a, **k):
        events.append("reestimate_bn")
        return real_bn(*a, **k)

    def cal(*a, **k):
        events.append("calibrate")
        return real_cal(*a, **k)

    monkeypatch.setattr(T2, "EMA", Recording)
    monkeypatch.setattr(T2, "reestimate_bn", bn)
    monkeypatch.setattr(TM, "fold", fold)
    monkeypatch.setattr(TT, "calibrate", cal)
    out = tmp_path / "ema"
    T2.main(base_args(mans, dirs, "ema", out, steps=5))                     # QAT from step 4, EMA from step 1
    float_ema, qat_ema = made
    assert events == ["copy_to0", "reestimate_bn", "calibrate", "fold", "copy_to1"]
    assert float_ema.steps == 4 and qat_ema.steps == 2                      # steps 1-4; folded + 1 QAT step
    # the folded network is the average's (its BatchNorm statistics re-estimated after the swap)
    assert all(torch.equal(a, b) for a, b in zip(float_ema.copied[0], folded_from[0]))
    assert all(torch.equal(p, a.float()) for p, a in zip(folded_from[0], float_ema.avg))
    # the run's model is the QAT average
    ck = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(w, a.float().double()) for w, a in zip(ck["w"] + ck["b"], qat_ema.avg))


def canvas(rng, h, w, boxes):
    g = rng.integers(0, 60, (h, w)).astype(np.uint8)
    for x0, y0, x1, y1 in boxes:
        g[y0:y1, x0:x1] = 230
    return g


def tiny_shards(tmp_path, weights=True):
    """The protocol's directories in miniature: synth_frames (a synthetic canvas wider than 600 px, halved at h2),
    lab_realplus and more_realplus (a real canvas each, lines with a 7th field), a synthetic manifest and a textspot2
    pin with skip lists."""
    rng = np.random.default_rng(11)
    dirs = {k: tmp_path / k for k in ("synth_frames", "lab_realplus", "more_realplus")}
    for d in dirs.values():
        d.mkdir()
    lines = [[100, 40, 400, 58, 0, "dialogue"], [50, 200, 650, 226, 0, "name"], [660, 20, 680, 280, 1, "dialogue"],
             [10, 100, 60, 106, 0, "ui"], [420, 120, 600, 140, 0, "choice"]]
    synth = {"canvas": canvas(rng, 300, 700, [ln[:4] for ln in lines]), "lines": lines,
             "ignore": [[300, 100, 340, 110]], "src": "synth:frame", "kind": "vita", "vn": None, "shot": None}
    w1, w2 = (0.75, 0.6) if weights else (1.0, 1.0)
    lab = {"canvas": canvas(rng, 136, 240, [[20, 50, 180, 66], [20, 90, 200, 110]]),
           "lines": [[20, 50, 180, 66, 0, "text", 1.0], [20, 90, 200, 110, 0, "text", w1]],
           "ignore": [[200, 10, 230, 30]], "src": "realplus", "kind": "psp", "vn": "v1", "shot": "v1_1"}
    more = {"canvas": canvas(rng, 120, 200, [[30, 40, 170, 58], [10, 80, 26, 118]]),
            "lines": [[30, 40, 170, 58, 0, "text", w2], [10, 80, 26, 118, 1, "text", 1.0]],
            "ignore": [], "src": "realplus", "kind": "psp", "vn": "v2", "shot": "v2_1"}
    skipped = {"canvas": canvas(rng, 64, 96, [[10, 20, 80, 36]]), "lines": [[10, 20, 80, 36, 0, "text", 1.0]],
               "ignore": [], "src": "realplus", "kind": "psp", "vn": "v9", "shot": "v9_1"}
    DATA.write_shard(dirs["synth_frames"] / "synth_frames_0000", [synth])
    DATA.write_shard(dirs["lab_realplus"] / "lab_realplus_0000", [lab, skipped])
    DATA.write_shard(dirs["more_realplus"] / "more_realplus_0000", [more])

    def pin(d, n):
        return {"dir": d.name, "name": n, "npy_sha256": sha(d / f"{n}.npy"), "json_sha256": sha(d / f"{n}.json")}

    man_s = tmp_path / "synth_man.json"
    man_s.write_text(json.dumps({"kind": "test-shards", "shards": [pin(dirs["synth_frames"], "synth_frames_0000")]}))
    man_2 = tmp_path / "textspot2_shards_v1.json"
    man_2.write_text(json.dumps({
        "kind": "textspot2-shards",
        "shards": [pin(dirs["lab_realplus"], "lab_realplus_0000"), pin(dirs["more_realplus"], "more_realplus_0000")],
        "real": {"measured": {"skip": {
            "eval_screenshot": {"shots": 1, "canvases": 1, "by_source": {"realplus": 1}, "skip_shots": ["v9_1"]},
            "linefind_list_vn": {"shots": 0, "canvases": 0, "by_source": {}, "skip_vns": ["v99"]}}}}}))
    return [man_s, man_2], dirs


def base_args(mans, dirs, recipe, out, more=True, shape="s4p_c6n4", seed=1, steps=3):
    d = [dirs["synth_frames"], dirs["lab_realplus"]] + ([dirs["more_realplus"]] if more else [])
    return ["--recipe", recipe, "--shape", shape, "--seed", str(seed), "--manifest", *map(str, mans),
            "--shards", *map(str, d), "--out", str(out), "--steps", str(steps), "--threads", "1",
            "--log-every", "1", "--cal-batches", "2", "--bn-batches", "2", "--ema-start", "1", "--workers", "0",
            "--allow-missing-sources"]


def procedural_frame(rng, H, W):
    g = np.full((H, W), int(rng.integers(0, 256)), np.int32)
    for _ in range(12):
        y0, x0 = int(rng.integers(0, H)), int(rng.integers(0, W))
        sub = g[y0:y0 + int(rng.integers(4, 40)), x0:x0 + int(rng.integers(8, 300))]
        sub[:] = int(rng.integers(0, 256))
        sub[:, ::int(rng.integers(2, 6))] = int(rng.integers(0, 256))
    return np.clip(g + rng.integers(-12, 13, g.shape), 0, 255).astype(np.uint8)


def check_blobs(run, shape):
    done = json.loads((run / "done.json").read_text())
    vtd, vtsp = (run / "final.vtd").read_bytes(), (run / "final.vtsp").read_bytes()
    assert done["vtd_sha256"] == sha(run / "final.vtd") and done["vtsp_sha256"] == sha(run / "final.vtsp")
    assert done["vtsp_bytes"] == len(vtsp) == len(vtd) + 32 and done["pt_sha256"] == sha(run / "final.pt")
    assert TD.Model(vtd).name == shape
    m = TS.Model(vtsp)
    assert (m.half_above, m.conf_min, m.name) == (600, 1, shape) and TS.pack(vtd, 600, 1) == vtsp
    rng = np.random.default_rng(12)
    for H, W in ((544, 960), (272, 480)):
        r = TS.spot(m, procedural_frame(rng, H, W))
        assert all(0 <= b.x and 0 <= b.y and b.x + b.w <= W and b.y + b.h <= H for b in r.boxes)
    return done


def log_of(run):
    return [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]


def test_a_three_step_run_of_every_recipe(tmp_path):
    mans, dirs = tiny_shards(tmp_path)
    runs = {}
    for recipe in ("lab", "data", "ema", "cls", "long"):
        out = tmp_path / f"{recipe}_run"
        T2.main(base_args(mans, dirs, recipe, out, more=recipe != "lab"))
        runs[recipe] = out
    out = tmp_path / "kd_run"
    T2.main(base_args(mans, dirs, "kd", out, shape="s4p_c8") + ["--teacher", str(runs["data"]), "--extras", "ema,cls"])
    runs["kd"] = out
    for recipe, run in runs.items():
        info = json.loads((run / "run.json").read_text())
        shape = "s4p_c8" if recipe == "kd" else "s4p_c6n4"
        assert info["name"] == f"{recipe}_h2_{shape}_seed1" and info["recipe"] == recipe
        assert (info["kind"], info["half_above"], info["scale"], info["steps"], info["qat_start"]) == \
            ("textspot2-run", 600, "h2", 3, 2)
        extras = {"lab": [], "data": [], "ema": ["ema"], "cls": ["cls"], "long": ["ema"], "kd": ["ema", "cls"]}
        assert info["extras"] == extras[recipe] and info["settings"]["extras"] == extras[recipe]
        assert (info["settings"]["ema"] is not None) == ("ema" in extras[recipe])
        assert (info["settings"]["cls"] is not None) == ("cls" in extras[recipe])
        assert info["settings"]["lr"]["peak"] == (2e-3 if recipe == "long" else 4e-3)
        assert info["skip"]["dropped_total"] == 1                     # v9_1 is on the pin's skip list
        pm = {r["manifest"]: r for r in info["skip"]["per_manifest"]}
        assert set(pm) == {str(m) for m in mans}
        t2 = pm[str(mans[1])]                                          # a textspot2 pin: 0 expected, 1 dropped here
        assert t2["textspot2_pin"] and t2["carries_skip_lists"] and t2["expected_dropped"] == 0
        assert t2["dropped"] == 1 and t2["as_expected"] is False and t2["pin_counts"]["eval_screenshot"]["shots"] == 1
        assert pm[str(mans[0])]["dropped"] == 0 and pm[str(mans[0])]["expected_dropped"] is None
        assert pm[str(mans[0])]["shards_read"] == 1 and t2["shards_read"] == (1 if recipe == "lab" else 2)
        assert info["pool"]["by_source"]["realplus"]["canvases"] == (1 if recipe == "lab" else 2)
        assert info["pool"]["weights"]["realplus"]["weighted"] == (1 if recipe == "lab" else 2)
        assert info["data_rule"]["unread"] == ({"more_realplus": 1} if recipe == "lab" else {})
        for f in ("train/textspot2_train.py", "train/textspot_train.py", "train/textdet_train.py",
                  "train/textspot2_shapes.py", "synth/textspot_twin.py"):
            assert info["code_sha256"][f] == sha(ROOT / f)
        assert info["protocol_sha256"] == sha(ROOT / "eval" / "textspot2_protocol.json")
        log = log_of(run)
        steps = [r for r in log if "lr" in r]
        assert [r["step"] for r in steps] == [1, 2, 3] and any(r.get("qat") for r in log)
        assert all(np.isfinite(r[k]) for r in steps for k in ("k_bce", "dice", "o", "a", "loss"))
        if "ema" in extras[recipe]:
            assert [r["step"] for r in log if r.get("ema") == "start"] == [1]
            swap = [r for r in log if r.get("ema_swap")]
            assert len(swap) == 1 and swap[0]["step"] == 2 and swap[0]["ema_steps"] == 2
            assert swap[0]["bn_batches"] == 2 and swap[0]["bn_layers"] > 0
            avg = [r for r in log if r.get("qat_average") == "the run's model"]
            assert len(avg) == 1 and avg[0]["qat_average_steps"] == 2         # the folded weights + 1 QAT step
            assert log.index(swap[0]) < next(i for i, r in enumerate(log) if r.get("qat"))
        else:
            assert not any(r.get("ema_swap") or r.get("ema") for r in log)
        if recipe == "kd":
            assert all(k in r for r in steps for k in ("kd_k", "kd_o", "kd_a"))
            t = info["settings"]["kd"]["teacher"]
            assert t["name"] == "data_h2_s4p_c6n4_seed1" and t["pt_sha256"] == sha(runs["data"] / "final.pt")
            assert t["shape"] == "s4p_c6n4" and t["stride"] == 4
            for r in steps:
                assert r["loss"] == pytest.approx(r["k_bce"] + r["dice"] + 0.5 * r["o"] + r["a"] + r["kd_k"] +
                                                  0.5 * r["kd_o"] + r["kd_a"], abs=2e-4)
        else:
            assert info["settings"]["kd"] is None
            for r in steps:
                assert r["loss"] == pytest.approx(r["k_bce"] + r["dice"] + 0.5 * r["o"] + r["a"], abs=2e-4)
        check_blobs(run, shape)
    ck = torch.load(runs["ema"] / "final.pt", map_location="cpu", weights_only=False)
    assert ck["kind"] == "textdet-qat" and ck["shape"]["name"] == "s4p_c6n4" and len(ck["act_scale"]) == len(ck["w"])


def test_a_lab_run_on_weight_1_labels_is_textspots_run(tmp_path):
    """The same shards, seed and steps through train/textspot_train.py and the lab recipe: the same losses and the
    same QAT checkpoint weights, so the recipe changes nothing but the weights."""
    mans, dirs = tiny_shards(tmp_path, weights=False)
    a, b = tmp_path / "textspot", tmp_path / "lab"
    ST.main(["--shape", "s4p_c8", "--scale", "h2", "--seed", "4", "--manifest", *map(str, mans),
             "--shards", str(dirs["synth_frames"]), str(dirs["lab_realplus"]), "--out", str(a), "--steps", "4",
             "--threads", "1", "--log-every", "1", "--cal-batches", "2", "--workers", "0",
             "--allow-missing-sources"])
    T2.main(base_args(mans, dirs, "lab", b, more=False, shape="s4p_c8", seed=4, steps=4))
    la = [r for r in log_of(a) if "lr" in r]
    lb = [r for r in log_of(b) if "lr" in r]
    assert [{k: r[k] for k in ("step", "lr", "k_bce", "dice", "o", "a")} for r in la] == \
        [{k: r[k] for k in ("step", "lr", "k_bce", "dice", "o", "a")} for r in lb]
    ca = torch.load(a / "final.pt", map_location="cpu", weights_only=False)
    cb = torch.load(b / "final.pt", map_location="cpu", weights_only=False)
    assert ca["act_scale"] == cb["act_scale"]
    assert all(torch.equal(x, y) for x, y in zip(ca["w"] + ca["b"], cb["w"] + cb["b"]))
    va, vb = (a / "final.vtd").read_bytes(), (b / "final.vtd").read_bytes()
    assert va[:12] == vb[:12] and va[16:24] == vb[16:24] and va[32:] == vb[32:]     # all but CRC and ckpt sha


def test_the_data_rule(tmp_path):
    mans, dirs = tiny_shards(tmp_path)
    with pytest.raises(SystemExit, match="not the protocol's directories"):
        T2.main(base_args(mans, dirs, "lab", tmp_path / "a", more=True))          # lab never reads more_*
    with pytest.raises(SystemExit, match="more_<source>"):
        T2.main(base_args(mans, dirs, "data", tmp_path / "b", more=False))        # data needs more_*
    real = tmp_path / "realplus"
    real.mkdir()
    DATA.write_shard(real / "real_0000", [{"canvas": np.zeros((40, 60), np.uint8), "lines": [], "ignore": [],
                                           "src": "realplus", "kind": "psp", "vn": "v3", "shot": "v3_1"}])
    man_97 = tmp_path / "textdet_like.json"
    man_97.write_text(json.dumps({"kind": "textdet-shards", "shards": [
        {"dir": "realplus", "name": "real_0000", "npy_sha256": sha(real / "real_0000.npy"),
         "json_sha256": sha(real / "real_0000.json")}]}))
    args = base_args([*mans, man_97], dirs, "data", tmp_path / "c")
    args[args.index("--shards") + 1:args.index("--shards") + 1] = [str(real)]
    with pytest.raises(SystemExit, match="not the protocol's directories"):
        T2.main(args)                                                              # never #97's real directories
    args = base_args([*mans, man_97], dirs, "data", tmp_path / "d")
    with pytest.raises(SystemExit, match="are not read"):
        T2.main(args)                                                              # a manifest's shard left unread
    T2.main(base_args([*mans, man_97], dirs, "data", tmp_path / "e") + ["--any-shards", "--probe", "1"])
    info = json.loads((tmp_path / "e" / "run.json").read_text())
    assert info["data_rule"].startswith("any shard directories")
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_the_refusals(tmp_path):
    mans, dirs = tiny_shards(tmp_path)
    out = tmp_path / "x"
    for extra, msg in ((["--shape", "s8_c24n4x2"], "d2s"), (["--extras", "cls"], "--extras is for"),
                       (["--teacher", str(tmp_path)], "--teacher is required for kd and only")):
        args = base_args(mans, dirs, "ema", out)
        if extra[0] == "--shape":
            args[args.index("--shape") + 1] = extra[1]
        else:
            args += extra
        with pytest.raises(SystemExit):
            T2.parse(args)
    with pytest.raises(SystemExit):
        T2.parse(base_args(mans, dirs, "kd", out) + ["--teacher", str(tmp_path)])            # kd needs --extras
    with pytest.raises(SystemExit):
        T2.parse(base_args(mans, dirs, "kd", out) + ["--extras", "none"])                     # and a teacher
    with pytest.raises(SystemExit):
        T2.parse(base_args(mans, dirs, "ema", out) + ["--ema-start", "3"])                   # after the QAT start
    assert T2.parse(base_args(mans, dirs, "long", out) + ["--extras", "ema,cls"]).extras_set == ("ema", "cls")
    assert T2.parse(base_args(mans, dirs, "data", out) + ["--extras", "ema"]).extras_set == ("ema",)
    a = T2.parse(["--recipe", "long", "--shape", "s4p_c8", "--seed", "1", "--manifest", "m", "--shards", "d",
                  "--out", "o"])
    assert (a.steps, a.extras_set, a.ema_start, a.bn_batches, a.cal_batches) == (60000, ("ema",), 1000, 32, 32)
    used = tmp_path / "used"
    used.mkdir()
    (used / "log.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="not empty"):
        T2.main(base_args(mans, dirs, "data", used))
    # a teacher at another stride or scale
    t8 = tmp_path / "t8"
    T2.main(base_args(mans, dirs, "data", tmp_path / "t4"))
    t8.mkdir()
    ck = torch.load(tmp_path / "t4" / "final.pt", map_location="cpu", weights_only=False)
    torch.save({**ck, "shape": {**ck["shape"], "stride": 8}}, t8 / "final.pt")
    (t8 / "run.json").write_text((tmp_path / "t4" / "run.json").read_text())
    with pytest.raises(SystemExit, match="stride"):
        T2.main(base_args(mans, dirs, "kd", tmp_path / "k1") + ["--teacher", str(t8), "--extras", "none"])
    info = json.loads((tmp_path / "t4" / "run.json").read_text())
    (t8 / "run.json").write_text(json.dumps({**info, "half_above": 0}))
    torch.save(ck, t8 / "final.pt")
    with pytest.raises(SystemExit, match="half_above"):
        T2.main(base_args(mans, dirs, "kd", tmp_path / "k2") + ["--teacher", str(t8), "--extras", "none"])
    # a textspot2 pin without its skip lists
    doc = json.loads(mans[1].read_text())
    del doc["real"]
    mans[1].write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="without real.measured.skip"):
        T2.main(base_args(mans, dirs, "data", tmp_path / "k3"))
    # a bad line weight
    with pytest.raises(ValueError):
        T2.at_scale([[0, 0, 50, 20, 0, "text", -0.5]], [], 1)


def test_a_speed_probe_writes_no_model_and_the_cli(tmp_path):
    mans, dirs = tiny_shards(tmp_path)
    run = tmp_path / "probe"
    args = base_args(mans, dirs, "cls", run, steps=30000) + ["--probe", "2"]
    p = subprocess.run([sys.executable, str(ROOT / "train" / "textspot2_train.py"), *args], capture_output=True,
                       text=True, timeout=900, cwd=ROOT)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]
    assert "TEXTSPOT2_PROBE_DONE" in p.stdout
    probe = json.loads((run / "probe.json").read_text())
    assert probe["steps_timed"] == 2 and probe["ms_per_step"] > 0 and probe["name"] == "cls_h2_s4p_c6n4_seed1"
    info = json.loads((run / "run.json").read_text())
    assert info["half_above"] == 600 and info["steps"] == 30000 and info["qat_start"] == 24000
    assert "train/textspot2_train.py" in info["code_sha256"]
    assert not any((run / f).exists() for f in ("final.pt", "final.vtd", "final.vtsp", "done.json"))


def test_the_files_textspot2_pins_are_unchanged():
    pins = dict(PROTOCOL["sources_sha256"])
    for amend in PROTOCOL.get("amendments", []):
        pins.update(amend.get("sources_sha256", {}))
    for f in ("train/textspot_train.py", "train/textdet_train.py", "train/textdet_model.py",
              "train/textdet_export.py", "synth/textspot_twin.py", "synth/textdet_twin.py"):
        assert sha(ROOT / f) == pins[f], f


# ---------------------------------------------------------------- the sweep's plan, check and scripts

REPRO = ROOT / "eval" / "reproducibility" / "textspot2-20261020"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))
import textspot2_plan as PL  # noqa: E402


def test_the_stage1_plan_is_the_protocols():
    runs = PL.stage1()
    names = [r["run"] for r in runs]
    assert len(runs) == len(set(names)) == 38
    by = {}
    for r in runs:
        by.setdefault((r["recipe"], r["shape"]), []).append(r["seed"])
    want = {**{(rec, "s4_c24n4"): [1, 2, 3] for rec in ("lab", "data", "ema", "cls", "long")},
            ("long", "s4p_c8"): [1, 2, 3], ("r0", "s4_c24n4"): [3], ("r0", "s4p_c8"): [3],
            **{("data", sh): [1, 2, 3] for sh in ("s4p_c8", "s4p_c6n4", "s4p_c8n4", "s4p_c10n4", "s4p_c12",
                                                    "s4_c32n3")}}
    assert by == want
    assert set(sh for _, sh in want) == set(S2.VTDT)                    # every kroa shape, no d2s one
    for r in runs:
        assert r["steps"] == (60000 if r["recipe"] == "long" else 30000)
        assert r["timeout_s"] == (43200 if r["recipe"] == "long" else 21600)
        assert r["trainer"] == ("train/textspot_train.py" if r["recipe"] == "r0" else "train/textspot2_train.py")
        assert r["data"] == {"r0": "r0", "lab": "lab"}.get(r["recipe"], "data") and r["teacher"] is None
    stages = " ".join(PROTOCOL["stages"]["1"])
    for sh in ("s4p_c8", "s4p_c6n4", "s4p_c8n4", "s4p_c10n4", "s4p_c12", "s4_c32n3", "s4_c24n4"):
        assert sh in stages
    # longest first: by the probe's estimates when every run has one, else by steps x MACs
    est = {n: float(i) for i, n in enumerate(names)}
    assert [r["run"] for r in PL.order(PL.stage1(), est)] == names[::-1]
    fallback = PL.order(PL.stage1())
    assert fallback[0]["recipe"] == "long" and all(r["est_s"] is None for r in fallback)
    cost = [r["steps"] * PL.macs(r["shape"]) for r in fallback]
    assert cost == sorted(cost, reverse=True)


def test_the_stage2_plan():
    runs = PL.stage2("ema", "s4p_c8n4", "/t/runs/data_h2_s4p_c12_seed2", ["s4p_c12"], [])
    assert [r["run"] for r in runs] == [f"kd_h2_s4p_c8n4_seed{n}" for n in (1, 2, 3)] + \
        [f"long_h2_s4p_c12_seed{n}" for n in (1, 2, 3)]
    assert {r["extras"] for r in runs} == {"ema"} and runs[0]["teacher"] == "/t/runs/data_h2_s4p_c12_seed2"
    runs = PL.stage2("cls", "s4p_c8", "/t", ["s4p_c12"], [])
    assert [r["extras"] for r in runs] == ["cls"] * 3 + ["ema,cls"] * 3          # long always has ema
    runs = PL.stage2("none", None, None, [], ["s8_c24n4x2"])
    assert [r["run"] for r in runs] == [f"data_h2_s8_c24n4x2_seed{n}" for n in (1, 2, 3)]
    assert {r["extras"] for r in runs} == {"none"} and PL.order(runs)
    with pytest.raises(SystemExit):
        PL.stage2("ema", "s4p_c8", None)
    with pytest.raises(SystemExit):
        PL.stage2("ema", None, None, ["s4p_c12", "s4p_c12"])


def fake_herd(tmp_path):
    herd, repo = tmp_path / "herd", tmp_path / "repo"
    t2 = herd / "shared" / "artifacts" / "textspot2" / "shards"
    p2 = herd / "shared" / "private" / "textspot2" / "shards"
    for d in (t2 / "lab" / "lab_realplus", t2 / "lab" / "lab_teacher", p2 / "lab" / "lab_owner",
              t2 / "more" / "more_realplus", p2 / "more" / "more_owner"):
        d.mkdir(parents=True)
    (repo / "data" / "manifests").mkdir(parents=True)
    shards = [{"dir": d, "name": f"{d}_0000"} for d in ("lab_realplus", "lab_teacher", "lab_owner", "more_realplus",
                                                         "more_owner")]
    (repo / PL.T2_MANIFEST).write_text(json.dumps({"kind": "textspot2-shards", "shards": shards}))
    return herd, repo, t2, p2


def test_the_plan_command_lines(tmp_path):
    herd, repo, t2, p2 = fake_herd(tmp_path)
    p = PL.paths(herd)
    runs = {r["run"]: r for r in PL.stage1()}
    lab = PL.argv(runs["lab_h2_s4_c24n4_seed2"], "/o", repo, herd)
    i, j = lab.index("--shards"), lab.index("--out")
    assert lab[:9] == ["train/textspot2_train.py", "--recipe", "lab", "--shape", "s4_c24n4", "--seed", "2",
                       "--manifest", PL.SYNTH_MANIFEST] and lab[9] == PL.T2_MANIFEST
    assert lab[i + 1:j] == [str(p["TS"] / "shards" / "synth_frames"), str(p2 / "lab" / "lab_owner"),
                            str(t2 / "lab" / "lab_realplus"), str(t2 / "lab" / "lab_teacher")]
    assert lab[j:] == ["--out", "/o/lab_h2_s4_c24n4_seed2", "--threads", "5"]
    data = PL.argv(runs["data_h2_s4p_c12_seed1"], "/o", repo, herd)
    dirs = data[data.index("--shards") + 1:data.index("--out")]
    assert sorted(Path(d).name for d in dirs) == ["lab_owner", "lab_realplus", "lab_teacher", "more_owner",
                                                  "more_realplus", "synth_frames"]
    assert str(p2 / "more" / "more_owner") in dirs and str(p2 / "lab" / "lab_owner") in dirs
    r0 = PL.argv(runs["r0_h2_s4p_c8_seed3"], "/o", repo, herd)
    assert r0 == ["train/textspot_train.py", "--shape", "s4p_c8", "--scale", "h2", "--seed", "3", "--manifest",
                  PL.SYNTH_MANIFEST, PL.TEXTDET_MANIFEST, "--shards", str(p["TS"] / "shards" / "synth_frames"),
                  str(p["TD"] / "shards" / "realplus"), str(p["TD"] / "shards" / "teacher"),
                  str(p["TD"] / "shards" / "devnative"), str(p["PD"] / "shards" / "owner"),
                  "--out", "/o/r0_h2_s4p_c8_seed3", "--threads", "5"]
    kd = PL.argv(PL.stage2("ema,cls", "s4p_c8", "/t/x")[0], "/o", repo, herd)
    assert kd[-4:] == ["--extras", "ema,cls", "--teacher", "/t/x"]
    (t2 / "more" / "lab_teacher").mkdir()                                  # a basename found twice
    with pytest.raises(SystemExit, match="found 2 times"):
        PL.argv(runs["lab_h2_s4_c24n4_seed1"], "/o", repo, herd)
    # the CLI: a plan file, its order and NUL-separated arguments
    plan = tmp_path / "stage1.jsonl"
    PL.main(["stage1", "--out", str(plan)])
    assert len(PL.load(plan)) == 38


def test_check_run_and_runs_summary_on_a_tiny_run(tmp_path):
    mans, dirs = tiny_shards(tmp_path)
    runs = tmp_path / "runs"
    run = runs / "data_h2_s4p_c6n4_seed1"
    T2.main(base_args(mans, dirs, "data", run))
    fin = runs / "finished.txt"
    p = subprocess.run([sys.executable, str(REPRO / "check_run.py"), str(run), "--finished", str(fin)],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert p.returncode == 0 and "CHECK_OK data_h2_s4p_c6n4_seed1 " in p.stdout, p.stdout + p.stderr
    done = json.loads((run / "done.json").read_text())
    assert fin.read_text().split() == ["data_h2_s4p_c6n4_seed1", done["vtsp_sha256"], str(done["vtsp_bytes"]), "3",
                                       str(done["wall_seconds"])]
    subprocess.run([sys.executable, str(REPRO / "check_run.py"), str(run), "--finished", str(fin)], check=True,
                   capture_output=True, timeout=600, cwd=ROOT)
    assert len(fin.read_text().splitlines()) == 1                           # recorded once
    other = runs / "data_h2_s4p_c6n4_seed2"
    run.rename(other)
    p = subprocess.run([sys.executable, str(REPRO / "check_run.py"), str(other)], capture_output=True, text=True,
                       timeout=600, cwd=ROOT)
    assert p.returncode == 1 and "BAD run.json names data_h2_s4p_c6n4_seed1" in p.stdout
    other.rename(run)
    plan = tmp_path / "plan.jsonl"
    PL.write([PL.spec("data", "s4p_c6n4", 1), PL.spec("data", "s4p_c6n4", 2)], plan)
    import runs_summary as RS
    doc = RS.summarise(runs, plan)
    assert (doc["finished"], doc["of"]) == (1, 2) and doc["runs"][1]["state"] == "missing"
    c = doc["runs"][0]["curve"]
    assert c["step_final"] == 3 and c["step_first_qat"] == 3 and c["step_last_float"] == 2
    assert "| data_h2_s4p_c6n4_seed1 |" in RS.markdown(doc)


def test_the_sweep_scripts_parse():
    for f in ("sweep.sh", "probe.sh"):
        p = subprocess.run(["bash", "-n", str(REPRO / f)], capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr

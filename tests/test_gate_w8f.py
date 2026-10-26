"""The head gate in W8F and F32 and the two-level head in W8F (runtime/vocr.c gate_tile / run_head2; vita-vn-ocr
#126, model B for the PSP, where W8I is the slowest mode). On generated data only: a random-init model, a random
gate, its head permuted and random escapes in a VOC2 trailer.

The C runtime must equal the exact twins bit for bit: the gate's scores (train/headgate.gate_scores, the same
int16 arithmetic in every mode) on the mode's own head input; a gated run equal to the ungated run of the same mode
with the skipped frames read as the blank (headgate.simulate); the W8F two-level head equal to
train/head2level.twolevel_exact(mode="w8f"); every escape firing equal to the full W8F head; and the arena equal to
train/export.plan_arena. No development, test or private data is read here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "runtime", ROOT / "train", ROOT / "runtime" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import head2level as H2  # noqa: E402
import headgate as HG  # noqa: E402

F32 = np.float32
VOCAB = [chr(0x4E00 + i) for i in range(205)] + list("「」…、。")   # 210 classes + the blank
TILINGS = [(0, 0), (1, 1), (2, 3), (3, 16), (7, 5), (128, 128)]


def _model():
    import torch
    from torch import nn

    from model import TinyVNOCR
    torch.manual_seed(11)
    m = TinyVNOCR(len(VOCAB) + 1, width=0.25).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.uniform_(0.5, 2.0)
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, 0, (2.0 / mod.weight[0].numel()) ** 0.5)
        m.head.bias.normal_(0, 1.0)
        m.head.bias[0] += 2.0                        # blank-heavy, like a trained head
    return m


@pytest.fixture(scope="module")
def blobs():
    """(int8 blob, float blob, the int8 blob with its head permuted, the permutation)."""
    import export
    m = _model()
    ops = export.model_ops(m, 32)
    w8, _ = export.build_blob(ops, VOCAB, 32, True, 1024, b"")
    f32, _ = export.build_blob(ops, VOCAB, 32, False, 1024, b"")
    rng = np.random.default_rng(1)
    perm = np.concatenate([[0], 1 + rng.permutation(len(VOCAB))])
    return w8, f32, H2.permute_blob(w8, perm), perm


def crops(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h, w = int(rng.integers(14, 40)), int(rng.integers(20, 420))
        g = rng.integers(0, 255, size=(h, w)).astype(np.uint8)
        g[:, rng.integers(0, w, size=w // 3)] //= 3
        out.append(g)
    return out


def head_inputs(blob, mode, imgs):
    from pyvocr import Vocr
    rt = Vocr(blob, mode=mode, max_width=1024)
    xs = []
    for g in imgs:
        rt.run(g, head_in=True)
        xs.append(rt.last_head_in.copy())
    return np.concatenate(xs)


def gate_for(blob, mode, imgs, seed=3, q=0.5):
    """A random gate on the head input whose threshold is the q-quantile of these crops' scores (in this mode)."""
    rng = np.random.default_rng(seed)
    K = HG.head_op(blob)["k"]
    gp = HG.quantise(rng.normal(0, 0.3, size=(16, K)), rng.normal(0, 0.5, 16), rng.normal(0, 0.5, 16), 0.0)
    t = F32(np.quantile(HG.gate_scores(gp, head_inputs(blob, mode, imgs)), q))
    return HG.append_gate(blob, gp, t), gp, t


def random_escape(rng, K, H=16, t=0.0):
    import export
    q, scale = export.quantize_rows(rng.normal(0, 0.3, size=(H, K)))
    return dict(q=q, scale=scale.astype(F32), bias=rng.normal(0, 0.5, H).astype(F32),
                wl1=rng.normal(0, 0.05, H).astype(F32), wb0=rng.normal(0, 0.05, H).astype(F32),
                w2=rng.normal(0, 0.5, H).astype(F32), b2=F32(rng.normal(0, 0.5)), t=F32(t))


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == np.float32:
        return a.shape == b.shape and np.array_equal(a.view(np.uint32), np.asarray(b, F32).view(np.uint32))
    return np.array_equal(a, b)


@pytest.mark.parametrize("mode", ["w8f", "f32"])
def test_gate_scores_equal_twin(blobs, mode):
    from pyvocr import Vocr
    base = blobs[0] if mode == "w8f" else blobs[1]
    imgs = crops(12, 2)
    gated, gp, t = gate_for(base, mode, imgs)
    rt = Vocr(gated, mode=mode, max_width=1024, head_gate="score")
    n = 0
    for g in crops(20, 5):
        o = rt.run_topk(g, 4, head_in=True, gate_score=True)
        assert same(o["gate_score"], HG.gate_scores(gp, o["head_in"])) and o["gated"] == 0
        n += o["frames"]
    assert n > 300


@pytest.mark.parametrize("mode", ["w8f", "f32"])
@pytest.mark.parametrize("tile,head", TILINGS)
def test_gate_on_equals_simulation(blobs, mode, tile, head):
    """head_gate ON in W8F / F32: the ungated run of the same mode with the skipped frames as the blank, its blank
    logit computed as that mode's head computes class 0; the text-only path the same argmax; logits asked for or
    ungated: nothing skipped."""
    from pyvocr import Vocr
    base = blobs[0] if mode == "w8f" else blobs[1]
    gated, gp, t = gate_for(base, mode, crops(12, 2))
    full = Vocr(base, mode=mode, max_width=1024)
    on = Vocr(gated, mode=mode, max_width=1024, head_gate="on", tile=tile, head=head)
    vocab = [full.class_text(c) for c in range(full.n_classes)]
    skipped = frames = 0
    for i, g in enumerate(crops(16, 7)):
        k = (1, 3, 8, 16)[i % 4]
        x = full.run_topk(g, k, head_in=True)
        skip = HG.gate_scores(gp, x["head_in"]) > t
        sim = HG.simulate(x, skip, vocab)
        y = on.run_topk(g, k, gate_score=True)
        assert y["text"] == sim["text"] and y["gated"] == sim["gated"] == int(skip.sum())
        for key in ("best", "topc", "topv", "blank"):
            assert same(y[key], sim[key]), key
        assert F32(y["conf_min"]) == F32(sim["conf_min"]) and F32(y["conf_mean"]) == F32(sim["conf_mean"])
        assert on.run(g) == sim["text"] and np.array_equal(on.last_best, sim["best"])
        txt, lg = on.run(g, logits=True)
        assert txt == x["text"] and on.last_gated == 0
        assert same(lg[:, 0], x["blank"])                        # the blank logits a skipped frame reads
        u = on.run_topk(g, k, ungated=True)
        assert u["text"] == x["text"] and u["gated"] == 0 and same(u["topv"], x["topv"])
        skipped += int(skip.sum())
        frames += len(skip)
    assert 0.2 < skipped / frames < 0.8


def test_gate_off_is_the_base(blobs):
    from pyvocr import Vocr
    for base, mode in ((blobs[0], "w8f"), (blobs[1], "f32")):
        gated, _, _ = gate_for(base, mode, crops(6, 2))
        a, b = Vocr(base, mode=mode, max_width=1024), Vocr(gated, mode=mode, max_width=1024)
        assert a.arena_bytes == b.arena_bytes
        for g in crops(8, 9):
            x, y = a.run_topk(g, 8), b.run_topk(g, 8)
            assert x["text"] == y["text"] and y["gated"] == 0
            assert same(x["topv"], y["topv"]) and same(x["topc"], y["topc"])


@pytest.mark.parametrize("l1_blocks,t", [(3, -1.0), (3, 1e9), (10, 0.5), (1, -1e9), (5, 0.0), (2, -0.5)])
def test_two_level_w8f_equals_twin(blobs, l1_blocks, t):
    from pyvocr import Vocr

    import export
    pblob = blobs[2]
    rng = np.random.default_rng(l1_blocks * 13)
    head = H2.head_arrays(pblob)
    esc = random_escape(rng, head["K"], t=t)
    blob = H2.attach(pblob, H2.write_trailer(pblob, l1_blocks, esc))
    two = H2.read_trailer(blob)
    full = Vocr(pblob, mode="w8f")
    rt = Vocr(blob, mode="w8f", two_level=1)
    assert rt.arena_bytes == export.plan_arena(export.read_blob(blob), 1024, 0, 0, export.MODE_W8F, two_level=two)
    fired = frames = 0
    for g in crops(20, l1_blocks + 1):
        full.run(g, head_in=True)
        x = full.last_head_in
        tw = H2.twolevel_exact(x, head, two, topk=8, mode="w8f")
        out = rt.run_topk(g, 8, esc=True)
        assert same(out["esc"], tw["score"]) and out["escaped"] == int(tw["fire"].sum())
        for key in ("best", "topc", "topv", "blank"):
            assert same(out[key], tw[key]), key
        greedy = "".join(rt.class_text(int(c)) for i, c in enumerate(tw["best"])
                         if c and (i == 0 or c != tw["best"][i - 1]))
        assert out["text"] == greedy and rt.text(g) == greedy
        o0 = full.run_topk(g, 8)                                  # a fired frame's top-K is the full head's
        f = tw["fire"]
        assert same(o0["topc"][f], out["topc"][f]) and same(o0["topv"][f], out["topv"][f])
        fired += int(f.sum())
        frames += len(x)
    if t <= -1e8:
        assert fired == frames
    if t >= 1e8:
        assert fired == 0


def test_two_level_w8f_all_fired_is_today(blobs):
    from pyvocr import Vocr
    pblob = blobs[2]
    rng = np.random.default_rng(9)
    blob = H2.attach(pblob, H2.write_trailer(pblob, 4, random_escape(rng, H2.head_arrays(pblob)["K"], t=-np.inf)))
    full, rt = Vocr(pblob, mode="w8f"), Vocr(blob, mode="w8f", two_level=1)
    for g in crops(10, 11):
        a, b = full.run_topk(g, 8), rt.run_topk(g, 8)
        for k in ("text", "topv", "topc", "blank", "conf_min", "conf_mean"):
            assert same(a[k], b[k]), k
        t0, l0 = full.run(g, logits=True)
        t1, l1 = rt.run(g, logits=True)                           # logits asked for: the full head, every row
        assert t0 == t1 and same(l0, l1) and rt.last_escaped == 0


def test_two_level_w8i_unchanged_by_w8f(blobs):
    """The same trailer in W8I still equals the W8I twin (the W8F port leaves W8I's arithmetic as it was)."""
    from pyvocr import Vocr
    pblob = blobs[2]
    rng = np.random.default_rng(21)
    head = H2.head_arrays(pblob)
    esc = random_escape(rng, head["K"], t=0.0)
    blob = H2.attach(pblob, H2.write_trailer(pblob, 3, esc))
    two = H2.read_trailer(blob)
    full, rt = Vocr(pblob, mode="w8i"), Vocr(blob, mode="w8i", two_level=1)
    for g in crops(8, 12):
        full.run(g, head_in=True)
        tw = H2.twolevel_exact(full.last_head_in, head, two, topk=8)
        out = rt.run_topk(g, 8, esc=True)
        assert same(out["esc"], tw["score"]) and same(out["topv"], tw["topv"]) and same(out["best"], tw["best"])


@pytest.mark.parametrize("tile,head", [(0, 0), (1, 3), (2, 1), (8, 5), (4, 64)])
def test_gate_and_two_level_w8f(blobs, tile, head):
    """Both on in W8F: a skipped frame reads as the blank (headgate.simulate of the W8F two-level run), the escape
    runs on the kept frames only."""
    from pyvocr import Vocr
    pblob = blobs[2]
    rng = np.random.default_rng(31 + tile)
    head_ = H2.head_arrays(pblob)
    K, l1b = head_["K"], 4
    gp = HG.quantise(rng.normal(0, 0.3, size=(16, K)), rng.normal(0, 0.5, 16), rng.normal(0, 0.5, 16), 0.0)
    esc = random_escape(rng, K)
    x = head_inputs(pblob, "w8f", crops(12, 40 + tile))
    gt = F32(np.median(HG.gate_scores(gp, x)))
    q, sx = H2.quant16_exact(x)
    lg1 = H2.mm_w8f_exact(x, head_["q"][:8 * l1b], head_["scale"][:8 * l1b], head_["bias"][:8 * l1b])
    l1, b0 = H2.level1_inputs(lg1, 8 * l1b)
    esc["t"] = F32(np.quantile(H2.escape_score(q, sx, l1, b0, esc), 0.7))
    blob = H2.attach(HG.append_gate(pblob, gp, gt), H2.write_trailer(pblob, l1b, esc))
    two = Vocr(blob, mode="w8f", tile=tile, head=head, two_level=1)
    both = Vocr(blob, mode="w8f", tile=tile, head=head, two_level=1, head_gate="on")
    vocab = [both.class_text(c) for c in range(both.n_classes)]
    skipped = escaped = 0
    for g in crops(12, 40 + tile):
        o2 = two.run_topk(g, 8, head_in=True, esc=True)
        skip = HG.gate_scores(gp, o2["head_in"]) > gt
        want = HG.simulate(o2, skip, vocab)
        ob = both.run_topk(g, 8, esc=True)
        for k in ("text", "topv", "topc", "blank", "best", "conf_min", "conf_mean"):
            assert same(want[k], ob[k]), k
        assert ob["gated"] == int(skip.sum())
        assert np.array_equal(np.isnan(ob["esc"]), skip) and same(ob["esc"][~skip], o2["esc"][~skip])
        assert both.text(g) == want["text"]
        skipped += int(skip.sum())
        escaped += ob["escaped"]
    assert skipped and escaped


def test_arena_mirror(blobs):
    from pyvocr import Vocr

    import export
    w8, f32, pblob, _ = blobs
    rng = np.random.default_rng(4)
    gw8, _, _ = gate_for(w8, "w8f", crops(4, 1))
    gf32, _, _ = gate_for(f32, "f32", crops(4, 1))
    esc = random_escape(rng, H2.head_arrays(pblob)["K"])
    K = H2.head_arrays(pblob)["K"]
    gp = HG.quantise(rng.normal(0, 0.3, size=(16, K)), rng.normal(0, 0.5, 16), rng.normal(0, 0.5, 16), 0.0)
    both = H2.attach(HG.append_gate(pblob, gp, F32(0)), H2.write_trailer(pblob, 3, esc))
    cases = [(gw8, "w8f", None), (gf32, "f32", None), (both, "w8f", 1), (both, "w8i", 1)]
    for blob, mode, tl in cases:
        b = export.read_blob(blob)
        two = H2.read_trailer(blob) if tl else None
        m = {"f32": export.MODE_F32, "w8f": export.MODE_W8F, "w8i": export.MODE_W8I}[mode]
        for tile, head, mw in ((0, 0, 0), (1, 1, 0), (2, 3, 256), (5, 7, 640), (16, 16, 0), (128, 128, 1024)):
            for hg in (0, 1, 2):
                rt = Vocr(blob, mode=mode, tile=tile, head=head, max_width=mw, head_gate=hg, two_level=tl or 0)
                assert rt.arena_bytes == export.plan_arena(b, mw, tile, head, m, hg, two), (mode, tile, head, mw, hg)
    with pytest.raises(ValueError):
        export.plan_arena(export.read_blob(f32), 0, 0, 0, 1, 1)       # no gate in the blob

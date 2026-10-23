"""The two-level output head (runtime/vocr.c run_head2, its twin train/head2level.py; vita-vn-ocr #117 item 2) on
generated data only: a random-init model with a 210-class vocab, its head permuted, random escapes in a
VOC2 trailer. The C runtime must equal the exact twin bit for bit (escape scores, fired frames, argmax, top-K, the
greedy text), the full logits and two_level = 0 must be today's head, and broken trailers must be refused. No
development, test or private data is read here.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "runtime", ROOT / "train", ROOT / "runtime" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import head2level as H2  # noqa: E402

F32 = np.float32
VOCAB = [chr(0x4E00 + i) for i in range(205)] + list("「」…、。")   # 210 classes + the blank


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    import torch
    from torch import nn

    import export
    from model import TinyVNOCR
    torch.manual_seed(5)
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
    blob, _ = export.build_blob(export.model_ops(m, 32), VOCAB, 32, True, 1024, b"")
    rng = np.random.default_rng(1)
    perm = np.concatenate([[0], 1 + rng.permutation(len(VOCAB))])
    return blob, H2.permute_blob(blob, perm), perm


def crops(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h, w = int(rng.integers(20, 40)), int(rng.integers(30, 400))
        g = rng.integers(0, 255, size=(h, w)).astype(np.uint8)
        g[:, : w // 3] //= 3
        out.append(g)
    return out


def random_escape(rng, K, H=16, t=0.0):
    w1 = rng.normal(0, 0.3, size=(H, K + 2))
    q, scale = __import__("export").quantize_rows(w1[:, :K])
    return dict(q=q, scale=scale.astype(F32), bias=rng.normal(0, 0.5, H).astype(F32),
                wl1=rng.normal(0, 0.05, H).astype(F32), wb0=rng.normal(0, 0.05, H).astype(F32),
                w2=rng.normal(0, 0.5, H).astype(F32), b2=F32(rng.normal(0, 0.5)), t=F32(t))


def random_r48(rng, K, R, RK=16):
    import export
    pq, ps = export.quantize_rows(rng.normal(0, 0.3, size=(RK, K)))
    rq, rs = export.quantize_rows(rng.normal(0, 0.3, size=(R, RK)))
    return dict(p_q=pq, p_scale=ps.astype(F32), p_bias=rng.normal(0, 0.1, RK).astype(F32), r_q=rq,
                r_scale=rs.astype(F32), r_bias=rng.normal(0, 0.5, R).astype(F32))


def make(base_blob, l1_blocks, esc, rare=None):
    return H2.attach(base_blob, H2.write_trailer(base_blob, l1_blocks, esc, rare))


def test_permuted_blob_same_logits(base):
    from pyvocr import Vocr
    blob, pblob, perm = base
    r0, r1 = Vocr(blob, mode="w8i"), Vocr(pblob, mode="w8i")
    for g in crops(12, 3):
        t0, l0 = r0.run(g, logits=True)
        t1, l1 = r1.run(g, logits=True)
        assert np.array_equal(l0[:, perm], l1)
        assert r0.class_text(int(perm[5])) == r1.class_text(5)


@pytest.mark.parametrize("l1_blocks,rk,t", [(3, 0, -1.0), (3, 0, 1e9), (10, 0, 0.5), (1, 0, -1e9), (5, 0, 0.0),
                                            (2, 0, -0.5)])
def test_c_equals_twin(base, l1_blocks, rk, t):
    from pyvocr import Vocr
    _, pblob, _ = base
    rng = np.random.default_rng(l1_blocks * 7 + rk)
    head = H2.head_arrays(pblob)
    K = head["K"]
    esc = random_escape(rng, K, t=t)
    rare = random_r48(rng, K, head["P"] - 8 * l1_blocks, rk) if rk else None
    blob = make(pblob, l1_blocks, esc, rare)
    two = H2.read_trailer(blob)
    assert two["l1_blocks"] == l1_blocks and two["rare_k"] == rk
    full = Vocr(pblob, mode="w8i")
    rt = Vocr(blob, mode="w8i", two_level=1)
    import export
    b = export.read_blob(blob)
    assert rt.arena_bytes == export.plan_arena(b, 1024, 0, 0, 3, two_level=two)
    fired = frames = 0
    for g in crops(25, l1_blocks):
        full.run(g, head_in=True)
        x = full.last_head_in
        tw = H2.twolevel_exact(x, head, two, topk=8)
        out = rt.run_topk(g, 8, esc=True)
        assert np.array_equal(out["esc"], tw["score"])
        assert out["escaped"] == int(tw["fire"].sum())
        assert np.array_equal(out["best"], tw["best"])
        assert np.array_equal(out["topc"], tw["topc"]) and np.array_equal(out["topv"], tw["topv"])
        assert np.array_equal(out["blank"], tw["blank"])
        greedy = "".join(rt.class_text(int(c)) for i, c in enumerate(tw["best"])
                         if c and (i == 0 or c != tw["best"][i - 1]))
        assert out["text"] == greedy
        assert rt.text(g) == greedy                  # the no-top-K path: the same argmax
        fired += int(tw["fire"].sum())
        frames += len(x)
        if rk == 0:                                  # a fired frame's top-K is today's
            o0 = full.run_topk(g, 8)
            f = tw["fire"]
            assert np.array_equal(o0["topc"][f], out["topc"][f]) and np.array_equal(o0["topv"][f], out["topv"][f])
    if t <= -1e8:
        assert fired == frames
    if t >= 1e8:
        assert fired == 0


def test_all_fired_is_today(base):
    """With every frame fired (t = -inf) the two-level head is the full head: same text, top-K, confidence."""
    from pyvocr import Vocr
    _, pblob, _ = base
    rng = np.random.default_rng(9)
    blob = make(pblob, 4, random_escape(rng, H2.head_arrays(pblob)["K"], t=-np.inf))
    full, rt = Vocr(pblob, mode="w8i"), Vocr(blob, mode="w8i", two_level=1)
    for g in crops(10, 11):
        a, b = full.run_topk(g, 8), rt.run_topk(g, 8)
        for k in ("text", "topv", "topc", "blank", "conf_min", "conf_mean"):
            assert np.array_equal(a[k], b[k]), k


def test_off_and_full_logits_are_today(base):
    from pyvocr import Vocr
    _, pblob, _ = base
    rng = np.random.default_rng(4)
    blob = make(pblob, 2, random_escape(rng, H2.head_arrays(pblob)["K"], t=1e9))
    full, off, on = Vocr(pblob, mode="w8i"), Vocr(blob, mode="w8i"), Vocr(blob, mode="w8i", two_level=1)
    assert off.arena_bytes == full.arena_bytes < on.arena_bytes
    for g in crops(6, 2):
        assert off.text(g) == full.text(g)
        t0, l0 = full.run(g, logits=True)
        t1, l1 = on.run(g, logits=True)              # logits asked for: every row of every frame
        assert t0 == t1 and np.array_equal(l0, l1) and on.last_escaped == 0


def test_mode_and_trailer_checks(base):
    from pyvocr import Vocr, VocrError
    _, pblob, _ = base
    rng = np.random.default_rng(6)
    esc = random_escape(rng, H2.head_arrays(pblob)["K"])
    blob = make(pblob, 3, esc)
    with pytest.raises((VocrError, ValueError)):     # vocr_arena_bytes() == 0: invalid options
        Vocr(blob, mode="w8f", two_level=1)          # W8I only
    with pytest.raises((VocrError, ValueError)):
        Vocr(pblob, mode="w8i", two_level=1)         # no trailer
    K = H2.head_arrays(pblob)["K"]                   # the trial's 48-wide rare block: the runtime reads none
    r48 = make(pblob, 3, esc, random_r48(rng, K, H2.head_arrays(pblob)["P"] - 24, 16))
    assert H2.read_trailer(r48)["rare_k"] == 16
    with pytest.raises(VocrError):
        Vocr(r48, mode="w8i")
    with pytest.raises(ValueError):
        __import__("export").plan_arena(__import__("export").read_blob(r48), 1024, 0, 0, 3,
                                        two_level=H2.read_trailer(r48))
    bad = bytearray(blob)
    bad[-1] ^= 1                                     # trailer CRC
    with pytest.raises(VocrError):
        Vocr(bytes(bad), mode="w8i")
    Vocr(bytes(bad), mode="w8i", check_crc=False)    # structure only: still loads
    off = H2._al16(len(pblob))
    for field, val in ((24, 0), (24, 200), (26, 12), (28, 4), (16, 7)):
        bad = bytearray(blob)
        struct.pack_into("<H" if field != 16 else "<I", bad, off + field, val)
        struct.pack_into("<I", bad, off + 12, __import__("zlib").crc32(bytes(bad[off + 16:])) & 0xFFFFFFFF)
        with pytest.raises(VocrError):
            Vocr(bytes(bad), mode="w8i")
    for field in (32, 36):                           # a NaN threshold or layer-2 bias
        bad = bytearray(blob)
        struct.pack_into("<f", bad, off + field, float("nan"))
        struct.pack_into("<I", bad, off + 12, __import__("zlib").crc32(bytes(bad[off + 16:])) & 0xFFFFFFFF)
        with pytest.raises(VocrError):
            Vocr(bytes(bad), mode="w8i")


def test_threshold_rule():
    """At most floor(0.001 n) positives score <= t (no denormal in play: another test may set flush-to-zero)."""
    s = np.array([5, 1, 3, 2, 4, 9, 7, 6, 8, 0.5] * 100, F32)
    t, m = H2.threshold_for(s, 0.999)
    assert m == 1 and (s <= t).sum() <= 1 and t < 0.5 and (s > t).sum() == len(s)
    v = np.arange(1, 20001, dtype=F32)
    t, m = H2.threshold_for(v, 0.999)
    assert m == 20 and (v <= t).sum() == 20
    t, m = H2.threshold_for(v, 0.9999)
    assert m == 2 and (v <= t).sum() == 2


@pytest.mark.parametrize("rk,tile,head", [(0, 0, 0), (0, 1, 3), (0, 2, 1), (0, 8, 5)])
def test_gate_and_two_level(base, rk, tile, head):
    """head_gate ON with two_level: a frame the gate skips reads as the blank (headgate.simulate of the two-level
    run), every other frame is the two-level head's, at several tilings; the escape runs on those frames only."""
    from pyvocr import Vocr

    import headgate as HG
    _, pblob, _ = base
    rng = np.random.default_rng(31 + rk + tile)
    head_ = H2.head_arrays(pblob)
    K = head_["K"]
    gp = HG.quantise(rng.normal(0, 0.3, size=(16, K)), rng.normal(0, 0.5, 16), rng.normal(0, 0.5, 16), 0.0)
    l1b = 4
    rare = random_r48(rng, K, head_["P"] - 8 * l1b, rk) if rk else None
    esc = random_escape(rng, K)
    # thresholds at the median gate score and the 70th percentile escape score of these crops' frames
    x = np.concatenate([(lambda r: (r.run(g, head_in=True), r.last_head_in)[1])(Vocr(pblob, mode="w8i"))
                        for g in crops(15, 40 + tile)])
    gt = np.float32(np.median(HG.gate_scores(gp, x)))
    q, sx = H2.quant16_exact(x)
    lg1 = H2.mm_exact(q, sx, head_["q"][:8 * l1b], head_["scale"][:8 * l1b], head_["bias"][:8 * l1b])
    l1, b0 = H2.level1_inputs(lg1, 8 * l1b)
    esc["t"] = np.float32(np.quantile(H2.escape_score(q, sx, l1, b0, esc), 0.7))
    gated = HG.append_gate(pblob, gp, gt)
    blob = H2.attach(gated, H2.write_trailer(pblob, l1b, esc, rare))
    assert HG.read_gate(blob) is not None and H2.read_trailer(blob)["l1_blocks"] == l1b
    two = Vocr(blob, mode="w8i", tile=tile, head=head, two_level=1)
    both = Vocr(blob, mode="w8i", tile=tile, head=head, two_level=1, head_gate="on")
    vocab = [both.class_text(c) for c in range(both.n_classes)]
    skipped = escaped = 0
    for g in crops(15, 40 + tile):
        o2 = two.run_topk(g, 8, head_in=True, esc=True)
        skip = HG.gate_scores(gp, o2["head_in"]) > gt
        want = HG.simulate(o2, skip, vocab)
        ob = both.run_topk(g, 8, esc=True)
        for k in ("text", "topv", "topc", "blank", "best", "conf_min", "conf_mean"):
            assert np.array_equal(np.asarray(want[k]), np.asarray(ob[k])), k
        assert ob["gated"] == int(skip.sum())
        assert np.array_equal(np.isnan(ob["esc"]), skip)              # no escape on a skipped frame
        assert np.array_equal(ob["esc"][~skip], o2["esc"][~skip])
        assert ob["escaped"] == int((o2["esc"][~skip] > esc["t"]).sum())
        u = both.run_topk(g, 8, ungated=True)                         # ungated: the full head, no skipping
        full = Vocr(pblob, mode="w8i").run_topk(g, 8)
        assert u["text"] == full["text"] and np.array_equal(u["topc"], full["topc"]) and u["gated"] == 0
        skipped += int(skip.sum())
        escaped += ob["escaped"]
    assert skipped and escaped

"""VOCR6 stage 3's light global-attention block (docs/vocr6-attn.md) end to end: the model (gate 0 = the model without
it, bit for bit; padding masked out of the keys), checkpoints (train2 --attn, old checkpoints, from_checkpoint,
export.load_checkpoint), the blob's attention op (export, read_blob, plan_arena), the C runtime against the bit-exact
numpy twin (runtime/tests/vocr_ref.py) and PyTorch, and the head gate and two-level head on an attention blob.
CPU only; random-init models, no data."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in ("train", "runtime", "runtime/tests", "synth", "eval"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

import model as M  # noqa: E402
from model import GlobalAttn, TinyVNOCR, from_checkpoint, load_state  # noqa: E402

torch.set_num_threads(1)
VOCAB = [chr(0x3042 + i) for i in range(30)]


def _bn_random(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_var.copy_(torch.rand(mod.running_var.shape, generator=g) * 1.5 + 0.5)
                mod.running_mean.copy_(torch.rand(mod.running_mean.shape, generator=g) * 0.4 - 0.2)
    return m


# ----------------------------------------------------------------------------- the model

def test_default_model_has_no_attention():
    m = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48)
    assert m.attn is None and "attn" not in m.arch
    assert not any(k.startswith("attn.") for k in m.state_dict())
    assert not any(isinstance(x, GlobalAttn) for x in m.modules())


def test_attention_is_built_after_the_head_so_the_other_modules_initialise_alike():
    torch.manual_seed(5)
    a = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48).state_dict()
    torch.manual_seed(5)
    b = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48, attn=1, attn_dim=48, attn_heads=2).state_dict()
    assert list(b)[:len(a)] == list(a) and all(torch.equal(a[k], b[k]) for k in a)
    assert [k for k in b if k not in a] == ["attn.gate", "attn.q.weight", "attn.q.bias", "attn.k.weight",
                                            "attn.v.weight", "attn.v.bias", "attn.o.weight", "attn.o.bias"]
    assert float(b["attn.gate"]) == 0.0


def test_bad_attention_shapes_are_refused():
    for dim, heads in ((96, 5), (40, 2), (12, 1), (96, 0)):
        with pytest.raises(ValueError):
            GlobalAttn(96, dim, heads)
    with pytest.raises(ValueError):
        TinyVNOCR(5, attn=2)


def test_an_old_checkpoint_in_an_attention_model_gives_the_same_logits_and_gradients():
    """load_state of a state_dict without the block: gate 0, so the logits are the old model's bit for bit, in eval
    and in training (same dropout draw, masked batch), and so are the gradients of every shared parameter."""
    torch.manual_seed(0)
    old = _bn_random(TinyVNOCR(len(VOCAB) + 1, width=0.5)).eval()
    new = TinyVNOCR(len(VOCAB) + 1, width=0.5, attn=1)
    with torch.no_grad():
        new.attn.gate.fill_(0.5)                     # load_state resets it
    assert load_state(new, old.state_dict()) == "fresh-attn" and float(new.attn.gate) == 0.0
    new.eval()
    x = torch.rand(3, 1, 32, 200, generator=torch.Generator().manual_seed(1)) * 2 - 1
    with torch.no_grad():
        assert torch.equal(old(x), new(x))
    old.train()
    new.train()
    lengths = torch.tensor([25, 17, 9])
    torch.manual_seed(2)
    yo = old(x)
    torch.manual_seed(2)
    yn = new(x, lengths=lengths)
    assert torch.equal(yo, yn)
    yo.square().mean().backward()
    yn.square().mean().backward()
    pn = dict(new.named_parameters())
    for k, p in old.named_parameters():
        assert torch.equal(p.grad, pn[k].grad), k
    assert float(new.attn.gate.grad.abs()) > 0           # the gate itself learns from the first step
    # a strict load of a state_dict with the block, and a state_dict that does not fit
    assert load_state(new, new.state_dict()) == "loaded"
    bad = dict(old.state_dict())
    bad["extra.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        load_state(new, bad)


def test_padding_is_masked_out_of_the_keys():
    torch.manual_seed(0)
    a = GlobalAttn(48, 48, 2).eval()
    with torch.no_grad():
        a.gate.fill_(0.9)
        z1, z2 = torch.randn(1, 48, 1, 12), torch.randn(1, 48, 1, 20)
        zb = torch.cat([torch.cat([z1, torch.randn(1, 48, 1, 8) * 50], 3), z2], 0)
        out = a(zb, lengths=torch.tensor([12, 20]))
        assert torch.allclose(a(z1), out[:1, :, :, :12], atol=1e-5)
        assert torch.allclose(a(z2), out[1:], atol=1e-5)
        assert not torch.allclose(a(z1), a(zb)[:1, :, :, :12], atol=1e-2)   # unmasked, the padding leaks in


def test_from_checkpoint_rebuilds_the_block_and_old_checkpoints_have_none():
    torch.manual_seed(0)
    m = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48, attn=1, attn_dim=48, attn_heads=3).eval()
    with torch.no_grad():
        m.attn.gate.fill_(0.3)
    ck = {"model": m.state_dict(), "vocab": VOCAB, "width": 0.25, "bottleneck": 48, "height": 32,
          "attn": 1, "attn_dim": 48, "attn_heads": 3}
    m2, meta = from_checkpoint(ck)
    assert meta["attn"] == 1 and meta["attn_dim"] == 48 and meta["attn_heads"] == 3 and m2.attn.heads == 3
    x = torch.rand(1, 1, 32, 96)
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))
    old = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48)
    m3, meta3 = from_checkpoint({"model": old.state_dict(), "vocab": VOCAB, "width": 0.25, "bottleneck": 48})
    assert m3.attn is None and meta3["attn"] == 0
    with pytest.raises(RuntimeError):                    # a checkpoint that claims a block it does not have
        from_checkpoint({"model": old.state_dict(), "vocab": VOCAB, "width": 0.25, "bottleneck": 48, "attn": 1})


def test_qat_and_rep_cover_the_block():
    m = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48, attn=1, attn_dim=48)
    M.set_weight_quant(m, 8)
    assert all(c.wbits == 8 for c in (m.attn.q, m.attn.k, m.attn.v, m.attn.o))
    r = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48, attn=1, attn_dim=48, rep=1)
    plain = M.fuse_rep(r.eval())
    assert plain.attn is not None and plain.arch["attn"] == 1
    assert all(torch.equal(plain.state_dict()[k], v) for k, v in r.state_dict().items() if k.startswith("attn."))


# ----------------------------------------------------------------------------- train2

class Exited(Exception):
    pass


def _write_shard(path, crops, labels):
    np.savez(path, data=np.concatenate([c.ravel() for c in crops]).astype(np.uint8),
             offsets=np.cumsum([0] + [c.size for c in crops]).astype(np.int64),
             shapes=np.array([c.shape for c in crops], np.int32), labels=json.dumps(labels, ensure_ascii=False))


def test_train2_attn_from_an_old_checkpoint_resumes_and_records_the_block(tmp_path, monkeypatch):
    import train2
    vocab = ["あ", "い", "う"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (32, int(w)), dtype=np.uint8) for w in (64, 40, 96, 56) * 2]
    _write_shard(tmp_path / "s.npz", crops, ["あい", "う", "いう", "あ"] * 2)
    torch.manual_seed(0)
    old = TinyVNOCR(len(vocab) + 1, width=0.25, bottleneck=16)
    torch.save({"model": old.state_dict(), "vocab": vocab, "width": 0.25, "bottleneck": 16, "step": 7},
               tmp_path / "old.pt")

    def boom(code=0):
        raise Exited(code)

    monkeypatch.setattr(os, "_exit", boom)
    argv = ["--out", str(tmp_path / "runs"), "--src", f"a:1:{tmp_path / 's.npz'}", "--val", f"v:{tmp_path / 's.npz'}",
            "--batch", "4", "--warmup", "1", "--threads", "1", "--vocab", str(tmp_path / "vocab.txt"),
            "--width", "0.25", "--bottleneck", "16", "--eval-every", "2", "--log-every", "1"]
    attn = ["--attn", "1", "--attn-dim", "16", "--attn-heads", "2"]
    with pytest.raises(Exited):
        train2.main(["--run", "r", "--init", str(tmp_path / "old.pt"), *attn, *argv, "--steps", "2"])
    with pytest.raises(Exited):                          # resume to step 4
        train2.main(["--run", "r", "--init", str(tmp_path / "old.pt"), *attn, *argv, "--steps", "4"])
    ck = torch.load(tmp_path / "runs/r/ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4 and ck["attn"] == 1 and ck["attn_dim"] == 16 and ck["attn_heads"] == 2
    m, meta = from_checkpoint(ck)
    assert m.attn is not None and float(m.attn.gate) != 0.0     # the gate moved
    with pytest.raises(ValueError):                      # resume without --attn
        train2.main(["--run", "r", *argv, "--steps", "6"])
    with pytest.raises(ValueError):                      # an attention checkpoint into a run without the block
        train2.main(["--run", "r0", "--init", str(tmp_path / "runs/r/ckpt_last.pt"), *argv, "--steps", "1"])
    with pytest.raises(Exited):                          # --attn 0 from the old checkpoint: no attention keys
        train2.main(["--run", "p", "--init", str(tmp_path / "old.pt"), *argv, "--steps", "2"])
    ck0 = torch.load(tmp_path / "runs/p/ckpt_last.pt", map_location="cpu", weights_only=False)
    assert "attn" not in ck0 and not any(k.startswith("attn.") for k in ck0["model"])


# ----------------------------------------------------------------------------- export and the C runtime

def attn_model(seed=3, gate=0.7, dim=48, heads=2, vocab=VOCAB, blank_bias=None):
    """A random-init attention model (width 0.25, bottleneck 96) with random BN, a peaked softmax (Q, K x 3) and,
    with blank_bias, a blank-heavy head (the blank wins about that share of frames, as a trained model's do)."""
    from torch import nn
    torch.manual_seed(seed)
    m = TinyVNOCR(len(vocab) + 1, width=0.25, attn=1, attn_dim=dim, attn_heads=heads).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.uniform_(0.5, 2.0)
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, 0, (2.0 / mod.weight[0].numel()) ** 0.5)
        m.attn.q.weight.mul_(3.0)
        m.attn.k.weight.mul_(3.0)
        m.attn.gate.fill_(gate)
        if blank_bias is not None:
            lg = m(torch.rand(8, 1, 32, 256, generator=torch.Generator().manual_seed(8)) * 2 - 1)
            m.head.bias[0] += float(torch.quantile((lg[:, :, 1:].max(2).values - lg[:, :, 0]).flatten(), blank_bias))
    return m


def blob_of(m, int8, vocab=VOCAB):
    import export
    return export.build_blob(export.model_ops(m, 32), vocab, 32, int8, 1024, b"")[0]


def crops(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h, w = int(rng.integers(14, 48)), int(rng.integers(6, 700))
        g = rng.integers(0, 256, (h, w), dtype=np.uint8)
        g[:, rng.integers(0, w, size=max(1, w // 3))] //= 3
        out.append(g)
    return out


def x_of(g, H=32):
    import cv2
    h, w = g.shape
    nw = max(8, min(1024, int(round(w * H / h))))
    r = cv2.resize(g, (nw, H), interpolation=cv2.INTER_LINEAR)
    W = (nw + 7) // 8 * 8
    return np.concatenate([r, np.repeat(r[:, -1:], W - nw, 1)], 1).astype(np.float32) / 127.5 - 1.0


MODES = {"f32": False, "w8f": True, "w8i": True}


def test_export_folds_the_scale_and_the_gate():
    import export
    m = attn_model(gate=-0.6, dim=48, heads=3)
    b = export.read_blob(blob_of(m, False))
    a = [op for op in b["ops"] if op["type"] == export.T_ATTN]
    assert len(a) == 1 and b["ops"][-2] is a[0] and b["ops"][-1]["type"] == export.T_HEAD
    a = a[0]
    assert (a["heads"], a["attn_dim"], a["cin"], a["cout"], a["k"], a["cout_pad"]) == (3, 48, 96, 96, 96, 144)
    s = 1 / np.sqrt(16)
    wq, wk, wv = (getattr(m.attn, n).weight.detach().double().numpy()[:, :, 0, 0] for n in "qkv")
    np.testing.assert_array_equal(a["w"], np.concatenate([wq * s, wk, wv]).astype(np.float32))
    np.testing.assert_array_equal(a["bias"][48:96], 0)                     # K: no bias
    np.testing.assert_array_equal(a["bias"][:48], (m.attn.q.bias.detach().double().numpy() * s).astype(np.float32))
    wo, g = m.attn.o.weight.detach().double().numpy()[:, :, 0, 0], float(m.attn.gate)     # g: float32(-0.6)
    np.testing.assert_array_equal(a["o_w"], (wo * g).astype(np.float32))
    np.testing.assert_array_equal(a["o_bias"], (m.attn.o.bias.detach().double().numpy() * g).astype(np.float32))
    plain = export.read_blob(blob_of(_without_attn(m), False))
    assert b["params"] == plain["params"] + 144 * 96 + 144 + 96 * 48 + 96
    macs, plain_macs = export.macs_per_frame(b)[0], export.macs_per_frame(plain)[0]
    assert macs == plain_macs + 4 * 48 * 96 and export.macs_per_frame(b, 80)[0] == macs + 2 * 80 * 48
    q = export.read_blob(blob_of(m, True))
    a8 = [op for op in q["ops"] if op["type"] == export.T_ATTN][0]
    assert a8["int8"] and a8["w"].dtype == np.int8 and a8["o_w"].dtype == np.int8 and len(a8["o_scale"]) == 96
    extra = 64 + (144 * 96 + 8 * 144) + (96 * 48 + 8 * 96)            # the record, QKV and O with scales and biases
    assert len(blob_of(m, True)) == len(blob_of(_without_attn(m), True)) + extra


def _without_attn(m):
    """The same model without its attention block (every other weight shared)."""
    p = TinyVNOCR(m.head.out_channels, **{k: v for k, v in m.arch.items() if not k.startswith("attn")}).eval()
    p.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("attn.")})
    return p


def test_gate_zero_reads_as_the_model_without_the_block():
    """The export of a block at gate 0 (a fine-tune's start) gives the C logits of the model without it, bit for bit,
    in every mode, though its blob carries the op."""
    from pyvocr import Vocr
    m = attn_model(gate=0.0)
    p = _without_attn(m)
    for mode, int8 in MODES.items():
        a, b = Vocr(blob_of(m, int8), mode=mode), Vocr(blob_of(p, int8), mode=mode)
        assert a.arena_bytes > b.arena_bytes
        for g in crops(12, 1):
            ta, la = a.run(g, logits=True)
            tb, lb = b.run(g, logits=True)
            assert ta == tb and np.array_equal(la.view(np.uint32), lb.view(np.uint32)), mode


@pytest.mark.parametrize("dim,heads", [(48, 2), (96, 2), (32, 4)])
def test_c_equals_the_twin_and_pytorch(dim, heads):
    import vocr_ref
    from pyvocr import Vocr

    import export
    m = attn_model(dim=dim, heads=heads)
    for mode, int8 in MODES.items():
        data = blob_of(m, int8)
        b = export.read_blob(data)
        rts = [Vocr(data, mode=mode, tile=t, head=h) for t, h in ((0, 0), (1, 1), (3, 7), (16, 2), (128, 128))]
        for g in crops(8, dim + heads):
            x = x_of(g)
            exact = vocr_ref.forward(b, x, mode)
            outs = [rt.run(g, logits=True) for rt in rts]
            for text, lg in outs:
                assert np.array_equal(lg.view(np.uint32), exact.view(np.uint32)), mode
                assert text == outs[0][0]
            if mode == "f32":
                with torch.no_grad():
                    ref = m(torch.from_numpy(x)[None, None])[:, 0].numpy()
                assert np.abs(outs[0][1] - ref).max() <= 1e-3 * max(1.0, np.abs(ref).max())


def test_arena_mirror_and_the_line():
    """vocr_arena_bytes == export.plan_arena over widths, tiles, batches and modes; the line of z grows with
    max_width (t_max x 96 floats) and nothing else of it does."""
    from pyvocr import Vocr

    import export
    m = attn_model()
    for mode, int8 in MODES.items():
        data = blob_of(m, int8)
        b = export.read_blob(data)
        mid = {"f32": export.MODE_F32, "w8f": export.MODE_W8F, "w8i": export.MODE_W8I}[mode]
        for t, h, mw in ((0, 0, 0), (1, 1, 0), (2, 3, 256), (5, 7, 640), (16, 16, 1000), (128, 128, 1024), (4, 16, 8)):
            rt = Vocr(data, mode=mode, tile=t, head=h, max_width=mw)
            assert rt.arena_bytes == export.plan_arena(b, mw, t, h, mid), (mode, t, h, mw)
    big = attn_model(dim=96)
    plain = export.read_blob(blob_of(_without_attn(big), True))
    b = export.read_blob(blob_of(big, True))
    for mw in (256, 640, 1024):
        # this model's rings are small, so the working set (K, V of max_width / 8 frames) can outgrow them
        assert export.plan_arena(b, mw, 0, 0, export.MODE_W8I) >= export.plan_arena(plain, mw, 0, 0, export.MODE_W8I) \
            + mw // 8 * 96 * 4


@pytest.fixture(scope="module")
def gated_attn():
    """An attention blob (blank-heavy head) and a head gate fit on its own z' frames (a few steps), median threshold."""
    from pyvocr import Vocr

    import headgate as HG
    base = blob_of(attn_model(seed=4, blank_bias=0.6), True)
    rt = Vocr(base, mode="w8i")
    xs, ys = [], []
    for g in crops(20, 5):
        rt.run(g, head_in=True)
        xs.append(rt.last_head_in.copy())
        ys.append(rt.last_best == 0)
    x, y = np.concatenate(xs), np.concatenate(ys)
    assert 0.2 < y.mean() < 0.95
    old = dict(HG.FIT)
    HG.FIT.update(epochs=2, batch=512)
    try:
        f = HG.fit_float(x, y)
    finally:
        HG.FIT.clear()
        HG.FIT.update(old)
    params = HG.quantise(f["w1"], f["b1"], f["w2"], f["b2"])
    t = np.float32(np.median(HG.gate_scores(params, x)))
    return base, HG.append_gate(base, params, t), params, t


def test_head_gate_reads_z_prime(gated_attn):
    """With the attention op the head's input is z': the C gate scores equal the twin's on vocr_extra.head_in, and
    head_gate on equals the ungated read with the skipped frames as blanks, at several tilings."""
    from pyvocr import Vocr

    import export
    import headgate as HG
    base, blob, params, t = gated_attn
    assert HG.read_gate(blob)["k"] == 96 and export.read_blob(blob)["gate"]["hidden"] == 16
    a = Vocr(base, mode="w8i")
    vocab = [a.class_text(c) for c in range(a.n_classes)]
    skipped = frames = 0
    for tile, head in ((0, 0), (1, 1), (3, 7), (16, 4)):
        on = Vocr(blob, mode="w8i", head_gate="on", tile=tile, head=head)
        sc = Vocr(blob, mode="w8i", head_gate="score", tile=tile, head=head)
        assert on.arena_bytes == export.plan_arena(export.read_blob(blob), 0, tile, head, export.MODE_W8I, 1)
        for i, g in enumerate(crops(10, 6)):
            k = (1, 3, 8)[i % 3]
            x = a.run_topk(g, k, head_in=True)
            s = sc.run_topk(g, k, head_in=True, gate_score=True)
            tw = HG.gate_scores(params, x["head_in"])
            assert np.array_equal(s["head_in"], x["head_in"]) and np.array_equal(s["gate_score"].view(np.uint32),
                                                                                 tw.view(np.uint32))
            sim = HG.simulate(x, tw > t, vocab)
            y = on.run_topk(g, k)
            assert y["text"] == sim["text"] and y["gated"] == sim["gated"]
            assert np.array_equal(y["best"], sim["best"]) and np.array_equal(y["topc"], sim["topc"])
            assert np.array_equal(y["topv"].view(np.uint32), sim["topv"].view(np.uint32))
            skipped += sim["gated"]
            frames += len(tw)
    assert 0.2 < skipped / frames < 0.8


def test_two_level_reads_z_prime():
    """The two-level head on an attention blob: its rows permuted, a random escape; the C escape scores, fired
    frames, argmax and top-K equal train/head2level.py's twin on the head's input z'."""
    from pyvocr import Vocr

    import export
    import head2level as H2
    vocab = [chr(0x4E00 + i) for i in range(120)]
    m = attn_model(seed=6, vocab=vocab)
    with torch.no_grad():
        m.head.bias.normal_(0, 1.0)
        m.head.bias[0] += 2.0
    blob = blob_of(m, True, vocab)
    rng = np.random.default_rng(2)
    pblob = H2.permute_blob(blob, np.concatenate([[0], 1 + rng.permutation(len(vocab))]))
    head = H2.head_arrays(pblob)
    w1 = rng.normal(0, 0.3, size=(16, head["K"] + 2))
    q, scale = export.quantize_rows(w1[:, :head["K"]])
    esc = dict(q=q, scale=scale.astype(np.float32), bias=rng.normal(0, 0.5, 16).astype(np.float32),
               wl1=rng.normal(0, 0.05, 16).astype(np.float32), wb0=rng.normal(0, 0.05, 16).astype(np.float32),
               w2=rng.normal(0, 0.5, 16).astype(np.float32), b2=np.float32(0.1), t=np.float32(0.0))
    two_blob = H2.attach(pblob, H2.write_trailer(pblob, 3, esc))
    two = H2.read_trailer(two_blob)
    full, rt = Vocr(pblob, mode="w8i"), Vocr(two_blob, mode="w8i", two_level=1)
    assert rt.arena_bytes == export.plan_arena(export.read_blob(two_blob), 1024, 0, 0, 3, two_level=two)
    fired = frames = 0
    for g in crops(12, 7):
        full.run(g, head_in=True)
        tw = H2.twolevel_exact(full.last_head_in, head, two, topk=8)
        out = rt.run_topk(g, 8, esc=True)
        assert np.array_equal(out["esc"], tw["score"]) and out["escaped"] == int(tw["fire"].sum())
        assert np.array_equal(out["best"], tw["best"]) and np.array_equal(out["topc"], tw["topc"])
        assert np.array_equal(out["topv"], tw["topv"])
        fired += int(tw["fire"].sum())
        frames += len(tw["fire"])
    assert 0 < fired < frames


def test_checkpoint_tools_rebuild_the_block(tmp_path):
    """export.load_checkpoint, pyvocr's from_checkpoint and eval's TinyTorch read an attention checkpoint."""
    import backends
    from pyvocr import Vocr

    import export
    vocab = [chr(0x3042 + i) for i in range(40)]
    m = attn_model(vocab=vocab)
    ck = {"model": m.state_dict(), "vocab": vocab, "width": 0.25, "bottleneck": 96, "height": 32,
          "attn": 1, "attn_dim": 48, "attn_heads": 2}
    torch.save(ck, tmp_path / "a.pt")
    m2, v2, h2 = export.load_checkpoint(tmp_path / "a.pt")
    assert m2.attn is not None and v2 == vocab and h2 == 32
    rt = Vocr.from_checkpoint(tmp_path / "a.pt", int8=False, mode="f32")
    tt = backends.TinyTorch(tmp_path / "a.pt")
    assert tt.model.attn is not None
    g = crops(1, 9)[0]
    x = x_of(g)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)[None, None])[:, 0].numpy()
    _, lg = rt.run(g, logits=True)
    assert np.abs(lg - ref).max() <= 1e-3 * max(1.0, np.abs(ref).max())


RELEASED = {"S": ("v5f_S_seed2", "f175bfcb8cec65d69e85dd5f2cf707c9a711ada7203c0d1099cd2d0142978b8d"),
            "N": ("v5f_N_seed3", "20be179def8033171435ba62388657b3d3c932b365093737948c636e025e3bdc"),
            "L": ("v5f_L_seed2", "50413b37dbd85b2acb0908967bdeaf7ca67803850e0627a7b7cc3d274edc0ba3")}


@pytest.mark.parametrize("tier", list(RELEASED))
def test_the_release_checkpoints_still_export_to_the_released_blobs(tier):
    """Local only (the VOCR5 run checkpoints under $HERD): with the attention op in the exporter, VOCR5-S/N/L re-export
    to the v0.3.0 blobs byte for byte (vocr5-assets.json's sha256)."""
    import hashlib

    import export
    herd = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
    ck = herd / "shared/artifacts/vocr5-train/runs" / RELEASED[tier][0] / "ckpt_step15000.pt"
    if not ck.exists():
        pytest.skip(f"no checkpoint {ck}")
    m, vocab, h = export.load_checkpoint(ck)
    blob = export.build_blob(export.model_ops(m, h), vocab, h, True, None, hashlib.sha256(ck.read_bytes()).digest())[0]
    assert hashlib.sha256(blob).hexdigest() == RELEASED[tier][1]


def test_committed_cost_records():
    """The sizes and time records docs/vocr6-attn.md quotes: the deltas it states, and estimates marked as such."""
    s = json.loads((ROOT / "eval/vocr6_attn_sizes_20261026.json").read_text())
    t = json.loads((ROOT / "eval/vocr6_attn_time_20261026.json").read_text())
    for tier in "SNL":
        d = s["tiers"][tier]["delta"]
        assert (d["train_params"], d["blob_w8_bytes"], d["macs_per_frame"], d["macs_per_frame_T80"]) == \
            (37153, 40000, 36864, 52224)
        assert d["arena"]["w8i"] == {"256": 12288, "640": 30720, "1024": 49152}
        assert s["tiers"][tier]["today"]["release_identical"] and s["tiers"][tier]["today"]["speed_identical"]
        for basis in ("nopld", "pld"):
            assert t["tiers"][tier][basis]["pack_texts_identical"]
    assert t["estimates_are_not_measurements"] and t["scale"] == 1.000676
    assert t["tiers"]["N"]["nopld"]["nominal_box_ms_333"] == {"ref": 758.4, "attn": 793.5, "delta": 35.1}

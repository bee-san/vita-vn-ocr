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

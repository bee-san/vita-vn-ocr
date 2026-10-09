"""ci/fixture_ckpt.py: the runtime parity fixture is deterministic, in train.py's checkpoint
format, and has non-trivial BatchNorm so the exporter's BN folding is really exercised."""
import torch
from torch import nn

import fixture_ckpt
from common import load_vocab
from model import TinyVNOCR


def test_same_seed_same_weights_other_seed_differs():
    a, _ = fixture_ckpt.make(seed=0, width=0.25)
    b, _ = fixture_ckpt.make(seed=0, width=0.25)
    c, _ = fixture_ckpt.make(seed=1, width=0.25)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa if sa[k].is_floating_point())


def test_batchnorm_is_not_the_identity():
    model, _ = fixture_ckpt.make(seed=0)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and not model.training
    for m in bns:
        assert not torch.allclose(m.running_mean, torch.zeros_like(m.running_mean))
        assert not torch.allclose(m.running_var, torch.ones_like(m.running_var))
        assert not torch.allclose(m.weight, torch.ones_like(m.weight))
        assert torch.isfinite(m.running_var).all() and (m.running_var > 0).all()


def test_checkpoint_loads_like_a_train_py_checkpoint(tmp_path):
    path = fixture_ckpt.save(tmp_path / "fixture.pt", seed=0)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert {"model", "vocab", "width", "step", "best_cer"} <= set(ck)
    assert ck["vocab"] == load_vocab() and ck["width"] == 1.0
    model = TinyVNOCR(len(ck["vocab"]) + 1, width=ck["width"])
    model.load_state_dict(ck["model"])
    with torch.no_grad():
        y = model.eval()(torch.rand(2, 1, 32, 64) * 2 - 1)
    assert torch.isfinite(y).all() and float(y.std()) > 0.1     # logits spread, not collapsed

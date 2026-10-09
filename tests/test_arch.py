"""Fast tests for the architecture-sweep pieces: QAT fake quantisation, height 24,
<unk> vocabs, checkpoint round trip and the cost model. CPU only, no data files."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "synth"))

import cost  # noqa: E402
from model import TinyVNOCR, fake_quant_weight, from_checkpoint, set_weight_quant  # noqa: E402
from train import UNK, encoder  # noqa: E402

torch.set_num_threads(1)


@pytest.mark.parametrize("height", [24, 32])
@pytest.mark.parametrize("width", [0.25, 1.0])
def test_output_frames(height, width):
    m = TinyVNOCR(101, width=width, bottleneck=48).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 1, height, 80))
    assert out.shape == (10, 2, 101)


def test_v1_shape_unchanged():
    m = TinyVNOCR(5355)
    assert sum(p.numel() for p in m.parameters()) == 761_771     # v1's 0.762M


def test_fake_quant_is_int8_grid_with_straight_through_grad():
    w = torch.randn(16, 8, 3, 3, requires_grad=True)
    q = fake_quant_weight(w)
    scale = w.detach().abs().flatten(1).amax(1).view(-1, 1, 1, 1) / 127
    k = q.detach() / scale
    assert torch.allclose(k, k.round(), atol=1e-4) and k.abs().max() <= 127 + 1e-4
    q.sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_fake_quant_commutes_with_bn_folding():
    """Quantising the raw conv weight per output channel and then folding BN equals
    folding first and quantising the folded weight (what an int8 exporter does)."""
    torch.manual_seed(0)
    conv, bn = nn.Conv2d(8, 16, 3, bias=False), nn.BatchNorm2d(16)
    bn.weight.data.uniform_(-2, 2)                      # negative gammas flip signs
    bn.running_var.data.uniform_(0.1, 3)
    k = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().view(-1, 1, 1, 1)
    a = fake_quant_weight(conv.weight.detach()) * k
    b = fake_quant_weight(conv.weight.detach() * k)
    assert torch.allclose(a, b, atol=1e-6)


def test_int8_set_matches_export():
    m = TinyVNOCR(50)
    set_weight_quant(m, 8)
    stem = m.features[0][0]
    dw = m.features[1].dw[0]
    assert stem.wbits == 0 and dw.wbits == 0
    assert m.features[1].pw[0].wbits == 8 and m.head.wbits == 8 and m.bottleneck.wbits == 8
    assert m.features[7][0].wbits == 8                      # 2x1 row collapse


def test_set_weight_quant_changes_and_restores_output():
    torch.manual_seed(0)
    m = TinyVNOCR(50, width=0.25).eval()
    x = torch.rand(1, 1, 32, 64) * 2 - 1
    with torch.no_grad():
        ref = m(x)
        set_weight_quant(m, 8)
        q = m(x)
        set_weight_quant(m, 0)
        back = m(x)
    assert not torch.equal(ref, q) and torch.equal(ref, back)


def test_unk_encoder():
    plain = encoder(["あ", "い"])
    assert plain("あい") == [1, 2] and plain("あう") == [1, -1]
    unk = encoder(["あ", "い", UNK])
    assert unk("あうい") == [1, 3, 2]


@pytest.mark.parametrize("size", [2500, 3500])
def test_reduced_vocabs(size):
    """data/vocab_N.txt: N entries, a code-point-sorted subset of data/vocab.txt plus the
    trailing <unk>, with every kana of the full vocab kept."""
    from common import DATA, load_vocab
    from vocab import is_kana
    full = load_vocab()
    v = load_vocab(DATA / f"vocab_{size}.txt")
    assert len(v) == size and v[-1] == UNK and UNK not in full
    assert v[:-1] == sorted(v[:-1]) and set(v[:-1]) <= set(full)
    assert {c for c in full if is_kana(c)} <= set(v)
    assert set("、。「」『』！？…ー") <= set(v)


def test_vocab_build_sizes_and_extra():
    from vocab import build
    full = list("あいうアイウ一二三四五")
    freq = {"五": 5.0, "三": 3.0, "一": 1.0}
    assert build(9, freq, full) == sorted("あいうアイウ五三") + [UNK]
    assert build(0, freq, full) == sorted(full) + [UNK]
    v = build(9, freq, full, extra=["龍"])                   # extras are always kept
    assert "龍" in v and "五" in v and "三" not in v and len(v) == 9
    with pytest.raises(ValueError):
        build(5, freq, full)                                 # kana alone need 6 entries


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    m = TinyVNOCR(4, width=0.25, bottleneck=48).to(memory_format=torch.channels_last)
    ck = {"model": {k: v.contiguous() for k, v in m.state_dict().items()},
          "vocab": ["あ", "い", UNK], "width": 0.25, "bottleneck": 48, "height": 24, "unk": UNK}
    torch.save(ck, tmp_path / "ck.pt")
    m2, meta = from_checkpoint(tmp_path / "ck.pt")
    assert meta["height"] == 24 and meta["unk"] == UNK and meta["bottleneck"] == 48
    x = torch.rand(1, 1, 24, 40)
    with torch.no_grad():
        assert torch.allclose(m.eval()(x), m2(x), atol=1e-6)


def test_cost_model_consistent():
    m = TinyVNOCR(5355)
    r = cost.report(m, 5355, 32)
    assert r["params"] == 761_771
    # BN folding: 4 BN tensors per channel become 1 bias, so folded < trained params
    assert r["params_folded"] < r["params"]
    assert r["weights_fp32"] == 4 * r["params_folded"]
    assert r["weights_int8"] < r["weights_fp32"] / 3
    assert r["arena_256"] < r["arena_640"] < r["arena_1024"]
    assert r["macs_1024"] == 4 * r["macs_256"]
    # height 24 uses the same parameters and needs less arena at the same width
    r24 = cost.report(m, 5355, 24)
    assert r24["params"] == r["params"] and r24["arena_1024"] < r["arena_1024"]

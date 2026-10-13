"""Default arguments give exactly the model main had before the architecture-candidate knobs
(docs/vocr3-arch-candidates.md: --chans, --rep, the guide's return_features).

The reference is tests/model_pre_arch_candidates.py, a verbatim copy of train/model.py at main
a0b4d22 (sha256 pinned below). For the v1, FL10, H15 and F20 configurations (plus a deeper one)
the current TinyVNOCR with default arguments has the same modules, the same state_dict keys,
shapes, dtypes and initial values for a seed, the same forward output and gradients, and the
same f32 and int8 blobs byte for byte; an old-format checkpoint loads and exports unchanged; and
blobs of exactly-filled weights match hashes computed with main's train/model.py and
train/export.py. The released checkpoints re-export to the released blobs where they are on disk.
CPU only."""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import export  # noqa: E402
import model as M  # noqa: E402

torch.set_num_threads(1)
LEGACY = ROOT / "tests/model_pre_arch_candidates.py"
LEGACY_SHA256 = "ba45bcaba8b383963bad097be323f19d894fb96eda22b3462907476b72fc8303"   # main a0b4d22 train/model.py
# name: (classes incl. the CTC blank, TinyVNOCR keyword arguments)
CONFIGS = {"v1": (5355, {}), "FL10": (3501, {"width": 1.0}), "H15": (5355, {"width": 1.5}),
           "F20": (5355, {"width": 2.0}),
           "deep": (5355, {"width": 1.5, "bottleneck": 128, "depth": 2, "ctx": 4, "ctx_k": 7})}
# sha256 of the blobs of fill_exact() models, built with main a0b4d22's train/model.py and
# train/export.py (bytes, sha256)
PINS = {
    "v1_h32_f32": (3068064, "e2fc932fb00f8fc245a5c0b212ec4c424b6268939584f11fbe22ae63465b0100"),
    "v1_h32_w8": (860128, "ee6232fa21ddcc1103de67a1108d46d4a5041c2c94a995583323abd49487d14e"),
    "v1_h24_f32": (3068064, "95b6a6a96f69ab3cd1b92d26d7045037726e98dbe2faa3701bbc2ecc38b9636c"),
    "v1_h24_w8": (860128, "d04a4ccc8a3d777916044c99571073aa4bd7198bc8bfb51eeb63d21694800a5e"),
    "FL10_h32_f32": (2338656, "bee48fc7f22d73af5c09ecda80bf9836eb9cc12a360e614796852a639bdb116d"),
    "FL10_h32_w8": (657824, "4ed0fc6269f3465e94ebcfc29f862271b1444145e99823a934c26d83eb65ca45"),
    "H15_h32_f32": (4184096, "73a09ce811bb32ea265e3d98f0bde00d557fa2112d421c2d5dfc439ac605f77c"),
    "H15_h32_w8": (1156192, "e4ca4ccf52c2997273f23038d139163f94bfb1968dbf89436a194d6088075796"),
    "F20_h32_f32": (5724064, "9eb409e981da953df7bf7c4d2164cd7a73aa0f09610bddcd91d0d3853536be21"),
    "F20_h32_w8": (1558240, "1ee6c5da57d10bb6b7fc8818a7c116c7db0b6c2d85a77eaf95e52320a62f5526"),
    "deep_h32_f32": (6034528, "1be662705187ecd6bdab18852844fea441b376540ec1037dcffae4874fe9a4bc"),
    "deep_h32_w8": (1660256, "83683c6cd42f90623b6c60a12dcafbe64397158de94572bc0406d3eedcf09f4f"),
}
# the v0.2.0 release blobs (docs/train2.md; the overlay's VOCR_H15 is the H15 one)
RELEASED_W8 = {"FL10": "500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84",
               "H15": "316ebc377ec34520440991c5085199d8e74031f01a867938be90a60c1e3f03cd",
               "F20": "f8c1abe4743a71a742e718523f4206f992d3bd4fb69de321997b2bd6f16779c0"}


@pytest.fixture(scope="module")
def L():
    spec = importlib.util.spec_from_file_location("model_pre_arch_candidates", LEGACY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fill_exact(model):
    """Deterministic, exactly representable weights and BatchNorm statistics without any RNG
    (multiples of 2^-13), so blob hashes do not depend on the machine's RNG or float paths."""
    with torch.no_grad():
        for i, (name, t) in enumerate(model.state_dict().items()):
            if not t.dtype.is_floating_point:
                continue
            k = torch.arange(t.numel(), dtype=torch.int64)
            v = ((k * 7919 + i * 104729) % 2003 - 1001).to(torch.float64) / 8192
            if name.endswith("running_var"):
                v = 0.5 + v.abs() * 8
            elif t.dim() == 1 and name.endswith(".weight"):
                v = 1 + v
            t.copy_(v.reshape(t.shape).to(t.dtype))
    return model.eval()


def vocab_of(n):
    return [chr(0x4E00 + i) for i in range(n - 1)]


def ops_of(module, model, height, monkeypatch):
    """export.model_ops with `module` as train/model.py (it recognises DS and Ctx from there)."""
    monkeypatch.setitem(sys.modules, "model", module)
    try:
        return export.model_ops(model, height)
    finally:
        monkeypatch.setitem(sys.modules, "model", M)


def test_the_reference_is_main_verbatim():
    assert hashlib.sha256(LEGACY.read_bytes()).hexdigest() == LEGACY_SHA256


@pytest.mark.parametrize("name", list(CONFIGS))
def test_same_modules_and_state_dict_for_a_seed(L, name):
    n, kw = CONFIGS[name]
    torch.manual_seed(0)
    old = L.TinyVNOCR(n, **kw)
    torch.manual_seed(0)
    new = M.TinyVNOCR(n, **kw)
    assert [(k, type(m).__name__) for k, m in old.named_modules()] == \
        [(k, type(m).__name__) for k, m in new.named_modules()]
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn)
    for k in so:
        assert so[k].shape == sn[k].shape and so[k].dtype == sn[k].dtype and torch.equal(so[k], sn[k]), k
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    assert [n_ for n_, _ in old.named_parameters()] == [n_ for n_, _ in new.named_parameters()]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_same_forward_and_gradients(L, name):
    n, kw = CONFIGS[name]
    old, new = fill_exact(L.TinyVNOCR(n, **kw)), fill_exact(M.TinyVNOCR(n, **kw))
    x = torch.rand(2, 1, 32, 96, generator=torch.Generator().manual_seed(1)) * 2 - 1
    with torch.no_grad():
        assert torch.equal(old(x), new(x))
    old.train()
    new.train()
    torch.manual_seed(2)                                  # dropout draws the same mask in both
    yo = old(x)
    torch.manual_seed(2)
    yn = new(x)
    assert torch.equal(yo, yn)
    yo.square().mean().backward()
    yn.square().mean().backward()
    for (ko, po), (kn, pn) in zip(old.named_parameters(), new.named_parameters()):
        assert ko == kn and torch.equal(po.grad, pn.grad), ko
    for (ko, bo), (kn, bn) in zip(old.named_buffers(), new.named_buffers()):
        assert ko == kn and torch.equal(bo, bn), ko       # BatchNorm running statistics updated alike


@pytest.mark.parametrize("name", list(CONFIGS))
def test_byte_identical_export(L, name, monkeypatch, tmp_path):
    n, kw = CONFIGS[name]
    old, new = fill_exact(L.TinyVNOCR(n, **kw)), fill_exact(M.TinyVNOCR(n, **kw))
    vocab = vocab_of(n)
    for height in (32, 24) if name == "v1" else (32,):
        o_ops, n_ops = ops_of(L, old, height, monkeypatch), ops_of(M, new, height, monkeypatch)
        for int8 in (False, True):
            ob = export.build_blob(o_ops, vocab, height, int8=int8)[0]
            nb = export.build_blob(n_ops, vocab, height, int8=int8)[0]
            assert ob == nb, (name, height, int8)
            assert (len(nb), hashlib.sha256(nb).hexdigest()) == PINS[f"{name}_h{height}_{'w8' if int8 else 'f32'}"]
    # an old-format checkpoint (no chans / rep keys) rebuilds and exports unchanged
    ck = {"model": old.state_dict(), "vocab": vocab, "width": kw.get("width", 1.0),
          "bottleneck": kw.get("bottleneck", 96), "height": 32,
          **{k: kw[k] for k in ("depth", "ctx", "ctx_k") if k in kw}}
    torch.save(ck, tmp_path / "old.pt")
    model, v, height = export.load_checkpoint(tmp_path / "old.pt")
    assert v == vocab and height == 32
    assert export.build_blob(export.model_ops(model, 32), vocab, 32, int8=True)[0] == \
        export.build_blob(ops_of(L, old, 32, monkeypatch), vocab, 32, int8=True)[0]
    m2, meta = M.from_checkpoint(ck)
    assert "chans" not in meta and all(torch.equal(a, b) for a, b in zip(m2.state_dict().values(),
                                                                           old.state_dict().values()))


def release_dir():
    d = os.environ.get("VOCR_RELEASE_DIR")
    if d:
        return Path(d)
    herd = os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")
    return Path(herd) / "shared/artifacts/train2/release"


@pytest.mark.parametrize("name", list(RELEASED_W8))
def test_released_checkpoints_reexport_to_the_released_blobs(name):
    """Local only: needs the v0.2.0 release checkpoints ($VOCR_RELEASE_DIR or $HERD's train2/release)."""
    d = release_dir()
    ck = d / f"{name}_ckpt_best.pt"
    if not ck.exists():
        pytest.skip(f"no release checkpoint {ck}")
    model, vocab, height = export.load_checkpoint(ck)
    blob = export.build_blob(export.model_ops(model, height), vocab, height, True, 1024,
                             hashlib.sha256(ck.read_bytes()).digest())[0]
    assert hashlib.sha256(blob).hexdigest() == RELEASED_W8[name]


def test_default_forward_returns_only_logits():
    m = M.TinyVNOCR(11, width=0.25, bottleneck=16).eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 1, 32, 64))
        logits, feats = m(torch.zeros(1, 1, 32, 64), return_features=True)
    assert isinstance(out, torch.Tensor) and torch.equal(out, logits) and feats.shape[2] == 1
    assert not any(isinstance(x, (M.RepCBR, nn.MultiheadAttention)) for x in m.modules())

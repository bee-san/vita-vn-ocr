"""The architecture-candidate knobs (docs/vocr3-arch-candidates.md): explicit per-stage channels
(--chans) and training-time re-parameterisation (--rep). CPU only, tiny models, no data files."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for d in ("train", "synth", "eval", "vita"):
    sys.path.insert(0, str(ROOT / d))

import export  # noqa: E402
from model import RepCBR, TinyVNOCR, from_checkpoint, fuse_rep, load_plain_into_rep  # noqa: E402

torch.set_num_threads(1)
LATE_WIDE = (16, 24, 40, 64, 72)        # a tiny late-wide shape: wider 2-row stage and frame row


def randomise_bn(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.num_features, generator=g) * 1.5 + 0.5)
                m.weight.copy_(torch.rand(m.num_features, generator=g) + 0.5)
                m.bias.copy_(torch.randn(m.num_features, generator=g) * 0.1)
    return model


def test_defaults_keep_v1_and_rep_only_adds_training_branches():
    v1 = TinyVNOCR(5355)
    assert sum(p.numel() for p in v1.parameters()) == 761_771
    plain_keys = list(TinyVNOCR(50, width=0.5).state_dict())
    rep = TinyVNOCR(50, width=0.5, rep=2)
    reps = [m for m in rep.modules() if isinstance(m, RepCBR)]
    assert len(reps) == 1 + 6 + 2                   # stem, six DS depthwise convs, two Ctx depthwise
    assert all(not isinstance(m.pw, RepCBR) for m in rep.features if hasattr(m, "pw"))   # int8 ops stay plain
    assert list(fuse_rep(rep.eval()).state_dict()) == plain_keys


@pytest.mark.parametrize("rep", [1, 3])
def test_rep_fuses_exactly_into_the_plain_model(rep):
    torch.manual_seed(rep)
    m = randomise_bn(TinyVNOCR(31, width=0.5, rep=rep), rep).eval()
    p = fuse_rep(m)
    x = torch.randn(2, 1, 32, 72)
    with torch.no_grad():
        a, b = m(x), p(x)
    assert (a - b).abs().max() <= 1e-5 * max(1.0, float(a.abs().max()))
    # the fused BN is the identity plus the bias
    bn = p.features[0][1]
    assert torch.all(bn.weight == 1) and torch.all(bn.running_mean == 0)


def test_load_plain_into_rep_preserves_the_function():
    torch.manual_seed(5)
    plain = randomise_bn(TinyVNOCR(31, width=0.5), 5).eval()
    rep = load_plain_into_rep(TinyVNOCR(31, width=0.5, rep=2), plain.state_dict()).eval()
    x = torch.randn(1, 1, 32, 48)
    with torch.no_grad():
        assert torch.equal(rep(x), plain(x))


def test_rep_rejects_even_or_unaligned_kernels():
    with pytest.raises(ValueError):
        RepCBR(8, 8, (2, 1), pad=(0, 0))
    with pytest.raises(ValueError):
        RepCBR(8, 8, 3, pad=0)


def test_chans_validation_and_checkpoint_round_trip():
    with pytest.raises(ValueError):
        TinyVNOCR(10, chans=(16, 24, 40, 60, 72))   # 60 is not a multiple of 8
    with pytest.raises(ValueError):
        TinyVNOCR(10, chans=(16, 24, 40, 64))
    m = TinyVNOCR(21, chans=LATE_WIDE)
    assert m.features[-1].pw[0].weight.shape == (72, 72, 1, 1)
    ck = {"model": m.state_dict(), "vocab": ["x"] * 20, "chans": list(LATE_WIDE)}
    m2, meta = from_checkpoint(ck)
    assert meta["chans"] == LATE_WIDE
    assert sum(p.numel() for p in m2.parameters()) == sum(p.numel() for p in m.parameters())


def test_late_wide_rep_model_exports_and_the_c_runtime_matches(tmp_path):
    """A late-wide, rep-trained model fuses to ops the runtime already has: export.load_checkpoint
    rebuilds it, and the C runtime's W8I logits match PyTorch's int8-weight model."""
    torch.manual_seed(0)
    vocab = [chr(0x3042 + i) for i in range(20)]
    rep = randomise_bn(TinyVNOCR(len(vocab) + 1, chans=LATE_WIDE, bottleneck=24, rep=2), 3).eval()
    plain = fuse_rep(rep)
    path = tmp_path / "lw.pt"
    torch.save({"model": plain.state_dict(), "vocab": vocab, "chans": list(LATE_WIDE), "bottleneck": 24,
                "height": 32}, path)
    model, v, height = export.load_checkpoint(path)
    ops = export.model_ops(model, height)
    kinds = {(op["type"], op["kh"], op["kw"]) for op in ops}
    assert kinds <= {(export.T_CONV, 3, 3), (export.T_DW, 3, 3), (export.T_DW, 1, 5), (export.T_PW, 1, 1),
                     (export.T_PW, 2, 1), (export.T_HEAD, 1, 1)}
    blob, _ = export.build_blob(ops, vocab, height, int8=True)
    if not (shutil.which("cc") or shutil.which("gcc")):
        pytest.skip("no C compiler")
    sys.path.insert(0, str(ROOT / "runtime"))
    from pyvocr import Vocr
    bpath = tmp_path / "lw_w8.vocr"
    bpath.write_bytes(blob)
    try:
        r = Vocr(str(bpath), mode="w8i")
    except OSError as e:
        pytest.skip(f"libvocr.so unavailable: {e}")
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (32, 96), dtype=np.uint8)
    _, c_logits = r.run(crop, logits=True)
    from model import set_weight_quant
    set_weight_quant(model, 8)
    x = torch.from_numpy(crop.astype(np.float32)[None, None] / 127.5 - 1.0)
    with torch.no_grad():
        t_logits = model(x)[:, 0].numpy()
    scale = max(1.0, float(np.abs(t_logits).max()))
    assert np.abs(c_logits - t_logits).max() / scale < 5e-3
    assert (c_logits.argmax(1) == t_logits.argmax(1)).mean() > 0.95


def write_shard(path, crops, labels):
    np.savez(path, data=np.concatenate([c.ravel() for c in crops]).astype(np.uint8),
             offsets=np.cumsum([0] + [c.size for c in crops]).astype(np.int64),
             shapes=np.array([c.shape for c in crops], np.int32),
             labels=json.dumps(labels, ensure_ascii=False))


class Exited(Exception):
    pass


def test_train2_trains_resumes_and_saves_the_fused_rep_model(tmp_path, monkeypatch):
    import train2
    vocab = ["あ", "い", "う"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (32, 64), dtype=np.uint8) for _ in range(8)]
    write_shard(tmp_path / "s.npz", crops, ["あい", "う", "いう", "あ"] * 2)

    def boom(code=0):
        raise Exited(code)

    monkeypatch.setattr(os, "_exit", boom)
    argv = ["--run", "r", "--out", str(tmp_path / "runs"), "--src", f"a:1:{tmp_path / 's.npz'}",
            "--val", f"v:{tmp_path / 's.npz'}", "--batch", "4", "--warmup", "1", "--threads", "1",
            "--vocab", str(tmp_path / "vocab.txt"), "--chans", ",".join(map(str, LATE_WIDE)),
            "--bottleneck", "16", "--rep", "2", "--ema", "0.9", "--eval-every", "2", "--log-every", "1"]
    with pytest.raises(Exited):
        train2.main(argv + ["--steps", "2"])
    with pytest.raises(Exited):                      # resume from ckpt_last.pt to step 4
        train2.main(argv + ["--steps", "4"])
    ck = torch.load(tmp_path / "runs/r/ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4 and ck["rep"] == 2 and ck["chans"] == list(LATE_WIDE)
    assert "model_rep" in ck and "model_raw" in ck
    model, v, _ = export.load_checkpoint(tmp_path / "runs/r/ckpt_last.pt")   # "model" is plain
    rep = TinyVNOCR(len(vocab) + 1, chans=LATE_WIDE, bottleneck=16, rep=2)
    rep.load_state_dict(ck["model_rep"])
    x = torch.randn(1, 1, 32, 64)
    with torch.no_grad():
        assert (rep.eval()(x) - model(x)).abs().max() < 1e-4
    # a plain fine-tune can start from the rep checkpoint's fused weights, and a rep one from either
    with pytest.raises(Exited):
        train2.main(["--run", "ft", "--init", str(tmp_path / "runs/r/ckpt_last.pt")] + argv[2:] + ["--steps", "1"])

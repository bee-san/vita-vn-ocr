"""Model B (#126): train2.py --height, the depth/ctx-drop warm start (train/modelb_init.py), and a height-24 model
through export and the C runtime. Generated data only."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in ("train", "eval", "runtime", "synth"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

import export as E  # noqa: E402
from model import Ctx, TinyVNOCR, from_checkpoint  # noqa: E402

VOCAB = [chr(0x3042 + i) for i in range(30)]


def _model(depth=1, ctx=2, seed=0):
    torch.manual_seed(seed)
    m = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48, depth=depth, ctx=ctx)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_var.uniform_(0.5, 2.0)
            mod.running_mean.uniform_(-0.2, 0.2)
    return m.eval()


@pytest.mark.parametrize("height", [17, 20, 24, 28, 32])
def test_every_height_collapses_to_one_row(height):
    m = _model()
    with torch.no_grad():
        y = m(torch.zeros(2, 1, height, 64))
    assert y.shape == (8, 2, len(VOCAB) + 1)


def test_train2_height_geometry_and_batches():
    import train2
    try:
        assert train2.set_height(24) == 24 and train2.H == 24
        rng = np.random.default_rng(0)
        items = [train2.Item(rng.integers(0, 255, (24, w)).astype(np.uint8), np.array([1, 2, 3])) for w in (50, 97)]
        x, in_len, *_ = train2.make_batch(items, None, [{}])
        assert x.shape[2] == 24 and x.shape[3] % 8 == 0 and in_len.tolist() == [math.ceil(50 / 8), math.ceil(97 / 8)]
        # augmentation keeps the height, finder augmentation resizes back to it
        aug = train2.augment(items[0].img, np.random.default_rng(3))
        assert aug.shape[0] == 24
        from finder_aug import finder_aug
        fa = finder_aug(items[1].img.astype(np.float32), np.random.default_rng(4), items[0].img, height=train2.H)
        assert fa.shape[0] == 24
        for bad in (16, 33, 0):
            with pytest.raises(SystemExit):
                train2.set_height(bad)
    finally:
        train2.set_height(32)
    assert train2.H == 32


def test_valset_reads_at_its_height(tmp_path):
    import json

    import train2
    rng = np.random.default_rng(1)
    crops = [rng.integers(0, 255, (h, w)).astype(np.uint8) for h, w in ((15, 120), (19, 200), (23, 64))]
    offs = np.cumsum([0] + [c.size for c in crops])
    np.savez(tmp_path / "v.npz", data=np.concatenate([c.ravel() for c in crops]), offsets=offs,
             shapes=np.array([c.shape for c in crops]), labels=json.dumps(["あい", "うえお", "か"]))
    try:
        train2.set_height(24)
        v = train2.ValSet("v", tmp_path / "v.npz", {})
        assert v.height == 24 and all(g.shape[0] == 24 for g in v.imgs)
        m = TinyVNOCR(len(VOCAB) + 1, width=0.25, bottleneck=48).eval()
        cer, exact, _ = v.score(m, VOCAB)
        assert 0 <= cer and 0 <= exact <= 1
    finally:
        train2.set_height(32)


def test_train2_records_and_checks_the_height(tmp_path):
    """A 3-step run at height 24 records it; a resume at another height is refused; an init from height 32 is
    allowed (the weights fit any height)."""
    import json
    rng = np.random.default_rng(2)
    crops = [rng.integers(0, 255, (int(rng.integers(14, 24)), int(rng.integers(60, 200)))).astype(np.uint8)
             for _ in range(40)]
    offs = np.cumsum([0] + [c.size for c in crops])
    labels = ["".join(VOCAB[int(i)] for i in rng.integers(0, len(VOCAB), 3)) for _ in crops]
    np.savez(tmp_path / "s.npz", data=np.concatenate([c.ravel() for c in crops]), offsets=offs,
             shapes=np.array([c.shape for c in crops]), labels=json.dumps(labels))
    (tmp_path / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    base = [sys.executable, str(ROOT / "train" / "train2.py"), "--out", str(tmp_path / "runs"), "--src",
            f"r:1:{tmp_path / 's.npz'}", "--vocab", str(tmp_path / "vocab.txt"), "--width", "0.25",
            "--bottleneck", "48", "--batch", "8", "--threads", "1", "--eval-every", "3", "--warmup", "1"]
    r = subprocess.run(base + ["--run", "h24", "--steps", "3", "--height", "24"], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ck = torch.load(tmp_path / "runs" / "h24" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["height"] == 24 and json.loads((tmp_path / "runs" / "h24" / "args.json").read_text())["height"] == "24"
    r = subprocess.run(base + ["--run", "h24", "--steps", "4", "--height", "32"], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode != 0 and "resume --height differs" in (r.stdout + r.stderr)
    r = subprocess.run(base + ["--run", "h32", "--steps", "3", "--height", "32", "--init",
                               str(tmp_path / "runs" / "h24" / "ckpt_last.pt")], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0 and "init from a height-24 checkpoint into height 32" in r.stdout


def test_shallower_keeps_the_kept_modules():
    import modelb_init as MI
    src = _model(depth=1, ctx=2, seed=3)
    for depth, ctx in ((0, 2), (0, 1), (1, 1), (1, 2), (0, 0)):
        dst = MI.shallower(src, depth, ctx)
        assert dst.arch["depth"] == depth and dst.arch["ctx"] == ctx
        s_roles = dict(zip(MI.roles(1, 2), src.features))
        for r, mod in zip(MI.roles(depth, ctx), dst.features):
            for (k, a), (k2, b) in zip(mod.state_dict().items(), s_roles[r].state_dict().items()):
                assert k == k2 and torch.equal(a, b), (r, k)
        assert torch.equal(dst.head.weight, src.head.weight) and torch.equal(dst.bottleneck.weight,
                                                                              src.bottleneck.weight)
        assert sum(isinstance(m, Ctx) for m in dst.features) == ctx
    with pytest.raises(ValueError):
        MI.shallower(src, 2, 2)
    # with depth and ctx kept, the copy is the same function
    x = torch.rand(2, 1, 24, 96)
    with torch.no_grad():
        assert torch.equal(MI.shallower(src, 1, 2)(x), src(x))


def test_modelb_init_cli_and_train2_init(tmp_path):
    src = _model(depth=1, ctx=2, seed=4)
    ck = {"model": src.state_dict(), "vocab": VOCAB, "width": 0.25, "bottleneck": 48, "depth": 1, "ctx": 2,
          "ctx_k": 5, "height": 32, "step": 1234}
    torch.save(ck, tmp_path / "src.pt")
    r = subprocess.run([sys.executable, str(ROOT / "train" / "modelb_init.py"), "--ckpt", str(tmp_path / "src.pt"),
                        "--out", str(tmp_path / "init.pt"), "--depth", "0", "--ctx", "1", "--height", "24"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    out = torch.load(tmp_path / "init.pt", map_location="cpu", weights_only=False)
    m, meta = from_checkpoint(out)
    assert (meta["depth"], meta["ctx"], meta["height"], meta["width"]) == (0, 1, 24, 0.25)
    assert out["step"] == 0 and out["init_from"]["step"] == 1234 and len(out["init_from"]["sha256"]) == 64


def test_height24_shallow_model_exports_and_runs_in_c():
    """A depth-0, height-24 model exports, loads in the C runtime and equals its PyTorch logits within the
    parity tolerance, in W8F and F32."""
    from pyvocr import Vocr
    m = _model(depth=0, ctx=1, seed=5)
    rng = np.random.default_rng(6)
    g = rng.integers(0, 255, (19, 211)).astype(np.uint8)
    for int8 in (False, True):
        blob, _ = E.build_blob(E.model_ops(m, 24), VOCAB, 24, int8, None, b"")
        info = E.read_blob(blob)
        assert info["height"] == 24 and info["stride"] == 8
        rt = Vocr(blob, mode="w8f" if int8 else "f32")
        _, lg = rt.run(g, logits=True)
        x, _ = rt.preprocess(g)
        with torch.no_grad():
            ref = m(torch.from_numpy(x[None, None] / 127.5 - 1.0).float())[:, 0].numpy()
        assert lg.shape == ref.shape
        tol = 0.05 if int8 else 1e-3
        assert np.abs(lg - ref).max() < tol * (1 + np.abs(ref).max())

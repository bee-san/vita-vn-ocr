"""Model B (#126): train/modelb_head.py (permute, W8F features, gate + escape fit, the C runtime equal to the twins)
on a random-init 700-class model and generated shards. No development, test or private data is read."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("train", "data", "runtime", "runtime/tests", "synth"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

import modelb_head as MH  # noqa: E402
import modelb_pools as MP  # noqa: E402

VOCAB = [chr(0x4E00 + i) for i in range(700)]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    import torch
    from torch import nn

    import export
    from model import TinyVNOCR
    d = tmp_path_factory.mktemp("mbh")
    torch.manual_seed(3)
    m = TinyVNOCR(len(VOCAB) + 1, width=0.25, depth=0).eval()
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
        # blank-heavy, as a trained model's frames are: the blank wins about 60% of frames
        lg = m(torch.rand(8, 1, 24, 256, generator=torch.Generator().manual_seed(8)) * 2 - 1)
        dd = (lg[:, :, 1:].max(2).values - lg[:, :, 0]).flatten()
        m.head.bias[0] += float(torch.quantile(dd, 0.6))
    blob, _ = export.build_blob(export.model_ops(m, 24), VOCAB, 24, True, None, b"")
    (d / "b.vocr").write_bytes(blob)
    rng = np.random.default_rng(0)
    order = [VOCAB[i] for i in rng.permutation(len(VOCAB))]
    (d / "rank.json").write_text(json.dumps({"order": order}, ensure_ascii=False))
    for s in range(3):
        crops = [rng.integers(0, 255, (int(rng.integers(14, 26)), int(rng.integers(40, 300)))).astype(np.uint8)
                 for _ in range(40)]
        meta = [{"vn": f"v99{k:04d}", "title_id": f"TEST{(s * 40 + k) % 25:03d}"} for k in range(40)]
        MP.write_shard(d / f"s{s}.npz", crops, ["一"] * 40, meta)
    return d


def test_permute_features_fit(setup):
    d = setup
    import argparse
    MH.cmd_permute(argparse.Namespace(blob=str(d / "b.vocr"), rank=str(d / "rank.json"), out=str(d / "p.vocr")))
    from pyvocr import Vocr
    a, b = Vocr(d / "b.vocr", mode="w8f"), Vocr(d / "p.vocr", mode="w8f")
    g = np.random.default_rng(1).integers(0, 255, (20, 150)).astype(np.uint8)
    order = json.loads((d / "rank.json").read_text())["order"]
    perm = MH.perm_from_rank([""] + VOCAB, order)
    assert np.array_equal(a.run(g, logits=True)[1][:, perm], b.run(g, logits=True)[1])
    MH.cmd_features(argparse.Namespace(blob=str(d / "p.vocr"), sources=[str(d / "s*.npz")], lines=90,
                                       out=str(d / "f.npz")))
    z = np.load(d / "f.npz")
    assert len(z["off"]) == 91 and z["x"].shape[1] == 96 and z["cal_line"].any() and (~z["cal_line"]).any()
    assert 0.2 < (z["am"] == 0).mean() < 0.95 and (z["am"] >= MH.L1_ROWS).any()
    MH.cmd_fit(argparse.Namespace(blob=str(d / "p.vocr"), features=str(d / "f.npz"), out=str(d / "fit"), eps=1e-3,
                                  recall=0.999, check_lines=50))
    rec = json.loads((d / "fit" / "fit.json").read_text())
    assert all(v for k, v in rec["c_equals_twin"].items() if k != "lines") and rec["c_equals_twin"]["lines"] > 0
    assert rec["escape"]["missed_cal"] <= rec["escape"]["max_missed"]
    assert rec["gate"]["nonblank_skipped_cal"] <= rec["gate"]["allowed"]
    both = Vocr(d / "fit" / "gated2.vocr", mode="w8f", head_gate="on", two_level=1)
    assert both.has_gate and isinstance(both.text(g), str)


def test_perm_from_rank_refuses_another_vocab():
    with pytest.raises(SystemExit):
        MH.perm_from_rank(["", "a", "b"], ["a", "c"])

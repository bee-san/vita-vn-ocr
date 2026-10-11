"""Release restoration and host data filters, with generated test pixels only."""
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from pyvocr import Vocr  # noqa: E402

import hiragana_experiment as HE  # noqa: E402
import hiragana_gate as HG  # noqa: E402
import render  # noqa: E402
from export import build_blob, model_ops  # noqa: E402
from model import TinyVNOCR  # noqa: E402
from restore_blob import freeze_bn_stats, restore  # noqa: E402

spec = importlib.util.spec_from_file_location("import_animetext", ROOT / "data/import_animetext.py")
AT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AT)
POLICY = json.loads(AT.PIN.read_text())


def row(text="あ", shape=(32, 32), **kwargs):
    g = np.full(shape, 30, np.uint8)
    g[9:23, 5:shape[1] - 5:3] = 200
    buf = io.BytesIO()
    Image.fromarray(g).save(buf, "PNG")
    return {"transcription": text, "language": "ja", "seq_confidence": .99,
            "min_token_confidence": .99, "image_id": "123", "crop_id": "123_1",
            "image": {"bytes": buf.getvalue()}, **kwargs}


@pytest.mark.parametrize("patch,reason", [({"language": "en"}, "language"),
    ({"seq_confidence": float("nan")}, "seq_confidence"),
    ({"min_token_confidence": .89}, "min_token_confidence"),
    ({"transcription": "あ\nい"}, "empty_or_whitespace"),
    ({"transcription": "あ犬"}, "vocabulary"),
    ({"image_id": ""}, "missing_id")])
def test_confidence_and_vocabulary_filters(patch, reason):
    assert AT.filter_row(row(**patch), set("あい"), POLICY)[2] == reason


def test_horizontal_single_glyph_ctc_and_multiline_filters():
    assert AT.filter_row(row(), set("あ"), POLICY)[2] is None
    assert AT.filter_row(row("あい", (64, 20)), set("あい"), POLICY)[2] == "vertical_or_multiline"
    assert AT.filter_row(row("あ" * 20, (32, 224)), set("あ"), POLICY)[2] == "ctc_length"
    r = row("あい", (64, 160))
    g = np.full((64, 160), 30, np.uint8)
    g[8:23, 8:150:3] = 230
    g[40:55, 8:150:3] = 230
    buf = io.BytesIO()
    Image.fromarray(g).save(buf, "PNG")
    r["image"]["bytes"] = buf.getvalue()
    assert AT.filter_row(r, set("あい"), POLICY)[2] == "multiple_bands"


def test_group_split_is_reproducible_and_shards_roundtrip(tmp_path):
    assert AT.group_split("123") == AT.group_split("123")
    assert {AT.group_split(str(i)) for i in range(100)} == {"train", "holdout"}
    writer = AT.ShardWriter(tmp_path, per_shard=2)
    for i in range(3):
        writer.add(np.full((8, 16), i, np.uint8), "あ", {"split": "train", "crop_id": str(i)})
    writer.close()
    records = [json.loads(x) for x in (tmp_path / "crops.jsonl").read_text().splitlines()]
    assert len(records) == 3
    for i, r in enumerate(records):
        with np.load(tmp_path / r["shard"]) as z:
            assert json.loads(str(z["labels"]))[r["index"]] == "あ"
            assert (z["data"][z["offsets"][r["index"]]:z["offsets"][r["index"] + 1]] == i).all()


def test_matrix_keeps_matched_budget_and_dev_only_selection():
    config = {"original": {k: f"/data/{k}.npz" for k in HE.ORIGINAL},
              "dev": "/data/dev.npz", "targeted": "/data/targeted.npz", "anime": "/data/anime.npz"}
    cmds = HE.commands(config, "init.pt", "vocab.txt", "out")
    assert len(cmds) == 9
    for cmd in cmds:
        assert cmd[cmd.index("--steps") + 1] == "5000"
        assert cmd[cmd.index("--batch") + 1] == "96"
        assert cmd[cmd.index("--lr") + 1] == "0.0001"
        assert cmd[cmd.index("--select-metric") + 1] == "hiragana"
        sources = [cmd[i + 1].split(":") for i, c in enumerate(cmd) if c == "--src"]
        assert sum(float(s[1]) for s in sources) == pytest.approx(1)
        assert all("kd=" not in s[-1] and "test" not in s[2] for s in sources)
    with pytest.raises(ValueError, match="missing training source"):
        HE.check_data({**config, "anime_audit": "none.json", "anime_manifest": "none.jsonl"})


def test_targeted_renderer_uses_natural_text_and_keeps_target_in_crops(tmp_path, monkeypatch, font):
    import sandbox
    repo = sandbox.build(tmp_path / "repo", font)
    data = repo / "data"
    monkeypatch.setattr(render, "DATA", data)
    monkeypatch.setattr(render, "FONT_DIRS", [data / "fonts", data / "fonts_extra"])
    resources = render.Resources(target_kana="っ")
    assert all("っ" in text for pools in resources.target_pools.values() for pool in pools.values() for text in pool)
    monkeypatch.setattr(render, "R", resources)
    out = tmp_path / "targeted"
    out.mkdir()
    render.run_shard((0, 4, 22, str(out), "grad", True))
    with np.load(out / "synth_00000.npz") as z:
        assert all("っ" in label for label in json.loads(str(z["labels"])))
    with pytest.raises(ValueError, match="absent"):
        render.Resources(target_kana="ゑ")


def test_kana_targets_cannot_be_derived_from_test_results(tmp_path):
    path = tmp_path / "dev.json"
    report = {"datasets": {"clannad": {"split": "dev"}},
              "sets": {"clannad": {"hiragana": {"confusions": [{"reference": "っ", "prediction": "つ"}]}}}}
    path.write_text(json.dumps(report))
    assert render.development_kana(path) == "っつ"
    report["datasets"]["clannad"]["split"] = "test"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="development-only"):
        render.development_kana(path)


def test_gate_rejects_regressions_dataset_changes_and_memory(monkeypatch, tmp_path):
    import copy
    import hashlib
    from types import SimpleNamespace
    blob = tmp_path / "blob.vocr"
    blob.write_bytes(b"test identity")
    monkeypatch.setattr(HG, "Vocr", lambda *a, **kw: SimpleNamespace(blob_bytes=657664, arena_bytes=124832))
    sets = {name: {"all": {"errors": 10, "chars": 100}, "hiragana": {"errors": 4, "chars": 60},
                   "lines": [{"id": f"{name}-1", "text": "あ" * 60 + "字" * 40}]}
            for name in ("vita-test", "psp-test", "clannad-test")}
    base = {"sets": sets, "datasets": {n: {"split": "test", "manifest_sha256": "same"} for n in sets},
            "polarity": "console", "pad": {"vita": 4, "psp": 2}, "finder": None,
            "model_info": {"backend": "vocr", "mode": "w8i", "sha256": hashlib.sha256(blob.read_bytes()).hexdigest()}}
    candidate = copy.deepcopy(base)
    candidate["sets"]["clannad-test"]["hiragana"]["errors"] = 3
    assert HG.check(base, base, candidate, blob)["accepted"]
    candidate["sets"]["psp-test"]["all"]["errors"] = 11
    assert not HG.check(base, base, candidate, blob)["accepted"]
    candidate["datasets"]["clannad-test"]["split"] = "dev"
    with pytest.raises(ValueError, match="held-out"):
        HG.check(base, base, candidate, blob)
    candidate = copy.deepcopy(base)
    assert not HG.check(base, base, candidate, blob)["accepted"]
    candidate["polarity"] = "native"
    with pytest.raises(ValueError, match="console polarity"):
        HG.check(base, base, candidate, blob)
    candidate = copy.deepcopy(base)
    candidate["pad"] = 0
    with pytest.raises(ValueError, match="padding"):
        HG.check(base, base, candidate, blob)
    candidate = copy.deepcopy(base)
    candidate["sets"]["clannad-test"]["lines"][0]["id"] = "other-capture"
    with pytest.raises(ValueError, match="scored line identities"):
        HG.check(base, base, candidate, blob)
    candidate = copy.deepcopy(base)
    candidate["sets"]["clannad-test"]["hiragana"]["errors"] = 3
    monkeypatch.setattr(HG, "Vocr", lambda *a, **kw: SimpleNamespace(blob_bytes=657680, arena_bytes=124832))
    assert not HG.check(base, base, candidate, blob)["accepted"]


def test_restore_preserves_vocab_collapse_bias_and_c_logits(tmp_path):
    torch.manual_seed(12)
    model = TinyVNOCR(4, width=.25, bottleneck=16).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-.5, .5)
            m.running_var.uniform_(.5, 1.5)
            m.weight.data.uniform_(.5, 1.5)
            m.bias.data.uniform_(-.5, .5)
    blob, _ = build_blob(model_ops(model, 32), ["い", "あ", "〓"], 32)
    path = tmp_path / "test.vocr"
    path.write_bytes(blob)
    recovered, ck = restore(path, width=.25, bottleneck=16)
    assert ck["vocab"] == ["い", "あ", "〓"] and ck["freeze_bn_stats"]
    runtime = Vocr(blob, mode="f32")
    g = np.random.default_rng(1).integers(0, 256, (32, 80), dtype=np.uint8)
    _, logits = runtime.run(g, logits=True)
    with torch.no_grad():
        x = torch.from_numpy(g.astype(np.float32) / 127.5 - 1)[None, None]
        np.testing.assert_allclose(recovered(x).squeeze(1).numpy(), logits, rtol=1e-4, atol=1e-4)
    recovered.train()
    freeze_bn_stats(recovered)
    before = {k: v.clone() for k, v in recovered.state_dict().items() if "running_" in k}
    recovered(x).sum().backward()
    assert all(torch.equal(v, recovered.state_dict()[k]) for k, v in before.items())
    assert recovered.features[0][1].bias.grad is not None
    with pytest.raises(ValueError, match="architecture"):
        restore(path)
    int8_blob, _ = build_blob(model_ops(model, 32), ["い", "あ", "〓"], 32, int8=True)
    path.write_bytes(int8_blob)
    with pytest.raises(ValueError, match="fp32"):
        restore(path, width=.25, bottleneck=16)

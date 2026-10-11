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


def test_pair_shards_cap_rare_exposure_and_return_missing_bands_to_common():
    from collections import Counter
    schedule = render.pair_schedule(np.random.default_rng(1), 250, {0: ["日目"], 1: ["己巳"],
                                                                  2: ["己已"], 3: ["寝寢"]})
    assert Counter(schedule) == {0: 213, 1: 25, 2: 10, 3: 2}
    assert Counter(render.pair_schedule(np.random.default_rng(1), 250, {0: ["日目"], 2: ["己已"]})) == {
        0: 240, 2: 10}


def test_generated_names_keep_common_kanji_dominant(monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    monkeypatch.setattr(render, "R", SimpleNamespace(katakana=list("アカ"), hiragana=list("あか"),
                                                     name_kanji=list("日寢"), name_probs=[.999, .001]))
    rng, counts = np.random.default_rng(9), Counter()
    for _ in range(2000):
        _, label = render.random_name(rng)
        counts.update(label)
    assert counts["日"] > 1000 and 0 < counts["寢"] < counts["日"] / 100


def test_crop_edges_preserve_ink_and_exclude_adjacent_rows():
    line = {"fs": 16, "ix0": 20, "ix1": 80, "y0": 20, "y1": 36}
    before, after = {"y0": 1, "y1": 15}, {"y0": 41, "y1": 56}
    rng, boxes = np.random.default_rng(21), set()
    for _ in range(100):
        x, y, w, h = render.edge_box((16, 16, 68, 24), line, [before, line, after], (60, 100), rng)
        assert x <= 20 and x + w >= 80 and y <= 20 and y + h >= 36
        assert y >= 15 and y + h <= 41 and x >= 0 and x + w <= 100
        boxes.add((x, y, w, h))
    assert len(boxes) > 10


def test_crop_edge_jitter_falls_back_when_fractional_neighbours_cannot_be_separated():
    line = {"fs": 16, "ix0": 20, "ix1": 80, "y0": 20.3, "y1": 36.3}
    before, after = {"y0": 1, "y1": 20.2}, {"y0": 36.4, "y1": 56}
    original = (16, 16, 68, 24)
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert render.edge_box(original, line, [before, line, after], (60, 100), rng) == original


@pytest.mark.parametrize("failure", ["no-region", "empty-extract", "wrong-target"])
def test_unrenderable_pair_endpoint_fails_after_bounded_attempts(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    resources = SimpleNamespace(kanji_pairs=True, pair_bands={0: ["人入"]}, kanji_policy=None,
                                crop_edge_jitter=False, target_kana=set(), active_kanji=None)
    monkeypatch.setattr(render, "R", resources)
    monkeypatch.setattr(render, "MAX_PAIR_ATTEMPTS", 3)
    monkeypatch.setattr(render, "sample_pair", lambda *args, **kwargs: ("人", "入"))
    attempts = []
    pixels = np.zeros((32, 64, 3), np.uint8)

    def region(rng):
        attempts.append(resources.active_kanji)
        return None if failure == "no-region" else (pixels, [], None, {})

    monkeypatch.setattr(render, "render_region", region)
    monkeypatch.setattr(render, "extract", lambda *args, **kwargs: [] if failure == "empty-extract" else [
        (pixels[:, :, 0], "日", (0, 0, 64, 32), {"kind": "dialogue"}, 0)])
    with pytest.raises(RuntimeError, match="pair 人入 endpoint 人 after 3 attempts"):
        render.run_shard((0, 2, 1, str(tmp_path), "grad", True))
    assert attempts == ["人", "人", "人"]
    assert not list(tmp_path.glob("*.npz"))


def test_failed_pair_crops_keep_endpoint_and_success_resets_attempt_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    resources = SimpleNamespace(kanji_pairs=True, pair_bands={0: ["人入"]}, kanji_policy=None,
                                crop_edge_jitter=False, target_kana=set(), active_kanji=None)
    monkeypatch.setattr(render, "R", resources)
    monkeypatch.setattr(render, "MAX_PAIR_ATTEMPTS", 3)
    monkeypatch.setattr(render, "sample_pair", lambda *args, **kwargs: ("人", "入"))
    pixels = np.zeros((32, 64, 3), np.uint8)
    monkeypatch.setattr(render, "render_region", lambda rng: (pixels, [], None, {}))
    monkeypatch.setattr(render, "crop_meta", lambda *args, **kwargs: {})
    attempts = []

    def extract(*args, **kwargs):
        attempts.append(resources.active_kanji)
        if len(attempts) in (1, 3, 4):
            return []
        return [(pixels[:, :, 0], resources.active_kanji, (0, 0, 64, 32), {"kind": "dialogue"}, 0)]

    monkeypatch.setattr(render, "extract", extract)
    render.run_shard((0, 2, 1, str(tmp_path), "grad", True))
    assert attempts == ["人", "人", "入", "入", "入"]
    with np.load(tmp_path / "synth_00000.npz") as shard:
        labels = json.loads(str(shard["labels"]))
        metadata = json.loads(str(shard["meta"]))
    assert labels == ["人", "入"]
    assert [row["kanji_target"] for row in metadata] == labels
    assert [row["kanji_pair"] for row in metadata] == ["人入", "人入"]
    assert [row["kanji_band"] for row in metadata] == [0, 0]


def test_pair_endpoints_ignore_generated_names_and_accept_natural_dialogue(tmp_path, monkeypatch):
    from types import SimpleNamespace
    resources = SimpleNamespace(kanji_pairs=True, pair_bands={0: ["人入"]}, kanji_policy=None,
                                crop_edge_jitter=False, target_kana=set(), active_kanji=None)
    monkeypatch.setattr(render, "R", resources)
    monkeypatch.setattr(render, "sample_pair", lambda *args, **kwargs: ("人", "入"))
    pixels = np.zeros((32, 64, 3), np.uint8)
    monkeypatch.setattr(render, "render_region", lambda rng: (pixels, [], None, {}))
    monkeypatch.setattr(render, "crop_meta", lambda meta, line, *args: {"kind": line["kind"]})
    targets = []

    def extract(*args, **kwargs):
        target = resources.active_kanji
        targets.append(target)
        return [(np.full((32, 64), 1, np.uint8), target, (0, 0, 64, 32), {"kind": "name"}, 0),
                (np.full((32, 64), 200, np.uint8), target, (0, 0, 64, 32), {"kind": "dialogue"}, 0)]

    monkeypatch.setattr(render, "extract", extract)
    render.run_shard((0, 2, 1, str(tmp_path), "grad", True))
    with np.load(tmp_path / "synth_00000.npz") as shard:
        labels = json.loads(str(shard["labels"]))
        metadata = json.loads(str(shard["meta"]))
        assert (shard["data"] == 200).all()
    assert targets == labels == ["人", "入"]
    assert [row["kind"] for row in metadata] == ["dialogue", "dialogue"]


@pytest.mark.parametrize("enabled, draw, changed", [
    (False, False, False), (True, False, False), (True, True, False), (True, True, True)])
def test_extraction_records_actual_jitter_and_preserves_layout(monkeypatch, enabled, draw, changed):
    from types import SimpleNamespace
    pixels = np.zeros((40, 40, 3), np.uint8)
    line = {"kind": "dialogue", "text": "人", "fs": 16, "ix0": 8, "ix1": 24, "y0": 8, "y1": 24}
    original_line = dict(line)
    original_box, moved_box = (4, 4, 24, 24), (2, 2, 28, 28)
    monkeypatch.setattr(render, "R", SimpleNamespace(crop_edge_jitter=enabled, policy_sha256=None))
    monkeypatch.setattr(render, "find_lines_ex", lambda *args: ([original_box], [0]))
    edge_calls = []

    def edge(box, *args):
        edge_calls.append(box)
        return moved_box if changed else box

    monkeypatch.setattr(render, "edge_box", edge)
    rng = SimpleNamespace(random=lambda: 0 if draw else 1)
    crops = render.extract(pixels, [line], finder="grad", rng=rng, edge_jitter=enabled)
    assert len(crops) == 1
    crop, label, box, derived_line, dark = crops[0]
    expected = moved_box if changed else original_box
    assert box == expected and crop.shape == (expected[3], expected[2]) and label == "人"
    assert line == original_line and derived_line is not line
    assert len(edge_calls) == int(enabled and draw)
    metadata = render.crop_meta({}, derived_line, dark)
    assert metadata["crop_edge_jitter_enabled"] is enabled
    assert metadata["crop_edge_jitter_applied"] is (enabled and draw)
    assert metadata["crop_edge_jitter_changed"] is changed
    assert metadata["original_crop_box"] == list(original_box)
    assert metadata["crop_box"] == list(expected)


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
    assert HG.check(base, None, candidate, blob)["accepted"]
    candidate["sets"]["psp-test"]["all"]["errors"] = 11
    assert not HG.check(base, base, candidate, blob)["accepted"]
    candidate["datasets"]["clannad-test"]["split"] = "dev"
    with pytest.raises(ValueError, match="held-out"):
        HG.check(base, base, candidate, blob)
    candidate = copy.deepcopy(base)
    assert not HG.check(base, base, candidate, blob)["accepted"]
    assert HG.check(base, None, candidate, blob)["failures"] == ["CLANNAD hiragana does not improve over baseline"]
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

"""Fresh fine-tuning provenance and separation, using generated pixels only."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import hiragana_fresh as HF  # noqa: E402
import kanji_policy as KP  # noqa: E402
from model import TinyVNOCR  # noqa: E402


def shard(path, text="あい", fill=40, meta=None, inverted=False):
    original = np.full((32, 64), fill, np.uint8)
    crop = 255 - original if inverted else original
    np.savez(path, data=crop.ravel(), shapes=np.asarray([crop.shape], np.int32),
             offsets=np.asarray([0, crop.size], np.int64), labels=json.dumps([text]),
             meta=json.dumps([meta or {}]))
    return hashlib.sha256(np.asarray(crop.shape, "<i4").tobytes() + original.tobytes()).hexdigest()


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    paths = {name: tmp_path / f"{name}.npz" for name in (*HF.SOURCES, "dev", "synthetic_dev")}
    shard(paths["general"], meta={"split": "train"})
    shard(paths["targeted"], "っつ", 55, {"split": "train"})
    digest = hashlib.sha256(np.asarray([32, 64], "<i4").tobytes()
                            + np.full((32, 64), 90, np.uint8).tobytes()).hexdigest()
    anime_meta = {"split": "train", "crop_id": "source-1", "image_id": "source",
                  "inverted": True, "pixel_sha256": digest}
    shard(paths["anime"], "え", 90, anime_meta, inverted=True)
    shard(paths["dev"], "ずっ", 111, {"split": "dev", "hand_checked": True,
                                     "session": "clannad-dev", "image_sha256": "capture-1"})
    shard(paths["synthetic_dev"], "あい", 149, {"split": "dev"})
    manifest = tmp_path / "anime.jsonl"
    record = {**anime_meta, "text": "え", "shard": paths["anime"].name, "index": 0}
    manifest.write_text(json.dumps(record) + "\n")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"checked": 200, "approved_training_manifest_sha256": HF.sha256(manifest)}))
    vocab = ["あ", "い", "ず", "っ", "つ", "え", "土", "士", "本", "〓"]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    jiten, similar = tmp_path / "jiten.csv", tmp_path / "similar.txt"
    jiten.write_text("Kanji,Rank\n土,454\n士,281\n")
    similar.write_text("土/士/\n士/土/\n")
    monkeypatch.setattr(KP, "JITEN_SHA256", KP.sha256(jiten))
    monkeypatch.setattr(KP, "SIMILAR_SHA256", KP.sha256(similar))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(KP.build_policy(jiten, similar, vocab_path)))
    shard(paths["kanji"], "土", 172, {"split": "train", "kanji_pairs": True, "kanji_pair": "土士", "kanji_target": "土",
                                     "kanji_policy_sha256": HF.sha256(policy_path)})
    kanji_manifest = tmp_path / "snapshot.json"
    kanji_manifest.write_text(json.dumps({"schema": 2, "shards": [{"source": "kanji", "split": "train",
        "file": paths["kanji"].name, "sha256": HF.sha256(paths["kanji"]),
        "bytes": paths["kanji"].stat().st_size, "samples": 1}]}))
    config = {**{name: str(path) for name, path in paths.items()}, "anime_manifest": str(manifest),
              "anime_audit": str(audit), "provenance": {"new_sources": True},
              "kanji_policy": str(policy_path), "kanji_manifest": str(kanji_manifest)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    init = tmp_path / "init.pt"
    model = TinyVNOCR(len(vocab) + 1)
    torch.save({"model": model.state_dict(), "vocab": vocab, **HF.FL10_ARCH}, init)
    return config, config_path, init, vocab_path


def test_fresh_matrix_uses_matched_budget_and_development_selection():
    config = {name: f"/private/{name}.npz" for name in (*HF.SOURCES, "dev", "synthetic_dev")}
    commands = HF.commands(config, "/private/init.pt", "/private/vocab.txt", "/private/out")
    assert len(commands) == 3
    for command, mix in zip(commands, HF.MIXES.values()):
        for flag, value in {"--steps": "1000", "--batch": "32", "--lr": "0.00002",
                            "--warmup": "50", "--threads": "6", "--seed": "1",
                            "--eval-every": "100", "--kd-weight": "0", "--init": "/private/init.pt",
                            "--select": "clannad", "--select-metric": "hiragana"}.items():
            assert command[command.index(flag) + 1] == value
        assert "--freeze-bn-stats" in command and "--keep-evals" in command
        assert [command[i + 1] for i, flag in enumerate(command) if flag == "--val"] == [
            "clannad:/private/dev.npz", "synthetic:/private/synthetic_dev.npz"]
        sources = [command[i + 1].split(":") for i, flag in enumerate(command) if flag == "--src"]
        assert [(s[0], float(s[1])) for s in sources] == [
            (name, weight) for name, weight in zip(HF.SOURCES, mix) if weight]
        assert sum(float(s[1]) for s in sources) == pytest.approx(1.0)
        assert all(s[3] == "inv=0.2" for s in sources)
        assert next(float(s[1]) for s in sources if s[0] == "kanji") == .02


def test_valid_fresh_inputs_record_hashes_and_refuse_changed_snapshot(fresh, tmp_path):
    config, config_path, init, vocab = fresh
    out = tmp_path / "runs"
    identity = HF.prepare(config_path, init, vocab, out)
    assert identity == json.loads((out / "experiment.json").read_text())
    assert identity == HF.prepare(config_path, init, vocab, out)
    for path, info in identity["inputs"].items():
        assert info["sha256"] == HF.sha256(path) and info["bytes"] == Path(path).stat().st_size
    assert identity["inputs"][str(Path(config["synthetic_dev"]).resolve())]["role"] == "dev"
    assert "train/train2.py" in identity["code_sha256"]
    assert "requirements-ci.lock" in identity["dependency_sha256"]
    assert identity["sampling"]["kanji_corrective_probability"] == .02
    assert identity["inputs"][config["kanji_policy"]]["role"] == "provenance"
    assert "approximate" in identity["reproducibility"]["checkpoint_resume"]
    assert "locked" in identity["selection"]["held_out"]
    shard(Path(config["general"]), fill=41, meta={"split": "train"})
    with pytest.raises(ValueError, match="another configuration or input"):
        HF.prepare(config_path, init, vocab, out)


def test_missing_source_and_training_evaluation_roles_fail(fresh):
    config = fresh[0]
    with pytest.raises(ValueError, match="missing training source"):
        HF.check_data({**config, "general": str(Path(config["general"]).with_name("absent*.npz"))})
    shard(Path(config["general"]), meta={"split": "test"})
    with pytest.raises(ValueError, match="evaluation crops in training"):
        HF.check_data(config)


def test_same_training_shard_cannot_enter_two_sources(fresh):
    config = fresh[0]
    with pytest.raises(ValueError, match="two training sources"):
        HF.check_data({**config, "targeted": config["general"]})


@pytest.mark.parametrize("shared", [{"session": "clannad-dev"}, {"image_sha256": "capture-1"}])
def test_development_session_and_image_overlap_fail(fresh, shared):
    config = fresh[0]
    shard(Path(config["general"]), meta={"split": "train", **shared})
    with pytest.raises(ValueError, match="overlap training"):
        HF.check_data(config)


def test_copied_pixels_cannot_masquerade_as_synthetic_development(fresh):
    config = fresh[0]
    shard(Path(config["synthetic_dev"]), meta={"split": "dev"})
    with pytest.raises(ValueError, match="pixels overlap training"):
        HF.check_data(config)


@pytest.mark.parametrize("role", [None, "val", "train", "test"])
def test_synthetic_development_requires_explicit_dev_role(fresh, role):
    config = fresh[0]
    shard(Path(config["synthetic_dev"]), fill=149, meta={"split": role})
    with pytest.raises(ValueError, match="invalid dev split"):
        HF.check_data(config)


def test_development_sets_cannot_alias_the_same_pixels(fresh):
    config = fresh[0]
    shard(Path(config["synthetic_dev"]), fill=111, meta={"split": "dev"})
    with pytest.raises(ValueError, match="development sources overlap each other"):
        HF.check_data(config)


def test_labels_outside_the_checkpoint_vocabulary_are_rejected(fresh, tmp_path):
    config, config_path, init, vocab = fresh
    shard(Path(config["general"]), "漢", meta={"split": "train"})
    with pytest.raises(ValueError, match="exact FL10 vocabulary"):
        HF.prepare(config_path, init, vocab, tmp_path / "runs")


def test_even_existing_vocabulary_kanji_require_jiten_membership(fresh):
    config = fresh[0]
    shard(Path(config["general"]), "本", meta={"split": "train"})
    with pytest.raises(ValueError, match="outside the Jiten kanji policy"):
        HF.check_data(config)


def test_corrective_kanji_source_is_required(fresh):
    config = fresh[0]
    with pytest.raises(ValueError, match="missing fresh experiment keys"):
        HF.check_data({k: v for k, v in config.items() if k != "kanji"})


@pytest.mark.parametrize("field,value", [("sha256", "wrong"), ("bytes", 1), ("samples", 2)])
def test_corrective_snapshot_pins_actual_shards(fresh, field, value):
    config = fresh[0]
    path = Path(config["kanji_manifest"])
    snapshot = json.loads(path.read_text())
    snapshot["shards"][0][field] = value
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="hash, size or crop count differs"):
        HF.check_data(config)


@pytest.mark.parametrize("change", [{"kanji_policy_sha256": "other"}, {"kanji_pairs": False},
                                    {"kanji_pair": "土本"}, {"kanji_target": "士"}, {"kanji_target": None}])
def test_corrective_crops_bind_policy_and_supported_pairs(fresh, change):
    config = fresh[0]
    path = Path(config["kanji"])
    meta = {"split": "train", "kanji_pairs": True, "kanji_pair": "土士", "kanji_target": "土",
            "kanji_policy_sha256": HF.sha256(config["kanji_policy"]), **change}
    shard(path, "土", 172, meta)
    snapshot_path = Path(config["kanji_manifest"])
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["shards"][0].update(sha256=HF.sha256(path), bytes=path.stat().st_size)
    snapshot_path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="exact policy identity|supported pair endpoint"):
        HF.check_data(config)


def test_clannad_development_requires_handchecked_capture_metadata(fresh):
    config = fresh[0]
    shard(Path(config["dev"]), "ずっ", 111, {"split": "dev", "session": "clannad-dev"})
    with pytest.raises(ValueError, match="hand-checked"):
        HF.check_data(config)


@pytest.mark.parametrize("change,message", [({"checked": 199}, "200-crop audit"),
    ({"approved_training_manifest_sha256": "wrong"}, "does not cover")])
def test_anime_audit_size_and_identity_are_required(fresh, change, message):
    config = fresh[0]
    path = Path(config["anime_audit"])
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(ValueError, match=message):
        HF.check_data(config)


def test_audited_manifest_must_cover_actual_labels_and_metadata(fresh):
    config = fresh[0]
    manifest = Path(config["anime_manifest"])
    record = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**record, "text": "あ"}) + "\n")
    audit = Path(config["anime_audit"])
    audit.write_text(json.dumps({"checked": 200, "approved_training_manifest_sha256": HF.sha256(manifest)}))
    with pytest.raises(ValueError, match="labels or metadata differ"):
        HF.check_data(config)


def test_changed_anime_pixels_fail_even_if_original_manifest_is_unchanged(fresh):
    config = fresh[0]
    meta = json.loads(Path(config["anime_manifest"]).read_text())
    shard(Path(config["anime"]), "え", 91, {k: v for k, v in meta.items()
                                            if k not in {"text", "shard", "index"}}, inverted=True)
    with pytest.raises(ValueError, match="pixels differ"):
        HF.check_data(config)


@pytest.mark.parametrize("change", ["vocab", "architecture"])
def test_initialization_requires_exact_class_order_and_architecture(fresh, tmp_path, change):
    _, config, init, vocab = fresh
    if change == "vocab":
        lines = vocab.read_text().splitlines()
        vocab.write_text("\n".join(reversed(lines)) + "\n")
    else:
        ck = torch.load(init, weights_only=False)
        torch.save({**ck, "ctx": 1}, init)
    with pytest.raises(ValueError, match="vocabulary|architecture"):
        HF.prepare(config, init, vocab, tmp_path / "runs")

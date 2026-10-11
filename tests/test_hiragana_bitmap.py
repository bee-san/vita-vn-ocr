"""Frozen bitmap recipe safeguards, without native game assets or corpora."""
import json

import pytest

import hiragana_bitmap as HB


@pytest.fixture
def snapshot(tmp_path):
    shard = tmp_path / "train.npz"
    shard.write_bytes(b"generated snapshot identity fixture")
    files = {str(shard): {"role": "train", "source": "bitmap_general", "crops": 3}}
    row = {"source": "general", "split": "train", "file": shard.name,
           "sha256": HB.sha256(shard), "bytes": shard.stat().st_size, "samples": 3}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": 2, "shards": [row]}))
    return shard, manifest, row, files


@pytest.mark.parametrize("field,value", [("split", "test"), ("sha256", "0" * 64),
                                         ("samples", 2), ("bytes", 1)])
def test_snapshot_cannot_change_role_bytes_or_count(snapshot, field, value):
    _, manifest, row, files = snapshot
    row[field] = value
    manifest.write_text(json.dumps({"schema": 2, "shards": [row]}))
    with pytest.raises(ValueError, match="snapshot"):
        HB.bind_snapshot(manifest, {"general": "bitmap_general"}, files)


def test_snapshot_requires_all_and_only_declared_shards(snapshot):
    shard, manifest, row, files = snapshot
    HB.bind_snapshot(manifest, {"general": "bitmap_general"}, files)
    manifest.write_text(json.dumps({"schema": 2, "shards": [row, row]}))
    with pytest.raises(ValueError, match="snapshot"):
        HB.bind_snapshot(manifest, {"general": "bitmap_general"}, files)
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot"):
        HB.bind_snapshot(manifest, {"general": "bitmap_general"}, files)


def test_recipe_preserves_bitmap_ink_and_bounds_real_and_kanji_replay():
    config = {key: f"fixture-{key}.npz" for key in (*HB.MIX, *HB.DEV_KEYS)}
    commands = HB.commands(config, "init.pt", "vocab.txt", "out", python="python")
    assert len(commands) == 2
    for scope, command in zip(("head", "all"), commands):
        assert command[command.index("--train-scope") + 1] == scope
        sources = [command[i + 1] for i, value in enumerate(command) if value == "--src"]
        assert sources == [f"{key}:{weight}:fixture-{key}.npz:inv=0" +
                           (",aug=0" if key in ("bitmap_general", "bitmap_targeted", "native") else "")
                           for key, weight in HB.MIX.items()]
        assert sum(HB.MIX.values()) == 1 and HB.MIX["kanji"] == HB.MIX["native"] == .02
        assert not any("test" in value for value in command)

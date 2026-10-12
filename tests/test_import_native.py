"""Native training labels require pinned, nonreserved, visually checked sources."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kanji_policy as KP  # noqa: E402
from data import fresh_backgrounds as FB  # noqa: E402
from data import import_native as N  # noqa: E402
from hiragana_native import native_inputs  # noqa: E402


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "exclusion_sets", lambda: (set(), set()))
    monkeypatch.setattr(FB, "is_test_vn", lambda _: False)
    csv, graph, vocab, policy = [tmp_path / name for name in ("kanji.csv", "graph.txt", "vocab.txt", "policy.json")]
    csv.write_text("Kanji,Rank\n人,1\n入,2\n")
    graph.write_text("人/入/\n")
    vocab.write_text("人\n入\nあ\n")
    monkeypatch.setattr(KP, "JITEN_SHA256", KP.sha256(csv))
    monkeypatch.setattr(KP, "SIMILAR_SHA256", KP.sha256(graph))
    policy.write_text(json.dumps(KP.build_policy(csv, graph, vocab), ensure_ascii=False))
    records = []
    for i, (platform, split) in enumerate((("psp", "train"), ("psp", "dev"), ("psv", "train"), ("psv", "dev")), 10):
        image = Image.new("RGB", FB.PLATFORMS[platform][0], (i, 2*i, 3*i))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG")
        blob = buffer.getvalue()
        record = {"vn": f"v{i}", "url": f"https://t.vndb.org/sf/00/{i}.jpg", "release": f"r{i}",
                  "release_platforms": [platform], "dims": list(image.size), "sexual": 0, "violence": 0,
                  "platform": platform, "split": split, "bytes": len(blob),
                  "scenery_box": [0, 0, image.width, 100],
                  "scenery_review": {"hand_checked": True, "reason": "generated fixture"}}
        folder = "assets" if split == "train" else "assets-dev"
        record["file"] = f"{folder}/{FB.PLATFORMS[platform][1]}/shots/v{i}_{i}.jpg"
        path = tmp_path / record["file"]
        path.parent.mkdir(parents=True)
        path.write_bytes(blob)
        record["sha256"] = N.sha256(path)
        records.append(record)
    doc = {"schema": 2, "counts": {"train": 1, "dev": 1}, "records": records}
    doc["content_sha256"] = FB.canonical_sha(doc)
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(doc))
    annotations = tmp_path / "annotations.jsonl"
    row = {"id": "own-line", "vn": "v10", "box": [10, 150, 80, 20],
           "text": "あ人", "split": "train", "hand_checked": True}
    annotations.write_text(json.dumps(row, ensure_ascii=False)+"\n")
    return tmp_path, sources, annotations, vocab, policy, row


def test_native_snapshot_contains_actual_pixels_and_hashes_without_line_text(inputs):
    root, sources, annotations, vocab, policy, _ = inputs
    report = N.pack(root, sources, annotations, vocab, policy, root / "out")
    assert report["kept_lines"] == 1
    assert report["source_images"][0]["split"] == "train"
    assert "text" not in report
    with np.load(root / "out/native_00000.npz") as shard:
        meta = json.loads(str(shard["meta"]))[0]
        crop = shard["data"].reshape(shard["shapes"][0])
        original = 255-crop if meta["inverted"] else crop
        assert N.pixel_sha(original) == meta["pixel_sha256"]


@pytest.mark.parametrize("change", [{"hand_checked": False}, {"split": "dev"}, {"vn": "v11"},
                                    {"box": [-1, 150, 80, 20]}, {"box": [10, 150, 800, 20]}])
def test_uncertain_development_or_clipped_native_lines_are_rejected(inputs, change):
    root, sources, annotations, vocab, policy, row = inputs
    annotations.write_text(json.dumps({**row, **change}, ensure_ascii=False)+"\n")
    with pytest.raises(ValueError):
        N.pack(root, sources, annotations, vocab, policy, root / "out")


def test_native_source_byte_changes_are_rejected(inputs):
    root, sources, annotations, vocab, policy, _ = inputs
    path = root / "assets/vndb_psp/shots/v10_10.jpg"
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="pinned source"):
        N.pack(root, sources, annotations, vocab, policy, root / "out")


@pytest.mark.parametrize("change", [None, "role", "hash", "alias"])
def test_native_trial_binds_the_snapshot_and_rejects_role_hash_or_source_alias_changes(inputs, change):
    root, sources, annotations, vocab, policy, _ = inputs
    out = root / "out"
    N.pack(root, sources, annotations, vocab, policy, out)
    manifest = out / "manifest.json"
    config = {"native_manifest": str(manifest), "native_sources": str(sources),
              "native": str(out / "*.npz"), "kanji_policy": str(policy)}
    base = {}
    if change:
        doc = json.loads(manifest.read_text())
        if change == "role":
            doc["role"] = "dev"
        elif change == "hash":
            doc["shards"][0]["sha256"] = "0" * 64
        else:
            base[str((out / "native_00000.npz").resolve())] = {"role": "train"}
        manifest.write_text(json.dumps(doc))
        with pytest.raises(ValueError):
            native_inputs(config, base)
    else:
        assert native_inputs(config, base)[str((out / "native_00000.npz").resolve())]["crops"] == 1

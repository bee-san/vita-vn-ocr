"""Native background provenance/cohort guards, using generated JPEGs only."""
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import fresh_backgrounds as FB  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic_holdout(monkeypatch):
    monkeypatch.setattr(FB, "is_test_vn", lambda _: False)


def jpeg(platform="psp", fill=70, dims=None):
    buffer = io.BytesIO()
    Image.new("RGB", dims or FB.PLATFORMS[platform][0], (fill, fill + 1, fill + 2)).save(buffer, "JPEG")
    return buffer.getvalue()


def row(index=10, platform="psp", split="train"):
    blob = jpeg(platform, index)
    result = {"vn": f"v{index}", "title": "generated fixture", "url": f"https://t.vndb.org/sf/00/{index}.jpg",
              "release": f"r{index}", "release_platforms": [platform], "dims": list(FB.PLATFORMS[platform][0]),
              "sexual": 0.0, "violence": 0.0, "platform": platform, "split": split,
              "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    base = "assets" if split == "train" else "assets-dev"
    result["file"] = f"{base}/{FB.PLATFORMS[platform][1]}/shots/{FB.filename(result)}"
    return result


def seal(manifest):
    manifest.pop("content_sha256", None)
    manifest["content_sha256"] = FB.canonical_sha(manifest)
    return manifest


@pytest.fixture
def manifest():
    return seal({"schema": 1, "counts": {"train": 1, "dev": 1}, "records": [
        row(10), row(11, split="dev"), row(12, "psv"), row(13, "psv", "dev")]})


@pytest.mark.parametrize("change", [{"vn": "v4"}, {"vn": "v11"}, {"release": "r40"},
                                    {"dims": [640, 480]}, {"release_platforms": ["win"]},
                                    {"url": "https://t.vndb.org/st/00/10.jpg"},
                                    {"url": "https://example.com/sf/00/10.jpg"},
                                    {"release": None}, {"release_platforms": None}, {"dims": None},
                                    {"sexual": .3}, {"violence": .5}])
def test_only_native_console_nonreserved_screenshots_are_eligible(change):
    assert not FB.eligible(dict(row(), **change), "psp", {"v11"}, {"r40"})


def test_filename_and_deterministic_benchmark_holdout_are_excluded(monkeypatch):
    assert not FB.eligible(row(), "psp", {"v10_10.jpg"}, set())
    monkeypatch.setattr(FB, "is_test_vn", lambda _: True)
    assert not FB.eligible(row(), "psp", set(), set())


def test_selection_is_stable_and_one_per_game_and_release():
    rows = [row(10), row(11), row(12), row(13)]
    duplicate = dict(rows[0], url="https://t.vndb.org/sf/00/20.jpg")
    shared_release = dict(row(14), release="r11")
    first = FB.choose(rows + [duplicate, shared_release], "psp", 3, set(), set(), ["v12"], ["r13"])
    second = FB.choose(list(reversed(rows + [duplicate, shared_release])), "psp", 3, set(), set(), ["v12"], ["r13"])
    assert first == second
    assert len(first) == 2
    assert len({x["vn"] for x in first}) == 2
    assert len({x["release"] for x in first}) == 2
    assert all(x["vn"] != "v12" and x["release"] != "r13" for x in first)


def test_manifest_has_separate_games_at_native_paths(manifest):
    assert FB.validate_manifest(manifest, set(), set()) is manifest


@pytest.mark.parametrize("identity", ["vn", "release", "sha256"])
def test_manifest_rejects_cohort_identity_overlap(manifest, identity):
    train, dev = manifest["records"][:2]
    dev[identity] = train[identity]
    if identity == "vn":
        dev["file"] = f"assets-dev/vndb_psp/shots/{FB.filename(dev)}"
    with pytest.raises(ValueError, match="identity is repeated"):
        FB.validate_manifest(seal(manifest), set(), set())


def test_manifest_rejects_changed_content_and_path_escape(manifest):
    manifest["records"][0]["title"] = "changed"
    with pytest.raises(ValueError, match="content hash changed"):
        FB.validate_manifest(manifest, set(), set())
    manifest["records"][0]["file"] = "../../outside.jpg"
    with pytest.raises(ValueError, match="exact cohort/platform path"):
        FB.validate_manifest(seal(manifest), set(), set())


def test_manifest_counts_are_not_silently_partial(manifest):
    manifest["records"].pop()
    with pytest.raises(ValueError, match="cohort counts"):
        FB.validate_manifest(seal(manifest), set(), set())


@pytest.mark.parametrize("kind", ["hash", "bytes", "dimensions"])
def test_download_rejects_changed_or_non_native_images(tmp_path, monkeypatch, kind):
    record = row()
    blob = jpeg(fill=10)
    if kind == "hash":
        record["sha256"] = "0" * 64
    elif kind == "bytes":
        record["bytes"] += 1
    else:
        blob = jpeg(fill=10, dims=(640, 480))
        record["sha256"], record["bytes"] = hashlib.sha256(blob).hexdigest(), len(blob)
    session = type("Session", (), {"get": lambda *_a, **_k: type("Response", (), {
        "content": blob, "raise_for_status": lambda _: None})()})()
    monkeypatch.setattr(FB.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="mismatch|native console JPEG"):
        FB.download(session, record, tmp_path)
    assert not (tmp_path / record["file"]).exists()


def test_replay_verifies_cached_bytes_without_network(tmp_path):
    record = row()
    path = tmp_path / record["file"]
    path.parent.mkdir(parents=True)
    path.write_bytes(jpeg(fill=10))
    assert FB.download(None, record, tmp_path) == record
    path.write_bytes(jpeg(fill=11))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        FB.download(None, record, tmp_path)


def test_query_cache_is_bound_to_exact_request(tmp_path):
    cache = tmp_path / "query.json"
    response = {"results": [], "more": False}
    cache.write_text(json.dumps({"request": {"results": 100}, "response": response}))
    queries = []
    assert FB.query(None, {"results": 100}, cache, queries) == response
    assert queries[0]["response_sha256"] == FB.canonical_sha(response)
    with pytest.raises(ValueError, match="different query identity"):
        FB.query(None, {"results": 50}, cache, [])

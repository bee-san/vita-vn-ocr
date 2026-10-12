"""Pinned fresh-data rebuilding and synthetic development isolation."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
import fresh_dataset as F  # noqa: E402


def corpus_fixture(root, count=40):
    path = root / "assets/corpus/aozora/aozorabunko-dedupe-clean.jsonl.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(count):
            row = {"meta": {"作品ID": str(i), "文字遣い種別": "新字新仮名"},
                   "text": "「こんにちは。いい天気ですね。」\n今日は晴れていました。"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(sorted(set(row["text"]) - {"\n"})) + "\n", encoding="utf-8")
    return vocab


def test_book_cohorts_and_natural_pools_are_reproducible(tmp_path):
    vocab = corpus_fixture(tmp_path)
    caps = {s: {"dialogue": 4, "narration": 4} for s in ("train", "dev")}
    first = F.build_pools(tmp_path, vocab, caps=caps)
    second = F.build_pools(tmp_path, vocab, caps=caps)
    assert first["pool_records"] == second["pool_records"]
    assert first["modern_kana_books"] == 40
    cohorts = {role: set(json.loads((tmp_path / folder / "corpus/book_ids.json").read_text())["book_ids"])
               for role, folder in (("train", "assets"), ("dev", "assets-dev"))}
    assert cohorts["train"] and cohorts["dev"]
    assert not cohorts["train"] & cohorts["dev"]
    assert set.union(*cohorts.values()) == {str(i) for i in range(40)}
    assert all(F.book_split(i, 20261011) == role for role, ids in cohorts.items() for i in ids)
    assert "wiki" not in F.POOL_WEIGHTS


def test_cached_assets_are_hash_checked(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"correct pinned bytes")
    sha = F.sha256(path)
    F.download("https://unused.invalid", path, sha, path.stat().st_size)
    path.write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="size/hash mismatch"):
        F.download("https://unused.invalid", path, sha)


def synthetic_shard(root, name, role, cohort_sha, value=40):
    path = root / name / "synth_00000.npz"
    path.parent.mkdir(parents=True)
    meta = [{"split": role, "source_kind": "synthetic", "book_split_sha256": cohort_sha,
             "crop_edge_jitter_requested": True, "crop_edge_jitter_enabled": True,
             "crop_edge_jitter_applied": True, "crop_edge_jitter_changed": False}]
    np.savez_compressed(path, data=np.full(16 * 40, value, np.uint8), shapes=np.array([[16, 40]], np.int32),
                        offsets=np.array([0, 16 * 40], np.int64), labels=np.array(json.dumps(["こんにちは"])),
                        meta=np.array(json.dumps(meta)))
    return path


def test_manifest_rejects_role_changes_and_pixel_overlap(tmp_path):
    vocab = corpus_fixture(tmp_path)
    pools = F.build_pools(tmp_path, vocab, caps={s: {"dialogue": 4, "narration": 4} for s in ("train", "dev")})
    F.write_json(tmp_path / "assets_manifest.json", {"assets": []})
    cohorts = {r["split"]: r["sha256"] for r in pools["pool_records"] if r.get("role") == "source_book_cohort"}
    sources = {"general": "train", "synthetic-dev": "dev"}
    synthetic_shard(tmp_path, "general", "train", cohorts["train"])
    dev = synthetic_shard(tmp_path, "synthetic-dev", "dev", cohorts["dev"], value=41)
    result = F.dataset_manifest(tmp_path, sources, vocab)
    assert result["train_dev_pixel_intersection"] == 0
    assert result["summary"]["general"]["samples"] == 1
    assert result["summary"]["general"]["crop_edge_jitter_enabled"] == 1
    assert result["summary"]["general"]["crop_edge_jitter_applied"] == 1
    assert result["summary"]["general"]["crop_edge_jitter_changed"] == 0
    assert result["shards"][0]["kanji_sampling"]["crop_edge_jitter_applied"] == 1
    with np.load(dev) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["data"][:] = 40
    np.savez_compressed(dev, **arrays)
    with pytest.raises(ValueError, match="pixels overlap"):
        F.dataset_manifest(tmp_path, sources)
    arrays["meta"] = np.array(json.dumps([{"split": "train", "book_split_sha256": cohorts["dev"]}]))
    np.savez_compressed(dev, **arrays)
    with pytest.raises(ValueError, match="unbound role"):
        F.dataset_manifest(tmp_path, sources)
    path = tmp_path / "assets/corpus/book_ids.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="pool/cohort hash mismatch"):
        F.dataset_manifest(tmp_path, sources)


def test_actual_kanji_exposure_uses_policy_ranks():
    policy = {"ranks": {"一": 1, "串": 2500, "噸": 3500, "龜": 4500}}
    counts = F.kanji_exposure({"一": 9, "串": 3, "噸": 2, "龜": 1, "あ": 20}, policy)
    assert counts == {"glyphs": 15, "bands": {"0": 9, "1": 3, "2": 2, "3": 1}, "outside_allowlist": 0}
    with pytest.raises(ValueError, match="outside the policy"):
        F.kanji_exposure({"字": 1}, policy)


def test_renderer_flags_are_forwarded_to_existing_renderer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(F, "render_shards",
                        lambda root, split, arguments, bg=None: calls.append((root, split, arguments)))
    monkeypatch.setattr(sys, "argv", ["fresh_dataset.py", "render", "--root", str(tmp_path),
                                    "--vocab", "vocab.txt", "--kanji-policy", "policy.json",
                                    "--background-manifest", "backgrounds.json",
                                    "--native-font-manifest", "native-fonts.json", "--native-font-root", "owned-game",
                                    "--native-font-probability", "0.25",
                                    "--kanji-pairs", "--crop-edge-jitter", "--finder", "grad",
                                    "--out", "kanji", "--shards", "3", "--per-shard", "500",
                                    "--workers", "2", "--seed", "20261014"])
    F.main()
    root, split, arguments = calls[0]
    assert root == tmp_path and split == "train"
    assert arguments[arguments.index("--kanji-policy") + 1] == "policy.json"
    assert arguments[arguments.index("--background-manifest") + 1] == "backgrounds.json"
    assert arguments[arguments.index("--native-font-manifest") + 1] == "native-fonts.json"
    assert arguments[arguments.index("--native-font-root") + 1] == "owned-game"
    assert arguments[arguments.index("--native-font-probability") + 1] == "0.25"
    assert arguments[arguments.index("--seed") + 1] == "20261014"
    assert "--kanji-pairs" in arguments and "--crop-edge-jitter" in arguments
    assert arguments[arguments.index("--finder") + 1] == "grad"


@pytest.mark.parametrize("field,value", [("native_label_ink_preserved", False),
                                         ("native_core_alpha_threshold", 64),
                                         ("native_core_box", [2, -1, 20, 14]),
                                         ("crop_box", [0, 0, 19, 16]),
                                         ("original_crop_box", [0, 4, 40, 12]),
                                         ("native_font_manifest_sha256", None),
                                         ("native_metrics_sha256", "wrong"),
                                         ("native_strike", True)])
def test_native_snapshot_rejects_forged_source_or_ink_preservation_claims(tmp_path, monkeypatch, field, value):
    import native_font_manifest

    vocab = corpus_fixture(tmp_path)
    pools = F.build_pools(tmp_path, vocab, caps={s: {"dialogue": 4, "narration": 4} for s in ("train", "dev")})
    F.write_json(tmp_path / "assets_manifest.json", {"assets": []})
    cohort = next(r["sha256"] for r in pools["pool_records"]
                  if r.get("role") == "source_book_cohort" and r["split"] == "train")
    path = synthetic_shard(tmp_path, "native-general", "train", cohort)
    identity = {"family": "fixture_bitmap", "category": "gothic",
                "fnt": {"sha256": "b" * 64}, "pro": {"sha256": "c" * 64}}
    binding = {"manifest_sha256": "a" * 64, "fonts": [identity]}
    coverage = set(vocab.read_text().splitlines())
    monkeypatch.setattr(native_font_manifest, "load_manifest",
                        lambda *_: ({}, [("gothic", "fixture_bitmap", None, coverage)], binding))
    with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
    meta = json.loads(str(arrays["meta"]))
    meta[0].update(native_font_used=True, fam="fixture_bitmap", cat="gothic", native_strike=0,
                   native_font_manifest_sha256="a" * 64, native_font_sha256="b" * 64,
                   native_metrics_sha256="c" * 64, native_font_probability=1,
                   native_label_ink_preserved=True, native_core_alpha_threshold=128,
                   native_core_box=[2, 3, 20, 14], crop_box=[0, 0, 40, 16], original_crop_box=[0, 0, 40, 16])
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    kwargs = {"native_font_manifest": "generated-identity", "native_font_root": tmp_path}
    assert F.dataset_manifest(tmp_path, {"native-general": "train"}, vocab, **kwargs)["summary"][
        "native-general"]["native_label_ink_preserved"] == 1
    meta[0][field] = value
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="native"):
        F.dataset_manifest(tmp_path, {"native-general": "train"}, vocab, **kwargs)

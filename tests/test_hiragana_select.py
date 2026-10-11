"""Development selection leakage guards and int8 deployment compatibility."""
import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

import export
import hiragana_select as HS
from model import ARCH_KEYS, TinyVNOCR


@pytest.fixture
def datasets(tmp_path):
    real = np.full((32, 64, 3), 220, np.uint8)
    real[9:22, 8:58:3] = 30
    image = tmp_path / "capture.png"
    Image.fromarray(real).save(image)
    row = {"id": "dev-1", "split": "dev", "session": "capture-block-1", "hand_checked": True,
           "image": image.name, "sha256": HS.sha256(image), "vn": "fixture",
           "text": "あい", "box": [0, 0, 64, 32]}
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    gray = np.full((32, 64), 20, np.uint8)
    gray[9:22, 8:58:3] = 220
    synthetic = tmp_path / "dev.npz"
    meta = [{"split": "dev", "book_split_sha256": "1" * 64, "inverted": True}]
    np.savez(synthetic, data=gray.ravel(), offsets=[0, gray.size], shapes=[gray.shape],
             labels=json.dumps(["あい"]), meta=json.dumps(meta))
    return manifest, synthetic, row, real, gray


def test_development_pixels_use_console_and_stored_polarity(datasets):
    manifest, synthetic, _, pixels, gray = datasets
    real, synth, identities = HS.load_development(manifest, synthetic)
    np.testing.assert_array_equal(real[0].crop_rgb, 255 - HS.to_gray(pixels))
    np.testing.assert_array_equal(synth[0].crop_rgb[:, :, 0], gray)
    assert identities["real_dev"]["manifest_sha256"] == HS.sha256(manifest)
    assert identities["synthetic_dev"]["npz_sha256"] == HS.sha256(synthetic)
    assert identities["synthetic_dev"]["book_split_sha256"] == "1" * 64
    assert "hiragana-selection-dev" not in HS.SETS
    assert "hiragana-selection-dev" not in HS.LOCAL_MANIFESTS


@pytest.mark.parametrize("role", ["test", "train", None])
def test_real_test_or_training_role_is_rejected_before_reading_images(datasets, role):
    manifest, synthetic, row, _, _ = datasets
    row.update(split=role, image="absent-image.png")
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="development-only"):
        HS.load_development(manifest, synthetic)


def test_private_development_requires_hand_checked_captures(datasets):
    manifest, synthetic, row, _, _ = datasets
    row["hand_checked"] = False
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="hand_checked"):
        HS.load_development(manifest, synthetic)
    assert "hiragana-selection-dev" not in HS.SETS


@pytest.mark.parametrize("meta, message", [
    ([{"split": "test", "book_split_sha256": "1" * 64}], "development-only"),
    ([{"split": "train", "book_split_sha256": "1" * 64}], "development-only"),
    ([{"split": "dev"}], "cohort"),
    ([{"split": "dev", "book_split_sha256": "malformed"}], "cohort"),
    ([], "development-only"),
])
def test_synthetic_roles_and_book_provenance_are_required(datasets, meta, message):
    manifest, synthetic, _, _, gray = datasets
    np.savez(synthetic, data=gray.ravel(), offsets=[0, gray.size], shapes=[gray.shape],
             labels=json.dumps(["あい"]), meta=json.dumps(meta))
    with pytest.raises(ValueError, match=message):
        HS.load_development(manifest, synthetic)


def test_synthetic_development_scores_every_row_without_default_cap(datasets):
    manifest, synthetic, _, _, gray = datasets
    count = 3001
    np.savez(synthetic, data=np.tile(gray.ravel(), count), offsets=np.arange(count + 1) * gray.size,
             shapes=np.tile(gray.shape, (count, 1)), labels=json.dumps(["あ"] * count),
             meta=json.dumps([{"split": "dev", "book_split_sha256": "1" * 64}] * count))
    _, synth, identities = HS.load_development(manifest, synthetic)
    assert len(synth) == identities["synthetic_dev"]["lines"] == count


def scored(name, hira=2, real=5, synth=10, step=500):
    return {"checkpoint": name, "step": step,
            "scores": {"real_dev": {"all": {"errors": real}, "hiragana": {"errors": hira}},
                       "synthetic_dev": {"all": {"errors": synth}}}}


def test_full_text_guards_override_hiragana_improvement():
    baseline = scored("baseline")
    regresses_real = scored("real", hira=0, real=6)
    regresses_synth = scored("synthetic", hira=0, synth=11)
    chosen, improved = HS.choose(baseline, [regresses_real, regresses_synth])
    assert chosen is baseline and not improved
    assert not regresses_real["qualified"] and not regresses_synth["qualified"]


def test_ranking_and_ties_are_deterministic_and_keep_baseline_without_strict_improvement():
    baseline = scored("baseline")
    assert HS.choose(baseline, [scored("equal", step=100)])[0] is baseline
    candidates = [scored("less-hira", hira=0, real=5, synth=10, step=900),
                  scored("less-real", hira=0, real=4, synth=10, step=900),
                  scored("later", hira=0, real=4, synth=9, step=900),
                  scored("z-path", hira=0, real=4, synth=9, step=500),
                  scored("a-path", hira=0, real=4, synth=9, step=500)]
    chosen, improved = HS.choose(baseline, list(reversed(candidates)))
    assert chosen["checkpoint"] == "a-path" and improved
    assert HS.choose(baseline, [{"status": "rejected"}]) == (baseline, False)


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = TinyVNOCR(4, **ARCH_KEYS).eval()
    vocab = ["あ", "い", "〓"]
    ck = {"model": model.state_dict(), "vocab": vocab, "height": 32, "step": 500, **ARCH_KEYS}
    checkpoint = tmp_path / "ckpt_step500.pt"
    torch.save(ck, checkpoint)
    raw, _ = export.build_blob(export.model_ops(model, 32), vocab, 32, int8=True)
    baseline = tmp_path / "release.vocr"
    baseline.write_bytes(raw)
    reference = export.read_blob(raw)
    # Generated fixture keeps the deployment topology but uses a three-character head.
    monkeypatch.setattr(HS, "MODEL_BYTES", len(raw))
    monkeypatch.setattr(HS, "ARENA_BYTES", export.plan_arena(reference, mode=export.MODE_W8I))
    return baseline, checkpoint, ck, reference


def test_export_preserves_order_architecture_memory_and_source_identity(model_files, tmp_path):
    _, checkpoint, _, reference = model_files
    result = HS.export_checkpoint(checkpoint, reference, tmp_path)
    blob = export.read_blob(result["blob"])
    assert blob["int8"] and blob["vocab"] == reference["vocab"]
    assert result["checkpoint_sha256"] == HS.sha256(checkpoint)
    assert blob["sha"] == result["checkpoint_sha256"][:32]
    assert result["blob_sha256"] == HS.sha256(result["blob"])
    assert result["step"] == 500
    HS.validate_blob(blob, reference)


@pytest.mark.parametrize("patch,message", [
    ({"width": .5}, "architecture"), ({"ctx": 3}, "architecture"),
    ({"height": 24}, "height"), ({"vocab": ["い", "あ", "〓"]}, "vocabulary order"),
    ({"step": "500"}, "integer step"),
])
def test_incompatible_checkpoint_is_rejected_before_export(model_files, tmp_path, patch, message):
    _, checkpoint, ck, reference = model_files
    torch.save({**ck, **patch}, checkpoint)
    with pytest.raises(ValueError, match=message):
        HS.export_checkpoint(checkpoint, reference, tmp_path)
    assert not list(tmp_path.glob("*-ckpt*.vocr"))


def test_blob_architecture_and_planned_memory_are_checked(model_files, monkeypatch):
    _, _, _, reference = model_files
    HS.validate_blob(reference)
    changed = {**reference, "max_width": 512}
    with pytest.raises(ValueError, match="width 1024"):
        HS.validate_blob(changed)
    monkeypatch.setattr(HS, "MODEL_BYTES", HS.MODEL_BYTES + 16)
    with pytest.raises(ValueError, match="memory changed"):
        HS.validate_blob(reference)
    model = TinyVNOCR(4, width=.25, bottleneck=16).eval()
    raw, _ = export.build_blob(export.model_ops(model, 32), ["あ", "い", "〓"], 32, int8=True)
    with pytest.raises(ValueError, match="architecture"):
        HS.validate_blob(export.read_blob(raw))


def test_c_runtime_actual_memory_checked_before_inference(monkeypatch):
    class WrongMemory:
        info = {"blob_bytes": HS.MODEL_BYTES + 16, "arena_bytes": HS.ARENA_BYTES}

        def read(self, crops):
            pytest.fail("memory mismatch reached inference")

        def close(self):
            pass

    monkeypatch.setattr(HS, "NativeVocr", lambda _: WrongMemory())
    with pytest.raises(ValueError, match="C runtime"):
        HS.score_blob("unused", [], [])


def test_native_c_scoring_is_aggregate_only(model_files, datasets):
    baseline, _, _, _ = model_files
    real, synth, _ = HS.load_development(datasets[0], datasets[1])
    result = HS.score_blob(baseline, real, synth)
    assert result["backend"] == "vocr" and result["mode"] == "w8i"
    assert result["scores"]["real_dev"]["all"]["chars"] == 2
    assert result["scores"]["synthetic_dev"]["all"]["lines"] == 1
    assert "lines" not in result["scores"]["real_dev"]
    assert "あい" not in json.dumps(result, ensure_ascii=False)


def test_selection_locks_chosen_hash_and_refuses_rerun(model_files, datasets, monkeypatch, tmp_path):
    baseline, checkpoint, _, _ = model_files
    calls = []

    def score(path, real, synth):
        calls.append(path)
        record = scored("baseline" if path == baseline else "candidate", hira=2 if path == baseline else 1)
        return {"scores": record["scores"], "memory": {"blob_bytes": HS.MODEL_BYTES,
                                                        "arena_bytes": HS.ARENA_BYTES},
                "backend": "vocr", "mode": "w8i"}

    monkeypatch.setattr(HS, "score_blob", score)
    out = tmp_path / "selection.json"
    result = HS.select(baseline, datasets[0], datasets[1], [checkpoint], out)
    assert calls[0] == baseline
    assert result["strict_dev_hiragana_improvement"]
    assert result["chosen"]["blob_sha256"] == HS.sha256(result["chosen"]["blob"])
    assert result["selection_data"] == "development_only" and not result["heldout_test_evaluated"]
    assert json.loads(out.read_text()) == result
    assert (tmp_path / "selection_exports" / ".gitignore").read_text() == "*\n"
    monkeypatch.setattr(HS, "load_development", lambda *args: pytest.fail("completed selection was reopened"))
    with pytest.raises(FileExistsError, match="completed selection"):
        HS.select(baseline, datasets[0], datasets[1], [checkpoint], out)


def test_selection_records_incompatible_candidate_and_retains_baseline(model_files, datasets, tmp_path):
    baseline, checkpoint, ck, _ = model_files
    ck["ctx"] = 3
    torch.save(ck, checkpoint)
    out = tmp_path / "selection.json"
    result = HS.select(baseline, datasets[0], datasets[1], [checkpoint], out)
    assert not result["strict_dev_hiragana_improvement"]
    assert result["chosen"]["blob"] == str(baseline)
    assert result["candidates"][0]["status"] == "rejected"
    assert result["candidates"][0]["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()

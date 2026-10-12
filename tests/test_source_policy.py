"""Generated-pixel source caps, provenance rejection, real forwards and exact resume."""
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "eval"))
from metrics import norm  # noqa: E402

import source_policy as SP  # noqa: E402
import train2 as T2  # noqa: E402
from train import encoder  # noqa: E402


def asset(kind, origin, key, **extra):
    return {"kind": kind, "origin": origin, "sha256": hashlib.sha256(key.encode()).hexdigest(), **extra}


ASSETS = {
    "ofl": asset("font", "ofl", "ofl", license="OFL-1.1"),
    "text": asset("text", "non_game", "text", game_free_reviewed=True),
    "bg": asset("background", "procedural", "bg"),
    "gamefont": asset("font", "game", "gamefont", vn="v4", release="r1"),
    "gametext": asset("text", "game", "gametext", vn="v4", release="r1"),
    "gamebg": asset("background", "game", "gamebg", vn="v5", release="r2"),
    "real": asset("real_crop", "game", "real", vn="v6", release="r3"),
}


def meta(sample_id, assets=("ofl", "text", "bg"), patterns=(), kanji=False, broad=True, **extra):
    return {"split": "train", "source_provenance": {
        "schema": 1, "sample_id": sample_id, "asset_ids": list(assets),
        "correction_pattern_ids": list(patterns), "kanji_corrective": kanji,
        "broad_sampling": broad}, **extra}


def write_shard(path, metas, object_pixels=False):
    crops = [np.full((32, 40 + i * 8), 70 + i, np.uint8) for i in range(len(metas))]
    pixels = np.concatenate([crop.ravel() for crop in crops])
    if object_pixels:
        pixels = pixels.astype(object)  # Access requires allow_pickle; metadata checks must precede it.
    np.savez(path, data=pixels, offsets=np.cumsum([0, *[crop.size for crop in crops]]),
             shapes=np.asarray([crop.shape for crop in crops]), labels=json.dumps(["あい"] * len(metas)),
             meta=json.dumps(metas))


def setup(tmp_path, candidate=None, fallback=None, assets=None):
    candidate = candidate if candidate is not None else [meta("game", ("gamefont", "text", "bg"))]
    fallback = fallback if fallback is not None else [meta("broad-0"), meta("broad-1")]
    write_shard(tmp_path / "candidate.npz", candidate)
    write_shard(tmp_path / "fallback.npz", fallback)
    rows = [{"file": f"{source}.npz", "source": source, "split": "train",
             "sha256": SP.sha256(tmp_path / f"{source}.npz"),
             "bytes": (tmp_path / f"{source}.npz").stat().st_size, "samples": len(metas)}
            for source, metas in (("candidate", candidate), ("fallback", fallback))]
    doc = {"schema": 1, "kind": "general-vn-source-policy", "assets": copy.deepcopy(assets or ASSETS),
           "fallback_source": "fallback", "forbidden_vns": [], "shards": rows}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(doc))
    return path, doc


def save(path, doc):
    path.write_text(json.dumps(doc))
    return SP.SourcePolicy(path)


def sources_for(path, seed=9):
    policy = SP.SourcePolicy(path)
    files = {source: [path.parent / f"{source}.npz"] for source in ("candidate", "fallback")}
    provenance = policy.bind(files)
    sources = [T2.RamSource(source, paths, encoder(["あ", "い"]), {"aug": "0"}, norm,
                            np.random.default_rng([seed, i]), provenance)
               for i, (source, paths) in enumerate(files.items())]
    return policy, sources


def mixer_for(path, seed=9, batch=16):
    policy, sources = sources_for(path, seed)
    return T2.PolicyMixer(sources, [1, 0], batch, np.random.default_rng(seed), policy,
                          [{"aug": "0"}, {"aug": "0"}], bucket=3)


@pytest.mark.parametrize("tag,denominator", [
    ("game", 20), ("game_assets", 5), ("pattern", 200), ("kanji", 50)])
def test_each_exact_prefix_cap_independently(tag, denominator):
    exposure = SP.Exposure("selected", unknown=False,
                           font_text_vns=("v4",) if tag == "game" else (),
                           game_assets=tag in ("game", "game_assets"),
                           correction_pattern_ids=("kana:ずっ",) if tag == "pattern" else (),
                           kanji_corrective=tag == "kanji")
    fallback = SP.Exposure("fallback", unknown=False, fallback_verified=True)
    ledger, selected = SP.ExposureLedger(), 0
    for n in range(1, 1001):
        chosen = exposure if ledger.reason(exposure) is None else fallback
        ledger.consume(chosen, "mixed")
        selected += chosen is exposure
        assert selected <= n // denominator
    assert selected == 1000 // denominator


def test_multi_game_row_counts_union_not_asset_count_and_background_has_global_cap(tmp_path):
    path, _ = setup(tmp_path)
    policy = SP.SourcePolicy(path)
    row = policy.exposure(meta("multi", ("gamefont", "gametext", "gamebg", "real")), "unused")
    assert row.font_text_vns == ("v4", "v6")
    assert row.game_assets and not row.unknown
    ledger = SP.ExposureLedger()
    for _ in range(19):
        ledger.consume(SP.Exposure("b", unknown=False), "b")
    ledger.consume(row, "mixed")
    assert ledger.by_game == {"v4": 1, "v6": 1} and ledger.game_assets == 1
    scenery = policy.exposure(meta("scenery", ("ofl", "text", "gamebg")), "unused")
    assert not scenery.font_text_vns and scenery.game_assets


def test_organic_pattern_is_not_deliberate_and_exposure_identity_is_immutable(tmp_path):
    path, _ = setup(tmp_path)
    policy = SP.SourcePolicy(path)
    organic = policy.exposure(meta("organic", visible_label="ずっ"), "unused")
    assert organic.correction_pattern_ids == () and organic.fallback_verified
    assert SP.ExposureLedger().reason(organic) is None
    deliberate = policy.exposure(meta("deliberate", patterns=("kana:ずっ",)), "unused")
    assert SP.ExposureLedger().reason(deliberate) == "correction_pattern_cap"
    with pytest.raises(FrozenInstanceError):
        organic.sample_id = "changed"


def test_procedural_generated_text_requires_explicit_non_game_review(tmp_path):
    path, doc = setup(tmp_path)
    doc["assets"]["names"] = asset("text", "procedural", "generator+reviewed-inputs")
    policy = save(path, doc)
    row = meta("names", assets=("ofl", "text", "names", "bg"))
    assert policy.exposure(row, "unused").unknown
    doc["assets"]["names"]["game_free_reviewed"] = True
    assert save(path, doc).exposure(row, "unused").fallback_verified


@pytest.mark.parametrize("change", ["missing_metadata", "missing_role", "missing_asset", "missing_vn",
                                     "missing_release", "missing_ofl_license", "unreviewed_text"])
def test_missing_provenance_is_unknown_and_replaced_never_game_free(tmp_path, change):
    row, assets = meta("unknown"), copy.deepcopy(ASSETS)
    if change == "missing_metadata":
        row = {}
    elif change == "missing_role":
        row.pop("split")
    elif change == "missing_asset":
        row["source_provenance"]["asset_ids"].append("unregistered")
    elif change in ("missing_vn", "missing_release"):
        assets["partial"] = asset("font", "game", "partial", vn="v9", release="r9")
        assets["partial"].pop("vn" if change == "missing_vn" else "release")
        row["source_provenance"]["asset_ids"].append("partial")
    elif change == "missing_ofl_license":
        assets["partial"] = asset("font", "ofl", "partial")
        row["source_provenance"]["asset_ids"].append("partial")
    else:
        assets["partial"] = asset("text", "non_game", "partial")
        row["source_provenance"]["asset_ids"].append("partial")
    path, _ = setup(tmp_path, [row], assets=assets)
    mixer = mixer_for(path)
    batch = mixer.next()
    assert all(item.src == 1 and item.provenance.fallback_verified for item in batch)
    assert mixer.ledger.forwarded == 0
    mixer.record_forward(batch)
    assert mixer.ledger.forwarded == 16 and mixer.ledger.game_assets == 0
    assert mixer.rejected == {"unknown_provenance": 16}


@pytest.mark.parametrize("refs", [("text", "bg"), ("ofl", "bg"), ("ofl", "text"), ("bg",)])
def test_partial_synthetic_provenance_cannot_bypass_game_or_pattern_caps(tmp_path, refs):
    path, _ = setup(tmp_path, candidate=[meta("partial", assets=refs)])
    mixer = mixer_for(path)
    batch = mixer.next()
    assert all(item.src == 1 and item.provenance.fallback_verified for item in batch)
    assert mixer.rejected == {"unknown_provenance": 16}
    mixer.record_forward(batch)
    assert mixer.ledger.forwarded == 16 and mixer.ledger.by_source == {"fallback": 16}


def test_declared_game_real_crop_covers_native_font_text_and_background(tmp_path):
    path, _ = setup(tmp_path, candidate=[meta("native", assets=("real",), broad=False)])
    policy = SP.SourcePolicy(path)
    exposure = policy.exposure(meta("native", assets=("real",), broad=False), "unused")
    assert not exposure.unknown and not exposure.fallback_verified
    assert exposure.font_text_vns == ("v6",) and exposure.game_assets
    ledger = SP.ExposureLedger()
    for _ in range(19):
        ledger.consume(policy.exposure(meta("broad"), "unused"), "fallback")
    ledger.consume(exposure, "native")
    assert ledger.by_game == {"v6": 1} and ledger.game_assets == 1


@pytest.mark.parametrize("role", ["dev", "test", "validation", None, 1])
def test_non_train_row_rejected_without_reading_pixels(tmp_path, role):
    row = meta("dev", split=role)
    path, doc = setup(tmp_path, [row])
    write_shard(tmp_path / "candidate.npz", [row], object_pixels=True)
    doc["shards"][0].update(sha256=SP.sha256(tmp_path / "candidate.npz"),
                             bytes=(tmp_path / "candidate.npz").stat().st_size)
    policy = save(path, doc)
    with pytest.raises(ValueError, match="non-training row"):
        policy.bind({name: [tmp_path / f"{name}.npz"] for name in ("candidate", "fallback")})


@pytest.mark.parametrize("field,value", [("split", "test"), ("sha256", "G" * 64),
                                         ("bytes", True), ("samples", 0), ("file", "../candidate.npz")])
def test_invalid_shard_pin_or_role_rejected(tmp_path, field, value):
    path, doc = setup(tmp_path)
    doc["shards"][0][field] = value
    with pytest.raises(ValueError):
        save(path, doc)


def test_corrupted_bytes_exact_inventory_and_policy_mutation_rejected(tmp_path):
    path, _ = setup(tmp_path)
    policy = SP.SourcePolicy(path)
    files = {name: [tmp_path / f"{name}.npz"] for name in ("candidate", "fallback")}
    with pytest.raises(ValueError, match="exact policy inventory"):
        policy.bind({"fallback": files["fallback"]})
    with (tmp_path / "candidate.npz").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="bytes/hash"):
        policy.bind(files)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="policy changed"):
        policy.verify_inputs()


def test_forbidden_vn_conflicting_asset_alias_and_duplicate_row_rejected(tmp_path):
    path, doc = setup(tmp_path)
    doc["forbidden_vns"] = ["v4"]
    with pytest.raises(ValueError, match="protected game"):
        save(path, doc)
    doc["forbidden_vns"] = []
    doc["assets"]["alias"] = {**doc["assets"]["gamefont"], "vn": "v99"}
    with pytest.raises(ValueError, match="conflicting ownership"):
        save(path, doc)
    path, _ = setup(tmp_path, [meta("broad-0")])
    with pytest.raises(ValueError, match="sample identity is repeated"):
        sources_for(path)


@pytest.mark.parametrize("bad_fallback", [meta("b", assets=("ofl", "text")),
                                         meta("b", patterns=("p",)), meta("b", kanji=True),
                                         meta("b", broad=False), meta("b", assets=("ofl", "text", "gamebg")), {}])
def test_fallback_cannot_be_unknown_game_backed_targeted_or_missing_background(tmp_path, bad_fallback):
    path, _ = setup(tmp_path, fallback=[bad_fallback])
    with pytest.raises(ValueError, match="fallback must be broad"):
        sources_for(path)


def test_bucketing_keeps_identity_prefetch_does_not_count_and_lossless_forward_does(tmp_path):
    path, _ = setup(tmp_path)
    mixer = mixer_for(path)
    batch = mixer.next()
    assert sum(mixer.counts) == 48 and mixer.ledger.forwarded == 0
    with pytest.raises(ValueError, match="previous batch"):
        mixer.next()
    with pytest.raises(ValueError, match="before recording"):
        mixer.state_dict()
    mixer.record_forward(batch)  # Models update BN on forwards even without a gradient.
    assert mixer.ledger.forwarded == 16 and mixer.ledger.optimizer == 0
    next_batch = mixer.next()
    assert sum(mixer.counts) == 48 and mixer.ledger.forwarded == 16
    mixer.record_forward(next_batch)
    mixer.record_optimizer(next_batch)
    assert mixer.ledger.forwarded == 32 and mixer.ledger.optimizer == 16
    with pytest.raises(ValueError, match="counted once"):
        mixer.record_optimizer(next_batch)
    report = mixer.report()
    assert report["buffered_rows"] == 16 and report["consumed"]["by_game"] == {"v4": 1}
    assert all(item.provenance.sample_id in ("game", "broad-0", "broad-1") for item in next_batch)


def test_resume_restores_sources_rng_buffers_and_actual_counts_exactly(tmp_path):
    path, _ = setup(tmp_path, [meta(f"game-{i}", ("gamefont", "text", "bg")) for i in range(5)])
    original = mixer_for(path, batch=17)
    first = original.next()
    original.record_forward(first)
    original.record_optimizer(first)
    state = copy.deepcopy(original.state_dict())
    resumed = mixer_for(path, seed=23, batch=17)
    resumed.load_state_dict(state)
    for i in range(35):
        batches = [mixer.next() for mixer in (original, resumed)]
        assert [(item.src, item.provenance, item.img.tobytes()) for item in batches[0]] == [
            (item.src, item.provenance, item.img.tobytes()) for item in batches[1]]
        for mixer, batch in zip((original, resumed), batches):
            mixer.record_forward(batch)
            if i % 3:
                mixer.record_optimizer(batch)
        assert original.rng.random() == resumed.rng.random()
        assert original.state_dict() == resumed.state_dict()
    assert resumed.ledger.forwarded == 612 and resumed.ledger.by_game["v4"] == 30


@pytest.mark.parametrize("change", ["hash", "settings", "ledger", "draws", "buffer", "source_cursor", "rng"])
def test_resume_rejects_changed_policy_accounting_cursor_or_identity(tmp_path, change):
    path, _ = setup(tmp_path)
    mixer = mixer_for(path)
    batch = mixer.next()
    mixer.record_forward(batch)
    state = copy.deepcopy(mixer.state_dict())
    if change == "hash":
        state["policy_sha256"] = "0" * 64
    elif change == "settings":
        state["settings"]["batch"] += 1
    elif change == "ledger":
        state["ledger"]["game_assets"] = 100
    elif change == "draws":
        state["counts"][0] += 1
    elif change == "buffer":
        state["ready"][0][0][1] = "absent"
    elif change == "source_cursor":
        state["sources"][0]["order"] = []
    else:
        state["rng"] = {"bit_generator": "invalid"}
    with pytest.raises((ValueError, KeyError, TypeError)):
        mixer_for(path).load_state_dict(state)


def test_cli_ram_guard_runs_before_bad_shard_pixels(tmp_path):
    path, _ = setup(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / "train/train2.py"), "--run", "generated",
                             "--out", str(tmp_path / "out"), "--steps", "1", "--threads", "1",
                             "--source-policy", str(path), "--src",
                             f"candidate:1:{tmp_path / 'candidate.npz'}:stream=1", "--src",
                             f"fallback:0:{tmp_path / 'fallback.npz'}:stream=0"],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode != 0 and "RAM-only" in result.stderr
    assert not (tmp_path / "out/generated/ckpt_last.pt").exists()


@pytest.mark.parametrize("selection_only", [False, True])
def test_policy_mode_rejects_unbound_validation_before_object_pixel_access(tmp_path, selection_only):
    path, _ = setup(tmp_path)
    val = tmp_path / "unbound-val.npz"
    write_shard(val, [meta("dev", split="dev")], object_pixels=True)
    arguments = ["--select", "unbound"] if selection_only else ["--val", f"unbound:{val}"]
    command = [sys.executable, str(ROOT / "train/train2.py"), "--run", "no-validation", "--out",
               str(tmp_path / "out"), "--steps", "0", "--source-policy", str(path), "--src",
               f"candidate:1:{tmp_path / 'candidate.npz'}", *arguments]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "separately frozen general-VN selector" in result.stderr
    assert "Object arrays cannot be loaded" not in result.stderr
    assert not (tmp_path / "out/no-validation").exists()


@pytest.mark.parametrize("options", [{"targets": "unbound"}, {"label": "teacher"},
                                     {"label": "gt|teacher"}, {"kd": "1"}, {"kdonly": "1"}])
def test_opt_in_teacher_inputs_require_future_inventory_not_silent_acceptance(options):
    with pytest.raises(ValueError, match="unpinned teacher"):
        T2.validate_policy_options(options)
    T2.validate_policy_options({"label": "gt", "kd": "0", "kdonly": "0", "aug": "1"})


def test_metadata_validation_cli_never_decodes_crop_arrays_and_refuses_overwrite(tmp_path):
    path, doc = setup(tmp_path)
    write_shard(tmp_path / "candidate.npz", [meta("game", ("gamefont", "text", "bg"))], object_pixels=True)
    doc["shards"][0].update(sha256=SP.sha256(tmp_path / "candidate.npz"),
                             bytes=(tmp_path / "candidate.npz").stat().st_size)
    save(path, doc)
    out = tmp_path / "validation.json"
    command = [sys.executable, str(ROOT / "train/source_policy.py"), "--policy", str(path), "--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text())
    assert record["model_forwards"] == 0 and record["sources"]["fallback"]["verified_broad"] == 2
    previous = out.read_bytes()
    again = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert again.returncode != 0 and out.read_bytes() == previous


def test_journal_durable_forward_only_hash_chain_and_checkpoint_tail(tmp_path):
    execution, path, policy_sha = {"initial_model": "generated"}, tmp_path / "exposures.jsonl", "1" * 64
    ledger = SP.ExposureLedger()
    journal = SP.ExposureJournal(path, execution, policy_sha)
    journal.append("intent", ledger)
    ledger.consume(SP.Exposure("broad", unknown=False), "broad")
    journal.append("forward", ledger)
    first_tail, first_state = journal.tail, copy.deepcopy(ledger.state_dict())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in records] == ["intent", "forward"]
    assert records[0]["ledger"]["forwarded"] == 0 and records[1]["ledger"]["forwarded"] == 1
    journal.close()
    with pytest.raises(ValueError, match="uncheckpointed exposures"):
        SP.ExposureJournal(path, execution, policy_sha, SP.ExposureLedger().state_dict(), "0" * 64)
    resumed = SP.ExposureJournal(path, execution, policy_sha, first_state, first_tail)
    ledger.optimized(["broad"])
    resumed.append("optimizer", ledger)
    final_tail = resumed.tail
    resumed.close()
    final = SP.ExposureJournal(path, execution, policy_sha, ledger.state_dict(), final_tail)
    final.close()
    assert len(path.read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="provenance mismatch"):
        SP.ExposureJournal(path, {"changed": True}, policy_sha, ledger.state_dict(), final_tail)
    content = path.read_text()
    path.write_text(content.replace('"broad": 1', '"broad": 2', 1))
    with pytest.raises(ValueError, match="hash/provenance"):
        SP.ExposureJournal(path, execution, policy_sha, ledger.state_dict(), final_tail)


def test_journal_truncation_reset_and_outside_writer_are_rejected(tmp_path):
    path = tmp_path / "exposures.jsonl"
    ledger = SP.ExposureLedger()
    journal = SP.ExposureJournal(path, {}, "1" * 64)
    journal.append("intent", ledger)
    ledger.consume(SP.Exposure("broad", unknown=False), "broad")
    journal.append("forward", ledger)
    tail = journal.tail
    with pytest.raises(ValueError, match="matching model checkpoint"):
        SP.ExposureJournal(path, {}, "1" * 64)
    with path.open("a") as stream:
        stream.write(" ")
    ledger.optimized(["broad"])
    with pytest.raises(ValueError, match="outside its writer"):
        journal.append("optimizer", ledger)
    journal.close()
    path.write_text("")
    with pytest.raises(ValueError, match="uncheckpointed exposures"):
        SP.ExposureJournal(path, {}, "1" * 64, ledger.state_dict(), tail)


def test_unresolved_intent_refuses_resume_even_with_matching_checkpoint_tail(tmp_path):
    ledger = SP.ExposureLedger()
    path = tmp_path / "intent.jsonl"
    journal = SP.ExposureJournal(path, {}, "1" * 64)
    journal.append("intent", ledger)
    tail = journal.tail
    journal.close()
    record = json.loads(path.read_text())
    assert record["event"] == "intent" and record["ledger"]["forwarded"] == 0
    with pytest.raises(ValueError, match="unresolved forward intent"):
        SP.ExposureJournal(path, {}, "1" * 64, ledger.state_dict(), tail)
    with pytest.raises(ValueError, match="unresolved forward intent"):
        SP.ExposureJournal(path, {}, "1" * 64)


def test_journal_rejects_duplicate_intents_missing_intents_and_corrupt_intent_hash(tmp_path):
    ledger, path = SP.ExposureLedger(), tmp_path / "intents.jsonl"
    journal = SP.ExposureJournal(path, {}, "1" * 64)
    with pytest.raises(ValueError, match="durable forward intent"):
        journal.append("forward", ledger)
    journal.append("intent", ledger)
    tail = journal.tail
    with pytest.raises(ValueError, match="duplicate forward intent"):
        journal.append("intent", ledger)
    assert journal.tail == tail and len(path.read_text().splitlines()) == 1
    journal.close()
    record = json.loads(path.read_text())
    record["sha256"] = "0" * 64
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="hash/provenance"):
        SP.ExposureJournal(path, {}, "1" * 64)


@pytest.mark.parametrize("changed", ["restore_helper", "host", "math_backend"])
def test_resume_rejects_changed_restore_helper_host_or_math_backend_identity(tmp_path, changed):
    path, _ = setup(tmp_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("あ\nい\n")
    identity = T2.execution_identity(vocab, ["あ", "い"])
    assert identity["code"]["train/restore_blob.py"] == SP.sha256(ROOT / "train/restore_blob.py")
    assert identity["host"]["libc"] and identity["math_backend"]["opencv_build_sha256"]
    assert identity["math_backend"]["torch_build_sha256"] and "mkldnn_enabled" in identity["math_backend"]
    if changed == "restore_helper":
        identity["code"]["train/restore_blob.py"] = "0" * 64
    elif changed == "host":
        identity["host"]["machine"] = "changed-host"
    else:
        identity["math_backend"]["opencv_build_sha256"] = "0" * 64
    identity["initial_model"] = {"kind": "seeded_random", "seed": 1, "state_sha256": "0" * 64}
    out = tmp_path / "out/changed-identity"
    out.mkdir(parents=True)
    # Identity must fail before any optimizer/model/checkpoint state is accessed.
    torch.save({"source_policy": {}, "source_policy_execution": identity}, out / "ckpt_last.pt")
    before = SP.sha256(out / "ckpt_last.pt")
    command = [sys.executable, str(ROOT / "train/train2.py"), "--run", out.name, "--out", str(out.parent),
               "--steps", "0", "--threads", "1", "--width", ".125", "--bottleneck", "8", "--depth", "0",
               "--ctx", "0", "--vocab", str(vocab), "--source-policy", str(path), "--src",
               f"candidate:1:{tmp_path / 'candidate.npz'}", "--src", f"fallback:0:{tmp_path / 'fallback.npz'}"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "execution/vocabulary identity differs" in result.stderr
    assert SP.sha256(out / "ckpt_last.pt") == before and not (out / "source_exposures.jsonl").exists()


@pytest.mark.parametrize("failure", ["model_forward", "journal_forward", "kill_before_journal_forward"])
def test_interrupted_forward_intent_is_durable_without_claiming_successful_rows(tmp_path, failure):
    path, _ = setup(tmp_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("あ\nい\n")
    arguments = ["--run", "interrupted-forward", "--out", str(tmp_path / "out"), "--steps", "1",
                 "--threads", "1", "--batch", "20", "--width", ".125", "--bottleneck", "8", "--depth", "0",
                 "--ctx", "0", "--vocab", str(vocab), "--source-policy", str(path), "--src",
                 f"candidate:1:{tmp_path / 'candidate.npz'}:aug=0", "--src",
                 f"fallback:0:{tmp_path / 'fallback.npz'}:aug=0"]
    injection = ["import os, sys", f"sys.path.insert(0, {str(ROOT / 'train')!r})", "import train2"]
    if failure == "model_forward":
        injection += ["original = train2.TinyVNOCR.forward", "def fail(self, x):",
                      "    original(self, x)", "    raise RuntimeError('generated forward failure')",
                      "train2.TinyVNOCR.forward = fail"]
    else:
        injection += ["original = train2.ExposureJournal.append", "def fail(self, event, ledger):",
                      "    if event == 'forward':"]
        injection += ["        os._exit(74)"] if failure == "kill_before_journal_forward" else [
            "        raise RuntimeError('generated append failure')"]
        injection += ["    return original(self, event, ledger)", "train2.ExposureJournal.append = fail"]
    injection += [f"sys.argv = ['train2.py', *{arguments!r}]", "train2.main()"]
    failed = subprocess.run([sys.executable, "-c", "\n".join(injection)], capture_output=True,
                            text=True, timeout=45)
    assert failed.returncode == (74 if failure == "kill_before_journal_forward" else 1)
    out = tmp_path / "out/interrupted-forward"
    journal = [json.loads(line) for line in (out / "source_exposures.jsonl").read_text().splitlines()]
    assert len(journal) == 1 and journal[0]["event"] == "intent"
    assert journal[0]["ledger"]["forwarded"] == journal[0]["ledger"]["optimizer"] == 0
    assert not (out / "ckpt_last.pt").exists()
    attempted = subprocess.run([sys.executable, str(ROOT / "train/train2.py"), *arguments],
                               capture_output=True, text=True, timeout=45)
    assert attempted.returncode != 0 and "unresolved forward intent" in attempted.stderr
    assert len((out / "source_exposures.jsonl").read_text().splitlines()) == 1


def test_two_training_processes_cannot_overwrite_run_artifacts(tmp_path):
    path, _ = setup(tmp_path)
    out = tmp_path / "out/owned"
    out.mkdir(parents=True)
    args_file = out / "args.json"
    args_file.write_text("sentinel")
    owner = SP.RunWriter(out)
    result = subprocess.run([sys.executable, str(ROOT / "train/train2.py"), "--run", "owned",
                             "--out", str(out.parent), "--steps", "1", "--threads", "1", "--src",
                             f"candidate:1:{tmp_path / 'candidate.npz'}", "--source-policy", str(path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "another train2 writer" in result.stderr
    assert args_file.read_text() == "sentinel" and not (out / "source_exposures.jsonl").exists()
    owner.close()
    new_owner = SP.RunWriter(out)
    new_owner.close()


def test_duplicate_json_roles_or_asset_identities_are_malformed():
    with pytest.raises(ValueError, match="duplicate JSON"):
        SP.parse_json('{"split":"dev","split":"train"}')


def test_readable_execution_identity_is_frozen_before_any_forward(tmp_path):
    path = tmp_path / "execution.json"
    SP.freeze_json(path, {"initial_model": "generated"})
    assert json.loads(path.read_text()) == {"initial_model": "generated"}
    SP.freeze_json(path, {"initial_model": "generated"})
    with pytest.raises(ValueError, match="execution identity differs"):
        SP.freeze_json(path, {"initial_model": "changed"})


def test_initial_checkpoint_byte_identity_is_frozen_and_checked_on_resume(tmp_path):
    path, _ = setup(tmp_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("あ\nい\n")
    init = tmp_path / "initial.pt"
    model = T2.TinyVNOCR(3, width=.125, bottleneck=8, depth=0, ctx=0)
    torch.save({"model": model.state_dict(), "vocab": ["あ", "い"]}, init)
    command = [sys.executable, str(ROOT / "train/train2.py"), "--run", "initial", "--out", str(tmp_path / "out"),
               "--steps", "1", "--threads", "1", "--batch", "20", "--width", ".125", "--bottleneck", "8",
               "--depth", "0", "--ctx", "0", "--vocab", str(vocab), "--init", str(init),
               "--source-policy", str(path), "--src", f"candidate:1:{tmp_path / 'candidate.npz'}:aug=0",
               "--src", f"fallback:0:{tmp_path / 'fallback.npz'}:aug=0"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    checkpoint = tmp_path / "out/initial/ckpt_last.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["source_policy_execution"]["initial_model"] == {"kind": "checkpoint", "sha256": SP.sha256(init)}
    prior = SP.sha256(checkpoint)
    with init.open("ab") as stream:
        stream.write(b"changed initial bytes")
    changed = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert changed.returncode != 0 and "initial model bytes differ" in changed.stderr
    assert SP.sha256(checkpoint) == prior


def test_loss_error_after_real_forward_is_durable_and_cannot_resume_from_stale_checkpoint(tmp_path):
    path, _ = setup(tmp_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("あ\nい\n")
    arguments = ["--run", "loss-error", "--out", str(tmp_path / "out"), "--steps", "1", "--threads", "1",
                 "--batch", "20", "--width", ".125", "--bottleneck", "8", "--depth", "0", "--ctx", "0",
                 "--vocab", str(vocab), "--source-policy", str(path), "--src",
                 f"candidate:1:{tmp_path / 'candidate.npz'}:aug=0", "--src",
                 f"fallback:0:{tmp_path / 'fallback.npz'}:aug=0"]
    injection = "\n".join([
        "import sys", f"sys.path.insert(0, {str(ROOT / 'train')!r})", "import train2",
        "def fail(*args, **kwargs):", "    raise ValueError('forced loss failure')",
        "train2.F.ctc_loss = fail", f"sys.argv = ['train2.py', *{arguments!r}]", "train2.main()"])
    failed = subprocess.run([sys.executable, "-c", injection], capture_output=True, text=True, timeout=45)
    assert failed.returncode != 0 and "forced loss failure" in failed.stderr
    out = tmp_path / "out/loss-error"
    journal = [json.loads(line) for line in (out / "source_exposures.jsonl").read_text().splitlines()]
    assert (out / "source_execution.json").is_file()
    assert [r["event"] for r in journal] == ["intent", "forward"]
    assert journal[0]["ledger"]["forwarded"] == 0 and journal[1]["ledger"]["forwarded"] == 20
    assert journal[1]["ledger"]["optimizer"] == 0
    assert not (out / "ckpt_last.pt").exists()
    attempted = subprocess.run([sys.executable, str(ROOT / "train/train2.py"), *arguments],
                               capture_output=True, text=True, timeout=45)
    assert attempted.returncode != 0 and "matching model checkpoint" in attempted.stderr
    assert len((out / "source_exposures.jsonl").read_text().splitlines()) == 2


def test_real_generated_training_resume_matches_dropout_weights_and_exposure_ledgers(tmp_path):
    """Exercise the driver, actual CTC/optimizer and dropout RNG; no real training data."""
    path, _ = setup(tmp_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("あ\nい\n")
    base = ["--steps", "4", "--batch", "10", "--threads", "1", "--eval-every", "2",
            "--log-every", "1", "--width", ".125", "--bottleneck", "8", "--depth", "0",
            "--ctx", "0", "--warmup", "0", "--out", str(tmp_path / "out"), "--vocab", str(vocab),
            "--source-policy", str(path), "--src", f"candidate:1:{tmp_path / 'candidate.npz'}:aug=1,inv=0",
            "--src", f"fallback:0:{tmp_path / 'fallback.npz'}:aug=1,inv=0"]
    driver = [sys.executable, str(ROOT / "train/train2.py")]
    full = subprocess.run([*driver, "--run", "full", *base], capture_output=True, text=True, timeout=45)
    assert full.returncode == 0, full.stderr
    interruption = "\n".join([
        "import pathlib, sys", f"sys.path.insert(0, {str(ROOT / 'train')!r})", "import train2",
        "original = train2.PolicyMixer.next",
        "def interrupt(self):",
        f"    if pathlib.Path({str(tmp_path / 'out/resumed/ckpt_last.pt')!r}).exists():",
        "        raise SystemExit(75)",
        "    return original(self)",
        "train2.PolicyMixer.next = interrupt",
        f"sys.argv = ['train2.py', '--run', 'resumed', *{base!r}]", "train2.main()"])
    first = subprocess.run([sys.executable, "-c", interruption], capture_output=True, text=True, timeout=45)
    assert first.returncode == 75, first.stderr
    mid = torch.load(tmp_path / "out/resumed/ckpt_last.pt", weights_only=False)
    assert mid["step"] == 2 and mid["source_policy"]["ledger"]["forwarded"] == 20
    resumed = subprocess.run([*driver, "--run", "resumed", *base], capture_output=True, text=True, timeout=45)
    assert resumed.returncode == 0, resumed.stderr
    a = torch.load(tmp_path / "out/full/ckpt_last.pt", weights_only=False)
    b = torch.load(tmp_path / "out/resumed/ckpt_last.pt", weights_only=False)
    assert a["source_policy"] == b["source_policy"]
    assert a["source_policy_execution"] == b["source_policy_execution"]
    assert a["source_policy_execution"]["initial_model"]["kind"] == "seeded_random"
    assert a["source_policy_execution"]["ordered_vocabulary_sha256"] == SP.json_digest(["あ", "い"])
    assert a["source_policy_journal_sha256"] == b["source_policy_journal_sha256"]
    assert a["source_policy"]["ledger"]["by_game"] == {"v4": 2}
    assert a["source_policy"]["ledger"]["forwarded"] == a["source_policy"]["ledger"]["optimizer"] == 40
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    changed_recipe = subprocess.run([*driver, "--run", "resumed", *base, "--lr", ".008"],
                                    capture_output=True, text=True, timeout=45)
    assert changed_recipe.returncode != 0 and "training recipe differs" in changed_recipe.stderr
    b.pop("source_policy")
    torch.save(b, tmp_path / "out/resumed/ckpt_last.pt")
    invalid_resume = subprocess.run([*driver, "--run", "resumed", *base], capture_output=True,
                                    text=True, timeout=45)
    assert invalid_resume.returncode != 0 and "cannot add or remove" in invalid_resume.stderr

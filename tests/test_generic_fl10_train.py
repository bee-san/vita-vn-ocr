"""Generated generic-development locks and tiny fixture training, never real VN data."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import generic_fl10_train as G
import kanji_policy as KP
import source_policy as SP
import train2
from test_general_vn_select import package  # noqa: F401 - reusable generated selector fixture
from test_general_vn_select import prepare as prepare_development


def pin(path):
    return {"file": str(path), "sha256": SP.sha256(path)}


@pytest.fixture
def training(request, monkeypatch):
    generated = request.getfixturevalue("package")
    root = generated["root"]
    development = prepare_development(generated)
    monkeypatch.setattr(G, "BASELINE_SHA", SP.sha256(generated["baseline"]))
    torch.set_num_threads(1)
    vocab = root / "vocab.txt"
    vocab.write_text("あ\nい\n人\n入\n漢\n〓\n", encoding="utf-8")
    jiten, similar = root / "jiten.csv", root / "similar.txt"
    jiten.write_text("Kanji,Rank\n人,1\n入,2\n", encoding="utf-8")
    similar.write_text("人/入/\n", encoding="utf-8")
    monkeypatch.setattr(KP, "JITEN_SHA256", SP.sha256(jiten))
    monkeypatch.setattr(KP, "SIMILAR_SHA256", SP.sha256(similar))
    kanji = root / "kanji.json"
    kanji.write_text(json.dumps(KP.build_policy(jiten, similar, vocab), ensure_ascii=False))
    init = root / "generated-initial.pt"
    model = train2.TinyVNOCR(7, width=1.0, bottleneck=96, depth=1, ctx=2, ctx_k=5)
    checkpoint = {"model": model.state_dict(), "vocab": vocab.read_text().splitlines(),
                  "width": 1.0, "bottleneck": 96, "depth": 1, "ctx": 2, "ctx_k": 5, "height": 32,
                  "freeze_bn_stats": True, "unk": "〓", "restored_from": {"sha256": "1" * 64}}
    torch.save(checkpoint, init)
    monkeypatch.setattr(G, "INIT_SHA", SP.sha256(init))
    monkeypatch.setattr(G, "VOCAB_SHA", SP.sha256(vocab))
    monkeypatch.setattr(G, "KANJI_POLICY_SHA", SP.sha256(kanji))
    monkeypatch.setattr(G, "FP32_SHA", "1" * 64)
    monkeypatch.setattr(G, "CLASSES", 6)
    shard = root / "train.npz"
    metas = [{"split": "train", "source_provenance": {"schema": 1, "sample_id": f"generated:{i}",
        "asset_ids": ["font", "text", "background"], "correction_pattern_ids": [],
        "kanji_corrective": False, "broad_sampling": True}} for i in range(2)]
    data = np.full(32 * 48 * 2, 200, np.uint8)
    np.savez(shard, data=data, offsets=[0, 32 * 48, 32 * 48 * 2], shapes=[[32, 48], [32, 48]],
             labels=json.dumps(["あい", "人入"]), meta=json.dumps(metas))
    policy = root / "source-policy.json"
    policy_doc = {"schema": 1, "kind": "general-vn-source-policy", "fallback_source": "broad",
        "forbidden_vns": sorted({record["vn"] for record in generated["rows"]}),
        "assets": {
            "font": {"kind": "font", "origin": "ofl", "license": "OFL-1.1", "sha256": "2" * 64},
            "text": {"kind": "text", "origin": "non_game", "game_free_reviewed": True, "sha256": "3" * 64},
            "background": {"kind": "background", "origin": "procedural", "sha256": "4" * 64}},
        "shards": [{"file": shard.name, "source": "broad", "split": "train", "samples": 2,
                    "sha256": SP.sha256(shard), "bytes": shard.stat().st_size}]}
    policy.write_text(json.dumps(policy_doc))
    config = root / "config.json"
    doc = {"schema": 1, "kind": "general-vn-fl10-training-config", "init": pin(init), "vocab": pin(vocab),
        "kanji_policy": pin(kanji), "development": pin(generated["out"]), "source_policy": pin(policy),
        "run": {"name": "generated", "out": str(root / "runs")},
        "recipe": {"steps": 2, "batch": 2, "lr": .00002, "final_lr": .05, "warmup": 0, "wd": .01,
                   "train_scope": "head", "seed": 1, "threads": 1, "eval_every": 1},
        "sources": [{"name": "broad", "weight": 1.0, "augmentation": False}]}
    config.write_text(json.dumps(doc))
    return {**generated, "config": config, "config_doc": doc, "policy": policy, "policy_doc": policy_doc,
            "init": init, "checkpoint": checkpoint, "vocab": vocab, "kanji": kanji, "shard": shard,
            "prepared": root / "frozen/prepared.json", "development": development}


def save_config(training):
    training["config"].write_text(json.dumps(training["config_doc"]))


def forbidden_load(*args, **kwargs):
    pytest.fail("rejected protocol/input reached checkpoint tensors or crop pixels")


@pytest.mark.parametrize("change", ["unready", "wrong_role", "weaker_protocol"])
def test_development_precondition_rejects_before_crop_or_checkpoint_and_creates_nothing(training, monkeypatch, change):
    lock = copy.deepcopy(training["development"])
    if change == "unready":
        lock["readiness"]["ready"] = False
    elif change == "wrong_role":
        lock["test_or_google_evaluated"] = True
    else:
        lock["protocol"]["minimums"]["vns_per_console"] = 1
    training["out"].write_text(json.dumps(lock))
    training["config_doc"]["development"] = pin(training["out"])
    save_config(training)
    monkeypatch.setattr(torch, "load", forbidden_load)
    monkeypatch.setattr(train2, "load_shard", forbidden_load)
    monkeypatch.setattr(SP.SourcePolicy, "bind", forbidden_load)
    with pytest.raises(ValueError, match="not ready|development-only|unsupported prospective"):
        G.prepare(training["config"], training["prepared"])
    assert not training["prepared"].parent.exists() and not (training["root"] / "runs").exists()


def test_true_readiness_flag_does_not_override_real_quota_verification(training, monkeypatch):
    records = training["manifest"].read_text().splitlines()[:1]
    training["manifest"].write_text("\n".join(records) + "\n")
    lock = training["development"]
    lock["inputs"][str(training["manifest"])]["sha256"] = SP.sha256(training["manifest"])
    training["out"].write_text(json.dumps(lock))
    training["config_doc"]["development"] = pin(training["out"])
    save_config(training)
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="mappings differ|readiness metadata differs"):
        G.prepare(training["config"], training["prepared"])
    assert not training["prepared"].parent.exists()


@pytest.mark.parametrize("change", ["height24", "bottleneck32", "vocabulary_order", "restoration", "bn_unfrozen"])
def test_generated_initializer_must_keep_the_released_geometry_and_class_order(training, monkeypatch, change):
    checkpoint = copy.deepcopy(training["checkpoint"])
    if change == "height24":
        checkpoint["height"] = 24
    elif change == "bottleneck32":
        checkpoint["bottleneck"] = 32
    elif change == "vocabulary_order":
        checkpoint["vocab"] = list(reversed(checkpoint["vocab"]))
    elif change == "restoration":
        checkpoint["restored_from"]["sha256"] = "0" * 64
    else:
        checkpoint["freeze_bn_stats"] = False
    torch.save(checkpoint, training["init"])
    # Explicit generated fixture pin; production release constants are unchanged.
    monkeypatch.setattr(G, "INIT_SHA", SP.sha256(training["init"]))
    training["config_doc"]["init"] = pin(training["init"])
    save_config(training)
    with pytest.raises(ValueError, match="architecture/restoration"):
        G.prepare(training["config"], training["prepared"])
    assert not training["prepared"].exists()


@pytest.mark.parametrize("change", ["unreserved_vn", "wrong_train_role", "unsafe_jiten_row"])
def test_source_boundaries_and_whole_label_whitelist_are_checked_before_tensors(training, monkeypatch, change):
    if change == "unreserved_vn":
        training["policy_doc"]["forbidden_vns"] = []
    elif change == "wrong_train_role":
        training["policy_doc"]["shards"][0]["split"] = "dev"
    else:
        with np.load(training["shard"], allow_pickle=False) as shard:
            arrays = {key: shard[key] for key in shard.files}
        arrays["labels"] = np.array(json.dumps(["あ漢い", "人入"]))
        arrays["data"] = arrays["data"].astype(object)  # any pixel access would fail
        np.savez(training["shard"], **arrays)
        training["policy_doc"]["shards"][0].update(sha256=SP.sha256(training["shard"]),
                                                  bytes=training["shard"].stat().st_size)
    training["policy"].write_text(json.dumps(training["policy_doc"]))
    training["config_doc"]["source_policy"] = pin(training["policy"])
    save_config(training)
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="reserve every|training shards|unsafe whole row"):
        G.prepare(training["config"], training["prepared"])
    assert not training["prepared"].exists()


@pytest.mark.parametrize("extra", ["val", "select", "height", "teacher", "arguments"])
def test_arbitrary_geometry_validation_and_teacher_arguments_cannot_enter_config(training, monkeypatch, extra):
    training["config_doc"][extra] = "must-not-open"
    save_config(training)
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="exact schema-1"):
        G.prepare(training["config"], training["prepared"])


def test_prepare_is_metadata_only_atomic_and_does_not_overwrite(training, monkeypatch):
    with np.load(training["shard"], allow_pickle=False) as shard:
        arrays = {key: shard[key] for key in shard.files}
    arrays["data"] = arrays["data"].astype(object)
    np.savez(training["shard"], **arrays)
    training["policy_doc"]["shards"][0].update(sha256=SP.sha256(training["shard"]),
                                              bytes=training["shard"].stat().st_size)
    training["policy"].write_text(json.dumps(training["policy_doc"]))
    training["config_doc"]["source_policy"] = pin(training["policy"])
    save_config(training)
    monkeypatch.setattr(train2, "load_shard", forbidden_load)
    monkeypatch.setattr(G.selector, "load_pixels", forbidden_load)
    identity = G.prepare(training["config"], training["prepared"])
    assert identity["quality_measured"] is False and identity["architecture"]["height"] == 32
    assert identity["architecture"]["bottleneck"] == 96 and identity["checkpoint_steps"] == [1, 2]
    assert "--keep-evals" in identity["argv"] and "--val" not in identity["argv"] and "--select" not in identity["argv"]
    raw = training["prepared"].read_bytes()
    with pytest.raises(FileExistsError):
        G.prepare(training["config"], training["prepared"])
    with pytest.raises(FileExistsError):
        G.atomic_freeze(training["prepared"], {"malicious": "rewrite"})
    assert training["prepared"].read_bytes() == raw
    assert list(training["prepared"].parent.iterdir()) == [training["prepared"]]


@pytest.mark.parametrize("change", ["config", "driver", "geometry", "policy", "init", "development", "kanji"])
def test_changed_fingerprint_refuses_before_checkpoint_or_crop_load(training, monkeypatch, change):
    G.prepare(training["config"], training["prepared"])
    monkeypatch.setattr(torch, "load", forbidden_load)
    monkeypatch.setattr(train2, "load_shard", forbidden_load)
    if change == "driver":
        original = G.sha256
        monkeypatch.setattr(G, "sha256", lambda path: "0" * 64 if str(path).endswith("generic_fl10_train.py")
                            else original(path))
    elif change == "geometry":
        monkeypatch.setattr(train2, "H", 24)
    else:
        path = training["out"] if change == "development" else training[change]
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint changed|input changed"):
        G.verify_prepared(training["prepared"])
    assert not (training["root"] / "runs").exists()


def test_fixed_height_flows_through_real_generated_driver_and_retains_every_checkpoint(training, monkeypatch):
    G.prepare(training["config"], training["prepared"])
    monkeypatch.setattr(train2.os, "_exit", lambda code: None)
    monkeypatch.setattr(G.selector, "select", forbidden_load)
    monkeypatch.setattr(G.selector, "load_pixels", forbidden_load)
    original = train2.TinyVNOCR.forward
    heights = []

    def forward(model, x):
        heights.append(x.shape[2])
        return original(model, x)

    monkeypatch.setattr(train2.TinyVNOCR, "forward", forward)
    G.worker(training["prepared"])
    identity, checksum = G.verify_prepared(training["prepared"])
    inventory = G.retain_inventory(training["prepared"], identity, checksum)
    assert heights == [32, 32] and [record["step"] for record in inventory["checkpoints"]] == [1, 2]
    for record in inventory["checkpoints"]:
        checkpoint = torch.load(record["file"], weights_only=False)
        assert checkpoint["height"] == 32 and checkpoint["bottleneck"] == 96 and checkpoint["val"] == {}
        execution = checkpoint["source_policy_execution"]
        assert execution["geometry"]["height"] == 32 and execution["prospective_driver"]["prepared_sha256"] == checksum
        assert execution["prospective_driver"]["development_sha256"] == identity["development_sha256"]
        assert checkpoint["source_policy"]["ledger"]["forwarded"] == record["step"] * 2
    assert not list(Path(identity["run_directory"]).glob("ckpt_step*.tmp"))
    last = Path(identity["run_directory"]) / "ckpt_last.pt"
    last_sha = SP.sha256(last)
    altered = G.context(training["prepared"], identity, checksum)
    altered["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="execution/vocabulary identity differs"):
        train2.main(identity["argv"], execution_context=altered)
    assert SP.sha256(last) == last_sha and heights == [32, 32]
    frozen = Path(identity["run_directory"]) / "retained-checkpoints.json"
    raw = frozen.read_bytes()
    assert G.retain_inventory(training["prepared"], identity, checksum) == inventory and frozen.read_bytes() == raw
    Path(inventory["checkpoints"][0]["file"]).unlink()
    with pytest.raises(ValueError, match="complete evenly spaced"):
        G.retain_inventory(training["prepared"], identity, checksum)


def test_nonuniform_checkpoint_schedule_is_refused(training, monkeypatch):
    training["config_doc"]["recipe"].update(steps=3, eval_every=2)
    save_config(training)
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="evenly spaced"):
        G.prepare(training["config"], training["prepared"])


def test_prospective_trainer_context_requires_a_policy_before_outputs_or_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr(train2, "load_shard", forbidden_load)
    with pytest.raises(ValueError, match="execution context requires a source policy"):
        train2.main(["--run", "invalid", "--out", str(tmp_path / "out"), "--steps", "1",
                     "--src", "unbound:1:must-not-open.npz"], execution_context={"driver": "generated"})
    assert not (tmp_path / "out").exists()


def test_development_lock_byte_drift_during_preparation_cannot_publish_identity(training, monkeypatch):
    original, calls = G.checked_development, []

    def drift(path):
        result = original(path)
        calls.append(path)
        if len(calls) == 2:
            with path.open("ab") as stream:
                stream.write(b"\n")  # semantically unchanged JSON still changes the frozen byte identity
        return result

    monkeypatch.setattr(G, "checked_development", drift)
    with pytest.raises(ValueError, match="development preparation changed during"):
        G.prepare(training["config"], training["prepared"])
    assert not training["prepared"].parent.exists()


def test_same_geometry_alternative_baseline_cannot_weaken_released_comparison(training, monkeypatch):
    # The generated selector lock is valid; only the prospective driver's released
    # baseline constraint differs, so matching geometry is deliberately insufficient.
    monkeypatch.setattr(G, "BASELINE_SHA", "0" * 64)
    monkeypatch.setattr(torch, "load", forbidden_load)
    monkeypatch.setattr(SP.SourcePolicy, "bind", forbidden_load)
    with pytest.raises(ValueError, match="exact released W8 baseline"):
        G.prepare(training["config"], training["prepared"])
    assert not training["prepared"].parent.exists()

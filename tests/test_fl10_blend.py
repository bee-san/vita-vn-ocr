"""Offline blend invariants using generated weights, without screenshots or inference."""

import copy
import hashlib
import json
import struct
import zlib
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import fl10_blend as FB


@pytest.fixture
def parents():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    # Match the released vocabulary byte budget with generated single characters.
    vocab = [chr(0x21 + index) for index in range(80)] + [chr(0x4E00 + index) for index in range(3419)] + ["〓"]
    model = FB.TinyVNOCR(len(vocab) + 1, **FB.ARCH_KEYS)
    initial = {**FB.ARCH, "vocab": vocab, "step": 0, "freeze_bn_stats": True, "unk": "〓",
               "model": model.state_dict()}
    initial["model"]["head.bias"].zero_()
    final = copy.deepcopy(initial)
    final["step"] = 2000
    final["model"]["head.bias"].add_(0.125)
    final["model"]["features.0.0.weight"].mul_(0.75)
    final.update(opt={"untrusted": "optimizer"}, torch_rng="rng", source_policy="ledger")
    yield initial, final, {"vocab": [""] + vocab}
    torch.set_num_threads(previous)


@pytest.fixture
def sources(tmp_path, monkeypatch, parents):
    initial, final, _ = parents
    initial_path, final_path = tmp_path / "initial.pt", tmp_path / "final.pt"
    for path, ck in ((initial_path, initial), (final_path, final)):
        path.write_bytes(FB.checkpoint_bytes(ck))
    baseline, final_blob = tmp_path / "baseline.vocr", tmp_path / "final.vocr"
    baseline.write_bytes(FB.export_bytes(initial, FB.sha256(initial_path)))
    final_blob.write_bytes(FB.export_bytes(final, FB.sha256(final_path)))
    for name, path in (("INIT_SHA", initial_path), ("FINAL_SHA", final_path),
                       ("BASELINE_SHA", baseline), ("FINAL_BLOB_SHA", final_blob)):
        monkeypatch.setattr(FB, name, FB.sha256(path))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"test_or_google_evaluated": False, "runtime_promotion": False,
                                    "strict_general_dev_improvement": False,
                                    "baseline": {"blob_sha256": FB.BASELINE_SHA},
                                    "candidates": [{"step": 2000, "status": "scored",
                                                    "checkpoint": str(final_path),
                                                    "checkpoint_sha256": FB.FINAL_SHA,
                                                    "blob": str(final_blob),
                                                    "blob_sha256": FB.FINAL_BLOB_SHA}]}))
    monkeypatch.setattr(FB, "SELECTION_SHA", FB.sha256(selection))
    return initial_path, selection, baseline, tmp_path / "derived"


def test_endpoints_copy_instead_of_cancellation_or_aliasing():
    initial = {"weight": torch.tensor([1e20, -0.0])}
    final = {"weight": torch.tensor([1.0, 0.0])}
    for alpha, endpoint in ((Fraction(0), initial), (Fraction(1), final)):
        blended = FB.blend_state(initial, final, alpha, {"weight"})
        assert FB.tensor_bytes(blended["weight"]) == FB.tensor_bytes(endpoint["weight"])
        blended["weight"].zero_()
        assert endpoint["weight"][0] != 0


def test_fraction_grid_and_whole_parameter_arithmetic(parents):
    initial, final, reference = parents
    parameters, buffers = FB.validate_parents(initial, final, reference)
    assert FB.GRID == tuple(map(Fraction, ("0", "1/16", "1/8", "1/4", "1/2", "3/4", "1")))
    state = FB.blend_state(initial["model"], final["model"], Fraction(1, 4), parameters)
    assert all(FB.tensor_bytes(state[k]) == FB.tensor_bytes(initial["model"][k]) for k in buffers)
    assert len(buffers) == 54
    assert torch.equal(state["head.bias"], initial["model"]["head.bias"] + 0.03125)
    toy = FB.blend_state({"weight": torch.tensor([1., 2., 0.])},
                         {"weight": torch.tensor([5., -8., 1.])}, Fraction(1, 4), {"weight"})
    assert torch.equal(toy["weight"], torch.tensor([2., -0.5, 0.25]))
    with pytest.raises(ValueError, match="grid"):
        FB.blend_state(initial["model"], final["model"], Fraction(1, 3), parameters)


@pytest.mark.parametrize("key", ["features.0.1.running_mean", "features.0.1.running_var",
                                 "features.0.1.num_batches_tracked"])
def test_any_bn_buffer_change_is_rejected(parents, key):
    initial, final, reference = parents
    final["model"][key].add_(1)
    with pytest.raises(ValueError, match="buffer changed"):
        FB.validate_parents(initial, final, reference)


def test_buffer_identity_includes_signed_zero(parents):
    initial, final, reference = parents
    final["model"]["features.0.1.running_mean"][0] = -0.0
    with pytest.raises(ValueError, match="buffer changed"):
        FB.validate_parents(initial, final, reference)


@pytest.mark.parametrize("change", ["width", "width_type", "step", "step_type", "bn", "vocab", "keys",
                                  "dtype", "shape", "nonfinite", "nontensor"])
def test_metadata_state_types_and_actual_architecture_are_required(parents, change):
    initial, final, reference = parents
    if change == "width":
        final["width"] = 0.5
    elif change == "width_type":
        final["width"] = True
    elif change == "step":
        final["step"] = 1200
    elif change == "step_type":
        final["step"] = 2000.0
    elif change == "bn":
        final["freeze_bn_stats"] = False
    elif change == "vocab":
        final["vocab"][0], final["vocab"][1] = final["vocab"][1], final["vocab"][0]
    elif change == "keys":
        final["model"].pop("head.bias")
    elif change == "dtype":
        final["model"]["head.bias"] = final["model"]["head.bias"].double()
    elif change == "shape":
        final["model"]["head.bias"] = final["model"]["head.bias"][:-1]
    elif change == "nonfinite":
        final["model"]["head.bias"][0] = float("nan")
    else:
        final["model"]["head.bias"] = []
    with pytest.raises(ValueError):
        FB.validate_parents(initial, final, reference)


def test_serialization_and_state_digest_are_stable_and_content_bound(parents):
    initial, _, _ = parents
    assert FB.checkpoint_bytes(initial) == FB.checkpoint_bytes(initial)
    digest = FB.state_sha256(initial["model"])
    assert digest == FB.state_sha256(dict(reversed(list(initial["model"].items()))))
    initial["model"]["head.bias"][0] += 1
    assert digest != FB.state_sha256(initial["model"])


def test_publishes_complete_pinned_inventory_without_forward_or_training_state(sources, parents, monkeypatch):
    def forbidden_forward(*args, **kwargs):
        raise AssertionError("blend producer must not run inference")

    monkeypatch.setattr(FB.TinyVNOCR, "forward", forbidden_forward)
    inventory = FB.produce(*sources)
    out = sources[-1]
    assert json.loads((out / "inventory.json").read_text()) == inventory
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"alpha_{a.numerator:02d}_{a.denominator:02d}.pt" for a in FB.GRID] + ["inventory.json"])
    assert inventory["endpoint_payloads_identical"] == ["0", "1"]
    assert inventory["batchnorm_buffers_identical"] == 54
    assert (inventory["model_forwards"], inventory["quality_measured"]) == (0, False)
    assert (inventory["blob_bytes"], inventory["arena_bytes"]) == (657664, 124832)
    for candidate, alpha in zip(inventory["candidates"], FB.GRID):
        path = Path(candidate["checkpoint"])
        assert FB.sha256(path) == candidate["checkpoint_sha256"]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert not {"opt", "torch_rng", "source_policy", "source_policy_execution", "args"} & ck.keys()
        assert ck["step"] == 2000 and ck["freeze_bn_stats"] is True
        assert ck["derived_from"]["alpha"] == str(alpha)
        assert candidate["model_state_sha256"] == FB.state_sha256(ck["model"])
        raw = FB.export_bytes(ck, FB.sha256(path))
        assert hashlib.sha256(raw).hexdigest() == candidate["export_sha256"]
    assert inventory["candidates"][0]["model_state_sha256"] == FB.state_sha256(parents[0]["model"])
    assert inventory["candidates"][-1]["model_state_sha256"] == FB.state_sha256(parents[1]["model"])


@pytest.mark.parametrize("which", [0, 2])
def test_changed_input_pin_rejected_before_destination_exists(sources, which):
    sources[which].write_bytes(sources[which].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="fixed pin"):
        FB.produce(*sources)
    assert not sources[-1].exists()


def test_original_selection_requires_exact_scheduled_endpoint(sources, monkeypatch):
    initial, selection, baseline, out = sources
    doc = json.loads(selection.read_text())
    doc["candidates"][0]["checkpoint_sha256"] = "0" * 64
    selection.write_text(json.dumps(doc))
    monkeypatch.setattr(FB, "SELECTION_SHA", FB.sha256(selection))
    with pytest.raises(ValueError, match="fixed checkpoint/export pins"):
        FB.produce(initial, selection, baseline, out)
    assert not out.exists()


def test_endpoint_export_failure_precedes_any_artifact_publication(sources, monkeypatch):
    original = FB.export_bytes

    def corrupt(checkpoint, digest):
        raw = bytearray(original(checkpoint, digest))
        raw[48] ^= 1  # Legal source header changes must remain irrelevant.
        if checkpoint["derived_from"]["alpha"] == "0":
            first_weight_offset = struct.unpack_from("<I", raw, 64 + 16)[0]
            raw[first_weight_offset] ^= 1
            struct.pack_into("<I", raw, 40, zlib.crc32(raw[64:]) & 0xFFFFFFFF)
        return bytes(raw)

    monkeypatch.setattr(FB, "export_bytes", corrupt)
    with pytest.raises(ValueError, match="endpoint"):
        FB.produce(*sources)
    assert not sources[-1].exists()


def test_existing_destination_is_never_modified_even_if_incomplete(sources):
    out = sources[-1]
    out.mkdir()
    sentinel = out / "alpha_00_01.pt"
    sentinel.write_bytes(b"preserve incomplete previous run")
    with pytest.raises(FileExistsError, match="already exists"):
        FB.produce(*sources)
    assert sentinel.read_bytes() == b"preserve incomplete previous run"
    assert sorted(p.name for p in out.iterdir()) == [sentinel.name]

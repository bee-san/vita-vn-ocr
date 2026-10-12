#!/usr/bin/env python3
"""Freeze seven pinned FL10 baseline blends; export checks only, never OCR scoring."""

import argparse
import hashlib
import io
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

import export
from hiragana_select import sha256, validate_blob
from model import ARCH_KEYS, TinyVNOCR

ROOT = Path(__file__).resolve().parents[1]
INIT_SHA = "9fbcb3b2c620a582a003353abb9481d0ee05eb8a5709d4bb1481d4e733597b47"
FINAL_SHA = "70edc583c9c798bcbaffb7ad445455017cf103afaeba0ec6fd31ddf076cb5a6f"
BASELINE_SHA = "500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84"
FINAL_BLOB_SHA = "faa275045bc01e8dd431ec1620bb988a3d38ca707a08ebe5655f1922885310e9"
SELECTION_SHA = "0ac687b9e74d960beb0aa400576c483b7ac31347050ae1860a84657052d8e547"
GRID = tuple(Fraction(n, d) for n, d in ((0, 1), (1, 16), (1, 8), (1, 4), (1, 2), (3, 4), (1, 1)))
ARCH = {**ARCH_KEYS, "height": 32}
ARITHMETIC = "CPU float32: delta=final-init; scaled=delta*float(alpha); result=init+scaled; endpoints cloned"
CODE = ("train/fl10_blend.py", "train/model.py", "train/export.py", "train/hiragana_select.py")


def pinned(path, expected):
    path = Path(path).resolve()
    if sha256(path) != expected:
        raise ValueError(f"input SHA256 differs from fixed pin: {path}")
    return path


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def validate_parents(initial, final, reference):
    """Match actual model state, metadata and vocabulary before deriving any artifact."""
    model = TinyVNOCR(len(reference["vocab"]), **ARCH_KEYS)
    schema = model.state_dict()
    parameters = set(dict(model.named_parameters()))
    buffers = set(dict(model.named_buffers()))
    if len(buffers) != 54:
        raise ValueError("fixed FL10 must have exactly 54 BatchNorm buffers")
    for ck, step in ((initial, 0), (final, 2000)):
        if not isinstance(ck, dict):
            raise ValueError("parent checkpoint must be a dictionary")
        if any(type(ck.get(k)) is not type(v) or ck[k] != v for k, v in ARCH.items()):
            raise ValueError("parent architecture differs from fixed FL10")
        if (type(ck.get("step")) is not int or ck["step"] != step
                or ck.get("freeze_bn_stats") is not True or ck.get("unk") != "〓"
                or ck.get("vocab") != reference["vocab"][1:]
                or len(ck["vocab"]) != 3500 or ck["vocab"][-1] != "〓"):
            raise ValueError("parent step, vocabulary or frozen-BN metadata differs")
        state = ck.get("model")
        if not isinstance(state, dict) or state.keys() != schema.keys():
            raise ValueError("parent state keys differ from actual FL10 architecture")
        for key, tensor in state.items():
            expected = schema[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.dtype != expected.dtype or tensor.shape != expected.shape
                    or tensor.device.type != "cpu"):
                raise ValueError(f"parent tensor type/layout/shape differs: {key}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"parent tensor is nonfinite: {key}")
    for key in initial["model"]:
        left, right = initial["model"][key], final["model"][key]
        if (key in buffers or not left.is_floating_point()) and tensor_bytes(left) != tensor_bytes(right):
            raise ValueError(f"BatchNorm/nonfloating buffer changed: {key}")
    return parameters, buffers


def blend_state(initial, final, alpha, parameters):
    """Apply three explicit float32 operations only to named model parameters."""
    if alpha not in GRID:
        raise ValueError("alpha is outside the fixed seven-fraction grid")
    result = {}
    for key, left in initial.items():
        if key not in parameters or alpha == 0:
            value = left.clone()
        elif alpha == 1:
            value = final[key].clone()
        else:
            delta = torch.sub(final[key], left)
            scaled = torch.mul(delta, float(alpha))
            value = torch.add(left, scaled)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"blend produced nonfinite tensor: {key}")
        result[key] = value.contiguous()
    return result


def checkpoint_bytes(checkpoint):
    # A file-like object avoids torch.save embedding the destination basename.
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    return stream.getvalue()


def export_bytes(checkpoint, checkpoint_sha):
    model = TinyVNOCR(len(checkpoint["vocab"]) + 1, **ARCH_KEYS).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    raw, _ = export.build_blob(export.model_ops(model, 32), checkpoint["vocab"], 32,
                              int8=True, max_width=1024, source_sha=bytes.fromhex(checkpoint_sha))
    return raw


def check_endpoint_payload(alpha, raw, expected):
    if raw[64:] != expected[64:]:
        raise ValueError(f"alpha {alpha} export payload differs from its pinned endpoint")


def produce(initial_path, selection_path, baseline_path, out):
    """Publish the complete immutable inventory last, with no inference or data loading."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("blend destination already exists; use a new directory")
    initial_path = pinned(initial_path, INIT_SHA)
    baseline_path = pinned(baseline_path, BASELINE_SHA)
    selection_path = pinned(selection_path, SELECTION_SHA)
    selection_hash = sha256(selection_path)
    selection = json.loads(selection_path.read_text())
    if (selection.get("test_or_google_evaluated") is not False
            or selection.get("runtime_promotion") is not False
            or selection.get("strict_general_dev_improvement") is not False
            or selection.get("baseline", {}).get("blob_sha256") != BASELINE_SHA):
        raise ValueError("require the completed failed development-only experiment")
    matches = [c for c in selection.get("candidates", [])
               if type(c.get("step")) is int and c["step"] == 2000]
    if len(matches) != 1 or matches[0].get("status") != "scored":
        raise ValueError("old selection must identify exactly one scored step-2000 endpoint")
    selected = matches[0]
    if selected.get("checkpoint_sha256") != FINAL_SHA or selected.get("blob_sha256") != FINAL_BLOB_SHA:
        raise ValueError("old endpoint differs from fixed checkpoint/export pins")
    final_path = pinned(selected["checkpoint"], FINAL_SHA)
    final_blob_path = pinned(selected["blob"], FINAL_BLOB_SHA)
    inputs = {name: {"path": str(path), "sha256": sha256(path)} for name, path in (
        ("initial", initial_path), ("final", final_path), ("baseline", baseline_path),
        ("final_export", final_blob_path), ("original_selection", selection_path))}
    code_hashes = {name: sha256(ROOT / name) for name in CODE}
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    reference = export.read_blob(baseline_path)
    validate_blob(reference)
    validate_blob(export.read_blob(final_blob_path), reference)
    parameters, buffers = validate_parents(initial, final, reference)
    parents = {name: {**inputs[name], "model_state_sha256": state_sha256(ck["model"]), "step": ck["step"]}
               for name, ck in (("initial", initial), ("final", final))}
    payloads, candidates = {}, []
    for alpha in GRID:
        name = f"alpha_{alpha.numerator:02d}_{alpha.denominator:02d}.pt"
        ck = {**ARCH, "vocab": list(initial["vocab"]), "unk": "〓", "step": 2000,
              "freeze_bn_stats": True,
              "derived_from": {"kind": "fl10-baseline-blend", "parents": parents, "alpha": str(alpha),
                               "scope": "all named floating model parameters", "arithmetic": ARITHMETIC},
              "model": blend_state(initial["model"], final["model"], alpha, parameters)}
        raw = checkpoint_bytes(ck)
        digest = hashlib.sha256(raw).hexdigest()
        blob = export_bytes(ck, digest)
        validate_blob(export.read_blob(blob), reference)
        if alpha in (0, 1):
            check_endpoint_payload(alpha, blob, (baseline_path if alpha == 0 else final_blob_path).read_bytes())
        payloads[name] = raw
        candidates.append({"alpha": str(alpha), "checkpoint": str(out / name), "checkpoint_sha256": digest,
                           "model_state_sha256": state_sha256(ck["model"]), "step": 2000,
                           "export_sha256": hashlib.sha256(blob).hexdigest(),
                           "export_payload_sha256": hashlib.sha256(blob[64:]).hexdigest()})
    inventory = {"schema": 1, "kind": "fl10-baseline-blend-inventory", "grid": [str(a) for a in GRID],
                 "scope": "all named floating model parameters", "arithmetic": ARITHMETIC,
                 "inputs": inputs, "parents": parents, "code_sha256": code_hashes,
                 "dependency_sha256": {name: sha256(ROOT / name) for name in
                                       ("requirements.lock", "requirements-ci.lock")},
                 "environment": {"python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
                                 "torch": torch.__version__, "numpy": np.__version__},
                 "batchnorm_buffers_identical": len(buffers), "endpoint_payloads_identical": ["0", "1"],
                 "blob_bytes": reference["bytes"],
                 "arena_bytes": export.plan_arena(reference, max_width=1024, mode=export.MODE_W8I),
                 "candidates": candidates, "model_forwards": 0, "quality_measured": False,
                 "test_or_google_evaluated": False, "runtime_promotion": False}
    if sha256(selection_path) != selection_hash or any(sha256(v["path"]) != v["sha256"] for v in inputs.values()):
        raise ValueError("parent/selection input changed during derivation")
    if code_hashes != {name: sha256(ROOT / name) for name in CODE}:
        raise ValueError("producer/export source changed during derivation")
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        with (out / name).open("xb") as stream:
            stream.write(raw)
    with (out / "inventory.json").open("x") as stream:
        json.dump(inventory, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--original-selection", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    inventory = produce(args.initial, args.original_selection, args.baseline, args.out)
    print(json.dumps({"inventory": str(args.out.resolve() / "inventory.json"),
                      "checkpoints": len(inventory["candidates"]), "model_forwards": 0}))


if __name__ == "__main__":
    main()

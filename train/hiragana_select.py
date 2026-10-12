#!/usr/bin/env python3
"""Lock an FL10 int8 checkpoint using development data alone.

Score the released baseline and checkpoint exports with the scalar C W8I runtime.
Both full-text guards must pass; otherwise keep the released baseline. This tool
never opens the shared or private test sets and does not establish test accuracy.
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backends import NativeVocr  # noqa: E402
from metrics import score_lines  # noqa: E402
from sets import LOCAL_MANIFESTS, SETS, load_real, load_synth_val, register_local  # noqa: E402

import export  # noqa: E402
from common import line_polarity, to_gray  # noqa: E402
from model import ARCH_KEYS, TinyVNOCR  # noqa: E402

MODEL_BYTES, ARENA_BYTES = 657664, 124832
OP_KEYS = "type kh kw sh sw ph pw cin cout h_in h_out relu res collapse".split()
RULE = {"qualify": "real_dev.full_errors <= baseline and synthetic_dev.full_errors <= baseline",
        "rank": ["real_dev.hiragana_errors", "real_dev.full_errors", "synthetic_dev.full_errors",
                 "step", "checkpoint_path"],
        "replace_baseline": "qualified checkpoint strictly reduces real_dev.hiragana_errors"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_blob(blob, reference=None):
    """Require the deployment FL10 topology, vocabulary order and planned memory."""
    if not blob["int8"] or (blob["height"], blob["max_width"], blob["stride"]) != (32, 1024, 8):
        raise ValueError("selection requires FL10 int8, height 32, width 1024, stride 8")
    expected = reference or {"ops": export.model_ops(TinyVNOCR(blob["n_classes"], **ARCH_KEYS).eval(), 32),
                             "vocab": blob["vocab"]}
    if blob["vocab"] != expected["vocab"]:
        raise ValueError("FL10 vocabulary order differs from the released baseline")
    if len(blob["ops"]) != len(expected["ops"]) or any(
            any(actual[k] != want[k] for k in OP_KEYS)
            or (actual["bias"] is None) != (want["bias"] is None)
            for actual, want in zip(blob["ops"], expected["ops"])):
        raise ValueError("incompatible FL10 architecture")
    if (blob["bytes"], export.plan_arena(blob, max_width=1024, mode=export.MODE_W8I)) != (
            MODEL_BYTES, ARENA_BYTES):
        raise ValueError("FL10 model or arena memory changed")


def load_real_development(manifest, pad=4):
    """Reject non-development roles before opening images, with explicit padding."""
    manifest = Path(manifest).resolve()
    if type(pad) is not int or not 0 <= pad <= 32:
        raise ValueError("development padding must be an integer between 0 and 32")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("split") != "dev" for row in rows):
        raise ValueError("real selection manifest must be development-only")
    name = "hiragana-selection-dev"
    if name in SETS or name in LOCAL_MANIFESTS:
        raise ValueError("development set name already registered")
    try:
        register_local(name, manifest)
        real, _ = load_real(name, ROOT / "data" / "eval", pad=pad)
    finally:
        SETS.pop(name, None)
        LOCAL_MANIFESTS.pop(name, None)
    if not real or any(not line.text for line in real):
        raise ValueError("development datasets must contain scored nonempty labels")
    for line in real:
        gray = to_gray(line.crop_rgb)
        line.crop_rgb = np.ascontiguousarray(255 - gray if line_polarity(gray) else gray)
    return real, {"split": "dev", "manifest_sha256": sha256(manifest),
                  "sessions": sorted({row["session"] for row in rows}),
                  "lines": len(real), "pad": pad, "polarity": "console"}


def load_synthetic_development(synthetic):
    """Validate role and book cohort; score all stored console-polarity crops."""
    synthetic = Path(synthetic).resolve()
    with np.load(synthetic, allow_pickle=False) as shard:
        if "meta" not in shard:
            raise ValueError("synthetic selection shard needs development-role metadata")
        labels, metas = json.loads(str(shard["labels"])), json.loads(str(shard["meta"]))
        if not labels or len(metas) != len(labels) or any(
                not isinstance(meta, dict) or meta.get("split") != "dev" for meta in metas):
            raise ValueError("synthetic selection shard must be development-only")
        cohorts = {meta.get("book_split_sha256") for meta in metas}
        if len(cohorts) != 1 or not isinstance(next(iter(cohorts)), str):
            raise ValueError("synthetic development needs one held-out book cohort hash")
        cohort = next(iter(cohorts))
        if len(cohort) != 64 or any(c not in "0123456789abcdef" for c in cohort):
            raise ValueError("invalid synthetic held-out book cohort hash")
    synth = load_synth_val(synthetic, limit=0)
    if not synth or any(not line.text for line in synth):
        raise ValueError("development datasets must contain scored nonempty labels")
    return synth, {"split": "dev", "npz_sha256": sha256(synthetic),
                   "book_split_sha256": cohort, "lines": len(synth),
                   "polarity": "stored console crops"}


def load_development(manifest, synthetic):
    real, real_identity = load_real_development(manifest)
    synth, synth_identity = load_synthetic_development(synthetic)
    return real, synth, {"real_dev": real_identity, "synthetic_dev": synth_identity}


def load_guards(real_guards=(), synthetic_guards=()):
    lines, identities = {}, {}
    for name, path, pad in real_guards:
        if not name or name in lines or name in ("real_dev", "synthetic_dev"):
            raise ValueError("development guards require distinct non-reserved names")
        lines[name], identities[name] = load_real_development(path, pad)
    for name, path in synthetic_guards:
        if not name or name in lines or name in ("real_dev", "synthetic_dev"):
            raise ValueError("development guards require distinct non-reserved names")
        lines[name], identities[name] = load_synthetic_development(path)
    return lines, identities


def score_blob(path, real, synth, guards=None):
    rec = NativeVocr(path)
    try:
        memory = {key: rec.info[key] for key in ("blob_bytes", "arena_bytes")}
        if (memory["blob_bytes"], memory["arena_bytes"]) != (MODEL_BYTES, ARENA_BYTES):
            raise ValueError("C runtime FL10 model or arena memory changed")
        scores = {}
        datasets = {"real_dev": real, "synthetic_dev": synth, **(guards or {})}
        for name, lines in datasets.items():
            predictions = rec.read([line.crop_rgb for line in lines])
            if len(predictions) != len(lines):
                raise ValueError("recognizer did not return one prediction per development crop")
            scores[name] = score_lines([{"text": line.text, "pred": prediction}
                                        for line, prediction in zip(lines, predictions)])
        if not scores["real_dev"]["hiragana"]["chars"]:
            raise ValueError("real development set has no reference hiragana")
        return {"scores": scores, "memory": memory, "backend": "vocr", "mode": "w8i"}
    finally:
        rec.close()


def export_checkpoint(path, reference, directory):
    """Check metadata and actual op layout, then export the exact checkpoint bytes."""
    path = Path(path).resolve()
    source_hash = sha256(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if any(ck.get(key, default) != default for key, default in ARCH_KEYS.items()):
        raise ValueError("incompatible FL10 checkpoint architecture")
    if ck.get("height", 32) != 32 or list(ck["vocab"]) != reference["vocab"][1:]:
        raise ValueError("checkpoint height or vocabulary order differs from released FL10")
    step = ck.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint requires a nonnegative integer step")
    model, vocab, height = export.load_checkpoint(path)
    raw, _ = export.build_blob(export.model_ops(model, height), vocab, height, int8=True,
                              max_width=1024, source_sha=bytes.fromhex(source_hash))
    if sha256(path) != source_hash:
        raise ValueError("checkpoint changed during export")
    validate_blob(export.read_blob(raw), reference)
    target = directory / f"{path.parent.name}-{path.stem}-{source_hash[:16]}.vocr"
    target.write_bytes(raw)
    return {"checkpoint": str(path), "checkpoint_sha256": source_hash, "step": step,
            "blob": str(target.resolve()), "blob_sha256": hashlib.sha256(raw).hexdigest()}


def choose(baseline, candidates):
    """Two full-text guards, deterministic ranking, and strict hiragana improvement."""
    base = baseline["scores"]
    qualified = []
    for candidate in candidates:
        if "scores" not in candidate:
            continue
        scores = candidate["scores"]
        candidate["qualified"] = all(scores[name]["all"]["errors"] <= base[name]["all"]["errors"]
                                     for name in ("real_dev", "synthetic_dev"))
        candidate["qualified"] &= all(
            name in scores and all(scores[name][script]["errors"] <= base[name][script]["errors"]
                                   for script in ("all", "hiragana"))
            for name in base if name not in ("real_dev", "synthetic_dev"))
        if candidate["qualified"]:
            qualified.append(candidate)
    ranked = sorted(qualified, key=lambda candidate: (
        candidate["scores"]["real_dev"]["hiragana"]["errors"],
        candidate["scores"]["real_dev"]["all"]["errors"],
        candidate["scores"]["synthetic_dev"]["all"]["errors"], candidate["step"], candidate["checkpoint"]))
    if ranked and ranked[0]["scores"]["real_dev"]["hiragana"]["errors"] < base["real_dev"]["hiragana"]["errors"]:
        return ranked[0], True
    return baseline, False


def select(baseline, manifest, synthetic, checkpoints, out, real_guards=(), synthetic_guards=()):
    out, baseline = Path(out).resolve(), Path(baseline).resolve()
    if out.exists():
        raise FileExistsError(f"completed selection already exists: {out}")
    real, synth, datasets = load_development(manifest, synthetic)
    guards, guard_identities = load_guards(real_guards, synthetic_guards)
    datasets.update(guard_identities)
    reference = export.read_blob(baseline)
    validate_blob(reference)
    baseline_hash = sha256(baseline)
    base = {"blob": str(baseline), "blob_sha256": baseline_hash, **score_blob(baseline, real, synth, guards)}
    directory = out.with_name(out.stem + "_exports")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".gitignore").write_text("*\n")
    candidates = []
    for path in sorted({str(Path(path).resolve()) for path in checkpoints}):
        record = {"checkpoint": path, "checkpoint_sha256": sha256(path)}
        try:
            record.update(export_checkpoint(path, reference, directory))
            record.update(score_blob(record["blob"], real, synth, guards))
            record["status"] = "scored"
        except (ValueError, AssertionError, RuntimeError, KeyError) as error:
            record.update(status="rejected", reason=str(error), qualified=False)
        candidates.append(record)
        print(json.dumps({"checkpoint": path, "status": record["status"],
                          "scores": record.get("scores"), "reason": record.get("reason")}), flush=True)
    if sha256(baseline) != baseline_hash:
        raise ValueError("baseline changed during selection")
    chosen, improved = choose(base, candidates)
    if sha256(chosen["blob"]) != chosen["blob_sha256"]:
        raise ValueError("chosen blob changed during selection")
    result = {"schema": 1, "selection_data": "development_only", "selection_rule": RULE,
              "additional_guards": {"names": list(guards),
                                    "qualify": "each full and hiragana error count <= its own baseline"},
              "datasets": datasets, "baseline": base, "candidates": candidates,
              "strict_dev_hiragana_improvement": improved,
              "chosen": {key: chosen[key] for key in ("blob", "blob_sha256")},
              "heldout_test_evaluated": False}
    if improved:
        result["chosen"].update({key: chosen[key] for key in ("checkpoint", "checkpoint_sha256", "step")})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-dev", type=Path, required=True)
    parser.add_argument("--checkpoints", action="append", required=True, help="checkpoint path/glob; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-guard", action="append", default=[], metavar="NAME:PAD:MANIFEST")
    parser.add_argument("--synthetic-guard", action="append", default=[], metavar="NAME:NPZ")
    args = parser.parse_args()
    paths = []
    for pattern in args.checkpoints:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            parser.error(f"no checkpoints match {pattern}")
        paths.extend(matches)
    try:
        real_guards = [(name, path, int(pad)) for name, pad, path in
                       (item.split(":", 2) for item in args.real_guard)]
        synthetic_guards = [tuple(item.split(":", 1)) for item in args.synthetic_guard]
        result = select(args.baseline, args.dev_manifest, args.synthetic_dev, paths, args.out,
                        real_guards, synthetic_guards)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"chosen": result["chosen"],
                      "strict_dev_hiragana_improvement": result["strict_dev_hiragana_improvement"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prospective source-balanced FL10 adaptation; development selection stays separate."""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import general_vn_select as selector
import general_vn_textbox_guard as textbox_guard
import train2
from kanji_policy import allowed_label, load_policy
from source_policy import SourcePolicy, json_digest, parse_json, sha256

ROOT = Path(__file__).resolve().parents[1]
ARCH = {"width": 1.0, "bottleneck": 96, "depth": 1, "ctx": 2, "ctx_k": 5, "height": 32}
INIT_SHA = "9fbcb3b2c620a582a003353abb9481d0ee05eb8a5709d4bb1481d4e733597b47"
VOCAB_SHA = "7f1cb2eb073ddae5f6c39bdd49562cd3bd3ac757f5ac8c0eca4bd98beda4fed8"
FP32_SHA = "a42818bfa81361ac907ce79ec471c9e0778aa2d7dbe1b0d801207d27d63ecc7e"
KANJI_POLICY_SHA = "8fc9ea20804cc45876e25b60ed669446dc75fc7f0b6b383d65cc4688e050f763"
BASELINE_SHA = "500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84"
CLASSES = 3500
RECIPE_KEYS = {"steps", "batch", "lr", "final_lr", "warmup", "wd", "train_scope", "seed", "threads", "eval_every"}
CODE = sorted(set(selector.CODE + textbox_guard.CODE + ["synth/kanji_policy.py", "train/generic_fl10_train.py",
                    "train/train2.py", "train/source_policy.py", "train/train.py", "train/restore_blob.py"]))


def pinned_path(row, base):
    if not isinstance(row, dict) or set(row) != {"file", "sha256"}:
        raise ValueError("config input requires exactly file and sha256")
    path = Path(row["file"])
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not isinstance(row["sha256"], str) or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None:
        raise ValueError("config input requires a lowercase SHA256")
    if sha256(path) != row["sha256"]:
        raise ValueError("frozen training input changed")
    return path


def checked_recipe(recipe):
    if not isinstance(recipe, dict) or set(recipe) != RECIPE_KEYS:
        raise ValueError("declare the complete bounded training recipe; arbitrary trainer arguments are unsupported")
    for key in ("steps", "batch", "threads", "eval_every"):
        if type(recipe[key]) is not int or recipe[key] <= 0:
            raise ValueError("training counts/threads must be positive integers")
    for key in ("seed", "warmup"):
        if type(recipe[key]) is not int or recipe[key] < 0:
            raise ValueError("seed/warmup must be nonnegative integers")
    for key in ("lr", "final_lr", "wd"):
        if type(recipe[key]) not in (int, float) or not math.isfinite(recipe[key]) or recipe[key] < 0:
            raise ValueError("optimizer recipe must contain finite nonnegative values")
    if (recipe["lr"] <= 0 or not 0 < recipe["final_lr"] <= 1 or recipe["warmup"] >= recipe["steps"]
            or recipe["train_scope"] not in ("all", "head") or recipe["steps"] % recipe["eval_every"]):
        raise ValueError("invalid scope/schedule; all retained checkpoints must be evenly spaced")
    return recipe


def checked_development(path):
    # Refuse a known unready/wrong-role artifact before source metadata or model tensors.
    raw = parse_json(path.read_bytes())
    if (not isinstance(raw, dict) or raw.get("schema") != 1
            or raw.get("test_or_google_evaluated") is not False or raw.get("candidates_evaluated") is not False):
        raise ValueError("training requires a separately frozen development-only preparation")
    selector.validate_protocol(raw.get("protocol"))
    if not isinstance(raw.get("readiness"), dict) or raw["readiness"].get("ready") is not True:
        raise ValueError("generic real development cohort is not ready; existing minimums remain required")
    lock, rows = selector.verify_lock(path)
    if lock["baseline_sha256"] != BASELINE_SHA:
        raise ValueError("generic FL10 preparation must compare against the exact released W8 baseline")
    return lock, rows


def checked_config(config_path):
    config_path = Path(config_path).resolve()
    before = sha256(config_path)
    doc = parse_json(config_path.read_bytes())
    required = {"schema", "kind", "init", "vocab", "kanji_policy", "source_policy", "development",
                "textbox_development", "run", "recipe", "sources"}
    if (not isinstance(doc, dict) or set(doc) != required or type(doc["schema"]) is not int
            or doc["schema"] != 2 or doc["kind"] != "general-vn-fl10-training-config"):
        raise ValueError("expected exact schema-2 general-vn-fl10-training config with mandatory textbox preparation")
    base = config_path.parent
    development = pinned_path(doc["development"], base)
    lock, rows = checked_development(development)
    textbox_development = pinned_path(doc["textbox_development"], base)
    textbox_guard.verify_prepared(textbox_development, line_lock=development)
    recipe = checked_recipe(doc["recipe"])
    run = doc["run"]
    if (not isinstance(run, dict) or set(run) != {"name", "out"} or not isinstance(run["name"], str)
            or re.fullmatch("[A-Za-z0-9_-]+", run["name"]) is None):
        raise ValueError("training run requires a simple name and explicit output root")
    out = Path(run["out"])
    out = (base / out).resolve() if not out.is_absolute() else out.resolve()
    policy_path = pinned_path(doc["source_policy"], base)
    policy = SourcePolicy(policy_path)
    development_vns = {row["vn"] for row in rows}
    if not development_vns <= policy.forbidden_vns:
        raise ValueError("training policy must reserve every generic development VN")
    # A matching path/hash cannot be hidden behind a different source name or role.
    dev_paths = {Path(name).resolve() for name in lock["inputs"]}
    dev_hashes = {row["sha256"] for row in lock["inputs"].values()}
    if set(policy.shards) & dev_paths or any(row["sha256"] in dev_hashes for row in policy.shards.values()):
        raise ValueError("training shard overlaps a bound development artifact")
    if any(row["sha256"] in dev_hashes or row.get("source_sha256") in dev_hashes for row in policy.assets.values()):
        raise ValueError("training asset overlaps a bound development source")
    sources, weights = {}, []
    if not isinstance(doc["sources"], list) or not doc["sources"]:
        raise ValueError("training requires explicitly weighted policy sources")
    for row in doc["sources"]:
        if not isinstance(row, dict) or set(row) != {"name", "weight", "augmentation"}:
            raise ValueError("source config supports only name, weight and augmentation")
        name, weight = row["name"], row["weight"]
        if not isinstance(name, str) or re.fullmatch("[A-Za-z0-9_-]+", name) is None or name in sources:
            raise ValueError("training source names must be unique simple strings")
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("training source weights must be finite and nonnegative")
        if type(row["augmentation"]) is not bool:
            raise ValueError("source augmentation must be an explicit boolean")
        sources[name] = sorted(path for path, record in policy.shards.items() if record["source"] == name)
        weights.append(weight)
    if sum(weights) <= 0:
        raise ValueError("training needs a positive source proposal weight")
    policy.bind(sources)  # byte/role/provenance checks only; never crop arrays
    init = pinned_path(doc["init"], base)
    vocab = pinned_path(doc["vocab"], base)
    if doc["init"]["sha256"] != INIT_SHA or doc["vocab"]["sha256"] != VOCAB_SHA:
        raise ValueError("training requires the exact restored released FL10 initializer and ordered vocabulary")
    values = vocab.read_text(encoding="utf-8").splitlines()
    if len(values) != CLASSES or len(set(values)) != CLASSES or any(len(c) != 1 for c in values) or values[-1] != "〓":
        raise ValueError("released FL10 requires 3500 ordered nonblank classes ending in the unknown class")
    kanji_path = pinned_path(doc["kanji_policy"], base)
    if doc["kanji_policy"]["sha256"] != KANJI_POLICY_SHA:
        raise ValueError("training requires the original frozen FL10 Jiten whitelist")
    kanji = load_policy(kanji_path)
    if kanji["vocab_sha256"] != VOCAB_SHA:
        raise ValueError("Jiten policy differs from the exact ordered released vocabulary")
    vocab_set = set(values)
    for path, binding in policy.shards.items():
        with np.load(path, allow_pickle=False) as shard:
            labels = parse_json(str(shard["labels"]))
        if (not isinstance(labels, list) or len(labels) != binding["samples"]
                or any(not isinstance(label, str) or not label or not set(label) <= vocab_set
                       or not allowed_label(label, kanji) for label in labels)):
            raise ValueError("training labels contain an unsafe whole row outside the vocabulary/Jiten whitelist")
    # Initial weights are accessed only after independently revalidated real readiness.
    checkpoint = torch.load(init, map_location="cpu", weights_only=False)
    if (checkpoint.get("vocab") != values or any(checkpoint.get(key) != value for key, value in ARCH.items())
            or checkpoint.get("freeze_bn_stats") is not True or checkpoint.get("unk") != "〓"
            or checkpoint.get("restored_from", {}).get("sha256") != FP32_SHA
            or tuple(checkpoint["model"]["head.weight"].shape) != (CLASSES + 1, ARCH["bottleneck"], 1, 1)):
        raise ValueError("initializer is not the fixed released FL10 architecture/restoration")
    if train2.H != ARCH["height"]:
        raise ValueError("training input height differs from released FL10 geometry")
    argv = ["--run", run["name"], "--out", str(out), "--init", str(init), "--vocab", str(vocab),
            "--source-policy", str(policy_path), "--keep-evals", "--freeze-bn-stats", "--kd-weight", "0"]
    for key in ("width", "bottleneck", "depth", "ctx", "ctx_k"):
        argv += ["--" + key.replace("_", "-"), str(ARCH[key])]
    for key, value in recipe.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    for row in doc["sources"]:
        paths = sources[row["name"]]
        if any(any(c in str(path) for c in ":+*?[]") for path in paths):
            raise ValueError("training shard path cannot contain trainer glob/spec metacharacters")
        opts = f"stream=0,label=gt,kd=0,kdonly=0,inv=0,aug={int(row['augmentation'])}"
        argv += ["--src", f"{row['name']}:{row['weight']}:{'+'.join(map(str, paths))}:{opts}"]
    if (sha256(config_path) != before or sha256(init) != INIT_SHA or sha256(vocab) != VOCAB_SHA
            or sha256(kanji_path) != KANJI_POLICY_SHA):
        raise ValueError("frozen training config/initializer/vocabulary changed during preparation")
    policy.verify_inputs()
    checked_development(development)
    textbox_guard.verify_prepared(textbox_development, line_lock=development)
    if sha256(development) != doc["development"]["sha256"]:
        raise ValueError("development preparation changed during training preparation")
    if sha256(textbox_development) != doc["textbox_development"]["sha256"]:
        raise ValueError("textbox preparation changed during training preparation")
    return {"schema": 1, "kind": "general-vn-fl10-training-prepared", "config": str(config_path),
            "config_sha256": before, "development": str(development),
            "development_sha256": doc["development"]["sha256"],
            "textbox_development_sha256": doc["textbox_development"]["sha256"],
            "source_policy_sha256": policy.sha256, "init_sha256": INIT_SHA, "vocab_sha256": VOCAB_SHA,
            "kanji_policy_sha256": KANJI_POLICY_SHA,
            "architecture": ARCH, "argv": argv, "run_directory": str(out / run["name"]),
            "checkpoint_steps": list(range(recipe["eval_every"], recipe["steps"] + 1, recipe["eval_every"])),
            "inputs": {name: {"file": str(path), "sha256": checksum} for name, path, checksum in (
                ("development", development, doc["development"]["sha256"]),
                ("textbox_development", textbox_development, doc["textbox_development"]["sha256"]),
                ("source_policy", policy_path, policy.sha256),
                ("init", init, INIT_SHA), ("vocab", vocab, VOCAB_SHA), ("kanji_policy", kanji_path, KANJI_POLICY_SHA))},
            "code_sha256": {name: sha256(ROOT / name) for name in CODE},
            "execution": train2.execution_identity(vocab, values), "selection": "separate frozen general-VN selector",
            "quality_measured": False}


def atomic_freeze(path, value):
    """Publish a complete JSON inode atomically without replacing any existing artifact."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic no-overwrite publication
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(config, out):
    if Path(out).exists():
        raise FileExistsError("prepared training configuration is immutable; use a new destination")
    identity = checked_config(config)
    atomic_freeze(out, identity)
    return identity


def verify_prepared(path):
    path = Path(path).resolve()
    checksum = sha256(path)
    frozen = parse_json(path.read_bytes())
    if not isinstance(frozen, dict) or frozen.get("kind") != "general-vn-fl10-training-prepared":
        raise ValueError("training requires a prepared generic FL10 identity")
    if (sha256(frozen["config"]) != frozen["config_sha256"] or frozen.get("architecture") != ARCH
            or set(frozen.get("code_sha256", {})) != set(CODE)
            or any(sha256(ROOT / name) != value for name, value in frozen["code_sha256"].items())
            or set(frozen.get("inputs", {})) != {
                "development", "textbox_development", "source_policy", "init", "vocab", "kanji_policy"}):
        raise ValueError("prepared training fingerprint changed before any crop/checkpoint load")
    for pin in frozen["inputs"].values():
        pinned_path(pin, path.parent)
    vocab = Path(frozen["inputs"]["vocab"]["file"])
    if frozen["execution"] != train2.execution_identity(vocab, vocab.read_text(encoding="utf-8").splitlines()):
        raise ValueError("prepared execution/geometry fingerprint changed before any crop/checkpoint load")
    current = checked_config(frozen["config"])
    if current != frozen or sha256(path) != checksum:
        raise ValueError("prepared training fingerprint changed")
    return frozen, checksum


def context(path, identity, checksum):
    return {"schema": 1, "driver": "generic_fl10_train", "prepared": str(Path(path).resolve()),
            "prepared_sha256": checksum, "wrapper_sha256": identity["code_sha256"]["train/generic_fl10_train.py"],
            "config_sha256": identity["config_sha256"], "development_sha256": identity["development_sha256"],
            "textbox_development_sha256": identity["textbox_development_sha256"],
            "architecture": identity["architecture"], "identity_sha256": json_digest(identity)}


def worker(path):
    identity, checksum = verify_prepared(path)
    train2.main(identity["argv"], execution_context=context(path, identity, checksum))


def retain_inventory(path, identity, checksum):
    directory = Path(identity["run_directory"])
    inventory_path = directory / "retained-checkpoints.json"
    expected = [directory / f"ckpt_step{step}.pt" for step in identity["checkpoint_steps"]]
    if set(directory.glob("ckpt_step*.pt")) != set(expected):
        raise ValueError("retained checkpoint inventory differs from the complete evenly spaced recipe")
    inventory = {"schema": 1, "kind": "general-vn-training-checkpoints", "prepared_sha256": checksum,
                 "selection": "separate frozen general-VN selector; no checkpoint scored here",
                 "checkpoints": [{"step": step, "file": str(file), "sha256": sha256(file)}
                                 for step, file in zip(identity["checkpoint_steps"], expected, strict=True)]}
    if verify_prepared(path) != (identity, checksum):
        raise ValueError("training preparation changed during execution")
    if inventory_path.exists():
        if parse_json(inventory_path.read_bytes()) != inventory:
            raise ValueError("completed retained checkpoint inventory changed")
    else:
        atomic_freeze(inventory_path, inventory)
    return inventory


def run(path):
    identity, checksum = verify_prepared(path)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--prepared", str(Path(path).resolve())],
                   check=True)
    return retain_inventory(path, identity, checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument("--out", type=Path, required=True)
    for name in ("run", "worker"):
        execution = commands.add_parser(name)
        execution.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        identity = prepare(args.config, args.out)
        print(json.dumps({"prepared": str(args.out.resolve()), "checkpoint_steps": identity["checkpoint_steps"],
                          "training_performed": False, "quality_measured": False}))
    elif args.command == "worker":
        worker(args.prepared)
    else:
        result = run(args.prepared)
        print(json.dumps({"retained_checkpoints": len(result["checkpoints"]), "quality_measured": False}))


if __name__ == "__main__":
    main()

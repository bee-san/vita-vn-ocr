#!/usr/bin/env python3
"""Freeze two matched native-bitmap trials without accessing test captures.

Replay OFL general dialogue, bounded similar-kanji corrections and a small real
training source. Native bitmap crops already contain size, noise and crop
variation, so their training augmentation is disabled to preserve labelled ink.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from hiragana_experiment import FL10_ARCH, sha256, shard_files
from hiragana_fresh import check_kanji_manifest, host_environment, read_shard
from hiragana_native import native_inputs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from kanji_policy import allowed_label, load_policy  # noqa: E402

MIX = {"general": .66, "bitmap_general": .20, "bitmap_targeted": .10, "native": .02, "kanji": .02}
INIT_SHA = "9fbcb3b2c620a582a003353abb9481d0ee05eb8a5709d4bb1481d4e733597b47"
DEV_KEYS = ("dev", "synthetic_dev", "bitmap_dev", "psp_dev", "vita_dev")


def bind_snapshot(path, expected, files):
    """Bind exactly the declared shards to a previously audited synthetic snapshot."""
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    if doc.get("schema") != 2:
        raise ValueError("bitmap trials require schema-2 synthetic snapshots")
    seen = set()
    for row in doc["shards"]:
        source = expected.get(row["source"])
        if source is None:
            continue
        name = str((path.parent / row["file"]).resolve())
        if (name not in files or name in seen or files[name]["source"] != source
                or files[name]["role"] != row["split"] or row["sha256"] != sha256(name)
                or row["bytes"] != Path(name).stat().st_size or row["samples"] != files[name]["crops"]):
            raise ValueError("synthetic shards differ from their declared snapshot")
        seen.add(name)
    required = {name for name, row in files.items() if row["source"] in expected.values()}
    if seen != required:
        raise ValueError("synthetic snapshot does not cover the exact declared inputs")
    return doc


def check_inputs(config, vocab):
    policy = load_policy(config["kanji_policy"])
    if policy["vocab_sha256"] != sha256(vocab):
        raise ValueError("bitmap trials require the exact policy vocabulary")
    allowed = set(Path(vocab).read_text().splitlines())
    files, shards = {}, {}
    for source in ("general", "bitmap_general", "bitmap_targeted", "kanji", *DEV_KEYS):
        role = "dev" if source in DEV_KEYS else "train"
        for filename in shard_files(config[source]):
            path = str(Path(filename).resolve())
            if path in files:
                raise ValueError("the same shard belongs to multiple sources")
            shard = read_shard(path, role)
            if any(not set(label) <= allowed or not allowed_label(label, policy) for label in shard["labels"]):
                raise ValueError("labels differ from the vocabulary or Jiten kanji policy")
            if source in ("dev", "psp_dev", "vita_dev") and any(
                    row.get("hand_checked") is not True or not row.get("session") for row in shard["meta"]):
                raise ValueError("real development requires checked labels and explicit sessions")
            shards[path] = shard
            files[path] = {"role": role, "source": source, "crops": len(shard["labels"])}
    train = [shards[name] for name, row in files.items() if row["role"] == "train"]
    for name, row in files.items():
        if row["role"] == "dev" and any(shards[name][key] & other[key]
                                        for other in train for key in ("images", "sessions", "pixels")):
            raise ValueError("development images, sessions or pixels overlap training")
    old = bind_snapshot(config["general_manifest"], {"general": "general", "synthetic-dev": "synthetic_dev"}, files)
    bitmap = bind_snapshot(config["bitmap_manifest"], {"general": "bitmap_general", "targeted": "bitmap_targeted",
                                                     "synthetic-dev": "bitmap_dev"}, files)
    if not bitmap.get("native_fonts"):
        raise ValueError("native bitmap snapshot must bind its source fonts")
    for snapshot in (old, bitmap):
        if snapshot["kanji_policy_sha256"] != sha256(config["kanji_policy"]):
            raise ValueError("synthetic snapshot kanji policy changed")
    check_kanji_manifest(config, files, shards, policy)
    files.update(native_inputs(config, files))
    for key in ("kanji_policy", "kanji_manifest", "general_manifest", "bitmap_manifest",
                "real_guard_manifest_psp", "real_guard_manifest_vita", "dev_manifest"):
        files[str(Path(config[key]).resolve())] = {"role": "provenance", "source": key}
    return files


def commands(config, init, vocab, out, python=sys.executable):
    result = []
    for scope, learning_rate in (("head", "0.0001"), ("all", "0.00002")):
        command = [python, str(ROOT / "train/train2.py"), "--run", f"bitmap_{scope}_s1", "--out", str(out),
                   "--init", str(init), "--vocab", str(vocab), "--width", "1.0", "--bottleneck", "96",
                   "--depth", "1", "--ctx", "2", "--ctx-k", "5", "--steps", "2000", "--batch", "32",
                   "--lr", learning_rate, "--warmup", "50", "--threads", "6", "--seed", "1",
                   "--wd", "0.01", "--final-lr", "0.05", "--freeze-bn-stats", "--train-scope", scope,
                   "--kd-weight", "0", "--eval-every", "200", "--keep-evals", "--log-every", "100",
                   "--val", f"clannad:{config['dev']}", "--val", f"synthetic:{config['synthetic_dev']}",
                   "--val", f"bitmap:{config['bitmap_dev']}", "--select", "clannad", "--select-metric", "hiragana"]
        for source, weight in MIX.items():
            aug = ",aug=0" if source in ("bitmap_general", "bitmap_targeted", "native") else ""
            command += ["--src", f"{source}:{weight}:{config[source]}:inv=0{aug}"]
        result.append(command)
    return result


def prepare(config_path, init, vocab, out):
    config = json.loads(Path(config_path).read_text())
    files = check_inputs(config, vocab)
    checkpoint = torch.load(init, map_location="cpu", weights_only=False)
    if (sha256(init) != INIT_SHA or checkpoint["vocab"] != Path(vocab).read_text().splitlines()
            or any(checkpoint.get(key, value) != value for key, value in FL10_ARCH.items())):
        raise ValueError("bitmap trials require the pinned released FL10 initialization")
    out = Path(out).resolve()
    code = ("train/hiragana_bitmap.py", "train/hiragana_fresh.py", "train/hiragana_native.py",
            "train/hiragana_experiment.py", "train/train2.py", "train/train.py", "train/model.py",
            "train/export.py", "train/hiragana_select.py", "synth/common.py", "synth/kanji_policy.py",
            "eval/sets.py", "eval/metrics.py", "eval/backends.py", "runtime/pyvocr.py")
    identity = {"schema": 1, "recipe": "private-native-bitmap-v1", "config": config,
                "config_sha256": sha256(config_path), "init_sha256": sha256(init),
                "vocab_sha256": sha256(vocab), "architecture": FL10_ARCH,
                "inputs": {path: {**row, "sha256": sha256(path), "bytes": Path(path).stat().st_size}
                           for path, row in sorted(files.items())},
                "commands": commands(config, Path(init).resolve(), Path(vocab).resolve(), out),
                "code_sha256": {path: sha256(ROOT / path) for path in code},
                "dependencies": {path: sha256(ROOT / path) for path in ("requirements.lock", "requirements-ci.lock")},
                "environment": host_environment(), "sampling": MIX,
                "augmentation": "inv=0 throughout; bitmap and real native sources aug=0; OFL general/kanji aug=1",
                "selection": "unchanged strict CLANNAD hiragana improvement and original full/synthetic guards; "
                             "separate PSP, Vita and bitmap dev full/hiragana non-regression guards",
                "test_data_accessed": False,
                "reproducibility": "Complete restart pinned; resume omits RNG/source/mixer state."}
    record = out / "experiment.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError("experiment directory belongs to a different frozen recipe")
    out.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "init", "vocab", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    identity = prepare(args.config, args.init, args.vocab, args.out)
    print("Frozen matched head/all native bitmap trials.", flush=True)
    if args.run:
        for command in identity["commands"]:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

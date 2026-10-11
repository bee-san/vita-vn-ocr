#!/usr/bin/env python3
"""Frozen-encoder native-font recovery after the fresh synthetic arms fail dev guards.

Use identical inputs/seed for two head-only learning rates, with native console polarity.
Only development data can select a deployment blob; test sets remain unopened.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from hiragana_experiment import FL10_ARCH, sha256, shard_files
from hiragana_fresh import check_data, host_environment, read_shard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from kanji_policy import allowed_label, load_policy  # noqa: E402


def native_inputs(config, base_files):
    manifest = Path(config["native_manifest"]).resolve()
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    policy = load_policy(config["kanji_policy"])
    if (doc.get("schema") != 1 or doc.get("role") != "train"
            or doc.get("kanji_policy_sha256") != sha256(config["kanji_policy"])
            or doc.get("background_manifest_sha256") != sha256(config["native_sources"])
            or doc.get("vocab_sha256") != policy["vocab_sha256"]):
        raise ValueError("native manifest identity or training role changed")
    sys.path.insert(0, str(ROOT))
    from data.fresh_backgrounds import validate_manifest
    source_doc = validate_manifest(json.loads(Path(config["native_sources"]).read_text()))
    approved = {row["vn"]: row for row in source_doc["records"]
                if row["split"] == "train" and row["scenery_box"]}
    source_images = {row["vn"]: row for row in doc["source_images"]}
    if not source_images or any(row != approved.get(vn) for vn, row in source_images.items()):
        raise ValueError("native labels use unapproved source images")
    paths = {str(Path(path).resolve()) for path in shard_files(config["native"])}
    pinned = {str((manifest.parent / row["file"]).resolve()): row for row in doc["shards"]}
    if paths != set(pinned) or paths & set(base_files):
        raise ValueError("native shards differ from their unique snapshot manifest")
    other = [read_shard(path, details["role"]) for path, details in base_files.items()
             if details["role"] in ("train", "dev")]
    for path in paths:
        record, shard = pinned[path], read_shard(path, "train")
        if (record["sha256"] != sha256(path) or record["bytes"] != Path(path).stat().st_size
                or record["samples"] != len(shard["labels"])):
            raise ValueError("native shard identity changed")
        for label, meta in zip(shard["labels"], shard["meta"]):
            source = source_images.get(meta.get("vn"))
            if (not source or not allowed_label(label, policy) or meta.get("hand_checked") is not True
                    or meta.get("source_kind") != "native" or meta.get("image_sha256") != source["sha256"]
                    or meta.get("url") != source["url"] or meta.get("release") != source["release"]
                    or meta.get("annotations_sha256") != doc["annotations_sha256"]
                    or meta.get("kanji_policy_sha256") != doc["kanji_policy_sha256"]):
                raise ValueError("native crop provenance or label differs from the approved snapshot")
        if any(shard[key] & previous[key] for previous in other for key in ("images", "sessions", "pixels")):
            raise ValueError("native source overlaps another training/development input")
        other.append(shard)
    return {**{path: {"role": "train", "source": "native", "crops": pinned[path]["samples"]} for path in paths},
            str(manifest): {"role": "provenance", "source": "native_manifest"},
            str(Path(config["native_sources"]).resolve()): {"role": "provenance", "source": "native_sources"}}


def prepare(config_path, init, vocab, out):
    config = json.loads(Path(config_path).read_text())
    values = Path(vocab).read_text().splitlines()
    files = check_data(config, values, sha256(vocab))
    files.update(native_inputs(config, files))
    ck = torch.load(init, map_location="cpu", weights_only=False)
    if ck["vocab"] != values or any(ck.get(k, v) != v for k, v in FL10_ARCH.items()):
        raise ValueError("native trials require exact released FL10 initialization")
    out = Path(out).resolve()
    commands = []
    for name, learning_rate in (("head_slow", "0.00002"), ("head", "0.0001")):
        command = [sys.executable, str(ROOT / "train/train2.py"), "--run", f"native_{name}_s1",
                   "--out", str(out), "--init", str(Path(init).resolve()), "--vocab", str(Path(vocab).resolve()),
                   "--width", "1.0", "--bottleneck", "96", "--depth", "1", "--ctx", "2", "--ctx-k", "5",
                   "--steps", "1000", "--batch", "32", "--lr", learning_rate, "--warmup", "50",
                   "--threads", "6", "--seed", "1", "--wd", "0.01", "--final-lr", "0.05",
                   "--freeze-bn-stats", "--train-scope", "head", "--kd-weight", "0",
                   "--eval-every", "100", "--keep-evals", "--log-every", "100",
                   "--val", f"clannad:{config['dev']}", "--val", f"synthetic:{config['synthetic_dev']}",
                   "--select", "clannad", "--select-metric", "hiragana"]
        for source, weight in (("general", .83), ("targeted", .10), ("native", .05), ("kanji", .02)):
            command += ["--src", f"{source}:{weight}:{config[source]}:inv=0"]
        commands.append(command)
    code = ["train/hiragana_native.py", "train/hiragana_fresh.py", "train/hiragana_experiment.py",
            "train/train2.py", "train/train.py", "train/model.py", "train/restore_blob.py", "train/export.py",
            "data/import_native.py", "data/fresh_backgrounds.py", "synth/kanji_policy.py", "synth/common.py",
            "eval/metrics.py"]
    identity = {"recipe": "native-font-recovery-v1", "config": config, "config_sha256": sha256(config_path),
                "init_sha256": sha256(init), "vocab_sha256": sha256(vocab), "architecture": FL10_ARCH,
                "inputs": {path: {**details, "sha256": sha256(path), "bytes": Path(path).stat().st_size}
                           for path, details in sorted(files.items())},
                "commands": commands, "code_sha256": {path: sha256(ROOT / path) for path in code},
                "dependencies": {path: sha256(ROOT / path) for path in ("requirements.lock", "requirements-ci.lock")},
                "environment": host_environment(),
                "selection": "C W8I real-development hiragana improvement with full-CER and synthetic guards",
                "targeted_kana": "reference substitutions from development only: ず and っ",
                "sampling": {"general": .83, "targeted": .10, "native": .05, "kanji": .02,
                             "inversion_augmentation": 0, "polarity": "stored native console rule"},
                "reproducibility": "Complete restart is pinned; checkpoint resume omits RNG/source/mixer states."}
    record = out / "experiment.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError("native experiment directory belongs to another frozen identity")
    out.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(identity, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "init", "vocab", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    identity = prepare(args.config, args.init, args.vocab, args.out)
    print("Prepared two head-only native-font trials.", flush=True)
    if args.run:
        for command in identity["commands"]:
            print(json.dumps(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

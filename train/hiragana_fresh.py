#!/usr/bin/env python3
"""Prepare or run fresh, matched FL10 fine-tuning without historical training shards.

Training uses new general/targeted synthetic crops, curated AnimeText and a bounded
Jiten-approved similar-kanji corrective source (2% of sampled lines in every arm).
Every arm starts at the same restored release checkpoint. CLANNAD development and
separate synthetic development crops are validation only. Exported C W8I development
selection with full-CER and synthetic guards must precede held-out evaluation.

Pinned inputs, code, dependencies, commands and seeds permit a complete restart.
train2's checkpoint resume is approximate: RNG/source/mixer states are not saved.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import hostinfo
from hiragana_experiment import FL10_ARCH, sha256, shard_files
from model import from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from kanji_policy import allowed_label, load_policy  # noqa: E402

MIXES = {"general": (.98, 0.0, 0.0, .02), "targeted": (.88, .1, 0.0, .02), "anime": (.83, .1, .05, .02)}
SOURCES = ("general", "targeted", "anime", "kanji")


def read_shard(path, role):
    """Validate hard labels and return identities without keeping image arrays in RAM."""
    with np.load(path) as z:
        labels = json.loads(str(z["labels"]))
        if not labels or any(not isinstance(s, str) or not s.strip() for s in labels):
            raise ValueError(f"{path}: hard-label snapshot is empty or has unlabelled crops")
        shapes, offsets, data = z["shapes"], z["offsets"], z["data"]
        meta = json.loads(str(z["meta"])) if "meta" in z else [{} for _ in labels]
        if (shapes.shape != (len(labels), 2) or len(meta) != len(labels)
                or shapes.dtype.kind not in "iu" or offsets.dtype.kind not in "iu"
                or data.dtype != np.uint8 or data.ndim != 1
                or offsets.shape != (len(labels) + 1,) or offsets[0] != 0
                or offsets[-1] != len(data) or np.any(shapes <= 0)
                or np.any(np.diff(offsets) != shapes[:, 0] * shapes[:, 1])):
            raise ValueError(f"{path}: crop arrays and labels disagree")
        if any(not isinstance(m, dict) or m.get("split") != role for m in meta):
            raise ValueError(f"{path}: evaluation crops in training or invalid {role} split")
        sessions = {m["session"] for m in meta if m.get("session")}
        images = {m[k] for m in meta for k in ("image_sha256", "sha256") if m.get(k)}
        pixels = set()
        for i, shape in enumerate(shapes):
            crop = data[offsets[i]:offsets[i + 1]]
            pixels.add(hashlib.sha256(np.asarray(shape, "<i4").tobytes() + crop.tobytes()).hexdigest())
            # The curated manifest pins the original, pre-console-inversion pixels.
            if role == "train" and meta[i].get("pixel_sha256"):
                original = 255 - crop if meta[i].get("inverted") else crop
                digest = hashlib.sha256(np.asarray(shape, "<i4").tobytes() + original.tobytes()).hexdigest()
                if digest != meta[i]["pixel_sha256"]:
                    raise ValueError(f"{path}: pixels differ from the training manifest")
    return {"labels": labels, "meta": meta, "sessions": sessions, "images": images, "pixels": pixels}


def check_kanji_manifest(config, files, shards, policy):
    path = Path(config["kanji_manifest"]).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 2 or not isinstance(manifest.get("shards"), list):
        raise ValueError("kanji source requires a schema-2 snapshot manifest with shard hashes")
    policy_sha = sha256(config["kanji_policy"])
    approved = set()
    for row in manifest["shards"]:
        if row.get("source") != "kanji":
            continue
        shard_path = str((path.parent / row["file"]).resolve())
        if (row.get("split") != "train" or shard_path in approved or shard_path not in files
                or files[shard_path]["source"] != "kanji"):
            raise ValueError("kanji manifest contains duplicate, unapproved or evaluation shards")
        if (row.get("sha256") != sha256(shard_path) or row.get("bytes") != Path(shard_path).stat().st_size
                or row.get("samples") != files[shard_path]["crops"]):
            raise ValueError("kanji manifest hash, size or crop count differs from its shard")
        for text, meta in zip(shards[shard_path]["labels"], shards[shard_path]["meta"]):
            if meta.get("kanji_policy_sha256") != policy_sha or meta.get("kanji_pairs") is not True:
                raise ValueError("corrective kanji crops require the exact policy identity and pair-sampling flag")
            if not any(c in policy["ranks"] for c in text):
                raise ValueError("corrective kanji crop has no supported kanji")
            pair, target = meta.get("kanji_pair"), meta.get("kanji_target")
            if (not isinstance(pair, str) or len(pair) != 2 or "".join(sorted(pair)) not in policy["pairs"]
                    or not isinstance(target, str) or target not in pair or len(target) != 1 or target not in text):
                raise ValueError("corrective kanji crop lacks its supported pair endpoint")
        approved.add(shard_path)
    expected = {p for p, info in files.items() if info["source"] == "kanji"}
    if approved != expected:
        raise ValueError("kanji shards differ from the policy-bound snapshot manifest")


def check_data(config, vocab=None, vocab_sha256=None):
    required = {*SOURCES, "anime_manifest", "anime_audit", "dev", "synthetic_dev",
                "kanji_policy", "kanji_manifest"}
    if not required <= config.keys():
        raise ValueError(f"missing fresh experiment keys: {sorted(required - config.keys())}")
    policy = load_policy(config["kanji_policy"])
    if vocab_sha256 is not None and policy["vocab_sha256"] != vocab_sha256:
        raise ValueError("kanji policy vocabulary identity differs from the exact FL10 vocabulary")
    files, shards = {}, {}
    train_sessions, train_images, train_pixels = set(), set(), set()
    for source in SOURCES:
        for name in shard_files(config[source]):
            path = str(Path(name).resolve())
            if path in shards:
                raise ValueError(f"{path}: the same shard belongs to two training sources")
            shard = read_shard(path, "train")
            shards[path] = shard
            files[path] = {"role": "train", "source": source, "crops": len(shard["labels"])}
            train_sessions.update(shard["sessions"])
            train_images.update(shard["images"])
            train_pixels.update(shard["pixels"])
    dev_sessions, dev_images, dev_pixels = set(), set(), set()
    for key in ("dev", "synthetic_dev"):
        path = str(Path(config[key]).resolve())
        if path in files:
            raise ValueError(f"{path}: development shard is also training or another development source")
        if not Path(path).is_file():
            raise ValueError(f"missing development source: {path}")
        shard = read_shard(path, "dev")
        if key == "dev" and any(m.get("split") != "dev" or m.get("hand_checked") is not True
                                or not m.get("session") for m in shard["meta"]):
            raise ValueError("CLANNAD development must be hand-checked with explicit dev sessions")
        if (shard["sessions"] & train_sessions or shard["images"] & train_images
                or shard["pixels"] & train_pixels):
            raise ValueError(f"{path}: development session, screenshot or pixels overlap training")
        if (shard["sessions"] & dev_sessions or shard["images"] & dev_images
                or shard["pixels"] & dev_pixels):
            raise ValueError(f"{path}: development sources overlap each other")
        dev_sessions.update(shard["sessions"])
        dev_images.update(shard["images"])
        dev_pixels.update(shard["pixels"])
        files[path] = {"role": "dev", "source": key, "crops": len(shard["labels"])}
        shards[path] = shard
    if vocab is not None:
        allowed = set(vocab)
        if any(c not in allowed for s in shards.values() for text in s["labels"] for c in text):
            raise ValueError("labels must contain only characters from the exact FL10 vocabulary")
    labels = "".join(text for shard in shards.values() for text in shard["labels"])
    if not allowed_label(labels, policy):
        raise ValueError("training/development labels contain characters outside the Jiten kanji policy")
    check_kanji_manifest(config, files, shards, policy)
    audit = json.loads(Path(config["anime_audit"]).read_text())
    checked = audit.get("checked", 0)
    if (not isinstance(checked, int) or isinstance(checked, bool) or checked < 200
            or not audit.get("approved_training_manifest_sha256")):
        raise ValueError("AnimeText requires a completed 200-crop audit and curated training manifest")
    if audit["approved_training_manifest_sha256"] != sha256(config["anime_manifest"]):
        raise ValueError("audit does not cover the curated AnimeText manifest")
    manifest = Path(config["anime_manifest"]).resolve()
    records = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    approved = set()
    for record in records:
        path = str((manifest.parent / record["shard"]).resolve())
        index = record["index"]
        if (path not in shards or files[path]["source"] != "anime" or record["split"] != "train"
                or not isinstance(index, int) or not 0 <= index < len(shards[path]["labels"])):
            raise ValueError("AnimeText manifest contains an unapproved shard, index or split")
        if (path, index) in approved:
            raise ValueError("duplicate crop in the curated AnimeText manifest")
        if (record["text"] != shards[path]["labels"][index]
                or {k: v for k, v in record.items() if k not in {"shard", "index", "text"}}
                != shards[path]["meta"][index]):
            raise ValueError("AnimeText crop labels or metadata differ from the audited manifest")
        approved.add((path, index))
    expected = {(p, i) for p, info in files.items() if info["source"] == "anime"
                for i in range(info["crops"])}
    if approved != expected:
        raise ValueError("AnimeText shards differ from the audited training manifest")
    for key in ("anime_audit", "anime_manifest", "kanji_policy", "kanji_manifest"):
        files[str(Path(config[key]).resolve())] = {"role": "provenance", "source": key}
    return files


def commands(config, ckpt, vocab, out, python=sys.executable):
    result = []
    for name, weights in MIXES.items():
        cmd = [python, str(ROOT / "train/train2.py"), "--run", f"fresh_{name}_s1", "--out", str(out),
               "--init", str(ckpt), "--vocab", str(vocab), "--width", "1.0", "--bottleneck", "96",
               "--depth", "1", "--ctx", "2", "--ctx-k", "5", "--steps", "1000", "--batch", "32",
               "--lr", "0.00002", "--warmup", "50", "--seed", "1", "--threads", "6",
               "--wd", "0.01", "--final-lr", "0.05", "--freeze-bn-stats", "--kd-weight", "0",
               "--eval-every", "100", "--keep-evals", "--log-every", "100",
               "--val", f"clannad:{config['dev']}", "--val", f"synthetic:{config['synthetic_dev']}",
               "--select", "clannad", "--select-metric", "hiragana"]
        for source, weight in zip(SOURCES, weights):
            if weight:
                cmd += ["--src", f"{source}:{weight}:{config[source]}:inv=0.2"]
        result.append(cmd)
    return result


def host_environment():
    cpu = platform.processor()
    info = Path("/proc/cpuinfo")
    if info.exists():
        cpu = next((s.split(":", 1)[1].strip() for s in info.read_text().splitlines()
                    if s.startswith("model name")), cpu)
    return {"host": hostinfo.host_label(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu, "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "cuda_available": torch.cuda.is_available(), "training_threads": 6}


def prepare(config_path, init, vocab, out, python=sys.executable):
    config = json.loads(Path(config_path).read_text())
    vocab_values = Path(vocab).read_text().splitlines()
    files = check_data(config, vocab_values, sha256(vocab))
    ck = torch.load(init, map_location="cpu", weights_only=False)
    if (ck["vocab"] != vocab_values
            or any(ck.get(key, value) != value for key, value in FL10_ARCH.items())):
        raise ValueError("fresh experiment must use exact FL10 architecture and checkpoint vocabulary order")
    _, meta = from_checkpoint(ck)
    if (meta["vocab"] != vocab_values
            or any(meta[key] != value for key, value in FL10_ARCH.items())):
        raise ValueError("fresh experiment must use exact FL10 architecture and checkpoint vocabulary order")
    code_paths = ["train/hiragana_fresh.py", "train/hiragana_experiment.py", "train/train2.py",
                  "train/train.py", "train/model.py", "train/restore_blob.py", "train/export.py",
                  "synth/render.py", "synth/common.py", "synth/bg_exclude.py", "eval/metrics.py",
                  "eval/run.py", "eval/sets.py", "eval/backends.py", "data/import_animetext.py",
                  "synth/kanji_policy.py"]
    identity = {"recipe": "fresh-fl10-hiragana-kanji-v2", "config": config,
                "config_sha256": sha256(config_path), "init_sha256": sha256(init),
                "vocab_sha256": sha256(vocab), "architecture": FL10_ARCH,
                "inputs": {p: {**details, "sha256": sha256(p), "bytes": Path(p).stat().st_size}
                           for p, details in sorted(files.items())},
                "commands": commands(config, Path(init).resolve(), Path(vocab).resolve(), Path(out).resolve(), python),
                "code_sha256": {p: sha256(ROOT / p) for p in code_paths},
                "dependency_sha256": {p: sha256(ROOT / p) for p in ("requirements.lock", "requirements-ci.lock")},
                "environment": host_environment(),
                "sampling": {"unit": "source probability per sampled line; independent of shard count",
                             "source_order": list(SOURCES), "arm_weights": {name: list(w) for name, w in MIXES.items()},
                             "kanji_corrective_probability": .02,
                             "kanji_policy_sha256": sha256(config["kanji_policy"]),
                             "within_kanji_source": "finite rendered schedule follows policy rank bands/pair weights"},
                "selection": {"train2": "preliminary development hiragana only; all checkpoints retained",
                              "final": "export C W8I; development hiragana/full CER with full-CER and synthetic guards",
                              "held_out": "evaluate only after the selected blob identity is locked"},
                "reproducibility": {"complete_restart": "pinned inputs, code, dependency locks, commands and seeds",
                                    "checkpoint_resume": "approximate; train2 omits RNG and source/mixer states"}}
    out = Path(out)
    record = out / "experiment.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError("experiment directory belongs to another configuration or input/code/environment identity")
    out.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "init", "vocab", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--run", action="store_true",
                    help="execute sequential arms; default validates and records commands")
    args = ap.parse_args()
    identity = prepare(args.config, args.init, args.vocab, args.out)
    print(f"Prepared {len(identity['commands'])} fresh arms; resume is approximate, complete restart is recorded.")
    if args.run:
        for command in identity["commands"]:
            print(json.dumps(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

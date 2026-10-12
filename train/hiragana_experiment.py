#!/usr/bin/env python3
"""Run the matched FL10 hard-label experiment after the data and audit are ready.

The config identifies the five original hard-label sources (nv/s2/tt/rd/ps),
targeted synthetic training shards, audited AnimeText training shards, and
session-separated CLANNAD development NPZ. No test set is used for selection.
"""
import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from model import from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = {"nv": .25, "s2": .25, "tt": .275, "rd": .125, "ps": .1}
MIXES = {"control": (1, 0, 0), "targeted": (.9, .1, 0), "anime": (.85, .1, .05)}
FL10_ARCH = {"width": 1.0, "bottleneck": 96, "depth": 1, "ctx": 2, "ctx_k": 5, "height": 32}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shard_files(pattern):
    paths = sorted({p for part in pattern.split("+") for p in glob.glob(part)})
    if not paths:
        raise ValueError(f"missing training source: {pattern}")
    return paths


def check_data(config):
    if set(config["original"]) != set(ORIGINAL):
        raise ValueError("original mixture requires nv, s2, tt, rd and ps hard-label snapshots")
    train_files, train_sessions, train_images = set(), set(), set()
    for pattern in [*config["original"].values(), config["targeted"], config["anime"]]:
        for p in shard_files(pattern):
            train_files.add(str(Path(p).resolve()))
            with np.load(p) as z:
                labels = json.loads(str(z["labels"]))
                if not labels or not all(labels):
                    raise ValueError(f"{p}: hard-label snapshot is empty or has unlabelled crops")
                if "meta" in z:
                    meta = json.loads(str(z["meta"]))
                    train_sessions.update(m["session"] for m in meta if m.get("session"))
                    train_images.update(m[k] for m in meta for k in ("image_sha256", "sha256") if m.get(k))
                    if any(m.get("split") in {"test", "val", "dev", "holdout"} for m in meta):
                        raise ValueError(f"{p}: evaluation crops in a training source")
    dev = Path(config["dev"]).resolve()
    if str(dev) in train_files:
        raise ValueError("development shard is also a training source")
    with np.load(dev) as z:
        meta = json.loads(str(z["meta"]))
        if not meta or any(m.get("split") != "dev" or m.get("hand_checked") is not True or not m.get("session")
                           for m in meta):
            raise ValueError("development data must be hand-checked captures with explicit dev sessions")
        if any(m["session"] in train_sessions or m.get("image_sha256") in train_images for m in meta):
            raise ValueError("development session or screenshot overlaps training")
    audit = json.loads(Path(config["anime_audit"]).read_text())
    if audit.get("checked", 0) < 200 or not audit.get("approved_training_manifest_sha256"):
        raise ValueError("AnimeText requires a completed 200-crop audit and a curated training manifest")
    if audit["approved_training_manifest_sha256"] != sha256(config["anime_manifest"]):
        raise ValueError("audit does not cover the curated AnimeText manifest")
    records = [json.loads(x) for x in Path(config["anime_manifest"]).read_text().splitlines()]
    approved = {str((Path(config["anime_manifest"]).parent / r["shard"]).resolve()) for r in records}
    if set(map(lambda p: str(Path(p).resolve()), shard_files(config["anime"]))) != approved:
        raise ValueError("AnimeText shards differ from audited training manifest")
    if any(r["split"] != "train" for r in records):
        raise ValueError("AnimeText holdout data cannot enter training")


def commands(config, ckpt, vocab, out, python=sys.executable):
    result = []
    for seed in (1, 2, 3):
        for name, (original, targeted, anime) in MIXES.items():
            cmd = [python, str(ROOT / "train/train2.py"), "--run", f"{name}_s{seed}", "--out", str(out),
                   "--init", str(ckpt), "--vocab", str(vocab), "--width", "1.0", "--steps", "5000",
                   "--lr", "0.0001", "--warmup", "100", "--batch", "96", "--seed", str(seed),
                   "--freeze-bn-stats", "--kd-weight", "0", "--eval-every", "500", "--keep-evals",
                   "--val", f"clannad:{config['dev']}", "--select", "clannad", "--select-metric", "hiragana"]
            for src, weight in ORIGINAL.items():
                inv = .2 if src in {"nv", "s2"} else .5
                cmd += ["--src", f"{src}:{original * weight}:{config['original'][src]}:inv={inv}"]
            if targeted:
                cmd += ["--src", f"targeted:{targeted}:{config['targeted']}:inv=0.2"]
            if anime:
                cmd += ["--src", f"anime:{anime}:{config['anime']}:inv=0.2"]
            result.append(cmd)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", action="store_true", help="execute; default validates and writes commands only")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    check_data(cfg)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    _, meta = from_checkpoint(ck)
    if meta["vocab"] != args.vocab.read_text().splitlines() or any(meta[k] != v for k, v in FL10_ARCH.items()):
        raise ValueError("experiment must use the exact restored FL10 vocabulary and architecture")
    args.out.mkdir(parents=True, exist_ok=True)
    cmds = commands(cfg, args.init.resolve(), args.vocab.resolve(), args.out.resolve())
    files = {p for pattern in [*cfg["original"].values(), cfg["targeted"], cfg["anime"]]
             for p in shard_files(pattern)} | {cfg["dev"], cfg["anime_audit"], cfg["anime_manifest"]}
    identity = {"config": cfg, "config_sha256": sha256(args.config), "init_sha256": sha256(args.init),
                "vocab_sha256": sha256(args.vocab), "inputs": {p: sha256(p) for p in sorted(files)}, "commands": cmds}
    record = args.out / "experiment.json"
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError("experiment directory belongs to another configuration")
    record.write_text(json.dumps(identity, indent=2) + "\n")
    if args.run:
        for cmd in cmds:
            print(json.dumps(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

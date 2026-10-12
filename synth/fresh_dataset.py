#!/usr/bin/env python3
"""Rebuild compact VN training data from pinned assets, with book-held-out dev pools.

The assets, pools and render steps keep all downloaded/generated data under --root.
The render step forwards synth/render.py options; for example:

  fresh_dataset.py assets --root build/fresh
  fresh_dataset.py pools --root build/fresh --vocab /private/fl10-vocab.txt
  fresh_dataset.py render --root build/fresh --split train --finder grad \\
      --invert-dark --vocab /private/fl10-vocab.txt --out build/fresh/general \\
      --seed 20261011 --shards 24 --per-shard 500 --workers 2
  fresh_dataset.py render --root build/fresh --split dev --finder grad \\
      --invert-dark --vocab /private/fl10-vocab.txt --out build/fresh/synthetic-dev \\
      --seed 20262011 --shards 1 --per-shard 500 --workers 1
  fresh_dataset.py manifest --root build/fresh

Use --target-kana or --target-report on a separate training render for targeted
natural text. The development books never enter either training text pool.
"""
import argparse
import collections
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import math
import platform
import random
import sys
import time
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PIN = HERE / "fresh_assets.json"
POOL_WEIGHTS = {"dialogue": .63, "narration": .37}
BOOK_POLICY = "sha256(seed:book_id) uint64 modulo10: dev0,train1..9"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def book_split(book_id, seed):
    key = hashlib.sha256(f"{seed}:{book_id}".encode()).digest()
    return "dev" if int.from_bytes(key[:8], "big") % 10 == 0 else "train"


def download(url, dest, expected_sha256, expected_bytes=None):
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "vita-vn-ocr-fresh-data/0.1"})
                with urllib.request.urlopen(request, timeout=120) as r, tmp.open("wb") as f:
                    while block := r.read(1 << 20):
                        f.write(block)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        if sha256(tmp) != expected_sha256:
            raise ValueError(f"sha256 mismatch for {url}")
        tmp.replace(dest)
    if sha256(dest) != expected_sha256 or expected_bytes is not None and dest.stat().st_size != expected_bytes:
        raise ValueError(f"size/hash mismatch for {dest}")


def fetch_assets(root, pin=PIN, workers=4):
    root = Path(root)
    spec = json.loads(Path(pin).read_text(encoding="utf-8"))

    def fetch(row):
        dest = root / row["file"]
        if "zip_url" in row:
            archive = root / "assets/downloads" / row["zip_url"].rsplit("/", 1)[1]
            download(row["zip_url"], archive, row["zip_sha256"])
            with zipfile.ZipFile(archive) as z:
                if not dest.exists():
                    member = next(n for n in z.namelist() if n.endswith(dest.name))
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(member))
                for name in z.namelist():
                    if not name.rsplit("/", 1)[-1].upper().startswith(("OFL", "LICENSE")):
                        continue
                    license_path = dest.parent / name
                    if dest.parent.resolve() not in license_path.resolve().parents:
                        raise ValueError(f"unsafe archive license path: {name}")
                    license_path.parent.mkdir(parents=True, exist_ok=True)
                    license_path.write_bytes(z.read(name))
            if sha256(dest) != row["sha256"] or dest.stat().st_size != row["bytes"]:
                raise ValueError(f"size/hash mismatch for {dest}")
        else:
            download(row["url"], dest, row["sha256"], row["bytes"])
        print(f"asset verified: {row['file']}", flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(fetch, spec["assets"]))
    write_json(root / "assets_manifest.json", {**spec, "assets": records, "pin_sha256": sha256(pin)})


def verify_backgrounds(root, manifest, fetch=False):
    """Verify native source images and keep all VN/release identities out of dev."""
    from PIL import Image

    from bg_exclude import load_exclusions
    from common import is_test_vn

    sys.path.insert(0, str(REPO))
    from data.fresh_backgrounds import validate_manifest

    root = Path(root).resolve()
    doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if doc.get("schema") != 2:
        raise ValueError("fresh rendering requires a reviewed schema2 background manifest")
    if any(row.get("scenery_review", {}).get("hand_checked") is not True for row in doc.get("records", [])):
        raise ValueError("every background record needs a hand-checked scenery review")
    validate_manifest(doc)
    rows = doc["records"]
    reserved = load_exclusions() | {"v4"}
    identities = {role: {"vn": set(), "release": set()} for role in ("train", "dev")}
    counts = collections.Counter()
    for row in rows:
        role, platform_name = row["split"], row["platform"]
        if role not in identities or platform_name not in ("psp", "psv"):
            raise ValueError("background role/platform must be train|dev and psp|psv")
        folder = "assets" if role == "train" else "assets-dev"
        platform_dir = "vndb_psp" if platform_name == "psp" else "vndb"
        expected_dir = root / folder / platform_dir / "shots"
        dest = (root / row["file"]).resolve()
        if dest.parent != expected_dir or not dest.name.startswith(row["vn"] + "_"):
            raise ValueError("background path differs from its role/platform/VN")
        if row["vn"] in reserved or is_test_vn(row["vn"]):
            raise ValueError(f"reserved evaluation background: {row['vn']}")
        if fetch:
            download(row["url"], dest, row["sha256"], row["bytes"])
        if sha256(dest) != row["sha256"] or dest.stat().st_size != row["bytes"]:
            raise ValueError(f"background size/hash mismatch: {dest}")
        native = [480, 272] if platform_name == "psp" else [960, 544]
        with Image.open(dest) as image:
            if list(image.size) != row["dims"] or row["dims"] != native:
                raise ValueError(f"background has non-native dimensions: {dest}")
        for kind in ("vn", "release"):
            identities[role][kind].add(row[kind])
        counts[f"{role}:{platform_name}"] += 1
    if any(identities["train"][kind] & identities["dev"][kind] for kind in ("vn", "release")):
        raise ValueError("background training and development games/releases overlap")
    return {"manifest_sha256": sha256(manifest), "source_manifest": doc,
            "counts": dict(counts), "train_dev_vn_overlap": 0, "train_dev_release_overlap": 0}


def build_pools(root, vocab_path, seed=20261011, caps=None):
    import build_corpus

    root = Path(root)
    caps = caps or {"train": {"dialogue": 50000, "narration": 75000},
                    "dev": {"dialogue": 10000, "narration": 15000}}
    vocab = set(Path(vocab_path).read_text(encoding="utf-8").splitlines())
    random.seed(seed)  # the existing reservoir uses the standard-library RNG
    pools = {role: {name: build_corpus.Reservoir(n) for name, n in limits.items()}
             for role, limits in caps.items()}
    book_ids = {role: [] for role in pools}
    books = modern = rejected = 0
    start = time.time()
    corpus = root / "assets/corpus/aozora/aozorabunko-dedupe-clean.jsonl.gz"
    with gzip.open(corpus, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            books += 1
            meta = obj.get("meta") or {}
            if meta.get("文字遣い種別") != "新字新仮名":
                continue
            modern += 1
            text = obj.get("text") or ""
            book_id = str(meta.get("作品ID") or hashlib.sha256(text.encode()).hexdigest())
            role = book_split(book_id, seed)
            book_ids[role].append(book_id)
            for paragraph in text.split("\n"):
                for chunk in build_corpus.chunks(build_corpus.norm(paragraph)):
                    if not build_corpus.keep(chunk):
                        continue
                    if not set(chunk) <= vocab:
                        rejected += 1
                        continue
                    pools[role]["dialogue" if chunk[0] in "「『" else "narration"].add(chunk)
    if set(book_ids["train"]) & set(book_ids["dev"]):
        raise ValueError("training and development books overlap")
    records = []
    for role, role_pools in pools.items():
        out = root / ("assets" if role == "train" else "assets-dev") / "corpus"
        cohort = out / "book_ids.json"
        write_json(cohort, {"split": role, "seed": seed, "policy": BOOK_POLICY,
                            "book_ids": sorted(set(book_ids[role]))})
        records.append({"split": role, "role": "source_book_cohort", "file": str(cohort.relative_to(root)),
                        "sha256": sha256(cohort), "books": len(set(book_ids[role]))})
        frequency = collections.Counter()
        for name, pool in role_pools.items():
            if not pool.items:
                raise ValueError(f"empty {role} {name} pool")
            path = out / f"{name}.txt"
            path.write_text("\n".join(pool.items) + "\n", encoding="utf-8")
            frequency.update("".join(pool.items))
            records.append({"split": role, "pool": name, "items": len(pool.items), "seen": pool.seen,
                            "file": str(path.relative_to(root)), "sha256": sha256(path)})
        path = out / "char_freq.json"
        # Match the compact JSON format used for the original replacement pools.
        path.write_text(json.dumps(dict(frequency.most_common()), ensure_ascii=False) + "\n", encoding="utf-8")
        records.append({"split": role, "file": str(path.relative_to(root)), "sha256": sha256(path)})
    result = {"seed": seed, "books_seen": books, "modern_kana_books": modern, "oov_chunks_excluded": rejected,
              "book_split_policy": BOOK_POLICY, "book_intersection": 0, "pool_weights": POOL_WEIGHTS,
              "pool_records": records, "vocab_sha256": sha256(vocab_path), "elapsed_seconds": time.time() - start,
              "build_corpus_source_sha256": sha256(HERE / "build_corpus.py")}
    write_json(root / "pools_manifest.json", result)
    return result


def render_shards(root, split, arguments, background_manifest=None):
    import render

    root = Path(root).resolve()
    render.DATA = root / ("assets" if split == "train" else "assets-dev")
    render.FONT_DIRS = [root / "assets/fonts", root / "assets/fonts_extra"]
    render.POOL_WEIGHT = POOL_WEIGHTS.copy()
    cohort_sha = sha256(render.DATA / "corpus/book_ids.json")
    backgrounds = verify_backgrounds(root, background_manifest) if background_manifest else None
    if not backgrounds and any(list((render.DATA / folder / "shots").glob("*.jpg"))
                               for folder in ("vndb", "vndb_psp")):
        raise ValueError("screenshot backgrounds require a reviewed --background-manifest")
    policy_path = None
    for i, argument in enumerate(arguments):
        if argument == "--kanji-policy":
            policy_path = Path(arguments[i + 1])
        elif argument.startswith("--kanji-policy="):
            policy_path = Path(argument.split("=", 1)[1])
    policy_sha = sha256(policy_path) if policy_path else None
    original_meta = render.crop_meta

    def bound_meta(*args, **kwargs):
        meta = original_meta(*args, **kwargs)
        meta.update(split=split, source_kind="synthetic", book_split_sha256=cohort_sha)
        if policy_sha:
            meta.update(kanji_policy_sha256=policy_sha, kanji_pairs="--kanji-pairs" in arguments)
        meta["crop_edge_jitter_requested"] = "--crop-edge-jitter" in arguments
        if backgrounds:
            meta["background_manifest_sha256"] = backgrounds["manifest_sha256"]
        return meta

    render.crop_meta = bound_meta
    event = {"split": split, "argv": [str(HERE / "render.py"), "shards", *arguments],
             "wrapper_argv": sys.argv.copy(),
             "book_split_sha256": cohort_sha, "kanji_policy_sha256": policy_sha,
             "background_manifest_sha256": backgrounds["manifest_sha256"] if backgrounds else None,
             "renderer_source_sha256": sha256(HERE / "render.py"), "helper_sha256": sha256(__file__),
             "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    root.mkdir(parents=True, exist_ok=True)
    with (root / "render_invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    original_argv = sys.argv
    sys.argv = [str(HERE / "render.py"), "shards", *arguments]
    try:
        render.main()
    finally:
        sys.argv = original_argv
        render.crop_meta = original_meta


def dataset_manifest(root, sources, vocab_path=None, policy_path=None, background_manifest=None):
    import cv2
    import fontTools
    import numpy as np
    import PIL
    from PIL import features

    root = Path(root).resolve()
    backgrounds = verify_backgrounds(root, background_manifest) if background_manifest else None
    if not backgrounds and any(list((root / folder / platform_dir / "shots").glob("*.jpg"))
                               for folder in ("assets", "assets-dev") for platform_dir in ("vndb", "vndb_psp")):
        raise ValueError("screenshot backgrounds require --background-manifest")
    assets = json.loads((root / "assets_manifest.json").read_text(encoding="utf-8"))
    pools = json.loads((root / "pools_manifest.json").read_text(encoding="utf-8"))
    for row in pools["pool_records"]:
        if sha256(root / row["file"]) != row["sha256"]:
            raise ValueError(f"pool/cohort hash mismatch: {row['file']}")
    cohort_files = {row["split"]: root / row["file"] for row in pools["pool_records"]
                    if row.get("role") == "source_book_cohort"}
    book_ids = {split: set(json.loads(path.read_text(encoding="utf-8"))["book_ids"])
                for split, path in cohort_files.items()}
    if book_ids.get("train", set()) & book_ids.get("dev", set()):
        raise ValueError("training and development books overlap")
    vocab = None
    if vocab_path:
        if sha256(vocab_path) != pools["vocab_sha256"]:
            raise ValueError("vocabulary differs from the text-pool vocabulary")
        vocab = set(Path(vocab_path).read_text(encoding="utf-8").splitlines())
    cohort_hashes = {row["split"]: row["sha256"] for row in pools["pool_records"]
                     if row.get("role") == "source_book_cohort"}
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8")) if policy_path else None
    policy_sha = sha256(policy_path) if policy_path else None
    records, summaries, frequencies, pixels = [], {}, {}, {}
    for name, split in sources.items():
        counts, chars, hashes, policy_hashes = collections.Counter(), collections.Counter(), set(), set()
        paths = sorted((root / name).glob("synth_?????.npz"))
        if not paths:
            raise ValueError(f"no shards for {name}")
        for path in paths:
            shard_counts, shard_chars = collections.Counter(), collections.Counter()
            with np.load(path) as z:
                data, shapes, offsets = z["data"], z["shapes"], z["offsets"]
                labels, meta = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
            if len(labels) != len(meta) or len(labels) != len(shapes) or len(offsets) != len(labels) + 1:
                raise ValueError(f"inconsistent shard arrays: {path}")
            for i, (label, row) in enumerate(zip(labels, meta)):
                if row.get("split") != split or row.get("book_split_sha256") != cohort_hashes[split]:
                    raise ValueError(f"unbound role/book cohort in {path}")
                if backgrounds and row.get("background_manifest_sha256") != backgrounds["manifest_sha256"]:
                    raise ValueError(f"unbound background sources in {path}")
                if row.get("source_kind") != "synthetic":
                    raise ValueError("source_kind must be synthetic")
                if vocab is not None and not set(label) <= vocab:
                    raise ValueError(f"out-of-vocabulary label in {path}")
                if row.get("hand_checked"):
                    raise ValueError("synthetic crops must not claim hand-checked screenshot status")
                counts["samples"] += 1
                counts["characters"] += len(label)
                for key in ("platform", "fam", "cat", "kind", "layout", "scheme"):
                    if key in row:
                        counts[f"{key}:{row[key]}"] += 1
                chars.update(label)
                shard_chars.update(label)
                if row.get("kanji_policy_sha256"):
                    policy_hashes.add(row["kanji_policy_sha256"])
                if policy_sha and row.get("kanji_policy_sha256") != policy_sha:
                    raise ValueError(f"kanji policy differs from crop metadata in {path}")
                if "kanji_band" in row:
                    shard_counts[f"pair_band:{row['kanji_band']}"] += 1
                    counts[f"pair_band:{row['kanji_band']}"] += 1
                    target = row.get("kanji_target")
                    if not target or target not in label:
                        raise ValueError(f"selected kanji target missing from label in {path}")
                    shard_counts[f"target:{target}"] += 1
                for key in ("crop_edge_jitter_requested", "crop_edge_jitter_enabled",
                            "crop_edge_jitter_applied", "crop_edge_jitter_changed"):
                    applied = bool(row.get(key))
                    counts[key] += applied
                    shard_counts[key] += applied
                counts["kanji_pairs"] += bool(row.get("kanji_pairs"))
                h, w = shapes[i]
                width = max(8, min(1024, round(w * 32 / h)))
                needed = len(label) + sum(a == b for a, b in zip(label, label[1:]))
                counts["ctc_infeasible"] += math.ceil(width / 8) < needed
                hashes.add(hashlib.sha256(shapes[i].tobytes() + data[offsets[i]:offsets[i + 1]].tobytes()).hexdigest())
            records.append({"source": name, "split": split, "file": str(path.relative_to(root)),
                            "sha256": sha256(path), "bytes": path.stat().st_size, "samples": len(labels),
                            "kanji_sampling": dict(shard_counts),
                            "kanji_character_exposure": kanji_exposure(shard_chars, policy) if policy else None})
        if counts["ctc_infeasible"]:
            raise ValueError(f"CTC-infeasible labels in {name}")
        counts["unique_pixels"] = len(hashes)
        summaries[name] = {**dict(counts), "kanji_policy_sha256": sorted(policy_hashes),
                           "kanji_character_exposure": kanji_exposure(chars, policy) if policy else None}
        frequencies[name] = dict(chars.most_common())
        pixels[name] = hashes
    train_pixels = set().union(*(pixels[n] for n, s in sources.items() if s == "train"))
    dev_pixels = set().union(*(pixels[n] for n, s in sources.items() if s == "dev"))
    if train_pixels & dev_pixels:
        raise ValueError("training and development pixels overlap")
    result = {"schema": 2, "assets": assets, "backgrounds": backgrounds, "pools": pools, "sources": sources,
              "shards": records,
              "summary": summaries, "train_dev_pixel_intersection": 0,
              "kanji_policy_sha256": policy_sha,
              "render_invocations": [json.loads(line) for line in
                                     (root / "render_invocations.jsonl").read_text().splitlines()]
              if (root / "render_invocations.jsonl").exists() else [],
              "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                                (Path(__file__), HERE / "build_corpus.py", HERE / "render.py", HERE / "common.py",
                                 HERE / "mask_finder.py", HERE / "bg_exclude.py", PIN, REPO / "requirements-ci.lock",
                                 HERE / "kanji_policy.py", REPO / "data/fresh_backgrounds.py") if p.exists()},
              "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                              "numpy": np.__version__, "pillow": PIL.__version__,
                              "freetype": features.version_module("freetype2"),
                              "opencv": cv2.__version__, "fonttools": fontTools.__version__}}
    write_json(root / "label_character_frequency.json", frequencies)
    write_json(root / "dataset_manifest.json", result)
    return result


def kanji_exposure(chars, policy):
    """Actual glyph counts by Jiten rank, independent of source mixing weights."""
    bands = {str(i): 0 for i in range(4)}
    outside = 0
    for char, count in chars.items():
        if not unicodedata.name(char, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
            continue
        rank = policy["ranks"].get(char)
        if rank is None:
            outside += count
            continue
        band = 0 if rank <= 2000 else 1 if rank <= 3000 else 2 if rank <= 4000 else 3
        bands[str(band)] += count
    if outside:
        raise ValueError(f"{outside} kanji glyphs outside the policy allowlist")
    return {"glyphs": sum(bands.values()), "bands": bands, "outside_allowlist": outside}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("assets", "pools", "render", "manifest"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kanji-policy", type=Path)
    parser.add_argument("--background-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--source", action="append", help="manifest source directory name:train|dev")
    args, remaining = parser.parse_known_args()
    if args.step != "render" and remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    if args.step == "assets":
        fetch_assets(args.root, workers=args.workers)
        if args.background_manifest:
            verify_backgrounds(args.root, args.background_manifest, fetch=True)
    elif args.step == "pools":
        if not args.vocab:
            parser.error("pools requires --vocab")
        build_pools(args.root, args.vocab, args.seed)
    elif args.step == "render":
        if not args.vocab:
            parser.error("render requires --vocab")
        extra = ["--kanji-policy", str(args.kanji_policy)] if args.kanji_policy else []
        if args.background_manifest:
            extra.extend(["--background-manifest", str(args.background_manifest)])
        render_shards(args.root, args.split, ["--vocab", str(args.vocab), "--seed", str(args.seed),
                                            "--workers", str(args.workers), *extra, *remaining],
                      args.background_manifest)
    else:
        sources = dict(s.split(":", 1) for s in args.source) if args.source else {
            "general": "train", "targeted": "train", "synthetic-dev": "dev"}
        if any(split not in ("train", "dev") for split in sources.values()):
            parser.error("manifest sources must have train or dev roles")
        dataset_manifest(args.root, sources, args.vocab, args.kanji_policy, args.background_manifest)


if __name__ == "__main__":
    main()

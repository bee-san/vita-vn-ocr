#!/usr/bin/env python3
"""Stream pinned AnimeText OCR Parquet shards into local hard-label training NPZs.

No YOLO or detector is added to the console. Confidence and layout checks are
filters, not proof of correct labels. Audit accepted crops before using them.
"""
import argparse
import hashlib
import io
import json
import math
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "eval"))
from metrics import align_text, is_hiragana  # noqa: E402

from common import find_lines_ex, line_polarity, load_vocab, to_gray  # noqa: E402

PIN = ROOT / "data/manifests/animetext_ocr.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_shard(path):
    # NpzFile is lazy: repeatedly indexing z['data'] inflates the entire ZIP
    # member for each crop. Load its arrays once, then release the archive.
    with np.load(path) as z:
        return {k: z[k] for k in ("data", "offsets", "shapes")}


def group_split(image_id, seed=1, holdout_percent=10):
    key = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return "holdout" if int.from_bytes(key[:8], "big") % 100 < holdout_percent else "train"


def filter_row(row, vocab, policy):
    """Return (native grey image, label, None) or (None, None, rejection reason)."""
    if row.get("language") != "ja":
        return None, None, "language"
    for key in ("seq_confidence", "min_token_confidence"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not policy[key] <= value <= 1:
            return None, None, key
    text = row.get("transcription") or ""
    if not text or any(c.isspace() for c in text):
        return None, None, "empty_or_whitespace"
    if any(c not in vocab for c in text):
        return None, None, "vocabulary"
    if not row.get("image_id") or not row.get("crop_id"):
        return None, None, "missing_id"
    try:
        with Image.open(io.BytesIO(row["image"]["bytes"])) as im:
            if im.width * im.height > 1_000_000:
                return None, None, "dimensions"
            g = to_gray(np.asarray(im.convert("RGB")))
    except (OSError, ValueError, KeyError, TypeError, Image.DecompressionBombError):
        return None, None, "image"
    h, w = g.shape
    if h < 8 or h > 256 or w < 4:
        return None, None, "dimensions"
    # Horizontal text, including square single glyphs. Conservative rejection
    # deliberately loses difficult crops; the audit catches remaining layout errors.
    if (len(text) == 1 and not 0.45 <= w / h <= 2.5) or (len(text) > 1 and w / h < max(1.2, .35 * len(text))):
        return None, None, "vertical_or_multiline"
    found = find_lines_ex(g, "grad")
    if found and (found[2] or len(found[0]) > 1):
        return None, None, "multiple_bands"
    resized_w = max(8, round(w * 32 / h))
    required = len(text) + sum(a == b for a, b in zip(text, text[1:]))
    if resized_w > 1024 or (resized_w + 7) // 8 < required:
        return None, None, "ctc_length"
    return g, text, None


class ShardWriter:
    def __init__(self, out, per_shard=2000):
        self.out, self.per_shard = Path(out), per_shard
        self.pending = {"train": [], "holdout": []}
        self.counts = Counter()
        self.manifest = (self.out / "crops.jsonl").open("w", encoding="utf-8")

    def add(self, g, text, meta):
        split = meta["split"]
        self.pending[split].append((g, text, meta))
        if len(self.pending[split]) >= self.per_shard:
            self.flush(split)

    def flush(self, split):
        rows = self.pending[split]
        if not rows:
            return
        rel = f"{split}/animetext_{self.counts[split]:05d}.npz"
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shapes = np.array([g.shape for g, _, _ in rows], np.int32)
        offsets = np.concatenate([[0], np.cumsum(shapes.prod(axis=1))]).astype(np.int64)
        np.savez_compressed(path, data=np.concatenate([g.ravel() for g, _, _ in rows]),
                            offsets=offsets, shapes=shapes,
                            labels=np.array(json.dumps([t for _, t, _ in rows], ensure_ascii=False)),
                            meta=np.array(json.dumps([m for _, _, m in rows], ensure_ascii=False)))
        for i, (_, text, meta) in enumerate(rows):
            self.manifest.write(json.dumps({**meta, "text": text, "shard": rel, "index": i},
                                           ensure_ascii=False) + "\n")
        self.manifest.flush()
        self.counts[split] += 1
        self.pending[split] = []

    def close(self):
        for split in self.pending:
            self.flush(split)
        self.manifest.close()


def download(entry, policy, cache):
    url = f"https://huggingface.co/datasets/{policy['repo']}/resolve/{policy['revision']}/{entry['path']}"
    path = cache / Path(entry["path"]).name
    expected = entry["lfs"]["oid"]
    if path.exists() and path.stat().st_size == entry["size"] and digest(path) == expected:
        return path, url, expected
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size != entry["size"] or digest(tmp) != expected:
        raise ValueError(f"download sha256/size mismatch: {entry['path']}")
    tmp.replace(path)
    return path, url, expected


def run_import(args):
    import pyarrow.parquet as pq
    policy = json.loads(PIN.read_text())
    vocab = set(load_vocab(args.vocab))
    if args.out.exists():
        raise ValueError("output already exists; use a fresh directory to avoid mixing imports")
    args.out.mkdir(parents=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/datasets/{policy['repo']}/tree/{policy['revision']}/data?limit=1000"
    with urllib.request.urlopen(url, timeout=60) as r:
        entries = sorted((e for e in json.load(r) if e["path"].endswith(".parquet")), key=lambda e: e["path"])
    writer = ShardWriter(args.out)
    seen, seen_ids, counts, sources = set(), set(), Counter(), []
    limit = args.max_crops or policy["max_crops"]
    try:
        for entry in entries[:args.max_shards]:
            path, source_url, sha = download(entry, policy, args.cache)
            sources.append({"url": source_url, "sha256": sha, "bytes": path.stat().st_size})
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    counts["read"] += 1
                    g, text, reason = filter_row(row, vocab, policy)
                    if reason:
                        counts[reason] += 1
                        continue
                    pixel_sha = hashlib.sha256(np.asarray(g.shape, "<i4").tobytes() + g.tobytes()).hexdigest()
                    if pixel_sha in seen or row["crop_id"] in seen_ids:
                        counts["duplicate"] += 1
                        continue
                    seen.add(pixel_sha)
                    seen_ids.add(row["crop_id"])
                    split = group_split(row["image_id"], policy["split_seed"], policy["holdout_percent"])
                    dark = bool(line_polarity(g))
                    meta = {"source": policy["repo"], "revision": policy["revision"],
                            "license": policy["license"], "source_url": source_url, "source_sha256": sha,
                            "image_id": str(row["image_id"]), "crop_id": str(row["crop_id"]),
                            "encoded_sha256": hashlib.sha256(row["image"]["bytes"]).hexdigest(),
                            "pixel_sha256": pixel_sha, "split": split, "inverted": dark,
                            **{k: row[k] for k in ("seq_confidence", "min_token_confidence")}}
                    writer.add(255 - g if dark else g, text, meta)
                    counts["accepted"] += 1
                    counts[f"accepted_{split}"] += 1
                    if counts["accepted"] >= limit:
                        break
                if counts["accepted"] >= limit:
                    break
            print(f"{entry['path']}: {dict(counts)}", flush=True)
            if counts["accepted"] >= limit:
                break
    finally:
        writer.close()
        (args.out / "import.json").write_text(json.dumps({"policy": policy, "sources": sources,
            "vocab_sha256": digest(args.vocab), "counts": dict(counts), "audit_required": True}, indent=2) + "\n")


def audit(args):
    rows = [json.loads(x) for x in (args.input / "crops.jsonl").read_text().splitlines()]
    rows = sorted(rows, key=lambda r: hashlib.sha256(f"{args.seed}:{r['crop_id']}".encode()).digest())[:args.n]
    args.out.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(str(args.font), 16)
    checks, cache = [], {}
    for start in range(0, len(rows), 20):
        sheet = Image.new("RGB", (1200, 20 * 60), (35, 35, 35))
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(rows[start:start + 20]):
            path = args.input / r["shard"]
            if path not in cache:
                cache[path] = read_shard(path)
            z, i = cache[path], r["index"]
            h, w = map(int, z["shapes"][i])
            g = z["data"][z["offsets"][i]:z["offsets"][i + 1]].reshape(h, w)
            im = Image.fromarray(g)
            scale = min(48 / h, 680 / w)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
            sheet.paste(im.convert("RGB"), (0, j * 60))
            draw.text((690, j * 60 + 8), f"{start + j}: {r['text']}", font=font, fill="white")
            checks.append({"crop_id": r["crop_id"], "pixel_sha256": r["pixel_sha256"],
                           "text": r["text"], "correct": None, "corrected_text": None,
                           "layout_ok": None})
        sheet.save(args.out / f"audit_{start // 20:02d}.png")
    (args.out / "audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1) + "\n")


def curate(args):
    """Conservatively exclude kana classes with observed label errors; train split only."""
    rows = [json.loads(x) for x in (args.input / "crops.jsonl").read_text().splitlines()]
    by_id = {r["crop_id"]: r for r in rows}
    checks = json.loads(args.audit.read_text())
    if len(checks) < 200 or len({r["crop_id"] for r in checks}) != len(checks):
        raise ValueError("curation requires at least 200 distinct audited crops")
    reject_kana, reject_ids, corrections = set(), set(), {}
    confusion, occurrences = Counter(), Counter()
    errors = layout_errors = uncertain = 0
    for check in checks:
        original = by_id[check["crop_id"]]
        if check["pixel_sha256"] != original["pixel_sha256"] or check["text"] != original["text"]:
            raise ValueError("audit crop identity differs from import")
        undecided = check.get("uncertain") is True and check["correct"] is None
        if (type(check["correct"]) is not bool and not undecided) or type(check["layout_ok"]) is not bool:
            raise ValueError("audit has undecided label/layout checks")
        if undecided:
            uncertain += 1
            reject_ids.add(check["crop_id"])
            reject_kana.update(c for c in check.get("uncertain_kana", []) if is_hiragana(c))
        occurrences.update(c for c in check["text"] if is_hiragana(c))
        if not check["layout_ok"]:
            layout_errors += 1
            reject_ids.add(check["crop_id"])
        if check["correct"] is False:
            errors += 1
            corrected = check.get("corrected_text")
            if corrected:
                for g, p in align_text(corrected, check["text"]):
                    if g != p and (is_hiragana(g) or is_hiragana(p)):
                        confusion[g, p] += 1
                        reject_kana.update(c for c in (g, p) if is_hiragana(c))
                corrections[check["crop_id"]] = corrected
            else:
                reject_kana.update(c for c in check["text"] if is_hiragana(c))
                reject_ids.add(check["crop_id"])
    if args.out.exists():
        raise ValueError("curated output already exists")
    args.out.mkdir(parents=True)
    writer = ShardWriter(args.out)
    counts, cache = Counter(), {}
    vocab = set(load_vocab(args.vocab))
    try:
        for r in rows:
            if r["split"] != "train":
                counts["holdout"] += 1
                continue
            corrected = corrections.get(r["crop_id"])
            if r["crop_id"] in reject_ids or (not corrected and any(c in reject_kana for c in r["text"])):
                counts["audit_rejected"] += 1
                continue
            text = corrected or r["text"]
            path = args.input / r["shard"]
            if path not in cache:
                cache = {path: read_shard(path)}
            z, i = cache[path], r["index"]
            h, w = map(int, z["shapes"][i])
            g = z["data"][z["offsets"][i]:z["offsets"][i + 1]].reshape(h, w)
            frames = (max(8, round(w * 32 / h)) + 7) // 8
            if any(c not in vocab for c in text) or frames < len(text) + sum(a == b for a, b in zip(text, text[1:])):
                counts["corrected_unusable"] += 1
                continue
            meta = {k: v for k, v in r.items() if k not in {"shard", "index", "text"}}
            meta["audit_corrected"] = bool(corrected)
            writer.add(g.copy(), text, meta)
            counts["accepted"] += 1
    finally:
        writer.close()
    summary = {"checked": len(checks), "label_errors": errors, "layout_errors": layout_errors,
               "uncertain_labels": uncertain, "decidable_labels": len(checks) - uncertain,
               "label_error_rate": errors / max(1, len(checks) - uncertain), "rejected_kana": sorted(reject_kana),
               "observed_kana": dict(occurrences),
               "kana_confusions": [{"reference": g, "label": p, "count": n}
                                   for (g, p), n in confusion.most_common()],
               "counts": dict(counts), "audit_sha256": digest(args.audit),
               "import_manifest_sha256": digest(args.input / "crops.jsonl"),
               "approved_training_manifest_sha256": digest(args.out / "crops.jsonl")}
    (args.out / "audit-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--vocab", type=Path, required=True)
    imp.add_argument("--out", type=Path, required=True)
    imp.add_argument("--cache", type=Path, default=ROOT / "build/animetext-cache")
    imp.add_argument("--max-crops", type=int, default=20000)
    imp.add_argument("--max-shards", type=int, default=339)
    aud = sub.add_parser("audit")
    aud.add_argument("--input", type=Path, required=True)
    aud.add_argument("--out", type=Path, required=True)
    aud.add_argument("--font", type=Path, required=True)
    aud.add_argument("--n", type=int, default=200)
    aud.add_argument("--seed", type=int, default=1)
    cur = sub.add_parser("curate")
    cur.add_argument("--input", type=Path, required=True)
    cur.add_argument("--audit", type=Path, required=True)
    cur.add_argument("--vocab", type=Path, required=True)
    cur.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "import":
        if args.max_crops < 1 or args.max_shards < 1:
            ap.error("limits must be positive")
        run_import(args)
    elif args.command == "audit":
        audit(args)
    else:
        curate(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pack privately transcribed native training crops from the pinned VNDB cohort.

Annotations stay local. Only source identities, hashes and aggregate counts belong
in Git. Every accepted line must be explicitly checked visually before packing.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
from common import line_polarity, to_gray  # noqa: E402
from data.fresh_backgrounds import sha256, validate_manifest  # noqa: E402
from kanji_policy import allowed_label, load_policy  # noqa: E402
from render import crop_ctc_feasible  # noqa: E402


def pixel_sha(g):
    return hashlib.sha256(np.asarray(g.shape, "<i4").tobytes() + g.tobytes()).hexdigest()


def pack(root, sources, annotations, vocab, policy_path, out):
    root, out = Path(root).resolve(), Path(out)
    if out.exists():
        raise ValueError("native snapshot output already exists")
    doc = validate_manifest(json.loads(Path(sources).read_text(encoding="utf-8")))
    if doc["schema"] != 2:
        raise ValueError("native import requires a visually reviewed source cohort")
    approved = {r["vn"]: r for r in doc["records"] if r["split"] == "train" and r["scenery_box"]}
    policy = load_policy(policy_path)
    if policy["vocab_sha256"] != sha256(vocab):
        raise ValueError("native import requires the exact policy vocabulary")
    rows = [json.loads(line) for line in Path(annotations).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("native annotations require unique, nonempty line identities")
    crops, labels, metas, counts, images, seen = [], [], [], Counter(), {}, {}
    for row in rows:
        if row.get("split") != "train" or row.get("hand_checked") is not True:
            raise ValueError("native annotations must be checked training lines")
        if row.get("vn") not in approved:
            raise ValueError("native image is absent from the approved training cohort")
        source = approved[row["vn"]]
        path = root / source["file"]
        if sha256(path) != source["sha256"] or path.stat().st_size != source["bytes"]:
            raise ValueError("native image differs from its pinned source")
        with Image.open(path) as image:
            if list(image.size) != source["dims"]:
                raise ValueError("native screenshot dimensions changed")
            rgb = np.asarray(image.convert("RGB"))
        box, pad, text = row.get("box"), row.get("pad", 2 if source["platform"] == "psp" else 4), row.get("text")
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or type(pad) is not int or not 0 <= pad <= 8):
            raise ValueError("native line geometry requires an integer box and bounded padding")
        x, y, w, h = box
        iw, ih = source["dims"]
        if not (0 <= x < x + w <= iw and 0 <= y < y + h <= ih):
            raise ValueError("native line box must fit entirely inside its screenshot")
        if not isinstance(text, str) or not text.strip() or any(c.isspace() for c in text):
            raise ValueError("native labels must be nonempty, unspaced strings")
        if not allowed_label(text, policy):
            counts["rejected_vocabulary_or_policy"] += 1
            continue
        g = to_gray(rgb[max(0, y-pad):min(ih, y+h+pad), max(0, x-pad):min(iw, x+w+pad)])
        if min(g.shape) < 8 or not crop_ctc_feasible(g.shape, text):
            counts["rejected_ctc_or_geometry"] += 1
            continue
        digest = pixel_sha(g)
        if digest in seen:
            if seen[digest] != text:
                raise ValueError("identical native pixels have conflicting labels")
            counts["rejected_duplicate_pixels"] += 1
            continue
        seen[digest] = text
        dark = bool(line_polarity(g))
        crops.append(255-g if dark else g)
        labels.append(text)
        metas.append({"id": row["id"], "split": "train", "source_kind": "native",
                      "hand_checked": True, "vn": source["vn"], "platform": source["platform"],
                      "session": f"vndb:{source['vn']}:{source['release']}",
                      "image_sha256": source["sha256"], "pixel_sha256": digest,
                      "url": source["url"], "release": source["release"],
                      "box": box, "pad": pad, "inverted": dark,
                      "kanji_policy_sha256": sha256(policy_path), "annotations_sha256": sha256(annotations)})
        images[source["vn"]] = source
        counts[f"kept:{source['platform']}"] += 1
    if not crops:
        raise ValueError("no native labels survived policy/CTC checks")
    shapes = np.asarray([g.shape for g in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes.prod(axis=1))]).astype(np.int64)
    out.mkdir(parents=True)
    shard = out / "native_00000.npz"
    np.savez_compressed(shard, data=np.concatenate([g.ravel() for g in crops]), shapes=shapes,
                        offsets=offsets, labels=np.asarray(json.dumps(labels, ensure_ascii=False)),
                        meta=np.asarray(json.dumps(metas, ensure_ascii=False)))
    result = {"schema": 1, "role": "train", "annotations_sha256": sha256(annotations),
              "background_manifest_sha256": sha256(sources), "kanji_policy_sha256": sha256(policy_path),
              "vocab_sha256": sha256(vocab), "importer_sha256": sha256(__file__),
              "review": "Crops/transcriptions inspected visually; ambiguous or mixed crops removed.",
              "annotated_lines": len(rows), "kept_lines": len(labels), "counts": dict(counts),
              "characters": len("".join(labels)), "label_character_counts": dict(Counter("".join(labels))),
              "source_images": list(images.values()),
              "shards": [{"file": shard.name, "samples": len(labels), "sha256": sha256(shard),
                          "bytes": shard.stat().st_size}]}
    (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "sources", "annotations", "vocab", "policy", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    result = pack(args.root, args.sources, args.annotations, args.vocab, args.policy, args.out)
    print(json.dumps({"kept_lines": result["kept_lines"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()

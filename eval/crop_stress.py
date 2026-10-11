#!/usr/bin/env python3
"""Development-only C W8I diagnostics for dialogue crop margins and ROI edges.

Every crop keeps its complete annotated ink box. Image-boundary padding is
clamped and reported separately. Optional signed ROI offsets expand (positive)
or shrink (negative) the dialogue selection while preserving every ink box.
This tool never opens shared test sets and never selects or trains a model.
"""
import argparse
import datetime
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "synth"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import finders  # noqa: E402
import metrics  # noqa: E402
from backends import NativeVocr  # noqa: E402
from sets import ImageCache  # noqa: E402

from common import line_polarity, to_gray  # noqa: E402

CASES = [{"name": f"symmetric-{margin}", "left": margin, "right": margin, "top": 4, "bottom": 4}
         for margin in (0, 2, 4, 8)] + [
    {"name": "left-0-right-8", "left": 0, "right": 8, "top": 4, "bottom": 4},
    {"name": "left-8-right-0", "left": 8, "right": 0, "top": 4, "bottom": 4}]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_bounds(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("boxes must contain four integer coordinates")
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("annotated ink or region box lies outside the image bounds")
    return x, y, x + w, y + h


def crop_bounds(box, margins, width, height):
    """Clamping may reduce requested padding, never the annotated ink extent."""
    x0, y0, x1, y1 = box_bounds(box, width, height)
    if any(type(margins[key]) is not int or margins[key] < 0 for key in ("left", "right", "top", "bottom")):
        raise ValueError("crop margins must be nonnegative integers")
    bounds = (max(0, x0 - margins["left"]), max(0, y0 - margins["top"]),
              min(width, x1 + margins["right"]), min(height, y1 + margins["bottom"]))
    actual = {"left": x0 - bounds[0], "right": bounds[2] - x1,
              "top": y0 - bounds[1], "bottom": bounds[3] - y1}
    assert bounds[0] <= x0 < x1 <= bounds[2] and bounds[1] <= y0 < y1 <= bounds[3]
    return bounds, actual


def load_development(manifest):
    path = Path(manifest).resolve()
    digest = sha256(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("split") != "dev" for row in rows):
        raise ValueError("crop stress manifest must be development-only")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate development line ids")
    if any(row.get("hand_checked") is not True or not row.get("session") for row in rows):
        raise ValueError("crop stress requires hand_checked=true and capture sessions")
    cache, images, image_hashes = ImageCache(), {}, {}
    for row in rows:
        image = path.parent / row["image"]
        if image not in images:
            image_hashes[image] = sha256(image)
            images[image] = cache.get(image)
        if image_hashes[image] != row["sha256"]:
            raise ValueError(f"development screenshot hash mismatch: {row['id']}")
        height, width = images[image].shape[:2]
        if "dims" in row and row["dims"] != [width, height]:
            raise ValueError("manifest frame dimensions do not match the screenshot")
        box_bounds(row["box"], width, height)
        if row.get("region") is not None:
            box_bounds(row["region"], width, height)
        if not row.get("uncertain") and not row.get("text"):
            raise ValueError("scored development lines need nonempty reference text")
        row["image_path"] = image
    if not any(not row.get("uncertain") for row in rows):
        raise ValueError("development manifest contains no scored lines")
    if sha256(path) != digest:
        raise ValueError("development manifest changed while loading")
    identity = {"split": "dev", "manifest_sha256": digest,
                "sessions": sorted({row["session"] for row in rows}),
                "scored_lines": sum(not row.get("uncertain", False) for row in rows),
                "uncertain_lines": sum(bool(row.get("uncertain")) for row in rows),
                "images": sorted([{"sha256": image_hashes[path], "width": image.shape[1],
                                   "height": image.shape[0]} for path, image in images.items()],
                                 key=lambda item: item["sha256"])}
    return rows, images, identity


def console_crops(crops):
    output, inverted = [], 0
    for crop in crops:
        gray = to_gray(crop)
        dark = bool(line_polarity(gray))
        output.append(np.ascontiguousarray(255 - gray if dark else gray))
        inverted += int(dark)
    return output, inverted


def aggregate(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("recognizer did not return one prediction per crop")
    scores = metrics.score_lines([{"text": row["text"], "pred": prediction}
                                  for row, prediction in zip(rows, predictions)])
    return {"full": scores["all"], "hiragana": scores["hiragana"]}


def line_case(rec, rows, images, case):
    scored = [row for row in rows if not row.get("uncertain")]
    crops, actuals, geometries = [], [], []
    for row in scored:
        image = images[row["image_path"]]
        bounds, actual = crop_bounds(row["box"], case, image.shape[1], image.shape[0])
        x0, y0, x1, y1 = bounds
        crops.append(image[y0:y1, x0:x1])
        actuals.append(actual)
        geometries.append([row["id"], row["sha256"], list(bounds)])
    crops, inverted = console_crops(crops)
    started = time.monotonic()
    predictions = rec.read(crops)
    seconds = time.monotonic() - started
    geometry_sha = hashlib.sha256(json.dumps(geometries, separators=(",", ":")).encode()).hexdigest()
    return {"case": case, **aggregate(scored, predictions), "inverted_lines": inverted,
            "geometry_sha256": geometry_sha, "fully_visible_ink_lines": len(scored),
            "padding_clamped_lines": sum(any(actual[key] != case[key] for key in actual) for actual in actuals),
            "actual_padding": {key: {"min": min(actual[key] for actual in actuals),
                                     "max": max(actual[key] for actual in actuals)}
                               for key in ("left", "right", "top", "bottom")},
            "host_inference_seconds": round(seconds, 6)}


def roi_case(rec, rows, images, offset):
    groups = {}
    for row in rows:
        if row.get("region") is None:
            raise ValueError("ROI stress requires a dialogue region on every line")
        groups.setdefault((row["image_path"], tuple(row["region"])), []).append(row)
    references, predictions, geometries = [], [], []
    found = detections = missed = rejected = invalid_regions = clamped = inverted = 0
    for (image_path, region), group in groups.items():
        image = images[image_path]
        height, width = image.shape[:2]
        x, y, w, h = region
        desired = (x - offset, y - offset, x + w + offset, y + h + offset)
        bounds = (max(0, desired[0]), max(0, desired[1]), min(width, desired[2]), min(height, desired[3]))
        rx0, ry0, rx1, ry1 = bounds
        if rx0 >= rx1 or ry0 >= ry1 or any(
                row["box"][0] < rx0 or row["box"][1] < ry0
                or row["box"][0] + row["box"][2] > rx1
                or row["box"][1] + row["box"][3] > ry1 for row in group):
            invalid_regions += 1
            continue
        clamped += bounds != desired
        geometries.append([group[0]["sha256"], list(bounds), sorted(row["id"] for row in group)])
        gts = [{**row, "box_r": (row["box"][0] - rx0, row["box"][1] - ry0,
                                 row["box"][2], row["box"][3])} for row in group]
        dets = finders.console(image[ry0:ry1, rx0:rx1])
        if dets is None:
            rejected += 1
            dets = []
        crops = []
        for det in dets:
            dx0, dy0, dx1, dy1 = box_bounds(list(map(int, det)), rx1 - rx0, ry1 - ry0)
            crops.append(image[ry0 + dy0:ry0 + dy1, rx0 + dx0:rx0 + dx1])
        crops, count = console_crops(crops)
        inverted += count
        preds = rec.read(crops) if crops else []
        if len(preds) != len(dets):
            raise ValueError("recognizer did not return one prediction per detection")
        pairs, unmatched = finders.match(dets, gts)
        matched = {j for _, j in pairs}
        ignored = sum(bool(gts[j].get("uncertain")) for _, j in pairs)
        detections += len(dets) - ignored
        for i, j in pairs:
            if not gts[j].get("uncertain"):
                references.append(gts[j])
                predictions.append(preds[i])
                found += 1
        for j, row in enumerate(gts):
            if j not in matched and not row.get("uncertain"):
                references.append(row)
                predictions.append("")
                missed += 1
        for i in unmatched:
            references.append({"text": ""})
            predictions.append(preds[i])
    geometry_sha = hashlib.sha256(json.dumps(geometries, separators=(",", ":")).encode()).hexdigest()
    return {"roi_offset": offset, "finder": "console Python bit-exact twin of runtime/vocr_lines.c",
            **aggregate(references, predictions), "regions": len(groups) - invalid_regions,
            "regions_skipped_to_preserve_ink": invalid_regions, "regions_rejected": rejected,
            "padding_clamped_regions": clamped, "fully_visible_gt_lines": found + missed,
            "found_lines": found, "detections": detections,
            "line_recall": round(found / (found + missed), 5) if found + missed else None,
            "line_precision": round(found / detections, 5) if detections else None,
            "inverted_lines": inverted, "geometry_sha256": geometry_sha}


def run(manifest, model, out, roi_offsets=()):
    out, model = Path(out).resolve(), Path(model).resolve()
    if out.exists():
        raise FileExistsError(f"crop stress result already exists: {out}")
    rows, images, dataset = load_development(manifest)
    model_hash = sha256(model)
    rec = NativeVocr(model)
    try:
        cases = [line_case(rec, rows, images, case) for case in CASES]
        rois = [roi_case(rec, rows, images, offset) for offset in roi_offsets]
        info = {key: rec.info[key] for key in ("backend", "mode", "sha256", "blob_bytes", "arena_bytes")}
    finally:
        rec.close()
    if sha256(model) != model_hash or info["sha256"] != model_hash:
        raise ValueError("evaluated model changed during crop stress")
    result = {"schema": 1, "purpose": "development_diagnostic_only", "dataset": dataset,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "host": {"platform": platform.platform(), "machine": platform.machine()},
              "model_info": info, "polarity": "console", "max_width": 1024,
              "runtime_source_sha256": sha256(ROOT / "runtime" / "vocr.c"),
              "cases": cases, "roi_cases": rois, "heldout_test_evaluated": False,
              "padding_contract": "all annotated ink visible; only image-boundary padding may clamp"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="int8 VOCR blob, scalar C W8I")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roi-offset", type=int, nargs="*", default=[])
    args = parser.parse_args()
    result = run(args.manifest, args.model, args.out, args.roi_offset)
    for case in result["cases"]:
        print(json.dumps({"case": case["case"]["name"], "full": case["full"], "hiragana": case["hiragana"]}))
    for case in result["roi_cases"]:
        print(json.dumps({"roi_offset": case["roi_offset"], "full": case["full"],
                          "hiragana": case["hiragana"], "line_recall": case["line_recall"],
                          "line_precision": case["line_precision"]}))


if __name__ == "__main__":
    main()

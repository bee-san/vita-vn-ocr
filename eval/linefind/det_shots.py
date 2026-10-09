#!/usr/bin/env python3
"""Run PP-OCRv5 server text detection over screenshots (build tool for the line-finder GT).

  det_shots.py --shots 'data/vndb/shots/*.jpg' --out DIR/det.jsonl

One JSON row per screenshot: file name, sha256, image size, and every detected text
quad with its score. Resumable: screenshots already in the output are skipped. The
detector only proposes line boxes; build_gt.py turns them into line-level ground
truth and every set is checked by eye on contact sheets before use.
Honours OMP_NUM_THREADS (run it through heavy.sh).
"""
import argparse
import glob
import hashlib
import json
import os
import time
from pathlib import Path

import cv2


def make_engine(threads):
    from rapidocr import ModelType, OCRVersion, RapidOCR
    return RapidOCR(params={"Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.SERVER,
                            "Global.log_level": "error",
                            "EngineConfig.onnxruntime.intra_op_num_threads": threads})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True, help="glob of screenshot files")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    paths = sorted(glob.glob(args.shots))
    done = set()
    if args.out.exists():
        done = {json.loads(x)["shot"] for x in args.out.read_text().splitlines() if x.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    eng = make_engine(threads)
    t0 = time.time()
    with open(args.out, "a") as f:
        for i, p in enumerate(paths):
            name = Path(p).name
            if name in done:
                continue
            raw = Path(p).read_bytes()
            img = cv2.imread(p)
            if img is None:
                continue
            r = eng(img, use_det=True, use_cls=False, use_rec=False)
            quads = [] if r.boxes is None else [[[round(float(v), 1) for v in pt] for pt in q] for q in r.boxes]
            scores = [round(float(s), 4) for s in (r.scores or [])]
            f.write(json.dumps({"shot": name, "sha256": hashlib.sha256(raw).hexdigest(),
                                "size": [img.shape[1], img.shape[0]], "quads": quads,
                                "scores": scores}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    print("DET_DONE", flush=True)


if __name__ == "__main__":
    main()

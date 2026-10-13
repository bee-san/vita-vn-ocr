#!/usr/bin/env python3
"""Read the selected screenshots with the two local engines (the slow, model-heavy stage).

  read.py run [--shots $OUT/shots.jsonl] [--procs 5] [--out $OUT/reads]

Run it offline, in one heavy.sh slot, with the teacher env (MeikiOCR):
  HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 unshare -rn teacher/.venv/bin/python devnative/read.py run --procs 5

Per screenshot, at the console's resolution (native shots as they are; a shot of another size
read as kind vita or psp is first scaled to fit that screen, paths.load_screen): MeikiOCR
(teacher/engines.Meiki, pinned files) detects and reads every text line;
teacher/geometry.lines_of merges row pieces, refines each box to its ink and attaches
furigana (eval's conventions). Every horizontal line then gets four readings, all kept for
the contact sheets:
  meiki   MeikiOCR's own reading of its detected (merged) line, with character boxes
  meikiv  MeikiOCR's recognizer on the line's view (the ink box widened by 0.6 line heights
          left and right, so a line the detector cut short reads longer)
  v6s     PP-OCRv6 small (pinned ONNX, teacher/engines.PaddleRec) on the view
  v6sc    PP-OCRv6 small on the ink crop itself (ink + furigana + 4/2 px)
Every raw detector box (any orientation) is kept too, for the GRAD crops' "mixed" test.
Nothing is decided here. Output: <out>/reads.jsonl (one row per screenshot) and engines.json
(the engine files' sha256).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from devnative import paths as P  # noqa: E402

_ENG = {}


def _init():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from teacher import engines
    _ENG.update(engines.load(["meiki", "v6s"], n_threads=1))


def read_shot(r, eng=None):
    import cv2

    from devnative import lines as L
    from teacher import geometry as G
    eng = eng or _ENG
    rgb = P.load_screen(r)
    if rgb is None:
        return {"shot": r["shot"], "error": "unreadable"}
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    t0 = time.perf_counter()
    dets = eng["meiki"].ocr(bgr)
    lines = G.lines_of(dets, rgb)
    out = []
    for k, ln in enumerate(lines):
        crop = L.crop_box(ln, r["kind"], W, H)
        rec = {"k": k, "det": ln["det"], "det_conf": ln["det_conf"], "parts": ln["parts"], "ink": ln["ink"],
               "crop_ink": ln["crop"], "crop": crop, "view": ln["view"], "edge": ln["edge"], "hint": ln["hint"],
               "reads": {"meiki": [ln["text"], ln["chars"]]}}
        if ln.get("furi"):
            rec["furi"], rec["furi_text"] = ln["furi"], ln["furi_text"]
        x0, y0, x1, y1 = ln["view"]
        view = bgr[y0:y1, x0:x1]
        if view.size:
            t, mean, low, chars = eng["meiki"](view)
            rec["reads"]["meikiv"] = [t, mean, low, [[c[0], c[1] + x0, c[2] + y0, c[3] + x0, c[4] + y0, c[5]]
                                                     for c in chars]]
            rec["reads"]["v6s"] = eng["v6s"](view)
        cx0, cy0, cx1, cy1 = crop
        if cx1 > cx0 and cy1 > cy0:
            rec["reads"]["v6sc"] = eng["v6s"](bgr[cy0:cy1, cx0:cx1])
        out.append(rec)
    return {"shot": r["shot"], "subset": r.get("subset"), "vn": r["vn"], "kind": r["kind"], "url": r["url"],
            "release": r.get("release"), "path": str(P.screen_path(r)), "dims": [W, H],
            "flags": r.get("flags", []),
            "dets": [{"det": d["det"], "vertical": d["vertical"], "text": d["text"]} for d in dets],
            "lines": out, "secs": round(time.perf_counter() - t0, 3)}


def engine_pins():
    from teacher import engines
    out = {}
    for name in ("meiki_det", "meiki_rec", "v6s"):
        p = engines.model_path(name)
        out[name] = {"file": p.name, "sha256": P.file_sha256(p), "pinned": engines.PINS[name][2]}
        if out[name]["sha256"] != out[name]["pinned"]:
            sys.exit(f"{p}: sha256 differs from the pin")
    import importlib.metadata as md
    out["packages"] = {k: md.version(k) for k in ("meikiocr", "onnxruntime", "numpy", "opencv-python-headless")}
    return out


def cmd_run(args):
    shots = [json.loads(x) for x in Path(args.shots).read_text(encoding="utf-8").splitlines() if x.strip()]
    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "engines.json").write_text(json.dumps(engine_pins(), indent=1) + "\n")
    path = outd / "reads.jsonl"
    done = set()
    if path.exists():
        for x in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(x)["shot"])
            except (ValueError, KeyError):
                continue
    todo = [r for r in shots if r["shot"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(shots)} screenshots, {len(done)} already read, {len(todo)} to read, {args.procs} workers",
          flush=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo, mp.get_context("spawn").Pool(args.procs, initializer=_init) as pool:
        for i, row in enumerate(pool.imap_unordered(read_shot, todo, chunksize=1)):
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(todo)} screenshots, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    print("READ_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--shots", default=str(P.OUT / "shots.jsonl"))
    p.add_argument("--out", default=str(P.OUT / "reads"))
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"            # one thread per worker; heavy.sh's slot has 5 cores
    main()

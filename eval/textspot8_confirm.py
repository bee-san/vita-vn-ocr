#!/usr/bin/env python3
"""textspot8's confirmation read (eval/textspot8_protocol.json "rule": guard_confirmation): eval/pagepath.py --boxes
files of textspot8's decodes on vocr6-spotter3's frozen confirmation units ($PV/vocr6-spotter3/confirm/units_v1.pkl,
sha256 afc93c81...; public manifest eval/textspot7_confirm_units_v1.json on feat/textspot7 3781f2f), read ONCE, after
the dev / hval decision is recorded.

  textspot8_confirm.py boxes --spot SPOT.jsonl --decode det|det+refine --out BOXES.jsonl
      SPOT: eval/textspot7.py spot-confirm's file (the twin on every confirmation frame in the console view: the grey
      frame, a PSP one upscaled 2x bilinear on its grey); each unit keeps the horizontal boxes whose centre lies in its
      region, clipped (eval/textspot6.region_boxes), det or the refined boxes (det+refine)
  textspot8_confirm.py boxes --blob B.vtsp --decode crop|crop+refine|ctx --out BOXES.jsonl [--margin 48]
      crop: the spotter on each unit's region alone (its grey image, a PSP one upscaled 2x as eval/pagepath.py upscales
      it) at the console frame's scale (960 x 544); ctx: on the region grown by the margin in the console frame

The units' uids are eval/pagepath.py's (dev:<i>:<unit>:<variant>). Everything here is private.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6 as E6  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
PRIV = HERD / "shared" / "private" / "vocr6-spotter3" / "confirm"
UNITS = Path(os.environ.get("T8_CONFIRM_UNITS", str(PRIV / "units_v1.pkl")))
SHOTS = PRIV / "shots"
UNITS_SHA256 = "afc93c818ba29dfa3f96177feeb7c3fac32645c219f53a2cc0f96f898e41de89"
FRAME_W, FRAME_H = 960, 544


def units():
    if E6.sha256_file(UNITS) != UNITS_SHA256:
        raise SystemExit(f"{UNITS}: not the frozen confirmation units")
    return pickle.load(open(UNITS, "rb"))["units"]


def frames():
    """{frame id: console-view grey frame} (eval/textspot7.confirm_frames' convention, each unit's region checked)."""
    import cv2

    from common import to_gray
    us = units()
    got = {r["id"]: r for r in E6.read_jsonl(SHOTS / "fetched.jsonl") if r.get("ok")}
    by = {}
    for u in us:
        by.setdefault(u["shot"], []).append(u)
    out = {}
    for fid, grp in sorted(by.items()):
        path = SHOTS / got[fid]["file"]
        if E6.sha256_file(path) != got[fid]["sha256"]:
            raise SystemExit(f"{path}: sha256 changed")
        g = to_gray(np.ascontiguousarray(cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1]))
        for u in grp:
            x0, y0, x1, y1 = u["region"]
            if not (g[y0:y1, x0:x1] == u["image"]).all():
                raise SystemExit(f"{u['unit']}: the frame's region is not the unit's image")
        if grp[0]["kind"] == "psp":
            g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        out[fid] = np.ascontiguousarray(g)
    return out


_W = {}


def _crop_one(job):
    uid, img = job
    return {"unit": uid, "boxes": E6.crop_boxes(_W["model"], img, FRAME_W, FRAME_H, _W["k"])}


def _ctx_one(job):
    import textspot6_twin as T6
    uid, img, reg = job
    m = _W["model"]
    r = T6.spot(m, img, scale=T6.scale_of(m, FRAME_W, FRAME_H))
    boxes = [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, b.tag, b.tconf] for b in r.boxes]
    return {"unit": uid, "boxes": E6.region_boxes(boxes, reg)}


def cmd_boxes(a):
    import multiprocessing as mp

    import cv2
    t0 = time.time()
    us = units()
    out = []
    if a.decode in ("det", "det+refine"):
        meta, rows = E6.load_spot(a.spot)
        by = {r["frame"]: r for r in rows}
        k = "det" if a.decode == "det" else "detref"
        for i, u in enumerate(us):
            s = 2 if u["kind"] == "psp" else 1
            reg = tuple(v * s for v in u["region"])
            out.append({"unit": f"dev:{i}:{u['unit']}:{u['variant']}",
                        "boxes": E6.region_boxes(E6.decoded(by[u["shot"]], k), reg)})
        src = {"spot": str(a.spot), "spot_sha256": E6.sha256_file(a.spot), "blob_sha256": meta.get("blob_sha256")}
    else:
        import textspot6_twin as T6
        blob = Path(a.blob).read_bytes()
        _W["model"], _W["k"] = T6.Model(blob), ("detref" if a.decode == "crop+refine" else "det")
        if a.decode in ("crop", "crop+refine"):
            jobs = []
            for i, u in enumerate(us):
                img = u["image"]
                if u["kind"] == "psp":
                    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                jobs.append((f"dev:{i}:{u['unit']}:{u['variant']}", np.ascontiguousarray(img)))
            fn = _crop_one
        else:
            fr = frames()
            jobs = []
            for i, u in enumerate(us):
                g = fr[u["shot"]]
                s = 2 if u["kind"] == "psp" else 1
                reg = tuple(v * s for v in u["region"])
                x0, y0 = max(0, reg[0] - a.margin), max(0, reg[1] - a.margin)
                x1, y1 = min(g.shape[1], reg[2] + a.margin), min(g.shape[0], reg[3] + a.margin)
                jobs.append((f"dev:{i}:{u['unit']}:{u['variant']}", np.ascontiguousarray(g[y0:y1, x0:x1]),
                             (reg[0] - x0, reg[1] - y0, reg[2] - x0, reg[3] - y0)))
            fn = _ctx_one
        with mp.get_context("fork").Pool(a.workers) as pool:
            out = list(pool.imap(fn, jobs, chunksize=8))
        import hashlib
        src = {"blob": str(a.blob), "blob_sha256": hashlib.sha256(blob).hexdigest(), "margin": a.margin}
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    Path(str(a.out).replace(".jsonl", ".meta.json")).write_text(json.dumps(
        {"decode": a.decode, "units": len(out), "units_sha256": UNITS_SHA256, **src,
         "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n", encoding="utf-8")
    print(f"textspot8 confirm boxes written: {a.decode} {len(out)} units {time.time() - t0:.0f} s -> {a.out}",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("boxes")
    b.add_argument("--decode", required=True, choices=("crop", "crop+refine", "ctx", "det", "det+refine"))
    b.add_argument("--spot")
    b.add_argument("--blob")
    b.add_argument("--out", required=True)
    b.add_argument("--margin", type=int, default=48)
    b.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    if a.decode.startswith("det") and not a.spot:
        ap.error("det decodes read --spot (eval/textspot7.py spot-confirm's file)")
    if not a.decode.startswith("det") and not a.blob:
        ap.error("crop and ctx decodes run --blob")
    cmd_boxes(a)


if __name__ == "__main__":
    main()

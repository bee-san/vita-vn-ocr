#!/usr/bin/env python3
"""textspot8's context decode (eval/textspot8_protocol.json amendment 4): the spotter on the player's region grown by
a margin, so it sees the region's surroundings without a whole-frame capture, then the horizontal boxes whose centre
lies in the region, clipped to it (eval/textspot6.region_boxes' rule). It is the crop decode with context: what
Rubi can run by capturing a slightly larger rectangle than the region.

  textspot8_ctx.py boxes --blob B.vtsp --set dev|hval --out BOXES.jsonl [--margin 48] [--units HVAL_UNITS] [--workers 5]
      eval/pagepath.py --boxes rows [[x, y, w, h, dark]] in each unit's region px at its view scale (dev: #119's units
      on textspot6's console-view dev frames; hval: the hval units cut from the console-view pages, as eval/textspot6.py
      spot-crop cuts them); the grown region is read at the scale the blob's rule gives its frame (amendment 3 of
      textspot6: textspot6_twin.spot(scale=scale_of(frame)))
  textspot8_ctx.py parity --blob B.vtsp --set dev|hval --out PARITY.json --cbin DIR [--margin 48] [--workers 5]
      runtime/vocr_textspot6.c (eval/textspot6_parity.py's driver, VOCR_TEXTSPOT6_SCALE1 / _SCALE2) against the twin on
      every grown region of the set

Everything read here is private (game frames); BOXES files hold coordinates only.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6 as E6  # noqa: E402

MARGIN = 48


def grown(region, margin, W, H):
    x0, y0, x1, y1 = region
    return max(0, x0 - margin), max(0, y0 - margin), min(W, x1 + margin), min(H, y1 + margin)


def ctx_jobs(set_name, margin, units_path=None):
    """[(uid, grown region grey, scale frame (W, H), region in grown px (x0, y0, x1, y1))]."""
    import cv2

    from common import to_gray
    if set_name == "dev":
        frames = {k["frame"]: g for k, g in E6.dev_frames()}
        for i, u in enumerate(E6.dev_units()):
            g = frames[E6.dev_frame_key(u)]
            s = 2 if u["kind"] == "psp" else 1
            reg = tuple(v * s for v in u["region"])
            gx0, gy0, gx1, gy1 = grown(reg, margin, g.shape[1], g.shape[0])
            yield (f"dev:{i}:{u['unit']}:{u['variant']}", np.ascontiguousarray(g[gy0:gy1, gx0:gx1]),
                   (g.shape[1], g.shape[0]), (reg[0] - gx0, reg[1] - gy0, reg[2] - gx0, reg[3] - gy0))
        return
    rows = E6.read_jsonl(units_path)
    cache = {}
    for u in (r for r in rows if "uid" in r):
        k = (u["title"], u["page"])
        if k not in cache:
            cache.clear()
            bgr = cv2.imread(str(Path(E6.PAGES) / u["title"] / "pages" / f"{u['page']}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"{u['title']}:{u['page']}")
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            if u["platform"] == "psp":
                rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cache[k] = to_gray(np.ascontiguousarray(rgb))
        g = cache[k]
        s = 2 if u["platform"] == "psp" else 1
        reg = tuple(v * s for v in u["region"])
        gx0, gy0, gx1, gy1 = grown(reg, margin, g.shape[1], g.shape[0])
        yield (u["uid"], np.ascontiguousarray(g[gy0:gy1, gx0:gx1]), (g.shape[1], g.shape[0]),
               (reg[0] - gx0, reg[1] - gy0, reg[2] - gx0, reg[3] - gy0))


_W = {}


def _boxes_one(job):
    import textspot6_twin as T6
    uid, img, (fw, fh), reg = job
    m = _W["model"]
    r = T6.spot(m, img, scale=T6.scale_of(m, fw, fh))
    boxes = [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, b.tag, b.tconf] for b in r.boxes]
    return {"unit": uid, "boxes": E6.region_boxes(boxes, reg)}


def cmd_boxes(a):
    import textspot6_twin as T6
    blob = Path(a.blob).read_bytes()
    _W["model"] = T6.Model(blob)
    t0 = time.time()
    jobs = list(ctx_jobs(a.set, a.margin, a.units))
    out = Path(a.out)
    tmp = Path(str(out) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for rec in pool.imap(_boxes_one, jobs, chunksize=8):
            f.write(json.dumps(rec) + "\n")
    tmp.replace(out)
    meta = {"blob": str(a.blob), "blob_sha256": hashlib.sha256(blob).hexdigest(), "set": a.set, "decode": "ctx",
            "margin": a.margin, "units": len(jobs), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(str(out).replace(".jsonl", ".meta.json")).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"textspot8 ctx boxes written: {a.set} {len(jobs)} units margin {a.margin} {time.time() - t0:.0f} s -> {out}",
          flush=True)


def _twin(job):
    import textspot6_parity as P6
    uid, img, (fw, fh), _reg = job
    import textspot6_twin as T6
    return P6.twin_line((uid, img, T6.scale_of(_W["model"], fw, fh)))


def cmd_parity(a):
    import textspot6_parity as P6
    import textspot6_twin as T6
    blob = Path(a.blob).read_bytes()
    _W["model"] = P6._W["model"] = T6.Model(blob)
    exe, src_sha = P6.build(a.cbin)
    jobs = list(ctx_jobs(a.set, a.margin, a.units))
    n = n_eq = 0
    first = None
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k in range(0, len(jobs), 400):
            chunk = jobs[k:k + 400]
            tw = pool.map(_twin, chunk, chunksize=8)
            cl = P6.c_lines(exe, a.blob, [(uid, img, T6.scale_of(_W["model"], fw, fh))
                                         for uid, img, (fw, fh), _ in chunk])
            for (key, t), c in zip(tw, cl):
                n += 1
                if t == c:
                    n_eq += 1
                elif first is None:
                    first = {"key": key, "twin": t[:300], "c": c[:300]}
    res = {"kind": "textspot8-ctx-parity", "blob": str(a.blob), "blob_sha256": hashlib.sha256(blob).hexdigest(),
           "runtime_sources_sha256": src_sha, "set": a.set, "margin": a.margin, "records": n, "equal": n_eq,
           "all_equal": n == n_eq, "first_difference": first}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"textspot8 ctx parity {a.set}: {n_eq}/{n} equal", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("boxes", cmd_boxes), ("parity", cmd_parity)):
        p = sub.add_parser(name)
        p.add_argument("--blob", required=True)
        p.add_argument("--set", required=True, choices=("dev", "hval"))
        p.add_argument("--out", required=True)
        p.add_argument("--margin", type=int, default=MARGIN)
        p.add_argument("--units", default=str(E6.ART / "vocr6-spotter" / "eval" / "units" / "hval_v1.jsonl"))
        p.add_argument("--workers", type=int, default=5)
        if name == "parity":
            p.add_argument("--cbin", required=True)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

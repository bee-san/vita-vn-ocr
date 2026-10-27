#!/usr/bin/env python3
"""textspot8's orientation check (report only; docs/textspot8.md "Vertical text in Rubi"): Rubi's spotter path hands a
region to GRAD's path, whose rows ocr_vertical = auto checks for vertical text (API 47's rule), when the spotter's crop
decode returns more vertical than horizontal boxes in it. How often does that happen on horizontal dialogue (it should
not), and on vertical text (it should)?

  textspot8_orient.py --blob B.vtsp --out RES.json [--workers 5]

Sets, each region read as Rubi reads a region of the console's 960 x 544 picture (a PSP region upscaled 2x bilinear;
the spotter at the 960 x 544 frame's scale):
  dev      #119's whole-dialogue units (horizontal; 1,161)
  confirm  vocr6-spotter3's frozen confirmation units (horizontal; 516), when present
  vertical eval/manifests/vertical-dev-regions.jsonl, from the vertical study's local copy with page paths
           ($A/vertical/sets/regions_dev.jsonl; vertical text: VN pages and manga; development only)
RES.json holds counts only.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_W = {}


def counts(img):
    import textspot6_twin as T6
    m = _W["model"]
    r = T6.spot(m, np.ascontiguousarray(img), scale=T6.scale_of(m, 960, 544))
    v = sum(1 for b in r.boxes if b.vertical)
    return v, len(r.boxes) - v


def _job(job):
    key, img = job
    return key, counts(img)


def unit_jobs(units):
    import cv2
    for i, u in enumerate(units):
        img = u["image"]
        if u["kind"] == "psp":
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        yield f"{u['vn']}|{i}", img


def vertical_jobs():
    import cv2
    import textspot6 as E6
    import vertical as V
    # the vertical study's local copy of eval/manifests/vertical-dev-regions.jsonl, with each page's local path
    path = Path(E6.ART) / "vertical" / "sets" / "regions_dev.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    cache = {}
    for r in rows:
        if r["page"] not in cache:
            cache.clear()
            cache[r["page"]] = V.load_page_grey(r)
        g = cache[r["page"]]
        if g is None:
            continue
        x, y, w, h = r["region"]
        img = g[y:y + h, x:x + w]
        if r["kind"] == "psp":
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        yield f"{r.get('cohort', 'vn')}|{r['id']}", img


def summarize(res):
    n = len(res)
    vmaj = sum(1 for _, (v, h) in res if v > h)
    nobox = sum(1 for _, (v, h) in res if v + h == 0)
    by = {}
    for k, (v, h) in res:
        c = by.setdefault(k.split("|")[0], [0, 0])
        c[0] += 1
        c[1] += v > h
    return {"regions": n, "vertical_majority": vmaj, "share": round(vmaj / max(1, n), 4), "no_box": nobox,
            "by_group": {k: {"regions": a, "vertical_majority": b} for k, (a, b) in sorted(by.items())}}


def main(argv=None):
    import multiprocessing as mp

    import textspot6 as E6

    import textspot6_twin as T6
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    blob = Path(a.blob).read_bytes()
    _W["model"] = T6.Model(blob)
    out = {"kind": "textspot8-orientation", "blob_sha256": hashlib.sha256(blob).hexdigest(), "sets": {}}
    sets = {"dev": lambda: unit_jobs(E6.dev_units())}
    conf = E6.HERD / "shared" / "private" / "vocr6-spotter3" / "confirm" / "units_v1.pkl"
    if conf.exists():
        sets["confirm"] = lambda: unit_jobs(pickle.load(open(conf, "rb"))["units"])
    sets["vertical"] = vertical_jobs
    with mp.get_context("fork").Pool(a.workers) as pool:
        for name, fn in sets.items():
            res = list(pool.imap(_job, list(fn()), chunksize=8))
            out["sets"][name] = summarize(res)
            s = out["sets"][name]
            print(f"textspot8 orient {name}: {s['vertical_majority']}/{s['regions']} regions with more vertical than "
                  f"horizontal boxes ({s['by_group'] if name == 'vertical' else ''})", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

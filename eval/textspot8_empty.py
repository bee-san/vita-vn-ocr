#!/usr/bin/env python3
"""textspot8's empty-region check (report only; docs/textspot8.md "Empty regions"): how often a spotter's crop decode
finds no line in a text-free region (a hidden text box, a CG screen), the case where Rubi's GRAD path reads junk.

  textspot8_empty.py --blob B.vtsp [--blob B2.vtsp ...] --out RES.json

The regions are eval/linefind's `empty` regions (eval/textgate.linefind_empties: development = vita_dev + psp_dev,
report = the two holdouts; a reserved test VN's regions left out), the same 66 + 25 textgate's study read. Each is
read as Rubi reads a region of the console's 960 x 544 picture: a PSP region upscaled 2x bilinear (Adrenaline), the
spotter at the 960 x 544 frame's scale (textspot6_twin.spot(scale=scale_of(m, 960, 544)): the crop decode). RES.json
holds counts and region ids only.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None):
    import cv2
    import textgate as TG

    import textspot6_twin as T6
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    empties = TG.linefind_empties()
    out = {"kind": "textspot8-empty-regions", "computed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "regions": {role: len(rows) for role, rows in empties.items()}, "blobs": {}}
    for path in a.blob:
        blob = Path(path).read_bytes()
        m = T6.Model(blob)
        f = T6.scale_of(m, 960, 544)
        rec = {"blob_sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob), "scale": f}
        for role, rows in empties.items():
            none, nboxes, ids_with = 0, [], []
            for r, g in rows:
                img = g
                if r.get("platform") == "psp":
                    img = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                res = T6.spot(m, np.ascontiguousarray(img), scale=f)
                hz = [b for b in res.boxes if not b.vertical]
                nboxes.append(len(hz))
                if not hz:
                    none += 1
                else:
                    ids_with.append(r["id"])
            rec[role] = {"regions": len(rows), "no_line": none, "no_line_share": round(none / max(1, len(rows)), 4),
                         "boxes_total": int(sum(nboxes)), "regions_with_a_box": ids_with}
        out["blobs"][str(path)] = rec
        print(f"{Path(path).name}: " + ", ".join(f"{role} {rec[role]['no_line']}/{rec[role]['regions']} no line"
                                                for role in empties), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

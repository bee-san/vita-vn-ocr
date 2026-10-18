#!/usr/bin/env python3
"""textgate: the C gate (runtime/vocr_textgate.c through its host driver runtime/vocr_textgate_test) against the
twin (eval/textgate.py), for the selection rule's eligibility (eval/textgate_protocol.json "selection_rule").

  textgate_c.py parity --dev DIR --cbin DRIVER --out JSON
      Every candidate of DIR/decisions.json (eval/textgate.py dev): its score from the C runtime on every
      development case and empty region, in the push form (what Rubi feeds from its finder's first pass) and the
      in-memory form, must equal the twin's score recorded in decisions.json. Writes {name: {"c_equals_twin", ...}}
      for eval/textgate.py select --parity.
"""
import argparse
import datetime
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FORMS = {"image": 0, "rows": 1, "push": 2}
FL_BOXES = 4


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def c_scores(cbin, blob, mode, k, images, boxes=None, form="push"):
    """The C runtime's score per grey image (runtime/vocr_textgate_test --batch); boxes: per image, the finder's
    (x, y, w, h) boxes for the boxes mode."""
    flags = FORMS[form] | (FL_BOXES if mode == "boxes" else 0) | (int(k) << 4)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for i, img in enumerate(images):
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape
            f.write(struct.pack("<4I", w, h, 1, flags))
            f.write(img.tobytes())
            if mode == "boxes":
                bx = list((boxes or [])[i] or [])
                f.write(struct.pack("<I", len(bx)))
                for b in bx:
                    f.write(struct.pack("<4I", *(int(v) for v in b)))
        path = f.name
    try:
        out = subprocess.run([str(cbin), "--batch", str(blob), path], check=True, capture_output=True,
                             text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(out) != len(images):
        raise RuntimeError(f"the C gate returned {len(out)} results for {len(images)} images")
    res = []
    for line in out:
        rc, sc = (int(t) for t in line.split()[:2])
        if rc:
            raise RuntimeError(f"vocr_textgate rc={rc}")
        res.append(sc)
    return res


def cmd_parity(args):
    import textdet as TD
    import textgate as G
    dec = json.loads((Path(args.dev) / "decisions.json").read_text(encoding="utf-8"))
    cases = [(n, c) for n in G.WD_COHORTS for c in TD.load_cohort(args.cohorts, n)]
    if [TD.case_key(n, c) for n, c in cases] != dec["wd_cases"]:
        raise SystemExit("the cohorts are not the decisions' cases")
    empties = G.linefind_empties()
    ident = G.finder_driver(Path(args.out).parent / f"parity-build-{os.getpid()}")
    gw = [g for g, _ in G.grad_refine(ident["driver"], [c["image"] for _, c in cases])]
    ge = {role: [g for g, _ in G.grad_refine(ident["driver"], [g for _, g in rows])] for role, rows in empties.items()}
    out = {}
    for key, cand in dec["candidates"].items():
        if cand["family"] != "learned":
            continue
        name = cand["name"]
        cal = json.loads(Path(args.calibration).read_text())["candidates"][name] if args.calibration else None
        blob = cal["blob"] if cal else args.blob_dir.rstrip("/") + f"/{name}/final.vtd"
        if sha256(blob) != cand["blob_sha256"]:
            raise SystemExit(f"{name}: blob differs from the decisions' sha256")
        entry = out.setdefault(name, {"c_equals_twin": True, "checked": [], "blob_sha256": cand["blob_sha256"]})
        for form in args.form:
            cw = c_scores(args.cbin, blob, cand["mode"], cand["k"], [c["image"] for _, c in cases], gw, form)
            ok = cw == cand["wd_scores"]
            for role, rows in empties.items():
                ce = c_scores(args.cbin, blob, cand["mode"], cand["k"], [g for _, g in rows], ge[role], form)
                ok = ok and ce == cand["empty_scores"][role]
            entry["checked"].append({"operating_point": key, "form": form, "equal": ok,
                                     "cases": len(cw) + sum(len(r) for r in empties.values())})
            entry["c_equals_twin"] = entry["c_equals_twin"] and ok
            print(f"{key} {form}: C == twin {ok}", flush=True)
    doc = {"kind": "textgate-parity", "computed": datetime.datetime.now(datetime.timezone.utc)
           .strftime("%Y-%m-%dT%H:%M:%SZ"), "driver_sha256": sha256(args.cbin),
           "decisions_sha256": sha256(Path(args.dev) / "decisions.json"), "forms": args.form, "candidates": out}
    Path(args.out).write_text(json.dumps(doc["candidates"], indent=1) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".record.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--dev", required=True)
    p.add_argument("--cbin", required=True)
    p.add_argument("--calibration", default="")
    p.add_argument("--blob-dir", default="")
    p.add_argument("--cohorts", default=str(Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
                                            / "shared" / "artifacts" / "vocr5-train" / "cohorts"))
    p.add_argument("--form", action="append", default=None, choices=sorted(FORMS))
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    args.form = args.form or ["push", "image"]
    {"parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    main()

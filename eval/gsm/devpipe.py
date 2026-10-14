#!/usr/bin/env python3
"""gsm-research dev harness (DEV DATA ONLY).

Runs the real console path on devnative v3 regions: the C GRAD finder cuts each
labelled dialogue region into line crops, an optional crop post-processor runs,
then the C W8I recognizer reads each crop. Detections are matched to the
hand-labelled lines (vertical overlap >= 0.5, one-to-one, greedy), exactly as
eval/linefind does, and CER is reported per stratum.

Strata:
  names    -- labelled lines tagged 'name'
  dialogue -- labelled lines not tagged 'name'
  notext   -- detections that match no labelled line (finder false positives on
              scene art / plate decoration): the hallucination surface. Scored as
              characters written (lower is better; there is no reference).

NEVER touches a protected set. Reads only devnative v3 (and teacher_val for a
separate recognizer-only false-rejection sanity check). v1331 (owner Memories
Off) is excluded.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve()
# repo clone passed on the command line; default to the gsm-research clone
REPO = Path("/local/home/skerraut/herd-synth3/clones/gsm-research")
A = Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts")


def _imports(repo):
    sys.path.insert(0, str(repo))
    import synth.common as common  # noqa
    from eval.linefind.cfinder import Finder  # noqa
    from runtime.pyvocr import Vocr  # noqa
    from eval.metrics import norm  # noqa
    return common, Finder, Vocr, norm


def load_regions(exclude_vn=("v1331",)):
    rows = [json.loads(x) for x in open(A / "devnative/v3/lines_full.jsonl")]
    rows = [r for r in rows if r["vn"] not in exclude_vn]
    shots = {json.loads(x)["shot"]: json.loads(x)
             for x in open(A / "devnative/shots.jsonl")}
    regs = defaultdict(list)
    for r in rows:
        regs[(r["shot"], tuple(r["region"]))].append(r)
    return regs, shots


def vmatch(det, line_box):
    """vertical overlap / min height, and horizontal overlap bool (eval rule)."""
    dx, dy, dw, dh = det
    lx, ly, lw, lh = line_box
    oy = max(0, min(dy + dh, ly + lh) - max(dy, ly))
    ox = max(0, min(dx + dw, lx + lw) - max(dx, lx))
    v = oy / max(1, min(dh, lh))
    return v, ox > 0


def run(repo, post=None, strata_out=None, limit=0, tail_conf=0, blob="VOCR3_W20_w8.vocr"):
    common, Finder, Vocr, norm = _imports(repo)
    regs, shots = load_regions()
    finder = Finder("grad")
    rec = Vocr(str(A / "owner-drive-ingest/blobs" / blob), mode="w8i", tail_conf=tail_conf)

    agg = {k: {"ed": 0, "ref": 0, "lines": 0, "exact": 0} for k in ("names", "dialogue")}
    notext = {"dets": 0, "wrote": 0, "chars": 0}
    perline = []
    nreg = 0
    for (shot, region), ls in regs.items():
        if limit and nreg >= limit:
            break
        nreg += 1
        im = np.asarray(Image.open(shots[shot]["path"]).convert("RGB"))
        rx, ry, rw, rh = region
        sub = im[ry:ry + rh, rx:rx + rw]
        boxes = finder(sub)
        if boxes is None:
            boxes, darks = [], []
        else:
            darks = finder.last_dark
        # labelled-line boxes are in screenshot pixels; shift into region frame
        labels = []
        for ln in ls:
            ix1, iy1, ix2, iy2 = ln["ink"]
            labels.append((ln, (ix1 - rx, iy1 - ry, ix2 - ix1, iy2 - iy1)))
        # greedy one-to-one match by vertical overlap
        used = set()
        det_match = {}
        pairs = []
        for di, det in enumerate(boxes):
            best, bj = 0.5, None
            for li, (ln, lb) in enumerate(labels):
                if li in used:
                    continue
                v, hov = vmatch(det, lb)
                if hov and v >= best:
                    best, bj = v, li
            if bj is not None:
                used.add(bj)
                det_match[di] = bj
                pairs.append((di, bj))
        # recognize each detection
        for di, det in enumerate(boxes):
            dx, dy, dw, dh = det
            crop = common.grey_of(sub)[dy:dy + dh, dx:dx + dw]
            dark = darks[di] if di < len(darks) else 0
            box = det
            if post is not None:
                crop, box, dark = post(sub, det, dark, common)
            txt = norm(rec.text(crop, invert=bool(dark)))
            if di in det_match:
                ln = labels[det_match[di]][0]
                ref = norm(ln["label"])
                ed = _lev(txt, ref)
                strat = "names" if "name" in ln["styles"] else "dialogue"
                agg[strat]["ed"] += ed
                agg[strat]["ref"] += len(ref)
                agg[strat]["lines"] += 1
                agg[strat]["exact"] += int(txt == ref)
                perline.append({"id": ln["id"], "strat": strat, "ref": ref,
                                "pred": txt, "ed": ed})
            else:
                notext["dets"] += 1
                notext["wrote"] += int(len(txt) > 0)
                notext["chars"] += len(txt)
                perline.append({"id": f"{shot}@{di}", "strat": "notext",
                                "ref": "", "pred": txt, "ed": len(txt)})
    out = {"regions": nreg,
           "names": _rate(agg["names"]), "dialogue": _rate(agg["dialogue"]),
           "notext": notext}
    if strata_out:
        Path(strata_out).write_text(json.dumps({"summary": out, "lines": perline},
                                               ensure_ascii=False, indent=1))
    return out


def _rate(d):
    return {"cer": round(100 * d["ed"] / max(1, d["ref"]), 3), "ed": d["ed"],
            "ref": d["ref"], "lines": d["lines"],
            "exact": round(100 * d["exact"] / max(1, d["lines"]), 2)}


def _lev(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la:
        return lb
    if not lb:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--post", default="none")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--blob", default="VOCR3_W20_w8.vocr")
    args = ap.parse_args()
    post = None
    if args.post != "none":
        import importlib.util
        spec = importlib.util.spec_from_file_location("postmod", args.post)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        post = mod.post
    r = run(args.repo, post=post, strata_out=args.out, limit=args.limit, tail_conf=args.tail, blob=args.blob)
    print(json.dumps(r, ensure_ascii=False, indent=1))

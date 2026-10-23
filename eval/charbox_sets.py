#!/usr/bin/env python3
"""Development sets for per-character screen boxes (docs/char-boxes.md, issue #114).

Every unit is one hand-checked reference line (or vertical column) of a development frame, with the frame it
sits on, its main-text ink box, its text and, where the console's finder would read it, the region a player
draws. MeikiOCR's per-character boxes on that line are its reference boxes.

  units    collect the units (no pixels) from
             textspot-dev  eval/manifests/textspot-dev.jsonl: every labelled, non-ignored reference
                           (dialogue, name, choice, ui, scene) of the 188 frames (parts vita, psp,
                           adrenaline = psp-test style PSP frames at 2x, pc = scaled to fit 960x544);
                           a region per frame around its horizontal dialogue (+12 px; PSP +6)
             devnative     devnative's lines of reserved.json dev_sample VNs (Vita and PSP native, the
                           line's dialogue region), and every PSP one again on its frame at 2x
                           (cv2.INTER_LINEAR, Adrenaline's picture: part adrenaline)
             vertical-dev  eval/manifests/vertical-dev-columns.jsonl's labelled columns of the vertical
                           study's development split (#99; VN pages and the manga cohort; no test VN) with
                           their regions (vertical-dev-regions.jsonl)
  meiki    MeikiOCR (teacher/engines.Meiki, teacher/.venv) reads each unit's view (its ink box plus a
           margin, on the unit's frame; columns with MeikiOCR's vertical recognizer): its text and one box
           per character, in frame pixels. A reference character's box is MeikiOCR's character it is
           aligned to when the two characters are equal (metrics.norm of each); a line whose MeikiOCR
           reading is more than MAX_GT_CER from the reference (MAX_GT_CER_V for a column) keeps no
           reference boxes (reference_boxes(), applied when the score reads the readings)

Only development data: every horizontal unit's VN is in reserved.json dev_sample, and no unit's VN is a test VN
(the sets check it); no frame of vita-test, psp-test or CLANNAD is read.

  .venv-ci/bin/python eval/charbox_sets.py units --out $A/sets/units_dev.jsonl
  teacher/.venv/bin/python eval/charbox_sets.py meiki --units $A/sets/units_dev.jsonl --out $A/sets/meiki_dev.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "synth"))
from metrics import norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
TS_FRAMES = ART / "textspot" / "sets" / "dev" / "frames.jsonl"
DN_LINES = ART / "devnative" / "lines_full.jsonl"
DN_SHOTS = ART / "devnative" / "shots"
VC_COLS = ART / "vertical" / "sets" / "columns_dev.jsonl"
VC_REGIONS = ART / "vertical" / "sets" / "regions_dev.jsonl"
MANIFESTS = ROOT / "eval" / "manifests"
PAD = {"vita": 4, "psp": 2, "adrenaline": 4, "pc": 4}
MARGIN = {"vita": 12, "psp": 6, "adrenaline": 12, "pc": 12}
MAX_GT_CER = 0.2
MAX_GT_CER_V = 0.6


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dev_vns():
    from common import dev_sample_vns
    return set(dev_sample_vns())


def test_vns():
    from common import test_set_vns
    return set(test_set_vns())


def clip_box(box, dims):
    """[x, y, w, h] clipped to a frame of dims (W, H)."""
    W, H = dims
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(W, int(box[0] + box[2])), min(H, int(box[1] + box[3]))
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def grow(box, m, dims):
    return clip_box([box[0] - m, box[1] - m, box[2] + 2 * m, box[3] + 2 * m], dims)


def union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


# ----------------------------------------------------------------------------- units

def units_textspot(dev):
    paths = {r["id"]: r.get("path") for r in jsonl(TS_FRAMES)}
    out = []
    for row in jsonl(MANIFESTS / "textspot-dev.jsonl"):
        if row["vn"] not in dev or row.get("text_free"):
            continue
        part = row["part"]
        frame = {"id": row["id"], "url": row["url"], "sha256": row["sha256"], "path": paths.get(row["id"]),
                 "scale": row.get("scale"), "dims": row["dims"], "src_dims": row.get("src_dims")}
        refs = [r for r in row["refs"] if not r.get("ignore") and r.get("text") and r.get("cls")]
        dlg = [r["box"] for r in refs if r["cls"] == "dialogue" and not r.get("vertical")]
        region = grow(union(dlg), MARGIN[part], row["dims"]) if dlg else None
        for k, r in enumerate(row["refs"]):
            if r.get("ignore") or not r.get("text") or not r.get("cls"):
                continue
            reg = region if (r["cls"] == "dialogue" and not r.get("vertical")) else \
                grow(r["box"], MARGIN[part], row["dims"]) if r["cls"] == "name" and not r.get("vertical") else None
            out.append({"uid": f"ts:{row['id']}:{k}", "src": "textspot-dev", "vn": row["vn"], "part": part,
                        "frame": frame, "box": [int(v) for v in r["box"]], "text": r["text"], "cls": r["cls"],
                        "vertical": bool(r.get("vertical")), "region": reg, "pad": PAD[part]})
    return out


def units_devnative(dev):
    shots = {r["shot"]: r for r in jsonl(ROOT / "devnative" / "manifests" / "shots.jsonl")}
    out = []
    for r in jsonl(DN_LINES):
        if r["vn"] not in dev or not r.get("label"):
            continue
        kind = r["kind"]
        sh = shots.get(r["shot"])
        if sh is None:                 # tv_native lines: teacher_val's screenshots, kept under devnative/shots
            p = DN_SHOTS / kind / f"{r['shot']}.jpg"
            rgb = read_rgb(p)
            sh = shots[r["shot"]] = {"url": r["url"], "sha256": sha256(p), "dims": [rgb.shape[1], rgb.shape[0]]}
        W, H = sh["dims"]
        x0, y0, x1, y1 = r["ink"]
        box = [x0, y0, x1 - x0, y1 - y0]
        rx0, ry0, rx1, ry1 = r["region"]
        region = clip_box([rx0, ry0, rx1 - rx0, ry1 - ry0], (W, H))
        cls = "name" if "name" in (r.get("styles") or []) else "dialogue"
        frame = {"id": r["shot"], "url": sh["url"], "sha256": sh["sha256"],
                 "path": str(DN_SHOTS / kind / f"{r['shot']}.jpg"), "scale": None, "dims": [W, H]}
        out.append({"uid": f"dn:{r['id']}", "src": "devnative", "vn": r["vn"], "part": kind, "frame": frame,
                    "box": box, "text": r["label"], "cls": cls, "vertical": False, "region": region,
                    "pad": PAD[kind]})
        if kind == "psp":
            f2 = dict(frame, id=r["shot"] + "_x2", scale={"method": "cv2.INTER_LINEAR", "factor": 2},
                      dims=[2 * W, 2 * H])
            out.append({"uid": f"dn2:{r['id']}", "src": "devnative", "vn": r["vn"], "part": "adrenaline",
                        "frame": f2, "box": [2 * v for v in box], "text": r["label"], "cls": cls,
                        "vertical": False, "region": [2 * v for v in region], "pad": PAD["adrenaline"]})
    return out


def units_vertical(dev):
    """The vertical study's development columns (#99: its own split, never read by VOCR5-V's training; no test
    VN), VN pages and the manga cohort."""
    regions = {}
    for g in jsonl(VC_REGIONS):
        for c in g["columns"]:
            regions[c["id"]] = g["region"]
    out = []
    for c in jsonl(VC_COLS):
        if c.get("skip") or not c.get("text") or c.get("split") != "dev":
            continue
        part = c["kind"] if c["kind"] in PAD else "vita"
        scale = None
        if c.get("path") and c.get("dims"):
            import cv2
            im = cv2.imread(c["path"], cv2.IMREAD_UNCHANGED)
            sw, sh = im.shape[1], im.shape[0]
            if [sw, sh] != list(c["dims"]):          # a PC page scaled to fit the console (vertical_data.load_page)
                up = c["dims"][0] >= sw
                scale = {"method": "cv2.INTER_LINEAR" if up else "cv2.INTER_AREA"}
                part = "pc"
        frame = {"id": c["page"], "url": c.get("url"), "sha256": c.get("sha256"), "path": c.get("path"),
                 "scale": scale, "dims": c.get("dims")}
        out.append({"uid": f"vc:{c['id']}", "src": "vertical-dev", "vn": c["vn"], "part": part,
                    "cohort": c.get("cohort"), "frame": frame, "box": [int(v) for v in c["box"]], "text": c["text"],
                    "cls": "name" if c.get("name") else "dialogue", "vertical": True,
                    "region": regions.get(c["id"]), "pad": PAD[part]})
    return out


def cmd_units(args):
    dev, test = dev_vns(), test_vns()
    units = units_textspot(dev) + units_devnative(dev) + units_vertical(dev)
    bad = [u["uid"] for u in units if u["vn"] in test]
    if bad:
        raise SystemExit(f"{len(bad)} units on test VNs: {bad[:3]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter((u["src"], u["part"], u["cls"], u["vertical"]) for u in units))
    print(f"{len(units)} units, {len({u['vn'] for u in units})} VNs -> {args.out}")


# ----------------------------------------------------------------------------- frames

def read_rgb(path):
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{path}: not an image")
    return np.ascontiguousarray(bgr[:, :, ::-1])


_FRAMES = {}


def frame_rgb(fr):
    """The unit's frame (RGB uint8): its source, sha256-checked, scaled as its record says."""
    key = (fr["id"], json.dumps(fr.get("scale"), sort_keys=True))
    if key in _FRAMES:
        return _FRAMES[key]
    if len(_FRAMES) > 64:
        _FRAMES.clear()
    p = Path(fr["path"])
    if fr.get("sha256") and sha256(p) != fr["sha256"]:
        raise SystemExit(f"{p}: sha256 differs from the record")
    rgb = read_rgb(p)
    sc = fr.get("scale")
    if sc:
        import cv2
        W, H = fr["dims"]
        meth = getattr(cv2, str(sc["method"]).replace("cv2.", ""))
        rgb = np.ascontiguousarray(cv2.resize(rgb, (int(W), int(H)), interpolation=meth))
    if fr.get("dims") and [rgb.shape[1], rgb.shape[0]] != list(fr["dims"]):
        raise SystemExit(f"{fr['id']}: frame {rgb.shape[1]}x{rgb.shape[0]} is not {fr['dims']}")
    _FRAMES[key] = rgb
    return rgb


# ----------------------------------------------------------------------------- MeikiOCR reference boxes

def char_keys(text):
    """One normalised key per character of text (metrics.norm of the character alone, '' for whitespace)."""
    return [norm(c) for c in text]


def align(a, b):
    """Levenshtein alignment of two key lists: [(i, j)] index pairs (None for a gap), in order."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, out = n, m, []
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1], int(d[n, m])


def view_box(u, dims):
    """MeikiOCR's view of a unit: its ink box plus the console pad across the reading direction and half a line
    height along it, clipped to the frame."""
    x, y, w, h = u["box"]
    p = u["pad"]
    if not u["vertical"]:
        m = max(p, h // 2)
        return clip_box([x - m, y - p, w + 2 * m, h + 2 * p], dims)
    m = max(p, w // 2)
    return clip_box([x - p, y - m, w + 2 * p, h + 2 * m], dims)


def meiki_read(meiki, rgb, u):
    """(text, [[char, x0, y0, x1, y1, conf]] in frame px) of MeikiOCR on the unit's view."""
    H, W = rgb.shape[:2]
    vx, vy, vw, vh = view_box(u, (W, H))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if not u["vertical"]:
        crop = np.ascontiguousarray(bgr[vy:vy + vh, vx:vx + vw])
        text, _, _, chars = meiki(crop)
        return text, [[c[0], c[1] + vx, c[2] + vy, c[3] + vx, c[4] + vy, c[5]] for c in chars]
    m = meiki.m
    tb = [{"bbox": [vx, vy, vx + vw, vy + vh]}]
    res = [{"text": "", "chars": [], "is_vertical": True}]
    m._process_recognition_pipeline(bgr, tb, [0], res, meiki.rec_thr, meiki.punct, True)
    text = res[0]["text"].replace("\ufffd", "")
    return text, meiki._raw(res[0]["chars"])


def reference_boxes(u, text, chars):
    """([box or None] per character of the unit's text, MeikiOCR's CER against it): MeikiOCR's box of the
    aligned, equal character, on a line MeikiOCR reads within MAX_GT_CER of the reference (MAX_GT_CER_V for a
    column: MeikiOCR's vertical recognizer reads PSP and scaled PC columns poorly, #99)."""
    ref = char_keys(u["text"])
    got = [norm(c[0]) for c in chars]
    pairs, dist = align(ref, got)
    boxes = [None] * len(ref)
    nref = sum(1 for k in ref if k)
    cer = dist / max(1, nref)
    if cer > (MAX_GT_CER_V if u["vertical"] else MAX_GT_CER):
        return boxes, cer
    for i, j in pairs:
        if i is not None and j is not None and ref[i] and ref[i] == got[j]:
            boxes[i] = [int(v) for v in chars[j][1:5]]
    return boxes, cer


def cmd_meiki(args):
    sys.path.insert(0, str(ROOT))
    from teacher.engines import Meiki
    meiki = Meiki(n_threads=int(os.environ.get("OMP_NUM_THREADS", "5")))
    units = jsonl(args.units)
    done = set()
    if Path(args.out).exists():
        done = {r["uid"] for r in jsonl(args.out)}
    with open(args.out, "a", encoding="utf-8") as f:
        for k, u in enumerate(units):
            if u["uid"] in done:
                continue
            rgb = frame_rgb(u["frame"])
            text, chars = meiki_read(meiki, rgb, u)
            boxes, cer = reference_boxes(u, text, chars)
            f.write(json.dumps({"uid": u["uid"], "meiki_text": text, "meiki_chars": chars, "ref_boxes": boxes,
                                "meiki_cer": round(cer, 4)}, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                f.flush()
                print(f"{k}/{len(units)}", flush=True)
    print("MEIKI_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("units")
    a.add_argument("--out", required=True)
    a = sub.add_parser("meiki")
    a.add_argument("--units", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"units": cmd_units, "meiki": cmd_meiki}[args.cmd](args)


if __name__ == "__main__":
    main()

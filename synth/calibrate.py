#!/usr/bin/env python3
"""Calibrate synthetic crops against real VN screenshots (synth2).

Real calibration crops come only from training-side games: screenshots of VNs with
is_test_vn() False that are not in data/manifests/synth2_bg_exclude.txt (eval's
reserved test games and the hand-labelled real sets used for the A/B). Sources:

  vita-det   linefind's text-line detections on data/vndb/shots
             ($HERD/shared/artifacts/linefind/det/vita_shots_det.jsonl), horizontal
             boxes in the lower 55% of the screen, re-cut to finder geometry by the
             console's line finder (common.find_lines(img, "grad")) inside the box
  psp-band   PSP 480x272 screenshots (data/vndb_psp): the console's finder on the
             lower 42% of the screen (dialogue area; also picks up some UI text)
  real-bright data/real pseudo-label crops (the overlay's bright finder; train split)

  calibrate.py sheets --synth 'DIR/*.npz' [--out runs/calib]
      side-by-side contact sheets (real left, synthetic right, 32 px rows as the
      model sees them; every sheet <= 1280x1280)
  calibrate.py stats --synth 'old=data/synth/synth_00000.npz' --synth 'new=DIR/synth_0000[0-1].npz'
      crop height, aspect, text polarity, contrast and background busyness per source (JSON)
  calibrate.py finders [--n 2000]
      how often the console's finder (grad) or the overlay's (legacy) yields a usable
      crop where the renderer's mask finder does, per text scheme
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bg_exclude import load_exclusions  # noqa: E402
from common import DATA, find_lines, is_test_vn, to_gray  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
DET = HERD / "shared" / "artifacts" / "linefind" / "det" / "vita_shots_det.jsonl"


def excluded():
    return load_exclusions()


def usable(name, ex):
    vn = name.split("_")[0]
    return not is_test_vn(vn) and vn not in ex and name not in ex


def grad_recut(rgb, box, pad=6):
    """Re-cut one detected line to finder geometry: the gradient finder inside the box."""
    x0, y0, x1, y1 = box
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    sub = rgb[y0:y1, x0:x1]
    if sub.shape[0] < 10 or sub.shape[1] < 10:
        return None
    dets = find_lines(sub, "grad")
    if not dets:
        return None
    x, y, w, h = max(dets, key=lambda d: d[3] * d[2])
    if h < 10 or w < 2 * h:
        return None
    return to_gray(sub[y:y + h, x:x + w])


def real_vita_det(limit, rng):
    ex = excluded()
    rows = [json.loads(s) for s in DET.read_text().splitlines()] if DET.exists() else []
    rows = [r for r in rows if usable(r["shot"], ex)]
    rng.shuffle(rows)
    out = []
    for r in rows:
        p = DATA / "vndb" / "shots" / r["shot"]
        if not p.exists():
            continue
        rgb = None
        for q in r["quads"]:
            xs, ys = [int(v[0]) for v in q], [int(v[1]) for v in q]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            h, w = y1 - y0, x1 - x0
            if not (12 <= h <= 70 and w >= 2.5 * h and (y0 + y1) / 2 > 0.45 * r["size"][1]):
                continue
            if rgb is None:
                rgb = np.asarray(Image.open(p).convert("RGB"))
            g = grad_recut(rgb, (x0, y0, x1, y1))
            if g is not None:
                out.append((g, r["shot"]))
        if len(out) >= limit:
            break
    return out[:limit]


def real_psp_band(limit, rng):
    ex = excluded()
    shots = [p for p in sorted((DATA / "vndb_psp" / "shots").glob("*.jpg")) if usable(p.name, ex)]
    rng.shuffle(shots)
    out = []
    for p in shots:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        if rgb.shape[:2] != (272, 480):
            continue
        y0 = int(272 * 0.58)
        dets = find_lines(rgb[y0:], "grad")
        for x, y, w, h in dets or []:
            if 10 <= h <= 40 and w >= 3 * h:
                out.append((to_gray(rgb[y0 + y:y0 + y + h, x:x + w]), p.name))
        if len(out) >= limit:
            break
    return out[:limit]


def real_bright(limit, rng):
    ex = excluded()
    recs = [json.loads(s) for s in (DATA / "real" / "lines.jsonl").read_text().splitlines()]
    recs = [r for r in recs if r["split"] == "train" and usable(r["shot"], ex)]
    rng.shuffle(recs)
    out = []
    for r in recs[:limit]:
        p = DATA / "real" / "crops" / f"{r['id']:05d}.png"
        if p.exists():
            out.append((to_gray(np.asarray(Image.open(p).convert("RGB"))), r["shot"]))
    return out


def synth_crops(pattern, limit, rng):
    paths = sorted(glob.glob(pattern))
    out = []
    for p in paths:
        z = np.load(p)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        metas = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(shapes)
        idx = rng.permutation(len(shapes))[:max(1, limit // len(paths))]
        for i in idx:
            h, w = (int(v) for v in shapes[i])
            out.append((data[offs[i]:offs[i + 1]].reshape(h, w), metas[i]))
    return out[:limit]


def crop_stats(g):
    """Style statistics of one grey crop (finder geometry: ~4 px margin above, 5 below)."""
    g = g.astype(np.float32)
    h, w = g.shape
    bg = float(np.median(g))
    lo, hi = np.percentile(g, [2, 98])
    light = (hi - bg) > (bg - lo)
    text = hi if light else lo
    margin = np.concatenate([g[:3].ravel(), g[-3:].ravel()])
    return {"h": h, "aspect": w / h, "light_text": bool(light), "contrast": float(abs(text - bg)),
            "bg": bg, "busy": float(margin.std())}


def summarise(crops):
    st = [crop_stats(g) for g, _ in crops]
    if not st:
        return {}
    a = {k: np.array([s[k] for s in st], np.float64) for k in st[0]}

    def q(v):
        return [round(float(x), 1) for x in np.percentile(v, [10, 50, 90])]
    return {"n": len(st), "h_p10_50_90": q(a["h"]), "aspect_p10_50_90": q(a["aspect"]),
            "light_text_share": round(float(a["light_text"].mean()), 3),
            "contrast_p10_50_90": q(a["contrast"]), "bg_p10_50_90": q(a["bg"]),
            "busy_p10_50_90": q(a["busy"])}


def row_img(g, width, tag, font):
    g32 = Image.fromarray(g).resize((max(8, round(g.shape[1] * 32 / g.shape[0])), 32), Image.BILINEAR)
    row = Image.new("RGB", (width, 34), (0, 0, 0))
    row.paste(g32.convert("RGB").crop((0, 0, min(g32.width, width - 4), 32)), (0, 1))
    if tag:
        ImageDraw.Draw(row).text((width - 120, 20), tag[:18], font=font, fill=(255, 255, 0))
    return row


def sheets(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(sorted((DATA / "fonts" / "notosansjp").glob("*.ttf"))[0]), 11)
    reals = {"vita-det": real_vita_det(400, rng), "psp-band": real_psp_band(200, rng),
             "real-bright": real_bright(200, rng)}
    syn = []
    for spec in args.synth:
        name, pat = spec.split("=", 1) if "=" in spec else ("synth", spec)
        syn.append((name, synth_crops(pat, 600, rng)))
    n_rows = 37
    for name, crops in syn:
        for k, (rname, rc) in enumerate(reals.items()):
            if not rc:
                continue
            want = "psp" if rname == "psp-band" else None
            pool = [c for c in crops if want is None or c[1].get("platform") in ("psp",)] or crops
            sheet = Image.new("RGB", (1280, n_rows * 34), (30, 30, 30))
            for i in range(n_rows):
                g, src = rc[int(rng.integers(len(rc)))]
                sheet.paste(row_img(g, 636, src.split("_")[0], font), (0, 34 * i))
                g, m = pool[int(rng.integers(len(pool)))]
                tag = f"{m.get('platform', '')} {m.get('scheme', '')}"
                sheet.paste(row_img(g, 636, tag, font), (644, 34 * i))
            path = out / f"side_{rname}_vs_{name}.png"
            sheet.save(path)
            print("wrote", path, sheet.size, flush=True)


def stats(args):
    rng = np.random.default_rng(args.seed)
    res = {"vita-det": summarise(real_vita_det(1500, rng)), "psp-band": summarise(real_psp_band(800, rng)),
           "real-bright": summarise(real_bright(800, rng))}
    for spec in args.synth:
        name, pat = spec.split("=", 1) if "=" in spec else ("synth", spec)
        crops = synth_crops(pat, 4000, rng)
        res[name] = summarise(crops)
        by = defaultdict(list)
        for g, m in crops:
            by[m.get("platform", "?")].append((g, m))
        for plat, cs in sorted(by.items()):
            res[f"{name}:{plat}"] = summarise(cs)
        if crops and "scheme" in crops[0][1]:
            # the polarity estimator against the renderer's ground truth
            ok = [crop_stats(g)["light_text"] == (m["scheme"] in ("light_on_dark", "outlined_on_light",
                                                                    "on_colour")) for g, m in crops]
            res[name]["estimator_agrees_with_meta"] = round(float(np.mean(ok)), 3)
            for key in ("platform", "layout", "scheme", "cat", "name", "cursor", "partial", "ruby", "kind"):
                c = Counter(str(m.get(key)) for _, m in crops)
                res[name][f"share_{key}"] = {k: round(v / len(crops), 3) for k, v in c.most_common()}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))


def finders(args):
    import render
    cv2.setNumThreads(1)
    render.init(100_000)
    rng = np.random.default_rng(args.seed)
    agree = defaultdict(lambda: [0, 0])
    yield_ = defaultdict(lambda: Counter())
    n = 0
    while n < args.n:
        r = render.render_region(rng)
        if r is None:
            continue
        img, layout, ink, meta = r
        n += 1
        key = meta["scheme"]
        mask = render.extract(img, layout, ink, "mask") or []
        grad = render.extract(img, layout, ink, "grad") or []
        bright = render.extract(img, layout, ink, "legacy") or []
        yield_[key]["regions"] += 1
        yield_[key]["mask"] += len(mask)
        yield_[key]["grad"] += len(grad)
        yield_[key]["bright"] += len(bright)
        gb = [b for _, _, b, _, _ in grad]
        for _, _, (x, y, w, h), _, _ in mask:
            agree[key][1] += 1
            for gx, gy, gw, gh in gb:
                ix = max(0, min(x + w, gx + gw) - max(x, gx))
                iy = max(0, min(y + h, gy + gh) - max(y, gy))
                if ix * iy >= 0.5 * (w * h + gw * gh - ix * iy):
                    agree[key][0] += 1
                    break
    res = {}
    for key in sorted(yield_):
        y = yield_[key]
        res[key] = {"regions": y["regions"], "mask_lines": y["mask"],
                    "grad_lines_per_mask_line": round(y["grad"] / max(1, y["mask"]), 3),
                    "legacy_lines_per_mask_line": round(y["bright"] / max(1, y["mask"]), 3),
                    "mask_boxes_matched_by_grad_iou50": round(agree[key][0] / max(1, agree[key][1]), 3)}
    tot = {k: sum(r[k] if isinstance(r[k], int) else 0 for r in res.values()) for k in ("regions", "mask_lines")}
    res["all"] = tot | {"legacy_lines_per_mask_line": round(sum(yield_[k]["bright"] for k in yield_)
                                                            / max(1, tot["mask_lines"]), 3),
                        "grad_lines_per_mask_line": round(sum(yield_[k]["grad"] for k in yield_)
                                                          / max(1, tot["mask_lines"]), 3),
                        "mask_boxes_matched_by_grad_iou50": round(
                            sum(a for a, _ in agree.values()) / max(1, sum(b for _, b in agree.values())), 3)}
    print(json.dumps(res, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sheets", "stats", "finders"])
    ap.add_argument("--synth", action="append", default=[], help="[name=]glob of shards")
    ap.add_argument("--out", default="runs/calib")
    ap.add_argument("--json", default="")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"sheets": sheets, "stats": stats, "finders": finders}[args.mode](args)


if __name__ == "__main__":
    main()

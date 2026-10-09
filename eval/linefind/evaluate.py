#!/usr/bin/env python3
"""Score dialogue line finders on real regions: line/region recall and precision, and
optionally the whole-pipeline CER with a recognizer checkpoint.

  evaluate.py --manifest eval/linefind/manifests/research_b.jsonl [--manifest ...]
              [--finders legacy,grad,research,oracle] [--rec models/v1/ckpt_best.pt]
              [--shots DIR ...] [--json out.json] [--sheets DIR]

Finders:
  legacy    synth/common.py find_lines(finder="legacy"): the overlay's bright-pixel
            projection (vjo_ocr_find_lines), which rejects light boxes
  grad      synth/common.py find_lines(finder="grad"): the polarity-agnostic finder,
            bit-exact twin of runtime/vocr_lines.c
  c-grad, c-legacy  the C finder itself (runtime/vocr_lines.c through its CLI)
  research  the research campaign's gradient-energy row projection (float, full-width
            crops; bf653e98 scratch/real/grad_finder.py), ported here as a baseline
  oracle    the ground-truth line boxes padded like the finders pad (upper bound)
  mod:fn    any function rgb -> boxes, imported from module `mod` (prototyping)

Line matching is IoU >= 0.5, one to one (metrics.py). Regions where a finder
rejects the input count as finding nothing. Pipeline CER = sum of edit distances /
sum of reference lengths over all regions with a transcription, comparing the
concatenated line texts after NFKC and whitespace removal; a rejected region scores
its full length. Lines marked uncertain are kept in the reference (as in research).
"""
import argparse
import importlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "train"))
import metrics  # noqa: E402
from build_gt import SHOT_DIRS, read, region_rgb  # noqa: E402

from common import find_lines, line_polarity, to_gray  # noqa: E402


def research_grad(rgb):
    """Port of the research campaign's gradient-energy finder (float arithmetic)."""
    g = to_gray(rgb).astype(np.float32)
    e = np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]
    prof = np.convolve(e.sum(1), np.ones(3) / 3, mode="same")
    on = prof > 0.25 * prof.max()
    runs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        if not v and start is not None:
            runs.append([start, y])
            start = None
    if start is not None:
        runs.append([start, len(on)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 3:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] >= 8]
    if runs:
        med = float(np.median([r[1] - r[0] for r in runs]))
        runs = [r for r in runs if r[1] - r[0] >= 0.55 * med]
    H, W = g.shape
    return [(0, max(0, a - 4), W, min(H, b + 6) - max(0, a - 4)) for a, b in runs]


def get_finder(name):
    if name == "legacy":
        return lambda rgb: find_lines(rgb, finder="legacy")
    if name == "grad":
        return lambda rgb: find_lines(rgb, finder="grad")
    if name == "research":
        return research_grad
    if name in ("c-grad", "c-legacy"):
        import cfinder
        return cfinder.Finder(name[2:])
    if ":" in name:
        mod, fn = name.split(":")
        return getattr(importlib.import_module(mod), fn)
    raise SystemExit(f"unknown finder {name}")


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if not ch.isspace())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Recognizer:
    """v1-style CTC checkpoint (train/model.py): grey crop, height 32, greedy decode."""

    def __init__(self, path):
        import torch

        from model import TinyVNOCR
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab = ck["vocab"]
        self.model = TinyVNOCR(len(self.vocab) + 1, width=ck.get("width", 1.0))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.torch = torch

    def __call__(self, grey):
        import cv2
        h, w = grey.shape
        nw = max(8, min(1024, int(round(w * 32 / h))))
        g = cv2.resize(grey, (nw, 32), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        W = (nw + 7) // 8 * 8
        x = np.empty((1, 1, 32, W), np.float32)
        x[0, 0, :, :nw] = g
        x[0, 0, :, nw:] = g[:, -1:]
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(x / 127.5 - 1.0))
        best = logits.argmax(2)[:, 0].tolist()[:(nw + 7) // 8]
        out, last = [], 0
        for t in best:
            if t and t != last:
                out.append(self.vocab[t - 1])
            last = t
        return "".join(out)


def to_xyxy(boxes):
    return [(x, y, x + w, y + h) for (x, y, w, h) in boxes]


def gt_boxes(row):
    x0, y0 = row["region"][:2]
    keep, ign = [], []
    for ln in row["lines"]:
        if not ln.get("box"):
            continue
        b = ln["box"]
        bb = (b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0)
        (ign if ln.get("ignore") else keep).append(bb)
    return keep, ign


def oracle(row, rgb):
    H, W = rgb.shape[:2]
    keep, _ = gt_boxes(row)
    out = []
    for x0, y0, x1, y1 in keep:
        a, b = max(0, x0), max(0, y0)
        out.append((a, b, min(W, x1) - a, min(H, y1) - b))
    return out


def evaluate(rows, finders, shots, rec=None, sheets=None, norm_pol=False):
    res = {}
    imgs = [region_rgb(r, shots) for r in rows]
    for name in finders:
        f = None if name == "oracle" else get_finder(name)
        per, errs, chars, t_total, out_rows = [], 0, 0, 0.0, []
        for r, rgb in zip(rows, imgs):
            t0 = time.perf_counter()
            boxes = oracle(r, rgb) if f is None else f(rgb)
            t_total += time.perf_counter() - t0
            keep, ign = gt_boxes(r)
            m = metrics.match(to_xyxy(boxes or []), keep, ign)
            m["rejected"] = boxes is None
            per.append(m)
            o = {"id": r["id"], "found": None if boxes is None else len(boxes), "gt": len(keep),
                 "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "boxes": boxes}
            texts = [ln["text"] for ln in r["lines"] if ln.get("text") and not ln.get("ignore")]
            if rec is not None and texts:
                grey = to_gray(rgb)
                crops = [grey[y:y + h, x:x + w] for (x, y, w, h) in (boxes or []) if w > 0 and h > 0]
                if norm_pol:
                    crops = [255 - c if line_polarity(c) else c for c in crops]
                pred = "".join(rec(c) for c in crops)
                ref = norm("".join(texts))
                e = lev(norm(pred), ref)
                errs += e
                chars += len(ref)
                o.update(pred=pred, cer=round(e / max(1, len(ref)), 4))
            out_rows.append(o)
        s = metrics.summarize(per)
        s["rejected"] = sum(m["rejected"] for m in per)
        s["ms_per_region"] = round(1000 * t_total / max(1, len(rows)), 2)
        if rec is not None and chars:
            s["pipeline_cer"] = errs / chars
            s["ref_chars"] = chars
        res[name] = {"summary": s, "rows": out_rows}
        if sheets is not None:
            draw_sheets(rows, imgs, out_rows, Path(sheets) / name)
    return res


def draw_sheets(rows, imgs, out_rows, out):
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for r, rgb, o in zip(rows, imgs, out_rows):
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        keep, ign = gt_boxes(r)
        for b in keep:
            d.rectangle((b[0], b[1], b[2] - 1, b[3] - 1), outline=(255, 0, 255))
        for (x, y, w, h) in o["boxes"] or []:
            d.rectangle((x, y, x + w - 1, y + h - 1), outline=(0, 255, 0))
        ok = o["fp"] == 0 and o["fn"] == 0
        tiles.append((f"{r['id']} gt={o['gt']} found={o['found']} {'OK' if ok else 'ERR'} {o.get('cer', '')}", im))
    page, cur, h = 0, [], 0
    for t in tiles + [None]:
        if t is None or (cur and h + t[1].height + 14 > 1250):
            if cur:
                W = min(1280, max(im.width for _, im in cur))
                s = Image.new("RGB", (W, sum(im.height + 14 for _, im in cur)), (40, 40, 40))
                d = ImageDraw.Draw(s)
                y = 0
                for name, im in cur:
                    d.text((2, y), name, fill=(255, 255, 0))
                    s.paste(im, (0, y + 12))
                    y += im.height + 14
                s.save(out / f"sheet_{page:02d}.png")
            page, cur, h = page + 1, [], 0
            if t is None:
                break
        cur.append(t)
        h += t[1].height + 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--finders", default="legacy,grad,research,oracle")
    ap.add_argument("--rec", default="")
    ap.add_argument("--shots", action="append", default=[])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--sheets")
    ap.add_argument("--norm", action="store_true", help="invert dark-text crops before recognition")
    args = ap.parse_args()
    shots = args.shots + [str(d) for d in SHOT_DIRS]
    rec = Recognizer(args.rec) if args.rec else None
    report = {}
    for man in args.manifest:
        rows = read(man)
        res = evaluate(rows, args.finders.split(","), shots, rec,
                       None if not args.sheets else Path(args.sheets) / Path(man).stem, args.norm)
        report[Path(man).stem] = res
        print(f"\n## {Path(man).stem}: {len(rows)} regions")
        print("| finder | line R | line P | F1 | region exact | merged | split | cut | rejected | CER | ms |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, v in res.items():
            s = v["summary"]
            cer = f"{100 * s['pipeline_cer']:.1f}%" if "pipeline_cer" in s else "-"
            print(f"| {name} | {s['line_recall']:.3f} | {s['line_precision']:.3f} | {s['line_f1']:.3f} | "
                  f"{s['region_exact']:.3f} | {s['merged']} | {s['split']} | {s['cut']} | {s['rejected']} | "
                  f"{cer} | {s['ms_per_region']} |")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

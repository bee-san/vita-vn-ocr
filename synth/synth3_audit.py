#!/usr/bin/env python3
"""Label audit of synthetic shards: how often PP-OCRv6 small (the teacher) reads a crop as its label.

  synth3_audit.py --shards 'S3/shards/train/synth3_000[0-9]?.npz' --n 2000 --out audit.json --sheet sheet.png

A random sample (seeded) of crops is read by PP-OCRv6 small through eval/backends.py (the pinned
ONNX file), in screen polarity (console-polarity crops are inverted back first). The report gives
the teacher-vs-label CER and exact share per stratum, and a contact sheet of the crops where they
disagree most, for reading by eye. Disagreement is not label error: the teacher misreads hard
renders too (tiny PSP text, heavy outlines, rare kanji, pixel fonts). It bounds the label-noise
rate and finds systematic rendering faults (a blank glyph under a label, cut ink), which show as
clusters on the sheet. Needs onnxruntime (the shared venv), not the CI env.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(HERE))
from metrics import lev, norm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet")
    ap.add_argument("--font", default="/local/home/skerraut/herd-vita-ocr/shared/data/fonts/notosansjp/NotoSansJP[wght].ttf")
    args = ap.parse_args()
    import backends
    paths = sorted(glob.glob(args.shards))
    rng = np.random.default_rng(args.seed)
    index = []
    for p in paths:
        z = np.load(p)
        index += [(p, i) for i in range(len(z["shapes"]))]
    pick = sorted(rng.choice(len(index), min(args.n, len(index)), replace=False))
    by_file = defaultdict(list)
    for k in pick:
        by_file[index[k][0]].append(index[k][1])
    crops, rows = [], []
    for p, ids in by_file.items():
        z = np.load(p)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        labels = json.loads(str(z["labels"]))
        meta = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(labels)
        for i in ids:
            h, w = (int(v) for v in shapes[i])
            g = data[offs[i]:offs[i + 1]].reshape(h, w)
            m = meta[i]
            if m.get("inverted"):
                g = 255 - g
            crops.append(np.repeat(g[..., None], 3, axis=2))
            rows.append({"file": Path(p).name, "i": int(i), "label": labels[i], "meta": m})
    rec = backends.make_recognizer("ppocr-v6-small")
    texts, _ = backends.timed_read(rec, crops)
    strata = defaultdict(lambda: [0, 0, 0, 0])
    for r, t in zip(rows, texts):
        ref, pred = norm(r["label"]), norm(t)
        e = lev(pred, ref)
        r.update(pred=pred, errors=e, chars=len(ref))
        m = r["meta"]
        for k in ("all", f"platform:{m.get('platform')}", f"crop:{m.get('crop')}", f"cat:{m.get('cat')}",
                  f"raster:{str(m.get('raster', ''))[:5]}", f"kind:{m.get('kind')}", f"layout:{m.get('layout')}"):
            s = strata[k]
            s[0] += e
            s[1] += len(ref)
            s[2] += 1
            s[3] += e == 0
    out = {"shards": args.shards, "files": len(paths), "sample": len(rows), "seed": args.seed,
           "teacher": rec.info.get("sha256"),
           "strata": {k: {"cer": round(v[0] / max(v[1], 1), 4), "exact": round(v[3] / max(v[2], 1), 4), "lines": v[2]}
                      for k, v in sorted(strata.items())},
           "worst": [{k: r[k] for k in ("file", "i", "label", "pred", "errors")}
                     for r in sorted(rows, key=lambda r: -r["errors"] / max(r["chars"], 1))[:200]]}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["strata"]["all"]), flush=True)
    if args.sheet:
        font = ImageFont.truetype(args.font, 15)
        worst = sorted(range(len(rows)), key=lambda k: -rows[k]["errors"] / max(rows[k]["chars"], 1))[:36]
        sheet = Image.new("RGB", (1280, 34 * len(worst)), (0, 0, 0))
        for j, k in enumerate(worst):
            g = crops[k][..., 0]
            im = Image.fromarray(g).resize((max(8, round(g.shape[1] * 32 / g.shape[0])), 32), Image.BILINEAR)
            sheet.paste(im.crop((0, 0, min(im.width, 640), 32)).convert("RGB"), (0, 34 * j))
            d = ImageDraw.Draw(sheet)
            d.text((650, 34 * j + 1), "L " + rows[k]["label"][:30], font=font, fill=(160, 255, 160))
            d.text((650, 34 * j + 17), "T " + rows[k]["pred"][:30], font=font, fill=(255, 200, 120))
        sheet.save(args.sheet)


if __name__ == "__main__":
    main()

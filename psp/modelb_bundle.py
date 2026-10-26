#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): the console test bundle, with no game data in it.

  modelb_bundle.py --blob B.vocr --out DIR --eboot lines=E.PBP --eboot lines_vfpu=V.PBP --eboot page=E.PBP
                   --eboot page_vfpu=V.PBP --eboot probe64=E64.PBP [--mode w8f] [--head-gate on] [--two-level]
                   [--tile T] [--head H] [--font FONT.ttf]

The test inputs are rendered here: lines of this repository's own CI corpus (ci/fixtures/corpus, written for this
project) in M+ 1p (SIL OFL 1.1, ci/fonts.py) at native PSP sizes (14-17 px) over procedural dialogue-box backgrounds
(no screenshot, no game font, no game text):
  lines.bin    LINES single-line crops, ink box + 2 px, grey
  regions.bin  REGIONS 480 x 90 (and smaller) dialogue regions of 2 or 3 lines, for the app's finder mode
DIR/PSP/GAME/<folder>/ per --eboot (EBOOT.PBP, model.vocr, the pack, expect.txt from the host runtime, vocr.cfg):
  lines, lines_vfpu   the line crops, timed (repeat 3), the model's shipped options
  page, page_vfpu     the regions through the line finder (GRAD) + refinement, then every line (the page path)
  probe64             the 64 MB build (PSP_LARGE_MEMORY): the lines once, plus every memory partition's largest
                      free block (probe=1)
Each app compares every line with the host's text and logits (expect.txt) and writes results.txt and stats.json.
The bundle holds no copyrighted game data: it may be published (the release's console test asset).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "runtime")]
import pack  # noqa: E402

LINES, REGIONS, SEED = 150, 40, 20261026
FOLDERS = {"lines": ("MODELB_L", -1), "lines_vfpu": ("MODELB_LV", -1), "page": ("MODELB_P", 0),
           "page_vfpu": ("MODELB_PV", 0), "probe64": ("MODELB_64", -1)}


def corpus():
    out = []
    for f in sorted((ROOT / "ci" / "fixtures" / "corpus").glob("*.txt")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            while len(ln) > 25:                          # PSP lines: at most about 26 characters at 16 px
                cut = max(ln.rfind("、", 0, 22), ln.rfind("。", 0, 22))
                cut = cut + 1 if cut >= 8 else 22
                out.append(ln[:cut])
                ln = ln[cut:]
            if ln:
                out.append(ln)
    return out


def background(rng, h, w):
    """A procedural dialogue box: a dark (or, a quarter of the time, light) vertical gradient, noise, a frame line."""
    light = rng.random() < 0.25
    top, bot = (rng.uniform(170, 235), rng.uniform(150, 225)) if light else (rng.uniform(10, 70), rng.uniform(0, 50))
    g = np.linspace(top, bot, h)[:, None] * np.ones((1, w)) + rng.normal(0, rng.uniform(1, 6), (h, w))
    if rng.random() < 0.4:
        y = int(rng.integers(0, 2))
        g[y, :] = 255 - g[y, :].mean()
    return np.clip(g, 0, 255).astype(np.uint8), light


def draw(canvas, light, text, x, y, font, rng):
    from PIL import Image, ImageDraw
    im = Image.fromarray(canvas, "L")
    d = ImageDraw.Draw(im)
    fill = int(rng.uniform(0, 50)) if light else int(rng.uniform(205, 255))
    stroke = int(rng.integers(0, 2)) if not light else 0
    d.text((x, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=0 if not light else None)
    box = d.textbbox((x, y), text, font=font, stroke_width=stroke)
    return np.asarray(im), box


def render(font_path, rng):
    from PIL import ImageFont
    texts = corpus()
    lines, regions = [], []
    for i in range(LINES):
        t = texts[int(rng.integers(len(texts)))]
        size = int(rng.integers(14, 18))
        font = ImageFont.truetype(str(font_path), size)
        w = int(font.getlength(t)) + 16
        bg, light = background(rng, size + 12, w)
        img, (x0, y0, x1, y1) = draw(bg, light, t, 6, 4, font, rng)
        lines.append(np.ascontiguousarray(img[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]))
    for i in range(REGIONS):
        size = int(rng.integers(14, 18))
        font = ImageFont.truetype(str(font_path), size)
        n = int(rng.integers(2, 4))
        pitch = size + int(rng.integers(5, 9))
        h = min(90, 10 + n * pitch) if i % 2 else 90
        w = 480 if i % 2 == 0 else int(rng.integers(300, 481))
        bg, light = background(rng, h, w)
        img = bg
        for k in range(n):
            t = texts[int(rng.integers(len(texts)))]
            img, _ = draw(img, light, t, 8, 5 + k * pitch, font, rng)
        regions.append(np.ascontiguousarray(img))
    return lines, regions


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eboot", action="append", required=True, help="KIND=PATH, KIND one of " + ", ".join(FOLDERS))
    ap.add_argument("--mode", default="w8f", choices=["f32", "w8f", "w8i"])
    ap.add_argument("--head-gate", default="off", choices=["off", "on"])
    ap.add_argument("--two-level", action="store_true")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--font", default="")
    a = ap.parse_args()
    font = a.font or subprocess.run([sys.executable, str(ROOT / "ci" / "fonts.py")], check=True, capture_output=True,
                                    text=True).stdout.strip().splitlines()[-1]
    rng = np.random.default_rng(SEED)
    lines, regions = render(font, rng)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    src = out / "src"
    for name, crops in (("lines", lines), ("regions", regions)):
        (src / name).mkdir(parents=True, exist_ok=True)
        pack.write_pack(src / name / "lines.bin", crops)
    manifest = {"kind": "modelb-console-test-bundle", "blob": Path(a.blob).name,
                "blob_sha256": hashlib.sha256(Path(a.blob).read_bytes()).hexdigest(),
                "inputs": {"text": "ci/fixtures/corpus (this repository)", "font": "M+ 1p Regular (SIL OFL 1.1)",
                           "lines": len(lines), "regions": len(regions), "seed": SEED},
                "options": {"mode": a.mode, "head_gate": a.head_gate, "two_level": bool(a.two_level), "tile": a.tile,
                            "head": a.head}, "apps": {}}
    for spec in a.eboot:
        kind, eboot = spec.split("=", 1)
        folder, finder = FOLDERS[kind]
        p = src / ("regions" if finder >= 0 else "lines")
        args = ["bundle", "--blob", a.blob, "--mode", a.mode, "--pack", str(p), "--out", str(out), "--eboot", eboot,
                "--name", folder, "--tile", str(a.tile), "--head", str(a.head), "--head-gate", a.head_gate,
                "--finder", str(finder), "--repeat", "1" if kind == "probe64" else "3"]
        if a.two_level:
            args.append("--two-level")
        if kind == "probe64":
            args.append("--probe")
        subprocess.run([sys.executable, str(HERE / "pack.py"), *args], check=True, cwd=ROOT, capture_output=True)
        app = out / "PSP" / "GAME" / folder
        manifest["apps"][folder] = {"kind": kind, "eboot_sha256": hashlib.sha256(Path(eboot).read_bytes()).hexdigest(),
                                    "finder": finder, "items": len(regions) if finder >= 0 else len(lines),
                                    "vocr_cfg": (app / "vocr.cfg").read_text()}
    shutil.rmtree(src)
    for stray in out.rglob("vocr_lines_refine_test"):
        stray.unlink()
    (out / "bundle.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "apps"} | {"apps": list(manifest["apps"])}))


if __name__ == "__main__":
    main()

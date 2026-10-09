#!/usr/bin/env python3
"""Write the test bundle the Vita app reads from ux0:data/vocr/ (vita/app/main.c).

  bundle.py --out DIR [--profile quick|full] [--ckpt CKPT] [--modes w8i,w8f,f32]
            [--sets synth-val,vita-test,psp-test,regions,psp-regions] [--eval-data DIR] [--no-font]

DIR gets:
  cases.bin     the cases (line crops and dialogue regions) with the expected results
  cases.json    the same cases for vita/compare.py: ids, ground truth, styles, expectations
  *.vocr        the weights blobs (train/export.py, from --ckpt)
  font.bin      16-px bitmap font for the console screen (from the OFL font M+ 1p, ci/fonts.py),
                with font-OFL.txt beside it
  config.txt    the app's settings (clock, repeats, ...), README.txt

Every expected result comes from the host build of the same C code: runtime/vocr.c through
runtime/pyvocr.py (libvocr.so) and runtime/vocr_lines.c through a ctypes build of it, so the
console has to reproduce each one bit for bit: the same text and the same CRC-32 (zlib) of
the logits, and for regions the same line boxes. Line crops of real screenshots carry the
console's polarity decision (vocr_line_dark over the crop) and run with VOCR_INVERT when
the text is dark, as the line finder's VocrLine.dark makes the overlay do.

The bundle holds crops of copyrighted game screenshots: keep it local; never commit or
publish it (AGENTS.md). Rebuild it from eval/fetch.py's screenshots instead.

cases.bin format (little-endian u32 unless noted; every record padded to 4 bytes):
  header, 64 bytes: "VOCRCASE", version 1, n_models, n_cases, max_case_bytes, max_lines,
                    max_frames, n_classes, max_region_w, max_region_h, 0 x 5
  per model, 96 bytes: name[16], file[64] (NUL-padded), mode (1 f32, 2 w8f, 3 w8i),
                    blob bytes, blob CRC-32, host arena bytes
  per case: u32 record bytes (this field included), kind (0 line, 1 region), flags
            (1 dark, 2 polarity from vocr_line_dark), width, height, channels (1 or 4),
            id length, ground-truth length, id, ground truth (UTF-8; regions: lines joined
            by \\n), pad; then per model: text length, logits CRC-32, frames, line count,
            line count x (x, y, w, h, dark), text, pad; then the pixels, pad.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "runtime"), str(ROOT / "train"), str(ROOT / "eval"), str(ROOT / "synth"),
                str(ROOT / "ci")]

MODES = {"f32": 1, "w8f": 2, "w8i": 3}
KIND_LINE, KIND_REGION = 0, 1
FLAG_DARK, FLAG_AUTO = 1, 2
LINES_MAX = 16
MAGIC = b"VOCRCASE"
PROFILES = {   # per set: number of cases (None = all); regions / psp-regions: dialogue regions
    "quick": {"synth-val": 8, "vita-test": 24, "psp-test": 8, "regions": 8, "psp-regions": 4},
    "full": {"synth-val": 100, "vita-test": None, "psp-test": None, "regions": None, "psp-regions": None},
}
REGION_SETS = {"regions": "vita-test", "psp-regions": "psp-test"}
# the v1 checkpoint (release v0.1.0-baseline): models/v1/ like runtime/README.md, or the fleet's copy
DEFAULT_CKPT = (Path(os.environ["HERD"]) / "shared/models/v1/ckpt_best.pt" if os.environ.get("HERD")
                else ROOT / "models/v1/ckpt_best.pt")


def pad4(b):
    return b + b"\0" * (-len(b) % 4)


# ---------------------------------------------------------------- the C line finder, host build

class _Img(C.Structure):
    _fields_ = [("pixels", C.c_void_p), ("width", C.c_uint), ("height", C.c_uint),
                ("stride", C.c_size_t), ("bpp", C.c_uint)]


class _Line(C.Structure):
    _fields_ = [("x", C.c_uint), ("y", C.c_uint), ("w", C.c_uint), ("h", C.c_uint), ("dark", C.c_uint)]


class _Set(C.Structure):
    _fields_ = [("lines", _Line * LINES_MAX), ("count", C.c_uint), ("truncated", C.c_uint)]


class Lines:
    """runtime/vocr_lines.c built for the host as a shared library (build/vita-host/)."""

    def __init__(self, cc=None):
        src = ROOT / "runtime" / "vocr_lines.c"
        so = ROOT / "build" / "vita-host" / "libvocr_lines.so"
        newest = max(src.stat().st_mtime, (ROOT / "runtime" / "vocr_lines.h").stat().st_mtime)
        if not so.exists() or so.stat().st_mtime < newest:
            so.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([cc or os.environ.get("CC", "cc"), "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
        self.lib = C.CDLL(str(so))
        self.lib.vocr_lines_arena_bytes.argtypes = [C.c_uint, C.c_uint]
        self.lib.vocr_lines_arena_bytes.restype = C.c_size_t
        self.lib.vocr_find_lines.argtypes = [C.POINTER(_Img), C.c_int, C.c_void_p, C.c_size_t, C.POINTER(_Set)]
        self.lib.vocr_line_dark.argtypes = [C.POINTER(_Img), C.c_uint, C.c_uint, C.c_uint, C.c_uint]

    @staticmethod
    def _img(px):
        px = np.ascontiguousarray(px, dtype=np.uint8)
        bpp = 1 if px.ndim == 2 else px.shape[2]
        return px, _Img(px.ctypes.data, px.shape[1], px.shape[0], px.strides[0], bpp)

    def dark(self, px):
        """vocr_line_dark over the whole crop."""
        keep, im = self._img(px)
        rc = self.lib.vocr_line_dark(C.byref(im), 0, 0, im.width, im.height)
        del keep
        if rc < 0:
            raise RuntimeError(f"vocr_line_dark: {rc}")
        return rc

    def find(self, rgba):
        """vocr_find_lines (GRAD) on an RGBA region: [(x, y, w, h, dark)]."""
        keep, im = self._img(rgba)
        n = self.lib.vocr_lines_arena_bytes(im.width, im.height)
        arena = C.create_string_buffer(n)
        out = _Set()
        rc = self.lib.vocr_find_lines(C.byref(im), 0, arena, n, C.byref(out))
        del keep
        if rc:
            raise RuntimeError(f"vocr_find_lines: {rc}")
        return [(ln.x, ln.y, ln.w, ln.h, ln.dark) for ln in out.lines[:out.count]]


# ---------------------------------------------------------------- cases

def rgba(rgb):
    return np.ascontiguousarray(np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2))


def spread(items, n, key):
    """At most n items, round-robin over key (so every game shows up early)."""
    if n is None or n >= len(items):
        return items
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    out, queues = [], list(groups.values())
    while len(out) < n:
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return sorted(out, key=items.index)


def line_cases(name, n, data_dir, lines_lib):
    import sets
    if name == "synth-val":
        rows = sets.load_synth_val(limit=n or sets.SYNTH_VAL_LINES)
        return [dict(id=ln.id, set=name, kind=KIND_LINE, flags=0, px=np.ascontiguousarray(ln.crop_rgb[..., 0]),
                     gt=ln.text, styles=ln.styles, vn=None) for ln in rows]
    if not (sets.MANIFESTS / f"{name}.jsonl").exists():
        print(f"{name}: no manifest on this branch, skipped", file=sys.stderr)
        return []
    rows, _ = sets.load(name, data_dir)
    rows = spread(rows, n, key=lambda ln: ln.vn)
    out = []
    for ln in rows:
        px = rgba(ln.crop_rgb)
        dark = lines_lib.dark(px)
        out.append(dict(id=ln.id, set=name, kind=KIND_LINE, flags=FLAG_AUTO | (FLAG_DARK if dark else 0), px=px,
                        gt=ln.text, styles=ln.styles, vn=ln.vn))
    return out


def region_cases(name, n, data_dir):
    import sets
    from PIL import Image
    if not (sets.MANIFESTS / f"{name}.jsonl").exists():
        print(f"{name}: no manifest on this branch, regions skipped", file=sys.stderr)
        return []
    regs = sets.regions(name, data_dir)
    regs = spread(regs, n, key=lambda r: r[2][0]["vn"])
    out = []
    for path, (rx, ry, rw, rh), rows in regs:
        img = np.asarray(Image.open(path).convert("RGB"))
        px = rgba(img[ry:ry + rh, rx:rx + rw])
        rows = sorted(rows, key=lambda r: (r["box"][1], r["box"][0]))
        out.append(dict(id=f"{rows[0]['id']}@region", set=name + ":regions", kind=KIND_REGION, flags=0, px=px,
                        gt="\n".join(r["text"] for r in rows if not r.get("uncertain")),
                        styles=sorted({s for r in rows for s in r.get("styles") or []}), vn=rows[0]["vn"],
                        region=[rx, ry, rw, rh], image=Path(path).name,
                        rows=[dict(id=r["id"], box=[r["box"][0] - rx, r["box"][1] - ry, r["box"][2], r["box"][3]],
                                   text=r["text"], uncertain=bool(r.get("uncertain"))) for r in rows]))
    return out


# ---------------------------------------------------------------- expected results (host runtime)

def expect(case, vocr, lines_lib, n_classes):
    """What the host build of the runtime gives for one case and one model."""
    def run(px, invert):
        text, logits = vocr.run(px, logits=True, invert=bool(invert))
        assert logits.shape[1] == n_classes
        return text, np.ascontiguousarray(logits, dtype="<f4").tobytes(), logits.shape[0]

    if case["kind"] == KIND_LINE:
        text, raw, frames = run(case["px"], case["flags"] & FLAG_DARK)
        return dict(text=text, crc=zlib.crc32(raw) & 0xFFFFFFFF, frames=frames, boxes=[], max_line_frames=frames)
    boxes = lines_lib.find(case["px"])
    crc, frames, texts, most = 0, 0, [], 0
    for x, y, w, h, dark in boxes:
        text, raw, f = run(case["px"][y:y + h, x:x + w], dark)
        crc = zlib.crc32(raw, crc) & 0xFFFFFFFF
        frames += f
        most = max(most, f)
        texts.append(text)
    return dict(text="\n".join(texts), crc=crc, frames=frames, boxes=[list(b) for b in boxes], max_line_frames=most)


def encode_case(case, exps, model_names):
    px = np.ascontiguousarray(case["px"])
    h, w = px.shape[:2]
    ch = 1 if px.ndim == 2 else px.shape[2]
    ident, gt = case["id"].encode(), case["gt"].encode()
    body = struct.pack("<7I", case["kind"], case["flags"], w, h, ch, len(ident), len(gt)) + pad4(ident + gt)
    for name in model_names:
        e = exps[name]
        t = e["text"].encode()
        body += struct.pack("<4I", len(t), e["crc"], e["frames"], len(e["boxes"]))
        body += b"".join(struct.pack("<5I", *b) for b in e["boxes"])
        body += pad4(t)
    body += pad4(px.tobytes())
    return struct.pack("<I", 4 + len(body)) + body


def write_cases_bin(path, models, cases, exps, n_classes):
    """cases.bin from the models (dicts with name, file, mode, bytes, crc, arena), the cases
    and their expectations (one {model name: expect()} per case). Returns the header sizes."""
    names = [m["name"] for m in models]
    records = [encode_case(c, e, names) for c, e in zip(cases, exps)]
    max_frames = max([1] + [x["max_line_frames"] for e in exps for x in e.values()])  # longest single line
    max_lines = max([0] + [len(x["boxes"]) for e in exps for x in e.values()])
    regions = [c["px"].shape for c in cases if c["kind"] == KIND_REGION]
    max_rh, max_rw = max([0] + [h for h, _, _ in regions]), max([0] + [w for _, w, _ in regions])
    header = MAGIC + struct.pack("<11I", 1, len(models), len(records), max(len(r) for r in records), max_lines,
                                 max_frames, n_classes, max_rw, max_rh, 0, 0)
    header += b"\0" * (64 - len(header))
    mrec = b"".join(m["name"].encode().ljust(16, b"\0") + m["file"].encode().ljust(64, b"\0")
                    + struct.pack("<4I", m["mode"], m["bytes"], m["crc"], m["arena"]) for m in models)
    Path(path).write_bytes(header + mrec + b"".join(records))
    return dict(max_frames=max_frames, max_lines=max_lines, max_region=(max_rw, max_rh))


# ---------------------------------------------------------------- the screen font

def write_font(out, texts, vocab):
    """font.bin (see vita/app/screen.c) from M+ 1p, the pinned OFL font of ci/fonts.py."""
    from fontTools.ttLib import TTFont
    from PIL import Image, ImageDraw, ImageFont

    import fonts
    path = fonts.ensure_font()
    cmap = TTFont(str(path)).getBestCmap()
    chars = {chr(c) for c in range(0x20, 0x7F)} | set("".join(vocab)) | set("".join(texts)) | set("…「」『』！？")
    cps = sorted(ord(c) for c in chars if ord(c) in cmap and not c.isspace() or c == " ")
    face = ImageFont.truetype(str(path), 15)
    entries, bitmaps = [], b""
    base = 16 + 12 * len(cps)
    for cp in cps:
        half = cp < 0x100 or 0xFF61 <= cp <= 0xFF9F
        w = 8 if half else 16
        img = Image.new("L", (w, 16), 0)
        ImageDraw.Draw(img).text((w / 2, 13), chr(cp), font=face, fill=255, anchor="ms")
        bits = np.packbits(np.asarray(img) >= 96, axis=1)
        entries.append(struct.pack("<IHHI", cp, w, 0, base + len(bitmaps)))
        bitmaps += bits.tobytes()
    (out / "font.bin").write_bytes(b"VFNT" + struct.pack("<3I", 1, len(cps), 16) + b"".join(entries) + bitmaps)
    (out / "font-OFL.txt").write_bytes((path.parent / "OFL.txt").read_bytes())
    return len(cps), str(path)


# ---------------------------------------------------------------- main

def export_blobs(ckpt, out, stem):
    import export
    model, vocab, height = export.load_checkpoint(ckpt)
    sha = hashlib.sha256(Path(ckpt).read_bytes()).digest()
    files = {}
    for key, int8 in (("f32", False), ("w8", True)):
        data, _ = export.build_blob(export.model_ops(model, height), vocab, height, int8, source_sha=sha)
        files[key] = f"{stem}_{key}.vocr"
        (out / files[key]).write_bytes(data)
    return files, vocab, sha.hex()


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


CONFIG = """# vocr_test settings (key=value; lines starting with # are ignored)
# ARM clock in MHz for the run: 444 (what the overlay's OCR job asks for), 333 or 0 (leave it)
clock=444
# timed runs per case and model (the first, untimed run checks the logits)
repeat={repeat}
# 1: exit when done (the Vita3K run); 0: wait for START
autoexit={autoexit}
# 1: also read the ARM performance counters through ScePerf, where the firmware allows it
pmu=1
# worker thread: stack in KiB (the overlay's job stack is 64) and CPU (0-2; -1 any)
stack_kb=64
cpu=0
# only run cases whose id contains this text, and at most this many (0 = all)
only=
limit=0
# models to run (comma list of names from cases.json; empty = all)
models=
"""

README = """vita-vn-ocr Vita test bundle ({profile}): copy this folder to ux0:data/vocr/ on the Vita.
Made by vita/bundle.py at git {git} on {date}. Install vocr_test.vpk (vita/README.md), run it,
then copy ux0:data/vocr/results.tsv back and run:  python vita/compare.py --bundle <this dir> results.tsv
It holds crops of copyrighted game screenshots: keep it private, never commit or share it.
font.bin is a bitmap rendering of M+ 1p (SIL Open Font License 1.1, font-OFL.txt).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                    help="train.py checkpoint (default: $HERD/shared/models/v1 or models/v1)")
    ap.add_argument("--stem", default="v1", help="blob file names: STEM_f32.vocr, STEM_w8.vocr")
    ap.add_argument("--modes", default="w8i,w8f,f32")
    ap.add_argument("--sets", default="synth-val,vita-test,psp-test,regions,psp-regions")
    ap.add_argument("--eval-data", type=Path, default=None,
                    help="eval screenshots (default $VOCR_EVAL_DATA, data/eval)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--autoexit", type=int, default=0)
    ap.add_argument("--no-font", action="store_true")
    args = ap.parse_args()
    import fetch
    from pyvocr import Vocr
    t0 = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data_dir = fetch.data_dir(args.eval_data)
    files, vocab, ckpt_sha = export_blobs(args.ckpt, out, args.stem)
    lines_lib = Lines()
    prof = PROFILES[args.profile]
    cases = []
    for s in args.sets.split(","):
        if s in REGION_SETS:
            cases += region_cases(REGION_SETS[s], prof[s], data_dir)
        else:
            cases += line_cases(s, prof[s], data_dir, lines_lib)
    modes = args.modes.split(",")
    models = []
    for m in modes:
        f = files["f32" if m == "f32" else "w8"]
        v = Vocr(out / f, mode=m)
        blob = (out / f).read_bytes()
        models.append(dict(name=m, file=f, mode=MODES[m], bytes=len(blob), crc=zlib.crc32(blob) & 0xFFFFFFFF,
                           arena=v.arena_bytes, sha256=hashlib.sha256(blob).hexdigest(), vocr=v))
    n_classes = models[0]["vocr"].n_classes
    exps = [{m["name"]: expect(case, m["vocr"], lines_lib, n_classes) for m in models} for case in cases]
    sizes = write_cases_bin(out / "cases.bin", models, cases, exps, n_classes)
    meta_cases = []
    for i, (case, e) in enumerate(zip(cases, exps)):
        meta = {k: v for k, v in case.items() if k != "px"}
        meta.update(index=i, width=int(case["px"].shape[1]), height=int(case["px"].shape[0]),
                    channels=1 if case["px"].ndim == 2 else int(case["px"].shape[2]), expected=e)
        meta_cases.append(meta)
    max_frames = sizes["max_frames"]
    font = None
    if not args.no_font:
        n_glyphs, font_path = write_font(out, [c["gt"] for c in cases] +
                                         [e["text"] for c in meta_cases for e in c["expected"].values()], vocab)
        font = dict(glyphs=n_glyphs, source=Path(font_path).name, commit=__import__("fonts").COMMIT)
    (out / "config.txt").write_text(CONFIG.format(repeat=args.repeat, autoexit=args.autoexit))
    date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "README.txt").write_text(README.format(profile=args.profile, git=git_sha(), date=date))
    manifest = dict(profile=args.profile, git=git_sha(), created=date, ckpt=str(args.ckpt), ckpt_sha256=ckpt_sha,
                    eval_data=str(data_dir), font=font, n_classes=n_classes, max_frames=max_frames,
                    models=[{k: v for k, v in m.items() if k != "vocr"} for m in models],
                    cases=meta_cases)
    (out / "cases.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    kinds = {}
    for c in cases:
        kinds[c["set"]] = kinds.get(c["set"], 0) + 1
    print(f"{out}: {len(cases)} cases {kinds}, models {modes}, cases.bin "
          f"{(out / 'cases.bin').stat().st_size:,} B, {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

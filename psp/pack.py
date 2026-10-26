#!/usr/bin/env python3
"""Crop packs for the PSP test app, the host runtime's expected outputs, and app bundles.

  pack.py build --set psp-test [--set vita-test --set synth-val] [--n N] --out DIR
      DIR/lines.bin    the crops in vocr_cli's pack format (runtime/tools/vocr_cli.c): u32 count,
                       then per crop u32 height, u32 width, u32 channels, the pixels. Real sets
                       are RGBA (eval/sets.py's GT-box crops, as runtime/tests/parity.py feeds
                       them), synth-val is grey, so both input paths run.
      DIR/lines.jsonl  per crop: id, set, ground truth, style tags
  pack.py expect --blob B.vocr --mode w8f --pack DIR/lines.bin --out DIR/expect_w8f.txt
      per crop: "<CRC-32 of the host runtime's logits, float32 LE, as 8 hex digits>\t<text>",
      from runtime/build/libvocr.so (pyvocr). The PSP app compares every line with it.
      --head-gate on|score, --two-level: the text is the gated / two-level run's (the app's timed runs, with
      --repeat >= 1); the CRC stays the full logits' (a run that asks for the logits skips nothing).
      --finder N [--refine FLAGS]: every crop is a dialogue region (grey or RGBA): the host's C line finder N
      and refinement (runtime/vocr_lines*.c, built now), then every line read with its dark flag as
      VOCR_INVERT; the CRC runs over the lines' logits in order, the text is their texts joined by tabs.
  pack.py bundle --blob B.vocr --mode w8f --pack DIR --out OUT [--eboot psp/EBOOT.PBP] [--name VOCR]
      OUT/PSP/GAME/VOCR/: EBOOT.PBP, model.vocr, lines.bin, expect.txt, vocr.cfg; copy OUT/PSP
      to the root of a memory stick (or into PPSSPP's memstick) and start "vocr OCR test".
  --mode defaults to w8f, the mode psp/README.md recommends for the PSP.

Screenshots come from --eval-data, $VOCR_EVAL_DATA or data/eval (eval/fetch.py rebuilds them);
manifests from --manifests (default eval/manifests; point it at a directory holding
psp-test.jsonl while eval's psp-test is not on main). The crops are cut from copyrighted
screenshots: packs stay local, like the screenshots.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "runtime")]


def rgba(rgb):
    return np.ascontiguousarray(np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2))


def load_set(name, n, eval_data, manifests):
    import fetch
    import sets
    if manifests:
        sets.MANIFESTS = Path(manifests)
    if name == "synth-val":
        out = []
        shard = os.environ.get("VOCR_SYNTH_VAL", sets.SYNTH_VAL)              # as eval/run.py
        for ln in sets.load_synth_val(shard, limit=n or sets.SYNTH_VAL_LINES):
            g = np.ascontiguousarray(ln.crop_rgb[:, :, 0])           # the shard's grey line
            out.append((ln.id, name, g, ln.text, ln.styles))
        return out
    lines, _ = sets.load(name, fetch.data_dir(eval_data))
    lines = lines[:n] if n else lines
    return [(ln.id, name, rgba(ln.crop_rgb), ln.text, ln.styles) for ln in lines]


def write_pack(path, crops):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(crops)))
        for img in crops:
            f.write(struct.pack("<III", img.shape[0], img.shape[1], 4 if img.ndim == 3 else 1))
            f.write(np.ascontiguousarray(img).tobytes())


def read_pack(path):
    raw = Path(path).read_bytes()
    (count,), off, out = struct.unpack_from("<I", raw), 4, []
    for _ in range(count):
        h, w, ch = struct.unpack_from("<III", raw, off)
        off += 12
        a = np.frombuffer(raw, np.uint8, h * w * ch, off)
        out.append(a.reshape(h, w) if ch == 1 else a.reshape(h, w, ch))
        off += h * w * ch
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return out


def cmd_build(a):
    rows, crops = [], []
    for name in a.set:
        for ident, s, img, text, styles in load_set(name, a.n, a.eval_data, a.manifests):
            rows.append({"id": ident, "set": s, "text": text, "styles": styles,
                         "h": int(img.shape[0]), "w": int(img.shape[1]), "ch": 4 if img.ndim == 3 else 1})
            crops.append(img)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_pack(out / "lines.bin", crops)
    (out / "lines.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                     encoding="utf-8")
    print(f"{len(crops)} crops ({', '.join(a.set)}) -> {out / 'lines.bin'} "
          f"({(out / 'lines.bin').stat().st_size} bytes)")


FINDER_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror")


def finder_driver(out_dir):
    """The host build of the line finder + refinement driver (runtime/vocr_lines_refine_test.c), built now."""
    import subprocess
    drv = Path(out_dir) / "vocr_lines_refine_test"
    subprocess.run(["gcc", *FINDER_FLAGS, "-o", str(drv), "runtime/vocr_lines.c", "runtime/vocr_lines_refine.c",
                    "runtime/vocr_lines_refine_test.c"], check=True, cwd=ROOT)
    return drv


def find_lines_c(driver, regions, finder, flags):
    """[[(x, y, w, h, dark)] per region]: the C finder `finder` then vocr_refine_lines(flags) on each region
    (grey or RGBA uint8), as the PSP app's finder mode runs them."""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img in regions:
            img = np.ascontiguousarray(img, np.uint8)
            h, w = img.shape[:2]
            bpp = 1 if img.ndim == 2 else img.shape[2]
            f.write(struct.pack("<6I", w, h, bpp, finder, w * bpp, flags))
            f.write(img.tobytes())
        path = f.name
    try:
        lines = subprocess.run([str(driver), path], check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(lines) != len(regions):
        raise RuntimeError(f"finder driver returned {len(lines)} results for {len(regions)} regions")
    out = []
    for ln in lines:
        v = [int(t) for t in ln.split()]
        if v[0]:
            raise RuntimeError(f"finder rc {v[0]}")
        out.append([tuple(v[5 + 5 * i:10 + 5 * i]) for i in range(v[2])])
    return out


def expected(blob, mode, crops, invert=False, tile=0, head=0, head_gate="off", two_level=False, repeat=1,
             finder=-1, refine=31, driver=None):
    """[(crc32 of the logits, text)] from the host runtime, as the PSP app checks them (psp/main.c): the CRC of
    the first, ungated run's full logits, the text of the last run (gated / two-level when the app repeats)."""
    from pyvocr import Vocr
    v = Vocr(blob, mode=mode, tile=tile, head=head, head_gate=head_gate, two_level=int(two_level))
    fast = repeat >= 1 and (head_gate == "on" or two_level)
    boxes = find_lines_c(driver, crops, finder, refine) if finder >= 0 else None
    out = []
    for i, img in enumerate(crops):
        if boxes is None:
            parts = [(img, invert)]
        else:
            parts = [(img[y:y + h, x:x + w], bool(d) != bool(invert)) for x, y, w, h, d in boxes[i]]
        crc, texts = 0, []
        for crop, inv in parts:
            text, logits = v.run(np.ascontiguousarray(crop), logits=True, invert=inv)
            crc = zlib.crc32(np.ascontiguousarray(logits, "<f4").tobytes(), crc)
            texts.append(v.run(np.ascontiguousarray(crop), invert=inv) if fast else text)
        out.append((crc, "\t".join(texts)))
    return out


def write_expect(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for crc, text in rows:
            if "\n" in text:
                raise ValueError(f"text with a newline: {text!r}")
            f.write(f"{crc:08x}\t{text}\n")


def _expected_args(a):
    driver = finder_driver(Path(a.out).parent if a.cmd == "expect" else Path(a.out)) if a.finder >= 0 else None
    return dict(invert=a.invert, tile=a.tile, head=a.head, head_gate=a.head_gate, two_level=a.two_level,
                repeat=a.repeat, finder=a.finder, refine=a.refine, driver=driver)


def cmd_expect(a):
    crops = read_pack(a.pack)
    write_expect(a.out, expected(a.blob, a.mode, crops, **_expected_args(a)))
    print(f"{len(crops)} {'regions' if a.finder >= 0 else 'lines'} ({a.mode}{', inverted' if a.invert else ''}, "
          f"gate {a.head_gate}, two-level {int(a.two_level)}) -> {a.out}")


def bundle_cfg(mode, invert=False, repeat=1, clock=333, tile=0, head=0, head_gate="off", two_level=False,
               finder=-1, refine=31, probe=0):
    """vocr.cfg of a memory-stick bundle (keys: psp/main.c cfg_set): check the blob's CRC,
    compare every line with expect.txt, wait for START at the end. tile / head 0: the
    runtime's defaults (4 / 16)."""
    return {"mode": mode, "invert": int(invert), "repeat": repeat, "crc": 1, "clock": clock, "wait": 1,
            "tile": tile, "head": head, "gate": head_gate, "two_level": int(two_level), "finder": finder,
            "refine": refine, "probe": int(probe)}


def cmd_bundle(a):
    src = Path(a.pack)
    app = Path(a.out) / "PSP" / "GAME" / a.name
    app.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.eboot, app / "EBOOT.PBP")
    shutil.copyfile(a.blob, app / "model.vocr")
    shutil.copyfile(src / "lines.bin", app / "lines.bin")
    if (src / "lines.jsonl").exists():
        shutil.copyfile(src / "lines.jsonl", app / "lines.jsonl")
    crops = read_pack(app / "lines.bin")
    write_expect(app / "expect.txt", expected(a.blob, a.mode, crops, **_expected_args(a)))
    cfg = bundle_cfg(a.mode, a.invert, a.repeat, a.clock, a.tile, a.head, a.head_gate, a.two_level, a.finder,
                     a.refine, a.probe)
    (app / "vocr.cfg").write_text("".join(f"{k}={v}\n" for k, v in cfg.items()), encoding="utf-8")
    print(f"{app}: EBOOT.PBP, model.vocr, {len(crops)} crops, expect.txt ({a.mode}), vocr.cfg")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--set", action="append", required=True,
                   choices=["psp-test", "vita-test", "synth-val"])
    b.add_argument("--n", type=int, default=0, help="at most N lines per set (0: all)")
    b.add_argument("--eval-data")
    b.add_argument("--manifests", help="directory with <set>.jsonl (default eval/manifests)")
    b.add_argument("--out", required=True)
    e = sub.add_parser("expect")
    for p in (e,):
        p.add_argument("--pack", required=True, help="a lines.bin")
        p.add_argument("--out", required=True)
    u = sub.add_parser("bundle")
    u.add_argument("--pack", required=True, help="the directory pack.py build wrote")
    u.add_argument("--out", required=True)
    u.add_argument("--eboot", default=str(HERE / "EBOOT.PBP"))
    u.add_argument("--name", default="VOCR", help="folder under PSP/GAME (default VOCR)")
    u.add_argument("--clock", type=int, default=333)
    u.add_argument("--probe", action="store_true", help="the app reports every memory partition's free block")
    for p in (e, u):
        p.add_argument("--blob", required=True)
        p.add_argument("--mode", default="w8f", choices=["f32", "w8f", "w8i"])
        p.add_argument("--invert", action="store_true")
    for p in (e, u):
        p.add_argument("--tile", type=int, default=0, help="CTC frames per tile (0: 4)")
        p.add_argument("--head", type=int, default=0, help="frames per head batch (0: 16)")
        p.add_argument("--head-gate", default="off", choices=["off", "on", "score"])
        p.add_argument("--two-level", action="store_true")
        p.add_argument("--repeat", type=int, default=1, help="the app's timed runs per line (expect: its text)")
        p.add_argument("--finder", type=int, default=-1, choices=[-1, 0, 2, 3],
                       help="page path: crops are regions, read through this line finder (vocr_lines.h)")
        p.add_argument("--refine", type=int, default=31, help="vocr_refine_lines flags in finder mode")
    a = ap.parse_args()
    {"build": cmd_build, "expect": cmd_expect, "bundle": cmd_bundle}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Crop packs for the PSP test app, the host runtime's expected outputs, and app bundles.

  pack.py build --set psp-test [--set vita-test --set synth-val] [--n N] --out DIR
      DIR/lines.bin    the crops in vocr_cli's pack format (runtime/tools/vocr_cli.c): u32 count,
                       then per crop u32 height, u32 width, u32 channels, the pixels. Real sets
                       are RGBA (eval/sets.py's GT-box crops, as runtime/tests/parity.py feeds
                       them), synth-val is grey, so both input paths run.
      DIR/lines.jsonl  per crop: id, set, ground truth, style tags
  pack.py expect --blob B.vocr --mode w8i --pack DIR/lines.bin --out DIR/expect_w8i.txt
      per crop: "<CRC-32 of the host runtime's logits, float32 LE, as 8 hex digits>\t<text>",
      from runtime/build/libvocr.so (pyvocr). The PSP app compares every line with it.
  pack.py bundle --blob B.vocr --mode w8i --pack DIR --out OUT [--eboot psp/EBOOT.PBP] [--name VOCR]
      OUT/PSP/GAME/VOCR/: EBOOT.PBP, model.vocr, lines.bin, expect.txt, vocr.cfg; copy OUT/PSP
      to the root of a memory stick (or into PPSSPP's memstick) and start "vocr OCR test".

Screenshots come from --eval-data, $VOCR_EVAL_DATA or data/eval (eval/fetch.py rebuilds them);
manifests from --manifests (default eval/manifests; point it at a directory holding
psp-test.jsonl while eval's psp-test is not on main). The crops are cut from copyrighted
screenshots: packs stay local, like the screenshots.
"""
import argparse
import json
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
        for ln in sets.load_synth_val(limit=n or sets.SYNTH_VAL_LINES):
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


def expected(blob, mode, crops, invert=False, tile=0, head=0):
    """[(crc32 of the logits, text)] from the host runtime."""
    from pyvocr import Vocr
    v = Vocr(blob, mode=mode, tile=tile, head=head)
    out = []
    for img in crops:
        text, logits = v.run(img, logits=True, invert=invert)
        out.append((zlib.crc32(np.ascontiguousarray(logits, "<f4").tobytes()), text))
    return out


def write_expect(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for crc, text in rows:
            if "\n" in text or "\t" in text:
                raise ValueError(f"text with a tab or newline: {text!r}")
            f.write(f"{crc:08x}\t{text}\n")


def cmd_expect(a):
    crops = read_pack(a.pack)
    write_expect(a.out, expected(a.blob, a.mode, crops, a.invert, a.tile, a.head))
    print(f"{len(crops)} lines ({a.mode}{', inverted' if a.invert else ''}) -> {a.out}")


def bundle_cfg(mode, invert=False, repeat=1, clock=333, tile=0, head=0):
    """vocr.cfg of a memory-stick bundle (keys: psp/main.c cfg_set): check the blob's CRC,
    compare every line with expect.txt, wait for START at the end. tile / head 0: the
    runtime's defaults (4 / 16)."""
    return {"mode": mode, "invert": int(invert), "repeat": repeat, "crc": 1, "clock": clock, "wait": 1,
            "tile": tile, "head": head}


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
    write_expect(app / "expect.txt", expected(a.blob, a.mode, crops, a.invert, a.tile, a.head))
    cfg = bundle_cfg(a.mode, a.invert, a.repeat, a.clock, a.tile, a.head)
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
    u.add_argument("--repeat", type=int, default=1)
    u.add_argument("--clock", type=int, default=333)
    for p in (e, u):
        p.add_argument("--blob", required=True)
        p.add_argument("--mode", default="w8i", choices=["f32", "w8f", "w8i"])
        p.add_argument("--invert", action="store_true")
    for p in (e, u):
        p.add_argument("--tile", type=int, default=0, help="CTC frames per tile (0: 4)")
        p.add_argument("--head", type=int, default=0, help="frames per head batch (0: 16)")
    a = ap.parse_args()
    {"build": cmd_build, "expect": cmd_expect, "bundle": cmd_bundle}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parity of runtime/vocr_charbox.c with its twin eval/charbox.py (issue #114, docs/char-boxes.md).

    charbox_parity.py --bin runtime/build/charbox/vocr_charbox_test [--lines 3000] [--aligns 3000] [--seed 0]

Random lines through vocr_charbox_find (greedy runs with blanks and repeats over data/vocab.txt's classes, the
punctuation groups weighted up; stride 1 and top-K arrays; crops from 1 to 1,200 px wide and 1 to 100 tall, so
resized widths from 8 to 1,024; every flag combination; calibrations from 0 to the int16 extremes; caps from 1
to 256; profiles with ties, zero runs and spikes, and GREY8 / RGBA8 crops) and random pairs of texts through
vocr_charbox_align (substitutions, insertions, deletions, swaps and repeats over small alphabets), through the
C driver (vocr_charbox_test --parity) and the twin: every class, frame, group, centre, cell, ink box and map
entry must be the same.
"""
import argparse
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import charbox as CB  # noqa: E402


def vocab():
    lines = (ROOT / "data" / "vocab.txt").read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [""] + lines


def class_groups(classes):
    return [CB.group(t) if len(t) == 1 else CB.G_OTHER for t in classes]


def make_line(rng, nprng, n_classes, punct):
    frames = rng.choice([1, 2, 5, rng.randint(1, 128), rng.randint(1, 128), rng.randint(60, 128)])
    seq, last = [], 0
    for _ in range(frames):
        r = rng.random()
        if r < 0.45:
            c = 0
        elif r < 0.65 and last:
            c = last                                     # the run goes on
        elif r < 0.8 and punct:
            c = rng.choice(punct)
        else:
            c = rng.randint(1, n_classes - 1)
        seq.append(c)
        last = c
    K = rng.choice([1, 1, 1, 2, 8, 16])
    cls = np.zeros((frames, K), np.uint16)
    cls[:, 0] = seq
    if K > 1:
        cls[:, 1:] = nprng.integers(0, n_classes, size=(frames, K - 1))
    w = rng.choice([1, 7, 8, 9, rng.randint(1, 1200), rng.randint(20, 960), rng.randint(100, 400)])
    h = rng.choice([1, 2, 16, 32, rng.randint(1, 100), rng.randint(10, 48)])
    flags = rng.randint(0, 7)
    if rng.random() < 0.3:
        cal = None
    elif rng.random() < 0.1:
        cal = [rng.choice([-32768, -32767, 32767, 0, rng.randint(-32768, 32767)]) for _ in range(9)]
    else:
        cal = [rng.randint(-200, 200) for _ in range(9)]
    cap = rng.choice([256, 256, 256, rng.randint(1, 256), 1, 2])
    ink = rng.choice([1, 2, 3]) if flags & CB.INK else rng.choice([0, 1, 2])
    prof = pix = None
    if ink == 1:
        kind = rng.random()
        if kind < 0.3:
            prof = nprng.integers(0, 4, size=w)                          # many ties and zeros
        elif kind < 0.5:
            prof = np.zeros(w, np.int64)                                 # no ink at all
        else:
            prof = nprng.integers(0, 4000, size=w)
            for _ in range(rng.randint(0, 20)):                           # gaps
                a = rng.randint(0, w)
                prof[a:a + rng.randint(1, 12)] = rng.choice([0, 0, 1, 5])
            if rng.random() < 0.2:
                prof[rng.randint(0, w - 1)] = 2 ** 31 + rng.randint(0, 1000)     # a spike past int32
        prof = np.asarray(prof, np.int64)
    elif ink in (2, 3):
        bpp = 1 if ink == 2 else 4
        a = nprng.integers(0, 256, size=(h, w, bpp), dtype=np.uint8) if rng.random() < 0.3 else \
            np.zeros((h, w, bpp), np.uint8)
        for _ in range(rng.randint(0, 30)):                               # strokes
            x0, y0 = rng.randint(0, w - 1), rng.randint(0, h - 1)
            a[y0:y0 + rng.randint(1, max(1, h // 2)), x0:x0 + rng.randint(1, 6)] = rng.randint(0, 255)
        pix = a if bpp == 4 else a[:, :, 0]
    return {"cls": cls, "w": w, "h": h, "flags": flags, "cal": cal, "cap": cap, "ink": ink, "prof": prof,
            "pix": pix}


def twin_line(c, groups):
    best = c["cls"][:, 0]
    nw = CB.resized_width(c["w"], c["h"])
    if c["ink"] == 1:
        prof = c["prof"]
    elif c["ink"] in (2, 3):
        p = c["pix"].astype(np.int64)
        grey = p if p.ndim == 2 else (77 * p[:, :, 0] + 150 * p[:, :, 1] + 29 * p[:, :, 2]) >> 8
        prof = CB.ink_profile(grey)
    else:
        prof = None
    out = CB.positions(best, c["w"], c["h"], nw, c["flags"], prof, c["cal"], groups, cap=c["cap"])
    toks = [0, len(out)]
    for o in out:
        toks += [o["cls"], o["f0"], o["f1"], groups[o["cls"]], o["c"], o["x0"], o["x1"], o["i0"], o["i1"]]
    return "F " + " ".join(str(v) for v in toks)


def mutate(rng, src, alphabet):
    out = list(src)
    for _ in range(rng.randint(0, 6)):
        r = rng.random()
        if not out or r < 0.25:
            out.insert(rng.randint(0, len(out)), rng.choice(alphabet))
        elif r < 0.5:
            del out[rng.randint(0, len(out) - 1)]
        elif r < 0.75:
            out[rng.randint(0, len(out) - 1)] = rng.choice(alphabet)
        elif len(out) > 1:
            i = rng.randint(0, len(out) - 2)
            out[i], out[i + 1] = out[i + 1], out[i]
    if rng.random() < 0.1:
        out = [rng.choice(alphabet) for _ in range(rng.randint(0, 40))]
    return out


def make_align(rng):
    alphabet = [ord(c) for c in rng.choice(["ab", "abc", "あいうえおかき「」、。", "の", "abcdefghij"])]
    src = [rng.choice(alphabet) for _ in range(rng.choice([0, 1, 5, rng.randint(0, 60), rng.randint(0, 300)]))]
    return src, mutate(rng, src, alphabet)


def write_cases(path, lines, aligns):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(lines) + len(aligns)))
        for c in lines:
            frames, K = c["cls"].shape
            f.write(struct.pack("<III", 0, frames, K))
            f.write(np.ascontiguousarray(c["cls"], "<u2").tobytes())
            cal = c["cal"] if c["cal"] is not None else [0] * 9
            f.write(struct.pack("<IIII", c["w"], c["h"], c["flags"], int(c["cal"] is not None)))
            f.write(struct.pack("<9h", *cal))
            f.write(struct.pack("<II", c["cap"], c["ink"]))
            if c["ink"] == 1:
                f.write(np.asarray(np.minimum(c["prof"], 2 ** 32 - 1), "<u4").tobytes())
            elif c["ink"] in (2, 3):
                f.write(np.ascontiguousarray(c["pix"], np.uint8).tobytes())
        for src, dst in aligns:
            f.write(struct.pack("<III", 1, len(src), len(dst)))
            f.write(struct.pack(f"<{len(src)}I", *src))
            f.write(struct.pack(f"<{len(dst)}I", *dst))


def real_lines(units_path, meiki_path, tier, n, seed):
    """Cases from development lines (the scorer's fixed crops of units with reference boxes), read by a released
    model: (classes of the model, cases). Every arm's flags; the model's committed calibration."""
    sys.path.insert(0, str(ROOT / "runtime"))
    sys.path.insert(0, str(ROOT / "synth"))
    import charbox_eval as CE
    import charbox_sets as CS
    units = {u["uid"]: u for u in CS.jsonl(units_path)}
    gts = CE.load_gt(units, meiki_path)
    cal = CE.load_cal()[tier]
    r = CE.Reader(tier)
    classes = [""] + [r.v.class_text(k) for k in range(1, r.v.n_classes)]
    rng = random.Random(seed)
    uids = sorted(u for u, b in gts.items() if any(x is not None for x in b) and units[u]["vertical"] == (tier == "V"))
    rng.shuffle(uids)
    cases = []
    for uid in uids[:n]:
        u = units[uid]
        grey = CE.to_gray(CS.frame_rgb(u["frame"]))
        x, y, w, h = CE.crop_box(u, (grey.shape[1], grey.shape[0]))
        crop = np.ascontiguousarray(grey[y:y + h, x:x + w])
        dark = CE.line_dark(grey, (x, y, w, h))
        if u["vertical"]:
            import vertical_twin as V
            crop = np.ascontiguousarray(V.rotate_ccw(255 - crop if dark else crop))
            text, best, nw = r.read(crop)
        else:
            text, best, nw = r.read(crop, invert=bool(dark))
        for flags in sorted({f for f, _ in CE.ARMS.values()} | {CB.CENTRE_RUN | CB.INK}):
            cases.append({"cls": np.asarray(best, np.uint16).reshape(-1, 1), "w": crop.shape[1], "h": crop.shape[0],
                          "flags": flags, "cal": cal, "cap": 256, "ink": 2, "prof": None, "pix": crop})
    return classes, cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="the driver, or a command with its prefix (qemu)")
    ap.add_argument("--lines", type=int, default=3000)
    ap.add_argument("--aligns", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-units", help="development units (charbox_sets.py): real lines instead of random ones")
    ap.add_argument("--real-meiki")
    ap.add_argument("--tier", default="N")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    if args.real_units:
        classes, lines = real_lines(args.real_units, args.real_meiki, args.tier, args.lines, args.seed)
        groups = class_groups(classes)
        aligns = []
    else:
        classes = vocab()
        groups = class_groups(classes)
        punct = [k for k, t in enumerate(classes) if k and groups[k] != CB.G_OTHER]
        lines = [make_line(rng, nprng, len(classes), punct) for _ in range(args.lines)]
        aligns = [make_align(rng) for _ in range(args.aligns)]
    for c in lines:     # a profile the C side reads as u32: the twin sees the same clipped values
        if c["ink"] == 1:
            c["prof"] = np.minimum(c["prof"], 2 ** 32 - 1)
    with tempfile.TemporaryDirectory(prefix="charbox_") as td:
        vp, cp = Path(td) / "vocab.txt", Path(td) / "cases.bin"
        vp.write_text("\n".join(classes[1:]) + "\n", encoding="utf-8")
        write_cases(cp, lines, aligns)
        r = subprocess.run(args.bin.split() + ["--parity", str(vp), str(cp)], capture_output=True, text=True,
                           timeout=3600)
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"driver failed: {r.returncode}")
    got = r.stdout.splitlines()
    want = [twin_line(c, groups) for c in lines] + \
        ["A 0" + "".join(f" {v}" for v in CB.align_map(s, d)) for s, d in aligns]
    if len(got) != len(want):
        raise SystemExit(f"{len(got)} results for {len(want)} cases")
    bad = [k for k, (g, w) in enumerate(zip(got, want)) if g != w]
    for k in bad[:5]:
        print(f"case {k}:\n  C    {got[k][:300]}\n  twin {want[k][:300]}", file=sys.stderr)
    n_chars = sum(int(w.split()[2]) for w in want[:len(lines)])
    print(f"CHARBOX_PARITY {'OK' if not bad else 'FAIL'} lines={len(lines)} chars={n_chars} "
          f"aligns={len(aligns)} differ={len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

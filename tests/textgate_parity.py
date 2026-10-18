#!/usr/bin/env python3
"""Parity check: runtime/vocr_textgate.c against its twin, eval/textgate.py (kmap on synth/textdet_twin.run_net,
kth, box_mask, score: the gate's specification).

  textgate_parity.py --bin DRIVER [--seeds 2] [--regions 8] [--seed 0] [--shapes p8_c4,s4_c4,...] [--wild 2]
                     [--chains 4] [--probes 1] [--screen-every 4] [--maps-every 2] [--finder-every 3]
                     [--selftest N --selftest-blobs B] [--blob [NAME=]FILE ...] [--cohort PKL ...] [--json OUT]

Blobs: random blobs (train/textdet_export.random_int_layers + pack) of every presence shape of
train/textgate_train.SHAPES (registered in textdet_model.SHAPES for that call only: the file is not touched) and
every #97 shape of train/textdet_model.SHAPES, --seeds seeds each; wild-requantisation blobs (tests/textdet_parity
.wild_blob: weights -128, biases over all of int32, multipliers 1 .. 2^31 - 1, shifts 1 .. 62); probe blobs (K =
the grey of every s-th pixel, so designed regions set the K map cell by cell: ties, all-equal maps, extremes);
random valid layer chains no shape uses (a 1x1 first layer, runs of 1x1 layers, a 3x3 dense or depthwise head,
dilations up to 8, up to 64 channels, ReLU anywhere but the head); --blob files (trained blobs).

Regions: edge sizes from 1x1, odd sizes, procedural regions (textdet_export.procedural_region; designed ones for
the probes), whole 960x544 / 480x272 screens, GREY or RGBA of the same grey; --cohort pickles (real regions, field
"image"). Each region runs through the C driver (runtime/vocr_textgate_test.c --batch) in the image, rows and push
forms, in both modes (boxes mode: random boxes inside, across the border, outside, empty, overlapping, up to 16), k
in {1, 2, 4, 8}, padded strides and odd arenas on some; the finder mode (vocr_find_lines_gated / _rows, finders 0-3,
t at and next to the twin's score) on every --finder-every-th region against the plain finder's lines gated by the
twin's score; and the K map (--maps, boxes mode) on every --maps-every-th. Every C score and map must equal
eval/textgate.score / kmap exactly. --selftest N also runs the driver's own invariants on N regions of a few blobs.

Prints TEXTGATE_PARITY_OK <blobs> blobs (<regions> regions, <records> records, <maps> maps; <seconds> s). On real
regions (--cohort) it prints parity only: a mismatch names the case, never a score or a decision. --json writes
{blob name: {"c_equals_twin": bool, "regions": n, "records": n, "sha256": ...}} (eval/textgate.py select --parity).
"""
import argparse
import hashlib
import json
import pickle
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_parity as TP  # noqa: E402
import textdet_twin as TW  # noqa: E402
import textgate as TG  # noqa: E402
import textgate_train as G  # noqa: E402

ROWS, PUSH, BOXES, ODD, FINDER, NOGATE = 1, 2, 4, 8, 256, 512     # vocr_textgate_test's record flags
KS = (1, 2, 4, 8)
UNSET = 99999
MODES = ("region", "boxes")


# ---------------------------------------------------------------- blobs

def with_shape(shape, fn, *args):
    """fn(shape name, *args) with shape registered in textdet_model.SHAPES for the call (export's random_int_layers
    looks shapes up there; a presence shape of textgate_train is not in it)."""
    name = shape["name"]
    added = name not in TM.SHAPES
    if added:
        TM.SHAPES[name] = shape
    try:
        return fn(name, *args)
    finally:
        if added:
            del TM.SHAPES[name]


def random_blob(shape, seed, n_cal=6):
    sh, q = with_shape(shape, E.random_int_layers, seed, n_cal)
    return E.pack(sh, q, hashlib.sha256(f"random:{shape['name']}:{seed}".encode()).digest())


def wild_blob(shape, seed):
    return with_shape(shape, TP.wild_blob, seed)


def _calibrated(specs, rng, n_cal=4):
    """Integer layers for layer specs (dicts as textdet_model.layers gives) with random int8 weights and the
    requantisation spread on procedural regions, as export's random_int_layers does."""
    xs = [E.procedural_region(rng).astype(np.int64)[None] - 128 for _ in range(n_cal)]
    out = []
    for L in specs:
        per = 1 if L["kind"] == "dw" else L["cin"]
        wq = rng.integers(-127, 128, (L["cout"], per, L["k"], L["k"])).astype(np.int8)
        wq[rng.random(wq.shape) < 0.3] = 0
        tl = TW.Layer()
        tl.kind, tl.k, tl.stride, tl.dil, tl.relu = (1 if L["kind"] == "dw" else 0), L["k"], L["stride"], L["dil"], 0
        tl.cin, tl.cout = L["cin"], L["cout"]
        tl.w = wq.astype(np.int64)
        tl.b = np.zeros(L["cout"], np.int64)
        a = np.concatenate([E._accumulate(tl, x).reshape(L["cout"], -1) for x in xs], 1).astype(np.float64)
        med, sd = np.median(a, 1), a.std(1) + 1.0
        M = np.full(L["cout"], 30.0) / sd
        if L["head"]:
            M = np.array([24.0, 24.0, 6.0, 6.0]) / sd
        ms = [E.quantize_multiplier(float(v)) for v in M]
        bq = np.floor((10.0 if L["relu"] else 0.0) / M - med + 0.5).astype(np.int64)
        q = {**L, "wq": wq, "bq": bq, "m": np.array([m for m, _ in ms], np.int64),
             "sh": np.array([s for _, s in ms], np.int64)}
        out.append(q)
        tl.relu, tl.b, tl.m, tl.sh = (1 if L["relu"] else 0), bq, q["m"], q["sh"]
        xs = [TW.run_layer(tl, x) for x in xs]
    return out


def chain_blob(seed):
    """A random valid layer chain the trained shapes never use: an optional 1x1 first layer, 3x3 layers dense or
    depthwise (stride 2 ones making the output stride 4 or 8; stride 1 ones dilated 1..8), runs of 1x1 layers,
    1..64 channels, ReLU anywhere but the head, and a 1x1, 3x3 dense or 3x3 depthwise head of 4 maps."""
    rng = np.random.default_rng(seed)
    stride = int(rng.choice([4, 8]))
    kinds = ["s2"] * (2 if stride == 4 else 3) + list(rng.choice(["1x1", "3x3", "dw", "1x1"], int(rng.integers(0, 5))))
    rng.shuffle(kinds)
    if rng.random() < 0.4:
        kinds.insert(0, "1x1")
    head = str(rng.choice(["1x1", "3x3", "dw"]))
    specs, cin = [], 1

    def chans():
        return int(rng.integers(1, 65)) if rng.random() < 0.1 else int(rng.integers(1, 13))
    for kd in kinds:
        relu = bool(rng.random() < 0.6)
        if kd == "s2":
            dw = cin > 1 and rng.random() < 0.5
            cout = cin if dw else chans()
            specs.append(dict(kind="dw" if dw else "conv", cin=cin, cout=cout, k=3, stride=2, dil=1, relu=relu))
        elif kd == "1x1":
            cout = chans()
            specs.append(dict(kind="conv", cin=cin, cout=cout, k=1, stride=1, dil=1, relu=relu))
        elif kd == "3x3":
            cout = chans()
            specs.append(dict(kind="conv", cin=cin, cout=cout, k=3, stride=1, dil=int(rng.integers(1, 9)), relu=relu))
        else:
            cout = cin
            specs.append(dict(kind="dw", cin=cin, cout=cout, k=3, stride=1, dil=int(rng.integers(1, 9)), relu=relu))
        cin = cout
    if head == "dw" and cin != 4:
        specs.append(dict(kind="conv", cin=cin, cout=4, k=1, stride=1, dil=1, relu=True))
        cin = 4
    if head == "1x1":
        specs.append(dict(kind="conv", cin=cin, cout=4, k=1, stride=1, dil=1, relu=False))
    else:
        specs.append(dict(kind="dw" if head == "dw" else "conv", cin=cin, cout=4, k=3, stride=1,
                          dil=int(rng.integers(1, 9)), relu=False))
    for i, L in enumerate(specs):
        L["head"] = i == len(specs) - 1
    q = _calibrated(specs, rng)
    shape = {"name": f"chain_{seed}", "stride": stride}
    return E.pack(shape, q, hashlib.sha256(shape["name"].encode()).digest())


def all_blobs(args):
    """[(name, bytes, probe stride or None)]."""
    shapes = [s for s in args.shapes.split(",") if s] or (list(G.SHAPES) + list(TM.SHAPES))
    lookup = {**TM.SHAPES, **G.SHAPES}
    out = []
    for i, name in enumerate(shapes):
        for k in range(args.seeds):
            seed = 1000 * (k + 1) + i
            out.append((f"{name}_seed{seed}", random_blob(lookup[name], seed, args.n_cal), None))
    for k, name in enumerate(shapes[::max(1, len(shapes) // max(1, args.wild))][:args.wild]):
        out.append((f"wild_{name}_{k}", wild_blob(lookup[name], 77 + k), None))
    for k in range(args.chains):
        out.append((f"chain_{k}", chain_blob(5000 + k + 100 * args.seed), None))
    for k in range(args.probes):
        for stride in (4, 8):
            out.append((f"probe_s{stride}_{k}", TP.probe_blob(stride, k % len(TP.PROBE_VARIANTS)), stride))
    for spec in args.blob:
        name, _, path = spec.rpartition("=")
        out.append((name or Path(path).parent.name or Path(path).name, Path(path).read_bytes(), None))
    return out


# ---------------------------------------------------------------- regions and boxes

def random_boxes(rng, w, h):
    """Up to 16 boxes: inside, across the border, outside, empty, overlapping, and huge unsigned values."""
    out = []
    for _ in range(int(rng.choice([0, 1, 1, 2, 3, 4, 16]) if rng.random() < 0.9 else rng.integers(0, 17))):
        kind = rng.random()
        if kind < 0.6:
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            bw, bh = int(rng.integers(1, w - x + 1)), int(rng.integers(1, h - y + 1))
        elif kind < 0.75:
            x, y = int(rng.integers(0, w + 20)), int(rng.integers(0, h + 20))
            bw, bh = int(rng.integers(1, w + 30)), int(rng.integers(1, h + 30))
        elif kind < 0.85:
            x, y, bw, bh = int(rng.integers(0, w + 9)), int(rng.integers(0, h + 9)), int(rng.integers(0, 3)), \
                int(rng.integers(0, 3))
        elif kind < 0.93:
            x, y = int(rng.integers(0, 2 ** 32)), int(rng.integers(0, 2 ** 32))
            bw, bh = int(rng.integers(0, 2 ** 32)), int(rng.integers(0, 2 ** 32))
        else:
            x, y, bw, bh = out[-1] if out else (0, 0, w, h)
        out.append((x, y, bw, bh))
    return out


def regions(rng, n, probe_stride=None, screen=None):
    return TP.regions_for(rng, n, probe_stride, screen)


def cohort_regions(paths, limit):
    out = []
    for p in paths:
        cases = pickle.load(open(p, "rb"))
        for i, c in enumerate(cases[:limit] if limit else cases):
            img = np.asarray(c["image"])
            out.append((f"{Path(p).stem}:{i}", img))
    return out


def to_grey(img):
    from common import grey_of
    return np.ascontiguousarray(grey_of(img))


def pixels_of(img, rng):
    """(pixels as the driver gets them, their grey): grey as is or RGBA of the same grey; RGB(A) inputs as RGBA."""
    img = np.asarray(img, np.uint8)
    if img.ndim == 3:
        rgba = np.empty(img.shape[:2] + (4,), np.uint8)
        rgba[..., :3] = img[..., :3]
        rgba[..., 3] = 255 if img.shape[2] == 3 else img[..., 3]
        return rgba, to_grey(rgba)
    return (TP.to_rgba(img, rng) if rng.random() < 0.5 else img), img


# ---------------------------------------------------------------- the driver

def record(pix, flags, boxes=None, finder=None):
    h, w = pix.shape[:2]
    bpp = 1 if pix.ndim == 2 else 4
    b = struct.pack("<4I", w, h, bpp, flags) + np.ascontiguousarray(pix).tobytes()
    if flags & BOXES:
        b += struct.pack("<I", len(boxes)) + b"".join(struct.pack("<4I", *bx) for bx in boxes)
    if flags & FINDER:
        b += struct.pack("<2i", *finder)
    return b


def run_driver(binary, args, timeout=3600):
    p = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    if p.returncode:
        raise RuntimeError(f"{binary} {' '.join(map(str, args))}: rc {p.returncode}\n{p.stderr[-3000:]}")
    return p.stdout.splitlines()


def check_blob(binary, name, blob, imgs, rng, td, args, real=False):
    """Every region of imgs [(label, image)] through C and the twin. Returns (records, maps, mismatches)."""
    model = TW.Model(blob)
    s = model.stride
    bpath = Path(td) / "blob.vtd"
    bpath.write_bytes(blob)
    recs, exp, finder_pairs = [], [], []
    maps_recs, maps_exp = [], []
    for ri, (label, img) in enumerate(imgs):
        pix, grey = pixels_of(img, rng)
        km = TG.kmap(model, grey)
        h, w = grey.shape
        for mode in (0, 1):
            for form in (0, ROWS, PUSH):
                k = int(rng.choice(KS))
                boxes = random_boxes(rng, w, h) if mode else None
                flags = form | (BOXES if mode else 0) | (k << 4) | (ODD if rng.random() < 0.3 else 0)
                recs.append(record(pix, flags, boxes))
                want = TG.score(km, MODES[mode], k, s, boxes)
                exp.append((label, f"{MODES[mode]} form {form} k {k}", f"0 {want}"))
        if args.maps_every and ri % args.maps_every == 0:
            boxes = random_boxes(rng, w, h)
            k = int(rng.choice(KS))
            maps_recs.append(record(grey, PUSH | BOXES | (k << 4), boxes))
            maps_exp.append((label, km, f"0 {TG.score(km, 'boxes', k, s, boxes)}"))
        if args.finder_every and ri % args.finder_every == 0:
            f = 0 if real else int(rng.integers(0, 4))
            mode = int(rng.integers(0, 2))
            k = int(rng.choice(KS))
            form = ROWS if rng.random() < 0.5 else 0
            base = TG.score(km, "region", k, s)
            t = int(base + rng.integers(0, 2)) if rng.random() < 0.8 else int(rng.integers(-130, 129))
            recs.append(record(pix, FINDER | NOGATE | (BOXES if mode else 0) | (k << 4), boxes=[], finder=(f, 0)))
            exp.append(None)
            recs.append(record(pix, form | FINDER | (BOXES if mode else 0) | (k << 4), boxes=[], finder=(f, t)))
            exp.append(None)
            finder_pairs.append((len(recs) - 2, label, km, mode, k, t))
    bad = []
    if recs:
        batch = Path(td) / "r.bin"
        batch.write_bytes(b"".join(recs))
        got = run_driver(binary, ["--batch", bpath, batch])
        if len(got) != len(recs):
            return len(recs), 0, [(name, "-", f"{len(got)} output lines for {len(recs)} records", "")]
        for e, g in zip(exp, got):
            if e is not None and g != e[2]:
                bad.append((name, e[0], e[1], "" if real else f"C {g} twin {e[2]}"))
        for i, label, km, mode, k, t in finder_pairs:
            plain, gated = got[i], got[i + 1]
            v = [int(x) for x in plain.split()]
            rc, n = v[0], v[1]
            boxes = [tuple(v[3 + 5 * j: 7 + 5 * j]) for j in range(n)]
            if mode == 0:
                text = TG.score(km, "region", k, s) >= t
                want = plain if text else "0 0 0"
            elif rc:
                want = plain
            else:
                want = plain if TG.score(km, "boxes", k, s, boxes) >= t else "0 0 0"
            if gated != want:
                bad.append((name, label, f"finder mode {MODES[mode]} k {k}", "" if real else f"C {gated} want {want}"))
    nmaps = 0
    if maps_recs:
        mb = Path(td) / "m.bin"
        mb.write_bytes(b"".join(maps_recs))
        out = run_driver(binary, ["--maps", bpath, mb])
        for i, (label, km, want) in enumerate(maps_exp):
            line, mline = out[2 * i], out[2 * i + 1].split()
            hm, wm = int(mline[1]), int(mline[2])
            vals = np.array(mline[3:], np.int64)
            ok = line == want and (hm, wm) == km.shape and vals.size == hm * wm and (vals.reshape(hm, wm) == km).all()
            nmaps += 1
            if not ok:
                bad.append((name, label, "K map", "" if real else f"C {line} {hm}x{wm} twin {want} {km.shape}"))
    return len(recs), nmaps, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--regions", type=int, default=8, help="regions per random blob")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cal", type=int, default=6, help="calibration regions of random_int_layers")
    ap.add_argument("--shapes", default="", help="comma-separated shapes (default: every gate and #97 shape)")
    ap.add_argument("--wild", type=int, default=2, help="shapes with a wild-requantisation blob")
    ap.add_argument("--chains", type=int, default=4, help="random valid layer chains")
    ap.add_argument("--probes", type=int, default=1, help="probe blob pairs (strides 4 and 8)")
    ap.add_argument("--screen-every", type=int, default=4, help="every k-th blob also reads a whole screen (0: none)")
    ap.add_argument("--maps-every", type=int, default=2, help="compare the K map of every k-th region (0: none)")
    ap.add_argument("--finder-every", type=int, default=3, help="the finder mode on every k-th region (0: none)")
    ap.add_argument("--selftest", type=int, default=0, help="regions of the C self-test per blob (0: skip)")
    ap.add_argument("--selftest-blobs", type=int, default=2)
    ap.add_argument("--blob", action="append", default=[], help="[NAME=]FILE: more blobs (trained ones)")
    ap.add_argument("--cohort", action="append", default=[], help="pickled real regions (field image)")
    ap.add_argument("--cohort-limit", type=int, default=0, help="cases per cohort (0: all)")
    ap.add_argument("--only-files", action="store_true", help="only the --blob files (no random blobs)")
    ap.add_argument("--json", default="", help="write per-blob parity records here")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.only_files:
        blobs = [(n or Path(p).parent.name, Path(p).read_bytes(), None)
                 for n, _, p in (b.rpartition("=") for b in args.blob)]
    else:
        blobs = all_blobs(args)
    rng = np.random.default_rng(args.seed)
    real = cohort_regions(args.cohort, args.cohort_limit) if args.cohort else []
    nrec = nreg = nmaps = 0
    bad, doc = [], {}
    with tempfile.TemporaryDirectory(prefix="textgate-parity-") as td:
        if args.selftest:
            pick = [b for b in blobs if not b[0].startswith(("probe", "chain"))][::3][:max(0, args.selftest_blobs - 1)]
            pick += [b for b in blobs if b[0].startswith("chain")][:1]
            paths = []
            for name, blob, _ in pick[:args.selftest_blobs]:
                p = Path(td) / f"self_{name}.vtd"
                p.write_bytes(blob)
                paths.append(p)
            print(" ".join(run_driver(args.bin, ["--selftest", args.selftest, *paths])))
        for bi, (name, blob, probe) in enumerate(blobs):
            if real:
                imgs, is_real = real, True
            else:
                screen = None
                if args.screen_every and bi % args.screen_every == 0:
                    screen = (960, 544) if (bi // args.screen_every) % 2 == 0 else (480, 272)
                imgs = [(f"r{i}", g) for i, g in enumerate(regions(rng, args.regions, probe, screen))]
                is_real = False
            r, mp, b = check_blob(args.bin, name, blob, imgs, rng, td, args, is_real)
            nrec += r
            nreg += len(imgs)
            nmaps += mp
            bad += b
            doc[name] = {"c_equals_twin": not b, "regions": len(imgs), "records": r, "maps": mp,
                         "sha256": hashlib.sha256(blob).hexdigest(), "real": is_real}
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    if bad:
        for name, label, what, detail in bad[:12]:
            print(f"MISMATCH {name} {label}: {what} {detail}")
        print(f"TEXTGATE_PARITY_FAIL {len(bad)} of {nrec} records")
        return 1
    print(f"TEXTGATE_PARITY_OK {len(blobs)} blobs ({nreg} regions, {nrec} records, {nmaps} maps; "
          f"{time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

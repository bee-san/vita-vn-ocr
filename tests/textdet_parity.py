#!/usr/bin/env python3
"""Parity check: runtime/vocr_textdet.c against its Python twin synth/textdet_twin.py (the specification).

  textdet_parity.py --bin build/textdet/vocr_textdet_test [--seeds 2] [--regions 10] [--probe-regions 24]
                    [--selftest 12] [--maps-every 1] [--seed 0] [--blob FILE ...] [--shapes s4_c8,...]

Writes random blobs with train/textdet_export.py's functions (random_int_layers + pack) for every shape in
train/textdet_model.SHAPES and --seeds seeds each, plus "probe" blobs whose network passes the grey of every
fourth (eighth) pixel through to the head's maps, so designed regions drive the decode into its corners
(more than 64 components, vertical lines, merging runs, the 4-px drop, boxes at the borders; min_cells 1-4).
Every blob reads procedural regions (textdet_export.procedural_region, edge sizes from 1x1, odd sizes and
whole 960x544 / 480x272 screens; GREY and RGBA) through the C driver in the in-memory and row-callback forms
(and the finder mode, padded strides and odd arenas on some), and the twin; any difference in the head's four
maps (--maps), boxes, dark, vertical, order or truncated fails. --selftest N also runs the driver's own
invariants (vocr_textdet_test --selftest) on N regions of a few of the blobs. Prints
TEXTDET_PARITY_OK <runs> runs (<regions> regions, ...).
"""
import argparse
import hashlib
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_twin as TW  # noqa: E402

ROWS, FINDER, PADDED, ODD = 1, 2, 4, 8
EDGE = ((1, 1), (2, 1), (1, 2), (3, 3), (5, 7), (7, 5), (1, 40), (40, 1), (9, 13), (17, 31), (33, 8), (63, 65))


# ---------------------------------------------------------------- blobs

def random_blob(shape, seed, n_cal=6):
    sh, q = E.random_int_layers(shape, seed, n_cal=n_cal)
    return E.pack(sh, q, hashlib.sha256(f"random:{shape}:{seed}".encode()).digest())


def _layer(kind, k, stride, cin, cout, wq, bq, m, sh, relu=False, dil=1, head=False):
    return {"kind": kind, "k": k, "stride": stride, "dil": dil, "relu": relu, "cin": cin, "cout": cout,
            "head": head, "wq": np.asarray(wq, np.int8), "bq": np.asarray(bq, np.int64),
            "m": np.asarray(m, np.int64), "sh": np.asarray(sh, np.int64)}


def set_min_cells(blob, n):
    """The blob with header byte 21 (min kernel cells) set to n and the CRC recomputed."""
    b = bytearray(blob)
    b[21] = n
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    TW.Model(bytes(b))
    return bytes(b)


PROBE_VARIANTS = (
    (1, 1, 6, 6),       # ordinary boxes about x/4 + 6 half-px either side
    (2, -1, 2, 10),     # a negative far distance on bright cells
    (-1, 3, 12, -4),    # a negative near distance
    (0, 0, 4, 4),       # 4 half-px either side: boxes 4-5 px across, at the drop's edge
    (1, 1, 0, 0),       # dim cells under 4 px across: dropped
    (-2, 4, 0, 0),      # boxes pushed past the cell, off the region at the far border: clipped to nothing
)


def probe_blob(stride, variant=0):
    """A network that carries x = grey - 128 of pixel (s i, s j) to cell (i, j) (a centre-tap stem, then stride-2
    depthwise layers, all identity requantisation), then a 1x1 head: K = x, O = x - 64 (vertical where grey >
    192), A1 and A2 = (slope x + 4 off) / 4 with PROBE_VARIANTS[variant]."""
    one = 1 << 30
    centre = np.zeros((1, 1, 3, 3), np.int8)
    centre[0, 0, 1, 1] = 1
    layers = [_layer("conv", 3, 2, 1, 1, centre, [0], [one], [30])]
    for _ in range(1 if stride == 4 else 2):
        layers.append(_layer("dw", 3, 2, 1, 1, centre, [0], [one], [30]))
    slope1, slope2, off1, off2 = PROBE_VARIANTS[variant]
    w = np.array([1, 1, slope1, slope2], np.int8).reshape(4, 1, 1, 1)
    layers.append(_layer("conv", 1, 1, 1, 4, w, [0, -64, 4 * off1, 4 * off2], [one] * 4, [30, 30, 32, 32],
                         head=True))
    shape = {"name": f"probe_s{stride}_v{variant}", "stride": stride}
    return E.pack(shape, layers, hashlib.sha256(shape["name"].encode()).digest())


def wild_blob(shape_name, seed):
    """random_int_layers' layers with some channels' requantisation replaced by extremes the loader accepts:
    weights -128, biases over the whole int32 range, multipliers 1 .. 2^31 - 1, shifts 1 .. 62 (the int64
    requantisation's corners; most such channels saturate)."""
    rng = np.random.default_rng(seed)
    shape, q = E.random_int_layers(shape_name, seed, n_cal=2)
    for L in q:
        wild = rng.random(L["cout"]) < 0.35
        L["wq"] = L["wq"].copy()
        L["wq"][rng.random(L["wq"].shape) < 0.05] = -128
        n = int(wild.sum())
        L["bq"] = np.where(wild, rng.integers(-(1 << 31), 1 << 31, L["cout"]), L["bq"])
        L["m"] = np.where(wild, rng.integers(1, 1 << 31, L["cout"]), L["m"])
        L["sh"] = np.where(wild, rng.integers(1, 63, L["cout"]), L["sh"])
        if n:
            k = int(rng.integers(0, L["cout"]))          # one channel at the very edges
            L["bq"][k], L["m"][k], L["sh"][k] = (-(1 << 31), (1 << 31) - 1, 1) if seed % 2 else ((1 << 31) - 1, 1, 62)
    return E.pack(shape, q, hashlib.sha256(f"wild:{shape_name}:{seed}".encode()).digest())


def all_blobs(seeds, shapes, extra=(), n_cal=6, wild=2):
    """[(name, bytes)]: random blobs of every shape and seed, wild-requantisation blobs of `wild` shapes, probe
    blobs (strides 4 and 8, min_cells 1-4), --blob files."""
    out = []
    for i, shape in enumerate(shapes):
        for k in range(seeds):
            seed = 1000 * (k + 1) + i                   # distinct per shape: the _ko twins get their own weights
            out.append((f"{shape}_seed{seed}", random_blob(shape, seed, n_cal)))
    for k, shape in enumerate(shapes[::max(1, len(shapes) // max(1, wild))][:wild]):
        out.append((f"wild_{shape}_{k}", wild_blob(shape, 77 + k)))
    for stride in (4, 8):
        for v in range(len(PROBE_VARIANTS)):
            b = probe_blob(stride, v)
            out.append((f"probe_s{stride}_v{v}", b))
            mc = 1 + (v + (stride == 8)) % 4
            out.append((f"probe_s{stride}_v{v}_mc{mc}", set_min_cells(b, mc)))
    for p in extra:
        out.append((Path(p).name, Path(p).read_bytes()))
    return out


# ---------------------------------------------------------------- regions

def to_rgba(grey, rng):
    """An RGBA region whose grey is exactly `grey`: R = G = B = grey (the luma of (g, g, g) is g), or a random
    colour with that luma where one exists, and random alpha."""
    h, w = grey.shape
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = grey[..., None]
    rgba[..., 3] = rng.integers(0, 256, (h, w))
    if rng.random() < 0.5:                                  # colour: keep G, solve R, B close to the luma
        r = rng.integers(0, 256, (h, w)).astype(np.int64)
        b = rng.integers(0, 256, (h, w)).astype(np.int64)
        g = np.clip((256 * grey.astype(np.int64) + 128 - 77 * r - 29 * b) // 150, 0, 255)
        ok = ((77 * r + 150 * g + 29 * b) >> 8) == grey
        rgba[..., 0] = np.where(ok, r, grey)
        rgba[..., 1] = np.where(ok, g, grey)
        rgba[..., 2] = np.where(ok, b, grey)
    c = rgba.astype(np.int64)
    assert (((77 * c[..., 0] + 150 * c[..., 1] + 29 * c[..., 2]) >> 8) == grey).all()
    return rgba


def designed_region(rng, stride):
    """A region for the probe blobs, designed cell by cell: the grey of pixel (s i, s j) is cell (i, j)'s value
    (> 128 a kernel cell, > 192 a vertical vote), every other pixel is random (only the dark flags read it).
    Marks: dots, bars along and across, U shapes and combs (runs that merge components of the row above), cell
    diagonals (not 4-connected), checkerboards (the most runs and live components a row can have, and far more
    than 64 candidates), full rows and blocks (big sums), marks on the borders."""
    h, w = int(rng.integers(1, 400)), int(rng.integers(1, 900))
    s = stride
    hm, wm = -(-h // s), -(-w // s)
    cell = rng.integers(0, 129, (hm, wm))
    for _ in range(int(rng.integers(1, 30))):
        kind = int(rng.integers(0, 9))
        i, j = int(rng.integers(0, hm)), int(rng.integers(0, wm))
        lo = 193 if rng.random() < 0.3 else 129
        v = lambda shape: rng.integers(lo, 256, shape)  # noqa: E731
        if kind == 0:
            cell[i, j] = v(())
        elif kind == 1:                                     # a bar along the row
            a, b = int(rng.integers(1, 3)), int(rng.integers(1, 80))
            cell[i:i + a, j:j + b] = v(cell[i:i + a, j:j + b].shape)
        elif kind == 2:                                     # a bar down the column
            a, b = int(rng.integers(1, 60)), int(rng.integers(1, 3))
            cell[i:i + a, j:j + b] = v(cell[i:i + a, j:j + b].shape)
        elif kind == 3:                                     # a U: legs joined at the bottom
            a, b = int(rng.integers(2, 12)), int(rng.integers(2, 12))
            cell[i:i + a, j] = v(cell[i:i + a, j].shape)
            cell[i:i + a, min(wm - 1, j + b)] = v(cell[i:i + a, min(wm - 1, j + b)].shape)
            cell[min(hm - 1, i + a - 1), j:j + b + 1] = v(cell[min(hm - 1, i + a - 1), j:j + b + 1].shape)
        elif kind == 4:                                     # a comb: teeth joined by a bar below them
            a, n = int(rng.integers(2, 10)), int(rng.integers(2, 20))
            for t in range(n):
                cell[i:i + a, min(wm - 1, j + 2 * t)] = v(cell[i:i + a, min(wm - 1, j + 2 * t)].shape)
            cell[min(hm - 1, i + a), j:j + 2 * n] = v(cell[min(hm - 1, i + a), j:j + 2 * n].shape)
        elif kind == 5:                                     # a diagonal: corner-touching cells, apart
            for t in range(int(rng.integers(2, 30))):
                cell[min(hm - 1, i + t), min(wm - 1, j + t)] = v(())
        elif kind == 6:                                     # a checkerboard patch
            a, b = int(rng.integers(2, 40)), int(rng.integers(2, 120))
            yy, xx = np.mgrid[i:min(hm, i + a), j:min(wm, j + b)]
            m = (yy + xx) % 2 == 0
            cell[yy[m], xx[m]] = v(int(m.sum()))
        elif kind == 7:                                     # a solid block
            a, b = int(rng.integers(1, 30)), int(rng.integers(1, 100))
            cell[i:i + a, j:j + b] = v(cell[i:i + a, j:j + b].shape)
        else:                                               # the border rows and columns
            cell[0, :] = np.where(rng.random(wm) < 0.5, v(wm), cell[0, :])
            cell[:, -1] = np.where(rng.random(hm) < 0.5, v(hm), cell[:, -1])
    if rng.random() < 0.08:                                 # all kernel: one component of every cell
        cell[:] = v(cell.shape)
    elif rng.random() < 0.08:                               # a full checkerboard
        cell[(np.add.outer(np.arange(hm), np.arange(wm)) % 2) == 0] = 200
    g = rng.integers(0, 256, (h, w))
    g[::s, ::s] = cell
    return g.astype(np.uint8)


def regions_for(rng, n, probe_stride=None, screen=None):
    """n grey regions: edge sizes, then procedural (or, for a probe blob, designed) regions and odd sizes; with
    screen = (w, h) the last one is a whole screen holding four such regions."""
    out = [rng.integers(0, 256, (h, w), dtype=np.uint8) for w, h in EDGE[:max(0, min(len(EDGE), n // 3))]]
    while len(out) < n - (screen is not None):
        if rng.random() < 0.1:
            w, h = int(rng.integers(1, 90)) * 2 + 1, int(rng.integers(1, 60)) * 2 + 1   # odd sizes
            out.append(rng.integers(0, 256, (h, w), dtype=np.uint8))
        else:
            out.append(E.procedural_region(rng) if probe_stride is None else designed_region(rng, probe_stride))
    if screen is not None:
        w, h = screen
        g = rng.integers(0, 256, (h, w)).astype(np.uint8) if probe_stride else np.zeros((h, w), np.uint8)
        for _ in range(4):
            p = E.procedural_region(rng) if probe_stride is None else designed_region(rng, probe_stride)
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            if probe_stride:                                # on the cell grid
                y, x = y - y % probe_stride, x - x % probe_stride
            ph, pw = min(p.shape[0], h - y), min(p.shape[1], w - x)
            g[y:y + ph, x:x + pw] = p[:ph, :pw]
        out.append(g)
    return out


# ---------------------------------------------------------------- the C driver and the twin

def write_batch(path, cases):
    """cases: [(img uint8 HxW or HxWx4, flags)] -> the driver's records (struct "<4I" w, h, bpp, flags; pixels)."""
    with open(path, "wb") as f:
        for img, flags in cases:
            h, w = img.shape[:2]
            bpp = 1 if img.ndim == 2 else 4
            f.write(struct.pack("<4I", w, h, bpp, flags))
            f.write(np.ascontiguousarray(img).tobytes())


def twin_line(model, img):
    lines, trunc = TW.find_lines(model, img)
    return f"0 {len(lines)} {int(trunc)}" + "".join(f" {x} {y} {w} {h} {d} {v}" for x, y, w, h, d, v in lines)


def finder_line(model, img):
    """The finder mode's expected line: the twin's horizontal lines (find_lines_ex), vertical printed 0."""
    boxes, dark, trunc = TW.find_lines_ex(model, img)
    return f"0 {len(boxes)} {int(trunc)}" + "".join(f" {x} {y} {w} {h} {d} 0" for (x, y, w, h), d in zip(boxes, dark))


def run_driver(binary, args, timeout=3600):
    p = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    if p.returncode:
        raise RuntimeError(f"{binary} {' '.join(map(str, args))}: rc {p.returncode}\n{p.stderr[-3000:]}")
    return p.stdout


def check_blob(binary, name, blob, imgs, rng, td, maps_every=1):
    """Every region of imgs through C and the twin. Returns (runs, maps, mismatches)."""
    model = TW.Model(blob)
    bpath = Path(td) / f"{name}.vtd"
    bpath.write_bytes(blob)
    cases, exp = [], []
    for k, g in enumerate(imgs):
        img = to_rgba(g, rng) if rng.random() < 0.5 else g
        base = ODD * int(rng.random() < 0.3) | PADDED * int(rng.random() < 0.3)
        line = twin_line(model, img)
        for form in (0, ROWS):
            cases.append((img, base | form))
            exp.append((k, form, line))
        if rng.random() < 0.3:
            cases.append((img, base | FINDER | int(rng.integers(0, 2))))
            exp.append((k, "finder", finder_line(model, img)))
    batch = Path(td) / f"{name}.bin"
    write_batch(batch, cases)
    got = run_driver(binary, ["--batch", bpath, batch]).splitlines()
    bad = []
    if len(got) != len(cases):
        return len(cases), 0, [(name, -1, f"{len(got)} output lines for {len(cases)} records", "")]
    for (k, form, e), g in zip(exp, got):
        if g != e:
            bad.append((name, k, f"form {form}, shape {imgs[k].shape}: C  {g[:400]}", f"py {e[:400]}"))
    nmaps = 0
    if maps_every:
        sel = imgs[::maps_every]
        mb = Path(td) / f"{name}_maps.bin"
        write_batch(mb, [(g, 0) for g in sel])
        out = run_driver(binary, ["--maps", bpath, mb]).splitlines()
        for i, g in enumerate(sel):
            rc, hm, wm = map(int, out[2 * i + 1].split())
            ref = TW.run_net(model, g)
            vals = np.array(out[2 * i].split()[1:], np.int64)
            ok = rc == 0 and vals.size == 4 * hm * wm and (hm, wm) == ref.shape[1:] and \
                (vals.reshape(hm, wm, 4).transpose(2, 0, 1) == ref).all()
            nmaps += 1
            if not ok:
                bad.append((name, i * maps_every, f"maps differ (rc {rc}, {hm}x{wm} vs {ref.shape[1:]})", ""))
    return len(cases), nmaps, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--regions", type=int, default=10, help="regions per random blob")
    ap.add_argument("--probe-regions", type=int, default=24, help="regions per probe blob")
    ap.add_argument("--selftest", type=int, default=0, help="regions of the C self-test per blob (0: skip)")
    ap.add_argument("--selftest-blobs", type=int, default=3, help="blobs the self-test reads")
    ap.add_argument("--maps-every", type=int, default=1, help="compare the maps of every k-th region (0: none)")
    ap.add_argument("--screen-every", type=int, default=1,
                    help="every k-th blob also reads a whole 960x544 or 480x272 screen (0: none)")
    ap.add_argument("--wild", type=int, default=2, help="shapes with a wild-requantisation blob")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cal", type=int, default=6, help="calibration regions of random_int_layers")
    ap.add_argument("--shapes", default="", help="comma-separated subset of SHAPES (default: all)")
    ap.add_argument("--blob", action="append", default=[], help="more blobs (e.g. trained sweep blobs)")
    args = ap.parse_args(argv)
    t0 = time.time()
    shapes = [s for s in args.shapes.split(",") if s] or list(TM.SHAPES)
    blobs = all_blobs(args.seeds, shapes, args.blob, args.n_cal, args.wild)
    rng = np.random.default_rng(args.seed)
    runs = regions = nmaps = 0
    bad = []
    with tempfile.TemporaryDirectory(prefix="textdet-parity-") as td:
        if args.selftest:
            rnd = [b for b in blobs if not b[0].startswith(("probe", "wild"))]
            probes = [b for b in blobs if b[0].startswith("probe")]
            wild = [b for b in blobs if b[0].startswith("wild")]
            order = []
            for k in range(max(len(rnd), len(probes), len(wild))):   # a small random blob, a probe, ... spread out
                for group in (rnd[::max(1, len(rnd) // 6)], probes[::max(1, len(probes) // 6)], wild):
                    if k < len(group):
                        order.append(group[k])
            paths = []
            for name, blob in order[:args.selftest_blobs]:
                p = Path(td) / f"self_{name}.vtd"
                p.write_bytes(blob)
                paths.append(p)
            out = run_driver(args.bin, ["--selftest", args.selftest, *paths])
            print(out.strip())
        for bi, (name, blob) in enumerate(blobs):
            probe = name.startswith("probe")
            n = args.probe_regions if probe else args.regions
            screen = None
            if args.screen_every and bi % args.screen_every == 0:
                screen = (960, 544) if (bi // args.screen_every) % 2 == 0 else (480, 272)
            imgs = regions_for(rng, n, TW.Model(blob).stride if probe else None, screen)
            r, mp, b = check_blob(args.bin, name, blob, imgs, rng, td, args.maps_every)
            runs += r
            regions += len(imgs)
            nmaps += mp
            bad += b
    if bad:
        for name, k, c, p in bad[:12]:
            print(f"MISMATCH {name} region {k}: {c}\n    {p}")
        print(f"TEXTDET_PARITY_FAIL {len(bad)} of {runs} runs")
        return 1
    print(f"TEXTDET_PARITY_OK {runs} runs ({regions} regions, {len(blobs)} blobs, {nmaps} maps; "
          f"{time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

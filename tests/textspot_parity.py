#!/usr/bin/env python3
"""Parity check: runtime/vocr_textspot.c against its Python twin synth/textspot_twin.py (the specification).

  textspot_parity.py --bin build/textspot/vocr_textspot_test [--seeds 1] [--frames 4] [--probe-frames 6]
                     [--full-every 4] [--selftest 12] [--selftest-blobs 3] [--maps-every 2] [--seed 0]
                     [--shapes s4_c8,...] [--blob FILE ...]

Blobs: random VTDT networks (train/textdet_export.random_int_layers + pack) of every shape of
eval/textspot_protocol.json's family B and of train/textdet_model.SHAPES (#97's runs), --seeds each, plus
wild-requantisation networks and #97's probe networks (tests/textdet_parity.probe_blob: the grey of every
fourth / eighth network-input pixel carried to the head's maps, min_cells 1-4), each wrapped by
textspot_twin.pack at half_above 0 and 600 (and some with conf_min > 1); --blob adds VTSP files (or VTDT ones,
wrapped both ways).

Frames: procedural full frames (textdet_export.procedural_region tiles; whole 960x544, 480x272, 725x544 and
870x544 screens on every --full-every-th blob), edge and odd sizes (widths about half_above too); for the probe
networks frames designed cell by cell (checkerboards, combs, U shapes, diagonals, bars, blocks: hundreds of
components, so more than 64 candidates), drawn at the network's input scale and doubled when the frame is
halved, so the halving is exact on them. GREY and RGBA; the in-memory and row-callback forms, the
VOCR_TEXTSPOT_NO_DARK flag (everything but dark compared, dark 0), padded strides, odd arenas and the arena for
every frame up to the size. The C driver and the twin (textspot_twin.spot; maps: textdet_twin.run_net on
textspot_twin.net_input) must agree exactly on the maps, boxes (x, y, w, h, vertical, conf, dark, block), blocks,
order, pick and truncated. --selftest N also runs the driver's own invariants (vocr_textspot_test --selftest) on
N frames of a few blobs. Prints TEXTSPOT_PARITY_OK <runs> (<frames> frames, ...).
"""
import argparse
import json
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_export as E  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_parity as TP  # noqa: E402
import textdet_twin as TD  # noqa: E402
import textspot_twin as S  # noqa: E402

ROWS, NO_DARK, PADDED, ODD, MOST = 1, 2, 4, 8, 16       # vocr_textspot_test's record flags
SCREENS = ((960, 544), (480, 272), (725, 544), (870, 544))
EDGE = ((1, 1), (2, 1), (1, 2), (3, 3), (5, 7), (7, 5), (1, 40), (40, 1), (601, 3), (599, 5), (63, 65))


def protocol_shapes():
    """Family B's shapes (h2 and f1) of the frozen protocol, then the rest of SHAPES (#97's)."""
    p = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text(encoding="utf-8"))
    fam = p["candidates"]["family_b"]["shapes"]
    out = []
    for s in [*fam["h2"], *fam["f1"], *TM.SHAPES]:
        if s not in out:
            out.append(s)
    return out


def set_conf_min(blob, conf_min):
    b = bytearray(blob)
    b[18] = conf_min
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    S.Model(bytes(b))
    return bytes(b)


def all_blobs(seeds, shapes, extra=(), wild=2, n_cal=6):
    """[(name, VTSP bytes)]: every VTDT network wrapped at half_above 0 and 600, every fourth with a conf_min > 1."""
    rng = np.random.default_rng(1234)
    nets = []
    for i, shape in enumerate(shapes):
        for k in range(seeds):
            nets.append((f"{shape}_seed{1000 * (k + 1) + i}", TP.random_blob(shape, 1000 * (k + 1) + i, n_cal)))
    for k, shape in enumerate(shapes[::max(1, len(shapes) // max(1, wild))][:wild]):
        nets.append((f"wild_{shape}_{k}", TP.wild_blob(shape, 77 + k)))
    for stride in (4, 8):
        for v in range(len(TP.PROBE_VARIANTS)):
            b = TP.probe_blob(stride, v)
            mc = 1 + (v + (stride == 8)) % 4
            nets.append((f"probe_s{stride}_v{v}", b if v % 2 else TP.set_min_cells(b, mc)))
    out = []
    for j, (name, vtdt) in enumerate(nets):
        for ha in (0, 600):
            cm = int(rng.integers(2, 48)) if (j + ha // 600) % 4 == 3 else 1
            out.append((f"{name}_h{ha}_c{cm}", S.pack(vtdt, ha, cm)))
    for p in extra:
        raw = Path(p).read_bytes()
        if raw[:4] == S.MAGIC:
            out.append((Path(p).name, raw))
        else:
            for ha in (0, 600):
                out.append((f"{Path(p).name}_h{ha}", S.pack(raw, ha, 1)))
    return out


# ---------------------------------------------------------------- frames

def tiled(rng, w, h):
    """A w x h grey frame tiled with procedural regions (text-like stripes and blocks on plain grounds)."""
    g = np.full((h, w), int(rng.integers(0, 256)), np.uint8)
    y = 0
    while y < h:
        x, rh = 0, 0
        while x < w:
            p = E.procedural_region(rng)
            ph, pw = min(p.shape[0], h - y), min(p.shape[1], w - x)
            g[y:y + ph, x:x + pw] = p[:ph, :pw]
            x += pw
            rh = max(rh, ph)
        y += rh
    return g


def designed(rng, stride, h, w):
    """An h x w network input designed cell by cell for the probe networks (textdet_parity.designed_region's
    marks, at any size): the grey of pixel (s i, s j) is cell (i, j)'s value (> 128 a kernel cell, > 192 a
    vertical vote), the other pixels random."""
    s = stride
    hm, wm = -(-h // s), -(-w // s)
    cell = rng.integers(0, 129, (hm, wm))
    for _ in range(int(rng.integers(1, 30 + hm * wm // 400))):
        kind = int(rng.integers(0, 9))
        i, j = int(rng.integers(0, hm)), int(rng.integers(0, wm))
        lo = 193 if rng.random() < 0.3 else 129
        v = lambda shape: rng.integers(lo, 256, shape)  # noqa: E731
        if kind == 0:
            cell[i, j] = v(())
        elif kind == 1:
            a, b = int(rng.integers(1, 3)), int(rng.integers(1, 80))
            cell[i:i + a, j:j + b] = v(cell[i:i + a, j:j + b].shape)
        elif kind == 2:
            a, b = int(rng.integers(1, 60)), int(rng.integers(1, 3))
            cell[i:i + a, j:j + b] = v(cell[i:i + a, j:j + b].shape)
        elif kind == 3:
            a, b = int(rng.integers(2, 12)), int(rng.integers(2, 12))
            cell[i:i + a, j] = v(cell[i:i + a, j].shape)
            cell[i:i + a, min(wm - 1, j + b)] = v(cell[i:i + a, min(wm - 1, j + b)].shape)
            cell[min(hm - 1, i + a - 1), j:j + b + 1] = v(cell[min(hm - 1, i + a - 1), j:j + b + 1].shape)
        elif kind == 4:
            a, n = int(rng.integers(2, 10)), int(rng.integers(2, 20))
            for t in range(n):
                cell[i:i + a, min(wm - 1, j + 2 * t)] = v(cell[i:i + a, min(wm - 1, j + 2 * t)].shape)
            cell[min(hm - 1, i + a), j:j + 2 * n] = v(cell[min(hm - 1, i + a), j:j + 2 * n].shape)
        elif kind == 5:
            for t in range(int(rng.integers(2, 30))):
                cell[min(hm - 1, i + t), min(wm - 1, j + t)] = v(())
        elif kind == 6:
            a, b = int(rng.integers(2, 40)), int(rng.integers(2, 120))
            yy, xx = np.mgrid[i:min(hm, i + a), j:min(wm, j + b)]
            m = (yy + xx) % 2 == 0
            cell[yy[m], xx[m]] = v(int(m.sum()))
        elif kind == 7:
            a, b = int(rng.integers(1, 30)), int(rng.integers(1, 100))
            cell[i:i + a, j:j + b] = v(cell[i:i + a, j:j + b].shape)
        else:
            cell[0, :] = np.where(rng.random(wm) < 0.5, v(wm), cell[0, :])
            cell[:, -1] = np.where(rng.random(hm) < 0.5, v(hm), cell[:, -1])
    if rng.random() < 0.06:
        cell[:] = 200
    elif rng.random() < 0.1:
        cell[(np.add.outer(np.arange(hm), np.arange(wm)) % 2) == 0] = rng.integers(129, 256)
    g = rng.integers(0, 256, (h, w))
    g[::s, ::s] = cell
    return g.astype(np.uint8)


def probe_frame(rng, model, w, h):
    """A w x h frame whose network input is designed(): drawn at the input scale and doubled when it is halved."""
    f = S.scale_of(model, w)
    if f == 1:
        return designed(rng, model.stride, h, w)
    d = designed(rng, model.stride, -(-h // 2), -(-w // 2))
    return np.repeat(np.repeat(d, 2, 0), 2, 1)[:h, :w]


def frames_for(rng, model, n, probe, screen):
    """n frames: edge sizes, then procedural (or designed) frames of random and odd sizes; screen = (w, h) adds a
    whole screen."""
    sizes = list(EDGE[:max(0, min(len(EDGE), n // 3))])
    while len(sizes) < n:
        r = rng.random()
        if r < 0.15:
            sizes.append((int(rng.integers(1, 120)) * 2 + 1, int(rng.integers(1, 80)) * 2 + 1))
        elif r < 0.3:
            sizes.append((int(rng.integers(560, 680)), int(rng.integers(1, 120))))
        else:
            sizes.append((int(rng.integers(8, 720)), int(rng.integers(8, 300))))
    if screen is not None:
        sizes.append(screen)
    out = []
    for w, h in sizes:
        if probe:
            out.append(probe_frame(rng, model, w, h))
        elif w * h < 64 or rng.random() < 0.1:
            out.append(rng.integers(0, 256, (h, w), dtype=np.uint8))
        else:
            out.append(tiled(rng, w, h))
    return out


# ---------------------------------------------------------------- the C driver and the twin

def write_batch(path, cases):
    """cases: [(img uint8 HxW or HxWx4, flags)] -> the driver's records ("<4I" w, h, bpp, flags; pixels)."""
    with open(path, "wb") as f:
        for img, flags in cases:
            h, w = img.shape[:2]
            f.write(struct.pack("<4I", w, h, 1 if img.ndim == 2 else 4, flags))
            f.write(np.ascontiguousarray(img).tobytes())


def twin_line(r, dark=True):
    out = [0, len(r.boxes), len(r.blocks), r.pick, r.truncated]
    for b in r.boxes:
        out += [b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark if dark else 0, b.block]
    for b in r.blocks:
        out += [b.x, b.y, b.w, b.h, b.vertical, b.first, b.count]
    return " ".join(map(str, out))


def run_driver(binary, args, timeout=3600):
    p = subprocess.run([*str(binary).split(), *map(str, args)], capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    if p.returncode:
        raise RuntimeError(f"{binary} {' '.join(map(str, args))}: rc {p.returncode}\n{p.stderr[-3000:]}")
    return p.stdout


class Coverage:
    def __init__(self):
        self.truncated = self.halved = self.vertical = self.boxes = self.blocks = self.multi = self.empty = 0
        self.nopick_ok = self.conf_min = 0

    def add(self, model, img, r):
        self.truncated += r.truncated
        self.halved += S.scale_of(model, img.shape[1]) == 2
        self.vertical += sum(b.vertical for b in r.boxes)
        self.boxes += len(r.boxes)
        self.blocks += len(r.blocks)
        self.multi += sum(b.count > 1 for b in r.blocks)
        self.empty += not r.boxes
        self.conf_min += model.conf_min > 1

    def __str__(self):
        return (f"{self.truncated} truncated, {self.halved} halved, {self.empty} empty, {self.boxes} boxes "
                f"({self.vertical} vertical) in {self.blocks} blocks ({self.multi} of several boxes)")


def check_blob(binary, name, blob, imgs, rng, td, cov, maps_every=1):
    """Every frame of imgs through C and the twin. Returns (runs, maps, mismatches)."""
    model = S.Model(blob)
    bpath = Path(td) / f"{name}.vtsp"
    bpath.write_bytes(blob)
    cases, exp = [], []
    for k, g in enumerate(imgs):
        img = TP.to_rgba(g, rng) if rng.random() < 0.5 else g
        r = S.spot(model, img)
        cov.add(model, img, r)
        line, line_nd = twin_line(r), twin_line(r, dark=False)
        base = ODD * int(rng.random() < 0.3) | PADDED * int(rng.random() < 0.3) | MOST * int(rng.random() < 0.3)
        for form in (0, ROWS):
            cases.append((img, base | form))
            exp.append((k, form, line))
        if rng.random() < 0.5:
            form = int(rng.integers(0, 2))
            cases.append((img, base | NO_DARK | form))
            exp.append((k, f"nodark{form}", line_nd))
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
            ref = TD.run_net(model.net, S.net_input(model, g)[0])
            vals = np.array(out[2 * i].split()[1:], np.int64)
            ok = rc == 0 and vals.size == 4 * hm * wm and (hm, wm) == ref.shape[1:] and \
                (vals.reshape(hm, wm, 4).transpose(2, 0, 1) == ref).all()
            nmaps += 1
            if not ok:
                bad.append((name, i * maps_every, f"maps differ (rc {rc}, {hm}x{wm} vs {ref.shape[1:]})", ""))
    return len(cases), nmaps, bad


def check_loader(binary, blobs, td):
    """The C loader accepts every blob and refuses damaged copies the twin refuses (a few edits per blob)."""
    rng = np.random.default_rng(5)
    paths, want = [], []
    for name, blob in blobs[:6]:
        for k in range(8):
            b = bytearray(blob)
            at = int(rng.integers(0, len(b)))
            b[at] ^= int(rng.integers(1, 256))
            if k % 2:                                   # fix the outer CRC: the inner checks must refuse it
                struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
            try:
                S.Model(bytes(b))
                ok = 0
            except ValueError:
                ok = -5
            p = Path(td) / f"load_{name}_{k}.bin"
            p.write_bytes(bytes(b))
            paths.append(p)
            want.append(ok)
        p = Path(td) / f"load_{name}_ok.bin"
        p.write_bytes(blob)
        paths.append(p)
        want.append(0)
    got = [int(v) for v in run_driver(binary, ["--load", *paths]).split()]
    return [(str(p), w, g) for p, w, g in zip(paths, want, got) if w != g]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="the driver (may carry a prefix, e.g. 'qemu-arm -cpu cortex-a9 ./x')")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--frames", type=int, default=4, help="frames per random blob")
    ap.add_argument("--probe-frames", type=int, default=6, help="frames per probe blob")
    ap.add_argument("--full-every", type=int, default=4, help="every k-th blob also reads a whole screen (0: none)")
    ap.add_argument("--selftest", type=int, default=0, help="frames of the C self-test per blob (0: skip)")
    ap.add_argument("--selftest-blobs", type=int, default=3)
    ap.add_argument("--maps-every", type=int, default=2, help="compare the maps of every k-th frame (0: none)")
    ap.add_argument("--wild", type=int, default=2, help="shapes with a wild-requantisation network")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cal", type=int, default=6)
    ap.add_argument("--shapes", default="", help="comma-separated shapes (default: family B, then SHAPES)")
    ap.add_argument("--blob", action="append", default=[], help="more blobs (VTSP, or VTDT wrapped at 0 and 600)")
    args = ap.parse_args(argv)
    t0 = time.time()
    shapes = [s for s in args.shapes.split(",") if s] or protocol_shapes()
    blobs = all_blobs(args.seeds, shapes, args.blob, args.wild, args.n_cal)
    rng = np.random.default_rng(args.seed)
    runs = nframes = nmaps = 0
    bad = []
    cov = Coverage()
    with tempfile.TemporaryDirectory(prefix="textspot-parity-") as td:
        if args.selftest:
            rnd = [b for b in blobs if not b[0].startswith(("probe", "wild"))]
            probes = [b for b in blobs if b[0].startswith("probe")]
            wild = [b for b in blobs if b[0].startswith("wild")]
            order = []
            for k in range(max(len(rnd), len(probes), len(wild))):
                for grp in (rnd[::max(1, len(rnd) // 6)], probes[::max(1, len(probes) // 6)], wild):
                    if k < len(grp):
                        order.append(grp[k])
            paths = []
            for name, blob in order[:args.selftest_blobs]:
                p = Path(td) / f"self_{name}.vtsp"
                p.write_bytes(blob)
                paths.append(p)
            print(run_driver(args.bin, ["--selftest", args.selftest, *paths]).strip())
        lbad = check_loader(args.bin, blobs, td)
        for name, w, g in lbad[:12]:
            bad.append((name, -1, f"loader: C {g}", f"twin {w}"))
        for bi, (name, blob) in enumerate(blobs):
            model = S.Model(blob)
            probe = name.startswith("probe")
            screen = SCREENS[(bi // args.full_every) % len(SCREENS)] if args.full_every and bi % args.full_every == 0 \
                else None
            imgs = frames_for(rng, model, args.probe_frames if probe else args.frames, probe, screen)
            r, mp, b = check_blob(args.bin, name, blob, imgs, rng, td, cov, args.maps_every)
            runs += r
            nframes += len(imgs)
            nmaps += mp
            bad += b
    if bad:
        for name, k, c, p in bad[:12]:
            print(f"MISMATCH {name} frame {k}: {c}\n    {p}")
        print(f"TEXTSPOT_PARITY_FAIL {len(bad)} of {runs} runs")
        return 1
    print(f"coverage: {cov}")
    print(f"TEXTSPOT_PARITY_OK {runs} runs ({nframes} frames, {len(blobs)} blobs, {nmaps} maps; "
          f"{time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Parity check: runtime/vocr_textspot2.c against its Python twin synth/textspot2_twin.py (the specification), for
VTSP version 1 and 2 (d2s) blobs; version 1 also against textspot's frozen runtime (runtime/vocr_textspot.c).

  textspot2_parity.py --bin build/textspot2/vocr_textspot2_test [--v1-bin build/textspot2/vocr_textspot_test]
                      [--seeds 1] [--frames 4] [--probe-frames 6] [--full-every 4] [--selftest 12]
                      [--selftest-blobs 4] [--maps-every 2] [--seed 0] [--shapes S,...] [--d2s-shapes S,...]
                      [--blob FILE ...]

Blobs, each wrapped at half_above 0 and 600 (every fourth with a conf_min > 1):
  - version 1: random VTDT networks (train/textdet_export.random_int_layers) of every "kroa" shape of
    eval/textspot2_protocol.json plus stride-8 and small ones of train/textdet_model.SHAPES, a wild-requantisation
    network (tests/textdet_parity.wild_blob), #97's probe networks at strides 4 and 8 (min cells 1-4);
  - version 2: random d2s networks (train/textspot2_export.random_int_layers) of every "kroa_x2" shape, a
    wild-requantisation d2s network, and d2s probe networks (probe_vtdt: sub-cell (dy, dx) of cell (i, j) carries
    the grey of network-input pixel (8 i + 4 dy, 8 j + 4 dx) to its four maps, as #97's stride-4 probe carries
    pixel (4 I, 4 J) of decode cell (I, J): so the twin of a d2s probe must give exactly the result of the
    stride-4 probe of the same variant, which is checked on every probe frame);
  - --blob adds VTSP files.
Frames: tests/textspot_parity's procedural full frames (960x544, 480x272, 725x544, 870x544 on every
--full-every-th blob), edge and odd sizes, and for the probes frames designed cell by cell at stride 4 (hundreds
of components: more than 64 candidates, truncated), GREY and RGBA; the in-memory and row-callback forms,
VOCR_TEXTSPOT_NO_DARK, padded strides, odd arenas, the arena for every frame up to the size. The C driver and the
twin (textspot2_twin.spot; maps: textspot2_twin.run_maps) must agree exactly on the maps the decode reads, the
boxes, blocks, order, pick and truncated flag; with --v1-bin every version-1 batch must also print exactly what
runtime/vocr_textspot_test prints for it, and every version-1 blob's arena (for the frame and up to it) must be
that driver's. The C loader must refuse exactly the damaged copies the twin refuses. --selftest N also runs the
driver's own invariants on N frames of a few blobs of both versions. Prints TEXTSPOT2_PARITY_OK <runs> runs
(<frames> frames, ...).
"""
import argparse
import hashlib
import struct
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
import textdet_parity as TP  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_twin as T2  # noqa: E402
import textspot_parity as P  # noqa: E402
import textspot_twin as S  # noqa: E402

ROWS, NO_DARK, PADDED, ODD, MOST = P.ROWS, P.NO_DARK, P.PADDED, P.ODD, P.MOST
SCREENS = P.SCREENS
V1_EXTRA = ("s8_c8", "s8_c24n4", "s4_c4")      # version-1 strides 8 and a small stride 4, beside the protocol's


def v1_shapes():
    S2.register()
    return [*(k for k, v in S2.SHAPES.items() if v["head"] == "kroa"), *V1_EXTRA]


def d2s_shapes():
    return list(S2.D2S)


# ---------------------------------------------------------------- d2s blobs

def set_min_cells(vtdt, n):
    """A network blob (either version) with header byte 21 (min kernel cells) set to n and its CRC recomputed."""
    b = bytearray(vtdt)
    b[21] = n
    struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
    return bytes(b)


def probe_vtdt(variant=0):
    """The d2s probe network: a centre-tap stem and stride-2 depthwise layer (the pixel (4 i, 4 j) at stride 4), a
    3x3 stride-2 conv to 4 channels whose channel 2 dy + dx taps (1 + dy, 1 + dx) (the pixel (8 i + 4 dy,
    8 j + 4 dx) at stride 8), all identity requantisation, then the 1x1 head: channel 4 (2 dy + dx) + m is
    tests/textdet_parity.probe_blob's map m of that channel (K = x, O = x - 64, A1 and A2 = (slope x + 4 off) / 4
    with PROBE_VARIANTS[variant])."""
    one = 1 << 30
    centre = np.zeros((1, 1, 3, 3), np.int8)
    centre[0, 0, 1, 1] = 1
    taps = np.zeros((4, 1, 3, 3), np.int8)
    for dy in (0, 1):
        for dx in (0, 1):
            taps[2 * dy + dx, 0, 1 + dy, 1 + dx] = 1
    layers = [TP._layer("conv", 3, 2, 1, 1, centre, [0], [one], [30]),
              TP._layer("dw", 3, 2, 1, 1, centre, [0], [one], [30]),
              TP._layer("conv", 3, 2, 1, 4, taps, [0] * 4, [one] * 4, [30] * 4)]
    slope1, slope2, off1, off2 = TP.PROBE_VARIANTS[variant]
    w = np.zeros((16, 4, 1, 1), np.int8)
    bias, mult, shift = [], [], []
    for dy in (0, 1):
        for dx in (0, 1):
            q = 2 * dy + dx
            for mp, (wt, b, sh) in enumerate(((1, 0, 30), (1, -64, 30), (slope1, 4 * off1, 32),
                                              (slope2, 4 * off2, 32))):
                w[T2.channel(dy, dx, mp), q, 0, 0] = wt
                bias.append(b)
                mult.append(one)
                shift.append(sh)
    layers.append(TP._layer("conv", 1, 1, 4, 16, w, bias, mult, shift, head=True))
    shape = {"name": f"probe_d2s_v{variant}", "stride": 8}
    return X.pack(shape, layers, hashlib.sha256(shape["name"].encode()).digest())


def wild_vtdt(shape_name, seed):
    """tests/textdet_parity.wild_blob for a d2s shape: random_int_layers with extreme requantisation corners."""
    rng = np.random.default_rng(seed)
    shape, q = X.random_int_layers(shape_name, seed, n_cal=2)
    for L in q:
        wild = rng.random(L["cout"]) < 0.35
        L["wq"] = L["wq"].copy()
        L["wq"][rng.random(L["wq"].shape) < 0.05] = -128
        n = int(wild.sum())
        L["bq"] = np.where(wild, rng.integers(-(1 << 31), 1 << 31, L["cout"]), L["bq"])
        L["m"] = np.where(wild, rng.integers(1, 1 << 31, L["cout"]), L["m"])
        L["sh"] = np.where(wild, rng.integers(1, 63, L["cout"]), L["sh"])
        if n:
            k = int(rng.integers(0, L["cout"]))
            L["bq"][k], L["m"][k], L["sh"][k] = (-(1 << 31), (1 << 31) - 1, 1) if seed % 2 else ((1 << 31) - 1, 1, 62)
    return X.pack(shape, q, hashlib.sha256(f"wild:{shape_name}:{seed}".encode()).digest())


def nets(seeds, shapes1, shapes2, wild=1, n_cal=6):
    """[(name, network blob)]: version-1 VTDT networks, then d2s ones (the VTSP version follows the outputs field)."""
    out = []
    for i, shape in enumerate(shapes1):
        for k in range(seeds):
            out.append((f"{shape}_seed{1000 * (k + 1) + i}", TP.random_blob(shape, 1000 * (k + 1) + i, n_cal)))
    for i, shape in enumerate(shapes2):
        for k in range(seeds):
            out.append((f"{shape}_seed{2000 * (k + 1) + i}", X.random_vtdt(shape, 2000 * (k + 1) + i, n_cal)))
    for k, shape in enumerate(shapes1[:wild]):
        out.append((f"wild_{shape}_{k}", TP.wild_blob(shape, 77 + k)))
    for k, shape in enumerate(shapes2[:wild]):
        out.append((f"wild_{shape}_{k}", wild_vtdt(shape, 88 + k)))
    for v in range(len(TP.PROBE_VARIANTS)):
        mc = 1 + v % 4
        for stride in (4, 8):
            b = TP.probe_blob(stride, v)
            out.append((f"probe_s{stride}_v{v}", b if v % 2 else TP.set_min_cells(b, 1 + (v + (stride == 8)) % 4)))
        b = probe_vtdt(v)
        out.append((f"probe_d2s_v{v}", b if v % 2 else set_min_cells(b, mc)))
    return out


def all_blobs(seeds, shapes1, shapes2, extra=(), wild=1, n_cal=6):
    """[(name, VTSP bytes)]: every network wrapped at half_above 0 and 600, every fourth with a conf_min > 1."""
    rng = np.random.default_rng(1234)
    out = []
    for j, (name, vtdt) in enumerate(nets(seeds, shapes1, shapes2, wild, n_cal)):
        for ha in (0, 600):
            cm = int(rng.integers(2, 48)) if (j + ha // 600) % 4 == 3 else 1
            out.append((f"{name}_h{ha}_c{cm}", T2.pack(vtdt, ha, cm)))
    for p in extra:
        out.append((Path(p).name, Path(p).read_bytes()))
    return out


def probe_twin(blob):
    """The stride-4 version-1 probe whose result a d2s probe blob must reproduce (None for any other blob)."""
    m = T2.Model(blob)
    if m.version != T2.VERSION_D2S or not m.name.startswith("probe_d2s_v"):
        return None
    v = int(m.name[len("probe_d2s_v"):])
    return S.Model(S.pack(set_min_cells(TP.probe_blob(4, v), m.min_cells), m.half_above, m.conf_min))


# ---------------------------------------------------------------- the C driver and the twin

def check_blob(binary, v1_binary, name, blob, imgs, rng, td, cov, maps_every=1, arenas=True):
    """Every frame of imgs through C and the twin (and, for version 1, the frozen driver). Returns (runs, maps,
    v1 runs, mismatches)."""
    model = T2.Model(blob)
    ref4 = probe_twin(blob)
    bpath = Path(td) / f"{name}.vtsp"
    bpath.write_bytes(blob)
    cases, exp, bad = [], [], []
    for k, g in enumerate(imgs):
        img = TP.to_rgba(g, rng) if rng.random() < 0.5 else g
        r = T2.spot(model, img)
        cov.add(model, img, r)
        if ref4 is not None and S.spot(ref4, img) != r:
            bad.append((name, k, "the d2s probe differs from its stride-4 probe in the twin", ""))
        line, line_nd = P.twin_line(r), P.twin_line(r, dark=False)
        base = ODD * int(rng.random() < 0.3) | PADDED * int(rng.random() < 0.3) | MOST * int(rng.random() < 0.3)
        for form in (0, ROWS):
            cases.append((img, base | form))
            exp.append((k, form, line))
        if rng.random() < 0.5:
            form = int(rng.integers(0, 2))
            cases.append((img, base | NO_DARK | form))
            exp.append((k, f"nodark{form}", line_nd))
    batch = Path(td) / f"{name}.bin"
    P.write_batch(batch, cases)
    got = P.run_driver(binary, ["--batch", bpath, batch]).splitlines()
    if len(got) != len(cases):
        return len(cases), 0, 0, [(name, -1, f"{len(got)} output lines for {len(cases)} records", "")]
    for (k, form, e), g in zip(exp, got):
        if g != e:
            bad.append((name, k, f"form {form}, shape {imgs[k].shape}: C  {g[:400]}", f"py {e[:400]}"))
    nv1 = 0
    if v1_binary and model.version == S.VERSION:
        old = P.run_driver(v1_binary, ["--batch", bpath, batch]).splitlines()
        nv1 = len(old)
        if old != got:
            bad.append((name, -1, "vocr_textspot2_test and vocr_textspot_test print different lines", ""))
        sizes = sorted({(g.shape[1], g.shape[0]) for g in imgs} | {(960, 544), (480, 272), (600, 7)})
        for w, h in sizes if arenas else []:
            for kind in ("--arena", "--frame-arena"):
                a1 = P.run_driver(v1_binary, [kind, bpath, w, h]).strip()
                a2 = P.run_driver(binary, [kind, bpath, w, h]).strip()
                if a1 != a2:
                    bad.append((name, -1, f"{kind} {w}x{h}: {a2} != vocr_textspot's {a1}", ""))
    nmaps = 0
    if maps_every:
        sel = imgs[::maps_every]
        mb = Path(td) / f"{name}_maps.bin"
        P.write_batch(mb, [(g, 0) for g in sel])
        out = P.run_driver(binary, ["--maps", bpath, mb]).splitlines()
        for i, g in enumerate(sel):
            rc, hm, wm = map(int, out[2 * i + 1].split())
            ref = T2.run_maps(model, T2.net_input(model, g)[0])
            vals = np.array(out[2 * i].split()[1:], np.int64)
            ok = rc == 0 and vals.size == 4 * hm * wm and (hm, wm) == ref.shape[1:] and \
                (vals.reshape(hm, wm, 4).transpose(2, 0, 1) == ref).all()
            nmaps += 1
            if not ok:
                bad.append((name, i * maps_every, f"maps differ (rc {rc}, {hm}x{wm} vs {ref.shape[1:]})", ""))
    return len(cases), nmaps, nv1, bad


def check_loader(binary, blobs, td, per=8):
    """The C loader accepts every blob and refuses exactly the damaged copies the twin refuses (a few edits per
    blob, every other one with the outer CRC fixed so the inner checks decide)."""
    rng = np.random.default_rng(5)
    paths, want = [], []
    v1 = [b for b in blobs if b[1][4] == 1][:4]
    v2 = [b for b in blobs if b[1][4] == 2][:4]
    for name, blob in v1 + v2:
        for k in range(per):
            b = bytearray(blob)
            at = int(rng.integers(0, len(b))) if k < per // 2 else int(rng.integers(4, 64))
            b[at] ^= int(rng.integers(1, 256))
            if k % 2:
                struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
            try:
                T2.Model(bytes(b))
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
    got = [int(v) for v in P.run_driver(binary, ["--load", *paths]).split()]
    return [(str(p), w, g) for p, w, g in zip(paths, want, got) if w != g], len(paths)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="the driver (may carry a prefix, e.g. 'qemu-arm -cpu cortex-a9 ./x')")
    ap.add_argument("--v1-bin", default="", help="runtime/vocr_textspot_test (version-1 batches compared with it)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--frames", type=int, default=4, help="frames per random blob")
    ap.add_argument("--probe-frames", type=int, default=6, help="frames per probe blob")
    ap.add_argument("--full-every", type=int, default=4, help="every k-th blob also reads a whole screen (0: none)")
    ap.add_argument("--selftest", type=int, default=0, help="frames of the C self-test per blob (0: skip)")
    ap.add_argument("--selftest-blobs", type=int, default=4)
    ap.add_argument("--maps-every", type=int, default=2, help="compare the maps of every k-th frame (0: none)")
    ap.add_argument("--wild", type=int, default=1, help="shapes of each version with a wild-requantisation network")
    ap.add_argument("--arena-every", type=int, default=1,
                    help="with --v1-bin, compare the arenas of every k-th version-1 blob (0: none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cal", type=int, default=6)
    ap.add_argument("--shapes", default="", help="version-1 shapes (default: the protocol's kroa shapes + a few)")
    ap.add_argument("--d2s-shapes", default="", help="version-2 shapes (default: the protocol's kroa_x2 shapes)")
    ap.add_argument("--blob", action="append", default=[], help="more VTSP blobs")
    args = ap.parse_args(argv)
    t0 = time.time()
    shapes1 = [s for s in args.shapes.split(",") if s] or v1_shapes()
    shapes2 = [s for s in args.d2s_shapes.split(",") if s] or d2s_shapes()
    S2.register()
    blobs = all_blobs(args.seeds, shapes1, shapes2, args.blob, args.wild, args.n_cal)
    rng = np.random.default_rng(args.seed)
    runs = nframes = nmaps = nv1 = 0
    bad = []
    cov = P.Coverage()
    with tempfile.TemporaryDirectory(prefix="textspot2-parity-") as td:
        if args.selftest:
            pick = []
            for want in (lambda n: n.startswith("probe_d2s"), lambda n: n.startswith(tuple(shapes2)),
                         lambda n: n.startswith(tuple(shapes1)), lambda n: n.startswith("wild_"),
                         lambda n: n.startswith("probe_s")):
                pick += [b for b in blobs if want(b[0]) and b not in pick][:1]
            paths = []
            for name, blob in pick[:args.selftest_blobs]:
                p = Path(td) / f"self_{name}.vtsp"
                p.write_bytes(blob)
                paths.append(p)
            print(P.run_driver(args.bin, ["--selftest", args.selftest, *paths]).strip())
        lbad, nload = check_loader(args.bin, blobs, td)
        for name, w, g in lbad[:12]:
            bad.append((name, -1, f"loader: C {g}", f"twin {w}"))
        for bi, (name, blob) in enumerate(blobs):
            model = T2.Model(blob)
            probe = name.startswith("probe")
            screen = SCREENS[(bi // args.full_every) % len(SCREENS)] if args.full_every and bi % args.full_every == 0 \
                else None
            imgs = P.frames_for(rng, model, args.probe_frames if probe else args.frames, probe, screen)
            arenas = bool(args.arena_every) and bi % args.arena_every == 0
            r, mp, v1, b = check_blob(args.bin, args.v1_bin, name, blob, imgs, rng, td, cov, args.maps_every, arenas)
            runs += r
            nframes += len(imgs)
            nmaps += mp
            nv1 += v1
            bad += b
    nv2 = sum(1 for _, b in blobs if b[4] == T2.VERSION_D2S)
    if bad:
        for name, k, c, p in bad[:12]:
            print(f"MISMATCH {name} frame {k}: {c}\n    {p}")
        print(f"TEXTSPOT2_PARITY_FAIL {len(bad)} of {runs} runs")
        return 1
    print(f"coverage: {cov}")
    print(f"TEXTSPOT2_PARITY_OK {runs} runs ({nframes} frames, {len(blobs)} blobs ({nv2} of version 2), {nmaps} maps, "
          f"{nload} loader cases, {nv1} version-1 runs equal to vocr_textspot_test's; {time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Export a trained text detector (train/textdet_train.py checkpoint) to the int8 blob the C runtime runs
(runtime/vocr_textdet.c; layout in synth/textdet_twin.py).

  textdet_export.py --ckpt RUN/final.pt --out det.vtd          a trained, quantisation-aware checkpoint
  textdet_export.py --random SHAPE --seed N --out det.vtd      random integer weights, requantisation set so the
                                                               activations spread (runtime tests, not a detector)
  textdet_export.py --info det.vtd                             sizes, MACs and the arena

Every conv becomes int8 weights per output channel (scale max|w_c| / 127), an int32 bias in the accumulator's
units and a fixed-point requantisation (multiplier, shift) to the next layer's int8 activation scale. The input
scale is 1/128 (codes = grey - 128); the head's K and O maps are written at 1/16 per code and A1, A2 at 1/2 px.
"""
import argparse
import hashlib
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_model as TM  # noqa: E402
import textdet_twin as TW  # noqa: E402

IN_SCALE = 1.0 / 128


def quantize_multiplier(M):
    """(m, shift) with m * 2^-shift ~= M, m in [2^30, 2^31), 1 <= shift <= 62."""
    if not M > 0:
        raise ValueError(f"multiplier {M} must be positive")
    mant, e = math.frexp(M)
    m = int(math.floor(mant * (1 << 31) + 0.5))
    if m == 1 << 31:
        m, e = 1 << 30, e + 1
    sh = 31 - e
    if sh > 62:
        m = max(1, m >> (sh - 62))
        sh = 62
    if sh < 1:
        raise ValueError(f"multiplier {M} too large")
    return m, sh


def quantize_layers(shape, folded, act_scale):
    """Integer layers from folded float weights and per-layer output scales: [dict(w, b, m, sh, ...)]."""
    out = []
    s_x = IN_SCALE
    for L, (w, b), s_y in zip(TM.layers(shape), folded, act_scale):
        w = np.asarray(w, np.float64)
        b = np.asarray(b, np.float64)
        cout = L["cout"]
        if L["head"]:
            s_out = np.array([1 / TM.KO_SCALE, 1 / TM.KO_SCALE, 1 / TM.A_SCALE, 1 / TM.A_SCALE])
        else:
            s_out = np.full(cout, float(s_y))
        mx = np.abs(w.reshape(cout, -1)).max(1)
        s_w = np.where(mx > 0, mx / 127, 1.0)
        wq = np.clip(np.floor(w / s_w.reshape(-1, 1, 1, 1) + 0.5), -127, 127).astype(np.int8)
        bq = np.clip(np.floor(b / (s_w * s_x) + 0.5), -(2 ** 31 - 1), 2 ** 31 - 1).astype(np.int64)
        ms = [quantize_multiplier(float(s_w[c] * s_x / s_out[c])) for c in range(cout)]
        out.append({**L, "wq": wq, "bq": bq, "m": np.array([m for m, _ in ms], np.int64),
                    "sh": np.array([s for _, s in ms], np.int64)})
        if not L["head"]:
            s_x = float(s_y)
    return out


def pack(shape, qlayers, ckpt_sha256=b"\0" * 8):
    """The VTDT blob bytes."""
    n = len(qlayers)
    data = bytearray()
    base = TW.HEADER + n * TW.RECORD
    recs = []

    def put(arr):
        while (base + len(data)) % 4:
            data.append(0)
        off = base + len(data)
        data.extend(arr.tobytes())
        return off

    for L in qlayers:
        wo = put(np.ascontiguousarray(L["wq"], np.int8))
        bo = put(np.asarray(L["bq"], "<i4"))
        mo = put(np.asarray(L["m"], "<i4"))
        so = put(np.asarray(L["sh"], np.uint8))
        recs.append(struct.pack("<BBBBB3xHHIIIII", 1 if L["kind"] == "dw" else 0, L["k"], L["stride"], L["dil"],
                                1 if L["relu"] else 0, L["cin"], L["cout"], wo, bo, mo, so, 0))
    while len(data) % 4:
        data.append(0)
    total = base + len(data)
    name = shape["name"].encode("ascii")[:31].ljust(32, b"\0")
    head = bytearray(TW.HEADER)
    head[:4] = TW.MAGIC
    struct.pack_into("<HBBBBH", head, 16, n, shape["stride"], TM.N_OUT, TM.A_SCALE, TM.MIN_CELLS[shape["stride"]], 0)
    head[24:32] = ckpt_sha256[:8]
    head[32:64] = name
    body = bytes(head[16:]) + b"".join(recs) + bytes(data)
    struct.pack_into("<III", head, 4, TW.VERSION, total, zlib.crc32(body) & 0xFFFFFFFF)
    blob = bytes(head) + b"".join(recs) + bytes(data)
    assert len(blob) == total
    TW.Model(blob)            # the loader's checks
    return blob


def export_checkpoint(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "textdet-qat":
        raise SystemExit(f"{path}: not a quantisation-aware textdet checkpoint")
    shape = ck["shape"]
    folded = [(w.double(), b.double()) for w, b in zip(ck["w"], ck["b"])]
    sha = hashlib.sha256(Path(path).read_bytes()).digest()
    return pack(shape, quantize_layers(shape, folded, ck["act_scale"]), sha)


def random_model(shape_name, seed, calib=8):
    """A random-init network with random BatchNorm statistics, folded, scales calibrated on random regions."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    shape = TM.SHAPES[shape_name]
    model = TM.TextDet(shape)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    model.eval()
    folded = fold(model)
    taps_max = [0.0] * len(folded)
    q = TM.QTextDet(shape, folded, [1.0] * len(folded))
    q.quant = False
    with torch.no_grad():
        for _ in range(calib):
            g = procedural_region(rng).astype(np.float32)[None, None]
            taps = []
            q((torch.from_numpy(g) - 128) / 128, taps=taps)
            for i, t in enumerate(taps):
                taps_max[i] = max(taps_max[i], float(t.abs().max()))
    scales = [max(m, 1e-3) / 127 for m in taps_max]
    # centre the head so the decode sees mixed kernel signs, both orientations and positive edge distances
    q = TM.QTextDet(shape, folded, scales)
    q.quant = False
    outs = []
    with torch.no_grad():
        for _ in range(calib):
            g = procedural_region(rng)
            outs.append(q((torch.from_numpy(g.astype(np.float32))[None, None] - 128) / 128)[0].flatten(1))
    o = torch.cat(outs, 1).double()
    w, b = folded[-1]
    shift = torch.stack([o[0].median(), o[1].median(), o[2].median() - 8, o[3].median() - 8])
    folded[-1] = (w, b - shift)
    return shape, folded, scales


def procedural_region(rng):
    """A grey region of random blocks and stripes (text-like rows of strokes), for random-weight tests."""
    h, w = int(rng.integers(24, 220)), int(rng.integers(24, 640))
    g = np.full((h, w), int(rng.integers(0, 256)), np.int32)
    for _ in range(int(rng.integers(1, 8))):
        y0, x0 = int(rng.integers(0, h)), int(rng.integers(0, w))
        bh, bw = int(rng.integers(4, 40)), int(rng.integers(8, 300))
        sub = g[y0:y0 + bh, x0:x0 + bw]
        sub[:] = int(rng.integers(0, 256))
        sub[:, ::int(rng.integers(2, 6))] = int(rng.integers(0, 256))
    g += rng.integers(-12, 13, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def fold(model):
    return TM.fold(model)


def random_int_layers(shape_name, seed, n_cal=6):
    """Random integer layers whose activations spread over the int8 range on procedural regions (for tests of the
    C runtime against the twin: arbitrary valid blobs, not a trained detector). The head's K and O maps are
    centred on 0 and A1, A2 on about 8 px, so the decode sees mixed signs, both orientations and real boxes."""
    rng = np.random.default_rng(seed)
    shape = TM.SHAPES[shape_name]
    regions = [procedural_region(rng) for _ in range(n_cal)]
    xs = [g.astype(np.int64)[None] - 128 for g in regions]
    out = []
    for L in TM.layers(shape):
        per = 1 if L["kind"] == "dw" else L["cin"]
        wq = rng.integers(-127, 128, (L["cout"], per, L["k"], L["k"])).astype(np.int8)
        wq[rng.random(wq.shape) < 0.3] = 0
        tl = TW.Layer()
        tl.kind, tl.k, tl.stride, tl.dil, tl.relu = (1 if L["kind"] == "dw" else 0), L["k"], L["stride"], L["dil"], 0
        tl.cin, tl.cout = L["cin"], L["cout"]
        tl.w = wq.astype(np.int64)
        tl.b = np.zeros(L["cout"], np.int64)
        tl.m = np.full(L["cout"], 1 << 30, np.int64)
        tl.sh = np.full(L["cout"], 30, np.int64)          # identity requantisation: the raw accumulator, unclamped
        accs = []
        for x in xs:
            acc = _accumulate(tl, x)
            accs.append(acc.reshape(L["cout"], -1))
        a = np.concatenate(accs, 1).astype(np.float64)
        med, sd = np.median(a, 1), a.std(1) + 1.0
        if L["head"]:
            target_med = np.array([0.0, 0.0, 16.0, 16.0])
            target_sd = np.array([24.0, 24.0, 6.0, 6.0])
        else:
            target_med = np.full(L["cout"], 10.0 if L["relu"] else 0.0)
            target_sd = np.full(L["cout"], 30.0)
        M = target_sd / sd
        ms = [quantize_multiplier(float(v)) for v in M]
        bq = np.floor(target_med / M - med + 0.5).astype(np.int64)
        q = {**L, "wq": wq, "bq": bq, "m": np.array([m for m, _ in ms], np.int64),
             "sh": np.array([s for _, s in ms], np.int64)}
        out.append(q)
        tl.relu, tl.b, tl.m, tl.sh = (1 if L["relu"] else 0), bq, q["m"], q["sh"]
        xs = [TW.run_layer(tl, x) for x in xs]
    return shape, out


def _accumulate(tl, x):
    """The twin's accumulator (bias included, before requantisation) of one layer."""
    cin, h, w = x.shape
    s, d, k = tl.stride, tl.dil, tl.k
    p = (k - 1) // 2 * d
    ho, wo = (h + s - 1) // s, (w + s - 1) // s
    xp = np.pad(x, ((0, 0), (p, p), (p, p))) if p else x
    acc = np.zeros((tl.cout, ho, wo), np.int64)
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, ky * d: ky * d + s * (ho - 1) + 1: s, kx * d: kx * d + s * (wo - 1) + 1: s]
            if tl.kind == 1:
                acc += tl.w[:, 0, ky, kx][:, None, None] * patch
            else:
                acc += np.tensordot(tl.w[:, :, ky, kx], patch, axes=(1, 0))
    return acc + tl.b[:, None, None]


def info(path):
    blob = Path(path).read_bytes()
    m = TW.Model(blob)
    shape = TM.SHAPES.get(m.name)
    print(f"{path}: {m.name}, stride {m.stride}, {len(m.layers)} layers, {m.blob_bytes} bytes, ckpt {m.ckpt}")
    if shape:
        for w, h in ((900, 150), (450, 80), (960, 544), (480, 272)):
            print(f"  {w}x{h}: {TM.cost(shape, w, h)['macs']:,} MACs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--info")
    args = ap.parse_args()
    if args.info:
        info(args.info)
        return
    if args.ckpt:
        blob = export_checkpoint(args.ckpt)
    elif args.random:
        shape, qlayers = random_int_layers(args.random, args.seed)
        blob = pack(shape, qlayers, hashlib.sha256(f"random:{args.random}:{args.seed}".encode()).digest())
    else:
        ap.error("--ckpt, --random or --info")
    Path(args.out).write_bytes(blob)
    print(f"{args.out}: {len(blob)} bytes")


if __name__ == "__main__":
    main()

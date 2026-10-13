#!/usr/bin/env python3
"""Console artefacts of native PS Vita / PSP frames, measured on training-side data only.

synth3's native console-frame mode (synth3.py, CFG["frames"]) renders whole 960x544 / 480x272
frames and applies the capture chain to the frame. Its settings come from these measurements, on
training-side native VNDB screenshots (realplus/native.py's selection: realdata's and teacher's
native frames of training-side VNs; never an eval, linefind, devnative, teacher_val, CLANNAD or
owner-game frame) and on training-side native line crops (teacher_train + real_train, kinds vita
and psp, reserved VNs dropped):

  native_calib.py jpeg   --shots SHOTS.jsonl --out J        JPEG quality (IJG scale fitted to the file's
                                                           luma table) and chroma subsampling per frame
  native_calib.py frames --shots SHOTS.jsonl --out J        resampling traces on the frame: the periodic
                                                           second-difference energy an upscale by 2
                                                           (PSP on Vita), 4/3 or 3/2 leaves, and the
                                                           odd/even column-pair test for nearest 2x
  native_calib.py crops  --real NPZ[+NPZ] --synth GLOB --out J   glyph aspect (ink width per full-width
                                                           character / ink height) and edge sharpness
                                                           (p90 gradient / contrast) per target and kind
                                                           for real and synthetic crops alike
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# libjpeg's standard luminance table (ITU T.81 K.1), quality-scaled by IJG's rule
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                     92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float64)


def ijg_table(q):
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((STD_LUMA * s + 50) / 100), 1, 255)


# natural (row-major) index of each zigzag position
ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,
                   6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
                   38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def jpeg_quality(path):
    """(quality, exact, subsampling): the IJG quality whose luma table is closest to the file's (exact
    when equal), and PIL's subsampling code (0 4:4:4, 1 4:2:2, 2 4:2:0, -1 other); None if no JPEG."""
    from PIL import Image, JpegImagePlugin
    with Image.open(path) as im:
        if im.format != "JPEG":
            return None
        q = getattr(im, "quantization", None) or {}
        sub = JpegImagePlugin.get_sampling(im)
    if 0 not in q:
        return None
    t = np.asarray(q[0], np.float64)
    # Pillow's table order has changed between versions (natural vs zigzag): fit both orders
    errs = [(min(float(np.abs(ijg_table(k) - t).mean()), float(np.abs(ijg_table(k)[ZIGZAG] - t).mean())), k)
            for k in range(1, 101)]
    err, best = min(errs)
    return best, err == 0.0, sub


def cmd_jpeg(args):
    rows = [json.loads(x) for x in open(args.shots, encoding="utf-8") if x.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        got = jpeg_quality(r["path"])
        by[r["kind"]].append(None if got is None else got)
    out = {}
    for kind, vals in by.items():
        ok = [v for v in vals if v]
        qs = np.array([v[0] for v in ok])
        out[kind] = {"frames": len(vals), "jpeg": len(ok), "exact_ijg_table": sum(v[1] for v in ok),
                     "quality_hist": dict(sorted(collections.Counter(int(q) for q in qs).items())),
                     "quality_p10_50_90": [float(np.percentile(qs, p)) for p in (10, 50, 90)] if len(qs) else None,
                     "subsampling": dict(collections.Counter({0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(v[2], "other")
                                                             for v in ok))}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- frames

def grey(path):
    import cv2
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if g is None else g.astype(np.float32)


def periodic_energy(g, axis):
    """Spectrum (DFT magnitude, normalised by its median) of the mean |second difference| along an
    axis: interpolation by a factor p/q leaves peaks at multiples of 1/p cycles per pixel."""
    d2 = np.abs(np.diff(g, n=2, axis=axis))
    prof = d2.mean(axis=1 - axis)
    prof = prof - prof.mean()
    f = np.abs(np.fft.rfft(prof))
    freqs = np.fft.rfftfreq(len(prof))
    med = float(np.median(f[1:])) + 1e-6
    return freqs, f / med


def peak(freqs, mag, f0, tol=0.004):
    sel = np.abs(freqs - f0) <= tol
    return float(mag[sel].max()) if sel.any() else 0.0


def pair_ratio(g):
    """Nearest-neighbour 2x upscale: columns 2k and 2k+1 are copies, so their mean |difference| is far
    below that of columns 2k+1 and 2k+2. Returns the smaller of the two phase ratios."""
    d = np.abs(np.diff(g, axis=1))
    a, b = float(d[:, 0::2].mean()), float(d[:, 1::2].mean())
    return min(a, b) / (max(a, b) + 1e-6)


def cmd_frames(args):
    rows = [json.loads(x) for x in open(args.shots, encoding="utf-8") if x.strip()]
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g = grey(r["path"])
        if g is None:
            continue
        k = r["kind"]
        fx, mx = periodic_energy(g, 1)
        fy, my = periodic_energy(g, 0)
        out[k]["pair_ratio"].append(pair_ratio(g))
        for name, f0 in (("x2_f0.5", 0.5), ("x4/3_f0.25", 0.25), ("x3/2_f0.333", 1 / 3), ("jpeg_f0.125", 0.125)):
            out[k][name + "_x"].append(peak(fx, mx, f0))
            out[k][name + "_y"].append(peak(fy, my, f0))
    res = {}
    for k, d in out.items():
        res[k] = {"frames": len(d["pair_ratio"])}
        for name, vals in d.items():
            v = np.asarray(vals)
            res[k][name] = {"p10_50_90": [round(float(np.percentile(v, p)), 3) for p in (10, 50, 90)]}
        pr = np.asarray(d["pair_ratio"])
        res[k]["nearest2x_like (pair_ratio < 0.35)"] = int((pr < 0.35).sum())
        # a frame whose 1/3 or 1/4 cycle peak stands far above the JPEG grid's own 1/8 harmonic
        for name in ("x4/3_f0.25", "x3/2_f0.333"):
            a = np.asarray(d[name + "_x"])
            res[k][f"{name}_x > 8"] = int((a > 8).sum())
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- crops

FULL = set("、。「」『』（）！？…―～・")


def full_width(lab):
    import unicodedata
    return all(unicodedata.east_asian_width(c) in ("W", "F") or c in FULL for c in lab)


def sharpness(g, box):
    """p90 of the gradient magnitude inside the ink box, over the crop's p95-p5 contrast."""
    x0, y0, x1, y1 = box
    a = np.asarray(g, np.float32)[y0:y1, x0:x1]
    if a.shape[0] < 3 or a.shape[1] < 3:
        return None
    gy, gx = np.gradient(a)
    m = np.hypot(gx, gy)
    c = float(np.percentile(a, 95) - np.percentile(a, 5))
    return float(np.percentile(m, 90)) / c if c > 20 else None


def crop_stats(paths, kinds, reserved, limit, rng):
    import synth3_stats as S
    out = collections.defaultdict(lambda: {"aspect": [], "sharp": []})
    for p in paths:
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        labels = json.loads(str(z["labels"]))
        shapes, offs, data = z["shapes"], z["offsets"], z["data"]
        idx = np.arange(len(meta))
        if limit and len(idx) > limit:
            idx = np.sort(rng.choice(idx, limit, replace=False))
        for i in idx:
            m = meta[i]
            if m.get("vn") in reserved:
                continue
            kind = S.kind_of(m)
            if kind not in kinds:
                continue
            h, w = shapes[i]
            g = data[offs[i]:offs[i + 1]].reshape(h, w)
            box = S.px_ink(g)
            if box is None:
                continue
            key = f"{m.get('target', kind)}:{kind}"
            lab = labels[i]
            if len(lab) >= 5 and full_width(lab):
                ih = box[3] - box[1]
                if ih >= 6:
                    out[key]["aspect"].append((box[2] - box[0]) / len(lab) / ih)
            s = sharpness(g, box)
            if s is not None:
                out[key]["sharp"].append(s)
    res = {}
    for key, d in out.items():
        res[key] = {}
        for name, vals in d.items():
            v = np.asarray(vals)
            if len(v):
                res[key][name] = {"n": int(len(v)), "p10_50_90": [round(float(np.percentile(v, q)), 3)
                                                                 for q in (10, 50, 90)]}
        a = np.asarray(d["aspect"])
        if len(a):
            res[key]["aspect_share_wide (>1.15)"] = round(float((a > 1.15).mean()), 4)
            res[key]["aspect_share_narrow (<0.80)"] = round(float((a < 0.80).mean()), 4)
    return res


def cmd_crops(args):
    import synth3_stats as S
    reserved = S.reserved_vns(S.DEFAULT_EXCLUDE)
    if args.exclude:
        reserved |= S.reserved_vns(args.exclude)
    rng = np.random.default_rng(0)
    res = {"real": crop_stats([p for spec in args.real for p in spec.split("+")], {"vita", "psp"}, reserved,
                              args.limit, rng)}
    if args.synth:
        res["synth"] = crop_stats(sorted(glob.glob(args.synth)), {"vita", "psp", "psp2x"}, set(), args.limit, rng)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("jpeg", "frames"):
        p = sub.add_parser(c)
        p.add_argument("--shots", required=True)
        p.add_argument("--out", required=True)
    p = sub.add_parser("crops")
    p.add_argument("--real", nargs="+", required=True)
    p.add_argument("--synth")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=0, help="at most this many crops per file (0: all)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"jpeg": cmd_jpeg, "frames": cmd_frames, "crops": cmd_crops}[a.cmd](a)


if __name__ == "__main__":
    main()

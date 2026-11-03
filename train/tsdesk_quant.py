#!/usr/bin/env python3
"""textspot-desktop's int8 detector (issue #173, round 3; docs/textspot-desktop.md): a version-1 VTSD blob (float32)
quantized to a version-2 one (train/tsdesk_blob.py, desktop/vocr_tsd_q8.c), and the numpy twin of its maps.

    python train/tsdesk_quant.py frames --out FRAMES.pkl            # training-side calibration / check frames
    python train/tsdesk_quant.py quant --blob K1.vtsd --frames FRAMES.pkl --out K1.q8.vtsd [--method mse]
    python train/tsdesk_quant.py err --blob K1.vtsd --q8 K1.q8.vtsd --frames FRAMES.pkl --part check

The scheme (tsdesk_blob's version 2): int8 weights per output channel (symmetric: max |w| -> 127), uint8 maps per
tensor with zero point 0 (every map of the network is a ReLU or a sum of ReLUs), static scales chosen from a
calibration pass of the float network over training-side frames (CALIB_METHODS), exact int32 sums and a float32
requantization per output channel. The head's maps stay float32, so the decode reads them as it reads the float
network's. q8_maps is the network's twin in numpy: the C int8 kernels (every build, every thread count) give its maps
bit for bit (tests/test_tsdesk_c.py).

Calibration frames (`frames`): training-side frames only, never the timing pages (eval/tsdesk_time.py's hval and 1b
frames are left out by key and VN) and no protected set: desktop-data's 1a (the hooked pools' train pages as desktop
captures), 1b (realplus v3 at its own size) and 1c (synthetic desktop frames), docs/desktop-data.md. The calibration
part and the check part (`err`) come from disjoint shards.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                       # desktopdata (the shards)

import tsdesk_blob as TB  # noqa: E402

CALIB_METHODS = ("max", "mse", "p99.9", "p99.99", "p99.999")
HIST_BINS = 8192
HERD = Path("/local/home/skerraut/herd-vita-ocr")
PAGES = HERD / "shared" / "artifacts" / "desktop-data" / "pages"
FRAME_SOURCES = {"1a": PAGES / "hooked_v1" / "train", "1b": PAGES / "realplus_v3" / "desk_realplus_v3",
                 "1c": PAGES / "synth_v1" / "desk_synth_v1"}
TIMING_PAGES = HERD / "shared" / "private" / "textspot-desktop" / "timing_pages.pkl"


# ----------------------------------------------------------------------------- the network's steps

def graph(h):
    """The steps in execution order (a read_header dict h): dicts of kind (stem, sep, gc, lat, head), name (the layer:
    a sep's are NAME.dw and NAME.pw, tsdesk_blob.layer_list), src (the input tensor), out (the output tensor), res (a
    sep's residual add), up (a lateral's coarser level, or None), stride, dil. Tensors are named after their steps
    (stem, s8.b0, gc, p32, smooth0, ...), a sep's depthwise output NAME.dw; the input is "in", the head's "maps"."""
    c, n, hs = h["c"], h["n"], h["hs"]
    steps = [{"kind": "stem", "name": "stem", "src": "in", "out": "stem"}]
    prev, fin = "stem", {}
    for s in range(4):
        st = 4 << s
        if s > 0:
            name = f"s{st}.down"
            steps.append({"kind": "sep", "name": name, "src": prev, "out": name, "res": False, "stride": 2, "dil": 1})
            prev = name
        for i in range(n[s]):
            name = f"s{st}.b{i}"
            steps.append({"kind": "sep", "name": name, "src": prev, "out": name, "res": True, "stride": 1,
                          "dil": h["dil"][s][i]})
            prev = name
        fin[s] = prev
    if h["gc"]:
        steps.append({"kind": "gc", "name": "gc", "src": fin[3], "out": "gc"})
        fin[3] = "gc"
    coarse = None
    for s in range(3, (0 if hs == 4 else 1) - 1, -1):
        steps.append({"kind": "lat", "name": f"lat{4 << s}", "src": fin[s], "out": f"p{4 << s}", "up": coarse})
        coarse = f"p{4 << s}"
    prev = coarse
    for i in range(h["smooth"]):
        name = f"smooth{i}"
        steps.append({"kind": "sep", "name": name, "src": prev, "out": name, "res": True, "stride": 1, "dil": 1})
        prev = name
    steps.append({"kind": "head", "name": "head", "src": prev, "out": "maps"})
    if len(c) != 4:
        raise ValueError(h)
    return steps


def tensors(h):
    """The quantized tensors (uint8 maps with a scale each) in execution order."""
    out = []
    for st in graph(h):
        if st["kind"] == "sep":
            out.append(st["out"] + ".dw")
        if st["kind"] != "head":
            out.append(st["out"])
    return out


# ----------------------------------------------------------------------------- the float network (torch)

class FloatNet:
    """The float32 network of a version-1 blob in torch (the weights tsdesk_blob.read_weights gives: the C path's
    network, within float rounding of it and of the ONNX graph): maps and, when asked, every tensor of tensors(h)."""

    def __init__(self, blob, threads=None):
        import torch
        self.torch = torch
        if threads:
            torch.set_num_threads(int(threads))
        self.h, W = TB.read_weights(blob)
        self.steps = graph(self.h)
        T = {}
        for L in TB.layer_list(self.h):
            w, b = W[L["name"]]["w"], W[L["name"]]["b"]
            if L["kind"] == "stem":
                wt = w.T.reshape(L["cout"], 1, 4, 4)
            elif L["kind"] == "dw":
                wt = w.T.reshape(L["cin"], 1, 3, 3)
            elif L["kind"] == "gc":
                wt = w
            else:
                wt = w.T.reshape(L["cout"], L["cin"], 1, 1)
            T[L["name"]] = (torch.from_numpy(np.ascontiguousarray(wt)), torch.from_numpy(b.copy()))
        self.W = T

    def __call__(self, x, keep=False):
        """(maps [out, H / hs, W / hs] float32 numpy, {tensor: torch [C, h, w]} when keep) of a padded grey input x
        (uint8 [H, W] or [1, 1, H, W], sides multiples of 32)."""
        torch, F = self.torch, self.torch.nn.functional
        x = np.ascontiguousarray(np.asarray(x, np.uint8).reshape(np.asarray(x).shape[-2:]))
        t = {}
        with torch.no_grad():
            g = torch.from_numpy(x).float()[None, None]
            t["in"] = (g - 128.0) / 128.0
            for st in self.steps:
                k, name = st["kind"], st["name"]
                a = t[st["src"]]
                if k == "stem":
                    w, b = self.W["stem"]
                    t["stem"] = F.relu(F.conv2d(a, w, b, stride=4))
                elif k == "sep":
                    wd, bd = self.W[name + ".dw"]
                    wp, bp = self.W[name + ".pw"]
                    d, s = st["dil"], st["stride"]
                    y = F.relu(F.conv2d(a, wd, bd, stride=s, padding=d, dilation=d, groups=a.shape[1]))
                    t[name + ".dw"] = y
                    z = F.conv2d(y, wp, bp)
                    t[name] = F.relu(z + a if st["res"] else z)
                elif k == "gc":
                    w, b = self.W["gc"]
                    g_ = F.relu(a.mean((2, 3)) @ w + b)
                    t["gc"] = a + g_[:, :, None, None]
                elif k == "lat":
                    w, b = self.W[name]
                    p = F.relu(F.conv2d(a, w, b))
                    if st["up"]:
                        p = p + F.interpolate(t[st["up"]], scale_factor=2.0, mode="nearest")
                    t[st["out"]] = p
                else:
                    w, b = self.W["head"]
                    t["maps"] = F.conv2d(a, w, b)
        maps = t["maps"][0, :self.h["out"]].numpy().astype(np.float32)
        if not keep:
            return maps, None
        return maps, {k: v[0] for k, v in t.items() if k not in ("in", "maps")}


# ----------------------------------------------------------------------------- calibration

class Calib:
    """Per-tensor statistics of the float network's maps over calibration inputs: max, then a histogram of the
    positive values on [0, max] (HIST_BINS bins) and the count of zeros (two passes over the same inputs)."""

    def __init__(self, net):
        self.net = net
        self.names = tensors(net.h)
        self.max = {k: 0.0 for k in self.names}
        self.hist = {k: np.zeros(HIST_BINS, np.float64) for k in self.names}
        self.zeros = {k: 0 for k in self.names}
        self.count = {k: 0 for k in self.names}

    def pass_max(self, xs):
        for x in xs:
            _, t = self.net(x, keep=True)
            for k in self.names:
                self.max[k] = max(self.max[k], float(t[k].max()))

    def pass_hist(self, xs):
        torch = self.net.torch
        for x in xs:
            _, t = self.net(x, keep=True)
            for k in self.names:
                v = t[k].flatten()
                pos = v[v > 0]
                self.count[k] += int(v.numel())
                self.zeros[k] += int(v.numel() - pos.numel())
                if self.max[k] > 0 and pos.numel():
                    h = torch.histc(pos.clamp(max=self.max[k]), HIST_BINS, 0.0, self.max[k])
                    self.hist[k] += h.double().numpy()

    def clip(self, k, method):
        """The clip value of tensor k (the largest value its uint8 codes hold: scale = clip / 255)."""
        mx, hist = self.max[k], self.hist[k]
        if mx <= 0 or hist.sum() <= 0:
            return 0.0
        edges = np.linspace(0.0, mx, HIST_BINS + 1)
        if method == "max":
            return mx
        if method.startswith("p"):
            p = float(method[1:]) / 100.0
            cum = np.cumsum(hist) / hist.sum()
            j = int(np.searchsorted(cum, p))
            return float(edges[min(j + 1, HIST_BINS)])
        if method == "mse":
            centers = (edges[:-1] + edges[1:]) / 2
            best, best_c = None, mx
            for c in mx * np.linspace(0.02, 1.0, 491):
                s = c / 255.0
                q = np.minimum(np.rint(centers / s), 255.0) * s
                err = float((hist * (centers - q) ** 2).sum())
                if best is None or err < best:
                    best, best_c = err, c
            return float(best_c)
        raise ValueError(f"calibration method {method!r}: one of {CALIB_METHODS}")

    def scales(self, method):
        return {k: (self.clip(k, method) / 255.0) or 1.0 for k in self.names}

    def summary(self):
        return {k: {"max": self.max[k], "zeros": self.zeros[k] / max(1, self.count[k])} for k in self.names}


# ----------------------------------------------------------------------------- quantization

def _wscale(w):
    """Per output channel (axis 1 of [K][Cout]): max |w| / 127, 1 where a channel is all zero."""
    s = np.abs(w.astype(np.float64)).max(axis=0) / 127.0
    return np.where(s > 0, s, 1.0)


def _wq(w, s):
    return np.clip(np.rint(w.astype(np.float64) / s), -127, 127).astype(np.int8)


def _bq(b, acc_scale):
    return np.clip(np.rint(b.astype(np.float64) / acc_scale), -TB.Q8_BIAS_MAX, TB.Q8_BIAS_MAX).astype(np.int64)


def quantize(blob, sa):
    """{layer name: quantized form} (tsdesk_blob.blob_bytes_q8's q) of a version-1 blob's network for the tensor scales
    sa ({tensor: float}; tensors(h))."""
    h, W = TB.read_weights(blob)
    q = {}
    for st in graph(h):
        k, name = st["kind"], st["name"]
        if k == "stem":
            w, b = W["stem"]["w"], W["stem"]["b"]
            sw = _wscale(w)
            wq = _wq(w, sw)
            acc = sw / 128.0                     # the input is (g - 128) / 128: acc of raw g, bq holds the -128 sum
            bq = np.clip(_bq(b, acc) - 128 * wq.astype(np.int64).sum(0), -TB.Q8_BIAS_MAX, TB.Q8_BIAS_MAX)
            q["stem"] = {"w": wq, "bq": bq, "m": (acc / sa["stem"]).astype(np.float32), "r": 0.0}
        elif k == "sep":
            s_in, s_dw, s_out = sa[st["src"]], sa[name + ".dw"], sa[st["out"]]
            w, b = W[name + ".dw"]["w"], W[name + ".dw"]["b"]
            sw = _wscale(w)
            q[name + ".dw"] = {"w": _wq(w, sw), "bq": _bq(b, s_in * sw), "m": (s_in * sw / s_dw).astype(np.float32)}
            w, b = W[name + ".pw"]["w"], W[name + ".pw"]["b"]
            sw = _wscale(w)
            q[name + ".pw"] = {"w": _wq(w, sw), "bq": _bq(b, s_dw * sw), "m": (s_dw * sw / s_out).astype(np.float32),
                               "r": float(np.float32(s_in / s_out)) if st["res"] else 0.0}
        elif k == "gc":
            q["gc"] = {"w": W["gc"]["w"], "b": W["gc"]["b"], "sx": sa[st["src"]], "so": sa["gc"]}
        elif k == "lat":
            s_in, s_out = sa[st["src"]], sa[st["out"]]
            w, b = W[name]["w"], W[name]["b"]
            sw = _wscale(w)
            q[name] = {"w": _wq(w, sw), "bq": _bq(b, s_in * sw), "m": (s_in * sw / s_out).astype(np.float32),
                       "r": float(np.float32(sa[st["up"]] / s_out)) if st["up"] else 0.0}
        else:
            s_in = sa[st["src"]]
            w, b = W["head"]["w"], W["head"]["b"]
            sw = _wscale(w)
            q["head"] = {"w": _wq(w, sw), "bq": _bq(b, s_in * sw), "m": (s_in * sw).astype(np.float32), "r": 0.0}
    return q


# ----------------------------------------------------------------------------- the int8 network's twin (numpy)

def _mm(x, w):
    """Exact integer products of uint8 x [P][K] and int8 w [K][N] (float64 sums of integers below 2^53)."""
    return np.rint(x.astype(np.float64) @ w.astype(np.float64)).astype(np.int64)


def _req(acc, e):
    t = acc + e["bq"].astype(np.int64)
    if t.size and np.abs(t).max() >= 2 ** 31:
        raise OverflowError("an int8 layer's sum leaves int32")
    return t.astype(np.int32).astype(np.float32) * e["m"]


def _u8(y):
    return np.rint(np.clip(y, np.float32(0.0), np.float32(255.0))).astype(np.uint8)


def _dw(x, e, stride, dil):
    H, W, C = x.shape
    Ho, Wo = (H - 1) // stride + 1, (W - 1) // stride + 1
    xp = np.zeros((H + 2 * dil, W + 2 * dil, C), np.int64)
    xp[dil:dil + H, dil:dil + W] = x
    acc = np.zeros((Ho, Wo, C), np.int64)
    w = e["w"].astype(np.int64)
    for ky in range(3):
        for kx in range(3):
            acc += xp[ky * dil:ky * dil + stride * (Ho - 1) + 1:stride,
                      kx * dil:kx * dil + stride * (Wo - 1) + 1:stride] * w[3 * ky + kx]
    return _u8(_req(acc, e))


def _pw(x, e, res=None, up=None):
    H, W, C = x.shape
    y = _req(_mm(x.reshape(-1, C), e["w"]), e).reshape(H, W, -1)
    if res is not None:
        y = y + res.astype(np.float32) * np.float32(e["r"])
    elif up is not None:
        u = np.repeat(np.repeat(up, 2, axis=0), 2, axis=1)
        y = np.maximum(y, np.float32(0.0)) + u.astype(np.float32) * np.float32(e["r"])
    return _u8(y)


def _gc(x, e):
    H, W, C = x.shape
    P = H * W
    s = x.reshape(-1, C).sum(axis=0, dtype=np.int64)
    sx, so = np.float64(e["sx"]), np.float64(e["so"])
    mean = s.astype(np.float64) * sx / np.float64(P)
    a = e["b"].astype(np.float64).copy()
    w = e["w"].astype(np.float64)
    for ci in range(C):
        a = a + mean[ci] * w[ci]
    G = (np.maximum(a, 0.0) / so).astype(np.float32)
    A = np.float32(sx / so)
    return _u8(x.astype(np.float32) * A + G)


def q8_maps(blob, x, keep=False):
    """The int8 network of a version-2 blob on a padded grey input x (uint8 [H, W] or [1, 1, H, W], multiples of 32):
    maps [out, H / hs, W / hs] float32, bit for bit the C int8 path's (and {tensor: uint8 [h, w, C]} when keep)."""
    h, q = TB.read_q8(blob) if isinstance(blob, (bytes, bytearray, memoryview)) else blob
    x = np.ascontiguousarray(np.asarray(x, np.uint8).reshape(np.asarray(x).shape[-2:]))
    H, W = x.shape
    if H % TB.ALIGN or W % TB.ALIGN:
        raise ValueError("the input's sides are multiples of 32")
    t = {}
    for st in graph(h):
        k, name = st["kind"], st["name"]
        if k == "stem":
            patches = x.reshape(H // 4, 4, W // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)
            t["stem"] = _u8(_req(_mm(patches, q["stem"]["w"]), q["stem"])).reshape(H // 4, W // 4, -1)
        elif k == "sep":
            a = t[st["src"]]
            y = _dw(a, q[name + ".dw"], st["stride"], st["dil"])
            t[name + ".dw"] = y
            t[name] = _pw(y, q[name + ".pw"], res=a if st["res"] else None)
        elif k == "gc":
            t["gc"] = _gc(t[st["src"]], q["gc"])
        elif k == "lat":
            t[st["out"]] = _pw(t[st["src"]], q[name], up=t[st["up"]] if st["up"] else None)
        else:
            a = t[st["src"]]
            hh, ww, C = a.shape
            maps = _req(_mm(a.reshape(-1, C), q["head"]["w"]), q["head"]).reshape(hh, ww, -1)
            out = np.ascontiguousarray(maps[:, :, :h["out"]].transpose(2, 0, 1))
    return (out, t) if keep else out


def q8_blob(blob, sa):
    """The version-2 blob of a version-1 blob for the tensor scales sa."""
    return TB.blob_bytes_q8(blob, quantize(blob, sa))


# ----------------------------------------------------------------------------- frames

def _timing_exclusions():
    """(keys, VNs) of the timing pages (never calibration frames): hval title:page keys and 1b VN_shot keys."""
    if not TIMING_PAGES.exists():
        return set(), set()
    d = pickle.loads(TIMING_PAGES.read_bytes())
    keys = {str(p["key"]) for p in d["pages"]}
    vns = {k.split("_")[0] for k in keys if k.startswith("v") and "_" in k}
    return keys, vns


def build_frames(per_source, seed=0, check_share=0.5, sources=("1a", "1b", "1c")):
    """{"calib": [...], "check": [...]} of training-side frames: {key, source, grey}. Shards alternate between the
    two parts (disjoint shards); each part takes per_source[source] frames of each source at random."""
    import desktopdata.shards as SH
    keys_out, vns_out = _timing_exclusions()
    rng = np.random.default_rng(seed)
    out = {"calib": [], "check": []}
    for src in sources:
        shards = sorted(p.with_suffix("") for p in FRAME_SOURCES[src].glob("*.json"))
        parts = {"calib": shards[0::2], "check": shards[1::2]}
        for part, lst in parts.items():
            need = int(per_source[src] * (1.0 if part == "calib" else check_share))
            cands = []
            for sh in lst:
                samples = SH.read_shard(sh)[1]
                for i, s in enumerate(samples):
                    key = str(s.get("shot") or f"{sh.name}:{i}")
                    if key in keys_out or str(s.get("vn")) in vns_out or s.get("split", "train") != "train":
                        continue
                    cands.append((sh, i, key))
            picks = sorted((cands[j] for j in rng.permutation(len(cands))[:need]), key=lambda c: (str(c[0]), c[1]))
            cache = {}
            for sh, i, key in picks:
                if sh not in cache:
                    cache.clear()
                    cache[sh] = SH.read_shard(sh)
                data, samples = cache[sh]
                out[part].append({"key": key, "source": src, "shard": sh.name,
                                  "grey": np.ascontiguousarray(SH.canvas(data, samples[i]))})
    return out


def _inputs(frames, policies, lib=None, blob=None):
    """The padded network inputs of frames under each policy (the C detector's preprocessing: integer resize, edge
    padding), as (key, policy, x [H, W])."""
    import tsdesk_c as TC
    cm = TC.CModel(blob, lib=lib)
    out = []
    for f in frames:
        for pol in policies:
            x, _ = cm.preprocess(f["grey"], pol)
            out.append((f["key"], pol, x[0, 0]))
    return out


# ----------------------------------------------------------------------------- the maps' error

def maps_error(fm, qm, iw=None, ih=None, hs=8):
    """The int8 maps' error against the float maps of one input ([out, h, w] each; iw x ih the image part in input
    px, None: all): K sign flips, |dK|, and on the float's kernel cells (K > 0) |dO|, |dA|, |dE| (input px) and tag
    argmax changes."""
    if iw is not None:
        mw, mh = -(-iw // hs), -(-ih // hs)
        fm, qm = fm[:, :mh, :mw], qm[:, :mh, :mw]
    k = fm[0] > 0
    d = np.abs(fm - qm)
    r = {"cells": int(fm[0].size), "kernel": int(k.sum()), "k_flip": int(((fm[0] > 0) != (qm[0] > 0)).sum()),
         "dk_mean": float(d[0].mean()), "dk_p99": float(np.percentile(d[0], 99))}
    if k.any():
        r.update({"do_mean": float(d[1][k].mean()), "da_mean": float(d[2:4][:, k].mean()),
                  "de_mean": float(d[4:6][:, k].mean()), "da_p99": float(np.percentile(d[2:4][:, k], 99)),
                  "de_p99": float(np.percentile(d[4:6][:, k], 99))})
        if fm.shape[0] > 6:
            r["tag_flip"] = int((fm[6:].argmax(0)[k] != qm[6:].argmax(0)[k]).sum())
    return r


def box_agreement(fb, qb, thr=0.5):
    """Greedy one-to-one matches of two box lists (tools/tsdesk.Box) at IoU >= thr: (matched, |fb|, |qb|, mean IoU)."""
    def iou(a, b):
        x0, y0 = max(a.x, b.x), max(a.y, b.y)
        x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        u = a.w * a.h + b.w * b.h - inter
        return inter / u if u > 0 else 0.0
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(fb) for j, b in enumerate(qb)), reverse=True)
    used_f, used_q, ious = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_f or j in used_q:
            continue
        used_f.add(i)
        used_q.add(j)
        ious.append(v)
    return len(ious), len(fb), len(qb), float(np.mean(ious)) if ious else 1.0


# ----------------------------------------------------------------------------- CLI

def cmd_frames(a):
    per = {"1a": a.n_1a, "1b": a.n_1b, "1c": a.n_1c}
    t0 = time.time()
    fr = build_frames(per, seed=a.seed, check_share=a.check_share)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"frames": fr, "seed": a.seed, "per_source": per, "check_share": a.check_share,
                     "sources": {k: str(v) for k, v in FRAME_SOURCES.items()}}, f, protocol=4)
    print(json.dumps({"out": a.out, "calib": len(fr["calib"]), "check": len(fr["check"]),
                      "by_source": {p: {s: sum(1 for f in fr[p] if f["source"] == s) for s in per} for p in fr},
                      "s": round(time.time() - t0, 1)}))


def _load_frames(path, part):
    d = pickle.loads(Path(path).read_bytes())
    return d["frames"][part]


def cmd_quant(a):
    blob = Path(a.blob).read_bytes()
    t0 = time.time()
    frames = _load_frames(a.frames, "calib")
    if a.limit:
        frames = frames[:a.limit]
    xs = [x for _, _, x in _inputs(frames, a.policy.split(","), a.lib, blob)]
    net = FloatNet(blob, a.threads)
    cal = Calib(net)
    cal.pass_max(xs)
    cal.pass_hist(xs)
    methods = a.method.split(",")
    rec = {"blob": a.blob, "frames": a.frames, "calib_frames": len(frames), "inputs": len(xs), "policy": a.policy,
           "stats": cal.summary(), "scales": {}}
    for i, meth in enumerate(methods):
        sa = cal.scales(meth)
        out = a.out if len(methods) == 1 else str(Path(a.out).with_suffix("")) + f".{meth}.vtsd"
        Path(out).write_bytes(q8_blob(blob, sa))
        rec["scales"][meth] = {"file": out, "sa": sa}
    rec["s"] = round(time.time() - t0, 1)
    Path(a.record or (str(Path(a.out).with_suffix("")) + ".quant.json")).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("stats", "scales")} |
                     {"files": [v["file"] for v in rec["scales"].values()]}))


def cmd_err(a):
    import tsdesk_c as TC
    blob = Path(a.blob).read_bytes()
    frames = _load_frames(a.frames, a.part)
    if a.limit:
        frames = frames[:a.limit]
    f32 = TC.CModel(blob, lib=a.lib)
    hs = f32.hs
    res = {}
    for qpath in a.q8:
        q8 = TC.CModel(Path(qpath).read_bytes(), lib=a.lib)
        tot, boxes = {}, [0, 0, 0, []]
        for f in frames:
            for pol in a.policy.split(","):
                x, (fx, fy, iw, ih) = f32.preprocess(f["grey"], pol)
                fm, qm = f32.maps(x[0, 0]), q8.maps(x[0, 0])
                e = maps_error(fm, qm, iw, ih, hs)
                for k, v in e.items():
                    tot.setdefault(k, []).append(v)
                fb, qb = f32.decode(fm, iw, ih), q8.decode(qm, iw, ih)
                m, nf, nq, mi = box_agreement(fb, qb)
                boxes[0] += m
                boxes[1] += nf
                boxes[2] += nq
                boxes[3].append(mi)
        summ = {"inputs": len(tot["cells"]), "k_flip_rate": sum(tot["k_flip"]) / sum(tot["cells"]),
                "k_flip_of_kernel": sum(tot["k_flip"]) / max(1, sum(tot["kernel"])),
                "dk_mean": float(np.mean(tot["dk_mean"])), "dk_p99": float(np.mean(tot["dk_p99"])),
                "box_recall": boxes[0] / max(1, boxes[1]), "box_precision": boxes[0] / max(1, boxes[2]),
                "box_iou_mean": float(np.mean(boxes[3])), "boxes_f32": boxes[1], "boxes_q8": boxes[2]}
        for k in ("do_mean", "da_mean", "de_mean", "da_p99", "de_p99"):
            if k in tot:
                summ[k] = float(np.mean(tot[k]))
        if "tag_flip" in tot:
            summ["tag_flip_of_kernel"] = sum(tot["tag_flip"]) / max(1, sum(tot["kernel"]))
        res[qpath] = summ
        print(json.dumps({"q8": qpath} | summ))
    if a.out:
        Path(a.out).write_text(json.dumps({"blob": a.blob, "frames": a.frames, "part": a.part, "policy": a.policy,
                                           "results": res}, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frames", help="training-side calibration and check frames (a private pickle)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-1a", type=int, default=96)
    p.add_argument("--n-1b", type=int, default=96)
    p.add_argument("--n-1c", type=int, default=48)
    p.add_argument("--check-share", type=float, default=0.5, help="check frames per source as a share of calib's")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("quant", help="a version-2 (int8) blob from a version-1 blob and the calibration frames")
    p.add_argument("--blob", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policy", default="long:1120", help="input policies of the calibration inputs (comma list)")
    p.add_argument("--method", default="mse", help=f"comma list of {CALIB_METHODS} (one blob each)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--lib", default=None)
    p.add_argument("--record", default=None)
    p.set_defaults(fn=cmd_quant)
    p = sub.add_parser("err", help="the int8 maps' and boxes' error against the float32 C network")
    p.add_argument("--blob", required=True)
    p.add_argument("--q8", required=True, nargs="+")
    p.add_argument("--frames", required=True)
    p.add_argument("--part", default="check")
    p.add_argument("--policy", default="long:1120")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lib", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_err)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

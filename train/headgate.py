#!/usr/bin/env python3
"""The blank-frame head gate (docs/headgate.md, issue #117 item 1, eval/headgate_protocol.json).

  headgate.py features --tier N --blob VOCR5_N_w8.vocr --out DIR        (heavy: the training-side frames)
  headgate.py fit --tier N --blob VOCR5_N_w8.vocr --features DIR --out DIR --record REC.json [--prefix VOCR5]
  headgate.py info GATED.vocr

A gate is a 96 -> 16 (ReLU) -> 1 network on the output head's input, as the W8I head sees it, fit to say
"this frame's argmax is the blank" before the head runs, so the runtime skips the head there
(runtime/vocr.h, vocr_opts.head_gate). It is stored as a section after the unchanged version-1 blob:

  section (little-endian, 16-byte aligned, at the blob's 16-byte aligned end; runtime/vocr.c parse_gate)
    0 char[4] "VOCG"   4 u16 version (1)   6 u16 header bytes (64)   8 u32 section bytes   12 u32 flags (0)
   16 u16 k (the head's K)   18 u16 hidden   20 u32 the blob's bytes   24 u32 the blob's CRC-32 (its field 40)
   28 f32 t   32 f32 b2   36 u32 w1 offset   40 u32 s1 offset   44 u32 b1 offset   48 u32 w2 offset
   52 u32 CRC-32 of the section's bytes [64, end)   56..63 zero
    w1 int8 [hidden/8][k][8] (a pointwise layer's packing), s1 / b1 / w2 float32 [hidden]
  The CRC covers the arrays, not the header: t and b2 are content outside it, as the base blob's header fields are
  outside its CRC; vocr_load checks the header's structure (and rejects a NaN t or b2), not their values.

A runtime without gate support reads the version-1 blob as before (it reads the header's byte count only);
a blob without the section runs exactly as before in this one.

The gate's arithmetic (vocr.c gate_tile; gate_scores below is a bit-exact twin):
  q, sx = the frame's head input quantised as W8I does (vocr.c quant16; tests/vocr_ref.quant16_exact)
  h[j] = relu(float(sum_k w1[j][k] q[k]) * (s1[j] * sx) + b1[j])          (exact int32 sums, float steps)
  score = b2, then score = score + w2[j] * h[j] for j = 0 .. hidden - 1     (float32, in this order)
and a frame is skipped iff score > t.

Training (the protocol's "training"): the frozen model's own frames on the console-cut crops of VOCR5's
fine-tune pools (GRAD crops and their refined copies); features = vocr_extra.head_in (the head's input, from
the C runtime in W8I), label = (the W8I argmax is the blank). Split by VN (sha1(vn) % 5 == 0: calibration).
One float fit per tier (standardised inputs, folded into layer 1 at export), layer 1 quantised to int8 per
hidden unit; the threshold from the exact int8 scores of the calibration frames with a non-blank argmax.
"""
import argparse
import hashlib
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "runtime", ROOT / "runtime" / "tests", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MAGIC, VERSION, HEADER = b"VOCG", 1, 64
HIDDEN = 16
EPS = (1e-3, 1e-4)                   # the protocol's threshold and its pre-registered fallback
FIT = dict(lr=3e-3, epochs=12, batch=4096, seed=0)
F32 = np.float32

HERD = Path("/local/home/skerraut/herd-vita-ocr")
POOLS = {   # eval/vocr5_ft_sources.json 'owg' and 'ntg' (eval/vocr5_pools.sha256): console-cut crops
    "own_grad": HERD / "shared/private/vocr5-train/pools/v3/own_grad.npz",
    "own_ref": HERD / "shared/private/vocr5-train/pools/v3/own_ref.npz",
    "nat_grad": HERD / "shared/artifacts/vocr5-train/pools/v3/nat_grad.npz",
    "nat_ref": HERD / "shared/artifacts/vocr5-train/pools/v3/nat_ref.npz",
    "dntv_grad": HERD / "shared/artifacts/vocr5-train/pools/v3/dntv_grad.npz",
    "dntv_ref": HERD / "shared/artifacts/vocr5-train/pools/v3/dntv_ref.npz",
    "dnst_grad": HERD / "shared/artifacts/vocr5-train/pools/v3/dnst_grad.npz",
    "dnst_ref": HERD / "shared/artifacts/vocr5-train/pools/v3/dnst_ref.npz",
}


def _align(n, a=16):
    return (n + a - 1) // a * a


def sha256(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the section

def base_of(data):
    """The version-1 blob's bytes (header field 8) of a blob with or without a gate section."""
    total = struct.unpack_from("<I", data, 8)[0]
    if data[:4] != b"VOCR" or total > len(data):
        raise ValueError("not a VOCR blob")
    return bytes(data[:total])


def gate_section(base, params, t):
    """The VOCG section for the version-1 blob `base` (bytes): params = quantise(...)'s dict, t the threshold."""
    q, s1, b1, w2 = params["q"], params["scale"], params["bias"], params["w2"]
    hid, k = q.shape
    if hid % 8 or not 8 <= hid <= 64 or q.dtype != np.int8:
        raise ValueError("hidden units: a multiple of 8, 8..64, int8 weights")
    w1 = np.ascontiguousarray(q.reshape(hid // 8, 8, k).transpose(0, 2, 1))       # [hid/8][k][8]
    o_w1 = HEADER
    o_s1 = _align(o_w1 + w1.size)
    o_b1 = _align(o_s1 + 4 * hid)
    o_w2 = _align(o_b1 + 4 * hid)
    n = _align(o_w2 + 4 * hid)
    sec = bytearray(n)
    sec[o_w1:o_w1 + w1.size] = w1.tobytes()
    sec[o_s1:o_s1 + 4 * hid] = np.asarray(s1, "<f4").tobytes()
    sec[o_b1:o_b1 + 4 * hid] = np.asarray(b1, "<f4").tobytes()
    sec[o_w2:o_w2 + 4 * hid] = np.asarray(w2, "<f4").tobytes()
    total = struct.unpack_from("<I", base, 8)[0]
    base_crc = struct.unpack_from("<I", base, 40)[0]
    if total != len(base):
        raise ValueError("base must be exactly the version-1 blob")
    head_k = head_op(base)["k"]
    if head_k != k:
        raise ValueError(f"gate input {k} != the head's K {head_k}")
    struct.pack_into("<4sHHII2HIIffIIII", sec, 0, MAGIC, VERSION, HEADER, n, 0, k, hid, total, base_crc,
                     F32(t), F32(params["b2"]), o_w1, o_s1, o_b1, o_w2)
    struct.pack_into("<I", sec, 52, zlib.crc32(bytes(sec[HEADER:])) & 0xFFFFFFFF)
    return bytes(sec)


def append_gate(base, params, t):
    """The gated blob: the version-1 blob, zero padding to 16 bytes, the section."""
    base = bytes(base)
    pad = _align(len(base)) - len(base)
    return base + bytes(pad) + gate_section(base, params, t)


def read_gate(data):
    """The section's fields and arrays (as the runtime reads them), or None without one. Checks the CRC."""
    data = bytes(data)
    total = struct.unpack_from("<I", data, 8)[0]
    off = _align(total)
    if len(data) < off + 4 or data[off:off + 4] != MAGIC:
        return None
    (magic, ver, hb, n, flags, k, hid, base_bytes, base_crc, t, b2, o_w1, o_s1, o_b1, o_w2,
     crc) = struct.unpack_from("<4sHHII2HIIffIIIII", data, off)
    sec = data[off:off + n]
    if ver != VERSION or hb != HEADER or flags or len(sec) != n or base_bytes != total:
        raise ValueError("bad VOCG section")
    if zlib.crc32(sec[HEADER:]) & 0xFFFFFFFF != crc or base_crc != struct.unpack_from("<I", data, 40)[0]:
        raise ValueError("VOCG CRC mismatch")
    w1 = np.frombuffer(sec, np.int8, hid * k, o_w1).reshape(hid // 8, k, 8).transpose(0, 2, 1).reshape(hid, k)
    return {"k": k, "hidden": hid, "t": F32(t), "b2": F32(b2), "bytes": n, "offset": off,
            "q": w1.copy(), "scale": np.frombuffer(sec, "<f4", hid, o_s1).copy(),
            "bias": np.frombuffer(sec, "<f4", hid, o_b1).copy(), "w2": np.frombuffer(sec, "<f4", hid, o_w2).copy()}


def head_op(blob):
    """The head op record's fields of a blob (type, k, cout_pad)."""
    n_ops, ops = struct.unpack_from("<H", blob, 22)[0], struct.unpack_from("<I", blob, 28)[0]
    r = ops + 64 * (n_ops - 1)
    return {"type": blob[r], "k": struct.unpack_from("<H", blob, r + 32)[0],
            "cout": struct.unpack_from("<H", blob, r + 10)[0]}


# ---------------------------------------------------------------- the twin

def quantise(w1, b1, w2, b2):
    """A float gate (w1 [hidden][k] on the raw head input, b1, w2 [hidden], b2) as stored: layer 1 int8 per
    hidden unit (train/export.quantize_rows), the rest float32."""
    import export
    q, scale = export.quantize_rows(np.asarray(w1, np.float64))
    return {"q": q, "scale": scale.astype(F32), "bias": np.asarray(b1, F32), "w2": np.asarray(w2, F32),
            "b2": F32(b2)}


def gate_scores(params, x):
    """Each frame's gate score, bit for bit as vocr.c gate_tile computes it. x: frames x k float32 (the
    head's input, vocr_extra.head_in)."""
    import vocr_ref as R
    x = np.ascontiguousarray(x, F32)
    if not len(x):
        return np.zeros(0, F32)
    h = R.matmul_exact(x, {"w": params["q"], "scale": params["scale"], "bias": params["bias"], "relu": True,
                           "cout": params["q"].shape[0]}, "w8i")
    s = np.full(len(x), params["b2"], F32)
    for j in range(h.shape[1]):
        s = (s + (params["w2"][j] * h[:, j]).astype(F32)).astype(F32)
    return s


def ctc(best, vocab):
    out, last = [], 0
    for c in best:
        c = int(c)
        if c and c != last:
            out.append(vocab[c])
        last = c
    return "".join(out)


def simulate(out, skip, vocab):
    """An ungated run's top-K output (pyvocr run_topk: text, topv, topc, blank, best) as the gated run returns
    it when the frames `skip` (bool per frame) are skipped: their best 0, top-K [blank logit, -inf, ...]
    with classes 0; the greedy text and conf_min / conf_mean recomputed as vocr.c does (float32)."""
    skip = np.asarray(skip, bool)
    tv = out["topv"].copy()
    tc = out["topc"].copy()
    best = out["best"].copy()
    bl = out["blank"]
    k = tv.shape[1]
    best[skip] = 0
    tv[skip, 0] = bl[skip]
    tv[skip, 1:] = -np.inf
    tc[skip] = 0
    res = dict(out, topv=tv, topc=tc, best=best, text=ctc(best, vocab), gated=int(skip.sum()))
    conf = []
    last = 0
    for f, c in enumerate(best):             # vocr.c run_head: a frame that emits a character
        c = int(c)
        if c and c != last:
            s = F32(1)
            for j in range(1, k):
                s = F32(s + exp_neg(F32(tv[f, j] - tv[f, 0])))
            conf.append(F32(F32(1) / s))
        last = c
    res["chars"] = len(conf)
    if conf:
        res["conf_min"] = float(min(conf))
        tot = F32(0)
        for p in conf:
            tot = F32(tot + p)
        res["conf_mean"] = float(F32(tot / F32(len(conf))))
    else:
        res["conf_min"] = res["conf_mean"] = 1.0
    return res


def _f32bits(u):
    return np.frombuffer(struct.pack("<I", u), "<f4")[0]


# vocr.c exp_neg's float literals, bit for bit
_LOG2E = _f32bits(0x3FB8AA3B)
_POLY = [_f32bits(u) for u in (0x39218489, 0x3AAEC3FF, 0x3C1D955B, 0x3D635847, 0x3E75FDF0, 0x3F317218, 0x3F800000)]


def exp_neg(x):
    """vocr.c exp_neg (float32 only; the confidence's exp)."""
    x = F32(x)
    if not x > F32(-30):
        x = F32(-30)
    if x > 0:
        x = F32(0)
    t = F32(x * _LOG2E)
    e = int(t)
    if F32(e) > t:
        e -= 1
    f = F32(t - F32(e))
    p = _POLY[0]
    for c in _POLY[1:]:
        p = F32(F32(p * f) + c)
    return F32(p * _f32bits((e + 127) << 23))


# ---------------------------------------------------------------- training data

def pool_rows():
    """Every row of the training pools: [(file name, row index, vn, kind)] and an iterator of crops."""
    rows = []
    for name, path in POOLS.items():
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        for i, m in enumerate(meta):
            rows.append((name, i, m["vn"], m.get("kind")))
    return rows


def check_hygiene(rows):
    """No row of a reserved test VN or of the development sample (the protocol's 'hygiene')."""
    r = json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text(encoding="utf-8"))
    bad = {vn for _, _, vn, _ in rows if vn in set(r["test"]) or vn in set(r["dev_sample"])}
    if bad:
        raise SystemExit(f"training rows of reserved/dev_sample VNs: {sorted(bad)[:10]}")
    return {"rows": len(rows), "vns": len({vn for _, _, vn, _ in rows}), "reserved_or_dev_sample_vns": 0,
            "reserved_sha256": sha256(ROOT / "eval" / "manifests" / "reserved.json")}


def is_cal(vn):
    return int(hashlib.sha1(vn.encode()).hexdigest(), 16) % 5 == 0


_W = {}


def _init(blob):
    from pyvocr import Vocr
    _W["rt"] = Vocr(blob, mode="w8i", max_width=1024)
    _W["pools"] = {}


def _crop(name, i):
    z = _W["pools"].get(name)
    if z is None:
        d = np.load(POOLS[name])
        z = _W["pools"][name] = (d["data"], d["offsets"], d["shapes"])
    data, offs, shapes = z
    h, w = (int(v) for v in shapes[i])
    return np.ascontiguousarray(data[offs[i]:offs[i + 1]].reshape(h, w))


def _work(job):
    j, name, i = job
    rt = _W["rt"]
    g = _crop(name, i)
    rt.run(g, head_in=True)
    return j, rt.last_head_in.copy(), rt.last_best.copy()


def cmd_features(a):
    import multiprocessing as mp
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pool_rows()
    hyg = check_hygiene(rows)
    jobs = [(j, name, i) for j, (name, i, _, _) in enumerate(rows)]
    feats, bests = [None] * len(rows), [None] * len(rows)
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.blob,)) as pool:
        for n, (j, x, b) in enumerate(pool.imap_unordered(_work, jobs, chunksize=16)):
            feats[j], bests[j] = x, b
            if n % 2000 == 0:
                print(f"  {a.tier}: {n}/{len(jobs)}", flush=True)
    off = np.zeros(len(rows) + 1, np.int64)
    off[1:] = np.cumsum([len(b) for b in bests])
    np.savez(out / f"features_{a.tier}.npz", x=np.concatenate(feats).astype(F32), best=np.concatenate(bests),
             off=off, vn=np.array([r[2] for r in rows]), kind=np.array([str(r[3]) for r in rows]),
             src=np.array([r[0] for r in rows]), idx=np.array([r[1] for r in rows], np.int64))
    info = {"tier": a.tier, "blob": Path(a.blob).name, "blob_sha256": sha256(a.blob), **hyg,
            "frames": int(off[-1]), "pools": {k: sha256(p) for k, p in POOLS.items()},
            "seconds": round(time.time() - t0, 1)}
    (out / f"features_{a.tier}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


# ---------------------------------------------------------------- fit

def fit_float(x, y, seed=FIT["seed"]):
    """The float gate on standardised inputs; returns its parameters on the raw input (folded)."""
    import torch
    torch.set_num_threads(1)              # deterministic
    torch.manual_seed(seed)
    mu = x.mean(0, dtype=np.float64)
    sd = x.std(0, dtype=np.float64) + 1e-6
    X = torch.from_numpy(((x - mu) / sd).astype(F32))
    Y = torch.from_numpy(y.astype(F32))
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 1))
    opt = torch.optim.Adam(net.parameters(), lr=FIT["lr"])
    n, bs = len(X), FIT["batch"]
    steps = FIT["epochs"] * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(FIT["epochs"]):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(X[b]).squeeze(1), Y[b])
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        final = float(lossf(net(X).squeeze(1), Y))
    w1 = net[0].weight.detach().double().numpy()
    b1 = net[0].bias.detach().double().numpy()
    w1f = w1 / sd[None, :]
    b1f = b1 - w1f @ mu
    return {"w1": w1f, "b1": b1f, "w2": net[2].weight.detach().double().numpy()[0],
            "b2": float(net[2].bias.detach().double().numpy()[0]), "loss": final}


def auc(score, y):
    order = np.argsort(score, kind="stable")
    r = np.empty(len(score), np.float64)
    s = score[order]
    i = 0
    while i < len(s):                      # average ranks of ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos = int(y.sum())
    nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None


def threshold(scores_nonblank, eps):
    """t = the (floor(eps n) + 1)-th largest score of the n calibration frames with a non-blank argmax: at most
    floor(eps n) of them score above it."""
    s = np.sort(np.asarray(scores_nonblank, F32))[::-1]
    m = int(np.floor(eps * len(s)))
    return F32(s[m]), m


def _c_scores(blob_bytes, rows, mode):
    """The C runtime's gate scores (and skipped counts) of training rows [(name, i)], one process."""
    from pyvocr import Vocr
    rt = Vocr(blob_bytes, mode="w8i", max_width=1024, head_gate=mode)
    _W["pools"] = {}
    scores, gated, best = [], [], []
    for name, i in rows:
        o = rt.run_topk(_crop(name, i), 2, gate_score=True)
        scores.append(o["gate_score"].copy())
        gated.append(o["gated"])
        best.append(o["best"].copy())
    return scores, gated, best


def cmd_fit(a):
    t0 = time.time()
    feat = Path(a.features)
    z = np.load(feat / f"features_{a.tier}.npz")
    info = json.loads((feat / f"features_{a.tier}.json").read_text())
    base = Path(a.blob).read_bytes()
    if sha256(base) != info["blob_sha256"]:
        raise SystemExit("the features were extracted with another blob")
    x, best, off = z["x"], z["best"], z["off"]
    vn = z["vn"]
    line = np.repeat(np.arange(len(off) - 1), np.diff(off))
    cal_line = np.array([is_cal(v) for v in vn])
    cal = cal_line[line]
    y = best == 0
    g = fit_float(x[~cal], y[~cal])
    params = quantise(g["w1"], g["b1"], g["w2"], g["b2"])
    s = gate_scores(params, x)
    rec = {"tier": a.tier, "blob": Path(a.blob).name, "blob_sha256": sha256(base), "features": info,
           "fit": {**FIT, "hidden": HIDDEN, "frames": int((~cal).sum()), "lines": int((~cal_line).sum()),
                   "vns": len(set(vn[~cal_line])), "blank_share": round(float(y[~cal].mean()), 5),
                   "loss": round(g["loss"], 6)},
           "calibration": {"frames": int(cal.sum()), "lines": int(cal_line.sum()), "vns": len(set(vn[cal_line])),
                           "blank_share": round(float(y[cal].mean()), 5),
                           "auc_int8": round(auc(s[cal], y[cal]), 6)},
           "gates": {}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nb = cal & ~y
    for eps in EPS:
        t, m = threshold(s[nb], eps)
        blob = append_gate(base, params, t)
        name = f"{a.prefix}_{a.tier}_w8g.vocr" if eps == EPS[0] else f"{a.prefix}_{a.tier}_w8g_eps{eps:g}.vocr"
        (out / name).write_bytes(blob)
        skip = s > t
        rec["gates"][f"{eps:g}"] = {
            "file": name, "sha256": sha256(blob), "bytes": len(blob), "base_bytes": len(base),
            "section_bytes": len(blob) - _align(len(base)), "t": float(t), "t_hex": struct.pack("<f", t).hex(),
            "nonblank_cal_frames": int(nb.sum()), "allowed": m,
            "cal_skip_share": round(float(skip[cal].mean()), 5),
            "cal_nonblank_skipped": int(skip[nb].sum()),
            "cal_nonblank_skipped_share": round(float(skip[nb].mean()), 6),
            "cal_blank_recall": round(float(skip[cal & y].mean()), 5),
            "fit_skip_share": round(float(skip[~cal].mean()), 5),
            "fit_nonblank_skipped_share": round(float(skip[~cal & ~y].mean()), 6),
            "per_platform_cal_skip": {k: round(float(skip[cal & (z["kind"][line] == k)].mean()), 5)
                                      for k in ("vita", "psp")}}
    # the C runtime gives the twin's scores and skips, bit for bit, on every calibration line
    blob = (out / rec["gates"][f"{EPS[0]:g}"]["file"]).read_bytes()
    t = F32(rec["gates"][f"{EPS[0]:g}"]["t"])
    cal_rows = [(str(z["src"][j]), int(z["idx"][j])) for j in np.nonzero(cal_line)[0]]
    c_s, c_g, c_b = _c_scores(blob, cal_rows, "score")
    tw = [s[off[j]:off[j + 1]] for j in np.nonzero(cal_line)[0]]
    same = all(np.array_equal(cs.view(np.uint32), ts.view(np.uint32)) for cs, ts in zip(c_s, tw))
    same_best = all(np.array_equal(cb, best[off[j]:off[j + 1]]) for cb, j in zip(c_b, np.nonzero(cal_line)[0]))
    _, c_g_on, c_b_on = _c_scores(blob, cal_rows, "on")
    same_skip = all(gn == int((ts > t).sum()) for gn, ts in zip(c_g_on, tw))
    blank_on = all(np.all(cb[ts > t] == 0) for cb, ts in zip(c_b_on, tw))
    rec["c_equals_twin"] = {"lines": len(cal_rows), "scores_bitwise": bool(same), "score_mode_best_unchanged":
                            bool(same_best), "on_mode_skipped_counts": bool(same_skip),
                            "on_mode_skipped_frames_blank": bool(blank_on)}
    if not (same and same_best and same_skip and blank_on):
        raise SystemExit(f"C runtime != twin: {rec['c_equals_twin']}")
    rec["seconds"] = round(time.time() - t0, 1)
    Path(a.record).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("tier", "fit", "calibration", "c_equals_twin")}), flush=True)
    for e, v in rec["gates"].items():
        print(e, json.dumps(v), flush=True)


def cmd_info(a):
    data = Path(a.blob).read_bytes()
    g = read_gate(data)
    if g is None:
        print(json.dumps({"blob": a.blob, "gate": None}))
        return
    print(json.dumps({"blob": a.blob, "bytes": len(data), "base_bytes": len(base_of(data)), "section_bytes": g["bytes"],
                      "k": g["k"], "hidden": g["hidden"], "t": float(g["t"]), "b2": float(g["b2"]),
                      "macs_per_frame": g["k"] * g["hidden"] + g["hidden"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("features")
    p.add_argument("--tier", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("fit")
    p.add_argument("--tier", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--prefix", default="VOCR5", help="the gated blobs' names: PREFIX_TIER_w8g.vocr (VOCR6: "
                                                     "eval/vocr6_gate_protocol.json)")
    p = sub.add_parser("info")
    p.add_argument("blob")
    a = ap.parse_args()
    {"features": cmd_features, "fit": cmd_fit, "info": cmd_info}[a.cmd](a)


if __name__ == "__main__":
    main()

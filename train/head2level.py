#!/usr/bin/env python3
"""head2level: the two-level output head of vita-vn-ocr #117 item 2 (eval/head2level_protocol.json).

  head2level.py rank --out RANK.json            class counts in the labels of VOCR5's fine-tune pools (real sources),
                                                and each tier vocab's frequency order
  head2level.py permute --blob B --rank RANK.json --out P.vocr
                                                the blob with its head rows and vocab in that order (a valid
                                                version-1 blob: every class's logit is the same, bit for bit)
  head2level.py info --blob P.vocr              the blob's VOC2 trailer, if any

Library (eval/head2level_*.py): the VOC2 trailer writer and reader, the exact numpy twin of runtime/vocr.c's
two-level head (run_head2: level 1, the escape, level 2, first maxima and top-K in vocr_run's order), the escape's
fit / export / threshold, and the 48-wide rare block (r48).

The two-level head (runtime/vocr.h vocr_2level): level 1 = the head's first l1_blocks 8-row blocks (the blank and
the most frequent classes), on every frame; the escape, a hidden-16 MLP on the head's quantised input, level 1's
largest non-blank logit l1 and its blank logit b0, says which frames also need the other rows (level 2: the head's
own rows, or with rare_k the rows of a rare_k-wide projection P of the head input).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "runtime" / "tests", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import export  # noqa: E402

F32 = np.float32
CHUNK = 512
TWO_MAGIC, TWO_VERSION, TWO_HEADER = b"VOC2", 1, 128
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _al16(n):
    return (n + 15) // 16 * 16


# ---------------------------------------------------------------- the ranking (protocol "ranking")

def ft_real_files(sources=ROOT / "eval" / "vocr5_ft_sources.json"):
    """[(group, file)] of VOCR5's fine-tune pools' real sources, refined copies included."""
    d = json.loads(Path(sources).read_text(encoding="utf-8"))
    out = []
    for e in d["real"]:
        for f in list(e["files"]) + list(e.get("refined", [])):
            out.append((e["group"], f))
    return out


def pinned_hashes(path=ROOT / "eval" / "vocr5_pools.sha256"):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            h, f = line.split(None, 1)
            out[f.strip()] = h
    return out


def pool_lines(files, check_hashes=True):
    """Every distinct line of the files once: {(group, id): (vn, label)}; asserts the pinned hashes and that no
    line belongs to a reserved VN (test or development sample)."""
    from common import is_reserved_vn
    pins = pinned_hashes()
    lines, rows = {}, 0
    for group, f in files:
        if check_hashes and pins.get(f) != sha256(f):
            raise SystemExit(f"{f}: not the file eval/vocr5_pools.sha256 pins")
        z = np.load(f)
        labels = json.loads(str(z["labels"]))
        meta = json.loads(str(z["meta"]))
        for lab, m in zip(labels, meta):
            rows += 1
            if is_reserved_vn(m["vn"]):
                raise SystemExit(f"{f}: a row of reserved VN {m['vn']}")
            lines.setdefault((group, m["id"]), (m["vn"], lab))
    return lines, rows


def rank_order(vocab, counts, order6459):
    """perm (new class -> old class) of a model vocab (vocab[0] = '' the blank): class 0 first, then decreasing
    count, ties and unseen classes in the 6,459 list's order, then any class outside it in vocab order."""
    big = 10 ** 9
    cls = list(range(1, len(vocab)))
    key = sorted(cls, key=lambda c: (-counts.get(vocab[c], 0), order6459.get(vocab[c], big), c))
    return np.array([0] + key, np.int64)


def cmd_rank(args):
    from collections import Counter
    files = ft_real_files()
    lines, rows = pool_lines(files)
    cnt = Counter()
    for vn, lab in lines.values():
        cnt.update(c for c in lab if not c.isspace())
    v6459 = (ROOT / "data" / "vocab_vn_6459.txt").read_text(encoding="utf-8").split("\n")
    v6459 = [c for c in v6459 if c]
    order6459 = {c: i for i, c in enumerate(v6459)}
    out = {"kind": "head2level-rank", "protocol": "eval/head2level_protocol.json 'ranking'",
           "files": {f: sha256(f) for _, f in files}, "rows": rows, "distinct_lines": len(lines),
           "vns": len({v for v, _ in lines.values()}), "characters": int(sum(cnt.values())),
           "distinct_characters": len(cnt), "vocab_6459_sha256": sha256(ROOT / "data" / "vocab_vn_6459.txt"),
           "counts": dict(cnt.most_common()), "tiers": {}}
    for name in ("vocab_4987_proof.txt", "vocab_vn_5631.txt"):
        p = ROOT / "data" / name
        vocab = [""] + [c for c in p.read_text(encoding="utf-8").split("\n") if c]
        perm = rank_order(vocab, cnt, order6459)
        out["tiers"][name] = {"sha256": sha256(p), "classes": len(vocab),
                              "order": [vocab[int(c)] for c in perm[1:]],
                              "seen_classes": sum(1 for c in vocab[1:] if cnt.get(c, 0)),
                              "cover": {str(k): round(sum(cnt.get(vocab[int(c)], 0) for c in perm[1:k + 1]) /
                                                      max(1, sum(cnt.values())), 6)
                                        for k in (500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000)}}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("counts", "tiers", "files")}))
    for k, v in out["tiers"].items():
        print(k, "seen", v["seen_classes"], "cover", v["cover"])


def perm_for_blob(vocab, rank):
    """perm (new -> old) of a blob vocab from a rank file's order for that vocab."""
    for t in rank["tiers"].values():
        if t["classes"] == len(vocab) and set(t["order"]) == set(vocab[1:]):
            pos = {c: i for i, c in enumerate(vocab)}
            if len(pos) != len(vocab):
                raise ValueError("the vocab repeats a class text")
            return np.array([0] + [pos[c] for c in t["order"]], np.int64)
    raise SystemExit("no ranking for this blob's vocab")


# ---------------------------------------------------------------- permuted blob

def _head_record(data):
    (_, _, _, total, flags, height, max_w, stride, n_ops, n_classes, ops_off, v_off, v_bytes, crc, _, _) = \
        struct.unpack_from("<4sHHII4HIIIIII16s", data, 0)
    rec = ops_off + export.OP_BYTES * (n_ops - 1)
    (t, fl, kh, kw, sh, sw, ph, pw, cin, cout, h_in, h_out, w_off, w_bytes, b_off, s_off, k, cout_pad) = \
        struct.unpack_from("<8B4H4I2H", data, rec)
    assert t == export.T_HEAD and fl & export.F_INT8, "an int8 blob's head"
    return dict(total=total, n_classes=n_classes, v_off=v_off, v_bytes=v_bytes, w_off=w_off, b_off=b_off,
                s_off=s_off, k=k, cout_pad=cout_pad, cout=cout, flags=fl)


def permute_blob(data, perm):
    """The int8 blob with head rows (weights, scale, bias) and vocab permuted: new class j is old class perm[j].
    Every byte else is unchanged but the CRC; padded rows stay at the end. Returns bytes."""
    data = bytes(data)
    h = _head_record(data)
    V, K, P = h["n_classes"], h["k"], h["cout_pad"]
    assert len(perm) == V and perm[0] == 0 and sorted(perm.tolist()) == list(range(V))
    out = bytearray(data)
    w = np.frombuffer(data, np.int8, P * K, h["w_off"]).reshape(P // 8, K, 8).transpose(0, 2, 1).reshape(P, K)
    full = np.concatenate([perm, np.arange(V, P)])
    w2 = w[full]
    out[h["w_off"]:h["w_off"] + P * K] = np.ascontiguousarray(w2.reshape(P // 8, 8, K).transpose(0, 2, 1)).tobytes()
    for off in (h["s_off"], h["b_off"]):
        if off:
            a = np.frombuffer(data, "<f4", P, off)
            out[off:off + 4 * P] = a[full].astype("<f4").tobytes()
    offs = np.frombuffer(data, "<u2", V + 1, h["v_off"]).astype(np.int64)
    raw = data[h["v_off"] + 2 * (V + 1):h["v_off"] + 2 * (V + 1) + h["v_bytes"]]
    enc = [raw[offs[c]:offs[c + 1]] for c in range(V)]
    enc2 = [enc[int(c)] for c in perm]
    no = np.zeros(V + 1, np.int64)
    no[1:] = np.cumsum([len(e) for e in enc2])
    assert no[-1] == h["v_bytes"]
    blob = no.astype("<u2").tobytes() + b"".join(enc2)
    out[h["v_off"]:h["v_off"] + len(blob)] = blob
    crc = zlib.crc32(bytes(out[export.HEADER_BYTES:h["total"]])) & 0xFFFFFFFF
    struct.pack_into("<I", out, 40, crc)
    return bytes(out)


def cmd_permute(args):
    data = Path(args.blob).read_bytes()
    rank = json.loads(Path(args.rank).read_text(encoding="utf-8"))
    vocab = export.read_blob(data)["vocab"]
    perm = perm_for_blob(vocab, rank)
    out = permute_blob(data, perm)
    export.read_blob(out)
    Path(args.out).write_bytes(out)
    print(json.dumps({"blob": args.blob, "sha256": sha256(args.blob), "out": args.out, "out_sha256": sha256(args.out),
                      "bytes": len(out)}))


# ---------------------------------------------------------------- the VOC2 trailer

def pack8(w):
    """int8 [rows][K] (rows a multiple of 8) -> [rows / 8][K][8] bytes, the layout mm_i16 reads."""
    r, k = w.shape
    assert r % 8 == 0
    return np.ascontiguousarray(w.astype(np.int8).reshape(r // 8, 8, k).transpose(0, 2, 1)).tobytes()


def write_trailer(base, l1_blocks, esc, rare=None):
    """The VOC2 trailer (runtime/vocr.c parse_two) for the base blob `base` (bytes): esc = dict(q int8 [H][K],
    scale, bias, wl1, wb0, w2 float32 [H], b2, t float32); rare = None (r96) or dict(p_q int8 [RK][K], p_scale,
    p_bias [RK], r_q int8 [R][RK] (R = the head's rare rows padded to a multiple of 8), r_scale, r_bias [R])."""
    total, crc = struct.unpack_from("<I", base, 8)[0], struct.unpack_from("<I", base, 40)[0]
    H = esc["q"].shape[0]
    secs, pos = [], [TWO_HEADER]

    def put(b):
        off = pos[0]
        secs.append((off, b))
        pos[0] = _al16(off + len(b))
        return off

    f4 = lambda a: np.asarray(a, "<f4").tobytes()  # noqa: E731
    offs = [put(pack8(esc["q"])), put(f4(esc["scale"])), put(f4(esc["bias"])), put(f4(esc["wl1"])),
            put(f4(esc["wb0"])), put(f4(esc["w2"]))]
    rk = 0
    if rare is not None:
        rk = rare["p_q"].shape[0]
        offs += [put(pack8(rare["p_q"])), put(f4(rare["p_scale"])), put(f4(rare["p_bias"])), put(pack8(rare["r_q"])),
                 put(f4(rare["r_scale"])), put(f4(rare["r_bias"]))]
    else:
        offs += [0] * 6
    size = pos[0]
    t = bytearray(size)
    for off, b in secs:
        t[off:off + len(b)] = b
    struct.pack_into("<4sHHII", t, 0, TWO_MAGIC, TWO_VERSION, TWO_HEADER, size, 0)
    struct.pack_into("<IIHHHHff12I", t, 16, total, crc, l1_blocks, H, rk, 0, F32(esc["t"]), F32(esc["b2"]), *offs)
    struct.pack_into("<I", t, 12, zlib.crc32(bytes(t[16:])) & 0xFFFFFFFF)
    return bytes(t)


def attach(base, trailer):
    """base blob (its own bytes, or base + earlier trailers) + zero padding to 16 + trailer."""
    base = bytes(base)
    return base + bytes(_al16(len(base)) - len(base)) + trailer


def read_trailer(data):
    """The VOC2 trailer of a blob as numpy arrays, or None (scans like runtime/vocr.c find_trailer)."""
    total = struct.unpack_from("<I", data, 8)[0]
    off = _al16(total)
    while off + 16 <= len(data):
        n = struct.unpack_from("<I", data, off + 8)[0]
        if data[off:off + 4] == TWO_MAGIC:
            break
        if n < 16 or n % 16 or n > len(data) - off:
            return None
        off += n
    else:
        return None
    p = data[off:off + n]
    if zlib.crc32(p[16:]) & 0xFFFFFFFF != struct.unpack_from("<I", p, 12)[0]:
        raise ValueError("VOC2 CRC mismatch")
    b_total, b_crc, l1b, H, rk, _, t, b2 = struct.unpack_from("<IIHHHHff", p, 16)
    o = struct.unpack_from("<12I", p, 40)
    hd = _head_record(data)
    K, P = hd["k"], hd["cout_pad"]

    def i8(off, rows, k):
        return np.frombuffer(p, np.int8, rows * k, off).reshape(rows // 8, k, 8).transpose(0, 2, 1).reshape(rows, k)

    def f(off, n):
        return np.frombuffer(p, "<f4", n, off).astype(F32)
    esc = dict(q=i8(o[0], H, K), scale=f(o[1], H), bias=f(o[2], H), wl1=f(o[3], H), wb0=f(o[4], H), w2=f(o[5], H),
               b2=F32(b2), t=F32(t))
    rare = None
    if rk:
        R = P - 8 * l1b
        rare = dict(p_q=i8(o[6], rk, K), p_scale=f(o[7], rk), p_bias=f(o[8], rk), r_q=i8(o[9], R, rk),
                    r_scale=f(o[10], R), r_bias=f(o[11], R))
    return dict(offset=off, bytes=n, base_bytes=b_total, base_crc=b_crc, l1_blocks=l1b, hidden=H, rare_k=rk, esc=esc,
                rare=rare)


def cmd_info(args):
    data = Path(args.blob).read_bytes()
    tr = read_trailer(data)
    b = export.read_blob(data)
    info = None if tr is None else {k: tr[k] for k in ("offset", "bytes", "l1_blocks", "hidden", "rare_k")}
    if info is not None:
        info["t"] = float(tr["esc"]["t"])
    print(json.dumps({"bytes": len(data), "base_bytes": b["bytes"], "classes": b["n_classes"], "trailer": info}))


# ---------------------------------------------------------------- the exact twin (runtime/vocr.c W8I)

def quant16_exact(x):
    """vocr.c quant16 per row: (q as float64 integers, sx float32)."""
    from vocr_ref import quant16_exact as q16
    return q16(np.asarray(x, F32))


def mm_exact(q, sx, w, scale, bias):
    """vocr.c mm_i16: q [P][K] (float64 integers), sx [P], w int8 [N][K] -> [P][N] float32:
    S * (scale * sx) + bias, S = 0 + float(c_0) + ... over 512-term chunks."""
    P, K = q.shape
    wd = np.asarray(w, np.float64)
    acc = np.zeros((P, wd.shape[0]), F32)
    for k0 in range(0, K, CHUNK):
        acc = (acc + (q[:, k0:k0 + CHUNK] @ wd[:, k0:k0 + CHUNK].T).astype(F32)).astype(F32)
    y = (acc * (np.asarray(scale, F32)[None, :] * np.asarray(sx, F32)[:, None]).astype(F32)).astype(F32)
    if bias is not None:
        y = (y + np.asarray(bias, F32)[None, :]).astype(F32)
    return y


def head_arrays(data):
    """The int8 head of a blob: q [V][K], scale [V], bias [V] (unpadded) and cout_pad."""
    b = export.read_blob(data)
    hd = b["ops"][-1]
    return dict(q=hd["w"], scale=hd["scale"].astype(F32), bias=hd["bias"].astype(F32), V=b["n_classes"],
                P=hd["cout_pad"], K=hd["k"], vocab=b["vocab"])


def escape_score(q, sx, l1, b0, esc):
    """The escape's score per frame, exactly as run_head2: layer 1 through mm_i16 (with its bias), then
    a = a + wl1 * l1, a = a + wb0 * b0, ReLU, s = b2 + w2[0] * h[0] + ... in order (float32, rounded each step)."""
    a = mm_exact(q, sx, esc["q"], esc["scale"], esc["bias"])
    l1 = np.asarray(l1, F32)[:, None]
    b0 = np.asarray(b0, F32)[:, None]
    a = (a + (esc["wl1"][None, :] * l1).astype(F32)).astype(F32)
    a = (a + (esc["wb0"][None, :] * b0).astype(F32)).astype(F32)
    h = np.where(a > 0, a, F32(0)).astype(F32)
    s = np.full(len(a), esc["b2"], F32)
    for j in range(h.shape[1]):
        s = (s + (esc["w2"][j] * h[:, j]).astype(F32)).astype(F32)
    return s


def level1_inputs(logits, L1):
    """(l1, b0) of level 1 (classes [0, L1) of a permuted head) from exact logits [F][>= L1]."""
    lg = np.asarray(logits, F32)
    return lg[:, 1:L1].max(1).astype(F32), lg[:, 0].astype(F32)


def rare_logits(q, sx, rare):
    """r48's rare logits: P (with bias) through mm_i16, its outputs re-quantised by quant16, then the rare rows."""
    y = mm_exact(q, sx, rare["p_q"], rare["p_scale"], rare["p_bias"])
    q2, s2 = quant16_exact(y)
    return mm_exact(q2, s2, rare["r_q"], rare["r_scale"], rare["r_bias"])


def mm_w8f_exact(x, w, scale, bias):
    """vocr.c's W8F head rows (dequant8 + mm_f32): x [P][K] float32, w int8 [N][K] -> [P][N] float32, with
    w_f = float(q) * scale and y = bias, then y = y + w_f[k] * x[k] for k = 0 .. K - 1 (each step rounded)."""
    x = np.asarray(x, F32)
    wf = (np.asarray(w, F32) * np.asarray(scale, F32)[:, None]).astype(F32)
    y = np.broadcast_to(np.asarray(bias, F32) if bias is not None else np.zeros(len(wf), F32),
                        (len(x), len(wf))).copy()
    for k in range(x.shape[1]):
        y = (y + (wf[:, k][None, :] * x[:, k][:, None]).astype(F32)).astype(F32)
    return y


def twolevel_exact(x, head, two, topk=8, mode="w8i"):
    """runtime/vocr.c run_head2 on head inputs x [F][K] float32: per frame the score, fired, argmax (first maximum
    over the computed classes) and the top-K (values decreasing, ties: the lower class first). head: head_arrays of
    the permuted blob; two: read_trailer; mode "w8i" or "w8f" (the head rows' arithmetic; the escape reads the
    int16-quantised input in both). Returns dict(score, fire, best, topv, topc, blank)."""
    if mode not in ("w8i", "w8f"):
        raise ValueError("the two-level head runs in w8i or w8f")
    q, sx = quant16_exact(x)
    V = head["V"]
    L1 = 8 * two["l1_blocks"]

    def rows(sel, a, b):
        if mode == "w8f":
            return mm_w8f_exact(np.asarray(x, F32)[sel], head["q"][a:b], head["scale"][a:b], head["bias"][a:b])
        return mm_exact(q[sel], sx[sel], head["q"][a:b], head["scale"][a:b], head["bias"][a:b])
    lg1 = rows(slice(None), 0, L1)
    l1, b0 = level1_inputs(lg1, L1)
    s = escape_score(q, sx, l1, b0, two["esc"])
    fire = s > two["esc"]["t"]
    F = len(x)
    full = np.full((F, V), -np.inf, F32)
    full[:, :L1] = lg1
    if fire.any():
        if two["rare"] is None:
            r = rows(fire, L1, V)
        else:
            if mode != "w8i":
                raise ValueError("the rare block is a W8I trial only")
            r = rare_logits(q[fire], sx[fire], two["rare"])[:, :V - L1]
        full[np.nonzero(fire)[0], L1:] = r
    best = full.argmax(1)                                  # first maximum
    o = np.argsort(-full, axis=1, kind="stable")[:, :topk]
    return dict(score=s, fire=fire, best=best.astype(np.uint16), topv=np.take_along_axis(full, o, 1),
                topc=o.astype(np.uint16), blank=b0)


# ---------------------------------------------------------------- the escape (protocol "escape_training")

def fit_escape(x, y, seed=0, epochs=12, bs=4096, hidden=16, threads=5):
    """BCE fit of Linear(d, 16) - ReLU - Linear(16, 1) on standardised inputs (Adam 3e-3, cosine, pos_weight =
    min(100, neg / pos)). Returns dict(mu, sd, w1 [16][d], b1, w2 [16], b2, loss, pos_weight)."""
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    x = np.asarray(x, F32)
    y = np.asarray(y, bool)
    mu, sd = x.mean(0), x.std(0) + F32(1e-6)
    X = torch.from_numpy(((x - mu) / sd).astype(F32))
    Y = torch.from_numpy(y.astype(F32))
    npos = int(y.sum())
    pw = min(100.0, float(len(y) - npos) / max(1, npos))
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n = len(X)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(X[b]).squeeze(1), Y[b])
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        final = float(lossf(net(X).squeeze(1), Y))
    sdict = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    return dict(mu=mu.astype(np.float64), sd=sd.astype(np.float64), w1=sdict["0.weight"], b1=sdict["0.bias"],
                w2=sdict["2.weight"][0], b2=float(sdict["2.bias"][0]), loss=final, pos_weight=pw)


def export_escape(fit, K=96):
    """Fold the standardisation into layer 1 (w / sd, b - w . (mu / sd)); its first K columns (the head input)
    int8 per hidden unit (export.quantize_rows); the last two (l1, b0), biases and layer 2 float32. t = +inf."""
    w = fit["w1"] / fit["sd"][None, :]
    b = fit["b1"] - w @ fit["mu"]
    q, scale = export.quantize_rows(w[:, :K])
    return dict(q=q, scale=scale.astype(F32), bias=b.astype(F32), wl1=w[:, K].astype(F32), wb0=w[:, K + 1].astype(F32),
                w2=np.asarray(fit["w2"], F32), b2=F32(fit["b2"]), t=F32(np.inf))


def threshold_for(scores_pos, recall=0.999):
    """t = the largest float32 below the (floor((1 - recall) n) + 1)-th smallest positive score: at most
    floor((1 - recall) n) of the n positives score <= t (do not fire). recall 0.999 or 0.9999 (exact integers)."""
    s = np.sort(np.asarray(scores_pos, F32))
    miss_per_10k = int(round((1.0 - recall) * 10000))
    m = len(s) * miss_per_10k // 10000
    return F32(np.nextafter(s[m], F32(-np.inf))), m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rank")
    p.add_argument("--out", required=True)
    p = sub.add_parser("permute")
    p.add_argument("--blob", required=True)
    p.add_argument("--rank", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("info")
    p.add_argument("--blob", required=True)
    a = ap.parse_args()
    {"rank": cmd_rank, "permute": cmd_permute, "info": cmd_info}[a.cmd](a)


if __name__ == "__main__":
    main()

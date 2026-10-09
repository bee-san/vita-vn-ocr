"""Bit-exact numpy model of the vocr C runtime's arithmetic (runtime/vocr.c).

    logits = forward(read_blob(path), x, mode)     # x: H x W float32 network input

Every float32 operation is done in the same order as the C kernels, each product and
sum rounded on its own (vocr.c is built with -ffp-contract=off):
  stem / depthwise: acc = bias; for dy (rows inside the image only), dx: acc = acc + w * x
  pointwise / head (f32, w8f): acc = bias; for k = 0..K-1: acc = acc + w[k] * x[k],
      where w8f uses w = float(q) * scale
  pointwise / head (w8i): per position q16 = trunc(x * (32767 / max|x|) +- 0.5);
      S = 0 + float(c_0) + float(c_1) + ... with c_i the exact sum of q8 * q16 over
      k in [512 i, 512 i + 512); y = S * (scale * (max|x| / 32767)) + bias
  then ReLU (x > 0 ? x : 0), then the residual add.
Columns outside the line are zeros, which the C code also adds (w * 0); rows outside
the image are skipped. So on IEEE hardware without flush-to-zero the C runtime must
return exactly these bits for any tile size; parity.py checks that. (ARMv7 NEON flushes
denormals to zero, so the NEON build can differ only if a denormal ever occurs.)
"""
import numpy as np

T_CONV, T_DW, T_PW, T_HEAD = 1, 2, 3, 4
F32 = np.float32
CHUNK = 512          # w8i: int32 sums over at most this many terms (vocr.c W8I_CHUNK)


def _relu(y):
    return np.where(y > 0, y, F32(0)).astype(F32)


def conv_exact(x, op):
    """Stem (x: Hi x Wi) or depthwise (x: Hi x Wi x C) convolution, C summation order."""
    stem = op["type"] == T_CONV
    hi, wi = x.shape[:2]
    kh, kw, sh, sw, ph, pw = op["kh"], op["kw"], op["sh"], op["sw"], op["ph"], op["pw"]
    ho, wo = op["h_out"], (wi + 2 * pw - kw) // sw + 1
    c = op["cout"]
    xp = np.zeros((hi, wi + 2 * pw + sw) + (() if stem else (c,)), F32)
    xp[:, pw:pw + wi] = x
    w = op["w"].astype(F32)                                   # [cout][taps]
    b = op["bias"] if op["bias"] is not None else np.zeros(c, F32)
    out = np.empty((ho, wo, c), F32)
    for y in range(ho):
        acc = np.broadcast_to(b.astype(F32), (wo, c)).copy()
        r0 = y * sh - ph
        for dy in range(kh):
            r = r0 + dy
            if r < 0 or r >= hi:
                continue
            for dx in range(kw):
                cols = xp[r, dx:dx + sw * (wo - 1) + 1:sw]      # [wo] or [wo][c]
                wt = w[:, dy * kw + dx]
                acc = acc + (wt[None, :] * cols[:, None] if stem else wt[None, :] * cols)
        out[y] = _relu(acc) if op["relu"] else acc
    return out


def quant16_exact(x):
    """Per-row int16 quantisation of vocr.c quant16(): (q float64, scale float32)."""
    mx = np.abs(x).max(axis=1).astype(F32)
    ok = mx > 0
    safe = np.where(ok, mx, F32(1)).astype(F32)
    inv = (F32(32767) / safe).astype(F32)
    t = (x * inv[:, None]).astype(F32)
    h = np.where(np.signbit(t), F32(-0.5), F32(0.5)).astype(F32)
    q = np.clip(np.trunc((t + h).astype(F32)), -32767, 32767)
    q[~ok] = 0
    sx = np.where(ok, (mx / F32(32767)).astype(F32), F32(0)).astype(F32)
    return q.astype(np.float64), sx


def matmul_exact(x, op, mode):
    """x: P x K float32 -> P x cout float32 (bias, ReLU; no residual)."""
    n = op["cout"]
    b = op["bias"].astype(F32) if op["bias"] is not None else None
    if mode == "w8i":
        q, sx = quant16_exact(x)
        w = op["w"].astype(np.float64)
        acc = np.zeros((x.shape[0], n), F32)
        for k0 in range(0, x.shape[1], CHUNK):                 # exact integer sums per chunk
            acc = (acc + (q[:, k0:k0 + CHUNK] @ w[:, k0:k0 + CHUNK].T).astype(F32)).astype(F32)
        y = (acc * (op["scale"].astype(F32)[None, :] * sx[:, None]).astype(F32)).astype(F32)
        if b is not None:
            y = (y + b[None, :]).astype(F32)
    else:
        w = op["w"].astype(F32) if not op.get("int8") else \
            (op["w"].astype(F32) * op["scale"].astype(F32)[:, None]).astype(F32)
        y = np.broadcast_to(b if b is not None else np.zeros(n, F32), (x.shape[0], n)).copy()
        for k in range(x.shape[1]):
            y = y + w[:, k][None, :] * x[:, k][:, None]
    return _relu(y) if op["relu"] else y


def forward(blob, x, mode):
    """x: H x W float32 input (W a multiple of the stride). Returns T x classes logits."""
    t = np.asarray(x, F32)
    dw_in = None
    for op in blob["ops"]:
        if op["type"] in (T_CONV, T_DW):
            dw_in = t
            t = conv_exact(t, op)
            continue
        h, w, c = t.shape
        if op["collapse"]:
            y = matmul_exact(t.transpose(1, 0, 2).reshape(w, h * c), op, mode).reshape(1, w, -1)
        else:
            y = matmul_exact(t.reshape(h * w, c), op, mode).reshape(h, w, -1)
        if op["res"]:
            y = (y + dw_in).astype(F32)
        t = y
    return t[0]

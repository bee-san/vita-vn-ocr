"""textspot7's per-region fallback (F) and decode post-pass (D) for a text spotter's boxes (issue #128,
docs/textspot7.md, eval/textspot7_protocol.json with amendment 1): the specification and the bit-exact twin of
runtime/vocr_textspot7.c. Integer arithmetic only; every box is (x, y, w, h) in the region's pixels.

F: choose(S, G, p) -> "S" | "G"
  S: the spotter's horizontal boxes of the region [(x, y, w, h, conf)] (conf 0..127, the decode's), in its order;
  G: rubi,fix=1's boxes of the same region [(x, y, w, h)]. p = Params(mode, tau, lam, kappa): lam and kappa in
  per-mille (0 = off).
  - cmed: the lower middle of S's sorted confidences (index (n - 1) // 2); hmed: the same of S's heights.
  - mode "S" (the spotter by default): G when S is empty and G is not; G when cmed < tau; G when some G box
    S_lacks (lam on) or S_short (kappa on); else S. An empty S with an empty G is S.
    S_lacks(b): 1000 h_b >= lam hmed and 2 cov < h_b, where cov is the length of the union of [y_s, y_s + h_s)
      intersected with [y_b, y_b + h_b) over the S boxes s that overlap b horizontally (by a pixel or more).
    S_short(b): R = the S boxes s with 2 vov(b, s) >= min(h_b, h_s) and a horizontal overlap with b; R not empty;
      with [u0, u1) the span of R in x, 1000 max(u0 - x_b, x_b + w_b - u1) > kappa h_b.
  - mode "G" (GRAD by default): S when G is empty and S is not; S when G_merge and cmed >= tau; else G.
    G_merge: some G box holds the centres of 2 or more S boxes of different S rows. A centre is (2x + w, 2y + h)
    in half pixels, inside b when 2 x_b <= cx < 2 (x_b + w_b) and the same in y; rows: the connected components of
    the pairs with 2 vov >= the smaller height.

D: post(frame_boxes, region, grey, steps) -> [(x, y, w, h, conf, dark)] in region pixels
  frame_boxes: the spotter's horizontal boxes in the frame's pixels [(x, y, w, h, conf, dark)], decode order;
  region (X0, Y0, X1, Y1) in the frame's pixels; grey: the region's pixels (uint8, H x W = Y1 - Y0, X1 - X0);
  steps: a set of "assign", "join5" / "join10" (gamma 0.5 / 1.0), "extend", "tighten", run in this order.
  - assign (else textspot6's centre rule): a box also belongs to the region when its part inside is at least 0.6
    of its height tall (10 ih >= 6 h) and at least its height wide (iw >= h); every box clipped to the region.
  - join: two boxes in one row (10 vov >= 6 min(h)) with gap = max(x_a, x_b) - min(x_a + w_a, x_b + w_b) and
    10 gap <= g10 max(h_a, h_b) (g10 = 5 or 10) become their union at the earlier box's place (conf the smaller,
    dark the wider box's, the earlier on a tie); the first such pair in order (a, then b > a) each time, until
    none is left.
  - extend: each box grows left and right over inked columns of its own rows, from the original set: a column is
    inked when at least 2 of the box's rows there have a 3 x 3 contrast (max - min, edges replicated) >= t,
    t = max(40, (2 p99) // 5), p99 the contrast inside the box at sorted index (99 (n - 1)) // 100. The scan goes
    outward from the box's edge, at most (3 h) // 2 columns, ends when the run of uninked columns passes h // 2,
    and stops at the region's edge and at the nearest other box of the row (10 vov >= 6 min(h)) on that side; the
    new edge is the last inked column plus max(2, (15 h + 50) // 100) px, never beyond the stop; a box only grows.
  - tighten: synth/lines_refine.refine_lines(grey, boxes, dark, TIGHTEN) (runtime/vocr_lines_refine.c with
    VOCR_REFINE_TIGHTEN): each box's rows cut to its text band plus 16%, its columns kept.
"""
from dataclasses import dataclass

import numpy as np

MODES = ("S", "G")
STEPS = ("assign", "join5", "join10", "extend", "tighten")
CONTRAST_MIN = 40


@dataclass(frozen=True)
class Params:
    mode: str = "S"
    tau: int = 0
    lam: int = 0          # per-mille, 0 = off
    kappa: int = 0        # per-mille, 0 = off

    def name(self):
        return f"{self.mode}_t{self.tau}_l{self.lam}_k{self.kappa}"


# ----------------------------------------------------------------------------- geometry

def vov(a, b):
    return max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))


def hov(a, b):
    return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))


def lower_median(values):
    v = sorted(values)
    return v[(len(v) - 1) // 2]


def rows_of(boxes):
    """Row id per box: the smallest index of its connected component under 2 vov >= min(h)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if 2 * vov(boxes[i], boxes[j]) >= min(boxes[i][3], boxes[j][3]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n)]


def centre_in(s, b):
    cx, cy = 2 * s[0] + s[2], 2 * s[1] + s[3]
    return 2 * b[0] <= cx < 2 * (b[0] + b[2]) and 2 * b[1] <= cy < 2 * (b[1] + b[3])


# ----------------------------------------------------------------------------- F

def s_lacks(b, S, lam, hmed):
    if not lam or 1000 * b[3] < lam * hmed:
        return False
    ivs = sorted((max(s[1], b[1]), min(s[1] + s[3], b[1] + b[3])) for s in S if hov(s, b) > 0)
    cov, end = 0, b[1]
    for a, e in ivs:
        if e <= a:
            continue
        a = max(a, end)
        if e > a:
            cov += e - a
            end = e
    return 2 * cov < b[3]


def s_short(b, S, kappa):
    if not kappa:
        return False
    R = [s for s in S if 2 * vov(b, s) >= min(b[3], s[3]) and hov(b, s) > 0]
    if not R:
        return False
    u0 = min(s[0] for s in R)
    u1 = max(s[0] + s[2] for s in R)
    return 1000 * max(u0 - b[0], b[0] + b[2] - u1) > kappa * b[3]


def g_merge(S, G):
    rows = rows_of(S)
    for b in G:
        held = {rows[i] for i, s in enumerate(S) if centre_in(s, b)}
        if len(held) >= 2:
            return True
    return False


def features(S, G, p):
    """The quantities choose() reads (for records): cmed, hmed, lacks, short, merge."""
    cmed = lower_median([s[4] for s in S]) if S else None
    hmed = lower_median([s[3] for s in S]) if S else None
    return {"cmed": cmed, "hmed": hmed,
            "lacks": bool(S) and any(s_lacks(b, S, p.lam, hmed) for b in G),
            "short": bool(S) and any(s_short(b, S, p.kappa) for b in G),
            "merge": bool(S) and g_merge(S, G)}


def choose(S, G, p):
    S = [tuple(int(v) for v in s[:5]) for s in S]
    G = [tuple(int(v) for v in g[:4]) for g in G]
    if p.mode == "S":
        if not S:
            return "G" if G else "S"
        cmed = lower_median([s[4] for s in S])
        if cmed < p.tau:
            return "G"
        hmed = lower_median([s[3] for s in S])
        if any(s_lacks(b, S, p.lam, hmed) or s_short(b, S, p.kappa) for b in G):
            return "G"
        return "S"
    if p.mode == "G":
        if not G:
            return "S" if S else "G"
        if not S:
            return "G"
        cmed = lower_median([s[4] for s in S])
        return "S" if (cmed >= p.tau and g_merge(S, G)) else "G"
    raise ValueError(f"unknown mode {p.mode!r}")


# ----------------------------------------------------------------------------- D

def assign(frame_boxes, region, overlap):
    """The region's boxes from the frame's (decode order), clipped, in region pixels: [(x, y, w, h, conf, dark)]."""
    X0, Y0, X1, Y1 = region
    W, H = X1 - X0, Y1 - Y0
    out = []
    for b in frame_boxes:
        x, y, w, h, conf, dark = (int(v) for v in b[:6])
        cx, cy = 2 * x + w, 2 * y + h
        keep = 2 * X0 <= cx < 2 * X1 and 2 * Y0 <= cy < 2 * Y1
        if not keep and overlap:
            ih = min(y + h, Y1) - max(y, Y0)
            iw = min(x + w, X1) - max(x, X0)
            keep = ih > 0 and iw > 0 and 10 * ih >= 6 * h and iw >= h
        if not keep:
            continue
        a0, b0 = max(0, x - X0), max(0, y - Y0)
        a1, b1 = min(W, x + w - X0), min(H, y + h - Y0)
        if a1 > a0 and b1 > b0:
            out.append((a0, b0, a1 - a0, b1 - b0, conf, dark))
    return out


def same_row6(a, b):
    return 10 * vov(a, b) >= 6 * min(a[3], b[3])


def join(boxes, g10):
    bx = [tuple(b) for b in boxes]
    while True:
        hit = None
        for i in range(len(bx)):
            for j in range(i + 1, len(bx)):
                a, b = bx[i], bx[j]
                if not same_row6(a, b):
                    continue
                gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
                if 10 * gap <= g10 * max(a[3], b[3]):
                    hit = (i, j)
                    break
            if hit:
                break
        if hit is None:
            return bx
        i, j = hit
        a, b = bx[i], bx[j]
        x0, y0 = min(a[0], b[0]), min(a[1], b[1])
        x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
        dark = a[5] if a[2] >= b[2] else b[5]
        bx[i] = (x0, y0, x1 - x0, y1 - y0, min(a[4], b[4]), dark)
        del bx[j]


def contrast3(g):
    p = np.pad(np.asarray(g, np.uint8), 1, mode="edge").astype(np.int16)
    mx = np.maximum(np.maximum(p[:, :-2], p[:, 1:-1]), p[:, 2:])
    mn = np.minimum(np.minimum(p[:, :-2], p[:, 1:-1]), p[:, 2:])
    mx = np.maximum(np.maximum(mx[:-2], mx[1:-1]), mx[2:])
    mn = np.minimum(np.minimum(mn[:-2], mn[1:-1]), mn[2:])
    return (mx - mn).astype(np.int32)


def ink_threshold(c, b):
    x, y, w, h = b[:4]
    v = np.sort(c[y:y + h, x:x + w].ravel())
    if v.size == 0:
        return 255
    p99 = int(v[(99 * (v.size - 1)) // 100])
    return max(CONTRAST_MIN, (2 * p99) // 5)


def extend(boxes, grey):
    H, W = grey.shape
    c = contrast3(grey)
    out = []
    for i, b in enumerate(boxes):
        x, y, w, h = b[:4]
        t = ink_threshold(c, b)
        band = c[y:y + h] >= t                      # rows of the box, every column
        inked = band.sum(axis=0) >= 2
        reach, gapmax = (3 * h) // 2, h // 2
        pad = max(2, (15 * h + 50) // 100)
        others = [o for k, o in enumerate(boxes) if k != i and same_row6(b, o)]
        # right
        stop = min([W] + [o[0] for o in others if o[0] >= x + w])
        last, gap = None, 0
        for X in range(x + w, min(stop, x + w + reach)):
            if inked[X]:
                last, gap = X, 0
            else:
                gap += 1
                if gap > gapmax:
                    break
        x1 = x + w
        if last is not None:
            x1 = max(x1, min(stop, last + 1 + pad))
        # left
        stopl = max([0] + [o[0] + o[2] for o in others if o[0] + o[2] <= x])
        last, gap = None, 0
        for X in range(x - 1, max(stopl, x - reach) - 1, -1):
            if inked[X]:
                last, gap = X, 0
            else:
                gap += 1
                if gap > gapmax:
                    break
        x0 = x
        if last is not None:
            x0 = min(x0, max(stopl, last - pad))
        out.append((x0, y, x1 - x0, h, *b[4:6]))
    return out


def tighten(boxes, grey):
    import lines_refine as LR
    if not boxes:
        return []
    rb, rd, org = LR.refine_lines(np.ascontiguousarray(grey, np.uint8), [b[:4] for b in boxes], [b[5] for b in boxes],
                                  LR.TIGHTEN)
    return [(*(int(v) for v in r), boxes[o][4], int(d)) for r, d, o in zip(rb, rd, org)]


def post(frame_boxes, region, grey, steps):
    steps = set(steps)
    bad = steps - set(STEPS)
    if bad:
        raise ValueError(f"unknown steps {sorted(bad)}")
    bx = assign(frame_boxes, region, "assign" in steps)
    if "join10" in steps:
        bx = join(bx, 10)
    elif "join5" in steps:
        bx = join(bx, 5)
    if "extend" in steps:
        bx = extend(bx, grey)
    if "tighten" in steps:
        bx = tighten(bx, grey)
    return bx

#!/usr/bin/env python3
"""finder_aug: "what the console's line finder hands over", train2's fa= augmentation (VOCR5 arm F).

  g2 = finder_aug(g, rng, donor=None, min_width=8)

The console reads each line from the box its GRAD finder returns (runtime/vocr_lines.c). That box is taller and
wider than the ink crop the real pools hold, and its extra rows and columns carry nearly all of the errors it adds
(docs/finder-tighten.md §1): the dialogue box's border or inner rule, half-cut glyphs of the previous or next line,
furigana, scenery behind a translucent box, a "next page" cursor after the text and a portrait or ornament at either
end. finder_aug gives a stored crop that context (eval/vocr5_protocol.json "finder_augmentation";
prompts/vocr5-aug.md part A). g is a crop at height L (train2: 32) as float32 grey, rng a numpy Generator and donor
another crop of the batch (None: no donor features). Every random choice is drawn from rng in a fixed order, so the
same generator state gives the same crop. All sizes are relative to L:

 1. Levels: bg = the median of the crop's top and bottom rows; ink = its 2nd or 98th percentile, whichever is
    farther from bg.
 2. Vertical context, always: E = round(L U(0.10, 0.60)) extra rows, t = round(E U(0, 1)) on top and E - t below,
    filled with bg + N(0, 2). Then, each side on its own:
    - p 0.35 a frame line: 1-3 px thick at a random row of the side, from x0 in [0, 0.2 w] to x1 in [0.8 w, w],
      grey = ink (p 0.5) or bg +- U(60, 160) clipped to [0, 255];
    - p 0.35 a neighbour line's half-cut glyphs (side of 2 rows or more): the donor at height L, cut (at a random x)
      or tiled to width w; its bottom rows above the crop (the previous line) or its top rows below it (the next
      line), as many as the side has;
    - p 0.15, top only, a furigana fragment: a piece of the donor 1-4 L wide, scaled by U(0.45, 0.55) in both axes,
      at a random x and a random row within the top rows (cut at the top when it is taller than they are);
    - p 0.20 scenery: a linear vertical gradient from bg at the crop to bg +- U(20, 80) at the side's far edge, plus
      N(0, U(3, 10)) noise.
    Donor pixels go onto bg as the donor's deviation from its own background (step 1's levels of the donor), scaled
    so that the donor's ink level lands on the crop's: a neighbour line has the line's polarity and contrast. A
    donor whose ink is within MIN_CONTRAST of its background holds no glyphs and adds nothing; the scale is capped
    at MAX_GAIN. The side is drawn back to front: scenery, neighbour, furigana, then the (opaque) frame line.
 3. Line-end clutter, p 0.5:
    - p 0.6 a trailing next-page cursor: U(0.5, 2.0) L of bg after the last column, then a shape U(0.3, 0.7) L in
      size (a filled square or circle, a triangle pointing down or right, or a flower of 4-5 overlapping circles),
      grey = ink (p 0.7) or an accent grey bg +- U(60, 160) (p 0.3), centred on the text band (the crop's middle
      row) +- 0.2 L; its edges anti-aliased;
    - else a portrait or ornament edge at the start (p 0.5) or the end: a block U(0.5, 3.0) L wide over the full new
      height, after a gap of U(0, 1) L of bg: a smooth random field plus 1-4 random ellipses, or (p 0.5, with a
      donor) the donor at the new height, flipped (horizontally, vertically or both) and contrast-shifted.
    New background columns are bg + N(0, 2), like the extra rows.
 4. Resize to `height` keeping the aspect ratio (width round(w' height / (L + E)), capped at max_w; INTER_AREA, a
    shrink). A width under min_width (train2: the frames the line's CTC label needs) returns g unchanged.

The label never changes. make_batch draws whether to call it (probability fa) and the donor (another item of the
batch) only for a source whose fa > 0, so fa=0 leaves train2's random stream and batches as they were.

Contact sheet (synthetic shards only, never a real or protected crop):

  python train/finder_aug.py sheet --shard SYNTH.npz --out sheet.png [--n 40] [--seed 1] [--max-px 1280]
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

H, MAX_W = 32, 1024                  # train2's line height and width cap
P_FRAME, P_NEIGHBOUR, P_FURIGANA, P_SCENERY = 0.35, 0.35, 0.15, 0.20
P_CLUTTER, P_CURSOR, P_CURSOR_INK, P_START, P_DONOR_BLOCK = 0.5, 0.6, 0.7, 0.5, 0.5
MIN_CONTRAST = 24.0                  # donor ink this close to its background: no glyphs to borrow
MAX_GAIN = 4.0                       # cap on the donor deviation's scale onto the crop's contrast
SS = 4                               # supersampling of the drawn shapes (anti-aliased edges)
SHAPES = ("square", "circle", "triangle-down", "triangle-right", "flower")


def levels(g):
    """(bg, ink) of a grey crop: the median of its top and bottom rows, and its 2nd or 98th percentile, whichever is
    farther from bg."""
    g = np.asarray(g, np.float32)
    bg = float(np.median(np.concatenate([g[0], g[-1]])))
    lo, hi = (float(v) for v in np.percentile(g, (2, 98)))
    return bg, (lo if abs(lo - bg) >= abs(hi - bg) else hi)


def _fill(rng, bg, h, w):
    return (bg + rng.normal(0.0, 2.0, (h, w))).astype(np.float32)


def _other_grey(rng, bg):
    """bg +- U(60, 160), clipped to [0, 255]: a frame line or cursor colour other than the ink's."""
    d = rng.uniform(60.0, 160.0)
    return float(np.clip(bg + (d if rng.random() < 0.5 else -d), 0.0, 255.0))


def _cut_or_tile(a, w, rng):
    """a cut at a random x to width w, or tiled to it when narrower."""
    aw = a.shape[1]
    if aw >= w:
        x = int(rng.integers(0, aw - w + 1))
        return a[:, x:x + w]
    return np.tile(a, (1, -(-w // aw)))[:, :w]


class _Donor:
    """A donor crop at height L, and its deviation from its own background on the crop's contrast."""

    def __init__(self, d, L, bg, ink):
        d = np.asarray(d, np.float32)
        if d.shape[0] != L:
            d = cv2.resize(d, (max(1, int(round(d.shape[1] * L / d.shape[0]))), L), interpolation=cv2.INTER_LINEAR)
        self.img = d
        dbg, dink = levels(d)
        self.glyphs = abs(dink - dbg) >= MIN_CONTRAST
        gain = (ink - bg) / (dink - dbg) if self.glyphs else 0.0
        self.dev = ((d - dbg) * float(np.clip(gain, -MAX_GAIN, MAX_GAIN))).astype(np.float32)


def _paste_add(dst, src, y, x):
    """dst[y:, x:] += src, clipped to dst."""
    h, w = dst.shape
    sy0, sx0 = max(0, -y), max(0, -x)
    y0, x0 = max(0, y), max(0, x)
    y1, x1 = min(h, y + src.shape[0]), min(w, x + src.shape[1])
    if y1 > y0 and x1 > x0:
        dst[y0:y1, x0:x1] += src[sy0:sy0 + y1 - y0, sx0:sx0 + x1 - x0]


def _side(rng, rows, top, L, bg, ink, donor, log):
    """Step 2's features on one side's extra rows (n x w, modified in place): the draws in the spec's order (frame,
    neighbour, furigana, scenery), the drawing back to front."""
    n, w = rows.shape
    if n == 0:
        return
    where = "top" if top else "bottom"
    frame = neighbour = furigana = scenery = None
    if rng.random() < P_FRAME:
        th = min(n, int(rng.integers(1, 4)))
        r0 = int(rng.integers(0, n - th + 1))
        x0 = int(round(rng.uniform(0.0, 0.2) * w))
        x1 = int(round(rng.uniform(0.8, 1.0) * w))
        grey = ink if rng.random() < 0.5 else _other_grey(rng, bg)
        frame = (r0, th, x0, x1, grey)
    if donor is not None and n >= 2 and rng.random() < P_NEIGHBOUR:
        strip = _cut_or_tile(donor.dev, w, rng)
        k = min(n, L)
        neighbour = (strip[L - k:], n - k) if top else (strip[:k], 0)
    if top and donor is not None and rng.random() < P_FURIGANA:
        s = rng.uniform(0.45, 0.55)
        dw = donor.dev.shape[1]
        pw = min(dw, max(1, int(round(L * rng.uniform(1.0, 4.0)))))
        px = int(rng.integers(0, dw - pw + 1))
        fw, fh = max(1, int(round(pw * s))), max(1, int(round(L * s)))
        small = cv2.resize(np.ascontiguousarray(donor.dev[:, px:px + pw]), (fw, fh), interpolation=cv2.INTER_AREA)
        x = int(rng.integers(0, max(0, w - fw) + 1))
        y = int(rng.integers(0, n - fh + 1)) if fh <= n else n - fh
        furigana = (small.reshape(fh, fw), y, x)
    if rng.random() < P_SCENERY:
        amp = rng.uniform(20.0, 80.0) * (1.0 if rng.random() < 0.5 else -1.0)
        sd = rng.uniform(3.0, 10.0)
        dist = np.arange(n, 0, -1) if top else np.arange(1, n + 1)        # rows from the crop: 1 .. n
        scenery = (amp * dist / n)[:, None] + rng.normal(0.0, sd, (n, w))
    if scenery is not None:
        rows += scenery.astype(np.float32)
        log.append(f"scenery-{where}")
    if neighbour is not None and donor.glyphs:
        _paste_add(rows, neighbour[0], neighbour[1], 0)
        log.append(f"neighbour-{where}")
    if furigana is not None and donor.glyphs:
        _paste_add(rows, *furigana)
        log.append("furigana")
    if frame is not None:
        r0, th, x0, x1, grey = frame
        rows[r0:r0 + th, x0:x1] = grey
        log.append(f"frame-{where}")


def _shape_mask(rng, kind, size):
    """size x size coverage in [0, 1] of a cursor shape."""
    S = size * SS
    m = np.zeros((S, S), np.uint8)
    if kind == "square":
        m[:] = 255
    elif kind == "circle":
        cv2.circle(m, (S // 2, S // 2), S // 2, 255, -1, cv2.LINE_AA)
    elif kind == "triangle-down":
        cv2.fillPoly(m, [np.array([[0, 0], [S - 1, 0], [S // 2, S - 1]], np.int32)], 255, cv2.LINE_AA)
    elif kind == "triangle-right":
        cv2.fillPoly(m, [np.array([[0, 0], [S - 1, S // 2], [0, S - 1]], np.int32)], 255, cv2.LINE_AA)
    else:                                    # a flower / star of overlapping circles
        k = int(rng.integers(4, 6))
        phase = rng.uniform(0.0, 2 * math.pi / k)
        c, r, R = S / 2, 0.22 * S, 0.27 * S
        for i in range(k):
            a = phase + 2 * math.pi * i / k
            cv2.circle(m, (int(round(c + R * math.cos(a))), int(round(c + R * math.sin(a)))), int(round(r)), 255, -1,
                       cv2.LINE_AA)
        cv2.circle(m, (int(round(c)), int(round(c))), int(round(0.8 * r)), 255, -1, cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255.0, (size, size), interpolation=cv2.INTER_AREA)


def _cursor(rng, img, t, L, bg, ink, log):
    Hn = img.shape[0]
    gap = int(round(L * rng.uniform(0.5, 2.0)))
    size = max(3, int(round(L * rng.uniform(0.3, 0.7))))
    kind = SHAPES[int(rng.integers(len(SHAPES)))]
    a = _shape_mask(rng, kind, size)
    grey = ink if rng.random() < P_CURSOR_INK else _other_grey(rng, bg)
    cy = t + L / 2 + L * rng.uniform(-0.2, 0.2)
    block = _fill(rng, bg, Hn, gap + size)
    y0 = int(round(cy - size / 2))
    ya, yb = max(0, y0), min(Hn, y0 + size)
    if yb > ya:
        aa = a[ya - y0:yb - y0]
        cols = block[ya:yb, gap:]
        block[ya:yb, gap:] = cols * (1.0 - aa) + grey * aa
    log.append(f"cursor-{kind}")
    return np.hstack([img, block])


def _field(rng, bg, h, w):
    """A smooth random field plus a few random ellipses (h x w)."""
    base = bg + rng.uniform(-80.0, 80.0)
    amp = rng.uniform(15.0, 60.0)
    lo = rng.normal(0.0, 1.0, (max(2, h // 8), max(2, w // 8))).astype(np.float32)
    f = cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC) * amp + base
    for _ in range(int(rng.integers(1, 5))):
        m = np.zeros((h * SS, w * SS), np.uint8)
        centre = (int(rng.uniform(0, w) * SS), int(rng.uniform(0, h) * SS))
        axes = (max(1, int(rng.uniform(0.1, 0.6) * w * SS)), max(1, int(rng.uniform(0.1, 0.6) * h * SS)))
        cv2.ellipse(m, centre, axes, rng.uniform(0.0, 180.0), 0, 360, 255, -1, cv2.LINE_AA)
        a = cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
        f = f * (1.0 - a) + rng.uniform(0.0, 255.0) * a
    return f.astype(np.float32)


def _donor_block(rng, donor, bg, h, w):
    """The donor at height h, flipped and contrast-shifted, cut or tiled to width w."""
    d = donor.img
    d = cv2.resize(d, (max(1, int(round(d.shape[1] * h / d.shape[0]))), h), interpolation=cv2.INTER_LINEAR)
    d = cv2.flip(d, int(rng.integers(-1, 2)))
    d = np.ascontiguousarray(_cut_or_tile(d, w, rng))
    gain = rng.uniform(0.3, 1.0) * (1.0 if rng.random() < 0.5 else -1.0)
    return ((d - d.mean()) * gain + bg + rng.uniform(-80.0, 80.0)).astype(np.float32)


def _portrait(rng, img, L, bg, donor, log):
    Hn = img.shape[0]
    start = rng.random() < P_START
    bw = max(1, int(round(L * rng.uniform(0.5, 3.0))))
    gap = int(round(L * rng.uniform(0.0, 1.0)))
    if donor is not None and rng.random() < P_DONOR_BLOCK:
        block, what = _donor_block(rng, donor, bg, Hn, bw), "donor"
    else:
        block, what = _field(rng, bg, Hn, bw), "field"
    gapcols = _fill(rng, bg, Hn, gap)
    log.append(f"portrait-{'start' if start else 'end'}-{what}")
    return np.hstack([block, gapcols, img] if start else [img, gapcols, block])


def finder_aug(g, rng, donor=None, min_width=8, height=H, max_w=MAX_W, log=None):
    """g (L x w grey, float32) with the finder's context, resized to height; g itself when the result would be
    narrower than min_width. donor: another crop (any height; None = no donor features). log: a list that gets the
    names of the features drawn (for sheets and tests; it changes nothing)."""
    log = [] if log is None else log
    g = np.asarray(g, np.float32)
    L, w = g.shape
    bg, ink = levels(g)
    d = _Donor(donor, L, bg, ink) if donor is not None else None
    E = int(round(L * rng.uniform(0.10, 0.60)))
    t = int(round(E * rng.uniform(0.0, 1.0)))
    log.append(f"rows-{t}-{E - t}")
    top, bottom = _fill(rng, bg, t, w), _fill(rng, bg, E - t, w)
    _side(rng, top, True, L, bg, ink, d, log)
    _side(rng, bottom, False, L, bg, ink, d, log)
    img = np.vstack([top, g, bottom])
    if rng.random() < P_CLUTTER:
        img = _cursor(rng, img, t, L, bg, ink, log) if rng.random() < P_CURSOR else _portrait(rng, img, L, bg, d, log)
    Hn, wn = img.shape
    nw = min(max_w, int(round(wn * height / Hn)))
    if nw < min_width:
        log.append("unchanged-min-width")
        return g
    out = cv2.resize(np.clip(img, 0.0, 255.0).astype(np.float32), (nw, height), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0.0, 255.0)


# ----------------------------------------------------------------------------- contact sheet

def _crops(path, height=H):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    out = []
    for i in range(len(shapes)):
        h, w = (int(v) for v in shapes[i])
        g = data[offs[i]:offs[i + 1]].reshape(h, w)
        nw = max(8, min(MAX_W, int(round(w * height / h))))    # train.resize_h
        out.append((cv2.resize(g, (nw, height), interpolation=cv2.INTER_LINEAR), labels[i]))
    return out


def sheet(shard, out, n=40, seed=1, max_px=1280, max_w=300):
    """A contact sheet of n crops of a synthetic shard (only crops up to max_w px wide at height 32, so they show at
    1:1): two columns, each row the crop as stored and finder_aug's result (always applied; the donor another crop of
    the sheet). Returns the features drawn per crop."""
    rng = np.random.default_rng(seed)
    crops = [c for c in _crops(shard) if c[0].shape[1] <= max_w and c[1]]
    pick = rng.choice(len(crops), size=n, replace=False)
    rows, logs = [], []
    for j, i in enumerate(pick):
        donor = crops[pick[(j + 1 + int(rng.integers(n - 1))) % n]][0]
        log = []
        a = finder_aug(crops[i][0].astype(np.float32), rng, donor, log=log)
        rows.append((crops[i][0], np.clip(a, 0, 255).astype(np.uint8)))
        logs.append({"row": j, "shard_index": int(i), "features": log, "w": int(crops[i][0].shape[1]),
                     "w_aug": int(a.shape[1])})
    cols, per = 2, -(-n // 2)
    cw, rh, pad = max_px // cols, H + 6, 4
    canvas = np.full((per * rh + 2 * pad, max_px), 96, np.uint8)
    for j, (o, a) in enumerate(rows):
        c, r = divmod(j, per)
        x, y = c * cw + 28, pad + r * rh
        cv2.putText(canvas, str(j), (c * cw + 2, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
        room = (cw - 28 - 3 * pad) // 2
        for k, im in enumerate((o, a)):
            if im.shape[1] > room:
                im = cv2.resize(im, (room, max(1, int(round(im.shape[0] * room / im.shape[1])))),
                                interpolation=cv2.INTER_AREA)
            xx = x + k * (room + pad)
            canvas[y:y + im.shape[0], xx:xx + im.shape[1]] = im
    assert max(canvas.shape) <= max_px
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return logs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet", help="contact sheet of augmented crops of a SYNTHETIC shard")
    s.add_argument("--shard", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--max-px", type=int, default=1280)
    args = ap.parse_args(argv)
    logs = sheet(args.shard, args.out, args.n, args.seed, args.max_px)
    Path(args.out).with_suffix(".json").write_text(json.dumps({"shard": str(args.shard), "seed": args.seed,
                                                               "rows": logs}, indent=1) + "\n")
    print(f"{args.out}: {len(logs)} crops")


if __name__ == "__main__":
    main()

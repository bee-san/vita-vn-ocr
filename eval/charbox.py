"""Per-character positions from the recognizer's CTC frames: the Python twin of runtime/vocr_charbox.c.

docs/char-boxes.md. The recognizer (runtime/vocr.c) resizes a line crop of w x h pixels to the model's height
H (32) and nw = vocr_resized_width(w, h) columns (OpenCV INTER_LINEAR), right-pads it to a multiple of the
stride (8) and emits one CTC frame per stride columns. Greedy decoding keeps a frame's class when it is not the
blank and differs from the frame before, so every character of the greedy text has a run of frames [f0, f1]
(the frames of its class before the next blank or class), and frame f covers resized columns [8 f, 8 f + 8).
In continuous coordinates (pixel i spans [i, i + 1)), INTER_LINEAR maps a resized coordinate u to the crop
coordinate u * w / nw.

positions() turns a line's per-frame argmax (vocr_extra.best, or topk_class[f * K]) into one entry per greedy
character, along the reading direction, in the crop's continuous pixel coordinates, fixed point (1/16 px):

  c        its centre. In resized px: the centre of the first frame of the character's run, 8 f0 + 4 (or of
           the whole run with CENTRE_RUN), plus the model's calibration (CAL): a shift for every character and
           an offset for the character's group (GROUPS: punctuation that sits at one side of its cell,
           brackets, small kana, ...), mapped to the crop (x w / nw) and clipped to it. With INK, the middle
           of the ink of its cell
  x0, x1   its cell: the line is cut between neighbouring centres (midpoints); the first and last cells reach
           half a pitch beyond their centres (the median spacing of the centres, else the crop height),
           clipped to the crop. The cells tile the line: a cursor or a touch on the screen picks the cell
           that holds the point
  i0, i1   its ink box: the cell, or with INK the columns of the cell that hold ink

Flags (0 = calibrated frame centres and their midpoints):

  PITCH    on a line of at least PITCH_MIN characters whose spacings are regular (every spacing within
           1/PITCH_TOL of the median: a fixed font pitch), each centre moves to the least-squares grid
           a + p k (k the character's index) before the cells are cut
  INK      given the crop (its column ink profile, ink_profile()), each cut between two centres moves to the
           middle of the emptiest run of columns between them (an ink-column gap), each ink box shrinks to
           the columns of its cell above 1/INK_FRAC of the line's peak, and the centre becomes the middle of
           that ink. A cell without such a column keeps its box and centre

Everything is integer arithmetic, so the C runtime computes the same numbers bit for bit
(tests/test_charbox.py, `make check-charbox`).

to_screen() maps crop boxes to the screen: a horizontal crop at (x, y, w, h) gives x + box with the line's rows
y .. y + h; a vertical column read turned 90 degrees anticlockwise (synth/vertical_twin.rotate_ccw: the column's
top is the strip's left) gives y + box along the column, with its columns x .. x + w.
"""
import numpy as np

STRIDE = 8
Q = 16                      # fixed point: 1/16 px
CENTRE_RUN, PITCH, INK = 1, 2, 4
PITCH_MIN = 4               # characters a line needs before its pitch is trusted
PITCH_TOL = 4               # spacing tolerance: within 1/PITCH_TOL (25%) of the median spacing
INK_FRAC = 4                # a column holds ink when its profile is above 1/INK_FRAC of the line's peak
MAX_CHARS = 256             # bound of the C runtime's arrays (vocr_charbox.h VOCR_CHARBOX_MAX)

# Character groups of the calibration (code points). 0 is every other character.
G_OTHER, G_CLOSE, G_OPEN, G_DOTS, G_SMALL, G_LONG, G_MARK, G_HALF = range(8)
N_GROUPS = 8
GROUP_NAMES = ["other", "close", "open", "dots", "small", "long", "mark", "half"]
_CLOSE = "、。，．」』）】〉》〕｝］〙〗”’"
_OPEN = "「『（【〈《〔｛［〘〖“‘"
_DOTS = "…‥・"
_SMALL = "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"
_LONG = "ー―─～〜－"
_MARK = "！？♪☆★♥♡※"


def group(cp):
    """The calibration group of a code point (int) or a one-character string."""
    ch = chr(cp) if isinstance(cp, int) else cp
    if not ch:
        return G_OTHER
    if ch in _CLOSE:
        return G_CLOSE
    if ch in _OPEN:
        return G_OPEN
    if ch in _DOTS:
        return G_DOTS
    if ch in _SMALL:
        return G_SMALL
    if ch in _LONG:
        return G_LONG
    if ch in _MARK:
        return G_MARK
    o = ord(ch)
    if o < 0x80 or 0xFF61 <= o <= 0xFF9F:
        return G_HALF
    return G_OTHER


def spans(best):
    """[(class, f0, f1)] of the greedy text: one entry per emitted character, f0..f1 its run of frames."""
    out = []
    last = 0
    for t, c in enumerate(np.asarray(best, np.int64).tolist()):
        if c and c != last:
            out.append([c, t, t])
        elif c and c == last:
            out[-1][2] = t
        last = c
    return [tuple(s) for s in out]


def _div_round(a, b):
    """round(a / b), half away from zero, for integers (b > 0): the C side's arithmetic."""
    return (a + b // 2) // b if a >= 0 else -((-a + b // 2) // b)


def resized_width(w, h, height=32, max_width=1024, min_width=8):
    """vocr_resized_width(): round(w * height / h), half to even, clamped to [min_width, max_width]."""
    num = w * height
    q = (2 * num + h) // (2 * h)
    if (2 * num + h) % (2 * h) == 0 and q & 1:
        q -= 1
    return max(min_width, min(max_width, q))


def ink_profile(grey):
    """Column ink of a line crop (uint8 H x W, grey as the recognizer reads it, either polarity): per column
    the sum over its rows of |g(y + 1, x) - g(y, x)|, the vertical gradient energy. int64 array of W."""
    g = np.asarray(grey, np.int64)
    if g.ndim != 2 or g.shape[0] < 2:
        return np.zeros(g.shape[-1] if g.ndim == 2 else 0, np.int64)
    return np.abs(np.diff(g, axis=0)).sum(axis=0)


def positions(best, w, h, nw, flags=0, profile=None, cal=None, groups=None, stride=STRIDE, cap=MAX_CHARS):
    """[{"cls", "f0", "f1", "c", "x0", "x1", "i0", "i1"}] for the greedy characters of one line crop of w x h
    px read at resized width nw (the first `cap` of them); c, x0, x1, i0, i1 in 1/16 crop px (divide by Q).
    profile: ink_profile(crop), used with INK. cal: N_GROUPS + 1 ints, resized px x 16: [shift, offset of
    group 0, ..., group 7] (None: zeros). groups: class -> group (a sequence or dict), needed with cal."""
    sp = spans(best)[:cap]
    n = len(sp)
    if not n or not w or not nw:
        return []
    cal = list(cal) if cal is not None else [0] * (N_GROUPS + 1)
    c = []
    for cls, f0, f1 in sp:
        r = Q * stride * (f0 + f1 + 1) // 2 if flags & CENTRE_RUN else Q * (stride * f0 + stride // 2)
        g = groups[cls] if groups is not None else G_OTHER
        r += cal[0] + cal[1 + g]
        c.append(min(max(_div_round(r * w, nw), 0), w * Q))
    d = sorted(c[i + 1] - c[i] for i in range(n - 1))
    pitch = d[(len(d) - 1) // 2] if d else h * Q
    if pitch <= 0:
        pitch = h * Q
    if (flags & PITCH) and n >= PITCH_MIN and all(abs(x - pitch) * PITCH_TOL <= pitch for x in d):
        sk = n * (n - 1) // 2                  # least squares of c_k = a + p k over k = 0 .. n - 1
        skk = (n - 1) * n * (2 * n - 1) // 6
        sc = sum(c)
        skc = sum(k * v for k, v in enumerate(c))
        den = n * skk - sk * sk
        p_num = n * skc - sk * sc
        a_num = sc * skk - sk * skc
        c = [min(max(_div_round(a_num + k * p_num, den), 0), w * Q) for k in range(n)]
        pitch = max(1, _div_round(p_num, den))
    cuts = [_div_round(c[i] + c[i + 1], 2) for i in range(n - 1)]
    lo = [max(0, c[0] - pitch // 2)] + cuts
    hi = cuts + [min(w * Q, c[-1] + pitch // 2)]
    out = []
    for i in range(n):
        x0, x1 = lo[i], max(lo[i], hi[i])
        out.append({"cls": sp[i][0], "f0": sp[i][1], "f1": sp[i][2], "c": c[i], "x0": x0, "x1": x1,
                    "i0": x0, "i1": x1})
    if (flags & INK) and profile is not None and len(profile) == w:
        _ink(out, np.asarray(profile, np.int64), w)
    return out


def _ink(out, prof, w):
    """INK (the module docstring): cuts to ink-column gaps, ink boxes and centres to the ink of each cell."""
    peak = int(prof.max()) if len(prof) else 0
    if peak <= 0:
        return
    thr = peak // INK_FRAC
    n = len(out)
    for i in range(n - 1):
        a = (out[i]["c"] + Q - 1) // Q          # whole columns strictly between the two centres
        b = out[i + 1]["c"] // Q
        if b - a < 1:
            continue
        m = int(prof[a:b].min())
        best_len, best_s, run_s = 0, a, -1      # the first longest run of minimal columns
        for x in range(a, b + 1):
            on = x < b and int(prof[x]) == m
            if on and run_s < 0:
                run_s = x
            if not on and run_s >= 0:
                if x - run_s > best_len:
                    best_len, best_s = x - run_s, run_s
                run_s = -1
        cut = best_s * Q + best_len * Q // 2
        out[i]["x1"] = out[i]["i1"] = cut
        out[i + 1]["x0"] = out[i + 1]["i0"] = cut
    for o in out:
        a = o["x0"] // Q
        b = min(w, (o["x1"] + Q - 1) // Q)
        first = last = -1
        for x in range(a, b):
            if int(prof[x]) > thr:
                if first < 0:
                    first = x
                last = x
        if first >= 0:
            i0, i1 = max(o["x0"], first * Q), min(o["x1"], (last + 1) * Q)
            if i1 > i0:
                o["i0"], o["i1"] = i0, i1
                o["c"] = (i0 + i1) // 2


def class_groups(vocr):
    """group() of every class of a pyvocr.Vocr model (class 0, the blank, is G_OTHER)."""
    out = [G_OTHER] * vocr.n_classes
    for k in range(1, vocr.n_classes):
        t = vocr.class_text(k)
        out[k] = group(t) if len(t) == 1 else G_OTHER
    return out


SYNC = 8                    # vocr_charbox.h VOCR_CHARBOX_SYNC


def align_map(src, dst, sync=SYNC):
    """dst[j] -> the index of the src character it is aligned to, or -1 (inserted): another text's characters
    (the beam + LM text, the normalised text, a hook's text) placed on a line's greedy characters. src, dst:
    sequences of code points (or one-character strings). Equal characters align; at a difference, the shortest
    skip (di on src, dj on dst; di + dj = s for s = 1 .. 2 sync, each at most sync) after which two
    characters agree (or a text ends right after the first) is taken, the most balanced first (|di - dj|,
    then dj); its first min(di, dj) characters are substitutions, the rest of dst's insertions and the rest
    of src's dropped. With no such skip the two characters are a substitution. runtime/vocr_charbox.c
    vocr_charbox_align computes the same map."""
    src = [ord(c) if isinstance(c, str) else int(c) for c in src]
    dst = [ord(c) if isinstance(c, str) else int(c) for c in dst]
    n, m = len(src), len(dst)
    out = [-1] * m
    i = j = 0
    while i < n and j < m:
        if src[i] == dst[j]:
            out[j] = i
            i += 1
            j += 1
            continue
        found = None
        for s in range(1, 2 * sync + 1):
            best = None
            for di in range(max(0, s - sync), min(s, sync) + 1):
                dj = s - di
                if i + di >= n or j + dj >= m or src[i + di] != dst[j + dj]:
                    continue
                if not (i + di + 1 >= n or j + dj + 1 >= m or src[i + di + 1] == dst[j + dj + 1]):
                    continue
                key = abs(di - dj) * 64 + dj
                if best is None or key < best[0]:
                    best = (key, di, dj)
            if best is not None:
                found = best[1:]
                break
        if found is None:
            out[j] = i
            i += 1
            j += 1
            continue
        di, dj = found
        for t in range(min(di, dj)):
            out[j + t] = i + t
        i += di
        j += dj
    return out


def to_screen(boxes, crop, vertical=False, ink=False):
    """[(box (x0, y0, x1, y1), centre (x, y))] in float screen px for positions() entries of a crop at
    crop = (x, y, w, h) on the screen (vertical: the column before its turn). ink: the ink boxes, else the
    cells."""
    x, y, w, h = crop
    out = []
    for o in boxes:
        a, b = (o["i0"], o["i1"]) if ink else (o["x0"], o["x1"])
        a, b, c = a / Q, b / Q, o["c"] / Q
        if not vertical:
            out.append(((x + a, y, x + b, y + h), (x + c, y + h / 2)))
        else:
            out.append(((x, y + a, x + w, y + b), (x + w / 2, y + c)))
    return out

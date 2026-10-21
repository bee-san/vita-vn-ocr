"""Python twin of runtime/vocr_post.c (issue #109): the counter text rule and the counter-context re-decode.

    from vocr_post import counters, redecode, greedy
    text, n = counters("ルッ月です」")                       # -> "ルヶ月です」", 1
    topc2, stats = redecode(vocab, topv, topc, tau=2000)     # vocab[c]: class c's text ("" for the blank)
    text = greedy(vocab, topc2)                              # the edited greedy text

Every decision is the C code's, bit for bit: the costs are float32 logit differences rounded to Q10 nats as
runtime/vocr_lm.c rounds them, so tests/post_parity.py can require identical output. The contexts and the
corpus counts behind them are in docs/counters.md.
"""
import numpy as np

TSU, KE, KA = "\u30c3", "\u30f6", "\u30f5"           # ッ ヶ ヵ
NUMERALS = set("〇一二三四五六七八九十百千万0123456789０１２３４５６７８９")
QUANT = set("何数幾")
COUNTERS = set("月所年国条")
OPEN_FOR = {"」": "「", "』": "『"}
SIDE, MAX_SPAN, MAX_CAND = 4, 16, 32
MAX_TAU = 64000
MAX_FRAMES = 1024


def _tsu_is_ke(left, right):
    """ッ between left and right is a misread ヶ (rule 1 of runtime/vocr_post.h)."""
    if right in ("月", "年"):
        return True
    if right in ("国", "条"):
        return left in NUMERALS or left in QUANT
    if right == "所":
        return (left in NUMERALS and left != "一") or left in QUANT
    return False


def counters(text):
    """(text with every counter ッ made ヶ, changes)."""
    out = list(text)
    n = 0
    prev = ""
    for i, c in enumerate(out):
        if c == TSU and _tsu_is_ke(prev, out[i + 1] if i + 1 < len(out) else ""):
            out[i] = KE
            c = KE
            n += 1
        prev = c
    return "".join(out), n


def milli_q10(x):
    a = abs(int(x))
    q = (a * 1024 + 500) // 1000
    return -q if x < 0 else q


def cost(top, v):
    """vocr_lm.c's rounding of a logit difference to Q10 nats, in float32."""
    d = np.float32(np.float32(top) - np.float32(v))
    if not d >= np.float32(0):
        d = np.float32(0)
    elif d > np.float32(64):
        d = np.float32(64)
    return int(np.float32(np.float32(d * np.float32(1024)) + np.float32(0.5)))


def _cp(vocab, c):
    t = vocab[c]
    return t if len(t) == 1 else None          # a class of several characters (or the blank) matches nothing


def _is_letter(ch):
    if ch is None or len(ch) != 1:
        return False
    o = ord(ch)
    return (0x30A1 <= o <= 0x30FA or o == 0x30FC or 0x3041 <= o <= 0x3096 or 0x4E00 <= o <= 0x9FFF
            or 0x3400 <= o <= 0x4DBF or o == 0x3005 or "A" <= ch <= "Z" or "a" <= ch <= "z"
            or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A)


def _numq(ch):
    return ch is not None and (ch in NUMERALS or ch in QUANT)


def _pos(tc, f, c):
    row = tc[f]
    for j in range(len(row)):
        if int(row[j]) == c:
            return j
    return -1


def _swap0(tc, f, j):
    if j > 0:
        tc[f, 0], tc[f, j] = tc[f, j], tc[f, 0]


def _try_numeral(vocab, tv, tc, x, prev_end, nxt, tau):
    cls, s, e = x
    if e - s + 1 > MAX_SPAN:
        return s, False
    w0 = s - SIDE if s > SIDE else 0
    if w0 <= prev_end:
        w0 = prev_end + 1
    w1 = e + SIDE if e + SIDE < nxt else nxt - 1
    n = w1 - w0 + 1
    bc, bok = [], []
    for i in range(n):
        f = w0 + i
        if int(tc[f, 0]) == 0:
            bc.append(0)
            bok.append(True)
        else:
            j = _pos(tc, f, 0)
            bok.append(j > 0)
            bc.append(cost(tv[f, 0], tv[f, j]) if j > 0 else 0)
    pb, pok = [0], [True]
    for i in range(n):
        pb.append(pb[-1] + bc[i])
        pok.append(pok[-1] and bok[i])
    sb, sok = [0] * (n + 1), [True] * (n + 1)
    for i in range(n - 1, -1, -1):
        sb[i] = sb[i + 1] + bc[i]
        sok[i] = sok[i + 1] and bok[i]
    cand = []
    for f in range(s, e + 1):
        if len(cand) >= MAX_CAND:
            break
        for j in range(1, tc.shape[1]):
            if len(cand) >= MAX_CAND:
                break
            c = int(tc[f, j])
            if not c or c == cls or c in cand:
                continue
            if _numq(_cp(vocab, c)):
                cand.append(c)
    best = None                                   # (cost, class, a, b)
    for d in cand:
        dok, dc = [], []
        for i in range(n):
            j = _pos(tc, w0 + i, d)
            dok.append(j >= 0)
            dc.append(cost(tv[w0 + i, 0], tv[w0 + i, j]) if j >= 0 else 0)
        for a in range(n):
            if not pok[a]:
                break
            if prev_end >= 0 and w0 + a == prev_end + 1 and int(tc[prev_end, 0]) == d:
                continue
            run = 0
            for b in range(a, n):
                if not dok[b]:
                    break
                run += dc[b]
                if not sok[b + 1]:
                    continue
                tot = pb[a] + run + sb[b + 1]
                if best is None or tot < best[0] or (tot == best[0] and d < best[1]):
                    best = (tot, d, w0 + a, w0 + b)
    if best is None or best[0] > tau:
        return s, False
    _, d, a, b = best
    for f in range(w0, w1 + 1):
        if a <= f <= b:
            _swap0(tc, f, _pos(tc, f, d))
        elif int(tc[f, 0]) != 0:
            _swap0(tc, f, _pos(tc, f, 0))
    return a, True


def _try_open(vocab, tv, tc, ch, end, open_q):
    best = None                                   # (cost, frame, position)
    for f in range(end):
        if int(tc[f, 0]) != 0:
            continue
        for j in range(1, tc.shape[1]):
            c = int(tc[f, j])
            if c and _cp(vocab, c) == ch:
                q = cost(tv[f, 0], tv[f, j])
                if best is None or q < best[0]:
                    best = (q, f, j)
                break
    if best is None or best[0] > open_q:
        return False
    _swap0(tc, best[1], best[2])
    return True


def redecode(vocab, topv, topc, tau=0, open=0):
    """The re-decode on one line's top-K (frames x K logits, decreasing, and their classes): (the edited
    classes, {"edits", "contexts", "numerals", "brackets"}). tau / open in 1/1000 nat, as vocr_post_opts."""
    tv = np.asarray(topv, np.float32)
    tc = np.array(topc, np.uint16, copy=True)
    st = {"edits": 0, "contexts": 0, "numerals": 0, "brackets": 0}
    if not (0 <= tau <= MAX_TAU and 0 <= open <= MAX_TAU) or tc.ndim != 2 or not 1 <= tc.shape[1] <= 16:
        raise ValueError("bad options or top-K")                 # VOCR_TOPK_MAX = 16, as the C code checks
    if tc.shape[0] > MAX_FRAMES:
        raise ValueError("too many frames")
    if tc.size and int(tc.max()) >= len(vocab):
        raise ValueError("a top-K class outside the model")
    if not tau and not open:
        return tc, st
    tq, oq = milli_q10(tau), milli_q10(open)
    T = tc.shape[0]
    want = None
    cls = 0
    for f in range(T):
        c = int(tc[f, 0])
        if c and c != cls:
            ch = _cp(vocab, c)
            if ch in ("「", "『"):
                break
            if ch in OPEN_FOR:
                want = OPEN_FOR[ch]
                break
        cls = c
    runs = []
    cls, start = 0, 0
    for f in range(T + 1):
        c = int(tc[f, 0]) if f < T else 0
        if c == cls:
            continue
        if cls:
            runs.append((cls, start, f - 1))
            runs = runs[-4:]
            if len(runs) >= 3:
                x, mid, ctr = runs[-3], runs[-2], runs[-1]
                xc = _cp(vocab, x[0])
                if (_cp(vocab, mid[0]) in (TSU, KE, KA) and _cp(vocab, ctr[0]) in COUNTERS and _is_letter(xc)
                        and not _numq(xc)):
                    prev_end = runs[0][2] if len(runs) == 4 else -1
                    st["contexts"] += 1
                    first, done = x[1], False
                    if tq:
                        first, done = _try_numeral(vocab, tv, tc, x, prev_end, mid[1], tq)
                    if done:
                        st["edits"] += 1
                        st["numerals"] += 1
                    if oq and want and prev_end < 0 and _try_open(vocab, tv, tc, want, first, oq):
                        want = None
                        st["edits"] += 1
                        st["brackets"] += 1
        cls, start = c, f
    return tc, st


def greedy(vocab, topc):
    """The greedy text of the top-1 classes (topc[:, 0]): repeats merged, blanks dropped."""
    out, last = [], 0
    for c in np.asarray(topc)[:, 0]:
        c = int(c)
        if c and c != last:
            out.append(vocab[c])
        last = c
    return "".join(out)

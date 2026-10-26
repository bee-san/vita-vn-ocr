"""VOLM: the character n-gram LM blob that runtime/vocr_lm.c reads, and a bit-exact Python model of it.

This module is the specification of the C code (runtime/vocr_lm.c): the blob layout, the integer
backoff scorer and the CTC prefix beam search, written so that each step is the C code's step. The
tests (tests/test_ctclm.py) require the C decoder to give exactly this module's output.

Blob (little-endian, every section 16-byte aligned, the whole blob 16-byte aligned in memory):

  header, 64 bytes
    0  magic "VOLM"            4  u32 version (1)       8  u32 total bytes
    12 u32 order N (2..6)      16 u32 tokens V          20 u32 flags (0)
    24 u32 CRC-32 (IEEE) of bytes [64, total)           28 u32 vocabulary offset (V - 3 u32 code points)
    32 16 bytes: the first 16 bytes of the sha256 of the build manifest (provenance), 48..63 zero
  order table, 6 x 32 bytes at 64: per order k = 1..N (unused rows zero)
    u32 n (entries; order 1: V)   u32 off_tok (k >= 2)   u32 off_prob   u32 off_bo (k < N)
    u32 off_child (k < N)          u32 off_cb_prob         u32 off_cb_bo (k < N)   u32 0
  sections: tok u16[n], prob u8[n], bo u8[n], child u32[n + 1], cb_prob i16[256], cb_bo i16[256];
            vocabulary u32[V - 3]: the code points of tokens 3..V-1, strictly increasing

Tokens: 0 <unk>, 1 <s>, 2 </s>, 3.. one Unicode code point each. Order 1 is indexed by token id;
order k >= 2 holds the n-grams sorted by (parent entry at order k-1, last token), so the children of
entry e of order k-1 are the order-k entries [child[e], child[e+1]). Log-probabilities and backoff
weights are natural-log, in Q10 (1/1024 nat), quantised to 256 levels per order and kind; backoff
code 0 is exactly 0. The model is prefix-closed (every n-gram's context is an entry).

Scores in the decoder are int32 in Q10 nats: a candidate's frame score is minus its logit margin to the
frame's top-1, rounded (top-1: 0; every other candidate at most -1, so with alpha = beta = 0 the greedy
path is the unique best path and the beam returns the greedy text, bit for bit).
"""
import struct
import zlib

import numpy as np

MAGIC = b"VOLM"
VERSION = 1
HEADER = 64
ORDER_ROW = 32
MAX_ORDER = 6
TABLE = HEADER + MAX_ORDER * ORDER_ROW
UNK, BOS, EOS, FIRST = 0, 1, 2, 3
NONE = -1
Q = 1024                       # Q10: 1/1024 nat
NEG = -(1 << 30)               # an impossible path score
MAX_BEAM = 16
MAX_TOPK = 16
MARGIN_CAP = 64.0              # nats: larger logit margins are clamped (scores stay far inside int32)
LP_FLOOR = -65536              # Q10: the LM term per character is clamped to [LP_FLOOR, 0]
MAX_FRAMES = 1024              # longest line the decoder takes (with the clamps, no score can reach NEG)
FLAG_BOS = 1                   # start each line from the <s> context (else: no context)
FLAG_EOS = 2                   # add alpha * log P(</s> | context) at the end of the line
CANCEL_EVERY = 8               # frames between two polls of the cancel callback


def al16(n):
    return (n + 15) & ~15


def crc32(b):
    return zlib.crc32(b) & 0xFFFFFFFF


# ---------------------------------------------------------------- writing

def write_blob(order, codepoints, orders, manifest_sha=b""):
    """orders[k-1] for k = 1..order: dict with numpy arrays
         tok  (k >= 2) uint16, prob uint8, bo (k < order) uint8, child (k < order) uint32 [n + 1],
         cb_prob int16[256], cb_bo (k < order) int16[256]
    codepoints: the code points of tokens 3.., strictly increasing."""
    codepoints = np.asarray(codepoints, np.uint32)
    if not 2 <= order <= MAX_ORDER or len(orders) != order:
        raise ValueError("order")
    if len(codepoints) and np.any(np.diff(codepoints.astype(np.int64)) <= 0):
        raise ValueError("code points must be strictly increasing")
    V = len(codepoints) + FIRST
    parts, rows, off = [], [], TABLE

    def put(arr):
        nonlocal off
        b = np.ascontiguousarray(arr).tobytes()
        o = off
        parts.append((o, b))
        off = al16(off + len(b))
        return o

    for k in range(1, order + 1):
        d = orders[k - 1]
        n = V if k == 1 else len(d["tok"])
        if len(d["prob"]) != n:
            raise ValueError(f"order {k}: prob length")
        row = [n, 0, 0, 0, 0, 0, 0, 0]
        if k >= 2:
            row[1] = put(np.asarray(d["tok"], "<u2"))
        row[2] = put(np.asarray(d["prob"], np.uint8))
        if k < order:
            if len(d["bo"]) != n or len(d["child"]) != n + 1:
                raise ValueError(f"order {k}: bo/child length")
            row[3] = put(np.asarray(d["bo"], np.uint8))
            row[4] = put(np.asarray(d["child"], "<u4"))
        row[5] = put(np.asarray(d["cb_prob"], "<i2"))
        if k < order:
            cb = np.asarray(d["cb_bo"], "<i2")
            if cb[0] != 0:
                raise ValueError("backoff code 0 must be exactly 0")
            row[6] = put(cb)
        rows.append(row)
    voc_off = put(codepoints.astype("<u4"))
    total = al16(off)
    buf = bytearray(total)
    for o, b in parts:
        buf[o:o + len(b)] = b
    for k, row in enumerate(rows):
        struct.pack_into("<8I", buf, HEADER + k * ORDER_ROW, *row)
    sha16 = (manifest_sha + bytes(16))[:16]
    struct.pack_into("<4sIIIIIII16s", buf, 0, MAGIC, VERSION, total, order, V, 0, 0, voc_off, sha16)
    struct.pack_into("<I", buf, 24, crc32(bytes(buf[HEADER:])))
    return bytes(buf)


# ---------------------------------------------------------------- reading

class LM:
    """A parsed VOLM blob, with the C scorer's exact integer arithmetic."""

    def __init__(self, blob):
        b = bytes(blob)
        if len(b) < TABLE or b[:4] != MAGIC:
            raise ValueError("not a VOLM blob")
        version, total, order, V, flags, crc, voc_off = struct.unpack_from("<7I", b, 4)
        if version != VERSION or total != len(b) or not 2 <= order <= MAX_ORDER or V < FIRST:
            raise ValueError("unsupported VOLM header")
        if crc != crc32(b[HEADER:total]):
            raise ValueError("VOLM CRC mismatch")
        self.blob, self.order, self.V = b, order, V
        self.codepoints = np.frombuffer(b, "<u4", V - FIRST, voc_off).astype(np.int64)
        self.n, self.tok, self.prob, self.bo, self.child, self.cbp, self.cbb = [], [], [], [], [], [], []
        for k in range(1, order + 1):
            n, ot, op, ob, oc, ocp, ocb, _ = struct.unpack_from("<8I", b, HEADER + (k - 1) * ORDER_ROW)
            self.n.append(n)
            self.tok.append(np.frombuffer(b, "<u2", n, ot).astype(np.int64) if k >= 2 else None)
            self.prob.append(np.frombuffer(b, np.uint8, n, op).astype(np.int64))
            self.bo.append(np.frombuffer(b, np.uint8, n, ob).astype(np.int64) if k < order else None)
            self.child.append(np.frombuffer(b, "<u4", n + 1, oc).astype(np.int64) if k < order else None)
            self.cbp.append(np.frombuffer(b, "<i2", 256, ocp).astype(np.int64))
            self.cbb.append(np.frombuffer(b, "<i2", 256, ocb).astype(np.int64) if k < order else None)
        self.sha16 = b[32:48]

    def token_of(self, text):
        """LM token of a recognizer class's text: one code point in the vocabulary, else <unk>."""
        if len(text) != 1:
            return UNK
        cp = ord(text)
        i = int(np.searchsorted(self.codepoints, cp))
        return FIRST + i if i < len(self.codepoints) and self.codepoints[i] == cp else UNK

    def find(self, k, parent, t):
        """Entry of order k + 1 under entry `parent` of order k with last token t, or NONE (binary search)."""
        lo, hi = int(self.child[k - 1][parent]), int(self.child[k - 1][parent + 1])
        toks = self.tok[k]
        while lo < hi:
            mid = (lo + hi) >> 1
            v = int(toks[mid])
            if v < t:
                lo = mid + 1
            elif v > t:
                hi = mid
            else:
                return mid
        return NONE

    def initial(self, flags):
        """The context state of an empty line: per order k = 1..N-1 the entry of the last k tokens."""
        s = [NONE] * (self.order - 1)
        if flags & FLAG_BOS:
            s[0] = BOS
        return tuple(s)

    def score(self, state, t):
        """(log P(t | state) in Q10, the state after t), as vocr_lm.c's lm_score computes them.

        state[k-1] is the order-k entry of the context's last k tokens (k = 1..N-1), or NONE. With
        e[k] = the order-(k+1) entry (last k context tokens, t), found among the children of state[k-1]:
            log P = prob(e[k]) for the largest k with e[k] present, plus the backoff weight of every
                    longer context entry that is present; with no e[k] at all, the unigram prob of t
                    plus every present context's backoff.
        The new state is (t, e[1], ..., e[N-2])."""
        N = self.order
        e = [NONE] * N
        for k in range(1, N):
            p = state[k - 1]
            e[k] = self.find(k, p, t) if p != NONE else NONE
        lp = 0
        for k in range(N - 1, 0, -1):
            if e[k] != NONE:
                lp += int(self.cbp[k][self.prob[k][e[k]]])
                break
            if state[k - 1] != NONE:
                lp += int(self.cbb[k - 1][self.bo[k - 1][state[k - 1]]])
        else:
            lp += int(self.cbp[0][self.prob[0][t]])
        return lp, (t, *e[1:N - 1])


# ---------------------------------------------------------------- the decoder

def tdiv(a, b):
    """C99 integer division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def milli_to_q10(x):
    """1/1000 -> Q10, rounded half away from zero (vocr_lm.c's milli_q10)."""
    a = abs(int(x))
    q = (a * Q + 500) // 1000
    return q if x >= 0 else -q


def margin_q10(top, v):
    """-round(min(top - v, 64) * 1024) in float32 arithmetic (a negative or NaN difference counts as 0),
    at most -1: a candidate's frame score (vocr_lm.c's cand_score)."""
    d = np.float32(np.float32(top) - np.float32(v))
    if not d >= np.float32(0):
        d = np.float32(0)
    elif d > np.float32(MARGIN_CAP):
        d = np.float32(MARGIN_CAP)
    m = int(np.float32(d * np.float32(Q)) + np.float32(0.5))
    return -max(m, 1)


def candidates(topv, topc, blank, k, gate):
    """One frame's candidates [(class, score)]: the top-1 at 0, then the top-K classes whose margin to the
    top-1 is below the gate, then the blank if it is not among them and within the gate."""
    out = [(int(topc[0]), 0)]
    for j in range(1, k):
        s = margin_q10(topv[0], topv[j])
        if -s >= gate:
            break
        out.append((int(topc[j]), s))
    if blank is not None and all(c != 0 for c, _ in out):
        s = margin_q10(topv[0], blank)
        if -s < gate:
            out.append((0, s))
    return out


def prefix_hash(h, c):
    return (h * 0x9E3779B1 + c + 1) & 0xFFFFFFFF


BRACKETS = ("「」", "『』", "（）", "【】", "〈〉", "《》", "〔〕", "［］")


def bracket_role(text):
    """0, or k (1..8) for the opener of bracket pair k, 0x80 | k for its closer (ASCII ( ) count as pair 3)."""
    if text == "(":
        return 3
    if text == ")":
        return 0x80 | 3
    for k, (o, c) in enumerate(BRACKETS, 1):
        if text == o:
            return k
        if text == c:
            return 0x80 | k
    return 0


def bracket_next(st, r):
    """(the bracket state after a character of role r, mismatch): the state is the last two unclosed openers
    (low byte the innermost); a closer of another pair than the innermost open one is a mismatch."""
    s0, s1 = st & 0xFF, st >> 8
    if not r:
        return st, False
    if not r & 0x80:
        return r | s0 << 8, False
    r &= 0x7F
    if s0 == r:
        return s1, False
    if not s0:
        return st, False
    return (0 if s1 == r else st), True


def decode(lm, cmap, topv, topc, blank, beam=8, topk=8, alpha=0, beta=0, gate=0, flags=0, bracket=0, roles=None,
           stats=None, lexicon=None, lex_bonus=0):
    """CTC prefix beam search (Viterbi: max over alignments) with shallow LM fusion, exactly as
    vocr_lm_decode. topv/topc: frames x K, blank: frames or None; cmap: class -> LM token.
    alpha, beta, gate in 1/1000 (nat). Returns the classes of the best prefix.

    Per frame, every kept prefix (score ending in blank sb, ending in a character snb, s = max) moves:
      blank      -> the same prefix, sb' = s + q
      its last c -> the same prefix, snb' = snb + q (a repeat), and, from sb only, prefix + c
      other c    -> prefix + c, snb = s + q + alpha * log P_LM(c | prefix) + beta
    with max (not sum) where paths meet. A new prefix that equals a kept prefix (the same characters)
    is merged into it. The beam keeps the `beam` best prefixes by max(sb', snb'), ties to the earlier
    one (kept prefixes in beam order first, then the new ones in creation order). bracket > 0 (1/1000 nat)
    costs a closing bracket that does not match the innermost open one; roles[c] is class c's bracket_role.

    lexicon (a Lexicon; the dictionary-aware prototype of docs/vocr6-lm.md, Python only: vocr_lm.c has no such
    term): a new prefix whose last L >= 2 characters spell one of its words, L the longest, gains lex_bonus
    (1/1000 nat) x (L - 1). With lexicon None or lex_bonus 0 the search is vocr_lm_decode's, unchanged."""
    T = len(topv)
    if T > MAX_FRAMES:
        raise ValueError(f"{T} frames: the decoder takes at most {MAX_FRAMES}")
    if T and int(np.max(np.asarray(topc)[:, :topk])) >= len(cmap):
        raise ValueError("a top-K class is not below the model's class count (vocr_lm_decode: VOCR_E_ARG)")
    a, b, g, br = milli_to_q10(alpha), milli_to_q10(beta), milli_to_q10(gate), milli_to_q10(bracket)
    lb = milli_to_q10(lex_bonus) if lexicon is not None else 0
    use_lm = a != 0 and lm is not None
    # nodes: parent, class, length, hash, LM state; node 0 is the empty prefix
    par, cls, ln, hs, bk = [NONE], [0], [0], [0], [0]
    st = [lm.initial(flags) if lm is not None else ()]
    hyps = [(0, 0, NEG)]                     # (node, sb, snb), best first
    lm_calls = 0
    for t in range(T):
        cand = candidates(topv[t], topc[t], None if blank is None else blank[t], min(topk, len(topc[t])), g)
        stay = [[NEG, NEG] for _ in hyps]
        new = []                             # [parent node, class, snb, state]
        for i, (n, sb, snb) in enumerate(hyps):
            s = max(sb, snb)
            last = cls[n] if n else 0
            for c, q in cand:
                if c == 0:
                    stay[i][0] = max(stay[i][0], s + q)
                    continue
                if c == last:
                    if snb > NEG:
                        stay[i][1] = max(stay[i][1], snb + q)
                    src = sb
                else:
                    src = s
                if src <= NEG:
                    continue
                h = prefix_hash(hs[n], c)
                j = next((j for j, (m, _, _) in enumerate(hyps) if m and cls[m] == c and ln[m] == ln[n] + 1
                          and hs[m] == h and same_prefix(par, cls, par[m], n)), None)
                if use_lm:
                    lp, nst = lm.score(st[n], cmap[c])
                    lp = min(max(lp, LP_FLOOR), 0)
                    lm_calls += 1
                    lmq = tdiv(a * lp, Q)
                else:
                    lmq, nst = 0, st[n]
                val = src + q + lmq + b
                if lb:
                    L = lexicon.longest(cmap[c], n, par, cls, cmap)
                    if L >= 2:
                        val += lb * (L - 1)
                bst = 0
                if br:
                    bst, mis = bracket_next(bk[n], roles[c])
                    if mis:
                        val -= br
                if j is not None:
                    stay[j][1] = max(stay[j][1], val)
                else:
                    new.append((n, c, val, nst, bst))
        ranked = [(max(x), i) for i, x in enumerate(stay) if max(x) > NEG]
        ranked += [(e[2], len(hyps) + j) for j, e in enumerate(new) if e[2] > NEG]
        ranked.sort(key=lambda r: (-r[0], r[1]))
        nxt = []
        for _, idx in ranked[:beam]:
            if idx < len(hyps):
                nxt.append((hyps[idx][0], stay[idx][0], stay[idx][1]))
            else:
                n, c, val, nst, bst = new[idx - len(hyps)]
                par.append(n)
                cls.append(c)
                ln.append(ln[n] + 1)
                hs.append(prefix_hash(hs[n], c))
                bk.append(bst)
                st.append(nst)
                nxt.append((len(par) - 1, NEG, val))
        hyps = nxt
    best, best_s = 0, None
    for n, sb, snb in hyps:
        s = max(sb, snb)
        if use_lm and flags & FLAG_EOS:
            lp, _ = lm.score(st[n], EOS)
            s += tdiv(a * min(max(lp, LP_FLOOR), 0), Q)
        if best_s is None or s > best_s:
            best, best_s = n, s
    out = []
    n = best
    while n:
        out.append(cls[n])
        n = par[n]
    if stats is not None:
        stats["lm_calls"] = stats.get("lm_calls", 0) + lm_calls
        stats["nodes"] = max(stats.get("nodes", 0), len(par))
    return out[::-1]


def same_prefix(par, cls, x, y):
    """Nodes x and y of equal length spell the same characters."""
    while x != y:
        if cls[x] != cls[y]:
            return False
        x, y = par[x], par[y]
    return True


def class_map(lm, class_texts):
    """Recognizer class -> LM token (class 0, the blank, maps to <unk> and is never scored). Every bracket pair
    maps to the tokens of 「 and 」: the LM's two or three characters of context cannot tell a pair's style."""
    def fold(t):
        r = bracket_role(t)
        return t if not r else "」" if r & 0x80 else "「"
    return [UNK] + [lm.token_of(fold(t)) for t in class_texts[1:]]


class Lexicon:
    """Words as LM-token tuples, for decode(lexicon=...): longest() walks a new prefix back from its last character
    through the suffix set (every suffix of every word) and stops at the first suffix no word ends in; `steps`
    counts the set lookups (the console-cost estimate's operation count)."""

    def __init__(self, words, max_len=8):
        self.words = {w for w in words if 2 <= len(w) <= max_len}
        self.suffixes = {w[i:] for w in self.words for i in range(len(w))}
        self.max_len = max_len
        self.steps = 0

    def longest(self, t, node, par, cls, cmap):
        """The longest word length ending in token t after the prefix ending at node (0: none)."""
        key, best, m = (t,), 0, node
        while len(key) < self.max_len and m:
            key = (cmap[cls[m]],) + key
            m = par[m]
            self.steps += 1
            if key not in self.suffixes:
                break
            if key in self.words:
                best = len(key)
        return best


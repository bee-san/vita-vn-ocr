#!/usr/bin/env python3
"""Build the character n-gram LM (a VOLM blob, lm/vlm.py) that runtime/vocr_lm.c reads.

  build.py --out LM.vlm [--order 4] [--bytes 1000000] [--corpora aozora_dialogue,aozora_narration]
           [--jiten vn=0.2,novel=0.1] [--profiles ...] [--heldout realplus] [--manifest LM.json]

Deterministic: the same pinned inputs (lm/sources.json, every file checked by sha256) and arguments give the
same bytes (the tests build a small model twice). Steps:

1. Text. The running-text corpora are synth3's text pools (synth/synth3_text.py: norm_vn display form,
   pieces of at most 60 characters, every character in data/vocab.txt), one piece per line. Each piece is
   <s> c1 ... cn </s>. The Jiten frequency lists (jiten.moe, CC BY-SA 4.0; word, reading, rank) give
   VN- and novel-specific words: each word (norm_vn, every character in the vocabulary, at least 2
   characters) is a pseudo-sentence WB c1 ... cn WB counted max(1, round(S / rank)) times, with S set so
   that the list carries the requested share of the corpus characters. WB is a counting-only boundary
   token (the n-grams that contain it are dropped from the model; their mass goes to the backoff).
2. Counts of every n-gram of order 1..N inside a sentence (N <= 5: keys are base-(V+1) numbers in int64).
3. Interpolated modified Kneser-Ney (Chen & Goodman 1998): raw counts at order N, continuation counts below
   (raw for n-grams that start with <s> or WB), three discounts per order from the count-of-counts,
   interpolation with the uniform distribution at order 1.
4. Pruning to a byte budget: an order-k n-gram is kept when its raw count is at least t_k, with
   t = (1, t2, t3, t4) non-decreasing (so every kept n-gram's context is kept: the model is prefix-closed);
   the profile (t2 : t3 : t4) and its scale are searched so that the blob fits --bytes, each profile's
   largest fitting scale is kept, and of those the one with the lowest held-out cross-entropy wins.
   Kept probabilities are the unpruned model's; backoff weights are recomputed bottom-up so every context
   is normalised over what is kept.
5. Quantisation: per order, 256 equal-population bins for log-probabilities and for backoff weights (code 0
   of a backoff codebook is exactly 0, used by contexts with no kept extension), in Q10 nats.

Held-out text for the profile choice is never development data: by default it is the realplus v2 labels
(real VN lines of training-side VNs, the recognizer's training pool; the LM is not trained on them), never
teacher_val, devnative, the owner pool or any protected set.

Protected titles (always applied, before step 1): the VNs of eval/manifests/test_games.json ("all": vita-test,
psp-test, their families and release partners) and data/manifests/synth3_exclude_clannad.json (the CLANNAD
family), plus any --exclude-vns list. Their VNDB character names and titles (synth3's pinned VNDB extracts) become
text markers by synth's rule (synth/owner_games.marker_candidates): every title (>= 3 Japanese or >= 5 Latin
characters) and every full name of two or more parts (joined, >= 3 characters) always; a one-part name of >= 3
characters only when it hits at most 50 corpus pieces (more: a common word). A corpus piece holding a marker is
dropped. A Jiten word is dropped when it is a marker, or a protected character's full name or name part (>= 2
characters) that hits at most 50 corpus pieces. The counts go into the manifest.
"""
import argparse
import bisect
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "synth"))
import vlm  # noqa: E402

JP = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\u3005]")
MIN_MARKER = 3           # synth/owner_games.MIN_MARKER
MAX_RARE_HITS = 50       # synth/owner_games.MAX_RARE_HITS: a name hitting more corpus pieces is a common word

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
SOURCES = HERE / "sources.json"
DEFAULT_PROFILES = {   # (t2 : ... : tN) ratios of the count thresholds, per order
    2: ((1,),),
    3: ((1, 1), (1, 2), (2, 3)),
    4: ((1, 1, 1), (1, 2, 2), (1, 2, 4), (1, 1, 2), (1, 3, 6), (2, 3, 4)),
    5: ((1, 1, 1, 1), (1, 1, 2, 2), (1, 2, 2, 2), (1, 1, 1, 2), (1, 2, 3, 4), (2, 3, 4, 5)),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(p, base=ROOT):
    """A sources.json path: $HERD expanded, a relative path taken from `base` (the repository)."""
    p = Path(str(p).replace("$HERD", str(HERD)))
    return p if p.is_absolute() else base / p


def load_sources(path=SOURCES):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked(entry):
    p = resolve(entry["path"])
    if entry.get("sha256") and sha256(p) != entry["sha256"]:
        raise SystemExit(f"{p}: sha256 differs from lm/sources.json")
    return p


# ---------------------------------------------------------------- text

class Vocab:
    """LM tokens: 0 <unk>, 1 <s>, 2 </s>, 3.. the characters of data/vocab.txt by code point; WB (counting only)."""

    def __init__(self, chars):
        self.cps = np.array(sorted({ord(c) for c in chars}), np.int64)
        self.V = len(self.cps) + vlm.FIRST
        self.WB = self.V
        self.lut = np.zeros(0x110000, np.int32)          # code point -> token (0 = <unk>)
        self.lut[self.cps] = np.arange(vlm.FIRST, self.V, dtype=np.int32)

    def encode(self, s):
        if not s:
            return np.zeros(0, np.int32)
        return self.lut[np.frombuffer(s.encode("utf-32-le"), "<u4").astype(np.int64)]


def read_lines(path, limit=0):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s:
                out.append(s)
                if limit and len(out) >= limit:
                    break
    return out


def jiten_words(path, vocab):
    """{word: best rank} of a Jiten frequency-list CSV (Word,Form,Rank), in display form, in-vocabulary,
    at least 2 characters."""
    from synth3_text import norm_vn
    best = {}
    with open(path, encoding="utf-8") as f:
        head = f.readline().strip()
        if head != "Word,Form,Rank":
            raise SystemExit(f"{path}: unexpected header {head!r}")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            w, rank = norm_vn(parts[0]), int(parts[-1])
            if len(w) < 2 or not (vocab.encode(w) >= vlm.FIRST).all():
                continue
            if rank < best.get(w, 1 << 30):
                best[w] = rank
    return best


def protected(src, extra_vns=()):
    """(VN ids, [name parts] per character, titles) of the protected titles and any extra VNs."""
    from synth3_text import norm_vn
    ex = src["exclusions"]
    vns = set(json.loads(checked(ex["test_games"]).read_text(encoding="utf-8"))["all"])
    vns |= set(json.loads(checked(ex["clannad"]).read_text(encoding="utf-8"))["vns"])
    vns |= set(extra_vns)
    chars, titles = [], []
    with open(checked(ex["vndb_chars"]), encoding="utf-8") as f:
        col = {k: i for i, k in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            r = line.rstrip("\n").split("\t")
            if set(r[col["vn_ids"]].split(",")) & vns:
                chars.append([norm_vn(x) for x in r[col["name"]].split(" ") if x])
    with open(checked(ex["vndb_titles"]), encoding="utf-8") as f:
        col = {k: i for i, k in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            r = line.rstrip("\n").split("\t")
            if r[col["vn_id"]] in vns:
                titles.append(r[col["title"]])
    return vns, chars, titles


def hit_lines(text, starts, s, cap=None):
    """Indices of the pieces of `text` (start offsets `starts`) holding s; stops after cap + 1 when cap is set."""
    out, j = set(), text.find(s)
    while j >= 0:
        out.add(bisect.bisect_right(starts, j) - 1)
        if cap is not None and len(out) > cap:
            break
        j = text.find(s, j + 1)
    return out


def exclude_protected(corpus_lines, jiten, vns, chars, titles):
    """Drop the corpus pieces and the Jiten words that name a protected title or character (module docstring).
    Returns (kept pieces, filtered jiten, stats)."""
    always, rare = set(), set()
    for t in titles:
        t = unicodedata.normalize("NFC", t or "").strip()
        if t and (JP.search(t) and len(t) >= 3 or not JP.search(t) and len(t) >= 5):
            always.add(t)
            full = unicodedata.normalize("NFKC", t)
            if full != t and len(full) >= 3:
                always.add(full)
    words = set()
    for parts in chars:
        s = "".join(parts)
        if len(s) >= MIN_MARKER and JP.search(s):
            (always if len(parts) >= 2 else rare).add(s)
        words |= {w for w in (s, *parts) if len(w) >= 2 and JP.search(w)}
    rare -= always
    text = "\n".join(corpus_lines)
    starts, pos = [0], text.find("\n")
    while pos >= 0:
        starts.append(pos + 1)
        pos = text.find("\n", pos + 1)
    drop, hits = set(), {}
    for m in sorted(always):
        drop |= hit_lines(text, starts, m)
    for m in sorted((rare | words) - always):
        h = hit_lines(text, starts, m, MAX_RARE_HITS)
        hits[m] = len(h)
        if m in rare and len(h) <= MAX_RARE_HITS:
            drop |= h
    markers = always | {m for m in rare if hits[m] <= MAX_RARE_HITS}
    bad = markers | {w for w in words if hits.get(w, 0) <= MAX_RARE_HITS}
    kept = [x for i, x in enumerate(corpus_lines) if i not in drop]
    out, jdrop = {}, {}
    for name, (wd, share) in jiten.items():
        out[name] = ({w: r for w, r in wd.items() if w not in bad}, share)
        jdrop[name] = len(wd) - len(out[name][0])
    stats = {"vns": len(vns), "characters": len(chars), "titles": len(titles), "markers": len(markers),
             "common_names_not_markers": len(rare) - len(markers & rare),
             "corpus_pieces_dropped": len(drop), "jiten_words_dropped": jdrop}
    return kept, out, stats


def sequences(corpus_lines, vocab, jiten, corpus_chars):
    """Token sequences with multiplicities: corpus pieces (<s> ... </s>, x1) and Jiten pseudo-sentences."""
    seqs, mult = [], []
    BOS, EOS = vlm.BOS, vlm.EOS
    for s in corpus_lines:
        t = vocab.encode(s)
        seqs.append(np.concatenate([[BOS], t, [EOS]]).astype(np.int32))
        mult.append(1)
    jstats = {}
    for name, (words, share) in jiten.items():
        if share <= 0:
            continue
        items = sorted(words.items(), key=lambda kv: (kv[1], kv[0]))
        weight = sum(len(w) / r for w, r in items)
        S = share * corpus_chars / weight
        n_chars = 0
        for w, r in items:
            m = max(1, int(round(S / r)))
            seqs.append(np.concatenate([[vocab.WB], vocab.encode(w), [vocab.WB]]).astype(np.int32))
            mult.append(m)
            n_chars += m * len(w)
        jstats[name] = {"words": len(items), "scale": S, "weighted_chars": n_chars}
    return seqs, np.asarray(mult, np.int64), jstats


# ---------------------------------------------------------------- counting

def count_ngrams(seqs, mult, order, V1):
    """counts[k-1] = (sorted int64 keys, int64 counts) of every order-k n-gram inside a sequence.
    Key of (t1..tk) = sum t_i * V1^(k-i)."""
    lens = np.array([len(s) for s in seqs], np.int64)
    toks = np.concatenate(seqs).astype(np.int64)
    sid = np.repeat(np.arange(len(seqs)), lens)
    w = np.repeat(mult, lens)
    counts = []
    for k in range(1, order + 1):
        n = len(toks) - k + 1
        if n <= 0:
            counts.append((np.zeros(0, np.int64), np.zeros(0, np.int64)))
            continue
        ok = sid[:n] == sid[k - 1:k - 1 + n]
        key = np.zeros(n, np.int64)
        for j in range(k):
            key = key * V1 + toks[j:j + n]
        key, ww = key[ok], w[:n][ok]
        u, inv = np.unique(key, return_inverse=True)
        c = np.bincount(inv, weights=ww).astype(np.int64)
        counts.append((u, c))
    return counts


# ---------------------------------------------------------------- modified Kneser-Ney

def discounts(a):
    n = [int(np.sum(a == i)) for i in (1, 2, 3, 4)]
    if min(n) == 0:
        return np.array([0.5, 1.0, 1.5])
    Y = n[0] / (n[0] + 2 * n[1])
    d = np.array([1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2]])
    return np.clip(d, 0.05, None)


def kneser_ney(counts, order, V1, V_pred):
    """Interpolated modified KN. Returns per order k: keys, raw counts, log p (natural), and for k < N the
    gamma (backoff) of every order-k entry as a context (0 where it has no extension)."""
    keys = [c[0] for c in counts]
    raw = [c[1] for c in counts]
    adj = []
    for k in range(1, order + 1):
        if k == order:
            adj.append(raw[k - 1].astype(np.float64))
            continue
        # continuation count of g = number of distinct v with (v g) an order-(k+1) n-gram
        hk = keys[k]
        suffix = hk % (V1 ** k)
        su, sc = np.unique(suffix, return_counts=True)
        a = np.zeros(len(keys[k - 1]), np.float64)
        pos = np.searchsorted(keys[k - 1], su)
        a[pos] = sc
        first = keys[k - 1] // (V1 ** (k - 1))
        starts = (first == vlm.BOS) | (first == V1 - 1)            # <s> or WB: no left context
        a[starts] = raw[k - 1][starts]
        adj.append(a)
    logp, gamma = [], []
    prev_p = None
    uni_floor = 0.0
    for k in range(1, order + 1):
        a = adj[k - 1]
        D = discounts(np.round(a).astype(np.int64))
        ai = np.round(a).astype(np.int64)
        disc = np.where(ai == 1, D[0], np.where(ai == 2, D[1], D[2]))
        disc = np.where(ai == 0, 0.0, disc)
        if k == 1:
            A = a.sum()
            g = (D[0] * np.sum(ai == 1) + D[1] * np.sum(ai == 2) + D[2] * np.sum(ai >= 3)) / A
            p = np.maximum(a - disc, 0) / A + g / V_pred
            uni_floor = g / V_pred
        else:
            ctx = keys[k - 1] // V1
            cu, cinv = np.unique(ctx, return_inverse=True)
            A = np.bincount(cinv, weights=a)
            n1 = np.bincount(cinv, weights=(ai == 1).astype(np.float64))
            n2 = np.bincount(cinv, weights=(ai == 2).astype(np.float64))
            n3 = np.bincount(cinv, weights=(ai >= 3).astype(np.float64))
            gctx = (D[0] * n1 + D[1] * n2 + D[2] * n3) / A
            lower = keys[k - 1] % (V1 ** (k - 1))
            lpos = np.searchsorted(keys[k - 2], lower)
            if np.any(keys[k - 2][np.minimum(lpos, len(keys[k - 2]) - 1)] != lower):
                raise RuntimeError(f"order {k}: a suffix is missing from order {k - 1}")
            p = np.maximum(a - disc, 0) / A[cinv] + gctx[cinv] * prev_p[lpos]
            # the gamma of each order-(k-1) context entry
            g_entry = np.zeros(len(keys[k - 2]), np.float64)
            cpos = np.searchsorted(keys[k - 2], cu)
            if np.any(keys[k - 2][cpos] != cu):
                raise RuntimeError(f"order {k}: a context is missing from order {k - 1}")
            g_entry[cpos] = gctx
            gamma.append(g_entry)
        logp.append(np.log(p))
        prev_p = p
    gamma.append(np.zeros(len(keys[order - 1]), np.float64))
    return {"keys": keys, "raw": raw, "logp": logp, "gamma": gamma, "adj": adj, "uni_floor": uni_floor}


# ---------------------------------------------------------------- pruning, backoffs

def prune_and_backoff(kn, order, V1, V, thresholds):
    """Keep order-k n-grams (k >= 2) with raw count >= thresholds[k-1] and no WB/<unk>-free check needed;
    drop every n-gram containing WB (token V1 - 1); renormalise order 1 over the real tokens; recompute
    backoff weights bottom-up. Returns per order: keys, log p, log backoff (k < N)."""
    WB = V1 - 1
    keys, logp = [], []
    for k in range(1, order + 1):
        kk, lp, raw = kn["keys"][k - 1], kn["logp"][k - 1], kn["raw"][k - 1]
        has_wb = np.zeros(len(kk), bool)
        x = kk.copy()
        for _ in range(k):
            has_wb |= (x % V1) == WB
            x //= V1
        keep = ~has_wb
        if k >= 2:
            keep &= raw >= thresholds[k - 1]
        keys.append(kk[keep])
        logp.append(lp[keep])
    # order 1: every real token is an entry (index = token id): tokens never counted keep KN's uniform
    # floor; <s> is never predicted (its probability only has to be a valid code)
    p1 = np.full(V, kn["uni_floor"])
    p1[keys[0]] = np.exp(logp[0])
    p1[vlm.BOS] = 0.0
    p1 /= p1.sum()
    lp1 = np.log(np.maximum(p1, 1e-300))
    lp1[vlm.BOS] = lp1[np.arange(V) != vlm.BOS].min()
    keys[0] = np.arange(V, dtype=np.int64)
    logp[0] = lp1
    # prefix closure: drop any n-gram whose context is not kept (cannot happen with non-decreasing counts,
    # but WB removal and the guards make it cheap to enforce)
    for k in range(2, order + 1):
        ctx = keys[k - 1] // V1
        pos = np.searchsorted(keys[k - 2], ctx)
        ok = (pos < len(keys[k - 2])) & (keys[k - 2][np.minimum(pos, len(keys[k - 2]) - 1)] == ctx)
        keys[k - 1], logp[k - 1] = keys[k - 1][ok], logp[k - 1][ok]
    # backoffs, bottom-up: bo(h) = (1 - sum_kept p(w|h)) / (1 - sum_kept P_lower(w|h'))
    logbo = []
    for k in range(2, order + 1):
        kk, lp = keys[k - 1], logp[k - 1]
        ctx = kk // V1
        lower = kk % (V1 ** (k - 1))
        plow = np.exp(prob_lower(keys, logp, logbo, k - 1, lower, V1))
        cu, cinv = np.unique(ctx, return_inverse=True)
        num = 1.0 - np.bincount(cinv, weights=np.exp(lp))
        den = 1.0 - np.bincount(cinv, weights=plow)
        num = np.maximum(num, 1e-12)
        den = np.maximum(den, 1e-12)
        bo = np.zeros(len(keys[k - 2]), np.float64)
        bo[np.searchsorted(keys[k - 2], cu)] = np.log(num / den)
        logbo.append(bo)
    return keys, logp, logbo


def prob_lower(keys, logp, logbo, k, grams, V1):
    """log P(w | h') under the pruned model for order-k n-grams `grams` (keys of order k), recursively
    backing off; logbo[j-1] are the backoffs of order-j contexts already computed (j < k)."""
    if k == 1:
        return logp[0][grams]
    kk = keys[k - 1]
    pos = np.searchsorted(kk, grams)
    pos_c = np.minimum(pos, len(kk) - 1)
    found = (pos < len(kk)) & (kk[pos_c] == grams)
    out = np.empty(len(grams), np.float64)
    out[found] = logp[k - 1][pos_c[found]]
    nf = ~found
    if np.any(nf):
        g = grams[nf]
        ctx = g // V1
        cpos = np.searchsorted(keys[k - 2], ctx)
        cpos_c = np.minimum(cpos, len(keys[k - 2]) - 1)
        cfound = (cpos < len(keys[k - 2])) & (keys[k - 2][cpos_c] == ctx)
        bo = np.where(cfound, logbo[k - 2][cpos_c], 0.0)
        out[nf] = bo + prob_lower(keys, logp, logbo, k - 1, g % (V1 ** (k - 1)), V1)
    return out


# ---------------------------------------------------------------- quantisation, blob

def quantise(values, zero_code=False):
    """256 equal-population bins in sorted order; code per value and the Q10 int16 codebook."""
    v = np.asarray(values, np.float64)
    codes = np.zeros(len(v), np.uint8)
    cb = np.zeros(256, np.int16)
    if zero_code:
        nz = np.flatnonzero(v != 0.0)
        levels = 255
    else:
        nz = np.arange(len(v))
        levels = 256
    if len(nz):
        order = np.argsort(v[nz], kind="stable")
        bins = (np.arange(len(nz)) * levels) // len(nz)
        sums = np.bincount(bins, weights=v[nz][order], minlength=levels)
        cnt = np.bincount(bins, minlength=levels)
        means = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
        q = np.clip(np.round(means * vlm.Q), -32767, 32767).astype(np.int16)
        off = 1 if zero_code else 0
        cb[off:off + levels] = q
        # empty bins (fewer values than levels) repeat the previous level
        for i in range(1, levels):
            if cnt[i] == 0:
                cb[off + i] = cb[off + i - 1]
        c = np.empty(len(nz), np.int64)
        c[order] = bins + off
        codes[nz] = c.astype(np.uint8)
    return codes, cb


def assemble(keys, logp, logbo, order, V1, vocab, manifest_sha=b""):
    orders = []
    for k in range(1, order + 1):
        kk = keys[k - 1]
        d = {}
        pc, pcb = quantise(logp[k - 1])
        d["prob"], d["cb_prob"] = pc, pcb
        if k >= 2:
            d["tok"] = (kk % V1).astype(np.uint16)
        if k < order:
            bc, bcb = quantise(logbo[k - 1], zero_code=True)
            d["bo"], d["cb_bo"] = bc, bcb
            # children of entry e: order-(k+1) entries whose context is e
            child_ctx = keys[k] // V1
            pos = np.searchsorted(kk, child_ctx)
            d["child"] = np.searchsorted(pos, np.arange(len(kk) + 1), side="left").astype(np.uint32)
        orders.append(d)
    return vlm.write_blob(order, vocab.cps, orders, manifest_sha)


def blob_bytes(nk, order, V):
    """Exact VOLM size for entry counts nk[k-1] (order 1: V)."""
    off = vlm.TABLE
    for k in range(1, order + 1):
        n = V if k == 1 else nk[k - 1]
        if k >= 2:
            off = vlm.al16(off + 2 * n)
        off = vlm.al16(off + n)
        if k < order:
            off = vlm.al16(off + n)
            off = vlm.al16(off + 4 * (n + 1))
        off = vlm.al16(off + 512)
        if k < order:
            off = vlm.al16(off + 512)
    off = vlm.al16(off + 4 * (V - vlm.FIRST))
    return off


def kept_counts(kn, order, V1, thresholds):
    WB = V1 - 1
    nk = []
    for k in range(1, order + 1):
        kk, raw = kn["keys"][k - 1], kn["raw"][k - 1]
        x, has_wb = kk.copy(), np.zeros(len(kk), bool)
        for _ in range(k):
            has_wb |= (x % V1) == WB
            x //= V1
        keep = ~has_wb if k == 1 else (~has_wb & (raw >= thresholds[k - 1]))
        nk.append(int(keep.sum()))
    return nk


# ---------------------------------------------------------------- held-out cross-entropy

def cross_entropy(lm, lines, vocab, flags):
    """Mean nats per character of `lines` (those wholly inside the vocabulary) under the blob's integer
    scorer."""
    total, n = 0, 0
    for s in lines:
        toks = vocab.encode(s)
        if not len(toks) or (toks < vlm.FIRST).any():
            continue
        st = lm.initial(flags)
        for t in toks:
            lp, st = lm.score(st, int(t))
            total += lp
            n += 1
    return -total / vlm.Q / max(n, 1)


def heldout_lines(spec, limit, sources=SOURCES, seed=20261014):
    """Held-out VN text: realplus v2 labels (training-side VNs; never development or test data), or a file."""
    if spec != "realplus":
        return read_lines(resolve(spec, Path.cwd()), limit)
    src = load_sources(sources)["heldout"]["realplus"]
    p = checked(src)
    from synth3_text import norm_vn
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    lab = sorted({norm_vn(r["label"]) for r in rows if r.get("label") and not r.get("name")})
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lab))[:limit]
    return [lab[i] for i in sorted(idx)]


# ---------------------------------------------------------------- main

def build(args):
    src = load_sources(args.sources)
    vocab_path = checked(src["vocab"])
    chars = [c for c in vocab_path.read_text(encoding="utf-8").split("\n") if c]
    if any(len(c) != 1 for c in chars):
        raise SystemExit("data/vocab.txt: every class must be one character")
    vocab = Vocab(chars)
    V1 = vocab.V + 1                                   # + WB
    corpora, corpus_lines, paths = [], [], {"vocab": src["vocab"]["path"]}
    for name in [c for c in args.corpora.split(",") if c]:
        e = src["corpora"][name]
        p = checked(e)
        lines = read_lines(p, args.limit)
        corpus_lines += lines
        corpora.append({"name": name, "sha256": e["sha256"], "licence": e["licence"], "attribution": e["attribution"],
                        "lines": len(lines), "chars": sum(map(len, lines))})
        paths[name] = e["path"]
    corpus_chars = sum(c["chars"] for c in corpora)
    jiten, jmeta = {}, []
    for part in [p for p in args.jiten.split(",") if p]:
        name, _, share = part.partition("=")
        e = src["jiten"][name]
        p = checked(e)
        jiten[name] = (jiten_words(p, vocab), float(share))
        jmeta.append({"name": name, "url": e["url"], "sha256": e["sha256"], "licence": e["licence"],
                      "attribution": e["attribution"], "share": float(share)})
        paths[f"jiten_{name}"] = e["path"]
    extra = []
    for path in args.exclude_vns:
        extra += [v.strip() for v in Path(path).read_text(encoding="utf-8").split() if v.strip()]
    vns, chars, titles = protected(src, extra)
    corpus_lines, jiten, xstats = exclude_protected(corpus_lines, jiten, vns, chars, titles)
    xstats["extra_vns"] = len(set(extra))
    xstats["extra_vns_sha256"] = hashlib.sha256("\n".join(sorted(set(extra))).encode()).hexdigest() if extra else None
    corpus_chars = sum(map(len, corpus_lines))
    print(f"exclusion: {xstats}", flush=True)
    seqs, mult, jstats = sequences(corpus_lines, vocab, jiten, corpus_chars)
    for m in jmeta:
        m.update(jstats.get(m["name"], {}))
    print(f"corpus {len(corpus_lines):,} pieces, {corpus_chars:,} chars; jiten {jstats}", flush=True)
    counts = count_ngrams(seqs, mult, args.order, V1)
    print("n-grams per order: " + ", ".join(f"{len(c[0]):,}" for c in counts), flush=True)
    kn = kneser_ney(counts, args.order, V1, V_pred=vocab.V - 1)
    held = heldout_lines(args.heldout, args.heldout_lines, args.sources) if args.heldout else []
    held_sha = hashlib.sha256("\n".join(held).encode()).hexdigest() if held else None
    profiles = [tuple(int(x) for x in p.split(":")) for p in args.profiles.split(",")] if args.profiles \
        else DEFAULT_PROFILES[args.order]
    if any(len(p) != args.order - 1 for p in profiles):
        raise SystemExit(f"--profiles: order {args.order} needs {args.order - 1} ratios per profile")
    if V1 ** args.order >= 1 << 63:
        raise SystemExit("order too high for int64 n-gram keys at this vocabulary")
    tried = []
    best = None
    for prof in profiles:
        # largest-fitting (smallest) scale s: thresholds (1, s*prof...) non-decreasing
        lo, hi = 1, 1 << 24
        while lo < hi:
            s = (lo + hi) // 2
            th = (1, *[max(1, s * p) for p in prof])[:args.order]
            th = tuple(int(x) for x in np.maximum.accumulate(th))
            if blob_bytes(kept_counts(kn, args.order, V1, th), args.order, vocab.V) <= args.bytes:
                hi = s
            else:
                lo = s + 1
        th = tuple(int(x) for x in np.maximum.accumulate((1, *[max(1, lo * p) for p in prof])[:args.order]))
        keys, logp, logbo = prune_and_backoff(kn, args.order, V1, vocab.V, th)
        blob = assemble(keys, logp, logbo, args.order, V1, vocab)
        lm = vlm.LM(blob)
        ce = cross_entropy(lm, held, vocab, vlm.FLAG_BOS) if held else None
        tried.append({"profile": list(prof), "thresholds": [int(x) for x in th], "bytes": len(blob),
                      "entries": [int(len(k)) for k in keys], "heldout_nats_per_char": ce})
        print(f"profile {prof}: thresholds {th}, {len(blob):,} B, entries {[len(k) for k in keys]}, "
              f"held-out {ce if ce is None else round(ce, 4)} nats/char", flush=True)
        if best is None or (ce is not None and ce < best[0]):
            best = (ce, th, keys, logp, logbo)
    _, th, keys, logp, logbo = best
    # what the model is made of, by content only (no paths): its sha256 prefix goes into the blob's header
    manifest = {"kind": "volm-build", "version": vlm.VERSION, "order": args.order, "budget_bytes": args.bytes,
                "vocab": {"sha256": src["vocab"]["sha256"], "tokens": vocab.V},
                "corpora": corpora, "jiten": jmeta, "heldout_lines": len(held), "heldout_text_sha256": held_sha,
                "exclusion": dict(xstats, sources={k: v["sha256"] for k, v in src["exclusions"].items()},
                                  corpus_chars_kept=corpus_chars),
                "ngrams_counted": [int(len(c[0])) for c in counts], "profiles": tried,
                "chosen_thresholds": [int(x) for x in th],
                "builder": {"script": "lm/build.py", "numpy": np.__version__}}
    msha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).digest()
    blob = assemble(keys, logp, logbo, args.order, V1, vocab, msha)
    Path(args.out).write_bytes(blob)
    manifest.update({"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                     "manifest_sha256_prefix": msha[:16].hex(), "entries": [int(len(k)) for k in keys],
                     "paths": paths, "heldout": args.heldout})
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(blob):,} B sha256 {manifest['sha256']}", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--sources", default=str(SOURCES))
    ap.add_argument("--order", type=int, default=4, choices=(2, 3, 4, 5))
    ap.add_argument("--bytes", type=int, default=1_000_000)
    ap.add_argument("--corpora", default="aozora_dialogue,aozora_narration")
    ap.add_argument("--jiten", default="vn=0.2,novel=0.1")
    ap.add_argument("--profiles", default="", help="t2:t3:t4 ratios, comma-separated (default: a fixed set)")
    ap.add_argument("--heldout", default="realplus")
    ap.add_argument("--heldout-lines", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="read at most this many lines per corpus (tests)")
    ap.add_argument("--exclude-vns", action="append", default=[],
                    help="a file of VN ids (whitespace-separated) excluded like the protected titles (ablations)")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

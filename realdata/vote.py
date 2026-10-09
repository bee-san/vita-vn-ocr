"""Normalisation of OCR readings, agreement keys and the voting rules.

canon(s) is the label form, following the synthetic corpus (synth/build_corpus.py):
NFC, half-width katakana widened, no spaces, ASCII letters and digits (full-width
forms folded), full-width ！？～（）, dash-like glyphs folded to ―, and every
ellipsis written as … with one … per character cell (… and ‥ are one cell each, a run
of three . or ． is one cell). The nakaguro ・ is kept as it is. Page-cursor icons
(▽▼△▲◇◆) at the end of a line are dropped, as synth/render.py never labels its cursor.

key(s)    the strict agreement key: NFKC of canon(s).
loose(s)  key(s) with small kana folded to full size, ellipsis cell counts dropped and
          ー/一/― treated as one glyph: the confusions that OCRs make most on game fonts.

A rule takes the voters' readings {name: text} and returns (label, how) or (None, why).
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# The same as synth/build_corpus.norm (tests/test_realdata.py checks it), copied because
# build_corpus imports pyarrow, which labelling and CI don't need.
_TRANS = str.maketrans({"〜": "～", "—": "―", "\u00a0": "", "\u3000": "", " ": "", "\t": ""})
_HALF_KANA = re.compile(r"[\uff61-\uff9f]+")


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = _HALF_KANA.sub(lambda m: unicodedata.normalize("NFKC", m.group()), s)
    s = s.translate(_TRANS)
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("C"))


DASHES = "—―─━－‐‑–-"
DOT_RUN = re.compile(r"[…‥⋯.．]*(?:[…‥⋯]|[.．]{2,})[…‥⋯.．]*")
WIDE = str.maketrans({"!": "！", "?": "？", "~": "～", "(": "（", ")": "）", "·": "・"})
CURSOR = "▽▼∇△▲◇◆"                # page-cursor icons after the last character: never labelled
SMALL = str.maketrans("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ", "あいうえおつやゆよわアイウエオツヤユヨワカケ")
LONG = str.maketrans({"ー": "―", "一": "―"})


def _cells(run):
    cells = sum(run.count(c) for c in "…‥⋯")
    return max(1, cells + round((run.count(".") + run.count("．")) / 3))


def canon(s):
    s = norm(s or "")
    s = "".join(unicodedata.normalize("NFKC", c) if 0xFF10 <= ord(c) <= 0xFF5A and c.isalnum() else c
                for c in s)
    s = DOT_RUN.sub(lambda m: "…" * _cells(m.group()), s)
    s = "".join("―" if c in DASHES and (c != "-" or not _ascii_word_dash(s, i)) else c
                for i, c in enumerate(s))
    return s.translate(WIDE).rstrip(CURSOR)


def _ascii_word_dash(s, i):
    """A hyphen between ASCII letters or digits (e.g. 'Re-start') stays a hyphen."""
    return 0 < i < len(s) - 1 and s[i - 1].isascii() and s[i - 1].isalnum() \
        and s[i + 1].isascii() and s[i + 1].isalnum()


def key(s):
    return "".join(c if c == "…" else unicodedata.normalize("NFKC", c) for c in canon(s))


def loose(s):
    k = key(s).translate(SMALL).translate(LONG)
    return re.sub("…+", "…", k)


def japanese(s):
    s = key(s)
    return bool(s) and sum(ord(c) < 128 for c in s) <= 0.5 * len(s) and any(
        0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF or c == "々" for c in s)


# ---------------------------------------------------------------------------- rules

def _groups(reads, f):
    g = {}
    for name, text in reads.items():
        if text is not None:
            g.setdefault(f(text), []).append(name)
    return sorted(g.items(), key=lambda kv: -len(kv[1]))


def resolve(reads, names, order):
    """Label from voters that agree loosely: per-position majority of their canonical
    readings when those have the same length, else the first reading in `order`."""
    cands = [canon(reads[n]) for n in names]
    c = Counter(cands).most_common()
    if len(c) == 1 or c[0][1] > c[1][1]:
        return c[0][0]
    if len({len(x) for x in cands}) == 1:
        out = []
        for chars in zip(*cands):
            cc = Counter(chars).most_common()
            if len(cc) > 1 and cc[0][1] == cc[1][1]:
                return canon(reads[next(n for n in order if n in names)])
            out.append(cc[0][0])
        return "".join(out)
    return canon(reads[next(n for n in order if n in names)])


def old_rule(reads):
    """data/label_real.py: PP-OCRv5 server and manga-ocr agree (exactly, or up to the
    ellipsis spelling and small-kana size)."""
    a, b = reads.get("v5s"), reads.get("manga")
    if a is None or b is None:
        return None, "missing"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from label_real import decide
    lab = decide(a, b)
    return (canon(lab), "v5s=manga") if lab else (None, "disagree")


def make_rule(n_strict=None, n_loose=None, voters=("v5s", "manga", "v6s", "meiki"),
              required=(), order=("v6s", "meiki", "manga", "v5s")):
    """At least n_strict voters agree on key(), or at least n_loose on loose() (the
    label is then resolved among them); `required` voters must be in the agreeing set."""
    def rule(reads):
        rs = {n: reads.get(n) for n in voters}
        if any(rs[n] is None for n in required):
            return None, "missing"
        if not any(t and japanese(t) for t in rs.values() if t is not None):
            return None, "not-japanese"
        if n_strict:
            k, names = _groups(rs, key)[0]
            if k and len(names) >= n_strict and all(n in names for n in required):
                return canon(rs[names[0]]), f"strict{len(names)}"
        if n_loose:
            k, names = _groups(rs, loose)[0]
            if k and len(names) >= n_loose and all(n in names for n in required):
                return resolve(rs, names, order), f"loose{len(names)}"
        return None, "disagree"
    return rule


EDGE = set("「」『』【】（）()［］[]〈〉《》|｜/*。、!！?？")


def _trim_to(q, p):
    """q without the leading/trailing bracket or punctuation characters that p lacks:
    manga-ocr and meiki often invent a bracket or a full stop at a crop edge."""
    while q and q[0] in EDGE and not p.startswith(q[0]):
        q = q[1:]
    while q and q[-1] in EDGE and not p.endswith(q[-1]):
        q = q[:-1]
    return q


def make_cross_rule(pp=("v6s", "v5s"), other=("manga", "meiki"), trim=False, dots=False):
    """A PP-OCR reading (v6s first, then v5s) is the label when a reader of another
    family (manga-ocr or meiki) agrees with it on key(): errors shared inside a model
    family (Chinese glyph variants, まあ/まぁ, 一/ー) then need an independent second
    reader to make the same mistake. trim: the other reader may have extra edge
    brackets/punctuation. dots: ellipsis cell counts may differ when the PP reading is
    v6s's (its counts were right on the hand-checked dev set; manga-ocr's often aren't)."""
    def rule(reads):
        if not any(t and japanese(t) for t in reads.values() if t is not None):
            return None, "not-japanese"
        for a in pp:
            if reads.get(a) is None or not key(reads[a]):
                continue
            ka = key(reads[a])
            for b in other:
                if reads.get(b) is None:
                    continue
                kb = key(reads[b])
                if trim:
                    kb = _trim_to(kb, ka)
                if kb == ka or (dots and a == "v6s" and re.sub("…+", "…", kb) == re.sub("…+", "…", ka)):
                    return canon(reads[a]), f"{a}={b}"
        return None, "disagree"
    return rule


def _dotless(k):
    return re.sub("…+", "…", k)


def make_count_rule(n=3, trim=True, dots=True, veto=False, pp=("v6s", "v5s"), other=("manga", "meiki")):
    """At least n of the four readers support one PP-OCR reading (v6s's, else v5s's):
    a PP reader supports it with the same key(); manga-ocr or meiki also when they only
    add edge brackets/punctuation (trim) or, against v6s's reading, differ only in
    ellipsis cell counts (dots). Three supporters always span both model families."""
    def rule(reads):
        if not any(t and japanese(t) for t in reads.values() if t is not None):
            return None, "not-japanese"
        best = None
        for a in pp:
            if reads.get(a) is None or not key(reads[a]):
                continue
            ka = key(reads[a])
            sup = [x for x in pp if reads.get(x) is not None and key(reads[x]) == ka]
            for b in other:
                if reads.get(b) is None:
                    continue
                kb = _trim_to(key(reads[b]), ka) if trim else key(reads[b])
                if kb == ka or (dots and a == "v6s" and re.sub("…+", "…", kb) == re.sub("…+", "…", ka)):
                    sup.append(b)
            if len(sup) >= n and (best is None or len(sup) > len(best[1])):
                best = (a, sup)
        if best is None:
            return None, "disagree"
        ka = key(reads[best[0]])
        if veto:
            # a reader outside the majority that differs only in small kana or ー/一/― is
            # a real dissent on exactly the glyphs OCRs confuse systematically
            for x, t in reads.items():
                if x not in best[1] and t is not None and key(t) and loose(t) == loose(ka) \
                        and _dotless(key(t)) != _dotless(ka):
                    return None, "confusable-dissent"
        return canon(reads[best[0]]), f"{len(best[1])}:{'+'.join(sorted(best[1]))}"
    return rule


def _strip_edges(k):
    while k and k[0] in EDGE:
        k = k[1:]
    while k and k[-1] in EDGE:
        k = k[:-1]
    return k


def v5s_can_matter(reads):
    """Whether PP-OCRv5 server's reading could change the default rule's decision, given
    the other three readings. It cannot when v6s's reading has no other supporter (so it
    tops out at two) and manga-ocr and meiki don't agree with each other once their edge
    punctuation is stripped (so no v5s reading can collect three). label.py skips v5s,
    the slowest reader, on exactly those lines."""
    v6, mg, mk = reads.get("v6s"), reads.get("manga"), reads.get("meiki")
    if v6 is None or mg is None or mk is None:
        return True
    k6 = key(v6)
    for q in (mg, mk):
        kq = _trim_to(key(q), k6)
        if k6 and (kq == k6 or _dotless(kq) == _dotless(k6)):
            return True
    return _strip_edges(key(mg)) == _strip_edges(key(mk))


ALL4 = ("v5s", "manga", "v6s", "meiki")
RULES = {
    "old": old_rule,                                                     # label_real.py
    "v5s=manga": make_rule(n_strict=2, voters=("v5s", "manga")),
    "v5s~manga": make_rule(n_strict=2, n_loose=2, voters=("v5s", "manga")),
    "v5s=manga=v6s": make_rule(n_strict=3, voters=("v5s", "manga", "v6s")),
    "v5s~manga~v6s": make_rule(n_strict=3, n_loose=3, voters=("v5s", "manga", "v6s")),
    "2of3": make_rule(n_strict=2, voters=("v5s", "manga", "v6s")),
    "3of4": make_rule(n_strict=3, voters=ALL4),
    "3of4~": make_rule(n_strict=3, n_loose=3, voters=ALL4),
    "4of4": make_rule(n_strict=4, voters=ALL4),
    "v6s=meiki": make_rule(n_strict=2, voters=("v6s", "meiki")),
    "3of4-trim": make_count_rule(3, trim=True, dots=False),
    "3of4-trim-dots": make_count_rule(3, trim=True, dots=True),
    "3of4-dots": make_count_rule(3, trim=False, dots=True),
    "3of4-trim-dots-veto": make_count_rule(3, trim=True, dots=True, veto=True),
    "3of4-veto": make_count_rule(3, trim=False, dots=False, veto=True),
    "cross": make_cross_rule(),
    "cross-trim": make_cross_rule(trim=True),
    "cross-dots": make_cross_rule(dots=True),
    "cross-trim-dots": make_cross_rule(trim=True, dots=True),
    "cross-v6s": make_cross_rule(pp=("v6s",)),
    "cross-v6s-trim-dots": make_cross_rule(pp=("v6s",), trim=True, dots=True),
    "cross-manga": make_cross_rule(other=("manga",)),
    "cross-manga-trim-dots": make_cross_rule(other=("manga",), trim=True, dots=True),
}
DEFAULT_RULE = "3of4-trim-dots-veto"


def without(rule, name):
    """The rule run as if reader `name` had not been run: the ablations in RULES."""
    return lambda reads: rule({k: v for k, v in reads.items() if k != name})


for _name, _pp, _other in (("v5s", ("v6s",), ("manga", "meiki")), ("manga", ("v6s", "v5s"), ("meiki",)),
                           ("meiki", ("v6s", "v5s"), ("manga",))):
    RULES[f"{DEFAULT_RULE} without {_name}"] = without(
        make_count_rule(3, trim=True, dots=True, veto=True, pp=_pp, other=_other), _name)

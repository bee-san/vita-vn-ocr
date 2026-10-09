"""Label form, agreement keys and the acceptance rule of the teacher labeller.

canon(s)  the label written to the training set. It follows the synthetic corpus
          (synth/build_corpus.norm) and realdata's labels: NFC, half-width katakana
          widened, no spaces, ASCII letters and digits (full-width forms folded),
          full-width ！？～（）, dash-like glyphs as ―, every ellipsis as … with one … per
          character cell, and no page-cursor icon (▽▼△▲◇◆) at the end of a line.
key(s)    the agreement key: canon(s) as eval/metrics.norm() scores it (NFKC, ellipses,
          dashes and tildes folded). Two readings with the same key score identically on
          the benchmark. Small kana, dakuten, ー/一 and lookalike kanji are NOT folded, so
          engines must agree on exactly the glyphs the recognizer is judged on.

Rule (see accept()): MeikiOCR reads the line first. Its reading becomes the label when
  * MeikiOCR is confident: its lowest character confidence (punctuation re-scaled, see
    engines.Meiki) is at least `min_conf` and the detector's box score at least `det_conf`,
  * at least `votes` of the voters (PP-OCRv6 small, manga-ocr, Chrome Screen AI, PP-OCRv5
    server: whichever the rule lists in `voters`) read the same key, and no listed voter
    that disagrees differs from it only in a confusable glyph (`veto`, optional),
  * no voter that disagrees reads the label plus extra characters at one end (`trunc`,
    optional: the line probably runs on past MeikiOCR's box),
  * the line is Japanese (kana or kanji, at most half ASCII), at least `min_len` characters.
Thresholds are chosen by measurement on eval's hand-checked sets (teacher/measure.py).
"""
import functools
import re
import unicodedata

# ----------------------------------------------------------------------------- label form

_TRANS = str.maketrans({"〜": "～", "—": "―", "\u00a0": "", "\u3000": "", " ": "", "\t": ""})
_HALF_KANA = re.compile(r"[\uff61-\uff9f]+")
DASHES = "—―─━－‐‑–-"
DOT_RUN = re.compile(r"[…‥⋯.．]*(?:[…‥⋯]|[.．]{2,}|・{2,})[…‥⋯.．・]*")
WIDE = str.maketrans({"!": "！", "?": "？", "~": "～", "(": "（", ")": "）", "·": "・"})
CURSOR = "▽▼∇△▲◇◆"
SMALL = str.maketrans("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ", "あいうえおつやゆよわアイウエオツヤユヨワカケ")


def norm(s):
    """synth/build_corpus.norm: NFC, half-width katakana widened, no spaces or controls."""
    s = unicodedata.normalize("NFC", s)
    s = _HALF_KANA.sub(lambda m: unicodedata.normalize("NFKC", m.group()), s)
    s = s.translate(_TRANS)
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("C"))


def _cells(run):
    """Ellipsis cells in a run of dots: … ‥ ⋯ are one cell each, three . ． or ・ are one."""
    cells = sum(run.count(c) for c in "…‥⋯")
    return max(1, cells + round((run.count(".") + run.count("．") + run.count("・")) / 3))


def _ascii_word_dash(s, i):
    return 0 < i < len(s) - 1 and s[i - 1].isascii() and s[i - 1].isalnum() \
        and s[i + 1].isascii() and s[i + 1].isalnum()


@functools.lru_cache(maxsize=1 << 18)
def canon(s):
    s = norm(s or "")
    s = "".join(unicodedata.normalize("NFKC", c) if 0xFF10 <= ord(c) <= 0xFF5A and c.isalnum() else c
                for c in s)
    s = DOT_RUN.sub(lambda m: "…" * _cells(m.group()), s)
    s = "".join("―" if c in DASHES and (c != "-" or not _ascii_word_dash(s, i)) else c
                for i, c in enumerate(s))
    return s.translate(WIDE).rstrip(CURSOR)


# ----------------------------------------------------------------------------- scoring key
# The same folding as eval/metrics.norm (kept identical by tests/test_teacher.py when eval/
# is present), so "agree" means "score the same on the benchmark".
_E_DASHES = "―—─━–‐‑‒−-"
_E_TILDES = "〜～~∼"
_E_TRANS = str.maketrans({**{c: "―" for c in _E_DASHES}, **{c: "~" for c in _E_TILDES}})
_E_SPACE = re.compile(r"\s+")
_E_DOTS = re.compile(r"・{2,}")


def eval_norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = _E_SPACE.sub("", s)
    s = s.replace("...", "…").replace("..", "‥")
    s = _E_DOTS.sub(lambda m: "…" * max(1, len(m.group()) // 3), s)
    return s.translate(_E_TRANS)


@functools.lru_cache(maxsize=1 << 18)
def key(s):
    return eval_norm(canon(s))


def loose(s):
    """key() with small kana folded and ー/一/― merged: what a confusable dissent agrees on."""
    return key(s).translate(SMALL).translate(str.maketrans({"ー": "―", "一": "―"}))


def lev(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_kana_kanji(c):
    o = ord(c)
    return 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or c in "々〆〇ヶ"


@functools.lru_cache(maxsize=1 << 18)
def japanese(s):
    """At least one kana or kanji, and at most half ASCII (English UI, copyright lines)."""
    k = key(s)
    return bool(k) and sum(ord(c) < 128 for c in k) <= 0.5 * len(k) and any(map(is_kana_kanji, k))


# ----------------------------------------------------------------------------- the rule

VOTERS = ("v6s", "manga", "screenai", "v5s", "meiki", "meikiv")
DEFAULT = {
    "primary": "meikiv",                     # MeikiOCR reading the line view ("meiki": its own det box)
    "voters": ("v6s", "manga", "screenai"),  # which other readings count as votes
    "votes": 2,                              # voters that must read the primary's key
    "min_conf": 0.0,                         # primary: lowest character confidence
    "det_conf": 0.0,                         # MeikiOCR detector box score
    "veto": False,                           # a confusable dissent (small kana, ー/一) rejects
    "trunc": False,                          # a dissent that reads the label plus more at an edge rejects
    "min_len": 2,                            # characters in the label
}


def reading(line, name):
    """[text, chars] of one engine's reading of a line, or None. chars are
    [c, x0, y0, x1, y1, conf] in screenshot pixels (MeikiOCR only)."""
    if name == "meiki":
        m = line.get("meiki") or {}
        return [m.get("text") or "", m.get("chars") or []]
    t = (line.get("reads") or {}).get(name)
    if t is None:
        return None
    return [t[0], t[3] if len(t) > 3 else []]


def low_conf(chars):
    return min((c[5] for c in chars), default=0.0)


def undecided(line, rule):
    """Whether reading another of the rule's voters could still change accept()'s result for
    this line (label.py's cascade). False once the primary fails its own tests, once enough
    voters agree (without veto), or once too few unread voters remain to reach `votes`."""
    r = {**DEFAULT, **(rule or {})}
    lab, why = accept(line, {**r, "voters": (), "votes": 0, "veto": False})
    if lab is None:
        return False
    k = key(reading(line, r["primary"])[0])
    voters = [v for v in r["voters"] if v != r["primary"]]
    read = [v for v in voters if reading(line, v) is not None]
    agree = sum(key(reading(line, v)[0]) == k for v in read)
    left = len(voters) - len(read)
    if agree + left < r["votes"]:
        return False
    if not r["veto"] and not r.get("trunc"):
        return agree < r["votes"]
    return accept(line, r)[1] not in ("confusable-dissent", "truncated") and left > 0


def accept(line, rule=None):
    """(label, how) for one line read by label.py, or (None, why)."""
    r = {**DEFAULT, **(rule or {})}
    if line.get("vertical"):
        return None, "vertical"
    if not line.get("targets", ["?"]):
        return None, "size"
    if line.get("edge"):
        return None, "screen-edge"
    p = reading(line, r["primary"])
    if p is None:
        return None, "unread"
    text, chars = p
    k = key(text)
    if not k:
        return None, "empty"
    if not japanese(text):
        return None, "not-japanese"
    if len(k) < r["min_len"]:
        return None, "short"
    if low_conf(chars) < r["min_conf"]:
        return None, "low-conf"
    if line.get("det_conf", 1.0) < r["det_conf"]:
        return None, "low-det"
    agree, dissent = [], []
    for v in r["voters"]:
        if v == r["primary"]:
            continue
        t = reading(line, v)
        if t is None:
            continue
        (agree if key(t[0]) == k else dissent).append(v)
    if len(agree) < r["votes"]:
        return None, "disagree"
    if r["veto"]:
        lk = loose(text)
        for v in dissent:
            t = reading(line, v)[0]
            if key(t) and loose(t) == lk:
                return None, "confusable-dissent"
    if r.get("trunc"):
        # a voter that reads the same text plus extra characters at an edge suggests the
        # line runs on past the crop (a bracket or a word the detector box left out)
        for v in r["trunc"] if isinstance(r["trunc"], (list, tuple)) else dissent:
            if v not in dissent:
                continue
            kv = key(reading(line, v)[0])
            if len(kv) > len(k) and (kv.endswith(k) or kv.startswith(k)):
                return None, "truncated"
    return canon(text), f"{r['primary']}+" + "+".join(agree)

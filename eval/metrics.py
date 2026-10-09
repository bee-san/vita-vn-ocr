"""Scoring for eval/run.py.

Both the prediction and the ground truth go through norm() before scoring, so
the score ignores differences that don't change the text a reader sees:

  * Unicode NFKC: full-width ASCII and digits fold to ASCII (！ -> !, １ -> 1),
    half-width katakana to full width, compatibility forms to their base;
  * all whitespace is removed (VN dialogue has none; models emit stray spaces);
  * an ellipsis is one character: NFKC's "..." (from …) and "‥" (from ..) fold
    back, and runs of katakana middle dots "・・・" read as an ellipsis;
  * wave dashes/tildes (〜 ～ ~ ∼) are one character, and so are the horizontal
    dashes (― — ─ ━ – ‐ ‑ − -); the long-vowel mark ー and the kanji 一 stay distinct.

Small kana (っ/つ), dakuten and lookalike kanji/katakana are NOT folded: those
are real recognition errors.

CER = sum of Levenshtein distances / sum of ground-truth lengths (both after
norm), over all scored lines. Exact-line accuracy = share of lines whose
normalised prediction equals the normalised ground truth.
"""
import re
import unicodedata

_DASHES = "―—─━–‐‑‒−-"
_TILDES = "〜～~∼"
_TRANS = str.maketrans({**{c: "―" for c in _DASHES}, **{c: "~" for c in _TILDES}})
_SPACE = re.compile(r"\s+")
_DOTS = re.compile(r"・{2,}")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = _SPACE.sub("", s)
    s = s.replace("...", "…").replace("..", "‥")
    s = _DOTS.sub(lambda m: "…" * max(1, len(m.group()) // 3), s)
    return s.translate(_TRANS)


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


class Tally:
    """Accumulates CER and exact-line accuracy."""

    __slots__ = ("errs", "chars", "exact", "lines")

    def __init__(self):
        self.errs = self.chars = self.exact = self.lines = 0

    def add(self, errs, chars):
        self.errs += errs
        self.chars += chars
        self.exact += errs == 0
        self.lines += 1

    def summary(self):
        return {"lines": self.lines, "chars": self.chars, "errors": self.errs,
                "cer": round(self.errs / self.chars, 5) if self.chars else None,
                "exact": round(self.exact / self.lines, 5) if self.lines else None}


def score_lines(rows):
    """rows: dicts with pred, text, styles, vn. Adds per-line 'errors'/'n' and
    returns {"all": ..., "styles": {tag: ...}, "games": {vn: ...}, "game_macro_cer": x}."""
    total, styles, games = Tally(), {}, {}
    for r in rows:
        p, g = norm(r["pred"]), norm(r["text"])
        e = lev(p, g)
        r["errors"], r["n"] = e, len(g)
        total.add(e, len(g))
        for tag in r.get("styles") or []:
            styles.setdefault(tag, Tally()).add(e, len(g))
        games.setdefault(r.get("vn") or "-", Tally()).add(e, len(g))
    per_game = {k: v.summary() for k, v in sorted(games.items())}
    cers = [v["cer"] for v in per_game.values() if v["cer"] is not None]
    return {"all": total.summary(),
            "styles": {k: v.summary() for k, v in sorted(styles.items())},
            "games": per_game,
            "game_macro_cer": round(sum(cers) / len(cers), 5) if cers else None}


def oov_chars(texts, vocab_set):
    """Ground-truth characters a model cannot emit (checked raw and NFKC-folded)."""
    if not vocab_set:
        return {}
    folded = {unicodedata.normalize("NFKC", c) for c in vocab_set}
    out = {}
    for t in texts:
        for c in t:
            if c.isspace() or c in vocab_set or unicodedata.normalize("NFKC", c) in folded:
                continue
            out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

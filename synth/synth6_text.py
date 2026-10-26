"""synth6 text: what the synthetic lines say.

Four sources, mixed by CFG weights (synth6.py):

  pool      synth3's VN-register pools (Aozora speech and narration, subtitles, roleplay, Tatoeba, spoken corpora,
            templates) through synth3's own dialogue_text / narration_text / short_text, with its focus control
            and exclusion markers (CLANNAD's names and title never appear)
  hook      the train titles' own hook text (train pages only: never an hval, guard or test page), re-wrapped to
            the drawing title's box; a title's text is drawn in another title's style, so a synthetic line never
            copies a real crop
  target    a line holding a character the real lines hold rarely or never (the vocab characters' counts in the
            train titles' verified lines, `real_counts.json`): weight (count + 5) ** -0.7, times 4 for a known
            confusable, restricted to what the drawing font can draw; its context is a pool line holding it, else
            a Jiten visual-novel word holding it in a short carrier sentence (synth4's carriers), else a name
  confusion the confusions reports/vocr5-analysis.md (#119) measured: ＝ between katakana words, Latin letters and
            digits inside Japanese lines, 『』 against 「」, small kana, katakana with (han)dakuten, ー against 一 and
            lookalike kana and kanji pairs, drawn in natural contexts or carriers

Every label is checked against the test titles' text: a line sharing an 8-character run with a test title's
line, or equal to one of four or more characters, is never drawn (`clean_label`). Nor is a line equal to one of the
train titles' hval or guard lines (the screening set): any such line from the corpus' own text, and one of six or
more characters from the other sources (shorter ones are generic: 「…………」, a name).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "teacher"))
import synth3 as S3  # noqa: E402
from vote import canon  # noqa: E402

NGRAM = 8
HELD_MIN = 6           # a pool, target or confusion line equal to an hval line is dropped from this length
KATA = re.compile(r"[ァ-ヺー]{3,}")
# confusable groups (synth3's CONFUSABLE plus #119's measured confusions)
CONFUSION_GROUPS = list(S3.CONFUSABLE) + ["＝ー", "『「", "』」", "Iｌ1！", "ヴブ", "ヶケ", "ヵカ", "ゎわ",
                                         "ゐい"]
LATIN_WORDS = ["I", "l", "D", "BGM", "VTuber", "mSv/h", "OK", "CD", "DVD", "No.1", "Lv.5", "HP", "MP", "SNS", "PC",
               "TV", "AI", "NG", "VS", "Ver.2", "LINE", "BL", "OL", "IT", "SF", "UFO", "DNA", "SOS", "VIP", "X",
               "A組", "B級", "Sランク", "Tシャツ", "Wi-Fi", "CM", "DJ", "MC", "PR", "GPS", "USB", "ID", "km", "kg",
               "cm", "mm", "kHz", "MHz", "ATM", "CEO", "POWER", "LOVE", "Yes", "No", "Hello", "Sir", "Mr.", "Dr."]
LATIN_CARRIERS = ["{w}って何？", "それは{w}だ。", "{w}のことだよ。", "「{w}」", "{w}が好きなんだ。", "あの{w}がさ……",
                  "{w}、ですか？", "これが{w}か。", "{w}で決まりだ！", "今日の{w}は最高だった。", "{w}と{w2}。"]
EQ_CARRIERS = ["{a}＝{b}", "「{a}＝{b}」", "{a}＝{b}さん", "{a}＝{b}の話だ。", "私は{a}＝{b}。", "{a}＝{b}……？",
               "{a}＝{b}だって？", "{a}＝{b}と申します。"]
DOUBLE_CARRIERS = ["『{s}』", "『{s}』と書いてあった。", "「『{s}』って言ってたよ」", "『{s}』……"]


def ngrams(s, n=NGRAM):
    return {s[i:i + n] for i in range(len(s) - n + 1)}


class TestFilter:
    """The test titles' text, for the decontamination check only (never drawn)."""

    def __init__(self, lines):
        keys = [canon(s) for s in lines if s]
        self.exact = {k for k in keys if len(k) >= 4}
        self.grams = set()
        for k in keys:
            self.grams |= ngrams(k)

    def ok(self, label):
        k = canon(label)
        if k in self.exact:
            return False
        return not (ngrams(k) & self.grams)


class TextResources(S3.Resources):
    """synth3's text resources (pools, contexts, names, lists, focus targets) without its fonts or scenes."""

    def _fonts(self, manifest_path, holdout):
        return []


class Text6:
    def __init__(self, assets, vocab, s3_assets=S3.ASSETS, curriculum=None, exclude=S3.EXCLUDE_MANIFESTS):
        assets = Path(assets)
        S3.R = TextResources(assets=s3_assets, split="train", profile="personal", vocab=vocab, scenes=False,
                             exclude=exclude)
        S3.S2.R = None
        self.R = S3.R
        self.vocab = [c for c in self.R.vocab if c != S3.UNK]
        self.vset = set(self.vocab)
        self.cid = {c: i for i, c in enumerate(self.R.vocab)}
        self.hook = {}            # title -> [(utterance, lines)], names separately
        self.hook_names = []
        with open(assets / "hooktext.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["kind"] == "line":
                    self.hook.setdefault(r["title"], []).append(r["text"])
                elif r["kind"] == "name":
                    self.hook_names.append(r["text"])
        self.hook_titles = sorted(self.hook)
        real = json.loads((assets / "real_counts.json").read_text(encoding="utf-8"))["chars"]
        self.real = real
        conf = set("".join(CONFUSION_GROUPS))
        cnt = np.array([real.get(c, 0) for c in self.R.vocab], np.float64)
        w = (cnt + 5.0) ** -0.7 * np.where([c in conf for c in self.R.vocab], 4.0, 1.0)
        self.words = {}
        if curriculum:
            cur = json.loads(Path(curriculum).read_text(encoding="utf-8"))
            self.words = {c: [(x, float(v)) for x, v in ws if all(ch in self.vset for ch in x)]
                          for c, ws in cur["words"].items() if c in self.vset}
        self.names_idx = {}
        for src, lst in self.R.names.items():
            for i, parts in enumerate(lst):
                for ch in set("".join(parts)):
                    if ch in self.vset:
                        lst2 = self.names_idx.setdefault(ch, [])
                        if len(lst2) < 32:
                            lst2.append((src, i))
        has = np.array([bool(self.R.has_ctx[i]) or bool(self.words.get(c)) or c in self.names_idx
                        for i, c in enumerate(self.R.vocab)])
        w[~has] = 0.0
        w[[i for i, c in enumerate(self.R.vocab) if c == S3.UNK]] = 0.0
        self.target_w = w
        self.real_counts = cnt
        tf = assets / "test_text.json"
        self.test = TestFilter(json.loads(tf.read_text(encoding="utf-8"))["lines"] if tf.exists() else [])
        hf = assets / "hval_text.json"
        self.held = {canon(x) for x in json.loads(hf.read_text(encoding="utf-8"))["lines"]} if hf.exists() else set()
        self._font_w = {}
        kata = []
        for src in ("vndb", "syosetu"):
            for parts in self.R.names.get(src, [])[:200000]:
                for p in parts:
                    if KATA.fullmatch(p) and 3 <= len(p) <= 7:
                        kata.append(p)
        self.kata_words = sorted(set(kata))

    # ---- targets
    def target_char(self, rng, font=None):
        """A vocab class index drawn by the rare-character weights, restricted to what `font` draws."""
        w = self.target_w
        if font is not None:
            key = id(font)
            if key not in self._font_w:
                m = np.array([font.has(c) for c in self.R.vocab], np.float64)
                fw = w * m
                self._font_w[key] = fw / fw.sum() if fw.sum() > 0 else None
            p = self._font_w[key]
            if p is None:
                return None
        else:
            p = w / w.sum()
        return int(rng.choice(len(p), p=p))

    def word_line(self, rng, ch, kind="dialogue"):
        """A Jiten VN word holding ch in a short carrier sentence (synth4's carriers), or None. A word holding an
        exclusion marker (CLANNAD's character names and title; the Jiten list covers CLANNAD's script) is never
        used, as in synth4's context4."""
        markers = self.R.exclusions["markers"]
        ws = [(w, v) for w, v in self.words.get(ch, ()) if not (markers and S3.marked(w, markers))]
        if not ws:
            return None
        x = np.array([v for _, v in ws], np.float64) ** 0.5
        word = ws[int(rng.choice(len(ws), p=x / x.sum()))][0]
        if kind == "narration":
            t = rng.choice(["{w}。", "{w}が見えた。", "そこには{w}があった。", "{w}を思い出す。", "目の前の{w}。"])
        elif word[-1] in "うくぐすつぬぶむるい":
            t = rng.choice(["{w}。", "{w}んだ。", "{w}のか？", "{w}……", "もう{w}。", "{w}かもしれない。",
                            "まだ{w}。"])
        else:
            t = rng.choice(["{w}", "{w}……", "{w}？", "{w}！", "{w}って、何？", "{w}だ。", "これが{w}か。",
                            "{w}のこと？", "あの{w}が……", "{w}を見た。", "それは{w}だ。", "{w}はどこ？",
                            "ああ、{w}ね。"])
        return t.format(w=word)

    def name_line(self, rng, ch):
        lst = self.names_idx.get(ch)
        if not lst:
            return None
        src, i = lst[int(rng.integers(len(lst)))]
        parts = self.R.names[src][i]
        holding = [p for p in parts if ch in p] or list(parts)
        n = holding[int(rng.integers(len(holding)))]
        return rng.choice(["{n}！", "{n}……", "{n}？", "{n}さん！", "{n}、待って！", "{n}の部屋。", "ねえ、{n}。",
                           "{n}が来た。", "{n}って誰？"]).format(n=n)

    def target_line(self, rng, font=None):
        c = self.target_char(rng, font)
        if c is None:
            return None, None
        ch = self.R.vocab[c]
        r = rng.random()
        s = None
        if r < 0.55 and self.R.has_ctx[c]:
            s, _ = self.R.context(rng, c, "dialogue" if rng.random() < 0.7 else "narration")
        if s is None and r < 0.85:
            s = self.word_line(rng, ch)
        if s is None:
            s = self.name_line(rng, ch) or self.word_line(rng, ch)
        if s is None and self.R.has_ctx[c]:
            s, _ = self.R.context(rng, c)
        if s is None:
            return None, ch
        s = S3.clean(s).strip("「」『』")
        if s and rng.random() < 0.5:
            s = "「" + s + "」"
        return s, ch

    # ---- confusions
    def confusion_line(self, rng, font=None):
        k = rng.random()
        if k < 0.2 and len(self.kata_words) > 10:
            a = self.kata_words[int(rng.integers(len(self.kata_words)))]
            b = self.kata_words[int(rng.integers(len(self.kata_words)))]
            return rng.choice(EQ_CARRIERS).format(a=a, b=b), "kata_eq"
        if k < 0.4:
            w = LATIN_WORDS[int(rng.integers(len(LATIN_WORDS)))]
            w2 = LATIN_WORDS[int(rng.integers(len(LATIN_WORDS)))]
            if rng.random() < 0.3:
                w = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in w)
            return rng.choice(LATIN_CARRIERS).format(w=w, w2=w2), "latin"
        if k < 0.55:
            s, _, _ = S3.dialogue_text(rng, False)
            s = s.strip("「」『』")
            return rng.choice(DOUBLE_CARRIERS).format(s=s[:30]), "double_bracket"
        g = CONFUSION_GROUPS[int(rng.integers(len(CONFUSION_GROUPS)))]
        chars = [c for c in g if c in self.cid]
        if not chars:
            return None, None
        parts = []
        for ch in chars[: int(rng.integers(1, len(chars) + 1))]:
            c = self.cid[ch]
            s = None
            if self.R.has_ctx[c]:
                s, _ = self.R.context(rng, c)
            if s is None:
                s = self.word_line(rng, ch)
            if s:
                parts.append(S3.clean(s).strip("「」『』"))
        if not parts:
            return None, None
        s = "".join(p if p[-1:] in "。！？…" else p + "。" for p in parts)
        return s, "group:" + g

    # ---- the mix
    def utterance(self, rng, mix, title=None, font=None, narration=0.18):
        """(text, source, focus) for one dialogue-box utterance; text may hold 「」 and full-width spaces."""
        src = S3.choose(rng, mix)
        if src == "hook":
            others = [t for t in self.hook_titles if t != title] or self.hook_titles
            t = others[int(rng.integers(len(others)))]
            lst = self.hook[t]
            return lst[int(rng.integers(len(lst)))], f"hook:{t}", None
        if src == "target":
            s, ch = self.target_line(rng, font)
            if s:
                return s, "target", ch
        if src == "confusion":
            s, why = self.confusion_line(rng, font)
            if s:
                return s, "confusion:" + why, None
        r = rng.random()
        if r < narration:
            s, pool, focus = S3.narration_text(rng)
        elif r < narration + 0.12:
            s, pool, focus = S3.short_text(rng, True)
        else:
            s, pool, focus = S3.dialogue_text(rng, rng.random() < 0.5)
        return s, "pool:" + str(pool), (None if focus is None else self.R.vocab[focus])

    def name(self, rng):
        """(display, source) of a speaker name: a train title's hook name or synth3's name sampler."""
        if self.hook_names and rng.random() < 0.4:
            return self.hook_names[int(rng.integers(len(self.hook_names)))], "hook_name"
        disp, _, src = S3.sample_name(rng)
        return disp, "name:" + src

    def clean_label(self, label, source=""):
        """Drawable as a label: in the vocab, no test title's text, and not an hval or guard line of the corpus
        (any length for text taken from the corpus itself, six characters or more for other sources)."""
        if not label or not all(c in self.vset for c in label) or not self.test.ok(label):
            return False
        if label in self.held and (len(label) >= HELD_MIN or (source or "").startswith("hook")):
            return False
        return True

#!/usr/bin/env python3
"""Build VN-style text pools and the recognizer vocabulary from the fetched corpora.

Reads data/corpus/aozora/*.jsonl.gz and data/corpus/wiki/**/*.parquet and writes
  data/corpus/dialogue.txt   Aozora chunks that are speech (start with 「 or 『)
  data/corpus/narration.txt  the other Aozora chunks
  data/corpus/wiki.txt       Japanese Wikipedia chunks
  data/corpus/char_freq.json per-character counts over the three pools
  data/vocab.txt             one recognizer class per line (CTC blank is index 0)
Spaces are removed: VN dialogue has none, and the recognizer never emits them.
"""
import collections
import gzip
import json
import random
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[1] / "data"
CORPUS = DATA / "corpus"
CAP = {"dialogue": 600_000, "narration": 600_000, "wiki": 400_000}
MAX_LEN = 120          # chunk length; a Vita text box shows about 3 x 28 characters
COVERAGE = 0.9998      # character coverage target for the vocabulary
MAX_VOCAB = 6000
MIN_COUNT = 20

HIRA = "".join(chr(c) for c in range(0x3041, 0x3097))
KATA = "".join(chr(c) for c in range(0x30A1, 0x30FB)) + "ーヽヾゝゞ"
PUNCT = ("、。，．・：；？！゛゜―‐／＼～｜…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】"
         "＋－±×÷＝≠＜＞≦≧∞∴♂♀°′″℃￥＄％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼※→←↑↓♪♭♯♥♡々〆〇〃")
FULL_ALNUM = "".join(chr(c) for c in [*range(0xFF10, 0xFF1A), *range(0xFF21, 0xFF3B),
                                      *range(0xFF41, 0xFF5B)])
ASCII = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.,:;'\"()-/&%#+=*~"
MANDATORY = set(HIRA + KATA + PUNCT + FULL_ALNUM + ASCII)

# Visually identical code points are merged so the recognizer is not asked to
# guess between them; whitespace is dropped.
TRANS = str.maketrans({"〜": "～", "—": "―", "\u00a0": "", "\u3000": "", " ": "", "\t": ""})
HALF_KANA = re.compile(r"[\uff61-\uff9f]+")
SENT_END = re.compile(r"(?<=[。！？」』])")


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = HALF_KANA.sub(lambda m: unicodedata.normalize("NFKC", m.group()), s)
    s = s.translate(TRANS)
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("C"))


def is_jp(ch):
    o = ord(ch)
    return 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or ch in "々〆〇"


def chunks(par):
    """Split a paragraph into pieces of at most MAX_LEN at sentence ends."""
    if len(par) <= MAX_LEN:
        yield par
        return
    cur = ""
    for sent in SENT_END.split(par):
        if cur and len(cur) + len(sent) > MAX_LEN:
            yield cur
            cur = ""
        cur += sent
        while len(cur) > MAX_LEN:
            yield cur[:MAX_LEN]
            cur = cur[MAX_LEN:]
    if cur:
        yield cur


def keep(chunk):
    return len(chunk) >= 4 and sum(map(is_jp, chunk)) >= 0.6 * len(chunk)


class Reservoir:
    def __init__(self, cap):
        self.cap, self.seen, self.items = cap, 0, []

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            j = random.randrange(self.seen)
            if j < self.cap:
                self.items[j] = item


def aozora_paragraphs():
    for path in sorted((CORPUS / "aozora").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield from (json.loads(line).get("text") or "").split("\n")


def wiki_paragraphs():
    for path in sorted((CORPUS / "wiki").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2000):
            for text in batch.column(0).to_pylist():
                yield from (p for p in (text or "").split("\n") if len(p) >= 10)


def cp932_ok(ch):
    try:
        ch.encode("cp932")
    except UnicodeEncodeError:
        return False
    return not unicodedata.category(ch).startswith(("C", "Z"))


def main():
    random.seed(1234)
    pools = {name: Reservoir(cap) for name, cap in CAP.items()}
    for par in aozora_paragraphs():
        for c in chunks(norm(par)):
            if keep(c):
                pools["dialogue" if c[0] in "「『" else "narration"].add(c)
    print("aozora seen:", {k: pools[k].seen for k in ("dialogue", "narration")}, flush=True)
    for par in wiki_paragraphs():
        for c in chunks(norm(par)):
            if keep(c):
                pools["wiki"].add(c)
    print("wiki seen:", pools["wiki"].seen, flush=True)

    freq = collections.Counter()
    for name, pool in pools.items():
        (CORPUS / f"{name}.txt").write_text("\n".join(pool.items) + "\n", encoding="utf-8")
        for item in pool.items:
            freq.update(item)
        print(f"{name}: {len(pool.items)} chunks", flush=True)
    (CORPUS / "char_freq.json").write_text(
        json.dumps(dict(freq.most_common()), ensure_ascii=False), encoding="utf-8")

    total = sum(freq.values())
    vocab = set(MANDATORY)
    covered = sum(freq[c] for c in vocab)
    for ch, n in freq.most_common():
        if ch in vocab or not cp932_ok(ch):
            continue
        if covered / total >= COVERAGE or len(vocab) >= MAX_VOCAB or n < MIN_COUNT:
            break
        vocab.add(ch)
        covered += n
    vocab = sorted(vocab)
    (DATA / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    oov = [(c, n) for c, n in freq.most_common() if c not in set(vocab)][:30]
    print(f"vocab {len(vocab)} classes, coverage {covered / total:.5f} of {total} chars", flush=True)
    print("top OOV:", oov, flush=True)
    print("CORPUS_DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Character coverage of recognizer vocabs on the NVIDIA Japanese text, from the
per-shard counts that fetch.py run writes (every horizontal line of every page,
after the corpus normalisation, whether or not the line was cropped).

  oov_report.py [--out DIR] [--vocab data/vocab.txt ...] [--top 40] [--top-kanji 30] [--json FILE]

For each vocab: the share of characters it misses, split by kind (an <unk>-terminated
vocab, as in the arch sweep, counts its last entry as <unk>, not as a character). Then
the most frequent characters missing from the first vocab, and the most frequent kana
and kanji missing from each further vocab, as Markdown tables.
"""
import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synth"))
import fetch  # noqa: E402
from common import load_vocab  # noqa: E402

UNK = "〓"


def in_jis0208(c):
    """True for characters of JIS X 0208 (two EUC-JP bytes, both 0xA1-0xFE)."""
    try:
        b = c.encode("euc_jp")
    except UnicodeEncodeError:
        return False
    return len(b) == 2 and b[0] >= 0xA1 and b[1] >= 0xA1


def kind(c):
    o = ord(c)
    if (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x3FFFF
            or c in "々〆〇"):
        return "kanji, JIS X 0208" if in_jis0208(c) else "kanji, not in JIS X 0208"
    if 0x3040 <= o <= 0x30FF:
        return "kana"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "Hangul"
    cat = unicodedata.category(c)
    if cat.startswith("M"):
        return "combining mark"
    if o < 0x80:
        return "ASCII"
    if cat.startswith("L"):
        return "letter (Latin, Cyrillic, Greek...)"
    if o >= 0x1F000 or 0x2600 <= o <= 0x27BF:
        return "emoji / dingbat"
    return "symbol / punctuation, JIS X 0208" if in_jis0208(c) else "symbol / punctuation"


def shown(c):
    return f"◌{c}" if unicodedata.category(c)[0] in "MZC" else c


def load(out):
    recs = [json.loads(p.read_text()) for p in
            [*sorted(Path(out).glob("nv_*.json")), *sorted((Path(out) / "val_parts").glob("nv_*.json"))]]
    chars, stats = Counter(), Counter()
    for r in recs:
        chars.update(r.get("chars", {}))
        stats.update(r["stats"])
    return recs, chars, stats


def vocab_chars(path):
    vocab = load_vocab(path)
    return set(vocab[:-1] if vocab and vocab[-1] == UNK else vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(fetch.DEFAULT_OUT))
    ap.add_argument("--vocab", nargs="*", default=[str(HERE.parent / "data" / "vocab.txt")])
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--top-kanji", type=int, default=30)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    recs, chars, stats = load(args.out)
    total = sum(chars.values())
    if not total:
        raise SystemExit(f"no character counts under {args.out}")
    print(f"{len(recs)} shards, {stats['pages']:,} pages, {stats['h_lines']:,} horizontal lines, "
          f"{total:,} characters, {len(chars):,} distinct\n")
    kinds = sorted({kind(c) for c in chars})
    report = {"shards": len(recs), "pages": stats["pages"], "h_lines": stats["h_lines"], "chars": total,
              "distinct": len(chars), "vocabs": {}}
    print("| vocab | size | missed characters | " + " | ".join(kinds) + " |")
    print("|---|---|---|" + "---|" * len(kinds))
    for v in args.vocab:
        vs = vocab_chars(v)
        miss = Counter({c: n for c, n in chars.items() if c not in vs})
        by_kind = Counter()
        for c, n in miss.items():
            by_kind[kind(c)] += n
        name = Path(v).name
        report["vocabs"][name] = {"size": len(vs), "char_oov": sum(miss.values()) / total,
                                  "distinct_missed": len(miss),
                                  "by_kind": {k: by_kind[k] / total for k in kinds},
                                  "top_missing": [[c, f"U+{ord(c):04X}", kind(c), n]
                                                  for c, n in miss.most_common(args.top)]}
        print(f"| {name} | {len(vs):,} | {100 * sum(miss.values()) / total:.3f}% | "
              + " | ".join(f"{100 * by_kind[k] / total:.3f}%" for k in kinds) + " |")
    print(f"\nLines with a character outside {Path(args.vocab[0]).name}: "
          f"{100 * stats['h_oov_lines'] / max(1, stats['h_lines']):.2f}% of horizontal lines.")
    first = vocab_chars(args.vocab[0])
    miss = Counter({c: n for c, n in chars.items() if c not in first})
    print(f"\nMost frequent characters missing from {Path(args.vocab[0]).name}:\n")
    print("| rank | char | code point | kind | count | per million chars |\n|---|---|---|---|---|---|")
    for i, (c, n) in enumerate(miss.most_common(args.top), 1):
        print(f"| {i} | {shown(c)} | U+{ord(c):04X} | {kind(c)} | {n:,} | {1e6 * n / total:.1f} |")
    for v in args.vocab[1:]:
        vs = vocab_chars(v)
        jp = Counter({c: n for c, n in chars.items() if c not in vs and kind(c) in ("kana", "kanji, JIS X 0208",
                                                                                   "kanji, not in JIS X 0208")})
        top = jp.most_common(args.top_kanji)
        print(f"\nMost frequent kana and kanji missing from {Path(v).name} "
              f"({100 * sum(jp.values()) / total:.3f}% of all characters): "
              + " ".join(f"{c}({1e6 * n / total:.0f})" for c, n in top) + " (per million)")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()

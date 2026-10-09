#!/usr/bin/env python3
"""Character coverage of a recognizer vocab on the NVIDIA Japanese text, from the
per-shard counts that fetch.py run writes (every horizontal line of every page,
after the corpus normalisation, whether or not the line was cropped).

  oov_report.py [--out DIR] [--vocab data/vocab.txt ...] [--top 40] [--json FILE]

For each vocab: the share of characters it misses (an <unk>-terminated vocab, as in
the arch sweep, counts its last entry as <unk>, not as a character) and the share
of lines with at least one missing character. Then the most frequent characters
missing from the first vocab, grouped by kind, as a Markdown table.
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


def kind(c):
    o = ord(c)
    try:
        jis = c.encode("iso2022_jp") is not None
    except UnicodeEncodeError:
        jis = False
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x3FFFF:
        return "kanji, JIS X 0208" if jis else "kanji, not in JIS X 0208"
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
    return "symbol / punctuation, JIS X 0208" if jis else "symbol / punctuation"


def load(out):
    recs = [json.loads(p.read_text()) for p in
            [*sorted(Path(out).glob("nv_*.json")), *sorted((Path(out) / "val_parts").glob("nv_*.json"))]]
    chars, stats = Counter(), Counter()
    for r in recs:
        chars.update(r.get("chars", {}))
        stats.update(r["stats"])
    return recs, chars, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(fetch.DEFAULT_OUT))
    ap.add_argument("--vocab", nargs="*", default=[str(HERE.parent / "data" / "vocab.txt")])
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    recs, chars, stats = load(args.out)
    total = sum(chars.values())
    if not total:
        raise SystemExit(f"no character counts under {args.out}")
    print(f"{len(recs)} shards, {stats['pages']:,} pages, {stats['h_lines']:,} horizontal lines, "
          f"{total:,} characters, {len(chars):,} distinct\n")
    print("| vocab | characters | missed characters | distinct missed |\n|---|---|---|---|")
    report = {"pages": stats["pages"], "h_lines": stats["h_lines"], "chars": total, "distinct": len(chars),
              "vocabs": {}}
    for v in args.vocab:
        vocab = load_vocab(v)
        vs = set(vocab[:-1] if vocab and vocab[-1] == UNK else vocab)
        missed = sum(n for c, n in chars.items() if c not in vs)
        report["vocabs"][Path(v).name] = {"size": len(vs), "char_oov": missed / total,
                                          "distinct_missed": sum(c not in vs for c in chars)}
        print(f"| {Path(v).name} | {len(vs):,} | {100 * missed / total:.3f}% | "
              f"{report['vocabs'][Path(v).name]['distinct_missed']:,} |")
    print(f"\nLines with a character outside {Path(args.vocab[0]).name}: "
          f"{100 * stats['h_oov_lines'] / max(1, stats['h_lines']):.2f}% of horizontal lines.\n")
    vocab = set(load_vocab(args.vocab[0]))
    miss = Counter({c: n for c, n in chars.items() if c not in vocab})
    by_kind = Counter()
    for c, n in miss.items():
        by_kind[kind(c)] += n
    print("| kind of missing character | share of all characters | share of missed |\n|---|---|---|")
    for k, n in by_kind.most_common():
        print(f"| {k} | {100 * n / total:.4f}% | {100 * n / max(1, sum(miss.values())):.1f}% |")
    print("\n| rank | char | code point | kind | count | per million chars |\n|---|---|---|---|---|---|")
    top = []
    for i, (c, n) in enumerate(miss.most_common(args.top), 1):
        shown = c if unicodedata.category(c)[0] not in "MZC" else f"◌{c}"
        print(f"| {i} | {shown} | U+{ord(c):04X} | {kind(c)} | {n:,} | {1e6 * n / total:.1f} |")
        top.append({"char": c, "cp": f"U+{ord(c):04X}", "kind": kind(c), "count": n})
    report["by_kind"] = dict(by_kind)
    report["top_missing"] = top
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()

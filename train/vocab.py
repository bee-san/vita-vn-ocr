#!/usr/bin/env python3
"""Reduced recognizer vocabularies ranked by VN text frequency, and their coverage.

  vocab.py names [--out data/vndb_chars.jsonl]
      Fetches the original-script names of characters in Japanese-original VNs with
      at least 100 VNDB votes from the VNDB kana API (rate-limited, User-Agent set),
      one JSON object per character with the ids of the VNs it appears in. Local only,
      never committed.
  vocab.py build --size 2500 [--out data/vocab_2500.txt] [--extra FILE]
      Always keeps every kana (small kana, ヴ and ー included), the VN punctuation and
      symbol set of build_corpus.PUNCT and the digits. The other characters are ranked
      by a VN-weighted frequency: 95% the synthetic text pools mixed with the
      renderer's weights (dialogue 0.45, narration 0.35, wiki 0.20), 5% the kanji of
      VNDB character names (VN scripts repeat their cast's names, which use kanji
      that literary and encyclopedic text rarely does: 葵, 杏, 椎, 隼, 皐...). A name
      is left out if any of its VNs is one of eval's test games (eval's
      manifests/test_games.json "all", or synth/common.is_test_vn): the test sets'
      games never inform the ranking. The
      top size - 1 characters are kept and the <unk> token 〓 is appended. Only
      characters of the full vocab (data/vocab.txt) are eligible, so a reduced vocab
      is a subset of v1's. --size 0 keeps every eligible character (v1's vocab plus
      <unk>). --extra FILE makes more characters eligible and always keeps them, e.g.
      the frequent out-of-vocab characters nvsynth reports for the NVIDIA pretraining
      text: a text file (every non-space character) or a JSON {char: count} object
      (characters with count >= --extra-min).
  vocab.py coverage [--vocab F ...] [--json OUT]
      Character coverage and <unk> rate of each vocab on real VN dialogue text:
      hand-transcribed research lines, the agreed pseudo-labels in data/real, and
      eval's test manifests (eval/manifests/*.jsonl) when they exist; plus the
      synthetic validation shard's labels and the dialogue pool for reference.
      None of these texts is used for ranking.

The file format is v1's: one class per line, class i + 1 is line i, class 0 is the
CTC blank. A vocab whose last line is 〓 is an <unk> vocab: train.py maps every
character outside it to that class, so the model learns to say "unknown" instead
of a lookalike, and decoders print 〓 (never in a ground truth, so it scores as an
error).
"""
import argparse
import collections
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from common import DATA, is_test_vn, load_vocab  # noqa: E402

UNK = "〓"
POOL_WEIGHT = {"dialogue": 0.45, "narration": 0.35, "wiki": 0.20}   # synth/render.py
NAME_WEIGHT = 0.05
NAMES = DATA / "vndb_chars.jsonl"
TEST_GAMES = ROOT / "eval" / "manifests" / "test_games.json"
DIGITS = "0123456789０１２３４５６７８９"
RESEARCH = Path(os.environ.get("VOCR_RESEARCH",
                               "/local/home/skerraut/herd-vita-ocr/shared/research/scratch/real"))


def is_kana(c):
    o = ord(c)
    return 0x3041 <= o <= 0x3096 or 0x30A1 <= o <= 0x30FA or c in "ーヽヾゝゞ"


def rel_freq(text):
    counts = collections.Counter(text)
    for ws in "\n \u3000\t":
        counts.pop(ws, None)
    total = sum(counts.values()) or 1
    return {c: n / total for c, n in counts.items()}


def reserved_vns(path=TEST_GAMES):
    """eval's reserved test games (test_games.json "all"); is_test_vn() adds the rest."""
    if not Path(path).exists():
        raise SystemExit(f"{path} missing: the vocab ranking must exclude eval's test games")
    return set(json.loads(Path(path).read_text(encoding="utf-8"))["all"])


def load_names(path=NAMES, reserved=None):
    """(names, dropped): the original names of a 'vocab.py names' JSONL, without every
    character that appears in one of eval's test games (any of its VNs reserved)."""
    reserved = reserved_vns() if reserved is None else reserved
    names, dropped = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if not c.get("original"):
            continue
        if any(v in reserved or is_test_vn(v) for v in c.get("vns", [])):
            dropped += 1
            continue
        names.append(c["original"])
    return names, dropped


def vn_freq(corpus=DATA / "corpus", names=NAMES):
    """VN-weighted relative character frequency (see the module docstring)."""
    mix = collections.Counter()
    for name, weight in POOL_WEIGHT.items():
        for c, f in rel_freq((corpus / f"{name}.txt").read_text(encoding="utf-8")).items():
            mix[c] += (1 - NAME_WEIGHT) * weight * f
    if names.exists():
        kept, dropped = load_names(names)
        print(f"{names}: {len(kept)} character names ({dropped} of eval's test games left out)")
        for c, f in rel_freq("\n".join(kept)).items():
            mix[c] += NAME_WEIGHT * f
    else:
        print(f"warning: {names} missing (run 'vocab.py names'); ranking without names")
    return mix


def build(size, freq, full, extra=()):
    """size - 1 characters (forced set first, then by frequency) plus UNK, code-point
    sorted. size 0 keeps every character of full + extra."""
    from build_corpus import PUNCT  # lazy: build_corpus imports pyarrow
    extra = [c for c in extra if c != UNK and not c.isspace()]
    full = list(dict.fromkeys([*full, *extra]))
    keep = {c for c in full if is_kana(c) or c in PUNCT or c in DIGITS} | set(extra)
    if size and len(keep) > size - 1:
        raise ValueError(f"{len(keep)} forced characters do not fit in size {size}")
    for c, _ in sorted(((c, freq.get(c, 0.0)) for c in full if c not in keep),
                       key=lambda t: (-t[1], t[0])):
        if size and len(keep) >= size - 1:
            break
        keep.add(c)
    return sorted(keep) + [UNK]


def read_extra(path, min_count=1):
    """Characters of an --extra file: JSON {char: count} or plain text."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        counts = json.loads(text)
    except json.JSONDecodeError:
        return list(dict.fromkeys(c for c in text if not c.isspace()))
    return [c for c, n in sorted(counts.items(), key=lambda t: -t[1]) if len(c) == 1 and n >= min_count]


def fetch_names(out, min_votes=100, max_pages=400):
    """One JSON object per character: VNDB id, original name, ids of its VNs."""
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "vita-vn-ocr-vocab/0.1 (character-name kanji frequency)"
    chars, last = [], None
    for _ in range(max_pages):
        flt = ["vn", "=", ["and", ["olang", "=", "ja"], ["votecount", ">=", min_votes]]]
        if last:
            flt = ["and", flt, ["id", ">", last]]
        body = {"filters": flt, "fields": "original,vns.id", "sort": "id", "results": 100}
        r = s.post("https://api.vndb.org/kana/character", json=body, timeout=60)
        if r.status_code == 429:
            time.sleep(60)
            continue
        r.raise_for_status()
        d = r.json()
        chars += [{"id": x["id"], "original": x.get("original"),
                   "vns": [v["id"] for v in x.get("vns", [])]} for x in d["results"]]
        if not d["results"] or not d.get("more"):
            break
        last = d["results"][-1]["id"]
        time.sleep(1.6)                     # stays under 200 requests per 5 minutes
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chars),
                   encoding="utf-8")
    print(f"{out}: {len(chars)} characters, {sum(bool(c['original']) for c in chars)} with a name")


def real_texts():
    """{set name: [line text]} of real VN dialogue available locally."""
    sets = {}
    hand = []
    for path in (RESEARCH / "gt.json", RESEARCH / "gt_real.json"):
        if path.exists():
            for e in json.loads(path.read_text(encoding="utf-8")):
                for ln in e["lines"]:
                    if not isinstance(ln, dict):
                        hand.append(ln)
                    elif ln.get("confident", True):
                        hand.append(ln["text"])
    if hand:
        sets["research-hand"] = sorted(set(hand))
    lines = DATA / "real" / "lines.jsonl"
    if lines.exists():
        recs = [json.loads(x) for x in lines.read_text(encoding="utf-8").splitlines()]
        sets["real-agreed"] = [r["label"] for r in recs if r.get("label")]
    for path in sorted(glob.glob(str(ROOT / "eval" / "manifests" / "*.jsonl"))):
        texts = []
        for x in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            if r.get("text") and not r.get("uncertain"):
                texts.append(r["text"])
        if texts:
            sets["eval-" + Path(path).stem] = texts
    return sets


def coverage(vocab, texts):
    vs = set(vocab)
    chars = [c for t in texts for c in t if not c.isspace()]
    oov = collections.Counter(c for c in chars if c not in vs)
    lines_oov = sum(any(c not in vs for c in t if not c.isspace()) for t in texts)
    n = max(len(chars), 1)
    return {"lines": len(texts), "chars": len(chars), "unk_rate": sum(oov.values()) / n,
            "line_unk_rate": lines_oov / max(len(texts), 1),
            "oov": "".join(c for c, _ in oov.most_common(40))}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("names")
    n.add_argument("--out", type=Path, default=NAMES)
    b = sub.add_parser("build")
    b.add_argument("--size", type=int, required=True, help="entries incl. <unk>; 0 = all eligible")
    b.add_argument("--out", type=Path)
    b.add_argument("--extra", type=Path, help="more eligible, always-kept characters")
    b.add_argument("--extra-min", type=int, default=1)
    c = sub.add_parser("coverage")
    c.add_argument("--vocab", type=Path, nargs="+",
                   default=[DATA / "vocab_2500.txt", DATA / "vocab_3500.txt", DATA / "vocab.txt"])
    c.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.cmd == "names":
        fetch_names(args.out)
        return
    if args.cmd == "build":
        full = load_vocab()
        extra = read_extra(args.extra, args.extra_min) if args.extra else []
        vocab = build(args.size, vn_freq(), full, extra)
        out = args.out or DATA / f"vocab_{args.size or len(vocab)}.txt"
        out.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        kanji = sum(0x4E00 <= ord(ch) <= 0x9FFF for ch in vocab)
        print(f"{out}: {len(vocab)} entries ({kanji} kanji, {sum(map(is_kana, vocab))} kana, "
              f"<unk> last), {len(vocab) + 1} classes with the blank")
        return

    sets = real_texts()
    import numpy as np
    z = np.load(sorted(glob.glob(str(DATA / "synth" / "synth_*.npz")))[-1])
    sets["synth-val"] = json.loads(str(z["labels"]))
    dlg = (DATA / "corpus" / "dialogue.txt").read_text(encoding="utf-8").splitlines()
    sets["aozora-dialogue-pool"] = dlg[::20]
    report = {}
    for path in args.vocab:
        vocab = [v for v in load_vocab(path) if v != UNK]
        report[path.name] = {name: coverage(vocab, texts) for name, texts in sets.items()}
    for name in sets:
        row = "  ".join(f"{p}: {report[p][name]['unk_rate'] * 100:6.3f}% chars "
                        f"{report[p][name]['line_unk_rate'] * 100:5.1f}% lines" for p in report)
        print(f"{name:22s} ({report[next(iter(report))][name]['chars']:>8} chars)  {row}")
    for p in report:
        for name in sets:
            if not name.startswith(("synth", "aozora")):
                print(f"  {p} OOV in {name}: {report[p][name]['oov']}")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The dictionary-aware decoding prototype's word list (docs/vocr6-lm.md): JMdict's forms ranked by Jiten's global
frequency list, the two files Rubi_for_vita ships in its dictionary build (docs/plaintext-dictionaries.json:
sources/JMdict-plain.zip and sources/jiten_freq_global.zip, checked by their sha256).

  lexicon.py build --jmdict JMdict-plain.zip --jiten jiten_freq_global.zip --vocab V.txt [--vocab V2.txt]
                   --out LEX.tsv [--max-len 8]

Every term-bank entry's expression and reading (Yomitan format 3) is a candidate form; a form is kept when it is
2..max-len characters (synth3_text.norm_vn display form), every character in the vocabularies, and Jiten ranks it
(its term_meta_bank's value: the form as the term, either with or without a reading). LEX.tsv: form TAB best rank,
sorted by rank then form. JMdict (EDRDG) and Jiten are CC BY-SA 4.0.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))

PINS = {"JMdict-plain.zip": "14729ca303da7b9f603233934618369e7089cb3c6254649da5887d25463235c9",
        "jiten_freq_global.zip": "df0ab5c6971144e338ce0f3e9c738ff253333390a0f4d10f74d6361292f3bd40"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jiten_ranks(path):
    """{form: best rank} of a Jiten (Yomitan term_meta_bank) frequency zip."""
    best = {}
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if not n.startswith("term_meta_bank"):
                continue
            for term, kind, data in json.loads(z.read(n)):
                if kind != "freq":
                    continue
                v = data.get("value", (data.get("frequency") or {}).get("value"))
                if isinstance(v, int) and v < best.get(term, 1 << 30):
                    best[term] = v
    return best


def jmdict_forms(path):
    forms = set()
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if n.startswith("term_bank"):
                for e in json.loads(z.read(n)):
                    forms.add(e[0])
                    if e[1]:
                        forms.add(e[1])
    return forms


def build(jmdict, jiten, vocab_paths, max_len=8):
    from synth3_text import norm_vn
    vocab = set()
    for p in vocab_paths:
        vocab |= {c for c in Path(p).read_text(encoding="utf-8").split("\n") if c}
    ranks = jiten_ranks(jiten)
    out = {}
    for f in jmdict_forms(jmdict):
        r = ranks.get(f)
        w = norm_vn(f)
        if r is None or not 2 <= len(w) <= max_len or any(c not in vocab for c in w):
            continue
        if r < out.get(w, 1 << 30):
            out[w] = r
    return sorted(out.items(), key=lambda kv: (kv[1], kv[0]))


def load(path, max_rank=None):
    """[(form, rank)] of a LEX.tsv, those ranked at most max_rank."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        w, r = line.split("\t")
        if max_rank is None or int(r) <= max_rank:
            out.append((w, int(r)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--jmdict", required=True)
    p.add_argument("--jiten", required=True)
    p.add_argument("--vocab", action="append", required=True)
    p.add_argument("--max-len", type=int, default=8)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in (a.jmdict, a.jiten):
        if Path(f).name in PINS and sha256(f) != PINS[Path(f).name]:
            raise SystemExit(f"{f}: not the file Rubi pins")
    words = build(a.jmdict, a.jiten, a.vocab, a.max_len)
    Path(a.out).write_text("".join(f"{w}\t{r}\n" for w, r in words), encoding="utf-8")
    meta = {"words": len(words), "chars": sum(len(w) for w, _ in words), "sha256": sha256(a.out),
            "jmdict_sha256": sha256(a.jmdict), "jiten_sha256": sha256(a.jiten), "max_len": a.max_len}
    Path(a.out + ".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

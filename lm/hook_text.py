#!/usr/bin/env python3
"""The hooked corpus' visual lines (the owner's PS Vita / PSP captures, #125) as a text corpus for lm/build.py.

  hook_text.py export --root MIRROR --splits SPLITS.json --keep train|final --vocab V.txt [--vocab V2.txt]
                      --out TEXT.txt --heldout HVAL.txt --manifest TEXT.json [--test-vns TEST_VNS.txt]

MIRROR is the bucket's local copy (<platform>/<TITLE_ID> <title>/lines.jsonl.zst + metadata.json); SPLITS.json is
data/import_hooked.py's page split (titles[T].assign: page -> train | guard | hval | test).

Which lines (docs/vocr6-lm.md):
- only titles whose metadata.json says `train` (the owner's split): a `test` title, and any title that names
  CLANNAD, never contributes a line;
- `--keep train` (screening): only the pages the split assigns to `train`, so hval pages and their guard bands
  never reach the LM that is screened on them; `--keep final`: every page of a train title (train, guard, hval:
  the final LMs fold hval back in, like the final recognizers);
- line records of kind `line` (dialogue and narration; vertical columns included: their text is a line's text);
  speaker names and ruby readings are left out (dialogue CER is the metric; names are a title's own and repeat);
- the record's `text` (the training-label form, teacher/vote.canon), every character in the union of the --vocab
  files, at least one character;
- each title's distinct lines once, in page order: a line the capture saw on several pages (an NVL page that keeps
  its earlier lines, a scene captured twice) is counted once.

The output is one line per piece, titles in id order; --heldout gets the hval pages' lines the same way (the
screening set's text, for held-out cross-entropy); --test-vns the VNDB ids of the `test` titles (lm/build.py
--exclude-vns: their names leave the public corpora). The manifest holds counts and hashes, never text. Every file
written here holds the owner's private, copyrighted game text: it stays under $HERD and is never committed.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

CLANNAD = re.compile(r"clannad|クラナド|ＣＬＡＮＮＡＤ", re.I)
KINDS = ("line",)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def title_dirs(root):
    """[(title id, folder)] of every title folder of the mirror, sorted by title id."""
    out = []
    for plat in ("vita", "psp"):
        for d in sorted((Path(root) / plat).glob("*")):
            if (d / "metadata.json").is_file():
                out.append((d.name.split(" ")[0], d))
    return sorted(out)


def read_zst_jsonl(path):
    data = subprocess.run(["zstd", "-dc", str(path)], capture_output=True, check=True).stdout
    return [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]


def load_vocab(paths):
    chars = set()
    for p in paths:
        chars |= {c for c in Path(p).read_text(encoding="utf-8").split("\n") if c}
    return chars


def select(records, assign, keep, vocab):
    """(kept texts in page order, counts) of one title's line records under the page split."""
    pages = {"train": ("train",), "final": ("train", "guard", "hval"), "hval": ("hval",)}[keep]
    counts = {"records": 0, "kind": 0, "page": 0, "oov": 0, "empty": 0, "duplicate": 0, "kept": 0, "chars": 0}
    seen, out = set(), []
    for r in sorted(records, key=lambda r: (int(r["page"]), r["id"])):
        counts["records"] += 1
        if r.get("kind") not in KINDS:
            counts["kind"] += 1
            continue
        if assign.get(r["page"]) not in pages:
            counts["page"] += 1
            continue
        t = r.get("text") or ""
        if not t:
            counts["empty"] += 1
            continue
        if any(c not in vocab for c in t):
            counts["oov"] += 1
            continue
        if t in seen:
            counts["duplicate"] += 1
            continue
        seen.add(t)
        out.append(t)
        counts["kept"] += 1
        counts["chars"] += len(t)
    return out, counts


def cmd_export(a):
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    vocab = load_vocab(a.vocab)
    pieces, held, titles, test_vns = [], [], {}, set()
    for tid, d in title_dirs(a.root):
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        vn = (meta.get("vndb") or {}).get("vn")
        if meta.get("split") != "train" or CLANNAD.search(meta.get("title", "")) or CLANNAD.search(d.name):
            titles[tid] = {"split": meta.get("split"), "used": False}
            if vn:
                test_vns.add(vn)
            test_vns |= set((meta.get("vndb") or {}).get("candidates") or [])
            continue
        if tid not in splits["titles"]:
            raise SystemExit(f"{tid}: not in {a.splits}")
        st = splits["titles"][tid]
        if st["title_split"] != "train":
            raise SystemExit(f"{tid}: the split file calls it {st['title_split']}, metadata.json train")
        recs = read_zst_jsonl(d / "lines.jsonl.zst")
        got, counts = select(recs, st["assign"], a.keep, vocab)
        hv, hcounts = select(recs, st["assign"], "hval", vocab)
        pieces += got
        held += hv
        titles[tid] = {"split": "train", "used": True, "platform": meta.get("platform"), "vndb": vn,
                       "lines_sha256": sha256(d / "lines.jsonl.zst"), "counts": counts, "hval": hcounts}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(t + "\n" for t in pieces), encoding="utf-8")
    Path(a.heldout).write_text("".join(t + "\n" for t in held), encoding="utf-8")
    if a.test_vns:
        Path(a.test_vns).write_text("".join(v + "\n" for v in sorted(test_vns)), encoding="utf-8")
    man = {"kind": "vocr6-lm-hook-text", "schema": 1, "keep": a.keep, "kinds": list(KINDS),
           "splits": {"file": Path(a.splits).name, "sha256": sha256(a.splits),
                      "params": splits.get("params")},
           "vocab_sha256": [sha256(p) for p in a.vocab], "titles": titles,
           "pieces": len(pieces), "chars": sum(map(len, pieces)), "sha256": sha256(out),
           "heldout": {"pieces": len(held), "chars": sum(map(len, held)), "sha256": sha256(a.heldout)},
           "test_vns": sorted(test_vns)}
    Path(a.manifest).write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("keep", "pieces", "chars", "sha256")} | {"heldout": man["heldout"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("--root", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--keep", choices=("train", "final"), required=True)
    p.add_argument("--vocab", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--test-vns")
    a = ap.parse_args()
    {"export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()

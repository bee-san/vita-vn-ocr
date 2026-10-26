#!/usr/bin/env python3
"""What a VOLM blob (lm/vlm.py) keeps of the game text it was built from (docs/vocr6-lm.md, "Memorisation").

  memorization.py report --lm NAME=B.vlm [--lm NAME=A.vlm ...] --train TRAIN.txt --heldout HELD.txt
                         --public CORPUS.txt [--public ...] --out MEMO.json

For each blob:
- its order, entries per order and pruning thresholds (from the build manifest next to it, NAME.json);
- game-only n-grams: the kept n-grams of order 2..N (tokens, with <s> and </s>) that occur in the game text
  (TRAIN.txt, one line a piece, <s> line </s>) and nowhere in the public corpora (--public, the same tokenisation):
  the fragments of game script the blob holds that public text does not explain;
- verbatim recovery: from every position of every line, the LM's own most probable next character (its argmax,
  from the line's context as the decoder's state holds it: <s> then the line's characters) is compared with the
  line's next character, again and again: the run of characters it reproduces before the first miss. Reported:
  the longest run over every start (the longest verbatim span greedy continuation recovers), the distribution of
  each line's longest run, and the share of characters predicted exactly as the next one. On TRAIN.txt (the text
  the blob was built from) and on HELD.txt (lines it never saw: held-out lines of the same games) - the gap
  between the two is what the blob memorised rather than generalised.
The JSON holds counts, never text. The tie of an argmax goes to the lower token.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Next:
    """The most probable next token of a VOLM state, exactly as LM.score ranks tokens: the candidates are every
    child of the state's longest context, of each shorter one, and the best unigrams; the probability of each is
    LM.score's (backoff weights included)."""

    def __init__(self, lm):
        self.lm = lm
        p1 = lm.cbp[0][lm.prob[0]].copy()
        p1[:vlm.FIRST] = np.iinfo(np.int64).min
        p1[vlm.EOS] = lm.cbp[0][lm.prob[0][vlm.EOS]]
        self.uni = np.argsort(-p1, kind="stable")       # unigram order (ties: the lower token)
        self.cache = {}

    def argmax(self, state):
        if state in self.cache:
            return self.cache[state]
        lm = self.lm
        cand = set()
        for k in range(1, lm.order):
            e = state[k - 1]
            if e != vlm.NONE:
                lo, hi = int(lm.child[k - 1][e]), int(lm.child[k - 1][e + 1])
                cand.update(int(t) for t in lm.tok[k][lo:hi])
        cand.update(int(t) for t in self.uni[:len(cand) + 1])
        best, bt = None, None
        for t in sorted(cand):
            if t < vlm.FIRST and t != vlm.EOS:
                continue
            lp, _ = lm.score(state, t)
            if best is None or lp > best:
                best, bt = lp, t
        self.cache[state] = bt
        return bt


def recovery(lm, lines):
    """Per line, the longest run of characters reproduced by greedy continuation from any start; and the share of
    characters (with </s>) that are the argmax of their context."""
    nx = Next(lm)
    longest, hit, n = [], 0, 0
    for s in lines:
        toks = [lm.token_of(c) for c in s] + [vlm.EOS]
        if vlm.UNK in toks:
            continue
        states = [lm.initial(vlm.FLAG_BOS)]
        for t in toks[:-1]:
            states.append(lm.score(states[-1], t)[1])
        pred = [nx.argmax(st) for st in states]          # the argmax after each prefix
        ok = [p == t for p, t in zip(pred, toks)]
        hit += sum(ok)
        n += len(toks)
        best, i = 0, 0
        while i < len(toks):                               # runs of greedy continuation from each start
            if not ok[i]:
                i += 1
                continue
            j = i
            st = states[i]
            while j < len(toks):                            # continue on the LM's own text (= the line's here)
                if nx.argmax(st) != toks[j]:
                    break
                st = lm.score(st, toks[j])[1]
                j += 1
            best = max(best, j - i)
            i += 1
        longest.append(best)
    a = np.array(longest)
    return {"lines": int(len(a)), "chars_with_eos": int(n), "next_char_exact_share": round(hit / max(n, 1), 4),
            "longest_run": {"max": int(a.max()) if len(a) else 0,
                            "p50": float(np.percentile(a, 50)) if len(a) else 0,
                            "p90": float(np.percentile(a, 90)) if len(a) else 0,
                            "p99": float(np.percentile(a, 99)) if len(a) else 0,
                            "lines_ge_8": int((a >= 8).sum()), "lines_ge_12": int((a >= 12).sum()),
                            "lines_ge_20": int((a >= 20).sum())},
            "longest_run_by_length": {str(k): int((a == k).sum()) for k in range(0, int(a.max()) + 1)} if len(a)
            else {}}


BOUND = {vlm.BOS: "<s>", vlm.EOS: "</s>", vlm.UNK: "<unk>"}


def as_chars(lm, gram):
    """A token tuple as a tuple of characters (<s>, </s>, <unk> as such): comparable across vocabularies."""
    return tuple(BOUND.get(t) or chr(int(lm.codepoints[t - vlm.FIRST])) for t in gram)


def kept_ngrams(lm):
    """{order k: set of token tuples} of the blob's entries, k = 1..N."""
    out = {1: {(t,) for t in range(lm.V)}}
    ent = {1: [(t,) for t in range(lm.V)]}
    for k in range(2, lm.order + 1):
        prev, cur = ent[k - 1], []
        child = lm.child[k - 2]
        for e, g in enumerate(prev):
            lo, hi = int(child[e]), int(child[e + 1])
            cur.extend(g + (int(t),) for t in lm.tok[k - 1][lo:hi])
        ent[k] = cur
        out[k] = set(cur)
    return out


def text_ngrams(paths, order):
    """{k: set of character tuples} of every n-gram (k = 2..order) inside <s> line </s> of the files."""
    out = {k: set() for k in range(2, order + 1)}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if not s:
                    continue
                t = ["<s>", *s, "</s>"]
                for k in range(2, order + 1):
                    out[k].update(tuple(t[i:i + k]) for i in range(len(t) - k + 1))
    return out


def cmd_report(a):
    train = [s for s in Path(a.train).read_text(encoding="utf-8").split("\n") if s]
    held = [s for s in Path(a.heldout).read_text(encoding="utf-8").split("\n") if s]
    if a.limit:
        train, held = train[:a.limit], held[:a.limit]
    out = {"kind": "volm-memorisation", "train_lines": len(train), "heldout_lines": len(held),
           "train_sha256": sha256(a.train), "heldout_sha256": sha256(a.heldout),
           "public_sha256": [sha256(p) for p in a.public], "lms": {}}
    pub = game = None
    order = max(vlm.LM(Path(spec.partition("=")[2]).read_bytes()).order for spec in a.lm)
    for spec in a.lm:
        name, _, path = spec.partition("=")
        lm = vlm.LM(Path(path).read_bytes())
        man = Path(path).with_suffix(".json")
        info = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
        row = {"sha256": sha256(path), "bytes": Path(path).stat().st_size, "order": lm.order, "entries": lm.n,
               "thresholds": info.get("chosen_thresholds"),
               "game_text": any(c.get("visual_lines") for c in info.get("corpora", []))}
        if pub is None:
            pub = text_ngrams(a.public, order)
            game = text_ngrams([a.train], order)
        kept = {k: {as_chars(lm, g) for g in v} for k, v in kept_ngrams(lm).items()}
        row["game_only_ngrams"] = {str(k): {"kept": len(kept[k]),
                                            "in_game_text_not_public": len((kept[k] & game[k]) - pub[k])}
                                   for k in range(2, lm.order + 1)}
        row["recovery_train"] = recovery(lm, train)
        row["recovery_heldout"] = recovery(lm, held)
        out["lms"][name] = row
        print(name, json.dumps({k: row[k] for k in ("order", "entries", "thresholds", "game_only_ngrams")}),
              json.dumps(row["recovery_train"]["longest_run"]), json.dumps(row["recovery_heldout"]["longest_run"]),
              flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--lm", action="append", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--public", action="append", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

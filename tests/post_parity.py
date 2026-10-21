#!/usr/bin/env python3
"""Parity of runtime/vocr_post.c with its Python twin eval/vocr_post.py (issue #109).

    post_parity.py --bin runtime/build/post/vocr_post_test [--rules 3000] [--lines 2000] [--seed 0]

Random texts through the counter rule, and random per-frame top-K lists through the re-decode (lines built
around X + ッ/ヶ/ヵ + 月/所/年/国/条 contexts, with brackets, numerals, blanks and look-alike letters as the
alternatives, every K from 1 to 16 and thresholds from 0 to the maximum), through the C driver
(vocr_post_test --parity) and the twin: every count, text and edited class must be the same.
"""
import argparse
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import vocr_post as P  # noqa: E402

RULE_ALPHABET = list("ッヶヵケっツ月所年国条日村町一二三九十百何数幾〇3５\nルあ「」谷ベドー")
RULE_LEFT = list("一二三四五六七八九十百千万〇0123456789０９何数幾ルえ為別ー\n「ッ")
RULE_RIGHT = list("月年所国条日村町谷\nッヶ")
LETTERS = list("ルニーしハトロ口ミセナ大王八ヤ了乙カケ巾九二三一十七")
TAILS = list("ですまかがのはにをとも」』、。！")
THRESHOLDS = [0, 1, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 16000, 64000]


def vocab_classes():
    lines = (ROOT / "data" / "vocab.txt").read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [""] + lines                                   # class 0: the blank


def make_line(rng, idx, K):
    """(frames x K logits, classes) for one random line around a counter context."""
    seq = []
    if rng.random() < 0.3:
        seq.append("「")
    if rng.random() < 0.2:
        seq.append(rng.choice(LETTERS))
    seq.append(rng.choice(LETTERS))
    seq.append(rng.choice(["ッ", "ヶ", "ヵ", "ケ", "ッ"]))
    seq.append(rng.choice(list("月所年国条日")))
    seq += [rng.choice(TAILS) for _ in range(rng.randint(0, 4))]
    if rng.random() < 0.5:
        seq.append(rng.choice(["」", "』", "」"]))
    if rng.random() < 0.15:                               # a second context on the same line
        seq += [rng.choice(LETTERS), rng.choice(["ッ", "ヶ"]), rng.choice(list("月年所"))]
    alts = ["", "「", "『", "」", "ル", "九", "一", "二", "三", "八", "十", "何", "数", "ッ", "ヶ", "ヵ", "ー", "ニ",
            "し", "で", "す"]
    frames = []
    for i, ch in enumerate(seq):
        for _ in range(rng.randint(0, 2)):                # blanks before the character
            frames.append("")
        for _ in range(rng.randint(1, 3)):
            frames.append(ch)
    frames += [""] * rng.randint(0, 2)
    T = len(frames)
    tv = np.zeros((T, K), np.float32)
    tc = np.zeros((T, K), np.uint16)
    for f, ch in enumerate(frames):
        top = idx[ch]
        pool = [idx[a] for a in alts if idx[a] != top]
        rng.shuffle(pool)
        others = pool[:K - 1]
        while len(others) < K - 1:                        # K above the pool: random classes
            c = rng.randrange(1, len(idx))
            if c != top and c not in others:
                others.append(c)
        v0 = np.float32(rng.uniform(5, 15))
        vals = sorted((np.float32(v0 - rng.choice([rng.uniform(0, 8), rng.uniform(0, 1), 0.0]))
                       for _ in range(K - 1)), reverse=True)
        tc[f, 0], tv[f, 0] = top, v0
        for j in range(1, K):
            tc[f, j], tv[f, j] = others[j - 1], vals[j - 1]
        # ties keep the lower class first, as vocr_run() orders them
        for j in range(1, K):
            if tv[f, j] == tv[f, 0] and tc[f, j] < tc[f, 0]:
                tv[f, j] = np.nextafter(tv[f, 0], np.float32(-np.inf))
        order = sorted(range(K), key=lambda j: (-float(tv[f, j]), int(tc[f, j])))
        tv[f], tc[f] = tv[f, order], tc[f, order]
    return tv, tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--rules", type=int, default=3000)
    ap.add_argument("--lines", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    vocab = vocab_classes()
    idx = {}
    for i, t in enumerate(vocab):
        idx.setdefault(t, i)
    missing = [c for c in "「』」『ルニーし九一二三八十何数ッヶヵ月所年国条日でま" if c not in idx]
    if missing:
        sys.exit(f"vocab lacks {missing}")
    cases, expect = [], []
    fixed = ["ルッ月です」", "一ッ所", "三ッ所", "えッ国上山", "九ッ\n月", "ッ", "ッ月", ""]
    for i in range(a.rules):
        if i < len(fixed):
            t = fixed[i]
        elif i % 2:                                       # a context: left + ッ + right, in noise
            t = "".join(rng.choice(RULE_ALPHABET) for _ in range(rng.randint(0, 3))) + rng.choice(RULE_LEFT) + \
                "ッ" + rng.choice(RULE_RIGHT) + "".join(rng.choice(RULE_ALPHABET) for _ in range(rng.randint(0, 3)))
        else:
            t = "".join(rng.choice(RULE_ALPHABET) for _ in range(rng.randint(1, 12)))
        b = t.encode("utf-8")
        cases.append(struct.pack("<II", 0, len(b)) + b)
        out, n = P.counters(t)
        expect.append(f"R {n} {out.encode('utf-8').hex() or '-'}")
    letters = list(LETTERS)
    for i in range(a.lines):
        K = rng.choice([1, 2, 3, 4, 6, 8, 8, 8, 12, 16])
        tv, tc = make_line(rng, idx, K)
        tau, opn = rng.choice(THRESHOLDS), rng.choice(THRESHOLDS)
        if rng.random() < 0.2:
            tau = rng.randint(0, 64000)
        cases.append(struct.pack("<IIIii", 1, tv.shape[0], K, tau, opn) + tv.tobytes() + tc.tobytes())
        tc2, st = P.redecode(vocab, tv, tc, tau, opn)
        txt = P.greedy(vocab, tc2).encode("utf-8").hex() or "-"
        expect.append(f"D {st['edits']} {st['contexts']} {st['numerals']} {st['brackets']} {txt} "
                      + " ".join(str(int(c)) for c in tc2.reshape(-1)))
    del letters
    with tempfile.TemporaryDirectory() as d:
        vp, cp = Path(d) / "vocab.txt", Path(d) / "cases.bin"
        vp.write_text("\n".join(vocab[1:]) + "\n", encoding="utf-8")
        cp.write_bytes(struct.pack("<I", len(cases)) + b"".join(cases))
        res = subprocess.run(a.bin.split() + ["--parity", str(vp), str(cp)], capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"driver failed rc={res.returncode}: {res.stderr[-2000:]}")
    got = res.stdout.splitlines()
    if len(got) != len(expect):
        sys.exit(f"{len(got)} results for {len(expect)} cases")
    bad = [i for i, (g, e) in enumerate(zip(got, expect)) if g != e]
    edits = sum(1 for e in expect if e.startswith("D ") and e.split()[1] != "0")
    rules = sum(1 for e in expect if e.startswith("R ") and e.split()[1] != "0")
    if bad:
        for i in bad[:5]:
            print("case", i, "\n  C:   ", got[i][:300], "\n  twin:", expect[i][:300], file=sys.stderr)
        sys.exit(f"post parity: {len(bad)} of {len(expect)} cases differ")
    print(f"post parity: {a.rules} rule cases ({rules} changed) and {a.lines} re-decode lines ({edits} edited): "
          "C == twin on every case")


if __name__ == "__main__":
    main()

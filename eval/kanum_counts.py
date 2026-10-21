#!/usr/bin/env python3
"""Counter-context counts behind runtime/vocr_post.c's text rule (issue #109, docs/counters.md): how often ッ,
and the counter forms ヶ ヵ ケ 箇 (and か カ っ ツ for reference), stand before 月 所 年 国 条 村 町 日 and the other
characters ヶ/ヵ precede, split by the character on their left, with up to 400 contexts of every ッ hit kept for
inspection. Training-side text only: Aozora Bunko and the Japanese Wikipedia shard data/fetch_assets.py fetched,
synth3's train-split text pools other than its Aozora and Wikipedia ones (the same text), and the real label
pools (teacher, realdata, realplus v3, native-frames, owner pool v2) with every reserved test VN and every
dev_sample VN (eval/manifests/reserved.json) removed. No eval set, no Jiten list.

    kanum_counts.py sources                      the inputs it would read
    kanum_counts.py count OUT.json [--procs 5]   (heavy: about 15 seconds in 5 processes)

The output stays local (it holds text contexts); docs/counters.md has the counts.
"""
import argparse
import collections
import glob
import gzip
import json
import multiprocessing as mp
import os
import re
import sys

HERD = os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")
A = f"{HERD}/shared/artifacts"
D = f"{HERD}/shared/data/corpus"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_KANJI = set("一二三四五六七八九十百千万〇零両廿卅")
NUM_DIGIT = set("0123456789０１２３４５６７８９")
QUANT = set("何数幾")
MIDS_ALL = "ッヶヵ"            # every right character counted
MIDS_REF = "ケ箇っツかカ"       # only before the counter set
COUNTER = "月所年国条村町日"
RE_ALL = re.compile(f"[{MIDS_ALL}]")
RE_REF = re.compile(f"[{MIDS_REF}][{COUNTER}]")
CTX_RIGHT = set(COUNTER) | {"EOS"}
CTX_KEEP = 400                  # contexts kept per (source, mid=ッ, right, left class)


def lclass(ch):
    if ch is None:
        return "BOS"
    if ch in NUM_KANJI:
        return "numk"
    if ch in NUM_DIGIT:
        return "numd"
    if ch in QUANT:
        return "quant"
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or ch == "々":
        return "kanji"
    if 0x30A1 <= o <= 0x30FA or ch == "ー":
        return "kata"
    if 0x3041 <= o <= 0x3096:
        return "hira"
    return "other"


def reserved():
    r = json.load(open(f"{REPO}/eval/manifests/reserved.json"))
    return set(r["test"]), set(r["dev_sample"])


def texts(src):
    """Yield (text) units of one source."""
    test, dev = reserved()
    drop = test | dev
    kind, path = src
    if kind == "aozora":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]
    elif kind == "wiki":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=2000, columns=["text"]):
            for t in b.column(0).to_pylist():
                yield t
    elif kind == "pool":
        with open(path, encoding="utf-8") as f:
            buf = []
            for line in f:
                buf.append(line)
                if len(buf) >= 20000:
                    yield "".join(buf)
                    buf = []
            if buf:
                yield "".join(buf)
    elif kind == "labels":
        # jsonl with a label/text field and a vn field (or an id 'v123_...')
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                vn = r.get("vn") or str(r.get("id", "")).split("_")[0]
                if vn in drop:
                    continue
                if r.get("status") not in (None, "kept"):
                    continue
                t = r.get("label") if r.get("label") is not None else r.get("text")
                if isinstance(t, str) and t:
                    yield t + "\n"


def count_src(src):
    name = src[0] + ":" + os.path.basename(src[1])
    cnt = collections.Counter()      # (lclass, mid, right) -> n
    numl = collections.Counter()     # (leftchar, mid, right) for numeral / quant lefts
    ctx = collections.defaultdict(list)
    chars = 0
    for t in texts(src):
        chars += len(t)
        for m in RE_ALL.finditer(t):
            i = m.start()
            mid = t[i]
            right = t[i + 1] if i + 1 < len(t) else "EOS"
            left = t[i - 1] if i > 0 else None
            lc = lclass(left)
            cnt[(lc, mid, right)] += 1
            if lc in ("numk", "numd", "quant"):
                numl[(left, mid, right)] += 1
            if mid == "ッ" and (right in CTX_RIGHT or lc in ("numk", "numd", "quant")):
                k = (mid, right, lc)
                if len(ctx[k]) < CTX_KEEP:
                    ctx[k].append(t[max(0, i - 12):i + 13].replace("\n", "⏎"))
        for m in RE_REF.finditer(t):
            i = m.start()
            left = t[i - 1] if i > 0 else None
            lc = lclass(left)
            cnt[(lc, t[i], t[i + 1])] += 1
            if lc in ("numk", "numd", "quant"):
                numl[(left, t[i], t[i + 1])] += 1
    return {"source": name, "chars": chars,
            "counts": [[*k, v] for k, v in cnt.items()],
            "numleft": [[*k, v] for k, v in numl.items()],
            "contexts": [[*k, v] for k, v in ctx.items()]}


def sources():
    out = [("aozora", f"{D}/aozora/aozorabunko-dedupe-clean.jsonl.gz")]
    out += [("wiki", p) for p in sorted(glob.glob(f"{D}/wiki/20231101.ja/*.parquet"))]
    for p in sorted(glob.glob(f"{A}/synth3/text/pools3/*.train.txt")):
        b = os.path.basename(p)
        if b.startswith(("aozora_", "wiki.")):
            continue                     # the same text as the aozora / wiki sources above
        out.append(("pool", p))
    for p in sorted(glob.glob(f"{REPO}/teacher/manifests/lines_*.jsonl")):
        out.append(("labels", p))
    for p in sorted(glob.glob(f"{REPO}/realdata/manifests/lines_*.jsonl")):
        out.append(("labels", p))
    out.append(("labels", f"{A}/realplus-v3/labels_v3i9.jsonl"))
    out.append(("labels", f"{A}/native-frames/pool/lines_full_v1.jsonl"))
    out.append(("labels", f"{HERD}/shared/private/owner-games/_build2/owner-pool-2/lines_full_v2.jsonl"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "sources"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    srcs = sources()
    for s in srcs:
        if not os.path.exists(s[1]):
            sys.exit(f"missing {s}")
    if a.cmd == "sources":
        for s in srcs:
            print(*s)
        return
    with mp.Pool(a.procs) as pool:
        res = pool.map(count_src, sorted(srcs, key=lambda s: -os.path.getsize(s[1])), chunksize=1)
    json.dump({"sources": res}, open(a.out, "w"), ensure_ascii=False)
    print("wrote", a.out, sum(r["chars"] for r in res), "chars")


if __name__ == "__main__":
    main()

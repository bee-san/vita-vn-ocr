#!/usr/bin/env python3
"""synth3 text pools: VN-like dialogue, narration and names, split by source document.

  synth3_text.py pools  --raw RAW --aozora AOZORA.jsonl.gz --wiki WIKI.txt --out POOLS [--workers 6]
  synth3_text.py names  --vndb-chars vndb_chars.tsv --syosetu syosetu-names.parquet \
                        --dev-vns dev_vns.txt --out POOLS
  synth3_text.py index  --pools POOLS            # char -> sampled contexts (focus-character control)
  synth3_text.py targets --pools POOLS --real REAL.npz[+REAL2.npz] --jiten-vn VN.csv --mining M.json

Every source is read from a pinned local copy (data/manifests/synth3_text_sources.json lists URL,
sha256, licence class); nothing is downloaded here. A line is kept only if every character is in
data/vocab.txt (whole lines are rejected, characters are never deleted), it has Japanese script
and at most MAX_LEN characters. Train/dev assignment is by source document (Aozora book id,
Tatoeba sentence id, subtitle block, roleplay conversation, wiki page), hashed with SPLIT_SALT:
1 in 20 documents is dev. Train lines that also occur in any dev pool are removed, so the
synthetic dev shards share no line with training. Names exclude every character that appears in
a reserved test VN (eval/manifests/test_games.json "all", synth/common.is_test_vn) or a
development VN (--dev-vns), by VNDB VN id.

Output (POOLS/): <pool>.<split>.txt (one line per utterance), <pool>.<split>.json (counts,
sha256, licence class), names.<split>.tsv, contexts.<split>.npz, targets.json.
"""
import argparse
import bz2
import glob
import gzip
import hashlib
import html
import json
import multiprocessing as mp
import re
import sys
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import DATA, is_test_vn, load_vocab  # noqa: E402

MAX_LEN = 60          # one utterance; render-time wrapping makes 1-4 line boxes from it
SPLIT_SALT = "synth3-text-split-v1"
DEV_MOD = 20
TRAIN_CAP = 400_000   # lines per pool and split (deterministic hash subsample)
DEV_CAP = 20_000

# licence classes: open = redistributable with attribution; nc = non-commercial; grey = unclear
SOURCES = {
    "aozora_dialogue": {"licence": "CC-BY-4.0 dataset / public-domain texts", "class": "open", "kind": "dialogue"},
    "aozora_narration": {"licence": "CC-BY-4.0 dataset / public-domain texts", "class": "open", "kind": "narration"},
    "wiki": {"licence": "CC-BY-SA-4.0", "class": "open", "kind": "narration"},
    "tatoeba": {"licence": "CC-BY-2.0-FR", "class": "open", "kind": "dialogue"},
    "jesc": {"licence": "CC-BY-SA-4.0", "class": "open", "kind": "dialogue"},
    "opensubtitles": {"licence": "none stated (OPUS attribution)", "class": "grey", "kind": "dialogue"},
    "jdd": {"licence": "CC-BY-NC-ND-4.0", "class": "nc", "kind": "dialogue"},
    "bsd": {"licence": "CC-BY-NC-SA-4.0", "class": "nc", "kind": "dialogue"},
    "nucc": {"licence": "CC-BY-NC-ND-4.0", "class": "nc", "kind": "dialogue"},
    "aratako_dsv3": {"licence": "MIT", "class": "open", "kind": "dialogue"},
    "aratako_gpt4omini": {"licence": "CC-BY-NC-SA-4.0 + OpenAI terms", "class": "nc", "kind": "dialogue"},
    "narikiri": {"licence": "Apache-2.0 as labelled (forum scrape)", "class": "grey", "kind": "dialogue"},
    "wikiquote": {"licence": "CC-BY-SA-4.0", "class": "open", "kind": "dialogue"},
    "wiktionary": {"licence": "CC-BY-SA-4.0", "class": "open", "kind": "dialogue"},
}

JP = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\u3005]")
HALF_KANA = re.compile(r"[\uff61-\uff9f]+")
SPACE = re.compile(r"[\s\u3000]+")
ASCII_DOTS = re.compile(r"\.{2,}|・{3,}")
COMMA = re.compile(r"(?<![0-9]),|,(?![0-9])")
TRANS = str.maketrans({"!": "！", "?": "？", "~": "～", "〜": "～", "—": "―", "–": "―", "─": "―",
                       "‐": "―", "(": "（", ")": "）", "\u00a0": ""})
SENT = re.compile(r"(?<=[。！？…])(?=[^。！？…」』）])")
QUOTE = re.compile(r"「[^「」]{1,200}」|『[^『』]{1,200}』")


def norm_vn(s):
    """Display form of VN text: NFC, full-width kana and ！？～（）, one … per ellipsis cell,
    ― for dashes, no spaces or control characters."""
    s = unicodedata.normalize("NFC", s or "")
    s = HALF_KANA.sub(lambda m: unicodedata.normalize("NFKC", m.group()), s)
    s = SPACE.sub("", s)
    s = ASCII_DOTS.sub(lambda m: "…" * max(1, len(m.group()) // 3), s)
    s = re.sub("…{4,}", "………", s)
    s = COMMA.sub("、", s).translate(TRANS)
    s = re.sub("―{3,}", "――", s)
    return "".join(c for c in s if not unicodedata.category(c).startswith("C"))


def keep(s, vocab):
    if not s or len(s) > MAX_LEN or not JP.search(s):
        return False
    if any(c not in vocab for c in s):
        return False
    return sum(bool(JP.match(c)) for c in s) >= 0.5 * len(s)


def sentences(text):
    """Split one utterance into sentence pieces of at most MAX_LEN characters."""
    text = text.strip()
    if len(text) <= MAX_LEN:
        return [text] if text else []
    out, cur = [], ""
    for part in SENT.split(text):
        if cur and len(cur) + len(part) > MAX_LEN:
            out.append(cur)
            cur = ""
        cur += part
        while len(cur) > MAX_LEN:
            cut = max(cur.rfind("、", 0, MAX_LEN), 20)
            out.append(cur[:cut + 1])
            cur = cur[cut + 1:]
    if cur:
        out.append(cur)
    return out


def split_of(source, doc):
    h = int(hashlib.sha256(f"{SPLIT_SALT}:{source}:{doc}".encode()).hexdigest()[:16], 16)
    return "dev" if h % DEV_MOD == 0 else "train"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- sources

def _aozora(path, want):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta = d.get("meta") or {}
            if meta.get("文字遣い種別") != "新字新仮名":
                continue
            book = meta.get("作品ID")
            for par in (d.get("text") or "").split("\n"):
                par = par.strip()
                if not par:
                    continue
                if want == "dialogue":
                    for q in QUOTE.findall(par):
                        yield book, q          # with its brackets: VN speech is shown in 「」
                else:
                    rest = QUOTE.sub("", par) if par[0] not in "「『" else ""
                    if rest:
                        yield book, rest


def src_aozora_dialogue(a):
    yield from _aozora(a.aozora, "dialogue")


def src_aozora_narration(a):
    yield from _aozora(a.aozora, "narration")


def src_wiki(a):
    with open(a.wiki, encoding="utf-8") as f:
        for i, line in enumerate(f):
            yield i // 50, line.strip()


def src_tatoeba(a):
    with bz2.open(Path(a.raw) / "tatoeba/jpn_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                yield p[0], p[2]


def src_jesc(a):
    with tarfile.open(Path(a.raw) / "jesc/raw.tar.gz") as t:
        member = next(m for m in t.getmembers() if m.isfile() and m.name.endswith("raw"))
        f = t.extractfile(member)
        for i, line in enumerate(f):
            p = line.decode("utf-8", "replace").rstrip("\n").split("\t")
            if len(p) >= 2:
                yield i // 200, p[1]


def src_opensubtitles(a):
    with gzip.open(Path(a.raw) / "opensubtitles/ja.txt.gz", "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                continue
            yield i // 200, s.lstrip("-−‐ ")


def src_jdd(a):
    for fn in sorted(glob.glob(str(Path(a.raw) / "jdd/data/topic*.json"))):
        for k, d in enumerate(json.load(open(fn, encoding="utf-8"))):
            for u in d["utterances"]:
                yield f"{Path(fn).name}:{k}", u["utterance"]


def src_bsd(a):
    for fn in ("train.json", "dev.json", "test.json"):
        for doc in json.load(open(Path(a.raw) / "bsd" / fn, encoding="utf-8")):
            for s in doc["conversation"]:
                yield doc.get("id", fn), s["ja_sentence"]


def src_nucc(a):
    import io
    import zipfile
    tag = re.compile(r"^[FMX]\d{3}：")
    with zipfile.ZipFile(Path(a.raw) / "nucc/nucc.zip") as z:
        for name in sorted(n for n in z.namelist() if re.search(r"data\d+\.txt$", n)):
            text = io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace")
            for line in text:
                s = line.strip()
                if not s or s[0] in "＠％":
                    continue
                s = tag.sub("", s)
                s = re.sub(r"＜[^＞]*＞", "", s)
                s = re.sub(r"（[^）]*）", "", s).replace("＊", "")
                if len(s) >= 2:
                    yield name, s


QUOTED = re.compile(r"「([^「」]{1,200})」")


def _turns(contents):
    for c in contents:
        c = c or ""
        qs = QUOTED.findall(c)
        if qs:
            for q in qs:
                yield "「" + q + "」"
            continue
        for ln in c.split("\n"):
            ln = re.sub(r"^\s*[^\s：:「（(]{1,12}[：:]\s*", "", ln)
            ln = re.sub(r"^\[[^\]]*\]\s*", "", ln)
            ln = re.sub(r"[（(][^）)]*[）)]", "", ln).strip("*＊ ")
            if ln:
                yield ln


def src_aratako_dsv3(a):
    import pyarrow.parquet as pq
    p = sorted(glob.glob(str(Path(a.raw) / "aratako_sfw_dsv3/**/*.parquet"), recursive=True))
    for f in p:
        t = pq.read_table(f, columns=["conversations"])
        for k, conv in enumerate(t.column("conversations").to_pylist()):
            for s in _turns(m["content"] for m in conv):
                yield f"{Path(f).name}:{k}", s


def src_aratako_gpt4omini(a):
    for fn in sorted(glob.glob(str(Path(a.raw) / "aratako_gpt4omini/*.jsonl"))):
        for k, line in enumerate(open(fn, encoding="utf-8")):
            r = json.loads(line)
            for s in _turns(m["content"] for m in r["conversations"]):
                yield f"{Path(fn).name}:{k}", s


def src_narikiri(a):
    path = Path(a.raw) / "narikiri_omniai/Japanese-Roleplay-Dialogues-Filtered.jsonl"
    for k, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        for s in _turns(p["post_content"] for p in r["posts"]):
            yield k, s


def _wiki_pages(path):
    import xml.etree.ElementTree as ET
    with bz2.open(path, "rb") as f:
        for _, el in ET.iterparse(f):
            if el.tag.rsplit("}", 1)[-1] == "page":
                ns, title, txt = el.find("{*}ns"), el.find("{*}title"), el.find(".//{*}text")
                yield (title.text if title is not None else ""), (ns.text if ns is not None else ""), \
                    (txt.text if txt is not None and txt.text else "")
                el.clear()


def dewiki(s):
    s = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", s, flags=re.S)
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    for _ in range(2):
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", s)
    return s.replace("'''", "").replace("''", "").strip()


def src_wikiquote(a):
    for title, ns, text in _wiki_pages(Path(a.raw) / "wikiquote/jawikiquote-latest-pages-articles.xml.bz2"):
        if ns == "0":
            for ln in text.split("\n"):
                if ln.startswith("*") and not ln.startswith("**"):
                    yield title, dewiki(ln[1:])


def src_wiktionary(a):
    for title, ns, text in _wiki_pages(Path(a.raw) / "wiktionary/jawiktionary-latest-pages-articles.xml.bz2"):
        if ns == "0":
            for ln in text.split("\n"):
                if ln.startswith("#*") or ln.startswith("#:"):
                    s = dewiki(ln.lstrip("#*: "))
                    if not re.search(r"[A-Za-z]{4,}", s):
                        yield title, s


READERS = {k[4:]: v for k, v in globals().items() if k.startswith("src_")}


def build_source(job):
    """Normalise, segment, filter and split one source; returns (name, {split: lines}, stats)."""
    name, args = job
    vocab = set(load_vocab(args.vocab))
    seen, out, stats = set(), {"train": [], "dev": []}, Counter()
    for doc, text in READERS[name](args):
        for piece in sentences(norm_vn(text)):
            stats["pieces"] += 1
            if not keep(piece, vocab):
                stats["rejected"] += 1
                continue
            if piece in seen:
                stats["duplicate"] += 1
                continue
            seen.add(piece)
            out[split_of(name, doc)].append(piece)
    for split, cap in (("train", TRAIN_CAP), ("dev", DEV_CAP)):
        lines = out[split]
        if len(lines) > cap:          # deterministic subsample: smallest line hashes
            keyed = sorted(lines, key=lambda s: hashlib.sha256(f"{SPLIT_SALT}:{s}".encode()).digest())
            stats[f"{split}_capped_from"] = len(lines)
            out[split] = keyed[:cap]
    return name, out, dict(stats)


def cmd_pools(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.sources or list(SOURCES)
    with mp.get_context("fork").Pool(min(args.workers, len(names))) as pool:
        results = pool.map(build_source, [(n, args) for n in names], chunksize=1)
    dev_all = set()
    for _, lines, _ in results:
        dev_all.update(lines["dev"])
    manifest = {}
    for name, lines, stats in results:
        train = [s for s in lines["train"] if s not in dev_all]
        stats["train_removed_dev_overlap"] = len(lines["train"]) - len(train)
        for split, items in (("train", train), ("dev", lines["dev"])):
            path = out / f"{name}.{split}.txt"
            path.write_text("\n".join(items) + "\n", encoding="utf-8")
            manifest[f"{name}.{split}"] = {"file": path.name, "lines": len(items),
                                           "chars": sum(map(len, items)), "sha256": file_sha256(path),
                                           "bytes": path.stat().st_size, **SOURCES[name]}
        manifest[f"{name}.stats"] = stats
        print(name, stats, flush=True)
    (out / "pools.json").write_text(json.dumps({"salt": SPLIT_SALT, "dev_mod": DEV_MOD, "max_len": MAX_LEN,
                                                "vocab_sha256": file_sha256(args.vocab), "pools": manifest},
                                               ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- names

def reserved_vns(extra=()):
    doc = json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())
    return set(doc["all"]) | set(extra)


def cmd_names(args):
    vocab = set(load_vocab(args.vocab))
    dev_vns = set(Path(args.dev_vns).read_text().split()) if args.dev_vns else set()
    forbid = reserved_vns(dev_vns)
    out = Path(args.out)
    rows = {"train": [], "dev": []}
    stats = Counter()
    with open(args.vndb_chars, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {k: i for i, k in enumerate(header)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[col["lang"]] != "ja":
                continue
            vns = p[col["vn_ids"]].split(",")
            if any(v in forbid or is_test_vn(v) for v in vns):
                stats["vndb_reserved_or_dev"] += 1
                continue
            parts = [norm_vn(x) for x in p[col["name"]].split(" ") if x]
            if not parts or any(not keep(x, vocab) for x in parts) or len("".join(parts)) > 16:
                stats["vndb_rejected"] += 1
                continue
            role = p[col["main_role"]] or "-"
            rows[split_of("vndb", p[col["char_id"]])].append(("vndb", role, *parts[:2]))
            stats["vndb_kept"] += 1
    if args.syosetu:
        import pyarrow.parquet as pq
        names = pq.read_table(args.syosetu, columns=["name"]).column("name").to_pylist()
        keep_syo = []
        for nm in names:
            s = norm_vn(nm or "")
            if 2 <= len(s) <= 8 and keep(s, vocab) and not re.search(r"[0-9A-Za-z０-９Ａ-Ｚａ-ｚ]", s):
                keep_syo.append(s)
        keep_syo = sorted(set(keep_syo), key=lambda s: hashlib.sha256(f"{SPLIT_SALT}:n:{s}".encode()).digest())
        keep_syo = keep_syo[:args.syosetu_cap]
        for s in keep_syo:
            rows[split_of("syosetu", s)].append(("syosetu", "-", s))
        stats["syosetu_kept"] = len(keep_syo)
    dev_names = {r[2:] for r in rows["dev"]}
    rows["train"] = [r for r in rows["train"] if r[2:] not in dev_names]
    for split, items in rows.items():
        path = out / f"names.{split}.tsv"
        path.write_text("".join("\t".join(r) + "\n" for r in items), encoding="utf-8")
        stats[f"{split}_rows"] = len(items)
    (out / "names.json").write_text(json.dumps({"stats": stats, "forbidden_vns": len(forbid),
                                                "dev_vns": sorted(dev_vns),
                                                "sha256": {s: file_sha256(out / f"names.{s}.tsv")
                                                           for s in rows}},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(dict(stats), flush=True)


# ----------------------------------------------------------------------------- contexts

def cmd_index(args):
    """For every vocab character, up to K train contexts per pool: the first K lines that hold
    it in a fixed pseudo-random line order (seeded by the pool name), so the index is
    deterministic and spread over the whole pool."""
    vocab = load_vocab(args.vocab)
    cid = {c: i for i, c in enumerate(vocab)}
    split = args.split
    pools = sorted(p.name[:-len(f".{split}.txt")] for p in Path(args.pools).glob(f"*.{split}.txt"))
    K = args.k
    chars, pool_ids, line_ids = [], [], []
    covered = set()
    for pi, pool in enumerate(pools):
        lines = (Path(args.pools) / f"{pool}.{split}.txt").read_text(encoding="utf-8").split("\n")
        seed = int.from_bytes(hashlib.sha256(f"{SPLIT_SALT}:index:{pool}".encode()).digest()[:8], "little")
        count = Counter()
        for li in np.random.default_rng(seed).permutation(len(lines)):
            s = lines[li]
            for c in set(s):
                if count[c] < K:
                    count[c] += 1
                    chars.append(cid[c])
                    pool_ids.append(pi)
                    line_ids.append(int(li))
        covered |= set(count)
        print(f"index {pool}: {len(lines)} lines, {len(count)} characters", flush=True)
    np.savez_compressed(Path(args.pools) / f"contexts.{split}.npz", char=np.asarray(chars, np.int32),
                        pool=np.asarray(pool_ids, np.int16), line=np.asarray(line_ids, np.int32),
                        pools=np.asarray(json.dumps(pools)))
    print(f"contexts: {len(chars)} refs, {len(covered)} of {len(vocab)} characters have a natural context",
          flush=True)


# ----------------------------------------------------------------------------- targets

def cmd_targets(args):
    """Character weights for focus sampling: real training-side label frequency, the Jiten VN
    word list (Zipf-weighted lemma kanji) and the mining report's per-character error rates."""
    vocab = load_vocab(args.vocab)
    real = Counter()
    for spec in args.real.split("+"):
        z = np.load(spec)
        for lab in json.loads(str(z["labels"])):
            real.update(lab)
    jiten = Counter()
    with open(args.jiten_vn, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split(",")
            if len(p) < 3:
                continue
            try:
                rank = int(p[-1])
            except ValueError:
                continue
            for c in p[0]:
                if JP.match(c):
                    jiten[c] += 1.0 / rank
    hard = {}
    if args.mining:
        rep = json.loads(Path(args.mining).read_text(encoding="utf-8"))
        errs = Counter()
        occ = Counter()
        for m in rep["by_model"].values():
            for c, (e, n) in m["char_errors"].items():
                errs[c] += e
                occ[c] += n
        for c in occ:
            hard[c] = round((errs[c] + 0.5) / (occ[c] + 25), 5)   # shrunk toward a low prior
    tot_r, tot_j = sum(real.values()) or 1, sum(jiten.values()) or 1
    out = {"vocab_sha256": file_sha256(args.vocab), "real_chars": sum(real.values()),
           "jiten_vn": {"file": str(args.jiten_vn), "sha256": file_sha256(args.jiten_vn)},
           "chars": {}}
    for c in vocab:
        out["chars"][c] = {"real": round(real[c] / tot_r, 8), "jiten_vn": round(jiten[c] / tot_j, 8),
                           "hard": hard.get(c, 0.0)}
    Path(args.pools, "targets.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"targets for {len(vocab)} characters; real {sum(real.values())} chars; "
          f"{sum(1 for c in vocab if real[c])} seen in real labels", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=DATA / "vocab.txt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pools")
    p.add_argument("--raw", required=True)
    p.add_argument("--aozora", required=True)
    p.add_argument("--wiki", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sources", nargs="*")
    p.add_argument("--workers", type=int, default=6)
    n = sub.add_parser("names")
    n.add_argument("--vndb-chars", required=True)
    n.add_argument("--syosetu")
    n.add_argument("--syosetu-cap", type=int, default=300_000)
    n.add_argument("--dev-vns")
    n.add_argument("--out", required=True)
    i = sub.add_parser("index")
    i.add_argument("--pools", required=True)
    i.add_argument("--k", type=int, default=48)
    i.add_argument("--split", choices=("train", "dev"), default="train")
    t = sub.add_parser("targets")
    t.add_argument("--pools", required=True)
    t.add_argument("--real", required=True, help="training-side real crops (labels only)")
    t.add_argument("--jiten-vn", required=True)
    t.add_argument("--mining")
    args = ap.parse_args()
    {"pools": cmd_pools, "names": cmd_names, "index": cmd_index, "targets": cmd_targets}[args.cmd](args)


if __name__ == "__main__":
    main()

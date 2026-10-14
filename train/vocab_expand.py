#!/usr/bin/env python3
"""Is the vocabulary too small? VN characters each vocab cannot emit, what the labelling pipelines hid,
what a bigger head costs on the console, and the vocab per tier (docs/vocab-expand.md).

  vocab_expand.py jiten  --jiten DIR --out chars.npz          per-deck character counts from Jiten's deck CSVs
  vocab_expand.py order  --chars chars.npz --out order.json   the VN-weighted character order (design decks only)
  vocab_expand.py curves --chars chars.npz --order order.json --out curves.json
  vocab_expand.py lines  --chars chars.npz --order order.json --out lines.json
  vocab_expand.py bias   --out bias.json                      lines the labelling pipelines dropped or never kept
  vocab_expand.py blobs  --out DIR                            random-init int8 blobs of every head option
                                                             (DIR/sizes.json; then vita/budget_sweep.py a9 --dir DIR)
  vocab_expand.py build  --order order.json --size N ... --out FILE
  vocab_expand.py proof  strata|probe ...                     the proof study's report-only measures
  vocab_expand.py tables FILE.json                            the markdown tables of docs/vocab-expand.md

Emittable. Every count here uses eval's scoring normalisation (eval/metrics.norm: NFKC, folded dashes,
tildes and ellipses): a reference character is emittable by a vocab when every character of its norm()
is produced by some class's norm(). So a vocab with ASCII 'P' emits 'Ｐ', and FL10 lacking full-width
capitals costs nothing on the score (the realdata/vote.canon label convention writes them as ASCII).

Jiten (https://jiten.moe, CC BY-SA 4.0; data/jiten_fetch.py) gives, per title, the occurrences of every
dictionary word form its parser found. A form's characters are counted with its occurrences; the
denominator is the title's characterCount (kana, kanji in U+4E00-9FAF, 々 and full-width Latin and
digits; punctuation is not counted), so "per 1,000 characters" is per 1,000 content characters. JMnedict
entries (word ids 5,000,000-5,999,999) are names. Titles linked to a reserved VNDB VN (eval's test
games, is_test_vn, every synth3 exclusion manifest including CLANNAD and the owner's test titles, the
development VNs) are held out of every design statistic: the order is built from the other titles and
the held-out titles are reported separately (character counts only).
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "synth", ROOT / "eval", ROOT / "realdata", ROOT / "vita"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import align_text, norm  # noqa: E402

HERD = Path("/local/home/skerraut/herd-vita-ocr")
A = HERD / "shared" / "artifacts"
FL10_VOCAB = A / "train2" / "release" / "FL10_vocab_3500.txt"
UNK = "〓"
NAME_IDS = (5_000_000, 6_000_000)          # JMnedict entry ids in Jiten's word table
TYPES = ("VisualNovel", "Novel", "WebNovel", "Anime", "Drama", "Movie", "VideoGame", "Manga", "NonFiction",
         "YouTube", "Audio")
LIST_GROUPS = {"VN": ("VisualNovel",), "Novel": ("Novel", "WebNovel"), "Anime/Drama": ("Anime", "Drama", "Movie"),
               "Game": ("VideoGame",), "Manga": ("Manga",)}
SIZE_GRID = tuple(range(3000, 8001, 100))   # output classes, CTC blank included
TIER_SIZES = (3501, 4987, 5630, 6500, 7500)


# ----------------------------------------------------------------------------- basics

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_vocab(path):
    return [s for s in Path(path).read_text(encoding="utf-8").split("\n") if s]


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F \
        or c == "々"


def is_kana(c):
    o = ord(c)
    return 0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF


def emit_set(vocab):
    """The characters a vocab can produce after eval's norm (the <unk> 〓 emits nothing a reference holds)."""
    out = set()
    for v in vocab:
        if v != UNK:
            out |= set(norm(v))
    return out


def emittable(c, es):
    """True when every character of norm(c) is producible (whitespace folds to nothing)."""
    return all(x in es for x in norm(c))


def reserved_vns(extra=()):
    """synth4_gap.reserved_vns(): test games, the exclusion manifests, v661 and v19373, plus extra lists."""
    import synth4_gap
    return synth4_gap.reserved_vns(extra)


def is_reserved(vns, reserved):
    from common import is_test_vn
    return any(v in reserved or is_test_vn(v) for v in vns)


def default_extra():
    return [p for p in (A / "synth3" / "mining" / "exclude_dev_vns.txt", A / "native-frames" / "owner_games_vns.json")
            if p.exists()]


def pct(xs, qs=(50, 90, 95, 99)):
    xs = np.asarray(xs, np.float64)
    return {f"p{q}": round(float(np.percentile(xs, q)), 4) for q in qs} if len(xs) else {}


# ----------------------------------------------------------------------------- jiten

def read_deck_csv(path):
    """(form, occurrences, word id) rows of one Jiten deck CSV export (gzip or plain)."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rb") as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig", newline="")
        r = csv.reader(text)
        head = next(r)
        iw, io_, iid = head.index("Word"), head.index("Occurences"), head.index("JmDictWordId")
        for row in r:
            if len(row) > max(iw, io_, iid):
                yield row[iw], int(row[io_]), int(row[iid])


def deck_counts(path):
    """(Counter of characters, Counter of characters in name entries, word-form characters) of one deck."""
    cnt, names, tot = Counter(), Counter(), 0
    for form, occ, wid in read_deck_csv(path):
        if occ <= 0:
            continue
        name = NAME_IDS[0] <= wid < NAME_IDS[1]
        for c in form:
            cnt[c] += occ
            if name:
                names[c] += occ
        tot += occ * len(form)
    return cnt, names, tot


def vndb_ids(deck):
    out = []
    for ln in deck.get("links") or []:
        u = ln.get("url") or ""
        if "vndb.org/v" in u:
            v = "v" + u.rsplit("/v", 1)[1].split("/")[0].split("?")[0]
            if v[1:].isdigit():
                out.append(v)
    return sorted(set(out))


def cmd_jiten(args):
    jd = Path(args.jiten)
    reserved = reserved_vns(args.exclude if args.exclude is not None else default_extra())
    chars, cidx = [], {}
    rows, cols, vals, nvals = [], [], [], []
    decks = []
    for t in TYPES:
        meta_p = jd / "decks" / f"decks_{t}.json"
        wdir = jd / "words" / t
        if not meta_p.exists() or not wdir.exists():
            continue
        meta = {d["deckId"]: d for d in json.loads(meta_p.read_text(encoding="utf-8"))}
        for p in sorted(wdir.glob("*.csv.gz"), key=lambda p: int(p.name.split(".")[0])):
            did = int(p.name.split(".")[0])
            d = meta.get(did)
            if d is None:
                continue
            cnt, names, tot = deck_counts(p)
            if not cnt:
                continue
            vn = vndb_ids(d)
            i = len(decks)
            decks.append({"deck": did, "type": t, "content": int(d.get("characterCount") or 0), "wordform": tot,
                          "vndb": vn, "reserved": bool(vn) and is_reserved(vn, reserved),
                          "unique_kanji": d.get("uniqueKanjiCount")})
            for c, n in cnt.items():
                j = cidx.get(c)
                if j is None:
                    j = cidx[c] = len(chars)
                    chars.append(c)
                rows.append(i)
                cols.append(j)
                vals.append(n)
                nvals.append(names.get(c, 0))
    np.savez_compressed(args.out, rows=np.asarray(rows, np.int32), cols=np.asarray(cols, np.int32),
                        vals=np.asarray(vals, np.int64), names=np.asarray(nvals, np.int64),
                        chars=np.asarray(json.dumps(chars, ensure_ascii=False)),
                        decks=np.asarray(json.dumps(decks, ensure_ascii=False)))
    by = Counter((d["type"], d["reserved"]) for d in decks)
    print(f"{len(decks)} decks, {len(chars)} distinct characters, {len(vals)} entries; "
          + ", ".join(f"{t}{' reserved' if r else ''} {n}" for (t, r), n in sorted(by.items())), flush=True)


class Chars:
    """Per-deck character counts (cmd_jiten's npz)."""

    def __init__(self, path):
        z = np.load(path)
        self.rows, self.cols, self.vals, self.names = z["rows"], z["cols"], z["vals"], z["names"]
        self.chars = json.loads(str(z["chars"]))
        self.decks = json.loads(str(z["decks"]))
        self.content = np.asarray([max(1, d["content"]) for d in self.decks], np.float64)
        self.path = str(path)

    def select(self, types=None, reserved=None):
        return np.asarray([(types is None or d["type"] in types) and (reserved is None or d["reserved"] == reserved)
                           for d in self.decks], bool)

    def per_char(self, mask, weight="corpus"):
        """{char: weight} over the decks in mask: corpus (occurrences) or deck (per-deck share, decks equal)."""
        m = mask[self.rows]
        v = self.vals[m].astype(np.float64)
        if weight == "deck":
            v = v / self.content[self.rows[m]] / max(1, mask.sum())
        w = np.bincount(self.cols[m], weights=v, minlength=len(self.chars))
        return {self.chars[i]: float(x) for i, x in enumerate(w) if x > 0}

    def doc_freq(self, mask):
        m = mask[self.rows]
        return np.bincount(self.cols[m], minlength=len(self.chars))


# ----------------------------------------------------------------------------- order

def load_real_text(paths):
    """Characters of training-side real labels (canonical label convention): {char: count}."""
    cnt = Counter()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("status") == "kept" and r.get("label"):
                cnt.update(r["label"])
    return cnt


def cmd_order(args):
    """The VN-weighted order: kanji by mean per-title share over the design VN titles (each title weighs
    the same, so a kanji one title leans on counts as much as one spread thin over many), ties by the
    number of titles using it; then the non-kanji block."""
    ch = Chars(args.chars)
    design = ch.select(("VisualNovel",), reserved=False)
    w_deck = ch.per_char(design, "deck")
    w_corp = ch.per_char(design, "corpus")
    df = ch.doc_freq(design)
    dfc = {ch.chars[i]: int(x) for i, x in enumerate(df) if x}
    kanji = sorted((c for c in w_deck if is_kanji(c)), key=lambda c: (-w_deck[c], -dfc.get(c, 0), c))
    h15 = read_vocab(args.vocab)
    real = load_real_text(args.real)
    # non-kanji: H15's (every one has renders and real or NVIDIA labels), ordered by real VN-label use, then
    # the non-kanji the real training-side labels use at least --min-real times that H15 lacks
    nk_h15 = [c for c in h15 if not is_kanji(c)]
    extra = sorted((c for c, n in real.items() if n >= args.min_real and not is_kanji(c) and c not in set(h15)
                    and norm(c) and c != UNK and not c.isspace()), key=lambda c: (-real[c], c))
    nk = sorted(nk_h15, key=lambda c: (-real.get(c, 0), -w_corp.get(c, 0), h15.index(c))) + extra
    h15_k = [c for c in h15 if is_kanji(c)]
    # kanji outside the parser's range (Jiten strips them: Ext A, compatibility ideographs): H15's, kept
    outside = [c for c in h15_k if not (0x4E00 <= ord(c) <= 0x9FAF or c == "々") and c not in w_deck]
    out = {"kind": "vocab-expand-order", "schema": 1,
           "inputs": {"chars": {"file": args.chars, "sha256": sha256(args.chars)}, "vocab_h15": sha256(args.vocab),
                      "real": [{"file": str(p), "sha256": sha256(p)} for p in args.real if Path(p).exists()]},
           "design_titles": int(design.sum()), "held_out_titles": int(ch.select(("VisualNovel",), True).sum()),
           "non_kanji": nk, "non_kanji_extra": extra, "kanji": kanji, "kanji_outside_jiten": outside,
           "weights": {c: [round(1e6 * w_deck[c], 4), dfc.get(c, 0)] for c in kanji}}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(kanji)} VN kanji over {int(design.sum())} design titles; "
          f"{len(nk)} non-kanji ({len(extra)} beyond H15)", flush=True)


def vocab_of_size(order, n_classes, unk=True, h15=None):
    """A vocab of n_classes output classes (CTC blank included): the non-kanji block, the compatibility
    kanji Jiten cannot see, then kanji in VN order, then the H15 kanji the VN titles never use (H15's order),
    then <unk>."""
    n = n_classes - 1 - (1 if unk else 0)
    base = list(order["non_kanji"]) + list(order.get("kanji_outside_jiten", []))
    tail = [c for c in (h15 or []) if is_kanji(c) and c not in set(order["kanji"]) and c not in set(base)]
    pool = base + list(order["kanji"]) + tail
    if n > len(pool):
        raise ValueError(f"only {len(pool)} ordered characters for {n_classes} classes")
    out = pool[:n] + ([UNK] if unk else [])
    assert len(out) == len(set(out))
    return out


# ----------------------------------------------------------------------------- curves

def deck_missed(ch, vocab_es):
    """Per deck: occurrences of characters the vocab cannot emit, and of those in name entries."""
    ok = np.asarray([emittable(c, vocab_es) for c in ch.chars], bool)
    miss = ~ok[ch.cols]
    n = np.bincount(ch.rows[miss], weights=ch.vals[miss].astype(np.float64), minlength=len(ch.decks))
    nn = np.bincount(ch.rows[miss], weights=ch.names[miss].astype(np.float64), minlength=len(ch.decks))
    return n, nn


def cover_row(ch, mask, missed, names=None):
    """Missed per 1,000 content characters over the decks in mask: corpus-weighted, per-title mean and
    percentiles; titles with any miss."""
    if not mask.any():
        return None
    per = 1000 * missed[mask] / ch.content[mask]
    r = {"titles": int(mask.sum()), "corpus": round(float(1000 * missed[mask].sum() / ch.content[mask].sum()), 4),
         "title_mean": round(float(per.mean()), 4), **pct(per), "max": round(float(per.max()), 3),
         "titles_with_miss": round(float((missed[mask] > 0).mean()), 4)}
    if names is not None:
        r["name_share"] = round(float(names[mask].sum() / max(1e-9, missed[mask].sum())), 4)
    return r


def mixture(rows, ch):
    """Jiten's global list as a mixture: each media type's corpus-weighted rate, weighted by the type's
    total characters on Jiten (all its titles, sampled or not)."""
    return None if not rows else rows


def cmd_curves(args):
    ch = Chars(args.chars)
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))
    h15 = read_vocab(args.vocab)
    jd = Path(args.jiten) if args.jiten else None
    type_chars = {}
    if jd:
        for t in TYPES:
            p = jd / "decks" / f"decks_{t}.json"
            if p.exists():
                type_chars[t] = sum(int(d.get("characterCount") or 0) for d in json.loads(p.read_text(encoding="utf-8"))
                                    if d.get("parentDeckId") is None)
    groups = {"VN design": ch.select(("VisualNovel",), False), "VN held out": ch.select(("VisualNovel",), True)}
    for g, ts in LIST_GROUPS.items():
        if g != "VN":
            groups[g] = ch.select(ts)
    fixed = {"FL10": read_vocab(FL10_VOCAB), "H15": h15}
    for spec in args.vocab_file or []:
        name, _, p = spec.partition("=")
        fixed[name] = read_vocab(p)
    out = {"kind": "vocab-expand-curves", "schema": 1,
           "inputs": {"chars": {"file": args.chars, "sha256": sha256(args.chars)},
                      "order": {"file": args.order, "sha256": sha256(args.order)}},
           "type_chars": type_chars, "fixed": {}, "curve": [], "global_weights": {}}

    def evaluate(vocab):
        missed, names = deck_missed(ch, emit_set(vocab))
        res = {g: cover_row(ch, m, missed, names) for g, m in groups.items()}
        per_type = {}
        for t in TYPES:
            m = ch.select((t,))
            if m.any():
                per_type[t] = float(missed[m].sum() / ch.content[m].sum())
        if type_chars:
            tot = sum(type_chars[t] for t in per_type)
            res["global (mixture)"] = {"corpus": round(1000 * sum(per_type[t] * type_chars[t] for t in per_type) / tot,
                                                       4), "types": len(per_type)}
        return res, missed, names
    for name, v in fixed.items():
        res, missed, names = evaluate(v)
        out["fixed"][name] = {"classes": len(v) + 1, **res}
    for n in SIZE_GRID:
        try:
            v = vocab_of_size(order, n, unk=True, h15=h15)
        except ValueError:
            break
        res, _, _ = evaluate(v)
        out["curve"].append({"classes": n, **res})
        print(f"{n}: VN design {res['VN design']['title_mean']:.3f} p95 {res['VN design']['p95']:.3f}; "
              f"held out {res['VN held out']['title_mean'] if res['VN held out'] else '-'}", flush=True)
    # where the misses sit at the tier sizes: names, the concentration over titles, per-kanji spread
    vn = ch.select(("VisualNovel",))
    out["concentration"] = {}
    for n in TIER_SIZES:
        try:
            v = vocab_of_size(order, n, unk=True, h15=h15) if n != 3501 else fixed["FL10"]
        except ValueError:
            continue
        es = emit_set(v)
        ok = np.asarray([emittable(c, es) for c in ch.chars], bool)
        miss = (~ok[ch.cols]) & vn[ch.rows]
        per_deck = np.bincount(ch.rows[miss], weights=ch.vals[miss].astype(np.float64), minlength=len(ch.decks))[vn]
        s = np.sort(per_deck)[::-1]
        tot = s.sum()
        top = {f"top{k}pct_titles": round(float(s[:max(1, int(len(s) * k / 100))].sum() / max(tot, 1e-9)), 4)
               for k in (1, 5, 10)}
        per_char_titles = Counter(ch.chars[c] for c in ch.cols[miss])
        tops = Counter()
        for c, val in zip(ch.cols[miss], ch.vals[miss]):
            tops[ch.chars[c]] += int(val)
        out["concentration"][str(n)] = {**top, "missed_kanji_types": len(per_char_titles),
                                        "titles_per_missed_char": pct(list(per_char_titles.values()), (50, 90)),
                                        "name_share": round(float(ch.names[miss].sum()
                                                                   / max(1, ch.vals[miss].sum())), 4),
                                        "top_missed": [[c, k, per_char_titles[c]] for c, k in tops.most_common(40)]}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- lines

def content_len(s):
    """Content characters of a label as Jiten counts them (kana, kanji, Latin letters and digits)."""
    return sum(1 for c in norm(s) if is_kana(c) or is_kanji(c) or c.isalnum())


def line_hits(lines, es):
    """(lines, lines holding a character the vocab cannot emit, characters, such characters)."""
    n = bad = chars = badc = 0
    for s in lines:
        t = norm(s)
        if not t:
            continue
        n += 1
        chars += len(t)
        miss = sum(1 for c in t if not emittable(c, es))
        badc += miss
        bad += miss > 0
    return {"lines": n, "per_1000_lines": round(1000 * bad / max(1, n), 3), "lines_with_miss": bad,
            "chars": chars, "per_1000_chars": round(1000 * badc / max(1, chars), 4)}


def predicted_line_share(q, lengths):
    """Share of lines with at least one miss when each content character misses with probability q,
    independently, over the empirical content-length distribution."""
    L = np.asarray(lengths, np.float64)
    return float(np.mean(1 - (1 - q) ** L))


def cmd_lines(args):
    ch = Chars(args.chars)
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))
    h15 = read_vocab(args.vocab)
    vocabs = {"FL10": read_vocab(FL10_VOCAB), "H15": h15}
    for n in TIER_SIZES[1:]:
        try:
            vocabs[f"VN{n}"] = vocab_of_size(order, n, unk=True, h15=h15)
        except ValueError:
            continue
    for spec in args.vocab_file or []:
        name, _, p = spec.partition("=")
        vocabs[name] = read_vocab(p)
    # the content-length distribution of real VN dialogue lines: training-side realplus v3 kept labels
    rows = (json.loads(x) for x in Path(args.lengths).read_text(encoding="utf-8").splitlines())
    lens = [content_len(r["label"]) for r in rows if r.get("status") == "kept" and r.get("label")]
    lens = [x for x in lens if x > 0]
    sets = {}
    for spec in args.set or []:
        name, _, p = spec.partition("=")
        sets[name] = load_line_set(p)
    out = {"kind": "vocab-expand-lines", "schema": 1, "lengths": {"file": args.lengths, "lines": len(lens),
                                                                  **pct(lens, (10, 50, 90))},
           "vocabs": {k: len(v) + 1 for k, v in vocabs.items()}, "measured": {}, "modelled": {}, "calibration": {}}
    vn_design, vn_held = ch.select(("VisualNovel",), False), ch.select(("VisualNovel",), True)
    for name, v in vocabs.items():
        es = emit_set(v)
        missed, _ = deck_missed(ch, es)
        q = missed / ch.content
        for tag, m in (("VN design", vn_design), ("VN held out", vn_held)):
            if not m.any():
                continue
            per = np.asarray([predicted_line_share(x, lens) for x in q[m]])
            out["modelled"].setdefault(name, {})[tag] = {
                "title_mean_per_1000_lines": round(1000 * float(per.mean()), 3),
                **{k: round(1000 * v_, 3) for k, v_ in pct(per).items()}}
        for sname, lines in sets.items():
            out["measured"].setdefault(name, {})[sname] = line_hits(lines, es)
            # the independence model checked on the same set: its own per-character miss rate q applied to
            # its own content-length distribution, against the observed per-1,000-line rate
            hit = out["measured"][name][sname]
            qq = hit["per_1000_chars"] / 1000
            ln = [content_len(s) for s in lines if content_len(s) > 0]
            out["calibration"].setdefault(name, {})[sname] = {
                "observed_per_1000_lines": hit["per_1000_lines"],
                "independence_per_1000_lines": round(1000 * predicted_line_share(qq, ln), 3)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_line_set(path):
    """Reference texts of one line set: a labels.jsonl (kept rows), an eval manifest (scored rows), a CLANNAD
    refs.jsonl (status ref) or an npz with labels."""
    p = Path(path)
    if p.suffix == ".npz":
        import synth4_gap
        return [x for x in synth4_gap.npz_text(p, ("labels",))["labels"] if x]
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "status" in r and r["status"] not in ("kept", "ref"):
            continue
        if r.get("uncertain") or r.get("notext"):
            continue
        t = r.get("label") or r.get("text")
        if t:
            out.append(t)
    return out


# ----------------------------------------------------------------------------- bias

def oov_stats(rows, vocabs):
    """rows: [(vn, text)]. Per vocab: lines, characters, VNs holding a character it cannot emit."""
    out = {}
    for name, es in vocabs.items():
        lines = chars = 0
        vns = set()
        miss_c = Counter()
        for vn, t in rows:
            bad = [c for c in norm(t) if not emittable(c, es)]
            if bad:
                lines += 1
                chars += len(bad)
                vns.add(vn)
                miss_c.update(bad)
        out[name] = {"lines": lines, "chars": chars, "vns": len(vns),
                     "top": "".join(c for c, _ in miss_c.most_common(25))}
    return out


def cmd_bias(args):
    """What the labelling pipelines dropped for being out of H15's vocab, and what they never kept."""
    reserved = reserved_vns(default_extra())
    from common import is_test_vn
    ok_vn = lambda v: bool(v) and v not in reserved and not is_test_vn(v)  # noqa: E731
    h15, fl10 = read_vocab(args.vocab), read_vocab(FL10_VOCAB)
    vocabs = {"H15": emit_set(h15), "FL10": emit_set(fl10)}
    for spec in args.vocab_file or []:
        name, _, p = spec.partition("=")
        vocabs[name] = emit_set(read_vocab(p))
    out = {"kind": "vocab-expand-bias", "schema": 1, "sources": {}, "protected": {}}
    # 1. kept labels the pool builders drop for a character outside data/vocab.txt (realplus pool.py and
    #    native.py, teacher/build.py and realdata/build.py all apply the same test after review)
    owner_pool = HERD / "shared" / "private" / "owner-games" / "_build" / "owner-pool" / "labels.jsonl"
    for name, path in (("realplus v2", A / "realplus" / "labels.jsonl"),
                       ("realplus v3", A / "realplus-v3" / "labels.jsonl"),
                       ("native-frames pool", A / "native-frames" / "pool" / "labels.jsonl"),
                       ("owner pool", owner_pool)):
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        kept = [(r.get("vn"), r["label"]) for r in rows if r.get("status") == "kept" and r.get("label")
                and (name == "owner pool" or ok_vn(r.get("vn")))]
        why = Counter(r.get("why") for r in rows if r.get("status") != "kept")
        out["sources"][name] = {"file": str(path), "sha256": sha256(path), "rows": len(rows), "kept": len(kept),
                                "kept_vns": len({v for v, _ in kept}), "kept_chars": sum(len(norm(t)) for _, t in kept),
                                "dropped_why": dict(why.most_common()), "kept_outside": oov_stats(kept, vocabs)}
        # 2. the engine readings of the lines never kept: how often both engine families read the same
        #    character outside the vocab, against the same rate on the kept lines (realplus' reads)
        reads_p = path.parent / "reads" / "reads.jsonl"
        if reads_p.exists() and name.startswith("realplus"):
            out["sources"][name]["engines"] = engine_oov(reads_p, rows, vocabs, ok_vn)
    # 3. teacher and realdata: re-run their acceptance on the raw reads and count the OOV drops
    out["sources"]["teacher (raw reads)"] = teacher_oov(vocabs, ok_vn)
    out["sources"]["realdata (raw reads)"] = realdata_oov(vocabs, ok_vn)
    # 4. protected and development sets: OOV rate only (character and line counts; nothing is selected)
    prot = {"vita-test": ROOT / "eval" / "manifests" / "vita-test.jsonl",
            "psp-test": ROOT / "eval" / "manifests" / "psp-test.jsonl",
            "CLANNAD refs": HERD / "shared" / "private" / "owner-drive-20261013" / "clannad-compare" / "refs.jsonl",
            "devnative (kept)": A / "devnative" / "labels.jsonl"}
    for name, p in prot.items():
        if not Path(p).exists():
            continue
        lines = load_line_set(p)
        out["protected"][name] = {"sha256": sha256(p), **{k: line_hits(lines, es) for k, es in vocabs.items()}}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def engine_oov(reads_p, labels, vocabs, ok_vn):
    """Lines of a realplus reads.jsonl: for kept and for not-kept lines, the share where MeikiOCR and PP-OCRv6
    small both read a character the vocab cannot emit (the same character), and where either does."""
    status = {r["id"]: r.get("status") for r in labels}
    res = {k: {"kept": Counter(), "not_kept": Counter()} for k in vocabs}
    vns_nk = {k: set() for k in vocabs}
    for line in open(reads_p, encoding="utf-8"):
        s = json.loads(line)
        if not ok_vn(s.get("vn")):
            continue
        for ln in s.get("lines") or []:
            lid = f"{s['shot']}_{ln['k']}"
            st = status.get(lid)
            if st is None:
                continue
            grp = "kept" if st == "kept" else "not_kept"
            rd = ln.get("reads") or {}
            m = norm((rd.get("meiki") or [""])[0])
            v = norm((rd.get("v6s") or [""])[0])
            for k, es in vocabs.items():
                om, ov = {c for c in m if not emittable(c, es)}, {c for c in v if not emittable(c, es)}
                res[k][grp]["lines"] += 1
                res[k][grp]["either"] += bool(om or ov)
                res[k][grp]["both"] += bool(om & ov)
                if grp == "not_kept" and om & ov:
                    vns_nk[k].add(s["vn"])
    out = {}
    for k, r in res.items():
        out[k] = {g: {"lines": c["lines"], "both_engines_per_1000": round(1000 * c["both"] / max(1, c["lines"]), 3),
                      "either_engine_per_1000": round(1000 * c["either"] / max(1, c["lines"]), 3),
                      "both_lines": c["both"]} for g, c in r.items()}
        out[k]["not_kept"]["both_vns"] = len(vns_nk[k])
    return out


def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def teacher_oov(vocabs, ok_vn):
    """teacher/build.py's acceptance (vote.accept with teacher/rule.json) on its raw reads: accepted labels
    holding a character outside each vocab (H15's are the lines build.py dropped as oov)."""
    tdir = ROOT / "teacher"
    if str(tdir) not in sys.path:
        sys.path.insert(0, str(tdir))
    try:
        G = _load_module("teacher_geometry", tdir / "geometry.py")
        TV = _load_module("teacher_vote", tdir / "vote.py")
    except Exception as e:     # teacher's modules need no extra packages, but stay optional here
        return {"error": repr(e)}
    rule = json.loads((tdir / "rule.json").read_text(encoding="utf-8"))["rule"]
    rows = []
    files = sorted((A / "teacher" / "reads").glob("train.*of5.jsonl"))
    for p_ in files:
        for line in open(p_, encoding="utf-8"):
            r = json.loads(line)
            if not ok_vn(r.get("vn")):
                continue
            for ln in G.refurigana(r.get("lines") or []):
                lab, _ = TV.accept(ln, rule)
                if lab:
                    rows.append((r["vn"], lab))
    return {"files": [{"file": str(p_), "sha256": sha256(p_)} for p_ in files], "accepted": len(rows),
            "accepted_vns": len({v for v, _ in rows}), "accepted_outside": oov_stats(rows, vocabs)}


def realdata_oov(vocabs, ok_vn):
    """realdata/build.py's default rule on its raw reads (labels/lines.*.jsonl): accepted labels outside each vocab."""
    try:
        RV = _load_module("realdata_vote", ROOT / "realdata" / "vote.py")
        rule = RV.RULES[RV.DEFAULT_RULE]
    except Exception as e:
        return {"error": repr(e)}
    rows = []
    files = sorted((A / "realdata" / "labels").glob("lines.*of10.jsonl"))
    for p_ in files:
        for line in open(p_, encoding="utf-8"):
            r = json.loads(line)
            if not ok_vn(r.get("vn")):
                continue
            lab, _ = rule({k: v[0] for k, v in r["r"].items()})
            if lab and RV.japanese(lab):
                rows.append((r["vn"], lab))
    return {"files": len(files), "accepted": len(rows), "accepted_vns": len({v for v, _ in rows}),
            "accepted_outside": oov_stats(rows, vocabs)}


# ----------------------------------------------------------------------------- blobs

# Head options at the always tier's and the H15 topology's widths (the model-tiers study covers the other
# widths): bottleneck 48-128, the classes of each vocab candidate.
BLOB_GRID = [(w, b, c) for w in (1.0, 1.5) for b in (64, 96, 128) for c in TIER_SIZES] + \
            [(1.0, 48, c) for c in TIER_SIZES]


def blob_name(width, bottleneck, classes):
    return f"w{int(round(width * 100)):03d}b{bottleneck}c{classes}"


def filler_vocab(n_classes):
    """n_classes - 1 characters, 3 UTF-8 bytes each (as kana and kanji), for size-exact random blobs."""
    return [chr(0x4E00 + i) for i in range(n_classes - 1)]


def cmd_blobs(args):
    import budget_sweep as BS

    import export
    out = Path(args.out)
    (out / "blobs").mkdir(parents=True, exist_ok=True)
    sizes_p = out / "sizes.json"
    sizes = json.loads(sizes_p.read_text()) if sizes_p.exists() else {}
    grid = list(BLOB_GRID)
    for spec in args.vocab_file or []:          # exact blobs of the recommended vocab files
        name, _, p = spec.partition("=")
        w, b = name.split(":")[1:3] if ":" in name else ("1.0", "96")
        grid.append((float(w), int(b), read_vocab(p), name.split(":")[0]))
    for g in grid:
        if len(g) == 3:
            w, b, c = g
            vocab, name = filler_vocab(c), blob_name(w, b, c)
        else:
            w, b, vocab, name = g
        if name in sizes and (out / "blobs" / f"{name}_w8.vocr").exists():
            continue
        model = BS.random_model(w, b, 1, 2, 5, len(vocab) + 1)
        data, meta = export.build_blob(export.model_ops(model, 32), vocab, 32, int8=True)
        (out / "blobs" / f"{name}_w8.vocr").write_bytes(data)
        info = BS.blob_info(data)
        info.update(config=dict(width=w, bottleneck=b, depth=1, ctx=2, ctx_k=5, classes=len(vocab) + 1, height=32),
                    random_init=True, params=meta["params"], sha256=hashlib.sha256(data).hexdigest())
        sizes[name] = info
        print(f"{name}: {len(data):,} B, arena {info['arena']['t4h16']['w8i']:,} B, "
              f"{info['macs_per_frame']:,} MACs/frame", flush=True)
    if BS.RELEASE:
        for rel in ("FL10", "H15"):
            p = BS.RELEASE / f"{rel}_w8.vocr"
            if p.exists() and rel not in sizes:
                data = p.read_bytes()
                (out / "blobs" / f"{rel}_w8.vocr").write_bytes(data)
                info = BS.blob_info(data)
                info.update(released=True, sha256=hashlib.sha256(data).hexdigest())
                sizes[rel] = info
    sizes_p.write_text(json.dumps(sizes, indent=1, ensure_ascii=False))


def tier_rows(sizes_dir):
    """Blob size, arena, MACs and the estimated ms of every blob with an a9 run (2 MiB L2, default tiling)."""
    import budget_sweep as BS
    d = Path(sizes_dir)
    sizes = json.loads((d / "sizes.json").read_text())
    runs = BS.load_runs(d)
    out = {}
    for name, s in sizes.items():
        r = runs.get((name, *BS.BASE))
        row = {"classes": s["classes"], "blob_bytes": s["blob_bytes"], "head_bytes": s["head_weight_bytes"],
               "arena_w8i": s["arena"]["t4h16"]["w8i"], "macs_per_frame": s["macs_per_frame"],
               "head_macs_per_frame": s["head_macs_per_frame"], "params": s["params"], "config": s.get("config")}
        if r:
            for mhz in (333, 444):
                lo, hi = BS._ms(r, mhz)
                row[f"ms_{mhz}"] = [round(lo, 1), round(hi, 1)]
                row[f"box3_s_{mhz}"] = [round(3 * lo / 1000, 2), round(3 * hi / 1000, 2)]
            row["tier"] = BS.tier(s, r)
        out[name] = row
    return out


# ----------------------------------------------------------------------------- build

def cmd_build(args):
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))
    h15 = read_vocab(args.vocab)
    v = vocab_of_size(order, args.size, unk=not args.no_unk, h15=h15)
    Path(args.out).write_text("\n".join(v) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(v)} entries, {len(v) + 1} classes, sha256 {sha256(args.out)}", flush=True)


# ----------------------------------------------------------------------------- proof (report-only measures)

def char_groups(vocab_a, vocab_b):
    """Group of a reference character for the proof: added (only B emits it), shared, dropped (only A), other."""
    ea, eb = emit_set(vocab_a), emit_set(vocab_b)

    def g(c):
        a, b = emittable(c, ea), emittable(c, eb)
        return "shared" if a and b else "added" if b else "dropped" if a else "other"
    return g


def char_recall(refs, preds, group):
    """{group: [correct, total]} and {char: [correct, total]} over aligned (reference, prediction) pairs."""
    per_g, per_c = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for ref, pred in zip(refs, preds):
        for gc, pc in align_text(ref, pred):
            if gc is None:
                continue
            k = group(gc)
            per_g[k][1] += 1
            per_c[gc][1] += 1
            if pc == gc:
                per_g[k][0] += 1
                per_c[gc][0] += 1
    return dict(per_g), dict(per_c)


def cmd_proof_strata(args):
    """Character-level strata of the proof's development scores (eval/synth3_ablation.py score files)."""
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    fa = read_vocab(FL10_VOCAB)
    fb = read_vocab(ROOT / proto["arms_detail"]["G_v4987"]["vocab"]["file"])
    group = char_groups(fa, fb)
    out = {"kind": "vocab-proof-strata", "protocol_sha256": sha256(args.protocol), "files": {}}
    for p in args.scores:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        refs = [x["ref"] for x in s["lines"]]
        res = {}
        for arm, runs in proto["arms"].items():
            g_tot = defaultdict(lambda: [0, 0])
            for run in runs:
                m = s["models"].get(run)
                if m is None:
                    continue
                for view in proto["views"]:
                    pg, _ = char_recall(refs, m["pred"][view], group)
                    for k, (a, b) in pg.items():
                        g_tot[k][0] += a
                        g_tot[k][1] += b
            res[arm] = {k: {"correct": a, "total": b, "recall": round(a / b, 4) if b else None}
                        for k, (a, b) in sorted(g_tot.items())}
        line_tags = Counter()
        for r in refs:
            gs = {group(c) for c in r}
            line_tags["line-added"] += "added" in gs
            line_tags["line-shared"] += gs <= {"shared"}
        out["files"][Path(p).name] = {"lines": len(refs), "line_tags": dict(line_tags), "arms": res}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cmd_proof_probe(args):
    """The synthetic probe: per-class recall on synth4's calibration render c2 through the C runtime W8I."""
    from pyvocr import Vocr

    import synth4_gap
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    fa = read_vocab(FL10_VOCAB)
    fb = read_vocab(ROOT / proto["arms_detail"]["G_v4987"]["vocab"]["file"])
    group = char_groups(fa, fb)
    exp = json.loads(Path(args.exposures).read_text(encoding="utf-8"))["chars_by"]
    crops, refs = [], []
    for name, rec in proto["probe"]["set"]["files"].items():
        if sha256(rec["file"]) != rec["sha256"]:
            raise SystemExit(f"{name} differs from the protocol")
        z = np.load(rec["file"])
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        labels = synth4_gap.npz_text(rec["file"], ("labels",))["labels"]
        for i, lab in enumerate(labels):
            if not lab:
                continue
            h, w = (int(v) for v in shapes[i])
            crops.append(data[offs[i]:offs[i + 1]].reshape(h, w))
            refs.append(norm(lab))
    if args.limit:
        crops, refs = crops[:args.limit], refs[:args.limit]
    out = {"kind": "vocab-proof-probe", "protocol_sha256": sha256(args.protocol), "crops": len(crops),
           "chars": sum(len(r) for r in refs), "exposures": {"file": args.exposures, "sha256": sha256(args.exposures)},
           "runs": {}}
    bins = ((0, 50), (50, 100), (100, 300), (300, 1000), (1000, 10 ** 12))
    for spec in args.blob:
        name, _, path = spec.partition("=")
        rt = Vocr(path, mode="w8i", max_width=1024)
        preds = [norm(rt.text(g)) for g in crops]
        pg, pc = char_recall(refs, preds, group)
        cer = sum(len([1 for a, b in align_text(r, p) if a != b]) for r, p in zip(refs, preds)) / max(1, out["chars"])
        by_bin = {}
        for k in ("added", "shared"):
            for lo, hi in bins:
                cs = [c for c in pc if group(c) == k and lo <= exp.get(c, {}).get("occ", 0) < hi]
                a = sum(pc[c][0] for c in cs)
                b = sum(pc[c][1] for c in cs)
                by_bin[f"{k} exposures {lo}-{hi if hi < 10 ** 12 else 'inf'}"] = {
                    "classes": len(cs), "chars": b, "recall": round(a / b, 4) if b else None}
        out["runs"][name] = {"blob_sha256": sha256(path), "cer": round(cer, 5),
                             "groups": {k: {"correct": a, "total": b, "recall": round(a / b, 4) if b else None}
                                        for k, (a, b) in sorted(pg.items())}, "by_exposure": by_bin,
                             "added_classes_seen": sum(1 for c in pc if group(c) == "added"),
                             "added_classes_learned": sum(1 for c in pc if group(c) == "added" and pc[c][0] > 0)}
        groups_s = ", ".join(f"{k} {v['recall']}" for k, v in out["runs"][name]["groups"].items())
        print(f"{name}: CER {cer:.4f}; " + groups_s, flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- tables

def fmt(x, nd=3):
    return "-" if x is None else f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


def cmd_tables(args):
    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    k = d.get("kind")
    if k == "vocab-expand-curves":
        groups = [g for g in d["curve"][0] if g not in ("classes",) and isinstance(d["curve"][0][g], dict)]
        print("| vocab | classes | " + " | ".join(f"{g}: per 1,000 (corpus / title mean / p95)" for g in groups[:2])
              + " | " + " | ".join(f"{g} corpus" for g in groups[2:]) + " |")
        print("|---|---|" + "---|" * len(groups))
        rows = [(n, r) for n, r in d["fixed"].items()] + [(f"VN-ordered {r['classes']:,}", r) for r in d["curve"]
                                                          if r["classes"] in (3000, 3500, 4000, 4500, 4987, 5000, 5630,
                                                                              6000, 6500, 7000, 7500, 8000)]
        for name, r in rows:
            cells = []
            for g in groups[:2]:
                x = r.get(g)
                cells.append("-" if not x else f"{x['corpus']:.3f} / {x['title_mean']:.3f} / {x['p95']:.3f}")
            for g in groups[2:]:
                x = r.get(g)
                cells.append("-" if not x else f"{x['corpus']:.3f}")
            print(f"| {name} | {r['classes']:,} | " + " | ".join(cells) + " |")
    elif k == "vocab-expand-bias":
        print("| source | kept lines | VNs | H15 cannot emit: lines / VNs | FL10 cannot emit: lines / VNs |")
        print("|---|---|---|---|---|")
        for name, r in d["sources"].items():
            o = r.get("kept_outside") or r.get("accepted_outside")
            if not o:
                continue
            n = r.get("kept", r.get("accepted"))
            print(f"| {name} | {n:,} | {r.get('kept_vns', r.get('accepted_vns')):,} | {o['H15']['lines']} / "
                  f"{o['H15']['vns']} | {o['FL10']['lines']} / {o['FL10']['vns']} |")
        print()
        print("| set | lines | H15: per 1,000 lines / chars | FL10: per 1,000 lines / chars |")
        print("|---|---|---|---|")
        for name, r in d["protected"].items():
            print(f"| {name} | {r['H15']['lines']:,} | {r['H15']['per_1000_lines']} / {r['H15']['per_1000_chars']} | "
                  f"{r['FL10']['per_1000_lines']} / {r['FL10']['per_1000_chars']} |")
    else:
        print(json.dumps(d, ensure_ascii=False, indent=1)[:4000])


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jiten")
    j.add_argument("--jiten", required=True)
    j.add_argument("--exclude", nargs="*", default=None)
    j.add_argument("--out", required=True)
    o = sub.add_parser("order")
    o.add_argument("--chars", required=True)
    o.add_argument("--vocab", default=str(ROOT / "data" / "vocab.txt"))
    o.add_argument("--real", nargs="*", default=[str(A / "realplus-v3" / "labels.jsonl"),
                                                 str(A / "native-frames" / "pool" / "labels.jsonl")])
    o.add_argument("--min-real", type=int, default=3)
    o.add_argument("--out", required=True)
    c = sub.add_parser("curves")
    c.add_argument("--chars", required=True)
    c.add_argument("--order", required=True)
    c.add_argument("--jiten")
    c.add_argument("--vocab", default=str(ROOT / "data" / "vocab.txt"))
    c.add_argument("--vocab-file", action="append", help="NAME=FILE: also report this vocab")
    c.add_argument("--out", required=True)
    ln = sub.add_parser("lines")
    ln.add_argument("--chars", required=True)
    ln.add_argument("--order", required=True)
    ln.add_argument("--vocab", default=str(ROOT / "data" / "vocab.txt"))
    ln.add_argument("--vocab-file", action="append")
    ln.add_argument("--lengths", default=str(A / "realplus-v3" / "labels.jsonl"))
    ln.add_argument("--set", action="append", help="NAME=FILE: a line set measured directly")
    ln.add_argument("--out", required=True)
    b = sub.add_parser("bias")
    b.add_argument("--vocab", default=str(ROOT / "data" / "vocab.txt"))
    b.add_argument("--vocab-file", action="append")
    b.add_argument("--out", required=True)
    bl = sub.add_parser("blobs")
    bl.add_argument("--out", required=True)
    bl.add_argument("--vocab-file", action="append", help="NAME[:WIDTH:BOTTLENECK]=FILE: an exact blob of a vocab")
    bu = sub.add_parser("build")
    bu.add_argument("--order", required=True)
    bu.add_argument("--vocab", default=str(ROOT / "data" / "vocab.txt"))
    bu.add_argument("--size", type=int, required=True, help="output classes, CTC blank included")
    bu.add_argument("--no-unk", action="store_true")
    bu.add_argument("--out", required=True)
    pr = sub.add_parser("proof")
    psub = pr.add_subparsers(dest="what", required=True)
    ps = psub.add_parser("strata")
    ps.add_argument("--protocol", default=str(ROOT / "eval" / "vocab_proof_protocol.json"))
    ps.add_argument("--scores", nargs="+", required=True)
    ps.add_argument("--out", required=True)
    pp = psub.add_parser("probe")
    pp.add_argument("--protocol", default=str(ROOT / "eval" / "vocab_proof_protocol.json"))
    pp.add_argument("--blob", action="append", required=True)
    pp.add_argument("--exposures", required=True, help="synth4_gap.py exposures of synth4's pinned render")
    pp.add_argument("--limit", type=int, default=0)
    pp.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("json")
    args = ap.parse_args()
    if args.cmd == "proof":
        {"strata": cmd_proof_strata, "probe": cmd_proof_probe}[args.what](args)
        return
    {"jiten": cmd_jiten, "order": cmd_order, "curves": cmd_curves, "lines": cmd_lines, "bias": cmd_bias,
     "blobs": cmd_blobs, "build": cmd_build, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()

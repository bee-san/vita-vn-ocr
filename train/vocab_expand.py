#!/usr/bin/env python3
"""Is the vocabulary too small? VN characters each vocab cannot emit, what the labelling pipelines hid,
what a bigger head costs on the console, and the vocab per tier (docs/vocab-expand.md).

  vocab_expand.py agg    --lists DIR --real NAME=FILE ... --out agg.json --order-out order.json
                         the aggregate Jiten lists: frequency expansion, the VN-first kanji order, coverage
                         curves per list, the Kanji CSV check, VNDB names
  vocab_expand.py build  --order order.json --size N --out FILE        a vocab file of N output classes
  vocab_expand.py sample --jiten DIR --out sample.npz                  character counts of the per-deck VN sample
  vocab_expand.py tail   --sample sample.npz --order order.json --out tail.json
                         the per-VN distribution (the sample): mean, p95, worst titles, reserved-title check
  vocab_expand.py lines  --agg agg.json --tail tail.json --out lines.json
  vocab_expand.py bias   --out bias.json                                lines the labelling pipelines dropped
  vocab_expand.py fonts  --vocab-file NAME=FILE ... --out fonts.json   fonts and contexts for the new classes
  vocab_expand.py blobs  --out DIR                                      random-init int8 blobs of every head option
                                                                        (then vita/budget_sweep.py a9 --dir DIR)
  vocab_expand.py proof  strata|probe ...                               the proof study's report-only measures
  vocab_expand.py tables FILE.json                                      markdown tables of docs/vocab-expand.md

Sources. Jiten (https://jiten.moe/frequency-dictionaries, CC BY-SA 4.0; data/jiten_fetch.py) publishes word
lists per media type and a kanji list. Both carry ranks, not counts: a word's rank orders
log(1 + its occurrences) x the number of titles that use it, a kanji's the same over the raw text of all
titles (Jiten.Core/JitenHelper.cs at Sirush/Jiten 7eac4d4, ApplyAggregatesAndRank and
ComputeKanjiFrequencies). So a list gives character frequencies only through a model: a word of rank r
weighs r^-s and so does each character of its form (docs/synth4-gap.md section 1), s fitted so that the kanji
distribution it implies is closest (KL) to the kanji of the training-side real labels. The rates are per
1,000 characters of the list's expanded word forms (dictionary spellings, no punctuation).

Order (owner 10:02Z: the Kanji CSV is the primary kanji source; where it and the VN counts disagree, VN wins
for ordering): data/vocab.txt's 486 non-kanji (kana, punctuation, ASCII and full-width Latin and digits:
every one has renders), then the kanji by VN-list weight, ties by Kanji CSV rank; then the kanji the VN list
never uses, by Kanji CSV rank; then data/vocab.txt's remaining kanji. A vocab of N classes (CTC blank
included) is the first N - 2 characters of that order plus <unk> (〓).

Emittable. A reference character is emittable when every character of its eval/metrics.norm() (NFKC, folded
dashes, tildes and ellipses) is produced by some class's norm(): the score's own rule. The training labels
follow realdata/vote.canon (full-width Latin and digits written as ASCII), so an ASCII class reads both.

The per-deck VN sample (2,162 decks fetched 2026-10-14 06:35-10:29Z before the owner stopped the per-deck
fetch; data/jiten_fetch.py) is used only for the per-VN tail (`tail`): each title's own rate, its percentiles,
and two checks (the aggregate model against the sample's measured rate; titles of reserved VNs against the
rest). It enters no curve and no ordering.
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
for _p in (HERE, ROOT / "synth", ROOT / "eval", ROOT / "realdata", ROOT / "vita", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import align_text, norm  # noqa: E402

HERD = Path("/local/home/skerraut/herd-vita-ocr")
A = HERD / "shared" / "artifacts"
FL10_VOCAB = A / "train2" / "release" / "FL10_vocab_3500.txt"
H15_VOCAB = ROOT / "data" / "vocab.txt"
PROOF_VOCAB = ROOT / "data" / "vocab_4987_proof.txt"
UNK = "〓"
NAME_IDS = (5_000_000, 6_000_000)          # JMnedict entry ids in Jiten's word table: names
LISTS = {"VN": "VisualNovel", "Novel": "Novel", "Anime": "Anime", "Drama": "Drama", "Video game": "VideoGame",
         "Global": "global"}
SIZE_GRID = tuple(range(3000, 8001, 100))   # output classes, CTC blank included
TIER_SIZES = (3501, 4987, 5631, 6459, 7501)
COST_SIZES = (3501, 4987, 5630, 6500, 7500)  # the class counts of the random-init cost grid (`blobs`)
S_GRID = tuple(round(0.70 + 0.05 * i, 2) for i in range(15))


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
    """synth4_gap.reserved_vns(): the test games, the reservation (docs/reservations.md), the CLANNAD exclusion
    manifest, plus extra lists."""
    import synth4_gap
    return synth4_gap.reserved_vns(extra)


def is_reserved(vns, reserved):
    from common import is_reserved_vn
    return any(v in reserved or is_reserved_vn(v) for v in vns)


def default_extra():
    """No extra lists since 2026-10-14 (owner 13:25Z, docs/reservations.md): the development sample is in
    common.is_reserved_vn; the old development list (exclude_dev_vns.txt) and the owner's former test series
    (owner_games_vns.json) are training data. Pass --exclude to add lists."""
    return []


def pct(xs, qs=(50, 90, 95, 99)):
    xs = np.asarray(xs, np.float64)
    return {f"p{q}": round(float(np.percentile(xs, q)), 4) for q in qs} if len(xs) else {}


def named_vocabs(specs):
    """NAME=FILE specs -> {name: vocab}."""
    out = {}
    for spec in specs or []:
        name, _, p = spec.partition("=")
        out[name] = read_vocab(p)
    return out


# ----------------------------------------------------------------------------- aggregate lists

def load_list(path):
    """(word form, rank) rows of a Jiten word list CSV (Word,Form,Rank)."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        head = next(r)
        iw, ir = head.index("Word"), head.index("Rank")
        for row in r:
            if len(row) > max(iw, ir) and row[ir].strip().isdigit() and row[iw].strip():
                rows.append((row[iw].strip(), int(row[ir])))
    return rows


def load_kanji_csv(path):
    """{kanji: rank} of Jiten's kanji list CSV (Kanji,Rank); a repeated kanji keeps its best rank."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        head = next(r)
        ik, ir = head.index("Kanji"), head.index("Rank")
        for row in r:
            if len(row) > max(ik, ir) and row[ir].strip().isdigit() and is_kanji(row[ik].strip()[:1] or " "):
                k = row[ik].strip()
                out[k] = min(out.get(k, 10 ** 9), int(row[ir]))
    return out


def char_weights(rows, s):
    """Character weights of running text implied by a list's ranks: a word of rank r weighs r^-s, and so does
    each character of its form (synth4_gap.jiten_char_weights)."""
    w = Counter()
    for word, rank in rows:
        x = rank ** -s
        for c in word:
            w[c] += x
    return w


def missed_rate(w, es, kanji_only=False):
    """Per 1,000 characters (or kanji) of w: the weight of characters the emit set cannot produce."""
    tot = miss = 0.0
    for c, x in w.items():
        if kanji_only and not is_kanji(c):
            continue
        tot += x
        if not emittable(c, es):
            miss += x
    return 1000 * miss / tot if tot else None


def jis_x0213(c):
    """True for a character of JIS X 0213 (which contains JIS X 0208): the Japanese kanji sets."""
    try:
        c.encode("euc_jis_2004")
        return True
    except UnicodeEncodeError:
        return False


def build_order(w_vn, kanji_rank, h15, japanese=None):
    """(non-kanji block, kanji order, counts): VN-list weight first (ties: Kanji CSV rank), then the Kanji CSV's
    other kanji: those with a Japanese context (`japanese`: the kanji of Jiten's global word list and of VNDB
    character names) and in JIS X 0213 first, by rank (the all-media list also ranks Chinese-only characters
    from Chinese text in some titles), then data/vocab.txt's remaining kanji in its order."""
    vn_k = sorted((c for c in w_vn if is_kanji(c)), key=lambda c: (-w_vn[c], kanji_rank.get(c, 10 ** 9), c))
    seen = set(vn_k)
    jp = japanese if japanese is not None else set(kanji_rank)
    rest = sorted((c for c in kanji_rank if c not in seen),
                  key=lambda c: (0 if c in jp and jis_x0213(c) else 1 if jis_x0213(c) else 2, kanji_rank[c], c))
    seen |= set(rest)
    left = [c for c in h15 if is_kanji(c) and c not in seen]
    idx = {c: i for i, c in enumerate(h15)}
    nk = sorted((c for c in h15 if not is_kanji(c)), key=lambda c: (-w_vn.get(c, 0.0), idx[c]))
    return nk, vn_k + rest + left, {"vn_list": len(vn_k), "kanji_csv_only": len(rest),
                                    "kanji_csv_only_not_jis": sum(1 for c in rest if not jis_x0213(c)),
                                    "kanji_csv_only_japanese": sum(1 for c in rest if c in jp and jis_x0213(c)),
                                    "vn_list_not_jis": "".join(c for c in vn_k if not jis_x0213(c)),
                                    "vocab_txt_only": len(left)}


def vocab_of_size(order, n_classes, unk=True, h15=None):
    """A vocab of n_classes output classes (CTC blank included): the order's first characters, then <unk>.
    (h15 is accepted for older callers; the order already holds data/vocab.txt's characters.)"""
    n = n_classes - 1 - (1 if unk else 0)
    pool = list(order["non_kanji"]) + list(order["kanji"])
    if n > len(pool):
        raise ValueError(f"only {len(pool)} ordered characters for {n_classes} classes")
    out = pool[:n] + ([UNK] if unk else [])
    assert len(out) == len(set(out))
    return out


def sized(order, sizes):
    """[(n, vocab)] for the sizes the order can fill (a small order skips the larger ones)."""
    out = []
    for n in sizes:
        try:
            out.append((n, vocab_of_size(order, n)))
        except ValueError:
            pass
    return out


def spearman(a, b):
    """Spearman rank correlation of two equal-length sequences (average ranks for ties)."""
    def ranks(x):
        x = np.asarray(x, np.float64)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / cnt)[inv]
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def list_manifest(lists_dir):
    """{list name: record} from data/jiten_fetch.py's fetch_log.jsonl (URL, sha256, retrieval time)."""
    log = Path(lists_dir).parent / "fetch_log.jsonl"
    recs = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("kind") == "list" and r.get("file"):
                recs[r["name"]] = {k: r.get(k) for k in ("url", "sha256", "bytes", "rows", "retrieved", "http_date")}
    return recs


def cmd_agg(args):
    import synth4_gap
    ld = Path(args.lists)
    h15 = read_vocab(args.vocab)
    reserved = reserved_vns(args.exclude if args.exclude is not None else default_extra())
    man = list_manifest(ld)
    lists = {name: load_list(ld / f"jiten_freq_{f}.csv") for name, f in LISTS.items()}
    kanji_rank = load_kanji_csv(ld / "jiten_freq_kanji.csv")
    for name, f in LISTS.items():
        rec = man.get(f, {})
        if rec.get("sha256") and rec["sha256"] != sha256(ld / f"jiten_freq_{f}.csv"):
            raise SystemExit(f"{f}: file differs from its fetch record")
    # the Zipf exponent: the kanji of the real training-side labels (VN screenshot text, deduplicated per VN)
    real = Counter()
    real_inputs = {}
    for spec in args.real:
        name, rows, dropped = synth4_gap.load_real_labels(spec, reserved)
        for _, lab in rows:
            real.update(lab)
        p = spec.partition("=")[2]
        real_inputs[name] = {"file": p, "sha256": sha256(p), "lines": len(rows), "dropped_reserved": dropped}
    real_kanji = {c: v for c, v in real.items() if is_kanji(c)}
    s, kl_curve = synth4_gap.fit_exponent(lists["VN"], real_kanji, S_GRID)
    W = {name: char_weights(rows, s) for name, rows in lists.items()}
    names = synth4_gap.vndb_names(args.vndb_chars, reserved) if args.vndb_chars and Path(args.vndb_chars).exists() \
        else []
    japanese = {c for w, _ in lists["Global"] for c in w if is_kanji(c)} | {c for nm in names for c in nm
                                                                          if is_kanji(c)}
    nk, korder, counts = build_order(W["VN"], kanji_rank, h15, japanese)
    order = {"kind": "vocab-expand-order", "schema": 1, "zipf_s": s, "non_kanji": nk, "kanji": korder,
             "counts": counts,
             "kanji_weights": {c: [round(1e6 * W["VN"].get(c, 0.0) / sum(W["VN"].values()), 4),
                                   kanji_rank.get(c)] for c in korder}}
    Path(args.order_out).write_text(json.dumps(order, ensure_ascii=False) + "\n", encoding="utf-8")

    fixed = {"FL10": read_vocab(FL10_VOCAB), "H15": h15, "proof (Small)": read_vocab(PROOF_VOCAB)}
    fixed.update(named_vocabs(args.vocab_file))
    out = {"kind": "vocab-expand-agg", "schema": 1,
           "inputs": {"lists": {f: dict(man.get(f, {}), file=str(ld / f"jiten_freq_{f}.csv"),
                                    sha256=sha256(ld / f"jiten_freq_{f}.csv"))
                                for f in list(LISTS.values()) + ["kanji"]},
                      "real": real_inputs, "reserved_vns": len(reserved),
                      "vocab_txt": sha256(args.vocab),
                      "order": {"file": args.order_out, "sha256": sha256(args.order_out)}},
           "licence": "Jiten frequency lists, CC BY-SA 4.0 (https://jiten.moe/frequency-dictionaries); "
                      "derived counts only",
           "zipf": {"s": s, "kl_curve": kl_curve, "real_kanji": int(sum(real_kanji.values()))},
           "lists": {name: {"rows": len(rows), "distinct_chars": len(W[name]),
                            "kanji": sum(1 for c in W[name] if is_kanji(c))} for name, rows in lists.items()},
           "kanji_csv": {"kanji": len(kanji_rank)}, "order_counts": counts, "fixed": {}, "curve": []}

    def rates(vocab, ws=W):
        es = emit_set(vocab)
        return {name: {"chars": round(missed_rate(w, es), 4), "kanji": round(missed_rate(w, es, True), 4)}
                for name, w in ws.items()}
    for name, v in fixed.items():
        out["fixed"][name] = {"classes": len(v) + 1, **rates(v)}
    for n, v in sized(order, sorted(set(SIZE_GRID) | set(TIER_SIZES))):
        out["curve"].append({"classes": n, **rates(v)})
    # exact membership (no norm folding) for FL10 and H15: the full-width Latin synth4-gap counted
    out["exact_membership"] = {name: round(1000 * sum(x for c, x in W["VN"].items() if c not in set(v))
                                           / sum(W["VN"].values()), 4) for name, v in (("FL10", fixed["FL10"]),
                                                                                       ("H15", h15))}
    # the most frequent VN-list characters each tier vocab misses
    out["missed_top"] = {}
    totv = sum(W["VN"].values())
    tiers = [(f"VN {n}", v_) for n, v_ in sized(order, TIER_SIZES[1:])]
    for name, v in [("FL10", fixed["FL10"]), ("H15", h15)] + tiers:
        es = emit_set(v)
        miss = sorted(((c, x) for c, x in W["VN"].items() if not emittable(c, es)), key=lambda t: -t[1])
        out["missed_top"][name] = [[c, round(1e6 * x / totv, 2)] for c, x in miss[:40]]
        out["missed_top"][name + " (count)"] = len(miss)
    # the Kanji CSV against the VN-derived counts
    vn_k = [c for c in korder if W["VN"].get(c, 0) > 0]
    # Jiten's kanji test leaves out the iteration mark 々 (in every vocab here): put it first in the Kanji CSV order
    k_k = ["々"] + sorted((c for c in kanji_rank if c != "々"), key=lambda c: (kanji_rank[c], c))
    wk = {c: x for c, x in W["VN"].items() if is_kanji(c)}
    totk = sum(wk.values())
    chk = []
    for n in (1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000):
        tv, tk = set(vn_k[:n]), set(k_k[:n])
        chk.append({"kanji": n, "vn_order_missed_per_1000_kanji": round(1000 * (1 - sum(wk.get(c, 0) for c in tv)
                                                                               / totk), 4),
                    "kanji_csv_order_missed_per_1000_kanji": round(1000 * (1 - sum(wk.get(c, 0) for c in tk) / totk),
                                                                   4),
                    "overlap": round(len(tv & tk) / n, 4)})
    both = [c for c in vn_k if c in kanji_rank]
    vr = {c: i + 1 for i, c in enumerate(vn_k)}
    out["kanji_csv_check"] = {
        "curve": chk, "vn_kanji": len(vn_k), "vn_kanji_in_kanji_csv": len(both),
        "vn_kanji_not_in_kanji_csv": "".join(c for c in vn_k if c not in kanji_rank)[:80],
        "spearman_vn_vs_kanji_csv": round(spearman([vr[c] for c in both], [kanji_rank[c] for c in both]), 4),
        "vn_heavy_csv_low": [[c, vr[c], kanji_rank.get(c)] for c in vn_k[:3000]
                             if kanji_rank.get(c, 10 ** 9) > 5000][:30],
        "csv_heavy_vn_low": [[c, kanji_rank.get(c), vr.get(c)] for c in k_k[:3000] if vr.get(c, 10 ** 9) > 5000][:30]}
    # sensitivity to the exponent: rates and the order's top kanji
    sens = {}
    base_top = set(korder[:4499])
    for ds in (-0.15, 0.15):
        s2 = round(s + ds, 2)
        w2 = {"VN": char_weights(lists["VN"], s2)}
        nk2, ko2, _ = build_order(w2["VN"], kanji_rank, h15, japanese)
        o2 = {"non_kanji": nk2, "kanji": ko2}
        sens[str(s2)] = {"overlap_top4499_kanji": round(len(set(ko2[:4499]) & base_top) / 4499, 4),
                         **{name: rates(v, w2)["VN"] for name, v in fixed.items()},
                         **{f"VN {n}": rates(v_, w2)["VN"] for n, v_ in sized(o2, TIER_SIZES[1:])}}
    out["sensitivity"] = sens
    # VNDB character names (usable VNs only): names holding a character a vocab cannot emit
    if names:
        nrow = {}
        for name, v in list(fixed.items()) + [(f"VN {n}", v_) for n, v_ in sized(order, TIER_SIZES[1:])]:
            es = emit_set(v)
            bad = [nm for nm in names if not all(emittable(c, es) for c in nm)]
            nrow[name] = {"names": len(names), "with_char_outside": len(bad), "share": round(len(bad) / len(names), 5)}
        out["names"] = {"file": args.vndb_chars, "sha256": sha256(args.vndb_chars), "rows": nrow}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"s = {s}; order: {counts}; FL10 {out['fixed']['FL10']['VN']}, H15 {out['fixed']['H15']['VN']} "
          f"per 1,000 VN chars / kanji", flush=True)


def cmd_build(args):
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))
    v = vocab_of_size(order, args.size, unk=not args.no_unk)
    Path(args.out).write_text("\n".join(v) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(v)} entries, {len(v) + 1} classes, sha256 {sha256(args.out)}", flush=True)


# ----------------------------------------------------------------------------- the per-deck VN sample

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


def cmd_sample(args, types=("VisualNovel",)):
    jd = Path(args.jiten)
    reserved = reserved_vns(args.exclude if args.exclude is not None else default_extra())
    chars, cidx = [], {}
    rows, cols, vals, nvals = [], [], [], []
    decks = []
    for t in types:
        meta_p, wdir = jd / "decks" / f"decks_{t}.json", jd / "words" / t
        if not meta_p.exists() or not wdir.exists():
            continue
        meta = {d["deckId"]: d for d in json.loads(meta_p.read_text(encoding="utf-8"))}
        for p in sorted(wdir.glob("*.csv.gz"), key=lambda p: int(p.name.split(".")[0])):
            d = meta.get(int(p.name.split(".")[0]))
            if d is None:
                continue
            cnt, names, tot = deck_counts(p)
            if not cnt:
                continue
            vn = vndb_ids(d)
            i = len(decks)
            decks.append({"deck": d["deckId"], "type": t, "title": d.get("originalTitle"),
                          "content": int(d.get("characterCount") or 0), "wordform": tot, "vndb": vn,
                          "reserved": bool(vn) and is_reserved(vn, reserved)})
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
    by = Counter(d["reserved"] for d in decks)
    print(f"{len(decks)} decks ({by[True]} of reserved VNs), {len(chars)} distinct characters", flush=True)


class Chars:
    """Per-deck character counts (cmd_sample's npz)."""

    def __init__(self, path):
        z = np.load(path)
        self.rows, self.cols, self.vals, self.names = z["rows"], z["cols"], z["vals"], z["names"]
        self.chars = json.loads(str(z["chars"]))
        self.decks = json.loads(str(z["decks"]))
        self.content = np.asarray([max(1, d["content"]) for d in self.decks], np.float64)
        self.wordform = np.asarray([max(1, d.get("wordform", 0)) for d in self.decks], np.float64)
        self.path = str(path)

    def select(self, types=None, reserved=None):
        return np.asarray([(types is None or d["type"] in types) and (reserved is None or d["reserved"] == reserved)
                           for d in self.decks], bool)

    def per_char(self, mask, weight="corpus"):
        """{char: weight} over the decks in mask: corpus (occurrences) or deck (per-deck share, decks equal)."""
        m = mask[self.rows]
        v = self.vals[m].astype(np.float64)
        if weight == "deck":
            v = v / self.wordform[self.rows[m]] / max(1, mask.sum())
        w = np.bincount(self.cols[m], weights=v, minlength=len(self.chars))
        return {self.chars[i]: float(x) for i, x in enumerate(w) if x > 0}


def deck_missed(ch, vocab_es):
    """Per deck: occurrences of characters the vocab cannot emit, and of those in name entries."""
    ok = np.asarray([emittable(c, vocab_es) for c in ch.chars], bool)
    miss = ~ok[ch.cols]
    n = np.bincount(ch.rows[miss], weights=ch.vals[miss].astype(np.float64), minlength=len(ch.decks))
    nn = np.bincount(ch.rows[miss], weights=ch.names[miss].astype(np.float64), minlength=len(ch.decks))
    return n, nn


def tail_row(ch, mask, missed, names):
    """The per-title distribution over the decks in mask (word-form basis, as the aggregate curves), the
    corpus rate, and the content-character rates the line model needs."""
    if not mask.any():
        return None
    per = 1000 * missed[mask] / ch.wordform[mask]
    per_c = 1000 * missed[mask] / ch.content[mask]
    return {"titles": int(mask.sum()), "corpus": round(float(1000 * missed[mask].sum() / ch.wordform[mask].sum()), 4),
            "title_mean": round(float(per.mean()), 4), **pct(per), "max": round(float(per.max()), 3),
            "titles_with_miss": round(float((missed[mask] > 0).mean()), 4),
            "name_share": round(float(names[mask].sum() / max(1e-9, missed[mask].sum())), 4),
            "content_basis": {"corpus": round(float(1000 * missed[mask].sum() / ch.content[mask].sum()), 4),
                              **pct(per_c)}}


def cmd_tail(args):
    ch = Chars(args.sample)
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))
    vocabs = {"FL10": read_vocab(FL10_VOCAB), "H15": read_vocab(H15_VOCAB), "proof (Small)": read_vocab(PROOF_VOCAB)}
    vocabs.update(named_vocabs(args.vocab_file))
    for n, v in sized(order, (3000, 3501, 4000, 4987, 5631, 6459, 7501, 8001)):
        vocabs[f"VN {n}"] = v
    design, held = ch.select(reserved=False), ch.select(reserved=True)
    out = {"kind": "vocab-expand-tail", "schema": 1,
           "sample": {"file": args.sample, "sha256": sha256(args.sample), "decks": len(ch.decks),
                      "of": args.of, "design_titles": int(design.sum()), "reserved_titles": int(held.sum()),
                      "note": "a sample: the VN decks fetched before the owner stopped the per-deck fetch (10:00Z); "
                              "word-form basis (each form's characters x its occurrences)"},
           "order": {"file": args.order, "sha256": sha256(args.order)}, "vocabs": {}}
    wf_ratio = float(ch.wordform[design].sum() / ch.content[design].sum())
    out["wordform_per_content_char"] = round(wf_ratio, 4)
    for name, v in vocabs.items():
        missed, names = deck_missed(ch, emit_set(v))
        r = {"classes": len(v) + 1, "design": tail_row(ch, design, missed, names),
             "reserved (OOV rate only)": tail_row(ch, held, missed, names)}
        per = 1000 * missed / ch.wordform
        worst = np.argsort(-np.where(design, per, -1))[:5]
        r["worst_design_titles"] = [[ch.decks[i]["title"], round(float(per[i]), 3)] for i in worst if design[i]]
        out["vocabs"][name] = r
    # reserved-title check: kanji of each tier vocab whose sample occurrences come mostly from reserved titles
    m_res = held[ch.rows]
    occ = np.bincount(ch.cols, weights=ch.vals.astype(np.float64), minlength=len(ch.chars))
    occ_res = np.bincount(ch.cols[m_res], weights=ch.vals[m_res].astype(np.float64), minlength=len(ch.chars))
    idx = {c: i for i, c in enumerate(ch.chars)}
    out["reserved_dominated"] = {}
    prot = {}
    for spec in args.protected or []:
        pname, _, pp = spec.partition("=")
        if Path(pp).exists():
            prot[pname] = load_line_set(pp)
    for name in [k for k in vocabs if k.startswith("VN ") or k.startswith("proof") or k in (args.check or [])]:
        ks = [c for c in vocabs[name] if is_kanji(c) and c in idx]
        dom = [c for c in ks if occ[idx[c]] > 0 and occ_res[idx[c]] / occ[idx[c]] >= 0.8]
        out["reserved_dominated"][name] = {"kanji_in_vocab_seen_in_sample": len(ks), "80pct_from_reserved": len(dom),
                                           "examples": "".join(sorted(dom, key=lambda c: -occ[idx[c]])[:30])}
        # their reach into the protected and development sets: reference characters only (counts, nothing selected)
        dset = set(dom)
        for sname, lines in prot.items():
            cs = [c for t in lines for c in norm(t)]
            out["reserved_dominated"][name].setdefault("protected_chars", {})[sname] = [
                sum(1 for c in cs if c in dset), len(cs)]
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name in ("FL10", "H15", "proof (Small)", "VN 5631", "VN 7501"):
        d = out["vocabs"][name]["design"]
        print(f"{name}: sample corpus {d['corpus']}, title mean {d['title_mean']}, p95 {d['p95']}, "
              f"max {d['max']} per 1,000", flush=True)


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


def cmd_lines(args):
    """Per 1,000 dialogue lines: lines holding at least one character a tier's vocab cannot emit. The model
    treats each content character of a line as an independent draw at the vocab's per-character miss rate,
    over the content-length distribution of real VN dialogue lines (training-side realplus v3 labels)."""
    agg = json.loads(Path(args.agg).read_text(encoding="utf-8"))
    tail = json.loads(Path(args.tail).read_text(encoding="utf-8")) if args.tail else None
    rows = (json.loads(x) for x in Path(args.lengths).read_text(encoding="utf-8").splitlines())
    lens = [content_len(r["label"]) for r in rows if r.get("status") == "kept" and r.get("label")]
    lens = [x for x in lens if x > 0]
    out = {"kind": "vocab-expand-lines", "schema": 1,
           "lengths": {"file": args.lengths, "sha256": sha256(args.lengths), "lines": len(lens),
                       "mean": round(float(np.mean(lens)), 2), **pct(lens, (10, 50, 90))}, "vocabs": {}}
    curve = {r["classes"]: r for r in agg["curve"]}
    names = {"FL10": ("fixed", "FL10"), "H15": ("fixed", "H15"), "proof (Small)": ("fixed", "proof (Small)")}
    for n in TIER_SIZES[1:]:
        names[f"VN {n}"] = ("curve", n)
    for k in agg["fixed"]:
        names.setdefault(k, ("fixed", k))
    for name, (kind, key) in names.items():
        a = agg["fixed"][key] if kind == "fixed" else curve[key]
        q = a["VN"]["chars"] / 1000
        r = {"aggregate_model": round(1000 * predicted_line_share(q, lens), 3)}
        t = tail["vocabs"].get(name) if tail else None
        if t and t.get("design"):
            cb = t["design"]["content_basis"]
            r["sample_corpus"] = round(1000 * predicted_line_share(cb["corpus"] / 1000, lens), 3)
            r["sample_p50_title"] = round(1000 * predicted_line_share(cb["p50"] / 1000, lens), 3)
            r["sample_p95_title"] = round(1000 * predicted_line_share(cb["p95"] / 1000, lens), 3)
        out["vocabs"][name] = r
    sets = {}
    for spec in args.set or []:
        name, _, p = spec.partition("=")
        sets[name] = load_line_set(p)
    if sets:
        out["calibration"] = {}
        for name, v in (("FL10", read_vocab(FL10_VOCAB)), ("H15", read_vocab(H15_VOCAB))):
            es = emit_set(v)
            for sname, lines in sets.items():
                hit = line_hits(lines, es)
                ln = [content_len(s) for s in lines if content_len(s) > 0]
                q = hit["per_1000_chars"] / 1000 * sum(len(norm(s)) for s in lines) / max(1, sum(ln))
                out["calibration"].setdefault(name, {})[sname] = {
                    "observed_per_1000_lines": hit["per_1000_lines"],
                    "independence_per_1000_lines": round(1000 * predicted_line_share(q, ln), 3)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- fonts and contexts

def font_files():
    """(path, category) of every font file of synth4's manifest (data/manifests/synth4_fonts.json)."""
    d = json.loads((ROOT / "data" / "manifests" / "synth4_fonts.json").read_text(encoding="utf-8"))
    roots = {k: v.replace("$HERD", str(HERD)) for k, v in d["roots"].items()}
    return [(Path(roots[f["root"]]) / x["file"], x.get("category") or f.get("category"), f["licence_class"])
            for f in d["families"] for x in f["files"]]


def cmd_fonts(args):
    """For the classes of each vocab file outside data/vocab.txt (the classes no render or NVIDIA crop holds
    yet): how many of synth4's font files draw them, and whether a context exists for the curriculum (a Jiten
    VN or global word whose characters are all in the vocab, or a VNDB character name)."""
    from fontTools.ttLib import TTFont

    import synth4_gap
    cmaps = []
    for path, cat, lic in font_files():
        try:
            cm = TTFont(str(path), lazy=True, fontNumber=0).getBestCmap() or {}
        except Exception:
            cm = {}
        cmaps.append((set(cm), cat, lic))
    h15 = set(read_vocab(H15_VOCAB))
    ld = Path(args.lists)
    vn_words = [w for w, _ in load_list(ld / "jiten_freq_VisualNovel.csv")]
    gl_words = [w for w, _ in load_list(ld / "jiten_freq_global.csv")]
    reserved = reserved_vns(default_extra())
    names = synth4_gap.vndb_names(args.vndb_chars, reserved) if args.vndb_chars else []
    out = {"kind": "vocab-expand-fonts", "schema": 1, "font_files": len(cmaps),
           "smooth_files": sum(1 for _, cat, _ in cmaps if cat != "pixel"), "vocabs": {}}
    for name, v in named_vocabs(args.vocab_file).items():
        vs = set(v)
        new = [c for c in v if c not in h15 and c != UNK]
        ctx_vn = {c for w in vn_words if set(w) <= vs for c in w}
        ctx_gl = {c for w in gl_words if set(w) <= vs for c in w}
        ctx_nm = {c for nm in names for c in nm}
        nf = np.asarray([sum(1 for cm, _, _ in cmaps if ord(c) in cm) for c in new])
        ns = np.asarray([sum(1 for cm, cat, _ in cmaps if cat != "pixel" and ord(c) in cm) for c in new])
        out["vocabs"][name] = {
            "entries": len(v), "new_classes": len(new), "new_kanji": sum(1 for c in new if is_kanji(c)),
            "fonts_p10_50": [int(np.percentile(nf, 10)), int(np.percentile(nf, 50))] if len(new) else None,
            "smooth_fonts_lt5": int((ns < 5).sum()), "smooth_fonts_lt20": int((ns < 20).sum()),
            "no_font": "".join(c for c, k in zip(new, nf) if k == 0)[:60],
            "with_vn_word": sum(1 for c in new if c in ctx_vn), "with_global_word": sum(1 for c in new if c in ctx_gl),
            "with_vndb_name": sum(1 for c in new if c in ctx_nm),
            "no_context": "".join(c for c in new if c not in ctx_vn | ctx_gl | ctx_nm)[:60]}
        print(name, json.dumps(out["vocabs"][name], ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


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
    from common import is_reserved_vn
    ok_vn = lambda v: bool(v) and v not in reserved and not is_reserved_vn(v)  # noqa: E731
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
BLOB_GRID = [(w, b, c) for w in (1.0, 1.5) for b in (64, 96, 128) for c in COST_SIZES] + \
            [(1.0, 48, c) for c in COST_SIZES]


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


def owner_tier(blob_bytes, box3_hi_444):
    """The owner's tier (06:33-07:33Z): weights ceiling and the high bound of the 3-line 640-px box at 444 MHz.
    Small < 1 MB and <= 1 s; Normal <= 4 MB and <= 2 s; Large <= 9 MB and <= 3 s; else none."""
    if box3_hi_444 is None:
        return None
    if blob_bytes < 1_000_000 and box3_hi_444 <= 1.0:
        return "Small"
    if blob_bytes <= 4_000_000 and box3_hi_444 <= 2.0:
        return "Normal"
    if blob_bytes <= 9_000_000 and box3_hi_444 <= 3.0:
        return "Large"
    return "none"


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
            row["tier_budget_doc"] = BS.tier(s, r)          # the always/preflight/never scheme (superseded)
            row["owner_tier"] = owner_tier(s["blob_bytes"], row["box3_s_444"][1])
        out[name] = row
    return out



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

def cmd_tables(args):
    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    k = d.get("kind")
    if k == "vocab-expand-agg":
        names = list(LISTS)
        print(f"Zipf s = {d['zipf']['s']}")
        print("| vocab | classes | " + " | ".join(f"{n}" for n in names) + " | VN kanji |")
        print("|---|---|" + "---|" * len(names) + "---|")
        rows = list(d["fixed"].items()) + [(f"VN-ordered {r['classes']:,}", r) for r in d["curve"]
                                           if r["classes"] in (3000, 3501, 4000, 4500, 4987, 5000, 5631, 6000, 6501,
                                                               7000, 7501, 8000)]
        for name, r in rows:
            print(f"| {name} | {r['classes']:,} | " + " | ".join(f"{r[n]['chars']:.3f}" for n in names)
                  + f" | {r['VN']['kanji']:.3f} |")
        print()
        print("| kanji | VN order: VN kanji missed / 1,000 | Kanji CSV order | overlap |")
        print("|---|---|---|---|")
        for r in d["kanji_csv_check"]["curve"]:
            print(f"| {r['kanji']:,} | {r['vn_order_missed_per_1000_kanji']:.3f} | "
                  f"{r['kanji_csv_order_missed_per_1000_kanji']:.3f} | {r['overlap']:.3f} |")
        print()
        for s, r in d["sensitivity"].items():
            print(f"s = {s}: top-4,499 overlap {r['overlap_top4499_kanji']}; "
                  + ", ".join(f"{n} {v['chars']}" for n, v in r.items() if isinstance(v, dict)))
        if d.get("names"):
            print()
            print("| vocab | VNDB names with a character it cannot emit |")
            print("|---|---|")
            for n, r in d["names"]["rows"].items():
                print(f"| {n} | {r['with_char_outside']:,} of {r['names']:,} ({100 * r['share']:.2f}%) |")
    elif k == "vocab-expand-tail":
        print("| vocab | classes | sample corpus | title mean | p50 | p90 | p95 | p99 | max | titles with a miss "
              "| name share | reserved titles: mean / p95 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n, r in d["vocabs"].items():
            x, y = r["design"], r["reserved (OOV rate only)"]
            print(f"| {n} | {r['classes']:,} | {x['corpus']:.3f} | {x['title_mean']:.3f} | {x['p50']:.3f} | "
                  f"{x['p90']:.3f} | {x['p95']:.3f} | {x['p99']:.3f} | {x['max']:.2f} | {x['titles_with_miss']:.2f} | "
                  f"{x['name_share']:.2f} | {y['title_mean']:.3f} / {y['p95']:.3f} |")
    elif k == "vocab-expand-lines":
        print("| vocab | aggregate model | sample corpus | median title | p95 title |")
        print("|---|---|---|---|---|")
        for n, r in d["vocabs"].items():
            print(f"| {n} | {r['aggregate_model']} | {r.get('sample_corpus', '-')} | {r.get('sample_p50_title', '-')} "
                  f"| {r.get('sample_p95_title', '-')} |")
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
        print("| set | lines | " + " | ".join(f"{v}: per 1,000 lines / chars" for v in d["protected"][
            next(iter(d["protected"]))] if v != "sha256") + " |")
        for name, r in d["protected"].items():
            print(f"| {name} | {r['H15']['lines']:,} | " + " | ".join(
                f"{x['per_1000_lines']} / {x['per_1000_chars']}" for v, x in r.items() if v != "sha256") + " |")
    else:
        print(json.dumps(d, ensure_ascii=False, indent=1)[:4000])


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("agg")
    g.add_argument("--lists", required=True, help="data/jiten_fetch.py's lists directory")
    g.add_argument("--real", action="append", required=True, help="NAME=labels.jsonl|labels.npz (fits s)")
    g.add_argument("--vocab", default=str(H15_VOCAB))
    g.add_argument("--vocab-file", action="append", help="NAME=FILE: also report this vocab")
    g.add_argument("--vndb-chars", default=str(A / "synth3" / "text" / "vndb_chars.tsv"))
    g.add_argument("--exclude", nargs="*", default=None)
    g.add_argument("--out", required=True)
    g.add_argument("--order-out", required=True)
    bu = sub.add_parser("build")
    bu.add_argument("--order", required=True)
    bu.add_argument("--size", type=int, required=True, help="output classes, CTC blank included")
    bu.add_argument("--no-unk", action="store_true")
    bu.add_argument("--out", required=True)
    sm = sub.add_parser("sample")
    sm.add_argument("--jiten", required=True)
    sm.add_argument("--exclude", nargs="*", default=None)
    sm.add_argument("--out", required=True)
    tl = sub.add_parser("tail")
    tl.add_argument("--sample", required=True)
    tl.add_argument("--order", required=True)
    tl.add_argument("--of", type=int, default=2162, help="titles of the population the sample is drawn from")
    tl.add_argument("--vocab-file", action="append")
    tl.add_argument("--check", nargs="*", help="also run the reserved-title check on these named vocabs")
    tl.add_argument("--protected", action="append", help="NAME=FILE: count reserved-dominated kanji in this set")
    tl.add_argument("--out", required=True)
    ln = sub.add_parser("lines")
    ln.add_argument("--agg", required=True)
    ln.add_argument("--tail")
    ln.add_argument("--lengths", default=str(A / "realplus-v3" / "labels.jsonl"))
    ln.add_argument("--set", action="append", help="NAME=FILE: a line set for the independence check")
    ln.add_argument("--out", required=True)
    b = sub.add_parser("bias")
    b.add_argument("--vocab", default=str(H15_VOCAB))
    b.add_argument("--vocab-file", action="append")
    b.add_argument("--out", required=True)
    fo = sub.add_parser("fonts")
    fo.add_argument("--lists", required=True)
    fo.add_argument("--vocab-file", action="append", required=True)
    fo.add_argument("--vndb-chars", default=str(A / "synth3" / "text" / "vndb_chars.tsv"))
    fo.add_argument("--out", required=True)
    bl = sub.add_parser("blobs")
    bl.add_argument("--out", required=True)
    bl.add_argument("--vocab-file", action="append", help="NAME[:WIDTH:BOTTLENECK]=FILE: an exact blob of a vocab")
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
    {"agg": cmd_agg, "build": cmd_build, "sample": cmd_sample, "tail": cmd_tail, "lines": cmd_lines,
     "bias": cmd_bias, "fonts": cmd_fonts, "blobs": cmd_blobs, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()

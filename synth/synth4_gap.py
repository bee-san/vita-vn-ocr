#!/usr/bin/env python3
"""synth4 gap analysis: what the real training-side VN data says synthetic data must supply.

  synth4_gap.py exposures --shards 'S3V3/synth3_000??.npz' --out exposures.json
  synth4_gap.py text --exposures exposures.json --jiten-vn VN.csv --real NAME=FILE ... \
      --vndb-chars vndb_chars.tsv --out text.json
  synth4_gap.py styles --source NAME=GLOB[+GLOB] ... --out styles.json
  synth4_gap.py mine --pool NAME=FILE ... --blob NAME=BLOB ... --out mine.json
  synth4_gap.py tables text.json|styles.json|mine.json     # the markdown tables of docs/synth4-gap.md

Only training-side data is read for anything that steers synth4: realplus (v2 crops, v3 labels),
the native-frames pool's labels, teacher_train / real_train, the Jiten visual-novel word list and
VNDB character names. Rows of every reserved VN are dropped and counted: eval's test games,
`common.is_test_vn`, every synth3 exclusion manifest (fresh native, CLANNAD + family, the owner's
test series) and any extra list given with --exclude (the development VNs, the owner's 13:10Z set).
The development cohorts (teacher_val, devnative) are measured only by `styles` (reported, never a
design input) and never by `mine`.

exposures  per vocab character, over a pinned synth3 render: occurrences in labels, crops holding
           it, distinct font files and families that drew it, and occurrences per platform and
           font category. A character's training exposures in a run are its occurrences times the
           share of the render the run streams (a 16,000-step FL10 pretrain at 96 lines and a 0.5
           synthetic share reads 768,000 crops, 0.77 of a 1,000,000-crop render)
text       the character-frequency tail and vocab coverage:
           - Jiten's visual-novel word list (2,104 VN scripts) gives ranks, not counts; a word of
             rank r is weighted r^-s, with s fitted so that the kanji distribution it implies is
             closest (KL) to the kanji of the real training-side labels (which are VN running
             text, but filtered to data/vocab.txt, so they cannot show what lies beyond it);
             every character of the word (its dictionary form) is counted with that weight
           - coverage of running text by FL10's 3,500 and H15's 5,354 classes, and by the top-N
             VN characters for other N (the vocab-size curve)
           - the share of real labels FL10's vocab cannot emit (measured on the labels directly)
           - VNDB character names whose name holds a character outside each vocab
           - vocab characters of VN text with fewer than N synth3 exposures
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT, ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import norm  # noqa: E402

DEFAULT_EXCLUDE = (ROOT / "data" / "manifests" / "synth3_exclude_fresh_native.json",
                   ROOT / "data" / "manifests" / "synth3_exclude_clannad.json",
                   ROOT / "data" / "manifests" / "synth3_exclude_owner_games.json")
# The conductor's note of the owner's 13:10Z reservation lists 弟切草 (v661) with Aiyoku no Eustia, 428 and
# 街; data/manifests/synth3_exclude_owner_games.json (a40fc23, #63) holds the other three but not v661.
EXTRA_RESERVED = frozenset({"v661"})
EXPOSURE_BINS = (1, 10, 50, 100, 300, 1000)
NAME_STOP = set("「」『』。、！？!?…―（）()")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F \
        or c == "々"


def script(c):
    o = ord(c)
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
    if 0x30A0 <= o <= 0x30FF:
        return "katakana"
    if is_kanji(c):
        return "kanji"
    if c.isascii() and c.isalnum():
        return "latin"
    return "other"


def read_vocab(path):
    return [s for s in Path(path).read_text(encoding="utf-8").split("\n") if s]


def reserved_vns(extra=()):
    """VN ids no statistic may use: eval's test games plus every exclusion manifest's VNs (and
    extra lists: .json with a "vns" list, or whitespace-separated .txt)."""
    out = set(json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())["all"]) | EXTRA_RESERVED
    for p in list(DEFAULT_EXCLUDE) + list(extra):
        p = Path(p)
        if p.suffix == ".json":
            out |= set(json.loads(p.read_text(encoding="utf-8"))["vns"])
        else:
            out |= set(p.read_text().split())
    return out


def usable(vn, reserved):
    from common import is_test_vn
    return bool(vn) and vn not in reserved and not is_test_vn(vn)


def expand(spec):
    out = []
    for g in spec.split("+"):
        hit = sorted(glob.glob(g))
        if not hit:
            raise SystemExit(f"no files match {g}")
        out += hit
    return out


def npz_text(path, members=("labels", "meta")):
    """Only the JSON members of an npz (its pixels are never decompressed)."""
    import zipfile
    out = {}
    with zipfile.ZipFile(path) as zf:
        for m in members:
            name = m + ".npy"
            if name not in zf.namelist():
                continue
            with zf.open(name) as f:
                out[m] = json.loads(str(np.lib.format.read_array(f, allow_pickle=False)))
    return out


# ----------------------------------------------------------------------------- exposures

def shard_exposures(path):
    z = npz_text(path)
    labels, meta = z["labels"], z.get("meta") or [{}] * len(z["labels"])
    occ, crops, plat, cat = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    fonts, fams = defaultdict(set), defaultdict(set)
    n_crops, n_chars = 0, 0
    for lab, m in zip(labels, meta):
        if not lab:
            continue
        n_crops += 1
        n_chars += len(lab)
        c_lab = Counter(lab)
        occ.update(c_lab)
        crops.update(c_lab.keys())
        p, k, f, fam = m.get("platform", "?"), m.get("cat", "?"), m.get("font"), m.get("fam")
        for c, v in c_lab.items():
            plat[c][p] += v
            cat[c][k] += v
            if f:
                fonts[c].add(f)
            if fam:
                fams[c].add(fam)
    return {"occ": occ, "crops": crops, "plat": plat, "cat": cat, "fonts": fonts, "fams": fams,
            "n_crops": n_crops, "n_chars": n_chars}


def cmd_exposures(args):
    import multiprocessing as mp
    paths = expand(args.shards)
    occ, crops, plat, cat = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    fonts, fams = defaultdict(set), defaultdict(set)
    n_crops = n_chars = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for r in pool.imap_unordered(shard_exposures, paths):
            occ.update(r["occ"])
            crops.update(r["crops"])
            for c, v in r["plat"].items():
                plat[c].update(v)
            for c, v in r["cat"].items():
                cat[c].update(v)
            for c, v in r["fonts"].items():
                fonts[c] |= v
            for c, v in r["fams"].items():
                fams[c] |= v
            n_crops += r["n_crops"]
            n_chars += r["n_chars"]
    doc = {"kind": "synth4-gap-exposures", "schema": 1, "shards": [{"file": p, "sha256": sha256(p)} for p in paths]
           if args.hash else [{"file": p} for p in paths],
           "crops": n_crops, "chars": n_chars,
           "chars_by": {c: {"occ": occ[c], "crops": crops[c], "fonts": len(fonts[c]), "families": len(fams[c]),
                            "platform": dict(plat[c]), "category": dict(cat[c])} for c in occ}}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{n_crops} crops, {n_chars} chars, {len(occ)} distinct -> {args.out}", flush=True)


# ----------------------------------------------------------------------------- text

def jiten_words(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 3:
                rows.append((row[0].strip(), int(row[2])))
    return rows


def jiten_char_weights(words, s):
    """Character weights of running text implied by Jiten's ranks: a word of rank r counts r^-s,
    and so does each of its characters."""
    w = Counter()
    for word, rank in words:
        x = rank ** -s
        for c in word:
            w[c] += x
    return w


def kl_div(p, q, keys, eps=1e-12):
    sp = sum(p.get(k, 0.0) for k in keys)
    sq = sum(q.get(k, 0.0) for k in keys)
    out = 0.0
    for k in keys:
        a = p.get(k, 0.0) / sp
        if a > 0:
            out += a * math.log(a / max(q.get(k, 0.0) / sq, eps))
    return out


def fit_exponent(words, real_kanji, grid):
    """The Zipf exponent whose implied kanji distribution is closest to the real labels' kanji
    (KL(real || jiten), over the kanji both hold)."""
    best = None
    curve = []
    for s in grid:
        w = jiten_char_weights(words, s)
        keys = [k for k in real_kanji if w.get(k, 0) > 0]
        d = kl_div(real_kanji, w, keys)
        curve.append([round(s, 3), round(d, 5)])
        if best is None or d < best[1]:
            best = (s, d)
    return best[0], curve


def load_real_labels(spec, reserved):
    """(name, [(vn, label)], dropped rows): kept labels of a labels.jsonl (realplus, native-frames)
    or the labels of an npz (teacher_train, real_train), without reserved VNs; a label is counted
    once per VN (repeated name plates and menu lines would otherwise dominate)."""
    name, _, path = spec.partition("=")
    rows, dropped = [], 0
    seen = set()
    if path.endswith(".jsonl"):
        for x in Path(path).read_text(encoding="utf-8").splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("status") != "kept" or not r.get("label"):
                continue
            vn = r.get("vn")
            if not usable(vn, reserved):
                dropped += 1
                continue
            k = (vn, r["label"])
            if k not in seen:
                seen.add(k)
                rows.append(k)
    else:
        z = npz_text(path)
        for lab, m in zip(z["labels"], z.get("meta") or [{}] * len(z["labels"])):
            vn = m.get("vn")
            if not lab:
                continue
            if not usable(vn, reserved):
                dropped += 1
                continue
            k = (vn, lab)
            if k not in seen:
                seen.add(k)
                rows.append(k)
    return name, rows, dropped


def coverage(weights, vocab_set):
    tot = sum(weights.values())
    cov = sum(v for c, v in weights.items() if c in vocab_set)
    return cov / tot if tot else None


def vndb_names(path, reserved):
    """Japanese character names of VNDB (one per character entry whose VNs are all usable)."""
    out = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(head)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(head) or p[ix["lang"]] != "ja":
                continue
            vns = [v for v in p[ix["vn_ids"]].split(",") if v]
            if not vns or not all(usable(v, reserved) for v in vns):
                continue
            nm = p[ix["name"]].replace(" ", "").replace("　", "")
            if nm:
                out.append(nm)
    return out


def cmd_text(args):
    reserved = reserved_vns(args.exclude)
    v_h15 = read_vocab(args.vocab)
    v_fl10 = read_vocab(args.vocab_fl10)
    S_h15, S_fl10 = set(v_h15), set(v_fl10)
    exp = json.loads(Path(args.exposures).read_text(encoding="utf-8"))
    E = {c: d["occ"] for c, d in exp["chars_by"].items()}
    F = {c: d["fonts"] for c, d in exp["chars_by"].items()}
    FA = {c: d["families"] for c, d in exp["chars_by"].items()}
    stream = args.stream_crops / max(1, exp["crops"])      # share of the render one run reads

    # ---- real training-side labels (VN running text, vocab-filtered by their pipelines)
    real_sets = {}
    real_all = Counter()
    for spec in args.real:
        name, rows, dropped = load_real_labels(spec, reserved)
        cnt = Counter()
        for _, lab in rows:
            cnt.update(lab)
        real_sets[name] = {"file": spec.partition("=")[2], "sha256": sha256(spec.partition("=")[2]),
                           "lines": len(rows), "vns": len({v for v, _ in rows}), "dropped_reserved": dropped,
                           "chars": sum(cnt.values()), "distinct": len(cnt),
                           "outside_h15_per_1000": round(1000 * (1 - coverage(cnt, S_h15)), 3),
                           "outside_fl10_per_1000": round(1000 * (1 - coverage(cnt, S_fl10)), 3),
                           "lines_outside_fl10": sum(1 for _, lab in rows if set(lab) - S_fl10),
                           "lines_outside_h15": sum(1 for _, lab in rows if set(lab) - S_h15)}
        real_all.update(cnt)
    real_kanji = {c: v for c, v in real_all.items() if is_kanji(c)}

    # ---- Jiten VN word list, Zipf weights fitted to the real kanji
    words = jiten_words(args.jiten_vn)
    grid = [round(0.70 + 0.05 * i, 2) for i in range(15)]
    s, curve = fit_exponent(words, real_kanji, grid)
    jw = jiten_char_weights(words, s)
    tot = sum(jw.values())
    kan_tot = sum(v for c, v in jw.items() if is_kanji(c))

    def cov_row(vset):
        cov_all = coverage(jw, vset)
        cov_k = sum(v for c, v in jw.items() if is_kanji(c) and c in vset) / kan_tot
        return {"chars_covered": round(cov_all, 6), "outside_per_1000_chars": round(1000 * (1 - cov_all), 3),
                "kanji_covered": round(cov_k, 6), "outside_per_1000_kanji": round(1000 * (1 - cov_k), 3)}
    vocab_cov = {"FL10_3500": cov_row(S_fl10), "H15_5354": cov_row(S_h15)}
    # the vocab-size curve: every non-kanji character of H15's vocab plus the N most frequent VN kanji
    kanji_order = [c for c, _ in sorted(((c, v) for c, v in jw.items() if is_kanji(c)), key=lambda x: -x[1])]
    non_kanji = {c for c in v_h15 if not is_kanji(c)}
    size_curve = []
    for n_k in (1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 10000):
        vs = non_kanji | set(kanji_order[:n_k])
        r = cov_row(vs)
        r.update({"kanji": n_k, "classes": len(vs)})
        size_curve.append(r)
    # what FL10 misses that H15 has, and what H15 misses, by VN weight
    miss_fl10 = sorted(((c, jw[c] / tot) for c in S_h15 - S_fl10 if c in jw), key=lambda x: -x[1])
    miss_h15 = sorted(((c, v / tot) for c, v in jw.items() if c not in S_h15 and is_kanji(c)), key=lambda x: -x[1])
    in_h15_not_vn = sorted(c for c in S_h15 if is_kanji(c) and jw.get(c, 0) == 0)

    # ---- names
    names = vndb_names(args.vndb_chars, reserved) if args.vndb_chars else []
    name_rows = {}
    for tag, vset in (("FL10_3500", S_fl10), ("H15_5354", S_h15)):
        bad = [n for n in names if set(n) - vset]
        name_rows[tag] = {"names": len(names), "with_char_outside": len(bad),
                          "share": round(len(bad) / max(1, len(names)), 4)}
    name_miss = Counter(c for n in names for c in set(n) if c not in S_h15 and is_kanji(c))

    # ---- the exposure tail of the VN characters in the vocab
    def tail(vocab_list, vset):
        rows = {}
        vn_chars = [c for c in vocab_list if jw.get(c, 0) > 0 or real_all.get(c, 0) > 0]
        for n in EXPOSURE_BINS:
            low = [c for c in vn_chars if E.get(c, 0) < n]
            rows[f"<{n}"] = {"chars": len(low), "kanji": sum(1 for c in low if is_kanji(c)),
                             "vn_text_per_1000": round(1000 * sum(jw.get(c, 0) for c in low) / tot, 4),
                             "real_label_per_1000": round(1000 * sum(real_all.get(c, 0) for c in low)
                                                          / max(1, sum(real_all.values())), 4)}
        never = [c for c in vocab_list if E.get(c, 0) == 0]
        few_fonts = {f"<{k}": sum(1 for c in vocab_list if F.get(c, 0) < k) for k in (2, 5, 10, 20)}
        few_fams = {f"<{k}": sum(1 for c in vocab_list if FA.get(c, 0) < k) for k in (2, 5, 10, 20)}
        occ = np.array([E.get(c, 0) for c in vocab_list], np.float64)
        return {"vocab": len(vocab_list), "vn_chars_in_vocab": len(vn_chars), "tail": rows,
                "never_rendered": len(never), "never_rendered_examples": "".join(never[:60]),
                "exposure_p10_50_90": [int(v) for v in np.percentile(occ, (10, 50, 90))],
                "fonts_lt": few_fonts, "families_lt": few_fams,
                "lowest_vn_weight": [{"char": c, "exposures": E.get(c, 0), "fonts": F.get(c, 0),
                                      "vn_per_million": round(1e6 * jw.get(c, 0) / tot, 3)}
                                     for c in sorted(vn_chars, key=lambda c: (E.get(c, 0), -jw.get(c, 0)))[:40]]}
    out = {
        "kind": "synth4-gap-text", "schema": 1,
        "inputs": {"exposures": {"file": args.exposures, "sha256": sha256(args.exposures), "crops": exp["crops"]},
                   "jiten_vn": {"file": args.jiten_vn, "sha256": sha256(args.jiten_vn), "words": len(words)},
                   "vocab_h15": {"file": args.vocab, "sha256": sha256(args.vocab), "classes": len(v_h15)},
                   "vocab_fl10": {"file": args.vocab_fl10, "sha256": sha256(args.vocab_fl10), "classes": len(v_fl10)},
                   "real": real_sets, "reserved_vns": len(reserved),
                   "vndb_chars": {"file": args.vndb_chars,
                                  "sha256": sha256(args.vndb_chars) if args.vndb_chars else None,
                                  "names": len(names)}},
        "stream_share": round(stream, 4),
        "zipf": {"s": s, "kl_curve": curve},
        "vocab_coverage": vocab_cov,
        "size_curve": size_curve,
        "fl10_missing_top": [{"char": c, "per_million": round(1e6 * v, 2), "exposures": E.get(c, 0)}
                             for c, v in miss_fl10[:60]],
        "fl10_missing_mass_per_1000": round(1000 * sum(v for _, v in miss_fl10), 4),
        "h15_missing_top": [{"char": c, "per_million": round(1e6 * v, 2)} for c, v in miss_h15[:80]],
        "h15_missing_kanji": len(miss_h15),
        "h15_kanji_not_in_vn_list": {"count": len(in_h15_not_vn), "examples": "".join(in_h15_not_vn[:80])},
        "names": name_rows,
        "name_missing_top": [{"char": c, "names": n} for c, n in name_miss.most_common(60)],
        "exposure_tail": {"H15_5354": tail(v_h15, S_h15), "FL10_3500": tail(v_fl10, S_fl10)},
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"zipf s={s}; FL10 covers {vocab_cov['FL10_3500']['chars_covered']:.5f} of VN chars, "
          f"H15 {vocab_cov['H15_5354']['chars_covered']:.5f} -> {args.out}", flush=True)


# ----------------------------------------------------------------------------- styles

def px_ink(g):
    """synth3_stats.px_ink: teacher's refine() rule on the stored grey crop."""
    from synth3_stats import px_ink as f
    return f(g)


def orient(g):
    """The crop with its text brighter than its background (console polarity stores dark text
    inverted, but the inversion rule can miss), the background level and whether it was flipped."""
    g = np.asarray(g, np.float32)
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = float(np.median(ring))
    hi, lo = np.percentile(g, (99, 1))
    if bg - lo > hi - bg:
        return 255.0 - g, 255.0 - bg, True
    return g, bg, False


def runs_1d(mask):
    """Lengths of the True runs along the last axis of a 2-D bool array (flattened per row)."""
    out = []
    for row in mask:
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out.extend((e - s).tolist())
    return out


def pair_asym(g, axis, tol):
    """|share of equal neighbour pairs at even offsets - at odd offsets| along `axis`, over pairs
    that touch text edges: a nearest-neighbour 2x upscale makes one phase's pairs equal."""
    a = np.moveaxis(g, axis, -1)
    if a.shape[-1] < 6:
        return 0.0, 0
    d = np.abs(np.diff(a, axis=-1))
    # only pairs next to a strong edge (within a 4-px window) carry information
    e = d > 24
    near = e.copy()
    for k in (1, 2, 3):
        near[..., k:] |= e[..., :-k]
        near[..., :-k] |= e[..., k:]
    eq = d <= tol
    ev, od = near[..., 0::2], near[..., 1::2]
    n_e, n_o = int(ev.sum()), int(od.sum())
    if n_e < 20 or n_o < 20:
        return 0.0, n_e + n_o
    return abs(float(eq[..., 0::2][ev].mean()) - float(eq[..., 1::2][od].mean())), n_e + n_o


def crop_features(g):
    """Style measurements of one grey crop (console polarity), the same for every source:

      ink_h       pixel-ink height (px_ink, teacher's refine() rule)
      contrast    p99 of the ink rows - background (border-ring median), grey levels
      aa          share of text-edge pixels at intermediate levels: among pixels at least 20% of
                  the way from background to text level, those below 80% (0 for 1-bit text on a
                  clean frame; JPEG, scaling and anti-aliasing raise it)
      stroke      p25 of the horizontal and vertical runs of text pixels (>= 50%) in the ink box,
                  px; stroke_rel = stroke / ink_h
      nearest2x   max over axes of the even/odd asymmetry of equal neighbour pairs at text edges
                  (a nearest-neighbour 2x upscale, as a PSP frame shown on a Vita)
      halo        (outer background level - darkest quarter of the 1-2 px ring around the fill)
                  / contrast: positive when a ring darker than the background surrounds the text
                  (an outline; in console polarity every outline is darker than its fill)
      shadow      (ring above-left - ring below-right) / contrast: positive when the ring is
                  darker below and right of the glyphs (a drop shadow)
      ruby        a separate ink run above the main line, 0.25-0.7 of its height, over at most
                  85% of its width (furigana)
      texture     std of the background pixels (outside the dilated text), grey levels
      bg_trend    |mean of the background's top third - bottom third| / contrast
      flipped     the stored crop's text was darker than its background"""
    import cv2
    g0 = np.asarray(g, np.float32)
    h, w = g0.shape
    if h < 6 or w < 6:
        return None
    g, bg, flipped = orient(g0)
    px = px_ink(g)
    if px is None:
        return None
    x0, y0, x1, y1 = px
    band = g[y0:y1, x0:x1]
    fg = float(np.percentile(band, 99))
    contrast = fg - bg
    if contrast < 12:
        return None
    v = (g - bg) / contrast
    vb = v[y0:y1, x0:x1]
    texty = vb > 0.2
    aa = float(((vb > 0.2) & (vb < 0.8)).sum() / max(1, texty.sum()))
    tm = vb > 0.5
    runs = runs_1d(tm) + runs_1d(tm.T)
    stroke = float(np.percentile(runs, 25)) if runs else 0.0
    ink_h = y1 - y0
    tol = max(2.0, 0.03 * contrast)
    a_h, n_h = pair_asym(g, 1, tol)
    a_v, n_v = pair_asym(g, 0, tol)
    fill = (v > 0.6).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    d2 = cv2.dilate(fill, k3, iterations=2)
    d3 = cv2.dilate(fill, k3, iterations=3)
    d5 = cv2.dilate(fill, k3, iterations=5)
    ring = (d2 > 0) & (fill == 0)
    outer = (d5 > 0) & (d3 == 0)
    halo = shadow = 0.0
    if ring.sum() >= 8 and outer.sum() >= 8:
        halo = (float(np.median(g[outer])) - float(np.percentile(g[ring], 25))) / contrast
        sh_dr = np.zeros_like(fill)
        sh_ul = np.zeros_like(fill)
        sh_dr[2:, 2:] = fill[:-2, :-2]
        sh_ul[:-2, :-2] = fill[2:, 2:]
        dr = (sh_dr > 0) & ring
        ul = (sh_ul > 0) & ring
        if dr.sum() >= 6 and ul.sum() >= 6:
            shadow = (float(np.median(g[ul])) - float(np.median(g[dr]))) / contrast
    # furigana: bands of rows with ink, split at near-empty rows; the main band holds the most
    # ink; ruby is a band right above it, 0.2-0.7 of its height, with less than half its ink,
    # over at most 85% of its width
    m = np.abs(g - bg) > max(28.0, 0.4 * float(np.percentile(np.abs(g - bg), 99)))
    rc = m.sum(1)
    on = rc > max(1, int(0.01 * w))
    ruby = False
    bands = []
    r = 0
    while r < h:
        if on[r]:
            s0 = r
            while r < h and on[r]:
                r += 1
            bands.append((s0, r))
        else:
            r += 1
    if len(bands) >= 2:
        ink_of = [int(rc[a:b].sum()) for a, b in bands]
        k = int(np.argmax(ink_of))
        if k > 0:
            ma, mb = bands[k]
            ra, rb = bands[k - 1]
            mh, rh = mb - ma, rb - ra
            mcols = np.flatnonzero(m[ma:mb].any(0))
            rcols = np.flatnonzero(m[ra:rb].any(0))
            if (0.2 * mh <= rh <= 0.7 * mh and ink_of[k - 1] < 0.5 * ink_of[k] and len(mcols) and len(rcols)
                    and (rcols[-1] - rcols[0] + 1) <= 0.85 * (mcols[-1] - mcols[0] + 1) and ma - rb <= max(3, mh // 2)):
                ruby = True
    bgm = d3 == 0
    texture = float(g[bgm].std()) if bgm.sum() > 20 else 0.0
    t3 = max(1, h // 3)
    top, bot = bgm[:t3], bgm[-t3:]
    trend = 0.0
    if top.sum() > 10 and bot.sum() > 10:
        trend = abs(float(g[:t3][top].mean()) - float(g[-t3:][bot].mean())) / contrast
    return {"h": h, "w": w, "ink_h": int(ink_h), "contrast": round(contrast, 1), "aa": round(aa, 4),
            "stroke": round(stroke, 2), "stroke_rel": round(stroke / max(1, ink_h), 4),
            "nearest2x": round(max(a_h, a_v), 4), "halo": round(halo, 4), "shadow": round(shadow, 4),
            "ruby": ruby, "texture": round(texture, 2), "bg_trend": round(trend, 4), "flipped": flipped}


def synth_truth(m):
    """What the generator drew, from a synth3 crop's metadata."""
    r = m.get("raster") or ""
    return {"pixel_font": m.get("cat") == "pixel", "binary": r.startswith("pixel") or r == "1bpp",
            "nearest2x": m.get("platform") == "psp2x" and "nearest" in (m.get("capture") or "")
            or r == "pixel2", "outline": (m.get("sw") or 0) > 0, "shadow": bool(m.get("shadow")),
            "ruby": bool(m.get("ruby")), "grad": bool(m.get("grad")), "coloured": bool(m.get("coloured")),
            "clean": (m.get("capture") or "").split("+")[-1] in ("clean", "") and m.get("platform") != "scaled"}


def style_kind(m):
    """Capture kind of a crop: native vita / psp (real kind, synth3 platform) or hires (PC shots
    scaled; synth3's scaled renders), plus psp2x for synth3's PSP frames upscaled on a Vita."""
    if m.get("real") or m.get("teacher") or m.get("pool"):
        return m.get("kind") or "?"
    return {"vita": "vita", "psp": "psp", "psp2x": "psp2x", "scaled": "hires"}.get(m.get("platform"), "?")


def _style_rows(task):
    path, idx, keep_meta = task
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(labels)
    out = []
    for i in idx:
        h, w = (int(v) for v in shapes[i])
        f = crop_features(data[offs[i]:offs[i + 1]].reshape(h, w))
        if f is None:
            continue
        m = meta[i]
        lab = labels[i]
        f.update({"target": m.get("target") or ("vita" if m.get("platform") in ("vita", "psp2x") else "psp"),
                  "kind": style_kind(m), "len": len(lab), "vn": m.get("vn"),
                  "namelike": len(lab) <= 6 and not (set(lab) & NAME_STOP)})
        if m.get("platform") == "scaled":
            f["target"] = m.get("target")
        if keep_meta:
            f["truth"] = synth_truth(m)
            f["meta"] = {k: m.get(k) for k in ("platform", "cat", "raster", "capture", "sw", "shadow", "ruby",
                                                "grad", "coloured", "fam", "crop", "bg", "box_alpha", "layout")}
        out.append(f)
    return out


def cmd_styles(args):
    import multiprocessing as mp
    reserved = reserved_vns(args.exclude)
    rng = np.random.default_rng(args.seed)
    res = {"kind": "synth4-gap-styles", "schema": 1, "sources": {}}
    for spec in args.source:
        name, _, globs = spec.partition("=")
        synth = name.startswith("s3") or name.startswith("synth")
        paths = expand(globs)
        rows, dropped = [], 0
        for p in paths:
            z = npz_text(p)
            meta = z.get("meta") or [{}] * len(z["labels"])
            for i, (lab, m) in enumerate(zip(z["labels"], meta)):
                if not lab:
                    continue
                vn = m.get("vn")
                if vn and not usable(vn, reserved):
                    dropped += 1
                    continue
                rows.append((p, i))
        if args.limit and len(rows) > args.limit:
            pick = rng.choice(len(rows), args.limit, replace=False)
            rows = [rows[i] for i in sorted(pick)]
        by = defaultdict(list)
        for p, i in rows:
            by[p].append(i)
        tasks = []
        for p, idx in by.items():
            for k in range(0, len(idx), 2000):
                tasks.append((p, idx[k:k + 2000], synth))
        feats = []
        with mp.get_context("fork").Pool(args.workers) as pool:
            for part in pool.imap_unordered(_style_rows, tasks):
                feats.extend(part)
        res["sources"][name] = {"files": [{"file": p, "sha256": sha256(p)} for p in paths], "rows": len(rows),
                                "measured": len(feats), "reserved_dropped": dropped, "synthetic": synth}
        Path(args.out_dir, f"{name}.features.jsonl").write_text(
            "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in feats), encoding="utf-8")
        print(f"{name}: {len(feats)} of {len(rows)} crops measured ({dropped} reserved rows dropped)", flush=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- mining

def cmd_minepool(args):
    """Crops of the realplus v3 lines no model has trained on: kept in v3's labels (Claude and
    Luna, or both engine families, agree), not kept in v2's labels (which every VOCR3 fine-tune
    read), cut by main's realplus/pool.py exactly as realplus builds its pool (ink crops at Vita
    and PSP scale, GRAD crops where the console finder isolates the line), without the lines whose
    final crop still waits for an edge check. Reserved and development VNs are dropped."""
    import os
    os.environ["REALPLUS"] = str(args.realplus)
    sys.path.insert(0, str(ROOT / "realplus"))
    sys.path.insert(0, str(ROOT))
    import pool as RP
    reserved = reserved_vns(args.exclude)
    v2_kept = {json.loads(x)["id"] for x in Path(args.v2_labels).read_text(encoding="utf-8").splitlines()
               if x.strip() and json.loads(x).get("status") == "kept"}
    labels = [json.loads(x) for x in Path(args.realplus, "labels.jsonl").read_text(encoding="utf-8").splitlines()
              if x.strip()]
    new = [r for r in labels if r.get("status") == "kept" and r["id"] not in v2_kept]
    drop_res = [r for r in new if not usable(r.get("vn"), reserved)]
    new = [r for r in new if usable(r.get("vn"), reserved)]
    shots = RP.load_reads(Path(args.realplus) / "reads" / "reads.jsonl")
    for sk, s in shots.items():         # reads rows hold the screen path; build_lines wants the shot manifest's
        s.setdefault("kind", "vita")
    trained_shots = set()
    for p in args.trained or []:        # screenshots any VOCR3 candidate fine-tuned on: their lines are not new
        for m in npz_text(p, ("meta",))["meta"]:
            if m.get("id"):
                trained_shots.add(m["id"].rsplit("_", 1)[0])
    n_before = len(new)
    new = [r for r in new if r["shot"] not in trained_shots]
    files, man, drops, kept = RP.build_lines(new, shots, procs=args.workers)
    if args.trained:
        drops["trained-shot"] = n_before - len(new)
    if args.native:                     # native console frames: keep each line's crop at its own console's scale
        kind_of = {sk: s.get("source_kind") or s.get("kind") for sk, s in shots.items()}
        for key in list(files):
            target = key[0]
            rows = []
            for img, lab, meta in files[key]:
                k = kind_of.get(meta["id"].rsplit("_", 1)[0])
                if k == target:
                    rows.append((img, lab, dict(meta, kind=k)))
            files[key] = rows
    out = {"kind": "synth4-minepool", "new_kept": len(new) + len(drop_res), "reserved_dropped": len(drop_res),
           "lines": len(kept), "drops": dict(drops), "files": {}}
    if args.native:
        out["native"] = True
    for (target, crop), rows in sorted(files.items()):
        if not rows:
            continue
        p = Path(args.out_dir) / f"{args.prefix}_{target}_{crop}.npz"
        RP.save_npz(p, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
        out["files"][f"{target}_{crop}"] = {"file": str(p), "sha256": sha256(p), "crops": len(rows)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "files"}), flush=True)


def mine_tags(m, lab, f, ranks):
    """Strata of a mining line from its own metadata, label and crop measurements (never from a
    model's output)."""
    t = [f"target:{m.get('target')}", f"kind:{m.get('kind')}", f"crop:{m.get('crop')}"]
    n = len(lab)
    t.append("len:1-4" if n <= 4 else "len:5-12" if n <= 12 else "len:13+")
    if n <= 6 and not set(lab) & NAME_STOP:
        t.append("namelike")
    if m.get("dark"):
        t.append("dark-text")
    if f:
        ih = f["ink_h"]
        t.append("ink:<14" if ih < 14 else "ink:14-19" if ih < 20 else "ink:20-27" if ih < 28 else "ink:28+")
        if f["aa"] < 0.3:
            t.append("binary-like")
        if f["halo"] > 0.12:
            t.append("halo")
        if f["ruby"]:
            t.append("ruby")
        if f["texture"] > 20:
            t.append("busy-bg")
    if any(is_kanji(c) and ranks.get(c, 99999) > 2000 for c in lab):
        t.append("rare-kanji")
    if any(c.isascii() and c.isalnum() for c in lab) or any("０" <= c <= "ｚ" for c in lab):
        t.append("latin-digits")
    return t


def cmd_mine(args):
    """Read every mining crop with each blob through the C runtime (W8I, console polarity, as the
    console reads it) and attribute errors to reference characters (eval's alignment)."""
    sys.path.insert(0, str(ROOT / "runtime"))
    from metrics import align_text, lev
    from pyvocr import Vocr
    ranks = json.loads((ROOT / "data" / "manifests" / "fl10_kanji_policy_20261011.json").read_text())["ranks"]
    rows = []
    for spec in args.pool:
        name, _, path = spec.partition("=")
        z = np.load(path)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        labels, meta = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
        for i, (lab, m) in enumerate(zip(labels, meta)):
            h, w = (int(v) for v in shapes[i])
            g = data[offs[i]:offs[i + 1]].reshape(h, w)
            f = crop_features(g)
            rows.append({"pool": name, "id": m.get("id"), "vn": m.get("vn"), "ref": norm(lab), "img": g,
                         "tags": mine_tags(m, lab, f, ranks)})
    res = {"kind": "synth4-gap-mine", "schema": 1,
           "pools": {s.partition("=")[0]: {"file": s.partition("=")[2], "sha256": sha256(s.partition("=")[2])}
                     for s in args.pool},
           "lines": len(rows), "chars": sum(len(r["ref"]) for r in rows), "vns": len({r["vn"] for r in rows}),
           "models": {}}
    for spec in args.blob:
        name, _, path = spec.partition("=")
        rt = Vocr(path, mode="w8i", max_width=1024)
        ref_n, err, sub, dele = Counter(), Counter(), Counter(), Counter()
        conf = Counter()
        ins = Counter()
        tag_e, tag_n = Counter(), Counter()
        per_line = []
        for r in rows:
            p = norm(rt.text(r["img"]))
            e = lev(p, r["ref"])
            per_line.append(e)
            for t in r["tags"]:
                tag_e[t] += e
                tag_n[t] += len(r["ref"])
            for a, b in align_text(r["ref"], p):
                if a is not None:
                    ref_n[a] += 1
                    if b is None:
                        err[a] += 1
                        dele[a] += 1
                    elif a != b:
                        err[a] += 1
                        sub[a] += 1
                        conf[(a, b)] += 1
                elif b is not None:
                    ins[b] += 1
        n = sum(len(r["ref"]) for r in rows)
        by_script = defaultdict(lambda: [0, 0])
        for c, k in ref_n.items():
            by_script[script(c)][0] += err[c]
            by_script[script(c)][1] += k
        worst = sorted((c for c in ref_n if ref_n[c] >= args.min_count), key=lambda c: -err[c] / ref_n[c])
        res["models"][name] = {
            "blob": path, "blob_sha256": sha256(path), "cer": round(sum(per_line) / max(1, n), 5),
            "exact": round(sum(1 for e in per_line if e == 0) / max(1, len(rows)), 4),
            "strata": {t: {"chars": tag_n[t], "cer": round(tag_e[t] / max(1, tag_n[t]), 5)} for t in sorted(tag_n)},
            "scripts": {k: {"chars": v[1], "err_rate": round(v[0] / max(1, v[1]), 5)} for k, v in by_script.items()},
            "insertions_top": [[c, k] for c, k in ins.most_common(15)],
            "confusions_top": [[a, b, k] for (a, b), k in conf.most_common(60)],
            "char_errors_top": [[c, err[c], ref_n[c]] for c in sorted(err, key=lambda c: -err[c])[:80]],
            "worst_rate": [[c, err[c], ref_n[c]] for c in worst[:60]],
            "char_err": {c: [err[c], ref_n[c]] for c in ref_n},
        }
        print(f"{name}: CER {sum(per_line) / max(1, n):.4f} on {len(rows)} lines", flush=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- style summary

DETECTORS = {           # feature, direction, truth key (synth3 metadata)
    "binary": ("aa", "lt", "binary"),
    "nearest2x": ("nearest2x", "gt", "nearest2x"),
    "outline": ("halo", "gt", "outline"),
    "shadow": ("shadow", "gt", "shadow"),
}


def youden(pos, neg, lower):
    """Threshold maximising TPR - FPR for `feature < t` (lower) or `feature > t`."""
    vals = np.unique(np.percentile(np.concatenate([pos, neg]), np.linspace(1, 99, 197)))
    best = None
    for t in vals:
        tpr = float(np.mean(pos < t)) if lower else float(np.mean(pos > t))
        fpr = float(np.mean(neg < t)) if lower else float(np.mean(neg > t))
        if best is None or tpr - fpr > best[1] - best[2]:
            best = (float(t), tpr, fpr)
    return best


def rogan_gladen(obs, tpr, fpr):
    """Prevalence corrected for a detector's error rates: (obs - FPR) / (TPR - FPR), in [0, 1]."""
    if tpr - fpr < 0.05:
        return None
    return float(min(1.0, max(0.0, (obs - fpr) / (tpr - fpr))))


def load_features(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def cmd_stylesum(args):
    synth = load_features(args.synth)
    det = {}
    for name, (feat, d, truth) in DETECTORS.items():
        pos = np.array([f[feat] for f in synth if f["truth"][truth]])
        neg = np.array([f[feat] for f in synth if not f["truth"][truth]])
        if name == "shadow":            # a shadow without an outline: the outline's halo masks it
            pos = np.array([f[feat] for f in synth if f["truth"]["shadow"] and not f["truth"]["outline"]])
            neg = np.array([f[feat] for f in synth if not f["truth"]["shadow"] and not f["truth"]["outline"]])
        t, tpr, fpr = youden(pos, neg, d == "lt")
        det[name] = {"feature": feat, "dir": d, "threshold": round(t, 4), "tpr": round(tpr, 4), "fpr": round(fpr, 4),
                     "pos": int(len(pos)), "neg": int(len(neg))}
        # per capture kind, with the overall threshold
        det[name]["by_kind"] = {}
        for k in sorted({f["kind"] for f in synth}):
            p = np.array([f[feat] for f in synth if f["kind"] == k and f["truth"][truth]])
            n = np.array([f[feat] for f in synth if f["kind"] == k and not f["truth"][truth]])
            if len(p) >= 30 and len(n) >= 30:
                hit = (lambda a: a < t) if d == "lt" else (lambda a: a > t)
                det[name]["by_kind"][k] = {"tpr": round(float(np.mean(hit(p))), 4),
                                           "fpr": round(float(np.mean(hit(n))), 4)}
    rt = synth  # ruby: flag detector
    rp = [f["ruby"] for f in rt if f["truth"]["ruby"]]
    rn = [f["ruby"] for f in rt if not f["truth"]["ruby"]]
    det["ruby"] = {"feature": "ruby", "tpr": round(float(np.mean(rp)), 4), "fpr": round(float(np.mean(rn)), 4),
                   "pos": len(rp), "neg": len(rn)}
    out = {"kind": "synth4-gap-stylesum", "schema": 1, "synth_reference": args.synth, "detectors": det, "sources": {}}
    for spec in [f"s3v3={args.synth}"] + list(args.source):
        name, _, path = spec.partition("=")
        fs = load_features(path)
        groups = defaultdict(list)
        for f in fs:
            groups["all"].append(f)
            groups[f"{f['target']}:{f['kind']}"].append(f)
            if f.get("namelike"):
                groups["namelike"].append(f)
        res = {}
        for g, rows in sorted(groups.items()):
            if len(rows) < 30:
                continue
            r = {"crops": len(rows)}
            for key in ("ink_h", "stroke", "stroke_rel", "aa", "contrast", "texture", "bg_trend", "halo", "shadow"):
                r[key] = [round(float(v), 3) for v in np.percentile([f[key] for f in rows], (10, 50, 90))]
            r["flipped"] = round(float(np.mean([f["flipped"] for f in rows])), 4)
            r["namelike"] = round(float(np.mean([f["namelike"] for f in rows])), 4)
            for dn, dd in det.items():
                if dn == "ruby":
                    obs = float(np.mean([f["ruby"] for f in rows]))
                else:
                    a = np.array([f[dd["feature"]] for f in rows])
                    hit = a < dd["threshold"] if dd["dir"] == "lt" else a > dd["threshold"]
                    obs = float(np.mean(hit))
                kk = g.split(":")[-1]
                tf = dd.get("by_kind", {}).get(kk, dd)
                r[f"{dn}_obs"] = round(obs, 4)
                est = rogan_gladen(obs, tf["tpr"], tf["fpr"])
                r[f"{dn}_est"] = None if est is None else round(est, 4)
            if rows[0].get("truth"):
                for tk in ("binary", "pixel_font", "nearest2x", "outline", "shadow", "ruby", "grad", "coloured"):
                    r[f"{tk}_truth"] = round(float(np.mean([f["truth"][tk] for f in rows])), 4)
            res[g] = r
        out["sources"][name] = res
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(det, indent=0)[:1500])


# ----------------------------------------------------------------------------- colour

def colour_features(rgb, g):
    """Text-fill colour of one crop: rgb is the crop as the screen shows it, g its grey crop in
    console polarity (for the text mask). coloured: the fill's median HSV saturation > 0.35 at a
    median value > 0.25 (not white, grey or black); part: 15-85% of the fill pixels are saturated
    (a coloured span, or a coloured name in a line); grad: the fill's luminance changes from the
    top third of the ink rows to the bottom third by more than 0.15 of the text contrast, with a
    row correlation |r| > 0.4 (a vertical gradient fill)."""
    gg, bg, _ = orient(g)
    px = px_ink(gg)
    if px is None:
        return None
    x0, y0, x1, y1 = px
    fg = float(np.percentile(gg[y0:y1, x0:x1], 99))
    contrast = fg - bg
    if contrast < 12:
        return None
    v = (gg - bg) / contrast
    fill = v > 0.7
    fill[:y0] = False
    fill[y1:] = False
    if fill.sum() < 12:
        return None
    px_rgb = rgb[fill].astype(np.float32) / 255.0
    mx, mn = px_rgb.max(1), px_rgb.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    coloured = bool(np.median(sat) > 0.35 and np.median(mx) > 0.25)
    share = float(np.mean((sat > 0.35) & (mx > 0.25)))
    part = 0.15 <= share <= 0.85
    ys = np.nonzero(fill)[0].astype(np.float32)
    lum = gg[fill].astype(np.float32)
    grad = False
    if y1 - y0 >= 6 and ys.std() > 0:
        r = float(np.corrcoef(ys, lum)[0, 1]) if lum.std() > 0 else 0.0
        t3 = (y1 - y0) / 3.0
        top = lum[ys < y0 + t3]
        bot = lum[ys >= y1 - t3]
        if len(top) > 4 and len(bot) > 4:
            grad = abs(r) > 0.4 and abs(float(top.mean()) - float(bot.mean())) > 0.15 * contrast
    hue = None
    if coloured:
        sel = (sat > 0.35) & (mx > 0.25)
        p = px_rgb[sel]
        mxi = p.argmax(1)
        hue = ["r", "g", "b"][int(np.bincount(mxi, minlength=3).argmax())]
    return {"coloured": coloured, "part": part, "grad": grad, "sat": round(float(np.median(sat)), 3), "hue": hue}


def _real_colour(task):
    import cv2

    from devnative.paths import fit_scale
    path, idx, shots_dir = task
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    meta = json.loads(str(z["meta"]))
    out = []
    cache = {}
    for i in idx:
        m = meta[i]
        if not m.get("box"):
            continue
        sk = m["id"].rsplit("_", 1)[0]
        kind = m["kind"]
        key = (sk, kind, m["target"])
        if key not in cache:
            p = Path(shots_dir) / kind / f"{sk}.jpg"
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                im = im[:, :, ::-1]
                if kind == "hires":
                    h, w = im.shape[:2]
                    s = fit_scale(w, h, m["target"])
                    im = cv2.resize(im, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                                    interpolation=cv2.INTER_AREA)
            cache[key] = im
        im = cache[key]
        if im is None:
            continue
        x0, y0, x1, y1 = m["box"]
        h, w = (int(v) for v in shapes[i])
        rgb = np.ascontiguousarray(im[y0:y1, x0:x1])
        if rgb.shape[:2] != (h, w):
            continue
        g = data[offs[i]:offs[i + 1]].reshape(h, w)
        from common import to_gray
        gg = to_gray(rgb)
        if m.get("inverted"):
            gg = 255 - gg
        if int(np.abs(gg.astype(np.int16) - g.astype(np.int16)).max()) > 1:
            continue                       # the cut is not the stored crop (box convention or rescale)
        f = colour_features(rgb, g)
        if f:
            f.update(kind=kind, target=m["target"])
            out.append(f)
    return out


_SYN = {}


def _synth_colour(task):
    import synth3 as S3
    seed, idx, n_regions = task
    rng = np.random.default_rng([seed, idx])
    out = []
    for _ in range(n_regions):
        try:
            r = S3.render_region(rng)
        except Exception:  # noqa: BLE001
            continue
        if r is None:
            continue
        img, layout, ink, meta = r
        mode = S3.choose(rng, S3.CFG["crop"])
        pad = S3.CFG["pad"]["psp" if meta["target"] == "psp" else "vita"]
        for g, lab, box, L, dark, kind in S3.extract(img, layout, ink, rng, mode, pad, meta.pop("_ruby_mask"),
                                                     meta.pop("_fill_mask")):
            x, y, w, h = box
            f = colour_features(np.ascontiguousarray(img[y:y + h, x:x + w]), g)
            if f:
                f.update(kind=style_kind(meta), target=meta["target"], truth_grad=bool(meta["grad"]),
                         truth_coloured=bool(L.get("coloured")), scheme=meta["scheme"])
                out.append(f)
    return out


def cmd_colour(args):
    import multiprocessing as mp
    reserved = reserved_vns(args.exclude)
    res = {"kind": "synth4-gap-colour", "schema": 1, "sources": {}}
    rows = []
    if args.real:
        z = npz_text(args.real)
        idx = [i for i, m in enumerate(z["meta"]) if usable(m.get("vn"), reserved) and z["labels"][i]]
        tasks = [(args.real, idx[k:k + 1500], args.shots) for k in range(0, len(idx), 1500)]
        with mp.get_context("fork").Pool(args.workers) as pool:
            for part in pool.imap_unordered(_real_colour, tasks):
                rows.extend(("real", f) for f in part)
    if args.synth_regions:
        import synth3 as S3
        S3.init(scene_hashes=args.scene_hashes, targets=args.targets)
        per = 50
        tasks = [(args.seed, k, per) for k in range(max(1, args.synth_regions // per))]
        with mp.get_context("fork").Pool(args.workers) as pool:
            for part in pool.imap_unordered(_synth_colour, tasks):
                rows.extend(("s3v3", f) for f in part)
    groups = defaultdict(list)
    for src, f in rows:
        groups[(src, "all")].append(f)
        groups[(src, f"{f['target']}:{f['kind']}")].append(f)
    for (src, g), fs in sorted(groups.items()):
        r = {"crops": len(fs)}
        for k in ("coloured", "part", "grad"):
            r[k] = round(float(np.mean([f[k] for f in fs])), 4)
        hues = Counter(f["hue"] for f in fs if f["hue"])
        r["hue"] = {k: round(v / max(1, sum(hues.values())), 3) for k, v in hues.items()}
        if "truth_grad" in fs[0]:
            for k in ("truth_grad", "truth_coloured"):
                r[k] = round(float(np.mean([f[k] for f in fs])), 4)
            pos = [f["grad"] for f in fs if f["truth_grad"]]
            neg = [f["grad"] for f in fs if not f["truth_grad"]]
            r["grad_tpr_fpr"] = [round(float(np.mean(pos)), 4) if pos else None,
                                 round(float(np.mean(neg)), 4) if neg else None]
        res["sources"].setdefault(src, {})[g] = r
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res["sources"], ensure_ascii=False)[:3000], flush=True)


# ----------------------------------------------------------------------------- tables

# Head cost per output class, measured in docs/vita-memory-budget.md's sweep (int8 W8I blob bytes
# and the high bound of the A9 estimate at 333 MHz for a 640-px line): width 1.0 FL10 (3,501
# classes) -> w100 (5,355), width 1.5 v3500 -> v7000 -> v11000. The arena does not depend on it.
TIER = {"FL10": {"width": 1.0, "classes": 3501, "blob": 657664, "ms_high": 235.0, "arena": 124832,
                 "per_class_bytes": (859904 - 657664) / (5355 - 3501), "per_class_ms": (280 - 235) / (5355 - 3501),
                 "limit": (900000, 131072, 300.0), "tier": "always safe"},
        "H15": {"width": 1.5, "classes": 5355, "blob": 1155968, "ms_high": 398.0, "arena": 182944,
                "per_class_bytes": (1335600 - 953680) / 3500, "per_class_ms": (535 - 438) / 4000,
                "limit": (1600000, 262144, 550.0), "tier": "safe with the preflight"}}


def tier_cost(model, classes):
    t = TIER[model]
    d = classes - t["classes"]
    return round(t["blob"] + d * t["per_class_bytes"]), round(t["ms_high"] + d * t["per_class_ms"], 1)


def cmd_tables(args):
    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    k = d.get("kind")
    if k == "synth4-gap-text":
        print("| vocab | classes | VN chars outside, per 1,000 | VN kanji outside, per 1,000 |")
        print("|---|---|---|---|")
        for name, r in d["vocab_coverage"].items():
            print(f"| {name} | {name.split('_')[1]} | {r['outside_per_1000_chars']} | {r['outside_per_1000_kanji']} |")
        print()
        print("| VN-ordered vocab: kanji + H15's non-kanji | classes | outside / 1,000 chars "
              "| FL10-topology blob, est. ms | H15-topology blob, est. ms |")
        print("|---|---|---|---|---|")
        for r in d["size_curve"]:
            if r["kanji"] > 6000:
                continue
            b1, m1 = tier_cost("FL10", r["classes"])
            b2, m2 = tier_cost("H15", r["classes"])
            print(f"| {r['kanji']:,} | {r['classes']:,} | {r['outside_per_1000_chars']} | {b1:,} B, {m1} ms | "
                  f"{b2:,} B, {m2} ms |")
        print()
        print("| source | lines | VNs | chars | FL10 cannot emit, per 1,000 | lines with such a char |")
        print("|---|---|---|---|---|---|")
        for name, r in d["inputs"]["real"].items():
            print(f"| {name} | {r['lines']:,} | {r['vns']:,} | {r['chars']:,} | {r['outside_fl10_per_1000']} | "
                  f"{r['lines_outside_fl10']} |")
        print()
        for name, r in d["exposure_tail"].items():
            print(f"{name}: never rendered {r['never_rendered']}, exposure p10/50/90 {r['exposure_p10_50_90']}, "
                  f"fonts<k {r['fonts_lt']}")
            print("| exposures in synth3 v3 (1M crops) | vocab chars of VN text | kanji "
                  "| their VN-text share, per 1,000 |")
            print("|---|---|---|---|")
            for b, x in r["tail"].items():
                print(f"| {b} | {x['chars']:,} | {x['kanji']:,} | {x['vn_text_per_1000']} |")
            print()
    elif k == "synth4-gap-stylesum":
        print("| detector | feature | threshold | TPR | FPR (synth3 v3 truth) |")
        print("|---|---|---|---|---|")
        for n, r in d["detectors"].items():
            print(f"| {n} | {r['feature']} | {r.get('threshold', 'flag')} | {r['tpr']} | {r['fpr']} |")
        print()
        cols = ("binary", "outline", "shadow", "nearest2x", "ruby")
        print("| source | crops | ink h p10/50/90 | contrast p50 | aa p10/50/90 | texture p50 | "
              + " | ".join(f"{c} obs (est)" for c in cols) + " |")
        print("|---|---|---|---|---|---|" + "---|" * len(cols))
        for src, groups in d["sources"].items():
            for g, r in groups.items():
                if g == "namelike":
                    continue
                cells = " | ".join(
                    f"{r[c + '_obs']:.3f} ({'-' if r[c + '_est'] is None else format(r[c + '_est'], '.2f')})"
                    for c in cols)
                ih = "/".join(f"{v:g}" for v in r["ink_h"])
                print(f"| {src} {g} | {r['crops']:,} | {ih} | {r['contrast'][1]:g} | "
                      f"{'/'.join(f'{v:.2f}' for v in r['aa'])} | {r['texture'][1]:.1f} | {cells} |")
    elif k == "synth4-gap-colour":
        print("| source | crops | coloured fill | partly coloured | gradient (detector) | hue r/g/b |")
        print("|---|---|---|---|---|---|")
        for src, groups in d["sources"].items():
            for g, r in groups.items():
                h = r["hue"]
                print(f"| {src} {g} | {r['crops']:,} | {r['coloured']:.3f} | {r['part']:.3f} | {r['grad']:.3f} | "
                      f"{h.get('r', 0):.2f}/{h.get('g', 0):.2f}/{h.get('b', 0):.2f} |")
    elif k == "synth4-gap-mine":
        names = list(d["models"])
        print("| stratum | chars | " + " | ".join(names) + " |")
        print("|---|---|" + "---|" * len(names))
        print("| all | {:,} | ".format(d["chars"])
              + " | ".join(f"{100 * d['models'][n]['cer']:.2f}%" for n in names) + " |")
        for t in d["models"][names[0]]["strata"]:
            print(f"| {t} | {d['models'][names[0]]['strata'][t]['chars']:,} | "
                  + " | ".join(f"{100 * d['models'][n]['strata'][t]['cer']:.2f}%" for n in names) + " |")
        print()
        for n in names:
            m = d["models"][n]
            print(f"{n} scripts: " + ", ".join(f"{s} {100 * v['err_rate']:.2f}% of {v['chars']:,}"
                                               for s, v in m["scripts"].items()))
            print(f"{n} confusions: " + " ".join(f"{a}→{b} {c}" for a, b, c in m["confusions_top"][:30]))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exposures")
    e.add_argument("--shards", required=True)
    e.add_argument("--workers", type=int, default=5)
    e.add_argument("--hash", action="store_true", help="record every shard's sha256")
    e.add_argument("--out", required=True)
    t = sub.add_parser("text")
    t.add_argument("--exposures", required=True)
    t.add_argument("--jiten-vn", required=True)
    t.add_argument("--real", action="append", required=True, help="NAME=labels.jsonl|labels.npz")
    t.add_argument("--vndb-chars")
    t.add_argument("--vocab", default=str(ROOT / "data" / "vocab.txt"))
    t.add_argument("--vocab-fl10", required=True)
    t.add_argument("--stream-crops", type=int, default=768000,
                   help="synthetic crops one run reads (study 1's FL10 pretrain: 16,000 x 96 x 0.5)")
    t.add_argument("--exclude", nargs="*", default=[])
    t.add_argument("--out", required=True)
    s = sub.add_parser("styles")
    s.add_argument("--source", action="append", required=True,
                   help="NAME=GLOB[+GLOB]; a NAME starting with s3/synth is synthetic (its metadata is kept as truth)")
    s.add_argument("--limit", type=int, default=20000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=5)
    s.add_argument("--exclude", nargs="*", default=[])
    s.add_argument("--out-dir", required=True)
    s.add_argument("--out", required=True)
    mp_ = sub.add_parser("minepool")
    mp_.add_argument("--realplus", required=True, help="realplus v3's directory (labels.jsonl, reads/)")
    mp_.add_argument("--v2-labels", required=True, help="realplus v2's labels.jsonl (its kept lines were trained on)")
    mp_.add_argument("--workers", type=int, default=5)
    mp_.add_argument("--native", action="store_true", help="native console frames (keep the console's own scale)")
    mp_.add_argument("--trained", nargs="*", default=[], help="npz pools whose screenshots were trained on")
    mp_.add_argument("--prefix", default="minepool")
    mp_.add_argument("--exclude", nargs="*", default=[])
    mp_.add_argument("--out-dir", required=True)
    mp_.add_argument("--out", required=True)
    mi = sub.add_parser("mine")
    mi.add_argument("--pool", action="append", required=True, help="NAME=NPZ")
    mi.add_argument("--blob", action="append", required=True, help="NAME=BLOB.vocr")
    mi.add_argument("--min-count", type=int, default=20)
    mi.add_argument("--out", required=True)
    ss = sub.add_parser("stylesum")
    ss.add_argument("--synth", required=True, help="the synthetic source's features.jsonl (detector reference)")
    ss.add_argument("--source", action="append", default=[], help="NAME=features.jsonl")
    ss.add_argument("--out", required=True)
    co = sub.add_parser("colour")
    co.add_argument("--real", help="teacher_train-format npz whose meta boxes index screenshots under --shots")
    co.add_argument("--shots", default=str(Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts/realdata/shots")))
    co.add_argument("--synth-regions", type=int, default=0, help="render this many synth3 regions in RGB")
    co.add_argument("--scene-hashes")
    co.add_argument("--targets")
    co.add_argument("--seed", type=int, default=4242)
    co.add_argument("--workers", type=int, default=5)
    co.add_argument("--exclude", nargs="*", default=[])
    co.add_argument("--out", required=True)
    tb = sub.add_parser("tables")
    tb.add_argument("json")
    args = ap.parse_args()
    {"exposures": cmd_exposures, "text": cmd_text, "styles": cmd_styles, "minepool": cmd_minepool,
     "mine": cmd_mine, "stylesum": cmd_stylesum, "colour": cmd_colour, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()

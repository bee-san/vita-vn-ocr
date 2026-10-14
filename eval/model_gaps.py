#!/usr/bin/env python3
"""Where the VOCR3 / VOCR4 recognizers still fail, on development data only (docs/model-tiers.md, part 2).

  model_gaps.py taxonomy --out JSON [--vocr3 DIR] [--vocr4 DIR] [--pools NPZ ...] [--exposures JSON]
  model_gaps.py sheets --vocr3 DIR --out DIR [--mode errors|sample]   contact sheets for an eye review
  model_gaps.py review --sheets DIR --decisions JSONL --out JSON      aggregate an eye review of those sheets
  model_gaps.py tables JSON [--review JSON]                           the markdown tables of docs/model-tiers.md

Data. The development cohorts VOCR3 and VOCR4 select on: teacher_val (the amendment-8 copy, 1,643 lines, 180
VNs) and devnative v3 (strict and tv_native, ink and GRAD crops; 986 crops of 43 VNs). Their per-line readings
are the selection score files (eval/synth3_ablation.py score: the plain-C runtime in W8I, view rv = the console
polarity, as the console reads them): $HERD/shared/artifacts/synth3/vocr3/select/<SIZE>/ for VOCR3 and
$HERD/shared/artifacts/vocr4-train/select/<SIZE>/scores/ for VOCR4 (a run counts once its tv_<run>.json exists,
which vocr4-train writes last). No protected set (vita-test, psp-test, the fresh native textbox test, CLANNAD)
is read; the committed JSON holds aggregate counts and single-character confusion pairs only.

Errors are eval's: metrics.norm on both sides, Levenshtein over reference characters. Each error is placed on
one reference character (a substitution or deletion) or counted as an insertion, by one optimal alignment
(substitution before deletion before insertion on ties), so per-class counts sum to the line's edit distance.

Classes of a reference character, first match wins: oov (the model's vocab cannot emit it), hiragana /
katakana with dakuten or handakuten, small kana, plain hiragana / katakana, the long-vowel mark, kanji by Jiten
rank (data/manifests/fl10_kanji_policy_20261011.json: <= 500, 501-2,000, > 2,000) and separately by how often
the VOCR3 fine-tune's real pools hold it (--pools: teacher_train, real_train, realplus v2 ink), Latin letters,
digits, punctuation, other symbols. After NFKC full-width Latin and digits are ASCII, so they count as Latin
and digits.

Finder against recognizer. devnative cuts each line twice: by hand around its ink (ink) and by the console's
own finder (GRAD), the latter only where the finder returns one box that holds exactly that line. On the lines
with both, errors(GRAD) - errors(ink) is what the console's crop adds; the share of lines without a clean GRAD
box is the finder's own miss rate on these screenshots.
"""
import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "train")]
from metrics import norm  # noqa: E402

COHORTS = ("tv", "dn_strict_ink", "dn_strict_grad", "dn_tv_native_ink", "dn_tv_native_grad")
VOCR3_FILES = {"tv": "teacher_val.json", "dn_strict_ink": "devnative_strict_ink.json",
               "dn_strict_grad": "devnative_strict_grad.json", "dn_tv_native_ink": "devnative_tv_native_ink.json",
               "dn_tv_native_grad": "devnative_tv_native_grad.json"}
# VOCR3's selections and the same-recipe runs of each size (realplus fine-tune, C_mix and A_s2 arms)
SELECTED = {"FL10": "rp_FL10_C_mix_pre128k_seed1", "H15": "rp_H15_C_mix_seed2", "W20": "rp_W20_C_mix_seed1"}
SIZE_OF = {"FL10": dict(blob=657_664, params=579_661, width=1.0, classes=3501),
           "H15": dict(blob=1_155_968, params=1_038_507, width=1.5, classes=5355),
           "W20": dict(blob=1_558_016, params=1_423_499, width=2.0, classes=5355),
           "LW": dict(blob=1_473_280, params=None, width=None, classes=5355)}

SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶ")
PUNCT = set("、。「」『』（）()［］[]【】〈〉《》〔〕｛｝{}・…‥!?！？~〜～―-—─:;：；,.，．'\"“”‘’")


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or c in "々〆ヶ"


def voiced(c):
    d = unicodedata.normalize("NFD", c)
    return len(d) > 1 and d[1] in "\u3099\u309a"


def char_class(c, vocab=None):
    """The coarse class of one (normalised) reference character."""
    if vocab is not None and c not in vocab:
        return "oov"
    o = ord(c)
    if 0x3041 <= o <= 0x309F:
        return "hira_small" if c in SMALL_KANA else "hira_voiced" if voiced(c) else "hiragana"
    if c == "ー":
        return "long_vowel"
    if 0x30A0 <= o <= 0x30FF:
        return "kata_small" if c in SMALL_KANA else "kata_voiced" if voiced(c) else "katakana"
    if is_kanji(c):
        return "kanji"
    if c.isascii() and c.isalpha():
        return "latin"
    if c.isdigit():
        return "digit"
    if c in PUNCT or unicodedata.category(c).startswith("P"):
        return "punct"
    return "symbol"


def kanji_band(c, ranks):
    r = ranks.get(c, 10 ** 9)
    return "kanji_r500" if r <= 500 else "kanji_r2000" if r <= 2000 else "kanji_rare"


def exposure_band(n):
    return "real_0" if n == 0 else "real_1-9" if n < 10 else "real_10-99" if n < 100 else "real_100+"


def align(ref, pred):
    """One optimal Levenshtein alignment: [(op, ref index or None, ref char, pred char)], op in
    {'=', 's', 'd', 'i'}. Ties prefer the diagonal, then a deletion, then an insertion."""
    n, m = len(ref), len(pred)
    D = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        Di, Dp, r = D[i], D[i - 1], ref[i - 1]
        for j in range(1, m + 1):
            a = Dp[j - 1] + (r != pred[j - 1])
            b = Dp[j] + 1
            c = Di[j - 1] + 1
            Di[j] = a if a <= b and a <= c else (b if b <= c else c)
    ops, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != pred[j - 1]):
            ops.append(("=" if ref[i - 1] == pred[j - 1] else "s", i - 1, ref[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(("d", i - 1, ref[i - 1], None))
            i -= 1
        else:
            ops.append(("i", None, None, pred[j - 1]))
            j -= 1
    return ops[::-1]


def pair_kind(r, p):
    """What kind of substitution r -> p is (the confusion families of docs/synth4-gap.md section 4)."""
    if r is None or p is None:
        return "other"
    br, bp = unicodedata.normalize("NFD", r)[0], unicodedata.normalize("NFD", p)[0]
    if br == bp and r != p:
        return "dakuten"                      # が/か, パ/バ, ...
    small = {"ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お", "っ": "つ", "ゃ": "や", "ゅ": "ゆ", "ょ": "よ",
             "ゎ": "わ", "ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ", "ッ": "ツ", "ャ": "ヤ", "ュ": "ユ",
             "ョ": "ヨ", "ヮ": "ワ", "ヵ": "カ", "ヶ": "ケ"}
    if small.get(r) == p or small.get(p) == r:
        return "small_kana"
    kata = lambda c: chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c  # noqa: E731
    if kata(r) == kata(p):
        return "hira_kata"
    if {r, p} <= set("ー一―-~〜"):
        return "dash_like"
    if r in PUNCT and p in PUNCT:
        return "punct_punct"
    if is_kanji(r) and is_kanji(p):
        return "kanji_kanji"
    if (r in PUNCT) != (p in PUNCT):
        return "punct_other"
    return "other"


# ---------------------------------------------------------------- data

def load_npz_meta(path):
    z = np.load(path)
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    shapes = z["shapes"]
    rows = [(m, tuple(int(v) for v in shapes[i])) for i, (lab, m) in enumerate(zip(labels, meta)) if lab]
    return rows


def load_scores(path):
    d = json.loads(Path(path).read_text())
    return d["dev"], d["lines"], d["models"]


def gather(vocr3=None, vocr4=None):
    """{cohort: {"dev", "lines", "meta", "preds": {run: [rv predictions]}}} over every score file found."""
    out = {}
    for size in ("FL10", "H15", "W20"):
        if vocr3:
            for cohort, fn in VOCR3_FILES.items():
                p = Path(vocr3) / size / fn
                if p.exists():
                    add_scores(out, cohort, p, prefix="")
        if vocr4:
            sd = Path(vocr4) / size / "scores"
            for tv in sorted(sd.glob("tv_*.json")) if sd.exists() else ():
                run = tv.name[3:-5]
                add_scores(out, "tv", tv, prefix="")
                for cohort in COHORTS[1:]:
                    p = sd / f"{cohort}_{run}.json"
                    if p.exists():
                        add_scores(out, cohort, p, prefix="")
    return out


def add_scores(out, cohort, path, prefix=""):
    dev, lines, models = load_scores(path)
    c = out.get(cohort)
    if c is None:
        rows = load_npz_meta(dev["file"]) if Path(dev["file"]).exists() else None
        if rows is not None and [m["id"] for m, _ in rows] != [ln["id"] for ln in lines]:
            raise SystemExit(f"{path}: line ids differ from {dev['file']}")
        c = out[cohort] = {"dev": dev, "lines": lines, "meta": rows, "preds": {}}
    elif [ln["id"] for ln in c["lines"]] != [ln["id"] for ln in lines]:
        raise SystemExit(f"{path}: lines differ from the first file of cohort {cohort}")
    for run, m in models.items():
        c["preds"].setdefault(prefix + run, m["pred"]["rv"])


def real_counts(pools):
    """Character occurrences in the real fine-tune pools' labels (normalised), each line once per pool."""
    cnt = Counter()
    for p in pools or ():
        z = np.load(p)
        for lab in json.loads(str(z["labels"])):
            cnt.update(norm(lab))
    return cnt


FL10_VOCAB = None   # set by --fl10-vocab: FL10's 3,500 characters ($HERD/shared/artifacts/train2/release/)


def vocab_for(run):
    """The output vocabulary of a run's size: FL10's 3,500 characters or data/vocab.txt (None: unknown)."""
    if "FL10" in run:
        if FL10_VOCAB is None or not Path(FL10_VOCAB).exists():
            return None
        v = Path(FL10_VOCAB).read_text(encoding="utf-8").splitlines()
    else:
        v = (ROOT / "data/vocab.txt").read_text(encoding="utf-8").splitlines()
    return set(norm("".join(v))) | set(v)


def size_of(run):
    for s in ("FL10", "H15", "W20", "LW"):
        if f"_{s}_" in run or run.endswith(f"_{s}"):
            return s
    return None


def frames_band(shape):
    h, w = shape
    f = -(-int(round(w * 32 / max(h, 1))) // 8)
    return "frames:1-15" if f < 16 else "frames:16-39" if f < 40 else "frames:40-79" if f < 80 else "frames:80+"


# ---------------------------------------------------------------- taxonomy

def per_run(cohorts, run, ranks, rcount, cohort_set=COHORTS):
    """Every count of one run, over the cohorts in cohort_set that hold it."""
    vocab = vocab_for(run)
    tot = Counter()
    by_cohort = defaultdict(Counter)
    by_stratum = defaultdict(Counter)
    by_class = defaultdict(Counter)
    conf = Counter()
    kinds = Counter()
    edge = Counter()
    for name in cohort_set:
        c = cohorts.get(name)
        if not c or run not in c["preds"]:
            continue
        preds = c["preds"][run]
        for k, (ln, pred) in enumerate(zip(c["lines"], preds)):
            ref = ln["ref"]
            ops = align(ref, pred)
            errs = sum(op != "=" for op, *_ in ops)
            tags = list(ln["tags"])
            if c["meta"] is not None:
                m, shape = c["meta"][k]
                tags.append(frames_band(shape))
                tags += [f"style:{s}" for s in (m.get("styles") or [])]
            tags.append(f"cohort:{'tv' if name == 'tv' else 'dn'}")
            for t in [name] + tags + ["all"]:
                s = by_stratum[t]
                s["chars"] += len(ref)
                s["errors"] += errs
                s["lines"] += 1
                s["lines_wrong"] += errs > 0
            by_cohort[name]["chars"] += len(ref)
            by_cohort[name]["errors"] += errs
            tot["chars"] += len(ref)
            tot["errors"] += errs
            pos = 0                                  # reference characters consumed so far
            for op, i, r, p in ops:
                if op == "i":
                    by_class[char_class(p)]["ins"] += 1
                    kinds["insertion"] += 1
                    edge["edge" if pos <= 1 or pos >= len(ref) - 1 else "inner"] += 1
                    continue
                pos += 1
                cl = char_class(r, vocab)
                keys = [cl]
                if cl == "kanji":
                    keys += [kanji_band(r, ranks), "k" + exposure_band(rcount.get(r, 0))]
                for key in keys:
                    b = by_class[key]
                    b["ref"] += 1
                    if op == "s":
                        b["sub"] += 1
                    elif op == "d":
                        b["del"] += 1
                if op == "s":
                    conf[(r, p)] += 1
                    kinds[pair_kind(r, p)] += 1
                elif op == "d":
                    kinds["deletion"] += 1
                if op != "=":
                    edge["edge" if i <= 1 or i >= len(ref) - 2 else "inner"] += 1
    return {"total": dict(tot), "by_cohort": {k: dict(v) for k, v in by_cohort.items()},
            "by_stratum": {k: dict(v) for k, v in by_stratum.items()},
            "by_class": {k: dict(v) for k, v in by_class.items()},
            "confusions": [[r, p, n] for (r, p), n in conf.most_common(60)],
            "confusion_kinds": dict(kinds), "edge": dict(edge)}


def finder_pairs(cohorts, run):
    """devnative lines with both crops: errors on the hand-cut ink crop against the console finder's GRAD crop."""
    out = {}
    for sub in ("strict", "tv_native"):
        ink, grad = cohorts.get(f"dn_{sub}_ink"), cohorts.get(f"dn_{sub}_grad")
        if not ink or not grad or run not in ink["preds"] or run not in grad["preds"]:
            continue
        gi = {ln["id"]: k for k, ln in enumerate(grad["lines"])}
        r = Counter()
        for k, ln in enumerate(ink["lines"]):
            ref = ln["ref"]
            e_ink = sum(op != "=" for op, *_ in align(ref, ink["preds"][run][k]))
            if ln["id"] not in gi:
                r["ink_only_lines"] += 1
                r["ink_only_chars"] += len(ref)
                r["ink_only_errors"] += e_ink
                continue
            g = gi[ln["id"]]
            e_grad = sum(op != "=" for op, *_ in align(ref, grad["preds"][run][g]))
            r["paired_lines"] += 1
            r["paired_chars"] += len(ref)
            r["paired_errors_ink"] += e_ink
            r["paired_errors_grad"] += e_grad
            r["grad_worse_lines"] += e_grad > e_ink
            r["grad_better_lines"] += e_grad < e_ink
        out[sub] = dict(r)
    return out


def dev_score(cohorts, run):
    """VOCR3/VOCR4's development score from rv only (the committed scores average rv and rvn)."""
    tv = cohorts.get("tv")
    if not tv or run not in tv["preds"]:
        return None
    e = c = 0
    for ln, p in zip(tv["lines"], tv["preds"][run]):
        e += sum(op != "=" for op, *_ in align(ln["ref"], p))
        c += len(ln["ref"])
    de = dc = 0
    for name in COHORTS[1:]:
        d = cohorts.get(name)
        if not d or run not in d["preds"]:
            continue
        for ln, p in zip(d["lines"], d["preds"][run]):
            de += sum(op != "=" for op, *_ in align(ln["ref"], p))
            dc += len(ln["ref"])
    return {"tv_cer": e / c, "dn_cer": de / dc if dc else None,
            "score_rv": (e / c + (de / dc if dc else e / c)) / 2}


def cmd_taxonomy(args):
    global FL10_VOCAB
    FL10_VOCAB = args.fl10_vocab
    cohorts = gather(args.vocr3, args.vocr4)
    ranks = json.loads((ROOT / "data/manifests/fl10_kanji_policy_20261011.json").read_text(encoding="utf-8"))["ranks"]
    rcount = real_counts(args.pools)
    runs = sorted({r for c in cohorts.values() for r in c["preds"]})
    focus = [r for r in runs if r in SELECTED.values() or r.startswith(("v4", "rp_"))]
    out = {"kind": "model-gaps", "schema": 1,
           "cohorts": {k: {"file": v["dev"]["file"], "sha256": v["dev"]["sha256"], "lines": len(v["lines"]),
                           "chars": sum(len(ln["ref"]) for ln in v["lines"]),
                           "vns": len({ln["vn"] for ln in v["lines"]})} for k, v in cohorts.items()},
           "real_pools": [{"file": str(p)} for p in (args.pools or [])],
           "real_chars": sum(rcount.values()), "runs": {}}
    for run in focus:
        r = per_run(cohorts, run, ranks, rcount)
        r["finder"] = finder_pairs(cohorts, run)
        r["dev_score_rv"] = dev_score(cohorts, run)
        r["size"] = size_of(run)
        out["runs"][run] = r
        print(f"{run}: {r['total']['errors']}/{r['total']['chars']} = {r['total']['errors'] / r['total']['chars']:.4f}",
              file=sys.stderr, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- eye-review sheets

def cmd_sheets(args):
    """Contact sheets (1280 x 1280 px at most) of the lines the W20 selection reads wrong (--mode errors) or
    of every --every'th line (--mode sample), each crop with the reference (R) and the three VOCR3
    selections' readings (W, H, F), for an eye review of the causes. Development cohorts only; the sheets
    hold crops of copyrighted screenshots, so --out must be a private directory."""
    import random

    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, str(ROOT / "ci"))
    import fonts
    font = ImageFont.truetype(str(fonts.ensure_font()), 15)
    cohorts = gather(args.vocr3, None)
    runs = [SELECTED["W20"], SELECTED["H15"], SELECTED["FL10"]]
    items = []
    for name in COHORTS:
        c = cohorts.get(name)
        if not c:
            continue
        z = np.load(c["dev"]["file"])
        labels = json.loads(str(z["labels"]))
        idx = [i for i, lab in enumerate(labels) if lab]
        for k, ln in enumerate(c["lines"]):
            preds = [c["preds"][r][k] if r in c["preds"] else None for r in runs]
            wrong = preds[0] is not None and preds[0] != ln["ref"]
            if args.mode == "errors" and not wrong:
                continue
            items.append({"cohort": name, "id": ln["id"], "vn": ln["vn"], "ref": ln["ref"], "preds": preds,
                          "tags": ln["tags"], "wrong": wrong, "i": idx[k], "z": c["dev"]["file"]})
    if args.mode == "sample":
        random.Random(args.seed).shuffle(items)
        items = items[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    W, H, col_w, text_h = 1280, 1280, 640, 17
    npz = {}
    sheets, cur, y, col = [], [], 0, 0
    for n, it in enumerate(items, 1):
        z = npz.setdefault(it["z"], np.load(it["z"]))
        i = it["i"]
        h, w = (int(v) for v in z["shapes"][i])
        g = z["data"][z["offsets"][i]:z["offsets"][i + 1]].reshape(h, w)
        scale = min(2.0, 40 / h, (col_w - 8) / w)
        im = Image.fromarray(g).resize((max(1, round(w * scale)), max(1, round(h * scale))))
        cell_h = im.height + 4 * text_h + 8
        if y + cell_h > H:
            col, y = col + 1, 0
            if col == 2:
                sheets.append(cur)
                cur, col = [], 0
        cur.append((n, it, im, col * col_w, y))
        y += cell_h
    if cur:
        sheets.append(cur)
    index = []
    for s_i, cells in enumerate(sheets, 1):
        sheet = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(sheet)
        for n, it, im, x0, y0 in cells:
            sheet.paste(im, (x0 + 4, y0 + 2))
            ty = y0 + im.height + 4
            d.text((x0 + 4, ty), f"#{n} {it['cohort']}  R: {it['ref']}", fill=0, font=font)
            for lab, p in zip(("W", "H", "F"), it["preds"]):
                ty += text_h
                d.text((x0 + 4, ty), f"{lab}: {p if p is not None else '-'}", fill=0, font=font)
            index.append({k: it[k] for k in ("cohort", "id", "vn", "ref", "preds", "tags", "wrong")}
                         | {"n": n, "sheet": f"S{s_i:03d}"})
        sheet.save(out / f"S{s_i:03d}.png", optimize=True)
    (out / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in index))
    print(f"{len(items)} lines on {len(sheets)} sheets in {out}")


def cmd_review(args):
    """Aggregate eye-review decisions (one JSON object per line: n, cause, styles, label_ok) into counts."""
    index = {r["n"]: r for r in map(json.loads, (Path(args.sheets) / "index.jsonl").read_text().splitlines())}
    dec = [json.loads(x) for x in Path(args.decisions).read_text().splitlines() if x.strip()]
    causes, styles, by_cohort = Counter(), Counter(), defaultdict(Counter)
    errors_by_cause = Counter()
    for d in dec:
        it = index[d["n"]]
        e = sum(op != "=" for op, *_ in align(it["ref"], it["preds"][0] or ""))
        causes[d["cause"]] += 1
        errors_by_cause[d["cause"]] += e
        by_cohort[it["cohort"]][d["cause"]] += 1
        for s in d.get("styles", []):
            styles[s] += 1
    out = {"lines": len(dec), "causes": dict(causes), "errors_by_cause": dict(errors_by_cause),
           "styles": dict(styles), "by_cohort": {k: dict(v) for k, v in by_cohort.items()}}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- tables

def per1000(e, n):
    return f"{1000 * e / n:.1f}" if n else "-"


def cmd_tables(args):
    d = json.loads(Path(args.json).read_text())
    sel = [r for r in SELECTED.values() if r in d["runs"]]
    print("| class | ref chars | " + " | ".join(f"{size_of(r)} errors/1,000 (share)" for r in sel) + " |")
    print("|---|---|" + "---|" * len(sel))
    classes = sorted({k for r in sel for k in d["runs"][r]["by_class"]})
    for cl in classes:
        cells = []
        for r in sel:
            b = d["runs"][r]["by_class"].get(cl, {})
            e = b.get("sub", 0) + b.get("del", 0) + b.get("ins", 0)
            tot = d["runs"][r]["total"]["errors"]
            cells.append(f"{per1000(e, b.get('ref', 0))} ({100 * e / tot:.1f}%)")
        print(f"| {cl} | {d['runs'][sel[0]]['by_class'].get(cl, {}).get('ref', 0):,} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("taxonomy")
    t.add_argument("--vocr3")
    t.add_argument("--vocr4")
    t.add_argument("--pools", nargs="*")
    t.add_argument("--fl10-vocab")
    t.add_argument("--out", required=True)
    s = sub.add_parser("sheets")
    s.add_argument("--vocr3", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=("errors", "sample"), default="errors")
    s.add_argument("--limit", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("review")
    r.add_argument("--sheets", required=True)
    r.add_argument("--decisions", required=True)
    r.add_argument("--out", required=True)
    b = sub.add_parser("tables")
    b.add_argument("json")
    args = ap.parse_args()
    {"taxonomy": cmd_taxonomy, "sheets": cmd_sheets, "review": cmd_review, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()

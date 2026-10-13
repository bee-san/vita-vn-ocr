#!/usr/bin/env python3
"""synth4: synth3 version 3 with a rare-character curriculum and a style mix set from real crops.

docs/synth4-gap.md measured the training-side real data against synth3 version 3's pinned render;
synth4 changes what those measurements found, and nothing else (every other probability, the text
pools, scenes, crops and capture chains are synth3 v3's, `synth/synth3.py` unchanged):

  curriculum  every vocab class at least K exposures per 1,000,000 crops (K = 80 for a class VN text
              uses, 25 for the rest): the deficit against synth3 v3's measured exposures is spread over
              the render's 100 shards, and a focus draw takes a class with shard quota left first
              (`curriculum.json`, pinned); focus lines 40% of dialogue (synth3 25%)
  contexts    a quota character's line is a natural pool line (synth3's contexts), a Jiten
              visual-novel word holding it in a short VN carrier sentence, or a VNDB / web-novel name
              holding it (also as a speaker name): real words and names, never random strings
  hard chars  synth3's focus weights with the hard-example weight blended with the error rates of
              the current VOCR3 selections on 2,890 unseen training-side lines (synth4_gap.py mine)
  1-bit text  pixel fonts 4% of native Vita and 8% of native PSP regions (synth3 10% / 28%), 1-bit
              raster 2% / 4% (6% / 28%): real native crops are almost all anti-aliased; the pixel
              families stay, at their exact strikes (10-11 px fonts now also at 2x nearest on Vita)
  fonts       synth3's pinned pool plus the owner's fonts (cinecaption 2.26, NDS12 at its 10-ppem
              strike), licence class "owner": only in --profile personal, never in --profile open
  outlines    synth2's own outline dropped on 45% of the schemes that do not need one, synth3's
              outline rates about halved (synth3 61-73% of crops outlined; real estimates lower)
  contrast    darker dialogue boxes, lighter light boxes, more pure-white fills (synth3's text was
              dimmer and its boxes lighter than real: p50 contrast 134-153 against 145-191)
  colour      coloured spans, lines and names rarer, half the gradient fills, green and cyan accents
              mostly redrawn (synth3 17% coloured fills, a quarter green; real 10.5%, a tenth green)
  scaled text the large renders scaled to the Vita 1.5 px smaller (synth3's were taller than real)

  synth4.py fonts --out data/manifests/synth4_fonts.json       synth3's manifest + the owner's fonts
  synth4.py fonts-verify [--manifest M]
  synth4.py curriculum --exposures E.json --mine MINE.json --out CURR.json --targets-out TARGETS.json
  synth4.py preview --out DIR [--n 16]
  synth4.py shards --out DIR --shards N --per-shard M [--first F] [--workers W] [--seed S]
  synth4.py manifest --dir DIR --out MANIFEST.json

Assets: synth3's (--assets, read only: text pools, lists, scenes), the font manifest's roots (the
owner's files under $HERD/shared/artifacts/synth4-design/fonts, never committed), the curriculum
and targets files (--curriculum, --targets) and a font-coverage cache of synth4's own (--font-cache).
"""
import argparse
import colorsys
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import render as S2  # noqa: E402
import synth3 as S3  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
S4_ART = HERD / "shared" / "artifacts" / "synth4-design"
FONT_MANIFEST4 = ROOT / "data" / "manifests" / "synth4_fonts.json"
OWNER_FONTS = [
    {"family": "owner_cinecaption", "category": "pen", "file": "owner/cinecaption/cinecaption226.ttf",
     "source": "owner-provided file cinecaption226.ttf (name table: しねきゃぷしょん, Version 2.26)",
     "licence_file": None, "pixel_em": None,
     "resembles": "hand-lettered film-subtitle face"},
    {"family": "owner_nds12", "category": "pixel", "file": "owner/nds12/NDS12.ttf",
     "source": "owner-provided archive nds12.zip (NDS12 by Caveras, FontStruct; NDS12.ttf)",
     "licence_file": "owner/nds12/license.txt", "pixel_em": 10,
     "resembles": "Nintendo DS system font; FontStruct grid of 10 bricks per em (409.6 of 4,096 units) with an "
                  "11 + 1 brick ascent and descent: pixel-exact at 10 ppem, a 12-px line cell (the '12')"},
]
OWNER_LICENCE = ("LicenseRef-owner-licensed: the owner (bee-san, 2026-10-13 14:28Z) is licensed to use this font "
                 "for this non-commercial private OCR project; never redistributed, nor its renders; "
                 "--profile personal only")

# synth3 CFG keys synth4 changes (merged by synth3.merge_cfg, which refuses an unknown key)
CFG_PATCH = {
    "category": {"vita": {"gothic": 0.42, "rounded": 0.21, "mincho": 0.19, "pen": 0.06, "display": 0.08,
                          "pixel": 0.04},
                 "psp": {"gothic": 0.45, "rounded": 0.18, "mincho": 0.17, "pen": 0.05, "display": 0.07,
                         "pixel": 0.08}},
    "raster": {"vita": {"aa8": 0.38, "aa4": 0.25, "atlas": 0.27, "ss2": 0.08, "mono": 0.02},
               "psp": {"aa8": 0.24, "aa4": 0.52, "atlas": 0.16, "ss2": 0.04, "mono": 0.04}},
    "focus": 0.40,
    "outline": {"vita": {"light_on_dark": 0.20, "dark_on_light": 0.06, "on_colour": 0.5},
                "psp": {"light_on_dark": 0.25, "dark_on_light": 0.08, "on_colour": 0.5},
                "scaled": {"light_on_dark": 0.35, "dark_on_light": 0.10, "on_colour": 0.6}},
    "coloured": {"span": 0.08, "line": 0.06},
    "name_coloured": 0.30,
    "scaled_em": {"vita": [19.0, 3.3, 13, 30], "psp": [13.5, 2.0, 9, 19]},
    "blur": 0.10,
}
# synth4's own knobs (recorded in the recipe next to synth3's CFG)
CFG4 = {
    "generator": "synth4",
    "version": 1,
    "curriculum": {"k_vn": 80, "k_other": 25, "shards": 100, "quota_share": 0.75},
    "contexts": {"word": 0.35, "name": 0.15},
    "name_focus": 0.3,
    "stroke_drop": 0.45,
    "grad_keep": 0.5,
    "contrast": {"dark_box": [0.3, 0.75], "light_box": [0.3, 0.7], "white_fill": 0.75, "dark_fill": [0.2, 0.7]},
    "accent_fill_drop": 0.4,
    "green_band": [0.17, 0.55],
    "green_redraw": 0.85,
    "pixel_vita_min_em": 10,
    "hard_blend": 0.5,
}
WORDS_PER_CHAR = 24
CARRIERS_NOUN = ["{w}", "{w}……", "{w}？", "{w}！", "{w}って、何？", "{w}だ。", "{w}なんだ。", "{w}か……",
                 "これが{w}か。", "{w}のこと？", "{w}、ですか？", "あの{w}が……", "{w}を見た。", "{w}の話だよ。",
                 "それは{w}だ。", "{w}だって？", "{w}って言った？", "{w}が好き。", "{w}はどこ？", "ああ、{w}ね。",
                 "{w}と{w2}。", "{w2}より{w}がいい。"]
CARRIERS_VERB = ["{w}。", "{w}んだ。", "{w}のか？", "{w}……", "もう{w}。", "{w}かもしれない。", "{w}つもりだ。",
                 "まだ{w}。", "{w}って、本当？"]
CARRIERS_NARR = ["{w}。", "{w}が見えた。", "{w}の前に立つ。", "そこには{w}があった。", "{w}を思い出す。",
                 "目の前の{w}。",
                 "{w}のことを考えていた。"]
NAME_LINES = ["{n}！", "{n}……", "{n}？", "{n}さん！", "{n}、待って！", "{n}の部屋。", "{n}だよ。", "ねえ、{n}。",
              "{n}が来た。", "{n}って誰？"]
VERB_END = set("うくぐすつぬぶむるい")
R4 = {}            # per-process curriculum state, set by init4() before workers fork


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- fonts

def font_roots():
    art = HERD / "shared" / "artifacts"
    return {"pool": HERD / "shared" / "data" / "fonts", "pool_extra": art / "synth2" / "fonts_extra",
            "synth3": art / "synth3" / "fonts", "synth4": S4_ART / "fonts"}


def cmd_fonts(args):
    """synth3's pinned font manifest plus the owner's fonts (root synth4, licence class owner)."""
    from fontTools.ttLib import TTFont
    doc = json.loads((ROOT / "data" / "manifests" / "synth3_fonts.json").read_text(encoding="utf-8"))
    root = font_roots()["synth4"]
    fams = list(doc["families"])
    for o in OWNER_FONTS:
        p = root / o["file"]
        if not p.exists():
            raise SystemExit(f"owner font missing: {p}")
        weight = int(TTFont(str(p), lazy=True)["OS/2"].usWeightClass)
        lic = root / o["licence_file"] if o["licence_file"] else None
        fams.append({"family": o["family"], "category": o["category"], "root": "synth4", "licence_id": OWNER_LICENCE,
                     "licence_class": "owner", "licence_file": o["licence_file"],
                     "licence_sha256": sha256(lic) if lic else None, "licence_url": None, "source": o["source"],
                     "source_revision": None, "resembles": o["resembles"],
                     "files": [{"file": o["file"], "bytes": p.stat().st_size, "sha256": sha256(p),
                                "category": o["category"], "weight": weight, "pixel_em": o["pixel_em"]}]})
    seen = {}
    for fam in fams:
        for f in fam["files"]:
            if f["sha256"] in seen:
                raise SystemExit(f"duplicate font bytes: {f['file']} and {seen[f['sha256']]}")
            seen[f["sha256"]] = f["file"]
    out = {"kind": "synth4-fonts", "schema": 1,
           "roots": dict(doc["roots"], synth4="$HERD/shared/artifacts/synth4-design/fonts"),
           "base_manifest": {"file": "data/manifests/synth3_fonts.json",
                             "sha256": sha256(ROOT / "data" / "manifests" / "synth3_fonts.json")},
           "families": fams,
           "counts": {"families": len(fams), "files": sum(len(f["files"]) for f in fams),
                      "bytes": sum(x["bytes"] for f in fams for x in f["files"]),
                      "nc_families": sorted(f["family"] for f in fams if f["licence_class"] == "nc"),
                      "owner_families": sorted(f["family"] for f in fams if f["licence_class"] == "owner")}}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{out['counts']} -> {args.out}")


def cmd_fonts_verify(args):
    doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    roots = font_roots()
    bad = 0
    for fam in doc["families"]:
        for f in fam["files"]:
            p = roots[fam["root"]] / f["file"]
            if not p.exists() or p.stat().st_size != f["bytes"] or sha256(p) != f["sha256"]:
                print(f"MISMATCH {fam['root']}:{f['file']}")
                bad += 1
    print(f"verified {sum(len(f['files']) for f in doc['families'])} files, {bad} mismatches")
    sys.exit(1 if bad else 0)


def font_root4(self, root):
    return {"pool": S3.SHARED_DATA / "fonts", "pool_extra": S3.HERD / "shared" / "artifacts" / "synth2" / "fonts_extra",
            "synth3": self.assets / "fonts", "synth4": S4_ART / "fonts"}[root]


def px_fits4(font, platform):
    """synth3's rule, except that 10-11 px pixel fonts also serve the Vita at 2x nearest (a PSP-class
    bitmap font on a Vita screen; synth3 kept them off the Vita entirely)."""
    em = font.pixel_em or 0
    return 10 <= em <= 16 if platform in ("psp", "psp2x") else CFG4["pixel_vita_min_em"] <= em <= 20


# ----------------------------------------------------------------------------- curriculum

def is_vn_char(c, jw, real):
    return jw.get(c, 0) > 0 or real.get(c, 0) > 0


def build_curriculum(vocab, exposures, jw, real, words_by_char, mined, base_targets):
    """The quota and contexts of every vocab class, and the focus targets with blended hard weights.

    exposures: synth3 v3's occurrences per class (1,000,000 crops); jw: Jiten VN character weights;
    real: real training-side label counts; words_by_char: {c: [(word, weight)]} Jiten words held
    entirely in the vocab; mined: {c: (errors, occurrences)} summed over the mined models;
    base_targets: synth3 v3's targets.json "chars"."""
    C = CFG4["curriculum"]
    k = {c: (C["k_vn"] if is_vn_char(c, jw, real) else C["k_other"]) for c in vocab}
    deficit = {c: max(0, k[c] - int(exposures.get(c, 0))) for c in vocab}
    hard3 = np.array([base_targets.get(c, {}).get("hard", 0.0) for c in vocab], np.float64)
    rate = np.array([mined[c][0] / (mined[c][1] + 10.0) if c in mined else 0.0 for c in vocab], np.float64)
    h3 = hard3 / max(1e-9, hard3.max())
    hm = rate / max(1e-9, rate.max())
    b = CFG4["hard_blend"]
    hard = (1 - b) * h3 + b * hm
    targets = {}
    for i, c in enumerate(vocab):
        t = dict(base_targets.get(c, {"real": 0.0, "jiten_vn": 0.0}))
        t["hard"] = round(float(hard[i]), 6)
        targets[c] = t
    words = {c: [[w, round(x, 8)] for w, x in sorted(words_by_char.get(c, []), key=lambda t: -t[1])[:WORDS_PER_CHAR]]
             for c in vocab if words_by_char.get(c)}
    return {"k": k, "deficit": deficit, "words": words, "targets": targets,
            "totals": {"deficit": int(sum(deficit.values())),
                       "classes_with_deficit": sum(1 for v in deficit.values() if v),
                       "vn_classes": sum(1 for c in vocab if is_vn_char(c, jw, real))}}


def cmd_curriculum(args):
    import synth4_gap as G
    vocab = S3.load_vocab()
    vset = set(vocab)
    exp = json.loads(Path(args.exposures).read_text(encoding="utf-8"))
    E = {c: d["occ"] for c, d in exp["chars_by"].items()}
    jwords = G.jiten_words(args.jiten_vn)
    s = json.loads(Path(args.text).read_text(encoding="utf-8"))["zipf"]["s"]
    jw = G.jiten_char_weights(jwords, s)
    reserved = G.reserved_vns(args.exclude)
    real = defaultdict(int)
    for spec in args.real:
        _, rows, _ = G.load_real_labels(spec, reserved)
        for _, lab in rows:
            for c in lab:
                real[c] += 1
    words_by_char = defaultdict(list)
    for w, rank in jwords:
        if len(w) < 1 or not all(c in vset for c in w):
            continue
        x = rank ** -s
        for c in set(w):
            words_by_char[c].append((w, x))
    mine = json.loads(Path(args.mine).read_text(encoding="utf-8"))
    mined = defaultdict(lambda: [0, 0])
    for name in args.mine_models:
        for c, (e, n) in mine["models"][name]["char_err"].items():
            mined[c][0] += e
            mined[c][1] += n
    base = json.loads(Path(args.base_targets).read_text(encoding="utf-8"))
    cur = build_curriculum(vocab, E, jw, real, words_by_char, mined, base["chars"])
    tdoc = dict(base, chars=cur.pop("targets"), synth4={"base_targets_sha256": sha256(args.base_targets),
                                                          "mine_sha256": sha256(args.mine), "models": args.mine_models,
                                                          "hard_blend": CFG4["hard_blend"]})
    Path(args.targets_out).write_text(json.dumps(tdoc, ensure_ascii=False) + "\n", encoding="utf-8")
    cur.update({"kind": "synth4-curriculum", "schema": 1, "cfg4_curriculum": CFG4["curriculum"],
                "inputs": {"exposures": {"file": args.exposures, "sha256": sha256(args.exposures)},
                           "jiten_vn": {"file": args.jiten_vn, "sha256": sha256(args.jiten_vn), "zipf_s": s},
                           "text": {"file": args.text, "sha256": sha256(args.text)},
                           "real": [{"spec": r} for r in args.real],
                           "mine": {"file": args.mine, "sha256": sha256(args.mine)},
                           "targets_out_sha256": sha256(args.targets_out),
                           "vocab_sha256": sha256(S3.DATA / "vocab.txt")}})
    Path(args.out).write_text(json.dumps(cur, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"curriculum: {cur['totals']} -> {args.out}; targets -> {args.targets_out}", flush=True)


def shard_targets(idx, seed, deficit, n_shards):
    """Per-shard exposure targets: each class's deficit spread over the render's shards, with
    stochastic rounding drawn from (seed, shard, 4), so a shard's targets never depend on how many
    shards one job renders (resume-safe) and every shard's rounding is independent."""
    rng = np.random.default_rng([seed, idx, 4])
    d = np.asarray(deficit, np.float64) / n_shards
    base = np.floor(d)
    return base + (rng.random(len(d)) < (d - base))


def carrier(rng, word, kind, other=None):
    if kind == "narration":
        t = CARRIERS_NARR[rng.integers(len(CARRIERS_NARR))]
    elif word[-1] in VERB_END:
        t = CARRIERS_VERB[rng.integers(len(CARRIERS_VERB))]
    else:
        t = CARRIERS_NOUN[rng.integers(len(CARRIERS_NOUN))]
        if "{w2}" in t and not other:
            t = "{w}……"
    return t.format(w=word, w2=other or "")


def pick_word(rng, c):
    ws = R4["words"].get(c)
    if not ws:
        return None
    w = np.array([x for _, x in ws], np.float64) ** 0.5
    return ws[int(rng.choice(len(ws), p=w / w.sum()))][0]


def name_holding(rng, c):
    """(src, parts) of a name that holds character c, or None."""
    lst = R4["name_idx"].get(c)
    if not lst:
        return None
    src, i = lst[int(rng.integers(len(lst)))]
    return src, S3.R.names[src][i]


def name_text(parts, c, rng):
    """The display form of a name that keeps character c: the part holding it, or the full name."""
    if len(parts) == 2:
        holding = [p for p in parts if c in p]
        if rng.random() < 0.4 or not holding:
            return parts[0] + ("　" if rng.random() < CFG_NAMES()["space"] else "") + parts[1]
        return holding[0]
    return parts[0]


def CFG_NAMES():  # noqa: N802 - mirrors synth3's CFG["names"] lookups
    return S3.CFG["names"]


def focus_char4(self, rng):
    """A quota class first (while the shard has quota left and with CFG4 quota_share), else
    synth3's tail- and hard-weighted draw."""
    q = getattr(self, "quota_t", None)
    self._from_quota = False
    if q is not None and rng.random() < CFG4["curriculum"]["quota_share"]:
        rem = np.maximum(0.0, q - self.counts)
        rem[~R4["has_any_ctx"]] = 0
        tot = rem.sum()
        if tot > 0:
            self._from_quota = True
            return int(rng.choice(len(rem), p=rem / tot))
    return _ORIG["focus_char"](self, rng)


def context4(self, rng, c, kind="dialogue"):
    """A line holding vocab class c. A quota class (the curriculum's) gets a Jiten VN word in a short
    carrier or a name holding it with CFG4 contexts' probabilities, else synth3's natural pool line
    (and a carrier when no pool line exists)."""
    ch = self.vocab[c]
    if getattr(self, "_from_quota", False):
        r = rng.random()
        C = CFG4["contexts"]
        if r < C["word"]:
            w = pick_word(rng, ch)
            if w:
                other = pick_word(rng, self.vocab[int(rng.integers(len(self.vocab)))]) if rng.random() < 0.3 else None
                line = carrier(rng, w, kind, other)
                if not S3.marked(line, self.exclusions["markers"]):
                    return line, "jiten_words"
        elif r < C["word"] + C["name"]:
            nm = name_holding(rng, ch)
            if nm:
                t = NAME_LINES[rng.integers(len(NAME_LINES))] if kind != "narration" else "{n}がいた。"
                return t.format(n=name_text(nm[1], ch, rng).replace("　", "")), "names"
    s, pool = _ORIG["context"](self, rng, c, kind)
    if s is None:
        w = pick_word(rng, ch)
        if w and not S3.marked(w, self.exclusions["markers"]):
            return carrier(rng, w, kind), "jiten_words"
    return s, pool


def sample_name4(rng):
    """synth3's speaker name, except that with CFG4 name_focus a name holding a quota class with
    quota left is drawn (rare name kanji in the plates where VNs show them)."""
    R = S3.R
    q = getattr(R, "quota_t", None)
    if q is not None and rng.random() < CFG4["name_focus"]:
        rem = np.maximum(0.0, q - R.counts) * R4["has_name"]
        tot = rem.sum()
        if tot > 0:
            c = R.vocab[int(rng.choice(len(rem), p=rem / tot))]
            nm = name_holding(rng, c)
            if nm:
                src, parts = nm
                name = name_text(parts, c, rng)
                N = S3.CFG["names"]
                if rng.random() < N["honorific"] and R.lists.get("honorifics"):
                    name += R.lists["honorifics"][rng.integers(len(R.lists["honorifics"]))]
                if rng.random() < N["brackets"]:
                    name = "【" + name + "】"
                name = "".join(x for x in name if x in R.vocab_set or x == "　")
                if c in name:
                    return name, name.replace("　", ""), f"{src}+focus"
    return _ORIG["sample_name"](rng)


# ----------------------------------------------------------------------------- style

def s2_style4(rng, platform):
    """synth2's palette and effects with synth4's corrections (docs/synth4-gap.md, section 3): fewer
    outlines where the scheme does not need one, half the gradient fills, darker dialogue boxes,
    lighter light boxes, more pure-white and darker dark fills."""
    st = _ORIG["s2_style"](rng, platform)
    sch = st["scheme"]
    C = CFG4["contrast"]
    if st["stroke"] and sch in ("light_on_dark", "dark_on_light", "on_colour") and rng.random() < CFG4["stroke_drop"]:
        st["stroke"] = None
        st["stroke2"] = None
    if st["grad"] is not None and rng.random() >= CFG4["grad_keep"]:
        st["grad"] = None
    box = np.asarray(st["box"], np.float32)
    fill = np.asarray(st["fill"], np.float32)
    sat = (max(fill) - min(fill)) / max(1.0, float(max(fill)))
    if sat > 0.3 and sch in ("light_on_dark", "dark_on_light", "outlined_on_light") and \
            rng.random() < CFG4["accent_fill_drop"]:      # an accent-coloured fill becomes a plain one
        fill = (S2.text_light(rng) if S2.luma(fill) > S2.luma(box) else S2.text_dark(rng)).astype(np.float32)
    if sch == "light_on_dark":
        box = box * float(rng.uniform(*C["dark_box"]))
        if S2.luma(fill) > 200 and max(fill) - min(fill) < 40 and rng.random() < C["white_fill"]:
            fill = np.full(3, 255.0, np.float32)
    elif sch == "dark_on_light":
        box = 255.0 - (255.0 - box) * float(rng.uniform(*C["light_box"]))
        if S2.luma(fill) < 90:
            fill = fill * float(rng.uniform(*C["dark_fill"]))
    st["box"] = np.clip(box, 0, 255).astype(np.float32)
    st["fill"] = np.clip(fill, 0, 255).astype(np.float32)
    st["on_dark"] = S2.luma(st["fill"]) > S2.luma(st["box"])
    return st


def accent4(rng, on_dark):
    """synth2's accent colour, with a green or cyan hue redrawn once with CFG4 green_redraw."""
    c = _ORIG["accent"](rng, on_dark)
    h = colorsys.rgb_to_hsv(*(np.asarray(c, np.float64) / 255.0))[0]
    lo, hi = CFG4["green_band"]
    if lo <= h <= hi and rng.random() < CFG4["green_redraw"]:
        c = _ORIG["accent"](rng, on_dark)
    return c


# ----------------------------------------------------------------------------- wiring

_ORIG = {}


def patch():
    """Install synth4 into synth3 (idempotent): CFG, licence profiles and the functions above."""
    if _ORIG:
        return
    merged = S3.merge_cfg(S3.CFG, CFG_PATCH)
    S3.CFG.clear()
    S3.CFG.update(merged)
    S3.LICENCE_PROFILES["personal"] = {"open", "nc", "grey", "owner"}
    S3.LICENCE_PROFILES["open"] = {"open"}
    _ORIG.update(focus_char=S3.Resources.focus_char, context=S3.Resources.context, sample_name=S3.sample_name,
                 s2_style=S2.sample_style, accent=S2.accent, font_root=S3.Resources.font_root)
    S3.Resources.focus_char = focus_char4
    S3.Resources.context = context4
    S3.Resources.font_root = font_root4
    S3.sample_name = sample_name4
    S3.px_fits = px_fits4
    S2.sample_style = s2_style4
    S2.accent = accent4
    S3.FONT_MANIFEST = FONT_MANIFEST4


def init4(curriculum, **kw):
    """synth3's resources with synth4's fonts, then the curriculum's per-class state."""
    patch()
    kw.setdefault("font_manifest", FONT_MANIFEST4)
    R = S3.init(**kw)
    cur = json.loads(Path(curriculum).read_text(encoding="utf-8"))
    if cur.get("inputs", {}).get("vocab_sha256") not in (None, sha256(S3.DATA / "vocab.txt")):
        raise SystemExit("the curriculum was built for another vocab")
    R4.clear()
    R4["file"] = str(curriculum)
    R4["sha256"] = sha256(curriculum)
    R4["deficit"] = np.array([cur["deficit"].get(c, 0) for c in R.vocab], np.float64)
    R4["words"] = {c: [(w, x) for w, x in v if all(ch in R.vocab_set for ch in w)] for c, v in cur["words"].items()}
    idx = defaultdict(list)
    for src, lst in R.names.items():
        for i, parts in enumerate(lst):
            for ch in set("".join(parts)):
                if ch in R.cid and len(idx[ch]) < 64:
                    idx[ch].append((src, i))
    R4["name_idx"] = dict(idx)
    R4["has_name"] = np.array([1.0 if c in idx else 0.0 for c in R.vocab])
    has_word = np.array([bool(R4["words"].get(c)) for c in R.vocab])
    R4["has_any_ctx"] = R.has_ctx | has_word | (R4["has_name"] > 0)
    return R


def run_shard4(job):
    """synth3's shard with this shard's curriculum targets."""
    idx, _, seed = job[0], job[1], job[2]
    S3.R.quota_t = shard_targets(idx, seed, R4["deficit"], CFG4["curriculum"]["shards"])
    return S3.run_shard(job)


def recipe4(args):
    rec = S3.recipe(args)
    rec["kind"] = "synth4-recipe"
    rec["cfg4"] = CFG4
    rec["cfg_patch"] = CFG_PATCH
    rec["code_sha256"]["synth4.py"] = sha256(HERE / "synth4.py")
    rec["font_manifest_sha256"] = sha256(FONT_MANIFEST4)
    rec["curriculum"] = {"file": R4["file"], "sha256": R4["sha256"], "deficit": int(R4["deficit"].sum())}
    rec["targets_sha256"] = sha256(args.targets) if args.targets else None
    lic = sorted({f.licence for f in S3.R.fonts})
    rec["fonts_by_licence"] = {k: sum(1 for f in S3.R.fonts if f.licence == k) for k in lic}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fonts", "fonts-verify", "curriculum", "preview", "shards", "manifest"])
    ap.add_argument("--assets", type=Path, default=S3.ASSETS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dir", action="append")
    ap.add_argument("--manifest", default=str(FONT_MANIFEST4))
    ap.add_argument("--profile", choices=["personal", "open"], default="personal")
    ap.add_argument("--shards", type=int, default=100)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--per-shard", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--scene-hashes", type=Path)
    ap.add_argument("--font-cache", type=Path, default=S4_ART / "fonts" / "coverage_cache.json")
    ap.add_argument("--targets", type=Path, help="focus targets (synth4.py curriculum --targets-out)")
    ap.add_argument("--curriculum", type=Path, help="synth4.py curriculum --out")
    # curriculum inputs
    ap.add_argument("--exposures")
    ap.add_argument("--jiten-vn", default=str(S3.ASSETS / "text" / "jiten" / "frequency_list_visual_novel.csv"))
    ap.add_argument("--text", help="synth4_gap.py text output (its fitted Zipf exponent)")
    ap.add_argument("--real", action="append", default=[])
    ap.add_argument("--mine")
    ap.add_argument("--mine-models", nargs="*", default=["FL10_sel", "H15_sel"])
    ap.add_argument("--base-targets")
    ap.add_argument("--targets-out")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.set_defaults(split="train")
    args = ap.parse_args()
    if args.mode == "fonts":
        return cmd_fonts(args)
    if args.mode == "fonts-verify":
        return cmd_fonts_verify(args)
    if args.mode == "curriculum":
        return cmd_curriculum(args)
    if args.mode == "manifest":
        return S3.manifest(args)
    import cv2
    cv2.setNumThreads(1)
    init4(args.curriculum, assets=args.assets, split="train", profile=args.profile, font_cache=args.font_cache,
          scene_hashes=args.scene_hashes, targets=args.targets)
    R = S3.R
    print(f"fonts {len(R.fonts)} files / {len({f.fam for f in R.fonts})} families "
          f"{ {c: len(v) for c, v in R.by_cat.items()} }; licences "
          f"{ {k: sum(1 for f in R.fonts if f.licence == k) for k in sorted({f.licence for f in R.fonts})} }; "
          f"pools {list(R.pools)}; excluded {len(R.exclusions['vns'])} VNs; curriculum deficit "
          f"{int(R4['deficit'].sum())}", flush=True)
    if args.mode == "preview":
        R.quota_t = shard_targets(0, args.seed, R4["deficit"], CFG4["curriculum"]["shards"])
        return S3.preview(args)
    args.out.mkdir(parents=True, exist_ok=True)
    rec = json.loads(json.dumps(recipe4(args), ensure_ascii=False))
    rp = args.out / "recipe.json"
    if rp.exists() and json.loads(rp.read_text(encoding="utf-8")) != rec:
        raise SystemExit(f"{rp} holds a different recipe: use a new --out")
    rp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    jobs = [(i, args.per_shard, args.seed, str(args.out), "synth4")
            for i in range(args.first, args.first + args.shards)]
    t0, total = time.time(), 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, regions, errors, dt in pool.imap_unordered(run_shard4, jobs):
            total += n
            print(f"shard {idx}: {n} lines from {regions} regions ({errors} errors) in {dt:.0f}s "
                  f"(total {total}, {time.time() - t0:.0f}s)", flush=True)
    print("SHARDS_DONE", flush=True)


if __name__ == "__main__":
    main()

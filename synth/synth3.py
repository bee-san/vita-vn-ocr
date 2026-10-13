#!/usr/bin/env python3
"""synth3: synthetic PS Vita / PSP visual-novel dialogue line crops, as the console reads them.

synth2 (render.py) taught the recognizer the VN look. synth3 keeps its box, outline and layout
machinery and changes what docs/synth3.md found mattered, measured against training-side real
crops (teacher_train + real_train) and the released models' errors on training-side pools:

  text       VN-register dialogue pools (synth3_text.py: Aozora speech in 「」, subtitles,
             roleplay, Tatoeba, spoken corpora, templates) and narration, split by source
             document; focus-character control: 35% of lines are chosen to contain a
             tail-weighted, hard-example-weighted vocab character, from a natural context;
             crop lengths steered to the real crops' length histogram (many short crops)
  names      VNDB character names (test and development VNs removed), web-novel names and
             generated names; plates, inline names, honorifics, 【】, ？？？
  fonts      the synth2 pool plus a pinned free/OFL pool (data/manifests/synth3_fonts.json),
             families sampled per category, weights biased to medium-bold, per-line coverage
  raster     8-bit FreeType anti-aliasing, 4-bit (16-level) glyph coverage, glyph-atlas
             rendering (bigger cell, 4-bit, bilinear or area downscale), 1-bit hinted
             glyphs, bitmap/pixel fonts at their exact 10-20 px strikes (nearest 2x on Vita),
             2x supersampling, coverage gamma
  screens    Vita 960x544, PSP 480x272 (12-20 px), PSP upscaled 2x as on a Vita (bilinear,
             nearest, 1.995x, scanlines), and large renders area-scaled down to either size
             (PC / home-console screenshots as the teacher's training crops have them)
  capture    clean framebuffer or VNDB-like JPEG chains (console encoder then Q90), PSP 16-bit
             dither, blur, noise
  crops      the eval/teacher convention (labelled ink incl. furigana + 4 px Vita / 2 px PSP,
             small jitter) or the console's own GRAD finder boxes (common.find_lines_ex),
             both stored in console polarity (dark text inverted by common.line_dark)

Version 3 (docs/synth3-v3.md) keeps every version-2 fix and closes the gaps measured against
the training-side real crops: more large renders area-scaled to the console (two thirds of real
crops are PC shots scaled down) at the real ink sizes, the VNDB JPEG applied at the source
resolution before that downscale, busier crop edges (more scenes, finer scene zoom, more
translucent boxes, a GRAD box that fails the one-line check falls back to the line's gt crop
instead of dropping a line in a busy region), Latin letters and digits near the real share (fewer
UI menus and Latin UI words, pool lines with Latin mostly redrawn, less alnum dressing), and
pages: text that overflows the box continues on a later page (v2 showed only the first, so a
long utterance never reached its closing 」). Names, scenes and screenshots of the fresh native
textbox titles and their VNDB relations (data/manifests/synth3_exclude_fresh_native.json) are
never used.

  synth3.py preview --out DIR [--n 16]                contact sheets (<= 1280 px)
  synth3.py shards  --out DIR --split train --shards N --per-shard M [--workers W] [--seed S]
  synth3.py manifest --dir DIR [--dir DIR2] --out MANIFEST.json
  synth3.py scene-hashes --out HASHES.json           file and RGB-pixel sha256 of every scene candidate

--cfg '{"platform": {...}}' (or --cfg @file.json) overrides CFG for calibration renders; it is
recorded in the recipe. Assets come from --assets (default $HERD/shared/artifacts/synth3): fonts/,
text/pools3/, and screenshot backgrounds (training-split games only; test, development and
excluded VNs never).
"""
import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import render as S2  # noqa: E402
from common import DATA, is_test_vn, line_dark, load_vocab, to_gray  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ASSETS = HERD / "shared" / "artifacts" / "synth3"
SHARED_DATA = HERD / "shared" / "data"
FONT_MANIFEST = ROOT / "data" / "manifests" / "synth3_fonts.json"
# Titles protected like eval's test games (the fresh native textbox test, its reserves, CLANNAD,
# and their VNDB relations): never a name, scene or screenshot of synth3 (version 3 on).
EXCLUDE_MANIFESTS = (ROOT / "data" / "manifests" / "synth3_exclude_fresh_native.json",
                     ROOT / "data" / "manifests" / "synth3_exclude_clannad.json")
LATIN = re.compile(r"[0-9A-Za-z０-９Ａ-Ｚａ-ｚ]")

# Every probability of the generator, recorded verbatim in the shard manifest.
CFG = {
    "version": 3,
    # v3: two thirds of the real training crops are PC / home-console shots area-scaled to a
    # console size, so large renders scaled down get a bigger share (docs/synth3-v3.md)
    "platform": {"vita": 0.24, "psp": 0.24, "psp2x": 0.05, "scaled": 0.47},
    "scaled_to_psp": 0.4,
    "category": {
        "vita": {"gothic": 0.40, "rounded": 0.20, "mincho": 0.18, "pen": 0.05, "display": 0.07, "pixel": 0.10},
        "psp": {"gothic": 0.36, "rounded": 0.14, "mincho": 0.14, "pen": 0.03, "display": 0.05, "pixel": 0.28},
    },
    "raster": {
        "vita": {"aa8": 0.36, "aa4": 0.24, "atlas": 0.26, "ss2": 0.08, "mono": 0.06},
        "psp": {"aa8": 0.18, "aa4": 0.40, "atlas": 0.12, "ss2": 0.02, "mono": 0.28},
    },
    "gamma": {"p": 0.5, "lo": 0.7, "hi": 1.45},
    "layout": {"adv": 0.50, "band": 0.12, "nvl": 0.09, "balloon": 0.04, "choice": 0.08, "bare": 0.12, "ui": 0.05},
    # v3: UI menus were 12% of v2's crops and carried half its Latin letters
    "ui": {"lines": [2, 5], "latin_word": 0.08, "places": 0.45},
    "short": 0.15,
    "narration": 0.18,
    "narrate_nvl": 0.4,
    "focus": 0.25,
    "focus_exponent": 0.6,
    "focus_hard_gain": 4.0,
    "focus_prior_exponent": 0.35,       # VN relevance: (real + Jiten-VN frequency + 1e-7) ** this
    "reference_pools": ["wiktionary", "wikiquote"],   # focus contexts only when nothing else has the char
    "brackets": {"named": 0.7, "unnamed": 0.35, "double": 0.03},
    "dressing": {"interjection": 0.08, "stammer": 0.03, "ending": 0.15, "terminal": 0.75, "dots": 0.06,
                 "elongation": 0.03, "alnum": 0.012},
    # v3: a pool line holding Latin letters or digits is kept with this probability, else redrawn
    # (at most latin_redraws times); focus contexts for a non-Latin character likewise
    "latin_keep": 0.22,
    "latin_redraws": 8,
    # v3: text longer than the box continues on later pages and a screenshot shows any page (v2
    # showed only the first, so a long utterance never reached its closing 」)
    "pages": {"wrap_lines": 48},
    "terminal": {"。": 0.45, "！": 0.17, "？": 0.13, "……": 0.12, "っ！": 0.04, "！？": 0.03, "～": 0.04, "♪": 0.02},
    "names": {"vndb": 0.6, "syosetu": 0.3, "generated": 0.1, "honorific": 0.2, "brackets": 0.1,
              "unknown": 0.03, "role": 0.03, "space": 0.3},
    # v3: names inline before 「 were 5.6% of crops and put 「 mid-line 3.5x as often as in real crops
    "name_mode": {"vita": {"tab": 0.58, "inline": 0.06, "line": 0.24, None: 0.12},
                  "psp": {"tab": 0.55, "inline": 0.08, "line": 0.25, None: 0.12}},
    "nvl_name": 0.3,
    "name_coloured": 0.65,
    "ruby": {"vita": 0.15, "psp": 0.20},
    "cursor": 0.33,
    "partial": {"p": 0.08, "ramp": 0.5},
    "join": 0.35,
    "drop_aside": 0.8,
    "coloured": {"span": 0.14, "line": 0.10},
    "outline": {"vita": {"light_on_dark": 0.40, "dark_on_light": 0.12, "on_colour": 0.7},
                "psp": {"light_on_dark": 0.60, "dark_on_light": 0.20, "on_colour": 0.8},
                "scaled": {"light_on_dark": 0.55, "dark_on_light": 0.2, "on_colour": 0.8}},
    "shadow": {"vita": 0.30, "psp": 0.25, "scaled": 0.5},
    # native PSP em sizes and their probabilities (v3: one step larger, real PSP ink p90 23 px)
    "psp_sizes": [[12, 13, 14, 15, 16, 17, 18, 20], [0.04, 0.10, 0.17, 0.21, 0.19, 0.12, 0.10, 0.07]],
    "capture": {"vita": {"clean": 0.5, "vndb2": 0.5},
                "psp": {"clean": 0.35, "q90": 0.35, "vndb2": 0.30},
                "psp2x": {"bilinear": 0.6, "nearest": 0.3, "x1995": 0.05, "scanlines": 0.05, "jpeg": 0.1},
                # v3: VNDB's JPEG happens at the source resolution, before the area downscale
                # that made the real crops (teacher/realdata), as on a real PC screenshot
                "scaled": {"q90": 0.85, "clean": 0.15}},
    "dither565": 0.45,
    "blur": 0.15,
    "noise": 0.15,
    "subnative": 0.03,
    "crop": {"gt": 0.25, "grad": 0.75},
    # v3: a line whose GRAD box fails the one-line check gets its gt crop instead of no crop
    # (v2 dropped those lines, which sit in busy regions: GRAD crops were the cleanest)
    "grad_fallback": True,
    "pad": {"vita": 4, "psp": 2},
    # extra gt margin per side, U(0, max x ink height) with probability p (v3: per target, from
    # one pixel-ink measure for real and synthetic crops; real crops are their ink plus 4 / 2 px,
    # a little more on the left; v2 copied MeikiOCR character-box margins, which sit inside the ink)
    "pad_extra": {"vita": {"p": 1.0, "top": 0.04, "bottom": 0.06, "left": 0.12, "right": 0.04},
                  "psp": {"p": 1.0, "top": 0.0, "bottom": 0.02, "left": 0.2, "right": 0.0}},
    # final em of large renders area-scaled down: N(mean, sd) clipped to [lo, hi] (v3: matched to
    # the pixel ink height of teacher's and realdata's hires crops, p10/50/90 14-15/21/31-32 px
    # on the Vita target)
    "scaled_em": {"vita": [20.5, 3.3, 13, 30], "psp": [13.5, 2.0, 9, 19]},
    "background": {"scene": 0.92},
    "scene_zoom": [1.0, 1.2],
    # dialogue-box opacity per platform group: [p of the low range, low range, high range] (v3:
    # PC renders scaled down see far more scene through their boxes, as in the real hires crops)
    "box_alpha": {"vita": {"on_dark": [0.55, [0.3, 0.7], [0.7, 0.93]], "light": [0.3, [0.65, 0.88], [0.88, 1.0]],
                           "on_colour": [0.7, 0.97]},
                  "psp": {"on_dark": [0.7, [0.2, 0.65], [0.65, 0.9]], "light": [0.5, [0.5, 0.82], [0.82, 1.0]],
                          "on_colour": [0.6, 0.95]},
                  "scaled": {"on_dark": [0.8, [0.15, 0.6], [0.6, 0.85]], "light": [0.65, [0.4, 0.75], [0.75, 0.95]],
                             "on_colour": [0.5, 0.9]}},
    "pitch": [1.12, 1.8],
    "dialogue_pools": {
        "aozora_dialogue": 0.16, "narikiri": 0.20, "aratako_dsv3": 0.07, "aratako_gpt4omini": 0.09,
        "opensubtitles": 0.09, "jesc": 0.09, "tatoeba": 0.07, "nucc": 0.03, "jdd": 0.02, "bsd": 0.01,
        "templates": 0.07, "wiktionary": 0.01, "wikiquote": 0.01},
    "narration_pools": {"aozora_narration": 0.85, "wiki": 0.15},
    # The training-side real crops' label-length histogram (synth3_stats.py on teacher_train +
    # real_train, 20,000 lines; bins synth3_stats.LEN_EDGES), the acceptance target.
    "length_edges": [0, 2, 4, 6, 9, 13, 18, 24, 31, 41, 200],
    "length_target": [0.0132, 0.2402, 0.154, 0.1029, 0.1199, 0.1419, 0.1216, 0.0867, 0.0191, 0.0003],
    "length_floor": 0.15,
}
LICENCE_PROFILES = {"personal": {"open", "nc", "grey"}, "open": {"open"}}
CONFUSABLE = ("ばはぱ バハパ ぶふぷ ブフプ べへぺ ベヘペ ぼほぽ ボホポ びひぴ ビヒピ がか ガカ ぎき ギキ ぐく グク "
              "げけ ゲケ ごこ ゴコ ざさ ザサ じし ジシ ずす ズス ぜせ ゼセ ぞそ ゾソ だた ダタ ぢち づつ でて デテ "
              "どと ドト っつ ッツ ゃや ャヤ ゅゆ ュユ ょよ ョヨ ぁあ ァア ぃい ィイ ぅう ゥウ ぇえ ェエ ぉお ォオ "
              "ゎわ ヮワ ー一 ロ口 カ力 エ工 ニ二 へヘ ぺペ べベ 『「 』」 夕タ 卜ト 八ハ 千干 土士 未末 己已巳 "
              "人入 日曰 目且 大犬太 天夭 白自百 ～〜 ・．").split()

R = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose(rng, weights):
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=np.float64)
    return names[rng.choice(len(names), p=p / p.sum())]


def merge_cfg(base, patch):
    """CFG with a calibration override merged in: nested dicts merge key by key, anything else
    replaces. A key the generator does not have is an error (a typo would be silently ignored)."""
    out = dict(base)
    for k, v in patch.items():
        if k not in base:
            raise KeyError(f"unknown CFG key {k!r}")
        out[k] = merge_cfg(base[k], v) if isinstance(v, dict) and isinstance(base[k], dict) else v
    return out


def load_exclusions(paths):
    """VN ids, screenshot hashes (file and decoded-RGB pixel sha256) and name hashes that synth3
    must never use, from exclusion manifests (data/manifests/synth3_exclude_*.json)."""
    ex = {"vns": set(), "sha256": set(), "pixel_sha256": set(), "name_sha256": set(), "markers": set(), "files": []}
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        if doc.get("kind") != "synth3-exclusion":
            raise ValueError(f"{p} is not a synth3 exclusion manifest")
        ex["vns"] |= set(doc["vns"])
        ex["sha256"] |= set(doc.get("screenshots", {}).get("sha256", []))
        ex["pixel_sha256"] |= set(doc.get("screenshots", {}).get("pixel_sha256", []))
        ex["name_sha256"] |= set(doc.get("names", {}).get("name_sha256", []))
        ex["markers"] |= set(doc.get("text_markers", {}).get("strings", []))
        ex["files"].append({"file": Path(p).name, "sha256": sha256(p), "vns": len(doc["vns"])})
    return ex


def marked(s, markers):
    """Does text s hold an exclusion marker (a protected title's character names or title)?"""
    return any(m in s for m in markers)


def name_hash(parts):
    """The exclusion manifests' name key: the name's parts as names.tsv stores them, joined."""
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def pixel_sha256(path):
    """sha256 of the decoded RGB pixels (eval/fresh_native_textbox.py's definition)."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- resources

class TextPool:
    """Lines of one pool split as one string plus offsets: fork-friendly and compact."""

    def __init__(self, path):
        text = Path(path).read_text(encoding="utf-8")
        lines = [s for s in text.split("\n") if s]
        self.blob = "".join(lines)
        self.off = np.concatenate([[0], np.cumsum([len(s) for s in lines])]).astype(np.int64)
        self.n = len(lines)
        self.sha256 = sha256(path)

    def __getitem__(self, i):
        return self.blob[self.off[i]:self.off[i + 1]]

    def sample(self, rng):
        return self[int(rng.integers(self.n))]


class Font:
    __slots__ = ("cat", "fam", "path", "chars", "pixel_em", "licence", "key", "weight")

    def __init__(self, cat, fam, path, chars, pixel_em, licence, key, weight):
        self.cat, self.fam, self.path, self.chars = cat, fam, path, chars
        self.pixel_em, self.licence, self.key, self.weight = pixel_em, licence, key, weight


def coverage(path, vocab, cache):
    """Vocab characters the font draws: mapped by its cmap to a glyph with an outline (TrueType
    contours or a composite, a CFF charstring with bounds). A mapped but empty glyph would render
    a blank under a label, so it does not count. Cached by file sha256."""
    key = sha256(path)
    if key not in cache:
        cache[key] = "".join(drawn_chars(path, vocab))
    return set(cache[key]), key


def drawn_chars(path, vocab):
    from fontTools.ttLib import TTFont
    font = TTFont(str(path), lazy=True, fontNumber=0)
    cmap = font.getBestCmap() or {}
    if "glyf" in font:
        glyf = font["glyf"]

        def ink(name):
            g = glyf[name]
            return g.numberOfContours != 0
    elif "CFF " in font or "CFF2" in font:
        table = font["CFF "] if "CFF " in font else font["CFF2"]
        cs = table.cff.topDictIndex[0].CharStrings

        def ink(name):
            return cs[name].calcBounds(cs) is not None
    else:
        def ink(name):
            return True
    out = []
    for c in vocab:
        g = cmap.get(ord(c))
        if g is None:
            continue
        try:
            if c.isspace() or ink(g):
                out.append(c)
        except Exception:  # noqa: BLE001 - a glyph fontTools cannot parse is not drawn
            continue
    return out


class Resources:
    def __init__(self, assets=ASSETS, split="train", profile="personal", font_manifest=FONT_MANIFEST,
                 vocab=None, holdout_fonts=(), scenes=True, exclude=EXCLUDE_MANIFESTS, font_cache=None,
                 scene_hashes=None, targets=None):
        self.assets, self.split, self.profile = Path(assets), split, profile
        self.targets_path = Path(targets) if targets else None
        self.exclusions = load_exclusions(exclude)
        self.font_cache = Path(font_cache) if font_cache else self.assets / "fonts" / "coverage_cache.json"
        self.vocab = load_vocab(vocab) if vocab else load_vocab()
        self.vocab_set = set(self.vocab)
        self.cid = {c: i for i, c in enumerate(self.vocab)}
        self.fonts = self._fonts(font_manifest, holdout_fonts)
        self.by_cat = {}
        for f in self.fonts:
            self.by_cat.setdefault(f.cat, {}).setdefault(f.fam, []).append(f)
        self._text()
        self._backgrounds(scenes, scene_hashes)
        self.hiragana = [c for c in self.vocab if 0x3041 <= ord(c) <= 0x3093 and c not in "ゎゐゑ"]
        self.katakana = [c for c in self.vocab if 0x30A1 <= ord(c) <= 0x30FA and c not in "ヵヶヮ"]
        self.native_faces = {}
        self.counts = np.zeros(len(self.vocab), np.float64)
        self._focus_w = None
        self._focus_age = 0

    # ---- fonts
    def _fonts(self, manifest_path, holdout):
        doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        allowed = LICENCE_PROFILES[self.profile]
        cache_path = self.font_cache
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        n_cache = len(cache)
        out = []
        for fam in doc["families"]:
            if fam["licence_class"] not in allowed:
                continue
            if (fam["family"] in holdout) != (self.split == "dev-fonts"):
                continue
            for f in fam["files"]:
                path = Path(self.font_root(fam["root"])) / f["file"]
                if not path.exists():
                    raise FileNotFoundError(f"pinned font missing: {path} (synth3.py fetch-fonts)")
                if path.stat().st_size != f["bytes"]:
                    raise ValueError(f"font size differs from the manifest: {path}")
                chars, key = coverage(path, self.vocab, cache)
                if key != f["sha256"]:
                    raise ValueError(f"font hash differs from the manifest: {path}")
                cat = f.get("category") or fam["category"]
                pix = f.get("pixel_em")
                if cat == "pixel" and not pix:
                    continue
                if len(chars) < (300 if cat == "pixel" else 1500):
                    continue
                out.append(Font(cat, fam["family"], path, chars, pix, fam["licence_class"], key[:12],
                                f.get("weight") or 400))
        if len(cache) != n_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        if not out:
            raise ValueError("no usable fonts")
        return out

    def font_root(self, root):
        return {"pool": SHARED_DATA / "fonts", "pool_extra": HERD / "shared" / "artifacts" / "synth2" / "fonts_extra",
                "synth3": self.assets / "fonts"}[root]

    # ---- text
    def _text(self):
        pools_dir = self.assets / "text" / "pools3"
        meta = json.loads((pools_dir / "pools.json").read_text(encoding="utf-8"))["pools"]
        split = "dev" if self.split.startswith("dev") else "train"
        allowed = LICENCE_PROFILES[self.profile]
        self.pools, self.pool_meta = {}, {}
        for name in list(CFG["dialogue_pools"]) + list(CFG["narration_pools"]):
            if name == "templates":
                continue
            m = meta.get(f"{name}.{split}")
            if m is None or m["class"] not in allowed or not m["lines"]:
                continue
            self.pools[name] = TextPool(pools_dir / m["file"])
            self.pool_meta[name] = {"file": m["file"], "sha256": self.pools[name].sha256, "lines": self.pools[name].n,
                                    "class": m["class"]}
        self.lists = json.loads((self.assets / "text" / "lists.json").read_text(encoding="utf-8"))
        self.lists = {k: [s for s in v if all(c in self.vocab_set for c in s)] for k, v in self.lists.items()
                      if isinstance(v, list)}
        z = np.load(pools_dir / f"contexts.{split}.npz")
        names = json.loads(str(z["pools"]))
        keep = np.array([names[p] in self.pools for p in z["pool"]], bool)
        order = np.argsort(z["char"][keep], kind="stable")
        self.ctx_char = z["char"][keep][order]
        self.ctx_pool = np.array([names[p] for p in z["pool"][keep][order]], dtype=object)
        self.ctx_line = z["line"][keep][order]
        self.ctx_start = np.searchsorted(self.ctx_char, np.arange(len(self.vocab) + 1))
        self.names = {"vndb": [], "syosetu": []}
        self.names_dropped = 0
        bad = self.exclusions["name_sha256"]
        for line in (pools_dir / f"names.{split}.tsv").read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if (bad and name_hash(p[2:]) in bad) or marked("".join(p[2:]), self.exclusions["markers"]):
                self.names_dropped += 1            # a character of an excluded VN, or its homonym
                continue
            self.names.setdefault(p[0], []).append(tuple(p[2:]))
        targets_path = self.targets_path or pools_dir / "targets.json"
        tg = json.loads(targets_path.read_text(encoding="utf-8"))["chars"]
        self.hard = np.array([tg[c]["hard"] if c in tg else 0.0 for c in self.vocab])
        conf = set("".join(CONFUSABLE))
        self.hard += np.array([0.05 if c in conf else 0.0 for c in self.vocab])
        self.hard = self.hard / max(1e-9, self.hard.max())
        self.prior = np.array([(tg[c]["real"] + tg[c]["jiten_vn"] if c in tg else 0.0) + 1e-7 for c in self.vocab])
        self.prior = self.prior ** CFG["focus_prior_exponent"]
        jk = [c for c in self.vocab if S2.is_kanji(c)]
        zero = {"real": 0.0, "jiten_vn": 0.0}
        jw = np.array([tg.get(c, zero)["jiten_vn"] + tg.get(c, zero)["real"] for c in jk]) + 1e-7
        order = np.argsort(-jw)
        self.name_kanji = [jk[i] for i in order[:2500]]
        w = jw[order[:2500]] ** 0.7
        self.name_kanji_p = w / w.sum()
        self.has_ctx = (self.ctx_start[1:] - self.ctx_start[:-1]) > 0
        self.text_identity = {"pools": self.pool_meta, "contexts": sha256(pools_dir / f"contexts.{split}.npz"),
                              "names": sha256(pools_dir / f"names.{split}.tsv"),
                              "names_dropped_excluded": self.names_dropped,
                              "targets": sha256(targets_path),
                              "lists": sha256(self.assets / "text" / "lists.json")}

    # ---- scenes
    def _backgrounds(self, scenes=True, scene_hashes=None):
        """Training-side screenshots for scenes (synth2's list plus realdata's shots), never a
        test, development or excluded VN, and never a screenshot whose file or RGB-pixel sha256 an
        exclusion manifest lists; split by VN between train and dev. scenes=False: none (the
        renderer then draws synthetic scenes only, as in CI). scene_hashes: a scene-hashes file
        (file sha256 -> pixel sha256), so the pixels need not be decoded here."""
        self.scene_dropped = {"excluded_vn": 0, "excluded_hash": 0}
        if not scenes:
            self.bg, self.bg_vns, self.bg_text = {"vita": [], "psp": [], "hires": []}, [], {}
            return
        forbid = set(json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())["all"])
        forbid |= set((self.assets / "mining" / "exclude_dev_vns.txt").read_text().split())
        excluded = self.exclusions["vns"]
        vita, psp, hires = scene_candidates()
        self.bg_text = scene_text_boxes(HERD / "shared" / "artifacts" / "realdata" / "shots")
        known = json.loads(Path(scene_hashes).read_text(encoding="utf-8"))["pixels"] if scene_hashes else {}

        def clean(p):        # not a screenshot an exclusion manifest lists (by bytes or by pixels)
            if not (self.exclusions["sha256"] or self.exclusions["pixel_sha256"]):
                return True
            fs = sha256(p)
            ps = known.get(fs) or pixel_sha256(p)
            if fs in self.exclusions["sha256"] or ps in self.exclusions["pixel_sha256"]:
                self.scene_dropped["excluded_hash"] += 1
                return False
            return True

        def ok(p):      # a scene needs a text detection record: its patches must avoid the screen's text
            vn = p.name.split("_")[0]
            if vn in excluded:
                self.scene_dropped["excluded_vn"] += 1
                return False
            return p.exists() and vn not in forbid and not is_test_vn(vn) and p in self.bg_text
        split_dev = self.split.startswith("dev")

        def mine(p):        # scenes split by VN between train and synthetic dev
            h = int(hashlib.sha256(f"synth3-bg:{p.name.split('_')[0]}".encode()).hexdigest()[:8], 16)
            return (h % 10 == 0) == split_dev
        self.bg = {k: [p for p in v if ok(p) and mine(p) and clean(p)]
                   for k, v in (("vita", vita), ("psp", psp), ("hires", hires))}
        self.bg_vns = sorted({p.name.split("_")[0] for v in self.bg.values() for p in v})

    # ---- focus characters
    def focus_char(self, rng):
        if self._focus_w is None or self._focus_age >= 64:
            w = (self.counts + 20.0) ** (-CFG["focus_exponent"]) * (1 + CFG["focus_hard_gain"] * self.hard) \
                * self.prior
            w[~self.has_ctx] = 0
            self._focus_w = w / w.sum()
            self._focus_age = 0
        self._focus_age += 1
        return int(rng.choice(len(self._focus_w), p=self._focus_w))

    def kind_of(self, pool):
        return "narration" if pool in CFG["narration_pools"] else "dialogue"

    def context(self, rng, c, kind="dialogue"):
        """A train line holding vocab character c, from a pool of `kind` when one has it."""
        a, b = self.ctx_start[c], self.ctx_start[c + 1]
        if a == b:
            return None, None
        ks = np.arange(a, b)
        ref = CFG["reference_pools"]
        same = [k for k in ks if self.kind_of(self.ctx_pool[k]) == kind and self.ctx_pool[k] not in ref]
        other = [k for k in ks if self.ctx_pool[k] not in ref]
        pick = same or other
        if self.exclusions["markers"]:
            pick = [k for k in pick if not marked(self.pools[self.ctx_pool[k]][int(self.ctx_line[k])],
                                                  self.exclusions["markers"])]
            if not pick:
                return None, None
        if pick and not LATIN.match(self.vocab[c]) and rng.random() >= CFG["latin_keep"]:
            plain = [k for k in pick if not LATIN.search(self.pools[self.ctx_pool[k]][int(self.ctx_line[k])])]
            pick = plain or pick         # v3: Latin letters only where the focus character needs them
        k = int(pick[rng.integers(len(pick))]) if pick else int(rng.integers(a, b))
        if self.exclusions["markers"] and marked(self.pools[self.ctx_pool[k]][int(self.ctx_line[k])],
                                                 self.exclusions["markers"]):
            return None, None
        pool = self.ctx_pool[k]
        return self.pools[pool][int(self.ctx_line[k])], pool


def scene_candidates():
    """(vita, psp, hires) screenshot paths synth3 may cut scenes from, before any filter."""
    bg = json.loads((HERD / "shared" / "artifacts" / "synth2" / "backgrounds.json").read_text())
    rd = HERD / "shared" / "artifacts" / "realdata" / "shots"
    vita = [SHARED_DATA / "vndb" / "shots" / s for s in bg["vita_shots"]] + sorted((rd / "vita").glob("*.jpg"))
    return vita, sorted((rd / "psp").glob("*.jpg")), sorted((rd / "hires").glob("*.jpg"))


def _hash_pair(p):
    return sha256(p), pixel_sha256(p)


def scene_hashes(args):
    """File sha256 -> decoded-RGB pixel sha256 of every scene candidate, so render workers check
    the exclusion manifests' pixel hashes without decoding thousands of screenshots each."""
    paths = [p for p in dict.fromkeys(sum(scene_candidates(), [])) if p.exists()]
    with mp.get_context("fork").Pool(args.workers) as pool:
        pairs = pool.map(_hash_pair, paths, chunksize=32)
    ex = load_exclusions(args.exclude)
    hits = [str(p) for p, (fs, ps) in zip(paths, pairs) if fs in ex["sha256"] or ps in ex["pixel_sha256"]]
    doc = {"kind": "synth3-scene-hashes", "schema": 1, "candidates": len(paths), "pixels": dict(pairs),
           "excluded_matches": hits, "exclusions": ex["files"]}
    Path(args.out).write_text(json.dumps(doc, indent=0) + "\n", encoding="utf-8")
    print(f"{len(paths)} scene candidates hashed; {len(hits)} match an exclusion manifest -> {args.out}", flush=True)


def scene_text_boxes(rd):
    """Text lines already detected on the scene screenshots, in their own pixels: realdata's
    PP-OCRv5 line boxes (realdata/labels) and linefind's detections on data/vndb/shots. A scene
    patch that touches any of them is not used, so no real screen text ends up unlabelled
    inside a synthetic crop (version 2; version 1 cut scene text into 2-5% of crops)."""
    boxes = {}
    for f in sorted((rd.parent / "labels").glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                path = rd / r["kind"] / (r["id"].rsplit("_", 1)[0] + ".jpg")
                boxes.setdefault(path, []).append(tuple(r["ink"]))
    det = HERD / "shared" / "artifacts" / "linefind" / "det" / "vita_shots_det.jsonl"
    if det.exists():
        for line in det.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            path = SHARED_DATA / "vndb" / "shots" / r["shot"]
            lst = boxes.setdefault(path, [])
            for q in r["quads"]:
                xs, ys = [v[0] for v in q], [v[1] for v in q]
                lst.append((min(xs), min(ys), max(xs), max(ys)))
    return {p: np.asarray(v, np.float32).reshape(-1, 4) for p, v in boxes.items()}


def init(**kw):
    global R
    R = Resources(**kw)
    S2.R = None          # synth2 helpers see no native faces
    return R


# ----------------------------------------------------------------------------- text

def clean(s):
    return "".join(c for c in s if c in R.vocab_set)


def pool_line(rng, pool):
    """A line of `pool`. One holding Latin letters or digits is kept with probability CFG
    latin_keep, else redrawn (at most latin_redraws times): subtitle, roleplay and Wikipedia lines
    carry about 3x the Latin share of real VN crops (version 3). A line holding an exclusion text
    marker (CLANNAD's character names and title) is always redrawn."""
    bad = R.exclusions["markers"]
    s = R.pools[pool].sample(rng)
    for _ in range(CFG["latin_redraws"]):
        if not LATIN.search(s) or rng.random() < CFG["latin_keep"]:
            break
        s = R.pools[pool].sample(rng)
    for _ in range(50):         # never a line naming a protected title's characters
        if not (bad and marked(s, bad)):
            return s
        s = R.pools[pool].sample(rng)
    return ""


def template_line(rng):
    L = R.lists
    r = rng.random()
    if r < 0.3 and L.get("interjections"):
        s = L["interjections"][rng.integers(len(L["interjections"]))]
        return s + ["", "！", "……", "？", "っ！", "……？"][rng.integers(6)]
    if r < 0.5 and L.get("stammers"):
        return L["stammers"][rng.integers(len(L["stammers"]))]
    if r < 0.75 and L.get("onomatopoeia"):
        s = L["onomatopoeia"][rng.integers(len(L["onomatopoeia"]))]
        return s + ["", "……", "っ", "！"][rng.integers(4)]
    name = sample_name(rng)[1]
    hon = L.get("honorifics") or ["さん"]
    return name + hon[rng.integers(len(hon))] + ["！", "……？", "っ！", "？", "……"][rng.integers(5)]


def dialogue_text(rng, named):
    """One spoken utterance and its source pool."""
    focus = None
    if rng.random() < CFG["focus"]:
        focus = R.focus_char(rng)
        s, pool = R.context(rng, focus)
        if s is None:
            focus = None
    if focus is None:
        weights = {k: v for k, v in CFG["dialogue_pools"].items() if k == "templates" or k in R.pools}
        pool = choose(rng, weights)
        s = template_line(rng) if pool == "templates" else pool_line(rng, pool)
    s = clean(s).lstrip("、。，．・").strip("「」『』").lstrip("、。")
    if pool != "templates" and focus is None and rng.random() < CFG["join"] and pool in R.pools:
        more = clean(pool_line(rng, pool)).strip("「」『』").lstrip("、。")   # a box holds several sentences
        if more and s and len(s) + len(more) <= 70:
            s = s + ("" if s[-1] in "。！？…―～♪" else "。") + more
    if "（" in s and rng.random() < CFG["drop_aside"]:
        s = re.sub(r"（[^（）]*）", "", s)
    if not s:
        return "", pool, focus
    D = CFG["dressing"]
    if R.kind_of(pool) == "narration":
        return s, pool, focus
    lists = R.lists
    r = rng.random()
    if r < D["interjection"] and lists.get("interjections"):
        lead = lists["interjections"][rng.integers(len(lists["interjections"]))]
        s = lead + ("、" if rng.random() < 0.6 else "……") + s
    elif r < D["interjection"] + D["stammer"] and len(s) > 1 and s[0] not in "…―":
        s = s[0] + "、" + s
    if rng.random() < D["ending"] and s.endswith("。"):
        ends = ["！", "？", "！？", "……", "っ！", "♪", "～", "ーっ！", "！！", "……？"]
        s = s[:-1] + ends[rng.integers(len(ends))]
    elif s[-1] not in "。！？…―～♪」』）" and rng.random() < D["terminal"]:
        s += choose(rng, CFG["terminal"])          # subtitle lines often lack the 。 a VN shows
    if rng.random() < D["dots"] and "、" in s:
        i = s.index("、")
        s = s[:i] + "……" + s[i + 1:]
    if rng.random() < D["elongation"] and lists.get("elongations"):
        s += lists["elongations"][rng.integers(len(lists["elongations"]))]
    if rng.random() < D["alnum"]:
        tok = alnum_token(rng)
        s = tok + "のはがをもでとに"[rng.integers(8)] + s
    B = CFG["brackets"]
    if rng.random() < (B["named"] if named else B["unnamed"]):
        s = ("『" + s + "』") if rng.random() < B["double"] else ("「" + s + "」")
    return clean(s), pool, focus


def alnum_token(rng):
    """Latin letters and digits as VN dialogue shows them (half or full width)."""
    full = rng.random() < 0.6
    kind = rng.random()
    if kind < 0.35:
        s = str(int(rng.integers(1, 3000)))
        s += ["", "円", "年", "人", "時", "分", "回", "％", "個", "日"][rng.integers(10)]
    elif kind < 0.6:
        s = ["CG", "SE", "BGM", "OK", "TV", "SNS", "No.", "Lv", "HP", "MP", "PC", "DVD", "CD", "ID", "NG", "VS",
             "AI", "UFO", "LOVE", "Yes", "No", "Hello", "Sir"][rng.integers(23)]
    elif kind < 0.8:
        s = f"{int(rng.integers(1, 4))}年{'ABCDEFG'[rng.integers(7)]}組"
    else:
        s = f"{int(rng.integers(0, 24))}:{int(rng.integers(0, 60)):02d}"
    if full:
        s = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in s)
    return clean(s)


def narration_text(rng):
    focus = None
    if rng.random() < CFG["focus"] * 0.5:
        focus = R.focus_char(rng)
        s, pool = R.context(rng, focus, "narration")
        if s is not None:
            return clean(s).strip("「」『』").lstrip("、。"), pool, focus
        focus = None
    pool = choose(rng, {k: v for k, v in CFG["narration_pools"].items() if k in R.pools})
    return clean(pool_line(rng, pool)).lstrip("、。"), pool, focus


def paginate(lines, max_lines, rng):
    """(page lines, page index, pages): text that overflows a box of max_lines continues on the
    next page, and a screenshot is equally likely to show any page. Version 2 always showed the
    first page, so an utterance that overflowed never showed its closing 」 (9.3 per 1,000
    characters against the real crops' 16.1)."""
    if len(lines) <= max_lines:
        return lines, 0, 1
    pages = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]
    k = int(rng.integers(len(pages)))
    return pages[k], k, len(pages)


UI_WORDS = [w.replace("_", " ") for w in """
セーブ ロード オート スキップ ログ バックログ システム コンフィグ 設定
環境設定 タイトル タイトルへ戻る タイトルに戻る ゲーム終了 終了 はい
いいえ 決定 キャンセル 戻る 次へ 前へ クイックセーブ クイックロード
Q.セーブ Q.ロード ボイス 音量 テキスト速度 ウィンドウ 既読スキップ
全画面 回想 CGモード ギャラリー エクストラ おまけ 音楽鑑賞 シーン回想
はじめから つづきから 続きから NEW_GAME LOAD CONFIG
EXTRA EXIT AUTO SKIP LOG SAVE Q.SAVE
Q.LOAD CLOSE BACK MENU データ セーブデータ
NO_DATA プロローグ エピローグ 選択肢 体験版 上書きしますか？
セーブしますか？ ロードしますか？ タイトル画面に戻りますか？ BGM SE
VOICE 画面設定 文字表示速度 オート速度 既読 未読 閉じる ヘルプ
マップ アイテム ステータス
""".split()]


UI_LATIN = [w for w in UI_WORDS if LATIN.search(w)]
UI_JAPANESE = [w for w in UI_WORDS if not LATIN.search(w)]
# v3: place and status labels of VN screens (map, scene title, status window), the other short
# unpunctuated crops the real training crops hold besides names and menus
UI_PLACES = """
教室 屋上 廊下 中庭 校門 校庭 体育館 図書室 保健室 職員室 生徒会室 部室 音楽室 理科室 食堂 購買 寮
自室 自宅 居間 台所 玄関 浴室 寝室 公園 駅前 商店街 神社 寺 喫茶店 繁華街 広場 河原 海岸 砂浜 港
病院 森 湖 山道 丘 城 城下町 王宮 宿屋 酒場 武器屋 道具屋 教会 墓地 村 町外れ 路地裏 屋敷 研究所
放課後 昼休み 早朝 午前 午後 夕方 夜 深夜 翌朝 数日後 回想 体力 気力 好感度 所持金 攻撃力 防御力
素早さ 魅力 知力 経験値 名前 年齢 職業 所属 装備 状態 移動 会話 調べる 話す 考える 持ち物 表情 主人公
""".split()


def ui_text(rng):
    r = rng.random()
    if r < CFG["ui"]["places"]:
        s = UI_PLACES[rng.integers(len(UI_PLACES))]
    elif r < 0.7:        # v3: Latin menu words (SAVE, Q.LOAD, CGモード) at CFG ui.latin_word, not 27%
        words = UI_LATIN if rng.random() < CFG["ui"]["latin_word"] else UI_JAPANESE
        s = words[rng.integers(len(words))]
    elif r < 0.85:
        s = f"第{'一二三四五六七八九十'[rng.integers(10)]}{['章', '話', '幕', '部'][rng.integers(4)]}"
    else:
        s = f"{int(rng.integers(1, 13))}月{int(rng.integers(1, 29))}日（{'月火水木金土日'[rng.integers(7)]}）"
    if rng.random() < 0.5:
        s = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in s)
    return clean(s), "ui", None


def short_text(rng, named):
    """A short spoken line: an interjection, stammer or name call, or a short pool line."""
    if rng.random() < 0.5:
        s, pool = template_line(rng), "templates"
    else:
        pool = choose(rng, {k: CFG["dialogue_pools"][k] for k in ("opensubtitles", "jesc", "nucc", "narikiri",
                                                                    "tatoeba") if k in R.pools})
        s = pool_line(rng, pool)
        for _ in range(30):
            if len(s) <= 7:
                break
            s = pool_line(rng, pool)
    s = clean(s).strip("「」『』")
    if s and s[-1] not in "。！？…―～♪っ" and rng.random() < 0.7:
        s += choose(rng, CFG["terminal"])
    B = CFG["brackets"]
    if s and rng.random() < (B["named"] if named else B["unnamed"]):
        s = "「" + s + "」"
    return clean(s), pool, None


def choice_text(rng):
    s, pool, focus = dialogue_text(rng, False)
    s = s.strip("「」『』")
    cut = [i + 1 for i, c in enumerate(s) if c in "、。！？" and 3 <= i < 18]
    s = s[:cut[0]] if cut else s[:int(rng.integers(5, 16))]
    return (s.rstrip("、。") or s), pool, focus


def sample_name(rng):
    """(display, label, source): the display may hold a full-width space (not labelled)."""
    N = CFG["names"]
    r = rng.random()
    if r < N["unknown"]:
        name, src = "？？？", "unknown"
    elif r < N["unknown"] + N["role"]:
        roles = ["少女", "少年", "男", "女", "先生", "店員", "男子生徒", "女子生徒", "母", "父", "妹", "姉", "兄",
                 "老人", "兵士", "声", "電話", "アナウンス", "謎の声", "クラスメイト"]
        name, src = roles[rng.integers(len(roles))], "role"
    else:
        src = choose(rng, {"vndb": N["vndb"], "syosetu": N["syosetu"], "generated": N["generated"]})
        if src != "generated" and R.names.get(src):
            parts = R.names[src][rng.integers(len(R.names[src]))]
            if len(parts) == 2:
                form = rng.random()
                if form < 0.45:
                    name = parts[1]
                elif form < 0.7:
                    name = parts[0]
                else:
                    name = parts[0] + ("　" if rng.random() < N["space"] else "") + parts[1]
            else:
                name = parts[0]
        else:
            src = "generated"
            if rng.random() < 0.45:
                name = "".join(rng.choice(R.katakana, size=int(rng.integers(2, 6))))
            else:
                k = int(rng.choice([1, 2, 2, 3]))
                name = "".join(rng.choice(R.name_kanji, size=k, p=R.name_kanji_p))
    if src not in ("unknown", "role") and rng.random() < N["honorific"] and R.lists.get("honorifics"):
        name += R.lists["honorifics"][rng.integers(len(R.lists["honorifics"]))]
    if rng.random() < N["brackets"]:
        name = "【" + name + "】"
    name = "".join(c for c in name if c in R.vocab_set or c == "　")
    return name, name.replace("　", ""), src


# ----------------------------------------------------------------------------- fonts & raster

def pick_font(rng, chars, platform, cat=None):
    weights = CFG["category"]["vita" if platform == "vita" else "psp"]
    for _ in range(4):
        c = cat or choose(rng, {k: v for k, v in weights.items() if k in R.by_cat})
        fams = [f for f in R.by_cat[c].values() if any(chars <= x.chars for x in f)]
        if c == "pixel":
            fams = [f for f in fams if any(px_fits(x, platform) for x in f)]
        if fams:
            fam = fams[rng.integers(len(fams))]
            files = [x for x in fam if chars <= x.chars and (c != "pixel" or px_fits(x, platform))]
            if files:
                w = np.array([1.0 + (0.8 if 500 <= x.weight <= 750 else 0.0) for x in files])
                return files[rng.choice(len(files), p=w / w.sum())]
        cat = None
    cands = [f for f in R.fonts if chars <= f.chars and f.cat != "pixel"]
    return cands[rng.integers(len(cands))] if cands else None


def px_fits(font, platform):
    em = font.pixel_em or 0
    return 10 <= em <= 16 if platform in ("psp", "psp2x") else 12 <= em <= 20


class Layer3:
    """Text masks (fills per colour index, outer and inner outline) drawn on a canvas at
    `scale` canvas px per output px, then resampled to the region: supersampling (BOX),
    glyph atlases (bilinear or area) and pixel-font integer upscales (nearest)."""

    def __init__(self, w, h, scale, aa, n_fill, sw, sw2, resample=Image.BOX, integer=False):
        self.w, self.h, self.ss, self.sw, self.sw2 = w, h, scale, sw, sw2
        self.resample, self.integer = resample, integer
        size = (max(1, int(math.ceil(w * scale))), max(1, int(math.ceil(h * scale))))
        self.fill = [Image.new("L", size, 0) for _ in range(n_fill)]
        self.s1 = Image.new("L", size, 0) if sw else None
        self.s2 = Image.new("L", size, 0) if sw2 else None
        self.ruby = Image.new("L", size, 0)
        self.d = [ImageDraw.Draw(m) for m in self.fill]
        self.d1 = ImageDraw.Draw(self.s1) if sw else None
        self.d2 = ImageDraw.Draw(self.s2) if sw2 else None
        self.dr = ImageDraw.Draw(self.ruby)
        for dr in self.d + [self.d1, self.d2, self.dr]:
            if dr is not None and not aa:
                dr.fontmode = "1"
        self.cursor_k = 0

    def _xy(self, x, base):
        xy = (x * self.ss, base * self.ss)
        return (round(xy[0]), round(xy[1])) if self.integer else xy

    def glyph(self, x, base, ch, font, k=0, small=False, ruby=False):
        xy = self._xy(x, base)
        self.d[k].text(xy, ch, font=font, fill=255, anchor="ls")
        if ruby:
            self.dr.text(xy, ch, font=font, fill=255, anchor="ls")
        for dr, w in ((self.d1, self.sw), (self.d2, self.sw2)):
            if dr is not None:
                sw = max(w - 1, 1) if small else w
                dr.text(xy, ch, font=font, fill=255, anchor="ls", stroke_width=max(1, round(sw * self.ss)),
                        stroke_fill=255)

    def poly(self, pts, k=0, ellipse=False):
        p = [(x * self.ss, y * self.ss) for x, y in pts]
        for dr in [self.d[k], self.d1, self.d2]:
            if dr is None:
                continue
            if ellipse:
                dr.ellipse((p[0][0], p[0][1], p[1][0], p[1][1]), fill=255)
            else:
                dr.polygon(p, fill=255)

    def arrays(self, q4_canvas=False, q4=False, gamma=1.0):
        def arr(im):
            if im is None:
                return None
            a = np.asarray(im, np.float32) / 255.0
            if q4_canvas:
                a = np.round(a * 15) / 15
            if a.shape != (self.h, self.w):
                im2 = Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
                a = np.asarray(im2.resize((self.w, self.h), self.resample), np.float32) / 255.0
            if q4:
                a = np.round(a * 15) / 15
            if gamma != 1.0:
                a = a ** gamma
            return a
        return [arr(m) for m in self.fill], arr(self.s1), arr(self.s2), arr(self.ruby)


# ----------------------------------------------------------------------------- style

def sample_style(rng, platform, effects=None):
    """synth2's palette and effects (render.sample_style, run with synth3's layout weights),
    then outline and shadow rates per console, or per `effects` (v3: "scaled", PC renders,
    whose text is more often outlined or shadowed; docs/synth3-v3.md)."""
    saved, saved_s = S2.LAYOUT_WEIGHT, S2.SCHEME_WEIGHT
    S2.LAYOUT_WEIGHT = CFG["layout"]
    S2.SCHEME_WEIGHT = dict(saved_s, ui=saved_s["choice"])
    try:
        st = S2.sample_style(rng, "vita" if platform == "vita" else "psp")
    finally:
        S2.LAYOUT_WEIGHT, S2.SCHEME_WEIGHT = saved, saved_s
    con = "vita" if platform == "vita" else "psp"
    fx = effects or con
    p_out = CFG["outline"][fx].get(st["scheme"])
    if st["stroke"] is None and p_out and rng.random() < p_out:
        w = 1 if con == "psp" or rng.random() < 0.5 else int(rng.choice([2, 3], p=[0.8, 0.2]))
        dark = S2.luma(st["fill"]) > 128
        st["stroke"] = {"w": w, "c": S2.dark_colour(rng, 0.25) if dark else S2.light_colour(rng, 0.15, 0.9)}
    if st["shadow"] is None and rng.random() < CFG["shadow"][fx] * 0.5:
        d = 1 if con == "psp" or rng.random() < 0.6 else 2
        st["shadow"] = {"dx": d, "dy": d, "blur": 0.7 if rng.random() < 0.3 else 0.0,
                        "c": S2.dark_colour(rng, 0.15) if st["on_dark"] else S2.light_colour(rng, 0.1, 0.85),
                        "a": float(rng.uniform(0.55, 1.0))}
    return st


# ----------------------------------------------------------------------------- region

def font_size(rng, platform, font):
    if font.cat == "pixel":
        em = font.pixel_em
        mult = 1 if platform in ("psp", "psp2x") or em >= 17 else 2
        return em * mult, em, mult
    if platform in ("psp", "psp2x"):
        sizes, p = CFG["psp_sizes"]
        fs = int(rng.choice(sizes, p=np.asarray(p) / np.sum(p)))
    else:   # native Vita crops: ink p50 24 px (teacher_train, kind vita)
        r = rng.random()
        fs = int(rng.integers(17, 21)) if r < 0.10 else int(rng.integers(30, 35)) if r < 0.18 else \
            int(np.clip(round(rng.normal(25, 2.4)), 21, 30))
    return fs, fs, 1


def raster_mode(rng, platform, font):
    if font.cat == "pixel":
        return "pixel"
    return choose(rng, CFG["raster"]["vita" if platform == "vita" else "psp"])


def raster_setup(rng, mode, fs, pix_mult):
    """Canvas parameters of a raster mode: (scale, anti-aliased, resample, integer origins, tag)."""
    if mode == "pixel":
        return 1.0 / pix_mult, False, Image.NEAREST, True, "pixel%d" % pix_mult
    if mode == "mono":
        return 1.0, False, Image.BOX, True, "1bpp"
    if mode == "ss2":
        return 2.0, True, Image.BOX, False, "ss2"
    if mode == "atlas":
        cell = float(rng.choice([32, 48])) if fs >= 18 else float(rng.choice([16, 24, 32]))
        res = Image.BILINEAR if rng.random() < 0.6 else Image.BOX
        return max(1.0, cell / fs), True, res, True, f"atlas{int(cell)}" + ("b" if res == Image.BILINEAR else "a")
    return 1.0, True, Image.BOX, False, mode


def render_region(rng, platform=None):
    """(rgb uint8, layout, full-ink bool mask, meta) or None for an unusable draw."""
    platform = platform or choose(rng, CFG["platform"])
    up = 1.0
    target = platform
    if platform == "scaled":
        target = "psp" if rng.random() < CFG["scaled_to_psp"] else "vita"
        up = float(rng.uniform(1.25, 2.0)) * (2.0 if target == "psp" else 1.0)
        geom = "vita"
    else:
        geom = "vita" if platform == "vita" else "psp"
    st = sample_style(rng, geom, "scaled" if platform == "scaled" else None)
    lay, sch = st["layout"], st["scheme"]
    psp = geom == "psp"
    SW, SH = (960, 544) if not psp else (480, 272)
    if platform == "scaled":
        SW, SH = int(SW * up / (2.0 if target == "psp" else 1.0)), int(SH * up / (2.0 if target == "psp" else 1.0))
    con = "psp" if psp else "vita"

    # ---- name and text
    name_mode = None
    if lay in ("adv", "band", "bare"):
        name_mode = choose(rng, CFG["name_mode"][con])
        if lay != "adv" and name_mode == "tab":
            name_mode = "line"
    elif lay == "nvl" and rng.random() < CFG["nvl_name"]:
        name_mode = "line"
    name_disp = name_label = name_src = None
    if name_mode:
        name_disp, name_label, name_src = sample_name(rng)
        if not name_label:
            name_mode = None
    narrate = (lay in ("nvl", "bare") and rng.random() < CFG["narrate_nvl"]) or \
        (name_mode is None and rng.random() < CFG["narration"])
    sources, focus = [], []
    if lay in ("choice", "ui"):
        lo, hi = CFG["ui"]["lines"]
        n_lines = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.2])) if lay == "choice" else int(rng.integers(lo, hi + 1))
        texts = []
        for _ in range(n_lines):
            t, pool, fc = choice_text(rng) if lay == "choice" else ui_text(rng)
            texts.append(t)
            sources.append(pool)
            focus.append(fc)
        if min(len(t) for t in texts) < 1:
            return None
        chars = set("".join(texts))
    else:
        if narrate:
            text, pool, fc = narration_text(rng)
        elif lay in ("adv", "band", "balloon") and rng.random() < CFG["short"]:
            text, pool, fc = short_text(rng, name_mode is not None)
        else:
            text, pool, fc = dialogue_text(rng, name_mode is not None)
        if not text:
            return None
        sources.append(pool)
        focus.append(fc)
        chars = set(text)
    if name_disp:
        chars |= set(name_label)
    font = (pick_font(rng, chars, "vita", cat=choose(rng, {k: v for k, v in CFG["category"]["vita"].items()
                                                          if k != "pixel" and k in R.by_cat}))
            if platform == "scaled" else pick_font(rng, chars, platform))
    if font is None:
        return None
    u = up / (2.0 if target == "psp" else 1.0) if platform == "scaled" else 1.0   # virtual px per geometry px
    fs, em, pix_mult = font_size(rng, "psp" if psp else "vita", font)
    if platform == "scaled":
        mu, sd, lo, hi = CFG["scaled_em"][target]
        fs = int(round(float(np.clip(rng.normal(mu, sd), lo, hi)) * up))
    wght = S2.pick_weight(rng, font.path)
    mode = raster_mode(rng, "psp" if psp else "vita", font)
    ck, aa, resample, integer, mode_detail = raster_setup(rng, mode, fs, pix_mult)

    def canvas_font(size):          # the font at canvas size for an output em of `size`
        return f if mode == "pixel" else S2.get_font(font.path, max(1, int(round(size * ck))), wght)
    f = S2.get_font(font.path, em, wght) if mode == "pixel" else None
    f = canvas_font(fs)
    track = 0 if rng.random() < 0.65 else int(rng.integers(-1, 3 if psp else 4))
    sw = int(round((st["stroke"]["w"] if st["stroke"] else 0) * u))
    sw2 = int(round((st["stroke2"]["w"] if st["stroke2"] else 0) * u))
    if st["stroke"]:
        st["stroke"] = dict(st["stroke"], w=max(1, sw))
        sw = st["stroke"]["w"]
    if st["stroke2"]:
        st["stroke2"] = dict(st["stroke2"], w=max(1, sw2))
        sw2 = st["stroke2"]["w"]
    sh = st["shadow"]
    if sh and u != 1.0:
        sh = st["shadow"] = dict(sh, dx=int(round(sh["dx"] * u)), dy=int(round(sh["dy"] * u)))
    dy = sh["dy"] if sh else 0

    def pad(a, b):
        return int(int(rng.integers(a, b)) * u) // (2 if psp else 1)

    def cadv(ch, fnt):         # advance in output px of a canvas-size font
        return fnt.getlength(ch) / ck

    def measure_w(fnt, text_):
        return sum(cadv(c, fnt) for c in text_) + track * max(len(text_) - 1, 0)

    # ---- lines
    page, pages = 0, 1
    if lay in ("choice", "ui"):
        lines = [S2.add_spaces(t, rng) for t in texts]
        box_w = None
    else:
        box_w = rng.uniform(0.55, 0.97) * SW
        cap = max(4, int((box_w - 2 * fs) // max(fs + track, 6)))
        cpl = max(5, int(cap * rng.uniform(0.6, 1.0)))
        max_lines = {"adv": int(rng.choice([1, 2, 3, 4], p=[0.25, 0.33, 0.32, 0.10])),
                     "band": int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])),
                     "nvl": int(rng.integers(3, 8)),
                     "balloon": int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])),
                     "bare": int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))}[lay]
        if lay == "balloon":
            cpl = min(cpl, int(rng.integers(8, 20)))
        wrapped = S2.wrap(S2.add_spaces(text, rng), cpl, CFG["pages"]["wrap_lines"], rng, at_punct=rng.random() < 0.35)
        lines, page, pages = paginate(wrapped, max_lines, rng)
        if lay == "nvl" and rng.random() < 0.5 and len(lines) > 1 and page == pages - 1:
            more, pool2, fc2 = narration_text(rng)
            extra = [ln for ln in S2.wrap(more, cpl, max(0, max_lines - len(lines)), rng)
                     if set(ln) <= font.chars | {"　"}]
            if extra:
                lines += extra
                sources.append(pool2)
                focus.append(fc2)
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    first = lines[0] if page == 0 else text.lstrip("　")
    indent = rng.random() < 0.5 and first[:1] in ("「", "『", "（") and lay not in ("choice", "ui", "balloon")
    inline_name = name_mode == "inline" and page == 0     # an inline name opens the utterance, not page 2
    if inline_name:
        lines[0] = name_disp + "　" * int(rng.choice([0, 0, 1, 2])) + lines[0]
    labels = [ln.replace("　", "") for ln in lines]

    # ---- colour spans
    n_fill = 1
    colour_of = [[0] * len(ln) for ln in lines]
    C = CFG["coloured"]
    if lay not in ("choice", "ui") and rng.random() < C["span"]:
        li = int(rng.integers(len(lines)))
        if len(lines[li]) >= 3:
            a = int(rng.integers(0, len(lines[li]) - 1))
            b = min(len(lines[li]), a + int(rng.integers(2, 7)))
            for k in range(a, b):
                colour_of[li][k] = 1
            n_fill = 2
    elif lay not in ("choice", "ui") and rng.random() < C["line"]:
        li = int(rng.integers(len(lines)))
        colour_of[li] = [1] * len(lines[li])
        n_fill = 2
    name_coloured = name_mode is not None and rng.random() < CFG["name_coloured"]
    if inline_name and name_coloured:
        for k in range(len(name_disp)):
            colour_of[0][k] = 1
        n_fill = 2
    cursor_k = n_fill
    n_fill += 1

    # ---- ruby / partial / cursor
    ruby = lay in ("adv", "band", "nvl", "bare") and rng.random() < CFG["ruby"][con] and mode != "pixel"
    ruby_fs = max(6 if psp else 9, int(round(fs * rng.uniform(0.42, 0.55))))
    ruby_gap = int(rng.choice([0, 1, 2], p=[0.3, 0.45, 0.25]))
    ruby_space = ruby_fs + ruby_gap if ruby else 0
    partial = None
    long_lines = [i for i, ln in enumerate(lines) if len(ln) >= 2]
    if lay in ("adv", "band", "nvl") and rng.random() < CFG["partial"]["p"] and long_lines:
        li = int(long_lines[rng.integers(len(long_lines))])
        keep = int(rng.integers(1, len(lines[li]) + 1))
        partial = {"line": li, "ramp": rng.random() < CFG["partial"]["ramp"],
                   "rest": lines[li][keep] if keep < len(lines[li]) else "",
                   "ghost": rng.random() < 0.4, "ghost_a": float(rng.uniform(0.06, 0.2))}
        lines, labels, colour_of = lines[:li + 1], labels[:li + 1], colour_of[:li + 1]
        lines[li] = lines[li][:keep]
        labels[li] = lines[li].replace("　", "")
        colour_of[li] = colour_of[li][:keep]
        if not labels[li]:
            return None
    cursor = None
    if lay in ("adv", "band", "balloon", "nvl") and (partial is None) and rng.random() < CFG["cursor"]:
        cursor = {"kind": S2.CURSORS[rng.integers(len(S2.CURSORS))], "inline": rng.random() < 0.6,
                  "size": fs * rng.uniform(0.5, 0.9), "same": rng.random() < 0.4}

    # ---- geometry
    widths = [measure_w(f, ln) for ln in lines]
    pitch = max(fs * rng.uniform(*CFG["pitch"]), fs + 2 * max(sw, sw2) + dy + (1 if psp else 2)) + ruby_space
    if lay == "nvl":
        pitch = max(pitch, fs * rng.uniform(1.4, 2.0) + ruby_space)
    pl, pr = pad(8, 48), pad(8, 48)
    pt, pb = pad(6, 26), pad(6, 28)
    if lay == "nvl":
        pl, pt = pad(16, 70), pad(10, 40)
    out_l = out_r = out_t = out_b = 0
    show_border = lay in ("adv", "balloon", "choice") and rng.random() < 0.55
    if show_border or lay in ("bare", "balloon", "choice"):
        out_l, out_r = pad(0, 26), pad(0, 26)
        out_t, out_b = pad(0, 22), pad(0, 22)
    name_fs = fs if mode == "pixel" else max(10 if psp else 14, int(round(fs * rng.uniform(0.8, 1.02))))
    nf = canvas_font(name_fs) if name_mode in ("tab", "line") else None
    tab = None
    if name_mode == "tab":
        nw = measure_w(nf, name_disp)
        tab_h = name_fs * rng.uniform(1.35, 1.9)
        tab_w = max(nw + 2 * name_fs * rng.uniform(0.35, 1.4), name_fs * rng.uniform(3, 7))
        overlap = tab_h * rng.uniform(-0.15, 0.55)
        out_t = max(out_t, int(math.ceil(tab_h - overlap)) + pad(0, 8))
        tab = {"w": tab_w, "h": tab_h, "overlap": overlap, "dx": name_fs * rng.uniform(0, 2.5), "nw": nw}
    name_line = name_mode == "line"
    name_line_h = (name_fs * 1.25 + pitch - fs) if name_line else 0
    inner_w = max(widths) + (fs if indent else 0) + ((cursor["size"] + fs * 0.8) if cursor else 0)
    if lay in ("adv", "band", "nvl") and rng.random() < 0.6:
        inner_w = max(inner_w, box_w - pl - pr)
    if lay in ("choice", "ui"):
        bar_w = max(widths) + 2 * fs * rng.uniform(0.8, 3.0)
        bar_h = fs * rng.uniform(1.5, 2.1) + 2 * max(sw, sw2)
        bar_gap = fs * rng.uniform(0.25, 1.0)
        w = int(out_l + bar_w + out_r)
        h = int(out_t + len(lines) * bar_h + (len(lines) - 1) * bar_gap + out_b)
    else:
        text_h = name_line_h + ruby_space + (len(lines) - 1) * pitch + fs + 2 * max(sw, sw2) + dy
        w = int(out_l + pl + inner_w + pr + out_r)
        h = int(out_t + pt + text_h + pb + out_b)
    if lay == "balloon":
        h += int(fs * 0.8)
    if w > SW or h > SH or w < 24 or h < 16:
        return None

    # ---- scene and boxes
    img, bg_tag = background(rng, w, h, platform, target, up)
    boxes = []
    rect = (out_l, out_t, w - out_r - 1, h - out_b - 1 - (int(fs * 0.8) if lay == "balloon" else 0))
    BA = CFG["box_alpha"]["scaled" if platform == "scaled" else "psp" if psp else "vita"]
    p_lo, lo, hi = BA["on_dark"] if st["on_dark"] else BA["light"]
    box_alpha = float(rng.uniform(*lo) if rng.random() < p_lo else rng.uniform(*hi))
    if sch == "on_colour":
        box_alpha = float(rng.uniform(*BA["on_colour"]))
    border = None
    if show_border:
        bw = 1 if psp else int(rng.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.2, 0.1]))
        bc = S2.light_colour(rng, 0.5, 0.75) if st["on_dark"] else (S2.dark_colour(rng, 0.4) if rng.random() < 0.5
                                                                    else S2.accent(rng, False))
        border = (bw, bc)
    pat = float(rng.uniform(4, 16)) if rng.random() < 0.3 else 0.0
    col2 = np.clip(st["box"] + rng.normal(0, 25, 3), 0, 255) if rng.random() < 0.35 else None
    if lay in ("adv", "nvl"):
        radius = 0 if rng.random() < 0.4 else int(rng.uniform(4, 22) / (2 if psp else 1))
        alpha = float(rng.uniform(0.35, 0.75)) if lay == "nvl" else box_alpha
        if lay == "adv" and rng.random() < 0.18:
            alpha = (float(rng.uniform(0.15, 0.6)) * box_alpha, box_alpha)
        boxes.append({"shape": "round" if radius else "rect", "rect": rect, "radius": radius, "colour": st["box"],
                      "colour2": col2, "alpha": alpha, "pattern": pat, "border": border,
                      "frame": S2.frame_style(rng, psp, st, border)})
    elif lay == "band":
        a0 = float(rng.uniform(0.0, 0.7)) * box_alpha if rng.random() < 0.5 else box_alpha
        boxes.append({"shape": "rect", "rect": (-2, rect[1], w + 2, h + 2), "colour": st["box"], "colour2": col2,
                      "alpha": (a0, box_alpha), "pattern": pat, "border": None})
    elif lay == "balloon":
        x0, y0, x1, y1 = rect
        tx = rng.uniform(x0 + (x1 - x0) * 0.2, x0 + (x1 - x0) * 0.8)
        tail = [(tx - fs * 0.5, y1 - 2), (tx + fs * 0.5, y1 - 2), (tx + rng.uniform(-1, 1) * fs, y1 + fs * 0.75)]
        boxes.append({"shape": "ellipse" if rng.random() < 0.3 else "round", "rect": rect,
                      "radius": int((y1 - y0) * rng.uniform(0.2, 0.5)), "colour": st["box"],
                      "alpha": float(rng.uniform(0.9, 1.0)), "pattern": 0.0, "tail": tail,
                      "border": (1 if psp else int(rng.integers(2, 4)), S2.dark_colour(rng, 0.2))})
    elif lay in ("choice", "ui"):
        radius = int(rng.uniform(0, 0.5) * bar_h)
        bar_alpha = float(rng.uniform(0.5, 0.85)) if rng.random() < 0.5 else box_alpha
        hl = int(rng.integers(len(lines))) if rng.random() < 0.5 else -1
        for i in range(len(lines)):
            y0 = out_t + i * (bar_h + bar_gap)
            c = st["box"] if i != hl else np.clip(st["box"] + rng.normal(0, 45, 3), 0, 255)
            boxes.append({"shape": "round" if radius else "rect", "rect": (out_l, y0, w - out_r - 1, y0 + bar_h),
                          "radius": radius, "colour": c, "alpha": bar_alpha, "pattern": 0.0, "border": border})
    for b in boxes:
        S2.draw_box(img, rng, b, st)
    if tab is not None:
        tab_c = st["box"] if rng.random() < 0.3 else (S2.dark_colour(rng, 0.35) if rng.random() < 0.4 else S2.hsv(
            rng.random(), rng.uniform(0.3, 0.9), rng.uniform(0.35, 0.95)))
        tx0 = out_l + tab["dx"]
        ty1 = out_t + tab["overlap"]
        tab["rect"] = (tx0, ty1 - tab["h"], min(w - 1, tx0 + tab["w"]), ty1)
        tr = int(rng.uniform(0, 0.5) * tab["h"])
        S2.draw_box(img, rng, {"shape": "round" if tr else "rect", "rect": tab["rect"], "radius": tr, "colour": tab_c,
                               "colour2": None, "alpha": float(rng.uniform(0.6, 1.0)),
                               "pattern": float(rng.uniform(4, 12)) if rng.random() < 0.25 else 0.0,
                               "border": border if rng.random() < 0.5 else None}, st)
        tab["c"] = tab_c

    # ---- text
    layer = Layer3(w, h, ck, aa, n_fill, sw, sw2, resample, integer)
    layer.cursor_k = cursor_k
    layout = []
    # v3: eval's and teacher's line boxes come from refine(), whose white-text and black-text
    # masks find the glyph fill (+1 px), so the outline or shadow of white or black text sits
    # inside the 4 / 2 px pad of a real crop; synth3's gt box follows that rule for such lines
    fill_box = (st["grad"] is None and bool(st["stroke"] or st["shadow"] or st["stroke2"])
                and (min(st["fill"]) >= 200 or max(st["fill"]) <= 70))
    x_left = out_l + pl
    y = out_t + pt + ruby_space
    base_k = 1.0 if mode == "pixel" else 0.88
    if name_line:
        nx = x_left
        base = y + base_k * name_fs
        for ch in name_disp:
            if ch != "　":
                layer.glyph(nx, base, ch, nf, 1 if name_coloured else 0)
            nx += cadv(ch, nf)
        if name_coloured:
            n_fill = max(n_fill, 2)
        layout.append({"text": name_label, "top": y, "fs": name_fs, "x0": x_left, "x1": nx, "kind": "name",
                       "xs": None, "fillbox": fill_box and not name_coloured})
        y += name_line_h
    tops = ([out_t + i * (bar_h + bar_gap) + (bar_h - fs) / 2 for i in range(len(lines))] if lay in ("choice", "ui")
            else [y + i * pitch for i in range(len(lines))])
    ghost = None
    for i, ln in enumerate(lines):
        top = tops[i]
        base = top + base_k * fs
        x = (w - widths[i]) / 2 if lay in ("choice", "ui", "balloon") else \
            x_left + (fs if indent and (i > 0 or page > 0) else 0)
        xs = []
        for k, ch in enumerate(ln):
            xs.append(x)
            if ch != "　":
                layer.glyph(x, base, ch, f, colour_of[i][k])
            x += cadv(ch, f) + track
        x_end = x - track
        if ruby:
            draw_ruby(rng, layer, ln, xs, f, top, ruby_fs, ruby_gap, font, wght, cadv, mode)
        if partial is not None and i == len(lines) - 1 and partial["ghost"] and partial["rest"]:
            ghost = (x, base, partial["rest"])
        if cursor is not None and i == len(lines) - 1:
            s = cursor["size"]
            if cursor["inline"]:
                cx, cy = x_end + fs * rng.uniform(0.15, 0.6) + s / 2, top + fs * rng.uniform(0.45, 0.8)
            else:
                cx, cy = w - out_r - pr * rng.uniform(0.2, 0.8) - s / 2, h - out_b - pb * rng.uniform(0.3, 0.9) - s / 2
            if cx + s / 2 < w - 1 and cy + s / 2 < h - 1:
                S2.draw_cursor(layer, cursor["kind"], cx, cy, s)
        kind = lay if lay in ("choice", "ui") else ("name+dialogue" if inline_name and i == 0 else
                                                 "narration" if narrate else "dialogue")
        layout.append({"text": labels[i], "top": top, "fs": fs, "x0": xs[0] if xs else x, "x1": x_end, "kind": kind,
                       "ruby": bool(ruby and any(S2.is_kanji(c) for c in ln)), "xs": xs, "line": ln,
                       "coloured": bool(any(colour_of[i])), "fillbox": fill_box and not any(colour_of[i])})
    q4c = mode == "atlas" and rng.random() < 0.7
    q4 = mode == "aa4"
    gamma = 1.0
    if rng.random() < CFG["gamma"]["p"] and mode not in ("pixel", "mono"):
        gamma = float(math.exp(rng.uniform(math.log(CFG["gamma"]["lo"]), math.log(CFG["gamma"]["hi"]))))
    fills, s1, s2_, ruby_m = layer.arrays(q4c, q4, gamma)
    # typewriter: the last glyphs fade in (per-glyph alpha); a glyph is labelled when alpha >= 0.5
    if partial is not None and partial["ramp"]:
        L = layout[-1]
        n = len(L["line"])
        ramp = min(n, int(rng.integers(2, 7)))
        alphas = np.ones(n)
        alphas[n - ramp:] = np.linspace(1.0, float(rng.uniform(0.05, 0.3)), ramp + 1)[1:]
        alphas = np.where((alphas > 0.3) & (alphas < 0.7), np.where(alphas >= 0.5, 0.75, 0.25), alphas)
        r0, r1 = max(0, int(L["top"] - ruby_space - 2)), min(h, int(L["top"] + fs * 1.3 + 2 * max(sw, sw2) + dy + 2))
        for k in range(n):
            if alphas[k] >= 1:
                continue
            c0 = max(0, int(math.floor(L["xs"][k])) - max(sw, sw2) - 1)
            c1 = min(w, int(math.ceil(L["xs"][k] + cadv(L["line"][k], f))) + max(sw, sw2) + 1)
            for m in fills + [s1, s2_]:
                if m is not None:
                    m[r0:r1, c0:c1] *= alphas[k]
        vis = "".join(ch for ch, a in zip(L["line"], alphas) if a >= 0.5).replace("　", "")
        if not vis:
            return None
        L["text"] = vis
        L["fade"] = int((alphas < 1).sum())
    cols = [st["fill"], st["accent"]] + [st["fill"]] * max(0, len(fills) - 2)
    if cursor is not None:
        cols[cursor_k] = st["fill"] if cursor["same"] else S2.accent(rng, st["on_dark"])
    grad_rows = None
    if st["grad"] is not None:
        t = np.full(h, 0.5, np.float32)
        for L in layout:
            r0, r1 = int(L["top"]), int(L["top"] + L["fs"])
            t[max(r0, 0):max(r1, 0)] = np.linspace(0, 1, max(r1, 0) - max(r0, 0), dtype=np.float32)
        grad_rows = (np.asarray(st["fill"], np.float32)[None, :] * (1 - t[:, None])
                     + np.asarray(st["grad"], np.float32)[None, :] * t[:, None])[:, None, :]
    core, ink = S2.compose_text(img, fills, s1, s2_, cols, st, grad_rows)
    if ghost is not None:                   # faint next glyph: drawn, never inked or labelled
        gl = Layer3(w, h, ck, aa, 1, 0, 0, resample, integer)
        gl.glyph(ghost[0], ghost[1], ghost[2], f)
        S2.over(img, gl.arrays()[0][0], st["fill"], partial["ghost_a"])
    if tab is not None:
        tl = Layer3(w, h, ck, aa, 1, sw if rng.random() < 0.5 else 0, 0, resample, integer)
        tnf = nf
        tk = ck
        on_tab_dark = S2.luma(tab["c"]) < 140
        nwid = sum(tnf.getlength(c) / tk for c in name_disp)
        nx = tab["rect"][0] + ((tab["w"] - nwid) / 2 if rng.random() < 0.5 else name_fs * 0.5)
        ntop = tab["rect"][1] + (tab["h"] - name_fs) / 2
        nbase = ntop + base_k * name_fs
        x = nx
        for ch in name_disp:
            if ch != "　":
                tl.glyph(x, nbase, ch, tnf)
            x += tnf.getlength(ch) / tk
        nfill = ((S2.light_colour(rng, 0.1, 0.9) if on_tab_dark else S2.dark_colour(rng, 0.25))
                 if not name_coloured else S2.accent(rng, on_tab_dark))
        tst = {"shadow": None, "glow": None, "stroke2": None,
               "stroke": {"w": tl.sw, "c": S2.dark_colour(rng, 0.2) if on_tab_dark else S2.light_colour(rng)}}
        tf, ts1, _, _ = tl.arrays(q4c, q4, gamma)
        tcore, tink = S2.compose_text(img, tf, ts1, None, [nfill], tst)
        np.maximum(core, tcore, out=core)
        np.maximum(ink, tink, out=ink)
        layout.insert(0, {"text": name_label, "top": ntop, "fs": name_fs, "x0": nx, "x1": x, "kind": "name",
                          "xs": None, "fillbox": bool(tl.sw) and (min(nfill) >= 200 or max(nfill) <= 70)})

    # ---- screen and capture
    cap = []
    if platform == "vita" and rng.random() < CFG["subnative"]:
        s = float(rng.choice([0.75, 2 / 3]))
        small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        cap.append("subnative")
    if psp and platform != "scaled" and rng.random() < CFG["dither565"]:
        img = dither565(img)
        cap.append("rgb565")
    if rng.random() < CFG["blur"]:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 0.7)))
        cap.append("blur")
    if rng.random() < CFG["noise"]:
        img = img + rng.normal(0, rng.uniform(1, 4), img.shape).astype(np.float32)
        cap.append("noise")
    out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    ink_f = ink.astype(np.float32)
    core_f = core.astype(np.float32)
    k3 = np.ones((3, 3), np.uint8)
    fill_src = cv2.dilate((core_f >= 0.5).astype(np.uint8), k3).astype(np.float32)   # refine(): fill + 1 px
    ruby_f = ruby_m
    k_final = 1.0
    if platform == "psp2x":
        mode2 = choose(rng, {k: v for k, v in CFG["capture"]["psp2x"].items() if k != "jpeg"})
        k_final = 1.995 if mode2 == "x1995" else 2.0
        interp = cv2.INTER_NEAREST if mode2 == "nearest" else cv2.INTER_LINEAR
        W2, H2 = int(round(w * k_final)), int(round(h * k_final))
        out = cv2.resize(out, (W2, H2), interpolation=interp)
        if mode2 == "scanlines":
            out[1::2] = (out[1::2].astype(np.float32) * 0.72).astype(np.uint8)
        cap.append(mode2)
        if rng.random() < CFG["capture"]["psp2x"]["jpeg"]:
            out = jpeg(out, 90, 2, rng)
            cap.append("q90")
    elif platform == "scaled":
        # v3: as a real PC screenshot reaches the training crops, VNDB's Q90 JPEG at the source
        # resolution, then the area downscale to the console size (teacher/realdata build.py)
        if choose(rng, CFG["capture"]["scaled"]) == "q90":
            out = jpeg(out, 90, 2 if rng.random() < 0.85 else 0, rng)
            cap.append("q90")
        k_final = 1.0 / up
        W2, H2 = max(8, int(round(w * k_final))), max(8, int(round(h * k_final)))
        out = cv2.resize(out, (W2, H2), interpolation=cv2.INTER_AREA)
        cap.append(f"area{k_final:.2f}")
    else:
        chain = choose(rng, CFG["capture"][con])
        if chain == "vndb2":
            out = jpeg(out, int(rng.integers(82, 95)), 2, rng)
            out = jpeg(out, 90, 2 if rng.random() < 0.85 else 0, rng)
        elif chain == "q90":
            out = jpeg(out, 90, 2, rng)
        cap.append(chain)
    if k_final != 1.0:
        H2, W2 = out.shape[:2]
        ink_f = cv2.resize(ink_f, (W2, H2), interpolation=cv2.INTER_AREA if k_final < 1 else cv2.INTER_LINEAR)
        ruby_f = cv2.resize(ruby_f, (W2, H2), interpolation=cv2.INTER_AREA if k_final < 1 else cv2.INTER_LINEAR)
        core_f = cv2.resize(core_f, (W2, H2), interpolation=cv2.INTER_AREA if k_final < 1 else cv2.INTER_LINEAR)
        if k_final < 1:     # refine() ran on the source screenshot; its box scales to every row it touches
            fill_src = cv2.resize(fill_src, (W2, H2), interpolation=cv2.INTER_AREA)
        for L in layout:
            for key in ("top", "x0", "x1"):
                L[key] *= k_final
            L["fs"] *= k_final
            if L.get("xs"):
                L["xs"] = [v * k_final for v in L["xs"]]
        sw, sw2, dy, ruby_space = sw * k_final, sw2 * k_final, dy * k_final, ruby_space * k_final
    ink_b = ink_f >= 0.5
    ruby_b = ruby_f >= 0.35
    for L in layout:
        L["y0"] = L["top"] - max(sw, sw2) - 1
        L["y1"] = L["top"] + L["fs"] + max(sw, sw2) + dy + 1
        rows = ink_b[max(int(L["y0"]), 0):max(int(L["y1"]), 0)]
        # columns of this line's own glyphs (plus outline and shadow): a neighbouring line whose
        # ink reaches into the band must not widen it (version 1 let 3% of crops span a neighbour)
        mx = max(sw, sw2) + dy + 2
        c0, c1 = max(0, int(math.floor(L["x0"] - mx))), max(0, int(math.ceil(L["x1"] + mx)))
        cols = np.flatnonzero(rows[:, c0:c1].any(axis=0)) + c0 if rows.size else []
        L["ix0"], L["ix1"] = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (-99, -99)
        L["ruby_top"] = None
        if L.get("ruby"):
            r0 = max(0, int(L["top"] - ruby_space - 2))
            band = ruby_b[r0:max(r0, int(L["top"] + 1)), max(0, L["ix0"]):max(0, L["ix1"])]
            rr = np.flatnonzero(band.any(axis=1))
            if len(rr):
                L["ruby_top"] = r0 + int(rr[0])
    meta = {"platform": platform, "target": target if platform == "scaled" else ("vita" if platform != "psp"
                                                                                else "psp"),
            "layout": lay, "scheme": sch, "cat": font.cat, "fam": font.fam, "font": font.key, "lic": font.licence,
            "fs": round(fs * k_final, 1), "em": em, "raster": mode_detail, "gamma": round(gamma, 2),
            "sw": round(sw, 2), "shadow": bool(sh), "glow": bool(st["glow"]), "grad": st["grad"] is not None,
            "name": name_mode, "name_src": name_src, "partial": partial is not None,
            "cursor": None if cursor is None else ("inline" if cursor["inline"] else "corner"),
            "capture": "+".join(cap), "border": bool(show_border), "pools": sources,
            "focus": [None if c is None else R.vocab[c] for c in focus],
            "bg": bg_tag, "box_alpha": round(box_alpha, 2) if boxes else None, "page": [page, pages]}
    meta["_ruby_mask"] = ruby_b
    if not any(L.get("fillbox") for L in layout):
        meta["_fill_mask"] = None
    elif k_final < 1:
        meta["_fill_mask"] = fill_src > 0.01
    else:                   # refine() runs on the screen as shown (native, or a 2x upscale)
        meta["_fill_mask"] = cv2.dilate((core_f >= 0.5).astype(np.uint8), k3).astype(bool)
    return out, layout, ink_b, meta


def draw_ruby(rng, layer, ln, xs, f, top, ruby_fs, gap, font, wght, cadv, mode):
    rf = S2.get_font(font.path, max(1, int(round(ruby_fs * layer.ss))), wght)
    i = 0
    while i < len(ln):
        if not S2.is_kanji(ln[i]):
            i += 1
            continue
        j = i
        while j < len(ln) and S2.is_kanji(ln[j]):
            j += 1
        if rng.random() < 0.65:
            kana = R.hiragana if rng.random() < 0.85 else R.katakana
            kana = [c for c in kana if c in font.chars]
            if kana:
                reading = "".join(rng.choice(kana, size=(j - i) * int(rng.integers(1, 3))))
                span0, span1 = xs[i], xs[j - 1] + cadv(ln[j - 1], f)
                width = sum(rf.getlength(c) / layer.ss for c in reading)
                x = (span0 + span1 - width) / 2
                base = top - gap - 0.12 * ruby_fs
                for ch in reading:
                    layer.glyph(x, base, ch, rf, 0, small=True, ruby=True)
                    x += rf.getlength(ch) / layer.ss
        i = j


def dither565(img):
    """PSP 16-bit framebuffer: RGB565 with a 4x4 ordered dither."""
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - 0.5
    h, w = img.shape[:2]
    t = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w, None]
    steps = np.array([8, 4, 8], np.float32)
    return np.clip(np.floor(img / steps + 0.5 + t) * steps, 0, 255)


def jpeg(arr, q, subsampling, rng):
    """JPEG round trip with a random 8x8 grid phase (the region is a crop of a screenshot)."""
    oy, ox = int(rng.integers(0, 8)), int(rng.integers(0, 8))
    h, w = arr.shape[:2]
    padded = np.pad(arr, ((oy, 0), (ox, 0), (0, 0)), mode="edge")
    buf = io.BytesIO()
    Image.fromarray(padded).save(buf, "JPEG", quality=int(q), subsampling=int(subsampling))
    return np.asarray(Image.open(buf).convert("RGB"))[oy:oy + h, ox:ox + w]


def load_bg(path, scale, box=None):
    return S2.load_bg(path, scale, box)


def background(rng, w, h, platform, target, up):
    """(region, tag): a scene crop (upper 62% of a training-side screenshot) that fits the region,
    else synthetic gradients, blobs and clutter (synth2's generator); tag "scene:KIND" or
    "synthetic"."""
    if rng.random() < CFG["background"]["scene"]:
        if platform == "scaled":
            pools = [("hires", up * (0.5 if target == "psp" else 1.0))]
        elif platform in ("psp", "psp2x"):
            pools = [("psp", 1.0), ("vita", 0.5), ("hires", 0.5)]
        else:
            pools = [("vita", 1.0), ("hires", 1.0)]
        kind, base = pools[rng.integers(len(pools))]
        paths = R.bg[kind]
        for _ in range(4 if paths else 0):
            p = paths[rng.integers(len(paths))]
            try:
                z = round(float(rng.uniform(*CFG["scene_zoom"])), 1)
                with Image.open(p) as pim:
                    sw0 = pim.width
                scale = round(960 * base * z / sw0, 3) if kind == "hires" else base * z
                im = load_bg(p, scale)
                H, W = im.shape[:2]
                safe_h = int(0.62 * H)
                if W >= w and safe_h >= h:
                    x0 = int(rng.integers(0, W - w + 1))
                    y0 = int(rng.integers(0, safe_h - h + 1))
                    k = W / sw0
                    tb = R.bg_text.get(p, np.zeros((0, 4), np.float32)) * k
                    if len(tb) and ((tb[:, 0] - 3 < x0 + w) & (tb[:, 2] + 3 > x0) &
                                    (tb[:, 1] - 3 < y0 + h) & (tb[:, 3] + 3 > y0)).any():
                        continue            # the patch would carry real, unlabelled screen text
                    arr = im[y0:y0 + h, x0:x0 + w].astype(np.float32)
                    return (arr[:, ::-1].copy() if rng.random() < 0.5 else arr), f"scene:{kind}"
            except OSError:
                pass
    saved = S2.R
    S2.R = _NoScenes
    try:
        return S2.background(rng, w, h, "vita"), "synthetic"
    finally:
        S2.R = saved


class _NoScenes:
    """Makes synth2's background() skip screenshots: synth3 picks scenes itself."""
    bg_psp, bg_vita, bg_boxes = [], [], {}


# ----------------------------------------------------------------------------- crops

def line_ink(L, layout, ink, ruby, shape):
    """(x0, y0, x1, y1) of a line's labelled ink: fill, outline and hard shadow, plus the
    furigana sitting on it; descenders below the nominal band count, the next line's ruby not."""
    H = shape[0]
    if L["ix0"] < 0:
        return None
    y0, y1 = max(0, int(math.floor(L["y0"]))), min(H, int(math.ceil(L["y1"] + 0.25 * L["fs"])))
    for o in layout:
        if o is not L and o["y0"] >= L["y0"] + 1:
            y1 = min(y1, int(math.floor(o["y0"] if o.get("ruby_top") is None else min(o["y0"], o["ruby_top"]))))
    y1 = max(y1, min(H, int(math.ceil(L["y1"]))))
    x0, x1 = L["ix0"], L["ix1"]
    band = ink[y0:y1, x0:x1] & ~ruby[y0:y1, x0:x1]
    rows = np.flatnonzero(band.any(axis=1))
    if not len(rows):
        return None
    top, bot = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
    if L.get("ruby_top") is not None:
        top = min(top, L["ruby_top"])
    return x0, top, x1, bot


def fill_ink(L, layout, fill, ruby, shape):
    """The line's box as refine() finds white or black text, vertically: the rows of its glyph fill
    dilated by 1 px (`fill`), plus attached furigana, so an outline or shadow above and below sits
    in the pad. The columns stay the line's full ink columns: a thin glyph whose fill vanishes
    (。」 at a small size) must never fall outside the crop."""
    return line_ink(L, layout, fill, ruby, shape)


def gt_box(L, layout, ink, ruby, pad, rng, shape, extra=None, fill=None):
    """The labelled ink padded like eval's boxes (4 px Vita, 2 px PSP), plus a random margin per
    side (CFG pad_extra of the target: with probability p, U(0, max) x ink height; `extra`
    overrides it). Another line's band may not enter the crop by more than a quarter."""
    H, W = shape[:2]
    ib = None
    if fill is not None and L.get("fillbox"):
        ib = fill_ink(L, layout, fill, ruby, shape)
    if ib is None:
        ib = line_ink(L, layout, ink, ruby, shape)
    if ib is None:
        return None, None
    x0, top, x1, bot = ib
    ih = bot - top
    E = extra if extra is not None else CFG["pad_extra"]["vita" if pad >= 4 else "psp"]
    more = [0, 0, 0, 0]
    if rng.random() < E["p"]:
        more = [int(rng.uniform(0, E[k]) * ih + 0.5) for k in ("left", "top", "right", "bottom")]
    bx0, by0 = max(0, x0 - pad - more[0]), max(0, top - pad - more[1])
    bx1, by1 = min(W, x1 + pad + more[2]), min(H, bot + pad + more[3])
    for o in layout:
        if o is L or o["ix0"] < 0:
            continue
        ov = min(by1, o["y1"]) - max(by0, o["y0"])
        if ov > 0.25 * (o["y1"] - o["y0"]) and o["ix1"] > bx0 and o["ix0"] < bx1:
            if o["y0"] >= top:
                by1 = min(by1, int(math.floor(o["y0"] + 0.25 * (o["y1"] - o["y0"]))))
            else:
                by0 = max(by0, int(math.ceil(o["y1"] - 0.25 * (o["y1"] - o["y0"]))))
            if by0 > top or by1 < bot:
                return None, None
    if bx1 - bx0 < 8 or by1 - by0 < 8:
        return None, None
    return (bx0, by0, bx1 - bx0, by1 - by0), ib


def grad_crops(img, layout):
    """The console's GRAD finder boxes that each hold exactly one labelled line: the box's band
    (inside the finder's 3 px pad) overlaps that line's band by at least half and no other
    labelled line by more than a row, line and box pick each other, the box hugs the line's
    ink on both sides within 3 px (unless that ink touches the region edge), and it is not
    taller than 1.6x the line's band plus pad (plus attached furigana). Anything else, such as
    a band merged with a neighbour or with screen text, yields no crop."""
    from common import PAD, find_lines_ex
    r = find_lines_ex(img, "grad")
    if r is None:
        return []
    dets, flags = r[0], r[1]
    lines = [L for L in layout if L.get("text") and L["ix0"] >= 0]
    if not dets or not lines:
        return []
    H, W = img.shape[:2]
    rows = [(y + (PAD if y > 0 else 0), y + h - (PAD if y + h < H else 0)) for x, y, w, h in dets]
    ov = np.array([[min(b, L["y1"]) - max(a, L["y0"]) for L in lines] for a, b in rows], np.float32)
    grey = to_gray(img)
    out = []
    for d, (x, y, w, h) in enumerate(dets):
        li = int(ov[d].argmax())
        L = lines[li]
        band = L["y1"] - L["y0"]
        if ov[d, li] <= 0 or int(ov[:, li].argmax()) != d or (np.delete(ov[d], li) > 1).any():
            continue
        if ov[d, li] < 0.5 * min(rows[d][1] - rows[d][0], band):
            continue
        extra = (L["y0"] - L["ruby_top"]) if L.get("ruby_top") is not None else 0
        if h > 1.6 * (band + 2 * PAD) + extra:
            continue
        if L["ix0"] > PAD + 1 and abs(x + PAD - L["ix0"]) > 3:
            continue
        if L["ix1"] < W - PAD - 1 and abs(x + w - PAD - L["ix1"]) > 3:
            continue
        dark = int(flags[d])
        g = grey[y:y + h, x:x + w]
        out.append((255 - g if dark else g.copy(), L["text"], (x, y, w, h), L, dark))
    return out


def extract(img, layout, ink, rng, mode, pad, ruby=None, fill=None):
    """[(grey crop in console polarity, label, box, line, dark, crop kind)]. In GRAD mode (v3),
    a labelled line without an accepted GRAD box gets its gt crop instead (kind "gt"), so a line
    next to screen clutter or a close neighbour is not dropped (CFG grad_fallback)."""
    ruby = ruby if ruby is not None else np.zeros_like(ink)
    out = []
    if mode == "grad":
        out = [g + ("grad",) for g in grad_crops(img, layout)]
        if not CFG["grad_fallback"]:
            return out
        done = {id(g[3]) for g in out}
        todo = [L for L in layout if L.get("text") and id(L) not in done]
    else:
        todo = [L for L in layout if L.get("text")]
    if not todo:
        return out
    grey = to_gray(img)
    for L in todo:
        box, _ = gt_box(L, layout, ink, ruby, pad, rng, img.shape, fill=fill)
        if box is None:
            continue
        x, y, w, h = box
        dark = line_dark(grey, box)
        g = grey[y:y + h, x:x + w]
        out.append((255 - g if dark else g.copy(), L["text"], box, L, dark, "gt"))
    return out


def crop_ink_box(L, layout, ink, ruby, box):
    """The line's labelled ink box in crop coordinates (for the margin statistics)."""
    ib = line_ink(L, layout, ink, ruby, ink.shape)
    if ib is None:
        return None
    x, y, w, h = box
    return [max(0, ib[0] - x), max(0, ib[1] - y), min(w, ib[2] - x), min(h, ib[3] - y)]


# ----------------------------------------------------------------------------- shards

def length_bin(n):
    e = CFG["length_edges"]
    return int(np.searchsorted(e, n, side="right") - 1)


class LengthGate:
    """Rejection sampling of crops toward CFG['length_target'] (label-length bins): a crop in bin
    b is kept with probability (t_b / q_b) / max_k (t_k / q_k), where q is the running share of
    proposed crops per bin (starting at the target), floored at CFG['length_floor'] so no
    length disappears and the render cost stays bounded."""

    def __init__(self):
        self.t = np.asarray(CFG["length_target"], np.float64)
        self.q = self.t * 200.0 + 1e-3

    def accept(self, rng, n):
        b = min(length_bin(n), len(self.t) - 1)
        self.q[b] += 1
        share = self.q / self.q.sum()
        ratio = np.where(self.t > 0.005, self.t / share, 0.0)
        p = float(np.clip(ratio[b] / max(ratio.max(), 1e-9), CFG["length_floor"], 1.0))
        return rng.random() < p


def run_shard(job):
    idx, n_lines, seed, out_dir, prefix = job
    cv2.setNumThreads(1)
    path = Path(out_dir) / f"{prefix}_{idx:05d}.npz"
    if path.exists():
        return idx, 0, 0, 0, 0.0
    rng = np.random.default_rng([seed, idx])
    R.counts[:] = 0
    R._focus_w = None
    gate = LengthGate()
    t0, crops, labels, metas, regions, errors = time.time(), [], [], [], 0, 0
    while len(crops) < n_lines:
        try:
            r = render_region(rng)
        except Exception as e:  # noqa: BLE001 - a rare bad draw must not end a long render
            errors += 1
            if errors <= 3:
                print(f"shard {idx}: render error {type(e).__name__}: {e}", flush=True)
            if errors > 1000:
                raise
            continue
        if r is None:
            continue
        img, layout, ink, meta = r
        regions += 1
        crop_mode = choose(rng, CFG["crop"])
        pad = CFG["pad"]["psp" if meta["target"] == "psp" else "vita"]
        ruby_mask = meta.pop("_ruby_mask")
        fill_mask = meta.pop("_fill_mask")
        for g, lab, box, L, dark, kind in extract(img, layout, ink, rng, crop_mode, pad, ruby_mask, fill_mask):
            if not lab or any(c not in R.vocab_set for c in lab) or not S2.crop_ctc_feasible(g.shape, lab):
                continue
            if set(lab) <= set("「」『』（）()、。・"):      # a stray bracket or comma is not a line
                continue
            if not gate.accept(rng, len(lab)):
                continue
            m = dict(meta)
            m.update(kind=L["kind"], crop=kind, crop_mode=crop_mode, dark=bool(dark), inverted=bool(dark),
                     ruby=bool(L.get("ruby_top") is not None), coloured=bool(L.get("coloured")),
                     fade=int(L.get("fade", 0)), box=[int(v) for v in box],
                     ink_box=crop_ink_box(L, layout, ink, ruby_mask, box))
            crops.append(g)
            labels.append(lab)
            metas.append(m)
            for c in lab:
                R.counts[R.cid[c]] += 1
            if len(crops) >= n_lines:
                break
    shapes = np.array([c.shape for c in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=np.concatenate([c.ravel() for c in crops]), offsets=offsets, shapes=shapes,
                        labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(metas, ensure_ascii=False)))
    tmp.rename(path)
    return idx, len(crops), regions, errors, time.time() - t0


def recipe(args):
    code = {p: sha256(HERE / p) for p in ("synth3.py", "render.py", "common.py", "synth3_text.py")}
    return {"kind": "synth3-recipe", "schema": 1, "split": args.split, "seed": args.seed, "profile": args.profile,
            "per_shard": args.per_shard, "cfg": CFG, "code_sha256": code,
            "font_manifest_sha256": sha256(FONT_MANIFEST), "vocab_sha256": sha256(DATA / "vocab.txt"),
            "fonts": len(R.fonts), "font_families": len({f.fam for f in R.fonts}),
            "text": R.text_identity, "background_vns": len(R.bg_vns),
            "background_shots": {k: len(v) for k, v in R.bg.items()},
            "exclusions": R.exclusions["files"], "scenes_dropped": R.scene_dropped,
            "scene_hashes_sha256": sha256(args.scene_hashes) if getattr(args, "scene_hashes", None) else None,
            "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": Image.__version__,
                         "freetype": __import__("PIL.features", fromlist=["version"]).version("freetype2")}}


# ----------------------------------------------------------------------------- preview / manifest

def preview(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label_font = ImageFont.truetype(str(sorted((SHARED_DATA / "fonts" / "notosansjp").glob("*.ttf"))[0]), 15)
    rows, regions = [], []
    while len(rows) < args.n * 3:
        r = render_region(rng)
        if r is None:
            continue
        img, layout, ink, meta = r
        mode = choose(rng, CFG["crop"])
        got = extract(img, layout, ink, rng, mode, CFG["pad"]["psp" if meta["target"] == "psp" else "vita"],
                      meta.pop("_ruby_mask"), meta.pop("_fill_mask"))
        if not got:
            continue
        vis = Image.fromarray(img)
        d = ImageDraw.Draw(vis)
        for _, _, (x, y, w, h), _, _, _ in got:
            d.rectangle((x, y, x + w - 1, y + h - 1), outline=(0, 255, 0))
        if len(regions) < args.n:
            regions.append(vis)
        for g, lab, _, L, dark, kind in got:
            g32 = Image.fromarray(g).resize((max(8, round(g.shape[1] * 32 / g.shape[0])), 32), Image.BILINEAR)
            row = Image.new("RGB", (1280, 34), (0, 0, 0))
            row.paste(g32.crop((0, 0, min(g32.width, 700), 32)).convert("RGB"), (0, 1))
            tag = (f"{lab[:24]} [{meta['platform']} {meta['fam']} {meta['raster']} {kind} "
                   f"{'inv ' if dark else ''}{meta['capture']}]")
            ImageDraw.Draw(row).text((710, 7), tag, font=label_font, fill=(255, 255, 160))
            rows.append(row)
    sheets, cur, y = [], [], 0
    for im in regions:
        im = im if im.width <= 1280 else im.resize((1280, im.height * 1280 // im.width))
        if y + im.height > 1280 and cur:
            sheets.append(cur)
            cur, y = [], 0
        cur.append(im)
        y += im.height + 4
    sheets.append(cur)
    for s, ims in enumerate(sheets[:6]):
        sheet = Image.new("RGB", (1280, min(1280, sum(im.height + 4 for im in ims))), (40, 40, 40))
        yy = 0
        for im in ims:
            sheet.paste(im, (0, yy))
            yy += im.height + 4
        sheet.save(out / f"regions_{s:02d}.png")
    for s in range(0, len(rows), 37):
        part = rows[s:s + 37]
        sheet = Image.new("RGB", (1280, 34 * len(part)), (0, 0, 0))
        for i, row in enumerate(part):
            sheet.paste(row, (0, 34 * i))
        sheet.save(out / f"crops_{s // 37:02d}.png")
    print(f"preview: {len(regions)} regions, {len(rows)} crops -> {out}", flush=True)


def manifest(args):
    shards = []
    recipes = {}
    for d in args.dir:
        d = Path(d)
        rec = json.loads((d / "recipe.json").read_text(encoding="utf-8"))
        recipes[d.name] = rec
        for p in sorted(d.glob("*.npz")):
            z = np.load(p)
            shards.append({"set": d.name, "file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
                           "samples": int(len(z["shapes"]))})
    doc = {"kind": "synth3-shards", "schema": 1, "sets": recipes, "shards": shards,
           "samples": sum(s["samples"] for s in shards), "bytes": sum(s["bytes"] for s in shards)}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(shards)} shards, {doc['samples']} samples, {doc['bytes'] / 1e9:.2f} GB -> {args.out}")


def apply_cfg(spec):
    """--cfg: a JSON object, or @path to one, merged into CFG in place (calibration renders)."""
    patch = json.loads(Path(spec[1:]).read_text(encoding="utf-8") if spec.startswith("@") else spec)
    merged = merge_cfg(CFG, patch)
    CFG.clear()
    CFG.update(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preview", "shards", "manifest", "scene-hashes"])
    ap.add_argument("--assets", type=Path, default=ASSETS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dir", action="append")
    ap.add_argument("--split", choices=["train", "dev", "dev-fonts"], default="train")
    ap.add_argument("--profile", choices=sorted(LICENCE_PROFILES), default="personal")
    ap.add_argument("--holdout-fonts", nargs="*", default=[])
    ap.add_argument("--shards", type=int, default=100)
    ap.add_argument("--first", type=int, default=0, help="first shard index (resume / split a render)")
    ap.add_argument("--per-shard", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cfg", help="JSON (or @file) merged into CFG, for calibration renders; recorded in the recipe")
    ap.add_argument("--exclude", nargs="*", type=Path, default=list(EXCLUDE_MANIFESTS),
                    help="exclusion manifests (VNs, screenshot and name hashes never used)")
    ap.add_argument("--scene-hashes", type=Path, help="scene-hashes output: pixel sha256 per scene file sha256")
    ap.add_argument("--font-cache", type=Path, help="font coverage cache (default ASSETS/fonts/coverage_cache.json)")
    ap.add_argument("--targets", type=Path, help="focus-character targets (default ASSETS/text/pools3/targets.json)")
    args = ap.parse_args()
    if args.mode == "manifest":
        manifest(args)
        return
    if args.mode == "scene-hashes":
        scene_hashes(args)
        return
    if args.cfg:
        apply_cfg(args.cfg)
    cv2.setNumThreads(1)
    init(assets=args.assets, split=args.split, profile=args.profile, holdout_fonts=set(args.holdout_fonts),
         exclude=args.exclude, font_cache=args.font_cache, scene_hashes=args.scene_hashes, targets=args.targets)
    print(f"fonts {len(R.fonts)} files / {len({f.fam for f in R.fonts})} families "
          f"{ {c: len(v) for c, v in R.by_cat.items()} }; pools {list(R.pools)}; "
          f"scenes { {k: len(v) for k, v in R.bg.items()} } from {len(R.bg_vns)} VNs; "
          f"excluded: {len(R.exclusions['vns'])} VNs, scenes dropped {R.scene_dropped}, "
          f"names dropped {R.names_dropped}", flush=True)
    if args.mode == "preview":
        preview(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    rec = json.loads(json.dumps(recipe(args), ensure_ascii=False))     # canonical (JSON keys are strings)
    rp = args.out / "recipe.json"
    if rp.exists() and json.loads(rp.read_text(encoding="utf-8")) != rec:
        raise SystemExit(f"{rp} holds a different recipe: use a new --out")
    rp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    prefix = "synth3" if args.split == "train" else f"synth3_{args.split}"
    jobs = [(i, args.per_shard, args.seed, str(args.out), prefix) for i in range(args.first, args.first + args.shards)]
    t0, total = time.time(), 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, regions, errors, dt in pool.imap_unordered(run_shard, jobs):
            total += n
            print(f"shard {idx}: {n} lines from {regions} regions ({errors} errors) in {dt:.0f}s "
                  f"(total {total}, {time.time() - t0:.0f}s)", flush=True)
    print("SHARDS_DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""desk_ui: synthetic dense-UI desktop frames for the desktop detector (textspot-desktop round 3, #173).

  ui_synth.py render   --out DIR --shards N [--per-shard 250] [--first F] [--seed S] [--workers W] [--version V]
  ui_synth.py manifest --dir DIR --out data/manifests/desktop_data/tsdesk_ui_train_v1.json [--version V]

--version 1 (the default) draws desk_ui v1 (tsdesk_ui_train_v1.json: its code path, mix and random draws, so its
shards re-render bit for bit); --version 2 draws desk_ui v2 (tsdesk_ui_train_v2.json; the "desk_ui v2" section
below): the v1 layouts at about 35% of the layout draws and six more aimed at characters read into a label from what
sits next to it (icons and badges touching a label or 0.1-0.5 line heights from it, leader dots and rules between a
label and its value, ornament marks, arrows and graphic brackets beside a label, label + value pairs at 0.5-1.75
line heights in one box, stat items written as one run in rows, UI rows inside wide plates), every mark background.

What dense_screen still misses (protocol v2 amendment 11): short UI labels, large or stylised glyphs where K stays
below 0, an icon right before a label, label + value pairs, stat panels, menus, button rows, list lines that start
with ・. Each frame is one of 1c's (desktopdata/synth.py, whose code this module only calls: its scenes, fonts,
coverage boxes, capture model and MeikiOCR guard) with 1-3 UI layouts on it and, in 30%, 1c's dialogue element:
  buttons   a row (or a column) of 3-10 buttons: pill, rounded, square, underlined or bare; an icon before the label
  menu      3-9 items stacked in large display glyphs; a cursor or icon before the selected one, a highlight bar
  stats     2-8 rows of label + value; gauges and star / heart ratings (background); values in a right column
  header    a large title (48-160 px) and a row of tabs
  config    label + slider or option buttons + value
  list      3-8 lines starting with ・ (or numbered, or after a bullet icon drawn as background)
  slots     a grid of save slots: No.NN, a date and time, a short title
  pairs     top-bar items (DAY 12, 所持金 3,450円, Lv.15), some after an icon
Fonts: synth3's non-pixel families, licence class open only (as 1c), weighted to gothic, display and rounded faces,
bold weights favoured. UI em 12-160 px at the frame's own size (MIX_UI["em_band"]), at most 12% of its height.
Boxes follow 1b's conventions (textspot-desktop-targets' convention_report.md and convention_gaps_*.json): one box
per text item, the ink box of its own coverage (1c's rule, textspot_data.instance_theta); ・ at a list line's start
and brackets are inside the box (text); icons, bullets, cursors, gauges, sliders, ratings, plates and separators are
background, and the marks among them are listed in the frame (fr.bg): the guard leaves alone a box lying mostly on
them, so they never become ignore areas; the ink gaps inside an item are at most GAP_IN x its height and two items
side by side on a row are at least GAP_BETWEEN x the taller one's height apart (nothing in between); class "ui" (1c's
dialogue element: "dialogue", "name"). Scenes: 1c's pool, cut by VN, screenshot id and file sha256 against every
protected set (train/tsdesk_dense_source.exclusion_sets). Shards: DIR/desk_ui_NNNN.{npy,json}, src
"synth:desktop-ui", kind "pc", lines [x0, y0, x1, y1, vertical, class]; the trainer reads them as an extra source
(--extra desk_ui=MANIFEST:ROOT).
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from desktopdata import capture as CAP  # noqa: E402
from desktopdata import shards as SH  # noqa: E402
from desktopdata import synth as D1  # noqa: E402

GEN_VERSION = 1
GAP_IN = 0.9            # x the item's ink height: the widest ink gap inside one item (1b: <= 1.0 h in 95% of boxes)
GAP_BETWEEN = 1.8       # x the taller item's height: two items on a row (1b: >= 1.63 h in 99% of side-by-side pairs)
ROW_GAP = 0.3           # x the taller item's height: the least gap between two stacked rows' ink
MIX_UI = {
    "screen": {k: v for k, v in D1.MIX["screen"].items() if k != "3840x2160"},
    "game": dict(D1.MIX["game"]),
    "captured": D1.MIX["captured"],
    "native_jpeg": dict(D1.MIX["native_jpeg"]),
    "n_layouts": [0.35, 0.4, 0.25],
    "layout": {"buttons": 0.18, "menu": 0.16, "stats": 0.14, "header": 0.1, "config": 0.09, "list": 0.13,
               "slots": 0.06, "pairs": 0.14},
    "dialogue": 0.3,
    "dialogue_layout": {"adv": 0.7, "adv_top": 0.05, "band": 0.1, "bare": 0.15},
    "em_band": {"12-24": 0.3, "24-48": 0.3, "48-96": 0.28, "96-160": 0.12},
    "em_max_frac": 0.12,
    "category": {"gothic": 0.45, "display": 0.25, "rounded": 0.15, "mincho": 0.1, "pen": 0.05},
    "bold_p": 0.55,
    "thin_p": 0.1,
    "latin_p": 0.4,
    "style": {"outline": 0.35, "outline_em": [0.03, 0.1], "double": 0.05, "glow": 0.15, "shadow": 0.3,
              "gradient": 0.2, "accent": 0.3, "spacing": 0.3, "spacing_em": [0.03, 0.3], "shear": 0.1,
              "shear_k": [0.12, 0.25]},
    "icon_before": 0.35,
    "icon_gap_em": [0.15, 0.6],
    "plate": {"pill": 0.25, "round": 0.25, "square": 0.2, "underline": 0.1, "bare": 0.2},
    "list_kind": {"nakaguro": 0.5, "numbered": 0.15, "icon": 0.25, "plain": 0.1},
    "scenes": dict(D1.MIX["scenes"]),
}
GUARD_BG_SHARE = 0.5    # a guard box at least this much on drawn marks (fr.bg) is left alone: marks are background

MENU_JP = ["はじめから", "つづきから", "ロード", "セーブ", "ギャラリー", "CGモード", "回想", "音楽鑑賞", "おまけ",
           "エクストラ", "コンフィグ", "設定", "終了", "ゲーム終了", "タイトルへ", "タイトルに戻る", "システム",
           "クイックセーブ", "クイックロード", "バックログ", "スキップ", "オート", "ウィンドウを消す", "既読スキップ",
           "環境設定", "ヘルプ", "マニュアル", "フローチャート", "用語集", "キャラクター", "プロフィール", "アルバム",
           "シーン回想", "オプション", "続ける", "最初から", "前に戻る", "次へ", "決定", "キャンセル", "はい", "いいえ",
           "閉じる", "戻る", "全画面", "ウィンドウ", "Q.セーブ", "Q.ロード", "ログ", "メニュー", "タイトル", "ボイス",
           "データ", "アイテム", "ステータス", "マップ", "移動", "調べる", "話す", "休む", "持ち物", "スケジュール"]
MENU_EN = ["NEW GAME", "START", "CONTINUE", "LOAD", "SAVE", "Q.SAVE", "Q.LOAD", "GALLERY", "EXTRA", "EXTRAS",
           "CONFIG", "OPTION", "OPTIONS", "SYSTEM", "EXIT", "QUIT", "TITLE", "BACK", "LOG", "BACKLOG", "AUTO", "SKIP",
           "HIDE", "MENU", "HELP", "MUSIC", "MEMORY", "SCENE", "ALBUM", "SOUND", "VOICE", "TEXT", "SCREEN", "WINDOW",
           "DEFAULT", "OK", "CANCEL", "YES", "NO", "CLOSE", "NEXT", "PREV", "STATUS", "ITEM", "MAP", "SKILL", "DATA",
           "Start", "Load", "Save", "Config", "Exit", "Gallery", "Extra", "Back", "Log", "Auto", "Skip", "Title"]
HEADERS = ["CONFIG", "SAVE", "LOAD", "GALLERY", "BACKLOG", "STATUS", "ITEM", "TIPS", "MAP", "EXTRA", "OPTION",
           "SYSTEM", "MUSIC ROOM", "SCENE REPLAY", "CHARACTER", "PROFILE", "システム", "セーブ", "ロード", "コンフィグ",
           "ステータス", "アイテム", "用語集", "人物紹介", "フローチャート", "環境設定", "音楽鑑賞", "CGギャラリー",
           "シーン回想", "スケジュール", "マップ", "おまけ", "データ"]
TABS = ["システム", "サウンド", "テキスト", "ボイス", "画面", "その他", "基本設定", "詳細設定", "GENERAL", "SOUND",
        "TEXT", "VOICE", "SCREEN", "OTHER", "KEY", "MOUSE", "PAGE 1", "PAGE 2", "1", "2", "3", "4", "5", "ALL",
        "人物", "用語", "世界観", "イベント", "Chapter 1", "Chapter 2", "第一章", "第二章", "共通", "個別"]
CONFIG_LABELS = ["BGM音量", "効果音", "SE音量", "ボイス音量", "マスター音量", "システム音量", "テキスト速度",
                 "文字速度", "オート速度", "ウィンドウ透明度", "既読スキップ", "未読スキップ", "画面モード",
                 "マウスホイール", "右クリック", "フォント", "キャラ別ボイス", "BGM", "SE", "VOICE", "TEXT SPEED",
                 "AUTO SPEED", "WINDOW ALPHA", "MASTER", "MOVIE", "ボイス再生", "選択肢後", "スキップ設定"]
OPTION_VALUES = ["ON", "OFF", "遅い", "普通", "速い", "あり", "なし", "する", "しない", "全画面", "ウィンドウ", "標準",
                 "高速", "瞬間", "有効", "無効", "SLOW", "NORMAL", "FAST", "継続", "解除", "全て", "既読のみ"]
STAT_LABELS = ["体力", "気力", "知力", "魅力", "筋力", "精神", "攻撃力", "防御力", "素早さ", "運", "器用さ", "魔力",
               "好感度", "信頼度", "所持金", "経験値", "レベル", "名声", "ストレス", "疲労", "学力", "芸術", "運動",
               "容姿", "話術", "根性", "日数", "残り日数", "行動力", "HP", "MP", "SP", "EXP", "ATK", "DEF", "SPD",
               "LUK", "INT", "STR", "AGI", "Lv", "GOLD", "RANK", "SCORE", "TIME", "所持品", "ランク", "クラス"]
ITEMS = ["薬草", "鍵", "手紙", "写真", "学生証", "携帯電話", "地図", "懐中電灯", "ハンカチ", "お守り", "古い日記",
         "ノート", "財布", "定期券", "指輪", "ペンダント", "傘", "腕時計", "弁当", "ジュース", "名刺", "招待状",
         "チケット", "メモ", "カード", "薬", "包帯", "ロープ", "マッチ", "鏡", "本", "楽譜", "花束", "クッキー"]
DOW = "月火水木金土日"


def choose(rng, w):
    return CAP.choose(rng, w)


FONT_LOG = []            # version 2: the font files this frame draws with (meta "fonts"); never read by a draw


def S3():
    return D1.S3


def TS():
    return D1.TS


# ----------------------------------------------------------------------------- text values

def value_text(rng):
    """A procedural UI value: a number, a fraction, a percentage, money, a date or time, a level, a rank."""
    n = int(rng.integers(0, 1000))
    m = int(rng.integers(max(1, n), 1000)) if n else int(rng.integers(1, 1000))
    k = int(rng.integers(14))
    if k == 0:
        return f"{n}"
    if k == 1:
        return f"{n}/{m}"
    if k == 2:
        return f"{int(rng.integers(0, 101))}%"
    if k == 3:
        return f"{int(rng.integers(0, 99999)):,}G"
    if k == 4:
        return f"{int(rng.integers(0, 99999)):,}円"
    if k == 5:
        return f"Lv.{int(rng.integers(1, 100))}"
    if k == 6:
        return f"{int(rng.integers(1, 13))}月{int(rng.integers(1, 29))}日"
    if k == 7:
        return f"{int(rng.integers(1, 13))}/{int(rng.integers(1, 29))}（{DOW[int(rng.integers(7))]}）"
    if k == 8:
        return f"{int(rng.integers(0, 24))}:{int(rng.integers(0, 60)):02d}"
    if k == 9:
        return ["S", "A", "B", "C", "D", "E", "SS", "MAX", "---", "???"][int(rng.integers(10))]
    if k == 10:
        return f"{'+-'[int(rng.integers(2))]}{int(rng.integers(1, 100))}"
    if k == 11:
        return f"{int(rng.integers(1, 365))}日目"
    if k == 12:
        return f"第{int(rng.integers(1, 13))}章"
    return f"{int(rng.integers(1, 99))}回"


def pair_text(rng):
    """(label, value) of a top-bar pair, or (text, None) for a single token."""
    k = int(rng.integers(8))
    if k == 0:
        return f"DAY {int(rng.integers(1, 100))}", None
    if k == 1:
        return f"{int(rng.integers(1, 13))}月{int(rng.integers(1, 29))}日（{DOW[int(rng.integers(7))]}）", None
    if k == 2:
        return f"Lv.{int(rng.integers(1, 100))}", None
    if k == 3:
        return f"{int(rng.integers(0, 24))}:{int(rng.integers(0, 60)):02d}", None
    if k == 4:
        return f"第{int(rng.integers(1, 30))}話", None
    if k == 5:
        return f"残り{int(rng.integers(1, 30))}日", None
    return STAT_LABELS[int(rng.integers(len(STAT_LABELS)))], value_text(rng)


def short_text(rng, lo=3, hi=16):
    """A short phrase from synth3's text pools (its choice texts), cut to lo-hi characters."""
    for _ in range(4):
        s, _, _ = S3().choice_text(rng)
        s = (s or "").strip().replace("　", "")
        if len(s) >= lo:
            return s[:int(rng.integers(lo, hi + 1))]
    return ITEMS[int(rng.integers(len(ITEMS)))]


def name_text(rng):
    disp, label, _ = S3().sample_name(rng)
    return (disp or "").replace("　", "") or None


# ----------------------------------------------------------------------------- fonts and items

def pick_ui_font(rng, chars):
    """(FontFile, wght or None): a non-pixel synth3 face of licence class open covering chars, by MIX_UI's
    category weights, bold weights favoured (MIX_UI["bold_p"]), thin ones sometimes (MIX_UI["thin_p"])."""
    R = S3().R
    cats = {k: v for k, v in MIX_UI["category"].items() if k in R.by_cat}
    chars = set(chars) - {" ", "　"}
    for _ in range(8):
        c = choose(rng, cats)
        fams = [f for f in R.by_cat[c].values() if any(chars <= x.chars and x.licence == "open" for x in f)]
        if not fams:
            continue
        fam = fams[int(rng.integers(len(fams)))]
        files = [x for x in fam if chars <= x.chars and x.licence == "open"]
        thin = [x for x in files if x.weight and x.weight <= 300]
        if thin and rng.random() < MIX_UI["thin_p"]:
            return _logged(thin[int(rng.integers(len(thin)))]), None
        bold = [x for x in files if x.weight and x.weight >= 600]
        if bold and rng.random() < MIX_UI["bold_p"]:
            return _logged(bold[int(rng.integers(len(bold)))]), None
        normal = [x for x in files if not x.weight or x.weight > 300] or files
        f = normal[int(rng.integers(len(normal)))]
        wght = S3().S2.pick_weight(rng, f.path)
        if wght is not None and rng.random() < MIX_UI["bold_p"]:
            wght = max(wght, 700)
        return _logged(f), wght
    return None, None


def _logged(f):
    if VERSION >= 2 and f is not None:
        FONT_LOG.append(f)
    return f


def ui_em(rng, fr, band=None):
    lo, hi = (int(v) for v in (band or choose(rng, MIX_UI["em_band"])).split("-"))
    cap = max(12, int(MIX_UI["em_max_frac"] * fr.H))
    return int(np.clip(rng.integers(lo, hi + 1), 12, cap))


def ui_style(rng, em, on_luma=None):
    """Fill, outline, glow, shadow and gradient of UI text drawn on a plate of luma on_luma (None: on the scene)."""
    S2 = S3().S2
    P = MIX_UI["style"]
    dark_text = on_luma is not None and on_luma > 150
    fill = S2.text_dark(rng) if dark_text else S2.text_light(rng)
    if rng.random() < P["accent"]:
        fill = S2.accent(rng, not dark_text)
    st = {"fill": fill}
    if rng.random() < P["outline"] or (on_luma is None and rng.random() < 0.6):
        st["stroke"] = int(np.clip(round(em * rng.uniform(*P["outline_em"])), 1, 6))
        st["stroke_c"] = S2.light_colour(rng, 0.15, 0.9) if dark_text else S2.dark_colour(rng, 0.25)
        if rng.random() < P["double"]:
            st["outer"] = st["stroke"] + max(1, int(round(em * 0.04)))
            st["outer_c"] = S2.dark_colour(rng, 0.2) if dark_text else S2.light_colour(rng, 0.3, 0.8)
    if rng.random() < P["glow"]:
        st["glow"] = {"r": max(2, int(round(em * rng.uniform(0.08, 0.2)))), "c": S2.accent(rng, True),
                      "a": float(rng.uniform(0.4, 0.9))}
    if rng.random() < P["shadow"] or (on_luma is None and "stroke" not in st and "glow" not in st):
        d = max(1, int(round(em * rng.uniform(0.04, 0.09))))
        hard = rng.random() < 0.6
        st["shadow"] = {"dx": d, "dy": d, "blur": 0.0 if hard else float(rng.uniform(0.8, 2.0)),
                        "c": S2.light_colour(rng, 0.1, 0.85) if dark_text else S2.dark_colour(rng, 0.15),
                        "a": float(rng.uniform(0.55, 1.0))}
    if rng.random() < P["gradient"]:
        other = S2.accent(rng, not dark_text) if rng.random() < 0.5 else \
            np.clip(np.asarray(fill, np.float32) * rng.uniform(0.55, 0.85), 0, 255)
        st["grad"] = (np.asarray(fill, np.float32), np.asarray(other, np.float32))
    return st


class Item:
    """One text item before it is placed: its masks (fill, outline, outer outline, shadow, glow), its coverage (1c's
    rule: fill, outlines and a hard shadow), its ink box in the masks (coverage >= textspot_data.instance_theta), the
    pen origin, and the widest ink-free run of columns inside the ink box over its height (gap)."""

    def __init__(self, text, font, style, spacing=0.0, shear=0.0):
        import cv2
        from PIL import Image, ImageDraw
        self.text, self.style, self.ok = text, style, False
        stroke, outer = int(style.get("stroke", 0)), int(style.get("outer", 0))
        sh, gl = style.get("shadow"), style.get("glow")
        asc, desc = font.getmetrics()
        advs = [font.getlength(ch) for ch in text]
        adv = sum(advs) + spacing * max(0, len(text) - 1)
        sw = max(stroke, outer)
        pad = sw + 4 + ((max(abs(sh["dx"]), abs(sh["dy"])) + int(math.ceil(2 * sh["blur"]))) if sh else 0) + \
            (2 * gl["r"] if gl else 0)
        w, h = int(math.ceil(adv)) + 2 * pad + 4, asc + desc + 2 * pad + 4
        org = (pad, pad + asc)

        def draw(stroke_w):
            im = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(im)
            kw = {"stroke_width": stroke_w, "stroke_fill": 255} if stroke_w else {}
            if not spacing:
                d.text(org, text, font=font, fill=255, anchor="ls", **kw)
            else:
                x = float(org[0])
                for ch, a in zip(text, advs):
                    d.text((x, org[1]), ch, font=font, fill=255, anchor="ls", **kw)
                    x += a + spacing
            return np.asarray(im, np.float32) / 255.0
        fill = draw(0)
        out = draw(stroke) if stroke else None
        outer_m = draw(outer) if outer else None
        base = outer_m if outer_m is not None else (out if out is not None else fill)
        shadow = None
        if sh:
            m = np.zeros_like(base)
            dx, dy = sh["dx"], sh["dy"]
            m[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
                base[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            if sh["blur"] > 0:
                m = cv2.GaussianBlur(m, (0, 0), sh["blur"])
            shadow = m
        glow = None
        if gl:
            k = np.ones((2 * gl["r"] + 1, 2 * gl["r"] + 1), np.uint8)
            glow = cv2.GaussianBlur(cv2.dilate(base, k), (0, 0), max(1.0, gl["r"] / 2))
        masks = {"fill": fill, "out": out, "outer": outer_m, "shadow": shadow, "glow": glow}
        if shear:
            extra = int(math.ceil(abs(shear) * h))
            M = np.float32([[1, -shear, shear * h if shear > 0 else 0], [0, 1, 0]])
            for k_, m in masks.items():
                if m is not None:
                    masks[k_] = cv2.warpAffine(m, M, (w + extra, h), flags=cv2.INTER_LINEAR)
            org = (org[0] + shear * (h - org[1]) if shear > 0 else org[0] + shear * (0 - org[1]), org[1])
            w += extra
        self.masks, self.w_mask, self.h_mask = masks, w, h
        cov = masks["fill"]
        for k_ in ("out", "outer"):
            if masks[k_] is not None:
                cov = np.maximum(cov, masks[k_])
        if sh and sh["blur"] == 0 and masks["shadow"] is not None:
            cov = np.maximum(cov, masks["shadow"] * sh["a"])
        self.cov = cov
        theta = D1.TS.instance_theta(cov)
        if theta is None:
            return
        ys, xs = np.nonzero(cov >= theta)
        if not len(ys):
            return
        self.ix0, self.iy0, self.ix1, self.iy1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        self.w, self.h = self.ix1 - self.ix0, self.iy1 - self.iy0
        self.ox, self.oy = float(org[0]), float(org[1])
        cols = (cov >= theta)[:, self.ix0:self.ix1].any(axis=0)
        best = run = 0
        for v in cols:
            run = 0 if v else run + 1
            best = max(best, run)
        self.gap = best / max(1, self.h)
        self.up = self.oy - self.iy0          # ink above the baseline
        self.down = self.iy1 - self.oy        # ink below it
        self.ok = self.w >= 2 and self.h >= 2


def make_item(text, font, style, spacing=0.0, shear=0.0):
    """An Item whose inside gaps meet GAP_IN: as asked, else without its letter spacing, else without its spaces;
    None when none does."""
    for sp, t in ((spacing, text), (0.0, text), (0.0, text.replace(" ", "").replace("　", ""))):
        it = Item(t, font, style, sp, shear)
        if it.ok and it.gap <= GAP_IN:
            return it
    return None


def paint(img, X, Y, m, colour, alpha=1.0, rows=None):
    """img (float32 H x W x 3) <- colour over it with coverage m at (X, Y), clipped to the frame; rows: a per-row
    colour ramp (h x 3) instead of one colour."""
    H, W = img.shape[:2]
    h, w = m.shape
    x0, y0, x1, y1 = max(0, X), max(0, Y), min(W, X + w), min(H, Y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = (m[y0 - Y:y1 - Y, x0 - X:x1 - X] * alpha)[..., None]
    col = np.asarray(colour, np.float32) if rows is None else rows[y0 - Y:y1 - Y][:, None, :]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + col * a


def put(fr, it, x_ink, base, role="ui", kind=None):
    """Draws an Item with its ink box's left edge at x_ink and its pen baseline at base (frame px) and adds it as an
    instance of role (its text and the layout's kind kept in fr.item_info). Returns its ink box [x0, y0, x1, y1], or
    None when its masks do not fit the frame."""
    X, Y = int(round(x_ink)) - it.ix0, int(round(base - it.oy))
    if X < 0 or Y < 0 or X + it.w_mask > fr.W or Y + it.h_mask > fr.H:
        return None
    st, m = it.style, it.masks
    if m["glow"] is not None:
        paint(fr.img, X, Y, m["glow"], st["glow"]["c"], st["glow"]["a"])
    if m["shadow"] is not None:
        paint(fr.img, X, Y, m["shadow"], st["shadow"]["c"], st["shadow"]["a"])
    if m["outer"] is not None:
        paint(fr.img, X, Y, m["outer"], st["outer_c"])
    if m["out"] is not None:
        paint(fr.img, X, Y, m["out"], st["stroke_c"])
    rows = None
    if "grad" in st:
        t = np.clip((np.arange(it.h_mask, dtype=np.float32) - it.iy0) / max(1, it.h), 0, 1)[:, None]
        rows = st["grad"][0][None, :] * (1 - t) + st["grad"][1][None, :] * t
    paint(fr.img, X, Y, m["fill"], st["fill"], 1.0, rows)
    n = len(fr.inst)
    if fr.add(X, Y, it.cov, 0, role):
        fr.item_info[n] = (it.text, kind or (fr.elements[-1].split(":")[0] if fr.elements else None))
    return [X + it.ix0, Y + it.iy0, X + it.ix1, Y + it.iy1]


# ----------------------------------------------------------------------------- marks (background)

ICONS = ("bullet", "ring", "square", "square_r", "diamond", "tri", "star5", "star5o", "heart", "note", "kome",
         "gear", "speaker", "disk", "folder", "clock", "check", "cross", "plus", "arrow_l", "arrow_r", "chev2",
         "pause", "magnifier", "envelope", "home", "key", "book", "cursor_down", "hourglass")
BULLETS = ("bullet", "diamond", "square", "tri", "star5", "star5o", "note", "kome", "ring", "chev2")


def icon_mask(kind, s):
    """A float coverage mask (s x s, s >= 4) of one icon or bullet, drawn 4x supersampled."""
    import cv2
    n = max(4, int(round(s)))
    S = 4
    N = n * S
    m = np.zeros((N, N), np.uint8)
    c = N / 2.0
    r = N / 2.0 * 0.9
    t = max(1, int(round(N / 9)))

    def P(pts):
        return np.round(np.asarray(pts, np.float64)).astype(np.int32)

    def ci(x, y, rr, fill=True):
        cv2.circle(m, (int(round(x)), int(round(y))), max(1, int(round(rr))), 255, -1 if fill else t, cv2.LINE_AA)

    if kind == "bullet":
        ci(c, c, r * 0.55)
    elif kind == "ring":
        ci(c, c, r * 0.75, False)
    elif kind == "square":
        cv2.rectangle(m, (int(c - r * 0.6), int(c - r * 0.6)), (int(c + r * 0.6), int(c + r * 0.6)), 255, -1)
    elif kind == "square_r":
        cv2.rectangle(m, (int(c - r * 0.8), int(c - r * 0.8)), (int(c + r * 0.8), int(c + r * 0.8)), 255, t)
        ci(c, c, r * 0.3)
    elif kind == "diamond":
        cv2.fillPoly(m, [P([[c, c - r * 0.8], [c + r * 0.8, c], [c, c + r * 0.8], [c - r * 0.8, c]])], 255, cv2.LINE_AA)
    elif kind == "tri":
        cv2.fillPoly(m, [P([[c - r * 0.6, c - r * 0.75], [c + r * 0.75, c], [c - r * 0.6, c + r * 0.75]])], 255,
                     cv2.LINE_AA)
    elif kind in ("star5", "star5o"):
        pts = [[c + (r if k % 2 == 0 else r * 0.42) * math.sin(math.pi * k / 5),
                c - (r if k % 2 == 0 else r * 0.42) * math.cos(math.pi * k / 5)] for k in range(10)]
        if kind == "star5":
            cv2.fillPoly(m, [P(pts)], 255, cv2.LINE_AA)
        else:
            cv2.polylines(m, [P(pts)], True, 255, t, cv2.LINE_AA)
    elif kind == "heart":
        ci(c - r * 0.38, c - r * 0.2, r * 0.42)
        ci(c + r * 0.38, c - r * 0.2, r * 0.42)
        cv2.fillPoly(m, [P([[c - r * 0.78, c - r * 0.05], [c + r * 0.78, c - r * 0.05], [c, c + r * 0.85]])], 255,
                     cv2.LINE_AA)
    elif kind == "note":
        cv2.ellipse(m, (int(c - r * 0.25), int(c + r * 0.5)), (int(r * 0.35), int(r * 0.26)), -20, 0, 360, 255, -1,
                    cv2.LINE_AA)
        cv2.line(m, (int(c + r * 0.08), int(c + r * 0.45)), (int(c + r * 0.08), int(c - r * 0.85)), 255, t)
        cv2.line(m, (int(c + r * 0.08), int(c - r * 0.85)), (int(c + r * 0.6), int(c - r * 0.45)), 255, t)
    elif kind == "kome":
        e = r * 0.55
        cv2.line(m, (int(c - e), int(c - e)), (int(c + e), int(c + e)), 255, t, cv2.LINE_AA)
        cv2.line(m, (int(c - e), int(c + e)), (int(c + e), int(c - e)), 255, t, cv2.LINE_AA)
        for dx, dy in ((0.75, 0), (-0.75, 0), (0, 0.75), (0, -0.75)):
            ci(c + dx * r, c + dy * r, r * 0.13)
    elif kind == "gear":
        for k in range(8):
            a = math.pi * k / 4
            q = [[c + r * 0.95 * math.cos(a + d) * f, c + r * 0.95 * math.sin(a + d) * f]
                 for d, f in ((-0.2, 1.0), (0.2, 1.0), (0.3, 0.6), (-0.3, 0.6))]
            cv2.fillPoly(m, [P(q)], 255, cv2.LINE_AA)
        ci(c, c, r * 0.68)
        cv2.circle(m, (int(c), int(c)), max(1, int(r * 0.28)), 0, -1, cv2.LINE_AA)
    elif kind == "speaker":
        cv2.fillPoly(m, [P([[c - r * 0.85, c - r * 0.3], [c - r * 0.45, c - r * 0.3], [c, c - r * 0.75],
                            [c, c + r * 0.75], [c - r * 0.45, c + r * 0.3], [c - r * 0.85, c + r * 0.3]])], 255,
                     cv2.LINE_AA)
        for rr in (0.45, 0.8):
            cv2.ellipse(m, (int(c), int(c)), (int(r * rr), int(r * rr)), 0, -45, 45, 255, t, cv2.LINE_AA)
    elif kind == "disk":
        cv2.rectangle(m, (int(c - r * 0.8), int(c - r * 0.8)), (int(c + r * 0.8), int(c + r * 0.8)), 255, -1)
        cv2.rectangle(m, (int(c - r * 0.45), int(c - r * 0.8)), (int(c + r * 0.35), int(c - r * 0.3)), 0, -1)
        cv2.rectangle(m, (int(c - r * 0.5), int(c + r * 0.1)), (int(c + r * 0.5), int(c + r * 0.65)), 0, -1)
    elif kind == "folder":
        cv2.fillPoly(m, [P([[c - r * 0.85, c - r * 0.55], [c - r * 0.25, c - r * 0.55], [c - r * 0.1, c - r * 0.35],
                            [c + r * 0.85, c - r * 0.35], [c + r * 0.85, c + r * 0.65], [c - r * 0.85, c + r * 0.65]])],
                     255, cv2.LINE_AA)
    elif kind == "clock":
        ci(c, c, r * 0.85, False)
        cv2.line(m, (int(c), int(c)), (int(c), int(c - r * 0.55)), 255, t, cv2.LINE_AA)
        cv2.line(m, (int(c), int(c)), (int(c + r * 0.4), int(c + r * 0.15)), 255, t, cv2.LINE_AA)
    elif kind == "check":
        cv2.polylines(m, [P([[c - r * 0.7, c], [c - r * 0.2, c + r * 0.55], [c + r * 0.75, c - r * 0.6]])], False,
                      255, t + 1, cv2.LINE_AA)
    elif kind == "cross":
        e = r * 0.65
        cv2.line(m, (int(c - e), int(c - e)), (int(c + e), int(c + e)), 255, t + 1, cv2.LINE_AA)
        cv2.line(m, (int(c - e), int(c + e)), (int(c + e), int(c - e)), 255, t + 1, cv2.LINE_AA)
    elif kind == "plus":
        e = r * 0.7
        cv2.line(m, (int(c - e), int(c)), (int(c + e), int(c)), 255, t + 1, cv2.LINE_AA)
        cv2.line(m, (int(c), int(c - e)), (int(c), int(c + e)), 255, t + 1, cv2.LINE_AA)
    elif kind in ("arrow_l", "arrow_r"):
        sgn = 1 if kind == "arrow_r" else -1
        pts = [[c - sgn * r * 0.8, c - r * 0.2], [c, c - r * 0.2], [c, c - r * 0.6], [c + sgn * r * 0.8, c],
               [c, c + r * 0.6], [c, c + r * 0.2], [c - sgn * r * 0.8, c + r * 0.2]]
        cv2.fillPoly(m, [P(pts)], 255, cv2.LINE_AA)
    elif kind == "chev2":
        for dx in (-0.35, 0.25):
            cv2.polylines(m, [P([[c + (dx - 0.25) * r, c - r * 0.55], [c + (dx + 0.25) * r, c],
                                 [c + (dx - 0.25) * r, c + r * 0.55]])], False, 255, t + 1, cv2.LINE_AA)
    elif kind == "pause":
        for dx in (-0.4, 0.15):
            cv2.rectangle(m, (int(c + dx * r), int(c - r * 0.7)), (int(c + (dx + 0.25) * r), int(c + r * 0.7)), 255,
                          -1)
    elif kind == "magnifier":
        ci(c - r * 0.15, c - r * 0.15, r * 0.5, False)
        cv2.line(m, (int(c + r * 0.22), int(c + r * 0.22)), (int(c + r * 0.8), int(c + r * 0.8)), 255, t + 1,
                 cv2.LINE_AA)
    elif kind == "envelope":
        cv2.rectangle(m, (int(c - r * 0.85), int(c - r * 0.55)), (int(c + r * 0.85), int(c + r * 0.55)), 255, t)
        cv2.polylines(m, [P([[c - r * 0.85, c - r * 0.55], [c, c + r * 0.1], [c + r * 0.85, c - r * 0.55]])], False,
                      255, t, cv2.LINE_AA)
    elif kind == "home":
        cv2.fillPoly(m, [P([[c, c - r * 0.85], [c + r * 0.85, c - r * 0.05], [c + r * 0.55, c - r * 0.05],
                            [c + r * 0.55, c + r * 0.8], [c - r * 0.55, c + r * 0.8], [c - r * 0.55, c - r * 0.05],
                            [c - r * 0.85, c - r * 0.05]])], 255, cv2.LINE_AA)
    elif kind == "key":
        ci(c - r * 0.45, c, r * 0.35, False)
        cv2.line(m, (int(c - r * 0.1), int(c)), (int(c + r * 0.85), int(c)), 255, t, cv2.LINE_AA)
        cv2.line(m, (int(c + r * 0.6), int(c)), (int(c + r * 0.6), int(c + r * 0.35)), 255, t, cv2.LINE_AA)
    elif kind == "book":
        cv2.rectangle(m, (int(c - r * 0.85), int(c - r * 0.6)), (int(c - r * 0.05), int(c + r * 0.7)), 255, -1)
        cv2.rectangle(m, (int(c + r * 0.05), int(c - r * 0.6)), (int(c + r * 0.85), int(c + r * 0.7)), 255, -1)
    elif kind == "cursor_down":
        cv2.fillPoly(m, [P([[c - r * 0.75, c - r * 0.45], [c + r * 0.75, c - r * 0.45], [c, c + r * 0.65]])], 255,
                     cv2.LINE_AA)
    elif kind == "hourglass":
        cv2.fillPoly(m, [P([[c - r * 0.6, c - r * 0.8], [c + r * 0.6, c - r * 0.8], [c, c],
                            [c + r * 0.6, c + r * 0.8], [c - r * 0.6, c + r * 0.8], [c, c]])], 255, cv2.LINE_AA)
    elif kind in ICONS2:
        icon2_draw(m, kind, c, r, t, P, ci)
    else:
        raise ValueError(kind)
    return cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def mark(fr, kind, cx, cy, s, colour, plate=None):
    """A background mark (icon, bullet, cursor) of size s centred at (cx, cy): drawn, and listed in fr.bg (never a
    positive, never an ignore area). plate: (colour, alpha) of a disc or rounded square behind it."""
    n = max(4, int(round(s)))
    x0, y0 = int(round(cx - n / 2)), int(round(cy - n / 2))
    if x0 < 0 or y0 < 0 or x0 + n > fr.W or y0 + n > fr.H:
        return None
    if plate is not None:
        pad = max(2, int(round(n * 0.25)))
        D1.draw_box(fr.img, (x0 - pad, y0 - pad, x0 + n + pad, y0 + n + pad), plate[0], plate[1], (n + 2 * pad) // 2)
        fr.bg.append([max(0, x0 - pad), max(0, y0 - pad), min(fr.W, x0 + n + pad), min(fr.H, y0 + n + pad)])
    paint(fr.img, x0, y0, icon_mask(kind, n), colour)
    fr.bg.append([x0, y0, x0 + n, y0 + n])
    return [x0, y0, x0 + n, y0 + n]


def gauge(fr, rng, x0, y0, x1, y1, colour, frac=None):
    """A gauge bar (background): a frame and a fill to frac of its width."""
    S2 = S3().S2
    frac = float(rng.uniform(0.05, 1.0)) if frac is None else frac
    border = (1, S2.light_colour(rng, 0.3, 0.6)) if rng.random() < 0.6 else None
    D1.draw_box(fr.img, (x0, y0, x1, y1), S2.dark_colour(rng, 0.2), float(rng.uniform(0.5, 0.9)),
                int((y1 - y0) * rng.uniform(0, 0.5)), border)
    D1.draw_box(fr.img, (x0 + 1, y0 + 1, x0 + 1 + (x1 - x0 - 2) * frac, y1 - 1), colour, 1.0,
                int((y1 - y0) * 0.3))
    fr.bg.append([int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))])


def slider(fr, rng, x0, cy, x1, colour, s):
    """A slider (background): a track and a knob."""
    t = max(1, int(round(s * 0.15)))
    D1.draw_box(fr.img, (x0, cy - t / 2, x1, cy + t / 2 + 1), colour, 1.0, t // 2)
    kx = float(rng.uniform(x0, x1))
    mark(fr, "bullet" if rng.random() < 0.6 else "square", kx, cy, s, S3().S2.light_colour(rng, 0.2, 0.85))
    fr.bg.append([int(x0), int(cy - t), int(math.ceil(x1)), int(cy + t + 1)])


def rating(fr, rng, x0, cy, s, n=5):
    kind = "star5" if rng.random() < 0.6 else "heart"
    k = int(rng.integers(0, n + 1))
    on, off = S3().S2.accent(rng, True), S3().S2.dark_colour(rng, 0.5)
    for i in range(n):
        mark(fr, kind if i < k or rng.random() < 0.5 else (kind + "o" if kind == "star5" else kind),
             x0 + s * (i + 0.5) * 1.15, cy, s, on if i < k else off)
    return x0 + s * n * 1.15


# ----------------------------------------------------------------------------- layout helpers

def plate_box(fr, rng, rect, kind, colour, alpha, em):
    S2 = S3().S2
    x0, y0, x1, y1 = rect
    if kind == "bare":
        return
    if kind == "underline":
        t = max(1, int(round(em * 0.06)))
        D1.draw_box(fr.img, (x0, y1 - t, x1, y1), colour, 1.0, 0)
        return
    radius = {"pill": (y1 - y0) // 2, "round": int(em * rng.uniform(0.15, 0.5)), "square": 0}[kind]
    border = (max(1, int(round(em * 0.05))), S2.light_colour(rng, 0.4, 0.6)) if rng.random() < 0.4 else None
    grad = (alpha * rng.uniform(0.5, 1.0), alpha) if rng.random() < 0.3 else None
    D1.draw_box(fr.img, (x0, y0, x1, y1), colour, alpha, int(radius), border, grad)


def plate_colour(rng):
    S2 = S3().S2
    c = S2.dark_colour(rng, 0.35) if rng.random() < 0.7 else S2.light_colour(rng, 0.25, 0.8)
    return c, float(rng.uniform(0.55, 0.97))


def font_for(rng, texts, em):
    f, wght = pick_ui_font(rng, set("".join(texts)))
    if f is None:
        return None, None
    return D1.get_font(f, em, wght), f


def items_for(rng, texts, font, em, st):
    P = MIX_UI["style"]
    latin = all(t.isascii() for t in texts)
    sp = em * rng.uniform(*P["spacing_em"]) if latin and rng.random() < P["spacing"] else 0.0
    sh = float(rng.uniform(*P["shear_k"])) if latin and rng.random() < P["shear"] else 0.0
    return [make_item(t, font, st, sp, sh) for t in texts]


def between(h):
    """The least ink gap between two items on a row, the taller of them h px high: GAP_BETWEEN x h, and 2 px for the
    rounding of their positions."""
    return GAP_BETWEEN * h + 2


def place_ui(fr, rng, w, h, where, items):
    """Top-left of a w x h layout placed on free space at least GAP_BETWEEN x its tallest item's height from every
    element placed before it (and keeping later ones that far: the grown rectangle is reserved), ROW_GAP above and
    below; None when there is no room."""
    hmax = max([it.h for it in items if it is not None] or [1])
    gx, gy = between(hmax), ROW_GAP * hmax + 1
    xy = TS().place(fr, rng, int(math.ceil(w + 2 * gx)), int(math.ceil(h + 2 * gy)), where)
    if xy is None:
        return None
    fr.reserve(xy[0], xy[1], xy[0] + w + 2 * gx, xy[1] + h + 2 * gy)
    return xy[0] + gx, xy[1] + gy


def fits_row(items, gaps):
    """The row's label-to-label gaps meet GAP_BETWEEN (gaps[i]: the ink gap between items i and i+1)."""
    return all(g >= GAP_BETWEEN * max(a.h, b.h) for a, b, g in zip(items, items[1:], gaps))


# ----------------------------------------------------------------------------- layouts

def lay_buttons(fr, rng):
    S2 = S3().S2
    latin = rng.random() < MIX_UI["latin_p"]
    pool = MENU_EN if latin else MENU_JP
    n = int(rng.integers(3, 11))
    texts = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    em = ui_em(rng, fr)
    font, f = font_for(rng, texts, em)
    if font is None:
        return False
    pk = choose(rng, MIX_UI["plate"])
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc) if pk not in ("bare", "underline") else None)
    items = [it for it in items_for(rng, texts, font, em, st) if it is not None]
    if len(items) < 2:
        return False
    vertical = rng.random() < 0.25
    icons = rng.random() < MIX_UI["icon_before"]
    isz = em * rng.uniform(0.7, 1.1) if icons else 0.0
    igap = em * rng.uniform(*MIX_UI["icon_gap_em"]) if icons else 0.0
    kinds = [ICONS[int(rng.integers(len(ICONS)))] for _ in items]
    padx, pady = em * rng.uniform(0.4, 1.2), em * rng.uniform(0.25, 0.6)
    up = max(it.up for it in items)
    down = max(it.down for it in items)
    bh = max(up + down, isz) + 2 * pady
    if vertical:
        bw = max(isz + igap + it.w for it in items) + 2 * padx
        pitch = bh + em * rng.uniform(0.1, 0.6)
        while len(items) > 2 and pitch * len(items) > fr.H * 0.9:
            items.pop()
        W_, H_ = bw, pitch * len(items)
    else:
        g = em * rng.uniform(0.1, 1.0)
        g = max(g, between(max(it.h for it in items)) - (2 * padx + isz + igap))
        while len(items) > 2 and sum(isz + igap + it.w + 2 * padx for it in items) + g * (len(items) - 1) > fr.W * 0.98:
            items.pop()
        W_ = sum(isz + igap + it.w + 2 * padx for it in items) + g * (len(items) - 1)
        H_ = bh
    xy = place_ui(fr, rng, W_, H_, "edge" if not vertical else "side", items)
    if xy is None:
        return False
    x, y = xy
    added = 0
    ic = S2.accent(rng, S2.luma(pc) < 128) if rng.random() < 0.5 else st["fill"]
    for i, it in enumerate(items):
        if vertical:
            bx, by = x, y + i * pitch
            w_i = bw
        else:
            bx = x + sum(isz + igap + jt.w + 2 * padx for jt in items[:i]) + g * i
            by = y
            w_i = isz + igap + it.w + 2 * padx
        plate_box(fr, rng, (bx, by, bx + w_i, by + bh), pk, pc, pa, em)
        base = by + pady + up + (bh - 2 * pady - up - down) / 2
        lx = bx + padx + isz + igap
        if icons:
            mark(fr, kinds[i], bx + padx + isz / 2, base + (it.down - it.up) / 2, isz, ic)
        if put(fr, it, lx, base, kind="buttons"):
            added += 1
    fr.elements.append(f"buttons:{'latin' if latin else 'jp'}:em{em}:{pk}:{'v' if vertical else 'h'}"
                       f"{':icons' if icons else ''}:{f.cat}")
    return added > 0


def lay_menu(fr, rng):
    S2 = S3().S2
    latin = rng.random() < MIX_UI["latin_p"]
    pool = MENU_EN if latin else MENU_JP
    n = int(rng.integers(3, 10))
    texts = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    em = ui_em(rng, fr, choose(rng, {"24-48": 0.3, "48-96": 0.45, "96-160": 0.25}))
    font, f = font_for(rng, texts, em)
    if font is None:
        return False
    st = ui_style(rng, em, None)
    items = [it for it in items_for(rng, texts, font, em, st) if it is not None]
    if len(items) < 2:
        return False
    pitch = em * rng.uniform(1.3, 2.2)
    pitch = max(pitch, max(it.h for it in items) * (1 + ROW_GAP) + 1)
    while len(items) > 2 and pitch * len(items) > fr.H * 0.92:
        items.pop()
    align = choose(rng, {"left": 0.5, "centre": 0.35, "right": 0.15})
    cur = rng.random() < 0.6
    csz = em * rng.uniform(0.5, 0.9)
    cgap = em * rng.uniform(0.2, 0.8)
    lead = (csz + cgap) if cur else 0.0
    wmax = max(it.w for it in items)
    W_, H_ = lead + wmax + em, pitch * len(items)
    xy = place_ui(fr, rng, W_, H_, "side" if rng.random() < 0.6 else "centre", items)
    if xy is None:
        return False
    x, y = xy
    sel = int(rng.integers(len(items)))
    bar = rng.random() < 0.4
    added = 0
    for i, it in enumerate(items):
        base = y + i * pitch + (pitch + it.up - it.down) / 2
        lx = x + lead + {"left": 0.0, "centre": (wmax - it.w) / 2, "right": wmax - it.w}[align]
        if i == sel and bar:
            D1.draw_box(fr.img, (x, y + i * pitch, x + W_, y + (i + 1) * pitch), S2.accent(rng, True),
                        float(rng.uniform(0.2, 0.5)), int(pitch * 0.2))
        if i == sel and cur:
            mark(fr, ["tri", "star5", "diamond", "bullet", "chev2", "heart"][int(rng.integers(6))], lx - cgap - csz / 2,
                 base + (it.down - it.up) / 2, csz, S2.accent(rng, True))
        if put(fr, it, lx, base, kind="menu"):
            added += 1
    fr.elements.append(f"menu:{'latin' if latin else 'jp'}:em{em}:{align}{':cursor' if cur else ''}:{f.cat}")
    return added > 0


def lay_stats(fr, rng):
    S2 = S3().S2
    n = int(rng.integers(2, 9))
    labels = [STAT_LABELS[i] for i in rng.choice(len(STAT_LABELS), size=n, replace=False)]
    values = [value_text(rng) for _ in labels]
    em = ui_em(rng, fr, choose(rng, {"12-24": 0.5, "24-48": 0.4, "48-96": 0.1}))
    joined = rng.random() < 0.3
    texts = [f"{a} {b}" for a, b in zip(labels, values)] if joined else labels + values
    font, f = font_for(rng, texts, em)
    if font is None:
        return False
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc))
    its = items_for(rng, texts, font, em, st)
    rows = list(zip(its, [None] * n)) if joined else list(zip(its[:n], its[n:]))
    rows = [(a, b) for a, b in rows if a is not None and (joined or b is not None)]
    if not rows:
        return False
    extra = choose(rng, {"none": 0.4, "gauge": 0.35, "rating": 0.25}) if not joined else "none"
    icons = rng.random() < MIX_UI["icon_before"]
    isz = em * rng.uniform(0.7, 1.0) if icons else 0.0
    igap = em * rng.uniform(*MIX_UI["icon_gap_em"]) if icons else 0.0
    hmax = max(max(a.h, b.h if b else 0) for a, b in rows)
    pitch = max(em * rng.uniform(1.3, 1.9), hmax * (1 + ROW_GAP) + 1)
    pad = em * rng.uniform(0.5, 1.2)
    lw = max(a.w for a, _ in rows)
    vw = max((b.w for _, b in rows if b is not None), default=0)
    mid = 0.0 if joined else max(between(hmax), em * rng.uniform(1.5, 6.0))
    if extra != "none":
        mid = max(mid, em * 5.5 + 2 * between(hmax))
    title = name_text(rng) if rng.random() < 0.35 else None
    tit = None
    if title:
        tf, _ = font_for(rng, [title], int(em * 1.2))
        tit = make_item(title, tf, st) if tf is not None else None
    th = (tit.h + em * 0.8) if tit else 0.0
    W_ = 2 * pad + isz + igap + lw + mid + vw
    H_ = 2 * pad + th + pitch * len(rows)
    if tit is not None:
        W_ = max(W_, 2 * pad + tit.w)
    xy = place_ui(fr, rng, W_, H_, "side" if rng.random() < 0.5 else "any", [x for row in rows for x in row] + [tit])
    if xy is None:
        return False
    x, y = xy
    D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa, int(em * rng.uniform(0, 0.6)),
                (max(1, int(em * 0.05)), S2.light_colour(rng, 0.4, 0.6)) if rng.random() < 0.5 else None)
    added = 0
    if tit is not None and put(fr, tit, x + pad, y + pad + tit.up, kind="stats"):
        added += 1
    ic = S2.accent(rng, S2.luma(pc) < 128)
    kind = ICONS[int(rng.integers(len(ICONS)))]
    for i, (a, b) in enumerate(rows):
        top = y + pad + th + i * pitch
        base = top + (pitch + max(a.up, b.up if b else 0) - max(a.down, b.down if b else 0)) / 2
        lx = x + pad + isz + igap
        if icons:
            mark(fr, kind, x + pad + isz / 2, top + pitch / 2, isz, ic)
        if put(fr, a, lx, base, kind="stats"):
            added += 1
        if b is not None:
            vx = x + W_ - pad - b.w
            if extra == "gauge":
                g0, g1 = lx + a.w + between(hmax), vx - between(hmax)
                if g1 - g0 > em:
                    gauge(fr, rng, g0, top + pitch * 0.35, g1, top + pitch * 0.65, S2.accent(rng, True))
            elif extra == "rating":
                s = em * 0.8
                r0 = lx + a.w + between(hmax)
                if r0 + s * 5 * 1.15 < vx - between(hmax):
                    rating(fr, rng, r0, top + pitch / 2, s)
            if put(fr, b, vx, base, kind="stats"):
                added += 1
    fr.elements.append(f"stats:em{em}:{'joined' if joined else 'column'}:{extra}{':icons' if icons else ''}:{f.cat}")
    return added > 0


def lay_header(fr, rng):
    S2 = S3().S2
    title = HEADERS[int(rng.integers(len(HEADERS)))]
    em = ui_em(rng, fr, choose(rng, {"48-96": 0.5, "96-160": 0.5}))
    font, f = font_for(rng, [title], em)
    if font is None:
        return False
    st = ui_style(rng, em, None)
    sp = em * rng.uniform(0.05, 0.3) if title.isascii() and rng.random() < 0.4 else 0.0
    tit = make_item(title, font, st, sp, float(rng.uniform(0.12, 0.22)) if title.isascii() and rng.random() < 0.15
                    else 0.0)
    if tit is None:
        return False
    n = int(rng.integers(0, 7))
    tabs = [TABS[i] for i in rng.choice(len(TABS), size=n, replace=False)] if n else []
    tem = ui_em(rng, fr, choose(rng, {"12-24": 0.6, "24-48": 0.4}))
    titems = []
    if tabs:
        tfont, _ = font_for(rng, tabs, tem)
        if tfont is not None:
            pc, pa = plate_colour(rng)
            tst = ui_style(rng, tem, S2.luma(pc))
            titems = [it for it in items_for(rng, tabs, tfont, tem, tst) if it is not None]
    padx = tem * rng.uniform(0.6, 1.4)
    pady = tem * rng.uniform(0.3, 0.6)
    tg = 0.0
    if titems:
        tg = max(tem * rng.uniform(0.1, 0.6), between(max(it.h for it in titems)) - 2 * padx)
        while len(titems) > 1 and sum(it.w + 2 * padx for it in titems) + tg * (len(titems) - 1) > fr.W * 0.95:
            titems.pop()
    tabs_w = sum(it.w + 2 * padx for it in titems) + tg * max(0, len(titems) - 1)
    tabs_h = (max(it.up + it.down for it in titems) + 2 * pady) if titems else 0.0
    W_ = max(tit.w + em * 0.5, tabs_w)
    H_ = tit.up + tit.down + em * 0.4 + (tabs_h + tem * 0.5 if titems else 0)
    if W_ > fr.W * 0.98:
        return False
    xy = place_ui(fr, rng, W_, H_, "upper", [tit] + titems)
    if xy is None:
        return False
    x, y = xy
    added = 0
    if rng.random() < 0.4:
        t = max(1, int(round(em * 0.05)))
        D1.draw_box(fr.img, (x, y + tit.up + tit.down + em * 0.2, x + W_, y + tit.up + tit.down + em * 0.2 + t),
                    S2.accent(rng, True), 1.0, 0)
    if put(fr, tit, x, y + tit.up, kind="header"):
        added += 1
    if titems:
        ty = y + tit.up + tit.down + em * 0.4 + tem * 0.5
        up = max(it.up for it in titems)
        down = max(it.down for it in titems)
        sel = int(rng.integers(len(titems)))
        cx = x
        for i, it in enumerate(titems):
            w_i = it.w + 2 * padx
            D1.draw_box(fr.img, (cx, ty, cx + w_i, ty + tabs_h), S2.accent(rng, True) if i == sel else pc,
                        pa, int(tabs_h * rng.uniform(0, 0.4)))
            if put(fr, it, cx + padx, ty + pady + up + (tabs_h - 2 * pady - up - down) / 2, kind="header"):
                added += 1
            cx += w_i + tg
    fr.elements.append(f"header:em{em}:tabs{len(titems)}:{f.cat}")
    return added > 0


def lay_config(fr, rng):
    S2 = S3().S2
    n = int(rng.integers(2, 8))
    labels = [CONFIG_LABELS[i] for i in rng.choice(len(CONFIG_LABELS), size=n, replace=False)]
    em = ui_em(rng, fr, choose(rng, {"12-24": 0.55, "24-48": 0.45}))
    rows = []
    for lab in labels:
        if rng.random() < 0.55:
            rows.append((lab, "slider", [f"{int(rng.integers(0, 101))}"]))
        else:
            k = int(rng.integers(2, 4))
            rows.append((lab, "options", [OPTION_VALUES[i] for i in rng.choice(len(OPTION_VALUES), size=k,
                                                                                replace=False)]))
    alltexts = [r[0] for r in rows] + [t for r in rows for t in r[2]]
    font, f = font_for(rng, alltexts, em)
    if font is None:
        return False
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc))
    built = []
    for lab, kind, vals in rows:
        a = make_item(lab, font, st)
        vs = [make_item(v, font, st) for v in vals]
        if a is None or any(v is None for v in vs):
            continue
        built.append((a, kind, vs))
    if not built:
        return False
    hmax = max(max([a.h] + [v.h for v in vs]) for a, _, vs in built)
    pitch = max(em * rng.uniform(1.5, 2.2), hmax * (1 + ROW_GAP) + 1 + em * 0.4)
    pad = em * rng.uniform(0.6, 1.2)
    lw = max(a.w for a, _, _ in built)
    gap = between(hmax)
    opad = em * rng.uniform(0.3, 0.7)
    def opt_w(vs):
        return sum(v.w + 2 * opad for v in vs) + max(0.0, gap - 2 * opad) * (len(vs) - 1)
    rw = max([em * 8 + gap + max((v.w for v in vs), default=0) for _, k, vs in built if k == "slider"] +
             [opt_w(vs) for _, k, vs in built if k == "options"])
    W_ = 2 * pad + lw + gap + rw
    H_ = 2 * pad + pitch * len(built)
    if W_ > fr.W * 0.98:
        return False
    xy = place_ui(fr, rng, W_, H_, "centre" if rng.random() < 0.5 else "any",
                  [a for a, _, _ in built] + [v for _, _, vs in built for v in vs])
    if xy is None:
        return False
    x, y = xy
    D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa, int(em * rng.uniform(0, 0.5)))
    added = 0
    oc, oa = plate_colour(rng)
    sel_c = S2.accent(rng, True)
    for i, (a, kind, vs) in enumerate(built):
        top = y + pad + i * pitch
        up = max([a.up] + [v.up for v in vs])
        down = max([a.down] + [v.down for v in vs])
        base = top + (pitch + up - down) / 2
        if put(fr, a, x + pad, base, kind="config"):
            added += 1
        rx = x + pad + lw + gap
        if kind == "slider":
            v = vs[0]
            slider(fr, rng, rx, top + pitch / 2, rx + em * 8, sel_c, em * 0.6)
            if put(fr, v, rx + em * 8 + gap, base, kind="config"):
                added += 1
        else:
            sel = int(rng.integers(len(vs)))
            cx = rx
            for j, v in enumerate(vs):
                D1.draw_box(fr.img, (cx, top + pitch * 0.15, cx + v.w + 2 * opad, top + pitch * 0.85),
                            sel_c if j == sel else oc, oa, int(pitch * 0.2))
                if put(fr, v, cx + opad, base, kind="config"):
                    added += 1
                cx += v.w + 2 * opad + max(0.0, gap - 2 * opad)
    fr.elements.append(f"config:em{em}:{len(built)}:{f.cat}")
    return added > 0


def lay_list(fr, rng):
    S2 = S3().S2
    n = int(rng.integers(3, 9))
    kind = choose(rng, MIX_UI["list_kind"])
    src = choose(rng, {"phrase": 0.5, "item": 0.3, "name": 0.2})
    texts = []
    for i in range(n):
        t = short_text(rng, 3, 14) if src == "phrase" else (ITEMS[int(rng.integers(len(ITEMS)))] if src == "item"
                                                              else (name_text(rng) or short_text(rng, 2, 8)))
        if kind == "nakaguro":
            t = "・" + t
        elif kind == "numbered":
            t = (f"{i + 1}." if rng.random() < 0.5 else "①②③④⑤⑥⑦⑧⑨"[i]) + t
        texts.append(t)
    em = ui_em(rng, fr, choose(rng, {"12-24": 0.55, "24-48": 0.4, "48-96": 0.05}))
    font, f = font_for(rng, texts, em)
    if font is None:
        return False
    on_plate = rng.random() < 0.7
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc) if on_plate else None)
    items = [make_item(t, font, st) for t in texts]
    items = [it for it in items if it is not None]
    if len(items) < 2:
        return False
    pitch = max(em * rng.uniform(1.3, 1.9), max(it.h for it in items) * (1 + ROW_GAP) + 1)
    pad = em * rng.uniform(0.5, 1.2)
    bsz = em * rng.uniform(0.4, 0.8)
    bgap = em * rng.uniform(0.2, 0.6)
    lead = (bsz + bgap) if kind == "icon" else 0.0
    W_ = 2 * pad + lead + max(it.w for it in items)
    H_ = 2 * pad + pitch * len(items)
    xy = place_ui(fr, rng, W_, H_, "any", items)
    if xy is None:
        return False
    x, y = xy
    if on_plate:
        D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa, int(em * rng.uniform(0, 0.6)))
    bk = BULLETS[int(rng.integers(len(BULLETS)))]
    bc = S2.accent(rng, True) if rng.random() < 0.5 else st["fill"]
    added = 0
    for i, it in enumerate(items):
        top = y + pad + i * pitch
        base = top + (pitch + it.up - it.down) / 2
        if kind == "icon":
            mark(fr, bk, x + pad + bsz / 2, top + pitch / 2, bsz, bc)
        if put(fr, it, x + pad + lead, base, kind="list"):
            added += 1
    fr.elements.append(f"list:{kind}:{src}:em{em}:{f.cat}")
    return added > 0


def lay_slots(fr, rng):
    S2 = S3().S2
    cols, nrows = int(rng.integers(1, 4)), int(rng.integers(2, 6))
    em = ui_em(rng, fr, choose(rng, {"12-24": 0.75, "24-48": 0.25}))
    tiles = []
    for k in range(cols * nrows):
        empty = rng.random() < 0.15
        no = f"No.{k + 1:02d}" if rng.random() < 0.7 else f"{k + 1:03d}"
        date = None if empty else f"20{int(rng.integers(10, 27))}/{int(rng.integers(1, 13)):02d}/" \
                                  f"{int(rng.integers(1, 29)):02d} {int(rng.integers(0, 24)):02d}:" \
                                  f"{int(rng.integers(0, 60)):02d}"
        title = ("NO DATA" if rng.random() < 0.5 else "EMPTY") if empty else short_text(rng, 4, 14)
        tiles.append((no, date, title))
    texts = [t for tile in tiles for t in tile if t]
    font, f = font_for(rng, texts, em)
    if font is None:
        return False
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc))
    built = []
    for no, date, title in tiles:
        a, b, c = make_item(no, font, st), make_item(date, font, st) if date else None, make_item(title, font, st)
        if a is None or c is None or (date and b is None):
            continue
        built.append((a, b, c))
    if len(built) < 2:
        return False
    hmax = max(max(a.h, c.h, b.h if b else 0) for a, b, c in built)
    pad = em * rng.uniform(0.4, 0.8)
    gap = between(hmax)
    thumb = em * rng.uniform(3.0, 5.0) if rng.random() < 0.6 else 0.0
    row1 = max(a.w + gap + (b.w if b else 0) for a, b, _ in built)
    tw = 2 * pad + thumb + (gap if thumb else 0) + max(row1, max(c.w for _, _, c in built))
    lh = max(em * 1.4, hmax * (1 + ROW_GAP) + 1)
    th = 2 * pad + max(2 * lh, thumb * 0.75)
    tg = max(em * rng.uniform(0.3, 1.0), between(hmax) - 2 * pad)
    cols = min(cols, max(1, int((fr.W * 0.95 + tg) // (tw + tg))))
    nrows = min((len(built) + cols - 1) // cols, max(1, int((fr.H * 0.9 + tg) // (th + tg))))
    built = built[:cols * nrows]
    W_, H_ = cols * tw + (cols - 1) * tg, nrows * th + (nrows - 1) * tg
    xy = place_ui(fr, rng, W_, H_, "centre", [x for t in built for x in t])
    if xy is None:
        return False
    x, y = xy
    added = 0
    for k, (a, b, c) in enumerate(built):
        cx, cy = x + (k % cols) * (tw + tg), y + (k // cols) * (th + tg)
        D1.draw_box(fr.img, (cx, cy, cx + tw, cy + th), pc, pa, int(em * rng.uniform(0, 0.4)),
                    (1, S2.light_colour(rng, 0.4, 0.6)) if rng.random() < 0.5 else None)
        tx = cx + pad
        if thumb:
            D1.draw_box(fr.img, (tx, cy + pad, tx + thumb, cy + pad + thumb * 0.75), S2.dark_colour(rng, 0.6), 0.9, 0)
            tx += thumb + gap
        base1 = cy + pad + (lh + max(a.up, b.up if b else 0) - max(a.down, b.down if b else 0)) / 2
        if put(fr, a, tx, base1, kind="slots"):
            added += 1
        if b is not None and put(fr, b, tx + a.w + gap, base1, kind="slots"):
            added += 1
        if put(fr, c, tx, cy + pad + lh + (lh + c.up - c.down) / 2, kind="slots"):
            added += 1
    fr.elements.append(f"slots:em{em}:{cols}x{nrows}{':thumb' if thumb else ''}:{f.cat}")
    return added > 0


def lay_pairs(fr, rng):
    S2 = S3().S2
    n = int(rng.integers(1, 5))
    added = 0
    for _ in range(n):
        lab, val = pair_text(rng)
        joined = val is not None and rng.random() < 0.5
        texts = [f"{lab} {val}"] if joined else ([lab] if val is None else [lab, val])
        em = ui_em(rng, fr, choose(rng, {"12-24": 0.4, "24-48": 0.45, "48-96": 0.15}))
        font, f = font_for(rng, texts, em)
        if font is None:
            continue
        on_plate = rng.random() < 0.5
        pc, pa = plate_colour(rng)
        st = ui_style(rng, em, S2.luma(pc) if on_plate else None)
        its = [make_item(t, font, st) for t in texts]
        if any(it is None for it in its):
            continue
        hmax = max(it.h for it in its)
        gap = between(hmax) + em * rng.uniform(0, 1.5)
        icon = rng.random() < MIX_UI["icon_before"] * 1.5
        isz = em * rng.uniform(0.7, 1.1) if icon else 0.0
        igap = em * rng.uniform(*MIX_UI["icon_gap_em"]) if icon else 0.0
        padx, pady = em * rng.uniform(0.4, 1.0), em * rng.uniform(0.2, 0.5)
        up, down = max(it.up for it in its), max(it.down for it in its)
        W_ = 2 * padx + isz + igap + sum(it.w for it in its) + gap * (len(its) - 1)
        H_ = max(up + down, isz) + 2 * pady
        xy = place_ui(fr, rng, W_, H_, "edge", its)
        if xy is None:
            continue
        x, y = xy
        if on_plate:
            D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa, int(H_ * rng.uniform(0, 0.5)))
        base = y + pady + up + (H_ - 2 * pady - up - down) / 2
        if icon:
            mark(fr, ICONS[int(rng.integers(len(ICONS)))], x + padx + isz / 2, y + H_ / 2, isz,
                 S2.accent(rng, True) if rng.random() < 0.6 else st["fill"])
        cx = x + padx + isz + igap
        for it in its:
            if put(fr, it, cx, base, kind="pairs"):
                added += 1
            cx += it.w + gap
        fr.elements.append(f"pairs:{'joined' if joined else ('single' if val is None else 'apart')}:em{em}"
                           f"{':icon' if icon else ''}:{f.cat}")
    return added > 0


LAYOUTS = {"buttons": lay_buttons, "menu": lay_menu, "stats": lay_stats, "header": lay_header, "config": lay_config,
           "list": lay_list, "slots": lay_slots, "pairs": lay_pairs}


# ----------------------------------------------------------------------------- desk_ui v2 (render --version 2)
#
# desk_ui v2 (textspot-desktop-ui2, 2026-11-04; scope: $A/textspot-desktop-ui2/desk_ui_v2_scope.md). I1 (desk_ui v1)
# still reads characters into a label from what sits next to it: an icon or badge touching it or a small gap from it
# (身, 百, 晶), leader dots or rules between a label and its value, brackets and marks the labels leave out, and UI
# rows tagged dialogue, which the reading stage's cut never cuts into items (自武将数15人自兵数930人, 攻撃力20). v2
# draws the v1 layouts (at about 35% of the layout draws, unchanged) and six layouts made of exactly those:
#   statbar   a row (or a grid) of 2-6 stat items on a top or bottom bar, in a wide box like a dialogue window, on a
#             panel or bare: runs (攻撃力20, 兵数930人), runs in two styles, one-box pairs, single tokens; items
#             at least GAP_BETWEEN apart with separators (bars, diamonds, slashes) and attached marks between them
#   ledger    3-8 rows: label, leader, value (two boxes; the leader starts and ends 0.1-0.5 h from their ink), a
#             one-box pair, a run, or label and value far apart; or a table of contents (第一章… 12)
#   iconlist  item, skill and menu lists (or a toolbar row) with an icon or badge touching or near each label,
#             counts as runs (薬草×3) or in a column
#   selector  config rows: a value between arrow marks (◀ 普通 ▶), options after check / radio boxes, a slider
#   framed    a title between ornament marks or graphic rules, inside graphic brackets, or with bracket glyphs of
#             its own font (【装備】: text, inside the box, as 1b keeps them)
#   pairs2    top-bar pairs in one box (DAY　12, 所持金　3,450円) or runs, each with an icon or badge
# 1b's conventions (textspot-desktop-targets' convention_report.md / convention_gaps_*.json; textspot-desktop-ui2's
# convention_pairs_1b.json): brackets in the text are inside the box, marks are outside; one box per item; two boxes
# on a row at least GAP_BETWEEN apart; a label and its value in ONE box when their gap is under 1.8 h (1b: 325 of
# 325 one-box pairs below 1.5 h, 12 of 15 at 1.5-1.8 h; split from 1.8 h, always from 2.5 h), so a pair's gap is
# drawn in GAP_PAIR and a two-box pair at GAP_BETWEEN or more. Marks, badges, leaders, separators, arrows,
# ornaments and graphic brackets are background: their ink boxes (coverage over MARK_INK) go to fr.bg (never a box,
# never an ignore area; the guard leaves them alone), an attached mark's gap to its label to fr.attached (meta
# "attached": [mark index, line index, side, gap px]) and a one-box pair's gap to fr.pair_gap (meta "pairs").

VERSIONS = (1, 2)
VERSION = 1              # the version being drawn (set_version; render --version)
MARK_INK = 0.05          # a background mark's ink: its coverage over this (the box listed in fr.bg)
GAP_PAIR = (0.5, 1.75)   # x the pair's height: the gap of a label + value pair drawn in one box
MIX_UI2 = {
    "screen": dict(MIX_UI["screen"]),
    "game": dict(MIX_UI["game"]),
    "captured": MIX_UI["captured"],
    "native_jpeg": dict(MIX_UI["native_jpeg"]),
    "n_layouts": [0.25, 0.4, 0.35],
    "layout": {"buttons": 0.063, "menu": 0.056, "stats": 0.049, "header": 0.035, "config": 0.0315, "list": 0.0455,
               "slots": 0.021, "pairs": 0.049,
               "statbar": 0.15, "ledger": 0.13, "iconlist": 0.13, "selector": 0.09, "framed": 0.07, "pairs2": 0.08},
    "dialogue": 0.3,
    "dialogue_layout": dict(MIX_UI["dialogue_layout"]),
    "mark_gap": {"touch_p": 0.3, "touch": [0.0, 0.05], "near": [0.1, 0.5]},
    "icon_size": [0.7, 1.25],
    "badge_size": [0.55, 1.0],
    "arrow_size": [0.4, 0.8],
    "orn_size": [0.4, 0.9],
    "badge_p": 0.35,
    "pair_gap": list(GAP_PAIR),
    "run2_gap": [0.04, 0.12],
    "leader": {"dots": 0.2, "squares": 0.08, "dash": 0.15, "rule": 0.17, "double": 0.07, "dotted": 0.08, "glyph": 0.25},
    "leader_gap": [0.1, 0.5],
    "glyph_leader_gap": [0.3, 0.5],
    "leader_min": 1.0,
    "glyph_leader_min": 3.5,
    "glyph_leaders": {"…": 0.15, "‥": 0.15, "．": 0.2, "－": 0.15, "―": 0.15, "─": 0.2},
    "separator": {"vbar": 0.45, "diamond": 0.2, "slash": 0.2, "chev": 0.15},
    "scenes": dict(MIX_UI["scenes"]),
}
STAT_LABELS2 = STAT_LABELS + [     # generic labels (not the words of the dense_dev errors that motivated the layouts)
    "武将数", "兵数", "部隊数", "兵士", "兵力", "兵糧", "武将", "城数", "国力", "金銭", "米", "民忠", "人口",
    "技術", "統率", "武力", "知略", "政治", "士気", "訓練", "ターン", "残りターン", "所持数", "合計", "最大", "現在",
    "攻撃", "防御", "命中", "回避", "会心", "耐久", "移動力", "射程", "重量", "価格", "売値", "在庫", "評価", "人気",
    "親密度", "愛情", "友情", "体重", "身長", "年齢", "撃破数", "討伐数", "達成率", "収集率", "プレイ時間", "既読率",
    "スコア", "コンボ", "ポイント", "所持ポイント", "ゴールド", "コイン", "ジェム", "スタミナ", "ライフ", "残機",
    "フロア", "階層", "ステージ", "WAVE", "COMBO", "BONUS", "TOTAL", "COST", "CP", "AP", "BP", "TP", "PT"]
UNITS = ("人", "円", "G", "個", "回", "日", "歳", "cm", "kg", "pt", "%", "枚", "本", "匹", "台", "石", "両", "名", "件",
         "点", "秒", "分", "時間", "階", "F")
ITEMS2 = ITEMS + [
    "回復薬", "エーテル", "鉄の剣", "木の盾", "皮の鎧", "魔法の杖", "銀の指輪", "解毒薬", "毒消し草", "聖水", "宝石",
    "金貨", "銀貨", "銅貨", "火打ち石", "釣り竿", "つるはし", "松明", "保存食", "水筒", "設計図", "古びた鍵", "謎の石",
    "魔導書", "羽ペン", "インク", "砂時計", "望遠鏡", "方位磁石", "首飾り", "髪飾り", "香水"]
SKILLS = ["ファイア", "ブリザード", "サンダー", "ヒール", "キュア", "斬撃", "連撃", "突き", "防御", "かばう", "挑発",
          "回復", "鑑定", "料理", "採取", "調合", "解錠", "隠密", "跳躍", "疾風", "烈火", "氷結", "雷撃", "聖なる光",
          "見切り", "反撃", "集中", "瞑想", "祈り", "応援"]
CHAPTERS = ["序章", "第一章", "第二章", "第三章", "第四章", "第五章", "第六章", "終章", "幕間", "プロローグ",
            "エピローグ", "Chapter 1", "Chapter 2", "Chapter 3", "EPISODE 1", "EPISODE 2", "第1話", "第2話", "第3話"]
ORN_TITLES = HEADERS + [
    "装備", "持ち物", "所持品", "仲間", "編成", "任務", "依頼", "図鑑", "実績", "称号", "ショップ", "購入", "売却",
    "合成", "強化", "スキル", "魔法", "作戦", "情報", "履歴", "サウンド", "テキスト", "画面設定", "基本設定"]
TEXT_BRACKETS = (("【", "】"), ("『", "』"), ("「", "」"), ("〔", "〕"), ("［", "］"), ("《", "》"), ("〈", "〉"),
                 ("（", "）"))
ICONS2 = ("doc", "grid", "stack", "calendar", "chest", "battery", "signal", "gem", "sword", "shield", "crown", "flag",
          "bell", "lock", "coin", "potion", "person", "spark", "flower", "tri_l", "chev_r", "chev_l", "angle_r",
          "angle_l", "arrow_up", "checkbox", "checkbox_on", "radio", "radio_on")
PICTOS = ICONS + ICONS2[:17]                         # attached icons: v1's icons and the kanji-like pictograms
BULLETS2 = ("bullet", "diamond", "square", "tri", "star5", "star5o", "spark", "flower", "heart")   # no ・ and no ※
ARROWS = (("tri_l", "tri"), ("chev_l", "chev_r"), ("angle_l", "angle_r"), ("arrow_l", "arrow_r"))
CHECKS = (("checkbox", "checkbox_on"), ("radio", "radio_on"))
BADGE_PLATES = ("disc", "rsquare", "pill", "burst", "shield", "hex")
BADGE_INNER = ("bullet", "check", "plus", "star5", "heart", "arrow_up", "chev2", "cross", "diamond", "tri")


def set_version(v):
    """The desk_ui version sample_frame draws: 1 (desk_ui v1, bit for bit) or 2."""
    global VERSION
    if int(v) not in VERSIONS:
        raise ValueError(f"desk_ui version {v}: one of {VERSIONS}")
    VERSION = int(v)


def mix():
    return MIX_UI if VERSION == 1 else MIX_UI2


def icon2_draw(m, kind, c, r, t, P, ci):
    """icon_mask's v2 kinds (ICONS2) on its 4x canvas m: pictograms that read like kanji at small sizes (a document
    with lines, a grid, a stack of boxes, a calendar, a chest, a battery, signal bars, a gem, a sword, a shield, a
    crown, a flag, a bell, a lock, a coin, a potion, a person), ornaments (a four-point spark, a flower), arrows and
    chevrons for selectors, and check / radio boxes."""
    import cv2

    def R(x0, y0, x1, y1, v=255, th=-1):
        cv2.rectangle(m, (int(round(c + x0 * r)), int(round(c + y0 * r))), (int(round(c + x1 * r)),
                                                                          int(round(c + y1 * r))), v, th)

    def L(x0, y0, x1, y1, th=t, v=255):
        cv2.line(m, (int(round(c + x0 * r)), int(round(c + y0 * r))), (int(round(c + x1 * r)), int(round(c + y1 * r))),
                 v, th, cv2.LINE_AA)

    def F(pts, v=255):
        cv2.fillPoly(m, [P([[c + x * r, c + y * r] for x, y in pts])], v, cv2.LINE_AA)

    if kind == "doc":
        R(-0.62, -0.85, 0.62, 0.85, th=t)
        for y in (-0.42, 0.0, 0.42):
            L(-0.36, y, 0.36, y)
    elif kind == "grid":
        for x0, y0 in ((-0.85, -0.85), (0.08, -0.85), (-0.85, 0.08), (0.08, 0.08)):
            R(x0, y0, x0 + 0.77, y0 + 0.77)
    elif kind == "stack":
        for x0, y0 in ((-0.38, -0.88), (-0.88, 0.06), (0.12, 0.06)):
            R(x0, y0, x0 + 0.76, y0 + 0.8, th=t)
            L(x0 + 0.1, y0 + 0.4, x0 + 0.66, y0 + 0.4)
    elif kind == "calendar":
        R(-0.82, -0.7, 0.82, 0.82, th=t)
        R(-0.82, -0.7, 0.82, -0.32)
        for x in (-0.45, 0.0, 0.45):
            for y in (0.05, 0.45):
                ci(c + x * r, c + y * r, r * 0.1)
        L(-0.4, -0.92, -0.4, -0.55)
        L(0.4, -0.92, 0.4, -0.55)
    elif kind == "chest":
        R(-0.88, -0.2, 0.88, 0.78)
        cv2.ellipse(m, (int(c), int(c - r * 0.2)), (int(r * 0.88), int(r * 0.55)), 0, 180, 360, 255, -1, cv2.LINE_AA)
        L(-0.88, -0.2, 0.88, -0.2, th=max(1, t), v=0)
        R(-0.16, -0.36, 0.16, 0.08, v=0)
    elif kind == "battery":
        R(-0.88, -0.48, 0.66, 0.48, th=t)
        R(0.66, -0.2, 0.88, 0.2)
        for k in range(3):
            R(-0.74 + k * 0.46, -0.32, -0.42 + k * 0.46, 0.32)
    elif kind == "signal":
        for k in range(4):
            R(-0.85 + k * 0.45, 0.82 - (k + 1) * 0.4, -0.55 + k * 0.45, 0.82)
    elif kind == "gem":
        F([(-0.5, -0.62), (0.5, -0.62), (0.88, -0.15), (0.0, 0.88), (-0.88, -0.15)])
        L(-0.88, -0.15, 0.88, -0.15, th=max(1, t // 2), v=0)
        L(-0.2, -0.62, 0.0, 0.88, th=max(1, t // 2), v=0)
        L(0.2, -0.62, 0.0, 0.88, th=max(1, t // 2), v=0)
    elif kind == "sword":
        L(-0.62, 0.62, 0.78, -0.78, th=int(t * 1.6))
        L(-0.72, 0.18, -0.18, 0.72, th=t)
        ci(c - r * 0.8, c + r * 0.8, r * 0.16)
    elif kind == "shield":
        F([(-0.75, -0.82), (0.75, -0.82), (0.75, 0.0), (0.0, 0.9), (-0.75, 0.0)])
        L(0.0, -0.66, 0.0, 0.6, th=max(1, t // 2), v=0)
        L(-0.6, -0.2, 0.6, -0.2, th=max(1, t // 2), v=0)
    elif kind == "crown":
        F([(-0.88, 0.62), (-0.88, -0.5), (-0.44, 0.02), (0.0, -0.78), (0.44, 0.02), (0.88, -0.5), (0.88, 0.62)])
    elif kind == "flag":
        L(-0.62, -0.88, -0.62, 0.9, th=t)
        F([(-0.62, -0.86), (0.82, -0.6), (-0.62, -0.08)])
    elif kind == "bell":
        cv2.ellipse(m, (int(c), int(c - r * 0.1)), (int(r * 0.55), int(r * 0.65)), 0, 180, 360, 255, -1, cv2.LINE_AA)
        F([(-0.55, -0.1), (0.55, -0.1), (0.85, 0.55), (-0.85, 0.55)])
        ci(c, c + r * 0.72, r * 0.16)
    elif kind == "lock":
        cv2.ellipse(m, (int(c), int(c - r * 0.2)), (int(r * 0.42), int(r * 0.5)), 0, 180, 360, 255, t, cv2.LINE_AA)
        L(-0.42, -0.2, -0.42, 0.05)
        L(0.42, -0.2, 0.42, 0.05)
        R(-0.68, 0.0, 0.68, 0.88)
        ci(c, c + r * 0.36, r * 0.13)
        cv2.circle(m, (int(c), int(c + r * 0.36)), max(1, int(r * 0.13)), 0, -1, cv2.LINE_AA)
    elif kind == "coin":
        ci(c, c, r * 0.88)
        cv2.circle(m, (int(c), int(c)), max(1, int(r * 0.62)), 0, max(1, t // 2), cv2.LINE_AA)
        R(-0.12, -0.12, 0.12, 0.12, v=0)
    elif kind == "potion":
        ci(c, c + r * 0.3, r * 0.58)
        R(-0.2, -0.62, 0.2, -0.1)
        R(-0.3, -0.85, 0.3, -0.62)
    elif kind == "person":
        ci(c, c - r * 0.45, r * 0.34)
        cv2.ellipse(m, (int(c), int(c + r * 0.85)), (int(r * 0.7), int(r * 0.68)), 0, 180, 360, 255, -1, cv2.LINE_AA)
    elif kind == "spark":
        F([(0.0, -0.9), (0.18, -0.18), (0.9, 0.0), (0.18, 0.18), (0.0, 0.9), (-0.18, 0.18), (-0.9, 0.0),
           (-0.18, -0.18)])
    elif kind == "flower":
        for k in range(5):
            a = 2 * math.pi * k / 5
            ci(c + 0.5 * r * math.cos(a), c + 0.5 * r * math.sin(a), r * 0.36)
        cv2.circle(m, (int(c), int(c)), max(1, int(r * 0.2)), 0, -1, cv2.LINE_AA)
    elif kind == "tri_l":
        F([(0.6, -0.75), (-0.75, 0.0), (0.6, 0.75)])
    elif kind in ("chev_r", "chev_l", "angle_r", "angle_l"):
        s = 1 if kind.endswith("_r") else -1
        th = t + 1 if kind.startswith("chev") else max(1, t // 2 + 1)
        w = 0.42 if kind.startswith("chev") else 0.3
        cv2.polylines(m, [P([[c - s * w * r, c - r * 0.7], [c + s * w * r, c], [c - s * w * r, c + r * 0.7]])], False,
                      255, th, cv2.LINE_AA)
    elif kind == "arrow_up":
        F([(-0.75, 0.55), (0.0, -0.7), (0.75, 0.55)])
    elif kind in ("checkbox", "checkbox_on"):
        R(-0.75, -0.75, 0.75, 0.75, th=t)
        if kind == "checkbox_on":
            cv2.polylines(m, [P([[c - r * 0.45, c], [c - r * 0.1, c + r * 0.38], [c + r * 0.5, c - r * 0.4]])], False,
                          255, t + 1, cv2.LINE_AA)
    elif kind in ("radio", "radio_on"):
        ci(c, c, r * 0.78, False)
        if kind == "radio_on":
            ci(c, c, r * 0.38)
    else:
        raise ValueError(kind)


def badge_masks(plate, inner, s):
    """(plate, inner) float coverage masks of one badge s px high (a pill is 1.7 s wide): a disc, rounded square,
    pill, burst, shield or hexagon with a pictogram (icon_mask) at its centre. No letters, digits, ! or ?."""
    import cv2
    n = max(6, int(round(s)))
    w = int(round(1.7 * n)) if plate == "pill" else n
    S = 4
    H_, W_ = n * S, w * S
    m = np.zeros((H_, W_), np.uint8)
    cx, cy, r = W_ / 2.0, H_ / 2.0, H_ / 2.0 * 0.95

    def P(pts):
        return np.round(np.asarray(pts, np.float64)).astype(np.int32)
    if plate == "disc":
        cv2.circle(m, (int(cx), int(cy)), int(r), 255, -1, cv2.LINE_AA)
    elif plate == "rsquare":
        k = int(r * 0.45)
        cv2.rectangle(m, (int(cx - r) + k, int(cy - r)), (int(cx + r) - k, int(cy + r)), 255, -1)
        cv2.rectangle(m, (int(cx - r), int(cy - r) + k), (int(cx + r), int(cy + r) - k), 255, -1)
        for x, y in ((cx - r + k, cy - r + k), (cx + r - k, cy - r + k), (cx - r + k, cy + r - k),
                     (cx + r - k, cy + r - k)):
            cv2.circle(m, (int(x), int(y)), k, 255, -1, cv2.LINE_AA)
    elif plate == "pill":
        cv2.rectangle(m, (int(r), int(cy - r)), (int(W_ - r), int(cy + r)), 255, -1)
        cv2.circle(m, (int(r), int(cy)), int(r), 255, -1, cv2.LINE_AA)
        cv2.circle(m, (int(W_ - r), int(cy)), int(r), 255, -1, cv2.LINE_AA)
    elif plate == "burst":
        pts = [[cx + (r if k % 2 == 0 else r * 0.78) * math.cos(math.pi * k / 8),
                cy + (r if k % 2 == 0 else r * 0.78) * math.sin(math.pi * k / 8)] for k in range(16)]
        cv2.fillPoly(m, [P(pts)], 255, cv2.LINE_AA)
    elif plate == "shield":
        cv2.fillPoly(m, [P([[cx - r * 0.85, cy - r], [cx + r * 0.85, cy - r], [cx + r * 0.85, cy + r * 0.05],
                            [cx, cy + r], [cx - r * 0.85, cy + r * 0.05]])], 255, cv2.LINE_AA)
    elif plate == "hex":
        cv2.fillPoly(m, [P([[cx + r * math.cos(math.pi * k / 3 + math.pi / 6), cy + r * math.sin(math.pi * k / 3 +
                                                                                                math.pi / 6)]
                            for k in range(6)])], 255, cv2.LINE_AA)
    else:
        raise ValueError(plate)
    pm = cv2.resize(m, (w, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    q = max(4, int(round(0.58 * n)))
    im = np.zeros_like(pm)
    y0, x0 = (n - q) // 2, (w - q) // 2
    im[y0:y0 + q, x0:x0 + q] = icon_mask(inner, q)
    return pm, im * pm


class Mark:
    """A background mark before it is placed: its paint layers [(mask, colour, alpha)] (one shape), the union's ink
    box (coverage over MARK_INK) in the masks, (glyph marks) the pen baseline oy in the masks, and its kind (meta
    "mark_kinds")."""

    def __init__(self, layers, oy=None, kind=None):
        self.layers, self.oy, self.kind = [(m, c, a) for m, c, a in layers if m is not None], oy, kind
        u = self.layers[0][0]
        for m_, _, _ in self.layers[1:]:
            u = np.maximum(u, m_)
        self.mh, self.mw = u.shape
        ys, xs = np.nonzero(u > MARK_INK)
        self.ok = bool(len(ys))
        if self.ok:
            self.ink = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            self.w, self.h = self.ink[2] - self.ink[0], self.ink[3] - self.ink[1]


def icon_mark(kind, s, colour):
    return Mark([(icon_mask(kind, s), colour, 1.0)], kind=f"icon:{kind}")


def badge_mark(rng, s, colour, inner_colour):
    """A badge (background): its plate in colour with a pictogram cut out of it or drawn in inner_colour."""
    pm, im = badge_masks(BADGE_PLATES[int(rng.integers(len(BADGE_PLATES)))],
                         BADGE_INNER[int(rng.integers(len(BADGE_INNER)))], s)
    if rng.random() < 0.5:
        return Mark([(pm * (1.0 - im), colour, 1.0)], kind="badge")
    return Mark([(pm, colour, 1.0), (im, inner_colour, 1.0)], kind="badge")


def item_layers(it, glow=True):
    """An Item's paint layers in put()'s order (glow, shadow, outer outline, outline, fill; one fill colour)."""
    st, m = it.style, it.masks
    out = []
    if glow and m["glow"] is not None:
        out.append((m["glow"], st["glow"]["c"], st["glow"]["a"]))
    if m["shadow"] is not None:
        out.append((m["shadow"], st["shadow"]["c"], st["shadow"]["a"]))
    if m["outer"] is not None:
        out.append((m["outer"], st["outer_c"], 1.0))
    if m["out"] is not None:
        out.append((m["out"], st["stroke_c"], 1.0))
    out.append((m["fill"], st["fill"], 1.0))
    return out


def glyph_mark(text, font, style):
    """Glyphs drawn as background (a leader run, an ornament glyph) in a text style without glow or gradient: a Mark
    with its pen baseline, or None."""
    st = {k: style[k] for k in ("fill", "stroke", "stroke_c", "shadow") if k in style}
    it = Item(text, font, st)
    if not hasattr(it, "oy"):
        return None
    mk = Mark(item_layers(it), oy=it.oy, kind="glyph")
    return mk if mk.ok else None


def place_mark(fr, mk, x_ink, cy=None, base=None):
    """Paints a Mark with its ink box's left edge at x_ink (frame px) and its ink centred on row cy, or (a glyph
    mark) its pen baseline at base; lists the ink box in fr.bg. Returns (index in fr.bg, box) or None (outside)."""
    X = int(round(x_ink)) - mk.ink[0]
    Y = int(round(base - mk.oy)) if base is not None else int(round(cy - (mk.ink[1] + mk.ink[3]) / 2.0))
    if X < 0 or Y < 0 or X + mk.mw > fr.W or Y + mk.mh > fr.H:
        return None
    for m, colour, a in mk.layers:
        paint(fr.img, X, Y, m, colour, a)
    box = [X + mk.ink[0], Y + mk.ink[1], X + mk.ink[2], Y + mk.ink[3]]
    fr.bg.append(box)
    if getattr(fr, "bg_kind", None) is not None:
        fr.bg_kind[len(fr.bg) - 1] = mk.kind or "mark"
    return len(fr.bg) - 1, box


def mark_gap(rng, h):
    """The ink gap of a mark to its label, px: touching (0-0.05 h) or near (0.1-0.5 h)."""
    P = MIX_UI2["mark_gap"]
    g = rng.uniform(*P["touch"]) if rng.random() < P["touch_p"] else rng.uniform(*P["near"])
    return int(round(g * h))


def side_mark(rng, h, colour, plate_luma=None):
    """An attached mark for a label of ink height h: an icon (PICTOS) or a badge (MIX_UI2["badge_p"])."""
    S2 = S3().S2
    if rng.random() < MIX_UI2["badge_p"]:
        s = h * rng.uniform(*MIX_UI2["badge_size"])
        bc = S2.accent(rng, plate_luma is None or plate_luma < 128)
        ic = S2.light_colour(rng, 0.1, 0.9) if S2.luma(bc) < 140 else S2.dark_colour(rng, 0.2)
        return badge_mark(rng, s, bc, ic)
    s = h * rng.uniform(*MIX_UI2["icon_size"])
    return icon_mark(PICTOS[int(rng.integers(len(PICTOS)))], s, colour)


class Group:
    """Items on one pen baseline drawn as ONE text item (put_group: one instance, one box): a label and its value
    apart (a one-box pair, its gap in GAP_PAIR x its height) or a run in two styles. w, h, up, down and text as an
    Item's; gaps: the ink gaps between the parts' ink boxes, px; pair: a one-box pair (its gap goes to fr.pair_gap,
    meta "pairs"), else a run."""

    def __init__(self, parts, gaps, sep=" ", pair=True):
        self.parts, self.gaps, self.pair = list(parts), [int(round(g)) for g in gaps], pair
        self.up = max(p.up for p in parts)
        self.down = max(p.down for p in parts)
        self.h = int(round(self.up + self.down))
        self.offs, x = [], 0
        for i, p in enumerate(parts):
            self.offs.append(x)
            x += p.w + (self.gaps[i] if i < len(self.gaps) else 0)
        self.w = x
        self.text = sep.join(p.text for p in parts)
        self.style = parts[0].style
        self.pair_gap = max(self.gaps) if self.gaps else 0
        self.ok = all(p.ok for p in parts)


def pair_group(rng, a, b):
    """A label and its value in one box, their ink gap GAP_PAIR x the pair's height (1b's convention)."""
    if a is None or b is None:
        return None
    h = max(a.up, b.up) + max(a.down, b.down)
    return Group([a, b], [max(1, round(rng.uniform(*MIX_UI2["pair_gap"]) * h))])


def run2_group(rng, a, b):
    """A run in two styles (the value in an accent colour or a bigger size): one box, a letter-spacing gap."""
    if a is None or b is None:
        return None
    h = max(a.up, b.up) + max(a.down, b.down)
    return Group([a, b], [max(1, round(rng.uniform(*MIX_UI2["run2_gap"]) * h))], sep="", pair=False)


def paint_item(fr, it, X, Y, layer):
    """One of put()'s paint layers of an Item at mask origin (X, Y)."""
    st, m = it.style, it.masks
    if layer == "glow" and m["glow"] is not None:
        paint(fr.img, X, Y, m["glow"], st["glow"]["c"], st["glow"]["a"])
    elif layer == "shadow" and m["shadow"] is not None:
        paint(fr.img, X, Y, m["shadow"], st["shadow"]["c"], st["shadow"]["a"])
    elif layer == "outer" and m["outer"] is not None:
        paint(fr.img, X, Y, m["outer"], st["outer_c"])
    elif layer == "out" and m["out"] is not None:
        paint(fr.img, X, Y, m["out"], st["stroke_c"])
    elif layer == "fill":
        rows = None
        if "grad" in st:
            t = np.clip((np.arange(it.h_mask, dtype=np.float32) - it.iy0) / max(1, it.h), 0, 1)[:, None]
            rows = st["grad"][0][None, :] * (1 - t) + st["grad"][1][None, :] * t
        paint(fr.img, X, Y, m["fill"], st["fill"], 1.0, rows)


def item_box(it, x_ink, base):
    """The ink box put() / put_group() gives an Item / a Group drawn at (x_ink, base)."""
    if isinstance(it, Group):
        xi, bi = int(round(x_ink)), int(round(base))
        ys = [(bi - int(round(p.oy)) + p.iy0, bi - int(round(p.oy)) + p.iy1) for p in it.parts]
        return [xi, min(a for a, _ in ys), xi + it.w, max(b for _, b in ys)]
    X, Y = int(round(x_ink)) - it.ix0, int(round(base - it.oy))
    return [X + it.ix0, Y + it.iy0, X + it.ix1, Y + it.iy1]


def put_group(fr, g, x_ink, base, role="ui", kind=None):
    """Draws a Group with its ink box's left edge at x_ink and its parts' pen baseline at base, all parts' layers in
    put()'s order, and adds ONE instance of role: the union of the parts' coverage. Returns its ink box (the
    instance's, as instance_boxes takes it) or None when it does not fit the frame."""
    xi, bi = int(round(x_ink)), int(round(base))
    pos = []
    for p, o in zip(g.parts, g.offs):
        X, Y = xi + o - p.ix0, bi - int(round(p.oy))
        if X < 0 or Y < 0 or X + p.w_mask > fr.W or Y + p.h_mask > fr.H:
            return None
        pos.append((X, Y))
    for layer in ("glow", "shadow", "outer", "out", "fill"):
        for p, (X, Y) in zip(g.parts, pos):
            paint_item(fr, p, X, Y, layer)
    x0, y0 = min(X for X, _ in pos), min(Y for _, Y in pos)
    x1 = max(X + p.w_mask for p, (X, _) in zip(g.parts, pos))
    y1 = max(Y + p.h_mask for p, (_, Y) in zip(g.parts, pos))
    cov = np.zeros((y1 - y0, x1 - x0), np.float32)
    for p, (X, Y) in zip(g.parts, pos):
        sl = cov[Y - y0:Y - y0 + p.h_mask, X - x0:X - x0 + p.w_mask]
        np.maximum(sl, p.cov, out=sl)
    n = len(fr.inst)
    if not fr.add(x0, y0, cov, 0, role):
        return None
    fr.item_info[n] = (g.text, kind or (fr.elements[-1].split(":")[0] if fr.elements else None))
    if g.pair and len(g.parts) > 1 and getattr(fr, "pair_gap", None) is not None:
        fr.pair_gap[n] = g.pair_gap
    x, y, c_, th = fr.inst[n][:4]
    return D1.TS._box(x, y, c_, th)


def put_any(fr, it, x_ink, base, kind):
    """put / put_group, and (version 2) the item's form (form()) in fr.item_form."""
    n = len(fr.inst)
    b = put_group(fr, it, x_ink, base, kind=kind) if isinstance(it, Group) else put(fr, it, x_ink, base, kind=kind)
    if b is not None and len(fr.inst) > n and getattr(fr, "item_form", None) is not None:
        fr.item_form[n] = getattr(it, "form", None)
    return b


def form(it, f):
    """Tags an Item / Group with its form (meta "forms"): run, run2, pair, single, label, value, option, title."""
    if it is not None:
        it.form = f
    return it


class Entry:
    """An item (Item or Group) with its attached marks: before / after = (Mark, ink gap px) or None."""

    def __init__(self, it, before=None, after=None):
        self.it, self.before, self.after = it, before, after

    @property
    def lead(self):
        return (self.before[0].w + self.before[1]) if self.before else 0

    @property
    def trail(self):
        return (self.after[0].w + self.after[1]) if self.after else 0

    @property
    def mh(self):
        return max([self.it.h] + [p[0].h for p in (self.before, self.after) if p is not None])


def draw_entry(fr, e, x, base, kind):
    """An Entry with the left edge of its extent (its before-mark's ink, else the item's) at x on pen baseline base:
    the marks first (centred on the item's ink box, at their gaps), then the item; each drawn mark of a drawn item
    goes to fr.attached. Returns the item's box or None."""
    x_ink = int(round(x)) + e.lead
    box = item_box(e.it, x_ink, base)
    placed = []
    for side, m in (("before", e.before), ("after", e.after)):
        if m is None:
            continue
        mk, gap = m
        mx = box[0] - gap - mk.w if side == "before" else box[2] + gap
        r = place_mark(fr, mk, mx, cy=(box[1] + box[3]) / 2.0)
        if r is not None:
            placed.append((r[0], side, gap))
    n = len(fr.inst)
    b = put_any(fr, e.it, x_ink, base, kind)
    if b is not None and len(fr.inst) > n and getattr(fr, "attached", None) is not None:
        for bi, side, gap in placed:
            fr.attached.append([bi, n, side, gap])
    return b


def leader_mark(rng, kind, span, h, colour, font=None, style=None, chars=None):
    """A leader (background) at most span px long for a row of ink height h: round or square dots, dashes, a rule, a
    double rule or a dotted rule (drawn, 4x supersampled), or a run of one leader glyph in the row's font and style
    (MIX_UI2["glyph_leaders"], those in chars: the font file's covered characters; none: a dotted rule). Returns
    (Mark, where: "mid" / "base" / "glyph") or None."""
    import cv2
    if span < 4:
        return None
    if kind == "glyph":
        cand = {c: w for c, w in MIX_UI2["glyph_leaders"].items() if chars is None or c in chars}
        if not cand:
            kind = "dotted"
    if kind == "glyph":
        ch = choose(rng, cand)
        adv = max(1.0, font.getlength(ch))
        k = int(span // adv)
        if k < 3:
            return None
        while k >= 3:
            mk = glyph_mark(ch * k, font, style)
            if mk is not None and mk.w <= span:
                mk.kind = f"leader:glyph:{ch}"
                return mk, "glyph"
            k -= 1
        kind = "rule"                                   # a glyph run too faint to draw (a thin ─): a rule
    S = 4
    where = "base" if kind in ("dots", "squares", "dotted") and rng.random() < 0.35 else "mid"
    if kind in ("dots", "squares"):
        rr = max(1.0, h * rng.uniform(0.05, 0.1))
        pitch = max(2 * rr + 2, h * rng.uniform(0.3, 0.6))
    elif kind == "dotted":
        rr = max(0.7, h * rng.uniform(0.025, 0.045))
        pitch = max(2 * rr + 1.5, h * rng.uniform(0.1, 0.18))
    else:
        rr = max(1.0, h * rng.uniform(0.03, 0.09)) / 2
    th = int(math.ceil(2 * rr)) + 2
    if kind == "double":
        th = int(math.ceil(2 * rr)) * 2 + max(2, int(round(h * 0.12))) + 2
    m = np.zeros((th * S, int(span) * S), np.uint8)
    cy = th * S / 2.0
    if kind in ("dots", "squares", "dotted"):
        k = int((span - 2 * rr) // pitch) + 1
        if k < 2:
            return None
        x0 = (span - (k - 1) * pitch) / 2.0
        for i in range(k):
            x = (x0 + i * pitch) * S
            if kind == "squares":
                cv2.rectangle(m, (int(x - rr * S), int(cy - rr * S)), (int(x + rr * S), int(cy + rr * S)), 255, -1)
            else:
                cv2.circle(m, (int(x), int(cy)), max(1, int(rr * S)), 255, -1, cv2.LINE_AA)
    elif kind == "dash":
        dl, dg = h * rng.uniform(0.2, 0.5), h * rng.uniform(0.12, 0.3)
        x = 0.0
        while x + dl <= span:
            cv2.rectangle(m, (int(x * S), int(cy - rr * S)), (int((x + dl) * S), int(cy + rr * S)), 255, -1)
            x += dl + dg
    elif kind == "rule":
        cv2.rectangle(m, (0, int(cy - rr * S)), (int(span * S) - 1, int(cy + rr * S)), 255, -1)
    elif kind == "double":
        sep = max(2, int(round(h * 0.12))) * S / 2.0
        for yy in (cy - sep, cy + sep):
            cv2.rectangle(m, (0, int(yy - rr * S)), (int(span * S) - 1, int(yy + rr * S)), 255, -1)
    else:
        raise ValueError(kind)
    mm = cv2.resize(m, (int(span), th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mk = Mark([(mm, colour, 1.0)], kind=f"leader:{kind}")
    return (mk, where) if mk.ok and mk.w <= span else None


def rule_mark(w, t, colour):
    """A plain horizontal rule (or, with w < t, a vertical one) w x t px: background."""
    m = np.ones((max(1, int(round(t))), max(1, int(round(w)))), np.float32)
    return Mark([(m, colour, 1.0)], kind="rule")


def value2(rng):
    """A stat value: v1's value_text, or a number with a unit (930人, 3,450円, 12個, 99%)."""
    if rng.random() < 0.45:
        return value_text(rng)
    n = int(rng.integers(0, 10 ** int(rng.integers(1, 5))))
    s = f"{n:,}" if n >= 1000 and rng.random() < 0.5 else str(n)
    if rng.random() < 0.75:
        s += UNITS[int(rng.integers(len(UNITS)))]
    return s


def stat_label(rng):
    return STAT_LABELS2[int(rng.integers(len(STAT_LABELS2)))]


def stat_spec(rng, w):
    """(kind, a, b) of a stat item: run (攻撃力20: one text), run2 (the value in another style, no gap), pair (a label
    and its value at GAP_PAIR, one box) or single (a top-bar token, a label or a value alone)."""
    k = choose(rng, w)
    if k == "single":
        t, v = pair_text(rng)
        return "single", (t if v is None or rng.random() < 0.6 else v), None
    return k, stat_label(rng), value2(rng)


def font_for2(rng, texts, em):
    """(font, FontFile, wght) as font_for, keeping the weight so a second size of the same face can be drawn."""
    f, wght = pick_ui_font(rng, set("".join(texts)))
    if f is None:
        return None, None, None
    return D1.get_font(f, em, wght), f, wght


def spec_item(rng, spec, font, st, f=None, wght=None, em=None):
    """An Item or Group of a stat spec (stat_spec), or None."""
    S2 = S3().S2
    k, a, b = spec
    if k == "single":
        return form(make_item(a, font, st), "single")
    if k == "run":
        return form(make_item(a + b, font, st), "run")
    ia = make_item(a, font, st)
    if ia is None:
        return None
    if k == "run2":
        vst = dict(st, fill=S2.accent(rng, S2.luma(st["fill"]) > 128)) if rng.random() < 0.6 else st
        vfont = font
        if f is not None and rng.random() < 0.5:
            vfont = D1.get_font(f, int(round(em * rng.uniform(1.1, 1.45))), wght)
        return form(run2_group(rng, ia, make_item(b, vfont, vst)), "run2")
    return form(pair_group(rng, ia, make_item(b, font, st)), "pair")


def entries_with_marks(rng, items, p, colour, plate_luma, before_p=0.75):
    """Entries of items, each (p) with a side_mark before it (before_p) or after it, at mark_gap."""
    out = []
    for it in items:
        e = Entry(it)
        if rng.random() < p:
            mk = side_mark(rng, it.h, colour, plate_luma)
            if mk.ok:
                if rng.random() < before_p:
                    e.before = (mk, mark_gap(rng, it.h))
                else:
                    e.after = (mk, mark_gap(rng, it.h))
        out.append(e)
    return out


def separator_mark(rng, h, colour):
    """A separator between two items on a row (background): a bar, a small diamond, a slash or a chevron."""
    import cv2
    k = choose(rng, MIX_UI2["separator"])
    if k in ("vbar", "diamond", "chev"):
        mk = (rule_mark(max(1, round(h * rng.uniform(0.05, 0.1))), h * rng.uniform(0.8, 1.2), colour) if k == "vbar"
              else icon_mark("diamond", h * rng.uniform(0.35, 0.6), colour) if k == "diamond"
              else icon_mark("chev_r", h * rng.uniform(0.5, 0.8), colour))
        mk.kind = f"sep:{k}"
        return mk
    n = max(4, int(round(h * rng.uniform(0.8, 1.1))))
    m = np.zeros((n * 4, n * 4), np.uint8)
    cv2.line(m, (int(n * 4 * 0.7), int(n * 4 * 0.05)), (int(n * 4 * 0.3), int(n * 4 * 0.95)), 255,
             max(1, int(round(n * 4 * 0.07))), cv2.LINE_AA)
    return Mark([(cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0, colour, 1.0)],
                kind="sep:slash")


def row_gaps(rng, entries, hmax, seps):
    """The ink gaps between consecutive entries' items on one row: at least GAP_BETWEEN x the row's tallest item
    (between()), and room for the marks attached between them and the separator (0.3 h clear of each side)."""
    out = []
    for i, (a, b) in enumerate(zip(entries, entries[1:])):
        s = seps[i]
        need = a.trail + b.lead + ((s.w + 0.6 * hmax) if s is not None else 0) + 0.2 * hmax
        out.append(int(math.ceil(max(between(hmax) + hmax * rng.uniform(0.0, 1.2), need))))
    return out


def lay_statbar(fr, rng):
    S2 = S3().S2
    n = int(rng.integers(2, 7))
    em = ui_em(rng, fr, choose(rng, {"12-24": 0.5, "24-48": 0.4, "48-96": 0.1}))
    specs = [stat_spec(rng, {"run": 0.45, "run2": 0.1, "pair": 0.2, "single": 0.25}) for _ in range(n)]
    font, f, wght = font_for2(rng, [t for s in specs for t in s[1:] if t], em)
    if font is None:
        return False
    plate = choose(rng, {"bar": 0.3, "window": 0.25, "panel": 0.25, "bare": 0.2})
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc) if plate != "bare" else None)
    items = [it for it in (spec_item(rng, s, font, st, f, wght, em) for s in specs) if it is not None]
    if len(items) < 2:
        return False
    hmax = max(it.h for it in items)
    ic = st["fill"] if rng.random() < 0.5 else S2.accent(rng, S2.luma(pc) < 128 or plate == "bare")
    entries = entries_with_marks(rng, items, rng.uniform(0.3, 0.9), ic, S2.luma(pc) if plate != "bare" else None)
    sep_p = rng.uniform(0.0, 0.8)
    sc = S2.light_colour(rng, 0.3, 0.6) if S2.luma(pc) < 128 else S2.dark_colour(rng, 0.4)
    nrows = 1 if len(entries) < 4 or rng.random() < 0.7 else int(rng.integers(2, 4))
    per = int(math.ceil(len(entries) / nrows))
    rows = [entries[i:i + per] for i in range(0, len(entries), per)]
    seps = [[separator_mark(rng, hmax, sc) if rng.random() < sep_p else None for _ in r[1:]] for r in rows]
    gaps = [row_gaps(rng, r, hmax, s) for r, s in zip(rows, seps)]
    widths = [sum(e.lead + e.it.w + e.trail for e in r) + sum(g) for r, g in zip(rows, gaps)]
    up = max(e.it.up for e in entries)
    down = max(e.it.down for e in entries)
    rh = max(up + down, max(e.mh for e in entries))
    pitch = rh * (1 + ROW_GAP) + 1 + hmax * rng.uniform(0.0, 0.5)
    padx, pady = hmax * rng.uniform(0.5, 1.5), hmax * rng.uniform(0.3, 0.8)
    W_ = 2 * padx + max(widths)
    H_ = 2 * pady + pitch * len(rows)
    gx = between(hmax)
    if plate == "bar":
        W_ = max(W_, min(fr.W - 2 * gx - 4, fr.W * rng.uniform(0.6, 1.0)))
    elif plate == "window":
        W_ = max(W_, min(fr.W - 2 * gx - 4, fr.W * rng.uniform(0.55, 0.92)))
        H_ = max(H_, min(fr.H * 0.4, fr.H * rng.uniform(0.12, 0.25)))
    if W_ + 2 * gx > fr.W:
        return False
    up_ = "bottom" if rng.random() < 0.7 else "upper"
    where = {"bar": "edge", "window": up_, "panel": "any", "bare": "edge"}[plate]
    xy = place_ui(fr, rng, W_, H_, where, items)
    if xy is None:
        return False
    x, y = xy
    if plate in ("bar", "window", "panel"):
        rad = {"bar": 0, "window": int(hmax * rng.uniform(0, 1.0)), "panel": int(hmax * rng.uniform(0, 0.6))}[plate]
        border = (max(1, int(hmax * 0.06)), S2.light_colour(rng, 0.4, 0.6)) if rng.random() < 0.4 else None
        D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa if plate != "window" else float(rng.uniform(0.4, 0.9)),
                    rad, border)
    align = choose(rng, {"left": 0.6, "centre": 0.3, "right": 0.1})
    added = 0
    top0 = y + (H_ - pitch * len(rows)) / 2
    for r_i, (row, gs, ss, w_) in enumerate(zip(rows, gaps, seps, widths)):
        cx = x + padx + {"left": 0.0, "centre": (W_ - 2 * padx - w_) / 2, "right": W_ - 2 * padx - w_}[align]
        base = top0 + r_i * pitch + (pitch + up - down) / 2
        boxes = []
        for i, e in enumerate(row):
            b = draw_entry(fr, e, cx, base, "statbar")
            added += b is not None
            boxes.append(b)
            cx += e.lead + e.it.w + e.trail + (gs[i] if i < len(gs) else 0)
        for i, s in enumerate(ss):
            a, b = boxes[i], boxes[i + 1]
            if s is None or a is None or b is None:
                continue
            l_ = a[2] + row[i].trail
            r_ = b[0] - row[i + 1].lead
            if r_ - l_ >= s.w + 0.6 * hmax:
                place_mark(fr, s, (l_ + r_ - s.w) / 2.0, cy=(a[1] + a[3]) / 2.0)
    fr.elements.append(f"statbar:em{em}:{plate}:{len(items)}x{len(rows)}:{f.cat}")
    return added > 0


def lay_ledger(fr, rng):
    S2 = S3().S2
    toc = rng.random() < 0.2
    n = int(rng.integers(3, 9))
    em = ui_em(rng, fr, choose(rng, {"12-24": 0.5, "24-48": 0.4, "48-96": 0.1}))
    kinds, pairs = [], []
    for i in range(n):
        k = "leader" if toc else choose(rng, {"leader": 0.45, "pair": 0.25, "run": 0.15, "far": 0.15})
        if toc:
            a = CHAPTERS[int(rng.integers(len(CHAPTERS)))] + (" " + short_text(rng, 2, 8) if rng.random() < 0.6
                                                              else "")
            b = str(int(rng.integers(1, 300)))
        else:
            a = stat_label(rng) if rng.random() < 0.7 else ITEMS2[int(rng.integers(len(ITEMS2)))]
            b = value2(rng)
        kinds.append(k)
        pairs.append((a, b))
    font, f, wght = font_for2(rng, [t for p in pairs for t in p], em)
    if font is None:
        return False
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc))
    rows = []
    for k, (a, b) in zip(kinds, pairs):
        if k == "run":
            rows.append((k, form(make_item(a + b, font, st), "run"), None))
        elif k == "pair":
            rows.append((k, form(pair_group(rng, make_item(a, font, st), make_item(b, font, st)), "pair"), None))
        else:
            rows.append((k, form(make_item(a, font, st), "label"), form(make_item(b, font, st), "value")))
    rows = [r for r in rows if r[1] is not None and (r[0] in ("run", "pair") or r[2] is not None)]
    if len(rows) < 2:
        return False
    hmax = max(max(r[1].h, r[2].h if r[2] is not None else 0) for r in rows)
    lkind = choose(rng, MIX_UI2["leader"])
    glyph = lkind == "glyph"
    g1, g2 = (rng.uniform(*MIX_UI2["glyph_leader_gap" if glyph else "leader_gap"]) for _ in range(2))
    lmin = MIX_UI2["glyph_leader_min" if glyph else "leader_min"] * hmax
    lc = st["fill"] if rng.random() < 0.6 else np.clip(0.55 * np.asarray(st["fill"], np.float32) +
                                                       0.45 * np.asarray(pc, np.float32), 0, 255)
    icons = rng.random() < 0.4
    same_icon = rng.random() < 0.4
    kind0 = PICTOS[int(rng.integers(len(PICTOS)))]
    ic = st["fill"] if rng.random() < 0.5 else S2.accent(rng, S2.luma(pc) < 128)
    igap = mark_gap(rng, hmax) if icons else 0
    marks = []
    for r in rows:
        if not icons:
            marks.append(None)
            continue
        if same_icon:
            mk = icon_mark(kind0, r[1].h * rng.uniform(*MIX_UI2["icon_size"]), ic)
        else:
            mk = side_mark(rng, r[1].h, ic, S2.luma(pc))
        marks.append(mk if mk.ok else None)
    lead = max([mk.w for mk in marks if mk is not None] + [0]) + (igap if icons else 0)
    two = [r for r in rows if r[0] in ("leader", "far")]
    lw = max([r[1].w for r in two] + [0])
    vw = max([r[2].w for r in two] + [0])
    pw = max([r[1].w for r in rows if r[0] in ("run", "pair")] + [0])
    mid = between(hmax)
    if any(r[0] == "leader" for r in rows):
        mid = max(mid, (g1 + g2) * hmax + lmin + 4)
    mid += hmax * rng.uniform(0.0, 3.0)
    inner = max(lw + mid + vw, pw)
    pad = hmax * rng.uniform(0.6, 1.4)
    rh = max([max(r[1].h, r[2].h if r[2] is not None else 0) for r in rows] +
             [mk.h for mk in marks if mk is not None])
    pitch = max(em * rng.uniform(1.3, 1.9), rh * (1 + ROW_GAP) + 1)
    W_, H_ = 2 * pad + lead + inner, 2 * pad + pitch * len(rows)
    if W_ + 2 * between(hmax) > fr.W:
        return False
    xy = place_ui(fr, rng, W_, H_, "side" if rng.random() < 0.5 else "any",
                  [r[1] for r in rows] + [r[2] for r in rows if r[2] is not None])
    if xy is None:
        return False
    x, y = xy
    D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa, int(em * rng.uniform(0, 0.6)),
                (max(1, int(em * 0.05)), S2.light_colour(rng, 0.4, 0.6)) if rng.random() < 0.5 else None)
    added = 0
    for i, ((k, a, b), mk) in enumerate(zip(rows, marks)):
        top = y + pad + i * pitch
        up = max(a.up, b.up if b is not None else 0)
        down = max(a.down, b.down if b is not None else 0)
        base = top + (pitch + up - down) / 2
        e = Entry(a, before=(mk, igap) if mk is not None else None)
        xl = x + pad + lead - e.lead
        ba = draw_entry(fr, e, xl, base, "ledger")
        added += ba is not None
        if b is None or ba is None:
            continue
        vx = x + W_ - pad - b.w
        bb = put_any(fr, b, vx, base, "ledger")
        added += bb is not None
        if bb is None or k != "leader":
            continue
        h = max(ba[3] - ba[1], bb[3] - bb[1])
        s0, s1 = ba[2] + int(math.ceil(g1 * h)), bb[0] - int(math.ceil(g2 * h))
        lm = leader_mark(rng, lkind, s1 - s0, h, lc, font, dict(st, fill=lc), f.chars)
        if lm is None:
            continue
        mk_, where = lm
        if where == "glyph":
            place_mark(fr, mk_, s0, base=base)
        else:
            cy = (ba[1] + ba[3]) / 2.0 if where == "mid" else ba[3] - mk_.h / 2.0
            place_mark(fr, mk_, s0 + (s1 - s0 - mk_.w) / 2.0 if lkind in ("dots", "squares", "dotted") else s0, cy=cy)
    fr.elements.append(f"ledger:em{em}:{'toc' if toc else 'stats'}:{lkind}{':icons' if icons else ''}:{f.cat}")
    return added > 0


def lay_iconlist(fr, rng):
    S2 = S3().S2
    horizontal = rng.random() < 0.3
    n = int(rng.integers(3, 8 if horizontal else 10))
    src = choose(rng, {"item": 0.4, "skill": 0.2, "menu": 0.25, "phrase": 0.15})
    counts = src in ("item", "skill") and rng.random() < 0.5
    cmode = choose(rng, {"run": 0.5, "column": 0.5}) if counts and not horizontal else ("run" if counts else None)
    texts, ctexts = [], []
    for _ in range(n):
        t = (ITEMS2[int(rng.integers(len(ITEMS2)))] if src == "item" else
             SKILLS[int(rng.integers(len(SKILLS)))] if src == "skill" else
             MENU_JP[int(rng.integers(len(MENU_JP)))] if src == "menu" else short_text(rng, 2, 10))
        c = ["×", "x", "×", "個"][int(rng.integers(4))]
        num = str(int(rng.integers(1, 100)))
        ctext = (c + num) if c != "個" else (num + c)
        if cmode == "run":
            t += ctext
        texts.append(t)
        ctexts.append(ctext)
    em = ui_em(rng, fr, choose(rng, {"12-24": 0.45, "24-48": 0.4, "48-96": 0.15}))
    font, f, wght = font_for2(rng, texts + ctexts, em)
    if font is None:
        return False
    on_plate = rng.random() < 0.75
    bars = not horizontal and rng.random() < 0.35
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc) if on_plate or bars else None)
    rows = [(form(make_item(t, font, st), "run" if cmode == "run" else "label"),
             form(make_item(c, font, st), "value") if cmode == "column" else None) for t, c in zip(texts, ctexts)]
    rows = [r for r in rows if r[0] is not None and (cmode != "column" or r[1] is not None)]
    if len(rows) < 2:
        return False
    hmax = max(max(a.h, b.h if b is not None else 0) for a, b in rows)
    ic = st["fill"] if rng.random() < 0.5 else S2.accent(rng, S2.luma(pc) < 128 or not on_plate)
    pl = S2.luma(pc) if on_plate or bars else None
    gb = mark_gap(rng, hmax)
    entries = []
    for a, _ in rows:
        e = Entry(a)
        side = choose(rng, {"before": 0.65, "after": 0.25, "both": 0.1})
        if side in ("before", "both"):
            mk = side_mark(rng, a.h, ic, pl)
            if mk.ok:
                e.before = (mk, gb if not horizontal else mark_gap(rng, a.h))
        if side in ("after", "both"):
            mk = side_mark(rng, a.h, ic, pl)
            if mk.ok:
                e.after = (mk, mark_gap(rng, a.h))
        entries.append(e)
    pad = hmax * rng.uniform(0.5, 1.2)
    if horizontal:
        gaps = row_gaps(rng, entries, hmax, [None] * len(entries))
        W_ = 2 * pad + sum(e.lead + e.it.w + e.trail for e in entries) + sum(gaps)
        while len(entries) > 2 and W_ + 2 * between(hmax) > fr.W:
            entries.pop()
            gaps = gaps[:len(entries) - 1]
            W_ = 2 * pad + sum(e.lead + e.it.w + e.trail for e in entries) + sum(gaps)
        rh = max(max(e.mh for e in entries), max(e.it.up for e in entries) + max(e.it.down for e in entries))
        H_ = 2 * pad + rh
    else:
        lead = max(e.lead for e in entries)
        body = max(e.it.w + e.trail for e in entries)
        cw = max([b.w for _, b in rows if b is not None] + [0])
        mid = (between(hmax) + hmax * rng.uniform(0.0, 2.5)) if cmode == "column" else 0
        W_ = 2 * pad + lead + body + mid + cw
        rh = max(e.mh for e in entries)
        pitch = max(em * rng.uniform(1.3, 2.0), rh * (1 + ROW_GAP) + 1)
        while len(entries) > 2 and pitch * len(entries) + 2 * pad > fr.H * 0.92:
            entries.pop()
        rows = rows[:len(entries)]
        H_ = 2 * pad + pitch * len(entries)
    if W_ + 2 * between(hmax) > fr.W:
        return False
    xy = place_ui(fr, rng, W_, H_, "edge" if horizontal else ("side" if rng.random() < 0.6 else "any"),
                  [e.it for e in entries])
    if xy is None:
        return False
    x, y = xy
    if on_plate:
        D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa, int(em * rng.uniform(0, 0.6)))
    added = 0
    if horizontal:
        up, down = max(e.it.up for e in entries), max(e.it.down for e in entries)
        base = y + (H_ + up - down) / 2
        cx = x + pad
        for i, e in enumerate(entries):
            added += draw_entry(fr, e, cx, base, "iconlist") is not None
            cx += e.lead + e.it.w + e.trail + (gaps[i] if i < len(gaps) else 0)
    else:
        sel = int(rng.integers(len(entries)))
        bc, ba_ = S2.accent(rng, True), float(rng.uniform(0.2, 0.5))
        for i, (e, (_, cb)) in enumerate(zip(entries, rows)):
            top = y + pad + i * pitch
            up = max(e.it.up, cb.up if cb is not None else 0)
            down = max(e.it.down, cb.down if cb is not None else 0)
            base = top + (pitch + up - down) / 2
            if bars or i == sel and rng.random() < 0.5:
                D1.draw_box(fr.img, (x + pad * 0.3, top + pitch * 0.06, x + W_ - pad * 0.3, top + pitch * 0.94),
                            bc if i == sel else pc, ba_ if i == sel else pa, int(pitch * 0.2))
            b = draw_entry(fr, e, x + pad + lead - e.lead, base, "iconlist")
            added += b is not None
            if cb is not None and b is not None:
                added += put_any(fr, cb, x + W_ - pad - cb.w, base, "iconlist") is not None
    fr.elements.append(f"iconlist:{src}:{'h' if horizontal else 'v'}:{cmode}:em{em}:{f.cat}")
    return added > 0


def lay_selector(fr, rng):
    S2 = S3().S2
    n = int(rng.integers(2, 7))
    labels = [CONFIG_LABELS[i] for i in rng.choice(len(CONFIG_LABELS), size=n, replace=False)]
    em = ui_em(rng, fr, choose(rng, {"12-24": 0.5, "24-48": 0.45, "48-96": 0.05}))
    ctl = []
    for _ in labels:
        k = choose(rng, {"arrows": 0.45, "check": 0.35, "slider": 0.2})
        vals = ([OPTION_VALUES[int(rng.integers(len(OPTION_VALUES)))]] if k == "arrows" else
                [OPTION_VALUES[i] for i in rng.choice(len(OPTION_VALUES), size=int(rng.integers(2, 4)), replace=False)]
                if k == "check" else [str(int(rng.integers(0, 101)))])
        ctl.append((k, vals))
    font, f, wght = font_for2(rng, labels + [v for _, vs in ctl for v in vs], em)
    if font is None:
        return False
    pc, pa = plate_colour(rng)
    st = ui_style(rng, em, S2.luma(pc))
    rows = []
    for lab, (k, vals) in zip(labels, ctl):
        a = form(make_item(lab, font, st), "label")
        vs = [form(make_item(v, font, st), "option" if k != "slider" else "value") for v in vals]
        if a is not None and all(v is not None for v in vs):
            rows.append((a, k, vs))
    if not rows:
        return False
    hmax = max(max([a.h] + [v.h for v in vs]) for a, _, vs in rows)
    ac = st["fill"] if rng.random() < 0.5 else S2.accent(rng, S2.luma(pc) < 128)
    arrows = ARROWS[int(rng.integers(len(ARROWS)))]
    asz = hmax * rng.uniform(*MIX_UI2["arrow_size"])
    agap = int(round(hmax * rng.uniform(*MIX_UI2["mark_gap"]["near"]))) if rng.random() < 0.85 else \
        int(round(hmax * rng.uniform(*MIX_UI2["mark_gap"]["touch"])))
    checks = CHECKS[int(rng.integers(len(CHECKS)))]
    csz = hmax * rng.uniform(0.6, 1.0)
    cgap = mark_gap(rng, hmax)
    built = []
    for a, k, vs in rows:
        if k == "arrows":
            la, ra = icon_mark(arrows[0], asz, ac), icon_mark(arrows[1], asz, ac)
            slot = max(v.w for v in vs) + int(hmax * rng.uniform(0.0, 0.4))
            built.append((a, k, [Entry(vs[0], (la, agap), (ra, agap))], slot))
        elif k == "check":
            sel = int(rng.integers(len(vs)))
            es = [Entry(v, (icon_mark(checks[1] if j == sel else checks[0], csz, ac), cgap)) for j, v in enumerate(vs)]
            built.append((a, k, es, 0))
        else:
            built.append((a, k, [Entry(vs[0])], 0))
    gap = between(hmax)
    for i, (a, k, es, slot) in enumerate(built):
        if k == "check":
            built[i] = (a, k, es, row_gaps(rng, es, hmax, [None] * len(es)))

    def ctl_w(k, es, extra):
        if k == "arrows":
            e = es[0]
            return e.before[0].w + e.before[1] + extra + e.after[1] + e.after[0].w
        if k == "check":
            return sum(e.lead + e.it.w for e in es) + sum(extra)
        return em * 8 + gap + es[0].it.w
    lw = max(a.w for a, _, _, _ in built)
    cw = max(ctl_w(k, es, ex) for _, k, es, ex in built)
    pad = em * rng.uniform(0.6, 1.2)
    mid = gap + hmax * rng.uniform(0.0, 2.0)
    rh = max(max(max([e.mh for e in es] + [a.h]) for a, _, es, _ in built), hmax)
    pitch = max(em * rng.uniform(1.5, 2.2), rh * (1 + ROW_GAP) + 1 + em * 0.3)
    W_, H_ = 2 * pad + lw + mid + cw, 2 * pad + pitch * len(built)
    if W_ + 2 * gap > fr.W:
        return False
    xy = place_ui(fr, rng, W_, H_, "centre" if rng.random() < 0.5 else "any",
                  [a for a, _, _, _ in built] + [e.it for _, _, es, _ in built for e in es])
    if xy is None:
        return False
    x, y = xy
    D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa, int(em * rng.uniform(0, 0.5)))
    added = 0
    sel_c = S2.accent(rng, True)
    for i, (a, k, es, ex) in enumerate(built):
        top = y + pad + i * pitch
        up = max([a.up] + [e.it.up for e in es])
        down = max([a.down] + [e.it.down for e in es])
        base = top + (pitch + up - down) / 2
        added += put_any(fr, a, x + pad, base, "selector") is not None
        rx = x + pad + lw + mid
        if k == "arrows":
            e = es[0]
            vx = rx + e.lead + (ex - e.it.w) / 2.0
            box = item_box(e.it, vx, base)
            r = []
            for side, (mk, g) in (("before", e.before), ("after", e.after)):
                mx = (rx if side == "before" else rx + e.lead + ex + g)
                q = place_mark(fr, mk, mx, cy=(box[1] + box[3]) / 2.0)
                if q is not None:
                    r.append((q[0], side, int(box[0] - q[1][2]) if side == "before" else int(q[1][0] - box[2])))
            nn = len(fr.inst)
            if put_any(fr, e.it, vx, base, "selector") is not None:
                added += 1
                fr.attached.extend([qi, nn, side, g] for qi, side, g in r)
        elif k == "check":
            cx = rx
            for j, e in enumerate(es):
                added += draw_entry(fr, e, cx, base, "selector") is not None
                cx += e.lead + e.it.w + (ex[j] if j < len(ex) else 0)
        else:
            slider(fr, rng, rx, top + pitch / 2, rx + em * 8, sel_c, em * 0.6)
            added += put_any(fr, es[0].it, rx + em * 8 + gap, base, "selector") is not None
    fr.elements.append(f"selector:em{em}:{len(built)}:{arrows[0]}:{checks[0]}:{f.cat}")
    return added > 0


def bracket_marks(fr, rng, box, h, colour, kind):
    """Graphic brackets (background) around a title's ink box: corner brackets (two arms at each corner) or tall
    square brackets (a bar and two serifs each side), 0.1-0.5 h from the box; each arm or bar is its own mark."""
    t = max(1, int(round(h * rng.uniform(0.04, 0.09))))
    gx = int(round(h * rng.uniform(0.1, 0.5)))
    if kind == "corner":
        gy = int(round(h * rng.uniform(0.1, 0.4)))
        arm = max(3, int(round(h * rng.uniform(0.35, 0.8))))
        x0, y0, x1, y1 = box[0] - gx - t, box[1] - gy - t, box[2] + gx + t, box[3] + gy + t
        segs = [(x0, y0, x0 + arm, y0 + t), (x0, y0 + t, x0 + t, y0 + arm), (x1 - arm, y0, x1, y0 + t),
                (x1 - t, y0 + t, x1, y0 + arm), (x0, y1 - t, x0 + arm, y1), (x0, y1 - arm, x0 + t, y1 - t),
                (x1 - arm, y1 - t, x1, y1), (x1 - t, y1 - arm, x1, y1 - t)]
    else:
        hh = h * rng.uniform(1.2, 1.8)
        cy = (box[1] + box[3]) / 2.0
        ya, yb = int(round(cy - hh / 2)), int(round(cy + hh / 2))
        ser = max(2, int(round(h * rng.uniform(0.15, 0.35))))
        xl, xr = box[0] - gx - t, box[2] + gx
        segs = [(xl, ya, xl + t, yb), (xl + t, ya, xl + t + ser, ya + t), (xl + t, yb - t, xl + t + ser, yb),
                (xr, ya, xr + t, yb), (xr - ser, ya, xr, ya + t), (xr - ser, yb - t, xr, yb)]
    out = []
    for a0, b0, a1, b1 in segs:
        if a1 > box[0] and a0 < box[2] and b1 > box[1] and b0 < box[3]:
            continue                                    # never over the title's own box
        mk = rule_mark(a1 - a0, b1 - b0, colour)
        mk.kind = f"bracket:{kind}"
        r = place_mark(fr, mk, a0, cy=(b0 + b1) / 2.0)
        if r is not None:
            out.append(r[1])
    return out


def lay_framed(fr, rng):
    S2 = S3().S2
    kind = choose(rng, {"marks": 0.35, "rules": 0.25, "brackets": 0.2, "text": 0.2})
    title = ORN_TITLES[int(rng.integers(len(ORN_TITLES)))]
    if kind == "text":
        o, c = TEXT_BRACKETS[int(rng.integers(len(TEXT_BRACKETS)))]
        title = o + title + c
    em = ui_em(rng, fr, choose(rng, {"24-48": 0.3, "48-96": 0.45, "96-160": 0.25}))
    font, f, wght = font_for2(rng, [title], em)
    if font is None:
        return False
    st = ui_style(rng, em, None)
    sp = em * rng.uniform(0.05, 0.3) if title.isascii() and rng.random() < 0.4 else 0.0
    tit = form(make_item(title, font, st, sp), "title")
    if tit is None:
        return False
    h = tit.h
    mc = st["fill"] if rng.random() < 0.5 else S2.accent(rng, True)
    left = right = None
    ext = 0.0
    if kind == "marks" or (kind == "text" and rng.random() < 0.3):
        k = BULLETS2[int(rng.integers(len(BULLETS2)))]
        s = h * rng.uniform(*MIX_UI2["orn_size"])
        g = mark_gap(rng, h) if rng.random() < 0.3 else int(round(h * rng.uniform(0.1, 0.5)))
        lk, rk = (("tri", "tri_l") if k == "tri" else (k, k))
        left, right = (icon_mark(lk, s, mc), g), (icon_mark(rk, s, mc), g)
    elif kind == "rules":
        ln_ = h * rng.uniform(1.0, 4.0)
        t = max(1, h * rng.uniform(0.04, 0.1))
        g = int(round(h * rng.uniform(0.1, 0.5)))
        if rng.random() < 0.25:
            lm = leader_mark(rng, "double", ln_, h, mc)
            mk = lm[0] if lm is not None else None
        else:
            mk = rule_mark(ln_, t, mc)
        if mk is None or not mk.ok:
            return False
        left, right = (mk, g), (mk, g)
    elif kind == "brackets":
        ext = h * 1.2
    e = Entry(tit, left, right)
    W_ = e.lead + tit.w + e.trail + 2 * ext
    H_ = max(e.mh, h * 1.9 if kind == "brackets" else 0) + em * 0.4
    if W_ + 2 * between(h) > fr.W:
        return False
    xy = place_ui(fr, rng, W_, H_, "upper" if rng.random() < 0.6 else "any", [tit])
    if xy is None:
        return False
    x, y = xy
    base = y + (H_ + tit.up - tit.down) / 2
    b = draw_entry(fr, e, x + ext, base, "framed")
    if b is None:
        return False
    if kind == "brackets":
        bracket_marks(fr, rng, b, h, mc, "corner" if rng.random() < 0.5 else "square")
    fr.elements.append(f"framed:{kind}:em{em}:{f.cat}")
    return True


def lay_pairs2(fr, rng):
    S2 = S3().S2
    n = int(rng.integers(1, 5))
    added = 0
    for _ in range(n):
        spec = stat_spec(rng, {"pair": 0.55, "run": 0.25, "run2": 0.2})
        em = ui_em(rng, fr, choose(rng, {"12-24": 0.4, "24-48": 0.45, "48-96": 0.15}))
        font, f, wght = font_for2(rng, [t for t in spec[1:] if t], em)
        if font is None:
            continue
        on_plate = rng.random() < 0.5
        pc, pa = plate_colour(rng)
        st = ui_style(rng, em, S2.luma(pc) if on_plate else None)
        it = spec_item(rng, spec, font, st, f, wght, em)
        if it is None:
            continue
        ic = S2.accent(rng, not on_plate or S2.luma(pc) < 128) if rng.random() < 0.6 else st["fill"]
        e = entries_with_marks(rng, [it], 0.7, ic, S2.luma(pc) if on_plate else None, 0.7)[0]
        padx, pady = it.h * rng.uniform(0.4, 1.0), it.h * rng.uniform(0.2, 0.5)
        W_ = 2 * padx + e.lead + it.w + e.trail
        H_ = max(it.up + it.down, e.mh) + 2 * pady
        xy = place_ui(fr, rng, W_, H_, "edge", [it])
        if xy is None:
            continue
        x, y = xy
        if on_plate:
            D1.draw_box(fr.img, (x, y, x + W_, y + H_), pc, pa, int(H_ * rng.uniform(0, 0.5)))
        base = y + (H_ + it.up - it.down) / 2
        if draw_entry(fr, e, x + padx, base, "pairs2") is not None:
            added += 1
        fr.elements.append(f"pairs2:{spec[0]}:em{em}{':mark' if (e.before or e.after) else ''}:{f.cat}")
    return added > 0


LAYOUTS2 = dict(LAYOUTS, statbar=lay_statbar, ledger=lay_ledger, iconlist=lay_iconlist, selector=lay_selector,
                framed=lay_framed, pairs2=lay_pairs2)


# ----------------------------------------------------------------------------- frames

def dialogue_logged(fr, rng, layout):
    """1c's dialogue element with the font files its pick_font returns logged in FONT_LOG (version 2's meta "fonts";
    1c's pick_font is put back afterwards, its draws unchanged)."""
    orig = D1.pick_font

    def logged(rng_, chars, cat=None):
        r = orig(rng_, chars, cat)
        if r[0] is not None:
            FONT_LOG.append(r[0])
        return r
    D1.pick_font = logged
    try:
        return D1.dialogue_element(fr, rng, layout)
    finally:
        D1.pick_font = orig


def compose_ui(rng, W, H):
    """A frame: 1c's scene, 1c's dialogue element in MIX_UI["dialogue"] of them, then 1-3 UI layouts (version 2:
    MIX_UI2's dialogue share and layouts, LAYOUTS2)."""
    fr = D1.new_frame(rng, W, H)
    fr.bg, fr.item_info = [], {}
    fr.empty = False
    mx, lays = (MIX_UI, LAYOUTS) if VERSION == 1 else (MIX_UI2, LAYOUTS2)
    if VERSION >= 2:
        fr.attached, fr.pair_gap, fr.bg_kind, fr.item_form = [], {}, {}, {}
        FONT_LOG.clear()
    if rng.random() < mx["dialogue"]:
        k = len(fr.rects)
        if VERSION >= 2:
            dialogue_logged(fr, rng, choose(rng, mx["dialogue_layout"]))
        else:
            D1.dialogue_element(fr, rng, choose(rng, mx["dialogue_layout"]))
        g = GAP_BETWEEN * 1.1 * fr.em            # its lines' height is at most about 1.1 em; its name tab sits above
        fr.rects[k:] = [(int(x0 - g), int(y0 - g), int(x1 + g), int(y1 + g)) for x0, y0, x1, y1 in fr.rects[k:]]
    n = 1 + int(rng.choice(len(mx["n_layouts"]), p=np.asarray(mx["n_layouts"]) / sum(mx["n_layouts"])))
    for _ in range(n):
        lays[choose(rng, mx["layout"])](fr, rng)
    return fr


def instance_boxes(fr):
    """(lines, ignore, info) in frame px: 1c's instance_boxes (an instance under MIN_ACROSS across an ignore area),
    and per line [text, kind] (None for 1c's dialogue element's lines, which keep no text)."""
    lines, ignore, info = [], [list(b) for b in fr.ignore], []
    for i, (x, y, cov, theta, v, role, _) in enumerate(fr.inst):
        b = D1.TS._box(x, y, cov, theta)
        if ((b[2] - b[0]) if v else (b[3] - b[1])) < D1.MIN_ACROSS:
            ignore.append(b)
            continue
        lines.append([b[0], b[1], b[2], b[3], v, role])
        t, k = fr.item_info.get(i, (None, "dialogue" if role in ("dialogue", "name") else None))
        info.append([t, k])
    return lines, ignore, info


def guard_ui(rgb, lines, ignore, bg):
    """1c's MeikiOCR guard, except that an area lying at least GUARD_BG_SHARE on the drawn marks (bg, grown by 2 px)
    is left out: the marks are background. Returns (areas added, areas left out)."""
    areas = D1.guard_ignore(rgb, lines, ignore)
    if not bg or not areas:
        return areas, 0
    H, W = rgb.shape[:2]
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in bg:
        m[max(0, y0 - 2):min(H, y1 + 2), max(0, x0 - 2):min(W, x1 + 2)] = True
    keep, left = [], 0
    for b in areas:
        a = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        if m[b[1]:b[3], b[0]:b[2]].sum() >= GUARD_BG_SHARE * a:
            left += 1
            continue
        keep.append(b)
    return keep, left


GUARD_ON = True


def inst_lines(fr):
    """{instance index: line index} of the instances instance_boxes keeps as lines."""
    out = {}
    for i, (x, y, cov, theta, v, role, _) in enumerate(fr.inst):
        b = D1.TS._box(x, y, cov, theta)
        if ((b[2] - b[0]) if v else (b[3] - b[1])) >= D1.MIN_ACROSS:
            out[i] = len(out)
    return out


def v2_meta(fr, line_of, bg_of):
    """meta "attached" ([mark index, line index, side, gap px]: a mark drawn beside its label, gap in render px),
    "pairs" ([line index, gap px]: a one-box pair's label / value ink gap) and "forms" (each line's form, None for
    v1 layouts and the dialogue element) with the indices of the sample's own lists (line_of / bg_of: render-time
    index to sample index, None when dropped)."""
    il = inst_lines(fr)
    att, pairs, forms = [], [], {}
    for n, li in il.items():
        k = line_of(li)
        if k is not None:
            forms[k] = fr.item_form.get(n)
    for bi, n, side, gap in fr.attached:
        li = line_of(il.get(n))
        mi = bg_of(bi)
        if li is not None and mi is not None:
            att.append([mi, li, side, int(gap)])
    for n, gap in sorted(fr.pair_gap.items()):
        li = line_of(il.get(n))
        if li is not None:
            pairs.append([li, int(gap)])
    return att, pairs, [forms.get(k) for k in range(len(forms))]


def sample_frame(rng):
    """One desk_ui frame (a sample dict and its grey canvas), or None for an unusable draw."""
    mx = mix()
    captured = rng.random() < mx["captured"]
    W, H = CAP.wh(choose(rng, mx["game" if captured else "screen"]))
    fr = compose_ui(rng, W, H)
    if not fr.inst:
        return None
    lines, ignore, info = instance_boxes(fr)
    if not lines:
        return None
    rgb = np.clip(fr.img + 0.5, 0, 255).astype(np.uint8)
    guard, left = guard_ui(rgb, lines, ignore, fr.bg) if GUARD_ON else ([], 0)
    ignore = ignore + guard
    bg = [list(b) for b in fr.bg]
    cap, tags = None, []

    def same(i):
        return i
    line_of = bg_of = same
    if captured:
        cap = CAP.sample(rng, fr.W, fr.H, None)
        rgb = cap.apply(rgb)
        if VERSION >= 2:
            lk = [i for i, ln in enumerate(lines) if cap.box(ln[:4]) is not None]
            bk = [i for i, b in enumerate(bg) if cap.box(b) is not None]
            lmap, bmap = {i: k for k, i in enumerate(lk)}, {i: k for k, i in enumerate(bk)}
            line_of, bg_of = lmap.get, bmap.get
        kept = [([*b, ln[4], ln[5]], inf) for ln, inf in zip(lines, info) for b in [cap.box(ln[:4])] if b is not None]
        lines, info = [k[0] for k in kept], [k[1] for k in kept]
        ignore = [b for b in (cap.box(x) for x in ignore) if b is not None]
        bg = [b for b in (cap.box(x) for x in bg) if b is not None]
        tags.append("captured")
    elif rng.random() < mx["native_jpeg"]["p"]:
        import cv2
        q = int(rng.integers(mx["native_jpeg"]["quality"][0], mx["native_jpeg"]["quality"][1] + 1))
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        rgb = np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])
        tags.append(f"jpeg{q}")
    Hh, Ww = rgb.shape[:2]
    meta = {"generator": f"ui_synth {VERSION}", "render": [fr.W, fr.H], "size": [Ww, Hh], "scene": fr.scene,
            "elements": fr.elements, "capture": cap.record() if cap else None, "tags": tags,
            "guard_ignore": len(guard), "guard_left_on_marks": left, "marks": bg, "items": info}
    if VERSION >= 2:
        meta["attached"], meta["pairs"], meta["forms"] = v2_meta(fr, (lambda i: None if i is None else line_of(i)),
                                                                 (lambda i: None if i is None else bg_of(i)))
        kinds = [fr.bg_kind.get(i, "v1") for i in range(len(fr.bg))]
        meta["mark_kinds"] = [kinds[i] for i in bk] if captured else kinds
        meta["fonts"] = sorted({(f.path.name, f.licence) for f in FONT_LOG})
    s = {"src": "synth:desktop-ui", "kind": "pc", "vn": None, "shot": None, "lines": lines, "ignore": ignore,
         "meta": meta}
    return s, SH.to_gray(rgb)


def extra_exclusion():
    """Version 2's sets on top of tsdesk_dense_source.exclusion_sets: the fresh native textbox set (eval/
    fresh_native_textbox_20261012.json's source manifest: its VNs and file sha256)."""
    p = ROOT / "eval" / "fresh_native_textbox_20261012.json"
    rows = json.loads(p.read_text(encoding="utf-8")).get("source_manifest") or []
    return {"fresh_native": {"vns": {r["vn"] for r in rows if r.get("vn")},
                             "sha256": {r[k] for r in rows for k in ("sha256", "pixel_sha256") if r.get(k)},
                             "shots": set(), "source": {p.name: SH.sha256_file(p)}}}


def exclude_scenes(hash_files=True, extra=None):
    """Cuts 1c's scene pool (textspot_data.SCENES, SCENE_ASPECT; in place) by VN, VNDB screenshot id and file sha256
    against every protected set (train/tsdesk_dense_source.exclusion_sets, and extra: more sets in its format).
    Returns the counts."""
    import tsdesk_dense_source as DSS
    TS_ = D1.TS
    ex = DSS.exclusion_sets()
    if extra:
        ex.update(extra)
    vns = set().union(*(e["vns"] for e in ex.values()))
    shas = set().union(*(e["sha256"] for e in ex.values()))
    shots = set().union(*(e["shots"] for e in ex.values()))
    dropped = Counter()

    def keep(p):
        name = Path(p).name
        if name.split("_")[0] in vns:
            dropped["vn"] += 1
            return False
        if DSS.shot_id(name) in shots:
            dropped["shot"] += 1
            return False
        if hash_files and SH.sha256_file(p) in shas:
            dropped["sha256"] += 1
            return False
        return True
    before = {k: len(v) for k, v in TS_.SCENES.items()}
    for k in TS_.SCENES:
        TS_.SCENES[k] = [p for p in TS_.SCENES[k] if keep(p)]
    TS_._aspects()
    return {"before": before, "after": {k: len(v) for k, v in TS_.SCENES.items()},
            "aspect_after": {k: len(v) for k, v in TS_.SCENE_ASPECT.items()}, "dropped": dict(dropped),
            "sets": {n: {"vns": len(e["vns"]), "sha256": len(e["sha256"]), "shots": len(e["shots"])}
                     for n, e in ex.items()}, "hash_files": hash_files}


def init(scenes=True, hash_files=True):
    """1c's resources (desktopdata/synth.init) and, with scenes, the scene pool cut by exclude_scenes (version 2: and
    by extra_exclusion)."""
    D1.init(scenes=scenes)
    return exclude_scenes(hash_files, extra_exclusion() if VERSION >= 2 else None) if scenes else None


def run_shard(job):
    idx, n, seed, out_dir = job
    path = Path(out_dir) / f"desk_ui_{idx:04d}"
    if path.with_suffix(".json").exists():
        return idx, 0, 0, 0.0
    rng = np.random.default_rng([seed, idx])
    t0, samples, greys, errors = time.time(), [], [], 0
    while len(samples) < n:
        try:
            r = sample_frame(rng)
        except Exception as e:  # noqa: BLE001 - a rare bad draw must not end a long render
            errors += 1
            if errors <= 3:
                import traceback
                print(f"shard {idx}: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            if errors > 200:
                raise
            continue
        if r is not None:
            samples.append(r[0])
            greys.append(r[1])
    SH.write_shard(path, samples, greys)
    return idx, len(samples), errors, time.time() - t0


def cmd_render(args):
    import cv2
    cv2.setNumThreads(1)
    global GUARD_ON
    GUARD_ON = not args.no_guard
    set_version(args.version)
    seed = args.seed if args.seed is not None else SEEDS[VERSION]
    rec = init(scenes=not args.no_scenes)
    if GUARD_ON:
        D1.guard_ignore(np.zeros((64, 64, 3), np.uint8), [], [])     # load MeikiOCR before the workers fork
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene_exclusion.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    jobs = [(i, args.per_shard, seed, str(out)) for i in range(args.first, args.first + args.shards)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, errors, dt in pool.imap_unordered(run_shard, jobs):
            print(f"desk ui shard {idx}: {n} frames ({errors} errors) in {dt:.0f}s ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"desk ui render: {len(jobs)} shards in {time.time() - t0:.0f} s", flush=True)


def cmd_manifest(args):
    set_version(getattr(args, "version", VERSION))
    d = Path(args.dir)
    shards = [SH.shard_record(js.with_suffix(""), d.name) for js in sorted(d.glob("desk_ui_*.json"))]
    gens = {s0["meta"].get("generator") for js in sorted(d.glob("desk_ui_*.json"))
            for s0 in json.loads(js.read_text(encoding="utf-8"))["samples"][:1]}
    if gens and gens != {f"ui_synth {VERSION}"}:
        raise SystemExit(f"{d}: shards drawn by {sorted(gens)}, not ui_synth {VERSION} (--version)")
    ex = d / "scene_exclusion.json"
    name = "desk_ui" if VERSION == 1 else f"desk_ui_v{VERSION}"
    doc = {"kind": "desktop-frames", "schema": 1, "issue": 173, "generator": "desktopdata/ui_synth.py",
           "generator_version": VERSION, "created": SH.utc(),
           "format": "1c's frames (desktopdata/synth.py); train/tsdesk_det.py reads them as an extra source "
                     f"(--extra {name}=MANIFEST:ROOT)",
           "mix": mix(), "gap_in": GAP_IN, "gap_between": GAP_BETWEEN, "row_gap": ROW_GAP,
           "guard": dict(D1.GUARD, bg_share=GUARD_BG_SHARE), "capture_mix": CAP.MIX,
           "seed": args.seed if args.seed is not None else SEEDS[VERSION],
           "fonts": "synth3's pool (data/manifests/synth3_fonts.json) without its pixel families, licence class open "
                    "only (pick_ui_font)",
           "scenes": "1c's scenes (textspot's clean pool, synth2's synthetic backgrounds), cut by VN, screenshot id "
                     "and sha256 against every protected set",
           "scene_exclusion": json.loads(ex.read_text()) if ex.exists() else None,
           "code_sha256": SH.code_sha256(ROOT, ["desktopdata/ui_synth.py", "desktopdata/synth.py",
                                               "desktopdata/capture.py", "desktopdata/shards.py",
                                               "synth/textspot_data.py", "synth/synth3.py"]),
           "shards": shards, "totals": SH.totals(shards),
           "note": "shards hold rendered frames over VNDB screenshot scenes and stay local; this manifest holds names, "
                   "counts, sha256"}
    if VERSION >= 2:
        doc.update(gap_pair=list(GAP_PAIR), mark_ink=MARK_INK,
                   meta_v2="meta attached: [mark index, line index, side, ink gap px]; meta pairs: [line index, "
                           "label / value ink gap px] (px of the render size); meta mark_kinds: each mark's kind "
                           "(icon:NAME, badge, leader:KIND, sep:KIND, rule, bracket:KIND, glyph; v1: v1's marks); meta "
                           "forms: each line's form (run, run2, pair, single, label, value, option, title; null: "
                           "v1 layouts and the dialogue element)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {doc['totals']['all']['samples'] if shards else 0} frames", flush=True)


SEEDS = {1: 20261102, 2: 20261104}       # each version's render seed (render / manifest --seed overrides)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--per-shard", type=int, default=250)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--seed", type=int, help="default: the version's (SEEDS)")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--version", type=int, default=1, choices=VERSIONS)
    p.add_argument("--no-scenes", action="store_true")
    p.add_argument("--no-guard", action="store_true", help="no MeikiOCR guard (tests: no teacher env)")
    p = sub.add_parser("manifest")
    p.add_argument("--dir", required=True)
    p.add_argument("--seed", type=int, help="default: the version's (SEEDS)")
    p.add_argument("--version", type=int, default=1, choices=VERSIONS)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"render": cmd_render, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

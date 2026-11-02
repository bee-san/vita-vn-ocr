#!/usr/bin/env python3
"""desk_ui: synthetic dense-UI desktop frames for the desktop detector (textspot-desktop round 3, #173).

  ui_synth.py render   --out DIR --shards N [--per-shard 250] [--first F] [--seed S] [--workers W]
  ui_synth.py manifest --dir DIR --out data/manifests/desktop_data/tsdesk_ui_train_v1.json

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
            return thin[int(rng.integers(len(thin)))], None
        bold = [x for x in files if x.weight and x.weight >= 600]
        if bold and rng.random() < MIX_UI["bold_p"]:
            return bold[int(rng.integers(len(bold)))], None
        normal = [x for x in files if not x.weight or x.weight > 300] or files
        f = normal[int(rng.integers(len(normal)))]
        wght = S3().S2.pick_weight(rng, f.path)
        if wght is not None and rng.random() < MIX_UI["bold_p"]:
            wght = max(wght, 700)
        return f, wght
    return None, None


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


# ----------------------------------------------------------------------------- frames

def compose_ui(rng, W, H):
    """A frame: 1c's scene, 1c's dialogue element in MIX_UI["dialogue"] of them, then 1-3 UI layouts."""
    fr = D1.new_frame(rng, W, H)
    fr.bg, fr.item_info = [], {}
    fr.empty = False
    if rng.random() < MIX_UI["dialogue"]:
        k = len(fr.rects)
        D1.dialogue_element(fr, rng, choose(rng, MIX_UI["dialogue_layout"]))
        g = GAP_BETWEEN * 1.1 * fr.em            # its lines' height is at most about 1.1 em; its name tab sits above
        fr.rects[k:] = [(int(x0 - g), int(y0 - g), int(x1 + g), int(y1 + g)) for x0, y0, x1, y1 in fr.rects[k:]]
    n = 1 + int(rng.choice(len(MIX_UI["n_layouts"]), p=np.asarray(MIX_UI["n_layouts"]) / sum(MIX_UI["n_layouts"])))
    for _ in range(n):
        LAYOUTS[choose(rng, MIX_UI["layout"])](fr, rng)
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


def sample_frame(rng):
    """One desk_ui frame (a sample dict and its grey canvas), or None for an unusable draw."""
    captured = rng.random() < MIX_UI["captured"]
    W, H = CAP.wh(choose(rng, MIX_UI["game" if captured else "screen"]))
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
    if captured:
        cap = CAP.sample(rng, fr.W, fr.H, None)
        rgb = cap.apply(rgb)
        kept = [([*b, ln[4], ln[5]], inf) for ln, inf in zip(lines, info) for b in [cap.box(ln[:4])] if b is not None]
        lines, info = [k[0] for k in kept], [k[1] for k in kept]
        ignore = [b for b in (cap.box(x) for x in ignore) if b is not None]
        bg = [b for b in (cap.box(x) for x in bg) if b is not None]
        tags.append("captured")
    elif rng.random() < MIX_UI["native_jpeg"]["p"]:
        import cv2
        q = int(rng.integers(MIX_UI["native_jpeg"]["quality"][0], MIX_UI["native_jpeg"]["quality"][1] + 1))
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        rgb = np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])
        tags.append(f"jpeg{q}")
    Hh, Ww = rgb.shape[:2]
    meta = {"generator": f"ui_synth {GEN_VERSION}", "render": [fr.W, fr.H], "size": [Ww, Hh], "scene": fr.scene,
            "elements": fr.elements, "capture": cap.record() if cap else None, "tags": tags,
            "guard_ignore": len(guard), "guard_left_on_marks": left, "marks": bg, "items": info}
    s = {"src": "synth:desktop-ui", "kind": "pc", "vn": None, "shot": None, "lines": lines, "ignore": ignore,
         "meta": meta}
    return s, SH.to_gray(rgb)


def exclude_scenes(hash_files=True):
    """Cuts 1c's scene pool (textspot_data.SCENES, SCENE_ASPECT; in place) by VN, VNDB screenshot id and file sha256
    against every protected set (train/tsdesk_dense_source.exclusion_sets). Returns the counts."""
    import tsdesk_dense_source as DSS
    TS_ = D1.TS
    ex = DSS.exclusion_sets()
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
    """1c's resources (desktopdata/synth.init) and, with scenes, the scene pool cut by exclude_scenes."""
    D1.init(scenes=scenes)
    return exclude_scenes(hash_files) if scenes else None


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
    rec = init(scenes=not args.no_scenes)
    if GUARD_ON:
        D1.guard_ignore(np.zeros((64, 64, 3), np.uint8), [], [])     # load MeikiOCR before the workers fork
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene_exclusion.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    jobs = [(i, args.per_shard, args.seed, str(out)) for i in range(args.first, args.first + args.shards)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, errors, dt in pool.imap_unordered(run_shard, jobs):
            print(f"desk ui shard {idx}: {n} frames ({errors} errors) in {dt:.0f}s ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"desk ui render: {len(jobs)} shards in {time.time() - t0:.0f} s", flush=True)


def cmd_manifest(args):
    d = Path(args.dir)
    shards = [SH.shard_record(js.with_suffix(""), d.name) for js in sorted(d.glob("desk_ui_*.json"))]
    ex = d / "scene_exclusion.json"
    doc = {"kind": "desktop-frames", "schema": 1, "issue": 173, "generator": "desktopdata/ui_synth.py",
           "generator_version": GEN_VERSION, "created": SH.utc(),
           "format": "1c's frames (desktopdata/synth.py); train/tsdesk_det.py reads them as an extra source "
                     "(--extra desk_ui=MANIFEST:ROOT)",
           "mix": MIX_UI, "gap_in": GAP_IN, "gap_between": GAP_BETWEEN, "row_gap": ROW_GAP,
           "guard": dict(D1.GUARD, bg_share=GUARD_BG_SHARE), "capture_mix": CAP.MIX, "seed": args.seed,
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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {doc['totals']['all']['samples'] if shards else 0} frames", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--per-shard", type=int, default=250)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--no-scenes", action="store_true")
    p.add_argument("--no-guard", action="store_true", help="no MeikiOCR guard (tests: no teacher env)")
    p = sub.add_parser("manifest")
    p.add_argument("--dir", required=True)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"render": cmd_render, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Training data of the full-frame text spotter (issue #104, docs/textspot.md): synthetic WHOLE frames with exact
ink boxes and text classes, in #97's shard format (synth/textdet_data.py).

  textspot_data.py synth --out DIR --shards N [--per-shard M] [--first F] [--seed S] [--workers W]
  textspot_data.py preview --dir DIR [--n 20] [--cols 4] [--seed S] --out PNG    contact sheet (<= 1280 px)
  textspot_data.py manifest --dir DIR --out data/manifests/textspot_shards_v1.json [--real-manifest JSON]
                   [--real-stats JSON]                   the pin: every shard's sha256 and counts, the generator
  textspot_data.py real-stats --root DIR ... --out JSON     #97's pinned real canvases, measured (report only)

A shard is DIR/<name>.npy (uint8 grey canvases, concatenated row-major) and DIR/<name>.json (#97's format):
  {"samples": [{"off", "h", "w", "src", "kind", "vn", "shot", "lines": [[x0, y0, x1, y1, vertical, role]],
                "ignore": [[x0, y0, x1, y1]], "meta"}]}
src is "synth:<frame kind>" (synth:vita, synth:psp, synth:adrenaline, synth:pc, synth:empty for a text-free frame);
kind is the canvas's platform target as #97's real canvases have it (psp for a 480 x 272 frame, else vita); vn and
shot are null. role is the protocol's class: dialogue, name, choice, ui or scene (eval/textspot_protocol.json
"labels"). Boxes are [x0, x1) x [y0, y1) in canvas pixels.

Frames (MIX "kind"), each a whole screen at the size the spotter reads:
  vita        960 x 544, the Vita's elements over a Vita, a PC or (upscaled) a PSP scene
  psp         480 x 272, the PSP's elements over a PSP (or a scaled-down) scene; 16-bit dither
  adrenaline  a PSP frame rendered at 480 x 272, then scaled 2x bilinear (cv2.INTER_LINEAR) to 960 x 544
  pc          rendered at a PC size (1280 x 720, 800 x 600, 1024 x 768) over a PC scene of that aspect, then scaled to
              fit 960 x 544 with its aspect (cv2.INTER_AREA): 960 x 540 or 725 x 544 (teacher/geometry.target_scale).
              Its synth3 regions keep the Vita geometry (box widths drawn from 960 px, pads and outlines in Vita px)
              at synth3 version 3's PC em (its final em at the Vita target over the fit scale), without bitmap fonts
              and with version 3's "scaled" outlines, shadows and box opacities
  empty       a text-free frame of any of the four: the scene alone, the scene with an empty dialogue box, window,
              choice stack or menu (render_region's region with its text inpainted away, as #97's `empty` kind), or
              a menu screen of plates and icons without text
Every geometric step (the 2x, the fit) acts on the whole frame and on every instance's own ink mask; an instance's
box is re-derived from its scaled mask (>= 0.5). JPEG chains, dither, blur and noise act on the frame only.

A frame with text holds what a game shows at once, over a full-screen scene (below):
  dialogue   synth3's render_region (version 3) with the layout forced: adv / band at the bottom, an nvl page over
             most of the screen, a balloon or bare text anywhere. Its lines are class dialogue (narration and an
             inline name+dialogue line too), its name plate or name line class name. The region's own background is
             the frame under it, and its capture is switched off (the frame is captured once, at the end)
  choice     render_region's choice layout: a stack of 2-4 choice bars, class choice, often without dialogue
  menu       render_region's ui layout: a list of 2-5 menu bars, class ui
  ui         0-8 small labels at the screen's edges and corners, class ui, one instance per word or label (a gap
             of 1.6 em or more between them): quick-menu bars (SAVE LOAD AUTO SKIP LOG CONFIG, セーブ ロード ...),
             system words, dates and times, chapter titles, save slots, page and status labels, and button hints
             whose button glyph (a drawn icon, never text) is not part of the box
  scene      text inside the picture: a sign, a poster, a note, a phone screen; class scene; horizontal or vertical
  vertical   vertical ADV columns (right to left) or manga balloons (Vita only), the column typesetting of
             synth/vertical_render.py (Pillow's raqm layout, direction ttb); one instance per column, class dialogue
Elements never overlap: each reserves its rectangle (plus a margin) and the next one is placed in free space.

Boxes and ignore areas: an instance is one line (or one column) and its box is its MAIN-TEXT ink box from the render
masks (fill, outline, hard shadow): the instance's ink coverage thresholded at 0.5 (synth3's ink rule), or at half
its peak when the peak is under 1 (a thin face at a small size never reaches 0.5; instance_theta). For
render_region's lines the coverage is synth2's compose_text ink (recorded during the render: the region's text, and
the name tab's own ink for the tab's name, so a tab never lends its ink to the first line) over
textdet_data.main_box's band and columns, less the furigana and any page cursor's ink; for the other elements the
instance's own masks. A region whose lines' boxes overlap by more than a shared row or two (a tab drawn over the first
line), or whose text ink outside its boxes and ignore areas exceeds UNLABELLED_MAX px, is drawn again; smaller
leftovers become ignore areas. Ignore areas: the furigana (connected parts of the ruby mask, grown by the outline and
shadow), the faded tail of a typewriter line, instances too faint to label (peak coverage under LABEL_MIN), and every
instance under 6 px across at the frame's size (the protocol ignores such text).

Scenes: textgate's clean scene pool (eval/textgate_data.clean_scene_pool's rule: synth3's scene pool, so no test,
development-sample or CLANNAD VN, without the VNs of eval/linefind's four detector-built sets, any screenshot
vita-test, psp-test or those six linefind manifests list, and CHAOS;CHILD v14018), cut further (scene_pool): no
screenshot any eval/manifests or eval/linefind/manifests list holds (the vertical sets, linefind's candidate lists and
psp_pool included; never a textspot set), no VN of any eval/linefind list, no scene of
data/manifests/vertical_exclude_scenes.json. A scene is cover-cropped to the frame (zoom 1-1.25, flipped at random)
and its known text is inpainted away first (cv2.INPAINT_TELEA): synth3's scene text boxes (realdata's line boxes,
linefind's detections) and the teacher's training reads of the screenshot (MeikiOCR's lines with their furigana,
Screen AI's boxes; never a test read). Some frames get synth2's synthetic background instead.

Deterministic: a shard's frames come from numpy's default_rng([seed, shard]); workers fork after init.
"""
import argparse
import contextlib
import hashlib
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_data as TDD  # noqa: E402
from common import to_gray  # noqa: E402

GEN_VERSION = 1
CLASSES = ("dialogue", "name", "choice", "ui", "scene")
MIN_ACROSS = 6                  # px across the line at the frame's size; smaller text is an ignore area
LABEL_MIN = 0.3                 # an instance whose peak ink coverage is lower is an ignore area (instance_theta)
FAINT = 0.1                     # coverage that counts as drawn, for ignore areas and the unlabelled-ink check
UNLABELLED_MAX = 40             # a region whose text ink outside its boxes and ignore areas exceeds this is redrawn;
                                # less is made ignore areas (region_element)
VITA, PSP = (960, 544), (480, 272)
FIT = VITA                      # PC frames are scaled to fit this with their aspect
URL_ID = re.compile(r"/(\d+)\.(?:jpg|png)$")

# Every probability of the generator, recorded in the shard manifest.
MIX = {
    "kind": {"vita": 0.35, "psp": 0.25, "adrenaline": 0.15, "pc": 0.10, "empty": 0.15},
    "empty_size": {"vita": 0.41, "psp": 0.29, "adrenaline": 0.18, "pc": 0.12},     # the four kinds' shares
    "empty_mode": {"scene": 0.35, "box": 0.45, "menu": 0.20},
    "empty_box_layout": {"adv": 0.5, "nvl": 0.15, "band": 0.1, "balloon": 0.05, "choice": 0.12, "ui": 0.08},
    "pc_size": {"1280x720": 0.4, "800x600": 0.4, "1024x768": 0.2},
    "main": {"horizontal": 0.80, "vertical": 0.07, "none": 0.13},
    # synth3 version 3's layout weights (adv .50, band .12, nvl .09, balloon .04, bare .12), nvl a little higher:
    # a whole page of text is a frame kind of its own
    "layout": {"adv": 0.55, "band": 0.12, "nvl": 0.12, "balloon": 0.05, "bare": 0.16},
    "vertical_manga": 0.4,              # of the vertical main elements on Vita-size frames; the rest ADV columns
    "choice": {"with_main": 0.07, "without_main": 0.55},
    "menu": {"with_main": 0.03, "without_main": 0.35},
    "ui_n": [0.25, 0.2, 0.16, 0.12, 0.09, 0.07, 0.05, 0.035, 0.025],    # P(0, 1, ..., 8 UI elements)
    "ui": {"quick": 0.22, "word": 0.18, "hint": 0.13, "date": 0.12, "chapter": 0.08, "slot": 0.07,
           "status": 0.12, "place": 0.08},
    "ui_plate": 0.55,
    "scene_text": 0.18,
    "scene_text_n": [1, 2],
    "scene_text_vertical": 0.3,
    "scene_text_kind": {"sign": 0.4, "poster": 0.25, "note": 0.2, "phone": 0.15},
    "vertical_extra": 0.05,
    "scenes": {"vita": {"vita": 0.30, "hires": 0.60, "psp": 0.05, "synthetic": 0.05},
               "psp": {"psp": 0.75, "hires": 0.12, "vita": 0.08, "synthetic": 0.05},
               "pc": {"hires": 0.95, "synthetic": 0.05}},
    "scene_zoom": [1.0, 1.25],
    "scene_max_text": 0.4,             # a scene whose known text covers more of the frame is drawn again
    "scene_text_pad": [0.15, 2],        # known text boxes grown by max(2 px, 0.15 x their height) before inpainting
}
# The frame's capture: synth3 version 4's native calibration (data/manifests/synth3_v4_native_cfg.json "frames":
# console JPEG then VNDB's Q90, the grid on the frame), and version 3's PC path (VNDB's Q90 at the source resolution,
# then the area downscale)
CAPTURE = {
    "vita": {"chain": {"clean": 0.25, "vndb2": 0.75}, "q420": 0.94},
    "psp": {"chain": {"clean": 0.25, "q90": 0.63, "vndb2": 0.12}, "q420": {"q90": 0.0, "vndb2": 1.0}},
    "adrenaline": {"chain": {"clean": 0.25, "vndb2": 0.75}, "q420": 0.94},
    "pc": {"chain": {"clean": 0.15, "q90": 0.85}, "q420": 0.85},
    "console_q": [82, 95], "dither565": 0.45, "blur": 0.15, "noise": 0.15,
}
ROLE = {"name": "name", "choice": "choice", "ui": "ui", "dialogue": "dialogue", "narration": "dialogue",
        "name+dialogue": "dialogue"}
PLACE_OF = {"adv": "bottom", "band": "bottom", "nvl": "centre", "balloon": "any", "bare": "any",
            "choice": "centre", "ui": "side"}

QUICK_LATIN = ["Q.SAVE", "Q.LOAD", "SAVE", "LOAD", "AUTO", "SKIP", "LOG", "CONFIG", "SYSTEM", "BACK", "MENU",
               "HIDE", "TITLE", "EXIT", "VOICE", "OPTION", "BACKLOG", "CLOSE"]
QUICK_JP = ["セーブ", "ロード", "オート", "スキップ", "ログ", "設定", "システム", "閉じる", "戻る", "Q.セーブ",
            "Q.ロード", "バックログ", "メニュー", "タイトル", "既読スキップ", "ボイス", "コンフィグ", "環境設定"]
HINTS = [("circle", "決定"), ("cross", "戻る"), ("cross", "キャンセル"), ("triangle", "メニュー"),
         ("square", "ログ"), ("triangle", "システム"), ("circle", "選択"), ("square", "オート"),
         ("circle", "OK"), ("cross", "BACK"), ("start", "ポーズ"), ("select", "スキップ"), ("lr", "ページ送り"),
         ("triangle", "ウィンドウ消去"), ("circle", "次へ"), ("square", "バックログ")]
STATUS = ["NEW", "SKIP", "AUTO", "Now Loading...", "Loading", "体験版", "好感度", "所持金", "Lv.{n}", "HP {n}/{m}",
          "{n}/{m}", "Page {n}", "No.{nn}", "DAY {n}", "{n}日目", "第{n}話", "残り{n}回", "既読", "未読", "CLEAR",
          "COMPLETE", "BGM", "VOICE", "SE", "CG {n}/{m}", "Chapter {n}", "Tips", "TIPS", "MAP", "STATUS", "ITEM"]
SIGNS = ["営業中", "本日休業", "立入禁止", "非常口", "出口", "入口", "注意", "危険", "準備中", "受付", "案内所",
         "駐輪場", "禁煙", "関係者以外立入禁止", "文化祭", "部員募集中", "生徒会室", "職員室", "図書室", "保健室",
         "喫茶", "商店街", "駅前", "改札", "のりば", "観光案内", "OPEN", "CLOSED", "CAFE", "BAR", "HOTEL", "EXIT",
         "STAFF ONLY", "WELCOME", "SALE"]
SIGN_SUFFIX = ["店", "駅", "学園", "高校", "神社", "公園", "病院", "食堂", "書店", "商店", "ホテル", "銀行", "寮"]
DIGITS = "0123456789"
SPACES = {" ", "　"}

S3 = None           # synth3, after init()
VR = None           # synth/vertical_render.py, after init()
SCENES = {"vita": [], "psp": [], "hires": []}
SCENE_ASPECT = {"16:9": [], "4:3": []}
SCENE_TEXT = {}     # scene path -> known text boxes, float32 N x 4, in the screenshot's own pixels
POOL = {}           # the pool's counts and identity (manifest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose(rng, weights):
    return TDD.choose(rng, weights)


# ----------------------------------------------------------------------------- scenes

def _rows(path):
    for x in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(x)
        except json.JSONDecodeError:
            continue


def eval_screenshots():
    """(image ids, file sha256, VNs) to keep out of the scene pool: every screenshot any eval/manifests or
    eval/linefind/manifests jsonl lists (the textspot sets are never opened: their VNs are reserved, so synth3's pool
    holds none of them), and every VN of an eval/linefind list."""
    ids, shas, vns = set(), set(), set()
    files = sorted((ROOT / "eval" / "manifests").glob("*.jsonl")) + \
        sorted((ROOT / "eval" / "linefind" / "manifests").glob("*.jsonl"))
    for f in files:
        if f.name.startswith("textspot"):
            continue
        linefind = f.parent.parent.name == "linefind"
        for r in _rows(f):
            if r.get("sha256"):
                shas.add(r["sha256"])
            m = URL_ID.search(r.get("url") or "")
            if m:
                ids.add(m.group(1))
            if linefind and r.get("vn"):
                vns.add(r["vn"])
    return ids, shas, vns


def scene_pool(hash_files=True):
    """Cuts synth3's scene pool (in place) to textgate's clean pool and then further (module docstring). Returns the
    counts and the pool's identity: {"synth3", "textgate", "pool", "sha256", "files"}."""
    R = S3.R
    synth3_counts = {k: len(v) for k, v in R.bg.items()}
    ids, shas, lf_vns = TDD._eval_exclusions()

    def tg_keep(p):         # eval/textgate_data.clean_scene_pool's rule
        name = Path(p).name
        vn = name.split("_")[0]
        sid = name.rsplit("_", 1)[-1].split(".")[0]
        return vn not in lf_vns and vn not in TDD.REPORT_ONLY_VNS and sid not in ids
    tg = {k: [p for p in v if tg_keep(p)] for k, v in R.bg.items()}
    tg_counts = {k: len(v) for k, v in tg.items()}
    ids2, shas2, vns2 = eval_screenshots()
    vex = set(json.loads((ROOT / "data" / "manifests" / "vertical_exclude_scenes.json").read_text())["shots"])
    files = {}
    dropped = {"vn": 0, "image_id": 0, "vertical_scene": 0, "sha256": 0}

    def keep(p):
        name = Path(p).name
        vn = name.split("_")[0]
        sid = name.rsplit("_", 1)[-1].split(".")[0]
        if vn in vns2:
            dropped["vn"] += 1
            return False
        if sid in ids2:
            dropped["image_id"] += 1
            return False
        if Path(p).stem in vex:
            dropped["vertical_scene"] += 1
            return False
        if hash_files:
            files[str(p)] = sha256(p)
            if files[str(p)] in shas2:
                dropped["sha256"] += 1
                del files[str(p)]
                return False
        return True
    R.bg = {k: [p for p in v if keep(p)] for k, v in tg.items()}
    R.bg_vns = sorted({Path(p).name.split("_")[0] for v in R.bg.values() for p in v})
    counts = {k: len(v) for k, v in R.bg.items()}
    ident = hashlib.sha256("".join(f"{Path(p).name} {files.get(str(p), '')}\n"
                                   for k in sorted(R.bg) for p in sorted(map(str, R.bg[k]))).encode()).hexdigest()
    return {"synth3": synth3_counts, "textgate": tg_counts, "pool": counts, "dropped_further": dropped,
            "vns": len(R.bg_vns), "sha256": ident,
            "sha256_rule": "sha256 of the lines '<file name> <file sha256>' of every pool scene, sorted by kind, path"}


def teacher_text_boxes(stems):
    """{shot: (dims, [box])}: every text box the teacher's TRAINING reads hold for those screenshots (MeikiOCR lines:
    det, ink, parts, furigana; Screen AI boxes), in the read's pixels. Never the psp-test reads."""
    import glob
    out = {}
    for p in sorted(glob.glob(str(TDD.ART / "teacher" / "reads" / "train.*of5.jsonl"))):
        for r in _rows(p):
            if r.get("shot") not in stems:
                continue
            boxes = []
            for ln in r.get("lines") or []:
                for k in ("det", "ink", "crop"):
                    if ln.get(k):
                        boxes.append(ln[k])
                boxes += [b for b in (ln.get("parts") or []) if b]
                boxes += [b for b in (ln.get("furi") or []) if b]
            boxes += [s["box"] for s in r.get("sai") or [] if s.get("box")]
            out[r["shot"]] = (tuple(r.get("dims") or ()), boxes)
    return out


def load_scene_text():
    """SCENE_TEXT for every pool scene: synth3's scene text boxes plus the teacher's, in the scene's own pixels."""
    from PIL import Image
    paths = [Path(p) for v in S3.R.bg.values() for p in v]
    teach = teacher_text_boxes({p.stem for p in paths})
    SCENE_TEXT.clear()
    for p in paths:
        boxes = [np.asarray(S3.R.bg_text.get(p, np.zeros((0, 4), np.float32)), np.float32).reshape(-1, 4)]
        t = teach.get(p.stem)
        if t and t[1]:
            with Image.open(p) as im:
                sw, sh = im.size
            dims, tb = t
            kx, ky = (sw / dims[0], sh / dims[1]) if len(dims) == 2 and dims[0] and dims[1] else (1.0, 1.0)
            b = np.asarray(tb, np.float32).reshape(-1, 4) * np.array([kx, ky, kx, ky], np.float32)
            boxes.append(b)
        SCENE_TEXT[str(p)] = np.concatenate(boxes)


def init(scenes=True, assets=None, font_manifest=None, hash_files=True):
    """synth3's resources (its `personal` profile, as #97 and VOCR5 used), the scene pool and its text boxes, and
    synth/vertical_render.py's typesetting. scenes=False (tests): no screenshots, synthetic backgrounds only."""
    global S3, VR
    import synth3
    cv2.setNumThreads(1)
    kw = {"scenes": scenes}
    if assets is not None:
        kw.update(assets=assets, font_manifest=font_manifest)
    synth3.init(**kw)                   # textdet_data.init_synth's call, with scenes
    TDD.S3 = S3 = synth3
    import vertical_render
    vertical_render.S3 = S3
    VR = vertical_render
    POOL.clear()
    if scenes and any(S3.R.bg.values()):
        POOL.update(scene_pool(hash_files))
        load_scene_text()
    else:
        S3.R.bg = {"vita": [], "psp": [], "hires": []}
        POOL.update({"pool": {"vita": 0, "psp": 0, "hires": 0}, "sha256": None})
    for k in SCENES:
        SCENES[k] = [str(p) for p in S3.R.bg.get(k, [])]
    _aspects()
    return S3


def _aspects():
    from PIL import Image
    for k in SCENE_ASPECT:
        SCENE_ASPECT[k] = []
    for p in SCENES["hires"]:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except OSError:
            continue
        a = w / h
        if abs(a - 16 / 9) < 0.06:
            SCENE_ASPECT["16:9"].append(p)
        elif abs(a - 4 / 3) < 0.06:
            SCENE_ASPECT["4:3"].append(p)


def synthetic_background(rng, w, h):
    saved = S3.S2.R
    S3.S2.R = S3._NoScenes
    try:
        return np.asarray(S3.S2.background(rng, w, h, "vita"), np.float32)
    finally:
        S3.S2.R = saved


def text_mask(path, sw, sh):
    """uint8 mask (255 = known text, grown) of a scene in its own pixels."""
    m = np.zeros((sh, sw), np.uint8)
    pad_k, pad_min = MIX["scene_text_pad"]
    for x0, y0, x1, y1 in SCENE_TEXT.get(str(path), np.zeros((0, 4), np.float32)):
        p = max(pad_min, pad_k * min(y1 - y0, x1 - x0))
        a0, b0 = max(0, int(math.floor(x0 - p))), max(0, int(math.floor(y0 - p)))
        a1, b1 = min(sw, int(math.ceil(x1 + p))), min(sh, int(math.ceil(y1 + p)))
        if a1 > a0 and b1 > b0:
            m[b0:b1, a0:a1] = 255
    return m


def load_scene(rng, W, H, pools):
    """(float32 H x W x 3 frame background, tag): a pool scene cover-cropped to W x H with its known text inpainted
    away, or a synthetic background."""
    from PIL import Image
    for _ in range(4):
        src = choose(rng, {k: v for k, v in pools.items() if k == "synthetic" or (SCENE_ASPECT.get(k) or
                                                                                    SCENES.get(k))})
        if src == "synthetic":
            break
        paths = SCENE_ASPECT[src] if src in SCENE_ASPECT else SCENES[src]
        p = paths[int(rng.integers(len(paths)))]
        try:
            with Image.open(p) as im:
                rgb = np.asarray(im.convert("RGB"))
        except OSError:
            continue
        sh, sw = rgb.shape[:2]
        z = float(rng.uniform(*MIX["scene_zoom"]))
        sc = max(W / sw, H / sh) * z
        cw, ch = min(sw, W / sc), min(sh, H / sc)
        cx = float(rng.uniform(0, sw - cw))
        cy = float(rng.uniform(0, sh - ch))
        x0, y0 = int(math.floor(cx)), int(math.floor(cy))
        x1, y1 = min(sw, int(math.ceil(cx + cw))), min(sh, int(math.ceil(cy + ch)))
        interp = cv2.INTER_AREA if (x1 - x0) > W else cv2.INTER_LINEAR
        img = cv2.resize(np.ascontiguousarray(rgb[y0:y1, x0:x1]), (W, H), interpolation=interp)
        mask = text_mask(p, sw, sh)[y0:y1, x0:x1]
        m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA) > 0
        if m.mean() > MIX["scene_max_text"]:
            continue
        if m.any():
            m = cv2.dilate(m.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
            img = cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        return np.ascontiguousarray(img, np.float32), f"{src}:{Path(p).stem}"
    return synthetic_background(rng, W, H), "synthetic"


# ----------------------------------------------------------------------------- the frame

class _NoRoom(Exception):
    pass


def instance_theta(cov):
    """The ink threshold of an instance's coverage: 0.5 (synth3's ink rule), or half the instance's peak coverage when
    that is under 1 (a thin face at a small size never reaches 0.5, yet is plainly drawn); None when the peak is under
    LABEL_MIN (too faint to label: an ignore area)."""
    q = float(np.max(cov)) if np.size(cov) else 0.0
    if q < LABEL_MIN:
        return None
    return min(0.5, 0.5 * q)


class Frame:
    """A frame being composed: its background (float32 RGB), the rectangles its elements reserve, the instances
    (frame x, y, bool mask, vertical, role, element) and the ignore areas."""

    def __init__(self, kind, W, H, platform, k, fit=None):
        self.kind, self.W, self.H, self.platform, self.k, self.fit = kind, W, H, platform, k, fit
        self.img = None
        self.rects = []
        self.inst = []
        self.ignore = []
        self.elements = []
        self.qa = {"overlap_redraw": 0, "unlabelled_redraw": 0, "unlabelled_px": 0}

    def free(self, x0, y0, x1, y1, margin=0):
        if x0 < 0 or y0 < 0 or x1 > self.W or y1 > self.H:
            return False
        for a0, b0, a1, b1 in self.rects:
            if x0 < a1 + margin and a0 < x1 + margin and y0 < b1 + margin and b0 < y1 + margin:
                return False
        return True

    def reserve(self, x0, y0, x1, y1):
        self.rects.append((int(x0), int(y0), int(x1), int(y1)))

    def add(self, x, y, cov, vertical, role, theta=None):
        """An instance: its ink coverage (float in [0, 1], local) at frame offset (x, y). Its ink is coverage >= theta
        (instance_theta: 0.5, lower for a thin or faint draw); an instance too faint to label is an ignore area.
        Stored trimmed to its nonzero coverage, so a scaling of the frame scales exactly this mask."""
        cov = np.asarray(cov, np.float32)
        if theta is None:
            theta = instance_theta(cov)
        if theta is None:               # nothing a reader could see as text: its faint ink is ignored
            ys, xs = np.nonzero(cov > FAINT)
            if len(ys):
                self.add_ignore(x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1)
            return False
        ys, xs = np.nonzero(cov > 0)
        if not len(ys) or not (cov >= theta).any():
            return False
        a, b, c, d = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        self.inst.append((int(x) + c, int(y) + a, np.ascontiguousarray(cov[a:b, c:d]), float(theta), int(vertical),
                          role, len(self.elements)))
        return True

    def add_ignore(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(self.W, int(x1)), min(self.H, int(y1))
        if x1 > x0 and y1 > y0:
            self.ignore.append([x0, y0, x1, y1])


def place(fr, rng, w, h, where, tries=30):
    """Top-left corner of a w x h element in free space, or None. where: bottom (ADV boxes and bands), centre
    (choices, NVL pages), side (menus), any (balloons, bare text), upper (scene text), edge (UI labels)."""
    W, H = fr.W, fr.H
    if w > W or h > H:
        return None
    m = max(2, int(round(4 * fr.k)))
    for t in range(tries):
        if where == "bottom":
            x = (W - w) / 2 + rng.uniform(-0.5, 0.5) * min(W - w, 0.2 * W)
            y = H - h - rng.uniform(0.0, 0.06 if t < 10 else 0.25) * H
        elif where == "centre":
            x = (W - w) / 2 + rng.uniform(-0.5, 0.5) * min(W - w, 0.3 * W)
            y = (H - h) / 2 + rng.uniform(-0.2, 0.2) * (H - h)
        elif where == "side":
            r = rng.random()
            x = (W - w) / 2 if r < 0.4 else (rng.uniform(0, 0.15) * (W - w) if r < 0.7 else
                                             (W - w) * (1 - rng.uniform(0, 0.15)))
            y = (H - h) / 2 + rng.uniform(-0.35, 0.35) * (H - h)
        elif where == "upper":
            x = rng.uniform(0, W - w)
            y = rng.uniform(0, max(0.0, min(H - h, 0.7 * H - h / 2)))
        elif where == "edge":
            inset_x = rng.uniform(0.005, 0.03) * W
            inset_y = rng.uniform(0.005, 0.04) * H
            a = int(rng.integers(8))
            xs = (inset_x, (W - w) / 2, W - w - inset_x)
            ys = (inset_y, (H - h) / 2, H - h - inset_y)
            x, y = [(xs[0], ys[0]), (xs[2], ys[0]), (xs[0], ys[2]), (xs[2], ys[2]), (xs[1], ys[0]),
                    (xs[1], ys[2]), (xs[0], ys[1]), (xs[2], ys[1])][a]
            if a >= 4 and rng.random() < 0.5:      # anywhere along that edge
                if a in (4, 5):
                    x = rng.uniform(0, W - w)
                else:
                    y = rng.uniform(0, H - h)
        else:
            x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
        x, y = int(np.clip(round(x), 0, W - w)), int(np.clip(round(y), 0, H - h))
        if fr.free(x, y, x + w, y + h, m):
            return x, y
    return None


# ----------------------------------------------------------------------------- synth3's regions in the frame

def line_rows(L, layout, H):
    """Rows [y0, y1) of a rendered line's band, as textdet_data.main_box takes them."""
    y0, y1 = max(0, int(math.floor(L["y0"]))), min(H, int(math.ceil(L["y1"] + 0.25 * L["fs"])))
    for o in layout:
        if o is not L and o["y0"] >= L["y0"] + 1:
            y1 = min(y1, int(math.floor(o["y0"] if o.get("ruby_top") is None else min(o["y0"], o["ruby_top"]))))
    y1 = max(y1, min(H, int(math.ceil(L["y1"]))))
    return y0, y1


def line_band(L, layout, H):
    """(y0, y1, x0, x1) of a rendered line's band: textdet_data.main_box's rows and columns (its box is the band's
    ink less the furigana), or None."""
    if L.get("ix0", -99) < 0:
        return None
    return (*line_rows(L, layout, H), L["ix0"], L["ix1"])


def line_window(L, W):
    """Columns [c0, c1) in which render_region looks for a line's own ink (its glyphs' advance plus its outline and
    shadow, recovered from the line's band: y0 = top - outline - 1, y1 = top + fs + outline + shadow + 1)."""
    outline = L["top"] - L["y0"] - 1
    shadow = L["y1"] - L["top"] - L["fs"] - outline - 1
    mx = outline + shadow + 2
    return max(0, int(math.floor(L["x0"] - mx))), min(W, max(0, int(math.ceil(L["x1"] + mx))))


def region_lines(lay, inks, tab, ruby, cursor, shape):
    """The instances of a rendered region: [(x0, y0, coverage crop, theta, role)] and its faint lines' ignore areas.
    A line's coverage is its ink source (the region's text ink, or the name tab's own ink for the tab's name) over
    textdet_data.main_box's band, its columns those of its ink in its nominal rows (render_region's ix0 / ix1, at
    the instance's threshold), less the furigana and the page cursors. Two tight lines may share a row or two of
    their bands (an outline or shadow reaching the next line), as in main_box."""
    H, W = shape
    keep = ~ruby if cursor is None else ~(ruby | cursor)
    out, ignore = [], []
    for i, L in enumerate(lay):
        src = inks[1] if (tab and i == 0 and len(inks) > 1) else inks[0]
        y0, y1 = line_rows(L, lay, H)
        c0, c1 = line_window(L, W)
        if y1 <= y0 or c1 <= c0:
            continue
        band = np.where(keep[y0:y1, c0:c1], src[y0:y1, c0:c1], 0.0).astype(np.float32)
        theta = instance_theta(band)
        if theta is None:
            ys, xs = np.nonzero(band > FAINT)
            if len(ys):
                ignore.append([c0 + int(xs.min()), y0 + int(ys.min()), c0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1])
            continue
        n0, n1 = max(int(L["y0"]), 0) - y0, max(int(L["y1"]), 0) - y0
        cols = np.flatnonzero((band[max(0, n0):max(0, n1)] >= theta).any(axis=0))
        if not len(cols):
            continue
        a, b = int(cols[0]), int(cols[-1]) + 1
        crop = np.zeros_like(band)
        crop[:, a:b] = band[:, a:b]
        out.append((c0, y0, crop, theta, ROLE.get(L.get("kind"), "dialogue")))
    return out, ignore


def ruby_ignore(ruby, sw, shadow, shape):
    """Ignore areas of a region's furigana: the parts of its ruby mask (the glyph fills), each grown by the outline
    (sw + 1 px) and, with a shadow, by 3 px more to the right and down."""
    H, W = shape
    g = int(math.ceil(sw)) + 1
    out = []
    for x0, y0, x1, y1 in mask_boxes(ruby):
        out.append([max(0, x0 - g), max(0, y0 - g), min(W, x1 + g + (3 if shadow else 0)),
                    min(H, y1 + g + (3 if shadow else 0))])
    return out


def _box(x, y, cov, theta):
    ys, xs = np.nonzero(cov >= theta)
    return [x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1]


def _overlap(a, b):
    """Two boxes that overlap by more than a shared row or two (two tight lines' outlines may touch: at most
    max(2 px, 15% of the smaller height) of rows)."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return ix > 0 and iy > max(2.0, 0.15 * min(a[3] - a[1], b[3] - b[1]))


def unlabelled_ink(inks, boxes, ruby, cursor):
    """The region's text ink (coverage >= 0.5, the furigana fills and the cursors left out) outside every box and
    ignore area: (pixel count, bool mask)."""
    ink = np.maximum.reduce(inks) >= 0.5
    ink &= ~ruby
    if cursor is not None:
        ink &= ~cursor
    for b in boxes:
        ink[b[1]:b[3], b[0]:b[2]] = False
    return int(ink.sum()), ink


def cursor_mask(cursors, shape, shadow):
    """The ink of synth2's page cursors drawn in a region (they are icons, not text): their shapes, grown by 1 px,
    and by up to 3 px right and down for a shadow."""
    from PIL import Image, ImageDraw
    h, w = shape
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for kind, cx, cy, s in cursors:
        for typ, pts in S3.S2.cursor_shape(kind, cx, cy, s):
            if typ == "ring":
                (x0, y0), (x1, y1) = pts
                d.ellipse((x0, y0, x1, y1), fill=255)
            else:
                d.polygon([(float(x), float(y)) for x, y in pts], fill=255)
    m = cv2.dilate(np.asarray(im, np.uint8), np.ones((3, 3), np.uint8)) > 0
    if shadow:
        out = m.copy()
        for dy in range(4):
            for dx in range(4):
                if dx or dy:
                    out[dy:, dx:] |= m[:h - dy, :w - dx]
        m = out
    return m


def mask_boxes(m, gx=2, gy=1):
    """Boxes of the parts of a bool mask (8-connected after growing gx px across and gy px up and down), each tight
    to the mask's own pixels."""
    if not m.any():
        return []
    d = cv2.dilate(m.astype(np.uint8), np.ones((2 * gy + 1, 2 * gx + 1), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        sub = m[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == i)
        ys, xs = np.nonzero(sub)
        if len(ys):
            out.append([x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1])
    return out


def _region_cfg(base, layout, pc):
    c = dict(base)
    c["layout"] = {layout: 1.0}
    c["capture"] = dict(base["capture"], vita={"clean": 1.0}, psp={"clean": 1.0})
    c["dither565"] = c["blur"] = c["noise"] = c["subnative"] = 0.0
    c["frames"] = None
    if pc:          # a PC screen: no bitmap fonts; synth3's "scaled" effects and box opacities
        c["category"] = dict(base["category"], vita={k: v for k, v in base["category"]["vita"].items()
                                                     if k != "pixel"})
        c["outline"] = dict(base["outline"], vita=base["outline"]["scaled"])
        c["shadow"] = dict(base["shadow"], vita=base["shadow"]["scaled"])
        c["box_alpha"] = dict(base["box_alpha"], vita=base["box_alpha"]["scaled"])
    return c


@contextlib.contextmanager
def _in_frame(fr, layout, where, placed, cursors, inks):
    """synth3 drawing one region into the frame: the layout forced, the region's background the frame under the
    place chosen for it, no region capture; page cursors and the text inks (synth2's compose_text: the region's
    text, then its name tab's) recorded; on a PC frame, synth3's version-3 PC em (its final em at the Vita target)
    at the PC frame's pixels."""
    base_cfg, base_bg, base_cursor, base_fs = S3.CFG, S3.background, S3.S2.draw_cursor, S3.font_size
    base_compose = S3.S2.compose_text

    def background(rng, w, h, platform, target, up):
        xy = place(fr, rng, w, h, where)
        if xy is None:
            raise _NoRoom
        x, y = xy
        placed[:] = [x, y, w, h]
        return fr.img[y:y + h, x:x + w].copy(), "frame"

    def draw_cursor(layer, kind, cx, cy, s):
        cursors.append((kind, cx, cy, s))
        return base_cursor(layer, kind, cx, cy, s)

    def compose_text(img, fills, s1, s2, cols, st, grad_rows=None):
        core, ink = base_compose(img, fills, s1, s2, cols, st, grad_rows)
        inks.append(np.array(ink, np.float32))      # a copy: render_region merges the tab's ink into this array
        return core, ink

    def font_size(rng, platform, font):
        mu, sd, lo, hi = base_cfg["scaled_em"]["vita"]
        fs = int(round(float(np.clip(rng.normal(mu, sd), lo, hi)) * fr.k))
        return fs, fs, 1
    S3.CFG = _region_cfg(base_cfg, layout, fr.kind == "pc")
    S3.background = background
    S3.S2.draw_cursor = draw_cursor
    S3.S2.compose_text = compose_text
    if fr.kind == "pc":
        S3.font_size = font_size
    try:
        yield
    finally:
        S3.CFG, S3.background, S3.S2.draw_cursor, S3.font_size = base_cfg, base_bg, base_cursor, base_fs
        S3.S2.compose_text = base_compose


def render_in_frame(fr, rng, layout):
    """One synth3 region of `layout` drawn into the frame: (x, y, rgb, layout lines, ink, ruby, cursors, inks, meta)
    or None (an unusable draw, or no room)."""
    placed, cursors, inks = [], [], []
    with _in_frame(fr, layout, PLACE_OF[layout], placed, cursors, inks):
        try:
            r = S3.render_region(rng, fr.platform)
        except _NoRoom:
            return None
    if r is None or not placed or not inks:
        return None
    out, lay, ink, meta = r
    x, y, w, h = placed
    if out.shape[:2] != (h, w) or any(a.shape != (h, w) for a in inks):
        return None
    ruby = meta.pop("_ruby_mask")
    meta.pop("_fill_mask", None)
    return x, y, out, lay, ink, ruby, cursors, inks, meta


def region_element(fr, rng, layout, tries=4):
    """A dialogue, choice or menu region of synth3 in the frame; its lines are instances. A draw whose lines' boxes
    overlap (a name tab drawn over the first line) or that leaves text ink outside every box is drawn again. True
    when placed."""
    for _ in range(tries):
        got = render_in_frame(fr, rng, layout)
        if got is None:
            continue
        x, y, out, lay, ink, ruby, cursors, inks, meta = got
        h, w = out.shape[:2]
        cm = cursor_mask(cursors, (h, w), meta.get("shadow")) if cursors else None
        lines, faint = region_lines(lay, inks, meta.get("name") == "tab", ruby, cm, (h, w))
        if not lines:
            continue
        boxes = [_box(c0, r0, crop, th) for c0, r0, crop, th, _ in lines]
        if any(_overlap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]):
            fr.qa["overlap_redraw"] += 1
            continue
        ignore = ruby_ignore(ruby, meta.get("sw") or 0, meta.get("shadow"), (h, w)) + faint
        if meta.get("partial"):         # the faded tail and ghost glyph of a typewriter line
            last = [b for b, ln in zip(boxes, lines) if ln[4] == "dialogue"]
            if last:
                b = last[-1]
                fs = max(L["fs"] for L in lay)
                ignore.append([b[2], b[1], min(w, b[2] + int(math.ceil(1.3 * fs)) + 2), b[3]])
        left, left_mask = unlabelled_ink(inks, boxes + ignore, ruby, cm)
        if left > UNLABELLED_MAX:
            fr.qa["unlabelled_redraw"] += 1
            continue
        if left:                        # stray ink (a furigana outline, a descender below a short next line)
            fr.qa["unlabelled_px"] += left
            ignore += [[max(0, b[0] - 1), max(0, b[1] - 1), min(w, b[2] + 1), min(h, b[3] + 1)]
                       for b in mask_boxes(left_mask, 1, 1)]
        fr.img[y:y + h, x:x + w] = out.astype(np.float32)
        for c0, r0, crop, th, role in lines:
            fr.add(x + c0, y + r0, crop, 0, role, th)
        for b in ignore:
            fr.add_ignore(x + b[0], y + b[1], x + b[2], y + b[3])
        fr.reserve(x, y, x + w, y + h)
        fr.elements.append(f"{layout}:{meta.get('layout')}")
        return True
    return False


def empty_region(fr, rng, layout, tries=4):
    """A synth3 region with its text inpainted away (#97's `empty` kind): an empty box, window, choice stack or
    menu. True when placed."""
    for _ in range(tries):
        got = render_in_frame(fr, rng, layout)
        if got is None:
            continue
        x, y, out, lay, ink, ruby, cursors, inks, meta = got
        if meta.get("partial"):
            continue
        drawn = np.maximum.reduce(inks) > FAINT
        m = cv2.dilate((ink | ruby | drawn).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        out = cv2.inpaint(np.ascontiguousarray(out), m, 3, cv2.INPAINT_TELEA)
        h, w = out.shape[:2]
        fr.img[y:y + h, x:x + w] = out.astype(np.float32)
        fr.reserve(x, y, x + w, y + h)
        fr.elements.append(f"empty:{layout}")
        return True
    return False


# ----------------------------------------------------------------------------- text of our own (UI, scene text)

def _font(rng, chars, platform, cats=None, allow_pixel=True, fs=None):
    """A font drawing every one of chars (spaces aside); for text under 20 px, never a static face of weight 300 or
    less (a thin face at a small size is mostly anti-aliasing: instance_theta still labels it, but it is rarely a
    game's choice)."""
    plat = "vita" if platform == "vita" else "psp"
    chars = set(chars) - SPACES
    if cats is not None:
        cats = {k: v for k, v in cats.items() if k in S3.R.by_cat and (allow_pixel or k != "pixel")}
    for _ in range(4):
        cat = choose(rng, cats) if cats else None
        f = S3.pick_font(rng, chars, plat, cat=cat)
        if f is None or (not allow_pixel and f.cat == "pixel"):
            continue
        if fs is not None and fs < 20 and f.weight <= 300 and "[wght]" not in f.path.name:
            continue
        return f
    return None


def _ui_cats(platform):
    if platform == "vita":
        return {"gothic": 0.55, "rounded": 0.2, "mincho": 0.05, "display": 0.1, "pixel": 0.1}
    return {"gothic": 0.45, "rounded": 0.15, "mincho": 0.05, "display": 0.07, "pixel": 0.28}


class Text:
    """One horizontal line of text with synth3's raster modes (Layer3) and synth2's compositing (compose_text)."""

    def __init__(self, rng, text, font, fs, platform, track=0):
        self.text, self.font = text, font
        geom = "vita" if platform == "vita" else "psp"
        if font.cat == "pixel":
            fs, em, mult = S3.font_size(rng, geom, font)
        else:
            em, mult = fs, 1
        self.fs = fs
        self.mode = S3.raster_mode(rng, geom, font)
        self.ck, self.aa, self.resample, self.integer, _ = S3.raster_setup(rng, self.mode, fs, mult)
        wght = S3.S2.pick_weight(rng, font.path)
        self.f = (S3.S2.get_font(font.path, em, wght) if self.mode == "pixel" else
                  S3.S2.get_font(font.path, max(1, int(round(fs * self.ck))), wght))
        self.track = track
        self.base_k = 1.0 if self.mode == "pixel" else 0.88
        self.q4c = self.mode == "atlas" and rng.random() < 0.7
        self.gamma = 1.0
        G = S3.CFG["gamma"]
        if rng.random() < G["p"] and self.mode not in ("pixel", "mono"):
            self.gamma = float(math.exp(rng.uniform(math.log(G["lo"]), math.log(G["hi"]))))
        self.width = sum(self.f.getlength(c) / self.ck for c in text) + track * max(0, len(text) - 1)

    def draw(self, img, x, top, fill, stroke=None, shadow=None):
        """Composites the line onto img (float32, local) with its left at x and its em top at `top`; returns its ink
        (float32, img's size)."""
        h, w = img.shape[:2]
        sw = stroke["w"] if stroke else 0
        layer = S3.Layer3(w, h, self.ck, self.aa, 1, sw, 0, self.resample, self.integer)
        base = top + self.base_k * self.fs
        for ch in self.text:
            if ch != "　":
                layer.glyph(x, base, ch, self.f, 0)
            x += self.f.getlength(ch) / self.ck + self.track
        fills, s1, s2, _ = layer.arrays(self.q4c, self.mode == "aa4", self.gamma)
        st = {"shadow": shadow, "glow": None, "stroke": stroke, "stroke2": None}
        _, ink = S3.S2.compose_text(img, fills, s1, None, [fill], st)
        return ink


def text_colours(rng, bg_luma, plate):
    """(fill, stroke or None, shadow or None) legible on a background of luma bg_luma; plate: on an opaque plate.
    The canvases are grey, so a fill whose luma is within 80 of the background's is replaced by near white or near
    black (a coloured fill that only differs in hue would be unreadable)."""
    S2 = S3.S2
    light = bg_luma < 128
    if light:
        fill = S2.text_light(rng) if rng.random() < 0.85 else S2.accent(rng, True)
    else:
        fill = S2.text_dark(rng) if rng.random() < 0.85 else S2.accent(rng, False)
    if abs(S2.luma(fill) - bg_luma) < 80:
        fill = S2.light_colour(rng, 0.05, 0.95) if light else S2.dark_colour(rng, 0.1)
    stroke = shadow = None
    p_effect = 0.3 if plate else 0.85
    if rng.random() < p_effect:
        if rng.random() < 0.6:
            stroke = {"w": 1 if rng.random() < 0.7 else 2,
                      "c": S2.dark_colour(rng, 0.2) if light else S2.light_colour(rng, 0.1, 0.9)}
        else:
            d = 1 if rng.random() < 0.7 else 2
            shadow = {"dx": d, "dy": d, "blur": 0.0, "a": float(rng.uniform(0.6, 1.0)),
                      "c": S2.dark_colour(rng, 0.15) if light else S2.light_colour(rng, 0.1, 0.85)}
    return fill, stroke, shadow


def luma_of(img):
    a = np.asarray(img, np.float32).reshape(-1, 3)
    return float((77 * a[:, 0] + 150 * a[:, 1] + 29 * a[:, 2]).mean() / 256.0)


def draw_plate(canvas, rng, rect, colour, alpha, radius=0, border=None):
    S3.S2.draw_box(canvas, rng, {"shape": "round" if radius else "rect", "rect": rect, "radius": radius,
                                 "colour": colour, "colour2": None, "alpha": alpha, "pattern": 0.0,
                                 "border": border}, {"on_dark": True})


def icon_mask(kind, size, sym_w):
    """A button glyph or menu icon as a drawn shape (never text): (body mask, symbol mask), float32 size x size."""
    from PIL import Image, ImageDraw
    n = 4 * size
    body = Image.new("L", (n, n), 0)
    sym = Image.new("L", (n, n), 0)
    db, ds = ImageDraw.Draw(body), ImageDraw.Draw(sym)
    c, r = n / 2, n / 2 - 2
    w = max(2, int(round(sym_w * 4)))
    if kind in ("circle", "cross", "triangle", "square"):
        db.ellipse((c - r, c - r, c + r, c + r), fill=255)
        q = r * 0.5
        if kind == "circle":
            ds.ellipse((c - q, c - q, c + q, c + q), outline=255, width=w)
        elif kind == "cross":
            ds.line((c - q, c - q, c + q, c + q), fill=255, width=w)
            ds.line((c - q, c + q, c + q, c - q), fill=255, width=w)
        elif kind == "triangle":
            ds.polygon([(c, c - q), (c + q, c + q * 0.75), (c - q, c + q * 0.75)], outline=255, width=w)
        else:
            ds.rectangle((c - q * 0.8, c - q * 0.8, c + q * 0.8, c + q * 0.8), outline=255, width=w)
    elif kind in ("start", "select", "lr"):
        db.rounded_rectangle((2, c - r * 0.55, n - 2, c + r * 0.55), radius=int(r * 0.4), fill=255)
        if kind == "start":
            ds.polygon([(c - r * 0.25, c - r * 0.3), (c + r * 0.35, c), (c - r * 0.25, c + r * 0.3)], fill=255)
        elif kind == "select":
            ds.rectangle((c - r * 0.35, c - r * 0.18, c + r * 0.35, c + r * 0.18), fill=255)
        else:
            ds.polygon([(c - r * 0.6, c), (c - r * 0.2, c - r * 0.3), (c - r * 0.2, c + r * 0.3)], fill=255)
            ds.polygon([(c + r * 0.6, c), (c + r * 0.2, c - r * 0.3), (c + r * 0.2, c + r * 0.3)], fill=255)
    elif kind == "star":
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.42
            pts.append((c + rr * math.cos(a), c + rr * math.sin(a)))
        db.polygon(pts, fill=255)
    elif kind == "heart":
        db.ellipse((c - r, c - r * 0.8, c, c + r * 0.1), fill=255)
        db.ellipse((c, c - r * 0.8, c + r, c + r * 0.1), fill=255)
        db.polygon([(c - r * 0.97, c - r * 0.2), (c + r * 0.97, c - r * 0.2), (c, c + r)], fill=255)
    elif kind == "gear":
        db.ellipse((c - r * 0.75, c - r * 0.75, c + r * 0.75, c + r * 0.75), fill=255)
        for i in range(8):
            a = i * math.pi / 4
            db.line((c, c, c + r * math.cos(a), c + r * math.sin(a)), fill=255, width=int(r * 0.35))
        ds.ellipse((c - r * 0.3, c - r * 0.3, c + r * 0.3, c + r * 0.3), fill=255)
    else:   # play / arrow
        db.polygon([(c - r * 0.6, c - r), (c + r, c), (c - r * 0.6, c + r)], fill=255)

    def down(im):
        return np.asarray(im.resize((size, size), Image.BOX), np.float32) / 255.0
    return down(body), down(sym)


def keep_spaces_clean(s):
    """synth3.clean, keeping ASCII and full-width spaces (drawn as gaps: dates, save slots, chapter titles)."""
    return "".join(c for c in s if c in S3.R.vocab_set or c in " 　").strip()


def ui_text(rng, kind):
    """(pieces, font size class): pieces are ("text", s) or ("icon", kind); every text piece is one instance."""
    clean = keep_spaces_clean
    if kind == "quick":
        words = QUICK_LATIN if rng.random() < 0.55 else QUICK_JP
        n = int(rng.integers(3, 7))
        pick = list(dict.fromkeys(words[int(i)] for i in rng.choice(len(words), size=n, replace=False)))
        return [("text", clean(w)) for w in pick if clean(w)], "small"
    if kind == "hint":
        out = []
        for i in rng.choice(len(HINTS), size=int(rng.integers(1, 4)), replace=False):
            icon, word = HINTS[int(i)]
            out += [("icon", icon), ("text", clean(word))]
        return out, "small"
    if kind == "date":
        r = rng.random()
        mo, d = int(rng.integers(1, 13)), int(rng.integers(1, 29))
        if r < 0.3:
            s = f"{mo}月{d}日（{'月火水木金土日'[int(rng.integers(7))]}）"
        elif r < 0.5:
            s = f"{int(rng.integers(1990, 2031))}/{mo:02d}/{d:02d}"
        elif r < 0.7:
            s = f"{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}"
        elif r < 0.85:
            s = f"{mo}月{d}日　{S3.UI_PLACES[int(rng.integers(len(S3.UI_PLACES)))]}"
        else:
            s = f"{int(rng.integers(1990, 2031))}/{mo:02d}/{d:02d} {int(rng.integers(0, 24)):02d}:" \
                f"{int(rng.integers(0, 60)):02d}"
        if rng.random() < 0.3:
            s = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in s)
        return [("text", clean(s))], "small"
    if kind == "chapter":
        num = "一二三四五六七八九十"[int(rng.integers(10))] if rng.random() < 0.7 else str(int(rng.integers(1, 30)))
        s = f"第{num}{['章', '話', '幕', '部', '節'][int(rng.integers(5))]}"
        if rng.random() < 0.7:
            t, _, _ = S3.narration_text(rng)
            t = re.split(r"[、。！？…「」『』（）　 ]", t)
            t = max(t, key=len) if t else ""
            if 2 <= len(t):
                s += "　" + t[:int(rng.integers(2, 9))]
        return [("text", clean(s))], "large"
    if kind == "slot":
        n = int(rng.integers(1, 100))
        s = f"No.{n:02d}" if rng.random() < 0.6 else f"セーブデータ{n:02d}"
        return [("text", clean(s))], "slot"
    if kind == "status":
        s = STATUS[int(rng.integers(len(STATUS)))]
        s = s.format(n=int(rng.integers(1, 99)), m=int(rng.integers(100, 999)), nn=f"{int(rng.integers(1, 99)):02d}")
        if s in ("好感度", "所持金") and rng.random() < 0.6:
            return [("text", clean(s)), ("text", clean(f"{int(rng.integers(1, 99999))}" +
                                                    ("円" if s == "所持金" else "")))], "small"
        return [("text", clean(s))], "small"
    if kind == "place":
        return [("text", clean(S3.UI_PLACES[int(rng.integers(len(S3.UI_PLACES)))]))], "small"
    w = S3.UI_WORDS[int(rng.integers(len(S3.UI_WORDS)))]
    return [("text", clean(w))], "small"


def ui_size(rng, fr, cls):
    if fr.platform == "psp":
        fs = int(rng.choice([9, 10, 10, 11, 11, 12, 12, 13, 14, 16]))
        if cls == "large":
            fs = int(rng.integers(13, 23))
        return fs
    if cls == "large":
        fs = float(np.clip(rng.normal(28, 6), 18, 44))
    else:
        fs = float(np.clip(rng.normal(18, 3.5), 12, 30))
    return int(round(fs * fr.k))


def text_row(fr, rng, pieces, fs, font, where, plate_p, role="ui", gap_em=(1.6, 3.5), vertical_stack=False,
             fill=None, colours=None, pad_em=None):
    """A row (or a column of rows) of text pieces and icons as separate instances, with an optional plate behind.
    True when placed."""
    if font is None or not pieces:
        return False
    texts = []
    for kind, s in pieces:
        if kind == "text":
            if not s or not set(s) - SPACES <= font.chars:
                return False
            texts.append(Text(rng, s, font, fs, fr.platform))
        else:
            texts.append(None)
    fs = next(t.fs for t in texts if t is not None)
    sw_max = 2
    ih = int(math.ceil(fs * 1.25)) + 2 * sw_max + 2
    pad = int(round(fs * (rng.uniform(0.4, 1.0) if pad_em is None else pad_em)))
    ixs, x = [], pad
    rows_y = []
    if vertical_stack:
        widths = [t.width if t is not None else fs * 1.1 for t in texts]
        w = int(math.ceil(max(widths))) + 2 * pad
        lead = int(round(fs * rng.uniform(0.6, 1.4)))
        h = pad * 2 + len(texts) * ih + (len(texts) - 1) * lead
        for i in range(len(texts)):
            ixs.append(pad)
            rows_y.append(pad + i * (ih + lead))
    else:
        for i, t in enumerate(texts):
            if i:
                prev_icon = texts[i - 1] is None
                g = rng.uniform(0.15, 0.4) if prev_icon else rng.uniform(*gap_em)
                x += int(round(g * fs))
            ixs.append(x)
            rows_y.append(pad)
            x += int(math.ceil(t.width)) if t is not None else int(round(fs * 1.1))
        w = x + pad
        h = ih + 2 * pad
    xy = place(fr, rng, w, h, where)
    if xy is None:
        return False
    X, Y = xy
    canvas = fr.img[Y:Y + h, X:X + w].copy()
    plate = rng.random() < plate_p
    if plate:
        S2 = S3.S2
        dark = rng.random() < 0.7
        col = S2.dark_colour(rng, 0.3) if dark else S2.light_colour(rng, 0.3, 0.8)
        if rng.random() < 0.25:
            col = S2.hsv(rng.random(), rng.uniform(0.4, 0.9), rng.uniform(0.25, 0.6) if dark else 0.85)
        alpha = float(rng.uniform(0.55, 0.95))
        border = (1, S2.light_colour(rng, 0.4, 0.7) if dark else S2.dark_colour(rng, 0.3)) if rng.random() < 0.4 \
            else None
        draw_plate(canvas, rng, (0, 0, w - 1, h - 1), col, alpha, int(rng.uniform(0, 0.5) * min(h, 24)), border)
    bg = luma_of(canvas)
    if colours is None:
        colours = text_colours(rng, bg, plate)
    tfill, stroke, shadow = colours
    if fill is not None:
        tfill = fill
    S2 = S3.S2
    for i, t in enumerate(texts):
        top = rows_y[i] + sw_max + 1
        if t is None:
            sz = int(round(fs * rng.uniform(0.95, 1.2)))
            body, sym = icon_mask(pieces[i][1], sz, max(1.0, fs / 9))
            y0 = rows_y[i] + (ih - sz) // 2
            x0 = ixs[i]
            sub = canvas[y0:y0 + sz, x0:x0 + sz]
            if sub.shape[:2] != (sz, sz):
                continue
            ic = S2.hsv(rng.random(), rng.uniform(0.3, 0.8), rng.uniform(0.5, 0.95)) if rng.random() < 0.6 else \
                (S2.light_colour(rng) if bg < 128 else S2.dark_colour(rng, 0.25))
            S2.over(sub, body, ic)
            S2.over(sub, sym, S2.light_colour(rng) if S2.luma(ic) < 150 else S2.dark_colour(rng, 0.2))
            continue
        ink = t.draw(canvas, ixs[i], top, tfill, stroke, shadow)
        fr.add(X, Y, ink, 0, role)
    fr.img[Y:Y + h, X:X + w] = canvas
    fr.reserve(X, Y, X + w, Y + h)
    fr.elements.append(f"{role}:row")
    return True


def ui_element(fr, rng):
    kind = choose(rng, MIX["ui"])
    pieces, cls = ui_text(rng, kind)
    chars = "".join(s for k, s in pieces if k == "text")
    if not chars:
        return False
    fs = ui_size(rng, fr, "large" if cls == "large" else "small")
    font = _font(rng, chars, fr.platform, _ui_cats(fr.platform), allow_pixel=fr.kind != "pc", fs=fs)
    if kind == "slot":
        return save_slots(fr, rng, fs, font)
    where = "edge" if kind != "chapter" or rng.random() < 0.6 else "upper"
    stack = kind == "quick" and rng.random() < 0.35
    return text_row(fr, rng, pieces, fs, font, where, MIX["ui_plate"], "ui", vertical_stack=stack)


def save_slots(fr, rng, fs, font):
    """1-4 save slots: a plate each, with a thumbnail box and two lines (slot number and date; a chapter or place)."""
    if font is None:
        return False
    n = int(rng.integers(1, 5))
    lines = []
    for i in range(n):
        a = f"No.{int(rng.integers(1, 100)):02d}　{int(rng.integers(2000, 2031))}/{int(rng.integers(1, 13)):02d}/" \
            f"{int(rng.integers(1, 29)):02d} {int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}"
        b = S3.UI_PLACES[int(rng.integers(len(S3.UI_PLACES)))] if rng.random() < 0.5 else \
            f"第{'一二三四五六七八九十'[int(rng.integers(10))]}章"
        lines.append((keep_spaces_clean(a), keep_spaces_clean(b)))
    if not all(set(a + b) - SPACES <= font.chars for a, b in lines):
        return False
    ta = [(Text(rng, a, font, fs, fr.platform), Text(rng, b, font, fs, fr.platform)) for a, b in lines]
    fs = ta[0][0].fs
    ih = int(math.ceil(fs * 1.25)) + 6
    thumb = int(round(2.3 * ih))
    pad = int(round(fs * 0.5))
    tw = int(math.ceil(max(max(t1.width, t2.width) for t1, t2 in ta)))
    sw_ = pad * 3 + int(thumb * 16 / 9) + tw
    sh_ = thumb + 2 * pad
    gap = int(round(fs * rng.uniform(0.3, 0.9)))
    w, h = sw_, n * sh_ + (n - 1) * gap
    xy = place(fr, rng, w, h, "side")
    if xy is None:
        return False
    X, Y = xy
    canvas = fr.img[Y:Y + h, X:X + w].copy()
    S2 = S3.S2
    dark = rng.random() < 0.7
    col = S2.dark_colour(rng, 0.3) if dark else S2.light_colour(rng, 0.3, 0.8)
    colours = text_colours(rng, 60 if dark else 200, True)
    for i, (t1, t2) in enumerate(ta):
        y0 = i * (sh_ + gap)
        draw_plate(canvas, rng, (0, y0, w - 1, y0 + sh_ - 1), col, float(rng.uniform(0.6, 0.95)),
                   int(rng.uniform(0, 8) * fr.k))
        tx0, ty0 = pad, y0 + pad
        tcol = rng.uniform(40, 200, 3).astype(np.float32)
        canvas[ty0:ty0 + thumb, tx0:tx0 + int(thumb * 16 / 9)] = tcol
        lx = pad * 2 + int(thumb * 16 / 9)
        for t, ty in ((t1, ty0 + 2), (t2, ty0 + thumb - ih + 2)):
            ink = t.draw(canvas, lx, ty, *colours)
            fr.add(X, Y, ink, 0, "ui")
    fr.img[Y:Y + h, X:X + w] = canvas
    fr.reserve(X, Y, X + w, Y + h)
    fr.elements.append("ui:slots")
    return True


def scene_text_element(fr, rng, vertical):
    """Text inside the picture (a sign, a poster, a note, a phone screen): class scene."""
    kind = choose(rng, MIX["scene_text_kind"])
    clean = keep_spaces_clean
    if kind == "sign":
        r = rng.random()
        if r < 0.5:
            s = SIGNS[int(rng.integers(len(SIGNS)))]
        elif r < 0.8:
            name = S3.sample_name(rng)[1]
            s = name[:4] + SIGN_SUFFIX[int(rng.integers(len(SIGN_SUFFIX)))]
        else:
            s = S3.UI_PLACES[int(rng.integers(len(S3.UI_PLACES)))]
        texts = [clean(s)]
    elif kind == "poster":
        texts = [clean(SIGNS[int(rng.integers(len(SIGNS)))])]
        if rng.random() < 0.7:
            t, _, _ = S3.narration_text(rng)
            texts.append(clean(t)[:int(rng.integers(5, 14))])
        if rng.random() < 0.4:
            texts.append(clean(f"{int(rng.integers(1, 13))}月{int(rng.integers(1, 29))}日"))
    elif kind == "note":
        t, _, _ = S3.narration_text(rng) if rng.random() < 0.5 else S3.dialogue_text(rng, False)[:3]
        t = clean(t).strip("「」『』")
        n = int(rng.integers(1, 4))
        cpl = int(rng.integers(5, 13))
        texts = [t[i * cpl:(i + 1) * cpl] for i in range(n)]
    else:
        t, _, _ = S3.short_text(rng, False)
        texts = [clean(t).strip("「」『』")]
        if rng.random() < 0.6:
            texts.insert(0, clean(S3.sample_name(rng)[1]))
    texts = [t for t in texts if t]
    if not texts:
        return False
    if vertical:
        return vertical_block(fr, rng, texts[:2], "scene", kind)
    cats = {"note": {"pen": 0.4, "mincho": 0.3, "rounded": 0.3}, "sign": {"gothic": 0.5, "mincho": 0.25,
                                                                          "display": 0.25}}.get(kind)
    if fr.platform == "psp":
        fs = int(np.clip(round(rng.normal(13, 3)), 8, 24))
    else:
        fs = int(round(float(np.clip(rng.normal(22, 6), 11, 44)) * fr.k))
    font = _font(rng, "".join(texts), fr.platform, cats or _ui_cats(fr.platform), allow_pixel=False, fs=fs)
    if font is None:
        return False
    return text_block(fr, rng, texts, font, fs, kind)


def text_block(fr, rng, texts, font, fs, kind):
    """1-3 lines on a plate (sign / poster / note / phone screen), each line an instance of class scene."""
    if not all(set(t) - SPACES <= font.chars for t in texts):
        return False
    ts = [Text(rng, t, font, fs if i == 0 or kind != "poster" else max(6, int(fs * rng.uniform(0.5, 0.8))),
               fr.platform) for i, t in enumerate(texts)]
    lead = [int(math.ceil(t.fs * rng.uniform(1.3, 1.7))) for t in ts]
    pad = int(round(ts[0].fs * rng.uniform(0.4, 1.0)))
    w = int(math.ceil(max(t.width for t in ts))) + 2 * pad
    h = sum(lead) + 2 * pad
    xy = place(fr, rng, w, h, "upper")
    if xy is None:
        return False
    X, Y = xy
    canvas = fr.img[Y:Y + h, X:X + w].copy()
    S2 = S3.S2
    if kind == "phone":
        col = S2.dark_colour(rng, 0.2)
    elif kind in ("poster", "note"):
        col = S2.light_colour(rng, 0.25, 0.8)
    else:
        col = S2.hsv(rng.random(), rng.uniform(0.0, 0.9), rng.uniform(0.15, 0.95))
    border = (max(1, int(round(rng.uniform(1, 3) * fr.k))), S2.dark_colour(rng, 0.3)) if rng.random() < 0.5 \
        else None
    draw_plate(canvas, rng, (0, 0, w - 1, h - 1), col, float(rng.uniform(0.85, 1.0)),
               int(rng.uniform(0, 0.3) * min(h, 30)) if kind == "phone" else 0, border)
    colours = text_colours(rng, S2.luma(col), True)
    y = pad
    for t, ld in zip(ts, lead):
        ink = t.draw(canvas, pad + (w - 2 * pad - t.width) / 2 if kind == "sign" else pad,
                     y + (ld - t.fs) / 2, *colours)
        fr.add(X, Y, ink, 0, "scene")
        y += ld
    fr.img[Y:Y + h, X:X + w] = canvas
    fr.reserve(X, Y, X + w, Y + h)
    fr.elements.append(f"scene:{kind}")
    return True


# ----------------------------------------------------------------------------- vertical text

def _column_texts(rng, n, cpc):
    s = ""
    for _ in range(4):
        t, _, _ = S3.narration_text(rng) if rng.random() < 0.4 else S3.dialogue_text(rng, rng.random() < 0.3)
        s += t.replace("　", "").replace(" ", "")
        if len(s) >= n * cpc:
            break
    cols = [s[i * cpc:(i + 1) * cpc] for i in range(n)]
    return [c for c in cols if c]


def vertical_columns(rng, texts, font_path, fs, pitch, gap, stroke, furi, ss=2):
    """Masks of vertical columns, right to left: (w, h, [(fill, stroke or None[, furigana]) per column]), each float32
    at output size. The typesetting is vertical_render.draw_column (raqm, direction ttb), one cell of `pitch` px per
    character; a column's furigana (when drawn) is a run of small kana at its right."""
    from PIL import Image
    n = len(texts)
    m = int(round(fs * 0.6))
    rw = int(round(fs * 0.55)) if furi else 0
    W = n * fs + (n - 1) * gap + 2 * m + rw
    H = max(len(t) for t in texts) * pitch + 2 * m
    F = VR.font(font_path, fs * ss)
    out = []
    for c, t in enumerate(texts):
        x_c = (W - m - rw - fs / 2 - c * (fs + gap)) * ss
        fill = Image.new("L", (W * ss, H * ss), 0)
        sk = Image.new("L", (W * ss, H * ss), 0) if stroke else None
        VR.draw_column(fill, x_c, m * ss, t, F, fs * ss, pitch * ss, stroke, sk, ss)
        fa = np.asarray(fill.resize((W, H), Image.BOX), np.float32) / 255.0
        sa = np.asarray(sk.resize((W, H), Image.BOX), np.float32) / 255.0 if sk is not None else None
        out.append((fa, sa))
        if furi and rng.random() < 0.5 and len(t) > 1:
            k0 = int(rng.integers(0, len(t) - 1))
            k1 = int(min(len(t), k0 + rng.integers(1, 4)))
            ruby = "".join(VR.FURI_KANA[int(rng.integers(len(VR.FURI_KANA)))] for _ in range(2 * (k1 - k0)))
            rl = Image.new("L", (W * ss, H * ss), 0)
            rx = x_c + (fs * 0.5 + fs * 0.28 + 1) * ss
            VR.draw_column(rl, rx, (m + k0 * pitch) * ss, ruby, VR.font(font_path, max(5, int(fs * ss * 0.5))),
                           int(fs * ss * 0.5), int(round(pitch * ss / 2)))
            ra = np.asarray(rl.resize((W, H), Image.BOX), np.float32) / 255.0
            out[-1] = (fa, sa, ra)
    return W, H, out


def vertical_block(fr, rng, texts, role, kind):
    """1-2 vertical columns of scene text (a vertical sign) on a plate."""
    fs = int(np.clip(round(rng.normal(13, 3)), 9, 22)) if fr.platform == "psp" else \
        int(round(float(np.clip(rng.normal(24, 5), 14, 40)) * fr.k))
    font = _font(rng, "".join(texts), fr.platform, {"gothic": 0.5, "mincho": 0.3, "display": 0.2},
                 allow_pixel=False, fs=fs)
    if font is None:
        return False
    texts = [t.replace(" ", "").replace("　", "")[:int(rng.integers(2, 9))] for t in texts]
    texts = [t for t in texts if t]
    return vertical_element(fr, rng, "sign", texts=texts, font=font, fs=fs, role=role)


def vertical_element(fr, rng, mode, texts=None, font=None, fs=None, role="dialogue"):
    """Vertical columns over a translucent box or the scene ("adv"), in a manga balloon ("manga") or on a sign
    ("sign"): one instance per column. True when placed."""
    S2 = S3.S2
    psp = fr.platform == "psp"
    if fs is None:
        if psp:
            fs = int(rng.choice([10, 11, 12, 12, 13, 13, 14, 14, 15, 16]))
        else:
            mu, sd = (24, 4.5) if mode == "manga" else (25, 3.5)
            fs = int(round(float(np.clip(rng.normal(mu, sd), 16, 34)) * fr.k))
    if texts is None:
        n = int(rng.integers(1, 5)) if mode == "manga" else int(rng.integers(2, 9))
        cpc = int(rng.integers(4, 10)) if mode == "manga" else int(rng.integers(6, 17))
        texts = _column_texts(rng, n, cpc)
    if not texts:
        return False
    if font is None:
        font = _font(rng, "".join(texts), fr.platform, None, allow_pixel=False, fs=fs)
        if font is None:
            return False
    if not set("".join(texts)) - SPACES <= font.chars:
        return False
    pitch = int(round(fs * rng.uniform(1.0, 1.18)))
    gap = int(round(fs * rng.uniform(0.35, 1.1 if mode != "manga" else 0.6)))
    furi = mode == "adv" and fs >= 12 and rng.random() < 0.15
    if furi:                                # room for a run of ruby between two columns
        gap = max(gap, int(round(0.7 * fs)))
    boxed = mode == "adv" and rng.random() < 0.6
    if mode == "manga":
        tc, oc = np.array([rng.uniform(0, 45)] * 3, np.float32), None
    elif mode == "sign":
        tc, oc = None, None
    else:
        light = rng.random() < 0.75
        tc = np.array([rng.uniform(215, 255)] * 3 if light else [rng.uniform(0, 50)] * 3, np.float32)
        # legible over a scene: without a box, an outline (70%) or a shadow (below); with one, sometimes an outline
        outline = rng.random() < (0.3 if boxed else 0.7)
        oc = np.array([rng.uniform(0, 40)] * 3 if light else [rng.uniform(215, 255)] * 3, np.float32) \
            if outline else None
    stroke = (1 if fs < 20 or rng.random() < 0.6 else 2) if oc is not None else 0
    w, h, cols = vertical_columns(rng, texts, font.path, fs, pitch, gap, stroke, furi)
    pad = int(round(fs * (0.5 if mode != "manga" else 0.9)))
    W, H = w + 2 * pad, h + 2 * pad
    if mode == "adv" and rng.random() < 0.3:
        H = max(H, min(fr.H, int(fr.H * rng.uniform(0.7, 0.95))))
    xy = place(fr, rng, W, H, "any" if mode != "adv" else ("side" if rng.random() < 0.7 else "any"))
    if xy is None:
        return False
    X, Y = xy
    canvas = fr.img[Y:Y + H, X:X + W].copy()
    if mode == "adv" and not boxed:     # text of the scene's own polarity gets a box under it
        bl = luma_of(canvas)
        boxed = (tc.mean() > 128 and bl > 140) or (tc.mean() <= 128 and bl < 110)
    if mode == "manga":
        S2.draw_box(canvas, rng, {"shape": "ellipse" if rng.random() < 0.7 else "round", "rect": (0, 0, W - 1, H - 1),
                                  "radius": int(min(W, H) * 0.3), "colour": np.array([rng.uniform(235, 255)] * 3),
                                  "colour2": None, "alpha": 1.0, "pattern": 0.0,
                                  "border": (max(1, int(round(2 * fr.k))), S2.dark_colour(rng, 0.15))}, {})
    elif mode == "sign":
        col = S2.hsv(rng.random(), rng.uniform(0.0, 0.9), rng.uniform(0.15, 0.95))
        draw_plate(canvas, rng, (0, 0, W - 1, H - 1), col, float(rng.uniform(0.85, 1.0)), 0,
                   (max(1, int(round(2 * fr.k))), S2.dark_colour(rng, 0.3)) if rng.random() < 0.5 else None)
        tc = np.asarray(text_colours(rng, S2.luma(col), True)[0], np.float32)
    elif boxed:
        dark = tc.mean() > 128
        draw_plate(canvas, rng, (0, 0, W - 1, H - 1), S2.dark_colour(rng, 0.2) if dark else
                   S2.light_colour(rng, 0.2, 0.85), float(rng.uniform(0.55, 0.9)), 0)
    shadow = mode == "adv" and oc is None and (not boxed or rng.random() < 0.3)
    for c in cols:
        fa, sa = c[0], c[1]
        sub = canvas[pad:pad + h, pad:pad + w]
        if shadow:
            sh_m = np.zeros_like(fa)
            d = 1 if psp else 2
            sh_m[d:, d:] = fa[:-d, :-d]
            S2.over(sub, sh_m, (10, 10, 10), 0.8)
        if sa is not None and oc is not None:
            S2.over(sub, sa, oc)
        S2.over(sub, fa, tc)
        ink = np.maximum(fa, sa) if sa is not None else fa.copy()
        if shadow:
            ink = np.maximum(ink, sh_m)
        fr.add(X + pad, Y + pad, ink, 1, role)
        if len(c) > 2:
            ra = c[2]
            S2.over(sub, ra, tc)
            for b in mask_boxes(ra >= 0.35, 1, 2):
                fr.add_ignore(X + pad + b[0], Y + pad + b[1], X + pad + b[2], Y + pad + b[3])
    fr.img[Y:Y + H, X:X + W] = canvas
    fr.reserve(X, Y, X + W, Y + H)
    fr.elements.append(f"vertical:{mode}")
    return True


# ----------------------------------------------------------------------------- text-free extras

def menu_plates(fr, rng):
    """A menu screen of plates and icons without text."""
    S2 = S3.S2
    n = int(rng.integers(3, 9))
    pw = int(round(rng.uniform(0.18, 0.45) * fr.W))
    ph = int(round(rng.uniform(0.05, 0.11) * fr.H))
    gap = int(round(rng.uniform(0.01, 0.04) * fr.H))
    grid = rng.random() < 0.3
    cols = 2 if grid else 1
    rows = int(math.ceil(n / cols))
    W, H = cols * pw + (cols - 1) * gap, rows * ph + (rows - 1) * gap
    xy = place(fr, rng, W, H, "side")
    if xy is None:
        return False
    X, Y = xy
    canvas = fr.img[Y:Y + H, X:X + W].copy()
    dark = rng.random() < 0.7
    col = S2.dark_colour(rng, 0.3) if dark else S2.light_colour(rng, 0.3, 0.8)
    for i in range(n):
        x0, y0 = (i % cols) * (pw + gap), (i // cols) * (ph + gap)
        draw_plate(canvas, rng, (x0, y0, x0 + pw - 1, y0 + ph - 1), col, float(rng.uniform(0.5, 0.95)),
                   int(rng.uniform(0, 0.5) * ph), (1, S2.light_colour(rng)) if rng.random() < 0.3 else None)
        if rng.random() < 0.6:
            sz = int(ph * rng.uniform(0.5, 0.8))
            kind = ["star", "heart", "gear", "play", "circle", "cross", "triangle", "square"][int(rng.integers(8))]
            body, sym = icon_mask(kind, sz, max(1.0, sz / 10))
            iy = y0 + (ph - sz) // 2
            ix = x0 + int(ph * 0.3)
            sub = canvas[iy:iy + sz, ix:ix + sz]
            if sub.shape[:2] == (sz, sz):
                S2.over(sub, body, S2.accent(rng, dark))
                S2.over(sub, sym, S2.light_colour(rng))
    fr.img[Y:Y + H, X:X + W] = canvas
    fr.reserve(X, Y, X + W, Y + H)
    fr.elements.append("menu:plates")
    return True


# ----------------------------------------------------------------------------- frames

def pc_size(rng):
    w, h = (int(v) for v in choose(rng, MIX["pc_size"]).split("x"))
    return w, h


def fit_size(w, h):
    """teacher/geometry.target_scale's fit to 960 x 544 (with the aspect): (scale, (W2, H2))."""
    s = min(FIT[0] / w, FIT[1] / h)
    return s, (max(1, int(round(w * s))), max(1, int(round(h * s))))


def new_frame(rng, kind):
    if kind == "vita":
        fr = Frame(kind, *VITA, "vita", 1.0)
        pools = MIX["scenes"]["vita"]
    elif kind in ("psp", "adrenaline"):
        fr = Frame(kind, *PSP, "psp", 0.5)
        pools = MIX["scenes"]["psp"]
    else:
        W, H = pc_size(rng)
        s, fit = fit_size(W, H)
        fr = Frame(kind, W, H, "vita", 1.0 / s, fit)
        pools = dict(MIX["scenes"]["pc"])
        pools["16:9" if abs(W / H - 16 / 9) < 0.06 else "4:3"] = pools.pop("hires")
    fr.img, fr.scene = load_scene(rng, fr.W, fr.H, pools)
    return fr


def text_frame(fr, rng):
    main = choose(rng, MIX["main"])
    has_main = False
    if main == "horizontal":
        has_main = region_element(fr, rng, choose(rng, MIX["layout"]))
    elif main == "vertical":
        mode = "manga" if fr.platform == "vita" and rng.random() < MIX["vertical_manga"] else "adv"
        has_main = vertical_element(fr, rng, mode)
        if has_main and mode == "manga" and rng.random() < 0.5:
            vertical_element(fr, rng, "manga")
    key = "with_main" if has_main else "without_main"
    if rng.random() < MIX["choice"][key]:
        region_element(fr, rng, "choice")
    if rng.random() < MIX["menu"][key]:
        region_element(fr, rng, "ui")
    if rng.random() < MIX["scene_text"]:
        for _ in range(int(rng.integers(MIX["scene_text_n"][0], MIX["scene_text_n"][1] + 1))):
            scene_text_element(fr, rng, rng.random() < MIX["scene_text_vertical"])
    if main == "horizontal" and rng.random() < MIX["vertical_extra"]:
        vertical_element(fr, rng, "manga" if fr.platform == "vita" else "adv")
    n_ui = int(rng.choice(len(MIX["ui_n"]), p=np.asarray(MIX["ui_n"]) / np.sum(MIX["ui_n"])))
    for _ in range(n_ui):
        ui_element(fr, rng)


def empty_frame(fr, rng, mode):
    if mode == "box":
        empty_region(fr, rng, choose(rng, MIX["empty_box_layout"]))
    elif mode == "menu":
        menu_plates(fr, rng)
        if rng.random() < 0.4:
            menu_plates(fr, rng)
    fr.elements.append(f"empty:{mode}")


def jpeg_chain(rng, img, kind):
    """The frame's capture chain (CAPTURE): (image, tag)."""
    C = CAPTURE[kind]
    chain = choose(rng, C["chain"])
    if chain == "clean":
        return img, "clean"
    q420 = C["q420"][chain] if isinstance(C["q420"], dict) else C["q420"]
    if chain == "vndb2":
        lo, hi = CAPTURE["console_q"]
        img = S3.jpeg_frame(img, int(rng.integers(lo, hi)), 2)
    sub = 2 if rng.random() < q420 else 0
    return S3.jpeg_frame(img, 90, sub), f"{chain}-{'420' if sub == 2 else '444'}"


def blur_noise(rng, img, tags):
    out = img
    if rng.random() < CAPTURE["blur"]:
        out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), float(rng.uniform(0.3, 0.7)))
        tags.append("blur")
    if rng.random() < CAPTURE["noise"]:
        out = out.astype(np.float32) + rng.normal(0, rng.uniform(1, 4), out.shape).astype(np.float32)
        tags.append("noise")
    return np.clip(np.asarray(out, np.float32) + 0.5, 0, 255).astype(np.uint8) if out is not img else img


def scale_box(x, y, cov, src, dst, interp, theta=0.5):
    """The box of an instance's ink after the frame is resized: its coverage (at frame offset x, y in a src = (W, H)
    frame) placed in a whole frame of zeros, resized to dst with interp, thresholded at theta; None when nothing is
    left. The whole-frame resize, so the box is the instance's own ink in the resized frame."""
    full = np.zeros((src[1], src[0]), np.float32)
    h, w = cov.shape
    full[y:y + h, x:x + w] = cov
    r = cv2.resize(full, dst, interpolation=interp) >= theta
    ys = np.flatnonzero(r.any(axis=1))
    if not len(ys):
        return None
    xs = np.flatnonzero(r.any(axis=0))
    return [int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1]


def scale_rect(b, kx, ky, W, H):
    return [max(0, int(math.floor(b[0] * kx))), max(0, int(math.floor(b[1] * ky))),
            min(W, int(math.ceil(b[2] * kx))), min(H, int(math.ceil(b[3] * ky)))]


def finalize(fr, rng):
    """The captured grey canvas, its lines and ignore areas: {"canvas", "lines", "ignore", "meta"}."""
    tags = []
    img = np.clip(fr.img + 0.5, 0, 255).astype(np.uint8)
    if fr.kind in ("psp", "adrenaline"):
        if rng.random() < CAPTURE["dither565"]:
            img = np.clip(S3.dither565(img.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8)
            tags.append("rgb565")
    img = blur_noise(rng, img, tags)
    ignore = [list(b) for b in fr.ignore]
    dst = interp = None
    if fr.kind == "adrenaline":
        dst, interp = VITA, cv2.INTER_LINEAR
        img = cv2.resize(img, dst, interpolation=interp)
        tags.append("2x")
        img, chain = jpeg_chain(rng, img, "adrenaline")
    elif fr.kind == "pc":
        img, chain = jpeg_chain(rng, img, "pc")
        dst, interp = fit_size(fr.W, fr.H)[1], cv2.INTER_AREA
        img = cv2.resize(img, dst, interpolation=interp)
        tags.append(f"fit{dst[0]}x{dst[1]}")
    else:
        img, chain = jpeg_chain(rng, img, fr.kind)
    tags.append(chain)
    H, W = img.shape[:2]
    kx, ky = W / fr.W, H / fr.H
    if dst is not None:
        ignore = [scale_rect(b, kx, ky, W, H) for b in ignore]
    lines, line_el = [], []
    for x, y, cov, theta, v, role, el in fr.inst:
        b = _box(x, y, cov, theta) if dst is None else scale_box(x, y, cov, (fr.W, fr.H), dst, interp, theta)
        if b is None:           # the instance's ink vanished in the scaling: its scaled rectangle is ignored
            ignore.append(scale_rect([x, y, x + cov.shape[1], y + cov.shape[0]], kx, ky, W, H))
            continue
        across = (b[2] - b[0]) if v else (b[3] - b[1])
        if across < MIN_ACROSS:
            ignore.append(b)
            continue
        lines.append([b[0], b[1], b[2], b[3], v, role])
        line_el.append(el)
    ignore = [b for b in ignore if b[2] > b[0] and b[3] > b[1]]
    meta = {"frame": fr.kind, "render": [fr.W, fr.H], "scene": fr.scene, "elements": fr.elements,
            "line_element": line_el, "capture": "+".join(tags), "qa": dict(fr.qa)}
    return {"canvas": to_gray(img), "lines": lines, "ignore": ignore, "meta": meta}


def sample_frame(rng):
    """One synthetic frame (a sample dict without off), or None for an unusable draw."""
    kind = choose(rng, MIX["kind"])
    empty = kind == "empty"
    if empty:
        kind = choose(rng, MIX["empty_size"])
    fr = new_frame(rng, kind)
    if empty:
        empty_frame(fr, rng, choose(rng, MIX["empty_mode"]))
    else:
        text_frame(fr, rng)
        if not fr.inst:
            return None
    s = finalize(fr, rng)
    if not empty and not s["lines"]:
        return None
    s["src"] = "synth:empty" if empty else f"synth:{kind}"
    s["kind"] = "psp" if kind == "psp" else "vita"
    s["vn"] = s["shot"] = None
    return s


def run_shard(job):
    idx, n, seed, out_dir = job
    path = Path(out_dir) / f"synth_frames_{idx:04d}"
    if path.with_suffix(".json").exists():
        return idx, 0, 0, 0.0
    rng = np.random.default_rng([seed, idx])
    t0, samples, errors = time.time(), [], 0
    while len(samples) < n:
        try:
            s = sample_frame(rng)
        except Exception as e:  # noqa: BLE001 - a rare bad draw must not end a long render
            errors += 1
            if errors <= 3:
                import traceback
                print(f"shard {idx}: error {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            if errors > 200:
                raise
            continue
        if s is not None:
            samples.append(s)
    TDD.write_shard(path, samples)
    return idx, len(samples), errors, time.time() - t0


# ----------------------------------------------------------------------------- commands

CODE = ("synth/textspot_data.py", "synth/textdet_data.py", "synth/synth3.py", "synth/render.py",
        "synth/vertical_render.py", "synth/common.py", "teacher/geometry.py")
COLOURS = {"dialogue": (0, 255, 0), "name": (0, 200, 255), "choice": (255, 255, 0), "ui": (255, 0, 255),
           "scene": (255, 140, 0)}


def cmd_synth(args):
    init()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool_path = out / "scene_pool.json"
    doc = dict(POOL, paths={k: sorted(Path(p).name for p in v) for k, v in SCENES.items()})
    if pool_path.exists():
        old = json.loads(pool_path.read_text())
        if old.get("sha256") != doc.get("sha256"):
            raise SystemExit(f"{pool_path}: the scene pool changed ({old.get('sha256')} -> {doc.get('sha256')})")
    else:
        pool_path.write_text(json.dumps(doc, indent=0) + "\n", encoding="utf-8")
    print(f"scene pool {POOL['textgate']} -> {POOL['pool']} ({POOL['sha256'][:12]})", flush=True)
    jobs = [(i, args.per_shard, args.seed, str(out)) for i in range(args.first, args.first + args.shards)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, errors, dt in pool.imap_unordered(run_shard, jobs):
            print(f"shard {idx}: {n} frames ({errors} errors) in {dt:.0f}s ({time.time() - t0:.0f}s)", flush=True)
    print("SYNTH_DONE", flush=True)


def draw_sample(canvas, s, scale=1.0):
    """RGB PIL image of a canvas with its boxes coloured by class (2 px for vertical) and its ignore areas hatched."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(canvas).convert("RGB")
    if scale != 1.0:
        im = im.resize((max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale)))), Image.BILINEAR)
    d = ImageDraw.Draw(im)

    def sc(b):
        return [b[0] * scale, b[1] * scale, b[2] * scale - 1, b[3] * scale - 1]
    for b in s["ignore"]:
        x0, y0, x1, y1 = sc(b)
        d.rectangle((x0, y0, x1, y1), outline=(255, 40, 40))
        t = x0 - (y1 - y0)
        while t < x1:
            # a hatch line from (t, y1) to (t + h, y0), clipped to the box
            ax, ay, bx, by = t, y1, t + (y1 - y0), y0
            if ax < x0:
                ay -= x0 - ax
                ax = x0
            if bx > x1:
                by += bx - x1
                bx = x1
            if ax <= bx:
                d.line((ax, ay, bx, by), fill=(255, 40, 40))
            t += 4
    for x0, y0, x1, y1, v, role in s["lines"]:
        d.rectangle(sc([x0, y0, x1, y1]), outline=COLOURS.get(role, (255, 255, 255)), width=2 if v else 1)
    return im


def cmd_preview(args):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(args.seed)
    paths = sorted(p.with_suffix("") for p in Path(args.dir).glob("*.npy") if ".tmp" not in p.name)
    sizes = [len(json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))["samples"]) for p in paths]
    flat = [(i, k) for i, n in enumerate(sizes) for k in range(n)]
    pick = rng.choice(len(flat), size=min(args.n, len(flat)), replace=False)
    cols = args.cols
    tw = 1280 // cols
    tiles, notes = [], []
    for i, j in enumerate(pick):
        p = paths[flat[int(j)][0]]
        k = flat[int(j)][1]
        data, samples = TDD.load_shard(p)
        s = samples[k]
        cv = TDD.canvas(data, s)
        im = draw_sample(cv, s, tw / s["w"])
        ImageDraw.Draw(im).text((2, 1), str(i + 1), fill=(255, 255, 0))
        tiles.append(im)
        notes.append({"n": i + 1, "shard": p.name, "index": k, "src": s["src"], "size": [s["w"], s["h"]],
                      "lines": len(s["lines"]), "ignore": len(s["ignore"]),
                      "roles": {r: sum(1 for ln in s["lines"] if ln[5] == r) for r in CLASSES},
                      "vertical": sum(ln[4] for ln in s["lines"]), "elements": s["meta"].get("elements"),
                      "capture": s["meta"].get("capture"), "scene": s["meta"].get("scene")})
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    H = sum(max(t.height for t in r) + 3 for r in rows)
    sheet = Image.new("RGB", (1280, H), (40, 0, 40))
    y = 0
    for r in rows:
        for j, t in enumerate(r):
            sheet.paste(t, (j * tw, y))
        y += max(t.height for t in r) + 3
    sheet.thumbnail((1280, 1280))
    sheet.save(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(notes, ensure_ascii=False, indent=0) + "\n",
                                                   encoding="utf-8")
    print(args.out, sheet.size)


def shard_stats(idx):
    by_kind, by_size, roles, orient = {}, {}, {r: 0 for r in CLASSES}, {"horizontal": 0, "vertical": 0}
    ignore = small = 0
    for s in idx:
        k = s["src"].split(":", 1)[1]
        by_kind[k] = by_kind.get(k, 0) + 1
        z = f"{s['w']}x{s['h']}"
        by_size[z] = by_size.get(z, 0) + 1
        for ln in s["lines"]:
            roles[ln[5]] += 1
            orient["vertical" if ln[4] else "horizontal"] += 1
        ignore += len(s["ignore"])
    return {"frames": len(idx), "by_kind": by_kind, "by_size": by_size, "lines": sum(roles.values()),
            "by_class": roles, "by_orientation": orient, "ignore": ignore, "small": small}


def _add(a, b):
    for k, v in b.items():
        if isinstance(v, dict):
            _add(a.setdefault(k, {}), v)
        else:
            a[k] = a.get(k, 0) + v


def cmd_manifest(args):
    d = Path(args.dir)
    shards, totals = [], {}
    for p in sorted(d.glob("synth_frames_*.json")):
        if p.name.endswith(".tmp.json") or not p.with_suffix(".npy").exists():
            continue
        idx = json.loads(p.read_text(encoding="utf-8"))["samples"]
        st = shard_stats(idx)
        st.pop("small")
        shards.append({"dir": d.name, "name": p.stem, "npy_sha256": sha256(p.with_suffix(".npy")),
                       "json_sha256": sha256(p), "npy_bytes": p.with_suffix(".npy").stat().st_size, **st})
        _add(totals, {k: v for k, v in st.items()})
    pool = json.loads((d / "scene_pool.json").read_text(encoding="utf-8"))
    pool.pop("paths", None)
    real = None
    if args.real_manifest:
        rm = Path(args.real_manifest)
        man = json.loads(rm.read_text(encoding="utf-8"))
        real = {"manifest": str(rm.relative_to(ROOT)) if rm.is_absolute() else str(rm), "sha256": sha256(rm),
                "shards": sorted(f"{s['dir']}/{s['name']}" for s in man["shards"] if s["dir"] != "synth"),
                "samples": sum(s["samples"] for s in man["shards"] if s["dir"] != "synth"),
                "note": "#97's real whole-screenshot canvases (realplus, teacher, devnative; the owner's are private), "
                        "pinned there; #97's own synthetic shards (dir synth) are not used"}
        if args.real_stats:
            real["measured"] = json.loads(Path(args.real_stats).read_text(encoding="utf-8"))
    doc = {"kind": "textspot-shards", "schema": 1, "generator": "synth/textspot_data.py", "gen_version": GEN_VERSION,
           "issue": 104, "seed": args.seed, "per_shard": args.per_shard,
           "code_sha256": {f: sha256(ROOT / f) for f in CODE},
           "mix": MIX, "capture": CAPTURE, "min_across": MIN_ACROSS, "classes": list(CLASSES),
           "synth3": "render_region of synth3 version 3 (CFG frames None), region capture off, layout forced",
           "reserved_sha256": sha256(ROOT / "eval" / "manifests" / "reserved.json"),
           "scene_pool": pool, "real": real, "shards": shards, "totals": totals}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {totals.get('frames')} frames, {totals.get('lines')} lines")


def real_stats(paths):
    """#97's real canvases measured from their JSON: sizes and kinds, lines, vertical, ignore areas, the share of
    lines under 6 px and over 72 px across, and whole frames (the canvas is the whole screenshot at its read size)."""
    out = {}
    for p in paths:
        p = Path(p)
        idx = json.loads(p.read_text(encoding="utf-8"))["samples"]
        for s in idx:
            src = s["src"]
            o = out.setdefault(src, {"canvases": 0, "by_size": {}, "by_kind": {}, "lines": 0, "vertical": 0,
                                     "ignore": 0, "across_lt6": 0, "across_gt72": 0, "lines_hist": {},
                                     "whole_frame": 0, "not_whole": 0, "no_lines": 0, "vns": set()})
            o["canvases"] += 1
            z = f"{s['w']}x{s['h']}"
            o["by_size"][z] = o["by_size"].get(z, 0) + 1
            o["by_kind"][s["kind"]] = o["by_kind"].get(s["kind"], 0) + 1
            n = len(s["lines"])
            o["lines"] += n
            b = str(min(n, 20)) if n < 20 else "20+"
            o["lines_hist"][b] = o["lines_hist"].get(b, 0) + 1
            o["no_lines"] += n == 0
            o["vertical"] += sum(ln[4] for ln in s["lines"])
            o["ignore"] += len(s["ignore"])
            for ln in s["lines"]:
                a = (ln[2] - ln[0]) if ln[4] else (ln[3] - ln[1])
                o["across_lt6"] += a < 6
                o["across_gt72"] += a > 72
            m = s.get("meta") or {}
            sw, sh = (m.get("src_size") or [None, None])
            sc = m.get("scale")
            whole = sw is not None and sc is not None and abs(round(sw * sc) - s["w"]) <= 1 and \
                abs(round(sh * sc) - s["h"]) <= 1
            if sw is not None and sc is None:
                whole = True
            o["whole_frame" if whole else "not_whole"] += 1
            if s.get("vn"):
                o["vns"].add(s["vn"])
    for o in out.values():
        o["vns"] = len(o["vns"])
        o["lines_per_canvas"] = round(o["lines"] / max(1, o["canvases"]), 3)
        o["share_lt6"] = round(o["across_lt6"] / max(1, o["lines"]), 5)
        o["share_gt72"] = round(o["across_gt72"] / max(1, o["lines"]), 5)
    return out


def cmd_real_stats(args):
    paths = []
    for d in args.root:
        paths += sorted(p for p in Path(d).glob("*.json") if not p.name.endswith(".tmp.json") and
                        p.with_suffix(".npy").exists())
    st = real_stats(paths)
    Path(args.out).write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")
    for k, v in st.items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("by_size", "lines_hist")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--per-shard", type=int, default=1000)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    p = sub.add_parser("preview")
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--per-shard", type=int, default=1000)
    p.add_argument("--real-manifest", default=str(ROOT / "data" / "manifests" / "textdet_shards_v1.json"))
    p.add_argument("--real-stats")
    p = sub.add_parser("real-stats")
    p.add_argument("--root", action="append", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"synth": cmd_synth, "preview": cmd_preview, "manifest": cmd_manifest, "real-stats": cmd_real_stats}[args.cmd](
        args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Training data of the learned text detector (issue #96, docs/textdet.md): pinned shards of grey canvases with
line boxes and ignore areas.

  textdet_data.py synth --out DIR --shards N --per-shard M [--first F] [--seed S] [--workers W]
  textdet_data.py real --source realplus|teacher|owner|devnative --out DIR [--per-shard M] [--workers W]
  textdet_data.py manifest --dir DIR ... --out data/manifests/textdet_shards_v1.json
  textdet_data.py preview --dir DIR [--n 24] --out PNG                    contact sheet (<= 1280 px) with the boxes
  textdet_data.py rows --source NAME --out JSONL                          the real screenshots a source would use

A shard is DIR/<name>.npy (uint8 grey canvases, concatenated row-major) and DIR/<name>.json:
  {"samples": [{"off", "h", "w", "src", "kind", "vn", "shot", "lines": [[x0, y0, x1, y1, vertical, role]],
                "ignore": [[x0, y0, x1, y1]], "meta"}]}
Boxes are [x0, y0, x1) x [y0, y1) in canvas pixels. A line box is the line's main-text ink box: fill, outline and
shadow; furigana, emphasis dots, page cursors and icons are not part of it (they are negatives). Ignore areas carry
no loss (text the source did not label precisely).

Synthetic canvases (synth3 version 3's renderer, its scenes, fonts, texts and exclusions; render_region's ink and
ruby masks give the exact boxes):
  region    a rendered dialogue region as a player would select it (ADV boxes, bands, NVL pages, balloons, choices,
            bare text over a scene, UI menus; Vita, PSP, PSP 2x and scaled PC sizes)
  loose     a region inside a larger canvas of scene (a loose selection or most of a screen)
  vertical  vertical text columns (right to left) over a scene or synthetic background, sometimes with a
            horizontal line beside them: the orientation output's positives
  empty     a rendered region with its text inpainted away (an empty dialogue box), or a scene patch: no line

Real canvases (MeikiOCR line boxes refined to ink, teacher/geometry.refine; detector boxes with their vertical
flag): whole screenshots at the console's target size, every horizontal line a positive, vertical detector boxes
vertical positives, any other detected text an ignore area:
  realplus   realplus's reads of VNDB PC screenshots at the Vita target (at most 2 screenshots per VN)
  teacher    the teacher's reads (Vita, PSP and PC screenshots; PC ones at the Vita target, a third of them also at
             the PSP target; Screen AI's boxes that no line covers are ignore areas; at most 6 per VN)
  owner      the owner's console captures (owner pools v1 + v2; private; at most 80 per VN)
  devnative  devnative's native Vita and PSP screenshots
Excluded everywhere: every reserved.json test and dev_sample VN and test session, every screenshot of vita-test,
psp-test and the six eval/linefind manifests, every VN of eval/linefind's four detector-built sets, and the owner's
CHAOS;CHILD (v14018, the report-only native cohort).
"""
import argparse
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
for _p in (HERE, ROOT / "teacher", ROOT / "eval" / "linefind"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from common import dev_sample_vns, reserved_sessions, test_set_vns, to_gray  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIVATE = HERD / "shared" / "private"
SCREEN = {"vita": (960, 544), "psp": (480, 272)}
SYNTH_MIX = {"region": 0.45, "loose": 0.2, "vertical": 0.15, "empty": 0.2}
INK_MIN, INK_MAX = 8, 72          # a positive's ink size across the line, px; outside: ignore
CAPS = {"realplus": 2, "teacher": 6, "owner": 80, "devnative": 1000}
REPORT_ONLY_VNS = {"v14018"}
VERT_ROTATE = set("ー―‐－～〜…‥「」『』（）()【】〔〕［］〈〉《》｛｝<>-~=＝")
VERT_SHIFT = set("、。，．")
SMALL = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
URL_ID = re.compile(r"/(\d+)\.(?:jpg|png)$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose(rng, weights):
    keys = list(weights)
    p = np.asarray([weights[k] for k in keys], np.float64)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


# ----------------------------------------------------------------------------- synthetic

S3 = None


def init_synth():
    global S3
    import synth3
    cv2.setNumThreads(1)
    synth3.init()
    S3 = synth3
    return synth3


def main_box(L, layout, ink, ruby, shape):
    """(x0, y0, x1, y1) of a rendered line's main-text ink (fill, outline, shadow; its furigana excluded), or None:
    synth3.line_ink without the ruby extension."""
    H = shape[0]
    if L.get("ix0", -99) < 0:
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
    cols = np.flatnonzero(band.any(axis=0))
    return int(x0 + cols[0]), int(y0 + rows[0]), int(x0 + cols[-1] + 1), int(y0 + rows[-1] + 1)


def render(rng):
    """synth3's region: (rgb, lines [[x0, y0, x1, y1, 0, role]], ink | ruby mask, meta) or None."""
    r = S3.render_region(rng)
    if r is None:
        return None
    img, layout, ink, meta = r
    ruby = meta.pop("_ruby_mask")
    meta.pop("_fill_mask", None)
    lines = []
    for L in layout:
        b = main_box(L, layout, ink, ruby, img.shape)
        if b is not None and b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
            lines.append([*b, 0, "name" if L.get("kind") == "name" else "text"])
    return img, lines, ink | ruby, meta


def background(rng, w, h, target):
    arr, tag = S3.background(rng, w, h, target, target, 1.0)
    return np.clip(np.asarray(arr, np.float32), 0, 255), tag


def capture(rng, img):
    """Light capture artefacts on a composed canvas (synth3's own regions carry theirs already)."""
    out = np.asarray(img, np.float32)
    if rng.random() < 0.2:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 0.7)))
    if rng.random() < 0.2:
        out = out + rng.normal(0, rng.uniform(1, 4), out.shape).astype(np.float32)
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = S3.jpeg(out, int(rng.integers(70, 96)), 2, rng)
    return out


def sample_region(rng):
    r = render(rng)
    if r is None or not r[1]:
        return None
    img, lines, _, meta = r
    return {"canvas": to_gray(img), "lines": lines, "ignore": [],
            "meta": {k: meta[k] for k in ("platform", "target", "layout", "scheme", "cat", "capture")}}


def sample_loose(rng):
    r = render(rng)
    if r is None or not r[1]:
        return None
    img, lines, _, meta = r
    target = meta["target"]
    SW, SH = SCREEN[target]
    h, w = img.shape[:2]
    k = 0.5 if target == "psp" else 1.0
    top, bot = (int(rng.integers(4, int(160 * k) + 5)) for _ in range(2))
    left, right = (int(rng.integers(4, int(220 * k) + 5)) for _ in range(2))
    H2, W2 = min(SH, h + top + bot), min(SW, w + left + right)
    if H2 < h or W2 < w:
        return None
    oy, ox = int(rng.integers(0, H2 - h + 1)), int(rng.integers(0, W2 - w + 1))
    bg, tag = background(rng, W2, H2, target)
    bg[oy:oy + h, ox:ox + w] = img
    out = capture(rng, bg)
    moved = [[x0 + ox, y0 + oy, x1 + ox, y1 + oy, v, role] for x0, y0, x1, y1, v, role in lines]
    return {"canvas": to_gray(out), "lines": moved, "ignore": [],
            "meta": {"platform": meta["platform"], "target": target, "layout": meta["layout"], "bg": tag}}


def _glyph(ch, font, size, rotate):
    """(fill mask, stroke-grown mask) of one glyph centred in a 2*size square (uint8), rotated 90 deg if asked."""
    from PIL import Image, ImageDraw
    n = 2 * size
    im = Image.new("L", (n, n), 0)
    ImageDraw.Draw(im).text((n / 2, n / 2), ch, font=font, fill=255, anchor="mm")
    if rotate:
        im = im.rotate(-90)
    return np.asarray(im)


def sample_vertical(rng):
    """Vertical text columns (right to left), and sometimes a horizontal line beside them."""
    target = "psp" if rng.random() < 0.35 else "vita"
    SW, SH = SCREEN[target]
    fs = int(rng.integers(11, 19)) if target == "psp" else int(rng.integers(16, 36))
    text = ""
    for _ in range(3):
        t, _, _ = (S3.narration_text(rng) if rng.random() < 0.4 else S3.dialogue_text(rng, False))
        text += t
        if len(text) > 30:
            break
    text = text.replace("　", "")
    if len(text) < 2:
        return None
    font = S3.pick_font(rng, set(text), target)
    if font is None or font.cat == "pixel":
        return None
    from render import get_font, pick_weight
    f = get_font(font.path, fs, pick_weight(rng, font.path))
    pitch = fs + int(rng.integers(0, max(2, fs // 4) + 1))
    gap = int(round(fs * rng.uniform(0.35, 1.1)))
    cpc = int(rng.integers(3, 16))
    ncol = int(min(rng.integers(1, 7), math.ceil(len(text) / cpc)))
    cols = [text[i * cpc:(i + 1) * cpc] for i in range(ncol)]
    m = int(rng.integers(4, 3 * fs))
    H = min(SH, 2 * m + cpc * pitch)
    W = min(SW, 2 * m + ncol * (fs + gap) + int(rng.integers(0, 6 * fs)))
    if H < 2 * fs or W < fs + 8:
        return None
    bg, tag = background(rng, W, H, target)
    fill = np.zeros((H, W), np.float32)
    lines = []
    x_right = W - m
    for c, col in enumerate(cols):
        cx = x_right - c * (fs + gap) - fs / 2
        if cx - fs / 2 < 0:
            break
        colmask = np.zeros((H, W), np.float32)
        for k, ch in enumerate(col):
            cy = m + k * pitch + fs / 2
            if cy + fs / 2 > H:
                break
            g = _glyph(ch, f, fs, ch in VERT_ROTATE).astype(np.float32) / 255
            dx = dy = 0.0
            if ch in VERT_SHIFT:
                dx, dy = 0.55 * fs, -0.55 * fs
            elif ch in SMALL:
                dx, dy = 0.1 * fs, -0.1 * fs
            x0, y0 = int(round(cx + dx - fs)), int(round(cy + dy - fs))
            gy0, gx0 = max(0, -y0), max(0, -x0)
            gy1, gx1 = min(2 * fs, H - y0), min(2 * fs, W - x0)
            if gy1 <= gy0 or gx1 <= gx0:
                continue
            sub = colmask[y0 + gy0:y0 + gy1, x0 + gx0:x0 + gx1]
            np.maximum(sub, g[gy0:gy1, gx0:gx1], out=sub)
        np.maximum(fill, colmask, out=fill)
        ys, xs = np.nonzero(colmask >= 0.5)
        if len(ys):
            lines.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, 1, "text"])
    if not lines:
        return None
    horiz = None
    if rng.random() < 0.3 and lines[-1][0] > 4 * fs:          # a horizontal line in the free space on the left
        from PIL import Image, ImageDraw
        t = text[:int(rng.integers(2, 12))]
        im = Image.new("L", (W, H), 0)
        hy = int(rng.integers(m, max(m + 1, H - fs - m)))
        ImageDraw.Draw(im).text((m, hy), t, font=f, fill=255)
        hm = np.asarray(im, np.float32) / 255
        hm[:, lines[-1][0] - gap // 2:] = 0
        ys, xs = np.nonzero(hm >= 0.5)
        if len(ys) and xs.max() - xs.min() > fs:
            np.maximum(fill, hm, out=fill)
            horiz = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, 0, "text"]
    light = float(bg.mean()) < 128 if rng.random() < 0.8 else rng.random() < 0.5
    tc = np.array([rng.uniform(215, 255)] * 3 if light else [rng.uniform(0, 50)] * 3, np.float32)
    out = bg.copy()
    if rng.random() < 0.35:                                 # an outline in the other polarity
        r = 1 if fs < 20 else 2
        grown = cv2.dilate(fill, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        oc = np.array([rng.uniform(0, 40)] * 3 if light else [rng.uniform(220, 255)] * 3, np.float32)
        out = out * (1 - grown[..., None]) + oc * grown[..., None]
        for ln in lines + ([horiz] if horiz else []):
            ln[0], ln[1] = max(0, ln[0] - r), max(0, ln[1] - r)
            ln[2], ln[3] = min(W, ln[2] + r), min(H, ln[3] + r)
    out = out * (1 - fill[..., None]) + tc * fill[..., None]
    out = capture(rng, out)
    if horiz:
        lines.append(horiz)
    return {"canvas": to_gray(out), "lines": lines, "ignore": [],
            "meta": {"target": target, "fs": fs, "cols": len(lines), "bg": tag, "font": font.fam}}


def sample_empty(rng):
    if rng.random() < 0.5:
        r = render(rng)
        if r is None:
            return None
        img, lines, mask, meta = r
        if meta.get("partial"):
            return None
        m = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        out = cv2.inpaint(np.ascontiguousarray(img), m, 3, cv2.INPAINT_TELEA)
        return {"canvas": to_gray(out), "lines": [], "ignore": [],
                "meta": {"empty": "inpainted", "target": meta["target"], "layout": meta["layout"]}}
    target = "psp" if rng.random() < 0.35 else "vita"
    SW, SH = SCREEN[target]
    k = 0.5 if target == "psp" else 1.0
    h = int(min(SH, rng.integers(int(30 * k), int(320 * k))))
    w = int(min(SW, rng.integers(int(100 * k), int(960 * k))))
    bg, tag = background(rng, w, h, target)
    return {"canvas": to_gray(capture(rng, bg)), "lines": [], "ignore": [], "meta": {"empty": tag, "target": target}}


SYNTH = {"region": sample_region, "loose": sample_loose, "vertical": sample_vertical, "empty": sample_empty}


def run_synth_shard(job):
    idx, n, seed, out_dir = job
    path = Path(out_dir) / f"synth_{idx:04d}"
    if path.with_suffix(".json").exists():
        return idx, 0, 0, 0.0
    rng = np.random.default_rng([seed, idx])
    t0, samples, errors = time.time(), [], 0
    while len(samples) < n:
        kind = choose(rng, SYNTH_MIX)
        try:
            s = SYNTH[kind](rng)
        except Exception as e:  # noqa: BLE001 - a rare bad draw must not end a long render
            errors += 1
            if errors <= 3:
                print(f"shard {idx}: {kind} error {type(e).__name__}: {e}", flush=True)
            if errors > 500:
                raise
            continue
        if s is None:
            continue
        s.update(src=f"synth:{kind}", kind=s["meta"].get("target"), vn=None, shot=None)
        samples.append(s)
    write_shard(path, samples)
    return idx, len(samples), errors, time.time() - t0


# ----------------------------------------------------------------------------- real

def _eval_exclusions():
    """Screenshot image ids and sha256 of vita-test, psp-test and eval/linefind, and the linefind sets' VNs."""
    ids, shas, vns = set(), set(), set()
    for name in ("vita-test", "psp-test"):
        for x in (ROOT / "eval" / "manifests" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            shas.add(r["sha256"])
            m = URL_ID.search(r["url"])
            if m:
                ids.add(m.group(1))
    for name in ("research_a", "research_b", "vita_dev", "vita_holdout", "psp_dev", "psp_holdout"):
        for x in (ROOT / "eval" / "linefind" / "manifests" / f"{name}.jsonl").read_text().splitlines():
            r = json.loads(x)
            shas.add(r["sha256"])
            m = URL_ID.search(r.get("url") or "")
            if m:
                ids.add(m.group(1))
            if not name.startswith("research"):
                vns.add(r["vn"])
    return ids, shas, vns


def _excluded(r, ex):
    ids, shas, lf_vns, test, dev, sessions = ex
    vn = r.get("vn")
    if vn in test or vn in dev or vn in lf_vns or vn in REPORT_ONLY_VNS:
        return True
    rel = r.get("release") or ""
    if rel in sessions or any(s in sessions for s in re.findall(r"r\d+", r.get("session") or "")):
        return True
    m = URL_ID.search(r.get("url") or "")
    if m and m.group(1) in ids:
        return True
    return r.get("sha256") in shas


def _cap(rows, cap):
    """At most cap rows per VN, chosen by the hash of the shot id (deterministic)."""
    by = {}
    for r in sorted(rows, key=lambda r: hashlib.sha256(r["shot"].encode()).hexdigest()):
        lst = by.setdefault(r["vn"], [])
        if len(lst) < cap:
            lst.append(r)
    return sorted((r for v in by.values() for r in v), key=lambda r: r["shot"])


def _jsonl(path):
    for x in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(x)
        except json.JSONDecodeError:
            continue


def real_rows(source):
    """The real screenshots of one source after the exclusions and the per-VN cap: reads rows (latest per shot)."""
    ids, shas, lf_vns = _eval_exclusions()
    ex = (ids, shas, lf_vns, set(test_set_vns()), set(dev_sample_vns()), set(reserved_sessions()))
    rows = {}
    if source == "realplus":
        for r in _jsonl(ART / "realplus" / "reads" / "reads.jsonl"):
            if r.get("dets") is not None:
                rows[r["shot"]] = r
        meta = {r["shot"]: r for r in _jsonl(ART / "realplus" / "reads" / "shots.jsonl")}
        for k, r in rows.items():
            m = meta.get(k) or {}
            for f in ("sha256", "session", "release", "url"):
                r.setdefault(f, m.get(f))
    elif source == "teacher":
        import glob
        for p in sorted(glob.glob(str(ART / "teacher" / "reads" / "train.*of5.jsonl"))):
            for r in _jsonl(p):
                r["path"] = str(ART / "realdata" / "shots" / r["kind"] / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}")
                rows[r["shot"]] = r
    elif source == "owner":
        for p in (PRIVATE / "owner-games" / "_build" / "owner-pool" / "reads" / "reads.jsonl",
                  PRIVATE / "owner-games" / "_build2" / "owner-pool-2" / "reads" / "reads.jsonl"):
            for r in _jsonl(p):
                if r.get("dets") is not None:
                    rows[r["shot"]] = r
        meta = {r["shot"]: r for r in _jsonl(PRIVATE / "owner-games" / "_build2" / "owner-pool-2" / "reads" /
                                             "shots.jsonl")}
        for k, r in rows.items():
            for f in ("session", "release", "url"):
                r.setdefault(f, (meta.get(k) or {}).get(f))
    elif source == "devnative":
        for r in _jsonl(ART / "devnative" / "reads" / "reads.jsonl"):
            rows[r["shot"]] = r
    else:
        raise SystemExit(f"unknown source {source}")
    kept = [r for r in rows.values() if not _excluded(r, ex)]
    return _cap(kept, CAPS[source]), len(rows)


def _covered(b, boxes, frac=0.5):
    area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    for c in boxes:
        ix = max(0, min(b[2], c[2]) - max(b[0], c[0]))
        iy = max(0, min(b[3], c[3]) - max(b[1], c[1]))
        if ix * iy >= frac * area:
            return True
    return False


def _scaled(b, s, W, H):
    x0, y0 = int(math.floor(b[0] * s)), int(math.floor(b[1] * s))
    x1, y1 = int(math.ceil(b[2] * s)), int(math.ceil(b[3] * s))
    return [max(0, x0), max(0, y0), min(W, x1), min(H, y1)]


def real_canvases(r, source, rng):
    """[(grey canvas, lines, ignore, meta)] of one screenshot: one per target the source reads it at."""
    import geometry as G
    bgr = cv2.imread(str(r["path"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(r["path"])
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    sh, sw = rgb.shape[:2]
    if source in ("realplus", "owner", "devnative"):
        # these reads are on the screenshot at its read size (dims: the Vita target for PC shots), boxes in it
        dw, dh = (int(v) for v in r["dims"])
        plans = [("vita" if r["kind"] != "psp" else "psp", (dw, dh), 1.0)]
    else:
        # the teacher's boxes are in the source screenshot's pixels; PC shots are read at their target scale
        if r["kind"] in ("vita", "psp"):
            plans = [(r["kind"], (sw, sh), 1.0)]
        else:
            plans = []
            for t in ("vita",) + (("psp",) if rng.random() < 1 / 3 else ()):
                s = G.target_scale(sw, sh, t)
                plans.append((t, (max(1, int(round(sw * s))), max(1, int(round(sh * s)))), s))
    out = []
    for target, (dw, dh), k in plans:
        # cv2.resize to the read size is teacher/geometry.scale_image's resize (INTER_AREA down, LINEAR up)
        img = rgb if (dw, dh) == (sw, sh) else cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA
                                                          if dw < sw else cv2.INTER_LINEAR)
        H, W = img.shape[:2]
        s = dw / sw
        lines, ignore, covers = [], [], []
        for ln in r.get("lines") or []:
            b = _scaled(ln["ink"], k, W, H)
            covers.append(b)
            for p in ln.get("parts") or []:
                covers.append(_scaled(p, k, W, H))
            for f in ln.get("furi") or []:
                covers.append(_scaled(f, k, W, H))
            hgt = b[3] - b[1]
            if b[2] - b[0] < 2 or hgt < 2:
                continue
            (lines if INK_MIN <= hgt <= INK_MAX else ignore).append(b)
        lines = [[*b, 0, "text"] for b in lines]
        for d in r.get("dets") or []:
            b = _scaled(d["det"], k, W, H)
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            if d.get("vertical"):
                wd = b[2] - b[0]
                if INK_MIN <= wd <= INK_MAX and b[3] - b[1] >= wd:
                    lines.append([*b, 1, "text"])
                else:
                    ignore.append(b)
            elif not _covered(b, covers):
                ignore.append(b)
        for s_ in r.get("sai") or []:
            b = _scaled(s_["box"], k, W, H)
            if b[2] - b[0] >= 2 and b[3] - b[1] >= 2 and not _covered(b, covers + [ln[:4] for ln in lines]):
                ignore.append(b)
        out.append((to_gray(img), lines, ignore, {"target": target, "scale": round(s, 4), "src_size": [sw, sh]}))
    return out


def run_real_shard(job):
    idx, rows, source, seed, out_dir = job
    path = Path(out_dir) / f"{source}_{idx:04d}"
    if path.with_suffix(".json").exists():
        return idx, 0, 0, 0.0
    rng = np.random.default_rng([seed, idx])
    t0, samples, errors = time.time(), [], 0
    for r in rows:
        try:
            for g, lines, ignore, meta in real_canvases(r, source, rng):
                samples.append({"canvas": g, "lines": lines, "ignore": ignore, "meta": meta, "src": source,
                                "kind": meta["target"], "vn": r["vn"], "shot": r["shot"]})
        except (OSError, FileNotFoundError, KeyError, ValueError) as e:
            errors += 1
            print(f"{source} {r.get('shot')}: {type(e).__name__}: {e}", flush=True)
    write_shard(path, samples)
    return idx, len(samples), errors, time.time() - t0


# ----------------------------------------------------------------------------- shards

def write_shard(path, samples):
    off, index = 0, []
    for s in samples:
        h, w = s["canvas"].shape
        index.append({"off": off, "h": int(h), "w": int(w), **{k: s.get(k) for k in ("src", "kind", "vn", "shot")},
                      "lines": [[int(v) if i < 5 else v for i, v in enumerate(ln)] for ln in s["lines"]],
                      "ignore": [[int(v) for v in b] for b in s["ignore"]], "meta": s.get("meta", {})})
        off += h * w
    data = np.concatenate([s["canvas"].ravel() for s in samples]) if samples else np.zeros(0, np.uint8)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, data)
    tmp.rename(path.with_suffix(".npy"))
    path.with_suffix(".json").write_text(json.dumps({"samples": index}, ensure_ascii=False) + "\n", encoding="utf-8")


def load_shard(path, mmap=True):
    """(data uint8 array, samples) of a shard path without suffix."""
    path = Path(path)
    data = np.load(path.with_suffix(".npy"), mmap_mode="r" if mmap else None)
    return data, json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["samples"]


def canvas(data, s):
    return np.asarray(data[s["off"]:s["off"] + s["h"] * s["w"]]).reshape(s["h"], s["w"])


# ----------------------------------------------------------------------------- commands

CODE = ("synth/textdet_data.py", "synth/synth3.py", "synth/render.py", "synth/common.py", "teacher/geometry.py")


def cmd_synth(args):
    init_synth()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.per_shard, args.seed, str(out)) for i in range(args.first, args.first + args.shards)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, errors, dt in pool.imap_unordered(run_synth_shard, jobs):
            print(f"synth shard {idx}: {n} samples ({errors} errors) in {dt:.0f}s ({time.time() - t0:.0f}s)",
                  flush=True)
    print("SYNTH_DONE", flush=True)


def cmd_real(args):
    rows, total = real_rows(args.source)
    print(f"{args.source}: {len(rows)} screenshots of {len({r['vn'] for r in rows})} VNs kept of {total}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = [rows[i:i + args.per_shard] for i in range(0, len(rows), args.per_shard)]
    jobs = [(i, c, args.source, args.seed, str(out)) for i, c in enumerate(chunks)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, errors, dt in pool.imap_unordered(run_real_shard, jobs):
            print(f"{args.source} shard {idx}: {n} canvases ({errors} errors) in {dt:.0f}s ({time.time() - t0:.0f}s)",
                  flush=True)
    print("REAL_DONE", flush=True)


def cmd_rows(args):
    rows, total = real_rows(args.source)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ("shot", "vn", "kind", "url", "release", "session", "sha256")})
                    + "\n")
    print(f"{args.source}: {len(rows)} of {total} -> {args.out}")


def cmd_manifest(args):
    shards = []
    for d in args.dir:
        for p in sorted(Path(d).glob("*.json")):
            if p.name.endswith(".tmp.json") or p.name in ("manifest.json", "rows.json"):
                continue
            data = p.with_suffix(".npy")
            if not data.exists():
                continue
            idx = json.loads(p.read_text(encoding="utf-8"))["samples"]
            src = {}
            for s in idx:
                src[s["src"]] = src.get(s["src"], 0) + 1
            shards.append({"dir": Path(d).name, "name": p.stem, "npy_sha256": sha256(data), "json_sha256": sha256(p),
                           "samples": len(idx), "lines": sum(len(s["lines"]) for s in idx),
                           "vertical": sum(ln[4] for s in idx for ln in s["lines"]), "by_src": src})
    doc = {"kind": "textdet-shards", "schema": 1, "code_sha256": {f: sha256(ROOT / f) for f in CODE},
           "synth_mix": SYNTH_MIX, "caps": CAPS, "ink_range": [INK_MIN, INK_MAX],
           "reserved_sha256": sha256(ROOT / "eval" / "manifests" / "reserved.json"),
           "report_only_vns": sorted(REPORT_ONLY_VNS), "shards": shards,
           "totals": {"samples": sum(s["samples"] for s in shards), "lines": sum(s["lines"] for s in shards),
                      "vertical": sum(s["vertical"] for s in shards)}}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {doc['totals']}")


def cmd_preview(args):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(args.seed)
    paths = sorted(p.with_suffix("") for p in Path(args.dir).glob("*.npy") if ".tmp" not in p.name)
    tiles = []
    for _ in range(args.n):
        data, samples = load_shard(paths[int(rng.integers(len(paths)))])
        s = samples[int(rng.integers(len(samples)))]
        im = Image.fromarray(canvas(data, s)).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in s["ignore"]:
            d.rectangle((b[0], b[1], b[2] - 1, b[3] - 1), outline=(255, 160, 0))
        for x0, y0, x1, y1, v, _ in s["lines"]:
            d.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 0, 255) if v else (0, 255, 0))
        im.thumbnail((620, 300))
        tiles.append(im)
    cols = 2
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    W = 1280
    H = sum(max(t.height for t in r) + 4 for r in rows)
    sheet = Image.new("RGB", (W, H), (60, 0, 60))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            sheet.paste(t, (x, y))
            x += 640
        y += max(t.height for t in r) + 4
    sheet.thumbnail((1280, 1280))
    sheet.save(args.out)
    print(args.out, sheet.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--per-shard", type=int, default=2000)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    p = sub.add_parser("real")
    p.add_argument("--source", required=True, choices=sorted(CAPS))
    p.add_argument("--out", required=True)
    p.add_argument("--per-shard", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    p = sub.add_parser("rows")
    p.add_argument("--source", required=True, choices=sorted(CAPS))
    p.add_argument("--out", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--dir", action="append", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("preview")
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"synth": cmd_synth, "real": cmd_real, "rows": cmd_rows, "manifest": cmd_manifest, "preview": cmd_preview}[
        args.cmd](args)


if __name__ == "__main__":
    main()

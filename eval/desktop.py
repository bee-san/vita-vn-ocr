#!/usr/bin/env python3
"""The desktop benchmark (issue #169, docs/desktop.md): a whole frame read the way a desktop app reads it, on ONNX
Runtime's CPU provider: a detector on the frame, every horizontal box it returns cut from the frame and read by a
recognizer (greedy CTC), the boxes of each dialogue unit scored as eval/pagepath.py scores a unit.

  desktop.py frames --set hval|dev|confirm|pc|test --out FRAMES.pkl [--protocol eval/desktop_protocol.json]
      the frames of a set and their dialogue units (frame px), no pixels (they are read from the shared box)
  desktop.py resample --frames FRAMES.pkl --height H --out FRAMES_H.pkl
      the same frames resampled to height H, the aspect kept (cv2.INTER_LINEAR up, INTER_AREA down; desktop
      windows of 720 to 2160 px), every unit's region, line boxes and cells scaled with them: CER per input
      resolution (report only; amendment 2)
  desktop.py detect --frames FRAMES.pkl --det SPEC --out BOXES.jsonl [--workers 5]
      a detector's boxes on every frame: SPEC = meiki (MeikiOCR 0.3.5's detector, threshold 0.5, its own fit to
      960 x 544) | ts8 (textspot8_exact.onnx on the whole frame, a frame larger than 960 x 544 fit to it first, as
      docs/onnx.md says) | desk:MODEL.onnx[@SIZE] (tools/vocr_desktop.py's detector, SIZE its input policy)
  desktop.py read --frames FRAMES.pkl --boxes BOXES.jsonl --rec MODEL.onnx --out RESULT.json [--pad auto|N]
                  [--rows ROWS.jsonl] [--workers 5]
      every box whose centre lies in a dialogue unit's region, cut from the frame (pad: MeikiOCR's boxes are grown
      by `auto` = round(4 H / 544) px, at least 2: 4 on a Vita frame, 2 on a native PSP one), read with its dark
      flag (vocr_line_dark) by a tools/vocr_onnx.py recognizer, greedy; scored
  desktop.py meiki --frames FRAMES.pkl --out RESULT.json [--rows ROWS.jsonl] [--workers 5]
      MeikiOCR 0.3.5 itself, detector and recognizer as its run_ocr runs them (its defaults: det_threshold 0.5,
      rec_threshold 0.1, punct_conf_factor 1.0, BGR input), every box's text kept with its box; scored the same way
  desktop.py compare --base A.json --arm B.json [--out C.json]
      the paired difference of dialogue CER per group, with eval/pagepath.py's cluster bootstrap (VN or title)
  desktop.py boxes-pr --frames FRAMES.pkl (hval or test) --boxes BOXES.jsonl --out PR.json [--pad auto|N|none]
      whole-page line boxes against every verified horizontal line of the page (dialogue and names, box_main for
      a line with furigana): a box matches a line when they overlap by at least half the smaller height vertically
      and half the smaller width horizontally (one to one, largest overlap first); a box that matches nothing and
      lies at least half inside an ignore area (ruby, a line dropped for geometry, conflict or label doubt, a
      vertical line, a renderer box no verified line holds) is not counted. Precision counts text the hook does not
      cover as false boxes, so it is a lower bound

Sets (no protected set: vita-test, psp-test and CLANNAD are never read here):
  hval     the hval pages of the hooked pools v2.1 (b2-ingest2, #166), every page's base dialogue regions (the page
           pool's regions: verified dialogue lines grouped as a player draws them, eval's margin), native frames:
           Vita 960 x 544, PSP 480 x 272
  dev      #119's whole-dialogue development units (base variant), without the VNs that pools v2.1 trains on (its
           titles' VNDB ids: 9 of #119's 93 VNs): "dev-v2". Frames rebuilt from the screenshots as #119 cut them
           (a PC screenshot scaled to fit 960 x 544, a PSP one at 480 x 272), each unit's region checked pixel for
           pixel against the unit's stored image
  confirm  textspot7's confirmation units (base variant) without the VNs pools v2.1 trains on (4 of 41): read once
           for the candidate set, at the end
  pc       the dev units cut from PC screenshots, read at the screenshot's own size (640 x 480 to 1920 x 1080): the
           units' regions, boxes and cells divided by the scale #119 cut them at
  test     the hooked test titles (index.json's test: PCSG00807, NPJH50882): refused unless --protocol names a
           protocol file that git has committed unchanged and whose attempt record does not exist yet

Scoring (eval/pagepath.py's, function for function): a unit's boxes in reading order (rows by vertical overlap,
left to right in a row), each box's text normalised (eval/metrics.norm) and joined, against the unit's reference
lines joined; line recall (whole lines / lines) and box precision (boxes that are not junk / boxes) from the lines'
inked cells (pagepath.ink_points). A vertical box is a box with no text for a horizontal recognizer.

Everything read here is private (copyrighted frames): FRAMES, BOXES and ROWS stay on the shared box; RESULT.json
holds counts only.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher", ROOT / "runtime", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PV = HERD / "shared" / "private"
POOL = ART / "b2-ingest2" / "pagepool" / "v2.1"
PAGES = ART / "b2-ingest2" / "pages"
VNDB_MAP = ART / "b2-ingest2" / "vndb_map_v2.json"
DEV_UNITS = ART / "vocr5-analysis" / "work" / "units.pkl"
CONFIRM_UNITS = PV / "vocr6-spotter3" / "confirm" / "units_v1.pkl"
CONFIRM_SHOTS = PV / "vocr6-spotter3" / "confirm" / "shots"
SHOT_DIRS = {"vita": [ART / "realdata" / "shots" / "vita", ART / "devnative" / "shots" / "vita"],
             "psp": [ART / "realdata" / "shots" / "psp", ART / "devnative" / "shots" / "psp"],
             "hires": [ART / "realdata" / "shots" / "hires", ART / "realplus" / "shots" / "hires"]}
ONNX_DIR = ART / "vocr-onnx" / "release"
SETS = ("hval", "dev", "confirm", "pc", "test")
SCHEMA = "vocr-desktop/1"


def sha256_file(p, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def hooked_vns():
    """Every VNDB VN id of the hooked pools v2.1's titles (b2-ingest2's map)."""
    m = json.loads(VNDB_MAP.read_text(encoding="utf-8"))["titles"]
    return {v for e in m.values() for v in e.get("vns") or []}


# ----------------------------------------------------------------------------- sets

def even_cells(box, text):
    """[[x0, y0, x1, y1]]: a line's box cut into one even cell per character (pagepath.dev_units' rule)."""
    import metrics
    x, y, w, h = box
    n = max(1, len(metrics.norm(text)))
    return [[x + k * w / n, y, x + (k + 1) * w / n, y + h] for k in range(n)]


def hooked_frames(split):
    """Frames of the page pool's pages of a split, with their base dialogue units (pagepath.hooked_units)."""
    import pagepath as PP
    units, meta = PP.hooked_units(POOL, {split})
    by = defaultdict(list)
    for u in units:
        if u["variant"] == "base":
            by[(u["title"], u["page"])].append(u)
    frames = []
    for (tid, page), us in sorted(by.items()):
        u0 = us[0]
        W, H = (960, 544) if u0["platform"] == "vita" else (480, 272)
        frames.append({"key": f"{tid}:{page}", "set": split, "view": u0["platform"], "cluster": tid,
                       "src": {"kind": "hooked", "image": f"{tid}/pages/{page}.png", "sha256": u0["image_sha256"]},
                       "W": W, "H": H,
                       "units": [{"uid": u["uid"], "cluster": tid, "group": u0["platform"], "region": u["region"],
                                  "lines": [{"text": ln["text"], "box": [float(v) for v in ln["box"]],
                                             "cells": ln["cells"]} for ln in u["lines"]]} for u in us]})
    return frames, meta


def _find_shot(shot, src_kind):
    for d in SHOT_DIRS[src_kind]:
        for p in sorted(glob.glob(f"{d}/{shot}.*")):
            return p
    raise FileNotFoundError(f"{shot}: no screenshot in {SHOT_DIRS[src_kind]}")


def _region_units(us, scale=1.0):
    """#119-style units (region [x0, y0, x1, y1] and line boxes [x0, y0, x1, y1] in region px) -> desktop units in
    frame px, every coordinate divided by `scale` (1: the frame the unit was cut from)."""
    out = []
    for u in us:
        x0, y0, x1, y1 = u["region"]
        lines = []
        for ln in u["lines"]:
            a0, b0, a1, b1 = ln["box"]
            box = [(x0 + a0) / scale, (y0 + b0) / scale, (a1 - a0) / scale, (b1 - b0) / scale]
            lines.append({"text": ln["text"], "box": box, "cells": even_cells(box, ln["text"])})
        region = [int(math.floor(x0 / scale)), int(math.floor(y0 / scale)), int(math.ceil(x1 / scale)),
                  int(math.ceil(y1 / scale))]
        out.append({"uid": u["_uid"], "cluster": u["vn"], "group": u["_group"], "region": region, "lines": lines})
    return out


def _check_regions(rgb, us):
    from common import to_gray
    g = to_gray(rgb)
    for u in us:
        x0, y0, x1, y1 = u["region"]
        if g[y0:y1, x0:x1].shape != u["image"].shape or not (g[y0:y1, x0:x1] == u["image"]).all():
            raise SystemExit(f"{u['unit']}: the rebuilt frame's region is not the unit's stored image")


def dev_frames(pc=False):
    """dev-v2's frames as #119 cut its units (pc=False), or its PC screenshots' units at their own size (pc=True)."""
    import cv2
    import geometry as G
    d = pickle.load(open(DEV_UNITS, "rb"))
    skip = hooked_vns()
    units = []
    for i, u in enumerate(d["units"]):
        if u["variant"] != "base" or u["vn"] in skip:
            continue
        u = dict(u, _uid=f"dev:{i}:{u['unit']}:{u['variant']}")
        u["_group"] = ("vita-pc" if u["src_kind"] == "hires" else "vita") if u["kind"] == "vita" else \
            ("psp-hires" if u["src_kind"] == "hires" else "psp")
        units.append(u)
    by = defaultdict(list)
    for u in units:
        by[(u["shot"], u["src_kind"], round(float(u["scale"]), 6), u["kind"])].append(u)
    frames = []
    for (shot, src_kind, scale, kind), us in sorted(by.items()):
        if pc and not (src_kind == "hires" and kind == "vita" and scale < 1.0):
            continue
        p = _find_shot(shot, src_kind)
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        native_h, native_w = rgb.shape[:2]
        if src_kind == "hires":
            rgb = np.ascontiguousarray(G.scale_image(rgb, float(us[0]["scale"])))
        _check_regions(rgb, us)
        if pc:
            for u in us:
                u["_group"] = "pc"
            view, W, H, s = "pc", native_w, native_h, float(us[0]["scale"])
        else:
            view, W, H, s = kind, rgb.shape[1], rgb.shape[0], 1.0
        frames.append({"key": f"{shot}|{src_kind}|{scale}|{kind}", "set": "pc" if pc else "dev", "view": view,
                       "cluster": us[0]["vn"], "W": W, "H": H,
                       "src": {"kind": "dev", "path": str(p), "sha256": sha256_file(p), "src_kind": src_kind,
                               "scale": None if pc else (float(us[0]["scale"]) if src_kind == "hires" else None)},
                       "units": _region_units(us, s)})
    return frames, {"dev_units_sha256": sha256_file(DEV_UNITS), "excluded_vns": sorted(
        {u["vn"] for u in d["units"]} & skip)}


def confirm_frames():
    import cv2
    d = pickle.load(open(CONFIRM_UNITS, "rb"))
    skip = hooked_vns()
    fetched = {r["id"]: r for r in read_jsonl(CONFIRM_SHOTS / "fetched.jsonl")} if (
        CONFIRM_SHOTS / "fetched.jsonl").exists() else {}
    units = []
    for i, u in enumerate(d["units"]):
        if u["variant"] != "base" or u["vn"] in skip:
            continue
        units.append(dict(u, _uid=f"confirm:{i}:{u['unit']}:{u['variant']}", _group=u["kind"]))
    by = defaultdict(list)
    for u in units:
        by[u["shot"]].append(u)
    frames = []
    for shot, us in sorted(by.items()):
        f = fetched.get(shot)
        p = CONFIRM_SHOTS / f["file"] if f else next(iter(sorted(CONFIRM_SHOTS.glob(f"{shot}.*"))))
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        _check_regions(rgb, us)
        frames.append({"key": shot, "set": "confirm", "view": us[0]["kind"], "cluster": us[0]["vn"],
                       "W": rgb.shape[1], "H": rgb.shape[0],
                       "src": {"kind": "confirm", "path": str(p), "sha256": sha256_file(p)},
                       "units": _region_units(us)})
    return frames, {"confirm_units_sha256": sha256_file(CONFIRM_UNITS), "excluded_vns": sorted(
        {u["vn"] for u in d["units"]} & skip)}


def protocol_gate(protocol):
    """The test titles are read only under a protocol that git has committed unchanged, once (its attempt
    record, outside git, must not exist; this call writes it)."""
    if not protocol:
        raise SystemExit("the test set needs --protocol (a committed eval/desktop_protocol.json)")
    p = Path(protocol).resolve()
    rel = p.relative_to(ROOT)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=ROOT, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(rel)], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    if tracked.returncode or dirty:
        raise SystemExit(f"{rel}: not committed unchanged")
    doc = json.loads(p.read_text(encoding="utf-8"))
    att = Path(doc["test"]["attempt_record"])
    if att.exists():
        raise SystemExit(f"the test titles were read already ({att})")
    att.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    att.write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "head": head,
                               "protocol_sha256": sha256_file(p)}) + "\n")
    return doc


def build_frames(name, protocol=None):
    if name == "hval":
        return hooked_frames("hval")
    if name == "test":
        protocol_gate(protocol)
        return hooked_frames("test")
    if name == "dev":
        return dev_frames()
    if name == "pc":
        return dev_frames(pc=True)
    if name == "confirm":
        return confirm_frames()
    raise SystemExit(f"unknown set {name!r}")


def load_rgb(fr):
    """A frame's RGB pixels at its view's size (and its resample, for a per-resolution set)."""
    import cv2
    import geometry as G
    s = fr["src"]
    path = PAGES / s["image"] if s["kind"] == "hooked" else Path(s["path"])
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    if s.get("scale"):
        rgb = np.ascontiguousarray(G.scale_image(rgb, float(s["scale"])))
    rs = fr.get("resample")
    if rs:
        if rgb.shape[:2] != (rs["from"][1], rs["from"][0]):
            raise SystemExit(f"{fr['key']}: {rgb.shape[1]}x{rgb.shape[0]} before the resample, expected {rs['from']}")
        interp = cv2.INTER_AREA if rs["to"][1] < rs["from"][1] else cv2.INTER_LINEAR
        rgb = np.ascontiguousarray(cv2.resize(rgb, tuple(rs["to"]), interpolation=interp))
    if rgb.shape[:2] != (fr["H"], fr["W"]):
        raise SystemExit(f"{fr['key']}: {rgb.shape[1]}x{rgb.shape[0]}, expected {fr['W']}x{fr['H']}")
    return rgb


def check_sha(fr):
    s = fr["src"]
    path = PAGES / s["image"] if s["kind"] == "hooked" else Path(s["path"])
    if sha256_file(path) != s["sha256"]:
        raise SystemExit(f"{path}: sha256 differs from the frames file's")


def load_frames(path):
    with open(path, "rb") as f:
        doc = pickle.load(f)
    return doc["frames"], doc["meta"]


# ----------------------------------------------------------------------------- detectors

def session_options(threads=1, opt="all", spinning=None, profile=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    so.graph_optimization_level = {"disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                                   "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                   "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                                   "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[opt]
    if spinning is not None:
        so.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
    if profile:
        so.enable_profiling = True
        so.profile_file_prefix = str(profile)
    return so


def meiki_object(threads=1, opt="all", spinning=None, rec=True):
    """A meikiocr.MeikiOCR on the pinned files (teacher/engines.PINS), with our session options."""
    import engines as E
    import onnxruntime as ort
    from meikiocr import MeikiOCR
    m = MeikiOCR.__new__(MeikiOCR)
    mk = lambda name: ort.InferenceSession(str(E.model_path(name)), session_options(threads, opt, spinning),  # noqa
                                           providers=["CPUExecutionProvider"])
    m.det_session = mk("meiki_det")
    if rec:
        m.rec_session = mk("meiki_rec")
        m.vrec_session = mk("meiki_vrec")
    m.active_provider = "CPUExecutionProvider"
    m.max_batch_size = 8
    return m


class MeikiDet:
    """MeikiOCR 0.3.5's detector as its run_detection runs it (BGR, fit to 960 x 544 with INTER_LINEAR, /255, the
    boxes with score > thr clipped and cast to int as it casts them, sorted by top edge)."""

    name = "meiki"

    def __init__(self, threads=1, thr=0.5, opt="all", spinning=None, m=None):
        self.m = m or meiki_object(threads, opt, spinning, rec=False)
        self.thr = thr

    def boxes(self, rgb):
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        m = self.m
        tensor, scale = m._preprocess_for_detection(bgr)
        _, b, s = m._run_detection_inference(tensor, scale)
        b, s = b[0], s[0]
        keep = s > self.thr
        h, w = bgr.shape[:2]
        b = np.clip(b[keep], 0, np.array([w, h, w, h])).astype(np.int32)
        s = s[keep]
        order = np.argsort(b[:, 1], kind="stable")
        out = []
        for i in order:
            x0, y0, x1, y1 = (int(v) for v in b[i])
            if x1 <= x0 or y1 <= y0:
                continue
            out.append([x0, y0, x1, y1, int(y1 - y0 > x1 - x0), round(float(s[i]), 4), -1])
        return out


class Ts8Det:
    """textspot8 (tools/vocr_onnx.Spotter on textspot8_exact.onnx: the C runtime's codes) on the whole frame. A frame
    larger than 960 x 544 is first fit to it (cv2.INTER_AREA), as docs/onnx.md tells apps to; boxes map back."""

    name = "ts8"

    def __init__(self, threads=1, opt="all", spinning=None, path=None):
        import vocr_onnx as V
        self.sp = V.Spotter(path or ONNX_DIR / "textspot8_exact.onnx",
                            session_options=session_options(threads, opt, spinning))
        self.V = V

    def boxes(self, rgb):
        import cv2
        g = self.V.grey_of(rgb)
        H, W = g.shape
        f = min(1.0, 960 / W, 544 / H)
        if f < 1.0:
            g = cv2.resize(g, (max(1, int(round(W * f))), max(1, int(round(H * f)))), interpolation=cv2.INTER_AREA)
        sp = self.sp.spot(g)
        out = []
        for b in sp.boxes:
            out.append([b.x / f, b.y / f, (b.x + b.w) / f, (b.y + b.h) / f, int(b.vertical), int(b.conf), int(b.tag)])
        return out


class CDeskDet:
    """The C path's detector (desktop/libvocrdesk.so through tools/vocr_desktop.CModel) on a VDSK blob."""

    name = "cdesk"

    def __init__(self, vdsk, size=None):
        import vocr_desktop as D
        self.m, self.size = D.CModel(vdsk), size

    def boxes(self, rgb):
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag] for b in
                self.m.detect(rgb, self.size)]


def detector(spec, threads=1, opt="all", spinning=None):
    if spec == "meiki":
        return MeikiDet(threads, opt=opt, spinning=spinning)
    if spec == "ts8":
        return Ts8Det(threads, opt, spinning)
    if spec.startswith("desk:"):
        import vocr_desktop as D
        path, _, rest = spec[5:].partition("@")
        size, _, pad = rest.partition("@")
        return D.Detector(path, session_options=session_options(threads, opt, spinning), size=size or None,
                          pad=pad or None)
    if spec.startswith("cdesk:"):
        import vocr_desktop as D
        path, _, size = spec[6:].partition("@")
        return CDeskDet(path, size or None)
    raise SystemExit(f"unknown detector {spec!r}")


# ----------------------------------------------------------------------------- reading and scoring

def pad_of(spec, H):
    """The pad MeikiOCR's boxes get before they are read (--pad): auto = round(4 H / 544), at least 2."""
    if spec == "auto":
        return max(2, int(round(4 * H / 544)))
    return int(spec)


def crop_box(b, pad, W, H):
    """[x0, y0, x1, y1] (floats) grown by pad, rounded outward, clipped: (x, y, w, h) or None."""
    x0, y0 = max(0, int(math.floor(b[0] - pad))), max(0, int(math.floor(b[1] - pad)))
    x1, y1 = min(W, int(math.ceil(b[2] + pad))), min(H, int(math.ceil(b[3] + pad)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def reading_order(boxes):
    """Indices of (x, y, w, h, ...) boxes in reading order: rows top to bottom (a box joins a row when it overlaps
    the row's first box vertically by at least half the smaller height), left to right within a row."""
    idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3] / 2.0, boxes[i][0]))
    rows = []
    for i in idx:
        x, y, w, h = boxes[i][:4]
        if rows:
            fx, fy, fw, fh = boxes[rows[-1][0]][:4]
            if min(fy + fh, y + h) - max(fy, y) >= 0.5 * min(fh, h):
                rows[-1].append(i)
                continue
        rows.append([i])
    return [i for r in rows for i in sorted(r, key=lambda j: boxes[j][0])]


def unit_boxes(u, crops):
    """The crop boxes (x, y, w, h, k) whose centre lies in the unit's region, in reading order."""
    x0, y0, x1, y1 = u["region"]
    inside = [c for c in crops if x0 <= c[0] + c[2] / 2.0 < x1 and y0 <= c[1] + c[3] / 2.0 < y1]
    return [inside[i] for i in reading_order(inside)]


def score_frame_units(fr, grey, crops, texts):
    """pagepath.score_unit for every unit of a frame. crops: [(x, y, w, h, k)] frame px (k indexes texts)."""
    import pagepath as PP
    rows = []
    for u in fr["units"]:
        x0, y0, x1, y1 = u["region"]
        sub = grey[y0:y1, x0:x1]
        cells = [[[c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0] for c in ln["cells"]] for ln in u["lines"]]
        lboxes = [[ln["box"][0] - x0, ln["box"][1] - y0, ln["box"][2], ln["box"][3]] for ln in u["lines"]]
        points = PP.ink_points(sub, cells, lboxes)
        ub = unit_boxes(u, crops)
        rb = [(c[0] - x0, c[1] - y0, c[2], c[3]) for c in ub]
        tx = [texts[c[4]] for c in ub]
        r = PP.score_unit(u, rb, tx, points)
        r.update(uid=u["uid"], cluster=u["cluster"], group=u["group"], set=fr["set"], view=fr["view"])
        r["texts"] = tx
        r["boxes_xywh"] = [list(c[:4]) for c in ub]
        rows.append(r)
    return rows


def summarize(rows):
    import pagepath as PP
    out = {"all": PP.summarize(rows)["all"]}
    out["by_view"] = PP.summarize(rows, lambda r: f"{r['set']}:{r['view']}")
    out["by_group"] = PP.summarize(rows, lambda r: f"{r['set']}:{r['group']}")
    return out


_W = {}


def _init_read(rec_path, threads):
    import vocr_onnx as V
    _W["rec"] = V.Recognizer(rec_path, session_options=session_options(threads))
    _W["V"] = V


def _read_job(job):
    fr, boxes, pad = job
    V = _W["V"]
    rgb = load_rgb(fr)
    g = V.grey_of(rgb)
    H, W = g.shape
    p = pad_of(pad, H) if pad is not None else 0
    want = set()
    for u in fr["units"]:
        x0, y0, x1, y1 = u["region"]
        for k, b in enumerate(boxes):
            cb = crop_box(b, p, W, H)
            if cb and x0 <= cb[0] + cb[2] / 2.0 < x1 and y0 <= cb[1] + cb[3] / 2.0 < y1:
                want.add(k)
    crops, texts = [], []
    for k, b in enumerate(boxes):
        if k not in want:
            continue
        cb = crop_box(b, p, W, H)
        t = ""
        if not b[4]:
            dark = V.line_dark(g, cb)
            t = _W["rec"].read(g[cb[1]:cb[1] + cb[3], cb[0]:cb[0] + cb[2]], invert=bool(dark)).text
        crops.append((*cb, len(texts)))
        texts.append(t)
    return score_frame_units(fr, g, crops, texts)


def _init_meiki(threads):
    _W["meiki"] = meiki_object(threads)


def meiki_ocr(m, bgr, det_thr=0.5, rec_thr=0.1, punct=1.0):
    """meikiocr.MeikiOCR.run_ocr (0.3.5) line for line, keeping each result's detector box."""
    text_boxes = m.run_detection(bgr, det_thr)
    if not text_boxes:
        return []
    results = [{"text": "", "chars": [], "is_vertical": False} for _ in range(len(text_boxes))]
    hi, vi = [], []
    for i, tb in enumerate(text_boxes):
        x1, y1, x2, y2 = tb["bbox"]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        (vi if h > w else hi).append(i)
    if hi:
        m._process_recognition_pipeline(bgr, text_boxes, hi, results, rec_thr, punct, False)
    if vi:
        m._process_recognition_pipeline(bgr, text_boxes, vi, results, rec_thr, punct, True)
    return [(tb["bbox"], r["text"], bool(r["is_vertical"])) for tb, r in zip(text_boxes, results)]


def _meiki_job(fr):
    import vocr_onnx as V
    rgb = load_rgb(fr)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    res = meiki_ocr(_W["meiki"], bgr)
    g = V.grey_of(rgb)
    crops, texts = [], []
    for (x0, y0, x1, y1), text, _vert in res:
        if x1 <= x0 or y1 <= y0:
            continue
        crops.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), len(texts)))
        texts.append(text.replace("\ufffd", ""))
    return score_frame_units(fr, g, crops, texts)


def _init_detect(spec, threads):
    _W["det"] = detector(spec, threads)


def _detect_job(fr):
    t0 = time.perf_counter()
    rgb = load_rgb(fr)
    b = _W["det"].boxes(rgb)
    return {"key": fr["key"], "W": fr["W"], "H": fr["H"], "boxes": b,
            "ms": round((time.perf_counter() - t0) * 1e3, 2)}


def _pool(workers, init, args):
    import multiprocessing as mp
    return mp.get_context("fork").Pool(workers, initializer=init, initargs=args)


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def result_doc(kind, frames_path, fmeta, rows, extra):
    pub = [{k: v for k, v in r.items() if k not in ("texts", "status", "kinds", "boxes_xywh")} for r in rows]
    doc = {"schema": SCHEMA, "kind": kind, "frames_file": str(frames_path), "frames_sha256": sha256_file(frames_path),
           "frames_meta": fmeta, **extra, "repo_head": git_head(), "desktop_py_sha256": sha256_file(__file__),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": summarize(pub),
           "units": [{k: r[k] for k in ("uid", "set", "cluster", "group", "view", "errors", "chars", "lines", "boxes",
                                        "whole", "cut", "missed", "noink", "merged", "junk")} for r in pub]}
    return doc


def print_summary(doc):
    for k, v in doc["summary"]["by_group"].items():
        print(f"{k}: units {v['units']} CER {v['cer']:.3f}% ({v['errors']}/{v['chars']}) recall "
              f"{v['line_recall']:.3f} precision {v['box_precision']:.3f} cut {v['cut']} missed {v['missed']} "
              f"merged {v['merged']} junk {v['junk']}", flush=True)
    a = doc["summary"]["all"]
    print(f"all: units {a['units']} CER {a['cer']:.3f}% recall {a['line_recall']:.3f} precision "
          f"{a['box_precision']:.3f}", flush=True)


# ----------------------------------------------------------------------------- commands

def cmd_frames(a):
    frames, meta = build_frames(a.set, a.protocol)
    meta = dict(meta, set=a.set, frames=len(frames), units=sum(len(f["units"]) for f in frames),
                clusters=len({f["cluster"] for f in frames}),
                views=dict(Counter(f["view"] for f in frames)),
                groups=dict(Counter(u["group"] for f in frames for u in f["units"])))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"frames": frames, "meta": meta}, f)
    print(json.dumps(meta), flush=True)


def resample_frame(fr, height):
    """A frame record resampled to `height` px (aspect kept): its units' geometry scaled by the exact factors."""
    W, H = fr["W"], fr["H"]
    nw, nh = max(1, int(round(W * height / H))), int(height)
    sx, sy = nw / W, nh / H

    def box(b):
        return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

    units = []
    for u in fr["units"]:
        x0, y0, x1, y1 = u["region"]
        region = [int(math.floor(x0 * sx)), int(math.floor(y0 * sy)), min(nw, int(math.ceil(x1 * sx))),
                  min(nh, int(math.ceil(y1 * sy)))]
        units.append(dict(u, uid=f"{u['uid']}@{nh}", region=region,
                          lines=[dict(ln, box=box(ln["box"]), cells=[[c[0] * sx, c[1] * sy, c[2] * sx, c[3] * sy]
                                                                     for c in ln["cells"]]) for ln in u["lines"]]))
    return dict(fr, key=f"{fr['key']}@{nh}", W=nw, H=nh, units=units,
                resample={"from": [W, H], "to": [nw, nh], "method": "cv2.INTER_LINEAR up, cv2.INTER_AREA down"})


def cmd_resample(a):
    frames, meta = load_frames(a.frames)
    out = [resample_frame(fr, a.height) for fr in frames]
    meta = dict(meta, resampled_to_height=a.height, source_frames_sha256=sha256_file(a.frames),
                views={**{f"{v}@{a.height}": n for v, n in Counter(f["view"] for f in out).items()}})
    for fr in out:
        fr["view"] = f"{fr['view']}@{a.height}"
    with open(a.out, "wb") as f:
        pickle.dump({"frames": out, "meta": meta}, f)
    print(json.dumps({k: meta[k] for k in ("resampled_to_height", "views")}), flush=True)


def cmd_detect(a):
    frames, _ = load_frames(a.frames)
    if a.limit:
        frames = frames[:a.limit]
    t0 = time.time()
    rows = []
    with _pool(a.workers, _init_detect, (a.det, a.threads)) as pool:
        for r in pool.imap(_detect_job, frames, chunksize=4):
            rows.append(r)
    meta = {"meta": {"det": a.det, "frames_file": str(a.frames), "frames_sha256": sha256_file(a.frames),
                     "threads_per_worker": a.threads, "workers": a.workers, "repo_head": git_head(),
                     "seconds": round(time.time() - t0, 1)}}
    write_jsonl(a.out, [meta, *rows])
    n = sum(len(r["boxes"]) for r in rows)
    print(f"desktop detect: {a.det} {len(rows)} frames, {n} boxes, {time.time() - t0:.0f} s -> {a.out}", flush=True)


def load_boxes(path):
    rows = read_jsonl(path)
    return {r["key"]: r["boxes"] for r in rows[1:]}, rows[0]["meta"]


def cmd_read(a):
    frames, fmeta = load_frames(a.frames)
    boxes, bmeta = load_boxes(a.boxes)
    if bmeta["frames_sha256"] != sha256_file(a.frames):
        raise SystemExit("the boxes were found on another frames file")
    pad = a.pad if a.pad not in (None, "none") else None
    jobs = [(fr, boxes[fr["key"]], pad) for fr in frames if fr["key"] in boxes]
    if len(jobs) != len(frames):
        raise SystemExit(f"{len(frames) - len(jobs)} frames have no boxes in {a.boxes}")
    t0 = time.time()
    rows = []
    with _pool(a.workers, _init_read, (a.rec, 1)) as pool:
        for rs in pool.imap(_read_job, jobs, chunksize=4):
            rows.extend(rs)
    doc = result_doc("vocr-desktop-read", a.frames, fmeta, rows,
                     {"det": bmeta["det"], "boxes_file": str(a.boxes), "boxes_sha256": sha256_file(a.boxes),
                      "pad": pad, "rec": Path(a.rec).name, "rec_sha256": sha256_file(a.rec), "decode": "greedy",
                      "lm": None, "filter": None, "seconds": round(time.time() - t0, 1)})
    if a.rows:
        write_jsonl(a.rows, rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print_summary(doc)


def cmd_meiki(a):
    import engines as E
    frames, fmeta = load_frames(a.frames)
    t0 = time.time()
    rows = []
    with _pool(a.workers, _init_meiki, (1,)) as pool:
        for rs in pool.imap(_meiki_job, frames, chunksize=4):
            rows.extend(rs)
    from importlib.metadata import version
    doc = result_doc("vocr-desktop-meiki", a.frames, fmeta, rows,
                     {"det": "meikiocr", "meikiocr": version("meikiocr"),
                      "files": {k: {"name": E.model_path(k).name, "sha256": sha256_file(E.model_path(k))}
                                for k in ("meiki_det", "meiki_rec", "meiki_vrec")},
                      "settings": {"det_threshold": 0.5, "rec_threshold": 0.1, "punct_conf_factor": 1.0,
                                   "input": "BGR"}, "seconds": round(time.time() - t0, 1)})
    if a.rows:
        write_jsonl(a.rows, rows)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print_summary(doc)


def _overlap(a, b):
    """(dx, dy) of two [x0, y0, x1, y1] boxes."""
    return min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])


def page_refs(rec):
    """(references, ignore areas) [x0, y0, x1, y1] of a page pool record (cmd_boxes_pr's rule)."""
    def xyxy(b, g=0):
        return [b[0] - g, b[1] - g, b[0] + b[2] + g, b[1] + b[3] + g]
    refs, ign = [], []
    held = set()
    for ln in rec["lines"]:
        held.update(ln.get("box_index") or [])
        box = ln.get("box_main") if ln.get("furigana") and ln.get("box_main") else ln["box"]
        if ln["kind"] in ("line", "name") and not ln["vertical"] and ln.get("drop") in (None, "oov"):
            refs.append(xyxy(box))
        else:
            ign.append(xyxy(box, 2))
    for i in rec.get("uncovered") or []:
        b = rec["boxes"][i - 1]["xyxy"] if rec["boxes"][i - 1]["index"] == i else \
            next(x["xyxy"] for x in rec["boxes"] if x["index"] == i)
        ign.append([b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2])
    return refs, ign


def box_pr(dets, refs, ign):
    """(true, false, missed) of horizontal detections against references (cmd_boxes_pr's rule)."""
    pairs = []
    for i, d in enumerate(dets):
        for j, r in enumerate(refs):
            dx, dy = _overlap(d, r)
            if dx >= 0.5 * min(d[2] - d[0], r[2] - r[0]) and dy >= 0.5 * min(d[3] - d[1], r[3] - r[1]):
                pairs.append((dx * dy, i, j))
    used_d, used_r = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_d and j not in used_r:
            used_d.add(i)
            used_r.add(j)
    false = 0
    for i, d in enumerate(dets):
        if i in used_d:
            continue
        area = max(1e-9, (d[2] - d[0]) * (d[3] - d[1]))
        inside = max((max(0, _overlap(d, g)[0]) * max(0, _overlap(d, g)[1]) for g in ign), default=0)
        if inside < 0.5 * area:
            false += 1
    return len(used_d), false, len(refs) - len(used_r)


def cmd_boxes_pr(a):
    frames, fmeta = load_frames(a.frames)
    boxes, bmeta = load_boxes(a.boxes)
    recs = {}
    for f in sorted({fr["cluster"] for fr in frames}):
        for r in read_jsonl(POOL / f"{f}.jsonl"):
            recs[f"{f}:{r['page']}"] = r
    tot = defaultdict(lambda: [0, 0, 0])
    for fr in frames:
        if fr["src"]["kind"] != "hooked":
            raise SystemExit("boxes-pr reads hooked pages only")
        refs, ign = page_refs(recs[fr["key"]])
        p = pad_of(a.pad, fr["H"]) if a.pad not in (None, "none") else 0
        dets = []
        for b in boxes[fr["key"]]:
            if b[4]:
                continue
            cb = crop_box(b, p, fr["W"], fr["H"])
            if cb:
                dets.append([cb[0], cb[1], cb[0] + cb[2], cb[1] + cb[3]])
        t, f_, m = box_pr(dets, refs, ign)
        for k in (fr["view"], "all"):
            tot[k][0] += t
            tot[k][1] += f_
            tot[k][2] += m
    res = {}
    for k, (t, f_, m) in sorted(tot.items()):
        p_, r_ = t / max(1, t + f_), t / max(1, t + m)
        res[k] = {"true": t, "false": f_, "missed": m, "precision": round(p_, 4), "recall": round(r_, 4),
                  "f1": round(2 * p_ * r_ / max(1e-9, p_ + r_), 4)}
        print(f"{k}: P {p_:.3f} R {r_:.3f} F1 {res[k]['f1']:.3f} (true {t}, false {f_}, missed {m})", flush=True)
    doc = {"schema": SCHEMA, "kind": "vocr-desktop-boxes-pr", "frames_sha256": sha256_file(a.frames),
           "boxes_sha256": sha256_file(a.boxes), "det": bmeta["det"], "pad": a.pad, "frames_meta": fmeta,
           "result": res}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def compare_docs(A, B):
    """{group: pagepath.cluster_bootstrap(arm A, base B)} over the units' groups (set:all, set:view, set:group)."""
    import pagepath as PP
    ua, ub = {u["uid"]: u for u in A["units"]}, {u["uid"]: u for u in B["units"]}
    if set(ua) != set(ub):
        raise SystemExit("compare: the runs hold different units")
    groups = defaultdict(lambda: ([], []))
    for k, u in ub.items():
        for g in (f"{u['set']}:all", f"{u['set']}:{u['view']}", f"{u['set']}:{u['group']}"):
            groups[g][0].append(ua[k])
            groups[g][1].append(u)
    return {g: PP.cluster_bootstrap(ra, rb) for g, (ra, rb) in sorted(groups.items())}


def cmd_compare(a):
    A = json.loads(Path(a.arm).read_text(encoding="utf-8"))
    B = json.loads(Path(a.base).read_text(encoding="utf-8"))
    res = compare_docs(A, B)
    for g, c in res.items():
        print(f"{g}: {c['base_cer']:.3f}% -> {c['arm_cer']:.3f}%  delta {c['delta']:+.3f} CI {c['ci95']}  "
              f"clusters {c['clusters']}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"base": a.base, "arm": a.arm, "base_sha256": sha256_file(a.base),
                                           "arm_sha256": sha256_file(a.arm), "compare": res}, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frames")
    p.add_argument("--set", required=True, choices=SETS)
    p.add_argument("--out", required=True)
    p.add_argument("--protocol")
    p = sub.add_parser("resample")
    p.add_argument("--frames", required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--frames", required=True)
    p.add_argument("--det", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("read")
    p.add_argument("--frames", required=True)
    p.add_argument("--boxes", required=True)
    p.add_argument("--rec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pad", default="none", help="auto | N | none (the detector's boxes as they are)")
    p.add_argument("--rows")
    p.add_argument("--workers", type=int, default=5)
    p = sub.add_parser("meiki")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows")
    p.add_argument("--workers", type=int, default=5)
    p = sub.add_parser("boxes-pr")
    p.add_argument("--frames", required=True)
    p.add_argument("--boxes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pad", default="none")
    p = sub.add_parser("compare")
    p.add_argument("--base", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    {"frames": cmd_frames, "resample": cmd_resample, "detect": cmd_detect, "read": cmd_read, "meiki": cmd_meiki,
     "boxes-pr": cmd_boxes_pr, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The textspot2 labels (arm b) and the enlarged real pool (arm c) of the full-frame text spotter (issue #106;
docs/textspot2.md "Data" b and c with amendment 1; eval/textspot2_protocol.json "data" and amendments[0]).

  textspot2_labels.py cache --arm lab|more --source S [--workers W] [--limit N]   colour, meiki, refine, winner
  textspot2_labels.py build --arm lab|more --source S                             the shards, from the cache
  textspot2_labels.py pool                                                        arm c: the new screenshots
  textspot2_labels.py leak                                                        arm c: thumbnail check
  textspot2_labels.py manifest [--out data/manifests/textspot2_shards_v1.json]    every shard, hashes, statistics
  textspot2_labels.py sheet --arm lab|more --sheet K --sources S,... --out PNG    QA contact sheet (<= 1280 px)
  textspot2_labels.py probe [--n 200] [--workers W]                               timing probe (nothing is kept)

Every heavy step runs through HEAVY_SLOTS=12 $HERD/heavy.sh timeout <secs> ... < /dev/null with W single-thread
workers (default: the slot's OMP_NUM_THREADS, 5), pinned to the slot's cores as teacher/label.py pins its readers.
The C driver comes from `eval/textspot.py cbuild --out $T2/labels/cbuild`.

Nothing of #97's or the teacher's code is changed; this file imports it:

  canvases   arm b: #97's real canvases (data/manifests/textdet_shards_v1.json: realplus, teacher, devnative, owner)
             minus textspot's skip lists (data/manifests/textspot_shards_v1.json real.measured.skip). Their source
             records are synth/textdet_data.real_rows(source), which gives #97's rows exactly (the same shots in the
             same order as #97's hygiene rows). Arm c: the new screenshots of `pool`, one canvas each, made as
             real_canvases makes one for its source (PC screenshots at the Vita target only).
  colour     the canvas's colour source at the canvas's size, made as synth/textdet_data.real_canvases made the
             canvas: the record's file (cv2.imread, BGR -> RGB), the read size (realplus, owner, devnative: the
             record's dims; the teacher: the screenshot's own size for Vita and PSP shots, round(w s) x round(h s)
             with s = teacher/geometry.target_scale for the canvas's target for PC shots) and the same cv2.resize
             (INTER_AREA when it shrinks, else INTER_LINEAR, none at the same size). Arm b uses it only when
             synth/common.to_gray of it equals the canvas byte for byte; a canvas whose record or file is missing
             ("missing") or whose rebuild differs ("mismatch") is read as its grey in three channels (counted).
  meiki      teacher/engines.Meiki(det_thr=0.2), one thread: its detector exactly as Meiki.ocr runs it (MeikiOCR's
             _preprocess_for_detection and _run_detection_inference, scores > det_thr, boxes clipped to the image
             and cast to int, in Meiki.ocr's order), BGR input as teacher/label.py feeds it, no recognition. Every
             box is refined to ink by teacher/geometry.refine twice, both cached: on the colour source ("ref_h")
             and on its transpose ("ref_t": the box transposed, the result transposed back).
  winner     the textspot winner (v0.3.3-textspot.1 textspot_s4_c24n4_w8.vtsp, sha256 checked), det decode on the
             grey canvas, through the C runtime's batch driver (runtime/vocr_textspot_test --batch); one canvas in
             RULES["winner_twin_check_every"] (by key hash) also through synth/textspot_twin.spot, which must agree.
  cache      one JSON line per canvas (a jsonl per shard; the owner's under $HERD/shared/private/textspot2),
             appended as results come in, so a killed job resumes where it stopped.
  build      docs/textspot2.md "Data" b with amendment 1 (build_labels; RULES holds every setting) on the cached
             boxes, in #97's shard format with lines [x0, y0, x1, y1, vertical, "text", weight] and ignore
             [[x0, y0, x1, y1]]; each sample's src is its source's name, so the mix per source stays textspot's:
             arm b  $T2/shards/lab/lab_<source>/<#97's shard name>.json ($P2/shards/lab/lab_owner/ for the owner):
                    #97's samples in #97's order minus the skipped ones, offsets into #97's canvases; <name>.npy is
                    a symlink to #97's npy;
             arm c  $T2/shards/more/more_<source>/more_<source>_NNNN.{npy,json} ($P2/shards/more/more_owner/):
                    the canvases written.
  manifest   data/manifests/textspot2_shards_v1.json (kind textspot2-shards): every shard (dir, name, npy_sha256,
             json_sha256, samples, lines, vertical, by_src), the code's sha256, meiki's and the winner's, the rules,
             textspot's skip lists (real.measured.skip, as train/textspot_train.skip_lists reads them), the pool's
             reserve check and thumbnail check, the statistics per arm and source, the totals.
             train/textspot_train.FramePool loads it.
"""
import argparse
import collections
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))
import common  # noqa: E402
import textdet_data as TD  # noqa: E402  (it puts teacher/ and eval/linefind/ on sys.path)
import textspot_real_half as RH  # noqa: E402
from teacher import geometry as G  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIVATE = HERD / "shared" / "private"
T2 = ART / "textspot2"
P2 = PRIVATE / "textspot2"
SOURCES = ("realplus", "teacher", "devnative", "owner")
ARMS = ("lab", "more")
TEXTDET_MANIFEST = ROOT / "data" / "manifests" / "textdet_shards_v1.json"
TEXTSPOT_MANIFEST = ROOT / "data" / "manifests" / "textspot_shards_v1.json"
PROTOCOL = ROOT / "eval" / "textspot2_protocol.json"
OUT_MANIFEST = ROOT / "data" / "manifests" / "textspot2_shards_v1.json"
WINNER = ART / "textspot" / "release" / "v0.3.3-textspot.1" / "textspot_s4_c24n4_w8.vtsp"
WINNER_SHA256 = "852506919ba94e54a4efc55e3695f365626d9d4ce6347f420cbf2afe3e8360cf"
CBUILD = T2 / "labels" / "cbuild"
URL_ID = re.compile(r"/(\d+)\.(?:jpg|png)$")
SHOT_ID = re.compile(r"^(v\d+)_(\d+)$")

# The label rules (docs/textspot2.md "Data" b with amendment 1), every setting in one place.
RULES = {
    "det_thr": 0.2,             # meiki boxes kept: score > det_thr (teacher/engines.Meiki's own comparison)
    "pos_min": 0.5,             # a meiki box of score >= 0.5 may become a positive; one below is an ignore area
    "weight_span": 0.3,         # weight 0.5 + 0.5 min(1, (score - 0.5) / 0.3)
    "cover": 0.5,               # covered: at least half of the box lies inside the union of the named boxes
    "vertical_ratio": 1.5,      # vertical: refined height >= 1.5 x width
    "across": [6, 96],          # a new positive's size across (height, or width when vertical) at canvas scale
    "replace_iou": 0.5,         # an existing ignore area that a new positive's raw meiki box meets at IoU >= 0.5
    "refine": "P1",             # refined_of
    "cover_box": "refined",     # the box of a meiki candidate that the existing positives must cover
    "low_covered": "ignore",    # a low-score box the existing positives cover: an ignore area ("ignore") or not
    "box": "every meiki-made label (positive or ignore area) is the refined box; the raw box only where the "
           "protocol says raw (the union a winner box must lie in, the IoU that removes an existing ignore area)",
    "furigana_hosts": "every other horizontal meiki box (refined, score > det_thr) and every horizontal existing "
                      "positive; teacher/geometry.is_furigana's geometry without its kana test; horizontal "
                      "candidates only",
    "winner_twin_check_every": 25,
}
CAPS_MORE = {"pc": 8, "native": 12, "owner": 200}
POOL_SALT = "textspot2-pool:"
PER_SHARD_MORE = 500
WEIGHT_BINS = ("0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", "1.0 by score", "1.0 by winner")
IGNORE_KINDS = ("low", "winner_only", "furigana_record", "furigana_geometry", "size_small", "size_large")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(obj, path, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(path)


def jsonl(path):
    p = Path(path)
    if not p.exists():
        return
    for x in p.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        try:
            yield json.loads(x)
        except json.JSONDecodeError:            # a line cut by a crash: that canvas is redone
            continue


def herd_rel(path):
    return str(path).replace(str(HERD), "$HERD")


# ----------------------------------------------------------------------------- where things live

def base_of(source):
    return P2 if source == "owner" else T2


def shard_dir(arm, source):
    """One arm's shards of one source; train/textspot_train.FramePool matches the directory's basename."""
    return base_of(source) / "shards" / arm / f"{arm}_{source}"


def cache_dir(arm, source):
    return base_of(source) / "labels" / arm / source


def cache_path(arm, source, name):
    return cache_dir(arm, source) / f"{name}.jsonl"


# ----------------------------------------------------------------------------- box geometry

def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / float(area(a) + area(b) - inter)


def union_cover(b, boxes):
    """The share of box b's area inside the union of boxes (exact, by coordinate compression)."""
    a = area(b)
    if a <= 0:
        return 0.0
    clips = []
    for c in boxes:
        x0, y0, x1, y1 = max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3])
        if x1 > x0 and y1 > y0:
            clips.append((x0, y0, x1, y1))
    if not clips:
        return 0.0
    xs = sorted({v for c in clips for v in (c[0], c[2])})
    ys = sorted({v for c in clips for v in (c[1], c[3])})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grid = np.zeros((len(ys) - 1, len(xs) - 1), bool)
    for x0, y0, x1, y1 in clips:
        grid[yi[y0]:yi[y1], xi[x0]:xi[x1]] = True
    cell = np.diff(np.asarray(ys, np.int64))[:, None] * np.diff(np.asarray(xs, np.int64))[None, :]
    return float((grid * cell).sum()) / a


def covered(b, boxes, frac=None):
    """At least half of b lies inside the union of boxes (amendment 1, "Coverage")."""
    return union_cover(b, boxes) >= (RULES["cover"] if frac is None else frac)


def tall(b, ratio=None):
    """The protocol's vertical rule: height >= 1.5 x width."""
    return (b[3] - b[1]) >= (RULES["vertical_ratio"] if ratio is None else ratio) * (b[2] - b[0])


def across(b, vertical):
    return (b[2] - b[0]) if vertical else (b[3] - b[1])


def is_furigana_geom(b, host, det):
    """teacher/geometry.is_furigana's geometry without its kana test (meiki's detector reads no text): b at most
    0.65 x the host's height (b's height from its raw box `det` when that is smaller), inside the host's horizontal
    extent +- 4 px, its centre above the host's top third, its bottom at most half the host's height above the
    host's top. The function itself is called, with a kana placeholder for the text it would test."""
    return G.is_furigana(list(b), list(host), "\u3042", list(det))


def refine_h(rgb, raw):
    """teacher/geometry.refine of a box on the colour source as it is (a horizontal line's refinement)."""
    return [int(v) for v in G.refine(rgb, [int(v) for v in raw])[0]]


def refine_t(rgb, raw):
    """teacher/geometry.refine on the transposed crop (amendment 1, a vertical box): the image and the box
    transposed, the result transposed back. A box refine returns unchanged keeps its raw extent."""
    x0, y0, x1, y1 = (int(v) for v in raw)
    r = G.refine(np.transpose(rgb, (1, 0, 2)), [y0, x0, y1, x1])[0]
    return [int(r[1]), int(r[0]), int(r[3]), int(r[2])]


def refined_of(m, procedure=None):
    """(refined box, vertical) of a cached meiki box m {"raw", "ref_h", "ref_t"} (amendment 1, "Refinement").

    P1, the amendment's words in order: the box is refined as geometry.refine refines (horizontally); it is
       vertical when that refined box is at least 1.5 x as tall as wide (the raw box's proportions when refine
       returned it unchanged: the same box); a vertical box is refined on the transposed crop instead, and stays
       vertical.
    P2: the raw box's proportions choose the refinement (on the transposed crop when the raw box is at least 1.5 x
       as tall as wide) and the refined box's proportions give the flag."""
    procedure = procedure or RULES["refine"]
    raw, rh, rt = m["raw"], m["ref_h"], m["ref_t"]
    if procedure == "P1":
        vertical = tall(rh)
        return (list(rt) if vertical else list(rh)), vertical
    if procedure == "P2":
        if tall(raw):
            return list(rt), tall(rt)
        return list(rh), tall(rh)
    raise ValueError(f"unknown refine procedure {procedure!r}")


def weight_of(score, agrees):
    """A new meiki positive's weight: 1 when the winner agrees (>= 50% of it inside the winner boxes' union), else
    0.5 + 0.5 min(1, (score - 0.5) / 0.3)."""
    if agrees:
        return 1.0
    return 0.5 + 0.5 * min(1.0, (float(score) - RULES["pos_min"]) / RULES["weight_span"])


def weight_bin(w, agrees):
    if agrees:
        return "1.0 by winner"
    if w >= 1.0:
        return "1.0 by score"
    return WEIGHT_BINS[min(4, max(0, int((w - 0.5) * 10 + 1e-9)))]


def build_labels(width, height, existing, existing_ignore, furi, meiki, winner, rules=None):
    """The textspot2 labels of one canvas (docs/textspot2.md "Data" b with amendment 1). Pure: boxes in, boxes out.

    existing         the canvas's existing positives [[x0, y0, x1, y1, vertical, ...]] (arm b: #97's lines; arm c: [])
    existing_ignore  its existing ignore areas [[x0, y0, x1, y1]] (arm b: #97's; arm c: [])
    furi             the source record's furigana boxes at the canvas's scale
    meiki            the cached meiki boxes [{"raw", "score", "ref_h", "ref_t"}] (score > det_thr)
    winner           the winner's boxes [[x0, y0, x1, y1]] (det decode)

    1. every meiki box: its refined box and orientation (refined_of), clipped to the canvas;
    2. score < 0.5: an ignore area (low); score >= 0.5: nothing when the existing positives cover it; else an
       ignore area when it is furigana (the record's furigana boxes cover it, or the geometry against a bigger
       horizontal meiki box or existing positive) or outside 6-96 px across; else a new positive, weight 1 when the
       winner boxes cover it, else 0.5 + 0.5 min(1, (score - 0.5) / 0.3);
    3. every winner box that the raw meiki boxes and the positives (existing and new) do not cover: an ignore area;
    4. the existing ignore areas, less those a new positive's raw meiki box meets at IoU >= 0.5.
    Returns (lines [[x0, y0, x1, y1, vertical, "text", weight]], ignore [[x0, y0, x1, y1]], stats)."""
    R = dict(RULES, **(rules or {}))
    ex_boxes = [[int(v) for v in ln[:4]] for ln in existing]
    ex_hosts = [b for b, ln in zip(ex_boxes, existing) if not int(ln[4])]
    ms = []
    for m in meiki:
        box, vertical = refined_of(m, R["refine"])
        box = [max(0, int(box[0])), max(0, int(box[1])), min(width, int(box[2])), min(height, int(box[3]))]
        ms.append({"raw": [int(v) for v in m["raw"]], "score": float(m["score"]), "box": box, "vertical": vertical})
    meiki_hosts = [(i, x["box"]) for i, x in enumerate(ms) if not x["vertical"] and area(x["box"]) > 0]
    st = collections.Counter()
    ign = {k: [] for k in IGNORE_KINDS}
    new = []
    for i, x in enumerate(ms):
        st["meiki"] += 1
        box = x["box"]
        if area(box) <= 0:
            st["meiki_empty"] += 1
            continue
        cover_box = box if R["cover_box"] == "refined" else x["raw"]
        if x["score"] < R["pos_min"]:
            st["meiki_low"] += 1
            if R["low_covered"] == "drop" and covered(cover_box, ex_boxes):
                st["low_dropped_covered"] += 1
                continue
            ign["low"].append(box)
            continue
        st["meiki_high"] += 1
        if covered(cover_box, ex_boxes):
            st["high_covered"] += 1
            continue
        if furi and covered(box, furi):
            ign["furigana_record"].append(box)
            continue
        if not x["vertical"] and (any(is_furigana_geom(box, o, x["raw"]) for j, o in meiki_hosts if j != i) or
                                  any(is_furigana_geom(box, o, x["raw"]) for o in ex_hosts)):
            ign["furigana_geometry"].append(box)
            continue
        a = across(box, x["vertical"])
        if a < R["across"][0]:
            ign["size_small"].append(box)
            continue
        if a > R["across"][1]:
            ign["size_large"].append(box)
            continue
        agrees = covered(box, winner)
        new.append({"box": box, "vertical": x["vertical"], "w": weight_of(x["score"], agrees), "raw": x["raw"],
                    "agrees": agrees, "score": x["score"]})
    positives = ex_boxes + [p["box"] for p in new]
    raw_all = [x["raw"] for x in ms]
    for wb in winner:
        wb = [int(v) for v in wb[:4]]
        if area(wb) > 0 and not covered(wb, raw_all + positives):
            ign["winner_only"].append(wb)
    kept_ignore, removed = [], 0
    for b in existing_ignore:
        if any(iou(p["raw"], b) >= R["replace_iou"] for p in new):
            removed += 1
        else:
            kept_ignore.append([int(v) for v in b[:4]])
    lines = [[*[int(v) for v in ln[:5]], "text", 1.0] for ln in existing]
    wh = collections.Counter()
    for p in sorted(new, key=lambda p: (p["box"][1], p["box"][0], p["box"][3], p["box"][2])):
        lines.append([*p["box"], int(p["vertical"]), "text", round(p["w"], 4)])
        wh[weight_bin(p["w"], p["agrees"])] += 1
    ignore = kept_ignore + [b for k in IGNORE_KINDS for b in ign[k]]
    st.update({"positives_kept": len(existing), "positives_added": len(new),
               "vertical_kept": sum(int(ln[4]) for ln in existing),
               "vertical_added": sum(int(p["vertical"]) for p in new),
               "ignore_kept": len(kept_ignore), "ignore_removed": removed,
               **{f"ignore_{k}": len(v) for k, v in ign.items()}})
    detail = {"new": [{k: p[k] for k in ("box", "raw", "w", "agrees", "vertical", "score")} for p in new],
              "low": [(x["box"], x["score"]) for x in ms if x["score"] < R["pos_min"] and x["box"] in ign["low"]],
              "winner_only": [wb for wb in ign["winner_only"]]}
    return lines, ignore, {"counts": dict(st), "weights": dict(wh), "n_existing": len(existing),
                           "ignore_kinds": {k: len(v) for k, v in ign.items()}, "detail": detail}


# ----------------------------------------------------------------------------- records and colour sources

def record_rows(source, uncapped=False):
    """synth/textdet_data.real_rows(source): #97's records (its exclusions and per-VN cap), or, uncapped, every record
    its exclusions keep (textdet_data.CAPS[source] is set aside for the call and restored)."""
    if not uncapped:
        return TD.real_rows(source)
    saved = TD.CAPS[source]
    TD.CAPS[source] = 1 << 30
    try:
        return TD.real_rows(source)
    finally:
        TD.CAPS[source] = saved


def plan_of(r, source, target, sw, sh):
    """(target, (dw, dh), k) of one canvas as synth/textdet_data.real_canvases plans it: the read size and the scale
    k of the record's boxes. target: the canvas's target (the teacher's PC screenshots, one per target)."""
    if source in ("realplus", "owner", "devnative"):
        dw, dh = (int(v) for v in r["dims"])
        return ("vita" if r["kind"] != "psp" else "psp"), (dw, dh), 1.0
    if r["kind"] in ("vita", "psp"):
        return r["kind"], (sw, sh), 1.0
    s = G.target_scale(sw, sh, target)
    return target, (max(1, int(round(sw * s))), max(1, int(round(sh * s)))), s


def read_rgb(path):
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return np.ascontiguousarray(bgr[:, :, ::-1])


def resize_as_97(rgb, dw, dh):
    """real_canvases' resize: none at the same size, cv2.INTER_AREA when it shrinks (dw < sw), else INTER_LINEAR."""
    import cv2
    sh, sw = rgb.shape[:2]
    if (dw, dh) == (sw, sh):
        return rgb
    return cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA if dw < sw else cv2.INTER_LINEAR)


def colour_source(r, source, target):
    """(rgb at the canvas's size, k, plan) of a record; (None, None, None) when the record or its file is missing."""
    if r is None or not r.get("path"):
        return None, None, None
    rgb = read_rgb(r["path"])
    if rgb is None:
        return None, None, None
    sh, sw = rgb.shape[:2]
    plan = plan_of(r, source, target, sw, sh)
    _, (dw, dh), k = plan
    return np.ascontiguousarray(resize_as_97(rgb, dw, dh)), k, plan


def record_furi(r, k, width, height):
    """The source record's furigana boxes ('furi' of its lines, teacher/geometry.attach_furigana) at the canvas's
    scale, scaled as real_canvases scales the record's boxes (textdet_data._scaled)."""
    out = []
    for ln in (r or {}).get("lines") or []:
        for f in ln.get("furi") or []:
            b = TD._scaled(f, k, width, height)
            if b[2] > b[0] and b[3] > b[1]:
                out.append(b)
    return out


def lab_k(r, source, smp):
    """k of a lab canvas without its file: the plan from #97's recorded source size (meta src_size)."""
    if r is None:
        return None
    sw, sh = (smp.get("meta") or {}).get("src_size") or (smp["w"], smp["h"])
    return plan_of(r, source, (smp.get("meta") or {}).get("target") or smp["kind"], int(sw), int(sh))[2]


def grey3(grey):
    return np.ascontiguousarray(np.repeat(np.asarray(grey, np.uint8)[:, :, None], 3, axis=2))


# ----------------------------------------------------------------------------- meiki and the winner

_W = {}


def init_worker(cbuild, blob_path, det_thr):
    """One single-thread worker: MeikiOCR's sessions (teacher/engines.Meiki, one thread) and the winner's driver."""
    import cv2
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    import textspot_twin as TS
    from teacher import engines as E
    _W["meiki"] = E.Meiki(det_thr=det_thr, n_threads=1)
    _W["exe"] = str(Path(cbuild) / "vocr_textspot_test")
    _W["blob"] = str(blob_path)
    _W["twin"] = TS.Model(Path(blob_path).read_bytes())
    _W["npy"] = {}


def meiki_detect(meiki, bgr):
    """teacher/engines.Meiki.ocr's detection without its recognition: [(raw box [x0, y0, x1, y1], score)] with
    score > meiki.det_thr, boxes clipped to the image and cast to int32, in Meiki.ocr's order (stable by top edge).
    bgr: as teacher/label.py feeds Meiki (cv2.imread's channel order; Meiki.rgb is False)."""
    img = np.ascontiguousarray(bgr[:, :, ::-1]) if meiki.rgb else bgr
    m = meiki.m
    tensor, scale = m._preprocess_for_detection(img)
    _, boxes, scores = m._run_detection_inference(tensor, scale)
    boxes, scores = boxes[0], scores[0]
    keep = scores > meiki.det_thr
    h, w = img.shape[:2]
    boxes = np.clip(boxes[keep], 0, np.array([w, h, w, h])).astype(np.int32)
    scores = scores[keep]
    order = np.argsort(boxes[:, 1], kind="stable")
    return [([int(v) for v in boxes[i]], float(scores[i])) for i in order]


def winner_c(exe, blob, greys):
    """runtime/vocr_textspot_test --batch on grey frames, with eval/textspot.CDrivers.spot's records and parsing:
    [[(x, y, w, h, vertical, conf, dark, block), ...] per frame]."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for g in greys:
            f.write(struct.pack("<4I", g.shape[1], g.shape[0], 1, 0) + np.ascontiguousarray(g).tobytes())
        path = f.name
    try:
        p = subprocess.run([exe, "--batch", blob, path], capture_output=True, text=True, timeout=3600,
                           stdin=subprocess.DEVNULL)
    finally:
        os.unlink(path)
    if p.returncode:
        raise RuntimeError(f"vocr_textspot_test: rc {p.returncode}: {p.stderr[-2000:]}")
    out = []
    for line in p.stdout.splitlines():
        v = [int(t) for t in line.split()]
        if v[0]:
            raise RuntimeError(f"vocr_textspot_find rc={v[0]}")
        n = v[1]
        out.append([tuple(v[5 + 8 * i:13 + 8 * i]) for i in range(n)])
    if len(out) != len(greys):
        raise RuntimeError("vocr_textspot_test: one result per frame expected")
    return out


def winner_twin(model, grey):
    import textspot_twin as TS
    return [(b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block) for b in TS.spot(model, grey).boxes]


def detect(rgb):
    """meiki on the colour source and both refinements of every box: (the cache's "meiki" list, det s, refine s)."""
    t0 = time.perf_counter()
    found = meiki_detect(_W["meiki"], np.ascontiguousarray(rgb[:, :, ::-1]))
    t1 = time.perf_counter()
    out = []
    for raw, score in found:
        if raw[2] <= raw[0] or raw[3] <= raw[1]:
            out.append({"raw": raw, "score": score, "ref_h": raw, "ref_t": raw})
        else:
            out.append({"raw": raw, "score": score, "ref_h": refine_h(rgb, raw), "ref_t": refine_t(rgb, raw)})
    return out, t1 - t0, time.perf_counter() - t1


def _npy(path):
    d = _W["npy"]
    if path not in d:
        d.clear()
        d[path] = np.load(path, mmap_mode="r")
    return d[path]


def twin_checked(key):
    every = RULES["winner_twin_check_every"]
    return bool(every) and int(hashlib.sha256(key.encode()).hexdigest(), 16) % every == 0


def work_chunk(job):
    """A chunk of canvases -> cache lines. job: {"arm", "source", "items"}; a lab item has #97's npy, its sample and
    its record, a more item its record and target."""
    arm, source = job["arm"], job["source"]
    out, greys = [], []
    ms = collections.Counter()
    for it in job["items"]:
        t0 = time.perf_counter()
        r = it.get("record")
        if arm == "lab":
            smp = it["sample"]
            data = _npy(it["npy"])
            grey = np.ascontiguousarray(np.asarray(data[smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(
                smp["h"], smp["w"]))
            rgb, k, plan = colour_source(r, source, (smp.get("meta") or {}).get("target") or smp["kind"])
            if rgb is None:
                status = "missing"
            elif rgb.shape[:2] != grey.shape or not np.array_equal(common.to_gray(rgb), grey):
                status = "mismatch"
            else:
                status = "ok"
            if status != "ok":
                rgb = grey3(grey)
                k = lab_k(r, source, smp)
        else:
            rgb, k, plan = colour_source(r, source, it["target"])
            if rgb is None:
                out.append({"key": it["key"], "status": "missing"})
                continue
            grey = common.to_gray(rgb)
            status = "ok"
        t1 = time.perf_counter()
        boxes, t_det, t_ref = detect(rgb)
        ms["colour"] += t1 - t0
        ms["meiki"] += t_det
        ms["refine"] += t_ref
        line = {"key": it["key"], "status": status, "w": int(grey.shape[1]), "h": int(grey.shape[0]),
                "grey_sha256": hashlib.sha256(np.ascontiguousarray(grey).tobytes()).hexdigest(), "k": k,
                "meiki": boxes}
        if arm == "more":
            line["plan"] = [plan[0], list(plan[1])]
        out.append(line)
        greys.append(grey)
    done = [ln for ln in out if "meiki" in ln]
    if greys:
        t3 = time.perf_counter()
        res = winner_c(_W["exe"], _W["blob"], greys)
        ms["winner"] += time.perf_counter() - t3
        for ln, g, boxes in zip(done, greys, res):
            ln["winner"] = [list(b[:6]) for b in boxes]
            if twin_checked(ln["key"]):
                t4 = time.perf_counter()
                ln["twin_equal"] = winner_twin(_W["twin"], g) == [tuple(b) for b in boxes]
                ms["twin"] += time.perf_counter() - t4
    n = max(1, len(done))
    for ln in done:
        ln["ms"] = {k: round(1000 * v / n, 2) for k, v in ms.items()}
    return out


def _work(arg):
    path, job = arg
    return path, work_chunk(job)


def pin_to_slot():
    """teacher/label.py's pinning: the job on the 5 cores of its heavy.sh slot (HEAVY_SLOT), when there is one."""
    from teacher import label as TL
    return TL.pin_to_slot()


# ----------------------------------------------------------------------------- arm b: #97's canvases

def textspot_skip():
    """textspot's skip lists (data/manifests/textspot_shards_v1.json real.measured.skip): (shots, vns, the block)."""
    doc = json.loads(TEXTSPOT_MANIFEST.read_text(encoding="utf-8"))
    skip = doc["real"]["measured"]["skip"]
    return (frozenset(skip["eval_screenshot"]["skip_shots"]), frozenset(skip["linefind_list_vn"]["skip_vns"]),
            skip)


def textdet_shards(source):
    """#97's pinned shards of one source: [(name, json path, npy path, pin)]; the json's sha256 is checked."""
    man = json.loads(TEXTDET_MANIFEST.read_text(encoding="utf-8"))
    out = []
    for s in man["shards"]:
        if s["dir"] != source:
            continue
        base = RH.SHARDS[source] / s["name"]
        js, npy = base.with_suffix(".json"), base.with_suffix(".npy")
        if sha256(js) != s["json_sha256"]:
            raise SystemExit(f"{js}: sha256 is not data/manifests/textdet_shards_v1.json's")
        out.append((s["name"], js, npy, s))
    return out


def lab_items(source, limit=0):
    """[(#97 shard name, npy path, [(index, sample)] kept)] of a source's canvases, and the skip counts."""
    skip_shots, skip_vns, _ = textspot_skip()
    out, skipped, n = [], collections.Counter(), 0
    for name, js, npy, _ in textdet_shards(source):
        kept = []
        for i, smp in enumerate(json.loads(js.read_text(encoding="utf-8"))["samples"]):
            by_shot, by_vn = smp.get("shot") in skip_shots, smp.get("vn") in skip_vns
            if by_shot or by_vn:
                skipped["by_shot"] += by_shot
                skipped["by_vn_only"] += by_vn and not by_shot
                skipped["total"] += 1
                continue
            kept.append((i, smp))
        if limit:
            kept = kept[:max(0, limit - n)]
        n += len(kept)
        out.append((name, str(npy), kept))
        if limit and n >= limit:
            break
    return out, dict(skipped)


def lab_key(name, i):
    return f"{name}:{i}"


# ----------------------------------------------------------------------------- arm c: the new screenshots

def eval_exclusions():
    """What arm c leaves out beyond #97's own exclusions (textdet_data._excluded): the screenshots of every
    eval/manifests and eval/linefind/manifests list but the textspot sets (textspot_real_half.eval_lists, by image
    id and sha256) and every eval/linefind list's VN; every textspot development and test frame's source screenshot
    (the rows of eval/manifests/textspot-{dev,test}.jsonl only, never their frames: by shot id, image id and
    sha256); the realplus pools' thumbnail leak drops (near-duplicates of a test screenshot)."""
    ids, shas, lf_vns = RH.eval_lists()
    ts_ids, ts_shas, ts_shots = set(), set(), set()
    for name in ("textspot-dev", "textspot-test"):
        for r in jsonl(ROOT / "eval" / "manifests" / f"{name}.jsonl"):
            m = URL_ID.search(r.get("url") or "")
            if m:
                ts_ids.add(m.group(1))
            if r.get("sha256"):
                ts_shas.add(r["sha256"])
            if r.get("id"):
                ts_shots.add(re.sub(r"_x2$", "", str(r["id"])))
    leak = set()
    for p in (ART / "realplus" / "manifests" / "leak_drop.json", ART / "realplus-v3" / "manifests" / "leak_drop.json"):
        if p.exists():
            leak |= set(json.loads(p.read_text(encoding="utf-8")))
    return {"ids": set(ids), "shas": set(shas), "lf_vns": set(lf_vns), "ts_ids": ts_ids, "ts_shas": ts_shas,
            "ts_shots": ts_shots, "leak_drop": leak}


def image_id(r):
    m = URL_ID.search(r.get("url") or "")
    if m:
        return m.group(1)
    m = SHOT_ID.match(r.get("shot") or "")
    return m.group(2) if m else None


def group_of(source, r):
    """The cap group of a screenshot: pc (realplus; the teacher's PC shots), native (the teacher's Vita and PSP
    shots; devnative) or owner."""
    if source == "owner":
        return "owner"
    if source == "realplus" or (source == "teacher" and (r or {}).get("kind") not in ("vita", "psp")):
        return "pc"
    return "native"


def owner_test_folder(r):
    return any(part.endswith("-TEST") for part in Path(r.get("path") or "").parts)


def pool_rank(shot):
    return hashlib.sha256((POOL_SALT + shot).encode("utf-8")).hexdigest()


def lab_existing():
    """{(group, vn): {shot}} of the lab canvases (#97's minus textspot's skip lists) and {shot: source} of every
    #97 real canvas (skipped ones included: a screenshot any #97 canvas holds is not new)."""
    skip_shots, skip_vns, _ = textspot_skip()
    held, existing = {}, collections.defaultdict(set)
    for source in SOURCES:
        by = {r["shot"]: r for r in record_rows(source)[0]}
        for _, js, _, _ in textdet_shards(source):
            for smp in json.loads(js.read_text(encoding="utf-8"))["samples"]:
                held.setdefault(smp["shot"], source)
                if smp["shot"] in skip_shots or smp.get("vn") in skip_vns:
                    continue
                existing[(group_of(source, by.get(smp["shot"]) or {"kind": smp["kind"]}), smp["vn"])].add(
                    smp["shot"])
    return existing, held


def cap_select(cand, existing, caps=None):
    """The caps of arm c: cand {(group, vn): {shot: anything}} (the new screenshots), existing {(group, vn): {shot}}
    (the lab screenshots) -> ([(group, vn, shot)] chosen, {group: counts}). Per VN and group, at most caps[group]
    screenshots, the existing ones included; the new ones in pool_rank order (sha256('textspot2-pool:' + shot))."""
    caps = CAPS_MORE if caps is None else caps
    picked, groups = [], collections.defaultdict(collections.Counter)
    for (g, vn), d in sorted(cand.items()):
        room = max(0, caps[g] - len(existing.get((g, vn), ())))
        order = sorted(d, key=pool_rank)
        picked += [(g, vn, shot) for shot in order[:room]]
        groups[g]["vns_with_candidates"] += 1
        groups[g]["candidates"] += len(order)
        groups[g]["new"] += min(room, len(order))
        groups[g]["capped_out"] += max(0, len(order) - room)
    return picked, groups


def more_pool():
    """Arm c's new screenshots: ({source: [record]} in shot order, the selection's record).

    Candidates: every record #97's exclusions keep (record_rows uncapped) of realplus, the teacher, devnative and the
    owner. Left out: a screenshot a #97 real canvas holds (not new), every reserved VN (synth/common.is_reserved_vn:
    test and dev_sample with their release partners), every owner -TEST folder, eval_exclusions() (by image id, the
    row's and the file's sha256, VN), a record without its file, and one whose row sha256 differs from its file's.
    Caps per VN and group, the existing screenshots of the group's lab canvases included: pc 8, native 12, owner 200;
    within a VN the new ones in sha256('textspot2-pool:' + shot) order. A screenshot in two sources' records joins
    the first in SOURCES order."""
    ex = eval_exclusions()
    existing, held = lab_existing()
    cand, why, totals, seen = collections.defaultdict(dict), collections.Counter(), {}, set()
    for source in SOURCES:
        rows, total = record_rows(source, uncapped=True)
        totals[source] = {"records": total, "kept_by_97s_exclusions": len(rows)}
        for r in rows:
            shot = r["shot"]
            iid = image_id(r) if source != "owner" else None
            if shot in held:
                why["held_by_a_97_canvas"] += 1
            elif shot in seen:
                why["in_an_earlier_source"] += 1
            elif common.is_reserved_vn(r["vn"]):
                why["reserved_vn"] += 1
            elif source == "owner" and owner_test_folder(r):
                why["owner_test_folder"] += 1
            elif r["vn"] in ex["lf_vns"]:
                why["linefind_list_vn"] += 1
            elif (iid and iid in ex["ids"]) or r.get("sha256") in ex["shas"]:
                why["eval_list_screenshot"] += 1
            elif shot in ex["ts_shots"] or (iid and iid in ex["ts_ids"]) or r.get("sha256") in ex["ts_shas"]:
                why["textspot_frame_source"] += 1
            elif shot in ex["leak_drop"]:
                why["thumbnail_leak_drop"] += 1
            elif not r.get("path") or not Path(r["path"]).exists():
                why["file_missing"] += 1
            else:
                fsha = sha256(r["path"])
                if fsha in ex["shas"] or fsha in ex["ts_shas"]:
                    why["eval_or_textspot_file_sha256"] += 1
                elif r.get("sha256") and r["sha256"] != fsha:
                    why["row_sha256_is_not_the_file"] += 1
                else:
                    seen.add(shot)
                    cand[(group_of(source, r), r["vn"])][shot] = (source, dict(r, file_sha256=fsha))
    chosen = collections.defaultdict(list)
    picked, groups = cap_select(cand, existing)
    for g, vn, shot in picked:
        source, r = cand[(g, vn)][shot]
        chosen[source].append(r)
    for g in groups:
        groups[g]["lab_vns"] = len({vn for (gg, vn) in existing if gg == g})
        groups[g]["lab_screenshots"] = sum(len(v) for (gg, _), v in existing.items() if gg == g)
    out = {s: sorted(chosen.get(s, []), key=lambda r: r["shot"]) for s in SOURCES}
    rec = {"made": utc(), "candidates": totals, "left_out": dict(sorted(why.items())),
           "groups": {g: dict(v) for g, v in sorted(groups.items())}, "caps": CAPS_MORE,
           "order": f"sha256('{POOL_SALT}' + shot) within a VN", "new_by_source": {s: len(v) for s, v in out.items()},
           "new_vns_by_source": {s: len({r['vn'] for r in v}) for s, v in out.items()},
           "exclusion_sets": {"eval_list_image_ids": len(ex["ids"]), "eval_list_sha256": len(ex["shas"]),
                              "linefind_list_vns": len(ex["lf_vns"]), "textspot_frame_sources": len(ex["ts_shots"]),
                              "textspot_frame_sha256": len(ex["ts_shas"]),
                              "thumbnail_leak_drop": len(ex["leak_drop"])}}
    return out, rec


POOL_FIELDS = ("shot", "vn", "kind", "url", "release", "session", "sha256", "file_sha256", "dims")


def pool_path(source):
    return cache_dir("more", source) / "pool.jsonl"


def write_pool(pool, rec):
    for source, rows in pool.items():
        p = pool_path(source)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                row = {k: r.get(k) for k in POOL_FIELDS}
                row["source"] = source
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(p)
    jdump(rec, T2 / "labels" / "more" / "pool_record.json", indent=1)


def more_rows(source):
    """The pool's rows of a source joined to their full records (record_rows uncapped), in pool order."""
    pool = list(jsonl(pool_path(source)))
    if not pool:
        return []
    by = {r["shot"]: r for r in record_rows(source, uncapped=True)[0]}
    out = []
    for p in pool:
        r = by.get(p["shot"])
        if r is None:
            raise SystemExit(f"{source} {p['shot']}: in the pool but not in the records")
        out.append(dict(r, file_sha256=p["file_sha256"]))
    return out


def more_target(source, r):
    """Arm c's canvas target: the Vita target for PC screenshots (as #97 made its realplus canvases), else the
    record's own screen (real_canvases' plan)."""
    if source == "teacher" and r["kind"] not in ("vita", "psp"):
        return "vita"
    return "psp" if r["kind"] == "psp" else "vita"


def more_shards(source):
    """[(shard name, [records])] of a source's pool: PER_SHARD_MORE screenshots (one canvas each) per shard."""
    rows = more_rows(source)
    return [(f"more_{source}_{i // PER_SHARD_MORE:04d}", rows[i:i + PER_SHARD_MORE])
            for i in range(0, len(rows), PER_SHARD_MORE)]


# ----------------------------------------------------------------------------- the cache stage

def cached_keys(path):
    return {ln["key"] for ln in jsonl(path) if ln.get("status") == "missing" or "winner" in ln}


def cache_shards(arm, source, limit=0):
    """[(cache file name, items)] of one arm and source."""
    shards = []
    if arm == "lab":
        by = {r["shot"]: r for r in record_rows(source)[0]}
        for name, npy, kept in lab_items(source, limit)[0]:
            shards.append((name, [{"key": lab_key(name, i), "npy": npy, "sample": smp,
                                   "record": by.get(smp["shot"])} for i, smp in kept]))
        return shards
    n = 0
    for name, rows in more_shards(source):
        if limit:
            rows = rows[:max(0, limit - n)]
        n += len(rows)
        shards.append((name, [{"key": r["shot"], "record": r, "target": more_target(source, r)} for r in rows]))
        if limit and n >= limit:
            break
    return shards


def run_cache(arm, source, workers, limit=0, chunk=10, cdir=None, shards=None):
    """Fill one arm's and source's cache (it resumes: cached canvases are not redone). Returns a record."""
    cdir = Path(cdir) if cdir else cache_dir(arm, source)
    cdir.mkdir(parents=True, exist_ok=True)
    if sha256(WINNER) != WINNER_SHA256:
        raise SystemExit(f"{WINNER}: not the textspot winner's sha256")
    if not (CBUILD / "vocr_textspot_test").exists():
        raise SystemExit(f"{CBUILD}/vocr_textspot_test missing: eval/textspot.py cbuild --out {CBUILD}")
    shards = cache_shards(arm, source, limit) if shards is None else shards
    jobs, canvases = [], 0
    for name, items in shards:
        path = cdir / f"{name}.jsonl"
        done = cached_keys(path)
        todo = [it for it in items if it["key"] not in done]
        canvases += len(items)
        for i in range(0, len(todo), chunk):
            jobs.append((str(path), {"arm": arm, "source": source, "items": todo[i:i + chunk]}))
    total = sum(len(j[1]["items"]) for j in jobs)
    print(f"{utc()} cache {arm}/{source}: {canvases} canvases in {len(shards)} shards, {total} to do, "
          f"{workers} workers", flush=True)
    t0, n_done, files = time.time(), 0, {}
    if jobs:
        with mp.get_context("fork").Pool(workers, initializer=init_worker,
                                          initargs=(str(CBUILD), str(WINNER), RULES["det_thr"])) as pool:
            for path, lines in pool.imap_unordered(_work, jobs):
                f = files.get(path)
                if f is None:
                    f = files[path] = open(path, "a", encoding="utf-8")
                for ln in lines:
                    f.write(json.dumps(ln, ensure_ascii=False) + "\n")
                f.flush()
                n_done += len(lines)
                if n_done % 500 < len(lines) or n_done == total:
                    dt = time.time() - t0
                    print(f"{utc()} {arm}/{source}: {n_done}/{total} canvases, {dt:.0f} s, "
                          f"{1000 * dt / n_done:.1f} ms per canvas (wall)", flush=True)
        for f in files.values():
            f.close()
    dt = time.time() - t0
    return {"arm": arm, "source": source, "canvases": canvases, "done_now": n_done, "seconds": round(dt, 1),
            "workers": workers}


# ----------------------------------------------------------------------------- the build stage

class Stats:
    """Per-source statistics of a build (counts only)."""

    def __init__(self):
        self.c = collections.Counter()
        self.w = collections.Counter()

    def add(self, st, ln, n_lines, n_ignore):
        self.c.update(st["counts"])
        self.w.update(st["weights"])
        self.c["canvases"] += 1
        self.c["lines"] += n_lines
        self.c["ignore_total"] += n_ignore
        self.c["winner_boxes"] += len(ln.get("winner") or [])
        self.c[f"colour_{ln['status']}"] += 1
        if "twin_equal" in ln:
            self.c["twin_checked"] += 1
            self.c["twin_equal"] += bool(ln["twin_equal"])

    def as_dict(self):
        c = self.c
        return {"canvases": c["canvases"], "skipped": {k[5:]: c[k] for k in sorted(c) if k.startswith("skip_")},
                "colour_source": {"ok": c["colour_ok"], "missing": c["colour_missing"],
                                  "mismatch": c["colour_mismatch"]},
                "positives": {"kept": c["positives_kept"], "added": c["positives_added"],
                              "total": c["positives_kept"] + c["positives_added"], "vertical_kept": c["vertical_kept"],
                              "vertical_added": c["vertical_added"]},
                "weights_added": {k: self.w.get(k, 0) for k in WEIGHT_BINS},
                "ignore": {"kept": c["ignore_kept"], "removed": c["ignore_removed"],
                           "added": {k: c[f"ignore_{k}"] for k in IGNORE_KINDS}, "total": c["ignore_total"]},
                "meiki": {"boxes": c["meiki"], "score_ge_0.5": c["meiki_high"], "score_0.2_0.5": c["meiki_low"],
                          "high_covered_by_existing": c["high_covered"], "empty_after_clip": c["meiki_empty"],
                          "low_dropped_covered": c["low_dropped_covered"]},
                "winner": {"boxes": c["winner_boxes"], "twin_checked": c["twin_checked"],
                           "twin_equal": c["twin_equal"]},
                "lines": c["lines"]}


def to_meiki(line):
    return [{"raw": m["raw"], "score": m["score"], "ref_h": m["ref_h"], "ref_t": m["ref_t"]} for m in line["meiki"]]


def winner_xyxy(line):
    return [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in line.get("winner") or []]


def link_npy(link, target):
    link, target = Path(link), Path(target).resolve()
    if link.is_symlink() and Path(os.readlink(link)) == target:
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def build_lab(source, cdir=None, out=None, limit=0):
    """Arm b's shards of one source from its cache: one json per #97 shard, its npy a symlink to #97's."""
    by = {r["shot"]: r for r in record_rows(source)[0]}
    items, skipped = lab_items(source, limit)
    stats = Stats()
    for k, v in skipped.items():
        stats.c[f"skip_{k}"] += v
    cdir = Path(cdir) if cdir else cache_dir("lab", source)
    out = Path(out) if out else shard_dir("lab", source)
    out.mkdir(parents=True, exist_ok=True)
    written, per_canvas = [], []
    for name, npy, kept in items:
        cache = {ln["key"]: ln for ln in jsonl(cdir / f"{name}.jsonl")}
        samples = []
        for i, smp in kept:
            ln = cache.get(lab_key(name, i))
            if ln is None or "winner" not in ln:
                raise SystemExit(f"lab/{source} {name}:{i}: not cached")
            if (ln["w"], ln["h"]) != (smp["w"], smp["h"]):
                raise SystemExit(f"lab/{source} {name}:{i}: the cached size is not the canvas's")
            r = by.get(smp["shot"])
            furi = record_furi(r, ln["k"], smp["w"], smp["h"]) if ln.get("k") is not None else []
            lines, ignore, st = build_labels(smp["w"], smp["h"], smp["lines"], smp["ignore"], furi, to_meiki(ln),
                                             winner_xyxy(ln))
            stats.add(st, ln, len(lines), len(ignore))
            meta = dict(smp.get("meta") or {})
            meta["t2"] = {"colour": ln["status"], "added": st["counts"].get("positives_added", 0),
                          "ignore_added": {k: v for k, v in st["ignore_kinds"].items() if v},
                          "ignore_removed": st["counts"].get("ignore_removed", 0)}
            samples.append({**{k: smp[k] for k in ("off", "h", "w", "src", "kind", "vn", "shot")},
                            "lines": lines, "ignore": ignore, "meta": meta})
            per_canvas.append({"key": ln["key"], "ms": ln.get("ms")})
        jdump({"samples": samples}, out / f"{name}.json")
        link_npy(out / f"{name}.npy", npy)
        written.append(name)
    return {"arm": "lab", "source": source, "dir": out.name, "shards": written, "stats": stats.as_dict(),
            "rules": RULES, "built": utc()}


def build_more(source, cdir=None, out=None, limit=0):
    """Arm c's shards of one source from its cache: the canvases written (npy) with their labels."""
    stats = Stats()
    cdir = Path(cdir) if cdir else cache_dir("more", source)
    out = Path(out) if out else shard_dir("more", source)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in cache_shards("more", source, limit):
        cache = {ln["key"]: ln for ln in jsonl(cdir / f"{name}.jsonl")}
        samples = []
        for it in rows:
            r = it["record"]
            ln = cache.get(it["key"])
            if ln is None:
                raise SystemExit(f"more/{source} {r['shot']}: not cached")
            if ln["status"] == "missing":
                stats.c["colour_missing"] += 1
                continue
            rgb, k, plan = colour_source(r, source, it["target"])
            grey = common.to_gray(rgb)
            if hashlib.sha256(np.ascontiguousarray(grey).tobytes()).hexdigest() != ln["grey_sha256"]:
                raise SystemExit(f"more/{source} {r['shot']}: the canvas is not the one meiki read")
            H, W = grey.shape
            lines, ignore, st = build_labels(W, H, [], [], record_furi(r, k, W, H), to_meiki(ln), winner_xyxy(ln))
            stats.add(st, ln, len(lines), len(ignore))
            src = read_rgb(r["path"])
            meta = {"target": plan[0], "scale": round(W / src.shape[1], 4), "src_size": [src.shape[1], src.shape[0]],
                    "t2": {"colour": "ok", "added": st["counts"].get("positives_added", 0),
                           "ignore_added": {kk: v for kk, v in st["ignore_kinds"].items() if v}}}
            samples.append({"canvas": grey, "lines": lines, "ignore": ignore, "meta": meta, "src": source,
                            "kind": plan[0], "vn": r["vn"], "shot": r["shot"]})
        if samples:
            TD.write_shard(out / name, samples)
            written.append(name)
    return {"arm": "more", "source": source, "dir": out.name, "shards": written, "stats": stats.as_dict(),
            "rules": RULES, "built": utc()}


def cmd_build(args):
    rec = (build_lab if args.arm == "lab" else build_more)(args.source)
    jdump(rec, cache_dir(args.arm, args.source) / "build.json", indent=1)
    print(json.dumps(rec["stats"], indent=1), flush=True)
    print(f"TEXTSPOT2_BUILD_DONE {args.arm}/{args.source} {len(rec['shards'])} shards", flush=True)


# ----------------------------------------------------------------------------- manifest

CODE = ("synth/textspot2_labels.py", "synth/textdet_data.py", "synth/textspot_real_half.py", "synth/common.py",
        "synth/textspot_twin.py", "synth/textdet_twin.py", "teacher/geometry.py", "teacher/engines.py",
        "teacher/label.py", "eval/textspot.py", "eval/tools/reserve_check.py", "eval/tools/leakcheck.py",
        "runtime/vocr_textspot.c", "runtime/vocr_textspot.h", "runtime/vocr_textspot_test.c", "runtime/vocr_lines.c",
        "runtime/vocr_lines_rows.c", "train/textspot_train.py")


def list_sha256(items):
    """train/textspot_train.list_sha256: sha256 of a list's items, each followed by a newline, in order."""
    return hashlib.sha256("".join(f"{x}\n" for x in items).encode("utf-8")).hexdigest()


def shard_entry(arm, source, js):
    idx = json.loads(Path(js).read_text(encoding="utf-8"))["samples"]
    npy = Path(js).with_suffix(".npy")
    return {"dir": Path(js).parent.name, "name": Path(js).stem, "arm": arm, "source": source,
            "npy_sha256": sha256(npy), "json_sha256": sha256(js),
            "npy_symlink_to": herd_rel(os.readlink(npy)) if npy.is_symlink() else None,
            "samples": len(idx), "lines": sum(len(s["lines"]) for s in idx),
            "vertical": sum(int(ln[4]) for s in idx for ln in s["lines"]),
            "ignore": sum(len(s["ignore"]) for s in idx), "by_src": dict(collections.Counter(s["src"] for s in idx))}


def skip_block(shards):
    """real.measured.skip: textspot's two lists, copied, with their sha256 (train/textspot_train.list_sha256), and
    the canvases of these shards they still hold (0: dropped when the shards were built)."""
    _, _, skip = textspot_skip()
    ev, lf = skip["eval_screenshot"], skip["linefind_list_vn"]
    shots, vns = set(ev["skip_shots"]), set(lf["skip_vns"])
    held = {"eval_screenshot": collections.Counter(), "linefind_list_vn": collections.Counter()}
    for e in shards:
        js = shard_dir(e["arm"], e["source"]) / f"{e['name']}.json"
        for s in json.loads(js.read_text(encoding="utf-8"))["samples"]:
            if s.get("shot") in shots:
                held["eval_screenshot"][s["src"]] += 1
            elif s.get("vn") in vns:
                held["linefind_list_vn"][s["src"]] += 1
    out = {"copied_from": {"path": "data/manifests/textspot_shards_v1.json", "sha256": sha256(TEXTSPOT_MANIFEST),
                           "key": "real.measured.skip"},
           "how": "the lists are textspot's; the shards were built without the canvases they name, so shots, "
                  "canvases and by_source count what these shards still hold (0); textspot's own counts of #97's "
                  "canvases are kept as textspot_counts"}
    for key, items, name in (("eval_screenshot", ev["skip_shots"], "skip_shots"),
                             ("linefind_list_vn", lf["skip_vns"], "skip_vns")):
        src = skip[key]
        h = held[key]
        out[key] = {name: list(items), f"{name}_sha256": list_sha256(items), "lists": src.get("lists"),
                    "shots": 0 if not h else None, "canvases": sum(h.values()),
                    "by_source": {s: h.get(s, 0) for s in SOURCES},
                    "textspot_counts": {k: src.get(k) for k in ("shots", "canvases", "by_source")}}
    return out


def meiki_record():
    import importlib.metadata as md

    from teacher import engines as E
    src, fname, want = E.PINS["meiki_det"]
    path = E.model_path("meiki_det")
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got} is not the pinned {want}")
    return {"class": "teacher/engines.Meiki", "repo": src[1], "revision": src[2], "file": fname, "sha256": got,
            "det_thr": RULES["det_thr"], "comparison": "score > det_thr (Meiki.ocr)", "input": "BGR (cv2.imread's "
            "order, as teacher/label.py feeds Meiki; Meiki.rgb False)", "recognition": False,
            "preprocessing": "MeikiOCR's _preprocess_for_detection (fit to 960x544, INTER_LINEAR, /255, top-left "
                             "padded)", "threads": 1, "packages": {k: md.version(k) for k in (
                                 "meikiocr", "onnxruntime", "numpy", "opencv-python-headless")}}


def totals_of(shards, arm=None):
    sel = [e for e in shards if arm is None or e["arm"] == arm]
    by = collections.Counter()
    for e in sel:
        by.update(e["by_src"])
    return {"shards": len(sel), "samples": sum(e["samples"] for e in sel), "lines": sum(e["lines"] for e in sel),
            "vertical": sum(e["vertical"] for e in sel), "ignore": sum(e["ignore"] for e in sel),
            "by_src": dict(sorted(by.items()))}


def cmd_manifest(args):
    textdet = json.loads(TEXTDET_MANIFEST.read_text(encoding="utf-8"))
    pins = {(s["dir"], s["name"]): s for s in textdet["shards"]}
    shards, stats, builds = [], {a: {} for a in ARMS}, {}
    for arm in ARMS:
        for source in SOURCES:
            rec_p = cache_dir(arm, source) / "build.json"
            if not rec_p.exists():
                continue
            rec = json.loads(rec_p.read_text(encoding="utf-8"))
            stats[arm][source] = rec["stats"]
            builds[f"{arm}/{source}"] = {"built": rec["built"], "shards": len(rec["shards"])}
            for name in rec["shards"]:
                e = shard_entry(arm, source, shard_dir(arm, source) / f"{name}.json")
                if arm == "lab":
                    pin = pins[(source, name)]
                    if e["npy_sha256"] != pin["npy_sha256"]:
                        raise SystemExit(f"{name}.npy: not #97's canvases")
                    e["textdet_json_sha256"] = pin["json_sha256"]
                shards.append(e)
    rc_path = T2 / "labels" / "reserve_check.json"
    rc = json.loads(rc_path.read_text(encoding="utf-8")) if rc_path.exists() else None
    leak_path = T2 / "labels" / "more" / "leakcheck.json"
    cb = json.loads((CBUILD / "build.json").read_text(encoding="utf-8"))
    pool_rec_p = T2 / "labels" / "more" / "pool_record.json"
    doc = {"kind": "textspot2-shards", "schema": 1, "issue": 106, "generator": "synth/textspot2_labels.py",
           "created": utc(),
           "protocol": {"path": "eval/textspot2_protocol.json", "sha256": sha256(PROTOCOL), "amendment": 1},
           "code_sha256": {f: sha256(ROOT / f) for f in CODE},
           "inputs": {"textdet_manifest": {"path": "data/manifests/textdet_shards_v1.json",
                                           "sha256": sha256(TEXTDET_MANIFEST)},
                      "textspot_manifest": {"path": "data/manifests/textspot_shards_v1.json",
                                            "sha256": sha256(TEXTSPOT_MANIFEST)},
                      "reserved_sha256": sha256(ROOT / "eval" / "manifests" / "reserved.json")},
           "meiki": meiki_record(),
           "winner": {"blob": "v0.3.3-textspot.1/textspot_s4_c24n4_w8.vtsp", "sha256": sha256(WINNER),
                      "decode": "det (the textspot decode, no refinement)", "on": "the grey canvas",
                      "driver": "runtime/vocr_textspot_test --batch, built by eval/textspot.py cbuild",
                      "cbuild": cb, "twin_check": f"one canvas in {RULES['winner_twin_check_every']} (by key hash) "
                                                  "through synth/textspot_twin.spot too"},
           "rules": RULES, "caps_more": CAPS_MORE, "pool_order": f"sha256('{POOL_SALT}' + shot) within a VN",
           "shard_dirs": {f"{a}/{s}": herd_rel(shard_dir(a, s)) for a in ARMS for s in SOURCES},
           "real": {"measured": {"skip": skip_block(shards)}},
           "pool": json.loads(pool_rec_p.read_text(encoding="utf-8")) if pool_rec_p.exists() else None,
           "reserve_check": None if rc is None else {**rc, "summary": herd_rel(rc_path),
                                                     "summary_sha256": sha256(rc_path)},
           "leakcheck": json.loads(leak_path.read_text(encoding="utf-8")) if leak_path.exists() else None,
           "builds": builds, "stats": stats, "shards": shards,
           "totals": {"lab": totals_of(shards, "lab"), "more": totals_of(shards, "more"), "all": totals_of(shards)}}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {json.dumps(doc['totals']['all'])}", flush=True)


# ----------------------------------------------------------------------------- reserve and thumbnail checks

def lab_rows(source):
    """The lab canvases' screenshots of a source (distinct shots, #97's order) with their record's identity fields and
    their file's sha256 (computed now): the rows reserve_check reads for arm b."""
    by = {r["shot"]: r for r in record_rows(source)[0]}
    seen, out = set(), []
    for _, _, kept in lab_items(source)[0]:
        for _, smp in kept:
            if smp["shot"] in seen:
                continue
            seen.add(smp["shot"])
            r = by.get(smp["shot"]) or {"shot": smp["shot"], "vn": smp["vn"]}
            row = {k: r.get(k) for k in POOL_FIELDS if k != "file_sha256"}
            row["file_sha256"] = sha256(r["path"]) if r.get("path") and Path(r["path"]).exists() else None
            row["source"] = source
            out.append(row)
    return out


def cmd_reserve(args):
    """eval/tools/reserve_check.py over the lab canvases' screenshots (conductor 09:30Z) and arm c's new pool, one
    pool per arm and source, with --require-hashes (every row carries its file's sha256) and --pixels as #97's
    hygiene ran it; the record under $T2/labels/reserve_check.json (counts, verdicts; owner keys stay in $P2)."""
    pools = []
    for arm in ARMS:
        for s in SOURCES:
            rows = lab_rows(s) if arm == "lab" else list(jsonl(pool_path(s)))
            if not rows:
                continue
            q = cache_dir(arm, s) / "reserve_rows.jsonl"
            q.parent.mkdir(parents=True, exist_ok=True)
            with open(q, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({**r, "sha256": r["file_sha256"] or r.get("sha256")}, ensure_ascii=False)
                            + "\n")
            pools.append(f"{arm}_{s}={q}")
    out = P2 / "labels" / "reserve_check.json"
    cmd = [sys.executable, str(ROOT / "eval" / "tools" / "reserve_check.py"), *pools, "--require-hashes",
           "--out", str(out)] + (["--pixels"] if args.pixels else [])
    print(" ".join(herd_rel(c) for c in cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=7200)
    print(p.stdout, p.stderr[-3000:], flush=True)
    last = (p.stdout.strip().splitlines() or [""])[-1]
    rc = json.loads(out.read_text(encoding="utf-8"))
    public = {"verdict": rc["verdict"], "line": last, "pixels": rc.get("pixels"),
              "dev_sample_mode": rc["dev_sample_mode"], "reservation_dump": rc.get("reservation_dump"),
              "argv": ["eval/tools/reserve_check.py", *[x.split("=")[0] + "=<rows>" for x in pools],
                       "--require-hashes"] + (["--pixels"] if args.pixels else []),
              "record_private": herd_rel(out), "record_sha256": sha256(out),
              "pools": {k: {"rows": v["rows"], "vns": v["vns"], "hash_coverage": v["hash_coverage"],
                            "blockers": v["blockers"],
                            "info": {c: x["rows"] for c, x in v["checks"].items() if c.startswith("info")}}
                        for k, v in rc["pools"].items()}}
    jdump(public, T2 / "labels" / "reserve_check.json", indent=1)
    print(f"reserve_check: {last}", flush=True)
    return 0 if last == "RESERVE_OK" else 1


def cmd_leak(args):
    """eval/tools/leakcheck.py on the new pool's screenshots (symlinks named <vn>_<shot>.<ext> under $T2/labels/more
    and $P2/labels/more, so its reserved-game skip and same-vn tag work): 32x18 grey thumbnails against the
    vita-test and psp-test screenshots, correlation >= 0.95 listed."""
    dirs = []
    for s in SOURCES:
        rows = more_rows(s)
        if not rows:
            continue
        d = cache_dir("more", s) / "leak_links"
        d.mkdir(parents=True, exist_ok=True)
        for r in rows:
            src = Path(r["path"])
            name = r["shot"] if r["shot"].startswith(r["vn"] + "_") else f"{r['vn']}_{r['shot']}"
            link = d / f"{name}{src.suffix.lower()}"
            if not link.is_symlink():
                link.symlink_to(src)
        dirs.append(str(d))
    cmd = [sys.executable, str(ROOT / "eval" / "tools" / "leakcheck.py"), *dirs, "--min-corr", "0.95",
           "--data", str(ART / "eval")]
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=7200)
    out = p.stdout.strip().splitlines()
    print("\n".join(herd_rel(x) for x in out[-30:]), p.stderr[-2000:], flush=True)
    summary = None
    for x in reversed(out):
        if x.startswith("{"):
            summary = json.loads(x)
            break
    pairs = [herd_rel(x.strip()) for x in out if " ~ " in x]
    jdump({"tool": "eval/tools/leakcheck.py", "min_corr": 0.95, "rc": p.returncode, "summary": summary,
           "pairs": pairs, "spread": [x for x in out if x.startswith("best pool correlation")]},
          T2 / "labels" / "more" / "leakcheck.json", indent=1)
    return p.returncode


# ----------------------------------------------------------------------------- QA contact sheets

def sheet_items(arm, sources, sheet, n=40):
    """n canvases of one sheet: a deterministic draw (by key hash) over the given sources' shards."""
    pool = []
    for s in sources:
        d = shard_dir(arm, s)
        for js in sorted(d.glob("*.json")):
            if js.name.endswith(".tmp"):
                continue
            for i, smp in enumerate(json.loads(js.read_text(encoding="utf-8"))["samples"]):
                key = f"{arm}:{js.stem}:{i}"
                pool.append((hashlib.sha256(f"textspot2-sheet:{key}".encode()).hexdigest(), js, smp))
    pool.sort(key=lambda t: t[0])
    return pool[sheet * n:(sheet + 1) * n]


def cmd_sheet(args):
    """A QA contact sheet (<= 1280 px a side, 5 x 8 canvases): existing positives green, new meiki positives blue
    with their weight, vertical positives with a second inner frame, ignore areas grey hatched, furigana ignores
    orange (the build's ignore order: kept, then the IGNORE_KINDS)."""
    from PIL import Image, ImageDraw, ImageFont
    sources = args.sources.split(",")
    items = sheet_items(args.arm, sources, args.sheet, args.n)
    cols, tw, th, cap = 5, 256, 145, 12
    rows = (len(items) + cols - 1) // cols
    head = 22
    sheet = Image.new("RGB", (cols * tw, head + rows * (th + cap)), (30, 30, 30))
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 9)
    except OSError:
        font = ImageFont.load_default()
    dr.text((4, 4), f"{args.arm} sheet {args.sheet} ({','.join(sources)}): green existing, blue new meiki (weight), "
                    "grey hatch ignore, orange furigana ignore, inner frame vertical", fill=(255, 255, 255), font=font)
    index = []
    for k, (_, js, smp) in enumerate(items):
        data = np.load(js.with_suffix(".npy"), mmap_mode="r")
        g = np.asarray(data[smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(smp["h"], smp["w"])
        im = Image.fromarray(g).convert("RGB")
        s = min(tw / smp["w"], th / smp["h"])
        im = im.resize((max(1, int(smp["w"] * s)), max(1, int(smp["h"] * s))), Image.BILINEAR)
        d = ImageDraw.Draw(im, "RGBA")
        kinds = ign_kinds_of(smp)
        for b, kind in zip(smp["ignore"], kinds):
            x0, y0, x1, y1 = (int(round(v * s)) for v in b)
            col = (255, 150, 0, 255) if kind.startswith("furigana") else (170, 170, 170, 255)
            d.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=col)
            for t in range(x0 - (y1 - y0), x1, 4):                      # hatch
                d.line((max(x0, t), y0 + max(0, x0 - t), min(x1, t + (y1 - y0)), y0 + min(y1 - y0, x1 - t)),
                       fill=col[:3] + (110,))
        n_ex = len(smp["lines"]) - (smp.get("meta") or {}).get("t2", {}).get("added", 0)
        for j, ln in enumerate(smp["lines"]):
            x0, y0, x1, y1 = (int(round(v * s)) for v in ln[:4])
            new = j >= n_ex
            col = (40, 120, 255) if new else (0, 230, 0)
            d.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=col)
            if int(ln[4]):
                d.rectangle((x0 + 1, y0 + 1, max(x0 + 1, x1 - 2), max(y0 + 1, y1 - 2)), outline=col)
            if new:
                d.text((x0, max(0, y0 - 9)), f"{ln[6]:.2f}", fill=(120, 190, 255), font=font)
        x, y = (k % cols) * tw, head + (k // cols) * (th + cap)
        sheet.paste(im, (x, y))
        t2 = (smp.get("meta") or {}).get("t2", {})
        dr.text((x + 2, y + th), f"{k} {smp['src'][:4]} +{t2.get('added', 0)} -{t2.get('ignore_removed', 0)} "
                                 f"{t2.get('colour', '')}", fill=(220, 220, 220), font=font)
        index.append({"tile": k, "shard": js.stem, "dir": js.parent.name, "off": smp["off"], "vn": smp["vn"],
                      "shot": smp["shot"] if smp["src"] != "owner" else None, "src": smp["src"],
                      "lines": len(smp["lines"]), "added": t2.get("added", 0), "ignore": len(smp["ignore"])})
    sheet.thumbnail((1280, 1280))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    jdump({"sheet": args.sheet, "arm": args.arm, "sources": sources, "tiles": index},
          Path(args.out).with_suffix(".json"), indent=1)
    print(args.out, sheet.size, flush=True)


def ign_kinds_of(smp):
    """The kind of each ignore area of a built sample: kept ones first, then the build's IGNORE_KINDS in order."""
    t2 = (smp.get("meta") or {}).get("t2", {})
    added = t2.get("ignore_added", {})
    n_added = sum(added.values())
    kinds = ["kept"] * (len(smp["ignore"]) - n_added)
    for k in IGNORE_KINDS:
        kinds += [k] * added.get(k, 0)
    return kinds


# ----------------------------------------------------------------------------- probe

def qa_population(rules=None):
    """Over every lab canvas (the cache, built in memory with `rules`): (a) the winner-only boxes, (b) the low-score
    boxes the existing positives do not cover, (c) the new meiki positives: {kind: [(source, shard, npy, index,
    sample, box, info)]}."""
    rules = dict({"low_covered": "drop"}, **(rules or {}))
    pop = {"winner_only": [], "low_uncovered": [], "new": []}
    for source in SOURCES:
        by = {r["shot"]: r for r in record_rows(source)[0]}
        for name, npy, kept in lab_items(source)[0]:
            cache = {ln["key"]: ln for ln in jsonl(cache_path("lab", source, name))}
            for i, smp in kept:
                ln = cache[lab_key(name, i)]
                r = by.get(smp["shot"])
                furi = record_furi(r, ln["k"], smp["w"], smp["h"]) if ln.get("k") is not None else []
                _, _, st = build_labels(smp["w"], smp["h"], smp["lines"], smp["ignore"], furi, to_meiki(ln),
                                        winner_xyxy(ln), rules)
                d = st["detail"]
                conf = {(b[0], b[1], b[0] + b[2], b[1] + b[3]): b[5] for b in ln["winner"]}
                ref = (source, name, npy, i, smp)
                pop["winner_only"] += [(*ref, wb, {"conf": conf.get(tuple(wb))}) for wb in d["winner_only"]]
                pop["low_uncovered"] += [(*ref, b, {"score": s}) for b, s in d["low"]]
                pop["new"] += [(*ref, p["box"], {k: p[k] for k in ("score", "w", "agrees", "vertical")})
                               for p in d["new"]]
    return pop


def crop_tile(rgb, box, vertical, tw, th, up=3.0):
    """The box with about one line of context around it, scaled to fit tw x th (at most `up` x), its box drawn."""
    from PIL import Image, ImageDraw
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    a = max(8, (x1 - x0) if vertical else (y1 - y0))
    cx0, cy0, cx1, cy1 = max(0, x0 - a), max(0, y0 - a), min(W, x1 + a), min(H, y1 + a)
    im = Image.fromarray(np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1]))
    s = min(tw / im.width, th / im.height, up)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle((int((x0 - cx0) * s) - 1, int((y0 - cy0) * s) - 1, int((x1 - cx0) * s), int((y1 - cy0) * s)),
                outline=(255, 0, 0))
    return im


def cmd_qa(args):
    """Three pre-build QA sheets (the conductor's 09:30Z request): 40 random (a) winner-only boxes, (b) uncovered
    low-score boxes, (c) new meiki positives with their weight, drawn over every lab canvas (seed --seed); each crop
    the box (red) with about one line of context, in colour. Written under $P2/labels/qa/ (owner crops can be in
    them), with an index json per sheet."""
    from PIL import Image, ImageDraw, ImageFont
    pop = qa_population()
    out = P2 / "labels" / "qa"
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    rows_by = {s: {r["shot"]: r for r in record_rows(s)[0]} for s in SOURCES}
    counts = {k: len(v) for k, v in pop.items()}
    for kind, letter in (("winner_only", "a"), ("low_uncovered", "b"), ("new", "c")):
        items = pop[kind]
        rng = np.random.default_rng(args.seed)
        pick = sorted(rng.choice(len(items), size=min(args.n, len(items)), replace=False).tolist())
        cols, tw, th, cap, head = 4, 320, 104, 13, 20
        nrow = (len(pick) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, head + nrow * (th + cap)), (25, 25, 25))
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 4), f"({letter}) {kind}: {len(pick)} of {len(items)} lab boxes (seed {args.seed}); red = the box",
                fill=(255, 255, 255), font=font)
        index = []
        for k, j in enumerate(pick):
            source, name, npy, i, smp, box, info = items[j]
            r = rows_by[source].get(smp["shot"])
            rgb = colour_source(r, source, (smp.get("meta") or {}).get("target") or smp["kind"])[0]
            if rgb is None or rgb.shape[:2] != (smp["h"], smp["w"]):
                data = np.load(npy, mmap_mode="r")
                rgb = grey3(np.asarray(data[smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(smp["h"],
                                                                                                    smp["w"]))
            vertical = bool(info.get("vertical")) or tall(box)
            tile = crop_tile(rgb, box, vertical, tw - 4, th - 2)
            x, y = (k % cols) * tw, head + (k // cols) * (th + cap)
            sheet.paste(tile, (x + 2, y + 1))
            label = f"{k} {source[:4]} {box[2] - box[0]}x{box[3] - box[1]}"
            if "score" in info:
                label += f" s{info['score']:.2f}"
            if "w" in info:
                label += f" w{info['w']:.2f}{'*' if info['agrees'] else ''}{' V' if info['vertical'] else ''}"
            if "conf" in info:
                label += f" c{info['conf']}"
            dr.text((x + 2, y + th), label, fill=(230, 230, 230), font=font)
            index.append({"tile": k, "source": source, "shard": name, "index": i, "vn": smp["vn"],
                          "shot": smp["shot"] if source != "owner" else None, "box": box, **info})
        sheet.thumbnail((1280, 1280))
        path = out / f"qa_{letter}_{kind}.png"
        sheet.save(path)
        jdump({"kind": kind, "population": len(items), "seed": args.seed, "tiles": index},
              path.with_suffix(".json"), indent=1)
        print(path, sheet.size, flush=True)
    print(json.dumps({"population": counts}), flush=True)


def probe_items(n):
    """n lab canvases spread over the sources (realplus 0.4, teacher 0.3, owner 0.2, devnative 0.1), drawn by key
    hash: the probe's canvases."""
    want = {"realplus": round(0.4 * n), "teacher": round(0.3 * n), "owner": round(0.2 * n)}
    want["devnative"] = n - sum(want.values())
    out = {}
    for source, k in want.items():
        by = {r["shot"]: r for r in record_rows(source)[0]}
        pool = []
        for name, npy, kept in lab_items(source)[0]:
            for i, smp in kept:
                key = lab_key(name, i)
                pool.append((hashlib.sha256(f"textspot2-probe:{key}".encode()).hexdigest(), name, npy, i, smp))
        pool.sort(key=lambda t: t[0])
        shards = collections.defaultdict(list)
        for _, name, npy, i, smp in pool[:k]:
            shards[name].append({"key": lab_key(name, i), "npy": npy, "sample": smp, "record": by.get(smp["shot"])})
        out[source] = sorted(shards.items())
    return out


def probe_analysis(lines_by_source):
    """Counts that bear on the rules' open readings, over the probe's cached canvases."""
    a = collections.Counter()
    for source, items in lines_by_source.items():
        for smp, ln, furi in items:
            ms = to_meiki(ln)
            ex = [[int(v) for v in x[:4]] for x in smp["lines"]]
            for m in ms:
                a["boxes"] += 1
                hi = m["score"] >= RULES["pos_min"]
                a["high" if hi else "low"] += 1
                p1, v1 = refined_of(m, "P1")
                p2, v2 = refined_of(m, "P2")
                a["vertical_P1"] += v1
                a["vertical_P2"] += v2
                a["P1_P2_flag_differs"] += v1 != v2
                a["P1_P2_box_differs"] += p1 != p2
                a["raw_tall"] += tall(m["raw"])
                a["ref_h_unchanged"] += m["ref_h"] == m["raw"]
                a["ref_t_unchanged"] += m["ref_t"] == m["raw"]
                if v1 and not tall(m["ref_t"]):
                    a["P1_vertical_but_transposed_not_tall"] += 1
                if hi:
                    cr, cw = covered(p1, ex), covered(m["raw"], ex)
                    a["high_covered_refined"] += cr
                    a["high_covered_raw"] += cw
                    a["high_cover_refined_vs_raw_differs"] += cr != cw
                elif covered(p1, ex):
                    a["low_covered_by_existing"] += 1
            lines, ignore, st = build_labels(smp["w"], smp["h"], smp["lines"], smp["ignore"], furi, ms,
                                             winner_xyxy(ln))
            newp = [x for x in lines[len(smp["lines"]):]]
            for i in range(len(newp)):
                for j in range(i + 1, len(newp)):
                    bi, bj = newp[i][:4], newp[j][:4]
                    inter = union_cover(bi, [bj]) * area(bi)
                    if inter >= 0.5 * min(area(bi), area(bj)):
                        a["new_pairs_overlapping_half_of_smaller"] += 1
            for p in newp:
                for e in ex:
                    inter = union_cover(p[:4], [e]) * area(p[:4])
                    if inter >= 0.25 * min(area(p[:4]), area(e)) and inter > 0:
                        a["new_overlapping_existing_quarter"] += 1
                        break
            a["canvases"] += 1
            for k, v in st["counts"].items():
                a[f"build_{k}"] += v
    return dict(sorted(a.items()))


def cmd_probe(args):
    """The timing probe: n lab canvases through colour, meiki, refine and the winner (W workers in this slot), then
    the build in memory; cache under $T2/labels/probe ($P2 for the owner's). Nothing of it is used."""
    print("cores", pin_to_slot(), flush=True)
    items = probe_items(args.n)
    t0 = time.time()
    recs = {}
    for source, shards in items.items():
        cdir = base_of(source) / "labels" / "probe" / source
        if cdir.exists():
            for p in cdir.glob("*.jsonl"):
                p.unlink()
        recs[source] = run_cache("lab", source, args.workers, cdir=cdir, shards=shards)
    t_cache = time.time() - t0
    t1 = time.time()
    by_source, ms = {}, collections.Counter()
    for source, shards in items.items():
        cdir = base_of(source) / "labels" / "probe" / source
        out = []
        for name, its in shards:
            cache = {ln["key"]: ln for ln in jsonl(cdir / f"{name}.jsonl")}
            for it in its:
                ln = cache[it["key"]]
                smp, r = it["sample"], it["record"]
                furi = record_furi(r, ln["k"], smp["w"], smp["h"]) if ln.get("k") is not None else []
                out.append((smp, ln, furi))
                for k, v in (ln.get("ms") or {}).items():
                    ms[k] += v
                ms["n"] += 1
                ms[f"colour_{ln['status']}"] += 1
        by_source[source] = out
    tb = time.time()
    n_built = 0
    for source, out in by_source.items():
        for smp, ln, furi in out:
            build_labels(smp["w"], smp["h"], smp["lines"], smp["ignore"], furi, to_meiki(ln), winner_xyxy(ln))
            n_built += 1
    t_build = (time.time() - tb) / max(1, n_built)
    analysis = probe_analysis(by_source)
    n = ms.pop("n")
    per = {k: round(v / n, 2) for k, v in ms.items() if not k.startswith("colour_")}
    res = {"made": utc(), "canvases": n, "workers": args.workers, "wall_cache_s": round(t_cache, 1),
           "wall_per_canvas_ms": round(1000 * t_cache / n, 1), "cpu_ms_per_canvas": per,
           "build_ms_per_canvas": round(1000 * t_build, 2), "colour": {k[7:]: v for k, v in ms.items()
                                                                        if k.startswith("colour_")},
           "by_source": {s: r["canvases"] for s, r in recs.items()}, "analysis": analysis,
           "build_and_read_s": round(time.time() - t1, 1)}
    jdump(res, T2 / "labels" / "probe" / "probe.json", indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print("TEXTSPOT2_PROBE_DONE", flush=True)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    workers = int(os.environ.get("OMP_NUM_THREADS", "5"))
    p = sub.add_parser("cache")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--workers", type=int, default=workers)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard-range", default="", metavar="A:B",
                   help="only the source's shards A..B-1 (python slice), so one source can run in several slots")
    p = sub.add_parser("build")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--source", required=True, choices=SOURCES)
    sub.add_parser("pool")
    p = sub.add_parser("reserve")
    p.add_argument("--pixels", action="store_true")
    sub.add_parser("leak")
    p = sub.add_parser("manifest")
    p.add_argument("--out", default=str(OUT_MANIFEST))
    p = sub.add_parser("sheet")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--sheet", type=int, required=True)
    p.add_argument("--sources", default="realplus,teacher,devnative")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--out", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--workers", type=int, default=workers)
    p = sub.add_parser("qa")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    if args.cmd == "cache":
        print("cores", pin_to_slot(), flush=True)
        shards = cache_shards(args.arm, args.source, args.limit)
        if args.shard_range:
            a, b = (int(v) if v else None for v in args.shard_range.split(":"))
            shards = shards[a:b]
        rec = run_cache(args.arm, args.source, args.workers, args.limit, shards=shards)
        print(json.dumps(rec), flush=True)
        print(f"TEXTSPOT2_CACHE_DONE {args.arm}/{args.source}", flush=True)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "pool":
        pool, rec = more_pool()
        write_pool(pool, rec)
        print(json.dumps(rec, indent=1), flush=True)
        print("TEXTSPOT2_POOL_DONE", flush=True)
    elif args.cmd == "reserve":
        sys.exit(cmd_reserve(args))
    elif args.cmd == "leak":
        sys.exit(cmd_leak(args))
    elif args.cmd == "manifest":
        cmd_manifest(args)
    elif args.cmd == "sheet":
        cmd_sheet(args)
    elif args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd == "qa":
        cmd_qa(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-character screen boxes from the CTC frames: calibration (training side) and the development score.

docs/char-boxes.md, eval/charboxes_protocol.json, issue #114. The positions are eval/charbox.py's (the twin of
runtime/vocr_charbox.c); the recognizers are the released VOCR5 blobs through the C runtime in W8I
(runtime/pyvocr.py, build/libvocr.so of this checkout).

  calibrate   training-side lines only (no dev_sample, test or CLANNAD VN): teacher_train's crops with
              MeikiOCR's character boxes (VOCR5-S/N/L), and vertical training columns read by MeikiOCR's
              vertical recognizer (VOCR5-V). For each model: shift = the median, over the characters of
              group 0, of MeikiOCR's box centre minus the frame centre (resized px), and per group the median
              of the same difference minus the shift (groups with fewer than CAL_MIN characters: 0), on lines
              the model reads exactly as MeikiOCR does (VOCR5-V: the equal characters of an alignment, on
              columns MeikiOCR reads within charbox_sets.MAX_GT_CER_V of VOCR5-V; its vertical recognizer
              rarely reads a whole column exactly). Writes eval/charboxes_cal.json.
  train-check the arms on training-side lines the calibration did not read (the evidence the method of record
              was chosen on, before any development line was read).
  score       the development units (eval/charbox_sets.py units + meiki) through every line source
              (fixed: the unit's ink box plus the console pad; finder: GRAD + refine on the unit's region, the
              column finder for columns; ts1 / ts3: textspot's and textspot3's released spotters on the whole
              frame, det+refine), every model and every arm; one record per aligned reference character.
  report      the protocol's tables from the score records, with VN-cluster bootstrap intervals.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "runtime"))
import charbox as CB  # noqa: E402
import charbox_sets as CS  # noqa: E402
from metrics import norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
MODELS = ART / "overlay-boxes" / "models"
BLOBS = {"S": MODELS / "v030" / "VOCR5_S_w8.vocr", "N": MODELS / "v030" / "VOCR5_N_w8.vocr",
         "L": MODELS / "v030" / "VOCR5_L_w8.vocr", "V": MODELS / "v032v" / "VOCR5_V_w8.vocr"}
SPOTTERS = {"ts1": MODELS / "v033ts" / "textspot_s4_c24n4_w8.vtsp",
            "ts3": MODELS / "v034ts3" / "textspot3_t64n8c16_kd_w8.vtsp"}
CAL_FILE = ROOT / "eval" / "charboxes_cal.json"
CAL_MIN = 50
CAL_LINES = 3000
CAL_SEED = 20261023
ARMS = {"raw": (0, False), "cal": (0, True), "ink": (CB.INK, False), "cal+ink": (CB.INK, True),
        "cal+pitch+ink": (CB.INK | CB.PITCH, True)}
RECORD_ARM = "cal+ink"


def sha256(p):
    return CS.sha256(p)


def to_gray(rgb):
    from common import to_gray as tg
    return tg(rgb)


def line_dark(grey, box):
    from common import line_dark as ld
    return int(ld(grey, tuple(int(v) for v in box)))


class Reader:
    """One VOCR5 model in W8I: (text, best frames, resized width) of a grey crop."""

    def __init__(self, tier):
        from pyvocr import Vocr
        self.tier, self.path = tier, BLOBS[tier]
        self.v = Vocr(str(self.path), mode="w8i")
        self.groups = CB.class_groups(self.v)
        self.sha = sha256(self.path)

    def read(self, grey, invert=False):
        g = np.ascontiguousarray(grey, np.uint8)
        text = self.v.run(g, invert=invert)
        return text, self.v.last_best.copy(), self.v.resized_width(g.shape[1], g.shape[0])


def class_keys(reader, boxes):
    return [norm(reader.v.class_text(o["cls"])) for o in boxes]


# ----------------------------------------------------------------------------- calibration

def cal_from(diffs):
    """[shift, group 0 .. N_GROUPS - 1] (resized px x 16) from {group: [MeikiOCR centre - frame centre]}."""
    shift = float(np.median(diffs[CB.G_OTHER])) if diffs[CB.G_OTHER] else 0.0
    out = [int(round(shift * CB.Q))]
    for g in range(CB.N_GROUPS):
        v = diffs.get(g, [])
        out.append(int(round((float(np.median(v)) - shift) * CB.Q)) if len(v) >= CAL_MIN and g != CB.G_OTHER else 0)
    return out


def excluded_vns():
    return CS.dev_vns() | CS.test_vns()


def cal_horizontal(reader, log=print):
    z = np.load(ART / "teacher" / "teacher_train.npz")
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    meta = json.loads(str(z["meta"]))
    excl = excluded_vns()
    perm = np.random.default_rng(CAL_SEED).permutation(len(meta))
    diffs, used, chars = defaultdict(list), 0, 0
    for i in perm:
        m = meta[i]
        if m["vn"] in excl:
            continue
        if used >= CAL_LINES:
            break
        used += 1
        h, w = (int(v) for v in shapes[i])
        g = data[offs[i]:offs[i + 1]].reshape(h, w)       # stored in console polarity
        text, best, nw = reader.read(g)
        mc = m["chars"]
        sp = CB.spans(best)
        if text != "".join(c[0] for c in mc) or len(sp) != len(mc):
            continue
        for (cls, f0, _f1), c in zip(sp, mc):
            fire = CB.STRIDE * f0 + CB.STRIDE / 2
            diffs[reader.groups[cls]].append((c[1] + c[3]) / 2 * nw / w - fire)
            chars += 1
    log(f"{reader.tier}: {used} training lines, {chars} characters")
    return diffs, {"lines_read": used, "characters": chars}


def cal_vertical(reader, log=print):
    sys.path.insert(0, str(ROOT))
    import vertical_twin as V
    from teacher.engines import Meiki
    meiki = Meiki(n_threads=int(os.environ.get("OMP_NUM_THREADS", "5")))
    pages = CS.jsonl(ART / "vertical" / "data" / "reads_train" / "reads.jsonl")
    excl = excluded_vns()
    rng = np.random.default_rng(CAL_SEED)
    cols = []
    for p in pages:
        if p.get("split") != "train" or p["vn"] in excl or not p.get("path"):
            continue
        for c in p["columns"]:
            r = c.get("reads", {})
            if r.get("meiki") and r.get("meiki") == r.get("v6s"):
                cols.append((p, c))
    order = rng.permutation(len(cols))[:CAL_LINES]
    diffs, chars, used = defaultdict(list), 0, 0
    for k in order:
        p, c = cols[k]
        rgb = CS.read_rgb(p["path"])
        if p.get("dims") and [rgb.shape[1], rgb.shape[0]] != list(p["dims"]):
            import cv2  # vertical_data.load_page: fit the console's screen
            W, H = p["dims"]
            meth = cv2.INTER_AREA if W < rgb.shape[1] else cv2.INTER_LINEAR
            rgb = np.ascontiguousarray(cv2.resize(rgb, (int(W), int(H)), interpolation=meth))
        grey = to_gray(rgb)
        x0, y0, x1, y1 = c["ink"]
        kind = "psp" if p.get("kind") == "psp" else "vita"
        u = {"box": [x0, y0, x1 - x0, y1 - y0], "pad": CS.PAD[kind], "vertical": True}
        mtext, mchars = CS.meiki_read(meiki, rgb, u)
        bx = CS.clip_box([x0 - u["pad"], y0 - u["pad"], x1 - x0 + 2 * u["pad"], y1 - y0 + 2 * u["pad"]],
                         (grey.shape[1], grey.shape[0]))
        if bx[2] < 2 or bx[3] < 2:
            continue
        crop = grey[bx[1]:bx[1] + bx[3], bx[0]:bx[0] + bx[2]]
        dark = line_dark(grey, bx)
        strip = V.rotate_ccw(255 - crop if dark else crop)
        text, best, nw = reader.read(strip)
        used += 1
        sp = CB.spans(best)
        mk = [norm(m[0]) for m in mchars]
        vk = [norm(reader.v.class_text(s[0])) for s in sp]
        pairs, dist = CS.align(vk, mk)
        if not mk or dist > CS.MAX_GT_CER_V * max(1, len(mk)):
            continue
        w = strip.shape[1]
        for i, j in pairs:
            if i is None or j is None or not vk[i] or vk[i] != mk[j]:
                continue
            cls, f0, _f1 = sp[i]
            m = mchars[j]
            mc = (m[2] + m[4]) / 2 - bx[1]                  # along the column = along the strip
            fire = CB.STRIDE * f0 + CB.STRIDE / 2
            diffs[reader.groups[cls]].append(mc * nw / w - fire)
            chars += 1
    log(f"V: {used} training columns, {chars} characters")
    return diffs, {"columns_read": used, "characters": chars}


def cmd_calibrate(args):
    out = {"kind": "charboxes-calibration", "schema": 1,
           "what": "per model [shift, offset of group 0..7] in resized px x 16 (eval/charbox.py positions(cal=)); "
                   "groups: " + ", ".join(f"{k} {n}" for k, n in enumerate(CB.GROUP_NAMES)),
           "rule": __doc__.split("calibrate")[1].split("train-check")[0].strip(),
           "seed": CAL_SEED, "lines": CAL_LINES, "min_per_group": CAL_MIN, "models": {}}
    if args.update and Path(args.out).exists():
        out["models"] = json.loads(Path(args.out).read_text(encoding="utf-8"))["models"]
    for tier in args.tiers:
        r = Reader(tier)
        diffs, info = (cal_vertical if tier == "V" else cal_horizontal)(r)
        cal = cal_from(diffs)
        out["models"][tier] = {"blob": r.path.name, "sha256": r.sha, "cal": cal, **info,
                               "per_group": {CB.GROUP_NAMES[g]: len(v) for g, v in sorted(diffs.items())}}
        print(tier, cal, flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


# ----------------------------------------------------------------------------- scoring

def load_cal(path=CAL_FILE):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v["cal"] for k, v in doc["models"].items()}


def crop_box(u, dims):
    x, y, w, h = u["box"]
    p = u["pad"]
    return CS.clip_box([x - p, y - p, w + 2 * p, h + 2 * p], dims)


def est_boxes(reader, grey, box, vertical, cal, dark=None):
    """Read the crop `box` (x, y, w, h) of the frame's grey and place its characters: (text, {arm: [(key,
    centre (x, y), cell (a, b), ink (a, b))]}) in frame px; a, b along the reading direction."""
    x, y, w, h = box
    crop = np.ascontiguousarray(grey[y:y + h, x:x + w])
    if dark is None:
        dark = line_dark(grey, box)
    if vertical:
        import vertical_twin as V
        strip = V.rotate_ccw(255 - crop if dark else crop)
        text, best, nw = reader.read(strip)
        sw, sh = strip.shape[1], strip.shape[0]
        prof = CB.ink_profile(strip)
    else:
        text, best, nw = reader.read(crop, invert=bool(dark))
        sw, sh = w, h
        prof = CB.ink_profile(crop)
    out = {}
    for arm, (flags, use_cal) in ARMS.items():
        bx = CB.positions(best, sw, sh, nw, flags, prof, cal if use_cal else None, reader.groups)
        keys = class_keys(reader, bx)
        rows = []
        for o, k in zip(bx, keys):
            c, a, b, i0, i1 = (o[f] / CB.Q for f in ("c", "x0", "x1", "i0", "i1"))
            if not vertical:
                rows.append((k, (x + c, y + h / 2), (x + a, x + b), (x + i0, x + i1)))
            else:
                rows.append((k, (x + w / 2, y + c), (y + a, y + b), (y + i0, y + i1)))
        out[arm] = rows
    return text, out


def iou1(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    uni = max(a[1], b[1]) - min(a[0], b[0])
    return inter / uni if uni > 0 else 0.0


def score_line(u, gt, rows, cross):
    """Records of one unit against one read: per reference character with a reference box, aligned to an
    estimated character (None: the read has no character there)."""
    ref = CS.char_keys(u["text"])
    est = [r[0] for r in rows]
    pairs, _ = CS.align(ref, est)
    out = []
    vert = u["vertical"]
    for i, j in pairs:
        if i is None or gt[i] is None:
            continue
        b = gt[i]
        g_along = (b[1], b[3]) if vert else (b[0], b[2])
        if j is None:
            out.append({"i": i, "est": None})
            continue
        k, c, cell, ink = rows[j]
        ca, cc = (c[1], c[0]) if vert else (c[0], c[1])
        hit = g_along[0] <= ca <= g_along[1] and cross[0] <= cc <= cross[1]
        out.append({"i": i, "est": k, "eq": k == ref[i], "hit": bool(hit), "cell_iou": round(iou1(cell, g_along), 4),
                    "ink_iou": round(iou1(ink, g_along), 4), "err": round(ca - (g_along[0] + g_along[1]) / 2, 2),
                    "gw": g_along[1] - g_along[0]})
    return out


def cross_range(u):
    x, y, w, h = u["box"]
    p = u["pad"]
    return (x - p, x + w + p) if u["vertical"] else (y - p, y + h + p)


def match_boxes(dets, refs, vertical):
    """One-to-one matching of detections (x, y, w, h) to reference boxes, eval's rule: overlap across the reading
    direction at least half the smaller extent, any overlap along it; the largest overlaps first."""
    cand = []
    for i, d in enumerate(dets):
        for j, r in enumerate(refs):
            if vertical:
                ov = min(d[0] + d[2], r[0] + r[2]) - max(d[0], r[0])
                al = min(d[1] + d[3], r[1] + r[3]) - max(d[1], r[1])
                sm = min(d[2], r[2])
            else:
                ov = min(d[1] + d[3], r[1] + r[3]) - max(d[1], r[1])
                al = min(d[0] + d[2], r[0] + r[2]) - max(d[0], r[0])
                sm = min(d[3], r[3])
            if ov >= 0.5 * sm and al > 0:
                cand.append((ov * max(al, 0), i, j))
    cand.sort(reverse=True)
    used_d, used_r, out = set(), set(), {}
    for _, i, j in cand:
        if i in used_d or j in used_r:
            continue
        used_d.add(i)
        used_r.add(j)
        out[j] = i
    return out


def finder_boxes(grey, region, vertical):
    """The console finder's boxes on a region (x, y, w, h) of the frame, in frame px, with their dark flags."""
    x, y, w, h = region
    reg = np.ascontiguousarray(grey[y:y + h, x:x + w])
    if vertical:
        import vertical_twin as V
        b, d, _ = V.find_columns(reg, 1)
    else:
        import lines_refine as LR
        from common import find_lines_ex
        r = find_lines_ex(reg, "grad")
        if r is None:
            return []
        b, d, _ = r
        b, d, _ = LR.refine_lines(reg, b, d, LR.DEFAULT)
    return [((bx[0] + x, bx[1] + y, bx[2], bx[3]), int(dk)) for bx, dk in zip(b, d)]


_SPOT = {}


def spot_boxes(key, grey):
    """A released spotter's det+refine boxes on the whole frame: [((x, y, w, h), vertical, dark)]."""
    import lines_refine as LR
    import textspot3_twin as T3
    if key not in _SPOT:
        _SPOT[key] = T3.Model(Path(SPOTTERS[key]).read_bytes())
    r = T3.spot(_SPOT[key], grey)
    boxes = [(b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark) for b in r.boxes]
    hz = [b for b in boxes if not b[4]]
    rb, rd, _ = LR.refine_lines(grey, [b[:4] for b in hz], [b[6] for b in hz], LR.DEFAULT)
    out = [((int(b[0]), int(b[1]), int(b[2]), int(b[3])), False, int(d)) for b, d in zip(rb, rd)]
    out += [((b[0], b[1], b[2], b[3]), True, int(b[6])) for b in boxes if b[4]]
    return out


def load_gt(units, meiki_path, hand_path=None):
    """{uid: [box or None] per reference character}: charbox_sets.reference_boxes of MeikiOCR's stored reading;
    with hand_path (the hand check, charbox_hand.py), only the characters whose box the check accepted."""
    gts = {}
    for r in CS.jsonl(meiki_path):
        u = units.get(r["uid"])
        if u is None:
            continue
        boxes, _ = CS.reference_boxes(u, r["meiki_text"], r["meiki_chars"])
        gts[r["uid"]] = boxes
    if hand_path:
        keep = {}
        for h in CS.jsonl(hand_path):
            if h["uid"] in gts and h.get("verdict") == "ok":
                b = list(gts[h["uid"]])
                for i in h.get("bad", []):
                    if 0 <= i < len(b):
                        b[i] = None
                keep[h["uid"]] = b
        gts = keep
    return gts


def cmd_score(args):
    cal = load_cal(args.cal)
    units = {u["uid"]: u for u in CS.jsonl(args.units)}
    gts = load_gt(units, args.meiki, args.hand)
    tiers = args.tiers
    readers = {t: Reader(t) for t in tiers + (["V"] if "V" not in tiers else [])}
    by_frame = defaultdict(list)
    for uid, u in units.items():
        if uid in gts and any(b is not None for b in gts[uid]):
            by_frame[(u["frame"]["id"], json.dumps(u["frame"].get("scale"), sort_keys=True))].append(u)
    sources = args.sources
    out = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    n = 0
    for fk, us in sorted(by_frame.items()):
        rgb = CS.frame_rgb(us[0]["frame"])
        grey = to_gray(rgb)
        H, W = grey.shape

        def emit(u, src, tier, text, ests, rd_box):
            gt = gts[u["uid"]]
            for arm, rows in ests.items():
                recs = score_line(u, gt, rows, cross_range(u))
                out.write(json.dumps({"uid": u["uid"], "vn": u["vn"], "part": u["part"], "cls": u["cls"],
                                      "vertical": u["vertical"], "cohort": u.get("cohort"), "src": u["src"],
                                      "lines": src, "tier": tier, "arm": arm, "read": text,
                                      "box": rd_box, "chars": recs}, ensure_ascii=False) + "\n")

        if "fixed" in sources:
            for u in us:
                cb = crop_box(u, (W, H))
                for tier in (["V"] if u["vertical"] else [t for t in tiers if t != "V"]):
                    text, ests = est_boxes(readers[tier], grey, cb, u["vertical"], cal[tier])
                    emit(u, "fixed", tier, text, ests, cb)
        if "finder" in sources:
            regs = defaultdict(list)
            for u in us:
                if u.get("region"):
                    regs[(tuple(u["region"]), u["vertical"])].append(u)
            for (reg, vert), rus in regs.items():
                dets = finder_boxes(grey, reg, vert)
                m = match_boxes([d[0] for d in dets], [u["box"] for u in rus], vert)
                for j, u in enumerate(rus):
                    if j not in m:
                        emit(u, "finder", "-", None, {a: [] for a in ARMS}, None)
                        continue
                    box, dark = dets[m[j]]
                    for tier in (["V"] if vert else [t for t in tiers if t != "V"]):
                        text, ests = est_boxes(readers[tier], grey, box, vert, cal[tier], dark)
                        emit(u, "finder", tier, text, ests, list(box))
        for key in [s for s in sources if s in SPOTTERS]:
            if us[0]["src"] != "textspot-dev":
                continue
            dets = spot_boxes(key, grey)
            for vert in (False, True):
                dv = [d for d in dets if d[1] == vert]
                vu = [u for u in us if u["vertical"] == vert]
                if not vu:
                    continue
                m = match_boxes([d[0] for d in dv], [u["box"] for u in vu], vert)
                for j, u in enumerate(vu):
                    if j not in m:
                        emit(u, key, "-", None, {a: [] for a in ARMS}, None)
                        continue
                    box, _, dark = dv[m[j]]
                    for tier in (["V"] if vert else [t for t in tiers if t != "V"]):
                        text, ests = est_boxes(readers[tier], grey, box, vert, cal[tier], dark)
                        emit(u, key, tier, text, ests, list(box))
        n += 1
        if n % 20 == 0:
            out.flush()
            print(f"{n}/{len(by_frame)} frames, {time.time() - t0:.0f}s", flush=True)
    out.close()
    print("SCORE_DONE", flush=True)


# ----------------------------------------------------------------------------- report

def boot_ratio(groups, seed=0, n=10000):
    """95% VN-cluster bootstrap interval of sum(hits) / sum(chars) over {vn: (hits, chars)}."""
    keys = sorted(groups)
    if not keys:
        return None
    h = np.array([groups[k][0] for k in keys], np.float64)
    c = np.array([groups[k][1] for k in keys], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    hs, cs = h[idx].sum(1), c[idx].sum(1)
    r = hs / np.maximum(cs, 1)
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def summarise(recs, key_fn, boot=False):
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    stats = defaultdict(lambda: defaultdict(float))
    for r in recs:
        k = key_fn(r)
        if k is None:
            continue
        for c in r["chars"]:
            s = stats[k]
            s["ref"] += 1
            if c["est"] is None:
                continue
            s["pairs"] += 1
            s["hit"] += c["hit"]
            s["eq"] += c["eq"]
            s["cell_iou"] += c["cell_iou"]
            s["ink_iou"] += c["ink_iou"]
            agg[k][r["vn"]][0] += c["hit"]
            agg[k][r["vn"]][1] += 1
    out = {}
    for k, s in stats.items():
        p = max(1, s["pairs"])
        out[k] = {"ref_chars": int(s["ref"]), "pairs": int(s["pairs"]), "hit": round(s["hit"] / p, 4),
                  "hit_all": round(s["hit"] / max(1, s["ref"]), 4), "cell_iou": round(s["cell_iou"] / p, 4),
                  "ink_iou": round(s["ink_iou"] / p, 4), "vns": len(agg[k])}
        if boot:
            out[k]["hit_ci"] = boot_ratio(agg[k])
    return out


def cmd_report(args):
    recs = CS.jsonl(args.scores)
    tabs = {}
    tabs["by_part"] = summarise(recs, lambda r: (r["lines"], r["tier"], r["arm"], r["cls"], r["part"],
                                                 "v" if r["vertical"] else "h"), boot=args.boot)
    doc = {"kind": "charboxes-report", "scores": str(args.scores), "tables": {"|".join(map(str, k)): v
                                                                             for k, v in tabs["by_part"].items()}}
    if args.units:
        # the record arm's dialogue characters by calibration group: where the misses are
        units = {u["uid"]: u for u in CS.jsonl(args.units)}
        grp = defaultdict(lambda: [0, 0])
        for r in recs:
            if r["arm"] != RECORD_ARM or r["cls"] != "dialogue" or r["vertical"]:
                continue
            text = units[r["uid"]]["text"]
            for c in r["chars"]:
                if c["est"] is None:
                    continue
                g = CB.GROUP_NAMES[CB.group(text[c["i"]])]
                k = "|".join((r["lines"], r["tier"], r["part"], g))
                grp[k][0] += c["hit"]
                grp[k][1] += 1
        doc["by_group"] = {k: {"hit": round(v[0] / max(1, v[1]), 4), "pairs": v[1]} for k, v in sorted(grp.items())}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, v in sorted(tabs["by_part"].items(), key=str):
        if k[2] in (RECORD_ARM, "raw") and k[3] == "dialogue":
            print(k, v)


def cmd_train_check(args):
    """The arms on training-side lines the calibration did not read (the next TRAIN_CHECK lines of its seeded
    order): the evidence the method of record was chosen on, before any development line was read."""
    cal = load_cal(args.cal)
    z = np.load(ART / "teacher" / "teacher_train.npz")
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    meta = json.loads(str(z["meta"]))
    excl = excluded_vns()
    order = [i for i in np.random.default_rng(CAL_SEED).permutation(len(meta)) if meta[i]["vn"] not in excl]
    pick = order[CAL_LINES:CAL_LINES + args.lines]
    out = {"kind": "charboxes-train-check", "lines": len(pick), "models": {}}
    for tier in args.tiers:
        r = Reader(tier)
        st = defaultdict(lambda: [0, 0])
        iou = defaultdict(list)
        exact = 0
        for i in pick:
            m = meta[i]
            h, w = (int(v) for v in shapes[i])
            g = data[offs[i]:offs[i + 1]].reshape(h, w)
            text, best, nw = r.read(g)
            mc = m["chars"]
            if text != "".join(c[0] for c in mc):
                continue
            exact += 1
            prof = CB.ink_profile(g)
            for arm, (flags, use_cal) in ARMS.items():
                bx = CB.positions(best, w, h, nw, flags, prof, cal[tier] if use_cal else None, r.groups)
                if len(bx) != len(mc):
                    continue
                for o, c in zip(bx, mc):
                    hit = c[1] <= o["c"] / CB.Q <= c[3]
                    gname = CB.GROUP_NAMES[r.groups[o["cls"]]]
                    for key in (f"{arm}|{m['target']}", f"{arm}|group:{gname}", f"{arm}|all"):
                        st[key][0] += hit
                        st[key][1] += 1
                    iou[f"{arm}|all"].append(iou1((o["x0"] / CB.Q, o["x1"] / CB.Q), (c[1], c[3])))
        out["models"][tier] = {"exact_lines": exact,
                               "hit": {k: [round(v[0] / max(1, v[1]), 4), v[1]] for k, v in sorted(st.items())},
                               "cell_iou": {k: round(float(np.mean(v)), 4) for k, v in sorted(iou.items())}}
        print(tier, exact, {k: v for k, v in out["models"][tier]["hit"].items() if k.endswith(("|vita", "|psp"))},
              flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("calibrate")
    a.add_argument("--tiers", nargs="+", default=["S", "N", "L", "V"])
    a.add_argument("--out", default=str(CAL_FILE))
    a.add_argument("--update", action="store_true", help="keep the other models of an existing --out")
    a = sub.add_parser("train-check")
    a.add_argument("--tiers", nargs="+", default=["S", "N", "L"])
    a.add_argument("--cal", default=str(CAL_FILE))
    a.add_argument("--lines", type=int, default=1500)
    a.add_argument("--out", required=True)
    a = sub.add_parser("score")
    a.add_argument("--units", required=True)
    a.add_argument("--meiki", required=True)
    a.add_argument("--hand", help="the hand check's verdicts (charbox_hand.py): score its accepted boxes only")
    a.add_argument("--cal", default=str(CAL_FILE))
    a.add_argument("--tiers", nargs="+", default=["S", "N", "L"])
    a.add_argument("--sources", nargs="+", default=["fixed", "finder", "ts1", "ts3"])
    a.add_argument("--out", required=True)
    a = sub.add_parser("report")
    a.add_argument("--scores", required=True)
    a.add_argument("--boot", action="store_true")
    a.add_argument("--units", help="the units the scores read: adds the record arm's table by group")
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"calibrate": cmd_calibrate, "train-check": cmd_train_check, "score": cmd_score,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

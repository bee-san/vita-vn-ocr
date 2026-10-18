"""translucent-box: dialogue text over textured backgrounds (issue #98, docs/translucent-box.md).

  translucent_box.py tag --out DIR [--source NAME ...] [--procs N]
      The background classifier below on every source: the development whole-dialogue regions (VOCR5's wd
      cohorts, base variant), eval/linefind's four detector-built sets, the training pools (realplus, teacher's
      training lines, devnative's frames outside the development sample) and the owner's frames. Private
      per-region rows (DIR/rows_SOURCE.jsonl: ids, boxes and statistics, no text, no pixels) and a summary.
  translucent_box.py summary --tags DIR --baseline DIR --out JSON
      The public prevalence record: per source and per VN the regions and the tagged ones, and the share of
      GRAD and GRAD + refine's development dialogue errors (VOCR5-S/N/L, #97's frozen baseline units) that sits
      in tagged regions.

The classifier (frozen 2026-10-18, before any candidate finder was scored; docs/translucent-box.md "Prevalence"):
a text group's region (its lines' ink boxes, eval's grouping rule, plus 12 px on Vita and 6 px on PSP) is grey
g at the console's scale. Its background sample is the box's own background next to the text: over the text's
columns, outside every line's ink box grown by 3 px (Vita) / 2 px (PSP), the rows between the lines and the rows
above the first and below the last line, scanned outward from the text until a row looks like the box's edge
(half or more of its sampled pixels are "on") or its median grey moves more than 24 levels from the first
scanned row's. A pixel is "on" when its 3x3 local contrast (max - min, the GRAD finder's c) reaches the region's
stroke level, GRAD's own threshold T = max(40, 2/5 of the 99th percentile of c over the region). busy = the share
of the sample that is on; a region with a sample of at least 48 pixels and busy >= 0.03 is tagged: structure in
the box background next to the text that is as contrasty as the text's strokes (a translucent box over a scene,
text without a box over a scene, or a textured box). Checked by eye on contact sheets (docs).
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
OWNER = HERD / "shared" / "private" / "owner-games" / "_build" / "owner-pool"
WD_COHORTS = ART / "vocr5-train" / "cohorts"
MARGIN = {"vita": 12, "psp": 6}
DIL = {"vita": 3, "psp": 2}
EDGE = 0.5          # a scanned row with this share of on-pixels or more is the box's edge: the scan stops
TONE = 24           # ... or one whose median grey is this far from the first scanned row's
N_MIN = 48          # the smallest background sample a region is classified on
TAG_MIN = 0.03      # busy at or above this: tagged
SOURCES = ("wd", "linefind", "realplus", "teacher", "devnative", "owner")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- the classifier

def contrast3(g):
    """max - min of each 3x3 window, edges replicated (the GRAD finder's c, synth/common.contrast3)."""
    from common import contrast3 as c3
    return c3(np.asarray(g, dtype=np.uint8))


def stroke_level(con):
    """GRAD's threshold: max(40, 2/5 of the 99th percentile of the region's contrast)."""
    cum = np.cumsum(np.bincount(con.ravel(), minlength=256)).astype(np.int64)
    p99 = int(np.argmax(cum * 100 >= con.size * 99))
    return max(40, 2 * p99 // 5)


def background_sample(g, boxes, kind):
    """(sample mask, on mask, T) of a region g (grey, HxW) whose text lines have ink boxes [(x0, y0, x1, y1)]
    in region coordinates. See the module docstring."""
    g = np.asarray(g, dtype=np.uint8)
    hh, ww = g.shape
    dl = DIL[kind]
    text = np.zeros((hh, ww), bool)
    for (a, b, c, d) in boxes:
        text[max(0, b - dl):max(0, d + dl), max(0, a - dl):max(0, c + dl)] = True
    cx0 = max(1, min(b[0] for b in boxes))
    cx1 = min(ww - 1, max(b[2] for b in boxes))
    cy0 = max(0, min(b[1] for b in boxes) - dl)
    cy1 = min(hh, max(b[3] for b in boxes) + dl)
    con = contrast3(g)
    T = stroke_level(con)
    on = con >= T
    bg = ~text
    bg[:, :cx0] = False
    bg[:, cx1:] = False
    bg[0, :] = False
    bg[-1, :] = False
    nb = bg.sum(1)
    frac = np.where(nb > 0, (bg & on).sum(1) / np.maximum(nb, 1), 0.0)
    keep = np.zeros(hh, bool)
    for rng in (range(cy0 - 1, 0, -1), range(cy1, hh - 1)):
        ref = None
        for y in rng:
            if not nb[y] or frac[y] >= EDGE:
                break
            med = float(np.median(g[y][bg[y]]))
            if ref is None:
                ref = med
            elif abs(med - ref) > TONE:
                break
            keep[y] = True
    inner = np.zeros(hh, bool)
    inner[cy0:cy1] = True
    keep |= inner & (frac < EDGE)
    return bg & keep[:, None], on, T


def busyness(g, boxes, kind):
    """{"busy", "n", "T", "tagged"} of one text group's region."""
    if not boxes:
        return {"busy": None, "n": 0, "T": None, "tagged": False}
    m, on, T = background_sample(g, boxes, kind)
    n = int(m.sum())
    busy = round(float((m & on).sum() / n), 4) if n >= N_MIN else 0.0
    return {"busy": busy, "n": n, "T": int(T), "tagged": bool(n >= N_MIN and busy >= TAG_MIN)}


# ---------------------------------------------------------------- regions of a frame

def frame_regions(grey, members, others, kind):
    """[(region [x0, y0, x1, y1] in the frame, member boxes in region coordinates)] of the text groups of one
    frame, as eval/finder_tighten.py cuts its whole-dialogue regions: the members (the frame's dialogue lines'
    ink boxes) grouped by eval's rule, each group's union plus MARGIN[kind], shrunk on the side facing any other
    text box (a speaker name, an unlabelled line, UI: finder_tighten.clip_region); a group another box overlaps
    is left out."""
    import finder_tighten as FT
    H, W = grey.shape
    out = []
    for grp in FT.groups_of(members):
        mem = [members[i] for i in grp]
        core = FT.union(mem)
        m = MARGIN[kind]
        reg = [max(0, core[0] - m), max(0, core[1] - m), min(W, core[2] + m), min(H, core[3] + m)]
        reg = FT.clip_region(reg, core, others)
        if reg is None:
            continue
        x0, y0 = reg[:2]
        out.append((reg, [(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in mem]))
    return out


def classify_frame(grey, members, others, kind):
    """Rows (one per text group) of a frame: region, lines and the classifier's statistics."""
    rows = []
    for reg, rel in frame_regions(grey, members, others, kind):
        x0, y0, x1, y1 = reg
        rows.append({"region": reg, "lines": len(rel), **busyness(grey[y0:y1, x0:x1], rel, kind)})
    return rows


def split_reads(s, keys, scale, W, H):
    """(members, others) of a reads frame: the lines whose k is in keys, and every other detected text box (the
    other lines and the detector boxes not mostly inside a member), at the target scale, clipped to the frame
    (eval/finder_tighten._frame_units)."""
    import finder_tighten as FT
    members, others = [], []
    boxes = FT._read_boxes(s, scale)
    for what, k, box in boxes:
        box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if what == "line" and k in keys:
            members.append(box)
        elif what == "line":
            others.append(box)
    for what, k, box in boxes:
        if what != "det":
            continue
        box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(FT.inter_area(box, m) >= 0.5 * max(1, (box[2] - box[0]) * (box[3] - box[1])) for m in members):
            continue
        others.append(box)
    return members, others


# ---------------------------------------------------------------- sources

def _teacher_reads(shots=None):
    reads = {}
    for p in sorted(glob.glob(str(ART / "teacher" / "reads" / "train.*of5.jsonl"))):
        for x in open(p, encoding="utf-8"):
            try:
                r = json.loads(x)
            except json.JSONDecodeError:
                continue
            if shots is None or r["shot"] in shots:
                reads[r["shot"]] = r
    return reads


def source_wd():
    """VOCR5's development whole-dialogue regions (base variant): the cohort's own region (clipped against the
    frame's other text boxes) and its member lines' ink boxes from the frame's reads."""
    import pickle

    import geometry as G
    dn_reads = {json.loads(x)["shot"]: json.loads(x)
                for x in open(ART / "devnative" / "reads" / "reads.jsonl", encoding="utf-8")}
    tv_reads = None
    for cn in ("tv_dlg", "dn_dlg"):
        with open(WD_COHORTS / f"{cn}.pkl", "rb") as f:
            cases = pickle.load(f)
        if cn == "tv_dlg":
            tv_reads = _teacher_reads({c["shot"] for c in cases})
        for c in cases:
            if c["variant"] != "base":
                continue
            r = (tv_reads if cn == "tv_dlg" else dn_reads)[c["shot"]]
            sc = 1.0
            if cn == "tv_dlg" and r["kind"] == "hires":
                sc = G.target_scale(r["dims"][0], r["dims"][1], c["kind"])
            by_k = {str(ln["k"]): ln for ln in r.get("lines") or []}
            x0, y0 = c["region"][:2]
            rel = []
            for ln in c["lines"]:
                b = by_k[ln["id"].rsplit("_", 1)[1]]
                b = G.crop_box(b.get("crop_ink") or b["ink"], sc, 0, 1 << 30, 1 << 30)
                rel.append((b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0))
            yield {"source": "wd", "cohort": cn, "unit": c["unit"], "shot": c["shot"], "vn": c["vn"],
                   "kind": c["kind"], "region": c["region"], "lines": len(rel),
                   **busyness(c["image"], rel, c["kind"])}


def linefind_members(keep):
    """eval/linefind holds no text, so a speaker name cannot be told by its text: the top line of a group of two
    or more lines that is at most 40% as wide as the group's widest line is taken for a name (another text box,
    as the wd cohorts treat names)."""
    import finder_tighten as FT
    members, others = [], []
    for grp in FT.groups_of(keep):
        mem = sorted((keep[i] for i in grp), key=lambda b: (b[1], b[0]))
        widest = max(b[2] - b[0] for b in mem)
        if len(mem) >= 2 and 10 * (mem[0][2] - mem[0][0]) <= 4 * widest:
            others.append(mem[0])
            mem = mem[1:]
        members += mem
    return members, others


def source_linefind():
    """eval/linefind's detector-built sets: the region's non-ignore line boxes (names taken out, linefind_members)
    grouped, each group's union plus the margin, clipped to the set's region (the region a player drew) and
    against the names and ignore boxes. Regions without a line are `empty`."""
    import textdet as TD
    BG = TD._linefind("build_gt")
    shots = [str(d) for d in BG.SHOT_DIRS]
    test_vns = TD.reserved_test_vns()
    for name in TD.LINEFIND_SETS:
        rows, _ = TD.linefind_rows(name, test_vns)          # regions of a reserved test VN left out (#97's rule)
        for r in rows:
            g = TD.region_grey(r, shots)
            kind = "vita" if r["platform"] == "psv" else "psp"
            keep, ign = TD.gt_of(r)
            base = {"source": "linefind", "set": name, "id": r["id"], "shot": r["shot"], "vn": r["vn"],
                    "kind": kind, "tags": r.get("tags", [])}
            if not keep:
                yield {**base, "region": r["region"], "lines": 0, "busy": None, "n": 0, "T": None,
                       "tagged": False, "empty": True}
                continue
            members, names = linefind_members(keep)
            x0, y0 = r["region"][:2]
            rows = classify_frame(g, members, names + list(ign), kind)
            if not rows:
                yield {**base, "region": r["region"], "lines": len(keep), "busy": None, "n": 0, "T": None,
                       "tagged": False, "unclassified": True}
            for row in rows:
                row["region"] = [row["region"][0] + x0, row["region"][1] + y0, row["region"][2] + x0,
                                 row["region"][3] + y0]
                yield {**base, **row}


def _pool_frame(job):
    """One frame of a pool: (source, shot, vn, kind, loader, reads row, member line keys)."""
    import geometry as G

    from common import to_gray
    from devnative import paths as P
    src, shot, vn, kind, loader, s, keys = job
    if loader[0] == "path":
        rgb = P.read_rgb(loader[1])
        if rgb is None:
            return []
        sc = 1.0
        if (rgb.shape[1], rgb.shape[0]) != P.DIMS[kind]:
            sc = G.target_scale(rgb.shape[1], rgb.shape[0], kind)
            rgb = G.scale_image(rgb, sc)
    else:
        rgb = P.load_screen(s)
        if rgb is None:
            return []
        sc = 1.0                  # devnative and owner reads are native frames (eval/finder_tighten.py's builders)
    grey = to_gray(np.ascontiguousarray(rgb))
    H, W = grey.shape
    if list(s.get("dims") or []) == [W, H]:
        sc = 1.0                  # realplus reads hold their boxes at the frame's console scale already
    members, others = split_reads(s, keys, sc, W, H)
    if not members:
        return []
    return [{"source": src, "shot": shot, "vn": vn, "kind": kind, **row}
            for row in classify_frame(grey, members, others, kind)]


def jobs_realplus():
    """realplus's kept, non-name lines (labels.jsonl) on their reads frames; PC shots at the Vita scale."""
    kept = {}
    for x in open(ART / "realplus" / "labels.jsonl", encoding="utf-8"):
        r = json.loads(x)
        if r["status"] == "kept" and not r.get("name"):
            kept.setdefault(r["shot"], set()).add(r["id"].rsplit("_", 1)[1])
    for x in open(ART / "realplus" / "reads" / "reads.jsonl", encoding="utf-8"):
        s = json.loads(x)
        ks = kept.get(s["shot"])
        if not ks:
            continue
        kind = s["kind"] if s["kind"] in MARGIN else "vita"
        yield ("realplus", s["shot"], s["vn"], kind, ("path", s["path"]), s, ks)


def jobs_teacher():
    """teacher's training lines (teacher_train.npz ids and targets, name-like labels left out) on their reads
    frames, at each target."""
    import finder_tighten as FT
    rows = FT.load_npz(ART / "teacher" / "teacher_train.npz")
    by = {}
    for m, lab, _ in rows:
        shot, k = m["id"].rsplit("_", 1)
        if not FT.namelike(lab):
            by.setdefault((shot, m["target"]), set()).add(k)
    reads = _teacher_reads({s for s, _ in by})
    for (shot, target), ks in sorted(by.items()):
        s = reads.get(shot)
        if s is None:
            continue
        path = ART / "realdata" / "shots" / s["kind"] / f"{s['vn']}_{s['url'].rsplit('/', 1)[-1]}"
        yield ("teacher", shot, s["vn"], target, ("path", str(path)), s, ks)


def jobs_devnative():
    """devnative's labelled non-name lines on frames whose VN is outside the development sample (training data
    since the 13:25Z policy), on their reads frames."""
    from common import dev_sample_vns
    dev = set(dev_sample_vns())
    reads = {json.loads(x)["shot"]: json.loads(x)
             for x in open(ART / "devnative" / "reads" / "reads.jsonl", encoding="utf-8")}
    by = {}
    for x in open(ART / "devnative" / "lines_full.jsonl", encoding="utf-8"):
        r = json.loads(x)
        if r["vn"] in dev or "name" in (r.get("styles") or []):
            continue
        by.setdefault(r["shot"], set()).add(r["id"].rsplit("_", 1)[1])
    for shot, ks in sorted(by.items()):
        s = reads[shot]
        yield ("devnative", shot, s["vn"], s["kind"], ("screen",), s, ks)


def jobs_owner():
    """The owner's frames (owner pool v2, private): its kept, non-name lines on their reads frames."""
    kept = {}
    for x in open(OWNER / "labels.jsonl", encoding="utf-8"):
        r = json.loads(x)
        if r["status"] == "kept" and not r.get("name"):
            kept.setdefault(r["shot"], set()).add(r["id"].rsplit("_", 1)[1])
    for x in open(OWNER / "reads" / "reads.jsonl", encoding="utf-8"):
        s = json.loads(x)
        ks = kept.get(s["shot"])
        if not ks:
            continue
        yield ("owner", s["shot"], s["vn"], s["kind"], ("screen",), s, ks)


POOL_JOBS = {"realplus": jobs_realplus, "teacher": jobs_teacher, "devnative": jobs_devnative, "owner": jobs_owner}


def pool_jobs(src):
    """A pool's frames, every VN that eval/manifests/reserved.json reserves (test or development sample) left
    out: the pools are training data."""
    from common import is_reserved_vn
    return [j for j in POOL_JOBS[src]() if not is_reserved_vn(j[2])]


# ---------------------------------------------------------------- the design cohorts (training data)

def _design_frame(job):
    """Whole-dialogue cases of one training frame (eval/finder_tighten.dialogue_units: base, +4, +8), each tagged
    by the classifier on its base region."""
    import finder_tighten as FT
    import geometry as G

    from common import to_gray
    from devnative import paths as P
    src, shot, vn, kind, path, s, labels = job
    rgb = P.read_rgb(path)
    if rgb is None:
        return []
    sc = 1.0
    if (rgb.shape[1], rgb.shape[0]) != P.DIMS[kind]:
        sc = G.target_scale(rgb.shape[1], rgb.shape[0], kind)
        rgb = G.scale_image(rgb, sc)
    grey = to_gray(np.ascontiguousarray(rgb))
    H, W = grey.shape
    if list(s.get("dims") or []) == [W, H]:
        sc = 1.0
    members, others = [], []
    boxes = FT._read_boxes(s, sc)
    for what, k, box in boxes:
        box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if what == "line" and k in labels:
            members.append((box, f"{shot}_{k}", labels[k]))
        elif what == "line":
            others.append(box)
    for what, k, box in boxes:
        if what != "det":
            continue
        box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(FT.inter_area(box, m[0]) >= 0.5 * max(1, (box[2] - box[0]) * (box[3] - box[1])) for m in members):
            continue
        others.append(box)
    if not members:
        return []
    cases = FT.dialogue_units(shot, kind, W, H, members, others, grey, f"design_{src}", vn)
    box_of = {m[1]: m[0] for m in members}

    def rel_ink(c):
        x0, y0 = c["region"][:2]
        return [[box_of[ln["id"]][0] - x0, box_of[ln["id"]][1] - y0, box_of[ln["id"]][2] - x0,
                 box_of[ln["id"]][3] - y0] for ln in c["lines"]]

    tag = {c["unit"]: busyness(c["image"], rel_ink(c), kind) for c in cases if c["variant"] == "base"}
    out = []
    for c in cases:
        b = tag.get(c["unit"])
        if b is None:
            continue
        c.update(busy=b["busy"], tagged=b["tagged"], src=src, ink=rel_ink(c))
        out.append(c)
    return out


def design_jobs(exclude_shots):
    """Training frames of realplus (kept lines) and teacher (teacher_train lines), name-like labels (eval's rule:
    6 or fewer characters and no punctuation, which also takes out most UI words) left out,
    every reserved VN and every excluded screenshot left out: (src, shot, vn, kind, path, reads row, {k: label})."""
    import finder_tighten as FT

    from common import is_reserved_vn
    jobs = []
    kept = {}
    for x in open(ART / "realplus" / "labels.jsonl", encoding="utf-8"):
        r = json.loads(x)
        if r["status"] == "kept" and not r.get("name") and r.get("label") and not FT.namelike(r["label"]):
            kept.setdefault(r["shot"], {})[r["id"].rsplit("_", 1)[1]] = r["label"]
    for x in open(ART / "realplus" / "reads" / "reads.jsonl", encoding="utf-8"):
        s = json.loads(x)
        lab = kept.get(s["shot"])
        if not lab or is_reserved_vn(s["vn"]) or s["shot"] in exclude_shots:
            continue
        jobs.append(("realplus", s["shot"], s["vn"], s["kind"] if s["kind"] in MARGIN else "vita", s["path"], s, lab))
    by = {}
    for m, lab, _ in FT.load_npz(ART / "teacher" / "teacher_train.npz"):
        shot, k = m["id"].rsplit("_", 1)
        if not FT.namelike(lab):
            by.setdefault((shot, m["target"]), {})[k] = lab
    reads = _teacher_reads({s for s, _ in by})
    for (shot, target), lab in sorted(by.items()):
        s = reads.get(shot)
        if s is None or is_reserved_vn(s["vn"]) or shot in exclude_shots:
            continue
        path = ART / "realdata" / "shots" / s["kind"] / f"{s['vn']}_{s['url'].rsplit('/', 1)[-1]}"
        jobs.append(("teacher", shot, s["vn"], target, str(path), s, lab))
    return jobs


def cmd_design(args):
    """The design cohorts: whole-dialogue cases of training frames (never development data), the tagged regions
    and an equal-sized sample of the untagged ones, at most args.per_vn regions per VN and set."""
    import pickle
    import random

    import textdet as TD
    exclude = set()
    for cn in ("tv_dlg", "dn_dlg"):
        with open(WD_COHORTS / f"{cn}.pkl", "rb") as f:
            exclude |= {c["shot"] for c in pickle.load(f)}
    BG = TD._linefind("build_gt")
    for name in ("vita_dev", "vita_holdout", "psp_dev", "psp_holdout", "research_a", "research_b"):
        exclude |= {r["shot"].rsplit(".", 1)[0] for r in BG.read(TD.LINEFIND / f"{name}.jsonl")}
    jobs = [j for j in design_jobs(exclude) if not args.kinds or j[3] in args.kinds]
    rng = random.Random(args.seed)
    rng.shuffle(jobs)
    if args.frames:
        jobs = jobs[:args.frames]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.procs) as pool:
        cases = [c for cs in pool.imap(_design_frame, jobs, chunksize=8) for c in cs]
    units = {}
    for c in cases:
        units.setdefault((c["src"], c["unit"]), []).append(c)
    keys = sorted(units)
    rng.shuffle(keys)
    pick = {"tagged": [], "rest": []}
    per_vn = {"tagged": {}, "rest": {}}
    for k in keys:
        base = next(c for c in units[k] if c["variant"] == "base")
        part = "tagged" if base["tagged"] else "rest"
        n = per_vn[part].get(base["vn"], 0)
        if n >= args.per_vn:
            continue
        per_vn[part][base["vn"]] = n + 1
        pick[part].append(k)
    n = min(len(pick["tagged"]), args.regions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"kind": "translucent-box-design", "computed": utc_now(), "seed": args.seed, "frames": len(jobs),
               "excluded_shots": len(exclude), "per_vn": args.per_vn, "sets": {}}
    for part, ks in (("tagged", pick["tagged"][:n]), ("rest", pick["rest"][:n])):
        sel = [c for k in sorted(ks) for c in units[k]]
        with open(out / f"{args.name}_{part}.pkl", "wb") as f:
            pickle.dump(sel, f, protocol=4)
        summary["sets"][part] = {"regions": len(ks), "cases": len(sel), "vns": len({c["vn"] for c in sel}),
                                 "by_src": {s: sum(1 for k in ks if k[0] == s) for s in ("realplus", "teacher")},
                                 "by_kind": {kd: sum(1 for c in sel if c["kind"] == kd and c["variant"] == "base")
                                             for kd in MARGIN},
                                 "sha256": sha256(out / f"{args.name}_{part}.pkl")}
        print(part, summary["sets"][part], flush=True)
    summary["kinds"] = args.kinds
    (out / f"{args.name}.json").write_text(json.dumps(summary, indent=1) + "\n")


# ---------------------------------------------------------------- scoring through the C path

MODES = {"grad": (0, 0), "refine": (0, 31), "split": (1, 0), "split_refine": (1, 31)}
SPLIT_SOURCES = ("runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.h",
                 "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h", "runtime/vocr_lines_split.c",
                 "runtime/vocr_lines_split.h", "runtime/vocr_lines_split_test.c")
DRIVER_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror")


def build_driver(out, compiler="gcc"):
    """The finder + split + refinement driver (runtime/vocr_lines_split_test.c), built now from this checkout."""
    import subprocess
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    drv = out / "vocr_lines_split_test"
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
    subprocess.run([compiler, *DRIVER_FLAGS, "-o", str(drv), "runtime/vocr_lines.c", "runtime/vocr_lines_refine.c",
                    "runtime/vocr_lines_split.c", "runtime/vocr_lines_split_test.c"], check=True, cwd=ROOT, env=env)
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return {"driver": str(drv), "driver_sha256": sha256(drv), "compiler": version, "flags": list(DRIVER_FLAGS)}


def c_lines(driver, images, sflags, rflags):
    """The C finder (GRAD), the split (sflags) and the refinement (rflags) on every grey image:
    [(boxes, dark, found, rows_read)]."""
    import struct
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img in images:
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape
            f.write(struct.pack("<7I", w, h, 1, 0, w, sflags, rflags))
            f.write(img.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(driver), path], check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(out) != len(images):
        raise RuntimeError(f"driver returned {len(out)} results for {len(images)} images")
    res = []
    for line in out:
        v = [int(t) for t in line.split()]
        if v[0] != 0:
            raise RuntimeError(f"finder/split/refinement rc={v[0]}")
        n = v[3]
        res.append(([tuple(v[6 + 5 * i:10 + 5 * i]) for i in range(n)], [v[10 + 5 * i] for i in range(n)], v[1],
                    v[5]))
    return res


def twin_lines(img, sflags, rflags):
    import lines_refine as LR
    import lines_split as LS
    from common import find_lines_ex
    b, d, _ = find_lines_ex(img, "grad")
    sb, sd, _ = LS.split_lines(img, b, d, sflags)
    rb, rd, _ = LR.refine_lines(img, sb, sd, rflags)
    return [tuple(map(int, x)) for x in rb], [int(x) for x in rd]


def mode_lines(driver, cases, mode):
    """The C lines of every case in one mode, checked against the twins box for box."""
    sf, rf = MODES[mode]
    res = c_lines(driver, [c["image"] for c in cases], sf, rf)
    for i, (c, (b, d, _, _)) in enumerate(zip(cases, res)):
        tb, td = twin_lines(c["image"], sf, rf)
        if b != tb or d != td:
            raise RuntimeError(f"{mode} case {i} ({c['unit']}|{c['variant']}): C and twin differ: {b} {d} / {tb} {td}")
    return res


def cmd_score(args):
    """Every arm on every cohort in the four modes through the C path (C == twin checked on every case)."""
    import pickle

    import finder_tighten as FT

    from general_vn_textbox_guard import transcript_score
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ident = build_driver(out / "build")
    lib = FT.build(out / "build-lib")
    meta = {"kind": "translucent-box-scores", "computed": utc_now(), "driver": ident,
            "library": {k: lib[k] for k in ("library_sha256", "compiler", "scalar_flags")},
            "sources_sha256": {f: sha256(ROOT / f) for f in SPLIT_SOURCES},
            "code_sha256": {f: sha256(ROOT / f) for f in ("eval/translucent_box.py", "synth/lines_split.py",
                                                         "synth/lines_refine.py", "synth/common.py")},
            "modes": {m: {"split_flags": s, "refine_flags": r} for m, (s, r) in MODES.items()},
            "cohorts": {}, "arms": {}}
    cases, lines = {}, {}
    for name in args.names:
        with open(Path(args.cohorts) / f"{name}.pkl", "rb") as f:
            cases[name] = pickle.load(f)
        lines[name] = {m: mode_lines(ident["driver"], cases[name], m) for m in MODES}
        meta["cohorts"][name] = {"sha256": sha256(Path(args.cohorts) / f"{name}.pkl"), "cases": len(cases[name])}
        print(f"{name}: {len(cases[name])} cases, C == twin in every mode", flush=True)
    for spec in args.arm:
        arm, _, blob = spec.partition("=")
        reader = FT.Reader(Path(blob), lib["library"])
        meta["arms"][arm] = {"blob": Path(blob).name, "sha256": reader.sha256, "blob_bytes": reader.blob_bytes,
                             "arena_w8i": reader.arena_bytes}
        cache = {}

        def read(g, dark):
            key = (hashlib.blake2b(np.ascontiguousarray(g).tobytes(), digest_size=16).hexdigest(), g.shape,
                   int(dark))
            if key not in cache:
                cache[key] = reader.read(g, dark)
            return cache[key]

        res = {}
        for name in args.names:
            res[name] = {}
            for mode in MODES:
                units = []
                for c, (boxes, dark, found, rows) in zip(cases[name], lines[name][mode]):
                    img = c["image"]
                    preds = [read(img[y:y + h, x:x + w], d) for (x, y, w, h), d in zip(boxes, dark)]
                    sc = transcript_score(c["lines"], preds)
                    units.append({"unit": f"{c['unit']}|{c['variant']}", "shot": c["shot"], "vn": c["vn"],
                                  "platform": c["kind"], "variant": c["variant"],
                                  "errors": sc["all"]["full"]["errors"], "chars": sc["all"]["full"]["chars"],
                                  "hira_errors": sc["all"]["hira"]["errors"],
                                  "hira_chars": sc["all"]["hira"]["chars"], "detections": len(boxes),
                                  "finder_lines": found, "rows_read": rows})
                res[name][mode] = units
        (out / f"units_{arm}.json").write_text(json.dumps(res) + "\n")
        print(f"{arm}: scored, {len(cache)} crops read", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


# ---------------------------------------------------------------- reports and the rule

PAIRS = (("split_refine", "refine"), ("split", "grad"), ("refine", "grad"), ("split_refine", "grad"))


def load_scores(directory, arm):
    return json.loads((Path(directory) / f"units_{arm}.json").read_text(encoding="utf-8"))


def table(units_by_mode, keep=lambda u: True):
    """CER per mode and the paired VN-cluster differences of PAIRS over the units keep selects."""
    import textdet as TD
    out = {m: TD.cer_of(u, keep) for m, u in units_by_mode.items()}
    out["units"] = sum(1 for u in next(iter(units_by_mode.values())) if keep(u))
    out["vns"] = len({u["vn"] for u in next(iter(units_by_mode.values())) if keep(u)})
    for a, b in PAIRS:
        out[f"{a}-{b}"] = TD.vn_diff([(units_by_mode[a], units_by_mode[b])], keep)
    return out


def cmd_design_report(args):
    """The design cohorts' record (training data): every model, mode, set and platform."""
    doc = {"kind": "translucent-box-design-record", "computed": utc_now(), "code_sha256": sha256(Path(__file__)),
           "scores": {}, "sets": {}}
    for spec in args.scores:
        arm, _, d = spec.partition("=")
        res = load_scores(d, arm)
        meta = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
        doc["scores"][arm] = {"meta_sha256": sha256(Path(d) / "meta.json"), "arm": meta["arms"][arm],
                              "sources_sha256": meta["sources_sha256"], "cohorts": meta["cohorts"]}
        for name, by_mode in res.items():
            for plat in ("all", "vita", "psp"):
                t = table(by_mode, (lambda u, p=plat: p == "all" or u["platform"] == p))
                if t["units"]:
                    doc["sets"].setdefault(name, {}).setdefault(arm, {})[plat] = t
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for name, e in doc["sets"].items():
        for arm, t in e.items():
            a = t["all"]
            print(f"{name:18s} {arm}: " + " ".join(f"{m} {a[m]['cer']}" for m in MODES) +
                  f" | split+ref-ref {a['split_refine-refine']['delta']} {a['split_refine-refine']['ci95']}"
                  f" | split-grad {a['split-grad']['delta']} {a['split-grad']['ci95']}")


# ---------------------------------------------------------------- the protocol (pre-registration)

PROTOCOL = ROOT / "eval" / "translucent_box_protocol.json"
UNCHANGED = ("runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.c", "runtime/vocr_lines_rows.h",
             "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h", "synth/common.py", "synth/lines_refine.py",
             "runtime/vocr.c", "runtime/vocr.h", "train/general_vn_textbox_guard.py", "eval/finder_tighten.py",
             "eval/textdet.py", "eval/manifests/reserved.json")
SOURCES = ("runtime/vocr_lines_split.c", "runtime/vocr_lines_split.h", "runtime/vocr_lines_split_test.c",
           "synth/lines_split.py", "tests/lines_split_parity.py")


def wd_cohort_counts(tags):
    """The development cohorts' sizes (VOCR5's wd pooling): units, base regions, VNs and characters, tagged and
    the rest, and the native-Vita subset."""
    import textdet as TD
    index = wd_tag_index(tags)
    res = json.loads((ART / "textdet" / "baseline" / "wd" / "VOCR5-N" / "units_VOCR5-N.json").read_text())
    units = TD.wd_units(res, "grad")
    native = TD.native_vita_shots()
    out = {}
    for name, keep in (("tagged", lambda u: unit_tagged(index, u)), ("rest", lambda u: not unit_tagged(index, u)),
                       ("native_vita", lambda u: u["shot"] in native)):
        sel = [u for u in units if keep(u)]
        out[name] = {"units": len(sel), "regions": len({u["unit"].rsplit("|", 1)[0] for u in sel}),
                     "vns": len({u["vn"] for u in sel}), "chars": sum(u["chars"] for u in sel),
                     "by_platform": {p: sum(1 for u in sel if u["platform"] == p) for p in ("vita", "psp")}}
    out["native_vita_tagged_units"] = sum(1 for u in units if u["shot"] in native and unit_tagged(index, u))
    return out


def cmd_protocol(args):
    import textdet as TD
    ft = json.loads((ROOT / "eval" / "finder_tighten_protocol.json").read_text())["refinement"]["source_sha256"]
    tdp = json.loads((ROOT / "eval" / "textdet_protocol.json").read_text())
    unchanged = {f: sha256(ROOT / f) for f in UNCHANGED}
    for f, d in unchanged.items():
        if not f.startswith(("runtime/", "synth/")):
            continue                     # harness files: their current hashes are recorded, not compared
        if f in ft and ft[f] != d:
            raise SystemExit(f"{f} differs from finder-tighten's frozen copy")
        if f in tdp["sources_sha256"] and tdp["sources_sha256"][f] != d:
            raise SystemExit(f"{f} differs from textdet's frozen copy")
    import lines_split as LS
    base = ART / "textdet" / "baseline" / "wd"
    design = {Path(d).stem: {k: {kk: vv for kk, vv in v.items() if kk != "sha256"} for k, v in
                             json.loads(Path(d).read_text())["sets"].items()} for d in args.design}
    doc = {
        "kind": "translucent-box-protocol", "issue": "https://github.com/bee-san/vita-vn-ocr/issues/98", "schema": 1,
        "frozen": utc_now(),
        "rule_of_this_file": "Written and pushed before any candidate is scored on development data. Nothing here "
                             "changes after that except through a dated amendment appended to 'amendments', made "
                             "before the scores it can affect.",
        "owner_steering": "2026-10-18 06:50Z: do all four workstreams, PRs and issues, auto-merge, issues left open; "
                          "'we need it to run on a ps vita, so smaller == better'",
        "sources_sha256": {f: sha256(ROOT / f) for f in SOURCES},
        "harness_sha256_at_freeze": {f: sha256(ROOT / f) for f in ("eval/translucent_box.py",)},
        "harness_note": "eval/translucent_box.py may gain report and protected-set commands after the freeze; the "
                        "classifier, the cohorts (the tag rows' sha256 below) and the score path do not change",
        "unchanged_sha256": unchanged,
        "recognizers": tdp["recognizers"],
        "candidate": {
            "name": "split", "split_flags": LS.DEFAULT, "flag_names": "VOCR_SPLIT_STROKE",
            "api": ["vocr_split_lines(im, flags, arena, bytes, set, rows_read)",
                    "vocr_split_lines_rows(src, flags, arena, bytes, set, rows_read)"],
            "where": "after the GRAD finder (runtime/vocr_lines.c, unchanged), before the refinement "
                     "(runtime/vocr_lines_refine.c, unchanged); reads each box's own pixels only",
            "off": "flags 0 returns the finder's set unchanged and reads nothing (bit-identical; tested)",
            "arena": "vocr_split_arena_bytes(w, h) = 3 + up4(3w) + 3 up4(2h) <= vocr_lines_arena_bytes(w, h) "
                     "(tested on a grid of sizes)",
            "method": "synth/lines_split.py's docstring (the twin is the specification): per GRAD box of at least "
                      "16 rows, bright and dark ridges along the rows (reach 3 px), S(y) = min(#bright, #dark) above "
                      "2/5 of their 99th percentiles (floor 24), valley segmentation (S[y] * 3 <= the smaller "
                      "neighbouring peak), bands (S * 4 >= the segment's peak), text bands (>= 70% of the tallest, "
                      ">= 10% of the heaviest mass, >= 6 rows), a continuity guard on the horizontal 3-pixel "
                      "contrast (the gap's smallest count <= 60% of the bands' upper median), lines = band +- 16%, "
                      "the box's columns and dark flag",
            "constants": {k: getattr(LS, k) for k in dir(LS) if k.startswith("S_")},
            "arms": {m: {"split_flags": s, "refine_flags": r} for m, (s, r) in MODES.items()},
            "design": "chosen on training data only (eval/translucent_box.py design: realplus and teacher training "
                      "frames, every reserved VN, every wd frame and every linefind screenshot left out; "
                      "the sets below); the record is eval/translucent_box_design_20261018.json",
            "design_sets": design,
        },
        "development": {
            "wd": "VOCR5's development whole dialogue, unchanged (eval/textdet.wd_units: tv_dlg plus the dn_dlg frames "
                  "tv_dlg does not hold, base / +4 / +8 px; VOCR5's cohorts, sha256 below); every crop a mode "
                  "returns read in order by the C runtime in W8I, joined, scored by transcript_score (dialogue text)",
            "cohorts_sha256": {c: sha256(WD_COHORTS / f"{c}.pkl") for c in ("tv_dlg", "dn_dlg")},
            "translucent_box_cohort": "the wd units whose base region the frozen classifier tags (eval/translucent_box"
                                      ".py busyness >= 0.03, sample >= 48 px; the tags as committed in "
                                      "eval/translucent_box_prevalence_20261018.json and the private rows file below)",
            "rest_of_dev": "every other wd unit",
            "native_vita": "#97's native-Vita subset (eval/textdet.native_vita_shots: teacher_val frames read as kind "
                           "'vita' and devnative's Vita frames)",
            "tags_rows_sha256": sha256(Path(args.tags) / "rows_wd.jsonl"),
            "counts": wd_cohort_counts(args.tags),
            "baseline": "#97's frozen baseline units (GRAD flags 0 / refine 31; $HERD/shared/artifacts/textdet/"
                        "baseline/wd): this run's grad and refine units must equal them unit for unit",
            "baseline_units_sha256": {m: sha256(base / m / f"units_{m}.json") for m in MODELS},
            "bootstrap": {**TD.BOOT, "method": "eval/textdet.vn_diff: VN clusters resampled with numpy "
                                               "default_rng(0).integers, 2.5 / 97.5 percentiles, variants pooled"},
        },
        "rule": {
            "adopted_iff": [
                "A. translucent-box cohort: split_refine - refine has a VN-cluster 95% interval entirely below 0, for "
                "each of VOCR5-S, -N and -L",
                "B. rest of dev: split_refine - refine has no interval entirely above 0, for each model",
                "C. native Vita: neither split_refine - refine nor split - grad has an interval entirely above 0, "
                "for any model",
                "D. the C driver equals the twins on every development case in all four modes",
                "E. cost: VitaSDK Thumb-2 text of vocr_lines_split.o at -Os <= 4,096 B; deepest stack frame "
                "(-fstack-usage) <= 4,096 B; the arena bound above; an estimated <= 25 ms per 900x150 box at 333 MHz "
                "on one Cortex-A9 core (the A9 cycle model, calibrated as #97 to VOCR3-H15's 5.0-5.2 ms per CTC "
                "frame)",
            ],
            "rubi_default": "if adopted: Rubi runs the split in every game (its own setting, default on) when, "
                            "for each model, also F. tagged: split - grad has an interval entirely below 0 and "
                            "G. rest of dev: split - grad has no interval entirely above 0 (Rubi's Vita games run "
                            "no refinement by default); otherwise the split runs only where the refinement does "
                            "(Rubi's ocr_refine_lines, PSP games by default)",
            "not_adopted": "the verdict is negative; a fallback (#97's detector gated on boxes the classifier "
                           "tags, or the textgate model) may be tried only through a dated amendment made before any "
                           "of its scores; the protected sets are read once for the candidate, labelled not adopted",
            "report_only": ["per platform and per variant", "linefind detection (vita_dev, psp_dev; holdouts)",
                            "#97's det+refine (s8_c24_seed4, its stored units) on the same cohorts",
                            "the owner's frames (own_dlg; seen during design as a no-false-split check)",
                            "the design record"],
        },
        "protected": {
            "when": "once, after the development record (scores, rule outcome) is committed and pushed",
            "sets": ["vita-test + psp-test whole dialogue: eval's regions (a region with an uncertain line left out "
                     "whole), base / +4 / +8, the four modes, each VOCR5 model; body text and all text; VN-cluster "
                     "intervals (eval/finder_tighten's protected procedure with the split's modes)",
                     "CLANNAD long run: owner-drive-ingest's frozen references (refs_sha256 35411688...), each "
                     "page's ROI, the C finder; a reference line reads every line derived from its GRAD box (the "
                     "split, then the refinement), joined in order (eval/finder_tighten's clannad procedure); "
                     "dialogue (no name style) and all lines; whole-page bootstraps"],
            "attempts": "attempt records created exclusively; a second run refuses",
            "role": "report only: they decide nothing",
        },
        "disclosures": [
            "The failure modes were diagnosed on development regions before this freeze, by viewing (GRAD and GRAD + "
            "refine boxes of v27449's four frames and of the 24 highest-error base regions, and the classifier's "
            "ranked samples); the classifier's statistic and threshold were chosen while viewing development "
            "regions. No candidate split was run on a development case before this freeze.",
            "During design the prototypes were also run on the owner's frames (finder-tighten's own_dlg cohort, "
            "training data) as a check for false splits on native console text; it is report only here.",
        ],
        "estimates": "every console time is a host estimate (vita/a9_cycles.py's model under qemu-arm, as #83 and "
                     "#97), labelled as such; nothing runs on a console (hardware work is paused by the owner)",
        "amendments": [],
    }
    PROTOCOL.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {PROTOCOL} ({doc['frozen']})")
    print(json.dumps(doc["development"]["counts"], indent=1))


def cmd_tag(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "translucent-box-tags", "computed": utc_now(), "code_sha256": sha256(Path(__file__)),
            "classifier": {"margin": MARGIN, "dilation": DIL, "edge": EDGE, "tone": TONE, "n_min": N_MIN,
                           "tag_min": TAG_MIN}, "sources": {}}
    for src in args.source or SOURCES:
        if src == "wd":
            rows = list(source_wd())
        elif src == "linefind":
            rows = list(source_linefind())
        else:
            jobs = pool_jobs(src)
            if args.limit:
                jobs = jobs[:args.limit]
            if args.procs > 1:
                import multiprocessing as mp
                with mp.get_context("fork").Pool(args.procs) as pool:
                    rows = [r for rs in pool.imap(_pool_frame, jobs, chunksize=16) for r in rs]
            else:
                rows = [r for j in jobs for r in _pool_frame(j)]
        with open(out / f"rows_{src}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = sum(1 for r in rows if r.get("busy") is not None)
        t = sum(1 for r in rows if r["tagged"])
        meta["sources"][src] = {"regions": len(rows), "classified": n, "tagged": t,
                                "rows_sha256": sha256(out / f"rows_{src}.jsonl")}
        print(f"{src}: {len(rows)} regions, {n} classified, {t} tagged", flush=True)
    prev = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    prev.setdefault("sources", {}).update(meta["sources"])
    meta["sources"] = prev["sources"]
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


# ---------------------------------------------------------------- the public summary

MODELS = ("VOCR5-S", "VOCR5-N", "VOCR5-L")


def load_rows(tags, src):
    return [json.loads(x) for x in open(Path(tags) / f"rows_{src}.jsonl", encoding="utf-8")]


def wd_tag_index(tags):
    """{"tv_dlg|<unit>": tagged} of the wd regions (a unit's three variants share its base region's tag)."""
    return {f"{r['cohort']}|{r['unit']}": r["tagged"] for r in load_rows(tags, "wd")}


def unit_tagged(index, u):
    """The tag of a wd unit (eval/textdet.wd_units format: '<cohort>|<unit>|<variant>')."""
    cohort, unit, _ = u["unit"].split("|")
    return index[f"{cohort}|{unit}"]


def per_vn(rows):
    out = {}
    for r in rows:
        if r.get("busy") is None:
            continue
        e = out.setdefault(r["vn"], [0, 0])
        e[0] += 1
        e[1] += int(r["tagged"])
    return out


def cmd_summary(args):
    import textdet as TD
    doc = {"kind": "translucent-box-prevalence", "computed": utc_now(), "code_sha256": sha256(Path(__file__)),
           "classifier": {"margin": MARGIN, "dilation": DIL, "edge": EDGE, "tone": TONE, "n_min": N_MIN,
                          "tag_min": TAG_MIN},
           "tags_meta_sha256": sha256(Path(args.tags) / "meta.json"), "sources": {}}
    for src in SOURCES:
        p = Path(args.tags) / f"rows_{src}.jsonl"
        if not p.exists():
            continue
        rows = load_rows(args.tags, src)
        if src == "wd":
            tv_shots = {r["shot"] for r in rows if r["cohort"] == "tv_dlg"}
            rows = [r for r in rows if not (r["cohort"] == "dn_dlg" and r["shot"] in tv_shots)]
        e = {"regions": len(rows), "classified": sum(r.get("busy") is not None for r in rows),
             "tagged": sum(r["tagged"] for r in rows)}
        e["tagged_share"] = round(e["tagged"] / e["classified"], 4) if e["classified"] else None
        by_kind = {}
        for r in rows:
            if r.get("busy") is None:
                continue
            k = by_kind.setdefault(r["kind"], [0, 0])
            k[0] += 1
            k[1] += int(r["tagged"])
        e["by_kind"] = {k: {"classified": v[0], "tagged": v[1]} for k, v in sorted(by_kind.items())}
        vns = per_vn(rows)
        e["vns"] = len(vns)
        e["vns_with_tagged"] = sum(1 for v in vns.values() if v[1])
        e["per_vn_tagged"] = {vn: {"classified": v[0], "tagged": v[1]}
                              for vn, v in sorted(vns.items(), key=lambda kv: (-kv[1][1], kv[0])) if v[1]}
        if src == "linefind":
            e["by_set"] = {}
            for name in TD.LINEFIND_SETS:
                rs = [r for r in rows if r["set"] == name]
                e["by_set"][name] = {"regions": len({r["id"] for r in rs}),
                                     "empty": sum(1 for r in rs if r.get("empty")),
                                     "classified": sum(r.get("busy") is not None for r in rs),
                                     "tagged": sum(r["tagged"] for r in rs)}
        doc["sources"][src] = e
    if args.baseline:
        index = wd_tag_index(args.tags)
        err = {}
        for m in MODELS:
            res = json.loads((Path(args.baseline) / m / f"units_{m}.json").read_text(encoding="utf-8"))
            err[m] = {}
            for mode in ("grad", "refine"):
                units = TD.wd_units(res, mode)
                tg = [u for u in units if unit_tagged(index, u)]
                rest = [u for u in units if not unit_tagged(index, u)]
                e_all = sum(u["errors"] for u in units)
                err[m][mode] = {"tagged": TD.cer_of(tg), "rest": TD.cer_of(rest), "all": TD.cer_of(units),
                                "tagged_error_share": round(sum(u["errors"] for u in tg) / e_all, 4),
                                "tagged_units": len(tg), "tagged_vns": len({u["vn"] for u in tg})}
        doc["wd_errors"] = err
        doc["wd_baseline"] = str(args.baseline)
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for src, e in doc["sources"].items():
        print(f"{src}: {e['tagged']} / {e['classified']} tagged ({e['tagged_share']}), {e['vns_with_tagged']} of "
              f"{e['vns']} VNs")
    for m, d in doc.get("wd_errors", {}).items():
        print(m, {k: (v["tagged"]["cer"], v["rest"]["cer"], v["tagged_error_share"]) for k, v in d.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tag")
    p.add_argument("--out", required=True)
    p.add_argument("--source", action="append", choices=SOURCES)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="first N frames of a pool (a smoke test)")
    p = sub.add_parser("summary")
    p.add_argument("--tags", required=True)
    p.add_argument("--baseline", default="")
    p.add_argument("--out", required=True)
    p = sub.add_parser("design")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--frames", type=int, default=0, help="read only the first N shuffled frames")
    p.add_argument("--regions", type=int, default=600, help="tagged regions (and as many untagged)")
    p.add_argument("--per-vn", type=int, default=2)
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--kinds", nargs="*", default=[], choices=sorted(MARGIN))
    p.add_argument("--name", default="design")
    p = sub.add_parser("score")
    p.add_argument("--cohorts", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--arm", action="append", required=True, help="NAME=BLOB")
    p.add_argument("--out", required=True)
    p = sub.add_parser("design-report")
    p.add_argument("--scores", action="append", required=True, help="ARM=DIR of a score output")
    p.add_argument("--out", required=True)
    p = sub.add_parser("protocol")
    p.add_argument("--tags", required=True)
    p.add_argument("--design", action="append", required=True, help="a design cohorts' record (design.json)")
    args = ap.parse_args(argv)
    if args.cmd == "protocol":
        cmd_protocol(args)
    if args.cmd == "design-report":
        cmd_design_report(args)
    if args.cmd == "score":
        cmd_score(args)
    if args.cmd == "tag":
        cmd_tag(args)
    elif args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "design":
        cmd_design(args)


if __name__ == "__main__":
    main()

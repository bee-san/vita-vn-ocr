#!/usr/bin/env python3
"""textspot-desktop round 2 (#173): hard-example sources for the detector, mined with a trained model on the training
frames: the frames where it misses, cuts or merges labelled lines, for the lead to oversample.

  tsdesk_hard_source.py mine --model M.onnx --out DIR [--policy long:960] [--suffix C6] [--pad k6v4] [--rec R.onnx]
                        [--sources 1b,1a] [--targets T] [--label-edits E] [--workers 5] [--limit N] [--shards N]
      the model (tools/tsdesk.Detector through eval/tsdesk_eval.detector_from_spec: tsd:M@POLICY plus the suffix's
      decode options, post-pass and refinement) on every training-side frame of desktop-data's 1b v3 (realplus PC
      screenshots at their own size) and 1a (the hooked pages as desktop captures), read from their stored grey
      frames (the integer luma the detector reads: identical to its own grey of the colour frame). The frames of the
      held-out and test sets are skipped before any model sees them (train/tsdesk_dense_source.exclusion_sets: the
      dense sets, dev-v2, the PC test set, confirm-v2, the hooked test titles, the reservation's test VNs; by VN,
      screenshot sha256 and VNDB screenshot id). The horizontal boxes are read by the recognizer (greedy, at the
      reading pad) for the suffix's reading filter (+drop). Each frame's boxes, before and after the filter, are
      matched against its positives (1a's as the trainer sees them with --targets and --label-edits: train/
      tsdesk_targets.patch_sample, train/tsdesk_coverage.apply_edits) by line_status.
      DIR/rows/<source>_<shard>.jsonl: one row per frame (statuses, counts, the detector's boxes; no pixels, no
      text); a finished shard is skipped; DIR/mine.json: the run record
  tsdesk_hard_source.py build --mined DIR --out OUT [--per-vn-1b 3] [--per-vn-1a 10] [--max-1b 2000] [--max-1a 1000]
                        [--min-score 1] [--manifest-out-1b M] [--manifest-out-1a M]
      the hard frames: score = the main lines (1b: text and name; 1a: dialogue and name) the detector (before the
      filter: what training can change) misses or cuts, plus its boxes that merge two or more lines; frames with a
      score of at least min-score, worst first (score, then the failed share of the frame's main lines), at most
      per-vn frames of a VN and max frames a source. Two sources, each loadable as --extra NAME=MANIFEST:ROOT:
        desk_hard       1b frames, #97's frames (as desk_realplus: boxes from realplus's labels, not exact)
        desk_hook_hard  1a pages, textspot6-pages samples as 1a stores them; train/tsdesk_det.py treats an extra
                        source named desk_hook_* as desk_hook (--targets, --label-edits, the junk recipe, exact ends)
      OUT/<dir>/hard_NNN.{npy,json} and OUT/manifest_{1b,1a}.json: the shards, every frame's VNDB ids, sha256, size
      and failure counts, the selection rule and the exclusion check (the sets reloaded, every written frame against
      them by VN, sha256 and screenshot id, its pixels against its source frame; no manifest unless every count is 0)

v2 (protocol v2's hard_v2; mine --v2, build --rank v2 --version 2): mined with a newer model and C8 (scripts/r2/
read_policy.sh's variant; --suffix C8), 1b's positives with the PC label edits too (--pc-edits: what the trainer's
arms with them see), 1a's labelled lines with their notdlg label edits too (MAIN_V2). Each frame's row adds: the
labelled lines missed split by length (short: at most SHORT glyph cells, the dense gap's 1-6 character labels), and
the extra-character reads: a labelled line one box holds whole (not a merge) whose box reaches EXT_MIN x its height
past either end is read twice (its box and its own label box, the recognizer at the reading pad); the box reads extra
characters when the line's reading is a shorter subsequence of the box's (an icon, leader dots or a bullet read into
it). build --rank v2 orders the frames by missed short labels, then other missed lines, extra-character reads, cuts
and merges (in that order), a salted hash last.

Line status (eval/pagepath.line_stats' rule on even cells): a positive [x0, y0, x1, y1] is cut into round(length /
size across) even cells, one per glyph; a box holds a cell when it holds the cell's centre; a line is whole when one
box holds all its cells, cut when the best box holds some, missed when none does; a box that holds more than half the
cells of two or more lines merges them ("rows": the lines are on different rows; "side": on one row, as buttons side
by side). Positives under MIN_ACROSS px across at the policy's scale are not scored.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "teacher", ROOT / "eval", ROOT / "tools", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path("/local/home/skerraut/herd-vita-ocr")
ART, PV = HERD / "shared" / "artifacts", HERD / "shared" / "private"
SOURCES = {"1b": (ROOT / "data" / "manifests" / "desktop_data" / "realplus_native_v3.json",
                  ART / "desktop-data" / "pages" / "realplus_v3"),
           "1a": (ROOT / "data" / "manifests" / "desktop_data" / "hooked_desktop_v1.json",
                  ART / "desktop-data" / "pages" / "hooked_v1")}
REC = ART / "vocr-onnx" / "release" / "VOCR6.2_L_fast.onnx"
# scripts/r2/read_arm.sh's C6 (protocol v2 amendment 4: the decode vote, the post-pass, the refinement, the filter)
C6 = ("+refine+dec:vote_pct=70+post:merge_pct=70,rows_pct=65,valley_pct=0,ext_pct=1200"
      "+drop:nojp=2,nojp_gap_pct=200,nojp_anchor_pct=50,rep_pct=50")
# scripts/r2/read_policy.sh's C8 (protocol v2 amendment 9: C7 plus the narrow join; the reading stage's cut and its
# filter); the miner applies the filter (+drop: drop_set), not the cut (the score is the detector's, before both)
C8 = ("+refine+dec:vote_pct=70+post:merge_pct=70,rows_pct=65,valley_pct=0,ext_pct=1200,vflip_pct=30,join_pct=10"
      "+drop:nojp=2,nojp_gap_pct=200,nojp_anchor_pct=50,rep_pct=50,cut_pct=25,cut_dlg=0")
SUFFIXES = {"C6": C6, "C8": C8}
MAIN = {"1b": ("text", "name"), "1a": ("dialogue", "name")}
MAIN_V2 = {"1b": ("text", "name"), "1a": ("dialogue", "name", "notdlg")}
SHORT = 6           # glyph cells: a short label
EXT_MIN = 0.5       # x the line's height: a whole line's box reaching this far past an end is read for extra characters
SALT_V2 = "textspot-desktop-hard-v2:"
MIN_ACROSS = 6.0
NAMES = {"1b": ("desk_hard", "desk_hard_v1"), "1a": ("desk_hook_hard", "desk_hook_hard_v1")}
SALT = "textspot-desktop-hard-v1:"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- matching

def line_points(b, vertical):
    """The centres of a positive's even cells, one per glyph (its length over its size across, at least 1)."""
    x0, y0, x1, y1 = (float(v) for v in b[:4])
    if vertical:
        n = max(1, int(round((y1 - y0) / max(1e-6, x1 - x0))))
        return [((x0 + x1) / 2.0, y0 + (k + 0.5) * (y1 - y0) / n) for k in range(n)]
    n = max(1, int(round((x1 - x0) / max(1e-6, y1 - y0))))
    return [(x0 + (k + 0.5) * (x1 - x0) / n, (y0 + y1) / 2.0) for k in range(n)]


def _inside(b, p):
    return b[0] <= p[0] < b[2] and b[1] <= p[1] < b[3]


def line_status(boxes, lines):
    """(statuses, merged) of positives (lines [x0, y0, x1, y1, vertical, ...]) against detector boxes [x0, y0, x1, y1]
    (module docstring): per line "whole", "cut" or "missed"; merged: [(box index, [line indices])]."""
    pts = [line_points(ln[:4], int(ln[4])) for ln in lines]
    cov = [[sum(1 for p in pl if _inside(b, p)) for pl in pts] for b in boxes]
    st = []
    for li, pl in enumerate(pts):
        best = max((cov[bi][li] for bi in range(len(boxes))), default=0)
        st.append("whole" if best == len(pl) else "cut" if best else "missed")
    merged = []
    for bi in range(len(boxes)):
        held = [li for li, pl in enumerate(pts) if 2 * cov[bi][li] > len(pl)]
        if len(held) > 1:
            merged.append((bi, held))
    return st, merged


def same_row(lines):
    """True when every line of lines overlaps the first vertically by at least half the smaller height."""
    a = lines[0]
    return all(min(a[3], b[3]) - max(a[1], b[1]) >= 0.5 * max(1e-6, min(a[3] - a[1], b[3] - b[1])) for b in lines[1:])


def _inside_frac(b, g):
    ix = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
    iy = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
    return ix * iy / max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))


def frame_eval(boxes, lines, ignore, main, f):
    """Counts of one frame's boxes ([x0, y0, x1, y1, vertical]) against its positives: lines [x0, y0, x1, y1,
    vertical, class, ...]; main: the main classes; f: the policy's scale (positives under MIN_ACROSS px across after
    it are not scored). Junk: a box that holds no cell of any positive and lies less than half inside every ignore
    area."""
    scored = [ln for ln in lines if ((ln[2] - ln[0]) if int(ln[4]) else (ln[3] - ln[1])) * f >= MIN_ACROSS]
    st, merged = line_status([b[:4] for b in boxes], scored)
    is_main = [ln[5] in main for ln in scored]
    c = Counter()
    c["main"] = sum(is_main)
    c["other"] = len(scored) - c["main"]
    c["tiny"] = len(lines) - len(scored)
    for s, m in zip(st, is_main):
        c[f"{'main' if m else 'other'}_{s}"] += 1
    for _, held in merged:
        if any(is_main[li] for li in held):
            c["merged_" + ("side" if same_row([scored[li] for li in held]) else "rows")] += 1
    pts = [line_points(ln[:4], int(ln[4])) for ln in lines]
    for b in boxes:
        if not any(_inside(b, p) for pl in pts for p in pl) and not any(_inside_frac(b, g) >= 0.5 for g in ignore):
            c["junk"] += 1
    c["score"] = c["main_missed"] + c["main_cut"] + c["merged_rows"] + c["merged_side"]
    return dict(c), ["w" if s == "whole" else s[0] for s, m in zip(st, is_main) if m]


def line_best(boxes, lines):
    """Per line the index of the box holding most of its cells (ties: the smaller box), or None when none holds one."""
    pts = [line_points(ln[:4], int(ln[4])) for ln in lines]
    out = []
    for pl in pts:
        best, key = None, None
        for bi, b in enumerate(boxes):
            n = sum(1 for p in pl if _inside(b, p))
            if not n:
                continue
            k = (-n, (b[2] - b[0]) * (b[3] - b[1]))
            if key is None or k < key:
                best, key = bi, k
        out.append(best)
    return out


def is_subseq(a, b):
    """True when the characters of a appear in b in order."""
    it = iter(b)
    return all(ch in it for ch in a)


def frame_eval_v2(boxes, lines, ignore, main, f, read=None):
    """frame_eval's counts and failure string plus v2's (module docstring): main_missed_short / main_missed_long, and
    with read (a function: box [x0, y0, x1, y1] -> its reading, or None) extra_lines, extra_chars, extra_read (the
    lines read); details: [(line index among the scored, which end reached, box reading length, line reading length)].
    score_v2: the missed lines, the extra-character reads, the cuts and the merges."""
    c, fail = frame_eval(boxes, lines, ignore, main, f)
    scored = [ln for ln in lines if ((ln[2] - ln[0]) if int(ln[4]) else (ln[3] - ln[1])) * f >= MIN_ACROSS]
    bx = [b[:4] for b in boxes]
    st, merged = line_status(bx, scored)
    for k in ("main_missed_short", "main_missed_long", "extra_lines", "extra_chars", "extra_read"):
        c[k] = 0
    for ln, s_ in zip(scored, st):
        if ln[5] in main and s_ == "missed":
            n = len(line_points(ln[:4], int(ln[4])))
            c["main_missed_short" if n <= SHORT else "main_missed_long"] += 1
    details = []
    if read is not None:
        merge_boxes = {bi for bi, _ in merged}
        best = line_best(bx, scored)
        for li, (ln, s_) in enumerate(zip(scored, st)):
            if ln[5] not in main or s_ != "whole" or int(ln[4]) or best[li] is None or best[li] in merge_boxes:
                continue
            b, h = bx[best[li]], max(1e-6, ln[3] - ln[1])
            left, right = ln[0] - b[0] >= EXT_MIN * h, b[2] - ln[2] >= EXT_MIN * h
            if not (left or right):
                continue
            rb, rl = read(b), read(ln[:4])
            c["extra_read"] += 1
            if rb is not None and rl is not None and len(rb) > len(rl) and is_subseq(rl, rb):
                c["extra_lines"] += 1
                c["extra_chars"] += len(rb) - len(rl)
                details.append((li, "both" if left and right else "left" if left else "right", len(rb), len(rl)))
    c["score_v2"] = (c["main_missed_short"] + c["main_missed_long"] + c["extra_lines"] + c.get("main_cut", 0) +
                     c.get("merged_rows", 0) + c.get("merged_side", 0))
    return c, fail, details


# ----------------------------------------------------------------------------- mining

_SH = {"ex": None, "cache": None, "edits": None, "pc": None}
_W = {}


def _init(spec, rec, pad, v2=False):
    import tsdesk_eval as TE
    import vocr_onnx as V

    import desktop as D
    _W["det"] = TE.detector_from_spec(TE.strip_drop(spec), 1, D.session_options(1))
    _W["drop"] = TE.drop_of_det(spec)
    _W["rec"] = V.Recognizer(str(rec), session_options=D.session_options(1)) if _W["drop"] or v2 else None
    _W["pad"] = pad
    _W["v2"] = v2


def read_box(g, b):
    """The recognizer's reading of box [x0, y0, x1, y1] of a grey frame at the reading pad (detect_read's crop and
    polarity), or None when the box has no crop."""
    import tsdesk as T
    import vocr_onnx as V
    H, W = g.shape
    cb = T.crop_of((b[0], b[1], b[2] - b[0], b[3] - b[1]), T.pad_of(_W["pad"], H), W, H)
    if cb is None:
        return None
    cx, cy, cw, ch = cb
    return _W["rec"].read(g[cy:cy + ch, cx:cx + cw], invert=bool(V.line_dark(g, cb))).text


def detect_read(g):
    """(boxes, gone): the detector's boxes [x0, y0, x1, y1, vertical] of a grey frame and the indices the reading
    filter removes (tools/tsdesk.read_frame's procedure: every horizontal box read at the pad, drop_set on the
    readings; a box with no crop is removed too)."""
    import tsdesk as T
    import vocr_onnx as V
    bx = _W["det"].detect(g)
    boxes = [[float(b.x), float(b.y), float(b.x + b.w), float(b.y + b.h), int(b.vertical)] for b in bx]
    if not _W["drop"]:
        return boxes, []
    H, W = g.shape
    p = T.pad_of(_W["pad"], H)
    items, ks, gone = [], [], set()
    for k, b in enumerate(bx):
        cb = T.crop_of((b.x, b.y, b.w, b.h), p, W, H)
        if cb is None:
            gone.add(k)
            continue
        if b.vertical:
            continue
        cx, cy, cw, ch = cb
        r = _W["rec"].read(g[cy:cy + ch, cx:cx + cw], invert=bool(V.line_dark(g, cb)))
        items.append(((b.x, b.y, b.w, b.h), r.text, r.conf_mean))
        ks.append(k)
    gone |= {ks[i] for i in T.drop_set(items, _W["drop"])}
    return boxes, sorted(gone)


def _frame_ids(smp):
    meta = smp.get("meta") or {}
    return {"vn": smp.get("vn"), "sha256": meta.get("sha256") or meta.get("image_sha256"), "shot": smp.get("shot")}


def _mine_shard(job):
    import tsdesk_coverage as TC
    import tsdesk_dense_source as DS
    import tsdesk_targets as TT
    from desktopdata import shards as SH
    src, root, s, out_dir, policy, limit = job
    path = Path(out_dir) / f"{src}_{s['name']}.jsonl"
    if path.exists():
        return src, s["name"], 0, {}, 0.0
    import tsdesk as T
    t0 = time.time()
    data, samples = SH.read_shard(Path(root) / s.get("dir", "") / s["name"])
    rows, skipped = [], Counter()
    for i, smp in enumerate(samples):
        if limit and len(rows) >= limit:
            break
        hits = DS.hits(_frame_ids(smp), _SH["ex"])
        if hits:
            skipped["frames"] += 1
            for x in hits:
                skipped[x] += 1
            continue
        patched = edited = pc_edited = False
        if src == "1b" and _SH["pc"] is not None:
            es = TC.apply_pc_edits(smp, _SH["pc"])
            if es is not None:
                smp, pc_edited = es, True
        if src == "1a":
            if _SH["cache"] is not None:
                ps = TT.patch_sample(smp, _SH["cache"], "desk_hook")
                if ps is not None:
                    smp, patched = ps, True
            if _SH["edits"] is not None:
                es = TC.apply_edits(smp, _SH["edits"], "desk_hook")
                if es is not None:
                    smp, edited = es, True
        g = SH.canvas(data, smp)
        H, W = g.shape
        boxes, gone = detect_read(g)
        f = T.scale_for(policy, W, H)
        kept = [b for k, b in enumerate(boxes) if k not in set(gone)]
        extra = {}
        if _W.get("v2"):
            pre, fail, details = frame_eval_v2(boxes, smp["lines"], smp["ignore"], MAIN_V2[src], f,
                                               lambda b: read_box(g, b))
            post, _, _ = frame_eval_v2(kept, smp["lines"], smp["ignore"], MAIN_V2[src], f)
            extra = {"pc_edited": pc_edited, "extra": details}
        else:
            pre, fail = frame_eval(boxes, smp["lines"], smp["ignore"], MAIN[src], f)
            post, _ = frame_eval(kept, smp["lines"], smp["ignore"], MAIN[src], f)
        rows.append({"src": src, "dir": s.get("dir", ""), "shard": s["name"], "i": i, **_frame_ids(smp),
                     "title": smp.get("title"), "page": smp.get("page"), "w": W, "h": H, "pre": pre, "post": post,
                     "fail": "".join(fail), "patched": patched, "edited": edited, **extra,
                     "boxes": [[round(v, 1) for v in b[:4]] + [b[4]] for b in boxes], "gone": gone})
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    return src, s["name"], len(rows), dict(skipped), time.time() - t0


def cmd_mine(a):
    import tsdesk_coverage as TC
    import tsdesk_dense_source as DS
    import tsdesk_targets as TT
    out = Path(a.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    a.suffix = SUFFIXES.get(a.suffix, a.suffix)
    spec = f"tsd:{a.model}@{a.policy}{a.suffix}"
    _SH["ex"] = DS.exclusion_sets()
    _SH["pc"] = TC.load_pc_edits(a.pc_edits)[0] if a.pc_edits else None
    srcs = a.sources.split(",")
    if "1a" in srcs:
        _SH["cache"] = TT.load_cache(a.targets)[0] if a.targets else None
        _SH["edits"] = TC.load_edits(a.label_edits)[0] if a.label_edits else None
    jobs = []
    for src in srcs:
        man, root = SOURCES[src]
        doc = json.loads(Path(man).read_text(encoding="utf-8"))
        shards = [s for s in doc["shards"] if s.get("split", "train") == "train"]
        for s in shards[:a.shards] if a.shards else shards:
            if sha256_file(Path(root) / s.get("dir", "") / f"{s['name']}.json") != s["json_sha256"]:
                raise SystemExit(f"{s['name']}.json: sha256 differs from {man}'s")
            jobs.append((src, str(root), s, str(out / "rows"), a.policy, a.limit))
    t0 = time.time()
    done = Counter()
    skipped = defaultdict(Counter)
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(spec, a.rec, a.pad, a.v2)) as p:
        for src, name, n, sk, sec in p.imap_unordered(_mine_shard, jobs):
            done[src] += n
            skipped[src].update(sk)
            print(json.dumps({"src": src, "shard": name, "frames": n, "skipped": sk, "sec": round(sec, 1),
                              "elapsed": round(time.time() - t0, 1)}), flush=True)
    rec = {"kind": "tsdesk-hard-mine", "issue": 173, "model": str(a.model), "model_sha256": sha256_file(a.model),
           "spec": spec, "policy": a.policy, "pad": a.pad, "rec": str(a.rec),
           "rec_sha256": sha256_file(a.rec) if "+drop:" in a.suffix or a.v2 else None,
           "targets": a.targets, "targets_sha256": sha256_file(Path(a.targets) / "targets.json") if a.targets else None,
           "label_edits": a.label_edits,
           "edits_sha256": sha256_file(Path(a.label_edits) / "edits.json") if a.label_edits else None,
           "sources": {s: {"manifest": str(SOURCES[s][0].relative_to(ROOT)), "manifest_sha256":
                           sha256_file(SOURCES[s][0]), "frames": done[s], "skipped": dict(skipped[s])} for s in srcs},
           "code_sha256": sha256_file(Path(__file__)), "seconds": round(time.time() - t0, 1),
           "rule": {"main": MAIN, "min_across_px": MIN_ACROSS, "status": "eval/pagepath.line_stats on even cells"}}
    if a.v2:
        rec["v2"] = {"main": MAIN_V2, "short_cells": SHORT, "ext_min": EXT_MIN, "pc_edits": a.pc_edits,
                     "pc_edits_sha256": sha256_file(Path(a.pc_edits) / "edits.json") if a.pc_edits else None,
                     "extra": "a whole labelled line's box reaching ext_min x its height past an end: the box and the "
                              "line read at the pad; extra when the line's reading is a shorter subsequence of the "
                              "box's"}
    (out / "mine.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("sources", "seconds")}), flush=True)


# ----------------------------------------------------------------------------- the sources

def load_rows(mined):
    rows = defaultdict(list)
    for p in sorted((Path(mined) / "rows").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows[r["src"]].append(r)
    return rows


def vn_key(r):
    """The VN a frame counts against for the per-VN cap: its VNDB id, or its title for a hooked title with none."""
    return r.get("vn") or f"title:{r.get('title')}"


def select(rows, per_vn, cap, min_score=1):
    """The hard frames of one source's rows: score >= min_score, worst first (score, then the failed share of the
    frame's main lines, then a salted hash of the frame for a fixed order), at most per_vn a VN (vn_key) and cap in
    all."""
    cand = [r for r in rows if r["pre"].get("score", 0) >= min_score]
    cand.sort(key=lambda r: (-r["pre"]["score"], -r["pre"]["score"] / max(1, r["pre"].get("main", 0)),
                             hashlib.sha256((SALT + str(r["shot"])).encode()).hexdigest()))
    out, per = [], Counter()
    for r in cand:
        if per[vn_key(r)] >= per_vn:
            continue
        per[vn_key(r)] += 1
        out.append(r)
        if len(out) >= cap:
            break
    return out, len(cand)


def select_v2(rows, per_vn, cap, min_score=1):
    """v2's hard frames: score_v2 >= min_score, ordered by missed short labels, other missed lines, extra-character
    reads, cuts, merges (most first), then a salted hash (SALT_V2); at most per_vn a VN and cap in all."""
    def k(r):
        c = r["pre"]
        return (-c.get("main_missed_short", 0), -c.get("main_missed_long", 0), -c.get("extra_lines", 0),
                -c.get("main_cut", 0), -(c.get("merged_rows", 0) + c.get("merged_side", 0)),
                hashlib.sha256((SALT_V2 + str(r["shot"])).encode()).hexdigest())
    cand = sorted((r for r in rows if r["pre"].get("score_v2", 0) >= min_score), key=k)
    out, per = [], Counter()
    for r in cand:
        if per[vn_key(r)] >= per_vn:
            continue
        per[vn_key(r)] += 1
        out.append(r)
        if len(out) >= cap:
            break
    return out, len(cand)


KINDS_V1 = ("main_missed", "main_cut", "merged_rows", "merged_side", "junk")
KINDS_V2 = KINDS_V1 + ("main_missed_short", "main_missed_long", "extra_lines", "extra_chars")


def _kinds(rows, key="pre"):
    c = Counter()
    for r in rows:
        for k, v in r[key].items():
            c[k] += v
        c["frames"] += 1
    return dict(sorted(c.items()))


def build_source(src, picked, out, shard_frames=200, names=None, kinds=KINDS_V1, score="score"):
    """Write one source's shards (the source frames' samples unchanged but for off, h, w) and run its check;
    returns (shard records, frame records, check)."""
    import tsdesk_dense_source as DS
    from desktopdata import shards as SH
    man, root = SOURCES[src]
    name, dname = (names or NAMES)[src]
    picked = sorted(picked, key=lambda r: (r["vn"] or "", str(r["shot"])))
    cache, frames, shards = {}, [], []

    def frame(r):
        key = (r["dir"], r["shard"])
        if key not in cache:
            cache.clear()
            cache[key] = SH.read_shard(Path(root) / r["dir"] / r["shard"])
        data, samples = cache[key]
        smp = samples[r["i"]]
        return smp, SH.canvas(data, smp).copy()
    for k in range(0, len(picked), shard_frames):
        part, samples, greys = picked[k:k + shard_frames], [], []
        for r in part:
            smp, g = frame(r)
            smp = json.loads(json.dumps(smp))
            if _frame_ids(smp)["shot"] != r["shot"]:
                raise SystemExit(f"{r['shard']}:{r['i']}: not the mined frame")
            samples.append(smp)
            greys.append(g)
            frames.append({"shot": r["shot"], "vn": r["vn"], "sha256": r["sha256"], "title": r.get("title"),
                           "size": [int(g.shape[1]), int(g.shape[0])], "score": r["pre"][score],
                           "main": r["pre"].get("main", 0), "fail": r["fail"],
                           "kinds": {x: r["pre"].get(x, 0) for x in kinds},
                           "pixels_sha256": hashlib.sha256(g.tobytes()).hexdigest(),
                           "from": f"{r['dir']}/{r['shard']}:{r['i']}"})
        sname = f"hard_{k // shard_frames:03d}"
        SH.write_shard(Path(out) / dname / sname, samples, greys)
        shards.append(SH.shard_record(Path(out) / dname / sname, dname, {"split": "train"} if src == "1a" else None))
    # the check, on what was written: the sets reloaded and compared without hits(), the pixels against the source
    ex = DS.exclusion_sets()
    check = {n: {"vn": 0, "sha256": 0, "shot": 0, "vns_in_set": len(e["vns"]), "sha256_in_set": len(e["sha256"]),
                 "shots_in_set": len(e["shots"]), "source": e["source"]} for n, e in ex.items()}
    src_of = {f["shot"]: f["from"] for f in frames}
    n_frames = n_pix = 0
    read = {}
    for rec in shards:
        data, samples = SH.read_shard(Path(out) / dname / rec["name"])
        for smp in samples:
            n_frames += 1
            ids = _frame_ids(smp)
            for n, e in ex.items():
                check[n]["vn"] += ids["vn"] in e["vns"]
                check[n]["sha256"] += ids["sha256"] in e["sha256"]
                check[n]["shot"] += ids["shot"] in e["shots"]
            sd, si = src_of[ids["shot"]].rsplit(":", 1)
            if sd not in read:
                read.clear()
                read[sd] = SH.read_shard(Path(root) / sd)
            d0, s0 = read[sd]
            if not np.array_equal(SH.canvas(data, smp), SH.canvas(d0, s0[int(si)])):
                n_pix += 1
    bad = sum(v[w] for v in check.values() for w in ("vn", "sha256", "shot")) + n_pix
    return shards, frames, {"frames_checked": n_frames, "by_set": check, "pixels_differ_from_source": n_pix,
                            "passed": bad == 0}


def cmd_build(a):
    from desktopdata import shards as SH
    mined = Path(a.mined)
    mine = json.loads((mined / "mine.json").read_text(encoding="utf-8"))
    rows = load_rows(mined)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caps = {"1b": (a.per_vn_1b, a.max_1b), "1a": (a.per_vn_1a, a.max_1a)}
    dense = set()
    dm = ROOT / "data" / "manifests" / "desktop_data" / "tsdesk_dense_train_v1.json"
    if dm.exists():
        dense = {f["shot"] for f in json.loads(dm.read_text(encoding="utf-8"))["frames"]}
    res = {}
    v2 = getattr(a, "rank", "v1") == "v2"
    names = {k: (v[0], f"{v[0]}_v{getattr(a, 'version', 1)}") for k, v in NAMES.items()}
    for src in sorted(rows):
        per_vn, cap = caps[src]
        picked, n_cand = (select_v2 if v2 else select)(rows[src], per_vn, cap, a.min_score)
        shards, frames, check = build_source(src, picked, out, names=names, kinds=KINDS_V2 if v2 else KINDS_V1,
                                             score="score_v2" if v2 else "score")
        name, dname = names[src]
        if not check["passed"]:
            print(json.dumps({"src": src, "check": check}, indent=1))
            raise SystemExit(f"{src}: the exclusion check failed: no manifest written")
        tot = SH.totals(shards)["all"] if shards else {}
        rec = {"kind": "textspot-desktop-hard-source", "schema": 1, "issue": 173, "name": name,
               "created": SH.utc(), "load": f"--extra {name}=MANIFEST:{out} (add {name}=W to --mix)",
               "format": "#97's frames" if src == "1b" else "textspot6-pages samples of 1a (src hook, kind desktop): "
                         "train/tsdesk_det.py treats an extra source named desk_hook_* as desk_hook",
               "mined_with": {**{k: mine[k] for k in ("model", "model_sha256", "spec", "policy", "pad", "rec_sha256",
                                                       "targets", "targets_sha256", "label_edits", "edits_sha256",
                                                       "code_sha256")}, **({"v2": mine.get("v2")} if v2 else {})},
               "source": mine["sources"][src],
               "rule": ({"score": "main lines missed or cut plus boxes merging two or more lines, by the detector "
                                  "before the reading filter (line_status on even cells)", "main": MAIN[src],
                         "per_vn_key": "the VNDB id, or the title for a hooked title without one",
                         "min_score": a.min_score, "order": "score, the failed share of the main lines, a salted hash",
                         "per_vn": per_vn, "cap": cap, "salt": SALT} if not v2 else
                        {"score": "score_v2: labelled lines missed (short: <= SHORT cells), extra-character reads, "
                                  "cuts, merges, by the detector before the reading filter", "main": MAIN_V2[src],
                         "short_cells": SHORT, "ext_min": EXT_MIN, "min_score": a.min_score,
                         "order": "missed short labels, missed other lines, extra-character reads, cuts, merges, "
                                  "a salted hash", "per_vn_key": "the VNDB id, or the title for a hooked title "
                                                                "without one", "per_vn": per_vn, "cap": cap,
                         "salt": SALT_V2}),
               "candidates": {"frames_mined": len(rows[src]), "frames_with_score": n_cand,
                              "rates_all_frames_pre": _kinds(rows[src], "pre"),
                              "rates_all_frames_post": _kinds(rows[src], "post")},
               "check": check,
               "totals": {"frames": len(frames), "vns": len({f["vn"] for f in frames}), "lines": tot.get("lines", 0),
                          "ignore": tot.get("ignore", 0), "pixels": tot.get("pixels", 0),
                          "by_class": tot.get("by_class", {}), "failure_kinds_pre": _kinds(picked, "pre"),
                          "failure_kinds_post": _kinds(picked, "post"),
                          "sizes": dict(Counter(f"{f['size'][0]}x{f['size'][1]}" for f in frames).most_common(15)),
                          "max_frames_per_vn": max(Counter(vn_key(f) for f in frames).values()) if frames else 0,
                          "in_desk_dense": sum(1 for f in frames if f["shot"] in dense),
                          "npy_bytes": sum((out / dname / f"{r['name']}.npy").stat().st_size for r in shards)},
               "code_sha256": sha256_file(Path(__file__)), "frames": frames, "shards": shards,
               "note": "shards hold copyrighted frames and stay local; this manifest holds names, counts, sha256 and "
                       "VNDB ids"}
        txt = json.dumps(rec, ensure_ascii=False, indent=1) + "\n"
        (out / f"manifest_{src}.json").write_text(txt, encoding="utf-8")
        mo = a.manifest_out_1b if src == "1b" else a.manifest_out_1a
        if mo:
            Path(mo).write_text(txt, encoding="utf-8")
        res[src] = {"name": name, "frames": len(frames), "vns": rec["totals"]["vns"], "candidates": n_cand,
                    "kinds": rec["totals"]["failure_kinds_pre"], "check_passed": True}
    print(json.dumps(res), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mine")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policy", default="long:960")
    p.add_argument("--suffix", default=C6, help="the decode options, post-pass, refinement and filter: a suffix, or "
                                               "C6 / C8 (SUFFIXES)")
    p.add_argument("--pad", default="k6v4")
    p.add_argument("--rec", default=str(REC))
    p.add_argument("--sources", default="1b,1a")
    p.add_argument("--targets")
    p.add_argument("--label-edits")
    p.add_argument("--pc-edits", help="a PC label-edit cache (train/tsdesk_coverage.py pcedits) for 1b's positives")
    p.add_argument("--v2", action="store_true", help="v2's rows: MAIN_V2, short / long misses, extra-character reads")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="frames a shard (smoke runs)")
    p.add_argument("--shards", type=int, default=0, help="shards a source (smoke runs)")
    p = sub.add_parser("build")
    p.add_argument("--mined", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-vn-1b", type=int, default=3)
    p.add_argument("--per-vn-1a", type=int, default=10)
    p.add_argument("--max-1b", type=int, default=2000)
    p.add_argument("--max-1a", type=int, default=1000)
    p.add_argument("--min-score", type=int, default=1)
    p.add_argument("--rank", default="v1", choices=("v1", "v2"), help="v2: select_v2's order on v2 rows")
    p.add_argument("--version", type=int, default=1, help="the sources' directories: desk_hard_vN, desk_hook_hard_vN")
    p.add_argument("--manifest-out-1b")
    p.add_argument("--manifest-out-1a")
    a = ap.parse_args(argv)
    {"mine": cmd_mine, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()

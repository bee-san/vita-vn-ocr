#!/usr/bin/env python3
"""textspot7's fresh confirmation set (issue #128, docs/textspot7.md "The confirmation set"): whole-dialogue regions
of unseen games that no spotter trained on and no diagnosis used, built and frozen before any read of a new arm.

  textspot7_confirm.py pool --out POOL.json
      the exclusions (VN ids) and the candidate frames: native-size VNDB screenshots (eval/'s VNDB listings, Vita
      960x544 and PSP 480x272, the repo's content filters: sexual < 0.3, violence < 0.5) and the fresh native
      textbox frames (eval/fresh_native_source_provenance_20261012.json: press and blog screenshots), of every VN
      outside the exclusions; no screenshot of a test set's capture session
  textspot7_confirm.py fetch --pool POOL.json --dir SHOTS [--delay 1.0]
      the candidate frames into SHOTS (private): a local copy where one exists (eval's shots, sha256 checked),
      else a download (one request per --delay seconds, eval/fetch.py's User-Agent); a fresh native frame must
      match its recorded sha256; SHOTS/fetched.jsonl records every frame's sha256 and dims
  textspot7_confirm.py propose --pool POOL.json --dir SHOTS --out PROPOSALS.jsonl [--workers 5]
      the labelling proposals (heavy: run it through heavy.sh): MeikiOCR (teacher/engines.Meiki, its detector at
      det_thr 0.3 and its recognizer) on every frame, and PP-OCRv6 small (teacher/engines.PaddleRec "v6s") on every
      horizontal Meiki box cut with eval's pad (4 px Vita, 2 px PSP)
  textspot7_confirm.py sheets --pool POOL.json --dir SHOTS --proposals PROPOSALS.jsonl --out SHEETS [--per-vn 6]
      one review sheet per chosen frame (<= 1280 px a side): the frame with the numbered proposal boxes and, under
      it, every horizontal proposal's crop at 2x with its two readings; plus SHEETS/batch_<k>.json for the reviewers
  textspot7_confirm.py build --pool POOL.json --dir SHOTS --proposals PROPOSALS.jsonl --decisions DIR \\
                             --out UNITS.pkl --manifest PUBLIC.json
      the reviewed regions as #119's unit format (eval/pagepath.py's dev units: base / expanded-4 / expanded-8) and
      the public manifest (ids, VN, URLs, sha256, regions and line boxes; no text: the references stay private)

The exclusions (agreed with vocr6-spotter2, 2026-10-27 06:45Z; VN ids, strict):
  hooked     every hooked title's VNs, all splits ($A/vocr6-ingest/audit/overlap.json)
  textspot2  textspot2's real half: every VN of its 52,785 canvases (the shard JSONs)
  scenes     the VNs of textspot's synthetic frames' scene backgrounds (scene_pool.json)
  dev        #119's units, the dev sample with its release partners (eval/manifests/reserved.json dev_sample),
             textspot's development frames (eval/manifests/textspot-dev.jsonl)
  test       vita-test, psp-test and the CLANNAD family with their release partners (reserved.json test); no
             screenshot of their capture sessions (reserved.json sessions)
  owner      every VN of the owner's screenshots (reserved.json owner_vns)
  linefind   eval/linefind's sets, where GRAD was tuned
  extra      --exclude-vns FILE: VNs a partner adds (vocr6-spotter2's further training frames)

Everything under SHOTS, the proposals, the sheets, the decisions and UNITS.pkl is private (copyrighted frames and
their text): local only, never committed, never sent anywhere.
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIV = HERD / "shared" / "private"
UA = "vita-vn-ocr-eval/0.1 (benchmark rebuild; github.com/bee-san/vita-vn-ocr)"
NATIVE = {"vita": [960, 544], "psp": [480, 272]}
LISTINGS = (("vita", "vndb_psv.jsonl"), ("vita", "vndb_psv_any.jsonl"), ("psp", "vndb_psp.jsonl"))
FRESH = ROOT / "eval" / "fresh_native_source_provenance_20261012.json"
MARGIN = {"vita": 12, "psp": 6}                     # eval's region rule: the lines' union plus this margin
VARIANTS = {"base": 0, "expanded-4": 4, "expanded-8": 8}
PAD = {"vita": 4, "psp": 2}                         # eval's crop pad at native size


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ----------------------------------------------------------------------------- pool

def exclusions(extra=None):
    """{name: set of VN ids} (module docstring)."""
    ex = {}
    ov = json.loads((ART / "vocr6-ingest" / "audit" / "overlap.json").read_text(encoding="utf-8"))
    ex["hooked"] = {v for t in ov["titles"].values() for v in t["vns"]}
    ts2 = set()
    files = sorted(glob.glob(str(ART / "textspot2" / "shards" / "*" / "*" / "*.json")) +
                   glob.glob(str(PRIV / "textspot2" / "shards" / "*" / "*" / "*.json")))
    canvases = 0
    for f in files:
        for s in json.loads(Path(f).read_text(encoding="utf-8"))["samples"]:
            canvases += 1
            if s.get("vn"):
                ts2.add(s["vn"])
    if canvases != 52785:
        raise SystemExit(f"textspot2's shards hold {canvases} canvases, not 52,785: the pool moved")
    ex["textspot2"] = ts2
    sp = json.loads((ART / "textspot" / "shards" / "synth_frames" / "scene_pool.json").read_text(encoding="utf-8"))
    ex["scenes"] = {n.split("_")[0] for k in sp["paths"] for n in sp["paths"][k]}
    res = json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text(encoding="utf-8"))
    units = pickle.load(open(ART / "vocr5-analysis" / "work" / "units.pkl", "rb"))["units"]
    ex["dev"] = (set(res["dev_sample"]) | {u["vn"] for u in units} |
                 {r["vn"] for r in read_jsonl(ROOT / "eval" / "manifests" / "textspot-dev.jsonl")})
    ex["test"] = set(res["test"])
    ex["owner"] = set(res["owner_vns"])
    lf = set()
    for f in sorted(glob.glob(str(ROOT / "eval" / "linefind" / "manifests" / "*.jsonl"))):
        for r in read_jsonl(f):
            if r.get("vn"):
                lf.add(r["vn"])
    ex["linefind"] = lf
    ex["extra"] = set(json.loads(Path(extra).read_text(encoding="utf-8"))) if extra else set()
    return ex, sorted(res["sessions"])


def cmd_pool(a):
    ex, sessions = exclusions(a.exclude_vns)
    bad = set().union(*ex.values())
    cands, seen = [], set()
    for plat, fn in LISTINGS:
        for s in read_jsonl(ART / "eval" / "vndb" / fn):
            if s.get("dims") != NATIVE[plat] or s["vn"] in bad or s.get("release") in sessions:
                continue
            if not (s.get("sexual", 1) < 0.3 and s.get("violence", 1) < 0.5):
                continue
            shot = f"{s['vn']}_{Path(s['url']).stem}"
            if shot in seen:
                continue
            seen.add(shot)
            cands.append({"id": f"vndb-{shot}", "source": "vndb", "vn": s["vn"], "platform": plat, "url": s["url"],
                          "dims": s["dims"], "release": s.get("release"), "sexual": s.get("sexual"),
                          "violence": s.get("violence")})
    fresh = json.loads(FRESH.read_text(encoding="utf-8"))["sources"]
    for s in fresh:
        if s["vn"] in bad or s.get("uncertain"):
            continue
        cands.append({"id": s["id"], "source": "fresh-native", "vn": s["vn"], "platform": s["platform"],
                      "url": s["url"], "dims": s["dims"], "release": s.get("release"), "sha256": s["sha256"],
                      "roi_hint": s["base_roi"]})
    cands.sort(key=lambda c: (c["vn"], c["id"]))
    vns = Counter((c["platform"], c["vn"]) for c in cands)
    doc = {"kind": "textspot7-confirm-pool", "schema": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "exclusions": {k: sorted(v) for k, v in ex.items()}, "exclusion_counts": {k: len(v) for k, v in ex.items()},
           "excluded_vns": len(bad), "sessions": sessions, "candidates": cands,
           "candidate_vns": len({v for _, v in vns}), "candidate_frames": len(cands),
           "by_platform": dict(Counter(c["platform"] for c in cands)),
           "by_source": dict(Counter(c["source"] for c in cands))}
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"pool: {len(cands)} frames of {doc['candidate_vns']} VNs {doc['by_platform']} {doc['by_source']}; "
          f"{len(bad)} VNs excluded -> {a.out}")


# ----------------------------------------------------------------------------- fetch

def local_copies():
    """{url: (path, sha256)} of the screenshots already on this host with a recorded sha256 (eval's pools)."""
    out = {}
    for plat in ("vita", "psp"):
        p = ART / "eval" / plat / "shots.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                out[r["url"]] = (ART / "eval" / plat / "shots" / r["shot"], r["sha256"])
    return out


def cmd_fetch(a):
    import cv2
    import numpy as np
    import requests
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    log = d / "fetched.jsonl"
    done = {r["id"]: r for r in read_jsonl(log)} if log.exists() else {}
    local = local_copies()
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    n_new = n_fail = 0
    with open(log, "a", encoding="utf-8") as lf:
        for c in pool["candidates"]:
            if c["id"] in done:
                continue
            ext = Path(c["url"]).suffix or ".jpg"
            dest = d / f"{c['id']}{ext}"
            data, how = None, None
            if c["url"] in local and local[c["url"]][0].exists():
                src, want = local[c["url"]]
                b = src.read_bytes()
                if sha256_bytes(b) == want:
                    data, how = b, "local"
            if data is None:
                for attempt in range(4):
                    try:
                        r = sess.get(c["url"], timeout=60)
                        if r.status_code == 200 and r.content:
                            data, how = r.content, "download"
                            break
                        if r.status_code in (404, 410):
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(5 * (attempt + 1))
                time.sleep(a.delay)
            rec = {"id": c["id"], "url": c["url"]}
            if data is None:
                rec.update(ok=False, why="download failed")
            else:
                sha = sha256_bytes(data)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                dims = [int(img.shape[1]), int(img.shape[0])] if img is not None else None
                if c.get("sha256") and sha != c["sha256"]:
                    rec.update(ok=False, why="sha256 differs from the recorded one", sha256=sha)
                elif dims != c["dims"]:
                    rec.update(ok=False, why=f"dims {dims}", sha256=sha)
                else:
                    dest.write_bytes(data)
                    rec.update(ok=True, file=dest.name, sha256=sha, dims=dims, how=how)
            n_new += 1
            n_fail += not rec["ok"]
            lf.write(json.dumps(rec) + "\n")
            lf.flush()
    print(f"fetch: {n_new} new records ({n_fail} failed); {log}")


def fetched(dirpath):
    return {r["id"]: r for r in read_jsonl(Path(dirpath) / "fetched.jsonl") if r.get("ok")}


# ----------------------------------------------------------------------------- the frozen frame list

def review_key(fid):
    """The review order within a VN: a fixed shuffle (textspot2's pool-order rule)."""
    return hashlib.sha256(f"textspot7-confirm:{fid}".encode()).hexdigest()


def thumb(path):
    import cv2
    import numpy as np
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    t = cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def cmd_freeze_frames(a):
    """The public frame list (committed before any new arm is read): every fetched candidate in its review order,
    a frame whose bytes or thumbnail (32 x 18 grey, correlation > 0.97) repeat an earlier frame's marked dropped."""
    import numpy as np
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    got = fetched(a.dir)
    frames = [dict(c, sha256=got[c["id"]]["sha256"], file=got[c["id"]]["file"]) for c in pool["candidates"]
              if c["id"] in got]
    frames.sort(key=lambda f: (f["vn"], review_key(f["id"])))
    kept, seen_sha = [], {}
    for f in frames:
        t = thumb(Path(a.dir) / f["file"])
        dup = seen_sha.get(f["sha256"])
        if dup is None:
            for g, tg in kept:
                if float(np.dot(t, tg)) / t.size > 0.97:
                    dup = g["id"]
                    break
        if dup is not None:
            f["dropped"] = f"duplicate of {dup}"
        else:
            kept.append((f, t))
            seen_sha[f["sha256"]] = f["id"]
    vns = defaultdict(int)
    for f in frames:
        if "dropped" not in f:
            vns[f["vn"]] += 1
            f["review_rank"] = vns[f["vn"]]
    rows = [{k: f.get(k) for k in ("id", "source", "vn", "platform", "url", "sha256", "dims", "release", "review_rank",
                                    "dropped") if f.get(k) is not None} for f in frames]
    doc = {"kind": "textspot7-confirm-frames", "schema": 1, "issue": 128,
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pool_rule": "eval/textspot7_confirm.py pool (module docstring): the exclusions agreed with vocr6-spotter2",
           "exclusion_counts": pool["exclusion_counts"], "excluded_vns": pool["excluded_vns"],
           "review_order": "within a VN, sha256('textspot7-confirm:' + id); a reviewer accepts at most "
                           f"{a.per_vn} frames per VN in this order",
           "per_vn": a.per_vn, "frames": rows, "kept": len(kept), "dropped": len(frames) - len(kept),
           "vns": sorted(vns), "by_platform": dict(Counter(f["platform"] for f, _ in kept)),
           "by_source": dict(Counter(f["source"] for f, _ in kept)),
           "note": "frames are copyrighted screenshots: the URLs and sha256 only; nothing else of them is committed"}
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"frames: {len(kept)} kept of {len(frames)} ({len(vns)} VNs) {doc['by_platform']} -> {a.out} "
          f"sha256 {sha256_file(a.out)}")



# ----------------------------------------------------------------------------- proposals

def frame_list(path, include_dropped=False):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [f for f in doc["frames"] if include_dropped or "dropped" not in f]


def frame_file(dirpath, fid):
    got = fetched(dirpath)
    return Path(dirpath) / got[fid]["file"]


def cmd_propose(a):
    """MeikiOCR (det_thr 0.3, its recognizer) and PP-OCRv6 small on each horizontal Meiki box (eval's pad)."""
    import cv2
    import engines as E
    frames = frame_list(a.frames)
    got = fetched(a.dir)
    meiki = E.Meiki(det_thr=0.3)
    v6 = E.PaddleRec("v6s")
    done = set()
    out = Path(a.out)
    if out.exists():
        done = {r["id"] for r in read_jsonl(out)}
    t0 = time.time()
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for fr in frames:
            if fr["id"] in done:
                continue
            path = Path(a.dir) / got[fr["id"]]["file"]
            if sha256_file(path) != fr["sha256"]:
                raise SystemExit(f"{path}: sha256 differs from the frozen frame list")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            H, W = bgr.shape[:2]
            pad = PAD[fr["platform"]]
            boxes = []
            for k, ln in enumerate(meiki.ocr(bgr)):
                x0, y0, x1, y1 = ln["det"]
                rec = {"k": k + 1, "det": [int(v) for v in ln["det"]], "det_conf": ln["det_conf"],
                       "vertical": ln["vertical"], "meiki": ln["text"]}
                if not ln["vertical"] and x1 > x0 and y1 > y0:
                    c = bgr[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
                    t, mean, mn = v6(np.ascontiguousarray(c))[:3]
                    rec.update(v6=t, v6_conf=round(float(mn), 3))
                boxes.append(rec)
            f.write(json.dumps({"id": fr["id"], "vn": fr["vn"], "platform": fr["platform"], "dims": [W, H],
                                "boxes": boxes}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    print(f"propose: {n} frames in {time.time() - t0:.0f} s -> {out}")


# ----------------------------------------------------------------------------- sheets

FONT = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"
SHEET_W = 1280
COLOURS = [(255, 64, 64), (64, 200, 255), (255, 200, 0), (80, 255, 80), (255, 80, 255), (255, 140, 0)]


def console_view(bgr, platform):
    """The frame as the reviewers see it: Vita as it is, PSP 2x (bilinear), RGB."""
    import cv2
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    if platform == "psp":
        rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return rgb


def make_sheet(rgb_console, platform, boxes, title):
    """One review sheet: the console view with a 100-px grid and the numbered proposals, then each horizontal
    proposal's crop with its two readings. <= 1280 px a side."""
    from PIL import Image, ImageDraw, ImageFont
    s = 2 if platform == "psp" else 1
    font = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 14)
    fr = Image.fromarray(rgb_console)
    H, W = rgb_console.shape[:2]
    d = ImageDraw.Draw(fr)
    for x in range(100, W, 100):
        d.line([(x, 0), (x, H)], fill=(128, 128, 128), width=1)
        d.text((x + 2, 2), str(x), fill=(255, 255, 0), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
    for y in range(100, H, 100):
        d.line([(0, y), (W, y)], fill=(128, 128, 128), width=1)
        d.text((2, y + 2), str(y), fill=(255, 255, 0), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
    for b in boxes:
        col = COLOURS[(b["k"] - 1) % len(COLOURS)]
        x0, y0, x1, y1 = (v * s for v in b["det"])
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=col, width=2)
        d.text((x0 + 1, max(0, y0 - 18)), str(b["k"]), fill=col, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    rows = []
    for b in boxes:
        if b["vertical"]:
            continue
        x0, y0, x1, y1 = (v * s for v in b["det"])
        pad = 4
        cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        crop = Image.fromarray(np.ascontiguousarray(rgb_console[cy0:cy1, cx0:cx1]))
        if crop.width > SHEET_W - 60:
            crop = crop.resize((SHEET_W - 60, max(1, round(crop.height * (SHEET_W - 60) / crop.width))))
        rows.append((b, crop))
    text_h = 26
    body = sum(c.height + text_h + 6 for _, c in rows) + 34
    sheet_h = H + body
    scale = 1.0
    if sheet_h > SHEET_W:
        scale = SHEET_W / sheet_h
    sheet = Image.new("RGB", (SHEET_W, sheet_h), (40, 40, 40))
    sheet.paste(fr, (0, 0))
    d = ImageDraw.Draw(sheet)
    d.text((W + 10, 10), title, fill=(255, 255, 255), font=small)
    y = H + 4
    d.text((4, y), "proposals: [k] M = MeikiOCR, P = PP-OCRv6 small (min conf); crops below at console scale",
           fill=(200, 200, 200), font=small)
    y += 26
    for b, crop in rows:
        col = COLOURS[(b["k"] - 1) % len(COLOURS)]
        d.text((4, y), f"[{b['k']}] M: {b['meiki']}", fill=col, font=font)
        d.text((SHEET_W // 2 + 10, y), f"P: {b.get('v6', '')}  ({b.get('v6_conf', '')})", fill=col, font=font)
        y += text_h
        sheet.paste(crop, (40, y))
        y += crop.height + 6
    if scale < 1.0:
        sheet = sheet.resize((max(1, round(SHEET_W * scale)), SHEET_W))
    return sheet


def cmd_sheets(a):
    import cv2
    frames = frame_list(a.frames)
    props = {r["id"]: r for r in read_jsonl(a.proposals)}
    got = fetched(a.dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by_vn = defaultdict(list)
    for fr in frames:
        by_vn[fr["vn"]].append(fr)
    vns = sorted(by_vn)
    batches = [vns[i::a.batches] for i in range(a.batches)]
    for bi, bvns in enumerate(batches, 1):
        items = []
        for vn in bvns:
            for fr in sorted(by_vn[vn], key=lambda f: f["review_rank"]):
                bgr = cv2.imread(str(Path(a.dir) / got[fr["id"]]["file"]), cv2.IMREAD_COLOR)
                p = props[fr["id"]]
                sheet = make_sheet(console_view(bgr, fr["platform"]), fr["platform"], p["boxes"],
                                   f"{fr['id']}  ({fr['platform']}, rank {fr['review_rank']} of {vn})")
                name = f"b{bi:02d}_{fr['id']}.png"
                sheet.save(out / name)
                items.append({"frame": fr["id"], "vn": vn, "platform": fr["platform"], "rank": fr["review_rank"],
                              "sheet": name, "proposals": [{k: b.get(k) for k in ("k", "det", "vertical", "meiki",
                                                                                     "v6", "v6_conf")}
                                                           for b in p["boxes"]]})
        (out / f"batch_{bi:02d}.json").write_text(json.dumps({"batch": bi, "vns": bvns, "per_vn": a.per_vn,
                                                              "items": items}, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")
        print(f"batch {bi}: {len(bvns)} VNs, {len(items)} sheets")


# ----------------------------------------------------------------------------- build

def split_rows(grey, box, n):
    """A box holding n stacked lines cut at the n - 1 emptiest rows between them (ink = 3x3 contrast >= 40),
    each cut at least a sixth of the box's height from the others and the edges. -> n boxes [x0, y0, x1, y1]."""
    from common import contrast3
    x0, y0, x1, y1 = box
    c = contrast3(grey[y0:y1, x0:x1]) >= 40
    prof = c.sum(axis=1)
    h = y1 - y0
    cuts, minsep = [], max(2, h // (3 * n))
    for _ in range(n - 1):
        best = None
        for yy in range(minsep, h - minsep):
            if any(abs(yy - k) < minsep for k in cuts):
                continue
            if best is None or prof[yy] < prof[best]:
                best = yy
        if best is None:
            break
        cuts.append(best)
    cuts = sorted(cuts)
    edges = [0, *cuts, h]
    return [[x0, y0 + edges[i], x1, y0 + edges[i + 1]] for i in range(len(edges) - 1)]


def cmd_build(a):
    """The units from the reviewers' decisions (DIR/*.jsonl): #119's format (a pickle {"units": [...]}: unit, variant,
    vn, kind, src_kind, scale 1.0, shot, cohort, region [x0, y0, x1, y1] native, image (grey region), lines [{id,
    text, box [x0, y0, x1, y1] in region px}]) and the public manifest (no text)."""
    import cv2
    import geometry as GEO

    from common import to_gray
    frames = {f["id"]: f for f in frame_list(a.frames)}
    props = {r["id"]: r for r in read_jsonl(a.proposals)}
    got = fetched(a.dir)
    dec = {}
    for f in sorted(Path(a.decisions).glob("*.jsonl")):
        for r in read_jsonl(f):
            dec[r["frame"]] = r                                    # the last decision of a frame wins
    accepted = defaultdict(list)
    for fid, r in dec.items():
        if r.get("accept"):
            accepted[frames[fid]["vn"]].append(fid)
    units, public, skipped = [], [], Counter()
    for vn in sorted(accepted):
        fids = sorted(accepted[vn], key=lambda i: frames[i]["review_rank"])[:a.per_vn]
        for fid in fids:
            fr, r, p = frames[fid], dec[fid], props[fid]
            bgr = cv2.imread(str(Path(a.dir) / got[fid]["file"]), cv2.IMREAD_COLOR)
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            grey = to_gray(rgb)
            H, W = grey.shape
            s = 2 if fr["platform"] == "psp" else 1
            kb = {b["k"]: b for b in p["boxes"]}
            for gi, grp in enumerate(r["units"]):
                lines = []
                for ln in grp["lines"]:
                    if "k" in ln and ln.get("split"):
                        parts = split_rows(grey, kb[ln["k"]]["det"], int(ln["split"]))
                        texts = ln["texts"]
                        if len(parts) != len(texts):
                            raise SystemExit(f"{fid}: box {ln['k']} split into {len(parts)}, {len(texts)} texts")
                        for bx, t in zip(parts, texts):
                            lines.append((bx, t))
                    elif "k" in ln:
                        lines.append((kb[ln["k"]]["det"], ln["text"]))
                    else:
                        x, y, w, h = ln["add"]
                        lines.append(([x // s, y // s, (x + w + s - 1) // s, (y + h + s - 1) // s], ln["text"]))
                ink = []
                for bx, t in lines:
                    ib, _ = GEO.refine(rgb, [max(0, bx[0]), max(0, bx[1]), min(W, bx[2]), min(H, bx[3])])
                    ink.append((ib, t))
                m = MARGIN[fr["platform"]]
                rx0 = max(0, min(b[0] for b, _ in ink) - m)
                ry0 = max(0, min(b[1] for b, _ in ink) - m)
                rx1 = min(W, max(b[2] for b, _ in ink) + m)
                ry1 = min(H, max(b[3] for b, _ in ink) + m)
                for var, e in VARIANTS.items():
                    x0, y0 = max(0, rx0 - e), max(0, ry0 - e)
                    x1, y1 = min(W, rx1 + e), min(H, ry1 + e)
                    uid = f"{fid}@{rx0},{ry0},{rx1},{ry1}"
                    rl = [{"id": f"{fid}_{gi}_{li}", "text": t,
                           "box": [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]} for li, (b, t) in enumerate(ink)]
                    units.append({"unit": uid, "variant": var, "vn": vn, "kind": fr["platform"],
                                  "src_kind": fr["platform"], "scale": 1.0, "shot": fid,
                                  "cohort": "textspot7_confirm", "region": [x0, y0, x1, y1],
                                  "image": np.ascontiguousarray(grey[y0:y1, x0:x1]), "lines": rl,
                                  "group_kind": grp.get("kind", "dialogue")})
                    public.append({"unit": uid, "variant": var, "vn": vn, "platform": fr["platform"], "frame": fid,
                                   "frame_sha256": fr["sha256"], "url": fr["url"], "region": [x0, y0, x1, y1],
                                   "lines": [{"id": x["id"], "box": x["box"], "chars": len(x["text"])} for x in rl],
                                   "kind": grp.get("kind", "dialogue")})
        skipped[vn] = len(accepted[vn]) - len(fids)
    out = Path(a.out)
    with open(out, "wb") as f:
        pickle.dump({"units": units, "kind": "textspot7-confirm-units", "created": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    doc = {"kind": "textspot7-confirm-units", "schema": 1, "issue": 128, "units_pkl_sha256": sha256_file(out),
           "frames_manifest": a.frames, "frames_manifest_sha256": sha256_file(a.frames),
           "reviewer_kind": "assistant", "decisions": len(dec),
           "accepted_frames": sum(min(len(v), a.per_vn) for v in accepted.values()),
           "vns": len({u["vn"] for u in units}), "units": len(units),
           "base_regions": sum(1 for u in units if u["variant"] == "base"),
           "lines": sum(len(u["lines"]) for u in units if u["variant"] == "base"),
           "chars": sum(len(x["text"]) for u in units if u["variant"] == "base" for x in u["lines"]),
           "by_platform": dict(Counter(u["kind"] for u in units if u["variant"] == "base")),
           "units_public": public}
    Path(a.manifest).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"build: {doc['base_regions']} base regions ({len(units)} units) of {doc['vns']} VNs, {doc['lines']} "
          f"lines -> {out} (sha256 {doc['units_pkl_sha256'][:12]}); {a.manifest}")


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-vns")
    p.set_defaults(fn=cmd_pool)
    p = sub.add_parser("fetch")
    p.add_argument("--pool", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--delay", type=float, default=1.0)
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("freeze-frames")
    p.add_argument("--pool", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-vn", type=int, default=6)
    p.set_defaults(fn=cmd_freeze_frames)
    p = sub.add_parser("propose")
    p.add_argument("--frames", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_propose)
    p = sub.add_parser("sheets")
    p.add_argument("--frames", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--proposals", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--per-vn", type=int, default=6)
    p.set_defaults(fn=cmd_sheets)
    p = sub.add_parser("build")
    p.add_argument("--frames", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--proposals", required=True)
    p.add_argument("--decisions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--per-vn", type=int, default=6)
    p.set_defaults(fn=cmd_build)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

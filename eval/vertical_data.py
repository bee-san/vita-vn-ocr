#!/usr/bin/env python3
"""Vertical-text development and test sets (issue #99, docs/vertical.md): pages, reads, columns, review.

  vertical_data.py pages   --census SHOTS_V2 --classes CLASSES --out DIR      the page list (pages.jsonl)
  vertical_data.py manga   --zip VOL.zip --vol N --out DIR [--pages K]        NOBORU-style views of a manga volume
  vertical_data.py read    --pages DIR/pages.jsonl --out DIR [--procs 5]      MeikiOCR + PP-OCRv6 small, per column
  vertical_data.py sheets  --reads DIR/reads.jsonl --out BATCHDIR --mode pass1|pass2 [--ids IDS]
  vertical_data.py decide  --reads ... --batches B1,B2 --out DIR             decisions -> columns and regions

A *column* is one vertical text line: its main text only (furigana, the small ruby set to the right of a
column, excluded), read top to bottom. A *region* is what a player would select around a group of columns (a
speech balloon, a vertical dialogue box, the text block of an NVL page): the union of its columns plus a margin;
its text is its columns' texts in reading order, right to left.

Pages (one screen each, at the console's resolution: 960x544 Vita, 480x272 PSP; other sizes scaled to fit the
Vita screen as devnative/paths.load_screen does):
  vndb     VNDB screenshots of Vita / PSP / PC releases that the census (eval/vertical_count.py) found vertical
           pages on, never of a reserved.json test VN or test session or of an eval screenshot
  owner    the owner's console captures (private: never committed, never sent to any third party)
  manga    pages of a manga volume shown as NOBORU (a PS Vita manga reader) shows them on the 960x544 screen in
           its fit-to-width mode: the page scaled to the screen's width, cut into 544-row windows (noboru_views)

Reads (offline, teacher/.venv, one heavy.sh slot): MeikiOCR's detector on the page; each detector box flagged
vertical is refined to the ink of its main text (column_box below) and read by MeikiOCR's vertical recognizer on
the padded column crop ("meiki"), by MeikiOCR on its own box ("meikid", the page read) and by PP-OCRv6 small on the
crop turned 90 degrees anticlockwise (the column's top to the left, as PaddleOCR turns tall crops: "v6s").
Horizontal detector boxes are kept as ignore areas.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts" / "vertical"
PRIVATE = HERD / "shared" / "private" / "vertical"
DUMP = HERD / "shared" / "artifacts" / "realplus" / "vndb" / "dump" / "db"
SCREEN = {"vita": (960, 544), "psp": (480, 272)}
PAD = {"vita": 4, "psp": 2}
JP = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u3005]")
UA = "vita-vn-ocr-vertical/0.1 (OCR evaluation, low volume; github.com/bee-san/vita-vn-ocr)"

# Series clusters of the console VNs with vertical dialogue (issue #99's census) and their split. A cluster is one
# series (VNDB's direct relations); the split is by cluster, so no VN of a test cluster is in development.
# Developers and engines do cross it (Rejet, Otomate: docs/vertical.md, "Limits and disclosures").
CONSOLE_CLUSTERS = {
    "harukanaru": (["v1794", "v3259", "v3260", "v3261", "v3262", "v3263"], "dev"),
    "genrou": (["v11109"], "dev"),
    "gekka-ryouran": (["v7716"], "dev"),
    "bakumatsu-rock": (["v15361"], "test"),
    "renai-banchou": (["v7912", "v11309"], "test"),
    "yunohana": (["v17343", "v19298"], "test"),
    "trick-logic": (["v11713", "v12957"], "test"),
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_rows(name):
    with open(DUMP / name, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def reserved():
    return json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text())


def eval_images():
    """URL ids of every screenshot in eval's sets (vita-test, psp-test, linefind); the vertical sets' own manifests
    (eval/manifests/vertical-*.jsonl, built by this tool) and the textspot sets' (textspot-*.jsonl, built after
    them) are left out, so a rerun selects the same pages."""
    ids = set()
    files = [f for f in (ROOT / "eval" / "manifests").glob("*.jsonl")
             if not f.name.startswith(("vertical-", "textspot-"))]
    files += list((ROOT / "eval" / "linefind" / "manifests").glob("*.jsonl"))
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.search(r"/(\d+)\.(?:jpg|png)", line)
            if m:
                ids.add(m.group(1))
    return ids


def vn_clusters(vns):
    """Union-find of VNs over VNDB relations (any relation), restricted to `vns`: series clusters."""
    parent = {v: v for v in vns}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for r in dump_rows("vn_relations"):
        a, b = r[0], r[1]
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {v: find(v) for v in vns}


def stable(key, seed="vertical-20261018"):
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16)


# ----------------------------------------------------------------------------- pages

def cmd_pages(args):
    census = json.load(open(args.census))
    classes = {c["shot"]: c["class"] for c in json.load(open(args.classes))}
    res = reserved()
    test_vns, sessions = set(res["test"]), set(res["sessions"])
    evimg = eval_images()
    pages, seen_px = [], set()
    vn2cluster = {v: (c, s) for c, (vs, s) in CONSOLE_CLUSTERS.items() for v in vs}

    def refuse(s):
        if s["vn"] in test_vns:
            return "test-vn"
        if s.get("release") in sessions:
            return "test-session"
        m = re.search(r"/(\d+)\.(?:jpg|png)", s.get("url") or "")
        if m and m.group(1) in evimg:
            return "eval-image"
        return None

    def add(s, kind, origin, cluster, split, extra=None):
        import cv2
        img = cv2.imread(s["path"], cv2.IMREAD_COLOR)
        if img is None:
            return
        px = sha256_bytes(np.ascontiguousarray(img).tobytes())
        if px in seen_px:                    # the same picture under two VNs (v3260 / v3261)
            return
        seen_px.add(px)
        row = {"page": s["shot"] if origin != "owner" else s["shot"], "origin": origin, "vn": s["vn"], "kind": kind,
               "url": s.get("url"), "sha256": sha256_file(s["path"]), "path": s["path"], "release": s.get("release"),
               "dims": [int(img.shape[1]), int(img.shape[0])], "cluster": cluster, "split": split,
               "plat": s["plat"]}
        row.update(extra or {})
        pages.append(row)

    refused = {}
    # console pages with vertical dialogue (the census's classes, checked by eye in issue #99)
    for s in census:
        if s["plat"] not in ("psv", "psp") or not s.get("vpage") or not s.get("path"):
            continue
        why = refuse(s)
        if why:
            refused[s["shot"]] = why
            continue
        if classes.get(s["shot"], "ui") == "ui" or s["vn"] not in vn2cluster:
            continue
        cl, sp = vn2cluster[s["vn"]]
        add(s, "vita" if s["plat"] == "psv" else "psp", s["origin"], cl, sp)
    # every other owner frame of a vertical cluster with any vertical line (Trick x Logic's NVL scenes)
    for s in census:
        if s["origin"] == "owner" and s["vn"] in vn2cluster and s.get("vl") and s.get("path") and not s.get("vpage"):
            cl, sp = vn2cluster[s["vn"]]
            add(s, "vita" if s["plat"] == "psv" else "psp", "owner", cl, sp)
    # PC pages read at the Vita size: one page per VN, VNs grouped into series clusters, clusters split
    pc = [s for s in census if s["plat"] == "pc" and s.get("vpage") and s.get("path") and os.path.exists(s["path"])]
    pc = [s for s in pc if not refuse(s) and s["vn"] not in vn2cluster]
    clusters = vn_clusters({s["vn"] for s in pc})
    by_cluster = {}
    for s in pc:
        by_cluster.setdefault(clusters[s["vn"]], []).append(s)
    order = sorted(by_cluster, key=lambda c: stable(c))
    n_dev = n_test = 0
    for c in order:
        split = "test" if stable(c, "split") % 5 < 2 else "dev"
        if split == "dev" and n_dev >= args.pc_dev or split == "test" and n_test >= args.pc_test:
            continue
        cand = sorted(by_cluster[c], key=lambda s: (-len(s["vl"]), stable(s["shot"])))
        s = cand[0]
        add(s, "vita", "vndb", f"pc:{c}", split)
        if split == "dev":
            n_dev += 1
        else:
            n_test += 1
    out = Path(args.out)
    write_jsonl(out / "pages.jsonl", [p for p in pages if p["origin"] != "owner"])
    PRIVATE.mkdir(parents=True, exist_ok=True)
    write_jsonl(PRIVATE / "pages_owner.jsonl", [p for p in pages if p["origin"] == "owner"])
    json.dump({"refused": refused}, open(out / "pages_refused.json", "w"), indent=1)
    from collections import Counter
    print(Counter((p["split"], p["origin"], p["kind"]) for p in pages))
    print("refused", Counter(refused.values()))


def cmd_train_pages(args):
    """Training-side pages: every census screenshot with a vertical line whose VN shares no VNDB relation
    component with a development or test page's VN, a reserved.json test or dev_sample VN, or a console cluster;
    never a test session's or an eval screenshot. Owner frames go to the private list."""
    census = json.load(open(args.census))
    res = reserved()
    banned_vn = set(res["test"]) | set(res["dev_sample"])
    sessions, evimg = set(res["sessions"]), eval_images()
    held = {p["vn"] for p in jsonl(Path(args.dev_pages))}
    for p in args.extra_pages.split(",") if args.extra_pages else []:
        held |= {r["vn"] for r in jsonl(p)}
    held |= {v for vs, _ in CONSOLE_CLUSTERS.values() for v in vs}
    vns = {s["vn"] for s in census} | held
    comp = vn_clusters(vns)
    held_roots = {comp[v] for v in held}
    pages, owner, why = [], [], {}
    seen = set()
    for s in census:
        if not s.get("vl") or not s.get("path") or not os.path.exists(s["path"]):
            continue
        m = re.search(r"/(\d+)\.(?:jpg|png)", s.get("url") or "")
        reason = ("reserved" if s["vn"] in banned_vn else "held-cluster" if comp[s["vn"]] in held_roots else
                  "session" if s.get("release") in sessions else "eval-image" if m and m.group(1) in evimg else None)
        if reason:
            why[reason] = why.get(reason, 0) + 1
            continue
        if s["shot"] in seen:
            continue
        seen.add(s["shot"])
        kind = "psp" if s["plat"] == "psp" else "vita"
        row = {"page": s["shot"], "origin": s["origin"], "vn": s["vn"], "kind": kind, "url": s.get("url"),
               "sha256": sha256_file(s["path"]), "path": s["path"], "release": s.get("release"),
               "dims": s.get("dims"), "cluster": f"train:{comp[s['vn']]}", "split": "train", "plat": s["plat"]}
        (owner if s["origin"] == "owner" else pages).append(row)
    write_jsonl(Path(args.out) / "pages_train.jsonl", pages)
    write_jsonl(PRIVATE / "pages_train_owner.jsonl", owner)
    from collections import Counter
    print(len(pages), "vndb pages,", len(owner), "owner pages;", Counter(p["plat"] for p in pages), "refused", why)


# ----------------------------------------------------------------------------- manga views

def noboru_views(page_rgb):
    """NOBORU-like 960x544 framebuffers of one manga page in its fit-to-width mode: the page scaled (area / bilinear,
    as the GPU's filtering) to the screen's width and cut into consecutive 544-row windows from the top, the last one
    filled with black below the page: [(name, image, (scale, ox, oy))], where a page pixel (px, py) lands at
    (px * scale + ox, py * scale + oy) of the window. Windows do not overlap, so no text is in two of them (a
    balloon a window edge cuts is cut in both and is left out at review)."""
    import cv2
    W, H = SCREEN["vita"]
    h, w = page_rgb.shape[:2]
    s = W / w
    ph = int(round(h * s))
    big = cv2.resize(page_rgb, (W, ph), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = []
    for i, oy in enumerate(range(0, ph, H)):
        win = np.zeros((H, W, 3), np.uint8)
        part = big[oy:oy + H]
        win[:part.shape[0]] = part
        if part.shape[0] < H // 4:
            continue
        out.append((f"w{i}", win, (s, 0, -oy)))
    return out


def cmd_manga(args):
    """Extracts the volume's page images, picks every --every-th page from --first, writes its NOBORU views."""
    import cv2
    z = zipfile.ZipFile(args.zip)
    names = sorted(n for n in z.namelist() if re.search(r"\.(jpe?g|png)$", n, re.I))
    out = Path(args.out)
    (out / "views").mkdir(parents=True, exist_ok=True)
    zsha = sha256_file(args.zip)
    rows = []
    picked = names[args.first::args.every][:args.pages]
    for name in picked:
        data = z.read(name)
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, 2)
        stem = re.sub(r"[^0-9A-Za-z]+", "_", Path(name).stem)
        for view, img, (s, ox, oy) in noboru_views(rgb):
            pid = f"bj{args.vol:02d}_{stem}_{view}"
            path = out / "views" / f"{pid}.png"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows.append({"page": pid, "origin": "manga", "vn": f"bj{args.vol:02d}", "kind": "vita", "url": None,
                         "sha256": sha256_file(path), "path": str(path), "release": None, "dims": [960, 544],
                         "cluster": "blackjack", "split": args.split, "plat": "manga",
                         "manga": {"zip_sha256": zsha, "member": name, "member_sha256": sha256_bytes(data),
                                   "view": view, "scale": s, "ox": ox, "oy": oy,
                                   "page_dims": [rgb.shape[1], rgb.shape[0]]}})
    write_jsonl(out / f"pages_bj{args.vol:02d}.jsonl", rows)
    print(len(picked), "pages,", len(rows), "views")


# ----------------------------------------------------------------------------- columns

def to_grey(rgb):
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def column_box(grey, det, pad):
    """The main-text ink box of a vertical detector box, or None.

    Within the detector box widened by 3 px, a pixel is ink when it differs from the median of the window's border
    ring by more than max(28, 0.4 x its 99th percentile) (teacher/geometry.refine's rule, either polarity). The
    column profile (ink pixels per x, over the detector box's rows) splits into runs at gaps of at least one empty
    column; the heaviest run is the main text; a run to its right narrower than 0.6 of it is furigana and one to
    its left is a neighbouring column: both are left out. The rows are those holding ink within the main run.
    Returns [x0, y0, x1, y1] (ink, exclusive) and the padded crop box, clipped to the image."""
    H, W = grey.shape
    x0, y0, x1, y1 = det
    wx0, wy0, wx1, wy1 = max(0, x0 - 3), max(0, y0 - 3), min(W, x1 + 3), min(H, y1 + 3)
    g = grey[wy0:wy1, wx0:wx1].astype(np.float32)
    if g.shape[0] < 6 or g.shape[1] < 4:
        return None
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    d = np.abs(g - float(np.median(ring)))
    t = max(28.0, 0.4 * float(np.percentile(d, 99)))
    m = d > t
    prof = m.sum(0)
    runs, s = [], None
    for x in range(len(prof) + 1):
        on = x < len(prof) and prof[x] > 0
        if on and s is None:
            s = x
        elif not on and s is not None:
            runs.append([s, x, int(prof[s:x].sum())])
            s = None
    if not runs:
        return None
    k = max(range(len(runs)), key=lambda i: runs[i][2])
    a, b = runs[k][0], runs[k][1]
    # merge runs that are part of the main text: gaps of one or two columns inside a glyph, wide enough runs
    for j in range(k + 1, len(runs)):
        if runs[j][0] - b <= 2 and (runs[j][1] - runs[j][0]) >= 0.6 * (b - a):
            b = runs[j][1]
        else:
            break
    for j in range(k - 1, -1, -1):
        if a - runs[j][1] <= 2 and (runs[j][1] - runs[j][0]) >= 0.3 * (b - a):
            a = runs[j][0]
        else:
            break
    rows = np.nonzero(m[:, a:b].any(1))[0]
    if not len(rows):
        return None
    ink = [wx0 + a, wy0 + int(rows[0]), wx0 + b, wy0 + int(rows[-1]) + 1]
    crop = [max(0, ink[0] - pad), max(0, ink[1] - pad), min(W, ink[2] + pad), min(H, ink[3] + pad)]
    return ink, crop


def load_page(row):
    """RGB of a page at its console resolution (devnative/paths.load_screen's scaling)."""
    import cv2
    img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    W, H = SCREEN[row["kind"]]
    if (w, h) == (W, H):
        return rgb
    s = min(W / w, H / h)
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


_ENG = {}


def _init():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from teacher import engines
    _ENG.update(engines.load(["meiki", "v6s"], n_threads=1))


def read_page(row):
    import cv2
    rgb = load_page(row)
    if rgb is None:
        return {"page": row["page"], "error": "unreadable"}
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    grey = to_grey(rgb)
    H, W = grey.shape
    t0 = time.perf_counter()
    dets = _ENG["meiki"].ocr(bgr)
    cols, horiz = [], []
    for k, d in enumerate(dets):
        x0, y0, x1, y1 = d["det"]
        if not d["vertical"]:
            horiz.append({"det": d["det"], "text": d["text"]})
            continue
        cb = column_box(grey, d["det"], PAD[row["kind"]])
        if cb is None:
            continue
        ink, crop = cb
        cx0, cy0, cx1, cy1 = crop
        c = bgr[cy0:cy1, cx0:cx1]
        rec = {"k": k, "det": d["det"], "det_conf": d["det_conf"], "ink": ink, "crop": crop,
               "reads": {"meikid": d["text"]}}
        if c.shape[0] > c.shape[1] and c.size:
            t, mean, low, _ = _ENG["meiki"](np.ascontiguousarray(c))
            rec["reads"]["meiki"] = t
            v = _ENG["v6s"](np.ascontiguousarray(np.rot90(c)))
            rec["reads"]["v6s"] = v[0]
            rec["v6s_conf"] = [v[1], v[2]]
        cols.append(rec)
    return {**{k: row[k] for k in ("page", "origin", "vn", "kind", "url", "sha256", "release", "cluster", "split",
                                   "plat")},
            "path": row["path"], "dims": [W, H], "manga": row.get("manga"), "columns": cols, "horizontal": horiz,
            "secs": round(time.perf_counter() - t0, 3)}


def cmd_read(args):
    pages = []
    for p in args.pages.split(","):
        pages += jsonl(p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "reads.jsonl"
    done = {r["page"] for r in jsonl(path)} if path.exists() else set()
    todo = [p for p in pages if p["page"] not in done]
    print(f"{len(pages)} pages, {len(done)} read, {len(todo)} to read", flush=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo, mp.get_context("spawn").Pool(args.procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(read_page, todo, chunksize=1)):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("READ_DONE", flush=True)


# ----------------------------------------------------------------------------- review sheets

def canon(t):
    """Candidate text in the label form: NFKC-ish full-width punctuation as the benchmark writes it."""
    import unicodedata
    t = unicodedata.normalize("NFKC", t or "").replace(" ", "")
    t = t.replace("!", "！").replace("?", "？").replace("~", "～").replace("...", "…")
    t = re.sub(r"・{2,3}", "…", t)
    return t


def candidates(col):
    """Distinct readings, most supported first: [(text, [engines])]."""
    order = ("meiki", "meikid", "v6s")
    sup = {}
    for e in order:
        t = canon((col.get("reads") or {}).get(e))
        if t:
            sup.setdefault(t, []).append(e)
    return sorted(sup.items(), key=lambda kv: (-len(kv[1]), min(order.index(e) for e in kv[1])))


def col_id(page, k):
    return f"{page}_c{k}"


def cmd_sheets(args):
    """Review sheets: each entry is one column, shown upright with a little context around it (zoomed by an integer
    factor to about 40 px of column width), entries side by side in rows; the candidates are in Sxxx.json."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(HERD / "shared" / "data" / "fonts" / "bizudgothic" / "BIZUDGothic-Regular.ttf"), 14)
    rows = jsonl(args.reads)
    want = set(json.load(open(args.ids))) if args.ids else None
    extra = json.load(open(args.extra)) if args.extra else {}
    ents = []
    for r in rows:
        if r.get("error") or (args.split and r["split"] != args.split):
            continue
        if r["origin"] == "owner" and not args.private:
            continue
        img = None
        for c in r["columns"]:
            cid = col_id(r["page"], c["k"])
            if want is not None and cid not in want:
                continue
            if img is None:
                img = load_page(r)
            ents.append((cid, r, c, img))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    SHEET, ROWH = 1280, 600
    k, sheet_no = 0, 0
    while k < len(ents):
        sheet_no += 1
        canvas = Image.new("RGB", (SHEET, SHEET), "white")
        dr = ImageDraw.Draw(canvas)
        meta = []
        x = y = 0
        rowh = 0
        while k < len(ents):
            cid, r, c, img = ents[k]
            x0, y0, x1, y1 = c["crop"]
            w, h = x1 - x0, y1 - y0
            z = max(1, min(4, int(round(40 / max(1, w)))))
            while h * z > ROWH - 20 and z > 1:
                z -= 1
            cx0, cy0 = max(0, x0 - w), max(0, y0 - w // 2)
            cx1, cy1 = min(img.shape[1], x1 + w), min(img.shape[0], y1 + w // 2)
            ctx = img[cy0:cy1, cx0:cx1]
            ctx = cv2.resize(ctx, (ctx.shape[1] * z, ctx.shape[0] * z), interpolation=cv2.INTER_NEAREST)
            if ctx.shape[0] > ROWH - 20:
                ctx = cv2.resize(ctx, (int(ctx.shape[1] * (ROWH - 20) / ctx.shape[0]), ROWH - 20),
                                 interpolation=cv2.INTER_AREA)
                z2 = (ROWH - 20) / (cy1 - cy0)
            else:
                z2 = z
            ew, eh = ctx.shape[1] + 8, ctx.shape[0] + 20
            if x + ew > SHEET:
                x, y = 0, y + rowh
                rowh = 0
            if y + eh > SHEET:
                break
            n = len(meta) + 1
            canvas.paste(Image.fromarray(ctx), (x + 4, y + 18))
            fx0, fy0 = x + 4 + (x0 - cx0) * z2, y + 18 + (y0 - cy0) * z2
            dr.rectangle([fx0, fy0, fx0 + w * z2 - 1, fy0 + h * z2 - 1], outline=(255, 0, 0), width=1)
            dr.text((x + 4, y + 1), str(n), fill=(0, 0, 0), font=font)
            cands = candidates(c)
            for t in extra.get(cid, []):        # the earlier reviewers' readings, unattributed
                t = canon(t)
                if t and all(t != x for x, _ in cands):
                    cands.append((t, ["review"]))
            if args.mode != "pass1":
                rnd = random.Random(cid)
                rnd.shuffle(cands)
            meta.append({"n": n, "id": cid, "candidates": [{"c": i + 1, "text": t}
                                                          | ({"engines": e} if args.mode == "pass1" else {})
                                                          for i, (t, e) in enumerate(cands)]})
            x += ew
            rowh = max(rowh, eh)
            k += 1
        name = f"S{sheet_no:03d}"
        canvas.save(out / f"{name}.png")
        json.dump({"sheet": name, "entries": meta}, open(out / f"{name}.json", "w"), ensure_ascii=False, indent=1)
    (out / "README.md").write_text(README.format(mode=args.mode, n=len(ents), sheets=sheet_no), encoding="utf-8")
    print(len(ents), "columns,", sheet_no, "sheets ->", out)


README = """# Vertical-text review batch ({mode}, {n} columns, {sheets} sheets)

Each sheet `Sxxx.png` (1280 x 1280 px) shows numbered entries side by side. An entry is one VERTICAL text column
of a PS Vita / PSP / PC visual-novel screenshot or a manga page, shown upright (read it top to bottom). The red
frame marks the column that gets the label; around it is a little of the screen for context (the neighbouring
columns, if any, are NOT part of the label). `Sxxx.json` lists each entry's numbered candidate readings (from OCR
engines; any of them, or none, may be right), so you can copy text exactly.

Decide what the framed column says. A little furigana or the edge of a neighbouring column inside the frame is
fine: the label is the framed column's main text all the same. Label conventions (the benchmark's):
- the main text of the column only: never furigana (the small ruby set to the RIGHT of a vertical column), page
  cursors, icons or button glyphs;
- write vertical forms as their ordinary characters: a vertical bar for a long vowel is ー, a vertical dash is ―,
  vertical dots ︙ are … (one … per cell of dots), ﹁﹂﹃﹄ are 「」『』, ︵︶ are （）, a vertical wave is ～;
- keep 「」『』（）, full-width ！ ？ ～, no spaces; a sideways Latin word or number as plain ASCII; two or three
  half-width digits set side by side in one cell (tate-chu-yoko) as those digits;
- small kana (ゃ っ ァ), dakuten / handakuten (は ば ぱ), ー versus 一 versus ｜, and similar-looking kanji matter:
  look at each one;
- never guess a character you cannot see; mark the entry unsure instead.

Write `decisions.jsonl` in this directory, one JSON object per entry, for every entry of every sheet:

```
{{"sheet": "S001", "n": 1, "pick": 2}}             candidate 2 is exactly right
{{"sheet": "S001", "n": 2, "text": "「ええと……」"}}    the correct text (when no candidate is exactly right)
{{"sheet": "S001", "n": 3, "unsure": true}}        you cannot settle it from the image
{{"sheet": "S001", "n": 4, "skip": "ui"}}          not dialogue / narration: ui (menus, buttons, system text,
                                                   titles, logos, credits, signs) | sfx (hand-drawn manga sound
                                                   effects or lettering that is part of the art) | cut (the frame
                                                   cuts off part of the column's main text, or holds most of two
                                                   columns) | furigana (the frame holds only furigana) | bad (not
                                                   one readable vertical column)
```

Add `"name": true` when the column is a speaker name. Optional `"note"`. Do not edit anything else.
View each sheet image once, one image per tool call; append decisions after each sheet (resume-safe).
When every entry has a decision, the batch is done.
"""


# ----------------------------------------------------------------------------- decisions

def cmd_decide(args):
    """Merges pass-1 and pass-2 decisions into labels. A column is labelled when pass 1 decided a text and either
    pass 2 (an independent reviewer, candidates shuffled and unattributed) chose the same text, or pass 1's text
    equals an engine reading that pass 2 also picked; disagreements go to --resolve (a third look) or are left
    out as unsure."""
    rows = jsonl(args.reads)
    by_id = {}
    for r in rows:
        for c in r.get("columns", []):
            by_id[col_id(r["page"], c["k"])] = (r, c)

    def load(batches):
        dec = {}
        for b in batches.split(",") if batches else []:
            ds = []
            for f in sorted(Path(b).glob("decisions*.jsonl")):
                ds += jsonl(f)
            for d in ds:
                sh = json.load(open(Path(b) / f"{d['sheet']}.json"))
                e = next(x for x in sh["entries"] if x["n"] == d["n"])
                cand = {c["c"]: c["text"] for c in e["candidates"]}
                if "pick" in d:
                    t = cand[d["pick"]]
                elif "text" in d:
                    t = d["text"]
                else:
                    t = None
                dec[e["id"]] = {"text": canon(t) if t is not None else None, "skip": d.get("skip"),
                                "unsure": bool(d.get("unsure")), "name": bool(d.get("name"))}
        return dec
    p1, p2, p3 = load(args.pass1), load(args.pass2), load(args.resolve)
    out, stats = [], {}
    for cid, (r, c) in by_id.items():
        a = p1.get(cid)
        if a is None:
            continue
        b, z = p2.get(cid), p3.get(cid)
        if z is not None:                     # a third look settles a disagreement
            ok = z["text"] and not z["skip"] and not z["unsure"]
            final = {"text": z["text"]} if ok else {"skip": z["skip"] or "unsure"}
        elif a["skip"] or a["unsure"]:
            final = {"skip": a["skip"] or "unsure"}
        elif b is not None and b["text"] == a["text"]:
            final = {"text": a["text"]}
        elif b is None and args.single:
            final = {"text": a["text"]}
        else:
            final = {"skip": "disagree"}
        why = final.get("skip") or "label"
        stats[why] = stats.get(why, 0) + 1
        out.append({"id": cid, "page": r["page"], **final, "name": a["name"]})
    write_jsonl(args.out, out)
    if args.queue:
        q = [cid for cid in by_id if cid in p1 and cid not in p3 and not (p1[cid]["skip"] or p1[cid]["unsure"])
             and (p2.get(cid) is None or p2[cid]["text"] != p1[cid]["text"])]
        q += [cid for cid in by_id if cid in p1 and cid not in p3 and (p1[cid]["skip"] or p1[cid]["unsure"]) and
              cid in p2 and p2[cid]["text"] and not p2[cid]["skip"] and not p2[cid]["unsure"]]
        json.dump(sorted(set(q)), open(args.queue, "w"))
        print(len(set(q)), "for a third look ->", args.queue)
    print(stats)


def cmd_shard(args):
    """Training crops of real vertical columns (training-side pages only): each column's crop (column_box's padded
    box) in the console's polarity, turned 90 degrees anticlockwise, with its label: "agree" when MeikiOCR's page
    read and PP-OCRv6 small's read of the turned crop are the same text (two engine families), "ps" when they are
    not but PP-OCRv6 small's lowest character confidence is at least --conf. --half also writes each crop at half
    scale (INTER_AREA), as PSP-sized text. train2's shard format."""
    import cv2
    sys.path.insert(0, str(ROOT / "synth"))
    from common import line_dark
    crops, labels, metas = [], [], []
    for path in args.reads.split(","):
        for r in jsonl(path):
            if r.get("error") or r.get("split") != "train":
                continue
            todo = []
            for c in r.get("columns", []):
                a, b = canon(c["reads"].get("meikid")), canon(c["reads"].get("v6s"))
                conf = (c.get("v6s_conf") or [0, 0])[1]
                if args.select == "agree":
                    ok = a and a == b and len(a) >= 2
                else:
                    ok = b and len(b) >= 2 and a != b and conf >= args.conf
                if ok:
                    todo.append((c, b))
            if not todo:
                continue
            rgb = load_page(r)
            if rgb is None:
                continue
            g = to_grey(rgb)
            for c, text in todo:
                x0, y0, x1, y1 = c["crop"]
                if x1 - x0 < 4 or y1 - y0 < 8:
                    continue
                for sc in ((1.0, 0.5) if args.half else (1.0,)):
                    size = (max(1, round(g.shape[1] * sc)), max(1, round(g.shape[0] * sc)))
                    gg = g if sc == 1.0 else cv2.resize(g, size, interpolation=cv2.INTER_AREA)
                    bx = [int(round(v * sc)) for v in (x0, y0, x1, y1)]
                    if sc != 1.0:            # the PSP pad: 2 px around the halved ink box
                        ix0, iy0, ix1, iy1 = [int(round(v * sc)) for v in c["ink"]]
                        bx = [max(0, ix0 - 2), max(0, iy0 - 2), min(gg.shape[1], ix1 + 2), min(gg.shape[0], iy1 + 2)]
                    w, h = bx[2] - bx[0], bx[3] - bx[1]
                    if w < 4 or h < 8:
                        continue
                    crop = gg[bx[1]:bx[3], bx[0]:bx[2]]
                    dark = line_dark(gg, (bx[0], bx[1], w, h))
                    if dark:
                        crop = 255 - crop
                    crops.append(np.ascontiguousarray(np.rot90(crop, 1)))
                    labels.append(text)
                    metas.append({"page": r["page"], "vn": r["vn"], "k": c["k"], "scale": sc, "dark": int(dark),
                                  "select": args.select})
    shapes = np.array([x.shape for x in crops], np.int32)
    offs = np.concatenate([[0], np.cumsum([x.size for x in crops])]).astype(np.int64)   # n + 1 entries
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, data=np.concatenate([x.ravel() for x in crops]), offsets=offs, shapes=shapes,
             labels=json.dumps(labels, ensure_ascii=False), meta=json.dumps({"items": metas}, ensure_ascii=False))
    print(len(crops), "crops ->", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pages")
    p.add_argument("--census", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pc-dev", type=int, default=40)
    p.add_argument("--pc-test", type=int, default=25)
    p = sub.add_parser("train-pages")
    p.add_argument("--census", required=True)
    p.add_argument("--dev-pages", required=True)
    p.add_argument("--extra-pages", default="")
    p.add_argument("--out", required=True)
    p = sub.add_parser("manga")
    p.add_argument("--zip", required=True)
    p.add_argument("--vol", type=int, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--first", type=int, default=6)
    p.add_argument("--every", type=int, default=6)
    p.add_argument("--pages", type=int, default=24)
    p = sub.add_parser("read")
    p.add_argument("--pages", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("sheets")
    p.add_argument("--reads", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="pass1")
    p.add_argument("--ids")
    p.add_argument("--extra", help="JSON {column id: [more candidate texts]} (a third look's earlier readings)")
    p.add_argument("--split")
    p.add_argument("--private", action="store_true")
    p = sub.add_parser("shard")
    p.add_argument("--reads", required=True)
    p.add_argument("--select", choices=("agree", "ps"), required=True)
    p.add_argument("--conf", type=float, default=0.9)
    p.add_argument("--half", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--reads", required=True)
    p.add_argument("--pass1", required=True)
    p.add_argument("--pass2", default="")
    p.add_argument("--resolve", default="")
    p.add_argument("--single", action="store_true")
    p.add_argument("--queue", help="write the ids that need a third look (pass 1 and 2 disagree) here")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"pages": cmd_pages, "train-pages": cmd_train_pages, "manga": cmd_manga, "read": cmd_read,
     "sheets": cmd_sheets, "shard": cmd_shard, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    main()

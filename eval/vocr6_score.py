#!/usr/bin/env python3
"""VOCR6 T1 scoring (issue #125): one recognizer blob read the way Rubi reads, on hval and the unseen-game dev sets.

  vocr6_score.py hval-plan --pools POOLS --pagepool PAGEPOOL --pages PAGES --out HVAL_PLAN.pkl
  vocr6_score.py read --plan PLAN.pkl --blob BLOB.vocr --lm LM.vlm --out DUMP.json [--workers 5]
  vocr6_score.py score --name RUN --blob BLOB.vocr --hval-plan HP --hval-dump HD --dev-plan DP --dev-units DU \\
                       --dev-dump DD --out SCORE.json
  vocr6_score.py all --name RUN --blob BLOB.vocr --protocol eval/vocr6_protocol.json --out DIR [--workers 5]

The shipped form (Rubi main bd130a8: core/vocr_ocr.c, game_ocr/main.c, core/textnorm.c), per line: the C runtime in
W8I, greedy with the top 8 kept, the counter re-decode (6000 / 4000), then beam 8 with the 1 MB character LM
(alpha 500, beta 3500, gate 8000, EOS, bracket 8000) when the greedy text is not empty, then core/textnorm.c (the real
C code, PROTOCOL["rubi_norm"]) and the Shell's non-Japanese line filter. The confidence cascade (a doubtful line read
again by the next tier) needs two tiers of one recipe, so screening scores each model alone; stage 4 adds it.

Plans hold every crop a score reads, built once (the finder does not depend on the recognizer):
  dev   reports/vocr5-analysis's plan (#119: units.pkl + plan.pkl): teacher_val's and devnative's development crops
        (tv, dn) and the whole-dialogue units' finder crops (wd: the C finder VOCR_LINES_TEXTBOX_RECOVERY, the
        refinement on PSP units), in the console view (PSP at Adrenaline's 2x bilinear; Vita as is)
  hval  the hooked corpus' screening block (vocr6-ingest's pools and page pool): every hval dialogue and name line as
        a clean crop (box + pad; PSP cut from the 2x page, Rubi's view of a PSP game) and every complete hval page's
        whole-dialogue regions (base, +4, +8 px) through the same C finder, in the same view
Scores (per row, so selection can resample games): dialogue CER, eval/metrics.norm + Levenshtein for lines and
train/general_vn_textbox_guard.transcript_score for regions. Rows carry their game (hval: title id; dev: VNDB id).
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in ("eval", "train", "runtime", "synth"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))
from metrics import lev, norm  # noqa: E402

FINDER_RECOVERY = 3          # vocr_lines.h VOCR_LINES_TEXTBOX_RECOVERY (Rubi's VJO_VOCR_FINDER)
REFINE_ALL = 0x1F            # VOCR_REFINE_ALL: Rubi's ocr_refine_lines = psp
LM_OPTS = dict(beam=8, topk=8, alpha=500, beta=3500, gate=8000, flags=2, bracket=8000)
REDECODE = (6000, 4000)
TOPK = 8
PUNCT = set("「」『』。、！？!?…―（）")
VARIANTS = ("base", "expanded-4", "expanded-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def namelike(label):
    """eval/synth3_ablation.strata's name-like tag (VOCR5's dialogue rule for teacher_val)."""
    t = norm(label)
    return len(t) <= 6 and not set(t) & PUNCT


def has_japanese(t):
    """Rubi's non_japanese_filter = lines: a line without hiragana, katakana or CJK is dropped."""
    return any(0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FAF for c in t)


def load_npz(path):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"])) if "meta" in z.files else [{} for _ in labels]
    out = []
    for i, (lab, m) in enumerate(zip(labels, meta)):
        h, w = (int(v) for v in shapes[i])
        out.append((m, lab, data[offs[i]:offs[i + 1]].reshape(h, w)))
    return out


# ---------------------------------------------------------------- the C finder (the console's, bit for bit)

def build_finder(out):
    """runtime/vocr_lines.c + vocr_lines_refine.c with their batch driver, as reports/vocr5-analysis built it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    drv = out / "vocr_lines_refine_test"
    flags = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror")
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
    subprocess.run(["gcc", *flags, "-o", str(drv), "runtime/vocr_lines.c", "runtime/vocr_lines_refine.c",
                    "runtime/vocr_lines_refine_test.c"], check=True, cwd=ROOT, env=env)
    return drv


def c_find(driver, images, finder, flags_of):
    """[(boxes [(x, y, w, h)], dark flags, rc)] per grey image."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img, fl in zip(images, flags_of):
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape
            f.write(struct.pack("<6I", w, h, 1, finder, w, fl))
            f.write(img.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(driver), path], check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(out) != len(images):
        raise SystemExit(f"finder driver returned {len(out)} records for {len(images)} images")
    res = []
    for line in out:
        v = [int(t) for t in line.split()]
        rc, n = v[0], v[2]
        if rc != 0:
            res.append(([], [], rc))
            continue
        res.append(([tuple(v[5 + 5 * i:9 + 5 * i]) for i in range(n)], [v[9 + 5 * i] for i in range(n)], 0))
    return res


# ---------------------------------------------------------------- hval plan

def covers(b, lb):
    """Finder box b = (x, y, w, h) covers reference line box lb = [x0, y0, x1, y1]: vertical overlap of at least half
    the line's height and some horizontal overlap (reports/vocr5-analysis's rule)."""
    x, y, w, h = b
    ov = min(y + h, lb[3]) - max(y, lb[1])
    hx = min(x + w, lb[2]) - max(x, lb[0])
    return ov >= 0.5 * max(1, lb[3] - lb[1]) and hx > 0


def _console_page(rgb, platform):
    """The frame Rubi reads: a PSP game through Adrenaline is its 480x272 frame shown 2x bilinear on the Vita."""
    import cv2

    from common import to_gray
    if platform == "psp":
        rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return to_gray(rgb), (2 if platform == "psp" else 1)


def hval_plan(args):
    """Clean crops: POOLS/hval/{line,name}[_x2]/<title>.npz (the console view: PSP titles' _x2 files). Regions: the
    page pool's hval rows that are complete (every renderer box accounted for by a verified line), each whole-dialogue
    region variant cut from the console frame and run through the C finder."""
    import cv2
    pools, pagepool, pages = Path(args.pools), Path(args.pagepool), Path(args.pages)
    crops, meta, fixed, units = [], [], [], []

    def add(img, inv, m):
        crops.append((np.ascontiguousarray(img, np.uint8), bool(inv)))
        meta.append(m)
        return len(crops) - 1

    platform_of = {}
    for rowfile in sorted(pagepool.glob("*.jsonl")):
        for line in rowfile.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                platform_of[r["title_id"]] = r["platform"]
                break
    n_files = 0
    for kind in ("line", "name"):
        for title in sorted(platform_of):
            plat = platform_of[title]
            p = pools / "hval" / (f"{kind}_x2" if plat == "psp" else kind) / f"{title}.npz"
            if not p.exists():
                continue
            n_files += 1
            for m, lab, img in load_npz(p):
                if not lab:
                    continue
                c = add(img, False, {"set": "hval_clean", "kind": kind, "title": title, "id": m.get("id")})
                fixed.append({"crop": c, "id": m.get("id"), "title": title, "platform": plat, "kind": kind,
                              "ref": lab, "file": str(p)})
    driver = build_finder(Path(args.out).parent / "build")
    jobs, skipped = [], {"untrusted": 0, "vertical_region": 0, "no_dialogue": 0}
    complete_pages = set()
    for rowfile in sorted(pagepool.glob("*.jsonl")):
        title = rowfile.stem
        for line in rowfile.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "hval":
                continue
            # the page pool's regions keep every other box (an unlabelled nameplate, UI text, an unverified row)
            # out of the region or drop its group, so a page need not be complete; a page with a label in doubt
            # has no regions
            if r.get("untrusted"):
                skipped["untrusted"] += 1
                continue
            if r.get("complete"):
                complete_pages.add((title, r["page"]))
            by_id = {ln["id"]: ln for ln in r.get("lines") or []}
            for reg in r.get("regions") or []:
                lines = [by_id[i] for i in reg["lines"]]
                if any(ln.get("vertical") for ln in lines):
                    skipped["vertical_region"] += 1
                    continue
                if not any(ln.get("kind") == "line" for ln in lines):
                    skipped["no_dialogue"] += 1
                    continue
                jobs.append((title, r, reg, lines))
    by_page = {}
    for title, r, reg, lines in jobs:
        by_page.setdefault((title, r["page"]), []).append((r, reg, lines))
    imgs, info = [], []
    for (title, page), regs in sorted(by_page.items()):
        r0 = regs[0][0]
        png = pages / r0["image"] if (pages / r0["image"]).exists() else pages / title / "pages" / f"{page}.png"
        bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"cannot read {png}")
        if sha256(png) != r0["image_sha256"]:
            raise SystemExit(f"{png} differs from the page pool's sha256")
        grey, s = _console_page(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), r0["platform"])
        for r, reg, lines in regs:
            x0, y0, x1, y1 = (int(v) * s for v in reg["region"])
            imgs.append(np.ascontiguousarray(grey[y0:y1, x0:x1]))
            info.append((title, r, reg, lines, s))
    found = c_find(driver, imgs, FINDER_RECOVERY, [REFINE_ALL if i[1]["platform"] == "psp" else 0 for i in info])
    for img, (title, r, reg, lines, s), (boxes, dark, rc) in zip(imgs, info, found):
        variant = reg.get("variant", "base")
        x0, y0 = int(reg["region"][0]) * s, int(reg["region"][1]) * s
        ref = sorted(lines, key=lambda q: (q["box"][1], q["box"][0]))
        lboxes = [[q["box"][0] * s - x0, q["box"][1] * s - y0, (q["box"][0] + q["box"][2]) * s - x0,
                   (q["box"][1] + q["box"][3]) * s - y0] for q in ref]
        cov = [[li for li, lb in enumerate(lboxes) if covers(b, lb)] for b in boxes]
        hit = {li for c in cov for li in c}
        u = {"title": title, "platform": r["platform"], "page": r["page"], "variant": variant,
             "complete_page": (title, r["page"]) in complete_pages,
             "unit": f"{title}:{r['page']}:{'+'.join(reg['lines'])}:{variant}",
             "region": [int(v) * s for v in reg["region"]],
             "lines": [{"id": ln["id"], "text": ln["text"], "styles": [] if ln.get("kind") == "line" else ["name"],
                        "box": lb} for ln, lb in zip(ref, lboxes)],
             "cov": cov, "junk": sum(1 for c in cov if not c), "merged": sum(1 for c in cov if len(c) > 1),
             "missed": sum(1 for li in range(len(ref)) if li not in hit),
             "rc": rc, "boxes": [list(b) for b in boxes], "dark": list(dark), "finder": []}
        for bi, ((x, y, w, h), dk) in enumerate(zip(boxes, dark)):
            u["finder"].append(add(img[y:y + h, x:x + w], dk, {"set": "hval_page", "unit": u["unit"], "b": bi}))
        units.append(u)
    doc = {"kind": "vocr6-hval-plan", "built": now(), "pools": str(pools), "pagepool": str(pagepool),
           "pages": str(pages), "finder": FINDER_RECOVERY, "refine_psp": REFINE_ALL, "driver_sha256": sha256(driver),
           "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                       text=True).stdout.strip(), "skipped": skipped,
           "crops": crops, "meta": meta, "fixed": fixed, "units": units}
    with open(args.out, "wb") as f:
        pickle.dump(doc, f, protocol=4)
    print(f"hval plan written: crops {len(crops)} clean {len(fixed)} ({n_files} files) units {len(units)} "
          f"pages {len(by_page)} titles {len({u['title'] for u in units})} skipped {skipped}", flush=True)


# ---------------------------------------------------------------- reading

_W = {}


def _init(lib, blob, lm, head_gate, two_level):
    from pyvocr import Vocr, VocrLM, load_lib
    L = load_lib(lib)
    kw = {}
    if head_gate:
        kw["head_gate"] = head_gate
    if two_level:
        kw["two_level"] = True
    r = Vocr(blob, mode="w8i", max_width=1024, lib=L, **kw)
    _W["r"] = r
    _W["lm"] = VocrLM(r, lm, **LM_OPTS) if lm else None


def _read(job):
    i, img, inv = job
    r, lm = _W["r"], _W["lm"]
    o = r.run_topk(np.ascontiguousarray(img), TOPK, invert=bool(inv), blank=True)
    red, st = r.redecode(o, *REDECODE) if o["text"] else (o, {"edits": 0})
    beam = lm.decode(red["topv"], red["topc"], red["blank"]) if (lm is not None and o["text"]) else ""
    return i, o["text"], red["text"], beam, float(o["conf_min"]), float(o["conf_mean"]), int(o["topv"].shape[0])


def read(plan_path, blob, lm, out, workers=5, head_gate="", two_level=False):
    lib = ROOT / "runtime/build/libvocr.so"
    if not lib.exists():
        raise SystemExit("build the runtime first: make -C runtime build/libvocr.so")
    t0 = time.time()
    plan = pickle.load(open(plan_path, "rb"))
    jobs = [(i, img, inv) for i, (img, inv) in enumerate(plan["crops"])]
    res = [None] * len(jobs)
    with Pool(workers, initializer=_init, initargs=(str(lib), str(blob), lm, head_gate, two_level)) as pool:
        for k, r in enumerate(pool.imap_unordered(_read, jobs, chunksize=16)):
            res[r[0]] = r
            if k and k % 5000 == 0:
                print(f"read {k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    doc = {"kind": "vocr6-dump", "blob": str(blob), "blob_sha256": sha256(blob), "lm": lm,
           "lm_sha256": sha256(lm) if lm else None, "lm_opts": LM_OPTS, "redecode": list(REDECODE),
           "head_gate": head_gate or "off", "two_level": bool(two_level),
           "library_sha256": sha256(lib), "plan": str(plan_path), "plan_sha256": sha256(plan_path),
           "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                       text=True).stdout.strip(),
           "crops": [{"g": r[1], "rg": r[2], "b": r[3], "cmin": r[4], "cmean": r[5], "fr": r[6]} for r in res],
           "seconds": round(time.time() - t0, 1), "at": now()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(f"dumped: {out} {len(jobs)} crops {time.time() - t0:.0f}s", flush=True)
    return doc


# ---------------------------------------------------------------- texts and scores

class Norm:
    """core/textnorm.c (Rubi's real code) over many texts in one process."""

    def __init__(self, binary):
        self.binary = str(binary)

    def __call__(self, texts):
        inp = "".join(t.replace("\n", " ") + "\n" for t in texts)
        out = subprocess.run([self.binary], input=inp.encode("utf-8"), capture_output=True,
                             check=True).stdout.decode("utf-8").split("\n")
        if len(out) != len(texts) + 1 or out[-1] != "":
            raise SystemExit("rubi_norm returned another number of lines")
        return out[:-1]


def texts(dump, path, cnorm):
    """Per crop: greedy (W8I argmax) or rubi (beam over the re-decoded top-K when the greedy text is not empty,
    then textnorm and the non-Japanese filter; no cascade)."""
    cr = dump["crops"]
    if path == "greedy":
        return [c["g"] for c in cr]
    if path == "rubi":
        out = cnorm([c["b"] if c["g"] else "" for c in cr])
        return [t if has_japanese(t) else "" for t in out]
    raise ValueError(path)


def transcript(lines, preds):
    from general_vn_textbox_guard import transcript_score
    sc = transcript_score(lines, preds)["all"]["full"]
    return sc["errors"], sc["chars"]


def score_hval(plan, dump, cnorm, paths=("rubi", "greedy")):
    T = {p: texts(dump, p, cnorm) for p in paths}
    clean = {"id": [], "title": [], "platform": [], "kind": [], "chars": [], "errors": {p: [] for p in paths}}
    for f in plan["fixed"]:
        ref = norm(f["ref"])
        clean["id"].append(f["id"])
        clean["title"].append(f["title"])
        clean["platform"].append(f["platform"])
        clean["kind"].append(f["kind"])
        clean["chars"].append(len(ref))
        for p in paths:
            clean["errors"][p].append(lev(norm(T[p][f["crop"]]), ref))
    page = {"unit": [], "title": [], "platform": [], "variant": [], "chars": [], "junk": [], "merged": [],
            "missed": [], "errors": {p: [] for p in paths}}
    for u in plan["units"]:
        page["unit"].append(u["unit"])
        page["title"].append(u["title"])
        page["platform"].append(u["platform"])
        page["variant"].append(u["variant"])
        for k in ("junk", "merged", "missed"):
            page[k].append(int(u.get(k, 0)))
        n = None
        for p in paths:
            e, n = transcript(u["lines"], [T[p][c] for c in u["finder"]])
            page["errors"][p].append(e)
        page["chars"].append(n)
    return {"hval_clean": clean, "hval_page": page}


def score_dev(units_doc, plan, dump, cnorm, exclude_vns, paths=("rubi", "greedy"), view="psp2x"):
    """reports/vocr5-analysis's cohorts in the console view: tv / dn fixed lines (rows of every development crop,
    the name-like rule of VOCR5's selection marking names), wd whole-dialogue units."""
    T = {p: texts(dump, p, cnorm) for p in paths}
    out = {}
    for cohort in ("tv", "dn"):
        rows = {"id": [], "vn": [], "name": [], "chars": [], "errors": {p: [] for p in paths}}
        for i, f in enumerate(units_doc["fixed"]):
            if f["cohort"] != cohort or f["vn"] in exclude_vns:
                continue
            c = plan["fixed_ix"][view][i]
            ref = norm(f["ref_raw"])
            nm = ("name" in (f["meta"].get("styles") or [])) if cohort == "dn" else namelike(f["ref_raw"])
            rows["id"].append(f"{f['file']}|{f['id']}")
            rows["vn"].append(f["vn"])
            rows["name"].append(bool(nm))
            rows["chars"].append(len(ref))
            for p in paths:
                rows["errors"][p].append(lev(norm(T[p][c]), ref))
        out[cohort] = rows
    wd = {"unit": [], "vn": [], "kind": [], "chars": [], "errors": {p: [] for p in paths}}
    for pu in plan["units"][view]:
        u = units_doc["units"][pu["u"]]
        if u["vn"] in exclude_vns:
            continue
        lines = [{"id": ln["id"], "text": ln["text"], "styles": ln.get("styles", [])} for ln in u["lines"]]
        wd["unit"].append(f"{u['cohort']}|{u['unit']}|{u['variant']}")
        wd["vn"].append(u["vn"])
        wd["kind"].append(u["kind"])
        n = None
        for p in paths:
            e, n = transcript(lines, [T[p][c] for c in pu["finder"]])
            wd["errors"][p].append(e)
        wd["chars"].append(n)
    out["wd"] = wd
    return out


def rate(rows, path, mask=None):
    e = np.asarray(rows["errors"][path], float)
    c = np.asarray(rows["chars"], float)
    if mask is not None:
        e, c = e[mask], c[mask]
    return round(100.0 * e.sum() / max(1.0, c.sum()), 4)


def summarize(sc, paths=("rubi", "greedy")):
    s = {}
    for p in paths:
        d = {}
        if "hval_clean" in sc:
            h = sc["hval_clean"]
            kind = np.asarray(h["kind"])
            plat = np.asarray(h["platform"])
            d["hval_clean_dlg"] = rate(h, p, kind == "line")
            d["hval_clean_dlg_vita"] = rate(h, p, (kind == "line") & (plat == "vita"))
            d["hval_clean_dlg_psp"] = rate(h, p, (kind == "line") & (plat == "psp"))
            d["hval_clean_name"] = rate(h, p, kind == "name")
            g = sc["hval_page"]
            plat = np.asarray(g["platform"])
            nj = np.asarray(g["junk"]) == 0
            d["hval_page"] = rate(g, p, nj)
            d["hval_page_vita"] = rate(g, p, nj & (plat == "vita"))
            d["hval_page_psp"] = rate(g, p, nj & (plat == "psp"))
            d["hval_page_all"] = rate(g, p)
        if "tv" in sc:
            tv, dn, wd = sc["tv"], sc["dn"], sc["wd"]
            d["tv"] = rate(tv, p, ~np.asarray(tv["name"], bool))
            d["dn"] = rate(dn, p, ~np.asarray(dn["name"], bool))
            d["wd"] = rate(wd, p)
            d["dev_score"] = round(0.25 * d["tv"] + 0.25 * d["dn"] + 0.5 * d["wd"], 4)
        s[p] = d
    return s


def blob_info(blob):
    from pyvocr import Vocr, load_lib
    r = Vocr(str(blob), mode="w8i", max_width=1024, lib=load_lib(str(ROOT / "runtime/build/libvocr.so")))
    return {"blob_bytes": int(r.blob_bytes), "arena_bytes": int(r.arena_bytes)}


def score(args, proto=None):
    cnorm = Norm(args.rubi_norm)
    sc = {}
    if args.hval_plan:
        hp = pickle.load(open(args.hval_plan, "rb"))
        hd = json.loads(Path(args.hval_dump).read_text(encoding="utf-8"))
        if hd["plan_sha256"] != sha256(args.hval_plan):
            raise SystemExit("the hval dump was read from another plan")
        sc.update(score_hval(hp, hd, cnorm))
    if args.dev_plan:
        dp = pickle.load(open(args.dev_plan, "rb"))
        du = pickle.load(open(args.dev_units, "rb"))
        dd = json.loads(Path(args.dev_dump).read_text(encoding="utf-8"))
        if dd["plan_sha256"] != sha256(args.dev_plan):
            raise SystemExit("the dev dump was read from another plan")
        sc.update(score_dev(du, dp, dd, cnorm, set(args.exclude_vns or [])))
    doc = {"kind": "vocr6-score", "name": args.name, "blob": str(args.blob), "blob_sha256": sha256(args.blob),
           **blob_info(args.blob), "hval_plan_sha256": sha256(args.hval_plan) if args.hval_plan else None,
           "dev_plan_sha256": sha256(args.dev_plan) if args.dev_plan else None,
           "exclude_vns": sorted(args.exclude_vns or []), "rubi_norm_sha256": sha256(args.rubi_norm),
           "path": "W8I greedy + top-8, counter re-decode 6000/4000, beam 8 + LM (alpha 500, beta 3500, gate 8000, "
                   "EOS, bracket 8000), textnorm, non-Japanese filter; no cascade",
           "summary": summarize(sc), "rows": sc, "at": now()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print("scored:", args.name, json.dumps(doc["summary"]), flush=True)
    return doc


def all_cmd(args):
    """read both plans and score: what an arm agent runs on its final blob."""
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    sp = proto["scoring"]
    use = {k: bool(sp[k]["path"]) for k in ("hval_plan", "dev_plan")}   # an empty path: a smoke test's skip
    need = ["lm", "rubi_norm"] + (["hval_plan"] if use["hval_plan"] else []) + \
        (["dev_plan", "dev_units"] if use["dev_plan"] else [])
    for k in need:
        if sha256(sp[k]["path"]) != sp[k]["sha256"]:
            raise SystemExit(f"{k} {sp[k]['path']} differs from the protocol's sha256")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hd, dd = out / f"dump_hval_{args.name}.json", out / f"dump_dev_{args.name}.json"
    if use["hval_plan"]:
        read(sp["hval_plan"]["path"], args.blob, sp["lm"]["path"], hd, args.workers)
    if use["dev_plan"]:
        read(sp["dev_plan"]["path"], args.blob, sp["lm"]["path"], dd, args.workers)
    ns = argparse.Namespace(name=args.name, blob=args.blob, rubi_norm=sp["rubi_norm"]["path"],
                            hval_plan=sp["hval_plan"]["path"] if use["hval_plan"] else "", hval_dump=hd,
                            dev_plan=sp["dev_plan"]["path"] if use["dev_plan"] else "",
                            dev_units=sp["dev_units"]["path"], dev_dump=dd, exclude_vns=sp["dev_exclude_vns"],
                            out=out / f"score_{args.name}.json")
    return score(ns)


def merge_npz(args):
    """One shard of up to --n rows drawn evenly over the --glob files (a run's logged --val set; nothing selects on
    it): the same row order for every run. Rows keep their meta, plus "file"."""
    import glob as _glob
    files = sorted(f for g in args.glob for f in _glob.glob(g))
    if not files:
        raise SystemExit("no files")
    rows = []
    for f in files:
        rows += [(m, lab, img, f) for m, lab, img in load_npz(f) if lab]
    rng = np.random.default_rng(args.seed)
    take = sorted(rng.choice(len(rows), size=min(args.n, len(rows)), replace=False))
    sel = [rows[i] for i in take]
    data = np.concatenate([r[2].ravel() for r in sel]).astype(np.uint8)
    offs = np.concatenate([[0], np.cumsum([r[2].size for r in sel])]).astype(np.int64)
    shapes = np.asarray([r[2].shape for r in sel], np.int32)
    labels = json.dumps([r[1] for r in sel], ensure_ascii=False)
    meta = json.dumps([dict(r[0], file=Path(r[3]).name) for r in sel], ensure_ascii=False)
    np.savez(args.out, data=data, offsets=offs, shapes=shapes, labels=np.asarray(labels), meta=np.asarray(meta))
    print(f"MERGED {len(sel)} of {len(rows)} rows from {len(files)} files -> {args.out} sha256 {sha256(args.out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge-npz")
    p.add_argument("--glob", nargs="+", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("hval-plan")
    p.add_argument("--pools", required=True)
    p.add_argument("--pagepool", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("read")
    p.add_argument("--plan", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--lm", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--head-gate", default="")
    p.add_argument("--two-level", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--name", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--rubi-norm", required=True)
    p.add_argument("--hval-plan", default="")
    p.add_argument("--hval-dump", default="")
    p.add_argument("--dev-plan", default="")
    p.add_argument("--dev-units", default="")
    p.add_argument("--dev-dump", default="")
    p.add_argument("--exclude-vns", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p = sub.add_parser("all")
    p.add_argument("--name", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--protocol", default=str(ROOT / "eval/vocr6_protocol.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "merge-npz":
        merge_npz(a)
    elif a.cmd == "hval-plan":
        hval_plan(a)
    elif a.cmd == "read":
        read(a.plan, a.blob, a.lm, a.out, a.workers, a.head_gate, a.two_level)
    elif a.cmd == "score":
        score(a)
    else:
        all_cmd(a)


if __name__ == "__main__":
    main()

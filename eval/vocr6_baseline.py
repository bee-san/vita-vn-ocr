#!/usr/bin/env python3
"""VOCR6's baseline: the released VOCR5 recognizers, as Rubi runs them, on the hooked corpus (docs/vocr6-data.md,
issue #125) and on VOCR5's development sample.

  vocr6_baseline.py build --set hval|test --pools POOLS --pagepool PAGEPOOL --pages PAGES --cache CACHE [--procs 5]
      the set's crops and scoring units (CACHE/<set>/crops.pkl, units.json):
        clean   the pools' rows of the set: line (dialogue), name, line_x2 (PSP lines cut from the page upscaled 2x,
                the Adrenaline view), vreflow (the vertical title's columns re-flowed), vertical (its columns turned,
                read by VOCR5-V only), as stored (console polarity)
        grad / refine   the C GRAD finder without and with the refinement (VOCR_REFINE_DEFAULT; finder_tighten's
                driver, checked box for box against its Python twin) on every whole-dialogue region of the set's
                pages (eval/finder_tighten.py's rule, base / +4 / +8 px): Vita at native size, PSP at native size
                (x1) and upscaled 2x bilinear (x2, the Adrenaline view); each finder line read with its dark flag
        spot    the textspot3 pick (T3_kd_t64n8c16_seed3_detref, its det+refine decode; eval/textspot.py's twin) on
                every frame with verified text, PSP frames upscaled 2x as textspot's sets do; horizontal boxes read by
                the tier, vertical ones by VOCR5-V; references from the page pool (main-text boxes, names, ruby and
                every unverified renderer box as ignore areas)
  vocr6_baseline.py dev-build --cache CACHE
      VOCR5's development crops (eval/head2level_dev.py: tv and dn as stored and in screen polarity, wd's C finder
      lines with and without the refinement) -> CACHE/dev
  vocr6_baseline.py read --set hval|test|dev --tier S|N|L|V --cache CACHE [--procs 5]
      every crop of the set through the tier's released arm (ARMS: W8I, the head gate, Normal / Large's two-level
      head; top-8, the counter re-decode {6000, 4000}, beam 8 + the 1 MB LM at Rubi's settings, and Rubi's ungated
      re-read of a line whose re-decode finds a counter or whose reading holds a full-size カ/ケ before a counter
      character); V: VOCR5-V greedy, the set's vertical crops only -> CACHE/<set>/reads_<tier>.json
  vocr6_baseline.py score --cache CACHE --out RESULTS.json [--sets hval,test,dev]
      decodes greedy, rubi_h2 (head2level's rubi: no re-read), rubi (Rubi's line path) and rubi_cascade (a line under
      700 confidence read again by the next tier up); dialogue CER (names a secondary column) per title, Vita / PSP /
      all with title-cluster bootstrap 95% intervals (10,000 resamples, seed 0; a one-title set: page clusters),
      per-title intervals over pages; textspot's frame metrics (F1, all-text and dialogue CER); the dev components
      and score of eval/vocr5_select.py with VN-cluster intervals
  vocr6_baseline.py log --results RESULTS.json --experiments EXPERIMENTS.jsonl
      one JSON line per tier, arm=baseline, appended under an exclusive flock

The hooked crops are the owner's private game data: caches stay under $HERD; RESULTS.json holds counts, rates and
hashes only.
"""
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT, ROOT / "data", ROOT / "runtime", ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
A = HERD / "shared" / "artifacts"
LM_FILE = A / "vocr5-train" / "release" / "v0.3.0" / "vocr_lm_o3_1000k.vlm"
LM_SHA = "993000611555171daf6361e29a503b0871a5f3dd240a19555d55f16aaa3ab3d9"
# the released arms: v0.3.5-headgate.1's Small, v0.3.6-head2level.1's Normal and Large (VOCR5_{N,L}_w8g2.vocr)
ARMS = {"S": {"blob": A / "head-gate" / "blobs" / "VOCR5_S_w8g.vocr", "gate": True, "two": False,
              "sha256": "76a1bc5de5cd0419a7edad31bc198af8016bda5592a7e595de75697671d3e928",
              "release": "v0.3.5-headgate.1 VOCR5_S_w8g.vocr (= v0.3.6-head2level.1's Small)"},
        "N": {"blob": A / "head-2level" / "blobs" / "VOCR5_N_2Lg_K500_r96.vocr", "gate": True, "two": True,
              "sha256": "f0c8e52c9c1d9f2a714d5ca3ecfba9c0b031ee7ac51a522a2c29fd558d51761b",
              "release": "v0.3.6-head2level.1 VOCR5_N_w8g2.vocr"},
        "L": {"blob": A / "head-2level" / "blobs" / "VOCR5_L_2Lg_K500_r96.vocr", "gate": True, "two": True,
              "sha256": "41c3543ff8dddd45dc953a6a5f5d579ef9ca779f0e43196406be59c1d985c5b5",
              "release": "v0.3.6-head2level.1 VOCR5_L_w8g2.vocr"},
        "V": {"blob": A / "vertical" / "release" / "v0.3.2-vertical.1" / "VOCR5_V_w8.vocr", "gate": False,
              "two": False, "sha256": "ccd20c63bb2620f054538b456904bdf19b74d483d3da8de865009e282bb04172",
              "release": "v0.3.2-vertical.1 VOCR5_V_w8.vocr (greedy)"}}
TIERS = ("S", "N", "L")
NEXT = {"S": "N", "N": "L", "L": None}
SPOT_CAND, SPOT_DECODE = "T3:kd_t64n8c16_seed3", "det+refine"
SPOT_SHA = "20c3f7cc4f07b443741bbb45fe4270f6fa8c74ce29045a796a5ed8887a3fb26a"   # v0.3.4-textspot3.1's asset
DEV_COUPLED = ("v12188", "v18432")   # dev-sample VNs the hooked corpus trains on (PCSG00576) or tests (PCSG00807)
DECODES = ("greedy", "rubi_h2", "rubi", "rubi_cascade")
N_BOOT = 10000
VARIANTS = ("base", "expanded-4", "expanded-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_inputs():
    bad = [t for t, a in ARMS.items() if sha256(a["blob"]) != a["sha256"]]
    if bad or sha256(LM_FILE) != LM_SHA:
        raise SystemExit(f"released files differ from their sha256: {bad or 'the LM'}")


def write_json(path, doc, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------- build

def page_rgb(pages, row, scale):
    import cv2
    p = Path(pages) / row["image"]
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
        raise SystemExit(f"{p}: differs from its sha256")
    rgb = np.ascontiguousarray(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[..., ::-1])
    if scale != 1:
        rgb = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
    return rgb


def clean_crops(pools, split):
    """[(key, crop, invert)] and units of the set's pool rows (keys clean|pool|title|index)."""
    import import_hooked as IH
    crops, units = [], []
    for pool in ("line", "name", "line_x2", "name_x2", "vreflow", "vertical"):
        for f in sorted((Path(pools) / split / pool).glob("*.npz")):
            for i, (m, lab, g) in enumerate(IH.read_npz(f)):
                key = f"clean|{pool}|{f.stem}|{i}"
                crops.append((key, np.ascontiguousarray(g), False))
                units.append({"key": key, "pool": pool, "title_id": f.stem, "platform": m["platform"],
                              "page": m["page"], "id": m["id"], "ref": norm(lab), "name": pool.startswith("name"),
                              "vertical": pool == "vertical"})
    return crops, units


def region_jobs(pagepool, split, pages):
    """[(title, page, scale, variant, gi, region grey, reference texts, platform)]: the set's dialogue regions."""
    from common import to_gray
    out = []
    for f in sorted(Path(pagepool).glob("*.jsonl")):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        for r in rows:
            if r["split"] != split or not r["regions"]:
                continue
            texts = {ln["id"]: ln["text"] for ln in r["lines"]}
            scales = (1, 2) if r["platform"] == "psp" else (1,)
            for s in scales:
                grey = to_gray(page_rgb(pages, r, s))
                for gi, g in enumerate(r["regions"]):
                    x0, y0, x1, y1 = (s * v for v in g["region"])
                    out.append((r["title_id"], r["page"], s, g["variant"], gi,
                                np.ascontiguousarray(grey[y0:y1, x0:x1]), [texts[i] for i in g["lines"]],
                                r["platform"]))
    return out


def finder_crops(jobs, cache):
    """The C finder (flags 0) and the refinement on every region: crops and units (keys grad|... / refine|...)."""
    import finder_tighten as FT

    import lines_refine as LR
    ident = FT.build(Path(cache) / "build")
    flags = LR.parse_flags("default")
    crops, units = [], []
    imgs = [j[5] for j in jobs]
    for mode, fl in (("grad", 0), ("refine", flags)):
        res = []
        for c0 in range(0, len(imgs), 2000):                  # the driver reads one file per call
            res += FT.lines_of(ident["driver"], imgs[c0:c0 + 2000], fl)
        for (tid, page, s, variant, gi, img, refs, plat), (boxes, dark, org, rows, found) in zip(jobs, res):
            keys = []
            for li, ((x, y, w, h), d) in enumerate(zip(boxes, dark)):
                key = f"{mode}|x{s}|{tid}|{page}|{variant}|{gi}|{li}"
                crops.append((key, np.ascontiguousarray(img[y:y + h, x:x + w]), bool(d)))
                keys.append(key)
            units.append({"mode": mode, "scale": s, "title_id": tid, "page": page, "variant": variant, "gi": gi,
                          "platform": plat, "keys": keys, "refs": refs, "finder_lines": found})
    return crops, units, {k: ident[k] for k in ("driver_sha256", "library_sha256", "compiler")}, flags


def spot_refs(r, s):
    """textspot references of a page-pool row at scale s: main-text boxes of verified lines (dialogue, names), ruby and
    every renderer box no verified record covers as ignore areas; a tall furigana band too."""
    refs, covered = [], set()
    for ln in r["lines"]:
        covered.update(ln.get("box_index") or [])
        box = ln["box_main"] if ln.get("box_main") else ln["box"]
        x, y, w, h = (s * int(v) for v in box)
        if ln["kind"] == "ruby":
            refs.append({"box": [x, y, w, h], "cls": "dialogue", "vertical": False, "text": None, "ignore": True,
                         "why": "furigana"})
            continue
        refs.append({"box": [x, y, w, h], "cls": "name" if ln["kind"] == "name" else "dialogue",
                     "vertical": bool(ln["vertical"]), "text": ln["text"], "ignore": False, "id": ln["id"]})
        if ln.get("tall_band"):
            bx, by, bw, _ = ln["box"]
            refs.append({"box": [s * bx, s * by, s * bw, s * (ln["box_main"][1] - by)], "cls": "dialogue",
                         "vertical": False, "text": None, "ignore": True, "why": "undecided"})
    for b in r["boxes"]:
        if b["index"] not in covered:
            x0, y0, x1, y1 = b["xyxy"]
            refs.append({"box": [int(s * x0), int(s * y0), int(np.ceil(s * x1)) - int(s * x0),
                                 int(np.ceil(s * y1)) - int(s * y0)], "cls": "dialogue", "vertical": False,
                         "text": None, "ignore": True, "why": "undecided"})
    return refs


def spot_crops(pagepool, split, pages, cache, procs):
    import textspot as TS
    import textspot3 as T3E
    import textspot3_protected as TP3

    from common import to_gray
    spec, version = TP3.cand_spec(SPOT_CAND, SPOT_DECODE)
    if TS.sha256_bytes(spec.blob) != SPOT_SHA:
        raise SystemExit("the textspot3 pick's blob is not the selection record's")
    frames, info = [], []
    for f in sorted(Path(pagepool).glob("*.jsonl")):
        for x in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            if r["split"] != split or not r["lines"] or r.get("untrusted"):
                continue
            s = 2 if r["platform"] == "psp" else 1
            rgb = page_rgb(pages, r, s)
            row = {"id": f"{r['title_id']}:{r['page']}", "vn": r["title_id"], "part": r["platform"],
                   "text_free": False, "refs": spot_refs(r, s)}
            frames.append(TS.Frame(row, to_gray(rgb), TS.pixel_sha256(rgb)))
            info.append({"title_id": r["title_id"], "page": r["page"], "platform": r["platform"], "scale": s})
    with T3E.dispatch({version}):
        nets = TS.load_nets([spec])
        det = TS.detect(frames, [spec], nets, (4, -8), workers=procs)
    crops, units = [], []
    for f, inf, (arms, _) in zip(frames, info, det):
        arm = arms[spec.name]
        keys = []
        for bi, b in enumerate(arm["boxes"]):
            key = f"spot|{inf['title_id']}|{inf['page']}|{bi}"
            crops.append((key, TS.crop_of(f.grey, b, b[4]), bool(b[6])))
            keys.append([key, bool(b[4])])
        units.append({**inf, "frame": f.id, "width": f.width, "height": f.height, "refs": f.refs,
                      "boxes": [list(map(int, b[:7])) for b in arm["boxes"]], "keys": keys})
    meta = {"arm": spec.name, "decode": spec.decode, "blob_sha256": TS.sha256_bytes(spec.blob),
            "blob_bytes": len(spec.blob), "vtsp_version": version}
    return crops, units, meta


def cmd_build(a):
    out = Path(a.cache) / a.set
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    crops, units = clean_crops(a.pools, a.set)
    jobs = region_jobs(a.pagepool, a.set, a.pages)
    fc, fu, ident, flags = finder_crops(jobs, a.cache)
    sc, su, smeta = spot_crops(a.pagepool, a.set, a.pages, a.cache, a.procs)
    allc = crops + fc + sc
    with open(out / "crops.pkl", "wb") as f:
        pickle.dump(allc, f, protocol=4)
    doc = {"set": a.set, "clean": units, "finder": fu, "spot": su, "finder_identity": ident, "refine_flags": flags,
           "spot_meta": smeta, "crops": len(allc), "seconds": round(time.time() - t0, 1),
           "pools_manifest_sha256": sha256(Path(a.pools) / "manifest.json"),
           "pagepool_manifest_sha256": sha256(Path(a.pagepool) / "manifest.json")}
    write_json(out / "units.json", doc)
    print(json.dumps({"set": a.set, "crops": len(allc), "clean": len(units), "regions": len(fu) // 2,
                      "frames": len(su), "seconds": doc["seconds"]}))


def cmd_dev_build(a):
    import head2level_dev as H2
    cache = Path(a.cache) / "dev"
    cache.mkdir(parents=True, exist_ok=True)
    H2.cmd_lines(argparse.Namespace(cache=str(cache)))
    crops = H2.all_crops(cache)
    with open(cache / "crops.pkl", "wb") as f:
        pickle.dump([(k, np.ascontiguousarray(g), bool(inv)) for k, g, inv in crops], f, protocol=4)
    print(json.dumps({"set": "dev", "crops": len(crops)}))


# ----------------------------------------------------------------------------- read

W = {}


def _init(tier):
    import head2level_dev as H2
    from pyvocr import Vocr, VocrLM
    arm = ARMS[tier]
    kw = {}
    if arm["gate"]:
        kw["head_gate"] = 1
    if arm["two"]:
        kw["two_level"] = 1
    W.update(rt=Vocr(str(arm["blob"]), mode="w8i", max_width=1024, **kw), tier=tier, arm=arm, H2=H2)
    if tier != "V":
        W["lm"] = VocrLM(W["rt"], str(LM_FILE), **H2.RUBI_LM)


def _lm_text(o):
    rt, lm, H2 = W["rt"], W["lm"], W["H2"]
    o2, st = rt.redecode(o, tau=H2.COUNTER["tau"], open=H2.COUNTER["open"])
    return lm.decode(o2["topv"], o2["topc"], o2["blank"]), st["contexts"]


def _read(job):
    i, img, inv = job
    rt, H2 = W["rt"], W["H2"]
    if W["tier"] == "V":
        return i, rt.text(img, invert=inv), "", "", 1000, 0, 0
    o = rt.run_topk(img, 8, invert=inv, esc=W["arm"]["two"])
    g = o["text"]
    conf = H2.line_conf(o["conf_min"], g)
    lm1 = lm2 = ""
    reread = 0
    if g:
        lm1, ctx = _lm_text(o)
        lm2 = lm1
        if (W["arm"]["gate"] or W["arm"]["two"]) and (ctx or H2.REREAD_RE.search(g)):
            reread = 1                                   # Rubi's ungated re-read (docs/head2level.md, "Use")
            u = rt.run_topk(img, 8, invert=inv, ungated=True)
            lm2 = _lm_text(u)[0] if u["text"] else ""
    return i, g, lm1, lm2, conf, int(o["frames"]), reread


def cmd_read(a):
    check_inputs()
    d = Path(a.cache) / a.set
    with open(d / "crops.pkl", "rb") as f:
        crops = pickle.load(f)
    if a.tier == "V":                                    # VOCR5-V reads the vertical crops only
        units = json.loads((d / "units.json").read_text(encoding="utf-8")) if a.set != "dev" else None
        vkeys = set()
        if units:
            vkeys |= {u["key"] for u in units["clean"] if u["vertical"]}
            vkeys |= {k for u in units["spot"] for k, v in u["keys"] if v}
        crops = [c for c in crops if c[0] in vkeys]
    t0 = time.time()
    res = [None] * len(crops)
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.tier,)) as pool:
        for n, r in enumerate(pool.imap_unordered(_read, [(i, c[1], c[2]) for i, c in enumerate(crops)],
                                                  chunksize=16)):
            res[r[0]] = r
            if n % 5000 == 0:
                print(f"  {a.set} {a.tier}: {n}/{len(crops)} {time.time() - t0:.0f}s", flush=True)
    from pyvocr import Vocr
    arm = ARMS[a.tier]
    kw = {"head_gate": 1} if arm["gate"] else {}
    if arm["two"]:
        kw["two_level"] = 1
    rt = Vocr(str(arm["blob"]), mode="w8i", max_width=1024, **kw)
    doc = {"set": a.set, "tier": a.tier, "blob": str(arm["blob"]), "blob_sha256": arm["sha256"],
           "release": arm["release"], "blob_bytes": rt.blob_bytes, "arena_w8i": rt.arena_bytes,
           "head_gate": arm["gate"], "two_level": arm["two"], "lm_sha256": LM_SHA if a.tier != "V" else None,
           "crops": len(crops), "seconds": round(time.time() - t0, 1),
           "keys": [c[0] for c in crops], "greedy": [r[1] for r in res], "lm": [r[2] for r in res],
           "lm_reread": [r[3] for r in res], "conf": [r[4] for r in res], "frames": [r[5] for r in res],
           "reread": [r[6] for r in res]}
    write_json(d / f"reads_{a.tier}.json", doc)
    print(json.dumps({k: v for k, v in doc.items() if not isinstance(v, list)}))


# ----------------------------------------------------------------------------- score

def load_reads(d, tier):
    p = Path(d) / f"reads_{tier}.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["index"] = {k: i for i, k in enumerate(doc["keys"])}
    return doc


def decodes(reads, lib):
    """{tier: {decode: {key: text}}} of every tier read."""
    from pyvocr import post_counters
    out = {}
    for t in TIERS:
        r = reads.get(t)
        if r is None:
            continue
        rubi_h2 = [post_counters(x, lib)[0] for x in r["lm"]]
        rubi = [post_counters(x, lib)[0] for x in r["lm_reread"]]
        out[t] = {"greedy": dict(zip(r["keys"], r["greedy"])), "rubi_h2": dict(zip(r["keys"], rubi_h2)),
                  "rubi": dict(zip(r["keys"], rubi)), "conf": dict(zip(r["keys"], r["conf"]))}
    for t in TIERS:
        if t not in out:
            continue
        nt = NEXT[t]
        if nt is None:
            out[t]["rubi_cascade"] = dict(out[t]["rubi"])
            out[t]["cascaded"] = 0
        elif nt in out:
            nxt = out[nt]["rubi"]
            out[t]["rubi_cascade"] = {k: (nxt[k] if out[t]["conf"][k] < 700 else v) for k, v in out[t]["rubi"].items()}
            out[t]["cascaded"] = sum(1 for c in out[t]["conf"].values() if c < 700)
    return out


def boot_ratio(errors, chars, clusters, seed=0, n=N_BOOT):
    """Point CER (%) and the 95% interval of a cluster bootstrap (clusters resampled with replacement)."""
    errors, chars = np.asarray(errors, float), np.asarray(chars, float)
    if not chars.sum():
        return {"cer": None, "errors": 0, "chars": 0, "ci95": None, "clusters": 0}
    ug, inv = np.unique(np.asarray(clusters), return_inverse=True)
    ge = np.bincount(inv, weights=errors, minlength=len(ug))
    gc = np.bincount(inv, weights=chars, minlength=len(ug))
    point = 100 * errors.sum() / chars.sum()
    out = {"cer": round(float(point), 4), "errors": int(errors.sum()), "chars": int(chars.sum()),
           "clusters": int(len(ug)), "ci95": None}
    if len(ug) >= 2:
        idx = np.random.default_rng(seed).integers(0, len(ug), size=(n, len(ug)))
        den = gc[idx].sum(1)
        ok = den > 0
        rep = 100 * ge[idx].sum(1)[ok] / den[ok]
        lo, hi = np.percentile(rep, [2.5, 97.5])
        out["ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def macro(rows, seed=0, n=N_BOOT):
    """Title-macro CER (%): the mean of the titles' CERs (eval's game-macro), with a title-cluster bootstrap."""
    by = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by[r[0]][0] += r[3]
        by[r[0]][1] += r[4]
    cers = np.asarray([100 * e / c for e, c in by.values() if c > 0])
    if not len(cers):
        return None
    out = {"macro_cer": round(float(cers.mean()), 4), "median_title_cer": round(float(np.median(cers)), 4),
           "macro_ci95": None}
    if len(cers) >= 2:
        idx = np.random.default_rng(seed).integers(0, len(cers), size=(n, len(cers)))
        lo, hi = np.percentile(cers[idx].mean(1), [2.5, 97.5])
        out["macro_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def summarize(rows, group_of_title=None):
    """rows: [(title, platform, page, errors, chars)] -> per title (page clusters), Vita / PSP / all (title clusters,
    or page clusters when one title), and each group's title-macro CER."""
    out = {"per_title": {}}
    titles = sorted({r[0] for r in rows})
    for t in titles:
        rs = [r for r in rows if r[0] == t]
        out["per_title"][t] = boot_ratio([r[3] for r in rs], [r[4] for r in rs], [r[2] for r in rs])
    for name, keep in (("vita", lambda r: r[1] == "vita"), ("psp", lambda r: r[1] == "psp"), ("all", lambda r: True)):
        rs = [r for r in rows if keep(r)]
        if not rs:
            continue
        ts = {r[0] for r in rs}
        cl = [r[0] for r in rs] if len(ts) > 1 else [f"{r[0]}:{r[2]}" for r in rs]
        out[name] = boot_ratio([r[3] for r in rs], [r[4] for r in rs], cl)
        out[name]["titles"] = len(ts)
        out[name]["cluster"] = "title" if len(ts) > 1 else "page"
        out[name].update(macro(rs) or {})
    return out


def score_clean(units, texts, label_in_train=None):
    """{pool: summary} of the clean crops (dialogue pools; names; the Adrenaline view; re-flowed / vertical). With
    label_in_train ({line id: bool}), also `line|label_unseen`: the line pool without the lines whose label occurs in
    the same title's train pool (they are not duplicates of it, but a model may have learnt the text)."""
    res = {}
    by = defaultdict(list)
    for u in units:
        if u["key"] in texts:
            row = (u["title_id"], u["platform"], u["page"], lev(norm(texts[u["key"]]), u["ref"]), len(u["ref"]))
            by[u["pool"]].append(row)
            if label_in_train is not None and u["pool"] == "line" and not label_in_train.get(u["id"]):
                by["line|label_unseen"].append(row)
    for pool, rows in sorted(by.items()):
        res[pool] = summarize(rows)
        res[pool]["exact"] = round(sum(1 for r in rows if r[3] == 0) / len(rows), 4)
        res[pool]["lines"] = len(rows)
    return res


UNSEEN_KEYS = ("refine|console|base", "grad|console|base", "refine|x1|base")


def score_finder(units, texts, region_seen=None):
    """Whole-dialogue scores per mode, scale and variant, the console view, and with region_seen ({(title, page, region
    index): bool}) `<key>|unseen` for UNSEEN_KEYS: the regions none of whose lines has a train copy."""
    from general_vn_textbox_guard import transcript_score
    res = {}
    by = defaultdict(list)
    seen_of = {}
    for u in units:
        preds = [norm(texts[k]) for k in u["keys"]]
        sc = transcript_score([{"text": t, "styles": []} for t in u["refs"]], preds)["all"]["full"]
        row = (u["title_id"], u["platform"], u["page"], sc["errors"], sc["chars"])
        by[(u["mode"], u["scale"], u["variant"])].append(row)
        seen_of[id(row)] = bool(region_seen and region_seen.get((u["title_id"], u["page"], u["gi"])))
    for (mode, s, variant), rows in sorted(by.items()):
        res[f"{mode}|x{s}|{variant}"] = summarize(rows)
        res[f"{mode}|x{s}|{variant}"]["regions"] = len(rows)
    for mode in ("grad", "refine"):                        # pooled over the three variants (VOCR5's wd pooling)
        for s in (1, 2):
            rows = [r for (m, sc, _), rs in by.items() if m == mode and sc == s for r in rs]
            if rows:
                res[f"{mode}|x{s}|all-variants"] = summarize(rows)
        for variant in VARIANTS:                           # the console's view: Vita native, PSP in Adrenaline (x2)
            rows = [r for r in by.get((mode, 1, variant), []) if r[1] == "vita"] + \
                [r for r in by.get((mode, 2, variant), []) if r[1] == "psp"]
            if rows:
                res[f"{mode}|console|{variant}"] = summarize(rows)
                if region_seen is not None and f"{mode}|console|{variant}" in UNSEEN_KEYS:
                    res[f"{mode}|console|{variant}|unseen"] = summarize([r for r in rows if not seen_of[id(r)]])
                    res[f"{mode}|console|{variant}"]["seen_regions"] = sum(seen_of[id(r)] for r in rows)
                    res[f"{mode}|console|{variant}"]["seen_chars"] = sum(r[4] for r in rows if seen_of[id(r)])
    if region_seen is not None and by.get(("refine", 1, "base")):
        rows = by[("refine", 1, "base")]
        res["refine|x1|base|unseen"] = summarize([r for r in rows if not seen_of[id(r)]])
    return res


def score_spot(units, texts, vtexts, page_seen=None):
    """textspot's frame metrics; with page_seen ({(title, page): bool}) also the dialogue CER of the frames without a
    line that has a train copy (`dialogue_unseen`)."""
    import textspot_metrics as M
    rows_all, rows_dlg, counts, rows_unseen = [], [], [], []
    for u in units:
        dets = []
        for (key, vert), b in zip(u["keys"], u["boxes"]):
            t = (vtexts.get(key, "") if vert else texts[key]) or ""
            dets.append({"box": list(b[:4]), "vertical": bool(vert), "text": t})
        m = M.match_frame(dets, u["refs"], u["width"], u["height"])
        e = M.e2e_frame(dets, u["refs"], u["width"], u["height"])
        rows_all.append((u["title_id"], u["platform"], u["page"], *e["all"]))
        rows_dlg.append((u["title_id"], u["platform"], u["page"], *e["dialogue"]))
        if page_seen is not None and not page_seen.get((u["title_id"], u["page"])):
            rows_unseen.append(rows_dlg[-1])
        counts.append((u["title_id"], u["platform"], m["tp"], m["fp"], m["fn"]))
    det = {}
    for name, keep in (("vita", lambda c: c[1] == "vita"), ("psp", lambda c: c[1] == "psp"), ("all", lambda c: True)):
        cs = [c for c in counts if keep(c)]
        if cs:
            tp, fp, fn = (sum(c[i] for c in cs) for i in (2, 3, 4))
            det[name] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 4)}
    out = {"frames": len(units), "detection": det, "dialogue": summarize(rows_dlg), "all_text": summarize(rows_all)}
    if page_seen is not None:
        out["dialogue_unseen"] = summarize(rows_unseen)
        out["seen_frames"] = len(units) - len(rows_unseen)
    return out


def seen_flags(pools, pagepool, split):
    """What of a set the training side has seen (pools v1's decisions, the page pool):
    line_seen {line id: the line's duplicate cluster has a train copy} (hval lines only: screening leaves those out of
    the clean set, but the page path reads every page), label_in_train {line id: the same label is in the title's
    train line pool}, region_seen {(title, page, region index): a member line is line_seen}, page_seen {(title, page):
    a line of the page is line_seen, or the page's image equals a train or guard page's}."""
    import import_hooked as IH
    line_seen, label_in_train, region_seen, page_seen, image_seen = {}, {}, {}, {}, 0
    for f in sorted(Path(pagepool).glob("*.jsonl")):
        tid = f.stem
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not any(r["split"] == split for r in rows):
            continue
        dec = {r["id"]: r for r in IH.read_jsonl(Path(pools) / "decisions" / f"{tid}.jsonl")}
        tp = Path(pools) / "train" / "line" / f"{tid}.npz"
        train_labels = set(json.loads(str(np.load(tp)["labels"]))) if tp.exists() else set()
        train_images = {r["image_sha256"] for r in rows if r["split"] in ("train", "guard")}
        for r in rows:
            if r["split"] != split:
                continue
            seen_here = False
            for ln in r["lines"]:
                e = dec[ln["id"]]["pools"].get("line") or {}
                line_seen[ln["id"]] = bool(e.get("seen_train"))      # set by the importer for hval lines
                label_in_train[ln["id"]] = ln["text"] in train_labels
                seen_here |= line_seen[ln["id"]]
            for gi, g in enumerate(r["regions"]):
                region_seen[(tid, r["page"], gi)] = any(line_seen.get(i) for i in g["lines"])
            img = r["image_sha256"] in train_images
            image_seen += img
            page_seen[(tid, r["page"])] = seen_here or img
    return {"line_seen": line_seen, "label_in_train": label_in_train, "region_seen": region_seen,
            "page_seen": page_seen, "counts": {"lines": len(line_seen), "lines_seen": sum(line_seen.values()),
                                               "lines_label_in_train": sum(label_in_train.values()),
                                               "pages": len(page_seen), "pages_seen": sum(page_seen.values()),
                                               "pages_image_equal_to_train": image_seen,
                                               "regions": len(region_seen),
                                               "regions_seen": sum(region_seen.values())}}


def dev_components(comps, m, VS, exclude=()):
    """The dev score's components (dialogue rows; rows of the `exclude` VNs left out) with VN-cluster intervals, and
    the score's point and interval (components resampled independently, eval/vocr5_select.py's seeds and weights)."""
    res, point, reps = {}, 0.0, 0.0
    for n, c in comps.items():
        mask = ~c.is_name & ~np.isin(c.vns, list(exclude))
        e, ch, vn = c.models[m][mask], c.chars[mask], c.vns[mask]
        res[n] = boot_ratio(e, ch, vn, seed=VS.SEEDS[n])
        ug, inv = np.unique(vn, return_inverse=True)
        res[n]["vns"] = int(len(ug))
        ge = np.bincount(inv, weights=e, minlength=len(ug))
        gc = np.bincount(inv, weights=ch, minlength=len(ug))
        idx = np.random.default_rng(VS.SEEDS[n]).integers(0, len(ug), size=(N_BOOT, len(ug)))
        point = point + VS.WEIGHTS[n] * 100 * e.sum() / ch.sum()
        reps = reps + VS.WEIGHTS[n] * 100 * ge[idx].sum(1) / gc[idx].sum(1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    res["score_point"] = round(float(point), 5)
    res["score_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    if exclude:
        res["excluded_vns"] = list(exclude)
        res["score"] = res["score_point"]
    return res


def dev_scores(cache, reads, dec):
    """VOCR5's development score components (eval/vocr5_select.py) of every tier and decode, with VN-cluster
    intervals per component and of the score (components resampled independently, seeds 0 / 1 / 2)."""
    import head2level_dev as H2
    import vocr5_select as VS
    out = {}
    for t, r in reads.items():
        if t not in dec:
            continue
        paths = {}
        for d in DECODES:
            if d not in dec[t]:
                continue
            texts = [dec[t][d][k] for k in r["keys"]]
            rd = {"index": r["index"], "blob": r["blob"], "blob_sha256": r["blob_sha256"],
                  "blob_bytes": r["blob_bytes"], "arena_w8i": r["arena_w8i"]}
            paths[f"base_{t}__{d}"] = H2.write_scores(cache, t, f"base_{t}__{d}", rd, texts)
        comps = H2.comps_for(paths)
        out[t] = {}
        for m in paths:
            res = dev_components(comps, m, VS)
            res["score"] = round(VS.score_of(comps, m), 5)
            for n, c in comps.items():
                if n != "wd":
                    res[f"{n}_names_cer"] = round(100 * c.rate([m], "names"), 4)
            # VOCR6's dev comparisons: without the dev-sample VNs the hooked corpus trains on or tests (DEV_COUPLED)
            res["without_" + "_".join(DEV_COUPLED)] = dev_components(comps, m, VS, exclude=DEV_COUPLED)
            out[t][m.split("__")[1]] = res
        out[t]["cascaded_lines"] = dec[t].get("cascaded")
    return out


def cmd_score(a):
    from pyvocr import load_lib
    lib = load_lib()
    res = {"kind": "vocr6-baseline", "arms": {t: {k: (str(v) if isinstance(v, Path) else v)
                                                  for k, v in ARMS[t].items()} for t in ARMS},
           "lm": {"file": LM_FILE.name, "sha256": LM_SHA}, "decodes": list(DECODES), "sets": {}}
    for s in a.sets.split(","):
        d = Path(a.cache) / s
        reads = {t: r for t in TIERS + ("V",) if (r := load_reads(d, t)) is not None}
        dec = decodes(reads, lib)
        if s == "dev":
            res["sets"][s] = {"dev": dev_scores(d, {t: reads[t] for t in TIERS if t in reads}, dec),
                              "crops": len(reads["N"]["keys"]) if "N" in reads else None}
            continue
        units = json.loads((d / "units.json").read_text(encoding="utf-8"))
        vtexts = dict(zip(reads["V"]["keys"], reads["V"]["greedy"])) if "V" in reads else {}
        sres = {"crops": units["crops"], "pools_manifest_sha256": units["pools_manifest_sha256"],
                "pagepool_manifest_sha256": units["pagepool_manifest_sha256"], "spot_meta": units["spot_meta"],
                "finder_identity": units["finder_identity"], "tiers": {}}
        flags = seen_flags(a.pools, a.pagepool, s) if a.pools and a.pagepool else None
        if flags:
            if sha256(Path(a.pools) / "manifest.json") != units["pools_manifest_sha256"]:
                raise SystemExit("--pools is not the pool version the set was built from")
            sres["seen"] = flags["counts"]
        if vtexts:
            sres["vertical_clean_VOCR5-V"] = score_clean([u for u in units["clean"] if u["vertical"]], vtexts)
        for t in TIERS:
            if t not in dec:
                continue
            tr = {"cascaded_lines": dec[t].get("cascaded"), "reread_lines": int(sum(reads[t]["reread"])),
                  "seconds": reads[t]["seconds"], "blob_bytes": reads[t]["blob_bytes"],
                  "arena_w8i": reads[t]["arena_w8i"], "decodes": {}}
            for dcd in DECODES:
                if dcd not in dec[t]:
                    continue
                texts = dec[t][dcd]
                tr["decodes"][dcd] = {
                    "clean": score_clean([u for u in units["clean"] if not u["vertical"]], texts,
                                         flags and flags["label_in_train"]),
                    "finder": score_finder(units["finder"], texts, flags and flags["region_seen"]),
                    "spot": score_spot(units["spot"], texts, vtexts, flags and flags["page_seen"])}
            sres["tiers"][t] = tr
        res["sets"][s] = sres
    write_json(a.out, res, indent=1)
    print(f"-> {a.out}")


# ----------------------------------------------------------------------------- log

def headline(res, t, d="rubi_cascade"):
    out = {}
    h = res["sets"].get("hval", {}).get("tiers", {}).get(t, {}).get("decodes", {}).get(d)
    if h:
        out["hval_clean_dialogue_cer"] = {k: h["clean"]["line"].get(k, {}).get("cer") for k in ("vita", "psp", "all")}
        out["hval_page_refine_console_base_cer"] = {k: h["finder"]["refine|console|base"].get(k, {}).get("cer")
                                                    for k in ("vita", "psp", "all")}
        out["hval_page_refine_console_base_ci95"] = h["finder"]["refine|console|base"]["all"]["ci95"]
        out["hval_page_refine_console_base_macro"] = h["finder"]["refine|console|base"]["all"].get("macro_cer")
        un = h["finder"].get("refine|console|base|unseen")
        if un:
            out["hval_page_refine_console_base_unseen_cer"] = un["all"]["cer"]
            out["hval_page_refine_console_base_unseen_ci95"] = un["all"]["ci95"]
        if "line|label_unseen" in h["clean"]:
            out["hval_clean_label_unseen_cer"] = h["clean"]["line|label_unseen"]["all"]["cer"]
        out["hval_page_refine_x1_base_cer"] = {k: h["finder"]["refine|x1|base"].get(k, {}).get("cer")
                                               for k in ("vita", "psp", "all")}
        if "refine|x2|base" in h["finder"]:
            out["hval_page_refine_psp_x2_base_cer"] = h["finder"]["refine|x2|base"]["psp"]["cer"]
        out["hval_page_spot_dialogue_cer"] = {k: h["spot"]["dialogue"].get(k, {}).get("cer")
                                              for k in ("vita", "psp", "all")}
        out["hval_spot_f1"] = h["spot"]["detection"]["all"]["f1"]
    dv = res["sets"].get("dev", {}).get("dev", {}).get(t, {}).get(d)
    if dv:
        out["dev_score"] = dv["score"]
        out["dev_cer"] = {n: dv[n]["cer"] for n in ("tv", "dn", "wd")}
        ex = dv.get("without_" + "_".join(DEV_COUPLED))
        if ex:
            out["dev_score_without_" + "_".join(DEV_COUPLED)] = ex["score"]
            out["dev_score_without_ci95"] = ex["score_ci95"]
    te = res["sets"].get("test", {}).get("tiers", {}).get(t, {}).get("decodes", {}).get(d)
    if te:
        out["test_clean_dialogue_cer"] = te["clean"]["line"]["all"]["cer"]
        out["test_page_refine_base_cer"] = te["finder"]["refine|x1|base"]["all"]["cer"]
        out["test_page_spot_dialogue_cer"] = te["spot"]["dialogue"]["all"]["cer"]
    return out


def cmd_log(a):
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    sha = sha256(a.results)
    man = Path(a.pools_manifest)
    lines = []
    for t in TIERS:
        arm = res["arms"][t]
        secs = sum(res["sets"].get(s, {}).get("tiers", {}).get(t, {}).get("seconds", 0) or 0
                   for s in ("hval", "test"))
        lines.append({"label": "vocr6-ingest", "arm": "baseline", "tier": t, "seed": None,
                      "model": arm["release"], "config_hash": arm["sha256"], "checkpoint": arm["blob"],
                      "data_manifest_hash": sha256(man) if man.exists() else None, "steps": 0,
                      "wall_clock_s": round(secs, 1), "slot_hours": round(secs / 3600, 3),
                      "decode": "rubi_cascade (W8I, head gate as released, two-level N/L, counter re-decode, "
                                "beam 8 + 1 MB LM, ungated re-read, cascade < 700)",
                      "results": headline(res, t), "results_greedy": headline(res, t, "greedy"),
                      "results_file": str(a.results), "results_sha256": sha, "time": time.strftime(
                          "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "kind": a.kind, "notes": a.note or "VOCR5 as released (head gate, two-level N/L; Rubi still pins "
                      "v0.3.0's plain blobs) through Rubi's LM/counter/cascade path; test title read once; tables: "
                      "docs/vocr6-data.md"})
    with open(a.experiments, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"{len(lines)} lines -> {a.experiments}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--set", required=True, choices=("hval", "test"))
    p.add_argument("--pools", required=True)
    p.add_argument("--pagepool", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("dev-build")
    p.add_argument("--cache", required=True)
    p = sub.add_parser("read")
    p.add_argument("--set", required=True, choices=("hval", "test", "dev"))
    p.add_argument("--tier", required=True, choices=TIERS + ("V",))
    p.add_argument("--cache", required=True)
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("score")
    p.add_argument("--cache", required=True)
    p.add_argument("--sets", default="hval,test,dev")
    p.add_argument("--pools", default="", help="the pools the sets were built from: seen flags (line, region, page)")
    p.add_argument("--pagepool", default="")
    p.add_argument("--out", required=True)
    p = sub.add_parser("log")
    p.add_argument("--results", required=True)
    p.add_argument("--experiments", required=True)
    p.add_argument("--pools-manifest", required=True)
    p.add_argument("--kind", default="result")
    p.add_argument("--note", default="")
    a = ap.parse_args(argv)
    {"build": cmd_build, "dev-build": cmd_dev_build, "read": cmd_read, "score": cmd_score, "log": cmd_log}[a.cmd](a)


if __name__ == "__main__":
    main()

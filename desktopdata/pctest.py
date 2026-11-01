#!/usr/bin/env python3
"""The real PC test set (docs/desktop-data.md, part 2): VNDB PC screenshots at their own size, of VNs that are in no
training, development or test pool, with dialogue lines read by MeikiOCR and PP-OCRv6 small and adjudicated by
Claude on contact sheets. Frozen with a manifest (ids, URLs, sha256, boxes, labels) before vocr-desktop's decision
(#169), and reserved from every training and tuning pool from then on.

  pctest.py pools      [--out P]          every VN of a pool, set or scene list -> P (pool_vns.json)
  pctest.py candidates [--dump DB]        the rules below on VNDB's dump -> candidates.jsonl, candidates.json
  pctest.py fetch      [--workers 2]      the candidates' screenshots (sha256, decoded-pixel sha256)
  pctest.py read       [--procs 5]        MeikiOCR + PP-OCRv6 small at the screenshot's own size -> reads.jsonl
  pctest.py select                        the lines to review (pre-registered caps) -> select.json
  pctest.py sheets     --mode pass1|pass2|check ...   contact-sheet batches (devnative/sheets.py's format)
  pctest.py final                         devnative's development rule -> labels, units, the frozen manifest,
                                          eval/desktop.py's frames file

Everything bulky (screenshots, reads, sheets, decisions) is under $HERD/shared/artifacts/desktop-data/pctest; the
repository holds this code and eval/manifests/desktop-pc-test-v1.jsonl (public VNDB ids, URLs, sha256, boxes, text).

Candidate rules (fixed before any screenshot was downloaded; `candidates` records them in candidates.json):
  * the VN is a Japanese original in VNDB's dump of 2026-10-31 (vndb-db-2026-10-31.tar.zst);
  * the VN is in no pool: no training pool (realplus v1-v3 and its reclaim tier, by every screenshot selected or on
    disk; teacher; realdata; the native pool; the owner's pools and games; the hooked pools v1 and v2.1), no
    development or test set (devnative, eval's manifests and records, eval/linefind, #119's units, textspot7's
    confirmation set, the reservation's test and dev_sample), no scene list (realdata's shots, data/vndb's shots,
    synth2's backgrounds) and no data manifest of the repository (pool_vns);
  * the VN shares no VNDB release with a pool or reserved VN;
  * the screenshot's VNDB flag averages are sexual < 0.3 and violence < 0.5 (the repository's filter);
  * its release is for a PC (win, mac or lin among its platforms) and lists Japanese among its languages;
  * it is at least 640 x 360 with an aspect ratio of 1.25 to 2.4 (4:3 to 21:9 landscape).

Amendment 1 (2026-11-01 07:21Z, after the reads and before any review or label): the first rules also left out every
VN with a family relation (alternative version, original, fan disc, parent, side story, sequel, prequel, same series)
to a pool or reserved VN. With them the review selection held 1,157 candidate lines of 136 VNs, too few for 1,000
labelled dialogue lines after the review's drops. Such VNs are in no pool, so they are kept now and flagged: a row's
"family_of" names the pool or reserved VNs it is related to, and the manifest reports the set with and without them.
The selection caps were raised at the same time (SELECT).

Amendment 2 (2026-11-01 07:48-07:51Z, while pass 1 ran; no label of a line it adds existed): pass 1's first finished
batches skipped about a third of the lines (mostly UI) and marked a tenth as names, so the 1,526 lines would give
about 850 labelled dialogue lines. Two groups the first rules left out are added, each flagged in its rows: (a) VNs
that share a VNDB release with a pool or reserved VN, when the screenshot's own release holds no such VN
("release_partner_of" names them); (b) screenshots of HD console releases (Switch, PS4, PS5, Xbox One / Series; at
least 1280 x 720), which desktop users capture from PC emulators: group "console-hd", reported apart from group
"pc". The selection caps went to 12 screenshots and 40 lines per VN, and a line of 12 px or more at the frame's own
size is kept although it is under 10 px at the 960 x 544 fit scale (this set is read at native size).
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import pickle
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, ROOT / "realplus", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
OUT = Path(os.environ.get("DESKTOP_PCTEST") or ART / "desktop-data" / "pctest")
DUMP = ART / "desktop-data" / "vndb" / "dump" / "db"
DUMP_NAME = "vndb-db-2026-10-31.tar.zst"
UA = "vita-vn-ocr-desktop-data/0.1 (PC test set, low volume; github.com/bee-san/vita-vn-ocr)"
PC_PLATFORMS = {"win", "mac", "lin"}
HD_CONSOLES = {"swi", "ps4", "ps5", "xbo", "xxs"}       # amendment 2: HD console frames (Switch / PS4 emulated on PCs)
AMEND2 = True
SEXUAL_MAX, VIOLENCE_MAX = 0.3, 0.5
MIN_W, MIN_H = 640, 360
ASPECT = (1.25, 2.4)
FAMILY = {"alt", "orig", "fan", "par", "side", "seq", "preq", "ser"}
VN_Q = re.compile(r'"(v\d+)"')
VN_KEY = re.compile(r"\b(v\d+)_\d+")
RULES = {"dump": DUMP_NAME, "olang": "ja", "pc_platforms": sorted(PC_PLATFORMS), "language": "ja in the release's",
         "sexual_max": SEXUAL_MAX, "violence_max": VIOLENCE_MAX, "min_size": [MIN_W, MIN_H], "aspect": list(ASPECT),
         "exclude": ["pool VNs (pool_vns)", "reserved VNs (realplus/shots.reserved_vns)",
                     "VNs on one VNDB release with a pool or reserved VN"],
         "flag": ["family_of: VNs with a family relation to a pool or reserved VN (amendment 1)",
                  "release_partner_of: VNs on one VNDB release with a pool or reserved VN, their screenshot's own "
                  "release holding none (amendment 2)", "group console-hd: HD console releases (amendment 2)"],
         "amendments": [{"n": 1, "time": "2026-11-01T07:21Z", "before": "any review or label",
                         "what": "family-related VNs kept and flagged (they were left out); selection caps 5 -> 8 "
                                 "screenshots and 14 -> 24 lines per VN",
                         "why": "1,157 candidate lines of 136 VNs under the first rules: too few for 1,000 labelled "
                                "dialogue lines after the review's drops"},
                        {"n": 2, "time": "2026-11-01T07:48Z-07:51Z", "before": "any label of a line it adds",
                         "what": "release-partner VNs with a clean own release, HD console screenshots (group "
                                 "console-hd), caps 12 screenshots / 40 lines per VN, lines of 12 px or more at the "
                                 "frame's own size kept when under 10 px at the fit scale",
                         "why": "pass 1's first batches: about a third of the lines skipped (UI), a tenth names: about "
                                "850 dialogue lines expected from 1,526"}]}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


# ----------------------------------------------------------------------------- the pools

def pool_sources(repo=ROOT, art=ART, herd=HERD):
    """(name, kind, path) of every file or directory that names a VN of a pool, set or scene list. kind: 'quoted'
    (VN ids quoted in a text file), 'keys' (also <vn>_<n> keys in it), 'files' (<vn>_<n>.* file names under a
    directory), 'hooked' (the hooked corpus' metadata.json files)."""
    src = []
    q = lambda name, p: src.append((name, "quoted", Path(p)))  # noqa: E731
    k = lambda name, p: src.append((name, "keys", Path(p)))  # noqa: E731
    f = lambda name, p: src.append((name, "files", Path(p)))  # noqa: E731
    for p in sorted((repo / "realplus" / "manifests").rglob("*.json*")):
        q("realplus", p)
    q("realplus", art / "realplus" / "manifests" / "shots.jsonl")
    q("realplus", art / "reserve-policy" / "realplus" / "manifests" / "shots.jsonl")
    f("realplus", art / "realplus" / "shots")
    f("realplus", art / "reserve-policy" / "realplus" / "shots")
    for name in ("teacher", "realdata", "devnative"):
        for p in sorted((repo / name / "manifests").glob("*.json*")):
            k(name, p)
    q("devnative", art / "devnative" / "shots.jsonl")
    f("scenes", art / "realdata" / "shots")
    f("scenes", herd / "shared" / "data" / "vndb" / "shots")
    f("scenes", art / "synth2" / "vndb_psp")
    k("scenes", art / "synth2" / "backgrounds.json")
    for p in sorted((repo / "data" / "manifests").rglob("*.json*")):
        k("data-manifests", p)
    for p in sorted((repo / "eval" / "manifests").glob("*")) + sorted((repo / "eval" / "linefind" / "manifests").glob(
            "*")) + sorted((repo / "eval").glob("*.json")):
        k("eval", p)
    src.append(("dev119", "pickle", art / "vocr5-analysis" / "work" / "units.pkl"))
    src.append(("hooked", "hooked", herd / "shared" / "private" / "hooked-b2"))
    return src


def _walk_vns(o, out):
    if isinstance(o, dict):
        for key, v in o.items():
            if key == "vn" and isinstance(v, str) and re.fullmatch(r"v\d+", v):
                out.add(v)
            else:
                _walk_vns(v, out)
    elif isinstance(o, (list, tuple)):
        for x in o:
            _walk_vns(x, out)


def pool_vns(sources=None):
    """({vn: [source names]}, [per-source record]) over pool_sources() and the owner's VNs."""
    import shots as RS
    out = collections.defaultdict(set)
    rec = []
    for name, kind, path in sources or pool_sources():
        found = set()
        if not path.exists():
            rec.append({"name": name, "kind": kind, "path": str(path), "exists": False})
            continue
        if kind in ("quoted", "keys"):
            text = path.read_text(encoding="utf-8", errors="replace")
            found |= set(VN_Q.findall(text))
            if kind == "keys":
                found |= set(VN_KEY.findall(text))
        elif kind == "files":
            for p in path.rglob("*"):
                m = re.match(r"^(v\d+)_\d+\.(?:jpg|jpeg|png|webp)$", p.name)
                if m:
                    found.add(m.group(1))
        elif kind == "pickle":
            with open(path, "rb") as fh:
                _walk_vns(pickle.load(fh), found)
        elif kind == "hooked":
            for meta in sorted(path.glob("*/*/metadata.json")):
                d = json.loads(meta.read_text(encoding="utf-8"))
                v = (d.get("vndb") or {}).get("vn")
                if v:
                    found.add(v)
                for v2 in (d.get("vndb") or {}).get("vns") or []:
                    found.add(v2)
        for v in found:
            out[v].add(name)
        rec.append({"name": name, "kind": kind, "path": str(path), "vns": len(found)})
    for v in RS.owner_vns():
        out[v].add("owner")
    rec.append({"name": "owner", "kind": "realplus/shots.owner_vns", "vns": len(RS.owner_vns())})
    return {v: sorted(s) for v, s in out.items()}, rec


def cmd_pools(args):
    vns, rec = pool_vns()
    doc = {"kind": "desktop-pctest-pools", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "vns": len(vns), "by_source": dict(collections.Counter(n for s in vns.values() for n in s)),
           "sources": rec, "pool": vns}
    out = Path(args.out or OUT / "pool_vns.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(vns)} pool VNs -> {out}: {doc['by_source']}", flush=True)


# ----------------------------------------------------------------------------- candidates

def load_dump_extra(base):
    """realplus/shots.load_dump plus release engines, resolutions and dates."""
    import shots as RS
    d = RS.load_dump(base)
    eng = {r[0]: r[1] for r in RS.rows("engines", base)}
    d["release"] = {}
    for r in RS.rows("releases", base):
        d["release"][r[0]] = {"engine": eng.get(r[24]) if r[24] != "\\N" else None,
                              "reso": [int(r[5]), int(r[6])] if r[5].isdigit() and r[6].isdigit() else None,
                              "released": r[3]}
    return d


def family_of(d, vns):
    """{vn: [VNs of vns it has a family relation to]} (either direction), vns themselves left out."""
    vns = set(vns)
    out = collections.defaultdict(set)
    for v, rs in d["relations"].items():
        if v in vns:
            for u in rs - vns:
                out[u].add(v)
        else:
            for u in rs & vns:
                out[v].add(u)
    return {v: sorted(s, key=lambda x: int(x[1:])) for v, s in out.items()}


def candidates(d, pool, reserved, family="flag"):
    """(rows, why): one row per screenshot that passes the module's rules, and the count of every reason a
    screenshot was left out (the first that applies). family: 'flag' (amendment 1: keep a VN with a family relation
    to a pool or reserved VN, its row naming them in "family_of") or 'drop'."""
    import shots as RS
    off = set(pool) | set(reserved)
    partners = RS.release_partners(d, off)
    fam = family_of(d, off)
    why = collections.Counter()
    rows = []
    for vn, image, rid in d["shots"]:
        if d["olang"].get(vn) != "ja":
            why["not-japanese-original"] += 1
            continue
        if vn in pool:
            why["pool"] += 1
            continue
        if vn in reserved:
            why["reserved"] += 1
            continue
        own = (d["rel_vns"].get(rid, set()) - {vn}) & off
        if vn in partners and (own or not AMEND2):
            why["shared-release"] += 1
            continue
        if vn in fam and family == "drop":
            why["family"] += 1
            continue
        if image not in d["img"]:
            why["no-image-row"] += 1
            continue
        w, h, sexual, violence = d["img"][image]
        if sexual >= SEXUAL_MAX or violence >= VIOLENCE_MAX:
            why["rating"] += 1
            continue
        plat = sorted(d["plat"].get(rid, ()))
        group = "pc" if set(plat) & PC_PLATFORMS else None
        if group is None and AMEND2 and set(plat) & HD_CONSOLES and w >= 1280 and h >= 720:
            group = "console-hd"
        if group is None:
            why["not-pc"] += 1
            continue
        langs = sorted(d["langs"].get(rid, ()))
        if "ja" not in langs:
            why["no-japanese-release"] += 1
            continue
        if w < MIN_W or h < MIN_H or not ASPECT[0] <= w / h <= ASPECT[1]:
            why["size"] += 1
            continue
        rel = d["release"].get(rid, {})
        titles = d["title"].get(vn) or [vn]
        rows.append({"shot": f"{vn}_{image}", "image": image, "vn": vn, "title": titles[0],
                     "url": RS.url_of(image), "dims": [w, h], "release": rid, "platforms": plat, "langs": langs,
                     "engine": rel.get("engine"), "reso": rel.get("reso"), "released": rel.get("released"),
                     "sexual": sexual, "violence": violence, "family_of": fam.get(vn, []), "group": group,
                     "release_partner_of": sorted(set(v for r in partners.get(vn, ()) for v in d["rel_vns"].get(r, ())
                                                      if v in off), key=lambda x: int(x[1:]))})
    rank = collections.defaultdict(int)
    for r in sorted(rows, key=lambda r: (r["vn"], RS.stable_rank("desktop-pctest", r["image"]))):
        r["rank"] = rank[r["vn"]]
        rank[r["vn"]] += 1
    rows.sort(key=lambda r: (r["rank"], RS.stable_rank("desktop-pctest", r["vn"]), r["image"]))
    for i, r in enumerate(rows):
        r["order"] = i
    return rows, why


def cmd_candidates(args):
    import shots as RS
    base = Path(args.dump or DUMP)
    d = load_dump_extra(base)
    pool_doc = json.loads((OUT / "pool_vns.json").read_text(encoding="utf-8"))
    pool = pool_doc["pool"]
    reserved = RS.reserved_vns(d)
    rows, why = candidates(d, pool, reserved)
    old = {r["shot"]: r for r in read_jsonl(OUT / "candidates.jsonl")}
    for r in rows:
        for key in ("sha256", "pixel_sha256", "bytes", "status", "file_dims"):
            if key in old.get(r["shot"], {}):
                r[key] = old[r["shot"]][key]
    write_jsonl(OUT / "candidates.jsonl", rows)
    stamp = (base.parent / "TIMESTAMP").read_text().strip() if (base.parent / "TIMESTAMP").exists() else None
    summary = {"kind": "desktop-pctest-candidates", "rules": RULES, "dump_timestamp": stamp,
               "pool_vns": len(pool), "pool_vns_sha256": hashlib.sha256(json.dumps(sorted(pool)).encode()).hexdigest(),
               "reserved_vns": len(reserved), "screenshots": len(rows), "vns": len({r["vn"] for r in rows}),
               "dropped": dict(sorted(why.items(), key=lambda kv: -kv[1])),
               "dims": collections.Counter(f"{r['dims'][0]}x{r['dims'][1]}" for r in rows).most_common(40),
               "langs": collections.Counter("+".join(r["langs"]) for r in rows).most_common(20),
               "engines": collections.Counter(str(r["engine"]) for r in rows).most_common(40)}
    (OUT / "candidates.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("pool_vns", "reserved_vns", "screenshots", "vns", "dropped")}),
          flush=True)


# ----------------------------------------------------------------------------- fetch

def shot_path(r):
    return OUT / "shots" / f"{r['shot']}.jpg"


def cmd_fetch(args):
    import requests
    import shots as RS
    rows = read_jsonl(OUT / "candidates.jsonl")
    todo = [r for r in rows if not r.get("sha256") and r.get("status") != "failed"]
    if args.limit:
        todo = todo[:args.limit]
    local = threading.local()
    lock = threading.Lock()
    done = collections.Counter()

    def session():
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.headers["User-Agent"] = UA
        return local.s

    def one(r):
        dest = shot_path(r)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            ok = False
            for attempt in range(5):
                try:
                    resp = session().get(r["url"], timeout=60)
                    if resp.status_code == 429:
                        time.sleep(30 * (attempt + 1))
                        continue
                    if resp.status_code == 404:
                        break
                    resp.raise_for_status()
                    tmp = dest.with_name(dest.name + ".part")
                    tmp.write_bytes(resp.content)
                    tmp.replace(dest)
                    ok = True
                    break
                except requests.RequestException as e:
                    print(f"retry {attempt + 1} {r['url']}: {e}", flush=True)
                    time.sleep(5 * (attempt + 1))
            time.sleep(args.delay)
            if not ok:
                r["status"] = "failed"
                return "failed"
        try:
            import cv2
            bgr = cv2.imread(str(dest), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("undecodable")
            r["sha256"], r["bytes"] = sha256_file(dest), dest.stat().st_size
            r["pixel_sha256"] = RS.pixel_sha256(dest)
            r["file_dims"] = [int(bgr.shape[1]), int(bgr.shape[0])]
            r.pop("status", None)
            return "ok"
        except Exception as e:  # noqa: BLE001 - a truncated or undecodable file
            print(f"bad image {dest}: {e}", flush=True)
            dest.unlink(missing_ok=True)
            r["status"] = "failed"
            return "failed"

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(one, todo), 1):
            done[res] += 1
            if i % 200 == 0:
                with lock:
                    write_jsonl(OUT / "candidates.jsonl", rows)
                print(f"{i}/{len(todo)} {dict(done)} {time.time() - t0:.0f}s", flush=True)
    write_jsonl(OUT / "candidates.jsonl", rows)
    print(f"fetch: {len(todo)} tried, {dict(done)}; {sum(1 for r in rows if r.get('sha256'))} of {len(rows)} hashed",
          flush=True)


# ----------------------------------------------------------------------------- reads (at the screenshot's size)

FIT = (960, 544)          # the Vita screen: line sizes are judged at the scale that fits a frame to it


def fit_scale(w, h):
    return min(FIT[0] / w, FIT[1] / h)


def native_pad(w, h):
    """eval's crop pad (4 px on a 960 x 544 frame) at a frame's own size: 4 / fit scale, at least 4."""
    return max(4, int(round(4 / min(1.0, fit_scale(w, h)))))


def native_margin(w, h):
    """eval's dialogue-region margin (12 px on a 960 x 544 frame) at a frame's own size, at least 12."""
    return max(12, int(round(12 / min(1.0, fit_scale(w, h)))))


_ENG = {}


def _init_engines():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from teacher import engines
    _ENG.update(engines.load(["meiki", "v6s"], n_threads=1))


def read_shot(r, eng=None):
    """devnative/read.py's four readings of every horizontal line, on the screenshot at its own size: MeikiOCR's
    detector (threshold 0.5) and recognizer, teacher/geometry.lines_of (rows merged, ink boxes, furigana); per line
    `meiki` (MeikiOCR on its own box), `meikiv` (MeikiOCR on the wider view), `v6s` (PP-OCRv6 small on the view),
    `v6sc` (PP-OCRv6 small on the crop: ink + furigana + native_pad)."""
    import cv2
    import numpy as np

    from teacher import geometry as G
    eng = eng or _ENG
    bgr = cv2.imread(str(shot_path(r)), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"shot": r["shot"], "error": "unreadable"}
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    H, W = bgr.shape[:2]
    pad = native_pad(W, H)
    t0 = time.perf_counter()
    dets = eng["meiki"].ocr(bgr)
    lines = G.lines_of(dets, rgb)
    out = []
    for k, ln in enumerate(lines):
        x0, y0, x1, y1 = ln["crop"]
        crop = [max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)]
        rec = {"k": k, "det": ln["det"], "det_conf": ln["det_conf"], "parts": ln["parts"], "ink": ln["ink"],
               "crop_ink": ln["crop"], "crop": crop, "view": ln["view"], "edge": ln["edge"], "hint": ln["hint"],
               "reads": {"meiki": [ln["text"], ln["chars"]]}}
        if ln.get("furi"):
            rec["furi"], rec["furi_text"] = ln["furi"], ln["furi_text"]
        vx0, vy0, vx1, vy1 = ln["view"]
        view = bgr[vy0:vy1, vx0:vx1]
        if view.size:
            t, mean, low, chars = eng["meiki"](view)
            rec["reads"]["meikiv"] = [t, mean, low, [[c[0], c[1] + vx0, c[2] + vy0, c[3] + vx0, c[4] + vy0, c[5]]
                                                     for c in chars]]
            rec["reads"]["v6s"] = eng["v6s"](view)
        cx0, cy0, cx1, cy1 = crop
        if cx1 > cx0 and cy1 > cy0:
            rec["reads"]["v6sc"] = eng["v6s"](bgr[cy0:cy1, cx0:cx1])
        out.append(rec)
    return {"shot": r["shot"], "subset": "pc", "vn": r["vn"], "kind": "pc", "url": r["url"],
            "release": r.get("release"), "path": str(shot_path(r)), "dims": [W, H], "pad": pad,
            "fit": round(fit_scale(W, H), 6),
            "dets": [{"det": d["det"], "det_conf": d["det_conf"], "vertical": d["vertical"], "text": d["text"]}
                     for d in dets],
            "lines": out, "secs": round(time.perf_counter() - t0, 3)}


def engine_pins():
    from teacher import engines
    out = {}
    for name in ("meiki_det", "meiki_rec", "v6s"):
        p = engines.model_path(name)
        out[name] = {"file": p.name, "sha256": sha256_file(p), "pinned": engines.PINS[name][2]}
        if out[name]["sha256"] != out[name]["pinned"]:
            sys.exit(f"{p}: sha256 differs from the pin")
    import importlib.metadata as md
    out["packages"] = {k: md.version(k) for k in ("meikiocr", "onnxruntime", "numpy", "opencv-python-headless")}
    return out


def cmd_read(args):
    import multiprocessing as mp
    rows = [r for r in read_jsonl(OUT / "candidates.jsonl") if r.get("sha256")]
    dups = json.loads((OUT / "dups.json").read_text()) if (OUT / "dups.json").exists() else {}
    rows = [r for r in rows if r["shot"] not in dups]
    outd = OUT / "reads"
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "engines.json").write_text(json.dumps(engine_pins(), indent=1) + "\n")
    path = outd / "reads.jsonl"
    done = {r["shot"] for r in read_jsonl(path)}
    todo = [r for r in rows if r["shot"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} screenshots, {len(done)} read, {len(todo)} to read, {args.procs} workers", flush=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo, mp.get_context("spawn").Pool(args.procs,
                                                                               initializer=_init_engines) as pool:
        for i, row in enumerate(pool.imap_unordered(read_shot, todo, chunksize=1)):
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            if i % 50 == 0:
                print(f"{i + 1}/{len(todo)} screenshots, {time.time() - t0:.0f}s", flush=True)
    print(f"read: {len(todo)} screenshots in {time.time() - t0:.0f}s", flush=True)


def cmd_dedupe(args):
    """Pixel duplicates among the candidates (the first in candidate order is kept) and against the frames
    vocr-desktop reads (its dev, PC sample and confirmation frames, by decoded-pixel sha256) -> dups.json."""
    import shots as RS
    rows = [r for r in read_jsonl(OUT / "candidates.jsonl") if r.get("pixel_sha256")]
    seen, out = {}, {}
    ext = {}
    for p in args.against or []:
        for f in sorted(Path(p).rglob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file():
                try:
                    ext[RS.pixel_sha256(f)] = str(f)
                except ValueError:
                    continue
    for r in sorted(rows, key=lambda r: r["order"]):
        h = r["pixel_sha256"]
        if h in ext:
            out[r["shot"]] = f"pixels of {ext[h]}"
        elif h in seen:
            out[r["shot"]] = f"pixels of {seen[h]}"
        else:
            seen[h] = r["shot"]
    (OUT / "dups.json").write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    print(f"dedupe: {len(rows)} screenshots, {len(ext)} external hashes, {len(out)} duplicates", flush=True)


# ----------------------------------------------------------------------------- the lines to review

SELECT = {"max_shots_per_vn": 12, "max_lines_per_vn": 40, "min_kana": 2, "ink_fit": [10, 72], "ink_native_min": 12,
          "order": "candidates.jsonl rank (a stable hash of the image id), then VN order"}
KANA = re.compile(r"[\u3041-\u3096\u30a1-\u30fa]")


def line_ok(s, ln):
    """(ok, why) for a line worth adjudicating: horizontal, not at the screen's edge, ink at most 72 px tall at the
    fit scale (devnative's size rule, judged as the frame would look on a 960 x 544 screen) and at least 10 px there
    or (amendment 2) 12 px at the frame's own size, some reading Japanese."""
    from devnative import sheets as DS
    from teacher import vote
    if ln.get("edge"):
        return False, "screen-edge"
    hn = ln["ink"][3] - ln["ink"][1]
    h = hn * min(1.0, float(s["fit"]))
    if h > SELECT["ink_fit"][1] or (h < SELECT["ink_fit"][0] and hn < SELECT["ink_native_min"]):
        return False, "size"
    if not any(vote.japanese(t) for t in DS.readings(ln).values()):
        return False, "not-japanese"
    return True, None


def shot_japanese(s):
    """A screenshot shows Japanese when some line's reading has at least min_kana kana (a Chinese or English
    screen of a multi-language release has none)."""
    from devnative import sheets as DS
    return any(len(KANA.findall(t or "")) >= SELECT["min_kana"] for ln in s.get("lines", [])
               for t in DS.readings(ln).values())


def select_lines(reads, cands):
    """The review set: per VN, its screenshots in candidate order, a screenshot taken when it shows Japanese and
    has a line worth adjudicating, until max_shots_per_vn screenshots or max_lines_per_vn lines (the screenshot that
    reaches the line cap is taken whole). Returns ({line id: (shot, k)}, per-VN record, why counter)."""
    order = {r["shot"]: (r["rank"], r["order"]) for r in cands}
    by_vn = collections.defaultdict(list)
    for s in reads.values():
        if "lines" in s:
            by_vn[s["vn"]].append(s)
    picked, rec, why = {}, {}, collections.Counter()
    for vn in sorted(by_vn, key=lambda v: int(v[1:])):
        n_shots = n_lines = 0
        for s in sorted(by_vn[vn], key=lambda s: order.get(s["shot"], (1 << 30, 0))):
            if n_shots >= SELECT["max_shots_per_vn"] or n_lines >= SELECT["max_lines_per_vn"]:
                why["vn-cap"] += 1
                continue
            if not shot_japanese(s):
                why["shot-not-japanese"] += 1
                continue
            ok = []
            for ln in s["lines"]:
                good, w = line_ok(s, ln)
                if good:
                    ok.append(ln["k"])
                else:
                    why[f"line-{w}"] += 1
            if not ok:
                why["shot-no-line"] += 1
                continue
            n_shots += 1
            n_lines += len(ok)
            for k in ok:
                picked[f"{s['shot']}_{k}"] = (s["shot"], k)
        if n_shots:
            rec[vn] = {"shots": n_shots, "lines": n_lines}
    return picked, rec, why


def cmd_select(args):
    reads = {r["shot"]: r for r in read_jsonl(OUT / "reads" / "reads.jsonl")}
    cands = read_jsonl(OUT / "candidates.jsonl")
    picked, rec, why = select_lines(reads, cands)
    doc = {"rule": SELECT, "lines": len(picked), "vns": len(rec),
           "shots": len({v[0] for v in picked.values()}), "why": dict(why), "per_vn": rec,
           "ids": sorted(picked)}
    (OUT / "select.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"select: {len(picked)} lines, {doc['shots']} screenshots, {len(rec)} VNs; {dict(why)}", flush=True)


# ----------------------------------------------------------------------------- contact sheets

SHEET_INK = (30, 44)       # a line's ink on a sheet: integer zoom up to 44 px; text taller than 40 px scaled to 30
README_PC = """# Adjudication batch {name} ({mode}, {n} lines, {sheets} sheets)

{intro}

Each sheet `Sxxx.png` (at most 1280 x 1280 px) shows numbered entries. An entry is one text line of a real PC
visual-novel screenshot (VNDB), shown at about its own size (large text scaled down, small text zoomed by an
integer factor): the red frame marks the line's crop; around it is a little of the screen for context. Under the
image are numbered candidate readings. Characters that differ from candidate 1 are red.{engines} `Sxxx.json` lists
the same entries with each candidate's exact text, so you can copy it.

Decide what the framed line says. Label conventions (the benchmark's):
- the main text only: never furigana (small ruby above the line), page-cursor icons (▼ ▽ ◆ ...)
  or button glyphs;
- written as on screen, keeping 「」『』（）, with full-width ！ ？ ～, … for an ellipsis (one … per
  cell of dots), ― for a long dash, no spaces; Latin letters and digits as plain ASCII;
- small kana (ゃ っ ァ), dakuten / handakuten (は ば ぱ), ー versus 一, and similar-looking kanji
  matter: check each one;
- never guess a character you cannot see; mark the line unsure instead;
- a screen in Chinese, Korean or English (multi-language releases) is not Japanese: skip its lines as "bad".

Write `decisions.jsonl` in this directory, one JSON object per entry, for every entry of every sheet:

```
{{"sheet": "S001", "n": 1, "pick": 2}}             candidate 2 is exactly right
{{"sheet": "S001", "n": 2, "text": "「ええと……」"}}    the correct text (when no candidate is exactly right)
{{"sheet": "S001", "n": 3, "unsure": true}}        you cannot settle it from the image
{{"sheet": "S001", "n": 4, "skip": "ui"}}          not dialogue: ui (menus, buttons, system text,
                                                   logos, credits, settings) | cut (the frame cuts the line
                                                   or text runs on outside it) | bad (not one
                                                   readable horizontal Japanese text line)
```

Add `"name": true` when the line is a speaker name (name plate or 【name】). Dialogue, narration, in-story
choices and a speaker's line are what this set keeps. Optional `"note"`.
{default}Do not edit anything else. When every entry has a decision, the batch is done.
"""


def load_frame(s):
    import cv2
    import numpy as np
    bgr = cv2.imread(str(s["path"]), cv2.IMREAD_COLOR)
    return None if bgr is None else np.ascontiguousarray(bgr[:, :, ::-1])


def render_entry(rgb, ln, n, cands, show_engines, fonts, frame):
    """devnative/sheets.render_entry for a frame at its own size: the context (a line height left and right, a third
    above and below) scaled so that the ink is SHEET_INK[0] px when it is taller than 40 px, else zoomed by an
    integer factor towards SHEET_INK[1]."""
    import numpy as np
    from PIL import Image, ImageDraw

    from devnative import sheets as DS
    font, bold, small = fonts
    H, W = rgb.shape[:2]
    cx0, cy0, cx1, cy1 = frame
    ih = max(1, ln["ink"][3] - ln["ink"][1])
    dx, dy = max(4, round(1.0 * ih)), max(3, round(0.35 * ih))
    ctx = [max(0, cx0 - dx), max(0, cy0 - dy), min(W, cx1 + dx), min(H, cy1 + dy)]
    if ih > 40:
        z = SHEET_INK[0] / ih
    else:
        z = float(int(min(3, max(1, round(SHEET_INK[1] / ih)))))
    if (ctx[2] - ctx[0]) * z > DS.SHEET - 20:
        ctx = [max(0, cx0 - 4), ctx[1], min(W, cx1 + 4), ctx[3]]
    while z > 1 and (ctx[2] - ctx[0]) * z > DS.SHEET - 20:
        z -= 1
    if (ctx[2] - ctx[0]) * z > DS.SHEET - 20:
        z = (DS.SHEET - 20) / (ctx[2] - ctx[0])
    patch = Image.fromarray(np.ascontiguousarray(rgb[ctx[1]:ctx[3], ctx[0]:ctx[2]]))
    size = (max(1, round(patch.width * z)), max(1, round(patch.height * z)))
    patch = patch.resize(size, Image.NEAREST if z >= 1 and z == int(z) else Image.LANCZOS)
    d = ImageDraw.Draw(patch)
    d.rectangle([(cx0 - ctx[0]) * z - 1, (cy0 - ctx[1]) * z - 1, (cx1 - ctx[0]) * z, (cy1 - ctx[1]) * z],
                outline=(255, 30, 30), width=1)
    tag_w = 54 if show_engines else 0
    text_w = max([int(font.getlength(x["text"])) for x in cands] + [60])
    w = max(patch.width, 34 + tag_w + text_w) + 8
    h = DS.HEAD_PX + 6 + patch.height + 4 + DS.ROW * max(1, len(cands)) + DS.GAP
    img = Image.new("RGB", (min(DS.SHEET, w), h), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    head = f"#{n}" + ("   ≠" if len(cands) > 1 and show_engines else "")
    dr.text((2, 1), head, font=bold, fill=(20, 20, 160))
    img.paste(patch, (4, DS.HEAD_PX + 6))
    y = DS.HEAD_PX + 6 + patch.height + 4
    base = cands[0]["text"] if cands else ""
    for i, cnd in enumerate(cands, 1):
        dr.text((4, y + 2), f"{i}", font=bold, fill=(0, 0, 0))
        x = 26
        if show_engines:
            tags = "".join(t if e in cnd["engines"] else "·" for e, t in DS.ENGINES)
            dr.text((x, y + 7), tags, font=small, fill=(110, 110, 110))
            x += tag_w
        DS._diff_text(dr, (x, y), cnd["text"], base if i > 1 else cnd["text"], font)
        y += DS.ROW
    if not cands:
        dr.text((26, y), "(no candidate shown: type the text)", font=small, fill=(120, 120, 120))
    dr.line([(0, h - 2), (img.width, h - 2)], fill=(200, 200, 200))
    return img


def make_batch(entries, out, mode, name=None):
    """devnative/sheets.make_batch with render_entry above. entries: [(line id, shot row, line, candidates,
    frame)]."""
    from PIL import Image

    from devnative import sheets as DS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fonts = DS._fonts()
    show = mode == "pass1"
    blocks, cur, rgb = [], None, None
    for n, (lid, s, ln, cands, frame) in enumerate(entries, 1):
        if s["shot"] != cur:
            cur, rgb = s["shot"], load_frame(s)
        blocks.append(render_entry(rgb, ln, n, cands, show, fonts, frame))
    sheets, sizes = DS.pack(blocks)
    index = []
    for si, (members, (w, h)) in enumerate(zip(sheets, sizes), 1):
        sid = f"S{si:03d}"
        page = Image.new("RGB", (max(64, w), max(64, h)), (255, 255, 255))
        rows = []
        for bi, x, y in members:
            lid, s, ln, cands = entries[bi][:4]
            page.paste(blocks[bi], (x, y))
            row = {"n": bi + 1, "id": lid, "candidates": [{"c": i, "text": c["text"]} for i, c in enumerate(cands, 1)]}
            if show:
                for i, c in enumerate(cands):
                    row["candidates"][i]["engines"] = c["engines"]
            rows.append(row)
            index.append({"sheet": sid, "n": bi + 1, "id": lid})
        assert page.width <= DS.SHEET and page.height <= DS.SHEET
        page.save(out / f"{sid}.png", optimize=True)
        (out / f"{sid}.json").write_text(json.dumps({"sheet": sid, "entries": rows}, ensure_ascii=False, indent=1)
                                         + "\n", encoding="utf-8")
    (out / "batch.json").write_text(json.dumps({"mode": mode, "lines": len(entries), "sheets": len(sheets),
                                                "index": index}, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    eng = (" Each candidate also shows which engines read it (M MeikiOCR on its own box, m MeikiOCR on a wider "
           "view, V PP-OCRv6 small on that view, v PP-OCRv6 small on the crop; · = not that engine).") if show else ""
    (out / "README.md").write_text(README_PC.format(name=name or out.name, mode=mode, n=len(entries),
                                                    sheets=len(sheets), intro=DS.INTRO[mode], engines=eng,
                                                    default=DS.DEFAULT if mode == "pass1" else ""),
                                   encoding="utf-8")
    return len(sheets)


def cmd_sheets(args):
    from devnative import sheets as DS
    from teacher import vote
    reads = {r["shot"]: r for r in read_jsonl(OUT / "reads" / "reads.jsonl")}
    idx = DS.line_index(reads)
    if args.mode == "pass1":
        ids = json.loads((OUT / "select.json").read_text())["ids"]
        extra = {}
        done = set()
        for b in batch_dirs([args.out]) if Path(args.out).exists() else []:
            done |= {e["id"] for e in json.loads((b / "batch.json").read_text(encoding="utf-8"))["index"]}
        ids = [i for i in ids if i not in done]
    else:
        spec = json.loads(Path(args.lines).read_text(encoding="utf-8"))
        ids, extra = (spec["ids"], spec.get("extra") or {}) if isinstance(spec, dict) else (spec, {})

    def key(lid):
        s, ln = idx[lid]
        return (int(s["vn"][1:]), s["shot"], ln["ink"][1], ln["ink"][0])
    entries = []
    for lid in sorted(ids, key=key) if args.mode == "pass1" else ids:
        s, ln = idx[lid]
        if args.mode == "check":
            cands = []
        else:
            cands = DS.candidates(ln)
            if args.mode != "pass1":
                cands = [{"text": c["text"], "key": c["key"], "engines": []} for c in cands]
                t = extra.get(lid)
                if t and vote.key(t) not in {c["key"] for c in cands}:
                    cands.append({"text": vote.canon(t), "key": vote.key(t), "engines": []})
                cands = DS.shuffled(cands, lid)
        entries.append((lid, s, ln, cands, ln["crop"]))
    chunk = args.chunk or len(entries)
    first = 1 + len(batch_dirs([args.out])) if Path(args.out).exists() else 1
    for k, i in enumerate(range(0, len(entries), chunk), first):
        sub = Path(args.out) / f"b{k:02d}"
        ns = make_batch(entries[i:i + chunk], sub, args.mode, name=f"{Path(args.out).name}/{sub.name}")
        print(f"{sub}: {len(entries[i:i + chunk])} lines, {ns} sheets", flush=True)


def last_wins(text):
    """decisions.jsonl text with every earlier line of an entry (same sheet and n) dropped: the last line counts."""
    keep, seen = [], set()
    for raw in reversed(text.splitlines()):
        try:
            x = json.loads(raw)
        except ValueError:
            keep.append(raw)
            continue
        key = (x.get("sheet"), x.get("n"), "default" in x) if isinstance(x, dict) else None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        keep.append(raw)
    return "\n".join(reversed(keep)) + ("\n" if keep else "")


def parse_batch(d):
    """devnative/decide.parse_decisions on a batch's decisions.jsonl, the last line of an entry counting."""
    from devnative import decide as DD
    batch, sheets = DD.load_batch(d)
    text = (Path(d) / "decisions.jsonl").read_text(encoding="utf-8")
    return DD.parse_decisions(last_wins(text), sheets, batch["mode"], Path(d).name), batch


def cmd_check(args):
    """Validate a batch's decisions.jsonl (devnative/decide.parse_decisions; the last line of an entry counts);
    list the sheets still to decide."""
    from devnative import decide as DD
    d = Path(args.batch)
    batch, sheets = DD.load_batch(d)
    path = d / "decisions.jsonl"
    text = last_wins(path.read_text(encoding="utf-8")) if path.exists() else ""
    decided = set()
    for raw in text.splitlines():
        try:
            x = json.loads(raw)
        except ValueError:
            continue
        if isinstance(x, dict) and x.get("sheet") in sheets:
            if "default" in x:
                decided |= {(x["sheet"], n) for n, e in sheets[x["sheet"]].items() if len(e["candidates"]) == 1}
            elif "n" in x:
                decided.add((x["sheet"], x["n"]))
    todo = [sid for sid in sorted(sheets) if any((sid, n) not in decided for n in sheets[sid])]
    try:
        recs = DD.parse_decisions(text, sheets, batch["mode"], d.name)
    except DD.DecisionError as e:
        errs = [x for x in str(e).splitlines() if "entries without a decision" not in x]
        if errs:
            print("PROBLEMS:\n" + "\n".join(errs), flush=True)
        print(f"NOT FINISHED: sheets to decide {todo[:12]}{' ...' if len(todo) > 12 else ''}" if todo else
              "NOT FINISHED", flush=True)
        return
    print(f"OK: {len(recs)} decisions {dict(collections.Counter(r['decision'] for r in recs))}; batch complete",
          flush=True)


# ----------------------------------------------------------------------------- decisions -> labels -> the frozen set

def batch_dirs(dirs):
    out = []
    for d in dirs:
        d = Path(d)
        out += [d] if (d / "batch.json").exists() else sorted(x for x in d.glob("b*") if (x / "batch.json").exists())
    return out


def collect(dirs):
    """{line id: parsed decision} over batch directories (each fully decided; the last line of an entry counts)."""
    out = {}
    for d in batch_dirs(dirs):
        recs, _ = parse_batch(d)
        for r in recs:
            r["batch"] = f"{d.parent.name}/{d.name}"
            out[r["id"]] = r
    return out


def cmd_queue2(args):
    """The independent second pass: every pass-1 label that both engine families do not read (devnative's
    single and changed labels), with pass 1's label as an extra, unmarked candidate."""
    from devnative import decide as DD
    from devnative import sheets as DS
    reads = {r["shot"]: r for r in read_jsonl(OUT / "reads" / "reads.jsonl")}
    idx = DS.line_index(reads)
    p1 = collect(args.pass1)
    ids, extra, why = [], {}, collections.Counter()
    for lid, r in sorted(p1.items()):
        if DD.needs_pass2(r, idx[lid][1]):
            ids.append(lid)
            extra[lid] = r["text"]
            why[DD.classify(r["text"], idx[lid][1])[0]] += 1
    Path(args.out).write_text(json.dumps({"ids": ids, "extra": extra}, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    print(f"pass 2: {len(ids)} lines {dict(why)} of {len(p1)} pass-1 decisions -> {args.out}", flush=True)


def provisional(args):
    """{line id: decide_line record} from pass 1 and pass 2 (and third looks)."""
    from devnative import decide as DD
    from devnative import sheets as DS
    reads = {r["shot"]: r for r in read_jsonl(OUT / "reads" / "reads.jsonl")}
    idx = DS.line_index(reads)
    ids = json.loads((OUT / "select.json").read_text())["ids"]
    p1, p2 = collect(args.pass1), collect(args.pass2 or [])
    res = {}
    for f in args.resolve or []:
        for r in read_jsonl(f):
            res[r["id"]] = r
    out = {}
    for lid in ids:
        s, ln = idx[lid]
        out[lid] = DD.decide_line(lid, ln, p1.get(lid), p2.get(lid), "dev", resolution=res.get(lid))
    return out, idx, reads


def cmd_checksample(args):
    """The hand-checked subset: up to `per_vn` kept lines of every VN, then random kept lines up to n (seeded)."""
    import random
    labels, _, _ = provisional(args)
    kept = sorted(lid for lid, r in labels.items() if r["status"] == "kept")
    rng = random.Random(args.seed)
    by_vn = collections.defaultdict(list)
    for lid in kept:
        by_vn[lid.split("_")[0]].append(lid)
    pick = []
    for vn in sorted(by_vn, key=lambda v: int(v[1:])):
        pick += rng.sample(by_vn[vn], min(args.per_vn, len(by_vn[vn])))
    rest = sorted(set(kept) - set(pick))
    rng.shuffle(rest)
    pick = sorted(set(pick + rest[:max(0, args.n - len(pick))]))
    if len(pick) > args.n:
        pick = sorted(rng.sample(pick, args.n))
    Path(args.out).write_text(json.dumps({"ids": pick, "seed": args.seed, "from": len(kept)}, indent=1) + "\n")
    print(f"hand-check sample: {len(pick)} of {len(kept)} kept lines ({len(by_vn)} VNs) -> {args.out}", flush=True)


def units_of(s, kept, others, W, H):
    """Dialogue units of one screenshot: its kept non-name lines grouped as a player draws a region (eval's groups
    over the text boxes: a gap of at most 1.6 line heights), each region their union plus native_margin. A unit is
    clean when no other text the reads found (a line not kept, a name, a raw detector box) has its centre inside the
    region; only clean units are page-path units."""
    from devnative import lines as DL
    order = sorted(kept, key=lambda r: (r["text_box"][1], r["text_box"][0]))
    groups = []
    for r in order:
        b = r["text_box"]
        if groups:
            g = groups[-1]
            bottom = max(x["text_box"][3] for x in g)
            hmax = max(x["text_box"][3] - x["text_box"][1] for x in g + [r])
            if b[1] - bottom <= DL.GROUP_GAP * hmax:
                g.append(r)
                continue
        groups.append([r])
    m = native_margin(W, H)
    out = []
    for g in groups:
        u = [min(x["text_box"][0] for x in g), min(x["text_box"][1] for x in g),
             max(x["text_box"][2] for x in g), max(x["text_box"][3] for x in g)]
        region = [max(0, u[0] - m), max(0, u[1] - m), min(W, u[2] + m), min(H, u[3] + m)]
        dirty = []
        for o in others:
            cx, cy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
            if region[0] <= cx < region[2] and region[1] <= cy < region[3] and \
                    not any(DL.area(DL.inter(o, x["text_box"])) >= 0.5 * DL.area(o) for x in g):
                dirty.append(o)
        out.append({"region": region, "lines": g, "clean": not dirty, "dirt": len(dirty)})
    return out


def even_cells(box, text):
    sys.path.insert(0, str(ROOT / "eval"))
    import metrics
    x, y, w, h = box
    n = max(1, len(metrics.norm(text)))
    return [[x + k * w / n, y, x + (k + 1) * w / n, y + h] for k in range(n)]


def cmd_final(args):
    """Labels (devnative's development rule, then the hand-check: a kept label that the blind hand check reads
    differently is dropped unless a third look keeps it), dialogue units, the committed manifest and summary, and
    eval/desktop.py's frames file (private)."""
    from teacher import geometry as G
    from teacher import vote
    labels, idx, reads = provisional(args)
    hc = collect(args.check or [])
    cands = {r["shot"]: r for r in read_jsonl(OUT / "candidates.jsonl")}
    third = {}
    for f in args.resolve or []:
        for r in read_jsonl(f):
            third[r["id"]] = r
    stats = collections.Counter()
    rows = []
    for lid, rec in sorted(labels.items(), key=lambda kv: kv[0]):
        s, ln = idx[lid]
        stats[f"{rec['status']}:{rec['how'] if rec['status'] == 'kept' else rec['why']}"] += 1
        if rec["status"] != "kept":
            continue
        looks = 1 + int(bool(rec.get("pass2") and rec["pass2"].get("agree")))
        h = hc.get(lid)
        hand = None
        if h is not None:
            agree = h["decision"] in ("pick", "text") and vote.key(h["text"] or "") == vote.key(rec["label"])
            hand = {"agree": agree, "decision": h["decision"], "batch": h["batch"]}
            stats[f"hand:{'agree' if agree else 'disagree'}"] += 1
            if agree:
                looks += 1
            elif not (third.get(lid) or {}).get("keep"):
                stats["dropped:hand-disagree"] += 1
                continue
        c = cands[s["shot"]]
        W, H = s["dims"]
        ink = ln["ink"]
        rows.append({"id": lid, "vn": s["vn"], "title": c["title"], "release": c["release"], "engine": c["engine"],
                     "reso": c["reso"], "platforms": c["platforms"], "langs": c["langs"], "shot": s["shot"],
                     "url": c["url"], "sha256": c["sha256"], "pixel_sha256": c["pixel_sha256"], "dims": [W, H],
                     "box": [ink[0], ink[1], ink[2] - ink[0], ink[3] - ink[1]],
                     "text_box": list(ln["crop_ink"]), "crop": list(ln["crop"]), "furigana": bool(ln.get("furi")),
                     "text": rec["label"], "name": bool(rec["name"]), "how": rec["how"], "support": rec["support"],
                     "looks": looks, "reviewer_kind": "assistant", "hand_checked": hand,
                     "polarity": G.polarity(ln.get("hint") or {}), "family_of": c.get("family_of") or [],
                     "release_partner_of": c.get("release_partner_of") or [], "group": c.get("group") or "pc",
                     "pass1": rec["pass1"], "pass2": rec["pass2"]})
    # units
    by_shot = collections.defaultdict(list)
    for r in rows:
        by_shot[r["shot"]].append(r)
    frames = []
    n_units = n_clean = 0
    for shot, rs in sorted(by_shot.items()):
        s = reads[shot]
        W, H = s["dims"]
        kept_k = {int(r["id"].rsplit("_", 1)[1]) for r in rs}
        others = [ln["crop_ink"] for ln in s["lines"] if ln["k"] not in kept_k]
        others += [r["text_box"] for r in rs if r["name"]]
        others += [d["det"] for d in s.get("dets", [])]
        dlg = [dict(r, text_box=r["text_box"]) for r in rs if not r["name"]]
        us = units_of(s, dlg, others, W, H)
        units = []
        for k, u in enumerate(us):
            n_units += 1
            uid = f"pctest:{shot}:{k}"
            for r in u["lines"]:
                r2 = next(x for x in rows if x["id"] == r["id"])
                r2["unit"], r2["unit_clean"] = uid, u["clean"]
            if not u["clean"]:
                continue
            n_clean += 1
            units.append({"uid": uid, "cluster": s["vn"], "group": cands[shot].get("group") or "pc",
                          "region": u["region"],
                          "lines": [{"text": r["text"], "box": [float(v) for v in r["box"]],
                                     "cells": even_cells(r["box"], r["text"])} for r in u["lines"]]})
        if units:
            c = cands[shot]
            frames.append({"key": shot, "set": "pctest", "view": "pc", "cluster": s["vn"], "W": W, "H": H,
                           "src": {"kind": "pctest", "path": str(shot_path(c)), "sha256": c["sha256"]},
                           "units": units})
    for r in rows:
        r.setdefault("unit", None)
        r.setdefault("unit_clean", None)
    dlg_rows = [r for r in rows if not r["name"]]
    summary = {
        "kind": "desktop-pc-test", "version": 1, "issue": 169, "created": SH_utc(),
        "rules": RULES, "select": SELECT, "label_rule": "devnative/decide.decide_line mode dev (a consensus label "
        "after pass 1; a single or changed label only when the independent pass 2 reads the same key), then the "
        "blind hand check of a subset (a disagreement drops the label unless a third look keeps it)",
        "reviewer_kind": "assistant",
        "lines": len(rows), "dialogue_lines": len(dlg_rows), "name_lines": len(rows) - len(dlg_rows),
        "dialogue_chars": sum(len(r["text"]) for r in dlg_rows),
        "vns": len({r["vn"] for r in dlg_rows}), "screenshots": len({r["shot"] for r in dlg_rows}),
        "units": n_units, "clean_units": n_clean, "frames": len(frames),
        "clean_unit_lines": sum(len(u["lines"]) for f in frames for u in f["units"]),
        "hand_checked": sum(1 for r in rows if r["hand_checked"]),
        "hand_check_agree": sum(1 for r in rows if r["hand_checked"] and r["hand_checked"]["agree"]),
        "family_flagged_vns": len({r["vn"] for r in dlg_rows if r["family_of"]}),
        "family_flagged_lines": sum(1 for r in dlg_rows if r["family_of"]),
        "partner_flagged_vns": len({r["vn"] for r in dlg_rows if r["release_partner_of"]}),
        "partner_flagged_lines": sum(1 for r in dlg_rows if r["release_partner_of"]),
        "by_group": {g: {"dialogue_lines": sum(1 for r in dlg_rows if r["group"] == g),
                         "vns": len({r["vn"] for r in dlg_rows if r["group"] == g}),
                         "clean_unit_lines": sum(len(u["lines"]) for f in frames for u in f["units"]
                                                 if u["group"] == g)}
                     for g in sorted({r["group"] for r in dlg_rows})},
        "by_status": dict(stats),
        "dims": collections.Counter(f"{r['dims'][0]}x{r['dims'][1]}" for r in dlg_rows).most_common(),
        "engines": collections.Counter(str(r["engine"]) for r in dlg_rows).most_common(),
        "how": dict(collections.Counter(r["how"] for r in dlg_rows)),
    }
    man = Path(args.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary["manifest"] = str(man.relative_to(ROOT)) if man.is_relative_to(ROOT) else str(man)
    summary["manifest_sha256"] = sha256_file(man)
    fp = Path(args.frames)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "wb") as f:
        pickle.dump({"frames": frames, "meta": {"set": "pctest", "manifest_sha256": summary["manifest_sha256"],
                                                "units": n_clean, "schema": "vocr-desktop/1 frames"}}, f, protocol=4)
    summary["frames_file_sha256"] = sha256_file(fp)
    Path(args.summary).write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("lines", "dialogue_lines", "vns", "screenshots", "clean_units",
                                              "clean_unit_lines", "hand_checked", "hand_check_agree",
                                              "manifest_sha256", "frames_file_sha256")}), flush=True)
    print(json.dumps(summary["by_status"]), flush=True)


def SH_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cmd_reserve(args):
    """Reserve the frozen set's VNs in eval/manifests/reserved.json ("desktop_pc_test"): synth/common.is_reserved_vn
    and realplus/shots.reserved_vns keep them out of every later pool (training, scenes, names, calibration)."""
    rows = read_jsonl(args.manifest)
    vns = sorted({r["vn"] for r in rows}, key=lambda v: int(v[1:]))
    path = Path(args.out)
    doc = json.loads(path.read_text(encoding="utf-8"))
    clash = sorted(set(vns) & (set(doc["test"]) | set(doc["dev_sample"])))
    if clash:
        raise SystemExit(f"test or dev_sample VNs in the PC test set: {clash}")
    doc["desktop_pc_test"] = {v: "desktop-pc-test" for v in vns}
    doc["desktop_pc_test_record"] = {
        "rule": "the frozen desktop PC test set (vocr-desktop #169, docs/desktop-data.md): never train, tune, "
                "calibrate, select or label-tune on these VNs, and never render synthetic scenes or names from them",
        "manifest": str(Path(args.manifest).resolve().relative_to(ROOT)), "manifest_sha256": sha256_file(args.manifest),
        "vns": len(vns), "code": "desktopdata/pctest.py reserve"}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{path}: {len(vns)} desktop PC test VNs reserved", flush=True)


def cmd_zoom(args):
    """An enlarged view of one entry (its context at about `ink` px of ink, at most 1280 px wide) -> PNG."""
    from PIL import Image

    from devnative import sheets as DS
    d = Path(args.batch)
    sheet = json.loads((d / f"{args.sheet}.json").read_text(encoding="utf-8"))
    entry = next((e for e in sheet["entries"] if e["n"] == args.n), None)
    if entry is None:
        raise SystemExit(f"no entry {args.n} on {args.sheet}")
    reads = {r["shot"]: r for r in read_jsonl(OUT / "reads" / "reads.jsonl")}
    s, ln = DS.line_index(reads)[entry["id"]]
    rgb = load_frame(s)
    H, W = rgb.shape[:2]
    cx0, cy0, cx1, cy1 = ln["crop"]
    ih = max(1, ln["ink"][3] - ln["ink"][1])
    ctx = [max(0, cx0 - ih // 2), max(0, cy0 - ih // 3), min(W, cx1 + ih // 2), min(H, cy1 + ih // 3)]
    z = min(args.ink / ih, 1260 / max(1, ctx[2] - ctx[0]), 1260 / max(1, ctx[3] - ctx[1]))
    im = Image.fromarray(rgb[ctx[1]:ctx[3], ctx[0]:ctx[2]])
    im = im.resize((max(1, round(im.width * z)), max(1, round(im.height * z))),
                   Image.NEAREST if z >= 2 else Image.LANCZOS)
    out = d / f"zoom_{args.sheet}_{args.n}.png"
    im.save(out)
    print(f"ZOOM {out} ({im.width}x{im.height}, x{z:.2f})", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pools")
    p.add_argument("--out")
    p = sub.add_parser("candidates")
    p.add_argument("--dump")
    p = sub.add_parser("fetch")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("dedupe")
    p.add_argument("--against", nargs="*", help="directories of frames the test set must not repeat")
    p = sub.add_parser("read")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    sub.add_parser("select")
    p = sub.add_parser("sheets")
    p.add_argument("--mode", choices=("pass1", "pass2", "blind", "check"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lines", help="pass2 / blind / check: JSON list of line ids or {ids, extra}")
    p.add_argument("--chunk", type=int, default=0)
    p = sub.add_parser("check")
    p.add_argument("--batch", required=True)
    p = sub.add_parser("zoom")
    p.add_argument("--batch", required=True)
    p.add_argument("--sheet", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--ink", type=float, default=80.0)
    p = sub.add_parser("queue2")
    p.add_argument("--pass1", nargs="+", required=True)
    p.add_argument("--out", required=True)
    for name in ("checksample", "final"):
        p = sub.add_parser(name)
        p.add_argument("--pass1", nargs="+", required=True)
        p.add_argument("--pass2", nargs="*")
        p.add_argument("--resolve", nargs="*", help="third looks: JSONL {id, keep, text?, why}")
        if name == "checksample":
            p.add_argument("--n", type=int, default=200)
            p.add_argument("--per-vn", type=int, default=1)
            p.add_argument("--seed", type=int, default=20261101)
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--check", nargs="*", help="the hand-check batches")
            p.add_argument("--manifest", default=str(ROOT / "eval" / "manifests" / "desktop-pc-test-v1.jsonl"))
            p.add_argument("--summary", default=str(ROOT / "eval" / "desktop_pc_test_v1.json"))
            p.add_argument("--frames", default=str(OUT / "frames_pctest_v1.pkl"))
    p = sub.add_parser("reserve")
    p.add_argument("--manifest", default=str(ROOT / "eval" / "manifests" / "desktop-pc-test-v1.jsonl"))
    p.add_argument("--out", default=str(ROOT / "eval" / "manifests" / "reserved.json"))
    args = ap.parse_args(argv)
    cmds = {"reserve": cmd_reserve, "pools": cmd_pools, "candidates": cmd_candidates, "fetch": cmd_fetch,
            "dedupe": cmd_dedupe, "read": cmd_read, "select": cmd_select, "sheets": cmd_sheets, "check": cmd_check,
            "zoom": cmd_zoom, "queue2": cmd_queue2, "checksample": cmd_checksample, "final": cmd_final}
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()

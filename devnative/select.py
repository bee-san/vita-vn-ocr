#!/usr/bin/env python3
"""Select the screenshots of the native development cohort (devnative).

  select.py list                 fresh VNDB listing of the reserved VNs + CLANNAD's relations -> $OUT/vndb/
  select.py refs                 every screenshot the cohort must avoid or flag, hashed -> $OUT/refs/   (heavy.sh)
  select.py select [--no-fetch]  rules -> fetch (local mirror, else VNDB) -> pixel/near-duplicate checks
                                 -> devnative/manifests/shots.jsonl + $OUT/select_report.json

Only native console screenshots count: a psv release at 960x544 (Vita) or a psp release at
480x272 (PSP), the release Japanese-only, VNDB flags sexual < 0.3 and violence < 0.5 (the
repo's filters). Only VNs reserved from synth3's names and scenes may appear
($HERD/shared/artifacts/synth3/mining/exclude_dev_vns.txt), and of those never:
  * a test VN: eval/manifests/test_games.json "all", or synth/common.is_test_vn(),
  * a title of the fresh native textbox test (eval/fresh_native_source_provenance_20261012.json:
    PSP v14909 v25459 v29454, Vita v21289 v21976 v20519, reserves v16509 v19140) or any VN VNDB
    relates to one of them (any relation kind), protected like vita-test and psp-test,
  * CLANNAD (v4) or any VN VNDB relates to it (any relation kind),
  * a VN of any training label pool: teacher_train, teacher_distill_train, real_train,
    real_unlabelled, realdata's and teacher's training-split screenshots, the FL10 native
    training sources (data/manifests/fl10_native_training_20261011.json).
No screenshot may be (by VNDB id, file sha256, decoded-pixel sha256, or the same frame
re-encoded: thumbnail correlation >= 0.95 and a full-size grey p99 difference < 48):
  * in eval's sets (vita-test, psp-test) or from one of their capture sessions, or one of the
    fresh native textbox test's source images (its file sha256 or RGB pixel sha256),
  * a training label screenshot, or from a release (capture session) one was taken from.
Screenshots of the repo's other sets (linefind's dev, holdout, pool and research sets, the
general-VN development sources) are kept out of strict and scene_vn; in tv_native they stay,
flagged (linefind, general_vn_dev).
Every screenshot is in exactly one subset (the conductor's decision of 2026-10-12):
  tv_native  a native screenshot teacher_val already labels: the same lines through the
             console's path (GRAD finder, ink + 4/2 px, console polarity), relabelled here.
             Scene-background images are allowed here and flagged (bg_shot).
  scene_vn   a VN whose other screenshots were scene backgrounds (synth2's list, data/vndb/
             shots, synth2's PSP shots, fresh_jiten_backgrounds), the background images
             themselves excluded; never a teacher_val screenshot.
  strict     a VN in no training pool at all (labels or scenes); never a teacher_val screenshot.
Near-duplicates that are not the same frame (thumbnail correlation >= 0.95: usually the same
scene with other dialogue) are kept and flagged with their partner; against eval's sets they
are dropped.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
from common import is_test_vn  # noqa: E402
from devnative import paths as P  # noqa: E402

API = "https://api.vndb.org/kana/vn"
FIELDS = ("id,title,alttitle,screenshots.id,screenshots.url,screenshots.dims,screenshots.sexual,"
          "screenshots.violence,screenshots.votecount,screenshots.release.id,screenshots.release.platforms,"
          "screenshots.release.languages.lang")
REL_FIELDS = "id,title,relations.id,relations.title,relations.relation,relations.relation_official"
CLANNAD = "v4"
# The fresh native textbox test's titles (the conductor's hygiene update of 2026-10-12, PR #49):
# protected exactly like vita-test / psp-test, with their VNDB relations.
FRESH_NATIVE = ("v14909", "v25459", "v29454", "v21289", "v21976", "v20519", "v16509", "v19140")
FRESH_PROVENANCE = ROOT / "eval" / "fresh_native_source_provenance_20261012.json"
SEXUAL_MAX, VIOLENCE_MAX = 0.3, 0.5
NEAR = 0.95          # thumbnail correlation of a near-duplicate (eval/tools/leakcheck.py's bar)
SAME_P99 = 48        # full-size grey p99 |difference| under this: the same frame re-encoded
SUBSETS = ("strict", "tv_native", "scene_vn")
RESERVED = P.ART / "synth3" / "mining" / "exclude_dev_vns.txt"


# ----------------------------------------------------------------------------- VNDB listing

def post(session, body, url=API):
    for attempt in range(8):
        r = session.post(url, json=body, timeout=120)
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json(), r.content
    raise RuntimeError("VNDB kept rate-limiting")


def reserved_vns(path=RESERVED):
    """The reserved VNs (synth3's exclusion list); empty where the shared file is absent (CI)."""
    if not Path(path).exists():
        return []
    return sorted({x.strip() for x in Path(path).read_text().split() if x.strip()}, key=lambda v: int(v[1:]))


def cmd_list(args):
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = P.UA
    vns = reserved_vns(args.reserved)
    out = P.OUT / "vndb"
    out.mkdir(parents=True, exist_ok=True)
    rows, titles, resp_hash, queries = [], {}, hashlib.sha256(), []
    for i in range(0, len(vns), 50):
        chunk = vns[i:i + 50]
        page = 1
        while True:
            body = {"filters": ["or", *[["id", "=", v] for v in chunk]], "fields": FIELDS, "results": 100,
                    "page": page, "sort": "id"}
            queries.append(body)
            d, raw = post(s, body)
            resp_hash.update(raw)
            for vn in d["results"]:
                titles[vn["id"]] = {"title": vn["title"], "alttitle": vn.get("alttitle")}
                for sh in vn.get("screenshots") or []:
                    r = sh.get("release") or {}
                    rows.append({"vn": vn["id"], "id": sh.get("id") or P.image_id(sh["url"]), "url": sh["url"],
                                 "dims": sh["dims"], "sexual": sh["sexual"], "violence": sh["violence"],
                                 "flag_votes": sh.get("votecount"), "release": r.get("id"),
                                 "platforms": sorted(r.get("platforms") or []),
                                 "langs": sorted({x["lang"] for x in r.get("languages") or []})})
            print(f"VNs {i + 1}-{i + len(chunk)} page {page}: {len(rows)} screenshots so far", flush=True)
            time.sleep(1.5)
            if not d.get("more"):
                break
            page += 1
    body = {"filters": ["id", "=", CLANNAD], "fields": REL_FIELDS}
    queries.append(body)
    d, raw = post(s, body)
    resp_hash.update(raw)
    rel = d["results"][0].get("relations") or []
    family = {"vn": CLANNAD, "title": d["results"][0]["title"],
              "relations": [{"id": x["id"], "title": x["title"], "relation": x["relation"],
                             "official": x.get("relation_official")} for x in rel]}
    time.sleep(1.5)
    body = {"filters": ["or", *[["id", "=", v] for v in FRESH_NATIVE]], "fields": REL_FIELDS, "results": 100}
    queries.append(body)
    d, raw = post(s, body)
    resp_hash.update(raw)
    fresh = {}
    for x in d["results"]:
        fresh[x["id"]] = {"title": x["title"],
                          "relations": [{"id": y["id"], "title": y["title"], "relation": y["relation"],
                                         "official": y.get("relation_official")} for y in x.get("relations") or []]}
    missing = set(FRESH_NATIVE) - set(fresh)
    if missing:
        sys.exit(f"VNDB returned no entry for fresh-test titles {sorted(missing)}")
    rows.sort(key=lambda r: (int(r["vn"][1:]), P.image_num(r["url"])))
    (out / "listing.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                       encoding="utf-8")
    meta = {"api": API, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reserved": len(vns),
            "reserved_sha256": P.file_sha256(args.reserved), "vns_returned": len(titles),
            "screenshots": len(rows), "queries_sha256": hashlib.sha256(
                json.dumps(queries, sort_keys=True).encode()).hexdigest(),
            "responses_sha256": resp_hash.hexdigest(), "titles": titles, "clannad_family": family,
            "fresh_native_family": fresh}
    (out / "listing_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} screenshots of {len(titles)} VNs; CLANNAD relations {len(rel)} -> {out}", flush=True)


# ----------------------------------------------------------------------------- references

def _jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _npz_vns(path):
    import numpy as np
    if not Path(path).exists():
        return set()
    return {m["vn"] for m in json.loads(str(np.load(path)["meta"]))}


def label_pool_vns():
    """VNs of every training label pool (see the module docstring), each with its sources."""
    A, R = P.ART, P.ART / "realdata"
    out = defaultdict(set)
    for name, path in (("teacher_train", A / "teacher" / "teacher_train.npz"),
                       ("teacher_distill_train", A / "teacher" / "teacher_distill_train.npz"),
                       ("real_train", R / "real_train.npz"), ("real_unlabelled", R / "real_unlabelled.npz")):
        for v in _npz_vns(path):
            out[v].add(name)
    for name, path in (("realdata_shots_train", ROOT / "realdata" / "manifests" / "games.json"),
                       ("teacher_shots_train", ROOT / "teacher" / "manifests" / "games.json")):
        if path.exists():
            for v, g in json.loads(path.read_text(encoding="utf-8"))["games"].items():
                if g.get("split") == "train":
                    out[v].add(name)
    p = ROOT / "data" / "manifests" / "fl10_native_training_20261011.json"
    if p.exists():
        for r in json.loads(p.read_text(encoding="utf-8")).get("source_images", []):
            out[r["vn"]].add("fl10_native_training")
    return out


def reference_rows():
    """Every screenshot the cohort must avoid or flag: {id, url, sha256, vn, release, roles, path}.
    roles: eval-test, eval-linefind, eval-general-vn, teacher-val, label-train, scene."""
    A, D = P.ART, P.SHARED_DATA
    rows = defaultdict(lambda: {"roles": set(), "paths": set(), "sha256": set(), "releases": set()})

    def add(role, ref, vn=None, url=None, sha=None, release=None, path=None):
        num = P.image_num(url or ref)
        r = rows[num]
        r["roles"].add(role)
        r["vn"] = r.get("vn") or vn or str(ref).rsplit("/", 1)[-1].split("_")[0]
        r["url"] = url or r.get("url") or P.vndb_url(num)
        if sha:
            r["sha256"].add(sha)
        if release:
            r["releases"].add(release)
        if path and Path(path).exists():
            r["paths"].add(str(path))

    ev = A / "eval"
    for which in ("vita", "psp"):
        for x in _jsonl(ROOT / "eval" / "manifests" / f"{which}-test.jsonl"):
            name = f"{x['vn']}_{x['url'].rsplit('/', 1)[-1]}"
            add("eval-test", x["url"], x["vn"], x["url"], x["sha256"], x.get("session"),
                ev / which / "shots" / name)
    for f in sorted((ROOT / "eval" / "linefind" / "manifests").glob("*.jsonl")):
        for x in _jsonl(f):
            if "url" not in x:
                continue
            shot = x.get("shot") or f"{x['vn']}_{x['url'].rsplit('/', 1)[-1]}"
            path = (D / "vndb" / "shots" / shot) if x.get("platform") == "psv" else \
                (A / "linefind" / "psp_dev" / "shots" / shot)
            add("eval-linefind", x["url"], x["vn"], x["url"], x.get("sha256"), None, path)
    p = ROOT / "data" / "manifests" / "general_vn_development_sources_20261011.json"
    if p.exists():
        for x in json.loads(p.read_text(encoding="utf-8")).get("records", []):
            add("eval-general-vn", x["url"], x["vn"], x["url"], x.get("sha256"), x.get("release"))
    # realdata's screenshot collection: teacher_val's screenshots, and every training-split one
    rd_games = json.loads((ROOT / "realdata" / "manifests" / "games.json").read_text(encoding="utf-8"))["games"]
    tv_shots = teacher_val_shots()
    for kind in ("vita", "psp", "hires"):
        for x in _jsonl(ROOT / "realdata" / "manifests" / f"shots_{kind}.jsonl"):
            key = P.shot_key(x["vn"], x["url"])
            path = A / "realdata" / "shots" / kind / f"{x['vn']}_{x['url'].rsplit('/', 1)[-1]}"
            if key in tv_shots:
                add("teacher-val", x["url"], x["vn"], x["url"], x.get("sha256"), x.get("release"), path)
            if rd_games.get(x["vn"], {}).get("split") == "train":
                add("label-train", x["url"], x["vn"], x["url"], x.get("sha256"), x.get("release"), path)
    for x in _jsonl(ROOT / "teacher" / "manifests" / "shots.jsonl"):
        if x.get("split") == "train":
            add("label-train", x["url"], x["vn"], x["url"], x.get("sha256"), x.get("release"),
                A / "realdata" / "shots" / x["kind"] / f"{x['vn']}_{x['url'].rsplit('/', 1)[-1]}")
    p = ROOT / "data" / "manifests" / "fl10_native_training_20261011.json"
    if p.exists():
        for x in json.loads(p.read_text(encoding="utf-8")).get("source_images", []):
            add("label-train", x["url"], x["vn"], x["url"], x.get("sha256"), x.get("release"))
    # scene backgrounds of synthetic training data
    bg = json.loads((A / "synth2" / "backgrounds.json").read_text(encoding="utf-8"))
    for name in bg.get("vita_shots", []):
        add("scene", name, path=D / "vndb" / "shots" / name)
    for name in bg.get("psp_shots", []):
        add("scene", name, path=A / "synth2" / "vndb_psp" / "shots" / name)
    for f in sorted((D / "vndb" / "shots").glob("*.jpg")):
        add("scene", f.name, path=f)
    for x in _jsonl(ROOT / "data" / "manifests" / "synth2_psp_shots.jsonl"):
        name = f"{x['vn']}_{x['url'].rsplit('/', 1)[-1]}"
        add("scene", x["url"], x["vn"], x["url"], x.get("sha256"), x.get("release"),
            A / "synth2" / "vndb_psp" / "shots" / name)
    p = ROOT / "data" / "manifests" / "fresh_jiten_backgrounds.json"
    if p.exists():
        for x in json.loads(p.read_text(encoding="utf-8")).get("records", []):
            add("scene", x["url"], x["vn"], x["url"], x.get("sha256"), x.get("release"))
    out = []
    for num, r in sorted(rows.items()):
        out.append({"id": f"sf{num}", "url": r["url"], "vn": r["vn"], "roles": sorted(r["roles"]),
                    "sha256": sorted(r["sha256"]), "releases": sorted(r["releases"]), "paths": sorted(r["paths"])})
    return out


def teacher_val_shots():
    """{shot key: [teacher_val line ids]} of teacher_val's lines."""
    import numpy as np
    out = defaultdict(list)
    p = P.ART / "teacher" / "teacher_val.npz"
    if not p.exists():
        return out
    for m in json.loads(str(np.load(p)["meta"])):
        out[m["id"].rsplit("_", 1)[0]].append(m["id"])
    return out


def cmd_refs(args):
    """Hash every locally available reference image (file and decoded pixels) and keep its
    thumbnail, for the duplicate checks of `select`."""
    import numpy as np
    rows = reference_rows()
    out = P.OUT / "refs"
    out.mkdir(parents=True, exist_ok=True)
    thumbs, tid = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        r["files"] = []
        for path in r["paths"]:
            rgb = P.read_rgb(path)
            if rgb is None:
                continue
            r["files"].append({"path": path, "sha256": P.file_sha256(path), "pixel_sha256": P.pixel_sha256(rgb),
                               "dims": [rgb.shape[1], rgb.shape[0]]})
            thumbs.append(P.thumb(rgb))
            tid.append(f"{r['id']}|{path}")
        if i % 1000 == 0:
            print(f"{i}/{len(rows)} references, {len(thumbs)} images, {time.time() - t0:.0f}s", flush=True)
    (out / "refs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    np.savez_compressed(out / "thumbs.npz", thumbs=np.stack(thumbs).astype(np.float32), ids=np.array(tid))
    roles = Counter(x for r in rows for x in r["roles"])
    print(json.dumps({"references": len(rows), "images": len(thumbs), "roles": roles}), flush=True)
    print("REFS_DONE", flush=True)


# ----------------------------------------------------------------------------- the rules

def native_kind(row):
    p, (w, h) = set(row["platforms"]), tuple(row["dims"])
    if "psv" in p and (w, h) == P.DIMS["vita"]:
        return "vita"
    if "psp" in p and (w, h) == P.DIMS["psp"]:
        return "psp"
    return None


def vn_exclusion(vn, ctx):
    """Why VN `vn` can't be in the cohort, or None."""
    if vn not in ctx["reserved"]:
        return "not-reserved"
    if is_test_vn(vn) or vn in ctx["test_vns"]:
        return "test-vn"
    if vn in ctx.get("fresh_vns", ()):
        return "fresh-test-vn"
    if vn in ctx["clannad"]:
        return "clannad-family"
    if vn in ctx["label_vns"]:
        return "label-pool-vn"
    return None


def shot_exclusion(row, ctx):
    """Why a listing row can't be used (before any pixels are looked at), or None."""
    why = vn_exclusion(row["vn"], ctx)
    if why:
        return why
    if native_kind(row) is None:
        return "not-native"
    if row["langs"] != ["ja"]:
        return "language"
    if row["sexual"] >= SEXUAL_MAX or row["violence"] >= VIOLENCE_MAX:
        return "rating"
    if row["release"] in ctx["test_sessions"]:
        return "test-session"
    if row["release"] in ctx["label_releases"]:
        return "label-release"
    roles = ctx["ref_roles"].get(P.image_num(row["url"]), set())
    if "eval-test" in roles:
        return "eval-shot"
    if "label-train" in roles:
        return "label-shot"
    return None


# Screenshots of the repo's other sets: line-finder ground truth and the research's hand sets
# (eval/linefind/manifests) and the general-VN development sources. Never in strict or
# scene_vn; in tv_native (teacher_val's own screenshots) they stay, flagged.
OTHER_SETS = {"eval-linefind": "linefind", "eval-general-vn": "general_vn_dev"}


def flags_of(roles, vn, ctx):
    flags = [f for role, f in OTHER_SETS.items() if role in roles]
    if "scene" in roles:
        flags.append("bg_shot")
    if vn in ctx["scene_vns"]:
        flags.append("scene_vn")
    return flags


def subset_of(row, ctx):
    """(subset, flags) of an eligible listing row, or (None, reason)."""
    roles = ctx["ref_roles"].get(P.image_num(row["url"]), set())
    flags = flags_of(roles, row["vn"], ctx)
    if "teacher-val" in roles or P.shot_key(row["vn"], row["url"]) in ctx["tv_shots"]:
        return "tv_native", flags
    if roles & set(OTHER_SETS):
        return None, "other-set-shot"
    if "scene" in roles:
        return None, "scene-shot"
    if row["vn"] in ctx["scene_vns"]:
        return "scene_vn", flags
    return "strict", flags


def fresh_sources():
    """(VN ids, file sha256s, RGB pixel sha256s) of the fresh native textbox test's sources."""
    if not FRESH_PROVENANCE.exists():
        return set(), set(), set()
    d = json.loads(FRESH_PROVENANCE.read_text(encoding="utf-8"))
    src = d.get("sources", [])
    vns = {s["vn"] for s in src} | {o["source_key"][0] for o in d.get("omitted", []) if o.get("source_key")}
    return vns, {s["sha256"] for s in src if s.get("sha256")}, {s["pixel_sha256"] for s in src if s.get("pixel_sha256")}


def pil_pixel_sha256(path):
    """The fresh-native harness's pixel hash: sha256 of PIL's RGB bytes (no shape prefix)."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        return hashlib.sha256(np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8)).tobytes()).hexdigest()


def context(refs, clannad_family, fresh_family=None):
    tg = json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text(encoding="utf-8"))
    label = label_pool_vns()
    ref_roles = defaultdict(set)
    label_releases, scene_vns = set(), set()
    for r in refs:
        n = P.image_num(r["id"])
        ref_roles[n] |= set(r["roles"])
        if "label-train" in r["roles"]:
            label_releases |= set(r["releases"])
        if "scene" in r["roles"]:
            scene_vns.add(r["vn"])
    fvns, fsha, fpix = fresh_sources()
    fresh = set(FRESH_NATIVE) | fvns
    for v in list(fresh):
        fresh |= {x["id"] for x in (fresh_family or {}).get(v, {}).get("relations", [])}
    return {"reserved": set(reserved_vns()), "test_vns": set(tg["all"]), "test_sessions": set(tg["sessions"]),
            "fresh_vns": fresh, "fresh_sha": fsha, "fresh_pix": fpix,
            "clannad": {CLANNAD} | {x["id"] for x in clannad_family.get("relations", [])},
            "label_vns": set(label), "label_sources": label, "label_releases": label_releases,
            "scene_vns": scene_vns, "ref_roles": ref_roles, "tv_shots": teacher_val_shots()}


# ----------------------------------------------------------------------------- fetch + dedupe

def mirrors(row, kind):
    name = f"{row['vn']}_{row['url'].rsplit('/', 1)[-1]}"
    A = P.ART
    return [A / "realdata" / "shots" / kind / name, P.SHARED_DATA / "vndb" / "shots" / name,
            A / "synth2" / "vndb_psp" / "shots" / name, A / "linefind" / "psp_dev" / "shots" / name,
            A / "eval" / kind / "shots" / name]


def fetch(row, kind, dest, session, delay):
    """Copy the screenshot from a local mirror or download it from VNDB. Returns how."""
    if dest.exists():
        return "present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for m in mirrors(row, kind):
        if m.exists() and P.read_rgb(m) is not None:
            shutil.copyfile(m, dest)
            return f"mirror:{m.parent.parent.name}/{m.parent.name}"
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(5):
        try:
            r = session.get(row["url"], timeout=60)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            time.sleep(delay)
            return "vndb"
        except Exception as e:  # noqa: BLE001
            print(f"  retry {row['url']}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def same_frame(a, b):
    """Whether two RGB screenshots are the same frame (re-encoded or rescaled): grey p99 of
    |a - b| at a's size below SAME_P99."""
    import cv2
    import numpy as np
    ga = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY).astype(np.int16)
    gb = cv2.cvtColor(np.ascontiguousarray(b), cv2.COLOR_RGB2GRAY)
    if gb.shape != ga.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.percentile(np.abs(ga - gb.astype(np.int16)), 99)) < SAME_P99


def cmd_select(args):
    import numpy as np
    meta = json.loads((P.OUT / "vndb" / "listing_meta.json").read_text(encoding="utf-8"))
    listing = _jsonl(P.OUT / "vndb" / "listing.jsonl")
    refs = _jsonl(P.OUT / "refs" / "refs.jsonl")
    if not refs:
        sys.exit("run `select.py refs` first")
    if "fresh_native_family" not in meta:
        sys.exit("listing predates the fresh-test relations: run `select.py list` again")
    ctx = context(refs, meta["clannad_family"], meta["fresh_native_family"])
    why, keep = Counter(), []
    for r in listing:
        reason = shot_exclusion(r, ctx)
        if reason is None:
            sub, flags = subset_of(r, ctx)
            if sub is None:
                reason = flags
            else:
                r = dict(r, kind=native_kind(r), subset=sub, flags=flags)
        if reason:
            why[reason] += 1
            continue
        keep.append(r)
    print(f"{len(listing)} listed, {len(keep)} pass the id rules: {dict(why)}", flush=True)
    # pixels: fetch, hash, then the duplicate checks against every reference image
    by_file_sha, by_pix = defaultdict(set), defaultdict(set)
    ref_by_id = {}
    for x in refs:
        ref_by_id[x["id"]] = x
        for s in x["sha256"]:
            by_file_sha[s].add(x["id"])
        for f in x.get("files", []):
            by_file_sha[f["sha256"]].add(x["id"])
            by_pix[f["pixel_sha256"]].add(x["id"])
    z = np.load(P.OUT / "refs" / "thumbs.npz")
    rthumbs, rids = z["thumbs"], [str(t) for t in z["ids"]]
    import requests
    sess = requests.Session()
    sess.headers["User-Agent"] = P.UA
    rows, fetched = [], Counter()
    for r in keep:
        dest = P.OUT / "shots" / r["kind"] / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}"
        how = "present" if dest.exists() else (None if args.no_fetch else fetch(r, r["kind"], dest, sess, args.delay))
        fetched[how or "missing"] += 1
        if how is None:
            why["unfetched"] += 1
            continue
        rgb = P.read_rgb(dest)
        if rgb is None or (rgb.shape[1], rgb.shape[0]) != P.DIMS[r["kind"]]:
            why["bad-image"] += 1
            continue
        r = dict(r, sha256=P.file_sha256(dest), pixel_sha256=P.pixel_sha256(rgb), path=str(dest))
        if r["sha256"] in ctx["fresh_sha"] or pil_pixel_sha256(dest) in ctx["fresh_pix"]:
            why["dup-fresh-test"] += 1
            continue
        own = P.image_id(r["url"])
        hits = (by_file_sha.get(r["sha256"], set()) | by_pix.get(r["pixel_sha256"], set())) - {own}
        r["_hits"] = sorted(hits)
        r["_thumb"] = P.thumb(rgb)
        r["_rgb"] = rgb
        rows.append(r)
    # identical bytes/pixels with another reference screenshot inherit its roles
    out, near_flags = [], Counter()
    T = np.stack([r["_thumb"] for r in rows]) if rows else np.zeros((0, 576), np.float32)
    corr = T @ rthumbs.T if len(rows) else np.zeros((0, 0))
    for i, r in enumerate(rows):
        roles = set()
        for h in r["_hits"]:
            roles |= set(ref_by_id[h]["roles"])
        dup = sorted(r["_hits"])
        near = []
        for j in np.flatnonzero(corr[i] >= NEAR):
            rid, path = rids[j].split("|", 1)
            if rid == P.image_id(r["url"]):
                continue
            rr = ref_by_id[rid]
            other = P.read_rgb(path)
            same = other is not None and same_frame(r["_rgb"], other)
            if same:
                dup.append(rid)
                roles |= set(rr["roles"])
            else:
                near.append({"id": rid, "vn": rr["vn"], "corr": round(float(corr[i, j]), 4), "roles": rr["roles"]})
        reason = None
        tv = r["subset"] == "tv_native"
        if "eval-test" in roles or any("eval-test" in n["roles"] for n in near):
            reason = "dup-eval"
        elif "label-train" in roles:
            reason = "dup-label"
        elif not tv and roles & set(OTHER_SETS):
            reason = "dup-other-set"
        elif not tv and "teacher-val" in roles:
            reason = "dup-teacher-val"
        elif not tv and "scene" in roles:
            reason = "dup-scene"
        if reason:
            why[reason] += 1
            continue
        flags = sorted(set(r["flags"]) | set(flags_of(roles, r["vn"], ctx)))
        if near:
            flags.append("near_dup")
            near_flags[r["subset"]] += 1
        # the same frame twice in the cohort (uploaded under two ids): keep the first
        twin = next((o for o in out if float(r["_thumb"] @ o["_thumb"]) >= NEAR
                     and (o["pixel_sha256"] == r["pixel_sha256"] or same_frame(r["_rgb"], o["_rgb"]))), None)
        if twin is not None:
            why["dup-intra"] += 1
            continue
        rec = dict(r)
        rec.update(flags=sorted(set(flags)), same_frame_refs=sorted(set(dup)), near=near,
                   shot=P.shot_key(r["vn"], r["url"]), session=r["release"],
                   tv_lines=ctx["tv_shots"].get(P.shot_key(r["vn"], r["url"]), []))
        out.append(rec)
    out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in out]
    out.sort(key=lambda r: (SUBSETS.index(r["subset"]), int(r["vn"][1:]), P.image_num(r["url"])))
    # the committed manifest: no pixels, no text
    fields = ("shot", "subset", "vn", "kind", "url", "id", "sha256", "pixel_sha256", "dims", "release", "session",
              "sexual", "violence", "flags", "same_frame_refs", "near")
    P.MANIFESTS.mkdir(parents=True, exist_ok=True)
    (P.MANIFESTS / "shots.jsonl").write_text(
        "".join(json.dumps({k: r[k] for k in fields}, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    (P.OUT / "shots.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                                       encoding="utf-8")
    report = {"listing": {k: v for k, v in meta.items() if k != "titles"}, "listed": len(listing),
              "excluded": dict(why.most_common()), "fetch": dict(fetched),
              "selected": len(out), "by_subset_kind": dict(Counter(f"{r['subset']}/{r['kind']}" for r in out)),
              "vns_by_subset": {s: len({r["vn"] for r in out if r["subset"] == s}) for s in SUBSETS},
              "flags": dict(Counter(f"{r['subset']}:{f}" for r in out for f in r["flags"])),
              "label_pool_vns_reserved": sorted(v for v in ctx["reserved"] if v in ctx["label_vns"]),
              "clannad_family": sorted(ctx["clannad"]), "fresh_test_family": sorted(ctx["fresh_vns"]),
              "fresh_test_family_reserved": sorted(ctx["fresh_vns"] & ctx["reserved"])}
    (P.OUT / "select_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                              encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("excluded", "fetch", "selected", "by_subset_kind", "vns_by_subset",
                                             "flags")}, ensure_ascii=False, indent=1), flush=True)
    print("SELECT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--reserved", default=str(RESERVED))
    sub.add_parser("refs")
    p = sub.add_parser("select")
    p.add_argument("--no-fetch", action="store_true", help="use only screenshots already under $OUT/shots")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between VNDB downloads")
    args = ap.parse_args()
    {"list": cmd_list, "refs": cmd_refs, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()

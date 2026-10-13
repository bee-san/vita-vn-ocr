#!/usr/bin/env python3
"""realplus: training-side VNDB screenshots for a large real training pool.

  shots.py dump    [--date YYYY-MM-DD]          VNDB's daily database dump -> $REALPLUS/vndb/
  shots.py select  [--per-vn-hires K]           -> $REALPLUS/manifests/shots.jsonl, selection.json
  shots.py fetch   [--workers N] [--delay S]    -> $REALPLUS/shots/<kind>/<vn>_<id>.jpg (sha256, pixel sha256)
  shots.py dedupe                               pixel duplicates of every existing pool and set -> dups.json

The listing comes from the near-complete database dump (https://vndb.org/d14, one file a day,
the API's usage terms) instead of thousands of API pages. A screenshot is a candidate when

  * its VN is a Japanese original and its release is Japanese-only or Japanese + English (no
    Chinese or Korean text on screen), with VNDB flag averages sexual < 0.3, violence < 0.5;
  * it is one of realdata's three kinds: vita (a psv release at 960x544), psp (a psp release
    at 480x272) or hires (any other release's shot, aspect 1.25-1.85, at least 640x400, later
    scaled to fit 960x544 and 480x272);
  * its VN is on the training side (reserved_vns): not a test game (synth/common.is_test_vn,
    eval/manifests/test_games.json "all"), not CLANNAD or its family, not a fresh native
    textbox title (FRESH_NATIVE) or any of its VNDB relations, not a development VN (synth3's
    exclude_dev_vns.txt, the general-VN / CLANNAD development VNs, devnative's cohort) or its
    family, not in the
    validation bucket of the repo's split (15 <= md5(vn) % 100 < 25), and sharing no VNDB
    release with any of those;
  * its release is not one of eval's test capture sessions, and the screenshot (VNDB image id)
    is in no existing pool or evaluation set (realdata/teacher, synth2/synth3 scenes, the
    data/vndb shots, linefind, eval's sets and candidates, fresh-synthetic backgrounds,
    general-VN development, devnative).

`dedupe`, after `fetch`, also drops every screenshot whose decoded pixels (pixel_sha256) equal
those of a screenshot in any of those pools or sets that is on disk, or an earlier realplus
screenshot. Screenshots are copyrighted: they stay under $REALPLUS, never in git.
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "synth"))
from common import is_test_vn  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
UA = "vita-vn-ocr-realplus/0.1 (training-line screenshots, low volume; github.com/bee-san/vita-vn-ocr)"
DUMP_URL = "https://dl.vndb.org/dump/vndb-db-{date}.tar.zst"
SEXUAL_MAX, VIOLENCE_MAX = 0.3, 0.5
KINDS = ("vita", "psp", "hires")
HIRES_PLATFORMS = {"win", "mac", "lin", "dos", "p98", "p88", "x68", "fmt", "swi", "ps2", "ps3",
                   "ps4", "ps5", "psx", "ps1", "xb3", "xbo", "xxs", "sat", "drc", "dvd", "wii", "wiu"}
FAMILY = {"alt", "orig", "fan", "par", "side", "seq", "preq", "ser"}   # as eval/tools/manifest.py
CLANNAD = {"v4"}
# CLANNAD's direct VNDB family (alt/fan/orig/par/preq/seq/ser/side), as the conductor listed it on
# 2026-10-12 (train2's old pools held 45 such crops). reserved_vns finds them from the dump too.
CLANNAD_FAMILY = set("v12 v4060 v28752 v27548 v10058 v29951 v23354 v16901 v5698 v7983 v54298".split())
CLANNAD_TITLE = re.compile(r"clannad|クラナド", re.I)
# Every VN of the general-VN and CLANNAD development manifests (synth3_text.GENERAL_VN_DEV).
GENERAL_VN_DEV = set(("v4 v1015 v1281 v2632 v5154 v5297 v10091 v11109 v11454 v12775 v13766 v15773 v16128 "
                      "v18161 v18163 v18643").split())
IMAGE_ID = re.compile(r"sf/\d\d/(\d+)\.jpg")
SHOT_FILE = re.compile(r"^v\d+_(\d+)\.(?:jpg|jpeg|png)$")


def realplus_dir():
    return Path(os.environ.get("REALPLUS") or ART / "realplus")


def dump_dir():
    return realplus_dir() / "vndb" / "dump" / "db"


# ---------------------------------------------------------------- reserved VNs and screenshots

def val_bucket(vn):
    """realdata/teacher's validation bucket (build.split_of): 15 <= md5(vn) % 100 < 25."""
    return 15 <= int(hashlib.md5(vn.encode()).hexdigest(), 16) % 100 < 25


def test_games():
    return json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text(encoding="utf-8"))


def json_vns(path):
    """Every VNDB VN id ("v123") quoted anywhere in a JSON file."""
    return set(re.findall(r'"(v\d+)"', Path(path).read_text(encoding="utf-8"))) if Path(path).exists() else set()


# The fresh native textbox test's titles (eval/fresh_native_source_provenance_20261012.json, PR #49):
# PSP v14909 v25459 v29454, Vita v21289 v21976 v20519, reserves v16509 v19140. Protected exactly like
# vita-test / psp-test (the conductor's hygiene update of 2026-10-12), with every VNDB relation.
FRESH_NATIVE = set("v14909 v25459 v29454 v21289 v21976 v20519 v16509 v19140".split())
FRESH_NATIVE_FILES = ("fresh_native_source_provenance_20261012.json", "fresh_native_textbox_20261012.json",
                      "fresh_native_textbox_independent_audit_20261012.json")


# The owner's own games (owner 2026-10-13 10:51Z and 13:10Z / 13:12Z, superseding the 09:25Z / 09:37Z notes).
# Test-only titles: the owner's live screenshots of them are fresh native test sets, never trained, tuned,
# calibrated or rendered from, and never sent to Codex; each is reserved with its series. Memories Off (PSP,
# its whole series), Aiyoku no Eustia (Vita) and 428 (PSP) were reserved before any harvest touched the last two.
# Training-eligible titles: the owner's screenshots of them may be labelled for training (never through Codex:
# they are private frames, not public VNDB images); each is re-checked against reserved_vns at ingest.
OWNER_TEST_SERIES = {"v1152": "Memories Off", "v3770": "Aiyoku no Eustia", "v1299": "428"}
OWNER_TEST_TITLE = re.compile(r"memories\s*off|メモリーズ\s*オフ|秋之回[忆憶]", re.I)   # Memories Off's titles
OWNER_TRAINING = {"v22505": "CHAOS;HEAD NOAH", "v14018": "CHAOS;CHILD", "v751": "Rewrite", "v27448": "ATRI",
                  "v57": "Sharin no Kuni", "v4137": "Toradora! Portable", "v183": "Kikokugai",
                  "v12369": "Fate/EXTRA CCC", "v434": "Togainu no Chi", "v195": "Otome wa Boku ni Koishiteru",
                  "v1552": "Mashiro-iro Symphony", "v4028": "Koi to Senkyo to Chocolate"}


def owner_test_vns(d, roots=None, title=OWNER_TEST_TITLE):
    """{vn: reason} for the owner's test series (Memories Off, Aiyoku no Eustia, 428), from a load_dump() dict:
    every VN reachable from a root through FAMILY relations (alternative versions, originals, fan discs,
    parent and side stories, sequels, prequels, same series), or titled like Memories Off
    (`owner-test-series`); every VNDB
    relation of any kind of those, in either direction (`owner-test-relation`: crossovers and same-setting
    titles such as KID Mix Section or Never 7); and every VN on one VNDB release with a series VN
    (`owner-test-shared-release`). A relation's own relations and release partners are not reserved."""
    roots = set(OWNER_TEST_SERIES if roots is None else roots)
    fam = d["relations"]
    anyrel = d.get("any_relations", fam)
    series = {v for v, ts in d["title"].items() if any(title.search(t) for t in ts)} | roots
    todo = list(series)
    while todo:
        v = todo.pop()
        nxt = set(fam.get(v, ())) | {u for u, rs in fam.items() if v in rs}
        for u in nxt - series:
            series.add(u)
            todo.append(u)
    out = {v: "owner-test-series" for v in series}
    for v in sorted(series):
        for u in anyrel.get(v, ()):
            out.setdefault(u, "owner-test-relation")
    for v, rs in anyrel.items():
        if rs & series:
            out.setdefault(v, "owner-test-relation")
    for rid, rv in sorted(d["rel_vns"].items()):     # compilations and bundles holding a series VN
        if rv & series:
            for u in sorted(rv):
                out.setdefault(u, "owner-test-shared-release")
    return out


def fresh_native_vns():
    out = set(FRESH_NATIVE)
    for name in FRESH_NATIVE_FILES:
        out |= json_vns(ROOT / "eval" / name)
    return out


def fresh_native_hashes():
    """Every file sha256 and RGB pixel sha256 the fresh native test records for its images."""
    out = set()
    for name in FRESH_NATIVE_FILES:
        p = ROOT / "eval" / name
        if p.exists():
            out |= set(re.findall(r'"(?:sha256|pixel_sha256)": "([0-9a-f]{64})"', p.read_text(encoding="utf-8")))
    return out


def dev_vns():
    """Development VNs: synth3's reservation (teacher_val's VNs and the general-VN / CLANNAD
    development VNs), the general-VN development manifests, and devnative's cohort."""
    out = set(GENERAL_VN_DEV)
    p = ART / "synth3" / "mining" / "exclude_dev_vns.txt"
    if p.exists():
        out |= set(p.read_text().split())
    for name in ("general_vn_development_review_20261011.json", "general_vn_dev_readiness_20261011.json"):
        out |= json_vns(ROOT / "eval" / name)
    gsrc = ROOT / "data" / "manifests" / "general_vn_development_sources_20261011.json"
    if gsrc.exists():
        out |= {r["vn"] for r in json.loads(gsrc.read_text(encoding="utf-8"))["records"] if r.get("vn")}
    out |= devnative_vns()
    return out


def devnative_files():
    """devnative's cohort files: its selected screenshots, its VNDB listing of the reserved VNs, its
    labels and its committed manifests (not refs/, which lists the pools it must avoid)."""
    dn = ART / "devnative"
    files = [dn / "shots.jsonl", dn / "vndb" / "listing.jsonl", *sorted(dn.glob("labels*.jsonl")),
             *sorted((ROOT / "devnative" / "manifests").glob("*.jsonl"))]
    return [p for p in files if p.exists()]


def devnative_vns():
    out = set()
    for p in devnative_files():
        out |= set(re.findall(r'"vn": ?"(v\d+)"', p.read_text(encoding="utf-8", errors="replace")))
    return out


def screenshot_sources():
    """(name, path) of every file that lists screenshots of an existing pool or set."""
    rd, ev = ROOT / "realdata" / "manifests", ROOT / "eval"
    src = [("realdata", rd / f"shots_{k}.jsonl") for k in KINDS]
    src += [("realdata", ART / "realdata" / "manifests" / f"shots_{k}.jsonl") for k in KINDS]
    src += [("teacher", ROOT / "teacher" / "manifests" / "shots.jsonl")]
    src += [("eval-test", ev / "manifests" / f"{s}.jsonl") for s in ("vita-test", "psp-test")]
    src += [("eval-candidates", ART / "eval" / k / "shots.jsonl") for k in ("vita", "psp")]
    src += [("linefind", p) for p in sorted((ev / "linefind" / "manifests").glob("*.jsonl"))]
    src += [("linefind", ART / "linefind" / "psp_dev" / "shots.jsonl")]
    src += [("synth2", ROOT / "data" / "manifests" / "synth2_psp_shots.jsonl"),
            ("synth2", ART / "synth2" / "backgrounds.json")]
    src += [("fresh-synthetic", p) for p in sorted((ROOT / "data" / "manifests").glob("fl10_*.json"))]
    src += [("fresh-synthetic", ROOT / "data" / "manifests" / "fresh_jiten_backgrounds.json")]
    src += [("general-vn-dev", ROOT / "data" / "manifests" / "general_vn_development_sources_20261011.json")]
    src += [("devnative", p) for p in devnative_files()]
    return src


def screenshot_dirs():
    """(name, dir) of every local copy of an existing pool's or set's screenshots."""
    return [("realdata", ART / "realdata" / "shots"), ("data-vndb", HERD / "shared" / "data" / "vndb" / "shots"),
            ("synth2", ART / "synth2" / "vndb_psp"), ("eval-candidates", ART / "eval" / "vita" / "shots"),
            ("eval-candidates", ART / "eval" / "psp" / "shots"), ("linefind", ART / "linefind" / "psp_dev" / "shots"),
            ("devnative", ART / "devnative")]


def image_ids_in(path):
    """VNDB screenshot ids named in a file: t.vndb.org URLs and <vn>_<id>.jpg file names."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    ids = {int(x) for x in IMAGE_ID.findall(text)}
    ids |= {int(x) for x in re.findall(r'"v\d+_(\d+)\.(?:jpg|jpeg|png)"', text)}
    return ids


def reserved_screenshots():
    """{VNDB image id: [pool or set names]} over every existing pool and set, by file and by id."""
    out = collections.defaultdict(set)
    for name, path in screenshot_sources():
        if Path(path).exists():
            for i in image_ids_in(path):
                out[i].add(name)
    for name, d in screenshot_dirs():
        if d.exists():
            for p in d.rglob("*"):
                m = SHOT_FILE.match(p.name)
                if m:
                    out[int(m.group(1))].add(name)
    return out


# ---------------------------------------------------------------- the dump

def rows(table, base=None):
    with open((base or dump_dir()) / table, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def load_dump(base=None):
    """The dump tables realplus needs, as plain dicts and sets."""
    d = {"olang": {}, "title": collections.defaultdict(list), "img": {}, "plat": collections.defaultdict(set),
         "langs": collections.defaultdict(set), "rel_vns": collections.defaultdict(set),
         "relations": collections.defaultdict(set), "shots": []}
    for r in rows("vn", base):
        d["olang"][r[0]] = r[3]
    for r in rows("vn_titles", base):
        d["title"][r[0]] += [x for x in r[3:5] if x and x != "\\N"]
    for r in rows("images", base):
        if r[0].startswith("sf"):
            d["img"][int(r[0][2:])] = (int(r[1]), int(r[2]), int(r[4]) / 100, int(r[6]) / 100)
    for r in rows("releases_platforms", base):
        d["plat"][r[0]].add(r[1])
    for r in rows("releases_titles", base):
        d["langs"][r[0]].add(r[1])
    for r in rows("releases_vn", base):
        d["rel_vns"][r[0]].add(r[1])
    d["any_relations"] = collections.defaultdict(set)
    for r in rows("vn_relations", base):
        d["any_relations"][r[0]].add(r[1])
        if r[2] in FAMILY:
            d["relations"][r[0]].add(r[1])
    for vn, scr, rid in rows("vn_screenshots", base):
        d["shots"].append((vn, int(scr[2:]), None if rid == "\\N" else rid))
    return d


def kind_of(platforms, w, h):
    """realdata's kinds (realdata/vndb.kind_of), with a wider hires kind: any platform's shot
    of at least 640x400 (PC-98 and early Windows games are 640x400), aspect 1.25-1.85, that is not a
    native Vita or PSP frame. Smaller retro and handheld frames (320x240, 256x192, ...) stay out:
    scaled up they look like nothing the consoles render."""
    p = set(platforms)
    if "psv" in p and (w, h) == (960, 544):
        return "vita"
    if "psp" in p and (w, h) == (480, 272):
        return "psp"
    if w >= 640 and h >= 400 and 1.25 <= w / h <= 1.85:
        return "hires"
    return None


LANGS_OK = ({"ja"}, {"ja", "en"})


def langs_ok(langs):
    """Japanese-only releases (realdata's rule) and Japanese + English ones; English lines are
    dropped later as not Japanese. Releases with Chinese or Korean are left out, since hanzi-only
    lines would pass as Japanese."""
    return set(langs or ()) in LANGS_OK


def reserved_vns(d, test_all=None, fresh=None, dev=None):
    """{vn: reason} for every VN off limits to realplus. Reasons, first match wins:

      clannad       CLANNAD by id or title
      clannad-family  CLANNAD_FAMILY, the conductor's list of CLANNAD's direct family
      owner-test-series  owner_test_vns(): the owner's test-only series (Memories Off, Aiyoku no Eustia,
                    428), every VN of them
      test-set      eval's test_games.json "all" (set games, their family, release partners)
      test-vn       synth/common.is_test_vn (the 15% of VNs held out for evaluation)
      fresh-native  a title of the fresh native textbox test
      development   synth3's exclude_dev_vns.txt, general-VN / CLANNAD development, devnative
      val-bucket    the repo's validation split, 15 <= md5(vn) % 100 < 25
      fresh-native-relation  any direct VNDB relation (every kind) of a fresh native title
      clannad-relation  a direct VNDB relation of CLANNAD (v4) of every kind, or a FAMILY relation of
                    a CLANNAD_FAMILY VN
      owner-test-relation, owner-test-shared-release  owner_test_vns(): a VNDB relation of any kind of
                    a series VN, or a VN on one VNDB release with a series VN or such a relation
      family        a direct VNDB relation of kind FAMILY of a CLANNAD, fresh-native or
                    development VN (test-set already holds the set games' family)
      shared-release  on one VNDB release with a test-set, CLANNAD, fresh-native, development or
                    family VN; or with a test-vn or val-bucket VN when the release links just the
                    two of them (one game filed twice, or a game and its fandisc) or the two are
                    FAMILY relations. Larger compilations of unrelated games don't count there.
    """
    test_all = set(test_games()["all"] if test_all is None else test_all)
    fresh = fresh_native_vns() if fresh is None else set(fresh)
    dev = dev_vns() if dev is None else set(dev)
    vns = set(d["olang"])
    clannad = set(CLANNAD) | {v for v in vns if any(CLANNAD_TITLE.search(t) for t in d["title"].get(v, []))}
    owner = owner_test_vns(d)
    owner_series = {v for v, r in owner.items() if r == "owner-test-series"}
    why = {}
    for v in vns | test_all | fresh | dev | CLANNAD_FAMILY:
        if v in clannad:
            why[v] = "clannad"
        elif v in CLANNAD_FAMILY:
            why[v] = "clannad-family"
        elif v in owner_series:
            why[v] = "owner-test-series"
        elif v in test_all:
            why[v] = "test-set"
        elif is_test_vn(v):
            why[v] = "test-vn"
        elif v in fresh:
            why[v] = "fresh-native"
        elif v in dev:
            why[v] = "development"
        elif val_bucket(v):
            why[v] = "val-bucket"
    roots = {v for v, r in why.items() if r in ("clannad", "fresh-native", "development")}
    fresh_roots = {v for v, r in why.items() if r == "fresh-native"}
    anyrel = d.get("any_relations", d["relations"])
    for v in sorted(fresh_roots):              # the fresh native titles: relations of every kind
        for u in anyrel.get(v, ()):
            why.setdefault(u, "fresh-native-relation")
    for v, rs in anyrel.items():
        if rs & fresh_roots:
            why.setdefault(v, "fresh-native-relation")
    # CLANNAD (audit of PR #57): CLANNAD's own relations of every kind (char-sharing parodies such
    # as NETANNAD reuse its characters, names and text), and the FAMILY relations of the
    # conductor's family list (a sequel of a family member, such as Kyou After 2).
    for v in sorted(CLANNAD):
        for u in anyrel.get(v, ()):
            why.setdefault(u, "clannad-relation")
    for v, rs in anyrel.items():
        if rs & CLANNAD:
            why.setdefault(v, "clannad-relation")
    for v in sorted(CLANNAD_FAMILY):
        for u in d["relations"].get(v, ()):
            why.setdefault(u, "clannad-relation")
    for v, rs in d["relations"].items():
        if rs & CLANNAD_FAMILY:
            why.setdefault(v, "clannad-relation")
    # the owner's test series (2026-10-13 10:51Z): relations of every kind and every release partner
    for v, r in sorted(owner.items()):
        why.setdefault(v, r)
    for v in sorted(roots):
        for u in d["relations"].get(v, ()):
            why.setdefault(u, "family")
    for v, rs in d["relations"].items():       # relations are listed both ways, but be safe
        if rs & roots:
            why.setdefault(v, "family")
    strict = {v for v, r in why.items() if r not in ("test-vn", "val-bucket")}
    loose = set(why) - strict
    for rid, rv in sorted(d["rel_vns"].items()):
        if len(rv) < 2:
            continue
        for u in sorted(rv):
            if u in why:
                continue
            others = rv - {u}
            if others & strict or (others & loose and (len(rv) == 2 or d["relations"].get(u, set()) & others)):
                why[u] = "shared-release"
    return why


SESSION_GAP = 50


def assign_sessions(rows_):
    """realdata's capture sessions (vndb.assign_sessions): "<vn>/<release>/<n>", screenshots of
    one release whose VNDB ids are consecutive (gaps under SESSION_GAP)."""
    by = collections.defaultdict(list)
    for r in rows_:
        by[(r["vn"], r["release"] or "r0")].append(r)
    for (vn, rel), rs in by.items():
        rs.sort(key=lambda r: r["image"])
        n, last = 0, None
        for r in rs:
            if last is not None and r["image"] - last >= SESSION_GAP:
                n += 1
            r["session"] = f"{vn}/{rel}/{n}"
            last = r["image"]
    return rows_


def url_of(image):
    return f"https://t.vndb.org/sf/{image % 100:02d}/{image}.jpg"


def stable_rank(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def select(d, reserved_vn, reserved_img, sessions, per_vn_hires=0, seed=20261012):
    """The candidate rows and the count of every reason a screenshot was dropped."""
    why = collections.Counter()
    keep = []
    for vn, image, rid in d["shots"]:
        if d["olang"].get(vn) != "ja":
            why["not-japanese-original"] += 1
            continue
        if vn in reserved_vn:
            why["vn:" + reserved_vn[vn]] += 1
            continue
        if rid in sessions:
            why["test-session"] += 1
            continue
        if image in reserved_img:
            why["in-pool:" + "+".join(sorted(reserved_img[image]))] += 1
            continue
        if image not in d["img"]:
            why["no-image-row"] += 1
            continue
        w, h, sexual, violence = d["img"][image]
        if sexual >= SEXUAL_MAX or violence >= VIOLENCE_MAX:
            why["rating"] += 1
            continue
        if not langs_ok(d["langs"].get(rid)):
            why["language"] += 1
            continue
        plat = sorted(d["plat"].get(rid, ()))
        k = kind_of(plat, w, h)
        if k is None:
            why["kind"] += 1
            continue
        keep.append({"image": image, "vn": vn, "kind": k, "url": url_of(image), "dims": [w, h], "release": rid,
                     "platforms": plat, "sexual": sexual, "violence": violence})
    # hires: at most K per VN (0 = all), the K with the lowest seeded hash of their image id,
    # so the choice of a VN's shots doesn't depend on any other VN.
    by_vn = collections.defaultdict(list)
    for r in keep:
        if r["kind"] == "hires":
            by_vn[r["vn"]].append(r)
    drop = set()
    for vn, rs in by_vn.items():
        rs.sort(key=lambda r: stable_rank(seed, r["image"]))
        if per_vn_hires:
            drop |= {r["image"] for r in rs[per_vn_hires:]}
    why["hires-cap"] = len(drop)
    keep = [r for r in keep if r["image"] not in drop]
    # download order: native kinds first, then round-robin over VNs (a VN's n-th shot after
    # every VN's (n-1)-th), so a partial download is spread over as many VNs as possible.
    rank = collections.defaultdict(int)
    for r in sorted(keep, key=lambda r: (r["vn"], stable_rank(seed, r["image"]))):
        r["rank"] = rank[r["vn"]]
        rank[r["vn"]] += 1
    order = {"vita": 0, "psp": 1, "hires": 2}
    keep.sort(key=lambda r: (order[r["kind"]], r["rank"], stable_rank(seed, r["vn"]), r["image"]))
    assign_sessions(keep)
    for i, r in enumerate(keep):
        r["order"] = i
    return keep, why


def manifest_path():
    return realplus_dir() / "manifests" / "shots.jsonl"


def load_shots(path=None):
    p = Path(path or manifest_path())
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def save_shots(rows_, path=None):
    p = Path(path or manifest_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_), encoding="utf-8")
    tmp.replace(p)


def cmd_select(args):
    d = load_dump()
    tg = test_games()
    rvn = reserved_vns(d)
    rimg = reserved_screenshots()
    keep, why = select(d, rvn, rimg, set(tg["sessions"]), args.per_vn_hires)
    old = {r["image"]: r for r in load_shots()}
    for r in keep:                       # keep the hashes of screenshots already fetched
        for k in ("sha256", "pixel_sha256", "rgb_sha256", "bytes", "status"):
            if k in old.get(r["image"], {}):
                r[k] = old[r["image"]][k]
    save_shots(keep)
    stamp = (dump_dir().parent / "TIMESTAMP").read_text().strip() if (dump_dir().parent / "TIMESTAMP").exists() \
        else None
    summary = {
        "dump": stamp, "per_vn_hires": args.per_vn_hires,
        "kinds": {k: {"screenshots": sum(r["kind"] == k for r in keep),
                      "vns": len({r["vn"] for r in keep if r["kind"] == k})} for k in KINDS},
        "screenshots": len(keep), "vns": len({r["vn"] for r in keep}),
        "dropped": dict(sorted(why.items(), key=lambda kv: -kv[1])),
        "reserved_vns": dict(collections.Counter(rvn.values())),
        "reserved_screenshots": len(rimg), "test_sessions": len(tg["sessions"]),
    }
    (realplus_dir() / "manifests" / "selection.json").write_text(json.dumps(summary, indent=1) + "\n")
    (realplus_dir() / "manifests" / "reserved_vns.json").write_text(json.dumps(rvn, indent=0, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- download and hashes

def shot_path(r, base=None):
    return (base or realplus_dir()) / "shots" / r["kind"] / f"{r['vn']}_{r['image']}.jpg"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha256(path):
    """devnative's decoded-pixel hash (devnative/paths.pixel_sha256: OpenCV's decode, then 'HxWx3'
    and the RGB bytes), so realplus, devnative and the pools compare like with like."""
    import cv2
    import numpy as np
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"undecodable {path}")
    a = np.ascontiguousarray(bgr[:, :, ::-1])
    h = hashlib.sha256(f"{a.shape[0]}x{a.shape[1]}x3".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def rgb_sha256(path):
    """eval/fresh_native_textbox.py's pixel hash: sha256 of PIL's RGB bytes, no header."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        a = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()


def cmd_fetch(args):
    import requests
    rows_ = load_shots()
    todo = [r for r in rows_ if (not r.get("sha256") or not r.get("rgb_sha256")) and r.get("status") != "failed"
            and (not args.kind or r["kind"] == args.kind)]
    if args.limit:
        todo = todo[:args.limit]
    lock, done = threading.Lock(), collections.Counter()
    local = threading.local()

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
            r["sha256"], r["bytes"] = file_sha256(dest), dest.stat().st_size
            r["pixel_sha256"], r["rgb_sha256"] = pixel_sha256(dest), rgb_sha256(dest)
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
            if i % 500 == 0:
                with lock:
                    save_shots(rows_)
                print(f"{i}/{len(todo)} {dict(done)} {time.time() - t0:.0f}s", flush=True)
    save_shots(rows_)
    print(f"fetch: {len(todo)} tried, {dict(done)}; {sum(1 for r in rows_ if r.get('sha256'))} of {len(rows_)} "
          f"have hashes", flush=True)
    print("FETCH_DONE" if not done["failed"] else "FETCH_INCOMPLETE", flush=True)


def pool_pixel_hashes(cache):
    """{pixel sha256: [pool names]} over the local copies of every existing pool and set
    (screenshot_dirs), cached by path, size and mtime."""
    cache = Path(cache)
    old = json.loads(cache.read_text()) if cache.exists() else {}
    new, out = {}, collections.defaultdict(set)
    for name, d in screenshot_dirs():
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not p.is_file():
                continue
            st = p.stat()
            key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
            h = old.get(key) or pixel_sha256(p)
            new[key] = h
            out[h].add(name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(new))
    return out


def dedupe(rows_, pools, fresh, read=frozenset()):
    """{image id (str): why} for every duplicate: a pixel match with an existing pool or set, a file
    or RGB hash the fresh native test records, or a pixel match with another realplus screenshot
    (the one already read is kept, else the first in download order)."""
    out, first = {}, {}
    order = sorted((r for r in rows_ if r.get("pixel_sha256")),
                   key=lambda r: (f"{r['vn']}_{r['image']}" not in read, r["order"]))
    for r in order:
        h = r["pixel_sha256"]
        if fresh & {r.get("sha256"), r.get("rgb_sha256")}:
            out[str(r["image"])] = "pool:fresh-native"
        elif h in pools:
            out[str(r["image"])] = "pool:" + "+".join(sorted(pools[h]))
        elif h in first:
            out[str(r["image"])] = f"realplus:{first[h]}"
        else:
            first[h] = r["image"]
    return out


def cmd_dedupe(args):
    """Writes $REALPLUS/manifests/dups.json (the fetcher owns shots.jsonl while it runs)."""
    rows_ = load_shots()
    pools = pool_pixel_hashes(realplus_dir() / "manifests" / "pool_pixel_hashes_v2.json")
    reads = realplus_dir() / "reads" / "reads.jsonl"
    read = set()
    if reads.exists():
        for x in reads.read_text(encoding="utf-8").splitlines():
            try:
                read.add(json.loads(x)["shot"])
            except (ValueError, KeyError):
                continue
    out = dedupe(rows_, pools, fresh_native_hashes(), read)
    p = realplus_dir() / "manifests" / "dups.json"
    p.write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    why = collections.Counter(v.split(":")[0] if v.startswith("realplus") else v for v in out.values())
    print(f"dedupe: {len(rows_)} rows, {sum(1 for r in rows_ if r.get('pixel_sha256'))} hashed; duplicates "
          f"{dict(why)}; {len(pools)} pool pixel hashes -> {p}", flush=True)


def cmd_dump(args):
    import subprocess
    out = realplus_dir() / "vndb"
    out.mkdir(parents=True, exist_ok=True)
    name = f"vndb-db-{args.date}.tar.zst"
    if not (out / name).exists():
        subprocess.run(["curl", "-sSfL", "-A", UA, "-o", str(out / (name + ".part")), DUMP_URL.format(date=args.date)],
                       check=True)
        (out / (name + ".part")).replace(out / name)
    (out / "dump").mkdir(exist_ok=True)
    subprocess.run(["tar", "-I", "zstd", "-xf", str(out / name), "-C", str(out / "dump")], check=True)
    print(f"{name} sha256 {file_sha256(out / name)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--date", required=True)
    p = sub.add_parser("select")
    p.add_argument("--per-vn-hires", type=int, default=0, help="at most this many hires shots per VN (0: all)")
    p = sub.add_parser("fetch")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--delay", type=float, default=0.5, help="seconds each worker waits after a download")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--limit", type=int, default=0)
    sub.add_parser("dedupe")
    args = ap.parse_args()
    {"dump": cmd_dump, "select": cmd_select, "fetch": cmd_fetch, "dedupe": cmd_dedupe}[args.cmd](args)


if __name__ == "__main__":
    main()

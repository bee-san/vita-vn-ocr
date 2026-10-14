#!/usr/bin/env python3
"""realplus: training-side VNDB screenshots for a large real training pool.

  shots.py dump    [--date YYYY-MM-DD]          VNDB's daily database dump -> $REALPLUS/vndb/
  shots.py select  [--per-vn-hires K]           -> $REALPLUS/manifests/shots.jsonl, selection.json
  shots.py fetch   [--workers N] [--delay S]    -> $REALPLUS/shots/<kind>/<vn>_<id>.jpg (sha256, pixel sha256)
  shots.py dedupe                               pixel duplicates of every existing pool and set -> dups.json
  shots.py reserve [--dump DB] [--test-games]   the reservation -> eval/manifests/reserved.json (docs/reservations.md)

The listing comes from the near-complete database dump (https://vndb.org/d14, one file a day,
the API's usage terms) instead of thousands of API pages. A screenshot is a candidate when

  * its VN is a Japanese original and its release is Japanese-only or Japanese + English (no
    Chinese or Korean text on screen), with VNDB flag averages sexual < 0.3, violence < 0.5;
  * it is one of realdata's three kinds: vita (a psv release at 960x544), psp (a psp release
    at 480x272) or hires (any other release's shot, aspect 1.25-1.85, at least 640x400, later
    scaled to fit 960x544 and 480x272);
  * its VN is on the training side (reserved_vns, docs/reservations.md; owner 2026-10-14 13:25Z): not
    in the test reservation (a vita-test or psp-test game, the CLANNAD family, or a VN on one VNDB
    release with one of them) and not in the development sample (eval/manifests/reserved.json,
    "dev_sample"). The old hash splits (synth/common.is_test_vn, val_bucket) reserve nothing any more;
  * its release is not one of eval's test capture sessions, and the screenshot (VNDB image id)
    is in no existing pool (realdata/teacher, synth2/synth3 scenes, the data/vndb shots,
    fresh-synthetic backgrounds, general-VN development, devnative) and is not a vita-test or
    psp-test screenshot. eval's unchosen candidates and linefind's finder sets are training-side.

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
    """FROZEN history: realdata/teacher's validation bucket (build.split_of), 15 <= md5(vn) % 100 < 25. teacher_val
    and real_val were built from it; it reserves nothing since 2026-10-14 (docs/reservations.md)."""
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
# Owner decisions that make a VN training data in all its releases although a relation or a shared release would
# reserve it (owner 2026-10-13 19:21Z and 20:47Z: planetarian, v34, which shares the PC Key 10th Memorial BOX
# r3748 with CLANNAD). reserved_vns never gives these VNs a derived reason (relations, release partners); a
# direct one (CLANNAD, a test or development set, is_test_vn, the val bucket) would still apply, and planetarian
# has none (it is in no eval or development set).
OWNER_ALLOWED = {"v34": "planetarian (owner 2026-10-13 19:21Z, 20:47Z: training data in all releases)"}


def owner_test_vns(d, roots=None, title=OWNER_TEST_TITLE):
    """{vn: reason} for the owner's test series (Memories Off, Aiyoku no Eustia, 428), from a load_dump() dict:
    every VN reachable from a root through FAMILY relations (alternative versions, originals, fan discs,
    parent and side stories, sequels, prequels, same series), or titled like Memories Off
    (`owner-test-series`); every VNDB relation of any kind of those, in either direction
    (`owner-test-relation`: crossovers and same-setting titles such as KID Mix Section or Never 7); and every
    VN on one VNDB release with a series VN
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
    """The old development reservation (information only since 2026-10-14; docs/reservations.md): synth3's list
    (teacher_val's VNs and the general-VN / CLANNAD development VNs), the general-VN development manifests, and
    devnative's cohort. These VNs are training data now, apart from the development sample in reserved_vns()."""
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


# Sets that no longer protect their screenshots (docs/reservations.md): eval's unchosen candidates and linefind's
# finder sets are training-side. Their vita-test / psp-test screenshots stay protected as "eval-test".
UNRESERVED_SETS = {"eval-candidates", "linefind"}


def eval_test_images(repo=ROOT):
    """VNDB image ids of vita-test's and psp-test's screenshots."""
    out = set()
    for s in ("vita-test", "psp-test"):
        p = repo / "eval" / "manifests" / f"{s}.jsonl"
        if p.exists():
            out |= image_ids_in(p)
    return out


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
    """{VNDB image id: [pool or set names]} over every existing pool and set, by file and by id. Sets in
    UNRESERVED_SETS count only for their vita-test / psp-test screenshots (named "eval-test")."""
    out = collections.defaultdict(set)
    test_img = eval_test_images()
    for name, path in screenshot_sources():
        if Path(path).exists():
            for i in image_ids_in(path):
                if name not in UNRESERVED_SETS:
                    out[i].add(name)
                elif i in test_img:
                    out[i].add("eval-test")
    for name, d in screenshot_dirs():
        if d.exists():
            for p in d.rglob("*"):
                m = SHOT_FILE.match(p.name)
                if m and name not in UNRESERVED_SETS:
                    out[int(m.group(1))].add(name)
                elif m and int(m.group(1)) in test_img:
                    out[int(m.group(1))].add("eval-test")
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


def release_partners(d, vns, close=False):
    """{partner: [release ids]}: every VN listed on one VNDB release with a VN of vns (compilations, bundles, one
    game filed under two VN entries), the VNs of vns themselves left out. close=True: only a release that links
    just the two VNs (one game filed twice, a game and its fan disc on one disc); compilations don't count."""
    vns = set(vns)
    out = collections.defaultdict(set)
    for rid, rv in d["rel_vns"].items():
        if len(rv) > 1 and rv & vns:
            for u in rv - vns:
                if not close or len(rv) == 2:
                    out[u].add(rid)
    return {u: sorted(rs, key=lambda r: (int(r[1:]) if r[1:].isdigit() else 1 << 62, r)) for u, rs in out.items()}


def clannad_vns(d):
    """{vn: reason} of the CLANNAD family as the repo protects it: CLANNAD (v4) and every VN titled CLANNAD
    (`clannad`), the conductor's family list (`clannad-family`), CLANNAD's direct VNDB relations of every kind and
    the FAMILY relations of the family list (`clannad-relation`: char-sharing parodies such as NETANNAD, sequels of
    family members such as Kyou After 2). OWNER_ALLOWED VNs never get the derived relation reason."""
    vns = set(d["olang"]) | set(d["title"])
    why = {v: "clannad" for v in set(CLANNAD) | {v for v in vns if any(CLANNAD_TITLE.search(t)
                                                                         for t in d["title"].get(v, []))}}
    for v in sorted(CLANNAD_FAMILY):
        why.setdefault(v, "clannad-family")
    anyrel = d.get("any_relations", d["relations"])
    rel = set()
    for v in CLANNAD:
        rel |= set(anyrel.get(v, ()))
    rel |= {v for v, rs in anyrel.items() if rs & CLANNAD}
    for v in CLANNAD_FAMILY:
        rel |= set(d["relations"].get(v, ()))
    rel |= {v for v, rs in d["relations"].items() if rs & CLANNAD_FAMILY}
    for v in sorted(rel - set(OWNER_ALLOWED)):
        why.setdefault(v, "clannad-relation")
    return why


def set_games(tg=None):
    """The games of the two scored test sets: vita-test's 16 and psp-test's 14 (28 VNs; two are in both)."""
    tg = test_games() if tg is None else tg
    return set(tg["vita-test"]) | set(tg["psp-test"])


def test_reservation(d, sets=None):
    """{vn: reason} of the test reservation (owner 2026-10-14 13:25Z: the only test sets are vita-test, psp-test
    and the CLANNAD family, "with those exact VNs plus their shared releases"), first reason wins:

      clannad, clannad-family, clannad-relation   clannad_vns(): the CLANNAD family
      test-set                 a vita-test or psp-test game
      clannad-shared-release   on one VNDB release with a CLANNAD-family VN (the Key boxes: Kanon, AIR, Little Busters!)
      test-shared-release      on one VNDB release with a test-set game (compilations, an original and its fan disc
                               on one disc, bundles of a series)

    OWNER_ALLOWED VNs (planetarian, v34, on the Key 10th Memorial BOX with CLANNAD: owner 2026-10-13 19:21Z and
    20:47Z) never get a shared-release reason. Nothing else is reserved: not the old hash splits (synth/common.
    is_test_vn, val_bucket), not the set games' other VNDB relations, not teacher_val, devnative, the fresh native
    textbox titles or the owner's former test series."""
    sets = set_games() if sets is None else set(sets)
    why = clannad_vns(d)
    for v in sorted(sets, key=lambda v: int(v[1:])):
        why.setdefault(v, "test-set")
    croot = {v for v, r in why.items() if r.startswith("clannad")}
    for u in sorted(release_partners(d, croot)):
        if u not in OWNER_ALLOWED:
            why.setdefault(u, "clannad-shared-release")
    for u in sorted(release_partners(d, sets)):
        if u not in OWNER_ALLOWED:
            why.setdefault(u, "test-shared-release")
    return why


def committed_reservation(path=None):
    """eval/manifests/reserved.json ({"test": {...}, "dev_sample": {...}, ...}), or None before it exists."""
    p = Path(path or ROOT / "eval" / "manifests" / "reserved.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def reserved_vns(d, sets=None, dev=None, committed=None):
    """{vn: reason} for every VN off limits to the training pools (docs/reservations.md):

      the test reservation    test_reservation(d): the CLANNAD family, the vita-test and psp-test games, their
                              release partners
      dev-sample              a VN of the development sample (eval/manifests/reserved.json "dev_sample"), kept out
                              of the training pools of any round that selects on it
      dev-sample-shared-release  on a two-VN release with a dev-sample VN (release_partners close=True: one game
                              filed twice, a game and its fan disc; OWNER_ALLOWED and owner-game VNs never)

    dev: the development sample's VNs (default: the committed file's). committed: the committed reservation
    (default: eval/manifests/reserved.json), unioned in so that a later dump never shrinks the reservation; pass
    False to compute from d alone. Everything else is training data."""
    why = test_reservation(d, sets)
    com = committed_reservation() if committed is None else (committed or None)
    if dev is None:
        dev = {v for v, r in (com or {}).get("dev_sample", {}).items() if r == "dev-sample"}
    for v in sorted(dev, key=lambda v: int(v[1:])):
        why.setdefault(v, "dev-sample")
    keep = set(OWNER_ALLOWED) | owner_vns()
    for u in sorted(release_partners(d, dev, close=True)):
        if u not in keep:
            why.setdefault(u, "dev-sample-shared-release")
    for part in ("test", "dev_sample"):
        for v, r in sorted((com or {}).get(part, {}).items()):
            why.setdefault(v, r)
    return why


# ---------------------------------------------------------------- the development sample

DEV_SAMPLE_SIZE = 100
DEV_SAMPLE_SEED = "dev-sample-20261014"
OWNER_GAMES = Path(os.environ.get("OWNER_GAMES") or HERD / "shared" / "private" / "owner-games")


def owner_vns():
    """Every VN the owner supplied screenshots of or named (owner 2026-10-13/14): OWNER_TRAINING, OWNER_ALLOWED, the
    former test series roots, the games of the owner's Drive drop and the owner-games folders (…-v123[-TEST]).
    All of their screenshots are training data (owner 13:25Z), so none of them is ever in the development sample."""
    out = set(OWNER_TRAINING) | set(OWNER_ALLOWED) | set(OWNER_TEST_SERIES)
    drop = ROOT / "data" / "manifests" / "owner_drop_20261013.json"
    if drop.exists():
        out |= {g["vn"] for g in json.loads(drop.read_text(encoding="utf-8")).get("games", []) if g.get("vn")}
    if OWNER_GAMES.exists():
        out |= {m.group(1) for p in OWNER_GAMES.iterdir() for m in [re.search(r"-(v\d+)(?:-TEST)?$", p.name)] if m}
    return {v for v in out if re.fullmatch(r"v\d+", v)}


def dev_line_counts(repo=ROOT):
    """{vn: Counter} of the labelled development lines that exist today, by source and kind: devnative's reviewed
    native lines (devnative/manifests/lines.jsonl), teacher_val's lines (teacher/manifests/lines_*.jsonl, split val)
    and realdata's real_val lines (realdata/manifests/lines_*.jsonl, split val)."""
    out = collections.defaultdict(collections.Counter)
    p = repo / "devnative" / "manifests" / "lines.jsonl"
    if p.exists():
        for x in p.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                out[r["vn"]][f"devnative:{r['kind']}"] += 1
    for src, name in (("teacher_val", "teacher"), ("real_val", "realdata")):
        for kind in KINDS:
            p = repo / name / "manifests" / f"lines_{kind}.jsonl"
            if not p.exists():
                continue
            for x in p.read_text(encoding="utf-8").splitlines():
                if x.strip():
                    r = json.loads(x)
                    if r.get("split") == "val":
                        out[r["id"].split("_")[0]][f"{src}:{kind}"] += 1
    return out


def training_pool_vns(repo=ROOT):
    """{vn: {pool names}} of the training pools built so far, from their committed manifests: teacher's and
    realdata's train splits and realdata's unlabelled pool, realplus's screenshots (v2 and v3 label the same ones),
    the native pool v1 and the owner's Drive-drop training games."""
    out = collections.defaultdict(set)

    def add(name, path, key="vn", where=lambda r: True):
        p = Path(path)
        if not p.exists():
            return
        for x in p.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                if where(r):
                    out[r[key] if key != "id" else r["id"].split("_")[0]].add(name)
    for kind in KINDS:
        add("teacher_train", repo / "teacher" / "manifests" / f"lines_{kind}.jsonl", "id",
            lambda r: r.get("split") == "train")
        add("real_train", repo / "realdata" / "manifests" / f"lines_{kind}.jsonl", "id",
            lambda r: r.get("split") == "train")
        add("real_unlabelled", repo / "realdata" / "manifests" / f"unlabelled_{kind}.jsonl", "id")
    for p in sorted((repo / "realplus" / "manifests").glob("shots_*.jsonl")):
        add("realplus", p)
    add("native_v1", repo / "realplus" / "manifests" / "native" / "lines_v1.jsonl")
    drop = repo / "data" / "manifests" / "owner_drop_20261013.json"
    if drop.exists():
        for g in json.loads(drop.read_text(encoding="utf-8")).get("games", []):
            if g.get("role") == "training" and g.get("vn"):
                out[g["vn"]].add("owner")
    return out


def dev_sample(d, test, counts=None, size=DEV_SAMPLE_SIZE, seed=DEV_SAMPLE_SEED, owner=None, pools=None):
    """The development sample: `size` VNs that already have labelled development lines, chosen by tier and, inside a
    tier, by a seeded hash of the VN id (stable_rank), so the choice of one VN never depends on another:

      tier 1  VNs with devnative's reviewed native console lines
      tier 2  VNs with native (Vita or PSP) teacher_val lines
      tier 3  VNs with scaled PC (hires) teacher_val lines
      tier 4  VNs with real_val lines only (realdata's 4-reader labels; teacher left them out as linked VNs)

    A VN is a candidate only when it is not in the test reservation, not an owner game (owner_vns()), shares no
    VNDB release with an owner game (their screenshots all train, owner 2026-10-14 13:25Z), is in no training pool
    built so far (training_pool_vns) and shares no two-VN release (one game filed twice, a game and its fan disc)
    with a VN of one, so the sample's games were never trained on. Returns (rows, why):
    one row per chosen VN (tier, lines by source and kind) and the count of every reason a candidate was passed over."""
    counts = dev_line_counts() if counts is None else counts
    owner = owner_vns() if owner is None else set(owner)
    near_owner = set(release_partners(d, owner))
    pools = training_pool_vns() if pools is None else pools
    near_pool = set(release_partners(d, set(pools), close=True))
    why = collections.Counter()

    def tier(c):
        if any(k.startswith("devnative:") for k in c):
            return 1
        if c["teacher_val:vita"] or c["teacher_val:psp"]:
            return 2
        return 3 if c["teacher_val:hires"] else 4

    cands = []
    for v, c in counts.items():
        if v in test:
            why["test"] += 1
        elif v in owner:
            why["owner-game"] += 1
        elif v in near_owner:
            why["owner-shared-release"] += 1
        elif v in pools:
            why["in-training-pool"] += 1
        elif v in near_pool:
            why["two-vn-release-with-a-training-pool-vn"] += 1
        else:
            cands.append((tier(c), stable_rank(seed, v), v))
    cands.sort()
    rows = [{"vn": v, "tier": t, "title": (d["title"].get(v) or [None])[0], "lines": dict(sorted(counts[v].items()))}
            for t, _, v in cands[:size]]
    why["not-drawn"] = max(0, len(cands) - size)
    return rows, why


def cmd_reserve(args):
    """eval/manifests/reserved.json (the test reservation, the development sample, the test sessions) from a dump,
    and test_games.json's "all" (the test-set games and their release partners)."""
    d = load_dump(Path(args.dump) if args.dump else None)
    stamp_file = (Path(args.dump) if args.dump else dump_dir()).parent / "TIMESTAMP"
    stamp = stamp_file.read_text().strip() if stamp_file.exists() else None
    tg = test_games()
    test = test_reservation(d)
    rows, passed = dev_sample(d, test)
    dev = {r["vn"]: "dev-sample" for r in rows}
    keep = set(OWNER_ALLOWED) | owner_vns()
    for u, rs in sorted(release_partners(d, dev, close=True).items()):
        if u not in test and u not in keep:
            dev.setdefault(u, "dev-sample-shared-release")
    partners = {u: rs for u, rs in release_partners(d, set_games(tg) | {v for v, r in test.items()
                                                                       if r.startswith("clannad")}).items()
                if u in test}
    total = collections.Counter()
    for r in rows:
        for k, n in r["lines"].items():
            total[k] += n
    out = {
        "kind": "vita-vn-ocr-reservation", "schema": 1,
        "policy": "owner 2026-10-14 13:25Z (docs/reservations.md)",
        "rule": ("The only test sets are vita-test, psp-test and the CLANNAD family. 'test' holds their VNs and every "
                 "VN on one VNDB release with one of them (planetarian, v34, excepted by the owner): never train, "
                 "tune, calibrate, select, label-tune on them or render synthetic scenes or names from them. "
                 "'sessions' are vita-test's and psp-test's capture sessions (VNDB release ids): no screenshot of "
                 "theirs is ever used. 'dev_sample' is the fixed development sample model selection uses, with its "
                 "release partners: keep it out of the training pools of any round that selects on it. Every "
                 "other VN is training data, including the old hash splits (synth/common.is_test_vn, "
                 "realplus/shots.val_bucket), teacher_val, devnative, the fresh native textbox titles, the owner's "
                 "former test series and every owner screenshot."),
        "dump": stamp, "code": "realplus/shots.py reserve",
        "test": dict(sorted(test.items(), key=lambda kv: int(kv[0][1:]))),
        "dev_sample": dict(sorted(dev.items(), key=lambda kv: int(kv[0][1:]))),
        "sessions": sorted(tg["sessions"], key=lambda r: int(r[1:])),
        "test_shared_releases": {u: partners[u] for u in sorted(partners, key=lambda v: int(v[1:]))},
        "counts": {"test": dict(collections.Counter(test.values())), "dev_sample": dict(collections.Counter(
            dev.values())), "dev_sample_lines": dict(sorted(total.items())),
                   "dev_sample_tiers": dict(collections.Counter(r["tier"] for r in rows)),
                   "dev_candidates_passed_over": dict(passed)},
        "dev_sample_size": DEV_SAMPLE_SIZE, "dev_sample_seed": DEV_SAMPLE_SEED,
        "dev_sample_vns": rows,
    }
    path = Path(args.out)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.test_games:
        tgp = ROOT / "eval" / "manifests" / "test_games.json"
        doc = json.loads(tgp.read_text(encoding="utf-8"))
        doc.setdefault("all_20261012", doc["all"])
        doc["all"] = sorted({v for v, r in test.items() if r in ("test-set", "test-shared-release")},
                            key=lambda v: int(v[1:]))
        doc["rule"] = TEST_GAMES_RULE
        order = ("rule", "all", "sessions", "family", "vita-test", "psp-test", "all_20261012")
        doc = {k: doc[k] for k in order if k in doc} | {k: v for k, v in doc.items() if k not in order}
        tgp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"dump": stamp, "test": len(test), "dev_sample": len(dev), **out["counts"]}, indent=1))


TEST_GAMES_RULE = (
    "Owner 2026-10-14 13:25Z (docs/reservations.md): a VN is a test game of these sets when its id is in 'all', the "
    "vita-test and psp-test games and every VN on one VNDB release with one of them; a screenshot is from a test "
    "capture session when its VNDB release id is in 'sessions'. Never train, tune, calibrate, select or render "
    "synthetic scenes or names from a test game or a test session, on any platform. The CLANNAD family is the other "
    "test set (eval/manifests/reserved.json). 'family' lists the set games' other VNDB relations for information: "
    "they are training data. 'all_20261012' is the old list (the set games, every synth/common.is_test_vn game with "
    "Vita or PSP screenshots, session partners and the family), kept as history; it reserves nothing.")


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
    test_img = eval_test_images()
    for name, d in screenshot_dirs():
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not p.is_file():
                continue
            if name in UNRESERVED_SETS:                 # only their vita-test / psp-test screenshots protect
                m = SHOT_FILE.match(p.name)
                if not (m and int(m.group(1)) in test_img):
                    continue
                name_ = "eval-test"
            else:
                name_ = name
            st = p.stat()
            key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
            h = old.get(key) or pixel_sha256(p)
            new[key] = h
            out[h].add(name_)
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
    p = sub.add_parser("reserve", help="write eval/manifests/reserved.json (docs/reservations.md)")
    p.add_argument("--dump", help="the dump's db directory (default: $REALPLUS/vndb/dump/db)")
    p.add_argument("--out", default=str(ROOT / "eval" / "manifests" / "reserved.json"))
    p.add_argument("--test-games", action="store_true", help="also rewrite test_games.json's 'all'")
    args = ap.parse_args()
    {"dump": cmd_dump, "select": cmd_select, "fetch": cmd_fetch, "dedupe": cmd_dedupe,
     "reserve": cmd_reserve}[args.cmd](args)


if __name__ == "__main__":
    main()

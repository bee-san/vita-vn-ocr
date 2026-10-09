#!/usr/bin/env python3
"""List, select and download VNDB screenshots of Japanese VNs for real training lines.

  vndb.py list   --platform psv|psp|pc [--top N]     -> $REALDATA/vndb/<platform>.jsonl
  vndb.py select [--per-vn-hires K]                  -> realdata/manifests/shots_<kind>.jsonl (no hashes yet)
  vndb.py fetch  [--delay S] [--mirror DIR ...]      -> $REALDATA/shots/<kind>/<vn>_<file>, fills sha256

A listing row is one screenshot of a Japanese-original VN, with the release it was
taken from (its platforms and languages). A manifest row is
  {vn, kind, url, dims, release, sexual, violence, sha256, session}
where session (assign_sessions) groups screenshots of one release uploaded together.
`select` keeps a screenshot only when

  * its VNDB flag averages are sexual < 0.3 and violence < 0.5 (the repo's filters),
  * its release is Japanese-only (so no English or Chinese text on screen),
  * the VN is not a test game: synth/common.is_test_vn(v), or listed in the eval
    agent's reserved list (realdata/eval_exclude.json, a copy of eval's test_games.json),
  * and it falls into one of three kinds:
      vita   a psv release at 960x544 (native Vita frame buffer)
      psp    a psp release at 480x272 (native PSP frame buffer)
      hires  a PC or home-console release (4:3 to 16:9, at least 640x480), later
             rescaled to Vita and PSP resolution by label.py

Every screenshot of a VN is in its listing, whatever the platform, so the psv and
psp listings also contribute PC shots. `fetch` stores files under $REALDATA (default
$HERD/shared/artifacts/realdata) and records each file's sha256 in the manifest, so
a later run (or another machine) can rebuild the exact inputs. Files already present
with the recorded hash are skipped; --mirror points at local copies to reuse
before downloading. Screenshots are copyrighted game images: keep them local.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "synth"))
from common import is_test_vn  # noqa: E402

API = "https://api.vndb.org/kana/vn"
UA = "vita-vn-ocr-realdata/0.1 (training-line listing, low volume; github.com/bee-san/vita-vn-ocr)"
FIELDS = ("id,title,alttitle,votecount,screenshots.url,screenshots.dims,screenshots.sexual,"
          "screenshots.violence,screenshots.votecount,screenshots.release.id,"
          "screenshots.release.platforms,screenshots.release.languages.lang")
KINDS = ("vita", "psp", "hires")
MANIFESTS = {k: HERE / "manifests" / f"shots_{k}.jsonl" for k in KINDS}
# The eval agent's reserved test games: its merged manifest once it exists, plus the
# provisional copy taken while eval was in review. The union is excluded.
EVAL_EXCLUDE = [ROOT / "eval" / "manifests" / "test_games.json", HERE / "eval_exclude.json"]
SEXUAL_MAX, VIOLENCE_MAX = 0.3, 0.5
# PC and home consoles whose screenshots are rescaled; handhelds and phones are left out
# (odd aspect ratios, and psv/psp are used natively).
HIRES_PLATFORMS = {"win", "mac", "lin", "dos", "p98", "p88", "x68", "fmt", "swi", "ps2", "ps3",
                   "ps4", "ps5", "psx", "ps1", "xb3", "xbo", "xxs", "sat", "drc", "dvd", "wii", "wiu"}


def realdata_dir():
    herd = os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")
    return Path(os.environ.get("REALDATA") or Path(herd) / "shared" / "artifacts" / "realdata")


def excluded_vns():
    """Test games of the eval sets: the reserved ids, on top of the is_test_vn() rule."""
    out = set()
    for path in EVAL_EXCLUDE:
        if path.exists():
            out |= set(json.loads(path.read_text(encoding="utf-8"))["all"])
    return out


def is_excluded(vn, extra):
    return is_test_vn(vn) or vn in extra


def post(session, body):
    for attempt in range(8):
        r = session.post(API, json=body, timeout=120)
        if r.status_code == 429:
            time.sleep(60 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("VNDB kept rate-limiting")


def cmd_list(args):
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = UA
    if args.platform == "pc":
        rel = ["and", ["platform", "=", "win"], ["lang", "=", "ja"]]
        sort, reverse = "votecount", True          # most-voted first; --top bounds the walk
    else:
        rel = ["and", ["platform", "=", args.platform], ["lang", "=", "ja"]]
        sort, reverse = "id", False
    rows, page, n_vn = [], 1, 0
    while True:
        body = {"filters": ["and", ["olang", "=", "ja"], ["release", "=", rel]], "fields": FIELDS,
                "sort": sort, "reverse": reverse, "results": 100, "page": page}
        d = post(s, body)
        for vn in d["results"]:
            n_vn += 1
            for sh in vn.get("screenshots") or []:
                r = sh.get("release") or {}
                rows.append({"vn": vn["id"], "title": vn["title"], "alttitle": vn.get("alttitle"),
                             "vn_votes": vn.get("votecount"), "url": sh["url"], "dims": sh["dims"],
                             "sexual": sh["sexual"], "violence": sh["violence"],
                             "flag_votes": sh.get("votecount"), "release": r.get("id"),
                             "platforms": r.get("platforms") or [],
                             "langs": [x["lang"] for x in r.get("languages") or []]})
        print(f"{args.platform} page {page}: {n_vn} VNs, {len(rows)} screenshots", flush=True)
        if not d.get("more") or (args.top and n_vn >= args.top):
            break
        page += 1
        time.sleep(1.2)
    out = realdata_dir() / "vndb" / f"{args.platform}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    tmp.rename(out)
    print(f"{len(rows)} screenshots of {n_vn} VNs -> {out}", flush=True)


def kind_of(row):
    p, (w, h) = set(row["platforms"]), row["dims"]
    if "psv" in p and (w, h) == (960, 544):
        return "vita"
    if "psp" in p and (w, h) == (480, 272):
        return "psp"
    if p & HIRES_PLATFORMS and not p & {"psv", "psp"} and w >= 640 and h >= 480 \
            and 1.25 <= w / h <= 1.85:
        return "hires"
    return None


def cmd_select(args):
    extra = excluded_vns()
    seen, keep, why = set(), [], {}
    for name in ("psv", "psp", "pc"):
        path = realdata_dir() / "vndb" / f"{name}.jsonl"
        if not path.exists():
            print(f"missing listing {path}; run `vndb.py list --platform {name}` first", flush=True)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            k = kind_of(r)
            reason = ("test-game" if is_excluded(r["vn"], extra) else
                      "rating" if r["sexual"] >= SEXUAL_MAX or r["violence"] >= VIOLENCE_MAX else
                      "language" if r["langs"] != ["ja"] else
                      "kind" if k is None else None)
            if reason:
                why[reason] = why.get(reason, 0) + 1
                continue
            keep.append({"vn": r["vn"], "kind": k, "url": r["url"], "dims": r["dims"],
                         "release": r["release"], "sexual": r["sexual"], "violence": r["violence"],
                         "sha256": None})
    # hires shots: at most K per VN (random but seeded), so popular PC VNs don't dominate.
    rng = random.Random(20261009)
    by_vn = {}
    for r in keep:
        if r["kind"] == "hires":
            by_vn.setdefault(r["vn"], []).append(r)
    drop = set()
    for vn, rs in by_vn.items():
        rng.shuffle(rs)
        drop |= {r["url"] for r in rs[args.per_vn_hires:]}
    why["hires-cap"] = len(drop)
    keep = [r for r in keep if r["url"] not in drop]
    old = {r["url"]: r["sha256"] for r in load_shots()}
    for r in keep:
        r["sha256"] = old.get(r["url"])
    keep.sort(key=lambda r: (r["kind"], int(r["vn"][1:]), r["url"]))
    save_shots(keep)
    for k in ("vita", "psp", "hires"):
        rs = [r for r in keep if r["kind"] == k]
        print(f"{k}: {len(rs)} screenshots from {len({r['vn'] for r in rs})} VNs", flush=True)
    print(f"dropped: {why}; eval reserved ids: {len(extra)}", flush=True)


def load_shots():
    """Every selected screenshot (all kinds), as written by `select` and hashed by `fetch`."""
    rows = []
    for k in KINDS:
        if MANIFESTS[k].exists():
            rows += [json.loads(x) for x in MANIFESTS[k].read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


SESSION_GAP = 50


def shot_number(url):
    """VNDB's screenshot id (the file name: https://t.vndb.org/sf/44/112044.jpg -> 112044)."""
    return int(url.rsplit("/", 1)[-1].split(".")[0])


def assign_sessions(rows):
    """Give every screenshot a capture-session id, "<vn>/<release>/<n>": screenshots of one
    release whose VNDB ids are consecutive (gaps under SESSION_GAP) were uploaded together,
    i.e. taken in one sitting. Splits are by VN, so a session never straddles a split."""
    by = {}
    for r in rows:
        by.setdefault((r["vn"], r["release"] or "r0"), []).append(r)
    for (vn, rel), rs in by.items():
        rs.sort(key=lambda r: shot_number(r["url"]))
        n, last = 0, None
        for r in rs:
            num = shot_number(r["url"])
            if last is not None and num - last >= SESSION_GAP:
                n += 1
            r["session"] = f"{vn}/{rel}/{n}"
            last = num
    return rows


def save_shots(rows):
    assign_sessions(rows)
    MANIFESTS["vita"].parent.mkdir(parents=True, exist_ok=True)
    for k in KINDS:
        tmp = MANIFESTS[k].with_name(MANIFESTS[k].name + ".part")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["kind"] == k),
                       encoding="utf-8")
        tmp.replace(MANIFESTS[k])


def shot_key(r):
    """<vn>_<VNDB screenshot id>: the file stem, and the prefix of every line id."""
    return f"{r['vn']}_{r['url'].rsplit('/', 1)[-1].rsplit('.', 1)[0]}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def shot_path(r, base=None):
    return (base or realdata_dir()) / "shots" / r["kind"] / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}"


def cmd_fetch(args):
    import requests
    rows = load_shots()
    s = requests.Session()
    s.headers["User-Agent"] = UA
    got = reused = bad = 0
    for i, r in enumerate(rows):
        if args.kind and r["kind"] != args.kind:
            continue
        dest = shot_path(r)
        if dest.exists() and (r["sha256"] is None or sha256(dest) == r["sha256"]):
            r["sha256"] = r["sha256"] or sha256(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        name = dest.name
        mirror = next((Path(m) / name for m in args.mirror if (Path(m) / name).exists()), None)
        if mirror is not None and (r["sha256"] is None or sha256(mirror) == r["sha256"]):
            shutil.copyfile(mirror, dest)
            reused += 1
        else:
            ok = False
            for attempt in range(5):
                try:
                    resp = s.get(r["url"], timeout=60)
                    if resp.status_code == 429:
                        time.sleep(30 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    tmp = dest.with_name(name + ".part")
                    tmp.write_bytes(resp.content)
                    tmp.replace(dest)
                    ok = True
                    break
                except requests.RequestException as e:
                    print(f"retry {attempt + 1} {r['url']}: {e}", flush=True)
                    time.sleep(5 * (attempt + 1))
            time.sleep(args.delay)
            if not ok:
                bad += 1
                continue
            got += 1
        h = sha256(dest)
        if r["sha256"] and h != r["sha256"]:
            print(f"WRONG HASH {dest}", flush=True)
            bad += 1
        r["sha256"] = h
        if (got + reused) % 200 == 0:
            print(f"{i + 1}/{len(rows)}: {got} downloaded, {reused} from mirrors", flush=True)
            save_shots(rows)
    save_shots(rows)
    print(f"fetch: {len(rows)} screenshots, {got} downloaded, {reused} from mirrors, {bad} failed", flush=True)
    print("FETCH_DONE" if not bad else "FETCH_INCOMPLETE", flush=True)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--platform", choices=["psv", "psp", "pc"], required=True)
    p.add_argument("--top", type=int, default=0, help="stop after this many VNs (pc: most-voted first)")
    p = sub.add_parser("select")
    p.add_argument("--per-vn-hires", type=int, default=4)
    p = sub.add_parser("fetch")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between downloads")
    p.add_argument("--mirror", action="append", default=[], help="directory of already-downloaded shots")
    p.add_argument("--kind", choices=["vita", "psp", "hires"], help="only this kind")
    args = ap.parse_args()
    {"list": cmd_list, "select": cmd_select, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    main()

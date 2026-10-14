#!/usr/bin/env python3
"""Leak check of training pools against the reservation (docs/reservations.md; owner 2026-10-14 13:25Z).

  reserve_check.py NAME=PATH [NAME=PATH ...] [--dump DB] [--pixels] [--dev-sample block|info] [--out FILE]

A pool is a .npz with a "meta" list (teacher_*.npz, realplus_train_*.npz, native/owner pools, devnative), a
.jsonl manifest (lines, shots, reads or label files: rows with any of vn, id, shot, image, url, release, session,
sha256, pixel_sha256, rgb_sha256) or a directory of screenshots named <vn>_<image id>.<ext>. Every row is checked;
the blocker checks must all be empty, else the verdict is RESERVE_FAIL and the exit status 1:

  test VN               its VN is in eval/manifests/reserved.json "test": the CLANNAD family (CLANNAD, every VN
                        titled CLANNAD, the family list, CLANNAD's relations of every kind and the family list's
                        FAMILY relations), the vita-test and psp-test games, and every VN on one VNDB release with
                        one of those (planetarian, v34, excepted by the owner)
  test VN (dump)        --dump: the same rule recomputed on that VNDB dump (realplus/shots.test_reservation), so a
                        newer dump's titles, relations and releases count too; plus any release (session) that
                        lists a test VN
  test session          its release is one of vita-test's / psp-test's capture sessions (test_games.json)
  test screenshot       its VNDB image id, URL or file sha256 is one of vita-test's / psp-test's screenshots
  test pixels           --pixels: its decoded-pixel sha256 (devnative's, shots.pixel_sha256) equals that of a
                        protected image on disk: vita-test / psp-test screenshots, the owner's CLANNAD frames
  development sample    its VN is in reserved.json "dev_sample" (--dev-sample info: reported only, for a final
                        refit that trains on the development sample on purpose)

Information only (training data under the policy, reported so the change is visible): rows of the old hash splits
(synth/common.is_test_vn, the val bucket), of a set game's other VNDB relations (test_games.json "family"), and of
the former reservations (teacher_val's and devnative's VNs, the fresh native textbox titles, the owner's former
test series). This extends realplus-v3's hygiene.py (feat/realplus-v3) to the new policy: the same VN, release,
image, hash and pixel checks, with the test reservation's release partners and the development sample added and
the hash splits moved to information.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "synth", ROOT / "realplus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

import common  # noqa: E402

SHOT_KEY = re.compile(r"^(v\d+)_(\d+)")
URL_ID = re.compile(r"sf/\d\d/(\d+)\.jpg")
SESSION_REL = re.compile(r"(?:^|/)(r\d+)(?:/|$)")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BLOCKERS = ("test VN", "test VN (dump)", "test session", "test screenshot", "test pixels", "development sample")


def row_facts(r):
    """(vn, image id, release, {hashes}, key) of one pool row, any of them None when the row lacks it."""
    vn = r.get("vn")
    key = r.get("shot") or r.get("id") or r.get("file") or ""
    m = SHOT_KEY.match(str(key))
    image = r.get("image") if isinstance(r.get("image"), int) else None
    if image is None and m:
        image = int(m.group(2))
    if image is None and r.get("url"):
        u = URL_ID.search(str(r["url"]))
        image = int(u.group(1)) if u else None
    vn = vn or (m.group(1) if m else None)
    rel = r.get("release")
    if not rel and r.get("session"):
        s = SESSION_REL.search(str(r["session"]))
        rel = s.group(1) if s else None
    hashes = {r[k] for k in ("sha256", "pixel_sha256", "rgb_sha256") if isinstance(r.get(k), str)}
    return vn, image, rel, hashes, str(key)


def load_pool(path):
    """The rows of a pool file or directory, as dicts."""
    p = Path(path)
    if p.is_dir():
        out = []
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in IMAGE_EXT:
                m = SHOT_KEY.match(f.name)
                out.append({"file": f.name, "vn": m.group(1) if m else None, "path": str(f)})
        return out
    if p.suffix == ".npz":
        import numpy as np
        with np.load(p) as z:
            return list(json.loads(str(z["meta"])))
    if p.suffix == ".json":
        doc = json.loads(p.read_text(encoding="utf-8"))
        return doc if isinstance(doc, list) else doc.get("rows") or doc.get("records") or []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def protected(repo=ROOT, reservation=None):
    """The protected identities: test VNs, development-sample VNs, test sessions, test screenshot ids/URLs/sha256s."""
    res = common.reservation() if reservation is None else reservation
    imgs, urls, shas = set(), set(), set()
    for s in ("vita-test", "psp-test"):
        p = repo / "eval" / "manifests" / f"{s}.jsonl"
        for x in p.read_text(encoding="utf-8").splitlines() if p.exists() else ():
            if x.strip():
                r = json.loads(x)
                urls.add(r["url"])
                shas.add(r["sha256"])
                m = URL_ID.search(r["url"])
                if m:
                    imgs.add(int(m.group(1)))
    return {"test": dict(res["test"]), "dev": dict(res["dev_sample"]), "sessions": set(res["sessions"]),
            "images": imgs, "urls": urls, "sha256": shas}


def dump_test(db):
    """The test reservation recomputed on a dump, and every release id that lists a test VN."""
    d = SH.load_dump(Path(db))
    test = SH.test_reservation(d)
    rels = {rid for rid, rv in d["rel_vns"].items() if rv & set(test)}
    return test, rels


def protected_pixels(repo=ROOT, owner_games=None):
    """{pixel sha256: names} of the protected images on disk: vita-test / psp-test screenshots in eval's local
    copies ($HERD/shared/artifacts/eval/{vita,psp}/shots) and the owner's CLANNAD frames."""
    test_img = SH.eval_test_images(repo)
    out = collections.defaultdict(set)
    for k in ("vita", "psp"):
        d = SH.ART / "eval" / k / "shots"
        for f in sorted(d.glob("*")) if d.exists() else ():
            m = SH.SHOT_FILE.match(f.name)
            if m and int(m.group(1)) in test_img:
                out[SH.pixel_sha256(f)].add(f"{k}-test:{f.name}")
    og = Path(owner_games) if owner_games else SH.OWNER_GAMES
    for d in sorted(og.glob("*clannad*")) if og.exists() else ():
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMAGE_EXT:
                try:
                    out[SH.pixel_sha256(f)].add(f"owner-clannad:{f.name}")
                except ValueError:
                    continue
    return out


def check_pool(rows, prot, dump=None, pixels=None, dev_mode="block", family=None):
    """{check: {"rows": n, "vns": [...], "examples": [...]}} over one pool's rows; blockers and information."""
    hits = collections.defaultdict(lambda: {"rows": 0, "vns": set(), "examples": []})
    dump_test_vns, dump_rels = dump if dump else ({}, set())
    family = family or set()
    info_sets = former_reservations()

    def hit(name, vn, key):
        h = hits[name]
        h["rows"] += 1
        if vn:
            h["vns"].add(vn)
        if len(h["examples"]) < 5:
            h["examples"].append(key or vn)
    vns = set()
    for r in rows:
        vn, image, rel, hashes, key = row_facts(r)
        vns.add(vn)
        if vn in prot["test"]:
            hit("test VN", vn, key)
        if dump and (vn in dump_test_vns or (rel and rel in dump_rels)):
            hit("test VN (dump)", vn, key)
        if rel and rel in prot["sessions"]:
            hit("test session", vn, key)
        if (image is not None and image in prot["images"]) or r.get("url") in prot["urls"] or hashes & prot["sha256"]:
            hit("test screenshot", vn, key)
        if pixels and hashes & set(pixels):
            hit("test pixels", vn, key)
        if pixels and r.get("path") and SH.pixel_sha256(r["path"]) in pixels:
            hit("test pixels", vn, key)
        if vn in prot["dev"]:
            hit("development sample" if dev_mode == "block" else "development sample (info)", vn, key)
        if vn and common.is_test_vn(vn):
            hit("info: old is_test_vn split", vn, key)
        if vn and SH.val_bucket(vn):
            hit("info: old val bucket", vn, key)
        if vn in family:
            hit("info: set-game family (training now)", vn, key)
        for name, s in info_sets.items():
            if vn in s:
                hit(f"info: {name}", vn, key)
    out = {k: {"rows": v["rows"], "vns": len(v["vns"]), "vn_list": sorted(v["vns"])[:50], "examples": v["examples"]}
           for k, v in sorted(hits.items())}
    return {"rows": len(rows), "vns": len(vns - {None}), "checks": out,
            "blockers": sorted(k for k in out if k in BLOCKERS)}


def former_reservations():
    """VN sets the old rules reserved that are training data now (information)."""
    out = {"former fresh native titles": set(SH.FRESH_NATIVE),
           "former owner test series roots": set(SH.OWNER_TEST_SERIES),
           "former development VNs (teacher_val, devnative, general-VN)": set(SH.dev_vns())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+", help="NAME=PATH (npz, jsonl, json or a screenshot directory)")
    ap.add_argument("--dump", help="a VNDB dump's db directory: recompute the test reservation on it too")
    ap.add_argument("--pixels", action="store_true", help="also compare decoded pixels with protected images")
    ap.add_argument("--dev-sample", choices=("block", "info"), default="block")
    ap.add_argument("--out")
    args = ap.parse_args()
    prot = protected()
    dump = dump_test(args.dump) if args.dump else None
    pix = protected_pixels() if args.pixels else None
    tg = SH.test_games()
    family = {v for ids in tg.get("family", {}).values() for v in ids} - set(prot["test"])
    res = {"policy": common.reservation()["policy"], "reservation_dump": common.reservation()["dump"],
           "dump": args.dump, "pixels": None if pix is None else {"protected_images": sum(map(len, pix.values())),
                                                                   "distinct": len(pix)},
           "dev_sample_mode": args.dev_sample, "pools": {}}
    for spec in args.pools:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        res["pools"][name] = {"path": path, **check_pool(load_pool(path), prot, dump, pix, args.dev_sample, family)}
        r = res["pools"][name]
        print(f"{name}: {r['rows']} rows, {r['vns']} VNs, blockers {r['blockers'] or 'none'}", flush=True)
    bad = {n: r["blockers"] for n, r in res["pools"].items() if r["blockers"]}
    res["verdict"] = "RESERVE_FAIL" if bad else "RESERVE_OK"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(res["verdict"], flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

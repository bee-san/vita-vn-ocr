#!/usr/bin/env python3
"""The owner's Google Drive drop of 2026-10-13 (vn-ocr-dataset-20261013.zip): identify, check and sort.

  owner_drop.py check  --drop DIR [--dump DB] [--leak] --out PLAN.json
  owner_drop.py sort   --plan PLAN.json [--dest DIR]
  owner_drop.py public --plan PLAN.json --out JSON

The drop (owner, 2026-10-13 19:21Z and 19:27Z) is private and copyrighted. It stays under shared/private/,
never in git, and never goes to Codex or any other third party. Its samples.jsonl lists every full frame
(title, platform, image, dimensions, sha256); vita/CLANNAD/training-data/ is the CLANNAD long run
(eval/clannad_longrun.py's session, with the published pins of eval/vocr3_clannad.py).

check   per full frame: the decoded size (the source resolution; every frame of this drop is a native
        capture, 960x544 Vita or 480x272 PSP, never an Adrenaline upscale) and three hashes (the file's, PIL's
        decoded RGB as the fresh-native harness hashes it, devnative's). Per game: its VNDB id (GAMES:
        identified on VNDB's 2026-10-13 dump by title, platform and the Japanese console release),
        realplus/shots.reserved_vns's reason on the dump, the VN's rows in eval's and the development sets
        (vita-test, psp-test, test_games.json "all", is_test_vn, teacher_val, devnative) and, with --leak, each
        frame's best thumbnail correlation (eval/tools/leakcheck.thumb) with every screenshot of eval's
        candidates, devnative's cohort and teacher_val. decide() gives each game a role:
          test      CLANNAD and its family (protected, owner); else any reserved VN (the VNDB check pattern);
          training  an unreserved VN, or an owner decision (OWNER_DECISIONS) that is in no eval or
                    development set;
          blocked   an owner decision whose VN is in an eval or development set, or a training game with a
                    frame at or above MIN_CORR against a protected screenshot.
sort    hardlinks each game's frames into owner-games/<platform>-<slug>-v<id>[-TEST]/ (an existing folder of
        the same VN and role is reused) with manifest.jsonl (per frame: file, drop path, hashes, size, source
        resolution, sequence) and ingest.json (the decision). It never overwrites or removes a file.
public  the per-game table without any frame's path or hash (data/manifests/owner_drop_20261013.json).
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "realplus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

from common import is_test_vn  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
OWNER_GAMES = HERD / "shared" / "private" / "owner-games"
DUMP = ART / "native-frames" / "vndb" / "dump" / "db"
NATIVE = {"vita": (960, 544), "psp": (480, 272)}
MIN_CORR = 0.95          # eval/tools/leakcheck.py's default: a near-duplicate of a protected screenshot

# The drop's folders, identified on VNDB's 2026-10-13 dump (title, platform, the Japanese release; the capture
# slot names the Vita title id). `release` is the Japanese console release; `release_vns` its other VNs.
GAMES = {
    ("psp", "Amagami"): {"vn": "v1281", "slug": "amagami", "title": "アマガミ (Amagami)",
                         "release": "r9959 エビコレ＋アマガミ (ULJS-00339)"},
    ("psp", "Corpse-Party-Blood-Covered"): {"vn": "v2632", "slug": "corpse-party", "title": "Corpse Party BloodCovered",
                                            "release": "r9521 (ULJM-05704)"},
    ("psp", "Grisaia-no-Kajitsu"): {"vn": "v5154", "slug": "grisaia-no-kajitsu", "title": "グリザイアの果実",
                                    "release": "r15389 (ULJM-06232)"},
    ("psp", "Narcissu-Portable"): {"vn": "v10", "slug": "narcissu-portable",
                                   "title": "ナルキッソス Portable (narcissu)", "release": "r7480 (ULJM-05674)",
                                   "release_vns": ["v70", "v1542", "v18736", "v18738"]},
    ("psp", "Planetarian"): {"vn": "v34", "slug": "planetarian", "title": "planetarian ～ちいさなほしのゆめ～",
                             "release": "r6626 (ULJM-05383)"},
    ("psp", "Sharin-no-Kuni"): {"vn": "v57", "slug": "sharin-no-kuni", "title": "車輪の国、向日葵の少女",
                                "release": "r4693 UMD-PG Edition or r17721 (ULJM-06022, with v84)"},
    ("vita", "CLANNAD"): {"vn": "v4", "slug": "clannad", "title": "CLANNAD",
                          "release": "r33730 (VLJM-35124, PCSG00415)"},
    ("vita", "Chaos-Child"): {"vn": "v14018", "slug": "chaos-child", "title": "CHAOS;CHILD",
                              "release": "r39462 (VLJM-35209, PCSG00555)"},
    ("vita", "Danganronpa-V3"): {"vn": "v18334", "slug": "danganronpa-v3", "title": "NEWダンガンロンパV3",
                                 "release": "r47935 (VLJS-05097, PCSG00936)"},
    ("vita", "Iwaihime-Matsuri"): {"vn": "v17863", "slug": "iwaihime-matsuri", "title": "祝姫 -祀-",
                                   "release": "r51223 (VLJM-38010, PCSG01083)"},
    ("vita", "Rewrite"): {"vn": "v751", "slug": "rewrite", "title": "Rewrite",
                          "release": "r27566 (VLJM-35125, PCSG00414)"},
}
# Owner decisions that override the VNDB check pattern. They never override an eval or development set.
OWNER_DECISIONS = {
    "v34": {"role": "training", "by": "owner 2026-10-13 19:21Z and 20:47Z",
            "note": "planetarian (Key, all releases) is training data, allowed even where it overlaps an eval set "
                    "(realplus/shots.OWNER_ALLOWED). Without that exemption reserved_vns reserves it only as "
                    "shared-release (the Key 10th Memorial BOX, r3748, a PC compilation with CLANNAD and Tomoyo "
                    "After); it has no rows in vita-test, psp-test, teacher_val or devnative. Its PSP port and "
                    "CLANNAD's Vita port are both Key/Prototype releases: a CLANNAD score of any model trained on "
                    "these frames must disclose that exposure."},
}
CLANNAD_REASONS = {"clannad", "clannad-family", "clannad-relation"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_facts(path):
    """(width, height, PIL-RGB sha256, devnative pixel sha256) of one image."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        a = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
    h = hashlib.sha256(f"{a.shape[0]}x{a.shape[1]}x3".encode())
    h.update(a.tobytes())
    return a.shape[1], a.shape[0], hashlib.sha256(a.tobytes()).hexdigest(), h.hexdigest()


def source_resolution(platform, w, h):
    return "native" if NATIVE.get(platform) == (w, h) else f"non-native {w}x{h}"


def read_samples(drop):
    rows = [json.loads(x) for x in (Path(drop) / "samples.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        p = Path(r["image"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe sample path {r['image']}")
    return rows


# ----------------------------------------------------------------------------- the sets a VN may sit in

def npz_vns(path):
    """Counter(vn -> rows) of a line pool's meta table (an npz without one holds no lines: empty)."""
    import numpy as np
    z = np.load(path)
    if "meta" not in z.files:
        return collections.Counter()
    return collections.Counter(m.get("vn") for m in json.loads(str(z["meta"])))


def set_rows(root=ROOT, art=ART):
    """{set name: Counter(vn -> rows)} of every evaluation and development set on disk."""
    out = {}
    for s in ("vita-test", "psp-test"):
        p = root / "eval" / "manifests" / f"{s}.jsonl"
        if p.exists():
            out[s] = collections.Counter(json.loads(x)["vn"] for x in p.read_text(encoding="utf-8").splitlines()
                                         if x.strip())
    p = art / "teacher" / "teacher_val.npz"
    if p.exists():
        out["teacher_val"] = npz_vns(p)
    dn = collections.Counter()
    for p in sorted((art / "devnative").rglob("*.npz")) if (art / "devnative").exists() else ():
        dn += npz_vns(p)
    for p in sorted((root / "devnative" / "manifests").glob("*.jsonl")):
        dn += collections.Counter(json.loads(x).get("vn") for x in p.read_text(encoding="utf-8").splitlines()
                                  if x.strip())
    out["devnative"] = dn
    return out


def eval_membership(vn, sets, test_all, dev, fresh):
    """The eval / development sets a VN is in (empty: none)."""
    hit = {k: c[vn] for k, c in sets.items() if c.get(vn)}
    if vn in test_all:
        hit["test_games_all"] = True
    if is_test_vn(vn):
        hit["is_test_vn"] = True
    if vn in dev:
        hit["development_vns"] = True
    if vn in fresh:
        hit["fresh_native"] = True
    return hit


EVAL_KEYS = ("vita-test", "psp-test", "teacher_val", "devnative", "test_games_all", "fresh_native")


def decide(vn, reason, sets_hit, owner=OWNER_DECISIONS, leak_frames=0):
    """(role, why) of one game; role is training, test or blocked."""
    if vn in SH.CLANNAD or vn in SH.CLANNAD_FAMILY or reason in CLANNAD_REASONS:
        return "test", f"protected: CLANNAD ({reason or 'clannad'}), test only (owner)"
    in_eval = sorted(k for k in sets_hit if k in EVAL_KEYS)
    if vn in owner:
        o = owner[vn]
        if in_eval:
            return "blocked", f"{o['by']} says {o['role']}, but the VN is in {', '.join(in_eval)}: stopped"
        role, why = o["role"], f"{o['by']}: {o['role']} (reserved_vns: {reason or 'not reserved'})"
    elif reason:
        role, why = "test", f"reserved: {reason}"
    elif in_eval:
        role, why = "test", f"in {', '.join(in_eval)}"
    else:
        role, why = "training", "not reserved, in no eval or development set"
    if role == "training" and leak_frames:
        return "blocked", f"{leak_frames} frame(s) at or above {MIN_CORR} against a protected screenshot"
    return role, why


def folder_name(platform, slug, vn, role, existing=()):
    """<platform>-<slug>-v<id>[-TEST]; an existing folder of the same VN and role wins."""
    test = role != "training"
    pat = re.compile(rf"-{re.escape(vn)}{'-TEST' if test else ''}$")
    same = sorted(n for n in existing if pat.search(n))
    return same[0] if same else f"{platform}-{slug}-{vn}{'-TEST' if test else ''}"


# ----------------------------------------------------------------------------- leak check (thumbnails)

def protected_shots(art=ART):
    """{group: [paths]}: eval's candidate screenshots (every test screenshot is one), devnative's cohort,
    teacher_val's screenshots."""
    dn = art / "devnative" / "shots"
    out = {"eval-vita": sorted((art / "eval" / "vita" / "shots").glob("*.jpg")),
           "eval-psp": sorted((art / "eval" / "psp" / "shots").glob("*.jpg")),
           "devnative": sorted(p for p in dn.rglob("*") if p.suffix.lower() in (".jpg", ".png")) if dn.exists() else []}
    tv = art / "teacher" / "teacher_val.npz"
    if tv.exists():
        import numpy as np
        meta = json.loads(str(np.load(tv)["meta"]))
        paths = {art / "realdata" / "shots" / m["kind"] / ("_".join(m["id"].split("_")[:2]) + ".jpg") for m in meta}
        out["teacher_val"] = sorted(p for p in paths if p.exists())
    return {k: v for k, v in out.items() if v}


def leak_scores(frames, groups):
    """Per frame index: {group: best thumbnail correlation}."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "eval" / "tools"))
    sys.path.insert(0, str(ROOT / "eval"))
    from leakcheck import thumb
    T = np.stack([thumb(p) for p in frames])
    out = [dict() for _ in frames]
    for g, paths in groups.items():
        M = np.stack([thumb(p) for p in paths])
        best = (T @ M.T).max(axis=1)
        for i, b in enumerate(best):
            out[i][g] = round(float(b), 4)
    return out


# ----------------------------------------------------------------------------- check

def check(drop, dump=DUMP, leak=False, owner_games=OWNER_GAMES, why=None, sets=None, why_strict=None):
    drop = Path(drop)
    samples = read_samples(drop)
    if why is None:
        d = SH.load_dump(dump)
        why = SH.reserved_vns(d)
        allowed, SH.OWNER_ALLOWED = SH.OWNER_ALLOWED, {}
        try:
            why_strict = SH.reserved_vns(d)                 # what the owner's exemptions lifted
        finally:
            SH.OWNER_ALLOWED = allowed
    why_strict = why if why_strict is None else why_strict
    test_all = set(SH.test_games()["all"])
    dev, fresh = SH.dev_vns(), SH.fresh_native_vns()
    sets = set_rows() if sets is None else sets
    existing = sorted(p.name for p in Path(owner_games).iterdir() if p.is_dir()) if Path(owner_games).exists() else []
    frames = []
    for r in samples:
        w, h, rgb, dn = frame_facts(drop / r["image"])
        if sha256_file(drop / r["image"]) != r["sha256"] or [w, h] != list(r["dimensions"]):
            raise ValueError(f"{r['image']}: hash or size differs from samples.jsonl")
        frames.append({"image": r["image"], "platform": r["platform"], "title": r["title"],
                       "collection": r["collection"], "sequence": r.get("sequence"), "sha256": r["sha256"],
                       "pixel_sha256": rgb, "dn_pixel_sha256": dn, "dims": [w, h],
                       "source_resolution": source_resolution(r["platform"], w, h)})
    if leak:
        for f, s in zip(frames, leak_scores([drop / f["image"] for f in frames], protected_shots())):
            f["leak"] = s
    by_game = collections.defaultdict(list)
    for f in frames:
        by_game[(f["platform"], f["title"])].append(f)
    games = []
    for key in sorted(set(GAMES) | set(by_game)):
        g = GAMES.get(key)
        if g is None:
            raise ValueError(f"unidentified game folder {key}: add it to GAMES")
        fs = by_game.get(key, [])
        vn = g["vn"]
        hit = eval_membership(vn, sets, test_all, dev, fresh)
        n_leak = sum(1 for f in fs if max((f.get("leak") or {"-": 0}).values()) >= MIN_CORR)
        role, reason = decide(vn, why.get(vn), hit, leak_frames=n_leak)
        dims = collections.Counter(tuple(f["dims"]) for f in fs)
        games.append({"platform": key[0], "drop_title": key[1], "vn": vn, "title": g["title"], "release": g["release"],
                      "release_vns": g.get("release_vns", []),
                      "release_vns_reserved": {v: why.get(v) for v in g.get("release_vns", []) if why.get(v)},
                      "frames": len(fs), "dims": {f"{a}x{b}": n for (a, b), n in sorted(dims.items())},
                      "source_resolution": sorted({f["source_resolution"] for f in fs}),
                      "collections": dict(collections.Counter(f["collection"] for f in fs)),
                      "reserved_vns": why.get(vn), "reserved_without_owner_allowed": why_strict.get(vn),
                      "sets": hit, "role": role, "why": reason,
                      "owner_decision": OWNER_DECISIONS.get(vn), "leak_frames": n_leak,
                      "leak_max": {k: max(f["leak"][k] for f in fs) for k in (fs[0].get("leak") or {})} if fs else {},
                      "folder": folder_name(key[0], g["slug"], vn, role, existing) if fs else None})
    return {"kind": "owner-drop-plan", "schema": 1, "drop": str(drop), "frames": frames, "games": games,
            "computed": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


# ----------------------------------------------------------------------------- sort

def dest_name(f):
    """The frame's file name in its game folder: the long-run CLANNAD frames keep inputs/'s name
    (NNNNNN-native.png), the current run's keep their six-digit sequence name."""
    return Path(f["image"]).name


def sort(plan, dest=OWNER_GAMES, link=True):
    """Hardlink (or copy) every frame into its game folder; writes manifest.jsonl and ingest.json there."""
    import shutil
    drop, dest = Path(plan["drop"]), Path(dest)
    games = {(g["platform"], g["drop_title"]): g for g in plan["games"]}
    done = collections.Counter()
    rows = collections.defaultdict(list)
    for f in plan["frames"]:
        g = games[(f["platform"], f["title"])]
        out = dest / g["folder"]
        out.mkdir(parents=True, exist_ok=True)
        target = out / dest_name(f)
        if target.exists():
            if sha256_file(target) != f["sha256"]:
                raise FileExistsError(f"{target} exists with other bytes; never overwritten")
        elif link:
            os.link(drop / f["image"], target)
        else:
            shutil.copy2(drop / f["image"], target)
        done[g["folder"]] += 1
        rows[g["folder"]].append({"file": target.name, "drop_image": f["image"], "sha256": f["sha256"],
                                  "pixel_sha256": f["pixel_sha256"], "dn_pixel_sha256": f["dn_pixel_sha256"],
                                  "dims": f["dims"], "source_resolution": f["source_resolution"],
                                  "platform": f["platform"], "vn": g["vn"], "collection": f["collection"],
                                  "sequence": f.get("sequence")})
    for folder, rs in rows.items():
        g = next(x for x in plan["games"] if x["folder"] == folder)
        names = [r["file"] for r in rs]
        if len(set(names)) != len(names):
            raise ValueError(f"{folder}: two frames share a file name")
        (dest / folder / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs),
                                                      encoding="utf-8")
        info = {k: g[k] for k in ("platform", "vn", "title", "release", "role", "why", "reserved_vns", "sets",
                                  "frames", "dims", "source_resolution", "owner_decision", "leak_max")}
        info.update(source="owner Google Drive drop 2026-10-13 (vn-ocr-dataset-20261013.zip)",
                    private="copyrighted frames: never commit, never send to Codex or any third party")
        (dest / folder / "ingest.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n",
                                                   encoding="utf-8")
    return dict(done)


# ----------------------------------------------------------------------------- public table

def public(plan, archive=None):
    keep = ("platform", "drop_title", "vn", "title", "release", "frames", "dims", "source_resolution", "collections",
            "reserved_vns", "reserved_without_owner_allowed", "release_vns_reserved", "sets", "role", "why",
            "leak_max", "leak_frames", "folder")
    out = {"kind": "owner-drop-summary", "schema": 1, "archive": archive or {},
           "frames": sum(g["frames"] for g in plan["games"]),
           "by_role": dict(collections.Counter(g["role"] for g in plan["games"] for _ in range(g["frames"]))),
           "games": [{k: g[k] for k in keep} for g in plan["games"]],
           "owner_decisions": OWNER_DECISIONS, "min_corr": MIN_CORR,
           "note": "counts, ids and reasons only: no frame, no text, no per-frame hash"}
    blob = json.dumps(out, ensure_ascii=False)
    if '"image"' in blob or ".png" in blob:
        raise ValueError("public table names a frame")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--drop", type=Path, required=True)
    p.add_argument("--dump", type=Path, default=DUMP)
    p.add_argument("--leak", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("sort")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--dest", type=Path, default=OWNER_GAMES)
    p.add_argument("--copy", action="store_true", help="copy instead of hardlinking")
    p = sub.add_parser("public")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--archive-sha256")
    p.add_argument("--archive-bytes", type=int)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "check":
        plan = check(a.drop, a.dump, a.leak)
        a.out.write_text(json.dumps(plan, ensure_ascii=False) + "\n", encoding="utf-8")
        for g in plan["games"]:
            print(f"{g['platform']:4} {g['drop_title']:28} {g['vn']:7} {g['frames']:5} {g['role']:8} {g['why']}")
    elif a.cmd == "sort":
        plan = json.loads(a.plan.read_text(encoding="utf-8"))
        print(json.dumps(sort(plan, a.dest, link=not a.copy), indent=1))
    else:
        plan = json.loads(a.plan.read_text(encoding="utf-8"))
        arc = {"name": "vn-ocr-dataset-20261013.zip", "sha256": a.archive_sha256, "bytes": a.archive_bytes}
        a.out.write_text(json.dumps(public(plan, arc), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

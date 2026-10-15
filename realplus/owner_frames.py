#!/usr/bin/env python3
"""The owner's live console screenshots (owner, 2026-10-13 10:51Z): inventory and read input.

  owner_frames.py inventory [--root DIR] [--out DIR]   every image under the owner's folders: title, role
                                                       (test or training), size, file sha256, decoded-pixel
                                                       sha256 (devnative's and the fresh-native harness's),
                                                       source resolution (a PSP frame scaled 2x to the Vita's
                                                       screen by Adrenaline, upscale_2x),
                                                       duplicates -> OUT/inventory.jsonl, summary.json
  owner_frames.py prep --folder NAME [--out DIR]       devnative/read.py's input for one folder's unique frames
                                                       -> OUT/NAME/reads/shots.jsonl
  owner_frames.py pool-prep --pool DIR [--folder NAME ...]
                                                       read.py's input for an owner training pool: the unique,
                                                       eligible frames of every training folder (or the named
                                                       ones) -> DIR/reads/shots.jsonl, DIR/manifests/frames.jsonl

Folders (shared/private/owner-games/): <console>-<title>-v<id>[-TEST]. Since 2026-10-14 (owner 13:25Z, docs/
reservations.md) every owner frame is training data unless its VN is in the test reservation (the CLANNAD family,
the vita-test and psp-test games and their release partners: eval/manifests/reserved.json "test"); the -TEST
suffix no longer decides (it is recorded as was_test_folder). Owner frames are private: never sent to Codex.
Everything stays under shared/private/: frames, readings, crops, labels, manifests.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

PRIVATE = SH.HERD / "shared" / "private" / "owner-games"
# Folder names: <console>-<title>-v<id>[-TEST]; the -TEST suffix marks a test-only title. memories-off-TEST
# names no id (its series' root is v1152).
FOLDER_VN = re.compile(r"-v(\d+)(?:-TEST)?$")
NAMED = {"memories-off-TEST": "v1152"}
# The owner's explicit decisions for a training title reserved_vns reserves. Each one is allowed only for the
# listed reservation reasons, so a test, development or CLANNAD reservation can never be overridden here.
OWNER_DECISIONS = {
    "v34": {"role": "training", "by": "owner 2026-10-13 19:21Z",
            "allowed": ("shared-release", "clannad-shared-release"),
            "note": "planetarian (Key) is training data. reserved_vns reserves it only as a shared-release partner "
                    "(the PC Key 10th Memorial BOX, r3748, with CLANNAD). It has no row in vita-test, psp-test, "
                    "teacher_val or devnative. Disclose that planetarian PSP and CLANNAD Vita are both "
                    "Key/Prototype ports whenever a model trained on it is scored on CLANNAD."},
}


def owner_allowed(vn, reason):
    """The owner decision that lets a reserved VN's owner frames train, or None."""
    d = OWNER_DECISIONS.get(vn)
    return d if d and d["role"] == "training" and reason in d["allowed"] else None


def folder_info(name, reserved=None):
    """{vn, role, eligible, why, was_test_folder} of an owner folder (docs/reservations.md): role "test" when its VN
    is in the test reservation (reserved: reserved_vns of a dump, else the committed eval/manifests/reserved.json),
    "training" otherwise, whatever the folder's -TEST suffix says (owner 2026-10-14 13:25Z: all owner screenshots
    train). Every folder with a VNDB id is eligible for its role; the owner never had a development-sample VN."""
    m = FOLDER_VN.search(name)
    vn = NAMED.get(name) or (f"v{m.group(1)}" if m else None)
    was_test = name.endswith("-TEST")
    if vn is None:
        return {"vn": None, "role": "test" if was_test else "training", "eligible": False,
                "why": "no VNDB id in the folder name", "was_test_folder": was_test}
    from common import test_set_vns
    test = {v: r for v, r in (reserved or {}).items() if not r.startswith("dev-sample")} | test_set_vns()
    r = test.get(vn)
    if r is not None and owner_allowed(vn, r):
        return {"vn": vn, "role": "training", "eligible": True, "why": None, "was_test_folder": was_test,
                "owner_decision": f"{OWNER_DECISIONS[vn]['by']} (reserved_vns: {r})"}
    if r is not None:
        return {"vn": vn, "role": "test", "eligible": True, "why": f"test reservation: {r}",
                "was_test_folder": was_test}
    return {"vn": vn, "role": "training", "eligible": True, "why": None, "was_test_folder": was_test}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- Adrenaline 2x detection

def down_up_residual(g, method):
    """mean |g - up2(down2(g))| over the mean neighbour difference: a frame that is a 2x upscale by
    `method` comes back almost unchanged (nearest: about 0.18 after VNDB's Q90), a native frame does not
    (native Vita frames 0.70 and above)."""
    import cv2
    g = np.asarray(g, np.float32)
    H, W = g.shape
    rec = cv2.resize(cv2.resize(g, (W // 2, H // 2), interpolation=cv2.INTER_AREA), (W, H), interpolation=method)
    ref = (np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()) / 2
    return float(np.abs(g - rec).mean() / (ref + 1e-6))


def hf_ratio(g, axis):
    """Power above 0.3 cycles per pixel over the power at 0.05-0.25 along an axis: a smooth 2x upscale
    (bilinear, bicubic) holds almost nothing above 0.25 (p99 0.031 along x after Q90 for PSP frames
    upscaled bilinearly), a native frame much more (native Vita frames: p1 0.031, p10 0.045)."""
    g = np.asarray(g, np.float32)
    g = g - g.mean()
    P = np.abs(np.fft.rfft(g, axis=axis)) ** 2
    e = P.sum(axis=1 - axis)
    f = np.fft.rfftfreq(g.shape[axis])
    return float(e[f > 0.3].sum() / (e[(f > 0.05) & (f < 0.25)].sum() + 1e-6))


UP2_NEAREST, UP2_HF_X, UP2_HF_Y = 0.40, 0.025, 0.04


def upscale_2x(g):
    """(source, method, stats) of a 960x544 frame: ("480x272", "nearest" | "smooth") when it is a PSP
    frame scaled 2x to the Vita's screen (an Adrenaline capture), else ("960x544", None). Thresholds from
    training-side native Vita frames against training-side PSP frames upscaled 2x and re-encoded at Q90."""
    import cv2
    stats = {"nearest_residual": round(down_up_residual(g, cv2.INTER_NEAREST), 4),
             "hf_x": round(hf_ratio(g, 1), 4), "hf_y": round(hf_ratio(g, 0), 4)}
    if stats["nearest_residual"] < UP2_NEAREST:
        return "480x272", "nearest", stats
    if stats["hf_x"] < UP2_HF_X and stats["hf_y"] < UP2_HF_Y:
        return "480x272", "smooth", stats
    return "960x544", None, stats



def frame_row(path, folder, info=None):
    """One inventory row (no pixels): hashes, size and source resolution of a screenshot."""
    import cv2
    from PIL import Image

    from devnative import paths as P
    info = info or folder_info(folder)
    rgb = P.read_rgb(path)
    row = {"folder": folder, "file": Path(path).name, "role": info["role"], "vn": info["vn"],
           "eligible": info["eligible"], "why_not": info["why"], "owner_decision": info.get("owner_decision"),
           "bytes": Path(path).stat().st_size, "sha256": file_sha256(path)}
    if rgb is None:
        return dict(row, error="undecodable")
    with Image.open(path) as im:
        pil = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
        row["format"] = im.format
    H, W = rgb.shape[:2]
    row.update(dims=[W, H], pixel_sha256=P.pixel_sha256(rgb), rgb_sha256=hashlib.sha256(pil.tobytes()).hexdigest())
    if (W, H) == (960, 544):
        src, method, stats = upscale_2x(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        row.update(source=src, upscale=method, upscale_stats=stats,
                   console="psp-on-vita" if src == "480x272" else "vita")
    elif (W, H) == (480, 272):
        row.update(source="480x272", upscale=None, console="psp")
    else:
        row.update(source=f"{W}x{H}", upscale=None, console="other")
    return row


def cmd_inventory(args):
    root, out = Path(args.root), Path(args.out)
    reserved = SH.reserved_vns(SH.load_dump(Path(args.dump))) if args.dump else {}
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        info = folder_info(d.name, reserved)
        for p in sorted(x for x in d.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXT):
            rows.append(frame_row(p, d.name, info))
    first = {}
    for r in rows:
        k = r.get("pixel_sha256")
        if k in first:
            r["duplicate_of"] = first[k]
        elif k:
            first[k] = f"{r['folder']}/{r['file']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = {}
    for folder in sorted({r["folder"] for r in rows}):
        rs = [r for r in rows if r["folder"] == folder]
        summary[folder] = {"role": rs[0]["role"], "vn": rs[0]["vn"], "eligible": rs[0]["eligible"],
                           "why_not": rs[0]["why_not"], "owner_decision": rs[0].get("owner_decision"),
                           "files": len(rs),
                           "unique": sum("duplicate_of" not in r and "error" not in r for r in rs),
                           "dims": dict(collections.Counter("x".join(map(str, r.get("dims", []))) for r in rs)),
                           "console": dict(collections.Counter(r.get("console") for r in rs)),
                           "upscale": dict(collections.Counter(str(r.get("upscale")) for r in rs)),
                           "formats": dict(collections.Counter(r.get("format") for r in rs))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


def cmd_prep(args):
    out = Path(args.out)
    rows = [json.loads(x) for x in (out / "inventory.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    keep = [r for r in rows if r["folder"] == args.folder and "duplicate_of" not in r and "error" not in r]
    if keep and not keep[0]["eligible"]:
        sys.exit(f"{args.folder}: refused ({keep[0]['why_not']})")
    reads = []
    for i, r in enumerate(keep):
        kind = "psp" if r["dims"] == [480, 272] else "vita"
        stem = f"{r['vn']}_owner{i:04d}"
        reads.append({"shot": stem, "vn": r["vn"], "kind": kind, "url": f"owner:{r['folder']}/{r['file']}",
                      "path": str(Path(args.root) / r["folder"] / r["file"]), "release": None,
                      "session": f"owner/{r['folder']}", "subset": r["role"], "flags": [],
                      "source_kind": r.get("console"), "dims": r["dims"], "sha256": r["sha256"]})
    p = out / args.folder / "reads" / "shots.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in reads), encoding="utf-8")
    print(f"{len(reads)} unique frames of {args.folder} -> {p}", flush=True)


def cmd_pool_prep(args):
    """read.py's input for the owner training pool: every unique, eligible frame of the training folders.
    A frame whose pixels another folder already holds is dropped (first folder in name order wins). --skip-pool
    leaves out the frames an earlier owner pool already read (its manifests/frames.jsonl, by decoded-pixel
    sha256), so a later pool reads only new frames; --prefix names its shots <vn>_<prefix><n> (a VN's n-th unique
    frame across its folders, in name order) apart from the first pool's <vn>_owner<n>."""
    out, pool = Path(args.out), Path(args.pool)
    rows = [json.loads(x) for x in (out / "inventory.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    want = set(args.folder or ())
    skip = {}
    for sp in getattr(args, "skip_pool", None) or ():
        for x in (Path(sp) / "manifests" / "frames.jsonl").read_text(encoding="utf-8").splitlines():
            if x.strip():
                f = json.loads(x)
                skip.setdefault(f["pixel_sha256"], f"{Path(sp).name}:{f['shot']}")
    prefix = getattr(args, "prefix", None) or "owner"
    reads, frames, refused, skipped = [], [], collections.Counter(), collections.Counter()
    index = collections.Counter()
    for folder in sorted({r["folder"] for r in rows}):
        rs = [r for r in rows if r["folder"] == folder]
        if (want and folder not in want) or rs[0]["role"] != "training":
            continue
        if not rs[0]["eligible"]:
            refused[f"{folder}: {rs[0]['why_not']}"] += len(rs)
            continue
        keep = [r for r in rs if "duplicate_of" not in r and "error" not in r]
        for i, r in enumerate(keep):
            n = i if prefix == "owner" else index[r["vn"]]
            index[r["vn"]] += 1
            kind = "psp" if r["source"] == "480x272" and r["dims"] == [480, 272] else (
                "vita" if r["dims"] == [960, 544] and r["source"] == "960x544" else None)
            if kind is None:
                refused[f"{folder}: not a native frame ({r['source']}, {r['dims']})"] += 1
                continue
            if r["pixel_sha256"] in skip:
                skipped[folder] += 1
                continue
            shot = f"{r['vn']}_{prefix}{n:04d}"
            reads.append({"shot": shot, "vn": r["vn"], "kind": kind, "url": f"owner:{folder}/{r['file']}",
                          "path": str(Path(args.root) / folder / r["file"]), "release": None,
                          "session": f"owner/{folder}", "subset": "native", "flags": [], "source_kind": kind,
                          "dims": r["dims"]})
            frames.append({"shot": shot, "vn": r["vn"], "kind": kind, "folder": folder, "file": r["file"],
                           "sha256": r["sha256"], "pixel_sha256": r["pixel_sha256"], "rgb_sha256": r["rgb_sha256"],
                           "dims": r["dims"], "source": r["source"], "upscale": r.get("upscale"),
                           "owner_decision": r.get("owner_decision")})
    for name, rows_ in (("reads/shots.jsonl", reads), ("manifests/frames.jsonl", frames)):
        p = pool / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_), encoding="utf-8")
    print(json.dumps({"frames": len(reads), "by_vn": dict(collections.Counter(r["vn"] for r in reads)),
                      "by_kind": dict(collections.Counter(r["kind"] for r in reads)), "refused": dict(refused),
                      "skipped_read_by_an_earlier_pool": dict(skipped)}, indent=1), flush=True)


def cmd_publish(args):
    """The committed, aggregate-only record of an owner pool (realplus/manifests/owner/stats.json): per-folder
    eligibility and sizes, leak-check pair counts, candidate / label / build counts per VN and the npz sha256.
    No file name, hash, box, text or pixel of a private frame."""
    out, pool = Path(args.out), Path(args.pool)

    def jl(p):
        p = Path(p)
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
    summary = json.loads((out / "summary.json").read_text())
    folders = {k: {f: v.get(f) for f in ("role", "vn", "eligible", "why_not", "owner_decision", "files", "unique",
                                          "dims", "console", "upscale")} for k, v in summary.items()}
    leak = {}
    for lp in sorted((pool / "logs").glob("leak_*.log")):
        last = [x for x in lp.read_text().splitlines() if x.startswith("{")]
        if last:
            d = json.loads(last[-1])
            leak[lp.stem[5:]] = {k: d.get(k) for k in ("tests", "pool", "min_corr", "pairs", "same-vn", "cross-vn")}
    vn = lambda lid: lid.split("_")[0]  # noqa: E731
    labels = jl(pool / "labels.jsonl")
    by_vn = collections.defaultdict(lambda: collections.Counter())
    for r in labels:
        by_vn[vn(r["id"])][f"{r['status']}:{r['how'] if r['status'] == 'kept' else r['why']}"] += 1
    build = json.loads((pool / "manifests" / f"build_stats_{args.tag}.json").read_text())
    man = jl(pool / "manifests" / f"lines_{args.tag}.jsonl")
    lines_vn = collections.defaultdict(lambda: collections.Counter())
    for m in man:
        lines_vn[m["vn"]]["lines"] += 1
        for c in m["crops"]:
            lines_vn[m["vn"]][c] += 1
    doc = {"kind": "owner-pool-stats", "tag": args.tag, "private_dir": "shared/private/owner-games/_build/owner-pool",
           "codex": "none: private frames; Claude + MeikiOCR / PP-OCRv6 agreement only",
           "folders": folders, "leak": leak,
           "frames_read": len(jl(pool / "reads" / "reads.jsonl")),
           "candidates_by_vn": dict(collections.Counter(vn(r["id"]) for r in labels)),
           "labels_by_vn": {k: dict(v) for k, v in sorted(by_vn.items())},
           "build": build, "lines_by_vn": {k: dict(v) for k, v in sorted(lines_vn.items())}}
    cp = pool / "review" / f"check_report_{args.check}.json"
    if cp.exists():
        d = json.loads(cp.read_text())
        doc["precision_check"] = {k: v for k, v in d.items() if k not in ("disagreements", "batches")}
    dst = ROOT / "realplus" / "manifests" / "owner"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "stats.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"published {dst / 'stats.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(PRIVATE))
    ap.add_argument("--out", default=os.environ.get("OWNER_BUILD") or str(PRIVATE / "_build"))
    ap.add_argument("--dump", default=str(SH.ART / "native-frames" / "vndb" / "dump" / "db"),
                    help="the VNDB dump's db/ directory, for the training folders' eligibility check")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory")
    p = sub.add_parser("prep")
    p.add_argument("--folder", required=True)
    p = sub.add_parser("pool-prep")
    p.add_argument("--pool", required=True, help="the owner pool directory (under shared/private)")
    p.add_argument("--folder", action="append", help="only these training folders (default: all)")
    p.add_argument("--prefix", default="owner", help="shot name prefix: <vn>_<prefix><n> (a later pool: e.g. p2f)")
    p.add_argument("--skip-pool", action="append", help="an earlier owner pool directory whose frames are skipped")
    p = sub.add_parser("publish")
    p.add_argument("--pool", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--check", default="pre", help="the build tag the precision sample was drawn from")
    a = ap.parse_args()
    {"inventory": cmd_inventory, "prep": cmd_prep, "pool-prep": cmd_pool_prep, "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    main()

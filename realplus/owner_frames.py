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

Folders (shared/private/owner-games/): `memories-off-TEST` (test only: a fresh native test set, never trained
on, never sent to Codex) and `chaos-head-noah-v22505` (training-eligible; private frames, so never sent to
Codex either). Everything stays under shared/private/: frames, readings, crops, labels, manifests.
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


def folder_info(name, reserved=None):
    """{vn, role, eligible, why} of an owner folder. A test folder's VN must be an owner test title
    (shots.OWNER_TEST_SERIES); a training folder's VN must be on the training side today (not in
    reserved_vns), else its frames are refused (eligible False)."""
    m = FOLDER_VN.search(name)
    vn = NAMED.get(name) or (f"v{m.group(1)}" if m else None)
    role = "test" if name.endswith("-TEST") else "training"
    if vn is None:
        return {"vn": None, "role": role, "eligible": False, "why": "no VNDB id in the folder name"}
    if role == "test":
        ok = vn in SH.OWNER_TEST_SERIES
        return {"vn": vn, "role": role, "eligible": ok, "why": None if ok else "not an owner test title"}
    r = (reserved or {}).get(vn)
    return {"vn": vn, "role": role, "eligible": r is None, "why": None if r is None else f"reserved: {r}"}
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
           "eligible": info["eligible"], "why_not": info["why"],
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
                           "why_not": rs[0]["why_not"], "files": len(rs),
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
    a = ap.parse_args()
    {"inventory": cmd_inventory, "prep": cmd_prep}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The pinned synth3 font pool: data/manifests/synth3_fonts.json.

  synth3_fonts.py build  --research A/synth3/fonts/fonts_manifest.json --out data/manifests/synth3_fonts.json
  synth3_fonts.py verify [--manifest data/manifests/synth3_fonts.json]

The pool is synth2's (google/fonts OFL families in $HERD/shared/data/fonts, Fusion Pixel in
synth2's fonts_extra) plus the families the synth3 font research downloaded and licence-checked
into $HERD/shared/artifacts/synth3/fonts (its fonts_manifest.json: source URL, revision, licence
text and per-file sha256). Fonts are never committed; this manifest pins every file by size and
sha256, and records its licence class:

  open   free to use and redistribute, commercial use included (OFL-1.1, Apache-2.0, IPA, M+,
         public-domain and author "free" licences; see licence_id and the saved licence file)
  nc     free for personal / non-commercial use only (excluded by synth3.py --profile open)

pixel_em is the design pixel size of a bitmap-style font: its embedded bitmap strike when it has
one (FreeType then draws those exact bitmaps), otherwise the grid size its name or notes give.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ROOTS = {"pool": HERD / "shared" / "data" / "fonts",
         "pool_extra": HERD / "shared" / "artifacts" / "synth2" / "fonts_extra",
         "synth3": HERD / "shared" / "artifacts" / "synth3" / "fonts"}
OUT = ROOT / "data" / "manifests" / "synth3_fonts.json"
POOL_CATEGORY = {   # synth2's render.CATEGORY
    "gothic": ["notosansjp", "mplus1p", "mplus2", "sawarabigothic", "zenkakugothicnew", "zenkakugothicantique",
               "bizudgothic", "bizudpgothic", "murecho", "ibmplexsansjp"],
    "rounded": ["mplusrounded1c", "zenmarugothic", "kiwimaru"],
    "mincho": ["notoserifjp", "sawarabimincho", "zenoldmincho", "zenantique", "shipporimincho", "shipporiantique",
               "bizudmincho", "hinamincho", "kaiseiopti", "kaiseitokumin", "yujisyuku"],
    "pen": ["kleeone", "yuseimagic", "yomogi", "zenkurenaido"],
    "pixel": ["dotgothic16", "fusionpixel10", "fusionpixel12"],
    "display": ["rocknrollone", "mochiypopone", "hachimarupop"],
}
PIXEL_NAME = [(r"ark-pixel-10px", 10), (r"ark-pixel-12px", 12), (r"Galmuri9", 10), (r"Galmuri11", 12),
              (r"Galmuri14", 15), (r"PixelMplus10", 10), (r"PixelMplus12", 12), (r"unifont", 16), (r"zpix", 12),
              (r"x\d+y12px", 12), (r"x\d+y16px", 16), (r"fusion-pixel-10px", 10), (r"fusion-pixel-12px", 12),
              (r"DotGothic16", 16), (r"misaki", 8)]
NC_FAMILIES = {"jfdot_noncommercial", "zpix"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_em(name, strikes):
    if strikes:
        return int(strikes[0])
    for pat, em in PIXEL_NAME:
        if re.search(pat, name, re.I):
            return em
    return None


def os2_weight(path):
    from fontTools.ttLib import TTFont
    try:
        return int(TTFont(str(path), lazy=True)["OS/2"].usWeightClass)
    except (KeyError, AttributeError):
        return 400


def pool_families():
    out = []
    for cat, fams in POOL_CATEGORY.items():
        for fam in fams:
            root = "pool_extra" if fam.startswith("fusionpixel") else "pool"
            d = ROOTS[root] / fam
            files = sorted(p for p in d.glob("*.[to]tf"))
            if not files:
                continue
            lic = d / "OFL.txt"
            out.append({
                "family": fam, "category": cat, "root": root, "licence_id": "OFL-1.1", "licence_class": "open",
                "licence_file": f"{fam}/OFL.txt", "licence_sha256": sha256(lic) if lic.exists() else None,
                "source": ("google/fonts ofl/" + fam + " (data/fetch_assets.py fonts, 2026-10-08)"
                           if root == "pool" else "synth2 data/manifests/synth2_fonts.json"),
                "files": [{"file": f"{fam}/{p.name}", "bytes": p.stat().st_size, "sha256": sha256(p),
                           "category": cat, "weight": os2_weight(p),
                           "pixel_em": pixel_em(p.name, []) if cat == "pixel" else None} for p in files]})
    return out


def research_families(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for fam in doc["families"]:
        files = []
        for f in fam["files"]:
            cat = f.get("category") or fam["category"]
            em = pixel_em(f["file"], f.get("bitmap_ppem") or []) if cat == "pixel" else None
            files.append({"file": f"{fam['family']}/{f['file']}", "bytes": f["bytes"], "sha256": f["sha256"],
                          "category": cat, "weight": f.get("os2_weight") or 400, "pixel_em": em,
                          "strikes": f.get("bitmap_ppem") or []})
        out.append({"family": fam["family"], "category": fam["category"], "root": "synth3",
                    "licence_id": fam["licence_id"],
                    "licence_class": "nc" if fam["family"] in NC_FAMILIES or not fam.get("commercial_ok") else "open",
                    "licence_file": fam.get("licence_file"), "licence_sha256": fam.get("licence_file_sha256"),
                    "licence_url": fam.get("licence_url"), "source": fam.get("source_url"),
                    "source_revision": fam.get("source_revision"), "resembles": fam.get("style_notes"),
                    "files": files})
    return out


def cmd_build(args):
    fams = pool_families() + research_families(args.research)
    seen = {}
    for fam in fams:
        for f in fam["files"]:
            if f["sha256"] in seen:
                raise SystemExit(f"duplicate font bytes: {f['file']} and {seen[f['sha256']]}")
            seen[f["sha256"]] = f["file"]
    doc = {"kind": "synth3-fonts", "schema": 1,
           "roots": {"pool": "$HERD/shared/data/fonts", "pool_extra": "$HERD/shared/artifacts/synth2/fonts_extra",
                     "synth3": "$HERD/shared/artifacts/synth3/fonts"},
           "research_manifest_sha256": sha256(args.research),
           "families": sorted(fams, key=lambda f: (f["root"], f["family"])),
           "counts": {"families": len(fams), "files": sum(len(f["files"]) for f in fams),
                      "bytes": sum(x["bytes"] for f in fams for x in f["files"]),
                      "nc_families": sorted(f["family"] for f in fams if f["licence_class"] == "nc")}}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{doc['counts']} -> {args.out}")


def cmd_verify(args):
    doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    bad = 0
    for fam in doc["families"]:
        for f in fam["files"]:
            p = ROOTS[fam["root"]] / f["file"]
            if not p.exists() or p.stat().st_size != f["bytes"] or sha256(p) != f["sha256"]:
                print(f"MISMATCH {fam['root']}:{f['file']}")
                bad += 1
    print(f"verified {sum(len(f['files']) for f in doc['families'])} files, {bad} mismatches")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--research", required=True)
    b.add_argument("--out", default=str(OUT))
    v = sub.add_parser("verify")
    v.add_argument("--manifest", default=str(OUT))
    args = ap.parse_args()
    {"build": cmd_build, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The owner's desktop captures (docs/desktop-data.md "Owner captures"): PC frames at their own size with the game's
hooked text and line boxes, in the hooked bucket's format (data/import_hooked.py; `<platform>/<TITLE_ID> <title>/`
with pages.tar.zst, pages.jsonl.zst, lines.jsonl.zst and metadata.json, platform `pc`, `resolution` the capture's
size), into the same pools and page pool as the console captures and into native-size detector pages.

  owner.py import --root BUCKET --work DIR [--vndb-map M] [--procs 5]   extract, split, lines, pages (import_hooked)
  owner.py meiki  --work DIR [--workers 5]       MeikiOCR's boxes (threshold 0.2) for the text the hook does not cover
  owner.py shards --work DIR --out SHARDS [--allow-missing-meiki]       textspot6-pages shards at the frame's size
  owner.py manifest --work DIR --shards SHARDS --out data/manifests/desktop_data/owner_desktop_vN.json

The import is data/import_hooked.py's, unchanged, with one addition for frames that are neither Vita nor PSP: the
page pool's dialogue regions (eval/finder_tighten.dialogue_units) take eval's 12 px margin scaled to the frame,
max(12, round(12 H / 544)), where the console platforms keep their 12 / 6 px. The splits are the bucket's (a title
marked test stays out; hval / guard blocks as for the console titles). The shards are synth/textspot6_pages.py's,
unchanged: a sample's kind is the page's platform ("pc"), so the trainers' PSP 2x step never applies, and its
positives are the verified lines (refined to ink) at the frame's own size. Nothing is re-rendered: these are real
desktop captures. Everything stays local (copyrighted game frames); the manifests hold ids, counts and sha256.
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, ROOT / "data", ROOT / "synth", ROOT / "eval", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONSOLE = {"vita", "psp"}


def margin_for(H):
    return max(12, int(round(12 * H / 544)))


@contextlib.contextmanager
def desktop_margins():
    """finder_tighten.dialogue_units with a frame-scaled margin for a platform it does not know (a desktop frame)."""
    import finder_tighten as FT
    orig = FT.dialogue_units

    def du(shot, kind, W, H, *a, **k):
        if kind in FT.MARGIN:
            return orig(shot, kind, W, H, *a, **k)
        FT.MARGIN[kind] = margin_for(H)
        try:
            return orig(shot, kind, W, H, *a, **k)
        finally:
            FT.MARGIN.pop(kind, None)
    FT.dialogue_units = du
    try:
        yield
    finally:
        FT.dialogue_units = orig


def cmd_import(args):
    import import_hooked as IH
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    root = str(args.root)
    IH.main(["extract", "--root", root, "--out", str(work / "pages"), "--procs", str(args.procs)])
    IH.main(["split", "--root", root, "--out", str(work / "splits.json"), "--min-pages", str(args.min_pages)])
    lines = ["lines", "--root", root, "--pages", str(work / "pages"), "--splits", str(work / "splits.json"),
             "--out", str(work / "pools"), "--procs", str(args.procs), "--pool-tag", args.pool_tag]
    if args.vndb_map:
        lines += ["--vndb-map", args.vndb_map]
    IH.main(lines)
    with desktop_margins():
        IH.main(["pages", "--root", root, "--pages", str(work / "pages"), "--splits", str(work / "splits.json"),
                 "--pools", str(work / "pools"), "--out", str(work / "pagepool"), "--procs", "1"])
    print(f"owner import: {work}", flush=True)


def cmd_meiki(args):
    import textspot6_pages as TP
    work = Path(args.work)
    TP.main(["meiki", "--pool", str(work / "pagepool"), "--pages", str(work / "pages"), "--out", str(work / "meiki"),
             "--workers", str(args.workers)])


def cmd_shards(args):
    import textspot6_pages as TP
    work = Path(args.work)
    os.environ["HOOKED"] = str(args.root) if args.root else os.environ.get("HOOKED", "")
    cmd = ["build", "--pool", str(work / "pagepool"), "--pages", str(work / "pages"), "--out", str(args.out),
           "--workers", str(args.workers)]
    if (work / "meiki").exists():
        cmd += ["--meiki", str(work / "meiki")]
    if args.allow_missing_meiki:
        cmd += ["--allow-missing-meiki"]
    TP.main(cmd)


def cmd_manifest(args):
    import json

    import textspot6_pages as TP
    work = Path(args.work)
    TP.main(["manifest", "--shards", str(args.shards), "--pool", str(work / "pagepool"), "--out", str(args.out)])
    doc = json.loads(Path(args.out).read_text(encoding="utf-8"))
    kinds = {}
    for s in doc["shards"]:
        js = Path(args.shards) / s["dir"] / f"{s['name']}.json"
        for smp in json.loads(js.read_text(encoding="utf-8"))["samples"]:
            kinds[smp["kind"]] = kinds.get(smp["kind"], 0) + 1
    doc["desktop"] = {"what": "the owner's desktop captures at their own size (docs/desktop-data.md)",
                      "kinds": kinds, "generator": "desktopdata/owner.py"}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(doc['shards'])} shards, kinds {kinds}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import")
    p.add_argument("--root", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--vndb-map", default="")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--min-pages", type=int, default=200)
    p.add_argument("--pool-tag", default="owner-desktop-v1")
    p = sub.add_parser("meiki")
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=5)
    p = sub.add_parser("shards")
    p.add_argument("--work", required=True)
    p.add_argument("--root", help="the bucket root (VNDB ids from its metadata.json files)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--allow-missing-meiki", action="store_true")
    p = sub.add_parser("manifest")
    p.add_argument("--work", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"import": cmd_import, "meiki": cmd_meiki, "shards": cmd_shards, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

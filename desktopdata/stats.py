#!/usr/bin/env python3
"""Per-resolution counts and text-height distributions of the desktop data against the Vita / PSP pools
(docs/desktop-data.md "Counts").

  stats.py run --set NAME=MANIFEST:ROOTS[:SPLIT[:KIND]] ... [--pctest eval/manifests/desktop-pc-test-v1.jsonl]
               --out data/manifests/desktop_data/stats_v1.json
  stats.py tables --stats data/manifests/desktop_data/stats_v1.json [--doc docs/desktop-data.md] [--check]

A set is a shard manifest (textspot6-pages, desktop-frames or #97's textspot shards) and ROOTS, comma-separated
directories: a shard is read from the one named like its "dir" (FramePool's rule), else from <root>/<dir>; SPLIT
restricts a textspot6-pages manifest to one split (train), KIND to the samples of one kind (vita, psp).

Per set: frames, frames per resolution class and per exact size, positive lines per class, and the height (y1 - y0)
of every horizontal positive line of a text class (dialogue, name, text, notdlg, choice, ui) at the frame's own size
and at vocr-desktop's default input policy fit:960 (the longer side down to 960 px when longer, never up:
tools/vocr_desktop.scale_for), as percentiles and a histogram. The PC test set's dialogue lines (their ink box
heights) are added for reference.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktopdata import shards as SH  # noqa: E402

BINS = [0, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 100000]
CLASSES = ("<=640", "641-1023", "1024-1279", "1280-1599", "1600-1919", "1920-2559", "2560-3839", ">=3840")
TEXT = {"dialogue", "name", "text", "notdlg", "choice", "ui"}
PCT = (5, 25, 50, 75, 95)


def res_class(w):
    for lo, hi, name in ((0, 640, CLASSES[0]), (641, 1023, CLASSES[1]), (1024, 1279, CLASSES[2]),
                         (1280, 1599, CLASSES[3]), (1600, 1919, CLASSES[4]), (1920, 2559, CLASSES[5]),
                         (2560, 3839, CLASSES[6])):
        if lo <= w <= hi:
            return name
    return CLASSES[7]


def fit960(w, h):
    return min(1.0, 960.0 / max(w, h))


def dist(values):
    if not values:
        return {"n": 0}
    a = np.asarray(values, np.float64)
    hist, _ = np.histogram(a, bins=BINS)
    return {"n": int(a.size), "mean": round(float(a.mean()), 2),
            **{f"p{p}": round(float(np.percentile(a, p)), 1) for p in PCT},
            "hist": {f"{BINS[i]}-{BINS[i + 1] if BINS[i + 1] < 100000 else 'inf'}": int(v) for i, v in enumerate(hist)}}


def shard_path(roots, d, name):
    """A shard's json: the root whose directory name is the shard's "dir" (FramePool's rule), else <root>/<dir>."""
    for r in roots:
        if Path(r).name == d:
            return Path(r) / f"{name}.json"
    for r in roots:
        if (Path(r) / d / f"{name}.json").exists():
            return Path(r) / d / f"{name}.json"
    return None


def set_stats(manifest, root, split=None, kind=None):
    doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
    roots = [x for x in str(root).split(",") if x]
    skipped = 0
    frames, by_class, by_size, lines_cls = 0, collections.Counter(), collections.Counter(), collections.Counter()
    h_native, h_fit, h_native_dlg, h_fit_dlg = [], [], [], []
    kinds = collections.Counter()
    mix = collections.Counter()
    for s in doc["shards"]:
        if split and s.get("split") != split:
            continue
        js = shard_path(roots, s["dir"], s["name"])
        if js is None:
            skipped += 1
            continue
        if SH.sha256_file(js) != s["json_sha256"]:
            raise SystemExit(f"{js}: sha256 is not {manifest}'s")
        for smp in json.loads(js.read_text(encoding="utf-8"))["samples"]:
            if split and smp.get("split") != split:
                continue
            if kind and smp.get("kind") != kind:
                continue
            frames += 1
            w, h = smp["w"], smp["h"]
            by_class[res_class(w)] += 1
            by_size[f"{w}x{h}"] += 1
            kinds[smp.get("kind")] += 1
            f = fit960(w, h)
            meta = smp.get("meta") or {}
            cap = meta.get("capture")
            if cap:
                mix["captured"] += 1
                mix[f"setup:{cap['setup']}"] += 1
                mix[f"resampler:{cap['resampler']}"] += 1
                mix["dpi"] += bool(cap.get("dpi"))
                mix["downscaled"] += bool(cap.get("down"))
                mix["jpeg"] += bool(cap.get("jpeg"))
                if cap["setup"] == "fullscreen" and cap.get("name"):
                    mix[f"screen:{cap['name'].split('/')[0]}"] += 1
            if meta.get("src_kind"):
                mix[f"src_kind:{meta['src_kind']}"] += 1
            for e in meta.get("elements") or []:
                parts = e.split(":")
                mix[f"element:{parts[0]}"] += 1
                if parts[0] == "dialogue" and len(parts) >= 5:
                    mix[f"layout:{parts[1]}"] += 1
                    mix[f"name:{parts[2]}"] += 1
                    mix[f"weight:{parts[3]}"] += 1
                    mix[f"category:{parts[4]}"] += 1
            for t in meta.get("tags") or []:
                mix[f"tag:{t.rstrip('0123456789')}"] += 1
            for k in ("other_line_ignore", "det05_ignore", "det02_ignore", "furigana_ignore"):
                mix[f"ignore:{k}"] += (meta.get("counts") or {}).get(k, 0)
            if meta.get("guard_ignore"):
                mix["ignore:guard"] += meta["guard_ignore"]
            mix["ignore_areas"] += len(smp["ignore"])
            if not smp["lines"]:
                mix["text_free"] += 1
            for ln in smp["lines"]:
                lines_cls[ln[5]] += 1
                if int(ln[4]) or ln[5] not in TEXT:
                    continue
                hh = ln[3] - ln[1]
                h_native.append(hh)
                h_fit.append(hh * f)
                if ln[5] in ("dialogue", "text"):
                    h_native_dlg.append(hh)
                    h_fit_dlg.append(hh * f)
    return {"manifest": str(manifest), "manifest_sha256": SH.sha256_file(manifest), "split": split, "kind": kind,
            "shards_not_found": skipped, "frames": frames, "kinds": dict(kinds),
            "by_class": {c: by_class.get(c, 0) for c in CLASSES},
            "by_size": dict(by_size.most_common(25)), "sizes": len(by_size), "lines_by_class": dict(lines_cls),
            "height_native": dist(h_native), "height_fit960": dist(h_fit),
            "height_native_dialogue": dist(h_native_dlg), "height_fit960_dialogue": dist(h_fit_dlg),
            "mix": dict(sorted(mix.items()))}


def pctest_stats(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if not r["name"]]
    shots = {}
    for r in rows:
        shots[r["shot"]] = r["dims"]
    by_class = collections.Counter(res_class(w) for w, h in shots.values())
    hn = [r["box"][3] for r in rows]
    hf = [r["box"][3] * fit960(*r["dims"]) for r in rows]
    return {"manifest": str(path), "manifest_sha256": SH.sha256_file(path), "frames": len(shots),
            "lines": len(rows), "by_class": {c: by_class.get(c, 0) for c in CLASSES},
            "by_size": dict(collections.Counter(f"{w}x{h}" for w, h in shots.values()).most_common(25)),
            "height_native_dialogue": dist(hn), "height_fit960_dialogue": dist(hf)}


def cmd_run(args):
    out = {"kind": "desktop-data-stats", "created": SH.utc(), "bins": BINS, "classes": CLASSES, "sets": {}}
    for spec in args.set:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        man, root = parts[0], parts[1]
        split = parts[2] if len(parts) > 2 and parts[2] else None
        kind = parts[3] if len(parts) > 3 and parts[3] else None
        out["sets"][name] = set_stats(man, root, split, kind)
        print(name, out["sets"][name]["frames"], out["sets"][name]["height_native"].get("p50"), flush=True)
    if args.pctest and Path(args.pctest).exists():
        out["sets"]["pc-test"] = pctest_stats(args.pctest)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"-> {args.out}", flush=True)


# ----------------------------------------------------------------------------- doc tables

def fmt(n):
    return f"{n:,}"


def table_resolution(st):
    names = list(st["sets"])
    head = "| resolution class (width) | " + " | ".join(names) + " |"
    sep = "|---|" + "---:|" * len(names)
    rows = [head, sep]
    for c in CLASSES:
        rows.append(f"| {c} | " + " | ".join(fmt(st["sets"][n]["by_class"].get(c, 0)) for n in names) + " |")
    rows.append("| frames | " + " | ".join(fmt(st["sets"][n]["frames"]) for n in names) + " |")
    rows.append("| distinct sizes | " + " | ".join(str(st["sets"][n].get("sizes", len(st["sets"][n]["by_size"])))
                                                   for n in names) + " |")
    return "\n".join(rows)


def table_heights(st):
    rows = ["| set | lines | native p5 / p25 / p50 / p75 / p95 (px) | at fit:960 p5 / p25 / p50 / p75 / p95 (px) |",
            "|---|---:|---|---|"]
    for n, s in st["sets"].items():
        a, b = s["height_native_dialogue"], s["height_fit960_dialogue"]
        if not a.get("n"):
            continue
        q = lambda d: " / ".join(f"{d[f'p{p}']:g}" for p in PCT)  # noqa: E731
        rows.append(f"| {n} | {fmt(a['n'])} | {q(a)} | {q(b)} |")
    return "\n".join(rows)


def table_hist(st):
    names = [n for n, s in st["sets"].items() if s["height_native_dialogue"].get("n")]
    keys = list(st["sets"][names[0]]["height_native_dialogue"]["hist"])
    rows = ["| dialogue line height at native size (px) | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names)]
    for k in keys:
        cells = []
        for n in names:
            d = st["sets"][n]["height_native_dialogue"]
            cells.append(f"{100.0 * d['hist'][k] / d['n']:.1f}%")
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_mix(st):
    sets = [n for n in ("1a hooked desktop", "1c synthetic desktop") if n in st["sets"]]
    keys = ["setup:fullscreen", "setup:window_multiple", "setup:window_free", "setup:stretch", "screen:16:9",
            "screen:16:10", "screen:21:9", "screen:4:3", "resampler:bilinear", "resampler:bicubic",
            "resampler:lanczos", "resampler:nearest", "resampler:sharp_bilinear", "resampler:bilinear_sharpen", "dpi",
            "downscaled", "jpeg"]
    label = {"setup:window_multiple": "setup: emulator window (multiple of the frame)",
             "setup:window_free": "setup: user-sized window", "dpi": "DPI stretch (125-175%)",
             "downscaled": "capture downscaled", "jpeg": "capture saved as JPEG"}
    rows = ["| capture (frames) | " + " | ".join(sets) + " |", "|---|" + "---:|" * len(sets)]
    for k in keys:
        head, _, tail = k.partition(":")
        name = label.get(k, f"fullscreen on {tail}" if head == "screen" else f"{head}: {tail}")
        rows.append(f"| {name} | " + " | ".join(fmt(st["sets"][n]["mix"].get(k, 0)) for n in sets) + " |")
    rows.append("| captured (of all frames) | " + " | ".join(
        f"{fmt(st['sets'][n]['mix'].get('captured', 0))} of {fmt(st['sets'][n]['frames'])}" for n in sets) + " |")
    return "\n".join(rows)


TABLES = {"resolution": table_resolution, "heights": table_heights, "hist": table_hist, "mix": table_mix}


def render_doc(text, st):
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- generated:{name} -->\n)(.*?)(<!-- /generated:{name} -->)", re.S)
        text = pat.sub(lambda m: m.group(1) + fn(st) + "\n" + m.group(3), text)
    return text


def cmd_tables(args):
    st = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    if not args.doc:
        for name, fn in TABLES.items():
            print(f"## {name}\n{fn(st)}\n")
        return
    doc = Path(args.doc)
    old = doc.read_text(encoding="utf-8")
    new = render_doc(old, st)
    if args.check:
        if new != old:
            raise SystemExit(f"{doc}: generated tables differ from {args.stats}")
        print("tables up to date")
        return
    doc.write_text(new, encoding="utf-8")
    print(f"{doc}: tables written")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--set", action="append", required=True)
    p.add_argument("--pctest")
    p.add_argument("--out", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--stats", required=True)
    p.add_argument("--doc")
    p.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    {"run": cmd_run, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()

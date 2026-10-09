#!/usr/bin/env python3
"""Score a recognizer (and optionally a line finder) on real VN dialogue.

  eval/run.py --model MODEL [--set vita-test|psp-test|synth-val|all] [--finder F]
              [--polarity native|console] [--out results.json] [--errors N] [--data DIR] [--pad PX]

MODEL is a train/model.py checkpoint (.pt), an ONNX export, a named PaddleOCR
baseline (ppocr-v5-mobile, ppocr-v5-mobile-jis, ppocr-v5-server, ppocr-v6-small),
an ncnn model dir (ncnn:DIR), manga-ocr, or any external program (cmd:COMMAND,
e.g. the C runtime's host CLI). See eval/backends.py and eval/README.md.

Recognizer mode (default) reads every hand-checked line crop (the line's box plus
--pad px: 4 on Vita, 2 on PSP) and reports CER, exact-line accuracy and per-style and per-game
breakdowns. With --finder (overlay, console, gradient, py:MOD:FUNC, cmd:COMMAND) it runs
the finder on each dialogue region instead, recognizes what it finds, and reports
line recall/precision and the whole-pipeline CER. Scores use eval/metrics.py norm().

--polarity console hands every crop over the way the console does (docs/linefind.md,
polarity contract): a crop whose text synth/common.line_dark calls dark is inverted
(255 - grey) before recognition. The default, native, passes crops as they are.
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import backends  # noqa: E402
import finders  # noqa: E402
import metrics  # noqa: E402
import sets  # noqa: E402
from fetch import data_dir  # noqa: E402


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def available_sets(names, data):
    out = []
    for n in names:
        if n == "synth-val":
            path = Path(os.environ.get("VOCR_SYNTH_VAL", sets.SYNTH_VAL))
            if path.exists():
                out.append(n)
            else:
                print(f"skipping synth-val: {path} not found", file=sys.stderr)
        elif (sets.MANIFESTS / f"{n}.jsonl").exists():
            out.append(n)
        else:
            print(f"skipping {n}: no manifest yet", file=sys.stderr)
    return out


def console_polarity(crops):
    """The console's polarity contract: crops whose text common.line_dark calls dark become
    255 - grey (in all three channels, so a grey recognizer sees exactly that); the others
    pass unchanged. Returns (crops, inverted flags)."""
    from common import line_polarity, to_gray
    out, flags = [], []
    for c in crops:
        g = to_gray(c) if c.ndim == 3 else c
        dark = bool(line_polarity(g))
        out.append(np.repeat((255 - g)[..., None], 3, axis=2) if dark else c)
        flags.append(dark)
    return out, flags


def prepare(crops, polarity):
    if polarity == "console":
        return console_polarity(crops)
    return crops, [False] * len(crops)


def fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{100 * v:.2f}%" if pct else str(v)


def recognizer_mode(rec, name, data, args):
    if name == "synth-val":
        path = os.environ.get("VOCR_SYNTH_VAL", sets.SYNTH_VAL)
        lines = sets.load_synth_val(path, limit=sets.SYNTH_VAL_LINES if args.limit is None else args.limit)
    else:
        lines, _ = sets.load(name, data, pad=args.pad, limit=args.limit)
    crops, inverted = prepare([ln.crop_rgb for ln in lines], getattr(args, "polarity", "native"))
    preds, sec = backends.timed_read(rec, crops)
    rows = [{"id": ln.id, "vn": ln.vn, "styles": ln.styles, "text": ln.text, "pred": p}
            for ln, p in zip(lines, preds)]
    for r, inv in zip(rows, inverted):
        if inv:
            r["inverted"] = True
    res = metrics.score_lines(rows)
    res["inverted_lines"] = sum(inverted)
    res["host_ms_per_line"] = round(1000 * sec, 2)
    res["oov_chars"] = metrics.oov_chars([ln.text for ln in lines], rec.info.get("vocab_set"))
    res["lines"] = rows
    return res


def finder_mode(rec, find, name, data, args):
    cache = sets.ImageCache()
    tot = {"regions": 0, "rejected": 0, "gt_lines": 0, "found": 0, "dets": 0, "dets_ok": 0,
           "errors": 0, "chars": 0, "concat_errors": 0, "concat_chars": 0}
    out_regions = []
    for path, (rx, ry, rw, rh), rows in sets.regions(name, data):
        img = cache.get(path)
        region = img[ry:ry + rh, rx:rx + rw]
        gts = [dict(r, box_r=(r["box"][0] - rx, r["box"][1] - ry, r["box"][2], r["box"][3])) for r in rows]
        dets = find(region)
        tot["regions"] += 1
        scored = [g for g in gts if not g.get("uncertain")]
        n_gt = sum(len(metrics.norm(g["text"])) for g in scored)
        tot["gt_lines"] += len(scored)
        tot["chars"] += n_gt
        rec_reg = {"region": [rx, ry, rw, rh], "image": Path(path).name, "rejected": dets is None,
                   "gt": [g["id"] for g in gts], "dets": []}
        if dets is None:
            tot["rejected"] += 1
            tot["errors"] += n_gt
            if not any(g.get("uncertain") for g in gts):
                tot["concat_errors"] += n_gt
                tot["concat_chars"] += n_gt
            out_regions.append(rec_reg)
            continue
        dets = [tuple(int(v) for v in d) for d in dets]
        crops = [region[y:y + h, x:x + w] for (x, y, w, h) in dets]
        crops, _ = prepare(crops, getattr(args, "polarity", "native"))
        preds = rec.read(crops) if crops else []
        pairs, unmatched = finders.match(dets, gts, args.match)
        # A detection on an uncertain line is neither a hit nor a false alarm (sets.py).
        on_uncertain = sum(1 for _, j in pairs if gts[j].get("uncertain"))
        tot["dets"] += len(dets) - on_uncertain
        tot["dets_ok"] += len(pairs) - on_uncertain
        matched_g = {j for _, j in pairs}
        for i, j in pairs:
            g = gts[j]
            if not g.get("uncertain"):
                tot["found"] += 1
                tot["errors"] += metrics.lev(metrics.norm(preds[i]), metrics.norm(g["text"]))
        for j, g in enumerate(gts):
            if j not in matched_g and not g.get("uncertain"):
                tot["errors"] += len(metrics.norm(g["text"]))
        for i in unmatched:
            tot["errors"] += len(metrics.norm(preds[i]))
        if not any(g.get("uncertain") for g in gts):
            order = sorted(range(len(dets)), key=lambda i: (dets[i][1], dets[i][0]))
            cat_p = "".join(metrics.norm(preds[i]) for i in order)
            cat_g = "".join(metrics.norm(g["text"]) for g in sorted(gts, key=lambda g: (g["box"][1], g["box"][0])))
            tot["concat_errors"] += metrics.lev(cat_p, cat_g)
            tot["concat_chars"] += len(cat_g)
        rec_reg["dets"] = [{"box": list(d), "pred": p, "gt": gts[dict(pairs).get(i, -1)]["id"]
                            if i in dict(pairs) else None} for i, (d, p) in enumerate(zip(dets, preds))]
        out_regions.append(rec_reg)
    summary = {
        "regions": tot["regions"], "regions_rejected": tot["rejected"],
        "gt_lines": tot["gt_lines"], "line_recall": round(tot["found"] / max(1, tot["gt_lines"]), 5),
        "detections": tot["dets"], "line_precision": round(tot["dets_ok"] / max(1, tot["dets"]), 5),
        "pipeline_cer": round(tot["errors"] / max(1, tot["chars"]), 5),
        "concat_cer": round(tot["concat_errors"] / max(1, tot["concat_chars"]), 5),
        "concat_regions_chars": tot["concat_chars"],
    }
    return {"finder": summary, "regions": out_regions}


def print_recognizer(name, res, errors):
    a = res["all"]
    inv = f" | {res['inverted_lines']} inverted" if res.get("inverted_lines") else ""
    print(f"\n{name}: {a['lines']} lines, {a['chars']} chars | CER {fmt(a['cer'])} | exact {fmt(a['exact'])}"
          f" | game-macro CER {fmt(res['game_macro_cer'])} | {res['host_ms_per_line']} ms/line (host){inv}")
    if res["styles"]:
        print(f"  {'style':16s} {'lines':>6s} {'chars':>6s} {'CER':>8s} {'exact':>8s}")
        for tag, s in res["styles"].items():
            print(f"  {tag:16s} {s['lines']:6d} {s['chars']:6d} {fmt(s['cer']):>8s} {fmt(s['exact']):>8s}")
    if res["oov_chars"]:
        oov = "".join(list(res["oov_chars"])[:40])
        print(f"  ground-truth chars the model cannot emit: {sum(res['oov_chars'].values())} ({oov})")
    if errors:
        worst = sorted((r for r in res["lines"] if r["errors"]), key=lambda r: -r["errors"])[:errors]
        for r in worst:
            print(f"  [{r['id']}] {r['errors']}/{r['n']}\n     GT   {r['text']}\n     PRED {r['pred']}")


def print_finder(name, res):
    f = res["finder"]
    print(f"\n{name} (finder): {f['regions']} regions ({f['regions_rejected']} rejected) | "
          f"line recall {fmt(f['line_recall'])} | precision {fmt(f['line_precision'])} | "
          f"pipeline CER {fmt(f['pipeline_cer'])} | concat CER {fmt(f['concat_cer'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint, ONNX, preset, ncnn:DIR, manga-ocr or cmd:COMMAND")
    ap.add_argument("--set", default="all", help="vita-test, psp-test, synth-val, comma list, or all")
    ap.add_argument("--finder", default=None, help="overlay, console, gradient, py:MOD:FUNC or cmd:COMMAND")
    ap.add_argument("--data", default=None, help="screenshot dir (default $VOCR_EVAL_DATA or data/eval)")
    ap.add_argument("--pad", type=int, default=None,
                    help="px added around each line box (default 4 on vita-test, 2 on psp-test)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N lines of each set (synth-val defaults to 3000; 0 = all)")
    ap.add_argument("--match", type=float, default=0.5, help="finder mode: min vertical overlap to match")
    ap.add_argument("--polarity", default="native", choices=["native", "console"],
                    help="console: invert dark-text crops (synth/common.line_dark) before recognition, "
                         "as the console does (docs/linefind.md)")
    ap.add_argument("--cmd-format", default="ppm", choices=["ppm", "pgm", "png"])
    ap.add_argument("--errors", type=int, default=0, help="print the N worst lines per set")
    ap.add_argument("--out", type=Path, default=None, help="write full results (per line) as JSON")
    args = ap.parse_args(argv)

    data = data_dir(args.data)
    names = ["vita-test", "psp-test", "synth-val"] if args.set == "all" else args.set.split(",")
    for n in names:
        if n not in sets.SETS:
            ap.error(f"unknown set {n!r}")
    names = available_sets(names, data)
    if args.finder:
        names = [n for n in names if n != "synth-val"]
    t0 = time.time()
    rec = backends.make_recognizer(args.model, args.cmd_format)
    info = {k: v for k, v in rec.info.items() if k != "vocab_set"}
    print(f"model {rec.name} {json.dumps(info)}" + (" | polarity console" if args.polarity == "console" else ""))
    find = finders.make_finder(args.finder) if args.finder else None
    results = {"model": args.model, "model_name": rec.name, "model_info": info, "finder": args.finder,
               "pad": args.pad if args.pad is not None else sets.PAD, "polarity": args.polarity, "sets": {},
               "env": {"git": git_sha(), "python": platform.python_version(), "host": platform.node(),
                       "threads": backends.threads(),
                       "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}}
    for n in names:
        if find:
            res = finder_mode(rec, find, n, data, args)
            print_finder(n, res)
        else:
            res = recognizer_mode(rec, n, data, args)
            print_recognizer(n, res, args.errors)
        results["sets"][n] = res
    rec.close()
    results["seconds"] = round(time.time() - t0, 1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return results


if __name__ == "__main__":
    main()

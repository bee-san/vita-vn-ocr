#!/usr/bin/env python3
"""Compare eval/run.py results with the owner's quality bar (RULES.md "Owner goal").

  bar.py RESULT.json [RESULT2.json ...] [--md]

The bar is the JIS-trimmed int8 PP-OCRv5 mobile (eval's `ncnn:.../ppocr-v5-mobile-jis-int8`
baseline). Every result is re-scored from its per-line predictions against the committed
manifests (eval/manifests/*.jsonl): eval/metrics.norm, Levenshtein over total ground-truth
characters, overall and per style tag, so a JSON written before a tag fix scores with
today's tags. Lines no longer in a manifest, or now uncertain, are left out. A result that
lacks some of the manifest's lines is scored on the rest, with a warning on stderr: its
numbers are then not comparable with the bar's.

RULES.md asks for an overall CER *below* the bar's and a CER *at or below* it in every style,
so `*` marks the overall row only when it is strictly below the bar (a tie is marked `=`) and
a style row when it is at or below.

The default bar files are eval's own baseline runs on the shared host: vita-test from
results/ncnn-jis-int8.json, psp-test from results/rev01-psp/jis-int8.json (the run on the
fixed psp-test). The owner quoted psp-test 5.58%, the figure before eval PR #5's fixes;
re-scored on today's manifest the same model gets 5.34%, which is the bar used here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
RES = HERD / "shared/artifacts/eval/results"
BAR = {"vita-test": RES / "ncnn-jis-int8.json", "psp-test": RES / "rev01-psp/jis-int8.json"}
SETS = ("vita-test", "psp-test")


def manifest(name):
    rows = [json.loads(x) for x in (ROOT / "eval/manifests" / f"{name}.jsonl").read_text("utf-8").splitlines()
            if x.strip()]
    return {r["id"]: r for r in rows if not r.get("uncertain")}


def score(preds, man):
    """preds: {line id: predicted text}. Returns {"all": (cer, exact, lines), tag: ...}."""
    acc = {}
    for lid, row in man.items():
        if lid not in preds:
            continue
        ref, hyp = norm(row["text"]), norm(preds[lid])
        e = lev(hyp, ref)
        for tag in ["all"] + list(row.get("styles") or []):
            a = acc.setdefault(tag, [0, 0, 0, 0])
            a[0] += e
            a[1] += len(ref)
            a[2] += e == 0
            a[3] += 1
    return {k: (v[0] / max(v[1], 1), v[2] / max(v[3], 1), v[3]) for k, v in acc.items()}


def mark(cer, bar_cer, overall):
    """'*' if a CER meets the bar (overall: strictly below it; a style: at or below it), '='
    for an overall tie, '' otherwise."""
    if overall:
        return "*" if cer < bar_cer - 1e-12 else "=" if abs(cer - bar_cer) <= 1e-12 else ""
    return "*" if cer <= bar_cer + 1e-12 else ""


def missing(preds, man):
    """Manifest line ids a result has no prediction for."""
    return [lid for lid in man if lid not in preds]


def preds_of(path, name):
    d = json.loads(Path(path).read_text("utf-8"))
    s = d["sets"].get(name)
    if not s:
        return None
    return {r["id"]: r["pred"] for r in s["lines"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--md", action="store_true", help="Markdown tables")
    a = ap.parse_args()
    for name in SETS:
        man = manifest(name)
        bar = score(preds_of(BAR[name], name), man)
        cols = []
        for r in a.results:
            p = preds_of(r, name)
            if p is not None:
                gone = missing(p, man)
                if gone:
                    print(f"warning: {r} {name}: no prediction for {len(gone)} of {len(man)} manifest lines "
                          f"(e.g. {gone[0]}); scored on the rest, so not comparable with the bar",
                          file=sys.stderr)
                cols.append((Path(r).stem, score(p, man)))
        if not cols:
            continue
        tags = ["all"] + sorted(t for t in bar if t != "all")
        print(f"\n{name} (CER / exact; * = meets the bar: overall below it, a style at or below it; "
              f"= overall tie)")
        head = ["style", "lines", "bar (JIS int8)"] + [c for c, _ in cols]
        if a.md:
            print("| " + " | ".join(head) + " |")
            print("|" + "---|" * len(head))
        else:
            print("  ".join(f"{h:>18s}" for h in head))
        for t in tags:
            b = bar.get(t)
            cells = [t, str(b[2]) if b else "", f"{100 * b[0]:.2f}% / {100 * b[1]:.1f}%" if b else ""]
            for _, sc in cols:
                v = sc.get(t)
                if v is None:
                    cells.append("")
                    continue
                m = mark(v[0], b[0], t == "all") if b is not None else ""
                cells.append(f"{100 * v[0]:.2f}% / {100 * v[1]:.1f}%" + (f" {m}" if m else ""))
            if a.md:
                print("| " + " | ".join(cells) + " |")
            else:
                print("  ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""stride16_floor.py: the CTC frame budget of every development row (eval/stride16_protocol.json "floor").

  stride16_floor.py build --out FLOOR.json [--cohorts DIR] [--dev-dir DIR]
  stride16_floor.py summary FLOOR.json

Model-independent: no recognizer reads anything. For every row the selection scores it records the frames the C
runtime gives it at a frame step of 8, 12 and 16 input columns, and NR, the frames CTC needs for its reference:
characters plus one blank between adjacent repeated characters (eval/metrics.norm on the reference; the same
floor train/train.py ctc_ok applies). A row with T_s < NR cannot be read exactly at stride s by any model; the
protocol's tight stratum is T_s < 1.25 NR.

  fixed lines (tv: teacher_val_dev.npz, dn: devnative's four development files; eval/synth3_ablation.py's rows,
      labelled lines in file order): the stored crop (h, w) is what the runtime resizes, so q = round(w * 32 / h)
      (half to even, runtime/vocr.c vocr_resized_width) clamped to [8, 1024 // s * s], and T_s = ceil(q / s);
  whole dialogue (tv_dlg, dn_dlg; eval/finder_tighten.py's units): each case's C finder lines in both finder modes
      (GRAD with flags 0, and the refinement with VOCR_REFINE_DEFAULT; finder_tighten.lines_of, C == its Python
      twin on every case) give T_s = the sum over the unit's lines of ceil(q_line / s), and NR = the sum over the
      region's reference lines (the textbox guard scores all of them) of characters + repeats. A unit is a region
      read as a whole, so its ratio is a proxy: a region whose lines all run short of frames is tight, one starved
      line beside roomy ones is diluted.

The output holds counts, ids, frame numbers and hashes only (no text).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import norm  # noqa: E402

HERD = Path("/local/home/skerraut/herd-vita-ocr")
DEV_DIR = HERD / "shared/artifacts/owner-drive-ingest-2/reclaim/pools"
COHORTS = HERD / "shared/artifacts/vocr5-train/cohorts"
TV_FILE = "teacher_val_dev.npz"
DN_FILES = ("devnative_tv_native_ink_dev.npz", "devnative_tv_native_grad_dev.npz", "devnative_strict_ink_dev.npz",
            "devnative_strict_grad_dev.npz")
STRIDES = (8, 12, 16)
HEIGHT = 32
TIGHT = 1.25


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resized_width(w, h, stride, height=HEIGHT):
    """runtime/vocr.c vocr_resized_width for a blob of this stride: round(w * H / h), half to even, clamped to
    [8, the blob's max width] (train/export.py default_max_width: 1024, 1020 at stride 12)."""
    num = w * height
    q = (2 * num + h) // (2 * h)
    if (2 * num + h) % (2 * h) == 0 and q & 1:
        q -= 1
    return max(8, min(1024 // stride * stride, q))


def frames(w, h):
    return {str(s): math.ceil(resized_width(w, h, s) / s) for s in STRIDES}


def need(text):
    """CTC's frames for a normalised reference: characters + adjacent repeats."""
    t = norm(text)
    return len(t) + sum(a == b for a, b in zip(t, t[1:]))


def fixed_rows(path):
    """{line id: {T: {s: frames}, nr, platform}} for the labelled lines of a development npz (synth3_ablation's
    rows: an empty label is skipped)."""
    z = np.load(path)
    shapes = z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    out = {}
    for i, (lab, m) in enumerate(zip(labels, meta)):
        if not lab:
            continue
        h, w = (int(v) for v in shapes[i])
        out[m["id"]] = {"T": frames(w, h), "nr": need(lab), "platform": m.get("target") or m.get("kind")}
    return out


def wd_rows(cohorts):
    """{'<cohort>|<mode>|<unit>|<variant>': {T, nr, platform, lines}} for both finder modes."""
    import tempfile

    import finder_tighten as FT

    import lines_refine as LR
    ident = FT.build(Path(tempfile.mkdtemp(prefix="stride16-floor-")) / "build")
    out, meta = {}, {"finder": {k: ident[k] for k in ("driver_sha256", "library_sha256", "compiler")}}
    for name in ("tv_dlg", "dn_dlg"):
        cases = FT.load_cohort(cohorts, name)
        imgs = [c["image"] for c in cases]
        meta[name] = {"sha256": sha256(Path(cohorts) / f"{name}.pkl"), "cases": len(cases)}
        for mode, flags in (("grad", 0), ("refine", LR.parse_flags("default"))):
            for c, (boxes, _dark, _org, _rows, _found) in zip(cases, FT.lines_of(ident["driver"], imgs, flags)):
                key = f"{name}|{mode}|{c['unit']}|{c['variant']}"
                t = {str(s): sum(math.ceil(resized_width(w, h, s) / s) for (_x, _y, w, h) in boxes)
                     for s in STRIDES}
                out[key] = {"T": t, "nr": sum(need(ln["text"]) for ln in c["lines"]), "platform": c["kind"],
                            "shot": c["shot"], "vn": c["vn"], "lines": len(boxes)}
    return out, meta


def cmd_build(args):
    dev = Path(args.dev_dir)
    doc = {"kind": "stride16-floor", "schema": 1, "what": __doc__.split("\n\n")[2].replace("\n", " "),
           "tight_ratio": TIGHT, "strides": list(STRIDES),
           "code_sha256": {f: sha256(ROOT / f) for f in ("eval/stride16_floor.py", "eval/finder_tighten.py",
                                                          "runtime/vocr_lines.c", "runtime/vocr_lines_refine.c")},
           "files": {}, "tv": {}, "dn": {}, "wd": {}}
    doc["files"][TV_FILE] = sha256(dev / TV_FILE)
    doc["tv"] = fixed_rows(dev / TV_FILE)
    for f in DN_FILES:
        doc["files"][f] = sha256(dev / f)
        doc["dn"][f] = fixed_rows(dev / f)
    doc["wd"], doc["wd_meta"] = wd_rows(args.cohorts)
    Path(args.out).write_text(json.dumps(doc) + "\n", encoding="utf-8")
    print(json.dumps(summarize(doc), indent=1))


def _strata(rows, s):
    r = np.array([v["T"][str(s)] for v in rows], float)
    nr = np.array([v["nr"] for v in rows], float)
    ok = nr > 0
    ratio = np.where(ok, r / np.maximum(nr, 1), np.inf)
    return {"rows": int(len(rows)), "below_floor": int((ratio < 1.0).sum()), "tight": int((ratio < TIGHT).sum()),
            "tight_share": round(float((ratio < TIGHT).mean()), 5) if len(rows) else 0.0}


def summarize(doc):
    out = {}
    fixed = list(doc["tv"].values()) + [v for f in doc["dn"].values() for v in f.values()]
    for s in STRIDES:
        out[f"fixed_s{s}"] = _strata(fixed, s)
        for plat in sorted({v["platform"] for v in fixed}):
            out[f"fixed_s{s}_{plat}"] = _strata([v for v in fixed if v["platform"] == plat], s)
        out[f"wd_s{s}"] = _strata(list(doc["wd"].values()), s)
    return out


def cmd_summary(args):
    print(json.dumps(summarize(json.loads(Path(args.floor).read_text(encoding="utf-8"))), indent=1))


def cmd_protected(args):
    """Report only (protocol locked_report.also), run once inside the locked report: the protected fixed lines'
    frame budget (vita-test, psp-test: eval/sets.py's crops, the line box plus the set's pad, as the report reads
    them) and each report arm's CER on the tight (T_s < 1.25 NR) and floor (T_s < NR) strata at strides 12 and 16,
    console polarity, dialogue (no 'name' style) and all text. Counts only."""
    import sets
    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    fl = rep["shared"]["fixed_line"]["console"]
    rows = {}
    for s in ("vita-test", "psp-test"):
        lines, _ = sets.load(s, args.data)
        for ln in lines:
            h, w = ln.crop_rgb.shape[:2]
            rows[(s, ln.id)] = {"T": frames(w, h), "nr": need(ln.text)}
    doc = {"kind": "stride16-floor-protected", "schema": 1, "tight_ratio": TIGHT,
           "lines": {s: sum(k[0] == s for k in rows) for s in ("vita-test", "psp-test")}, "strata": {}, "arms": {}}
    for st in (8, 12, 16):
        for which, lim in (("tight", TIGHT), ("floor", 1.0)):
            for s in ("vita-test", "psp-test"):
                doc["strata"][f"s{st}_{which}_{s}"] = sum(v["T"][str(st)] < lim * v["nr"]
                                                          for k, v in rows.items() if k[0] == s)
    for arm, a in fl.items():
        rec = {}
        for st in (12, 16):
            for which, lim in (("tight", TIGHT), ("floor", 1.0)):
                for part in ("dialogue", "all"):
                    e = c = n = 0
                    for s in ("vita-test", "psp-test"):
                        for x in a["lines"][s]:
                            r = rows[(s, x["id"])]
                            if r["T"][str(st)] >= lim * r["nr"] or (part == "dialogue" and "name" in x["styles"]):
                                continue
                            e, c, n = e + x["errors"], c + x["n"], n + 1
                    rec[f"s{st}_{which}_{part}"] = {"lines": n, "chars": c, "errors": e,
                                                    "cer": round(100 * e / c, 3) if c else None}
        doc["arms"][arm] = rec
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"strata": doc["strata"], "arms": {k: v["s16_tight_dialogue"] for k, v in doc["arms"].items()}},
                     indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--cohorts", default=str(COHORTS))
    b.add_argument("--dev-dir", default=str(DEV_DIR))
    s = sub.add_parser("summary")
    s.add_argument("floor")
    pr = sub.add_parser("protected")
    pr.add_argument("--report", required=True, help="the locked report's report.json (eval/vocr3_report.py render)")
    pr.add_argument("--data", required=True, help="eval's screenshot directory")
    pr.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"build": cmd_build, "summary": cmd_summary, "protected": cmd_protected}[args.cmd](args)


if __name__ == "__main__":
    main()

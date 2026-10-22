#!/usr/bin/env python3
"""Development check of the edge continuation (runtime/vocr_lines_widen.c, issue #109, docs/counters.md).

    kanum_widen.py outer --out DIR        the cohorts' regions again from their frames, each with the 16 px of the
                                          frame on its left and right (private: crops of copyrighted frames)
    kanum_widen.py score --blob S=VOCR5_S_w8.vocr [--blob N=...] [--outer DIR] --out widen.json      (heavy)

Whole-dialogue cohorts tv_dlg and dn_dlg (VOCR5's, $HERD/shared/artifacts/vocr5-train/cohorts): every region goes
through the C GRAD finder (flags 0) and the C refinement (VOCR_REFINE_DEFAULT), each line read by the C runtime in
W8I with its dark flag as VOCR_INVERT and scored with the textbox guard's transcript_score, as VOCR5's
development score reads them (eval/finder_tighten.py), in two arms:
  base   the finder's lines as they are;
  widen  every line continued past the region's left or right edge by vocr_line_widen() (the twin, checked
         against the C driver on every strip), into the pixels around the region.
Two sets of regions:
  standard  the cohorts' own regions (the labelled lines plus 12 px on Vita, 6 on PSP, at base, +4 and +8 px),
            with the frame's 16 px on each side as the pixels widen may continue into (`outer`: the regions are
            rebuilt from their frames by finder_tighten's own builders, and every rebuilt region must equal the
            cohort's pixel for pixel);
  tight     each base region cut to a region drawn against its text: its left edge at the lines' first ink
            column plus 0, 2, 4 and 6 px (into the text), its right edge where the region's is; the margin the
            base region holds around it (12 / 6 px) is the pixels widen may continue into.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
COHORTS = HERD / "shared" / "artifacts" / "vocr5-train" / "cohorts"
WD = ("tv_dlg", "dn_dlg")
CUTS = (0, 2, 4, 6)
MARGIN = 16                       # px of the frame kept on each side of a region (outer)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def core_of(case):
    """The labelled lines' union (the unit's id) in the region's coordinates."""
    x0, y0, x1, y1 = (int(v) for v in case["unit"].split("@")[1].split(","))
    rx, ry = case["region"][0], case["region"][1]
    return x0 - rx, y0 - ry, x1 - rx, y1 - ry


def build_widen(out):
    exe = Path(out) / "vocr_lines_widen_test"
    subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(exe),
                    "runtime/vocr_lines_widen.c", "runtime/vocr_lines_widen_test.c"], check=True, cwd=ROOT)
    return exe


def c_widen(exe, jobs):
    """The C vocr_line_widen on [(strip, rx, rw, x, w)]: [(sides, x, w)]."""
    if not jobs:
        return []
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(struct.pack("<I", len(jobs)))
        for s, rx, rw, x, w in jobs:
            s = np.ascontiguousarray(s, np.uint8)
            f.write(struct.pack("<7I", s.shape[1], s.shape[0], 1, rx, rw, x, w) + s.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(exe), "--parity", path], check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    return [tuple(int(t) for t in line.split()) for line in out.splitlines()]


def widen_lines(exe, img, rx, rw, boxes):
    """Every box (region coordinates) continued into img's columns around [rx, rx + rw): boxes in img's
    coordinates, and how many were widened. The twin decides, the C driver must agree."""
    import lines_widen as LW
    jobs, out, n = [], [], 0
    for (x, y, w, h) in boxes:
        strip = img[y:y + h]
        nx, nw, sides = LW.widen(strip, rx, rw, x, w)
        jobs.append((strip, rx, rw, x, w))
        out.append((nx, y, nw, h))
        n += sides != 0
    c = c_widen(exe, jobs)
    twin = [(LW.widen(s, rx, rw, x, w)[2], *LW.widen(s, rx, rw, x, w)[:2]) for s, rx, rw, x, w in jobs]
    if c != twin:
        raise RuntimeError("vocr_line_widen: C and twin differ on a development strip")
    return out, n


def edge_lines(img, boxes):
    """Lines whose box reaches the region's left or right edge with ink in the edge's columns: the ones widen
    could change if the region had pixels around it."""
    import lines_widen as LW
    n = 0
    W = img.shape[1]
    for (x, y, w, h) in boxes:
        g = LW.grey(img[y:y + h])
        rng = g.max(axis=0) - g.min(axis=0)
        t = max(int(rng[x:x + w].max()) // 3, LW.MIN_INK)
        left = x == 0 and (rng[0] >= t or (W > 1 and rng[1] >= t))
        right = x + w == W and (rng[W - 1] >= t or (W > 1 and rng[W - 2] >= t))
        n += bool(left or right)
    return n


def cmd_outer(a):
    """The cohorts' cases rebuilt from their frames with MARGIN px of the frame on the left and right."""
    import pickle

    import finder_tighten as FT
    orig = FT.dialogue_units

    def with_outer(shot, kind, W, H, members, others, grey, cohort, vn):
        out = orig(shot, kind, W, H, members, others, grey, cohort, vn)
        for c in out:
            x0, y0, x1, y1 = c["region"]
            ox0, ox1 = max(0, x0 - MARGIN), min(W, x1 + MARGIN)
            c["outer"] = np.ascontiguousarray(grey[y0:y1, ox0:ox1])
            c["outer_x"] = x0 - ox0
        return out
    FT.dialogue_units = with_outer
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    for name, build in (("tv_dlg", FT.build_tv_dlg), ("dn_dlg", FT.build_dn_dlg)):
        cases = FT.load_cohort(COHORTS, name)
        built = {(c["unit"], c["variant"]): c for c in build()}
        out, missing = [], 0
        for c in cases:
            b = built.get((c["unit"], c["variant"]))
            if b is None or b["image"].shape != c["image"].shape or not np.array_equal(b["image"], c["image"]):
                missing += 1
                out.append(None)
                continue
            out.append({"outer": b["outer"], "outer_x": b["outer_x"]})
        with open(d / f"{name}_outer.pkl", "wb") as f:
            pickle.dump(out, f, protocol=4)
        print(f"{name}: {len(cases)} cases, {len(cases) - missing} rebuilt pixel for pixel, {missing} not", flush=True)


def cmd_score(a):
    import finder_tighten as FT
    from pyvocr import Vocr, load_lib

    import lines_refine as LR
    from general_vn_textbox_guard import transcript_score
    out = Path(a.out)
    work = out.parent / (out.stem + "_build")
    work.mkdir(parents=True, exist_ok=True)
    ident = FT.build(work, "gcc")
    exe = build_widen(work)
    flags = LR.parse_flags("default")
    cohorts = {name: FT.load_cohort(COHORTS, name) for name in WD}
    outer = {}
    if a.outer:
        import pickle
        for name in WD:
            with open(Path(a.outer) / f"{name}_outer.pkl", "rb") as f:
                outer[name] = pickle.load(f)
    tv_shots = {c["shot"] for c in cohorts["tv_dlg"]}
    # the regions: standard (as they are) and tight (cut)
    regions = []          # (set, cohort, case index, cut, inner image, rx, rw, outer image)
    for name, cases in cohorts.items():
        for ci, c in enumerate(cases):
            if name == "dn_dlg" and c["shot"] in tv_shots:
                continue                                    # pooled as VOCR5 pools them
            img = c["image"]
            o = outer.get(name, [None] * len(cases))[ci]
            if o is not None:
                regions.append(("standard", name, ci, None, img, o["outer_x"], img.shape[1], o["outer"]))
            else:
                regions.append(("standard", name, ci, None, img, 0, img.shape[1], img))
            if c["variant"] != "base":
                continue
            cx0, _, cx1, _ = core_of(c)
            for cut in CUTS:
                left = cx0 + cut
                if left <= 0 or left >= img.shape[1] - 8:
                    continue
                regions.append(("tight", name, ci, cut, np.ascontiguousarray(img[:, left:]), left,
                                img.shape[1] - left, img))
    imgs = [r[4] for r in regions]
    found = {"grad": FT.lines_of(ident["driver"], imgs, 0), "refine": FT.lines_of(ident["driver"], imgs, flags)}
    lib = load_lib(ident["library"])
    report = {"kind": "kanum-fix-widen-dev", "cohorts": {n: sha256(COHORTS / f"{n}.pkl") for n in WD},
              "native": ident, "cuts": list(CUTS), "models": {}}
    for spec in a.blob:
        mname, _, path = spec.partition("=")
        rt = Vocr(path, mode="w8i", max_width=1024, lib=lib)
        units = {"base": [], "widen": []}
        counts = {"standard_edge_lines": 0, "standard_lines": 0, "standard_widened": 0, "tight_widened": 0,
                  "tight_lines": 0}
        for mode, per in found.items():
            for r, (boxes, dark, _, _, _) in zip(regions, per):
                kind, name, ci, cut, inner, rx, rw, outer = r
                case = cohorts[name][ci]
                base = [norm(rt.text(np.ascontiguousarray(inner[y:y + h, x:x + w]), invert=bool(d)))
                        for (x, y, w, h), d in zip(boxes, dark)]
                if kind == "standard":
                    counts["standard_edge_lines"] += edge_lines(inner, boxes)
                    counts["standard_lines"] += len(boxes)
                wb, n = widen_lines(exe, outer, rx, rw, boxes)
                counts[f"{kind}_widened"] += n
                if kind == "tight":
                    counts["tight_lines"] += len(boxes)
                wid = base if n == 0 else [
                    norm(rt.text(np.ascontiguousarray(outer[y:y + h, x:x + w]), invert=bool(d)))
                    for (x, y, w, h), d in zip(wb, dark)]
                for arm, preds in (("base", base), ("widen", wid)):
                    sc = transcript_score(case["lines"], preds)
                    units[arm].append({"unit": f"{kind}|{cut}|{name}|{mode}|{case['unit']}|{case['variant']}",
                                       "set": kind, "cut": cut, "mode": mode, "vn": case["vn"],
                                       "errors": sc["all"]["full"]["errors"], "chars": sc["all"]["full"]["chars"],
                                       "pred": " / ".join(preds)})
        res = {"blob": Path(path).name, "sha256": sha256(path), "counts": counts, "sets": {}}
        sets = [("standard", lambda u: u["set"] == "standard"), ("tight", lambda u: u["set"] == "tight")]
        sets += [(f"tight_cut{c}", lambda u, c=c: u["set"] == "tight" and u["cut"] == c) for c in CUTS]
        for label, keep in sets:
            b = [u for u in units["base"] if keep(u)]
            w = [u for u in units["widen"] if keep(u)]
            tb, tw = FT.totals(b), FT.totals(w)
            entry = {"base": [*tb, FT.pct(tb)], "widen": [*tw, FT.pct(tw)],
                     "better": sum(y["errors"] < x["errors"] for x, y in zip(b, w)),
                     "worse": sum(y["errors"] > x["errors"] for x, y in zip(b, w))}
            if tb != tw:
                entry["compare"] = FT.compare(w, b)
            res["sets"][label] = entry
        res["changed"] = [{"unit": x["unit"], "errors": [x["errors"], y["errors"]], "before": x["pred"],
                           "after": y["pred"]} for x, y in zip(units["base"], units["widen"])
                          if x["pred"] != y["pred"]][:400]
        report["models"][mname] = res
        print(mname, counts, {k: (v["base"][2], v["widen"][2], v["better"], v["worse"])
                              for k, v in res["sets"].items()}, flush=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--blob", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--outer")
    p = sub.add_parser("outer")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"score": cmd_score, "outer": cmd_outer}[a.cmd](a)


if __name__ == "__main__":
    main()

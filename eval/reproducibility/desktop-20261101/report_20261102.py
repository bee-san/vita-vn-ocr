"""The counts-only record behind docs/desktop.md's derived figures (audit-vocr-desktop N1, N2, N7, N12, N13, F3):
eval/desktop_report_20261102.json. It reads the stored result files, rows and boxes (private, on the shared box) and
writes counts and rates only: no text, box or frame.

  report_20261102.py --out eval/desktop_report_20261102.json --checks CHECKS.json --mida-info MIDA_INFO.json

- pc_test_width_classes: the PC test set's units by their frame's width (<= 800, 801-1279, 1280-1919, >= 1920 px):
  frames, VNs, units, characters, errors and CER per arm;
- dev_v2_decomposition: per arm against R, the extra errors in the arm's units with a detection fault (a cut or missed
  line, a merged pair, a junk box) and in the rest;
- c_crop_differences: on dev-v2 at long:640 (the release model), the frames whose ONNX and C crops differ, by the
  largest shift of a matched box edge, and the boxes only one path finds; plus CHECKS.json (the resampler and whole-page
  checks, the luma timings);
- mida: mida_ocr_l.onnx's size and MACs (eval/mida_det.py info) and its reads by mida-ocr-eval against R on the same
  frames (eval/desktop_record.py's rows);
- budget_t4: each arm's p50 at amendment 1's budget (the fastest of its detector-thread variants, per view) in the
  report's 4-thread grid.
"""
import argparse
import collections
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "tools", ROOT / "eval", ROOT / "synth", ROOT / "teacher"):
    sys.path.insert(0, str(_p))

HERD = os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")
A = f"{HERD}/shared/artifacts/vocr-desktop"
PV = f"{HERD}/shared/private/vocr-desktop"
MIDA = f"{HERD}/shared/artifacts/mida-ocr-eval/res"
POLS = ["fit960", "fit800", "fit640", "fit480", "long960", "long800", "long640"]
CORRECTIONS = [
    "eval/desktop_timing_20261101.json and eval/desktop_timing_decide_20261101.json: in the records of "
    "eval/desktop_time.py run, host.load_1min_at_start was read after the passes, so it equals load_1min_at_end "
    "(audit-vocr-desktop N6); the records are left as written, and desktop_time.py now reads it before the passes",
    "eval/desktop_protocol.json registered_in: the rule was in #169's body at 03:20:18Z, before any baseline was "
    "scored (the first at 03:33:26Z); the experiment log's protocol line (03:40Z) came after the dev-v2 and PC-sample "
    "baselines (audit-vocr-desktop F1); the file is left as registered",
    "the hooked test titles were read with SKIP_F3=1: the representative's 7 policies, which amendment 6 lists, "
    "were not read on them (audit-vocr-desktop F2)",
]


def width_class(w):
    return "<=800" if w <= 800 else "801-1279" if w < 1280 else "1280-1919" if w < 1920 else ">=1920"


def units(name):
    return json.load(open(f"{A}/res/{name}.json"))["units"]


def pc_width_classes(D):
    frames, _ = D.load_frames(f"{PV}/frames/pctest.pkl")
    width = {u["uid"]: fr["W"] for fr in frames for u in fr["units"]}
    out = {"frames": {}, "arms": {}}
    fr_by = collections.defaultdict(lambda: [0, set()])
    for fr in frames:
        c = fr_by[width_class(fr["W"])]
        c[0] += 1
        c[1].add(fr["cluster"])
    out["frames"] = {k: {"frames": v[0], "vns": len(v[1])} for k, v in sorted(fr_by.items())}
    arms = ["meiki4_L_fast", "meikiocr", "ts8_L_fast", "mida_L_fast"] + [f"F3_{p}_L_fast" for p in POLS] + \
        ["final3_long640_L_fast"]
    for n in arms:
        agg = collections.defaultdict(lambda: [0, 0, 0])
        for u in units(f"pctest_{n}"):
            a = agg[width_class(width[u["uid"]])]
            a[0] += 1
            a[1] += u["chars"]
            a[2] += u["errors"]
        out["arms"][n] = {k: {"units": a[0], "chars": a[1], "errors": a[2], "cer": round(100 * a[2] / a[1], 4)}
                          for k, a in sorted(agg.items())}
    return out


def fault(u):
    return any(s in ("cut", "missed") for s in u["status"]) or u.get("merged", 0) > 0 or u.get("junk", 0) > 0


def dev_decomposition():
    def rows(n):
        return {json.loads(ln)["uid"]: json.loads(ln) for ln in open(f"{PV}/rows/{n}.rows.jsonl")}
    ref = rows("dev_meiki4_L_fast")
    out = {}
    for n in ("dev_F3_long960_L_fast", "dev_F3_long640_L_fast", "dev_final3_long640_L_fast"):
        arm = rows(n)
        agg = collections.defaultdict(lambda: {"units": 0, "extra_errors": 0, "cut": 0, "missed": 0, "merged": 0,
                                               "junk": 0})
        for k, r in ref.items():
            a = arm[k]
            cls = "arm_fault" if fault(a) else ("whole_both" if not fault(r) else "whole_arm_only")
            g = agg[cls]
            g["units"] += 1
            g["extra_errors"] += a["errors"] - r["errors"]
            g["cut"] += sum(s == "cut" for s in a["status"])
            g["missed"] += sum(s == "missed" for s in a["status"])
            g["merged"] += a.get("merged", 0)
            g["junk"] += a.get("junk", 0)
        out[n.replace("dev_", "").replace("_L_fast", "")] = dict(sorted(agg.items()))
    return out


def c_crop_differences(D, VD):
    frames, _ = D.load_frames(f"{PV}/frames/dev.pkl")
    size = {f["key"]: (f["W"], f["H"]) for f in frames}
    a, _ = D.load_boxes(f"{PV}/boxes/dev_final3_long640.jsonl")
    c, _ = D.load_boxes(f"{PV}/boxes/dev_final3c_long640.jsonl")

    def crops(bs, W, H):
        return sorted(cb + (int(b[4]),) for b in bs
                      for cb in [VD.crop_of((b[0], b[1], b[2] - b[0], b[3] - b[1]), W, H)] if cb)
    shift = collections.Counter()
    only_onnx = only_c = one_path_frames = differ = 0
    for k in a:
        W, H = size[k]
        ca, cc = crops(a[k], W, H), crops(c[k], W, H)
        if ca == cc:
            continue
        differ += 1
        used, worst, oa = set(), 0, 0
        for x in ca:
            best = None
            for j, y in enumerate(cc):
                if j in used:
                    continue
                ix = min(x[0] + x[2], y[0] + y[2]) - max(x[0], y[0])
                iy = min(x[1] + x[3], y[1] + y[3]) - max(x[1], y[1])
                if ix <= 0 or iy <= 0:
                    continue
                iou = ix * iy / (x[2] * x[3] + y[2] * y[3] - ix * iy)
                if best is None or iou > best[0]:
                    best = (iou, j, y)
            if best is None:
                oa += 1
                continue
            used.add(best[1])
            y = best[2]
            worst = max(worst, abs(x[0] - y[0]), abs(x[1] - y[1]), abs(x[0] + x[2] - y[0] - y[2]),
                        abs(x[1] + x[3] - y[1] - y[3]))
        oc = len(cc) - len(used)
        only_onnx += oa
        only_c += oc
        one_path_frames += bool(oa or oc)
        shift[str(worst)] += 1
    return {"frames": len(a), "frames_differ": differ, "worst_edge_shift_px": dict(sorted(shift.items(),
            key=lambda t: int(t[0]))), "boxes_only_onnx": only_onnx, "boxes_only_c": only_c,
            "frames_with_a_box_one_path_finds": one_path_frames,
            "note": "crops matched by largest IoU; the shift is the largest edge move of a matched crop on the frame "
                    "(0: only the box count differs)"}


def mida_records(REC):
    out = {}
    pairs = {"dev": (f"{A}/res/dev_meiki4_L_fast.json", f"{MIDA}/dev_box960_L_fast.json"),
             "hval": (f"{A}/res/hval_meiki4_L_fast.json", f"{MIDA}/hval_box960_L_fast.json"),
             "dev960": (f"{MIDA}/dev960_meiki4_L_fast.json", f"{MIDA}/dev960_mida_native_L_fast.json"),
             "hval960": (f"{MIDA}/hval960_meiki4_L_fast.json", f"{MIDA}/hval960_mida_native_L_fast.json")}
    for s, (r, m) in pairs.items():
        with tempfile.TemporaryDirectory() as t:
            shutil.copy(r, f"{t}/{s}_meiki4_L_fast.json")
            shutil.copy(m, f"{t}/{s}_mida_L_fast.json")
            REC.main(["--res", t, "--set", s, "--ref", "meiki4_L_fast", "--names", "mida_L_fast",
                      "--out", f"{t}/rec.json", "--note", "mida-ocr-eval's reads (eval/mida_det.py), measurement only"])
            rec = json.load(open(f"{t}/rec.json"))
        rec["source_files"] = {"reference": r, "arm": m}
        out[s] = rec
    return out


def budget_t4():
    grid = json.load(open(f"{A}/timing/report_grid_t4.json"))["results"]
    base = {}
    for name in grid:
        b = name.split("|")[0].split("#d")[0]
        base.setdefault(b, []).append(name)
    out = {}
    for b, names in base.items():
        out[b] = {v: {"p50": min(grid[n][v]["p50"] for n in names if v in grid[n]),
                      "variant": min((grid[n][v]["p50"], n) for n in names if v in grid[n])[1]}
                  for v in ("vita", "psp", "pc")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", required=True)
    ap.add_argument("--mida-info", required=True)
    a = ap.parse_args(argv)
    import desktop_record as REC
    import vocr_desktop as VD

    import desktop as D
    checks = json.load(open(a.checks))
    info = json.load(open(a.mida_info))
    rec = {"kind": "vocr-desktop-report-record", "issue": 169,
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repo_head": D.git_head(),
           "note": "derived figures of docs/desktop.md (audit-vocr-desktop's N1, N2, N7, N12, N13 and F3), counts and "
                   "rates only, from the stored reads; no new read of a protected set",
           "pc_test_width_classes": pc_width_classes(D), "dev_v2_decomposition": dev_decomposition(),
           "c_crop_differences": c_crop_differences(D, VD),
           "c_checks": {k: checks[k] for k in ("resampler", "page")}, "luma_ms": checks["luma"],
           "checks_meta": {k: checks[k] for k in checks if k.endswith("sha256") or k in ("created", "repo_head")},
           "mida": {"model": {k: info[k] for k in ("file", "bytes", "sha256", "params")},
                    "macs": info["macs"], "reads": mida_records(REC)},
           "budget_t4": budget_t4(), "corrections": CORRECTIONS}
    Path(a.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pc_classes_R": rec["pc_test_width_classes"]["arms"]["meiki4_L_fast"],
                      "c_crop_differences": rec["c_crop_differences"], "budget_mida": rec["budget_t4"].get(
                          "mida@box:960x544@along=1.0@min_cells=2+L")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

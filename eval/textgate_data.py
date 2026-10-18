#!/usr/bin/env python3
"""textgate: training-side data of the text-presence gate (docs/textgate.md, eval/textgate_protocol.json).

  textgate_data.py cal --out DIR [--workers W]
      The calibration cohorts, from the calibration split's VNs only (calib_vn): never a reserved test or
      development-sample VN or session, a VN of eval/linefind's four detector-built sets, any screenshot an eval or
      eval/linefind manifest lists, or v14018 (synth/textdet_data._excluded, #97's rule for its real canvases):
        cal_dlg    whole-dialogue regions built exactly as the development wd cohort is (eval/finder_tighten.py
                   _frame_units: labelled non-name lines grouped at 1.6 line heights, their union plus 12 / 6 px,
                   grown by 0 / 4 / 8 px, shrunk away from every other detected text box): realplus's kept labels
                   (PC screenshots at the Vita target), the teacher's training labels (Vita, PSP and scaled PC
                   frames) and the owner pool's kept labels (native console captures; private)
        cal_empty  text-free regions cut as eval/linefind's `empty` regions are (build_gt.det_set): 300-600 x
                   60-140 px (half on the PSP target) in the upper 45% of the frame, at least 10 px (5 on the PSP)
                   from every detected text box (lines, detector boxes, Screen AI boxes), one per frame (20 tries)
      Each is a pickle of cases {"cohort", "unit", "shot", "vn", "kind", "variant", "region", "image", "lines"}
      (the wd cohorts' format; cal_empty cases have no lines), plus cohorts.json (counts and sha256).
  textgate_data.py synth --out DIR --shards N [--per-shard M] [--workers W]
      Synthetic canvases from #97's generator (synth/textdet_data.py: synth3 version 3's regions, loose regions,
      vertical text and empty negatives) with synth3's scene pool cut further: no screenshot of a VN of
      eval/linefind's four detector-built sets (vita_dev, psp_dev and the two holdouts, whose empty regions are
      scored), no screenshot any eval or eval/linefind manifest lists (research_a/b's included), and no CHAOS;CHILD
      (v14018): textdet_data._eval_exclusions, the rule #97 applied to its real canvases.
      #97's own synthetic shards drew their scenes from a pool that held 54 of the 66 development empty regions'
      screenshots (docs/textdet.md, "What changed after the freeze"); the gate trains on these instead.
  textgate_data.py manifest --dir DIR ... --out JSON
      sha256 of every shard (as synth/textdet_data.py manifest does), for the pin.

The gate's training split: every VN for which calib_vn() is false. The real canvases are #97's real shards
(data/manifests/textdet_shards_v1.json: realplus, teacher, devnative and the owner's), whose exclusions are the
protocol's; a canvas of a calibration VN is never trained on.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "realplus", ROOT / "teacher", ROOT / "synth", ROOT / "eval", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import finder_tighten as FT  # noqa: E402  (before textdet_data, which puts eval/linefind first on sys.path)

import textdet_data as TDD  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIVATE = HERD / "shared" / "private"
SPLIT_SALT = "textgate-split:"
CAL_CAPS = {"realplus": 2, "teacher": 10, "owner": 20}
EMPTY = {"w": (300, 600), "h": (60, 140), "top": 0.45, "pad": 10, "tries": 20}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calib_vn(vn):
    """The calibration split: a fifth of the training-side VNs, by hash (never trained on)."""
    return int(hashlib.sha256(f"{SPLIT_SALT}{vn}".encode()).hexdigest()[:8], 16) % 5 == 0


def exclusions():
    """(the textdet_data._excluded tuple, eval/linefind VNs, eval and linefind screenshot ids)."""
    from common import dev_sample_vns, reserved_sessions, test_set_vns
    ids, shas, lf_vns = TDD._eval_exclusions()
    return (ids, shas, lf_vns, set(test_set_vns()), set(dev_sample_vns()), set(reserved_sessions())), lf_vns, ids


def excluded(row, ex):
    return TDD._excluded(row, ex)


def _cap(rows, cap, key=lambda r: r["shot"]):
    by = {}
    for r in sorted(rows, key=lambda r: hashlib.sha256(key(r).encode()).hexdigest()):
        lst = by.setdefault(r["vn"], [])
        if len(lst) < cap:
            lst.append(r)
    return sorted((r for v in by.values() for r in v), key=key)


def _jsonl(path):
    for x in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(x)
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------- calibration frames

def realplus_frames(ex):
    namelike = FT.namelike
    kept = {}
    for r in _jsonl(ART / "realplus" / "labels.jsonl"):
        if r["status"] == "kept" and not r.get("name") and r.get("label") and not namelike(r["label"]):
            kept.setdefault(r["shot"], {})[r["id"].rsplit("_", 1)[1]] = (r["id"], r["label"])
    reads = {}
    for r in _jsonl(ART / "realplus" / "reads" / "reads.jsonl"):
        if r["shot"] in kept and r.get("dets") is not None:
            reads[r["shot"]] = r
    meta = {r["shot"]: r for r in _jsonl(ART / "realplus" / "reads" / "shots.jsonl")}
    rows = []
    for shot, r in reads.items():
        m = meta.get(shot) or {}
        for f in ("sha256", "session", "release", "url"):
            r.setdefault(f, m.get(f))
        if calib_vn(r["vn"]) and not excluded(r, ex):
            rows.append(r)
    return [("realplus", r, r["kind"], 1.0, kept[r["shot"]]) for r in _cap(rows, CAL_CAPS["realplus"])]


def teacher_frames(ex):
    namelike = FT.namelike
    labelled = {}
    for r in _jsonl(ART / "teacher" / "lines_full.jsonl"):
        if r.get("split") != "train" or not r.get("label") or namelike(r["label"]):
            continue
        for t in r.get("targets") or []:
            labelled.setdefault((r["shot"], t), {})[r["id"].rsplit("_", 1)[1]] = (r["id"], r["label"])
    shots = {s for s, _ in labelled}
    reads = {}
    for p in sorted((ART / "teacher" / "reads").glob("train.*of5.jsonl")):
        for r in _jsonl(p):
            if r["shot"] in shots:
                reads[r["shot"]] = r
    rows = [r for r in reads.values() if calib_vn(r["vn"]) and not excluded(r, ex)]
    by_shot = {}
    for (shot, t), lab in sorted(labelled.items()):
        by_shot.setdefault(shot, []).append((t, lab))
    return [("teacher", r, t, None, lab) for r in _cap(rows, CAL_CAPS["teacher"]) for t, lab in by_shot[r["shot"]]]


def owner_frames(ex):
    reads = FT._owner_reads()
    kept = {}
    for r in FT._owner_kept():
        if not r.get("name"):
            kept.setdefault(r["shot"], {})[r["id"].rsplit("_", 1)[1]] = (r["id"], r["label"])
    rows = [reads[s] for s in kept if s in reads and calib_vn(reads[s]["vn"]) and not excluded(reads[s], ex)]
    return [("owner", r, r["kind"], 1.0, kept[r["shot"]]) for r in _cap(rows, CAL_CAPS["owner"])]


def frame_grey(src, r, target):
    """(grey frame at the read's scale, scale of the read boxes)."""
    import cv2
    import geometry as G

    from common import to_gray
    from devnative import paths as P
    if src == "teacher":
        path = ART / "realdata" / "shots" / r["kind"] / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        sc = G.target_scale(rgb.shape[1], rgb.shape[0], target) if r["kind"] == "hires" else 1.0
        return to_gray(np.ascontiguousarray(G.scale_image(rgb, sc))), sc
    rgb = P.load_screen(r)
    if rgb is None:
        raise FileNotFoundError(r.get("path") or r["shot"])
    return to_gray(np.ascontiguousarray(rgb)), 1.0


def text_boxes(r, sc, W, H):
    """Every detected text box of a reads frame at scale sc: lines (ink, parts, furigana), detector boxes, Screen
    AI boxes."""
    import geometry as G
    out = []
    for ln in r.get("lines") or []:
        for b in [ln.get("crop_ink") or ln["ink"], ln.get("det")] + list(ln.get("parts") or []) + \
                list(ln.get("furi") or []):
            if b:
                out.append(G.crop_box(b, sc, 0, W, H))
    for d in r.get("dets") or []:
        out.append(G.crop_box(d["det"], sc, 0, W, H))
    for s_ in r.get("sai") or []:
        out.append(G.crop_box(s_["box"], sc, 0, W, H))
    return out


def empty_case(src, r, target, grey, sc, rng):
    H, W = grey.shape
    k = 0.5 if target == "psp" else 1.0
    boxes = text_boxes(r, sc, W, H)
    for _ in range(EMPTY["tries"]):
        w = int(rng.integers(EMPTY["w"][0], EMPTY["w"][1] + 1) * k)
        h = int(rng.integers(EMPTY["h"][0], EMPTY["h"][1] + 1) * k)
        top = int(EMPTY["top"] * H)
        if w > W or top - h < 1:
            return None
        x0, y0 = int(rng.integers(0, W - w + 1)), int(rng.integers(0, top - h + 1))
        x1, y1 = x0 + w, y0 + h
        pad = EMPTY["pad"] * k
        if all(b[2] < x0 - pad or b[0] > x1 + pad or b[3] < y0 - pad or b[1] > y1 + pad for b in boxes):
            return {"cohort": "cal_empty", "unit": f"{r['shot']}@{target}@empty", "shot": r["shot"], "vn": r["vn"],
                    "kind": target, "variant": "base", "region": [x0, y0, x1, y1], "source": src,
                    "image": np.ascontiguousarray(grey[y0:y1, x0:x1]), "lines": []}
    return None


def _frame_job(job):
    src, r, target, _, lab = job
    try:
        grey, sc = frame_grey(src, r, target)
    except (OSError, FileNotFoundError) as e:
        return [], None, f"{r['shot']}: {e}"
    cases = FT._frame_units("cal_dlg", r, target, grey, lab, lambda k, lab=lab: FT.namelike(lab[k][1]), scale=sc)
    for c in cases:
        c["source"] = src
    rng = np.random.default_rng(int(hashlib.sha256(f"textgate-empty:{r['shot']}:{target}".encode())
                                    .hexdigest()[:8], 16))
    return cases, empty_case(src, r, target, grey, sc, rng), None


def cmd_cal(args):
    ex, _, _ = exclusions()
    jobs = realplus_frames(ex) + teacher_frames(ex) + owner_frames(ex)
    print(f"{len(jobs)} calibration frames: " + ", ".join(f"{s} {sum(j[0] == s for j in jobs)}"
                                                          for s in CAL_CAPS), flush=True)
    dlg, empty, errors = [], [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for cases, e, err in pool.imap(_frame_job, jobs, chunksize=8):
            dlg += cases
            if e is not None:
                empty.append(e)
            if err:
                errors.append(err)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, cases in (("cal_dlg", dlg), ("cal_empty", empty)):
        with open(out / f"{name}.pkl", "wb") as f:
            pickle.dump(cases, f, protocol=4)
        base = [c for c in cases if c["variant"] == "base"]
        summary[name] = {"cases": len(cases), "regions": len(base), "vns": len({c["vn"] for c in cases}),
                         "by_source": {s: sum(c["source"] == s for c in cases) for s in CAL_CAPS},
                         "by_kind": {k: sum(c["kind"] == k for c in cases) for k in ("vita", "psp")},
                         "lines": sum(len(c["lines"]) for c in base), "sha256": sha256(out / f"{name}.pkl")}
        print(name, summary[name], flush=True)
    summary["errors"] = errors
    summary["split"] = {"salt": SPLIT_SALT, "calib": "sha256(salt + vn)[:8] % 5 == 0", "caps": CAL_CAPS,
                        "empty": EMPTY}
    (out / "cohorts.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- clean synthetic shards

def clean_scene_pool():
    """synth3's scene pool without the VNs of eval/linefind's four detector-built sets, any screenshot an eval or
    eval/linefind manifest lists, and v14018 (in place, in textdet_data's synth3)."""
    _, lf_vns, ids = exclusions()
    TD = TDD
    S3 = TD.S3
    before = {k: len(v) for k, v in S3.R.bg.items()}

    def keep(p):
        name = Path(p).name
        vn = name.split("_")[0]
        sid = name.rsplit("_", 1)[-1].split(".")[0]
        return vn not in lf_vns and vn not in TD.REPORT_ONLY_VNS and sid not in ids
    S3.R.bg = {k: [p for p in v if keep(p)] for k, v in S3.R.bg.items()}
    S3.R.bg_vns = sorted({Path(p).name.split("_")[0] for v in S3.R.bg.values() for p in v})
    return before, {k: len(v) for k, v in S3.R.bg.items()}


def cmd_synth(args):
    import time
    TD = TDD
    TD.init_synth()
    before, after = clean_scene_pool()
    print(f"scene pool {before} -> {after}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene_pool.json").write_text(json.dumps({"before": before, "after": after,
                                                     "paths": sorted(str(p) for v in TD.S3.R.bg.values()
                                                                     for p in v)}, indent=0) + "\n")
    jobs = [(i, args.per_shard, args.seed, str(out)) for i in range(args.first, args.first + args.shards)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for idx, n, errors, dt in pool.imap_unordered(TD.run_synth_shard, jobs):
            print(f"synth shard {idx}: {n} samples ({errors} errors) in {dt:.0f}s ({time.time() - t0:.0f}s)",
                  flush=True)
    print("SYNTH_DONE", flush=True)


def cmd_manifest(args):
    TD = TDD
    shards = []
    for d in args.dir:
        for p in sorted(Path(d).glob("*.json")):
            if p.name.endswith(".tmp.json") or p.name in ("scene_pool.json", "cohorts.json"):
                continue
            data = p.with_suffix(".npy")
            if not data.exists():
                continue
            idx = json.loads(p.read_text(encoding="utf-8"))["samples"]
            src = {}
            for s in idx:
                src[s["src"]] = src.get(s["src"], 0) + 1
            shards.append({"dir": Path(d).name, "name": p.stem, "npy_sha256": sha256(data), "json_sha256": sha256(p),
                           "samples": len(idx), "lines": sum(len(s["lines"]) for s in idx), "by_src": src})
    doc = {"kind": "textgate-shards", "schema": 1,
           "code_sha256": {f: sha256(ROOT / f) for f in ("eval/textgate_data.py",) + TD.CODE},
           "scene_pool": json.loads((Path(args.dir[0]) / "scene_pool.json").read_text())["after"]
           if (Path(args.dir[0]) / "scene_pool.json").exists() else None,
           "shards": shards, "totals": {"samples": sum(s["samples"] for s in shards),
                                        "lines": sum(s["lines"] for s in shards)}}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {doc['totals']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cal")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    p = sub.add_parser("synth")
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--per-shard", type=int, default=2000)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    p = sub.add_parser("manifest")
    p.add_argument("--dir", action="append", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"cal": cmd_cal, "synth": cmd_synth, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

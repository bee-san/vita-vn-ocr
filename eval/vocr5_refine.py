#!/usr/bin/env python3
"""VOCR5's refined copies of the GRAD pools (eval/vocr5_protocol.json "refined_copies"; prompts/vocr5-aug.md part B).

  vocr5_refine.py build --src GRAD.npz --out REF.npz [--report JSON] [--name NAME] [--force] [--no-reserve-check]
  vocr5_refine.py realplus SNAPSHOT [--force]       one command for a realplus-v3 snapshot (v3i6, v3i7, ..., v3)
  vocr5_refine.py all [--realplus SNAPSHOT]          every GRAD pool VOCR5 fine-tunes on
  vocr5_refine.py check-devnative [--out JSON]       refining a stored GRAD crop = refining the frame's GRAD box
  vocr5_refine.py summary --reports REPORT... [--devnative JSON] --out JSON   the results, counts only (no text)

build: for each crop of a GRAD npz (train2's shard format: data, offsets, shapes, labels, meta) the screen grey is
  255 - g when meta says inverted, else g; synth/lines_refine.refine_lines (the bit-exact twin of
  runtime/vocr_lines_refine.c, flags VOCR_REFINE_DEFAULT) runs on it with the whole crop as the one finder box and the
  line's dark flag (meta dark). A refined copy is kept iff the refinement returns exactly one line whose box differs
  from the whole crop and is at least MIN_H px tall; it is cut, inverted again when the stored crop was, and keeps the
  label and the meta, plus
    refined_from  {"file", "sha256", "row"}: the source npz and the row the copy was cut from
    refine_box    the refined line's box in the frame, [x0, y0, x1, y1] like meta box (box stays the finder's GRAD box;
                  chars stay relative to it)
  Lines the refinement splits are skipped (their label cannot be divided). Counts (kept, unchanged, split, too small)
  and geometry go to the report (default REF.report.json, no text). The output is written deterministically (fixed
  zip timestamps), so the same input and code give the same sha256; an existing output is never replaced without
  --force. eval/tools/reserve_check.py --dump DUMP --pixels then runs on the output (REF.reserve_check.json, next to
  it); its verdict is the exit status. A source under $HERD/shared/private (the owner's frames) may only be written
  under it, and *_dev.npz files (the development sample) are refused.
realplus / all: the same with the pools' fixed paths, outputs in $HERD/shared/artifacts/vocr5-train/aug/ and the
  owner pool's in $HERD/shared/private/vocr5-train/aug/. Names follow the conductor's pool names (rp3_grad, own_grad,
  nat_grad, dntv_grad, dnst_grad), for eval/vocr5_pools.py sources --refined NAME=REF.npz.
check-devnative: devnative keeps each line's dialogue region and its GRAD crop. On every training-side line (the
  *_grad_train.npz parts of #84's re-carve; never the *_dev.npz files) the frame is loaded, the C finder's twin runs on
  the region, the line's finder box is found again, and the refinement of the whole region's finder set (the console's
  call) must give that line exactly the box(es) and dark flag(s) that refining the stored crop alone gives. Prints
  DEVNATIVE_AGREE or DEVNATIVE_DISAGREE (exit 1).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "synth", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lines_refine as RF  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIVATE = HERD / "shared" / "private"
OUT_DIR = ART / "vocr5-train" / "aug"
PRIVATE_OUT_DIR = PRIVATE / "vocr5-train" / "aug"
DUMP = ART / "native-frames" / "vndb" / "dump" / "db"
RECLAIM = ART / "owner-drive-ingest-2" / "reclaim" / "pools"
DEVNATIVE_TRAIN = (RECLAIM / "devnative_tv_native_grad_train.npz", RECLAIM / "devnative_strict_grad_train.npz")
DEVNATIVE_LINES = ART / "devnative" / "v3" / "lines_full.jsonl"
REALPLUS_SNAPSHOT = "v3i6"
MIN_H = 8
STATUSES = ("kept", "unchanged", "split", "too_small")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def pools(realplus=REALPLUS_SNAPSHOT):
    """{name: source GRAD npz} of every GRAD pool VOCR5's fine-tunes read (eval/vocr5_protocol.json real_pools)."""
    return {"rp3_grad": ART / "realplus-v3" / f"realplus_train_grad_{realplus}.npz",
            "own_grad": PRIVATE / "owner-games" / "_build2" / "owner-pool-2" / "owner_train_grad_v2.npz",
            "nat_grad": ART / "native-frames" / "pool" / "native_train_grad_v1.npz",
            "dntv_grad": DEVNATIVE_TRAIN[0], "dnst_grad": DEVNATIVE_TRAIN[1]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_private(path):
    p = Path(path).resolve()
    return p == PRIVATE.resolve() or PRIVATE.resolve() in p.parents


def out_path(src, private):
    """Where a source's refined copy goes: <stem>_ref.npz in OUT_DIR, or PRIVATE_OUT_DIR for the owner's frames."""
    return (PRIVATE_OUT_DIR if private else OUT_DIR) / f"{Path(src).stem}_ref.npz"


def save_npz(path, arrays):
    """np.savez_compressed's format with fixed zip timestamps: the same arrays give the same bytes."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name, a in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(a), allow_pickle=False)
    tmp.replace(path)


def refine_stored(g, inverted, dark, flags=RF.DEFAULT):
    """The refinement of a stored crop with the whole crop as the one finder box: (boxes [(x, y, w, h)], dark flags),
    crop-relative, on the screen grey (255 - g for an inverted crop)."""
    h, w = g.shape
    screen = 255 - g if inverted else g
    boxes, darks, _ = RF.refine_lines(screen, [(0, 0, w, h)], [int(dark)], flags)
    return boxes, darks


def classify(boxes, shape, min_h=MIN_H):
    """kept / unchanged / split / too_small for a stored crop's refined boxes."""
    h, w = shape
    if len(boxes) != 1:
        return "split"
    if tuple(boxes[0]) == (0, 0, w, h):
        return "unchanged"
    return "too_small" if boxes[0][3] < min_h else "kept"


def refined_copy(g, inverted, box):
    """The stored-polarity crop of box (x, y, w, h) of stored crop g: cut from the screen grey, inverted again."""
    x, y, w, h = box
    screen = 255 - g if inverted else g
    cut = screen[y:y + h, x:x + w]
    return np.ascontiguousarray(255 - cut if inverted else cut, dtype=np.uint8)


def ctc_frames_short(label, shape):
    """True when the crop at height 32 (train.resize_h's width) has fewer CTC frames than the label needs (train2's
    load_shard would drop it): one class per character, plus a blank between adjacent repeats."""
    h, w = shape
    nw = max(8, min(1024, int(round(w * 32 / h))))
    need = len(label) + sum(a == b for a, b in zip(label, label[1:]))
    return need > -(-nw // 8)


def build(src, out, name=None, report=None, force=False, reserve=True, dump=DUMP, flags=RF.DEFAULT):
    """Write src's refined copy to out (and its report); returns the report dict."""
    src, out = Path(src), Path(out)
    if src.name.endswith("_dev.npz"):
        raise SystemExit(f"{src}: a development-sample file; refined copies are built from training pools only")
    if is_private(src) and not is_private(out):
        raise SystemExit(f"{src} holds the owner's private frames: write its copy under {PRIVATE}, not {out}")
    if src.resolve() == out.resolve():
        raise SystemExit("--out would overwrite --src")
    if out.exists() and not force:
        raise SystemExit(f"{out} exists (it may be pinned); pass --force to replace it")
    report = Path(report) if report else out.with_name(out.stem + ".report.json")
    t0 = time.time()
    src_sha = sha256(src)
    z = np.load(src)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    if not (len(labels) == len(meta) == len(shapes) == len(offs) - 1):
        raise ValueError(f"{src}: data, labels and meta have different row counts")
    if any("refined_from" in m for m in meta):
        raise SystemExit(f"{src} is a refined copy already; refine the GRAD pool it came from")
    counts, by_target, split_n = Counter(), {}, Counter()
    crops, out_labels, out_meta = [], [], []
    removed_rows, trimmed_cols, ctc_short, src_ctc_short = [], 0, 0, 0
    for i, m in enumerate(meta):
        h, w = (int(v) for v in shapes[i])
        g = data[offs[i]:offs[i + 1]].reshape(h, w)
        if "dark" not in m or "inverted" not in m:
            raise ValueError(f"{src} row {i}: meta has no dark / inverted flag")
        boxes, _ = refine_stored(g, bool(m["inverted"]), int(m["dark"]), flags)
        st = classify(boxes, (h, w))
        counts[st] += 1
        by_target.setdefault(str(m.get("target")), Counter())[st] += 1
        if st == "split":
            split_n[len(boxes)] += 1
        if st != "kept":
            continue
        x, y, bw, bh = boxes[0]
        b = m.get("box")
        if not b or (b[3] - b[1], b[2] - b[0]) != (h, w):
            raise ValueError(f"{src} row {i}: meta box {b} is not the crop's {w}x{h} frame box")
        crops.append(refined_copy(g, bool(m["inverted"]), boxes[0]))
        out_labels.append(labels[i])
        out_meta.append(dict(m, refined_from={"file": src.name, "sha256": src_sha, "row": i},
                             refine_box=[b[0] + x, b[1] + y, b[0] + x + bw, b[1] + y + bh]))
        removed_rows.append(h - bh)
        trimmed_cols += bw < w
        ctc_short += ctc_frames_short(labels[i], (bh, bw))
        src_ctc_short += ctc_frames_short(labels[i], (h, w))
    out.parent.mkdir(parents=True, exist_ok=True)
    shp = np.array([c.shape for c in crops], np.int32).reshape(-1, 2)
    off = np.concatenate([[0], np.cumsum(shp[:, 0].astype(np.int64) * shp[:, 1])]).astype(np.int64)
    arrays = {"data": np.concatenate([c.ravel() for c in crops]) if crops else np.zeros(0, np.uint8),
              "offsets": off, "shapes": shp,
              "labels": np.array(json.dumps(out_labels, ensure_ascii=False)),
              "meta": np.array(json.dumps(out_meta, ensure_ascii=False))}
    save_npz(out, arrays)
    with np.load(out) as chk:                      # read back what train2 will read
        if (len(json.loads(str(chk["labels"]))) != len(crops) or not np.array_equal(chk["offsets"], off)
                or not np.array_equal(chk["data"], arrays["data"])):
            raise RuntimeError(f"{out}: read-back differs from what was written")
    rr = np.asarray(removed_rows, np.int64)
    rep = {"kind": "vocr5-refined-copy", "schema": 1, "name": name, "src": str(src), "src_sha256": src_sha,
           "out": str(out), "out_sha256": sha256(out), "flags": flags, "min_h": MIN_H, "rows": len(meta),
           "counts": {k: counts[k] for k in STATUSES}, "by_target": {t: {k: c[k] for k in STATUSES}
                                                                     for t, c in sorted(by_target.items())},
           "split_lines": {str(k): v for k, v in sorted(split_n.items())},
           "kept_geometry": {"rows_removed_mean": round(float(rr.mean()), 3) if len(rr) else None,
                             "rows_removed_median": float(np.median(rr)) if len(rr) else None,
                             "columns_trimmed": int(trimmed_cols)},
           "ctc_short_info": {"refined": int(ctc_short), "their_sources": int(src_ctc_short),
                              "note": "kept copies whose height-32 width holds fewer CTC frames than the label needs "
                                      "(train2's load_shard drops such a line); information only"},
           "code": {"eval/vocr5_refine.py": sha256(Path(__file__)),
                    "synth/lines_refine.py": sha256(ROOT / "synth" / "lines_refine.py")},
           "git": _git_head(), "seconds": round(time.time() - t0, 1)}
    print(f"{name or src.stem}: {len(meta)} rows -> kept {counts['kept']}, unchanged {counts['unchanged']}, "
          f"split {counts['split']}, too small {counts['too_small']} -> {out}", flush=True)
    if reserve:
        rep["reserve_check"] = reserve_check(name or out.stem, out, dump)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return rep


def reserve_check(name, out, dump=DUMP):
    """eval/tools/reserve_check.py NAME=OUT --dump DUMP --pixels, its JSON next to OUT; returns verdict and path."""
    out = Path(out)
    js = out.with_name(out.stem + ".reserve_check.json")
    cmd = [sys.executable, str(HERE / "tools" / "reserve_check.py"), f"{name}={out}", "--dump", str(dump), "--pixels",
           "--out", str(js)]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=ROOT)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr[-4000:])
    verdict = json.loads(js.read_text(encoding="utf-8"))["verdict"] if js.exists() else "RESERVE_ERROR"
    return {"verdict": verdict, "json": str(js), "returncode": r.returncode, "command": cmd[1:]}


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ----------------------------------------------------------------------------- devnative agreement

def frame_agreement(region, gbox_rel, stored, meta_dark, meta_inverted, flags=RF.DEFAULT):
    """One line: the finder's twin on the region, the line found again by its box (x, y, w, h) relative to the region,
    and the refinement of the region's whole finder set against that of the stored crop alone. Returns a dict of
    checks (all True when the two agree)."""
    from common import find_lines_ex
    boxes, dark, _ = find_lines_ex(region, "grad")
    hit = [k for k, b in enumerate(boxes) if tuple(b) == tuple(gbox_rel)]
    res = {"box_reproduced": len(hit) == 1, "crop_reproduced": False, "dark_equal": False, "refine_agree": False,
           "status": None}
    if len(hit) != 1:
        return res
    k = hit[0]
    x, y, w, h = boxes[k]
    mine = region[y:y + h, x:x + w]
    mine = 255 - mine if dark[k] else mine
    res["crop_reproduced"] = mine.shape == stored.shape and bool((mine == stored).all())
    res["dark_equal"] = int(dark[k]) == int(meta_dark) and bool(meta_inverted) == bool(meta_dark)
    rb, rd, ro = RF.refine_lines(region, boxes, dark, flags)
    framed = ([(bx - x, by - y, bw, bh) for (bx, by, bw, bh), o in zip(rb, ro) if o == k],
              [d for d, o in zip(rd, ro) if o == k])
    alone = refine_stored(stored, bool(meta_inverted), int(meta_dark), flags)
    res["refine_agree"] = ([tuple(b) for b in framed[0]], list(framed[1])) == ([tuple(b) for b in alone[0]],
                                                                             list(alone[1]))
    res["status"] = classify(alone[0], stored.shape)
    return res


def check_devnative(train_npz=DEVNATIVE_TRAIN, lines=DEVNATIVE_LINES, shots=ROOT / "devnative" / "manifests" /
                    "shots.jsonl"):
    from common import to_gray
    from devnative import paths as P
    rows = []
    files = []
    for f in map(Path, train_npz):
        if f.name.endswith("_dev.npz"):
            raise SystemExit(f"{f}: the development sample is never read here")
        z = np.load(f)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        meta = json.loads(str(z["meta"]))
        files.append({"file": str(f), "sha256": sha256(f), "rows": len(meta)})
        for i, m in enumerate(meta):
            h, w = (int(v) for v in shapes[i])
            rows.append((f.name, i, m, data[offs[i]:offs[i + 1]].reshape(h, w)))
    want = {m["id"] for _, _, m, _ in rows}
    full = {}
    for x in Path(lines).read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r["id"] in want:                         # training-side lines only
            full[r["id"]] = {k: r[k] for k in ("id", "shot", "region", "grad")}
    man = {}
    for x in Path(shots).read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        man[r["shot"]] = r
    screens, out, totals = {}, [], Counter()
    for fname, i, m, g in rows:
        f = full.get(m["id"])
        rec = {"file": fname, "row": i, "id": m["id"]}
        if f is None or f["grad"]["box"] != m["box"]:
            rec.update(found=False)
            out.append(rec)
            totals["line_missing"] += 1
            continue
        if f["shot"] not in screens:
            rgb = P.load_screen(man[f["shot"]])
            if rgb is None:
                raise SystemExit(f"{f['shot']}: screenshot not readable")
            screens[f["shot"]] = to_gray(np.ascontiguousarray(rgb))
        x0, y0, x1, y1 = f["region"]
        region = np.ascontiguousarray(screens[f["shot"]][y0:y1, x0:x1])
        gb = m["box"]
        res = frame_agreement(region, (gb[0] - x0, gb[1] - y0, gb[2] - gb[0], gb[3] - gb[1]), g, m["dark"],
                              m["inverted"])
        rec.update(found=True, **res)
        out.append(rec)
        for k in ("box_reproduced", "crop_reproduced", "dark_equal", "refine_agree"):
            totals[k] += bool(res[k])
        totals[f"status:{res['status']}"] += 1
    n = len(rows)
    ok = all(totals[k] == n for k in ("box_reproduced", "crop_reproduced", "dark_equal", "refine_agree"))
    return {"kind": "vocr5-refine-devnative-agreement", "schema": 1, "files": files, "lines": n,
            "screenshots": len(screens), "flags": RF.DEFAULT, "totals": dict(sorted(totals.items())),
            "disagree": [r for r in out if not (r.get("refine_agree") and r.get("crop_reproduced")
                                                and r.get("dark_equal"))],
            "verdict": "DEVNATIVE_AGREE" if ok and n else "DEVNATIVE_DISAGREE",
            "code": {"eval/vocr5_refine.py": sha256(Path(__file__)),
                     "synth/lines_refine.py": sha256(ROOT / "synth" / "lines_refine.py"),
                     "synth/common.py": sha256(ROOT / "synth" / "common.py")}, "git": _git_head()}


# ----------------------------------------------------------------------------- CLI

SUMMARY_KEYS = ("name", "src", "src_sha256", "out", "out_sha256", "flags", "min_h", "rows", "counts", "by_target",
                "split_lines", "kept_geometry", "ctc_short_info", "code", "git")


def summary(reports, devnative=None):
    """One results document from the per-pool reports and the devnative agreement (counts, paths, sha256; no text)."""
    pools_ = []
    for p in reports:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        e = {k: r.get(k) for k in SUMMARY_KEYS}
        rc = r.get("reserve_check") or {}
        e["reserve_check"] = {"verdict": rc.get("verdict"), "json": rc.get("json")}
        if rc.get("json") and Path(rc["json"]).exists():
            pool = next(iter(json.loads(Path(rc["json"]).read_text(encoding="utf-8"))["pools"].values()))
            e["reserve_check"].update(blockers=pool["blockers"], hash_coverage=pool["hash_coverage"])
        pools_.append(e)
    doc = {"kind": "vocr5-refined-copies", "schema": 1, "spec": "prompts/vocr5-aug.md part B; eval/vocr5_protocol.json "
           "data.real_pools.refined_copies", "pools": pools_,
           "totals": {k: sum(e["counts"][k] for e in pools_) for k in STATUSES}}
    if devnative:
        d = json.loads(Path(devnative).read_text(encoding="utf-8"))
        doc["devnative_agreement"] = {k: d[k] for k in ("verdict", "lines", "screenshots", "totals", "files", "flags",
                                                        "code", "git")}
        doc["devnative_agreement"]["disagree"] = len(d["disagree"])
    return doc


def _build_named(name, src, force):
    rep = build(src, out_path(src, is_private(src)), name=name, force=force)
    return rep.get("reserve_check", {}).get("verdict") == "RESERVE_OK"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the refined copy of one GRAD npz")
    b.add_argument("--src", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--report")
    b.add_argument("--name", help="pool name for the report and reserve_check (default: the output's stem)")
    b.add_argument("--force", action="store_true", help="replace an existing output")
    b.add_argument("--no-reserve-check", action="store_true")
    b.add_argument("--dump", default=str(DUMP), help="VNDB dump db directory for reserve_check --dump")
    r = sub.add_parser("realplus", help="the refined copy of a realplus-v3 snapshot's GRAD npz")
    r.add_argument("snapshot", help="v3i6, v3i7, ..., v3: realplus-v3/realplus_train_grad_<SNAPSHOT>.npz")
    r.add_argument("--force", action="store_true")
    a = sub.add_parser("all", help="every GRAD pool of VOCR5's fine-tunes")
    a.add_argument("--realplus", default=REALPLUS_SNAPSHOT)
    a.add_argument("--force", action="store_true")
    c = sub.add_parser("check-devnative", help="stored crop refinement = frame GRAD box refinement (devnative train)")
    c.add_argument("--out")
    s = sub.add_parser("summary", help="one text-free results JSON from the reports and the devnative check")
    s.add_argument("--reports", nargs="+", required=True)
    s.add_argument("--devnative")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "summary":
        doc = summary(args.reports, args.devnative)
        Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{args.out}: {len(doc['pools'])} pools, totals {doc['totals']}")
        return 0
    if args.cmd == "build":
        rep = build(args.src, args.out, name=args.name, report=args.report, force=args.force,
                    reserve=not args.no_reserve_check, dump=args.dump)
        v = rep.get("reserve_check", {}).get("verdict")
        return 0 if v in (None, "RESERVE_OK") else 1
    if args.cmd == "realplus":
        return 0 if _build_named("rp3_grad", pools(args.snapshot)["rp3_grad"], args.force) else 1
    if args.cmd == "all":
        ok = [_build_named(n, p, args.force) for n, p in pools(args.realplus).items()]
        return 0 if all(ok) else 1
    res = check_devnative()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"{res['verdict']}: {res['lines']} training-side lines on {res['screenshots']} screenshots, "
          f"{json.dumps(res['totals'])}")
    return 0 if res["verdict"] == "DEVNATIVE_AGREE" else 1


if __name__ == "__main__":
    sys.exit(main())

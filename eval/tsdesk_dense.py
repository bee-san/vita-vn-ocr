#!/usr/bin/env python3
"""textspot-desktop's held-out dense PC sets (issue #173, eval/textspot_desktop_protocol.json "sets.dense_rule"):
desktop-data's 1b v3 frames (realplus PC screenshots at their own size, #171) with at least 8 labelled horizontal
text lines, from 200 VNs held out of every training source: 100 VNs for screening (dense_screen), 100 for the
decision (dense_dev). MeikiOCR's author found textspot8 weak on screens with a lot of text; these sets measure it.

  tsdesk_dense.py build --out-dir DIR [--manifest-out data/manifests/tsdesk_dense_v1.json]
      the VN draw, the frames files DIR/dense_screen.pkl and DIR/dense_dev.pkl (eval/desktop.py's frames format,
      with each frame's 1b ignore areas), the skip list DIR/skip_vns.txt (every frame of the 200 VNs leaves the
      training mix) and the committed manifest (VNDB ids, screenshot ids and sha256, counts; no pixels, no text)

The rule, as registered:
  candidates  1b v3's frames (data/manifests/desktop_data/realplus_native_v3.json) with >= 8 horizontal positives of
              class text; their VNs minus every VN of a hooked pools v2.1 title, of dev-v2 and of the PC test set
  draw        the VNs sorted by sha256("textspot-desktop-dense-v1:" + vn): the first 100 dense_screen, the next 100
              dense_dev; per VN its first 3 dense frames by sha256("textspot-desktop-dense-v1:" + shot)
  units       one per labelled text line (mida-ocr-eval's pc1b.py rule): the line's region is its box grown by 0.3
              line heights, the vertical growth capped at half the gap to an x-overlapping line above or below; its
              text is realplus v3's label (lines_full_v3.jsonl), matched to the 1b line by box (the plan's ink box at
              the fit-960 scale divided by the fit, IoU >= 0.3); a label matched by two lines is dropped
  reading     a detector box at least half inside one of the frame's 1b ignore areas is dropped before reading
              (install_ignore below), as pc1b.py does
"""
import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "tools", ROOT / "synth", ROOT / "train", ROOT / "teacher", ROOT / "runtime", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import desktop as D  # noqa: E402

DD = D.ART / "desktop-data"
MAN = ROOT / "data" / "manifests" / "desktop_data" / "realplus_native_v3.json"
SHARDS = DD / "pages" / "realplus_v3"
PLAN = DD / "realplus" / "plan_v1.json"
LINES = D.ART / "realplus-v3" / "lines_full_v3.jsonl"
PCTEST = ROOT / "eval" / "manifests" / "desktop-pc-test-v1.jsonl"
DEV_FRAMES = D.PV / "vocr-desktop" / "frames" / "dev.pkl"
SALT = "textspot-desktop-dense-v1:"
MIN_LINES = 8
N_VNS = 100
PER_VN = 3


def h(s):
    return hashlib.sha256((SALT + s).encode()).hexdigest()


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def regions(boxes, W, H, grow=0.3):
    """[x0, y0, x1, y1] int regions of line boxes [x0, y0, x1, y1] (pc1b.py's rule)."""
    out = []
    for i, b in enumerate(boxes):
        hh = b[3] - b[1]
        up = down = grow * hh
        for j, c in enumerate(boxes):
            if j == i or min(b[2], c[2]) - max(b[0], c[0]) <= 0:
                continue
            if c[3] <= b[1] + 1e-6:
                up = min(up, max(0.0, (b[1] - c[3]) / 2))
            elif c[1] >= b[3] - 1e-6:
                down = min(down, max(0.0, (c[1] - b[3]) / 2))
        x0, x1 = max(0, int(b[0] - grow * hh)), min(W, int(b[2] + grow * hh + 0.999))
        y0, y1 = max(0, int(b[1] - up)), min(H, int(b[3] + down + 0.999))
        out.append([x0, y0, x1, y1])
    return out


def excluded_vns():
    out = set(D.hooked_vns())
    frames, _ = D.load_frames(DEV_FRAMES)
    out |= {f["cluster"] for f in frames}
    for line in PCTEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(json.loads(line)["vn"])
    return out


def dense_samples(man):
    """{shot: sample} of 1b v3's frames with >= MIN_LINES horizontal text positives (shard json sha256 checked)."""
    out = {}
    for s in man["shards"]:
        p = SHARDS / s["dir"] / f"{s['name']}.json"
        if D.sha256_file(p) != s["json_sha256"]:
            raise SystemExit(f"{p}: sha256 differs from the manifest's")
        for smp in json.loads(p.read_text(encoding="utf-8"))["samples"]:
            if sum(1 for ln in smp["lines"] if not ln[4] and ln[5] == "text") >= MIN_LINES:
                out[smp["shot"]] = smp
    return out


def frame_of(shot, smp, pr, labels, stats):
    W, H = smp["w"], smp["h"]
    fit = float(smp["meta"]["fit"])
    cands = []
    for pl in pr["lines"]:
        lab = labels.get(pl["id"])
        if lab is None or pl["name"]:
            continue
        crops = lab.get("crops") or {}
        ink = pl.get("vita_ink") or crops.get("vita_ink") or crops.get("vita_grad")
        if not ink:
            stats["label_without_box"] += 1
            continue
        x0, y0, x1, y1 = ink
        cands.append(([x0 / fit, y0 / fit, x1 / fit, y1 / fit], lab["label"], pl["id"]))
    lines = []
    for ln in smp["lines"]:
        if ln[4] or ln[5] != "text":
            continue
        stats["lines"] += 1
        best = max(cands, key=lambda c: iou(c[0], ln[:4]), default=None)
        if best is None or iou(best[0], ln[:4]) < 0.3:
            stats["unmatched"] += 1
            continue
        lines.append(([float(v) for v in ln[:4]], best[1], best[2]))
    ids = Counter(i for _, _, i in lines)
    dup = {i for i, n in ids.items() if n > 1}
    stats["dup_dropped"] += sum(1 for _, _, i in lines if i in dup)
    lines = [ln for ln in lines if ln[2] not in dup]
    regs = regions([ln[0] for ln in lines], W, H)
    units = []
    for k, ((x0, y0, x1, y1), text, lid) in enumerate(lines):
        box = [x0, y0, x1 - x0, y1 - y0]
        units.append({"uid": f"dense:{lid}", "cluster": smp["vn"], "group": "dense", "region": regs[k],
                      "lines": [{"text": text, "box": box, "cells": D.even_cells(box, text)}]})
    return {"key": shot, "set": "dense", "view": "dense", "cluster": smp["vn"], "W": W, "H": H,
            "src": {"kind": "dense", "path": pr["path"], "sha256": pr["sha256"]},
            "ignore": [[float(v) for v in b[:4]] for b in smp["ignore"]], "units": units}


def cmd_build(a):
    man = json.loads(MAN.read_text(encoding="utf-8"))
    if D.sha256_file(PLAN) != man["plan_sha256"]:
        raise SystemExit("plan_v1.json is not the plan 1b v3 was built from")
    plan = {r["shot"]: r for r in json.loads(PLAN.read_text(encoding="utf-8"))["rows"]}
    dense = dense_samples(man)
    skip = excluded_vns()
    by_vn = defaultdict(list)
    for shot, smp in dense.items():
        if smp["vn"] not in skip:
            by_vn[smp["vn"]].append(shot)
    vns = sorted(by_vn, key=h)
    if len(vns) < 2 * N_VNS:
        raise SystemExit(f"only {len(vns)} candidate VNs")
    draw = {"dense_screen": vns[:N_VNS], "dense_dev": vns[N_VNS:2 * N_VNS]}
    want = {name: [s for v in vs for s in sorted(by_vn[v], key=h)[:PER_VN]] for name, vs in draw.items()}
    need = {s for ss in want.values() for s in ss}
    labels = {}
    with open(LINES, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["shot"] in need:
                labels[r["id"]] = r
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {"kind": "textspot-desktop-dense-sets", "schema": 1, "issue": 173,
              "protocol": "eval/textspot_desktop_protocol.json sets.dense_rule", "salt": SALT,
              "min_text_lines": MIN_LINES, "vns_per_set": N_VNS, "frames_per_vn": PER_VN,
              "source": {"manifest": str(MAN.relative_to(ROOT)), "manifest_sha256": D.sha256_file(MAN),
                         "plan_sha256": man["plan_sha256"], "lines_v3_sha256": D.sha256_file(LINES)},
              "candidates": {"dense_frames": len(dense), "dense_vns": len({s["vn"] for s in dense.values()}),
                             "excluded_vns_present": len({s["vn"] for s in dense.values()} & skip),
                             "eligible_vns": len(vns)},
              "sets": {}}
    for name in ("dense_screen", "dense_dev"):
        stats = Counter()
        frames = [frame_of(s, dense[s], plan[s], labels, stats) for s in want[name]]
        frames = [fr for fr in frames if fr["units"]]
        for fr in frames:
            D.check_sha(fr)
        meta = {"set": name, "frames": len(frames), "units": sum(len(f["units"]) for f in frames),
                "clusters": len({f["cluster"] for f in frames}), "stats": dict(stats),
                "chars": sum(len(ln["text"]) for f in frames for u in f["units"] for ln in u["lines"]),
                "sizes": dict(Counter(f"{f['W']}x{f['H']}" for f in frames).most_common(15))}
        with open(out / f"{name}.pkl", "wb") as fo:
            pickle.dump({"frames": frames, "meta": meta}, fo)
        record["sets"][name] = dict(meta, frames_file_sha256=D.sha256_file(out / f"{name}.pkl"),
                                    vns=draw[name],
                                    shots=[{"shot": f["key"], "vn": f["cluster"], "sha256": f["src"]["sha256"],
                                            "size": [f["W"], f["H"]], "units": len(f["units"])} for f in frames])
        print(json.dumps({k: meta[k] for k in ("set", "frames", "units", "clusters", "chars")}), flush=True)
    held = sorted(draw["dense_screen"] + draw["dense_dev"])
    (out / "skip_vns.txt").write_text("\n".join(held) + "\n", encoding="utf-8")
    held_frames = sum(1 for smp in dense.values() if smp["vn"] in set(held))
    record["held_out"] = {"vns": len(held), "skip_vns_sha256": D.sha256_file(out / "skip_vns.txt"),
                          "dense_frames_of_held_vns": held_frames}
    if a.manifest_out:
        Path(a.manifest_out).write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(record["candidates"]), json.dumps(record["held_out"]), flush=True)


# ----------------------------------------------------------------------------- reading with the ignore filter

def inside_frac(b, g):
    ix = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
    iy = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
    area = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / area


_BASE = {}
_CUR = {"ign": []}


def _keep(b):
    return not any(inside_frac(b[:4], g) >= 0.5 for g in _CUR["ign"])


def _ign_read_job(job):
    fr, boxes, pad = job
    _CUR["ign"] = fr.get("ignore") or []
    _CUR["all"] = (fr.get("key"), boxes)        # the unfiltered boxes, for a reading stage that cuts lines first
    return _BASE["read"]((fr, [b for b in boxes if _keep(b)], pad))


def _ign_meiki_job(fr):
    _CUR["ign"] = fr.get("ignore") or []
    return _BASE["meiki_job"](fr)


def _ign_meiki_ocr(m, bgr, *args, **kw):
    return [r for r in _BASE["meiki_ocr"](m, bgr, *args, **kw) if _keep(list(r[0]))]


def install_ignore(Dm):
    """Every pipeline's boxes lose the boxes at least half inside one of the frame's ignore areas, for frames that
    carry them (fr["ignore"]: the dense sets); other frames as before (pc1b.py's install_ignore)."""
    if _BASE:
        return
    _BASE.update(read=Dm._read_job, meiki_job=Dm._meiki_job, meiki_ocr=Dm.meiki_ocr)
    Dm._read_job, Dm._meiki_job, Dm.meiki_ocr = _ign_read_job, _ign_meiki_job, _ign_meiki_ocr


def resample_dense(fr, fit):
    """eval/desktop.resample_frame for a dense frame: its units as there, and its ignore areas scaled by the same
    factors (desktop.py's resample carries fr["ignore"] unscaled; protocol v2 amendment 1's l544 dense frames)."""
    out = D.resample_frame(fr, fit=fit, tag=f"fit{fit[0]}x{fit[1]}")
    sx, sy = out["W"] / fr["W"], out["H"] / fr["H"]
    if fr.get("ignore"):
        out["ignore"] = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in fr["ignore"]]
    return out


def cmd_resample(a):
    import pickle
    frames, meta = D.load_frames(a.frames)
    fit = tuple(int(v) for v in a.fit.split("x"))
    out = [resample_dense(fr, fit) for fr in frames]
    meta = dict(meta, resampled_fit=a.fit, source_frames_sha256=D.sha256_file(a.frames), ignore_scaled=True)
    with open(a.out, "wb") as f:
        pickle.dump({"frames": out, "meta": meta}, f)
    print(json.dumps({"frames": len(out), "fit": a.fit, "ignore_areas": sum(len(f.get("ignore") or []) for f in out)}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest-out")
    p = sub.add_parser("resample", help="fit a dense frames file into WxH, its ignore areas scaled too")
    p.add_argument("--frames", required=True)
    p.add_argument("--fit", default="960x544")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"build": cmd_build, "resample": cmd_resample}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): native-PSP scoring of a recognizer blob in the shipped form, through the C runtime.

  modelb_eval.py dev-plan --out DEV.pkl
  modelb_eval.py hval-plan --pools POOLS --pagepool PAGEPOOL --pages PAGES --out HVAL.pkl
  modelb_eval.py read --plan PLAN.pkl --blob B.vocr [--mode w8f] [--head-gate on] [--two-level] --out ROWS.json
  modelb_eval.py compare --cand ROWS.json [ROWS.json ...] --base ROWS.json [...] --out CMP.json

Everything is native 480x272 PSP (never the Adrenaline 2x view): the PSP itself reads its own frame.

Plans hold every crop a score reads, cut once (the line finder does not depend on the recognizer):
  dev   the development sample's native PSP rows (eval/manifests/reserved.json dev_sample VNs, which no model of this
        round trains on): tv = teacher_val_dev's PSP lines, dn = devnative's four development files' PSP lines,
        both in two views (rv: as stored, dark text inverted as the console hands it over; rvn: screen polarity),
        as VOCR5's development score reads them; wd = the whole-dialogue PSP regions of VOCR5's cohorts (tv_dlg,
        and the dn_dlg frames tv_dlg does not hold) at base, +4 and +8 px, through the C line finder at native size
        in two paths: `grad` (VOCR_LINES_GRAD + vocr_refine_lines(VOCR_REFINE_DEFAULT), model B's pre-registered
        page path, #126) and `recovery` (VOCR_LINES_TEXTBOX_RECOVERY + the refinement, Rubi's finder; report only)
  hval  the hooked corpus' PSP screening block (vocr6-ingest's pools and page pool): hl = every hval PSP line and
        name as a clean crop (box + pad, console view, as stored), hp = every hval PSP page's whole-dialogue regions
        (the page pool's regions: verified horizontal dialogue with every other renderer box kept out, on pages with
        no label in doubt) through the same finder paths; a region's reference is its lines' verified text
Rows: per fixed line, errors = Levenshtein(norm(pred), norm(ref)), chars = len(norm(ref)); per region,
train/general_vn_textbox_guard.transcript_score's whole-transcript alignment (every finder line read with its dark
flag as VOCR_INVERT). Rows carry their cluster (dev: the VNDB id; hval: the title id) and frame counts (the time
model's input: frames, frames the head gate skipped, frames the two-level escape fired on). ROWS.json holds no text.

compare: CAND - BASE in CER points per component, CAND the mean over its rows files (seeds) of each unit's errors,
with a paired VN-cluster bootstrap (eval/vocr3_report.cluster_bootstrap: 10,000 resamples, seed 0) and a unit
bootstrap. Components: wd_grad (the primary), wd_recovery, fixed (tv + dn dialogue, both views), tv, dn, names,
hl (hval lines, dialogue), hp_grad, hp_recovery.
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in ("eval", "train", "runtime", "synth"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
DEV = {
    "tv": HERD / "shared/artifacts/owner-drive-ingest-2/reclaim/pools/teacher_val_dev.npz",
    "dn": [HERD / "shared/artifacts/owner-drive-ingest-2/reclaim/pools" / f for f in (
        "devnative_tv_native_ink_dev.npz", "devnative_tv_native_grad_dev.npz", "devnative_strict_ink_dev.npz",
        "devnative_strict_grad_dev.npz")],
    "wd": [HERD / "shared/artifacts/vocr5-train/cohorts/tv_dlg.pkl",
           HERD / "shared/artifacts/vocr5-train/cohorts/dn_dlg.pkl"],
}
DEV_SHA256 = {   # eval/vocr5_protocol.json cohorts
    "teacher_val_dev.npz": "e2f0300fd9f417b6c3c01e1dab36641bfd72749e2cac5db896226ef0d29b65dd",
    "devnative_tv_native_ink_dev.npz": "c198c9ea147620e2f75b929dbb676eda30cd10137b2b3c14eae17f28204bc0c4",
    "devnative_tv_native_grad_dev.npz": "4bda0139ae52f6d472513e96d2f79d30cc25be96dbc2d2316b3bce22f5e4702a",
    "devnative_strict_ink_dev.npz": "ad4207cdfc6f3e667ffe776a80ac3f97fe58fc3e27dab6285a621e35e24c7645",
    "devnative_strict_grad_dev.npz": "5e5b147254f9e65ccd9d904d79db68224b8fd26e902a0b167dff005b725891a1",
    "tv_dlg.pkl": "f1007ab4e943012330b229cc2630f3fc9d33aa94d6dac03130bc5ea5126c1af3",
    "dn_dlg.pkl": "c6dfd35293e7716bc3772fffc88c8d8aeba62c8fa20f97c0d5041b4f1dbc1e1f",
}
FINDERS = {"grad": 0, "recovery": 3}          # vocr_lines.h VOCR_LINES_GRAD, VOCR_LINES_TEXTBOX_RECOVERY
REFINE = 0x1F                                 # VOCR_REFINE_DEFAULT
VARIANTS = ("base", "expanded-4", "expanded-8")
PUNCT = set("「」『』。、！？!?…―（）")
BOOT = {"reps": 10000, "seed": 0, "min_clusters": 5}
COMPONENTS = ("wd_grad", "wd_recovery", "fixed", "tv", "dn", "names", "hl", "hl_names", "hp_grad", "hp_recovery")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def namelike(label):
    """eval/synth3_ablation.strata's name-like tag: VOCR5's dialogue rule for teacher_val."""
    t = norm(label)
    return len(t) <= 6 and not set(t) & PUNCT


def load_npz(path):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(labels)
    crops = [data[offs[i]:offs[i + 1]].reshape(int(shapes[i][0]), int(shapes[i][1])) for i in range(len(labels))]
    return crops, labels, meta


def check(path):
    want = DEV_SHA256.get(Path(path).name)
    if want and sha256(path) != want:
        raise SystemExit(f"{path}: sha256 differs from the pinned development cohort")


# ---------------------------------------------------------------- the C line finder (host build, as on the PSP)

FINDER_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror")


def build_finder(out_dir):
    drv = Path(out_dir) / "vocr_lines_refine_test"
    drv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcc", *FINDER_FLAGS, "-o", str(drv), "runtime/vocr_lines.c", "runtime/vocr_lines_refine.c",
                    "runtime/vocr_lines_refine_test.c"], check=True, cwd=ROOT)
    return drv


def c_find(driver, images, finder, flags=REFINE):
    """[[(x, y, w, h, dark)] per grey image]: the C finder, then vocr_refine_lines(flags)."""
    if not images:
        return []
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img in images:
            img = np.ascontiguousarray(img, np.uint8)
            if img.ndim != 2:
                raise ValueError("grey regions only")
            h, w = img.shape
            f.write(struct.pack("<6I", w, h, 1, finder, w, flags))
            f.write(img.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(driver), path], check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(out) != len(images):
        raise RuntimeError(f"finder driver: {len(out)} results for {len(images)} regions")
    res = []
    for ln in out:
        v = [int(t) for t in ln.split()]
        if v[0]:
            raise RuntimeError(f"finder rc {v[0]}")
        res.append([tuple(v[5 + 5 * i:10 + 5 * i]) for i in range(v[2])])
    return res


def region_rows(comp, regions, driver):
    """Plan rows of whole-dialogue regions: per finder path, the region's finder crops and its reference lines."""
    rows = []
    imgs = [r["image"] for r in regions]
    for path, finder in FINDERS.items():
        for r, boxes in zip(regions, c_find(driver, imgs, finder)):
            img = r["image"]
            crops = [(np.ascontiguousarray(img[y:y + h, x:x + w]), bool(d)) for x, y, w, h, d in boxes]
            rows.append({"comp": f"{comp}_{path}", "unit": f"{r['unit']}|{r['variant']}", "vn": r["vn"],
                         "variant": r["variant"], "crops": crops, "lines": r["lines"], "name": False})
    return rows


def fixed_rows(comp, crops, labels, meta, ids, vn_of, name_of, views=("rv", "rvn")):
    rows = []
    for i in ids:
        for view in views:
            inv = view == "rvn" and bool(meta[i].get("inverted"))
            g = 255 - crops[i] if inv else crops[i]
            rows.append({"comp": comp, "unit": f"{comp}|{i}|{meta[i].get('id', i)}|{view}", "vn": vn_of(i),
                         "view": view, "crops": [(np.ascontiguousarray(g), False)], "ref": labels[i],
                         "name": bool(name_of(i))})
    return rows


# ---------------------------------------------------------------- plans

def dev_plan(args):
    driver = build_finder(Path(args.out).parent / "build")
    rows = []
    check(DEV["tv"])
    crops, labels, meta = load_npz(DEV["tv"])
    ids = [i for i, m in enumerate(meta) if m.get("kind") == "psp" and labels[i]]
    rows += fixed_rows("tv", crops, labels, meta, ids, lambda i: meta[i]["vn"], lambda i: namelike(labels[i]))
    for f in DEV["dn"]:
        check(f)
        c, lab, mt = load_npz(f)
        ids = [i for i, m in enumerate(mt) if m.get("kind") == "psp" and lab[i]]
        rows += [dict(r, unit=f"{Path(f).stem}|{r['unit']}") for r in
                 fixed_rows("dn", c, lab, mt, ids, lambda i, mt=mt: mt[i]["vn"],
                            lambda i, mt=mt: "name" in (mt[i].get("styles") or []))]
    rows += region_rows("wd", dev_regions(), driver)
    plan = {"kind": "modelb-plan", "name": "dev", "built": now(), "rows": rows,
            "inputs": {Path(p).name: sha256(p) for p in [DEV["tv"], *DEV["dn"], *DEV["wd"]]},
            "finder_driver_sha256": sha256(driver), "finders": FINDERS, "refine": REFINE}
    with open(args.out, "wb") as f:
        pickle.dump(plan, f, protocol=4)
    print(summary(plan))


def dev_regions():
    """The development sample's native PSP whole-dialogue regions: VOCR5's cohorts tv_dlg, and the dn_dlg frames
    tv_dlg does not hold (finder-tighten's pooling), every variant: [{unit, vn, variant, image (grey), lines}]."""
    regions = []
    for f in DEV["wd"]:
        check(f)
    with open(DEV["wd"][0], "rb") as fh:
        tv = pickle.load(fh)
    with open(DEV["wd"][1], "rb") as fh:
        dn = pickle.load(fh)
    tv_shots = {u["shot"] for u in tv}
    for name, cases in (("tv_dlg", tv), ("dn_dlg", dn)):
        for u in cases:
            if u["kind"] != "psp" or (name == "dn_dlg" and u["shot"] in tv_shots):
                continue
            regions.append({"unit": f"{name}|{u['unit']}", "vn": u["vn"], "variant": u["variant"],
                            "image": u["image"], "lines": [{"text": ln["text"], "styles": ln.get("styles", [])}
                                                           for ln in u["lines"]]})
    return regions


def hval_regions(pagepool, pages):
    """The hooked corpus' hval PSP whole-dialogue regions (the page pool's regions of pages with no label in doubt),
    every variant, cut from the page in grey: [{unit, vn (the title id), variant, image, lines}]."""
    import cv2

    from common import to_gray
    regions = []
    for f in sorted(Path(pagepool).glob("*.jsonl")):
        for ln in open(f, encoding="utf-8"):
            p = json.loads(ln)
            if p.get("platform") != "psp" or p.get("split") != "hval" or p.get("untrusted"):
                continue
            text_of = {x["id"]: x["text"] for x in p["lines"]}
            img = None
            for r in p.get("regions", []):
                if r["variant"] not in VARIANTS or not r["lines"] or any(i not in text_of for i in r["lines"]):
                    continue
                if img is None:
                    path = Path(pages) / p["image"]
                    if sha256(path) != p["image_sha256"]:
                        raise SystemExit(f"{path}: sha256 differs from the page record")
                    img = to_gray(cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                x0, y0, x1, y1 = r["region"]
                regions.append({"unit": f"{p['title_id']}:{p['page']}", "vn": p["title_id"], "variant": r["variant"],
                                "image": np.ascontiguousarray(img[y0:y1, x0:x1]),
                                "lines": [{"text": text_of[i], "styles": []} for i in r["lines"]]})
    return regions


def hval_plan(args):
    """The PSP hval rows from vocr6-ingest's pools (POOLS/hval/{line,name}/<title>.npz) and page pool."""
    driver = build_finder(Path(args.out).parent / "build")
    rows = []
    pools = Path(args.pools)
    for pool, is_name in (("line", False), ("name", True)):
        for f in sorted((pools / "hval" / pool).glob("*.npz")):
            crops, labels, meta = load_npz(f)
            ids = [i for i, m in enumerate(meta) if m.get("platform") == "psp" and labels[i]]
            title = f.stem
            rows += [dict(r, comp="hl", unit=f"hl|{pool}|{r['unit']}") for r in
                     fixed_rows("hl", crops, labels, meta, ids, lambda i, t=title: t, lambda i, n=is_name: n,
                                views=("rv",))]
    regions = hval_regions(args.pagepool, args.pages)
    rows += [dict(r, comp=r["comp"].replace("wd_", "hp_")) for r in region_rows("wd", regions, driver)]
    plan = {"kind": "modelb-plan", "name": "hval", "built": now(), "rows": rows,
            "inputs": {"pools": str(pools), "pagepool": str(args.pagepool),
                       "pool_files": {str(f.relative_to(pools)): sha256(f)
                                      for f in sorted(pools.glob("hval/*/*.npz"))}},
            "finder_driver_sha256": sha256(driver), "finders": FINDERS, "refine": REFINE}
    with open(args.out, "wb") as f:
        pickle.dump(plan, f, protocol=4)
    print(summary(plan))


def summary(plan):
    from collections import Counter
    c = Counter(r["comp"] for r in plan["rows"])
    v = {k: len({r["vn"] for r in plan["rows"] if r["comp"] == k}) for k in c}
    return json.dumps({"plan": plan["name"], "rows": dict(c), "clusters": v})


# ---------------------------------------------------------------- reading and scoring

def read(args):
    from pyvocr import Vocr

    from general_vn_textbox_guard import transcript_score
    with open(args.plan, "rb") as f:
        plan = pickle.load(f)
    rt = Vocr(args.blob, mode=args.mode, max_width=1024, head_gate=args.head_gate, two_level=int(args.two_level))
    out = []
    for r in plan["rows"]:
        preds, frames, gated, escaped, blank = [], 0, 0, 0, 0
        for g, inv in r["crops"]:
            preds.append(norm(rt.text(g, invert=inv)))
            frames += len(rt.last_best)
            blank += int((rt.last_best == 0).sum())
            gated += rt.last_gated
            escaped += rt.last_escaped
        row = {"comp": r["comp"], "unit": r["unit"], "vn": r["vn"], "name": r["name"], "frames": frames,
               "blank": blank, "gated": gated, "escaped": escaped, "lines_out": len(r["crops"])}
        if "ref" in r:
            ref = norm(r["ref"])
            row.update(errors=lev(preds[0], ref), chars=len(ref))
        else:
            sc = transcript_score(r["lines"], preds)["all"]["full"]
            row.update(errors=sc["errors"], chars=sc["chars"])
        out.append(row)
    res = {"kind": "modelb-rows", "plan": plan["name"], "plan_built": plan["built"], "blob": str(args.blob),
           "blob_sha256": sha256(args.blob), "blob_bytes": rt.blob_bytes, "arena_bytes": rt.arena_bytes,
           "mode": args.mode, "head_gate": args.head_gate, "two_level": bool(args.two_level), "read": now(),
           "rows": out, "totals": totals(out)}
    Path(args.out).write_text(json.dumps(res) + "\n")
    print(json.dumps({"blob": Path(args.blob).name, "mode": args.mode, **res["totals"]}))


def component(rows, comp):
    """The rows of a component (dialogue only unless the component is a names one)."""
    if comp == "fixed":
        return [r for r in rows if r["comp"] in ("tv", "dn") and not r["name"]]
    if comp == "names":
        return [r for r in rows if r["comp"] in ("tv", "dn") and r["name"]]
    if comp == "hl_names":
        return [r for r in rows if r["comp"] == "hl" and r["name"]]
    if comp in ("tv", "dn", "hl"):
        return [r for r in rows if r["comp"] == comp and not r["name"]]
    return [r for r in rows if r["comp"] == comp]


def totals(rows):
    out = {}
    for comp in COMPONENTS:
        sel = component(rows, comp)
        if not sel:
            continue
        e, n = sum(r["errors"] for r in sel), sum(r["chars"] for r in sel)
        f, g, x = sum(r["frames"] for r in sel), sum(r["gated"] for r in sel), sum(r["escaped"] for r in sel)
        bl = sum(r.get("blank", 0) for r in sel)
        out[comp] = {"cer": round(100 * e / n, 4) if n else None, "errors": e, "chars": n, "units": len(sel),
                     "clusters": len({r["vn"] for r in sel}), "frames": f,
                     "blank_share": round(bl / f, 4) if f else None,
                     "gated_share": round(g / f, 4) if f else None,
                     "escaped_share_of_kept": round(x / (f - g), 4) if f > g else None}
    return out


def compare(args):
    import vocr3_report as R
    cands = [json.loads(Path(p).read_text()) for p in args.cand]
    bases = [json.loads(Path(p).read_text()) for p in args.base]
    out = {"kind": "modelb-compare", "cand": [c["blob_sha256"] for c in cands],
           "base": [b["blob_sha256"] for b in bases], "cand_files": args.cand, "base_files": args.base,
           "boot": BOOT, "components": {}}
    for comp in COMPONENTS:
        cs = [{r["unit"]: r for r in component(c["rows"], comp)} for c in cands]
        bs = [{r["unit"]: r for r in component(b["rows"], comp)} for b in bases]
        if not cs[0]:
            continue
        units = sorted(cs[0])
        if any(set(c) != set(units) for c in cs + bs):
            raise SystemExit(f"{comp}: the rows files hold different units")
        a = {u: (float(np.mean([c[u]["errors"] for c in cs])), cs[0][u]["chars"]) for u in units}
        b = {u: (float(np.mean([x[u]["errors"] for x in bs])), bs[0][u]["chars"]) for u in units}
        vn = {u: cs[0][u]["vn"] for u in units}
        lines = R.paired_bootstrap(a, b, BOOT["reps"], BOOT["seed"])
        clus = R.cluster_bootstrap(a, b, vn, BOOT["reps"], BOOT["seed"], BOOT["min_clusters"])
        n = sum(v[1] for v in a.values())
        out["components"][comp] = {
            "cand_cer": round(100 * sum(v[0] for v in a.values()) / n, 4) if n else None,
            "base_cer": round(100 * sum(v[0] for v in b.values()) / n, 4) if n else None,
            "delta": lines["delta"], "units_ci95": lines["ci95"], "vn_ci95": clus.get("ci95"),
            "units": lines["units"], "clusters": clus["clusters"], "chars": lines["chars"],
            "per_seed_delta": [round(100 * (sum(c[u]["errors"] for u in units) - sum(v[0] for v in b.values())) / n, 4)
                               for c in cs] if n else None}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["components"].items():
        print(f"{k:12s} cand {v['cand_cer']} base {v['base_cer']} delta {v['delta']} vn_ci {v['vn_ci95']} "
              f"({v['units']} units, {v['clusters']} clusters) seeds {v['per_seed_delta']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev-plan")
    d.add_argument("--out", required=True)
    h = sub.add_parser("hval-plan")
    h.add_argument("--pools", required=True)
    h.add_argument("--pagepool", required=True)
    h.add_argument("--pages", required=True)
    h.add_argument("--out", required=True)
    r = sub.add_parser("read")
    r.add_argument("--plan", required=True)
    r.add_argument("--blob", required=True)
    r.add_argument("--mode", default="w8f", choices=("f32", "w8f", "w8i"))
    r.add_argument("--head-gate", default="off", choices=("off", "on"))
    r.add_argument("--two-level", action="store_true")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--cand", nargs="+", required=True)
    c.add_argument("--base", nargs="+", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"dev-plan": dev_plan, "hval-plan": hval_plan, "read": read, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()

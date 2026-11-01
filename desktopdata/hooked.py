#!/usr/bin/env python3
"""1a: the hooked pools v2.1 train pages re-rendered as desktop captures (docs/desktop-data.md).

  hooked.py plan     --labels-manifest M --labels-shards D [--n 20000] [--seed 20261101] --out PLAN.json
  hooked.py build    --plan PLAN.json --pool P --pages DIR --labels-shards D --out SHARDS [--part i/n] [--workers 5]
  hooked.py manifest --plan PLAN.json --shards SHARDS --out data/manifests/desktop_data/hooked_desktop_v1.json

Input: the train pages of vocr-desktop's hooked page shards (data/manifests/desktop_pages_v21.json on
feat/vocr-desktop: synth/textspot6_pages.py on b2-ingest2's page pool v2.1 with MeikiOCR's boxes for the text the
hook does not cover). Their labels are reused as they are, in the page's native pixels: positives (dialogue and name
lines, refined to ink; MeikiOCR's other text as class notdlg) and ignore areas (ruby, dropped lines, renderer boxes no
verified line holds, weak MeikiOCR boxes). Only split train is read: hval, guard and the test titles stay out exactly
as in pools v2.1 (`plan` refuses any other split; the test titles have no train pages).

The plan: N pages of the train split, spread over the titles as HookPool samples them (title weight pages^0.5, a
title never more than its pages), the pages of a title drawn uniformly (seeded). The build: each page's colour PNG
(its sha256 checked against the shard's meta) through one capture of desktopdata/capture.py (seeded by title and
page), to grey; every box through the capture's exact map. Output shards are textspot6-pages shards (HookPool reads
them unchanged): SHARDS/train/<TID>_NNN.{npy,json}, at most 200 pages each, kind "desktop" (so the trainers' PSP
Adrenaline 2x step never applies), meta.capture the capture's settings, meta.src_kind the page's platform.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktopdata import capture as C  # noqa: E402
from desktopdata import shards as SH  # noqa: E402

KIND = "desktop"
SHARD_PAGES = 200
TITLE_TEMP = 0.5
CODE = ["desktopdata/hooked.py", "desktopdata/capture.py", "desktopdata/shards.py"]


def load_label_shards(manifest, root, split="train"):
    """{title: [(shard name, sample)]} of a textspot6-pages manifest's shards of `split`, each shard's json checked
    against the manifest's sha256. Refuses a sample of another split."""
    doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if doc.get("kind") != "textspot6-pages":
        raise SystemExit(f"{manifest}: not a textspot6-pages manifest")
    out = {}
    for s in doc["shards"]:
        if s["split"] != split:
            continue
        js = Path(root) / s["dir"] / f"{s['name']}.json"
        if SH.sha256_file(js) != s["json_sha256"]:
            raise SystemExit(f"{js}: sha256 is not {manifest}'s")
        for smp in json.loads(js.read_text(encoding="utf-8"))["samples"]:
            if smp["split"] != split:
                raise SystemExit(f"{js}: a {smp['split']} page in a {split} shard")
            out.setdefault(smp["title"], []).append((s["name"], smp))
    return out, doc


def allocate(sizes, n, temp=TITLE_TEMP):
    """{title: pages to take}: n split over titles by weight size^temp, capped at each title's size, the cap's
    leftover handed on to the other titles (largest remainders first, ties by title)."""
    left = dict(sizes)
    out = {t: 0 for t in sizes}
    need = min(n, sum(sizes.values()))
    while need > 0 and left:
        w = {t: left[t] ** temp for t in left}
        tot = sum(w.values())
        share = {t: need * w[t] / tot for t in left}
        give = {t: min(left[t], int(math.floor(share[t]))) for t in left}
        rem = need - sum(give.values())
        for t in sorted(left, key=lambda t: (-(share[t] - math.floor(share[t])), t)):
            if rem <= 0:
                break
            if give[t] < left[t]:
                give[t] += 1
                rem -= 1
        for t, g in give.items():
            out[t] += g
            left[t] -= g
        need -= sum(give.values())
        left = {t: v for t, v in left.items() if v > 0}
        if not sum(give.values()):
            break
    return out


def seed_of(*parts):
    return [zlib.crc32(str(p).encode()) for p in parts]


def cmd_plan(args):
    by_title, doc = load_label_shards(args.labels_manifest, args.labels_shards)
    sizes = {t: len(v) for t, v in by_title.items()}
    alloc = allocate(sizes, args.n)
    pages = {}
    for t in sorted(by_title):
        rng = np.random.default_rng(seed_of(args.seed, t))
        lst = sorted(by_title[t], key=lambda x: x[1]["page"])
        k = alloc[t]
        idx = sorted(rng.choice(len(lst), size=k, replace=False).tolist()) if k else []
        pages[t] = [{"page": lst[i][1]["page"], "shard": lst[i][0]} for i in idx]
    plan = {"kind": "desktop-hooked-plan", "created": SH.utc(), "seed": args.seed, "n": args.n,
            "title_temp": TITLE_TEMP, "labels_manifest": str(args.labels_manifest),
            "labels_manifest_sha256": SH.sha256_file(args.labels_manifest),
            "labels_totals": doc.get("totals"), "titles": len(pages),
            "pages": sum(len(v) for v in pages.values()), "train_pages": sum(sizes.values()),
            "per_title": {t: {"train_pages": sizes[t], "taken": len(pages[t])} for t in sorted(pages)},
            "selection": pages}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(f"plan: {plan['pages']} of {plan['train_pages']} train pages, {len(pages)} titles -> {args.out}", flush=True)


# ----------------------------------------------------------------------------- build

_G = {}


def map_lines(cap, lines):
    out = []
    for ln in lines:
        b = cap.box(ln[:4])
        if b is None:
            continue
        out.append([*b, int(ln[4]), ln[5], float(ln[6]) if len(ln) > 6 else 1.0])
    return out


def map_ignore(cap, ignore):
    return [b for b in (cap.box(x) for x in ignore) if b is not None]


def render_page(job):
    """(sample, grey) of one page: its capture and mapped labels."""
    import cv2
    tid, smp, path, seed = job
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != smp["meta"]["image_sha256"]:
        raise SystemExit(f"{path}: sha256 differs from the shard's meta")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    if (w, h) != (smp["w"], smp["h"]):
        raise SystemExit(f"{path}: {w}x{h}, the shard says {smp['w']}x{smp['h']}")
    rng = np.random.default_rng(seed_of(seed, tid, smp["page"]))
    cap = C.sample(rng, w, h, smp["kind"])
    out = cap.apply(rgb)
    grey = SH.to_gray(out)
    lines = map_lines(cap, smp["lines"])
    ignore = map_ignore(cap, smp["ignore"])
    s = {"src": "hook", "kind": KIND, "vn": smp.get("vn"), "shot": smp["shot"], "title": tid, "page": smp["page"],
         "split": "train", "lines": lines, "ignore": ignore,
         "meta": {"image_sha256": smp["meta"]["image_sha256"], "src_kind": smp["kind"],
                  "src_dims": [w, h], "capture": cap.record(), "src_lines": len(smp["lines"]),
                  "src_ignore": len(smp["ignore"]), "counts": smp["meta"].get("counts"),
                  "roi": smp["meta"].get("roi")}}
    if len(lines) != len(smp["lines"]):
        s["meta"]["lines_outside"] = len(smp["lines"]) - len(lines)
    return s, grey


def cmd_build(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    by_title, _ = load_label_shards(plan["labels_manifest"], args.labels_shards)
    images = {}
    for f in sorted(Path(args.pool).glob("*.jsonl")):
        for x in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            images[(r["title_id"], r["page"])] = r["image"]
    titles = sorted(plan["selection"])
    if args.part:
        i, n = (int(v) for v in args.part.split("/"))
        titles = titles[i::n]
    out = Path(args.out) / "train"
    t0 = time.time()
    done = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for tid in titles:
            want = {p["page"] for p in plan["selection"][tid]}
            smps = sorted((s for _, s in by_title[tid] if s["page"] in want), key=lambda s: s["page"])
            if len(smps) != len(want):
                raise SystemExit(f"{tid}: {len(want)} planned pages, {len(smps)} found in the label shards")
            for k in range(0, len(smps), SHARD_PAGES):
                name = f"{tid}_{k // SHARD_PAGES:03d}"
                if (out / f"{name}.json").exists() and not args.force:
                    continue
                jobs = [(tid, s, str(Path(args.pages) / images[(tid, s["page"])]), plan["seed"])
                        for s in smps[k:k + SHARD_PAGES]]
                res = list(pool.imap(render_page, jobs, chunksize=2))
                SH.write_shard(out / name, [r[0] for r in res], [r[1] for r in res])
                done += len(res)
                print(f"  train/{name}: {len(res)} pages ({done} in {time.time() - t0:.0f} s)", flush=True)
    print(f"desktop hooked build: {done} pages in {time.time() - t0:.0f} s", flush=True)


def cmd_manifest(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root = Path(args.shards)
    shards = []
    for js in sorted((root / "train").glob("*.json")):
        rec = SH.shard_record(js.with_suffix(""), "train")
        samples = json.loads(js.read_text(encoding="utf-8"))["samples"]
        rec.update(split="train", title=samples[0]["title"] if samples else None,
                   by_kind={k: sum(1 for s in samples if s["meta"]["src_kind"] == k) for k in ("vita", "psp")})
        shards.append(rec)
    n = sum(r["samples"] for r in shards)
    if n != plan["pages"]:
        raise SystemExit(f"{n} pages in the shards, the plan has {plan['pages']}")
    doc = {"kind": "textspot6-pages", "schema": 1, "issue": 169, "generator": "desktopdata/hooked.py",
           "created": SH.utc(), "desktop": {
               "what": "hooked pools v2.1 train pages re-rendered as desktop captures (1a, docs/desktop-data.md)",
               "kind": KIND, "capture_mix": C.MIX, "plan": {k: plan[k] for k in (
                   "seed", "n", "title_temp", "titles", "pages", "train_pages", "labels_manifest",
                   "labels_manifest_sha256")},
               "plan_sha256": SH.sha256_file(args.plan), "labels_source": args.labels_source,
               "code_sha256": SH.code_sha256(ROOT, CODE)},
           "shards": shards, "totals": SH.totals(shards, "split"),
           "note": "shards hold copyrighted game frames and stay local; this manifest holds names, counts, sha256"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(shards)} shards, {n} pages", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--labels-manifest", required=True)
    p.add_argument("--labels-shards", required=True)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build")
    p.add_argument("--plan", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--labels-shards", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part")
    p.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "5")))
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("manifest")
    p.add_argument("--plan", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--labels-source", default="data/manifests/desktop_pages_v21.json (feat/vocr-desktop 1b4e074)")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import cv2
    cv2.setNumThreads(1)
    {"plan": cmd_plan, "build": cmd_build, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

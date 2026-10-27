#!/usr/bin/env python3
"""textspot7's evaluation (issue #128, docs/textspot7.md, eval/textspot7_protocol.json): the spotter's boxes per
region with their confidences, the fit of F (the per-region fallback to rubi,fix=1) from stored page-path rows, the
--boxes files eval/pagepath.py reads for F, D and F+D, and the C == twin check.

  textspot7.py fit-f --spot SPOT.jsonl --s-rows S.rows.jsonl --g-rows G.rows.jsonl --set hval --out FIT.json
      F's grid (protocol "arms.F.grid", amendment 1) on one set's stored rows: per unit the spotter's region boxes
      (textspot6's det decode, reproduced from SPOT and checked box for box against S's rows) and GRAD's (G's rows);
      per grid point the choice per unit and the errors of the chosen run's row (F shows crops those runs read);
      the fit: the fewest errors, ties to mode S, then the larger tau, the smaller lam (off last), the smaller kappa
      (off last). Counts only in FIT.json
  textspot7.py boxes-f --spot SPOT.jsonl --s-rows S.rows.jsonl --g-rows G.rows.jsonl --set hval|dev
                       --params MODE,TAU,LAM,KAPPA --out BOXES.jsonl
      eval/pagepath.py --boxes rows of F with those parameters: each unit's chosen boxes [[x, y, w, h, dark]] in
      region pixels; BOXES.choice.json beside it (per unit S or G, and the counts)
  textspot7.py boxes-d --spot SPOT.jsonl --set hval|dev --steps assign,join5,join10,extend,tighten --out BOXES.jsonl
      D's boxes (synth/textspot7_twin.post) on each unit's region in the page path's own pixels (dev: the unit's
      grey region, a PSP one upscaled 2x bilinear; hval: the page in the console view, cut)
  textspot7.py parity --spot SPOT.jsonl --set hval|dev [--steps ...] [--s-rows ... --g-rows ... --params ...]
                      --out PARITY.json
      runtime/vocr_textspot7.c (built with gcc -O2) against the twin on every unit: D's boxes, F's choice

Units: dev = #119's units ($A/vocr5-analysis/work/units.pkl; uid dev:<i>:<unit>:<variant>), hval = the hval units
file ($A/vocr6-spotter/eval/units/hval_v1.jsonl). Everything here reads private frames; the outputs other than
FIT.json and PARITY.json hold boxes of private frames and stay local.
"""
import argparse
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot7_twin as T7  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
DEV_UNITS = ART / "vocr5-analysis" / "work" / "units.pkl"
HVAL_UNITS = ART / "vocr6-spotter" / "eval" / "units" / "hval_v1.jsonl"
PAGES = ART / "vocr6-ingest" / "pages"
PRIV = HERD / "shared" / "private" / "vocr6-spotter3" / "confirm"
CONFIRM_UNITS = Path(os.environ.get("T7_CONFIRM_UNITS", str(PRIV / "units_v1.pkl")))
CONFIRM_SHOTS = PRIV / "shots"
TAUS = (0, 8, 16, 24, 32, 48, 64, 80, 96)
LAMS = (600, 750, 900, 0)
KAPPAS = (500, 1000, 0)
STEP_BITS = {"assign": 1, "join5": 2, "join10": 4, "extend": 8, "tighten": 16}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def grid():
    """F's grid in the fit's tie order: mode S first, then the larger tau, the smaller lam (off last), the smaller
    kappa (off last). Mode G ignores lam and kappa (one point per tau)."""
    out = []
    for tau in sorted(TAUS, reverse=True):
        for lam in LAMS:
            for kappa in KAPPAS:
                out.append(T7.Params("S", tau, lam, kappa))
    for tau in sorted(TAUS, reverse=True):
        out.append(T7.Params("G", tau, 0, 0))
    return out


def parse_params(text):
    m, t, lam, k = text.split(",")
    return T7.Params(m, int(t), int(lam), int(k))


# ----------------------------------------------------------------------------- units and their frames

def dev_frame_key(u):
    return f"{u['shot']}|{u['src_kind']}|{round(float(u['scale']), 6)}|{u['kind']}"


def units_of(set_name):
    """[{uid, cluster, platform, region (frame px, console view), frame key, s}] in the units file's order."""
    out = []
    if set_name == "dev":
        for i, u in enumerate(pickle.load(open(DEV_UNITS, "rb"))["units"]):
            s = 2 if u["kind"] == "psp" else 1
            out.append({"uid": f"dev:{i}:{u['unit']}:{u['variant']}", "cluster": u["vn"], "platform": u["kind"],
                        "region": [v * s for v in u["region"]], "key": dev_frame_key(u), "s": s, "i": i})
        return out
    if set_name == "confirm":
        for i, u in enumerate(pickle.load(open(CONFIRM_UNITS, "rb"))["units"]):
            s = 2 if u["kind"] == "psp" else 1
            out.append({"uid": f"dev:{i}:{u['unit']}:{u['variant']}", "cluster": u["vn"], "platform": u["kind"],
                        "region": [v * s for v in u["region"]], "key": u["shot"], "s": s, "i": i, "confirm": True})
        return out
    if set_name != "hval":
        raise SystemExit(f"unknown set {set_name}")
    for r in read_jsonl(HVAL_UNITS)[1:]:
        s = 2 if r["platform"] == "psp" else 1
        view = "psp2x" if s == 2 else "vita"
        out.append({"uid": r["uid"], "cluster": r["cluster"], "platform": r["platform"],
                    "region": [v * s for v in r["region"]], "key": (r["title"], r["page"], view), "s": s,
                    "title": r["title"], "page": r["page"], "image_sha256": r["image_sha256"]})
    return out


def spot_rows(path, set_name):
    rows = read_jsonl(path)
    meta, rows = rows[0]["meta"], rows[1:]
    if set_name in ("dev", "confirm"):
        return meta, {r["frame"]: r for r in rows}
    return meta, {(r["title"], r["page"], r["view"]): r for r in rows}


def frame_boxes_of(r):
    """The det decode's horizontal boxes of a spot row as (x, y, w, h, conf, dark), in decode order."""
    return [(b[0], b[1], b[2], b[3], b[5], b[6]) for b in r["boxes"] if not b[4]]


def spotter_boxes(units, spot):
    """Per uid the spotter's region boxes (x, y, w, h, conf, dark) by textspot6's centre rule."""
    out = {}
    for u in units:
        r = spot[u["key"]]
        out[u["uid"]] = T7.assign(frame_boxes_of(r), u["region"], overlap=False)
    return out


def load_rows(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        r.pop("texts", None)
        out[r["uid"]] = r
    return out


def check_reproduction(sbox, srows):
    bad = [u for u, bx in sbox.items() if [list(b[:4]) for b in bx] != [list(b) for b in srows[u]["boxes_xywh"]]]
    if bad:
        raise SystemExit(f"{len(bad)} units' spotter boxes differ from the stored rows' (first {bad[:3]})")


# ----------------------------------------------------------------------------- F

def f_choices(units, sbox, grows, p):
    return {u["uid"]: T7.choose(sbox[u["uid"]], grows[u["uid"]]["boxes_xywh"], p) for u in units}


def cmd_fit_f(a):
    units = units_of(a.set)
    _, spot = spot_rows(a.spot, a.set)
    srows, grows = load_rows(a.s_rows), load_rows(a.g_rows)
    if a.s_boxes:                                   # F+D: the spotter's boxes are D's (eval/textspot7.py boxes-d)
        sbox = {r["unit"]: [tuple(b) for b in r["boxes6"]] for r in read_jsonl(a.s_boxes)}
    else:
        sbox = spotter_boxes(units, spot)
    check_reproduction(sbox, srows)
    chars = sum(srows[u["uid"]]["chars"] for u in units)
    se = sum(srows[u["uid"]]["errors"] for u in units)
    ge = sum(grows[u["uid"]]["errors"] for u in units)
    points = []
    for p in grid():
        ch = f_choices(units, sbox, grows, p)
        err = sum((srows if ch[u["uid"]] == "S" else grows)[u["uid"]]["errors"] for u in units)
        points.append({"params": p.name(), "mode": p.mode, "tau": p.tau, "lam": p.lam, "kappa": p.kappa,
                       "errors": err, "grad_units": sum(1 for v in ch.values() if v == "G")})
    best = min(range(len(points)), key=lambda i: (points[i]["errors"], i))     # grid() is in the tie order
    doc = {"kind": "textspot7-fit-f", "set": a.set, "units": len(units), "chars": chars, "spotter_errors": se,
           "grad_errors": ge, "oracle_errors": sum(min(srows[u["uid"]]["errors"], grows[u["uid"]]["errors"])
                                                   for u in units),
           "s_boxes": a.s_boxes, "spot_sha256": sha256_file(a.spot), "s_rows_sha256": sha256_file(a.s_rows),
           "g_rows_sha256": sha256_file(a.g_rows), "grid": points, "fit": points[best],
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    f = points[best]
    print(f"fit-f {a.set}: spotter {se}, grad {ge}, oracle {doc['oracle_errors']} of {chars}; fit {f['params']} "
          f"-> {f['errors']} errors ({100 * f['errors'] / chars:.3f}%), grad on {f['grad_units']} units")


def write_boxes(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def cmd_boxes_f(a):
    units = units_of(a.set)
    _, spot = spot_rows(a.spot, a.set)
    if a.s_boxes:
        sbox = {r["unit"]: [tuple(b) for b in r["boxes6"]] for r in read_jsonl(a.s_boxes)}
    else:
        sbox = spotter_boxes(units, spot)
    srows, grows = load_rows(a.s_rows), load_rows(a.g_rows)
    if not a.s_boxes:
        check_reproduction(sbox, srows)
    p = parse_params(a.params)
    ch = f_choices(units, sbox, grows, p)
    rows = []
    for u in units:
        r = srows[u["uid"]] if ch[u["uid"]] == "S" else grows[u["uid"]]
        rows.append({"unit": u["uid"], "boxes": [[*b, int(d)] for b, d in zip(r["boxes_xywh"], r["dark"])]})
    write_boxes(a.out, rows)
    c = Counter(ch.values())
    Path(str(a.out).replace(".jsonl", ".choice.json")).write_text(json.dumps(
        {"params": p.name(), "set": a.set, "S": c["S"], "G": c["G"], "choice": ch}, indent=0) + "\n",
        encoding="utf-8")
    print(f"boxes-f {a.set} {p.name()}: spotter on {c['S']} units, GRAD on {c['G']} -> {a.out}")


# ----------------------------------------------------------------------------- D

def region_images(units):
    """(uid, grey region in the page path's pixels) per unit, in order."""
    import cv2

    from common import to_gray
    if units and units[0]["uid"].startswith("dev:"):
        pk = pickle.load(open(CONFIRM_UNITS if units[0].get("confirm") else DEV_UNITS, "rb"))["units"]
        for u in units:
            img = pk[u["i"]]["image"]
            if u["s"] == 2:
                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            yield u, np.ascontiguousarray(img)
        return
    cache = {}
    for u in units:
        k = (u["title"], u["page"])
        if k not in cache:
            cache.clear()
            path = PAGES / u["title"] / "pages" / f"{u['page']}.png"
            if sha256_file(path) != u["image_sha256"]:
                raise SystemExit(f"{path}: sha256 differs from the units file's")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            if u["s"] == 2:
                rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cache[k] = to_gray(rgb)
        x0, y0, x1, y1 = u["region"]
        yield u, np.ascontiguousarray(cache[k][y0:y1, x0:x1])


_W = {}


def _post_one(job):
    uid, fb, region, img, steps = job
    return uid, T7.post(fb, region, img, steps)


def d_boxes(units, spot, steps, workers=5):
    import multiprocessing as mp
    need_px = bool(set(steps) & {"extend", "tighten"})
    jobs = []
    for u, img in (region_images(units) if need_px else ((u, None) for u in units)):
        jobs.append((u["uid"], frame_boxes_of(spot[u["key"]]), tuple(u["region"]), img, tuple(steps)))
    if not need_px:
        return {uid: T7.post(fb, rg, None, st) for uid, fb, rg, _, st in jobs}
    with mp.get_context("fork").Pool(workers) as pool:
        return dict(pool.imap(_post_one, jobs, chunksize=16))


def cmd_boxes_d(a):
    units = units_of(a.set)
    _, spot = spot_rows(a.spot, a.set)
    steps = [s for s in a.steps.split(",") if s]
    t0 = time.time()
    bx = d_boxes(units, spot, steps, a.workers)
    rows = [{"unit": u["uid"], "boxes": [[b[0], b[1], b[2], b[3], b[5]] for b in bx[u["uid"]]],
             "boxes6": [list(b) for b in bx[u["uid"]]]} for u in units]
    write_boxes(a.out, rows)
    print(f"boxes-d {a.set} {'+'.join(steps) or 'plain'}: {sum(len(r['boxes']) for r in rows)} boxes on "
          f"{len(rows)} units, {time.time() - t0:.0f} s -> {a.out}")


# ----------------------------------------------------------------------------- the confirmation frames

def confirm_frames():
    """[(key, grey)] of every frame the confirmation units are cut from, in the console view (the dev units'
    convention: the grey frame, a PSP one upscaled 2x bilinear on its grey), each checked: every unit's region of
    the native grey frame equals its stored image."""
    import cv2

    from common import to_gray
    units = pickle.load(open(CONFIRM_UNITS, "rb"))["units"]
    got = {r["id"]: r for r in read_jsonl(CONFIRM_SHOTS / "fetched.jsonl") if r.get("ok")}
    by = {}
    for u in units:
        by.setdefault(u["shot"], []).append(u)
    for fid, us in sorted(by.items()):
        path = CONFIRM_SHOTS / got[fid]["file"]
        if sha256_file(path) != got[fid]["sha256"]:
            raise SystemExit(f"{path}: sha256 changed")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        g = to_gray(np.ascontiguousarray(bgr[:, :, ::-1]))
        for u in us:
            x0, y0, x1, y1 = u["region"]
            if not (g[y0:y1, x0:x1] == u["image"]).all():
                raise SystemExit(f"{u['unit']}: the frame's region is not the unit's image")
        if us[0]["kind"] == "psp":
            g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        yield {"frame": fid, "view": "psp2x" if us[0]["kind"] == "psp" else "vita", "vn": us[0]["vn"]}, \
            np.ascontiguousarray(g)


def cmd_spot_confirm(a):
    import textspot6 as S6E
    jobs = list(confirm_frames())
    out = S6E.write_spot(a.blob, jobs, a.out, "confirm", a.workers)
    print(f"spot-confirm: {len(jobs)} frames -> {out}")


# ----------------------------------------------------------------------------- C == twin

def build_driver(out_dir):
    rt = ROOT / "runtime"
    srcs = [rt / "vocr_textspot7_test.c", rt / "vocr_textspot7.c", rt / "vocr_lines_refine.c", rt / "vocr_lines.c"]
    h = hashlib.sha256()
    for s in srcs + [rt / "vocr_textspot7.h", rt / "vocr_lines_refine.h", rt / "vocr_lines.h"]:
        h.update(s.read_bytes())
    exe = Path(out_dir) / f"vocr_textspot7_test_{h.hexdigest()[:16]}"
    if not exe.exists():
        exe.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcc", "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", f"-I{rt}",
                        "-o", str(exe), *map(str, srcs)], check=True)
    return exe, h.hexdigest()


def i32(*v):
    return struct.pack(f"<{len(v)}i", *[int(x) for x in v])


def rec_choose(S, G, p):
    b = i32(1, 0 if p.mode == "S" else 1, p.tau, p.lam, p.kappa, len(S), len(G))
    for s in S:
        b += i32(*s[:5])
    for g in G:
        b += i32(*g[:4])
    return b


def rec_post(fb, region, img, steps):
    bits = sum(STEP_BITS[s] for s in steps)
    b = i32(2, bits, *region, len(fb))
    for x in fb:
        b += i32(*x[:6])
    if bits & (STEP_BITS["extend"] | STEP_BITS["tighten"]):
        raw = np.ascontiguousarray(img, np.uint8).tobytes()
        b += raw + b"\0" * ((-len(raw)) % 4)
    return b


def run_c(exe, recs):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for r in recs:
            f.write(r)
        path = f.name
    try:
        return subprocess.run([str(exe), path], check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(path)


def parse_post(line):
    v = line.split()
    if v[1].startswith("E"):
        return v[1]
    n = int(v[1])
    xs = list(map(int, v[2:]))
    return [tuple(xs[6 * k:6 * k + 6]) for k in range(n)]


def cmd_parity(a):
    units = units_of(a.set)
    _, spot = spot_rows(a.spot, a.set)
    exe, dsha = build_driver(Path(a.work))
    doc = {"kind": "textspot7-parity", "set": a.set, "units": len(units), "driver_sha256": dsha}
    if a.steps is not None:
        steps = [s for s in a.steps.split(",") if s]
        need_px = bool(set(steps) & {"extend", "tighten"})
        twin = d_boxes(units, spot, steps, a.workers)
        recs, uids = [], []
        it = region_images(units) if need_px else ((u, None) for u in units)
        for u, img in it:
            recs.append(rec_post(frame_boxes_of(spot[u["key"]]), u["region"], img, steps))
            uids.append(u["uid"])
        out = []
        for k in range(0, len(recs), 2000):
            out += run_c(exe, recs[k:k + 2000])
        got = {uid: parse_post(line) for uid, line in zip(uids, out)}
        diff = [uid for uid in uids if [tuple(b) for b in twin[uid]] != got[uid]]
        doc["post"] = {"steps": steps, "equal": not diff and len(out) == len(uids), "differ": len(diff),
                       "first": diff[:5]}
    if a.params:
        p = parse_params(a.params)
        if a.s_boxes:
            sbox = {r["unit"]: [tuple(b) for b in r["boxes6"]] for r in read_jsonl(a.s_boxes)}
        else:
            sbox = spotter_boxes(units, spot)
        grows = load_rows(a.g_rows)
        twin = f_choices(units, sbox, grows, p)
        recs = [rec_choose(sbox[u["uid"]], grows[u["uid"]]["boxes_xywh"], p) for u in units]
        out = run_c(exe, recs)
        got = {u["uid"]: ("S" if line == "C 0" else "G" if line == "C 1" else line) for u, line in zip(units, out)}
        diff = [u["uid"] for u in units if twin[u["uid"]] != got[u["uid"]]]
        doc["choose"] = {"params": p.name(), "equal": not diff and len(out) == len(units), "differ": len(diff),
                         "first": diff[:5]}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print("parity", a.set, json.dumps({k: v for k, v in doc.items() if k in ("post", "choose")}))


def result_units(path):
    return {u["uid"]: u for u in json.loads(Path(path).read_text(encoding="utf-8"))["units"]}


def cmd_d_fit(a):
    """D's fit (protocol arms.D.fit): each step alone against the plain det decode, pooled over the representatives'
    hval page-path errors; a step is kept when it lowers them; of join5 / join10 the one with fewer errors (join5 on
    a tie). The plain decode through --boxes must reproduce the stored det rows unit for unit first."""
    runs = a.runs.split(",")
    variants = ("assign", "join5", "join10", "extend", "tighten")
    doc = {"kind": "textspot7-fit-d", "set": "hval", "runs": runs, "per_run": {}, "pooled": {},
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pooled = Counter()
    for r in runs:
        plain = result_units(Path(a.dir) / r / "hval_D_plain.json")
        stored = result_units(Path(a.stored) / r / "hval_det.json")
        bad = [u for u in stored if (plain[u]["errors"], plain[u]["chars"]) != (stored[u]["errors"],
                                                                                  stored[u]["chars"])]
        if set(plain) != set(stored) or bad:
            raise SystemExit(f"{r}: the plain decode does not reproduce the stored det rows ({len(bad)} units)")
        row = {"plain": sum(u["errors"] for u in plain.values()), "chars": sum(u["chars"] for u in plain.values())}
        for v in variants:
            res = result_units(Path(a.dir) / r / f"hval_D_{v}.json")
            row[v] = sum(u["errors"] for u in res.values())
        doc["per_run"][r] = row
        for k in ("plain", *variants):
            pooled[k] += row[k]
    doc["pooled"] = dict(pooled)
    kept = [v for v in ("assign", "extend", "tighten") if pooled[v] < pooled["plain"]]
    joins = [v for v in ("join5", "join10") if pooled[v] < pooled["plain"]]
    if joins:
        kept.append(min(joins, key=lambda v: (pooled[v], v != "join5")))
    order = [v for v in ("assign", "join5", "join10", "extend", "tighten") if v in kept]
    doc["kept"] = order
    doc["D"] = ",".join(order) if order else None
    doc["reproduced_plain"] = True
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"d-fit: pooled {dict(pooled)} -> D = {doc['D']}")


def cmd_a9_batch(a):
    """vocr_textspot7_test.c records for eval/textspot7_a9.py t7: D's post-pass (--steps) and F's choice (--params,
    GRAD's boxes from --g-rows) on every base unit of the set, in one batch file, and the five largest regions' records
    one per file (OUT.one1.bin ...)."""
    units = [u for u in units_of(a.set) if u["uid"].endswith(":base")]
    _, spot = spot_rows(a.spot, a.set)
    steps = [s for s in (a.steps or "").split(",") if s]
    grows = load_rows(a.g_rows) if a.params else None
    p = parse_params(a.params) if a.params else None
    need_px = bool(set(steps) & {"extend", "tighten"})
    recs = []
    for u, img in (region_images(units) if need_px else ((u, None) for u in units)):
        r = rec_post(frame_boxes_of(spot[u["key"]]), u["region"], img, steps)
        if p is not None:
            S = T7.post(frame_boxes_of(spot[u["key"]]), u["region"], img, steps)
            r += rec_choose(S, grows[u["uid"]]["boxes_xywh"], p)
        x0, y0, x1, y1 = u["region"]
        recs.append(((x1 - x0) * (y1 - y0), r))
    Path(a.out).write_bytes(b"".join(r for _, r in recs))
    order = sorted(range(len(recs)), key=lambda i: -recs[i][0])[:5]
    for k, i in enumerate(order, 1):
        Path(f"{a.out}.one{k}.bin").write_bytes(recs[i][1])
    print(f"a9-batch {a.set}: {len(recs)} units ({'post ' + '+'.join(steps) if steps else 'post plain'}"
          f"{', choose ' + p.name() if p else ''}) -> {a.out} (+ 5 largest)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit-f")
    for k in ("spot", "s-rows", "g-rows", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--set", default="hval")
    p.add_argument("--s-boxes", help="D's boxes-d file (F+D)")
    p.set_defaults(fn=cmd_fit_f)
    p = sub.add_parser("boxes-f")
    for k in ("spot", "s-rows", "g-rows", "set", "params", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--s-boxes", help="D's boxes-d file: F over D's boxes (F+D)")
    p.set_defaults(fn=cmd_boxes_f)
    p = sub.add_parser("boxes-d")
    for k in ("spot", "set", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--steps", default="")
    p.add_argument("--workers", type=int, default=5)
    p.set_defaults(fn=cmd_boxes_d)
    p = sub.add_parser("d-fit")
    p.add_argument("--runs", required=True)
    p.add_argument("--dir", default=str(ART / "vocr6-spotter3" / "eval"))
    p.add_argument("--stored", default=str(ART / "vocr6-spotter" / "eval"))
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_d_fit)
    p = sub.add_parser("a9-batch")
    for k in ("spot", "set", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--steps", default="")
    p.add_argument("--params")
    p.add_argument("--g-rows")
    p.set_defaults(fn=cmd_a9_batch)
    p = sub.add_parser("spot-confirm")
    p.add_argument("--blob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.set_defaults(fn=cmd_spot_confirm)
    p = sub.add_parser("parity")
    for k in ("spot", "set", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--steps")
    p.add_argument("--params")
    p.add_argument("--s-boxes")
    p.add_argument("--g-rows")
    p.add_argument("--work", default=str(ART / "vocr6-spotter3" / "eval" / "bin"))
    p.add_argument("--workers", type=int, default=5)
    p.set_defaults(fn=cmd_parity)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

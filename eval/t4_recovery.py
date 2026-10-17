#!/usr/bin/env python3
"""#194 bug 2: GRAD against TEXTBOX_RECOVERY, the overlay owner's finder choice, evaluated offline.

Rubi_for_vita commit 842980e (the owner's branch test4-console-fixes) runs every pinned model with
VOCR_LINES_TEXTBOX_RECOVERY (runtime/vocr_lines.h mode 3, "tbr" here) instead of GRAD (mode 0), so that a short last
line GRAD drops is kept (#194). This scores both modes through the C finder at this checkout (the overlay's pin
d214a05 for every finder file) and the C recognizer in deployment W8I (finder_tighten.Reader: greedy decode), for
every --arm (VOCR5 S/N/L), with the post-finder refinement (runtime/vocr_lines_refine.c, VOCR_REFINE_ALL) off and on.
The overlay's default, ocr_refine_lines = psp, refines PSP games only: the "deployed" view.

  t4_recovery.py dev       --cohorts DIR --boxes DIR --arm NAME=BLOB ... --out DIR
  t4_recovery.py protected --data EVAL_DATA --arm ... --out DIR          once (DIR/protected.attempt.json)
  t4_recovery.py clannad   --work WORK --arm ... --out DIR               once (DIR/clannad.attempt.json)
  t4_recovery.py report    --dir DIR ... --out JSON [--md MD]
  t4_recovery.py sheet     --dir DIR --part dev|protected|clannad-longrun|clannad-console --out DIR
  t4_recovery.py bench     --dir DIR [--reps 20]                         host time of both finders (not the console's)

Data:
  dev        finder-tighten's whole-dialogue development cohorts (tv_dlg, dn_dlg, own_dlg at base / +4 / +8 px), each
             labelled line's ink box rebuilt by finder_tighten's own builders and checked equal to the frozen cohort
             case for case (BOXES/<cohort>.boxes.json). The cohorts keep names and every other detected text out of the
             region. Pooled as finder_tighten pools them: dn_dlg frames that teacher_val holds are not counted twice.
  protected  vita-test and psp-test whole dialogue: eval's regions at base / +4 / +8 (vocr3_report.expand); a region
             with an uncertain line is left out whole. Dialogue = lines without the 'name' style.
  clannad    CLANNAD's long run (owner-drive-ingest's frozen references, eval/clannad_refs.py): every successful page's
             native frame at the long run's ROI (a reference line is the GRAD box its stored crop came from) and at
             test.4's console region 852x135+50+409 (#194; no references there), refinement off (a Vita game).

Measures, per finder and refinement view (no text in any public output):
  lines found      detections; ground-truth lines matched one to one (eval/finders.match: vertical overlap >= 0.5 of
                   the smaller height, overlapping horizontally)
  dialogue CER     transcript_score (train/general_vn_textbox_guard.py) of the region's lines against the detections'
                   readings in finder order; dialogue = its 'body' scope (no name line's characters)
  short last line  the bottom dialogue line of a region with 2+ dialogue lines, at most SHORT (40%) as wide as the
                   region's widest dialogue line; found = matched by a detection
  extra boxes      detections matching no ground-truth line (dev: scene art, frames, cursors, furigana, unlabelled
                   text); boxes matching a name line are counted apart
  new / lost       tbr's boxes that GRAD does not return in the same view, and GRAD's that tbr does not
Paired tbr - grad differences: eval/vocr3_report's line-unit and VN-cluster bootstraps (finder_tighten.compare);
CLANNAD: whole-page bootstraps (clannad_refs). Every case is also run through the row-callback finder
(vocr_lines_rows.c, what the overlay calls) and must return the same lines as vocr_lines.c. The protocol is
eval/t4_recovery_protocol.json, committed before the protected and CLANNAD runs.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train", ROOT / "teacher", ROOT / "realplus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import finder_tighten as FT  # noqa: E402
from finders import match, overlap  # noqa: E402
from metrics import norm  # noqa: E402

FINDERS = {"grad": 0, "tbr": 3}                     # runtime/vocr_lines.h: GRAD, TEXTBOX_RECOVERY
REFINE = {"off": 0, "on": 0x1F}                     # VOCR_REFINE_ALL, the overlay's VJO_VOCR_REFINE_ALL
MODES = [f"{f}/{r}" for f in FINDERS for r in REFINE]
VIEWS = ("off", "on", "deployed")                   # deployed: on in a PSP game, off in a Vita game
VARIANTS = FT.VARIANTS
SHORT = 0.4
CONSOLE_ROI = [50, 409, 852, 135]                    # #194: this console's CLANNAD region (x, y, w, h)
PROTOCOL = ROOT / "eval" / "t4_recovery_protocol.json"
CODE = ("eval/t4_recovery.py", "eval/finder_tighten.py", "eval/finders.py", "eval/metrics.py", "eval/vocr3_report.py",
        "eval/sets.py", "eval/clannad_refs.py", "train/general_vn_textbox_guard.py", "synth/common.py",
        "runtime/pyvocr.py", "runtime/vocr.c", "runtime/vocr.h", "runtime/vocr_lines.c", "runtime/vocr_lines.h",
        "runtime/vocr_lines_rows.c", "runtime/vocr_lines_rows.h", "runtime/vocr_lines_refine.c",
        "runtime/vocr_lines_refine.h", "runtime/vocr_lines_refine_test.c", "runtime/vocr_lines_test.c")
DEV = ("tv_dlg", "dn_dlg", "own_dlg")

ROWS_DRIVER = r"""/* t4_recovery: vocr_find_lines_rows on a batch (w h bpp finder stride, pixels; uint32 LE). */
#include <stdio.h>
#include <stdlib.h>

#include "vocr_lines_rows.h"

typedef struct { const unsigned char *p; size_t stride; } Mem;

static const unsigned char *mem_row(void *user, unsigned y)
{
    const Mem *m = (const Mem *)user;
    return m->p + (size_t)y * m->stride;
}

static unsigned rd32(const unsigned char *b)
{
    return (unsigned)b[0] | (unsigned)b[1] << 8 | (unsigned)b[2] << 16 | (unsigned)b[3] << 24;
}

int main(int argc, char **argv)
{
    unsigned char hdr[20];
    FILE *f;
    if (argc != 2 || !(f = fopen(argv[1], "rb"))) return 2;
    while (fread(hdr, 1, 20, f) == 20) {
        unsigned w = rd32(hdr), h = rd32(hdr + 4), bpp = rd32(hdr + 8), finder = rd32(hdr + 12), i;
        size_t stride = rd32(hdr + 16), bytes = stride * h, need = vocr_lines_arena_bytes(w, h);
        unsigned char *px = (unsigned char *)malloc(bytes ? bytes : 1), *arena = (unsigned char *)malloc(need + 1);
        VocrLinesRows src;
        VocrLineSet out;
        Mem m;
        int rc;
        if (!px || !arena || fread(px, 1, bytes, f) != bytes) return 2;
        m.p = px;
        m.stride = stride;
        src.width = w; src.height = h; src.bpp = bpp; src.row = mem_row; src.user = &m;
        rc = vocr_find_lines_rows(&src, (int)finder, arena + 1, need, &out);
        printf("%d %u %u", rc, out.count, out.truncated);
        for (i = 0; i < out.count; i++)
            printf(" %u %u %u %u %u", out.lines[i].x, out.lines[i].y, out.lines[i].w, out.lines[i].h,
                   out.lines[i].dark);
        printf("\n");
        free(px);
        free(arena);
    }
    fclose(f);
    return 0;
}
"""


def sha256(path):
    return FT.sha256(path)


def code_identity():
    return {f: sha256(ROOT / f) for f in CODE}


def build(out, compiler="gcc"):
    """finder_tighten's finder + refinement driver and scalar recognizer library, plus the row-form finder driver."""
    out = Path(out).resolve()
    ident = FT.build(out, compiler)
    src, exe = out / "rows_driver.c", out / "rows_driver"
    src.write_text(ROWS_DRIVER)
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
    subprocess.run([compiler, *FT.FINDER_FLAGS, "-Iruntime", "-o", str(exe), "runtime/vocr_lines.c",
                    "runtime/vocr_lines_rows.c", str(src)], check=True, cwd=ROOT, env=env)
    lt = out / "vocr_lines_test"
    subprocess.run([compiler, *FT.FINDER_FLAGS, "-o", str(lt), "runtime/vocr_lines.c", "runtime/vocr_lines_test.c"],
                   check=True, cwd=ROOT, env=env)
    ident.update(rows_driver=str(exe), rows_driver_sha256=sha256(exe), lines_test=str(lt),
                 lines_test_sha256=sha256(lt))
    return ident


def _batch(images, finder, flags):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False, dir=os.environ.get("TMPDIR", "/tmp"))
    with f:
        for img in images:
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape
            hdr = (w, h, 1, finder, w) if flags is None else (w, h, 1, finder, w, flags)
            f.write(struct.pack(f"<{len(hdr)}I", *hdr))
            f.write(img.tobytes())
    return f.name


def run_finder(ident, images, finder, flags, chunk=400):
    """[(found, boxes, dark, truncated)]: flags None = the row-form finder alone, else finder then refinement."""
    out = []
    for s in range(0, len(images), chunk):
        part = images[s:s + chunk]
        path = _batch(part, finder, flags)
        try:
            exe = ident["rows_driver"] if flags is None else ident["driver"]
            got = subprocess.run([exe, path], check=True, capture_output=True, text=True).stdout.splitlines()
        finally:
            os.unlink(path)
        if len(got) != len(part):
            raise RuntimeError(f"driver returned {len(got)} results for {len(part)} images")
        out += got
    res = []
    for line in out:
        v = [int(t) for t in line.split()]
        if v[0] != 0:
            raise RuntimeError(f"finder rc={v[0]}")
        if flags is None:
            found, n, trunc, k = v[1], v[1], v[2], 3
        else:
            found, n, trunc, k = v[1], v[2], v[3], 5
        boxes = [tuple(v[k + 5 * i:k + 5 * i + 4]) for i in range(n)]
        res.append((found, boxes, [v[k + 5 * i + 4] for i in range(n)], trunc))
    return res


def all_modes(ident, images):
    """{mode: [(boxes, dark, truncated)]} for the four modes, after checking that the row-callback finder returns
    vocr_lines.c's lines on every image, for both finders."""
    out = {}
    for fname, fid in FINDERS.items():
        mem = run_finder(ident, images, fid, 0)
        rows = run_finder(ident, images, fid, None)
        for i, (a, b) in enumerate(zip(mem, rows)):
            if (a[1], a[2], a[3]) != (b[1], b[2], b[3]) or a[0] != len(a[1]):
                raise RuntimeError(f"{fname} case {i}: vocr_find_lines_rows differs from vocr_find_lines")
        out[f"{fname}/off"] = [(b, d, t) for _, b, d, t in mem]
        out[f"{fname}/on"] = [(b, d, t) for _, b, d, t in run_finder(ident, images, fid, REFINE["on"])]
    return out


# ---------------------------------------------------------------- recognition (deduplicated, in processes)

_READERS = {}


def _init(arms, library):
    for name, blob in arms:
        _READERS[name] = FT.Reader(Path(blob), library)


def _read(job):
    key, crop, inv = job
    return key, {a: r.read(crop, inv) for a, r in _READERS.items()}


def crop_key(crop, inv):
    return hashlib.sha1(crop.tobytes() + struct.pack("<3I", crop.shape[0], crop.shape[1], int(inv))).hexdigest()


def read_crops(jobs, arms, library, procs):
    """jobs: {key: (grey crop, invert)} -> {key: {arm: normalised text}}."""
    if not jobs:
        return {}
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, _init, (arms, library)) as pool:
        it = pool.imap_unordered(_read, ((k, c, i) for k, (c, i) in jobs.items()), chunksize=8)
        return dict(it)


def arm_list(specs):
    arms = []
    for spec in specs:
        name, blob = FT.parse_arm(spec)
        arms.append((name, str(blob)))
    return arms


# ---------------------------------------------------------------- scoring

def band(img, box, boxes):
    """A private record's context: the box's rows +- 1.5 box heights (the region's full width), its y offset and
    the given boxes in band coordinates."""
    x, y, w, h = box
    y0, y1 = max(0, y - (3 * h) // 2), min(img.shape[0], y + h + (3 * h) // 2)
    shift = [(bx, by - y0, bw, bh) for bx, by, bw, bh in boxes]
    return np.ascontiguousarray(img[y0:y1]), (x, y - y0, w, h), shift


NEW_KINDS = ("short_last", "dialogue", "name", "union", "duplicate", "offtext")


def new_kind(i, box, det_line, gts, is_name, last):
    """What one of tbr's new boxes is: it matches the short last line, another dialogue line or a name line; or it
    matches none and covers two or more lines (union: their text is read again), one line another box matched
    (duplicate) or no line (offtext: scene art, a frame, a cursor, furigana or unlabelled text)."""
    if i in det_line:
        j = det_line[i]
        return "name" if is_name[j] else "short_last" if j == last else "dialogue"
    covered = sum(1 for g in gts if overlap(box, g["box_r"]) >= 0.5)
    return "union" if covered >= 2 else "duplicate" if covered == 1 else "offtext"


def short_last(lines):
    """Index of the region's short last dialogue line (see the module docstring), or None; and every short one."""
    dia = [i for i, ln in enumerate(lines) if "name" not in (ln.get("styles") or [])]
    if len(dia) < 2:
        return None, []
    widest = max(lines[i]["box"][2] for i in dia)
    short = [i for i in dia if lines[i]["box"][2] <= SHORT * widest]
    bottom = max(dia, key=lambda i: (lines[i]["box"][1] + lines[i]["box"][3], lines[i]["box"][1]))
    return (bottom if bottom in short else None), short


def score_cases(cases, ident, arms, procs, part):
    """cases: [{unit, vn, platform, variant, image (grey), lines: [{id, text, styles, box (x, y, w, h)}], ...}].
    Returns ({mode: [public unit records]}, [private records of tbr's new boxes])."""
    from general_vn_textbox_guard import transcript_score
    images = [c["image"] for c in cases]
    found = all_modes(ident, images)
    jobs = {}
    for mode, res in found.items():
        for c, (boxes, dark, _) in zip(cases, res):
            for (x, y, w, h), d in zip(boxes, dark):
                crop = np.ascontiguousarray(c["image"][y:y + h, x:x + w])
                jobs.setdefault(crop_key(crop, d), (crop, d))
    texts = read_crops(jobs, arms, ident["library"], procs)
    names = [a for a, _ in arms]
    units = {m: [] for m in MODES}
    new_private = []
    for ci, c in enumerate(cases):
        lines = c["lines"]
        gts = [{"box_r": tuple(ln["box"])} for ln in lines]
        is_name = ["name" in (ln.get("styles") or []) for ln in lines]
        last, short = short_last(lines)
        for mode in MODES:
            boxes, dark, trunc = found[mode][ci]
            pairs, unmatched = match(list(boxes), gts)
            det_line = {i: j for i, j in pairs}
            reads = [texts[crop_key(np.ascontiguousarray(c["image"][y:y + h, x:x + w]), d)]
                     for (x, y, w, h), d in zip(boxes, dark)]
            rec = {"unit": c["unit"], "vn": c["vn"], "platform": c["platform"], "variant": c["variant"],
                   "part": part, "detections": len(boxes), "truncated": int(trunc), "gt_lines": len(lines),
                   "gt_dialogue": sum(not n for n in is_name),
                   "found": sorted(j for _, j in pairs),
                   "found_dialogue": sum(1 for _, j in pairs if not is_name[j]),
                   "name_boxes": sum(1 for _, j in pairs if is_name[j]),
                   "extra_boxes": len(unmatched), "short_last": last, "short": short,
                   "has_furigana": any("furigana" in (ln.get("styles") or []) for ln in lines)}
            for a in names:
                preds = [r[a] for r in reads]
                sc = transcript_score(lines, preds)
                rec[a] = {"all_errors": sc["all"]["full"]["errors"], "all_chars": sc["all"]["full"]["chars"],
                          "body_errors": sc["body"]["full"]["errors"], "body_chars": sc["body"]["full"]["chars"],
                          "extra_chars": sum(len(norm(preds[i])) for i in unmatched)}
            if mode.startswith("tbr/"):
                gboxes = found["grad/" + mode.split("/")[1]][ci][0]
                new = [i for i, b in enumerate(boxes) if b not in gboxes]
                kinds = {i: new_kind(i, boxes[i], det_line, gts, is_name, last) for i in new}
                rec["new_boxes"] = len(new)
                for k in NEW_KINDS:
                    rec[f"new_{k}"] = sum(1 for i in new if kinds[i] == k)
                rec["new_extra"] = sum(1 for i in new if i not in det_line)
                rec["lost_boxes"] = sum(1 for b in gboxes if b not in boxes)
                for i in new:
                    img, bb, gshift = band(c["image"], boxes[i], gboxes)
                    new_private.append({"part": part, "unit": c["unit"], "variant": c["variant"], "mode": mode,
                                        "vn": c["vn"], "kind": kinds[i], "box": bb, "image": img, "grad_boxes": gshift,
                                        "reads": {a: reads[i][a] for a in names}})
            units[mode].append(rec)
    return units, new_private


# ---------------------------------------------------------------- dev

def load_dev(cohorts, boxes_dir, limit=0):
    out = {}
    for name in DEV:
        cases = FT.load_cohort(cohorts, name)
        bx = json.loads((Path(boxes_dir) / f"{name}.boxes.json").read_text())
        if bx["frozen_sha256"] != sha256(Path(cohorts) / f"{name}.pkl"):
            raise SystemExit(f"{name}: boxes.json was built from another cohort file")
        rows = []
        for c in cases[:limit or None]:
            b = bx["boxes"][f"{c['unit']}|{c['variant']}"]
            if len(b) != len(c["lines"]):
                raise SystemExit(f"{name} {c['unit']}: boxes and lines differ")
            lines = [dict(ln, box=(x0, y0, x1 - x0, y1 - y0)) for ln, (x0, y0, x1, y1) in zip(c["lines"], b)]
            rows.append({"unit": f"{name}|{c['unit']}|{c['variant']}", "shot": c["shot"], "vn": c["vn"],
                         "platform": c["kind"], "variant": c["variant"], "image": c["image"], "lines": lines,
                         "cohort": name})
        out[name] = rows
    return out


def cmd_dev(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(exist_ok=True)
    ident = build(out / "build", args.compiler)
    arms = arm_list(args.arm)
    data = load_dev(args.cohorts, args.boxes, args.limit)
    tv_shots = {c["shot"] for c in data["tv_dlg"]}
    meta = {"kind": "t4-recovery-dev", "native": ident, "code_sha256": code_identity(), "arms": {},
            "cohorts": {n: {"sha256": sha256(Path(args.cohorts) / f"{n}.pkl"), "cases": len(v)}
                        for n, v in data.items()}}
    for name, blob in arms:
        meta["arms"][name] = {"blob": Path(blob).name, "sha256": sha256(blob)}
    units, new = {m: [] for m in MODES}, []
    for name in DEV:
        cases = data[name]
        u, n = score_cases(cases, ident, arms, args.procs, name)
        for m in MODES:
            for rec, c in zip(u[m], cases):
                rec["pooled"] = not (name == "dn_dlg" and c["shot"] in tv_shots)
            units[m] += u[m]
        new += n
        print(f"{name}: {len(cases)} cases scored", flush=True)
    (out / "units.json").write_text(json.dumps(units) + "\n")
    with open(out / "private" / "new_boxes.pkl", "wb") as f:
        pickle.dump(new, f, protocol=4)
    with open(out / "private" / "bench_regions.pkl", "wb") as f:
        pickle.dump([c["image"] for n in DEV for c in data[n] if c["variant"] == "base"], f, protocol=4)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(summary_table(units, [a for a, _ in arms], lambda u: u["pooled"]), indent=1))


# ---------------------------------------------------------------- protected sets

def protected_cases(set_name, data):
    import sets
    import vocr3_report as V

    from common import to_gray
    platform = sets.SETS[set_name]
    cache = sets.ImageCache()
    cases, excluded = [], 0
    for path, region, rows in sets.regions(set_name, data):
        if any(r.get("uncertain") for r in rows):
            excluded += 1
            continue
        img = cache.get(path)
        dims = [img.shape[1], img.shape[0]]
        rows = sorted(rows, key=lambda r: (r["box"][1], r["box"][0]))
        rid = f"{Path(path).name}@{','.join(map(str, region))}"
        for variant, pad in VARIANTS.items():
            x, y, w, h = V.expand(region, pad, dims)
            lines = []
            for r in rows:
                bx, by, bw, bh = r["box"]
                x0, y0 = max(bx, x), max(by, y)
                x1, y1 = min(bx + bw, x + w), min(by + bh, y + h)
                lines.append({"id": r["id"], "text": r["text"], "styles": r.get("styles") or [],
                              "box": (x0 - x, y0 - y, max(1, x1 - x0), max(1, y1 - y0))})
            grey = np.ascontiguousarray(to_gray(np.ascontiguousarray(img[y:y + h, x:x + w])))
            cases.append({"unit": f"{set_name}|{rid}|{variant}", "vn": rows[0]["vn"], "platform": platform,
                          "variant": variant, "image": grey, "lines": lines, "set": set_name})
    return cases, excluded


def cmd_protected(args):
    out = Path(args.out)
    proto = json.loads(PROTOCOL.read_text())
    if proto["code_sha256"] != {k: v for k, v in code_identity().items() if k in proto["code_sha256"]}:
        raise SystemExit("the code differs from the protocol's")
    FT.protected_once(out / "protected.attempt.json", {"started": FT.utc_now(), "protocol_sha256": sha256(PROTOCOL)})
    (out / "private").mkdir(exist_ok=True)
    ident = build(out / "build", args.compiler)
    arms = arm_list(args.arm)
    for name, blob in arms:
        if proto["arms"][name] != sha256(blob):
            raise SystemExit(f"{name}: blob differs from the protocol's")
    units, new, account = {m: [] for m in MODES}, [], {}
    for set_name in ("vita-test", "psp-test"):
        cases, excluded = protected_cases(set_name, args.data)
        u, n = score_cases(cases, ident, arms, args.procs, set_name)
        for m in MODES:
            units[m] += u[m]
        new += n
        account[set_name] = {"cases": len(cases), "regions_excluded_uncertain": excluded}
        print(f"{set_name}: {len(cases)} cases scored ({excluded} regions with an uncertain line left out)", flush=True)
    (out / "private" / "units.json").write_text(json.dumps(units) + "\n")
    with open(out / "private" / "new_boxes.pkl", "wb") as f:
        pickle.dump(new, f, protocol=4)
    meta = {"kind": "t4-recovery-protected", "native": ident, "code_sha256": code_identity(), "account": account,
            "arms": {a: {"blob": Path(b).name, "sha256": sha256(b)} for a, b in arms}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "units.json").write_text(json.dumps(units) + "\n")   # counts only: no text, no image
    print(json.dumps(summary_table(units, [a for a, _ in arms], lambda u: True), indent=1))


# ---------------------------------------------------------------- CLANNAD long run

def cmd_clannad(args):
    import clannad_refs as C
    import vocr3_report as V
    from PIL import Image

    from common import to_gray
    out = Path(args.out)
    work = Path(args.work)
    proto = json.loads(PROTOCOL.read_text())
    if proto["code_sha256"] != {k: v for k, v in code_identity().items() if k in proto["code_sha256"]}:
        raise SystemExit("the code differs from the protocol's")
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if sha256(work / "refs.jsonl") != freeze["refs_sha256"] or freeze["refs_sha256"] != proto["clannad_refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record or the protocol")
    FT.protected_once(out / "clannad.attempt.json", {"started": FT.utc_now(), "refs_sha256": freeze["refs_sha256"],
                                                      "protocol_sha256": sha256(PROTOCOL)})
    (out / "private").mkdir(exist_ok=True)
    ident = build(out / "build", args.compiler)
    arms = arm_list(args.arm)
    for name, blob in arms:
        if proto["arms"][name] != sha256(blob):
            raise SystemExit(f"{name}: blob differs from the protocol's")
    names = [a for a, _ in arms]
    rows, meta = C.load_lines(work)
    session = Path(meta["session"])
    refs = {x["id"]: x for x in C.read_jsonl(work / "refs.jsonl")}
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)
    pages, rois = sorted(by_page), {"longrun": None, "console": CONSOLE_ROI}
    imgs = {k: [] for k in rois}
    for page in pages:
        src = by_page[page][0]["source"]
        path = session / src["path"]
        if sha256(path) != src["sha256"]:
            raise SystemExit(f"{src['path']}: frame hash differs from the capture journal")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        for k, roi in rois.items():
            x, y, w, h = roi or by_page[page][0]["roi"]
            imgs[k].append(np.ascontiguousarray(to_gray(np.ascontiguousarray(rgb[y:y + h, x:x + w]))))
    found = {k: all_modes(ident, v) for k, v in imgs.items()}
    jobs = {}
    for k in rois:
        for mode in ("grad/off", "tbr/off"):
            for img, (boxes, dark, _) in zip(imgs[k], found[k][mode]):
                for (x, y, w, h), d in zip(boxes, dark):
                    crop = np.ascontiguousarray(img[y:y + h, x:x + w])
                    jobs.setdefault(crop_key(crop, d), (crop, d))
    texts = read_crops(jobs, arms, ident["library"], args.procs)

    def read(img, box, d):
        x, y, w, h = box
        return texts[crop_key(np.ascontiguousarray(img[y:y + h, x:x + w]), d)]

    result = {"kind": "t4-recovery-clannad", "refs_sha256": freeze["refs_sha256"], "pages": len(pages),
              "native": ident, "code_sha256": code_identity(), "arms": {a: {"sha256": sha256(b)} for a, b in arms},
              "rois": {"longrun": by_page[pages[0]][0]["roi"], "console": CONSOLE_ROI}, "lines": {}, "refs": {}}
    new_private = []
    for k in rois:
        g, t = found[k]["grad/off"], found[k]["tbr/off"]
        st = {"grad_lines": sum(len(b) for b, _, _ in g), "tbr_lines": sum(len(b) for b, _, _ in t),
              "grad_truncated": sum(x[2] for x in g), "tbr_truncated": sum(x[2] for x in t),
              "new_boxes": 0, "lost_boxes": 0, "pages_with_new": 0, "pages_with_lost": 0,
              "pages_by_lines": {f: {} for f in ("grad", "tbr")}}
        for page, img, (gb, gd, _), (tb, td, _) in zip(pages, imgs[k], g, t):
            new = [i for i, b in enumerate(tb) if b not in gb]
            lost = [b for b in gb if b not in tb]
            st["new_boxes"] += len(new)
            st["lost_boxes"] += len(lost)
            st["pages_with_new"] += bool(new)
            st["pages_with_lost"] += bool(lost)
            for f, bb in (("grad", gb), ("tbr", tb)):
                st["pages_by_lines"][f][str(len(bb))] = st["pages_by_lines"][f].get(str(len(bb)), 0) + 1
            for i in new:
                covered = sum(1 for b in gb if overlap(tb[i], b) >= 0.5)
                kind = "union" if covered >= 2 else "duplicate" if covered == 1 else "offgrad"
                st[f"new_{kind}"] = st.get(f"new_{kind}", 0) + 1
                im2, bb, gshift = band(img, tb[i], gb)
                new_private.append({"part": f"clannad-{k}", "unit": page, "variant": "base", "mode": "tbr/off",
                                    "vn": "clannad", "kind": kind, "box": bb, "image": im2,
                                    "grad_boxes": gshift, "reads": {a: read(img, tb[i], td[i])[a] for a in names}})
        result["lines"][k] = st
    # the referenced lines (long-run ROI): each is a GRAD box; tbr's reading is the same box's, or the tbr boxes
    # inside it when tbr replaced it
    preds = {a: {"grad": {}, "tbr": {}} for a in names}
    replaced = missing = 0
    for page, img, (gb, gd, _), (tb, td, _) in zip(pages, imgs["longrun"], found["longrun"]["grad/off"],
                                                  found["longrun"]["tbr/off"]):
        for r in by_page[page]:
            if refs.get(r["id"], {}).get("status") != "ref":
                continue
            box = tuple(r["box"])
            if box not in gb:
                missing += 1
                continue
            gr = read(img, box, gd[gb.index(box)])
            if box in tb:
                tr = read(img, box, td[tb.index(box)])
            else:
                replaced += 1
                x, y, w, h = box
                inside = [i for i, (bx, by, bw, bh) in enumerate(tb)
                          if bx >= x and by >= y and bx + bw <= x + w and by + bh <= y + h]
                tr = {a: "".join(read(img, tb[i], td[i])[a] for i in inside) for a in names}
            for a in names:
                preds[a]["grad"][r["id"]] = gr[a]
                preds[a]["tbr"][r["id"]] = tr[a]
    scored = [x for x in refs.values() if x["status"] == "ref" and x["id"] in preds[names[0]]["grad"]]
    for scope, keep in (("dialogue", lambda x: not x["name"]), ("all", lambda x: True)):
        sel = [x for x in scored if keep(x)]
        if not sel:
            continue
        pg = [x["page"] for x in sel]
        ids = sorted(set(pg))
        idx = C.boot_idx(len(ids), C.REPS, C.SEED)
        e = {"lines": len(sel), "pages": len(ids)}
        for a in names:
            sums = {}
            for f in ("grad", "tbr"):
                u = [C.line_unit(x["text"], preds[a][f][x["id"]]) for x in sel]
                sums[f] = C.page_sums(u, pg)
                e[f"{a}/{f}"] = C.summary(sums[f], ids, idx)
            pb = V.paired_bootstrap({p: tuple(sums["tbr"][p][:2]) for p in ids},
                                    {p: tuple(sums["grad"][p][:2]) for p in ids}, C.REPS, C.SEED)
            e[f"{a}/tbr-grad"] = {"cer_delta": pb["delta"], "cer_ci95": pb["ci95"],
                                  "candidate_errors": pb["candidate_errors"], "baseline_errors": pb["baseline_errors"],
                                  "chars": pb["chars"]}
        result["refs"][scope] = e
    result["refs"]["missing_from_grad"] = missing
    result["refs"]["replaced_by_tbr"] = replaced
    FT.assert_public(result)
    (out / "clannad.json").write_text(json.dumps(result, indent=1) + "\n")
    with open(out / "private" / "new_boxes.pkl", "wb") as f:
        pickle.dump(new_private, f, protocol=4)
    with open(out / "private" / "bench_regions.pkl", "wb") as f:
        pickle.dump(imgs["console"], f, protocol=4)
    (out / "meta.json").write_text(json.dumps({"kind": "t4-recovery-clannad", "native": ident}, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("lines", "refs")}, indent=1))


# ---------------------------------------------------------------- report

def view_units(units, view, finder):
    """The view's unit records for a finder: off, on, or deployed (on for PSP units)."""
    if view != "deployed":
        return units[f"{finder}/{view}"]
    off, on = units[f"{finder}/off"], units[f"{finder}/on"]
    return [b if a["platform"] == "psp" else a for a, b in zip(off, on)]


def counts(us, arms):
    c = {k: sum(u[k] for u in us) for k in ("detections", "truncated", "gt_lines", "gt_dialogue", "found_dialogue",
                                             "name_boxes", "extra_boxes")}
    c["found_lines"] = sum(len(u["found"]) for u in us)
    c["short_last"] = sum(u["short_last"] is not None for u in us)
    c["short_last_found"] = sum(u["short_last"] is not None and u["short_last"] in u["found"] for u in us)
    c["short"] = sum(len(u["short"]) for u in us)
    c["short_found"] = sum(sum(j in u["found"] for j in u["short"]) for u in us)
    c["furigana_regions_extra"] = sum(u["extra_boxes"] for u in us if u["has_furigana"])
    for k in ("new_boxes", "new_extra", "lost_boxes", *(f"new_{x}" for x in NEW_KINDS)):
        if k in us[0]:
            c[k] = sum(u[k] for u in us)
    for a in arms:
        e, n = sum(u[a]["body_errors"] for u in us), sum(u[a]["body_chars"] for u in us)
        ea, na = sum(u[a]["all_errors"] for u in us), sum(u[a]["all_chars"] for u in us)
        c[a] = {"dialogue_errors": e, "dialogue_chars": n, "dialogue_cer": round(100 * e / n, 3) if n else None,
                "all_errors": ea, "all_chars": na, "all_cer": round(100 * ea / na, 3) if na else None,
                "extra_chars": sum(u[a]["extra_chars"] for u in us)}
    return c


def summary_table(units, arms, keep, view_list=VIEWS, variants=("all",)):
    out = {}
    for view in view_list:
        for v in variants:
            sel = {f: [u for u in view_units(units, view, f) if keep(u) and (v == "all" or u["variant"] == v)]
                   for f in FINDERS}
            if not sel["grad"]:
                continue
            row = {f: counts(sel[f], arms) for f in FINDERS}
            for a in arms:
                cand = [dict(unit=u["unit"], vn=u["vn"], errors=u[a]["body_errors"], chars=u[a]["body_chars"])
                        for u in sel["tbr"]]
                base = [dict(unit=u["unit"], vn=u["vn"], errors=u[a]["body_errors"], chars=u[a]["body_chars"])
                        for u in sel["grad"]]
                row[f"{a}/tbr-grad"] = FT.compare(cand, base)
            gf = {u["unit"]: set(u["found"]) for u in sel["grad"]}
            row["lines_only_grad"] = sum(len(gf[u["unit"]] - set(u["found"])) for u in sel["tbr"])
            row["lines_only_tbr"] = sum(len(set(u["found"]) - gf[u["unit"]]) for u in sel["tbr"])
            out[f"{view}/{v}"] = row
    return out


def cmd_report(args):
    rep = {"kind": "t4-recovery-report", "parts": {}}
    for d in args.dir:
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        units = json.loads((d / "units.json").read_text())
        arms = list(meta["arms"])
        if meta["kind"] == "t4-recovery-dev":
            parts = {"dev-pooled": lambda u: u["pooled"]}
            parts.update({n: (lambda u, n=n: u["part"] == n) for n in DEV})
        else:
            parts = {"vita-test": lambda u: u["part"] == "vita-test", "psp-test": lambda u: u["part"] == "psp-test",
                     "protected-pooled": lambda u: True}
        for name, keep in parts.items():
            variants = ("all",) if name in DEV else ("base", "expanded-4", "expanded-8", "all")
            rep["parts"][name] = {"arms": meta["arms"], "code_sha256": meta["code_sha256"],
                                  "table": summary_table(units, arms, keep, VIEWS, variants)}
            print(name, flush=True)
    FT.assert_public(rep)
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    if args.md:
        Path(args.md).write_text(markdown(rep) + "\n")
    print(markdown(rep))


def markdown(rep):
    md = []
    for name, p in rep["parts"].items():
        arms = list(p["arms"])
        md.append(f"### {name}\n")
        md.append("| view | lines found GRAD → TBR (of GT) | short last found GRAD → TBR (of) | "
                  "extra boxes GRAD → TBR | TBR new: short last + dialogue / name / union / duplicate / off-text; "
                  "lost | " +
                  " | ".join(f"{a} dialogue CER GRAD → TBR, Δ (VN CI)" for a in arms) + " |")
        md.append("|---" * (5 + len(arms)) + "|")
        for key, row in p["table"].items():
            g, t = row["grad"], row["tbr"]
            cells = [f"{key}", f"{g['found_lines']} → {t['found_lines']} ({g['gt_lines']})",
                     f"{g['short_last_found']} → {t['short_last_found']} ({g['short_last']})",
                     f"{g['extra_boxes']} → {t['extra_boxes']}",
                     f"{t.get('new_short_last', 0)} + {t.get('new_dialogue', 0)} / {t.get('new_name', 0)} / "
                     f"{t.get('new_union', 0)} / {t.get('new_duplicate', 0)} / {t.get('new_offtext', 0)}; "
                     f"{t.get('lost_boxes', 0)}"]
            for a in arms:
                c = row[f"{a}/tbr-grad"]
                cells.append(f"{g[a]['dialogue_cer']:.2f} → {t[a]['dialogue_cer']:.2f}, {FT.fmt_ci(c)}")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    return "\n".join(md)


# ---------------------------------------------------------------- contact sheets of tbr's new boxes (private)

def cmd_sheet(args):
    from PIL import Image, ImageDraw
    rows = []
    for d in args.dir:
        with open(Path(d) / "private" / "new_boxes.pkl", "rb") as f:
            rows += [r for r in pickle.load(f)
                     if args.part == "all" or r["part"] == args.part or (args.part == "dev" and r["part"] in DEV)]
    if args.mode:
        rows = [r for r in rows if r["mode"] == args.mode]
    if args.variant:
        rows = [r for r in rows if r["variant"] == args.variant]
    if args.kind:
        rows = [r for r in rows if r["kind"] in args.kind.split(",")]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows)) if args.shuffle else np.arange(len(rows))
    rows = [rows[i] for i in order][:args.limit or None]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tile_w, per = 620, args.per
    for s in range(0, len(rows), per):
        chunk, tiles = rows[s:s + per], []
        for r in chunk:
            im = Image.fromarray(r["image"]).convert("RGB")
            dr = ImageDraw.Draw(im)
            x, y, w, h = r["box"]
            for bx, by, bw, bh in r["grad_boxes"]:
                dr.rectangle([bx, by, bx + bw - 1, by + bh - 1], outline=(60, 140, 255))
            dr.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 40, 40), width=2)
            sc = min(1.0, tile_w / im.width)
            if sc < 1:
                im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.BILINEAR)
            tiles.append(im)
        cols = 2
        hrow = [max(t.height for t in tiles[i:i + cols]) + 16 for i in range(0, len(tiles), cols)]
        sheet = Image.new("RGB", (cols * (tile_w + 10), sum(hrow)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        yy = 0
        for i in range(0, len(tiles), cols):
            for j, t in enumerate(tiles[i:i + cols]):
                xx = j * (tile_w + 10)
                dr.text((xx + 2, yy + 1), str(s + i + j), fill=(0, 0, 0))
                sheet.paste(t, (xx, yy + 14))
            yy += hrow[i // cols]
        if sheet.height > 1280:
            sc = 1280 / sheet.height
            sheet = sheet.resize((int(sheet.width * sc), 1280), Image.BILINEAR)
        name = f"{args.part}_{s // per:02d}"
        sheet.save(out / f"{name}.png")
        (out / f"{name}.json").write_text(json.dumps(
            [{"n": s + i, "part": r["part"], "unit": r["unit"], "variant": r["variant"], "mode": r["mode"],
              "kind": r["kind"], "box": list(r["box"]), "reads": r["reads"]} for i, r in enumerate(chunk)],
            ensure_ascii=False, indent=0) + "\n")
        print(f"{out / name}.png: {len(chunk)} boxes")


# ---------------------------------------------------------------- host time of the finder (not the console's)

def cmd_bench(args):
    ident = json.loads((Path(args.dir) / "meta.json").read_text())["native"]
    with open(Path(args.dir) / "private" / "bench_regions.pkl", "rb") as f:
        images = pickle.load(f)
    res = {}
    for fname, fid in FINDERS.items():
        path = _batch(images, fid, None)
        try:
            out = subprocess.run([ident["lines_test"], "--bench", path, str(args.reps)], check=True,
                                 capture_output=True, text=True).stdout.splitlines()
        finally:
            os.unlink(path)
        best = [float(x.split()[3]) for x in out[:-1]]
        res[fname] = {"regions": len(best), "sum_best_us": round(sum(best), 1),
                      "median_best_us": float(np.median(best))}
    res["ratio_sum"] = round(res["tbr"]["sum_best_us"] / res["grad"]["sum_best_us"], 3)
    print(json.dumps(res, indent=1))
    (Path(args.dir) / "bench.json").write_text(json.dumps(res, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("dev", cmd_dev), ("protected", cmd_protected), ("clannad", cmd_clannad)):
        p = sub.add_parser(name)
        p.add_argument("--arm", action="append", required=True)
        p.add_argument("--compiler", default="gcc")
        p.add_argument("--procs", type=int, default=5)
        p.add_argument("--out", required=True)
        if name == "dev":
            p.add_argument("--cohorts", required=True)
            p.add_argument("--boxes", required=True)
            p.add_argument("--limit", type=int, default=0)
        elif name == "protected":
            p.add_argument("--data", required=True)
        else:
            p.add_argument("--work", required=True)
        p.set_defaults(fn=fn)
    p = sub.add_parser("report")
    p.add_argument("--dir", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--md")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("sheet")
    p.add_argument("--dir", nargs="+", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--mode")
    p.add_argument("--variant")
    p.add_argument("--kind")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--per", type=int, default=24)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("bench")
    p.add_argument("--dir", required=True)
    p.add_argument("--reps", type=int, default=20)
    p.set_defaults(fn=cmd_bench)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

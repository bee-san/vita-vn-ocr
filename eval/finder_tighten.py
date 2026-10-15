"""finder-tighten: development harness for the post-finder line refinement (runtime/vocr_lines_refine.c).

  finder_tighten.py cohorts --out DIR            build the development cohorts (private: crops of copyrighted frames)
  finder_tighten.py score --cohorts DIR --arm NAME=BLOB ... --mode MODE ... --out DIR
                                                 the C finder (+ refinement) and the C runtime in W8I on every case
  finder_tighten.py report --scores DIR ... --out JSON [--md MD]
                                                 integer counts, CER, paired line and VN-cluster bootstraps

Cohorts (development only; the 13:25Z policy makes everything outside CLANNAD, vita-test and psp-test development
or training data). Each is a list of cases with a grey region image (synth/common.to_gray of the frame, the grey
the console's finder and recognizer compute from RGBA):
  dn_pairs   devnative v3 (the VOCR3/VOCR4 select-pool copies): every line with both a hand-cut ink crop and a
             clean GRAD crop. The region is the one devnative cut the GRAD crop from; the case records which
             finder line is the GRAD crop, so a refinement mode reads its own box for that line.
  own_pairs  the owner pool v2 (private frames): the same for its GRAD/ink pairs.
  tv_dlg     teacher_val (the amendment-8 copy) as whole dialogue: dialogue regions a player would draw.
  dn_dlg     devnative v3 as whole dialogue.
  own_dlg    the owner pool as whole dialogue.
Whole-dialogue regions: the labelled non-name lines of a frame grouped with eval's rule (vertical gap <= 1.6 line
heights), their union plus 12 px (Vita) / 6 px (PSP), then grown by 0 / 4 / 8 px (the VOCR3 harness's base,
expanded-4, expanded-8). Every other detected text box (speaker names, unlabelled or rejected lines, UI, vertical
text) is kept out: the region is shrunk on the side facing it, as a player crops a name plate out; a region whose
text such a box overlaps is dropped whole. A name is a reviewed name style (devnative) or name-like (6 or fewer
characters and no punctuation, eval/synth3_ablation.py's rule) elsewhere.

Scoring. Pairs: the line's crop from each mode (C finder box, then the refinement's box for the same line) against
the hand-cut ink crop, every crop read by the C runtime (pyvocr, deployment W8I) with its dark flag as VOCR_INVERT.
Dialogue: every crop the mode returns, in finder order, recognized and joined, scored by the textbox guard's
transcript_score (train/general_vn_textbox_guard.py). Dialogue CER is the primary metric (owner 2026-10-14
13:57Z): pairs report dialogue lines and names separately, and dialogue regions hold no name.
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train", ROOT / "teacher", ROOT / "realplus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIVATE = HERD / "shared" / "private" / "owner-games" / "_build" / "owner-pool"
SELECT_POOLS = ART / "synth3" / "vocr3" / "select_pools"
MARGIN = {"vita": 12, "psp": 6}
VARIANTS = {"base": 0, "expanded-4": 4, "expanded-8": 8}
PUNCT = set("「」『』。、！？!?…―（）")
GROUP_GAP = 1.6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def namelike(label):
    t = norm(label)
    return len(t) <= 6 and not set(t) & PUNCT


def load_npz(path):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    out = []
    for i, (lab, m) in enumerate(zip(labels, meta)):
        h, w = (int(v) for v in shapes[i])
        out.append((m, lab, data[offs[i]:offs[i + 1]].reshape(h, w)))
    return out


# ---------------------------------------------------------------- geometry

def inter_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def groups_of(boxes):
    """eval's groups() over [x0, y0, x1, y1) boxes, top to bottom: indices per group."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    groups = []
    for i in order:
        b = boxes[i]
        if groups:
            g = groups[-1]
            bottom = max(boxes[j][3] for j in g)
            hmax = max(boxes[j][3] - boxes[j][1] for j in g + [i])
            if b[1] - bottom <= GROUP_GAP * hmax:
                g.append(i)
                continue
        groups.append([i])
    return groups


def clip_region(reg, core, others):
    """reg shrunk so that no other text box enters it (the side facing each box moves in); None when another
    box overlaps the text itself (core: the union of the member lines' boxes, one pixel of slack)."""
    r = list(reg)
    for o in others:
        if inter_area(o, r) <= 0:
            continue
        if inter_area(o, [core[0] - 1, core[1] - 1, core[2] + 1, core[3] + 1]) > 0:
            return None
        if o[3] <= core[1]:
            r[1] = max(r[1], o[3])
        elif o[1] >= core[3]:
            r[3] = min(r[3], o[1])
        elif o[2] <= core[0]:
            r[0] = max(r[0], o[2])
        else:
            r[2] = min(r[2], o[0])
    if r[2] - r[0] < 8 or r[3] - r[1] < 8:
        return None
    for o in others:
        if inter_area(o, r) > 0:
            return None
    return r


def dialogue_units(shot, kind, W, H, members, others, grey, cohort, vn):
    """Whole-dialogue cases of one frame: members = [(box, id, label)] of labelled dialogue lines, others = every
    other text box. One case per group and variant."""
    out = []
    boxes = [m[0] for m in members]
    for g in groups_of(boxes):
        core = union([boxes[j] for j in g])
        lines = sorted((members[j] for j in g), key=lambda m: (m[0][1], m[0][0]))
        for variant, extra in VARIANTS.items():
            m = MARGIN[kind] + extra
            reg = [max(0, core[0] - m), max(0, core[1] - m), min(W, core[2] + m), min(H, core[3] + m)]
            reg = clip_region(reg, core, others)
            if reg is None:
                if variant == "base":
                    break                      # the frame's group is dropped whole, every variant
                continue
            x0, y0, x1, y1 = reg
            out.append({"cohort": cohort, "unit": f"{shot}@{','.join(map(str, core))}", "shot": shot, "vn": vn,
                        "kind": kind, "variant": variant, "region": reg,
                        "image": np.ascontiguousarray(grey[y0:y1, x0:x1]),
                        "lines": [{"id": i, "text": lab, "styles": []} for _, i, lab in lines]})
    return out


# ---------------------------------------------------------------- cohort builders

def _devnative_lines():
    full = {}
    for line in (ART / "devnative" / "v3" / "lines_full.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        full[r["id"]] = r
    return full


def _devnative_select_ids():
    """The line ids of the VOCR3/VOCR4 select-pool copy of devnative v3 (ink and GRAD files)."""
    ids = {}
    for sub in ("strict", "tv_native"):
        for crop in ("ink", "grad"):
            for m, lab, g in load_npz(SELECT_POOLS / "devnative_v3" / f"devnative_{sub}_{crop}.npz"):
                ids.setdefault(m["id"], {})[crop] = (m, lab, g)
    return ids


def build_dn_pairs():
    from common import find_lines_ex, to_gray
    from devnative import paths as P
    full = _devnative_lines()
    shots = {json.loads(x)["shot"]: json.loads(x)
             for x in (ROOT / "devnative" / "manifests" / "shots.jsonl").read_text().splitlines()}
    sel = _devnative_select_ids()
    screens, cases = {}, []
    for lid in sorted(sel):
        crops = sel[lid]
        if "grad" not in crops or "ink" not in crops:
            continue
        f = full[lid]
        if f["shot"] not in screens:
            screens[f["shot"]] = to_gray(np.ascontiguousarray(P.load_screen(shots[f["shot"]])))
        g = screens[f["shot"]]
        x0, y0, x1, y1 = f["region"]
        reg = np.ascontiguousarray(g[y0:y1, x0:x1])
        boxes, dark, _ = find_lines_ex(reg, "grad")
        gb = f["grad"]["box"]
        idx = [k for k, (x, y, w, h) in enumerate(boxes) if [x0 + x, y0 + y, x0 + x + w, y0 + y + h] == gb]
        if len(idx) != 1:
            raise RuntimeError(f"{lid}: the GRAD box is not reproduced")
        mg, lab, gcrop = crops["grad"]
        k = idx[0]
        x, y, w, h = boxes[k]
        mine = reg[y:y + h, x:x + w]
        if dark[k]:
            mine = 255 - mine
        if mine.shape != gcrop.shape or (mine != gcrop).any():
            raise RuntimeError(f"{lid}: the GRAD crop is not reproduced")
        mi, _, icrop = crops["ink"]
        cases.append({"cohort": "dn_pairs", "id": lid, "vn": f["vn"], "kind": f["kind"],
                      "name": "name" in (f.get("styles") or []), "ref": lab, "image": reg, "line": k,
                      "ink_crop": icrop, "ink_inverted": bool(mi["inverted"])})
    return cases


def _owner_kept():
    """The owner pool's kept labels (labels.jsonl, before realplus's per-VN repeat cap): the lines
    realplus/native.py grouped into regions."""
    rows = [json.loads(x) for x in (PRIVATE / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r["status"] == "kept"]


def _owner_reads():
    reads = {}
    for x in (PRIVATE / "reads" / "reads.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        reads[r["shot"]] = r
    return reads


def build_own_pairs():
    """The owner pool's GRAD/ink pairs, with their regions rebuilt as realplus/native.py cut them."""
    import pool as PL

    from common import find_lines_ex, to_gray
    from devnative import lines as L
    from devnative import paths as P
    grad = {m["id"]: (m, lab, g) for m, lab, g in load_npz(PRIVATE / "native_train_grad_v2.npz")}
    ink = {m["id"]: (m, lab, g) for m, lab, g in load_npz(PRIVATE / "native_train_ink_v2.npz")}
    reads = _owner_reads()
    by_shot = {}
    for r in _owner_kept():
        by_shot.setdefault(r["shot"], []).append(r)
    cases = []
    for shot, recs in sorted(by_shot.items()):
        if not any(r["id"] in grad for r in recs):
            continue
        s = reads[shot]
        grey = to_gray(np.ascontiguousarray(P.load_screen(s)))
        H, W = grey.shape
        idx = {str(ln["k"]): ln for ln in s["lines"]}
        recs = sorted(recs, key=lambda r: r["id"])
        lines = [PL.read_line(idx[r["id"].rsplit("_", 1)[1]]) for r in recs]
        for reg, members in L.regions(lines, s["kind"], W, H):
            x0, y0, x1, y1 = reg
            im = np.ascontiguousarray(grey[y0:y1, x0:x1])
            boxes, dark, _ = find_lines_ex(im, "grad")
            for j in members:
                lid = recs[j]["id"]
                if lid not in grad or lid not in ink:
                    continue
                gb = grad[lid][0]["box"]
                k = [i for i, (x, y, w, h) in enumerate(boxes) if [x0 + x, y0 + y, x0 + x + w, y0 + y + h] == gb]
                if len(k) != 1:
                    raise RuntimeError(f"{lid}: the GRAD box is not reproduced")
                mi, lab, icrop = ink[lid]
                cases.append({"cohort": "own_pairs", "id": lid, "vn": s["vn"], "kind": s["kind"],
                              "name": bool(recs[j].get("name")), "ref": lab, "image": im, "line": k[0],
                              "ink_crop": icrop, "ink_inverted": bool(mi["inverted"])})
    return cases


def _read_boxes(s, scale=1.0):
    """Every detected text box of a reads frame, at the target scale: the horizontal lines' ink boxes (with
    furigana) and every detector box (any orientation)."""
    import geometry as G
    out = []
    for ln in s.get("lines") or []:
        b = ln.get("crop_ink") or ln["ink"]
        out.append(("line", str(ln["k"]), G.crop_box(b, scale, 0, 1 << 30, 1 << 30)))
    for d in s.get("dets") or []:
        out.append(("det", None, G.crop_box(d["det"], scale, 0, 1 << 30, 1 << 30)))
    return out


def _frame_units(cohort, s, kind, grey, labelled, name_of, scale=1.0):
    """labelled: {read line k: (line id, label)} of this frame and target."""
    H, W = grey.shape
    members, others = [], []
    for what, k, box in _read_boxes(s, scale):
        box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        if what == "line" and k in labelled and not name_of(k):
            lid, lab = labelled[k]
            members.append((box, lid, lab))
        elif what == "line":
            others.append(box)
    for what, k, box in _read_boxes(s, scale):
        if what != "det":
            continue
        box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        # a detector box that is mostly inside a member line belongs to it; any other is another text
        if any(inter_area(box, m[0]) >= 0.5 * max(1, (box[2] - box[0]) * (box[3] - box[1])) for m in members):
            continue
        others.append(box)
    if not members:
        return []
    return dialogue_units(s["shot"], kind, W, H, members, others, grey, cohort, s["vn"])


def build_dn_dlg():
    from common import to_gray
    from devnative import paths as P
    full = _devnative_lines()
    sel = _devnative_select_ids()
    reads = {}
    for x in (ART / "devnative" / "reads" / "reads.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        reads[r["shot"]] = r
    by_shot = {}
    for lid in sel:
        f = full[lid]
        lab = (sel[lid].get("ink") or sel[lid]["grad"])[1]
        by_shot.setdefault(f["shot"], {})[lid.rsplit("_", 1)[1]] = (lid, lab, "name" in (f.get("styles") or []))
    cases = []
    for shot, lab in sorted(by_shot.items()):
        s = reads[shot]
        grey = to_gray(np.ascontiguousarray(P.load_screen(s)))
        names = {k for k, v in lab.items() if v[2]}
        cases += _frame_units("dn_dlg", s, s["kind"], grey, {k: v[:2] for k, v in lab.items()},
                              lambda k, names=names: k in names)
    return cases


def build_own_dlg():
    from common import to_gray
    from devnative import paths as P
    reads = _owner_reads()
    by_shot = {}
    for r in _owner_kept():
        by_shot.setdefault(r["shot"], {})[r["id"].rsplit("_", 1)[1]] = (r["id"], r["label"], bool(r.get("name")))
    cases = []
    for shot, lab in sorted(by_shot.items()):
        s = reads[shot]
        grey = to_gray(np.ascontiguousarray(P.load_screen(s)))
        names = {k for k, v in lab.items() if v[2]}
        cases += _frame_units("own_dlg", s, s["kind"], grey, {k: v[:2] for k, v in lab.items()},
                              lambda k, names=names: k in names)
    return cases


def build_tv_dlg():
    import cv2
    import geometry as G

    from common import to_gray
    rows = load_npz(SELECT_POOLS / "teacher_val.npz")
    labelled = {}
    for m, lab, _ in rows:
        shot, k = m["id"].rsplit("_", 1)
        labelled.setdefault((shot, m["target"]), {})[k] = (m["id"], lab)
    shots = {s for s, _ in labelled}
    reads = {}
    for p in sorted((ART / "teacher" / "reads").glob("train.*of5.jsonl")):
        for x in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(x)
            except json.JSONDecodeError:
                continue
            if r["shot"] in shots:
                reads[r["shot"]] = r
    cases = []
    for (shot, target), lab in sorted(labelled.items()):
        s = reads[shot]
        path = ART / "realdata" / "shots" / s["kind"] / f"{s['vn']}_{s['url'].rsplit('/', 1)[-1]}"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        sc = G.target_scale(rgb.shape[1], rgb.shape[0], target) if s["kind"] == "hires" else 1.0
        grey = to_gray(np.ascontiguousarray(G.scale_image(rgb, sc)))
        cases += _frame_units("tv_dlg", s, target, grey, lab, lambda k, lab=lab: namelike(lab[k][1]), scale=sc)
    return cases


BUILDERS = {"dn_pairs": build_dn_pairs, "own_pairs": build_own_pairs, "tv_dlg": build_tv_dlg,
            "dn_dlg": build_dn_dlg, "own_dlg": build_own_dlg}


def cmd_cohorts(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = out / "cohorts.json"
    summary = json.loads(record.read_text()) if record.exists() else {}
    for name in args.only or BUILDERS:
        cases = BUILDERS[name]()
        with open(out / f"{name}.pkl", "wb") as f:
            pickle.dump(cases, f, protocol=4)
        if name.endswith("pairs"):
            s = {"lines": len(cases), "dialogue_lines": sum(not c["name"] for c in cases),
                 "chars": sum(len(norm(c["ref"])) for c in cases), "vns": len({c["vn"] for c in cases})}
        else:
            base = [c for c in cases if c["variant"] == "base"]
            s = {"cases": len(cases), "regions": len(base), "lines": sum(len(c["lines"]) for c in base),
                 "chars": sum(len(norm(x["text"])) for c in base for x in c["lines"]),
                 "vns": len({c["vn"] for c in cases}),
                 "variants": {v: sum(c["variant"] == v for c in cases) for v in VARIANTS}}
        s["sha256"] = sha256(out / f"{name}.pkl")
        summary[name] = s
        print(name, s, flush=True)
    record.write_text(json.dumps(dict(sorted(summary.items())), indent=1) + "\n")


# ---------------------------------------------------------------- the C path (finder, refinement, recognizer)

FINDER_SOURCES = ("runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.h",
                  "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h", "runtime/vocr_lines_refine_test.c")
FINDER_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror")
SCALAR_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror",
                "-Wpedantic", "-Wshadow", "-Wstrict-prototypes", "-Wmissing-prototypes", "-Wcast-align",
                "-Wpointer-arith", "-fPIC", "-shared")
CODE = ("eval/finder_tighten.py", "synth/lines_refine.py", "synth/common.py", "eval/metrics.py",
        "train/general_vn_textbox_guard.py", "runtime/vocr.c", "runtime/vocr.h", "runtime/pyvocr.py", *FINDER_SOURCES)
DEV_COHORTS = ("dn_pairs", "own_pairs", "tv_dlg", "dn_dlg", "own_dlg")
BOOT = {"reps": 10000, "seed": 0, "min_clusters": 5}
FORBIDDEN = {"text", "pred", "preds", "ref", "reference_text"}


PROTOCOL = ROOT / "eval" / "finder_tighten_protocol.json"


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dialogue_line(u):
    return not u["name"]


def code_identity():
    return {f: sha256(ROOT / f) for f in CODE}


def build(out, compiler="gcc"):
    """The finder + refinement driver and the scalar recognizer library, built now from this checkout."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    drv, lib = out / "vocr_lines_refine_test", out / "libvocr_scalar.so"
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
    subprocess.run([compiler, *FINDER_FLAGS, "-o", str(drv), "runtime/vocr_lines.c", "runtime/vocr_lines_refine.c",
                    "runtime/vocr_lines_refine_test.c"], check=True, cwd=ROOT, env=env)
    subprocess.run([compiler, *SCALAR_FLAGS, "-o", str(lib), "runtime/vocr.c", "-lm"], check=True, cwd=ROOT, env=env)
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return {"driver": str(drv), "driver_sha256": sha256(drv), "library": str(lib), "library_sha256": sha256(lib),
            "compiler": version, "finder_flags": list(FINDER_FLAGS), "scalar_flags": list(SCALAR_FLAGS)}


def c_refine(driver, images, flags, rows=False):
    """The C finder then vocr_refine_lines with `flags` on every grey image: [(found, boxes, dark, rows_read)]."""
    if not images:
        return []
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img in images:
            img = np.ascontiguousarray(img, dtype=np.uint8)
            if img.ndim != 2:
                raise ValueError("grey images only")
            h, w = img.shape
            f.write(struct.pack("<6I", w, h, 1, 0, w, flags))
            f.write(img.tobytes())
        path = f.name
    try:
        cmd = [str(driver)] + (["--rows"] if rows else []) + [path]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(out) != len(images):
        raise RuntimeError(f"driver returned {len(out)} results for {len(images)} images")
    res = []
    for line in out:
        v = [int(t) for t in line.split()]
        if v[0] != 0:
            raise RuntimeError(f"finder/refinement rc={v[0]}")
        n = v[2]
        res.append((v[1], [tuple(v[5 + 5 * i:9 + 5 * i]) for i in range(n)], [v[9 + 5 * i] for i in range(n)], v[4]))
    return res


def twin_lines(images, flags):
    """The Python twin on the same images: (boxes, dark, origin) each (origin maps a line to its finder line)."""
    import lines_refine as LR
    from common import find_lines_ex
    out = []
    for img in images:
        b, d, _ = find_lines_ex(img, "grad")
        out.append(LR.refine_lines(img, b, d, flags))
    return out


def lines_of(driver, images, flags):
    """C boxes, checked against the twin box for box (a real-data parity check on every case), plus the twin's
    origin of every line, and the rows the refinement read."""
    c = c_refine(driver, images, flags)
    t = twin_lines(images, flags)
    for i, ((found, boxes, dark, _), (tb, td, _)) in enumerate(zip(c, t)):
        if boxes != tb or dark != td:
            raise RuntimeError(f"case {i}: C and twin differ: {boxes} {dark} / {tb} {td}")
    return [(boxes, dark, org, rows, found) for (found, boxes, dark, rows), (_, _, org) in zip(c, t)]


class Reader:
    """The C runtime in deployment W8I (pyvocr, max width 1024) from an explicitly built library."""

    def __init__(self, blob, library):
        from pyvocr import Vocr, load_lib
        self.rt = Vocr(blob, mode="w8i", max_width=1024, lib=load_lib(library))
        self.sha256 = sha256(blob)
        self.blob_bytes, self.arena_bytes = self.rt.blob_bytes, self.rt.arena_bytes

    def read(self, g, invert):
        return norm(self.rt.text(np.ascontiguousarray(g), invert=bool(invert)))


def load_cohort(directory, name):
    with open(Path(directory) / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def score_cohort(name, cases, reader, modes):
    """Integer per-unit records of one cohort for one arm: {mode: [unit records]} (no text)."""
    from general_vn_textbox_guard import transcript_score
    out = {m: [] for m in ["ink"] * name.endswith("pairs") + list(modes)}
    for mode, lines in modes.items():
        for c, (boxes, dark, org, rows, found) in zip(cases, lines):
            img = c["image"]
            if name.endswith("pairs"):
                pred = "".join(reader.read(img[y:y + h, x:x + w], d)
                               for (x, y, w, h), d, o in zip(boxes, dark, org) if o == c["line"])
                ref = norm(c["ref"])
                out[mode].append({"unit": c["id"], "vn": c["vn"], "platform": c["kind"], "name": bool(c["name"]),
                                  "errors": lev(pred, ref), "chars": len(ref),
                                  "lines_out": sum(o == c["line"] for o in org)})
            else:
                preds = [reader.read(img[y:y + h, x:x + w], d) for (x, y, w, h), d in zip(boxes, dark)]
                sc = transcript_score(c["lines"], preds)
                out[mode].append({"unit": f"{c['unit']}|{c['variant']}", "shot": c["shot"], "vn": c["vn"],
                                  "platform": c["kind"], "variant": c["variant"],
                                  "errors": sc["all"]["full"]["errors"], "chars": sc["all"]["full"]["chars"],
                                  "hira_errors": sc["all"]["hira"]["errors"], "hira_chars": sc["all"]["hira"]["chars"],
                                  "detections": len(boxes), "finder_lines": found, "rows_read": rows})
    if name.endswith("pairs"):
        for c in cases:
            ref = norm(c["ref"])
            out["ink"].append({"unit": c["id"], "vn": c["vn"], "platform": c["kind"], "name": bool(c["name"]),
                               "errors": lev(reader.read(c["ink_crop"], False), ref), "chars": len(ref)})
    return out


def parse_arm(text):
    name, _, path = text.partition("=")
    if not name or not path:
        raise SystemExit(f"--arm NAME=BLOB, not {text!r}")
    return name, Path(path)


def cmd_score(args):
    """Every arm on every development cohort, GRAD (flags 0) and refined (--flags), through the C path."""
    import lines_refine as LR
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flags = LR.parse_flags(args.flags)
    ident = build(out / "build", args.compiler)
    meta = {"kind": "finder-tighten-dev-scores", "flags": flags, "native": ident, "code_sha256": code_identity(),
            "cohorts": {}, "arms": {}}
    lines = {}
    for name in args.cohort or DEV_COHORTS:
        cases = load_cohort(args.cohorts, name)
        imgs = [c["image"] for c in cases]
        lines[name] = {"grad": lines_of(ident["driver"], imgs, 0), "refine": lines_of(ident["driver"], imgs, flags)}
        for (b0, d0, _, r0, f0), c in zip(lines[name]["grad"], cases):
            if r0 != 0 or len(b0) != f0:
                raise RuntimeError(f"{name}: flags 0 is not the finder's output")
        meta["cohorts"][name] = {"sha256": sha256(Path(args.cohorts) / f"{name}.pkl"), "cases": len(cases)}
        print(f"{name}: {len(cases)} cases, C == twin on every case", flush=True)
    for spec in args.arm:
        arm, blob = parse_arm(spec)
        reader = Reader(blob, ident["library"])
        meta["arms"][arm] = {"blob": blob.name, "sha256": reader.sha256, "blob_bytes": reader.blob_bytes,
                             "arena_w8i": reader.arena_bytes}
        res = {}
        for name in lines:
            res[name] = score_cohort(name, load_cohort(args.cohorts, name), reader, lines[name])
        (out / f"units_{arm}.json").write_text(json.dumps(res) + "\n")
        print(f"{arm}: scored", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


# ---------------------------------------------------------------- report

def totals(units, keep=lambda u: True):
    sel = [u for u in units if keep(u)]
    return [sum(u["errors"] for u in sel), sum(u["chars"] for u in sel)]


def pct(t):
    return round(100 * t[0] / t[1], 3) if t[1] else None


def compare(cand, base, keep=lambda u: True):
    """CAND - BASE in points: unit and VN-cluster paired bootstraps (eval/vocr3_report's)."""
    import vocr3_report as R
    c = {u["unit"]: u for u in cand if keep(u)}
    b = {u["unit"]: u for u in base if keep(u)}
    if set(c) != set(b):
        raise ValueError("candidate and baseline units differ")
    a = {k: (c[k]["errors"], c[k]["chars"]) for k in c}
    bb = {k: (b[k]["errors"], b[k]["chars"]) for k in c}
    lines = R.paired_bootstrap(a, bb, BOOT["reps"], BOOT["seed"])
    vn = R.cluster_bootstrap(a, bb, {k: c[k]["vn"] for k in c}, BOOT["reps"], BOOT["seed"], BOOT["min_clusters"])
    return {"delta": lines["delta"], "units_ci95": lines["ci95"], "vn_ci95": vn.get("ci95"), "units": lines["units"],
            "vn_clusters": vn["clusters"], "chars": lines["chars"], "candidate_errors": lines["candidate_errors"],
            "baseline_errors": lines["baseline_errors"]}


def pooled_dialogue(res, tv_shots):
    """The primary whole-dialogue pool: tv_dlg + own_dlg + the dn_dlg frames teacher_val does not hold
    (teacher_val and devnative's tv_native share screenshots: never pooled twice)."""
    out = {}
    for mode in ("grad", "refine"):
        units = []
        for name in ("tv_dlg", "own_dlg", "dn_dlg"):
            if name not in res:
                continue
            for u in res[name][mode]:
                if name == "dn_dlg" and u["shot"] in tv_shots:
                    continue
                units.append(dict(u, unit=f"{name}|{u['unit']}"))
        out[mode] = units
    return out


def arm_report(res):
    r = {}
    for name, modes in res.items():
        if name.endswith("pairs"):
            e = {}
            for part, keep in (("dialogue", lambda u: not u["name"]), ("name", lambda u: u["name"]),
                               ("all", lambda u: True)):
                t = {m: totals(modes[m], keep) for m in ("ink", "grad", "refine")}
                gap = t["grad"][0] - t["ink"][0]
                e[part] = {**{m: {"errors": v[0], "chars": v[1], "cer": pct(v)} for m, v in t.items()},
                           "refine_minus_grad": compare(modes["refine"], modes["grad"], keep) if t["grad"][1] else None,
                           "gap_closed": round((t["grad"][0] - t["refine"][0]) / gap, 4) if gap > 0 else None}
            r[name] = e
        else:
            e = {}
            for v in list(VARIANTS) + ["all"]:
                keep = (lambda u: True) if v == "all" else (lambda u, v=v: u["variant"] == v)
                t = {m: totals(modes[m], keep) for m in ("grad", "refine")}
                e[v] = {**{m: {"errors": x[0], "chars": x[1], "cer": pct(x)} for m, x in t.items()},
                        "refine_minus_grad": compare(modes["refine"], modes["grad"], keep),
                        "detections": {m: sum(u["detections"] for u in modes[m] if keep(u))
                                       for m in ("grad", "refine")}}
            r[name] = e
    return r


def merged_meta(dirs):
    """The score directories' meta, merged: one flag set, native build, code and cohort set across them."""
    metas = [json.loads((Path(d) / "meta.json").read_text()) for d in dirs]
    first = metas[0]
    for m in metas[1:]:
        for k in ("flags", "code_sha256", "cohorts"):
            if m[k] != first[k]:
                raise SystemExit(f"score directories differ in {k}")
        if (m["native"]["driver_sha256"], m["native"]["library_sha256"]) != \
                (first["native"]["driver_sha256"], first["native"]["library_sha256"]):
            raise SystemExit("score directories were built differently")
    arms, where = {}, {}
    for d, m in zip(dirs, metas):
        for arm, v in m["arms"].items():
            if arm in arms:
                raise SystemExit(f"arm {arm} scored twice")
            arms[arm], where[arm] = v, Path(d)
    return dict(first, arms=arms), where


def cmd_report(args):
    meta, where = merged_meta(args.scores)
    report = {"kind": "finder-tighten-dev-report", "flags": meta["flags"], "native": meta["native"],
              "code_sha256": meta["code_sha256"], "cohorts": meta["cohorts"], "arms": meta["arms"],
              "bootstrap": BOOT, "results": {}, "pooled": {}}
    for arm in meta["arms"]:
        res = json.loads((where[arm] / f"units_{arm}.json").read_text())
        rep = arm_report(res)
        tv_shots = {u["shot"] for u in res.get("tv_dlg", {}).get("grad", [])}
        pool = pooled_dialogue(res, tv_shots)
        pd = {}
        for v in list(VARIANTS) + ["all"]:
            keep = (lambda u: True) if v == "all" else (lambda u, v=v: u["variant"] == v)
            t = {m: totals(pool[m], keep) for m in ("grad", "refine")}
            pd[v] = {**{m: {"errors": x[0], "chars": x[1], "cer": pct(x)} for m, x in t.items()},
                     "refine_minus_grad": compare(pool["refine"], pool["grad"], keep)}
        pairs = {m: [dict(u, unit=f"{n}|{u['unit']}") for n in ("dn_pairs", "own_pairs") if n in res for u in res[n][m]]
                 for m in ("ink", "grad", "refine")}
        tp = {m: totals(pairs[m], dialogue_line) for m in pairs}
        report["results"][arm] = rep
        report["pooled"][arm] = {
            "whole_dialogue": pd,
            "pairs_dialogue": {**{m: {"errors": x[0], "chars": x[1], "cer": pct(x)} for m, x in tp.items()},
                               "refine_minus_grad": compare(pairs["refine"], pairs["grad"], dialogue_line)},
            "dn_dlg_frames_not_in_tv": sum(1 for u in res.get("dn_dlg", {}).get("grad", [])
                                           if u["shot"] not in tv_shots)}
    # the mean over arms of the headline rates
    head = {}
    for arm, p in report["pooled"].items():
        rr = report["results"][arm]
        row = {"dn_pairs_dialogue": [rr["dn_pairs"]["dialogue"][m]["cer"] for m in ("grad", "refine", "ink")],
               "dn_pairs_all": [rr["dn_pairs"]["all"][m]["cer"] for m in ("grad", "refine", "ink")],
               "whole_dialogue": {v: [p["whole_dialogue"][v][m]["cer"] for m in ("grad", "refine")]
                                  for v in list(VARIANTS) + ["all"]}}
        head[arm] = row
    report["headline"] = head
    assert_public(report)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    if args.md:
        Path(args.md).write_text(markdown(report) + "\n")
    print(markdown(report))


def assert_public(value, where="report"):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN:
                raise ValueError(f"{where}: public output would contain text field {k!r}")
            assert_public(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            assert_public(v, f"{where}[{i}]")


def fmt_ci(c):
    if not c or c.get("delta") is None:
        return "-"
    vn = c.get("vn_ci95")
    vs = f"{vn[0]:+.2f} to {vn[1]:+.2f}" if vn else f"n/a ({c['vn_clusters']} VNs)"
    return f"{c['delta']:+.2f} (VN {vs})"


def markdown(rep):
    md = ["| arm | dn pairs dialogue: GRAD → refined (ink) | Δ (VN-cluster 95% CI) | gap closed | "
          "whole dialogue base | +4 | +8 | Δ all variants (VN CI) |", "|---|---|---|---|---|---|---|---|"]
    for arm in rep["arms"]:
        r, p = rep["results"][arm]["dn_pairs"]["dialogue"], rep["pooled"][arm]["whole_dialogue"]
        cells = [f"{p[v]['grad']['cer']:.2f} → {p[v]['refine']['cer']:.2f}" for v in VARIANTS]
        md.append(f"| {arm} | {r['grad']['cer']:.2f} → {r['refine']['cer']:.2f} ({r['ink']['cer']:.2f}) | "
                  f"{fmt_ci(r['refine_minus_grad'])} | {r['gap_closed']:.0%} | " + " | ".join(cells) +
                  f" | {fmt_ci(p['all']['refine_minus_grad'])} |")
    return "\n".join(md)


# ---------------------------------------------------------------- protected sets: report once

def protected_once(path, record):
    """Exclusive creation of the attempt record: a second run refuses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=1)


def cmd_protected(args):
    """vita-test and psp-test whole dialogue (eval's regions, base / +4 / +8; regions with an uncertain line left out
    whole), GRAD against refined, every arm, once. Fixed lines use no finder and are unchanged by the refinement."""

    import sets
    import vocr3_report as V
    from group_metrics import body  # noqa: F401  (transcript_score's body rule)

    import lines_refine as LR
    from common import to_gray
    from general_vn_textbox_guard import transcript_score
    out = Path(args.out)
    flags = LR.parse_flags(args.flags)
    protected_once(out / "protected.attempt.json", {"started": utc_now(), "flags": flags,
                                                     "protocol_sha256": sha256(PROTOCOL)})
    ident = build(out / "build", args.compiler)
    readers = {}
    for spec in args.arm:
        arm, blob = parse_arm(spec)
        readers[arm] = Reader(blob, ident["library"])
    units = {arm: {"grad": [], "refine": []} for arm in readers}
    for set_name in ("vita-test", "psp-test"):
        regions, excluded = V.dialogue_regions(set_name, args.data)
        cache = sets.ImageCache()
        cases = []
        for reg in regions:
            img = cache.get(reg["path"])
            dims = [img.shape[1], img.shape[0]]
            for variant, pad in VARIANTS.items():
                x, y, w, h = V.expand(reg["region"], pad, dims)
                cases.append((reg, variant, np.ascontiguousarray(to_gray(np.ascontiguousarray(img[y:y + h, x:x + w])))))
        imgs = [c[2] for c in cases]
        found = {"grad": lines_of(ident["driver"], imgs, 0), "refine": lines_of(ident["driver"], imgs, flags)}
        for mode, ls in found.items():
            for (reg, variant, img), (boxes, dark, _, rows, _) in zip(cases, ls):
                for arm, rd in readers.items():
                    preds = [rd.read(img[y:y + h, x:x + w], d) for (x, y, w, h), d in zip(boxes, dark)]
                    sc = transcript_score(reg["lines"], preds)
                    units[arm][mode].append({
                        "unit": f"{set_name}|{reg['region_id']}|{variant}", "set": set_name, "vn": reg["vn"],
                        "platform": reg["platform"], "variant": variant, "detections": len(boxes), "rows_read": rows,
                        **{f"{sc_}_{t}_{k}": sc[sc_][t][k] for sc_ in ("all", "body") for t in ("full", "hira")
                           for k in ("errors", "chars")}})
        print(f"{set_name}: {len(regions)} regions, {len(cases)} cases (excluded {excluded})", flush=True)
    result = {"kind": "finder-tighten-protected", "flags": flags, "native": ident, "code_sha256": code_identity(),
              "arms": {a: {"sha256": r.sha256, "blob_bytes": r.blob_bytes} for a, r in readers.items()},
              "bootstrap": BOOT, "sets": {}}
    for arm in readers:
        e = {}
        for scope in ("body", "all"):
            for v in list(VARIANTS) + ["all"]:
                for plat in ("vita", "psp", "pooled"):
                    def keep(u, v=v, plat=plat):
                        return (v == "all" or u["variant"] == v) and (plat == "pooled" or u["platform"] == plat)
                    cand = [dict(u, errors=u[f"{scope}_full_errors"], chars=u[f"{scope}_full_chars"])
                            for u in units[arm]["refine"]]
                    base = [dict(u, errors=u[f"{scope}_full_errors"], chars=u[f"{scope}_full_chars"])
                            for u in units[arm]["grad"]]
                    tg, tr = totals(base, keep), totals(cand, keep)
                    e[f"{scope}/{v}/{plat}"] = {"grad": {"errors": tg[0], "chars": tg[1], "cer": pct(tg)},
                                                "refine": {"errors": tr[0], "chars": tr[1], "cer": pct(tr)},
                                                "refine_minus_grad": compare(cand, base, keep)}
        result["sets"][arm] = e
    assert_public(result)
    (out / "protected.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "private").mkdir(exist_ok=True)
    (out / "private" / "units.json").write_text(json.dumps(units) + "\n")
    print(json.dumps({a: {k: (v["grad"]["cer"], v["refine"]["cer"]) for k, v in e.items() if k.endswith("/pooled")}
                      for a, e in result["sets"].items()}, indent=1))


def cmd_clannad(args):
    """CLANNAD's long run against owner-drive-ingest's frozen Claude references (eval/clannad_refs.py), once: every
    page's ROI from its native frame, the C finder, then GRAD's boxes and the refined boxes read by every arm. A
    reference line is the finder box its stored crop came from; its refined text is every refined line from that
    box, joined. Scored with clannad_refs' rules (metrics.norm, Levenshtein; whole-page bootstraps)."""

    import clannad_refs as C
    import vocr3_report as V
    from PIL import Image

    import lines_refine as LR
    from common import to_gray
    out = Path(args.out)
    work = Path(args.work)
    flags = LR.parse_flags(args.flags)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if sha256(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    protected_once(out / "clannad.attempt.json", {"started": utc_now(), "refs_sha256": freeze["refs_sha256"],
                                                   "protocol_sha256": sha256(PROTOCOL)})
    rows, meta = C.load_lines(work)
    session = Path(meta["session"])
    refs = {x["id"]: x for x in C.read_jsonl(work / "refs.jsonl")}
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)
    ident = build(out / "build", args.compiler)
    readers = {}
    for spec in args.arm:
        arm, blob = parse_arm(spec)
        readers[arm] = Reader(blob, ident["library"])
    pages, imgs = [], []
    for page, rs in sorted(by_page.items()):
        if not any(refs.get(r["id"], {}).get("status") == "ref" for r in rs):
            continue
        src = rs[0]["source"]
        path = session / src["path"]
        if sha256(path) != src["sha256"]:
            raise SystemExit(f"{src['path']}: frame hash differs from the capture journal")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        x, y, w, h = rs[0]["roi"]
        pages.append((page, rs))
        imgs.append(np.ascontiguousarray(to_gray(np.ascontiguousarray(rgb[y:y + h, x:x + w]))))
    grad = lines_of(ident["driver"], imgs, 0)
    refd = lines_of(ident["driver"], imgs, flags)
    preds = {arm: {} for arm in readers}
    matched = unmatched = 0
    for (page, rs), img, (gb, gd, _, _, _), (rb, rdk, org, _, _) in zip(pages, imgs, grad, refd):
        for r in rs:
            if refs.get(r["id"], {}).get("status") != "ref":
                continue
            box = tuple(r["box"])
            if box not in gb:
                unmatched += 1
                continue
            matched += 1
            k = gb.index(box)
            for arm, rd in readers.items():
                x, y, w, h = box
                preds[arm].setdefault("grad", {})[r["id"]] = rd.read(img[y:y + h, x:x + w], gd[k])
                preds[arm].setdefault("refine", {})[r["id"]] = "".join(
                    rd.read(img[yy:yy + hh, xx:xx + ww], dd) for (xx, yy, ww, hh), dd, o in zip(rb, rdk, org) if o == k)
    first = preds[next(iter(readers))].get("grad", {})
    scored = [x for x in refs.values() if x["status"] == "ref" and x["id"] in first]
    pg = [x["page"] for x in scored]
    ids = sorted(set(pg))
    idx = C.boot_idx(len(ids), C.REPS, C.SEED)
    result = {"kind": "finder-tighten-clannad", "flags": flags, "refs_sha256": freeze["refs_sha256"],
              "scored_lines": len(scored), "matched_finder_boxes": matched, "unmatched": unmatched,
              "pages": len(ids), "native": ident, "code_sha256": code_identity(),
              "arms": {a: {"sha256": r.sha256} for a, r in readers.items()}, "systems": {}, "pairs": {}}
    for arm in readers:
        sums = {}
        for mode in ("grad", "refine"):
            u = [C.line_unit(x["text"], preds[arm][mode][x["id"]]) for x in scored]
            sums[mode] = C.page_sums(u, pg)
            result["systems"][f"{arm}/{mode}"] = C.summary(sums[mode], ids, idx)
        a = {p: (sums["refine"][p][0], sums["refine"][p][1]) for p in ids}
        b = {p: (sums["grad"][p][0], sums["grad"][p][1]) for p in ids}
        cer = V.paired_bootstrap(a, b, C.REPS, C.SEED)
        result["pairs"][f"{arm}/refine:{arm}/grad"] = {"cer_delta": cer["delta"], "cer_ci95": cer["ci95"],
                                                      "candidate_errors": cer["candidate_errors"],
                                                      "baseline_errors": cer["baseline_errors"], "chars": cer["chars"]}
    assert_public(result)
    (out / "clannad.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: (v["cer"], v["cer_ci95"]) for k, v in result["systems"].items()}, indent=1))
    print(json.dumps(result["pairs"], indent=1))


# ---------------------------------------------------------------- A9 estimate of the refinement

def cmd_a9(args):
    """Cortex-A9 cycles of the refinement alone, as a9_cycles.py estimates the recognizer: the Vita compiler's machine
    code for vocr_lines_refine.c (arm-vita-eabi-gcc, Thumb-2, -O2) linked into the host driver, run under the
    plugin-enabled qemu-arm with the Vita's cache sizes on development regions, once with flags 0 (the finder
    alone) and once with --flags; the refinement's cost is the difference, per finder line. Estimates only."""
    sys.path.insert(0, str(ROOT / "vita"))
    import shutil

    import a9_cycles as A

    import lines_refine as LR
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vitasdk = args.vitasdk or os.environ.get("VITASDK")
    arm_cc = args.arm_cc or os.environ.get("ARM_CC") or A.default_arm_cc()
    q = Path(args.qemu) if args.qemu else A.default_qemu()
    flags = LR.parse_flags(args.flags)
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    objs = []
    for src in ("vocr_lines.c", "vocr_lines_refine.c"):
        o = out / src.replace(".c", "_vita.o")
        subprocess.run([str(vcc), "-std=c99", "-Wall", "-Wextra", "-Werror", *A.ARCH, "-O2", "-c", "-o", str(o),
                        str(ROOT / "runtime" / src)], check=True)
        objs.append(o)
    drv = out / "refine_driver_vitacc"
    subprocess.run([arm_cc, "-std=c99", "-Wall", "-Wextra", "-Werror", *A.ARCH, "-O2", "-static",
                    "-Wl,--no-enum-size-warning", "-o", str(drv), *map(str, objs),
                    str(ROOT / "runtime/vocr_lines_refine_test.c")], check=True, capture_output=True)
    objdump = arm_cc[:-3] + "objdump" if arm_cc.endswith("gcc") else "objdump"
    dis = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(drv)], check=True, capture_output=True,
                         text=True).stdout
    (out / "driver.dis").write_text(dis)
    rows = A.parse_dis(dis)
    with open(out / "classes.txt", "w") as f:
        for a, (_, mn, ops) in sorted(rows.items()):
            unit = A.classify(mn, ops)[0]
            if unit != "I":
                f.write(f"{a:x} {unit}\n")
    # development regions: every whole-dialogue base region of the cohorts (grey)
    imgs = []
    for name in args.cohort:
        imgs += [c["image"] for c in load_cohort(args.cohorts, name) if c.get("variant", "base") == "base"]
    imgs = imgs[:args.limit] if args.limit else imgs
    meta = {"regions": len(imgs), "flags": flags, "runs": {}}
    stage = Path(tempfile.mkdtemp(prefix="vocr-a9-", dir="/tmp"))
    try:
        shutil.copy2(drv, stage / drv.name)
        for tag, fl in (("off", 0), ("on", flags)):
            batch = stage / f"batch_{tag}.bin"
            with open(batch, "wb") as f:
                for img in imgs:
                    h, w = img.shape
                    f.write(struct.pack("<6I", w, h, 1, 0, w, fl))
                    f.write(np.ascontiguousarray(img).tobytes())
            plugins = ["-plugin", f"{q}/plugins/libinsn_mix.so,out={out}/counts_{tag}.txt,classes={out}/classes.txt",
                       "-plugin", f"{q}/plugins/libcache.so,{A.CACHE_ARGS}", "-d", "plugin", "-D",
                       str(out / f"cache_{tag}.txt")]
            p = subprocess.run([str(q / "bin/qemu-arm"), "-cpu", "cortex-a9", "-seed", "0", *plugins,
                                f"./{drv.name}", batch.name], capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=7200, env={}, cwd=stage)
            if p.returncode:
                raise SystemExit(f"qemu run failed: {p.stderr[-2000:]}")
            res = [ln.split() for ln in p.stdout.splitlines()]
            meta["runs"][tag] = {"finder_lines": sum(int(v[1]) for v in res), "lines_out": sum(int(v[2]) for v in res),
                                 "rows_read": sum(int(v[4]) for v in res)}
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    c_off, x_off = A.read_counts(out / "counts_off.txt")
    c_on, x_on = A.read_counts(out / "counts_on.txt")
    delta = {a: c_on.get(a, 0) - c_off.get(a, 0) for a in set(c_on) | set(c_off)}
    neg = {a: v for a, v in delta.items() if v < 0}
    delta = {a: v for a, v in delta.items() if v > 0}
    extra = {k: x_on.get(k, 0) - x_off.get(k, 0) for k in set(x_on) | set(x_off)}
    cache_on, cache_off = A.read_cache(out / "cache_on.txt"), A.read_cache(out / "cache_off.txt")
    cache = {k: max(0, cache_on[k] - cache_off[k]) for k in cache_on} if cache_on and cache_off else None
    funcs = {fn for fn, _, _ in rows.values()}
    lines = meta["runs"]["off"]["finder_lines"]
    r = A.model_mode(rows, delta, extra, funcs, lines, cache)
    r["per_finder_line_ms"] = {mhz: [c / (mhz * 1000) for c in r["cycles_per_frame"][mhz]] for mhz in A.CLOCKS}
    meta.update(per_line=r, negative_count_addresses=len(neg),
                note="cycles per finder line (frames = finder lines); a host qemu estimate, not a console measurement")
    (out / "a9.json").write_text(json.dumps(meta, indent=1) + "\n")
    lo, hi = r["per_finder_line_ms"][444]
    lo3, hi3 = r["per_finder_line_ms"][333]
    print(f"{len(imgs)} regions, {lines} finder lines -> {meta['runs']['on']['lines_out']} refined lines; "
          f"{r['insns_per_frame']:,.0f} instructions per finder line; est. {lo:.3f}-{hi:.3f} ms per line at 444 MHz, "
          f"{lo3:.3f}-{hi3:.3f} ms at 333 MHz; rows read per finder line "
          f"{meta['runs']['on']['rows_read'] / max(1, lines):.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cohorts")
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", choices=list(BUILDERS))
    p.set_defaults(fn=cmd_cohorts)
    p = sub.add_parser("score")
    p.add_argument("--cohorts", required=True)
    p.add_argument("--cohort", action="append", choices=DEV_COHORTS)
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--flags", default="default")
    p.add_argument("--compiler", default="gcc")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("report")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--md")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("protected")
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--flags", default="default")
    p.add_argument("--compiler", default="gcc")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_protected)
    p = sub.add_parser("clannad")
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--flags", default="default")
    p.add_argument("--compiler", default="gcc")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_clannad)
    p = sub.add_parser("a9")
    p.add_argument("--cohorts", required=True)
    p.add_argument("--cohort", action="append", default=[])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--flags", default="default")
    p.add_argument("--vitasdk")
    p.add_argument("--arm-cc")
    p.add_argument("--qemu")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_a9)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

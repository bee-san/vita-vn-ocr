#!/usr/bin/env python3
"""head2level: the development scoring of the two-level head trial (eval/head2level_protocol.json "development").

  head2level_dev.py lines --cache DIR
      the C GRAD finder's lines (flags 0) and the C refinement's (VOCR_REFINE_DEFAULT) on wd's regions, once
      (eval/finder_tighten.py lines_of: C == twin box for box), -> DIR/wd_lines.pkl
  head2level_dev.py read --tier N --arm NAME=BLOB[:two_level][:gate] --cache DIR [--procs 5]
      every development crop through one arm: tv and dn as stored (rv) and in screen polarity (rvn), wd's finder
      lines with their dark flag as VOCR_INVERT. Per crop the greedy text, Rubi's line confidence, Rubi's LM text
      (counter re-decode {6000, 4000}, then beam 8 + the 1 MB LM), frames, escaped frames and the per-frame
      argmax -> DIR/reads/<tier>/<arm>.npz + .json
  head2level_dev.py score --cache DIR --fits FIT.json ... --out RESULTS.json
      the decodes (greedy; rubi = counters(LM text); rubi_cascade = the next tier's same arm on a line whose
      confidence is below 700), the score files of eval/vocr5_select.py's formats, its comparisons (VN-cluster
      bootstraps, seeds 0 / 1 / 2), rare-character recall and its bootstrap, escape rates, and the rule

Development data only: VOCR5's fixed development sample (reserved.json dev_sample) as eval/headgate_protocol.json
"development" "cohorts" pins it. Nothing here reads vita-test, psp-test or CLANNAD. Reads hold texts of public
VNDB-screenshot cohorts only and stay under the artifacts directory (never committed).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "runtime", ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
A = HERD / "shared" / "artifacts"
POOLS = A / "owner-drive-ingest-2" / "reclaim" / "pools"
WDC = A / "vocr5-train" / "cohorts"
LM_FILE = A / "vocr5-train" / "release" / "v0.3.0" / "vocr_lm_o3_1000k.vlm"
LM_SHA = "993000611555171daf6361e29a503b0871a5f3dd240a19555d55f16aaa3ab3d9"
COHORT_SHA = {"teacher_val_dev.npz": "e2f0300fd9f417b6c3c01e1dab36641bfd72749e2cac5db896226ef0d29b65dd",
              "devnative_tv_native_ink_dev.npz": "c198c9ea147620e2f75b929dbb676eda30cd10137b2b3c14eae17f28204bc0c4",
              "devnative_tv_native_grad_dev.npz": "4bda0139ae52f6d472513e96d2f79d30cc25be96dbc2d2316b3bce22f5e4702a",
              "devnative_strict_ink_dev.npz": "ad4207cdfc6f3e667ffe776a80ac3f97fe58fc3e27dab6285a621e35e24c7645",
              "devnative_strict_grad_dev.npz": "5e5b147254f9e65ccd9d904d79db68224b8fd26e902a0b167dff005b725891a1",
              "tv_dlg.pkl": "f1007ab4e943012330b229cc2630f3fc9d33aa94d6dac03130bc5ea5126c1af3",
              "dn_dlg.pkl": "c6dfd35293e7716bc3772fffc88c8d8aeba62c8fa20f97c0d5041b4f1dbc1e1f"}
DN_FILES = ["devnative_tv_native_ink_dev", "devnative_tv_native_grad_dev", "devnative_strict_ink_dev",
            "devnative_strict_grad_dev"]
RUBI_LM = dict(beam=8, topk=8, alpha=500, beta=3500, gate=8000, flags=2, bracket=8000)
COUNTER = dict(tau=6000, open=4000)
CASCADE_CONF = 700
NEXT = {"S": "N", "N": "L", "L": None}
DECODES = ("greedy", "rubi", "rubi_cascade")
SEEDS = {"tv": 0, "dn": 1, "wd": 2}
WEIGHTS = {"tv": 0.25, "dn": 0.25, "wd": 0.5}
N_BOOT = 10000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_cohorts():
    files = {POOLS / "teacher_val_dev.npz": COHORT_SHA["teacher_val_dev.npz"]}
    files.update({POOLS / f"{f}.npz": COHORT_SHA[f"{f}.npz"] for f in DN_FILES})
    files.update({WDC / f: COHORT_SHA[f] for f in ("tv_dlg.pkl", "dn_dlg.pkl")})
    bad = [str(f) for f, h in files.items() if sha256(f) != h]
    if bad:
        raise SystemExit(f"cohort files differ from the protocol: {bad}")
    if sha256(LM_FILE) != LM_SHA:
        raise SystemExit("the LM differs from the protocol")


def line_conf(conf_min, greedy):
    """Rubi core/vocr_ocr.c line_conf: round(conf_min x 1000 + 0.5) clamped, 1000 for an empty greedy line."""
    if not greedy:
        return 1000
    c = np.float32(conf_min) * np.float32(1000.0) + np.float32(0.5)
    return 0 if c <= 0 else 1000 if c >= 1000 else int(c)


# ---------------------------------------------------------------- crops

def fixed_rows():
    """tv and dn rows (eval/synth3_ablation.load_dev), as {cohort file stem: rows}."""
    from synth3_ablation import load_dev
    out = {"teacher_val_dev": load_dev(POOLS / "teacher_val_dev.npz")}
    for f in DN_FILES:
        out[f] = load_dev(POOLS / f"{f}.npz")
    return out


def cmd_lines(a):
    import finder_tighten as FT

    import lines_refine as LR
    check_cohorts()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    ident = FT.build(cache / "build")
    flags = LR.parse_flags("default")
    out = {"flags": flags, "native": {k: v for k, v in ident.items()}, "cohorts": {}}
    for name in ("tv_dlg", "dn_dlg"):
        cases = FT.load_cohort(WDC, name)
        imgs = [c["image"] for c in cases]
        out["cohorts"][name] = {"grad": FT.lines_of(ident["driver"], imgs, 0),
                                "refine": FT.lines_of(ident["driver"], imgs, flags)}
        print(f"{name}: {len(cases)} cases, C == twin on every case", flush=True)
    with open(cache / "wd_lines.pkl", "wb") as f:
        pickle.dump(out, f)


def all_crops(cache):
    """[(key, image, invert)] of every development read, in a fixed order."""
    import finder_tighten as FT
    crops = []
    for stem, rows in fixed_rows().items():
        for i, r in enumerate(rows):
            g = r["img"]
            crops.append((f"{stem}|{i}|rv", g, False))
            crops.append((f"{stem}|{i}|rvn", 255 - g if r["inverted"] else g, False))
    with open(Path(cache) / "wd_lines.pkl", "rb") as f:
        wl = pickle.load(f)
    for name in ("tv_dlg", "dn_dlg"):
        cases = FT.load_cohort(WDC, name)
        for mode in ("grad", "refine"):
            for ci, (c, (boxes, dark, org, rows, found)) in enumerate(zip(cases, wl["cohorts"][name][mode])):
                img = c["image"]
                for li, ((x, y, w, h), d) in enumerate(zip(boxes, dark)):
                    crops.append((f"{name}|{mode}|{ci}|{li}", np.ascontiguousarray(img[y:y + h, x:x + w]), bool(d)))
    return crops


# ---------------------------------------------------------------- reads

W = {}


def parse_arm(spec):
    name, _, rest = spec.partition("=")
    parts = rest.split(":")
    return name, parts[0], "two_level" in parts[1:], "gate" in parts[1:]


def _init(blob, two, gate):
    from pyvocr import Vocr, VocrLM
    kw = {"two_level": 1} if two else {}
    if gate:
        kw["head_gate"] = 1
    W["rt"] = Vocr(blob, mode="w8i", max_width=1024, **kw)
    W["lm"] = VocrLM(W["rt"], str(LM_FILE), **RUBI_LM)
    W["two"] = two


def _read(job):
    i, img, inv = job
    rt, lm = W["rt"], W["lm"]
    out = rt.run_topk(img, 8, invert=inv, esc=W["two"])
    g = out["text"]
    conf = line_conf(out["conf_min"], g)
    t = ""
    if g:
        o2, _ = rt.redecode(out, tau=COUNTER["tau"], open=COUNTER["open"])
        t = lm.decode(o2["topv"], o2["topc"], o2["blank"])
    return i, g, t, conf, int(out["frames"]), int(out["escaped"]), out["best"].astype(np.uint16)


def cmd_read(a):
    check_cohorts()
    t0 = time.time()
    name, blob, two, gate = parse_arm(a.arm)
    out_dir = Path(a.cache) / "reads" / a.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = all_crops(a.cache)
    if a.limit:
        crops = crops[:a.limit]
    res = [None] * len(crops)
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(blob, two, gate)) as pool:
        for n, r in enumerate(pool.imap_unordered(_read, [(i, c[1], c[2]) for i, c in enumerate(crops)],
                                                  chunksize=8)):
            res[r[0]] = r
            if n % 2000 == 0:
                print(f"  {a.tier} {name}: {n}/{len(crops)} {time.time() - t0:.0f}s", flush=True)
    off = np.zeros(len(res) + 1, np.int64)
    off[1:] = np.cumsum([len(r[6]) for r in res])
    np.savez_compressed(out_dir / f"{name}.npz", off=off, best=np.concatenate([r[6] for r in res]),
                        frames=np.array([r[4] for r in res]), escaped=np.array([r[5] for r in res]),
                        conf=np.array([r[3] for r in res]))
    from pyvocr import Vocr
    kw = {"two_level": 1} if two else {}
    if gate:
        kw["head_gate"] = 1
    rt = Vocr(blob, mode="w8i", max_width=1024, **kw)
    meta = {"tier": a.tier, "arm": name, "blob": blob, "blob_sha256": sha256(blob), "blob_bytes": rt.blob_bytes,
            "arena_w8i": rt.arena_bytes, "two_level": two, "gate": gate, "crops": len(crops),
            "keys": [c[0] for c in crops], "greedy": [r[1] for r in res], "lm": [r[2] for r in res],
            "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("keys", "greedy", "lm")}), flush=True)


# ---------------------------------------------------------------- decodes and score files

def load_reads(cache, tier, arm):
    d = Path(cache) / "reads" / tier
    meta = json.loads((d / f"{arm}.json").read_text(encoding="utf-8"))
    z = np.load(d / f"{arm}.npz")
    meta.update(off=z["off"], best=z["best"], frames=z["frames"], escaped=z["escaped"], conf=z["conf"])
    meta["index"] = {k: i for i, k in enumerate(meta["keys"])}
    return meta


def decoded(reads, nxt, lib):
    """{decode: [text per crop]} of an arm (nxt: the next tier's reads of the same arm kind, or None)."""
    from pyvocr import post_counters
    rubi = [post_counters(t, lib)[0] for t in reads["lm"]]
    if nxt is None:
        casc = rubi
    else:
        if nxt["keys"] != reads["keys"]:
            raise SystemExit("the cascade's reads hold other crops")
        casc = [post_counters(nt, lib)[0] if c < CASCADE_CONF else r
                for r, c, nt in zip(rubi, reads["conf"], nxt["lm"])]
    return {"greedy": list(reads["greedy"]), "rubi": rubi, "rubi_cascade": casc,
            "cascaded": int(sum(1 for c in reads["conf"] if c < CASCADE_CONF)) if nxt is not None else 0}


def write_scores(cache, tier, model, reads, texts):
    """The score files of one model (an arm and a decode) in eval/vocr5_select.py's input formats:
    synth3_ablation score JSONs (tv, the four dn files) and a finder_tighten units directory (wd)."""
    import finder_tighten as FT
    from metrics import lev, norm

    from general_vn_textbox_guard import transcript_score
    sdir = Path(cache) / "scores" / tier
    sdir.mkdir(parents=True, exist_ok=True)
    idx = reads["index"]
    paths = {"tv": [], "dn": [], "wd": None}
    for stem, rows in fixed_rows_cached().items():
        preds = {v: [norm(texts[idx[f"{stem}|{i}|{v}"]]) for i in range(len(rows))] for v in ("rv", "rvn")}
        errs = {v: [lev(p, r["ref"]) for p, r in zip(preds[v], rows)] for v in ("rv", "rvn")}
        n = sum(len(r["ref"]) for r in rows)
        f = POOLS / f"{stem}.npz"
        doc = {"kind": "synth3-dev-scores", "dev": {"file": str(f), "sha256": COHORT_SHA[f.name], "lines": len(rows),
                                                    "chars": n, "vns": len({r["vn"] for r in rows})},
               "lines": [{"id": r["id"], "vn": r["vn"], "ref": r["ref"], "tags": r["tags"]} for r in rows],
               "models": {model: {"blob": reads["blob"], "blob_sha256": reads["blob_sha256"],
                                  "blob_bytes": reads["blob_bytes"], "arena_bytes": reads["arena_w8i"],
                                  "pred": preds, "errors": errs,
                                  "cer": {k: round(sum(v) / n, 5) for k, v in errs.items()}}}}
        p = sdir / (f"tv_{model}.json" if stem == "teacher_val_dev" else f"dn_{stem}_{model}.json")
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        paths["tv" if stem == "teacher_val_dev" else "dn"].append(str(p))
    with open(Path(cache) / "wd_lines.pkl", "rb") as f:
        wl = pickle.load(f)
    res = {}
    for name in ("tv_dlg", "dn_dlg"):
        cases = load_cohort_cached(name)
        res[name] = {}
        for mode in ("grad", "refine"):
            units = []
            for ci, (c, (boxes, dark, org, rows_read, found)) in enumerate(zip(cases, wl["cohorts"][name][mode])):
                preds = [norm(texts[idx[f"{name}|{mode}|{ci}|{li}"]]) for li in range(len(boxes))]
                sc = transcript_score(c["lines"], preds)
                units.append({"unit": f"{c['unit']}|{c['variant']}", "shot": c["shot"], "vn": c["vn"],
                              "platform": c["kind"], "variant": c["variant"], "errors": sc["all"]["full"]["errors"],
                              "chars": sc["all"]["full"]["chars"], "hira_errors": sc["all"]["hira"]["errors"],
                              "hira_chars": sc["all"]["hira"]["chars"], "detections": len(boxes),
                              "finder_lines": found, "rows_read": rows_read})
            res[name][mode] = units
    wdir = sdir / "wd" / model
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / f"units_{model}.json").write_text(json.dumps(res) + "\n")
    meta = {"kind": "finder-tighten-dev-scores", "flags": wl["flags"], "native": wl["native"],
            "code_sha256": FT.code_identity(), "cohorts": {n: {"sha256": COHORT_SHA[f"{n}.pkl"]} for n in res},
            "arms": {model: {"blob": Path(reads["blob"]).name, "sha256": reads["blob_sha256"],
                             "blob_bytes": reads["blob_bytes"], "arena_w8i": reads["arena_w8i"]}}}
    (wdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    paths["wd"] = str(wdir)
    return paths


_FIXED, _CASES = {}, {}


def fixed_rows_cached():
    if not _FIXED:
        _FIXED.update(fixed_rows())
    return _FIXED


def load_cohort_cached(name):
    import finder_tighten as FT
    if name not in _CASES:
        _CASES[name] = FT.load_cohort(WDC, name)
    return _CASES[name]


# ---------------------------------------------------------------- rare characters

def align_recall(pred, ref, rare):
    """(rare characters of ref, of them recalled): a minimum-edit alignment of pred to ref, backtrace preference
    match or substitution, then deletion, then insertion; a rare reference character is recalled when it is
    matched to the same character."""
    n, m = len(ref), len(pred)
    if not any(c in rare for c in ref):
        return 0, 0
    D = [list(range(m + 1))]
    for i in range(1, n + 1):
        ri, prev, row = ref[i - 1], D[i - 1], [i]
        for j in range(1, m + 1):
            row.append(min(prev[j - 1] + (ri != pred[j - 1]), prev[j] + 1, row[j - 1] + 1))
        D.append(row)
    i, j, tot, hit = n, m, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != pred[j - 1]):
            if ref[i - 1] in rare:
                tot += 1
                hit += ref[i - 1] == pred[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            tot += ref[i - 1] in rare
            i -= 1
        else:
            j -= 1
    return tot, hit


def rare_set(blob, L1):
    """Normalised characters some class of the blob emits but none of its first L1 (level-1) classes."""
    from metrics import norm

    import export
    vocab = export.read_blob(Path(blob).read_bytes())["vocab"]
    emit = [norm(v) for v in vocab]
    lvl1 = {e for e in emit[1:L1] if len(e) == 1}
    return {e for e in emit[1:] if len(e) == 1} - lvl1


def rare_rows(cache, tier, texts, reads, rare):
    """Per component: (rare counts, recalled counts, VN per row) over its dialogue rows / wd units, in the row
    order of eval/vocr5_select.py's components (tv, dn: both views pooled per row; wd: pooled_dialogue units)."""
    from metrics import norm
    idx = reads["index"]
    out = {}
    names = {}
    for stem in DN_FILES:
        z = np.load(POOLS / f"{stem}.npz")
        names.update({m["id"]: "name" in (m.get("styles") or []) for m in json.loads(str(z["meta"]))})
    for comp, stems in (("tv", ["teacher_val_dev"]), ("dn", DN_FILES)):
        tot, hit, vns = [], [], []
        for stem in sorted(stems, key=lambda s: str(POOLS / f"{s}.npz")):
            for i, r in enumerate(fixed_rows_cached()[stem]):
                is_name = ("namelike" in r["tags"]) if comp == "tv" else names[r["id"]]
                if is_name:
                    continue
                t = h = 0
                for v in ("rv", "rvn"):
                    a, b = align_recall(norm(texts[idx[f"{stem}|{i}|{v}"]]), r["ref"], rare)
                    t, h = t + a, h + b
                tot.append(t)
                hit.append(h)
                vns.append(r["vn"])
        out[comp] = (np.array(tot, float), np.array(hit, float), np.array(vns))
    with open(Path(cache) / "wd_lines.pkl", "rb") as f:
        wl = pickle.load(f)
    tv_shots = {c["shot"] for c in load_cohort_cached("tv_dlg")}
    tot, hit, vns = [], [], []
    for name in ("tv_dlg", "dn_dlg"):
        for mode in ("grad", "refine"):
            for ci, (c, (boxes, *_)) in enumerate(zip(load_cohort_cached(name), wl["cohorts"][name][mode])):
                if name == "dn_dlg" and c["shot"] in tv_shots:
                    continue
                pred = "".join(norm(texts[idx[f"{name}|{mode}|{ci}|{li}"]]) for li in range(len(boxes)))
                ref = "".join(norm(ln["text"]) for ln in c["lines"])
                a, b = align_recall(pred, ref, rare)
                tot.append(a)
                hit.append(b)
                vns.append(c["vn"])
    out["wd"] = (np.array(tot, float), np.array(hit, float), np.array(vns))
    return out


def recall_compare(ref_rr, arm_rr, n_boot=N_BOOT):
    """Rare-character recall loss (reference - arm) in points per component and combined (0.25 / 0.25 / 0.5),
    VN-cluster bootstrap of ratios of sums, seeds 0 / 1 / 2."""
    res, point, reps = {}, 0.0, 0.0
    for comp in ("tv", "dn", "wd"):
        t0, h0, v0 = ref_rr[comp]
        t1, h1, v1 = arm_rr[comp]
        assert (t0 == t1).all() and (v0 == v1).all()
        ug, inv = np.unique(v0, return_inverse=True)
        gt = np.bincount(inv, weights=t0, minlength=len(ug))
        g0 = np.bincount(inv, weights=h0, minlength=len(ug))
        g1 = np.bincount(inv, weights=h1, minlength=len(ug))
        p = (h0.sum() - h1.sum()) / max(1.0, t0.sum()) * 100
        ix = np.random.default_rng(SEEDS[comp]).integers(0, len(ug), size=(n_boot, len(ug)))
        den = np.maximum(gt[ix].sum(1), 1.0)
        r = (g0[ix].sum(1) - g1[ix].sum(1)) / den * 100
        lo, hi = np.percentile(r, [2.5, 97.5])
        res[comp] = {"rare_chars": int(t0.sum()), "recall_ref": round(float(h0.sum() / max(1, t0.sum())), 5),
                     "recall_arm": round(float(h1.sum() / max(1, t1.sum())), 5), "loss_points": round(float(p), 3),
                     "ci95": [round(float(lo), 3), round(float(hi), 3)]}
        point, reps = point + WEIGHTS[comp] * p, reps + WEIGHTS[comp] * r
    lo, hi = np.percentile(reps, [2.5, 97.5])
    res["combined"] = {"loss_points": round(float(point), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}
    return res


def comps_for(paths_by_model):
    """eval/vocr5_select.py components (tv, dn, wd) over every model's score files."""
    import vocr5_select as VS
    ns = argparse.Namespace(tv=[p for m in paths_by_model.values() for p in m["tv"]],
                            dn=[p for m in paths_by_model.values() for p in m["dn"]],
                            wd=[m["wd"] for m in paths_by_model.values()])
    return VS.load(ns)


def crop_groups(keys, cache):
    """{group: crop indices} for the escape statistics: component (tv, dn, wd), wd's finder-cut lines by mode,
    platform (as stored rows' kind; wd cases' kind), dialogue only (tv: not name-like; dn: no name style)."""
    names = {}
    for stem in DN_FILES:
        z = np.load(POOLS / f"{stem}.npz")
        names.update({m["id"]: "name" in (m.get("styles") or []) for m in json.loads(str(z["meta"]))})
    kinds = {}
    for stem in ["teacher_val_dev"] + DN_FILES:
        z = np.load(POOLS / f"{stem}.npz")
        kinds[stem] = [m["kind"] for m, lab in zip(json.loads(str(z["meta"])), json.loads(str(z["labels"]))) if lab]
    g = {}

    def add(k, i):
        g.setdefault(k, []).append(i)
    cases = {n: load_cohort_cached(n) for n in ("tv_dlg", "dn_dlg")}
    for i, k in enumerate(keys):
        p = k.split("|")
        if p[0] in ("tv_dlg", "dn_dlg"):
            c = cases[p[0]][int(p[2])]
            add("all", i)
            add("dialogue", i)
            add("wd", i)
            add(f"wd|{p[1]}", i)
            add(f"dialogue|platform:{c['kind']}", i)
            continue
        stem, r = p[0], fixed_rows_cached()[p[0]][int(p[1])]
        comp = "tv" if stem == "teacher_val_dev" else "dn"
        add("all", i)
        is_name = ("namelike" in r["tags"]) if comp == "tv" else names[r["id"]]
        if is_name:
            add("names", i)
            continue
        add("dialogue", i)
        add(comp, i)
        kind = kinds[stem][int(p[1])]
        add(f"dialogue|platform:{ {'vita': 'vita', 'psp': 'psp', 'hires': 'pc'}.get(kind, kind)}", i)
    return {k: np.array(v) for k, v in g.items()}


def perm_ranks(released_blob, perm_blob):
    """rank[c] = the permuted class of released class c."""
    import export
    v0 = export.read_blob(Path(released_blob).read_bytes())["vocab"]
    v1 = export.read_blob(Path(perm_blob).read_bytes())["vocab"]
    pos = {c: i for i, c in enumerate(v1)}
    return np.array([pos[c] for c in v0], np.int64)


def escape_stats(reads, groups, today=None, ranks=None, L1=None, Vpad=None):
    """Per group: frames, the escape's fire share among the frames level 1 ran on (no gate: every frame), head
    rows per frame, and (today's argmax mapped to the permuted classes) the oracle share outside level 1."""
    esc, frames = reads["escaped"], reads["frames"]
    out = {}
    for gname, idx in groups.items():
        F = int(frames[idx].sum())
        e = {"crops": int(len(idx)), "frames": F, "escaped": int(esc[idx].sum()),
             "fire_share": round(float(esc[idx].sum() / max(1, F)), 5)}
        if L1 is not None and Vpad is not None:
            e["head_rows_per_frame"] = round(L1 + e["fire_share"] * (Vpad - L1), 1)
        if today is not None and ranks is not None:
            fr = np.concatenate([np.arange(today["off"][i], today["off"][i + 1]) for i in idx]) if len(idx) else []
            b = ranks[today["best"][fr].astype(np.int64)] if len(fr) else np.zeros(0, np.int64)
            e["oracle_share"] = round(float((b >= L1).mean()), 5) if len(b) else None
            e["frames_nonblank_share"] = round(float((b != 0).mean()), 5) if len(b) else None
        out[gname] = e
    return out


def cmd_score(a):
    import vocr5_select as VS
    from pyvocr import load_lib
    lib = load_lib()
    cache = Path(a.cache)
    fits = {t: json.loads(Path(f).read_text()) for t, f in (x.split("=", 1) for x in a.fits)}
    timing = json.loads(Path(a.timing).read_text()) if a.timing else None
    tiers = [t for t in ("S", "N", "L") if (cache / "reads" / t).exists()]
    arms = {t: sorted(p.stem for p in (cache / "reads" / t).glob("*.json")) for t in tiers}
    reads = {t: {arm: load_reads(cache, t, arm) for arm in arms[t]} for t in tiers}
    res = {"kind": "head2level-dev-results", "protocol": "eval/head2level_protocol.json",
           "protocol_sha256": sha256(ROOT / "eval" / "head2level_protocol.json"), "tiers": {}}
    rel = A / "vocr5-train" / "release" / "v0.3.0"
    for t in tiers:
        t0 = time.time()
        paths, texts_by = {}, {}
        for arm in arms[t]:
            nt = NEXT[t]
            nxt = reads[nt].get(arm) if nt in reads else None
            if nt is not None and nxt is None:
                print(f"{t} {arm}: no {nt} reads for the cascade; rubi_cascade not scored", flush=True)
            dec = decoded(reads[t][arm], nxt, lib)
            texts_by[arm] = dec
            for d in DECODES:
                if d == "rubi_cascade" and nt is not None and nxt is None:
                    continue
                paths[f"{arm}__{d}"] = write_scores(cache, t, f"{arm}__{d}", reads[t][arm], dec[d])
        comps = comps_for(paths)
        groups = crop_groups(reads[t]["today"]["keys"], cache)
        ranks = perm_ranks(rel / f"VOCR5_{t}_w8.vocr", A / "head-2level" / "blobs" / f"VOCR5_{t}_perm.vocr")
        tier = {"arms": {}, "scores": {}}
        rr_cache = {}
        for m in paths:
            tier["scores"][m] = {"score": round(VS.score_of(comps, m), 5),
                                 **{f"{n}_cer": round(c.rate([m]), 6) for n, c in comps.items()},
                                 **{f"{n}_names_cer": round(c.rate([m], "names"), 6) for n, c in comps.items()
                                    if n != "wd"}}
        for arm in arms[t]:
            info = {"blob": reads[t][arm]["blob"], "blob_sha256": reads[t][arm]["blob_sha256"],
                    "blob_bytes": reads[t][arm]["blob_bytes"], "arena_w8i": reads[t][arm]["arena_w8i"],
                    "two_level": reads[t][arm]["two_level"], "gate": reads[t][arm]["gate"],
                    "cascaded_lines": texts_by[arm]["cascaded"]}
            refs = ["today"] + (["gate"] if reads[t][arm]["gate"] and arm != "gate" and "gate" in arms[t] else [])
            if arm in ("today", "gate"):
                info["escape"] = escape_stats(reads[t][arm], groups)
                tier["arms"][arm] = info
                continue
            K = int(arm.split("_K")[1].split("_")[0]) if "_K" in arm and "_Kalt" not in arm else \
                fits[t]["K_alt"]
            L1 = 8 * ((K + 8) // 8)
            info.update(K=K, L1=L1, rare_variant=arm.rsplit("_", 1)[-1])
            info["escape"] = escape_stats(reads[t][arm], groups, reads[t]["today"], ranks, L1,
                                          8 * ((len(ranks) + 7) // 8))
            rare = rare_set(reads[t][arm]["blob"], L1)
            info["rare_classes"] = len(rare)
            info["vs"] = {}
            for ref in refs:
                cmp = {}
                for d in DECODES:
                    mx, my = f"{ref}__{d}", f"{arm}__{d}"
                    if mx not in paths or my not in paths:
                        continue
                    c = VS.compare(comps, [mx], [my])
                    kr, ka = (ref, d, L1), (arm, d, L1)
                    if kr not in rr_cache:
                        rr_cache[kr] = rare_rows(cache, t, texts_by[ref][d], reads[t][ref], rare)
                    if ka not in rr_cache:
                        rr_cache[ka] = rare_rows(cache, t, texts_by[arm][d], reads[t][arm], rare)
                    rc = recall_compare(rr_cache[kr], rr_cache[ka])
                    a_ok = not (c["score"]["ci95"][0] > 0) and all(not (c[n]["ci95"][0] > 0) for n in comps)
                    b_ok = c["score"]["ci95"][1] <= 0.05
                    c_ok = rc["combined"]["loss_points"] <= 1.0 and rc["combined"]["ci95"][1] <= 2.0
                    cmp[d] = {"dev": c, "rare_recall": rc, "rule": {"a_not_resolved_worse": bool(a_ok),
                                                                    "b_noninferior_0.05": bool(b_ok),
                                                                    "c_rare_recall": bool(c_ok)}}
                info["vs"][ref] = cmp
            tier["arms"][arm] = info
            print(f"{t} {arm}: " + "; ".join(f"{ref}/{d} score {v['dev']['score']['delta_points']:+.3f} "
                                             f"{v['dev']['score']['ci95']} rare {v['rare_recall']['combined']}"
                                             for ref, cm in info["vs"].items() for d, v in cm.items()), flush=True)
        if timing is not None and t in timing.get("tiers", {}):
            tier["timing"] = timing["tiers"][t]
        tier["seconds"] = round(time.time() - t0, 1)
        res["tiers"][t] = tier
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def cmd_counter(a):
    """#109's synthetic counter-context set (eval/kanum_dev.py render), as stored: CER and lines changed, the
    reference arm against each arm, greedy and rubi (re-decode + LM + the counter rule). Report only."""
    import kanum_dev as KD
    from metrics import lev, norm
    from pyvocr import Vocr, VocrLM, post_counters
    rows = KD.sy_rows(a.synth)
    out = {"file": str(a.synth), "file_sha256": sha256(a.synth), "lines": len(rows), "arms": {}}
    texts = {}
    for spec in [a.ref] + a.arm:
        name, blob, two, gate = parse_arm(spec)
        kw = {"two_level": 1} if two else {}
        if gate:
            kw["head_gate"] = 1
        rt = Vocr(blob, mode="w8i", max_width=1024, **kw)
        lm = VocrLM(rt, str(LM_FILE), **RUBI_LM)
        g, r = [], []
        for row in rows:
            o = rt.run_topk(row["img"], 8)
            g.append(norm(o["text"]))
            if not o["text"]:
                r.append("")
                continue
            o2, _ = rt.redecode(o, tau=COUNTER["tau"], open=COUNTER["open"])
            r.append(norm(post_counters(lm.decode(o2["topv"], o2["topc"], o2["blank"]), rt.lib)[0]))
        texts[name] = {"greedy": g, "rubi": r}
    ref = parse_arm(a.ref)[0]
    chars = sum(len(row["ref"]) for row in rows)
    for name in texts:
        rep = {}
        for d in ("greedy", "rubi"):
            e0 = [lev(p, row["ref"]) for p, row in zip(texts[ref][d], rows)]
            e1 = [lev(p, row["ref"]) for p, row in zip(texts[name][d], rows)]
            rep[d] = {"cer_ref_pct": round(100 * sum(e0) / chars, 4), "cer_pct": round(100 * sum(e1) / chars, 4),
                      "lines_changed": sum(x != y for x, y in zip(texts[ref][d], texts[name][d])),
                      "lines_worse": sum(y > x for x, y in zip(e0, e1)),
                      "lines_better": sum(y < x for x, y in zip(e0, e1))}
        out["arms"][name] = rep
        print(name, json.dumps(rep), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("counter")
    p.add_argument("--synth", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("lines")
    p.add_argument("--cache", required=True)
    p = sub.add_parser("read")
    p.add_argument("--tier", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("score")
    p.add_argument("--cache", required=True)
    p.add_argument("--fits", nargs="+", required=True, help="TIER=fit_TIER.json")
    p.add_argument("--timing", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"lines": cmd_lines, "read": cmd_read, "score": cmd_score, "counter": cmd_counter}[a.cmd](a)


if __name__ == "__main__":
    main()

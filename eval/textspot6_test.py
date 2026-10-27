#!/usr/bin/env python3
"""textspot6's test reads (issue #128, docs/textspot6.md "Test sets"; eval/textspot6_protocol.json "test",
amendment 13), by the conductor only, once each, after the selection record (eval/textspot6_dev_<date>.json) is
committed and pushed, on the final weights (the pick retrained on train + guard + hval: --final RUN_DIR). Every
check that needs no protected pixel runs first; then OUT/<command>.attempt.json is created exclusively
(open(..., "x")), before the first protected pixel is read, so a second run is refused. OUT/<command>.json holds
counts, rates, intervals and hashes only; units, boxes and readings stay under OUT/private/.

  textspot6_test.py title   --selection SEL --final RUN --out OUT --pagepath PAGEPATH [--workers 5]
      the page pool's test title(s) (split "test": Dance with Devils, PCSG00807) through eval/pagepath.py, the
      headline's path: units by `pagepath.py units --set test` (hval's units were built that way), read with
      GRAD + refine (--finder rubi), Rubi's next finder (--finder rubi,fix=1, report), textspot3's pick and the
      final weights, both in the pick's decode exactly as score_run.sh reads hval; paired differences with page
      clusters (one title: a title interval would be degenerate)
  textspot6_test.py clannad --selection SEL --final RUN --out OUT --pagepath PAGEPATH [--work CLANNAD_WORK]
      CLANNAD's frozen long run (owner-drive-20261013/clannad-compare, refs.freeze.json).
      (a) whole frames, textspot3's way (eval/textspot3_protected.py clannad): the final weights in det and
      det+refine against textspot3's pick and textspot's stored arms (grad_refine among them; stored boxes, only
      read, checked against their committed records before the attempt), each reference line reading the
      detections assigned to its GRAD box (eval/textspot2_protected.clannad_preds); textspot3's pick is also
      re-spotted through this harness and checked box for box against its stored boxes.
      (b) report only: each page's ROI (the player's region) through eval/pagepath.py, as Rubi reads a region:
      GRAD + refine, rubi,fix=1, textspot3's pick and the final weights in the pick's decode, attributed to the
      reference lines the same way as (a)
  textspot6_test.py tests   --selection SEL --final RUN --out OUT [--data EVAL] [--shots DIR ...]
      vita-test + psp-test whole frames (eval/manifests/textspot-test.jsonl, textspot's pin; psp-test at 2x report
      only), textspot3's way (eval/textspot3_protected.py tests): the final weights in det and det+refine against
      textspot3's pick (stored units, checked against its record, and re-read through this harness) and
      textspot's stored arms; plus the tagged dialogue CER and the text-free frames' presence (as tsdev)

--research (amendment 15): with a negative verdict, the research pre-release's blob (the selection record's
best_eligible_if_negative, as trained on train: --final is its own run) is read the same way, report only.

--dry-run reads a non-protected stand-in through the same code, for testing before the one read: title with
--split hval --titles A,B; tests with the development manifest (no stored arms); clannad with --work a stand-in
directory built from hval pages by `textspot6_test.py standin`. A dry run refuses the protected inputs, and the
selection and final run are then not checked.
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6 as E6  # noqa: E402
import textspot6_select as SEL  # noqa: E402

HERD, ART = E6.HERD, E6.ART
PRIV = HERD / "shared" / "private"
CLANNAD_WORK = PRIV / "owner-drive-20261013" / "clannad-compare"
MODELS = ART / "vvo-public" / "v030"
LM = MODELS / "vocr_lm_o3_1000k.vlm"
RUBI = Path(os.environ.get("RUBI", str(ART / "vocr6-spotter" / "rubi_textnorm")))   # Rubi 213fd60 core/textnorm
T3_BLOB = ART / "textspot3" / "release" / "v0.3.4-textspot3.1" / "textspot3_t64n8c16_kd_w8.vtsp"
T3_BLOB_SHA256 = "20c3f7cc4f07b443741bbb45fe4270f6fa8c74ce29045a796a5ed8887a3fb26a"
T3_ARM = "T3_kd_t64n8c16_seed3_detref"
T3_STORED = PRIV / "textspot3" / "protected" / "kd_t64n8c16_seed3_detref"
T3_TEST_RECORD = ROOT / "eval" / "textspot3_protected_20261022.json"
T3_CLANNAD_RECORD = ROOT / "eval" / "textspot3_clannad_20261022.json"
T3_CACHE = PRIV / "textspot3" / "cache"
T3_DEV_CACHE = ART / "textspot3" / "eval" / "cache"
FINAL_SPLITS = "train,guard,hval"
PROTECTED_OUT = PRIV / "vocr6-spotter" / "protected"      # the real reads' outputs (private)
KIND = {"crop": "det", "crop+refine": "detref", "det": "det", "det+refine": "detref"}
RULE_MODEL = "VOCR5-N"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def once(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=1) + "\n")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def load_selection(path, dry, research=False):
    """The selection record and the model it names: its pick (arm, decode, representative seed), or with --research
    and a negative verdict its best eligible candidate (amendment 15), as sel["pick"]. A real read needs the record
    committed, unmodified since, and pushed."""
    path = Path(path)
    sel = json.loads(path.read_text(encoding="utf-8"))
    if sel.get("kind") != "textspot6-selection":
        raise SystemExit(f"{path}: not a textspot6 selection record")
    if research:
        if sel.get("pick") or not sel.get("best_eligible_if_negative"):
            raise SystemExit(f"{path}: --research reads the research release of a negative verdict only")
        sel = dict(sel, pick=dict(sel["best_eligible_if_negative"], research=True))
    if not sel.get("pick"):
        raise SystemExit(f"{path}: no pick (negative verdict): nothing is read")
    rec = {"path": str(path), "sha256": sha256_file(path), "pick": sel["pick"]}
    if dry:
        return sel, dict(rec, dry_run=True)
    if git("ls-files", "--error-unmatch", str(path.resolve())).returncode:
        raise SystemExit(f"{path}: the selection record must be committed before any test read")
    if git("diff", "--quiet", "HEAD", "--", str(path.resolve())).returncode:
        raise SystemExit(f"{path}: modified since its commit")
    commit, when = git("log", "-1", "--format=%H %cI", "--", str(path.resolve())).stdout.split()
    if not git("branch", "-r", "--contains", commit).stdout.strip():
        raise SystemExit(f"{path}: its commit {commit[:12]} is not pushed")
    return sel, dict(rec, commit=commit, committed=when)


def check_final(run_dir, sel, dry):
    """(the final blob, its record): the pick's recipe, shape and seed retrained on train + guard + hval."""
    d = Path(run_dir)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    blob = d / "final.vtsp"
    if not (d / "done.json").exists() or not blob.exists():
        raise SystemExit(f"{d}: not a finished run")
    recipe, shape = sel["pick"]["arm"].split("_", 1)
    info = {"run": d.name, "recipe": run["recipe"], "shape": run["shape"]["name"], "seed": run["seed"],
            "hook_splits": run["args"].get("hook_splits"), "teacher": run.get("teacher"),
            "blob_sha256": sha256_file(blob), "blob_bytes": blob.stat().st_size}
    want = (recipe, shape, sel["pick"]["seed"], "train" if sel["pick"].get("research") else FINAL_SPLITS)
    have = (info["recipe"], info["shape"], info["seed"], info["hook_splits"])
    if not dry and have != want:
        raise SystemExit(f"{d}: {have} is not the pick's final retrain (or the research release's run) {want}")
    return blob, info


def check_t3():
    if sha256_file(T3_BLOB) != T3_BLOB_SHA256:
        raise SystemExit(f"{T3_BLOB}: not textspot3's released pick")
    return {"blob": str(T3_BLOB), "blob_sha256": T3_BLOB_SHA256, "arm": T3_ARM}


def load_models(*blobs):
    import textspot6_twin as T6
    return [T6.Model(Path(b).read_bytes()) for b in blobs]


def guard_out(out, dry):
    """A real read writes under PROTECTED_OUT (private); a dry run never does."""
    out = Path(out)
    inside = str(out.resolve()).startswith(str(PROTECTED_OUT.resolve()) + os.sep)
    if inside == dry:
        raise SystemExit(f"{out}: a real read writes under {PROTECTED_OUT}, a dry run elsewhere")
    return out


# ----------------------------------------------------------------------------- textspot's reader

def textspot_readers(models, proto, out, record, reader_src):
    """The readers textspot's protected records were read with, exactly: the C runtime and its ctypes wrapper of a
    checkout of that commit (d214a05: runtime/vocr.c, runtime/pyvocr.py), built here with finder_tighten's scalar
    flags; the library must hash to the record's reader (eval/textspot2.readers_for's check). Today's runtime reads
    the same by design (its head gate and two-level head are off by default) but differs in bytes, which that check
    refuses, and its wrapper does not fit the old library's structs."""
    import importlib.util

    import finder_tighten as FT
    import textspot as TS
    src = Path(reader_src)
    lib = Path(out) / "private" / "rbuild_textspot" / "libvocr_scalar.so"
    lib.parent.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
    subprocess.run(["gcc", *FT.SCALAR_FLAGS, "-o", str(lib), "runtime/vocr.c", "-lm"], check=True, cwd=src, env=env)
    sha = sha256_file(lib)
    want = record["reader"]["library_sha256"]
    if sha != want:
        raise SystemExit(f"{src}: its reader library {sha[:12]} is not the record's {want[:12]}")
    spec = importlib.util.spec_from_file_location("pyvocr_textspot", src / "runtime" / "pyvocr.py")
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)

    class Reader:                      # finder_tighten.Reader on the old wrapper and library
        def __init__(self, blob):
            self.rt = pv.Vocr(blob, mode="w8i", max_width=1024, lib=pv.load_lib(str(lib)))
            self.sha256 = FT.sha256(blob)
            self.blob_bytes, self.arena_bytes = self.rt.blob_bytes, self.rt.arena_bytes

        def read(self, g, invert):
            return FT.norm(self.rt.text(np.ascontiguousarray(g), invert=bool(invert)))

    rd = TS.Readers(TS.recognizer_paths(models, proto), Reader)
    version = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
    rd.identity = {"library_sha256": sha, "compiler": version, "scalar_flags": list(FT.SCALAR_FLAGS),
                   "source": {"commit": commit, "vocr_c_sha256": sha256_file(src / "runtime" / "vocr.c"),
                              "pyvocr_sha256": sha256_file(src / "runtime" / "pyvocr.py")}}
    return rd


def record_readers(models, proto, out, compiler, record, reader_src):
    import textspot2_protected as TP2
    if reader_src:
        return textspot_readers(models, proto, out, record, reader_src)
    return TP2.build_readers(models, proto, out, compiler, record)


# ----------------------------------------------------------------------------- the page path

def pp_run(pagepath, units, base, extra, priv, pages=None, workers=5):
    """One eval/pagepath.py run (score_run.sh's flags) with this read's private reading cache."""
    cmd = [sys.executable, str(pagepath), "run", "--units", str(units), *(["--pages", str(pages)] if pages else []),
           *extra, "--models", str(MODELS), "--lm", str(LM), "--rubi", str(RUBI),
           "--cache", str(priv / "pp_cache" / "reads.sqlite"), "--work", str(priv / "pp_work"),
           "--workers", str(workers), "--rows", f"{base}.rows.jsonl", "--out", f"{base}.json"]
    (priv / "pp_cache").mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return Path(f"{base}.json")


def page_rows(result):
    """pagepath's unit rows with each unit's cluster set to its page (uid "<title>:<page>:...")."""
    doc = json.loads(Path(result).read_text(encoding="utf-8"))
    return {u["uid"]: dict(u, cluster=":".join(u["uid"].split(":")[:2])) for u in doc["units"]}, doc


def arm_summary(rows, doc):
    return {"units": len(rows), "cer": SEL.cer(rows), "platform": SEL.per_platform(rows),
            "errors": sum(r["errors"] for r in rows.values()), "chars": sum(r["chars"] for r in rows.values()),
            "by_platform": doc["summary"]["by_platform"], "finder": doc.get("finder"),
            "textnorm": doc.get("textnorm"), "result_sha256": sha256_file(doc["_path"]) if "_path" in doc else None}


def cmd_title(a):
    dry = a.dry_run
    if dry and a.split == "test":
        raise SystemExit("--dry-run reads a non-protected split (--split hval --titles ...)")
    if not dry and (a.split != "test" or a.titles):
        raise SystemExit("the test read is split test, every title")
    import textspot as TS
    sel, sel_rec = load_selection(a.selection, dry, a.research)
    dec = sel["pick"]["decode"]
    blob, final = check_final(a.final, sel, dry)
    t3 = check_t3()
    load_models(blob, T3_BLOB)
    pagepath = Path(a.pagepath)
    out = guard_out(a.out, dry)
    priv = out / "private" / "title"
    titles = a.titles.split(",") if a.titles else None
    once(out / "title.attempt.json", {"started": utc(), "split": a.split, "titles": titles, "selection": sel_rec,
                                      "final": final, "decode": dec, "textspot3": t3,
                                      "pagepath_sha256": sha256_file(pagepath), "rubi_textnorm": str(RUBI),
                                      "code_sha256": sha256_file(__file__)})
    priv.mkdir(parents=True, exist_ok=True)
    units = priv / "units.jsonl"
    subprocess.run([sys.executable, str(pagepath), "units", "--pagepool", str(E6.POOL), "--set", a.split,
                    *(["--titles", a.titles] if a.titles else []), "--out", str(units)], check=True,
                   stdin=subprocess.DEVNULL)
    res = {"grad_refine": pp_run(pagepath, units, priv / "grad_refine", ["--finder", "rubi"], priv, E6.PAGES,
                                 a.workers),
           "rubi_fix1": pp_run(pagepath, units, priv / "rubi_fix1", ["--finder", "rubi,fix=1"], priv, E6.PAGES,
                               a.workers)}
    frames = None
    for name, b in (("textspot6", blob), ("textspot3", T3_BLOB)):
        boxes = priv / f"boxes_{name}.jsonl"
        if dec.startswith("crop"):
            E6.write_crop(Path(b).read_bytes(), list(E6.crop_jobs("hval", units)), KIND[dec], boxes, a.workers)
            flag = "--boxes"
        else:
            if frames is None:
                frames = list(E6.hval_frames(split=a.split, titles=set(titles) if titles else None))
            spot = E6.write_spot(b, frames, priv / f"spot_{name}.jsonl", a.split, a.workers)
            with open(boxes, "w", encoding="utf-8") as f:
                for r in E6.frame_boxes(E6.load_spot(spot)[1], KIND[dec]):
                    f.write(json.dumps(r) + "\n")
            flag = "--frame-boxes"
        res[name] = pp_run(pagepath, units, priv / name, [flag, str(boxes)], priv, E6.PAGES, a.workers)
    rows, docs = {}, {}
    for k, p in res.items():
        rows[k], docs[k] = page_rows(p)
        docs[k]["_path"] = str(p)
    n_units = sum(1 for _ in open(units, encoding="utf-8")) - 1
    result = {"kind": "textspot6-test-title" + ("-dry-run" if dry else ""), "computed": utc(), "split": a.split,
              "titles": sorted({r["uid"].split(":")[0] for r in rows["grad_refine"].values()}),
              "selection": sel_rec, "final": final, "decode": dec, "textspot3": t3,
              "pagepath_sha256": sha256_file(pagepath), "units": n_units, "units_sha256": sha256_file(units),
              "pages": len({r["cluster"] for r in rows["grad_refine"].values()}),
              "clusters": "pages (<title>:<page>)", "arms": {k: arm_summary(rows[k], docs[k]) for k in rows},
              "pairs": {f"textspot6-{b}": SEL.diff(rows["textspot6"], rows[b])
                        for b in ("grad_refine", "textspot3", "rubi_fix1")}}
    result["pairs"]["textspot3-grad_refine"] = SEL.diff(rows["textspot3"], rows["grad_refine"])
    TS.assert_public(result)
    (out / "title.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for k, v in result["arms"].items():
        print(f"title {k}: dialogue CER {v['cer']}% over {v['units']} units", flush=True)
    for k, v in result["pairs"].items():
        print(f"title {k}: {v['delta']:+.4f} {v.get('ci95')}", flush=True)
    return result


# ----------------------------------------------------------------------------- vita-test + psp-test

def spot_rows(blob, frames, path, set_name, workers):
    """The twin's spot rows ({frame}, boxes, blocks, pick, presence, truncated, rboxes) of textspot Frames."""
    E6.write_spot(blob, [({"frame": f.id}, f.grey) for f in frames], path, set_name, workers)
    meta, rows = E6.load_spot(path)
    if [r["frame"] for r in rows] != [f.id for f in frames]:
        raise SystemExit(f"{path}: rows out of order")
    return rows


def seed_cache(cache_dir, src):
    """The reading cache seeded once with a byte-for-byte copy of src's shards (only read)."""
    cache_dir = Path(cache_dir)
    marker = cache_dir / "SEEDED.json"
    if marker.exists() or not src or not Path(src).exists():
        return json.loads(marker.read_text()) if marker.exists() else None
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in sorted(Path(src).glob("*.jsonl")):
        shutil.copyfile(f, cache_dir / f.name)
        files[f.name] = sha256_file(cache_dir / f.name)
    rec = {"source": str(src), "files": files, "seeded": utc()}
    marker.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def cmd_tests(a):
    import textspot as TS
    import textspot2_protected as TP2
    import textspot3_rule as R
    import textspot_metrics as M
    import textspot_protected as TP
    dry = a.dry_run
    sel, sel_rec = load_selection(a.selection, dry, a.research)
    dec = sel["pick"]["decode"]
    blob, final = check_final(a.final, sel, dry)
    t3 = check_t3()
    load_models(blob, T3_BLOB)
    out = guard_out(a.out, dry)
    priv = out / "private"
    proto = TS.protocol()
    man = ROOT / "eval" / "manifests" / ("textspot-dev.jsonl" if dry else "textspot-test.jsonl")
    digest = sha256_file(man)
    if not dry and digest != TS.pinned_sha(proto, TP.TEST_MANIFEST):
        raise SystemExit(f"{man}: sha256 {digest} is not textspot's pinned test manifest")
    rows = TS.jsonl(man)
    models = [RULE_MODEL]
    report_ids = TP2.report_ids_of(rows, TP.REPORT_PARTS)
    stored, stored_sha, t3_units = {}, {}, None
    if not dry:
        record = json.loads(TP2.TEST_RECORD.read_text(encoding="utf-8"))
        if record["manifest"]["sha256"] != digest:
            raise SystemExit("the stored test record is of another manifest")
        su, _, stored_sha = TP2.stored_tests(TP2.STORED / "tests" / "private", models, rows, record, TP.REPORT_PARTS)
        stored = {k: v[RULE_MODEL]["units"] for k, v in su.items()}
        t3rec = json.loads(T3_TEST_RECORD.read_text(encoding="utf-8"))
        f3 = T3_STORED / "tests" / "private" / T3_ARM / f"units_{RULE_MODEL}.json"
        t3_units = json.loads(f3.read_text(encoding="utf-8"))["units"]
        want = dict(t3rec["arms"][T3_ARM][RULE_MODEL])
        want.pop("all_frames_per_part", None)
        if TS.summarize(TP.scored(t3_units, report_ids)) != want:
            raise SystemExit(f"{f3}: does not reproduce textspot3's committed record")
        stored_sha[T3_ARM] = sha256_file(f3)
        rd = record_readers(models, proto, out, a.compiler, record, a.reader_src)
        seeded = seed_cache(priv / "cache", T3_CACHE)
    else:
        rd = TS.c_readers({m: TS.BLOBS[m] for m in [*models, TS.V_MODEL]}, priv / "cbuild")
        seeded = seed_cache(priv / "cache", T3_DEV_CACHE)
    once(out / "tests.attempt.json", {"started": utc(), "manifest_sha256": digest, "selection": sel_rec,
                                      "reader": getattr(rd, "identity", None),
                                      "final": final, "decode": dec, "textspot3": t3, "stored_sha256": stored_sha,
                                      "code_sha256": sha256_file(__file__)})
    frames = TS.load_frames(rows, list(TS.SHOT_DIRS) if dry else TP.test_sources(a))
    spots = {"textspot6": spot_rows(blob, frames, priv / "spot_textspot6.jsonl", "tests", a.workers),
             "textspot3": spot_rows(T3_BLOB, frames, priv / "spot_textspot3.jsonl", "tests", a.workers)}
    names = [("textspot6_det", "textspot6", "det"), ("textspot6_detref", "textspot6", "detref"),
             ("textspot3_reread", "textspot3", "detref")]
    per_frame, tags = [], {n: [] for n, _, _ in names}
    for i, f in enumerate(frames):
        d = {}
        for n, src, k in names:
            d[n], bx = E6.arm_of(spots[src][i], k, f.height)
            tags[n].append(bx)
        per_frame.append(d)
    cache = TS.Cache(priv / "cache")
    try:
        n_read = TS.read_all(frames, per_frame, models, rd, cache, workers=a.workers)
        units, tagged = {}, {}
        for n, src, _ in names:
            units[n], tagged[n] = [], []
            for i, f in enumerate(frames):
                texts = TS.texts_of(f, per_frame[i][n]["boxes"], RULE_MODEL, rd, cache)
                units[n].append(TS.frame_unit(f, per_frame[i][n], texts))
                dets = [{"box": list(b[:4]), "vertical": bool(b[4]), "text": t, "tag": b[8]}
                        for b, t in zip(tags[n][i], texts)]
                e, c = E6.tagged_dialogue(dets, f.refs, f.width, f.height)
                tagged[n].append({"unit": f.id, "vn": f.vn, "part": f.part, "errors": e, "chars": c,
                                  "presence": spots[src][i]["presence"], "text_free": f.text_free,
                                  "report": f.id in report_ids})
    finally:
        cache.close()
    for n in units:
        d = priv / n
        d.mkdir(parents=True, exist_ok=True)
        (d / f"units_{RULE_MODEL}.json").write_text(json.dumps({"units": units[n], "tagged": tagged[n]}) + "\n",
                                                   encoding="utf-8")
    result = {"kind": "textspot6-test-tests" + ("-dry-run" if dry else ""), "computed": utc(),
              "manifest": {"path": str(man.relative_to(ROOT)), "sha256": digest, "frames": len(rows)},
              "report_only_frames": len(report_ids), "scored_frames": len(rows) - len(report_ids),
              "selection": sel_rec, "final": final, "decode": dec, "primary": f"textspot6_{KIND[dec]}",
              "textspot3": t3, "models": models, "read": n_read, "cache_seed": seeded,
              "reader": getattr(rd, "identity", None), "arms": {},
              "comparisons": {}, "recall_by_class": {}}
    for n in units:
        sc = TP.scored(units[n], report_ids)
        s = TS.summarize(sc)
        s["all_frames_per_part"] = TS.summarize(units[n])["per_part"]
        tg = [t for t in tagged[n] if not t["report"]]
        tf = [t for t in tg if t["text_free"]]
        s["tagged_dialogue"] = M.cer_of(tg)
        s["presence_text_free"] = {"frames": len(tf), "with_dialogue_box": sum(1 for t in tf if t["presence"])}
        result["arms"][n] = s
        result["recall_by_class"][n] = R.recall_by_class(sc)
    if not dry:
        result["stored_sha256"] = stored_sha
        others = {**stored, T3_ARM: t3_units}
        for n in ("textspot6_det", "textspot6_detref"):
            ca = TS.rule_arm(TP.scored(units[n], report_ids))
            result["comparisons"][n] = {b: M.compare(ca, TS.rule_arm(TP.scored(u, report_ids)))
                                        for b, u in others.items()}
        result["recall_by_class"][T3_ARM] = R.recall_by_class(TP.scored(t3_units, report_ids))
        re3 = TS.summarize(TP.scored(units["textspot3_reread"], report_ids))
        result["textspot3_reread_equals_stored"] = re3 == TS.summarize(TP.scored(t3_units, report_ids))
    TS.assert_public(result)
    (out / "tests.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for n, s in result["arms"].items():
        print(f"tests {n}: F1 {s['detection']['f1']:.4f} CER all {s['cer_all']['cer']} dialogue "
              f"{s['cer_dialogue']['cer']} tagged {s['tagged_dialogue']['cer']}", flush=True)
    return result


# ----------------------------------------------------------------------------- CLANNAD

def scope_tables(sel_lines, preds, names, focus, idx_cache=None):
    """{scope: lines, pages, systems {name: CR.summary}, pairs {focus-other: paired page bootstrap}} over the
    reference lines (dialogue: the lines not flagged name)."""
    import clannad_refs as CR
    import vocr3_report as V
    out = {}
    for scope in ("dialogue", "all"):
        sl = [x for x in sel_lines if scope == "all" or not x.get("name")]
        pg = [x["page"] for x in sl]
        ids = sorted(set(pg))
        e = {"lines": len(sl), "pages": len(ids), "systems": {}, "pairs": {}}
        if ids:
            idx = CR.boot_idx(len(ids), CR.REPS, CR.SEED)
            sums = {}
            for n in names:
                sums[n] = CR.page_sums([CR.line_unit(x["text"], preds[n].get(x["id"], "")) for x in sl], pg)
                e["systems"][n] = CR.summary(sums[n], ids, idx)
            for f in focus:
                for b in names:
                    if b == f:
                        continue
                    d = V.paired_bootstrap({p: tuple(sums[f][p][:2]) for p in ids},
                                           {p: tuple(sums[b][p][:2]) for p in ids}, CR.REPS, CR.SEED)
                    e["pairs"][f"{f}-{b}"] = {"cer_delta": d["delta"], "cer_ci95": d["ci95"],
                                              "candidate_errors": d["candidate_errors"],
                                              "baseline_errors": d["baseline_errors"], "chars": d["chars"]}
        out[scope] = e
    return out


def keyed_refs(rs, refs, gb):
    """The page's reference lines keyed to its GRAD boxes (ROI px) as clannad_preds keys them."""
    import textspot as TS
    gbt = [TS.ints(b) for b in gb]
    out = []
    for r in rs:
        if refs.get(r["id"], {}).get("status") != "ref":
            continue
        box = TS.ints(r["box"])
        if box in gbt:
            out.append((r, gbt.index(box)))
    return out


def roi_units(pages, frames, rois, refs):
    """(b)'s units in eval/pagepath.py's development-unit form: one per page with reference lines, its ROI's grey;
    the lines are the page's reference lines (GRAD boxes in ROI px as x0, y0, x1, y1)."""
    units = []
    for (page, rs, _), f, roi in zip(pages, frames, rois):
        ref_rows = [r for r in rs if refs.get(r["id"], {}).get("status") == "ref"]
        if not ref_rows:
            continue
        x, y, w, h = rs[0]["roi"]
        units.append({"cohort": "clannad", "unit": page, "shot": page, "variant": "base", "vn": "clannad",
                      "kind": "vita", "src_kind": "vita", "scale": 1.0, "region": [x, y, x + w, y + h],
                      "image": roi, "frame": (f.width, f.height),
                      "lines": [{"id": r["id"], "text": refs[r["id"]]["text"], "fixed": refs[r["id"]]["text"],
                                 "hint": "", "styles": ["name"] if refs[r["id"]].get("name") else [],
                                 "box": [r["box"][0], r["box"][1], r["box"][0] + r["box"][2],
                                         r["box"][1] + r["box"][3]]} for r in ref_rows]})
    return units


def cmd_clannad(a):
    dry = a.dry_run
    work = Path(a.work)
    if dry == (work.resolve() == CLANNAD_WORK.resolve()):
        raise SystemExit("the real read is of CLANNAD's long run; a dry run of a stand-in (textspot6_test.py standin)")
    import clannad_refs as CR
    import finder_tighten as FT
    import textspot as TS
    import textspot2 as T2E
    import textspot2_protected as TP2
    import textspot_protected as TP
    from textspot3_protected import COUNT_KEYS, stored_clannad_counts

    from common import to_gray
    sel, sel_rec = load_selection(a.selection, dry, a.research)
    dec = sel["pick"]["decode"]
    blob, final = check_final(a.final, sel, dry)
    t3 = check_t3()
    load_models(blob, T3_BLOB)
    pagepath = Path(a.pagepath)
    out = guard_out(a.out, dry)
    priv = out / "private"
    proto = TS.protocol()
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if sha256_file(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    refs = {x["id"]: x for x in CR.read_jsonl(work / "refs.jsonl")}
    lines, _ = CR.load_lines(work)
    models = [RULE_MODEL]
    stored, stored_sha, record, t3rec = {}, {}, None, None
    if not dry:
        record = json.loads(TP2.CLANNAD_RECORD.read_text(encoding="utf-8"))
        t3rec = json.loads(T3_CLANNAD_RECORD.read_text(encoding="utf-8"))
        if record["refs_sha256"] != freeze["refs_sha256"] or t3rec["refs_sha256"] != freeze["refs_sha256"]:
            raise SystemExit("the stored CLANNAD records are of other references")
        for arm in TP2.STORED_ARMS:
            stored[arm], _ = T2E.load_arm(TP2.STORED / "clannad" / "private" / arm, model="none")
        stored[T3_ARM], _ = T2E.load_arm(T3_STORED / "clannad" / "private" / T3_ARM, model="none")
        stored_sha = {k: v["_sha256"] for k, v in stored.items()}
        counts0, n_pages = stored_clannad_counts(stored, lines, refs)
        want = {**{k: record["arms"][k] for k in TP2.STORED_ARMS}, T3_ARM: t3rec["arms"][T3_ARM]}
        diffs = [f"{k}.{c}" for k in stored for c in COUNT_KEYS if counts0[k][c] != want[k][c]]
        if n_pages != record["counts"]["pages"]:
            diffs.append("counts.pages")
        if diffs:
            raise SystemExit(f"the stored arms do not reproduce their CLANNAD records (nothing read): {diffs[:5]}")
        rd = record_readers(models, proto, out, a.compiler, record, a.reader_src)
        seeded = seed_cache(priv / "cache", T3_CACHE)
    else:
        rd = TS.c_readers({m: TS.BLOBS[m] for m in [*models, TS.V_MODEL]}, priv / "cbuild")
        seeded = None
    once(out / "clannad.attempt.json", {"started": utc(), "refs_sha256": freeze["refs_sha256"],
                                        "reader": getattr(rd, "identity", None),
                                        "selection": sel_rec, "final": final, "decode": dec, "textspot3": t3,
                                        "stored_sha256": stored_sha, "pagepath_sha256": sha256_file(pagepath),
                                        "code_sha256": sha256_file(__file__)})
    pages = TP.load_pages(work)
    frames, rois = [], []
    for page, rs, rgb in pages:
        frames.append(TS.Frame({"id": page, "vn": "clannad", "part": "clannad", "refs": []}, to_gray(rgb),
                               TS.pixel_sha256(rgb)))
        x, y, w, h = rs[0]["roi"]
        rois.append(np.ascontiguousarray(frames[-1].grey[y:y + h, x:x + w]))
    grad = TP.c_grad(FT.build(priv / "fbuild", a.compiler)["driver"])(rois)
    # (a) whole frames, textspot3's way. The stored arms come first: they must reproduce their committed records
    # before any box of the final weights (or the re-spotted textspot3 pick) is computed on a CLANNAD page.
    mine = [("textspot6_det", "textspot6", "det"), ("textspot6_detref", "textspot6", "detref"),
            ("textspot3_reread", "textspot3", "detref")]
    cache = TS.Cache(priv / "cache")
    try:
        sp = [{arm: dict(doc["frames"][f.id], boxes=[tuple(b) for b in doc["frames"][f.id]["boxes"]],
                         blocks=[tuple(b) for b in doc["frames"][f.id]["blocks"]]) for arm, doc in stored.items()}
              for f in frames]
        preds, counts, extra, n_read, diffs = {}, {}, None, 0, []
        if stored:
            n_read += TS.read_all(frames, sp, models, rd, cache, a.workers)
            p0, counts, extra = TP2.clannad_preds(pages, frames, sp, grad, refs, list(stored), models, rd, cache)
            preds = {n: p0[n][RULE_MODEL] for n in stored}
            sel0 = [x for x in refs.values() if x["status"] == "ref" and x["id"] in preds[TP2.REF_ARM]]
            sc0 = scope_tables(sel0, preds, list(stored), [])
            diffs = [k for k, v in (("scored_lines", len(sel0)), ("counts", extra)) if v != record[k]]
            diffs += [f"arms.{k}" for k in stored if counts[k] != want[k]]
            for scope in ("dialogue", "all"):
                sy = sc0[scope]["systems"]
                diffs += [f"{scope}.{k}" for k in TP2.STORED_ARMS
                          if sy[k] != record["scopes"][scope]["systems"].get(f"{k}/{RULE_MODEL}")]
                if sy[T3_ARM] != t3rec["scopes"][scope]["systems"].get(f"{T3_ARM}/{RULE_MODEL}"):
                    diffs.append(f"{scope}.{T3_ARM}")
            if diffs:
                (out / "clannad.stopped.json").write_text(json.dumps(
                    {"stopped": utc(), "why": "the stored arms do not reproduce their records; no box of the final "
                     "weights was computed", "differences": diffs}, indent=1) + "\n", encoding="utf-8")
                raise SystemExit(f"the stored arms do not reproduce their CLANNAD records: {diffs[:5]}")
        spots = {"textspot6": spot_rows(blob, frames, priv / "spot_textspot6.jsonl", "clannad", a.workers),
                 "textspot3": spot_rows(T3_BLOB, frames, priv / "spot_textspot3.jsonl", "clannad", a.workers)}
        per_frame = [{n: E6.arm_of(spots[src][i], k, f.height)[0] for n, src, k in mine} for i, f in enumerate(frames)]
        n_read += TS.read_all(frames, per_frame, models, rd, cache, a.workers)
        p1, c1, extra1 = TP2.clannad_preds(pages, frames, per_frame, grad, refs, [n for n, _, _ in mine], models, rd,
                                           cache)
    finally:
        cache.close()
    if extra is not None and extra1 != extra:
        raise SystemExit("the page counts differ between the stored arms' pass and the final weights'")
    extra = extra1
    preds.update({n: p1[n][RULE_MODEL] for n, _, _ in mine})
    counts.update(c1)
    names = [n for n, _, _ in mine] + list(stored)
    sel_lines = [x for x in refs.values() if x["status"] == "ref" and x["id"] in preds[names[0]]]
    (priv / "preds_whole.json").write_text(json.dumps(preds, ensure_ascii=False) + "\n", encoding="utf-8")
    whole = {"scored_lines": len(sel_lines), "counts": extra, "arms": {n: counts[n] for n in names},
             "read": n_read, "cache_seed": seeded, "primary": f"textspot6_{KIND[dec]}",
             "scopes": scope_tables(sel_lines, preds, names, ["textspot6_det", "textspot6_detref"])}
    if not dry:
        same = sum(1 for i, f in enumerate(frames)
                   if [list(b) for b in per_frame[i]["textspot3_reread"]["boxes"]]
                   == [list(b) for b in stored[T3_ARM]["frames"][f.id]["boxes"]])
        whole["stored"] = {"sha256": stored_sha, "reproduces_records": not diffs, "differences": diffs,
                           "textspot3_reread_same_boxes_pages": same}
    # (b) report only: each page's ROI through the page path
    units = roi_units(pages, frames, rois, refs)
    pkl = priv / "roi_units.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"units": units}, f)
    res = {"grad_refine": pp_run(pagepath, pkl, priv / "roi_grad_refine", ["--finder", "rubi"], priv,
                                 workers=a.workers),
           "rubi_fix1": pp_run(pagepath, pkl, priv / "roi_rubi_fix1", ["--finder", "rubi,fix=1"], priv,
                               workers=a.workers)}
    idx_of = {p[0]: i for i, p in enumerate(pages)}
    for name, b in (("textspot6", blob), ("textspot3", T3_BLOB)):
        bf = priv / f"roi_boxes_{name}.jsonl"
        if dec.startswith("crop"):
            jobs = [(f"dev:{i}:{u['unit']}:{u['variant']}", u["image"], *u["frame"]) for i, u in enumerate(units)]
            E6.write_crop(Path(b).read_bytes(), jobs, KIND[dec], bf, a.workers)
        else:
            with open(bf, "w", encoding="utf-8") as f:
                for i, u in enumerate(units):
                    r = spots[name][idx_of[u["unit"]]]
                    f.write(json.dumps({"unit": f"dev:{i}:{u['unit']}:{u['variant']}",
                                        "boxes": E6.region_boxes(E6.decoded(r, KIND[dec]), u["region"])}) + "\n")
        res[name] = pp_run(pagepath, pkl, priv / f"roi_{name}", ["--boxes", str(bf)], priv, workers=a.workers)
    rp = {n: {} for n in res}
    for n, p in res.items():
        rows = {r["uid"]: r for r in E6.read_jsonl(str(p).replace(".json", ".rows.jsonl"))}
        for i, u in enumerate(units):
            r = rows[f"dev:{i}:{u['unit']}:{u['variant']}"]
            page, rs, _ = pages[idx_of[u["unit"]]]
            owner = TP.assign([tuple(bx[:4]) for bx in r["boxes_xywh"]], [tuple(g) for g in grad[idx_of[page]]])
            for ref_row, k in keyed_refs(rs, refs, grad[idx_of[page]]):
                rp[n][ref_row["id"]] = "".join(t for t, o in zip(r["texts"], owner) if o == k)
    roi_lines = [x for x in sel_lines if x["id"] in rp["grad_refine"]]
    (priv / "preds_roi.json").write_text(json.dumps(rp, ensure_ascii=False) + "\n", encoding="utf-8")
    roi = {"units": len(units), "units_sha256": sha256_file(pkl), "decode": dec, "scored_lines": len(roi_lines),
           "assignment": "pagepath's boxes (ROI px) go to the GRAD box of the ROI they lie mostly inside "
                         "(textdet_protected.assign); a reference line reads its boxes' page-path texts in box order",
           "pagepath_finder": {n: json.loads(Path(p).read_text(encoding="utf-8")).get("finder")
                               for n, p in res.items()},
           "scopes": scope_tables(roi_lines, rp, list(rp), ["textspot6"])}
    result = {"kind": "textspot6-test-clannad" + ("-dry-run" if dry else ""), "computed": utc(),
              "refs_sha256": freeze["refs_sha256"], "selection": sel_rec, "final": final, "decode": dec,
              "textspot3": t3, "models": models, "pagepath_sha256": sha256_file(pagepath),
              "reader": getattr(rd, "identity", None), "whole_frames": whole, "roi_page_path": roi}
    TS.assert_public(result)
    (out / "clannad.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for part, doc in (("whole", whole), ("roi", roi)):
        for n, v in doc["scopes"]["dialogue"]["systems"].items():
            print(f"clannad {part} {n}: dialogue CER {v['cer']} {v.get('cer_ci95')}", flush=True)
    return result


# ----------------------------------------------------------------------------- the dry run's CLANNAD stand-in

def cmd_standin(a):
    """A CLANNAD-shaped work directory from hval pages (never a protected one), for clannad --dry-run: per page one
    of its hval units' regions as the ROI, the ROI's C GRAD boxes as the line boxes (ROI px), each with the text of
    the unit's line it overlaps most as a reference (status "noref" when none, and for every line of the last
    --no-text pages)."""
    out = Path(a.out)
    if str(out.resolve()).startswith(str(PRIV.resolve())) or out.exists():
        raise SystemExit("standin: a new directory outside the private tree")
    import cv2
    import finder_tighten as FT
    import textspot_protected as TP

    from common import to_gray
    rows = [r for r in E6.read_jsonl(a.units) if "uid" in r and r["platform"] == "vita" and r["variant"] == "base"]
    titles = a.titles.split(",") if a.titles else sorted({r["title"] for r in rows})[:2]
    seen, picked = set(), []
    for r in rows:
        k = (r["title"], r["page"])
        if r["title"] in titles and k not in seen:
            seen.add(k)
            picked.append(r)
    picked = picked[:a.pages]
    session = out / "session"
    session.mkdir(parents=True)
    rois, srcs = [], []
    for r in picked:
        src = E6.PAGES / r["title"] / "pages" / f"{r['page']}.png"
        dst = session / f"{r['title']}_{r['page']}.png"
        os.symlink(src, dst)
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        g = to_gray(np.ascontiguousarray(bgr[:, :, ::-1]))
        x0, y0, x1, y1 = r["region"]
        rois.append(np.ascontiguousarray(g[y0:y1, x0:x1]))
        srcs.append({"path": dst.name, "sha256": sha256_file(dst)})
    grad = TP.c_grad(FT.build(out / "fbuild")["driver"])(rois)
    lines, refs = [], []
    for j, (r, gb, src) in enumerate(zip(picked, grad, srcs)):
        x0, y0, x1, y1 = r["region"]
        page = f"{r['title']}:{r['page']}"
        no_text = j >= len(picked) - a.no_text
        for k, b in enumerate(gb):
            lid = f"{page}:{k}"
            lines.append({"id": lid, "page": page, "roi": [x0, y0, x1 - x0, y1 - y0], "box": [int(v) for v in b],
                          "source": src})
            best, ov = None, 0
            for ln in r["lines"]:
                lx, ly, lw, lh = ln["box"]
                ix = max(0, min(lx + lw, x0 + b[0] + b[2]) - max(lx, x0 + b[0]))
                iy = max(0, min(ly + lh, y0 + b[1] + b[3]) - max(ly, y0 + b[1]))
                if ix * iy > ov:
                    best, ov = ln, ix * iy
            ok = best is not None and not no_text
            refs.append({"id": lid, "page": page, "status": "ref" if ok else "noref",
                         "text": best["text"] if ok else "", "name": False})
    with open(out / "lines.jsonl", "w", encoding="utf-8") as f:
        for x in lines:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (out / "lines.meta.json").write_text(json.dumps({"session": str(session),
                                                     "lines_sha256": sha256_file(out / "lines.jsonl")}) + "\n")
    with open(out / "refs.jsonl", "w", encoding="utf-8") as f:
        for x in refs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (out / "refs.freeze.json").write_text(json.dumps({"refs_sha256": sha256_file(out / "refs.jsonl"),
                                                      "standin": True, "frozen": utc()}) + "\n")
    print(f"standin: {len(picked)} pages, {len(lines)} lines, {sum(1 for x in refs if x['status'] == 'ref')} "
          f"references -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("title", "clannad", "tests"):
        p = sub.add_parser(name)
        p.add_argument("--selection", required=True, help="the committed eval/textspot6_dev_<date>.json")
        p.add_argument("--final", required=True, help="the final run's directory (the pick on train + guard + hval)")
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=5)
        p.add_argument("--compiler", default="gcc")
        p.add_argument("--dry-run", action="store_true", help="a non-protected stand-in through the same code")
        p.add_argument("--research", action="store_true",
                       help="a negative verdict's research release (amendment 15), as trained")
        if name != "tests":
            p.add_argument("--pagepath", required=True, help="the pinned eval/pagepath.py")
        if name == "title":
            p.add_argument("--split", default="test")
            p.add_argument("--titles", help="dry runs only: some titles of the split")
        if name == "clannad":
            p.add_argument("--work", default=str(CLANNAD_WORK))
        if name in ("tests", "clannad"):
            p.add_argument("--reader-src", help="a checkout of the commit textspot's records were read with (d214a05): "
                                                "its runtime and wrapper are the readers (the record's library)")
        if name == "tests":
            p.add_argument("--data", default=str(ART / "eval"), help="eval's data directory (vita/shots, psp/shots)")
            p.add_argument("--shots", action="append", default=[], help="more source directories (text-free shots)")
    p = sub.add_parser("standin")
    p.add_argument("--units", required=True, help="hval units (eval/pagepath.py units)")
    p.add_argument("--titles")
    p.add_argument("--pages", type=int, default=30)
    p.add_argument("--no-text", type=int, default=3)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"title": cmd_title, "clannad": cmd_clannad, "tests": cmd_tests, "standin": cmd_standin}[a.cmd](a)


if __name__ == "__main__":
    main()

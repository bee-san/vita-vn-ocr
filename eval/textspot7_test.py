#!/usr/bin/env python3
"""textspot7's test reads (issue #128, eval/textspot7_protocol.json rule.with_pick: report only), once each, after
the selection records (eval/textspot7_dev_<date>.json, eval/textspot7_confirm_<date>.json) are committed and pushed,
on the final weights (the pick's spotter retrained on train + guard + hval). textspot6's reads (eval/textspot6_test.py,
its amendment 13), with the pick's decode D (join10, extend: synth/textspot7_twin.post) beside the plain det decode:

  textspot7_test.py title   --selection SEL --final RUN --out OUT --pagepath PAGEPATH
      the hooked test title(s) (split "test": Dance with Devils, PCSG00807) through the page path: GRAD + refine,
      rubi,fix=1, the final weights in det and in D; page clusters
  textspot7_test.py clannad --selection SEL --final RUN --out OUT --pagepath PAGEPATH [--work CLANNAD_WORK]
      CLANNAD's frozen long run: (a) whole frames, textspot3's way (eval/textspot6_test.py's harness: textspot's
      stored arms checked against their records first), the final weights in det and in D (D on the whole frame:
      the frame is the region; vertical boxes kept from det; blocks regrouped), textspot3's pick re-spotted;
      (b) each page's ROI through the page path: GRAD + refine, rubi,fix=1, the final weights in det and in D
  textspot7_test.py tests   --selection SEL --final RUN --out OUT
      vita-test + psp-test whole frames, textspot3's way, the final weights in det and in D (on the whole frame)

Every check that needs no protected pixel runs first; then OUT/<command>.attempt.json is created exclusively, so a
second read is refused. OUT/<command>.json holds counts, rates, intervals and hashes only. --dry-run reads the
non-protected stand-ins of eval/textspot6_test.py (title: --split hval --titles A,B; tests: the development manifest;
clannad: a stand-in from `textspot6_test.py standin`).
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6 as E6  # noqa: E402
import textspot6_select as SEL  # noqa: E402
import textspot6_test as T6T  # noqa: E402
import textspot7 as E7  # noqa: E402

import textspot7_twin as T7  # noqa: E402

PROTECTED_OUT = T6T.PRIV / "vocr6-spotter3" / "protected"
FINAL_SPLITS = "train,guard,hval"
SELECTION_KIND = "textspot7-selection-confirm"


def load_selection(path, dry):
    """The confirmation-phase record and its pick (arm, decode, seed, D's steps); a real read needs it (and the dev
    record it names) committed, unmodified, and pushed."""
    path = Path(path)
    sel = json.loads(path.read_text(encoding="utf-8"))
    if sel.get("kind") != SELECTION_KIND or not sel.get("pick"):
        raise SystemExit(f"{path}: not a textspot7 confirmation record with a pick")
    q = next(c for c in sel["qualifiers"] if c["name"] == sel["pick"])
    pick = {"name": q["name"], "arm": q["arm"], "run": q["run"], "decode": q["decode"],
            "seed": int(q["run"].rsplit("seed", 1)[1]), "steps": (q.get("params") or {}).get("steps")}
    if pick["decode"] != "D" or not pick["steps"]:
        raise SystemExit(f"{path}: this harness reads a D pick")
    rec = {"path": str(path), "sha256": T6T.sha256_file(path), "pick": pick}
    if dry:
        return pick, dict(rec, dry_run=True)
    for p in (path, ROOT / "eval" / "textspot7_dev_20261027.json"):
        if T6T.git("ls-files", "--error-unmatch", str(p.resolve())).returncode:
            raise SystemExit(f"{p}: must be committed before any test read")
        if T6T.git("diff", "--quiet", "HEAD", "--", str(p.resolve())).returncode:
            raise SystemExit(f"{p}: modified since its commit")
        commit = T6T.git("log", "-1", "--format=%H", "--", str(p.resolve())).stdout.split()[0]
        if not T6T.git("branch", "-r", "--contains", commit).stdout.strip():
            raise SystemExit(f"{p}: its commit {commit[:12]} is not pushed")
    rec["commit"] = T6T.git("log", "-1", "--format=%H", "--", str(path.resolve())).stdout.split()[0]
    return pick, rec


def check_final(run_dir, pick, dry):
    d = Path(run_dir)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    blob = d / "final.vtsp"
    if not (d / "done.json").exists() or not blob.exists():
        raise SystemExit(f"{d}: not a finished run")
    recipe, shape = pick["arm"].split("_", 1)
    info = {"run": d.name, "recipe": run["recipe"], "shape": run["shape"]["name"], "seed": run["seed"],
            "hook_splits": run["args"].get("hook_splits"), "blob_sha256": T6T.sha256_file(blob),
            "blob_bytes": blob.stat().st_size}
    have = (info["recipe"], info["shape"], info["seed"], info["hook_splits"])
    want = (recipe, shape, pick["seed"], FINAL_SPLITS)
    if not dry and have != want:
        raise SystemExit(f"{d}: {have} is not the pick's final retrain {want}")
    return blob, info


def guard_out(out, dry):
    out = Path(out)
    inside = str(out.resolve()).startswith(str(PROTECTED_OUT.resolve()))
    if inside == dry:
        raise SystemExit(f"{out}: a real read writes under {PROTECTED_OUT}, a dry run elsewhere")
    return out


def steps_of(pick):
    return [s for s in pick["steps"].split(",") if s]


def d_frame_arm(r, grey, steps, height):
    """D on a whole frame (the frame is the region): the det decode's horizontal boxes through the post-pass, each
    keeping the tag of the det box it holds most of; det's vertical boxes kept; blocks regrouped and ordered
    (textspot6_twin.group_and_order). -> (textspot's arm dict, the boxes with their tags), as textspot6.arm_of."""
    import textspot6_twin as T6
    H, W = grey.shape
    det = [b for b in r["boxes"]]
    hz = [b for b in det if not b[4]]
    out = T7.post([(b[0], b[1], b[2], b[3], b[5], b[6]) for b in hz], (0, 0, W, H), grey, steps)
    boxes = []
    for x, y, w, h, conf, dark in out:
        src = max(hz, key=lambda b: (max(0, min(x + w, b[0] + b[2]) - max(x, b[0])) *
                                     max(0, min(y + h, b[1] + b[3]) - max(y, b[1]))), default=None)
        tag, tconf = (src[8], src[9]) if src is not None else (0, 0)
        boxes.append((x, y, w, h, 0, conf, dark, tag, tconf))
    boxes += [(b[0], b[1], b[2], b[3], 1, b[5], b[6], b[8], b[9]) for b in det if b[4]]
    ob, bl, pk = T6.group_and_order(boxes, height)
    bx = [[*b[:7], b[9], b[7], b[8]] for b in ob]
    return {"boxes": [tuple(b[:8]) for b in bx], "blocks": [tuple(k[:7]) for k in bl], "pick": pk,
            "truncated": r["truncated"]}, bx


def unit_d_boxes(units, spot_by_key, steps):
    """--boxes rows of D for page-path units (E7.units_of's form), from their frames' spot rows."""
    rows = []
    for u, img in E7.region_images(units):
        r = spot_by_key[u["key"]]
        bx = T7.post(E7.frame_boxes_of(r), tuple(u["region"]), img, steps)
        rows.append({"unit": u["uid"], "boxes": [[b[0], b[1], b[2], b[3], b[5]] for b in bx]})
    return rows


def hooked_units(path):
    """E7.units_of's records for a hooked units file (the test title's)."""
    out = []
    for r in E6.read_jsonl(path)[1:]:
        s = 2 if r["platform"] == "psp" else 1
        view = "psp2x" if s == 2 else "vita"
        out.append({"uid": r["uid"], "cluster": r["cluster"], "platform": r["platform"],
                    "region": [v * s for v in r["region"]], "key": (r["title"], r["page"], view), "s": s,
                    "title": r["title"], "page": r["page"], "image_sha256": r["image_sha256"]})
    return out


def pairs(rows, names, base, clusters="page"):
    out = {}
    for n in names:
        for b in base:
            if n != b:
                out[f"{n}-{b}"] = SEL.diff(rows[n], rows[b])
    return out


# ----------------------------------------------------------------------------- the test title

def cmd_title(a):
    dry = a.dry_run
    if dry and a.split == "test":
        raise SystemExit("--dry-run reads a non-protected split (--split hval --titles ...)")
    if not dry and (a.split != "test" or a.titles):
        raise SystemExit("the test read is split test, every title")
    pick, sel_rec = load_selection(a.selection, dry)
    blob, final = check_final(a.final, pick, dry)
    T6T.load_models(blob)
    pagepath = Path(a.pagepath)
    out = guard_out(a.out, dry)
    priv = out / "private" / "title"
    titles = a.titles.split(",") if a.titles else None
    T6T.once(out / "title.attempt.json", {"started": T6T.utc(), "split": a.split, "titles": titles,
                                          "selection": sel_rec, "final": final, "steps": pick["steps"],
                                          "pagepath_sha256": T6T.sha256_file(pagepath),
                                          "code_sha256": T6T.sha256_file(__file__)})
    priv.mkdir(parents=True, exist_ok=True)
    units = priv / "units.jsonl"
    subprocess.run([sys.executable, str(pagepath), "units", "--pagepool", str(E6.POOL), "--set", a.split,
                    *(["--titles", a.titles] if a.titles else []), "--out", str(units)], check=True,
                   stdin=subprocess.DEVNULL)
    res = {"grad_refine": T6T.pp_run(pagepath, units, priv / "grad_refine", ["--finder", "rubi"], priv, E6.PAGES,
                                     a.workers),
           "rubi_fix1": T6T.pp_run(pagepath, units, priv / "rubi_fix1", ["--finder", "rubi,fix=1"], priv, E6.PAGES,
                                   a.workers)}
    frames = list(E6.hval_frames(split=a.split, titles=set(titles) if titles else None))
    spot = E6.write_spot(blob, frames, priv / "spot_textspot7.jsonl", a.split, a.workers)
    rows_spot = E6.load_spot(spot)[1]
    det = priv / "boxes_det.jsonl"
    with open(det, "w", encoding="utf-8") as f:
        for r in E6.frame_boxes(rows_spot, "det"):
            f.write(json.dumps(r) + "\n")
    res["textspot7_det"] = T6T.pp_run(pagepath, units, priv / "textspot7_det", ["--frame-boxes", str(det)], priv,
                                      E6.PAGES, a.workers)
    by_key = {(r["title"], r["page"], r["view"]): r for r in rows_spot}
    dfile = priv / "boxes_D.jsonl"
    with open(dfile, "w", encoding="utf-8") as f:
        for r in unit_d_boxes(hooked_units(units), by_key, steps_of(pick)):
            f.write(json.dumps(r) + "\n")
    res["textspot7_D"] = T6T.pp_run(pagepath, units, priv / "textspot7_D", ["--boxes", str(dfile)], priv, E6.PAGES,
                                    a.workers)
    rows, docs = {}, {}
    for k, p in res.items():
        rows[k], docs[k] = T6T.page_rows(p)
        docs[k]["_path"] = str(p)
    result = {"kind": "textspot7-test-title" + ("-dry-run" if dry else ""), "computed": T6T.utc(),
              "split": a.split, "titles": sorted({r["uid"].split(":")[0] for r in rows["grad_refine"].values()}),
              "selection": sel_rec, "final": final, "decode": "D", "steps": pick["steps"],
              "units": len(rows["grad_refine"]), "pagepath_sha256": T6T.sha256_file(pagepath),
              "arms": {k: T6T.arm_summary(rows[k], docs[k]) for k in rows},
              "pairs": pairs(rows, ["textspot7_D", "textspot7_det"], ["grad_refine", "rubi_fix1", "textspot7_det"]),
              "clusters": "pages (one title)"}
    (out / "title.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for k, v in result["arms"].items():
        print(f"title {k}: CER {v['cer']}", flush=True)
    return result


# ----------------------------------------------------------------------------- vita-test + psp-test whole frames

def cmd_tests(a):
    import textspot as TS
    import textspot2_protected as TP2
    import textspot_metrics as M
    import textspot_protected as TP
    dry = a.dry_run
    pick, sel_rec = load_selection(a.selection, dry)
    blob, final = check_final(a.final, pick, dry)
    T6T.load_models(blob, T6T.T3_BLOB)
    out = guard_out(a.out, dry)
    priv = out / "private"
    proto = TS.protocol()
    man = ROOT / "eval" / "manifests" / ("textspot-dev.jsonl" if dry else "textspot-test.jsonl")
    digest = T6T.sha256_file(man)
    if not dry and digest != TS.pinned_sha(proto, TP.TEST_MANIFEST):
        raise SystemExit(f"{man}: not textspot's pinned test manifest")
    rows = TS.jsonl(man)
    models = [T6T.RULE_MODEL]
    report_ids = TP2.report_ids_of(rows, TP.REPORT_PARTS)
    if not dry:
        record = json.loads(TP2.TEST_RECORD.read_text(encoding="utf-8"))
        if record["manifest"]["sha256"] != digest:
            raise SystemExit("the stored test record is of another manifest")
        rd = T6T.record_readers(models, proto, out, a.compiler, record, a.reader_src)
        seeded = T6T.seed_cache(priv / "cache", T6T.T3_CACHE)
    else:
        rd = TS.c_readers({m: TS.BLOBS[m] for m in [*models, TS.V_MODEL]}, priv / "cbuild")
        seeded = T6T.seed_cache(priv / "cache", T6T.T3_DEV_CACHE)
    T6T.once(out / "tests.attempt.json", {"started": T6T.utc(), "manifest_sha256": digest, "selection": sel_rec,
                                          "reader": getattr(rd, "identity", None), "final": final,
                                          "steps": pick["steps"], "code_sha256": T6T.sha256_file(__file__)})
    frames = TS.load_frames(rows, list(TS.SHOT_DIRS) if dry else TP.test_sources(a))
    spots = {"textspot7": T6T.spot_rows(blob, frames, priv / "spot_textspot7.jsonl", "tests", a.workers),
             "textspot3": T6T.spot_rows(T6T.T3_BLOB, frames, priv / "spot_textspot3.jsonl", "tests", a.workers)}
    names = ["textspot7_det", "textspot7_D", "textspot3_reread"]
    per_frame, tags = [], {n: [] for n in names}
    for i, f in enumerate(frames):
        d = {}
        d["textspot7_det"], bx = E6.arm_of(spots["textspot7"][i], "det", f.height)
        tags["textspot7_det"].append(bx)
        d["textspot7_D"], bx = d_frame_arm(spots["textspot7"][i], f.grey, steps_of(pick), f.height)
        tags["textspot7_D"].append(bx)
        d["textspot3_reread"], bx = E6.arm_of(spots["textspot3"][i], "detref", f.height)
        tags["textspot3_reread"].append(bx)
        per_frame.append(d)
    cache = TS.Cache(priv / "cache")
    try:
        n_read = TS.read_all(frames, per_frame, models, rd, cache, workers=a.workers)
        units, tagged = {}, {}
        for n in names:
            units[n], tagged[n] = [], []
            for i, f in enumerate(frames):
                texts = TS.texts_of(f, per_frame[i][n]["boxes"], T6T.RULE_MODEL, rd, cache)
                units[n].append(TS.frame_unit(f, per_frame[i][n], texts))
                dets = [{"box": list(b[:4]), "vertical": bool(b[4]), "text": t, "tag": b[8]}
                        for b, t in zip(tags[n][i], texts)]
                e, c = E6.tagged_dialogue(dets, f.refs, f.width, f.height)
                tagged[n].append({"unit": f.id, "vn": f.vn, "part": f.part, "errors": e, "chars": c,
                                  "report": f.id in report_ids})
    finally:
        cache.close()
    result = {"kind": "textspot7-test-tests" + ("-dry-run" if dry else ""), "computed": T6T.utc(),
              "manifest": {"path": str(man.relative_to(ROOT)), "sha256": digest, "frames": len(rows)},
              "report_only_frames": len(report_ids), "selection": sel_rec, "final": final, "steps": pick["steps"],
              "read": n_read, "cache_seed": seeded, "reader": getattr(rd, "identity", None), "arms": {},
              "comparisons": {}}
    for n in names:
        s = TS.summarize(TP.scored(units[n], report_ids))
        s["tagged_dialogue"] = M.cer_of([t for t in tagged[n] if not t["report"]])
        result["arms"][n] = s
    for a_, b_ in (("textspot7_D", "textspot7_det"), ("textspot7_D", "textspot3_reread"),
                   ("textspot7_det", "textspot3_reread")):
        result["comparisons"][f"{a_}-{b_}"] = M.compare(TS.rule_arm(TP.scored(units[a_], report_ids)),
                                                        TS.rule_arm(TP.scored(units[b_], report_ids)))
    TS.assert_public(result)
    (out / "tests.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for n, s in result["arms"].items():
        print(f"tests {n}: F1 {s['detection']['f1']:.4f} CER all {s['cer_all']['cer']} dialogue "
              f"{s.get('cer_dialogue', {}).get('cer')}", flush=True)
    return result


# ----------------------------------------------------------------------------- CLANNAD

def cmd_clannad(a):
    dry = a.dry_run
    work = Path(a.work)
    if dry == (work.resolve() == T6T.CLANNAD_WORK.resolve()):
        raise SystemExit("the real read is of CLANNAD's long run; a dry run of a stand-in")
    import clannad_refs as CR
    import finder_tighten as FT
    import textspot as TS
    import textspot2_protected as TP2
    import textspot_protected as TP

    from common import to_gray
    pick, sel_rec = load_selection(a.selection, dry)
    blob, final = check_final(a.final, pick, dry)
    T6T.load_models(blob, T6T.T3_BLOB)
    pagepath = Path(a.pagepath)
    out = guard_out(a.out, dry)
    priv = out / "private"
    proto = TS.protocol()
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if T6T.sha256_file(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    refs = {x["id"]: x for x in CR.read_jsonl(work / "refs.jsonl")}
    models = [T6T.RULE_MODEL]
    if not dry:
        record = json.loads(TP2.CLANNAD_RECORD.read_text(encoding="utf-8"))
        if record["refs_sha256"] != freeze["refs_sha256"]:
            raise SystemExit("the stored CLANNAD record is of other references")
        rd = T6T.record_readers(models, proto, out, a.compiler, record, a.reader_src)
        seeded = T6T.seed_cache(priv / "cache", T6T.T3_CACHE)
    else:
        rd = TS.c_readers({m: TS.BLOBS[m] for m in [*models, TS.V_MODEL]}, priv / "cbuild")
        seeded = None
    T6T.once(out / "clannad.attempt.json", {"started": T6T.utc(), "refs_sha256": freeze["refs_sha256"],
                                            "reader": getattr(rd, "identity", None), "selection": sel_rec,
                                            "final": final, "steps": pick["steps"],
                                            "pagepath_sha256": T6T.sha256_file(pagepath),
                                            "code_sha256": T6T.sha256_file(__file__)})
    pages = TP.load_pages(work)
    frames, rois = [], []
    for page, rs, rgb in pages:
        frames.append(TS.Frame({"id": page, "vn": "clannad", "part": "clannad", "refs": []}, to_gray(rgb),
                               TS.pixel_sha256(rgb)))
        x, y, w, h = rs[0]["roi"]
        rois.append(np.ascontiguousarray(frames[-1].grey[y:y + h, x:x + w]))
    grad = TP.c_grad(FT.build(priv / "fbuild", a.compiler)["driver"])(rois)
    # (a) whole frames, textspot3's way
    spots = {"textspot7": T6T.spot_rows(blob, frames, priv / "spot_textspot7.jsonl", "clannad", a.workers),
             "textspot3": T6T.spot_rows(T6T.T3_BLOB, frames, priv / "spot_textspot3.jsonl", "clannad", a.workers)}
    names = ["textspot7_det", "textspot7_D", "textspot3_reread"]
    per_frame = []
    for i, f in enumerate(frames):
        per_frame.append({"textspot7_det": E6.arm_of(spots["textspot7"][i], "det", f.height)[0],
                          "textspot7_D": d_frame_arm(spots["textspot7"][i], f.grey, steps_of(pick), f.height)[0],
                          "textspot3_reread": E6.arm_of(spots["textspot3"][i], "detref", f.height)[0]})
    cache = TS.Cache(priv / "cache")
    try:
        n_read = TS.read_all(frames, per_frame, models, rd, cache, a.workers)
        p1, c1, extra = TP2.clannad_preds(pages, frames, per_frame, grad, refs, names, models, rd, cache)
    finally:
        cache.close()
    preds = {n: p1[n][T6T.RULE_MODEL] for n in names}
    sel_lines = [x for x in refs.values() if x["status"] == "ref" and x["id"] in preds[names[0]]]
    whole = {"scored_lines": len(sel_lines), "counts": extra, "arms": {n: c1[n] for n in names}, "read": n_read,
             "cache_seed": seeded, "scopes": T6T.scope_tables(sel_lines, preds, names, ["textspot7_D",
                                                                                        "textspot7_det"])}
    # (b) each page's ROI through the page path
    units = T6T.roi_units(pages, frames, rois, refs)
    pkl = priv / "roi_units.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"units": units}, f)
    res = {"grad_refine": T6T.pp_run(pagepath, pkl, priv / "roi_grad_refine", ["--finder", "rubi"], priv,
                                     workers=a.workers),
           "rubi_fix1": T6T.pp_run(pagepath, pkl, priv / "roi_rubi_fix1", ["--finder", "rubi,fix=1"], priv,
                                   workers=a.workers)}
    idx_of = {p[0]: i for i, p in enumerate(pages)}
    for name in ("textspot7_det", "textspot7_D"):
        bf = priv / f"roi_boxes_{name}.jsonl"
        with open(bf, "w", encoding="utf-8") as f:
            for i, u in enumerate(units):
                r = spots["textspot7"][idx_of[u["unit"]]]
                if name == "textspot7_det":
                    bx = E6.region_boxes(E6.decoded(r, "det"), u["region"])
                else:
                    d = T7.post(E7.frame_boxes_of(r), tuple(u["region"]), u["image"], steps_of(pick))
                    bx = [[b[0], b[1], b[2], b[3], b[5]] for b in d]
                f.write(json.dumps({"unit": f"dev:{i}:{u['unit']}:{u['variant']}", "boxes": bx}) + "\n")
        res[name] = T6T.pp_run(pagepath, pkl, priv / f"roi_{name}", ["--boxes", str(bf)], priv, workers=a.workers)
    rp = {n: {} for n in res}
    for n, p in res.items():
        rows = {r["uid"]: r for r in E6.read_jsonl(str(p).replace(".json", ".rows.jsonl"))}
        for i, u in enumerate(units):
            r = rows[f"dev:{i}:{u['unit']}:{u['variant']}"]
            page, rs, _ = pages[idx_of[u["unit"]]]
            owner = TP.assign([tuple(bx[:4]) for bx in r["boxes_xywh"]], [tuple(g) for g in grad[idx_of[page]]])
            for ref_row, k in T6T.keyed_refs(rs, refs, grad[idx_of[page]]):
                rp[n][ref_row["id"]] = "".join(t for t, o in zip(r["texts"], owner) if o == k)
    roi_lines = [x for x in sel_lines if x["id"] in rp["grad_refine"]]
    roi = {"units": len(units), "units_sha256": T6T.sha256_file(pkl), "scored_lines": len(roi_lines),
           "scopes": T6T.scope_tables(roi_lines, rp, list(rp), ["textspot7_D", "textspot7_det"])}
    result = {"kind": "textspot7-test-clannad" + ("-dry-run" if dry else ""), "computed": T6T.utc(),
              "refs_sha256": freeze["refs_sha256"], "selection": sel_rec, "final": final, "steps": pick["steps"],
              "models": models, "pagepath_sha256": T6T.sha256_file(pagepath),
              "reader": getattr(rd, "identity", None), "whole_frames": whole, "roi_page_path": roi}
    TS.assert_public(result)
    (out / "clannad.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for part, doc in (("whole", whole), ("roi", roi)):
        for n, v in doc["scopes"]["dialogue"]["systems"].items():
            print(f"clannad {part} {n}: dialogue CER {v['cer']} {v.get('cer_ci95')}", flush=True)
    return result


def cmd_record(a):
    out = Path(a.out)
    rec = {"kind": "textspot7-test-record", "computed": T6T.utc(), "out": str(out), "reads": {}}
    for c in ("title", "tests", "clannad"):
        att, res = out / c / f"{c}.attempt.json", out / c / f"{c}.json"
        if not att.exists() or not res.exists():
            raise SystemExit(f"{c}: no finished read under {out}")
        doc = json.loads(res.read_text(encoding="utf-8"))
        a_doc = json.loads(att.read_text(encoding="utf-8"))
        rec["reads"][c] = {"attempt": {"started": a_doc["started"], "sha256": T6T.sha256_file(att)},
                           "result_sha256": T6T.sha256_file(res), "result": doc}
    rec = T6T.sanitize(rec)
    Path(a.json).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{a.json}: written")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("title", cmd_title), ("clannad", cmd_clannad), ("tests", cmd_tests)):
        p = sub.add_parser(name)
        p.add_argument("--selection", required=True)
        p.add_argument("--final", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=5)
        p.add_argument("--compiler", default="gcc")
        p.add_argument("--dry-run", action="store_true")
        if name != "tests":
            p.add_argument("--pagepath", required=True)
        if name == "title":
            p.add_argument("--split", default="test")
            p.add_argument("--titles")
        if name == "clannad":
            p.add_argument("--work", default=str(T6T.CLANNAD_WORK))
        if name in ("tests", "clannad"):
            p.add_argument("--reader-src", help="a checkout of d214a05 (textspot's reader), as textspot6_test.py")
        if name == "tests":
            p.add_argument("--data", default=str(T6T.ART / "eval"))
            p.add_argument("--shots", nargs="*", default=[])
        p.set_defaults(fn=fn)
    p = sub.add_parser("record")
    p.add_argument("--out", required=True)
    p.add_argument("--json", required=True)
    p.set_defaults(fn=cmd_record)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

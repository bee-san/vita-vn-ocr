#!/usr/bin/env python3
"""textspot2: the protected reads of one selected model, once per command, for the conductor only (docs/textspot2.md
"The rule", "Protected sets"; eval/textspot2_protocol.json "rule" "protected"). Nothing here is ever run by a helper.

  textspot2_protected.py tests --cand SPEC [--decode det|det+refine] --out $P2/protected/<pick>/tests [--data EVAL]
                               [--shots DIR ...] [--cbin DIR|auto] [--models N,S,L] [--workers 5]
      every frame of eval/manifests/textspot-test.jsonl (pinned by textspot amendment 3 and reused by textspot2's
      protocol), as eval/textspot_protected.py's tests reads them: the scored frames, and psp-test at 2x
      (adrenaline) report only.
  textspot2_protected.py clannad --cand SPEC [--decode ...] --work CLANNAD_WORK --out $P2/protected/<pick>/clannad ...
      CLANNAD's long run, whole frames in, as eval/textspot_protected.py's clannad reads it.

SPEC: T2:RUN (a finished run under $T2/runs, both decodes; --decode picks one) or eval/textspot.py's det:NAME=VTSP /
detref:NAME=VTSP. A VTSP version-2 (d2s) blob runs through textspot2.v2_dispatch.

Only the pick runs: no baseline is re-run. Its comparisons are against textspot's stored protected units of ref
(B_h2_s4_c24n4_seed2_detref) and of the eight baselines ($HERD/shared/private/textspot/protected/{tests,clannad}/
private/<arm>/, only read): every stored file's sha256 is recorded, and the stored files must reproduce textspot's
committed records (eval/textspot_protected_20261019.json, eval/textspot_clannad_20261019.json) exactly. The
secondary column: on the test set textspot2_rule.dnc_units (every reference's errors recomputed from the boxes and
the cached readings, checked against each arm's units); on CLANNAD its "all" scope (dialogue and speaker-name lines).

Each command runs once: OUT/<command>.attempt.json is created exclusively (textspot_protected.once, open(..., "x"))
before any protected pixel is read, and refuses a second run. Readings go to $P2/cache, seeded once with a copy of
textspot's private cache ($P/cache, only read). Public outputs (OUT/tests.json, OUT/clannad.json) hold counts, rates
and hashes only; the pick's units and boxes go to OUT/private/.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot as TS  # noqa: E402
import textspot2 as T2E  # noqa: E402
import textspot_metrics as M  # noqa: E402
import textspot_protected as TP  # noqa: E402

TEST_RECORD = ROOT / "eval" / "textspot_protected_20261019.json"
CLANNAD_RECORD = ROOT / "eval" / "textspot_clannad_20261019.json"
STORED = TS.P / "protected"
REF_ARM = "B_h2_s4_c24n4_seed2_detref"
STORED_ARMS = (REF_ARM,) + TS.BASELINES


def cand_spec(text, decode=None, runs_dir=T2E.DEFAULT["runs"]):
    """(the pick's one arm, its blob's VTSP version): T2:RUN gives the run's arms T2_<run>_det / _detref
    (textspot2.run_specs), det:NAME=VTSP / detref:NAME=VTSP eval/textspot.py's; --decode picks one."""
    if text.startswith("T2:"):
        run = text[3:]
        runs, _ = T2E.finished_runs(runs_dir, [run])
        specs, version = T2E.run_specs(runs), runs[run]["version"]
    else:
        kind, _, rest = text.partition(":")
        if kind not in T2E.KINDS or "=" not in rest:
            raise SystemExit(f"--cand {text!r}: T2:RUN, det:NAME=VTSP or detref:NAME=VTSP")
        version = T2E.blob_version(Path(rest.partition("=")[2]).read_bytes())
        with T2E.v2_dispatch(version != 1):
            specs = TS.parse_arms(text)
    if decode:
        specs = [s for s in specs if s.decode == decode]
    if len(specs) != 1:
        raise SystemExit(f"--cand {text!r} (--decode {decode}) names {len(specs)} arms, not one")
    return specs[0], version


def meta_of(spec, version):
    with T2E.v2_dispatch(version != 1):
        return spec.meta()


def report_ids_of(rows, report_parts):
    return {r["id"] for r in rows if r.get("report_only") or r.get("part") in report_parts}


def stored_tests(stored_dir, models, rows, record, report_parts):
    """textspot's stored protected test units of ref and the eight baselines: ({arm: {model: units doc}}, {arm: boxes
    doc}, {arm: {"boxes", model: sha256}}). Every file must reproduce textspot's committed record: the units score
    the stored boxes, their scored frames' summary is the record's and their per-part rows over every frame are its
    all_frames_per_part."""
    if record["candidate"]["arm"] != REF_ARM or list(record["report_parts"]) != list(report_parts):
        raise SystemExit("the stored test record is not textspot's read of ref with these report parts")
    report = report_ids_of(rows, report_parts)
    units, boxes, sha = {}, {}, {}
    for arm in STORED_ARMS:
        d = Path(stored_dir) / arm
        bdoc, _ = T2E.load_arm(d, model="none")
        boxes[arm], sha[arm] = bdoc, {"boxes": bdoc["_sha256"]}
        for m in models:
            f = d / f"units_{m}.json"
            doc = json.loads(f.read_text(encoding="utf-8"))
            want = dict(record["arms"][arm][m])
            allp = want.pop("all_frames_per_part")
            if (doc["meta"]["boxes_sha256"] != bdoc["_sha256"] or TS.summarize(TP.scored(doc["units"], report)) != want
                    or TS.summarize(doc["units"])["per_part"] != allp):
                raise SystemExit(f"{f}: does not reproduce textspot's committed record {TEST_RECORD.name}")
            units.setdefault(arm, {})[m] = doc
            sha[arm][m] = TS.sha256(f)
    return units, boxes, sha


def build_readers(models, proto, out, compiler, record, readers=None):
    """The pick's readers (eval/textspot.py's C readers), whose library must be the one textspot's protected units
    were read with (the record's reader); injected readers (tests) are used as they are."""
    if readers is not None:
        return readers
    return T2E.readers_for(models, proto, Path(out) / "rbuild", compiler, record["reader"]["library_sha256"])


def seed_private_cache(cache_dir, src):
    return T2E.seed_cache(cache_dir, src) if src else None


def cmd_tests(args, readers=None, log=print):
    proto, proto2 = TS.protocol(args.protocol), T2E.protocol2(args.protocol2)
    out = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else T2E.P2 / "cache"
    T2E.check_writable(out, cache_dir)
    man = Path(args.manifest) if args.manifest else ROOT / TP.TEST_MANIFEST
    digest = TS.sha256(man)
    if digest != TS.pinned_sha(proto, TP.TEST_MANIFEST) or digest != proto2["reused"]["test_manifest"]["sha256"]:
        raise SystemExit(f"{man}: sha256 {digest} is not the test manifest both protocols pin")
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    if record["manifest"]["sha256"] != digest:
        raise SystemExit("the stored test record is of another manifest")
    cand, version = cand_spec(args.cand, args.decode or None, args.runs)
    models = TS.parse_models(args.models)
    rows = TS.jsonl(man)
    report_ids = report_ids_of(rows, args.report_parts)
    stored, stored_boxes, stored_sha = stored_tests(args.stored, models, rows, record, args.report_parts)
    rd = build_readers(models, proto, out, args.compiler, record, readers)
    seeded = seed_private_cache(cache_dir, args.seed_from)
    cdir = T2E.resolve_cbin(args.cbin, out, args.compiler)
    meta = meta_of(cand, version)
    TP.once(out / "tests.attempt.json", {"started": TS.utc_now(), "protocol_sha256": TS.sha256(args.protocol),
                                         "protocol2_sha256": TS.sha256(args.protocol2), "manifest_sha256": digest,
                                         "candidate": meta, "blob_version": version, "models": models,
                                         "stored_sha256": stored_sha, "code_sha256": T2E.code_identity()})
    frames = TS.load_frames(rows, TP.test_sources(args))
    manifest = {"set": "test", "path": TS.rel(man), "sha256": digest, "frames": len(rows),
                "vns": len({r["vn"] for r in rows})}
    cache = TS.Cache(cache_dir)
    try:
        with T2E.v2_dispatch(version != 1):
            TS.evaluate(frames, [cand], manifest, out / "private", models, rd, cache, cdir, args.workers, log=log,
                        proto=proto)
        per = [{a: stored_boxes[a]["frames"][f.id] for a in STORED_ARMS} for f in frames]
        new = TS.read_all(frames, per, models, rd, cache, args.workers, log)
    finally:
        cache.close()
    shas = {f.id: f.sha for f in frames}
    cand_boxes, _ = T2E.load_arm(out / "private" / cand.name, model="none")
    result = {"kind": "textspot2-protected-tests", "computed": TS.utc_now(), "manifest": manifest,
              "protocol_sha256": TS.sha256(args.protocol), "protocol2_sha256": TS.sha256(args.protocol2),
              "candidate": meta, "blob_version": version, "report_parts": sorted(args.report_parts),
              "report_only_frames": len(report_ids), "scored_frames": len(rows) - len(report_ids), "models": models,
              "bootstrap": M.BOOT, "recognizers_sha256": dict(rd.sha), "reader": getattr(rd, "identity", None),
              "code_sha256": T2E.code_identity(), "cache_seed": seeded,
              "stored": {"dir": TS.rel(args.stored), "record": TS.rel(args.record),
                         "record_sha256": TS.sha256(args.record), "sha256": stored_sha, "reproduces_record": True,
                         "crops_read_now": new},
              "arms": {cand.name: {}}, "comparisons": {}, "secondary": {}}
    for m in models:
        cu = TS.load_units(out / "private" / cand.name / f"units_{m}.json")["units"]
        result["arms"][cand.name][m] = TS.summarize(TP.scored(cu, report_ids))
        result["arms"][cand.name][m]["all_frames_per_part"] = TS.summarize(cu)["per_part"]
        ca = TS.rule_arm(TP.scored(cu, report_ids))
        result["comparisons"][m] = {a: M.compare(ca, TS.rule_arm(TP.scored(stored[a][m]["units"], report_ids)))
                                    for a in STORED_ARMS}
        rec = (rd.sha[m], rd.sha[TS.V_MODEL])
        dnc = {cand.name: T2E.dnc_of(cand_boxes, rows, shas, cache, rec, cu)}
        for a in STORED_ARMS:
            dnc[a] = T2E.dnc_of(stored_boxes[a], rows, shas, cache, rec, stored[a][m]["units"])
        sc = {a: TP.scored(u, report_ids) for a, u in dnc.items()}
        result["secondary"][m] = {"dnc": {a: M.cer_of(u) for a, u in sc.items()},
                                  "vs": {a: M.vn_diff(sc[cand.name], sc[a]) for a in STORED_ARMS}}
    TS.assert_public(result)
    TS.write_json(out / "tests.json", result, indent=1)
    for m in models:
        s = result["arms"][cand.name][m]
        log(f"{cand.name} {m}: F1 {TS.fmt(s['detection']['f1'])} CER all {s['cer_all']['cer']} dialogue "
            f"{s['cer_dialogue']['cer']} D+N+C {result['secondary'][m]['dnc'][cand.name]['cer']}")
        for a, c in result["comparisons"][m].items():
            log(f"  - {a}: F1 {c['f1']['delta']:+.4f} {c['f1']['ci95']}  CER {c['cer_all']['delta']} "
                f"{c['cer_all']['ci95']}  dialogue {c['cer_dialogue']['delta']} {c['cer_dialogue']['ci95']}")
    return result


# ----------------------------------------------------------------------------- CLANNAD

def clannad_preds(pages, frames, per_frame, grad, refs, names, models, rd, cache):
    """eval/textspot_protected.py's CLANNAD assignment for every named arm: each detection goes to the GRAD box (of
    the page's ROI, in frame pixels) it overlaps most when that box holds at least half its area, and a reference
    line reads its detections joined in the arm's reading order. -> (preds, counts per arm, page counts)."""
    preds = {n: {m: {} for m in models} for n in names}
    counts = {n: {"boxes": 0, "vertical": 0, "boxes_on_no_text_pages": 0, "boxes_outside_ref_boxes": 0,
                  "c_equal_pages": 0, "c_differ_pages": 0, "c_unchecked_pages": 0} for n in names}
    extra = {"pages": len(pages), "pages_with_ref": 0, "no_text_pages": 0, "unmatched": 0, "matched": 0}
    for (page, rs, _), f, arms, gb in zip(pages, frames, per_frame, grad):
        x0, y0 = rs[0]["roi"][:2]
        gbf = [(x + x0, y + y0, w, h) for x, y, w, h in gb]
        ref_rows = [r for r in rs if refs.get(r["id"], {}).get("status") == "ref"]
        extra["pages_with_ref" if ref_rows else "no_text_pages"] += 1
        gbt = [TS.ints(b) for b in gb]
        keyed = []
        for r in ref_rows:
            box = TS.ints(r["box"])
            if box in gbt:
                keyed.append((r, gbt.index(box)))
                extra["matched"] += 1
            else:
                extra["unmatched"] += 1
        ks = {k for _, k in keyed}
        for n in names:
            a, c = arms[n], counts[n]
            c["boxes"] += len(a["boxes"])
            c["vertical"] += sum(1 for b in a["boxes"] if b[4])
            ce = a.get("c_equal")
            c["c_equal_pages" if ce is True else "c_differ_pages" if ce is False else "c_unchecked_pages"] += 1
            if not ref_rows:
                c["boxes_on_no_text_pages"] += len(a["boxes"])
                continue
            owner = TP.assign([b[:4] for b in a["boxes"]], gbf)
            c["boxes_outside_ref_boxes"] += sum(1 for o in owner if o not in ks)
            for m in models:
                texts = TS.texts_of(f, a["boxes"], m, rd, cache)
                for r, k in keyed:
                    preds[n][m][r["id"]] = "".join(t for t, o in zip(texts, owner) if o == k)
    return preds, counts, extra


def cmd_clannad(args, readers=None, grad_fn=None, log=print):
    import clannad_refs as CR
    import vocr3_report as V

    from common import to_gray
    proto, proto2 = TS.protocol(args.protocol), T2E.protocol2(args.protocol2)
    if proto2.get("kind") != "vita-vn-ocr-textspot2-protocol":
        raise SystemExit(f"{args.protocol2}: not textspot2's protocol")
    out = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else T2E.P2 / "cache"
    T2E.check_writable(out, cache_dir)
    work = Path(args.work)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if TS.sha256(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    if record["refs_sha256"] != freeze["refs_sha256"] or record["candidate"]["arm"] != REF_ARM:
        raise SystemExit("the stored CLANNAD record is not textspot's read of ref on these references")
    cand, version = cand_spec(args.cand, args.decode or None, args.runs)
    models = TS.parse_models(args.models)
    if list(record["models"])[:len(models)] != models:
        raise SystemExit(f"--models {models}: the stored record read {record['models']}")
    stored, stored_sha = {}, {}
    for a in STORED_ARMS:
        stored[a], _ = T2E.load_arm(Path(args.stored) / a, model="none")
        stored_sha[a] = stored[a]["_sha256"]
    rd = build_readers(models, proto, out, args.compiler, record, readers)
    seeded = seed_private_cache(cache_dir, args.seed_from)
    cdir = T2E.resolve_cbin(args.cbin, out, args.compiler)
    meta = meta_of(cand, version)
    TP.once(out / "clannad.attempt.json", {"started": TS.utc_now(), "refs_sha256": freeze["refs_sha256"],
                                           "protocol_sha256": TS.sha256(args.protocol),
                                           "protocol2_sha256": TS.sha256(args.protocol2), "candidate": meta,
                                           "blob_version": version, "models": models, "stored_sha256": stored_sha,
                                           "code_sha256": T2E.code_identity()})
    refs = {x["id"]: x for x in CR.read_jsonl(work / "refs.jsonl")}
    pages = TP.load_pages(work)
    frames, rois = [], []
    for page, rs, rgb in pages:
        frames.append(TS.Frame({"id": page, "vn": "clannad", "part": "clannad", "refs": []}, to_gray(rgb),
                               TS.pixel_sha256(rgb)))
        x, y, w, h = rs[0]["roi"]
        rois.append(frames[-1].grey[y:y + h, x:x + w].copy())
    if grad_fn is None:
        import finder_tighten as FT
        grad_fn = TP.c_grad(FT.build(out / "fbuild", args.compiler)["driver"])
    grad = grad_fn(rois)
    manifest = {"set": "clannad", "path": TS.rel(work / "refs.jsonl"), "sha256": freeze["refs_sha256"],
                "frames": len(frames), "vns": 1}
    cache = TS.Cache(cache_dir)
    try:
        with T2E.v2_dispatch(version != 1):
            per_frame = TS.run_arms(frames, [cand], manifest, out / "private", models, rd, cache, cdir, args.workers,
                                    log=log, proto=proto)
        for i, f in enumerate(frames):
            for a in STORED_ARMS:
                fr = stored[a]["frames"][f.id]
                per_frame[i][a] = dict(fr, boxes=[tuple(b) for b in fr["boxes"]],
                                       blocks=[tuple(b) for b in fr["blocks"]])
        new = TS.read_all(frames, [{a: p[a] for a in STORED_ARMS} for p in per_frame], models, rd, cache,
                          args.workers, log)
    finally:
        cache.close()
    names = [cand.name, *STORED_ARMS]
    preds, counts, extra = clannad_preds(pages, frames, per_frame, grad, refs, names, models, rd, cache)
    first = preds[cand.name][models[0]] if models else {}
    lines = [x for x in refs.values() if x["status"] == "ref" and x["id"] in first]
    result = {"kind": "textspot2-clannad", "computed": TS.utc_now(), "refs_sha256": freeze["refs_sha256"],
              "protocol_sha256": TS.sha256(args.protocol), "protocol2_sha256": TS.sha256(args.protocol2),
              "candidate": meta, "blob_version": version, "models": models, "recognizers_sha256": dict(rd.sha),
              "reader": getattr(rd, "identity", None), "code_sha256": T2E.code_identity(), "cache_seed": seeded,
              "scored_lines": len(lines), "counts": extra, "arms": {cand.name: counts[cand.name]},
              "assignment": "a detection goes to the GRAD box (of the page's ROI, in frame pixels) it overlaps most "
                            "when that box holds at least half its area; a reference line reads its detections "
                            "joined in the arm's reading order",
              "secondary": "the 'all' scope (dialogue and speaker-name lines; CLANNAD has no choice lines) is the "
                           "dialogue + name + choice column", "scopes": {}}
    diffs = [k for k in ("scored_lines", "counts") if result[k] != record[k]]
    diffs += [f"arms.{a}" for a in STORED_ARMS if counts[a] != record["arms"][a]]
    for scope in ("dialogue", "all"):
        sel = [x for x in lines if scope == "all" or not x.get("name")]
        pg = [x["page"] for x in sel]
        ids = sorted(set(pg))
        e = {"lines": len(sel), "pages": len(ids), "systems": {}, "pairs": {}}
        if ids:
            idx = CR.boot_idx(len(ids), CR.REPS, CR.SEED)
            for m in models:
                sums = {}
                for n in names:
                    sums[n] = CR.page_sums([CR.line_unit(x["text"], preds[n][m][x["id"]]) for x in sel], pg)
                    e["systems"][f"{n}/{m}"] = CR.summary(sums[n], ids, idx)
                for b in STORED_ARMS:
                    d = V.paired_bootstrap({p: tuple(sums[cand.name][p][:2]) for p in ids},
                                           {p: tuple(sums[b][p][:2]) for p in ids}, CR.REPS, CR.SEED)
                    e["pairs"][f"{cand.name}/{m}:{b}/{m}"] = {
                        "cer_delta": d["delta"], "cer_ci95": d["ci95"], "candidate_errors": d["candidate_errors"],
                        "baseline_errors": d["baseline_errors"], "chars": d["chars"]}
        want = record["scopes"][scope]
        diffs += [f"scopes.{scope}.{k}" for k in ("lines", "pages") if e[k] != want[k]]
        diffs += [f"scopes.{scope}.systems.{k}" for k in e["systems"] if not k.startswith(cand.name + "/")
                  and e["systems"][k] != want["systems"].get(k)]
        result["scopes"][scope] = e
    result["stored"] = {"dir": TS.rel(args.stored), "record": TS.rel(args.record),
                        "record_sha256": TS.sha256(args.record), "boxes_sha256": stored_sha,
                        "reproduces_record": not diffs, "differences": diffs, "crops_read_now": new}
    TS.assert_public(result)
    TS.write_json(out / "clannad.json", result, indent=1)
    for k, v in result["scopes"]["dialogue"]["systems"].items():
        if k.startswith(cand.name + "/"):
            log(f"{k}: dialogue CER {v['cer']} {v['cer_ci95']}")
    if diffs:
        raise SystemExit(f"the stored arms do not reproduce textspot's CLANNAD record: {diffs[:5]}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tests", "clannad"):
        p = sub.add_parser(name)
        p.add_argument("--cand", required=True, help="T2:RUN, det:NAME=VTSP or detref:NAME=VTSP")
        p.add_argument("--decode", default="", choices=("", "det", "det+refine"))
        p.add_argument("--out", required=True, help="$P2/protected/<pick>/" + name)
        p.add_argument("--runs", default=str(T2E.DEFAULT["runs"]))
        p.add_argument("--cbin", default="", help="a cbuild directory (textspot2.py cbuild), or 'auto'")
        p.add_argument("--models", default="N,S,L")
        p.add_argument("--workers", type=int, default=5)
        p.add_argument("--cache", default="", help="the reading cache (default $P2/cache)")
        p.add_argument("--seed-from", default=str(TS.P / "cache"), help="textspot's private cache, copied once")
        p.add_argument("--compiler", default="gcc")
        p.add_argument("--protocol", default=str(TS.PROTOCOL))
        p.add_argument("--protocol2", default=str(T2E.PROTOCOL2))
        p.add_argument("--stored", default=str(STORED / name / "private"))
        p.add_argument("--record", default=str(TEST_RECORD if name == "tests" else CLANNAD_RECORD))
        if name == "tests":
            p.add_argument("--manifest", default="")
            p.add_argument("--data", default=str(TS.ART / "eval"), help="eval's data directory (vita/shots, psp/shots)")
            p.add_argument("--shots", action="append", default=[], help="more source directories (text-free shots)")
            p.add_argument("--report-parts", nargs="*", default=list(TP.REPORT_PARTS))
        else:
            p.add_argument("--work", required=True)
    args = ap.parse_args(argv)
    {"tests": cmd_tests, "clannad": cmd_clannad}[args.cmd](args)


if __name__ == "__main__":
    main()

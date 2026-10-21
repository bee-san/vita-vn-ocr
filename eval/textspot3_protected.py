#!/usr/bin/env python3
"""textspot3: the protected reads of the pick, once per command, for the conductor only (docs/textspot3.md
"Protected sets, release and Rubi"; eval/textspot3_protocol.json "rule" "protected"). Nothing here is ever run by a
helper on real data.

  textspot3_protected.py tests --cand T3:RUN --decode det|det+refine --dev eval/textspot3_dev_<date>.json
                               --out $P3/protected/<pick>/tests [--data EVAL] [--shots DIR ...] [--cbin DIR|auto]
                               [--models N,S,L] [--workers 5]
      every frame of eval/manifests/textspot-test.jsonl (pinned by textspot amendment 3 and reused by textspot3's
      protocol), as eval/textspot_protected.py's tests reads them: the scored frames, and psp-test at 2x
      (adrenaline) report only.
  textspot3_protected.py clannad --cand T3:RUN --decode ... --dev ... --work CLANNAD_WORK
                                 --out $P3/protected/<pick>/clannad ...
      CLANNAD's long run, whole frames in, as eval/textspot_protected.py's clannad reads it.

--cand: T3:RUN (a finished run under $T3/runs, both decodes; --decode picks one) or eval/textspot.py's det:NAME=VTSP
/ detref:NAME=VTSP; a VTSP version-2 blob runs through textspot2's path, a version-3 blob through textspot3's
(eval/textspot3.py dispatch). Only the pick runs: --dev is the committed selection record, and the arm must be its
winner with the winner's blob (no pick: nothing is read). No baseline is re-run.

Its comparisons are against textspot's stored protected units of ref (B_h2_s4_c24n4_seed2_detref) and, report only,
of the eight baselines ($HERD/shared/private/textspot/protected/{tests,clannad}/private/<arm>/, only read). Before
the attempt, every stored file's sha256 is recorded and the stored files must reproduce textspot's committed records
(eval/textspot_protected_20261019.json: the units' summaries exactly, eval/textspot2_protected.stored_tests;
eval/textspot_clannad_20261019.json: every count of its arms that needs no pixel); after the CLANNAD read every
stored arm's systems must be the record's too.

Each command runs once: OUT/<command>.attempt.json is created exclusively (textspot_protected.once, open(..., "x"))
before any protected pixel is read, and refuses a second run. Readings go to $P3/cache, seeded once with a copy of
textspot2's private cache ($P2/cache) when it has one, else textspot's ($P/cache); both are only read. Public outputs
(OUT/tests.json, OUT/clannad.json) hold counts, rates and hashes only; the pick's units and boxes go to OUT/private/.
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
import textspot2_protected as TP2  # noqa: E402
import textspot3 as T3E  # noqa: E402
import textspot3_rule as R  # noqa: E402
import textspot_metrics as M  # noqa: E402
import textspot_protected as TP  # noqa: E402

TEST_RECORD, CLANNAD_RECORD = TP2.TEST_RECORD, TP2.CLANNAD_RECORD
STORED = TS.P / "protected"
REF_ARM, STORED_ARMS = TP2.REF_ARM, TP2.STORED_ARMS
P3 = T3E.P3
CACHE_SOURCES = (T2E.P2 / "cache", TS.P / "cache")
COUNT_KEYS = ("boxes", "vertical", "boxes_on_no_text_pages", "c_equal_pages", "c_differ_pages", "c_unchecked_pages")


def seed_source(sources=CACHE_SOURCES):
    """The private cache to copy: textspot2's when it holds a shard, else textspot's."""
    for d in sources:
        if any(Path(d).glob("*.jsonl")):
            return Path(d)
    return None


def cand_spec(text, decode=None, runs_dir=T3E.DEFAULT["runs"]):
    """(the pick's one arm, its blob's VTSP version): T3:RUN gives the run's arms T3_<run>_det / _detref
    (textspot3.run_specs), det:NAME=VTSP / detref:NAME=VTSP eval/textspot.py's; --decode picks one."""
    if text.startswith("T3:"):
        run = text[3:]
        runs, _ = T3E.finished_runs(runs_dir, [run])
        specs, version = T3E.run_specs(runs), runs[run]["version"]
    else:
        kind, _, rest = text.partition(":")
        if kind not in T3E.KINDS or "=" not in rest:
            raise SystemExit(f"--cand {text!r}: T3:RUN, det:NAME=VTSP or detref:NAME=VTSP")
        version = T3E.blob_version(Path(rest.partition("=")[2]).read_bytes())
        with T3E.dispatch({version}):
            specs = TS.parse_arms(text)
    if decode:
        specs = [s for s in specs if s.decode == decode]
    if len(specs) != 1:
        raise SystemExit(f"--cand {text!r} (--decode {decode}) names {len(specs)} arms, not one")
    return specs[0], version


def meta_of(spec, version):
    with T3E.dispatch({version}):
        return spec.meta()


def check_pick(dev, spec):
    """Only the pick runs: the selection record (eval/textspot3.py select) names a winner, and it is this arm with
    this blob. -> the record's identity for the attempt record."""
    if not dev:
        raise SystemExit("--dev: the committed selection record (only its pick is read)")
    doc = json.loads(Path(dev).read_text(encoding="utf-8"))
    if doc.get("kind") != "textspot3-selection":
        raise SystemExit(f"{dev}: not a textspot3 selection record")
    win = doc["result"]["winner"]
    if win is None:
        raise SystemExit(f"{dev}: no pick: the verdict is negative and nothing is read")
    row = next((s for s in doc["seeds"] if s["name"] == win), None)
    if win != spec.name or row is None or row["blob_sha256"] != TS.sha256_bytes(spec.blob):
        raise SystemExit(f"--cand {spec.name}: the record's pick is {win} (only the pick is read)")
    return {"record": TS.rel(dev), "sha256": TS.sha256(dev), "winner": win}


def seed_private_cache(cache_dir, src):
    if not src:
        return None
    return T3E.seed_cache(cache_dir, src, kind="textspot3-private-cache-seed")


def cmd_tests(args, readers=None, log=print):
    proto, proto3 = TS.protocol(args.protocol), T3E.protocol3(args.protocol3)
    out = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else P3 / "cache"
    T3E.check_writable(out, cache_dir)
    man = Path(args.manifest) if args.manifest else ROOT / TP.TEST_MANIFEST
    digest = TS.sha256(man)
    if digest != TS.pinned_sha(proto, TP.TEST_MANIFEST) or digest != proto3["reused"]["test_manifest"]["sha256"]:
        raise SystemExit(f"{man}: sha256 {digest} is not the test manifest both protocols pin")
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    if record["manifest"]["sha256"] != digest:
        raise SystemExit("the stored test record is of another manifest")
    cand, version = cand_spec(args.cand, args.decode or None, args.runs)
    pick = check_pick(args.dev, cand)
    models = TS.parse_models(args.models)
    if not models or models[0] != TS.RULE_MODEL:
        raise SystemExit(f"--models: {TS.RULE_MODEL} first (the rule's recognizer), then the report ones")
    rows = TS.jsonl(man)
    report_ids = TP2.report_ids_of(rows, args.report_parts)
    stored, _, stored_sha = TP2.stored_tests(args.stored, models, rows, record, args.report_parts)
    rd = TP2.build_readers(models, proto, out, args.compiler, record, readers)
    src = Path(args.seed_from) if args.seed_from else seed_source()
    seeded = seed_private_cache(cache_dir, src)
    cdir = T3E.resolve_cbin(args.cbin, out, args.compiler)
    meta = meta_of(cand, version)
    TP.once(out / "tests.attempt.json", {"started": TS.utc_now(), "protocol_sha256": TS.sha256(args.protocol),
                                         "protocol3_sha256": TS.sha256(args.protocol3), "manifest_sha256": digest,
                                         "candidate": meta, "blob_version": version, "pick": pick, "models": models,
                                         "stored_sha256": stored_sha, "code_sha256": T3E.code_identity()})
    frames = TS.load_frames(rows, TP.test_sources(args))
    manifest = {"set": "test", "path": TS.rel(man), "sha256": digest, "frames": len(rows),
                "vns": len({r["vn"] for r in rows})}
    cache = TS.Cache(cache_dir)
    try:
        with T3E.dispatch({version}):
            TS.evaluate(frames, [cand], manifest, out / "private", models, rd, cache, cdir, args.workers, log=log,
                        proto=proto)
    finally:
        cache.close()
    result = {"kind": "textspot3-protected-tests", "computed": TS.utc_now(), "manifest": manifest,
              "protocol_sha256": TS.sha256(args.protocol), "protocol3_sha256": TS.sha256(args.protocol3),
              "candidate": meta, "blob_version": version, "pick": pick, "report_parts": sorted(args.report_parts),
              "report_only_frames": len(report_ids), "scored_frames": len(rows) - len(report_ids), "models": models,
              "bootstrap": M.BOOT, "recognizers_sha256": dict(rd.sha), "reader": getattr(rd, "identity", None),
              "code_sha256": T3E.code_identity(), "cache_seed": seeded,
              "stored": {"dir": TS.rel(args.stored), "record": TS.rel(args.record),
                         "record_sha256": TS.sha256(args.record), "sha256": stored_sha, "reproduces_record": True},
              "arms": {cand.name: {}}, "recall_by_class": {}, "comparisons": {}, "vs_ref": {}}
    for m in models:
        cu = TS.load_units(out / "private" / cand.name / f"units_{m}.json")["units"]
        sc = TP.scored(cu, report_ids)
        result["arms"][cand.name][m] = TS.summarize(sc)
        result["arms"][cand.name][m]["all_frames_per_part"] = TS.summarize(cu)["per_part"]
        result["recall_by_class"][m] = {cand.name: R.recall_by_class(sc),
                                        REF_ARM: R.recall_by_class(TP.scored(stored[REF_ARM][m]["units"], report_ids))}
        ca = TS.rule_arm(sc)
        result["comparisons"][m] = {a: M.compare(ca, TS.rule_arm(TP.scored(stored[a][m]["units"], report_ids)))
                                    for a in STORED_ARMS}
        vr = result["comparisons"][m][REF_ARM]
        result["vs_ref"][m] = {"comparison": REF_ARM, "dialogue_resolved_better_and_guards_hold": R.qualifies(vr),
                               "bar": all(M.beats(result["comparisons"][m][b]) for b in TS.BASELINES)}
    TS.assert_public(result)
    TS.write_json(out / "tests.json", result, indent=1)
    for m in models:
        s = result["arms"][cand.name][m]
        log(f"{cand.name} {m}: F1 {TS.fmt(s['detection']['f1'])} CER all {s['cer_all']['cer']} dialogue "
            f"{s['cer_dialogue']['cer']}")
        for a, c in result["comparisons"][m].items():
            log(f"  - {a}: F1 {c['f1']['delta']:+.4f} {c['f1']['ci95']}  CER {c['cer_all']['delta']} "
                f"{c['cer_all']['ci95']}  dialogue {c['cer_dialogue']['delta']} {c['cer_dialogue']['ci95']}")
    return result


# ----------------------------------------------------------------------------- CLANNAD

def stored_clannad_counts(stored, lines, refs):
    """Each stored arm's counts of textspot's CLANNAD record that need no pixel: boxes, vertical boxes, boxes on the
    pages the references hold no text for, and its C parity pages (from its stored boxes, the line list and the
    references' status)."""
    pages = {}
    for r in lines:
        pages.setdefault(r["page"], []).append(r)
    out = {}
    for a, doc in stored.items():
        c = dict.fromkeys(COUNT_KEYS, 0)
        for page, rs in sorted(pages.items()):
            fr = doc["frames"][page]
            c["boxes"] += len(fr["boxes"])
            c["vertical"] += sum(1 for b in fr["boxes"] if b[4])
            ce = fr.get("c_equal")
            c["c_equal_pages" if ce is True else "c_differ_pages" if ce is False else "c_unchecked_pages"] += 1
            if not any(refs.get(r["id"], {}).get("status") == "ref" for r in rs):
                c["boxes_on_no_text_pages"] += len(fr["boxes"])
        out[a] = c
    return out, len(pages)


def cmd_clannad(args, readers=None, grad_fn=None, log=print):
    import clannad_refs as CR
    import vocr3_report as V

    from common import to_gray
    proto, proto3 = TS.protocol(args.protocol), T3E.protocol3(args.protocol3)
    if proto3.get("kind") != "vita-vn-ocr-textspot3-protocol":
        raise SystemExit(f"{args.protocol3}: not textspot3's protocol")
    out = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else P3 / "cache"
    T3E.check_writable(out, cache_dir)
    work = Path(args.work)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if TS.sha256(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    if record["refs_sha256"] != freeze["refs_sha256"] or record["candidate"]["arm"] != REF_ARM:
        raise SystemExit("the stored CLANNAD record is not textspot's read of ref on these references")
    cand, version = cand_spec(args.cand, args.decode or None, args.runs)
    pick = check_pick(args.dev, cand)
    models = TS.parse_models(args.models)
    if not models or list(record["models"])[:len(models)] != models:
        raise SystemExit(f"--models {models}: the stored record read {record['models']}")
    stored, stored_sha = {}, {}
    for a in STORED_ARMS:
        stored[a], _ = T2E.load_arm(Path(args.stored) / a, model="none")
        stored_sha[a] = stored[a]["_sha256"]
    refs = {x["id"]: x for x in CR.read_jsonl(work / "refs.jsonl")}
    lines, _ = CR.load_lines(work)
    counts0, n_pages = stored_clannad_counts(stored, lines, refs)
    diffs0 = [f"arms.{a}.{k}" for a in STORED_ARMS for k in COUNT_KEYS if counts0[a][k] != record["arms"][a][k]]
    if n_pages != record["counts"]["pages"]:
        diffs0.append("counts.pages")
    if diffs0:
        raise SystemExit(f"the stored arms do not reproduce textspot's CLANNAD record (nothing read): {diffs0[:5]}")
    rd = TP2.build_readers(models, proto, out, args.compiler, record, readers)
    src = Path(args.seed_from) if args.seed_from else seed_source()
    seeded = seed_private_cache(cache_dir, src)
    cdir = T3E.resolve_cbin(args.cbin, out, args.compiler)
    meta = meta_of(cand, version)
    TP.once(out / "clannad.attempt.json", {"started": TS.utc_now(), "refs_sha256": freeze["refs_sha256"],
                                           "protocol_sha256": TS.sha256(args.protocol),
                                           "protocol3_sha256": TS.sha256(args.protocol3), "candidate": meta,
                                           "blob_version": version, "pick": pick, "models": models,
                                           "stored_sha256": stored_sha, "code_sha256": T3E.code_identity()})
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
        with T3E.dispatch({version}):
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
    preds, counts, extra = TP2.clannad_preds(pages, frames, per_frame, grad, refs, names, models, rd, cache)
    first = preds[cand.name][models[0]]
    sel_lines = [x for x in refs.values() if x["status"] == "ref" and x["id"] in first]
    result = {"kind": "textspot3-clannad", "computed": TS.utc_now(), "refs_sha256": freeze["refs_sha256"],
              "protocol_sha256": TS.sha256(args.protocol), "protocol3_sha256": TS.sha256(args.protocol3),
              "candidate": meta, "blob_version": version, "pick": pick, "models": models,
              "recognizers_sha256": dict(rd.sha), "reader": getattr(rd, "identity", None),
              "code_sha256": T3E.code_identity(), "cache_seed": seeded, "scored_lines": len(sel_lines),
              "counts": extra, "arms": {cand.name: counts[cand.name]},
              "assignment": "a detection goes to the GRAD box (of the page's ROI, in frame pixels) it overlaps most "
                            "when that box holds at least half its area; a reference line reads its detections "
                            "joined in the arm's reading order", "scopes": {}}
    diffs = [k for k in ("scored_lines", "counts") if result[k] != record[k]]
    diffs += [f"arms.{a}" for a in STORED_ARMS if counts[a] != record["arms"][a]]
    for scope in ("dialogue", "all"):
        sel = [x for x in sel_lines if scope == "all" or not x.get("name")]
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
                        "counts_checked_before": list(COUNT_KEYS), "reproduces_record": not diffs,
                        "differences": diffs, "crops_read_now": new}
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
        p.add_argument("--cand", required=True, help="T3:RUN, det:NAME=VTSP or detref:NAME=VTSP")
        p.add_argument("--decode", default="", choices=("", "det", "det+refine"))
        p.add_argument("--dev", required=True, help="the committed selection record (eval/textspot3_dev_<date>.json)")
        p.add_argument("--out", required=True, help="$P3/protected/<pick>/" + name)
        p.add_argument("--runs", default=str(T3E.DEFAULT["runs"]))
        p.add_argument("--cbin", default="", help="a cbuild directory (textspot3.py cbuild), or 'auto'")
        p.add_argument("--models", default="N,S,L")
        p.add_argument("--workers", type=int, default=5)
        p.add_argument("--cache", default="", help="the reading cache (default $P3/cache)")
        p.add_argument("--seed-from", default="", help="the private cache copied once (default: $P2/cache when it "
                                                       "has shards, else $P/cache)")
        p.add_argument("--compiler", default="gcc")
        p.add_argument("--protocol", default=str(TS.PROTOCOL))
        p.add_argument("--protocol3", default=str(T3E.PROTOCOL3))
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

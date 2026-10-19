#!/usr/bin/env python3
"""textspot: the protected sets, once, after the selection is committed (eval/textspot_protocol.json "sets": "test"
and "protected"). Only the conductor runs it, once per command.

  textspot_protected.py tests --cand SPEC [--decode det|det+refine] --out DIR [--data EVAL] [--shots DIR ...]
                              [--cbin DIR|auto] [--models N,S,L] [--workers 5]
      Every frame of eval/manifests/textspot-test.jsonl, which a protocol amendment must pin: vita-test's and
      psp-test's screenshots as whole frames (the sources in eval's data directory, --data), the text-free test
      frames (--shots) and, report only, psp-test at 2x (the part "adrenaline", or rows marked report_only). The
      candidate (the selection record's winner or report candidate: SPEC as eval/textspot.py takes it, e.g.
      A:RUN@600 with --decode, B:RUN, det:NAME=VTSP) and every baseline run on every frame (the C drivers too with
      --cbin), read by VOCR5-N and the VOCR5-S / VOCR5-L report arms (vertical boxes by VOCR5-V), scored by the
      frozen metrics; candidate minus each baseline (textspot_metrics.compare: F1, CER all and dialogue, VN-cluster
      intervals) on the scored parts pooled, and every part's own summary.
  textspot_protected.py clannad --cand SPEC [--decode ...] --work CLANNAD_WORK --out DIR [--cbin ...] [--models ...]
      CLANNAD's long run against owner-drive-ingest's frozen references (eval/clannad_refs.py), whole frames in:
      every arm on each page's native frame. A reference line is a GRAD box of the page's ROI (the C finder,
      twin-checked, reproduces it, as eval/textdet_protected.py reads them); each detection goes to the GRAD box it
      overlaps most when that box holds at least half its area (textdet_protected.assign), and a reference line
      reads its detections joined in the arm's reading order. clannad_refs' scoring (metrics.norm, Levenshtein),
      whole-page bootstraps, dialogue (no speaker-name line) and all lines; candidate minus each baseline (paired
      page bootstrap). Also counted per arm: the boxes on the pages the references hold no text for, and the boxes
      outside every reference line's GRAD box on the others.

Each command runs once: its attempt record (OUT/*.attempt.json) is created exclusively before any protected pixel
is read and refuses a second run. Public outputs (OUT/tests.json, OUT/clannad.json) hold counts, rates and hashes
only; units and boxes go to OUT/private/, readings to the private cache ($P/cache).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot as TS  # noqa: E402
import textspot_metrics as M  # noqa: E402

TEST_MANIFEST = "eval/manifests/textspot-test.jsonl"
REPORT_PARTS = ("adrenaline",)
FORBIDDEN = TS.FORBIDDEN


def once(path, record):
    """The attempt record, created exclusively: a second attempt fails here, before anything is read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=1) + "\n")


def assert_public(value, where="result"):
    TS.assert_public(value, where)


def arm_specs(cand, decode=None):
    """The candidate's one arm (SPEC of eval/textspot.py; --decode picks det or det+refine where it expands to both)
    and the eight baselines."""
    cs = [s for s in TS.parse_arms(cand) if s.kind in ("det", "detref")]
    if decode:
        cs = [s for s in cs if s.decode == decode]
    if len(cs) != 1:
        raise SystemExit(f"--cand {cand!r} (--decode {decode}) names {len(cs)} candidate arms, not one")
    return cs[0], TS.parse_arms("baselines")


def recognizers(models, proto, out, compiler, readers=None):
    if readers is not None:
        return readers
    return TS.c_readers(TS.recognizer_paths(models, proto), Path(out) / "rbuild", compiler)


def cdir_of(args, out):
    if not args.cbin:
        return None
    if args.cbin == "auto":
        TS.cbuild(Path(out) / "cbuild", args.compiler)
        return Path(out) / "cbuild"
    return Path(args.cbin)


def test_sources(args):
    """Where the test frames' source screenshots are: eval's data directory (vita/shots, psp/shots, as
    eval/fetch.shot_path names them) and the text-free test shots (--shots)."""
    dirs = [Path(d) for d in args.shots]
    if args.data:
        dirs += [Path(args.data) / "vita" / "shots", Path(args.data) / "psp" / "shots"]
    return dirs + [TS.P / "sets" / "test" / "shots", TS.T / "sets" / "test" / "shots"]


def scored(units, report_ids):
    return [u for u in units if u["unit"] not in report_ids]


def cmd_tests(args, readers=None, log=print):
    proto = TS.protocol(args.protocol)
    out = Path(args.out)
    man = Path(args.manifest) if args.manifest else ROOT / TEST_MANIFEST
    digest = TS.sha256(man)
    pin = TS.pinned_sha(proto, TEST_MANIFEST)
    if pin is None or pin != digest:
        raise SystemExit(f"{man}: sha256 {digest} is not the test manifest a protocol amendment pins ({pin})")
    TS.check_baseline_blobs(proto)
    cand, bases = arm_specs(args.cand, args.decode)
    models = TS.parse_models(args.models)
    once(out / "tests.attempt.json", {"started": TS.utc_now(), "protocol_sha256": TS.sha256(args.protocol),
                                      "manifest_sha256": digest, "candidate": cand.meta(), "models": models,
                                      "code_sha256": TS.code_identity()})
    rows = TS.jsonl(man)
    report_ids = {r["id"] for r in rows if r.get("report_only") or r.get("part") in args.report_parts}
    frames = TS.load_frames(rows, test_sources(args))
    manifest = {"set": "test", "path": TS.rel(man), "sha256": digest, "frames": len(rows),
                "vns": len({r["vn"] for r in rows})}
    rd = recognizers(models, proto, out, args.compiler, readers)
    cache = TS.Cache(args.cache or TS.CACHES["test"])
    specs = [cand] + bases
    try:
        TS.evaluate(frames, specs, manifest, out / "private", models, rd, cache, cdir_of(args, out), args.workers,
                    log=log, proto=proto)
    finally:
        cache.close()
    result = {"kind": "textspot-protected-tests", "computed": TS.utc_now(), "manifest": manifest,
              "protocol_sha256": TS.sha256(args.protocol), "candidate": cand.meta(),
              "report_parts": sorted(args.report_parts), "report_only_frames": len(report_ids),
              "scored_frames": len(rows) - len(report_ids), "models": models, "bootstrap": M.BOOT,
              "recognizers_sha256": dict(rd.sha), "reader": getattr(rd, "identity", None),
              "code_sha256": TS.code_identity(), "arms": {}, "comparisons": {}}
    for m in models:
        units = {s.name: TS.load_units(out / "private" / s.name / f"units_{m}.json")["units"] for s in specs}
        for s in specs:
            result["arms"].setdefault(s.name, {})[m] = TS.summarize(scored(units[s.name], report_ids))
            result["arms"][s.name][m]["all_frames_per_part"] = TS.summarize(units[s.name])["per_part"]
        ca = TS.rule_arm(scored(units[cand.name], report_ids))
        result["comparisons"][m] = {b.name: M.compare(ca, TS.rule_arm(scored(units[b.name], report_ids)))
                                    for b in bases}
    assert_public(result)
    TS.write_json(out / "tests.json", result, indent=1)
    for m in models:
        s = result["arms"][cand.name][m]
        log(f"{cand.name} {m}: F1 {TS.fmt(s['detection']['f1'])} CER all {s['cer_all']['cer']} dialogue "
            f"{s['cer_dialogue']['cer']}")
        for b, c in result["comparisons"][m].items():
            log(f"  - {b}: F1 {c['f1']['delta']:+.4f} {c['f1']['ci95']}  CER {c['cer_all']['delta']} "
                f"{c['cer_all']['ci95']}  dialogue {c['cer_dialogue']['delta']} {c['cer_dialogue']['ci95']}")
    return result


# ----------------------------------------------------------------------------- CLANNAD

def assign(det_boxes, grad_boxes):
    """For each detection, the index of the GRAD box it lies mostly inside (the largest overlap, holding at least
    half its area), or None: eval/textdet_protected.assign."""
    import textdet_protected as TP
    return TP.assign(det_boxes, grad_boxes)


def c_grad(driver):
    """GRAD boxes of each grey ROI through the C finder, checked against the twin (eval/finder_tighten.lines_of)."""
    import finder_tighten as FT
    return lambda greys: [g[0] for g in FT.lines_of(driver, greys, 0)]


def twin_grad(greys):
    from common import find_lines_ex
    return [[TS.ints(b) for b in find_lines_ex(g, "grad")[0]] for g in greys]


def load_pages(work):
    """[(page, rows, frame RGB)] of the long run, every frame's file hash checked against the capture journal."""
    import clannad_refs as CR
    from PIL import Image
    rows, meta = CR.load_lines(work)
    session = Path(meta["session"])
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)
    out = []
    for page, rs in sorted(by_page.items()):
        src = rs[0]["source"]
        path = session / src["path"]
        if TS.sha256(path) != src["sha256"]:
            raise SystemExit(f"{src['path']}: frame hash differs from the capture journal")
        with Image.open(path) as im:
            rgb = np.ascontiguousarray(np.asarray(im.convert("RGB")))
        out.append((page, rs, rgb))
    return out


def cmd_clannad(args, readers=None, grad_fn=None, log=print):
    import clannad_refs as CR
    import vocr3_report as V

    from common import to_gray
    proto = TS.protocol(args.protocol)
    out = Path(args.out)
    work = Path(args.work)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if TS.sha256(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    TS.check_baseline_blobs(proto)
    cand, bases = arm_specs(args.cand, args.decode)
    models = TS.parse_models(args.models)
    once(out / "clannad.attempt.json", {"started": TS.utc_now(), "refs_sha256": freeze["refs_sha256"],
                                        "protocol_sha256": TS.sha256(args.protocol), "candidate": cand.meta(),
                                        "models": models, "code_sha256": TS.code_identity()})
    refs = {x["id"]: x for x in CR.read_jsonl(work / "refs.jsonl")}
    pages = load_pages(work)
    frames, rois = [], []
    for page, rs, rgb in pages:
        frames.append(TS.Frame({"id": page, "vn": "clannad", "part": "clannad", "refs": []}, to_gray(rgb),
                               TS.pixel_sha256(rgb)))
        x, y, w, h = rs[0]["roi"]
        rois.append(np.ascontiguousarray(frames[-1].grey[y:y + h, x:x + w]))
    if grad_fn is None:
        import finder_tighten as FT
        grad_fn = c_grad(FT.build(out / "fbuild", args.compiler)["driver"])
    grad = grad_fn(rois)
    manifest = {"set": "clannad", "path": TS.rel(work / "refs.jsonl"), "sha256": freeze["refs_sha256"],
                "frames": len(frames), "vns": 1}
    rd = recognizers(models, proto, out, args.compiler, readers)
    cache = TS.Cache(args.cache or TS.CACHES["test"])
    specs = [cand] + bases
    try:
        per_frame = TS.run_arms(frames, specs, manifest, out / "private", models, rd, cache, cdir_of(args, out),
                                args.workers, log=log, proto=proto)
    finally:
        cache.close()
    preds = {s.name: {m: {} for m in models} for s in specs}
    counts = {s.name: {"boxes": 0, "vertical": 0, "boxes_on_no_text_pages": 0, "boxes_outside_ref_boxes": 0,
                       "c_equal_pages": 0, "c_differ_pages": 0, "c_unchecked_pages": 0} for s in specs}
    extra = {"pages": len(pages), "pages_with_ref": 0, "no_text_pages": 0, "unmatched": 0, "matched": 0}
    for (page, rs, _), f, arms, gb in zip(pages, frames, per_frame, grad):
        x0, y0 = rs[0]["roi"][:2]
        gbf = [(x + x0, y + y0, w, h) for x, y, w, h in gb]
        ref_rows = [r for r in rs if refs.get(r["id"], {}).get("status") == "ref"]
        extra["pages_with_ref" if ref_rows else "no_text_pages"] += 1
        keyed = []
        for r in ref_rows:
            box = TS.ints(r["box"])
            if box in [TS.ints(b) for b in gb]:
                keyed.append((r, [TS.ints(b) for b in gb].index(box)))
                extra["matched"] += 1
            else:
                extra["unmatched"] += 1
        for s in specs:
            a = arms[s.name]
            c = counts[s.name]
            c["boxes"] += len(a["boxes"])
            c["vertical"] += sum(1 for b in a["boxes"] if b[4])
            ce = a.get("c_equal")
            c["c_equal_pages" if ce is True else "c_differ_pages" if ce is False else "c_unchecked_pages"] += 1
            if not ref_rows:
                c["boxes_on_no_text_pages"] += len(a["boxes"])
                continue
            owner = assign([b[:4] for b in a["boxes"]], gbf)
            ks = {k for _, k in keyed}
            c["boxes_outside_ref_boxes"] += sum(1 for o in owner if o not in ks)
            for m in models:
                texts = TS.texts_of(f, a["boxes"], m, rd, cache)
                for r, k in keyed:
                    preds[s.name][m][r["id"]] = "".join(t for t, o in zip(texts, owner) if o == k)
    first = preds[cand.name][models[0]] if models else {}
    lines = [x for x in refs.values() if x["status"] == "ref" and x["id"] in first]
    result = {"kind": "textspot-clannad", "computed": TS.utc_now(), "refs_sha256": freeze["refs_sha256"],
              "protocol_sha256": TS.sha256(args.protocol), "candidate": cand.meta(), "models": models,
              "recognizers_sha256": dict(rd.sha), "reader": getattr(rd, "identity", None),
              "code_sha256": TS.code_identity(), "scored_lines": len(lines), "counts": extra, "arms": counts,
              "assignment": "a detection goes to the GRAD box (of the page's ROI, in frame pixels) it overlaps most "
                            "when that box holds at least half its area; a reference line reads its detections "
                            "joined in the arm's reading order", "scopes": {}}
    for scope in ("dialogue", "all"):
        sel = [x for x in lines if scope == "all" or not x.get("name")]
        pg = [x["page"] for x in sel]
        ids = sorted(set(pg))
        e = {"lines": len(sel), "pages": len(ids), "systems": {}, "pairs": {}}
        if ids:
            idx = CR.boot_idx(len(ids), CR.REPS, CR.SEED)
            for m in models:
                sums = {}
                for s in specs:
                    u = [CR.line_unit(x["text"], preds[s.name][m][x["id"]]) for x in sel]
                    sums[s.name] = CR.page_sums(u, pg)
                    e["systems"][f"{s.name}/{m}"] = CR.summary(sums[s.name], ids, idx)
                for b in bases:
                    a_ = {p: (sums[cand.name][p][0], sums[cand.name][p][1]) for p in ids}
                    b_ = {p: (sums[b.name][p][0], sums[b.name][p][1]) for p in ids}
                    d = V.paired_bootstrap(a_, b_, CR.REPS, CR.SEED)
                    e["pairs"][f"{cand.name}/{m}:{b.name}/{m}"] = {
                        "cer_delta": d["delta"], "cer_ci95": d["ci95"], "candidate_errors": d["candidate_errors"],
                        "baseline_errors": d["baseline_errors"], "chars": d["chars"]}
        result["scopes"][scope] = e
    assert_public(result)
    TS.write_json(out / "clannad.json", result, indent=1)
    for k, v in result["scopes"]["dialogue"]["systems"].items():
        log(f"{k}: dialogue CER {v['cer']} {v['cer_ci95']}")
    log(json.dumps(extra))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tests", "clannad"):
        p = sub.add_parser(name)
        p.add_argument("--cand", required=True, help="the selected or report candidate (eval/textspot.py SPEC)")
        p.add_argument("--decode", default="", choices=("", "det", "det+refine"))
        p.add_argument("--out", required=True)
        p.add_argument("--cbin", default="", help="a cbuild directory, or 'auto'")
        p.add_argument("--models", default="N,S,L")
        p.add_argument("--workers", type=int, default=5)
        p.add_argument("--cache", default="")
        p.add_argument("--compiler", default="gcc")
        p.add_argument("--protocol", default=str(TS.PROTOCOL))
        if name == "tests":
            p.add_argument("--manifest", default="")
            p.add_argument("--data", default=str(TS.ART / "eval"), help="eval's data directory (vita/shots, psp/shots)")
            p.add_argument("--shots", action="append", default=[], help="more source directories (text-free shots)")
            p.add_argument("--report-parts", nargs="*", default=list(REPORT_PARTS))
        else:
            p.add_argument("--work", required=True)
    args = ap.parse_args(argv)
    {"tests": cmd_tests, "clannad": cmd_clannad}[args.cmd](args)


if __name__ == "__main__":
    main()

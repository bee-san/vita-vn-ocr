#!/usr/bin/env python3
"""VOCR3 adapter for the CLANNAD long run (eval/clannad_longrun.py session): any pinned W8I blobs.

  vocr3_clannad.py prepare --session DIR --spec SPEC.json --blobs DIR --directory NEW_DIR [--pair CAND:BASE]
  vocr3_clannad.py verify  --prepared NEW_DIR/prepared.json
  vocr3_clannad.py compare --prepared NEW_DIR/prepared.json --authorization AUTH.json

The session is the owner's retained private long run: lock.json, captures.jsonl, comparisons.jsonl and
the stored inputs/ PNGs (docs/clannad-longrun-results-20261012.md). By default the three files must
have the published hashes (eval/reproducibility/clannad-vocr2-20261012/README.md) and the published
counts (2,204 captures, 2,177 successful pages, 27 excluded, 4,271 lines; 36,915 / 40,114 reference
characters; cached FL10 3,176 / 5,618 edit disagreements). The comparison is the one the released
VOCR2 run used, for every spec arm:

  - paired line: each of the original processed line PNGs (polarity already applied, so invert=False)
    against the cached Google reading of the same pixels;
  - whole ROI: each original RGB ROI PNG, grey, through the arm's own C finder (all returned rows, up
    to 16; truncation is fatal), dark rows inverted once, recognized, joined in finder order, against
    the cached whole-ROI Google reading.

The 27 originally failed pages stay excluded for every arm; nothing is retried. The cached FL10
predictions are the arm "cached" (no forwards); an arm with the lock's model hash also reports how
many of its readings equal the cache. Pairs CAND:BASE (spec pairs, plus --pair, BASE may be
"cached") get paired bootstraps over lines and pages. CLANNAD is one title, so there are no VN
clusters. Google is an automated reference proxy, not hand-checked ground truth: these are edit
disagreements, never CER, and they cannot qualify or select a model. Same one-shot protocol as
vocr3_fresh.py: a new preparation directory (zero forwards), a reviewer-written authorization equal to
vocr3_report.expected_authorization(prepared, KIND), one attempt, private predictions.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import clannad_longrun as LR  # noqa: E402
import vocr3_report as R  # noqa: E402

KIND = "clannad-longrun-vocr3"
CODE = tuple(sorted(set(R.CODE) | {"eval/vocr3_clannad.py", "eval/clannad_longrun.py"}))
PUBLISHED = {
    "lock_sha256": "14ba9297269b1cc741a9993425b3f901a1086bed74be4eaad274381a83ac6425",
    "captures_sha256": "b6cd582c582d22af883710f202f60f9c44848be23c4a8fe2f726799c7f8f81ed",
    "comparisons_sha256": "f2754897c8b69f555706b93a3cd4f709b8066c4a7578c03baf828458db584cd9",
    "baseline_model_sha256": LR.BASELINE_SHA,
    "counts": {"captured": 2204, "successful_pages": 2177, "excluded_failed_pages": 27, "paired_lines": 4271,
               "paired_reference_chars": 36915, "whole_roi_reference_chars": 40114,
               "cached_paired_edits": 3176, "cached_whole_roi_edits": 5618},
}
PROTOCOL = {
    "schema": 1, "kind": KIND, "split": "test",
    "reference": "Google Lens automated proxy (cached); not hand-checked ground truth or CER",
    "paired_line": "original processed line PNGs, invert=False, vs cached same-line Google text",
    "whole_roi": "original RGB ROI PNGs, grey, each arm's C finder (<= 16 rows, truncation fatal), dark rows "
                 "inverted once, joined in finder order, vs cached whole-ROI Google text",
    "exclusions": "originally failed/empty-reference pages excluded for every arm, retained, never retried",
    "baseline": "cached released FL10 predictions (arm 'cached'), no baseline forwards",
    "normalization": "eval.metrics.norm; small kana, dakuten and similar kanji stay distinct",
    "pair_statistics": "CAND - BASE in points; paired bootstrap over lines (paired) and pages (whole ROI)",
    "model_selection": False, "promotion": "none; a Google proxy alone cannot qualify a model",
    "one_attempt": True, "retry": False,
}
CACHED = "cached"


def score_rows(rows):
    """fixed_compare.score: reference chars, edit disagreements and hiragana counts (no text)."""
    chars = sum(len(LR.norm(r["text"])) for r in rows)
    errors = sum(LR.lev(LR.norm(r["text"]), LR.norm(r["pred"])) for r in rows)
    h = LR.score_hiragana(rows)
    h.pop("confusions")
    h["disagreement_rate"] = h.pop("cer")
    return {"units": len(rows), "reference_chars": chars, "edit_disagreements": errors,
            "disagreement_rate": errors / chars if chars else None, "hiragana": h}


def unit(row):
    h = LR.score_hiragana([row])
    return [LR.lev(LR.norm(row["text"]), LR.norm(row["pred"])), len(LR.norm(row["text"])), h["errors"], h["chars"]]


def bound_session(session, expect):
    """Pins, hash chains and the cache binding of fixed_compare.bound_cache; returns pages in capture order."""
    for name, key in (("lock.json", "lock_sha256"), ("captures.jsonl", "captures_sha256"),
                      ("comparisons.jsonl", "comparisons_sha256")):
        if R.sha256(session / name) != expect[key]:
            raise ValueError(f"{name} differs from the expected pin {expect[key][:12]}…")
    lock = json.loads((session / "lock.json").read_text())
    LR.require_test(lock.get("split"))
    lock_sha = R.sha256(session / "lock.json")
    if lock.get("schema") != 1 or lock["runtime"]["model_sha256"] != expect["baseline_model_sha256"]:
        raise ValueError("lock does not bind the expected cached baseline model")
    captures = [r for r in LR.read_journal(session / "captures.jsonl") if r.get("kind") == "capture"]
    comparisons = LR.read_journal(session / "comparisons.jsonl")
    if len({c["event_sha256"] for c in captures}) != len(captures) or \
            len({c["signature"] for c in captures}) != len(captures):
        raise ValueError("duplicate capture events or foreground signatures")
    by_capture = {c["event_sha256"]: c for c in captures}
    if len(comparisons) != len(captures) or {r["capture_sha256"] for r in comparisons} != set(by_capture):
        raise ValueError("pending, extra or retried comparisons")
    order = {c["event_sha256"]: i for i, c in enumerate(captures)}
    ok, excluded = [], []
    for result in sorted(comparisons, key=lambda r: order[r["capture_sha256"]]):
        capture = by_capture[result["capture_sha256"]]
        LR.require_test(result.get("split"))
        LR.require_test(capture.get("split"))
        if result.get("kind") != "comparison" or result["lock_sha256"] != lock_sha or \
                capture["lock_sha256"] != lock_sha or len(result["lines"]) != len(capture["lines"]):
            raise ValueError("comparison/capture protocol or line binding differs")
        for entry, line in zip(capture["lines"], result["lines"], strict=True):
            for ek, rk in (("sha256", "input_sha256"), ("pixels_sha256", "pixels_sha256"), ("box", "box"),
                           ("dark", "dark")):
                if entry[ek] != line[rk]:
                    raise ValueError("line cache/pixel/geometry/polarity binding differs")
        whole = result["whole_roi_google"]
        good = bool(result["lines"]) and whole.get("ok") is True and bool(LR.norm(whole.get("text"))) and all(
            line[who].get("ok") is True for line in result["lines"] for who in ("google", "vocr"))
        if result["ok"] is not good:
            raise ValueError("cached success status is inconsistent")
        (ok if good else excluded).append((capture, result))
    return lock, ok, excluded


def cached_scores(ok):
    lines = [{"text": ln["google"]["text"], "pred": ln["vocr"]["text"]} for _, r in ok for ln in r["lines"]]
    roi = [{"text": r["whole_roi_google"]["text"], "pred": "".join(ln["vocr"]["text"] for ln in r["lines"])}
           for _, r in ok]
    return {"paired_line_proxy": score_rows(lines), "whole_roi_proxy": score_rows(roi)}


def counts_of(ok, excluded, baseline):
    return {"captured": len(ok) + len(excluded), "successful_pages": len(ok), "excluded_failed_pages": len(excluded),
            "paired_lines": sum(len(c["lines"]) for c, _ in ok),
            "paired_reference_chars": baseline["paired_line_proxy"]["reference_chars"],
            "whole_roi_reference_chars": baseline["whole_roi_proxy"]["reference_chars"],
            "cached_paired_edits": baseline["paired_line_proxy"]["edit_disagreements"],
            "cached_whole_roi_edits": baseline["whole_roi_proxy"]["edit_disagreements"]}


def check_pairs(pairs, arm_names):
    out = []
    for p in pairs:
        p = R.parse_pair(p) if isinstance(p, str) else dict(p)
        known = set(arm_names) | {CACHED}
        if p["candidate"] not in arm_names or p["baseline"] not in known or p["candidate"] == p["baseline"]:
            raise ValueError(f"pair {p}: candidate must be a blob arm, baseline a blob arm or {CACHED!r}")
        out.append({"candidate": p["candidate"], "baseline": p["baseline"]})
    return out


def prepare(session, spec_path, blobs_dir, directory, pairs=(), expect=None, compiler="gcc"):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("a preparation directory must be new; never replace a preparation")
    session = Path(session).resolve()
    expect = dict(PUBLISHED if expect is None else expect)
    lock, ok, excluded = bound_session(session, expect)          # roles and bindings before pixels
    spec = R.build_spec(spec_path)
    arms = R.pinned_arms(spec, blobs_dir)
    pairs = [p for p in spec["pairs"]] + check_pairs(pairs, [a.name for a in arms])
    for capture, _ in ok:
        LR.verified_image(session, capture["roi"])
        for entry in capture["lines"]:
            LR.verified_image(session, entry)
    baseline = cached_scores(ok)
    counts = counts_of(ok, excluded, baseline)
    if expect.get("counts") is not None and counts != expect["counts"]:
        raise ValueError(f"session counts differ from the expected ones: {counts}")
    directory.mkdir(parents=True)
    native = R.build_native(directory / "native", compiler)
    lib = R.load_library(native)
    protocol = dict(PROTOCOL, spec=R.public_spec(spec), spec_sha256=R.digest(R.public_spec(spec)), pairs=pairs,
                    published_session=expect == PUBLISHED, expect={k: v for k, v in expect.items()})
    R.write_new(directory / "protocol.json", protocol)
    fp = {"lock": expect["lock_sha256"], "captures": expect["captures_sha256"],
          "comparisons": expect["comparisons_sha256"], "pages": [c["event_sha256"] for c, _ in ok]}
    prepared = {"schema": 1, "kind": KIND + "-prepared", "directory": str(directory), "session": str(session),
                "protocol_sha256": R.sha256(directory / "protocol.json"), "spec": str(Path(spec_path).resolve()),
                "spec_sha256": R.sha256(spec_path), "blobs": str(Path(blobs_dir).resolve()) if blobs_dir else None,
                "arms": {a.name: a.identity() for a in arms}, "resources": {a.name: R.resources(a, lib) for a in arms},
                "native": native, "code_sha256": R.code_identity(CODE), "environment": R.stable_environment(),
                "counts": counts, "baseline": baseline, "input_fingerprint": R.digest(fp),
                "lock_model_sha256": lock["runtime"]["model_sha256"], "verified_stored_inputs":
                    sum(len(c["lines"]) + 1 for c, _ in ok), "finder_calls": 0, "model_forwards": 0}
    R.write_new(directory / "prepared.json", prepared)
    return prepared


def verify(prepared_path, images=False):
    prepared_path = Path(prepared_path).resolve()
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("schema") != 1 or prepared.get("kind") != KIND + "-prepared":
        raise ValueError("not a vocr3 CLANNAD preparation")
    directory, session = Path(prepared["directory"]), Path(prepared["session"])
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    expect = protocol["expect"]
    lock, ok, excluded = bound_session(session, expect)
    spec = R.build_spec(prepared["spec"])
    fixed = {k: v for k, v in protocol.items() if k not in ("spec", "spec_sha256", "pairs", "published_session",
                                                             "expect")}
    fp = {"lock": expect["lock_sha256"], "captures": expect["captures_sha256"],
          "comparisons": expect["comparisons_sha256"], "pages": [c["event_sha256"] for c, _ in ok]}
    baseline = cached_scores(ok)
    if (prepared_path != directory / "prepared.json" or fixed != PROTOCOL
            or R.sha256(directory / "protocol.json") != prepared["protocol_sha256"]
            or protocol["spec"] != R.public_spec(spec) or R.sha256(prepared["spec"]) != prepared["spec_sha256"]
            or R.digest(fp) != prepared["input_fingerprint"] or baseline != prepared["baseline"]
            or counts_of(ok, excluded, baseline) != prepared["counts"]
            or R.code_identity(CODE) != prepared["code_sha256"]
            or R.stable_environment() != prepared["environment"]):
        raise ValueError("frozen protocol, session, cache, spec, code or environment changed")
    arms = R.pinned_arms(spec, prepared["blobs"])
    if {a.name: a.identity() for a in arms} != prepared["arms"]:
        raise ValueError("arm blobs changed since preparation")
    R.verify_native(prepared["native"])
    if images:
        for capture, _ in ok:
            LR.verified_image(session, capture["roi"])
            for entry in capture["lines"]:
                LR.verified_image(session, entry)
    return {"prepared": prepared, "protocol": protocol, "spec": spec, "session": session, "ok": ok, "arms": arms}


def score(state, record, reader_factory=None):
    prepared, session, ok, arms = state["prepared"], state["session"], state["ok"], state["arms"]
    native = prepared["native"]
    lib = R.load_library(native) if reader_factory is None else None
    readers = {a.name: (reader_factory(a) if reader_factory else R.CReader(a, lib)) for a in arms}
    per = {a.name: {"lines": [], "pages": [], "line_units": {}, "page_units": {}, "histogram": Counter(),
                    "blank_lines": 0, "blank_roi_lines": 0, "empty_pages": 0, "forwards": 0,
                    "same_line_as_cache": 0, "same_page_as_cache": 0} for a in arms}
    cache = {"line_units": {}, "page_units": {}}
    modes = sorted({a.finder for a in arms})
    for capture, result in ok:
        roi = LR.verified_image(session, capture["roi"])
        grey = np.ascontiguousarray(LR.to_gray(roi))
        found = {m: R.run_finder(native["finder"]["path"], [grey], R.FINDERS[m])[0] for m in modes}
        line_pixels = [LR.verified_image(session, e) for e in capture["lines"]]
        page = capture["event_sha256"]
        whole_ref = result["whole_roi_google"]["text"]
        cached_roi = "".join(ln["vocr"]["text"] for ln in result["lines"])
        cache["page_units"][page] = unit({"text": whole_ref, "pred": cached_roi})
        for i, ln in enumerate(result["lines"]):
            cache["line_units"][f"{page}:{i}"] = unit({"text": ln["google"]["text"], "pred": ln["vocr"]["text"]})
        for arm in arms:
            p = per[arm.name]
            texts = []
            for i, (pixels, ln) in enumerate(zip(line_pixels, result["lines"], strict=True)):
                t = readers[arm.name].text(pixels, False)
                texts.append(t)
                row = {"text": ln["google"]["text"], "pred": t}
                p["lines"].append(row)
                p["line_units"][f"{page}:{i}"] = unit(row)
                p["blank_lines"] += not LR.norm(t)
                p["same_line_as_cache"] += t == ln["vocr"]["text"]
            rc, boxes, dark, truncated = found[arm.finder]
            if rc != 0 or truncated or len(boxes) != len(dark):
                record({"page": page, "arm": arm.name, "status": "failed", "rc": rc, "truncated": truncated})
                raise RuntimeError("finder failure or truncation on a whole ROI is fatal; rows are never sliced")
            roi_texts = []
            for (x, y, w, h), d in zip(boxes, dark):
                crop = grey[LR.checked_box([x, y, w, h], grey.shape)].copy()
                roi_texts.append(readers[arm.name].text(255 - crop if d else crop, False))
            joined = "".join(roi_texts)
            row = {"text": whole_ref, "pred": joined}
            p["pages"].append(row)
            p["page_units"][page] = unit(row)
            p["histogram"][len(boxes)] += 1
            p["empty_pages"] += not boxes
            p["blank_roi_lines"] += sum(not LR.norm(t) for t in roi_texts)
            p["same_page_as_cache"] += joined == cached_roi
            record({"page": page, "arm": arm.name, "status": "ok", "paired": texts,
                    "whole_roi": [{"box": list(b), "dark": int(d), "text": t}
                                  for b, d, t in zip(boxes, dark, roi_texts)]})
    for arm in arms:
        per[arm.name]["forwards"] = readers[arm.name].forwards if hasattr(readers[arm.name], "forwards") else None
    return per, cache


def pair_stats(a_units, b_units, bootstrap):
    out = {}
    for metric, (e, n) in {"full": (0, 1), "hira": (2, 3)}.items():
        a = {k: (v[e], v[n]) for k, v in a_units.items()}
        b = {k: (v[e], v[n]) for k, v in b_units.items()}
        out[metric] = R.paired_bootstrap(a, b, bootstrap["reps"], bootstrap["seed"])
    return out


def execute(state, record, reader_factory=None):
    prepared, protocol, spec = state["prepared"], state["protocol"], state["spec"]
    per, cache = score(state, record, reader_factory)
    arms = {}
    for name, p in per.items():
        entry = {"sha256": prepared["arms"][name]["sha256"], "finder": prepared["arms"][name]["finder"],
                 "paired_line_proxy": score_rows(p["lines"]), "whole_roi_proxy": score_rows(p["pages"]),
                 "whole_roi_line_count_histogram": {str(k): v for k, v in sorted(p["histogram"].items())},
                 "whole_roi_empty_pages": p["empty_pages"], "blank_paired_lines": p["blank_lines"],
                 "blank_whole_roi_lines": p["blank_roi_lines"], "recognizer_calls": p["forwards"]}
        if prepared["arms"][name]["sha256"] == prepared["lock_model_sha256"]:
            entry["cache_agreement"] = {"paired_lines_identical": p["same_line_as_cache"],
                                        "paired_lines": len(p["lines"]),
                                        "whole_roi_pages_identical": p["same_page_as_cache"],
                                        "pages": len(p["pages"])}
        arms[name] = entry
    units = {name: (p["line_units"], p["page_units"]) for name, p in per.items()}
    units[CACHED] = (cache["line_units"], cache["page_units"])
    pairs = {}
    for p in protocol["pairs"]:
        c, b = units[p["candidate"]], units[p["baseline"]]
        pairs[f"{p['candidate']}:{p['baseline']}"] = {
            "paired_line": {"lines": pair_stats(c[0], b[0], spec["bootstrap"])},
            "whole_roi": {"pages": pair_stats(c[1], b[1], spec["bootstrap"])},
            "vn_clusters": "not applicable: one title (CLANNAD)"}
    result = {"schema": 1, "kind": KIND + "-result", "status": "complete", "protocol": PROTOCOL,
              "protocol_sha256": prepared["protocol_sha256"], "published_session": protocol["published_session"],
              "session_pins": {k: v for k, v in protocol["expect"].items() if k != "counts"},
              "counts": prepared["counts"], "cached_baseline": prepared["baseline"],
              "arms_identity": prepared["arms"], "resources": prepared["resources"], "arms": arms, "pairs": pairs,
              "native": R.public_native(prepared["native"]), "code_sha256": prepared["code_sha256"],
              "reference": PROTOCOL["reference"], "promotion_decision": PROTOCOL["promotion"],
              "training_or_selection_use": False, "teacher_requests": 0}
    result["markdown"] = markdown(result)
    return result


def markdown(result):
    md = ["Google is an automated proxy: edit disagreements, not CER.", "",
          "| arm | paired line: disagreements / reference chars | hiragana | whole ROI: disagreements / reference "
          "chars | hiragana |", "|---|---|---|---|---|"]
    rows = [(CACHED, result["cached_baseline"])] + list(result["arms"].items())
    for name, r in rows:
        pl, wr = r["paired_line_proxy"], r["whole_roi_proxy"]
        md.append(f"| {name} | {pl['edit_disagreements']}/{pl['reference_chars']} | "
                  f"{pl['hiragana']['errors']}/{pl['hiragana']['chars']} | {wr['edit_disagreements']}/"
                  f"{wr['reference_chars']} | {wr['hiragana']['errors']}/{wr['hiragana']['chars']} |")
    return "\n".join(md)


def compare(prepared_path, authorization_path):
    return R.run_once(prepared_path, authorization_path, KIND, lambda images: verify(prepared_path, images), execute)


def expect_from_args(args):
    if not any((args.expect_lock, args.expect_captures, args.expect_comparisons, args.expect_baseline)):
        return None
    if not all((args.expect_lock, args.expect_captures, args.expect_comparisons, args.expect_baseline)):
        raise SystemExit("override all four --expect-* pins together, or none (the published session)")
    return {"lock_sha256": args.expect_lock, "captures_sha256": args.expect_captures,
            "comparisons_sha256": args.expect_comparisons, "baseline_model_sha256": args.expect_baseline,
            "counts": None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("session", "spec", "directory"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--blobs", type=Path)
    p.add_argument("--pair", action="append", default=[], help=f"CANDIDATE:BASELINE (BASELINE may be {CACHED})")
    p.add_argument("--compiler", default="gcc")
    for name in ("lock", "captures", "comparisons", "baseline"):
        p.add_argument(f"--expect-{name}", help="another session's pin (default: the published CLANNAD session)")
    v = sub.add_parser("verify")
    v.add_argument("--prepared", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--prepared", type=Path, required=True)
    c.add_argument("--authorization", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        prepared = prepare(args.session, args.spec, args.blobs, args.directory, args.pair, expect_from_args(args),
                           args.compiler)
        print(json.dumps({"prepared": str(Path(args.directory) / "prepared.json"), "counts": prepared["counts"],
                          "input_fingerprint": prepared["input_fingerprint"]}))
    elif args.command == "verify":
        verify(args.prepared, images=True)
        print("preparation verified; zero finder calls and model forwards")
    else:
        print(compare(args.prepared, args.authorization)["markdown"])


if __name__ == "__main__":
    main()

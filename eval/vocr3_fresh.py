#!/usr/bin/env python3
"""VOCR3 adapter for the fresh native textbox test (eval/fresh_native_textbox.py): any pinned W8I blobs.

  vocr3_fresh.py prepare --manifest M.json --annotations A.json --spec SPEC.json --blobs DIR --directory NEW_DIR
  vocr3_fresh.py verify  --prepared NEW_DIR/prepared.json
  vocr3_fresh.py compare --prepared NEW_DIR/prepared.json --authorization AUTH.json

The cohort contract is the frozen harness's, reused unchanged: fresh_native_textbox.checked_inputs
(source roles, exposure-evidence binding, references, deterministic deduplication, the fixed minima
of three fresh titles per console, five frames, 500 body / 200 hiragana characters per title), its
image_rgb (file, decoded native RGB and per-ROI pixel hashes), its base/+4/+8 ROI variants and its
transcript_score (one alignment per frame; every unmatched insertion charged to body). What changes
is the comparison: each spec arm (blob + the finder it ships with) reads every variant of every
frame: all returned crops, in C order, with C polarity; zero detections score as an empty prediction.

prepare validates everything before any pixel, then hashes pixels, blobs, sources and the freshly
built scalar library and finder driver, and writes protocol.json and prepared.json into a new
directory: zero finder calls, zero model forwards. compare runs once, after a reviewer writes the
authorization equal to vocr3_report.expected_authorization(prepared, KIND) (this CLI never writes it):
exclusive attempt.json, private-predictions.jsonl (private: reference-free but holds model text), and
result.json with integer per-title counters per arm and variant, paired bootstraps (frames; title
clusters) for the spec's pairs, and the VOCR2 whole-dialogue guard for the spec's gates. No model
selection, tuning, retry or default change follows from this file; the protocol decides how a result
is used. Keep manifests, annotations, images and the run directory outside Git.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import fresh_native_textbox as fresh  # noqa: E402
import vocr3_report as R  # noqa: E402
from metrics import norm  # noqa: E402

KIND = "fresh-native-textbox-vocr3"
CODE = tuple(sorted(set(R.CODE) | {"eval/vocr3_fresh.py", "eval/fresh_native_textbox.py"}))
PROTOCOL = {
    "schema": 1, "kind": KIND, "split": "test",
    "cohort_contract": "eval/fresh_native_textbox.py checked_inputs/image_rgb, unchanged (roles, exclusions, "
                       "dedup, fixed minima)",
    "minimums": fresh.MINIMUMS, "variants": fresh.VARIANTS,
    "pipeline": "each arm's C finder on the RGBA ROI; every returned crop, C order, C polarity; W8I max width "
                "1024; zero detections = empty prediction; finder failure/truncation aborts",
    "scoring": "fresh_native_textbox.transcript_score per frame (all/body x full/hiragana)",
    "pairs": "CAND - BASE in points; paired bootstrap over frames and over title clusters",
    "gates": "VOCR2 whole-dialogue guard (vocr3_report.dialogue_guard) per spec gate",
    "model_selection": False, "runtime_promotion": False, "one_attempt": True, "retry": False,
    "original_training_history": "unknown; recorded-manifest exclusions do not prove historical training "
                                 "independence",
}


def source_fingerprint(sources):
    return R.digest([{k: v for k, v in s.items() if k not in ("_lines", "_image")} for s in sources])


def reference_fingerprint(sources):
    return R.digest([s["_reference_sha256"] for s in sources])


def prepare(manifest, annotations, spec_path, blobs_dir, directory, compiler="gcc"):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("a preparation directory must be new; never replace a preparation")
    document, sources, readiness = fresh.checked_inputs(manifest, annotations)   # roles before pixels
    spec = R.build_spec(spec_path)
    arms = R.pinned_arms(spec, blobs_dir)
    for source in sources:
        fresh.image_rgb(source)
    directory.mkdir(parents=True)
    native = R.build_native(directory / "native", compiler)
    lib = R.load_library(native)
    protocol = dict(PROTOCOL, spec=R.public_spec(spec), spec_sha256=R.digest(R.public_spec(spec)))
    R.write_new(directory / "protocol.json", protocol)
    fp = {"sources": source_fingerprint(sources), "references": reference_fingerprint(sources)}
    prepared = {
        "schema": 1, "kind": KIND + "-prepared", "directory": str(directory),
        "protocol_sha256": R.sha256(directory / "protocol.json"),
        "manifest": str(Path(manifest).resolve()), "manifest_sha256": R.sha256(manifest),
        "annotations": str(Path(annotations).resolve()), "annotations_sha256": R.sha256(annotations),
        "exposure_evidence_sha256": document["exclusions"]["evidence_sha256"],
        "spec": str(Path(spec_path).resolve()), "spec_sha256": R.sha256(spec_path),
        "blobs": str(Path(blobs_dir).resolve()) if blobs_dir else None,
        "arms": {a.name: a.identity() for a in arms}, "resources": {a.name: R.resources(a, lib) for a in arms},
        "native": native, "code_sha256": R.code_identity(CODE), "environment": R.stable_environment(),
        "readiness": readiness, "fingerprints": fp, "input_fingerprint": R.digest(fp),
        "finder_calls": 0, "model_forwards": 0,
    }
    R.write_new(directory / "prepared.json", prepared)
    return prepared


def verify(prepared_path, images=False):
    prepared_path = Path(prepared_path).resolve()
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("schema") != 1 or prepared.get("kind") != KIND + "-prepared":
        raise ValueError("not a vocr3 fresh preparation")
    document, sources, readiness = fresh.checked_inputs(prepared["manifest"], prepared["annotations"])
    directory = Path(prepared["directory"])
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    spec = R.build_spec(prepared["spec"])
    if (prepared_path != directory / "prepared.json"
            or R.sha256(directory / "protocol.json") != prepared["protocol_sha256"]
            or {k: v for k, v in protocol.items() if k not in ("spec", "spec_sha256")} != PROTOCOL
            or protocol["spec"] != R.public_spec(spec)
            or R.sha256(prepared["manifest"]) != prepared["manifest_sha256"]
            or R.sha256(prepared["annotations"]) != prepared["annotations_sha256"]
            or R.sha256(prepared["spec"]) != prepared["spec_sha256"]
            or document["exclusions"]["evidence_sha256"] != prepared["exposure_evidence_sha256"]
            or readiness != prepared["readiness"]
            or {"sources": source_fingerprint(sources), "references": reference_fingerprint(sources)}
            != prepared["fingerprints"]
            or R.code_identity(CODE) != prepared["code_sha256"]
            or R.stable_environment() != prepared["environment"]):
        raise ValueError("frozen protocol, cohort, references, spec, code or environment changed")
    arms = R.pinned_arms(spec, prepared["blobs"])
    if {a.name: a.identity() for a in arms} != prepared["arms"]:
        raise ValueError("arm blobs changed since preparation")
    R.verify_native(prepared["native"])
    if images:
        for source in sources:
            fresh.image_rgb(source)
    return {"prepared": prepared, "spec": spec, "sources": sources, "arms": arms}


def score(state, record, reader_factory=None):
    """Every arm on every source and variant. Returns groups/units/accounting (integer counts)."""
    prepared, sources, arms = state["prepared"], state["sources"], state["arms"]
    native = prepared["native"]
    lib = R.load_library(native) if reader_factory is None else None
    readers = {a.name: (reader_factory(a) if reader_factory else R.CReader(a, lib)) for a in arms}
    groups = {a.name: {v: {} for v in fresh.VARIANTS} for a in arms}
    units = {a.name: [] for a in arms}
    acc = {a.name: {"forwards": 0, "detections": 0, "blank_outputs": 0, "zero_detection_frames": 0} for a in arms}
    modes = sorted({a.finder for a in arms})
    for source in sources:
        rgb = fresh.image_rgb(source)
        key = source["platform"] + ":" + source["vn"]
        title_ref = R.digest([s["_reference_sha256"] for s in sources if s["platform"] + ":" + s["vn"] == key])
        cases = []
        for variant, value in source["variants"].items():
            x, y, w, h = value["roi"]
            cases.append((variant, np.ascontiguousarray(np.concatenate(
                [rgb[y:y + h, x:x + w], np.full((h, w, 1), 255, np.uint8)], axis=2))))
        found = {m: R.run_finder(native["finder"]["path"], [c[1] for c in cases], R.FINDERS[m]) for m in modes}
        for i, (variant, rgba) in enumerate(cases):
            for arm in arms:
                rc, boxes, dark, truncated = found[arm.finder][i]
                if rc != 0 or truncated or len(boxes) != len(dark) or any(d not in (0, 1) for d in dark):
                    record({"source_id": source["id"], "arm": arm.name, "variant": variant, "status": "failed",
                            "failure_stage": "finder", "rc": rc, "truncated": truncated})
                    raise RuntimeError("finder error/truncation/polarity; never coerced to a blank success")
                preds = []
                for box, d in zip(boxes, dark):
                    a, b, c, e = fresh.checked_box(list(box), [rgba.shape[1], rgba.shape[0]])
                    text = readers[arm.name].text(np.ascontiguousarray(rgba[b:b + e, a:a + c]), bool(d))
                    if not isinstance(text, str):
                        raise RuntimeError("reader returned no text")
                    preds.append(text)
                s = fresh.transcript_score(source["_lines"], preds)
                g = groups[arm.name][variant].setdefault(key, {
                    "platform": source["platform"], "vn": source["vn"], "frames": 0, "detections": 0,
                    "blank_outputs": 0, "zero_detection_frames": 0, "reference_sha256": title_ref,
                    **{p: {t: {"chars": 0, "errors": 0} for t in R.SCRIPTS} for p in R.SCOPES}})
                g["frames"] += 1
                g["detections"] += len(boxes)
                g["blank_outputs"] += sum(not norm(p) for p in preds)
                g["zero_detection_frames"] += not boxes
                for p in R.SCOPES:
                    for t in R.SCRIPTS:
                        for k in ("chars", "errors"):
                            g[p][t][k] += s[p][t][k]
                ac = acc[arm.name]
                ac["forwards"] += len(boxes)
                ac["detections"] += len(boxes)
                ac["blank_outputs"] += sum(not norm(p) for p in preds)
                ac["zero_detection_frames"] += not boxes
                units[arm.name].append({"unit": source["id"], "vn": source["vn"], "platform": source["platform"],
                                        "cluster": key, "variant": variant,
                                        **{f"{p}_{t}_{k}": s[p][t][k] for p in R.SCOPES for t in R.SCRIPTS
                                           for k in ("chars", "errors")}})
                record({"source_id": source["id"], "arm": arm.name, "variant": variant, "status": "ok",
                        "boxes": [list(b) for b in boxes], "dark": list(dark), "predictions": preds, "score": s})
    return groups, units, acc


def execute(state, record, reader_factory=None):
    prepared, spec, sources = state["prepared"], state["spec"], state["sources"]
    groups, units, acc = score(state, record, reader_factory)
    result = {"schema": 1, "kind": KIND + "-result", "status": "complete", "protocol": PROTOCOL,
              "protocol_sha256": prepared["protocol_sha256"], "spec_sha256": R.digest(R.public_spec(spec)),
              "arms": prepared["arms"], "resources": prepared["resources"], "readiness": prepared["readiness"],
              "fingerprints": prepared["fingerprints"], "manifest_sha256": prepared["manifest_sha256"],
              "annotations_sha256": prepared["annotations_sha256"],
              "exposure_evidence_sha256": prepared["exposure_evidence_sha256"],
              "native": R.public_native(prepared["native"]), "code_sha256": prepared["code_sha256"],
              "groups": groups, "totals": {a: R.dialogue_totals(g) for a, g in groups.items()},
              "accounting": acc, "pairs": {}, "gates": {},
              "source_manifest": [{k: s[k] for k in ("id", "url", "sha256", "pixel_sha256", "platform", "vn",
                                                     "release", "session", "dims", "reviewer_kind")}
                                  for s in sources],
              "model_selection": False, "runtime_promotion": False,
              "scope": "fresh native screenshot test; host C runtime; no hardware timing"}
    for p in spec["pairs"]:
        if p["candidate"] in units and p["baseline"] in units:
            result["pairs"][f"{p['candidate']}:{p['baseline']}"] = R.compare_dialogue(
                units[p["candidate"]], units[p["baseline"]], spec["bootstrap"])
    for g in spec["gates"]:
        result["gates"][f"{g['candidate']}:{g['baseline']}"] = R.dialogue_guard(groups[g["baseline"]],
                                                                               groups[g["candidate"]])
    result["markdown"] = markdown(result)
    return result


def markdown(result):
    md = ["| arm | variant | platform | all full | all hiragana | body full | body hiragana | detections |",
          "|---|---|---|---|---|---|---|---:|"]
    for arm, by_var in result["totals"].items():
        for variant, by_plat in by_var.items():
            for plat, t in by_plat.items():
                cells = [f"{t[s][x]['errors']}/{t[s][x]['chars']}" for s in R.SCOPES for x in R.SCRIPTS]
                md.append(f"| {arm} | {variant} | {plat} | " + " | ".join(cells) + f" | {t['detections']} |")
    for key, gate in result["gates"].items():
        md.append(f"\nGuard {key}: {'pass' if gate['passed'] else 'FAIL'}; {len(gate['failed_cells'])} of "
                  f"{gate['title_checks']} title checks regress.")
    return "\n".join(md)


def compare(prepared_path, authorization_path):
    return R.run_once(prepared_path, authorization_path, KIND, lambda images: verify(prepared_path, images), execute)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("manifest", "annotations", "spec", "directory"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--blobs", type=Path)
    p.add_argument("--compiler", default="gcc")
    v = sub.add_parser("verify")
    v.add_argument("--prepared", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--prepared", type=Path, required=True)
    c.add_argument("--authorization", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "prepare":
        prepared = prepare(args.manifest, args.annotations, args.spec, args.blobs, args.directory, args.compiler)
        print(json.dumps({"prepared": str(Path(args.directory) / "prepared.json"),
                          "input_fingerprint": prepared["input_fingerprint"]}))
    elif args.command == "verify":
        verify(args.prepared, images=True)
        print("preparation verified; zero finder calls and model forwards")
    else:
        result = compare(args.prepared, args.authorization)
        print(result["markdown"])


if __name__ == "__main__":
    main()

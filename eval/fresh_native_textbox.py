#!/usr/bin/env python3
"""One frozen, explicitly authorized fresh native textbox test; never select models."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from fractions import Fraction
from pathlib import Path

from group_metrics import MINIMUMS
from metrics import is_hiragana, norm

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84"
LIBRARY_SHA = "4a5635ba9a0721373f35e8957c0788b74e1725ea8cd96b8200cc5e427536b11b"
FINDER_SHA = "90c30a1c1e1707b818c2bd919243ee589700ff44ceac8b8d7a842b9d55d8d9ca"
SOURCE_COMMIT = "cfa7ee6ce5c381ab57dcbca37bbde931d3282e54"
VARIANTS = {"base": 0, "expanded-4": 4, "expanded-8": 8}
MODES = {"original_GRAD": 0, "pure_C3": 2, "C4": 3}
DIMENSIONS = {"psp": [480, 272], "vita": [960, 544]}
ELIGIBILITY = ("used_in_recorded_training", "used_in_calibration", "used_in_development",
               "related_to_used_title")
FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror")
CODE = ("eval/fresh_native_textbox.py", "eval/group_metrics.py", "eval/metrics.py",
        "eval/linefind/cfinder.py", "train/general_vn_textbox_guard.py",
        "train/general_vn_select.py", "runtime/pyvocr.py", "runtime/vocr.c", "runtime/vocr.h",
        "runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_test.c",
        "requirements.lock", "requirements-ci.lock")
PROTOCOL = {
    "schema": 1, "kind": "fresh-native-textbox-fixed-family-test", "split": "test",
    "candidate_source_commit": SOURCE_COMMIT, "candidate_source_sha256": FINDER_SHA,
    "model_sha256": MODEL_SHA, "scalar_library_sha256": LIBRARY_SHA,
    "modes": MODES, "variants": VARIANTS, "minimums": MINIMUMS,
    "model_selection": False, "runtime_promotion": False, "teacher": None,
    "blob_bytes": 657664, "arena_bytes": 124832, "classes": 3501,
    "mode": "W8I", "max_width": 1024,
    "normalization": "eval.metrics.norm separately per main reference line and returned crop; no renorm after join",
    "alignment": "existing transcript_score: diagonal, deletion, insertion ties",
    "body_attribution": "body reference substitutions/deletions and ALL unmatched insertions",
    "prediction_policy": "every C crop in output order and polarity; blanks/zero detections remain; errors abort",
    "ruby": "supplementary only; no main reference or predicted-crop filtering",
    "dedup": ("stable source-id order: unique RGB pixels globally; whole normalized body transcript "
              "and unique body lines within each platform/title for quotas"),
    "gate": "72+ title/variant all+body full+hira nonreg versus mode0; all12 pooled metrics nonreg versus mode2",
    "material_gain": "100*C4 pooled all full errors <= 90*mode2 errors EACH variant",
    "body_hiragana_gain": "strict console-balanced rate reduction EACH variant; >=2 improving titles per console",
    "weights": "consoles 1/2 each; titles equal within console; exact fractions",
    "original_training_history": "unknown; recorded-manifest exclusions do not prove historical training independence",
    "one_attempt": True, "retry": False,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze(path, value):
    """Exclusive, atomic JSON creation. Existing snapshots are never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checked_hash(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("require lowercase SHA256")
    return value


def checked_box(box, dims):
    if (not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box)
            or not isinstance(dims, list) or len(dims) != 2):
        raise ValueError("native box must contain four integers")
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > dims[0] or y + h > dims[1]:
        raise ValueError("native box exceeds source")
    return list(box)


def roi_variants(box, dims):
    x, y, w, h = checked_box(box, dims)
    return {name: [max(0, x - pad), max(0, y - pad),
                   min(dims[0], x + w + pad) - max(0, x - pad),
                   min(dims[1], y + h + pad) - max(0, y - pad)]
            for name, pad in VARIANTS.items()}


def checked_manifest(document):
    """Validate all source roles before annotation, image, model or library access."""
    if (not isinstance(document, dict) or type(document.get("schema")) is not int
            or document["schema"] != 1 or document.get("split") != "test"
            or document.get("purpose") != "fresh-native-textbox"
            or document.get("original_training_history") != "unknown"
            or not isinstance(document.get("sources"), list) or not document["sources"]):
        raise ValueError("require explicit fresh test manifest and unknown historical initializer provenance")
    exclusions = document.get("exclusions")
    if not isinstance(exclusions, dict) or type(exclusions.get("schema")) is not int or exclusions["schema"] != 1:
        raise ValueError("require frozen recorded-use/family exclusion evidence")
    checked_hash(exclusions.get("evidence_sha256"))
    if not isinstance(exclusions.get("evidence_file"), str) or not exclusions["evidence_file"]:
        raise ValueError("require exposure evidence file")
    for field in ("vns", "release_ids", "urls", "sha256"):
        if (not isinstance(exclusions.get(field), list) or len(set(exclusions[field])) != len(exclusions[field])
                or any(not isinstance(v, str) or not v for v in exclusions[field])):
            raise ValueError("exclusion lists must be explicit and unique")
    seen = set()
    for source in document["sources"]:
        if not isinstance(source, dict) or source.get("split") != "test":
            raise ValueError("every native source must retain test role")
        for key in ("id", "session", "image", "url"):
            if not isinstance(source.get(key), str) or not source[key]:
                raise ValueError("source identity/provenance missing")
        if (re.fullmatch(r"[A-Za-z0-9_.:-]+", source["id"]) is None
                or re.fullmatch(r"[A-Za-z0-9_.:-]+", source["session"]) is None
                or not source["url"].startswith("https://")):
            raise ValueError("source ID must be a stable slug and source URL must use HTTPS")
        if source["id"] in seen:
            raise ValueError("duplicate source id")
        seen.add(source["id"])
        if re.fullmatch(r"v[1-9][0-9]*", str(source.get("vn"))) is None:
            raise ValueError("invalid VNDB title identity")
        if re.fullmatch(r"r[1-9][0-9]*", str(source.get("release"))) is None:
            raise ValueError("invalid native release identity")
        if source.get("platform") not in DIMENSIONS or source.get("dims") != DIMENSIONS[source["platform"]]:
            raise ValueError("platform/dimensions must prove a native PSP or Vita source")
        checked_hash(source.get("sha256"))
        checked_hash(source.get("pixel_sha256"))
        if (source.get("hand_checked") is not True or source.get("complete_eligible_text") is not True
                or source.get("uncertain") is not False
                or source.get("reviewer_kind") not in ("human", "assistant")
                or not isinstance(source.get("scope_exclusions"), list)
                or any(not isinstance(s, str) for s in source["scope_exclusions"])):
            raise ValueError("complete source transcription and explicit review provenance required")
        if source.get("eligibility") != dict.fromkeys(ELIGIBILITY, False):
            raise ValueError("recorded training/calibration/development/family exposure excludes a source")
        for field, excluded in (("vn", "vns"), ("release", "release_ids"), ("url", "urls"),
                                ("sha256", "sha256")):
            if source[field] in exclusions[excluded]:
                raise ValueError("source intersects frozen recorded exposure exclusions")
        line_ids = source.get("line_ids")
        if (not isinstance(line_ids, list) or not line_ids or any(not isinstance(s, str) or not s for s in line_ids)
                or len(set(line_ids)) != len(line_ids)):
            raise ValueError("main reading order requires unique line IDs")
        expected = roi_variants(source.get("base_roi"), source["dims"])
        variants = source.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
            raise ValueError("all three fixed crop variants required")
        for name, box in expected.items():
            variant = variants[name]
            if not isinstance(variant, dict) or set(variant) != {"roi", "pixel_sha256"} or variant["roi"] != box:
                raise ValueError("ROI variant differs from symmetric native-pixel rule")
            checked_hash(variant["pixel_sha256"])
    return document


def checked_cohort(document, annotations):
    """Pure metadata/reference validation, deterministic dedup and distinct-text quotas."""
    checked_manifest(document)
    if (not isinstance(annotations, dict) or type(annotations.get("schema")) is not int or annotations["schema"] != 1
            or annotations.get("split") != "test" or not isinstance(annotations.get("lines"), list)
            or not isinstance(annotations.get("supplementary"), list)):
        raise ValueError("annotations must be private explicit test main lines and supplementary ruby")
    lines = {}
    for original in annotations["lines"]:
        if (not isinstance(original, dict) or original.get("split") != "test"
                or not isinstance(original.get("id"), str) or not original["id"]
                or original["id"] in lines or original.get("content_kind") not in ("body", "name")
                or original.get("uncertain") is not False or not isinstance(original.get("text"), str)
                or not norm(original["text"])):
            raise ValueError("main lines require unique IDs, certain test roles and explicit body/name text")
        lines[original["id"]] = dict(original, styles=["name"] if original["content_kind"] == "name" else [])
    bound, selected, removed, pixels, bodies = set(), [], [], {}, {}
    for source in sorted(document["sources"], key=lambda s: s["id"]):
        ordered = []
        for line_id in source["line_ids"]:
            if line_id not in lines or line_id in bound or lines[line_id].get("source_id") != source["id"]:
                raise ValueError("reading order must bind every main line exactly once to its source")
            line = lines[line_id]
            box = checked_box(line.get("box"), source["dims"])
            x, y, w, h = source["base_roi"]
            a, b, c, d = box
            if a < x or b < y or a + c > x + w or b + d > y + h:
                raise ValueError("base ROI clips labelled main ink")
            bound.add(line_id)
            ordered.append(line)
        body_text = "".join(norm(line["text"]) for line in ordered if line["content_kind"] == "body")
        if not body_text:
            raise ValueError("native textbox requires nonempty main body")
        body_sha = hashlib.sha256(body_text.encode()).hexdigest()
        title = (source["platform"], source["vn"])
        semantic = [(x["content_kind"], norm(x["text"]), x["box"]) for x in ordered]
        if source["pixel_sha256"] in pixels:
            prior = pixels[source["pixel_sha256"]]
            if prior["title"] != title or prior["semantic"] != semantic:
                raise ValueError("duplicate native pixels have conflicting provenance or main annotations")
            removed.append({"source_id": source["id"], "reason": "duplicate_rgb", "retained_id": prior["id"]})
            continue
        pixels[source["pixel_sha256"]] = {"id": source["id"], "title": title, "semantic": semantic}
        body_key = (*title, body_sha)
        if body_key in bodies:
            removed.append({"source_id": source["id"], "reason": "repeated_body", "retained_id": bodies[body_key]})
            continue
        source = dict(source, _lines=ordered, _body_sha256=body_sha, _reference_sha256=digest(ordered))
        bodies[body_key] = source["id"]
        selected.append(source)
    if bound != set(lines):
        raise ValueError("unbound main annotation lines")
    source_map = {s["id"]: s for s in document["sources"]}
    for ruby in annotations["supplementary"]:
        if not isinstance(ruby, dict) or ruby.get("kind") != "ruby" or ruby.get("source_id") not in source_map:
            raise ValueError("only separately bound supplementary ruby is allowed")
        checked_box(ruby.get("box"), source_map[ruby["source_id"]]["dims"])
    groups, body_lines = {}, set()
    for source in selected:
        key = source["platform"] + ":" + source["vn"]
        group = groups.setdefault(key, {"platform": source["platform"], "vn": source["vn"],
                                        "frames": 0, "body_chars": 0, "body_hiragana": 0})
        group["frames"] += 1
        for line in source["_lines"]:
            text = norm(line["text"])
            if line["content_kind"] != "body" or (key, text) in body_lines:
                continue
            body_lines.add((key, text))
            group["body_chars"] += len(text)
            group["body_hiragana"] += sum(is_hiragana(c) for c in text)
    reasons = []
    for key, group in groups.items():
        for field, quota in (("frames", "frames_per_vn"), ("body_chars", "body_chars_per_vn"),
                             ("body_hiragana", "body_hiragana_per_vn")):
            if group[field] < MINIMUMS[quota]:
                reasons.append(f"{key}: {field} below fixed minimum")
    for console in DIMENSIONS:
        if sum(g["platform"] == console for g in groups.values()) < MINIMUMS["vns_per_console"]:
            reasons.append(console + ": fewer than three fresh titles")
    readiness = {"ready": not reasons, "minimums": MINIMUMS, "groups": groups, "reasons": reasons,
                 "selected_sources": len(selected), "deduplicated_sources": removed}
    if reasons:
        raise ValueError("fresh cohort is ineligible: " + "; ".join(reasons))
    return selected, readiness


def checked_inputs(manifest, annotations):
    """Source-role checks deliberately precede even opening the annotation file."""
    document = checked_manifest(read(manifest))
    evidence = Path(document["exclusions"]["evidence_file"])
    if not evidence.is_absolute():
        evidence = Path(manifest).resolve().parent / evidence
    if sha256(evidence) != document["exclusions"]["evidence_sha256"]:
        raise ValueError("recorded exposure evidence changed")
    excluded = read(evidence).get("excluded")
    if not isinstance(excluded, dict) or any(
            sorted(document["exclusions"][key]) != sorted(excluded.get(key, []))
            for key in ("vns", "release_ids", "urls", "sha256")):
        raise ValueError("manifest exclusion lists differ from the frozen recorded evidence")
    sources, readiness = checked_cohort(document, read(annotations))
    for source in sources:
        source["_image"] = str((Path(manifest).resolve().parent / source["image"]).resolve())
    return document, sources, readiness


def image_rgb(source):
    """Only called after the whole cohort's metadata and reference roles pass."""
    import numpy as np
    from PIL import Image

    if sha256(source["_image"]) != source["sha256"]:
        raise ValueError("native source file changed")
    with Image.open(source["_image"]) as image:
        if list(image.size) != source["dims"]:
            raise ValueError("source is not the declared native size")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() != source["pixel_sha256"]:
        raise ValueError("native RGB pixel hash differs")
    for variant in source["variants"].values():
        x, y, w, h = variant["roi"]
        if hashlib.sha256(np.ascontiguousarray(rgb[y:y + h, x:x + w]).tobytes()).hexdigest() != variant["pixel_sha256"]:
            raise ValueError("fixed ROI pixel hash differs")
    return rgb


def validate_blob_bytes(data):
    """Static header/CRC check without loading a model or running inference."""
    if len(data) != 657664 or data[:4] != b"VOCR":
        raise ValueError("released FL10 blob size/magic differs")
    version, header, size, flags = struct.unpack_from("<HHII", data, 4)
    classes = struct.unpack_from("<I", data, 24)[0]
    maximum_width = struct.unpack_from("<H", data, 18)[0]
    if (version, header, size, flags & 1, classes) != (1, 64, 657664, 1, 3501):
        raise ValueError("released W8I header contract differs")
    if maximum_width != 1024:
        raise ValueError("released FL10 resized-width contract differs")
    if zlib.crc32(data[64:]) != struct.unpack_from("<I", data, 40)[0]:
        raise ValueError("released FL10 CRC differs")


def code_identity():
    identity = {p: sha256(ROOT / p) for p in CODE}
    if identity["runtime/vocr_lines.c"] != FINDER_SHA:
        raise ValueError("candidate finder differs from the prospectively fixed source")
    return identity


def environment_identity():
    """Record and enforce actual installed versions from the committed lock."""
    lock = (ROOT / "requirements-ci.lock").read_text()
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "torch")}
    for name, version in packages.items():
        expected = re.search(r"^" + re.escape(name.lower()) + r"==([^\s]+)", lock, re.MULTILINE)
        if expected is None or version != expected[1]:
            raise ValueError("installed dependency differs from the committed pinned version: " + name)
    return {"system": platform.system(), "machine": platform.machine(), "Python": sys.version,
            "python_executable_sha256": sha256(sys.executable), "packages": packages}


def build_finder(directory, compiler):
    executable = Path(shutil.which(compiler) or compiler).resolve()
    binary = Path(directory) / "fixed-vocr-lines"
    argv = [str(executable), *FLAGS, str(ROOT / "runtime/vocr_lines.c"),
            str(ROOT / "runtime/vocr_lines_test.c"), "-o", str(binary)]
    subprocess.run(argv, check=True, env={"PATH": os.defpath, "LC_ALL": "C"})
    return {"binary": str(binary), "sha256": sha256(binary), "argv": argv,
            "compiler": str(executable), "compiler_sha256": sha256(executable),
            "version": subprocess.run([str(executable), "--version"], check=True,
                                      capture_output=True, text=True).stdout}


def prepare(manifest, annotations, model, library, directory, compiler="cc"):
    """Freeze one cohort/model/source family; no finder calls or model forwards."""
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("fresh audit directory must be new; never replace a preparation")
    document, sources, readiness = checked_inputs(manifest, annotations)
    identity = code_identity()
    for source in sources:
        image_rgb(source)
    if sha256(model) != MODEL_SHA or sha256(library) != LIBRARY_SHA:
        raise ValueError("model or explicitly pinned scalar library differs")
    validate_blob_bytes(Path(model).read_bytes())
    directory.mkdir(parents=True)
    finder = build_finder(directory, compiler)
    freeze(directory / "protocol.json", PROTOCOL)
    prepared = {
        "schema": 1, "kind": "fresh-native-textbox-prepared", "directory": str(directory),
        "protocol_sha256": sha256(directory / "protocol.json"), "code_sha256": identity,
        "manifest": str(Path(manifest).resolve()), "manifest_sha256": sha256(manifest),
        "annotations": str(Path(annotations).resolve()), "annotations_sha256": sha256(annotations),
        "model": str(Path(model).resolve()), "model_sha256": MODEL_SHA,
        "library": str(Path(library).resolve()), "library_sha256": LIBRARY_SHA,
        "finder": finder, "readiness": readiness,
        "source_fingerprint": digest([{k: v for k, v in s.items() if k != "_lines"} for s in sources]),
        "reference_fingerprint": digest([s["_reference_sha256"] for s in sources]),
        "exposure_evidence_sha256": document["exclusions"]["evidence_sha256"],
        "environment": environment_identity(),
        "finder_calls": 0, "model_forwards": 0,
    }
    freeze(directory / "prepared.json", prepared)
    return prepared


def verify(prepared_path, images=False):
    prepared = read(prepared_path)
    if prepared.get("schema") != 1 or prepared.get("kind") != "fresh-native-textbox-prepared":
        raise ValueError("invalid preparation")
    # Revalidate roles before image/model/library file hashes or native construction.
    document, sources, readiness = checked_inputs(prepared["manifest"], prepared["annotations"])
    directory = Path(prepared["directory"])
    if (Path(prepared_path).resolve() != directory / "prepared.json"
            or read(directory / "protocol.json") != PROTOCOL
            or sha256(directory / "protocol.json") != prepared["protocol_sha256"]
            or code_identity() != prepared["code_sha256"]
            or sha256(prepared["manifest"]) != prepared["manifest_sha256"]
            or sha256(prepared["annotations"]) != prepared["annotations_sha256"]
            or readiness != prepared["readiness"]
            or digest([{k: v for k, v in s.items() if k != "_lines"} for s in sources])
            != prepared["source_fingerprint"]
            or digest([s["_reference_sha256"] for s in sources]) != prepared["reference_fingerprint"]
            or document["exclusions"]["evidence_sha256"] != prepared["exposure_evidence_sha256"]):
        raise ValueError("frozen protocol/code/cohort/reference changed")
    if (prepared["model_sha256"] != MODEL_SHA or sha256(prepared["model"]) != MODEL_SHA
            or prepared["library_sha256"] != LIBRARY_SHA or sha256(prepared["library"]) != LIBRARY_SHA
            or sha256(prepared["finder"]["binary"]) != prepared["finder"]["sha256"]
            or sha256(prepared["finder"]["compiler"]) != prepared["finder"]["compiler_sha256"]
            or environment_identity() != prepared["environment"]):
        raise ValueError("frozen model/library/finder/compiler/interpreter changed")
    if images:
        for source in sources:
            image_rgb(source)
    return prepared, sources


def expected_authorization(prepared_path):
    prepared = read(prepared_path)
    return {"schema": 1, "authorization": "root-reviewed-one-shot-fresh-native-test",
            "prepared_sha256": sha256(prepared_path), "protocol_sha256": prepared["protocol_sha256"],
            "source_fingerprint": prepared["source_fingerprint"], "model_sha256": MODEL_SHA,
            "candidate_source_sha256": FINDER_SHA}


class NativeReader:
    def __init__(self, prepared):
        sys.path.insert(0, str(ROOT / "runtime"))
        from pyvocr import Vocr, load_lib

        if sha256(prepared["model"]) != MODEL_SHA or sha256(prepared["library"]) != LIBRARY_SHA:
            raise ValueError("frozen model/scalar library changed before construction")
        self.reader = Vocr(prepared["model"], mode="w8i", max_width=1024,
                           lib=load_lib(prepared["library"]), check_crc=True)
        if (self.reader.blob_bytes, self.reader.arena_bytes, self.reader.n_classes) != (657664, 124832, 3501):
            raise ValueError("FL10 native memory/class contract differs")

    def text(self, rgba, dark):
        return self.reader.text(rgba, invert=dark)

    def close(self):
        self.reader = None


def transcript_score(lines, predictions):
    # Reuse the frozen scoring behavior without changing the strict DEV validator.
    sys.path.insert(0, str(ROOT / "train"))
    from general_vn_textbox_guard import transcript_score as score

    if not isinstance(predictions, list) or any(not isinstance(p, str) for p in predictions):
        raise ValueError("every returned crop needs an explicit string, including blank")
    return score(lines, predictions)


def balanced_body_hira(groups):
    rates = []
    for console in DIMENSIONS:
        scores = [g["body"]["hira"] for g in groups.values() if g["platform"] == console]
        if not scores or any(s["chars"] <= 0 for s in scores):
            raise ValueError("positive body-hiragana references required on both consoles")
        rates.append(sum((Fraction(s["errors"], s["chars"]) for s in scores), Fraction()) / len(scores))
    return sum(rates, Fraction()) / 2


def qualify(results):
    """Fixed family gate. No adjustable minima, tolerances, weights or fallback."""
    reasons, gains, aggregates = [], {}, {}
    if set(results) != set(MODES) or any(set(r) != set(VARIANTS) for r in results.values()):
        return {"passed": False, "reasons": ["fixed mode/variant identities differ"]}
    for variant in VARIANTS:
        base, prior, candidate = (results[side][variant] for side in MODES)
        if set(base) != set(prior) or set(base) != set(candidate):
            reasons.append(variant + ": title identities differ")
            continue
        valid = True
        for title, ref in base.items():
            if (ref.get("platform") not in DIMENSIONS or not isinstance(ref.get("vn"), str)
                    or title != ref["platform"] + ":" + ref["vn"]
                    or type(ref.get("frames")) is not int or ref["frames"] < MINIMUMS["frames_per_vn"]):
                reasons.append(variant + "/" + title + ": invalid native reference metadata")
                valid = False
            for side in (prior[title], candidate[title]):
                if any(side.get(k) != ref.get(k) for k in ("platform", "vn", "frames", "reference_sha256")):
                    reasons.append(variant + "/" + title + ": reference identity changed")
                    valid = False
            for side in (ref, prior[title], candidate[title]):
                if any(type(side.get(k)) is not int or side[k] < 0
                       for k in ("detections", "blank_outputs", "zero_detection_frames")):
                    reasons.append(variant + "/" + title + ": invalid detection accounting")
                    valid = False
            for scope in ("all", "body"):
                for script in ("full", "hira"):
                    counts = [s[scope][script] for s in (ref, prior[title], candidate[title])]
                    if (any(type(c.get(k)) is not int or c[k] < 0 for c in counts for k in ("chars", "errors"))
                            or any(c["chars"] != counts[0]["chars"] for c in counts[1:])):
                        reasons.append(variant + "/" + title + ": invalid/mismatching counters")
                        valid = False
                    elif counts[2]["errors"] > counts[0]["errors"]:
                        reasons.append(f"{variant}/{title}/{scope}/{script}: regression versus mode0")
        if not valid:
            continue
        gains[variant] = {}
        for console in DIMENSIONS:
            titles = [t for t, g in candidate.items() if g["platform"] == console]
            improved = sorted(candidate[t]["vn"] for t in titles
                              if candidate[t]["body"]["hira"]["errors"] < base[t]["body"]["hira"]["errors"])
            gains[variant][console] = improved
            if len(titles) < 3 or len(set(improved)) < 2:
                reasons.append(variant + "/" + console + ": insufficient fresh titles/body gains")
        if any(g["body"]["hira"]["chars"] <= 0 for g in candidate.values()):
            reasons.append(variant + ": empty body-hiragana stratum")
            continue
        if not all(any(g["platform"] == console for g in candidate.values()) for console in DIMENSIONS):
            reasons.append(variant + ": missing console stratum")
            continue
        if balanced_body_hira(candidate) >= balanced_body_hira(base):
            reasons.append(variant + ": no strict console-balanced body-hiragana gain")
        aggregates[variant] = {}
        for scope in ("all", "body"):
            for script in ("full", "hira"):
                sums = {side: {k: sum(g[scope][script][k] for g in groups.values()) for k in ("chars", "errors")}
                        for side, groups in (("pure_C3", prior), ("C4", candidate))}
                aggregates[variant][scope + "/" + script] = sums
                if sums["C4"]["errors"] > sums["pure_C3"]["errors"]:
                    reasons.append(f"{variant}/{scope}/{script}: pooled regression versus mode2")
        old = aggregates[variant]["all/full"]["pure_C3"]["errors"]
        new = aggregates[variant]["all/full"]["C4"]["errors"]
        if old <= 0 or 100 * new > 90 * old:
            reasons.append(variant + ": less than10 percent pooled full-error reduction versus mode2")
    return {"passed": not reasons, "reasons": reasons, "body_gain_titles": gains,
            "pooled": aggregates, "material_fraction": "9/10", "weights": PROTOCOL["weights"]}


def score_family(prepared, sources, reader_factory=NativeReader, finder_factory=None, record=None):
    """Run all fixed modes once; injection supports procedural tests, never CLI options."""
    import numpy as np
    from linefind.cfinder import Finder

    finders = {side: (finder_factory(mode) if finder_factory else
                     Finder({0: "grad", 2: "textbox", 3: "textbox_recovery"}[mode],
                            binary=prepared["finder"]["binary"])) for side, mode in MODES.items()}
    results = {side: {v: {} for v in VARIANTS} for side in MODES}
    forwards = Counter()
    try:
        reader = reader_factory(prepared)
    except Exception:
        if record:
            record({"status": "failed", "failure_stage": "reader-construction", "reader_error": True})
        raise
    try:
        for source in sources:
            rgb = image_rgb(source)
            key = source["platform"] + ":" + source["vn"]
            for variant, value in source["variants"].items():
                x, y, w, h = value["roi"]
                rgba = np.concatenate([rgb[y:y + h, x:x + w], np.full((h, w, 1), 255, np.uint8)], axis=2)
                for side, finder in finders.items():
                    try:
                        boxes = finder(rgba)
                    except Exception:
                        if record:
                            record({"source_id": source["id"], "mode": MODES[side], "variant": variant,
                                    "status": "failed", "failure_stage": "finder", "finder_error": True})
                        raise
                    if boxes is None or finder.last_truncated or len(boxes) != len(finder.last_dark):
                        if record:
                            record({"source_id": source["id"], "mode": MODES[side], "variant": variant,
                                    "status": "failed", "failure_stage": "finder",
                                    "finder_error": boxes is None or len(boxes) != len(finder.last_dark),
                                    "truncated": bool(finder.last_truncated),
                                    "polarity_mismatch": boxes is not None and len(boxes) != len(finder.last_dark)})
                        raise RuntimeError("finder error/truncation/polarity mismatch; no blank coercion")
                    predictions, detections = [], []
                    for box, dark in zip(boxes, finder.last_dark):
                        try:
                            a, b, c, d = checked_box(list(box), [w, h])
                            if dark not in (0, 1):
                                raise RuntimeError("invalid C polarity")
                        except Exception:
                            if record:
                                record({"source_id": source["id"], "mode": MODES[side], "variant": variant,
                                        "status": "failed", "failure_stage": "finder", "finder_error": True,
                                        "completed_crops": len(predictions)})
                            raise
                        crop = np.ascontiguousarray(rgba[b:b + d, a:a + c])
                        try:
                            prediction = reader.text(crop, bool(dark))
                        except Exception:
                            if record:
                                record({"source_id": source["id"], "mode": MODES[side], "variant": variant,
                                        "status": "failed", "failure_stage": "reader", "reader_error": True,
                                        "completed_crops": len(predictions), "detections": detections,
                                        "predictions": predictions})
                            raise
                        if not isinstance(prediction, str):
                            if record:
                                record({"source_id": source["id"], "mode": MODES[side], "variant": variant,
                                        "status": "failed", "failure_stage": "reader", "reader_error": True,
                                        "completed_crops": len(predictions)})
                            raise RuntimeError("reader returned missing/non-string output")
                        forwards[side] += 1
                        predictions.append(prediction)
                        detections.append({"box": list(box), "dark": bool(dark)})
                    score = transcript_score(source["_lines"], predictions)
                    group = results[side][variant].setdefault(key, {
                        "platform": source["platform"], "vn": source["vn"], "frames": 0,
                        "detections": 0, "blank_outputs": 0, "zero_detection_frames": 0,
                        "reference_sha256": digest([s["_reference_sha256"] for s in sources
                                                     if s["platform"] + ":" + s["vn"] == key]),
                        "all": {s: {"chars": 0, "errors": 0} for s in ("full", "hira")},
                        "body": {s: {"chars": 0, "errors": 0} for s in ("full", "hira")},
                    })
                    group["frames"] += 1
                    group["detections"] += len(boxes)
                    group["blank_outputs"] += sum(not norm(p) for p in predictions)
                    group["zero_detection_frames"] += not boxes
                    for part in ("all", "body"):
                        for script in ("full", "hira"):
                            for counter in ("chars", "errors"):
                                group[part][script][counter] += score[part][script][counter]
                    if record:
                        record({"source_id": source["id"], "mode": MODES[side], "variant": variant, "status": "ok",
                                "detections": detections, "predictions": predictions, "score": score})
        return results, dict(forwards)
    finally:
        reader.close()


def compare(prepared_path, authorization_path):
    directory = Path(prepared_path).resolve().parent
    approved_sha = sha256(prepared_path)
    if read(authorization_path) != expected_authorization(prepared_path):
        raise ValueError("root-reviewed exact preparation authorization required before execution")
    with (directory / "execution.writer.lock").open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "attempt.json").exists():
            raise FileExistsError("fresh comparison was already attempted; no retries")
        prepared, sources = verify(prepared_path, images=False)
        if sha256(prepared_path) != approved_sha:
            raise ValueError("authorized preparation changed before the one-shot attempt")
        freeze(directory / "attempt.json", {"schema": 1, "prepared_sha256": approved_sha,
                                           "authorization_sha256": sha256(authorization_path)})
        try:
            verify(prepared_path, images=True)
            if (sha256(prepared_path) != approved_sha
                    or read(authorization_path) != expected_authorization(prepared_path)):
                raise ValueError("authorized preparation changed before scoring")
            with (directory / "private-predictions.jsonl").open("x", encoding="utf-8") as stream:
                def record(row):
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()

                results, forwards = score_family(prepared, sources, record=record)
            verify(prepared_path, images=True)
            if (sha256(prepared_path) != approved_sha
                    or read(authorization_path) != expected_authorization(prepared_path)):
                raise ValueError("authorized preparation changed before completion")
            report = {"schema": 1, "kind": "fresh-native-textbox-result", "status": "complete",
                      "scope": "fresh native screenshot test; original training history unknown; no hardware timing",
                      "protocol": PROTOCOL, "prepared_sha256": sha256(prepared_path),
                      "manifest_sha256": prepared["manifest_sha256"],
                      "annotations_sha256": prepared["annotations_sha256"],
                      "source_fingerprint": prepared["source_fingerprint"],
                      "reference_fingerprint": prepared["reference_fingerprint"],
                      "exposure_evidence_sha256": prepared["exposure_evidence_sha256"],
                      "source_sha256": prepared["code_sha256"], "model_sha256": MODEL_SHA,
                      "library_sha256": LIBRARY_SHA, "finder_sha256": prepared["finder"]["sha256"],
                      "readiness": prepared["readiness"], "results": results, "decision": qualify(results),
                      "successful_model_forwards": forwards, "finder_errors": 0, "truncations": 0,
                      "reader_errors": 0, "raw_predictions_sha256": sha256(directory / "private-predictions.jsonl"),
                      "source_manifest": [{k: s[k] for k in ("id", "url", "sha256", "pixel_sha256", "platform",
                                                           "vn", "release", "session", "dims", "reviewer_kind")}
                                          for s in sources], "model_selection": False, "runtime_promotion": False}
            freeze(directory / "result.json", report)
            return report
        except Exception as error:
            predictions = directory / "private-predictions.jsonl"
            failure_rows = [json.loads(line) for line in predictions.read_text().splitlines()
                            if line.strip()] if predictions.exists() else []
            failure_doc = {
                "schema": 1, "status": "failed", "error_type": type(error).__name__,
                "prepared_sha256": approved_sha, "retry_allowed": False,
                "partial_predictions_retained": predictions.exists(),
                "completed_cases": sum(r.get("status") == "ok" for r in failure_rows),
                "finder_errors": sum(r.get("finder_error", False) for r in failure_rows),
                "truncations": sum(r.get("truncated", False) for r in failure_rows),
                "reader_errors": sum(r.get("reader_error", False) for r in failure_rows),
            }
            freeze(directory / "failure.json", failure_doc)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for arg in ("manifest", "annotations", "model", "library", "directory"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--compiler", default="cc")
    v = sub.add_parser("verify")
    v.add_argument("--prepared", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--prepared", type=Path, required=True)
    c.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.annotations, args.model, args.library, args.directory, args.compiler)
    elif args.command == "verify":
        verify(args.prepared, images=True)
    else:
        compare(args.prepared, args.authorization)


if __name__ == "__main__":
    main()

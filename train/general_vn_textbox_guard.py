#!/usr/bin/env python3
"""Prospective DEV textbox guard for one already selected model; never rank models."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

import general_vn_select as selector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "runtime"))
from group_metrics import body  # noqa: E402
from linefind.cfinder import Finder  # noqa: E402
from metrics import is_hiragana, norm  # noqa: E402
from pyvocr import Vocr, load_lib  # noqa: E402

BASELINE_SHA = "500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84"
MARGINS = {"base": 0, "expanded-4": 4, "expanded-8": 8}
PROTOCOL = {
    "schema": 1,
    "purpose": "development-only end-to-end guard after independent line-crop selection",
    "model_comparison": "one frozen selection.chosen export versus the exact released baseline; no ranking",
    "roi_variants": MARGINS,
    "roi_rule": "symmetric native-pixel expansion, clamped to native image; no labelled ink clipped",
    "input": "RGBA8 native ROI, C GRAD finder; each returned RGBA crop to C W8I with its C dark flag",
    "reading_order": ("C finder output order; GT explicit ordered line IDs; "
                      "normalize each line/detection before joining"),
    "alignment": ("guard-local already-normalized Levenshtein; diagonal then deletion then insertion ties; "
                  "no renormalizing"),
    "body_attribution": "reference body substitutions/deletions; ALL unmatched insertions charged to body",
    "hiragana": "reference hiragana substitutions/deletions and hiragana insertions; no kana folding",
    "guard": "every variant and VN/platform all/body full+hira nonregression; no matching/filtering/masking",
    "gain": ("every variant strictly improves console-balanced all AND body hira; "
             "each variant has at least two distinct VN gains"),
    "platform_weights": {"psp": "1/2", "vita": "1/2"},
    "vn_weights": "equal within each console; exact fractions",
    "max_width": 1024,
    "blob_bytes": selector.MODEL_BYTES,
    "arena_bytes": selector.ARENA_BYTES,
    "runtime_promotion": False,
}
CODE = sorted(set(selector.CODE + [
    "train/general_vn_textbox_guard.py", "train/generic_fl10_train.py",
    "eval/linefind/cfinder.py", "runtime/vocr_lines.c", "runtime/vocr_lines.h",
    "runtime/vocr_lines_test.c",
]))
FINDER_SOURCES = ("runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_test.c")
FINDER_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-Wall", "-Wextra", "-Werror")
sha256 = selector.sha256


def atomic_freeze(path, value):
    path = Path(path).resolve()
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
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checked_hash(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("require lowercase SHA256 pins")
    return value


def checked_box(box, dims):
    if (not isinstance(dims, list) or len(dims) != 2
            or any(type(n) is not int or n <= 0 for n in dims)):
        raise ValueError("native dimensions require two positive integers")
    if not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box):
        raise ValueError("ROI requires four native integers")
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > dims[0] or y + h > dims[1]:
        raise ValueError("ROI exceeds native source bounds")
    return box


def roi_variants(base, dims):
    x, y, w, h = checked_box(base, dims)
    result = {}
    for name, margin in MARGINS.items():
        left, top = max(0, x - margin), max(0, y - margin)
        right, bottom = min(dims[0], x + w + margin), min(dims[1], y + h + margin)
        result[name] = [left, top, right - left, bottom - top]
    return result


def pixel_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()).hexdigest()


def read_document(path):
    doc = json.loads(Path(path).read_text())
    return checked_document(doc)


def checked_document(doc):
    sources = doc.get("sources") if isinstance(doc, dict) else None
    if (not isinstance(doc, dict) or type(doc.get("schema")) is not int or doc.get("schema") != 1
            or doc.get("split") != "dev" or not isinstance(sources, list)
            or not sources or any(not isinstance(s, dict) or s.get("split") != "dev" for s in sources)):
        raise ValueError("textbox manifest requires development-only roles before source bytes or pixels")
    return doc


def checked_sources(path, rows, doc=None):
    """Check the entire metadata cohort before any image decode or model forward."""
    path = Path(path).resolve()
    doc = read_document(path) if doc is None else doc
    checked_document(doc)
    references = {r["id"]: r for r in rows}
    seen, seen_sources, sources = set(), set(), []
    for original in doc["sources"]:
        source = dict(original)
        if (not isinstance(source.get("id"), str) or not source["id"] or source["id"] in seen_sources
                or source.get("hand_checked") is not True or source.get("complete_eligible_text") is not True
                or source.get("uncertain") is not False or not isinstance(source.get("scope_exclusions"), list)
                or any(not isinstance(x, str) for x in source["scope_exclusions"])
                or not isinstance(source.get("session"), str) or not source["session"]
                or not isinstance(source.get("release"), str) or not source["release"]):
            raise ValueError("every source needs reviewed complete horizontal-text scope and explicit exclusions")
        seen_sources.add(source["id"])
        ids = source.get("line_ids")
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids) or any(i in seen or i not in references for i in ids)):
            raise ValueError("ROI reading order must contain unique bound development line IDs")
        lines = [references[i] for i in ids]
        if any(r.get("uncertain") or not norm(r["text"])
               or (r.get("content_kind") == "body" and "name" in r["styles"])
               or r.get("content_kind") not in ("body", "name") for r in lines):
            raise ValueError("textbox references require certain explicit body/name kinds")
        seen.update(ids)
        first = lines[0]
        for field in ("platform", "vn", "session", "release", "dims", "sha256"):
            if source.get(field) != first.get(field) or any(r.get(field) != first.get(field) for r in lines):
                raise ValueError("ROI source/VN/platform/release provenance differs from bound lines")
        image = (path.parent / source["image"]).resolve()
        if any(Path(r["_image"]).resolve() != image for r in lines):
            raise ValueError("ROI image differs from bound line source")
        checked_hash(source["sha256"])
        checked_hash(source.get("pixel_sha256"))
        variants = source.get("variants")
        expected = roi_variants(source.get("base_roi"), source["dims"])
        if not isinstance(variants, dict) or set(variants) != set(MARGINS):
            raise ValueError("all fixed ROI variants are mandatory")
        for name, geometry in expected.items():
            variant = variants[name]
            if not isinstance(variant, dict) or set(variant) != {"roi", "pixel_sha256"} or variant["roi"] != geometry:
                raise ValueError("ROI variant differs from fixed symmetric/clamped rule")
            checked_hash(variant["pixel_sha256"])
            x, y, w, h = geometry
            for r in lines:
                a, b, c, d = r["box"]
                if a < x or b < y or a + c > x + w or b + d > y + h:
                    raise ValueError("ROI clips labelled ink")
        source.update(_image=str(image), _lines=lines)
        sources.append(source)
    if seen != set(references):
        raise ValueError("textbox cohort must cover every ready line-crop reference exactly once")
    # The full cohort and roles are valid before any source byte access.
    for source in sources:
        if sha256(source["_image"]) != source["sha256"]:
            raise ValueError("textbox source image SHA mismatch")
    return sources


def materialize_plan(line_lock, plan_path, out):
    """Bind reviewed metadata ROIs to decoded pixels; no detector/model execution."""
    if Path(out).exists():
        raise FileExistsError("ROI manifest is immutable")
    plan_sha = sha256(plan_path)
    plan = read_document(plan_path)
    _, rows = selector.verify_lock(line_lock)
    draft = copy.deepcopy(plan)
    for source in draft["sources"]:
        source.setdefault("pixel_sha256", "0" * 64)
        source["variants"] = {name: {"roi": box, "pixel_sha256": "0" * 64}
                              for name, box in roi_variants(source.get("base_roi"), source.get("dims")).items()}
    sources = checked_sources(plan_path, rows, draft)  # all roles/geometry first
    for entry, source in zip(draft["sources"], sources):
        with Image.open(source["_image"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        if [rgb.shape[1], rgb.shape[0]] != source["dims"]:
            raise ValueError("ROI plan source is not native dimensions")
        actual_sha = pixel_sha(rgb)
        if source["pixel_sha256"] != "0" * 64 and source["pixel_sha256"] != actual_sha:
            raise ValueError("ROI plan source decoded RGB hash differs")
        entry["image"] = source["_image"]  # output can live in another private directory
        entry["pixel_sha256"] = actual_sha
        for variant in entry["variants"].values():
            x, y, w, h = variant["roi"]
            variant["pixel_sha256"] = pixel_sha(rgb[y:y + h, x:x + w])
        if sha256(source["_image"]) != source["sha256"]:
            raise ValueError("ROI source changed while binding pixels")
    if sha256(plan_path) != plan_sha:
        raise ValueError("ROI metadata plan changed while binding pixels")
    draft.update(plan_path=str(Path(plan_path).resolve()), plan_sha256=plan_sha,
                 detector_evaluated=False, model_forwards=0)
    checked_sources(out, rows, draft)
    atomic_freeze(out, draft)
    return draft


def build_finder(directory, compiler="gcc", expected=None):
    """Compile the C host driver unconditionally; never use another worktree/cache."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".gitignore").write_text("*\n")
    compiler_info = selector.compiler_identity(compiler)
    binary, staging = directory / "vocr_lines_guard", directory / "vocr_lines_guard.building"
    sources = {name: sha256(ROOT / name) for name in FINDER_SOURCES}
    argv = [compiler_info["executable"], *FINDER_FLAGS,
            str(ROOT / "runtime/vocr_lines.c"), str(ROOT / "runtime/vocr_lines_test.c"), "-o", str(staging)]
    with (directory / "finder-build.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            staging.unlink(missing_ok=True)
            subprocess.run(argv, check=True, capture_output=True, text=True, cwd=ROOT, env=selector.BUILD_ENV)
            if sources != {name: sha256(ROOT / name) for name in FINDER_SOURCES}:
                raise ValueError("finder sources changed during build")
            if compiler_info != selector.compiler_identity(compiler_info["executable"]):
                raise ValueError("finder compiler changed during build")
            identity = {"directory": str(directory), "binary": str(binary), "binary_sha256": sha256(staging),
                        "compiler": compiler_info, "argv": argv, "cwd": str(ROOT),
                        "build_environment": dict(selector.BUILD_ENV), "source_sha256": sources}
            if expected is not None and expected != identity:
                raise ValueError("finder rebuild differs from frozen identity")
            os.replace(staging, binary)
            return identity
        finally:
            staging.unlink(missing_ok=True)


def verify_finder(identity):
    compiler = selector.compiler_identity(identity["compiler"]["executable"])
    directory = Path(identity["directory"])
    binary, staging = directory / "vocr_lines_guard", directory / "vocr_lines_guard.building"
    current = {"directory": str(directory.resolve()), "binary": str(binary.resolve()),
               "binary_sha256": sha256(binary), "compiler": compiler,
               "argv": [compiler["executable"], *FINDER_FLAGS, str(ROOT / "runtime/vocr_lines.c"),
                        str(ROOT / "runtime/vocr_lines_test.c"), "-o", str(staging)],
               "cwd": str(ROOT), "build_environment": dict(selector.BUILD_ENV),
               "source_sha256": {name: sha256(ROOT / name) for name in FINDER_SOURCES}}
    if identity != current:
        raise ValueError("frozen C finder/compiler/binary changed")


def prepare(line_lock, manifest, out, result_out, compiler="gcc"):
    if Path(out).exists():
        raise FileExistsError("textbox preparation is immutable")
    candidate_out = Path(out).resolve().with_name(Path(out).stem + "-candidate.json")
    destinations = {str(Path(out).resolve()), str(candidate_out), str(Path(result_out).resolve())}
    if (len(destinations) != 3 or destinations & {str(Path(line_lock).resolve()), str(Path(manifest).resolve())}):
        raise ValueError("textbox artifacts need distinct new destinations separate from input locks")
    if Path(result_out).exists() or candidate_out.exists():
        raise FileExistsError("textbox candidate/result destinations must be new before training")
    doc = read_document(manifest)  # wrong ROI role rejected before even line-source bytes
    generic, rows = selector.verify_lock(line_lock)
    if generic["baseline_sha256"] != BASELINE_SHA:
        raise ValueError("textbox guard requires exact released FL10 W8 baseline")
    checked_sources(manifest, rows, doc)
    directory = Path(out).resolve().with_name(Path(out).stem + "_runtime")
    lock = {"schema": 1, "kind": "general-vn-textbox-prepared", "protocol": PROTOCOL,
            "line_lock": str(Path(line_lock).resolve()), "line_lock_sha256": sha256(line_lock),
            "manifest": str(Path(manifest).resolve()), "manifest_sha256": sha256(manifest),
            "baseline": generic["baseline"], "baseline_sha256": BASELINE_SHA,
            "readiness": generic["readiness"], "result_out": str(Path(result_out).resolve()),
            "candidate_out": str(candidate_out),
            "code_sha256": {name: sha256(ROOT / name) for name in CODE},
            "environment": selector.environment_identity(),
            "dependency_sha256": {name: sha256(ROOT / name) for name in selector.DEPENDENCIES},
            "runtime": selector.build_scalar_runtime(directory, compiler),
            "finder": build_finder(directory, compiler), "model_forwards": 0,
            "test_or_google_evaluated": False, "candidate_bound": False}
    # Verify all original inputs again before publication; no decoded pixel arrays or model execution.
    _verify_prepared(lock)
    atomic_freeze(out, lock)
    return lock


def _verify_prepared(lock):
    if (not isinstance(lock, dict) or lock.get("schema") != 1 or lock.get("kind") != "general-vn-textbox-prepared"
            or lock.get("protocol") != PROTOCOL or lock.get("model_forwards") != 0
            or lock.get("test_or_google_evaluated") is not False or lock.get("candidate_bound") is not False):
        raise ValueError("require frozen development-only textbox preparation")
    # Role metadata precedes pixel/forward access even for a corrupted/replaced manifest.
    doc = read_document(lock["manifest"])
    if (sha256(lock["manifest"]) != lock["manifest_sha256"]
            or sha256(lock["line_lock"]) != lock["line_lock_sha256"]
            or sha256(lock["baseline"]) != lock["baseline_sha256"] or lock["baseline_sha256"] != BASELINE_SHA):
        raise ValueError("frozen textbox inputs changed")
    for namespace, names in (("code_sha256", CODE), ("dependency_sha256", selector.DEPENDENCIES)):
        if set(lock[namespace]) != set(names) or any(sha256(ROOT / n) != lock[namespace][n] for n in names):
            raise ValueError("textbox code/dependency fingerprint changed")
    selector.verify_execution(lock)
    verify_finder(lock["finder"])
    generic, rows = selector.verify_lock(lock["line_lock"])
    if (generic["readiness"] != lock["readiness"] or generic["baseline"] != lock["baseline"]
            or generic["baseline_sha256"] != lock["baseline_sha256"]):
        raise ValueError("textbox and line selector readiness/baseline differ")
    sources = checked_sources(lock["manifest"], rows, doc)
    return lock, sources


def verify_prepared(path, line_lock=None):
    lock = json.loads(Path(path).read_text())
    if line_lock is not None and Path(lock.get("line_lock", "")).resolve() != Path(line_lock).resolve():
        raise ValueError("textbox preparation belongs to another line-crop development lock")
    return _verify_prepared(lock)


def bind_selection(lock_path, selection_path, out):
    if Path(out).exists():
        raise FileExistsError("single selected candidate binding is immutable")
    lock, _ = verify_prepared(lock_path)
    if str(Path(out).resolve()) != lock["candidate_out"]:
        raise ValueError("candidate binding differs from its preregistered destination")
    selection = json.loads(Path(selection_path).read_text())
    line_identity = json.loads(Path(lock["line_lock"]).read_text())
    if str(Path(selection_path).resolve()) != line_identity["selection_out"]:
        raise ValueError("selection differs from the line selector's preregistered output")
    if (selection.get("schema") != 1 or selection.get("protocol_lock_sha256") != lock["line_lock_sha256"]
            or selection.get("strict_general_dev_improvement") is not True
            or selection.get("test_or_google_evaluated") is not False
            or selection.get("runtime_promotion") is not False
            or selection.get("protocol") != selector.PROTOCOL
            or selection.get("baseline", {}).get("blob_sha256") != BASELINE_SHA
            or selection.get("environment") != line_identity["environment"]
            or selection.get("runtime") != line_identity["runtime"]):
        raise ValueError("requires completed improved selection from the exact frozen generic DEV protocol")
    chosen = selection.get("chosen", {})
    matching = [r for r in selection.get("candidates", []) if r.get("blob") == chosen.get("blob")
                and r.get("blob_sha256") == chosen.get("blob_sha256") and r.get("eligible") is True
                and r.get("status") == "scored"]
    if len(matching) != 1 or checked_hash(chosen.get("blob_sha256")) == BASELINE_SHA:
        raise ValueError("selection must identify exactly one eligible nonbaseline export")
    candidate = Path(chosen["blob"]).resolve()
    if sha256(candidate) != chosen["blob_sha256"]:
        raise ValueError("selected export hash changed before binding")
    reference = selector.export.read_blob(lock["baseline"])
    selector.validate_blob(reference)
    selector.validate_blob(selector.export.read_blob(candidate), reference)
    binding = {"schema": 1, "kind": "general-vn-textbox-candidate", "prepared": str(Path(lock_path).resolve()),
               "prepared_sha256": sha256(lock_path), "selection": str(Path(selection_path).resolve()),
               "selection_sha256": sha256(selection_path), "candidate": str(candidate),
               "candidate_sha256": chosen["blob_sha256"], "result_out": lock["result_out"],
               "test_or_google_evaluated": False, "models_ranked": False}
    verify_prepared(lock_path)
    if sha256(candidate) != binding["candidate_sha256"]:
        raise ValueError("selected export changed during binding")
    atomic_freeze(out, binding)
    return binding


def verify_binding(path):
    binding = json.loads(Path(path).read_text())
    if (binding.get("schema") != 1 or binding.get("kind") != "general-vn-textbox-candidate"
            or binding.get("test_or_google_evaluated") is not False or binding.get("models_ranked") is not False):
        raise ValueError("require single frozen selected-candidate binding")
    for name, checksum in (("prepared", "prepared_sha256"), ("selection", "selection_sha256"),
                           ("candidate", "candidate_sha256")):
        if sha256(binding[name]) != binding[checksum]:
            raise ValueError("selected candidate or frozen selection/binding changed")
    lock, sources = verify_prepared(binding["prepared"])
    line_identity = json.loads(Path(lock["line_lock"]).read_text())
    if (str(Path(path).resolve()) != lock["candidate_out"]
            or binding["selection"] != line_identity["selection_out"]):
        raise ValueError("candidate binding or selection destination differs from preregistration")
    selection = json.loads(Path(binding["selection"]).read_text())
    if (selection.get("chosen", {}).get("blob_sha256") != binding["candidate_sha256"]
            or str(Path(selection["chosen"]["blob"]).resolve()) != binding["candidate"]
            or selection.get("protocol_lock_sha256") != lock["line_lock_sha256"]
            or binding["result_out"] != lock["result_out"]):
        raise ValueError("single candidate differs from completed DEV selection")
    return binding, lock, sources


def load_cases(sources):
    cases = []
    for source in sources:
        if sha256(source["_image"]) != source["sha256"]:
            raise ValueError("source bytes changed before pixel access")
        with Image.open(source["_image"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        if [rgb.shape[1], rgb.shape[0]] != source["dims"] or pixel_sha(rgb) != source["pixel_sha256"]:
            raise ValueError("native source decoded RGB identity differs")
        for name in MARGINS:
            x, y, w, h = source["variants"][name]["roi"]
            roi = np.ascontiguousarray(rgb[y:y + h, x:x + w])
            if pixel_sha(roi) != source["variants"][name]["pixel_sha256"]:
                raise ValueError("frozen ROI pixel identity differs")
            rgba = np.concatenate([roi, np.full((h, w, 1), 255, np.uint8)], axis=2)
            cases.append({"source": source, "variant": name, "rgba": rgba})
    return cases


def find_crops(cases, finder):
    for case in cases:
        rgba = case["rgba"]
        boxes = finder(rgba)
        if boxes is None or finder.last_truncated or len(boxes) != len(finder.last_dark):
            raise RuntimeError("C finder failed/truncated; guard cannot replace missing results with blank success")
        crops = []
        for box, dark in zip(boxes, finder.last_dark):
            x, y, w, h = checked_box(list(box), [rgba.shape[1], rgba.shape[0]])
            if dark not in (0, 1):
                raise RuntimeError("C finder returned invalid polarity")
            crops.append((np.ascontiguousarray(rgba[y:y + h, x:x + w]), bool(dark)))
        case.update(crops=crops, detections=[{"box": list(b), "dark": bool(d)}
                                            for b, d in zip(boxes, finder.last_dark)])
    return cases


def align_normalized(reference, prediction):
    """Align already-normalized units without merging punctuation at their boundaries."""
    table = [list(range(len(prediction) + 1))]
    for i, reference_char in enumerate(reference, 1):
        row = [i]
        for j, predicted_char in enumerate(prediction, 1):
            row.append(min(table[-1][j] + 1, row[-1] + 1,
                           table[-1][j - 1] + (reference_char != predicted_char)))
        table.append(row)
    pairs, i, j = [], len(reference), len(prediction)
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            pairs.append((reference[i - 1], prediction[j - 1]))
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, prediction[j - 1]))
            j -= 1
    return pairs[::-1]


def transcript_score(lines, predictions):
    """One full alignment; never remove detections using GT/name boxes."""
    # A single string represents one detection; the pipeline supplies one entry per C crop.
    if isinstance(predictions, str):
        predictions = [predictions]
    prediction = "".join(norm(text) for text in predictions)
    text = "".join(norm(line["text"]) for line in lines)
    body_mask = [body(line) for line in lines for _ in norm(line["text"])]
    result = {part: {script: {"chars": 0, "errors": 0} for script in ("full", "hira")}
              for part in ("all", "body")}
    position = 0
    for reference, predicted in align_normalized(text, prediction):
        in_body = True if reference is None else body_mask[position]
        if reference is not None:
            position += 1
        for part in ("all", "body") if in_body else ("all",):
            result[part]["full"]["chars"] += int(reference is not None)
            result[part]["hira"]["chars"] += int(is_hiragana(reference))
            if reference != predicted:
                result[part]["full"]["errors"] += 1
                result[part]["hira"]["errors"] += int(is_hiragana(reference) or
                                                       (reference is None and is_hiragana(predicted)))
    if position != len(body_mask):
        raise ValueError("normalized transcript/body attribution length differs")
    return result


class NativeReader:
    def __init__(self, path, runtime):
        if sha256(runtime["library"]) != runtime["library_sha256"]:
            raise ValueError("frozen scalar library changed before model construction")
        self.reader = Vocr(path, mode="w8i", max_width=1024, lib=load_lib(runtime["library"]))
        if (self.reader.blob_bytes, self.reader.arena_bytes) != (selector.MODEL_BYTES, selector.ARENA_BYTES):
            raise ValueError("FL10 blob/arena memory contract changed")

    def text(self, rgba, dark):
        return self.reader.text(rgba, invert=dark)

    def close(self):
        self.reader = None


def score_cases(blob, cases, runtime, reader_factory=None):
    groups = {name: {} for name in MARGINS}
    references = {name: {} for name in MARGINS}
    reader = (reader_factory or NativeReader)(blob, runtime)
    try:
        for case in cases:
            source = case["source"]
            predictions = [reader.text(crop, dark) for crop, dark in case["crops"]]
            if any(not isinstance(text, str) for text in predictions):
                raise RuntimeError("reader must return explicit text or raise; no missing result coercion")
            score = transcript_score(source["_lines"], predictions)
            key = f"{source['platform']}:{source['vn']}"
            group = groups[case["variant"]].setdefault(key, {
                "platform": source["platform"], "vn": source["vn"], "frames": 0, "detections": 0,
                "all": {s: {"chars": 0, "errors": 0} for s in ("full", "hira")},
                "body": {s: {"chars": 0, "errors": 0} for s in ("full", "hira")},
            })
            group["frames"] += 1
            group["detections"] += len(case["crops"])
            references[case["variant"]].setdefault(key, []).append({
                "source_id": source.get("id"), "source_sha256": source.get("sha256"),
                "lines": [{name: row.get(name) for name in ("id", "text", "content_kind", "styles", "box")}
                          for row in source["_lines"]]})
            for part in ("all", "body"):
                for script in ("full", "hira"):
                    for counter in ("chars", "errors"):
                        group[part][script][counter] += score[part][script][counter]
        for variant, strata in groups.items():
            for key, group in strata.items():
                group["reference_sha256"] = selector.digest(references[variant][key])
        return groups
    finally:
        reader.close()


def balanced_rate(groups, part="all", script="hira"):
    rates = []
    for platform in ("psp", "vita"):
        scores = [g[part][script] for g in groups.values() if g["platform"] == platform]
        if not scores or any(s["chars"] <= 0 for s in scores):
            raise ValueError("textbox macro requires positive reference counts on both consoles")
        rates.append(sum((Fraction(s["errors"], s["chars"]) for s in scores), Fraction()) / len(scores))
    return sum(rates, Fraction()) / 2


def qualify(baseline, candidate):
    reasons, gains, variant_gains = [], set(), {}
    if set(baseline) != set(MARGINS) or set(candidate) != set(MARGINS):
        return {"passed": False, "reasons": ["ROI variants differ"], "gain_vns": []}
    for variant in MARGINS:
        local_gains = set()
        base, got = baseline[variant], candidate[variant]
        if set(base) != set(got):
            reasons.append(variant + ": VN/platform identities differ")
            variant_gains[variant] = []
            continue
        for key, reference in base.items():
            current = got[key]
            if any(current.get(f) != reference.get(f) for f in (
                    "platform", "vn", "frames", "detections", "reference_sha256")):
                reasons.append(f"{variant}/{key}: reference/detection counts differ")
            for part in ("all", "body"):
                for script in ("full", "hira"):
                    a, b = reference[part][script], current[part][script]
                    if (type(b["chars"]) is not int or b["chars"] != a["chars"] or
                            type(b["errors"]) is not int or not 0 <= b["errors"] <= a["errors"]):
                        reasons.append(f"{variant}/{key}/{part}/{script}: regression/reference mismatch")
            if (current["all"]["hira"]["errors"] < reference["all"]["hira"]["errors"]
                    and current["body"]["hira"]["errors"] < reference["body"]["hira"]["errors"]):
                local_gains.add(reference["vn"])
        variant_gains[variant] = sorted(local_gains)
        gains.update(local_gains)
        if len(local_gains) < 2:
            reasons.append(f"{variant}: fewer than two distinct VNs improve all/body hiragana")
        for part in ("all", "body"):
            if balanced_rate(got, part) >= balanced_rate(base, part):
                reasons.append(f"{variant}/{part}: no strict console-balanced hiragana gain")
    return {"passed": not reasons, "reasons": reasons, "gain_vns": sorted(gains),
            "gain_vns_by_variant": variant_gains}


def compare(binding_path, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("completed textbox decision is immutable")
    binding_sha = sha256(binding_path)
    binding, lock, sources = verify_binding(binding_path)
    if str(out) != binding["result_out"]:
        raise ValueError("output differs from preregistered textbox result destination")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name + ".writer.lock")).open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists():
            raise FileExistsError("completed textbox decision is immutable")
        # Rebuild only after original cached bytes/compiler/source identities were verified.
        selector.build_scalar_runtime(lock["runtime"]["directory"], lock["runtime"]["compiler"]["executable"],
                                      expected=lock["runtime"])
        build_finder(lock["finder"]["directory"], lock["finder"]["compiler"]["executable"], expected=lock["finder"])
        verify_binding(binding_path)
        cases = find_crops(load_cases(sources), Finder("grad", binary=lock["finder"]["binary"]))
        verify_binding(binding_path)  # all source/ROI pixels checked before either model sees a crop
        baseline = score_cases(lock["baseline"], cases, lock["runtime"])
        verify_binding(binding_path)
        candidate = score_cases(binding["candidate"], cases, lock["runtime"])
        verify_binding(binding_path)
        if sha256(binding_path) != binding_sha:
            raise ValueError("candidate binding changed during comparison")
        result = {"schema": 1, "kind": "general-vn-textbox-result", "protocol": lock["protocol"],
                  "binding_sha256": binding_sha, "prepared_sha256": binding["prepared_sha256"],
                  "selection_sha256": binding["selection_sha256"], "manifest_sha256": lock["manifest_sha256"],
                  "baseline_sha256": lock["baseline_sha256"], "candidate_sha256": binding["candidate_sha256"],
                  "readiness": lock["readiness"], "runtime": lock["runtime"], "finder": lock["finder"],
                  "environment": lock["environment"], "code_sha256": lock["code_sha256"],
                  "dependency_sha256": lock["dependency_sha256"], "baseline": baseline, "candidate": candidate,
                  "decision": qualify(baseline, candidate), "test_or_google_evaluated": False,
                  "models_ranked": False, "runtime_promotion": False,
                  "console_balanced_rates": {
                      model: {variant: {part: {script: str(balanced_rate(groups, part, script))
                                               for script in ("full", "hira")} for part in ("all", "body")}
                              for variant, groups in scored.items()}
                      for model, scored in (("baseline", baseline), ("candidate", candidate))},
                  "successful_model_forwards": {
                      model: sum(g["detections"] for strata in scored.values() for g in strata.values())
                      for model, scored in (("baseline", baseline), ("candidate", candidate))},
                  "cases": [{"id": c["source"]["id"], "variant": c["variant"],
                             "platform": c["source"]["platform"], "vn": c["source"]["vn"],
                             "session": c["source"]["session"], "release": c["source"]["release"],
                             "source_sha256": c["source"]["sha256"],
                             "roi": c["source"]["variants"][c["variant"]]["roi"],
                             "roi_pixel_sha256": c["source"]["variants"][c["variant"]]["pixel_sha256"],
                             "line_ids": c["source"]["line_ids"], "detections": c["detections"]} for c in cases]}
        atomic_freeze(out, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--line-lock", required=True, type=Path)
    materialize.add_argument("--plan", required=True, type=Path)
    materialize.add_argument("--out", required=True, type=Path)
    prep = sub.add_parser("prepare")
    prep.add_argument("--line-lock", required=True, type=Path)
    prep.add_argument("--roi-manifest", required=True, type=Path)
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--result-out", required=True, type=Path)
    prep.add_argument("--compiler", default="gcc")
    bind = sub.add_parser("bind")
    bind.add_argument("--lock", required=True, type=Path)
    bind.add_argument("--selection", required=True, type=Path)
    bind.add_argument("--out", required=True, type=Path)
    run = sub.add_parser("compare")
    run.add_argument("--candidate-lock", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "materialize":
        result = materialize_plan(args.line_lock, args.plan, args.out)
        print(json.dumps({"sources": len(result["sources"]), "model_forwards": 0}))
    elif args.command == "prepare":
        result = prepare(args.line_lock, args.roi_manifest, args.out, args.result_out, args.compiler)
        print(json.dumps({"readiness": result["readiness"], "model_forwards": 0}))
    elif args.command == "bind":
        result = bind_selection(args.lock, args.selection, args.out)
        print(json.dumps({"candidate_sha256": result["candidate_sha256"], "models_ranked": False}))
    else:
        result = compare(args.candidate_lock, args.out)
        print(json.dumps(result["decision"]))


if __name__ == "__main__":
    main()

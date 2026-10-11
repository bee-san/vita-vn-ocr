#!/usr/bin/env python3
"""Prospective, independent general-VN development selection. No test data loading."""

import argparse
import fcntl
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "eval", ROOT / "synth", ROOT / "train"):
    sys.path.insert(0, str(directory))
from backends import NativeVocr as CachedNativeVocr  # noqa: E402
from group_metrics import MINIMUMS, balanced_rate, group_score, normalize_row, readiness  # noqa: E402
from metrics import score_lines  # noqa: E402
from sets import pad_box  # noqa: E402

import export  # noqa: E402
from common import line_polarity, to_gray  # noqa: E402
from hiragana_select import (  # noqa: E402
    ARENA_BYTES,
    MODEL_BYTES,
    export_checkpoint,
    load_synthetic_development,
    sha256,
    validate_blob,
)

PROTOCOL = {
    "schema": 1,
    "purpose": "prospective general-VN development only; never test or Google selection",
    "group_by": ["platform", "vn"],
    "minimums": MINIMUMS,
    "platform_weights": {"psp": "1/2", "vita": "1/2"},
    "guard": "each VN all/body full and hiragana errors <= its own baseline; each synthetic full/hira <= baseline",
    "rank": [
        "console_balanced_hiragana",
        "console_balanced_full",
        "total_full_errors",
        "synthetic_full_errors",
        "step",
        "checkpoint_path",
    ],
    "replace": "strict console-balanced hiragana reduction and gains in at least two distinct VN IDs",
    "minimum_improving_vns": 2,
    "body_definition": "exclude styles=name or content_kind=name",
    "pads": {"psp": 2, "vita": 4},
    "backend": "scalar C W8I",
    "max_width": 1024,
    "blob_bytes": MODEL_BYTES,
    "arena_bytes": ARENA_BYTES,
    "normalization": "eval.metrics.norm; no kana folding",
    "font_family_metadata": "not inferred",
}
CODE = [
    "train/general_vn_select.py",
    "eval/group_metrics.py",
    "train/hiragana_select.py",
    "train/export.py",
    "train/model.py",
    "eval/backends.py",
    "eval/metrics.py",
    "eval/sets.py",
    "runtime/pyvocr.py",
    "runtime/Makefile",
    "runtime/vocr.c",
    "runtime/vocr.h",
    "synth/common.py",
]
DEPENDENCIES = ("requirements.lock", "requirements-ci.lock")
RUNTIME_SOURCES = ("runtime/vocr.c", "runtime/vocr.h")
BUILD_ENV = {"PATH": os.defpath, "LC_ALL": "C", "TZ": "UTC", "SOURCE_DATE_EPOCH": "0"}
SCALAR_FLAGS = (
    "-std=c99",
    "-O2",
    "-ffp-contract=off",
    "-DVOCR_NO_NEON",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wpedantic",
    "-Wshadow",
    "-Wstrict-prototypes",
    "-Wmissing-prototypes",
    "-Wcast-align",
    "-Wpointer-arith",
    "-fPIC",
    "-shared",
)


def environment_identity():
    executable = Path(sys.executable).resolve()
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "executable": str(executable),
            "sha256": sha256(executable),
        },
        "host": {"system": platform.system(), "machine": platform.machine(), "libc": list(platform.libc_ver())},
        "packages": {"torch": torch.__version__, "numpy": np.__version__, "Pillow": Image.__version__},
    }


def compiler_identity(compiler="gcc"):
    found = shutil.which(str(compiler), path=BUILD_ENV["PATH"])
    if found is None:
        raise ValueError("scalar C compiler not found")
    executable = Path(found).resolve()
    version = subprocess.run(
        [str(executable), "--version"], check=True, capture_output=True, text=True, env=BUILD_ENV
    ).stdout.strip()
    if not version:
        raise ValueError("scalar C compiler did not report its version")
    return {"executable": str(executable), "sha256": sha256(executable), "version": version}


def build_scalar_runtime(directory, compiler="gcc", expected=None):
    """Always invoke the bound compiler; never trust the pyvocr mtime cache."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".gitignore").write_text("*\n")
    library, staging = directory / "libvocr_scalar.so", directory / "libvocr_scalar.building.so"
    compiler_info = compiler_identity(compiler)
    sources = {name: sha256(ROOT / name) for name in RUNTIME_SOURCES}
    argv = [compiler_info["executable"], *SCALAR_FLAGS, "-o", str(staging), str(ROOT / "runtime/vocr.c"), "-lm"]
    with (directory / "build.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            staging.unlink(missing_ok=True)
            subprocess.run(argv, check=True, capture_output=True, text=True, cwd=ROOT, env=BUILD_ENV)
            if sources != {name: sha256(ROOT / name) for name in RUNTIME_SOURCES}:
                raise ValueError("C sources changed during the scalar build")
            if compiler_info != compiler_identity(compiler_info["executable"]):
                raise ValueError("compiler changed during the scalar build")
            identity = {
                "directory": str(directory),
                "library": str(library),
                "library_sha256": sha256(staging),
                "compiler": compiler_info,
                "argv": argv,
                "cwd": str(ROOT),
                "build_environment": dict(BUILD_ENV),
                "source_sha256": sources,
            }
            if expected is not None and identity != expected:
                raise ValueError("rebuilt scalar runtime differs from frozen compiler/library identity")
            os.replace(staging, library)
            return identity
        finally:
            staging.unlink(missing_ok=True)


def verify_execution(lock):
    """Check original artifacts before a rebuild can hide drift or corruption."""
    if lock["environment"] != environment_identity():
        raise ValueError("installed Python/dependency environment changed after preparation")
    runtime = lock["runtime"]
    compiler = compiler_identity(runtime["compiler"]["executable"])
    directory = Path(runtime["directory"])
    library = directory / "libvocr_scalar.so"
    staging = directory / "libvocr_scalar.building.so"
    expected = {
        "directory": str(directory.resolve()),
        "library": str(library.resolve()),
        "library_sha256": sha256(library),
        "compiler": compiler,
        "argv": [compiler["executable"], *SCALAR_FLAGS, "-o", str(staging), str(ROOT / "runtime/vocr.c"), "-lm"],
        "cwd": str(ROOT),
        "build_environment": BUILD_ENV,
        "source_sha256": {name: sha256(ROOT / name) for name in RUNTIME_SOURCES},
    }
    if runtime != expected:
        raise ValueError("frozen scalar compiler/library identity changed after preparation")


class NativeVocr(CachedNativeVocr):
    """Use only the explicitly built scalar library, bypassing the default cache."""

    def __init__(self, path, runtime):
        sys.path.insert(0, str(ROOT / "runtime"))
        from pyvocr import Vocr, load_lib

        if sha256(runtime["library"]) != runtime["library_sha256"]:
            raise ValueError("scalar runtime library changed before reader construction")
        self.runtime = Vocr(path, mode="w8i", max_width=1024, lib=load_lib(runtime["library"]))
        self.name = Path(path).stem + "/scalar-w8i"
        self.info = {"blob_bytes": self.runtime.blob_bytes, "arena_bytes": self.runtime.arena_bytes}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def frozen_file(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("frozen general-VN protocol differs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return sha256(path)


def validate_protocol(protocol):
    if protocol != PROTOCOL:
        raise ValueError("unsupported prospective protocol; declare changes as a separate preregistration/version")


def read_rows(manifests):
    rows, bindings = [], {}
    for filename in manifests:
        path = Path(filename).resolve()
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records or any(row.get("split") != "dev" for row in records):
            raise ValueError("general-VN selector requires development-only manifests before image access")
        if any(row.get("hand_checked") is not True or not row.get("session") for row in records):
            raise ValueError("development references need reviewed labels and sessions")
        bindings[str(path)] = {"sha256": sha256(path), "role": "dev-manifest"}
        for original in records:
            row = normalize_row(original)
            if not row["text"] or row.get("uncertain"):
                if row.get("uncertain"):
                    continue
                raise ValueError("empty development reference")
            dims, box = row.get("dims"), row.get("box")
            if dims is None or not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError("development lines require native dimensions and integer ink boxes")
            x, y, w, h = box
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > dims[0] or y + h > dims[1]:
                raise ValueError("development ink box exceeds native image bounds")
            if not isinstance(row.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"]):
                raise ValueError("development source image requires a pinned SHA256")
            row["_image"] = str((path.parent / row["image"]).resolve())
            rows.append(row)
    # All manifest roles checked before any source-image bytes or decoded pixels.
    readiness(rows)
    image_owners = {}
    for row in rows:
        owner = (row["platform"], row["vn"])
        if row.get("sha256") in image_owners and image_owners[row["sha256"]] != owner:
            raise ValueError("one source image cannot represent different VN/platform groups")
        image_owners[row.get("sha256")] = owner
        image = Path(row["_image"])
        if sha256(image) != row.get("sha256"):
            raise ValueError("development source image SHA mismatch")
        bindings[str(image)] = {"sha256": row["sha256"], "bytes": image.stat().st_size, "role": "dev-image"}
    return rows, bindings


def synthetic_metadata(path):
    """Only role/label metadata is read during preparation, not rendered pixel arrays."""
    with np.load(path, allow_pickle=False) as shard:
        metas = json.loads(str(shard["meta"]))
        labels = json.loads(str(shard["labels"]))
    if (
        not labels
        or len(labels) != len(metas)
        or any(not isinstance(row, dict) or row.get("split") != "dev" for row in metas)
    ):
        raise ValueError("synthetic guard must be development-only")
    cohorts = {row.get("book_split_sha256") for row in metas}
    if len(cohorts) != 1 or any(
        not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) for value in cohorts
    ):
        raise ValueError("synthetic development requires explicit held-out book cohort provenance")
    return {"sha256": sha256(path), "role": "synthetic-dev", "samples": len(labels)}


def prepare(protocol_path, baseline, manifests, synthetic, out, selection_out=None, compiler="gcc"):
    protocol_path, baseline = Path(protocol_path).resolve(), Path(baseline).resolve()
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    existing = json.loads(Path(out).read_text()) if Path(out).exists() else None
    if existing is not None:
        verify_execution(existing)
    rows, inputs = read_rows(manifests)
    for name, filename in synthetic.items():
        if not name or name in ("groups", "real"):
            raise ValueError("synthetic guards need distinct non-reserved names")
        path = Path(filename).resolve()
        inputs[str(path)] = synthetic_metadata(path)
    if not synthetic:
        raise ValueError("at least one book-separated synthetic development guard is required")
    reference = export.read_blob(baseline)
    validate_blob(reference)
    selection_out = (
        Path(selection_out).resolve()
        if selection_out
        else Path(out).resolve().with_name(Path(out).stem + "-selection.json")
    )
    identity = {
        "schema": 1,
        "selection_out": str(selection_out),
        "protocol": protocol,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "baseline": str(baseline),
        "baseline_sha256": sha256(baseline),
        "manifests": [str(Path(p).resolve()) for p in manifests],
        "synthetic": {name: str(Path(p).resolve()) for name, p in synthetic.items()},
        "inputs": inputs,
        "readiness": readiness(rows, protocol["minimums"]),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE},
        "dependency_sha256": {name: sha256(ROOT / name) for name in DEPENDENCIES},
        "environment": environment_identity(),
        "runtime": build_scalar_runtime(
            Path(out).with_name(Path(out).stem + "_runtime"),
            compiler,
            expected=existing["runtime"] if existing else None,
        ),
        "test_or_google_evaluated": False,
        "candidates_evaluated": False,
    }
    verify_execution(identity)
    frozen_file(out, identity)
    return identity


def verify_lock(path):
    lock = json.loads(Path(path).read_text())
    validate_protocol(lock["protocol"])
    if (
        lock.get("schema") != 1
        or lock.get("test_or_google_evaluated") is not False
        or lock.get("candidates_evaluated") is not False
    ):
        raise ValueError("not a development-only protocol lock")
    if sha256(lock["protocol_path"]) != lock["protocol_sha256"]:
        raise ValueError("protocol changed after preparation")
    if sha256(lock["baseline"]) != lock["baseline_sha256"]:
        raise ValueError("baseline changed after preparation")
    for namespace, expected in (("code_sha256", CODE), ("dependency_sha256", DEPENDENCIES)):
        if set(lock[namespace]) != set(expected):
            raise ValueError("code/dependency identity namespace differs from complete frozen protocol")
        for name, checksum in lock[namespace].items():
            if sha256(ROOT / name) != checksum:
                raise ValueError("code/dependencies changed after preparation")
    verify_execution(lock)
    for name, binding in lock["inputs"].items():
        if binding["role"] not in ("dev-manifest", "dev-image", "synthetic-dev") or sha256(name) != binding["sha256"]:
            raise ValueError("development input changed after preparation")
    rows, bindings = read_rows(lock["manifests"])
    if not lock["synthetic"]:
        raise ValueError("frozen protocol has no synthetic development guards")
    for path in lock["synthetic"].values():
        path = str(Path(path).resolve())
        if path in bindings:
            raise ValueError("synthetic/real input roles overlap")
        bindings[path] = synthetic_metadata(path)
    if bindings != lock["inputs"]:
        raise ValueError("manifest/synthetic mappings differ from exact bound inputs")
    actual = readiness(rows, lock["protocol"]["minimums"])
    if actual != lock["readiness"]:
        raise ValueError("readiness metadata differs from actual frozen references")
    if not actual["ready"]:
        raise ValueError("development cohort not ready: " + "; ".join(actual["reasons"]))
    return lock, rows


def qualify(baseline, candidate):
    reasons = []
    if (
        baseline["groups"].keys() != candidate["groups"].keys()
        or baseline["synthetic"].keys() != candidate["synthetic"].keys()
    ):
        return ["development group identities differ"]
    for key, base in baseline["groups"].items():
        got = candidate["groups"][key]
        if any(got.get(field) != base.get(field) for field in ("platform", "vn", "provenance", "reference_sha256")):
            reasons.append(f"{key}: reference provenance differs")
        for part in ("all", "body"):
            if got[part]["lines"] != base[part]["lines"]:
                reasons.append(f"{key}/{part}: line counts differ")
            for script in ("full", "hira"):
                current, reference = got[part][script], base[part][script]
                if current["chars"] != reference["chars"]:
                    reasons.append(f"{key}/{part}/{script}: reference counts differ")
                if (
                    type(current["errors"]) is not int
                    or current["errors"] < 0
                    or current["errors"] > reference["errors"]
                ):
                    reasons.append(f"{key}/{part}/{script}: regression or invalid error count")
    for key, base in baseline["synthetic"].items():
        for script in ("full", "hira"):
            got = candidate["synthetic"][key][script]
            if (
                got["chars"] != base[script]["chars"]
                or type(got["errors"]) is not int
                or not 0 <= got["errors"] <= base[script]["errors"]
            ):
                reasons.append(f"{key}/{script}: synthetic regression/reference mismatch")
    return reasons


def gain_vns(baseline, candidate):
    return sorted(
        {
            base["vn"]
            for key, base in baseline["groups"].items()
            if candidate["groups"][key]["all"]["hira"]["errors"] < base["all"]["hira"]["errors"]
        }
    )


def choose(baseline, candidates):
    eligible = []
    base = baseline["scores"]
    for candidate in candidates:
        if "scores" not in candidate:
            continue
        scores = candidate["scores"]
        reasons = qualify(base, scores)
        candidate["qualified"] = not reasons
        candidate["reasons"] = reasons
        candidate["gain_vns"] = gain_vns(base, scores) if not reasons else []
        candidate["eligible"] = (
            not reasons
            and len(candidate["gain_vns"]) >= 2
            and balanced_rate(scores["groups"]) < balanced_rate(base["groups"])
        )
        if candidate["eligible"]:
            eligible.append(candidate)

    def rank(candidate):
        scores = candidate["scores"]
        return (
            balanced_rate(scores["groups"]),
            balanced_rate(scores["groups"], "full"),
            sum(group["all"]["full"]["errors"] for group in scores["groups"].values()),
            sum(score["full"]["errors"] for score in scores["synthetic"].values()),
            candidate["step"],
            candidate["checkpoint"],
        )

    return (min(eligible, key=rank), True) if eligible else (baseline, False)


def load_pixels(rows, protocol):
    cache, crops = {}, []
    for row in rows:
        path = row["_image"]
        if path not in cache:
            with Image.open(path) as image:
                image.load()
                if list(image.size) != row["dims"]:
                    raise ValueError("native development dimensions differ from frozen manifest")
                cache[path] = np.asarray(image.convert("RGB")).copy()
        rgb = cache[path]
        x0, y0, x1, y1 = pad_box(row["box"], protocol["pads"][row["platform"]], rgb.shape[1], rgb.shape[0])
        gray = to_gray(rgb[y0:y1, x0:x1])
        if gray.size == 0:
            raise ValueError("empty development crop")
        crops.append(np.ascontiguousarray(255 - gray if line_polarity(gray) else gray))
    return crops


def score_blob(blob, rows, crops, synthetic, runtime):
    reader = NativeVocr(blob, runtime)
    try:
        if (reader.info["blob_bytes"], reader.info["arena_bytes"]) != (MODEL_BYTES, ARENA_BYTES):
            raise ValueError("fixed FL10 memory budget changed")
        predictions = reader.read(crops)
        if len(predictions) != len(rows):
            raise ValueError("missing real development prediction")
        groups = group_score([dict(row, pred=prediction) for row, prediction in zip(rows, predictions)])
        guards = {}
        for name, lines in synthetic.items():
            predictions = reader.read([line.crop_rgb for line in lines])
            if len(predictions) != len(lines):
                raise ValueError("missing synthetic development prediction")
            score = score_lines([{"text": line.text, "pred": pred} for line, pred in zip(lines, predictions)])
            guards[name] = {
                "full": {"chars": score["all"]["chars"], "errors": score["all"]["errors"]},
                "hira": {"chars": score["hiragana"]["chars"], "errors": score["hiragana"]["errors"]},
            }
        return {"groups": groups, "synthetic": guards}
    finally:
        reader.close()


def select(lock_path, checkpoints, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("completed general-VN selection already exists")
    lock, rows = verify_lock(lock_path)  # readiness rejected before pixels/model execution
    if str(out) != lock["selection_out"]:
        raise ValueError("output differs from frozen selection destination")
    paths = sorted({str(Path(path).resolve()) for path in checkpoints})
    if not paths:
        raise ValueError("checkpoint inventory is empty")
    lock_sha = sha256(lock_path)
    inventory = {
        "protocol_lock_sha256": lock_sha,
        "selection_out": str(out),
        "checkpoints": {path: sha256(path) for path in paths},
    }
    inventory_path = Path(lock_path).with_name(Path(lock_path).stem + "-candidates.json")
    inventory_sha = frozen_file(inventory_path, inventory)  # binds the full cohort before any scoring
    runtime = build_scalar_runtime(
        lock["runtime"]["directory"], lock["runtime"]["compiler"]["executable"], expected=lock["runtime"]
    )
    verify_execution(lock)
    reference = export.read_blob(lock["baseline"])
    validate_blob(reference)
    crops = load_pixels(rows, lock["protocol"])
    synthetic = {name: load_synthetic_development(path)[0] for name, path in lock["synthetic"].items()}
    baseline = {
        "blob": lock["baseline"],
        "blob_sha256": lock["baseline_sha256"],
        "scores": score_blob(lock["baseline"], rows, crops, synthetic, runtime),
    }
    directory = out.with_name(out.stem + "_exports")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".gitignore").write_text("*\n")
    candidates = []
    for filename in paths:
        verify_execution(lock)  # installed dependencies/compiler/library must still match before each candidate
        record = {"checkpoint": filename, "checkpoint_sha256": sha256(filename)}
        try:
            record.update(export_checkpoint(filename, reference, directory))
            record["scores"] = score_blob(record["blob"], rows, crops, synthetic, runtime)
            record["status"] = "scored"
        except (ValueError, RuntimeError, AssertionError, KeyError) as error:
            record.update(status="rejected", reason=str(error))
        candidates.append(record)
    chosen, improved = choose(baseline, candidates)
    # Any mutation during a run invalidates it; lock is rechecked before publication.
    verify_lock(lock_path)
    if any(sha256(path) != checksum for path, checksum in inventory["checkpoints"].items()):
        raise ValueError("checkpoint cohort changed during selection")
    if sha256(inventory_path) != inventory_sha:
        raise ValueError("checkpoint inventory changed during selection")
    if sha256(lock_path) != lock_sha or sha256(chosen["blob"]) != chosen["blob_sha256"]:
        raise ValueError("chosen blob or protocol lock changed during selection")
    result = {
        "schema": 1,
        "protocol_lock_sha256": lock_sha,
        "candidate_inventory_sha256": inventory_sha,
        "protocol": lock["protocol"],
        "environment": lock["environment"],
        "runtime": runtime,
        "readiness": lock["readiness"],
        "baseline": baseline,
        "candidates": candidates,
        "strict_general_dev_improvement": improved,
        "chosen": {key: chosen[key] for key in ("blob", "blob_sha256")},
        "test_or_google_evaluated": False,
        "runtime_promotion": False,
    }
    if improved:
        result["chosen"].update({key: chosen[key] for key in ("checkpoint", "checkpoint_sha256", "step")})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--protocol", type=Path, required=True)
    prep.add_argument("--baseline", type=Path, required=True)
    prep.add_argument("--dev-manifest", type=Path, action="append", required=True)
    prep.add_argument("--synthetic-dev", action="append", required=True, help="NAME=PATH, repeatable")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--selection-out", type=Path)
    prep.add_argument("--compiler", default="gcc")
    sel = sub.add_parser("select")
    sel.add_argument("--lock", type=Path, required=True)
    sel.add_argument("--checkpoints", action="append", required=True)
    sel.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "prepare":
        synthetic = {}
        for value in args.synthetic_dev:
            name, path = value.split("=", 1)
            if name in synthetic:
                raise ValueError("duplicate synthetic guard name")
            synthetic[name] = path
        result = prepare(
            args.protocol, args.baseline, args.dev_manifest, synthetic, args.out, args.selection_out, args.compiler
        )
        print(json.dumps(result["readiness"], indent=2))
    else:
        paths = []
        for pattern in args.checkpoints:
            found = glob.glob(pattern)
            if not found:
                raise ValueError("checkpoint glob is empty")
            paths.extend(found)
        result = select(args.lock, paths, args.out)
        print(
            json.dumps(
                {"chosen": result["chosen"], "strict_general_dev_improvement": result["strict_general_dev_improvement"]}
            )
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check pinned MNN reader/path equivalence and recovery on private VN tensors.

The committed summary contains hashes and aggregate comparisons only. This is
host loading/ownership evidence, not console timing, physical RAM or a CER run.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PORT = Path(__file__).resolve().parent
ROOT = PORT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnn-source", type=Path, required=True)
    parser.add_argument("--mnn-library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--matrix", type=Path,
                        help="Pinned private 451-crop ground_truth.json; two complete path/reader passes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path,
                        help="Private cases.tsv supplying original/effective widths; no dialogue is copied")
    parser.add_argument("--sanitize", action="store_true",
                        help="Instrument the adapter/probe with ASan/UBSan; supplied MNN archive is unchanged")
    args = parser.parse_args()
    if bool(args.input) == bool(args.matrix):
        parser.error("Provide --input or --matrix")
    matrix = None
    if args.matrix:
        expected = json.loads((PORT / "results-arm.json").read_text())["provenance"]
        if digest(args.matrix) != expected["private_fixture_metadata_sha256"]:
            parser.error("Matrix metadata does not match the pinned hand-checked tensor bundle")
        matrix = json.loads(args.matrix.read_text())
        args.input = [args.matrix.parent / f"{row['id']}.f32" for row in matrix]
    pins = json.loads((PORT / "dependencies.json").read_text())
    if args.model.stat().st_size != pins["converted"]["bytes"] or digest(args.model) != pins["converted"]["sha256"]:
        parser.error("Model does not match the pinned recognizer")
    for source in args.input:
        if source.stat().st_size != 3 * 32 * 960 * 4:
            parser.error("Every input must be a BGR NCHW float32 tensor")
    dimensions = {}
    if matrix:
        for row, source in zip(matrix, args.input, strict=True):
            if digest(source) != row["input_sha256"]:
                parser.error("Matrix tensor hash mismatch")
            dimensions[source.name] = (row["width"], row["effective_w"])
    if args.cases:
        for line in args.cases.read_text().splitlines():
            name, crop, effective, *_ = line.split("\t")
            dimensions[name] = (int(crop), int(effective))
        if any(p.name not in dimensions for p in args.input):
            parser.error("Each input must have private case dimensions")
    args.out.mkdir(parents=True, exist_ok=True)
    args.out = args.out.resolve()
    cc, cxx = os.environ.get("CC", "cc"), os.environ.get("CXX", "c++")
    flags = ["-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", f"-I{PORT}"]
    if args.sanitize:
        flags += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    objects = []
    for name in ("arena", "decode", "model_reader"):
        obj = args.out / f"{name}.o"
        subprocess.run([cc, "-std=c99", *flags, "-c", str(PORT / f"{name}.c"), "-o", str(obj)], check=True)
        objects.append(str(obj))
    probe = args.out / "reader-probe"
    command = [cxx, "-std=c++17", *flags, f"-I{args.mnn_source / 'include'}",
               str(PORT / "reader_probe.cpp"), str(PORT / "runtime.cpp"), *objects,
               str(args.mnn_library), "-pthread", "-lm", "-ldl",
               "-Wl,--wrap=MNNMemoryAllocAlign", "-Wl,--wrap=MNNMemoryCallocAlign",
               "-Wl,--wrap=MNNMemoryFreeAlign",
               "-Wl,--wrap=_ZN3MNN11Interpreter14createFromFileEPKc", "-o", str(probe)]
    subprocess.run(command, check=True)
    run = [str(probe), *(["--matrix"] if matrix else []), str(args.model.resolve())]
    for source in args.input:
        crop, effective = dimensions.get(source.name, (960, 960))
        run.append(f"{source.resolve()}\t{crop}\t{effective}")
    result = subprocess.run(run, capture_output=True, text=True, timeout=600)
    (args.out / "reader.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Reader probe exited {result.returncode}; see reader.log")
    measured = json.loads(result.stdout.splitlines()[-1])
    datasets = None
    if matrix:
        sys.path.insert(0, str(ROOT / "eval"))
        from metrics import score_lines

        predictions = {row["fixture"]: bytes.fromhex(row["text_hex"]).decode()
                       for line in result.stdout.splitlines() if line.startswith('{"fixture":')
                       for row in [json.loads(line)]}
        if len(predictions) != len(matrix):
            raise RuntimeError("Incomplete matrix predictions")
        datasets = {name: score_lines([{**row, "pred": predictions[f"{row['id']}.f32"]}
                                      for row in matrix if row["dataset"] == name])["all"]
                    for name in ("vita-test", "psp-test")}
    input_hashes = [{"name": p.name, "sha256": digest(p),
                     "crop_effective_widths": dimensions.get(p.name, (960, 960))} for p in args.input]
    summary = {
        "scope": "Host pinned static MNN reader/path raw-output and cleanup comparison; "
                 "not console RAM/timing" + ("; full shared real-crop CER" if matrix else "; not CER"),
        "command": "python ports/meiki/verify_reader.py --mnn-source <pinned-MNN> "
                   "--mnn-library <host-static-MNN> --model <pinned-recognizer> "
                   + ("--matrix <pinned-private-ground_truth.json>" if matrix else
                      "--input <private-hand-checked-tensors>") + " --out <private-output>"
                   + (" --cases <private-cases.tsv>" if args.cases else "")
                   + (" --sanitize" if args.sanitize else ""),
        "platform": platform.platform(),
        "compiler": subprocess.check_output([cxx, "--version"], text=True).splitlines()[0],
        "adapter_sanitizers": args.sanitize,
        "mnn_archive_instrumented_by_this_command": False,
        "mnn_version": pins["mnn"]["version"], "mnn_commit": pins["mnn"]["commit"],
        "mnn_library_sha256": digest(args.mnn_library),
        "mnn_patch_sha256": digest(PORT / "mnn-vita.patch"),
        "model_bytes": args.model.stat().st_size, "model_sha256": digest(args.model),
        "workspace_bytes": 16 * 1024 * 1024, "reader_chunk_bytes": 8192,
        "reader_repetitions_per_fixture": 2 if matrix else 3,
        "path_repetitions_per_fixture": 2 if matrix else 1,
        "inputs": {"count": len(input_hashes),
                   "manifest_sha256": hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()}
                  if matrix else input_hashes,
        "case_manifest_sha256": digest(args.cases) if args.cases else None,
        "matrix_manifest_sha256": digest(args.matrix) if matrix else None,
        "eval_manifests": {name: digest(ROOT / "eval/manifests" / name)
                           for name in ("vita-test.jsonl", "psp-test.jsonl")},
        "source_sha256": {name: digest(PORT / name) for name in
                          ("runtime.cpp", "runtime.h", "model_reader.c", "model_reader.h", "reader_probe.cpp")},
        "measured": measured,
        "datasets": datasets,
        "limits": ("Complete451 hand-checked crop tensors from the pinned private bundle. " if matrix else
                   "Four supplied real-VN preprocessing tensors, not the full shared evaluation set. ") +
                  "System host C++ metadata heap is not the isolated Vita 4MiB heap. "
                  "Actual Vita engine metadata bounds remain separately guarded. "
                  "Reader buffer and MNN owning copy are both included in the measured 16MiB pool peak.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()

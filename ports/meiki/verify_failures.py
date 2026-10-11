#!/usr/bin/env python3
"""Compile and fault-check the C binding against pinned, patched static MNN.

Weights and input tensors are supplied privately. Checks cover repeated invalid
loads, sampled allocator failures/C++ exceptions and recovery on the same pool.
The result does not establish physical-Vita RAM or unrestricted MNN OOM safety.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
from pathlib import Path

PORT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnn-source", type=Path, required=True)
    parser.add_argument("--mnn-library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=128,
                        help="Allocation-ordinal sampling stride, in addition to load/output boundaries")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    args.out = args.out.resolve()
    expected = json.loads((PORT / "dependencies.json").read_text())["converted"]["sha256"]
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != expected:
        parser.error("Model does not match the pinned converted Meiki artifact")
    if args.input.stat().st_size != 3 * 32 * 960 * 4:
        parser.error("Input must contain the upstream BGR NCHW float32 tensor")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cc, cxx = os.environ.get("CC", "cc"), os.environ.get("CXX", "c++")
    flags = ["-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", f"-I{PORT}"]
    arena = args.out / "arena.o"
    subprocess.run([cc, "-std=c99", *flags, "-c", str(PORT / "arena.c"), "-o", str(arena)], check=True)
    probe = args.out / "failure-probe"
    subprocess.run([cxx, "-std=c++17", *flags, f"-I{args.mnn_source / 'include'}",
                    str(PORT / "runtime_failure_probe.cpp"), str(PORT / "runtime.cpp"),
                    str(PORT / "model_reader.c"), str(arena),
                    str(args.mnn_library), "-pthread", "-lm", "-ldl",
                    "-Wl,--wrap=MNNMemoryAllocAlign", "-Wl,--wrap=MNNMemoryCallocAlign",
                    "-Wl,--wrap=MNNMemoryFreeAlign", "-Wl,--wrap=meiki_arena_alloc",
                    "-Wl,--wrap=_Znwm", "-Wl,--wrap=_Znam",
                    "-Wl,--wrap=MNNEmbeddingExceptionSupport", "-o", str(probe)], check=True)
    invalid = args.out / "invalid.mnn"
    invalid.write_bytes(b"invalid Meiki model" * 4)
    missing = args.out / "absent.mnn"
    if missing.exists():
        parser.error(f"Expected missing fixture already exists: {missing}")
    rows = []

    def check(mode, ordinal):
        bad_path = missing if mode == "missing" else invalid
        command = [str(probe), mode, str(args.model.resolve()), str(args.input.resolve()),
                   str(bad_path), str(ordinal)]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{mode} ordinal {ordinal} timed out") from error
        (args.out / f"{mode}-{ordinal}.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(f"{mode} ordinal {ordinal} exited {process.returncode}; see its log")
        row = json.loads(process.stdout.splitlines()[-1])
        if not row["recovered"] or (mode in ("missing", "invalid") and row["maximum_retained"]):
            raise RuntimeError(f"Failed recovery or invalid-model cleanup: {row}")
        rows.append(row)
        return row

    baseline = check("baseline", 1)
    check("missing", 100)
    check("invalid", 100)
    rejected_backend = check("no-exceptions", 1)
    if rejected_backend["aligned_calls"] or rejected_backend["metadata_calls"]:
        raise RuntimeError("Unsupported backend was rejected after model allocations")
    total = baseline["aligned_calls"]
    points = sorted(set(range(1, min(97, total + 1))) |
                    set(range(max(1, total - 15), total + 1)) |
                    set(range(1, total + 1, args.stride)))
    for mode in ("arena", "exception", "other"):
        for ordinal in points:
            row = check(mode, ordinal)
            if not row["injected"]:
                raise RuntimeError(f"Expected fault was not reached: {row}")
        print(f"{mode}: {len(points)} failures recovered", flush=True)
    metadata_total = baseline["metadata_calls"]
    metadata_points = sorted({1, 2, 3, 4, 5, 10, 20, 40, 80} |
                             {max(1, metadata_total * i // 16) for i in range(1, 17)} |
                             set(range(max(1, metadata_total - 4), metadata_total + 1)))
    for ordinal in metadata_points:
        row = check("metadata", ordinal)
        if not row["injected"]:
            raise RuntimeError(f"Expected metadata fault was not reached: {row}")
    summary = {
        "kind": "host static MNN fault injection; not physical-Vita memory proof",
        "platform": platform.platform(),
        "compiler": subprocess.check_output([cxx, "--version"], text=True).splitlines()[0],
        "model_sha256": expected,
        "library_sha256": hashlib.sha256(args.mnn_library.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256((PORT / "mnn-vita.patch").read_bytes()).hexdigest(),
        "missing_model_failures": 100,
        "invalid_model_failures": 100,
        "exception_support_guard": True,
        "aligned_failure_ordinals": points,
        "cpp_exception_ordinals": points,
        "other_exception_ordinals": points,
        "metadata_exception_ordinals": metadata_points,
        "all_failures_recovered": True,
        "failure_cases_with_retained_pool_bytes": sum(bool(r["maximum_retained"]) for r in rows),
        "maximum_retained_pool_bytes_on_failure": max(r["maximum_retained"] for r in rows),
        "limits": "Sampled failures in the pinned recognizer, with a working host metadata heap; "
                  "constructor unwinding may leave pool allocations, which are reported and discarded. "
                  "No exhaustive metadata-OOM or game-process heap proof.",
    }
    (args.out / "failures.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("ordinals")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify allocation-free RGBA preprocessing against pinned upstream Meiki.

Private images, tensors and per-line predictions remain below --out. The
summary contains only derived counts. Optionally run the existing bounded
host recognizer to measure any observed tensor differences on the full set.
"""

import argparse
import ctypes as ct
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import score_lines  # noqa: E402
from sets import load_real  # noqa: E402

ELEMENTS = 3 * 32 * 960
SCRATCH_BYTES = 13440
ROWS = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.POINTER(ct.c_ubyte))
CANCEL = ct.CFUNCTYPE(ct.c_int, ct.c_void_p)


class Image(ct.Structure):
    _fields_ = [("ud", ct.c_void_p), ("width", ct.c_uint), ("height", ct.c_uint), ("rows", ROWS), ("cancelled", CANCEL)]


class Crop(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ("x", "y", "w", "h")]


class Info(ct.Structure):
    _fields_ = [("effective_w", ct.c_uint), ("effective_h", ct.c_uint)]


class Binding:
    def __init__(self, library):
        self.function = ct.CDLL(str(library)).meiki_preprocess_rgba
        self.function.argtypes = [
            ct.POINTER(Image),
            ct.POINTER(Crop),
            ct.POINTER(ct.c_float),
            ct.c_size_t,
            ct.c_void_p,
            ct.c_size_t,
            ct.POINTER(Info),
        ]
        self.function.restype = ct.c_int

    def prepare(
        self,
        rgba,
        crop=None,
        *,
        fail_read=None,
        cancel_after=None,
        output_elements=ELEMENTS,
        scratch_bytes=SCRATCH_BYTES,
        unaligned=False,
        unaligned_output=False,
        overlap=False,
    ):
        height, width = rgba.shape[:2]
        crop = crop or (0, 0, width, height)
        state = {"reads": 0, "cancel_checks": 0}

        @ROWS
        def read(_ud, y, n, dest):
            state["reads"] += 1
            assert 0 <= y < height and n == 1
            if fail_read is not None and state["reads"] >= fail_read:
                return 0
            packed = np.ascontiguousarray(rgba[y])
            ct.memmove(dest, packed.ctypes.data, width * 4)
            return 1

        @CANCEL
        def cancelled(_ud):
            state["cancel_checks"] += 1
            return cancel_after is not None and state["cancel_checks"] >= cancel_after

        image = Image(None, width, height, read, cancelled)
        result = np.full(ELEMENTS + 16, -123.25, dtype=np.float32)
        scratch = np.full(SCRATCH_BYTES + 64, 0xA5, dtype=np.uint8)
        scratch_offset = 16 + int(unaligned)
        info = Info(99, 99)
        output_address = result.ctypes.data + int(unaligned_output)
        scratch_address = result.ctypes.data + 8 if overlap else scratch.ctypes.data + scratch_offset
        rc = self.function(
            ct.byref(image),
            ct.byref(Crop(*crop)),
            ct.cast(output_address, ct.POINTER(ct.c_float)),
            output_elements,
            scratch_address,
            scratch_bytes,
            ct.byref(info),
        )
        assert np.all(result[ELEMENTS:] == -123.25), "Output guard was overwritten"
        assert np.all(scratch[:scratch_offset] == 0xA5), "Scratch prefix was overwritten"
        assert np.all(scratch[scratch_offset + SCRATCH_BYTES :] == 0xA5), "Scratch suffix was overwritten"
        if rc:
            assert (info.effective_w, info.effective_h) == (0, 0)
        return rc, result[:ELEMENTS].reshape(3, 32, 960), info, state


def expected(upstream, rgba, crop=None):
    x, y, w, h = crop or (0, 0, rgba.shape[1], rgba.shape[0])
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    batch, _, metadata = upstream._preprocess_for_recognition(bgr, [{"bbox": [x, y, x + w, y + h]}], [0], False)
    return batch[0], metadata[0]


def boundaries(binding, upstream):
    rng = np.random.default_rng(81246)
    geometry = [
        (1, 1),
        (3, 64),
        (5, 64),
        (7, 64),
        (960, 1),
        (960, 544),
        (480, 272),
        (17, 32),
        (960, 32),
        (2, 2),
        (16, 64),
        (1, 63),
    ]
    counts = {
        "successful_geometries": 0,
        "geometry_tensor_differences": 0,
        "invalid_cases": 0,
        "source_failure_cases": 0,
        "cancellation_cases": 0,
    }
    for width, height in geometry:
        # View with extra columns gives a real stride; callback must pack rows.
        rgba = rng.integers(0, 256, (height, width + 7, 4), dtype=np.uint8)[:, :width]
        rc, tensor, info, _ = binding.prepare(rgba)
        assert rc == 0, (width, height, rc)
        reference, metadata = expected(upstream, rgba)
        assert (info.effective_w, info.effective_h) == (metadata["effective_w"], metadata["effective_h"])
        counts["successful_geometries"] += 1
        counts["geometry_tensor_differences"] += not np.array_equal(tensor, reference)
        alpha_changed = rgba.copy()
        alpha_changed[:, :, 3] ^= 0xFF
        assert np.array_equal(tensor, binding.prepare(alpha_changed)[1]), "Alpha affected BGR input"
    image = rng.integers(0, 256, (80, 60, 4), dtype=np.uint8)
    crop = (3, 7, 31, 19)
    rc, tensor, info, _ = binding.prepare(image, crop)
    reference, metadata = expected(upstream, image, crop)
    assert rc == 0 and (info.effective_w, info.effective_h) == (metadata["effective_w"], metadata["effective_h"])
    counts["successful_geometries"] += 1
    counts["geometry_tensor_differences"] += not np.array_equal(tensor, reference)
    for kwargs in (
        {"crop": (0, 0, 0, 19)},
        {"crop": (59, 0, 2, 19)},
        {"crop": (0, 79, 31, 2)},
        {"output_elements": ELEMENTS - 1},
        {"scratch_bytes": SCRATCH_BYTES - 1},
        {"unaligned": True},
        {"unaligned_output": True},
        {"overlap": True},
    ):
        rc, _, _, state = binding.prepare(image, **kwargs)
        assert rc == -1 and state["reads"] == 0
        counts["invalid_cases"] += 1
    for width, height in ((1, 64), (1, 544), (961, 32), (32, 545)):
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rc, _, _, state = binding.prepare(rgba)
        assert rc == -1 and state["reads"] == 0
        counts["invalid_cases"] += 1
    for after in (1, 2, 3, 6, 20):
        rc, _, _, _ = binding.prepare(image, cancel_after=after)
        assert rc == -3
        counts["cancellation_cases"] += 1
    for after in (1, 2, 20):
        rc, _, _, _ = binding.prepare(image, fail_read=after)
        assert rc == -2
        counts["source_failure_cases"] += 1
    return counts


def reply(process):
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("Bounded recognizer stopped")
        if line.startswith('{"ms":'):
            result = json.loads(line)
            if result["rc"] or result["allocation_failed"] or result["heap_remaining"]:
                raise RuntimeError(f"Recognizer failed: {result}")
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/eval")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if any((args.probe, args.model, args.reference)) and not all((args.probe, args.model, args.reference)):
        parser.error("--probe, --model and --reference must be given together")
    args.out.mkdir(parents=True, exist_ok=True)
    library = args.out / "preprocess.so"
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(ROOT / "ports/meiki/preprocess.c"),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    sys.path.insert(0, str(args.upstream))
    from meikiocr import MeikiOCR

    upstream = MeikiOCR.__new__(MeikiOCR)
    binding = Binding(library)
    boundary_results = boundaries(binding, upstream)
    references = (
        {}
        if not args.reference
        else {row["id"]: row["pred"] for row in map(json.loads, args.reference.read_text().splitlines())}
    )
    counts = {
        "cases": 0,
        "tensor_different_cases": 0,
        "tensor_different_elements": 0,
        "max_pixel_difference": 0,
        "metadata_differences": 0,
    }
    predictions, text_differences, peaks = {}, 0, []
    log = (args.out / "host.log").open("w")
    process = (
        None
        if not args.probe
        else subprocess.Popen(
            [str(args.probe), str(args.model), "16"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
    )
    try:
        with (args.out / "cases.jsonl").open("w") as output_log:
            for dataset in ("vita-test", "psp-test"):
                lines, _ = load_real(dataset, args.data_dir)
                predictions[dataset] = []
                for index, line in enumerate(lines, 1):
                    rgba = cv2.cvtColor(line.crop_rgb, cv2.COLOR_RGB2RGBA)
                    rgba[:, :, 3] = np.arange(rgba.shape[1], dtype=np.uint8)
                    rc, tensor, info, _ = binding.prepare(rgba)
                    assert rc == 0, (line.id, rc)
                    reference, metadata = expected(upstream, rgba)
                    changed = int(np.count_nonzero(tensor != reference))
                    pixel_delta = int(np.rint(np.max(np.abs(tensor - reference)) * 255))
                    counts["cases"] += 1
                    counts["tensor_different_cases"] += changed != 0
                    counts["tensor_different_elements"] += changed
                    counts["max_pixel_difference"] = max(counts["max_pixel_difference"], pixel_delta)
                    counts["metadata_differences"] += (info.effective_w, info.effective_h) != (
                        metadata["effective_w"],
                        metadata["effective_h"],
                    )
                    row = {"id": line.id, "changed_elements": changed, "max_pixel_difference": pixel_delta}
                    if process:
                        path = args.out / "input.f32"
                        tensor.astype("<f4").tofile(path)
                        process.stdin.write(f"{path}\t{rgba.shape[1]}\t{info.effective_w}\n")
                        process.stdin.flush()
                        result = reply(process)
                        text = bytes.fromhex(result["text_hex"]).decode()
                        text_differences += text != references[line.id]
                        peaks.append(result["heap_peak"])
                        predictions[dataset].append(
                            {"id": line.id, "pred": text, "text": line.text, "styles": line.styles, "vn": line.vn}
                        )
                        row["prediction"] = text
                        row["text_changed"] = text != references[line.id]
                    output_log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if index % 25 == 0:
                        print(f"{dataset}: {index}/{len(lines)}", flush=True)
    finally:
        if process:
            process.stdin.close()
            rc = process.wait(timeout=20)
            process.stdout.close()
            assert rc == 0, rc
        log.close()
    summary = {
        "kind": "Host C RGBA binding vs Meiki 0.3.5/OpenCV 4.12.0 uint8 INTER_LINEAR",
        "scratch_bytes": SCRATCH_BYTES,
        "output_bytes": ELEMENTS * 4,
        "dynamic_allocations": 0,
        **counts,
        "boundaries": boundary_results,
    }
    if process:
        summary["recognition"] = {
            "text_differences_from_upstream_preprocessed_int8_MNN": text_differences,
            "peak_model_tensor_bytes": max(peaks),
            "datasets": {name: score_lines(rows) for name, rows in predictions.items()},
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "recognition"}, indent=2))
    if process:
        print(json.dumps({key: value for key, value in summary["recognition"].items() if key != "datasets"}, indent=2))
    if (
        counts["metadata_differences"]
        or counts["tensor_different_cases"]
        or boundary_results["geometry_tensor_differences"]
        or text_differences
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

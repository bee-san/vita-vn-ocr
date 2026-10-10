#!/usr/bin/env python3
"""Compare bounded detector RGBA preprocessing with pinned upstream/OpenCV.

Real images and generated shared libraries/reports stay in caller-supplied
private directories. This verifies the existing320x192 author model's input.
"""

import argparse
import ctypes as C
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ELEMENTS = 3 * 192 * 320
SCRATCH = 9600
Rows = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint, C.c_uint, C.POINTER(C.c_ubyte))
Cancel = C.CFUNCTYPE(C.c_int, C.c_void_p)


class Image(C.Structure):
    _fields_ = [("ud", C.c_void_p), ("width", C.c_uint), ("height", C.c_uint), ("rows", Rows), ("cancelled", Cancel)]


class Crop(C.Structure):
    _fields_ = [(name, C.c_uint) for name in ("x", "y", "w", "h")]


class Info(C.Structure):
    _fields_ = [
        ("effective_w", C.c_uint),
        ("effective_h", C.c_uint),
        ("target_width", C.c_int32),
        ("target_height", C.c_int32),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=False)
    library = args.out / "detect-preprocess.so"
    command = [
        "cc",
        "-std=c99",
        "-O2",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(Path(__file__).with_name("detect_preprocess.c")),
        "-lm",
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True)
    function = C.CDLL(str(library)).meiki_detect_preprocess_rgba
    function.argtypes = [
        C.POINTER(Image),
        C.POINTER(Crop),
        C.POINTER(C.c_float),
        C.c_size_t,
        C.c_void_p,
        C.c_size_t,
        C.POINTER(Info),
    ]
    function.restype = C.c_int
    sys.path[:0] = [str(root / "eval"), str(args.upstream)]
    import meikiocr.ocr as upstream
    import sets

    # The upstream function uses model-resolution globals, rather than a
    # parameter. Substitute the author's existing320x192 variant dimensions.
    upstream.INPUT_DET_WIDTH = 320
    upstream.INPUT_DET_HEIGHT = 192
    ocr = upstream.MeikiOCR.__new__(upstream.MeikiOCR)
    comparisons = []
    controls = []

    def run(
        rgba, box, *, cancel_at=None, fail_row=None, output_elements=ELEMENTS, scratch_bytes=SCRATCH, overlap=False
    ):
        calls = {"rows": 0, "cancel": 0}

        @Rows
        def rows(_ud, y, n, destination):
            calls["rows"] += 1
            if fail_row is not None and y >= fail_row:
                return -7
            if y + n > rgba.shape[0]:
                return -8
            data = np.ascontiguousarray(rgba[y : y + n]).tobytes()
            C.memmove(destination, data, len(data))
            return n

        @Cancel
        def cancelled(_ud):
            calls["cancel"] += 1
            return int(cancel_at is not None and calls["cancel"] >= cancel_at)

        image = Image(None, rgba.shape[1], rgba.shape[0], rows, cancelled)
        output_guard = np.full(ELEMENTS + 8, np.float32(-99.0))
        output = output_guard[4:-4]
        scratch_guard = (C.c_ubyte * (SCRATCH + 16))(*([0xA7] * (SCRATCH + 16)))
        scratch = C.addressof(scratch_guard) + 8
        if overlap:
            scratch = output.ctypes.data
        info = Info()
        rc = function(
            C.byref(image),
            C.byref(Crop(*box)),
            output.ctypes.data_as(C.POINTER(C.c_float)),
            output_elements,
            scratch,
            scratch_bytes,
            C.byref(info),
        )
        assert np.all(output_guard[:4] == -99) and np.all(output_guard[-4:] == -99)
        assert list(scratch_guard[:8]) == [0xA7] * 8 and list(scratch_guard[-8:]) == [0xA7] * 8
        return rc, output.reshape(1, 3, 192, 320), info, calls

    def compare(rgba, box, tag):
        rc, actual, info, _ = run(rgba, box)
        x, y, w, h = box
        bgr = cv2.cvtColor(rgba[y : y + h, x : x + w], cv2.COLOR_RGBA2BGR)
        expected, scale = ocr._preprocess_for_detection(bgr)
        geometry = [int(w * scale), int(h * scale), int(320 / scale), int(192 / scale)]
        measured = [info.effective_w, info.effective_h, info.target_width, info.target_height]
        differences = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
        comparisons.append(
            {
                "tag": tag,
                "rc": rc,
                "different_elements": differences,
                "geometry": measured,
                "geometry_equal": measured == geometry,
            }
        )

    cache = sets.ImageCache()
    for dataset in ("vita-test", "psp-test"):
        for i, (path, (x, y, w, h), _) in enumerate(sets.regions(dataset, args.eval_data)):
            rgba = cv2.cvtColor(cache.get(path), cv2.COLOR_RGB2RGBA)
            compare(rgba, (x, y, w, h), f"{dataset}:{i}")
    rng = np.random.default_rng(46)
    real_regions = len(comparisons)
    for w, h in (
        (1, 1),
        (2, 2),
        (640, 384),
        (960, 544),
        (319, 191),
        (320, 192),
        (321, 193),
        (13, 47),
        (47, 13),
        (33, 544),
    ):
        storage = rng.integers(0, 256, (h, w + 7, 4), dtype=np.uint8)
        compare(storage[:, :w], (0, 0, w, h), f"geometry:{w}x{h}:strided")
    source = rng.integers(0, 256, (64, 96, 4), dtype=np.uint8)
    for tag, box, options, expected_rc in [
        ("empty", (0, 0, 0, 1), {}, -1),
        ("outside", (90, 0, 7, 4), {}, -1),
        ("short_output", (0, 0, 96, 64), {"output_elements": ELEMENTS - 1}, -1),
        ("short_scratch", (0, 0, 96, 64), {"scratch_bytes": SCRATCH - 1}, -1),
        ("overlap", (0, 0, 96, 64), {"overlap": True}, -1),
        ("read_failure", (0, 0, 96, 64), {"fail_row": 20}, -2),
        *[(f"cancel:{point}", (0, 0, 96, 64), {"cancel_at": point}, -3) for point in (1, 2, 3, 12, 100)],
    ]:
        rc, _, info, calls = run(source, box, **options)
        assert rc == expected_rc and not any(
            (info.effective_w, info.effective_h, info.target_width, info.target_height)
        ), (tag, rc)
        if expected_rc == -1:
            assert calls["rows"] == 0
        controls.append({"tag": tag, "rc": rc, "row_calls": calls["rows"]})
    summary = {
        "kind": "Exact detector preprocessing tensors against existing upstream and OpenCV",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "command": command,
        "real_regions": real_regions,
        "comparisons": len(comparisons),
        "different_elements": sum(row["different_elements"] for row in comparisons),
        "geometry_differences": sum(not row["geometry_equal"] for row in comparisons),
        "controls": controls,
        "rows": comparisons,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    if summary["different_elements"] or summary["geometry_differences"] or any(row["rc"] for row in comparisons):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

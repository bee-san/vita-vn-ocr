#!/usr/bin/env python3
"""Prepare the author's existing Meiki320 detector without training.

Pins weights, fixes batch one and simplifies ONNX shape logic. Both original
inputs and all64 raw outputs remain. An optional pinned MNN converter applies
weight-int8 storage; runtime Memory_Low CPU convolution quantization is separate.
Generated models belong in private storage or release assets, never Git.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import onnx
import onnxsim
from normalize_mnn_uuid import normalize

SOURCE_SHA = "8cdc5daa5c13a5f93adb40ee2e53ad1a3d2f7e372584c5c665b038adeb74ae6d"
PREPARED_SHA = "c4219d056b083d97f3a2d1d44d669458d7ae3f65ace8ca19250c5cedcb1f8fbd"
MNN_SHA = "f0865ed291846a130905527f67c33f17e64d0f2c5eb38378ed1faea57f74b1ba"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--converter", type=Path)
    args = parser.parse_args()
    if onnx.__version__ != "1.18.0" or onnxsim.__version__ != "0.4.36":
        raise ValueError("Need pinned onnx1.18.0 and onnxsim0.4.36")
    if args.source.stat().st_size != 14084361 or digest(args.source) != SOURCE_SHA:
        raise ValueError("Source does not match the pinned existing author model")
    args.out.mkdir(parents=True, exist_ok=False)
    model = onnx.load(args.source)
    for tensor in list(model.graph.input) + list(model.graph.output):
        batch = tensor.type.tensor_type.shape.dim[0]
        batch.ClearField("dim_param")
        batch.dim_value = 1
    prepared, valid = onnxsim.simplify(model)
    if not valid:
        raise ValueError("Simplifier equivalence check failed")
    onnx.checker.check_model(prepared)
    path = args.out / "meiki-detect-batch1.onnx"
    onnx.save(prepared, path)
    if digest(path) != PREPARED_SHA:
        raise ValueError("Prepared bytes do not match the pinned conversion")
    manifest = json.loads(Path(__file__).with_name("detector-dependencies.json").read_text())
    manifest["preparation_command"] = [
        "python",
        "ports/meiki/prepare_detect.py",
        "--source",
        "<private-source>",
        "--out",
        "<private-output>",
    ]
    if args.converter:
        converted = args.out / "meiki-detect-int8.mnn"
        command = [
            str(args.converter.resolve()),
            "-f",
            "ONNX",
            "--modelFile",
            str(path.resolve()),
            "--MNNModel",
            str(converted.resolve()),
            "--bizCode",
            "meiki-detector-diagnostic",
            "--weightQuantBits",
            "8",
        ]
        with (args.out / "convert.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        normalize(
            converted,
            manifest["converted"]["mnn_uuid"],
            expected_sha=MNN_SHA,
            expected_bytes=4119220,
            biz_code="meiki-detector-diagnostic",
        )
        if converted.stat().st_size != 4119220 or digest(converted) != MNN_SHA:
            raise ValueError("Converted bytes do not match pinned MNN3.2.0 conversion")
        manifest["conversion_command"] = command
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

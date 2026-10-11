#!/usr/bin/env python3
"""Prepare the existing Meiki recognizer for a single low-memory console job.

This does not train or replace Meiki. It fixes batch size to one and computes
the encoder's class maximum in chunks instead of materializing its 11 MiB
630 x 4372 score matrix. Outputs retain Meiki's Unicode codes, boxes and scores.
Only the horizontal recognizer is loaded; the console supplies line crops.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxsim
from onnx import helper, numpy_helper

REVISION = "a28cf5874dc2438ebb1c86336be26bcec51e3375"
FILENAME = "meiki.text.rec.v0.960x32.onnx"
URL = f"https://huggingface.co/rtr46/meiki.txt.recognition.v0/resolve/{REVISION}/{FILENAME}"
SHA256 = "3e96bc772fbee9717e536a6353032bb944c3382dd2f6960ef4890decda43b000"
SIZE = 18593254


def source_model(path):
    """Fetch the pinned LGPL-3.0 weights once, or validate an existing copy."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "vita-vn-ocr-meiki-port/1"})
        with urllib.request.urlopen(request, timeout=180) as response:
            data = response.read(SIZE + 1)
        if len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Meiki download does not match the pinned source")
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    data = path.read_bytes()
    if len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Meiki source does not match the pinned model")
    return onnx.load_model_from_string(data)


def stream_class_max(model, chunk_size):
    """Equivalent MatMul + bias + ReduceMax, with bounded intermediate width."""
    if not 1 <= chunk_size <= 4372:
        raise ValueError("chunk size must be between 1 and 4372")
    nodes = list(model.graph.node)
    by_name = {node.name: node for node in nodes}
    prefix = "/model/decoder/"
    multiply = by_name[prefix + "enc_score_head/MatMul"]
    addition = by_name[prefix + "enc_score_head/Add"]
    reduction = by_name[prefix + "ReduceMax"]
    weights = {tensor.name: tensor for tensor in model.graph.initializer}
    matrix = numpy_helper.to_array(weights[multiply.input[1]])
    bias_name = next(name for name in addition.input if name != multiply.output[0])
    bias = numpy_helper.to_array(weights[bias_name])
    if matrix.shape != (128, 4372) or bias.shape != (4372,):
        raise ValueError("unexpected Meiki encoder head")
    if list(reduction.input) != list(addition.output):
        raise ValueError("unexpected head connection")
    attributes = {a.name: helper.get_attribute_value(a) for a in reduction.attribute}
    if attributes != {"axes": [-1], "keepdims": 0}:
        raise ValueError("unexpected class reduction")
    consumers = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node.name)
    if consumers[multiply.output[0]] != [addition.name] or consumers[addition.output[0]] != [reduction.name]:
        raise ValueError("score matrix has another consumer; cannot stream it")

    replacement, constants = [], []
    previous = None
    for index, start in enumerate(range(0, matrix.shape[1], chunk_size)):
        end = min(start + chunk_size, matrix.shape[1])
        name = f"vita_classmax_{index}"
        constants.extend(
            [
                numpy_helper.from_array(np.ascontiguousarray(matrix[:, start:end]), name + "_weight"),
                numpy_helper.from_array(bias[start:end].copy(), name + "_bias"),
            ]
        )
        replacement.extend(
            [
                helper.make_node(
                    "MatMul", [multiply.input[0], name + "_weight"], [name + "_product"], name=name + "_matmul"
                ),
                helper.make_node("Add", [name + "_product", name + "_bias"], [name + "_scores"], name=name + "_add"),
                helper.make_node(
                    "ReduceMax", [name + "_scores"], [name + "_maximum"], name=name + "_reduce", axes=[-1], keepdims=0
                ),
            ]
        )
        if previous is None:
            previous = name + "_maximum"
        else:
            replacement.append(
                helper.make_node("Max", [previous, name + "_maximum"], [name + "_accumulator"], name=name + "_max")
            )
            previous = name + "_accumulator"
    replacement.append(helper.make_node("Identity", [previous], list(reduction.output), name="vita_classmax_output"))
    removed = {multiply.name, addition.name, reduction.name}
    transformed = []
    for node in nodes:
        if node.name == multiply.name:
            transformed.extend(replacement)
        elif node.name not in removed:
            transformed.append(node)
    del model.graph.node[:]
    model.graph.node.extend(transformed)
    retained = [tensor for tensor in model.graph.initializer if tensor.name not in (multiply.input[1], bias_name)]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained + constants)
    # The original matrix's shape metadata is now stale.
    del model.graph.value_info[:]
    return model


def prepare(model, chunk_size=128):
    model = stream_class_max(model, chunk_size)
    # Boxes are expressed in model pixels, then mapped to the console crop by
    # the caller. Constant geometry also removes the ONNX export's shape logic.
    inputs = [item for item in model.graph.input if item.name != "orig_target_sizes"]
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    model.graph.initializer.append(numpy_helper.from_array(np.array([[960, 32]], dtype=np.int64), "orig_target_sizes"))
    result, valid = onnxsim.simplify(model, overwrite_input_shapes={"images": [1, 3, 32, 960]})
    if not valid:
        raise ValueError("ONNX simplification failed its equivalence check")
    onnx.checker.check_model(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("models/meiki") / FILENAME)
    parser.add_argument("--output", type=Path, default=Path("models/meiki/meiki-stream.onnx"))
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    output = prepare(source_model(args.source), args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output, args.output)
    metadata = {
        "source_url": URL,
        "source_sha256": SHA256,
        "source_license": "LGPL-3.0",
        "batch": 1,
        "input": [1, 3, 32, 960],
        "box_coordinates": "model pixels",
        "class_chunk": args.chunk_size,
        "largest_head_tensor_bytes": 630 * args.chunk_size * 4,
        "original_head_tensor_bytes": 630 * 4372 * 4,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "onnx_version": onnx.__version__,
        "onnxsim_version": onnxsim.__version__,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

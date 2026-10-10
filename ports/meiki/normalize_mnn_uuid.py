#!/usr/bin/env python3
"""Normalize MNN3.2.0 random UUID metadata, then verify strict model bytes.

Only the existing36-byte Net.mnn_uuid string changes. No tensor, operator,
coefficient, graph, layout or quantization byte changes. A mismatched resulting
hash leaves the original file intact. This makes strict model guards usable.
"""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


def normalize(path, uuid, *, expected_sha, expected_bytes, biz_code):
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", uuid):
        raise ValueError("Expected canonical36-byte UUID")
    data = bytearray(path.read_bytes())
    if len(data) != expected_bytes:
        raise ValueError("Model size does not match the pinned conversion")
    # Pinned schema/current/MNN_generated.h: Net::mnn_uuid uses vtable offset26;
    # Net::bizCode uses offset4. This is a fixed-layout metadata adapter, with
    # full-result hash validation before writing any bytes to the file.
    root = struct.unpack_from("<I", data)[0]
    vtable = root - struct.unpack_from("<i", data, root)[0]
    if struct.unpack_from("<H", data, vtable)[0] < 28:
        raise ValueError("Unexpected MNN Net vtable")

    def string_field(offset):
        member = struct.unpack_from("<H", data, vtable + offset)[0]
        if not member:
            raise ValueError("Missing MNN metadata field")
        field = root + member
        string = field + struct.unpack_from("<I", data, field)[0]
        length = struct.unpack_from("<I", data, string)[0]
        start = string + 4
        if start + length >= len(data) or data[start + length] != 0:
            raise ValueError("Unexpected MNN metadata string layout")
        return start, length

    code_start, code_length = string_field(4)
    if data[code_start : code_start + code_length].decode("ascii") != biz_code:
        raise ValueError("Model bizCode does not match the pinned conversion command")
    start, length = string_field(26)
    if length != 36:
        raise ValueError("Unexpected MNN UUID length")
    previous = data[start : start + 36].decode("ascii")
    data[start : start + 36] = uuid.encode("ascii")
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha:
        raise ValueError("Model differs beyond its random UUID; original file preserved")
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": digest, "previous_uuid": previous, "pinned_uuid": uuid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("recognizer", "detector"), required=True)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    manifest_name = "dependencies.json" if args.model == "recognizer" else "detector-dependencies.json"
    conversion = json.loads(Path(__file__).with_name(manifest_name).read_text())["converted"]
    result = normalize(
        args.path,
        conversion["mnn_uuid"],
        expected_sha=conversion["sha256"],
        expected_bytes=conversion["bytes"],
        biz_code=conversion["mnn_biz_code"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

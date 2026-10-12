#!/usr/bin/env python3
"""Persistent JSONL Google Lens worker; optional owocr environment only."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import owocr.ocr

    root = Path(owocr.ocr.__file__).parent
    deps = {}
    installed = {}
    for name in ("owocr", "numpy", "Pillow", "loguru", "curl_cffi", "protobuf", "cffi", "certifi", "pycparser"):
        try:
            dist = importlib.metadata.distribution(name)
            deps[name] = dist.version
            files = []
            for relative in sorted(dist.files or [], key=str):
                if str(relative).endswith((".pyc", ".pyo")) or "__pycache__" in str(relative):
                    continue
                path = Path(dist.locate_file(relative))
                if path.is_file():
                    files.append([str(relative), sha(path)])
            installed[name] = hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    if deps["owocr"] != "1.26.7":
        raise ValueError("this protocol requires owocr==1.26.7")
    return {
        "engine": "GoogleLens",
        "png_encoder": {
            "optimized": owocr.ocr.optimized_png_encode,
            "module_sha256": sha(owocr.ocr.fpng_py_fix.__file__) if owocr.ocr.optimized_png_encode else None,
            "format": "lossless PNG; no resizing below 3000000 pixels",
        },
        "python": sys.version,
        "remote_service": "Google Lens is unversioned; only local engine/dependencies can be frozen",
        "dependencies": deps,
        "installed_sources_sha256": installed,
        "owocr_sources": {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))},
        "bridge_sha256": sha(__file__),
    }


def result_text(result):
    """Respect word separators; Lens normally leaves Line.text unset."""
    lines = []
    boxes = []
    for paragraph in result.paragraphs:
        for line in paragraph.lines:
            text = (
                line.text
                if line.text is not None
                else "".join(word.text + (word.separator or "") for word in line.words)
            )
            lines.append(text)
            box = line.bounding_box
            boxes.append({"text": text, "box": [box.left, box.top, box.right, box.bottom] if box else None})
    return "\n".join(lines), boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        info = environment()
    if args.info:
        print(json.dumps(info, sort_keys=True))
        return
    with contextlib.redirect_stdout(sys.stderr):
        from owocr.ocr import GoogleLens
        from PIL import Image

        engine = GoogleLens()
    if not engine.available:
        raise RuntimeError("Google Lens dependencies are unavailable")
    for raw in sys.stdin:
        request = json.loads(raw)
        started = time.perf_counter()
        try:
            path = Path(request["path"])
            if sha(path) != request["sha256"]:
                raise ValueError("Google input file SHA mismatch")
            with Image.open(path) as image:
                image.load()
                pixels = hashlib.sha256(image.tobytes()).hexdigest()
                if pixels != request["pixels_sha256"] or image.mode != request["mode"]:
                    raise ValueError("Google input decoded pixel/mode mismatch")
                with contextlib.redirect_stdout(sys.stderr):
                    ok, result = engine(image)
            if not ok:
                raise RuntimeError(str(result))
            text, lines = result_text(result)
            answer = {"id": request["id"], "ok": True, "text": text, "lines": lines}
        except Exception as exc:
            answer = {"id": request.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        answer["requested_at_unix"] = time.time() - (time.perf_counter() - started)
        answer["host_seconds"] = time.perf_counter() - started
        print(json.dumps(answer, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Immutable real-VN test capture and Google/C OCR proxy comparison, never selection."""

import argparse
import ctypes as C
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "eval"))
from metrics import lev, norm, score_hiragana  # noqa: E402
from pyvocr import Vocr, load_lib  # noqa: E402

from common import to_gray  # noqa: E402

BASELINE_SHA = "500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84"
BLOB_BYTES, ARENA_BYTES = 657664, 124832
SOURCE_FILES = [
    "eval/clannad_longrun.py",
    "eval/owocr_bridge.py",
    "eval/metrics.py",
    "eval/requirements-owocr-longrun.txt",
    "synth/common.py",
    "runtime/pyvocr.py",
    "runtime/vocr.c",
    "runtime/vocr.h",
    "runtime/vocr_lines.c",
    "runtime/vocr_lines.h",
]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_test(split):
    if split != "test":
        raise ValueError("long-run capture/comparison requires split=test; never training or selection")


def rect(value):
    values = [int(v) for v in value.split(",")]
    if len(values) != 4 or any(v < 0 for v in values[:2]) or min(values[2:]) <= 0:
        raise argparse.ArgumentTypeError("box must be nonnegative x,y and positive w,h")
    return values


def native_size(value):
    values = [int(v) for v in value.split(",")]
    if len(values) != 2 or min(values) <= 0:
        raise argparse.ArgumentTypeError("native size must be positive width,height")
    return values


def checked_box(box, shape):
    x, y, w, h = box
    if x < 0 or y < 0 or min(w, h) <= 0 or x + w > shape[1] or y + h > shape[0]:
        raise ValueError("box outside native image")
    return slice(y, y + h), slice(x, x + w)


def _decode_journal(raw):
    if raw and not raw.endswith(b"\n"):
        raise ValueError("partial journal record: recover explicitly, never silently truncate")
    rows, previous = [], None
    for line in raw.splitlines():
        row = json.loads(line)
        checksum = row.pop("event_sha256")
        if row.get("sequence") != len(rows) or row.get("previous_sha256") != previous or digest(row) != checksum:
            raise ValueError("journal sequence/hash corruption")
        row["event_sha256"] = checksum
        previous = checksum
        rows.append(row)
    return rows


def append_event(path, event):
    """Hash-chained, fsync'd append, locked against simultaneous follow reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        records = _decode_journal(stream.read())
        record = dict(event, sequence=len(records), previous_sha256=records[-1]["event_sha256"] if records else None)
        record["event_sha256"] = digest(record)
        stream.seek(0, 2)
        stream.write(canonical(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)
    return record


def read_journal(path):
    if not path.exists():
        return []
    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        raw = stream.read()
        fcntl.flock(stream, fcntl.LOCK_UN)
    return _decode_journal(raw)


def frozen_file(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen protocol mismatch: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return file_sha(path)


def store_image(directory, name, pixels):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"capture input already exists: {path}")
    image = Image.fromarray(np.ascontiguousarray(pixels))
    image.save(path)
    return {
        "path": str(path.relative_to(directory)),
        "sha256": file_sha(path),
        "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "mode": image.mode,
        "size": list(image.size),
    }


def verified_image(directory, entry):
    path = (directory / entry["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("capture path escapes session")
    if file_sha(path) != entry["sha256"]:
        raise ValueError("capture file hash corruption")
    with Image.open(path) as image:
        image.load()
        if (
            image.mode != entry["mode"]
            or list(image.size) != entry["size"]
            or hashlib.sha256(image.tobytes()).hexdigest() != entry["pixels_sha256"]
        ):
            raise ValueError("capture decoded pixel corruption")
        return np.asarray(image).copy()


class _LinesImage(C.Structure):
    _fields_ = [
        ("pixels", C.c_void_p),
        ("width", C.c_uint),
        ("height", C.c_uint),
        ("stride", C.c_size_t),
        ("bpp", C.c_uint),
    ]


class _Line(C.Structure):
    _fields_ = [("x", C.c_uint), ("y", C.c_uint), ("w", C.c_uint), ("h", C.c_uint), ("dark", C.c_uint)]


class _LineSet(C.Structure):
    _fields_ = [("lines", _Line * 16), ("count", C.c_uint), ("truncated", C.c_uint)]


class Native:
    def __init__(self, directory, model):
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / "libvocr-longrun.so"
        command = [
            "gcc",
            "-std=c99",
            "-O2",
            "-ffp-contract=off",
            "-DVOCR_NO_NEON",
            "-fPIC",
            "-shared",
            str(ROOT / "runtime/vocr.c"),
            str(ROOT / "runtime/vocr_lines.c"),
            "-lm",
            "-o",
            str(library),
        ]
        if not library.exists():
            subprocess.run(command, check=True)
        self.lib = load_lib(library)
        self.lib.vocr_lines_arena_bytes.argtypes = [C.c_uint, C.c_uint]
        self.lib.vocr_lines_arena_bytes.restype = C.c_size_t
        self.lib.vocr_find_lines.argtypes = [
            C.POINTER(_LinesImage),
            C.c_int,
            C.c_void_p,
            C.c_size_t,
            C.POINTER(_LineSet),
        ]
        if file_sha(model) != BASELINE_SHA or Path(model).stat().st_size != BLOB_BYTES:
            raise ValueError("requires immutable released FL10 W8I baseline")
        self.reader = Vocr(model, mode="w8i", max_width=1024, lib=self.lib)
        if self.reader.arena_bytes != ARENA_BYTES:
            raise ValueError("baseline runtime memory mismatch")
        self.info = {
            "model_sha256": BASELINE_SHA,
            "blob_bytes": self.reader.blob_bytes,
            "recognizer_arena_bytes": self.reader.arena_bytes,
            "mode": "w8i",
            "max_width": 1024,
            "library_sha256": file_sha(library),
            "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
            "compile_flags": ["-std=c99", "-O2", "-ffp-contract=off", "-DVOCR_NO_NEON", "-fPIC", "-shared", "-lm"],
            "finder": "scalar C GRAD; unchanged baseline geometry",
            "polarity": "C finder dark flag; invert crop once before both engines",
        }

    def crops(self, rgb):
        grey = np.ascontiguousarray(to_gray(rgb))
        h, w = grey.shape
        arena_bytes = self.lib.vocr_lines_arena_bytes(w, h)
        arena = C.create_string_buffer(arena_bytes)
        image = _LinesImage(grey.ctypes.data, w, h, grey.strides[0], 1)
        lines = _LineSet()
        rc = self.lib.vocr_find_lines(C.byref(image), 0, arena, arena_bytes, C.byref(lines))
        if rc:
            raise RuntimeError(f"C finder failed: {rc}")
        crops = []
        for line in lines.lines[: lines.count]:
            box = [line.x, line.y, line.w, line.h]
            crop = grey[checked_box(box, grey.shape)].copy()
            crops.append((255 - crop if line.dark else crop, box, bool(line.dark)))
        return crops, bool(lines.truncated), arena_bytes

    def text(self, crop):
        return self.reader.text(crop, invert=False)


def teacher_info(python):
    return json.loads(subprocess.check_output([str(python), str(ROOT / "eval/owocr_bridge.py"), "--info"], text=True))


def protocol(args, native):
    require_test(args.split)
    if args.complete_box is None and args.control_file is None:
        raise ValueError("require completion indicator box or operator control file; stability is not completeness")
    return {
        "schema": 1,
        "purpose": "new real-VN test diagnostics only; forbidden for model/preprocessing selection",
        "split": args.split,
        "session": args.session,
        "native_size": getattr(args, "native_size", [960, 544]),
        "roi": args.roi,
        "ignore_boxes": args.ignore_box + ([args.complete_box] if args.complete_box else []),
        "complete_box": args.complete_box,
        "complete_threshold": args.complete_threshold,
        "complete_pixels": args.complete_pixels,
        "control_file": str(args.control_file.resolve()) if args.control_file else None,
        "ink_threshold": args.ink_threshold,
        "stable_seconds": args.stable_seconds,
        "stable_samples": args.stable_samples,
        "interval": args.interval,
        "max_lines": args.max_lines,
        "idle_timeout": args.idle_timeout,
        "advance_key": args.advance_key,
        "key_hold_seconds": getattr(args, "key_hold_seconds", 0.3),
        "display": args.display,
        "window": args.window,
        "runtime": native.info,
        "sources": {name: file_sha(ROOT / name) for name in SOURCE_FILES},
        "host": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()},
        "host_dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow")},
        "teacher": teacher_info(args.owocr_python),
        "normalization": "eval.metrics.norm; no small-kana/dakuten/kanji folding",
        "reference": "Google Lens output is an automated proxy, not ground truth or measured human CER",
    }


def validate_lock(directory):
    lock_path = directory / "lock.json"
    lock = json.loads(lock_path.read_text())
    require_test(lock.get("split"))  # role checked before any capture pixels
    if lock.get("schema") != 1 or lock["runtime"]["model_sha256"] != BASELINE_SHA:
        raise ValueError("invalid long-run lock/model")
    for name, sha in lock["sources"].items():
        if file_sha(ROOT / name) != sha:
            raise ValueError(f"locked source changed: {name}")
    for name, version in lock["host_dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"locked host dependency changed: {name}")
    return lock, file_sha(lock_path)


def dialogue_signature(rgb, roi, ignore_boxes, threshold):
    """Only foreground brightness mask participates; no animated scenery/cursor pixels."""
    grey = to_gray(rgb)
    mask = grey[checked_box(roi, grey.shape)] >= threshold
    x, y, w, h = roi
    for box in ignore_boxes:
        checked_box(box, grey.shape)
        bx, by, bw, bh = box
        left, top, right, bottom = max(x, bx), max(y, by), min(x + w, bx + bw), min(y + h, by + bh)
        if left < right and top < bottom:
            mask[top - y : bottom - y, left - x : right - x] = False
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(), int(mask.sum())


class Stability:
    def __init__(self, seconds, samples, seen=()):
        self.seconds, self.samples, self.seen = seconds, samples, set(seen)
        self.last, self.since, self.count = None, None, 0

    def observe(self, signature, now, complete, kind="dialogue"):
        if signature != self.last:
            self.last, self.since, self.count = signature, now, 1
        else:
            self.count += 1
        if kind != "dialogue" or not complete:
            return False
        return signature not in self.seen and self.count >= self.samples and now - self.since >= self.seconds

    def accept(self, signature):
        self.seen.add(signature)


def capture_event(directory, rgb, roi, native, signature, lock_sha, session, completion):
    rows = read_journal(directory / "captures.jsonl")
    committed = sum(row.get("kind") == "capture" for row in rows)
    # Preserve interrupted allocations; never reuse/overwrite their input bytes.
    allocated = [
        int(path.name.split("-", 1)[0])
        for path in (directory / "inputs").glob("*")
        if path.name.split("-", 1)[0].isdigit()
    ]
    number = max([committed - 1, *allocated]) + 1
    prefix = f"inputs/{number:06d}"
    source = store_image(directory, prefix + "-native.png", rgb)
    cropped = rgb[checked_box(roi, rgb.shape)].copy()
    roi_image = store_image(directory, prefix + "-roi.png", cropped)
    lines, truncated, finder_arena = native.crops(cropped)
    if truncated:
        raise ValueError("finder truncated: pause and inspect; never count incomplete detections")
    entries = []
    for i, (crop, box, dark) in enumerate(lines):
        entry = store_image(directory, prefix + f"-line-{i:02d}.png", crop)
        entry.update(box=box, dark=dark)
        entries.append(entry)
    return {
        "kind": "capture",
        "split": "test",
        "session": session,
        "lock_sha256": lock_sha,
        "signature": signature,
        "completion": completion,
        "source": source,
        "roi": roi_image,
        "lines": entries,
        "finder_arena_bytes": finder_arena,
        "captured_at_unix": time.time(),
    }


def read_control(path):
    if path is None:
        return {"kind": "dialogue", "complete": False, "page": None}
    if not path.exists():
        return {"kind": "pause", "complete": False, "page": None}
    value = json.loads(path.read_text())
    if value.get("kind") not in ("dialogue", "choice", "menu", "pause", "transition"):
        raise ValueError("operator control has invalid screen kind")
    return value


def advance(args, env):
    subprocess.run(["xdotool", "windowfocus", args.window], env=env, check=True)
    subprocess.run(["xdotool", "keydown", args.advance_key], env=env, check=True)
    try:
        time.sleep(getattr(args, "key_hold_seconds", 0.3))
    finally:
        subprocess.run(["xdotool", "keyup", args.advance_key], env=env, check=True)


@contextmanager
def writer_guard(directory, name):
    """One collector and one comparator per session; controller remains independent."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"{name} already running for this session") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + "\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def collect(args):
    require_test(args.split)
    with writer_guard(args.session_dir, "collector"):
        _collect(args)


def _collect(args):
    require_test(args.split)  # before screenshot access
    if (
        args.count <= 0
        or args.interval <= 0
        or args.stable_samples < 2
        or args.stable_seconds <= 0
        or args.idle_timeout <= 0
        or getattr(args, "key_hold_seconds", 0.3) <= 0
    ):
        raise ValueError("positive count/interval/stability and >=2 observations required")
    args.session_dir.mkdir(parents=True, exist_ok=True)
    native = Native(args.session_dir / "runtime", args.model)
    lock = protocol(args, native)
    lock_sha = frozen_file(args.session_dir / "lock.json", lock)
    journal = args.session_dir / "captures.jsonl"
    old = read_journal(journal)
    for row in old:
        if row.get("kind") == "capture":
            verify_capture(args.session_dir, row, lock_sha)
    captures = [row for row in old if row.get("kind") == "capture"]
    tracker = Stability(args.stable_seconds, args.stable_samples, [row["signature"] for row in captures])
    used_pages = {row["completion"].get("page") for row in captures if row["completion"].get("page") is not None}
    env = dict(os.environ, DISPLAY=args.display)
    scratch = args.session_dir / "sampling.png"
    accepted, observations = 0, 0
    sample_counts = Counter()
    last_accept = time.monotonic()
    try:
        while accepted < args.count:
            if (args.session_dir / "STOP").exists():
                print("STOP requested; immutable captures preserved", flush=True)
                break
            if time.monotonic() - last_accept >= args.idle_timeout:
                append_event(
                    journal,
                    {
                        "kind": "pause",
                        "split": "test",
                        "reason": "no_new_completed_dialogue_timeout",
                        "counts": dict(sample_counts),
                    },
                )
                print(
                    "PAUSED: no new completed dialogue; controller must inspect transition/menu/choice/duplicate",
                    flush=True,
                )
                break
            control = read_control(args.control_file)
            if control["kind"] != "dialogue":
                time.sleep(args.interval)
                continue  # menus/choices/transitions need controller, never auto advance
            subprocess.run(["import", "-window", args.window, str(scratch)], env=env, check=True)
            with Image.open(scratch) as image:
                if list(image.size) != lock["native_size"]:
                    raise ValueError(f"expected native {lock['native_size']}, got {image.size}")
                rgb = np.asarray(image.convert("RGB")).copy()
            signature, ink_pixels = dialogue_signature(rgb, args.roi, lock["ignore_boxes"], args.ink_threshold)
            complete = (
                control.get("complete") is True
                and control.get("page") is not None
                and control["page"] not in used_pages
            )
            if args.complete_box is not None:
                grey = to_gray(rgb)
                indicator = grey[checked_box(args.complete_box, grey.shape)]
                complete = int((indicator >= args.complete_threshold).sum()) >= args.complete_pixels
            observations += 1
            sample_counts["samples"] += 1
            sample_counts["incomplete_observations"] += int(not complete)
            sample_counts["duplicate_signature_observations"] += int(signature in tracker.seen)
            ready = tracker.observe(signature, time.monotonic(), complete and ink_pixels > 0, control["kind"])
            if ready:
                roi = rgb[checked_box(args.roi, rgb.shape)]
                found, truncated, _ = native.crops(roi)
                if truncated or not found or len(found) > args.max_lines:
                    append_event(
                        journal,
                        {
                            "kind": "pause",
                            "split": "test",
                            "signature": signature,
                            "reason": "finder_empty_truncated_or_many_lines",
                            "count": len(found),
                        },
                    )
                    print("PAUSED: suspicious dialogue geometry; controller review required", flush=True)
                    break
                completion = {
                    "method": "indicator" if args.complete_box else "operator",
                    "page": control.get("page"),
                    "stable_observations": tracker.count,
                    "stable_seconds": time.monotonic() - tracker.since,
                    "kind": control["kind"],
                }
                event = capture_event(
                    args.session_dir, rgb, args.roi, native, signature, lock_sha, args.session, completion
                )
                append_event(journal, event)
                tracker.accept(signature)
                if control.get("page") is not None:
                    used_pages.add(control["page"])
                accepted += 1
                last_accept = time.monotonic()
                print(
                    json.dumps(
                        {
                            "accepted": accepted,
                            "total": len(captures) + accepted,
                            "lines": len(found),
                            "signature": signature,
                        }
                    ),
                    flush=True,
                )
                if args.advance_key:
                    advance(args, env)
            if observations % 40 == 0:
                append_event(
                    journal,
                    {
                        "kind": "observations",
                        "split": "test",
                        "counts": dict(sample_counts),
                        "last_signature": signature,
                        "last_complete": complete,
                    },
                )
                sample_counts.clear()
                print(json.dumps({"samples": observations, "accepted": accepted, "complete": complete}), flush=True)
            time.sleep(args.interval)
    finally:
        if sample_counts:
            append_event(journal, {"kind": "observations", "split": "test", "counts": dict(sample_counts)})


def verify_capture(directory, capture, lock_sha):
    require_test(capture.get("split"))
    if capture.get("lock_sha256") != lock_sha:
        raise ValueError("capture protocol hash mismatch")
    for entry in [capture["source"], capture["roi"], *capture["lines"]]:
        verified_image(directory, entry)


class Teacher:
    def __init__(self, python, directory):
        self.log = (directory / "google-worker.log").open("a")
        self.process = subprocess.Popen(
            [str(python), str(ROOT / "eval/owocr_bridge.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
        )
        self.number = 0

    def request(self, directory, entry):
        self.number += 1
        request = dict(entry, path=str((directory / entry["path"]).resolve()), id=self.number)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        raw = self.process.stdout.readline()
        if not raw:
            raise RuntimeError("Google worker exited without result")
        result = json.loads(raw)
        if result.get("id") != self.number:
            raise RuntimeError("Google worker response ID mismatch")
        return result

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.log.close()


def teacher_request(teacher, directory, entry):
    started = time.perf_counter()
    try:
        return teacher.request(directory, entry)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "host_seconds": time.perf_counter() - started}


def compare_capture(directory, capture, native, teacher):
    """Failures remain failures; line engines consume one identical stored L image."""
    results = []
    for entry in capture["lines"]:
        pixels = verified_image(directory, entry)
        started = time.perf_counter()
        try:
            text = native.text(pixels)
            vocr = {"ok": True, "text": text, "host_seconds": time.perf_counter() - started}
        except Exception as exc:
            vocr = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "host_seconds": time.perf_counter() - started}
        google = teacher_request(teacher, directory, entry)
        results.append(
            {
                "input_sha256": entry["sha256"],
                "pixels_sha256": entry["pixels_sha256"],
                "box": entry["box"],
                "dark": entry["dark"],
                "google": google,
                "vocr": vocr,
            }
        )
    whole = teacher_request(teacher, directory, capture["roi"])
    ok = (
        bool(results)
        and whole.get("ok") is True
        and bool(norm(whole.get("text")))
        and all(line["google"].get("ok") is True and line["vocr"].get("ok") is True for line in results)
    )
    return {
        "kind": "comparison",
        "split": "test",
        "capture_sha256": capture["event_sha256"],
        "compared_at_unix": time.time(),
        "lock_sha256": capture["lock_sha256"],
        "ok": ok,
        "lines": results,
        "whole_roi_google": whole,
    }


def aggregate(captures, comparisons):
    by_capture = {}
    for result in comparisons:
        require_test(result.get("split"))
        by_capture[result["capture_sha256"]] = result
    rows, roi_rows, finder_rows = [], [], []
    seen, counts, host = (
        set(),
        Counter(
            {
                key: 0
                for key in (
                    "captures",
                    "pending_captures",
                    "failed_captures",
                    "successful_unique_pages",
                    "successful_paired_lines",
                    "duplicate_dialogue_signatures",
                    "whole_roi_google_lines",
                    "line_google_failures",
                    "line_vocr_failures",
                    "roi_google_failures",
                    "empty_roi_reference",
                    "finder_empty_captures",
                )
            }
        ),
        {"vocr": [], "google_lines": [], "google_roi": []},
    )
    for capture in captures:
        require_test(capture.get("split"))
        if capture.get("kind") == "observations":
            for key, value in capture["counts"].items():
                counts["sampled_frames" if key == "samples" else key] += value
            continue
        if capture.get("kind") != "capture":
            continue
        counts["captures"] += 1
        result = by_capture.get(capture["event_sha256"])
        if result is None:
            counts["pending_captures"] += 1
            continue
        for line in result["lines"]:
            counts["line_google_failures"] += int(line["google"].get("ok") is not True)
            counts["line_vocr_failures"] += int(line["vocr"].get("ok") is not True)
        counts["roi_google_failures"] += int(result["whole_roi_google"].get("ok") is not True)
        counts["empty_roi_reference"] += int(
            result["whole_roi_google"].get("ok") is True and not norm(result["whole_roi_google"].get("text"))
        )
        counts["finder_empty_captures"] += int(not result["lines"])
        if not result["ok"]:
            counts["failed_captures"] += 1
            continue
        whole = result["whole_roi_google"]["text"]
        key = capture["signature"]
        if key in seen:
            counts["duplicate_dialogue_signatures"] += 1
            continue
        seen.add(key)
        counts["successful_unique_pages"] += 1
        counts["successful_paired_lines"] += len(result["lines"])
        google_text = "".join(line["google"]["text"] for line in result["lines"])
        vocr_text = "".join(line["vocr"]["text"] for line in result["lines"])
        for line in result["lines"]:
            rows.append({"text": line["google"]["text"], "pred": line["vocr"]["text"]})
            host["vocr"].append(line["vocr"]["host_seconds"])
            host["google_lines"].append(line["google"]["host_seconds"])
        roi_rows.append({"text": whole, "pred": vocr_text})
        finder_rows.append({"text": whole, "pred": google_text})
        host["google_roi"].append(result["whole_roi_google"]["host_seconds"])
        counts["whole_roi_google_lines"] += len(result["whole_roi_google"].get("lines", []))

    def score(values):
        chars = sum(len(norm(row["text"])) for row in values)
        errors = sum(lev(norm(row["text"]), norm(row["pred"])) for row in values)
        hira = score_hiragana(values)
        hira.pop("confusions")  # aggregate output has no source text/confusion characters
        hira["disagreement_rate"] = hira.pop("cer")
        return {
            "units": len(values),
            "reference_chars": chars,
            "edit_disagreements": errors,
            "disagreement_rate": errors / chars if chars else None,
            "hiragana": hira,
        }

    timing = {
        key: {
            "samples": len(values),
            "total_seconds": sum(values),
            "median_seconds": float(np.median(values)) if values else None,
            "p95_seconds": float(np.percentile(values, 95)) if values else None,
        }
        for key, values in host.items()
    }
    return {
        "schema": 1,
        "split": "test",
        "reference": "Google Lens automated proxy; not hand-checked ground truth/CER",
        "counts": dict(counts),
        "paired_line_proxy": score(rows),
        "whole_roi_vocr_proxy": score(roi_rows),
        "whole_roi_vs_google_finder_proxy": score(finder_rows),
        "target_chars": 40000,
        "target_met": sum(len(norm(row["text"])) for row in roi_rows) >= 40000,
        "timing": timing,
        "timing_scope": "host only; Google includes network latency; not console estimates",
        "dedup": "foreground dialogue-mask identity; never Google text, which could collapse distinct OCR failures",
    }


def write_summary(directory, summary):
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(directory / "summary.json")


def compare(args):
    validate_lock(args.session_dir)  # reject roles/changed protocol before side effects
    with writer_guard(args.session_dir, "comparator"):
        _compare(args)


def _compare(args):
    if args.poll_interval <= 0:
        raise ValueError("poll interval must be positive")
    directory = args.session_dir
    lock, lock_sha = validate_lock(directory)
    if teacher_info(args.owocr_python) != lock["teacher"]:
        raise ValueError("locked Google engine/dependency sources changed")
    native = Native(directory / "runtime", args.model)
    if native.info != lock["runtime"]:
        raise ValueError("locked native runtime changed")
    captures = read_journal(directory / "captures.jsonl")
    for capture in captures:
        if capture.get("kind") == "capture":
            verify_capture(directory, capture, lock_sha)
    result_path = directory / "comparisons.jsonl"
    comparisons = read_journal(result_path)
    known = {capture["event_sha256"] for capture in captures if capture.get("kind") == "capture"}
    for result in comparisons:
        require_test(result.get("split"))
        if result["capture_sha256"] not in known or result["lock_sha256"] != lock_sha:
            raise ValueError("comparison input/protocol mismatch")
    done = {row["capture_sha256"] for row in comparisons if row["ok"] or not args.retry_errors}
    teacher = Teacher(args.owocr_python, directory)
    try:
        while True:
            captures = read_journal(directory / "captures.jsonl")
            for capture in captures:
                if capture.get("kind") != "capture" or capture["event_sha256"] in done:
                    continue
                verify_capture(directory, capture, lock_sha)
                result = compare_capture(directory, capture, native, teacher)
                comparisons.append(append_event(result_path, result))
                done.add(capture["event_sha256"])
                summary = aggregate(captures, comparisons)
                summary.update(
                    lock_sha256=lock_sha,
                    model_sha256=BASELINE_SHA,
                    capture_journal_sha256=file_sha(directory / "captures.jsonl"),
                    comparison_journal_sha256=file_sha(result_path),
                    sources=lock["sources"],
                    runtime=lock["runtime"],
                    teacher_sha256=digest(lock["teacher"]),
                )
                write_summary(directory, summary)
                print(
                    json.dumps(
                        {
                            "captures": summary["counts"],
                            "reference_chars": summary["whole_roi_vocr_proxy"]["reference_chars"],
                        }
                    ),
                    flush=True,
                )
                if summary["target_met"] and args.stop_at_target:
                    return
            if not args.follow or (directory / "STOP").exists():
                break
            time.sleep(args.poll_interval)
    finally:
        teacher.close()
    summary = aggregate(captures, comparisons)
    summary.update(
        lock_sha256=lock_sha,
        model_sha256=BASELINE_SHA,
        capture_journal_sha256=file_sha(directory / "captures.jsonl")
        if (directory / "captures.jsonl").exists()
        else None,
        comparison_journal_sha256=file_sha(result_path) if result_path.exists() else None,
        sources=lock["sources"],
        runtime=lock["runtime"],
        teacher_sha256=digest(lock["teacher"]),
    )
    write_summary(directory, summary)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("collect")
    cap.add_argument("--session-dir", type=Path, required=True)
    cap.add_argument("--session", required=True)
    cap.add_argument("--split", choices=["test", "dev", "train"], required=True)
    cap.add_argument("--model", type=Path, required=True)
    cap.add_argument("--owocr-python", type=Path, required=True)
    cap.add_argument("--display", required=True)
    cap.add_argument("--window", required=True)
    cap.add_argument("--roi", type=rect, required=True)
    cap.add_argument("--native-size", type=native_size, default=[960, 544])
    cap.add_argument("--ignore-box", type=rect, action="append", default=[])
    cap.add_argument("--complete-box", type=rect)
    cap.add_argument("--complete-threshold", type=int, default=200)
    cap.add_argument("--complete-pixels", type=int, default=3)
    cap.add_argument("--control-file", type=Path)
    cap.add_argument("--ink-threshold", type=int, default=200)
    cap.add_argument("--count", type=int, default=1000, help="additional unique completed pages this invocation")
    cap.add_argument("--interval", type=float, default=0.25)
    cap.add_argument("--stable-seconds", type=float, default=2.0)
    cap.add_argument("--stable-samples", type=int, default=3)
    cap.add_argument("--max-lines", type=int, default=5)
    cap.add_argument(
        "--idle-timeout", type=float, default=120.0, help="pause for controller review after no new completed page"
    )
    cap.add_argument("--advance-key", help="optional authorized game advance after completed capture")
    cap.add_argument("--key-hold-seconds", type=float, default=0.3)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--session-dir", type=Path, required=True)
    cmp.add_argument("--model", type=Path, required=True)
    cmp.add_argument("--owocr-python", type=Path, required=True)
    cmp.add_argument("--follow", action="store_true")
    cmp.add_argument("--poll-interval", type=float, default=2.0)
    cmp.add_argument("--retry-errors", action="store_true")
    cmp.add_argument("--stop-at-target", action="store_true")
    args = ap.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()

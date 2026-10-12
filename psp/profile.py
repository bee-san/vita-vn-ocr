#!/usr/bin/env python3
"""Measure fixed PSP execution profiles on generated inputs, with host-logit checks.

This measures emulator arithmetic, allocation and timing, never OCR accuracy.
Models, packs and detailed predictions remain in the private output directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pack
import ppsspp_cli

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from pyvocr import Vocr, load_lib  # noqa: E402

PROFILES = (("scalar-w8i-t4", "scalar", "w8i", 4),
            ("scalar-w8f-t4", "scalar", "w8f", 4),
            ("vfpu-w8f-t4", "vfpu", "w8f", 4),
            ("vfpu-w8f-t16", "vfpu", "w8f", 16))
SHAPES = ((1, 1, 1), (16, 64, 1), (20, 128, 4), (24, 256, 1),
          (24, 480, 4), (32, 512, 1), (32, 1024, 4), (48, 640, 1),
          (64, 1920, 4), (32, 48, 1), (32, 64, 4), (24, 96, 1))
SEED = 20261012


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def generated_inputs():
    rng = np.random.default_rng(SEED)
    images = []
    for index, (height, width, channels) in enumerate(SHAPES):
        yy, xx = np.mgrid[:height, :width]
        ink = ((xx % 19 < 3) | (yy % 11 < 2)) & (xx % 23 < 17)
        pixels = np.where(ink, 220, 35).astype(np.int16)
        pixels += rng.integers(-12, 13, (height, width), dtype=np.int16)
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        if index >= len(SHAPES) - 2:
            pixels[:] = 255 if index % 2 else 0
        if channels == 4:
            pixels = np.stack((pixels, pixels, pixels, np.full_like(pixels, 255)), axis=2)
        images.append(pixels)
    return images


def checked_run(app, count, expected, mode, tile, invert):
    stats = json.loads((app / "stats.json").read_text())
    result = (app / "results.txt").read_text()
    if "# done:" not in result or "# ERROR" in result:
        raise ValueError("PSP app did not complete every generated case")
    ppsspp_cli.check_results(result, stats)
    if (stats["lines"] != count or stats["text_ok"] != count or stats["logits_ok"] != count
            or stats["expect"] is not True):
        raise ValueError("PSP text or float-logit CRC differs from explicit host reference")
    if ((stats["mode"], stats["tile"], stats["head"], stats["max_width"], stats["invert"],
         stats["cpu_mhz"], stats["repeat"], stats["stack_bytes"])
            != (mode, tile, 16, 1024, int(invert), 333, 3, 4096)):
        raise ValueError("PSP execution settings differ from fixed profile")
    if not 0 < stats["stack_used"] < stats["stack_bytes"]:
        raise ValueError("PSP worker has no measured stack headroom")
    if any(stats[key] != expected[key] for key in ("blob_bytes", "classes", "arena_bytes")):
        raise ValueError("PSP model/arena contract differs from host")
    rows = [line.split("\t") for line in result.splitlines() if line and not line.startswith("#")]
    if [int(row[0]) for row in rows] != list(range(count)) or any(row[4:6] != ["1", "1"] for row in rows):
        raise ValueError("missing, reordered or numerically mismatching PSP case")
    return stats, [{"case": int(row[0]), "frames": int(row[1]), "emulated_us": int(row[3])}
                   for row in rows]


def profile(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("use a new output directory; preserve earlier measurements")
    model = args.model.resolve()
    if sha(model) != args.model_sha256:
        raise ValueError("model differs from the declared SHA256")
    binaries = {name: Path(value).resolve() for name, value in (
        ("scalar", args.scalar_eboot), ("vfpu", args.vfpu_eboot),
        ("host_library", args.library), ("emulator", args.ppsspp))}
    pins = {name: sha(path) for name, path in binaries.items()}
    source_names = ("psp/profile.py", "psp/pack.py", "psp/ppsspp_cli.py", "psp/main.c",
                    "psp/Makefile", "runtime/vocr.c", "runtime/vocr.h", "runtime/pyvocr.py")
    sources = {name: sha(ROOT / name) for name in source_names}
    images = generated_inputs()
    out.mkdir(parents=True)
    pack.write_pack(out / "generated-lines.bin", images)
    protocol = {
        "schema": 1, "purpose": "generated PSP numerical parity and emulator profile; no accuracy claim",
        "model_sha256": args.model_sha256, "binary_sha256": pins, "source_sha256": sources,
        "generated_pack_sha256": sha(out / "generated-lines.bin"), "seed": SEED,
        "shapes_hwc": SHAPES, "profiles": PROFILES, "polarities": [False, True],
        "boots_per_profile_polarity": 2, "timed_repeats_after_first_check": 3,
        "cpu_mhz": 333, "cpu_core": "jit", "max_width": 1024, "head": 16, "stack_bytes": 4096,
        "host": {"platform": platform.platform(), "python": sys.version,
                 "numpy": importlib.metadata.version("numpy")},
    }
    write(out / "protocol.json", protocol)
    lib = load_lib(binaries["host_library"])
    measurements = []
    for name, build, mode, tile in PROFILES:
        reader = Vocr(model, lib=lib, mode=mode, tile=tile, head=16, max_width=1024, check_crc=True)
        memory = {"blob_bytes": reader.blob_bytes, "arena_bytes": reader.arena_bytes,
                  "classes": reader.n_classes}
        for invert in (False, True):
            expected = []
            for pixels in images:
                text, logits = reader.run(pixels, invert=invert, logits=True)
                expected.append((zlib.crc32(np.ascontiguousarray(logits, "<f4").tobytes()), text))
            for boot_index in range(2):
                app = out / f"{name}-inv{int(invert)}-boot{boot_index}"
                app.mkdir()
                shutil.copyfile(binaries[build], app / "EBOOT.PBP")
                shutil.copyfile(model, app / "model.vocr")
                shutil.copyfile(out / "generated-lines.bin", app / "lines.bin")
                pack.write_expect(app / "expect.txt", expected)
                config = pack.bundle_cfg(mode, invert, repeat=3, clock=333, tile=tile, head=16)
                config.update(wait=0, max_width=1024, stack_kb=4)
                (app / "vocr.cfg").write_text("".join(f"{key}={value}\n" for key, value in config.items()))
                options = SimpleNamespace(ppsspp=str(binaries["emulator"]), cpu="jit", timeout=120)
                _, log = ppsspp_cli.boot(options, app, app)
                (app / "emulator.log").write_text(log)
                stats, rows = checked_run(app, len(images), memory, mode, tile, invert)
                measurements.append({"profile": name, "invert": invert, "boot": boot_index,
                                     "stats": stats, "cases": rows})
                print(f"{name} invert={invert} boot={boot_index}: {len(rows)} numerical checks passed", flush=True)
        del reader
    if (sha(model) != args.model_sha256 or pins != {name: sha(path) for name, path in binaries.items()}
            or sources != {name: sha(ROOT / name) for name in source_names}):
        raise ValueError("model, build or source changed during profiling")
    summary = []
    for name, _, mode, tile in PROFILES:
        rows = [row for row in measurements if row["profile"] == name]
        stats = [row["stats"] for row in rows]
        frames = sum(s["frames"] for s in stats)
        total = sum(s["us_total"] for s in stats)
        times = [s["us_total"] / s["frames"] for s in stats]
        same_repeats = all(rows[i]["cases"] == rows[i + 1]["cases"] for i in (0, 2))
        summary.append({"profile": name, "mode": mode, "tile": tile, "boots": len(rows),
                        "numerical_checks": sum(s["lines"] for s in stats),
                        "blob_bytes": stats[0]["blob_bytes"], "arena_bytes": stats[0]["arena_bytes"],
                        "stack_used_max": max(s["stack_used"] for s in stats),
                        "stack_reserved_bytes": stats[0]["stack_bytes"],
                        "model_and_context_bytes": stats[0]["sizeof_model"] + stats[0]["sizeof_ctx"],
                        "emulated_us_per_frame": total / frames,
                        "emulated_us_per_frame_range": [min(times), max(times)],
                        "repeated_case_timings_identical": same_repeats})
    result = {"schema": 1, "status": "completed", "protocol": protocol,
              "protocol_sha256": sha(out / "protocol.json"), "profiles": summary,
              "measurements": measurements,
              "scope": ("generated numerical checks only; PPSSPP emulated allocation/timing, "
                        "not physical PSP or real-VN CER"),
              "accuracy_evaluated": False, "default_changed": False}
    write(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "scalar-eboot", "vfpu-eboot", "library", "ppsspp", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    profile(parser.parse_args())


if __name__ == "__main__":
    main()

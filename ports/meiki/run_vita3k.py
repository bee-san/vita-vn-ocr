#!/usr/bin/env python3
"""Run the manual test app in an isolated Vita3K installation.

All emulator files stay under --out; this never connects to a physical Vita.
Emulator timing and partition sizes are not physical-console measurements.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emulator", default="vita3k")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pref = out / "pref"
    log_dir = pref / "ux0/data/MeikiProbe"
    log_dir.mkdir(parents=True)
    (log_dir / "emulator-autoexit.flag").touch()
    config = out / "config.yml"
    config.write_text(
        "initial-setup: true\nbackend-renderer: OpenGL\nvalidation-layer: false\n"
        "discord-rich-presence: false\nshow-live-area-screen: false\n"
        "delay-start: 0\ndelay-background: 0\nlog-level: 2\n"
        f"pref-path: {json.dumps(str(pref) + '/')}\n"
    )
    env = dict(os.environ)
    env.update(
        XDG_CONFIG_HOME=str(out / "config"),
        XDG_DATA_HOME=str(out / "data"),
        XDG_CACHE_HOME=str(out / "cache"),
        SDL_VIDEODRIVER="x11",
    )
    command = [
        "timeout",
        str(args.timeout),
        "xvfb-run",
        "-a",
        "-s",
        "-screen 0 1280x720x24",
        args.emulator,
        "--config-location",
        str(config),
        "--load-config",
        "--frontend",
        "sdl",
        str(args.vpk.resolve()),
    ]
    started = time.monotonic()
    with (out / "stdout.log").open("w") as log:
        result = subprocess.run(command, env=env, cwd=out, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    results = log_dir / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    terminal = rows[-1] if rows else None
    summary = {
        "kind": "Vita3K; not physical Vita memory or speed",
        "command": command,
        "exit_code": result.returncode,
        "elapsed_s": time.monotonic() - started,
        "terminal": terminal,
    }
    if rows:
        cases = [row for row in rows if "index" in row]
        summary["peak_model_tensor_bytes"] = max((row["heap_peak"] for row in cases), default=0)
        summary["all_arenas_released"] = all(row["heap_remaining"] == 0 for row in cases)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not terminal or terminal.get("status") != "complete" or terminal.get("failed") or result.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

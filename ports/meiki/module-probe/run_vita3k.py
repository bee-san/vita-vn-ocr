#!/usr/bin/env python3
"""Run the isolated native module loader probe locally, without device access."""

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
    logs = pref / "ux0/data/MeikiModuleProbe"
    logs.mkdir(parents=True)
    (logs / "emulator-autoexit.flag").touch()
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
    with (out / "stdout.log").open("w") as output:
        run = subprocess.run(command, env=env, cwd=out, stdin=subprocess.DEVNULL, stdout=output, stderr=output)
    result_file = logs / "results.jsonl"
    rows = [json.loads(line) for line in result_file.read_text().splitlines()] if result_file.exists() else []
    summary = {
        "kind": "Vita3K native module loader; not physical Shell memory or console speed",
        "command": command,
        "exit_code": run.returncode,
        "elapsed_s": time.monotonic() - started,
        "terminal": rows[-1] if rows else None,
    }
    if rows:
        cases = [row for row in rows if "index" in row or row.get("status") in ("detector", "detected_line")]
        summary["all_arenas_released"] = all(row["heap_remaining"] == 0 for row in cases)
        summary["peak_model_tensor_bytes"] = max((row["heap_peak"] for row in cases), default=0)
        summary["peak_metadata_brk_bytes"] = max((row["metadata_brk_peak"] for row in cases), default=0)
        summary["stops"] = [row for row in rows if row.get("status") == "unloaded"]
        summary["guard_checks"] = [
            row for row in rows if row.get("status") in ("guard_checks", "busy_checks", "detector_guards")
        ]
        summary["detector"] = [row for row in rows if row.get("status") == "detector_summary"]
        summary["faults"] = [row for row in rows if row.get("status") == "detector_oom"]
        summary["all_buffers_released"] = all(
            not row["workspace_rc"] and not row["metadata_rc"] for row in rows if row.get("status") == "buffers_freed"
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    terminal = summary["terminal"]
    if not terminal or terminal.get("status") != "complete" or terminal.get("failed") or run.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

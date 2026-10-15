"""The machine a result was produced on, as result files record it.

host_label() is a generic description: "<CPUs>-core <CPU model> host", e.g. "64-core AMD EPYC 9R14
host" on the build host (os.cpu_count() logical CPUs; that host runs one thread per core), or
"build-host" when the CPU model is unknown. It is never the machine's network name (its node name,
socket host name or FQDN), which can name a private machine: this repository is public.
tests/test_repo_policy.py rejects code that reads that name, and internal host names, in every
tracked file (docs/build-host.md).
"""
import os

FALLBACK = "build-host"


def cpu_model(cpuinfo="/proc/cpuinfo"):
    """The CPU's model name from /proc/cpuinfo (e.g. "AMD EPYC 9R14"), or "" if there is none."""
    try:
        with open(cpuinfo, encoding="utf-8", errors="replace") as f:
            for line in f:
                key, sep, value = line.partition(":")
                if sep and key.strip() == "model name":
                    return " ".join(value.split())
    except OSError:
        pass
    return ""


def host_label(cpuinfo="/proc/cpuinfo"):
    """What a result file's host field holds: the CPU count and model, never the machine's name."""
    model, cpus = cpu_model(cpuinfo), os.cpu_count()
    return f"{cpus}-core {model} host" if model and cpus else FALLBACK

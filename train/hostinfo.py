"""The machine a result was produced on, as result files record it.

host_label() is a generic description: "<cores>-core <CPU model> host", e.g. "64-core AMD EPYC 9R14
host" on the build host, or "build-host" when the CPU model is unknown. Cores are physical cores
(distinct physical id / core id pairs in /proc/cpuinfo; os.cpu_count()'s logical CPUs where the
kernel lists no core ids). It is never the machine's network name (its node name, socket host name
or FQDN), which can name a private machine: this repository is public. tests/test_repo_policy.py
rejects code that reads that name, and internal host names, in every tracked file (docs/build-host.md).
"""
import os

FALLBACK = "build-host"


def cpu_info(cpuinfo="/proc/cpuinfo"):
    """(CPU model name, physical cores) from /proc/cpuinfo: e.g. ("AMD EPYC 9R14", 64); ("", 0) if unknown."""
    model, socket_id, cores = "", None, set()
    try:
        with open(cpuinfo, encoding="utf-8", errors="replace") as f:
            for line in f:
                key, sep, value = line.partition(":")
                key, value = key.strip(), " ".join(value.split())
                if not sep:
                    continue
                if key == "model name" and not model:
                    model = value
                elif key == "physical id":
                    socket_id = value
                elif key == "core id":
                    cores.add((socket_id, value))
    except OSError:
        pass
    return model, len(cores)


def host_label(cpuinfo="/proc/cpuinfo"):
    """What a result file's host field holds: the core count and CPU model, never the machine's name."""
    model, cores = cpu_info(cpuinfo)
    cores = cores or os.cpu_count()
    return f"{cores}-core {model} host" if model and cores else FALLBACK

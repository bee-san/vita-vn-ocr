#!/usr/bin/env python3
"""Worst-case stack depth of each entry point, from gcc's -fcallgraph-info=su output (.ci).

  stack.py vocr.ci [--json]

Every function's frame comes from the .ci labels ("N bytes (static)"); the deepest call chain
below each public vocr_* function is the sum along the heaviest path. Calls to functions
outside the file (memcpy, memset, libgcc's soft-float helpers) are listed with their callers:
their own frames are not in the .ci file, so the totals are lower bounds by those frames.
The PSP test app measures the real total, callees included, with
sceKernelGetThreadStackFreeSize() (psp/main.c).
"""
import argparse
import json
import re
import sys
from functools import lru_cache


def parse(text):
    nodes, edges = {}, {}
    for m in re.finditer(r'node:\s*\{\s*title:\s*"([^"]+)"\s*label:\s*"([^"]*)"', text):
        title, label = m.group(1), m.group(2)
        sz = re.search(r"(\d+) bytes \((static|dynamic[^)]*)\)", label)
        nodes[title] = (int(sz.group(1)), sz.group(2)) if sz else None
    for m in re.finditer(r'edge:\s*\{\s*sourcename:\s*"([^"]+)"\s*targetname:\s*"([^"]+)"', text):
        edges.setdefault(m.group(1), []).append(m.group(2))
    return nodes, edges


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ci")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    nodes, edges = parse(open(args.ci, encoding="utf-8").read())
    name = {t: t.split(":")[-1] for t in nodes}

    @lru_cache(maxsize=None)
    def deepest(t, depth=0):
        if depth > 64:
            raise SystemExit(f"recursion through {t}")
        own = nodes.get(t)
        if own is None:
            return 0, (name.get(t, t) + " (extern)",)
        best, path = 0, ()
        for c in edges.get(t, []):
            s, p = deepest(c, depth + 1)
            if s > best or not path:
                best, path = s, p
        return own[0] + best, (f"{name[t]} {own[0]}",) + path

    dynamic = sorted(name[t] for t, v in nodes.items() if v and v[1] != "static")
    entries = sorted(t for t, v in nodes.items() if v and name[t].startswith("vocr_"))
    out = {}
    for t in entries:
        total, path = deepest(t)
        out[name[t]] = {"bytes": total, "chain": list(path)}
    externs = sorted({name.get(c, c) for t in nodes for c in edges.get(t, []) if nodes.get(c) is None})
    if args.json:
        json.dump({"entries": out, "externs": externs, "dynamic": dynamic}, sys.stdout, indent=1)
        print()
        return
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"{k:22s} {v['bytes']:6d} B  " + " -> ".join(v["chain"]))
    print("extern callees (frames not counted):", " ".join(externs))
    if dynamic:
        print("dynamic frames:", " ".join(dynamic))


if __name__ == "__main__":
    main()

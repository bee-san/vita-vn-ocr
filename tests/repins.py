"""eval/runtime_repins.json for the tests of frozen pins: the sha256 a pinned runtime file may have now."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "eval" / "runtime_repins.json"


def reachable(path, pinned):
    """{pinned} and every sha256 a chain of registered re-pins of `path` reaches from it."""
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))["repins"]
    seen, todo = {pinned}, [pinned]
    while todo:
        h = todo.pop()
        for r in reg:
            if r["file"] == path and r["from"] == h and r["to"] not in seen:
                seen.add(r["to"])
                todo.append(r["to"])
    return seen

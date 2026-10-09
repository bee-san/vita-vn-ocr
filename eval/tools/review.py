#!/usr/bin/env python3
"""Record hand-check decisions for candidate lines (build tool).

  review.py init vita|psp                 data/eval/<p>/review.jsonl from lines_cand.jsonl
                                          (keeps decisions already made for the same key)
  review.py apply vita|psp FILE           apply a decisions file (or - for stdin)
  review.py status vita|psp               counts per status and per VN

Decision lines ("#" starts a comment), IDX = row index shown on the line sheets:
  IDX a|b|c [+tag,..] [-tag,..] [@x,y,w,h]   ground truth = that reading
  IDX = TEXT [+tags] [@box]                  ground truth typed by hand
  IDX ? [TEXT]                               uncertain: kept as don't-care, never scored
  IDX x                                      not a dialogue line: dropped
  IDX +tag,..  /  IDX -tag,..  /  IDX @box   change tags or box only
  S SHOT +tag,.. [-tag,..]                   tags for every line of a screenshot
  A SHOT @x,y,w,h TEXT [+tags]               add a line the detector missed or merged away
The text is stored in the training-label convention (data/label_real.py canon()).
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fetch import data_dir  # noqa: E402

ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("label_real", ROOT / "data" / "label_real.py")


def canon(s):
    if not hasattr(canon, "f"):
        sys.path.insert(0, str(ROOT / "synth"))
        mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(mod)
        canon.f = mod.canon
    return canon.f(s)


def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def init(which):
    d = data_dir() / which
    cands = load(d / "lines_cand.jsonl")
    old = {r["key"]: r for r in load(d / "review.jsonl")} if (d / "review.jsonl").exists() else {}
    rows = []
    for i, c in enumerate(cands):
        r = dict(c, idx=i, status=None, text=None, tags_add=[], tags_del=[])
        o = old.get(c["key"])
        if o:
            for k in ("status", "text", "tags_add", "tags_del", "box_override", "region_override", "note"):
                if k in o:
                    r[k] = o[k]
            if o.get("status") is not None:            # keep the box that was checked by eye
                for k in ("box", "hint", "furigana", "furigana_text", "a", "b", "c"):
                    r[k] = o[k]
        rows.append(r)
    for o in old.values():                       # hand-added lines (op A) survive a re-init
        if "#m" in o["key"]:
            rows.append(dict(o, idx=len(rows)))
    save(d / "review.jsonl", rows)
    print(f"{len(rows)} candidate lines, {sum(r['status'] is not None for r in rows)} already decided")


def add_manual(rows, parts, raw):
    """A SHOT @x,y,w,h TEXT [+tags]: a line the detector missed or merged away."""
    shot, box = parts[1], [int(v) for v in parts[2].lstrip("@").split(",")]
    body = [t for t in parts[3:] if not TAGS.match(t)]
    tags = [t for t in parts[3:] if TAGS.match(t)]
    n = sum(1 for r in rows if r["key"].startswith(shot + "#m"))
    ref = next((r for r in rows if r["shot"] == shot), None)
    row = {"key": f"{shot}#m{n}", "shot": shot, "vn": shot.split("_")[0],
           "dims": ref["dims"] if ref else None, "box": box, "det": None, "a": "", "b": "", "c": "",
           "prop": "", "agree": 0, "hint": {}, "furigana": [], "furigana_text": [], "p": [],
           "idx": len(rows), "status": "ok", "text": canon("".join(body)), "tags_add": [], "tags_del": [],
           "note": "added by hand"}
    for tok in tags:
        m = TAGS.match(tok)
        (row["tags_add"] if m.group(1) == "+" else row["tags_del"]).extend(m.group(2).split(","))
    if not body:
        raise ValueError(f"manual line without text: {raw}")
    rows.append(row)


TAGS = re.compile(r"^([+-])([\w,-]+)$")


def apply(which, src):
    d = data_dir() / which
    rows = load(d / "review.jsonl")
    by_shot = {}
    for r in rows:
        by_shot.setdefault(r["shot"], []).append(r)
    text = sys.stdin.read() if src == "-" else Path(src).read_text(encoding="utf-8")
    n = 0
    for raw in text.splitlines():
        line = raw.split("  #")[0].rstrip() if "  #" in raw else raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(" ")
        if parts[0] == "A":
            add_manual(rows, parts, raw)
            n += 1
            continue
        if parts[0] == "S":
            shot = parts[1]
            targets = by_shot[shot] if shot in by_shot else by_shot[sorted(by_shot)[int(shot)]]
            for tok in parts[2:]:
                m = TAGS.match(tok)
                if not m:
                    raise ValueError(f"bad shot op: {raw}")
                for r in targets:
                    key = "tags_add" if m.group(1) == "+" else "tags_del"
                    other = "tags_del" if key == "tags_add" else "tags_add"
                    for t in m.group(2).split(","):
                        if t not in r[key]:
                            r[key].append(t)
                        if t in r[other]:
                            r[other].remove(t)
            n += 1
            continue
        r = rows[int(parts[0])]
        op = parts[1] if len(parts) > 1 else ""
        rest = parts[2:]
        if op in ("a", "b", "c"):
            r["status"], r["text"] = "ok", canon(r[op])
        elif op == "=":
            body = [t for t in rest if not (TAGS.match(t) or t.startswith("@"))]
            rest = [t for t in rest if TAGS.match(t) or t.startswith("@")]
            r["status"], r["text"] = "ok", canon(" ".join(body).replace(" ", ""))
        elif op == "?":
            body = [t for t in rest if not (TAGS.match(t) or t.startswith("@"))]
            rest = [t for t in rest if TAGS.match(t) or t.startswith("@")]
            r["status"], r["text"] = "uncertain", canon("".join(body)) if body else None
        elif op == "x":
            r["status"], r["text"] = "drop", None
        else:
            rest = parts[1:]
        for tok in rest:
            m = TAGS.match(tok)
            if m:
                key = "tags_add" if m.group(1) == "+" else "tags_del"
                other = "tags_del" if key == "tags_add" else "tags_add"
                for t in m.group(2).split(","):
                    if t not in r[key]:
                        r[key].append(t)
                    if t in r[other]:
                        r[other].remove(t)
            elif tok.startswith("@"):
                r["box_override"] = [int(v) for v in tok[1:].split(",")]
            elif tok:
                raise ValueError(f"bad token {tok!r} in: {raw}")
        n += 1
    save(d / "review.jsonl", rows)
    print(f"applied {n} decisions")
    status(which)


def status(which):
    rows = load(data_dir() / which / "review.jsonl")
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "ok"]
    vns = Counter(r["vn"] for r in ok)
    print(f"status: {dict(c)}; ok lines {len(ok)} from {len(vns)} VNs; chars {sum(len(r['text']) for r in ok)}")
    print("per VN:", dict(sorted(vns.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    cmd, which = sys.argv[1], sys.argv[2]
    {"init": lambda: init(which), "apply": lambda: apply(which, sys.argv[3]),
     "status": lambda: status(which)}[cmd]()

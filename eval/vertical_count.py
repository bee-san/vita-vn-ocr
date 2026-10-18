#!/usr/bin/env python3
"""Census of vertical text in every stored real-screenshot read (issue #99, docs/vertical.md).

  vertical_count.py OUT.json     one row per screenshot; totals per platform and origin on stdout

Per screenshot, every stored read is merged: MeikiOCR's detector boxes with their vertical flag (devnative/read.py's
reads: devnative, native-frames, realplus v3 and its reclaim, the owner pools) and Chrome Screen AI's lines with
theirs (the teacher's reads, whose label pipeline drops vertical lines before it votes). A vertical line is a box
flagged vertical, at least twice as tall as wide, holding at least 3 kana or kanji; a screenshot's vertical lines are
those of the read that finds the most; a "vertical page" has at least two of at least 6 characters. Platforms are
the VNDB release's (psv, psp, else pc), the owner's frames by their console. Inputs live under $HERD/shared
(artifacts and the private owner pools); nothing here is committed but the aggregate record."""
import collections
import json
import os
import os as _os
import re
import sys

HERD = _os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")
A = f"{HERD}/shared/artifacts"
P = f"{HERD}/shared/private/owner-games"
D = f"{A}/realplus/vndb/dump/db"
SRC = [
    ("teacher-sai", [f"{A}/teacher/reads/train.{i}of5.jsonl" for i in range(5)]),
    ("devnative", [f"{A}/devnative/reads/reads.jsonl"]),
    ("native-frames", [f"{A}/native-frames/pool/reads/reads.jsonl"]),
    ("realplus", [f"{A}/realplus-v3/reads/reads.jsonl"]),
    ("realplus-reclaim", [f"{A}/realplus-v3/reclaim/reads/reads.jsonl"]),
    ("owner-1", [f"{P}/_build/owner-pool/reads/reads.jsonl"]),
    ("owner-2", [f"{P}/_build2/owner-pool-2/reads/reads.jsonl"]),
]
JP = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u3005]")


def rows(name):
    with open(f"{D}/{name}", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def dets_of(d):
    if "dets" in d:
        for x in d["dets"]:
            yield x["det"], bool(x.get("vertical")), x.get("text", "")
    else:
        for x in d.get("sai", []):
            yield x["box"], bool(x.get("vertical")), x.get("text", "")


def vlines(d):
    out = []
    for box, vert, text in dets_of(d):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        n = len(JP.findall(text))
        if vert and h >= 2 * max(w, 1) and n >= 3:
            out.append([n, [int(v) for v in box], text])
    return out


def main(out_path):
    relp = collections.defaultdict(set)
    for r in rows("releases_platforms"):
        relp[r[0]].add(r[1])
    shots = {}
    for src, files in SRC:
        for f in files:
            for line in open(f):
                d = json.loads(line)
                owner = src.startswith("owner")
                sid = ("owner:" + d["url"]) if owner else d["shot"]
                rel = d.get("release")
                ps = relp.get(rel, set()) if rel else set()
                if owner:
                    plat = "psv" if d["kind"] == "vita" else "psp"
                else:
                    plat = "psv" if "psv" in ps else "psp" if "psp" in ps else "pc"
                s = shots.setdefault(sid, {"shot": d["shot"], "vn": d["vn"], "plat": plat, "release": rel,
                                           "origin": "owner" if owner else "vndb", "url": d.get("url"),
                                           "path": None, "dims": d.get("dims"), "reads": {}})
                if d.get("path") and os.path.exists(d["path"]):
                    s["path"] = d["path"]
                s["reads"][src] = vlines(d)
    for s in shots.values():
        best = max(s["reads"].items(), key=lambda kv: (len(kv[1]), sum(n for n, _, _ in kv[1])))
        s["vsrc"], s["vl"] = best
        s["vpage"] = sum(1 for n, _, _ in s["vl"] if n >= 6) >= 2
        if s["path"] is None and s["origin"] == "vndb" and s["plat"] in ("psv", "psp"):
            k = "vita" if s["plat"] == "psv" else "psp"
            c = f"{A}/realdata/shots/{k}/{s['shot']}.jpg"
            s["path"] = c if os.path.exists(c) else None
        del s["reads"]
    json.dump(list(shots.values()), open(out_path, "w"), ensure_ascii=False)
    tot = collections.Counter()
    vns = collections.defaultdict(set)
    for s in shots.values():
        k = (s["plat"], s["origin"])
        tot[k + ("shots",)] += 1
        tot[k + ("shots_v",)] += bool(s["vl"])
        tot[k + ("vpages",)] += s["vpage"]
        tot[k + ("vlines",)] += len(s["vl"])
        vns[k].add(s["vn"])
        if s["vpage"]:
            vns[k + ("v",)].add(s["vn"])
    for k in sorted(tot):
        print(k, tot[k])
    for k in sorted(vns):
        print(k, "VNs", len(vns[k]))


if __name__ == "__main__":
    main(sys.argv[1])

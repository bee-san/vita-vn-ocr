#!/usr/bin/env python3
"""Fetch the VNDB relations of the candidate test VNs (build tool).

  relations.py [VN ...]      default: every VN in data/eval/{vita,psp}/shots.jsonl

Writes data/eval/vndb/relations.json: {vn: [{"id", "relation", "official", "title"}]}.
Sequels, fandiscs, alternative versions and series entries often reuse a
game's scenes, UI and fonts, so manifest.py reserves a test game's direct
relations along with it. Same rules as vndb_list.py: the public kana API,
one request per second, with a User-Agent.
"""
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fetch import data_dir  # noqa: E402
from vndb_list import API, UA  # noqa: E402


def fetch(ids):
    s = requests.Session()
    s.headers["User-Agent"] = UA
    out = {}
    for k in range(0, len(ids), 100):
        chunk = ids[k:k + 100]
        body = {"filters": ["or", *[["id", "=", v] for v in chunk]], "results": 100,
                "fields": "id,relations.id,relations.relation,relations.relation_official,relations.title"}
        while True:
            r = s.post(API, json=body, timeout=120)
            if r.status_code != 429:
                break
            time.sleep(60)
        r.raise_for_status()
        for vn in r.json()["results"]:
            out[vn["id"]] = [{"id": x["id"], "relation": x["relation"], "official": x["relation_official"],
                              "title": x["title"]} for x in vn.get("relations") or []]
        time.sleep(1)
    return out


def main():
    data = data_dir()
    ids = sys.argv[1:]
    if not ids:
        ids = sorted({json.loads(x)["vn"] for p in ("vita", "psp") if (data / p / "shots.jsonl").exists()
                      for x in (data / p / "shots.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()},
                     key=lambda v: int(v[1:]))
    rel = fetch(ids)
    path = data / "vndb" / "relations.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(rel)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rel)} VNs, {sum(len(v) for v in rel.values())} relations -> {path}")


if __name__ == "__main__":
    main()

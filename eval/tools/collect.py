#!/usr/bin/env python3
"""Gather the candidate screenshots of the reserved test games (build tool).

  collect.py vita    copy the reserved Vita games' 960x544 shots from data/vndb/shots
                     (downloading any that are missing) into data/eval/vita/shots
  collect.py psp     download the reserved PSP games' 480x272 shots into data/eval/psp/shots

Both keep only screenshots attached to a release on that platform at its native
size, rated sexual < 0.3 and violence < 0.5 (the repo's existing filters), and
write data/eval/<platform>/shots.jsonl (VN, title, URL, sha256, release languages).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fetch import UA, data_dir, download, sha256, shot_name  # noqa: E402

PLATFORMS = {"vita": ("psv", [960, 544], "vita_hash", "vita_extra"),
             "psp": ("psp", [480, 272], "psp_hash", None)}


def main(which):
    data = data_dir()
    plat, dims, k1, k2 = PLATFORMS[which]
    games = json.loads((data / "test_games.json").read_text(encoding="utf-8"))
    vns = set(games[k1]) | (set(games[k2]) if k2 else set())
    listing = data / "vndb" / f"vndb_{plat}_any.jsonl"
    listing = listing if listing.exists() else data / "vndb" / f"vndb_{plat}.jsonl"
    rows = [json.loads(x) for x in listing.read_text(
        encoding="utf-8").splitlines()]
    keep = [x for x in rows if x["vn"] in vns and plat in x["platforms"] and x["dims"] == dims
            and x["sexual"] < 0.3 and x["violence"] < 0.5]
    seen, uniq = set(), []
    for x in keep:
        if x["url"] not in seen:
            seen.add(x["url"])
            uniq.append(x)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    old = data.parent / "vndb" / "shots"
    out = []
    for i, x in enumerate(uniq):
        name = shot_name(x["vn"], x["url"])
        dest = data / which / "shots" / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if (old / name).exists():
                shutil.copyfile(old / name, dest)
            else:
                if not download(s, x["url"], dest):
                    print(f"FAILED {x['url']}", flush=True)
                    continue
                time.sleep(0.5)
        out.append({"shot": name, "vn": x["vn"], "title": x["title"], "alttitle": x.get("alttitle"),
                    "url": x["url"], "sha256": sha256(dest), "dims": x["dims"], "langs": x["langs"],
                    "release": x["release"], "sexual": x["sexual"], "violence": x["violence"]})
        if i % 50 == 0:
            print(f"{i}/{len(uniq)}", flush=True)
    (data / which / "shots.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{which}: {len(out)} shots of {len({r['vn'] for r in out})} VNs", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

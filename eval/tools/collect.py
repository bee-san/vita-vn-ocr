#!/usr/bin/env python3
"""Gather the candidate screenshots of the reserved test games (build tool).

  collect.py vita    copy the candidate Vita games' 960x544 shots from data/vndb/shots
                     (downloading any that are missing) into data/eval/vita/shots
  collect.py psp     download the candidate PSP games' 480x272 shots into data/eval/psp/shots

The candidate games are every synth/common.py is_test_vn() game in the Japanese-original
listing (data/eval/vndb/vndb_<platform>.jsonl from vndb_list.py) plus EXTRA, the games
added by hand to reach 15+ Vita games (the any-language listing is searched for their
screenshots). Both keep only screenshots attached to a release on that platform at its
native size, rated sexual < 0.3 and violence < 0.5 (the repo's existing filters), and
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
sys.path.insert(0, str(HERE.parents[1] / "synth"))
from fetch import UA, data_dir, download, sha256, shot_name  # noqa: E402

from common import is_test_vn  # noqa: E402

PLATFORMS = {"vita": ("psv", [960, 544]), "psp": ("psp", [480, 272])}
# Not is_test_vn() games, reviewed as extra vita-test candidates; only v16566 and v19850
# kept lines (v28755's ja-tagged shots show English text).
EXTRA = {"vita": ["v1715", "v6618", "v7679", "v11179", "v12402", "v12424", "v14908", "v15061", "v15253",
                  "v16566", "v18027", "v19238", "v19850", "v19913", "v20727", "v28755"],
         "psp": []}


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def candidates(which, data):
    """Listing rows of the candidate screenshots, one per URL, in listing order."""
    plat, dims = PLATFORMS[which]
    vns = {x["vn"] for x in load_jsonl(data / "vndb" / f"vndb_{plat}.jsonl") if is_test_vn(x["vn"])}
    vns |= set(EXTRA[which])
    listing = data / "vndb" / f"vndb_{plat}_any.jsonl"
    rows = load_jsonl(listing if listing.exists() else data / "vndb" / f"vndb_{plat}.jsonl")
    keep = [x for x in rows if x["vn"] in vns and plat in x["platforms"] and x["dims"] == dims
            and x["sexual"] < 0.3 and x["violence"] < 0.5]
    seen, uniq = set(), []
    for x in keep:
        if x["url"] not in seen:
            seen.add(x["url"])
            uniq.append(x)
    return uniq


def main(which):
    data = data_dir()
    uniq = candidates(which, data)
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

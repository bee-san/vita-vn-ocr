#!/usr/bin/env python3
"""List VNDB screenshots of Japanese-original VNs released on one platform.

  vndb_list.py --platform psp --out data/eval/vndb_psp.jsonl

One JSON row per screenshot (VN id and title, URL, dims, sexual/violence
ratings, the release it belongs to with its platforms and languages). Only the
public kana API is used, one request per second with a User-Agent. This is a
build tool for the test sets; run.py and fetch.py never call it.
"""
import argparse
import json
import time
from pathlib import Path

import requests

UA = "vita-vn-ocr-eval/0.1 (benchmark manifests; github.com/bee-san/vita-vn-ocr)"
API = "https://api.vndb.org/kana/vn"
FIELDS = ("id,title,alttitle,screenshots.url,screenshots.dims,screenshots.sexual,"
          "screenshots.violence,screenshots.release.id,screenshots.release.platforms,"
          "screenshots.release.languages.lang")


def fetch(platform, olang="ja"):
    s = requests.Session()
    s.headers["User-Agent"] = UA
    rows, page = [], 1
    rel = ["release", "=", ["platform", "=", platform]]
    filters = ["and", ["olang", "=", olang], rel] if olang != "any" else rel
    while True:
        body = {"filters": filters,
                "fields": FIELDS, "results": 100, "page": page, "sort": "id"}
        r = s.post(API, json=body, timeout=120)
        if r.status_code == 429:
            time.sleep(60)
            continue
        r.raise_for_status()
        d = r.json()
        for vn in d["results"]:
            for sh in vn.get("screenshots") or []:
                rel = sh.get("release") or {}
                rows.append({"vn": vn["id"], "title": vn["title"], "alttitle": vn.get("alttitle"),
                             "url": sh["url"], "dims": sh["dims"], "sexual": sh["sexual"],
                             "violence": sh["violence"], "release": rel.get("id"),
                             "platforms": rel.get("platforms") or [],
                             "langs": [x["lang"] for x in rel.get("languages") or []]})
        print(f"page {page}: {len(d['results'])} VNs, {len(rows)} screenshots", flush=True)
        if not d.get("more"):
            return rows
        page += 1
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--olang", default="ja", help="original language filter, or 'any'")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = fetch(args.platform, args.olang)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    tmp.rename(args.out)
    print(f"{len(rows)} screenshots of {len({x['vn'] for x in rows})} VNs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

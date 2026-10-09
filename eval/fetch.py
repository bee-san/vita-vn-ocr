#!/usr/bin/env python3
"""Rebuild the eval screenshots from VNDB and check them against the manifests.

  eval/fetch.py [--set vita-test|psp-test|all] [--data DIR] [--check]

The manifests (eval/manifests/<set>.jsonl) hold every screenshot's VNDB URL and
sha256. Images go to DIR/<platform>/shots/<vn>_<file>.jpg, where DIR is --data,
else $VOCR_EVAL_DATA, else <repo>/data/eval. Files already present with the right
hash are skipped, so a killed run can be restarted. --check only verifies.
Exit status is non-zero if any image is missing or has the wrong hash.

The screenshots are copyrighted game images: keep them local, never commit them.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UA = "vita-vn-ocr-eval/0.1 (benchmark rebuild; github.com/bee-san/vita-vn-ocr)"
SETS = {"vita-test": "vita", "psp-test": "psp"}


def data_dir(arg=None):
    return Path(arg or os.environ.get("VOCR_EVAL_DATA") or ROOT / "data" / "eval")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def shot_name(vn, url):
    return f"{vn}_{url.rsplit('/', 1)[-1]}"


def shot_path(data, platform, vn, url):
    return Path(data) / platform / "shots" / shot_name(vn, url)


def download(session, url, dest, tries=5):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return True
        except Exception as e:  # network errors: retry with backoff
            print(f"retry {attempt + 1} {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return False


def shots_of(set_name):
    rows = [json.loads(x) for x in (HERE / "manifests" / f"{set_name}.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    seen = {}
    for r in rows:
        seen.setdefault(r["url"], (r["vn"], r["sha256"]))
    return [(url, vn, sha) for url, (vn, sha) in seen.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="all", choices=["all", *SETS])
    ap.add_argument("--data", default=None)
    ap.add_argument("--check", action="store_true", help="verify only, download nothing")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between downloads")
    args = ap.parse_args()
    data = data_dir(args.data)
    bad = 0
    session = None
    for set_name in (SETS if args.set == "all" else [args.set]):
        platform = SETS[set_name]
        shots = shots_of(set_name)
        got = ok = 0
        for url, vn, want in shots:
            dest = shot_path(data, platform, vn, url)
            if dest.exists() and sha256(dest) == want:
                ok += 1
                continue
            if args.check:
                print(f"{'WRONG HASH' if dest.exists() else 'MISSING'} {dest}", flush=True)
                bad += 1
                continue
            if session is None:
                import requests
                session = requests.Session()
                session.headers["User-Agent"] = UA
            if not download(session, url, dest):
                print(f"FAILED {url}", flush=True)
                bad += 1
                continue
            got += 1
            time.sleep(args.delay)
            if sha256(dest) != want:
                print(f"WRONG HASH {dest} (VNDB served different bytes for {url})", flush=True)
                bad += 1
            else:
                ok += 1
        print(f"{set_name}: {ok}/{len(shots)} screenshots ok ({got} downloaded) in {data / platform}",
              flush=True)
    if bad:
        print(f"{bad} screenshots missing or wrong", flush=True)
        sys.exit(1)
    print("FETCH_OK", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Jiten (https://jiten.moe) frequency data for the vocabulary study (docs/vocab-expand.md).

  jiten_fetch.py lists --out DIR       the official frequency lists (CSV): global, every media type, kanji
  jiten_fetch.py decks --out DIR       the top-level deck list of every media type (metadata, VNDB links)
  jiten_fetch.py words --out DIR --type VisualNovel [--sample N] [--seed S]
                                       per-deck word occurrences (the deck's CSV export: one row per word
                                       form with its occurrence count), every deck or a seeded sample
  jiten_fetch.py manifest --out DIR    DIR/manifest.json: every fetched file with its URL, sha256 and date

Why per-deck data: the official lists give ranks, not counts, and Jiten ranks a word by
log(1 + occurrences) x the number of decks that use it (Jiten.Core/JitenHelper.cs,
ApplyAggregatesAndRank, at Sirush/Jiten 7eac4d4), so no rank-to-count model recovers occurrences.
A deck's CSV export (POST /api/media-deck/{id}/download, format Csv, type Full) has the occurrences of
every word form the parser found, which gives character frequencies weighted by occurrence and a
per-title distribution (the 95th-percentile title, not only the corpus mean).

Licence: Jiten's decks, frequency lists and derived data are CC BY-SA 4.0 (jiten.moe footer, retrieved
2026-10-14). The downloads stay under DIR (local, never committed); only aggregate counts and the
manifest are committed, with attribution. CC BY-SA 4.0 is one-way compatible with GPL-3.0.

Politeness: one request at a time, at most one per --interval seconds (default 6.5 s, under the
anonymous download limit of 10 per 60 s in Jiten.Api/Program.cs), gzip transfer, retries with
backoff on 429 and 5xx, an identifying User-Agent. Idempotent: existing files are skipped.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.jiten.moe/api"
UA = "vita-vn-ocr-vocab-study/1.0 (character-frequency research; one request at a time)"
LICENCE = {"name": "CC BY-SA 4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/",
           "source": "https://jiten.moe/frequency-dictionaries"}
# MediaType enum of Jiten.Core/Data/MediaType.cs
MEDIA_TYPES = {"Anime": 1, "Drama": 2, "Movie": 3, "Novel": 4, "NonFiction": 5, "VideoGame": 6,
               "VisualNovel": 7, "WebNovel": 8, "Manga": 9, "Audio": 10, "YouTube": 11}
# DeckDownloadRequest: format Csv (2), downloadType Full (1), order DeckFrequency (3), no filters
DOWNLOAD_BODY = {"format": 2, "downloadType": 1, "order": 3, "minFrequency": 0, "maxFrequency": 0,
                 "excludeKana": False, "excludeMatureMasteredBlacklisted": False, "excludeAllTrackedWords": False,
                 "excludeExampleSentences": True}
_last = [0.0]


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def request(url, body=None, interval=6.5, tries=8):
    """(bytes, headers) of one GET or JSON POST, decompressed, after waiting out the interval."""
    data = None if body is None else json.dumps(body).encode()
    for attempt in range(tries):
        wait = _last[0] + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                     headers={"User-Agent": UA, "Accept-Encoding": "gzip",
                                              **({"Content-Type": "application/json"} if data else {})})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw, dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                back = min(600, 30 * (attempt + 1))
                print(f"{e.code} on {url}; retry in {back} s", file=sys.stderr, flush=True)
                time.sleep(back)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            back = min(600, 20 * (attempt + 1))
            print(f"{e!r} on {url}; retry in {back} s", file=sys.stderr, flush=True)
            time.sleep(back)
    raise RuntimeError(f"giving up on {url}")


def record(out, rec):
    with open(Path(out) / "fetch_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_lists(args):
    out = Path(args.out) / "lists"
    out.mkdir(parents=True, exist_ok=True)
    urls = {"global": f"{API}/frequency-list/download?downloadType=csv"}
    urls.update({t: f"{API}/frequency-list/download?mediaType={t}&downloadType=csv" for t in MEDIA_TYPES})
    urls["kanji"] = f"{API}/frequency-list/download-kanji?downloadType=csv"
    for name, url in urls.items():
        path = out / f"jiten_freq_{name}.csv"
        if path.exists():
            continue
        try:
            raw, hdr = request(url, interval=args.interval)
        except urllib.error.HTTPError as e:
            print(f"{name}: HTTP {e.code}", file=sys.stderr)
            record(args.out, {"kind": "list", "name": name, "url": url, "status": e.code, "retrieved": now()})
            continue
        path.write_bytes(raw)
        rec = {"kind": "list", "name": name, "url": url, "file": str(path.relative_to(args.out)),
               "sha256": sha256_bytes(raw), "bytes": len(raw), "rows": raw.count(b"\n") - 1,
               "retrieved": now(), "http_date": hdr.get("Date") or hdr.get("date")}
        record(args.out, rec)
        print(f"{name}: {len(raw):,} B, {rec['rows']:,} rows", file=sys.stderr, flush=True)


def cmd_decks(args):
    out = Path(args.out) / "decks"
    out.mkdir(parents=True, exist_ok=True)
    for t in MEDIA_TYPES:
        path = out / f"decks_{t}.json"
        if path.exists():
            continue
        url = f"{API}/media-deck/get-media-decks-by-type/{t}"
        raw, hdr = request(url, interval=args.interval)
        path.write_bytes(raw)
        n = len(json.loads(raw))
        record(args.out, {"kind": "decks", "type": t, "url": url, "file": str(path.relative_to(args.out)),
                          "sha256": sha256_bytes(raw), "bytes": len(raw), "decks": n, "retrieved": now(),
                          "http_date": hdr.get("Date") or hdr.get("date")})
        print(f"{t}: {n} decks", file=sys.stderr, flush=True)


def vndb_ids(deck):
    """VNDB VN ids of a deck's links (linkType 2 links to vndb.org/vN)."""
    out = []
    for ln in deck.get("links") or []:
        u = ln.get("url") or ""
        if "vndb.org/v" in u:
            v = "v" + u.rsplit("/v", 1)[1].split("/")[0].split("?")[0]
            if v[1:].isdigit():
                out.append(v)
    return sorted(set(out))


def cmd_words(args):
    decks = json.loads((Path(args.out) / "decks" / f"decks_{args.type}.json").read_text(encoding="utf-8"))
    decks = [d for d in decks if d.get("parentDeckId") is None]
    order = sorted(decks, key=lambda d: d["deckId"])
    random.Random(args.seed).shuffle(order)        # any prefix is a random sample
    if args.sample:
        order = order[:args.sample]
    out = Path(args.out) / "words" / args.type
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for d in order:
        path = out / f"{d['deckId']}.csv.gz"
        if path.exists():
            done += 1
            continue
        url = f"{API}/media-deck/{d['deckId']}/download"
        try:
            raw, hdr = request(url, body=DOWNLOAD_BODY, interval=args.interval)
        except urllib.error.HTTPError as e:
            record(args.out, {"kind": "words", "type": args.type, "deck": d["deckId"], "url": url,
                              "status": e.code, "retrieved": now()})
            print(f"{d['deckId']}: HTTP {e.code}", file=sys.stderr, flush=True)
            continue
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(raw, 6))
        tmp.rename(path)
        done += 1
        record(args.out, {"kind": "words", "type": args.type, "deck": d["deckId"], "url": url,
                          "body": DOWNLOAD_BODY, "file": str(path.relative_to(args.out)),
                          "sha256": sha256_bytes(raw), "bytes": len(raw), "rows": raw.count(b"\n") - 1,
                          "vndb": vndb_ids(d), "characterCount": d.get("characterCount"),
                          "wordCount": d.get("wordCount"), "retrieved": now()})
        if done % 25 == 0:
            print(f"{args.type}: {done}/{len(order)} decks {now()}", file=sys.stderr, flush=True)
    print(f"{args.type}: {done}/{len(order)} decks done", file=sys.stderr, flush=True)


def cmd_manifest(args):
    """The latest successful record per file, plus the licence and the API's source revision."""
    recs = {}
    for line in (Path(args.out) / "fetch_log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("file"):
            recs[r["file"]] = r
    doc = {"kind": "jiten-fetch", "schema": 1, "licence": LICENCE, "api": API, "user_agent": UA,
           "jiten_source": {"repo": "https://github.com/Sirush/Jiten", "commit": args.jiten_commit},
           "files": sorted(recs.values(), key=lambda r: r["file"])}
    (Path(args.out) / "manifest.json").write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n",
                                                  encoding="utf-8")
    print(f"{len(recs)} files -> {Path(args.out) / 'manifest.json'}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lists", "decks", "words", "manifest"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--interval", type=float, default=6.5, help="seconds between requests")
        if name == "words":
            p.add_argument("--type", required=True, choices=sorted(MEDIA_TYPES))
            p.add_argument("--sample", type=int, default=0, help="first N decks of the seeded order (0: all)")
            p.add_argument("--seed", type=int, default=20261014)
        if name == "manifest":
            p.add_argument("--jiten-commit", default="7eac4d44ef10ee3cf0044b333ce7f6a5bb524f85")
    args = ap.parse_args()
    {"lists": cmd_lists, "decks": cmd_decks, "words": cmd_words, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()

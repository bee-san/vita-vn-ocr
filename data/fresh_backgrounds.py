#!/usr/bin/env python3
"""Pin a small native PSP/Vita background cohort, then replay its URL/hash manifest.

The screenshots stay in ignored build/ storage. Reviewed, text-free rectangles
supply synthetic scenery only; their dialogue is never treated as an OCR label.
Each platform gets 16 training games and four different development games. All
40 games are distinct, and CLANNAD plus the shared benchmark games/sessions are
excluded. Existing PSP image hashes are retained from synth2_psp_shots.jsonl.

  python data/fresh_backgrounds.py --out build/hiragana/fresh-jiten

The first invocation creates the small manifest; later invocations fetch exactly
those pinned files without repeating the mutable VNDB selection query.
"""
import argparse
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synth.bg_exclude import load_exclusions  # noqa: E402
from synth.common import is_reserved_vn  # noqa: E402

API = "https://api.vndb.org/kana/vn"
FIELDS = ("id,title,screenshots.url,screenshots.dims,screenshots.sexual,"
          "screenshots.violence,screenshots.release.id,screenshots.release.platforms")
PLATFORMS = {"psp": ((480, 272), "vndb_psp"), "psv": ((960, 544), "vndb")}
PSP_MANIFEST = ROOT / "data/manifests/synth2_psp_shots.jsonl"
TEST_GAMES = ROOT / "eval/manifests/test_games.json"
EXCLUSIONS = ROOT / "data/manifests/synth2_bg_exclude.txt"
DEFAULT_MANIFEST = ROOT / "data/manifests/fresh_jiten_backgrounds.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def exclusion_sets():
    """Read both the central exclusion loader and reserved capture sessions."""
    games = json.loads(TEST_GAMES.read_text(encoding="utf-8"))
    return load_exclusions() | {"v4"}, set(games.get("sessions", []))


def filename(row):
    return f"{row['vn']}_{row['url'].rsplit('/', 1)[-1]}"


def eligible(row, platform, excluded, sessions):
    """Require screenshot-specific native dimensions and a console release."""
    if platform not in PLATFORMS:
        return False
    vn, release, url = row.get("vn", ""), row.get("release", ""), row.get("url", "")
    if not all(isinstance(value, str) for value in (vn, release, url)):
        return False
    parsed = urlparse(url)
    if (not re.fullmatch(r"v[1-9][0-9]*", vn)
            or not re.fullmatch(r"r[1-9][0-9]*", release)
            or parsed.scheme != "https" or parsed.netloc != "t.vndb.org"
            or not re.fullmatch(r"/sf/[0-9]{2}/[0-9]+\.jpg", parsed.path)
            or parsed.query or parsed.fragment):
        return False
    return (vn != "v4" and vn not in excluded and not is_reserved_vn(vn)
            and filename(row) not in excluded and release not in sessions
            and platform in (row.get("release_platforms") or [])
            and tuple(row.get("dims") or []) == PLATFORMS[platform][0]
            and isinstance(row.get("sexual"), (int, float)) and 0 <= row["sexual"] < .3
            and isinstance(row.get("violence"), (int, float)) and 0 <= row["violence"] < .5)


def screenshot_rows(response):
    rows = []
    for vn in response["results"]:
        for shot in vn.get("screenshots") or []:
            release = shot.get("release") or {}
            rows.append({"vn": vn["id"], "title": vn["title"], "url": shot["url"],
                         "dims": shot["dims"], "sexual": shot["sexual"],
                         "violence": shot["violence"], "release": release.get("id"),
                         "release_platforms": release.get("platforms") or []})
    return rows


def choose(rows, platform, count, excluded, sessions, used_vns=(), used_releases=()):
    """Deterministic one-per-game selection; no game/release crosses cohorts."""
    vns, releases, selected = set(used_vns), set(used_releases), []
    order = sorted(rows, key=lambda r: (hashlib.sha256(f"1:{r['vn']}".encode()).hexdigest(),
                                        r["url"]))
    for row in order:
        if (row["vn"] in vns or row["release"] in releases
                or not eligible(row, platform, excluded, sessions)):
            continue
        selected.append(dict(row, platform=platform))
        vns.add(row["vn"])
        releases.add(row["release"])
        if len(selected) == count:
            break
    return selected


def validate_manifest(manifest, excluded=None, sessions=None):
    if manifest.get("schema") not in (1, 2):
        raise ValueError("unsupported fresh background manifest schema")
    if manifest.get("content_sha256") != canonical_sha(
            {k: v for k, v in manifest.items() if k != "content_sha256"}):
        raise ValueError("background manifest content hash changed")
    if excluded is None or sessions is None:
        excluded, sessions = exclusion_sets()
    counts = manifest.get("counts", {})
    if (set(counts) != {"train", "dev"}
            or any(type(v) is not int or v <= 0 for v in counts.values())):
        raise ValueError("positive training/development counts are required")
    vns, releases, files, hashes, seen_counts = set(), set(), set(), set(), {}
    for row in manifest.get("records", []):
        platform, split = row.get("platform"), row.get("split")
        if split not in counts or not eligible(row, platform, excluded, sessions):
            raise ValueError(f"excluded or non-native background: {row.get('vn')}")
        if manifest["schema"] == 2:
            review = row.get("scenery_review", {})
            if (review.get("hand_checked") is not True
                    or not isinstance(review.get("reason"), str) or not review["reason"].strip()
                    or "scenery_box" not in row):
                raise ValueError("every scenery patch requires an explicit hand-checked review")
            box = row["scenery_box"]
            if box is not None:
                w, h = PLATFORMS[platform][0]
                if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                        or not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h):
                    raise ValueError("scenery rectangle must fit inside the native screenshot")
        expected = f"{'assets' if split == 'train' else 'assets-dev'}/{PLATFORMS[platform][1]}/shots/{filename(row)}"
        if row.get("file") != expected:
            raise ValueError("background file must have its exact cohort/platform path")
        digest = row.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise ValueError("background byte count and SHA256 are required")
        if (row["vn"] in vns or row["release"] in releases or row["file"] in files
                or digest in hashes):
            raise ValueError("background game/release/image identity is repeated across cohorts")
        vns.add(row["vn"])
        releases.add(row["release"])
        files.add(row["file"])
        hashes.add(digest)
        key = platform, split
        seen_counts[key] = seen_counts.get(key, 0) + 1
    if any(seen_counts.get((p, s), 0) != n for p in PLATFORMS for s, n in counts.items()):
        raise ValueError("background cohort counts do not match the manifest")
    return manifest


def verify_blob(blob, row):
    if row.get("sha256") and hashlib.sha256(blob).hexdigest() != row["sha256"]:
        raise ValueError(f"background SHA256 mismatch: {row['url']}")
    if row.get("bytes") is not None and len(blob) != row["bytes"]:
        raise ValueError(f"background size mismatch: {row['url']}")
    with Image.open(io.BytesIO(blob)) as image:
        if image.format != "JPEG" or image.size != PLATFORMS[row["platform"]][0]:
            raise ValueError(f"background is not a native console JPEG: {row['url']}")
        image.load()


def download(session, row, out):
    dest = Path(out) / row["file"]
    if dest.exists():
        blob = dest.read_bytes()
    else:
        response = session.get(row["url"], timeout=60)
        response.raise_for_status()
        blob = response.content
        verify_blob(blob, row)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".jpg.part")
        tmp.write_bytes(blob)
        tmp.replace(dest)
        time.sleep(.3)
    verify_blob(blob, row)
    return dict(row, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))


def query(session, body, cache, queries):
    """Cache only selection metadata locally; record request and response hashes."""
    cache = Path(cache)
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["request"] != body:
            raise ValueError(f"different query identity in reused cache: {cache}")
    else:
        for attempt in range(3):
            response = session.post(API, json=body, timeout=60)
            if response.status_code != 429:
                break
            delay = min(60, max(1, int(response.headers.get("Retry-After", "30"))))
            print(f"VNDB rate limit, waiting {delay}s", flush=True)
            time.sleep(delay)
        response.raise_for_status()
        saved = {"request": body, "response": response.json()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    queries.append({"url": API, "request": body, "response_sha256": canonical_sha(saved["response"])})
    return saved["response"]


def create_manifest(session, out, train=16, dev=4):
    excluded, sessions = exclusion_sets()
    pins = [json.loads(line) for line in PSP_MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    pin_map = {(r["vn"], r["url"], r["release"]): r for r in pins}
    ids = sorted({r["vn"] for r in pins if r["vn"] not in excluded and not is_reserved_vn(r["vn"])
                  and r["release"] not in sessions}, key=lambda v: int(v[1:]))
    if len(ids) > 100:
        raise ValueError("PSP source exceeds this deliberately small selection query")
    queries = []
    body = {"filters": ["or", *[["id", "=", v] for v in ids]], "fields": FIELDS, "results": 100}
    response = query(session, body, Path(out) / "background-query-psp.json", queries)
    rows = []
    for row in screenshot_rows(response):
        pin = pin_map.get((row["vn"], row["url"], row["release"]))
        if pin:
            rows.append(dict(row, sha256=pin["sha256"], source="synth2_psp_shots.jsonl+VNDB-kana"))
    psp = choose(rows, "psp", train + dev, excluded, sessions)
    if len(psp) != train + dev:
        raise ValueError("not enough pinned native PSP games")
    psv_rows, psv = [], []
    for page in range(1, 21):
        body = {"filters": ["and", ["olang", "=", "ja"], ["release", "=", ["platform", "=", "psv"]]],
                "fields": FIELDS, "results": 100, "page": page}
        response = query(session, body, Path(out) / f"background-query-psv-{page}.json", queries)
        psv_rows.extend(dict(r, source="VNDB-kana") for r in screenshot_rows(response))
        psv = choose(psv_rows, "psv", train + dev, excluded, sessions,
                     [r["vn"] for r in psp], [r["release"] for r in psp])
        print(f"Vita page {page}: {len(psv)} eligible distinct games", flush=True)
        if len(psv) == train + dev or not response.get("more"):
            break
    if len(psv) != train + dev:
        raise ValueError("not enough native Vita games separate from PSP/background exclusions")
    records = []
    for rows in (psp, psv):
        for index, row in enumerate(rows):
            split = "train" if index < train else "dev"
            base = "assets" if split == "train" else "assets-dev"
            row = dict(row, split=split, file=f"{base}/{PLATFORMS[row['platform']][1]}/shots/{filename(row)}")
            row = download(session, row, out)
            records.append(row)
            print(f"{row['platform']} {split} {row['vn']}: {row['bytes']} bytes, SHA256 verified", flush=True)
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "counts": {"train": train, "dev": dev}, "selection_seed": 1,
                "usage": "Synthetic scenery only, upper 62 percent; no screenshot-derived OCR labels.",
                "provenance": {"api_documentation": "https://api.vndb.org/kana",
                               "psp_manifest": {"file": str(PSP_MANIFEST.relative_to(ROOT)),
                                                "sha256": sha256(PSP_MANIFEST)},
                               "test_games_sha256": sha256(TEST_GAMES),
                               "background_exclusions_sha256": sha256(EXCLUSIONS),
                               "fetcher_sha256": sha256(__file__), "queries": queries},
                "records": records}
    manifest["content_sha256"] = canonical_sha(manifest)
    return validate_manifest(manifest, excluded, sessions)


def main():
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "vita-vn-ocr-fresh-backgrounds/1.0 (local OCR research)"
    if args.manifest.exists():
        manifest = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        for row in manifest["records"]:
            download(session, row, args.out)
    else:
        manifest = create_manifest(session, args.out)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(manifest['records'])} native backgrounds: {args.manifest}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch training assets for the Vita VN OCR into vita-ocr/data/.

  fonts   OFL Japanese fonts from google/fonts (a few weights per family)
  corpus  Aozora Bunko (public domain, cleaned) + one Japanese Wikipedia shard
  vndb    PS Vita VN screenshots listed by the VNDB kana API, then downloaded
          (960x544 only, platform psv, sexual < 0.3 and violence < 0.5)
  all     everything above, in that order

Idempotent: existing files are skipped, so a killed run can simply be restarted.
Screenshots are copyrighted game images: keep them local and never commit them.
"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
UA = "vita-jp-overlay-ocr-research/0.1 (local OCR training data)"
S = requests.Session()
S.headers["User-Agent"] = UA

# google/fonts ofl/ directory names. Gothic, rounded and mincho faces cover what
# Vita VNs mostly use (Shin Go/Rodin-like, Seurat-like rounded, Ryumin-like);
# a few pen/pixel faces add robustness. Missing families are logged and skipped.
FONT_FAMILIES = [
    "notosansjp", "notoserifjp", "mplus1p", "mplusrounded1c", "mplus2", "kosugi",
    "kosugimaru", "sawarabigothic", "sawarabimincho", "zenmarugothic",
    "zenkakugothicnew", "zenkakugothicantique", "zenoldmincho", "zenantique",
    "shipporimincho", "shipporiantique", "kleeone", "bizudgothic", "bizudpgothic",
    "bizudmincho", "dotgothic16", "kiwimaru", "yujisyuku", "murecho", "ibmplexsansjp",
    "hinamincho", "kaiseiopti", "kaiseitokumin", "rocknrollone", "mochiypopone",
    "zenkurenaido", "hachimarupop", "yuseimagic", "yomogi",
]
WEIGHTS = ("-Regular.", "-Medium.", "-Bold.", "[wght]")


def get(url, allow_404=False, **kw):
    for attempt in range(6):
        try:
            r = S.get(url, timeout=120, **kw)
            if allow_404 and r.status_code == 404:
                return None
            if r.status_code in (403, 429) and "api.github.com" in url:
                print(f"rate limited by GitHub, sleeping: {url}", flush=True)
                time.sleep(120 * (attempt + 1))
                continue
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"retry {attempt} {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with get(url, stream=True) as r, open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    tmp.rename(dest)
    return True


def fonts():
    out = ROOT / "fonts"
    for fam in FONT_FAMILIES:
        r = get(f"https://api.github.com/repos/google/fonts/contents/ofl/{fam}", allow_404=True)
        if r is None:
            print(f"font family missing: {fam}", flush=True)
            continue
        entries = r.json()
        faces = [e for e in entries if e["name"].endswith((".ttf", ".otf"))]
        keep = [e for e in faces if any(w in e["name"] for w in WEIGHTS)] or faces[:1]
        for e in keep + [e for e in entries if e["name"] == "OFL.txt"]:
            if download(e["download_url"], out / fam / e["name"]):
                print(f"font {fam}/{e['name']}", flush=True)
        time.sleep(1)
    n = len(list(out.glob("*/*.[to]tf")))
    print(f"fonts done: {n} files", flush=True)


def corpus():
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    out = ROOT / "corpus"
    repo = "globis-university/aozorabunko-clean"
    files = api.list_repo_files(repo, repo_type="dataset")
    print(f"aozora files: {files}", flush=True)
    for f in files:
        if f.endswith((".jsonl.gz", ".jsonl", ".parquet")):
            hf_hub_download(repo, f, repo_type="dataset", local_dir=out / "aozora")
            print(f"aozora {f}", flush=True)
    repo = "wikimedia/wikipedia"
    shards = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                    if f.startswith("20231101.ja/") and f.endswith(".parquet"))
    print(f"wiki ja shards: {len(shards)}", flush=True)
    for f in shards[:1]:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=out / "wiki")
        print(f"wiki {f}", flush=True)


def vndb():
    out = ROOT / "vndb"
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "screenshots.jsonl"
    if not meta.exists():
        rows, page = [], 1
        fields = ("id,title,screenshots.url,screenshots.dims,screenshots.sexual,"
                  "screenshots.violence,screenshots.release.id,screenshots.release.platforms")
        while True:
            body = {"filters": ["and", ["olang", "=", "ja"],
                                ["release", "=", ["platform", "=", "psv"]]],
                    "fields": fields, "results": 100, "page": page}
            r = S.post("https://api.vndb.org/kana/vn", json=body, timeout=120)
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            d = r.json()
            for vn in d["results"]:
                for s in vn.get("screenshots") or []:
                    rel = s.get("release") or {}
                    rows.append({"vn": vn["id"], "title": vn["title"], "url": s["url"],
                                 "dims": s["dims"], "sexual": s["sexual"],
                                 "violence": s["violence"], "release": rel.get("id"),
                                 "platforms": rel.get("platforms") or []})
            print(f"vndb page {page}: {len(d['results'])} VNs, {len(rows)} screenshots", flush=True)
            if not d.get("more"):
                break
            page += 1
            time.sleep(1)
        tmp = meta.with_name(meta.name + ".part")
        tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        tmp.rename(meta)
    rows = [json.loads(line) for line in meta.read_text().splitlines()]
    keep = [x for x in rows if "psv" in x["platforms"] and x["dims"] == [960, 544]
            and x["sexual"] < 0.3 and x["violence"] < 0.5]
    print(f"vndb: {len(rows)} listed, {len(keep)} safe 960x544 PSV screenshots", flush=True)
    for i, x in enumerate(keep):
        name = f"{x['vn']}_{x['url'].rsplit('/', 1)[-1]}"
        if download(x["url"], out / "shots" / name):
            time.sleep(0.3)
        if i % 100 == 0:
            print(f"shots {i}/{len(keep)}", flush=True)
    print(f"vndb done: {len(list((out / 'shots').glob('*.jpg')))} screenshots", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"fonts": [fonts], "corpus": [corpus], "vndb": [vndb],
             "all": [fonts, corpus, vndb]}[what]
    for step in steps:
        step()
    print("FETCH_DONE", flush=True)

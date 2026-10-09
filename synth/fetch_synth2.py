#!/usr/bin/env python3
"""Fetch the extra assets the synth2 renderer uses, with pinned URLs and sha256.

  fonts  OFL pixel fonts (Fusion Pixel 10/12 px proportional, Japanese glyph
         variant) from a pinned GitHub release, into data/fonts_extra/
  psp    PSP screenshots (480x272, VNDB platform psp, sexual < 0.3, violence < 0.5)
         of *training-side* VNs only: VNs in data/vndb/screenshots.jsonl for which
         common.is_test_vn() is false. Used as PSP backgrounds and for style
         calibration, never for evaluation. Into data/vndb_psp/shots/.
  all    both

Downloads are skipped when the file exists with the expected sha256. Each step
writes a manifest (URL + sha256) to data/manifests/, which is committed; the
fonts and screenshots themselves are not. Screenshots are copyrighted game
images: keep them local.
"""
import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, is_test_vn  # noqa: E402

UA = "vita-vn-ocr-synth2/0.1 (synthetic VN OCR training data; github.com/bee-san/vita-vn-ocr)"
MANIFESTS = DATA / "manifests"

# Fusion Pixel Font (TakWolf), fonts under SIL OFL 1.1. The proportional "ja"
# build covers 100% of the JIS X 0208 kanji (upstream docs/info-*px-proportional.md).
FUSION_TAG = "2026.09.25"
FUSION = {
    "fusionpixel10": ("fusion-pixel-font-10px-proportional-ttf-v2026.09.25.zip",
                      "fusion-pixel-10px-proportional-ja.ttf"),
    "fusionpixel12": ("fusion-pixel-font-12px-proportional-ttf-v2026.09.25.zip",
                      "fusion-pixel-12px-proportional-ja.ttf"),
}
FUSION_URL = "https://github.com/TakWolf/fusion-pixel-font/releases/download/{tag}/{name}"


def session():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def get(s, url):
    for attempt in range(5):
        try:
            r = s.get(url, timeout=120)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"retry {attempt} {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def fonts(out=DATA / "fonts_extra"):
    s = session()
    rows = []
    for fam, (zname, member) in FUSION.items():
        dest = out / fam / member
        url = FUSION_URL.format(tag=FUSION_TAG, name=zname)
        if not dest.exists():
            blob = get(s, url)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                names = z.namelist()
                ttf = next(n for n in names if n.endswith(member))
                # OFL.txt at the zip root is the font's own licence; LICENSES/<upstream>/
                # holds the upstream fonts' licences. Keep the paths so none overwrites another.
                lic = [n for n in names if n.rsplit("/", 1)[-1].upper().startswith(("OFL", "LICENSE"))]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(ttf))
                for n in lic:
                    (dest.parent / n).parent.mkdir(parents=True, exist_ok=True)
                    (dest.parent / n).write_bytes(z.read(n))
            rows.append({"family": fam, "zip_url": url, "zip_sha256": sha256(blob)})
            print(f"font {fam}: {member} from {zname}", flush=True)
        else:
            rows.append({"family": fam, "zip_url": url, "zip_sha256": None})
        rows[-1].update({"file": f"{fam}/{member}", "sha256": sha256(dest.read_bytes()),
                         "license": "OFL-1.1"})
    old = MANIFESTS / "synth2_fonts.json"
    if old.exists():   # keep zip hashes recorded by the first download
        prev = {r["family"]: r for r in json.loads(old.read_text())}
        for r in rows:
            r["zip_sha256"] = r["zip_sha256"] or prev.get(r["family"], {}).get("zip_sha256")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps(rows, indent=1) + "\n")
    print("fonts done", flush=True)


def psp(per_vn=8, out=DATA / "vndb_psp" / "shots"):
    rows = [json.loads(x) for x in (DATA / "vndb" / "screenshots.jsonl").read_text().splitlines()]
    keep = [x for x in rows if "psp" in x["platforms"] and x["dims"] == [480, 272]
            and x["sexual"] < 0.3 and x["violence"] < 0.5 and not is_test_vn(x["vn"])]
    by_vn = {}
    for x in keep:
        by_vn.setdefault(x["vn"], []).append(x)
    picks = [x for vn in sorted(by_vn) for x in sorted(by_vn[vn], key=lambda r: r["url"])[:per_vn]]
    print(f"psp: {len(keep)} training-side screenshots of {len(by_vn)} VNs, fetching {len(picks)}",
          flush=True)
    out.mkdir(parents=True, exist_ok=True)
    s, man = session(), []
    for i, x in enumerate(picks):
        dest = out / f"{x['vn']}_{x['url'].rsplit('/', 1)[-1]}"
        if not dest.exists():
            dest.write_bytes(get(s, x["url"]))
            time.sleep(1.0)
        man.append({"vn": x["vn"], "title": x["title"], "url": x["url"],
                    "sha256": sha256(dest.read_bytes()), "release": x["release"]})
        if i % 50 == 0:
            print(f"psp {i}/{len(picks)}", flush=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "synth2_psp_shots.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man))
    print(f"psp done: {len(man)} screenshots", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fonts", "psp", "all"])
    ap.add_argument("--per-vn", type=int, default=8)
    a = ap.parse_args()
    if a.what in ("fonts", "all"):
        fonts()
    if a.what in ("psp", "all"):
        psp(a.per_vn)
    print("FETCH_DONE", flush=True)

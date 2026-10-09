#!/usr/bin/env python3
"""Fetch the extra assets the synth2 renderer uses, pinned by the committed manifests.

  fonts  OFL pixel fonts (Fusion Pixel 10/12 px proportional, Japanese glyph
         variant) from a pinned GitHub release, into data/fonts_extra/
  psp    PSP screenshots (480x272, VNDB platform psp, sexual < 0.3, violence < 0.5)
         of *training-side* VNs only, into data/vndb_psp/shots/. Used as PSP
         backgrounds and for style calibration, never for evaluation.
  all    both

By default each step fetches exactly what its committed manifest in data/manifests/
lists (synth2_fonts.json, synth2_psp_shots.jsonl) and checks every file's sha256
against it, local copies included; a mismatch is an error. --refresh rebuilds a
manifest instead: the fonts from the pinned release, the PSP pick from the local VNDB
screenshot list (data/vndb/screenshots.jsonl), skipping every is_test_vn() game and
every VN eval reserves (bg_exclude.load_exclusions). The fonts and screenshots
themselves are never committed; screenshots are copyrighted game images, so keep
them local. Requests carry a User-Agent; screenshots are fetched 1 per second.
"""
import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bg_exclude import load_exclusions  # noqa: E402
from common import DATA, is_test_vn  # noqa: E402

UA = "vita-vn-ocr-synth2/0.1 (synthetic VN OCR training data; github.com/bee-san/vita-vn-ocr)"
MANIFESTS = HERE.parent / "data" / "manifests"
FONTS_MANIFEST = MANIFESTS / "synth2_fonts.json"
PSP_MANIFEST = MANIFESTS / "synth2_psp_shots.jsonl"

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
    import requests  # only needed to download (the pin checks run without it)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def check(what, got, want):
    if want is not None and got != want:
        raise SystemExit(f"sha256 mismatch for {what}: got {got}, the manifest pins {want}")


def get(s, url):
    import requests
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


def fonts(out=DATA / "fonts_extra", refresh=False):
    old = {r["family"]: r for r in json.loads(FONTS_MANIFEST.read_text())} if FONTS_MANIFEST.exists() else {}
    pins = {} if refresh else old
    s = session()
    rows = []
    for fam, (zname, member) in FUSION.items():
        pin = pins.get(fam, {})
        dest = out / fam / member
        url = FUSION_URL.format(tag=FUSION_TAG, name=zname)
        zip_sha = old.get(fam, {}).get("zip_sha256")     # kept when the font is already local
        if not dest.exists():
            blob = get(s, url)
            zip_sha = sha256(blob)
            check(url, zip_sha, pin.get("zip_sha256"))
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
            print(f"font {fam}: {member} from {zname}", flush=True)
        file_sha = sha256(dest.read_bytes())
        check(dest, file_sha, pin.get("sha256"))
        rows.append({"family": fam, "zip_url": url, "zip_sha256": zip_sha,
                     "file": f"{fam}/{member}", "sha256": file_sha, "license": "OFL-1.1"})
    if refresh or not FONTS_MANIFEST.exists():
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        FONTS_MANIFEST.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {FONTS_MANIFEST}", flush=True)
    print("fonts done (sha256 match the manifest)", flush=True)


def psp_pick(per_vn):
    """Up to per_vn training-side 480x272 PSP screenshots per VN from the local VNDB list."""
    rows = [json.loads(x) for x in (DATA / "vndb" / "screenshots.jsonl").read_text().splitlines()]
    reserved = load_exclusions()
    keep = [x for x in rows if "psp" in x["platforms"] and x["dims"] == [480, 272]
            and x["sexual"] < 0.3 and x["violence"] < 0.5 and not is_test_vn(x["vn"])
            and x["vn"] not in reserved]
    by_vn = {}
    for x in keep:
        by_vn.setdefault(x["vn"], []).append(x)
    print(f"psp: {len(keep)} training-side screenshots of {len(by_vn)} VNs", flush=True)
    return [{"vn": x["vn"], "title": x["title"], "url": x["url"], "sha256": None, "release": x["release"]}
            for vn in sorted(by_vn) for x in sorted(by_vn[vn], key=lambda r: r["url"])[:per_vn]]


def psp(per_vn=8, out=DATA / "vndb_psp" / "shots", refresh=False):
    if refresh or not PSP_MANIFEST.exists():
        picks = psp_pick(per_vn)
    else:
        picks = [json.loads(x) for x in PSP_MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"psp: fetching {len(picks)} screenshots", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    s, man = session(), []
    for i, x in enumerate(picks):
        dest = out / f"{x['vn']}_{x['url'].rsplit('/', 1)[-1]}"
        if not dest.exists():
            blob = get(s, x["url"])
            check(x["url"], sha256(blob), x["sha256"])
            dest.write_bytes(blob)
            time.sleep(1.0)
        check(dest, sha256(dest.read_bytes()), x["sha256"])
        man.append(dict(x, sha256=sha256(dest.read_bytes())))
        if i % 50 == 0:
            print(f"psp {i}/{len(picks)}", flush=True)
    if refresh or not PSP_MANIFEST.exists():
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        PSP_MANIFEST.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man))
        print(f"wrote {PSP_MANIFEST}", flush=True)
    print(f"psp done: {len(man)} screenshots (sha256 match the manifest)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fonts", "psp", "all"])
    ap.add_argument("--per-vn", type=int, default=8)
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the manifests instead of fetching and checking what they pin")
    a = ap.parse_args()
    if a.what in ("fonts", "all"):
        fonts(refresh=a.refresh)
    if a.what in ("psp", "all"):
        psp(a.per_vn, refresh=a.refresh)
    print("FETCH_DONE", flush=True)

#!/usr/bin/env python3
"""The one OFL font CI renders with, fetched from google/fonts at a pinned commit.

  fonts.py [--cache DIR]     download (or reuse) the font and its licence, print the font path

Both files are checked against their sha256 before use, so a changed upstream file
or a corrupted cache fails loudly instead of silently changing what CI renders.
M+ 1p (SIL Open Font License 1.1) is a gothic face, the style most Vita VNs use;
it covers every character of ci/fixtures/corpus and 4,805 of the 5,354 vocab classes.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

COMMIT = "2eb0b48d5f760f62e286216f0859a8c540dbc1bd"       # google/fonts main, 2026-10-08
FAMILY = "mplus1p"                                         # render.py CATEGORY family dir
FILES = {   # name: (sha256, bytes)
    "MPLUS1p-Regular.ttf": ("2f294ad496432b1608f070d310e3aa2adcf1de4af429f4901df97ec4bd361ed1", 1758688),
    "OFL.txt": ("da15da6b1496d4de18f97e2ad1b722ef8a1c121149c2c93b2cf7eac6ac27b35c", 4389),
}
FONT_FILE = "MPLUS1p-Regular.ttf"
URL = "https://raw.githubusercontent.com/google/fonts/{commit}/ofl/{family}/{name}"
UA = "vita-vn-ocr-ci/1.0 (+https://github.com/bee-san/vita-vn-ocr)"
DEFAULT_CACHE = Path(os.environ.get("VOCR_FONT_CACHE", Path.home() / ".cache" / "vita-vn-ocr" / "fonts"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url, dest, attempts=4):
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")    # the cache may be shared
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except OSError as e:          # URLError and HTTPError are OSErrors
            if attempt == attempts - 1:
                raise RuntimeError(f"could not download {url}: {e}") from e
            time.sleep(2 ** attempt)


def ensure_font(cache=DEFAULT_CACHE):
    """Return the path of the verified font in cache/<commit>/<family>/ (OFL.txt beside it),
    downloading whatever is missing or does not match its pinned hash."""
    d = Path(cache) / COMMIT / FAMILY
    d.mkdir(parents=True, exist_ok=True)
    for name, (digest, size) in FILES.items():
        dest = d / name
        if dest.exists() and dest.stat().st_size == size and sha256(dest) == digest:
            continue
        _fetch(URL.format(commit=COMMIT, family=FAMILY, name=name), dest)
        got = sha256(dest)
        if got != digest:
            dest.unlink()
            raise RuntimeError(f"{name}: sha256 {got} does not match the pinned {digest}")
    return d / FONT_FILE


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()
    print(ensure_font(args.cache))


if __name__ == "__main__":
    sys.exit(main())

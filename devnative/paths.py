"""Paths, screenshot ids and hashes shared by the devnative tools.

Everything bulky (screenshots, readings, crops, sheets) lives under $DEVNATIVE_OUT, by
default $HERD/shared/artifacts/devnative, and never in git. The repo holds code and the
manifests in devnative/manifests/ (URLs, hashes, ids, provenance; no text, no pixels).
"""
import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
SHARED_DATA = HERD / "shared" / "data"
OUT = Path(os.environ.get("DEVNATIVE_OUT") or ART / "devnative")
MANIFESTS = ROOT / "devnative" / "manifests"

KINDS = ("vita", "psp")
DIMS = {"vita": (960, 544), "psp": (480, 272)}
PAD = {"vita": 4, "psp": 2}                 # ink-box crop padding, as eval and teacher crop
MARGIN = {"vita": 12, "psp": 6}             # dialogue-region margin, as eval/tools/manifest.py
UA = "vita-vn-ocr-devnative/0.1 (development cohort listing, low volume; github.com/bee-san/vita-vn-ocr)"

_NUM = re.compile(r"(?:/sf/\d+/|_|^sf)(\d+)(?:\.\w+)?$")


def image_num(ref):
    """VNDB screenshot number from a URL (https://t.vndb.org/sf/46/97646.jpg), a file name
    (v36_97646.jpg), a shot key (v36_97646) or an id (sf97646)."""
    m = _NUM.search(str(ref).rsplit("/", 1)[-1] if "/sf/" not in str(ref) else str(ref))
    if not m:
        raise ValueError(f"no VNDB screenshot number in {ref!r}")
    return int(m.group(1))


def image_id(ref):
    """'sf<number>': VNDB's id of a screenshot, unique across VNs."""
    return f"sf{image_num(ref)}"


def shot_key(vn, url):
    """<vn>_<number>, the file stem every tool in the repo uses for a VNDB screenshot."""
    return f"{vn}_{image_num(url)}"


def vndb_url(num):
    """The VNDB URL of screenshot <num> (its directory is the number mod 100, two digits)."""
    return f"https://t.vndb.org/sf/{num % 100:02d}/{num}.jpg"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(path):
    """HxWx3 uint8 RGB as OpenCV decodes it (the decoder every tool here uses), or None."""
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if bgr is None else bgr[:, :, ::-1].copy()


def pixel_sha256(rgb):
    """sha256 of decoded pixels: 'HxWx3' then the RGB bytes. Two files with different bytes
    (metadata, a lossless re-save) but the same picture share it."""
    import numpy as np
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    h = hashlib.sha256(f"{a.shape[0]}x{a.shape[1]}x{a.shape[2] if a.ndim == 3 else 1}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def thumb(rgb):
    """32x18 grey thumbnail, zero mean and unit norm: eval/tools/leakcheck.py's fingerprint
    (there from the file with PIL's draft decode; here from decoded pixels, area-resized)."""
    import cv2
    import numpy as np
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def screen_path(row):
    """A screenshot row's file: its "path", else $OUT/shots/<kind>/<vn>_<file>."""
    if row.get("path"):
        return Path(row["path"])
    return OUT / "shots" / row["kind"] / f"{row['vn']}_{row['url'].rsplit('/', 1)[-1]}"


def fit_scale(w, h, kind):
    tw, th = DIMS[kind]
    return min(tw / w, th / h)


def load_screen(row):
    """RGB of a screenshot at the console's resolution: native Vita/PSP shots as they are; any
    other size (a PC or home-console shot read as `kind`) scaled to fit the console screen with
    cv2.INTER_AREA, as teacher/build.py scales its hires shots. None if unreadable."""
    import cv2
    rgb = read_rgb(screen_path(row))
    if rgb is None:
        return None
    h, w = rgb.shape[:2]
    if (w, h) == DIMS[row["kind"]]:
        return rgb
    s = fit_scale(w, h, row["kind"])
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

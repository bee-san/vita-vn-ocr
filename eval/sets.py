"""Test sets for eval/run.py.

  vita-test   hand-checked dialogue lines from real PS Vita screenshots (960x544)
  psp-test    hand-checked dialogue lines from real PSP screenshots (480x272)
  synth-val   the held-out synthetic shard (data/synth/synth_00039.npz, the shard
              train/train.py validates on), first 3000 lines like train.py

Manifest rows (eval/manifests/<set>.jsonl), one per text line:
  id          stable line id, e.g. "vita-0001"
  vn, title   VNDB VN id and title (a test game; see manifests/test_games.json)
  url, sha256 the VNDB screenshot and the sha256 of its bytes (eval/fetch.py)
  dims        screenshot size [w, h]
  region      [x, y, w, h]: the dialogue region a player would draw around the box
  box         [x, y, w, h]: the line's ink extent (main text plus any furigana
              attached above it); crops add --pad px on every side (default 4 on
              Vita like the overlay's line finder, 2 on PSP, whose 480x272 screen
              is half the scale and leaves about 4 px between lines)
  text        ground truth: the main text only (furigana is never part of it),
              in the training-label convention (full-width ！？～, … for ellipsis)
  styles      tags, see STYLES
  uncertain   true if the text could not be decided by eye; such lines are never
              scored (in finder mode a detection on them is neither hit nor false alarm)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFESTS = HERE / "manifests"
SETS = {"vita-test": "vita", "psp-test": "psp", "synth-val": None}
PAD = {"vita": 4, "psp": 2}  # default px around each line box, per platform
SYNTH_VAL = ROOT / "data" / "synth" / "synth_00039.npz"
SYNTH_VAL_LINES = 3000

STYLES = {
    "bright-on-dark": "text lighter than its background",
    "dark-on-light": "text darker than its background",
    "outlined": "text with an outline or heavy drop shadow",
    "coloured": "text in a colour other than white, black or grey",
    "coloured-name": "a speaker name in a colour other than the dialogue's",
    "name": "a speaker-name line (name plate or 【name】)",
    "furigana": "furigana (ruby) above the line; never part of the label",
    "no-box": "text drawn over the scene without a dialogue box",
    "mincho": "serif (mincho) face",
    "small": "glyphs under 16 px tall at native resolution",
}


class Line:
    __slots__ = ("id", "set", "vn", "title", "image", "box", "text", "styles", "uncertain",
                 "region", "region_id", "crop_rgb")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def read_manifest(name):
    path = MANIFESTS / f"{name}.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def pad_box(box, pad, w_img, h_img):
    x, y, w, h = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    return x0, y0, x1, y1


class ImageCache:
    def __init__(self):
        self.cache = {}

    def get(self, path):
        img = self.cache.get(path)
        if img is None:
            from PIL import Image
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} missing: run eval/fetch.py first")
            img = np.asarray(Image.open(path).convert("RGB"))
            self.cache[path] = img
        return img


def load_real(name, data_dir, pad=None, include_uncertain=False):
    from fetch import shot_path
    platform = SETS[name]
    pad = PAD[platform] if pad is None else pad
    cache = ImageCache()
    lines = []
    for r in read_manifest(name):
        if r.get("uncertain") and not include_uncertain:
            continue
        path = shot_path(data_dir, platform, r["vn"], r["url"])
        img = cache.get(path)
        x0, y0, x1, y1 = pad_box(r["box"], pad, img.shape[1], img.shape[0])
        lines.append(Line(id=r["id"], set=name, vn=r["vn"], title=r.get("title"), image=str(path),
                          box=r["box"], text=r["text"], styles=r.get("styles") or [],
                          uncertain=bool(r.get("uncertain")), region=r.get("region"),
                          region_id=f"{Path(path).name}@{','.join(map(str, r['region']))}"
                          if r.get("region") else None,
                          crop_rgb=np.ascontiguousarray(img[y0:y1, x0:x1])))
    return lines, cache


def synth_styles(meta):
    tags = ["bright-on-dark"]
    if meta.get("sw"):
        tags.append("outlined")
    if meta.get("ruby"):
        tags.append("furigana")
    if meta.get("cat") == "mincho":
        tags.append("mincho")
    if meta.get("fs", 99) < 16:
        tags.append("small")
    return tags


def load_synth_val(path=SYNTH_VAL, limit=SYNTH_VAL_LINES):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    metas = json.loads(str(z["meta"])) if "meta" in z else [{}] * len(labels)
    n = len(labels) if not limit else min(limit, len(labels))
    lines = []
    for i in range(n):
        h, w = (int(v) for v in shapes[i])
        g = data[offs[i]:offs[i + 1]].reshape(h, w)
        lines.append(Line(id=f"synth-{i:05d}", set="synth-val", vn=None, title=None, image=None,
                          box=None, text=labels[i], styles=synth_styles(metas[i]), uncertain=False,
                          region=None, region_id=None,
                          crop_rgb=np.repeat(g[:, :, None], 3, axis=2)))
    return lines


def load(name, data_dir, pad=None, limit=None):
    if name == "synth-val":
        return load_synth_val(limit=limit if limit is not None else SYNTH_VAL_LINES), None
    lines, cache = load_real(name, data_dir, pad)
    return (lines[:limit] if limit else lines), cache


def regions(name, data_dir):
    """Regions of a real set for finder mode: [(image path, region box, [manifest rows])]
    with every row of the region, uncertain ones included (as don't-care)."""
    from fetch import shot_path
    out = {}
    for r in read_manifest(name):
        if not r.get("region"):
            continue
        path = str(shot_path(data_dir, SETS[name], r["vn"], r["url"]))
        out.setdefault((path, tuple(r["region"])), []).append(r)
    return [(p, list(reg), rows) for (p, reg), rows in out.items()]

"""Generated fixtures for the VOCR3 harness tests: random-init W8I blobs and procedural captures.

Nothing here reads a protected set, a screenshot or a trained model. The blobs come from
train/model.py with random weights (so their text is meaningless but deterministic) and the
images are drawn bar patterns that the C line finder detects as text rows.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "runtime"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

VOCAB = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ漢字名！？")


def fixture_blob(path, seed=0, width=0.25):
    """A deterministic random-init int8 blob (exported exactly as train/export.py --int8 does)."""
    import torch
    from torch import nn

    import export
    from model import TinyVNOCR
    torch.manual_seed(seed)
    m = TinyVNOCR(len(VOCAB) + 1, width=width).eval()
    with torch.no_grad():                 # He-normal convs and non-trivial BN: input-dependent text
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.uniform_(0.5, 2.0)
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, 0, (2.0 / mod.weight[0].numel()) ** 0.5)
    blob, _ = export.build_blob(export.model_ops(m, 32), VOCAB, 32, True, 1024, b"")
    Path(path).write_bytes(blob)
    return Path(path), hashlib.sha256(blob).hexdigest()


def draw_rows(rgb, rows, dark=False, seed=0):
    """Text-like rows of glyph blocks: rows = [(x, y, w, h)] in image coordinates."""
    rng = np.random.default_rng(seed)
    ink = 20 if dark else 235
    for x, y, w, h in rows:
        cx = x
        while cx + h <= x + w:
            g = rng.integers(0, 2, (h, h)).astype(bool)
            g[: h // 4] = False
            block = rgb[y:y + h, cx:cx + h]
            block[g] = ink
            cx += h + max(2, h // 5)
    return rgb


def screen(dims, region, rows, dark=False, seed=0):
    w, h = dims
    rgb = np.full((h, w, 3), 40, np.uint8)
    x, y, rw, rh = region
    rgb[y:y + rh, x:x + rw] = 230 if dark else 25
    return draw_rows(rgb, rows, dark, seed)


def local_set(directory, n_regions=4, vns=("v900001", "v900002", "v900003"), split="test"):
    """A private-capture-style manifest (eval/sets.register_local) with generated 960x544 images:
    each image has one dialogue region with a name row and two body rows."""
    from PIL import Image
    directory = Path(directory)
    (directory / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    texts = ["あいうえおかきく", "さしすせそたちつてと", "なにぬねのはひふ", "漢字のなまえ", "かきくけこ"]
    for i in range(n_regions):
        vn = vns[i % len(vns)]
        region = [40, 360, 880, 160]
        name_box = [60, 372, 120, 22]
        body1 = [60, 404, 600, 26]
        body2 = [60, 440, 520, 26]
        rgb = screen((960, 544), region, [name_box, body1, body2], dark=(i % 2 == 1), seed=i)
        name = f"images/frame{i:02d}.png"
        Image.fromarray(rgb).save(directory / name)
        sha = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for j, (box, styles) in enumerate(((name_box, ["name"]), (body1, []), (body2, []))):
            rows.append({"id": f"fx-{i:02d}-{j}", "vn": vn, "title": vn, "session": f"fixture-{vn}",
                         "split": split, "hand_checked": True, "image": name, "sha256": sha, "dims": [960, 544],
                         "region": region, "box": box, "text": texts[(i + j) % len(texts)],
                         "styles": styles + (["dark-on-light"] if i % 2 else ["bright-on-dark"])})
    manifest = directory / "fixture-test.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return manifest, rows

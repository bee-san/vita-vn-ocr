"""Renderer smoke: synth/render.py writes a valid shard from the pinned OFL font and the CI
fixture corpus (in a sandbox copy, ci/sandbox.py), and train.py can load it. Plus the
line finder and the integer luma that the crops go through."""
import json

import numpy as np
import pytest

import fonts
import sandbox
import train as T
from common import find_lines, load_vocab, to_gray

VOCAB = load_vocab()


@pytest.fixture(scope="module")
def shard(font, tmp_path_factory):
    repo = sandbox.build(tmp_path_factory.mktemp("render") / "repo", font)
    return sandbox.render(repo, shards=1, per_shard=24, seed=1234, workers=1, timeout=300)[0]


def test_shard_format(shard):
    """The keys and layout train.py's load() relies on."""
    z = np.load(shard)
    assert {"data", "offsets", "shapes", "labels"} <= set(z.files)
    labels = json.loads(str(z["labels"]))
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    n = len(labels)
    assert n >= 24
    assert shapes.shape == (n, 2) and offs.shape == (n + 1,)
    assert data.dtype == np.uint8 and offs[0] == 0 and offs[-1] == data.size
    assert (np.diff(offs) == shapes[:, 0] * shapes[:, 1]).all()
    assert (shapes[:, 0] >= 8).all() and (shapes[:, 1] >= 8).all()
    if "meta" in z.files:                                   # the sandbox has exactly one font
        meta = json.loads(str(z["meta"]))
        assert len(meta) == n and {m.get("fam") for m in meta} <= {fonts.FAMILY, None}


def test_labels_are_fixture_text_in_the_vocab(shard):
    labels = json.loads(str(np.load(shard)["labels"]))
    corpus = "\n".join(s for pool in sandbox.fixture_lines().values() for s in pool)
    assert all(lab and set(lab) <= set(VOCAB) for lab in labels)
    # Most labels are verbatim pieces of the fixture sentences (render.py can add brackets,
    # ellipses or a speaker name), which shows the text came from the sandbox's data/.
    verbatim = [lab for lab in labels if lab in corpus]
    assert len(verbatim) >= len(labels) // 2, labels


def test_train_loader_reads_the_shard(shard):
    char2id = {c: i + 1 for i, c in enumerate(VOCAB)}
    imgs, labels, dropped = T.load([shard], char2id)
    assert imgs and dropped <= len(imgs) // 10
    assert all(g.shape[0] == 32 and 8 <= g.shape[1] <= T.MAX_W for g in imgs)


def test_fixture_corpus_is_in_vocab_and_font(font):
    vocab = set(VOCAB)
    sandbox.check_fixture(sandbox.fixture_lines(), vocab, font)    # raises if not


@pytest.mark.parametrize("finder", ["grad", "legacy"])
def test_find_lines_on_bright_text_bars(finder):
    """Two bright bars on a dark box: one crop per bar, each containing its bar. The bars
    carry 1-px dark strokes every 6 px like glyphs do: grad (the default) finds text by its
    texture, and a solid block has contrast only along its outline."""
    img = np.full((80, 200, 3), 30, np.uint8)
    bars = [(20, 10, 130, 20), (20, 45, 80, 20)]           # x, y, w, h
    for x, y, w, h in bars:
        img[y:y + h, x:x + w] = 240
        img[y:y + h, x + 2:x + w - 2:6] = 30
    boxes = find_lines(img, finder)
    assert boxes is not None and len(boxes) == len(bars), boxes
    for (bx, by, bw, bh), (x, y, w, h) in zip(sorted(boxes, key=lambda b: b[1]), bars):
        assert bx <= x and by <= y and bx + bw >= x + w and by + bh >= y + h
        assert bh < 2 * h + 16, "a crop should hold one line, not both"


def test_to_gray_matches_the_c_luma():
    rgb = np.array([[[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 20, 30]]], np.uint8)
    expect = [(77 * r + 150 * g + 29 * b) >> 8 for r, g, b in rgb[0].tolist()]
    assert to_gray(rgb)[0].tolist() == expect == [255, 76, 149, 28, 18]

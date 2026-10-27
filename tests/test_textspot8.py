"""textspot8 (docs/textspot8.md): the trainer's additions (train/textspot8_train.py) and the MeikiOCR reference's
box decode (eval/textspot8.py). No frames, models or shards needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "eval", ROOT / "teacher", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")
import textspot6_train as B  # noqa: E402
import textspot8_train as T  # noqa: E402

LINES = [[20, 30, 300, 52, 0, 1.0, 1], [20, 60, 250, 82, 0, 0.5, 1], [400, 10, 420, 200, 1, 1.0, 2]]
IGN = [[0, 200, 50, 220]]


@pytest.fixture
def t8():
    saved = dict(T.T8)
    yield T.T8
    T.T8.clear()
    T.T8.update(saved)


def test_identity_without_changes(t8):
    t8.update(edge_weight=1.0, across_shrink=0.0)
    a = T.targets(LINES, IGN, 512, 256)
    b = T._base_targets(LINES, IGN, 512, 256)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_edge_weights_only_at_line_ends(t8):
    t8.update(edge_weight=4.0, across_shrink=0.0, edge_cells=1)
    K, O, A1, A2, v, kern, W, M = T.targets(LINES, IGN, 512, 256)  # noqa: E741
    _, _, _, _, _, kern0, W0, _ = T._base_targets(LINES, IGN, 512, 256)
    assert np.array_equal(kern, kern0)
    changed = W != W0
    assert changed.any() and (changed <= kern).all()
    assert np.allclose(W[changed], 4.0 * W0[changed])
    # horizontal line 1 spans columns 5..74 at stride 4: its ends are columns 5 and 74
    cols = set(np.nonzero(changed[:30])[1].tolist())
    assert {5, 74, 62} <= cols and not cols & set(range(6, 62))


def test_edge_weight_skips_ends_cut_by_the_window(t8):
    t8.update(edge_weight=4.0, across_shrink=0.0, edge_cells=1)
    cut = [[-40, 30, 200, 52, 0, 1.0, 1]]                 # starts left of the window: only its right end is real
    _, _, _, _, _, kern, W, _ = T.targets(cut, [], 512, 256)
    cols = set(np.nonzero((W > 1) & kern)[1].tolist())
    assert cols == {(200 - 1) // 4}


def test_across_shrink_moves_the_extent_targets(t8):
    t8.update(edge_weight=1.0, across_shrink=1.0)
    _, _, A1, A2, _, kern, _, _ = T.targets(LINES, IGN, 512, 256)
    _, _, B1, B2, _, _, _, _ = T._base_targets(LINES, IGN, 512, 256)
    assert np.allclose(B1[kern] - A1[kern], 1.0) and np.allclose(B2[kern] - A2[kern], 1.0)


def test_icons_never_cover_text_or_ignore_areas():
    rng = np.random.default_rng(3)
    pos = [[20, 30, 300, 52, 0, 1.0, 1], [20, 60, 250, 82, 0, 1.0, 1]]
    for _ in range(50):
        img = np.full((256, 512), 40, np.uint8)
        before = img.copy()
        drawn = T.add_icons(img, pos, IGN, rng)
        for x0, y0, x1, y1 in drawn:
            for b in pos + IGN:
                assert not (x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1)
        for b in pos + IGN:
            assert np.array_equal(img[b[1]:b[3], b[0]:b[2]], before[b[1]:b[3], b[0]:b[2]])


def test_every_icon_kind_draws():
    rng = np.random.default_rng(0)
    for kind in T.ICON_KINDS:
        z = np.zeros((40, 40), np.uint8)
        T.draw_icon(z, kind, 20, 20, 24, 255, rng)
        assert (z > 0).sum() > 20, kind


def test_tight_crop_keeps_the_block_and_shifts_boxes():
    rng = np.random.default_rng(1)
    img = np.arange(256 * 512, dtype=np.uint32).reshape(256, 512).astype(np.uint8)
    pos = [[20, 30, 300, 52, 0, 1.0, 1], [20, 60, 250, 82, 0, 1.0, 1]]
    for _ in range(20):
        im2, p2, i2 = T.tight_crop(img, pos, IGN, rng)
        dx, dy = pos[0][0] - p2[0][0], pos[0][1] - p2[0][1]
        assert np.array_equal(im2, img[dy:dy + im2.shape[0], dx:dx + im2.shape[1]])
        for a, b in zip(pos, p2):
            assert [a[0] - dx, a[1] - dy, a[2] - dx, a[3] - dy] == b[:4] and a[4:] == b[4:]
        for b in p2:                                    # the block is inside the cut
            assert b[0] >= 0 and b[1] >= 0 and b[2] <= im2.shape[1] and b[3] <= im2.shape[0]


def test_apply_settings_registers_the_recipe_and_functions():
    T.apply_settings()
    assert "t8" in B.RECIPES and B.targets is T.targets and B.make_window is T.make_window
    assert B.AUG["zoom"] == list(T.T8["zoom"]) and "train/textspot8_train.py" in B.code_sha256()


def test_meiki_decode_orders_pads_and_thresholds():
    import textspot8 as E8
    grey = np.full((544, 960), 30, np.uint8)
    grey[100:120, 100:400] = 220
    grey[130:150, 100:360] = 220
    rgb = np.repeat(grey[:, :, None], 3, 2)
    rec = {"frame": "f", "view": "vita", "W": 960, "H": 544,
           "meiki": [[100, 130, 360, 150, 0.9], [100, 100, 400, 120, 0.8], [600, 10, 700, 30, 0.4]]}
    row = E8.decode_meiki(rec, rgb, grey, thr=0.5, decode="raw")
    assert [b[:4] for b in row["boxes"]] == [[100, 100, 300, 20], [100, 130, 260, 20]]
    assert [b[5] for b in row["boxes"]] == [102, 114] and all(b[8] == 0 for b in row["boxes"])
    row = E8.decode_meiki(rec, rgb, grey, thr=0.5, decode="pad")
    assert [b[:4] for b in row["boxes"]] == [[96, 96, 308, 28], [96, 126, 268, 28]]
    row = E8.decode_meiki(dict(rec, view="psp"), rgb, grey, thr=0.3, decode="pad")
    assert len(row["boxes"]) == 3 and row["boxes"][0][:4] == [598, 8, 104, 24]


def test_shift_boxes(tmp_path):
    import textspot8 as E8
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"uid": "u1", "boxes_xywh": [[10, 10, 100, 20], [0, 1, 50, 3]], "dark": [0, 1]}) + "\n")
    out = tmp_path / "out.jsonl"
    E8.cmd_shift_boxes(type("A", (), {"rows": str(rows), "dx": 0, "dy": 2, "out": str(out)}))
    got = json.loads(out.read_text())
    assert got == {"unit": "u1", "boxes": [[10, 8, 100, 24, 0], [0, 0, 50, 6, 1]]}
    E8.cmd_shift_boxes(type("A", (), {"rows": str(rows), "dx": 0, "dy": -2, "out": str(out)}))
    got = json.loads(out.read_text())
    assert got["boxes"] == [[10, 12, 100, 16, 0], [0, 1, 50, 2, 1]]

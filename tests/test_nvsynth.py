"""nvsynth/extract.py and fetch.py: line filters, ink box, grey visibility, crops and the shard format.

Pages are drawn in memory (rectangles stand in for glyph rows), so nothing is downloaded.
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for d in ("nvsynth", "synth", "train"):
    sys.path.insert(0, str(ROOT / d))

import extract as ex  # noqa: E402
import fetch  # noqa: E402

VOCAB = ex.vocab_set()


def line(text, x, y, w, h, quad=None):
    quad = quad or [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return {"text": text, "bbox": [x, y, w, h], "quad": quad}


def page(w=400, h=200, bg=(235, 235, 235), ink=(20, 20, 20), rows=((40, 60, 30, 300),)):
    """RGB page with one dark band per (y0, ink height, x0, width) standing in for a text line."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    rng = np.random.default_rng(0)
    for y0, ih, x0, iw in rows:
        band = rng.random((ih, iw)) < 0.45                       # glyph-like texture, not a solid block
        img[y0:y0 + ih, x0:x0 + iw][band] = ink
    return img


def jpeg(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=100)
    return buf.getvalue()


def test_check_keeps_a_plain_horizontal_japanese_line():
    assert ex.check(line("こんにちは、世界。", 10, 10, 200, 30), 400, 200, VOCAB) == (None, "こんにちは、世界。")


@pytest.mark.parametrize("ln, reason", [
    (line("縦書きの行", 10, 10, 30, 150), "vertical"),
    (line("回転した行", 10, 10, 200, 30, quad=[[10, 10], [210, 30], [210, 60], [10, 40]]), "rotated"),
    (line("はみ出した行", 300, 10, 200, 30), "clipped"),
    (line("小さい文字", 10, 10, 100, 10), "small"),
    (line("あ", 10, 10, 30, 30), "short"),
    (line("価格は€5です", 10, 10, 200, 30), "oov"),
    (line("Hello World", 10, 10, 200, 30), "ascii"),
    (line("ABCDEFGHIJあ", 10, 10, 200, 30), "ascii"),
])
def test_check_rejects(ln, reason):
    assert ex.check(ln, 400, 200, VOCAB)[0] == reason


def test_multi_word_lines_are_rejected_but_still_counted_for_oov():
    ln = dict(line("76 そのため、安心してください。", 10, 10, 300, 30), word_indices=[0, 1])
    assert ex.check(ln, 400, 200, VOCAB) == ("words", "76そのため、安心してください。")
    assert ex.check(dict(ln, word_indices=[3]), 400, 200, VOCAB)[0] is None


def test_labels_use_the_corpus_normalisation():
    # spaces (half and full width) go, 〜 becomes ～, half-width katakana become full-width
    assert ex.check(line("見え ます　〜ｶﾀｶﾅ", 10, 10, 300, 30), 400, 200, VOCAB) == (None, "見えます～カタカナ")


def test_ink_box_finds_the_glyph_rows_inside_a_loose_box():
    rgb = page(rows=((40, 20, 30, 300),))                       # ink rows 40..59, cols 30..329
    gray = ex.to_gray(rgb)
    (x0, y0, x1, y1), clean = ex.ink_box(gray, [25, 33, 320, 34])
    assert clean
    assert abs(y0 - 40) <= 1 and abs(y1 - 60) <= 1
    assert abs(x0 - 30) <= 1 and abs(x1 - 330) <= 1


def test_ink_box_falls_back_to_the_median_trim_on_a_blank_box():
    gray = np.full((200, 400), 128, np.uint8)
    (x0, y0, x1, y1), clean = ex.ink_box(gray, [20, 50, 200, 30])
    assert not clean and (x0, x1) == (20, 220) and 50 < y0 < y1 < 80


def test_crop_is_grey_height_32_with_a_margin_proportional_to_the_ink():
    rgb = page(rows=((40, 20, 30, 300),))
    g, meta = ex.crop(rgb, ex.to_gray(rgb), line("テスト用の行です", 25, 33, 320, 34), "テスト用の行です",
                      np.random.default_rng(1))
    assert g.dtype == np.uint8 and g.shape[0] == 32
    x, y, w, h = meta["crop"]
    pad = 40 - y
    assert 0.1 * 20 - 1 <= pad <= 0.24 * 20 + 1 and h == 20 + 2 * pad
    assert meta["polarity"] == "dark-on-light" and meta["contrast"] > 100
    assert abs(g.shape[1] - round(w * 32 / h)) <= 1


def test_tall_crops_take_the_area_path_and_still_end_at_height_32():
    rgb = page(w=1200, h=400, rows=((50, 120, 40, 1000),))
    g, meta = ex.crop(rgb, ex.to_gray(rgb), line("大きな見出しです", 30, 30, 1020, 160), "大きな見出しです",
                      np.random.default_rng(2))
    assert g.shape[0] == 32 and meta["ink_h"] > ex.AREA_ABOVE


def test_grey_visibility_filters():
    gray_bg = page(bg=(128, 128, 128), ink=(128, 128, 128))
    g, why = ex.crop(gray_bg, ex.to_gray(gray_bg), line("見えない行です", 25, 33, 320, 34), "見えない行です",
                     np.random.default_rng(0))
    assert g is None and why == "faint"
    # same luma, different hue: (77 R + 150 G + 29 B) >> 8 is 113 and 114
    hue = page(bg=(200, 80, 60), ink=(55, 150, 90))
    lumas = ex.to_gray(np.array([[[200, 80, 60], [55, 150, 90]]], np.uint8))[0]
    assert lumas.tolist() == [113, 114]
    band_rgb, band_grey = hue[40:60, 30:330], ex.to_gray(hue)[40:60, 30:330]
    assert ex.grey_ratio(band_rgb, band_grey) < ex.GREY_RATIO
    bw = page()
    assert ex.grey_ratio(bw[40:60, 30:330], ex.to_gray(bw)[40:60, 30:330]) > 0.9


def test_process_page_is_deterministic_and_counts_reasons():
    rgb = page(rows=((30, 20, 20, 300), (110, 20, 20, 250)))
    ann = {"line_bboxes": [line("一行目のテキスト", 15, 23, 310, 34), line("二行目のテキスト", 15, 103, 260, 34),
                           line("縦", 350, 10, 20, 150), line("縦書きです", 360, 10, 20, 150)]}
    runs = []
    for _ in range(2):
        stats, oov = ex.new_stats(), Counter()
        out = ex.process_page(jpeg(rgb), json.dumps(ann), [200, 400], 7, "train_000", 1, 5, VOCAB, stats, oov)
        runs.append([(lab, m["crop"]) for _, lab, m in out])
    assert runs[0] == runs[1] and len(runs[0]) == 1
    assert stats["pages"] == 1 and stats["lines"] == 4 and stats["vertical"] == 1 and stats["short"] == 1
    assert stats["usable"] == 2 and stats["kept"] == 1 and stats["pages_kept"] == 1
    all_lines = ex.process_page(jpeg(rgb), ann, [200, 400], 7, "train_000", 5, 5, VOCAB, ex.new_stats(), Counter())
    assert sorted(lab for _, lab, _ in all_lines) == ["一行目のテキスト", "二行目のテキスト"]


def test_oov_statistics_cover_every_horizontal_line():
    rgb = page()
    ann = {"line_bboxes": [line("価格は€5です", 10, 10, 200, 30), line("¥100です", 10, 60, 200, 30)]}
    stats, oov, chars = ex.new_stats(), Counter(), Counter()
    ex.process_page(jpeg(rgb), ann, [200, 400], 0, "t", 1, 0, VOCAB, stats, oov, chars)
    assert oov == Counter({"€": 1, "¥": 1}) and stats["h_oov_lines"] == 2 and stats["h_lines"] == 2
    assert chars == Counter("価格は€5です") + Counter("¥100です") and stats["h_chars"] == sum(chars.values())


def test_shards_load_in_train_py_unchanged(tmp_path):
    import train as T
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (32, w), dtype=np.uint8) for w in (40, 96, 300)]
    labels = ["あい", "テスト", "これは行です"]
    ex.save_shard(tmp_path / "nv_train_000.npz", crops, labels, [{"src": "x"}] * 3)
    char2id = {c: i + 1 for i, c in enumerate(ex.load_vocab())}
    imgs, ids, dropped = T.load([str(tmp_path / "nv_train_000.npz")], char2id)
    assert dropped == 0 and len(imgs) == 3
    for a, b in zip(imgs, crops):
        assert np.array_equal(a, b)                              # height 32 already: resize32 is a no-op
    assert ["".join(ex.load_vocab()[i - 1] for i in x) for x in ids] == labels


def test_fetch_helpers():
    assert fetch.parse_range("0-2,5", 10) == [0, 1, 2, 5]
    assert fetch.parse_range("", 3) == [0, 1, 2]
    b = fetch.Budget(10)
    b.take(6)
    b.give(6)
    b.take(25)                                                  # a single file over the limit still fits alone
    assert b.used == 25


def test_oov_report_character_classes():
    import oov_report as rep
    assert rep.in_jis0208("亜") and rep.in_jis0208("…") and rep.in_jis0208("ぁ")
    assert not rep.in_jis0208("¥") and not rep.in_jis0208("—") and not rep.in_jis0208("A")
    assert rep.kind("麺") == "kanji, JIS X 0208" and rep.kind("々") == "kanji, JIS X 0208"
    assert rep.kind("髙") == "kanji, not in JIS X 0208" and rep.kind("𠮟") == "kanji, not in JIS X 0208"
    assert rep.kind("^") == "ASCII" and rep.kind("이") == "Hangul" and rep.kind("\u0301") == "combining mark"

"""realdata/: label normalisation, the voting rule, the v5s cascade, line geometry and the
committed manifests. Fast, no OCR models, no network, no screenshots."""
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("realdata", "synth"):
    sys.path.insert(0, str(ROOT / d))

import build  # noqa: E402
import lines as L  # noqa: E402
import vndb  # noqa: E402
import vote  # noqa: E402
from common import is_test_vn, load_vocab  # noqa: E402


@pytest.mark.parametrize("raw, label", [
    ("．．．妙なヤツだ", "…妙なヤツだ"),                # manga-ocr's dots: one … per three
    ("......。", "……。"),
    ("‥‥えっ", "……えっ"),                             # ‥ is one cell
    ("てちょうだい!!」", "てちょうだい！！」"),
    ("だもんね~", "だもんね～"),
    ("--そうか", "――そうか"),
    ("Re-start", "Re-start"),                         # a hyphen inside a word stays
    ("ＯＫ！２人", "OK！2人"),
    ("ﾊﾟﾝ", "パン"),
    ("「あ、 い」", "「あ、い」"),
    ("ボナ・ヨルナ", "ボナ・ヨルナ"),
    ("·犯人は", "・犯人は"),
    ("進める。▽", "進める。"),                         # page cursor dropped
    ("(笑)", "（笑）"),
])
def test_canon(raw, label):
    assert vote.canon(raw) == label


def test_keys():
    assert vote.key("てちょうだい！！") == vote.key("てちょうだい!!")
    assert vote.key("……") != vote.key("…")
    assert vote.loose("まぁいい") == vote.loose("まあいい")
    assert vote.loose("がお一がおー") == vote.loose("がおーがおー")
    assert vote.loose("……っ") == vote.loose("…つ")
    assert vote.japanese("「……決めた」") and not vote.japanese("SKIP") and not vote.japanese("©2015 FIFS")


def test_norm_matches_build_corpus():
    pytest.importorskip("pyarrow")
    import build_corpus
    for s in ("ﾊﾟﾝ　です〜—", "a\u00a0b\tc", "\u200bx\x07y"):
        assert vote.norm(s) == build_corpus.norm(s)


RULE = vote.RULES[vote.DEFAULT_RULE]


def test_rule_needs_three_supporters():
    t = "「そう驚くようなことなのか」"
    assert RULE({"v6s": t, "v5s": t, "manga": t, "meiki": "x"})[0] == t
    assert RULE({"v6s": t, "v5s": t, "manga": "y", "meiki": "x"})[0] is None
    # manga-ocr and meiki may add brackets or a full stop at the crop edge
    assert RULE({"v6s": "頭がすっきりしてるわ。", "v5s": "z", "manga": "【頭がすっきりしてるわ。",
                 "meiki": "「頭がすっきりしてるわ。」"})[0] == "頭がすっきりしてるわ。"
    # ...but a missing leading nakaguro is a real difference
    assert RULE({"v6s": "犯人は", "v5s": "x", "manga": "・犯人は", "meiki": "y"})[0] is None


def test_rule_ellipsis_counts_come_from_v6s():
    lab = RULE({"v6s": "……っつっても", "v5s": "…っつっても", "manga": "．．．っつっても",
                "meiki": "……っつっても"})[0]
    assert lab == "……っつっても"


def test_rule_vetoes_confusable_dissent():
    # three readers say 一, PP-OCRv5 server says ー: a dissent on exactly the confusable glyph
    wrong = "「がお一がおー」"
    reads = {"v6s": wrong, "meiki": wrong, "manga": wrong, "v5s": "「がおーがおー」"}
    assert RULE(reads) == (None, "confusable-dissent")
    reads = {"v6s": "まあいいでしょう。", "v5s": "まあいいでしょう。", "manga": "まあいいでしょう。",
             "meiki": "まぁいいでしょう。"}
    assert RULE(reads)[0] is None


def test_v5s_cascade_never_changes_the_decision():
    """label.py skips v5s when v5s_can_matter() says so: check that no v5s reading could
    have changed the decision, over a grid of readings with the usual confusions."""
    pool = ["ジェイク様だ！」", "「ジェイク様だ！」", "ジエイク様だ!」", "ジェイク様だ", "ジェイク様だ！」。",
            "……決めた", "…決めた", "「……決めた」", "決めた", ""]
    for v6, mg, mk in itertools.product(pool, repeat=3):
        reads = {"v6s": v6, "manga": mg, "meiki": mk}
        if vote.v5s_can_matter(reads):
            continue
        base = RULE(reads)[0]
        for v5 in pool:
            assert RULE({**reads, "v5s": v5})[0] == base, (reads, v5)


def text_image(polarity, h=60, w=400):
    bg, fg = (30, 200) if polarity == "light" else (225, 40)
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.putText(img, "VN LINE 123", (60, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (fg, fg, fg), 2, cv2.LINE_AA)
    ys, xs = np.nonzero(np.abs(img[..., 0].astype(int) - bg) > 40)
    return img, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


@pytest.mark.parametrize("polarity", ["light", "dark"])
def test_ink_box_either_polarity(polarity):
    img, (x0, y0, x1, y1) = text_image(polarity)
    det = [x0 - 6, y0 - 5, x1 + 4, y1 + 6]                 # a detector box with some slack
    got = L.ink_box(img, det)
    assert got is not None
    assert all(abs(a - b) <= 2 for a, b in zip(got, (x0, y0, x1, y1))), (got, (x0, y0, x1, y1))
    assert L.polarity(img[got[1]:got[3], got[0]:got[2]]) == ("light-on-dark" if polarity == "light"
                                                              else "dark-on-light")


def test_ink_box_rejects_text_cut_by_the_screen_edge():
    img, (x0, y0, x1, y1) = text_image("light")
    assert L.ink_box(img[:, x0 + 5:], [0, y0 - 5, x1 - x0, y1 + 5]) is None


def test_merge_rows_joins_a_split_line_only():
    a, b, c = [10, 100, 200, 130], [215, 101, 400, 131], [10, 140, 300, 170]
    rows = L.merge_rows([a, b, c])
    assert sorted(map(tuple, rows)) == [(10, 100, 400, 131), (10, 140, 300, 170)]


def test_train_box_pads_like_the_line_finder():
    assert L.train_box((100, 50, 300, 74), 1.0, 960, 544) == (96, 46, 304, 79)
    assert L.train_box((2, 1, 300, 541), 1.0, 960, 544) == (0, 0, 304, 544)
    assert L.train_box((200, 100, 600, 140), 0.375, 480, 270) == (71, 33, 229, 58)
    assert L.target_scale(1280, 720, "vita") == 0.75 and L.target_scale(1280, 720, "psp") == 0.375


def test_val_split_is_disjoint_from_the_test_hash():
    vns = [f"v{i}" for i in range(1, 6001)]
    val = [v for v in vns if build.split_of(v) == "val"]
    assert not any(is_test_vn(v) for v in val)
    assert 0.08 < len(val) / len(vns) < 0.12


def test_kind_of():
    row = {"platforms": ["psv"], "dims": [960, 544]}
    assert vndb.kind_of(row) == "vita"
    assert vndb.kind_of({"platforms": ["psp"], "dims": [480, 272]}) == "psp"
    assert vndb.kind_of({"platforms": ["win"], "dims": [1280, 720]}) == "hires"
    assert vndb.kind_of({"platforms": ["win"], "dims": [1920, 600]}) is None     # too wide
    assert vndb.kind_of({"platforms": ["psv", "win"], "dims": [1280, 720]}) is None


def test_committed_manifests_are_consistent():
    shots = {vndb.shot_key(s): s for s in vndb.load_shots()}
    assert shots, "no shots manifests"
    extra = vndb.excluded_vns()
    for s in shots.values():
        assert len(s["sha256"] or "") == 64, s
        assert s["sexual"] < vndb.SEXUAL_MAX and s["violence"] < vndb.VIOLENCE_MAX, s
        assert not is_test_vn(s["vn"]) and s["vn"] not in extra, s
    rows = build.load_lines()
    vocab = set(load_vocab())
    for r in rows:
        assert r["id"].rsplit("_", 1)[0] in shots, r["id"]
        assert r["label"] and set(r["label"]) <= vocab, r
        assert r["split"] == build.split_of(shots[r["id"].rsplit("_", 1)[0]]["vn"]), r
        assert set(r["targets"]) <= {"vita", "psp"} and r["targets"], r
    hand = [json.loads(x) for x in build.HAND.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len({h["id"] for h in hand}) == len(hand)

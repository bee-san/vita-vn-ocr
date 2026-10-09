"""Fast tests of the teacher labeller's pure parts: label form, agreement, the rule, geometry,
matching to eval's ground truth. No OCR engine or model file is needed."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "teacher"))
sys.path.insert(0, str(ROOT / "synth"))
import build  # noqa: E402
import geometry as G  # noqa: E402
import measure  # noqa: E402
import vote  # noqa: E402


@pytest.mark.parametrize("raw,want", [
    ("「ええと...厳密に言うと」", "「ええと…厳密に言うと」"),
    ("そうなんだ・・・", "そうなんだ…"),
    ("待って……！", "待って……！"),
    ("本当?!", "本当？！"),
    ("だもんね~", "だもんね～"),
    ("ﾊﾞｶ", "バカ"),
    ("１年は経ってる", "1年は経ってる"),
    ("渋谷 友千香", "渋谷友千香"),
    ("そう──だね", "そう――だね"),
    ("Re-start", "Re-start"),
    ("おやすみ▼", "おやすみ"),
    ("ジョン・スミス", "ジョン・スミス"),
])
def test_canon(raw, want):
    assert vote.canon(raw) == want


def test_key_matches_eval_norm():
    """vote.eval_norm must stay identical to eval/metrics.norm (agreement = same score)."""
    path = ROOT / "eval" / "metrics.py"
    if not path.exists():
        pytest.skip("eval/ not in this checkout")
    spec = importlib.util.spec_from_file_location("eval_metrics", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for s in ["「ええと...厳密に」", "ﾊﾞｶ！？", "・・・・・・", "そう――だね〜", "１２３ ＡＢＣ", "ー一―", "まぁ", "‥…"]:
        assert vote.eval_norm(s) == m.norm(s), s


def test_agreement_is_strict_on_confusables():
    assert vote.key("まあ") != vote.key("まぁ")
    assert vote.key("ゲーム") != vote.key("ゲ一ム")
    assert vote.key("本当!?") == vote.key("本当！？")
    assert vote.loose("まあ") == vote.loose("まぁ")
    assert vote.japanese("こんにちは") and not vote.japanese("Copyright 2012") and not vote.japanese("60/60")


def line(meikiv="こんにちは", conf=0.95, **reads):
    chars = [[c, 10 * i, 0, 10 * i + 9, 20, conf] for i, c in enumerate(meikiv)]
    return {"k": 0, "det_conf": 0.9, "ink": [0, 0, 100, 20], "targets": ["vita"], "edge": False,
            "meiki": {"text": meikiv, "chars": chars},
            "reads": {"meikiv": [meikiv, conf, conf, chars], **{k: [v, 0.9, 0.9] for k, v in reads.items()}}}


def test_accept_votes_and_thresholds():
    ln = line(v6s="こんにちは", manga="こんにちわ", screenai="こんにちは")
    assert vote.accept(ln, {"votes": 2}) == ("こんにちは", "meikiv+v6s+screenai")
    assert vote.accept(ln, {"votes": 3})[0] is None
    assert vote.accept(ln, {"votes": 2, "min_conf": 0.99}) == (None, "low-conf")
    assert vote.accept(ln, {"votes": 2, "det_conf": 0.95}) == (None, "low-det")
    assert vote.accept(dict(ln, edge=True), {"votes": 1}) == (None, "screen-edge")
    assert vote.accept(dict(ln, targets=[]), {"votes": 1}) == (None, "size")
    assert vote.accept(line("60/60", v6s="60/60"), {"votes": 1}) == (None, "not-japanese")


def test_accept_veto_on_confusable_dissent():
    ln = line("まあね", v6s="まあね", manga="まあね", screenai="まぁね")
    assert vote.accept(ln, {"votes": 2})[0] == "まあね"
    assert vote.accept(ln, {"votes": 2, "veto": True}) == (None, "confusable-dissent")


def test_undecided_never_skips_a_voter_that_could_change_the_result():
    import itertools
    rule = {"voters": ("v6s", "manga", "screenai", "v5s"), "votes": 2, "veto": True}
    texts = ["まあね", "まぁね", "ちがう", None]
    for combo in itertools.product(texts, repeat=4):
        reads = {v: t for v, t in zip(rule["voters"], combo) if t is not None}
        ln = line("まあね", **reads)
        full = vote.accept(ln, rule)
        for v in rule["voters"]:                    # any single read still missing...
            if v in reads:
                continue
            if not vote.undecided(ln, rule):        # ...was declared unable to matter: check it
                for t in texts[:3]:
                    assert vote.accept(line("まあね", **reads, **{v: t}), rule)[0] == full[0], (reads, v, t)


def test_merge_rows_joins_fragments_left_to_right():
    a = {"det": [100, 10, 200, 40], "det_conf": 0.9, "vertical": False, "text": "「ええと",
         "chars": [["「", 100, 10, 110, 40, 0.9]]}
    b = {"det": [220, 12, 300, 41], "det_conf": 0.8, "vertical": False, "text": "厳密に」",
         "chars": [["厳", 220, 12, 240, 41, 0.9]]}
    c = {"det": [100, 60, 300, 90], "det_conf": 0.9, "vertical": False, "text": "次の行", "chars": []}
    rows = G.merge_rows([b, a, c])
    assert len(rows) == 2
    top = min(rows, key=lambda r: r["det"][1])
    assert top["text"] == "「ええと厳密に」" and top["det"] == [100, 10, 300, 41] and top["det_conf"] == 0.8


def test_refine_finds_ink_on_either_polarity():
    for fg, bg in ((240, 30), (20, 230)):
        img = np.full((80, 300, 3), bg, np.uint8)
        img[30:52, 50:250] = fg                     # a "line" of ink
        img[30:52, 60:70] = bg                      # with gaps between glyphs
        box, hint = G.refine(img, [40, 22, 260, 60])
        assert box == [50, 30, 250, 52], (fg, box)
        assert G.polarity(hint) == ("bright-on-dark" if fg > bg else "dark-on-light")


def test_furigana_attaches_to_the_line_below():
    rows = [{"ink": [100, 100, 400, 130], "text": "漢字の行"}, {"ink": [120, 88, 160, 98], "text": "かんじ"},
            {"ink": [100, 150, 400, 180], "text": "次の行"}]
    keep = G.attach_furigana(rows)
    assert len(keep) == 2
    host = keep[0]
    assert host["furi"] == [[120, 88, 160, 98]] and host["crop"] == [100, 88, 400, 130]


def test_voter_view_stops_short_of_neighbours():
    ink, above, beside = [100, 100, 400, 130], [100, 60, 400, 90], [440, 100, 500, 130]
    v = G.voter_view(ink, [above, beside], 960, 544)
    assert v[1] >= 95 and v[1] <= 100          # halfway to the line above at most
    assert v[2] <= 420                          # halfway to the neighbour on the row
    assert v[0] == 100 - round(G.VIEW_X * 30)


def test_targets_and_crop_box():
    assert G.targets_of("vita", [960, 544], [0, 0, 50, 20]) == ["vita"]
    assert G.targets_of("vita", [960, 544], [0, 0, 50, 8]) == []
    assert G.targets_of("hires", [1920, 1080], [0, 0, 100, 40]) == ["vita", "psp"]
    assert G.targets_of("hires", [1920, 1080], [0, 0, 100, 24]) == ["vita"]
    assert G.crop_box([10, 10, 50, 30], 0.5, 2, 480, 272) == [3, 3, 27, 17]


def test_splits_follow_the_repo_rules():
    from common import is_test_vn
    for vn in (f"v{i}" for i in range(1, 400)):
        assert build.is_test_vn(vn) == is_test_vn(vn)
        if build.split_of(vn) == "val":
            assert not is_test_vn(vn)


def test_match_and_score():
    gts = {"v1_1": [{"id": "a", "vn": "v1", "xyxy": [100, 100, 300, 130], "text": "【名前】"},
                    {"id": "b", "vn": "v1", "xyxy": [320, 100, 600, 130], "text": "「こんにちは」"},
                    {"id": "c", "vn": "v1", "xyxy": [100, 200, 600, 230], "text": "読めない", "uncertain": True}]}
    good = line("【名前】「こんにちは」", v6s="【名前】「こんにちは」", screenai="【名前】「こんにちは」")
    good["crop"] = [98, 99, 602, 131]
    wrong = line("ありがとう", v6s="ありがとう", screenai="ありがとう")
    wrong["crop"] = [100, 200, 600, 230]
    menu = line("セーブ", v6s="セーブ", screenai="セーブ")
    menu["crop"] = [800, 500, 900, 520]
    shots = {"v1_1": {"vn": "v1", "lines": [good, wrong, menu]}}
    items, scored = measure.match(shots, gts, "vita-test")
    assert [(x[3], x[5]) for x in items] == [("【名前】「こんにちは」", "scored"), (None, "uncertain"),
                                             (None, "unmatched")]       # joined row; uncertain and menu unscored
    s = measure.score(items, scored, {"votes": 2})
    assert s["accepted"] == 3 and s["accepted_scored"] == 1 and s["precision"] == 1.0 and s["recall"] == 1.0
    whole = line("「こんにちは」", v6s="「こんにちは」", screenai="「こんにちは」")
    whole["crop"] = [318, 99, 602, 131]                            # exactly the second gt line: scored
    half = line("「こんに", v6s="「こんに", screenai="「こんに")
    half["crop"] = [318, 99, 450, 131]                             # half of it: not scorable by boxes
    items, _ = measure.match({"v1_1": {"vn": "v1", "lines": [whole, half]}}, gts, "vita-test")
    assert [(x[3], x[5]) for x in items] == [("「こんにちは」", "scored"), (None, "partial")]

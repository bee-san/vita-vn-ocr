"""Fast tests of the teacher labeller's pure parts: label form, agreement, the rule, geometry,
matching to eval's ground truth. No OCR engine or model file is needed."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
from teacher import build, measure, vote  # noqa: E402
from teacher import geometry as G  # noqa: E402


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


def test_accept_trunc_on_longer_dissent():
    ln = line("ちょっと、あんた", v6s="ちょっと、あんた", manga="ちょっと、あんた", screenai="「ちょっと、あんた")
    assert vote.accept(ln, {"votes": 2})[0] == "ちょっと、あんた"
    assert vote.accept(ln, {"votes": 2, "trunc": True}) == (None, "truncated")
    ln = line("ちょっと、あんた", v6s="ちょっと、あんた", manga="ちょっと、あんた", screenai="ちよっと、あんた")
    assert vote.accept(ln, {"votes": 2, "trunc": True})[0] == "ちょっと、あんた"     # not a truncation


@pytest.mark.parametrize("extra", [{"veto": True}, {"trunc": True}, {"veto": True, "trunc": True}, {}])
def test_undecided_never_skips_a_voter_that_could_change_the_result(extra):
    import itertools
    rule = {"voters": ("v6s", "manga", "screenai", "v5s"), "votes": 2, **extra}
    texts = ["まあね", "まぁね", "「まあね", None]
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


def test_furigana_inside_the_hosts_box_and_names_are_not_furigana():
    # the host's ink box already took the ruby in (its top is the ruby's top)
    rows = [{"ink": [100, 86, 400, 130], "text": "お父さんが最期の"}, {"ink": [130, 86, 150, 96], "text": "とう"}]
    keep = G.attach_furigana(rows)
    assert [r["text"] for r in keep] == ["お父さんが最期の"] and keep[0]["furi"] == [[130, 86, 150, 96]]
    # a small name plate above the dialogue is a line of its own, not furigana
    rows = [{"ink": [100, 100, 400, 130], "text": "「行こう」"}, {"ink": [110, 84, 150, 98], "text": "倫也"}]
    assert len(G.attach_furigana(rows)) == 2


def test_furigana_overlapping_the_hosts_top():
    # v10856: rubies at y 434-445 over a line whose ink starts at 442 (overlap of 3 px)
    host = {"ink": [53, 442, 875, 475], "det": [49, 445, 874, 475], "text": "お父さんが最期のときを"}
    rubies = [{"ink": [83, 434, 107, 445], "det": [79, 432, 107, 445], "text": "とう"},
              {"ink": [207, 434, 250, 445], "det": [201, 432, 251, 444], "text": "さいご"}]
    # v3820: the ruby's ink box ran into the host's glyph tops (20 px), its detector box is 14 px
    host2 = {"ink": [214, 417, 584, 446], "det": [214, 420, 583, 446], "text": "あはは、杏紅ったら怖い顔」"}
    ruby2 = {"ink": [341, 403, 404, 423], "det": [337, 406, 401, 420], "text": "キョウコウ"}
    keep = G.attach_furigana([host, *rubies, host2, ruby2])
    assert [r["text"] for r in keep] == ["お父さんが最期のときを", "あはは、杏紅ったら怖い顔」"]
    assert keep[0]["crop"] == [53, 434, 875, 475] and keep[1]["crop"] == [214, 403, 584, 446]


def test_refurigana_on_stored_reads():
    host = line("お父さんが最期の")
    host.update(ink=[100, 86, 400, 130], crop=[100, 86, 400, 130])
    ruby = line("とう")
    ruby.update(ink=[130, 86, 150, 96], crop=[130, 86, 150, 96])
    out = G.refurigana([host, ruby])
    assert len(out) == 1 and out[0]["furi"] == [[130, 86, 150, 96]] and out[0]["crop"] == [100, 86, 400, 130]


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


def test_build_decide_and_render_end_to_end(tmp_path):
    """decide() + render() on one synthetic screenshot: accepted lines go to teacher_*.npz in
    train.py's format with character boxes, rejected real lines to teacher_distill_*.npz with
    empty labels, a test-game screenshot is dropped, and nothing is written into the repo."""
    import json
    import os
    import subprocess

    import cv2
    out, real, man = tmp_path / "out", tmp_path / "realdata", tmp_path / "man"
    (out / "reads").mkdir(parents=True)
    (out / "inputs" / "eval").mkdir(parents=True)
    vns = [f"v{i}" for i in range(1, 200)]
    train_vn = next(v for v in vns if build.split_of(v) == "train")
    test_vn = next(v for v in vns if build.is_test_vn(v))
    rows, shots = [], []
    for vn in (train_vn, test_vn):
        url = f"https://t.vndb.org/sf/00/{vn[1:]}.jpg"
        img = np.full((544, 960, 3), 20, np.uint8)
        img[400:430, 100:400] = 230                     # light text on a dark ground
        img[436:494, 90:410] = 230                      # a light box with dark text strokes
        img[452:478, 110:390:6] = 15
        (real / "shots" / "vita").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(real / "shots" / "vita" / f"{vn}_{vn[1:]}.jpg"), img)
        shots.append({"vn": vn, "kind": "vita", "url": url, "dims": [960, 544], "release": "r1", "sha256": vn,
                      "session": f"{vn}/r1/0"})
        good = line("こんにちは", v6s="こんにちは", screenai="こんにちは")
        good.update(k=0, crop=[100, 400, 400, 430], ink=[100, 400, 400, 430], view=[82, 392, 418, 438])
        bad = line("ありがとう", v6s="ありがどう", screenai="あいがとう")
        bad.update(k=1, crop=[100, 450, 400, 480], ink=[100, 450, 400, 480], view=[82, 442, 418, 488])
        rows.append({"shot": f"{vn}_{vn[1:]}", "url": url, "sha256": vn, "vn": vn, "kind": "vita", "release": "r1",
                     "session": f"{vn}/r1/0", "dims": [960, 544], "lines": [good, bad]})
    (out / "reads" / "train.0of1.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                   encoding="utf-8")
    for kind in ("vita", "psp", "hires"):
        (out / "inputs" / f"shots_{kind}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in shots if s["kind"] == kind), encoding="utf-8")
    (out / "rule.json").write_text(json.dumps({"rule": {"votes": 2}}), encoding="utf-8")
    env = dict(os.environ, TEACHER_OUT=str(out), REALDATA=str(real), TEACHER_MANIFESTS=str(man))
    r = subprocess.run([sys.executable, str(ROOT / "teacher" / "build.py"), "decide", "--rule", str(out / "rule.json")],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "BUILD_DONE" in r.stdout
    lines = [json.loads(x) for x in (man / "lines_vita.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["label"] for x in lines] == ["こんにちは"] and lines[0]["id"].startswith(train_vn + "_")
    distill = [json.loads(x) for x in (man / "distill_vita.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(distill) == 1 and distill[0]["why"] == "disagree" and "label" not in distill[0]
    z = np.load(out / "teacher_train.npz")
    labels, metas = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    assert labels == ["こんにちは"]
    # the fake character boxes start at x = 0: the crop widens to them, but only to the view (82)
    assert z["shapes"][0].tolist() == [30 + 2 * G.PAD["vita"], 400 - 82 + 2 * G.PAD["vita"]]
    assert lines[0]["crop"] == [82, 400, 400, 430]
    assert metas[0]["chars"][0][0] == "こ" and metas[0]["session"] == f"{train_vn}/r1/0"
    assert metas[0]["dark"] == 0 and metas[0]["inverted"] is False
    zd = np.load(out / "teacher_distill_train.npz")
    assert json.loads(str(zd["labels"])) == [""]
    dmeta = json.loads(str(zd["meta"]))[0]
    assert dmeta["reads"]["v6s"] == "ありがどう"
    assert dmeta["dark"] == 1 and dmeta["inverted"] is True   # dark text stored inverted (console contract)
    assert zd["data"].mean() < 128                            # the light box became the dark ground
    sys.path.insert(0, str(ROOT / "train"))
    import train  # the shard loads as a --real shard
    char2id = {c: i + 1 for i, c in enumerate(build.load_vocab())}
    imgs, labs, dropped = train.load([str(out / "teacher_train.npz")], char2id)
    assert len(imgs) == 1 and dropped == 0
    imgs, labs, dropped = train.load([str(out / "teacher_distill_train.npz")], char2id)
    assert len(imgs) == 0 and dropped == 1               # empty labels never enter supervised training


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
    good["crop"] = good["ink"] = [98, 99, 602, 131]
    wrong = line("ありがとう", v6s="ありがとう", screenai="ありがとう")
    wrong["crop"] = wrong["ink"] = [100, 200, 600, 230]
    menu = line("セーブ", v6s="セーブ", screenai="セーブ")
    menu["crop"] = menu["ink"] = [800, 500, 900, 520]
    shots = {"v1_1": {"vn": "v1", "lines": [good, wrong, menu]}}
    items, scored = measure.match(shots, gts, "vita-test")
    got = [(measure._m(x, None)[0], measure._m(x, None)[2]) for x in items]
    assert got == [("【名前】「こんにちは」", "scored"), (None, "uncertain"), (None, "unmatched")]
    s = measure.score(items, scored, {"votes": 2})
    assert s["accepted"] == 3 and s["accepted_scored"] == 1 and s["precision"] == 1.0 and s["recall"] == 1.0
    whole = line("「こんにちは」", v6s="「こんにちは」", screenai="「こんにちは」")
    whole["crop"] = whole["ink"] = [318, 99, 602, 131]                            # exactly the second gt line: scored
    half = line("「こんに", v6s="「こんに", screenai="「こんに")
    half["crop"] = half["ink"] = [318, 99, 450, 131]                             # half of it: not scorable by boxes
    items, _ = measure.match({"v1_1": {"vn": "v1", "lines": [whole, half]}}, gts, "vita-test")
    assert [(measure._m(x, None)[0], measure._m(x, None)[2]) for x in items] == [("「こんにちは」", "scored"),
                                                                               (None, "partial")]

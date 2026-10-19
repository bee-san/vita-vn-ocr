"""Tests of the textspot evaluation sets' tool (eval/textspot_sets.py) and of the committed manifests
(eval/manifests/textspot-dev.jsonl, textspot-test.jsonl): the frame selection's ranking and caps on a fake
inventory, the frame scaling (an Adrenaline frame's boxes are exactly its native boxes times 2), the review
decisions' parsers and checks, and the manifests' schema and split hygiene. No screenshots, no network.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot_sets as T  # noqa: E402

MAN = ROOT / "eval" / "manifests"
DEV = MAN / "textspot-dev.jsonl"
TEST = MAN / "textspot-test.jsonl"


# ----------------------------------------------------------------------------- fake inventory

def shot(vn, num, kind, dims=None):
    dims = dims or {"vita": [960, 544], "psp": [480, 272], "hires": [1280, 720]}[kind]
    return {"shot": f"{vn}_{num}", "sid": f"sf{num}", "num": num, "vn": vn, "kind": kind, "url": T.url_of(num),
            "release": f"r{vn[1:]}", "sha256": hashlib.sha256(str(num).encode()).hexdigest(), "src_dims": dims,
            "path": f"/nonexistent/{vn}_{num}.jpg", "rank": T.frame_rank("dev", f"sf{num}")}


def fake_inventory():
    inv, num = [], 1000
    for v in range(30):                          # 30 VNs: 7 vita, 4 psp, 2 hires shots each
        vn = f"v{100 + v}"
        for kind, n in (("vita", 7), ("psp", 4), ("hires", 2)):
            for _ in range(n):
                num += 1
                inv.append(shot(vn, num, kind))
    for v in range(20):                          # 20 more VNs with PC shots only
        vn = f"v{500 + v}"
        for _ in range(3):
            num += 1
            inv.append(shot(vn, num, "hires", [800, 600]))
    return sorted(inv, key=lambda r: r["rank"])


def test_ranks_are_the_protocols_hashes():
    assert T.frame_rank("dev", "sf57781") == hashlib.sha256(b"textspot-dev:sf57781").hexdigest()
    assert T.frame_rank("test", "sf96036") == hashlib.sha256(b"textspot-test:sf96036").hexdigest()
    assert T.vn_rank("v1315") == hashlib.sha256(b"textspot-dev-vn:v1315").hexdigest()


def test_dev_selection_ranks_and_caps():
    inv = fake_inventory()
    frames, cands = T.select_dev(inv)
    by = {}
    for f in frames:
        by.setdefault(f["part"], []).append(f)
    by_vn_kind = {}
    for r in inv:
        by_vn_kind.setdefault((r["vn"], r["kind"]), []).append(r)
    # vita: the 5 lowest-ranked shots of every VN
    for vn in {r["vn"] for r in inv if r["kind"] == "vita"}:
        got = sorted(f["shot"] for f in by["vita"] if f["vn"] == vn)
        want = sorted(r["shot"] for r in sorted(by_vn_kind[(vn, "vita")], key=lambda r: r["rank"])[:5])
        assert got == want
    # psp: 2 per VN; adrenaline: the first 20 PSP VNs by VN rank, their 3rd PSP shot by rank
    psp_vns = sorted({r["vn"] for r in inv if r["kind"] == "psp"}, key=T.vn_rank)
    assert all(sum(1 for f in by["psp"] if f["vn"] == vn) == 2 for vn in psp_vns)
    assert [f["vn"] for f in by["adrenaline"]] == psp_vns[:20]
    for f in by["adrenaline"]:
        third = sorted(by_vn_kind[(f["vn"], "psp")], key=lambda r: r["rank"])[2]
        assert f["shot"] == third["shot"] and f["id"] == third["shot"] + "_x2"
        assert f["dims"] == [960, 544] and f["scale"] == {"method": "cv2.INTER_LINEAR", "factor": 2}
    # pc: the first 40 VNs with PC shots by VN rank, their first shot by rank, scaled to fit 960x544
    pc_vns = sorted({r["vn"] for r in inv if r["kind"] == "hires"}, key=T.vn_rank)
    assert [f["vn"] for f in by["pc"]] == pc_vns[:40]
    for f in by["pc"]:
        first = sorted(by_vn_kind[(f["vn"], "hires")], key=lambda r: r["rank"])[0]
        assert f["shot"] == first["shot"] and f["scale"]["method"] == "cv2.INTER_AREA"
        assert f["dims"] == ([960, 540] if f["src_dims"] == [1280, 720] else [725, 544])
    assert "text-free" not in by and cands == []          # no detector counts yet: no text-free candidate
    ids = [f["id"] for f in frames]
    assert len(ids) == len(set(ids))


def test_dev_selection_replaces_dropped_frames_and_takes_confirmed_text_free_frames():
    inv = fake_inventory()
    frames, _ = T.select_dev(inv)
    first_vita = next(f for f in frames if f["part"] == "vita")
    frames2, _ = T.select_dev(inv, drops={first_vita["shot"]})
    vita = [f for f in frames2 if f["part"] == "vita" and f["vn"] == first_vita["vn"]]
    ranked = sorted((r for r in inv if r["vn"] == first_vita["vn"] and r["kind"] == "vita"),
                    key=lambda r: r["rank"])
    want = [r["shot"] for r in ranked if r["shot"] != first_vita["shot"]][:5]
    assert sorted(f["shot"] for f in vita) == sorted(want)
    # text-free: candidates are unused shots with at most one detector box, in rank order; the first 25 confirmed
    used = {f["shot"] for f in frames}
    rest = [r for r in inv if r["shot"] not in used]
    dets = {r["shot"]: (0 if i % 2 == 0 else 5) for i, r in enumerate(rest)}
    _, cands = T.select_dev(inv, dets=dets)
    assert [c["shot"] for c in cands] == [r["shot"] for r in rest if dets[r["shot"]] <= 1]
    tf = {c["shot"]: ("no" if i % 3 == 0 else "yes") for i, c in enumerate(cands)}
    frames3, _ = T.select_dev(inv, tf=tf, dets=dets)
    tfree = [f["shot"] for f in frames3 if f["part"] == "text-free"]
    assert tfree == [c["shot"] for c in cands if tf[c["shot"]] == "yes"][:25]
    assert all(f["text_free"] for f in frames3 if f["part"] == "text-free")
    look = T.tf_candidates_to_look(cands, {}, need=3, spare=2)
    assert look == [c["shot"] for c in cands[:5]]


# ----------------------------------------------------------------------------- scaling

def test_frame_scaling():
    import cv2
    rng = np.random.default_rng(0)
    psp = rng.integers(0, 256, (272, 480, 3), dtype=np.uint8)
    dims, scale = T.frame_spec("adrenaline", [480, 272])
    assert dims == [960, 544] and scale == {"method": "cv2.INTER_LINEAR", "factor": 2}
    assert np.array_equal(T.build_frame(psp, dims, scale), cv2.resize(psp, (960, 544),
                                                                       interpolation=cv2.INTER_LINEAR))
    dims, scale = T.frame_spec("psp", [480, 272])
    assert dims == [480, 272] and scale is None and T.build_frame(psp, dims, scale) is psp
    pc = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    dims, scale = T.frame_spec("pc", [800, 600])
    assert dims == [725, 544] and scale["method"] == "cv2.INTER_AREA"
    assert np.array_equal(T.build_frame(pc, dims, scale), cv2.resize(pc, (725, 544), interpolation=cv2.INTER_AREA))
    assert T.frame_spec("pc", [1920, 1080])[0] == [960, 540]
    assert T.frame_spec("text-free", [960, 544]) == ([960, 544], None)


def test_adrenaline_references_are_the_native_boxes_times_two(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "SETS", tmp_path)
    base = {"vn": "v1", "shot": "v1_5", "sid": "sf5", "url": T.url_of(5), "session": "r1", "sha256": "0" * 64,
            "src_dims": [480, 272], "rank": "a", "path": "/x"}
    frames = [dict(base, id="v1_5", part="psp", scale=None, dims=[480, 272]),
              dict(base, id="v1_5_x2", part="adrenaline", scale={"method": "cv2.INTER_LINEAR", "factor": 2},
                   dims=[960, 544])]
    refs = [{"ink": [10, 20, 110, 36], "cls": "dialogue", "vertical": False, "text": "テスト", "ignore": False,
             "why": None},
            {"ink": [400, 30, 415, 200], "cls": "scene", "vertical": True, "text": None, "ignore": True,
             "why": "rotated"}]
    rows, problems = T.manifest_rows({"frames": frames, "split": "test"}, {"v1_5": {"status": "final",
                                                                                   "refs": refs}})
    assert not problems
    one, two = rows[0]["refs"], rows[1]["refs"]
    assert one[0]["box"] == [10, 20, 100, 16] and one[1]["why"] == "rotated" and "why" not in one[0]
    for a, b in zip(one, two):
        assert b["box"] == [2 * v for v in a["box"]]
        assert {k: v for k, v in a.items() if k != "box"} == {k: v for k, v in b.items() if k != "box"}


# ----------------------------------------------------------------------------- review decisions

def prop(k, ink, vertical=False, text="テキスト"):
    return {"k": k, "det": ink, "ink": ink, "vertical": vertical, "det_conf": 0.9,
            "reads": {"meiki": text, "meikic": text, "v6s": text}}


def proposals():
    return {"shot": "v1_1", "dims": [960, 544], "n_dets": 5,
            "props": [prop(0, [100, 400, 300, 420], text="「あいう」"), prop(1, [100, 430, 200, 450]),
                      prop(2, [210, 430, 300, 450]), prop(3, [800, 500, 900, 520], text="セーブロード"),
                      prop(4, [900, 50, 920, 200], vertical=True)]}


def test_review_proposals_are_numbered_in_reading_order():
    plist = T.review_proposals(proposals())
    assert [q["n"] for q in plist] == [1, 2, 3, 4, 5]
    assert [q["k"] for q in plist] == [4, 0, 1, 2, 3]          # the column at y 50 first


def test_pass1_instances_from_a_box_decision():
    p = proposals()
    dec = {"frame": "F001", "screen": True, "boxes": [
        {"n": 1, "cls": "dialogue"},                             # the column, kept
        {"n": 2, "cls": "dialogue", "box": [98, 398, 210, 26]},   # kept with a new box
        {"n": 3, "merge": [4], "cls": "dialogue"},                # pieces of one line
        {"n": 5, "split": [850], "cls": ["ui", "ui"]}],
        "missed": [{"box": [10, 10, 50, 20], "cls": "ui"}, {"box": [500, 10, 100, 40], "ignore": "logo"}]}
    insts = T.p1_instances("v1_1", p, dec, {})
    by = {q["iid"]: q for q in insts}
    assert set(by) == {"p4", "p0.b", "p1.m", "p3.s0", "p3.s1", "m1.0", "m1.1"}
    assert by["p4"]["vertical"] and by["p4"]["cls"] == "dialogue" and not by["p4"].get("pending")
    assert by["p0.b"]["pending"] and by["p0.b"]["rect"] == [98, 398, 308, 424]
    assert by["p1.m"]["rect"] == [100, 430, 300, 450] and by["p1.m"]["reads"]["meiki"] == "テキストテキスト"
    assert by["p3.s0"]["rect"] == [800, 500, 850, 520] and by["p3.s1"]["rect"] == [850, 500, 900, 520]
    assert by["m1.1"]["ignore"] == "logo" and not by["m1.1"]["needs_text"]
    # once reread, the instances carry the refined box and the readings
    rr = {("v1_1", "p0.b"): {"rect": [98, 398, 308, 424], "vertical": False, "ink": [101, 401, 299, 419],
                             "reads": {"v6s": "x"}}}
    q = next(x for x in T.p1_instances("v1_1", p, dec, rr) if x["iid"] == "p0.b")
    assert not q["pending"] and q["ink"] == [101, 401, 299, 419] and q["reads"] == {"v6s": "x"}
    # a dropped or ignored proposal is not an instance to read
    dec2 = {"frame": "F001", "screen": True, "boxes": [{"n": 1, "ignore": "rotated"}, {"n": 2, "drop": True},
                                                        {"n": 3, "cls": "ui"}, {"n": 4, "cls": "ui"},
                                                        {"n": 5, "cls": "ui"}], "missed": []}
    insts2 = T.p1_instances("v1_1", p, dec2, {})
    assert [q["iid"] for q in insts2] == ["p4", "p1", "p2", "p3"] and insts2[0]["ignore"] == "rotated"


def write_frame_json(d, F, js):
    (d / f"{F}.json").write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")


def test_pass1_box_check(tmp_path):
    js = {"frame": "F001", "dims": [960, 544], "text_free_candidate": False,
          "proposals": [{"n": 1, "box": [100, 400, 200, 20], "vertical": False, "src": "meiki"},
                        {"n": 2, "box": [100, 430, 200, 20], "vertical": False, "src": "meiki"}]}
    write_frame_json(tmp_path, "F001", js)
    write_frame_json(tmp_path, "F002", dict(js, frame="F002", text_free_candidate=True, proposals=[]))
    meta = {"frames": [{"F": "F001", "shot": "a"}, {"F": "F002", "shot": "b"}]}
    good = [("r", {"frame": "F001", "screen": True, "boxes": [{"n": 1, "cls": "name"}, {"n": 2, "split": [200],
                                                                                         "cls": ["ui", "ui"]}],
                   "missed": [{"box": [5, 5, 30, 12], "cls": "ui"}]}),
            ("r", {"frame": "F002", "screen": True, "text_free": True, "boxes": [], "missed": []})]
    assert T.validate_p1b(meta, tmp_path, good) == []
    bad = [("r", {"frame": "F001", "screen": True, "boxes": [{"n": 1, "cls": "speech"}, {"n": 2, "split": [50]},
                                                             {"n": 9, "cls": "ui"}],
                  "missed": [{"box": [950, 5, 30, 12], "cls": "ui"}]}),
           ("r", {"frame": "F002", "screen": True, "boxes": [], "missed": []})]
    errs = " | ".join(T.validate_p1b(meta, tmp_path, bad))
    for want in ("n=1 needs cls", "n=2 split cuts", "n=9 is not a proposal", "inside the frame",
                 "F002: text-free candidate"):
        assert want in errs
    # the last line for a frame counts; a frame marked not a game screen needs nothing else
    later = bad + [("r", {"frame": "F001", "screen": False}), ("r", good[1][1])]
    assert T.validate_p1b(meta, tmp_path, later) == []
    assert "no decision" in " ".join(T.validate_p1b(meta, tmp_path, good[:1]))


def test_text_decisions_check_and_parse(tmp_path):
    sheet = {"sheet": "S001", "entries": [{"n": 1, "key": "a|p0", "candidates": [{"c": 1, "text": "あ"},
                                                                                 {"c": 2, "text": "ぁ"}]},
                                          {"n": 2, "key": "a|p1", "candidates": []}]}
    (tmp_path / "S001.json").write_text(json.dumps(sheet, ensure_ascii=False), encoding="utf-8")
    meta = {"sheets": ["S001"]}
    ok = [("r", {"sheet": "S001", "n": 1, "pick": 2}), ("r", {"sheet": "S001", "n": 2, "text": "「え...」!"})]
    assert T.validate_text(meta, tmp_path, ok, "p1t") == []
    errs = T.validate_text(meta, tmp_path, [("r", {"sheet": "S001", "n": 1, "pick": 3}),
                                            ("r", {"sheet": "S001", "n": 2, "bad": "maybe"})], "p1t")
    assert len(errs) == 2
    e1, e2 = sheet["entries"]
    assert T.text_of({"pick": 2}, e1) == ("ぁ", "text")
    assert T.text_of({"text": "「え...」!?~"}, e2) == ("「え…」！？～", "text")
    assert T.text_of({"unsure": True}, e2) == (None, "unsure")
    assert T.text_of({"bad": "cut"}, e2) == (None, "bad")
    # agreement of two looks under eval's normalisation; a third look settles a disagreement
    assert T._agree(("あい。", "text"), ("あい。", "text"), None) == ("あい。", "agree")
    assert T._agree(("あい", "text"), ("あぃ", "text"), None) == (None, "need-p3")
    assert T._agree(("あい", "text"), ("あぃ", "text"), ("あぃ", "text")) == ("あぃ", "p3-second")
    assert T._agree(("あい", "text"), ("あぃ", "text"), ("あう", "text")) == (None, "p3-none")
    assert T._agree((None, "unsure"), (None, "unsure"), None) == (None, "both-unsure")
    assert T._agree(("あい", "text"), None, None) == (None, "open")


def test_blind_candidates_are_shuffled_and_unattributed():
    e = {"key": "v1_1|p0", "src": "meiki", "reads": {"meiki": "あい", "meikic": "あい", "v6s": "あぃ"}}
    c1 = T.candidates_of(e, "p1t")
    assert c1[0] == {"c": 1, "text": "あい", "engines": ["meiki", "meikic"]}
    c2 = T.candidates_of(e, "p2t")
    assert sorted(x["text"] for x in c2) == sorted(["あい", "あぃ"]) and all("engines" not in x for x in c2)
    assert T.candidates_of(e, "p2t") == c2                     # deterministic
    assert T.candidates_of(dict(e, src="missed"), "p2t") == []  # no candidates for a missed instance


def test_pass2_box_check(tmp_path):
    js = {"frame": "F001", "dims": [480, 272], "text_free_candidate": False,
          "instances": [{"n": 1, "box": [10, 200, 200, 14], "cls": "dialogue", "vertical": False},
                        {"n": 2, "box": [10, 220, 100, 14], "cls": "dialogue", "vertical": False, "eval_line": True},
                        {"n": 3, "box": [300, 10, 40, 10], "ignore": "logo", "vertical": False}]}
    write_frame_json(tmp_path, "F001", js)
    meta = {"frames": [{"F": "F001", "shot": "a"}]}
    assert T.validate_p2b(meta, tmp_path, [("r", {"frame": "F001", "screen": True, "ok": True})]) == []
    good = {"frame": "F001", "screen": True,
            "flags": [{"n": 1, "split": [100], "cls": ["dialogue", "dialogue"], "texts": ["あ", "い"]},
                      {"n": 3, "unignore": True, "cls": "ui", "text": "設定"}],
            "missed": [{"box": [400, 250, 60, 12], "cls": "ui", "text": "MENU"}]}
    assert T.validate_p2b(meta, tmp_path, [("r", good)]) == []
    bad = {"frame": "F001", "screen": True, "flags": [{"n": 2, "split": [50], "texts": ["a", "b"]},
                                                      {"n": 1, "box": [10, 200, 220, 14], "cuts": True}],
           "missed": [{"box": [400, 250, 60, 12], "cls": "ui"}]}
    errs = " | ".join(T.validate_p2b(meta, tmp_path, [("r", bad)]))
    for want in ("n=2 is an eval line", "cuts: true needs", "missed entry needs cls and text"):
        assert want in errs


# ----------------------------------------------------------------------------- the committed manifests

FIELDS = {"id", "vn", "part", "session", "url", "sha256", "src_dims", "scale", "dims", "text_free", "refs", "looks",
          "reviewer_kind"}


def manifest(path):
    if not path.exists():
        pytest.skip(f"{path.name} not committed yet")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def check_rows(rows, parts):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert FIELDS <= set(r), r["id"]
        assert r["part"] in parts and r["reviewer_kind"] == "assistant" and r["looks"]
        assert r["url"].startswith("https://t.vndb.org/sf/") and len(r["sha256"]) == 64
        W, H = r["dims"]
        if r["part"] == "adrenaline":
            assert r["scale"] == {"method": "cv2.INTER_LINEAR", "factor": 2} and r["dims"] == [2 * v for v in
                                                                                             r["src_dims"]]
        elif r["scale"] is None:
            assert r["dims"] == r["src_dims"] and tuple(r["dims"]) in ((960, 544), (480, 272))
        else:
            sw, sh = r["src_dims"]
            s = min(960 / sw, 544 / sh)
            assert r["scale"]["method"] == "cv2.INTER_AREA" and r["dims"] == [round(sw * s), round(sh * s)]
        assert r["text_free"] == (r["part"] == "text-free")
        if r["text_free"]:
            assert r["refs"] == []
        for x in r["refs"]:
            x0, y0, w, h = x["box"]
            assert w > 0 and h > 0 and x0 >= 0 and y0 >= 0 and x0 + w <= W and y0 + h <= H, (r["id"], x)
            assert isinstance(x["vertical"], bool) and isinstance(x["ignore"], bool)
            if x["ignore"]:
                assert x["why"] in T.IGNORE_WHY and x["text"] is None, (r["id"], x)
            else:
                assert x["cls"] in T.CLASSES, (r["id"], x)
                assert isinstance(x["text"], str) and T.norm(x["text"]), (r["id"], x)
                assert "why" not in x


def reserved():
    return json.loads((MAN / "reserved.json").read_text(encoding="utf-8"))


def eval_urls():
    out = {}
    for s in ("vita-test", "psp-test"):
        for line in (MAN / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["url"]] = r
    return out


def test_dev_manifest():
    rows = manifest(DEV)
    check_rows(rows, T.PARTS)
    dev, test = set(reserved()["dev_sample"]), set(reserved()["test"])
    sessions = set(reserved()["sessions"])
    ev = eval_urls()
    for r in rows:
        assert r["vn"] in dev and r["vn"] not in test, r["id"]
        assert r["session"] not in sessions and r["url"] not in ev, r["id"]
    count = {}
    for r in rows:
        count.setdefault(r["part"], {}).setdefault(r["vn"], 0)
        count[r["part"]][r["vn"]] += 1
    assert all(n <= 5 for n in count.get("vita", {}).values())
    assert all(n <= 2 for n in count.get("psp", {}).values())
    assert len(count.get("adrenaline", {})) <= 20 and all(n == 1 for n in count.get("adrenaline", {}).values())
    assert len(count.get("pc", {})) <= 40 and all(n == 1 for n in count.get("pc", {}).values())
    assert sum(count.get("text-free", {}).values()) <= 25
    urls = [r["url"] for r in rows]
    assert len(urls) == len(set(urls)), "one screenshot in two frames"


def test_test_manifest():
    rows = manifest(TEST)
    check_rows(rows, ("vita", "psp", "adrenaline", "text-free"))
    tg = json.loads((MAN / "test_games.json").read_text(encoding="utf-8"))
    games = set(tg["all"])
    ev = eval_urls()
    for r in rows:
        assert r["vn"] in games, r["id"]
    main = {r["url"]: r for r in rows if r["part"] in ("vita", "psp")}
    assert set(main) == set(ev), "every vita-test and psp-test screenshot is a frame"
    for url, r in main.items():
        assert r["sha256"] == ev[url]["sha256"] and r["session"] == ev[url]["session"]
    # report-only Adrenaline frames: every psp-test frame at 2x, with its references times 2 exactly
    adr = {r["url"]: r for r in rows if r["part"] == "adrenaline"}
    assert set(adr) == {u for u, r in main.items() if r["part"] == "psp"}
    for url, r in adr.items():
        assert r.get("report_only") is True
        assert [x["box"] for x in r["refs"]] == [[2 * v for v in x["box"]] for x in main[url]["refs"]]
    tf = [r for r in rows if r["part"] == "text-free"]
    assert len(tf) <= 25 and all(r["url"] not in ev for r in tf)
    # eval's hand-checked lines keep their text (uncertain ones are ignore areas)
    by_url = {}
    for e in ev.values():
        by_url.setdefault(e["url"], [])
    for line in (MAN / "vita-test.jsonl").read_text().splitlines() + (MAN / "psp-test.jsonl").read_text().splitlines():
        e = json.loads(line)
        by_url[e["url"]].append(e)
    for url, r in main.items():
        texts = [x["text"] for x in r["refs"] if not x["ignore"]]
        for e in by_url[url]:
            if not e.get("uncertain"):
                assert e["text"] in texts or any(x["ignore"] and x["why"] == "undecided" for x in r["refs"]), \
                    (r["id"], e["id"])


def test_no_frame_or_session_in_both_sets():
    dev, test = manifest(DEV), manifest(TEST)
    assert not {r["url"] for r in dev} & {r["url"] for r in test}
    assert not {r["sha256"] for r in dev} & {r["sha256"] for r in test}
    assert not {r["session"] for r in dev} & {r["session"] for r in test}
    assert not {r["vn"] for r in dev} & {r["vn"] for r in test}

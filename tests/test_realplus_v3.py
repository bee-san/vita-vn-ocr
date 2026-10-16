"""realplus v3: the Claude + Luna agreement rule (realplus/luna.py, pool.apply_luna), the independent
precision check (pool.check_*), the hygiene tool's blockers and the frozen-batch guarantee of
decide.parsed. No network, screenshots or shared artifacts."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("realplus", "synth", ""):
    sys.path.insert(0, str(ROOT / d) if d else str(ROOT))

import hygiene as HY  # noqa: E402
import luna as LU  # noqa: E402
import pool as PL  # noqa: E402

from devnative import decide as D  # noqa: E402


def _dec(kind, text=None, skip=None, pick=None):
    return {"decision": kind, "text": text, "skip": skip, "pick": pick, "reviewer_kind": "assistant"}


def test_vote_rule_agree_disagree_edge_skip_missing():
    c = _dec("pick", "「ええと……」", pick=1)
    assert LU.vote(c, _dec("text", "「ええと......」"))["luna"] == "agree"            # same key: six dots = ……
    v = LU.vote(c, _dec("text", "ええと……"))
    assert v["luna"] == "disagree" and v["edge_only"]                                 # edge 「」 only: still a drop
    v = LU.vote(c, _dec("text", "「ええど……」"))
    assert v["luna"] == "disagree" and not v["edge_only"]                             # dakuten: a real difference
    assert LU.vote(c, _dec("skip", skip="ui"))["luna"] == "skip-ui"
    assert LU.vote(c, _dec("skip", skip="cut"))["luna"] == "skip-cut"
    assert LU.vote(c, _dec("unsure"))["luna"] == "unsure"
    assert LU.vote(c, None)["luna"] == "missing"
    v = LU.vote(_dec("skip", skip="ui"), _dec("pick", "何か", pick=1))
    assert v["luna"] == "n/a" and v["luna_vote"]["decision"] == "pick"               # Claude's skip stands
    assert LU.vote(_dec("unsure"), None)["luna"] == "n/a"
    luna = dict(_dec("text", "「ええと……」"), reviewer_kind="codex:gpt-6-luna")      # as parse_lenient records it
    assert LU.vote(c, luna)["luna_vote"]["reviewer_kind"] == "codex:gpt-6-luna"
    assert LU.vote(c, {"decision": "text", "text": "「ええと……」"})["luna_vote"]["reviewer_kind"] == LU.KIND


def test_vote_keys_fold_only_what_eval_folds():
    c = _dec("text", "そう・・・だね")
    assert LU.vote(c, _dec("text", "そう…だね"))["luna"] == "agree"                    # dot runs fold to …
    assert LU.vote(c, _dec("text", "そう…だねー"))["luna"] == "disagree"                # ー is not edge punctuation
    assert LU.vote(_dec("text", "ＡＢＣ１２"), _dec("text", "ABC12"))["luna"] == "agree"


def _sheets():
    return {"S001": {1: {"id": "v1_1_0", "candidates": [{"c": 1, "text": "あい"}, {"c": 2, "text": "あぃ"}]},
                     2: {"id": "v1_1_1", "candidates": [{"c": 1, "text": "名前"}]},
                     3: {"id": "v1_1_2", "candidates": [{"c": 1, "text": "はい"}]}}}


def test_parse_lenient_skips_bad_lines_and_expands_default():
    text = "\n".join([json.dumps({"sheet": "S001", "n": 1, "pick": 2}),
                      "not json at all",
                      json.dumps({"sheet": "S001", "n": 9, "pick": 1}),               # no such entry
                      json.dumps({"sheet": "S001", "n": 3, "pick": 1, "skip": "ui"}),  # pick and skip: the skip
                      json.dumps({"sheet": "S001", "default": 1})])
    got, bad = LU.parse_lenient(text, _sheets(), "pass1", "b0018")
    assert bad == 2
    assert got["v1_1_0"]["decision"] == "pick" and got["v1_1_0"]["text"] == "あぃ"
    assert got["v1_1_1"]["via"] == "default" and got["v1_1_1"]["text"] == "名前"      # one candidate: default applies
    assert got["v1_1_2"]["decision"] == "skip" and "via" not in got["v1_1_2"]           # explicit line wins
    assert all(r["reviewer_kind"] == "codex:gpt-6-luna" for r in got.values())


def _label(lid, batch, how="consensus", status="kept", confirm=None, pass2=None):
    r = {"id": lid, "status": status, "why": None if status == "kept" else "skip-ui", "label": "あい", "how": how,
         "support": ["meiki", "v6s"] if how == "consensus" else ["meiki"], "readers": {}, "name": False,
         "reviewer_kind": "assistant", "pass1": {"decision": "pick", "batch": batch} if batch else None,
         "pass2": pass2, "looks": 1}
    if confirm is not None:
        r["confirm"] = {"decision": "pick", "agree": confirm, "batch": "b0001"}
    return r


def _vote(lid, status):
    return {"id": lid, "luna": status, "luna_vote": {"decision": "pick"} if status != "missing" else None,
            "edge_only": False}


def test_apply_luna_keeps_agreements_drops_disagreements_and_pends_the_rest():
    labels = [_label("a", "b0018"), _label("b", "b0018"), _label("c", "b0018"), _label("d", "b0018"),
              _label("e", "b0018", confirm=True), _label("f", "b0018", how="single"),
              _label("g", "b0018", how="single", confirm=True), _label("h", "b0003"), _label("i", None, how="auto"),
              _label("j", "b0018", status="dropped")]
    votes = {"a": _vote("a", "agree"), "b": _vote("b", "disagree"), "c": _vote("c", "skip-ui"),
             "e": _vote("e", "skip-cut"), "f": _vote("f", "agree"), "g": _vote("g", "agree"), "j": _vote("j", "n/a")}
    c = PL.apply_luna(labels, votes, v3={"b0018"})
    by = {r["id"]: r for r in labels}
    assert by["a"]["status"] == "kept" and by["a"]["looks"] == 2 and by["a"]["luna"] == "agree"
    assert by["b"]["status"] == "dropped" and by["b"]["why"] == "luna-disagree" and by["b"]["label"] is None
    assert by["c"]["status"] == "pending" and by["c"]["why"] == "luna-unconfirmed" and by["c"]["label"] == "あい"
    assert by["d"]["status"] == "dropped" and by["d"]["why"] == "luna-missing"
    assert by["e"]["status"] == "kept"                                   # the confirm look read the same key
    assert by["f"]["status"] == "pending" and by["f"]["why"] == "confirm-missing"   # one family: needs the look
    assert by["g"]["status"] == "kept"
    assert by["h"]["status"] == "kept" and "luna" not in by["h"] and by["h"]["reviewer_kinds"] == ["assistant"]
    assert by["i"]["status"] == "kept" and by["i"]["reviewer_kinds"] == ["assistant"]
    assert by["j"]["status"] == "dropped" and by["j"]["why"] == "skip-ui"
    assert all(by[i]["reviewer_kinds"] == ["assistant", "codex:gpt-6-luna"] for i in "abcdefgj")
    assert c["kept:agree"] == 2 and c["dropped:luna-disagree"] == 1 and c["pending:luna-unconfirmed"] == 1


def _man(lid, how, crops, batch=None):
    return {"id": lid, "how": how, "label": "あい", "pass1": {"batch": batch} if batch else None,
            "crops": {c: [0, 0, 10, 10] for c in crops}}


def test_check_sample_is_stratified_and_picks_a_crop_the_pool_holds():
    man = [_man(f"v1_{i}_0", "auto", ["vita_ink", "psp_ink"]) for i in range(20)]               # v2 auto
    man += [_man(f"v2_{i}_0", "consensus", ["vita_ink"], "b0005") for i in range(5)]           # v2 reviewed
    man += [_man(f"v3_{i}_0", "consensus", ["vita_ink", "vita_grad"], "b0020") for i in range(30)]   # new t1
    man += [_man(f"v4_{i}_0", "consensus", ["vita_ink", "psp_grad"], "b0030") for i in range(6)]     # new t2
    man += [_man(f"v5_{i}_0", "auto", ["vita_ink", "psp_ink", "vita_grad"]) for i in range(8)]       # new auto
    man += [_man(f"v6_{i}_0", "consensus", ["vita_ink", "psp_ink"], "b0007") for i in range(5)]     # new edge
    v2 = {m["id"] for m in man if m["id"].startswith(("v1_", "v2_"))}
    tiers = {m["id"]: ("t2" if m["id"].startswith("v4_") else "t1") for m in man}
    strata = {"new:t1": 10, "new:t2": 10, "new:auto": 4, "new:edge": 3, "v2:auto": 5, "v2:reviewed": 2}
    sample = PL.check_sample(man, v2, {"b0020", "b0030"}, tiers, "seed", strata)
    by = {}
    for e in sample:
        by.setdefault(e["stratum"], []).append(e)
    assert {k: len(v) for k, v in by.items()} == {"new:t1": 10, "new:t2": 6, "new:auto": 4, "new:edge": 3,
                                                   "v2:auto": 5, "v2:reviewed": 2}
    rows = {m["id"]: m for m in man}
    assert all(e["crop"] in rows[e["id"]]["crops"] for e in sample)
    assert sample == PL.check_sample(man, v2, {"b0020", "b0030"}, tiers, "seed", strata)     # pre-registered
    assert {e["crop"] for e in by["new:auto"]} <= {"vita_ink", "psp_ink", "vita_grad"}
    assert {e["crop"] for e in by["new:edge"]} <= {"vita_ink", "psp_ink"}


def test_check_score_counts_skips_as_disagreements_by_stratum_and_crop():
    man = [_man("a", "auto", ["vita_ink"]), _man("b", "auto", ["psp_ink"]), _man("c", "consensus", ["vita_grad"])]
    spec = {"entries": [{"id": "a", "crop": "vita_ink", "stratum": "v2:auto"},
                        {"id": "b", "crop": "psp_ink", "stratum": "v2:auto"},
                        {"id": "c", "crop": "vita_grad", "stratum": "new:t1"},
                        {"id": "d", "crop": "vita_ink", "stratum": "new:t1"}]}          # not decided yet
    decided = {"a@vita_ink": {"decision": "text", "text": "あい"}, "b@psp_ink": {"decision": "skip", "skip": "cut"},
               "c@vita_grad": {"decision": "text", "text": "あぃ"}}
    summ, dis, n = PL.check_score(spec, man, decided)
    assert n == 3 and summ["all"]["agree"] == 1 and summ["all"]["lines"] == 3
    assert summ["stratum:v2:auto"]["agree"] == 1 and summ["stratum:new:t1"]["agree"] == 0
    assert summ["crop:psp_ink"]["agree"] == 0 and summ["side:new"]["lines"] == 1
    assert [r["id"] for r in dis] == ["b", "c"] and len(summ["all"]["wilson95"]) == 2


def test_check_stratum_new_edge_is_an_unreviewed_v2_label_outside_v2():
    assert PL.check_stratum(_man("x", "auto", ["vita_ink"]), set(), {"b0020"}, {}) == "new:auto"
    assert PL.check_stratum(_man("x", "consensus", ["vita_ink"], "b0003"), set(), {"b0020"}, {}) == "new:edge"
    assert PL.check_stratum(_man("x", "consensus", ["vita_ink"], "b0020"), set(), {"b0020"}, {"x": "t2"}) == "new:t2"
    assert PL.check_stratum(_man("x", "consensus", ["vita_ink"], "b0003"), {"x"}, {"b0020"}, {}) == "v2:reviewed"


def _dis(i, label, decision, text=None, crop="vita_ink", stratum="new:t1"):
    return {"id": i, "crop": crop, "stratum": stratum, "decision": decision, "text": text, "label": label}


def test_look3_candidates_show_the_label_and_a_typed_other_reading_shuffled_and_fixed():
    one = PL.look3_candidates(_dis("a", "「はい」", "unsure"))
    assert [c["text"] for c in one] == ["「はい」"]
    two = PL.look3_candidates(_dis("b", "「はい」", "text", "「はい。」"))
    assert sorted(c["text"] for c in two) == ["「はい。」", "「はい」"]
    assert two == PL.look3_candidates(_dis("b", "「はい」", "text", "「はい。」"))            # same order every run
    same_key = PL.look3_candidates(_dis("c", "あ…", "text", "あ..."))   # the check typed the label's key
    assert [c["text"] for c in same_key] == ["あ…"]


def test_look3_resolutions_keep_only_a_third_look_that_reads_the_label():
    dis = [_dis("a", "あい", "unsure"), _dis("b", "あい", "text", "あぃ"), _dis("c", "あい", "skip"),
           _dis("d", "あい", "unsure", crop="psp_grad"), _dis("e", "あい", "text", "かき")]
    looks = {"a@vita_ink": {"decision": "pick", "text": "あい"}, "b@vita_ink": {"decision": "text", "text": "あぃ"},
             "c@vita_ink": {"decision": "skip", "skip": "cut"}, "d@psp_grad": {"decision": "unsure", "note": "blur"},
             "e@vita_ink": {"decision": "text", "text": "か き"}}       # another reading, not the label's key
    res = PL.look3_resolutions(dis, looks, "v3pre")
    assert [(r["id"], r["keep"], r["why"]) for r in res] == [
        ("a", True, "label-right"), ("b", False, "label-wrong"), ("c", False, "cut"), ("d", False, "unsure"),
        ("e", False, "label-wrong")]
    assert res[3]["note"].endswith('"unsure"; blur') and "after the v3pre check" in res[0]["note"]
    with pytest.raises(KeyError):
        PL.look3_resolutions(dis, {}, "v3pre")


def test_check_after_looks_counts_right_wrong_and_unsettled_by_stratum_and_crop():
    spec = {"entries": [{"id": i, "crop": c, "stratum": s} for i, c, s in
                        [("a", "vita_ink", "new:t1"), ("b", "psp_grad", "new:t1"), ("c", "vita_ink", "v2:auto"),
                         ("d", "psp_ink", "v2:auto")]]}
    report = {"disagreements": [{"id": "b", "crop": "psp_grad"}, {"id": "c", "crop": "vita_ink"},
                                {"id": "d", "crop": "psp_ink"}]}
    res = {"b": {"keep": True, "why": "label-right"}, "c": {"keep": False, "why": "label-wrong"},
           "d": {"keep": False, "why": "unsure"}}
    out = PL.check_after_looks(spec, report, res)
    assert out["all"]["lines"] == 4 and out["all"]["right"] == 2 and out["all"]["look3-wrong"] == 1
    assert out["all"]["look3-unsettled"] == 1 and out["stratum:new:t1"]["rate"] == 1.0
    assert out["target:psp"]["right"] == 1 and out["crop:vita_ink"]["look3-wrong"] == 1
    assert len(out["all"]["wilson95"]) == 2 and out["side:v2"]["right"] == 0


def test_luna_summary_replaces_the_batch_list_by_its_range():
    s = PL.luna_summary({"batches": ["b0018", "b0019", "b0368"], "lines": 3, "agree_of_both_read": [1, 2, 0.5]})
    assert s == {"batches": "b0018-b0368 (3)", "lines": 3, "agree_of_both_read": [1, 2, 0.5]}


def test_hygiene_blockers_skip_information_only_checks():
    hits = {"test_games.all": {"vn": [], "sizes": {}}, "CLANNAD v4": {"vn": ["v4"], "sizes": {}},
            "CLANNAD 2nd ring": {"vn": ["v18"], "sizes": {}}, "family relation to test VN": {"vn": ["v1"], "pairs": {}},
            "devnative cohort: relations of every kind (2nd ring)": {"vn": ["v7318"]}}
    assert HY.blockers(hits) == {"CLANNAD v4": {"vn": ["v4"]}}


def test_hygiene_dump_relations_and_pool_hits(tmp_path):
    (tmp_path / "vn_relations").write_text("v4\tv11057\tchar\tf\nv16901\tv16902\tseq\tt\n")
    (tmp_path / "vn_titles").write_text("v64642\tja\tt\tQUIZ de CLANNAD\t\\N\nv4\tja\tt\tCLANNAD\t\\N\n")
    (tmp_path / "releases_vn").write_text("r1\tv4\nr1\tv999\nr2\tv7\n")
    (tmp_path / "releases_titles").write_text("r2\tja\tt\tクラナド 光坂\t\\N\n")
    d = HY.Dump(tmp_path)
    assert d.relations({"v4"}) == {"v11057"} and d.relations({"v11057"}) == {"v4"}
    assert d.family_relations({"v16901"}) == {"v16902"} and d.family_relations({"v4"}) == set()
    assert d.partners({"v4"}) == {"v999"}
    assert d.titled(HY.CLANNAD_TITLE) == {"v64642", "v4", "v7"}
    rows = {"v999_5": {"vn": "v999", "image": 5, "release": "r1", "sha256": "a" * 64, "pixel_sha256": "b" * 64}}
    pool = HY.Pool({"v999_5", "v3_6"}, rows)
    h = pool.hit(vns={"v999", "v8"}, imgs={6, 7}, hashes={"b" * 64}, rels={"r1"})
    assert h["vn"] == ["v999"] and h["image"] == [6] and h["hash"] == ["b" * 64] and h["release"] == ["r1"]
    assert h["sizes"] == {"vns": 2, "images": 2, "hashes": 1, "releases": 1}


def test_parsed_leaves_a_frozen_batch_untouched(tmp_path):
    (tmp_path / "batch.json").write_text(json.dumps({"mode": "pass1", "lines": 1, "sheets": 1,
                                                     "index": [{"sheet": "S001", "n": 1, "id": "v1_1_0"}]}))
    (tmp_path / "S001.json").write_text(json.dumps({"sheet": "S001", "entries": [
        {"n": 1, "id": "v1_1_0", "candidates": [{"c": 1, "text": "あい"}]}]}, ensure_ascii=False))
    (tmp_path / "decisions.jsonl").write_text(json.dumps({"sheet": "S001", "n": 1, "pick": 1}))
    recs = D.parsed(tmp_path)
    cache = tmp_path / "decisions.parsed.jsonl"
    assert recs[0]["text"] == "あい" and cache.exists()
    old = time.time() - 1000
    os.utime(cache, (old, old))
    D.parsed(tmp_path)
    assert abs(cache.stat().st_mtime - old) < 1                      # unchanged content: not rewritten


def test_parse_lenient_counts_a_reading_plus_skip_as_the_skip():
    text = json.dumps({"sheet": "S001", "n": 1, "pick": 1, "skip": "ui"}) + "\n" + \
        json.dumps({"sheet": "S001", "n": 2, "text": "名前", "unsure": True})
    got, bad = LU.parse_lenient(text, _sheets(), "pass1", "b0019")
    assert bad == 0
    assert got["v1_1_0"]["decision"] == "skip" and got["v1_1_0"]["skip"] == "ui" and got["v1_1_0"]["text"] is None
    assert got["v1_1_1"]["decision"] == "unsure"
    assert "[also read a candidate]" in got["v1_1_0"]["note"]


def test_helper_batches_edge_new_leaves_v2s_withdrawn_edge_batches_for_last(tmp_path, monkeypatch):
    for n in (19, 20, 30, 31, 32):
        d = tmp_path / "edge" / f"b{n:04d}"
        d.mkdir(parents=True)
        (d / "batch.json").write_text("{}")
    monkeypatch.setattr(PL, "REVIEW", tmp_path)
    assert [b.name for b in PL.helper_batches("edge-new")] == ["b0031", "b0032"]
    assert [b.name for b in PL.helper_batches("edge")] == ["b0019", "b0020", "b0030", "b0031", "b0032"]
    assert PL.helper_batches("pass1") == []


def test_plan_left_counts_unbatched_plan_lines_of_the_tiers(tmp_path, monkeypatch):
    monkeypatch.setattr(PL, "REVIEW", tmp_path)
    (tmp_path / "plan.jsonl").write_text("".join(json.dumps({"id": i, "tier": t}) + "\n"
                                                 for i, t in (("a", "t1"), ("b", "t1"), ("c", "t2"), ("d", "t3"))))
    b = tmp_path / "pass1" / "b0001"
    b.mkdir(parents=True)
    (b / "batch.json").write_text(json.dumps({"index": [{"id": "a"}]}))
    assert PL.plan_left("t1,t2") == 2 and PL.plan_left("t1") == 1 and PL.plan_left() == 3
    (b / "batch.json").write_text(json.dumps({"index": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}))
    assert PL.plan_left("t1,t2") == 0


def test_make_batches_renders_in_a_heavy_slot_unless_already_in_one(monkeypatch):
    calls = []
    monkeypatch.setattr(PL.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(PL, "cmd_batches", lambda a: calls.append(("inline", a.n, a.tiers)))
    monkeypatch.setenv("HERD", "/herd")
    monkeypatch.delenv("HEAVY_SLOT", raising=False)
    PL.make_batches(4, 200, "t1,t2")
    assert calls[-1][:3] == ["/herd/heavy.sh", "timeout", "3600"] and calls[-1][-2:] == ["--tiers", "t1,t2"]
    monkeypatch.setenv("HEAVY_SLOT", "3")
    PL.make_batches(4, 200, "t1,t2")
    assert calls[-1] == ("inline", 4, "t1,t2")


def test_build_exclusions_reserve_by_todays_rule_and_drop_reclaim_copies(tmp_path, monkeypatch):
    """The build (and hygiene) leave out the VNs shots.reserved_vns reserves today (the committed reserved.json
    unioned in) and the dedupe copies of both v2's selection and the reclaim tier; a reserved_vns.json saved at
    selection under the old rules is not read. The selection rows of both sources come back by shot key."""
    v2 = [{"vn": "v1", "image": 10}, {"vn": "v4", "image": 11}, {"vn": "v2", "image": 12}]
    rcl = [{"vn": "v9", "image": 20}, {"vn": "v9", "image": 21}]
    monkeypatch.setattr(PL.SH, "load_shots", lambda: v2)
    monkeypatch.setattr(PL, "reclaim_shot_rows", lambda: rcl)
    monkeypatch.setattr(PL.SH, "load_dump", lambda *a: {})
    monkeypatch.setattr(PL.SH, "reserved_vns", lambda d: {"v4": "clannad"})
    monkeypatch.setattr(PL, "dups", lambda: {"12": "pool:x"})
    monkeypatch.setattr(PL, "reclaim_dups", lambda: {"21": "pool:eval-candidates"})
    monkeypatch.setattr(PL, "OUT", tmp_path)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "reserved_vns.json").write_text(json.dumps({"v1": "test-vn"}))   # old rules: ignored
    (tmp_path / "manifests" / "leak_drop.json").write_text(json.dumps(["v1_10"]))
    current, vns, shots, src = PL.build_exclusions()
    assert current == {"v4": "clannad"} and vns == {"v4"}
    assert shots == {"v2_12", "v9_21", "v1_10"}
    assert set(src) == {"v1_10", "v4_11", "v2_12", "v9_20", "v9_21"}


def test_hygiene_final_blockers_narrow_to_clannad_and_set_games():
    hits = {"CLANNAD v4": {"vn": ["v4"]}, "test_games.vita-test": {"vn": ["v10"]},
            "test_games.psp-test": {"vn": ["v20"]}, "test_games.all": {"vn": ["v99"]},
            "is_test_vn": {"vn": ["v31"]}, "val-bucket": {"vn": ["v32"]},
            "teacher_val": {"vn": ["v33"]}, "owner test series (Memories Off, Eustia, 428): series, relations, "
            "release partners": {"vn": ["v1152"]}, "eval vita-test": {"vn": ["v10"]},
            "reserved.json test": {"vn": ["v4"]}, "reserved.json dev-sample": {"vn": ["v777"]},
            "leak-check drops": {"shot": ["v9_1"]}}
    final = HY.blockers(hits, final=True)
    assert set(final) == {"CLANNAD v4", "test_games.vita-test", "test_games.psp-test", "eval vita-test",
                          "reserved.json test", "reserved.json dev-sample", "leak-check drops"}
    # superset policy still blocks everything
    assert "is_test_vn" in HY.blockers(hits, final=False) and "teacher_val" in HY.blockers(hits, final=False)


def test_load_reads_merges_the_reclaim_tier(tmp_path, monkeypatch):
    import pool as PL
    main = tmp_path / "reads.jsonl"
    main.write_text(json.dumps({"shot": "v1_1", "vn": "v1", "kind": "vita"}) + "\n")
    rcl = tmp_path / "reclaim" / "reads" / "reads.jsonl"
    rcl.parent.mkdir(parents=True)
    rcl.write_text(json.dumps({"shot": "v9_9", "vn": "v9", "kind": "vita",
                               "path": "/x/reserve-policy/realplus/shots/hires/v9_9.jpg"}) + "\n")
    monkeypatch.setattr(PL, "READS", main)
    monkeypatch.setattr(PL, "RECLAIM_READS", rcl)
    shots = PL.load_reads()
    assert set(shots) == {"v1_1", "v9_9"}                       # both sources merged
    assert shots["v9_9"]["path"].endswith("reserve-policy/realplus/shots/hires/v9_9.jpg")
    assert set(PL.load_reads(main)) == {"v1_1", "v9_9"}         # path==READS also triggers the merge
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"shot": "v2_2", "vn": "v2", "kind": "vita"}) + "\n")
    assert set(PL.load_reads(other)) == {"v2_2"}                # a different path: no merge


def test_helpers_skip_a_batch_the_agent_reserved_for_itself(tmp_path, monkeypatch):
    """pool.py reserve marks a batch self-<name>: the feed queues no helper for it while it is open, but does for
    an open batch nobody holds (and again for the reserved one once the reservation is cancelled)."""
    for n in (1, 2):
        d = tmp_path / "pass1" / f"b{n:04d}"
        d.mkdir(parents=True)
        (d / "batch.json").write_text("{}")
    monkeypatch.setattr(PL, "REVIEW", tmp_path)
    monkeypatch.setattr(PL, "HERD_DIR", tmp_path)
    monkeypatch.setattr(PL, "live_helpers", lambda: set())
    queued = []
    monkeypatch.setattr(PL.subprocess, "run", lambda cmd, **kw: queued.append(cmd[1]))
    PL.cmd_reserve(argparse.Namespace(batch=[str(tmp_path / "pass1" / "b0001")], model="claude-opus-5.5"))
    PL.cmd_helpers(argparse.Namespace(max=6, kinds="pass1", retry=True))
    log = PL.helper_log()
    assert [h["batch"].rsplit("/", 1)[1] for h in log] == ["b0001", "b0002"] and len(queued) == 1
    assert log[0]["label"] == "self-b0001" and log[1]["label"] == f"{PL.HELPER}1"

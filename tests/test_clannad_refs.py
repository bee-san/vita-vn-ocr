"""Synthetic, offline checks of eval/clannad_refs.py: decision parsing, the two-of-three reference rule,
candidate filtering, page-safe batches, page-bootstrap scoring, the public guard and one end-to-end
collect -> refs -> score run (plus a post-freeze extra arm) on a tiny fake session. No CLANNAD data."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


CR = module("clannad_refs_tested", "clannad_refs.py")


def sheet(ids_cands, sid="S001"):
    """One sheet {sid: {n: entry}} from [(line id, [candidate texts])]."""
    return {sid: {n: {"n": n, "id": i, "candidates": [{"c": c, "text": t} for c, t in enumerate(cs, 1)]}
                  for n, (i, cs) in enumerate(ids_cands, 1)}}


def dumps(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def rec(kind, value=None):
    return {"decision": kind, "text": value if kind in ("pick", "text") else None,
            "skip": value if kind == "skip" else None, "name": False}


# ----------------------------------------------------------------------------- decisions

def test_parse_decisions_accepts_each_kind_and_the_aliases():
    sh = sheet([("a-0", ["「あ」", "「ぁ」"]), ("a-1", ["智代"]), ("a-2", []), ("a-3", ["x"])])
    text = dumps([{"sheet": "S001", "n": 1, "p": 2},
                  {"sheet": "S001", "n": 2, "pick": 1, "name": True},
                  {"sheet": "S001", "n": 2, "pick": 1, "name": True},       # repeated identically: one record
                  {"sheet": "S001", "n": 3, "s": "notext"},
                  {"sheet": "S001", "n": 4, "text": "「えーと…」", "note": "typed"}]) + "\n"
    recs = CR.parse_decisions(text, sh, "b01")
    assert [r["id"] for r in recs] == ["a-0", "a-1", "a-2", "a-3"]
    assert [r["decision"] for r in recs] == ["pick", "pick", "skip", "text"]
    assert recs[0]["text"] == CR.canon("「ぁ」") and recs[0]["pick"] == 2
    assert recs[1]["name"] is True and recs[0]["name"] is False
    assert recs[2]["skip"] == "notext" and recs[3]["text"] == CR.canon("「えーと…」") and recs[3]["note"] == "typed"
    assert {r["batch"] for r in recs} == {"b01"} and {r["reviewer_kind"] for r in recs} == {"assistant"}


def test_parse_decisions_lists_every_problem_and_has_no_shortcut():
    sh = sheet([(f"a-{i}", ["「はい」"]) for i in range(7)])
    lines = ["not json",
             json.dumps({"sheet": "S001", "n": 1, "pick": 1, "skip": "cut"}),   # two kinds
             json.dumps({"sheet": "S001", "n": 2, "pick": 5}),                    # no such candidate
             json.dumps({"sheet": "S001", "n": 3, "skip": "blurry"}),             # not a skip kind
             json.dumps({"sheet": "S001", "n": 4, "text": "  "}),                 # empty after normalisation
             json.dumps({"sheet": "S002", "n": 1, "pick": 1}),                    # no such sheet
             json.dumps({"sheet": "S001", "n": 5, "pick": 1}),
             json.dumps({"sheet": "S001", "n": 5, "unsure": True}),               # decided twice, differently
             json.dumps({"sheet": "S001", "n": 6, "unsure": False})]              # no decision at all
    with pytest.raises(CR.DecisionError) as e:
        CR.parse_decisions("\n".join(lines), sh, "b01")
    msg = str(e.value)
    for want in ("line 1: not JSON", "needs exactly one of", "has no candidate 5", "skip must be one of",
                 "empty after normalisation", "no entry 'S002'", "decided twice, differently",
                 "6 entries without a decision", "S001 #7"):
        assert want in msg, want
    with pytest.raises(CR.DecisionError, match="7 entries without a decision"):
        CR.parse_decisions("", sh, "b01")                                       # every entry needs a line


# ----------------------------------------------------------------------------- the reference rule

def test_reference_rule_two_reads_with_one_key():
    def t(s):
        return rec("text", s)
    assert CR.decide_ref(t("「はい」"), rec("pick", "「はい」")) == ("ref", CR.canon("「はい」"), "r1=r2")
    assert not CR.needs_r3(rec("skip", "notext"), rec("unsure"))
    assert CR.decide_ref(rec("skip", "notext"), rec("skip", "cut"))[:2] == ("excluded", None)
    assert CR.needs_r3(t("まぁ"), t("まあ"))                                      # small kana are distinct keys
    assert CR.needs_r3(t("あ"), rec("unsure")) and CR.needs_r3(rec("skip", "multi"), t("あ"))
    with pytest.raises(ValueError, match="third read"):
        CR.decide_ref(t("まぁ"), t("まあ"))
    assert CR.decide_ref(t("まぁ"), t("まあ"), t("まぁ")) == ("ref", CR.canon("まぁ"), "two of three")
    assert CR.decide_ref(t("あ"), rec("unsure"), t("あ"))[0] == "ref"
    assert CR.decide_ref(t("あ"), rec("unsure"), rec("skip", "cut")) == ("excluded", None, "two of three non-text")
    assert CR.decide_ref(t("あ"), t("ぁ"), t("ア")) == ("excluded", None, "no two reads agree")
    with pytest.raises(ValueError):
        CR.needs_r3(t("あ"), None)


def test_plausible_and_distinct_candidates():
    assert CR.plausible("「はい」") and CR.plausible("A子")                     # half ASCII is still shown
    assert not CR.plausible("") and not CR.plausible(None) and not CR.plausible("20IOOEI.0")
    assert not CR.plausible("AB子")
    got = CR.distinct(["「はい」", None, "", "20IOOEI.0", "「はい」 ", "「いいえ」", "「はい」"])
    assert [x["key"] for x in got] == [CR.key("「はい」"), CR.key("「いいえ」")]
    assert got[0]["text"] == CR.canon("「はい」") and got[0]["engines"] == []


def test_r2_candidates_are_shuffled_per_line_and_unattributed():
    d1 = rec("text", "「はい」")
    eng = {"meiki": "「はい」", "v6s": "「はぃ」"}
    a = CR.r2_candidates("p00001-0", d1, eng, "「はい」", ["「はし」", "20IOOEI.0"])
    assert sorted(x["key"] for x in a) == sorted({CR.key(s) for s in ("「はい」", "「はぃ」", "「はし」")})
    assert a == CR.r2_candidates("p00001-0", d1, eng, "「はい」", ["「はし」", "20IOOEI.0"])     # deterministic
    assert all(set(x) == {"text", "key", "engines"} and x["engines"] == [] for x in a)        # no sources


def test_chunks_never_split_a_page():
    pages = ["a"] * 3 + ["b"] * 4 + ["c"] + ["d"] * 2
    out = CR.chunks([({"page": p}, None) for p in pages], 3)
    assert [len(c) for c in out] == [3, 4, 3]
    seen = [{e[0]["page"] for e in c} for c in out]
    assert all(not (seen[i] & seen[j]) for i in range(len(seen)) for j in range(i + 1, len(seen)))
    assert sum(map(len, out)) == len(pages)


# ----------------------------------------------------------------------------- scoring

def test_page_sums_summary_and_the_bootstrap_matches_vocr3_report():
    import vocr3_report as R
    units = [(1, 4, 0, 0, 2), (0, 2, 1, 0, 0), (2, 5, 0, 1, 3)]
    sums = CR.page_sums(units, ["p1", "p1", "p2"])
    assert sums == {"p1": [1, 6, 1, 2, 0, 2], "p2": [2, 5, 0, 1, 1, 3]}
    s = CR.summary(sums, ["p1", "p2"], CR.boot_idx(2, 200))
    assert (s["lines"], s["chars"], s["errors"], s["exact_lines"]) == (3, 11, 3, 1)
    assert s["cer"] == round(100 * 3 / 11, 4) and s["exact"] == round(100 / 3, 4) and s["hiragana_cer"] == 20.0
    rng = np.random.default_rng(5)
    ids = [f"p{i:03d}" for i in range(40)]
    n = rng.integers(1, 30, 40)
    ea, eb = rng.integers(0, 5, 40), rng.integers(0, 5, 40)
    ref = R.paired_bootstrap({p: (int(ea[i]), int(n[i])) for i, p in enumerate(ids)},
                             {p: (int(eb[i]), int(n[i])) for i, p in enumerate(ids)}, 2000, 0)
    ours = CR.diff_ci(ea, eb, n, CR.boot_idx(len(ids), 2000, 0))
    assert ours == pytest.approx(ref["ci95"], abs=1e-4)                          # the same resamples


def test_assert_public_refuses_text_fields_at_any_depth():
    CR.assert_public({"scores": {"systems": {"A": {"all": {"cer": 1.0}}}}, "x": [{"lines": 3}]})
    for bad in ({"text": "x"}, {"a": {"b": [{"pred": "x"}]}}, {"a": [[{"lens": ""}]]}, {"readings": []},
                {"reference_text": "x"}):
        with pytest.raises(ValueError):
            CR.assert_public(bad)


def test_parse_extra_needs_a_pinned_blob():
    sha = "ab" * 32
    assert CR.parse_extra(f"VOCR4=/b/x_w8.vocr#{sha}") == {"name": "VOCR4", "file": "/b/x_w8.vocr",
                                                            "sha256": sha, "finder": "grad"}
    assert CR.parse_extra(f"V4.b=x.vocr#{sha}@textbox")["finder"] == "textbox"
    for bad in ("VOCR4=x.vocr", f"VOCR4=x.vocr#{sha[:63]}", f"=x.vocr#{sha}", f"VOCR4=x.vocr#{sha}@other",
                f"V/4=x.vocr#{sha}"):
        with pytest.raises(SystemExit):
            CR.parse_extra(bad)


# ----------------------------------------------------------------------------- end to end (fake session)

LINES = [  # id, page, dark, Lens, cached, (r1, r2, r3) reads
    ("p00001-0", "p00001", False, "智代", "智代ム",
     ({"text": "智代", "name": True}, {"text": "智代", "name": True}, None)),
    ("p00001-1", "p00001", True, "「はい」", "「はい」", ({"pick": 1}, {"text": "「はい」"}, None)),
    ("p00002-0", "p00002", False, "「まあ、そう」", "「まぁ、そう」",
     ({"text": "「まぁ、そう」"}, {"text": "「まあ、そう」"}, {"text": "「まぁ、そう」"})),
    ("p00002-1", "p00002", False, "", "", ({"skip": "notext"}, {"skip": "notext"}, None)),
    ("p00003-0", "p00003", False, "「あ」", "あ", ({"text": "「あ」"}, {"unsure": True}, {"skip": "cut"})),
]


def write_batch(work, rnd, rows):
    """A hand-made batch (sheets JSON and decisions, no images) plus the round's registry."""
    d = work / rnd / "b01"
    d.mkdir(parents=True)
    k = CR.ROUNDS.index(rnd)
    entries = [{"n": n, "id": lid, "candidates": [{"c": 1, "text": lens or "「x」"}]}
               for n, (lid, _, _, lens, _, _) in enumerate(rows, 1)]
    (d / "S001.json").write_text(json.dumps({"sheet": "S001", "entries": entries}, ensure_ascii=False))
    (d / "batch.json").write_text(json.dumps({"round": rnd, "lines": len(rows), "index": [{"id": r[0]} for r in rows]}))
    (d / "decisions.jsonl").write_text(dumps([{"sheet": "S001", "n": n, **r[5][k]} for n, r in enumerate(rows, 1)]))
    CR.write_json(work / f"{rnd}.sheets.json", {"round": rnd, "ids": [r[0] for r in rows], "lines": len(rows)})
    return d


def fake_work(tmp_path):
    work = tmp_path / "work"
    rows = [{"id": i, "page": p, "dark": dark, "lens": lens, "cached": cached}
            for i, p, dark, lens, cached, _ in LINES]
    sha = CR.write_jsonl(work / "lines.jsonl", rows)
    CR.write_json(work / "lines.meta.json", {"lines_sha256": sha, "pages": 3, "excluded_pages": 1, "lines": len(rows),
                                             "lock_sha256": "l", "captures_sha256": "c", "comparisons_sha256": "m",
                                             "session": str(tmp_path / "session")})
    names = [x["name"] for x in CR.protocol()["arms"]]
    refs = {"p00001-0": "智代", "p00001-1": "「はい」", "p00002-0": "「まぁ、そう」", "p00002-1": "",
            "p00003-0": "「あ」"}
    arm_rows = [{"id": i, **{n: ("" if n == "v1" else "ー" if n == "VOCR3_H15" and i == "p00002-1" else refs[i])
                             for n in names}} for i in refs]
    asha = CR.write_jsonl(work / "arms.jsonl", arm_rows)
    CR.write_json(work / "arms.meta.json", {
        "arms_sha256": asha, "arms": {x["name"]: {"sha256": x["sha256"]} for x in CR.protocol()["arms"]},
        "resources": {}, "native": {}, "cache_check": {"equal": 5, "lines": 5}, "code_sha256": {},
        "environment": {"python": "3", "system": "Linux", "machine": "x86_64", "libc": "glibc", "packages": {}}})
    return work


def test_end_to_end_collect_refs_and_one_score(tmp_path, monkeypatch):
    work = fake_work(tmp_path)
    write_batch(work, "r1", LINES)
    r2 = write_batch(work, "r2", LINES[:3])
    with pytest.raises(SystemExit, match="covers 3 of 5 lines"):                 # R2 must cover every line
        CR.main(["collect", "--work", str(work), "--round", "r2", "--batches", str(work / "r2")])
    shutil.rmtree(r2)
    write_batch(work, "r2", LINES)
    for rnd in ("r1", "r2"):
        CR.main(["collect", "--work", str(work), "--round", rnd, "--batches", str(work / rnd)])
    with pytest.raises(ValueError, match="third read"):
        CR.main(["refs", "--work", str(work)])                                  # R3 is missing
    assert not (work / "refs.jsonl").exists() and not (work / "refs.freeze.json").exists()
    r3 = [x for x in LINES if x[5][2] is not None]
    write_batch(work, "r3", r3)
    CR.main(["collect", "--work", str(work), "--round", "r3", "--batches", str(work / "r3")])
    CR.main(["refs", "--work", str(work)])
    freeze = json.loads((work / "refs.freeze.json").read_text())
    assert freeze["scored_lines"] == 3 and freeze["protocol_sha256"] == CR.sha256_file(CR.PROTOCOL_PATH)
    refs = {r["id"]: r for r in CR.read_jsonl(work / "refs.jsonl")}
    assert refs["p00002-0"]["text"] == CR.canon("「まぁ、そう」") and refs["p00002-0"]["why"] == "two of three"
    assert refs["p00002-0"]["lens_disagree"] and not refs["p00001-1"]["lens_disagree"]
    assert refs["p00002-1"]["notext"] and refs["p00003-0"]["status"] == "excluded"
    assert refs["p00001-0"]["name"] and not refs["p00001-1"]["name"]

    changed = tmp_path / "protocol.json"                                         # a protocol edited after the freeze
    changed.write_text(CR.PROTOCOL_PATH.read_text(encoding="utf-8").replace('"schema": 1', '"schema": 1, "x": 0', 1))
    monkeypatch.setattr(CR, "PROTOCOL_PATH", changed)
    with pytest.raises(SystemExit, match="differs from the one the references were frozen under"):
        CR.main(["score", "--work", str(work), "--out", str(tmp_path / "r.json")])
    assert not (work / "score.attempt.json").exists()                           # the one attempt is not used up
    monkeypatch.undo()

    out, md = tmp_path / "result.json", tmp_path / "result.md"
    CR.main(["score", "--work", str(work), "--out", str(out), "--md", str(md)])
    res = json.loads(out.read_text())
    CR.assert_public(res)
    blob = out.read_text(encoding="utf-8")
    assert "まぁ" not in blob and "智代" not in blob                              # no text of any line
    s = res["scores"]["systems"]
    import metrics
    chars = sum(len(metrics.norm(refs[i]["text"])) for i in ("p00001-0", "p00001-1", "p00002-0"))
    assert s["VOCR1"]["all"]["errors"] == 0 and s["VOCR1"]["all"]["exact"] == 100.0
    assert s["v1"]["all"]["errors"] == chars and s["v1"]["all"]["cer"] == 100.0
    assert s["Lens"]["all"]["errors"] == 1 and s["Lens"]["all"]["exact_lines"] == 2      # まあ for まぁ
    assert s["cached"]["all"]["errors"] == 1 and s["cached"]["all"]["exact_lines"] == 2  # 智代ム
    assert s["VOCR1"]["name"]["lines"] == 1 and s["VOCR1"]["inverted"]["lines"] == 1
    assert res["scores"]["bootstrap"]["pages"] == 2
    assert res["scores"]["pairs"]["VOCR1:v1"]["cer_delta"] == -100.0
    assert set(res["scores"]["pairs"]) == {":".join(p) for p in CR.protocol()["pairs"]}
    assert res["references"]["flags"]["lens_disagree"] == 1 and res["references"]["flags"]["excluded_lines"] == 2
    assert res["references"]["agreement"]["r1_text_r2_text"] == 3 and res["references"]["agreement"]["same_key"] == 2
    assert res["references"]["agreement"]["r1_decisions"] == {"typed": 3, "pick": 1, "skip:notext": 1}
    assert res["references"]["agreement"]["r2_decisions"] == {"typed": 3, "skip:notext": 1, "unsure": 1}
    assert res["notext_output"]["VOCR3_H15"] == {"lines": 1, "lines_with_output": 1, "output_chars": 1}
    assert res["notext_output"]["VOCR1"]["lines_with_output"] == 0
    assert res["references"]["freeze"]["protocol_sha256"] == freeze["protocol_sha256"]
    assert md.read_text().startswith("| system |") and (work / "score.done.json").exists()
    with pytest.raises(FileExistsError):                                         # one attempt only
        CR.main(["score", "--work", str(work), "--out", str(tmp_path / "again.json")])

    # a later blob scored after the freeze, from another agent's directory: WORK is only read
    into = tmp_path / "vocr4"
    before = sorted(p.name for p in work.iterdir())
    esha = CR.write_jsonl(into / "arms.t1.jsonl", [{"id": i, "VOCR4": r["text"] or ""} for i, r in refs.items()])
    CR.write_json(into / "arms.t1.meta.json", {"arms_sha256": esha, "arms": {"VOCR4": {"sha256": "ab" * 32}},
                                               "resources": {}, "native": {}, "code_sha256": {},
                                               "after_freeze": freeze["refs_sha256"]})
    base = ["score", "--work", str(work), "--into", str(into)]
    with pytest.raises(FileNotFoundError):                                       # no arms.t2.jsonl
        CR.main(base + ["--out", str(into / "x.json"), "--suffix", "t2", "--pair", "VOCR4:VOCR1"])
    for bad in ([], ["--pair", "VOCR4:NOPE"], ["--pair", "VOCR4"]):
        with pytest.raises(SystemExit, match="--pair"):
            CR.main(base + ["--out", str(into / "y.json"), "--suffix", "t1"] + bad)
    with pytest.raises(SystemExit, match="go with --suffix"):
        CR.main(["score", "--work", str(work), "--out", str(into / "z.json"), "--pair", "VOCR4:VOCR1"])
    assert not list(into.glob("score*.attempt.json"))                            # usage errors use no attempt
    eo = into / "extra.json"
    CR.main(base + ["--out", str(eo), "--suffix", "t1", "--pair", "VOCR4:VOCR1"])
    extra = json.loads(eo.read_text())
    assert extra["kind"] == "clannad-refs-extra-result" and list(extra["scores"]["pairs"]) == ["VOCR4:VOCR1"]
    assert extra["scores"]["systems"]["VOCR4"]["all"]["errors"] == 0
    assert sorted(p.name for p in work.iterdir()) == before                     # nothing written into WORK

"""eval/textspot6_test.py's guards: nothing protected is read unless the selection is committed and pushed, the final
run is the pick's retrain, the output is private, and only once; a dry run never touches a protected input."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "synth"), str(ROOT / "train"), str(ROOT / "runtime"), str(ROOT)]
import textspot6_test as T  # noqa: E402


def selection(tmp_path, pick=None, kind="textspot6-selection"):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"kind": kind, "pick": pick}))
    return p


PICK = {"arm": "base_qS4", "decode": "crop", "seed": 2}


def final_run(tmp_path, recipe="base", shape="qS4", seed=2, splits="train,guard,hval", done=True):
    d = tmp_path / "final_run"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"recipe": recipe, "shape": {"name": shape}, "seed": seed,
                                            "args": {"hook_splits": splits}}))
    (d / "final.vtsp").write_bytes(b"VTSP")
    if done:
        (d / "done.json").write_text("{}")
    return d


def test_selection_kind_and_negative_verdict(tmp_path):
    with pytest.raises(SystemExit, match="not a textspot6 selection"):
        T.load_selection(selection(tmp_path, PICK, kind="other"), dry=True)
    with pytest.raises(SystemExit, match="nothing is read"):
        T.load_selection(selection(tmp_path, None), dry=True)


def test_selection_must_be_committed_for_a_real_read(tmp_path):
    with pytest.raises(SystemExit, match="must be committed"):
        T.load_selection(selection(tmp_path, PICK), dry=False)
    sel, rec = T.load_selection(selection(tmp_path, PICK), dry=True)
    assert sel["pick"] == PICK and rec["dry_run"]


def test_research_read_only_with_a_negative_verdict(tmp_path):
    best = {"arm": "base_qS4", "decode": "crop", "seed": 2, "dev_cer": 1.39}
    p = tmp_path / "neg.json"
    p.write_text(json.dumps({"kind": "textspot6-selection", "pick": None, "best_eligible_if_negative": best}))
    with pytest.raises(SystemExit, match="nothing is read"):
        T.load_selection(p, dry=True)
    sel, rec = T.load_selection(p, dry=True, research=True)
    assert sel["pick"]["research"] and sel["pick"]["arm"] == "base_qS4"
    blob, info = T.check_final(final_run(tmp_path, splits="train"), sel, dry=False)     # its own run, as trained
    assert info["hook_splits"] == "train"
    with pytest.raises(SystemExit, match="research release of a negative verdict only"):
        T.load_selection(selection(tmp_path, PICK), dry=True, research=True)


def test_final_run_is_the_picks_retrain(tmp_path):
    sel = {"pick": PICK}
    blob, info = T.check_final(final_run(tmp_path), sel, dry=False)
    assert blob.name == "final.vtsp" and info["hook_splits"] == "train,guard,hval"


@pytest.mark.parametrize("kw", [{"splits": "train"}, {"seed": 1}, {"shape": "qL5"}, {"recipe": "kd"}])
def test_final_run_mismatch_refused(tmp_path, kw):
    with pytest.raises(SystemExit, match="final retrain"):
        T.check_final(final_run(tmp_path, **kw), {"pick": PICK}, dry=False)
    T.check_final(tmp_path / "final_run", {"pick": PICK}, dry=True)     # a dry run does not check it


def test_unfinished_final_refused(tmp_path):
    with pytest.raises(SystemExit, match="not a finished run"):
        T.check_final(final_run(tmp_path, done=False), {"pick": PICK}, dry=True)


def test_outputs_private_for_real_reads_only(tmp_path):
    with pytest.raises(SystemExit, match="a real read writes under"):
        T.guard_out(tmp_path / "out", dry=False)
    with pytest.raises(SystemExit, match="a real read writes under"):
        T.guard_out(T.PROTECTED_OUT / "x", dry=True)
    assert T.guard_out(tmp_path / "out", dry=True) == tmp_path / "out"
    assert T.guard_out(T.PROTECTED_OUT / "x", dry=False) == T.PROTECTED_OUT / "x"


def test_once_refuses_a_second_attempt(tmp_path):
    T.once(tmp_path / "a" / "title.attempt.json", {"x": 1})
    with pytest.raises(FileExistsError):
        T.once(tmp_path / "a" / "title.attempt.json", {"x": 2})


def test_title_split_rules_before_any_attempt(tmp_path):
    sel, run = selection(tmp_path, PICK), final_run(tmp_path)
    base = ["title", "--selection", str(sel), "--final", str(run), "--out", str(tmp_path / "o"),
            "--pagepath", str(tmp_path / "pagepath.py")]
    with pytest.raises(SystemExit, match="non-protected split"):
        T.main(base + ["--dry-run"])                       # a dry run of split test
    with pytest.raises(SystemExit, match="split test, every title"):
        T.main(base + ["--split", "hval"])                 # a real read of another split
    with pytest.raises(SystemExit, match="split test, every title"):
        T.main(base + ["--titles", "PCSG00807"])           # a real read of part of the split
    assert not (tmp_path / "o").exists()


def test_clannad_work_rules(tmp_path):
    sel, run = selection(tmp_path, PICK), final_run(tmp_path)
    base = ["clannad", "--selection", str(sel), "--final", str(run), "--out", str(tmp_path / "o"),
            "--pagepath", str(tmp_path / "pagepath.py")]
    with pytest.raises(SystemExit, match="stand-in"):
        T.main(base + ["--dry-run"])                       # a dry run of CLANNAD's own work directory
    with pytest.raises(SystemExit, match="stand-in"):
        T.main(base + ["--work", str(tmp_path)])           # a real read of anything but CLANNAD's
    assert not (tmp_path / "o").exists()


def test_standin_never_in_the_private_tree(tmp_path):
    with pytest.raises(SystemExit, match="outside the private tree"):
        T.main(["standin", "--units", str(tmp_path / "u.jsonl"), "--out", str(T.PRIV / "vocr6-spotter" / "x")])


def test_keyed_refs_match_grad_boxes():
    refs = {"a": {"status": "ref"}, "b": {"status": "noref"}, "c": {"status": "ref"}}
    rs = [{"id": "a", "box": [1, 2, 30, 10]}, {"id": "b", "box": [1, 20, 30, 10]}, {"id": "c", "box": [0, 0, 5, 5]}]
    got = T.keyed_refs(rs, refs, [(1, 20, 30, 10), (1, 2, 30, 10)])
    assert [(r["id"], k) for r, k in got] == [("a", 1)]

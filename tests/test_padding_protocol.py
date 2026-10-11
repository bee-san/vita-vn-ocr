"""Generated manifests and predictions verify the independent preprocessing gate."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import padding_protocol as PP  # noqa: E402


def test_expansion_requires_same_count_and_vertical_geometry():
    original = [(0, 8, 27, 20), (52, 30, 27, 20)]
    PP.verify_expansion(original, [(0, 8, 31, 20), (48, 30, 32, 20)], 80)
    for changed in [[(0, 8, 31, 20)], [(0, 7, 31, 21), (48, 30, 32, 20)], original]:
        with pytest.raises(ValueError, match="detections or vertical"):
            PP.verify_expansion(original, changed, 80)


def test_expanded_matching_cannot_hide_old_unmatched_line_and_false_positive():
    gts = [{"id": "line", "text": "あ", "box_r": (12, 0, 10, 10)}]
    predictions = ["あ"]
    fixed_rows, fixed_counts, old_pairs = PP.scored([(0, 0, 10, 10)], gts, predictions)
    new_rows, new_counts, new_pairs = PP.scored([(0, 0, 14, 10)], gts, predictions)
    assert old_pairs == [] and new_pairs == [(0, 0)]
    assert fixed_counts == {"found_lines": 0, "missed_lines": 1, "false_positive_detections": 1}
    assert new_counts == {"found_lines": 1, "missed_lines": 0, "false_positive_detections": 0}
    assert PP.metrics.score_lines(fixed_rows)["all"]["errors"] == 2
    assert PP.metrics.score_lines(new_rows)["all"]["errors"] == 0


def test_uncertain_matches_are_ignored_but_real_missing_lines_are_charged():
    gts = [{"text": "あ", "box_r": (0, 0, 20, 10), "uncertain": True},
           {"text": "いう", "box_r": (0, 20, 20, 10)}]
    rows, counts, _ = PP.scored([(0, 0, 20, 10)], gts, ["え"])
    assert rows == [{"text": "いう", "pred": ""}]
    assert counts == {"found_lines": 0, "missed_lines": 1, "false_positive_detections": 0}


def passing_results():
    score = {"baseline": {"full": {"chars": 20, "errors": 4}, "hiragana": {"chars": 10, "errors": 2}},
             "candidate_fixed_matching": {"full": {"chars": 20, "errors": 3},
                                          "hiragana": {"chars": 10, "errors": 1}},
             "candidate_emitted_matching": {"full": {"chars": 20, "errors": 3},
                                            "hiragana": {"chars": 10, "errors": 1}}}
    return {name: {"scores": copy.deepcopy(score), "matching_changed_regions": 0}
            for name in ("vita-test", "psp-test", "clannad-test")}


def test_gate_requires_improvement_without_regression_under_both_matching_methods():
    result = passing_results()
    assert PP.gate(result)["accepted"]
    result["vita-test"]["scores"]["candidate_fixed_matching"]["full"]["errors"] = 5
    assert not PP.gate(result)["accepted"]
    result = passing_results()
    result["clannad-test"]["scores"]["candidate_fixed_matching"]["hiragana"]["errors"] = 2
    assert not PP.gate(result)["accepted"]  # better expanded matching alone cannot pass
    result = passing_results()
    result["psp-test"]["scores"]["candidate_emitted_matching"]["full"]["errors"] = 5
    assert not PP.gate(result)["accepted"]


def test_gate_rejects_incomplete_sets_and_changed_reference_counts():
    result = passing_results()
    del result["psp-test"]
    with pytest.raises(ValueError, match="all three"):
        PP.gate(result)
    result = passing_results()
    result["clannad-test"]["scores"]["candidate_fixed_matching"]["hiragana"]["chars"] = 9
    with pytest.raises(ValueError, match="reference counts"):
        PP.gate(result)


def test_dev_role_rejected_before_backend_or_pixels(tmp_path, monkeypatch):
    manifest = tmp_path / "capture.jsonl"
    manifest.write_text(json.dumps({"id": "1", "split": "test", "session": "A", "hand_checked": True,
                                    "region": [0, 0, 80, 40], "box": [8, 8, 30, 16], "text": "あ"}) + "\n")
    monkeypatch.setattr(PP, "sha", lambda path: PP.BLOB_SHA)
    monkeypatch.setattr(PP, "git", lambda *args: "baseline")
    monkeypatch.setattr(PP, "Backend", lambda *args: pytest.fail("backend opened before role guard"))
    args = SimpleNamespace(command="dev", out=tmp_path / "result.json", blob=tmp_path / "unused.vocr",
                           manifest=manifest)
    with pytest.raises(ValueError, match="hand-checked dev"):
        PP.evaluate(args)
    assert not list(tmp_path.glob("*.started.json"))


def test_run_requires_lock_hash_and_committed_candidate_before_loading_pixels(tmp_path, monkeypatch):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema": 1, "purpose": "one_candidate_preprocessing_holdout_lock",
                               "rule": PP.RULE, "heldout_evaluated": False, "candidate_revision": "locked"}))
    args = SimpleNamespace(command="run", out=tmp_path / "result.json", lock=lock, lock_sha256="wrong")
    with pytest.raises(ValueError, match="preregistered SHA256"):
        PP.evaluate(args)
    args.lock_sha256 = PP.sha(lock)
    monkeypatch.setattr(PP, "git", lambda *args: "different")
    monkeypatch.setattr(PP, "Backend", lambda *args: pytest.fail("opened backend"))
    with pytest.raises(ValueError, match="locked clean commit"):
        PP.evaluate(args)


def test_completed_or_started_records_cannot_be_overwritten(tmp_path):
    path = tmp_path / "experiment.used.json"
    PP.write_new(path, {"started": True})
    with pytest.raises(FileExistsError):
        PP.write_new(path, {"started": False})
    assert json.loads(path.read_text()) == {"started": True}


def test_source_lock_rejects_uncommitted_or_changed_protocol_files(tmp_path, monkeypatch):
    source = tmp_path / "protocol.py"
    source.write_text("committed\n")
    monkeypatch.setattr(PP, "ROOT", tmp_path)
    monkeypatch.setattr(PP, "SOURCES", (source.name,))
    monkeypatch.setattr(PP, "source_at", lambda revision, path: b"committed\n")
    assert PP.committed_sources("head") == {source.name: PP.sha(source)}
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="differs from locked commit"):
        PP.committed_sources("head")

    def missing(revision, path):
        raise subprocess.CalledProcessError(128, ["git", "show"])

    monkeypatch.setattr(PP, "source_at", missing)
    with pytest.raises(ValueError, match="absent from locked commit"):
        PP.committed_sources("head")


@pytest.fixture
def locked_mock(tmp_path, monkeypatch):
    # No images or weights: preflight identities are real tiny manifests, while
    # the recognizer and scored reports are isolated from this guard regression.
    row = {"id": "1", "split": "test", "session": "A", "hand_checked": True,
           "region": [0, 0, 80, 40], "box": [8, 8, 30, 16], "text": "あ"}
    for name in ("vita-test", "psp-test", "clannad-test"):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(PP.sets, "MANIFESTS", tmp_path)
    monkeypatch.setattr(PP, "git", lambda *args: "" if args[0] == "status"
                        else "candidate" if args[-1] == "HEAD" else "baseline")
    monkeypatch.setattr(PP, "source_at", lambda revision, path: (ROOT / path).read_bytes())
    source_hashes = {path: PP.sha(ROOT / path) for path in PP.SOURCES}
    monkeypatch.setattr(PP, "committed_sources", lambda revision: source_hashes)
    blob = tmp_path / "generated.vocr"
    blob.write_bytes(b"generated fixture only")
    monkeypatch.setattr(PP, "BLOB_SHA", PP.sha(blob))
    record = {"schema": 1, "purpose": "one_candidate_preprocessing_holdout_lock", "rule": PP.RULE,
              "heldout_evaluated": False, "candidate_revision": "candidate", "baseline_revision": "baseline",
              "source_sha256": source_hashes, "artifact_dir": str(tmp_path),
              "baseline_source_sha256": {path: PP.sha(ROOT / path)
                                         for path in ("runtime/vocr_lines.c", "runtime/vocr_lines.h")},
              "blob": str(blob), "blob_sha256": PP.sha(blob),
              "datasets": PP.inputs(tmp_path / "clannad-test.jsonl", tmp_path)}
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(record))
    return path, record


def test_all_dataset_roles_checked_before_any_backend_or_pixels(locked_mock, tmp_path, monkeypatch):
    path, record = locked_mock
    monkeypatch.setattr(PP, "Backend", lambda *args: pytest.fail("backend opened before all roles validated"))
    args = SimpleNamespace(command="run", lock=path, lock_sha256=PP.sha(path), out=tmp_path / "report.json")
    record["source_sha256"] = {}
    path.write_text(json.dumps(record))
    args.lock_sha256 = PP.sha(path)
    with pytest.raises(ValueError, match="source hashes"):
        PP.evaluate(args)
    record["source_sha256"] = {p: PP.sha(ROOT / p) for p in PP.SOURCES}
    record["datasets"]["clannad-test"]["split"] = "dev"
    path.write_text(json.dumps(record))
    args.lock_sha256 = PP.sha(path)
    with pytest.raises(ValueError, match="dataset roles"):
        PP.evaluate(args)
    assert not list(tmp_path.glob("*.used.json"))


def test_same_lock_cannot_rerun_with_another_output(locked_mock, tmp_path, monkeypatch):
    path, _ = locked_mock
    calls = []

    def backend(directory, blob, baseline=None):
        calls.append(directory)
        return SimpleNamespace(binary=blob, command=[], max_finder_arena_bytes=0,
                               runtime=SimpleNamespace(blob_bytes=657664, arena_bytes=124832))

    monkeypatch.setattr(PP, "Backend", backend)
    result = passing_results()
    monkeypatch.setattr(PP, "score_dataset", lambda identity, old, new: result[Path(identity["manifest"]).stem])
    args = SimpleNamespace(command="run", lock=path, lock_sha256=PP.sha(path), out=tmp_path / "first.json")
    PP.evaluate(args)
    assert len(calls) == 2
    args.out = tmp_path / "second.json"
    with pytest.raises(FileExistsError):
        PP.evaluate(args)
    assert len(calls) == 2 and not args.out.exists()

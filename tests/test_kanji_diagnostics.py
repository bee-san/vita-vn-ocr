"""Kanji attribution and native-report provenance, with generated strings only."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import kanji_diagnostics as KD  # noqa: E402


@pytest.fixture
def policy():
    return {"vocab_sha256": "f" * 64, "ranks": {"土": 454, "士": 281, "巳": 2533, "已": 3222,
        "己": 815, "龜": 5001, "寢": 5010, "日": 17, "曰": 2032, "猫": 300, "本": 94},
        "pairs": ["土士", "己已", "已巳"]}


def native_report(rows):
    return {"model": "vocr:/private/fixture.vocr", "model_name": "fixture/w8i", "finder": None,
            "model_info": {"backend": "vocr", "mode": "w8i", "sha256": "a" * 64},
            "datasets": {"fixture-dev": {"manifest_sha256": "b" * 64, "split": "dev", "sessions": ["A"]}},
            "provenance": {"dev_only": True, "test_opened": False}, "evaluation_role": "development",
            "sets": {"fixture-dev": {"lines": rows}}}


def test_full_alignment_charges_reference_han_and_lists_into_han_separately(policy):
    result = KD.score([{"text": "土あ", "pred": "あ土"}], policy)
    assert result["kanji"]["chars"] == result["kanji"]["errors"] == 1
    assert result["kanji"]["cer"] == 1.0 and result["kanji"]["into_kanji"] == 1
    assert result["into_kanji_by_prediction_band"]["common"] == 1
    assert result["similar"]["substitutions"] == 0


def test_reference_rank_bands_exclude_insertions_and_preserve_unsupported(policy):
    rows = [{"text": g, "pred": p} for g, p in [("土ぁ", "士ぁ"), ("巳", ""), ("已", "己"),
        ("龜", ""), ("𠮷", "あ"), ("", "寢"), ("あ", "土"), ("日", "曰"), ("猫", "猫"), ("あ 本い", "あ本い")]]
    result = KD.score(rows, policy)
    total = result["kanji"]
    assert (total["chars"], total["substitutions"], total["deletions"], total["insertions"]) == (8, 4, 2, 1)
    assert total["errors"] == 7 and total["cer"] == .875 and total["into_kanji"] == 1
    assert result["insertions_by_prediction_band"]["very_rare"] == 1
    assert result["reference_rank_bands"]["common"]["reference_error_rate"] == .5
    assert result["reference_rank_bands"]["unsupported"]["substitutions"] == 1
    assert sum(b["errors"] for b in result["reference_rank_bands"].values()) == 6
    assert result["similar"]["substitutions"] == 2 and result["similar"]["reference_chars"] == 3
    assert {r["pair"] for r in result["similar"]["confusions"]} == {"土士", "己已"}
    assert all(r["pair"] != "日曰" for r in result["similar"]["confusions"])


def test_normalization_precedes_han_and_similarity_attribution(policy):
    result = KD.score([{"text": "あ ⼟い", "pred": "あ士い"}], policy)
    assert result["kanji"]["chars"] == result["similar"]["substitutions"] == 1
    assert result["similar"]["confusions"] == [{"reference": "土", "prediction": "士", "count": 1, "pair": "土士"}]


def test_zero_reference_bands_are_no_evidence_even_with_insertions(policy):
    result = KD.score([{"text": "", "pred": "寢"}], policy)
    assert result["kanji"]["cer"] is None and result["kanji"]["errors"] == 1
    assert result["kanji"]["evidence"] == result["similar"]["evidence"] == "no_evidence"
    assert all(b["reference_error_rate"] is None and b["evidence"] == "no_evidence"
               for b in result["reference_rank_bands"].values())


@pytest.mark.parametrize("rank,expected", [(2000, "common"), (2001, "uncommon"), (3000, "uncommon"),
    (3001, "rare"), (4000, "rare"), (4001, "very_rare")])
def test_rank_band_boundaries(rank, expected, policy):
    policy["ranks"]["土"] = rank
    assert KD.band("土", policy) == expected


def test_report_preserves_identity_and_roles_without_line_text(policy):
    report = native_report([{"text": "あ土い", "pred": "あ士い"}])
    before = copy.deepcopy(report)
    result = KD.diagnose(report, policy)
    assert report == before
    assert result["source"]["datasets"] == report["datasets"]
    assert result["source"]["provenance"] == report["provenance"]
    assert result["source"]["evaluation_role"] == "development"
    assert result["source"]["model_info"]["sha256"] == "a" * 64
    serialized = json.dumps(result, ensure_ascii=False)
    assert "あ土い" not in serialized and "あ士い" not in serialized


@pytest.mark.parametrize("change", [{"finder": "grad"}, {"model_info": {"backend": "torch"}}, {"sets": {}},
                                   {"sets": {"fixture-dev": {"lines": [{"text": "土", "pred": None}]}}}])
def test_refuses_finder_non_native_or_unmeasured_report(change, policy):
    with pytest.raises(ValueError):
        KD.diagnose(dict(native_report([]), **change), policy)


def test_cli_records_input_code_hashes_and_validates_pinned_policy(tmp_path, monkeypatch):
    import kanji_policy as KP
    jiten, similar, vocab = (tmp_path / name for name in ("jiten.csv", "similar.txt", "vocab.txt"))
    jiten.write_text("Kanji,Rank\n土,454\n士,281\n")
    similar.write_text("土/士/\n")
    vocab.write_text("土\n士\nあ\nい\n")
    monkeypatch.setattr(KP, "JITEN_SHA256", KP.sha256(jiten))
    monkeypatch.setattr(KP, "SIMILAR_SHA256", KP.sha256(similar))
    policy_path, report_path, output = (tmp_path / name for name in ("policy.json", "report.json", "out.json"))
    policy_path.write_text(json.dumps(KP.build_policy(jiten, similar, vocab)))
    report_path.write_text(json.dumps(native_report([{"text": "あ土い", "pred": "あ士い"}])))
    monkeypatch.setattr(sys, "argv", ["kanji_diagnostics", "--report", str(report_path), "--policy", str(policy_path),
                                     "--out", str(output)])
    KD.main()
    result = json.loads(output.read_text())
    assert result["inputs"]["report_sha256"] == KP.sha256(report_path)
    assert result["inputs"]["policy_sha256"] == KP.sha256(policy_path)
    assert result["inputs"]["metrics_code_sha256"] == KP.sha256(ROOT / "eval/metrics.py")
    assert result["sets"]["fixture-dev"]["similar"]["substitutions"] == 1
    policy = json.loads(policy_path.read_text())
    policy["pairs"] = []
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="content hash mismatch"):
        KD.main()

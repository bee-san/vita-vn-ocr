"""Pinned allowlists and rare-band bounds, using generated source tables only."""
import copy
import json

import numpy as np
import pytest

import kanji_policy as KP


@pytest.fixture
def policy(tmp_path, monkeypatch):
    jiten = tmp_path / "kanji.csv"
    jiten.write_text("Kanji,Rank\n人,1\n入,2\n日,50\n末,2000\n未,2500\n士,2000\n土,3500\n曰,4500\n龍,5000\n")
    similar = tmp_path / "similar.txt"
    similar.write_text("人/入/日/\n入/人/\n未/末/\n土/士/\n曰/日/\n龍/人/\n")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("人\n入\n日\n末\n未\n士\n土\n曰\n猫\nあ\nっ\n々\n！\n〓\n")
    monkeypatch.setattr(KP, "JITEN_SHA256", KP.sha256(jiten))
    monkeypatch.setattr(KP, "SIMILAR_SHA256", KP.sha256(similar))
    result = KP.build_policy(jiten, similar, vocab)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(result, ensure_ascii=False))
    return result, path, jiten, similar, vocab


@pytest.mark.parametrize("csv_text,message", [
    ("Kanji,Rank,Rank\n人,1,1\n", "header"),
    ("Rank,Kanji\n1,人\n", "header"),
    ("Kanji,Rank\n人,1\n人,2\n", "duplicate"),
    ("Kanji,Rank\n人,-1\n", "positive"),
    ("Kanji,Rank\n人,0\n", "positive"),
    ("Kanji,Rank\n人,1.5\n", "positive"),
    ("Kanji,Rank\nあ,1\n", "Han"),
    ("Kanji,Rank\n人人,1\n", "Han"),
    ("Kanji,Rank\n人,1,extra\n", "one rank"),
    ("Kanji,Rank\n", "empty"),
])
def test_csv_rejects_bad_headers_characters_duplicate_classes_and_ranks(tmp_path, csv_text, message):
    path = tmp_path / "kanji.csv"
    path.write_text(csv_text)
    with pytest.raises(ValueError, match=message):
        KP.read_jiten(path)


def test_positive_tied_ranks_are_valid(tmp_path):
    path = tmp_path / "kanji.csv"
    path.write_text("Kanji,Rank\n人,1\n入,1\n")
    assert KP.read_jiten(path) == {"人": 1, "入": 1}


def test_build_keeps_vocab_fixed_and_filters_graph_and_labels(policy):
    result, path, _, _, vocab = policy
    loaded = KP.load_policy(path)
    assert result == loaded
    assert loaded["vocab_sha256"] == KP.sha256(vocab)
    assert set(loaded["allowed_kanji"]) == set("人入日末未士土曰")
    assert loaded["coverage"]["missing_jiten_kanji"] == 1
    assert loaded["coverage"]["model_kanji_outside_jiten"] == 1
    assert KP.allowed_label("人あっ々！", loaded)
    assert not KP.allowed_label("猫", loaded)  # In the model, but absent from Jiten.
    assert not KP.allowed_label("龍", loaded)  # In Jiten, but absent from the model.
    assert not KP.allowed_label("あ?", loaded)  # Non-Han characters also retain the exact vocabulary.
    assert not KP.allowed_label("", loaded)
    assert "人龍" not in loaded["pairs"]
    assert "入日" not in loaded["pairs"]  # A neighbour list is not a complete clique.
    assert loaded["pairs"].count("人入") == 1


def test_source_hash_change_is_rejected(policy):
    _, _, jiten, similar, vocab = policy
    jiten.write_text(jiten.read_text() + "犬,7000\n")
    with pytest.raises(ValueError, match="pinned"):
        KP.build_policy(jiten, similar, vocab)


def test_policy_hash_and_schema_protect_the_derived_table(policy):
    original, path, _, _, _ = policy
    altered = copy.deepcopy(original)
    altered["ranks"]["人"] = 3
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="content hash"):
        KP.load_policy(path)
    path.write_text(json.dumps({**original, "schema": True}))
    with pytest.raises(ValueError, match="schema"):
        KP.load_policy(path)


@pytest.mark.parametrize("mutation,message", [
    (lambda p: p["provenance"]["jiten"].update(sha256="0" * 64), "source hashes"),
    (lambda p: p["ranks"].update(人=0), "positive integers"),
    (lambda p: p["ranks"].update(人=True), "positive integers"),
    (lambda p: p["pairs"].append("入日"), "canonical"),
    (lambda p: p["corrective"].update(total_training_probability=.03), "2%"),
    (lambda p: p["coverage"].update(supported_kanji=999), "coverage"),
    (lambda p: p.update(effective_bands={}), "effective band"),
])
def test_rehashed_invalid_policies_still_fail_schema_invariants(policy, mutation, message):
    original, path, _, _, _ = policy
    altered = copy.deepcopy(original)
    mutation(altered)
    altered["content_sha256"] = KP.content_sha256(altered)
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match=message):
        KP.load_policy(path)


def test_name_probabilities_favour_common_supported_kanji(policy):
    result = policy[0]
    weights = KP.name_weights(["人", "入", "曰", "龍"], result)
    assert sum(weights) == pytest.approx(1)
    assert weights[0] / weights[1] == pytest.approx(2)
    assert weights[0] / weights[2] == pytest.approx(4500)
    assert weights[3] == 0
    with pytest.raises(ValueError, match="no supported"):
        KP.name_weights(["龍", "猫"], result)


def test_endpoint_rank_defines_band_and_empty_bands_move_mass_toward_common(policy):
    result = policy[0]
    assert KP.rank_band("人入", result) == "common"
    assert KP.rank_band("末未", result) == "uncommon"
    assert KP.rank_band("士土", result) == "rare"
    assert KP.rank_band("日曰", result) == "very_rare"
    distribution = KP.pair_distribution(result)
    assert [distribution[b["name"]]["probability"] for b in KP.BANDS] == [.85, .10, .04, .01]
    subset = ["人入", "日曰"]
    distribution = KP.pair_distribution(result, subset)
    assert distribution["common"]["probability"] == .99
    assert distribution["very_rare"]["probability"] == .01
    assert distribution["rare"]["probability"] == 0
    assert KP.pair_distribution(result, ["土士"])["rare"]["probability"] == 1
    with pytest.raises(ValueError, match="subset"):
        KP.pair_distribution(result, ["人龍"])
    with pytest.raises(ValueError, match="subset"):
        KP.pair_distribution(result, [])


def test_pair_sampling_respects_eligible_subset_and_returns_both_endpoints(policy):
    result = policy[0]
    rng = np.random.default_rng(1)
    orders = {KP.sample_pair(rng, result, pairs=["土士"]) for _ in range(40)}
    assert orders == {("士", "土"), ("土", "士")}
    assert all(set(KP.sample_pair(rng, result, pairs=["人入", "日曰"])) in (set("人入"), set("日曰"))
               for _ in range(40))


def test_pair_weights_use_inverse_maximum_rank_within_scheduled_band(policy):
    result = policy[0]

    class RecordProbabilities:
        def __init__(self):
            self.probabilities = []

        def choice(self, count, p):
            self.probabilities.append(p)
            return 0

        def random(self):
            return 0

    rng = RecordProbabilities()
    KP.sample_pair(rng, result, pairs=["人入", "人日"])
    assert rng.probabilities[0] == [1, 0, 0, 0]
    assert rng.probabilities[1][0] / rng.probabilities[1][1] == pytest.approx(25)

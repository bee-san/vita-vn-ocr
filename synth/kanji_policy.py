#!/usr/bin/env python3
"""Pinned Jiten allowlist and bounded similar-kanji sampling for an unchanged model.

The official CSV supplies ranks, not observed occurrence counts. Inverse-rank
weights are an explicit heuristic. A corrective source gets 2% of total training
draws; its rare-band caps must also be enforced by the finite renderer schedule.
sample_pair returns both endpoints, so render one corpus-backed crop for each.
"""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

JITEN_URL = "https://api.jiten.moe/api/frequency-list/download-kanji?downloadType=csv"
JITEN_SHA256 = "bdb0eaef29e5b7bdf52efaa91376dba8935b860a57a56d7ae2203138de918c78"
SIMILAR_REVISION = "0edcfebd5204676dec72b6b12fa26c148489bb1c"
SIMILAR_URL = f"https://raw.githubusercontent.com/siikamiika/similar-kanji/{SIMILAR_REVISION}/kanji.tgz_similars.ut8"
SIMILAR_SHA256 = "2f80888f8c0afbf8793327cdf7dad0778f028467594183e53811b2ed461ef026"
SIMILAR_LICENSE_SHA256 = "06c3b7d4828de91d21c16b577d7bfad3601a15249fa5577f38b48858dfa1754f"
SHA = re.compile(r"[0-9a-f]{64}\Z")
BANDS = [{"name": "common", "max_rank": 2000, "probability": .85},
         {"name": "uncommon", "max_rank": 3000, "probability": .10},
         {"name": "rare", "max_rank": 4000, "probability": .04},
         {"name": "very_rare", "max_rank": None, "probability": .01}]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_han(character):
    return isinstance(character, str) and len(character) == 1 and unicodedata.name(character, "").startswith(
        ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))


def read_jiten(path):
    """Strict two-column CSV; tied ranks are allowed, duplicate characters are not."""
    ranks = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Kanji", "Rank"]:
            raise ValueError("Jiten CSV header must be exactly Kanji,Rank")
        for row in reader:
            character, rank = row.get("Kanji"), row.get("Rank")
            if set(row) != {"Kanji", "Rank"} or not is_han(character):
                raise ValueError("each Jiten row must contain one Han character and one rank")
            if not isinstance(rank, str) or not re.fullmatch(r"[1-9][0-9]*", rank):
                raise ValueError("Jiten ranks must be positive integers")
            if character in ranks:
                raise ValueError(f"duplicate Jiten character: {character}")
            ranks[character] = int(rank)
    if not ranks:
        raise ValueError("empty Jiten kanji list")
    return ranks


def read_pairs(path):
    """Each slash-separated record is a source and its neighbours, not a clique."""
    pairs = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        characters = line.rstrip("/").split("/")
        if not characters or any(not is_han(character) for character in characters):
            raise ValueError("similar-kanji records must contain single Han characters")
        source = characters[0]
        for other in characters[1:]:
            if source != other:
                pairs.add("".join(sorted((source, other))))
    return sorted(pairs)


def content_sha256(policy):
    body = {key: value for key, value in policy.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def build_policy(jiten, similar, vocab):
    jiten, similar, vocab = map(Path, (jiten, similar, vocab))
    if sha256(jiten) != JITEN_SHA256 or sha256(similar) != SIMILAR_SHA256:
        raise ValueError("source hash differs from the pinned Jiten or similar-kanji data")
    all_ranks = read_jiten(jiten)
    characters = vocab.read_text(encoding="utf-8").splitlines()
    if not characters or len(set(characters)) != len(characters) or any(len(c) != 1 for c in characters):
        raise ValueError("vocabulary must contain unique single-character classes in original order")
    ranks = {c: all_ranks[c] for c in characters if c in all_ranks}
    source_pairs = read_pairs(similar)
    pairs = [pair for pair in source_pairs if all(c in ranks for c in pair)]
    if not ranks or not pairs:
        raise ValueError("no supported Jiten kanji or similar-kanji pairs")
    policy = {"schema": 1, "vocab_sha256": sha256(vocab),
              "provenance": {"jiten": {"url": JITEN_URL, "sha256": JITEN_SHA256,
                                       "bytes": jiten.stat().st_size, "license": "CC-BY-SA-4.0",
                                       "revision": "Jiten (Kanji) 26-08-09", "columns": ["Kanji", "Rank"]},
                             "similar": {"url": SIMILAR_URL, "sha256": SIMILAR_SHA256,
                                         "bytes": similar.stat().st_size, "revision": SIMILAR_REVISION,
                                         "license": "MIT", "copyright": "Copyright (c) 2017 siikamiika",
                                         "license_url": f"https://raw.githubusercontent.com/siikamiika/similar-kanji/{SIMILAR_REVISION}/LICENSE",
                                         "license_sha256": SIMILAR_LICENSE_SHA256}},
              "allowed_kanji": sorted(ranks, key=lambda c: (ranks[c], c)),
              "allowed_non_kanji": [c for c in characters if not is_han(c)],
              "ranks": ranks, "pairs": pairs,
              "coverage": {"jiten_kanji": len(all_ranks), "model_vocabulary": len(characters),
                           "model_kanji": sum(is_han(c) for c in characters), "supported_kanji": len(ranks),
                           "missing_jiten_kanji": len(all_ranks) - len(ranks),
                           "model_kanji_outside_jiten": sum(is_han(c) and c not in ranks for c in characters),
                           "source_pairs": len(source_pairs), "supported_pairs": len(pairs)},
              "corrective": {"total_training_probability": .02, "rank_bands": BANDS,
                             "pair_weight": "1 / max(endpoint ranks)",
                             "endpoint_balance": "one rendered context crop per endpoint",
                             "empty_bands": "redistribute to the most common available band",
                             "finite_schedule": "floor rare quotas; assign remaining pairs to common"},
              "names": {"weight": "1 / character rank", "interpretation": "rank heuristic, not observed counts"}}
    policy["effective_bands"] = pair_distribution(policy)
    policy["content_sha256"] = content_sha256(policy)
    return policy


def load_policy(path):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or type(policy.get("schema")) is not int or policy["schema"] != 1:
        raise ValueError("kanji policy requires schema 1")
    required = {"vocab_sha256", "provenance", "allowed_kanji", "allowed_non_kanji", "ranks", "pairs",
                "coverage", "corrective", "names", "effective_bands", "content_sha256"}
    if not required <= policy.keys() or not SHA.fullmatch(str(policy["vocab_sha256"])):
        raise ValueError("kanji policy is missing required fields or vocabulary hash")
    if policy["content_sha256"] != content_sha256(policy):
        raise ValueError("kanji policy content hash mismatch")
    provenance = policy["provenance"]
    if not isinstance(provenance, dict) or any(not isinstance(provenance.get(key), dict)
                                             for key in ("jiten", "similar")):
        raise ValueError("kanji policy requires source provenance objects")
    if (provenance.get("jiten", {}).get("sha256"), provenance.get("similar", {}).get("sha256")) != (
            JITEN_SHA256, SIMILAR_SHA256):
        raise ValueError("kanji policy source hashes differ from pinned data")
    ranks = policy["ranks"]
    if not isinstance(ranks, dict) or not ranks or any(
            not is_han(c) or type(rank) is not int or rank < 1 for c, rank in ranks.items()):
        raise ValueError("kanji policy ranks must be single Han characters with positive integers")
    if policy["allowed_kanji"] != sorted(ranks, key=lambda c: (ranks[c], c)):
        raise ValueError("kanji allowlist and ranks disagree")
    other = policy["allowed_non_kanji"]
    if not isinstance(other, list) or len(other) != len(set(other)) or any(
            not isinstance(c, str) or len(c) != 1 or is_han(c) for c in other):
        raise ValueError("non-kanji classes must be unique single non-Han characters")
    pairs = policy["pairs"]
    if not isinstance(pairs, list) or not pairs or pairs != sorted(set(pairs)) or any(
            not isinstance(pair, str) or len(pair) != 2 or pair[0] >= pair[1]
            or any(c not in ranks for c in pair) for pair in pairs):
        raise ValueError("kanji pairs must be unique supported canonical two-character strings")
    corrective = policy["corrective"]
    if not isinstance(corrective, dict) or not isinstance(policy["coverage"], dict):
        raise ValueError("kanji corrective and coverage fields must be objects")
    if corrective.get("total_training_probability") != .02 or corrective.get("rank_bands") != BANDS:
        raise ValueError("kanji corrective source must retain the 2% budget and fixed rank-band caps")
    coverage = policy["coverage"]
    if coverage.get("supported_kanji") != len(ranks) or coverage.get("supported_pairs") != len(pairs):
        raise ValueError("kanji coverage counts disagree with the table")
    if policy["effective_bands"] != pair_distribution(policy):
        raise ValueError("kanji effective band distribution disagrees with the pairs")
    return policy


def allowed_label(text, policy):
    """Labels must fit the unchanged vocabulary and the Jiten Han allowlist."""
    other = set(policy["allowed_non_kanji"])
    return isinstance(text, str) and bool(text) and all(c in policy["ranks"] or c in other for c in text)


def name_weights(characters, policy):
    """Normalized inverse-rank probabilities for a list of candidate kanji classes."""
    weights = [1 / policy["ranks"][c] if c in policy["ranks"] else 0 for c in characters]
    total = sum(weights)
    if not total:
        raise ValueError("name candidates contain no supported Jiten kanji")
    return [weight / total for weight in weights]


def rank_band(pair, policy):
    rank = max(policy["ranks"][c] for c in pair)
    return next(band["name"] for band in BANDS if band["max_rank"] is None or rank <= band["max_rank"])


def pair_distribution(policy, pairs=None):
    """Conditional subset distribution; unavailable mass moves toward common ranks."""
    pairs = policy["pairs"] if pairs is None else list(pairs)
    allowed = set(policy["pairs"])
    if not pairs or len(set(pairs)) != len(pairs) or any(pair not in allowed for pair in pairs):
        raise ValueError("eligible pairs must be a nonempty unique subset of the policy graph")
    groups = {band["name"]: [] for band in BANDS}
    for pair in pairs:
        groups[rank_band(pair, policy)].append(pair)
    first = next(band["name"] for band in BANDS if groups[band["name"]])
    missing = sum(band["probability"] for band in BANDS if not groups[band["name"]])
    return {band["name"]: {"pairs": len(groups[band["name"]]),
                           "probability": round((band["probability"] if groups[band["name"]] else 0)
                                                + (missing if band["name"] == first else 0), 12)}
            for band in BANDS}


def sample_pair(rng, policy, pairs=None):
    """Return a pair in random endpoint order, with rank weighting inside each band.

    A renderer may pass only corpus-backed pairs, or a pre-scheduled band subset.
    For a single-band subset, selection stays in that band; the caller enforces
    overall band caps through its finite schedule and renders both endpoints.
    """
    pairs = policy["pairs"] if pairs is None else list(pairs)
    distribution = pair_distribution(policy, pairs)
    band = BANDS[int(rng.choice(len(BANDS), p=[distribution[b["name"]]["probability"] for b in BANDS]))]
    candidates = [pair for pair in pairs if rank_band(pair, policy) == band["name"]]
    weights = [1 / max(policy["ranks"][c] for c in pair) for pair in candidates]
    total = sum(weights)
    pair = candidates[int(rng.choice(len(candidates), p=[weight / total for weight in weights]))]
    return tuple(pair if rng.random() < .5 else pair[::-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jiten", type=Path, required=True)
    parser.add_argument("--similar", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policy = build_policy(args.jiten, args.similar, args.vocab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(policy, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "sha256": sha256(args.out),
                      "coverage": policy["coverage"], "effective_bands": policy["effective_bands"]}))


if __name__ == "__main__":
    main()

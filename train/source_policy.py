"""Opt-in, exact-prefix training exposure limits, independent of prefetch draws.

The registry is an assertion made by reviewed producers, not an ownership detector.
Unknown ownership never qualifies as a game-free fallback.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAPS = {"game_font_text": (1, 20), "all_game_assets": (1, 5),
        "correction_pattern": (1, 200), "kanji_corrective": (1, 50)}
KINDS = {"font", "text", "background", "real_crop"}
ORIGINS = {"game", "ofl", "non_game", "procedural", "unknown"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("source identity requires a lowercase SHA256")
    return value


def identifier(value):
    if (not isinstance(value, str) or not 0 < len(value) <= 256
            or any(ord(c) < 32 for c in value)):
        raise ValueError("source identity must be a bounded nonempty string")
    return value


def ids(value):
    if not isinstance(value, list):
        raise ValueError("source identities must be a list")
    result = tuple(identifier(v) for v in value)
    if len(set(result)) != len(result):
        raise ValueError("source identities must not repeat")
    return result


def parse_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON identity/role field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


@dataclass(frozen=True)
class Exposure:
    sample_id: str
    asset_ids: tuple = ()
    font_text_vns: tuple = ()
    game_assets: bool = False
    correction_pattern_ids: tuple = ()
    kanji_corrective: bool = False
    unknown: bool = True
    fallback_verified: bool = False


class SourcePolicy:
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        doc = parse_json(raw)
        if doc.get("schema") != 1 or doc.get("kind") != "general-vn-source-policy":
            raise ValueError("expected schema-1 general VN source policy")
        self.fallback_source = identifier(doc.get("fallback_source"))
        self.forbidden_vns = set(ids(doc.get("forbidden_vns")))
        if any(re.fullmatch(r"v[1-9][0-9]*", v) is None for v in self.forbidden_vns):
            raise ValueError("forbidden VNs require canonical VNDB identities")
        assets = doc.get("assets")
        if not isinstance(assets, dict):
            raise ValueError("source policy requires an explicit asset registry")
        self.assets, identities = {}, {}
        for key, row in assets.items():
            identifier(key)
            if not isinstance(row, dict):
                raise ValueError("asset registry rows must be objects")
            digest(row.get("sha256"))
            kind, origin = row.get("kind"), row.get("origin", "unknown")
            if kind is not None and kind not in KINDS or origin not in ORIGINS:
                raise ValueError("unknown asset kind or origin")
            vn, release = row.get("vn"), row.get("release")
            if (vn is not None and (not isinstance(vn, str) or re.fullmatch(r"v[1-9][0-9]*", vn) is None)
                    or release is not None and (not isinstance(release, str)
                                               or re.fullmatch(r"r[1-9][0-9]*", release) is None)):
                raise ValueError("game ownership requires canonical VNDB VN/release IDs")
            if vn in self.forbidden_vns:
                raise ValueError("development/protected game appears in training registry")
            if origin != "game" and origin != "unknown" and (vn is not None or release is not None):
                raise ValueError("game-free asset cannot carry game ownership")
            if (kind is None or origin == "game" and (vn is None or release is None)
                    or origin == "ofl" and (kind != "font" or row.get("license") != "OFL-1.1")
                    or origin == "non_game" and (kind != "text" or row.get("game_free_reviewed") is not True)
                    or origin == "procedural" and kind not in ("text", "background")
                    or origin == "procedural" and kind == "text" and row.get("game_free_reviewed") is not True
                    or kind == "real_crop" and origin != "game"):
                origin = "unknown"
            identity = (kind, origin, vn, release)
            if row["sha256"] in identities and identities[row["sha256"]] != identity:
                raise ValueError("same asset bytes have conflicting ownership/kind identities")
            identities[row["sha256"]] = identity
            self.assets[key] = {**row, "origin": origin}
        self.shards = {}
        rows = doc.get("shards")
        if not isinstance(rows, list) or not rows:
            raise ValueError("source policy requires a nonempty exact shard inventory")
        for row in rows:
            if not isinstance(row, dict) or row.get("split") != "train":
                raise ValueError("only declared training shards may enter the source policy")
            relative = Path(identifier(row.get("file")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("shard paths must stay within the policy directory")
            path = (self.path.parent / relative).resolve()
            if not path.is_relative_to(self.path.parent) or path in self.shards:
                raise ValueError("duplicate or escaping source shard")
            digest(row.get("sha256"))
            if any(type(row.get(k)) is not int or row[k] <= 0 for k in ("bytes", "samples")):
                raise ValueError("source shard bytes and samples must be positive integers")
            identifier(row.get("source"))
            self.shards[path] = row
        self.provenance = {}

    def verify_inputs(self):
        if sha256(self.path) != self.sha256:
            raise ValueError("source policy changed during input loading")
        for path, row in self.shards.items():
            if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError("training shard bytes/hash differ from the policy")

    def exposure(self, meta, fallback_id):
        if not isinstance(meta, dict):
            raise ValueError("training row metadata must be an object")
        if "split" in meta and meta["split"] != "train":
            raise ValueError("non-training row rejected before crop pixels")
        prov = meta.get("source_provenance")
        if prov is None:
            return Exposure(fallback_id)
        if not isinstance(prov, dict) or prov.get("schema") != 1:
            raise ValueError("training provenance requires schema 1")
        sample_id = identifier(prov.get("sample_id"))
        asset_ids = ids(prov.get("asset_ids"))
        patterns = ids(prov.get("correction_pattern_ids"))
        if any(type(prov.get(k)) is not bool for k in ("kanji_corrective", "broad_sampling")):
            raise ValueError("provenance flags must be explicit booleans")
        refs = [self.assets.get(key) for key in asset_ids]
        unknown = (meta.get("split") != "train" or not refs
                   or any(row is None or row["origin"] == "unknown" for row in refs))
        known = [row for row in refs if row is not None]
        games = [row for row in known if row["origin"] == "game"]
        vns = tuple(sorted({row["vn"] for row in games if row["kind"] in ("font", "text", "real_crop")}))
        kinds = {row.get("kind") for row in known}
        complete = ({"font", "text", "background"} <= kinds
                    or any(row["kind"] == "real_crop" and row["origin"] == "game" for row in known))
        unknown |= not complete
        fallback = (not unknown and prov["broad_sampling"] and not patterns
                    and not prov["kanji_corrective"] and {"font", "text", "background"} <= kinds
                    and all((row["kind"] == "font" and row["origin"] == "ofl"
                             or row["kind"] == "text" and row["origin"] in ("non_game", "procedural")
                             or row["kind"] == "background" and row["origin"] == "procedural") for row in known))
        return Exposure(sample_id, asset_ids, vns, bool(games), patterns,
                        prov["kanji_corrective"], unknown, fallback)

    def bind(self, sources):
        """Validate all inventories and roles before train2 accesses any crop arrays."""
        actual = {}
        for name, paths in sources.items():
            for path in paths:
                path = Path(path).resolve()
                if path in actual:
                    raise ValueError("same shard used in multiple training sources")
                actual[path] = name
        if set(actual) != set(self.shards) or any(actual[p] != self.shards[p]["source"] for p in actual):
            raise ValueError("training sources differ from the exact policy inventory")
        if self.fallback_source not in sources:
            raise ValueError("verified fallback source is absent")
        self.verify_inputs()
        seen = set()
        for path, row in self.shards.items():
            with np.load(path, allow_pickle=False) as z:
                metas = parse_json(str(z["meta"])) if "meta" in z.files else [{}] * row["samples"]
                if not isinstance(metas, list) or len(metas) != row["samples"]:
                    raise ValueError("training metadata row count differs from the policy")
                exposures = [self.exposure(meta, f"unknown:{row['sha256']}:{i}") for i, meta in enumerate(metas)]
            for exposure in exposures:
                if exposure.sample_id in seen:
                    raise ValueError("immutable training sample identity is repeated")
                seen.add(exposure.sample_id)
                if row["source"] == self.fallback_source and not exposure.fallback_verified:
                    raise ValueError("fallback must be broad OFL/non-game/procedural and fully verified")
            self.provenance[path] = exposures
        return self.provenance


class ExposureLedger:
    def __init__(self):
        self.forwarded = self.optimizer = self.game_assets = self.kanji = 0
        self.by_source, self.by_game, self.by_pattern, self.optimizer_by_source = (Counter() for _ in range(4))

    def reason(self, exposure):
        if exposure is None or exposure.unknown:
            return "unknown_provenance"
        n = self.forwarded + 1
        if any(self.by_game[vn] + 1 > n // 20 for vn in exposure.font_text_vns):
            return "game_font_text_cap"
        if exposure.game_assets and self.game_assets + 1 > n // 5:
            return "all_game_assets_cap"
        if any(self.by_pattern[p] + 1 > n // 200 for p in exposure.correction_pattern_ids):
            return "correction_pattern_cap"
        if exposure.kanji_corrective and self.kanji + 1 > n // 50:
            return "kanji_corrective_cap"
        return None

    def consume(self, exposure, source):
        reason = self.reason(exposure)
        if reason:
            raise ValueError(f"forwarded exposure violates source policy: {reason}")
        self.forwarded += 1
        self.by_source[source] += 1
        self.by_game.update(exposure.font_text_vns)
        self.by_pattern.update(exposure.correction_pattern_ids)
        self.game_assets += exposure.game_assets
        self.kanji += exposure.kanji_corrective

    def optimized(self, sources):
        extra = Counter(sources)
        if self.optimizer + len(sources) > self.forwarded or any(
                self.optimizer_by_source[key] + value > self.by_source[key] for key, value in extra.items()):
            raise ValueError("optimizer accounting exceeds forwarded exposures")
        self.optimizer += len(sources)
        self.optimizer_by_source.update(extra)

    def state_dict(self):
        return {"forwarded": self.forwarded, "optimizer": self.optimizer,
                "game_assets": self.game_assets, "kanji": self.kanji,
                **{k: dict(getattr(self, k)) for k in ("by_source", "by_game", "by_pattern", "optimizer_by_source")}}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        if not isinstance(state, dict) or set(state) != set(obj.state_dict()):
            raise ValueError("invalid source exposure ledger")
        for key in ("forwarded", "optimizer", "game_assets", "kanji"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError("invalid source exposure counter")
            setattr(obj, key, state[key])
        for key in ("by_source", "by_game", "by_pattern", "optimizer_by_source"):
            values = state[key]
            if not isinstance(values, dict) or any(type(v) is not int or v < 0 for v in values.values()):
                raise ValueError("invalid grouped source exposure counter")
            for name in values:
                identifier(name)
            setattr(obj, key, Counter(values))
        if (sum(obj.by_source.values()) != obj.forwarded or sum(obj.optimizer_by_source.values()) != obj.optimizer
                or obj.optimizer > obj.forwarded or obj.game_assets > obj.forwarded // 5
                or obj.kanji > obj.forwarded // 50
                or any(v > obj.forwarded // 20 for v in obj.by_game.values())
                or any(v > obj.forwarded // 200 for v in obj.by_pattern.values())
                or any(v > obj.by_source[k] for k, v in obj.optimizer_by_source.items())):
            raise ValueError("resumed source exposure ledger violates exact caps/accounting")
        return obj

    def reserve(self):
        return copy.deepcopy(self)


class RunWriter:
    """One cooperating train2 process may mutate a run's artifacts at a time."""
    def __init__(self, directory):
        self.file = (Path(directory) / ".train2-writer.lock").open("a")
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.file.close()
            raise ValueError("another train2 writer owns this run") from error

    def close(self):
        self.file.close()


def json_digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def freeze_json(path, value):
    """Preserve readable execution provenance even if the first forward later fails."""
    path = Path(path)
    if path.exists():
        if parse_json(path.read_bytes()) != value:
            raise ValueError("frozen source execution identity differs")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class ExposureJournal:
    """Durable aggregate boundaries; arbitrary crash recovery is deliberately refused."""
    def __init__(self, path, execution, policy_sha256, resume=None, checkpoint_tail=None):
        self.path, self.execution = Path(path), json_digest(execution)
        self.policy_sha256 = policy_sha256
        self.tail, self.ledger = "0" * 64, ExposureLedger().state_dict()
        self.pending_intent = False
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = parse_json(line)
                    if not isinstance(record, dict):
                        raise ValueError("invalid aggregate exposure journal record")
                    observed = record.pop("sha256", None)
                    keys = {"schema", "event", "previous_sha256", "execution_sha256", "policy_sha256", "ledger"}
                    if (set(record) != keys
                            or record.get("schema") != 1 or record.get("previous_sha256") != self.tail
                            or observed != json_digest(record)
                            or record.get("execution_sha256") != self.execution
                            or record.get("policy_sha256") != self.policy_sha256):
                        raise ValueError("exposure journal hash/provenance mismatch")
                    self._transition(record.get("event"), record.get("ledger"))
                    self.tail, self.ledger = observed, record["ledger"]
                    self.pending_intent = record["event"] == "intent"
        if self.pending_intent:
            raise ValueError("unresolved forward intent; interrupted model exposures cannot resume")
        if resume is None:
            if self.tail != "0" * 64 or checkpoint_tail is not None:
                raise ValueError("existing exposure journal requires a matching model checkpoint")
        elif self.tail != checkpoint_tail or self.ledger != resume:
            raise ValueError("exposure journal differs from checkpoint; uncheckpointed exposures cannot be reset")
        self.stream = self.path.open("a", encoding="utf-8")
        self.size = self.path.stat().st_size

    def _transition(self, event, ledger):
        ExposureLedger.from_state(ledger)
        if event not in ("intent", "forward", "optimizer"):
            raise ValueError("invalid exposure journal boundary")
        before = self.ledger
        if event == "intent":
            if self.pending_intent or ledger != before:
                raise ValueError("duplicate forward intent or intent changed successful exposure counts")
            return
        if event == "forward":
            if not self.pending_intent:
                raise ValueError("successful forward requires a durable forward intent")
            if (ledger["forwarded"] <= before["forwarded"] or ledger["optimizer"] != before["optimizer"]
                    or ledger["optimizer_by_source"] != before["optimizer_by_source"]):
                raise ValueError("invalid forwarded exposure journal transition")
        elif (self.pending_intent or ledger["optimizer"] <= before["optimizer"] or any(
                ledger[key] != before[key] for key in ledger if key not in ("optimizer", "optimizer_by_source"))):
            raise ValueError("invalid optimizer exposure journal transition")
        for key, value in before.items():
            if isinstance(value, dict):
                if any(ledger[key].get(group, 0) < count for group, count in value.items()):
                    raise ValueError("exposure journal counters moved backwards")
            elif ledger[key] < value:
                raise ValueError("exposure journal counters moved backwards")

    def append(self, event, ledger):
        state = ledger.state_dict()
        self._transition(event, state)
        if self.path.stat().st_size != self.size:
            raise ValueError("exposure journal changed outside its writer")
        record = {"schema": 1, "event": event, "previous_sha256": self.tail,
                  "execution_sha256": self.execution, "policy_sha256": self.policy_sha256, "ledger": state}
        tail = json_digest(record)
        self.stream.write(json.dumps({**record, "sha256": tail}, sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.tail, self.ledger, self.size = tail, state, self.path.stat().st_size
        self.pending_intent = event == "intent"

    def close(self):
        self.stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="aggregate metadata validation only; refuses overwrite")
    args = parser.parse_args()
    policy = SourcePolicy(args.policy)
    sources = {}
    for path, row in policy.shards.items():
        sources.setdefault(row["source"], []).append(path)
    policy.bind(sources)
    counts = {}
    for path, exposures in policy.provenance.items():
        source = policy.shards[path]["source"]
        row = counts.setdefault(source, {"rows": 0, "unknown": 0, "verified_broad": 0})
        row["rows"] += len(exposures)
        row["unknown"] += sum(exposure.unknown for exposure in exposures)
        row["verified_broad"] += sum(exposure.fallback_verified for exposure in exposures)
    result = {"schema": 1, "policy_sha256": policy.sha256, "caps": CAPS, "sources": counts,
              "validation": "metadata_and_source_bytes_only", "model_forwards": 0,
              "training_readiness": "not assessed; development and source isolation gates remain separate"}
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()

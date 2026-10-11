"""Generic VN/platform aggregates; no images, model execution or dataset loading."""

import hashlib
import re
from fractions import Fraction

from metrics import is_hiragana, lev, norm, score_hiragana

MINIMUMS = {"vns_per_console": 3, "frames_per_vn": 5, "body_chars_per_vn": 500, "body_hiragana_per_vn": 200}


def normalize_row(row):
    result = dict(row)
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("line requires a stable id")
    if not isinstance(row.get("vn"), str) or not re.fullmatch(r"v[1-9][0-9]*", row["vn"]):
        raise ValueError("line requires a VNDB game id")
    if "platform" in row and row["platform"] not in ("psp", "psv", "vita"):
        raise ValueError("invalid explicit platform provenance")
    if "dims" in row and (
        not isinstance(row["dims"], (list, tuple))
        or len(row["dims"]) != 2
        or any(type(value) is not int for value in row["dims"])
    ):
        raise ValueError("native dimensions must be two integers")
    inferred = {(480, 272): "psp", (960, 544): "vita"}.get(tuple(row.get("dims", ())))
    if "dims" in row and inferred is None:
        raise ValueError("unsupported native dimensions")
    platform = {"psp": "psp", "psv": "vita", "vita": "vita"}.get(row.get("platform"), inferred)
    if platform not in ("psp", "vita") or (inferred is not None and platform != inferred):
        raise ValueError("missing or contradictory native platform provenance")
    styles = row.get("styles", [])
    if not isinstance(styles, list) or any(not isinstance(style, str) for style in styles):
        raise ValueError("styles must be a list of strings")
    if not isinstance(row.get("text"), str):
        raise ValueError("reference text must be a string")
    result.update(platform=platform, styles=sorted(set(styles)))
    return result


def body(row):
    return "name" not in row["styles"] and row.get("content_kind") != "name"


def provenance(row):
    return {
        key: row.get(key)
        for key in (
            "id",
            "platform",
            "vn",
            "session",
            "release",
            "styles",
            "sha256",
            "dims",
            "content_kind",
            "source_manifest_sha256",
            "box",
            "region",
            "url",
        )
    }


def partition(rows):
    groups, seen = {}, set()
    for original in rows:
        row = normalize_row(original)
        if row["id"] in seen:
            raise ValueError("duplicate development line id")
        seen.add(row["id"])
        if row.get("uncertain"):
            continue
        groups.setdefault(f"{row['platform']}:{row['vn']}", []).append(row)
    return groups


def _score(rows):
    full = {
        "chars": sum(len(norm(row["text"])) for row in rows),
        "errors": sum(lev(norm(row["text"]), norm(row["pred"])) for row in rows),
    }
    hira = score_hiragana(rows)
    return {"lines": len(rows), "full": full, "hira": {key: hira[key] for key in ("chars", "errors")}}


def group_score(rows):
    result = {}
    for key, group in sorted(partition(rows).items()):
        if any(not isinstance(row.get("pred"), str) for row in group):
            raise ValueError("each scored line requires an explicit prediction")
        result[key] = {
            "platform": group[0]["platform"],
            "vn": group[0]["vn"],
            "all": _score(group),
            "body": _score([row for row in group if body(row)]),
            "provenance": sorted((provenance(row) for row in group), key=lambda row: row["id"]),
            "reference_sha256": hashlib.sha256(
                "\n".join(row["id"] + "\0" + row["text"] for row in sorted(group, key=lambda row: row["id"])).encode()
            ).hexdigest(),
        }
    return result


def readiness(rows, minimums=None):
    minimums = dict(MINIMUMS if minimums is None else minimums)
    if set(minimums) != set(MINIMUMS) or any(
        type(value) is not int or value < MINIMUMS[key] for key, value in minimums.items()
    ):
        raise ValueError("readiness quotas cannot be lowered below the prospective protocol")
    groups, reasons, totals = {}, [], {"psp": set(), "vita": set()}
    for key, group in sorted(partition(rows).items()):
        bodies = [row for row in group if body(row)]
        chars = sum(len(norm(row["text"])) for row in bodies)
        hira = sum(sum(is_hiragana(char) for char in norm(row["text"])) for row in bodies)
        frames = {
            row["sha256"]
            for row in group
            if isinstance(row.get("sha256"), str) and re.fullmatch("[0-9a-f]{64}", row["sha256"])
        }
        item = {
            "platform": group[0]["platform"],
            "vn": group[0]["vn"],
            "scored_lines": len(group),
            "frames": len(frames),
            "body_lines": len(bodies),
            "body_chars": chars,
            "body_hiragana": hira,
        }
        groups[key] = item
        totals[item["platform"]].add(item["vn"])
        for field, quota in [
            ("frames", "frames_per_vn"),
            ("body_chars", "body_chars_per_vn"),
            ("body_hiragana", "body_hiragana_per_vn"),
        ]:
            if item[field] < minimums[quota]:
                reasons.append(f"{key}: {field} {item[field]} < {minimums[quota]}")
    for platform, games in totals.items():
        if len(games) < minimums["vns_per_console"]:
            reasons.append(f"{platform}: VNs {len(games)} < {minimums['vns_per_console']}")
    return {
        "ready": not reasons,
        "minimums": minimums,
        "groups": groups,
        "reasons": reasons,
        "vns_per_console": {platform: len(games) for platform, games in totals.items()},
        "distinct_vns": len(set().union(*totals.values())),
    }


def balanced_rate(groups, script="hira"):
    if script not in ("hira", "full"):
        raise ValueError("unknown ranking script")
    platform_rates = []
    for platform in ("psp", "vita"):
        rates = []
        for group in groups.values():
            if group["platform"] == platform:
                score = group["all"][script]
                if score["chars"] <= 0:
                    raise ValueError("macro ranking requires nonempty reference strata")
                rates.append(Fraction(score["errors"], score["chars"]))
        if not rates:
            raise ValueError("macro ranking requires both consoles")
        platform_rates.append(sum(rates, Fraction()) / len(rates))
    return sum(platform_rates, Fraction()) / 2

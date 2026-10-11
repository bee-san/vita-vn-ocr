#!/usr/bin/env python3
"""Aggregate kanji errors from an existing native-C recognizer report; never select a model."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "eval"))
from metrics import align_text  # noqa: E402

from kanji_policy import BANDS, is_han, load_policy, sha256  # noqa: E402

BAND_NAMES = [b["name"] for b in BANDS] + ["unsupported"]


def band(character, policy):
    rank = policy["ranks"].get(character)
    return "unsupported" if rank is None else next(
        b["name"] for b in BANDS if b["max_rank"] is None or rank <= b["max_rank"])


def summary(counter, total=False):
    keys = ["chars", "substitutions", "deletions"] + (["insertions", "into_kanji"] if total else [])
    result = {k: counter[k] for k in keys}
    errors = counter["substitutions"] + counter["deletions"] + (counter["insertions"] if total else 0)
    result.update(errors=errors, evidence="observed" if counter["chars"] else "no_evidence")
    result["cer" if total else "reference_error_rate"] = (
        round(errors / counter["chars"], 5) if counter["chars"] else None)
    return result


def confusion_rows(counter):
    return [{"reference": g, "prediction": p, "count": n} for (g, p), n in sorted(
        counter.items(), key=lambda kv: (-kv[1], kv[0][0] or "", kv[0][1] or ""))]


def score(rows, policy):
    total, bands, confusions, similar = Counter(), {b: Counter() for b in BAND_NAMES}, Counter(), Counter()
    insertions, into = {b: 0 for b in BAND_NAMES}, {b: 0 for b in BAND_NAMES}
    pairs, graph_characters = set(policy["pairs"]), set("".join(policy["pairs"]))
    graph_references = 0
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ("text", "pred")):
            raise ValueError("recognizer rows require string text and prediction")
        for g, p in align_text(row["text"], row["pred"]):
            if is_han(g):
                total["chars"] += 1
                bands[band(g, policy)]["chars"] += 1
                graph_references += g in graph_characters
            if g == p:
                continue
            if is_han(g):
                kind = "deletions" if p is None else "substitutions"
                total[kind] += 1
                bands[band(g, policy)][kind] += 1
            elif g is None and is_han(p):
                total["insertions"] += 1
                insertions[band(p, policy)] += 1
            elif is_han(p):
                total["into_kanji"] += 1
                into[band(p, policy)] += 1
            if is_han(g) or is_han(p):
                confusions[g, p] += 1
            if is_han(g) and is_han(p) and "".join(sorted((g, p))) in pairs:
                similar[g, p] += 1
    sim_rows = [dict(r, pair="".join(sorted((r["reference"], r["prediction"]))))
                for r in confusion_rows(similar)]
    return {"lines": len(rows), "kanji": dict(summary(total, True), confusions=confusion_rows(confusions)),
            "reference_rank_bands": {b: summary(c) for b, c in bands.items()},
            "insertions_by_prediction_band": insertions, "into_kanji_by_prediction_band": into,
            "similar": {"substitutions": sum(similar.values()), "reference_chars": graph_references,
                        "evidence": "observed" if graph_references else "no_evidence", "confusions": sim_rows}}


def diagnose(report, policy):
    if report.get("finder", "missing") is not None or report.get("model_info", {}).get("backend") != "vocr":
        raise ValueError("requires an existing native-C recognizer report without a finder")
    sets = report.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError("recognizer report has no sets")
    identity_keys = ("model", "model_name", "model_info", "finder", "pad", "polarity", "env", "datasets",
                     "provenance", "role", "roles", "selection_role", "evaluation_role", "dev_only", "held_out")
    scored = {}
    for name, result in sets.items():
        if not isinstance(result, dict) or not isinstance(result.get("lines"), list):
            raise ValueError("recognizer sets require their already measured lines")
        scored[name] = score(result["lines"], policy)
    return {"schema": 1, "purpose": "aggregate diagnostics only; no model selection",
            "source": {k: report[k] for k in identity_keys if k in report},
            "policy_vocab_sha256": policy["vocab_sha256"],
            "definitions": {"alignment": "eval/metrics.py full normalized minimum-edit alignment",
                "kanji_cer": "(Han reference substitutions + deletions + Han insertions) / Han reference count.",
                "reference_rank_bands": "Reference substitutions/deletions only; <=2000, <=3000, <=4000, >4000.",
                "prediction_bands": "Insertions/into-Han use predicted rank; excluded from reference rank bands.",
                "unsupported": "Outside the pinned model-vocabulary/Jiten intersection; rank unavailable.",
                "similar": "Substitutions whose canonical pair exists in the pinned policy graph only."},
            "sets": scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(json.loads(args.report.read_text(encoding="utf-8")), load_policy(args.policy))
    result["inputs"] = {"report_sha256": sha256(args.report), "policy_sha256": sha256(args.policy),
                        "diagnostic_code_sha256": sha256(__file__),
                        "metrics_code_sha256": sha256(ROOT / "eval/metrics.py")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sha256": sha256(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""textspot3's pre-registered rule (eval/textspot3_rule.py) and its protocol's pins (eval/textspot3_protocol.json)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import textspot3_rule as R  # noqa: E402

PROTOCOL = ROOT / "eval" / "textspot3_protocol.json"


def units(errs, chars, vns):
    return [{"unit": f"u{i}", "vn": v, "errors": e, "chars": c} for i, (e, c, v) in enumerate(zip(errs, chars, vns))]


def arm(tp, err, dlg=3, n=40, chars=50):
    """A synthetic arm summary: every frame the same (tp, 10 - tp, 10 - tp) and errors, so every paired interval
    is the point difference itself."""
    vns = [f"v{i % 10}" for i in range(n)]
    return {"counts": [{"unit": f"u{i}", "vn": v, "tp": tp, "fp": 10 - tp, "fn": 10 - tp} for i, v in enumerate(vns)],
            "all": units([err] * n, [chars] * n, vns), "dialogue": units([dlg] * n, [chars] * n, vns)}


def cand(name, nbytes, err, tp=6, dlg=2, ms=200.0, arena=200000, c_equal=True, decode="det"):
    return {"name": name, "arm_id": name, "seed": 1, "decode": decode, "blob_bytes": nbytes, "arena_960": arena,
            "est_ms_333_960": ms, "c_equal": c_equal, "arm": arm(tp, err, dlg)}


def cmp(f1=None, cer=None, dlg=None):
    return {"f1": {"ci95": f1}, "cer_all": {"ci95": cer}, "cer_dialogue": {"ci95": dlg}}


# ----------------------------------------------------------------------------- seeds

def test_median_seed_by_dialogue():
    runs = [{"seed": 1, "cer_dialogue": 2.0, "cer_all": 30.0, "f1": 0.5},
            {"seed": 2, "cer_dialogue": 1.0, "cer_all": 40.0, "f1": 0.5},
            {"seed": 3, "cer_dialogue": 3.0, "cer_all": 20.0, "f1": 0.6}]
    rep, info = R.median_seed(runs)
    assert rep["seed"] == 1 and info == {"order": [2, 1, 3], "index": 1}
    rep, _ = R.median_seed(runs[:2])                    # two finished seeds: the worse one
    assert rep["seed"] == 1
    rep, _ = R.median_seed(runs[2:])
    assert rep["seed"] == 3
    tie = [dict(r, cer_dialogue=2.0) for r in runs]     # ties: all-text CER, then -F1, then seed
    assert [r["seed"] for r in sorted(tie, key=R.seed_key)] == [3, 1, 2]
    tie2 = [dict(r, cer_dialogue=2.0, cer_all=30.0, f1=0.5) for r in runs]
    assert [r["seed"] for r in sorted(tie2, key=R.seed_key)] == [1, 2, 3]
    with pytest.raises(ValueError):
        R.median_seed([])


# ----------------------------------------------------------------------------- comparisons

def test_qualifies():
    assert R.qualifies(cmp(f1=[-0.03, 0.02], cer=[-1.0, 6.0], dlg=[-3.0, -0.5]))     # dialogue resolved, rest level
    assert R.qualifies(cmp(f1=[0.01, 0.05], cer=[-5.0, -1.0], dlg=[-3.0, -0.5]))     # better everywhere
    assert not R.qualifies(cmp(f1=[-0.03, 0.02], cer=[-1.0, 6.0], dlg=[-3.0, 0.5]))  # dialogue not resolved
    assert not R.qualifies(cmp(f1=[-0.03, 0.02], cer=[0.5, 6.0], dlg=[-3.0, -0.5]))  # all-text resolved worse
    assert not R.qualifies(cmp(f1=[-0.05, -0.01], cer=[-1.0, 6.0], dlg=[-3.0, -0.5]))  # F1 resolved worse
    assert not R.qualifies(cmp(f1=None, cer=None, dlg=None))                          # no interval: nothing
    assert R.qualifies(cmp(f1=None, cer=None, dlg=[-3.0, -0.5]))                      # unformed guards resolve nothing


def test_replaces():
    assert R.replaces(cmp(f1=[-0.02, 0.03], cer=[-6.0, -1.0], dlg=[-1.0, 1.0]))
    assert not R.replaces(cmp(f1=[-0.02, 0.03], cer=[-6.0, 0.5], dlg=[-1.0, 1.0]))    # all-text not resolved
    assert not R.replaces(cmp(f1=[-0.02, 0.03], cer=[-6.0, -1.0], dlg=[0.1, 1.0]))    # dialogue resolved worse
    assert not R.replaces(cmp(f1=[-0.05, -0.01], cer=[-6.0, -1.0], dlg=[-1.0, 1.0]))  # F1 resolved worse
    assert not R.replaces(cmp(f1=[0.01, 0.05], cer=[-1.0, 1.0], dlg=[-3.0, -1.0]))    # F1 / dialogue gains alone: no


def test_eligible_and_cost_key():
    c = cand("a", 512000, 5, ms=300.0, arena=262144)
    assert R.eligible(c)
    assert not R.eligible(dict(c, blob_bytes=512001))
    assert not R.eligible(dict(c, est_ms_333_960=300.01))
    assert not R.eligible(dict(c, arena_960=262145))
    assert not R.eligible(dict(c, c_equal=None)) and not R.eligible(dict(c, c_equal=False))
    assert not R.eligible(dict(c, arena_960=None)) and not R.eligible(dict(c, est_ms_333_960=None))
    a, b = cand("a", 9000, 5, ms=100.0), cand("b", 3000, 5, ms=250.0, decode="det+refine")
    assert sorted([a, b], key=R.cost_key)[0]["name"] == "b"          # fewer bytes first, whatever the time
    d = cand("d", 3000, 5, ms=250.0)
    assert sorted([b, d], key=R.cost_key)[0]["name"] == "d"          # same bytes and ms: det before det+refine
    e = cand("e", 3000, 5, ms=240.0, decode="det+refine")
    assert sorted([b, d, e], key=R.cost_key)[0]["name"] == "e"       # same bytes: faster first


# ----------------------------------------------------------------------------- the rule

def test_chain():
    ref = {"name": "ref", "arm": arm(6, 8, dlg=3)}
    cands = [cand("a", 20000, 8, dlg=2),                 # qualifies: dialogue better, all-text and F1 level
             cand("b", 60000, 8, dlg=2),                 # level with a: no replacement
             cand("c", 100000, 7, tp=7, dlg=2),          # all-text better than a: replaces it
             cand("d", 200000, 6, tp=7, dlg=2.5),        # all-text better than c, dialogue worse than c: no
             cand("e", 300000, 5, tp=8, dlg=2),          # better than c: replaces it
             cand("f", 600000, 1, tp=9, dlg=1),          # over 512,000 B
             cand("g", 50000, 2, tp=9, dlg=1, c_equal=False),   # C != twin
             cand("h", 70000, 2, tp=9, dlg=1, ms=301.0),        # over 300 ms
             cand("q", 80000, 4, tp=8, dlg=3)]           # dialogue level with ref: never qualifies
    r = R.apply_rule3(cands, ref)
    assert r["eligible"] == ["a", "b", "q", "c", "d", "e"]
    assert r["ineligible"] == ["f", "g", "h"]
    assert r["qualifying"] == ["a", "b", "c", "d", "e"]
    assert r["winner"] == "e" and r["comparator"] == "ref"
    assert [(s["replaces"], s["by"]) for s in r["steps"]] == [("a", "c"), ("c", "e")]
    assert r["bar"] == [] and "baselines" not in r["comparisons"]["a"]


def test_all_text_regression_blocks_and_bar_is_report_only():
    ref = {"name": "ref", "arm": arm(6, 5, dlg=3)}
    baselines = {"grad": arm(5, 10, dlg=10), "det": arm(7, 4, dlg=2)}
    cands = [cand("x", 10000, 6, dlg=2),               # dialogue better but all-text resolved worse than ref
             cand("y", 40000, 5, tp=6, dlg=1)]         # qualifies; does not beat "det" on all-text (bar: no)
    r = R.apply_rule3(cands, ref, baselines)
    assert r["qualifying"] == ["y"] and r["winner"] == "y"
    assert r["comparisons"]["y"]["bar"] is False and r["bar"] == []
    assert set(r["comparisons"]["y"]["baselines"]) == {"det", "grad"}


def test_no_pick():
    ref = {"name": "ref", "arm": arm(6, 5, dlg=2)}
    r = R.apply_rule3([cand("x", 10000, 4, dlg=2), cand("y", 40000, 3, dlg=3)], ref)
    assert r["winner"] is None and r["qualifying"] == [] and r["steps"] == []


def test_recall_by_class():
    m = [{"tp_cls": {"dialogue": 2, "name": 1, "choice": 0, "ui": 1, "scene": 0},
          "fn_cls": {"dialogue": 0, "name": 1, "choice": 0, "ui": 3, "scene": 0}}] * 2
    r = R.recall_by_class(m)
    assert r["dialogue"] == {"tp": 4, "n": 4, "recall": 1.0}
    assert r["ui"] == {"tp": 2, "n": 8, "recall": 0.25} and r["choice"]["recall"] is None


# ----------------------------------------------------------------------------- the protocol

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_protocol_pins_and_rule():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = dict(p["sources_sha256"])
    for amend in p.get("amendments", []):
        pins.update(amend.get("sources_sha256", {}))
    for path, digest in pins.items():
        assert sha(path) == digest, f"{path} changed after the textspot3 protocol froze it (re-pin by an amendment)"
    assert "eval/textspot3_rule.py" in pins and "eval/textspot_metrics.py" in pins
    r = p["rule"]
    assert (r["vtsp_max"], r["arena_max"], r["ms_max_333_960"]) == (512000, 262144, 300)
    assert {k: r[k] for k in R.RULE} == R.RULE
    assert p["budget"]["ms_max_333_960"] == 300 and p["seeds"]["per_arm"] == [1, 2, 3]
    assert p["reused"]["dev_manifest"]["sha256"] == sha("eval/manifests/textspot-dev.jsonl")
    ref = p["references"]["ref"]
    assert ref["name"] == "B_h2_s4_c24n4_seed2_detref"
    assert len(ref["units_sha256"]) == 64 and len(ref["blob_sha256"]) == 64
    assert set(p["families"]) == {"T", "M"} and p["half_above"]["T"] == 480

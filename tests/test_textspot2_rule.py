"""textspot2's pre-registered rule (eval/textspot2_rule.py) and its protocol's pins (eval/textspot2_protocol.json)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import textspot2_rule as R  # noqa: E402

PROTOCOL = ROOT / "eval" / "textspot2_protocol.json"


def units(errs, chars, vns):
    return [{"unit": f"u{i}", "vn": v, "errors": e, "chars": c} for i, (e, c, v) in enumerate(zip(errs, chars, vns))]


def arm(tp, err, dlg=3, n=40, chars=50):
    """A synthetic arm summary: every frame the same (tp, 10 - tp, 10 - tp) and errors, so every paired interval
    is the point difference itself."""
    vns = [f"v{i % 10}" for i in range(n)]
    return {"counts": [{"unit": f"u{i}", "vn": v, "tp": tp, "fp": 10 - tp, "fn": 10 - tp} for i, v in enumerate(vns)],
            "all": units([err] * n, [chars] * n, vns), "dialogue": units([dlg] * n, [chars] * n, vns)}


def cand(name, ms, err, tp=6, dlg=3, nbytes=5000, arena=60000, c_equal=True, decode="det"):
    return {"name": name, "arm_id": name, "seed": 1, "decode": decode, "blob_bytes": nbytes, "arena_960": arena,
            "est_ms_333_960": ms, "c_equal": c_equal, "arm": arm(tp, err, dlg)}


RULE = {"vtsp_max": 262144, "arena_max": 131072, "ms_max_333_960": 150, "fast_ms_max_333_960": 80}


def cmp(f1=None, cer=None, dlg=None):
    return {"f1": {"ci95": f1}, "cer_all": {"ci95": cer}, "cer_dialogue": {"ci95": dlg}}


# ----------------------------------------------------------------------------- seeds

def test_median_seed_three_two_one():
    runs = [{"seed": 1, "cer_all": 30.0, "cer_dialogue": 3.0, "f1": 0.5},
            {"seed": 2, "cer_all": 28.0, "cer_dialogue": 4.0, "f1": 0.5},
            {"seed": 3, "cer_all": 35.0, "cer_dialogue": 2.0, "f1": 0.6}]
    rep, info = R.median_seed(runs)
    assert rep["seed"] == 1 and info == {"order": [2, 1, 3], "index": 1}
    rep, _ = R.median_seed(runs[:2])                    # two finished seeds: the worse one
    assert rep["seed"] == 1
    rep, _ = R.median_seed(runs[1:2])
    assert rep["seed"] == 2
    tie = [dict(r, cer_all=30.0) for r in runs]         # ties: dialogue CER, then -F1, then seed
    assert [r["seed"] for r in sorted(tie, key=R.seed_key)] == [3, 1, 2]
    with pytest.raises(ValueError):
        R.median_seed([])


# ----------------------------------------------------------------------------- comparisons

def test_better():
    assert R.better(cmp(f1=[-0.01, 0.02], cer=[-3.0, -0.5], dlg=[-1.0, 1.0]))      # CER resolved, nothing worse
    assert R.better(cmp(f1=[0.01, 0.05], cer=[-1.0, 2.0], dlg=[-1.0, 1.0]))        # F1 resolved
    assert not R.better(cmp(f1=[0.01, 0.05], cer=[-3.0, -1.0], dlg=[0.2, 1.0]))    # dialogue resolved worse
    assert not R.better(cmp(f1=[0.01, 0.05], cer=[0.5, 2.0], dlg=[-1.0, 1.0]))     # CER resolved worse
    assert not R.better(cmp(f1=[-0.05, -0.01], cer=[-3.0, -1.0], dlg=[-1.0, 1.0]))  # F1 resolved worse
    assert not R.better(cmp(f1=[-0.01, 0.05], cer=[-1.0, 2.0], dlg=[-1.0, 1.0]))   # nothing resolved
    assert not R.better(cmp(f1=None, cer=None, dlg=None))                           # no interval: nothing resolved


def test_eligible_and_cost_key():
    c = cand("a", 150, 5)
    assert R.eligible(c, RULE) and not R.eligible(c, RULE, ms_max=80)
    assert not R.eligible(dict(c, est_ms_333_960=150.01), RULE)
    assert not R.eligible(dict(c, arena_960=131073), RULE)
    assert not R.eligible(dict(c, blob_bytes=262145), RULE)
    assert not R.eligible(dict(c, c_equal=None), RULE)
    assert not R.eligible(dict(c, arena_960=None), RULE)
    a, b = cand("a", 90, 5, nbytes=9000), cand("b", 90, 5, nbytes=3000, decode="det+refine")
    assert sorted([a, b], key=R.cost_key)[0]["name"] == "b"          # same ms: fewer bytes first
    d = cand("d", 90, 5, nbytes=3000)
    assert sorted([b, d], key=R.cost_key)[0]["name"] == "d"          # then det before det+refine


# ----------------------------------------------------------------------------- the rule

def test_main_chain_and_fast_tier1():
    baselines = {"grad": arm(5, 10, dlg=10)}
    ref = {"name": "ref", "arm": arm(6, 8)}
    fast_ref = {"name": "fref", "arm": arm(5, 9)}
    cands = [cand("a", 60, 7),                       # better than ref on CER: the cheapest qualifier
             cand("b", 100, 7),                      # the same as a: no replacement
             cand("c", 120, 5, tp=7, dlg=2),         # better than a: replaces it
             cand("d", 140, 4, tp=7, dlg=3),         # better CER, worse dialogue than c: no
             cand("e", 149, 4, tp=8, dlg=2),         # better than c: replaces it
             cand("f", 200, 1, tp=9),                # over 150 ms
             cand("g", 70, 2, tp=9, c_equal=False),  # C != twin
             cand("q", 130, 3, tp=8, dlg=4)]         # resolved worse on dialogue than ref: never qualifies
    r = R.apply_rule2(cands, baselines, ref, fast_ref, RULE)
    assert r["eligible"] == ["a", "b", "c", "q", "d", "e"]
    assert r["main"]["qualifying"] == ["a", "b", "c", "d", "e"]
    assert r["main"]["winner"] == "e"
    assert [(s["replaces"], s["by"]) for s in r["main"]["steps"]] == [("a", "c"), ("c", "e")]
    assert r["fast"] == {"pool": ["a"], "qualifying": ["a"], "tier": 1, "winner": "a", "steps": [],
                         "comparator": "ref"}


def test_bar_blocks_and_fast_tier2():
    baselines = {"grad": arm(5, 10, dlg=10), "det": arm(4, 12, dlg=2)}
    ref = {"name": "ref", "arm": arm(6, 5)}
    fast_ref = {"name": "fref", "arm": arm(5, 9)}
    cands = [cand("h", 70, 7, dlg=2),               # not better than ref (CER 7 > 5); better than fast_ref
             cand("k", 100, 4, tp=7, dlg=3)]         # better than ref, but resolved worse on dialogue than "det"
    r = R.apply_rule2(cands, baselines, ref, fast_ref, RULE)
    assert r["bar"] == ["h"] and r["main"]["winner"] is None and r["main"]["qualifying"] == []
    assert r["fast"]["tier"] == 2 and r["fast"]["winner"] == "h" and r["fast"]["comparator"] == "fref"
    assert r["comparisons"]["h"]["better_than_fast_ref"] and not r["comparisons"]["h"]["better_than_ref"]


def test_no_pick():
    baselines = {"grad": arm(5, 10, dlg=10)}
    ref = {"name": "ref", "arm": arm(6, 5)}
    fast_ref = {"name": "fref", "arm": arm(6, 6)}
    r = R.apply_rule2([cand("x", 70, 8), cand("y", 140, 6)], baselines, ref, fast_ref, RULE)
    assert r["main"]["winner"] is None and r["fast"]["winner"] is None and r["fast"]["tier"] is None


def test_dnc_units():
    refs = [{"cls": "dialogue", "text": "あいう"}, {"cls": "name", "text": "名前"}, {"cls": "ui", "text": "SAVE"},
            {"cls": "choice", "text": "はい"}, {"cls": "dialogue", "text": None, "ignore": True}]
    u = R.dnc_units([{"unit": "f1", "vn": "v1", "refs": refs, "per_ref": [1, 0, 4, 2, None]}])
    assert u == [{"unit": "f1", "vn": "v1", "errors": 3, "chars": 7}]


# ----------------------------------------------------------------------------- the protocol

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_protocol_pins_and_rule():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = dict(p["sources_sha256"])
    for amend in p.get("amendments", []):
        pins.update(amend.get("sources_sha256", {}))
    for path, digest in pins.items():
        assert sha(path) == digest, f"{path} changed after the textspot2 protocol froze it (re-pin by an amendment)"
    assert "eval/textspot2_rule.py" in pins and "eval/textspot_metrics.py" in pins
    r = p["rule"]
    assert (r["ms_max_333_960"], r["fast_ms_max_333_960"], r["arena_max"], r["vtsp_max"]) == (150, 80, 131072, 262144)
    assert p["budget"]["ms_max_333_960"] == 150 and p["seeds"]["per_arm"] == [1, 2, 3]
    assert p["reused"]["dev_manifest"]["sha256"] == sha("eval/manifests/textspot-dev.jsonl")
    for k in ("ref", "fast_ref"):
        ref = p["references"][k]
        assert len(ref["units_sha256"]) == 64 and len(ref["blob_sha256"]) == 64
    shapes = {k for k in p["shapes"] if isinstance(p["shapes"][k], dict)}
    assert {"s4p_c8", "s4_c24n4", "s4p_c6n4", "s4p_c8n4", "s4p_c10n4", "s4p_c12", "s4_c32n3"} <= shapes
    assert all(p["shapes"][k]["head"] in ("kroa", "kroa_x2") for k in shapes)

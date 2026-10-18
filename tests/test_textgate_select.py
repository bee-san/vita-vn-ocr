"""textgate (docs/textgate.md): the calibration of an operating point, the gated development units, the selection
rule's bookkeeping, the data split and the presence trainer's targets. No data, no network."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import textgate as G  # noqa: E402


def test_threshold_keeps_the_target_share():
    s = list(range(1000))                       # 0..999
    t = G.threshold_at(s, 0.998)                # q = 2: the third smallest
    assert t == 2 and np.mean(np.asarray(s) >= t) >= 0.998
    assert G.threshold_at([5] * 10, 0.999) == 5           # ties are kept
    assert G.threshold_at([3, 1, 2], 0.999) == 1          # q = 0: the smallest


def test_calibrate_scores_counts_only_cases_the_finder_found():
    # 1,000 dialogue cases; the two lowest scores belong to cases GRAD + refine returned nothing for
    d = {1: [-100, -90] + list(range(998)), 2: [0] * 1000}
    found = [0, 0] + [1] * 998
    e = {1: [-50] * 9 + [500], 2: [0] * 10}
    emp_found = [1] * 9 + [0]
    out = G.calibrate_scores(d, e, found, emp_found, ["vita"] * 1000, ["realplus"] * 1000)
    op = out["0.998"]
    assert op["cases"] == 998 and op["k"] == 1
    assert op["t"] == 1                         # q = floor(0.002 * 998) = 1: the second smallest live score
    assert op["recall"] >= 0.998 and op["dropped"] == 1
    assert op["empty_rate"] == 1.0              # nine below t, the tenth has no finder line
    assert op["gate_rejects"] == 0.9 and op["screen_pass"]
    assert out["0.999"]["t"] == 0               # q = 0 with 998 cases


def test_calibrate_scores_prefers_the_smaller_k_on_a_tie():
    d = {1: list(range(100)), 2: list(range(100)), 4: list(range(100))}
    e = {1: [-5] * 10, 2: [-5] * 10, 4: [-5] * 10}
    op = G.calibrate_scores(d, e, [1] * 100, [1] * 10, ["vita"] * 100, ["x"] * 100)["0.998"]
    assert op["k"] == 1 and op["empty_rate"] == 1.0


def _units():
    def unit(name, i, vn, chars, errors, shot, plat="vita", variant="base"):
        return {"unit": f"{name}_{i}|{variant}", "shot": shot, "vn": vn, "platform": plat, "variant": variant,
                "errors": errors, "chars": chars, "hira_errors": 0, "hira_chars": 0, "detections": 1}
    base = {"tv_dlg": {"refine": [unit("a", i, f"v{i % 7}", 20, i % 3, f"s{i}") for i in range(40)]},
            "dn_dlg": {"refine": [unit("b", i, f"v{100 + i % 5}", 10, 0, f"d{i}", "psp") for i in range(20)]}}
    return base


def test_gated_units_stop_reads_every_character():
    base = _units()
    keep = {f"tv_dlg|{u['unit']}": True for u in base["tv_dlg"]["refine"]}
    keep.update({f"dn_dlg|{u['unit']}": True for u in base["dn_dlg"]["refine"]})
    stopped = base["tv_dlg"]["refine"][3]
    keep[f"tv_dlg|{stopped['unit']}"] = False
    g = G.gated_units(base, keep)
    assert sum(u.get("gated", False) for u in g["tv_dlg"]["gate"]) == 1
    u = next(x for x in g["tv_dlg"]["gate"] if x.get("gated"))
    assert u["errors"] == u["chars"] == 20 and u["detections"] == 0
    assert g["tv_dlg"]["refine"] == base["tv_dlg"]["refine"]       # the baseline is untouched
    with pytest.raises(KeyError):
        G.gated_units(base, {})                                     # every case needs a decision


def test_evaluate_counts_drops_and_empties():
    base = _units()
    models = {m: base for m in G.MODELS}
    cases = [f"tv_dlg|{u['unit']}" for u in base["tv_dlg"]["refine"]] + \
            [f"dn_dlg|{u['unit']}" for u in base["dn_dlg"]["refine"]]
    keep = [True] * len(cases)
    meta = [{"finder_lines": 1 if i < 8 else 0, "in_scene_pool": i % 2 == 0} for i in range(10)]
    dec = {"wd_cases": cases, "empties": {"development": meta, "report": meta[:4]}}
    cand = {"wd_keep": keep, "empty_keep": {"development": [False] * 6 + [True] * 4, "report": [True] * 4}}
    rep = G.evaluate(cand, dec, models, native=set())
    for m in G.MODELS:
        assert rep["models"][m]["gate_minus_refine"]["delta"] == 0
        assert rep["models"][m]["units_gated"] == 0
    assert rep["wd_cases_dropped"] == 0
    e = rep["empty_development"]
    assert e["none_returned"] == 8 and e["regions"] == 10 and e["rate"] == 0.8   # 6 stopped + 2 the finder missed
    assert e["in_scene_pool"] == [4, 5] and e["not_in_scene_pool"] == [4, 5]
    assert rep["empty_report"]["none_returned"] == 0


def test_protocol_rule_fields():
    p = json.loads((ROOT / "eval" / "textgate_protocol.json").read_text(encoding="utf-8"))
    r = p["selection_rule"]
    assert r["wd_cost_cap_points"] == 0.10 and r["empty_rate_min"] == 0.85
    assert p["gate"]["k_set"] == list(G.K_SET) and p["gate"]["targets"] == list(G.TARGETS)
    assert G.SCREEN_EMPTY == 0.85 and p["issue"] == 100
    runs = {x["run"]: x["blob_bytes"] for x in p["candidates"]["new_presence_runs"]}
    assert all(b < 2100 for b in runs.values()) and len(runs) == 8
    assert all(x["blob_bytes"] < 4100 for x in p["candidates"]["r97_runs"])


def test_calibration_split_is_deterministic_and_a_fifth():
    import textgate_data as D

    import textgate_train as GT
    vns = [f"v{i}" for i in range(20000)]
    share = np.mean([D.calib_vn(v) for v in vns])
    assert 0.19 < share < 0.21
    assert all(D.calib_vn(v) == GT.calib_vn(v) for v in vns[:2000])   # the trainer's copy agrees


def test_presence_targets():
    import textgate_train as GT
    s = 8
    lines = [[16, 16, 80, 40, 0, "text"],          # a horizontal line: cells with centres inside are text
             [200, 0, 220, 100, 1, "text"],        # vertical: no loss over its blocks
             [300, 180, 340, 200, 0, "text"]]      # cut by the window's bottom edge (h = 190): no loss
    P, V = GT.targets(lines, [[400, 8, 440, 30]], 0, 0, 480, 190, s)
    assert P.shape == (GT.WIN_H // s, GT.WIN_W // s)
    rows, cols = np.nonzero(P)
    assert set(rows.tolist()) == {2, 3, 4} and cols.min() == 2 and cols.max() == 9
    assert not V[0:13, 25:28].any()                # the vertical line's blocks
    assert not V[1:4, 50:55].any()                 # the ignore area
    assert not V[22:24, 37:43].any()               # the cut line
    assert V[10, 15] and P[10, 15] == 0            # far from any text: a negative
    assert not V[:, 60:].any()                     # outside the 480-px window: invalid
    P2, V2 = GT.targets([[0, 44, 30, 47, 0, "text"]], [], 0, 0, 512, 192, s)   # 3 px: no cell centre inside
    assert P2[5].sum() == 4 and P2.sum() == 4      # the row nearest the line's centre (45.5 -> row 5), its 4 cells

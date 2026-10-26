"""eval/vocr6_gate.py's pure parts (VOCR6 stage 4: the head gate and the two-level head on the VOCR6 blobs, #125;
eval/vocr6_gate_protocol.json): the arms and their cascades, #120's and #122's rules as frozen, the VOCR6-path check,
Rubi's re-read and cascade, the A9 arithmetic (checked against #120's and #122's committed records), the components
with excluded VNs, the step records and the lead's message. No development, test or private data is read here.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "eval", ROOT / "train", ROOT / "runtime", ROOT / "synth"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import vocr6_gate as G  # noqa: E402


def cmp_of(score, tv=(-0.1, 0.1), dn=(-0.1, 0.1), wd=(-0.1, 0.1)):
    return {"score": {"delta_points": 0.0, "ci95": list(score)},
            **{c: {"delta_points": 0.0, "ci95": list(v)} for c, v in (("tv", tv), ("dn", dn), ("wd", wd))}}


# ---------------------------------------------------------------- arms

@pytest.mark.parametrize("arm,parts", [
    ("today", {"two": False, "gate": None, "rc": None}),
    ("gate", {"two": False, "gate": "e3", "rc": None}),
    ("gate_e4", {"two": False, "gate": "e4", "rc": None}),
    ("2L", {"two": True, "gate": None, "rc": "r0.999"}),
    ("2L_rc9999", {"two": True, "gate": None, "rc": "r0.9999"}),
    ("2Lg", {"two": True, "gate": "e3", "rc": "r0.999"}),
    ("2Lg_e4", {"two": True, "gate": "e4", "rc": "r0.999"}),
    ("2Lg_rc9999", {"two": True, "gate": "e3", "rc": "r0.9999"}),
    ("2Lg_e4_rc9999", {"two": True, "gate": "e4", "rc": "r0.9999"})])
def test_arm_parts_round_trip(arm, parts):
    assert G.arm_parts(arm) == parts
    assert G.arm_name(parts["two"], parts["gate"], parts["rc"]) == arm


@pytest.mark.parametrize("bad", ["", "gate_rc9999", "2L_e4", "2Lgx", "gate_e3", "Today", "2L_e4_rc9999"])
def test_arm_parts_refuses(bad):
    with pytest.raises(ValueError):
        G.arm_parts(bad)


def test_cascade_and_references():
    # development.cascade: the next tier's arm of the same kind at its primary settings
    assert [G.cascade_arm(a) for a in ("today", "gate", "gate_e4", "2L", "2L_rc9999", "2Lg", "2Lg_e4",
                                       "2Lg_e4_rc9999")] == ["today", "gate", "gate", "2L", "2L", "2Lg", "2Lg", "2Lg"]
    assert G.gate_arm_of("2Lg_e4_rc9999") == "gate_e4" and G.gate_arm_of("2Lg") == "gate"
    assert G.gate_arm_of("2L") is None and G.gate_arm_of("today") is None
    assert G.NEXT == {"S": "N", "N": "L", "L": None} and G.ORDER == ("L", "N", "S")


def test_two_candidate():
    assert G.two_candidate("S", "gate") is None
    assert G.two_candidate("N", "gate") == "2Lg"
    assert G.two_candidate("L", "gate_e4") == "2Lg_e4"
    assert G.two_candidate("N", None) == "2L"
    assert G.two_candidate("L", "gate", "r0.9999") == "2Lg_rc9999"
    assert G.two_candidate("N", None, "r0.9999") == "2L_rc9999"


def test_a9_configurations():
    assert G.a9_cfgs("today") == [("today", (0, 1))]
    assert G.a9_cfgs("gate_e4") == [(f"gate_e4_{m}", (0, 1)) for m in ("off", "score", "on")]
    two = dict(G.a9_cfgs("2Lg_e4_rc9999"))
    assert two == {"2Lg_e4_rc9999": (0, 1), "today": (0, 1), "2L_tinf": (0,), "2L_t-inf": (0,), "gate_e4_on": (0, 1)}
    assert "gate_on" not in dict(G.a9_cfgs("2L"))
    assert [G.a9_cfg_of_arm(a) for a in ("today", "gate", "2Lg", "2L")] == ["today", "gate_on", "2Lg", "2L"]


def test_names():
    assert G.shipping_name("VOCR6", "S", "gate") == "VOCR6_S_w8g.vocr"
    assert G.shipping_name("VOCR6", "N", "2Lg_e4") == "VOCR6_N_w8g2.vocr"
    assert G.shipping_name("VOCR6", "L", "2L") == "VOCR6_L_w8_2l.vocr"
    assert G.shipping_name("VOCR6", "L", "today") == "VOCR6_L_w8.vocr"
    assert G.final_arm(None, None) == "today" and G.final_arm("gate", None) == "gate"
    assert G.final_arm("gate", "2Lg") == "2Lg"


# ---------------------------------------------------------------- rules

def test_gate_rule_edges():
    # a lower bound of exactly 0 is not "entirely above 0"; the margin and the saving are inclusive
    r = G.gate_rule(cmp_of((0.0, 0.05)), 0.05)
    assert r == {"not_resolved_worse": True, "upper_within_margin": True, "saving_at_least_5pct": True}
    assert not G.gate_rule(cmp_of((0.001, 0.02)), 0.2)["not_resolved_worse"]
    assert not G.gate_rule(cmp_of((-0.1, 0.02), dn=(0.001, 0.1)), 0.2)["not_resolved_worse"]
    assert not G.gate_rule(cmp_of((-0.1, 0.0501)), 0.2)["upper_within_margin"]
    assert not G.gate_rule(cmp_of((-0.1, 0.01)), 0.0499)["saving_at_least_5pct"]
    assert not G.gate_rule(cmp_of((-0.1, 0.01)), None)["saving_at_least_5pct"]


def test_gate_verdict():
    ok = G.gate_rule(cmp_of((-0.01, 0.01)), 0.1)
    edge = G.gate_rule(cmp_of((-0.01, 0.06)), 0.1)          # margin fails: behind the switch
    worse = G.gate_rule(cmp_of((0.01, 0.06)), 0.1)
    assert G.gate_verdict({d: ok for d in G.DECODES}) == "adopted"
    assert G.gate_verdict({"greedy": ok, "rubi": edge, "rubi_cascade": ok}) == "behind_the_switch"
    assert G.gate_verdict({"greedy": ok, "rubi": edge, "rubi_cascade": worse}) == "rejected"
    slow = G.gate_rule(cmp_of((-0.01, 0.01)), 0.04)
    assert G.gate_verdict({d: slow for d in G.DECODES}) == "behind_the_switch"


def test_two_rule_and_verdict():
    rare_ok = {"combined": {"loss_points": 1.0, "ci95": [0.0, 2.0]}}
    rare_bad = {"combined": {"loss_points": 0.5, "ci95": [0.0, 2.01]}}
    good = G.two_rule(cmp_of((-0.01, 0.01)), rare_ok)
    assert good == {"a_not_resolved_worse": True, "b_noninferior_0.05": True, "c_rare_recall": True}
    assert not G.two_rule(cmp_of((-0.01, 0.01)), rare_bad)["c_rare_recall"]
    checks = {f"{r}/{d}": good for r in ("today", "gate") for d in G.DECODES}
    assert G.two_verdict(checks, True, True, True) == ("adoptable", [])
    v, fails = G.two_verdict(checks, False, True, True)
    assert v == "not adopted" and fails == ["d_saving_at_least_3pct"]
    v, fails = G.two_verdict(checks, True, True, False)
    assert v == "not adopted" and "gate reference missing" in fails
    checks["gate/rubi"] = G.two_rule(cmp_of((0.001, 0.01)), rare_ok)
    v, fails = G.two_verdict(checks, True, True, True)
    assert v == "rejected" and fails == ["gate/rubi: a_not_resolved_worse"]


def test_path_check():
    def doc(hv, dv):
        return {"total": {"hval_page": {"ci95": list(hv)}, "dev": {"ci95": list(dv)}}}
    assert G.path_check(doc((-0.1, 0.0), (0.0, 0.3)))["not_worse"]
    r = G.path_check(doc((0.001, 0.2), (-0.1, 0.1)))
    assert not r["not_worse"] and not r["hval_page_not_resolved_worse"] and r["dev_not_resolved_worse"]
    assert not G.path_check(doc((-0.1, 0.1), (0.01, 0.1)))["not_worse"]


def test_read_again_is_rubis():
    # gated frames and a counter context; the two-level head's full-size カ / ケ before a counter character
    assert G.read_again(3, 0, 50, 1, False, "三ヶ月")
    assert not G.read_again(0, 0, 50, 1, False, "三ヶ月")                 # nothing skipped
    assert not G.read_again(3, 0, 50, 0, False, "三カ月")                 # the kana rule needs the two-level head
    assert G.read_again(0, 10, 50, 0, True, "三カ月")                     # level 1 only on 40 frames
    assert not G.read_again(0, 50, 50, 0, True, "三カ月")                 # every frame escaped: nothing skipped
    assert not G.read_again(0, 10, 50, 0, True, "カメラ")
    assert G.read_again(5, 10, 50, 0, True, "一ケ所")


def test_line_conf_and_cascade():
    assert G.line_conf(0.2, "") == 1000
    assert G.line_conf(0.6994, "あ") == 699 and G.line_conf(0.6995, "あ") == 700
    assert G.line_conf(1.5, "あ") == 1000 and G.line_conf(-1.0, "あ") == 0
    assert G.cascade(["a", "b", "c"], [699, 700, 1000], ["A", "B", "C"]) == ["A", "b", "c"]
    assert G.cascade(["a"], [0], None) == ["a"]
    with pytest.raises(ValueError):
        G.cascade(["a"], [0], ["A", "B"])


# ---------------------------------------------------------------- A9 arithmetic against the committed records

def test_gate_saving_reproduces_120():
    rec = json.loads((ROOT / "eval" / "headgate_a9_20261023.json").read_text())
    for t in ("S", "N", "L"):
        tr = rec["tiers"][t]
        for pld in ("pld0", "pld1"):
            r = tr["runs"][pld]
            ph = r["phases"]
            box0 = G.box_ms({"ms_per_frame": ph["off"]["ms_per_frame"], "frames": 1609})
            assert box0 == pytest.approx(r["box_ms_off"], abs=0.06)
            g = G.gate_saving(ph["off"], ph["score"], ph["on"], tr["pack_skip_share"], tr["dev_wd_skip_share"], box0)
            assert g["kappa"] == pytest.approx(r["kappa"], abs=2e-3)
            for name in ("pack", "dev_wd"):
                mine, ref = g["saving"][name], r["saving"][name]
                assert mine["ms_per_frame"] == pytest.approx(ref["ms_per_frame"], abs=2e-3)
                assert mine["nominal_share"] == pytest.approx(ref["nominal_share"], abs=5e-4)
                assert mine["nominal_box_ms"] == pytest.approx(ref["nominal_box_ms"], abs=0.6)


def test_two_saving_reproduces_122():
    tim = json.loads((ROOT / "eval" / "head2level_dev_20261023.json").read_text())["timing"]
    for t in ("N", "L"):
        today = tim["tiers"][t]["today"]["nopld"]
        box_today = G.box_ms(today)
        for arm, gated in (("2Lg_K500_r96", True), ("2L_K500_r96", False)):
            a = tim["tiers"][t]["arms"][arm]
            m, pj = a["measured"]["nopld"], a["projected"]
            assert box_today == pytest.approx(m["nominal_box_ms_today"], abs=0.06)
            wd = {"gated_share": pj.get("wd_gated_share"), "fire_share_all_frames": pj.get("wd_fire_share_all_frames"),
                  "fire_share": pj.get("wd_fire_share")}
            out = G.two_saving(today, {"ms_per_frame": m["ref_ms_per_frame"]}, {"ms_per_frame": m["ms_per_frame"]},
                               {"head_ms": pj["head_ms_never"]}, {"head_ms": pj["head_ms_always"]}, wd, gated,
                               box_today)
            assert out["measured"]["saved_pct_nominal_box"] == pytest.approx(m["saved_pct_nominal_box"], abs=0.02)
            assert out["projected"]["saved_pct_nominal_box"] == pytest.approx(pj["saved_pct_nominal_box"], abs=0.03)
            assert out["d_saving_at_least_3pct"] == a["d_saving_at_least_3pct"]


def test_box_with_attention():
    r = {"ms_per_frame": 3.0, "frames": 1609, "attn_core_ms": 0.0}
    assert G.box_ms(r) == pytest.approx(240 * 3.0 + 100)
    # the attention core (per frame on the pack) scaled by T: the pack's sum of T^2 against 3 lines of 80 frames
    r = {"ms_per_frame": 3.0, "frames": 1609, "attn_core_ms": 0.068}
    st2 = 108719
    core_t2 = 0.068 * 1609 / st2
    assert G.box_ms(r, st2) == pytest.approx(240 * (3.0 - 0.068) + 3 * 80 * 80 * core_t2 + 100)
    assert G.box_ms(r, st2) > G.box_ms(r)          # the pack's lines average fewer frames than 80


def test_saving_pct():
    s = G.saving_pct(3.0, 2.5, 820.0)
    assert s["saved_ms_nominal_box"] == 120.0 and s["saved_pct_nominal_box"] == pytest.approx(14.63, abs=0.01)
    assert s["saved_ms_nominal_box_444"] == 90.0 and s["saved_ms_real_box"] == 44.5


def test_effective_macs():
    est = {"arm": "2Lg", "macs_per_frame_full_head": 1_000_000, "head_macs_per_frame": 96 * 5632, "L1": 504}
    skip = {"gated_share": 0.6, "fire_share_all_frames": 0.08}
    head = 96 * 5632
    want = 1_000_000 - head + 1552 + 0.4 * (504 * 96 + G.GATE_ESC_MACS) + 0.08 * (5632 - 504) * 96
    assert G.effective_macs(est, skip) == round(want)
    est["arm"] = "gate"
    assert G.effective_macs(est, skip) == round(1_000_000 - head + 1552 + 0.4 * head)
    est["arm"] = "today"
    assert G.effective_macs(est, skip) == 1_000_000


# ---------------------------------------------------------------- components without the excluded VNs

def tiny_dev():
    """A Dev without files: a teacher_val stem with three rows (two VNs and a name), one devnative row, one wd case per
    cohort (the second of the excluded VN v18432)."""
    d = G.Dev.__new__(G.Dev)
    d.pools = Path("/nonexistent")
    rows = [{"id": "a", "vn": "v1", "ref": "あいう", "tags": []},
            {"id": "b", "vn": "v18432", "ref": "かき", "tags": []},
            {"id": "c", "vn": "v2", "ref": "名前", "tags": ["namelike"]}]
    dn = [{"id": "d1", "vn": "v3", "ref": "なに", "tags": []}]
    d.fixed = {G.TV: rows, G.DN[0]: dn, **{s: [] for s in G.DN[1:]}}
    d.cases = {"tv_dlg": [{"shot": "s1", "vn": "v1", "unit": "u1", "variant": "base",
                           "lines": [{"id": "l1", "text": "さしす", "styles": []}]}],
               "dn_dlg": [{"shot": "s2", "vn": "v18432", "unit": "u2", "variant": "base",
                           "lines": [{"id": "l2", "text": "たち", "styles": []}]}]}
    d.n_lines = {"tv_dlg": {"grad": [1], "refine": [1]}, "dn_dlg": {"grad": [1], "refine": [1]}}
    d.tv_shots = {"s1"}
    d.dn_names = {"d1": False}
    keys = [f"{G.TV}|{i}|{v}" for i in range(3) for v in ("rv", "rvn")] + [f"{G.DN[0]}|0|rv", f"{G.DN[0]}|0|rvn"]
    keys += [f"{n}|{m}|0|0" for n in G.WD for m in ("grad", "refine")]
    d.keys, d.index = keys, {k: i for i, k in enumerate(keys)}
    return d


GOOD = ["あいう", "あいう", "かき", "かき", "名前", "名前", "なに", "なに", "さしす", "さしす", "たち", "たち"]
BAD = ["あいx", "あいう", "", "", "名前", "名", "な", "なに", "さしす", "さ", "", ""]


def test_rows_leave_out_the_excluded_vns():
    import vocr5_select as V
    d = tiny_dev()
    assert d.vn_of(f"{G.TV}|1|rv") == "v18432" and d.vn_of("dn_dlg|grad|0|0") == "v18432"
    rows_all = d.rows(BAD, frozenset())
    rows_ex = d.rows(BAD, frozenset(G.EXCLUDE_VNS))
    assert rows_all["tv"][2] == ["v1", "v18432", "v2"] and rows_ex["tv"][2] == ["v1", "v2"]
    assert rows_all["tv"][0] == [1, 4, 1] and rows_ex["tv"][0] == [1, 1]
    assert rows_all["dn"][0] == [1] and rows_all["dn"][1] == [4]
    # wd rows in vocr5_select's sorted key order: dn_dlg's (v18432) before tv_dlg's
    assert rows_all["wd"][2] == ["v18432", "v18432", "v1", "v1"] and rows_ex["wd"][2] == ["v1", "v1"]
    assert rows_ex["wd"][0] == [0, 2] and rows_all["wd"][0] == [2, 2, 0, 2]
    comps = d.components({"x": d.rows(GOOD, frozenset(G.EXCLUDE_VNS)), "y": rows_ex})
    assert comps["tv"].rate(["y"]) == pytest.approx(1 / 6)           # names left out of dialogue
    assert comps["tv"].rate(["y"], "name") == pytest.approx(1 / 4)
    assert comps["wd"].rate(["y"]) == pytest.approx(2 / 6)
    out = V.compare(comps, ["x"], ["y"], n_boot=200)
    assert out["tv"]["vns"] == 1 and out["dn"]["vns"] == 1 and out["wd"]["vns"] == 1
    assert out["wd"]["delta_points"] == pytest.approx(100 * 2 / 6, abs=1e-3)


# ---------------------------------------------------------------- steps, messages, the protocol

class _Args:
    prefix, all_vns, frozen_only, path_lm, path_lm_opts, procs = "VOCR6", False, True, "lm", "", 5

    def __init__(self, priv):
        self.priv = str(priv)


def test_step_runs_once_and_records_slot_hours(tmp_path):
    ctx = G.Ctx(tmp_path / "out", _Args(tmp_path / "priv"))
    calls = []
    a = G.step(ctx, "N", "demo", lambda: calls.append(1) or {"value": 3})
    b = G.step(ctx, "N", "demo", lambda: calls.append(1) or {"value": 4})
    assert calls == [1] and a["value"] == b["value"] == 3 and a["slots"] == 1
    assert a["slot_hours"] == round(a["wall_s"] / 3600, 4)
    t = G.timing(ctx)
    assert t["rows"][0]["step"] == "demo" and set(t["slot_hours"]) == set(G.TIERS)
    again = G.Ctx(tmp_path / "out")                      # the configuration is the first run's
    assert again.exclude == frozenset(G.EXCLUDE_VNS) and again.frozen_only


def test_parse_final_blobs():
    msg = "vocr6-lead: FINAL BLOBS S=/a/S_w8.vocr N=/a/N_w8.vocr L=/a/L_w8.vocr."
    assert G.parse_final_blobs(msg) == {"S": "/a/S_w8.vocr", "N": "/a/N_w8.vocr", "L": "/a/L_w8.vocr"}
    two = msg + "\nFINAL BLOBS S=/b/S N=/b/N L=/b/L"
    assert G.parse_final_blobs(two)["N"] == "/b/N"
    assert G.parse_final_blobs("FINAL BLOBS S=<path> N=<path>") is None
    assert G.parse_final_blobs("nothing") is None


def test_protocol_matches_the_driver():
    proto = json.loads(G.PROTOCOL.read_text(encoding="utf-8"))
    assert proto["kind"] == "vocr6-gate-protocol" and proto["issue"] == 125
    assert tuple(proto["development"]["exclude_vns"]) == G.EXCLUDE_VNS
    assert G.RANK_SHA in proto["two_level"]["ranking"]
    assert proto["builds_on"]["vocr6"]["kit"] == G.KIT_PINS
    assert proto["vocr6_path"]["plans"]["hval"]["sha256"] == G.PATH_PLANS["hval"][1]
    assert proto["vocr6_path"]["plans"]["dev"]["sha256"] == G.PATH_PLANS["dev"][1]
    assert proto["vocr6_path"]["plans"]["dev"]["units_sha256"] == G.DEV_UNITS[1]
    assert G.RUBI_NORM[1][:8] in proto["vocr6_path"]["tool"]
    assert G.PACK_SHA in proto["estimates"]["method"] and str(G.SCALE) in proto["estimates"]["method"]
    assert G.DEV_LM_SHA in proto["development"]["decodes"]
    assert G.TEST_TITLE in proto["vocr6_path"]["plan_hygiene"]
    assert proto["decision"]["order"].startswith("Large, then Normal, then Small")
    assert G.TWO_LEVEL_TIERS == tuple(proto["two_level"]["tiers"])


def test_completion_tokens_never_printed():
    src = (ROOT / "eval" / "vocr6_gate.py").read_text(encoding="utf-8") + \
        (ROOT / "eval" / "vocr6_gate.sh").read_text(encoding="utf-8")
    import re
    assert not re.search(r"\b[A-Z0-9_]+_(DONE|BLOCKED)\b", src.replace("[A-Z0-9_]+_(DONE|BLOCKED)", ""))


def test_rows_refuse_duplicate_wd_units():
    d = tiny_dev()
    d.cases["dn_dlg"][0].update(unit="u1", shot="s3")
    d.cases["tv_dlg"][0]["variant"] = "base"
    d.cases["dn_dlg"][0]["variant"] = "base"
    d.cases["tv_dlg"][0]["unit"] = "u1"
    rows = d.rows(GOOD, frozenset())           # different cohorts: distinct keys
    assert len(rows["wd"][0]) == 4

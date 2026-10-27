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
    d.kinds = {G.TV: ["vita", "psp", "hires"], G.DN[0]: ["psp"], **{s: [] for s in G.DN[1:]}}
    d.cases["tv_dlg"][0]["kind"] = "vita"
    d.cases["dn_dlg"][0]["kind"] = "psp"
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
    prefix, all_vns, frozen_only, path_lm_hval, path_lm_dev, path_lm_opts, procs = "VOCR6", False, True, "a", "b", "", 5
    no_plans4 = False

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
    assert t["rows"][0]["step"] == "demo" and set(t["slot_hours"]) == {"dev", *G.TIERS}
    again = G.Ctx(tmp_path / "out")                      # the configuration is the first run's
    assert again.exclude == frozenset(G.EXCLUDE_VNS) and again.frozen_only


def test_vocr5_released_pins_match_the_protocols():
    for f in ("headgate_protocol.json", "head2level_protocol.json"):
        models = json.loads((ROOT / "eval" / f).read_text(encoding="utf-8"))["models"]
        for t in G.TIERS:
            tier, gate, two = G.VOCR5_RELEASED[models[t]["sha256"]]
            assert tier == t and gate == "gate" and two == (None if t == "S" else "2Lg")


def test_parse_final_blobs():
    yes = lambda p: True  # noqa: E731
    msg = "vocr6-lead: FINAL BLOBS S=/a/S_w8.vocr N=/a/N_w8.vocr L=/a/L_w8.vocr."
    assert G.parse_final_blobs(msg, yes) == {"S": "/a/S_w8.vocr", "N": "/a/N_w8.vocr", "L": "/a/L_w8.vocr"}
    two = msg + "\nFINAL BLOBS S=/b/S N=/b/N L=/b/L"
    assert G.parse_final_blobs(two, yes)["N"] == "/b/N"
    assert G.parse_final_blobs("FINAL BLOBS S=<path> N=<path>", yes) is None
    assert G.parse_final_blobs("nothing", yes) is None
    # audit -b G3: a placeholder after the real message, a backtick-quoted message, a missing file
    real = {"/a/S_w8.vocr", "/a/N_w8.vocr", "/a/L_w8.vocr"}
    after = msg + "\nthe template: `FINAL BLOBS S=… N=… L=…`"
    assert G.parse_final_blobs(after, real.__contains__) == {"S": "/a/S_w8.vocr", "N": "/a/N_w8.vocr",
                                                            "L": "/a/L_w8.vocr"}
    quoted = "`FINAL BLOBS S=/a/S_w8.vocr N=/a/N_w8.vocr L=/a/L_w8.vocr`"
    assert G.parse_final_blobs(quoted, real.__contains__)["L"] == "/a/L_w8.vocr"
    assert G.parse_final_blobs(msg.replace("/a/L_w8", "/x/L_w8"), real.__contains__) is None


def test_redo_plan():
    same = {"S": "s", "N": "n", "L": "l"}
    assert G.redo_plan(same, same) == ([], [])
    assert G.redo_plan(same, dict(same, S="s2")) == (["S"], ["S"])
    assert G.redo_plan(same, dict(same, N="n2")) == (["N"], ["N", "S"])
    assert G.redo_plan(same, dict(same, L="l2")) == (["L"], ["L", "N", "S"])
    assert G.redo_plan({"S": "s", "N": "n"}, same) == (["L"], ["L", "N", "S"])        # L never prepared


def test_steps_of_two_jobs_run_once(tmp_path):
    import threading
    ctx = G.Ctx(tmp_path / "out", _Args(tmp_path / "priv"))
    runs = []

    def slow():
        runs.append(1)
        import time
        time.sleep(0.3)
        return {"v": len(runs)}
    res = []
    th = [threading.Thread(target=lambda: res.append(G.step(ctx, "L", "shared", slow))) for _ in range(3)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    assert runs == [1] and [r["v"] for r in res] == [1, 1, 1]


def test_timing_reports_superseded_steps_apart(tmp_path):
    ctx = G.Ctx(tmp_path / "out", _Args(tmp_path / "priv"))
    G.step(ctx, "N", "kept", lambda: {})
    d = ctx.out / "superseded" / "N-abc-000000" / "steps"
    d.mkdir(parents=True)
    (d / "old.json").write_text(json.dumps({"tier": "N", "step": "old", "start": "a", "end": "b", "wall_s": 360.0,
                                            "slot_hours": 0.1}))
    t = G.timing(ctx)
    assert [r["step"] for r in t["rows"]] == ["kept"] and t["superseded_slot_hours"] == 0.1
    assert t["superseded_rows"][0]["tier"] == "N-abc-000000 (superseded)"


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
    a1 = proto["amendments"][0]                         # the LM per plan and the stage-4 plans' report
    assert a1["n"] == 1 and all(h in a1["what"] for h in G.PATH_LM_SHA.values())
    assert all(h in a1["also"] for _, h in G.PLANS4.values())
    assert G.PATH_LM_OPTS in a1["what"]
    a2 = proto["amendments"][1]                         # audit F1 / N2 / N3 / N4 / N7, before step 5
    assert a2["n"] == 2 and "VOCR5_S_w8g.vocr" in a2["vocr5_final_as_implemented"]
    assert "VOCR5_{N,L}_w8g2.vocr" in a2["vocr5_final_as_implemented"]
    assert "read_again" in a2["reported_beside_as_produced"]
    assert "8ff76d83" in a2["runtime"] and "eps = 1e-4" in a2["deviation_added"]
    assert proto["amendments"][0]["n"] == 1
    a3 = proto["amendments"][2]                         # provisional runs, confirmed by FINAL BLOBS
    assert a3["n"] == 3 and "redo_plan" in a3["confirmation"] and "superseded" in a3["confirmation"]
    assert set(a3["provisional_inputs"]) >= set(G.TIERS)
    a4 = proto["amendments"][3]                         # Small kept as VOCR5-S by stage 4's decision
    assert a4["n"] == 4 and "keep-vocr5" in a4["change"] and "VOCR5_S_w8g.vocr" in a4["change"]


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


# ---------------------------------------------------------------- the decision flow, stubbed (audit F2)

class _Live(_Args):
    frozen_only = False


def stub_decide(monkeypatch, tmp_path, tier, gate_verdicts, two_verdicts=None, path_worse=(), sha="0" * 64,
                next_final="2Lg"):
    """decide() with its scoring stubbed: gate_verdicts {arm: #120 verdict}, two_verdicts {arm: #122 verdict}, the arms
    in path_worse resolved worse on the VOCR6 path. Returns (ctx, the calls in order, the blob of an arm)."""
    ctx = G.Ctx(tmp_path / "out", _Live(tmp_path / "priv"))
    base = tmp_path / "in_w8.vocr"
    base.write_bytes(b"base")
    ctx.tdir(tier)
    (ctx.out / tier / "input.json").write_text(json.dumps({"blob": str(base), "sha256": sha}))
    nt = G.NEXT[tier]
    if nt and next_final:
        ctx.tdir(nt)
        (ctx.out / nt / "decision.json").write_text(json.dumps({"final_arm": next_final}))
    calls = []

    def blob_of(c, t, arm):
        if arm == "today":
            return base
        b = tmp_path / f"{t}_{arm}.vocr"
        if not b.exists():
            b.write_bytes(f"{t}:{arm}".encode())
        return b

    def frozen_gate(c, t, arm):
        calls.append(("gate", arm))
        return {"verdict": gate_verdicts[arm], "blob_sha256": "-"}

    def frozen_two(c, t, arm):
        calls.append(("two", arm))
        v = (two_verdicts or {})[arm]
        return {"verdict": v, "fails": [] if v == "adoptable" else ["x"], "blob_sha256": "-"}

    def path(c, t, arm):
        calls.append(("path", arm))
        ok = arm not in path_worse
        return {"not_worse": ok, "check": {"not_worse": ok}}

    def estimate(c, t, arm):
        return {"arm": arm, "macs_per_frame_full_head": 2_000_000, "head_macs_per_frame": 96 * 5632}

    monkeypatch.setattr(G, "frozen_gate", frozen_gate)
    monkeypatch.setattr(G, "frozen_two", frozen_two)
    monkeypatch.setattr(G, "vocr6_path", path)
    monkeypatch.setattr(G, "config_estimate", estimate)
    monkeypatch.setattr(G, "arm_wd_shares", lambda c, t, a: {"gated_share": 0.6, "fire_share_all_frames": 0.08,
                                                             "fire_share": 0.2})
    monkeypatch.setattr(G, "arm_l1", lambda c, t, a: 504)
    monkeypatch.setattr(G, "ensure_blob", blob_of)
    monkeypatch.setattr(G, "shipped_cascade", lambda c, t, a: calls.append(("shipped", a)) or {"shipped_vs": {}})
    monkeypatch.setattr(G, "counter_report", lambda c, t, arms: calls.append(("counter", tuple(arms))) or {"arms": {}})
    return ctx, calls, blob_of


def test_decide_gate_fallback_then_two_level_on_it(monkeypatch, tmp_path):
    ctx, calls, _ = stub_decide(monkeypatch, tmp_path, "N", {"gate": "rejected", "gate_e4": "adopted"},
                                {"2Lg_e4": "adoptable"})
    dec = G.decide(ctx, "N")
    assert (dec["gate_final"], dec["two_final"], dec["final_arm"]) == ("gate_e4", "2Lg_e4", "2Lg_e4")
    assert calls[:6] == [("gate", "gate"), ("path", "gate"), ("gate", "gate_e4"), ("path", "gate_e4"),
                         ("two", "2Lg_e4"), ("path", "2Lg_e4")]
    assert ("counter", ("today", "gate", "gate_e4", "2Lg_e4")) in calls and ("shipped", "2Lg_e4") in calls
    assert dec["final_blob"]["name"] == "VOCR6_N_w8g2.vocr" and (ctx.out / "final" / "VOCR6_N_w8g2.vocr").exists()
    assert not dec["gate"]["gate"]["adopted"] and dec["gate"]["gate_e4"]["adopted"]
    assert json.loads((ctx.out / "N" / "decision.json").read_text())["final_arm"] == "2Lg_e4"


def test_decide_path_check_is_part_of_adoption(monkeypatch, tmp_path):
    ctx, calls, _ = stub_decide(monkeypatch, tmp_path, "N", {"gate": "adopted", "gate_e4": "adopted"},
                                {"2Lg_e4": "adoptable"}, path_worse=("gate",))
    dec = G.decide(ctx, "N")
    assert not dec["gate"]["gate"]["adopted"] and dec["gate_final"] == "gate_e4"
    assert ("gate", "gate_e4") in calls


def test_decide_two_level_alone_and_the_recall_fallback(monkeypatch, tmp_path):
    ctx, calls, _ = stub_decide(monkeypatch, tmp_path, "L", {"gate": "rejected", "gate_e4": "behind_the_switch"},
                                {"2L": "not adopted", "2L_rc9999": "adoptable"})
    dec = G.decide(ctx, "L")
    assert dec["gate_final"] is None and dec["two_final"] == "2L_rc9999"
    assert [c for c in calls if c[0] == "two"] == [("two", "2L"), ("two", "2L_rc9999")]
    assert dec["final_blob"]["name"] == "VOCR6_L_w8_2l.vocr"
    assert not any(c[0] == "shipped" for c in calls)                # Large has no next tier


def test_decide_path_check_is_part_of_the_two_level_adoption(monkeypatch, tmp_path):
    ctx, calls, _ = stub_decide(monkeypatch, tmp_path, "N", {"gate": "adopted"},
                                {"2Lg": "adoptable", "2Lg_rc9999": "adoptable"}, path_worse=("2Lg",))
    dec = G.decide(ctx, "N")
    assert not dec["two_level"]["2Lg"]["adopted"] and dec["two_final"] == "2Lg_rc9999"
    assert ("path", "2Lg_rc9999") in calls


def test_decide_a_vocr5_blob_of_another_tier_is_ruled_as_any_blob(monkeypatch, tmp_path):
    n_sha = next(h for h, v in G.VOCR5_RELEASED.items() if v[0] == "N")
    ctx, _, _ = stub_decide(monkeypatch, tmp_path, "L", {"gate": "rejected", "gate_e4": "adopted"},
                            {"2Lg_e4": "adoptable"}, sha=n_sha)
    dec = G.decide(ctx, "L")
    assert "kept_vocr5_verdict" not in dec and dec["final_arm"] == "2Lg_e4"


def test_decide_recall_fallback_on_the_gate(monkeypatch, tmp_path):
    ctx, _, _ = stub_decide(monkeypatch, tmp_path, "N", {"gate": "adopted"}, {"2Lg": "rejected",
                                                                               "2Lg_rc9999": "adoptable"})
    dec = G.decide(ctx, "N")
    assert (dec["gate_final"], dec["two_final"]) == ("gate", "2Lg_rc9999")


def test_decide_nothing_adopted_ships_the_input(monkeypatch, tmp_path):
    ctx, calls, _ = stub_decide(monkeypatch, tmp_path, "S", {"gate": "rejected", "gate_e4": "rejected"},
                                next_final="gate")
    dec = G.decide(ctx, "S")
    assert dec["final_arm"] == "today" and dec["two_final"] is None
    assert dec["final_blob"]["path"] == str(tmp_path / "in_w8.vocr") and dec["final_blob"]["name"] == "VOCR6_S_w8.vocr"
    assert not any(c[0] in ("two", "shipped") for c in calls)


def test_decide_needs_the_next_tier_decided(monkeypatch, tmp_path):
    ctx, calls, _ = stub_decide(monkeypatch, tmp_path, "N", {"gate": "adopted"}, {"2Lg": "adoptable"},
                                next_final=None)
    with pytest.raises(SystemExit):
        G.decide(ctx, "N")
    assert calls == []


def test_decide_keeps_a_released_vocr5_verdict_when_the_route_diverges(monkeypatch, tmp_path):
    n_sha = next(h for h, v in G.VOCR5_RELEASED.items() if v[0] == "N")
    ctx, calls, blob_of = stub_decide(monkeypatch, tmp_path, "N", {"gate": "rejected", "gate_e4": "adopted"},
                                      {"2Lg_e4": "adoptable", "2Lg": "adoptable"}, sha=n_sha)
    rel = tmp_path / "rel2"
    rel.mkdir()
    (rel / "VOCR5_N_w8g2.vocr").write_bytes(blob_of(ctx, "N", "2Lg").read_bytes())
    monkeypatch.setitem(G.REF, "two_release", rel)
    dec = G.decide(ctx, "N")
    k = dec["kept_vocr5_verdict"]
    assert k["by_the_rules"] == {"gate": "gate_e4", "two_level": "2Lg_e4"}
    assert k["kept"] == {"gate": "gate", "two_level": "2Lg"} and list(k["kept_arms_report"]) == ["2Lg"]
    assert (dec["gate_final"], dec["two_final"], dec["final_arm"]) == ("gate", "2Lg", "2Lg")
    assert ("two", "2Lg") in calls and ("path", "2Lg") in calls
    assert dec["final_blob"]["name"] == "VOCR5_N_w8g2.vocr" and k["released_asset"]["identical"]
    assert ("counter", ("today", "gate", "gate_e4", "2Lg_e4", "2Lg")) in calls


def test_decide_refuses_a_kept_blob_unlike_the_release(monkeypatch, tmp_path):
    s_sha = next(h for h, v in G.VOCR5_RELEASED.items() if v[0] == "S")
    ctx, _, _ = stub_decide(monkeypatch, tmp_path, "S", {"gate": "adopted"}, sha=s_sha, next_final="2Lg")
    rel = tmp_path / "rel1"
    rel.mkdir()
    (rel / "VOCR5_S_w8g.vocr").write_bytes(b"another blob")
    monkeypatch.setitem(G.REF, "gate_release", rel)
    with pytest.raises(SystemExit):
        G.decide(ctx, "S")


def test_frozen_only_run_decides_large_but_not_normal_for_small(monkeypatch, tmp_path):
    ctx = G.Ctx(tmp_path / "out", _Args(tmp_path / "priv"))            # frozen only
    calls = []
    monkeypatch.setattr(G, "prepare", lambda c, t, b: calls.append(("prepare", t)))
    monkeypatch.setattr(G, "decide", lambda c, t: calls.append(("decide", t)))
    G.run_tier(ctx, "N", "n.vocr", "l.vocr")
    assert calls == [("prepare", "L"), ("decide", "L"), ("prepare", "N"), ("decide", "N")]
    calls.clear()
    G.run_tier(ctx, "S", "s.vocr", "n.vocr")
    assert calls == [("prepare", "N"), ("prepare", "S"), ("decide", "S")]


# ---------------------------------------------------------------- rule edges (audit F2)

def test_two_level_d_needs_both_estimates():
    today = {"ms_per_frame": 3.75, "head_ms": 1.0}
    never, always = {"head_ms": 0.1}, {"head_ms": 1.1}
    box = G.box_ms(today)                                                  # 1,000 ms
    assert box == pytest.approx(1000.0)
    ref = {"ms_per_frame": 3.0}

    def d(measured_pct, s, f):
        arm = {"ms_per_frame": 3.0 - measured_pct * box / 100 / G.NOMINAL_FRAMES}
        return G.two_saving(today, ref, arm, never, always, {"gated_share": s, "fire_share_all_frames": f}, True, box)
    big = d(3.5, 0.6, 0.08)                      # projected (0.4 x 0.9 - 0.08) x 240 / 1000 = 6.72%
    assert big["projected"]["saved_pct_nominal_box"] == pytest.approx(6.72, abs=0.01) and big["d_saving_at_least_3pct"]
    lo_proj = d(3.5, 0.9, 0.05)                  # projected 0.96%: the measured 3.5% alone does not pass
    assert lo_proj["projected"]["saved_pct_nominal_box"] < 3 and not lo_proj["d_saving_at_least_3pct"]
    lo_meas = d(2.9, 0.6, 0.08)
    assert lo_meas["measured"]["saved_pct_nominal_box"] == pytest.approx(2.9, abs=0.01)
    assert not lo_meas["d_saving_at_least_3pct"] and lo_meas["saved_pct_nominal_box_safe_min"] == pytest.approx(2.9,
                                                                                                                abs=0.01)
    edge = d(3.0, 0.6, 0.08)
    assert edge["d_saving_at_least_3pct"]


def test_two_level_b_margin_edge():
    rare = {"combined": {"loss_points": 0.0, "ci95": [0.0, 0.0]}}
    assert G.two_rule(cmp_of((-0.01, 0.05)), rare)["b_noninferior_0.05"]
    assert not G.two_rule(cmp_of((-0.01, 0.0501)), rare)["b_noninferior_0.05"]
    assert not G.two_rule(cmp_of((-0.01, 0.01), wd=(0.0001, 0.1)), rare)["a_not_resolved_worse"]


def test_rows_skip_dn_dlg_frames_tv_dlg_holds():
    d = tiny_dev()
    d.cases["dn_dlg"][0]["shot"] = "s1"                    # the same frame as tv_dlg's case
    rows = d.rows(GOOD, frozenset())
    assert rows["wd"][2] == ["v1", "v1"] and len(rows["wd"][0]) == 2


def test_platforms_of_the_rows_and_their_rates():
    d = tiny_dev()
    pl = d.row_platforms(frozenset(G.EXCLUDE_VNS))
    assert pl == {"tv": ["vita", "pc"], "dn": ["psp"], "wd": ["vita", "vita"]}
    assert d.platform_of(f"{G.TV}|1|rv") == "psp" and d.platform_of("dn_dlg|grad|0|0") == "psp"
    rx, ry = d.rows(GOOD, frozenset(G.EXCLUDE_VNS)), d.rows(BAD, frozenset(G.EXCLUDE_VNS))
    r = G.platform_rates(rx, ry, pl)
    assert r["tv"]["vita"] == {"rows": 1, "chars": 6, "cer_x": 0.0, "cer_y": pytest.approx(100 / 6, abs=1e-3)}
    assert r["tv"]["pc"]["rows"] == 0                      # the name row is not dialogue
    assert r["wd"]["vita"]["cer_y"] == pytest.approx(100 * 2 / 6, abs=1e-3)


def test_counter_report_tallies(monkeypatch, tmp_path):
    import numpy as np
    meta = [{"text": "三ヶ月", "cut": 3}, {"text": "二ヶ所", "cut": 0}, {"text": "何ヶ国", "cut": -2}]
    f = tmp_path / "synth.npz"
    np.savez(f, meta=np.array(json.dumps(meta, ensure_ascii=False)))
    monkeypatch.setattr(G, "COUNTER_SET", (f, "-"))
    ctx = G.Ctx(tmp_path / "out", _Args(tmp_path / "priv"))
    docs = {"today": {"greedy": ["三ヶ月", "二ヶ所", "何ヶ国"], "rubi": ["三ヶ月", "二ヶ所", "何ヶ国"],
                      "rubi_reread": ["三ヶ月", "二ヶ所", "何ヶ国"], "reread": [0, 0, 0]},
            "gate": {"greedy": ["三ケ月", "二ヶ所", "何ヶ国"], "rubi": ["三ケ月", "二ヵ所", "何ヶ国"],
                     "rubi_reread": ["三ヶ月", "二ヵ所", "何ヶ国"], "reread": [1, 0, 0]}}

    def fake_counter(c, t, arm):
        d = c.pdir(t, "counter")
        (d / f"{arm}.json").write_text(json.dumps(docs[arm], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(G, "ensure_counter", fake_counter)
    rep = G.counter_report(ctx, "N", ["today", "gate"])
    a = rep["arms"]["gate"]
    assert rep["lines"] == 3 and rep["chars"] == 9 and a["reread_lines"] == 1
    assert (a["greedy"]["lines_worse"], a["greedy"]["lines_changed"]) == (1, 1)
    assert a["greedy"]["cer_arm_pct"] == pytest.approx(100 / 9, abs=1e-3) and a["greedy"]["cer_today_pct"] == 0
    assert a["rubi"]["lines_worse"] == 2 and a["rubi_reread"]["lines_worse"] == 1
    assert a["greedy"]["by_cut"]["3"]["lines_worse"] == 1 and a["greedy"]["by_cut"]["0"]["lines_worse"] == 0


def test_dumps_job_waits_for_each_arms_blob(monkeypatch, tmp_path):
    ctx = G.Ctx(tmp_path / "out", _Args(tmp_path / "priv"))
    st = ctx.tdir("N", "steps")
    (ctx.out / "N" / "input.json").write_text("{}")
    order = []

    def wait_for(paths, timeout_s=0, every_s=0):
        order.append(("wait", tuple(Path(q).name for q in paths)))
    monkeypatch.setattr(G, "wait_for", wait_for)
    for f in ("ensure_dumps", "ensure_reads", "ensure_a9", "ensure_counter"):
        monkeypatch.setattr(G, f, lambda c, t, a, f=f: order.append((f, a)))
    G.dumps_job(ctx, "N", ["today", "gate", "2Lg"], extra=True)
    assert order[:6] == [("wait", ("input.json",)), ("wait", ()), ("ensure_dumps", "today"),
                         ("wait", ("gate_fit.json",)), ("ensure_dumps", "gate"),
                         ("wait", ("gate_fit.json", "two_timing_blobs.json"))]
    assert ("ensure_dumps", "2Lg_e4") in order and ("ensure_counter", "2Lg") in order
    assert order.index(("ensure_reads", "gate_e4")) < order.index(("ensure_counter", "today"))
    assert st.exists()


def test_keep_vocr5_records_the_released_verdict(monkeypatch, tmp_path):
    import argparse
    ctx = G.Ctx(tmp_path / "out", _Live(tmp_path / "priv"))
    base = tmp_path / "VOCR5_S_w8.vocr"
    base.write_bytes(b"s-int8")
    rel = tmp_path / "rel"
    rel.mkdir()
    (rel / "VOCR5_S_w8g.vocr").write_bytes(b"s-gated")
    import hashlib
    sha = hashlib.sha256(b"s-int8").hexdigest()
    monkeypatch.setitem(G.REF, "gate_release", rel)
    monkeypatch.setattr(G, "VOCR5_RELEASED", {sha: ("S", "gate", None)})
    monkeypatch.setattr(G, "blob_info", lambda p: {"n_classes": 4987, "has_gate": False, "attention": None})
    repro = tmp_path / "repro" / "S"
    repro.mkdir(parents=True)
    (repro / "decision.json").write_text(json.dumps({
        "final_arm": "gate", "estimate": {"today": {"x": 1}, "final": {"y": 2}}, "counter_check": {"arms": {}},
        "final_blob": {"sha256": hashlib.sha256(b"s-gated").hexdigest()}}))
    a = argparse.Namespace(out=str(ctx.out), tier="S", blob=str(base), repro=str(tmp_path / "repro"), reason="lead")
    G.cmd_keep_vocr5(a)
    dec = json.loads((ctx.out / "S" / "decision.json").read_text())
    assert dec["final_arm"] == "gate" and dec["final_blob"]["name"] == "VOCR5_S_w8g.vocr"
    assert (ctx.out / "final" / "VOCR5_S_w8g.vocr").read_bytes() == b"s-gated"
    assert dec["estimate"] == {"today": {"x": 1}, "final": {"y": 2}} and not (ctx.out / "S" / "prepared.json").exists()
    with pytest.raises(SystemExit):                     # the asset is not the blob step 2 rebuilt
        (rel / "VOCR5_S_w8g.vocr").write_bytes(b"another")
        G.cmd_keep_vocr5(a)
    with pytest.raises(SystemExit):                     # not the released blob of this tier
        G.cmd_keep_vocr5(argparse.Namespace(**dict(vars(a), tier="N")))


def test_confirm_compares_the_version1_part(monkeypatch, tmp_path):
    import argparse
    import hashlib
    ctx = G.Ctx(tmp_path / "out", _Live(tmp_path / "priv"))
    for t, sha in (("S", "s1"), ("N", "n1"), ("L", "l1")):
        ctx.tdir(t)
        (ctx.out / t / "input.json").write_text(json.dumps({"sha256": hashlib.sha256(sha.encode()).hexdigest()}))
    fb = {t: {"path": str(tmp_path / f"{t}.vocr"), "sha256": "-"} for t in G.TIERS}
    names = {"S": "s1", "N": "n2", "L": "l1"}                # N named otherwise
    monkeypatch.setattr(G, "base_sha256", lambda p: hashlib.sha256(names[Path(p).stem].encode()).hexdigest())
    (tmp_path / "fb.json").write_text(json.dumps(fb))
    G.cmd_confirm(argparse.Namespace(out=str(ctx.out), final_blobs=str(tmp_path / "fb.json")))
    doc = json.loads((ctx.out / "confirm.json").read_text())
    assert doc["confirmed"] == ["S", "L"] and doc["redo"] == ["N"] and doc["decide_again"] == ["N", "S"]

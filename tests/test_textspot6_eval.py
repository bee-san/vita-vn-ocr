"""textspot6's evaluation code without data: the selection rule (eval/textspot6_select.py) on hand-made page-path
results, the tagged dialogue CER and the box filters (eval/textspot6.py), and the sweep's plans."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "eval" / "reproducibility" / "textspot6-20261026"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6 as E6  # noqa: E402
import textspot6_select as SEL  # noqa: E402


def result(path, units):
    path.write_text(json.dumps({"units": units}))
    return str(path)


def units(n_clusters, errs, chars=100, platform="vita"):
    """One unit per cluster with the given errors."""
    return [{"uid": f"u{i}", "cluster": f"c{i}", "platform": platform, "errors": e, "chars": chars}
            for i, e in zip(range(n_clusters), errs)]


def test_rule_picks_the_cheapest_qualifier_and_replaces_only_when_resolved(tmp_path):
    n = 30
    base_dev = result(tmp_path / "bd.json", units(n, [10] * n))
    base_hval = result(tmp_path / "bh.json", units(n, [10] * n))
    cands = []
    # a cheap arm, clearly better than the baseline on dev, level on hval
    for seed, e in ((1, 5), (2, 6), (3, 7)):
        cands.append({"arm": "small", "seed": seed, "decode": "crop",
                      "dev": result(tmp_path / f"s{seed}d.json", units(n, [e] * n)),
                      "hval": result(tmp_path / f"s{seed}h.json", units(n, [10] * n)),
                      "vtsp_bytes": 50000, "est_ms_333": 260.0, "arena": 240000, "c_equal": True})
    # a bigger arm, better still on dev and resolved better than the small one
    for seed, e in ((1, 1), (2, 2), (3, 2)):
        cands.append({"arm": "big", "seed": seed, "decode": "crop",
                      "dev": result(tmp_path / f"b{seed}d.json", units(n, [e] * n)),
                      "hval": result(tmp_path / f"b{seed}h.json", units(n, [10] * n)),
                      "vtsp_bytes": 400000, "est_ms_333": 230.0, "arena": 250000, "c_equal": True})
    # an ineligible arm (too slow) that would be best
    cands.append({"arm": "slow", "seed": 1, "decode": "crop", "dev": result(tmp_path / "x.json", units(n, [0] * n)),
                  "hval": result(tmp_path / "xh.json", units(n, [0] * n)), "vtsp_bytes": 10000, "est_ms_333": 400.0,
                  "arena": 1000, "c_equal": True})
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"baseline": {"dev": base_dev, "hval": base_hval}, "seen_dev_clusters": [],
                                 "candidates": cands}))
    out = tmp_path / "sel.json"
    SEL.main(["--table", str(table), "--out", str(out)])
    r = json.loads(out.read_text())
    c = {x["arm"]: x for x in r["candidates"]}
    assert c["small"]["representative_seed"] == 2 and c["big"]["representative_seed"] == 2
    assert c["small"]["qualifies"] and c["big"]["qualifies"] and not c["slow"]["eligible"]
    assert r["chain"][0]["pick"] == "small|crop" and r["pick"]["arm"] == "big"


def test_hval_resolved_worse_disqualifies(tmp_path):
    n = 30
    base = result(tmp_path / "b.json", units(n, [10] * n))
    worse = result(tmp_path / "w.json", units(n, [20] * n))
    good = result(tmp_path / "g.json", units(n, [2] * n))
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"baseline": {"dev": base, "hval": base}, "candidates": [
        {"arm": "a", "seed": 1, "decode": "det", "dev": good, "hval": worse, "vtsp_bytes": 1, "est_ms_333": 1.0,
         "arena": 1, "c_equal": True}]}))
    SEL.main(["--table", str(table), "--out", str(tmp_path / "o.json")])
    r = json.loads((tmp_path / "o.json").read_text())
    assert not r["candidates"][0]["qualifies"] and r["verdict"] == "negative"


def test_rubi_default_needs_a_resolved_win_over_rubi_next(tmp_path):
    n = 30
    base = result(tmp_path / "b.json", units(n, [10] * n))
    pick_dev = result(tmp_path / "p.json", units(n, [4] * n))
    for nxt_e, want in ((8, "spotter"), (4, "grad"), (3, "grad")):
        nxt = result(tmp_path / f"n{nxt_e}.json", units(n, [nxt_e] * n))
        table = tmp_path / "t.json"
        table.write_text(json.dumps({"baseline": {"dev": base, "hval": base},
                                     "rubi_next": {"name": "rubi,fix=1", "dev": nxt, "hval": nxt},
                                     "reports": {"rubi,fix=5": {"dev": nxt, "hval": nxt}}, "candidates": [
            {"arm": "a", "seed": 1, "decode": "crop", "dev": pick_dev, "hval": pick_dev, "vtsp_bytes": 1,
             "est_ms_333": 1.0, "arena": 1, "c_equal": True}]}))
        SEL.main(["--table", str(table), "--out", str(tmp_path / "o.json")])
        r = json.loads((tmp_path / "o.json").read_text())
        assert r["verdict"] == "positive" and r["rubi_default"]["finder"] == want, (nxt_e, r["rubi_default"])
        assert r["reports"]["rubi,fix=5"]["dev_cer"] == round(100 * nxt_e / 100, 4)
        assert r["best_eligible_if_negative"] is None


def test_negative_verdict_names_the_best_eligible(tmp_path):
    n = 30
    base = result(tmp_path / "b.json", units(n, [10] * n))
    lvl = result(tmp_path / "l.json", units(n, [10] * n))
    lo = result(tmp_path / "lo.json", units(n, [9] * 15 + [11] * 15))
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"baseline": {"dev": base, "hval": base}, "candidates": [
        {"arm": "a", "seed": 1, "decode": "det", "dev": lvl, "hval": lvl, "vtsp_bytes": 1, "est_ms_333": 1.0,
         "arena": 1, "c_equal": True},
        {"arm": "b", "seed": 1, "decode": "det", "dev": lo, "hval": lvl, "vtsp_bytes": 2, "est_ms_333": 1.0,
         "arena": 1, "c_equal": False}]}))
    SEL.main(["--table", str(table), "--out", str(tmp_path / "o.json")])
    r = json.loads((tmp_path / "o.json").read_text())
    assert r["verdict"] == "negative" and r["rubi_default"] is None
    assert r["best_eligible_if_negative"]["arm"] == "a"         # b is not eligible (C differs from the twin)


def test_seen_clusters_are_left_out(tmp_path):
    u = units(3, [0, 0, 50])
    p = result(tmp_path / "u.json", u)
    assert SEL.cer(SEL.load_rows(p)) == round(100 * 50 / 300, 4)
    assert SEL.cer(SEL.load_rows(p, ("c2",))) == 0.0


def test_tagged_dialogue_counts_other_texts_read_as_dialogue():
    refs = [{"box": [10, 10, 200, 20], "cls": "dialogue", "vertical": False, "text": "あいう", "ignore": False},
            {"box": [10, 100, 80, 20], "cls": "ui", "vertical": False, "text": "メニュー", "ignore": False}]
    dets = [{"box": [8, 8, 204, 24], "vertical": False, "text": "あいう", "tag": 0},
            {"box": [8, 98, 84, 24], "vertical": False, "text": "メニュー", "tag": 0},     # UI read as dialogue
            {"box": [300, 300, 50, 20], "vertical": False, "text": "xx", "tag": 3}]      # tagged UI: not read
    e, n = E6.tagged_dialogue(dets, refs, 960, 544)
    assert n == 3 and e == 4
    dets[1]["tag"] = 3
    assert E6.tagged_dialogue(dets, refs, 960, 544) == (0, 3)


def test_box_filters():
    b = [1, 2, 3, 4, 0, 50, 1, 0, 2, 7]
    assert E6.keep_box(b, "all") and not E6.keep_box(b, "dialogue")
    assert not E6.keep_box([1, 2, 3, 4, 1, 50, 1, 0, 0, 0], "all")
    r = {"boxes": [b], "rboxes": [[1, 2, 3, 4, 0, 50, 1, 2]]}
    assert E6.decoded(r, "det") == [b] and E6.decoded(r, "detref") == [[1, 2, 3, 4, 0, 50, 1, -1, 2, 0]]


def test_sweep_plans_and_cap(tmp_path):
    import sweep as SW
    p1 = SW.plan_stage1()
    assert len(p1) == 8 and {r["seed"] for r in p1} == {1, 2}
    p1b = SW.plan_stage1b()
    assert len(p1b) == 11 and "base_t64n8c16_seed1" not in {r["run"] for r in p1b}
    cap = tmp_path / "cap.txt"
    assert SW.read_cap(cap, 3) == 3
    cap.write_text("2\n")
    assert SW.read_cap(cap, 3) == 2
    cmd = SW.command({"recipe": "base", "shape": "qS4", "seed": 1}, tmp_path / "run")
    assert "--hook-splits" in cmd and cmd[cmd.index("--hook-splits") + 1] == "train"

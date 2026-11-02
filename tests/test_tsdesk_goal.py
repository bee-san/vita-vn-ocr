"""The round-2 goal rule (eval/tsdesk_goal.py) on synthetic reads."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import tsdesk_goal as G  # noqa: E402


def _doc(errs, view_of, set_="dev", junk=None):
    units = []
    for i, e in enumerate(errs):
        v = view_of(i)
        units.append({"uid": f"u{i}", "set": set_, "view": v, "group": v, "cluster": f"c{i % 20}", "errors": e,
                      "chars": 50, "junk": (junk or {}).get(i, 0)})
    return {"units": units}


def _write(tmp, name, doc):
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(doc))
    return str(p)


def test_goal_passes_only_when_every_clause_holds(tmp_path):
    view = lambda i: "vita" if i % 2 else "psp"  # noqa: E731
    r = [1 if i % 3 == 0 else 0 for i in range(80)]
    better = [0 if i % 6 == 0 else e for i, e in enumerate(r)]
    worse = [e + 1 for e in r]
    t8 = [e + 2 for e in r]
    reads = {}
    for s in G.G1_SETS:
        reads[s] = {"cand": _write(tmp_path, f"{s}_c", _doc(better, view, s)),
                    "R": _write(tmp_path, f"{s}_r", _doc(r, view, s))}
        if s in G.VIEW_SETS:
            reads[s]["T8"] = _write(tmp_path, f"{s}_t", _doc(t8, view, s))
    hv = {"cand": _write(tmp_path, "h_c", _doc(r, view, "hval")), "R": _write(tmp_path, "h_r", _doc(r, view, "hval"))}
    reads["hval"] = hv
    timing = {"det_1t": {"cand": {"vita": 15.0}, "meiki": {"vita": 109.5}},
              "page_4t": {"cand": {"vita": 20, "psp": 20, "pc": 22}, "R": {"vita": 74, "psp": 71, "pc": 77}},
              "interleaved": True}
    spec = {"candidate": "x", "policy": "long:960", "reads": reads, "timing": _write(tmp_path, "t", timing)}
    out = G.check(spec)
    assert out["met_on_dev"] and out["gap"] < 0
    # a worse dense_dev fails G1; a slow detector fails G2; a junk box fails G4; a non-interleaved timing fails G2/G3
    bad = json.loads(json.dumps(spec))
    bad["reads"]["dense_dev"]["cand"] = _write(tmp_path, "dd_w", _doc(worse, view, "dense_dev"))
    o = G.check(bad)
    assert not o["G1"]["pass"] and not o["met_on_dev"] and o["gap"] > 0
    slow = dict(timing, det_1t={"cand": {"vita": 23.0}, "meiki": {"vita": 109.5}})
    assert not G.check(dict(spec, timing=_write(tmp_path, "t2", slow)))["G2"]["pass"]
    assert not G.check(dict(spec, timing=_write(tmp_path, "t3", dict(timing, interleaved=False))))["G3"]["pass"]
    j = json.loads(json.dumps(spec))
    j["reads"]["hval"]["cand"] = _write(tmp_path, "h_j", _doc(r, lambda i: "vita", "hval", junk={0: 1}))
    for u in json.loads(Path(j["reads"]["hval"]["cand"]).read_text())["units"][:1]:
        assert u["junk"] == 1
    doc = json.loads(Path(j["reads"]["hval"]["cand"]).read_text())
    doc["units"][0]["cluster"] = G.GUARD_CLUSTER
    Path(j["reads"]["hval"]["cand"]).write_text(json.dumps(doc))
    assert not G.check(j)["G4"]["pass"]
    # a missing read is unread, never a pass
    m = json.loads(json.dumps(spec))
    del m["reads"]["dev_l544"]
    o = G.check(m)
    assert o["G1"]["dev_l544"]["state"] == "unread" and not o["met_on_dev"] and o["gap"] is None


def test_views_of_reads_own_size_and_fit_views():
    cmp = {"dev:all": 1, "dev:vita": 2, "dev:psp": 3, "dev:psp-hires": 4, "dev:vita-pc": 5}
    assert G.views_of(cmp) == {"vita": 2, "psp": 3}
    cmp = {"dev:all": 1, "dev:vita@fit960x544": 2, "dev:psp@fit960x544": 3, "dev:psp-hires": 4}
    assert G.views_of(cmp) == {"vita": 2, "psp": 3}

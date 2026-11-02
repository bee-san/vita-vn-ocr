"""The desktop decision and its public records (issue #169, docs/desktop.md): eval/desktop_rule.py applies the
registered rule to result records, eval/desktop_record.py writes counts only. Synthetic records, no private data."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import desktop_record as REC  # noqa: E402
import desktop_rule as RULE  # noqa: E402

VNS = [f"v{i}" for i in range(12)]


def doc(set_name, errors, chars=100, views=("vita", "psp"), frames="f" * 64):
    """A result record of eval/desktop.py's shape: one unit per (VN, view), `errors(vn, view)` errors each."""
    units = []
    for vn in VNS:
        for view in views:
            units.append({"uid": f"{set_name}:{vn}:{view}", "cluster": vn, "set": set_name, "view": view,
                          "group": view, "errors": errors(vn, view), "chars": chars, "texts": ["秘密"],
                          "boxes_xywh": [[1, 2, 3, 4]]})
    e = sum(u["errors"] for u in units)
    c = sum(u["chars"] for u in units)
    s = {"units": len(units), "errors": e, "chars": c, "cer": round(100 * e / c, 4), "line_recall": 1.0,
         "box_precision": 1.0}
    return {"frames_sha256": frames, "det": "x", "pad": None, "rec": "VOCR6.2_L_fast.onnx", "units": units,
            "summary": {"all": s, "by_view": {f"{set_name}:{v}": s for v in views},
                        "by_group": {f"{set_name}:{v}": s for v in views}}}


def write(d, name, obj):
    (d / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    """R reads 1 error per unit everywhere (CER 1%). At fit:960, seeds 1-3 read dev-v2 at 1.25 / 1.083 / 1.083% (seeds
    2 and 3 tie, as the real seeds did); seed 3 is the median. Seed 3: long:640 qualifies on every clause, fit:960 and
    fit:640 are too slow, fit:480 reads dev-v2 at 1.5 R, and long:960 has no confirm-v2 record (undecided)."""
    res = tmp_path / "res"
    res.mkdir()
    one = lambda vn, view: 1  # noqa: E731
    for s in ("dev", "hval", "confirm", "pctest"):
        write(res, f"{s}_R", doc(s, one, views=("pc",) if s == "pctest" else ("vita", "psp")))
    # dev-v2 CER at fit:960: seed 1 1.25%, seeds 2 and 3 1.083% (a tie the seed number breaks)
    write(res, "dev_s1_fit960", doc("dev", lambda vn, view: 1 + (vn in VNS[:3])))
    for s in (2, 3):
        write(res, f"dev_s{s}_fit960", doc("dev", lambda vn, view: 1 + (vn in VNS[:2] and view == "vita"),
                                           chars=100))
    good = doc("dev", lambda vn, view: 1 + (vn == "v0"))           # +2 errors of 2,400 chars: within 1.2 R
    write(res, "dev_s3_long640", good)
    write(res, "dev_s3_fit640", good)
    write(res, "dev_s3_long960", good)
    write(res, "dev_s3_fit480", doc("dev", lambda vn, view: 1 + (vn in VNS[:6])))   # 1.5 R on Vita and PSP
    for pol in ("fit960", "long640", "fit640", "fit480", "long960"):
        write(res, f"hval_s3_{pol}", doc("hval", one))
        write(res, f"pctest_s3_{pol}", doc("pctest", one, views=("pc",)))
        if pol != "long960":
            write(res, f"confirm_s3_{pol}", doc("confirm", one))
    timing = {"threads": 4, "results": {
        "R": {"vita": {"p50": 80.0}, "psp": {"p50": 76.0}},
        "C640": {"vita": {"p50": 19.0}, "psp": {"p50": 18.0}},
        "C640#d1": {"vita": {"p50": 21.0}, "psp": {"p50": 17.5}},
        "Cfit640": {"vita": {"p50": 21.0}, "psp": {"p50": 15.0}},
        "C480": {"vita": {"p50": 12.0}, "psp": {"p50": 12.0}},
        "C960": {"vita": {"p50": 15.0}, "psp": {"p50": 15.0}},
        "Cfit960": {"vita": {"p50": 30.0}, "psp": {"p50": 14.0}}}}
    (tmp_path / "t4.json").write_text(json.dumps(timing), encoding="utf-8")

    def entry(seed, pol, t):
        return {"dev": f"dev_s{seed}_{pol}", "hval": f"hval_s{seed}_{pol}", "confirm": f"confirm_s{seed}_{pol}",
                "pc": f"pctest_s{seed}_{pol}", "timing": t}
    cand = {"reference": {"dev": "dev_R", "hval": "hval_R", "confirm": "confirm_R", "pc": "pctest_R", "timing": "R"},
            "seeds": {"1": {"fit:960": entry(1, "fit960", [])}, "2": {"fit:960": entry(2, "fit960", [])},
                      "3": {"fit:960": entry(3, "fit960", ["Cfit960"]),
                            "long:640": entry(3, "long640", ["C640", "C640#d1"]),
                            "fit:640": entry(3, "fit640", ["Cfit640"]), "fit:480": entry(3, "fit480", ["C480"]),
                            "long:960": entry(3, "long960", ["C960"])}}}
    (tmp_path / "cand.json").write_text(json.dumps(cand), encoding="utf-8")
    return tmp_path


def run_rule(world, out="decision.json"):
    RULE.main(["--res", str(world / "res"), "--timing", str(world / "t4.json"), "--candidates",
               str(world / "cand.json"), "--out", str(world / out)])
    return json.loads((world / out).read_text(encoding="utf-8"))


def test_rule_clauses_representative_and_pick(world):
    d = run_rule(world)
    assert d["seeds_dev_cer_fit960"] == {"1": 1.25, "2": 1.0833, "3": 1.0833}
    assert d["representative"] == "3"                     # sorted by (CER, seed): 2, 3, 1; the median is seed 3
    r = d["results"]
    assert r["long:640"]["qualifies"] is True
    assert r["long:640"]["speed"]["views"]["psp"]["p50"] == 17.5      # the fastest of the detector's variants
    assert r["fit:640"]["speed"]["views"]["vita"]["ok"] is False      # 21.0 > 80 / 4
    assert r["fit:640"]["qualifies"] is False
    assert r["fit:960"]["dev"]["ok"] is True and r["fit:960"]["qualifies"] is False   # 30 ms > 80 / 4 on Vita
    assert r["fit:480"]["dev"]["ok"] is False                          # 1.5 R
    assert r["fit:480"]["speed"]["ok"] is True
    assert r["long:960"]["confirm"]["ok"] is None and r["long:960"]["qualifies"] is None
    assert d["qualifying"] == ["long:640"] and d["pick"] == "long:640"
    assert d["undecided"] == ["long:960"] and d["verdict"] == "pick"


def test_rule_needs_the_four_thread_run_and_an_odd_number_of_seeds(world):
    t = json.loads((world / "t4.json").read_text(encoding="utf-8"))
    t["threads"] = 8
    (world / "t8.json").write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(SystemExit):
        RULE.main(["--res", str(world / "res"), "--timing", str(world / "t8.json"), "--candidates",
                   str(world / "cand.json"), "--out", str(world / "x.json")])
    c = json.loads((world / "cand.json").read_text(encoding="utf-8"))
    del c["seeds"]["1"]
    (world / "cand.json").write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_rule(world)


def test_record_holds_counts_only_and_refuses_other_frames(world, monkeypatch):
    res = world / "res"
    monkeypatch.setattr(REC.D, "git_head", lambda: "0" * 40)
    out = world / "rec.json"
    REC.main(["--res", str(res), "--set", "dev", "--ref", "R", "--names", "s3_long640,s3_fit480", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    rec = json.loads(text)
    assert "秘密" not in text and "boxes_xywh" not in text and "texts" not in text
    assert set(rec["rows"]) == {"R", "s3_long640", "s3_fit480"}
    ag = rec["rows"]["s3_fit480"]["against_reference"]["dev:all"]
    assert ag["delta"] == pytest.approx(0.5) and ag["clusters"] == 12  # 12 more errors in 2,400 characters
    assert "against_reference" not in rec["rows"]["R"]
    write(res, "dev_other", doc("dev", lambda vn, view: 1, frames="0" * 64))
    with pytest.raises(SystemExit):
        REC.main(["--res", str(res), "--set", "dev", "--ref", "R", "--names", "other", "--out", str(out)])

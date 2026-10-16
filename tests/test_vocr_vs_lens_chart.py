"""docs/charts/vocr-vs-lens.json, the README chart's numbers: every value is recomputed from the committed score files
it cites (their sha256 must match), and the headline's claims hold on those numbers.

No matplotlib, no font, no network: it checks the data the committed SVG/PNGs were drawn from.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "docs" / "charts"
DATA = CHARTS / "vocr-vs-lens.json"

spec = importlib.util.spec_from_file_location("vocr_vs_lens", CHARTS / "vocr_vs_lens.py")
chart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chart)

needs_data = pytest.mark.skipif(not DATA.exists(), reason="no chart data committed")


def data():
    return json.loads(DATA.read_text())


def source(d, name):
    return json.loads((ROOT / d["sources"][name]["file"]).read_text())


@needs_data
def test_sources_are_the_committed_files():
    d = data()
    assert set(d["sources"]) == {"clannad", "dialogue"}
    for name, src in d["sources"].items():
        path = ROOT / src["file"]
        assert path.is_file(), f"{name}: {src['file']} is not in the repo"
        assert chart.sha256(path) == src["sha256"], f"{name}: {src['file']} changed since the chart was drawn"


@needs_data
def test_every_value_recomputes_from_its_source():
    d = data()
    arms = [s["key"] for s in d["systems"] if s["key"] != "Lens"]
    cl = source(d, "clannad")["scores"]["systems"]
    dl = source(d, "dialogue")
    assert [s["key"] for s in d["sets"]] == ["clannad", "vita-test", "psp-test"]
    for s in d["sets"]:
        vals = d["values"][s["key"]]
        assert set(vals) == {*arms, "Lens"}
        if s["key"] == "clannad":
            for k in [*arms, "Lens"]:
                v = cl[k]["not_name"]
                assert (vals[k]["errors"], vals[k]["chars"], vals[k]["lines"]) == (v["errors"], v["chars"], v["lines"])
                assert vals[k]["cer"] == v["cer"] and vals[k]["cer_ci95"] == v["cer_ci95"]
                assert abs(v["cer"] - 100 * v["errors"] / v["chars"]) < 5e-4
        else:
            for k in [*arms, "Lens"]:
                want = chart.dialogue_value(dl, k, s["key"])
                assert {x: y for x, y in vals[k].items() if x != "vs"} == want
            for k in arms:
                assert vals[k]["vs"]["Lens"] == chart.dialogue_pair(dl, k, s["key"])
        assert s["lines"] == vals["Lens"]["lines"] == vals[arms[0]]["lines"], "the label's count is the scored lines"
    for set_name in ("vita-test", "psp-test"):
        f = d["facts"][f"lens_furigana_{set_name}"]
        lens = dl["arms"]["native"]["Lens"]
        assert f["furigana_errors"] == lens["styles"]["furigana"][set_name]["errors"]
        assert f["furigana_lines"] == lens["styles"]["furigana"][set_name]["lines"]
        assert f["all_errors"] == lens["all"][set_name]["errors"]


@needs_data
def test_the_headline_claims_hold():
    d = data()
    assert d["copy"].get("headline"), "no headline"
    assert d["copy"].get("claims"), "the headline's comparison must be written down as claims"
    assert chart.check(d) == []


@needs_data
def test_the_words_say_what_is_measured():
    d = data()
    words = " ".join([d["copy"].get("subhead", ""), *d["copy"].get("footnote", [])])
    assert "lower is better" in words.lower()
    for s in d["sets"]:
        assert s["label"].lower() in words.lower(), f"the footnote says what {s['label']} is"
        assert f"{s['lines']:,}" in words, f"the footnote gives {s['label']}'s line count"
    assert "furigana" in words.lower(), "Lens's furigana handicap on vita-test/psp-test is stated"
    assert "estimate" in " ".join(d["copy"]["panel"].get("lines", [])).lower(), "console times are labelled estimates"


@needs_data
def test_the_rendered_files_are_committed():
    for theme in ("light", "dark"):
        for ext in (".svg", "-2x.png"):
            f = CHARTS / f"vocr-vs-lens-{theme}{ext}"
            assert f.is_file() and f.stat().st_size < 1 << 20, f


def test_claim_checker_rejects_a_false_claim():
    d = {"values": {"x": {"A": {"cer": 1.0}, "Lens": {"cer": 0.5}}},
         "copy": {"claims": [{"set": "x", "system": "A", "op": "<", "than": "Lens"}]}}
    assert chart.check(d)
    d["copy"]["claims"][0]["op"] = ">"
    assert chart.check(d) == []
    d["copy"]["claims"][0].update(op="<", than=2.0, ci=False)
    assert chart.check(d) == []
    d["copy"]["claims"] = [{"set": "x", "system": "A", "op": "<", "than": "Lens", "ci": True}]
    d["values"]["x"]["Lens"]["cer"] = 2.0
    assert chart.check(d), "a ci claim needs the pair interval"
    d["values"]["x"]["A"]["vs"] = {"Lens": {"ci95": [-1.5, -0.2]}}
    assert chart.check(d) == []
    d["values"]["x"]["A"]["vs"]["Lens"]["ci95"] = [-1.5, 0.3]
    assert chart.check(d)
    d["values"]["y"] = {"A": {"cer": 1.0}, "Lens": None}
    d["copy"]["claims"] = [{"set": "y", "system": "A", "op": "<", "than": "Lens"}]
    assert chart.check(d) == ["{'set': 'y', 'system': 'A', 'op': '<', 'than': 'Lens'}: not measured"]

"""docs/charts/vocr-vs-lens.json, the README chart's numbers: every value is recomputed from the committed score files
it cites (their sha256 must match), the headline's claims hold on those numbers, the panel's and the footnote's numbers
come from the data, the committed images are the ones drawn from this JSON, and the README's chart text and model
table agree with it.

No matplotlib, no font, no network.
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest

NUMBER = re.compile(r"(?<![A-Za-z0-9-])\d[\d,.]*(?![A-Za-z0-9])")   # a standalone number (not int8, A9, Vita3K)


def typed_numbers(text):
    """Numbers typed into a template, after removing its {PLACEHOLDERS}."""
    return NUMBER.findall(re.sub(r"\{[A-Z0-9_]+\}", "", text))

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "docs" / "charts"
DATA = CHARTS / "vocr-vs-lens.json"
IMAGES = ("vocr-vs-lens-light.svg", "vocr-vs-lens-dark.svg", "vocr-vs-lens-light-2x.png", "vocr-vs-lens-dark-2x.png")

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
    assert set(d["sources"]) == {"clannad", "dialogue", "report", "protocol"}
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
                assert vals[k]["exact_lines"] == v["exact_lines"]
                assert vals[k]["exact_pct"] == round(100 * v["exact_lines"] / v["lines"], 2)
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
        g = dl["pairs"][f"{arms[0]}-Lens"]["console_vs_lens_native"]["dialogue"][set_name]["vn_clusters"]
        assert d["facts"][f"games_{set_name}"]["games"] == g


@needs_data
def test_the_panel_numbers_come_from_the_report_and_protocol():
    d = data()
    assert d["resources"] == chart.resources(ROOT / d["sources"]["report"]["file"],
                                             ROOT / d["sources"]["protocol"]["file"])
    p = chart.panel_text(d)
    r = d["resources"]
    assert p["big"] == chart.mb(r["VOCR5_N"]["bytes"]) == "1.2 MB"
    words = " ".join(p["lines"])
    assert chart.mb(r["VOCR5_S"]["bytes"]) in words and chart.mb(r["VOCR5_L"]["bytes"]) in words
    assert f"{r['VOCR5_N']['box_s_444']:.1f}-{r['VOCR5_N']['box_s_333']:.1f} s" in words
    assert "estimate" in words.lower(), "console times are labelled estimates"
    raw = json.dumps(d["copy"]["panel"], ensure_ascii=False)
    assert typed_numbers(raw) == [], "the panel's numbers are filled in, not typed"


@needs_data
def test_the_footnote_numbers_come_from_the_data():
    d = data()
    text = " ".join(chart.footnote_text(d))
    raw = " ".join(d["copy"]["footnote"])
    # only the estimate's conditions may be typed (a 3-line box, the two clocks); every result is filled in
    assert sorted(typed_numbers(raw)) == ["3", "333", "444"], "the footnote's numbers are filled in, not typed"
    assert "lower is better" in d["copy"]["subhead"].lower()
    for s in d["sets"]:
        assert s["label"].lower() in text.lower(), f"the footnote says what {s['label']} is"
        assert f"{s['lines']:,}" in text, f"the footnote gives {s['label']}'s line count"
    f = d["facts"]
    for set_name in ("psp-test", "vita-test"):
        fur = f[f"lens_furigana_{set_name}"]
        assert f"{fur['furigana_errors']} of its {fur['all_errors']} {set_name}" in text, set_name
    assert "furigana" in text and "estimate" in text and "not run on a console" in text
    assert "planetarian" in text, "the CLANNAD exposure is disclosed"


@needs_data
def test_the_headline_claims_hold():
    d = data()
    assert d["copy"].get("headline"), "no headline"
    assert d["copy"].get("claims"), "the headline's comparison must be written down as claims"
    assert chart.check(d) == []


@needs_data
def test_the_committed_images_are_drawn_from_this_json():
    d = data()
    assert d["rendered"]["inputs_sha256"] == chart.inputs_sha256(d), "the JSON changed since the images were drawn"
    assert set(d["rendered"]["files"]) == set(IMAGES)
    for name in IMAGES:
        f = CHARTS / name
        assert f.is_file() and f.stat().st_size < 1 << 20, f
        assert chart.sha256(f) == d["rendered"]["files"][name], f"{name} is not the image drawn from this JSON"


@needs_data
def test_the_readme_agrees_with_the_chart():
    d = data()
    readme = (ROOT / "README.md").read_text()
    V, r = d["values"], d["resources"]
    alt = re.search(r'<img src="docs/charts/vocr-vs-lens-light\.svg"[^>]*alt="([^"]+)"', readme)
    assert alt, "the README shows the light chart with alt text"
    assert 'srcset="docs/charts/vocr-vs-lens-dark.svg"' in readme
    for s in d["sets"]:
        assert f"{s['lines']:,} lines" in alt.group(1)
    # the alt text gives, set by set, every bar's value in bar order
    bars = [f"{V[s['key']][x['key']]['cer']:.2f}%" for s in d["sets"] for x in d["systems"]]
    assert re.findall(r"\d+\.\d\d%", alt.group(1)) == bars, "the alt text's values are the chart's, in order"
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| \[`(VOCR5_[SNL])_w8\.vocr`\].*$", readme, re.M)}
    assert set(rows) == {"VOCR5_S", "VOCR5_N", "VOCR5_L"}
    for arm, row in rows.items():
        triple = " / ".join(f"{V[s][arm]['cer']:.2f}%" for s in ("clannad", "vita-test", "psp-test"))
        assert triple in row, arm
        assert f"{r[arm]['bytes']:,} B" in row and f"{r[arm]['arena_w8i']:,} B" in row, arm
        assert f"{r[arm]['box_s_333']:.2f} / {r[arm]['box_s_444']:.2f} s" in row, arm
    lens = re.search(r"^\| Google Lens \|.*$", readme, re.M).group(0)
    want = (f"{V['clannad']['Lens']['cer']:.2f}% / {V['vita-test']['Lens']['cer']:.2f}%\\* / "
            f"{V['psp-test']['Lens']['cer']:.2f}%\\*")
    assert want in lens, (want, lens)


# The chart's words, frozen: a change to the headline, subhead, panel, footnote or marks must be re-read for
# truthfulness (README "How to read the chart") and this sha256 updated in the same PR.
WORDS_SHA256 = "9f77f93e43fbb0812ecbce662f78a5aeb74439e9f3677ad6ff9459d03e3015b6"


@needs_data
def test_the_words_are_the_reviewed_ones():
    c = data()["copy"]
    words = {k: c[k] for k in ("eyebrow", "headline", "subhead", "panel", "footnote", "marks")}
    got = chart.hashlib.sha256(json.dumps(words, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert got == WORDS_SHA256, "the chart's words changed: review them against the data, then update WORDS_SHA256"


@needs_data
def test_every_set_and_tier_has_a_claim_and_the_test_sets_need_intervals():
    d = data()
    arms = [s["key"] for s in d["systems"] if s["key"] != "Lens"]
    claims = {(c["set"], c["system"]): c for c in d["copy"]["claims"] if c["than"] == "Lens" and c["op"] == "<"}
    assert set(claims) == {(s["key"], a) for s in d["sets"] for a in arms}, "one 'below Lens' claim per set and tier"
    for set_name in ("vita-test", "psp-test"):
        for a in arms:
            assert claims[(set_name, a)].get("ci") is True, f"{set_name} {a}: the claim must need its interval"
            pair = source(d, "dialogue")["pairs"][f"{a}-Lens"]["console_vs_lens_native"]["dialogue"][set_name]
            assert pair["lines_ci95"][1] < 0 and pair["vn_ci95"][1] < 0, "straight from the source: below 0"
    assert all(not c.get("ci") for (s_, _), c in claims.items() if s_ == "clannad"), \
        "CLANNAD is level within the noise: no interval claim there"


@needs_data
def test_the_svgs_say_what_the_json_says():
    d = data()
    V = d["values"]
    p = chart.panel_text(d)
    want = [d["copy"]["eyebrow"], d["copy"]["headline"], d["copy"]["subhead"]]
    for s in d["sets"]:
        for x in d["systems"]:
            mark = d["copy"].get("marks", {}).get(s["key"], {}).get(x["key"], "")
            label = "Lens" if x["key"] == "Lens" else x["label"].split()[-1]
            want += [label, chart.fmt_pct(V[s["key"]][x["key"]]["cer"]) + mark]
        want += [s["label"], f"{s['platform']} · {s['lines']:,} lines"]
    want += [p["kicker"], p["big"], p["big_sub"], *p["lines"], *p["lens_lines"]]
    foot = " ".join(chart.footnote_text(d)).split()
    for name in ("vocr-vs-lens-light.svg", "vocr-vs-lens-dark.svg"):
        texts = re.findall(r"<!-- (.*?) -->", (CHARTS / name).read_text(encoding="utf-8"), re.S)
        n = len(want)
        assert texts[:n] == want, name
        assert " ".join(texts[n:]).split() == foot, f"{name}: the footnote drawn is the footnote filled from the data"


@needs_data
def test_the_readme_notes_agree_with_the_sources():
    d = data()
    readme = (ROOT / "README.md").read_text()
    notes = readme[readme.index("How to read the chart:"):readme.index("The chart is drawn by")]
    V = d["values"]
    cl = source(d, "clannad")["scores"]
    dl = source(d, "dialogue")
    rows = {s["key"]: V[s["key"]] for s in d["sets"]}
    assert f"below Lens's {rows['clannad']['Lens']['cer']:.2f}%" in notes
    lo, hi = rows["clannad"]["Lens"]["cer_ci95"]
    assert f"({lo:.2f} to {hi:.2f})" in notes
    exact = rows["clannad"]
    assert f"{exact['Lens']['exact_pct']:.1f}% against {exact['VOCR5_L']['exact_pct']:.1f}% for Large" in notes
    p = cl["pairs"]["VOCR5_L:Lens"]
    assert f"Large minus Lens is −{abs(p['cer_delta']):.2f} points" in notes
    assert f"(paired 95% interval −{abs(p['cer_ci95'][0]):.2f} to +{p['cer_ci95'][1]:.2f})" in notes
    names = {k: cl["systems"][k]["name"]["cer"] for k in ("Lens", "VOCR5_S", "VOCR5_N", "VOCR5_L")}
    ours = sorted(names[k] for k in ("VOCR5_S", "VOCR5_N", "VOCR5_L"))
    assert f"({names['Lens']:.2f}% against {ours[0]:.2f}-{ours[-1]:.2f}%)" in notes
    for set_name in ("vita-test", "psp-test"):
        nm = {k: dl["arms"]["native" if k == "Lens" else "console"][k]["names"][set_name] for k in names}
        rate = {k: 100 * v["errors"] / v["chars"] for k, v in nm.items()}
        ours = sorted(rate[k] for k in ("VOCR5_S", "VOCR5_N", "VOCR5_L"))
        assert f"({rate['Lens']:.2f}% against {ours[0]:.2f}-{ours[-1]:.2f}%)" in notes, set_name
    f = d["facts"]
    for set_name in ("psp-test", "vita-test"):
        assert str(f[f"lens_furigana_{set_name}"]["furigana_errors"]) in notes
    lv = json.loads((ROOT / "eval/vocr5_clannad_lens_view_20261016.json").read_text())["view"]["dialogue"]["systems"]
    assert " / ".join(f"{lv[k]['cer_vs_lens']:.2f}%" for k in ("VOCR5_S", "VOCR5_N", "VOCR5_L")) in notes
    lm = " / ".join(f"{dl['arms']['console'][k + '+LM']['dialogue']['pooled']['cer']:.2f}%"
                    for k in ("VOCR5_S", "VOCR5_N", "VOCR5_L"))
    assert lm in readme, "the README gives the LM-decoded figures"


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

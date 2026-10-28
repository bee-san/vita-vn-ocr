"""docs/charts/vocr-vs-lens.json, the README chart's numbers: every value is recomputed from the committed records it
cites (their sha256 must match), the headline's claims hold on those numbers, the row notes' and the footnote's numbers
come from the data, the committed images are the ones drawn from this JSON, and the README's chart text, tables and
notes agree with it.

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
VOCR6 = ("VOCR6_S", "VOCR6_N", "VOCR6_L")
VOCR5 = ("VOCR5_S", "VOCR5_N", "VOCR5_L")
OTHERS = ("Lens", "MeikiOCR", "PPOCRv6_small", "PPOCRv5_mobile_int8", "manga_ocr")
SET_KEYS = ("clannad", "vita-test", "psp-test", "pcsg00807")

spec = importlib.util.spec_from_file_location("vocr_vs_lens", CHARTS / "vocr_vs_lens.py")
chart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chart)

needs_data = pytest.mark.skipif(not DATA.exists(), reason="no chart data committed")


def data():
    return json.loads(DATA.read_text(encoding="utf-8"))


def source(d, name):
    return json.loads((ROOT / d["sources"][name]["file"]).read_text(encoding="utf-8"))


def readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def section(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


@needs_data
def test_sources_are_the_committed_files():
    d = data()
    assert set(d["sources"]) == {"readers", "vocr6", "followup", "vocr5_report", "vocr5_protocol", "lens_dialogue"}
    for name, src in d["sources"].items():
        path = ROOT / src["file"]
        assert path.is_file(), f"{name}: {src['file']} is not in the repo"
        assert chart.sha256(path) == src["sha256"], f"{name}: {src['file']} changed since the chart was drawn"


@needs_data
def test_the_systems_and_sets_are_the_charts():
    d = data()
    assert [(s["key"], s["label"], s["family"]) for s in d["systems"]] == [tuple(x) for x in chart.SYSTEMS]
    assert [s["key"] for s in d["sets"]] == list(SET_KEYS)
    assert {s["key"] for s in d["systems"]} == {*VOCR6, *VOCR5, *OTHERS}


@needs_data
def test_every_value_recomputes_from_the_readers_record():
    d = data()
    rec = source(d, "readers")
    for s in d["sets"]:
        t = rec["sets"][s["key"]]
        assert (s["lines"], s["chars"], s["clusters"], s["cluster"]) == (t["units"], t["chars"], t["clusters"],
                                                                          t["cluster"])
        for k in [x["key"] for x in d["systems"]]:
            v, want = d["values"][s["key"]][k], t["values"].get(k)
            if want is None:
                assert v is None and k == "Lens" and s["key"] == "pcsg00807", (s["key"], k)
                continue
            assert {x: y for x, y in v.items() if x != "vs"} == chart.value_of(want, f"sets.{s['key']}.values.{k}")
            assert abs(v["cer"] - 100 * v["errors"] / v["chars"]) < 5e-4
            assert v["lines"] == s["lines"] and v["chars"] == s["chars"], "every system read the same lines"
            if k in VOCR6 + VOCR5:
                assert set(v["vs"]) == {o for o in OTHERS if t["values"].get(o) is not None}
                for o, p in v["vs"].items():
                    assert p == chart.pair_of(t["pairs"][f"{k}-{o}"], f"sets.{s['key']}.pairs.{k}-{o}")


@needs_data
def test_the_readers_record_reproduces_the_committed_vocr_lens_and_meiki_numbers():
    """The other engines' scorer recounted VOCR6's, VOCR5's, Lens's and MeikiOCR's committed numbers from their reads'
    private records before scoring anything new; here the chart's VOCR numbers are checked against those committed
    records directly."""
    d = data()
    rec = source(d, "readers")
    assert rec["checks"]["compared"] == rec["checks"]["equal"] == 28
    fu = source(d, "followup")["path_a"]
    lens = source(d, "lens_dialogue")["arms"]["native"]["Lens"]["dialogue"]
    for k in VOCR6 + VOCR5:
        fam, tier = k.split("_")
        key = f"{fam}:{tier}:greedy"
        want = {"clannad": fu["clannad_lines"]["dialogue"]["rates"][key],
                "vita-test": fu["vita_psp_fixed_lines"]["vita-test|dialogue"]["rates"][key],
                "psp-test": fu["vita_psp_fixed_lines"]["psp-test|dialogue"]["rates"][key],
                "pcsg00807": fu["pcsg00807"]["clean"]["rates"][key]}
        for s, w in want.items():
            v = d["values"][s][k]
            assert (v["errors"], v["chars"], v["lines"]) == (w["errors"], w["chars"], w["lines"]), (s, k)
            assert v["cer"] == w["cer"], (s, k)
    for s in ("vita-test", "psp-test"):
        v = d["values"][s]["Lens"]
        assert (v["errors"], v["chars"], v["lines"]) == (lens[s]["errors"], lens[s]["chars"], lens[s]["lines"])


@needs_data
def test_rubi_path_numbers_are_the_final_reads():
    d = data()
    assert d["rubi_path"] == chart.rubi_path(ROOT / d["sources"]["followup"]["file"])


@needs_data
def test_the_resources_come_from_their_records():
    d = data()
    src = {k: ROOT / d["sources"][k]["file"] for k in ("vocr6", "vocr5_report", "vocr5_protocol", "readers")}
    assert d["resources"] == chart.resources(src["vocr6"], src["vocr5_report"], src["vocr5_protocol"],
                                             src["readers"])
    g = source(d, "vocr6")["follow_up_gate"]["tiers"]
    for t in "SNL":
        assert d["resources"][f"VOCR6_{t}"]["bytes"] == g[t]["blob"]["bytes"]
        assert d["resources"][f"VOCR6_{t}"]["box_s_333"] * 1000 == pytest.approx(g[t]["est_box_ms"]["shipped_333"])
    for k in OTHERS[1:]:
        assert d["resources"][k]["bytes"] == source(d, "readers")["sizes"][k]["bytes"]


@needs_data
def test_the_row_notes_numbers_come_from_the_data():
    d = data()
    notes = d["copy"]["row_notes"]
    assert typed_numbers(json.dumps(notes, ensure_ascii=False)) == [], "the row notes' numbers are filled in"
    for k in VOCR6 + VOCR5:
        r = d["resources"][k]
        note = chart.row_note(d, k)
        assert chart.mb(r["bytes"]) in note and "int8" in note
        assert f"{r['box_s_444']:.1f}-{r['box_s_333']:.1f} s" in note and "est." in note, "times are estimates"
    assert "internet" in chart.row_note(d, "Lens")
    for k in OTHERS[1:]:
        assert chart.mb(d["resources"][k]["bytes"]) in chart.row_note(d, k)


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
    for set_name in ("psp-test", "vita-test"):
        fur = d["facts"][f"lens_furigana_{set_name}"]
        assert f"{fur['furigana_errors']} of its {fur['all_errors']} {set_name}" in text, set_name
    for word in ("furigana", "estimate", "not run on a console", "glyph raster", "not an unseen game",
                 "first candidates", "no language model"):
        assert word in text, f"the footnote discloses: {word}"


@needs_data
def test_the_headline_claims_hold():
    d = data()
    assert d["copy"].get("headline"), "no headline"
    assert d["copy"].get("claims"), "the headline's comparison must be written down as claims"
    assert chart.check(d) == []


@needs_data
def test_every_vocr6_tier_beats_lens_with_its_interval_and_large_beats_every_engine():
    d = data()
    claims = {(c["set"], c["system"], c["than"]): c for c in d["copy"]["claims"] if c["op"] == "<"}
    for s in ("clannad", "vita-test", "psp-test"):
        for k in VOCR6:
            assert claims[(s, k, "Lens")].get("ci") is True, (s, k)
    for s in SET_KEYS:
        for o in OTHERS + ("VOCR5_L",):
            if s == "pcsg00807" and o == "Lens":
                continue
            assert (s, "VOCR6_L", o) in claims, (s, o)


@needs_data
def test_the_committed_images_are_drawn_from_this_json():
    d = data()
    assert d["rendered"]["inputs_sha256"] == chart.inputs_sha256(d), "the JSON changed since the images were drawn"
    assert set(d["rendered"]["files"]) == set(IMAGES)
    for name in IMAGES:
        f = CHARTS / name
        assert f.is_file() and f.stat().st_size < 1 << 20, f
        assert chart.sha256(f) == d["rendered"]["files"][name], f"{name} is not the image drawn from this JSON"


# The chart's words, frozen: a change to the headline, subhead, row notes, footnote or marks must be re-read for
# truthfulness against the data and this sha256 updated in the same PR.
WORDS_SHA256 = "5ef385e78370c915cb51291d4cf7746becda05271ad2699237eae72b3d623e47"


@needs_data
def test_the_words_are_the_reviewed_ones():
    c = data()["copy"]
    words = {k: c[k] for k in ("eyebrow", "headline", "subhead", "row_notes", "not_measured", "footnote", "marks")}
    got = chart.hashlib.sha256(json.dumps(words, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert got == WORDS_SHA256, "the chart's words changed: review them against the data, then update WORDS_SHA256"


@needs_data
def test_the_svgs_say_what_the_json_says():
    d = data()
    c = d["copy"]
    V = d["values"]
    want = [c["eyebrow"], c["headline"], c["subhead"]]
    for s in d["sets"]:
        want += [s["label"], f"{s['platform']} · {s['lines']:,} lines"]
        for x in d["systems"]:
            v = V[s["key"]][x["key"]]
            if v is None:
                want.append(c["not_measured"][s["key"]][x["key"]])
            else:
                want.append(chart.fmt_pct(v["cer"]) + c.get("marks", {}).get(s["key"], {}).get(x["key"], ""))
    for x in d["systems"]:
        want += [x["label"], chart.row_note(d, x["key"])]
    foot = " ".join(chart.footnote_text(d)).split()
    for name in ("vocr-vs-lens-light.svg", "vocr-vs-lens-dark.svg"):
        texts = re.findall(r"<!-- (.*?) -->", (CHARTS / name).read_text(encoding="utf-8"), re.S)
        texts = [t.replace("&amp;", "&") for t in texts]
        n = len(want)
        assert texts[:n] == want, name
        assert " ".join(texts[n:]).split() == foot, f"{name}: the footnote drawn is the footnote filled from the data"


def pct(v):
    return f"{v:.2f}%"


@needs_data
def test_the_readme_alt_text_gives_every_bar():
    d = data()
    alt = re.search(r'<img src="docs/charts/vocr-vs-lens-light\.svg"[^>]*alt="([^"]+)"', readme())
    assert alt, "the README shows the light chart with alt text"
    assert 'srcset="docs/charts/vocr-vs-lens-dark.svg"' in readme()
    bars = []
    for s in d["sets"]:
        assert f"{s['label']}, {s['platform']}, {s['lines']:,} lines" in alt.group(1)
        bars += [pct(v["cer"]) for x in d["systems"] if (v := d["values"][s["key"]][x["key"]]) is not None]
    assert re.findall(r"\d+\.\d\d%", alt.group(1)) == bars, "the alt text's values are the chart's, in order"


def table_rows(text):
    """{first cell's plain label: [cells]} of a markdown table."""
    out = {}
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            out[re.sub(r"\*\*|\[|\]\([^)]*\)|`[^`]*`", "", cells[0]).strip()] = cells
    return out


@needs_data
def test_the_readme_comparison_table_is_the_charts():
    d = data()
    sec = section(readme(), "## VOCR6 against other OCRs", "As Rubi runs them")
    rows = table_rows(sec)
    labels = {"VOCR6_S": "VOCR6 Small", "VOCR6_N": "VOCR6 Normal (the default)", "VOCR6_L": "VOCR6 Large",
              "VOCR5_S": "VOCR5 Small", "VOCR5_N": "VOCR5 Normal", "VOCR5_L": "VOCR5 Large", "Lens": "Google Lens",
              "MeikiOCR": "MeikiOCR (meikiocr 0.3.5)", "PPOCRv6_small": "PP-OCRv6 small",
              "PPOCRv5_mobile_int8": "PP-OCRv5 mobile, trimmed int8", "manga_ocr": "manga-ocr"}
    best = {s: min(v["cer"] for v in d["values"][s].values() if v) for s in SET_KEYS}
    for k, label in labels.items():
        cells = rows[label]
        r = d["resources"][k]
        if r.get("bytes"):
            assert cells[1].startswith(f"{r['bytes']:,} B"), (k, cells[1])
        if r.get("box_s_333"):
            assert cells[2] == f"{r['box_s_444']:.2f} / {r['box_s_333']:.2f} s", (k, cells[2])
        for i, s in enumerate(SET_KEYS):
            v = d["values"][s][k]
            want = "not run" if v is None else pct(v["cer"])
            if v is not None and v["cer"] == best[s]:
                want = f"**{want}**"
            if k == "Lens" and s in ("vita-test", "psp-test"):
                want += "\\*"
            assert cells[3 + i] == want, (k, s, cells[3 + i], want)
    assert "VOCR6.2_S_w8.vocr" in rows["VOCR6 Small"][0] and "releases/download/v0.5.0/" in rows["VOCR6 Small"][0]


@needs_data
def test_the_readme_rubi_path_table_is_the_final_reads():
    d = data()
    sec = section(readme(), "As Rubi runs them", "- **The sets.**")
    rows = table_rows(sec)
    for fam in ("VOCR6", "VOCR5"):
        cells = rows[f"{fam} Small / Normal / Large"]
        for i, s in enumerate(SET_KEYS):
            want = " / ".join(f"{d['rubi_path'][s][f'{fam}_{t}']['cer']:.2f}" for t in "SNL") + "%"
            assert cells[1 + i] == want, (fam, s)


@needs_data
def test_the_readme_filter_note_is_the_final_reads_floor():
    d = data()
    fl = source(d, "followup")["path_b"]["filter_floor"]["pcsg00807_clean"]
    note = section(readme(), "- **Rubi's text filter**", "- **The sets.**")
    assert f"costs {fl['points']:.2f} points ({fl['lines']} lines,\n  {fl['chars']} reference characters)" in note
    assert d["rubi_path"]["pcsg00807"]["VOCR6_L"]["errors"] >= fl["chars"], "the floor is part of every Rubi-path read"


@needs_data
def test_the_readme_notes_agree_with_the_record():
    d = data()
    notes = section(readme(), "- **The sets.**", "### The whole page")
    V = d["values"]
    for s in d["sets"]:
        assert f"{s['lines']:,}" in notes
    for set_name in ("psp-test", "vita-test"):
        fur = d["facts"][f"lens_furigana_{set_name}"]
        assert f"{fur['furigana_errors']} of its {fur['all_errors']} {set_name} errors" in notes
    # every VOCR6 tier below Lens on the three sets Lens read, with the pairs' deltas as written
    for k, label in zip(VOCR6, ("Small", "Normal", "Large")):
        deltas = [V[s][k]["vs"]["Lens"] for s in ("clannad", "vita-test", "psp-test")]
        assert all(p["ci95"][1] < 0 for p in deltas), k
        assert f"{label} " + " / ".join(f"−{abs(p['delta']):.2f}" for p in deltas) in notes, k
    lens_exact = 100 * V["clannad"]["Lens"]["exact_lines"] / V["clannad"]["Lens"]["lines"]
    large_exact = 100 * V["clannad"]["VOCR6_L"]["exact_lines"] / V["clannad"]["VOCR6_L"]["lines"]
    assert f"{large_exact:.1f}" == f"{lens_exact:.1f}" and f"exactly as Lens ({lens_exact:.1f}%)" in notes
    # VOCR6 Large lowest everywhere; its intervals exclude 0 against every engine but the one named
    for s in SET_KEYS:
        lo = min(v["cer"] for k, v in V[s].items() if v and k != "VOCR6_L")
        assert V[s]["VOCR6_L"]["cer"] < lo, s
        for o, p in V[s]["VOCR6_L"]["vs"].items():
            if (s, o) == ("psp-test", "PPOCRv6_small"):
                assert p["ci95"][0] < 0 < p["ci95"][1]
                assert f"({p['delta']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}], level)".replace("-", "−") \
                    in notes
            else:
                assert p["ci95"][1] < 0, (s, o)
    # Normal: lowest of the engines on three sets with intervals below 0; psp-test level with PP-OCRv6 small
    for s in ("clannad", "vita-test", "pcsg00807"):
        assert all(p["ci95"][1] < 0 for p in V[s]["VOCR6_N"]["vs"].values()), s
    p = V["psp-test"]["VOCR6_N"]["vs"]["PPOCRv6_small"]
    assert f"({p['delta']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}], level)".replace("-", "−") in notes
    # Small: below every engine in point except PP-OCRv6 small on psp-test; level with MeikiOCR on CLANNAD
    for s in SET_KEYS:
        for o in OTHERS:
            if V[s][o] is None or (s, o) == ("psp-test", "PPOCRv6_small"):
                continue
            assert V[s]["VOCR6_S"]["cer"] < V[s][o]["cer"], (s, o)
    assert all(p["ci95"][1] < 0 for p in V["pcsg00807"]["VOCR6_S"]["vs"].values())
    p = V["clannad"]["VOCR6_S"]["vs"]["MeikiOCR"]
    assert f"(−{abs(p['delta']):.2f} [−{abs(p['ci95'][0]):.2f}, +{p['ci95'][1]:.2f}])" in notes
    p = V["psp-test"]["VOCR6_S"]["vs"]["PPOCRv6_small"]
    assert f"(+{p['delta']:.2f} [−{abs(p['ci95'][0]):.2f}, +{p['ci95'][1]:.2f}], not resolved)" in notes
    # against VOCR5: lower everywhere in point; the resolved pairs as written
    rec = source(d, "readers")["sets"]
    for s in SET_KEYS:
        for t in "SNL":
            assert V[s][f"VOCR6_{t}"]["cer"] < V[s][f"VOCR5_{t}"]["cer"], (s, t)
    for s in ("clannad", "pcsg00807"):
        ps = [rec[s]["pairs"][f"VOCR6_{t}-VOCR5_{t}"] for t in "SNL"]
        assert all(p["ci95"][1] < 0 for p in ps), s
        assert " / ".join(f"−{abs(p['delta']):.2f}" for p in ps) in notes, s
    p = rec["vita-test"]["pairs"]["VOCR6_N-VOCR5_N"]
    assert p["ci95"][1] < 0 and f"(−{abs(p['delta']):.2f} [−{abs(p['ci95'][0]):.2f}, −{abs(p['ci95'][1]):.2f}])" \
        in notes
    rp = d["rubi_path"]["psp-test"]
    assert f"(Small {rp['VOCR6_S']['cer']:.2f}%\n    against {rp['VOCR5_S']['cer']:.2f}%, " \
           f"Normal {rp['VOCR6_N']['cer']:.2f}% against {rp['VOCR5_N']['cer']:.2f}%); " \
           f"Large reads it at {rp['VOCR6_L']['cer']:.2f}% against {rp['VOCR5_L']['cer']:.2f}%" in notes
    assert rp["VOCR6_S"]["cer"] > rp["VOCR5_S"]["cer"] and rp["VOCR6_N"]["cer"] > rp["VOCR5_N"]["cer"]
    assert rp["VOCR6_L"]["cer"] < rp["VOCR5_L"]["cer"]
    # the sizes the notes give for the PC engines
    mbs = sorted(d["resources"][k]["bytes"] for k in ("MeikiOCR", "PPOCRv6_small", "manga_ocr"))
    assert f"from {mbs[0] / 1e6:.1f} to {mbs[-1] / 1e6:.0f} MB of fp32 weights" in notes
    for word in ("glyph raster", "Collar x Malice", "planetarian", "Nothing of VOCR6 has run on a console",
                 "28 of 28 equal", "eval/other_readers_protocol.json"):
        assert word in notes, word


@needs_data
def test_the_readme_headline_is_backed():
    d = data()
    head = readme().split("## VOCR6 against other OCRs")[0]
    assert "better than Google Lens on every test set Lens read" in head
    assert "fewest errors of every OCR engine we measured, on all four test sets" in head
    V = d["values"]
    for s in ("clannad", "vita-test", "psp-test"):
        for k in VOCR6:
            assert V[s][k]["vs"]["Lens"]["ci95"][1] < 0, (s, k)
    assert V["pcsg00807"]["Lens"] is None
    for s in SET_KEYS:
        assert V[s]["VOCR6_L"]["cer"] == min(v["cer"] for v in V[s].values() if v), s
    lo, hi = (d["resources"][k]["bytes"] for k in ("VOCR6_S", "VOCR6_L"))
    assert f"{lo / 1e6:.1f} to {hi / 1e6:.1f} MB of int8 weights" in head


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
